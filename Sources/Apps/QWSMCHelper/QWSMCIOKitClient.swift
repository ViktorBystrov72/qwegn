// QWSMCIOKitClient.swift — Low-level AppleSMC 80-byte ABI reader and writer.
// Shared between the privileged qwg-smc-helper daemon and QWServer telemetry
// so both speak the exact IOKit struct ABI without duplicated definitions.

import Darwin
import Foundation
import IOKit

/// Low-level IOKit AppleSMC key reader/writer matching the 80-byte kernel ABI.
public final class QWSMCClient: @unchecked Sendable {
    public struct SMCKeyInfoData: Sendable {
        public var dataSize: UInt32 = 0
        public var dataType: UInt32 = 0
        public var dataAttributes: UInt8 = 0
        public init(dataSize: UInt32 = 0, dataType: UInt32 = 0, dataAttributes: UInt8 = 0) {
            self.dataSize = dataSize
            self.dataType = dataType
            self.dataAttributes = dataAttributes
        }
    }

    public typealias SMCBytes32 = (
        UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8
    )

    /// Explicit 80-byte C ABI container (prevents Swift tail-padding field packing).
    public struct SMCKeyData: Sendable {
        private typealias Storage = (
            UInt64, UInt64, UInt64, UInt64, UInt64,
            UInt64, UInt64, UInt64, UInt64, UInt64
        )
        private var raw: Storage = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        public init() {}

        public var key: UInt32 {
            get { load(0, as: UInt32.self) }
            set { store(newValue, at: 0) }
        }
        public var dataSize: UInt32 {
            get { load(28, as: UInt32.self) }
            set { store(newValue, at: 28) }
        }
        public var dataType: UInt32 {
            get { load(32, as: UInt32.self) }
            set { store(newValue, at: 32) }
        }
        public var dataAttributes: UInt8 {
            get { load(36, as: UInt8.self) }
            set { store(newValue, at: 36) }
        }
        public var result: UInt8 {
            get { load(40, as: UInt8.self) }
            set { store(newValue, at: 40) }
        }
        public var status: UInt8 {
            get { load(41, as: UInt8.self) }
            set { store(newValue, at: 41) }
        }
        public var data8: UInt8 {
            get { load(42, as: UInt8.self) }
            set { store(newValue, at: 42) }
        }
        public var data32: UInt32 {
            get { load(44, as: UInt32.self) }
            set { store(newValue, at: 44) }
        }
        public var bytes: SMCBytes32 {
            get { load(48, as: SMCBytes32.self) }
            set { store(newValue, at: 48) }
        }

        private func load<T>(_ offset: Int, as type: T.Type) -> T {
            withUnsafeBytes(of: raw) { (buf: UnsafeRawBufferPointer) in
                buf.loadUnaligned(fromByteOffset: offset, as: T.self)
            }
        }

        private mutating func store<T>(_ value: T, at offset: Int) {
            withUnsafeMutableBytes(of: &raw) { (buf: UnsafeMutableRawBufferPointer) in
                buf.storeBytes(of: value, toByteOffset: offset, as: T.self)
            }
        }
    }

    private static let kSMCUserClientOpen: UInt32 = 2
    private static let kSMCReadKey: UInt8 = 5
    private static let kSMCWriteKey: UInt8 = 6
    private static let kSMCGetKeyInfo: UInt8 = 9
    private static let typeFlt: UInt32 = fourCC("flt ")
    private static let typeUi8: UInt32 = fourCC("ui8 ")
    private static let typeUi16: UInt32 = fourCC("ui16")
    private static let typeSp78: UInt32 = fourCC("sp78")

    private let lock = NSLock()
    private var connection: io_connect_t = 0
    private var openedOnce = false
    private var keyInfoCache: [UInt32: SMCKeyInfoData] = [:]
    private var unavailableKeys: Set<UInt32> = []

    public init() {}

    deinit {
        if connection != 0 {
            IOServiceClose(connection)
            connection = 0
        }
    }

    public static func fourCC(_ s: String) -> UInt32 {
        var val: UInt32 = 0
        for byte in s.utf8.prefix(4) {
            val = (val << 8) | UInt32(byte)
        }
        return val
    }

    private func ensureConnected() -> io_connect_t {
        if connection != 0 { return connection }
        if openedOnce { return 0 }
        openedOnce = true
        let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleSMC"))
        guard service != 0 else { return 0 }
        defer { IOObjectRelease(service) }
        var conn: io_connect_t = 0
        let kr = IOServiceOpen(service, mach_task_self_, 0, &conn)
        guard kr == KERN_SUCCESS else { return 0 }
        connection = conn
        return conn
    }

    private func callSMC(_ conn: io_connect_t, input: inout SMCKeyData, output: inout SMCKeyData) -> kern_return_t {
        var outSize = MemoryLayout<SMCKeyData>.stride
        return IOConnectCallStructMethod(
            conn,
            Self.kSMCUserClientOpen,
            &input,
            MemoryLayout<SMCKeyData>.stride,
            &output,
            &outSize
        )
    }

    private func cachedKeyInfo(_ conn: io_connect_t, keyCode: UInt32) -> SMCKeyInfoData? {
        if let info = keyInfoCache[keyCode] { return info }
        if unavailableKeys.contains(keyCode) { return nil }
        var input = SMCKeyData()
        var output = SMCKeyData()
        input.key = keyCode
        input.data8 = Self.kSMCGetKeyInfo
        let kr = callSMC(conn, input: &input, output: &output)
        guard kr == KERN_SUCCESS, output.result == 0, output.dataSize > 0 else {
            unavailableKeys.insert(keyCode)
            return nil
        }
        let info = SMCKeyInfoData(dataSize: output.dataSize, dataType: output.dataType, dataAttributes: output.dataAttributes)
        keyInfoCache[keyCode] = info
        return info
    }

    public func readValue(_ key: String) -> Double? {
        lock.withLock {
            readValueLocked(key)
        }
    }

    private func readValueLocked(_ key: String) -> Double? {
        let conn = ensureConnected()
        guard conn != 0 else { return nil }
        let code = Self.fourCC(key)
        guard let info = cachedKeyInfo(conn, keyCode: code) else { return nil }
        var input = SMCKeyData()
        var output = SMCKeyData()
        input.key = code
        input.dataSize = info.dataSize
        input.data8 = Self.kSMCReadKey
        let kr = callSMC(conn, input: &input, output: &output)
        guard kr == KERN_SUCCESS, output.result == 0 else { return nil }
        return Self.decode(bytes: output.bytes, size: info.dataSize, type: info.dataType)
    }

    @discardableResult
    public func writeFloat(_ key: String, value: Float) -> kern_return_t {
        lock.withLock {
            let conn = ensureConnected()
            guard conn != 0 else { return kIOReturnNotOpen }
            let code = Self.fourCC(key)
            guard let info = cachedKeyInfo(conn, keyCode: code), info.dataSize == 4 else {
                return kIOReturnNotFound
            }
            var input = SMCKeyData()
            var output = SMCKeyData()
            input.key = code
            input.dataSize = 4
            input.data8 = Self.kSMCWriteKey
            let bits = value.bitPattern
            var b = input.bytes
            if info.dataType == Self.typeFlt {
                b.0 = UInt8(bits & 0xFF)
                b.1 = UInt8((bits >> 8) & 0xFF)
                b.2 = UInt8((bits >> 16) & 0xFF)
                b.3 = UInt8((bits >> 24) & 0xFF)
            } else {
                let raw = UInt32(max(0, min(65535, Int(value.rounded()))))
                b.0 = UInt8((raw >> 8) & 0xFF)
                b.1 = UInt8(raw & 0xFF)
            }
            input.bytes = b
            let kr = callSMC(conn, input: &input, output: &output)
            return kr == KERN_SUCCESS ? (output.result == 0 ? KERN_SUCCESS : kIOReturnNotWritable) : kr
        }
    }

    @discardableResult
    public func writeUInt8(_ key: String, value: UInt8) -> kern_return_t {
        lock.withLock {
            let conn = ensureConnected()
            guard conn != 0 else { return kIOReturnNotOpen }
            let code = Self.fourCC(key)
            guard let info = cachedKeyInfo(conn, keyCode: code), info.dataSize >= 1 else {
                return kIOReturnNotFound
            }
            var input = SMCKeyData()
            var output = SMCKeyData()
            input.key = code
            input.dataSize = info.dataSize
            input.data8 = Self.kSMCWriteKey
            var b = input.bytes
            b.0 = value
            input.bytes = b
            let kr = callSMC(conn, input: &input, output: &output)
            return kr == KERN_SUCCESS ? (output.result == 0 ? KERN_SUCCESS : kIOReturnNotWritable) : kr
        }
    }

    public func fanCount() -> Int {
        max(0, min(4, Int((readValue("FNum") ?? 0).rounded())))
    }

    public func fanBounds(index: Int) -> (minRPM: Double, maxRPM: Double) {
        let rawMin = readValue("F\(index)Mn") ?? 1200.0
        let rawMax = readValue("F\(index)Mx") ?? 5000.0
        let safeMin = max(1000.0, rawMin)
        let safeMax = max(safeMin, rawMax)
        return (safeMin, safeMax)
    }

    public static func decode(bytes: SMCBytes32, size: UInt32, type: UInt32) -> Double? {
        if type == typeFlt && size == 4 {
            let bits = UInt32(bytes.0) | (UInt32(bytes.1) << 8)
                | (UInt32(bytes.2) << 16) | (UInt32(bytes.3) << 24)
            let val = Float(bitPattern: bits)
            return val.isFinite ? Double(val) : nil
        }
        if type == typeUi8 && size >= 1 {
            return Double(bytes.0)
        }
        if type == typeUi16 && size >= 2 {
            return Double((UInt16(bytes.0) << 8) | UInt16(bytes.1))
        }
        if type == typeSp78 && size >= 2 {
            let raw = Int16(bitPattern: (UInt16(bytes.0) << 8) | UInt16(bytes.1))
            return Double(raw) / 256.0
        }
        return nil
    }
}
