// QWSMCClient.swift — AppleSMC sensor reader and fan override controller.
// Reads host GPU/CPU/memory temperatures, fan speeds, and system power directly
// from IOKit AppleSMC; never fabricates numbers when sensors are absent.

import Darwin
import Foundation
import IOKit

/// Cooling policy controlling host AppleSMC fan behavior during inference.
public enum QWCoolingPolicy: String, Codable, Sendable, CaseIterable {
    case auto = "auto"
    case proactive = "proactive"
    case max = "max"

    public var displayTitle: String {
        switch self {
        case .auto: return "System Auto"
        case .proactive: return "Proactive Pre-Spin"
        case .max: return "Max Blast"
        }
    }
}

/// Host thermal, fan, and power telemetry read from AppleSMC and ProcessInfo.
public struct QWSMCSnapshot: Sendable, Equatable {
    public let gpuTemperatureCelsius: Double?
    public let gpuTemperatureUnavailableReason: String?
    public let cpuTemperatureCelsius: Double?
    public let memoryTemperatureCelsius: Double?
    public let fanSpeedRPM: Double?
    public let fanSpeedUnavailableReason: String?
    public let fanMinRPM: Double?
    public let fanMaxRPM: Double?
    public let fanControlStatus: String
    public let systemPowerWatts: Double?
    public let systemPowerUnavailableReason: String?
    public let thermalStateLevel: String

    public init(
        gpuTemperatureCelsius: Double? = nil,
        gpuTemperatureUnavailableReason: String? = nil,
        cpuTemperatureCelsius: Double? = nil,
        memoryTemperatureCelsius: Double? = nil,
        fanSpeedRPM: Double? = nil,
        fanSpeedUnavailableReason: String? = nil,
        fanMinRPM: Double? = nil,
        fanMaxRPM: Double? = nil,
        fanControlStatus: String = "System Auto",
        systemPowerWatts: Double? = nil,
        systemPowerUnavailableReason: String? = nil,
        thermalStateLevel: String = "Nominal"
    ) {
        self.gpuTemperatureCelsius = gpuTemperatureCelsius
        self.gpuTemperatureUnavailableReason = gpuTemperatureUnavailableReason
        self.cpuTemperatureCelsius = cpuTemperatureCelsius
        self.memoryTemperatureCelsius = memoryTemperatureCelsius
        self.fanSpeedRPM = fanSpeedRPM
        self.fanSpeedUnavailableReason = fanSpeedUnavailableReason
        self.fanMinRPM = fanMinRPM
        self.fanMaxRPM = fanMaxRPM
        self.fanControlStatus = fanControlStatus
        self.systemPowerWatts = systemPowerWatts
        self.systemPowerUnavailableReason = systemPowerUnavailableReason
        self.thermalStateLevel = thermalStateLevel
    }
}

/// Low-level IOKit AppleSMC key reader/writer matching the 80-byte kernel ABI.
final class QWSMCClient: @unchecked Sendable {
    struct SMCKeyInfoData {
        var dataSize: UInt32 = 0
        var dataType: UInt32 = 0
        var dataAttributes: UInt8 = 0
    }

    typealias SMCBytes32 = (
        UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8
    )

    /// Explicit 80-byte C ABI container (prevents Swift tail-padding field packing).
    struct SMCKeyData {
        private typealias Storage = (
            UInt64, UInt64, UInt64, UInt64, UInt64,
            UInt64, UInt64, UInt64, UInt64, UInt64
        )
        private var raw: Storage = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        var key: UInt32 {
            get { load(0, as: UInt32.self) }
            set { store(newValue, at: 0) }
        }
        var dataSize: UInt32 {
            get { load(28, as: UInt32.self) }
            set { store(newValue, at: 28) }
        }
        var dataType: UInt32 {
            get { load(32, as: UInt32.self) }
            set { store(newValue, at: 32) }
        }
        var dataAttributes: UInt8 {
            get { load(36, as: UInt8.self) }
            set { store(newValue, at: 36) }
        }
        var result: UInt8 {
            get { load(40, as: UInt8.self) }
            set { store(newValue, at: 40) }
        }
        var status: UInt8 {
            get { load(41, as: UInt8.self) }
            set { store(newValue, at: 41) }
        }
        var data8: UInt8 {
            get { load(42, as: UInt8.self) }
            set { store(newValue, at: 42) }
        }
        var data32: UInt32 {
            get { load(44, as: UInt32.self) }
            set { store(newValue, at: 44) }
        }
        var bytes: SMCBytes32 {
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

    deinit {
        if connection != 0 {
            IOServiceClose(connection)
            connection = 0
        }
    }

    static func fourCC(_ s: String) -> UInt32 {
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

    func readValue(_ key: String) -> Double? {
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
    func writeFloat(_ key: String, value: Float) -> kern_return_t {
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
    func writeUInt8(_ key: String, value: UInt8) -> kern_return_t {
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

    static func decode(bytes: SMCBytes32, size: UInt32, type: UInt32) -> Double? {
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

/// Manages AppleSMC thermal/fan sampling and cooling policy enforcement.
public final class QWFanController: @unchecked Sendable {
    public static let shared = QWFanController()
    public static let helperSocketPath = "/var/run/qwg-smc.sock"

    private static let gpuTempKeys = ["Tg08", "Tg0C", "Tg0O", "Tg0R", "Tg0U", "Tg0X", "Tg04", "Tg0D", "Tg0P"]
    private static let cpuTempKeys = ["Tp00", "Tp04", "Tp08", "Tp0C", "Tp0G", "Tp0K"]
    private static let memTempKeys = ["Tm00", "Tm04", "Tm08", "Tm0C"]
    private static let proactiveCooldownSeconds: TimeInterval = 15.0

    private let client = QWSMCClient()
    private let lock = NSLock()
    private var policy: QWCoolingPolicy = .auto
    private var inferencePhase: String = "idle"
    private var lastActiveInferenceAt: Date?
    private var didApplyManualOverride = false
    private var lastWriteAuthorized: Bool?

    public init() {
        atexit { QWFanController.shared.restoreAutoOnExit() }
    }

    public var currentPolicy: QWCoolingPolicy {
        lock.withLock { policy }
    }

    public func setPolicy(_ newPolicy: QWCoolingPolicy) {
        lock.withLock {
            policy = newPolicy
        }
        _ = sampleAndReconcile()
    }

    public func updateInferencePhase(_ phase: String) {
        let now = Date()
        lock.withLock {
            inferencePhase = phase
            if phase == "prefill" || phase == "decode" {
                lastActiveInferenceAt = now
            }
        }
        _ = sampleAndReconcile()
    }

    public func sampleAndReconcile(now: Date = Date()) -> QWSMCSnapshot {
        let gpuTemp = Self.maxSensor(keys: Self.gpuTempKeys, client: client)
        let cpuTemp = Self.maxSensor(keys: Self.cpuTempKeys, client: client)
        let memTemp = Self.maxSensor(keys: Self.memTempKeys, client: client)
        let thermalState = Self.currentThermalState()

        let fanCountRaw = client.readValue("FNum")
        let fanCount = Int((fanCountRaw ?? 0).rounded())
        var fanActuals: [Double] = []
        var fanMins: [Double] = []
        var fanMaxs: [Double] = []
        if fanCount > 0 {
            for idx in 0..<min(fanCount, 4) {
                if let ac = client.readValue("F\(idx)Ac"), ac >= 0 { fanActuals.append(ac) }
                if let mn = client.readValue("F\(idx)Mn"), mn > 0 { fanMins.append(mn) }
                if let mx = client.readValue("F\(idx)Mx"), mx > 0 { fanMaxs.append(mx) }
            }
        }

        let avgActual = fanActuals.isEmpty ? nil : (fanActuals.reduce(0, +) / Double(fanActuals.count))
        let avgMin = fanMins.isEmpty ? nil : (fanMins.reduce(0, +) / Double(fanMins.count))
        let avgMax = fanMaxs.isEmpty ? nil : (fanMaxs.reduce(0, +) / Double(fanMaxs.count))
        let power = client.readValue("PSTR").flatMap { ($0 >= 0.5 && $0 <= 500.0) ? $0 : nil }

        let controlStatus = reconcileFans(
            fanCount: fanCount,
            actualRPMs: fanActuals,
            minRPMs: fanMins,
            maxRPMs: fanMaxs,
            gpuTemp: gpuTemp,
            thermalState: thermalState,
            now: now
        )

        return QWSMCSnapshot(
            gpuTemperatureCelsius: gpuTemp,
            gpuTemperatureUnavailableReason: gpuTemp == nil ? "this host does not publish AppleSMC GPU thermal sensors" : nil,
            cpuTemperatureCelsius: cpuTemp,
            memoryTemperatureCelsius: memTemp,
            fanSpeedRPM: avgActual,
            fanSpeedUnavailableReason: avgActual == nil ? "this host has no active cooling fans or AppleSMC fan telemetry" : nil,
            fanMinRPM: avgMin,
            fanMaxRPM: avgMax,
            fanControlStatus: controlStatus,
            systemPowerWatts: power,
            systemPowerUnavailableReason: power == nil ? "this host does not publish AppleSMC system power (PSTR)" : nil,
            thermalStateLevel: thermalState
        )
    }

    private func reconcileFans(
        fanCount: Int,
        actualRPMs: [Double],
        minRPMs: [Double],
        maxRPMs: [Double],
        gpuTemp: Double?,
        thermalState: String,
        now: Date
    ) -> String {
        guard fanCount > 0 else { return "Unavailable" }
        let snapshotState = lock.withLock {
            (policy, inferencePhase, lastActiveInferenceAt, didApplyManualOverride, lastWriteAuthorized)
        }
        let activePolicy = snapshotState.0
        let phase = snapshotState.1
        let lastActive = snapshotState.2
        let manualWasSet = snapshotState.3
        let writeAuth = snapshotState.4
        let isHot = (gpuTemp ?? 0) >= 85.0 || thermalState == "Serious" || thermalState == "Critical"

        switch activePolicy {
        case .auto:
            if manualWasSet { _ = writeAutoMode(fanCount: fanCount) }
            return "System Auto"
        case .proactive:
            let withinCooldown = lastActive.map { now.timeIntervalSince($0) < Self.proactiveCooldownSeconds } ?? false
            let active = phase == "prefill" || phase == "decode" || withinCooldown || (gpuTemp ?? 0) >= 72.0 || isHot
            if !active && (gpuTemp ?? 0) < 58.0 {
                if manualWasSet { _ = writeAutoMode(fanCount: fanCount) }
                return isPrivilegedOrHelperPresent(knownAuth: writeAuth) ? "Proactive (Idle · Auto)" : "System Auto (SMC write requires root)"
            }
            let ratio = isHot ? 1.0 : 0.80
            return applyManualRatio(ratio, fanCount: fanCount, actualRPMs: actualRPMs, minRPMs: minRPMs, maxRPMs: maxRPMs, label: "Proactive Boost")
        case .max:
            return applyManualRatio(1.0, fanCount: fanCount, actualRPMs: actualRPMs, minRPMs: minRPMs, maxRPMs: maxRPMs, label: "Max Blast")
        }
    }

    private func applyManualRatio(
        _ ratio: Double,
        fanCount: Int,
        actualRPMs: [Double],
        minRPMs: [Double],
        maxRPMs: [Double],
        label: String
    ) -> String {
        var appliedTargets: [Double] = []
        var writeFailed = false
        for idx in 0..<min(fanCount, 4) {
            let maxRPM = idx < maxRPMs.count ? maxRPMs[idx] : 5000.0
            let minRPM = idx < minRPMs.count ? minRPMs[idx] : 1200.0
            let curRPM = idx < actualRPMs.count ? actualRPMs[idx] : 0.0
            let safeTarget = min(maxRPM, max(curRPM, minRPM, maxRPM * ratio))
            if writeTarget(fanIndex: idx, rpm: safeTarget) {
                appliedTargets.append(safeTarget)
            } else {
                writeFailed = true
            }
        }
        lock.withLock {
            lastWriteAuthorized = !writeFailed
            if !writeFailed { didApplyManualOverride = true }
        }
        if writeFailed || appliedTargets.isEmpty {
            return "System Auto (SMC write requires root)"
        }
        let avg = appliedTargets.reduce(0, +) / Double(appliedTargets.count)
        return "\(label) (\(Int(avg.rounded())) RPM)"
    }

    @discardableResult
    private func writeAutoMode(fanCount: Int) -> Bool {
        if FileManager.default.fileExists(atPath: Self.helperSocketPath), sendSocketCommand("AUTO\n") {
            lock.withLock { didApplyManualOverride = false; lastWriteAuthorized = true }
            return true
        }
        var ok = true
        for idx in 0..<min(fanCount, 4) {
            if client.writeUInt8("F\(idx)md", value: 0) != KERN_SUCCESS { ok = false }
        }
        lock.withLock {
            if ok { didApplyManualOverride = false }
            lastWriteAuthorized = ok
        }
        return ok
    }

    private func writeTarget(fanIndex: Int, rpm: Double) -> Bool {
        let roundedRPM = Int(rpm.rounded())
        if FileManager.default.fileExists(atPath: Self.helperSocketPath),
           sendSocketCommand("SET \(fanIndex) \(roundedRPM)\n") {
            return true
        }
        guard geteuid() == 0 else { return false }
        let modeRes = client.writeUInt8("F\(fanIndex)md", value: 1)
        let tgRes = client.writeFloat("F\(fanIndex)Tg", value: Float(rpm))
        return modeRes == KERN_SUCCESS && tgRes == KERN_SUCCESS
    }

    private func isPrivilegedOrHelperPresent(knownAuth: Bool?) -> Bool {
        if let knownAuth { return knownAuth }
        return geteuid() == 0 || FileManager.default.fileExists(atPath: Self.helperSocketPath)
    }

    private func sendSocketCommand(_ command: String) -> Bool {
        let fd = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { return false }
        defer { Darwin.close(fd) }
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = Array(Self.helperSocketPath.utf8CString)
        let maxLen = MemoryLayout.size(ofValue: addr.sun_path)
        guard pathBytes.count <= maxLen else { return false }
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: maxLen) { dst in
                for (i, ch) in pathBytes.enumerated() { dst[i] = ch }
            }
        }
        let len = socklen_t(MemoryLayout<sockaddr_un>.size)
        let connRes = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.connect(fd, $0, len) }
        }
        guard connRes == 0 else { return false }
        return command.withCString { cstr in
            let bytes = strlen(cstr)
            return Darwin.write(fd, cstr, bytes) == bytes
        }
    }

    private func restoreAutoOnExit() {
        let shouldRestore = lock.withLock { didApplyManualOverride }
        guard shouldRestore else { return }
        _ = writeAutoMode(fanCount: 4)
    }

    private static func maxSensor(keys: [String], client: QWSMCClient) -> Double? {
        var maxVal: Double?
        for key in keys {
            if let v = client.readValue(key), v >= 15.0 && v <= 135.0 {
                maxVal = max(maxVal ?? v, v)
            }
        }
        return maxVal
    }

    private static func currentThermalState() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "Nominal"
        case .fair: return "Fair"
        case .serious: return "Serious"
        case .critical: return "Critical"
        @unknown default: return "Nominal"
        }
    }
}
