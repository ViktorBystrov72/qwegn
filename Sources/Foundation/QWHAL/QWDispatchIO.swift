import Foundation
import QWCore

/// Direct POSIX NVMe file I/O with optional `F_NOCACHE` unbuffered transfer.
public enum QWDispatchIOError: Error, CustomStringConvertible, Sendable {
    case openFailed(path: String, errno: Int32)
    case readFailed(offset: Int64, length: Int, errno: Int32)
    case writeFailed(offset: Int64, length: Int, errno: Int32)
    case channelClosed
    case shortRead(expected: Int, actual: Int)

    public var description: String {
        switch self {
        case .openFailed(let path, let err):
            return "QWDispatchIO: open failed for '\(path)': \(String(cString: strerror(err)))"
        case .readFailed(let off, let len, let err):
            return "QWDispatchIO: read failed at offset \(off) (length \(len)): \(String(cString: strerror(err)))"
        case .writeFailed(let off, let len, let err):
            return "QWDispatchIO: write failed at offset \(off) (length \(len)): \(String(cString: strerror(err)))"
        case .channelClosed:
            return "QWDispatchIO: channel is closed"
        case .shortRead(let exp, let act):
            return "QWDispatchIO: short read (expected \(exp) bytes, got \(act))"
        }
    }
}

public final class QWDispatchIO: @unchecked Sendable {
    /// Host VM page size (16 KiB on Apple Silicon) — sysconf, never hardcoded.
    public static let pageSize: Int = Int(sysconf(_SC_PAGESIZE))

    public enum AccessMode: Sendable {
        case readOnly
        case writeOnly
        case readWrite

        var posixFlags: Int32 {
            switch self {
            case .readOnly: return O_RDONLY
            case .writeOnly: return O_WRONLY | O_CREAT
            case .readWrite: return O_RDWR | O_CREAT
            }
        }
    }

    public let path: String
    public let mode: AccessMode
    public let directIO: Bool
    public let fd: Int32

    private let lock = NSLock()
    private var isClosed = false

    public init(path: String,
                mode: AccessMode = .readOnly,
                directIO: Bool = true) throws {
        self.path = path
        self.mode = mode
        self.directIO = directIO

        let openFlags = mode.posixFlags
        let fd = open(path, openFlags, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)
        guard fd >= 0 else {
            throw QWDispatchIOError.openFailed(path: path, errno: errno)
        }
        self.fd = fd

        if directIO {
            // Bypass page cache for direct DMA transfers from NVMe SSD
            _ = fcntl(fd, F_NOCACHE, 1)
        }
    }

    deinit {
        close()
    }

    /// Explicitly close the underlying POSIX file descriptor.
    public func close() {
        lock.lock()
        defer { lock.unlock() }
        if !isClosed {
            isClosed = true
            Darwin.close(fd)
        }
    }

    private func checkOpen() throws {
        lock.lock()
        defer { lock.unlock() }
        if isClosed {
            throw QWDispatchIOError.channelClosed
        }
    }

    /// Synchronous read using POSIX `pread` with an `EINTR` and partial-read retry loop.
    public func readSync(offset: Int64, into dst: UnsafeMutableRawPointer, length: Int) throws -> Int {
        try checkOpen()
        do {
            return try QWPOSIX.preadUpTo(fd: fd, into: dst, count: length, offset: offset)
        } catch let posixErr as QWPOSIXError {
            switch posixErr {
            case .ioFailed(let off, let len, let err):
                throw QWDispatchIOError.readFailed(offset: off, length: len, errno: err)
            case .shortRead(_, let expected, let actual):
                throw QWDispatchIOError.shortRead(expected: expected, actual: actual)
            case .invalidArgument:
                throw QWDispatchIOError.readFailed(offset: offset, length: length, errno: EINVAL)
            }
        }
    }

    /// Synchronous write fallback using POSIX `pwrite` with an `EINTR` retry loop.
    public func writeSync(offset: Int64, buffer: UnsafeRawPointer, length: Int) throws {
        try checkOpen()
        var totalWritten = 0
        var ptr = buffer

        while totalWritten < length {
            let toWrite = length - totalWritten
            let bytesWritten = pwrite(fd, ptr, toWrite, off_t(offset + Int64(totalWritten)))
            if bytesWritten < 0 {
                if errno == EINTR { continue }
                throw QWDispatchIOError.writeFailed(offset: offset + Int64(totalWritten), length: toWrite, errno: errno)
            }
            totalWritten += bytesWritten
            ptr = ptr.advanced(by: bytesWritten)
        }
    }
}
