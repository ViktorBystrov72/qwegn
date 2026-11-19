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

    private let stateCondition = NSCondition()
    private var isClosed = false
    private var didCloseFD = false
    private var activeIO = 0

    public init(path: String,
                mode: AccessMode = .readOnly,
                directIO: Bool = true) throws {
        self.path = path
        self.mode = mode
        self.directIO = directIO

        let openFlags = mode.posixFlags | O_CLOEXEC
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

    /// Explicitly close the underlying POSIX file descriptor after waiting for in-flight I/O.
    public func close() {
        stateCondition.withLock {
            if !isClosed {
                isClosed = true
            }
            while activeIO > 0 {
                stateCondition.wait()
            }
            if !didCloseFD {
                didCloseFD = true
                Darwin.close(fd)
            }
        }
    }

    private func withActiveFD<R>(_ body: (Int32) throws -> R) throws -> R {
        let activeFD = try stateCondition.withLock { () -> Int32 in
            if isClosed {
                throw QWDispatchIOError.channelClosed
            }
            activeIO += 1
            return fd
        }
        defer {
            stateCondition.withLock {
                activeIO -= 1
                if isClosed && activeIO == 0 {
                    stateCondition.broadcast()
                }
            }
        }
        return try body(activeFD)
    }

    private static func mapPOSIXError(_ posixErr: QWPOSIXError, offset: Int64, length: Int) -> QWDispatchIOError {
        switch posixErr {
        case .ioFailed(let off, let len, let err):
            return .readFailed(offset: off, length: len, errno: err)
        case .shortRead(_, let expected, let actual):
            return .shortRead(expected: expected, actual: actual)
        case .invalidArgument:
            return .readFailed(offset: offset, length: length, errno: EINVAL)
        }
    }

    /// Synchronous read using POSIX `pread` with an `EINTR` and partial-read retry loop.
    public func readSync(offset: Int64, into dst: UnsafeMutableRawPointer, length: Int) throws -> Int {
        try withActiveFD { activeFD in
            do {
                return try QWPOSIX.preadUpTo(fd: activeFD, into: dst, count: length, offset: offset)
            } catch let posixErr as QWPOSIXError {
                throw Self.mapPOSIXError(posixErr, offset: offset, length: length)
            }
        }
    }

    /// Synchronous exact-count read using `QWPOSIX.preadExact`, throwing `.shortRead` on premature EOF.
    public func readExactSync(offset: Int64, into dst: UnsafeMutableRawPointer, length: Int) throws {
        try withActiveFD { activeFD in
            do {
                try QWPOSIX.preadExact(fd: activeFD, into: dst, count: length, offset: offset)
            } catch let posixErr as QWPOSIXError {
                throw Self.mapPOSIXError(posixErr, offset: offset, length: length)
            }
        }
    }

    /// Synchronous write fallback using POSIX `pwrite` with an `EINTR` retry loop.
    public func writeSync(offset: Int64, buffer: UnsafeRawPointer, length: Int) throws {
        guard offset >= 0, length >= 0 else {
            throw QWDispatchIOError.writeFailed(offset: offset, length: length, errno: EINVAL)
        }
        let (_, endOv) = offset.addingReportingOverflow(Int64(length))
        guard !endOv else {
            throw QWDispatchIOError.writeFailed(offset: offset, length: length, errno: EOVERFLOW)
        }
        try withActiveFD { activeFD in
            var totalWritten = 0
            while totalWritten < length {
                let (curOff, offOv) = offset.addingReportingOverflow(Int64(totalWritten))
                guard !offOv else {
                    throw QWDispatchIOError.writeFailed(offset: offset, length: length - totalWritten, errno: EOVERFLOW)
                }
                let toWrite = length - totalWritten
                let bytesWritten = Darwin.pwrite(activeFD, buffer.advanced(by: totalWritten), toWrite, off_t(curOff))
                if bytesWritten < 0 {
                    let err = errno
                    if err == EINTR { continue }
                    throw QWDispatchIOError.writeFailed(offset: curOff, length: toWrite, errno: err)
                }
                if bytesWritten == 0 {
                    throw QWDispatchIOError.writeFailed(offset: curOff, length: toWrite, errno: EIO)
                }
                totalWritten += bytesWritten
            }
        }
    }
}
