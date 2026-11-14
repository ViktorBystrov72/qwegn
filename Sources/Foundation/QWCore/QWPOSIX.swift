// QWPOSIX.swift — Checked POSIX pread helpers with EINTR and partial-read loops.

import Darwin
import Foundation

public enum QWPOSIXError: Error, CustomStringConvertible, Sendable, Equatable {
    case invalidArgument(String)
    case ioFailed(offset: Int64, length: Int, errno: Int32)
    case shortRead(offset: Int64, expected: Int, actual: Int)

    public var description: String {
        switch self {
        case .invalidArgument(let msg):
            return "QWPOSIX: invalid argument (\(msg))"
        case .ioFailed(let off, let len, let err):
            return "QWPOSIX: pread failed at offset \(off) (length \(len)): \(String(cString: strerror(err)))"
        case .shortRead(let off, let expected, let actual):
            return "QWPOSIX: short read at offset \(off) (expected \(expected), got \(actual))"
        }
    }
}

public enum QWPOSIX {
    /// Reads up to `count` bytes at `offset`, retrying `EINTR` and partial reads until EOF.
    public static func preadUpTo(
        fd: Int32,
        into dst: UnsafeMutableRawPointer,
        count: Int,
        offset: Int64
    ) throws -> Int {
        guard fd >= 0, count >= 0, offset >= 0 else {
            throw QWPOSIXError.invalidArgument("fd=\(fd) count=\(count) offset=\(offset)")
        }
        if count == 0 { return 0 }
        var totalRead = 0
        while totalRead < count {
            let (curOff, overflow) = offset.addingReportingOverflow(Int64(totalRead))
            guard !overflow else {
                throw QWPOSIXError.invalidArgument("offset overflow at \(offset)+\(totalRead)")
            }
            let remaining = count - totalRead
            let got = Darwin.pread(fd, dst.advanced(by: totalRead), remaining, off_t(curOff))
            if got < 0 {
                let code = errno
                if code == EINTR { continue }
                throw QWPOSIXError.ioFailed(offset: curOff, length: remaining, errno: code)
            }
            if got == 0 { break }
            totalRead += got
        }
        return totalRead
    }

    /// Reads exactly `count` bytes at `offset`, throwing `shortRead` on premature EOF.
    public static func preadExact(
        fd: Int32,
        into dst: UnsafeMutableRawPointer,
        count: Int,
        offset: Int64
    ) throws {
        let got = try preadUpTo(fd: fd, into: dst, count: count, offset: offset)
        guard got == count else {
            throw QWPOSIXError.shortRead(offset: offset, expected: count, actual: got)
        }
    }
}
