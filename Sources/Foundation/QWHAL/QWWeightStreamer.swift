// QWWeightStreamer.swift — Synchronous unbuffered direct DMA weight streamer via QWDispatchIO
import Foundation

public enum QWWeightStreamerError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case invalidRange(offset: Int, count: Int, fileLength: Int)
    case readFailed(String)

    public var description: String {
        switch self {
        case .fileNotFound(let path):
            return "QWWeightStreamer: file not found at '\(path)'"
        case .invalidRange(let off, let cnt, let len):
            let (end, ov) = off.addingReportingOverflow(cnt)
            let endStr = ov ? "overflow" : "\(end)"
            return "QWWeightStreamer: invalid range [\(off), \(endStr)) in file of length \(len)"
        case .readFailed(let msg):
            return "QWWeightStreamer: read failed: \(msg)"
        }
    }
}

public final class QWWeightStreamer: Sendable {
    public let url: URL
    public let dispatchIO: QWDispatchIO
    public let fileLength: Int

    public init(url: URL, unbuffered: Bool = true) throws {
        self.url = url
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw QWWeightStreamerError.fileNotFound(url.path)
        }
        let io = try QWDispatchIO(path: url.path, mode: .readOnly, directIO: unbuffered)
        var st = Darwin.stat()
        guard Darwin.fstat(io.fd, &st) == 0, let len = Int(exactly: st.st_size), len >= 0 else {
            io.close()
            throw QWWeightStreamerError.readFailed("Unable to determine file size")
        }
        self.dispatchIO = io
        self.fileLength = len
    }

    public func streamTensor(offset: Int, byteCount: Int, destination: UnsafeMutableRawPointer) throws {
        let (end, ov) = offset.addingReportingOverflow(byteCount)
        guard offset >= 0, byteCount >= 0, !ov, end <= fileLength else {
            throw QWWeightStreamerError.invalidRange(offset: offset, count: byteCount, fileLength: fileLength)
        }
        try dispatchIO.readExactSync(offset: Int64(offset), into: destination, length: byteCount)
    }
}
