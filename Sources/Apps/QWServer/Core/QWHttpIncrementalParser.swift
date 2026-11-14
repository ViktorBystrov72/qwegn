// QWHttpIncrementalParser.swift — Stateful O(1) amortized HTTP/1.1 frame parser.

import Foundation

public struct QWHttpParsedRequest: Sendable, Equatable {
    public let method: String
    public let path: String
    public let headers: [String: String]
    public let body: Data

    public init(method: String, path: String, headers: [String: String], body: Data) {
        self.method = method
        self.path = path
        self.headers = headers
        self.body = body
    }
}

public enum QWHttpParseError: Error, Equatable, Sendable {
    case headerTooLarge
    case payloadTooLarge
    case badEncoding
    case malformedRequestLine
    case invalidContentLength
    case unsupportedTransferEncoding

    public var httpStatus: Int {
        switch self {
        case .headerTooLarge: return 431
        case .payloadTooLarge: return 413
        case .badEncoding, .malformedRequestLine, .invalidContentLength: return 400
        case .unsupportedTransferEncoding: return 501
        }
    }

    public var message: String {
        switch self {
        case .headerTooLarge: return "Headers too large"
        case .payloadTooLarge: return "Payload too large"
        case .badEncoding: return "Bad request encoding"
        case .malformedRequestLine: return "Malformed request line"
        case .invalidContentLength: return "Invalid Content-Length header"
        case .unsupportedTransferEncoding: return "Transfer-Encoding is not supported"
        }
    }
}

public enum QWHttpParseStep: Equatable, Sendable {
    case needMoreData
    case complete(QWHttpParsedRequest)
}

public struct QWHttpIncrementalParser: Sendable {
    public static let defaultMaxHeaderBytes = 64 * 1024
    public static let defaultMaxBodyBytes = 32 * 1024 * 1024
    private static let headerDelimiter = Data([13, 10, 13, 10])

    public let maxHeaderBytes: Int
    public let maxBodyBytes: Int

    private var buffer = Data()
    private var scannedOffset: Int = 0
    private var parsedHeader: (method: String, path: String, headers: [String: String], bodyStart: Int, contentLength: Int)?

    public init(
        maxHeaderBytes: Int = defaultMaxHeaderBytes,
        maxBodyBytes: Int = defaultMaxBodyBytes
    ) {
        self.maxHeaderBytes = maxHeaderBytes
        self.maxBodyBytes = maxBodyBytes
    }

    public mutating func append(_ chunk: Data) throws -> QWHttpParseStep {
        if !chunk.isEmpty {
            buffer.append(chunk)
        }
        if parsedHeader == nil {
            let searchStart = max(0, scannedOffset - 3)
            let headerEndRange = buffer.range(of: Self.headerDelimiter, options: [], in: searchStart..<buffer.count)
            guard let headerEndRange else {
                scannedOffset = buffer.count
                if buffer.count > maxHeaderBytes {
                    throw QWHttpParseError.headerTooLarge
                }
                return .needMoreData
            }
            if headerEndRange.upperBound > maxHeaderBytes {
                throw QWHttpParseError.headerTooLarge
            }
            parsedHeader = try parseHead(headerData: buffer.subdata(in: 0..<headerEndRange.lowerBound), bodyStart: headerEndRange.upperBound)
        }
        guard let head = parsedHeader else { return .needMoreData }
        let (totalNeeded, overflow) = head.bodyStart.addingReportingOverflow(head.contentLength)
        if overflow {
            throw QWHttpParseError.invalidContentLength
        }
        guard buffer.count >= totalNeeded else {
            return .needMoreData
        }
        let bodyData = head.contentLength > 0
            ? buffer.subdata(in: head.bodyStart..<totalNeeded)
            : Data()
        buffer.removeAll(keepingCapacity: false)
        scannedOffset = 0
        parsedHeader = nil
        return .complete(QWHttpParsedRequest(
            method: head.method,
            path: head.path,
            headers: head.headers,
            body: bodyData
        ))
    }

    private func parseHead(
        headerData: Data,
        bodyStart: Int
    ) throws -> (method: String, path: String, headers: [String: String], bodyStart: Int, contentLength: Int) {
        guard let headerStr = String(data: headerData, encoding: .utf8) else {
            throw QWHttpParseError.badEncoding
        }
        let lines = headerStr.components(separatedBy: "\r\n")
        guard let requestLine = lines.first, !requestLine.isEmpty else {
            throw QWHttpParseError.malformedRequestLine
        }
        let parts = requestLine.split(separator: " ", omittingEmptySubsequences: true)
        guard parts.count >= 2 else {
            throw QWHttpParseError.malformedRequestLine
        }
        let method = String(parts[0])
        let path = String(parts[1])

        var headers: [String: String] = [:]
        var parsedContentLength: Int? = nil

        for line in lines.dropFirst() where !line.isEmpty {
            let kv = line.split(separator: ":", maxSplits: 1)
            guard kv.count == 2 else { continue }
            let key = kv[0].trimmingCharacters(in: .whitespaces).lowercased()
            let val = kv[1].trimmingCharacters(in: .whitespaces)
            if key == "transfer-encoding" {
                throw QWHttpParseError.unsupportedTransferEncoding
            }
            if key == "content-length" {
                guard let len = Int(val), len >= 0 else {
                    throw QWHttpParseError.invalidContentLength
                }
                if len > maxBodyBytes {
                    throw QWHttpParseError.payloadTooLarge
                }
                if let prev = parsedContentLength, prev != len {
                    throw QWHttpParseError.invalidContentLength
                }
                parsedContentLength = len
            }
            if let existing = headers[key] {
                headers[key] = "\(existing), \(val)"
            } else {
                headers[key] = val
            }
        }

        return (method, path, headers, bodyStart, parsedContentLength ?? 0)
    }
}
