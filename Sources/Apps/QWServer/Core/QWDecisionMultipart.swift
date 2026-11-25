// QWDecisionMultipart.swift — RFC 7578 multipart/form-data parser for multimodal decision requests.

import Foundation
import QWCore
import QWDecision

public struct QWDecisionMultipartPayload: Sendable {
    public let jsonBody: Data
    public let namedParts: [String: Data]
}

public enum QWDecisionMultipartParser {
    private static let crlfcrlf = Data([0x0D, 0x0A, 0x0D, 0x0A])
    private static let lflf = Data([0x0A, 0x0A])

    public static func parse(headers: [String: String], body: Data) throws -> QWDecisionMultipartPayload {
        guard let boundary = extractBoundary(from: headers["content-type"]) else {
            throw QWDecisionError.invalidQuestion("multipart/form-data request requires a boundary parameter")
        }
        let delim = Data("--\(boundary)".utf8)
        let maxBinaryParts = max(8, QWImageDecodePolicy.defaultMaxImagesPerTurn * 2)
        var jsonCandidate: Data?
        var explicitRequestJSON: Data?
        var parts: [String: Data] = [:]

        guard let firstDelim = body.range(of: delim) else {
            throw QWDecisionError.invalidQuestion("malformed multipart/form-data payload: boundary not found")
        }
        var cursor = firstDelim.upperBound
        while cursor < body.endIndex {
            if cursor + 2 <= body.endIndex, body[cursor] == 0x2D, body[cursor + 1] == 0x2D {
                break
            }
            if cursor + 2 <= body.endIndex, body[cursor] == 0x0D, body[cursor + 1] == 0x0A {
                cursor += 2
            } else if body[cursor] == 0x0A {
                cursor += 1
            } else {
                throw QWDecisionError.invalidQuestion("malformed multipart delimiter line ending")
            }

            let sepRange: Range<Data.Index>
            let contentStart: Data.Index
            if let r = body.range(of: crlfcrlf, in: cursor..<body.endIndex) {
                sepRange = r
                contentStart = r.upperBound
            } else if let r = body.range(of: lflf, in: cursor..<body.endIndex) {
                sepRange = r
                contentStart = r.upperBound
            } else {
                throw QWDecisionError.invalidQuestion("malformed multipart part header")
            }

            let headerText = String(decoding: body[cursor..<sepRange.lowerBound], as: UTF8.self)
            let meta = parsePartHeaders(headerText)

            guard let nextDelim = body.range(of: delim, in: contentStart..<body.endIndex) else {
                throw QWDecisionError.invalidQuestion("unterminated multipart/form-data part")
            }
            var contentEnd = nextDelim.lowerBound
            if contentEnd >= contentStart + 2, body[contentEnd - 2] == 0x0D, body[contentEnd - 1] == 0x0A {
                contentEnd -= 2
            } else if contentEnd >= contentStart + 1, body[contentEnd - 1] == 0x0A {
                contentEnd -= 1
            }

            let partData = Data(body[contentStart..<contentEnd])
            if let name = meta.name {
                if name == "request" || name == "payload" {
                    explicitRequestJSON = partData
                } else {
                    if let ct = meta.contentType, ct.contains("application/json"), jsonCandidate == nil {
                        jsonCandidate = partData
                    }
                    guard parts.count < maxBinaryParts else {
                        throw QWDecisionError.invalidQuestion("Too many multipart binary parts (max \(maxBinaryParts))")
                    }
                    parts[name] = partData
                }
            }
            cursor = nextDelim.upperBound
        }

        guard let resolvedJSON = explicitRequestJSON ?? jsonCandidate else {
            throw QWDecisionError.invalidQuestion("multipart/form-data request missing 'request' JSON part")
        }
        return QWDecisionMultipartPayload(jsonBody: resolvedJSON, namedParts: parts)
    }

    static func extractBoundary(from contentType: String?) -> String? {
        guard let contentType else { return nil }
        for param in contentType.split(separator: ";").dropFirst() {
            let trimmed = param.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.lowercased().hasPrefix("boundary=") else { continue }
            var val = String(trimmed.dropFirst("boundary=".count)).trimmingCharacters(in: .whitespacesAndNewlines)
            if val.hasPrefix("\"") && val.hasSuffix("\"") && val.count >= 2 {
                val = String(val.dropFirst().dropLast())
            }
            return val.isEmpty ? nil : val
        }
        return nil
    }

    private static func parsePartHeaders(_ text: String) -> (name: String?, contentType: String?) {
        var name: String?
        var contentType: String?
        for line in text.split(whereSeparator: \.isNewline) {
            let parts = line.split(separator: ":", maxSplits: 1)
            guard parts.count == 2 else { continue }
            let key = parts[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let val = parts[1].trimmingCharacters(in: .whitespacesAndNewlines)
            if key == "content-disposition" {
                name = extractQuotedParam("name", in: val)
            } else if key == "content-type" {
                contentType = val.lowercased()
            }
        }
        return (name, contentType)
    }

    private static func extractQuotedParam(_ paramName: String, in header: String) -> String? {
        for segment in header.split(separator: ";") {
            let trimmed = segment.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.lowercased().hasPrefix("\(paramName)=") else { continue }
            var val = String(trimmed.dropFirst(paramName.count + 1)).trimmingCharacters(in: .whitespacesAndNewlines)
            if val.hasPrefix("\"") && val.hasSuffix("\"") && val.count >= 2 {
                val = String(val.dropFirst().dropLast())
            }
            return val.isEmpty ? nil : val
        }
        return nil
    }
}
