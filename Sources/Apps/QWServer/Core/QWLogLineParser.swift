// QWLogLineParser.swift — Structured classification for server and HTTP log lines

import Foundation

/// Filter category in the native Menu Bar log console.
public enum QWLogFilterLevel: String, CaseIterable, Identifiable, Sendable {
    case all = "ALL"
    case http = "HTTP"
    case info = "INFO"
    case warn = "WARN"
    case error = "ERROR"
    case debug = "DEBUG"

    public var id: String { rawValue }
}

/// Origin category for a parsed log line.
public enum QWLogEntryKind: String, Sendable {
    case system
    case http
}

/// Immutable, stably identified log entry for SwiftUI lists.
public struct QWLogEntry: Identifiable, Hashable, Sendable {
    public let id: UInt64
    public let line: String
    public let timestamp: String?
    public let level: QWLogLevel
    public let kind: QWLogEntryKind
    public let body: String

    /// Returns whether the entry passes both filter chip and search text.
    public func matches(filter: QWLogFilterLevel, query: String) -> Bool {
        let levelMatch: Bool = switch filter {
        case .all: true
        case .http: kind == .http
        case .info: level == .info
        case .warn: level == .warn
        case .error: level == .error
        case .debug: level == .debug
        }
        guard levelMatch else { return false }
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty || line.localizedCaseInsensitiveContains(trimmed)
    }
}

/// Parses formatted logger lines (`[ts] LEVEL msg`, `METHOD /path -> CODE`) cleanly.
public enum QWLogLineParser {
    private static let httpMethods = ["GET ", "POST ", "PUT ", "DELETE ", "PATCH ", "HEAD ", "OPTIONS "]

    public static func parse(id: UInt64, line: String) -> QWLogEntry {
        let (timestamp, rest) = splitTimestamp(line)
        let (level, kind, body) = classify(rest: rest, fullLine: line)
        return QWLogEntry(
            id: id,
            line: line,
            timestamp: timestamp,
            level: level,
            kind: kind,
            body: body
        )
    }

    private static func splitTimestamp(_ line: String) -> (String?, String) {
        guard line.hasPrefix("["),
              let closeIdx = line.firstIndex(of: "]") else {
            return (nil, line)
        }
        let tsStart = line.index(after: line.startIndex)
        let ts = String(line[tsStart..<closeIdx])
        // Guard against bracketed tags like `[ERROR]` being mistaken for timestamps.
        guard ts.contains("-") && ts.contains(":") else {
            return (nil, line)
        }
        let afterBracket = line.index(after: closeIdx)
        let remainder = String(line[afterBracket...]).trimmingCharacters(in: .whitespaces)
        return (ts, remainder)
    }

    private static func classify(rest: String, fullLine: String) -> (QWLogLevel, QWLogEntryKind, String) {
        let isHTTPMethod = httpMethods.contains { rest.hasPrefix($0) }
        let isHTTPArrow = rest.contains(" -> ") && (rest.contains(" OK |") || rest.contains(" ERR |"))
        if isHTTPMethod || isHTTPArrow {
            let isErr = rest.contains(" ERR |") || isClientOrServerError(rest)
            return (isErr ? .error : .info, .http, rest)
        }

        if let (lvl, stripped) = parseLeadingLevelTag(rest) {
            return (lvl, .system, stripped)
        }
        if fullLine.contains(" ERROR ") || fullLine.contains("[ERROR]") || rest.hasPrefix("ERROR:") {
            return (.error, .system, rest)
        }
        if fullLine.contains(" WARN ") || fullLine.contains("[WARN]") || rest.hasPrefix("WARNING:") {
            return (.warn, .system, rest)
        }
        if fullLine.contains(" DEBUG ") || fullLine.contains("[DEBUG]") {
            return (.debug, .system, rest)
        }
        return (.info, .system, rest)
    }

    private static func isClientOrServerError(_ rest: String) -> Bool {
        guard let range = rest.range(of: " -> ") else { return false }
        let suffix = rest[range.upperBound...].trimmingCharacters(in: .whitespaces)
        return suffix.hasPrefix("4") || suffix.hasPrefix("5")
    }

    private static func parseLeadingLevelTag(_ rest: String) -> (QWLogLevel, String)? {
        let candidates: [(String, QWLogLevel)] = [
            ("ERROR", .error), ("WARN", .warn), ("INFO", .info), ("DEBUG", .debug),
            ("[ERROR]", .error), ("[WARN]", .warn), ("[INFO]", .info), ("[DEBUG]", .debug),
        ]
        for (prefix, lvl) in candidates {
            if rest == prefix {
                return (lvl, "")
            }
            if rest.hasPrefix(prefix + " ") || rest.hasPrefix(prefix + "\t") {
                let tail = String(rest.dropFirst(prefix.count))
                    .trimmingCharacters(in: .whitespaces)
                return (lvl, tail)
            }
        }
        return nil
    }
}
