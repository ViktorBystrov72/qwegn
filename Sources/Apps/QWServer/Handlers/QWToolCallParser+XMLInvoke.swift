// QWToolCallParser+XMLInvoke.swift — Dialect-agnostic tool envelopes and XML <invoke name="..."><parameter name="..."> parsing.
import Foundation
import QWCore
import QWToken

/// Open/close XML wrapper pair enclosing structured tool calls on the wire.
public struct QWToolCallEnvelope: Equatable, Hashable, Sendable {
    public let openTag: String
    public let closeTag: String
    public let closeTagPrefix: String

    public static let standard = QWToolCallEnvelope(openTag: "<tool_call>", closeTag: "</tool_call>")
    public static let knownEnvelopes: [QWToolCallEnvelope] = [
        .standard,
        QWToolCallEnvelope(openTag: "<ernie:tool_call>", closeTag: "</ernie:tool_call>"),
    ]

    public init(openTag: String, closeTag: String) {
        self.openTag = openTag
        self.closeTag = closeTag
        self.closeTagPrefix = closeTag.hasSuffix(">") ? String(closeTag.dropLast()) : closeTag
    }

    /// Builds an open/close pair from `<...>` (e.g. `<acme:tool_call>` -> `</acme:tool_call>`).
    public static func fromOpenTag(_ openTag: String) -> QWToolCallEnvelope? {
        guard openTag.hasPrefix("<"), openTag.hasSuffix(">"), !openTag.hasPrefix("</"), openTag.count > 2 else {
            return nil
        }
        return QWToolCallEnvelope(openTag: openTag, closeTag: "</" + String(openTag.dropFirst()))
    }

    /// Detects the active tool-call envelope declared by a chat template source.
    public static func detect(in templateSource: String) -> QWToolCallEnvelope {
        if let regex = try? NSRegularExpression(pattern: #"<([A-Za-z0-9_]+):tool_call>"#),
           let m = regex.firstMatch(
               in: templateSource,
               range: NSRange(templateSource.startIndex..<templateSource.endIndex, in: templateSource)
           ),
           let r = Range(m.range, in: templateSource),
           let env = fromOpenTag(String(templateSource[r])) {
            return env
        }
        if templateSource.contains("<|ernie:im_start|>") && templateSource.contains("<ernie:global_setting>") {
            return QWToolCallEnvelope(openTag: "<ernie:tool_call>", closeTag: "</ernie:tool_call>")
        }
        return .standard
    }
}

extension QWToolCallParser {
    static func earliestToolSegmentOpen(
        in text: Substring,
        envelopes: [QWToolCallEnvelope] = QWToolCallEnvelope.knownEnvelopes
    ) -> (range: Range<String.Index>, envelope: QWToolCallEnvelope)? {
        var best: (range: Range<String.Index>, envelope: QWToolCallEnvelope)?
        for env in envelopes {
            guard let r = text.range(of: env.openTag) else { continue }
            if best.map({ r.lowerBound < $0.range.lowerBound }) ?? true {
                best = (r, env)
            }
        }
        return best
    }

    static func firstCompleteToolCallEnd(
        in text: String,
        envelopes: [QWToolCallEnvelope] = QWToolCallEnvelope.knownEnvelopes
    ) -> String.Index? {
        var earliest: String.Index?
        for env in envelopes {
            guard let open = text.range(of: env.openTag),
                  let close = matchingToolCallClose(in: text[open.upperBound...], envelope: env) else {
                continue
            }
            if earliest.map({ close.upperBound < $0 }) ?? true {
                earliest = close.upperBound
            }
        }
        return earliest
    }

    static func parseToolSegmentCalls(inner: String, schemas: ToolSchemas) -> [ToolCall] {
        if inner.hasPrefix("{") && inner.hasSuffix("}"),
           let call = parseSingleToolCall(inner: inner, schemas: schemas) {
            return [call]
        }
        if inner.contains("<function=") || inner.contains("<arg_key>"),
           let call = parseSingleToolCall(inner: inner, schemas: schemas) {
            return [call]
        }
        if inner.contains("<invoke") {
            let xmlCalls = parseXMLInvokeToolCalls(inner, schemas: schemas)
            if !xmlCalls.isEmpty { return xmlCalls }
        }
        if let call = parseSingleToolCall(inner: inner, schemas: schemas) {
            return [call]
        }
        return parseXMLInvokeToolCalls(inner, schemas: schemas)
    }

    static func isBalancedToolPayload(_ raw: Substring) -> Bool {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("{") && trimmed.hasSuffix("}") {
            var depth = 0
            var inString = false
            var escaped = false
            let lastIdx = trimmed.index(before: trimmed.endIndex)
            for idx in trimmed.indices {
                let ch = trimmed[idx]
                if inString {
                    if escaped {
                        escaped = false
                    } else if ch == "\\" {
                        escaped = true
                    } else if ch == "\"" {
                        inString = false
                    }
                    continue
                }
                if ch == "\"" {
                    inString = true
                } else if ch == "{" {
                    depth += 1
                } else if ch == "}" {
                    depth -= 1
                    if depth < 0 { return false }
                    if depth == 0 && idx < lastIdx { return false }
                }
            }
            return !inString && depth == 0
        }
        if trimmed.hasPrefix("<function=") && trimmed.hasSuffix("</function>") {
            return trimmed.components(separatedBy: "<function=").count == 2
                && trimmed.components(separatedBy: "</function>").count == 2
                && trimmed.components(separatedBy: "<parameter=").count
                    == trimmed.components(separatedBy: "</parameter>").count
        }
        return isBalancedXMLInvokePayload(raw)
    }

    static func hasUnclosedLiteralContainer(_ raw: Substring) -> Bool {
        let trimmed = raw.drop(while: { $0.isWhitespace })
        if trimmed.hasPrefix("{") {
            var depth = 0, inString = false, escaped = false
            for ch in trimmed {
                if inString {
                    if escaped { escaped = false } else if ch == "\\" { escaped = true } else if ch == "\"" { inString = false }
                    continue
                }
                if ch == "\"" { inString = true } else if ch == "{" { depth += 1 } else if ch == "}" { depth -= 1 }
            }
            return inString || depth > 0
        }
        if trimmed.hasPrefix("<function=") {
            return trimmed.components(separatedBy: "<parameter=").count > trimmed.components(separatedBy: "</parameter>").count
        }
        return hasUnclosedXMLInvoke(trimmed)
    }

    static func matchingToolCallClose(
        in afterOpen: Substring,
        envelope: QWToolCallEnvelope = .standard
    ) -> Range<String.Index>? {
        var searchFrom = afterOpen.startIndex
        while let close = afterOpen[searchFrom...].range(of: envelope.closeTag) {
            let prefix = afterOpen[..<close.lowerBound]
            if isBalancedToolPayload(prefix) {
                return close
            }
            let tail = afterOpen[close.upperBound...].drop(while: { $0.isWhitespace })
            guard let nextClose = tail.range(of: envelope.closeTag) else { return close }
            if !hasUnclosedLiteralContainer(prefix) {
                if tail.hasPrefix(envelope.openTag) { return close }
                if let nextOpen = tail.range(of: envelope.openTag), nextOpen.lowerBound < nextClose.lowerBound {
                    return close
                }
            }
            searchFrom = close.upperBound
        }
        return nil
    }

    static func isBalancedXMLInvokePayload(_ raw: Substring) -> Bool {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasSuffix("</invoke>") else { return false }
        let invokeOpens = trimmed.components(separatedBy: "<invoke").count - 1
        let invokeCloses = trimmed.components(separatedBy: "</invoke>").count - 1
        let paramOpens = trimmed.components(separatedBy: "<parameter").count - 1
        let paramCloses = trimmed.components(separatedBy: "</parameter>").count - 1
        return invokeOpens >= 1 && invokeOpens == invokeCloses && paramOpens == paramCloses
    }

    static func hasUnclosedXMLInvoke(_ raw: Substring) -> Bool {
        let trimmed = raw.drop(while: { $0.isWhitespace })
        guard trimmed.contains("<invoke") else { return false }
        let invokeOpens = trimmed.components(separatedBy: "<invoke").count - 1
        let invokeCloses = trimmed.components(separatedBy: "</invoke>").count - 1
        let paramOpens = trimmed.components(separatedBy: "<parameter").count - 1
        let paramCloses = trimmed.components(separatedBy: "</parameter>").count - 1
        return invokeOpens > invokeCloses || paramOpens > paramCloses
    }

    static func parseXMLInvokeToolCalls(_ inner: String, schemas: ToolSchemas) -> [ToolCall] {
        var calls: [ToolCall] = []
        var cursor = inner.startIndex
        while let (fnName, bodyStart) = nextNamedTag(in: inner, from: cursor, tag: "<invoke") {
            let body: String
            let nextCursor: String.Index
            if let (end, after) = findBoundaryCloseTag(
                in: inner, from: bodyStart, closeTag: "</invoke>", nextOpenTag: "<invoke"
            ) {
                body = String(inner[bodyStart..<end])
                nextCursor = after
            } else {
                body = String(inner[bodyStart...])
                nextCursor = inner.endIndex
            }
            var members: [String] = []
            for (pName, pVal) in extractXMLNamedParameters(in: body) {
                members.append("\(jsonLiteral(pName)):\(jsonValue(pVal, type: schemas.type(function: fnName, parameter: pName)))")
            }
            calls.append(ToolCall(
                id: generateCallId(),
                type: "function",
                function: ToolCallFunction(name: fnName, arguments: "{" + members.joined(separator: ",") + "}")
            ))
            cursor = nextCursor
        }
        return calls
    }

    private static func extractXMLNamedParameters(in body: String) -> [(String, String)] {
        var out: [(String, String)] = []
        var cursor = body.startIndex
        while let (pName, valStart) = nextNamedTag(in: body, from: cursor, tag: "<parameter") {
            if let (valEnd, nextCursor) = findBoundaryCloseTag(
                in: body, from: valStart, closeTag: "</parameter>", nextOpenTag: "<parameter"
            ) {
                out.append((pName, String(body[valStart..<valEnd])))
                cursor = nextCursor
            } else {
                out.append((pName, String(body[valStart...])))
                break
            }
        }
        return out
    }

    private static func nextNamedTag(in text: String, from start: String.Index, tag: String) -> (name: String, bodyStart: String.Index)? {
        var search = start
        while let open = text[search...].range(of: tag) {
            var idx = open.upperBound
            guard idx < text.endIndex, text[idx].isWhitespace else {
                search = open.upperBound
                continue
            }
            while idx < text.endIndex, text[idx].isWhitespace { idx = text.index(after: idx) }
            guard text[idx...].hasPrefix("name=\"") else {
                search = open.upperBound
                continue
            }
            let nameStart = text.index(idx, offsetBy: 6)
            guard let quoteEnd = text[nameStart...].firstIndex(of: "\"") else { return nil }
            let afterQuote = text.index(after: quoteEnd)
            guard afterQuote < text.endIndex, text[afterQuote] == ">" else {
                search = open.upperBound
                continue
            }
            let name = String(text[nameStart..<quoteEnd])
            if !name.isEmpty, !name.contains("\n") {
                return (name, text.index(after: afterQuote))
            }
            search = open.upperBound
        }
        return nil
    }
}
