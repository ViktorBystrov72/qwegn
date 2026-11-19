import Foundation
import QWCore
import QWToken

public struct ToolCallFunction: Codable, Sendable, Equatable {
    public let name: String
    public let arguments: String

    public init(name: String, arguments: String) {
        self.name = name
        self.arguments = arguments
    }
}

public struct ToolCall: Codable, Sendable, Equatable, Identifiable {
    public let id: String
    public let type: String
    public let function: ToolCallFunction

    public init(id: String = QWToolCallParser.generateCallId(), type: String = "function", function: ToolCallFunction) {
        self.id = id
        self.type = type
        self.function = function
    }
}

public struct ToolCallChunk: Codable, Sendable {
    public struct FunctionChunk: Codable, Sendable {
        public let name: String?
        public let arguments: String?

        public init(name: String? = nil, arguments: String? = nil) {
            self.name = name
            self.arguments = arguments
        }
    }

    public let index: Int
    public let id: String?
    public let type: String?
    public let function: FunctionChunk?

    public init(index: Int, id: String? = nil, type: String? = "function", function: FunctionChunk? = nil) {
        self.index = index
        self.id = id
        self.type = type
        self.function = function
    }
}

public enum QWToolCallParser {
    public struct ParseResult: Sendable {
        public let content: String?
        public let toolCalls: [ToolCall]?
        public let reasoningContent: String?
        public let finishReason: String

        public init(content: String?, toolCalls: [ToolCall]?, reasoningContent: String?, finishReason: String) {
            self.content = content
            self.toolCalls = toolCalls
            self.reasoningContent = reasoningContent
            self.finishReason = finishReason
        }
    }

    /// Generates standard OpenAI-compatible tool call ID (e.g. "call_a1b2c3d4e5f6")
    public static func generateCallId() -> String {
        let hex = UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12).lowercased()
        return "call_\(hex)"
    }

/// Parses complete generated text into reasoning, content and tool calls;
/// supports both Qwen XML-tag and JSON tool-call formats. Unterminated
/// calls on a truncated generation are dropped; finish reason becomes length.
    public static func parse(text: String,
                             tools: [QWJson]? = nil,
                             startsInThinking: Bool = false,
                             truncated: Bool = false,
                             allowToolCalls: Bool = true,
                             allowReasoningTags: Bool = true) -> ParseResult {
        let parser = QWChatStreamParser(
            startsInReasoning: startsInThinking && allowReasoningTags,
            allowToolCalls: allowToolCalls,
            allowReasoningTags: allowReasoningTags
        )
        _ = parser.feed(text)
        _ = parser.finish()
        return parser.result(tools: allowToolCalls ? tools : nil, truncated: truncated)
    }

    /// Splits accumulated `<tool_call>` segments into calls. A segment without its closing
    /// tag is reported through `incomplete` and only returned as a call when `dropIncomplete`
    /// is false.
    static func parseToolSegments(_ text: String, tools: [QWJson]?, dropIncomplete: Bool)
        -> (calls: [ToolCall], incomplete: Bool) {
        let schemas = ToolSchemas(tools)
        var calls: [ToolCall] = []
        var incomplete = false
        var rest = Substring(text)
        while let open = rest.range(of: "<tool_call>") {
            let afterOpen = rest[open.upperBound...]
            let inner: Substring
            let closed: Bool
            if let close = matchingToolCallClose(in: afterOpen) {
                inner = afterOpen[..<close.lowerBound]
                rest = afterOpen[close.upperBound...]
                closed = true
            } else {
                inner = afterOpen
                rest = Substring("")
                closed = false
            }
            if !closed {
                incomplete = true
                if dropIncomplete { continue }
            }
            let trimmed = inner.trimmingCharacters(in: .whitespacesAndNewlines)
            if let call = parseSingleToolCall(inner: trimmed, schemas: schemas) {
                calls.append(call)
            }
        }
        return (calls, incomplete)
    }

    static func firstCompleteToolCallEnd(in text: String) -> String.Index? {
        guard let open = text.range(of: "<tool_call>") else { return nil }
        return matchingToolCallClose(in: text[open.upperBound...])?.upperBound
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
        return false
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
        return false
    }

    static func matchingToolCallClose(in afterOpen: Substring) -> Range<String.Index>? {
        var searchFrom = afterOpen.startIndex
        while let close = afterOpen[searchFrom...].range(of: "</tool_call>") {
            let prefix = afterOpen[..<close.lowerBound]
            if isBalancedToolPayload(prefix) {
                return close
            }
            let tail = afterOpen[close.upperBound...].drop(while: { $0.isWhitespace })
            guard let nextClose = tail.range(of: "</tool_call>") else { return close }
            if !hasUnclosedLiteralContainer(prefix) {
                if tail.hasPrefix("<tool_call>") { return close }
                if let nextOpen = tail.range(of: "<tool_call>"), nextOpen.lowerBound < nextClose.lowerBound {
                    return close
                }
            }
            searchFrom = close.upperBound
        }
        return nil
    }

    private static func parseSingleToolCall(inner: String, schemas: ToolSchemas) -> ToolCall? {
        // Format A: JSON format inside <tool_call>
        if inner.hasPrefix("{") && inner.hasSuffix("}") {
            if let data = inner.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                let name = (json["name"] as? String) ?? (json["function"] as? String) ?? ""
                if !name.isEmpty {
                    let argsString: String
                    if let rawArgs = json["arguments"] ?? json["parameters"] {
                        if let dict = rawArgs as? [String: Any] {
                            let coerced = schemas.coerceDictionary(dict, function: name)
                            if let d = try? JSONSerialization.data(withJSONObject: coerced, options: [.sortedKeys, .withoutEscapingSlashes]),
                               let str = String(data: d, encoding: .utf8) {
                                argsString = str
                            } else {
                                argsString = "{}"
                            }
                        } else if let s = rawArgs as? String {
                            if let sData = s.data(using: .utf8),
                               let dict = (try? JSONSerialization.jsonObject(with: sData)) as? [String: Any],
                               let d = try? JSONSerialization.data(
                                   withJSONObject: schemas.coerceDictionary(dict, function: name),
                                   options: [.sortedKeys, .withoutEscapingSlashes]
                               ),
                               let str = String(data: d, encoding: .utf8) {
                                argsString = str
                            } else {
                                argsString = s
                            }
                        } else if let d = try? JSONSerialization.data(withJSONObject: rawArgs, options: [.sortedKeys, .withoutEscapingSlashes]),
                                  let str = String(data: d, encoding: .utf8) {
                            argsString = str
                        } else {
                            argsString = "{}"
                        }
                    } else {
                        argsString = "{}"
                    }
                    return ToolCall(
                        id: generateCallId(),
                        type: "function",
                        function: ToolCallFunction(name: name, arguments: argsString)
                    )
                }
            }
        }

        // Format B: XML tag format (function/parameter open-close pairs);
        // Format A above is the JSON-object variant.
        if let funcOpen = inner.range(of: "<function="),
           let gtIdx = inner[funcOpen.upperBound...].firstIndex(of: ">") {
            let name = String(inner[funcOpen.upperBound..<gtIdx]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !name.isEmpty, !name.contains("\n") {
                let bodyStart = inner.index(after: gtIdx)
                let afterOpen = inner[bodyStart...]
                let body: String
                if let funcClose = afterOpen.range(of: "</function>", options: .backwards),
                   afterOpen[funcClose.upperBound...].allSatisfy(\.isWhitespace) {
                    body = String(afterOpen[..<funcClose.lowerBound])
                } else {
                    body = String(afterOpen)
                }

                // Arguments are assembled by hand so the parameter order the model produced is
                // preserved: the chat template re-renders history from this JSON, and a reordered
                // call would no longer match what the model actually generated.
                var members: [String] = []
                for (pName, rawVal) in Self.extractXMLParameters(in: body) {
                    var pVal = rawVal
                    if pVal.hasPrefix("\n") { pVal.removeFirst() }
                    if pVal.hasSuffix("\n") { pVal.removeLast() }
                    members.append("\(jsonLiteral(pName)):\(jsonValue(pVal, type: schemas.type(function: name, parameter: pName)))")
                }

                return ToolCall(
                    id: generateCallId(),
                    type: "function",
                    function: ToolCallFunction(name: name, arguments: "{" + members.joined(separator: ",") + "}")
                )
            }
        }

        // Format C: <tool_call>{function-name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>
        if let firstArgRange = inner.range(of: "<arg_key>") {
            let name = String(inner[..<firstArgRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else { return nil }
            let body = String(inner[firstArgRange.lowerBound...])
            var members: [String] = []
            for (pName, rawVal) in Self.extractGLMArgPairs(in: body) {
                var pVal = rawVal
                if pVal.hasPrefix("\n") { pVal.removeFirst() }
                if pVal.hasSuffix("\n") { pVal.removeLast() }
                members.append("\(jsonLiteral(pName)):\(jsonValue(pVal, type: schemas.type(function: name, parameter: pName)))")
            }
            return ToolCall(
                id: generateCallId(),
                type: "function",
                function: ToolCallFunction(name: name, arguments: "{" + members.joined(separator: ",") + "}")
            )
        }

        let bareName = inner.trimmingCharacters(in: .whitespacesAndNewlines)
        if !bareName.isEmpty,
           bareName.range(of: #"^[A-Za-z_][A-Za-z0-9_.-]*$"#, options: .regularExpression) != nil {
            return ToolCall(
                id: generateCallId(),
                type: "function",
                function: ToolCallFunction(name: bareName, arguments: "{}")
            )
        }

        return nil
    }

    private static func extractXMLParameters(in body: String) -> [(String, String)] {
        var out: [(String, String)] = []
        var cursor = body.startIndex
        while let open = body[cursor...].range(of: "<parameter="),
              let gt = body[open.upperBound...].firstIndex(of: ">") {
            let rawName = String(body[open.upperBound..<gt])
            if rawName.contains("\n") {
                cursor = body.index(after: open.lowerBound)
                continue
            }
            let pName = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
            let valStart = body.index(after: gt)
            if let (valEnd, nextCursor) = findBoundaryCloseTag(
                in: body, from: valStart, closeTag: "</parameter>", nextOpenTag: "<parameter="
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

    private static func extractGLMArgPairs(in body: String) -> [(String, String)] {
        var out: [(String, String)] = []
        var cursor = body.startIndex
        while let kOpen = body[cursor...].range(of: "<arg_key>"),
              let kClose = body[kOpen.upperBound...].range(of: "</arg_key>"),
              let vOpen = body[kClose.upperBound...].range(of: "<arg_value>") {
            let between = body[kClose.upperBound..<vOpen.lowerBound]
            guard between.allSatisfy(\.isWhitespace) else {
                cursor = kClose.upperBound
                continue
            }
            let pName = String(body[kOpen.upperBound..<kClose.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            let valStart = vOpen.upperBound
            if let (valEnd, nextCursor) = findBoundaryCloseTag(
                in: body, from: valStart, closeTag: "</arg_value>", nextOpenTag: "<arg_key>"
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

    private static func findBoundaryCloseTag(
        in body: String,
        from valStart: String.Index,
        closeTag: String,
        nextOpenTag: String
    ) -> (valEnd: String.Index, nextCursor: String.Index)? {
        var search = valStart
        var fallbackNextOpen: String.Index? = nil
        while let close = body[search...].range(of: closeTag) {
            let after = body[close.upperBound...].drop(while: { $0.isWhitespace })
            if after.isEmpty || after.hasPrefix(nextOpenTag) {
                return (close.lowerBound, close.upperBound)
            }
            search = close.upperBound
        }
        if fallbackNextOpen == nil, let nextOpen = body[valStart...].range(of: nextOpenTag) {
            fallbackNextOpen = nextOpen.lowerBound
        }
        if let end = fallbackNextOpen {
            return (end, end)
        }
        return nil
    }

    private static func jsonLiteral(_ string: String) -> String {
        if let d = try? JSONSerialization.data(withJSONObject: string, options: [.fragmentsAllowed, .withoutEscapingSlashes]),
           let s = String(data: d, encoding: .utf8) {
            return s
        }
        return "\"\""
    }

    /// Converts a raw parameter body into a JSON value. When the tool schema declares the
    /// parameter as a string the body is NEVER interpreted: file contents that happen to be
    /// valid JSON (package.json, a bare number, ...) must reach the tool unchanged.
    private static func jsonValue(_ raw: String, type: ToolSchemas.ParamType) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        switch type {
        case .string:
            return jsonLiteral(raw)
        case .boolean:
            let low = trimmed.lowercased()
            if low == "true" || low == "\"true\"" || low == "1" { return "true" }
            if low == "false" || low == "\"false\"" || low == "0" { return "false" }
            return jsonLiteral(raw)
        case .integer:
            let unquoted = (trimmed.hasPrefix("\"") && trimmed.hasSuffix("\"") && trimmed.count >= 2)
                ? String(trimmed.dropFirst().dropLast()).trimmingCharacters(in: .whitespacesAndNewlines)
                : trimmed
            if let iv = Int64(unquoted) { return String(iv) }
            if let dv = Double(unquoted), dv.isFinite, let iv = Int64(exactly: dv) { return String(iv) }
            return jsonLiteral(raw)
        case .number:
            let unquoted = (trimmed.hasPrefix("\"") && trimmed.hasSuffix("\"") && trimmed.count >= 2)
                ? String(trimmed.dropFirst().dropLast()).trimmingCharacters(in: .whitespacesAndNewlines)
                : trimmed
            if let iv = Int64(unquoted) { return String(iv) }
            if let dv = Double(unquoted), dv.isFinite {
                if let d = try? JSONSerialization.data(withJSONObject: dv, options: [.fragmentsAllowed]),
                   let s = String(data: d, encoding: .utf8) {
                    return s
                }
            }
            return jsonLiteral(raw)
        case .structured, .unknown:
            let low = trimmed.lowercased()
            if low == "true" { return "true" }
            if low == "false" { return "false" }
            if low == "null" || low == "none" { return "null" }
            if let data = trimmed.data(using: .utf8),
               let value = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]),
               let out = try? JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed, .withoutEscapingSlashes]),
               let str = String(data: out, encoding: .utf8) {
                return str
            }
            return jsonLiteral(raw)
        }
    }

    /// Per-function parameter type table extracted from the request's OpenAI tool list.
    struct ToolSchemas {
        enum ParamType { case string, boolean, integer, number, structured, unknown }
        private var table: [String: [String: ParamType]] = [:]

        init(_ tools: [QWJson]?) {
            for tool in tools ?? [] {
                let fn = tool.get("function").isUndefined ? tool : tool.get("function")
                guard case .string(let name) = fn.get("name") else { continue }
                var params: [String: ParamType] = [:]
                let paramRoot = fn.get("parameters").isUndefined ? fn.get("input_schema") : fn.get("parameters")
                if case .object(let props) = paramRoot.get("properties") {
                    for (key, schema) in props { params[key] = Self.classify(schema) }
                }
                table[name] = params
            }
        }

        func type(function: String, parameter: String) -> ParamType {
            table[function]?[parameter] ?? .unknown
        }

        func coerceDictionary(_ dict: [String: Any], function: String) -> [String: Any] {
            guard let params = table[function] else { return dict }
            var out = dict
            for (k, v) in dict {
                guard let pType = params[k] else { continue }
                if let strVal = v as? String {
                    let trimmed = strVal.trimmingCharacters(in: .whitespacesAndNewlines)
                    switch pType {
                    case .boolean:
                        let low = trimmed.lowercased()
                        if low == "true" || low == "1" { out[k] = true }
                        else if low == "false" || low == "0" { out[k] = false }
                    case .integer:
                        if let iv = Int64(trimmed) { out[k] = iv }
                        else if let dv = Double(trimmed), dv.isFinite, let iv = Int64(exactly: dv) { out[k] = iv }
                    case .number:
                        if let iv = Int64(trimmed) { out[k] = iv }
                        else if let dv = Double(trimmed), dv.isFinite { out[k] = dv }
                    default:
                        break
                    }
                }
            }
            return out
        }

        private static func classify(_ schema: QWJson) -> ParamType {
            switch schema.get("type") {
            case .string(let t):
                switch t.lowercased() {
                case "string": return .string
                case "boolean", "bool": return .boolean
                case "integer", "int": return .integer
                case "number", "float", "double": return .number
                default: return .structured
                }
            case .array(let types):
                if types.contains(where: { $0.isString("string") }) { return .string }
                if types.contains(where: { $0.isString("boolean") }) { return .boolean }
                if types.contains(where: { $0.isString("integer") }) { return .integer }
                if types.contains(where: { $0.isString("number") }) { return .number }
                return .structured
            default:
                break
            }
            for key in ["anyOf", "oneOf"] {
                if case .array(let options) = schema.get(key) {
                    if options.contains(where: { $0.get("type").isString("string") }) { return .string }
                    if options.contains(where: { $0.get("type").isString("boolean") }) { return .boolean }
                    if options.contains(where: { $0.get("type").isString("integer") }) { return .integer }
                    if options.contains(where: { $0.get("type").isString("number") }) { return .number }
                }
            }
            return .unknown
        }
    }
}

/// Incremental splitter for model output. Starts in reasoning mode when the
/// generation prompt ended inside the thinking block; switches on close/call
/// tags. Tags split across token pieces are held back until classifiable.
public final class QWChatStreamParser: @unchecked Sendable {
    public enum Delta: Equatable, Sendable {
        case reasoning(String)
        case content(String)
    }

    private enum Mode { case reasoning, content, toolCall }

    private static let thinkOpenTags = ["<think>", "<ernie:think>"]
    private static let thinkCloseTags = ["</think>", "</ernie:think>"]
    private static let toolOpen = "<tool_call>"

    private var mode: Mode
    private let allowToolCalls: Bool
    private var allowReasoningTags: Bool
    private var buffer = ""
    private var leadingPending = true
    private var contentEmitted = false

    public private(set) var reasoning = ""
    public private(set) var content = ""
    public private(set) var toolText = ""

    public init(startsInReasoning: Bool, allowToolCalls: Bool = true, allowReasoningTags: Bool = true) {
        self.mode = (startsInReasoning && allowReasoningTags) ? .reasoning : .content
        self.allowToolCalls = allowToolCalls
        self.allowReasoningTags = allowReasoningTags
    }

    public func setStartsInReasoning(_ startsInReasoning: Bool, allowReasoningTags: Bool? = nil) {
        guard buffer.isEmpty && reasoning.isEmpty && content.isEmpty && toolText.isEmpty else { return }
        if let allowReasoningTags { self.allowReasoningTags = allowReasoningTags }
        mode = (startsInReasoning && self.allowReasoningTags) ? .reasoning : .content
    }

    public func feed(_ piece: String) -> [Delta] {
        buffer += piece
        return drain(final: false)
    }

    public func finish() -> [Delta] {
        drain(final: true)
    }

    /// Longest suffix of `text` that is a proper prefix of one of `tags`.
    private static func heldSuffix(_ text: String, tags: [String]) -> Int {
        var best = 0
        for tag in tags {
            let maxLen = min(tag.count - 1, text.count)
            guard maxLen > best else { continue }
            for len in stride(from: maxLen, to: best, by: -1) where text.hasSuffix(String(tag.prefix(len))) {
                best = len
                break
            }
        }
        return best
    }

    private func emit(_ text: Substring, as mode: Mode, into out: inout [Delta]) {
        guard !text.isEmpty else { return }
        let s = String(text)
        if mode == .reasoning {
            reasoning += s
            out.append(.reasoning(s))
        } else {
            content += s
            contentEmitted = true
            out.append(.content(s))
        }
    }

    private static func earliestTagRange(in text: String, tags: [String]) -> Range<String.Index>? {
        var earliest: Range<String.Index>? = nil
        for tag in tags {
            if let r = text.range(of: tag), earliest.map({ r.lowerBound < $0.lowerBound }) ?? true {
                earliest = r
            }
        }
        return earliest
    }

    private func drain(final: Bool) -> [Delta] {
        var out: [Delta] = []
        while true {
            switch mode {
            case .toolCall:
                if buffer.hasPrefix(Self.toolOpen) {
                    let afterOpen = buffer.dropFirst(Self.toolOpen.count)
                    if let close = QWToolCallParser.matchingToolCallClose(in: afterOpen) {
                        let inner = afterOpen[..<close.lowerBound]
                        let isClosed = QWToolCallParser.isBalancedToolPayload(inner)
                            || (!QWToolCallParser.hasUnclosedLiteralContainer(inner)
                                && (final || !afterOpen[close.upperBound...].trimmingCharacters(in: .whitespacesAndNewlines).isEmpty))
                        if isClosed {
                            toolText += String(buffer[..<close.upperBound])
                            buffer = String(afterOpen[close.upperBound...])
                            mode = .content
                            leadingPending = false
                            continue
                        }
                    }
                }
                if final {
                    toolText += buffer
                    buffer = ""
                }
                return out

            case .reasoning:
                if leadingPending {
                    buffer = String(buffer.drop(while: { $0.isWhitespace }))
                    if let openTag = Self.thinkOpenTags.first(where: { buffer.hasPrefix($0) }) {
                        buffer.removeFirst(openTag.count)
                        continue
                    }
                    if buffer.isEmpty { return out }
                    if !final && Self.thinkOpenTags.contains(where: { $0.hasPrefix(buffer) }) { return out }
                    leadingPending = false
                }
                let close = Self.earliestTagRange(in: buffer, tags: Self.thinkCloseTags)
                let tool = allowToolCalls ? buffer.range(of: Self.toolOpen) : nil
                if let close, tool.map({ close.lowerBound < $0.lowerBound }) ?? true {
                    emit(buffer[..<close.lowerBound], as: .reasoning, into: &out)
                    buffer = String(buffer[close.upperBound...])
                    mode = .content
                    leadingPending = false
                    continue
                }
                if let tool {
                    emit(buffer[..<tool.lowerBound], as: .reasoning, into: &out)
                    buffer = String(buffer[tool.lowerBound...])
                    mode = .toolCall
                    continue
                }
                let activeTags = allowToolCalls ? (Self.thinkCloseTags + [Self.toolOpen]) : Self.thinkCloseTags
                let hold = final ? 0 : Self.heldSuffix(buffer, tags: activeTags)
                let cut = buffer.index(buffer.endIndex, offsetBy: -hold)
                emit(buffer[..<cut], as: .reasoning, into: &out)
                buffer = String(buffer[cut...])
                return out

            case .content:
                if leadingPending {
                    if allowReasoningTags && !contentEmitted {
                        if let nonWS = buffer.firstIndex(where: { !$0.isWhitespace }) {
                            let afterWS = buffer[nonWS...]
                            if let openTag = Self.thinkOpenTags.first(where: { afterWS.hasPrefix($0) }) {
                                buffer = String(afterWS.dropFirst(openTag.count))
                                mode = .reasoning
                                leadingPending = true
                                continue
                            }
                            if !final && Self.thinkOpenTags.contains(where: { $0.hasPrefix(afterWS) }) {
                                return out
                            }
                            leadingPending = false
                        } else if !final && buffer.count < 16 {
                            return out
                        } else {
                            leadingPending = false
                        }
                    } else {
                        leadingPending = false
                    }
                }
                if allowToolCalls, let tool = buffer.range(of: Self.toolOpen) {
                    emit(buffer[..<tool.lowerBound], as: .content, into: &out)
                    buffer = String(buffer[tool.lowerBound...])
                    mode = .toolCall
                    continue
                }
                let activeTags = allowToolCalls ? [Self.toolOpen] : []
                let hold = final ? 0 : Self.heldSuffix(buffer, tags: activeTags)
                let cut = buffer.index(buffer.endIndex, offsetBy: -hold)
                emit(buffer[..<cut], as: .content, into: &out)
                buffer = String(buffer[cut...])
                return out
            }
        }
    }

    /// Final structured result. Call after `finish()`.
    public func result(tools: [QWJson]?, truncated: Bool) -> QWToolCallParser.ParseResult {
        let (calls, _) = allowToolCalls
            ? QWToolCallParser.parseToolSegments(toolText, tools: tools, dropIncomplete: truncated)
            : ([], false)
        let trimmedContent = content.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedReasoning = reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
        let finish: String
        if !calls.isEmpty && !truncated {
            finish = "tool_calls"
        } else if truncated {
            finish = "length"
        } else {
            finish = "stop"
        }
        return QWToolCallParser.ParseResult(
            content: trimmedContent.isEmpty ? nil : trimmedContent,
            toolCalls: calls.isEmpty ? nil : calls,
            reasoningContent: trimmedReasoning.isEmpty ? nil : trimmedReasoning,
            finishReason: finish
        )
    }
}

