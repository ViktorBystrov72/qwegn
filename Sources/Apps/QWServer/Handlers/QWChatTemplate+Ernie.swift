// QWChatTemplate+Ernie.swift — Authoritative native Swift ERNIE chatv2p3 wire renderer:
// preserves Python-spaced sorted JSON `tojson`, JSON-string `tool_calls.arguments` decoding,
// `<ernie:think>` / `thoughts` extraction, and strict `reasoning_effort` validation over compat Jinja.
import Foundation
import QWToken

extension QWChatTemplate {
    func renderErnie(messages: [QWJson], options: QWChatTemplateOptions) throws -> String {
        guard !messages.isEmpty else { throw QWChatTemplateError.emptyConversation }
        let thinkingOff: Bool = {
            switch options.enableThinking {
            case .bool(false): return true
            case .int(0): return true
            case .string(let s): return s.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "false"
            default: return false
            }
        }()
        let toolList: [QWJson]
        if case .array(let arr) = options.tools, !arr.isEmpty {
            toolList = arr
        } else {
            toolList = []
        }

        var out = "<|ERNIE_MASK|><|ernie:im_start|>system\n"
        if messages[0].get("role").ernieString == "system" {
            out += "<ernie:system_setting>"
            let sysText = Self.ernieTextContent(messages[0].get("content"))
            switch messages[0].get("content") {
            case .string, .array:
                out += "\n" + sysText
            default:
                break
            }
            out += "\n</ernie:system_setting>\n"
        }
        out += "<ernie:global_setting>"
        if !thinkingOff {
            let effort = try Self.resolveErnieEffort(options.reasoningEffort)
            out += "\nreasoning_effort=" + effort
        }
        if !toolList.isEmpty {
            out += "\nparallel_tool_calls=True"
        }
        out += "\n</ernie:global_setting>"
        if !toolList.isEmpty {
            let toolStrings = toolList.map { Self.encodePythonSortedJSONValue($0) }
            out += "\n<ernie:tool_list>\n[" + toolStrings.joined(separator: ", ") + "]\n</ernie:tool_list>"
        }
        out += "<|ernie:im_end|>\n"

        var body = ""
        for (idx, msg) in messages.enumerated() {
            guard let role = msg.get("role").ernieString else {
                throw QWChatTemplateRenderError.invalid("Unknown message role")
            }
            if role == "user" || (role == "system" && idx > 0) {
                body += "<|ernie:im_start|>\(role)\n\(Self.ernieTextContent(msg.get("content")))<|ernie:im_end|>\n"
            } else if role == "assistant" {
                body += try Self.renderErnieAssistant(msg, thinkingOff: thinkingOff)
            } else if role == "tool" {
                if idx == 0 || messages[idx - 1].get("role").ernieString != "tool" {
                    body += "<|ernie:im_start|>tool"
                }
                body += "\n<ernie:tool_output>\(Self.ernieTextContent(msg.get("content")))</ernie:tool_output>"
                if idx == messages.count - 1 || messages[idx + 1].get("role").ernieString != "tool" {
                    body += "<|ernie:im_end|>\n"
                }
            } else if role != "system" {
                throw QWChatTemplateRenderError.invalid("Unsupported role: \(role)")
            }
        }
        out += Self.stripOuterNewlines(body)
        if options.addGenerationPrompt {
            out += "\n<|ernie:im_start|>assistant\n" + (thinkingOff ? "<ernie:think></ernie:think>" : "<ernie:think>")
        }
        return out
    }

    private static func resolveErnieEffort(_ raw: QWJson) throws -> String {
        switch raw {
        case .undefined, .null:
            return "high"
        case .string(let s):
            switch s.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
            case "low", "minimal", "min", "light", "fast", "quick", "none", "off", "false", "disabled", "disable":
                return "low"
            case "medium", "mid", "med", "normal", "auto", "default":
                return "medium"
            case "high", "xhigh", "max", "maximum", "deep":
                return "high"
            default:
                throw QWChatTemplateRenderError.invalid("reasoning_effort must be one of low, medium, high (got \(s))")
            }
        default:
            throw QWChatTemplateRenderError.invalid("Unsupported reasoning_effort type")
        }
    }

    private static func renderErnieAssistant(_ msg: QWJson, thinkingOff: Bool) throws -> String {
        var content = ernieTextContent(msg.get("content"))
        var reasoning = msg.get("thoughts").ernieString ?? msg.get("reasoning_content").ernieString
        if reasoning == nil, let endRange = content.range(of: "</ernie:think>") {
            let rStart = content[..<endRange.lowerBound].range(of: "<ernie:think>", options: .backwards)?.upperBound
                ?? content.startIndex
            reasoning = String(content[rStart..<endRange.lowerBound])
            let lastEnd = content.range(of: "</ernie:think>", options: .backwards)?.upperBound ?? endRange.upperBound
            content = String(content[lastEnd...])
        }
        var out = "<|ernie:im_start|>assistant\n<ernie:think>"
        if !thinkingOff, let r = reasoning, !r.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            out += r
        }
        out += "</ernie:think>" + content
        if case .array(let calls) = msg.get("tool_calls"), !calls.isEmpty {
            out += "<ernie:tool_call>\n"
            for tc in calls {
                let fnVal = tc.get("function")
                let fn = fnVal.isUndefined ? tc : fnVal
                guard let fnName = fn.get("name").ernieString, !fnName.isEmpty else {
                    throw QWChatTemplateRenderError.invalid("tool call function name must be a string")
                }
                let argsObj = try resolveCallArgsDict(fn.get("arguments"))
                out += "<invoke name=\"\(fnName)\">"
                for (k, v) in argsObj.sorted(by: { $0.key < $1.key }) {
                    let valText = v.ernieString ?? encodePythonSortedJSONValue(v)
                    out += "<parameter name=\"\(k)\">\(valText)</parameter>"
                }
                out += "</invoke>\n"
            }
            out += "</ernie:tool_call>"
        }
        out += "<|ernie:im_end|>\n"
        return out
    }

    private static func ernieTextContent(_ val: QWJson) -> String {
        if case .string(let s) = val { return s }
        if case .array(let arr) = val {
            var parts: [String] = []
            for item in arr {
                let kind = item.get("type").ernieString
                if kind == "text" || kind == "input_text" || kind == nil,
                   let txt = item.get("text").ernieString {
                    parts.append(txt)
                }
            }
            return parts.joined()
        }
        return ""
    }

    private static func stripOuterNewlines(_ s: String) -> String {
        var lo = s.startIndex
        while lo < s.endIndex && s[lo] == "\n" { lo = s.index(after: lo) }
        var hi = s.endIndex
        while hi > lo && s[s.index(before: hi)] == "\n" { hi = s.index(before: hi) }
        return String(s[lo..<hi])
    }

    private static func encodePythonSortedJSONValue(_ json: QWJson) -> String {
        switch json {
        case .string(let s):
            return encodeJSONStringLiteral(s)
        case .int(let i):
            return String(i)
        case .double(let d):
            return String(d)
        case .bool(let b):
            return b ? "true" : "false"
        case .null, .undefined:
            return "null"
        case .array(let arr):
            return "[" + arr.map { encodePythonSortedJSONValue($0) }.joined(separator: ", ") + "]"
        case .object(let pairs):
            let body = pairs
                .filter { !$0.1.isUndefined }
                .sorted { $0.0 < $1.0 }
                .map { "\(encodeJSONStringLiteral($0.0)): \(encodePythonSortedJSONValue($0.1))" }
                .joined(separator: ", ")
            return "{" + body + "}"
        }
    }
}

private extension QWJson {
    var ernieString: String? {
        if case .string(let s) = self { return s }
        return nil
    }
}
