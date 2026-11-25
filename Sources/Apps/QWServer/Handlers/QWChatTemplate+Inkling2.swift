// QWChatTemplate+Inkling2.swift — Native Inkling-2 chat template wire renderer.
import Foundation
import QWToken

extension QWChatTemplate {
    private static let ink2EffortMap: [String: String] = [
        "none": "0",
        "minimal": "0.1",
        "low": "0.2",
        "medium": "0.7",
        "high": "0.9",
        "xhigh": "0.99",
        "max": "0.99",
    ]

    private static let ink2RoleTokens: [String: String] = [
        "user": "<|message_user|>",
        "assistant": "<|message_model|>",
        "system": "<|message_system|>",
        "tool": "<|message_tool|>",
    ]

    func renderInkling2(messages: [QWJson], options: QWChatTemplateOptions) throws -> String {
        guard !messages.isEmpty else { throw QWChatTemplateError.emptyConversation }
        let effortLiteral = try Self.resolveInk2EffortLiteral(options: options)
        var out = ""
        out.reserveCapacity(1024)

        if case .array(let toolList) = options.tools, !toolList.isEmpty {
            let specs: [[String: Any]] = toolList.map { tool in
                let fnVal = tool.get("function")
                let fn = fnVal.isUndefined ? tool : fnVal
                let desc = fn.get("description").stringValue ?? ""
                let name = fn.get("name").stringValue ?? ""
                let typeStr = tool.get("type").stringValue ?? "function"
                let paramsObj: Any = fn.get("parameters").isUndefined
                    ? [String: Any]()
                    : Self.foundationObject(from: fn.get("parameters"))
                return [
                    "description": desc,
                    "name": name,
                    "parameters": paramsObj,
                    "type": typeStr,
                ]
            }
            let data = try JSONSerialization.data(withJSONObject: specs, options: [.sortedKeys, .withoutEscapingSlashes])
            let jsonText = String(data: data, encoding: .utf8) ?? "[]"
            out += "<|message_system|>tool_declare<|content_xml|>" + jsonText + "<|end_message|>"
        }

        var effortEmitted = false
        for msg in messages {
            guard let role = msg.get("role").stringValue, let roleTok = Self.ink2RoleTokens[role] else {
                throw QWChatTemplateRenderError.invalid("Unknown message role: \(msg.get("role").stringValue ?? "")")
            }
            if !effortEmitted && role != "system" {
                out += "<|message_system|><|content_text|>Thinking effort level: \(effortLiteral)<|end_message|>"
                effortEmitted = true
            }

            var header = roleTok
            if role == "tool" {
                if let toolName = msg.get("name").stringValue, !toolName.isEmpty {
                    header += toolName
                } else if let tcId = msg.get("tool_call_id").stringValue, !tcId.isEmpty {
                    header += Self.lookupInk2ToolName(toolCallId: tcId, in: messages)
                }
            } else if role == "assistant", let aName = msg.get("name").stringValue, !aName.isEmpty {
                header += aName
            }

            let rc = msg.get("reasoning_content").stringValue ?? ""
            if role == "assistant" && !rc.isEmpty {
                out += "\(header)<|content_thinking|>\(rc)<|end_message|>"
            }

            let hasCalls: Bool = {
                if case .array(let arr) = msg.get("tool_calls"), !arr.isEmpty { return true }
                return false
            }()
            let contentVal = msg.get("content")
            if case .string(let text) = contentVal {
                let marker = "<|vision_start|><|image_pad|><|vision_end|>"
                if (role == "user" || role == "tool") && text.contains(marker) {
                    let segments = text.components(separatedBy: marker)
                    for (idx, seg) in segments.enumerated() {
                        if idx > 0 {
                            out += "\(header)<|content_image|><|unused_200054|><|end_message|>"
                        }
                        if !seg.isEmpty {
                            out += "\(header)<|content_text|>\(seg)<|end_message|>"
                        }
                    }
                } else if !text.isEmpty || !(role == "assistant" && (hasCalls || !rc.isEmpty)) {
                    out += "\(header)<|content_text|>\(text)<|end_message|>"
                }
            } else if case .array(let parts) = contentVal, !parts.isEmpty {
                for part in parts {
                    if case .string(let s) = part {
                        out += "\(header)<|content_text|>\(s)<|end_message|>"
                        continue
                    }
                    let pType = part.get("type").stringValue
                    if pType == nil || pType == "text" || pType == "input_text" {
                        let txt = part.get("text").stringValue ?? ""
                        out += "\(header)<|content_text|>\(txt)<|end_message|>"
                    } else if pType == "image" || pType == "input_image" || pType == "image_url" {
                        guard role == "user" || role == "tool" else {
                            throw QWChatTemplateRenderError.invalid("Image content is only supported for user and tool messages")
                        }
                        out += "\(header)<|content_image|><|unused_200054|><|end_message|>"
                    } else if pType == "audio" || pType == "input_audio" || pType == "audio_url" {
                        throw QWChatTemplateRenderError.invalid("Inkling 2 does not accept audio input")
                    } else {
                        throw QWChatTemplateRenderError.invalid("Unsupported content part type: \(pType ?? "")")
                    }
                }
            } else if case .null = contentVal, !(role == "assistant" && (hasCalls || !rc.isEmpty)) {
                out += "\(header)<|content_text|><|end_message|>"
            } else if contentVal.isUndefined, !(role == "assistant" && (hasCalls || !rc.isEmpty)) {
                out += "\(header)<|content_text|><|end_message|>"
            }

            if role == "assistant", case .array(let calls) = msg.get("tool_calls"), !calls.isEmpty {
                let aName = msg.get("name").stringValue
                for tc in calls {
                    let fn = tc.get("function")
                    guard let fnName = fn.get("name").stringValue else {
                        throw QWChatTemplateRenderError.invalid("tool call function name must be a string")
                    }
                    if let aName, !aName.isEmpty, aName != fnName {
                        throw QWChatTemplateRenderError.invalid("assistant name \(aName) does not match its tool call \(fnName)")
                    }
                    let argsObj = try Self.resolveCallArgsDict(fn.get("arguments"))
                    let pairs = argsObj.sorted { $0.key < $1.key }.map { (k, v) in
                        let kj = Self.encodeJSONStringLiteral(k)
                        let vj = Self.encodeCompactSortedJSONValue(v)
                        return "\(kj):\(vj)"
                    }.joined(separator: ",")
                    let fnJson = Self.encodeJSONStringLiteral(fnName)
                    out += "<|message_model|>\(fnName)<|content_invoke_tool_json|>{\"name\":\(fnJson),\"args\":{\(pairs)}}<|end_message|>"
                }
            }

            if role == "assistant" {
                out += "<|content_model_end_sampling|>"
            }
        }

        if !effortEmitted {
            out += "<|message_system|><|content_text|>Thinking effort level: \(effortLiteral)<|end_message|>"
        }
        return out
    }

    private static func resolveInk2EffortLiteral(options: QWChatTemplateOptions) throws -> String {
        if case .bool(false) = options.enableThinking { return "0" }
        let val = options.reasoningEffort
        switch val {
        case .undefined, .null:
            return "0.9"
        case .string(let s):
            let key = s.trimmingCharacters(in: .whitespacesAndNewlines)
            if let mapped = ink2EffortMap[key] { return mapped }
            throw QWChatTemplateRenderError.invalid("Unknown reasoning_effort: \(s)")
        case .int(let i):
            let d = Double(i)
            guard d >= 0.0 && d <= 0.99 else {
                throw QWChatTemplateRenderError.invalid("reasoning_effort must be in [0.0, 0.99]")
            }
            return d == 0.0 ? "0" : String(d)
        case .double(let d):
            guard d >= 0.0 && d <= 0.99 else {
                throw QWChatTemplateRenderError.invalid("reasoning_effort must be in [0.0, 0.99]")
            }
            return d == 0.0 ? "0" : String(d)
        default:
            throw QWChatTemplateRenderError.invalid("Unsupported reasoning_effort type")
        }
    }

    private static func lookupInk2ToolName(toolCallId: String, in messages: [QWJson]) -> String {
        for prev in messages {
            guard prev.get("role").stringValue == "assistant",
                  case .array(let calls) = prev.get("tool_calls") else { continue }
            for tc in calls where tc.get("id").stringValue == toolCallId {
                if let name = tc.get("function").get("name").stringValue { return name }
            }
        }
        return ""
    }

    static func resolveCallArgsDict(_ raw: QWJson) throws -> [String: QWJson] {
        switch raw {
        case .undefined, .null:
            return [:]
        case .object(let pairs):
            var dict: [String: QWJson] = [:]
            for (k, v) in pairs { dict[k] = v }
            return dict
        case .string(let s):
            let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { return [:] }
            let parsed = try QWJson.parse(trimmed)
            guard case .object(let pairs) = parsed else {
                throw QWChatTemplateRenderError.invalid("tool call arguments must be an object")
            }
            var dict: [String: QWJson] = [:]
            for (k, v) in pairs { dict[k] = v }
            return dict
        default:
            throw QWChatTemplateRenderError.invalid("tool call arguments must be an object")
        }
    }

    static func foundationObject(from json: QWJson) -> Any {
        switch json {
        case .undefined, .null: return NSNull()
        case .bool(let b): return b
        case .int(let i): return i
        case .double(let d): return d
        case .string(let s): return s
        case .array(let arr): return arr.map { foundationObject(from: $0) }
        case .object(let pairs):
            var dict: [String: Any] = [:]
            for (k, v) in pairs where !v.isUndefined {
                dict[k] = foundationObject(from: v)
            }
            return dict
        }
    }

    static func encodeJSONStringLiteral(_ s: String) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: [s], options: [.withoutEscapingSlashes]),
              let arrStr = String(data: data, encoding: .utf8),
              arrStr.count >= 2 else { return "\"\"" }
        return String(arrStr.dropFirst().dropLast())
    }

    static func encodeCompactSortedJSONValue(_ json: QWJson) -> String {
        switch json {
        case .string(let s): return encodeJSONStringLiteral(s)
        case .int(let i): return String(i)
        case .double(let d): return String(d)
        case .bool(let b): return b ? "true" : "false"
        case .null, .undefined: return "null"
        case .array, .object:
            let obj = foundationObject(from: json)
            guard let data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys, .withoutEscapingSlashes]),
                  let str = String(data: data, encoding: .utf8) else { return "null" }
            return str
        }
    }
}

public enum QWChatTemplateRenderError: Error, CustomStringConvertible, Sendable {
    case invalid(String)
    public var description: String {
        switch self { case .invalid(let m): return m }
    }
}

private extension QWJson {
    var stringValue: String? {
        if case .string(let s) = self { return s }
        return nil
    }
}
