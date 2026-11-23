// QWToolCallParser+Inkling2.swift — Inkling-2 channel marker detection and tool-call parsing.
import Foundation
import QWToken

extension QWToolCallParser {
    static let ink2MessageModel = "<|message_model|>"
    static let ink2ContentThinking = "<|content_thinking|>"
    static let ink2ContentText = "<|content_text|>"
    static let ink2InvokeToolJson = "<|content_invoke_tool_json|>"
    static let ink2EndMessage = "<|end_message|>"
    static let ink2EndSampling = "<|content_model_end_sampling|>"

    static let ink2AllMarkers: [String] = [
        ink2MessageModel,
        ink2ContentThinking,
        ink2ContentText,
        ink2InvokeToolJson,
        ink2EndMessage,
        ink2EndSampling,
    ]

    static func parseInk2ToolPayloads(
        _ payloads: [String],
        trailingIncomplete: String?,
        tools: [QWJson]?,
        dropIncomplete: Bool
    ) -> [ToolCall] {
        var candidates = payloads
        if !dropIncomplete, let trailing = trailingIncomplete {
            candidates.append(trailing)
        }
        guard !candidates.isEmpty else { return [] }
        let schemas = ToolSchemas(tools)
        var calls: [ToolCall] = []
        for raw in candidates {
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.hasPrefix("{"), trimmed.hasSuffix("}"),
                  let data = trimmed.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let name = (obj["name"] as? String) ?? (obj["function"] as? String),
                  !name.isEmpty else { continue }
            let rawArgs = obj["args"] ?? obj["arguments"] ?? obj["parameters"]
            let argsStr: String
            if let dict = rawArgs as? [String: Any] {
                let coerced = schemas.coerceDictionary(dict, function: name)
                if let d = try? JSONSerialization.data(withJSONObject: coerced, options: [.sortedKeys, .withoutEscapingSlashes]),
                   let s = String(data: d, encoding: .utf8) {
                    argsStr = s
                } else {
                    argsStr = "{}"
                }
            } else if let str = rawArgs as? String {
                argsStr = str
            } else {
                argsStr = "{}"
            }
            calls.append(ToolCall(function: ToolCallFunction(name: name, arguments: argsStr)))
        }
        return calls
    }
}

extension QWChatStreamParser {
    enum Ink2ChannelState {
        case awaitingChannel
        case thinking
        case text
        case tool
    }

    static func earliestTagMatch(in text: String, tags: [String]) -> (tag: String, range: Range<String.Index>)? {
        var best: (tag: String, range: Range<String.Index>)? = nil
        for tag in tags {
            if let r = text.range(of: tag), best.map({ r.lowerBound < $0.range.lowerBound }) ?? true {
                best = (tag, r)
            }
        }
        return best
    }

    func drainInk2(final: Bool, out: inout [Delta]) -> Bool {
        while true {
            if let eosRange = buffer.range(of: QWToolCallParser.ink2EndSampling) {
                let prefix = String(buffer[..<eosRange.lowerBound])
                buffer = prefix
                let _ = drainInk2(final: true, out: &out)
                buffer = ""
                return true
            }

            switch ink2State {
            case .awaitingChannel:
                let channelTags = [
                    QWToolCallParser.ink2ContentThinking,
                    QWToolCallParser.ink2ContentText,
                    QWToolCallParser.ink2InvokeToolJson,
                    QWToolCallParser.ink2EndMessage,
                ]
                if let match = Self.earliestTagMatch(in: buffer, tags: channelTags) {
                    buffer.removeSubrange(..<match.range.upperBound)
                    switch match.tag {
                    case QWToolCallParser.ink2ContentThinking:
                        ink2State = .thinking
                    case QWToolCallParser.ink2ContentText:
                        ink2State = .text
                    case QWToolCallParser.ink2InvokeToolJson:
                        ink2State = .tool
                    default:
                        ink2State = .awaitingChannel
                    }
                    continue
                }
                if final {
                    var rem = buffer.replacingOccurrences(of: QWToolCallParser.ink2MessageModel, with: "")
                    rem = rem.replacingOccurrences(of: QWToolCallParser.ink2EndSampling, with: "")
                    emitSubstring(rem[...], as: .content, into: &out)
                    buffer = ""
                }
                return true

            case .thinking, .text:
                let stopTags = [
                    QWToolCallParser.ink2EndMessage,
                    QWToolCallParser.ink2MessageModel,
                    QWToolCallParser.ink2ContentThinking,
                    QWToolCallParser.ink2ContentText,
                    QWToolCallParser.ink2InvokeToolJson,
                ]
                let emitAsReasoning = (ink2State == .thinking)
                if let match = Self.earliestTagMatch(in: buffer, tags: stopTags) {
                    if match.range.lowerBound > buffer.startIndex {
                        emitSubstring(buffer[..<match.range.lowerBound], as: emitAsReasoning ? .reasoning : .content, into: &out)
                    }
                    if match.tag == QWToolCallParser.ink2EndMessage || match.tag == QWToolCallParser.ink2MessageModel {
                        buffer.removeSubrange(..<match.range.upperBound)
                        ink2State = .awaitingChannel
                    } else {
                        buffer.removeSubrange(..<match.range.lowerBound)
                        ink2State = .awaitingChannel
                    }
                    continue
                }
                let hold = final ? 0 : heldMarkerSuffixCount(buffer, tags: QWToolCallParser.ink2AllMarkers)
                let cut = buffer.index(buffer.endIndex, offsetBy: -hold)
                emitSubstring(buffer[..<cut], as: emitAsReasoning ? .reasoning : .content, into: &out)
                buffer = String(buffer[cut...])
                return true

            case .tool:
                let endTags = [QWToolCallParser.ink2EndMessage, QWToolCallParser.ink2MessageModel]
                if let match = Self.earliestTagMatch(in: buffer, tags: endTags) {
                    let payload = String(buffer[..<match.range.lowerBound])
                    if allowToolCalls {
                        ink2ToolPayloads.append(payload)
                    }
                    buffer.removeSubrange(..<match.range.upperBound)
                    ink2State = .awaitingChannel
                    continue
                }
                if final {
                    if allowToolCalls && !buffer.isEmpty {
                        ink2IncompleteToolPayload = buffer
                    }
                    buffer = ""
                }
                return true
            }
        }
    }
}
