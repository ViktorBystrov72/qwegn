// QWServerSchema+OpenAIResponses.swift — OpenAI /v1/responses request/resource DTOs
// matching OpenAI Responses API schemas and streaming event contracts.

import Foundation
import QWCore
import QWToken

public struct OpenAIResponseCreateRequest: Decodable, Sendable {
    public struct ReasoningOption: Codable, Sendable {
        public let effort: String?
        public let summary: String?
    }

    public struct TextOption: Codable, Sendable {
        public let format: QWJson?
        public let verbosity: String?
    }

    public let model: String?
    public let input: QWJson?
    public let previousResponseId: String?
    public let instructions: String?
    public let tools: [QWJson]?
    public let toolChoice: QWJson?
    public let parallelToolCalls: Bool?
    public let maxOutputTokens: Int?
    public let maxToolCalls: Int?
    public let temperature: Double?
    public let topP: Double?
    public let presencePenalty: Double?
    public let frequencyPenalty: Double?
    public let topLogprobs: Int?
    public let reasoning: ReasoningOption?
    public let text: TextOption?
    public let truncation: String?
    public let stream: Bool?
    public let background: Bool?
    public let store: Bool?
    public let serviceTier: String?
    public let metadata: [String: String]?
    public let safetyIdentifier: String?
    public let promptCacheKey: String?
    public let user: String?
    public let cachePrompt: Bool?
    public let cacheSalt: String?
    public let reasoningBudget: Int?
    public let enableThinking: Bool?

    enum CodingKeys: String, CodingKey {
        case model, input, instructions, tools, temperature, stream, background, store
        case reasoning, text, truncation, metadata, user
        case previousResponseId = "previous_response_id"
        case toolChoice = "tool_choice"
        case parallelToolCalls = "parallel_tool_calls"
        case maxOutputTokens = "max_output_tokens"
        case maxToolCalls = "max_tool_calls"
        case topP = "top_p"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case topLogprobs = "top_logprobs"
        case serviceTier = "service_tier"
        case safetyIdentifier = "safety_identifier"
        case promptCacheKey = "prompt_cache_key"
        case cachePrompt = "cache_prompt"
        case cacheSalt = "cache_salt"
        case reasoningBudget = "reasoning_budget"
        case enableThinking = "enable_thinking"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        model = try c.decodeIfPresent(String.self, forKey: .model)
        input = try c.decodeIfPresent(QWJson.self, forKey: .input)
        previousResponseId = try c.decodeIfPresent(String.self, forKey: .previousResponseId)
        instructions = try c.decodeIfPresent(String.self, forKey: .instructions)
        tools = try c.decodeIfPresent([QWJson].self, forKey: .tools)
        toolChoice = try c.decodeIfPresent(QWJson.self, forKey: .toolChoice)
        parallelToolCalls = try c.decodeIfPresent(Bool.self, forKey: .parallelToolCalls)
        maxOutputTokens = try c.decodeIfPresent(Int.self, forKey: .maxOutputTokens)
        maxToolCalls = try c.decodeIfPresent(Int.self, forKey: .maxToolCalls)
        temperature = try c.decodeIfPresent(Double.self, forKey: .temperature)
        topP = try c.decodeIfPresent(Double.self, forKey: .topP)
        presencePenalty = try c.decodeIfPresent(Double.self, forKey: .presencePenalty)
        frequencyPenalty = try c.decodeIfPresent(Double.self, forKey: .frequencyPenalty)
        topLogprobs = try c.decodeIfPresent(Int.self, forKey: .topLogprobs)
        reasoning = try c.decodeIfPresent(ReasoningOption.self, forKey: .reasoning)
        text = try c.decodeIfPresent(TextOption.self, forKey: .text)
        truncation = try c.decodeIfPresent(String.self, forKey: .truncation)
        stream = try c.decodeIfPresent(Bool.self, forKey: .stream)
        background = try c.decodeIfPresent(Bool.self, forKey: .background)
        store = try c.decodeIfPresent(Bool.self, forKey: .store)
        serviceTier = try c.decodeIfPresent(String.self, forKey: .serviceTier)
        safetyIdentifier = try c.decodeIfPresent(String.self, forKey: .safetyIdentifier)
        promptCacheKey = try c.decodeIfPresent(String.self, forKey: .promptCacheKey)
        user = try c.decodeIfPresent(String.self, forKey: .user)
        cachePrompt = try c.decodeIfPresent(Bool.self, forKey: .cachePrompt)
        cacheSalt = try c.decodeIfPresent(String.self, forKey: .cacheSalt)
        reasoningBudget = try c.decodeIfPresent(Int.self, forKey: .reasoningBudget)
        enableThinking = try c.decodeIfPresent(Bool.self, forKey: .enableThinking)

        if c.contains(.metadata), (try? c.decodeNil(forKey: .metadata)) != true {
            let raw = try c.decode(QWJson.self, forKey: .metadata)
            guard case .object(let pairs) = raw else {
                throw DecodingError.typeMismatch(
                    [String: String].self,
                    DecodingError.Context(codingPath: c.codingPath + [CodingKeys.metadata],
                                          debugDescription: "metadata must be a JSON object")
                )
            }
            var coerced: [String: String] = [:]
            for (k, v) in pairs {
                switch v {
                case .string(let s): coerced[k] = s
                case .bool(let b): coerced[k] = b ? "true" : "false"
                case .int(let i): coerced[k] = String(i)
                case .double(let d): coerced[k] = String(d)
                case .null, .undefined: break
                case .array, .object:
                    throw DecodingError.typeMismatch(
                        String.self,
                        DecodingError.Context(codingPath: c.codingPath + [CodingKeys.metadata],
                                              debugDescription: "metadata['\(k)'] must be scalar")
                    )
                }
            }
            metadata = coerced
        } else {
            metadata = nil
        }
    }
}

public struct OpenAIResponseResource: Codable, Sendable {
    public struct IncompleteDetails: Codable, Sendable {
        public let reason: String
        public init(reason: String) { self.reason = reason }
    }

    public struct ErrorInfo: Codable, Sendable {
        public let code: String
        public let message: String
        public init(code: String, message: String) {
            self.code = code
            self.message = message
        }
    }

    public struct FunctionToolInfo: Codable, Sendable {
        public let type: String
        public let name: String
        public let description: String?
        public let parameters: QWJson?
        public let strict: Bool?

        enum CodingKeys: String, CodingKey {
            case type, name, description, parameters, strict
        }

        public init(name: String, description: String? = nil, parameters: QWJson? = nil, strict: Bool? = nil) {
            self.type = "function"
            self.name = name
            self.description = description
            self.parameters = parameters
            self.strict = strict
        }

        public func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(type, forKey: .type)
            try c.encode(name, forKey: .name)
            try encodeNullable(description, to: &c, forKey: .description)
            try encodeNullable(parameters, to: &c, forKey: .parameters)
            try encodeNullable(strict, to: &c, forKey: .strict)
        }
    }

    public struct TextConfig: Codable, Sendable {
        public let format: QWJson
        public let verbosity: String?

        public init(format: QWJson = .object([("type", .string("text"))]), verbosity: String? = nil) {
            self.format = format
            self.verbosity = verbosity
        }
    }

    public struct ReasoningInfo: Codable, Sendable {
        public let effort: String?
        public let summary: String?

        enum CodingKeys: String, CodingKey { case effort, summary }

        public init(effort: String?, summary: String?) {
            self.effort = effort
            self.summary = summary
        }

        public func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try encodeNullable(effort, to: &c, forKey: .effort)
            try encodeNullable(summary, to: &c, forKey: .summary)
        }
    }

    public struct UsageInfo: Codable, Sendable {
        public struct InputDetails: Codable, Sendable {
            public let cachedTokens: Int
            enum CodingKeys: String, CodingKey { case cachedTokens = "cached_tokens" }
            public init(cachedTokens: Int) { self.cachedTokens = cachedTokens }
        }
        public struct OutputDetails: Codable, Sendable {
            public let reasoningTokens: Int
            enum CodingKeys: String, CodingKey { case reasoningTokens = "reasoning_tokens" }
            public init(reasoningTokens: Int) { self.reasoningTokens = reasoningTokens }
        }

        public let inputTokens: Int
        public let outputTokens: Int
        public let totalTokens: Int
        public let inputTokensDetails: InputDetails
        public let outputTokensDetails: OutputDetails

        enum CodingKeys: String, CodingKey {
            case inputTokens = "input_tokens"
            case outputTokens = "output_tokens"
            case totalTokens = "total_tokens"
            case inputTokensDetails = "input_tokens_details"
            case outputTokensDetails = "output_tokens_details"
        }

        public init(inputTokens: Int, outputTokens: Int, cachedTokens: Int = 0, reasoningTokens: Int = 0) {
            self.inputTokens = inputTokens
            self.outputTokens = outputTokens
            self.totalTokens = inputTokens + outputTokens
            self.inputTokensDetails = InputDetails(cachedTokens: min(max(0, cachedTokens), inputTokens))
            self.outputTokensDetails = OutputDetails(reasoningTokens: min(max(0, reasoningTokens), outputTokens))
        }
    }

    public let id: String
    public let object: String
    public let createdAt: Int
    public let completedAt: Int?
    public let status: String
    public let incompleteDetails: IncompleteDetails?
    public let model: String
    public let previousResponseId: String?
    public let instructions: String?
    public let output: [QWJson]
    public let error: ErrorInfo?
    public let tools: [FunctionToolInfo]
    public let toolChoice: QWJson
    public let truncation: String
    public let parallelToolCalls: Bool
    public let text: TextConfig
    public let topP: Double
    public let presencePenalty: Double
    public let frequencyPenalty: Double
    public let topLogprobs: Int
    public let temperature: Double
    public let reasoning: ReasoningInfo?
    public let usage: UsageInfo?
    public let maxOutputTokens: Int?
    public let maxToolCalls: Int?
    public let store: Bool
    public let background: Bool
    public let serviceTier: String
    public let metadata: [String: String]
    public let safetyIdentifier: String?
    public let promptCacheKey: String?

    enum CodingKeys: String, CodingKey {
        case id, object, status, model, instructions, output, error, tools, truncation, text, temperature, reasoning, usage, store, background, metadata
        case createdAt = "created_at"
        case completedAt = "completed_at"
        case incompleteDetails = "incomplete_details"
        case previousResponseId = "previous_response_id"
        case toolChoice = "tool_choice"
        case parallelToolCalls = "parallel_tool_calls"
        case topP = "top_p"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case topLogprobs = "top_logprobs"
        case maxOutputTokens = "max_output_tokens"
        case maxToolCalls = "max_tool_calls"
        case serviceTier = "service_tier"
        case safetyIdentifier = "safety_identifier"
        case promptCacheKey = "prompt_cache_key"
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(object, forKey: .object)
        try c.encode(createdAt, forKey: .createdAt)
        try encodeNullable(completedAt, to: &c, forKey: .completedAt)
        try c.encode(status, forKey: .status)
        try encodeNullable(incompleteDetails, to: &c, forKey: .incompleteDetails)
        try c.encode(model, forKey: .model)
        try encodeNullable(previousResponseId, to: &c, forKey: .previousResponseId)
        try encodeNullable(instructions, to: &c, forKey: .instructions)
        try c.encode(output, forKey: .output)
        try encodeNullable(error, to: &c, forKey: .error)
        try c.encode(tools, forKey: .tools)
        try c.encode(toolChoice, forKey: .toolChoice)
        try c.encode(truncation, forKey: .truncation)
        try c.encode(parallelToolCalls, forKey: .parallelToolCalls)
        try c.encode(text, forKey: .text)
        try c.encode(topP, forKey: .topP)
        try c.encode(presencePenalty, forKey: .presencePenalty)
        try c.encode(frequencyPenalty, forKey: .frequencyPenalty)
        try c.encode(topLogprobs, forKey: .topLogprobs)
        try c.encode(temperature, forKey: .temperature)
        try encodeNullable(reasoning, to: &c, forKey: .reasoning)
        try encodeNullable(usage, to: &c, forKey: .usage)
        try encodeNullable(maxOutputTokens, to: &c, forKey: .maxOutputTokens)
        try encodeNullable(maxToolCalls, to: &c, forKey: .maxToolCalls)
        try c.encode(store, forKey: .store)
        try c.encode(background, forKey: .background)
        try c.encode(serviceTier, forKey: .serviceTier)
        try c.encode(metadata, forKey: .metadata)
        try encodeNullable(safetyIdentifier, to: &c, forKey: .safetyIdentifier)
        try encodeNullable(promptCacheKey, to: &c, forKey: .promptCacheKey)
    }
}

private func encodeNullable<T: Encodable, K: CodingKey>(
    _ value: T?,
    to container: inout KeyedEncodingContainer<K>,
    forKey key: K
) throws {
    if let value {
        try container.encode(value, forKey: key)
    } else {
        try container.encodeNil(forKey: key)
    }
}

public struct OpenAIResponseTemplate: Sendable {
    public let id: String
    public let createdAt: Int
    public let model: String
    public let req: OpenAIResponseCreateRequest
    public let tools: [OpenAIResponseResource.FunctionToolInfo]
    public let toolChoice: QWJson
    public let resolvedTemperature: Double
    public let resolvedTopP: Double
    public let textFormat: OpenAIResponseResource.TextConfig

    public init(
        id: String,
        createdAt: Int,
        model: String,
        req: OpenAIResponseCreateRequest,
        tools: [OpenAIResponseResource.FunctionToolInfo],
        toolChoice: QWJson,
        resolvedTemperature: Double,
        resolvedTopP: Double,
        textFormat: OpenAIResponseResource.TextConfig
    ) {
        self.id = id
        self.createdAt = createdAt
        self.model = model
        self.req = req
        self.tools = tools
        self.toolChoice = toolChoice
        self.resolvedTemperature = resolvedTemperature
        self.resolvedTopP = resolvedTopP
        self.textFormat = textFormat
    }

    public func makeResource(
        status: String,
        completedAt: Int?,
        output: [QWJson],
        usage: OpenAIResponseResource.UsageInfo?,
        incompleteDetails: OpenAIResponseResource.IncompleteDetails? = nil,
        error: OpenAIResponseResource.ErrorInfo? = nil
    ) -> OpenAIResponseResource {
        let reasoningInfo = req.reasoning.map {
            OpenAIResponseResource.ReasoningInfo(effort: $0.effort, summary: $0.summary)
        }
        return OpenAIResponseResource(
            id: id,
            object: "response",
            createdAt: createdAt,
            completedAt: completedAt,
            status: status,
            incompleteDetails: incompleteDetails,
            model: model,
            previousResponseId: req.previousResponseId,
            instructions: req.instructions,
            output: output,
            error: error,
            tools: tools,
            toolChoice: toolChoice,
            truncation: req.truncation ?? "disabled",
            parallelToolCalls: req.parallelToolCalls ?? true,
            text: textFormat,
            topP: resolvedTopP,
            presencePenalty: req.presencePenalty ?? 0.0,
            frequencyPenalty: req.frequencyPenalty ?? 0.0,
            topLogprobs: req.topLogprobs ?? 0,
            temperature: resolvedTemperature,
            reasoning: reasoningInfo,
            usage: usage,
            maxOutputTokens: req.maxOutputTokens,
            maxToolCalls: req.maxToolCalls,
            store: req.store ?? true,
            background: req.background ?? false,
            serviceTier: req.serviceTier ?? "default",
            metadata: req.metadata ?? [:],
            safetyIdentifier: req.safetyIdentifier,
            promptCacheKey: req.promptCacheKey
        )
    }
}

public struct OpenAIResponseValidationError: Error, CustomStringConvertible, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var description: String { message }
}

public enum OpenAIResponsesMapper {
    public static func parseInput(_ input: QWJson?, hasPrevious: Bool) throws -> [ChatMessage] {
        guard let input, !input.isUndefined, input != .null else {
            if hasPrevious { return [] }
            throw OpenAIResponseValidationError("input is required")
        }
        switch input {
        case .string(let text):
            return [ChatMessage(role: "user", text: text)]
        case .array(let items):
            return try parseItemArray(items)
        default:
            throw OpenAIResponseValidationError("input must be a string or an array of items")
        }
    }

    private static func parseItemArray(_ items: [QWJson]) throws -> [ChatMessage] {
        var out: [ChatMessage] = []
        var pendingReasoning: String? = nil

        for item in items {
            guard case .object = item else {
                throw OpenAIResponseValidationError("each item in input must be a JSON object")
            }
            let itemType: String
            if case .string(let t) = item.get("type") { itemType = t.lowercased() } else { itemType = "" }
            switch itemType {
            case "function_call":
                let callId: String
                if case .string(let cid) = item.get("call_id"), !cid.isEmpty { callId = cid }
                else if case .string(let id) = item.get("id"), !id.isEmpty { callId = id }
                else { callId = QWToolCallParser.generateCallId() }
                guard case .string(let name) = item.get("name"), !name.isEmpty else {
                    throw OpenAIResponseValidationError("function_call item is missing name")
                }
                let argsStr = jsonStringValue(item.get("arguments")) ?? "{}"
                let tc = ToolCall(id: callId, type: "function", function: ToolCallFunction(name: name, arguments: argsStr))
                if let last = out.last, last.role == "assistant" {
                    var merged = last.toolCalls ?? []
                    merged.append(tc)
                    out[out.count - 1] = ChatMessage(
                        role: "assistant", content: last.content, toolCalls: merged,
                        reasoningContent: last.reasoningContent ?? pendingReasoning
                    )
                } else {
                    out.append(ChatMessage(
                        role: "assistant", content: nil, toolCalls: [tc], reasoningContent: pendingReasoning
                    ))
                }
                pendingReasoning = nil
            case "function_call_output":
                guard case .string(let callId) = item.get("call_id"), !callId.isEmpty else {
                    throw OpenAIResponseValidationError("function_call_output item is missing call_id")
                }
                out.append(ChatMessage(
                    role: "tool",
                    content: ChatMessageContent(string: extractToolOutputText(item.get("output"))),
                    toolCallId: callId
                ))
            case "reasoning":
                let text = extractReasoningText(from: item)
                if !text.isEmpty {
                    pendingReasoning = pendingReasoning.map { $0 + "\n" + text } ?? text
                }
            default:
                guard case .string(let rawRole) = item.get("role") else {
                    throw OpenAIResponseValidationError("input item must specify role or a supported item type")
                }
                let role = rawRole.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                let content = try parseMessageContent(item.get("content"))
                var msgReasoning: String? = nil
                if case .string(let r) = item.get("reasoning_content") { msgReasoning = r }
                else if case .string(let r) = item.get("reasoning") { msgReasoning = r }
                if role == "assistant", let pending = pendingReasoning {
                    msgReasoning = msgReasoning.map { pending + "\n" + $0 } ?? pending
                    pendingReasoning = nil
                }
                out.append(ChatMessage(role: role, content: content, reasoningContent: msgReasoning))
            }
        }
        return out
    }

    private static func parseMessageContent(_ raw: QWJson) throws -> ChatMessageContent {
        switch raw {
        case .string(let s):
            return ChatMessageContent(string: s)
        case .array(let arr):
            var parts: [ChatMessageContent.Part] = []
            for p in arr {
                guard case .object = p else {
                    throw OpenAIResponseValidationError("message content part must be an object")
                }
                let pType: String
                if case .string(let s) = p.get("type") { pType = s.lowercased() } else { pType = "" }
                switch pType {
                case "input_text", "output_text", "text", "summary_text", "reasoning_text":
                    if case .string(let t) = p.get("text") { parts.append(.text(t)) }
                case "refusal":
                    if case .string(let t) = p.get("refusal") { parts.append(.text(t)) }
                    else if case .string(let t) = p.get("text") { parts.append(.text(t)) }
                case "input_image", "image_url":
                    if case .string(let u) = p.get("image_url"), !u.isEmpty {
                        parts.append(.imageUrl(u))
                    } else if case .string(let u) = p.get("image_url").get("url"), !u.isEmpty {
                        parts.append(.imageUrl(u))
                    }
                default:
                    if case .string(let t) = p.get("text") { parts.append(.text(t)) }
                }
            }
            return ChatMessageContent(parts: parts)
        case .null, .undefined:
            return ChatMessageContent(string: "")
        default:
            throw OpenAIResponseValidationError("message content must be a string or array")
        }
    }

    private static func extractToolOutputText(_ raw: QWJson) -> String {
        switch raw {
        case .string(let s): return s
        case .array(let arr):
            return arr.compactMap { part -> String? in
                if case .string(let t) = part.get("text") { return t }
                return nil
            }.joined(separator: "\n")
        case .undefined, .null: return ""
        default: return jsonStringValue(raw) ?? ""
        }
    }

    private static func extractReasoningText(from item: QWJson) -> String {
        var chunks: [String] = []
        for key in ["summary", "content"] {
            if case .array(let arr) = item.get(key) {
                for part in arr {
                    if case .string(let t) = part.get("text"), !t.isEmpty { chunks.append(t) }
                }
            }
        }
        return chunks.joined(separator: "")
    }

    private static func jsonStringValue(_ raw: QWJson) -> String? {
        switch raw {
        case .string(let s): return s
        case .undefined, .null: return nil
        default:
            let enc = JSONEncoder()
            enc.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            guard let data = try? enc.encode(raw), let str = String(data: data, encoding: .utf8) else { return nil }
            return str
        }
    }

    public static func normalizeTools(
        tools: [QWJson]?,
        toolChoice: QWJson?,
        maxToolCalls: Int?
    ) -> ([QWJson]?, [OpenAIResponseResource.FunctionToolInfo], QWJson?, QWJson) {
        var allowedNames: Set<String>? = nil
        var mappedChoice: QWJson? = toolChoice
        if let tc = toolChoice, case .string(let t) = tc.get("type"), t.lowercased() == "allowed_tools" {
            if case .array(let list) = tc.get("tools") {
                var set = Set<String>()
                for entry in list {
                    if case .string(let name) = entry.get("name"), !name.isEmpty { set.insert(name) }
                }
                allowedNames = set
            }
            if case .string(let mode) = tc.get("mode") { mappedChoice = .string(mode) }
        }
        if maxToolCalls == 0 { mappedChoice = .string("none") }

        var chatTools: [QWJson] = []
        var resTools: [OpenAIResponseResource.FunctionToolInfo] = []
        for rawTool in tools ?? [] {
            let fnObj = rawTool.get("function").isUndefined ? rawTool : rawTool.get("function")
            guard case .string(let name) = fnObj.get("name"), !name.isEmpty else { continue }
            if let allowedNames, !allowedNames.contains(name) { continue }
            let desc: String? = { if case .string(let d) = fnObj.get("description") { return d }; return nil }()
            let params: QWJson? = {
                let p = fnObj.get("parameters")
                return p.isUndefined || p == .null ? nil : p
            }()
            let strict: Bool? = {
                if case .bool(let b) = fnObj.get("strict") { return b }
                if case .bool(let b) = rawTool.get("strict") { return b }
                return nil
            }()
            resTools.append(OpenAIResponseResource.FunctionToolInfo(
                name: name, description: desc, parameters: params, strict: strict
            ))
            var fnPairs: [(String, QWJson)] = [("name", .string(name))]
            if let desc { fnPairs.append(("description", .string(desc))) }
            if let params { fnPairs.append(("parameters", params)) }
            chatTools.append(.object([("type", .string("function")), ("function", .object(fnPairs))]))
        }
        return (chatTools.isEmpty ? nil : chatTools, resTools, mappedChoice, toolChoice ?? .string("auto"))
    }

    public static func resourceTextConfig(from text: OpenAIResponseCreateRequest.TextOption?) -> OpenAIResponseResource.TextConfig {
        guard let fmt = text?.format, case .string(let rawType) = fmt.get("type") else {
            return OpenAIResponseResource.TextConfig(verbosity: text?.verbosity)
        }
        switch rawType.lowercased() {
        case "json_object":
            return OpenAIResponseResource.TextConfig(
                format: .object([("type", .string("json_object"))]), verbosity: text?.verbosity
            )
        case "json_schema":
            let name: String = { if case .string(let n) = fmt.get("name") { return n }; return "output" }()
            let desc: QWJson = { if case .string(let d) = fmt.get("description") { return .string(d) }; return .null }()
            let strict: Bool = { if case .bool(let s) = fmt.get("strict") { return s }; return true }()
            return OpenAIResponseResource.TextConfig(
                format: .object([
                    ("type", .string("json_schema")), ("name", .string(name)),
                    ("description", desc), ("schema", .null), ("strict", .bool(strict)),
                ]),
                verbosity: text?.verbosity
            )
        default:
            return OpenAIResponseResource.TextConfig(verbosity: text?.verbosity)
        }
    }

    public static func buildChatRequest(
        from req: OpenAIResponseCreateRequest,
        messages: [ChatMessage],
        tools: [QWJson]?,
        toolChoice: QWJson?,
        headers: [String: String]
    ) throws -> (request: ChatCompletionRequest, thinkingExplicitlyDisabled: Bool) {
        var responseFormat: QWJson? = nil
        if let fmt = req.text?.format, case .string(let fmtType) = fmt.get("type") {
            let lower = fmtType.lowercased()
            if lower == "json_object" {
                responseFormat = .object([("type", .string("json_object"))])
            } else if lower == "json_schema" {
                var jsPairs: [(String, QWJson)] = []
                if case .string(let name) = fmt.get("name") { jsPairs.append(("name", .string(name))) }
                if !fmt.get("schema").isUndefined { jsPairs.append(("schema", fmt.get("schema"))) }
                if case .bool(let strict) = fmt.get("strict") { jsPairs.append(("strict", .bool(strict))) }
                else { jsPairs.append(("strict", .bool(true))) }
                responseFormat = .object([("type", .string("json_schema")), ("json_schema", .object(jsPairs))])
            }
        }

        let effort = req.reasoning?.effort?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let isExplicitOff = effort.map { ["none", "off", "false", "disabled", "disable"].contains($0) } ?? false
        let rawEffort = req.reasoning?.effort
        let enableThinking: Bool? = isExplicitOff ? false : req.enableThinking

        var payload: [String: Any] = [
            "messages": try JSONSerialization.jsonObject(with: JSONEncoder().encode(messages), options: .fragmentsAllowed)
        ]
        if let model = req.model { payload["model"] = model }
        if let tools {
            payload["tools"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(tools), options: .fragmentsAllowed)
        }
        if let toolChoice {
            payload["tool_choice"] = try JSONSerialization.jsonObject(
                with: JSONEncoder().encode(toolChoice), options: .fragmentsAllowed
            )
        }
        if let p = req.parallelToolCalls { payload["parallel_tool_calls"] = p }
        if let rf = responseFormat {
            payload["response_format"] = try JSONSerialization.jsonObject(
                with: JSONEncoder().encode(rf), options: .fragmentsAllowed
            )
        }
        if let maxOut = req.maxOutputTokens { payload["max_tokens"] = maxOut }
        if let temp = req.temperature { payload["temperature"] = temp }
        if let topP = req.topP { payload["top_p"] = topP }
        if let rawEffort, !rawEffort.isEmpty { payload["reasoning_effort"] = rawEffort }
        if let eth = enableThinking { payload["enable_thinking"] = eth }
        if let budget = req.reasoningBudget { payload["reasoning_budget"] = budget }
        if let meta = req.metadata { payload["metadata"] = meta }
        if let user = req.user ?? req.safetyIdentifier { payload["user"] = user }
        if let pck = req.promptCacheKey { payload["prompt_cache_key"] = pck }
        if let cp = req.cachePrompt { payload["cache_prompt"] = cp }
        if let cs = req.cacheSalt { payload["cache_salt"] = cs }
        if let st = req.store { payload["store"] = st }

        var chatReq = try JSONDecoder().decode(ChatCompletionRequest.self, from: JSONSerialization.data(withJSONObject: payload))
        chatReq.httpHeaders = headers
        return (chatReq, isExplicitOff || (enableThinking == false))
    }

    public static func capToolCalls(_ calls: [ToolCall]?, parallelToolCalls: Bool?, maxToolCalls: Int?) -> [ToolCall]? {
        guard let calls, !calls.isEmpty else { return nil }
        var limit = calls.count
        if parallelToolCalls == false { limit = min(limit, 1) }
        if let maxToolCalls { limit = min(limit, max(0, maxToolCalls)) }
        let sliced = Array(calls.prefix(limit))
        return sliced.isEmpty ? nil : sliced
    }
}
