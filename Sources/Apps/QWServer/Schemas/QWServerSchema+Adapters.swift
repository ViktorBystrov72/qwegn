// QWServerSchema+Adapters.swift — Typed Anthropic Messages and Ollama request DTOs
// that map directly into ChatCompletionRequest in Swift without JSON round-trips.

import Foundation
import QWCore
import QWToken

public struct AnthropicValidationError: Error, Sendable {
    public let status: Int
    public let type: String
    public let message: String

    public init(status: Int = 400, type: String = "invalid_request_error", message: String) {
        self.status = status
        self.type = type
        self.message = message
    }
}

public struct AnthropicMessagesRequest: Decodable, Sendable {
    public struct MessageInput: Decodable, Sendable {
        public let role: String?
        public let content: QWJson?
    }

    public struct ThinkingInput: Decodable, Sendable {
        public let type: String?
        public let budgetTokens: QWJson?

        enum CodingKeys: String, CodingKey {
            case type
            case budgetTokens = "budget_tokens"
        }
    }

    public struct ToolInput: Decodable, Sendable {
        public let name: String?
        public let description: String?
        public let inputSchema: QWJson?

        enum CodingKeys: String, CodingKey {
            case name, description
            case inputSchema = "input_schema"
        }
    }

    public struct ToolChoiceInput: Decodable, Sendable {
        public let type: String?
        public let name: String?
        public let disableParallelToolUse: Bool?

        enum CodingKeys: String, CodingKey {
            case type, name
            case disableParallelToolUse = "disable_parallel_tool_use"
        }
    }

    public let model: String?
    public let messages: [MessageInput]?
    public let maxTokens: QWJson?
    public let system: QWJson?
    public let stream: Bool?
    public let temperature: QWJson?
    public let topP: QWJson?
    public let topK: QWJson?
    public let stopSequences: [String]?
    public let thinking: ThinkingInput?
    public let metadata: [String: String]?
    public let user: String?
    public let sessionId: String?
    public let promptCacheKey: String?
    public let cacheSalt: String?
    public let cachePrompt: Bool?
    public let store: Bool?
    public let tools: [ToolInput]?
    public let toolChoice: ToolChoiceInput?

    enum CodingKeys: String, CodingKey {
        case model, messages, system, stream, temperature, thinking, metadata, user, store, tools
        case maxTokens = "max_tokens"
        case topP = "top_p"
        case topK = "top_k"
        case stopSequences = "stop_sequences"
        case sessionId = "session_id"
        case promptCacheKey = "prompt_cache_key"
        case cacheSalt = "cache_salt"
        case cachePrompt = "cache_prompt"
        case toolChoice = "tool_choice"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        model = try c.decodeIfPresent(String.self, forKey: .model)
        messages = try c.decodeIfPresent([MessageInput].self, forKey: .messages)
        maxTokens = try c.decodeIfPresent(QWJson.self, forKey: .maxTokens)
        system = try c.decodeIfPresent(QWJson.self, forKey: .system)
        stream = try c.decodeIfPresent(Bool.self, forKey: .stream)
        temperature = try c.decodeIfPresent(QWJson.self, forKey: .temperature)
        topP = try c.decodeIfPresent(QWJson.self, forKey: .topP)
        topK = try c.decodeIfPresent(QWJson.self, forKey: .topK)
        stopSequences = try c.decodeIfPresent([String].self, forKey: .stopSequences)
        thinking = try c.decodeIfPresent(ThinkingInput.self, forKey: .thinking)
        metadata = try QWSchemaMetadataCoercion.decodeOptionalMetadata(from: c, forKey: .metadata)
        user = try c.decodeIfPresent(String.self, forKey: .user)
        sessionId = try c.decodeIfPresent(String.self, forKey: .sessionId)
        promptCacheKey = try c.decodeIfPresent(String.self, forKey: .promptCacheKey)
        cacheSalt = try c.decodeIfPresent(String.self, forKey: .cacheSalt)
        cachePrompt = try c.decodeIfPresent(Bool.self, forKey: .cachePrompt)
        store = try c.decodeIfPresent(Bool.self, forKey: .store)
        tools = try c.decodeIfPresent([ToolInput].self, forKey: .tools)
        toolChoice = try c.decodeIfPresent(ToolChoiceInput.self, forKey: .toolChoice)
    }

    public func toChatCompletionRequest(
        replyModel: String,
        requireMaxTokens: Bool,
        headers: [String: String]
    ) throws -> (request: ChatCompletionRequest, thinkingEnabled: Bool) {
        guard let messages, !messages.isEmpty else {
            throw AnthropicValidationError(message: "messages: expected required field present")
        }
        let resolvedMaxTokens: Int
        if requireMaxTokens {
            guard let mt = QWSchemaMetadataCoercion.strictInt(maxTokens), mt > 0 else {
                throw AnthropicValidationError(message: "max_tokens: expect integer greater than 0")
            }
            resolvedMaxTokens = mt
        } else {
            if let rawMT = maxTokens, rawMT != .null, rawMT != .undefined,
               QWSchemaMetadataCoercion.strictInt(rawMT) == nil {
                throw AnthropicValidationError(message: "max_tokens: expect integer")
            }
            resolvedMaxTokens = QWSchemaMetadataCoercion.strictInt(maxTokens) ?? 1
        }

        var translated: [ChatMessage] = []
        if let system {
            switch system {
            case .string(let s) where !s.isEmpty:
                translated.append(ChatMessage(role: "system", text: s))
            case .array(let blocks):
                let joined = blocks.compactMap { b -> String? in
                    guard case .string(let t) = b.get("type"), t == "text",
                          case .string(let txt) = b.get("text") else { return nil }
                    return txt
                }.joined(separator: "\n")
                if !joined.isEmpty {
                    translated.append(ChatMessage(role: "system", text: joined))
                }
            default:
                break
            }
        }
        for msg in messages {
            translated.append(contentsOf: Self.translateMessage(msg))
        }

        let thinkingEnabled = thinking?.type?.lowercased() == "enabled"
        let budget = thinkingEnabled ? QWSchemaMetadataCoercion.strictInt(thinking?.budgetTokens) : nil
        let effectiveBudget = (budget ?? 0) > 0 ? budget : nil
        let requested = (model ?? "").trimmingCharacters(in: .whitespacesAndNewlines)

        let mappedTools: [QWJson]? = tools.flatMap { list in
            let converted: [QWJson] = list.compactMap { t in
                guard let name = t.name, !name.isEmpty else { return nil }
                var fnPairs: [(String, QWJson)] = [("name", .string(name))]
                if let desc = t.description { fnPairs.append(("description", .string(desc))) }
                if let schema = t.inputSchema, !schema.isUndefined, schema != .null {
                    fnPairs.append(("parameters", schema))
                }
                return .object([("type", .string("function")), ("function", .object(fnPairs))])
            }
            return converted.isEmpty ? nil : converted
        }

        var mappedToolChoice: QWJson? = nil
        var parallelCalls: Bool? = nil
        if let tc = toolChoice {
            let tcType = (tc.type ?? "auto").lowercased()
            switch tcType {
            case "none": mappedToolChoice = .string("none")
            case "any", "required": mappedToolChoice = .string("required")
            case "tool":
                if let name = tc.name, !name.isEmpty {
                    mappedToolChoice = .object([
                        ("type", .string("function")),
                        ("function", .object([("name", .string(name))])),
                    ])
                }
            default:
                mappedToolChoice = .string("auto")
            }
            if tc.disableParallelToolUse == true {
                parallelCalls = false
            }
        }

        let resolvedUser = (user?.isEmpty == false ? user : nil)
            ?? (metadata?["user_id"].flatMap { $0.isEmpty ? nil : $0 })
        let req = ChatCompletionRequest(
            model: requested.isEmpty ? nil : replyModel,
            messages: translated,
            tools: mappedTools,
            toolChoice: mappedToolChoice,
            parallelToolCalls: parallelCalls,
            stream: requireMaxTokens ? (stream == true) : nil,
            maxTokens: resolvedMaxTokens,
            temperature: QWSchemaMetadataCoercion.strictDouble(temperature),
            topP: QWSchemaMetadataCoercion.strictDouble(topP),
            topK: QWSchemaMetadataCoercion.strictInt(topK),
            stop: (stopSequences?.isEmpty == false) ? stopSequences : nil,
            enableThinking: thinkingEnabled,
            reasoningBudget: effectiveBudget,
            metadata: metadata,
            user: resolvedUser,
            sessionId: sessionId,
            promptCacheKey: promptCacheKey,
            cacheSalt: cacheSalt,
            cachePrompt: cachePrompt,
            store: store,
            httpHeaders: headers
        )
        return (req, thinkingEnabled)
    }

    private static func translateMessage(_ msg: MessageInput) -> [ChatMessage] {
        let role = msg.role ?? "user"
        guard let content = msg.content else {
            return [ChatMessage(role: role, text: "")]
        }
        if case .string(let s) = content {
            return [ChatMessage(role: role, text: s)]
        }
        guard case .array(let blocks) = content else {
            return [ChatMessage(role: role, text: "")]
        }

        var textParts: [String] = []
        var imageUrls: [String] = []
        var videoSpecs: [QWVideoInputSpec] = []
        var reasoningParts: [String] = []
        var toolCalls: [ToolCall] = []
        var toolResults: [ChatMessage] = []

        for block in blocks {
            let bType: String
            if case .string(let s) = block.get("type") { bType = s } else { bType = "" }
            switch bType {
            case "text":
                if case .string(let t) = block.get("text") { textParts.append(t) }
            case "thinking":
                if case .string(let th) = block.get("thinking") { reasoningParts.append(th) }
            case "image":
                let src = block.get("source")
                if case .string(let data) = src.get("data"), !data.isEmpty {
                    let mediaType: String
                    if case .string(let mt) = src.get("media_type") { mediaType = mt } else { mediaType = "image/png" }
                    imageUrls.append("data:\(mediaType);base64,\(data)")
                } else if case .string(let url) = src.get("url"), !url.isEmpty {
                    imageUrls.append(url)
                }
            case "video":
                let src = block.get("source")
                let fps = QWSchemaMetadataCoercion.strictDouble(block.get("fps"))
                let maxFrames = QWSchemaMetadataCoercion.strictInt(block.get("max_frames"))
                if case .string(let data) = src.get("data"), !data.isEmpty {
                    let mediaType: String
                    if case .string(let mt) = src.get("media_type") { mediaType = mt } else { mediaType = "video/mp4" }
                    videoSpecs.append(QWVideoInputSpec(
                        url: "data:\(mediaType);base64,\(data)", fps: fps, maxFrames: maxFrames
                    ))
                } else if case .string(let url) = src.get("url"), !url.isEmpty {
                    videoSpecs.append(QWVideoInputSpec(url: url, fps: fps, maxFrames: maxFrames))
                }
            case "tool_use":
                let id: String
                if case .string(let bid) = block.get("id") { id = bid } else { id = QWToolCallParser.generateCallId() }
                let name: String
                if case .string(let n) = block.get("name") { name = n } else { name = "" }
                let argsJson = QWSchemaMetadataCoercion.canonicalJSONString(block.get("input")) ?? "{}"
                toolCalls.append(ToolCall(id: id, type: "function", function: ToolCallFunction(name: name, arguments: argsJson)))
            case "tool_result":
                let callId: String
                if case .string(let cid) = block.get("tool_use_id") { callId = cid } else { callId = "" }
                let resultText = extractBlockText(block.get("content"))
                toolResults.append(ChatMessage(
                    role: "tool",
                    content: ChatMessageContent(string: resultText),
                    toolCallId: callId
                ))
            default:
                break
            }
        }

        var out: [ChatMessage] = toolResults
        let joinedText = textParts.joined(separator: "")
        if !imageUrls.isEmpty || !videoSpecs.isEmpty {
            var parts: [ChatMessageContent.Part] = []
            if !joinedText.isEmpty { parts.append(.text(joinedText)) }
            for url in imageUrls { parts.append(.imageUrl(url)) }
            for vid in videoSpecs { parts.append(.videoUrl(vid)) }
            out.append(ChatMessage(role: role, content: ChatMessageContent(parts: parts)))
        } else if !toolCalls.isEmpty || !joinedText.isEmpty || !reasoningParts.isEmpty || out.isEmpty {
            let msgContent: ChatMessageContent? = (!joinedText.isEmpty || toolCalls.isEmpty)
                ? ChatMessageContent(string: joinedText)
                : nil
            out.append(ChatMessage(
                role: role,
                content: msgContent,
                toolCalls: toolCalls.isEmpty ? nil : toolCalls,
                reasoningContent: reasoningParts.isEmpty ? nil : reasoningParts.joined(separator: "\n")
            ))
        }
        return out
    }

    private static func extractBlockText(_ content: QWJson) -> String {
        if case .string(let s) = content { return s }
        if case .array(let blocks) = content {
            return blocks.compactMap { b -> String? in
                guard case .string(let t) = b.get("type"), t == "text",
                      case .string(let txt) = b.get("text") else { return nil }
                return txt
            }.joined(separator: "")
        }
        return ""
    }
}

public struct OllamaOptionsInput: Decodable, Sendable {
    public let temperature: QWJson?
    public let topP: QWJson?
    public let topK: QWJson?
    public let seed: QWJson?
    public let repeatPenalty: QWJson?
    public let numPredict: QWJson?
    public let stop: QWJson?

    enum CodingKeys: String, CodingKey {
        case temperature, seed, stop
        case topP = "top_p"
        case topK = "top_k"
        case repeatPenalty = "repeat_penalty"
        case numPredict = "num_predict"
    }

    var resolvedTemperature: Double? { QWSchemaMetadataCoercion.strictDouble(temperature) }
    var resolvedTopP: Double? { QWSchemaMetadataCoercion.strictDouble(topP) }
    var resolvedTopK: Int? { QWSchemaMetadataCoercion.strictInt(topK) }
    var resolvedSeed: Int? { QWSchemaMetadataCoercion.strictInt(seed) }
    var resolvedRepeatPenalty: Double? { QWSchemaMetadataCoercion.strictDouble(repeatPenalty) }
    var resolvedMaxTokens: Int? {
        guard let n = QWSchemaMetadataCoercion.strictInt(numPredict), n > 0 else { return nil }
        return n
    }
    var resolvedStop: [String]? {
        OllamaAdapterMapper.parseStop(stop)
    }
}

enum OllamaAdapterMapper {
    static func parseStop(_ raw: QWJson?) -> [String]? {
        guard let raw else { return nil }
        switch raw {
        case .string(let s): return [s]
        case .array(let arr):
            return arr.compactMap { if case .string(let s) = $0 { return s }; return nil }
        default:
            return nil
        }
    }

    static func responseFormat(from format: QWJson?) -> QWJson? {
        guard let format else { return nil }
        if case .string(let s) = format, s.lowercased() == "json" {
            return .object([("type", .string("json_object"))])
        }
        if case .object = format {
            return .object([
                ("type", .string("json_schema")),
                ("json_schema", .object([
                    ("name", .string("ollama_schema")),
                    ("schema", format),
                    ("strict", .bool(true)),
                ])),
            ])
        }
        return nil
    }

    static func messageContent(text: String, images: [String]?) -> ChatMessageContent {
        guard let imgs = images?.filter({ !$0.isEmpty }), !imgs.isEmpty else {
            return ChatMessageContent(string: text)
        }
        var parts: [ChatMessageContent.Part] = [.text(text)]
        for img in imgs {
            let url = img.hasPrefix("data:") || img.hasPrefix("http") ? img : "data:image/jpeg;base64,\(img)"
            parts.append(.imageUrl(url))
        }
        return ChatMessageContent(parts: parts)
    }
}

public struct OllamaChatRequest: Decodable, Sendable {
    public struct MessageInput: Decodable, Sendable {
        public let role: String
        public let content: ChatMessageContent?
        public let images: [String]?
        public let toolCalls: [ToolCall]?
        public let toolCallId: String?
        public let name: String?
        public let reasoningContent: String?

        enum CodingKeys: String, CodingKey {
            case role, content, images, name
            case toolCalls = "tool_calls"
            case toolCallId = "tool_call_id"
            case reasoningContent = "reasoning_content"
        }
    }

    public let model: String?
    public let messages: [MessageInput]
    public let tools: [QWJson]?
    public let toolChoice: QWJson?
    public let parallelToolCalls: Bool?
    public let stream: Bool?
    public let options: OllamaOptionsInput?
    public let format: QWJson?
    public let stop: [String]?
    public let maxTokens: Int?
    public let temperature: Double?
    public let topP: Double?
    public let topK: Int?
    public let repetitionPenalty: Double?
    public let seed: Int?
    public let enableThinking: Bool?
    public let reasoningEffort: String?
    public let reasoningBudget: Int?
    public let metadata: [String: String]?
    public let user: String?
    public let sessionId: String?
    public let promptCacheKey: String?
    public let cacheSalt: String?
    public let cachePrompt: Bool?
    public let store: Bool?

    enum CodingKeys: String, CodingKey {
        case model, messages, tools, stream, options, format, stop, temperature, seed, metadata, user, store
        case toolChoice = "tool_choice"
        case parallelToolCalls = "parallel_tool_calls"
        case maxTokens = "max_tokens"
        case topP = "top_p"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
        case enableThinking = "enable_thinking"
        case reasoningEffort = "reasoning_effort"
        case reasoningBudget = "reasoning_budget"
        case sessionId = "session_id"
        case promptCacheKey = "prompt_cache_key"
        case cacheSalt = "cache_salt"
        case cachePrompt = "cache_prompt"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        model = try c.decodeIfPresent(String.self, forKey: .model)
        messages = try c.decode([MessageInput].self, forKey: .messages)
        tools = try c.decodeIfPresent([QWJson].self, forKey: .tools)
        toolChoice = try c.decodeIfPresent(QWJson.self, forKey: .toolChoice)
        parallelToolCalls = try c.decodeIfPresent(Bool.self, forKey: .parallelToolCalls)
        stream = try c.decodeIfPresent(Bool.self, forKey: .stream)
        options = try c.decodeIfPresent(OllamaOptionsInput.self, forKey: .options)
        format = try c.decodeIfPresent(QWJson.self, forKey: .format)
        stop = try QWSchemaMetadataCoercion.decodeOptionalStop(from: c, forKey: .stop)
        maxTokens = try c.decodeIfPresent(Int.self, forKey: .maxTokens)
        temperature = try c.decodeIfPresent(Double.self, forKey: .temperature)
        topP = try c.decodeIfPresent(Double.self, forKey: .topP)
        topK = try c.decodeIfPresent(Int.self, forKey: .topK)
        repetitionPenalty = try c.decodeIfPresent(Double.self, forKey: .repetitionPenalty)
        seed = try c.decodeIfPresent(Int.self, forKey: .seed)
        enableThinking = try c.decodeIfPresent(Bool.self, forKey: .enableThinking)
        reasoningEffort = try c.decodeIfPresent(String.self, forKey: .reasoningEffort)
        reasoningBudget = try c.decodeIfPresent(Int.self, forKey: .reasoningBudget)
        metadata = try QWSchemaMetadataCoercion.decodeOptionalMetadata(from: c, forKey: .metadata)
        user = try c.decodeIfPresent(String.self, forKey: .user)
        sessionId = try c.decodeIfPresent(String.self, forKey: .sessionId)
        promptCacheKey = try c.decodeIfPresent(String.self, forKey: .promptCacheKey)
        cacheSalt = try c.decodeIfPresent(String.self, forKey: .cacheSalt)
        cachePrompt = try c.decodeIfPresent(Bool.self, forKey: .cachePrompt)
        store = try c.decodeIfPresent(Bool.self, forKey: .store)
    }

    public func toChatCompletionRequest() -> ChatCompletionRequest {
        let convertedMessages = messages.map { msg -> ChatMessage in
            let content: ChatMessageContent?
            if let imgs = msg.images, !imgs.isEmpty {
                content = OllamaAdapterMapper.messageContent(text: msg.content?.textValue ?? "", images: imgs)
            } else {
                content = msg.content
            }
            return ChatMessage(
                role: msg.role,
                content: content,
                toolCalls: msg.toolCalls,
                toolCallId: msg.toolCallId,
                name: msg.name,
                reasoningContent: msg.reasoningContent
            )
        }
        return ChatCompletionRequest(
            model: model,
            messages: convertedMessages,
            tools: tools,
            toolChoice: toolChoice,
            parallelToolCalls: parallelToolCalls,
            responseFormat: OllamaAdapterMapper.responseFormat(from: format),
            stream: stream,
            maxTokens: options?.resolvedMaxTokens ?? maxTokens,
            temperature: options?.resolvedTemperature ?? temperature,
            topP: options?.resolvedTopP ?? topP,
            topK: options?.resolvedTopK ?? topK,
            repetitionPenalty: options?.resolvedRepeatPenalty ?? repetitionPenalty,
            seed: options?.resolvedSeed ?? seed,
            stop: stop ?? options?.resolvedStop,
            enableThinking: enableThinking,
            reasoningEffort: reasoningEffort,
            reasoningBudget: reasoningBudget,
            metadata: metadata,
            user: user,
            sessionId: sessionId,
            promptCacheKey: promptCacheKey,
            cacheSalt: cacheSalt,
            cachePrompt: cachePrompt,
            store: store
        )
    }
}

public struct OllamaGenerateRequest: Decodable, Sendable {
    public let model: String?
    public let prompt: String?
    public let system: String?
    public let images: [String]?
    public let stream: Bool?
    public let options: OllamaOptionsInput?
    public let format: QWJson?
    public let stop: [String]?
    public let metadata: [String: String]?
    public let user: String?
    public let sessionId: String?
    public let promptCacheKey: String?
    public let cacheSalt: String?
    public let cachePrompt: Bool?
    public let store: Bool?

    enum CodingKeys: String, CodingKey {
        case model, prompt, system, images, stream, options, format, stop, metadata, user, store
        case sessionId = "session_id"
        case promptCacheKey = "prompt_cache_key"
        case cacheSalt = "cache_salt"
        case cachePrompt = "cache_prompt"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        model = try c.decodeIfPresent(String.self, forKey: .model)
        prompt = try c.decodeIfPresent(String.self, forKey: .prompt)
        system = try c.decodeIfPresent(String.self, forKey: .system)
        images = try c.decodeIfPresent([String].self, forKey: .images)
        stream = try c.decodeIfPresent(Bool.self, forKey: .stream)
        options = try c.decodeIfPresent(OllamaOptionsInput.self, forKey: .options)
        format = try c.decodeIfPresent(QWJson.self, forKey: .format)
        stop = try QWSchemaMetadataCoercion.decodeOptionalStop(from: c, forKey: .stop)
        metadata = try QWSchemaMetadataCoercion.decodeOptionalMetadata(from: c, forKey: .metadata)
        user = try c.decodeIfPresent(String.self, forKey: .user)
        sessionId = try c.decodeIfPresent(String.self, forKey: .sessionId)
        promptCacheKey = try c.decodeIfPresent(String.self, forKey: .promptCacheKey)
        cacheSalt = try c.decodeIfPresent(String.self, forKey: .cacheSalt)
        cachePrompt = try c.decodeIfPresent(Bool.self, forKey: .cachePrompt)
        store = try c.decodeIfPresent(Bool.self, forKey: .store)
    }

    public func toChatCompletionRequest(promptText: String) -> ChatCompletionRequest {
        var msgs: [ChatMessage] = []
        if let sys = system, !sys.isEmpty {
            msgs.append(ChatMessage(role: "system", text: sys))
        }
        msgs.append(ChatMessage(
            role: "user",
            content: OllamaAdapterMapper.messageContent(text: promptText, images: images)
        ))
        let nonEmptyStop = stop.flatMap { $0.isEmpty ? nil : $0 } ?? options?.resolvedStop
        return ChatCompletionRequest(
            model: model,
            messages: msgs,
            responseFormat: OllamaAdapterMapper.responseFormat(from: format),
            stream: stream ?? true,
            maxTokens: options?.resolvedMaxTokens,
            temperature: options?.resolvedTemperature,
            topP: options?.resolvedTopP,
            topK: options?.resolvedTopK,
            repetitionPenalty: options?.resolvedRepeatPenalty,
            seed: options?.resolvedSeed,
            stop: nonEmptyStop,
            metadata: metadata,
            user: user,
            sessionId: sessionId,
            promptCacheKey: promptCacheKey,
            cacheSalt: cacheSalt,
            cachePrompt: cachePrompt,
            store: store
        )
    }
}
