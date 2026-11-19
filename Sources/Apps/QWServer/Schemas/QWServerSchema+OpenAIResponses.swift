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
        metadata = try QWSchemaMetadataCoercion.decodeOptionalMetadata(from: c, forKey: .metadata)
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
