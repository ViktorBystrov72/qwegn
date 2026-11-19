// QWServerSchema+Responses.swift — Non-streaming and streaming chat completion DTOs.

import Foundation
import QWCore
import QWToken

public struct ChatCompletionResponse: Codable, Sendable {
    public struct Choice: Codable, Sendable {
        public struct Message: Codable, Sendable {
            public let role: String
            public let content: String?
            public let toolCalls: [ToolCall]?
            public let reasoningContent: String?

            enum CodingKeys: String, CodingKey {
                case role
                case content
                case toolCalls = "tool_calls"
                case reasoningContent = "reasoning_content"
            }

            public init(role: String, content: String?, toolCalls: [ToolCall]? = nil, reasoningContent: String? = nil) {
                self.role = role
                self.content = content
                self.toolCalls = toolCalls
                self.reasoningContent = reasoningContent
            }

            public func encode(to encoder: Encoder) throws {
                var container = encoder.container(keyedBy: CodingKeys.self)
                try container.encode(role, forKey: .role)
                if let content {
                    try container.encode(content, forKey: .content)
                } else {
                    try container.encodeNil(forKey: .content)
                }
                try container.encodeIfPresent(toolCalls, forKey: .toolCalls)
                try container.encodeIfPresent(reasoningContent, forKey: .reasoningContent)
            }
        }
        public let index: Int
        public let message: Message
        public let logprobs: ChoiceLogprobs?
        public let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index
            case message
            case logprobs
            case finishReason = "finish_reason"
        }
    }

    public struct Usage: Codable, Sendable {
        public struct PromptTokensDetails: Codable, Sendable {
            public let cachedTokens: Int

            enum CodingKeys: String, CodingKey {
                case cachedTokens = "cached_tokens"
            }

            public init(cachedTokens: Int) {
                self.cachedTokens = cachedTokens
            }
        }

        public struct CompletionTokensDetails: Codable, Sendable {
            public let reasoningTokens: Int

            enum CodingKeys: String, CodingKey {
                case reasoningTokens = "reasoning_tokens"
            }

            public init(reasoningTokens: Int) {
                self.reasoningTokens = reasoningTokens
            }
        }

        public let promptTokens: Int
        public let completionTokens: Int
        public let totalTokens: Int
        public let promptTokensDetails: PromptTokensDetails?
        public let completionTokensDetails: CompletionTokensDetails?

        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case completionTokens = "completion_tokens"
            case totalTokens = "total_tokens"
            case promptTokensDetails = "prompt_tokens_details"
            case completionTokensDetails = "completion_tokens_details"
        }

        public init(promptTokens: Int, completionTokens: Int, totalTokens: Int,
                    cachedTokens: Int? = nil, reasoningTokens: Int? = nil) {
            self.promptTokens = promptTokens
            self.completionTokens = completionTokens
            self.totalTokens = totalTokens
            self.promptTokensDetails = cachedTokens.map { PromptTokensDetails(cachedTokens: $0) }
            self.completionTokensDetails = reasoningTokens.map { CompletionTokensDetails(reasoningTokens: $0) }
        }
    }

    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [Choice]
    public let usage: Usage

    public init(
        id: String,
        model: String,
        content: String?,
        toolCalls: [ToolCall]? = nil,
        reasoningContent: String? = nil,
        logprobs: ChoiceLogprobs? = nil,
        promptTokens: Int,
        completionTokens: Int,
        cachedTokens: Int? = nil,
        reasoningTokens: Int? = nil,
        finishReason: String = "stop"
    ) {
        self.id = id
        self.object = "chat.completion"
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        self.choices = [
            Choice(
                index: 0,
                message: Choice.Message(
                    role: "assistant",
                    content: content,
                    toolCalls: toolCalls,
                    reasoningContent: reasoningContent
                ),
                logprobs: logprobs,
                finishReason: finishReason
            )
        ]
        self.usage = Usage(
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            totalTokens: promptTokens + completionTokens,
            cachedTokens: cachedTokens,
            reasoningTokens: reasoningTokens
        )
    }
}

public struct ChatCompletionChunk: Codable, Sendable {
    public struct Choice: Codable, Sendable {
        public struct Delta: Codable, Sendable {
            public let role: String?
            public let content: String?
            public let toolCalls: [ToolCallChunk]?
            public let reasoningContent: String?

            enum CodingKeys: String, CodingKey {
                case role
                case content
                case toolCalls = "tool_calls"
                case reasoningContent = "reasoning_content"
            }

            public init(
                role: String? = nil,
                content: String? = nil,
                toolCalls: [ToolCallChunk]? = nil,
                reasoningContent: String? = nil
            ) {
                self.role = role
                self.content = content
                self.toolCalls = toolCalls
                self.reasoningContent = reasoningContent
            }
        }
        public let index: Int
        public let delta: Delta
        public let logprobs: ChoiceLogprobs?
        public let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index
            case delta
            case logprobs
            case finishReason = "finish_reason"
        }
    }

    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [Choice]
    public let usage: ChatCompletionResponse.Usage?

    public init(
        id: String,
        model: String,
        role: String? = nil,
        content: String? = nil,
        toolCall: ToolCallChunk? = nil,
        reasoningContent: String? = nil,
        logprobs: ChoiceLogprobs? = nil,
        finishReason: String? = nil,
        usage: ChatCompletionResponse.Usage? = nil
    ) {
        self.id = id
        self.object = "chat.completion.chunk"
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        let toolCalls = toolCall.map { [$0] }
        self.choices = [
            Choice(
                index: 0,
                delta: Choice.Delta(
                    role: role,
                    content: content,
                    toolCalls: toolCalls,
                    reasoningContent: reasoningContent
                ),
                logprobs: logprobs,
                finishReason: finishReason
            )
        ]
        self.usage = usage
    }
}

public struct ChatCompletionExecutionResult: Sendable {
    public let content: String
    public let promptTokens: Int
    public let completionTokens: Int
    public let reasoningTokens: Int?
    public let cachedTokens: Int
    public let cacheActive: Bool
    public let truncated: Bool
    public let startsInThinking: Bool
    public let matchedStopSequence: String?
    public let logprobs: [ChatLogprobTokenEntry]?
    public let prefillSec: Double?
    public let decodeSec: Double?

    public init(
        content: String,
        promptTokens: Int,
        completionTokens: Int,
        reasoningTokens: Int?,
        cachedTokens: Int,
        cacheActive: Bool = false,
        truncated: Bool,
        startsInThinking: Bool,
        matchedStopSequence: String?,
        logprobs: [ChatLogprobTokenEntry]?,
        prefillSec: Double?,
        decodeSec: Double?
    ) {
        self.content = content
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.reasoningTokens = reasoningTokens
        self.cachedTokens = cachedTokens
        self.cacheActive = cacheActive
        self.truncated = truncated
        self.startsInThinking = startsInThinking
        self.matchedStopSequence = matchedStopSequence
        self.logprobs = logprobs
        self.prefillSec = prefillSec
        self.decodeSec = decodeSec
    }
}
