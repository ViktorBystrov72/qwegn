// QWServerSchema+Completions.swift — Legacy OpenAI /v1/completions request/response DTOs.

import Foundation
import QWCore

public enum CompletionPromptInput: Sendable, Equatable {
    case strings([String])
    case tokenArrays([[Int]])
}

public struct CompletionRequest: Decodable, Sendable {
    public struct StreamOptions: Codable, Sendable {
        public let includeUsage: Bool?
        enum CodingKeys: String, CodingKey { case includeUsage = "include_usage" }
    }

    public let model: String?
    public let prompt: CompletionPromptInput
    public let maxTokens: Int?
    public let minTokens: Int?
    public let temperature: Double?
    public let topP: Double?
    public let topK: Int?
    public let repetitionPenalty: Double?
    public let presencePenalty: Double?
    public let frequencyPenalty: Double?
    public let seed: Int?
    public let stop: [String]?
    public let stream: Bool?
    public let streamOptions: StreamOptions?
    public let logprobs: Int?
    public let echo: Bool?
    public let n: Int?
    public let suffix: String?
    public let user: String?
    public let ignoreEos: Bool?
    public let cachePrompt: Bool?
    public let cacheSalt: String?
    public let promptCacheKey: String?
    public let metadata: [String: String]?
    public var httpHeaders: [String: String]? = nil

    enum CodingKeys: String, CodingKey {
        case model, prompt, temperature, seed, stop, stream, logprobs, echo, n, suffix, user, metadata
        case maxTokens = "max_tokens"
        case minTokens = "min_tokens"
        case topP = "top_p"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case streamOptions = "stream_options"
        case ignoreEos = "ignore_eos"
        case cachePrompt = "cache_prompt"
        case cacheSalt = "cache_salt"
        case promptCacheKey = "prompt_cache_key"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        model = try c.decodeIfPresent(String.self, forKey: .model)
        maxTokens = try c.decodeIfPresent(Int.self, forKey: .maxTokens)
        minTokens = try c.decodeIfPresent(Int.self, forKey: .minTokens)
        temperature = try c.decodeIfPresent(Double.self, forKey: .temperature)
        topP = try c.decodeIfPresent(Double.self, forKey: .topP)
        topK = try c.decodeIfPresent(Int.self, forKey: .topK)
        repetitionPenalty = try c.decodeIfPresent(Double.self, forKey: .repetitionPenalty)
        presencePenalty = try c.decodeIfPresent(Double.self, forKey: .presencePenalty)
        frequencyPenalty = try c.decodeIfPresent(Double.self, forKey: .frequencyPenalty)
        seed = try c.decodeIfPresent(Int.self, forKey: .seed)
        stream = try c.decodeIfPresent(Bool.self, forKey: .stream)
        streamOptions = try c.decodeIfPresent(StreamOptions.self, forKey: .streamOptions)
        echo = try c.decodeIfPresent(Bool.self, forKey: .echo)
        n = try c.decodeIfPresent(Int.self, forKey: .n)
        suffix = try c.decodeIfPresent(String.self, forKey: .suffix)
        user = try c.decodeIfPresent(String.self, forKey: .user)
        ignoreEos = try c.decodeIfPresent(Bool.self, forKey: .ignoreEos)
        cachePrompt = try c.decodeIfPresent(Bool.self, forKey: .cachePrompt)
        cacheSalt = try c.decodeIfPresent(String.self, forKey: .cacheSalt)
        promptCacheKey = try c.decodeIfPresent(String.self, forKey: .promptCacheKey)
        metadata = try c.decodeIfPresent([String: String].self, forKey: .metadata)

        if let s = try? c.decode(String.self, forKey: .prompt) {
            prompt = .strings([s])
        } else if let arr = try? c.decode([String].self, forKey: .prompt) {
            guard !arr.isEmpty else {
                throw DecodingError.dataCorruptedError(forKey: .prompt, in: c, debugDescription: "prompt must not be empty")
            }
            prompt = .strings(arr)
        } else if let ids = try? c.decode([Int].self, forKey: .prompt) {
            guard !ids.isEmpty else {
                throw DecodingError.dataCorruptedError(forKey: .prompt, in: c, debugDescription: "prompt must not be empty")
            }
            prompt = .tokenArrays([ids])
        } else if let nestedIds = try? c.decode([[Int]].self, forKey: .prompt) {
            guard !nestedIds.isEmpty, nestedIds.allSatisfy({ !$0.isEmpty }) else {
                throw DecodingError.dataCorruptedError(forKey: .prompt, in: c, debugDescription: "prompt must not be empty")
            }
            prompt = .tokenArrays(nestedIds)
        } else {
            throw DecodingError.typeMismatch(
                CompletionPromptInput.self,
                DecodingError.Context(codingPath: c.codingPath + [CodingKeys.prompt],
                                      debugDescription: "prompt must be string, [string], [int], or [[int]]")
            )
        }

        if let singleStop = try? c.decode(String.self, forKey: .stop) {
            stop = [singleStop]
        } else {
            stop = try c.decodeIfPresent([String].self, forKey: .stop)
        }

        if let intLP = try? c.decode(Int.self, forKey: .logprobs) {
            logprobs = intLP
        } else if let boolLP = try? c.decode(Bool.self, forKey: .logprobs) {
            logprobs = boolLP ? 1 : nil
        } else if c.contains(.logprobs), (try? c.decodeNil(forKey: .logprobs)) != true {
            throw DecodingError.typeMismatch(
                Int.self,
                DecodingError.Context(codingPath: c.codingPath + [CodingKeys.logprobs],
                                      debugDescription: "logprobs must be an integer or null")
            )
        } else {
            logprobs = nil
        }
    }
}

public struct CompletionLogprobs: Codable, Sendable, Equatable {
    public let textOffset: [Int]
    public let tokenLogprobs: [Float?]
    public let tokens: [String]
    public let topLogprobs: [[String: Float]?]

    enum CodingKeys: String, CodingKey {
        case tokens
        case textOffset = "text_offset"
        case tokenLogprobs = "token_logprobs"
        case topLogprobs = "top_logprobs"
    }

    public init(
        textOffset: [Int],
        tokenLogprobs: [Float?],
        tokens: [String],
        topLogprobs: [[String: Float]?]
    ) {
        self.textOffset = textOffset
        self.tokenLogprobs = tokenLogprobs
        self.tokens = tokens
        self.topLogprobs = topLogprobs
    }
}

public struct CompletionChoice: Codable, Sendable {
    public let text: String
    public let index: Int
    public let logprobs: CompletionLogprobs?
    public let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case text, index, logprobs
        case finishReason = "finish_reason"
    }

    public init(text: String, index: Int, logprobs: CompletionLogprobs? = nil, finishReason: String? = nil) {
        self.text = text
        self.index = index
        self.logprobs = logprobs
        self.finishReason = finishReason
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(text, forKey: .text)
        try c.encode(index, forKey: .index)
        if let logprobs { try c.encode(logprobs, forKey: .logprobs) } else { try c.encodeNil(forKey: .logprobs) }
        if let finishReason { try c.encode(finishReason, forKey: .finishReason) } else { try c.encodeNil(forKey: .finishReason) }
    }
}

public struct CompletionUsage: Codable, Sendable {
    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }

    public init(promptTokens: Int, completionTokens: Int) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = promptTokens + completionTokens
    }
}

public struct CompletionResponse: Codable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [CompletionChoice]
    public let usage: CompletionUsage?

    public init(
        id: String,
        model: String,
        choices: [CompletionChoice],
        usage: CompletionUsage? = nil,
        created: Int = Int(Date().timeIntervalSince1970)
    ) {
        self.id = id
        self.object = "text_completion"
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
    }
}

public struct TokenizerInfoResponse: Codable, Sendable, Equatable {
    public let eosToken: String
    public let bosToken: String?
    public let padToken: String?
    public let unkToken: String?
    public let chatTemplate: String?

    enum CodingKeys: String, CodingKey {
        case eosToken = "eos_token"
        case bosToken = "bos_token"
        case padToken = "pad_token"
        case unkToken = "unk_token"
        case chatTemplate = "chat_template"
    }

    public init(
        eosToken: String,
        bosToken: String? = nil,
        padToken: String? = nil,
        unkToken: String? = nil,
        chatTemplate: String? = nil
    ) {
        self.eosToken = eosToken
        self.bosToken = (bosToken?.isEmpty == false) ? bosToken : nil
        self.padToken = (padToken?.isEmpty == false) ? padToken : nil
        self.unkToken = (unkToken?.isEmpty == false) ? unkToken : nil
        self.chatTemplate = (chatTemplate?.isEmpty == false) ? chatTemplate : nil
    }
}

public struct TokenizeRequest: Decodable, Sendable {
    public let model: String?
    public let prompt: String?
    public let text: String?
    public let addSpecialTokens: Bool?

    enum CodingKeys: String, CodingKey {
        case model, prompt, text
        case addSpecialTokens = "add_special_tokens"
    }
}

public struct TokenizeResponse: Codable, Sendable, Equatable {
    public let tokens: [Int]
    public let count: Int
    public let maxModelLen: Int

    enum CodingKeys: String, CodingKey {
        case tokens, count
        case maxModelLen = "max_model_len"
    }

    public init(tokens: [Int], count: Int, maxModelLen: Int) {
        self.tokens = tokens
        self.count = count
        self.maxModelLen = maxModelLen
    }
}

public struct DetokenizeRequest: Decodable, Sendable {
    public let model: String?
    public let tokens: [Int]
}

public struct DetokenizeResponse: Codable, Sendable, Equatable {
    public let prompt: String

    public init(prompt: String) {
        self.prompt = prompt
    }
}

public struct CompletionInvalidInputError: Error, CustomStringConvertible, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var description: String { message }
}

public struct CompletionExecutionChoice: Sendable {
    public let index: Int
    public let text: String
    public let finishReason: String
    public let promptTokens: Int
    public let completionTokens: Int
    public let logprobs: CompletionLogprobs?

    public init(
        index: Int,
        text: String,
        finishReason: String,
        promptTokens: Int,
        completionTokens: Int,
        logprobs: CompletionLogprobs?
    ) {
        self.index = index
        self.text = text
        self.finishReason = finishReason
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.logprobs = logprobs
    }
}

public struct CompletionExecutionResult: Sendable {
    public let choices: [CompletionExecutionChoice]
    public let totalPromptTokens: Int
    public let totalCompletionTokens: Int

    public init(
        choices: [CompletionExecutionChoice],
        totalPromptTokens: Int,
        totalCompletionTokens: Int
    ) {
        self.choices = choices
        self.totalPromptTokens = totalPromptTokens
        self.totalCompletionTokens = totalCompletionTokens
    }
}
