// QWServerSchema+Messages.swift — Chat message DTOs, logprob types,
// metadata validation, and shared scalar metadata coercion helpers.

import Foundation
import QWCore
import QWToken

public struct ChatMessageContent: Codable, Sendable {
    public enum Part: Codable, Sendable {
        case text(String)
        case imageUrl(String)
        case videoUrl(QWVideoInputSpec)

        enum CodingKeys: String, CodingKey {
            case type
            case text
            case imageUrl = "image_url"
            case videoUrl = "video_url"
            case video
            case url
            case fps
            case maxFrames = "max_frames"
            case maxPixels = "max_pixels"
        }

        enum ImageUrlKeys: String, CodingKey {
            case url
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let type = try container.decode(String.self, forKey: .type).lowercased()
            if type == "text" || type == "input_text" || type == "output_text" {
                let text = (try? container.decode(String.self, forKey: .text)) ?? ""
                self = .text(text)
            } else if type == "image_url" || type == "input_image" {
                if let direct = try? container.decode(String.self, forKey: .imageUrl) {
                    self = .imageUrl(direct)
                } else {
                    let imgContainer = try container.nestedContainer(keyedBy: ImageUrlKeys.self, forKey: .imageUrl)
                    let url = try imgContainer.decode(String.self, forKey: .url)
                    self = .imageUrl(url)
                }
            } else if type == "video_url" || type == "input_video" {
                let topFps = try container.decodeIfPresent(Double.self, forKey: .fps)
                let topFrames = try container.decodeIfPresent(Int.self, forKey: .maxFrames)
                let topPixels = try container.decodeIfPresent(Int.self, forKey: .maxPixels)
                if let spec = try? container.decode(QWVideoInputSpec.self, forKey: .videoUrl) {
                    self = .videoUrl(QWVideoInputSpec(
                        url: spec.url,
                        fps: spec.fps ?? topFps,
                        maxFrames: spec.maxFrames ?? topFrames,
                        maxPixels: spec.maxPixels ?? topPixels
                    ))
                } else if let direct = try? container.decode(String.self, forKey: .videoUrl) {
                    self = .videoUrl(QWVideoInputSpec(
                        url: direct, fps: topFps, maxFrames: topFrames, maxPixels: topPixels
                    ))
                } else if let fallbackURL = try? container.decode(String.self, forKey: .url) {
                    self = .videoUrl(QWVideoInputSpec(
                        url: fallbackURL, fps: topFps, maxFrames: topFrames, maxPixels: topPixels
                    ))
                } else {
                    throw DecodingError.dataCorruptedError(
                        forKey: .videoUrl, in: container, debugDescription: "video_url requires a url string or object"
                    )
                }
            } else if type == "video" {
                let topFps = try container.decodeIfPresent(Double.self, forKey: .fps)
                let topFrames = try container.decodeIfPresent(Int.self, forKey: .maxFrames)
                let topPixels = try container.decodeIfPresent(Int.self, forKey: .maxPixels)
                if let spec = try? container.decode(QWVideoInputSpec.self, forKey: .video) {
                    self = .videoUrl(QWVideoInputSpec(
                        url: spec.url,
                        fps: spec.fps ?? topFps,
                        maxFrames: spec.maxFrames ?? topFrames,
                        maxPixels: spec.maxPixels ?? topPixels
                    ))
                } else if let direct = (try? container.decode(String.self, forKey: .video))
                    ?? (try? container.decode(String.self, forKey: .url)) {
                    self = .videoUrl(QWVideoInputSpec(
                        url: direct, fps: topFps, maxFrames: topFrames, maxPixels: topPixels
                    ))
                } else {
                    throw DecodingError.dataCorruptedError(
                        forKey: .video, in: container, debugDescription: "video part requires a url or video string"
                    )
                }
            } else {
                self = .text("")
            }
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .text(let t):
                try container.encode("text", forKey: .type)
                try container.encode(t, forKey: .text)
            case .imageUrl(let u):
                try container.encode("image_url", forKey: .type)
                var imgContainer = container.nestedContainer(keyedBy: ImageUrlKeys.self, forKey: .imageUrl)
                try imgContainer.encode(u, forKey: .url)
            case .videoUrl(let v):
                try container.encode("video_url", forKey: .type)
                try container.encode(v, forKey: .videoUrl)
            }
        }
    }

    public let rawText: String?
    public let parts: [Part]?

    public init(string: String) {
        self.rawText = string
        self.parts = nil
    }

    public init(parts: [Part]) {
        self.rawText = nil
        self.parts = parts
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            self.rawText = str
            self.parts = nil
        } else if let p = try? container.decode([Part].self) {
            self.rawText = nil
            self.parts = p
        } else {
            self.rawText = ""
            self.parts = nil
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if let rawText {
            try container.encode(rawText)
        } else if let parts {
            try container.encode(parts)
        } else {
            try container.encode("")
        }
    }

    public var textValue: String {
        if let rawText { return rawText }
        if let parts {
            var res = ""
            for p in parts {
                if case .text(let t) = p {
                    if !res.isEmpty { res += "\n" }
                    res += t
                }
            }
            return res
        }
        return ""
    }

    /// Content string formatted for chat-template rendering, inserting canonical `<|vision_start|>...<|vision_end|>`
    /// markers for `image_url` and `video_url` parts when the text does not already include explicit media markers.
    public var templateTextValue: String {
        if let rawText { return rawText }
        guard let parts else { return "" }
        let plainText = textValue
        let hasImageParts = !imageUrls.isEmpty
        let hasVideoParts = !videoInputs.isEmpty
        let explicitImg = !hasImageParts || plainText.contains("<|image_pad|>") || plainText.contains("<|image|>")
        let explicitVid = !hasVideoParts || plainText.contains("<|video_pad|>") || plainText.contains("<|video|>")
        if explicitImg && explicitVid && (hasImageParts || hasVideoParts) {
            return plainText
        }
        var res = ""
        for p in parts {
            switch p {
            case .text(let t):
                if !res.isEmpty && !res.hasSuffix(">") { res += "\n" }
                res += t
            case .imageUrl:
                res += "<|vision_start|><|image_pad|><|vision_end|>"
            case .videoUrl:
                res += "<|vision_start|><|video_pad|><|vision_end|>"
            }
        }
        return res
    }

    public var imageUrls: [String] {
        guard let parts else { return [] }
        var urls: [String] = []
        for p in parts {
            if case .imageUrl(let u) = p {
                urls.append(u)
            }
        }
        return urls
    }

    public var videoInputs: [QWVideoInputSpec] {
        guard let parts else { return [] }
        var items: [QWVideoInputSpec] = []
        for p in parts {
            if case .videoUrl(let v) = p {
                items.append(v)
            }
        }
        return items
    }
}

public struct ChatMessage: Codable, Sendable {
    public let role: String
    public let content: ChatMessageContent?
    public let toolCalls: [ToolCall]?
    public let toolCallId: String?
    public let name: String?
    public let reasoningContent: String?

    enum CodingKeys: String, CodingKey {
        case role
        case content
        case toolCalls = "tool_calls"
        case toolCallId = "tool_call_id"
        case name
        case reasoningContent = "reasoning_content"
    }

    public init(
        role: String,
        content: ChatMessageContent? = nil,
        toolCalls: [ToolCall]? = nil,
        toolCallId: String? = nil,
        name: String? = nil,
        reasoningContent: String? = nil
    ) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolCallId = toolCallId
        self.name = name
        self.reasoningContent = reasoningContent
    }

    public init(role: String, text: String) {
        self.role = role
        self.content = ChatMessageContent(string: text)
        self.toolCalls = nil
        self.toolCallId = nil
        self.name = nil
        self.reasoningContent = nil
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.role = try container.decode(String.self, forKey: .role)
        self.content = try container.decodeIfPresent(ChatMessageContent.self, forKey: .content)
        self.toolCalls = try container.decodeIfPresent([ToolCall].self, forKey: .toolCalls)
        self.toolCallId = try container.decodeIfPresent(String.self, forKey: .toolCallId)
        self.name = try container.decodeIfPresent(String.self, forKey: .name)
        self.reasoningContent = try container.decodeIfPresent(String.self, forKey: .reasoningContent)
    }

    public func extractTextContent() -> String {
        content?.textValue ?? ""
    }
}

public struct ChatLogprobTokenEntry: Codable, Sendable {
    public struct TopLogprob: Codable, Sendable {
        public let token: String
        public let logprob: Float
        public let bytes: [Int]?

        public init(token: String, logprob: Float, bytes: [Int]? = nil) {
            self.token = token
            self.logprob = logprob
            self.bytes = bytes
        }
    }

    public let token: String
    public let logprob: Float
    public let bytes: [Int]?
    public let topLogprobs: [TopLogprob]

    enum CodingKeys: String, CodingKey {
        case token
        case logprob
        case bytes
        case topLogprobs = "top_logprobs"
    }

    public init(token: String, logprob: Float, bytes: [Int]? = nil, topLogprobs: [TopLogprob] = []) {
        self.token = token
        self.logprob = logprob
        self.bytes = bytes
        self.topLogprobs = topLogprobs
    }
}

public struct ChoiceLogprobs: Codable, Sendable {
    public let content: [ChatLogprobTokenEntry]

    public init(content: [ChatLogprobTokenEntry]) {
        self.content = content
    }
}

public struct QWMetadataValidationError: Error, CustomStringConvertible, Sendable {
    public let message: String

    public init(_ message: String) {
        self.message = message
    }

    public var description: String { message }
}

/// Shared scalar metadata coercion and strict JSON numeric extraction helpers.
enum QWSchemaMetadataCoercion {
    static func coerceMetadataDictionary(
        from rawMeta: QWJson,
        codingPath: [CodingKey]
    ) throws -> [String: String] {
        guard case .object(let pairs) = rawMeta else {
            throw DecodingError.typeMismatch(
                [String: String].self,
                DecodingError.Context(
                    codingPath: codingPath,
                    debugDescription: "metadata must be a JSON object"
                )
            )
        }
        var coerced: [String: String] = [:]
        for (k, v) in pairs {
            switch v {
            case .string(let s):
                coerced[k] = s
            case .bool(let b):
                coerced[k] = b ? "true" : "false"
            case .int(let i):
                coerced[k] = String(i)
            case .double(let d):
                if d.isFinite && d >= Double(Int64.min) && d < Double(Int64.max)
                    && d.truncatingRemainder(dividingBy: 1) == 0 {
                    coerced[k] = String(Int64(d))
                } else {
                    coerced[k] = String(d)
                }
            case .null, .undefined:
                break
            case .array, .object:
                throw DecodingError.typeMismatch(
                    String.self,
                    DecodingError.Context(
                        codingPath: codingPath,
                        debugDescription: "metadata['\(k)'] must be a scalar string, number, or boolean"
                    )
                )
            }
        }
        return coerced
    }

    static func decodeOptionalMetadata<K: CodingKey>(
        from container: KeyedDecodingContainer<K>,
        forKey key: K
    ) throws -> [String: String]? {
        guard container.contains(key), (try? container.decodeNil(forKey: key)) != true else {
            return nil
        }
        let rawMeta = try container.decode(QWJson.self, forKey: key)
        return try coerceMetadataDictionary(from: rawMeta, codingPath: container.codingPath + [key])
    }

    static func decodeOptionalStop<K: CodingKey>(
        from container: KeyedDecodingContainer<K>,
        forKey key: K
    ) throws -> [String]? {
        if let singleStop = try? container.decode(String.self, forKey: key) {
            return [singleStop]
        }
        return try container.decodeIfPresent([String].self, forKey: key)
    }

    static func strictInt(_ value: QWJson?) -> Int? {
        guard let value else { return nil }
        switch value {
        case .int(let i): return i
        case .double(let d):
            guard d.isFinite, d.rounded(.towardZero) == d,
                  d >= Double(Int.min), d <= Double(Int.max) else { return nil }
            return Int(d)
        default:
            return nil
        }
    }

    static func strictDouble(_ value: QWJson?) -> Double? {
        guard let value else { return nil }
        switch value {
        case .double(let d): return d.isFinite ? d : nil
        case .int(let i): return Double(i)
        default: return nil
        }
    }

    static func canonicalJSONString(_ raw: QWJson) -> String? {
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
}
