// QWDecisionTypes.swift — Core domain types for single-pass System-1 decision evaluation.
import Foundation
import QWCore
import QWToken

public enum QWDecisionKind: String, Sendable, Hashable, Codable, CaseIterable {
    case noul
    case choice
    case multiChoice = "multiple"
    case score
    case read
    case estimate
    case box
    case boxes
}

public enum QWReadKind: String, Sendable, Hashable, Codable, CaseIterable {
    case text
    case number
    case date
    case path

    public var spanTokenCap: Int {
        switch self {
        case .text, .path: return 512
        case .number, .date: return 32
        }
    }
}

public enum QWScoreReadout: String, Sendable, Hashable, Codable, CaseIterable {
    case mean
    case median
    case argmax
}

public enum QWTruncationPolicy: String, Sendable, Hashable, Codable, CaseIterable {
    case error
    case head
    case tail
    case middle
    case autoState = "auto_state"
}

public struct QWDecisionCriterion: Sendable, Hashable, Codable, Equatable {
    public let key: String
    public let description: String?

    public init(key: String, description: String? = nil) {
        self.key = key
        self.description = description
    }
}

public enum QWDecisionState: Sendable, Hashable, Codable, Equatable {
    case text(String)
    case json(Data)
    case empty

    public func rendered(style: String = "json_only") -> String {
        switch self {
        case .empty:
            return ""
        case .text(let s):
            return s
        case .json(let data):
            if style.hasPrefix("json"), let parsed = try? QWJson.parse(data) {
                return Self.prettyJSON(parsed)
            }
            return String(decoding: data, as: UTF8.self)
        }
    }

    private static func prettyJSON(_ value: QWJson, indent: Int = 0) -> String {
        let pad = String(repeating: " ", count: indent)
        let pad2 = String(repeating: " ", count: indent + 2)
        switch value {
        case .undefined, .null:
            return "null"
        case .bool, .int, .double, .string:
            return value.tojson
        case .array(let items):
            guard !items.isEmpty else { return "[]" }
            let body = items.map { pad2 + prettyJSON($0, indent: indent + 2) }.joined(separator: ",\n")
            return "[\n\(body)\n\(pad)]"
        case .object(let pairs):
            guard !pairs.isEmpty else { return "{}" }
            let body = pairs.map { pad2 + QWJson.jsonQuote($0.0) + ": " + prettyJSON($0.1, indent: indent + 2) }
                .joined(separator: ",\n")
            return "{\n\(body)\n\(pad)}"
        }
    }
}

public struct QWDecisionQuestion: Sendable, Hashable, Codable, Equatable {
    public let name: String
    public let kind: QWDecisionKind
    public let instructions: String
    public let criteria: [QWDecisionCriterion]
    public let abstainBelow: Double?
    public let selectThreshold: Double?
    public let scoreReadout: QWScoreReadout?
    public let readKind: QWReadKind
    public let unit: String?
    public let estimateLow: Double?
    public let estimateHigh: Double?
    public let estimateStep: Double?
    public let maxBoxes: Int?
    public let jsonPath: String?
    public let chain: String?
    public let alwaysEmitBox: Bool

    enum CodingKeys: String, CodingKey {
        case name, kind, instructions, criteria
        case abstainBelow, selectThreshold, scoreReadout, readKind, unit
        case estimateLow, estimateHigh, estimateStep, maxBoxes, jsonPath, chain
        case alwaysEmitBox
    }

    public init(
        name: String,
        kind: QWDecisionKind,
        instructions: String,
        criteria: [QWDecisionCriterion] = [],
        abstainBelow: Double? = nil,
        selectThreshold: Double? = nil,
        scoreReadout: QWScoreReadout? = nil,
        readKind: QWReadKind = .text,
        unit: String? = nil,
        estimateLow: Double? = nil,
        estimateHigh: Double? = nil,
        estimateStep: Double? = nil,
        maxBoxes: Int? = nil,
        jsonPath: String? = nil,
        chain: String? = nil,
        alwaysEmitBox: Bool = false
    ) {
        self.name = name
        self.kind = kind
        self.instructions = instructions
        self.criteria = criteria
        self.abstainBelow = abstainBelow
        self.selectThreshold = selectThreshold
        self.scoreReadout = scoreReadout
        self.readKind = readKind
        self.unit = unit
        self.estimateLow = estimateLow
        self.estimateHigh = estimateHigh
        self.estimateStep = estimateStep
        self.maxBoxes = maxBoxes
        self.jsonPath = jsonPath
        self.chain = chain
        self.alwaysEmitBox = alwaysEmitBox
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.name = try c.decode(String.self, forKey: .name)
        self.kind = try c.decode(QWDecisionKind.self, forKey: .kind)
        self.instructions = try c.decode(String.self, forKey: .instructions)
        self.criteria = try c.decodeIfPresent([QWDecisionCriterion].self, forKey: .criteria) ?? []
        self.abstainBelow = try c.decodeIfPresent(Double.self, forKey: .abstainBelow)
        self.selectThreshold = try c.decodeIfPresent(Double.self, forKey: .selectThreshold)
        self.scoreReadout = try c.decodeIfPresent(QWScoreReadout.self, forKey: .scoreReadout)
        self.readKind = try c.decodeIfPresent(QWReadKind.self, forKey: .readKind) ?? .text
        self.unit = try c.decodeIfPresent(String.self, forKey: .unit)
        self.estimateLow = try c.decodeIfPresent(Double.self, forKey: .estimateLow)
        self.estimateHigh = try c.decodeIfPresent(Double.self, forKey: .estimateHigh)
        self.estimateStep = try c.decodeIfPresent(Double.self, forKey: .estimateStep)
        self.maxBoxes = try c.decodeIfPresent(Int.self, forKey: .maxBoxes)
        self.jsonPath = try c.decodeIfPresent(String.self, forKey: .jsonPath)
        self.chain = try c.decodeIfPresent(String.self, forKey: .chain)
        self.alwaysEmitBox = try c.decodeIfPresent(Bool.self, forKey: .alwaysEmitBox) ?? false
    }
}

public struct QWDecisionRequest: Sendable, Hashable, Equatable {
    public let state: QWDecisionState
    public let questions: [QWDecisionQuestion]
    public let truncation: QWTruncationPolicy?
    public let contextBudget: Int?
    public let systemPrompt: String?
    public let stateStyle: String
    public let chain: String
    public let images: [Data]

    public init(
        state: QWDecisionState,
        questions: [QWDecisionQuestion],
        truncation: QWTruncationPolicy? = nil,
        contextBudget: Int? = nil,
        systemPrompt: String? = nil,
        stateStyle: String = "json_only",
        chain: String = "default",
        images: [Data] = []
    ) {
        self.state = state
        self.questions = questions
        self.truncation = truncation
        self.contextBudget = contextBudget
        self.systemPrompt = systemPrompt
        self.stateStyle = stateStyle
        self.chain = chain
        self.images = images
    }
}

public struct QWDecisionBox: Sendable, Hashable, Codable, Equatable {
    public let top: Double
    public let bottom: Double
    public let left: Double
    public let right: Double
    public let found: Double
    public let confidence: Double

    public init(top: Double, bottom: Double, left: Double, right: Double, found: Double, confidence: Double) {
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.found = found
        self.confidence = confidence
    }
}

public struct QWDecisionProbability: Sendable, Hashable, Codable, Equatable {
    public let key: String
    public let value: Double

    public init(key: String, value: Double) {
        self.key = key
        self.value = value
    }
}

public struct QWDecisionAnswer: Sendable, Hashable, Codable, Equatable {
    public let name: String
    public let kind: QWDecisionKind
    public let choice: String?
    public let selectedChoices: [String]
    public let abstained: Bool
    public let noul: Bool?
    public let confidence: Double
    public let entropyConfidence: Double?
    public let probs: [QWDecisionProbability]
    public let score: Double?
    public let sigma: Double?
    public let spreadReported: Bool
    public let text: String?
    public let number: Double?
    public let date: String?
    public let span: Range<Int>?
    public let charSpan: Range<Int>?
    public let jsonPath: String?
    public let found: Bool?
    public let foundProb: Double?
    public let estimate: Double?
    public let interval: [Double]?
    public let boxes: [QWDecisionBox]
    public let atLeast: Bool
    public let presence: [Double]

    public init(
        name: String,
        kind: QWDecisionKind,
        choice: String? = nil,
        selectedChoices: [String] = [],
        abstained: Bool = false,
        noul: Bool? = nil,
        confidence: Double = 0,
        entropyConfidence: Double? = nil,
        probs: [QWDecisionProbability] = [],
        score: Double? = nil,
        sigma: Double? = nil,
        spreadReported: Bool = false,
        text: String? = nil,
        number: Double? = nil,
        date: String? = nil,
        span: Range<Int>? = nil,
        charSpan: Range<Int>? = nil,
        jsonPath: String? = nil,
        found: Bool? = nil,
        foundProb: Double? = nil,
        estimate: Double? = nil,
        interval: [Double]? = nil,
        boxes: [QWDecisionBox] = [],
        atLeast: Bool = false,
        presence: [Double] = []
    ) {
        self.name = name
        self.kind = kind
        self.choice = choice
        self.selectedChoices = selectedChoices
        self.abstained = abstained
        self.noul = noul
        self.confidence = confidence
        self.entropyConfidence = entropyConfidence
        self.probs = probs
        self.score = score
        self.sigma = sigma
        self.spreadReported = spreadReported
        self.text = text
        self.number = number
        self.date = date
        self.span = span
        self.charSpan = charSpan
        self.jsonPath = jsonPath
        self.found = found
        self.foundProb = foundProb
        self.estimate = estimate
        self.interval = interval
        self.boxes = boxes
        self.atLeast = atLeast
        self.presence = presence
    }
}

public struct QWDecisionUsage: Sendable, Hashable, Codable, Equatable {
    public let promptTokens: Int
    public let outputTokens: Int
    public let truncatedTokens: Int
    public let latencyMs: Double

    public init(promptTokens: Int, outputTokens: Int = 0, truncatedTokens: Int = 0, latencyMs: Double = 0) {
        self.promptTokens = promptTokens
        self.outputTokens = outputTokens
        self.truncatedTokens = truncatedTokens
        self.latencyMs = latencyMs
    }
}

public struct QWDecisionResponse: Sendable, Hashable, Codable, Equatable {
    public let answers: [QWDecisionAnswer]
    public let usage: QWDecisionUsage
    public let chain: String

    public init(answers: [QWDecisionAnswer], usage: QWDecisionUsage, chain: String = "default") {
        self.answers = answers
        self.usage = usage
        self.chain = chain
    }
}

public enum QWDecisionError: Error, Equatable, CustomStringConvertible {
    case invalidQuestion(String)
    case contextOverflow(tokens: Int, limit: Int)
    case unsupportedLabelSubset(requested: [String], allowed: [String])
    case missingSingleTokenAlias(optionCount: Int)
    case calibrationFailed(String)
    case backendFailed(String)

    public var description: String {
        switch self {
        case .invalidQuestion(let msg): return "invalid decision question: \(msg)"
        case .contextOverflow(let tokens, let limit):
            return "decision input (\(tokens) tokens) exceeds context budget (\(limit))"
        case .unsupportedLabelSubset(let req, let allowed):
            return "requested labels \(req) are not a subset of fixed classifier labels \(allowed)"
        case .missingSingleTokenAlias(let count):
            return "tokenizer lacks distinct single-token aliases for \(count) options"
        case .calibrationFailed(let msg): return "calibration failed: \(msg)"
        case .backendFailed(let msg): return "decision backend failed: \(msg)"
        }
    }
}
