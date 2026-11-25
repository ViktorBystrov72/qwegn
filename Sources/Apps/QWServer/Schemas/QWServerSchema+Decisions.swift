// QWServerSchema+Decisions.swift — Wire DTOs for /v1/decisions, /v1/systemone, and /v1/laya.

import Foundation
import QWCore
import QWDecision
import QWEngine
import QWToken

/// Incoming question specification supporting both `/v1/decisions` and `/v1/systemone` dialects.
public struct QWDecisionHTTPQuestion: Codable, Sendable {
    public let name: String?
    public let kind: String?
    public let type: String?
    public let instructions: String?
    public let question: String?
    public let prompt: String?
    public let criteria: QWJson?
    public let options: [String]?
    public let labels: [String]?
    public let abstainBelow: Double?
    public let selectThreshold: Double?
    public let scoreReadout: String?
    public let readKind: String?
    public let unit: String?
    public let estimateLow: Double?
    public let estimateHigh: Double?
    public let estimateStep: Double?
    public let minValue: Double?
    public let maxValue: Double?
    public let stepValue: Double?
    public let maxBoxes: Int?
    public let jsonPath: String?
    public let chain: String?

    enum CodingKeys: String, CodingKey {
        case name, kind, type, instructions, question, prompt, criteria, options, labels, unit, chain
        case abstainBelow = "abstain_below"
        case selectThreshold = "select_threshold"
        case scoreReadout = "score_readout"
        case readKind = "read_kind"
        case estimateLow = "estimate_low"
        case estimateHigh = "estimate_high"
        case estimateStep = "estimate_step"
        case minValue = "min"
        case maxValue = "max"
        case stepValue = "step"
        case maxBoxes = "max_boxes"
        case jsonPath = "json_path"
    }

    public init(
        name: String? = nil,
        kind: String? = nil,
        type: String? = nil,
        instructions: String? = nil,
        question: String? = nil,
        prompt: String? = nil,
        criteria: QWJson? = nil,
        options: [String]? = nil,
        labels: [String]? = nil,
        abstainBelow: Double? = nil,
        selectThreshold: Double? = nil,
        scoreReadout: String? = nil,
        readKind: String? = nil,
        unit: String? = nil,
        estimateLow: Double? = nil,
        estimateHigh: Double? = nil,
        estimateStep: Double? = nil,
        minValue: Double? = nil,
        maxValue: Double? = nil,
        stepValue: Double? = nil,
        maxBoxes: Int? = nil,
        jsonPath: String? = nil,
        chain: String? = nil
    ) {
        self.name = name
        self.kind = kind
        self.type = type
        self.instructions = instructions
        self.question = question
        self.prompt = prompt
        self.criteria = criteria
        self.options = options
        self.labels = labels
        self.abstainBelow = abstainBelow
        self.selectThreshold = selectThreshold
        self.scoreReadout = scoreReadout
        self.readKind = readKind
        self.unit = unit
        self.estimateLow = estimateLow
        self.estimateHigh = estimateHigh
        self.estimateStep = estimateStep
        self.minValue = minValue
        self.maxValue = maxValue
        self.stepValue = stepValue
        self.maxBoxes = maxBoxes
        self.jsonPath = jsonPath
        self.chain = chain
    }

    public func toDecisionQuestion(
        fallbackName: String,
        defaultScoreReadout: QWScoreReadout
    ) throws -> QWDecisionQuestion {
        let resolvedName = (name?.trimmingCharacters(in: .whitespacesAndNewlines)).flatMap { $0.isEmpty ? nil : $0 } ?? fallbackName
        let parsedCriteria = try parseCriteria()
        let rawKind = (kind ?? type ?? (parsedCriteria.isEmpty ? "noul" : "choice"))
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let (mappedKind, inferredReadKind) = try Self.mapKind(rawKind)
        let explicitRead = readKind.flatMap { QWReadKind(rawValue: $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()) }
        let resolvedReadout = try parseScoreReadout(scoreReadout) ?? defaultScoreReadout
        let resolvedPrompt = instructions ?? question ?? prompt ?? ""
        return QWDecisionQuestion(
            name: resolvedName,
            kind: mappedKind,
            instructions: resolvedPrompt,
            criteria: parsedCriteria,
            abstainBelow: abstainBelow,
            selectThreshold: selectThreshold,
            scoreReadout: resolvedReadout,
            readKind: explicitRead ?? inferredReadKind,
            unit: unit,
            estimateLow: estimateLow ?? minValue,
            estimateHigh: estimateHigh ?? maxValue,
            estimateStep: estimateStep ?? stepValue,
            maxBoxes: maxBoxes,
            jsonPath: jsonPath,
            chain: chain
        )
    }

    private func parseCriteria() throws -> [QWDecisionCriterion] {
        if let criteria {
            switch criteria {
            case .object(let pairs):
                return pairs.map { key, val in
                    let desc: String? = switch val {
                    case .null, .undefined: nil
                    case .string(let s): s.isEmpty ? nil : s
                    default: val.tojson
                    }
                    return QWDecisionCriterion(key: key, description: desc)
                }
            case .array(let arr):
                return arr.compactMap { item in
                    switch item {
                    case .string(let s):
                        return QWDecisionCriterion(key: s)
                    case .object:
                        if case .string(let k) = item.get("key") {
                            let d: String? = {
                                if case .string(let ds) = item.get("description") { return ds }
                                return nil
                            }()
                            return QWDecisionCriterion(key: k, description: d)
                        }
                        return QWDecisionCriterion(key: item.tojson)
                    case .null, .undefined:
                        return nil
                    default:
                        return QWDecisionCriterion(key: item.stringified)
                    }
                }
            case .null, .undefined:
                break
            default:
                throw QWDecisionError.invalidQuestion("criteria must be a JSON object or string array")
            }
        }
        if let list = options ?? labels {
            return list.map { QWDecisionCriterion(key: $0) }
        }
        return []
    }

    private func parseScoreReadout(_ raw: String?) throws -> QWScoreReadout? {
        guard let trimmed = raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              !trimmed.isEmpty else { return nil }
        guard let parsed = QWScoreReadout(rawValue: trimmed) else {
            throw QWDecisionError.invalidQuestion("unsupported score_readout '\(trimmed)' (supported: mean, median, argmax)")
        }
        return parsed
    }

    private static func mapKind(_ raw: String) throws -> (QWDecisionKind, QWReadKind) {
        switch raw {
        case "noul", "binary", "predicate", "bool": return (.noul, .text)
        case "choice", "select", "classify", "label", "rank": return (.choice, .text)
        case "multiple", "multi_choice", "multichoice", "multi": return (.multiChoice, .text)
        case "score", "ordinal", "rating": return (.score, .text)
        case "read", "extract", "span", "extract_text": return (.read, .text)
        case "extract_number", "number": return (.read, .number)
        case "extract_date", "date": return (.read, .date)
        case "path", "json_path": return (.read, .path)
        case "estimate": return (.estimate, .text)
        case "box": return (.box, .text)
        case "boxes": return (.boxes, .text)
        default:
            throw QWDecisionError.invalidQuestion("unsupported decision kind '\(raw)'")
        }
    }
}

/// Canonical `POST /v1/decisions` and `POST /v1/systemone` request payload.
public struct QWDecisionHTTPRequest: Codable, Sendable {
    public let model: String?
    public let state: QWJson?
    public let context: QWJson?
    public let questions: QWJson
    public let truncation: String?
    public let contextBudget: Int?
    public let systemPrompt: String?
    public let stateStyle: String?
    public let chain: String?
    public let scoreReadout: String?
    public let images: [String]?

    enum CodingKeys: String, CodingKey {
        case model, state, context, questions, truncation, chain, images
        case contextBudget = "context_budget"
        case systemPrompt = "system_prompt"
        case stateStyle = "state_style"
        case scoreReadout = "score_readout"
    }

    public func toDecisionRequest() throws -> QWDecisionRequest {
        let defaultReadout: QWScoreReadout
        if let raw = scoreReadout?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
           !raw.isEmpty {
            guard let parsed = QWScoreReadout(rawValue: raw) else {
                throw QWDecisionError.invalidQuestion("unsupported score_readout '\(raw)'")
            }
            defaultReadout = parsed
        } else {
            defaultReadout = .mean
        }

        let truncationPolicy: QWTruncationPolicy?
        if let rawTrunc = truncation?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
           !rawTrunc.isEmpty {
            guard let parsed = QWTruncationPolicy(rawValue: rawTrunc) else {
                throw QWDecisionError.invalidQuestion("unsupported truncation policy '\(rawTrunc)'")
            }
            truncationPolicy = parsed
        } else {
            truncationPolicy = nil
        }

        let parsedQuestions = try Self.decodeQuestions(questions, defaultScoreReadout: defaultReadout)
        let parsedState = try Self.decodeState(state ?? context)
        let decodedImages = try Self.decodeHTTPImages(images)
        return QWDecisionRequest(
            state: parsedState,
            questions: parsedQuestions,
            truncation: truncationPolicy,
            contextBudget: contextBudget,
            systemPrompt: systemPrompt,
            stateStyle: stateStyle ?? "json_only",
            chain: chain ?? "default",
            images: decodedImages
        )
    }

    static func decodeHTTPImages(_ raw: [String]?) throws -> [Data] {
        guard let raw, !raw.isEmpty else { return [] }
        guard raw.count <= QWImageDecodePolicy.defaultMaxImagesPerTurn else {
            throw QWDecisionError.invalidQuestion(
                "Too many images in decision request (\(raw.count)); maximum is \(QWImageDecodePolicy.defaultMaxImagesPerTurn)"
            )
        }
        return try raw.map { source in
            let data: Data
            do {
                data = try QWImageDecoder.resolveSourceData(urlOrDataUri: source, policy: .serverDefault)
            } catch let imgErr as QWImageDecoderError {
                throw QWDecisionError.invalidQuestion(imgErr.description)
            }
            guard !data.isEmpty else {
                throw QWDecisionError.invalidQuestion("Decoded HTTP image payload is empty")
            }
            return data
        }
    }

    static func decodeState(_ raw: QWJson?) throws -> QWDecisionState {
        guard let raw else { return .empty }
        switch raw {
        case .undefined, .null:
            return .empty
        case .string(let s):
            return .text(s)
        case .object:
            if case .string(let kind) = raw.get("type") {
                switch kind.lowercased() {
                case "text":
                    if case .string(let txt) = raw.get("text") { return .text(txt) }
                case "json":
                    let payload = raw.get("json")
                    if !payload.isUndefined { return .json(Data(payload.tojson.utf8)) }
                default:
                    break
                }
            }
            return .json(Data(raw.tojson.utf8))
        default:
            return .json(Data(raw.tojson.utf8))
        }
    }

    static func decodeQuestions(
        _ raw: QWJson,
        defaultScoreReadout: QWScoreReadout
    ) throws -> [QWDecisionQuestion] {
        switch raw {
        case .array(let items):
            return try items.enumerated().map { idx, item in
                let dto = try QWServerRequestDecoder.decode(QWDecisionHTTPQuestion.self, from: Data(item.tojson.utf8))
                return try dto.toDecisionQuestion(fallbackName: "q\(idx)", defaultScoreReadout: defaultScoreReadout)
            }
        case .object(let pairs):
            return try pairs.map { key, val in
                let dto = try QWServerRequestDecoder.decode(QWDecisionHTTPQuestion.self, from: Data(val.tojson.utf8))
                return try dto.toDecisionQuestion(fallbackName: key, defaultScoreReadout: defaultScoreReadout)
            }
        default:
            throw QWDecisionError.invalidQuestion("questions must be an array or keyed object")
        }
    }
}

public struct QWDecisionBatchHTTPRequest: Codable, Sendable {
    public let items: [QWDecisionHTTPRequest]?
    public let requests: [QWDecisionHTTPRequest]?

    public var resolvedItems: [QWDecisionHTTPRequest] {
        items ?? requests ?? []
    }
}

public struct QWDecisionHTTPResponse: Codable, Sendable {
    public let answers: [QWDecisionAnswer]
    public let usage: QWDecisionUsage
    public let chain: String
    public let model: String
    public let backend: String

    public init(envelope: QWDecisionExecutionEnvelope) {
        self.answers = envelope.response.answers
        self.usage = envelope.response.usage
        self.chain = envelope.response.chain
        self.model = envelope.modelName
        self.backend = envelope.backendKind
    }
}

public struct QWDecisionBatchHTTPResponse: Codable, Sendable {
    public let results: [QWDecisionHTTPResponse]
}

/// Wire payload for `POST /v1/laya` enforcing the 60,000-character rail (`ADR-008`).
public struct QWLayaHTTPRequest: Codable, Sendable {
    public static let maxContextCharacters = 60_000

    public let model: String?
    public let context: QWJson?
    public let state: QWJson?
    public let question: String?
    public let instructions: String?
    public let kind: String?
    public let mode: String?
    public let command: String?
    public let options: [String]?
    public let labels: [String]?
    public let criteria: QWJson?
    public let questions: QWJson?
    public let chain: String?
    public let guardMode: Bool?
    public let scoreReadout: String?
    public let images: [String]?

    enum CodingKeys: String, CodingKey {
        case model, context, state, question, instructions, kind, mode, command
        case options, labels, criteria, questions, chain, images
        case guardMode = "guard"
        case scoreReadout = "score_readout"
    }

    public func toDecisionRequest() throws -> QWDecisionRequest {
        let resolvedState: QWDecisionState
        if let cmd = command, state == nil, context == nil {
            resolvedState = .text("About to execute command: \(cmd)")
        } else {
            resolvedState = try QWDecisionHTTPRequest.decodeState(state ?? context)
        }

        let charCount: Int = switch resolvedState {
        case .empty: 0
        case .text(let s): s.count
        case .json(let d): String(decoding: d, as: UTF8.self).count
        }
        guard charCount <= Self.maxContextCharacters else {
            throw QWDecisionError.invalidQuestion(
                "state length \(charCount) exceeds \(Self.maxContextCharacters)-character Laya rail"
            )
        }

        let readout: QWScoreReadout
        if let rawR = scoreReadout?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(), !rawR.isEmpty {
            guard let parsed = QWScoreReadout(rawValue: rawR) else {
                throw QWDecisionError.invalidQuestion("unsupported score_readout '\(rawR)'")
            }
            readout = parsed
        } else {
            readout = .mean
        }

        let qList: [QWDecisionQuestion]
        if let questions, !questions.isUndefined, questions != .null {
            qList = try QWDecisionHTTPRequest.decodeQuestions(questions, defaultScoreReadout: readout)
        } else if guardMode == true || mode?.lowercased() == "guard" || (command != nil && question == nil && instructions == nil) {
            let guardOptions = options ?? ["allow", "confirm", "deny"]
            let guardCriteria = [
                "allow": "safe to run without operator confirmation",
                "confirm": "potentially risky, ask operator before executing",
                "deny": "destructive or policy-violating command",
            ]
            let criteria = guardOptions.map { key in
                QWDecisionCriterion(key: key, description: guardCriteria[key])
            }
            let promptText = question ?? instructions ?? "How should the harness treat this command?"
            qList = [QWDecisionQuestion(
                name: "guard",
                kind: .choice,
                instructions: promptText,
                criteria: criteria,
                chain: chain ?? "guardrail"
            )]
        } else {
            let promptText = question ?? instructions ?? ""
            guard !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw QWDecisionError.invalidQuestion("Laya request requires 'question', 'questions', or 'command'")
            }
            let single = QWDecisionHTTPQuestion(
                name: "answer",
                kind: kind,
                instructions: promptText,
                criteria: criteria,
                options: options,
                labels: labels,
                scoreReadout: scoreReadout,
                chain: chain
            )
            qList = [try single.toDecisionQuestion(fallbackName: "answer", defaultScoreReadout: readout)]
        }

        let decodedImages = try QWDecisionHTTPRequest.decodeHTTPImages(images)
        return QWDecisionRequest(
            state: resolvedState,
            questions: qList,
            truncation: nil,
            chain: chain ?? "default",
            images: decodedImages
        )
    }
}

public struct QWLayaHTTPResponse: Codable, Sendable {
    public let answer: QWJson
    public let decision: String?
    public let confidence: Double
    public let probabilities: [String: Double]
    public let probs: [QWDecisionProbability]
    public let answers: [String: QWDecisionAnswer]
    public let usage: QWDecisionUsage
    public let chain: String
    public let model: String
    public let backend: String

    public init(envelope: QWDecisionExecutionEnvelope) {
        let primary = envelope.response.answers.first
        let probPairs = primary?.probs ?? []
        var probDict: [String: Double] = [:]
        for p in probPairs { probDict[p.key] = p.value }

        let summaryValue: QWJson
        let decisionLabel: String?
        if let primary {
            switch primary.kind {
            case .noul:
                let b = primary.noul ?? false
                summaryValue = .bool(b)
                decisionLabel = b ? "yes" : "no"
            case .choice, .multiChoice:
                if let c = primary.choice {
                    summaryValue = .string(c)
                    decisionLabel = c
                } else {
                    summaryValue = .array(primary.selectedChoices.map { .string($0) })
                    decisionLabel = primary.selectedChoices.first
                }
            case .score:
                let val = primary.score ?? 0
                summaryValue = .double(val)
                decisionLabel = nil
            case .estimate:
                let val = primary.estimate ?? 0
                summaryValue = .double(val)
                decisionLabel = nil
            case .read:
                if let num = primary.number {
                    summaryValue = .double(num)
                } else {
                    summaryValue = .string(primary.date ?? primary.text ?? "")
                }
                decisionLabel = primary.text ?? primary.date
            case .box, .boxes:
                summaryValue = .string("\(primary.boxes.count) box(es)")
                decisionLabel = primary.boxes.isEmpty ? "none" : "detected"
            }
        } else {
            summaryValue = .null
            decisionLabel = nil
        }

        var keyed: [String: QWDecisionAnswer] = [:]
        for ans in envelope.response.answers { keyed[ans.name] = ans }

        self.answer = summaryValue
        self.decision = decisionLabel
        self.confidence = primary?.confidence ?? 0
        self.probabilities = probDict
        self.probs = probPairs
        self.answers = keyed
        self.usage = envelope.response.usage
        self.chain = envelope.response.chain
        self.model = envelope.modelName
        self.backend = envelope.backendKind
    }
}

public struct QWDecisionLoadHTTPRequest: Codable, Sendable {
    public let packPath: String?
    public let model: String?
    public let computeUnit: String?
    public let loadVision: Bool?

    enum CodingKeys: String, CodingKey {
        case model
        case packPath = "pack_path"
        case computeUnit = "compute_unit"
        case loadVision = "load_vision"
    }
}

public struct QWDecisionErrorPayload: Codable, Sendable {
    public struct Detail: Codable, Sendable {
        public let message: String
        public let type: String
        public let code: Int
        public let allowedLabels: [String]?

        enum CodingKeys: String, CodingKey {
            case message, type, code
            case allowedLabels = "allowed_labels"
        }
    }

    public let error: Detail
    public let allowedLabels: [String]?

    enum CodingKeys: String, CodingKey {
        case error
        case allowedLabels = "allowed_labels"
    }

    public init(message: String, type: String = "invalid_request_error", code: Int, allowedLabels: [String]? = nil) {
        self.error = Detail(message: message, type: type, code: code, allowedLabels: allowedLabels)
        self.allowedLabels = allowedLabels
    }
}
