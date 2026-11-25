// QWServerSchema+Decisions.swift — Wire request DTOs & dialect normalization for decision APIs.

import Foundation
import QWCore
import QWDecision
import QWEngine
import QWToken

public struct QWDecisionHTTPAsset: Codable, Sendable {
    public let type: String?
    public let format: String?
    public let dataBase64: String?
    public let part: String?

    enum CodingKeys: String, CodingKey {
        case type, format, part
        case dataBase64 = "data_base64"
    }
}

public struct QWDecisionHTTPOptionItem: Codable, Sendable {
    public let value: String?
    public let label: String?
    public let description: String?
}

public struct QWDecisionQuestionWireMeta: Sendable {
    public let name: String
    public let incomingType: String
    public let scoreLegend: [String: QWJson]

    public init(name: String, incomingType: String, scoreLegend: [String: QWJson] = [:]) {
        self.name = name
        self.incomingType = incomingType
        self.scoreLegend = scoreLegend
    }
}

public struct QWDecisionRequestDefaults: Sendable {
    public let scoreReadout: QWScoreReadout
    public let truncation: QWTruncationPolicy?
    public let alwaysEmitBox: Bool

    public init(
        scoreReadout: QWScoreReadout = .mean,
        truncation: QWTruncationPolicy? = nil,
        alwaysEmitBox: Bool = false
    ) {
        self.scoreReadout = scoreReadout
        self.truncation = truncation
        self.alwaysEmitBox = alwaysEmitBox
    }
}

public struct QWDecisionPreparedHTTPInput: Sendable {
    public let request: QWDecisionRequest
    public let questionMeta: [QWDecisionQuestionWireMeta]
    public let isOpenAIDialect: Bool
    public let preferredModel: String?
}

/// Incoming question specification supporting `/v1/decisions`, `/v1/systemone`, and OpenAI dialects.
public struct QWDecisionHTTPQuestion: Codable, Sendable {
    public let name: String?
    public let kind: String?
    public let type: String?
    public let instructions: QWJson?
    public let question: String?
    public let prompt: String?
    public let criteria: QWJson?
    public let choices: [QWDecisionHTTPOptionItem]?
    public let levels: [QWDecisionHTTPOptionItem]?
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
    public let alwaysEmitBox: Bool?

    enum CodingKeys: String, CodingKey {
        case name, kind, type, instructions, question, prompt, criteria
        case choices, levels, options, labels, unit, chain
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
        case alwaysEmitBox = "always_emit_box"
    }

    public init(
        name: String? = nil,
        kind: String? = nil,
        type: String? = nil,
        instructions: String? = nil,
        question: String? = nil,
        prompt: String? = nil,
        criteria: QWJson? = nil,
        choices: [QWDecisionHTTPOptionItem]? = nil,
        levels: [QWDecisionHTTPOptionItem]? = nil,
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
        chain: String? = nil,
        alwaysEmitBox: Bool? = nil
    ) {
        self.name = name
        self.kind = kind
        self.type = type
        self.instructions = instructions.map { .string($0) }
        self.question = question
        self.prompt = prompt
        self.criteria = criteria
        self.choices = choices
        self.levels = levels
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
        self.alwaysEmitBox = alwaysEmitBox
    }

    public var usesOpenAIDialect: Bool {
        if choices != nil || levels != nil { return true }
        let raw = (type ?? kind ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return raw == "predicate"
    }

    public func toDecisionQuestion(
        fallbackName: String,
        defaultScoreReadout: QWScoreReadout,
        defaultAlwaysEmitBox: Bool = false,
        assetPlaceholders: [String] = []
    ) throws -> QWDecisionQuestion {
        try toDecisionQuestionWithMeta(
            fallbackName: fallbackName,
            defaultScoreReadout: defaultScoreReadout,
            defaultAlwaysEmitBox: defaultAlwaysEmitBox,
            assetPlaceholders: assetPlaceholders
        ).0
    }

    func toDecisionQuestionWithMeta(
        fallbackName: String,
        defaultScoreReadout: QWScoreReadout,
        defaultAlwaysEmitBox: Bool = false,
        assetPlaceholders: [String] = []
    ) throws -> (QWDecisionQuestion, QWDecisionQuestionWireMeta) {
        let resolvedName = (name?.trimmingCharacters(in: .whitespacesAndNewlines)).flatMap { $0.isEmpty ? nil : $0 } ?? fallbackName
        let (parsedCriteria, scoreLegend) = try parseCriteria(assetPlaceholders: assetPlaceholders)
        let rawKind = (kind ?? type ?? (parsedCriteria.isEmpty ? "noul" : "choice"))
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let (mappedKind, inferredReadKind) = try Self.mapKind(rawKind)
        let explicitRead = readKind.flatMap { QWReadKind(rawValue: $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()) }
        let resolvedReadout = try parseScoreReadout(scoreReadout) ?? defaultScoreReadout

        let rawInstText = Self.renderJSONText(instructions)
        let rawPrompt = rawInstText ?? question ?? prompt ?? ""
        let resolvedPrompt = QWDecisionHTTPRequest.stripAssetPlaceholders(rawPrompt, placeholders: assetPlaceholders)

        let q = QWDecisionQuestion(
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
            chain: chain,
            alwaysEmitBox: alwaysEmitBox ?? defaultAlwaysEmitBox
        )
        let meta = QWDecisionQuestionWireMeta(name: resolvedName, incomingType: rawKind, scoreLegend: scoreLegend)
        return (q, meta)
    }

    private static func renderJSONText(_ value: QWJson?) -> String? {
        guard let value else { return nil }
        switch value {
        case .undefined, .null: return nil
        case .string(let s): return s
        default: return value.tojson
        }
    }

    private func parseCriteria(assetPlaceholders: [String]) throws -> ([QWDecisionCriterion], [String: QWJson]) {
        func clean(_ s: String?) -> String? {
            guard let s else { return nil }
            let stripped = QWDecisionHTTPRequest.stripAssetPlaceholders(s, placeholders: assetPlaceholders)
            return stripped.isEmpty ? nil : stripped
        }

        if let choices, !choices.isEmpty {
            let crit = choices.compactMap { item -> QWDecisionCriterion? in
                guard let key = (item.value ?? item.label), !key.isEmpty else { return nil }
                return QWDecisionCriterion(key: key, description: clean(item.description))
            }
            return (crit, [:])
        }
        if let levels, !levels.isEmpty {
            var legend: [String: QWJson] = [:]
            let crit = levels.enumerated().map { idx, item -> QWDecisionCriterion in
                let key = item.label ?? "\(idx)"
                if let desc = item.description {
                    legend["\(idx)"] = .string(desc)
                } else {
                    legend["\(idx)"] = .string(key)
                }
                return QWDecisionCriterion(key: key, description: clean(item.description))
            }
            return (crit, legend)
        }
        if let criteria {
            switch criteria {
            case .object(let pairs):
                let crit = pairs.map { key, val -> QWDecisionCriterion in
                    let rawDesc: String? = switch val {
                    case .null, .undefined: nil
                    case .string(let s): s.isEmpty ? nil : s
                    default: val.tojson
                    }
                    return QWDecisionCriterion(key: key, description: clean(rawDesc))
                }
                return (crit, [:])
            case .array(let arr):
                var legend: [String: QWJson] = [:]
                var crit: [QWDecisionCriterion] = []
                for (idx, item) in arr.enumerated() {
                    switch item {
                    case .string(let s):
                        legend["\(idx)"] = .string(s)
                        crit.append(QWDecisionCriterion(key: clean(s) ?? s, description: clean(s)))
                    case .object:
                        if case .string(let k) = item.get("key") {
                            let d: String? = {
                                if case .string(let ds) = item.get("description") { return ds }
                                return nil
                            }()
                            legend["\(idx)"] = item
                            crit.append(QWDecisionCriterion(key: k, description: clean(d)))
                        } else {
                            let j = item.tojson
                            legend["\(idx)"] = item
                            crit.append(QWDecisionCriterion(key: j, description: clean(j)))
                        }
                    case .null, .undefined:
                        break
                    default:
                        legend["\(idx)"] = item
                        let s = item.stringified
                        crit.append(QWDecisionCriterion(key: s, description: clean(s)))
                    }
                }
                return (crit, legend)
            case .null, .undefined:
                break
            default:
                throw QWDecisionError.invalidQuestion("criteria must be a JSON object or array")
            }
        }
        if let list = options ?? labels {
            var legend: [String: QWJson] = [:]
            for (idx, s) in list.enumerated() { legend["\(idx)"] = .string(s) }
            return (list.map { QWDecisionCriterion(key: $0) }, legend)
        }
        return ([], [:])
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

/// Canonical decision request supporting `/v1/decisions`, `/v1/systemone`, and OpenAI Decisions.
public struct QWDecisionHTTPRequest: Codable, Sendable {
    public let model: String?
    public let state: QWJson?
    public let context: QWJson?
    public let input: QWJson?
    public let questions: QWJson
    public let assets: [String: QWDecisionHTTPAsset]?
    public let truncation: String?
    public let contextBudget: Int?
    public let systemPrompt: String?
    public let stateStyle: String?
    public let chain: String?
    public let scoreReadout: String?
    public let images: [String]?
    public let alwaysEmitBox: Bool?

    enum CodingKeys: String, CodingKey {
        case model, state, context, input, questions, assets, truncation, chain, images
        case contextBudget = "context_budget"
        case systemPrompt = "system_prompt"
        case stateStyle = "state_style"
        case scoreReadout = "score_readout"
        case alwaysEmitBox = "always_emit_box"
    }

    public func toDecisionRequest() throws -> QWDecisionRequest {
        try prepareExecution().request
    }

    public func prepareExecution(
        defaults: QWDecisionRequestDefaults = QWDecisionRequestDefaults(),
        multipartParts: [String: Data] = [:]
    ) throws -> QWDecisionPreparedHTTPInput {
        let defaultReadout: QWScoreReadout
        if let raw = scoreReadout?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
           !raw.isEmpty {
            guard let parsed = QWScoreReadout(rawValue: raw) else {
                throw QWDecisionError.invalidQuestion("unsupported score_readout '\(raw)'")
            }
            defaultReadout = parsed
        } else {
            defaultReadout = defaults.scoreReadout
        }

        let truncationPolicy: QWTruncationPolicy?
        if let rawTrunc = truncation?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
           !rawTrunc.isEmpty {
            guard let parsed = QWTruncationPolicy(rawValue: rawTrunc) else {
                throw QWDecisionError.invalidQuestion("unsupported truncation policy '\(rawTrunc)'")
            }
            truncationPolicy = parsed
        } else {
            truncationPolicy = defaults.truncation
        }

        var collectedImages = try Self.decodeHTTPImages(images)
        let rawState = input ?? state ?? context
        let orderedAssetKeys = try Self.resolveAssets(
            assets,
            rawState: rawState,
            rawQuestions: questions,
            multipartParts: multipartParts,
            into: &collectedImages
        )
        let defaultBoxEmit = alwaysEmitBox ?? defaults.alwaysEmitBox
        let (parsedQuestions, questionMeta, questionUsesOpenAI) = try Self.decodeQuestionsWithMeta(
            questions,
            defaultScoreReadout: defaultReadout,
            defaultAlwaysEmitBox: defaultBoxEmit,
            assetPlaceholders: orderedAssetKeys
        )
        let parsedState = try Self.decodeState(
            rawState,
            assetPlaceholders: orderedAssetKeys,
            imagesOut: &collectedImages
        )

        guard collectedImages.count <= QWImageDecodePolicy.defaultMaxImagesPerTurn else {
            throw QWDecisionError.invalidQuestion(
                "Too many images in decision request (\(collectedImages.count)); maximum is \(QWImageDecodePolicy.defaultMaxImagesPerTurn)"
            )
        }

        let domainReq = QWDecisionRequest(
            state: parsedState,
            questions: parsedQuestions,
            truncation: truncationPolicy,
            contextBudget: contextBudget,
            systemPrompt: systemPrompt,
            stateStyle: stateStyle ?? "json_only",
            chain: chain ?? "default",
            images: collectedImages
        )
        return QWDecisionPreparedHTTPInput(
            request: domainReq,
            questionMeta: questionMeta,
            isOpenAIDialect: (input != nil) || questionUsesOpenAI,
            preferredModel: model
        )
    }

    private static func resolveAssets(
        _ assets: [String: QWDecisionHTTPAsset]?,
        rawState: QWJson?,
        rawQuestions: QWJson,
        multipartParts: [String: Data],
        into imagesOut: inout [Data]
    ) throws -> [String] {
        guard let assets, !assets.isEmpty else { return [] }
        let searchBlob = (rawState?.tojson ?? "") + "\n" + rawQuestions.tojson
        let ordered = assets.sorted { lhs, rhs in
            if let li = partNumber(lhs.value.part), let ri = partNumber(rhs.value.part), li != ri {
                return li < ri
            }
            let lPos = searchBlob.range(of: lhs.key)?.lowerBound
            let rPos = searchBlob.range(of: rhs.key)?.lowerBound
            switch (lPos, rPos) {
            case let (a?, b?) where a != b: return a < b
            case (.some, .none): return true
            case (.none, .some): return false
            default: return lhs.key < rhs.key
            }
        }
        for (key, asset) in ordered {
            if let part = asset.part, !part.isEmpty {
                guard let partData = multipartParts[part], !partData.isEmpty else {
                    throw QWDecisionError.invalidQuestion("missing multipart part '\(part)' for asset '\(key)'")
                }
                imagesOut.append(partData)
            } else if let b64 = asset.dataBase64, !b64.isEmpty {
                let data: Data
                if b64.hasPrefix("data:") {
                    data = try resolveSingleImageURL(b64)
                } else if let decoded = Data(base64Encoded: b64, options: .ignoreUnknownCharacters), !decoded.isEmpty {
                    data = decoded
                } else {
                    throw QWDecisionError.invalidQuestion("invalid base64 data in asset '\(key)'")
                }
                imagesOut.append(data)
            } else {
                throw QWDecisionError.invalidQuestion("asset '\(key)' requires 'part' or 'data_base64'")
            }
        }
        return ordered.map(\.key)
    }

    private static func partNumber(_ part: String?) -> Int? {
        guard let part, let dash = part.lastIndex(of: "-") else { return nil }
        return Int(part[part.index(after: dash)...])
    }

    static func stripAssetPlaceholders(_ text: String, placeholders: [String]) -> String {
        guard !placeholders.isEmpty, !text.isEmpty else { return text }
        var out = text
        for key in placeholders where !key.isEmpty {
            out = out.replacingOccurrences(of: key, with: "")
        }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func decodeHTTPImages(_ raw: [String]?) throws -> [Data] {
        guard let raw, !raw.isEmpty else { return [] }
        guard raw.count <= QWImageDecodePolicy.defaultMaxImagesPerTurn else {
            throw QWDecisionError.invalidQuestion(
                "Too many images in decision request (\(raw.count)); maximum is \(QWImageDecodePolicy.defaultMaxImagesPerTurn)"
            )
        }
        return try raw.map(resolveSingleImageURL)
    }

    private static func resolveSingleImageURL(_ source: String) throws -> Data {
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

    static func decodeState(_ raw: QWJson?) throws -> QWDecisionState {
        var ignored: [Data] = []
        return try decodeState(raw, assetPlaceholders: [], imagesOut: &ignored)
    }

    static func decodeState(
        _ raw: QWJson?,
        assetPlaceholders: [String],
        imagesOut: inout [Data]
    ) throws -> QWDecisionState {
        guard let raw else { return .empty }
        switch raw {
        case .undefined, .null:
            return .empty
        case .string(let s):
            let cleaned = stripAssetPlaceholders(s, placeholders: assetPlaceholders)
            return (!assetPlaceholders.isEmpty && cleaned.isEmpty) ? .empty : .text(assetPlaceholders.isEmpty ? s : cleaned)
        case .array(let items):
            if let extracted = try extractMessagesState(items, assetPlaceholders: assetPlaceholders, imagesOut: &imagesOut) {
                return extracted
            }
            return try sanitizedJSONState(raw, assetPlaceholders: assetPlaceholders)
        case .object:
            if case .array(let messages) = raw.get("messages"),
               let extracted = try extractMessagesState(messages, assetPlaceholders: assetPlaceholders, imagesOut: &imagesOut) {
                return extracted
            }
            if case .string(let imgSource) = raw.get("image") {
                imagesOut.append(try resolveSingleImageURL(imgSource))
                if case .string(let txt) = raw.get("text") {
                    let cleaned = stripAssetPlaceholders(txt, placeholders: assetPlaceholders)
                    return cleaned.isEmpty ? .empty : .text(cleaned)
                }
                return .empty
            }
            if case .string(let kind) = raw.get("type") {
                switch kind.lowercased() {
                case "text":
                    if case .string(let txt) = raw.get("text") {
                        let cleaned = stripAssetPlaceholders(txt, placeholders: assetPlaceholders)
                        return (!assetPlaceholders.isEmpty && cleaned.isEmpty) ? .empty : .text(cleaned)
                    }
                case "json":
                    let payload = raw.get("json")
                    if !payload.isUndefined {
                        return try sanitizedJSONState(payload, assetPlaceholders: assetPlaceholders)
                    }
                default:
                    break
                }
            }
            return try sanitizedJSONState(raw, assetPlaceholders: assetPlaceholders)
        default:
            return .json(Data(raw.tojson.utf8))
        }
    }

    private static func sanitizedJSONState(_ json: QWJson, assetPlaceholders: [String]) throws -> QWDecisionState {
        if assetPlaceholders.isEmpty {
            return .json(Data(json.tojson.utf8))
        }
        let cleaned = stripPlaceholdersFromJSON(json, placeholders: assetPlaceholders)
        if isEffectivelyEmptyJSON(cleaned) { return .empty }
        return .json(Data(cleaned.tojson.utf8))
    }

    private static func stripPlaceholdersFromJSON(_ json: QWJson, placeholders: [String]) -> QWJson {
        switch json {
        case .string(let s):
            return .string(stripAssetPlaceholders(s, placeholders: placeholders))
        case .array(let arr):
            return .array(arr.map { stripPlaceholdersFromJSON($0, placeholders: placeholders) })
        case .object(let pairs):
            return .object(pairs.map { ($0.0, stripPlaceholdersFromJSON($0.1, placeholders: placeholders)) })
        default:
            return json
        }
    }

    private static func isEffectivelyEmptyJSON(_ json: QWJson) -> Bool {
        switch json {
        case .undefined, .null: return true
        case .string(let s): return s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        case .array(let arr): return arr.allSatisfy(isEffectivelyEmptyJSON)
        case .object(let pairs): return pairs.allSatisfy { isEffectivelyEmptyJSON($0.1) }
        default: return false
        }
    }

    private static func extractMessagesState(
        _ messages: [QWJson],
        assetPlaceholders: [String],
        imagesOut: inout [Data]
    ) throws -> QWDecisionState? {
        guard !messages.isEmpty,
              messages.allSatisfy({
                  if case .object = $0 { return $0.get("content") != .undefined }
                  return false
              }) else {
            return nil
        }
        var chunks: [String] = []
        for msg in messages {
            let content = msg.get("content")
            if case .string(let s) = content {
                let c = stripAssetPlaceholders(s, placeholders: assetPlaceholders)
                if !c.isEmpty { chunks.append(c) }
            } else if case .array(let parts) = content {
                for part in parts {
                    let pType: String = {
                        if case .string(let t) = part.get("type") { return t.lowercased() }
                        return ""
                    }()
                    if pType == "text" || pType == "input_text", case .string(let txt) = part.get("text") {
                        let c = stripAssetPlaceholders(txt, placeholders: assetPlaceholders)
                        if !c.isEmpty { chunks.append(c) }
                    } else if pType == "image_url" || pType == "input_image" {
                        let imgVal = part.get("image_url")
                        if case .string(let urlStr) = imgVal {
                            imagesOut.append(try resolveSingleImageURL(urlStr))
                        } else if case .object = imgVal, case .string(let urlStr) = imgVal.get("url") {
                            imagesOut.append(try resolveSingleImageURL(urlStr))
                        }
                    }
                }
            }
        }
        return chunks.isEmpty ? .empty : .text(chunks.joined(separator: "\n"))
    }

    static func decodeQuestions(
        _ raw: QWJson,
        defaultScoreReadout: QWScoreReadout
    ) throws -> [QWDecisionQuestion] {
        try decodeQuestionsWithMeta(
            raw,
            defaultScoreReadout: defaultScoreReadout,
            defaultAlwaysEmitBox: false,
            assetPlaceholders: []
        ).0
    }

    private static func decodeQuestionsWithMeta(
        _ raw: QWJson,
        defaultScoreReadout: QWScoreReadout,
        defaultAlwaysEmitBox: Bool,
        assetPlaceholders: [String]
    ) throws -> ([QWDecisionQuestion], [QWDecisionQuestionWireMeta], Bool) {
        switch raw {
        case .array(let items):
            var questions: [QWDecisionQuestion] = []
            var metas: [QWDecisionQuestionWireMeta] = []
            var anyOpenAI = false
            for (idx, item) in items.enumerated() {
                let dto = try QWServerRequestDecoder.decode(QWDecisionHTTPQuestion.self, from: Data(item.tojson.utf8))
                if dto.usesOpenAIDialect { anyOpenAI = true }
                let (q, m) = try dto.toDecisionQuestionWithMeta(
                    fallbackName: "q\(idx)",
                    defaultScoreReadout: defaultScoreReadout,
                    defaultAlwaysEmitBox: defaultAlwaysEmitBox,
                    assetPlaceholders: assetPlaceholders
                )
                questions.append(q)
                metas.append(m)
            }
            return (questions, metas, anyOpenAI)
        case .object(let pairs):
            var questions: [QWDecisionQuestion] = []
            var metas: [QWDecisionQuestionWireMeta] = []
            var anyOpenAI = false
            for (key, val) in pairs {
                let dto = try QWServerRequestDecoder.decode(QWDecisionHTTPQuestion.self, from: Data(val.tojson.utf8))
                if dto.usesOpenAIDialect { anyOpenAI = true }
                let (q, m) = try dto.toDecisionQuestionWithMeta(
                    fallbackName: key,
                    defaultScoreReadout: defaultScoreReadout,
                    defaultAlwaysEmitBox: defaultAlwaysEmitBox,
                    assetPlaceholders: assetPlaceholders
                )
                questions.append(q)
                metas.append(m)
            }
            return (questions, metas, anyOpenAI)
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
