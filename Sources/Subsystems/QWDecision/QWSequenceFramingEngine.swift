// QWSequenceFramingEngine.swift — Unified framing for verbalizer, option-marker, and sequence-pair models.
import Foundation
import QWCore
import QWToken

/// Escaped text holding U+200B breaks with exact UTF-8 byte mapping back to caller text.
public struct QWPlainEscapedText: Sendable, Hashable, Equatable {
    public let original: String
    public let escaped: String
    private let boundaryLo: [Int]
    private let boundaryHi: [Int]

    public init(original: String, escaped: String, boundaryLo: [Int], boundaryHi: [Int]) {
        self.original = original
        self.escaped = escaped
        self.boundaryLo = boundaryLo
        self.boundaryHi = boundaryHi
    }

    public func originalUTF8Range(forEscapedUTF8Range range: Range<Int>) -> Range<Int> {
        let maxEsc = max(0, boundaryLo.count - 1)
        let eLo = max(0, min(range.lowerBound, maxEsc))
        let eHi = max(eLo, min(range.upperBound, maxEsc))
        let oLo = boundaryLo[eLo]
        let oHi = max(oLo, boundaryHi[eHi])
        return oLo..<oHi
    }
}

public enum QWSlotBindingKind: Sendable, Equatable {
    case categorical(labels: [String])
    case noul
    case binaryOption(optionIndex: Int, key: String)
    case pairRelevance(optionIndex: Int, key: String, totalOptions: Int)
    case fixedLabelSubset(labels: [String], classIndices: [Int])
    case score(labels: [String])
    case readPointer(readKind: QWReadKind, stateTokenCount: Int)
    case estimateGrid(low: Double, high: Double, size: Int)
    case boxPresence(blockIndex: Int, isMany: Bool)
    case boxEdge(blockIndex: Int, edgeIndex: Int, isMany: Bool, gridSize: Int)
}

public struct QWBoundSlot: Sendable, Equatable {
    public let slotIndex: Int
    public let questionIndex: Int
    public let kind: QWSlotBindingKind

    public init(slotIndex: Int, questionIndex: Int, kind: QWSlotBindingKind) {
        self.slotIndex = slotIndex
        self.questionIndex = questionIndex
        self.kind = kind
    }
}

public struct QWBoundQuestion: Sendable, Equatable {
    public let questionIndex: Int
    public let question: QWDecisionQuestion
    public let slotBindings: [QWBoundSlot]
    public let stateTokenSpans: [QWTokenSpan]
    public let escapedState: QWPlainEscapedText
    public let pointerTokenSpans: [QWTokenSpan?]

    public init(
        questionIndex: Int,
        question: QWDecisionQuestion,
        slotBindings: [QWBoundSlot],
        stateTokenSpans: [QWTokenSpan],
        escapedState: QWPlainEscapedText,
        pointerTokenSpans: [QWTokenSpan?] = []
    ) {
        self.questionIndex = questionIndex
        self.question = question
        self.slotBindings = slotBindings
        self.stateTokenSpans = stateTokenSpans
        self.escapedState = escapedState
        self.pointerTokenSpans = pointerTokenSpans
    }
}

public struct QWBuiltDecisionBatch: Sendable, Equatable {
    public let request: QWDecisionBackendBatchRequest
    public let questionBindings: [QWBoundQuestion]
    public let truncatedTokens: Int
    public let promptTokens: Int
    public let defaultScoreReadout: QWScoreReadout
    public let boxEdgesMode: String
    public let usesNumbersThousandthsGrid: Bool
}

public enum QWSequenceFramingEngine {
    private static let zeroWidthBreak = Unicode.Scalar(0x200B)!

    private static let fallbackAliasPool: [String] = {
        var pool: [String] = []
        for c in 65...90 { pool.append(String(UnicodeScalar(c)!)) }
        for i in 0..<100 { pool.append(String(format: "%02d", i)) }
        for c in 97...122 { pool.append(String(UnicodeScalar(c)!)) }
        for i in 0..<200 { pool.append("#\(i)") }
        for a in 65...90 {
            for b in 65...90 {
                pool.append(String(UnicodeScalar(a)!) + String(UnicodeScalar(b)!))
            }
        }
        return pool
    }()

    /// Neutralizes any registered special token string in caller text with `U+200B`, keeping bijective UTF-8 bounds.
    public static func escapeSpecialTokens(_ text: String, specials: [String]) -> QWPlainEscapedText {
        let patterns = specials.filter { $0.unicodeScalars.count > 1 && !$0.contains("\u{200B}") }
        guard !patterns.isEmpty, !text.isEmpty else {
            let n = text.utf8.count
            return QWPlainEscapedText(
                original: text, escaped: text,
                boundaryLo: Array(0...n), boundaryHi: Array(0...n)
            )
        }
        struct TrackedScalar {
            let scalar: Unicode.Scalar
            let origLo: Int
            let origHi: Int
        }
        var items: [TrackedScalar] = []
        items.reserveCapacity(text.unicodeScalars.count + 8)
        var cursor = 0
        for sc in text.unicodeScalars {
            let w = sc.utf8.count
            items.append(TrackedScalar(scalar: sc, origLo: cursor, origHi: cursor + w))
            cursor += w
        }
        let patScalars = patterns.map { Array($0.unicodeScalars) }
        var changed = true
        while changed {
            changed = false
            var insertAt: [Int] = []
            var i = 0
            while i < items.count {
                var matchedLen = 0
                for ps in patScalars where i + ps.count <= items.count {
                    var ok = true
                    for k in 0..<ps.count where items[i + k].scalar != ps[k] {
                        ok = false; break
                    }
                    if ok { matchedLen = ps.count; break }
                }
                if matchedLen > 0 {
                    insertAt.append(i + 1)
                    changed = true
                    i += matchedLen
                } else {
                    i += 1
                }
            }
            for idx in insertAt.reversed() {
                let anchor = items[idx - 1].origHi
                items.insert(TrackedScalar(scalar: zeroWidthBreak, origLo: anchor, origHi: anchor), at: idx)
            }
        }
        var escapedStr = String.UnicodeScalarView()
        escapedStr.reserveCapacity(items.count)
        var loMap: [Int] = [0]
        var hiMap: [Int] = [0]
        for item in items {
            escapedStr.append(item.scalar)
            let w = item.scalar.utf8.count
            for k in 1...w {
                if k < w {
                    loMap.append(item.origLo)
                    hiMap.append(item.origHi)
                } else {
                    loMap.append(item.origHi)
                    hiMap.append(item.origHi)
                }
            }
        }
        return QWPlainEscapedText(
            original: text, escaped: String(escapedStr),
            boundaryLo: loMap, boundaryHi: hiMap
        )
    }

    /// Rewrites `<|name|>` to `<¦name¦>` so untrusted text cannot forge option markers or delimiters.
    public static func escapeMarkerTags(_ text: String) -> String {
        guard text.contains("<|"), let regex = try? NSRegularExpression(pattern: "<\\|([A-Za-z0-9_]+)\\|>") else {
            return text
        }
        let ns = text as NSString
        return regex.stringByReplacingMatches(
            in: text,
            range: NSRange(location: 0, length: ns.length),
            withTemplate: "<¦$1¦>"
        )
    }

    public static func optionCodes(labels: [String], scheme: Int = 1) -> [String] {
        let trimmed = labels.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        if !trimmed.isEmpty, trimmed.allSatisfy({ $0.count == 1 && $0.unicodeScalars.first?.properties.isAlphabetic == true }) {
            return trimmed
        }
        let letters = (65...90).map { String(UnicodeScalar($0)!) }
        if trimmed.count <= 26 {
            return Array(letters.prefix(trimmed.count))
        }
        if scheme == 2 {
            var all = letters
            for a in letters { for b in letters { all.append(a + b) } }
            return Array(all.prefix(trimmed.count))
        }
        return (0..<trimmed.count).map { String(format: "%02d", $0) }
    }

    /// Resolves single-token alias groups `(code, [id, optionalSpacePrefixedId])` for verbalizer questions.
    public static func resolveVerbalizerAliases(
        labels: [String],
        scheme: Int,
        tokenizer: QWTokenizer
    ) throws -> [(code: String, tokenIds: [Int])] {
        let codes = optionCodes(labels: labels, scheme: scheme)
        var used = Set<Int>()
        var primary: [(code: String, id: Int)] = []
        primary.reserveCapacity(codes.count)
        for code in codes {
            if let tid = tokenizer.singleTokenId(forText: code), !used.contains(tid) {
                used.insert(tid)
                primary.append((code, tid))
                continue
            }
            var found: (String, Int)?
            for candidate in fallbackAliasPool {
                if let tid = tokenizer.singleTokenId(forText: candidate), !used.contains(tid) {
                    used.insert(tid)
                    found = (candidate, tid)
                    break
                }
            }
            guard let chosen = found else {
                throw QWDecisionError.missingSingleTokenAlias(optionCount: labels.count)
            }
            primary.append(chosen)
        }
        return primary.map { entry in
            var forms = [entry.id]
            if let spaced = tokenizer.singleTokenId(forText: " " + entry.code), !used.contains(spaced) {
                forms.append(spaced)
            }
            return (entry.code, forms)
        }
    }

    public static func noulReadoutGroups(tokenizer: QWTokenizer) -> [[Int]] {
        let primaryNo = ["no", "No", "NO"].compactMap { tokenizer.singleTokenId(forText: $0) }
        let primaryYes = ["yes", "Yes", "YES"].compactMap { tokenizer.singleTokenId(forText: $0) }
        if !primaryNo.isEmpty && !primaryYes.isEmpty {
            return [primaryNo, primaryYes]
        }
        let noForms = ["no", "No", "NO", "false", "0"].compactMap { tokenizer.singleTokenId(forText: $0) }
        let yesForms = ["yes", "Yes", "YES", "true", "1"].compactMap { tokenizer.singleTokenId(forText: $0) }
        if !noForms.isEmpty && !yesForms.isEmpty {
            return [noForms, yesForms]
        }
        return [[0], [1]]
    }

    public static func scoreReadoutGroups(
        count: Int,
        tokenizer: QWTokenizer,
        includeSpaced: Bool = true
    ) throws -> [[Int]] {
        var groups: [[Int]] = []
        groups.reserveCapacity(count)
        var used = Set<Int>()
        for i in 0..<count {
            var forms: [Int] = []
            let candidates = includeSpaced ? [String(i), " " + String(i)] : [String(i)]
            for str in candidates {
                if let tid = tokenizer.singleTokenId(forText: str), !used.contains(tid) {
                    used.insert(tid)
                    forms.append(tid)
                }
            }
            if forms.isEmpty {
                let aliases = try resolveVerbalizerAliases(labels: (0..<count).map(String.init), scheme: 1, tokenizer: tokenizer)
                return aliases.map(\.tokenIds)
            }
            groups.append(forms)
        }
        return groups
    }

    public static func numberGridTokenGroups(
        low: Double,
        high: Double,
        size: Int,
        scale: Int,
        tokenizer: QWTokenizer
    ) -> [[Int]]? {
        guard size >= 2, low < high else { return nil }
        let step = (high - low) / Double(size - 1)
        var groups: [[Int]] = []
        groups.reserveCapacity(size)
        for i in 0..<size {
            let value = (i == size - 1) ? high : (low + Double(i) * step)
            let scaled = value * Double(scale)
            let n = Int(scaled.rounded())
            guard abs(Double(n) - scaled) <= 1e-6,
                  let primary = tokenizer.singleTokenId(forText: String(n)) else {
                return nil
            }
            var forms = [primary]
            if let spaced = tokenizer.singleTokenId(forText: " \(n)"), spaced != primary {
                forms.append(spaced)
            }
            groups.append(forms)
        }
        return groups
    }

    public static func gridTokenGroups(size: Int, scaleFactor: Int, tokenizer: QWTokenizer) throws -> [[Int]] {
        if scaleFactor > 1 {
            var numericGroups: [[Int]] = []
            numericGroups.reserveCapacity(size)
            var allSingle = true
            for i in 0..<size {
                guard let primary = tokenizer.singleTokenId(forText: String(i)) else {
                    allSingle = false
                    break
                }
                var forms = [primary]
                if let spaced = tokenizer.singleTokenId(forText: " \(i)"), spaced != primary {
                    forms.append(spaced)
                }
                numericGroups.append(forms)
            }
            if allSingle {
                return numericGroups
            }
        }
        var groups: [[Int]] = []
        groups.reserveCapacity(size)
        for i in 0..<size {
            let label = scaleFactor > 1 ? String(format: "%03d", i) : "c\(i)"
            if let tid = tokenizer.singleTokenId(forText: label) ?? tokenizer.singleTokenId(forText: String(i)) {
                groups.append([tid])
            } else {
                groups.append([i])
            }
        }
        return groups
    }

    public static func buildBatch(
        request: QWDecisionRequest,
        config: QWDecisionModelConfig,
        tokenizer: QWTokenizer,
        maxContextLength: Int
    ) throws -> QWBuiltDecisionBatch {
        let budgetLimit = max(16, min(request.contextBudget ?? maxContextLength, maxContextLength))
        let stateRaw = request.state.rendered(style: request.stateStyle)
        let defaultReadout = QWScoreReadout(rawValue: config.defaultScoreReadout) ?? .mean
        switch config.framing {
        case .turnVerbalizer(let codesPerOption, _, let packSet, let maxSetSize, _, let sep, let lead, let chatML):
            if chatML {
                return try buildChatMLTurnVerbalizerBatch(
                    request: request, stateRaw: stateRaw, config: config, tokenizer: tokenizer,
                    budgetLimit: budgetLimit, scheme: codesPerOption, packSet: packSet,
                    maxSetSize: maxSetSize, assistantLead: lead, defaultReadout: defaultReadout
                )
            }
            return try buildTurnVerbalizerBatch(
                request: request, stateRaw: stateRaw, config: config, tokenizer: tokenizer,
                budgetLimit: budgetLimit, scheme: codesPerOption, packSet: packSet, separator: sep,
                defaultReadout: defaultReadout
            )
        case .optionMarkers(let layout, let markerTok, let bosTok, let maxLen, _, _):
            return try buildOptionMarkersBatch(
                request: request, stateRaw: stateRaw, config: config, tokenizer: tokenizer,
                budgetLimit: min(budgetLimit, maxLen), layout: layout, markerToken: markerTok,
                bosToken: bosTok, defaultReadout: defaultReadout
            )
        case .sequencePair(let maxLen):
            return try buildSequencePairBatch(
                request: request, stateRaw: stateRaw, config: config, tokenizer: tokenizer,
                budgetLimit: min(budgetLimit, maxLen), defaultReadout: defaultReadout
            )
        }
    }

    internal static func buildTurnVerbalizerBatch(
        request: QWDecisionRequest,
        stateRaw: String,
        config: QWDecisionModelConfig,
        tokenizer: QWTokenizer,
        budgetLimit: Int,
        scheme: Int,
        packSet: Bool,
        separator: String,
        defaultReadout: QWScoreReadout
    ) throws -> QWBuiltDecisionBatch {
        let escaped = escapeSpecialTokens(stateRaw, specials: tokenizer.allSpecialTokenStrings())
        let stateEncoded = try tokenizer.encodeWithOffsets(escaped.escaped)
        let bosIds = resolveBosIds(tokenizer: tokenizer)

        var renderedQuestions: [(ids: [Int], groups: [[[Int]]], pointerOffsets: [Int], labels: [String], kinds: [QWSlotBindingKind])] = []
        for q in request.questions {
            let built = try renderVerbalizerQuestion(
                q: q, scheme: scheme, tokenizer: tokenizer,
                numbersThousandths: config.usesNumbersThousandthsGrid,
                stateTokenCount: stateEncoded.ids.count
            )
            renderedQuestions.append(built)
        }

        let policy = request.truncation ?? .error
        var truncatedTotal = 0
        var sequences: [QWDecisionSequenceInput] = []
        var slots: [QWDecisionSlotTarget] = []
        var bindings: [QWBoundQuestion] = []
        var uniqueTokensSet = Set<Int>()
        var totalPromptTokens = 0

        if packSet && !renderedQuestions.isEmpty {
            let qTotal = renderedQuestions.reduce(0) { $0 + $1.ids.count }
            let room = max(0, budgetLimit - bosIds.count - qTotal)
            let (keptIds, keptSpans, trimmed) = try applyTruncation(
                stateEncoded: stateEncoded, room: room, policy: policy,
                totalNeeded: bosIds.count + stateEncoded.ids.count + qTotal, budgetLimit: budgetLimit
            )
            truncatedTotal = trimmed
            var allIds = bosIds + keptIds
            let stateEndPos = allIds.count
            var qSpans: [Range<Int>] = []
            for (qIdx, rq) in renderedQuestions.enumerated() {
                let qStart = allIds.count
                allIds.append(contentsOf: rq.ids)
                let qEnd = allIds.count
                qSpans.append(qStart..<qEnd)
                let slotBindings = appendSlots(
                    seqIdx: 0, qStart: qStart, qEnd: qEnd, questionIdx: qIdx, rq: rq,
                    stateCount: keptIds.count, slots: &slots, uniqueTokens: &uniqueTokensSet
                )
                bindings.append(QWBoundQuestion(
                    questionIndex: qIdx, question: request.questions[qIdx], slotBindings: slotBindings,
                    stateTokenSpans: keptSpans, escapedState: escaped
                ))
            }
            sequences.append(QWDecisionSequenceInput(
                tokenIds: allIds, topology: config.topology, stateEndPosition: stateEndPos,
                questionSpans: qSpans, chain: request.chain, images: request.images
            ))
            totalPromptTokens = allIds.count
        } else {
            let sepIds = (try? tokenizer.encode(separator, addSpecialTokens: false)) ?? []
            for (qIdx, rq) in renderedQuestions.enumerated() {
                let room = max(0, budgetLimit - bosIds.count - sepIds.count - rq.ids.count)
                let totalNeeded = bosIds.count + stateEncoded.ids.count + sepIds.count + rq.ids.count
                let (keptIds, keptSpans, trimmed) = try applyTruncation(
                    stateEncoded: stateEncoded, room: room, policy: policy,
                    totalNeeded: totalNeeded, budgetLimit: budgetLimit
                )
                truncatedTotal = max(truncatedTotal, trimmed)
                var seqIds = bosIds + keptIds + sepIds
                let stateEndPos = seqIds.count
                let qStart = seqIds.count
                seqIds.append(contentsOf: rq.ids)
                let qEnd = seqIds.count
                let slotBindings = appendSlots(
                    seqIdx: qIdx, qStart: qStart, qEnd: qEnd, questionIdx: qIdx, rq: rq,
                    stateCount: keptIds.count, slots: &slots, uniqueTokens: &uniqueTokensSet
                )
                sequences.append(QWDecisionSequenceInput(
                    tokenIds: seqIds, topology: config.topology, stateEndPosition: stateEndPos,
                    questionSpans: [qStart..<qEnd], chain: rq.labels.first.flatMap { _ in request.questions[qIdx].chain } ?? request.chain,
                    images: request.images
                ))
                bindings.append(QWBoundQuestion(
                    questionIndex: qIdx, question: request.questions[qIdx], slotBindings: slotBindings,
                    stateTokenSpans: keptSpans, escapedState: escaped
                ))
                totalPromptTokens += seqIds.count
            }
        }

        return QWBuiltDecisionBatch(
            request: QWDecisionBackendBatchRequest(
                sequences: sequences, slots: slots, uniqueCandidateTokenIds: uniqueTokensSet.sorted()
            ),
            questionBindings: bindings,
            truncatedTokens: truncatedTotal,
            promptTokens: totalPromptTokens,
            defaultScoreReadout: defaultReadout,
            boxEdgesMode: config.boxEdgesMode,
            usesNumbersThousandthsGrid: config.usesNumbersThousandthsGrid
        )
    }

    private static func resolveBosIds(tokenizer: QWTokenizer) -> [Int] {
        if let id = tokenizer.tokenId(for: "<|startoftext|>") ?? tokenizer.tokenId(for: "<bos>") ?? tokenizer.tokenId(for: "<s>") {
            return [id]
        }
        return [1]
    }

    private static func renderVerbalizerQuestion(
        q: QWDecisionQuestion,
        scheme: Int,
        tokenizer: QWTokenizer,
        numbersThousandths: Bool,
        stateTokenCount: Int
    ) throws -> (ids: [Int], groups: [[[Int]]], pointerOffsets: [Int], labels: [String], kinds: [QWSlotBindingKind]) {
        let baseHeader = "QUESTION: \(q.instructions)\n"
        switch q.kind {
        case .noul:
            let prompt = baseHeader + "ANSWER:"
            let ids = try tokenizer.encode(prompt, addSpecialTokens: false)
            let g = noulReadoutGroups(tokenizer: tokenizer)
            return (ids, [g], [ids.count - 1], ["no", "yes"], [.noul])
        case .choice:
            let aliases = try resolveVerbalizerAliases(labels: q.criteria.map(\.key), scheme: scheme, tokenizer: tokenizer)
            var lines = baseHeader
            for (idx, c) in q.criteria.enumerated() {
                let desc = c.description ?? c.key
                lines += "\(aliases[idx].code): \(desc)\n"
            }
            lines += "ANSWER:"
            let ids = try tokenizer.encode(lines, addSpecialTokens: false)
            return (ids, [aliases.map(\.tokenIds)], [ids.count - 1], q.criteria.map(\.key), [.categorical(labels: q.criteria.map(\.key))])
        case .multiChoice:
            var text = baseHeader
            var ids: [Int] = []
            var groups: [[[Int]]] = []
            var offsets: [Int] = []
            var kinds: [QWSlotBindingKind] = []
            let g = noulReadoutGroups(tokenizer: tokenizer)
            for (idx, c) in q.criteria.enumerated() {
                let seg = "\(text)\(c.key) (\(c.description ?? c.key)):"
                text = "\n"
                ids.append(contentsOf: try tokenizer.encode(seg, addSpecialTokens: false))
                offsets.append(ids.count - 1)
                groups.append(g)
                kinds.append(.binaryOption(optionIndex: idx, key: c.key))
            }
            return (ids, groups, offsets, q.criteria.map(\.key), kinds)
        case .score:
            let labels = q.criteria.enumerated().map { String($0.offset) }
            let groups = try scoreReadoutGroups(count: labels.count, tokenizer: tokenizer)
            var lines = baseHeader
            for (idx, c) in q.criteria.enumerated() {
                lines += "\(idx): \(c.description ?? c.key)\n"
            }
            lines += "ANSWER:"
            let ids = try tokenizer.encode(lines, addSpecialTokens: false)
            return (ids, [groups], [ids.count - 1], labels, [.score(labels: labels)])
        case .read:
            let prompt = baseHeader + "SPAN:"
            let ids = try tokenizer.encode(prompt, addSpecialTokens: false)
            return (ids, [[]], [ids.count - 1], [q.name], [.readPointer(readKind: q.readKind, stateTokenCount: stateTokenCount)])
        case .estimate:
            let lo = q.estimateLow ?? 0.0
            let hi = q.estimateHigh ?? 1.0
            let size = q.estimateStep.map { max(2, min(128, Int(((hi - lo) / $0).rounded()) + 1)) } ?? 128
            let prompt = baseHeader + "ESTIMATE:"
            let ids = try tokenizer.encode(prompt, addSpecialTokens: false)
            let groups = try gridTokenGroups(size: size, scaleFactor: 1, tokenizer: tokenizer)
            return (ids, [groups], [ids.count - 1], [q.name], [.estimateGrid(low: lo, high: hi, size: size)])
        case .box, .boxes:
            let isMany = (q.kind == .boxes)
            let blocks = isMany ? min(64, max(1, q.maxBoxes ?? 16)) : 1
            let edgeSize = numbersThousandths ? 1000 : 128
            let scale = numbersThousandths ? 1000 : 1
            let presenceGroups = noulReadoutGroups(tokenizer: tokenizer)
            let edgeGroups = try gridTokenGroups(size: edgeSize, scaleFactor: scale, tokenizer: tokenizer)
            let prompt = baseHeader + "BOX:"
            let baseIds = try tokenizer.encode(prompt, addSpecialTokens: false)
            let dummyTok = baseIds.last ?? 0
            var ids = baseIds
            var groups: [[[Int]]] = []
            var offsets: [Int] = []
            var kinds: [QWSlotBindingKind] = []
            for blk in 0..<blocks {
                offsets.append(ids.count - 1)
                groups.append(presenceGroups)
                kinds.append(.boxPresence(blockIndex: blk, isMany: isMany))
                for edge in 0..<4 {
                    ids.append(dummyTok)
                    offsets.append(ids.count - 1)
                    groups.append(edgeGroups)
                    kinds.append(.boxEdge(blockIndex: blk, edgeIndex: edge, isMany: isMany, gridSize: edgeSize))
                }
                if blk + 1 < blocks { ids.append(dummyTok) }
            }
            return (ids, groups, offsets, [q.name], kinds)
        }
    }

    internal static func applyTruncation(
        stateEncoded: QWTokenizedSpanList,
        room: Int,
        policy: QWTruncationPolicy,
        totalNeeded: Int,
        budgetLimit: Int
    ) throws -> (ids: [Int], spans: [QWTokenSpan], truncated: Int) {
        let count = stateEncoded.ids.count
        guard totalNeeded > budgetLimit else {
            return (stateEncoded.ids, stateEncoded.spans, 0)
        }
        switch policy {
        case .error:
            throw QWDecisionError.contextOverflow(tokens: totalNeeded, limit: budgetLimit)
        case .head, .autoState:
            let keep = max(0, min(room, count))
            return (Array(stateEncoded.ids.prefix(keep)), Array(stateEncoded.spans.prefix(keep)), count - keep)
        case .tail:
            let keep = max(0, min(room, count))
            let drop = count - keep
            return (Array(stateEncoded.ids.dropFirst(drop)), Array(stateEncoded.spans.dropFirst(drop)), drop)
        case .middle:
            let keep = max(0, min(room, count))
            let headKeep = keep / 2
            let tailKeep = keep - headKeep
            let ids = Array(stateEncoded.ids.prefix(headKeep)) + Array(stateEncoded.ids.suffix(tailKeep))
            let spans = Array(stateEncoded.spans.prefix(headKeep)) + Array(stateEncoded.spans.suffix(tailKeep))
            return (ids, spans, count - keep)
        }
    }

    private static func appendSlots(
        seqIdx: Int,
        qStart: Int,
        qEnd: Int,
        questionIdx: Int,
        rq: (ids: [Int], groups: [[[Int]]], pointerOffsets: [Int], labels: [String], kinds: [QWSlotBindingKind]),
        stateCount: Int,
        slots: inout [QWDecisionSlotTarget],
        uniqueTokens: inout Set<Int>
    ) -> [QWBoundSlot] {
        var result: [QWBoundSlot] = []
        for idx in rq.kinds.indices {
            let slotIdx = slots.count
            let groups = idx < rq.groups.count ? rq.groups[idx] : []
            for g in groups { for tid in g { uniqueTokens.insert(tid) } }
            let pos = qStart + (idx < rq.pointerOffsets.count ? rq.pointerOffsets[idx] : max(0, rq.ids.count - 1))
            let isPtr = {
                if case .readPointer = rq.kinds[idx] { return true }
                return false
            }()
            slots.append(QWDecisionSlotTarget(
                sequenceIndex: seqIdx,
                tokenPosition: min(max(qStart, pos), max(qStart, qEnd - 1)),
                candidateTokenGroups: groups,
                isPointerSlot: isPtr,
                pointerEndPosition: isPtr ? min(max(qStart, qEnd - 1), qStart + rq.ids.count - 1) : nil
            ))
            let updatedKind: QWSlotBindingKind
            if case .readPointer(let rk, _) = rq.kinds[idx] {
                updatedKind = .readPointer(readKind: rk, stateTokenCount: stateCount)
            } else {
                updatedKind = rq.kinds[idx]
            }
            result.append(QWBoundSlot(slotIndex: slotIdx, questionIndex: questionIdx, kind: updatedKind))
        }
        return result
    }
}
