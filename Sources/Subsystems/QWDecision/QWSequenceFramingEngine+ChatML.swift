// QWSequenceFramingEngine+ChatML.swift — Exact ChatML turn-verbalizer framing and StatePositions alignment.
import Foundation
import QWCore
import QWToken

extension QWSequenceFramingEngine {
    internal struct ChatMLRenderedQuestion {
        let suffix: String
        let groups: [[[Int]]]
        let relOffsets: [Int]
        let kinds: [QWSlotBindingKind]
        let isRead: Bool
        let readKind: QWReadKind
    }

    internal static func buildChatMLTurnVerbalizerBatch(
        request: QWDecisionRequest,
        stateRaw: String,
        config: QWDecisionModelConfig,
        tokenizer: QWTokenizer,
        budgetLimit: Int,
        scheme: Int,
        packSet: Bool,
        maxSetSize: Int,
        assistantLead: String,
        defaultReadout: QWScoreReadout
    ) throws -> QWBuiltDecisionBatch {
        let specials = tokenizer.allSpecialTokenStrings()
        let bosStr = resolveBosString(tokenizer: tokenizer)
        let systemTurn = resolveSystemTurn(request.systemPrompt)
        let mediaPrefix = buildChatMLMediaPrefix(imageCount: request.images.count, tokenizer: tokenizer)
        let isEmptyState: Bool = {
            if case .empty = request.state { return true }
            return false
        }()

        let stateBlockRaw = isEmptyState ? "" : (stateRaw + "\n\n")
        let escapedStateOnly = escapeSpecialTokens(isEmptyState ? "" : stateRaw, specials: specials)
        let escapedStateBlock = escapeSpecialTokens(stateBlockRaw, specials: specials)
        let questionHeader = isEmptyState ? "" : "QUESTION:\n"
        let headPrefix = "\(bosStr)\(systemTurn)<|im_start|>user\n\(mediaPrefix)"
        let fullPrefix = isEmptyState
            ? headPrefix
            : "\(headPrefix)\(escapedStateBlock.escaped)\n\(questionHeader)"

        var rendered: [ChatMLRenderedQuestion] = []
        rendered.reserveCapacity(request.questions.count)
        for q in request.questions {
            rendered.append(try renderChatMLQuestion(
                q: q, scheme: scheme, tokenizer: tokenizer, specials: specials,
                numbersThousandths: config.usesNumbersThousandthsGrid, assistantLead: assistantLead
            ))
        }

        let step = packSet ? max(1, maxSetSize) : 1
        let policy = request.truncation ?? .error
        var sequences: [QWDecisionSequenceInput] = []
        var slots: [QWDecisionSlotTarget] = []
        var bindings: [QWBoundQuestion] = []
        var uniqueTokensSet = Set<Int>()
        var totalPromptTokens = 0
        var truncatedTotal = 0

        if policy != .error && !isEmptyState {
            let maxQuestionIds = try rendered.enumerated().reduce(0) { acc, item in
                let hdr = (item.offset % step == 0) ? "" : questionHeader
                let c = try tokenizer.encode(hdr + item.element.suffix, addSpecialTokens: false).count
                return acc + c
            }
            let overhead = try tokenizer.encode("\(headPrefix)\n\n\n\(questionHeader)", addSpecialTokens: false).count
            if overhead + maxQuestionIds > budgetLimit {
                throw QWDecisionError.contextOverflow(tokens: overhead + maxQuestionIds, limit: budgetLimit)
            }
        }

        var groupStart = 0
        while groupStart < rendered.count {
            let groupEnd = min(rendered.count, groupStart + step)
            let chunk = Array(rendered[groupStart..<groupEnd])

            var workingStateRaw = stateRaw
            var workingPrefix = fullPrefix
            var workingEscapedState = escapedStateOnly
            let firstFull = workingPrefix + chunk[0].suffix
            var firstEncoded = try tokenizer.encodeWithOffsets(firstFull, addSpecialTokens: false)

            var tailIdsByMember: [[Int]] = [firstEncoded.ids]
            for k in 1..<chunk.count {
                tailIdsByMember.append(try tokenizer.encode(questionHeader + chunk[k].suffix, addSpecialTokens: false))
            }
            let totalNeeded = firstEncoded.ids.count + tailIdsByMember.dropFirst().reduce(0) { $0 + $1.count }

            if totalNeeded > budgetLimit {
                if policy == .error || isEmptyState {
                    throw QWDecisionError.contextOverflow(tokens: totalNeeded, limit: budgetLimit)
                }
                let nonStateCount = firstEncoded.ids.count
                    - (try tokenizer.encode(escapedStateOnly.escaped, addSpecialTokens: false)).count
                    + tailIdsByMember.dropFirst().reduce(0) { $0 + $1.count }
                let room = max(0, budgetLimit - nonStateCount)
                let origEncoded = try tokenizer.encodeWithOffsets(escapedStateOnly.escaped, addSpecialTokens: false)
                let (keptIds, _, trimmed) = try applyTruncation(
                    stateEncoded: origEncoded, room: room, policy: policy,
                    totalNeeded: totalNeeded, budgetLimit: budgetLimit
                )
                truncatedTotal = max(truncatedTotal, trimmed)
                workingStateRaw = (try? tokenizer.decode(keptIds, skipSpecialTokens: false)) ?? ""
                workingEscapedState = escapeSpecialTokens(workingStateRaw, specials: specials)
                let rebuiltBlock = escapeSpecialTokens(workingStateRaw + "\n\n", specials: specials)
                workingPrefix = "\(headPrefix)\(rebuiltBlock.escaped)\n\(questionHeader)"
                firstEncoded = try tokenizer.encodeWithOffsets(workingPrefix + chunk[0].suffix, addSpecialTokens: false)
                tailIdsByMember[0] = firstEncoded.ids
            }

            let suffix0Ids = try tokenizer.encode(chunk[0].suffix, addSpecialTokens: false)
            let stateEndPos = max(0, firstEncoded.ids.count - suffix0Ids.count)
            let ptrAlign = alignStatePositions(
                isEmptyState: isEmptyState,
                headPrefixUTF8Count: headPrefix.utf8.count,
                escapedState: workingEscapedState,
                firstFull: workingPrefix + chunk[0].suffix,
                firstEncoded: firstEncoded
            )

            var allIds = firstEncoded.ids
            var qSpans: [Range<Int>] = [stateEndPos..<firstEncoded.ids.count]
            var seqQuestionRanges: [Range<Int>] = [stateEndPos..<firstEncoded.ids.count]
            for k in 1..<chunk.count {
                let qStart = allIds.count
                allIds.append(contentsOf: tailIdsByMember[k])
                let qEnd = allIds.count
                qSpans.append(qStart..<qEnd)
                seqQuestionRanges.append(qStart..<qEnd)
            }

            let seqIdx = sequences.count
            let seqChain = (chunk.count == 1 ? request.questions[groupStart].chain : nil) ?? request.chain
            sequences.append(QWDecisionSequenceInput(
                tokenIds: allIds,
                topology: config.topology,
                stateEndPosition: stateEndPos,
                questionSpans: qSpans,
                chain: seqChain,
                images: request.images,
                pointerValidMask: ptrAlign.validMask
            ))
            totalPromptTokens += allIds.count

            for (offsetInChunk, rq) in chunk.enumerated() {
                let globalQIdx = groupStart + offsetInChunk
                let qEnd = seqQuestionRanges[offsetInChunk].upperBound
                let slotBindings = appendChatMLSlots(
                    seqIdx: seqIdx, qEnd: qEnd, questionIdx: globalQIdx,
                    rq: rq, stateTokenCount: ptrAlign.validTokenSpans.compactMap({ $0 }).count,
                    slots: &slots, uniqueTokens: &uniqueTokensSet
                )
                bindings.append(QWBoundQuestion(
                    questionIndex: globalQIdx,
                    question: request.questions[globalQIdx],
                    slotBindings: slotBindings,
                    stateTokenSpans: ptrAlign.validTokenSpans.compactMap { $0 },
                    escapedState: workingEscapedState,
                    pointerTokenSpans: rq.isRead ? ptrAlign.validTokenSpans : []
                ))
            }
            groupStart = groupEnd
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

    private static func resolveBosString(tokenizer: QWTokenizer) -> String {
        for candidate in ["<|startoftext|>", "<bos>", "<s>"] {
            if tokenizer.tokenId(for: candidate) != nil { return candidate }
        }
        return ""
    }

    private static func resolveSystemTurn(_ systemPrompt: String?) -> String {
        guard let sp = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines),
              !sp.isEmpty, sp.lowercased() != "none" else {
            return ""
        }
        let body = (sp.lowercased() == "d1") ? "You are d1, a decision model trained by Liquid AI." : sp
        return "<|im_start|>system\n\(body)<|im_end|>\n"
    }

    private static func buildChatMLMediaPrefix(imageCount: Int, tokenizer: QWTokenizer) -> String {
        guard imageCount > 0, tokenizer.tokenId(for: "<|image|>") != nil else { return "" }
        if tokenizer.tokenId(for: "<|vision_start|>") != nil && tokenizer.tokenId(for: "<|vision_end|>") != nil {
            return String(repeating: "<|vision_start|><|image|><|vision_end|>", count: imageCount)
        }
        return String(repeating: "<|image|>", count: imageCount)
    }

    private static func alignStatePositions(
        isEmptyState: Bool,
        headPrefixUTF8Count: Int,
        escapedState: QWPlainEscapedText,
        firstFull: String,
        firstEncoded: QWTokenizedSpanList
    ) -> (validTokenSpans: [QWTokenSpan?], validMask: [Bool]) {
        guard !firstEncoded.spans.isEmpty else {
            return ([nil], [true])
        }
        if isEmptyState {
            return ([nil], [true])
        }
        let fullBytes = Array(firstFull.utf8)
        let stateStart = headPrefixUTF8Count
        let stateStop = stateStart + escapedState.escaped.utf8.count
        var spansOut: [QWTokenSpan?] = []
        var maskOut: [Bool] = []
        spansOut.reserveCapacity(firstEncoded.spans.count)
        maskOut.reserveCapacity(firstEncoded.spans.count)
        for (tokIdx, tokSpan) in firstEncoded.spans.enumerated() {
            if tokIdx == 0 {
                spansOut.append(nil)
                maskOut.append(true)
                continue
            }
            let a = tokSpan.utf8Range.lowerBound
            let b = tokSpan.utf8Range.upperBound
            let lo = max(a, stateStart)
            let hi = min(b, stateStop)
            var matched: QWTokenSpan? = nil
            if lo < hi && lo <= b && a <= lo && hi <= b && b <= fullBytes.count {
                let prefixSlice = String(decoding: fullBytes[a..<lo], as: UTF8.self)
                let suffixSlice = String(decoding: fullBytes[hi..<b], as: UTF8.self)
                let innerSlice = String(decoding: fullBytes[lo..<hi], as: UTF8.self)
                if prefixSlice.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                   suffixSlice.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                   !innerSlice.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    let escRange = (lo - stateStart)..<(hi - stateStart)
                    matched = QWTokenSpan(
                        tokenId: tokSpan.tokenId,
                        utf8Range: escRange,
                        utf16Range: escRange,
                        scalarRange: escRange
                    )
                }
            }
            spansOut.append(matched)
            maskOut.append(matched != nil)
        }
        let lastValid = maskOut.lastIndex(of: true) ?? 0
        return (Array(spansOut[0...lastValid]), Array(maskOut[0...lastValid]))
    }

    private static func renderChatMLQuestion(
        q: QWDecisionQuestion,
        scheme: Int,
        tokenizer: QWTokenizer,
        specials: [String],
        numbersThousandths: Bool,
        assistantLead: String
    ) throws -> ChatMLRenderedQuestion {
        let said = escapeSpecialTokens(q.instructions, specials: specials).escaped
        switch q.kind {
        case .noul:
            var extra = ""
            if !q.criteria.isEmpty {
                let y = q.criteria.first(where: { $0.key.lowercased() == "true" || $0.key.lowercased() == "yes" })?.description ?? ""
                let n = q.criteria.first(where: { $0.key.lowercased() == "false" || $0.key.lowercased() == "no" })?.description ?? ""
                if !y.isEmpty || !n.isEmpty { extra = "\nYes: \(y)\nNo: \(n)" }
            }
            let body = "\(said)\(extra)\n\nReply with yes or no only."
            let text = "\(body)<|im_end|>\n<|im_start|>assistant\n\(assistantLead)"
            return ChatMLRenderedQuestion(
                suffix: text, groups: [noulReadoutGroups(tokenizer: tokenizer)],
                relOffsets: [0], kinds: [.noul], isRead: false, readKind: .text
            )
        case .choice:
            let aliases = try resolveVerbalizerAliases(labels: q.criteria.map(\.key), scheme: scheme, tokenizer: tokenizer)
            var lines: [String] = []
            lines.reserveCapacity(q.criteria.count)
            for (idx, c) in q.criteria.enumerated() {
                let desc = escapeSpecialTokens(c.description ?? c.key.replacingOccurrences(of: "_", with: " "), specials: specials).escaped
                lines.append("\(aliases[idx].code) \(desc)")
            }
            let body = "\(said)\n\nOptions:\n\(lines.joined(separator: "\n"))\n\nReply with the option code only."
            let text = "\(body)<|im_end|>\n<|im_start|>assistant\n\(assistantLead)"
            return ChatMLRenderedQuestion(
                suffix: text, groups: [aliases.map(\.tokenIds)],
                relOffsets: [0], kinds: [.categorical(labels: q.criteria.map(\.key))], isRead: false, readKind: .text
            )
        case .multiChoice:
            let lines = q.criteria.map { c -> String in
                let nameJSON = jsonEncodedString(c.key)
                let desc = escapeSpecialTokens(c.description ?? c.key, specials: specials).escaped
                return "\(nameJSON): \(desc)"
            }.joined(separator: "\n")
            let body = "\(said)\n\nOptions:\n\(lines)\n\nReply with yes or no for each option."
            let names = q.criteria.map { jsonEncodedString($0.key) }
            let (text, relOffsets) = try renderMarkedSuffix(
                escapedBody: body, markerNames: names, lead: assistantLead,
                after: "", specials: specials, tokenizer: tokenizer
            )
            let yesNo = noulReadoutGroups(tokenizer: tokenizer)
            let kinds = q.criteria.enumerated().map { QWSlotBindingKind.binaryOption(optionIndex: $0.offset, key: $0.element.key) }
            return ChatMLRenderedQuestion(
                suffix: text, groups: [[ [Int] ]](repeating: yesNo, count: names.count),
                relOffsets: relOffsets, kinds: kinds, isRead: false, readKind: .text
            )
        case .score:
            let labels = q.criteria.enumerated().map { String($0.offset) }
            let groups = try scoreReadoutGroups(count: labels.count, tokenizer: tokenizer, includeSpaced: false)
            let legend = q.criteria.enumerated().map { idx, c in
                let name = escapeSpecialTokens(c.description ?? c.key, specials: specials).escaped
                return "\(idx) \(name)"
            }.joined(separator: "\n")
            let body = "\(said)\n\n\(legend)\n\nReply with a single digit 0-\(labels.count - 1) only."
            let text = "\(body)<|im_end|>\n<|im_start|>assistant\n\(assistantLead)"
            return ChatMLRenderedQuestion(
                suffix: text, groups: [groups], relOffsets: [0],
                kinds: [.score(labels: labels)], isRead: false, readKind: .text
            )
        case .read:
            let unitNote = (q.readKind == .number && !(q.unit ?? "").isEmpty) ? " (unit: \(q.unit!))" : ""
            let body = "\(said)\n\nRead the \(q.readKind.rawValue)\(unitNote) from the state, or nothing if it is absent."
            let (text, relOffsets) = try renderMarkedSuffix(
                escapedBody: body, markerNames: ["start", "end"], lead: assistantLead,
                after: "", specials: specials, tokenizer: tokenizer
            )
            return ChatMLRenderedQuestion(
                suffix: text, groups: [[], []], relOffsets: relOffsets,
                kinds: [.readPointer(readKind: q.readKind, stateTokenCount: 0)], isRead: true, readKind: q.readKind
            )
        case .estimate:
            let lo = q.estimateLow ?? 0.0
            let hi = q.estimateHigh ?? 1.0
            let size = q.estimateStep.map { max(2, min(128, Int(((hi - lo) / $0).rounded()) + 1)) } ?? 128
            let step = (hi - lo) / Double(max(1, size - 1))
            let unitSpaced = q.unit.map { " \($0)" } ?? ""
            let groups: [[[Int]]]
            let body: String
            if numbersThousandths, let numGroups = numberGridTokenGroups(low: lo, high: hi, size: size, scale: 1, tokenizer: tokenizer) {
                groups = [numGroups]
                body = "\(said)\n\nEstimate from \(formatG(lo)) to \(formatG(hi))\(unitSpaced), in steps of \(formatG(step)). Reply with the number only."
            } else {
                let firstCode: String
                let lastCode: String
                if size <= 10 {
                    let scoreG = try scoreReadoutGroups(count: size, tokenizer: tokenizer, includeSpaced: false)
                    groups = [scoreG]
                    firstCode = "0"
                    lastCode = "\(size - 1)"
                } else {
                    let aliases = try resolveVerbalizerAliases(labels: (0..<size).map(String.init), scheme: scheme, tokenizer: tokenizer)
                    groups = [aliases.map(\.tokenIds)]
                    firstCode = aliases.first?.code ?? "A"
                    lastCode = aliases.last?.code ?? "A"
                }
                body = "\(said)\n\nEstimate from \(formatG(lo)) to \(formatG(hi))\(unitSpaced) on \(size) equally spaced values, step \(formatG(step)): \(firstCode) is \(formatG(lo)), \(lastCode) is \(formatG(hi)). Reply with the code only."
            }
            let text = "\(body)<|im_end|>\n<|im_start|>assistant\n\(assistantLead)"
            return ChatMLRenderedQuestion(
                suffix: text, groups: groups, relOffsets: [0],
                kinds: [.estimateGrid(low: lo, high: hi, size: size)], isRead: false, readKind: .text
            )
        case .box, .boxes:
            return try renderChatMLBoxQuestion(
                q: q, said: said, scheme: scheme, numbersThousandths: numbersThousandths,
                assistantLead: assistantLead, specials: specials, tokenizer: tokenizer
            )
        }
    }

    private static func renderChatMLBoxQuestion(
        q: QWDecisionQuestion,
        said: String,
        scheme: Int,
        numbersThousandths: Bool,
        assistantLead: String,
        specials: [String],
        tokenizer: QWTokenizer
    ) throws -> ChatMLRenderedQuestion {
        let isMany = (q.kind == .boxes)
        let blocks = isMany ? min(16, max(1, q.maxBoxes ?? 16)) : 1
        let useNumbers = numbersThousandths && numberGridTokenGroups(low: 0.0, high: 0.999, size: 1000, scale: 1000, tokenizer: tokenizer) != nil
        let edgeSize = useNumbers ? 1000 : 128
        let edgeGroups: [[Int]]
        let edgeLegend: String
        if useNumbers {
            edgeGroups = try gridTokenGroups(size: 1000, scaleFactor: 1000, tokenizer: tokenizer)
            edgeLegend = "Each edge is a number from 0 to 999, in thousandths of the picture: top and bottom from its top edge, left and right from its left edge."
        } else {
            let aliases = try resolveVerbalizerAliases(labels: (0..<128).map(String.init), scheme: scheme, tokenizer: tokenizer)
            edgeGroups = aliases.map(\.tokenIds)
            edgeLegend = "Each edge is a code on a 128-value scale: \(aliases.first?.code ?? "A") is 0 (top or left), \(aliases.last?.code ?? "A") is 1 (bottom or right). Values are equally spaced."
        }
        let body: String
        let markerNames: [String]
        let after: String
        let edges = ["top", "bottom", "left", "right"]
        if isMany {
            body = "\(said)\n\nEvery one of them, 16 at most, in reading order (left edge, then top). Present k is yes when there are at least k. \(edgeLegend) Reply with yes or no at each present, then its top, bottom, left and right."
            markerNames = (1...blocks).flatMap { k in ["present \(k)"] + edges.map { "\($0) \(k)" } }
            after = blocks < 16 ? "\n" : ""
        } else {
            let one = useNumbers ? "one number" : "one code"
            body = "\(said)\n\nPresence is yes or no. \(edgeLegend) Reply with yes or no at present, then \(one) at top, bottom, left and right."
            markerNames = ["present"] + edges
            after = ""
        }
        let (text, relOffsets) = try renderMarkedSuffix(
            escapedBody: body, markerNames: markerNames, lead: assistantLead,
            after: after, specials: specials, tokenizer: tokenizer
        )
        let presenceG = noulReadoutGroups(tokenizer: tokenizer)
        var groups: [[[Int]]] = []
        var kinds: [QWSlotBindingKind] = []
        groups.reserveCapacity(markerNames.count)
        kinds.reserveCapacity(markerNames.count)
        for blk in 0..<blocks {
            groups.append(presenceG)
            kinds.append(.boxPresence(blockIndex: blk, isMany: isMany))
            for edge in 0..<4 {
                groups.append(edgeGroups)
                kinds.append(.boxEdge(blockIndex: blk, edgeIndex: edge, isMany: isMany, gridSize: edgeSize))
            }
        }
        return ChatMLRenderedQuestion(
            suffix: text, groups: groups, relOffsets: relOffsets,
            kinds: kinds, isRead: false, readKind: .text
        )
    }

    private static func renderMarkedSuffix(
        escapedBody: String,
        markerNames: [String],
        lead: String,
        after: String,
        specials: [String],
        tokenizer: QWTokenizer
    ) throws -> (text: String, relOffsets: [Int]) {
        var text = "\(escapedBody)<|im_end|>\n<|im_start|>assistant\n\(lead)"
        var endBytes: [Int] = []
        endBytes.reserveCapacity(markerNames.count)
        for (i, name) in markerNames.enumerated() {
            let escapedName = escapeSpecialTokens(name, specials: specials).escaped
            text += (i == 0 ? "" : "\n") + escapedName + ":"
            endBytes.append(text.utf8.count)
        }
        if !after.isEmpty,
           let afterIds = try? tokenizer.encode(text + after, addSpecialTokens: false),
           let lastId = afterIds.last,
           let decoded = try? tokenizer.decode([lastId], skipSpecialTokens: false),
           decoded.replacingOccurrences(of: "\\s+$", with: "", options: .regularExpression).hasSuffix(":") {
            text += after
        }
        let encoded = try tokenizer.encodeWithOffsets(text, addSpecialTokens: false)
        let textBytes = Array(text.utf8)
        let len = encoded.ids.count
        var offsets: [Int] = []
        offsets.reserveCapacity(endBytes.count)
        for chars in endBytes {
            let matches = encoded.spans.enumerated().filter { _, span in
                span.utf8Range.lowerBound <= chars - 1 && (chars - 1) < span.utf8Range.upperBound
            }
            guard matches.count == 1 else {
                throw QWDecisionError.invalidQuestion("slot marker must have an exact colon token position")
            }
            let (tokIdx, span) = matches[0]
            let tail = chars < span.utf8Range.upperBound && span.utf8Range.upperBound <= textBytes.count
                ? String(decoding: textBytes[chars..<span.utf8Range.upperBound], as: UTF8.self)
                : ""
            let tokStr = (try? tokenizer.decode([encoded.ids[tokIdx]], skipSpecialTokens: false)) ?? ""
            let tailTrimmed = tail.trimmingCharacters(in: .whitespacesAndNewlines)
            let tokTrimmed = tokStr.replacingOccurrences(of: "\\s+$", with: "", options: .regularExpression)
            guard tailTrimmed.isEmpty && tokTrimmed.hasSuffix(":") else {
                throw QWDecisionError.invalidQuestion("slot marker token must end with colon and optional whitespace")
            }
            offsets.append(tokIdx - len + 1)
        }
        return (text, offsets)
    }

    private static func appendChatMLSlots(
        seqIdx: Int,
        qEnd: Int,
        questionIdx: Int,
        rq: ChatMLRenderedQuestion,
        stateTokenCount: Int,
        slots: inout [QWDecisionSlotTarget],
        uniqueTokens: inout Set<Int>
    ) -> [QWBoundSlot] {
        let lastPos = max(0, qEnd - 1)
        if rq.isRead {
            let slotIdx = slots.count
            let startOffset = rq.relOffsets.first ?? 0
            let endOffset = rq.relOffsets.count > 1 ? rq.relOffsets[1] : startOffset
            let startPos = max(0, min(lastPos, lastPos + startOffset))
            let endPos = max(0, min(lastPos, lastPos + endOffset))
            slots.append(QWDecisionSlotTarget(
                sequenceIndex: seqIdx,
                tokenPosition: startPos,
                candidateTokenGroups: [],
                isPointerSlot: true,
                pointerEndPosition: endPos
            ))
            let kind: QWSlotBindingKind = .readPointer(readKind: rq.readKind, stateTokenCount: stateTokenCount)
            return [QWBoundSlot(slotIndex: slotIdx, questionIndex: questionIdx, kind: kind)]
        }
        var out: [QWBoundSlot] = []
        out.reserveCapacity(rq.kinds.count)
        for idx in rq.kinds.indices {
            let slotIdx = slots.count
            let groups = idx < rq.groups.count ? rq.groups[idx] : []
            for g in groups { for tid in g { uniqueTokens.insert(tid) } }
            let rel = idx < rq.relOffsets.count ? rq.relOffsets[idx] : 0
            let pos = max(0, min(lastPos, lastPos + rel))
            slots.append(QWDecisionSlotTarget(
                sequenceIndex: seqIdx, tokenPosition: pos, candidateTokenGroups: groups,
                isPointerSlot: false, pointerEndPosition: nil
            ))
            out.append(QWBoundSlot(slotIndex: slotIdx, questionIndex: questionIdx, kind: rq.kinds[idx]))
        }
        return out
    }

    private static func jsonEncodedString(_ raw: String) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: [raw], options: []),
              let arr = String(data: data, encoding: .utf8),
              arr.hasPrefix("[") && arr.hasSuffix("]") else {
            return "\"\(raw)\""
        }
        return String(arr.dropFirst().dropLast())
    }

    private static func formatG(_ x: Double) -> String {
        String(format: "%g", x)
    }
}
