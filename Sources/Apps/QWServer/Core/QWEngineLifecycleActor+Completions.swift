// QWEngineLifecycleActor+Completions.swift — Execution helpers for
// POST /v1/completions (raw prompts, echo logprobs) and POST /v1/messages/count_tokens.

import Foundation
import QWCli
import QWCore
import QWEngine
import QWHead
import QWToken
import QWVision

private final class QWSingleTokenConstraint: QWTokenConstraint, @unchecked Sendable {
    private let tokenId: Int
    private let vocabularySize: Int

    init(tokenId: Int, vocabularySize: Int) {
        self.tokenId = tokenId
        self.vocabularySize = vocabularySize
    }

    var allowsSpeculation: Bool { false }
    var isUnconstrainedNow: Bool { false }
    var speculationBoundaryToken: Int? { nil }

    func allowedTokens() throws -> [Int]? {
        guard tokenId >= 0, tokenId < vocabularySize else {
            throw QWTokenConstraintError.emptySupport
        }
        return [tokenId]
    }

    func accept(_ token: Int) throws {}
}

private typealias CompletionLPEntry = (tok: String, lp: Float?, top: [String: Float]?)

private final class RawCompletionCapture: @unchecked Sendable {
    private let lock = NSLock()
    let tok: QWTokenizer
    let stopScanner: QWEngineLifecycleActor.IncrementalStopScanner
    var decoder: QWStreamDecoder
    var fullUTF8: [UInt8] = []
    var emittedUTF8 = 0
    var stoppedByCustomRule = false
    var pendingLPEntries: [CompletionLPEntry] = []
    var allCompletionLPEntries: [CompletionLPEntry] = []
    var streamOffset: Int
    let choiceIdx: Int
    let onChoiceToken: (@Sendable (Int, String, CompletionLogprobs?) -> Void)?
    let shouldStop: (@Sendable () -> Bool)?

    init(
        tok: QWTokenizer, customStops: [String], streamOffset: Int, choiceIdx: Int,
        onChoiceToken: (@Sendable (Int, String, CompletionLogprobs?) -> Void)?,
        shouldStop: (@Sendable () -> Bool)?
    ) {
        self.tok = tok
        self.stopScanner = QWEngineLifecycleActor.IncrementalStopScanner(stops: customStops)
        self.decoder = QWStreamDecoder(tok)
        self.streamOffset = streamOffset
        self.choiceIdx = choiceIdx
        self.onChoiceToken = onChoiceToken
        self.shouldStop = shouldStop
    }

    func appendLogprob(_ entry: CompletionLPEntry, trackPending: Bool) {
        lock.withLock {
            allCompletionLPEntries.append(entry)
            if trackPending {
                pendingLPEntries.append(entry)
            }
        }
    }

    func processToken(_ id: Int) -> Bool {
        if shouldStop?() == true || Task.isCancelled {
            lock.withLock { stoppedByCustomRule = true }
            return false
        }
        let result: (shouldContinue: Bool, emitted: (String, CompletionLogprobs?)?) = lock.withLock {
            guard let piece = try? decoder.decode(id, skipSpecialTokens: true), !piece.isEmpty else {
                return (true, nil)
            }
            let prevCount = fullUTF8.count
            fullUTF8.append(contentsOf: piece.utf8)
            if !stopScanner.isEmpty {
                if let match = stopScanner.firstMatch(in: fullUTF8, previousCount: prevCount) {
                    stoppedByCustomRule = true
                    fullUTF8.removeSubrange(match.byteOffset..<fullUTF8.count)
                    return (false, takeEmissionLocked(upTo: fullUTF8.count))
                }
                let hold = stopScanner.heldSuffixUTF8Count(in: fullUTF8)
                return (true, takeEmissionLocked(upTo: max(0, fullUTF8.count - hold)))
            }
            return (true, takeEmissionLocked(upTo: fullUTF8.count))
        }
        if let (str, lpChunk) = result.emitted {
            onChoiceToken?(choiceIdx, str, lpChunk)
        }
        return result.shouldContinue
    }

    func finishAndSnapshot() -> (fullText: String, stoppedByCustomRule: Bool, logprobs: [CompletionLPEntry]) {
        let state = lock.withLock { () -> (String, CompletionLogprobs?, String, Bool, [CompletionLPEntry]) in
            let tail = decoder.flush()
            if !tail.isEmpty && !stoppedByCustomRule {
                let prevCount = fullUTF8.count
                fullUTF8.append(contentsOf: tail.utf8)
                if !stopScanner.isEmpty, let match = stopScanner.firstMatch(in: fullUTF8, previousCount: prevCount) {
                    stoppedByCustomRule = true
                    fullUTF8.removeSubrange(match.byteOffset..<fullUTF8.count)
                }
            }
            let emitted = takeEmissionLocked(upTo: fullUTF8.count)
            let fullText = String(decoding: fullUTF8, as: UTF8.self)
            return (emitted?.0 ?? "", emitted?.1, fullText, stoppedByCustomRule, allCompletionLPEntries)
        }
        if !state.0.isEmpty || state.1 != nil {
            onChoiceToken?(choiceIdx, state.0, state.1)
        }
        return (state.2, state.3, state.4)
    }

    private func takeEmissionLocked(upTo utf8Limit: Int) -> (String, CompletionLogprobs?)? {
        guard onChoiceToken != nil, utf8Limit > emittedUTF8, utf8Limit <= fullUTF8.count else {
            emittedUTF8 = max(emittedUTF8, utf8Limit)
            return nil
        }
        guard let str = String(bytes: fullUTF8[emittedUTF8..<utf8Limit], encoding: .utf8), !str.isEmpty else {
            return nil
        }
        emittedUTF8 = utf8Limit
        let lpChunk: CompletionLogprobs? = pendingLPEntries.isEmpty
            ? nil
            : QWEngineLifecycleActor.buildCompletionLogprobs(entries: pendingLPEntries, startOffset: streamOffset)
        streamOffset += str.count
        pendingLPEntries.removeAll(keepingCapacity: true)
        return (str, lpChunk)
    }
}

private func finiteLogprob(_ value: Float) -> Float {
    value.isFinite ? value : -9999.0
}

private func completionLogprobTuple(
    from lp: QWTokenLogprob,
    tokenizer tok: QWTokenizer,
    topCount: Int
) -> CompletionLPEntry {
    let tokStr = (try? tok.decode([lp.tokenId], skipSpecialTokens: false)) ?? ""
    let safeLP = finiteLogprob(lp.logprob)
    var topDict: [String: Float]? = nil
    if topCount > 0 && !lp.topCandidates.isEmpty {
        var d: [String: Float] = [:]
        for cand in lp.topCandidates {
            let s = (try? tok.decode([cand.tokenId], skipSpecialTokens: false)) ?? ""
            let candLP = finiteLogprob(cand.logprob)
            if d[s].map({ candLP > $0 }) ?? true {
                d[s] = candLP
            }
        }
        topDict = d
    }
    return (tokStr, safeLP, topDict)
}

extension QWEngineLifecycleActor {

    nonisolated public func countChatPromptTokens(request: ChatCompletionRequest) async throws -> Int {
        try await selectModelIfNeeded(request.model, allowSameFamilyQuantSwitch: false)
        let memState = await autoMountAndPrepareMemoryState(for: request)
        let loaded = try await ensureLoaded()
        let config = getConfig()
        let tok = loaded.tokenizer
        let hint = request.maxTokens ?? request.maxCompletionTokens ?? config.maxTokens
        let prepared = try QWChatTurnPreparer.preparePrompt(
            request: request,
            config: config,
            chatTemplate: loaded.chatTemplate,
            tokenizer: tok,
            descriptor: loaded.descriptor,
            supportsTokenConstraints: loaded.modelBackend.supportsTokenConstraints,
            memoryManifest: memState.manifest,
            effectiveMaxNewTokens: hint
        )
        let encoded = try encodeWireIncrementallyWithSystemPrefix(
            wire: prepared.wire, tokenizer: tok, packPath: config.packPath,
            cachePrompt: request.effectiveCachePrompt, cacheSalt: request.effectiveCacheSalt
        )
        let imageUrls = request.extractImageUrls()
        let videoInputs = request.extractVideoInputs()
        guard !imageUrls.isEmpty || !videoInputs.isEmpty else { return encoded.tokens.count }
        try validateRequestCapabilities(request)
        let policy = QWImageDecodePolicy(
            allowDataURLs: true,
            allowLocalImagePaths: config.allowLocalImagePaths,
            allowRemoteImageURLs: config.allowRemoteImageURLs
        )
        let mmTurn = try await visionCoordinator.withLock {
            try await QWImageDecoder.prepareMultimodalTurn(
                promptTokens: encoded.tokens,
                imageUrls: imageUrls,
                videoInputs: videoInputs,
                hiddenDim: loaded.descriptor.hidden,
                visionBackend: loaded.visionBackend,
                policy: policy
            )
        }
        return mmTurn.tokens.count
    }

    nonisolated public func handleTextCompletion(
        request: CompletionRequest,
        onStreamReady: (@Sendable () async -> Void)? = nil,
        onEchoPrompt: (@Sendable (Int, String, CompletionLogprobs?) -> Void)? = nil,
        onChoiceToken: (@Sendable (Int, String, CompletionLogprobs?) -> Void)? = nil,
        shouldStop: (@Sendable () -> Bool)? = nil
    ) async throws -> CompletionExecutionResult {
        if let lp = request.logprobs, lp < 0 {
            throw CompletionInvalidInputError("logprobs must be >= 0")
        }
        if let maxT = request.maxTokens, maxT < 0 {
            throw CompletionInvalidInputError("max_tokens must be >= 0")
        }
        let admitted = try await beginAdmittedRequest(
            requestedModel: request.model,
            allowSameFamilyQuantSwitch: false
        )
        do {
            let res = try await executeTextCompletion(
                request: request,
                admitted: admitted,
                onStreamReady: onStreamReady,
                onEchoPrompt: onEchoPrompt,
                onChoiceToken: onChoiceToken,
                shouldStop: shouldStop
            )
            await endInFlightRequest()
            return res
        } catch {
            await endInFlightRequest()
            throw error
        }
    }

    nonisolated private func executeTextCompletion(
        request: CompletionRequest,
        admitted: AdmittedRequest,
        onStreamReady: (@Sendable () async -> Void)?,
        onEchoPrompt: (@Sendable (Int, String, CompletionLogprobs?) -> Void)?,
        onChoiceToken: (@Sendable (Int, String, CompletionLogprobs?) -> Void)?,
        shouldStop: (@Sendable () -> Bool)?
    ) async throws -> CompletionExecutionResult {
        let loaded = admitted.loaded
        let config = admitted.config
        let tok = loaded.tokenizer
        let vocab = loaded.descriptor.vocab
        let specialTokens = tok.resolveSpecialTokens(fallback: loaded.descriptor.specialTokens)

        QWConcurrentBatchTracker.beginPreparing()
        var preparingActive = true
        defer {
            if preparingActive { QWConcurrentBatchTracker.endPreparing() }
        }

        let prompts = try Self.resolveCompletionPrompts(
            request.prompt, tokenizer: tok, vocabulary: vocab, specialTokens: specialTokens
        )
        let effectiveMaxContext = QWChatPolicyResolver.effectiveContextBudget(
            configMaxContext: config.maxContext,
            backendContextBudget: loaded.modelBackend.parametersInfo.contextBudget
        )
        for p in prompts where p.tokens.count >= effectiveMaxContext {
            throw QWServerContextLengthError(promptTokens: p.tokens.count, maxContext: effectiveMaxContext)
        }

        await onStreamReady?()

        let requestedMaxNew = request.maxTokens ?? 16
        let collectLP = request.logprobs != nil
        let topLPCount = min(max(0, request.logprobs ?? 0), 20)
        let echo = request.echo ?? false
        let customStops = (request.ignoreEos == true) ? [] : (request.stop ?? []).filter { !$0.isEmpty }
        let usePrefixCache = request.cachePrompt ?? true
        let cacheSalt = request.cacheSalt ?? request.promptCacheKey
        let sampling = QWChatPolicyResolver.resolveSampling(for: request, config: config).parameters

        @Sendable func runPrompt(
            choiceIdx: Int,
            promptItem: (tokens: [Int], echoText: String),
            tracksPreparation: Bool
        ) async throws -> CompletionExecutionChoice {
            var localPreparing = tracksPreparation
            if tracksPreparation { QWConcurrentBatchTracker.beginPreparing() }
            defer { if localPreparing { QWConcurrentBatchTracker.endPreparing() } }

            let promptTokens = promptItem.tokens
            let maxNew = requestedMaxNew == 0
                ? 0
                : min(requestedMaxNew, max(1, effectiveMaxContext - promptTokens.count))

            var promptLPEntries: [CompletionLPEntry] = []
            var offset = 0
            if echo {
                if collectLP {
                    promptLPEntries = try await computeEchoPromptLogprobs(
                        promptTokens: promptTokens,
                        tokenizer: tok,
                        vocabulary: vocab,
                        backend: loaded.modelBackend,
                        config: config,
                        topLPCount: topLPCount,
                        usePrefixCache: usePrefixCache,
                        cacheSalt: cacheSalt,
                        shouldStop: shouldStop
                    )
                    let echoLP = Self.buildCompletionLogprobs(entries: promptLPEntries, startOffset: 0)
                    onEchoPrompt?(choiceIdx, promptItem.echoText, echoLP)
                    offset = promptItem.echoText.count
                } else {
                    onEchoPrompt?(choiceIdx, promptItem.echoText, nil)
                    offset = promptItem.echoText.count
                }
            }

            if maxNew == 0 {
                if localPreparing {
                    localPreparing = false
                    QWConcurrentBatchTracker.endPreparing()
                }
                let lpOut = echo && collectLP ? Self.buildCompletionLogprobs(entries: promptLPEntries, startOffset: 0) : nil
                let fullText = echo ? promptItem.echoText : ""
                return CompletionExecutionChoice(
                    index: choiceIdx, text: fullText, finishReason: "stop",
                    promptTokens: promptTokens.count, completionTokens: 0, logprobs: lpOut
                )
            }

            var stops = specialTokens.defaultStopTokenIds
            if request.ignoreEos == true || (request.minTokens ?? 0) >= maxNew {
                stops.removeAll()
            }

            let capture = RawCompletionCapture(
                tok: tok, customStops: customStops, streamOffset: offset, choiceIdx: choiceIdx,
                onChoiceToken: onChoiceToken, shouldStop: shouldStop
            )
            let tokenCallback: @Sendable (Int) -> Bool = { id in
                capture.processToken(id)
            }

            var lpCallback: (@Sendable (QWTokenLogprob) -> Void)? = nil
            if collectLP {
                let wantPending = (onChoiceToken != nil)
                lpCallback = { (lp: QWTokenLogprob) in
                    let tuple = completionLogprobTuple(from: lp, tokenizer: tok, topCount: topLPCount)
                    capture.appendLogprob(tuple, trackPending: wantPending)
                }
            }

            let genReq = QWModelGenerationRequest(
                promptTokens: promptTokens,
                maxNewTokens: maxNew,
                sampling: sampling,
                stopTokenIds: stops,
                collectLogprobs: collectLP,
                topLogprobsCount: topLPCount,
                onTokenLogprob: lpCallback,
                exactSpeculation: sampling.hasExplicitSeed,
                usePrefixCache: usePrefixCache,
                cacheSalt: cacheSalt
            )

            if localPreparing {
                localPreparing = false
                QWConcurrentBatchTracker.endPreparing()
            }
            let metrics = try await withCheckedThrowingContinuation { cont in
                DispatchQueue.global(qos: .userInitiated).async {
                    do {
                        let m = try loaded.modelBackend.generate(request: genReq, config: config, onToken: tokenCallback)
                        cont.resume(returning: m)
                    } catch {
                        cont.resume(throwing: error)
                    }
                }
            }

            let snap = capture.finishAndSnapshot()
            let completionCount = metrics.tokens.count
            let truncated = !snap.stoppedByCustomRule
                && (completionCount >= maxNew)
                && (metrics.tokens.last.map { !stops.contains($0) } ?? true)
            let finishReason = truncated ? "length" : "stop"
            let suffixStr = (!truncated ? request.suffix : nil) ?? ""
            let finalChoiceText = (echo ? promptItem.echoText : "") + snap.fullText + suffixStr

            var finalLogprobs: CompletionLogprobs? = nil
            if collectLP {
                let completionEntries: [CompletionLPEntry]
                if !snap.logprobs.isEmpty {
                    completionEntries = snap.logprobs
                } else {
                    completionEntries = metrics.tokenLogprobs.map {
                        completionLogprobTuple(from: $0, tokenizer: tok, topCount: topLPCount)
                    }
                }
                let combined = echo ? (promptLPEntries + completionEntries) : completionEntries
                finalLogprobs = Self.buildCompletionLogprobs(entries: combined, startOffset: 0)
            }

            return CompletionExecutionChoice(
                index: choiceIdx, text: finalChoiceText, finishReason: finishReason,
                promptTokens: promptTokens.count, completionTokens: completionCount, logprobs: finalLogprobs
            )
        }

        var choices: [CompletionExecutionChoice] = []
        var totalIn = 0
        var totalOut = 0

        if prompts.count > 1 && !echo && requestedMaxNew > 0 {
            preparingActive = false
            QWConcurrentBatchTracker.endPreparing()
            var byIdx = [CompletionExecutionChoice?](repeating: nil, count: prompts.count)
            try await withThrowingTaskGroup(of: CompletionExecutionChoice.self) { group in
                for (choiceIdx, promptItem) in prompts.enumerated() {
                    group.addTask {
                        try await runPrompt(choiceIdx: choiceIdx, promptItem: promptItem, tracksPreparation: true)
                    }
                }
                for try await c in group {
                    byIdx[c.index] = c
                }
            }
            for c in byIdx.compactMap({ $0 }) {
                choices.append(c)
                totalIn += c.promptTokens
                totalOut += c.completionTokens
            }
        } else {
            for (choiceIdx, promptItem) in prompts.enumerated() {
                if shouldStop?() == true || Task.isCancelled { break }
                let tracksLocal = !preparingActive
                if choiceIdx == 0 && preparingActive {
                    preparingActive = false
                    QWConcurrentBatchTracker.endPreparing()
                }
                let c = try await runPrompt(
                    choiceIdx: choiceIdx,
                    promptItem: promptItem,
                    tracksPreparation: tracksLocal
                )
                choices.append(c)
                totalIn += c.promptTokens
                totalOut += c.completionTokens
            }
        }

        return CompletionExecutionResult(choices: choices, totalPromptTokens: totalIn, totalCompletionTokens: totalOut)
    }

    nonisolated private static func isValidRemoteTokenizerToken(_ token: String, tokenizer tok: QWTokenizer) -> Bool {
        guard !token.isEmpty else { return false }
        if let ids = try? tok.encode(token, addSpecialTokens: false), !ids.isEmpty {
            return true
        }
        return tok.tokenId(for: token) != nil
    }

    nonisolated public func handleTokenizerInfo(model: String? = nil) async throws -> TokenizerInfoResponse {
        try await selectModelIfNeeded(model, allowSameFamilyQuantSwitch: false)
        let loaded = try await ensureLoaded()
        let tok = loaded.tokenizer
        let tpl = loaded.chatTemplate

        func contextToken(_ key: String) -> String? {
            guard let s = tpl.tokenizerTokenString(key),
                  Self.isValidRemoteTokenizerToken(s, tokenizer: tok) else { return nil }
            return s
        }

        let eosCandidates: [String?] = [
            tpl.tokenizerTokenString("eos_token"),
            tok.piece(loaded.descriptor.eosTokenId),
            try? tok.decode([loaded.descriptor.eosTokenId], skipSpecialTokens: false),
            "<|endoftext|>"
        ]
        let eosToken = eosCandidates
            .compactMap { $0 }
            .first { Self.isValidRemoteTokenizerToken($0, tokenizer: tok) }
            ?? "<|endoftext|>"

        return TokenizerInfoResponse(
            eosToken: eosToken,
            bosToken: contextToken("bos_token"),
            padToken: contextToken("pad_token"),
            unkToken: contextToken("unk_token"),
            chatTemplate: tpl.rawSource.isEmpty ? nil : tpl.rawSource
        )
    }

    nonisolated public func handleTokenize(request: TokenizeRequest) async throws -> TokenizeResponse {
        guard let rawText = request.prompt ?? request.text else {
            throw CompletionInvalidInputError("tokenize request requires a 'prompt' or 'text' string")
        }
        try await selectModelIfNeeded(request.model, allowSameFamilyQuantSwitch: false)
        let loaded = try await ensureLoaded()
        let config = getConfig()
        let tok = loaded.tokenizer
        var ids = try tok.encode(rawText, addSpecialTokens: request.addSpecialTokens ?? false)
        if ids.isEmpty && !rawText.isEmpty, let singleId = tok.tokenId(for: rawText) {
            ids = [singleId]
        }
        let effectiveMaxContext = QWChatPolicyResolver.effectiveContextBudget(
            configMaxContext: config.maxContext,
            backendContextBudget: loaded.modelBackend.parametersInfo.contextBudget
        )
        let maxModelLen = effectiveMaxContext > 0 ? effectiveMaxContext : config.maxContext
        return TokenizeResponse(tokens: ids, count: ids.count, maxModelLen: maxModelLen)
    }

    nonisolated public func handleDetokenize(request: DetokenizeRequest) async throws -> DetokenizeResponse {
        try await selectModelIfNeeded(request.model, allowSameFamilyQuantSwitch: false)
        let loaded = try await ensureLoaded()
        let vocab = loaded.descriptor.vocab
        for id in request.tokens where id < 0 || id >= vocab {
            throw CompletionInvalidInputError("token id \(id) is out of vocabulary 0..<\(vocab)")
        }
        let prompt: String
        do {
            prompt = try loaded.tokenizer.decode(request.tokens, skipSpecialTokens: false)
        } catch {
            throw CompletionInvalidInputError("unable to detokenize token ids: \(error)")
        }
        return DetokenizeResponse(prompt: prompt)
    }

    nonisolated private func computeEchoPromptLogprobs(
        promptTokens: [Int],
        tokenizer tok: QWTokenizer,
        vocabulary: Int,
        backend: any QWModelBackend,
        config: QWEngineLifecycleActor.Config,
        topLPCount: Int,
        usePrefixCache: Bool,
        cacheSalt: String?,
        shouldStop: (@Sendable () -> Bool)?
    ) async throws -> [CompletionLPEntry] {
        guard !promptTokens.isEmpty else { return [] }
        let firstStr = (try? tok.decode([promptTokens[0]], skipSpecialTokens: false)) ?? ""
        var out: [CompletionLPEntry] = [(firstStr, nil, nil)]
        guard promptTokens.count > 1 else { return out }
        let greedy = QWSamplingParameters(temperature: 0, topP: 1, topK: 1, repetitionPenalty: 1.0)
        if let singlePass = try await withCheckedThrowingContinuation({ (cont: CheckedContinuation<[QWTokenLogprob]?, any Error>) in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    cont.resume(returning: try backend.computePromptLogprobs(
                        promptTokens: promptTokens,
                        topLogprobsCount: topLPCount,
                        sampling: greedy,
                        shouldStop: shouldStop
                    ))
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }) {
            out.append(contentsOf: singlePass.map {
                completionLogprobTuple(from: $0, tokenizer: tok, topCount: topLPCount)
            })
            return out
        }
        for i in 1..<promptTokens.count {
            if shouldStop?() == true || Task.isCancelled { break }
            let targetId = promptTokens[i]
            let tokStr = (try? tok.decode([targetId], skipSpecialTokens: false)) ?? ""
            guard backend.supportsTokenConstraints else {
                out.append((tokStr, nil, nil))
                continue
            }
            let prefix = Array(promptTokens[0..<i])
            let constraint = QWSingleTokenConstraint(tokenId: targetId, vocabularySize: vocabulary)
            let req = QWModelGenerationRequest(
                promptTokens: prefix, maxNewTokens: 1, sampling: greedy, stopTokenIds: [],
                collectLogprobs: true, topLogprobsCount: topLPCount,
                constraint: constraint, exactSpeculation: true,
                usePrefixCache: usePrefixCache, cacheSalt: cacheSalt
            )
            let m = try await withCheckedThrowingContinuation { cont in
                DispatchQueue.global(qos: .userInitiated).async {
                    do {
                        cont.resume(returning: try backend.generate(request: req, config: config, onToken: nil))
                    } catch {
                        cont.resume(throwing: error)
                    }
                }
            }
            if let lp = m.tokenLogprobs.first, lp.tokenId == targetId {
                out.append(completionLogprobTuple(from: lp, tokenizer: tok, topCount: topLPCount))
            } else {
                out.append((tokStr, nil, nil))
            }
        }
        return out
    }

    nonisolated static func buildCompletionLogprobs(
        entries: [(tok: String, lp: Float?, top: [String: Float]?)],
        startOffset: Int
    ) -> CompletionLogprobs {
        var offsets: [Int] = []
        var tokenLPs: [Float?] = []
        var tokens: [String] = []
        var topLPs: [[String: Float]?] = []
        var cursor = startOffset
        for e in entries {
            offsets.append(cursor)
            tokens.append(e.tok)
            tokenLPs.append(e.lp)
            topLPs.append(e.top)
            cursor += e.tok.count
        }
        return CompletionLogprobs(textOffset: offsets, tokenLogprobs: tokenLPs, tokens: tokens, topLogprobs: topLPs)
    }

    nonisolated private static func resolveCompletionPrompts(
        _ input: CompletionPromptInput,
        tokenizer tok: QWTokenizer,
        vocabulary: Int,
        specialTokens: QWModelSpecialTokens
    ) throws -> [(tokens: [Int], echoText: String)] {
        switch input {
        case .strings(let list):
            var out: [(tokens: [Int], echoText: String)] = []
            for raw in list {
                var ids = try tok.encode(raw, addSpecialTokens: false)
                if ids.isEmpty {
                    let fallback = tok.tokenId(for: "<bos>") ?? tok.tokenId(for: "<s>") ?? tok.tokenId(for: "<|endoftext|>") ?? specialTokens.eosTokenId
                    if fallback >= 0 && fallback < vocabulary {
                        ids = [fallback]
                    } else {
                        throw CompletionInvalidInputError("prompt produced zero tokens")
                    }
                }
                out.append((ids, raw))
            }
            return out
        case .tokenArrays(let batches):
            var out: [(tokens: [Int], echoText: String)] = []
            for ids in batches {
                guard !ids.isEmpty else {
                    throw CompletionInvalidInputError("token prompt array must not be empty")
                }
                for id in ids where id < 0 || id >= vocabulary {
                    throw CompletionInvalidInputError("token id \(id) is out of vocabulary 0..<\(vocabulary)")
                }
                let decoded = (try? tok.decode(ids, skipSpecialTokens: false)) ?? ""
                out.append((ids, decoded))
            }
            return out
        }
    }
}
