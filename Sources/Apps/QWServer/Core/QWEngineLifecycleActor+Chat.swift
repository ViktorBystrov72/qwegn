// QWEngineLifecycleActor+Chat.swift — Chat inference handler delegating policy
// resolution to QWChatPolicyResolver and prompt assembly to QWChatTurnPreparer.

import Foundation
import QWCli
import QWCore
import QWEngine
import QWHAL
import QWHead
import QWKV
import QWMM
import QWQuant
import QWToken
import QWVision

private final class ChatStreamCapture: @unchecked Sendable {
    private static let endThinkUTF8 = Array("</think>".utf8)
    private static let endErnieThinkUTF8 = Array("</ernie:think>".utf8)
    private static let toolOpenUTF8 = Array("<tool_call>".utf8)
    private static let toolCloseUTF8 = Array("</tool_call>".utf8)
    private static let funcOpenUTF8 = Array("<function=".utf8)
    private static let funcCloseUTF8 = Array("</function>".utf8)

    private let lock = NSLock()
    let tok: QWTokenizer
    let stopScanner: QWEngineLifecycleActor.IncrementalStopScanner
    let endThinkCloseTag: String
    var streamDecoder: QWStreamDecoder
    var fullUTF8: [UInt8]
    var emittedUTF8Count = 0
    var remainingHealUTF8: Int
    var healedConsumedTokens = 0
    var hasReceivedPiece = false
    var ttftSec: Double?
    var inThinking: Bool
    var reasoningTokenCount = 0
    var matchedStopSequence: String?
    var stoppedByCustomRule = false
    var jsonTracker = QWEngineLifecycleActor.IncrementalJSONObjectTracker()
    var sawToolCallOpen = false
    let onToken: ((String) -> Void)?
    let onTokenLogprob: ((ChatLogprobTokenEntry) -> Void)?
    let modelBackend: any QWModelBackend

    init(
        tok: QWTokenizer,
        initialPrefix: String,
        initialHealUTF8: Int,
        startsInThinking: Bool,
        customStops: [String],
        endThinkCloseTag: String,
        onToken: ((String) -> Void)?,
        onTokenLogprob: ((ChatLogprobTokenEntry) -> Void)?,
        modelBackend: any QWModelBackend
    ) {
        self.tok = tok
        self.stopScanner = QWEngineLifecycleActor.IncrementalStopScanner(stops: customStops)
        self.endThinkCloseTag = endThinkCloseTag
        self.streamDecoder = QWStreamDecoder(tok)
        let initialBytes = Array(initialPrefix.utf8)
        self.fullUTF8 = initialBytes
        self.jsonTracker.advance(utf8: initialBytes)
        self.remainingHealUTF8 = initialHealUTF8
        self.inThinking = startsInThinking
        self.onToken = onToken
        self.onTokenLogprob = onTokenLogprob
        self.modelBackend = modelBackend
    }

    private static func matchesAt(_ utf8: [UInt8], offset: Int, needle: [UInt8]) -> Bool {
        for j in 0..<needle.count where utf8[offset + j] != needle[j] { return false }
        return true
    }

    static func containsBytesNearTail(_ utf8: [UInt8], needle: [UInt8], prevCount: Int) -> Bool {
        guard utf8.count >= needle.count else { return false }
        let start = max(0, prevCount - needle.count)
        let lastStart = utf8.count - needle.count
        guard start <= lastStart else { return false }
        for i in start...lastStart where utf8[i] == needle[0] && matchesAt(utf8, offset: i, needle: needle) {
            return true
        }
        return false
    }

    static func lastOffsetOfBytes(_ utf8: [UInt8], needle: [UInt8]) -> Int? {
        guard utf8.count >= needle.count else { return nil }
        for i in stride(from: utf8.count - needle.count, through: 0, by: -1)
        where utf8[i] == needle[0] && matchesAt(utf8, offset: i, needle: needle) {
            return i
        }
        return nil
    }

    private func appendPieceLocked(_ piece: String) -> Int {
        let prevCount = fullUTF8.count
        fullUTF8.append(contentsOf: piece.utf8)
        jsonTracker.advance(utf8: fullUTF8)
        if !sawToolCallOpen && Self.containsBytesNearTail(fullUTF8, needle: Self.toolOpenUTF8, prevCount: prevCount) {
            sawToolCallOpen = true
        }
        return prevCount
    }

    private func insideOpenToolFunctionLocked() -> Bool {
        guard let lo = Self.lastOffsetOfBytes(fullUTF8, needle: Self.funcOpenUTF8) else { return false }
        if let lc = Self.lastOffsetOfBytes(fullUTF8, needle: Self.funcCloseUTF8) { return lo > lc }
        return true
    }

    private func emitAndReturnStringLocked(upTo utf8Limit: Int) -> String? {
        guard onToken != nil else {
            emittedUTF8Count = max(emittedUTF8Count, min(utf8Limit, fullUTF8.count))
            return nil
        }
        let clamped = min(utf8Limit, fullUTF8.count)
        guard clamped > emittedUTF8Count else { return nil }
        if let str = String(bytes: fullUTF8[emittedUTF8Count..<clamped], encoding: .utf8) {
            emittedUTF8Count = clamped
            return str.isEmpty ? nil : str
        }
        return nil
    }

    func processToken(
        _ id: Int,
        inferStart: Date,
        promptTokCount: Int,
        telemetry: QWEngineTelemetryRegistry,
        userTextContainsImEnd: Bool,
        imEndTokenId: Int,
        endThinkTokenId: Int,
        isConstrainedJson: Bool,
        stopOnSingleTool: Bool
    ) -> (shouldContinue: Bool, emitChunk: String?) {
        lock.withLock {
            if ttftSec == nil {
                ttftSec = max(0.0001, Date().timeIntervalSince(inferStart))
                telemetry.beginDecode(promptTokens: promptTokCount)
            } else {
                telemetry.recordDecodeToken()
            }
            let wasInThinking = inThinking
            if inThinking {
                reasoningTokenCount += 1
                if id == endThinkTokenId { inThinking = false }
            }
            var piece = ""
            if userTextContainsImEnd && id == imEndTokenId {
                let insideOpenJson = isConstrainedJson && (jsonTracker.completedByteOffset == nil)
                if insideOpenToolFunctionLocked() || insideOpenJson {
                    piece = "<|im_end|>"
                } else {
                    stoppedByCustomRule = true
                    return (false, nil)
                }
            } else if let decoded = try? streamDecoder.decode(id, skipSpecialTokens: true) {
                piece = (decoded.isEmpty && wasInThinking && id == endThinkTokenId) ? endThinkCloseTag : decoded
            }
            guard !piece.isEmpty else { return (true, nil) }

            if remainingHealUTF8 > 0 {
                let pBytes = Array(piece.utf8)
                if pBytes.count <= remainingHealUTF8 {
                    remainingHealUTF8 -= pBytes.count
                    healedConsumedTokens += 1
                    return (true, nil)
                }
                let tail = Array(pBytes.dropFirst(remainingHealUTF8))
                remainingHealUTF8 = 0
                piece = String(decoding: tail, as: UTF8.self)
                guard !piece.isEmpty else { return (true, nil) }
            }

            if !hasReceivedPiece {
                hasReceivedPiece = true
                if isConstrainedJson {
                    let trimmed = piece.drop(while: { $0.isWhitespace })
                    if trimmed.hasPrefix("{") { piece = String(trimmed.dropFirst()) }
                }
            }
            let prevCount = appendPieceLocked(piece)
            if inThinking && (Self.containsBytesNearTail(fullUTF8, needle: Self.endThinkUTF8, prevCount: prevCount)
                || Self.containsBytesNearTail(fullUTF8, needle: Self.endErnieThinkUTF8, prevCount: prevCount)) {
                inThinking = false
            }

            if !stopScanner.isEmpty, let match = stopScanner.firstMatch(in: fullUTF8, previousCount: prevCount) {
                matchedStopSequence = match.stop
                stoppedByCustomRule = true
                fullUTF8.removeSubrange(match.byteOffset..<fullUTF8.count)
                return (false, emitAndReturnStringLocked(upTo: fullUTF8.count))
            }
            if isConstrainedJson, let jsonEnd = jsonTracker.completedByteOffset {
                fullUTF8.removeSubrange(jsonEnd..<fullUTF8.count)
                stoppedByCustomRule = true
                return (false, emitAndReturnStringLocked(upTo: fullUTF8.count))
            }
            if stopOnSingleTool && sawToolCallOpen
                && Self.containsBytesNearTail(fullUTF8, needle: Self.toolCloseUTF8, prevCount: prevCount) {
                let text = String(decoding: fullUTF8, as: UTF8.self)
                if let closeEnd = QWToolCallParser.firstCompleteToolCallEnd(in: text) {
                    fullUTF8 = Array(text[..<closeEnd].utf8)
                    stoppedByCustomRule = true
                    return (false, emitAndReturnStringLocked(upTo: fullUTF8.count))
                }
            }
            let safeUTF8 = stopScanner.isEmpty
                ? fullUTF8.count
                : max(0, fullUTF8.count - stopScanner.heldSuffixUTF8Count(in: fullUTF8))
            return (true, emitAndReturnStringLocked(upTo: safeUTF8))
        }
    }

    func markStoppedByConstraint() {
        lock.withLock { stoppedByCustomRule = true }
    }

    func emitRemainderAndSnapshot() -> (
        fullText: String,
        ttftSec: Double?,
        matchedStopSequence: String?,
        stoppedByCustomRule: Bool,
        healedConsumedTokens: Int,
        reasoningTokenCount: Int,
        finalChunk: String?
    ) {
        lock.withLock {
            let tail = streamDecoder.flush()
            if !tail.isEmpty && !stoppedByCustomRule {
                let prev = appendPieceLocked(tail)
                if !stopScanner.isEmpty && matchedStopSequence == nil,
                   let match = stopScanner.firstMatch(in: fullUTF8, previousCount: prev) {
                    matchedStopSequence = match.stop
                    stoppedByCustomRule = true
                    fullUTF8.removeSubrange(match.byteOffset..<fullUTF8.count)
                }
            }
            let chunk = emitAndReturnStringLocked(upTo: fullUTF8.count)
            return (String(decoding: fullUTF8, as: UTF8.self), ttftSec, matchedStopSequence,
                    stoppedByCustomRule, healedConsumedTokens, reasoningTokenCount, chunk)
        }
    }
}

extension QWEngineLifecycleActor {

    /// Dynamically loads or unloads the vision tower in place without unloading the LLM or resetting the KV cache.
    public func setVisionLoaded(_ loadVision: Bool) async throws {
        config.loadVision = loadVision
        statusBox.updateConfig(config)
        if case .loaded(let loaded) = state {
            try await visionCoordinator.withLock {
                try loaded.visionBackend?.setVisionLoaded(loadVision)
            }
            logger.log(.info, "Vision tower \(loadVision ? "loaded into" : "unloaded from") memory in place (KV cache preserved).")
        }
    }

    nonisolated public func handleChatCompletion(
        request: ChatCompletionRequest,
        onStreamReady: (@Sendable () async -> Void)? = nil,
        onPromptTokensComputed: (@Sendable (Int) -> Void)? = nil,
        onToken: ((String) -> Void)? = nil,
        onTokenLogprob: ((ChatLogprobTokenEntry) -> Void)? = nil,
        shouldStop: (@Sendable () -> Bool)? = nil
    ) async throws -> ChatCompletionExecutionResult {
        try await waitForModelTransitionsAndSelectIfNeeded(
            requestedModel: request.model,
            allowSameFamilyQuantSwitch: false
        )
        let memState = await autoMountAndPrepareMemoryState(for: request)
        let admitted = try await beginAdmittedRequest(
            requestedModel: request.model,
            allowSameFamilyQuantSwitch: false
        )
        do {
            let result = try await executeChatCompletion(
                request: request,
                admitted: admitted,
                memState: memState,
                onStreamReady: onStreamReady,
                onPromptTokensComputed: onPromptTokensComputed,
                onToken: onToken,
                onTokenLogprob: onTokenLogprob,
                shouldStop: shouldStop
            )
            await endInFlightRequest()
            return result
        } catch {
            await endInFlightRequest()
            throw error
        }
    }

    nonisolated private func executeChatCompletion(
        request: ChatCompletionRequest,
        admitted: AdmittedRequest,
        memState: (mountedTokens: [Int], deltaTokens: [Int], manifest: QWMemoryManifest?),
        onStreamReady: (@Sendable () async -> Void)?,
        onPromptTokensComputed: (@Sendable (Int) -> Void)?,
        onToken: ((String) -> Void)?,
        onTokenLogprob: ((ChatLogprobTokenEntry) -> Void)?,
        shouldStop: (@Sendable () -> Bool)?
    ) async throws -> ChatCompletionExecutionResult {
        let loaded = admitted.loaded
        let config = admitted.config
        QWConcurrentBatchTracker.beginPreparing()
        var preparingActive = true
        defer { if preparingActive { QWConcurrentBatchTracker.endPreparing() } }
        let inferStart = Date()
        let tok = loaded.tokenizer
        let imageUrls = request.extractImageUrls()

        let prepared = try QWChatTurnPreparer.preparePrompt(
            request: request,
            config: config,
            chatTemplate: loaded.chatTemplate,
            tokenizer: tok,
            descriptor: loaded.descriptor,
            supportsTokenConstraints: loaded.modelBackend.supportsTokenConstraints,
            memoryManifest: memState.manifest,
            effectiveMaxNewTokens: request.maxTokens ?? request.maxCompletionTokens ?? config.maxTokens
        )
        let cacheSalt = request.effectiveCacheSalt
        let usePrefixCache = request.effectiveCachePrompt
        let encodedWire = try encodeWireIncrementallyWithSystemPrefix(
            wire: prepared.wire, tokenizer: tok, packPath: config.packPath,
            cachePrompt: usePrefixCache, cacheSalt: cacheSalt
        )
        let mmTurn = try await prepareMultimodalInput(
            promptTokens: encodedWire.tokens,
            imageUrls: imageUrls,
            loaded: loaded,
            config: config
        )
        let visualSpans = mmTurn.visualSpans
        let budget = try QWChatPolicyResolver.resolveChatBudget(
            for: request,
            promptTokenCount: mmTurn.tokens.count,
            config: config,
            backendContextBudget: loaded.modelBackend.parametersInfo.contextBudget
        )
        let budgetGuard = QWMemoryVault.applyElasticBudgetGuard(
            mountedMemoryTokens: memState.mountedTokens,
            deltaTokens: memState.deltaTokens,
            agentPromptTokens: mmTurn.tokens.count,
            maxNewTokens: budget.requestedMaxNew,
            maxContext: budget.effectiveMaxContext
        )
        if budgetGuard.bypassedOversizedCache {
            logger.log(.info, "[MemoryPack] Elastic budget guard bypassed oversized .qwcache (\(memState.mountedTokens.count) tokens) to preserve \(mmTurn.tokens.count)-token agent prompt within maxContext=\(budget.effectiveMaxContext)")
        }
        let memoryPrefixTokens = budgetGuard.prefixTokens
        let effectivePromptTokens = (visualSpans.isEmpty && !memoryPrefixTokens.isEmpty)
            ? memoryPrefixTokens + mmTurn.tokens
            : mmTurn.tokens
        let effectiveSystemPrefixTokenCount = (visualSpans.isEmpty && !memoryPrefixTokens.isEmpty)
            ? encodedWire.systemPrefixTokenCount.map { memoryPrefixTokens.count + $0 }
            : encodedWire.systemPrefixTokenCount

        let reportedPromptTokenCount = effectivePromptTokens.count
        guard reportedPromptTokenCount < budget.effectiveMaxContext else {
            throw QWServerContextLengthError(promptTokens: reportedPromptTokenCount, maxContext: budget.effectiveMaxContext)
        }
        let maxNew = min(budget.requestedMaxNew, max(1, budget.effectiveMaxContext - reportedPromptTokenCount))
        var stops = prepared.stops
        let heal = Self.prepareTokenHeal(
            prepared: prepared, request: request, promptTokens: effectivePromptTokens,
            systemPrefixTokenCount: effectiveSystemPrefixTokenCount, stops: stops,
            tokenizer: tok, vocabulary: loaded.descriptor.vocab,
            supportsTokenConstraints: loaded.modelBackend.supportsTokenConstraints,
            hasImages: !visualSpans.isEmpty
        )
        let genPromptTokens = heal.promptTokens
        let activeGenConstraint = QWChatPolicyResolver.resolveActiveConstraint(
            prepared: prepared, request: request, didHealPeel: heal.didPeel,
            healConstraint: heal.constraint, requestedMaxNew: budget.requestedMaxNew,
            clampedMaxNew: maxNew, tokenizer: tok, vocabularySize: loaded.descriptor.vocab,
            supportsTokenConstraints: loaded.modelBackend.supportsTokenConstraints
        )
        let endThinkCloseTag = (tok.tokenId(for: "</ernie:think>") == prepared.endThinkTokenId)
            ? "</ernie:think>"
            : "</think>"
        let capture = ChatStreamCapture(
            tok: tok, initialPrefix: prepared.isConstrainedJson ? "{" : "",
            initialHealUTF8: heal.stripUTF8Count, startsInThinking: prepared.shouldThink,
            customStops: prepared.customStopStrings, endThinkCloseTag: endThinkCloseTag,
            onToken: onToken, onTokenLogprob: onTokenLogprob, modelBackend: loaded.modelBackend
        )
        let genMaxNew = heal.didPeel ? (maxNew + 1) : maxNew
        if request.ignoreEos == true || (request.minTokens ?? 0) >= maxNew { stops.removeAll() }

        onPromptTokensComputed?(reportedPromptTokenCount)
        await onStreamReady?()
        let telemetry = QWEngineTelemetryRegistry.shared
        telemetry.beginPrefill(promptTokens: genPromptTokens.count)
        let promptTokCount = genPromptTokens.count
        let stopOnSingleTool = (request.parallelToolCalls == false) && prepared.hasActiveTools
        let cancelFlag = QWGenerationCancelFlag()

        let tokenCallback: @Sendable (Int) -> Bool = { id in
            if cancelFlag.isCancelled || shouldStop?() == true {
                capture.markStoppedByConstraint()
                return false
            }
            let step = capture.processToken(
                id, inferStart: inferStart, promptTokCount: promptTokCount, telemetry: telemetry,
                userTextContainsImEnd: prepared.userTextContainsImEnd, imEndTokenId: prepared.imEndTokenId,
                endThinkTokenId: prepared.endThinkTokenId, isConstrainedJson: prepared.isConstrainedJson,
                stopOnSingleTool: stopOnSingleTool
            )
            if let chunk = step.emitChunk { capture.onToken?(chunk) }
            if step.shouldContinue, let jsonConstraint = prepared.jsonConstraint, jsonConstraint.isComplete {
                capture.markStoppedByConstraint()
                if let tail = capture.emitRemainderAndSnapshot().finalChunk { capture.onToken?(tail) }
                return false
            }
            return step.shouldContinue
        }

        let collectLP = request.logprobs == true
        let topLPCount = min(max(0, request.topLogprobs ?? 0), 20)
        var logprobCallback: (@Sendable (QWTokenLogprob) -> Void)? = nil
        if collectLP && capture.onTokenLogprob != nil {
            logprobCallback = { (lp: QWTokenLogprob) in
                capture.onTokenLogprob?(Self.makeLogprobEntry(lp, tokenizer: capture.tok))
            }
        }

        let samplingResolution = QWChatPolicyResolver.resolveSampling(for: request, config: config)
        logSamplingSelectionIfChanged(samplingResolution.parameters.denseNaxSelectionLogMessage(
            paramsSource: samplingResolution.sourceLabel))

        let firstVisualOffset = visualSpans.map(\.tokenOffset).min() ?? genPromptTokens.count
        let cachePolicy = QWChatPolicyResolver.resolveCacheAndSpeculation(
            for: request,
            systemPrefixTokenCount: effectiveSystemPrefixTokenCount,
            firstVisualTokenOffset: firstVisualOffset,
            isConstrainedJson: prepared.isConstrainedJson,
            hasExplicitSeed: samplingResolution.parameters.hasExplicitSeed
        )
        let genRequest = QWModelGenerationRequest(
            promptTokens: genPromptTokens,
            maxNewTokens: genMaxNew,
            sampling: samplingResolution.parameters,
            stopTokenIds: stops,
            retainPrefixLength: cachePolicy.retainPrefixLength,
            hasImages: !visualSpans.isEmpty,
            collectLogprobs: collectLP,
            topLogprobsCount: topLPCount,
            onTokenLogprob: logprobCallback,
            constraint: activeGenConstraint,
            exactSpeculation: cachePolicy.exactSpeculation,
            usePrefixCache: cachePolicy.usePrefixCache,
            cacheSalt: cachePolicy.cacheSalt,
            visualSpans: visualSpans,
            onPrefillProgress: { completed, total in
                telemetry.updatePrefillProgress(completedTokens: completed, totalTokens: total)
            }
        )

        let metrics: QWModelCompletionMetrics
        preparingActive = false
        QWConcurrentBatchTracker.endPreparing()
        do {
            metrics = try await withTaskCancellationHandler(
                operation: {
                    if Task.isCancelled || shouldStop?() == true { cancelFlag.cancel() }
                    return try await withCheckedThrowingContinuation { cont in
                        DispatchQueue.global(qos: .userInitiated).async {
                            do {
                                cont.resume(returning: try capture.modelBackend.generate(
                                    request: genRequest, config: config, onToken: tokenCallback
                                ))
                            } catch {
                                cont.resume(throwing: error)
                            }
                        }
                    }
                },
                onCancel: { cancelFlag.cancel() }
            )
        } catch {
            telemetry.resetToIdle()
            Self.recordChatOutcome(
                request: request, config: config, hasMountedManifest: memState.manifest != nil,
                promptTokens: effectivePromptTokens.count, generatedTokens: 0, outcome: "failed"
            )
            throw error
        }

        let finalState = capture.emitRemainderAndSnapshot()
        if let tailChunk = finalState.finalChunk { capture.onToken?(tailChunk) }

        var fullText = finalState.fullText
        if let fmtType = request.responseFormatType,
           (fmtType == "json_schema" || fmtType == "json_object"),
           !prepared.shouldThink,
           (prepared.isConstrainedJson || !fullText.contains("<tool_call>")),
           let startBrace = fullText.firstIndex(of: "{"),
           let endBrace = Self.firstCompleteJsonObjectEnd(in: String(fullText[startBrace...])) {
            let sub = String(fullText[startBrace...])
            fullText = String(sub[..<endBrace])
        }

        let totalDuration = Date().timeIntervalSince(inferStart)
        let ttftMs = (finalState.ttftSec ?? metrics.firstTokenSec).map { $0 * 1000.0 }
        let decodeSec = metrics.decodeSec
        let decodedTokens = metrics.tokens.count > 1 ? metrics.tokens.count - 1 : metrics.tokens.count
        let decodeTokPerSec = decodeSec.flatMap { $0 > 0 && decodedTokens > 0 ? Double(decodedTokens) / $0 : nil }

        if metrics.cacheActive {
            telemetry.recordCacheLookup(hitTier: metrics.cacheTier, matchedTokens: metrics.cacheHitTokens)
            telemetry.updateCacheUsage(
                warmBytes: metrics.warmCacheBytes, warmLimit: config.cacheConfig.warmByteLimit,
                ssdBytes: metrics.ssdCacheBytes, ssdLimit: config.cacheConfig.ssdByteLimit
            )
        }
        telemetry.endRequest(
            tokensIn: effectivePromptTokens.count,
            tokensOut: metrics.tokens.count,
            cachedTokens: metrics.cacheHitTokens,
            prefillSec: metrics.prefillSec,
            decodeSec: decodeSec,
            ttftSec: finalState.ttftSec ?? metrics.firstTokenSec
        )
        telemetry.updateContext(
            active: effectivePromptTokens.count + metrics.tokens.count,
            capacity: metrics.activeContextCapacity
        )
        logger.logRequest(
            method: "POST", path: "/v1/chat/completions", statusCode: 200,
            promptTokens: effectivePromptTokens.count, completionTokens: metrics.tokens.count,
            imagesCount: imageUrls.count, cacheHitTokens: metrics.cacheHitTokens,
            cacheTier: metrics.cacheTier, cacheActive: metrics.cacheActive,
            specSummary: metrics.speculativeSummary,
            ttftMs: ttftMs, decodeTokPerSec: decodeTokPerSec, totalSec: totalDuration
        )
        Self.recordChatOutcome(
            request: request, config: config, hasMountedManifest: memState.manifest != nil,
            promptTokens: effectivePromptTokens.count, generatedTokens: metrics.tokens.count,
            cachedTokens: metrics.cacheHitTokens, cacheHitTier: metrics.cacheTier,
            ttftMs: ttftMs, prefillMs: metrics.prefillSec.map { $0 * 1000.0 },
            decodeMs: decodeSec.map { $0 * 1000.0 }, decodeTps: decodeTokPerSec,
            stageGPU: metrics.stageGPU, outcome: "ok"
        )

        let reportedCompletionTokens = max(0, metrics.tokens.count - finalState.healedConsumedTokens)
        let truncated = !finalState.stoppedByCustomRule
            && (reportedCompletionTokens >= maxNew)
            && (metrics.tokens.last.map { !stops.contains($0) } ?? true)
        let logprobEntries: [ChatLogprobTokenEntry]? = collectLP
            ? metrics.tokenLogprobs.map { Self.makeLogprobEntry($0, tokenizer: tok) }
            : nil

        return ChatCompletionExecutionResult(
            content: fullText,
            promptTokens: reportedPromptTokenCount,
            completionTokens: reportedCompletionTokens,
            reasoningTokens: prepared.shouldThink ? finalState.reasoningTokenCount : nil,
            cachedTokens: metrics.cacheHitTokens,
            cacheActive: metrics.cacheActive,
            truncated: truncated,
            startsInThinking: prepared.shouldThink,
            matchedStopSequence: finalState.matchedStopSequence,
            logprobs: logprobEntries,
            prefillSec: metrics.prefillSec,
            decodeSec: metrics.decodeSec
        )
    }

    nonisolated private func prepareMultimodalInput(
        promptTokens: [Int],
        imageUrls: [String],
        loaded: LoadedState,
        config: Config
    ) async throws -> (tokens: [Int], visualSpans: [QWVisualSpan], loadedOnDemand: Bool, unloadedOnText: Bool) {
        if imageUrls.isEmpty && (loaded.visionBackend?.isLoaded != true || config.loadVision) {
            return (promptTokens, [], false, false)
        }
        let imagePolicy = QWImageDecodePolicy(
            allowDataURLs: true,
            allowLocalImagePaths: config.allowLocalImagePaths,
            allowRemoteImageURLs: config.allowRemoteImageURLs
        )
        let mmTurn = try await visionCoordinator.withLock {
            try await QWImageDecoder.prepareMultimodalTurn(
                promptTokens: promptTokens,
                imageUrls: imageUrls,
                hiddenDim: loaded.descriptor.hidden,
                visionBackend: loaded.visionBackend,
                policy: imagePolicy,
                onImageEncoded: { [logger] idx, total, h, w, ms in
                    logger.log(.info, String(format: "vision_encode_ms=%.1f image=%d/%d patches=%dx%d",
                                             ms, idx, total, h, w))
                }
            )
        }
        if mmTurn.loadedOnDemand {
            logger.log(.info, "vision_tower_loaded_on_demand=true")
        } else if mmTurn.unloadedOnText {
            logger.log(.info, "vision_tower_unloaded_on_text_prompt=true")
        }
        return mmTurn
    }

    private static func recordChatOutcome(
        request: ChatCompletionRequest,
        config: Config,
        hasMountedManifest: Bool,
        promptTokens: Int,
        generatedTokens: Int,
        cachedTokens: Int = 0,
        cacheHitTier: String? = nil,
        ttftMs: Double? = nil,
        prefillMs: Double? = nil,
        decodeMs: Double? = nil,
        decodeTps: Double? = nil,
        stageGPU: [String: Double] = [:],
        outcome: String
    ) {
        guard request.effectiveStore else { return }
        QWRequestRecordStore.shared.append(QWRequestRecord(
            id: request.effectiveRequestId ?? QWRequestRecordStore.shared.nextId(),
            time: Date(),
            modelId: config.packName,
            promptTokens: promptTokens,
            generatedTokens: generatedTokens,
            cachedTokens: cachedTokens,
            cacheHitTier: cacheHitTier,
            ttftMs: ttftMs,
            prefillMs: prefillMs,
            decodeMs: decodeMs,
            decodeTps: decodeTps,
            stageGPU: stageGPU,
            config: QWChatPolicyResolver.buildLedgerConfig(
                request: request, config: config, hasMountedManifest: hasMountedManifest
            ),
            outcome: outcome
        ))
    }
}
