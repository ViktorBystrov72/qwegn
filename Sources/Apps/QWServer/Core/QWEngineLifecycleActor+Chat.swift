// QWEngineLifecycleActor+Chat.swift — the inference handler the
// HTTP routes call. Split out so the actor file stays under the
// repository's file-size rule.
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

public struct ChatCompletionExecutionResult: Sendable {
    public let content: String
    public let promptTokens: Int
    public let completionTokens: Int
    public let reasoningTokens: Int?
    public let cachedTokens: Int
    public let truncated: Bool
    public let startsInThinking: Bool
    public let matchedStopSequence: String?
    public let logprobs: [ChatLogprobTokenEntry]?
    public let prefillSec: Double?
    public let decodeSec: Double?
}

extension QWEngineLifecycleActor {

    /// Dynamically loads or unloads the vision tower in place without unloading the LLM or resetting the KV cache.
    public func setVisionLoaded(_ loadVision: Bool) throws {
        config.loadVision = loadVision
        statusBox.updateConfig(config)
        if case .loaded(let loaded) = state {
            try loaded.visionBackend?.setVisionLoaded(loadVision)
            logger.log(.info, "Vision tower \(loadVision ? "loaded into" : "unloaded from") memory in place (KV cache preserved).")
        }
    }

    nonisolated public func handleChatCompletion(
        request: ChatCompletionRequest,
        onToken: ((String) -> Void)? = nil,
        onTokenLogprob: ((ChatLogprobTokenEntry) -> Void)? = nil,
        shouldStop: (@Sendable () -> Bool)? = nil
    ) async throws -> ChatCompletionExecutionResult {
        try await selectModelIfNeeded(request.model, allowSameFamilyQuantSwitch: false)
        await beginInFlightRequest()
        do {
            let result = try await executeChatCompletion(
                request: request,
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
        onToken: ((String) -> Void)?,
        onTokenLogprob: ((ChatLogprobTokenEntry) -> Void)?,
        shouldStop: (@Sendable () -> Bool)?
    ) async throws -> ChatCompletionExecutionResult {
        let memState = await autoMountAndPrepareMemoryState(for: request)
        let loaded = try await ensureLoaded()
        let config = getConfig()
        QWConcurrentBatchTracker.beginPreparing()
        var preparingActive = true
        defer {
            if preparingActive { QWConcurrentBatchTracker.endPreparing() }
        }
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
            memoryManifest: memState.manifest
        )
        let shouldThink = prepared.shouldThink
        let isConstrainedJson = prepared.isConstrainedJson
        let wire = prepared.wire

        let usePrefixCache = request.effectiveCachePrompt
        let cacheSalt = request.effectiveCacheSalt
        let encodedWire = try await encodeWireIncrementallyWithSystemPrefix(
            wire: wire, tokenizer: tok, packPath: config.packPath,
            cachePrompt: usePrefixCache, cacheSalt: cacheSalt
        )
        let promptTokens = encodedWire.tokens
        let systemPrefixTokenCount = encodedWire.systemPrefixTokenCount
        let imagePolicy = QWImageDecodePolicy(
            allowDataURLs: true,
            allowLocalImagePaths: config.allowLocalImagePaths,
            allowRemoteImageURLs: config.allowRemoteImageURLs
        )
        let mmTurn = try await QWImageDecoder.prepareMultimodalTurn(
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
        if mmTurn.loadedOnDemand {
            logger.log(.info, "vision_tower_loaded_on_demand=true")
        } else if mmTurn.unloadedOnText {
            logger.log(.info, "vision_tower_unloaded_on_text_prompt=true")
        }
        let visualSpans = mmTurn.visualSpans
        let requestedMaxNew = request.maxTokens ?? request.maxCompletionTokens
            ?? config.modelMaxLength.map { max(0, $0 - mmTurn.tokens.count) }
            ?? config.maxTokens
        let budgetGuard = QWMemoryVault.applyElasticBudgetGuard(
            mountedMemoryTokens: memState.mountedTokens,
            deltaTokens: memState.deltaTokens,
            agentPromptTokens: mmTurn.tokens.count,
            maxNewTokens: requestedMaxNew,
            maxContext: config.maxContext
        )
        if budgetGuard.bypassedOversizedCache {
            logger.log(.info, "[MemoryPack] Elastic budget guard bypassed oversized .qwcache (\(memState.mountedTokens.count) tokens) to preserve \(mmTurn.tokens.count)-token agent prompt within maxContext=\(config.maxContext)")
        }
        let memoryPrefixTokens = budgetGuard.prefixTokens
        let effectivePromptTokens: [Int]
        let effectiveSystemPrefixTokenCount: Int?
        if visualSpans.isEmpty && !memoryPrefixTokens.isEmpty {
            effectivePromptTokens = memoryPrefixTokens + mmTurn.tokens
            effectiveSystemPrefixTokenCount = systemPrefixTokenCount.map { memoryPrefixTokens.count + $0 }
        } else {
            effectivePromptTokens = mmTurn.tokens
            effectiveSystemPrefixTokenCount = systemPrefixTokenCount
        }

        var stops = prepared.stops
        let fmtType = request.responseFormatType
        let hasActiveTools = prepared.hasActiveTools
        let imEndTokenId = prepared.imEndTokenId
        let endThinkTokenId = prepared.endThinkTokenId
        let userTextContainsImEnd = prepared.userTextContainsImEnd
        let jsonConstraint = prepared.jsonConstraint
        let customStopStrings = prepared.customStopStrings
        let reportedPromptTokenCount = effectivePromptTokens.count
        let heal = Self.prepareTokenHeal(
            prepared: prepared,
            request: request,
            promptTokens: effectivePromptTokens,
            systemPrefixTokenCount: effectiveSystemPrefixTokenCount,
            stops: stops,
            tokenizer: tok,
            vocabulary: loaded.descriptor.vocab,
            supportsTokenConstraints: loaded.modelBackend.supportsTokenConstraints,
            hasImages: !visualSpans.isEmpty
        )
        let genPromptTokens = heal.promptTokens
        let activeGenConstraint = heal.constraint

        final class StreamCapture: @unchecked Sendable {
            private static let endThinkUTF8 = Array("</think>".utf8)
            let tok: QWTokenizer
            var streamDecoder: QWStreamDecoder
            var fullText: String
            var fullUTF8: [UInt8]
            var emittedUTF8Count: Int = 0
            var pendingPrefix: String
            var remainingHealUTF8: Int
            var healedConsumedTokens: Int = 0
            var hasReceivedPiece: Bool = false
            var ttftSec: Double? = nil
            var inThinking: Bool
            var reasoningTokenCount: Int = 0
            var matchedStopSequence: String? = nil
            var stoppedByCustomRule: Bool = false
            let onToken: ((String) -> Void)?
            let onTokenLogprob: ((ChatLogprobTokenEntry) -> Void)?
            let modelBackend: any QWModelBackend
            init(tok: QWTokenizer, initialPrefix: String, initialHealUTF8: Int,
                 startsInThinking: Bool,
                 onToken: ((String) -> Void)?,
                 onTokenLogprob: ((ChatLogprobTokenEntry) -> Void)?,
                 modelBackend: any QWModelBackend) {
                self.tok = tok
                self.streamDecoder = QWStreamDecoder(tok)
                self.fullText = initialPrefix
                self.fullUTF8 = Array(initialPrefix.utf8)
                self.pendingPrefix = initialPrefix
                self.remainingHealUTF8 = initialHealUTF8
                self.inThinking = startsInThinking
                self.onToken = onToken
                self.onTokenLogprob = onTokenLogprob
                self.modelBackend = modelBackend
            }

            func appendPiece(_ piece: String) {
                fullText += piece
                fullUTF8.append(contentsOf: piece.utf8)
            }

            func truncateFullText(to endIndex: String.Index) {
                fullText = String(fullText[..<endIndex])
                fullUTF8 = Array(fullText.utf8)
            }

            func containsEndThinkNearTail(lastPieceUTF8Count: Int) -> Bool {
                let needle = Self.endThinkUTF8
                guard fullUTF8.count >= needle.count else { return false }
                let start = max(0, fullUTF8.count - lastPieceUTF8Count - needle.count)
                let lastStart = fullUTF8.count - needle.count
                guard start <= lastStart else { return false }
                for i in start...lastStart where fullUTF8[i] == needle[0] {
                    var match = true
                    for j in 1..<needle.count where fullUTF8[i + j] != needle[j] {
                        match = false
                        break
                    }
                    if match { return true }
                }
                return false
            }

            func emitUpTo(_ utf8Limit: Int) {
                guard let onToken else {
                    emittedUTF8Count = utf8Limit
                    return
                }
                guard utf8Limit > emittedUTF8Count, utf8Limit <= fullUTF8.count else { return }
                if let str = String(bytes: fullUTF8[emittedUTF8Count..<utf8Limit], encoding: .utf8) {
                    emittedUTF8Count = utf8Limit
                    if !str.isEmpty { onToken(str) }
                }
            }
        }
        let capture = StreamCapture(
            tok: tok,
            initialPrefix: isConstrainedJson ? "{" : "",
            initialHealUTF8: heal.stripUTF8Count,
            startsInThinking: shouldThink,
            onToken: onToken,
            onTokenLogprob: onTokenLogprob,
            modelBackend: loaded.modelBackend
        )
        let maxNew = request.maxTokens ?? request.maxCompletionTokens
            ?? config.modelMaxLength.map { max(0, $0 - reportedPromptTokenCount) }
            ?? config.maxTokens
        let genMaxNew = heal.didPeel ? (maxNew + 1) : maxNew
        if request.ignoreEos == true || (request.minTokens ?? 0) >= maxNew {
            stops.removeAll()
        }

        let telemetry = QWEngineTelemetryRegistry.shared
        telemetry.beginPrefill(promptTokens: genPromptTokens.count)
        let promptTokCount = genPromptTokens.count
        let stopOnSingleTool = (request.parallelToolCalls == false) && hasActiveTools
        let cancelFlag = QWGenerationCancelFlag()

        let tokenCallback: @Sendable (Int) -> Bool = { id in
            if cancelFlag.isCancelled || shouldStop?() == true {
                capture.stoppedByCustomRule = true
                return false
            }
            if capture.ttftSec == nil {
                let firstElapsed = max(0.0001, Date().timeIntervalSince(inferStart))
                capture.ttftSec = firstElapsed
                telemetry.beginDecode(promptTokens: promptTokCount)
            } else {
                telemetry.recordDecodeToken()
            }
            if capture.inThinking {
                capture.reasoningTokenCount += 1
                if id == endThinkTokenId {
                    capture.inThinking = false
                }
            }
            var piece = ""
            if userTextContainsImEnd && id == imEndTokenId {
                let lastOpen = capture.fullText.range(of: "<function=", options: .backwards)
                let lastClose = capture.fullText.range(of: "</function>", options: .backwards)
                let insideOpenToolFunction: Bool
                if let lo = lastOpen {
                    insideOpenToolFunction = lastClose.map { lo.lowerBound > $0.lowerBound } ?? true
                } else {
                    insideOpenToolFunction = false
                }
                let insideOpenJson = isConstrainedJson && (Self.firstCompleteJsonObjectEnd(in: capture.fullText) == nil)
                if insideOpenToolFunction || insideOpenJson {
                    piece = "<|im_end|>"
                } else {
                    capture.stoppedByCustomRule = true
                    return false
                }
            } else if let decoded = try? capture.streamDecoder.decode(id, skipSpecialTokens: true) {
                piece = decoded
            }
            guard !piece.isEmpty else { return true }

            if capture.remainingHealUTF8 > 0 {
                let pBytes = Array(piece.utf8)
                if pBytes.count <= capture.remainingHealUTF8 {
                    capture.remainingHealUTF8 -= pBytes.count
                    capture.healedConsumedTokens += 1
                    return true
                }
                let tail = Array(pBytes.dropFirst(capture.remainingHealUTF8))
                capture.remainingHealUTF8 = 0
                piece = String(decoding: tail, as: UTF8.self)
                guard !piece.isEmpty else { return true }
            }

            if !capture.hasReceivedPiece {
                capture.hasReceivedPiece = true
                if isConstrainedJson {
                    let trimmed = piece.drop(while: { $0.isWhitespace })
                    if trimmed.hasPrefix("{") {
                        piece = String(trimmed.dropFirst())
                    }
                }
            }
            capture.appendPiece(piece)
            if capture.inThinking && capture.containsEndThinkNearTail(lastPieceUTF8Count: piece.utf8.count) {
                capture.inThinking = false
            }

            if !customStopStrings.isEmpty {
                var earliestRange: Range<String.Index>? = nil
                var matchedStop: String? = nil
                for stopStr in customStopStrings {
                    if let r = capture.fullText.range(of: stopStr),
                       earliestRange.map({ r.lowerBound < $0.lowerBound }) ?? true {
                        earliestRange = r
                        matchedStop = stopStr
                    }
                }
                if let range = earliestRange, let matchedStop {
                    capture.matchedStopSequence = matchedStop
                    capture.stoppedByCustomRule = true
                    capture.truncateFullText(to: range.lowerBound)
                    capture.emitUpTo(capture.fullUTF8.count)
                    return false
                }
            }

            if let jsonConstraint, jsonConstraint.isComplete {
                capture.stoppedByCustomRule = true
                capture.emitUpTo(capture.fullUTF8.count)
                return false
            }

            if isConstrainedJson, let jsonEnd = Self.firstCompleteJsonObjectEnd(in: capture.fullText) {
                capture.truncateFullText(to: jsonEnd)
                capture.stoppedByCustomRule = true
                capture.emitUpTo(capture.fullUTF8.count)
                return false
            }

            if stopOnSingleTool, let closeEnd = QWToolCallParser.firstCompleteToolCallEnd(in: capture.fullText) {
                capture.truncateFullText(to: closeEnd)
                capture.stoppedByCustomRule = true
                capture.emitUpTo(capture.fullUTF8.count)
                return false
            }

            if customStopStrings.isEmpty {
                capture.emitUpTo(capture.fullUTF8.count)
            } else {
                let hold = Self.heldStopSuffixUTF8(capture.fullText, stops: customStopStrings)
                let safeUTF8 = max(0, capture.fullUTF8.count - hold)
                capture.emitUpTo(safeUTF8)
            }
            return true
        }

        let collectLP = request.logprobs == true
        let topLPCount = min(max(0, request.topLogprobs ?? 0), 20)
        var logprobCallback: (@Sendable (QWTokenLogprob) -> Void)? = nil
        if collectLP && capture.onTokenLogprob != nil {
            logprobCallback = { (lp: QWTokenLogprob) in
                let entry = Self.makeLogprobEntry(lp, tokenizer: capture.tok)
                capture.onTokenLogprob?(entry)
            }
        }

        let sampling = config.samplingParameters.merging(
            temperature: request.temperature, topP: request.topP, topK: request.topK,
            repetitionPenalty: request.repetitionPenalty, seed: request.seed
        )
        let requestCarriesSampling = request.temperature != nil || request.topP != nil
            || request.topK != nil || request.repetitionPenalty != nil || request.seed != nil
        logSamplingSelectionIfChanged(sampling.denseNaxSelectionLogMessage(
            paramsSource: requestCarriesSampling ? "request override"
                : Self.samplingParamsSourceLabel(config.sourceSummary)))

        let firstVisualOffset = visualSpans.map(\.tokenOffset).min() ?? genPromptTokens.count
        let retainPrefixLen: Int? = (usePrefixCache
            && (effectiveSystemPrefixTokenCount ?? 0) >= 128
            && (effectiveSystemPrefixTokenCount ?? 0) < firstVisualOffset)
            ? effectiveSystemPrefixTokenCount
            : nil
        let prefillProgressCallback: @Sendable (Int, Int) -> Void = { completed, total in
            telemetry.updatePrefillProgress(completedTokens: completed, totalTokens: total)
        }
        let genRequest = QWModelGenerationRequest(
            promptTokens: genPromptTokens,
            maxNewTokens: genMaxNew,
            sampling: sampling,
            stopTokenIds: stops,
            retainPrefixLength: retainPrefixLen,
            hasImages: !visualSpans.isEmpty,
            collectLogprobs: collectLP,
            topLogprobsCount: topLPCount,
            onTokenLogprob: logprobCallback,
            constraint: activeGenConstraint,
            exactSpeculation: isConstrainedJson || sampling.hasExplicitSeed || request.effectiveExactSpeculation,
            usePrefixCache: usePrefixCache,
            cacheSalt: cacheSalt,
            visualSpans: visualSpans,
            onPrefillProgress: prefillProgressCallback
        )

        let metrics: QWModelCompletionMetrics
        preparingActive = false
        QWConcurrentBatchTracker.endPreparing()
        do {
            metrics = try await withTaskCancellationHandler(
                operation: {
                    if Task.isCancelled || shouldStop?() == true {
                        cancelFlag.cancel()
                    }
                    return try await withCheckedThrowingContinuation { cont in
                        DispatchQueue.global(qos: .userInitiated).async {
                            do {
                                let m = try capture.modelBackend.generate(
                                    request: genRequest,
                                    config: config,
                                    onToken: tokenCallback
                                )
                                cont.resume(returning: m)
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
            if request.effectiveStore {
                QWRequestRecordStore.shared.append(QWRequestRecord(
                    id: request.effectiveRequestId ?? QWRequestRecordStore.shared.nextId(),
                    time: Date(),
                    modelId: config.packName,
                    promptTokens: effectivePromptTokens.count,
                    generatedTokens: 0,
                    config: Self.buildLedgerConfig(
                        request: request, config: config, hasMountedManifest: memState.manifest != nil
                    ),
                    outcome: "failed"
                ))
            }
            throw error
        }

        let flushTail = capture.streamDecoder.flush()
        if !flushTail.isEmpty && !capture.stoppedByCustomRule {
            capture.appendPiece(flushTail)
        }
        if !customStopStrings.isEmpty && capture.matchedStopSequence == nil {
            for stopStr in customStopStrings {
                if let r = capture.fullText.range(of: stopStr) {
                    capture.matchedStopSequence = stopStr
                    capture.stoppedByCustomRule = true
                    capture.truncateFullText(to: r.lowerBound)
                    break
                }
            }
        }
        capture.emitUpTo(capture.fullUTF8.count)

        var fullText = capture.fullText
        if let fmtType, (fmtType == "json_schema" || fmtType == "json_object"),
           (isConstrainedJson || !fullText.contains("<tool_call>")),
           let startBrace = fullText.firstIndex(of: "{"),
           let endBrace = Self.firstCompleteJsonObjectEnd(in: String(fullText[startBrace...])) {
            let sub = String(fullText[startBrace...])
            fullText = String(sub[..<endBrace])
        }
        let ttftSec = capture.ttftSec

        let totalDuration = Date().timeIntervalSince(inferStart)
        let ttftMs = (ttftSec ?? metrics.firstTokenSec).map { $0 * 1000.0 }
        let decodeSec = metrics.decodeSec
        let decodedTokens = metrics.tokens.count > 1 ? metrics.tokens.count - 1 : metrics.tokens.count
        let decodeTokPerSec = decodeSec.flatMap { $0 > 0 && decodedTokens > 0 ? Double(decodedTokens) / $0 : nil }

        if metrics.cacheActive {
            telemetry.recordCacheLookup(hitTier: metrics.cacheTier, matchedTokens: metrics.cacheHitTokens)
            telemetry.updateCacheUsage(
                warmBytes: metrics.warmCacheBytes,
                warmLimit: config.cacheConfig.warmByteLimit,
                ssdBytes: metrics.ssdCacheBytes,
                ssdLimit: config.cacheConfig.ssdByteLimit
            )
        }
        telemetry.endRequest(
            tokensIn: effectivePromptTokens.count,
            tokensOut: metrics.tokens.count,
            cachedTokens: metrics.cacheHitTokens,
            prefillSec: metrics.prefillSec,
            decodeSec: decodeSec,
            ttftSec: ttftSec ?? metrics.firstTokenSec
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

        if request.effectiveStore {
            QWRequestRecordStore.shared.append(QWRequestRecord(
                id: request.effectiveRequestId ?? QWRequestRecordStore.shared.nextId(),
                time: Date(),
                modelId: config.packName,
                promptTokens: effectivePromptTokens.count,
                generatedTokens: metrics.tokens.count,
                cachedTokens: metrics.cacheHitTokens,
                cacheHitTier: metrics.cacheTier,
                ttftMs: ttftMs,
                prefillMs: metrics.prefillSec.map { $0 * 1000.0 },
                decodeMs: metrics.decodeSec.map { $0 * 1000.0 },
                decodeTps: decodeTokPerSec,
                stageGPU: metrics.stageGPU,
                config: Self.buildLedgerConfig(
                    request: request, config: config, hasMountedManifest: memState.manifest != nil
                ),
                outcome: "ok"
            ))
        }

        let reportedCompletionTokens = max(0, metrics.tokens.count - capture.healedConsumedTokens)
        let truncated = !capture.stoppedByCustomRule
            && (reportedCompletionTokens >= maxNew)
            && (metrics.tokens.last.map { !stops.contains($0) } ?? true)
        let logprobEntries: [ChatLogprobTokenEntry]? = collectLP
            ? metrics.tokenLogprobs.map { Self.makeLogprobEntry($0, tokenizer: tok) }
            : nil
        let reasoningToks: Int? = shouldThink ? capture.reasoningTokenCount : nil

        return ChatCompletionExecutionResult(
            content: fullText,
            promptTokens: reportedPromptTokenCount,
            completionTokens: reportedCompletionTokens,
            reasoningTokens: reasoningToks,
            cachedTokens: metrics.cacheHitTokens,
            truncated: truncated,
            startsInThinking: shouldThink,
            matchedStopSequence: capture.matchedStopSequence,
            logprobs: logprobEntries,
            prefillSec: metrics.prefillSec,
            decodeSec: metrics.decodeSec
        )
    }

    private static func buildLedgerConfig(
        request: ChatCompletionRequest,
        config: QWEngineLifecycleActor.Config,
        hasMountedManifest: Bool
    ) -> [String: String] {
        var out: [String: String] = [
            "kvMode": String(describing: config.kvMode),
            "mtpDrafts": String(config.mtpDrafts),
            "speculative": config.speculativeDecodingEnabled ? "on" : "off",
            "cachePrompt": request.effectiveCachePrompt ? "on" : "off",
            "memPack": hasMountedManifest ? "mounted" : (request.effectiveUseMemoryPack ? "auto" : "off")
        ]
        if let sid = request.effectiveSessionId { out["sessionId"] = sid }
        if let preset = request.effectivePresetId { out["preset"] = preset }
        if let salt = request.effectiveCacheSalt { out["cacheKey"] = salt }
        if request.effectiveExactSpeculation { out["exactSpec"] = "on" }
        if let user = request.effectiveUser { out["user"] = user }
        return out
    }
}
