// DriverBatchDecode+Interleaved.swift — Slice-interleaved prefill and Radix wave admission for QWEngineDriver.

import Foundation
import Metal
import QWCore
import QWHAL
import QWKV
import QWHead

enum QWBatchSlotPhase {
    case prefilling(
        cursor: Int,
        matchedTokens: Int,
        retainPoints: [Int],
        stochasticOwner: QWStochasticSession?,
        startedAt: Date,
        prefillMark: Int,
        prefillWallAcc: Double
    )
    case decoding
}

struct QWBatchSlotState {
    let job: QWBatchScheduledJob
    let session: QWEngineSession
    let maxNew: Int
    let stopIDs: Set<Int>
    let constraint: (any QWTokenConstraint)?
    let constraintSampler: QWSampler?
    let lpSampler: QWSampler?
    let topKCount: Int
    let speculativeEnabled: Bool
    let relaxedMarginOverride: Float?
    let relaxedMinProb: Double?
    var phase: QWBatchSlotPhase
    var output: QWBatchSlotOutputState
    var cacheAnswer: QWCacheAnswer?
    var logprobs: [QWTokenLogprob] = []
    var pendingToken: Int

    var isDecoding: Bool {
        if case .decoding = phase { return true }
        return false
    }
}

extension QWEngineDriver {
    static func resolveSlotRelaxedOverrides(
        request: QWGenerationRequest,
        defaultMinProb: Double
    ) -> (margin: Float?, minProb: Double?) {
        let reqBackend = request.speculative.mtpBackend?.lowercased()
        if request.sampling.hasExplicitSeed {
            return (0.0, nil)
        }
        if reqBackend == "variational" || reqBackend == "relaxed" || reqBackend == "plm" {
            return (Float(max(0.0, request.speculative.mtpRelaxedMargin)), request.speculative.mtpRelaxedMinProb)
        }
        if reqBackend == "standard" || reqBackend == "dflash2" || reqBackend == "dspine" {
            return (0.0, nil)
        }
        _ = defaultMinProb
        return (nil, nil)
    }

    /// Finds the deepest common prefix (`>= 128` tokens) shared by at least two cacheable jobs in an arriving wave.
    static func commonWavePrefixLength(for jobs: [QWBatchScheduledJob], minTokens: Int = 128) -> Int? {
        guard QWRadixKVGate.isEnabled, jobs.count >= 2 else { return nil }
        let eligible = jobs.filter { $0.request.usePrefixCache && $0.request.visualSpans.isEmpty }
        guard eligible.count >= 2 else { return nil }
        var best = 0
        for i in 0..<eligible.count {
            let a = eligible[i].request.prompt
            guard a.count > minTokens else { continue }
            for j in (i + 1)..<eligible.count {
                let b = eligible[j].request.prompt
                guard b.count > minTokens else { continue }
                let limit = min(a.count, b.count)
                var common = 0
                while common < limit && a[common] == b[common] { common += 1 }
                if common >= minTokens && common < a.count && common < b.count {
                    best = max(best, common)
                }
            }
        }
        return best >= minTokens ? best : nil
    }

    func admitBatchJobForScheduling(
        _ job: QWBatchScheduledJob,
        commonWavePrefix: Int?,
        isWaveLeader: Bool,
        hasDecodingPeers: Bool,
        permit: QWGenerationPermit
    ) throws -> QWBatchSlotState? {
        let req = job.request
        if req.usePrefixCache && !hasPrefixCache {
            throw QWEngineDriverError.state("the request wants a prefix cache this engine does not have")
        }
        guard QWBatchPartitionSelection.validateRequestBounds(
            prompt: req.prompt, maxNew: req.maxNew, stopIDs: req.stopIDs, vocab: geometry.vocab
        ) else {
            throw QWEngineDriverError.state("invalid batch generation request bounds or token IDs")
        }
        guard req.prompt.count < ropeRows else {
            throw QWEngineDriverError.posBeyondTable(req.prompt.count)
        }
        let effectiveMaxNew = QWBatchPartitionSelection.clampedMaxNew(
            requested: req.maxNew, promptCount: req.prompt.count, activeCapacity: ropeRows
        )
        if effectiveMaxNew == 0 {
            job.complete(.success(QWGenerationResult(tokens: [], stepMs: [], stepTokenCounts: [])))
            return nil
        }

        let started = Date()
        let pureGreedy = (req.sampling.isGreedy && req.sampling.repetitionPenalty == 1.0)
            || (req.constraint != nil && !(req.constraint?.allowsSpeculation ?? false))
        var stochConfig = req.sampling
        if !pureGreedy {
            stochConfig.uniformVersion = .upper24
            if stochConfig.temperature <= 0 {
                stochConfig.temperature = 1
                stochConfig.topK = 1
                stochConfig.topP = 1
            }
        }
        let stochasticOwner = pureGreedy ? nil : try QWStochasticSession(geometry: geometry, config: stochConfig)
        let saved = (activeCacheSalt, activeVisualSpans, activeConstraint, activeCollectLogprobs,
                     activePrefillProgress, activePrefillTotalTokens, activePrefillBasePosition)
        defer {
            (activeCacheSalt, activeVisualSpans, activeConstraint, activeCollectLogprobs,
             activePrefillProgress, activePrefillTotalTokens, activePrefillBasePosition) = saved
        }
        (activeCacheSalt, activeVisualSpans, activeConstraint) = (req.cacheSalt, req.visualSpans, nil)
        activeCollectLogprobs = req.collectLogprobs || req.constraint != nil || stochasticOwner != nil || req.usePrefixCache
        (activePrefillProgress, activePrefillTotalTokens, activePrefillBasePosition) = (req.onPrefillProgress, nil, 0)
        if req.collectTimings { prefillTimingLedger.setEnabled(true) }
        let prefillMark = prefillTimingLedger.values().count

        let session: QWEngineSession
        var exactHitFirst: Int? = nil
        var cacheMatched = 0
        var cacheTier: String? = nil
        if req.usePrefixCache, let store = prefixStore {
            let loaded = try prefixPagePressure(store) {
                try cachedAdmission(prompt: req.prompt, mode: gdnPrefillMode, store: store, permit: permit)
            }
            session = loaded.session; exactHitFirst = loaded.first
            cacheMatched = loaded.matched; cacheTier = loaded.tier
        } else {
            session = try maybePrefixPagePressure { try admitSession(promptTokens: req.prompt.count) }
        }

        let constraintSampler = req.constraint != nil ? QWSampler(config: req.sampling) : nil
        let lpSampler = req.collectLogprobs ? QWSampler(config: req.sampling) : nil
        let speculativeEnabled = QWBatchPartitionSelection.canEnableSpeculation(
            requested: req.speculative.enabled, mtpDrafts: mtpDrafts,
            collectLogprobs: req.collectLogprobs,
            allowsSpeculation: req.constraint?.allowsSpeculation ?? true,
            hasExplicitSeed: req.sampling.hasExplicitSeed, isGreedy: req.sampling.isGreedy
        )
        let relaxed = Self.resolveSlotRelaxedOverrides(request: req, defaultMinProb: mtpRelaxedMinProb)
        let cacheAnswer = req.usePrefixCache
            ? QWCacheAnswer(hit: cacheMatched > 0, matchedTokens: cacheMatched,
                            totalTokens: req.prompt.count, tier: cacheTier)
            : nil

        do {
            if let cachedFirst = exactHitFirst {
                let prefillElapsed = Date().timeIntervalSince(started)
                req.onPrefillProgress?(0, 0)
                if let stochasticOwner { session.stochastic = stochasticOwner }
                var slot = QWBatchSlotState(
                    job: job, session: session, maxNew: effectiveMaxNew, stopIDs: req.stopIDs,
                    constraint: req.constraint, constraintSampler: constraintSampler,
                    lpSampler: lpSampler, topKCount: max(0, min(20, req.topLogprobsCount)),
                    speculativeEnabled: speculativeEnabled,
                    relaxedMarginOverride: relaxed.margin, relaxedMinProb: relaxed.minProb,
                    phase: .decoding,
                    output: QWBatchSlotOutputState(
                        startedAt: started, prefillSeconds: prefillElapsed,
                        firstTokenSeconds: Date().timeIntervalSince(started),
                        prefillGPUSeconds: prefillGPUSecondsSince(mark: prefillMark)
                    ),
                    cacheAnswer: cacheAnswer,
                    pendingToken: cachedFirst
                )
                syncTokenHistoryAndRecycler(session, tokens: req.prompt, promptCount: req.prompt.count)
                try consumeBatchPredictedToken(
                    slot: &slot, rawArgmax: cachedFirst,
                    logitsTensor: sc.logits, logitBuf: batchScLogitsBuffer, stepMsValue: nil
                )
                if !slot.output.finished { slot.output.markDecodeStartedIfNeeded() }
                return slot
            }

            try prepareSuffixProgressAndWrites(
                session, matched: cacheMatched, promptCount: req.prompt.count,
                onProgress: req.onPrefillProgress, permit: permit
            )
            let mode = gdnPrefillMode
            let retainPoints: [Int]
            if req.usePrefixCache, let store = prefixStore, req.visualSpans.isEmpty {
                let policyKey = prefixPolicyKey(mode: mode)
                let periodic = (QWRadixKVGate.isEnabled && !canSlicePureAttentionPages && mode != .naive) ? 4096 : 0
                retainPoints = QWTwoTierPrefixCache.computeRetainPoints(
                    explicitRetainLength: req.retainPrefixLength,
                    matchedTokens: cacheMatched,
                    firstVisualTokenOffset: req.prompt.count,
                    promptCount: req.prompt.count,
                    sharedBranchPrefixLength: store.sharedBranchPrefixLength(for: req.prompt, mode: policyKey),
                    commonWavePrefixLength: commonWavePrefix,
                    periodicCheckpointInterval: periodic,
                    includeTrailingAutoRetain: false
                )
            } else {
                retainPoints = []
            }

            let sliceBudget = QWInterleavedPrefillGate.sliceTokens
            let canSliceInterleave = sliceBudget > 0 && mode != .naive && req.visualSpans.isEmpty

            if !canSliceInterleave {
                let tPrefill = Date()
                let unconditionedFirst = try prefillWithRetainPoints(
                    session: session, prompt: req.prompt, retainPoints: retainPoints,
                    mode: mode, store: req.usePrefixCache ? prefixStore : nil, permit: permit
                )
                let prefillElapsed = Date().timeIntervalSince(tPrefill)
                if req.usePrefixCache, let store = prefixStore, req.visualSpans.isEmpty,
                   !(req.retainPrefixLength != nil && req.retainPrefixLength! > cacheMatched && req.retainPrefixLength! < req.prompt.count) {
                    _ = try? retainCachedPrompt(session, prompt: req.prompt, mode: mode,
                                                first: unconditionedFirst, store: store, permit: permit)
                }
                if let stochasticOwner { session.stochastic = stochasticOwner }
                var slot = QWBatchSlotState(
                    job: job, session: session, maxNew: effectiveMaxNew, stopIDs: req.stopIDs,
                    constraint: req.constraint, constraintSampler: constraintSampler,
                    lpSampler: lpSampler, topKCount: max(0, min(20, req.topLogprobsCount)),
                    speculativeEnabled: speculativeEnabled,
                    relaxedMarginOverride: relaxed.margin, relaxedMinProb: relaxed.minProb,
                    phase: .decoding,
                    output: QWBatchSlotOutputState(
                        startedAt: started, prefillSeconds: prefillElapsed,
                        firstTokenSeconds: Date().timeIntervalSince(started),
                        prefillGPUSeconds: prefillGPUSecondsSince(mark: prefillMark)
                    ),
                    cacheAnswer: cacheAnswer,
                    pendingToken: unconditionedFirst
                )
                syncTokenHistoryAndRecycler(session, tokens: req.prompt, promptCount: req.prompt.count)
                try consumeBatchPredictedToken(
                    slot: &slot, rawArgmax: unconditionedFirst,
                    logitsTensor: sc.logits, logitBuf: batchScLogitsBuffer, stepMsValue: nil
                )
                if !slot.output.finished { slot.output.markDecodeStartedIfNeeded() }
                return slot
            }

            for t in req.prompt where t < 0 || t >= geometry.vocab {
                throw QWEngineDriverError.badToken(t)
            }
            syncTokenHistoryAndRecycler(session, tokens: req.prompt, promptCount: req.prompt.count)

            var prefillWallAcc: Double = 0
            // When leading a sibling wave with no active decode peers, materialize the shared wave checkpoint
            // immediately so sibling jobs admitted right behind this leader hit the Radix node without recomputing it.
            if isWaveLeader, !hasDecodingPeers,
               let waveLen = commonWavePrefix,
               waveLen > session.position, waveLen < req.prompt.count,
               let store = prefixStore, req.usePrefixCache {
                let tSeg = Date()
                let midFirst = try prefill(
                    session, tokens: Array(req.prompt[session.position..<waveLen]),
                    mode: mode, permit: permit, evictScratchOnCompletion: false
                )
                prefillWallAcc += Date().timeIntervalSince(tSeg)
                _ = try? retainCachedPrompt(
                    session, prompt: Array(req.prompt.prefix(waveLen)),
                    mode: mode, first: midFirst, store: store, permit: permit, pinToSlot: false
                )
                resetIntermediatePrediction(session)
                session.tokenHistory = req.prompt
                session.promptTokenCount = req.prompt.count
                try prepareSuffixKVCacheWrites(session, store: store, from: waveLen, to: req.prompt.count, permit: permit)
            }

            let filteredRetains = retainPoints.filter { $0 > session.position && $0 < req.prompt.count }
            var slot = QWBatchSlotState(
                job: job, session: session, maxNew: effectiveMaxNew, stopIDs: req.stopIDs,
                constraint: req.constraint, constraintSampler: constraintSampler,
                lpSampler: lpSampler, topKCount: max(0, min(20, req.topLogprobsCount)),
                speculativeEnabled: speculativeEnabled,
                relaxedMarginOverride: relaxed.margin, relaxedMinProb: relaxed.minProb,
                phase: .prefilling(
                    cursor: session.position,
                    matchedTokens: cacheMatched,
                    retainPoints: filteredRetains,
                    stochasticOwner: stochasticOwner,
                    startedAt: started,
                    prefillMark: prefillMark,
                    prefillWallAcc: prefillWallAcc
                ),
                output: QWBatchSlotOutputState(startedAt: started),
                cacheAnswer: cacheAnswer,
                pendingToken: 0
            )
            // Short prompts without active decode peers complete their single chunk immediately on admission.
            let remaining = req.prompt.count - session.position
            if !hasDecodingPeers && filteredRetains.isEmpty && remaining <= min(sliceBudget, psc.maxChunk) {
                try stepPrefillingSlot(&slot, hasDecodingPeers: false, permit: permit)
            }
            return slot
        } catch {
            finishSession(session)
            throw error
        }
    }

    private func prefillWithRetainPoints(
        session: QWEngineSession,
        prompt: [Int],
        retainPoints: [Int],
        mode: GDNPrefillMode,
        store: QWEnginePrefixStore<QWEngineCachedPrefix>?,
        permit: QWGenerationPermit
    ) throws -> Int {
        if let store, !retainPoints.isEmpty {
            for retainLen in retainPoints where retainLen > session.position && retainLen < prompt.count {
                let midFirst = try prefill(
                    session, tokens: Array(prompt[session.position..<retainLen]),
                    mode: mode, permit: permit, evictScratchOnCompletion: false
                )
                _ = try? retainCachedPrompt(
                    session, prompt: Array(prompt.prefix(retainLen)),
                    mode: mode, first: midFirst, store: store, permit: permit
                )
                resetIntermediatePrediction(session)
                session.tokenHistory = Array(prompt.prefix(retainLen))
                try prepareSuffixKVCacheWrites(session, store: store, from: retainLen, to: prompt.count, permit: permit)
            }
        }
        return try prefill(session, tokens: Array(prompt[session.position..<prompt.count]), mode: mode, permit: permit)
    }

    /// Advances a `.prefilling` slot by one slice (`512` tokens when decode peers are active; full chunk when solo).
    func stepPrefillingSlot(
        _ slot: inout QWBatchSlotState,
        hasDecodingPeers: Bool,
        permit: QWGenerationPermit
    ) throws {
        guard case .prefilling(
            let cursor,
            let matchedTokens,
            var retainPoints,
            let stochasticOwner,
            let startedAt,
            let prefillMark,
            var prefillWallAcc
        ) = slot.phase else { return }

        let req = slot.job.request
        let prompt = req.prompt
        let totalCount = prompt.count
        guard cursor < totalCount else { return }

        let mode = gdnPrefillMode
        let sliceCap: Int
        if hasDecodingPeers {
            let configured = QWInterleavedPrefillGate.sliceTokens
            sliceCap = max(32, min(psc.maxChunk, configured > 0 ? configured : QWInterleavedPrefillGate.defaultSliceTokens))
        } else {
            let adaptiveCap = Self.adaptivePrefillChunkCap(
                position: slot.session.position,
                maxChunk: psc.maxChunk,
                isSubquadraticAttention: usesSubquadraticLongPrefill
            )
            sliceCap = Self.nextBalancedChunkCount(remaining: totalCount - cursor, stepCap: adaptiveCap)
        }

        while let firstRetain = retainPoints.first, firstRetain <= cursor {
            retainPoints.removeFirst()
        }
        let targetBoundary = retainPoints.first.map { min($0, totalCount) } ?? totalCount
        let count = max(1, min(sliceCap, targetBoundary - cursor))
        let newCursor = cursor + count
        let reachesRetain = (retainPoints.first == newCursor) && newCursor < totalCount
        let reachesEnd = (newCursor == totalCount)
        let runsHead = reachesRetain || reachesEnd

        let saved = (activeCacheSalt, activeVisualSpans, activeConstraint, activeCollectLogprobs,
                     activePrefillProgress, activePrefillTotalTokens, activePrefillBasePosition)
        defer {
            (activeCacheSalt, activeVisualSpans, activeConstraint, activeCollectLogprobs,
             activePrefillProgress, activePrefillTotalTokens, activePrefillBasePosition) = saved
        }
        (activeCacheSalt, activeVisualSpans, activeConstraint) = (req.cacheSalt, req.visualSpans, nil)
        activeCollectLogprobs = runsHead && (req.collectLogprobs || req.constraint != nil || stochasticOwner != nil || req.usePrefixCache)
        activePrefillProgress = req.onPrefillProgress
        activePrefillBasePosition = matchedTokens
        activePrefillTotalTokens = max(0, totalCount - matchedTokens)

        if geometry.sharedExpertIntermediate > 0 && !geometry.sharedExpertHasGate, let s = psc.moe {
            let gPtr = s.gate.region.pointer.advanced(by: s.gate.offset).assumingMemoryBound(to: Float.self)
            for i in 0..<min(count, psc.maxChunk) { gPtr[i] = 1.0 }
        }

        let tSlice = Date()
        let rawToken = try maybePrefixPagePressure(session: slot.session) {
            try prefillChunk(
                slot.session, tokens: prompt, start: cursor, count: count,
                runsHead: runsHead, mode: mode, permit: permit, dflashWindowEnd: totalCount
            )
        }
        prefillWallAcc += Date().timeIntervalSince(tSlice)
        let doneTokens = min(activePrefillTotalTokens ?? totalCount, max(0, newCursor - matchedTokens))
        req.onPrefillProgress?(doneTokens, activePrefillTotalTokens ?? totalCount)

        if reachesRetain {
            retainPoints.removeFirst()
            if req.usePrefixCache, let store = prefixStore {
                _ = try? retainCachedPrompt(
                    slot.session, prompt: Array(prompt.prefix(newCursor)),
                    mode: mode, first: rawToken, store: store, permit: permit
                )
                resetIntermediatePrediction(slot.session)
                slot.session.tokenHistory = prompt
                slot.session.promptTokenCount = totalCount
                try prepareSuffixKVCacheWrites(slot.session, store: store, from: newCursor, to: totalCount, permit: permit)
            }
        }

        guard reachesEnd else {
            slot.phase = .prefilling(
                cursor: newCursor,
                matchedTokens: matchedTokens,
                retainPoints: retainPoints,
                stochasticOwner: stochasticOwner,
                startedAt: startedAt,
                prefillMark: prefillMark,
                prefillWallAcc: prefillWallAcc
            )
            return
        }

        if prefillScratchEviction.shouldEvict(wiredEngineBytes: arena.committedBytes) {
            evictPrefillScratch()
        }
        let needsFullRetain = !(req.retainPrefixLength != nil
            && req.retainPrefixLength! > matchedTokens
            && req.retainPrefixLength! < totalCount)
        if req.usePrefixCache, needsFullRetain, let store = prefixStore, req.visualSpans.isEmpty {
            _ = try? retainCachedPrompt(
                slot.session, prompt: prompt, mode: mode,
                first: rawToken, store: store, permit: permit
            )
        }
        if let stochasticOwner { slot.session.stochastic = stochasticOwner }
        slot.output.prefillSeconds = prefillWallAcc
        slot.output.firstTokenSeconds = Date().timeIntervalSince(startedAt)
        slot.output.prefillGPUSeconds = prefillGPUSecondsSince(mark: prefillMark)
        slot.pendingToken = rawToken
        slot.phase = .decoding
        syncTokenHistoryAndRecycler(slot.session, tokens: prompt, promptCount: totalCount)
        try consumeBatchPredictedToken(
            slot: &slot, rawArgmax: rawToken,
            logitsTensor: sc.logits, logitBuf: batchScLogitsBuffer, stepMsValue: nil
        )
        if !slot.output.finished { slot.output.markDecodeStartedIfNeeded() }
    }
}
