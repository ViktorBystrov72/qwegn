// DriverBatchDecode.swift — Multi-session continuous batched decode for QWEngineDriver.

import Foundation
import Metal
import ObjectiveC
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWKV
import QWHead
import QWMTP

/// Preallocated 4-row head output and argmax scratch shared across batched steps.
final class QWBatchHeadScratch {
    static let maxBatchRows = 4
    let buffer: any MTLBuffer
    let vocab: Int
    let argmaxSlots: Int
    private let region: QWMRegion
    private let logitsOffset: Int
    private let keysHiOffset: Int
    private let keysLoOffset: Int
    private let tokensOffset: Int

    init(hal: QWHal, vocab: Int) throws {
        func align256(_ bytes: Int) -> Int { (bytes + 255) / 256 * 256 }
        let slots = max(1, QWHead.argmaxSlots(vocab))
        let rows = Self.maxBatchRows
        let logitsBytes = align256(rows * max(1, vocab) * 4)
        let keysBytes = align256(rows * slots * 4)
        let tokensBytes = align256(rows * 4)
        let total = logitsBytes + 2 * keysBytes + tokensBytes
        guard let buf = hal.device.makeBuffer(length: total, options: .storageModeShared) else {
            throw QWEngineDriverError.state("failed to allocate batched head scratch")
        }
        self.buffer = buf; self.vocab = vocab; self.argmaxSlots = slots
        self.region = QWMRegion(buffer: buf, base: 0, length: total)
        self.logitsOffset = 0; self.keysHiOffset = logitsBytes
        self.keysLoOffset = logitsBytes + keysBytes; self.tokensOffset = logitsBytes + 2 * keysBytes
    }

    private func rowTensor(_ base: Int, stride: Int, _ row: Int, _ count: Int, _ dt: QWDType) -> QWTensorValue {
        QWTensorValue(region: region, offset: base + row * stride, shape: QWTensorShape([count], dt))
    }

    func logitsView(rows: Int) -> QWTensorValue { rowTensor(logitsOffset, stride: 0, 0, rows * vocab, .f32) }
    func logitRow(_ row: Int) -> QWTensorValue { rowTensor(logitsOffset, stride: vocab * 4, row, vocab, .f32) }
    func keysHiRow(_ row: Int) -> QWTensorValue { rowTensor(keysHiOffset, stride: argmaxSlots * 4, row, argmaxSlots, .i32) }
    func keysLoRow(_ row: Int) -> QWTensorValue { rowTensor(keysLoOffset, stride: argmaxSlots * 4, row, argmaxSlots, .i32) }
    func tokenRow(_ row: Int) -> QWTensorValue { rowTensor(tokensOffset, stride: 4, row, 1, .i32) }

    func readToken(_ row: Int) -> Int {
        Int(buffer.contents().advanced(by: tokensOffset + row * 4).assumingMemoryBound(to: Int32.self).pointee)
    }

    func logitBuffer(_ row: Int) -> UnsafeBufferPointer<Float> {
        let ptr = buffer.contents().advanced(by: logitsOffset + row * vocab * 4).assumingMemoryBound(to: Float.self)
        return UnsafeBufferPointer(start: ptr, count: vocab)
    }
}

/// Thread-safe handle for one submitted request tracked across a batch wave.
final class QWBatchScheduledJob: @unchecked Sendable {
    let request: QWGenerationRequest
    let onToken: ((Int) -> Bool)?
    private let promise = QWBatchJobPromise<QWGenerationResult>()

    init(request: QWGenerationRequest, onToken: ((Int) -> Bool)?) {
        self.request = request
        self.onToken = onToken
    }

    func complete(_ result: Result<QWGenerationResult, Error>) {
        promise.complete(result)
    }

    func wait() throws -> QWGenerationResult {
        do {
            return try promise.wait()
        } catch QWBatchSchedulerError.missingJobOutcome {
            throw QWEngineDriverError.state("batch job finished without an outcome")
        }
    }
}

extension QWEngineDriver {
    nonisolated(unsafe) private static var batchHeadScratchAssocKey: UInt8 = 0
    nonisolated(unsafe) private static var batchMLPAssocKey: UInt8 = 0

    var canRunBatchedDecode: Bool {
        mlaOperator == nil && hcaOperator == nil && echoDrafter == nil && mbsaOperator == nil
            && psc.maxChunk >= QWBatchHeadScratch.maxBatchRows
            && geometry.vocab > 0
            && (geometry.mhcConfig == nil || (psc.mhcStream != nil && psc.mhcControls != nil))
            && (geometry.gdnLayerCount == 0 || psc.dn32.shape.elementCount >= QWBatchHeadScratch.maxBatchRows * 2 * geometry.gdnVHeads)
    }

    func sharedBatchHeadScratch() throws -> QWBatchHeadScratch {
        try slotLock.withLock {
            if let existing = objc_getAssociatedObject(self, &Self.batchHeadScratchAssocKey) as? QWBatchHeadScratch {
                return existing
            }
            let created = try QWBatchHeadScratch(hal: hal, vocab: geometry.vocab)
            objc_setAssociatedObject(self, &Self.batchHeadScratchAssocKey, created, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
            return created
        }
    }

    func sharedBatchMLP() throws -> QWMTPBatchedMLP {
        if let existing = mtpBatchedMLP, existing.rows >= QWBatchHeadScratch.maxBatchRows {
            return existing
        }
        return try slotLock.withLock {
            if let existing = objc_getAssociatedObject(self, &Self.batchMLPAssocKey) as? QWMTPBatchedMLP {
                return existing
            }
            let bytes = QWMTPBatchedMLP.allocationBytes(geometry: geometry, rows: QWBatchHeadScratch.maxBatchRows)
            guard let buf = hal.device.makeBuffer(length: bytes + 1024, options: .storageModeShared) else {
                throw QWEngineDriverError.state("failed to allocate batched MLP scratch")
            }
            let alloc = QWSubAllocator(region: QWMRegion(buffer: buf, base: 0, length: bytes + 1024))
            let created = try QWMTPBatchedMLP(geometry: geometry, hal: hal, allocator: alloc, rows: QWBatchHeadScratch.maxBatchRows)
            objc_setAssociatedObject(self, &Self.batchMLPAssocKey, created, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
            return created
        }
    }

    private var scLogitsBuffer: UnsafeBufferPointer<Float> {
        let ptr = sc.logits.region.pointer.advanced(by: sc.logits.offset).assumingMemoryBound(to: Float.self)
        return UnsafeBufferPointer(start: ptr, count: geometry.vocab)
    }

    /// Executes one batched trunk + LM head step across B=2...4 active sessions in one command buffer.
    func forwardBatchStep(
        _ sessions: [QWEngineSession], tokens: [Int],
        permit: QWGenerationPermit? = nil, headScratch: QWBatchHeadScratch? = nil
    ) throws -> (rawIds: [Int], stepMs: Double, scratch: QWBatchHeadScratch) {
        let B = sessions.count
        guard (2...QWBatchHeadScratch.maxBatchRows).contains(B), tokens.count == B else {
            throw QWEngineDriverError.state("batched decode requires B=2...4 matching tokens")
        }
        guard canRunBatchedDecode else {
            throw QWEngineDriverError.state("model geometry/operators require quantum-interleaved batching")
        }
        return try withFlightPermit(permit) {
            try forwardBatchStepLocked(sessions, tokens: tokens, headScratch: headScratch)
        }
    }

    private func forwardBatchStepLocked(
        _ sessions: [QWEngineSession], tokens: [Int], headScratch: QWBatchHeadScratch?
    ) throws -> (rawIds: [Int], stepMs: Double, scratch: QWBatchHeadScratch) {
        let B = sessions.count
        guard verifiedSession == nil else {
            throw QWEngineDriverError.state("resolve pending verification before batched decode")
        }
        let tStep0 = Date()
        for b in 0..<B {
            let sess = sessions[b]
            guard !sess.sessionFinished, !sess.stateFailed, !sess.verificationPending else {
                throw QWEngineDriverError.state("cannot run batched step on finished or failed session")
            }
            let pos = sess.position
            guard pos < ropeRows else { throw QWEngineDriverError.posBeyondTable(pos) }
            guard tokens[b] >= 0, tokens[b] < geometry.vocab else { throw QWEngineDriverError.badToken(tokens[b]) }
            try maybePrefixPagePressure(session: sess) {
                if sess.kv.tokenCapacity <= pos {
                    try kvCache.grow(sess.kv, to: pos + QWKVCache.pageTokens)
                    sess.attnKVViews = try kvCache.layerViewsAll(sess.kv, layers: attnLayers)
                }
                try kvCache.prepareWrites(sess.kv, positions: pos..<(pos + 1))
            }
        }
        let scratch = try headScratch ?? sharedBatchHeadScratch()
        try prepareLSASessionsForBatchStep(sessions)
        try copyEmbedRows(tokens, start: 0, count: B, to: prefixRows(psc.x, rows: B, cols: geometry.hidden, dt: .f16))
        if geometry.sharedExpertIntermediate > 0 && !geometry.sharedExpertHasGate, let s = psc.moe {
            let gPtr = s.gate.region.pointer.advanced(by: s.gate.offset).assumingMemoryBound(to: Float.self)
            for i in 0..<B { gPtr[i] = 1.0 }
        }
        guard let pair = hal.begin() else { throw QWEngineDriverError.state("cmdbuffer") }
        var cb: any MTLCommandBuffer = pair.0, enc: any MTLComputeCommandEncoder = pair.1
        activeCB = cb; activeEnc = enc; stepCBs = []
        var encoderEnded = false, committedFinalCB = false, stepCBsDrained = false
        defer {
            if !encoderEnded { (activeEnc ?? enc).endEncoding() }
            if !stepCBsDrained {
                if committedFinalCB { cb.waitUntilCompleted() }
                for sliceCb in stepCBs ?? [] { sliceCb.waitUntilCompleted() }
                if let pipe = coldEvents {
                    for bcb in pipe.bridgeCBs { bcb.waitUntilCompleted() }
                    pipe.shutdown(); coldEvents = nil
                }
                flushPendingColdUnloads()
            }
            activeCB = nil; activeEnc = nil; stepCBs = nil
        }
        for (v, p) in geometry.executionPlan.enumerated() {
            guard let denseKernel = layerKernels[p] as? DenseLayerKernel else {
                throw QWEngineDriverError.state("non-dense layer kernel encountered in batched decode")
            }
            try denseKernel.encodeBatchDecode(activeEnc ?? enc, driver: self, sessions: sessions, layer: v, weights: layersW[p])
        }
        cb = activeCB ?? cb; enc = activeEnc ?? enc
        try encodeBatchedHead(enc, scratch: scratch, count: B)
        encoderEnded = true
        hal.end(cb, enc)
        committedFinalCB = true
        let allCbs = (stepCBs?.isEmpty == false) ? (stepCBs! + [cb]) : [cb]
        defer { stepCBsDrained = true }
        try drainBatchStepCommandBuffers(allCbs)
        var rawIds: [Int] = []
        rawIds.reserveCapacity(B)
        for b in 0..<B {
            let pos = sessions[b].position
            sessions[b].position += 1; sessions[b].generated += 1
            sessions[b].batchedDecodeTouched = true; sessions[b].mlaIndexerStash = nil
            kvCache.markWritten(sessions[b].kv, upTo: pos + 1)
            rawIds.append(scratch.readToken(b))
        }
        return (rawIds, Date().timeIntervalSince(tStep0) * 1000, scratch)
    }

    private func drainBatchStepCommandBuffers(_ cbs: [any MTLCommandBuffer]) throws {
        for sliceCb in cbs { sliceCb.waitUntilCompleted() }
        let coldFailure: (any Error)? = coldEvents.flatMap { pipe in
            let r = pipe.takeStepResult()
            pipe.tuneBridge()
            return r.failure
        }
        flushPendingColdUnloads()
        if let coldFailure { throw QWEngineDriverError.state("cold event staging: \(coldFailure)") }
        for sliceCb in cbs {
            if let err = sliceCb.error {
                commandBufferErrors += 1
                throw QWEngineDriverError.state("batched decode cb: \(err.localizedDescription)")
            }
            if sliceCb.status != .completed {
                throw QWEngineDriverError.state("batched decode cb did not complete")
            }
        }
    }

    private func encodeBatchedHead(
        _ enc: any MTLComputeCommandEncoder,
        scratch: QWBatchHeadScratch,
        count B: Int
    ) throws {
        enc.useResource(scratch.buffer, usage: [.read, .write])
        let xBatch = prefixRows(psc.ln1, rows: B, cols: geometry.hidden, dt: .f16)
        let logitsBatch = scratch.logitsView(rows: B)
        if case .f16 = headStorage {
            encodeFwht32Fast(enc, x: xBatch)
            enc.memoryBarrier(scope: .buffers)
        }
        try head.encode(enc, storage: headStorage, x: xBatch, gs: nil, y: logitsBatch, M: B)
        if let mm = multimaxParams {
            enc.memoryBarrier(scope: .buffers)
            try head.encodeMultiMax(enc, logits: logitsBatch, params: mm, count: B * geometry.vocab)
        }
        enc.memoryBarrier(scope: .buffers)
        for b in 0..<B {
            try head.encodeArgmax(enc, y: scratch.logitRow(b), keysHi: scratch.keysHiRow(b),
                                  keysLo: scratch.keysLoRow(b), token: scratch.tokenRow(b), N: geometry.vocab)
            enc.memoryBarrier(scope: .buffers)
        }
    }

    // MARK: - Continuous multi-session scheduler loop

    private struct BatchSlotState {
        let job: QWBatchScheduledJob
        let session: QWEngineSession
        let maxNew: Int
        let stopIDs: Set<Int>
        let constraint: (any QWTokenConstraint)?
        let constraintSampler: QWSampler?
        let lpSampler: QWSampler?
        let topKCount: Int
        let speculativeEnabled: Bool
        var output: QWBatchSlotOutputState
        var cacheAnswer: QWCacheAnswer?
        var logprobs: [QWTokenLogprob] = []
        var pendingToken: Int
    }

    private func runScheduledBatchLocked(
        initialJobs: [QWBatchScheduledJob],
        headScratch: QWBatchHeadScratch?,
        permit: QWGenerationPermit,
        pollPending: (_ freeSlots: Int) -> [QWBatchScheduledJob],
        requeueDelayed: ([QWBatchScheduledJob]) -> Void
    ) {
        let scratch = headScratch ?? (try? sharedBatchHeadScratch())
        var active: [BatchSlotState] = []
        defer { for slot in active where !slot.session.sessionFinished { finishSession(slot.session) } }
        var wave = QWBatchWaveState()

        func failSlot(at idx: Int, error: any Error) {
            let failed = active.remove(at: idx)
            finishSession(failed.session)
            failed.job.complete(.failure(error))
            wave.clearBlock()
        }
        func pruneSlots(where predicate: (BatchSlotState) -> Bool) {
            for b in stride(from: active.count - 1, through: 0, by: -1) where predicate(active[b]) {
                finishAndCompleteSlot(active.remove(at: b))
                wave.clearBlock()
            }
        }
        func admitQueue(_ jobs: [QWBatchScheduledJob]) {
            var deferred: [QWBatchScheduledJob] = []
            for job in jobs {
                if !deferred.isEmpty { deferred.append(job); continue }
                do {
                    if let slot = try admitAndPrefillBatchJob(job, permit: permit) {
                        if slot.output.finished { finishAndCompleteSlot(slot) } else { active.append(slot) }
                    }
                } catch let kvErr as QWKVError where !active.isEmpty {
                    if case .outOfPages = kvErr { deferred.append(job) } else { job.complete(.failure(kvErr)) }
                } catch {
                    job.complete(.failure(error))
                }
            }
            if deferred.isEmpty { wave.clearBlock() }
            else { wave.markBlocked(freeWatermark: kvCache.freePageCount); requeueDelayed(deferred) }
        }

        admitQueue(initialJobs)
        while !active.isEmpty {
            wave.refreshBlock(currentFree: kvCache.freePageCount)
            let openSlots = slotLock.withLock { freeSlots.count }
            if wave.canPollPending(openSlots: openSlots) {
                let newlyPolled = pollPending(openSlots)
                if !newlyPolled.isEmpty { admitQueue(newlyPolled) }
            }
            pruneSlots { $0.session.position >= ropeRows }
            if active.isEmpty { break }

            let stepIndices = eligibleBatchSlotIndices(
                slots: active.map { ($0.session.position, $0.maxNew - $0.output.generated.count) },
                maxRows: QWBatchHeadScratch.maxBatchRows)
            let useBatched = QWBatchPartitionSelection.shouldRunBatchedStep(
                stepIndices: stepIndices, activeCount: active.count, roundRobinIndex: wave.roundRobinIndex,
                canRunBatchedDecode: canRunBatchedDecode,
                hasSpeculative: stepIndices.contains { canSpeculateSlot(active[$0]) },
                allowsSpeculativeBatch: geometry.numExperts == 0)
            if useBatched {
                let stepResult: (rawIds: [Int], stepMs: Double, scratch: QWBatchHeadScratch)
                do {
                    stepResult = try forwardBatchStep(
                        stepIndices.map { active[$0].session }, tokens: stepIndices.map { active[$0].pendingToken },
                        permit: permit, headScratch: scratch)
                } catch {
                    for slotIdx in stepIndices.sorted(by: >) { failSlot(at: slotIdx, error: error) }
                    continue
                }
                var failed: [(Int, any Error)] = []
                for (batchIdx, slotIdx) in stepIndices.enumerated() {
                    active[slotIdx].output.markDecodeStartedIfNeeded()
                    do {
                        try consumePredictedToken(
                            slot: &active[slotIdx], rawArgmax: stepResult.rawIds[batchIdx],
                            logitsTensor: stepResult.scratch.logitRow(batchIdx),
                            logitBuf: stepResult.scratch.logitBuffer(batchIdx), stepMsValue: stepResult.stepMs)
                    } catch { failed.append((slotIdx, error)) }
                }
                for (slotIdx, slotError) in failed.sorted(by: { $0.0 > $1.0 }) { failSlot(at: slotIdx, error: slotError) }
                pruneSlots { $0.output.finished }
                wave.advanceRoundRobin(
                    stepIndices: stepIndices, activeCount: active.count, mixedShortAndLong: stepIndices.count < active.count)
            } else {
                let idx = wave.clampedSlotIndex(activeCount: active.count)
                let quantum = QWBatchPartitionSelection.serialQuantum(
                    canRunBatchedDecode: canRunBatchedDecode, activeCount: active.count,
                    canSpeculateCurrent: canSpeculateSlot(active[idx]))
                for _ in 0..<quantum where idx < active.count && !active[idx].output.finished {
                    if active[idx].session.position >= ropeRows { active[idx].output.finished = true; break }
                    do { try stepSingleSlot(&active[idx], permit: permit) } catch { failSlot(at: idx, error: error); break }
                }
                if idx < active.count, active[idx].output.finished {
                    finishAndCompleteSlot(active.remove(at: idx))
                    wave.finishSerialSlotRemoved(activeCount: active.count)
                } else {
                    wave.advanceSerialSlot(activeCount: active.count)
                }
            }
        }
    }

    func runScheduledBatch(
        initialJobs: [QWBatchScheduledJob],
        headScratch: QWBatchHeadScratch? = nil,
        pollPending: (_ freeSlots: Int) -> [QWBatchScheduledJob],
        requeueDelayed: ([QWBatchScheduledJob]) -> Void
    ) {
        withDriverGenerateLock {
            let permit: QWGenerationPermit
            do { permit = try acquireRequest() } catch {
                for job in initialJobs { job.complete(.failure(error)) }
                return
            }
            defer { releaseRequest(permit) }
            runScheduledBatchLocked(
                initialJobs: initialJobs, headScratch: headScratch, permit: permit,
                pollPending: pollPending, requeueDelayed: requeueDelayed)
        }
    }

    private func admitAndPrefillBatchJob(
        _ job: QWBatchScheduledJob,
        permit: QWGenerationPermit
    ) throws -> BatchSlotState? {
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
        let stochasticOwner = pureGreedy ? nil : try QWStochasticSession(geometry: geometry, config: req.sampling)
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

        do {
            let unconditionedFirst: Int
            let prefillElapsed: Double
            if let cachedFirst = exactHitFirst {
                unconditionedFirst = cachedFirst
                prefillElapsed = Date().timeIntervalSince(started)
                req.onPrefillProgress?(0, 0)
            } else {
                try prepareSuffixProgressAndWrites(
                    session, matched: cacheMatched, promptCount: req.prompt.count,
                    onProgress: req.onPrefillProgress, permit: permit
                )
                let tPrefill = Date()
                unconditionedFirst = try prefill(session, tokens: Array(req.prompt.dropFirst(cacheMatched)),
                                                 mode: nil, permit: permit)
                prefillElapsed = Date().timeIntervalSince(tPrefill)
                if req.usePrefixCache, let store = prefixStore,
                   req.visualSpans.isEmpty {
                    _ = try? retainCachedPrompt(session, prompt: req.prompt, mode: gdnPrefillMode,
                                                first: unconditionedFirst, store: store, permit: permit)
                }
            }

            if let stochasticOwner { session.stochastic = stochasticOwner }
            let constraintSampler = req.constraint != nil ? QWSampler(config: req.sampling) : nil
            let lpSampler = req.collectLogprobs ? QWSampler(config: req.sampling) : nil
            let speculativeEnabled = QWBatchPartitionSelection.canEnableSpeculation(
                requested: req.speculative.enabled, mtpDrafts: mtpDrafts,
                collectLogprobs: req.collectLogprobs,
                allowsSpeculation: req.constraint?.allowsSpeculation ?? true,
                hasExplicitSeed: req.sampling.hasExplicitSeed, isGreedy: req.sampling.isGreedy)
            var slot = BatchSlotState(
                job: job, session: session, maxNew: effectiveMaxNew, stopIDs: req.stopIDs,
                constraint: req.constraint, constraintSampler: constraintSampler,
                lpSampler: lpSampler, topKCount: max(0, min(20, req.topLogprobsCount)),
                speculativeEnabled: speculativeEnabled,
                output: QWBatchSlotOutputState(
                    startedAt: started, prefillSeconds: prefillElapsed,
                    firstTokenSeconds: Date().timeIntervalSince(started),
                    prefillGPUSeconds: prefillGPUSecondsSince(mark: prefillMark)),
                cacheAnswer: req.usePrefixCache
                    ? QWCacheAnswer(hit: cacheMatched > 0, matchedTokens: cacheMatched,
                                    totalTokens: req.prompt.count, tier: cacheTier) : nil,
                pendingToken: unconditionedFirst)
            syncTokenHistoryAndRecycler(session, tokens: req.prompt, promptCount: req.prompt.count)
            try consumePredictedToken(
                slot: &slot, rawArgmax: unconditionedFirst,
                logitsTensor: sc.logits, logitBuf: scLogitsBuffer, stepMsValue: nil)
            if !slot.output.finished { slot.output.markDecodeStartedIfNeeded() }
            return slot
        } catch {
            finishSession(session)
            throw error
        }
    }

    private func canSpeculateSlot(_ slot: BatchSlotState) -> Bool {
        QWBatchPartitionSelection.isSlotSpeculatable(
            speculativeEnabled: slot.speculativeEnabled,
            remaining: slot.maxNew - slot.output.generated.count,
            hasLogprobsSampler: slot.lpSampler != nil,
            isUnconstrainedNow: slot.constraint?.isUnconstrainedNow ?? true,
            hasConstraint: slot.constraint != nil,
            hasConstraintSampler: slot.constraintSampler != nil
        ) && canRunSpeculativeRound(slot.session)
    }

    private func stepSingleSlot(_ slot: inout BatchSlotState, permit: QWGenerationPermit) throws {
        slot.output.markDecodeStartedIfNeeded()
        let saved = (activeConstraint, activeCollectLogprobs)
        defer { (activeConstraint, activeCollectLogprobs) = saved }
        let remaining = slot.maxNew - slot.output.generated.count
        if canSpeculateSlot(slot) {
            activeConstraint = slot.constraint.flatMap { c in slot.constraintSampler.map { (c, $0) } }
            activeCollectLogprobs = false
            let t0 = Date()
            let decision = try executeSpeculativeRoundStep(
                slot.session, remaining: remaining, stopIDs: slot.stopIDs,
                drain: true, store: prefixStore, permit: permit
            )
            if let next = decision.pendingToken { slot.pendingToken = next }
            slot.output.recordSpeculativeRound(
                tokens: decision.tokens, stepMsValue: Date().timeIntervalSince(t0) * 1000,
                maxNew: slot.maxNew, stopped: decision.stopped, stoppedToken: decision.stoppedToken,
                onToken: slot.job.onToken
            )
            return
        }
        activeConstraint = nil
        activeCollectLogprobs = (slot.lpSampler != nil || slot.constraint != nil || slot.session.stochastic != nil)
        let t0 = Date()
        let nextArgmax: Int = try withFlightPermit(permit) {
            guard let id = try forwardStep(slot.session, token: slot.pendingToken) else {
                throw QWEngineDriverError.state("serial forward step returned nil")
            }
            return id
        }
        try consumePredictedToken(
            slot: &slot, rawArgmax: nextArgmax, logitsTensor: sc.logits,
            logitBuf: scLogitsBuffer, stepMsValue: Date().timeIntervalSince(t0) * 1000
        )
    }

    private func selectBatchToken(
        slot: BatchSlotState, rawArgmax: Int, logitsTensor: QWTensorValue,
        logitBuf: UnsafeBufferPointer<Float>, repArr: [Int]
    ) throws -> Int {
        func sampleStochastic(_ owner: QWStochasticSession) throws -> Int {
            try owner.stream.sampleTarget(owner.distribution(logits: logitsTensor, repetitionHistory: repArr))
        }
        if let constraint = slot.constraint, let sampler = slot.constraintSampler {
            let chosen: Int
            if let allowed = try constraint.allowedTokens() {
                let rep = sampler.config.repetitionPenalty == 1.0 ? Set<Int>() : Set(repArr)
                chosen = try sampler.constrainedToken(logits: logitBuf, allowed: allowed, repetitionHistory: rep)
            } else if let owner = slot.session.stochastic {
                chosen = try sampleStochastic(owner)
            } else if sampler.config.isGreedy {
                chosen = sampler.config.repetitionPenalty == 1.0 ? rawArgmax
                    : try sampler.penalisedGreedyToken(logits: logitBuf, repetitionHistory: Set(repArr))
            } else {
                chosen = try sampler.sampleToken(
                    logits: logitBuf, repetitionHistory: sampler.config.repetitionPenalty == 1.0 ? [] : repArr
                )
            }
            try constraint.accept(chosen)
            return chosen
        }
        return try slot.session.stochastic.map(sampleStochastic) ?? rawArgmax
    }

    private func consumePredictedToken(
        slot: inout BatchSlotState, rawArgmax: Int, logitsTensor: QWTensorValue,
        logitBuf: UnsafeBufferPointer<Float>, stepMsValue: Double?
    ) throws {
        let repArr = slot.session.activeRepetitionHistory()
        let chosen = try selectBatchToken(
            slot: slot, rawArgmax: rawArgmax, logitsTensor: logitsTensor, logitBuf: logitBuf, repArr: repArr
        )
        if let owner = slot.session.stochastic {
            slot.session.roundRecord.samplerDrawIndex = owner.stream.drawsConsumed
        }
        slot.pendingToken = chosen
        slot.session.roundRecord.pendingToken = chosen
        slot.session.roundRecord.outputCount = slot.session.generated
        guard slot.output.beginPredictedToken(chosen, stopIDs: slot.stopIDs, stepMsValue: stepMsValue) else { return }
        if let lpSampler = slot.lpSampler {
            let lp = lpSampler.computeTokenLogprob(logits: logitBuf, chosenToken: chosen, topK: slot.topKCount)
            slot.logprobs.append(lp)
            slot.job.request.onTokenLogprob?(lp)
        }
        if (slot.constraint?.isUnconstrainedNow ?? true), let prev1 = slot.session.tokenHistory.last {
            let prev2 = slot.session.tokenHistory.count >= 2
                ? slot.session.tokenHistory[slot.session.tokenHistory.count - 2] : nil
            slot.session.tokenRecycler.recordTransition(previous: prev2, current: prev1, next: chosen)
        }
        slot.session.tokenHistory.append(chosen)
        slot.output.finishPredictedToken(maxNew: slot.maxNew, accepted: slot.job.onToken?(chosen) ?? true)
    }

    private func finishAndCompleteSlot(_ slot: BatchSlotState) {
        finishSession(slot.session)
        let summary = slot.output.speculativeSummary(
            enabled: slot.speculativeEnabled, proposals: mtpDraftProposals, accepted: mtpDraftAccepted
        )
        slot.job.complete(.success(QWGenerationResult(
            tokens: slot.output.generated, stepMs: slot.output.stepMs,
            stepTokenCounts: slot.output.stepTokenCounts,
            prefillSeconds: slot.output.prefillSeconds,
            firstTokenSeconds: slot.output.firstTokenSeconds,
            decodeSeconds: slot.output.decodeDurationSeconds(),
            prefillGPUSeconds: slot.output.prefillGPUSeconds > 0 ? slot.output.prefillGPUSeconds : nil,
            stopTokenID: slot.output.stoppedToken, cache: slot.cacheAnswer,
            speculativeSummary: summary, tokenLogprobs: slot.logprobs
        )))
    }
}
