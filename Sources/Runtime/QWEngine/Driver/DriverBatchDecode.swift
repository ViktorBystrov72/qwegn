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
        self.buffer = buf
        self.vocab = vocab
        self.argmaxSlots = slots
        self.region = QWMRegion(buffer: buf, base: 0, length: total)
        self.logitsOffset = 0
        self.keysHiOffset = logitsBytes
        self.keysLoOffset = logitsBytes + keysBytes
        self.tokensOffset = logitsBytes + 2 * keysBytes
    }

    func logitsView(rows: Int) -> QWTensorValue {
        QWTensorValue(region: region, offset: logitsOffset, shape: QWTensorShape([rows * vocab], .f32))
    }

    func logitRow(_ row: Int) -> QWTensorValue {
        QWTensorValue(region: region, offset: logitsOffset + row * vocab * 4, shape: QWTensorShape([vocab], .f32))
    }

    func keysHiRow(_ row: Int) -> QWTensorValue {
        QWTensorValue(region: region, offset: keysHiOffset + row * argmaxSlots * 4, shape: QWTensorShape([argmaxSlots], .i32))
    }

    func keysLoRow(_ row: Int) -> QWTensorValue {
        QWTensorValue(region: region, offset: keysLoOffset + row * argmaxSlots * 4, shape: QWTensorShape([argmaxSlots], .i32))
    }

    func tokenRow(_ row: Int) -> QWTensorValue {
        QWTensorValue(region: region, offset: tokensOffset + row * 4, shape: QWTensorShape([1], .i32))
    }

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
    let done = DispatchSemaphore(value: 0)
    private let outcomeLock = NSLock()
    private var outcome: Result<QWGenerationResult, Error>?

    init(request: QWGenerationRequest, onToken: ((Int) -> Bool)?) {
        self.request = request
        self.onToken = onToken
    }

    func complete(_ result: Result<QWGenerationResult, Error>) {
        let shouldSignal = outcomeLock.withLock { () -> Bool in
            guard outcome == nil else { return false }
            outcome = result
            return true
        }
        if shouldSignal { done.signal() }
    }

    func wait() throws -> QWGenerationResult {
        done.wait()
        guard let result = outcomeLock.withLock({ outcome }) else {
            throw QWEngineDriverError.state("batch job finished without an outcome")
        }
        return try result.get()
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
        slotLock.lock()
        defer { slotLock.unlock() }
        if let existing = objc_getAssociatedObject(self, &Self.batchHeadScratchAssocKey) as? QWBatchHeadScratch {
            return existing
        }
        let created = try QWBatchHeadScratch(hal: hal, vocab: geometry.vocab)
        objc_setAssociatedObject(self, &Self.batchHeadScratchAssocKey, created, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        return created
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

    /// Executes one batched trunk + LM head step across B=2...4 active sessions in one command buffer.
    func forwardBatchStep(
        _ sessions: [QWEngineSession],
        tokens: [Int],
        permit: QWGenerationPermit? = nil,
        headScratch: QWBatchHeadScratch? = nil
    ) throws -> (rawIds: [Int], stepMs: Double, scratch: QWBatchHeadScratch) {
        let B = sessions.count
        guard (2...QWBatchHeadScratch.maxBatchRows).contains(B), tokens.count == B else {
            throw QWEngineDriverError.state("batched decode requires B=2...4 matching tokens")
        }
        guard canRunBatchedDecode else {
            throw QWEngineDriverError.state("model geometry/operators require quantum-interleaved batching")
        }
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        try requestGate.validate(permit)
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
            let ensureKV: () throws -> Void = {
                if sess.kv.tokenCapacity <= pos {
                    try self.kvCache.grow(sess.kv, to: pos + QWKVCache.pageTokens)
                    sess.attnKVViews = try self.kvCache.layerViewsAll(sess.kv, layers: self.attnLayers)
                }
                try self.kvCache.prepareWrites(sess.kv, positions: pos..<(pos + 1))
            }
            if let store = prefixStore {
                try prefixPagePressure(store, session: sess, ensureKV)
            } else {
                try ensureKV()
            }
        }
        let scratch = try headScratch ?? sharedBatchHeadScratch()
        try prepareLSASessionsForBatchStep(sessions)
        try copyEmbedRows(tokens, start: 0, count: B,
                          to: prefixRows(psc.x, rows: B, cols: geometry.hidden, dt: .f16))
        if geometry.sharedExpertIntermediate > 0 && !geometry.sharedExpertHasGate, let s = psc.moe {
            let gPtr = s.gate.region.pointer.advanced(by: s.gate.offset).assumingMemoryBound(to: Float.self)
            for i in 0..<B { gPtr[i] = 1.0 }
        }

        guard let pair = hal.begin() else { throw QWEngineDriverError.state("cmdbuffer") }
        var cb: any MTLCommandBuffer = pair.0
        var enc: any MTLComputeCommandEncoder = pair.1
        activeCB = cb
        activeEnc = enc
        stepCBs = []
        var encoderEnded = false
        var stepCBsDrained = false
        defer {
            if !encoderEnded { (activeEnc ?? enc).endEncoding() }
            if !stepCBsDrained {
                for sliceCb in stepCBs ?? [] { sliceCb.waitUntilCompleted() }
                _ = coldEvents?.takeStepResult()
                flushPendingColdUnloads()
            }
            activeCB = nil; activeEnc = nil; stepCBs = nil
        }

        for (v, p) in geometry.executionPlan.enumerated() {
            let lw = layersW[p]
            guard let denseKernel = layerKernels[p] as? DenseLayerKernel else {
                throw QWEngineDriverError.state("non-dense layer kernel encountered in batched decode")
            }
            try denseKernel.encodeBatchDecode(activeEnc ?? enc, driver: self, sessions: sessions, layer: v, weights: lw)
        }
        cb = activeCB ?? cb
        enc = activeEnc ?? enc
        try encodeBatchedHead(enc, scratch: scratch, count: B)
        encoderEnded = true
        hal.end(cb, enc)
        let allCbs = (stepCBs?.isEmpty == false) ? (stepCBs! + [cb]) : [cb]
        for sliceCb in allCbs { sliceCb.waitUntilCompleted() }
        let coldFailure: (any Error)? = coldEvents.flatMap { pipe in
            let r = pipe.takeStepResult()
            pipe.tuneBridge()
            return r.failure
        }
        flushPendingColdUnloads()
        stepCBsDrained = true
        if let coldFailure { throw QWEngineDriverError.state("cold event staging: \(coldFailure)") }
        for sliceCb in allCbs {
            if let err = sliceCb.error {
                commandBufferErrors += 1
                throw QWEngineDriverError.state("batched decode cb: \(err.localizedDescription)")
            }
            if sliceCb.status != .completed {
                throw QWEngineDriverError.state("batched decode cb did not complete")
            }
        }
        var rawIds: [Int] = []
        rawIds.reserveCapacity(B)
        for b in 0..<B {
            let pos = sessions[b].position
            sessions[b].position += 1
            sessions[b].generated += 1
            sessions[b].batchedDecodeTouched = true
            sessions[b].mlaIndexerStash = nil
            kvCache.markWritten(sessions[b].kv, upTo: pos + 1)
            rawIds.append(scratch.readToken(b))
        }
        return (rawIds, Date().timeIntervalSince(tStep0) * 1000, scratch)
    }

    private func encodeBatchedHead(
        _ enc: any MTLComputeCommandEncoder,
        scratch: QWBatchHeadScratch,
        count B: Int
    ) throws {
        enc.useResource(scratch.buffer, usage: [.read, .write])
        let xBatch = prefixRows(psc.ln1, rows: B, cols: geometry.hidden, dt: .f16)
        let logitsBatch = scratch.logitsView(rows: B)
        switch headStorage {
        case .f16:
            encodeFwht32Fast(enc, x: xBatch)
            enc.memoryBarrier(scope: .buffers)
            try head.encode(enc, storage: headStorage, x: xBatch, gs: nil, y: logitsBatch, M: B)
        case .q4g:
            try head.encode(enc, storage: headStorage, x: xBatch, gs: nil, y: logitsBatch, M: B)
        }
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
        let startedAt: Date
        var prefillSeconds: Double
        var firstTokenSeconds: Double
        var prefillGPUSeconds: Double
        var decodeStartedAt: Date?
        var cacheAnswer: QWCacheAnswer?
        var generated: [Int]
        var stepMs: [Double] = []
        var stepTokenCounts: [Int] = []
        var speculativeRounds: Int = 0
        var speculativeRoundTokens: Int = 0
        var logprobs: [QWTokenLogprob] = []
        var pendingToken: Int
        var stoppedToken: Int? = nil
        var finished: Bool = false
    }

    func runScheduledBatch(
        initialJobs: [QWBatchScheduledJob],
        headScratch: QWBatchHeadScratch? = nil,
        pollPending: (_ freeSlots: Int) -> [QWBatchScheduledJob],
        requeueDelayed: ([QWBatchScheduledJob]) -> Void
    ) {
        withDriverGenerateLock {
            let permit: QWGenerationPermit
            do {
                permit = try acquireRequest()
            } catch {
                for job in initialJobs { job.complete(.failure(error)) }
                return
            }
            defer { releaseRequest(permit) }

            let scratch = headScratch ?? (try? sharedBatchHeadScratch())
            var active: [BatchSlotState] = []
            defer {
                for slot in active where !slot.session.sessionFinished {
                    finishSession(slot.session)
                }
            }

            var blockedFreeKVPages: Int? = nil

            func admitQueue(_ jobs: [QWBatchScheduledJob]) {
                var deferred: [QWBatchScheduledJob] = []
                for job in jobs {
                    if !deferred.isEmpty {
                        deferred.append(job)
                        continue
                    }
                    do {
                        if let slot = try admitAndPrefillBatchJob(job, permit: permit) {
                            if slot.finished {
                                finishAndCompleteSlot(slot)
                            } else {
                                active.append(slot)
                            }
                        }
                    } catch let kvErr as QWKVError where !active.isEmpty {
                        if case .outOfPages = kvErr {
                            deferred.append(job)
                        } else {
                            job.complete(.failure(kvErr))
                        }
                    } catch {
                        job.complete(.failure(error))
                    }
                }
                if !deferred.isEmpty {
                    blockedFreeKVPages = kvCache.freePageCount
                    requeueDelayed(deferred)
                } else {
                    blockedFreeKVPages = nil
                }
            }

            admitQueue(initialJobs)
            var rrIndex = 0
            while !active.isEmpty {
                if let blockedAt = blockedFreeKVPages, kvCache.freePageCount > blockedAt {
                    blockedFreeKVPages = nil
                }
                let openSlots = slotLock.withLock { freeSlots.count }
                if openSlots > 0 && blockedFreeKVPages == nil {
                    let newlyPolled = pollPending(openSlots)
                    if !newlyPolled.isEmpty { admitQueue(newlyPolled) }
                }
                for b in stride(from: active.count - 1, through: 0, by: -1) where active[b].session.position >= ropeRows {
                    finishAndCompleteSlot(active.remove(at: b))
                    blockedFreeKVPages = nil
                }
                if active.isEmpty { break }

                let stepIndices = eligibleBatchSlotIndices(
                    slots: active.map { ($0.session.position, $0.maxNew - $0.generated.count) },
                    maxRows: QWBatchHeadScratch.maxBatchRows)
                let mixedShortAndLong = stepIndices.count < active.count
                let rrAllowsBatch = !mixedShortAndLong || stepIndices.contains(rrIndex)
                let useBatchedStep = stepIndices.count >= 2 && canRunBatchedDecode && rrAllowsBatch
                    && (geometry.numExperts == 0 || !stepIndices.contains { canSpeculateSlot(active[$0]) })
                if useBatchedStep {
                    let stepSessions = stepIndices.map { active[$0].session }
                    let stepTokens = stepIndices.map { active[$0].pendingToken }
                    do {
                        let (rawIds, stepMs, usedScratch) = try forwardBatchStep(
                            stepSessions, tokens: stepTokens, permit: permit, headScratch: scratch)
                        for (batchIdx, slotIdx) in stepIndices.enumerated() {
                            if active[slotIdx].decodeStartedAt == nil { active[slotIdx].decodeStartedAt = Date() }
                            try consumePredictedToken(
                                slot: &active[slotIdx], rawArgmax: rawIds[batchIdx],
                                logitsTensor: usedScratch.logitRow(batchIdx),
                                logitBuf: usedScratch.logitBuffer(batchIdx),
                                stepMsValue: stepMs)
                        }
                    } catch {
                        for slotIdx in stepIndices.sorted(by: >) {
                            finishSession(active[slotIdx].session)
                            active[slotIdx].job.complete(.failure(error))
                            active.remove(at: slotIdx)
                        }
                        blockedFreeKVPages = nil
                    }
                    for b in stride(from: active.count - 1, through: 0, by: -1) where active[b].finished {
                        let completed = active.remove(at: b)
                        finishAndCompleteSlot(completed)
                        blockedFreeKVPages = nil
                    }
                    if mixedShortAndLong && !active.isEmpty {
                        rrIndex = ((stepIndices.last ?? rrIndex) + 1) % active.count
                    } else if rrIndex >= active.count {
                        rrIndex = 0
                    }
                } else {
                    if rrIndex >= active.count { rrIndex = 0 }
                    let canSpecCur = canSpeculateSlot(active[rrIndex])
                    let quantum = (!canRunBatchedDecode && active.count > 1 && !canSpecCur) ? 4 : 1
                    var stepsDone = 0
                    while stepsDone < quantum, rrIndex < active.count, !active[rrIndex].finished {
                        if active[rrIndex].session.position >= ropeRows {
                            active[rrIndex].finished = true
                            break
                        }
                        do {
                            try stepSingleSlot(&active[rrIndex], permit: permit)
                        } catch {
                            let failed = active.remove(at: rrIndex)
                            finishSession(failed.session)
                            failed.job.complete(.failure(error))
                            blockedFreeKVPages = nil
                            break
                        }
                        stepsDone += 1
                    }
                    if rrIndex < active.count, active[rrIndex].finished {
                        let completed = active.remove(at: rrIndex)
                        finishAndCompleteSlot(completed)
                        blockedFreeKVPages = nil
                        if rrIndex >= active.count { rrIndex = 0 }
                    } else if !active.isEmpty {
                        rrIndex = (rrIndex + 1) % active.count
                    }
                }
            }
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
        guard req.maxNew >= 0, !req.prompt.isEmpty,
              req.prompt.allSatisfy({ (0..<geometry.vocab).contains($0) }),
              req.stopIDs.allSatisfy({ (0..<geometry.vocab).contains($0) }) else {
            throw QWEngineDriverError.state("invalid batch generation request bounds or token IDs")
        }
        guard req.prompt.count < ropeRows else {
            throw QWEngineDriverError.posBeyondTable(req.prompt.count)
        }
        let effectiveMaxNew = min(req.maxNew, ropeRows - req.prompt.count)
        if effectiveMaxNew == 0 {
            job.complete(.success(QWGenerationResult(tokens: [], stepMs: [], stepTokenCounts: [])))
            return nil
        }

        let started = Date()
        let pureGreedy = (req.sampling.isGreedy && req.sampling.repetitionPenalty == 1.0) || req.constraint != nil
        let stochasticOwner = pureGreedy ? nil : try QWStochasticSession(geometry: geometry, config: req.sampling)
        let savedSalt = activeCacheSalt
        let savedVisual = activeVisualSpans
        let savedConstraint = activeConstraint
        let savedCollectLP = activeCollectLogprobs
        let savedPrefillProg = activePrefillProgress
        let savedPrefillTot = activePrefillTotalTokens
        let savedPrefillBase = activePrefillBasePosition
        defer {
            activeCacheSalt = savedSalt; activeVisualSpans = savedVisual
            activeConstraint = savedConstraint; activeCollectLogprobs = savedCollectLP
            activePrefillProgress = savedPrefillProg; activePrefillTotalTokens = savedPrefillTot
            activePrefillBasePosition = savedPrefillBase
        }
        activeCacheSalt = req.cacheSalt; activeVisualSpans = req.visualSpans
        activeConstraint = nil
        activeCollectLogprobs = req.collectLogprobs || req.constraint != nil || stochasticOwner != nil || req.usePrefixCache
        activePrefillProgress = req.onPrefillProgress; activePrefillTotalTokens = nil; activePrefillBasePosition = 0
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
            session = loaded.session
            exactHitFirst = loaded.first
            cacheMatched = loaded.matched
            cacheTier = loaded.tier
        } else if let store = prefixStore {
            session = try prefixPagePressure(store) { try admitSession(promptTokens: req.prompt.count) }
        } else {
            session = try admitSession(promptTokens: req.prompt.count)
        }

        do {
            let unconditionedFirst: Int
            let prefillElapsed: Double
            if let cachedFirst = exactHitFirst {
                unconditionedFirst = cachedFirst
                prefillElapsed = Date().timeIntervalSince(started)
                req.onPrefillProgress?(0, 0)
            } else {
                activePrefillBasePosition = cacheMatched
                let computedTotal = max(0, req.prompt.count - cacheMatched)
                activePrefillTotalTokens = computedTotal
                req.onPrefillProgress?(0, computedTotal)
                if cacheMatched > 0, let store = prefixStore {
                    try prefixPagePressure(store, session: session) {
                        flight.lock()
                        defer { flight.unlock() }
                        try requestGate.validate(permit)
                        try kvCache.prepareWrites(session.kv, positions: session.position..<req.prompt.count)
                        if let mtpCache, let handle = session.mtpKV {
                            let rows = mtpCache.written(handle)
                            try mtpCache.prepareWrites(handle, positions: rows..<(rows + computedTotal))
                        }
                    }
                }
                let tPrefill = Date()
                unconditionedFirst = try prefill(session, tokens: Array(req.prompt.dropFirst(cacheMatched)),
                                                 mode: nil, permit: permit)
                prefillElapsed = Date().timeIntervalSince(tPrefill)
                if req.usePrefixCache, let store = prefixStore, stochasticOwner == nil,
                   req.visualSpans.isEmpty {
                    _ = try? retainCachedPrompt(session, prompt: req.prompt, mode: gdnPrefillMode,
                                                first: unconditionedFirst, store: store, permit: permit)
                }
            }

            if let stochasticOwner { session.stochastic = stochasticOwner }
            let constraintSampler = req.constraint != nil ? QWSampler(config: req.sampling) : nil
            let lpSampler = req.collectLogprobs ? QWSampler(config: req.sampling) : nil
            let speculativeEnabled = req.speculative.enabled && mtpDrafts > 0
                && !req.collectLogprobs && (req.constraint?.allowsSpeculation ?? true)
                && !(req.sampling.hasExplicitSeed && !req.sampling.isGreedy)
            let logitsPtr = sc.logits.region.pointer.advanced(by: sc.logits.offset).assumingMemoryBound(to: Float.self)
            let logitsBuf = UnsafeBufferPointer(start: logitsPtr, count: geometry.vocab)
            var slot = BatchSlotState(
                job: job, session: session, maxNew: effectiveMaxNew, stopIDs: req.stopIDs,
                constraint: req.constraint, constraintSampler: constraintSampler,
                lpSampler: lpSampler, topKCount: max(0, min(20, req.topLogprobsCount)),
                speculativeEnabled: speculativeEnabled,
                startedAt: started, prefillSeconds: prefillElapsed,
                firstTokenSeconds: Date().timeIntervalSince(started),
                prefillGPUSeconds: prefillGPUSecondsSince(mark: prefillMark),
                decodeStartedAt: nil,
                cacheAnswer: req.usePrefixCache
                    ? QWCacheAnswer(hit: cacheMatched > 0, matchedTokens: cacheMatched,
                                    totalTokens: req.prompt.count, tier: cacheTier)
                    : nil,
                generated: [], pendingToken: unconditionedFirst
            )
            session.tokenHistory = req.prompt
            session.tokenRecycler.reset()
            session.tokenRecycler.ingestSequence(req.prompt)
            try consumePredictedToken(
                slot: &slot, rawArgmax: unconditionedFirst,
                logitsTensor: sc.logits, logitBuf: logitsBuf, stepMsValue: nil)
            if !slot.finished { slot.decodeStartedAt = Date() }
            return slot
        } catch {
            finishSession(session)
            throw error
        }
    }

    private func canSpeculateSlot(_ slot: BatchSlotState) -> Bool {
        let remaining = slot.maxNew - slot.generated.count
        guard slot.speculativeEnabled, remaining >= 2, slot.lpSampler == nil,
              slot.constraint?.isUnconstrainedNow ?? true,
              slot.constraint == nil || slot.constraintSampler != nil else {
            return false
        }
        return canRunSpeculativeRound(slot.session)
    }

    private func stepSingleSlot(_ slot: inout BatchSlotState, permit: QWGenerationPermit) throws {
        if slot.decodeStartedAt == nil { slot.decodeStartedAt = Date() }
        let savedConstraint = activeConstraint
        let savedCollectLP = activeCollectLogprobs
        defer {
            activeConstraint = savedConstraint
            activeCollectLogprobs = savedCollectLP
        }
        let remaining = slot.maxNew - slot.generated.count
        if canSpeculateSlot(slot) {
            activeConstraint = slot.constraint.flatMap { c in slot.constraintSampler.map { (c, $0) } }
            activeCollectLogprobs = false
            let t0 = Date()
            func runWithPressure<T>(_ body: () throws -> T) throws -> T {
                if let store = prefixStore {
                    return try prefixPagePressure(store, session: slot.session, body)
                }
                return try body()
            }
            let decision: QWGreedyRoundDecision
            if slot.session.stochastic != nil {
                let round = try runWithPressure {
                    try prepareStochasticRound(slot.session, maxNew: remaining, stopIDs: slot.stopIDs, permit: permit)
                }
                try commitStochasticVerification(slot.session, round: round, drain: true, permit: permit)
                decision = round.decision
            } else {
                let round = try runWithPressure {
                    try prepareGreedyRound(slot.session, maxNew: remaining, stopIDs: slot.stopIDs, permit: permit)
                }
                try commitGreedyVerification(slot.session, round: round, drain: true, permit: permit)
                decision = round.decision
            }
            slot.stepMs.append(Date().timeIntervalSince(t0) * 1000)
            slot.stepTokenCounts.append(decision.tokens.count)
            slot.speculativeRounds += 1
            slot.speculativeRoundTokens += decision.tokens.count
            slot.generated.append(contentsOf: decision.tokens)
            if let next = decision.pendingToken { slot.pendingToken = next }
            var cancelled = false
            for tok in decision.tokens {
                if slot.job.onToken?(tok) == false { cancelled = true }
            }
            if cancelled { slot.finished = true }
            if decision.stopped {
                slot.stoppedToken = decision.stoppedToken
                slot.finished = true
            } else if slot.generated.count >= slot.maxNew {
                slot.finished = true
            }
            return
        }
        activeConstraint = nil
        activeCollectLogprobs = (slot.lpSampler != nil || slot.constraint != nil || slot.session.stochastic != nil)
        let t0 = Date()
        let nextArgmax: Int = try {
            guard flight.try() else { throw QWEngineDriverError.busy }
            defer { flight.unlock() }
            try requestGate.validate(permit)
            guard let id = try forwardStep(slot.session, token: slot.pendingToken) else {
                throw QWEngineDriverError.state("serial forward step returned nil")
            }
            return id
        }()
        let logitsPtr = sc.logits.region.pointer.advanced(by: sc.logits.offset).assumingMemoryBound(to: Float.self)
        let logitsBuf = UnsafeBufferPointer(start: logitsPtr, count: geometry.vocab)
        try consumePredictedToken(
            slot: &slot, rawArgmax: nextArgmax,
            logitsTensor: sc.logits, logitBuf: logitsBuf,
            stepMsValue: Date().timeIntervalSince(t0) * 1000)
    }

    private func consumePredictedToken(
        slot: inout BatchSlotState,
        rawArgmax: Int,
        logitsTensor: QWTensorValue,
        logitBuf: UnsafeBufferPointer<Float>,
        stepMsValue: Double?
    ) throws {
        let chosen: Int
        if let constraint = slot.constraint, let sampler = slot.constraintSampler {
            if let allowed = try constraint.allowedTokens() {
                let rep = sampler.config.repetitionPenalty == 1.0 ? Set<Int>() : Set(slot.session.tokenHistory)
                chosen = try sampler.constrainedToken(logits: logitBuf, allowed: allowed, repetitionHistory: rep)
            } else if sampler.config.isGreedy && sampler.config.repetitionPenalty == 1.0 {
                chosen = rawArgmax
            } else if sampler.config.isGreedy {
                chosen = try sampler.penalisedGreedyToken(logits: logitBuf, repetitionHistory: Set(slot.session.tokenHistory))
            } else {
                let rep = sampler.config.repetitionPenalty == 1.0 ? [] : slot.session.tokenHistory
                chosen = try sampler.sampleToken(logits: logitBuf, repetitionHistory: rep)
            }
            try constraint.accept(chosen)
        } else if let owner = slot.session.stochastic {
            chosen = try owner.stream.sampleTarget(
                owner.distribution(logits: logitsTensor, repetitionHistory: slot.session.tokenHistory))
            slot.session.roundRecord.samplerDrawIndex = owner.stream.drawsConsumed
        } else {
            chosen = rawArgmax
        }

        if let stepMsValue {
            slot.stepMs.append(stepMsValue)
            slot.stepTokenCounts.append(1)
        }
        slot.pendingToken = chosen
        slot.session.roundRecord.pendingToken = chosen
        slot.session.roundRecord.outputCount = slot.session.generated
        if slot.stopIDs.contains(chosen) {
            slot.stoppedToken = chosen
            slot.finished = true
            return
        }
        if let lpSampler = slot.lpSampler {
            let lp = lpSampler.computeTokenLogprob(logits: logitBuf, chosenToken: chosen, topK: slot.topKCount)
            slot.logprobs.append(lp)
            slot.job.request.onTokenLogprob?(lp)
        }
        if slot.constraint == nil, let prev1 = slot.session.tokenHistory.last {
            let prev2 = slot.session.tokenHistory.count >= 2
                ? slot.session.tokenHistory[slot.session.tokenHistory.count - 2] : nil
            slot.session.tokenRecycler.recordTransition(previous: prev2, current: prev1, next: chosen)
        }
        slot.session.tokenHistory.append(chosen)
        slot.generated.append(chosen)
        if let cb = slot.job.onToken, cb(chosen) == false {
            slot.finished = true
            return
        }
        if slot.generated.count >= slot.maxNew {
            slot.finished = true
        }
    }

    private func finishAndCompleteSlot(_ slot: BatchSlotState) {
        finishSession(slot.session)
        let decodeSec = slot.decodeStartedAt.map { Date().timeIntervalSince($0) }
        let specSummary: String? = slot.speculativeRounds > 0
            ? String(format: "rounds=%d, mean=%.2f tok/rnd",
                     slot.speculativeRounds,
                     Double(slot.speculativeRoundTokens) / Double(slot.speculativeRounds))
            : (slot.speculativeEnabled
                ? String(format: "mtp=%d/%d", mtpDraftAccepted, mtpDraftProposals)
                : nil)
        slot.job.complete(.success(QWGenerationResult(
            tokens: slot.generated,
            stepMs: slot.stepMs,
            stepTokenCounts: slot.stepTokenCounts,
            prefillSeconds: slot.prefillSeconds,
            firstTokenSeconds: slot.firstTokenSeconds,
            decodeSeconds: decodeSec,
            prefillGPUSeconds: slot.prefillGPUSeconds > 0 ? slot.prefillGPUSeconds : nil,
            stopTokenID: slot.stoppedToken,
            cache: slot.cacheAnswer,
            speculativeSummary: specSummary,
            tokenLogprobs: slot.logprobs
        )))
    }
}
