// MTPVerification+Rounds.swift — Speculative draft-verify round lifecycle on QWEngineDriver.

import Foundation
import Darwin
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWMLP
import QWHead
import QWMTP
import QWQuant

extension QWEngineDriver {
    /// True while a speculative verify window fits below both `ropeRows` and the draft encoder's RoPE ceiling.
    func canRunSpeculativeRound(_ session: QWEngineSession) -> Bool {
        guard let verifyWindow, !session.batchedDecodeTouched,
              !session.sessionFinished, !session.stateFailed, !session.verificationPending,
              session.trunkHiddenValid,
              session.roundRecord.consumedCount == session.position,
              session.roundRecord.absolutePosition == session.position,
              session.roundRecord.outputCount == session.generated else {
            return false
        }
        if let mtpCache {
            guard let handle = session.mtpKV,
                  mtpCache.written(handle) == session.roundRecord.mtpPhysicalRows else {
                return false
            }
        }
        if dflashDrafter != nil {
            guard session.dflashSlot != nil, session.dflashCacheOffset == session.position else {
                return false
            }
        }
        let count = verifyWindow.rows - 1
        let draftCapacity = (dflashDrafter != nil ? nil : mtpDraftEncoder?.positionCapacity) ?? ropeRows
        return session.position > 0
            && session.position <= ropeRows - verifyWindow.rows
            && session.position <= draftCapacity - count
    }

    /// Diagnostic two-phase round owner. Output is tentative until commit succeeds;
    /// scratch stays leased across prefix selection until commit/discard/finish.
    /// No user callback runs under flight, and ordinary generation never calls it.
    func prepareGreedyRound(_ session: QWEngineSession, maxNew: Int,
                            stopIDs: Set<Int> = [],
                            permit: QWGenerationPermit? = nil) throws -> QWPreparedGreedyRound {
        try prepareRound(session, maxNew: maxNew, stopIDs: stopIDs, stochastic: false, permit: permit)
    }

    func prepareStochasticRound(_ session: QWEngineSession, maxNew: Int,
                                stopIDs: Set<Int> = [],
                                permit: QWGenerationPermit? = nil) throws -> QWPreparedStochasticRound {
        QWPreparedStochasticRound(authorization:
            try prepareRound(session, maxNew: maxNew, stopIDs: stopIDs, stochastic: true, permit: permit))
    }

    func prepareRound(_ session: QWEngineSession, maxNew: Int, stopIDs: Set<Int>,
                              stochastic: Bool, permit: QWGenerationPermit?) throws -> QWPreparedGreedyRound {
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        try requestGate.validate(permit)
        guard !mtpAdaptiveDepth || !stochastic else {
            throw QWEngineDriverError.state("QW_MTP_ADAPTIVE_DEPTH currently supports greedy rounds only")
        }
        let profileStart = mtpStageProfiling ? DispatchTime.now().uptimeNanoseconds : nil
        if (echoDrafter != nil || mtpDraftEncoder != nil || dflashDrafter != nil), let cb = session.lastCommitCB {
            cb.waitUntilCompleted()
            if cb.status != .completed {
                session.stateFailed = true
            }
            session.lastCommitCB = nil
        } else if let cb = session.lastCommitCB, cb.status == .error {
            session.stateFailed = true
        }
        let dflashReady = (dflashDrafter == nil) || (session.dflashSlot != nil && session.dflashCacheOffset == session.position)
        guard (session.stochastic != nil) == stochastic,
              session.stochastic?.stream.isRoundActive != true,
              session.stochastic?.stream.drawsConsumed == (stochastic ? session.roundRecord.samplerDrawIndex : nil),
              verifiedSession == nil, let verifyWindow,
              (mtpDraftEncoder != nil || echoDrafter != nil || dflashDrafter != nil),
              dflashReady,
              let mtpCache, let handle = session.mtpKV, let hidden = session.trunkHidden,
              let pending = session.roundRecord.pendingToken,
              !session.sessionFinished, !session.stateFailed, !session.verificationPending,
              session.trunkHiddenValid, session.position > 0,
              session.position <= ropeRows - verifyWindow.rows,
              session.roundRecord.consumedCount == session.position,
              session.roundRecord.absolutePosition == session.position,
              session.roundRecord.outputCount == session.generated,
              maxNew > 0, session.generated <= Int.max - verifyWindow.rows,
              (0..<geometry.vocab).contains(pending),
              stopIDs.allSatisfy({ (0..<geometry.vocab).contains($0) }) else {
            throw QWEngineDriverError.state("greedy round requires a seeded boundary and valid output budget")
        }
        let activeDraftEncoder = dflashDrafter == nil ? mtpDraftEncoder : nil
        let count = verifyWindow.rows - 1, basePairs = session.roundRecord.mtpPhysicalRows
        let nextNResolving = activeDraftEncoder != nil && echoDrafter == nil && !mtpStageProfiling
        let nextNMaxPairs = nextNResolving
            ? min(verifyWindow.rows, max(count, (activeDraftEncoder?.positionCapacity ?? ropeRows) - session.position))
            : count
        let pairSlotsNeeded = echoDrafter != nil ? verifyWindow.rows : nextNMaxPairs
        try kvCache.validate(session.kv)
        try mtpCache.validate(handle)
        guard kvCache.written(session.kv) == session.position,
              mtpCache.written(handle) == basePairs,
              basePairs >= 0, basePairs <= QWKVCache.maxTokensPerSeq - pairSlotsNeeded,
              session.position <= (activeDraftEncoder?.positionCapacity ?? ropeRows) - count else {
            throw QWEngineDriverError.state("greedy round cache or RoPE boundary mismatch")
        }
        try claimMLAIndexerState(session)
        // Reserve both windows before draft execution; never grow a cache while
        // one of its command buffers is in flight.
        if session.position + verifyWindow.rows > session.kv.tokenCapacity {
            try kvCache.grow(session.kv, to: session.position + verifyWindow.rows)
            session.attnKVViews = try kvCache.layerViewsAll(session.kv, layers: attnLayers)
        }
        if basePairs + pairSlotsNeeded > handle.tokenCapacity {
            try mtpCache.grow(handle, to: basePairs + pairSlotsNeeded)
            session.mtpKVViews = [try mtpCache.layerViews(handle, layer: 0)]
        }
        try kvCache.prepareWrites(session.kv, positions: session.position..<(session.position + verifyWindow.rows))
        try mtpCache.prepareWrites(handle, positions: basePairs..<(basePairs + pairSlotsNeeded))
        if session.attnKVViews.isEmpty {
            session.attnKVViews = try kvCache.layerViewsAll(session.kv, layers: attnLayers)
        }
        let views: (k: QWKVLayerView, v: QWKVLayerView)
        if let current = session.mtpKVViews.first {
            views = current
        } else {
            let v = try mtpCache.layerViews(handle, layer: 0)
            session.mtpKVViews = [v]
            views = v
        }
        let boundary = try greedyRoundBoundary(session)
        lastMTPRoundProfile = nil
        pendingMTPProfile = nil
        pendingMTPAdaptiveEntropy = mtpAdaptiveDepth ? [] : nil
        preparedStochasticWitness = nil
        lastRoundUsedPLD = false
        lastDraftedPairs = 0
        do {
            let owner = session.stochastic
            var proposals: [QWMTPGumbelProposal] = []
            var drafts: [Int] = []
            var draftedPairs = 0
            let draftAccumulator = profileStart.map { _ in QWMTPStageAccumulator() }
            let draftCompleted: ((any MTLCommandBuffer) throws -> Void)? = draftAccumulator.map { accumulator in
                { try accumulator.record($0) }
            }
            if let dflashDrafter {
                guard let dSlot = session.dflashSlot else {
                    throw QWEngineDriverError.state("DFlash slot unavailable for speculative round")
                }
                let resolvedStrategy = (try? QWDraftStrategy.resolve(mtpDraftStrategy)) ?? .hybrid
                let canAdapt = (owner == nil && !mtpStageProfiling && resolvedStrategy != .mtp)
                var usedPLD = false
                if canAdapt && activeConstraint == nil {
                    let ctx = pendingHistoryContext(session: session, pending: pending)
                    let minPldTokens = resolvedStrategy == .ngram ? min(4, count) : min(7, count)
                    var cand: [Int] = []
                    if let pld = QWContextCopyDrafter.periodicDrafts(
                        history: session.tokenHistory, extraToken: ctx.extraPending,
                        maxCount: count, minRepeats: 3),
                       pld.count >= minPldTokens {
                        cand = Array(pld.prefix(count))
                    } else if let match = QWContextCopyDrafter.drafts(
                        history: session.tokenHistory, extraToken: ctx.extraPending,
                        maxCount: count, minMatchLen: 6),
                        !match.tokens.isEmpty {
                        var extended = Array(match.tokens.prefix(count))
                        if extended.count < count {
                            session.tokenRecycler.appendDrafts(
                                into: &extended, maxCount: count,
                                previousToken: ctx.previousToken, pendingToken: pending,
                                maxUnigramFallbacks: 0)
                        }
                        if extended.count >= minPldTokens {
                            cand = extended
                        }
                    }
                    if cand.count > 7 && cand.count < 12 {
                        cand = Array(cand.prefix(7))
                    }
                    if !cand.isEmpty {
                        drafts = cand
                        proposals = []
                        draftedPairs = drafts.count
                        lastRoundUsedPLD = true
                        mtpDraftProposals += drafts.count
                        usedPLD = true
                    }
                }
                if usedPLD {
                    session.dflashRoundStartNs = nil
                } else {
                    let activeCount = canAdapt ? session.dflashPolicy.selectDraftCount(maxDrafts: count) : count
                    session.dflashRoundStartNs = canAdapt ? DispatchTime.now().uptimeNanoseconds : nil
                    if let owner {
                        try owner.stream.beginGumbelRound(drafts: activeCount, vocabulary: geometry.vocab, maxDrafts: 15)
                    }
                    let anchorMaskEmbed: ((any MTLComputeCommandEncoder, QWTensorValue, Int, Int, Int) throws -> Void)? =
                        self.mtpDeviceEmbedding.map { devEmb in
                            { enc, propX32, anchorToken, maskToken, rows in
                                dflashDrafter.encodeStagePropBF16(
                                    enc, tableBuffer: devEmb.buffer, byteOffset: devEmb.byteOffset,
                                    propX: propX32, anchorToken: anchorToken, maskToken: maskToken, rows: rows
                                )
                            }
                        }
                    let proposed: (drafts: [Int], proposals: [QWMTPGumbelProposal])
                    do {
                        proposed = try dflashDrafter.propose(
                            hal: hal, slot: dSlot, cacheOffset: session.dflashCacheOffset,
                            pendingToken: pending, draftCount: activeCount,
                            headStorage: headStorage,
                            encodeHeadLogits: { enc, xRot16, xNorm16, _, yLogits32, rows, vocabRows in
                                if let draftHead = self.mtpDraftHead {
                                    if rows == 1 {
                                        try draftHead.encode(enc, hal: self.hal, x: xRot16, y: yLogits32)
                                    } else {
                                        try draftHead.encodeBatched(enc, hal: self.hal, x: xRot16, y: yLogits32, m: rows)
                                    }
                                } else {
                                    switch self.headStorage {
                                    case .q4g(let hw):
                                        if self.hal.caps.isMetal4TensorCapable && rows >= 9
                                            && vocabRows % 64 == 0 && hw.view.k % 64 == 0 {
                                            try QWGemm.encodeNaxF32(
                                                enc, hal: self.hal, weight: hw.view,
                                                x16: xRot16, y16: xRot16, y32: yLogits32, rows: rows
                                            )
                                        } else {
                                            try self.head.encode(enc, storage: self.headStorage, x: xRot16, y: yLogits32, M: rows)
                                        }
                                    case .f16:
                                        try self.head.encode(enc, storage: self.headStorage, x: xNorm16, y: yLogits32, M: rows)
                                    }
                                }
                            },
                            encodeAnchorMaskEmbed: anchorMaskEmbed,
                            embedding: { try self.copyEmbedRow($0, to: $1) },
                            stochastic: owner.map { (Float($0.config.temperature), $0.stream) },
                            completed: draftCompleted)
                    } catch {
                        commandBufferErrors += 1
                        throw error
                    }
                    drafts = proposed.drafts
                    proposals = proposed.proposals
                    draftedPairs = drafts.count
                    mtpDraftProposals += drafts.count
                }
            } else if let echoDrafter {
                let proposed = try proposeEchoOrPLDRound(
                    session: session, echoDrafter: echoDrafter, hidden: hidden,
                    views: views, pending: pending, count: count, basePairs: basePairs,
                    owner: owner, draftCompleted: draftCompleted)
                drafts = proposed.drafts
                proposals = proposed.proposals
                draftedPairs = proposed.draftedPairs
            } else {
                guard let mtpDraftEncoder else {
                    throw QWEngineDriverError.state("MTP draft encoder is absent")
                }
                let proposed = try proposeNextNOrPLDRound(
                    session: session, draftEncoder: mtpDraftEncoder, hidden: hidden,
                    views: views, pending: pending, count: count, basePairs: basePairs,
                    owner: owner, draftCompleted: draftCompleted)
                drafts = proposed.drafts
                proposals = proposed.proposals
                draftedPairs = proposed.draftedPairs
            }
        

            mtpCache.markWritten(handle, upTo: basePairs + draftedPairs)
            let verifyStart = mtpStageProfiling ? DispatchTime.now().uptimeNanoseconds : nil
            let verifyAccumulator = verifyStart.map { _ in QWMTPStageAccumulator() }
            let verifyCompleted: ((any MTLCommandBuffer) throws -> Void)? = verifyAccumulator.map { accumulator in
                { try accumulator.record($0) }
            }
            let window = try verifyTokensLocked(session, tokens: [pending] + drafts, completed: verifyCompleted)
            let decision: QWGreedyRoundDecision
            let evidence: QWStochasticRoundEvidence?
            let margin = effectiveRelaxedMargin
            if let owner {
                var targets: [QWSamplingDistribution] = []
                for row in 0..<(drafts.count + 1) {
                    let history = session.tokenHistory + [pending] + Array(drafts.prefix(row))
                    targets.append(try owner.distribution(logits: window.logitRow(row),
                                                          repetitionHistory: Array(history)))
                }
                let result = margin > 0
                    ? try owner.stream.resolveRelaxedGumbelRound(proposals: proposals,
                                                                 targetDistributions: targets,
                                                                 margin: Double(margin))
                    : try owner.stream.resolveGumbelRound(proposals: proposals,
                                                          targetDistributions: targets)
                decision = try QWGreedyRoundDecision.sampled(geometry: geometry, 
                    proposals: proposals, result: result, maxNew: maxNew, stopIDs: stopIDs,
                    maxDrafts: dflashDrafter != nil ? 15 : 7)
                preparedStochasticWitness = QWStochasticRoundWitness(owner: owner,
                    drawEnd: owner.stream.drawsConsumed)
                evidence = QWStochasticRoundEvidence(proposals: proposals, targets: targets, result: result,
                    drawStart: boundary.record.samplerDrawIndex, drawEnd: owner.stream.drawsConsumed)
            } else {
                var rawDecision = try window.greedyDecision(drafts: drafts, maxNew: maxNew, stopIDs: stopIDs,
                                                            activeRows: drafts.count + 1,
                                                            relaxedMargin: margin,
                                                            relaxedMinProb: effectiveRelaxedMinProb)
                if let boundary = activeConstraint?.constraint.speculationBoundaryToken,
                   let idx = rawDecision.tokens.firstIndex(of: boundary),
                   idx + 1 < rawDecision.tokens.count {
                    rawDecision = QWGreedyRoundDecision(
                        acceptedDrafts: rawDecision.acceptedDrafts,
                        tokens: Array(rawDecision.tokens.prefix(idx + 1)),
                        stopped: false,
                        stoppedToken: nil
                    )
                }
                decision = rawDecision
                evidence = nil
            }
            let lease = QWGreedyRoundLease(boundary: boundary, decision: decision, stochasticEvidence: evidence)
            if let profileStart, let verifyStart, let draftAccumulator, let verifyAccumulator {
                let rawDraftHostMs = Double(verifyStart - profileStart) / 1_000_000
                let rawVerifyHostMs = Double(DispatchTime.now().uptimeNanoseconds - verifyStart) / 1_000_000
                pendingMTPProfile = (
                    try QWMTPStageTiming(hostMs: max(draftAccumulator.gpuMs, rawDraftHostMs),
                                         gpuMs: draftAccumulator.gpuMs, commandBuffers: draftAccumulator.commandBuffers),
                    try QWMTPStageTiming(hostMs: max(verifyAccumulator.gpuMs, rawVerifyHostMs),
                                         gpuMs: verifyAccumulator.gpuMs, commandBuffers: verifyAccumulator.commandBuffers))
            }
            preparedGreedyLease = lease
            return lease.prepared
        } catch {
            // A submitted prefix may have changed pair/trunk bytes. Without a
            // completed verified lease, do not expose an apparently retryable round.
            session.stateFailed = true
            session.stochastic?.stream.cancelActiveRound()
            if let owner = session.stochastic { session.roundRecord.samplerDrawIndex = owner.stream.drawsConsumed }
            throw error
        }
    }

    func commitPreparedVerification(_ session: QWEngineSession, round: QWPreparedGreedyRound,
                                            stochastic: Bool, drain: Bool = true,
                                            permit: QWGenerationPermit?) throws {
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        try requestGate.validate(permit)
        let profileStart = mtpStageProfiling ? DispatchTime.now().uptimeNanoseconds : nil
        guard (preparedStochasticWitness != nil) == stochastic,
              (session.stochastic != nil) == stochastic,
              verifiedSession === session, session.verificationPending,
              !session.stateFailed, !session.sessionFinished,
              let verifyWindow, let gdnReplayCheckpoint, let mtpCache, let mtpHandle = session.mtpKV,
              let preparedGreedyLease,
              let trunkHidden = session.trunkHidden, (2...verifyWindow.rows).contains(verifiedInputs.count),
              verifiedInputs.first == session.roundRecord.pendingToken else {
            throw QWEngineDriverError.state("greedy commit requires a live verified draft round")
        }
        try preparedGreedyLease.validate(round, current: greedyRoundBoundary(session))
        try preparedStochasticWitness?.validate(session)
        guard !mtpStageProfiling || pendingMTPProfile != nil else {
            throw QWEngineDriverError.state("MTP profile is missing its prepared stages")
        }
        let decision = round.decision
        let drafts = Array(verifiedInputs.dropFirst())
        if !stochastic {
            let (full, rawTargets) = try verifyWindow.greedyDecisionWithTargets(
                drafts: drafts, maxNew: verifiedInputs.count, activeRows: verifiedInputs.count,
                relaxedMargin: effectiveRelaxedMargin, relaxedMinProb: effectiveRelaxedMinProb)
            guard decision.acceptedDrafts == full.acceptedDrafts,
                  decision.tokens == Array(full.tokens.prefix(decision.tokens.count)),
                  decision.tokens.count <= full.tokens.count else {
                throw QWEngineDriverError.state("greedy commit decision does not match verified target rows")
            }
            if effectiveRelaxedMargin == 0, activeConstraint == nil {
                let pendingInHistory = session.tokenHistory.count > session.position
                    && session.tokenHistory.last == verifiedInputs[0]
                let prevBeforePending: Int? = pendingInHistory
                    ? (session.tokenHistory.count >= 2 ? session.tokenHistory[session.tokenHistory.count - 2] : nil)
                    : session.tokenHistory.last
                for row in (0..<min(verifiedInputs.count, rawTargets.count)).reversed() {
                    let prev = (row == 0) ? prevBeforePending : verifiedInputs[row - 1]
                    session.tokenRecycler.recordTransition(
                        previous: prev, current: verifiedInputs[row], next: rawTargets[row])
                }
            }
        }
        guard session.generated <= Int.max - decision.tokens.count else {
            throw QWEngineDriverError.state("round output count overflows")
        }
        try kvCache.validate(session.kv)
        try mtpCache.validate(mtpHandle)
        let basePairs = session.roundRecord.mtpPhysicalRows
        let usesManagedPairs = echoDrafter != nil || (mtpDraftEncoder != nil && dflashDrafter == nil
            && (lastRoundUsedPLD || lastDraftedPairs < drafts.count || (mtpQuadraticSeeding && !mtpStageProfiling)))
        let draftedPairs = (echoDrafter != nil || (mtpDraftEncoder != nil && dflashDrafter == nil))
            ? lastDraftedPairs : drafts.count
        let maxAddedPairs = echoDrafter != nil ? verifiedInputs.count
            : (usesManagedPairs && mtpDraftEncoder != nil
                ? min(verifiedInputs.count, max(drafts.count, mtpDraftEncoder!.positionCapacity - session.position))
                : drafts.count)
        guard basePairs >= 0, basePairs <= QWKVCache.maxTokensPerSeq - maxAddedPairs,
              kvCache.written(session.kv) == session.position + verifiedInputs.count,
              mtpCache.written(mtpHandle) == basePairs + draftedPairs else {
            throw QWEngineDriverError.state("greedy commit requires complete trunk and draft pair windows")
        }
        let retained = decision.retainedRows
        let retainedPairs: Int
        if echoDrafter != nil {
            retainedPairs = retained
        } else if usesManagedPairs, let mtpDraftEncoder {
            retainedPairs = min(retained, max(0, mtpDraftEncoder.positionCapacity - session.position))
        } else {
            retainedPairs = try decision.retainedPairRows(drafts: drafts.count)
        }
        guard let cb = hal.queue.makeCommandBuffer() else { throw QWEngineDriverError.state("greedy commit command buffer") }
        if let echoDrafter {
            try encodeEchoCommitResolution(cb, session: session, echoDrafter: echoDrafter,
                                           window: verifyWindow, trunkHidden: trunkHidden,
                                           basePairs: basePairs, retained: retained,
                                           usedPLD: lastRoundUsedPLD)
        } else if usesManagedPairs, let mtpDraftEncoder {
            try encodeNextNCommitResolution(cb, session: session, draft: mtpDraftEncoder,
                                            window: verifyWindow, trunkHidden: trunkHidden,
                                            basePairs: basePairs, retained: retained,
                                            usedPLD: lastRoundUsedPLD)
        }
        try verifyWindow.encodeRetainedPrefix(cb, hal: hal, checkpoint: gdnReplayCheckpoint, decoder: gdnDecode,
                                              rec: session.recSlab, ring: session.ringSlab,
                                              trunkHidden: trunkHidden, activeRows: verifiedInputs.count,
                                              retaining: retained)
        if mtpStageProfiling && retained == 0 && geometry.gdnLayerCount == 0 {
            let src4 = QWTensorValue(region: verifyWindow.hidden.region, offset: verifyWindow.hidden.offset,
                                     shape: QWTensorShape([2], .f16))
            let dst4 = QWTensorValue(region: verifyWindow.embeddings.region, offset: verifyWindow.embeddings.offset,
                                     shape: QWTensorShape([2], .f16))
            if let enc = cb.makeComputeCommandEncoder() {
                defer { enc.endEncoding() }
                try QWVerifyWindow.encodeCopy(enc, hal: hal, source: src4, destination: dst4)
            }
        }
        if let dflashDrafter, let dSlot = session.dflashSlot, retained > 0 {
            try dflashDrafter.encodeContextUpdate(cb, slot: dSlot, startPosition: session.position, totalCount: retained)
        }
        if !mtpStageProfiling && !drain {
            session.lastCommitCB = cb
            cb.addCompletedHandler { [weak self, weak session] buffer in
                if buffer.status != .completed {
                    self?.commandBufferErrors += 1
                    session?.stateFailed = true
                }
            }
        }
        cb.commit()
        if mtpStageProfiling || drain {
            cb.waitUntilCompleted()
            guard cb.status == .completed else {
                commandBufferErrors += 1
                session.stateFailed = true
                throw QWEngineDriverError.state("greedy commit GPU failure: \(String(describing: cb.error))")
            }
        }
        let usedPLDThisRound = lastRoundUsedPLD
        let resolutionGPU: Double?
        do {
            resolutionGPU = mtpStageProfiling ? try QWMTPStageTiming.gpuDurationMs(cb) : nil
            let trunkRollback = verifiedInputs.count - retained
            if trunkRollback > 0 {
                try kvCache.rollback(session.kv, by: trunkRollback)
                session.attnKVViews = try kvCache.layerViewsAll(session.kv, layers: attnLayers)
                mlaOperator?.truncatePosition(session.position + retained)
                hcaOperator?.truncatePosition(session.position + retained)
            }
            if echoDrafter != nil || usesManagedPairs {
                try finalizeMTPPairCursor(session: session, mtpCache: mtpCache, mtpHandle: mtpHandle,
                                          basePairs: basePairs, draftedPairs: draftedPairs,
                                          retainedPairs: retainedPairs)
            } else {
                let pairRollback = drafts.count - retainedPairs
                if pairRollback > 0 {
                    try mtpCache.rollback(mtpHandle, by: pairRollback)
                    session.mtpKVViews = [try mtpCache.layerViews(mtpHandle, layer: 0)]
                }
            }
            lastRoundUsedPLD = false
            lastDraftedPairs = 0
        

        } catch {
            // GPU state has already resolved; partly changed host tables cannot
            // be retried as a fresh round.
            session.stateFailed = true
            throw error
        }
        let profile: QWMTPRoundProfile?
        do {
            if let profileStart, let resolutionGPU {
                guard let pendingMTPProfile else {
                    throw QWEngineDriverError.state("MTP profile is missing its prepared stages")
                }
                let rawResHostMs = Double(DispatchTime.now().uptimeNanoseconds - profileStart) / 1_000_000
                profile = QWMTPRoundProfile(drafts: drafts.count, retainedRows: retained,
                    draft: pendingMTPProfile.draft, verification: pendingMTPProfile.verification,
                    resolution: try QWMTPStageTiming(
                        hostMs: max(resolutionGPU, rawResHostMs),
                        gpuMs: resolutionGPU, commandBuffers: 1),
                    draftTokens: drafts, acceptedDrafts: decision.acceptedDrafts,
                    backend: mtpBackend.rawValue,
                    verificationGpuMsByStage: verificationGpuMsByStage,
                    adaptiveEntropyByDraft: pendingMTPAdaptiveEntropy)
            } else {
                profile = nil
            }
        } catch {
            session.stateFailed = true
            throw error
        }
        session.position += retained
        if dflashDrafter != nil, session.dflashSlot != nil {
            session.dflashCacheOffset = session.position
        }
        session.generated += decision.tokens.count
        session.roundRecord.consumedCount = session.position
        session.roundRecord.absolutePosition = session.position
        session.roundRecord.outputCount = session.generated
        session.roundRecord.mtpPhysicalRows = basePairs + retainedPairs
        mtpDraftAccepted += decision.acceptedDrafts
        if dflashDrafter != nil {
            if !stochastic && !mtpStageProfiling && !usedPLDThisRound {
                let roundMs = session.dflashRoundStartNs.map {
                    Double(DispatchTime.now().uptimeNanoseconds - $0) / 1_000_000
                }
                session.dflashPolicy.observeRound(
                    draftCount: drafts.count, accepted: decision.acceptedDrafts, roundMs: roundMs)
            }
            session.dflashRoundStartNs = nil
        }
    

        session.tokenHistory.append(contentsOf: decision.tokens)
        if let active = activeConstraint {
            for tok in decision.tokens {
                try active.constraint.accept(tok)
            }
        }
        if let owner = session.stochastic { session.roundRecord.samplerDrawIndex = owner.stream.drawsConsumed }
        if let pending = decision.pendingToken { session.roundRecord.pendingToken = pending }
        if retained > 0 {
            session.roundRecord.trunkHiddenSeed = nil // GPU carry is authoritative.
        }
        session.verificationPending = false
        verifiedSession = nil
        verifiedInputs.removeAll(keepingCapacity: true)
        self.preparedGreedyLease = nil
        preparedStochasticWitness = nil
        pendingMTPProfile = nil
        pendingMTPAdaptiveEntropy = nil
        lastMTPRoundProfile = profile
    }

}
