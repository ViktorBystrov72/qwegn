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
        if (echoDrafter != nil || mtpDraftEncoder != nil), let cb = session.lastCommitCB {
            cb.waitUntilCompleted()
            if cb.status != .completed {
                session.stateFailed = true
            }
            session.lastCommitCB = nil
        } else if let cb = session.lastCommitCB, cb.status == .error {
            session.stateFailed = true
        }
        guard (session.stochastic != nil) == stochastic,
              session.stochastic?.stream.isRoundActive != true,
              session.stochastic?.stream.drawsConsumed == (stochastic ? session.roundRecord.samplerDrawIndex : nil),
              verifiedSession == nil, let verifyWindow,
              (mtpDraftEncoder != nil || echoDrafter != nil || dflashDrafter != nil),
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
        let count = verifyWindow.rows - 1, basePairs = session.roundRecord.mtpPhysicalRows
        let nextNResolving = mtpDraftEncoder != nil && echoDrafter == nil && dflashDrafter == nil && !mtpStageProfiling
        let nextNMaxPairs = nextNResolving
            ? min(verifyWindow.rows, max(count, (mtpDraftEncoder?.positionCapacity ?? ropeRows) - session.position))
            : count
        let pairSlotsNeeded = echoDrafter != nil ? verifyWindow.rows : nextNMaxPairs
        try kvCache.validate(session.kv)
        try mtpCache.validate(handle)
        guard kvCache.written(session.kv) == session.position,
              mtpCache.written(handle) == basePairs,
              basePairs >= 0, basePairs <= QWKVCache.maxTokensPerSeq - pairSlotsNeeded,
              session.position <= (mtpDraftEncoder?.positionCapacity ?? ropeRows) - count else {
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
            if let dflashDrafter, let dSlot = session.dflashSlot {
                if let owner {
                    try owner.stream.beginGumbelRound(drafts: count, vocabulary: geometry.vocab, maxDrafts: 15)
                }
                let proposed: (drafts: [Int], proposals: [QWMTPGumbelProposal])
                do {
                    proposed = try dflashDrafter.propose(
                        hal: hal, slot: dSlot, cacheOffset: session.dflashCacheOffset,
                        pendingToken: pending, draftCount: count, head: head,
                        headStorage: headStorage, draftHead: mtpDraftHead,
                        embedding: { try self.copyEmbedRow($0, to: $1) },
                        stochasticOwner: owner, completed: draftCompleted)
                } catch {
                    commandBufferErrors += 1
                    throw error
                }
                drafts = proposed.drafts
                proposals = proposed.proposals
                draftedPairs = drafts.count
                mtpDraftProposals += drafts.count
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
                    maxDrafts: dflashDrafter != nil ? 15 : 5)
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
            let full = try verifyWindow.greedyDecision(drafts: drafts, maxNew: verifiedInputs.count,
                                                       activeRows: verifiedInputs.count,
                                                       relaxedMargin: effectiveRelaxedMargin,
                                                       relaxedMinProb: effectiveRelaxedMinProb)
            guard decision.acceptedDrafts == full.acceptedDrafts,
                  decision.tokens == Array(full.tokens.prefix(decision.tokens.count)),
                  decision.tokens.count <= full.tokens.count else {
                throw QWEngineDriverError.state("greedy commit decision does not match verified target rows")
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
        if dflashDrafter != nil {
            session.dflashCacheOffset = session.position
        }
        session.generated += decision.tokens.count
        session.roundRecord.consumedCount = session.position
        session.roundRecord.absolutePosition = session.position
        session.roundRecord.outputCount = session.generated
        session.roundRecord.mtpPhysicalRows = basePairs + retainedPairs
        mtpDraftAccepted += decision.acceptedDrafts
    

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
