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
        if lsaOperator != nil,
           session.position < lsaConfig.denseBypassThreshold,
           session.position + verifyWindow.rows > lsaConfig.denseBypassThreshold {
            return false
        }
        if isWeightFreeSpeculation {
            guard (activeConstraint?.constraint.isUnconstrainedNow ?? true),
                  let pending = session.roundRecord.pendingToken,
                  proposeWeightFreeDrafts(session: session, pending: pending, count: count) != nil else {
                return false
            }
        }
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
        let profileStart = mtpStageProfiling ? DispatchTime.now().uptimeNanoseconds : nil
        let canOverlapPriorCommit = !stochastic && !mtpStageProfiling && !mtpVerificationProfiling
            && coldEvents == nil && mtpDeviceEmbedding != nil
            && (dflashDrafter != nil || echoDrafter != nil || mtpDraftEncoder != nil)
        var pendingCommitCB: (any MTLCommandBuffer)?
        if let cb = session.lastCommitCB {
            session.lastCommitCB = nil
            if canOverlapPriorCommit {
                if cb.status == .error {
                    commandBufferErrors += 1
                    session.stateFailed = true
                }
                pendingCommitCB = cb
            } else if echoDrafter != nil || mtpDraftEncoder != nil || dflashDrafter != nil || isWeightFreeSpeculation {
                cb.waitUntilCompleted()
                if cb.status != .completed {
                    session.stateFailed = true
                }
            } else if cb.status == .error {
                session.stateFailed = true
            }
        }
        func drainPendingCommitIfNeeded() throws {
            if let cb = pendingCommitCB {
                pendingCommitCB = nil
                try waitForPriorCommitCB(cb, session: session)
            }
        }
        let dflashReady = (dflashDrafter == nil) || (session.dflashSlot != nil && session.dflashCacheOffset == session.position)
        let hasValidMTPCache = (mtpCache != nil && session.mtpKV != nil)
            || (isWeightFreeSpeculation && mtpCache == nil && session.mtpKV == nil)
        guard (session.stochastic != nil) == stochastic,
              session.stochastic?.stream.isRoundActive != true,
              session.stochastic?.stream.drawsConsumed == (stochastic ? session.roundRecord.samplerDrawIndex : nil),
              verifiedSession == nil, let verifyWindow,
              (mtpDraftEncoder != nil || echoDrafter != nil || dflashDrafter != nil || isWeightFreeSpeculation),
              dflashReady, hasValidMTPCache,
              let hidden = session.trunkHidden,
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
            pendingCommitCB?.waitUntilCompleted()
            throw QWEngineDriverError.state("greedy round requires a seeded boundary and valid output budget")
        }
        let activeDraftEncoder = dflashDrafter == nil ? mtpDraftEncoder : nil
        let count = verifyWindow.rows - 1, basePairs = session.roundRecord.mtpPhysicalRows
        let nextNResolving = activeDraftEncoder != nil && echoDrafter == nil && !mtpStageProfiling
        let nextNMaxPairs = nextNResolving
            ? min(verifyWindow.rows, max(count, (activeDraftEncoder?.positionCapacity ?? ropeRows) - session.position))
            : count
        let pairSlotsNeeded = echoDrafter != nil ? verifyWindow.rows : nextNMaxPairs
        do {
            try kvCache.validate(session.kv)
            if let mtpCache, let handle = session.mtpKV {
                try mtpCache.validate(handle)
                guard mtpCache.written(handle) == basePairs,
                      basePairs >= 0, basePairs <= QWKVCache.maxTokensPerSeq - pairSlotsNeeded else {
                    throw QWEngineDriverError.state("greedy round cache or RoPE boundary mismatch")
                }
            }
        } catch {
            pendingCommitCB?.waitUntilCompleted()
            throw error
        }
        guard kvCache.written(session.kv) == session.position,
              session.position <= (activeDraftEncoder?.positionCapacity ?? ropeRows) - count else {
            pendingCommitCB?.waitUntilCompleted()
            throw QWEngineDriverError.state("greedy round cache or RoPE boundary mismatch")
        }
        let needsSparseClaim = (hcaOperator.map { $0.stateOwner !== session } ?? false)
            || (mlaOperator.map { $0.stateOwner !== session } ?? false)
            || (lsaOperator.map { $0.stateOwner !== session } ?? false)
            || (mbsaOperator.map { $0.stateOwner !== session } ?? false)
        let needsMTPCacheWait = (mtpCache != nil && session.mtpKV != nil) && (
            basePairs + pairSlotsNeeded > session.mtpKV!.tokenCapacity
            || mtpCache!.hasSharedPages(session.mtpKV!, positions: basePairs..<(basePairs + pairSlotsNeeded))
        )
        if needsSparseClaim
            || session.position + verifyWindow.rows > session.kv.tokenCapacity
            || needsMTPCacheWait {
            try drainPendingCommitIfNeeded()
        }
        try claimMLAIndexerState(session)
        // Reserve both windows before draft execution; never grow a cache while
        // one of its command buffers is in flight.
        if session.position + verifyWindow.rows > session.kv.tokenCapacity {
            try kvCache.grow(session.kv, to: session.position + verifyWindow.rows)
            session.attnKVViews = try kvCache.layerViewsAll(session.kv, layers: attnLayers)
        }
        try kvCache.prepareWrites(session.kv, positions: session.position..<(session.position + verifyWindow.rows))
        if session.attnKVViews.isEmpty {
            session.attnKVViews = try kvCache.layerViewsAll(session.kv, layers: attnLayers)
        }
        let views: (k: QWKVLayerView, v: QWKVLayerView)?
        if let mtpCache, let handle = session.mtpKV {
            if basePairs + pairSlotsNeeded > handle.tokenCapacity {
                try mtpCache.grow(handle, to: basePairs + pairSlotsNeeded)
                session.mtpKVViews = [try mtpCache.layerViews(handle, layer: 0)]
            }
            try mtpCache.prepareWrites(handle, positions: basePairs..<(basePairs + pairSlotsNeeded))
            if let current = session.mtpKVViews.first {
                views = current
            } else {
                let v = try mtpCache.layerViews(handle, layer: 0)
                session.mtpKVViews = [v]
                views = v
            }
        } else {
            views = nil
        }
        let boundary = try greedyRoundBoundary(session)
        lastMTPRoundProfile = nil
        pendingMTPProfile = nil
        pendingMTPAdaptiveEntropy = (mtpAdaptiveDepth && !stochastic) ? [] : nil
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
            let verifyStart = mtpStageProfiling ? DispatchTime.now().uptimeNanoseconds : nil
            let verifyAccumulator = verifyStart.map { _ in QWMTPStageAccumulator() }
            let verifyCompleted: ((any MTLCommandBuffer) throws -> Void)? = verifyAccumulator.map { accumulator in
                { try accumulator.record($0) }
            }
            let window: QWVerifyWindow
            if owner == nil, !mtpStageProfiling, !mtpVerificationProfiling, coldEvents == nil,
               !mtpAdaptiveDepth, activeConstraint == nil, dflashDrafter == nil, !isWeightFreeSpeculation,
               let mtpCache, let handle = session.mtpKV, let views,
               let table = mtpDeviceEmbedding,
               let draftPlan = try prepareAsyncNeuralDraft(
                   session: session, hidden: hidden, views: views, pending: pending,
                   count: count, basePairs: basePairs, mtpCache: mtpCache, mtpHandle: handle,
                   table: table) {
                let commitCB = pendingCommitCB
                pendingCommitCB = nil
                switch draftPlan {
                case .pld(let chosen):
                    window = try verifyTokensCore(
                        session, explicitTokens: [pending] + chosen, rowCount: 1 + chosen.count,
                        pendingToken: nil, useDeviceEmbed: table, priorCommitCB: commitCB,
                        inFlightDraftCBs: []).window
                    drafts = chosen
                    draftedPairs = 0
                    lastRoundUsedPLD = true
                    lastDraftedPairs = 0
                    mtpDraftProposals += chosen.count
                case .asyncNeural(let draftCBs, let neuralCount):
                    let (verifiedWindow, resolvedTokens) = try verifyTokensCore(
                        session, explicitTokens: nil, rowCount: 1 + neuralCount,
                        pendingToken: pending, useDeviceEmbed: table,
                        priorCommitCB: commitCB, inFlightDraftCBs: draftCBs)
                    window = verifiedWindow
                    drafts = Array(resolvedTokens.dropFirst())
                    draftedPairs = neuralCount
                    lastRoundUsedPLD = false
                    lastDraftedPairs = neuralCount
                    mtpDraftProposals += drafts.count
                }
            } else {
                if owner != nil || mtpAdaptiveDepth || mtpDeviceEmbedding == nil || coldEvents != nil {
                    try drainPendingCommitIfNeeded()
                }
                if let dflashDrafter {
                    guard let dSlot = session.dflashSlot else {
                        try? drainPendingCommitIfNeeded()
                        throw QWEngineDriverError.state("DFlash slot unavailable for speculative round")
                    }
                    let proposed = try proposeDFlashOrPLDRound(
                        session: session, dflashDrafter: dflashDrafter, dSlot: dSlot,
                        pending: pending, count: count, owner: owner, draftCompleted: draftCompleted)
                    drafts = proposed.drafts
                    proposals = proposed.proposals
                    draftedPairs = proposed.draftedPairs
                } else if let echoDrafter {
                    guard let views else {
                        try? drainPendingCommitIfNeeded()
                        throw QWEngineDriverError.state("ECHO round requires MTP KV views")
                    }
                    let proposed = try proposeEchoOrPLDRound(
                        session: session, echoDrafter: echoDrafter, hidden: hidden,
                        views: views, pending: pending, count: count, basePairs: basePairs,
                        owner: owner, draftCompleted: draftCompleted)
                    drafts = proposed.drafts
                    proposals = proposed.proposals
                    draftedPairs = proposed.draftedPairs
                } else if isWeightFreeSpeculation {
                    guard let chosen = proposeWeightFreeDrafts(session: session, pending: pending, count: count) else {
                        try? drainPendingCommitIfNeeded()
                        throw QWEngineDriverError.state("weight-free round had no context or recycler drafts")
                    }
                    drafts = chosen
                    proposals = []
                    draftedPairs = 0
                    lastRoundUsedPLD = true
                    mtpDraftProposals += drafts.count
                } else {
                    guard let mtpDraftEncoder, let views else {
                        try? drainPendingCommitIfNeeded()
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
                if let mtpCache, let handle = session.mtpKV {
                    mtpCache.markWritten(handle, upTo: basePairs + draftedPairs)
                }
                if canOverlapPriorCommit {
                    let embedTable = mtpDeviceEmbedding
                    let commitCB = pendingCommitCB
                    pendingCommitCB = nil
                    window = try verifyTokensCore(
                        session, explicitTokens: [pending] + drafts, rowCount: 1 + drafts.count,
                        pendingToken: nil, useDeviceEmbed: embedTable,
                        priorCommitCB: commitCB, inFlightDraftCBs: [], completed: verifyCompleted).window
                } else {
                    try drainPendingCommitIfNeeded()
                    window = try verifyTokensLocked(session, tokens: [pending] + drafts, completed: verifyCompleted)
                }
            }
            var rawDecision: QWGreedyRoundDecision
            let evidence: QWStochasticRoundEvidence?
            let rawTargetsForLease: [Int]?
            let margin: Float = lastRoundUsedPLD ? 0 : effectiveRelaxedMargin
            let minProb: Float = margin > 0 ? effectiveRelaxedMinProb : 0
            if let owner {
                let ctx = pendingHistoryContext(session: session, pending: pending)
                let extraPending = ctx.extraPending.map { [$0] } ?? []
                if proposals.isEmpty {
                    var targetTokens = [Int](repeating: 0, count: drafts.count + 1)
                    for row in 0..<(drafts.count + 1) {
                        let history = session.activeRepetitionHistory(extraTokens: extraPending + Array(drafts.prefix(row)))
                        let dist = try owner.distribution(logits: window.logitRow(row), repetitionHistory: history)
                        let sampled = try owner.stream.sampleTarget(dist)
                        targetTokens[row] = sampled
                        if row < drafts.count && sampled != drafts[row] { break }
                    }
                    rawDecision = try QWGreedyRoundDecision.make(
                        geometry: geometry, drafts: drafts, targetTokens: targetTokens,
                        maxNew: maxNew, stopIDs: stopIDs)
                    preparedStochasticWitness = QWStochasticRoundWitness(
                        owner: owner, drawEnd: owner.stream.drawsConsumed)
                    evidence = nil
                } else {
                    var targets: [QWSamplingDistribution] = []
                    for row in 0..<(drafts.count + 1) {
                        let history = session.activeRepetitionHistory(extraTokens: extraPending + Array(drafts.prefix(row)))
                        targets.append(try owner.distribution(logits: window.logitRow(row),
                                                              repetitionHistory: history))
                    }
                    let result = margin > 0
                        ? try owner.stream.resolveRelaxedGumbelRound(proposals: proposals,
                                                                     targetDistributions: targets,
                                                                     margin: Double(margin))
                        : try owner.stream.resolveGumbelRound(proposals: proposals,
                                                              targetDistributions: targets)
                    rawDecision = try QWGreedyRoundDecision.sampled(geometry: geometry,
                        proposals: proposals, result: result, maxNew: maxNew, stopIDs: stopIDs,
                        maxDrafts: dflashDrafter != nil ? 15 : 7)
                    preparedStochasticWitness = QWStochasticRoundWitness(owner: owner,
                        drawEnd: owner.stream.drawsConsumed)
                    evidence = QWStochasticRoundEvidence(proposals: proposals, targets: targets, result: result,
                        drawStart: boundary.record.samplerDrawIndex, drawEnd: owner.stream.drawsConsumed)
                }
                rawTargetsForLease = nil
            } else {
                let withTargets = try window.greedyDecisionWithTargets(
                    drafts: drafts, maxNew: maxNew, stopIDs: stopIDs,
                    activeRows: drafts.count + 1,
                    relaxedMargin: margin,
                    relaxedMinProb: minProb)
                rawDecision = withTargets.decision
                rawTargetsForLease = withTargets.targets
                evidence = nil
            }
            if let boundaryTok = activeConstraint?.constraint.speculationBoundaryToken,
               let idx = rawDecision.tokens.firstIndex(of: boundaryTok),
               idx + 1 < rawDecision.tokens.count {
                rawDecision = QWGreedyRoundDecision(
                    acceptedDrafts: rawDecision.acceptedDrafts,
                    tokens: Array(rawDecision.tokens.prefix(idx + 1)),
                    stopped: false,
                    stoppedToken: nil
                )
            }
            let decision = rawDecision
            let lease = QWGreedyRoundLease(boundary: boundary, decision: decision,
                                           stochasticEvidence: evidence, rawTargets: rawTargetsForLease)
            if let profileStart, let verifyStart, let draftAccumulator, let verifyAccumulator {
                let rawDraftHostMs = Double(verifyStart - profileStart) / 1_000_000
                let rawVerifyHostMs = Double(DispatchTime.now().uptimeNanoseconds - verifyStart) / 1_000_000
                let draftTiming: QWMTPStageTiming
                if isWeightFreeSpeculation && draftAccumulator.commandBuffers == 0 {
                    draftTiming = try QWMTPStageTiming.hostOnly(hostMs: rawDraftHostMs)
                } else {
                    draftTiming = try QWMTPStageTiming(
                        hostMs: max(draftAccumulator.gpuMs, rawDraftHostMs),
                        gpuMs: draftAccumulator.gpuMs, commandBuffers: draftAccumulator.commandBuffers)
                }
                pendingMTPProfile = (
                    draftTiming,
                    try QWMTPStageTiming(hostMs: max(verifyAccumulator.gpuMs, rawVerifyHostMs),
                                         gpuMs: verifyAccumulator.gpuMs, commandBuffers: verifyAccumulator.commandBuffers))
            }
            preparedGreedyLease = lease
            return lease.prepared
        } catch {
            // A submitted prefix may have changed pair/trunk bytes. Without a
            // completed verified lease, do not expose an apparently retryable round.
            try? drainPendingCommitIfNeeded()
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
        let hasValidMTPCache = (mtpCache != nil && session.mtpKV != nil)
            || (isWeightFreeSpeculation && mtpCache == nil && session.mtpKV == nil)
        guard (preparedStochasticWitness != nil) == stochastic,
              (session.stochastic != nil) == stochastic,
              verifiedSession === session, session.verificationPending,
              !session.stateFailed, !session.sessionFinished,
              let verifyWindow, let gdnReplayCheckpoint, hasValidMTPCache,
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
            let commitMargin: Float = lastRoundUsedPLD ? 0 : effectiveRelaxedMargin
            let commitMinProb: Float = commitMargin > 0 ? effectiveRelaxedMinProb : 0
            let rawTargets: [Int]
            if let cachedTargets = preparedGreedyLease.rawTargets, cachedTargets.count == verifiedInputs.count {
                rawTargets = cachedTargets
            } else {
                let (full, computedTargets) = try verifyWindow.greedyDecisionWithTargets(
                    drafts: drafts, maxNew: verifiedInputs.count, activeRows: verifiedInputs.count,
                    relaxedMargin: commitMargin, relaxedMinProb: commitMinProb)
                guard decision.acceptedDrafts == full.acceptedDrafts,
                      decision.tokens == Array(full.tokens.prefix(decision.tokens.count)),
                      decision.tokens.count <= full.tokens.count else {
                    throw QWEngineDriverError.state("greedy commit decision does not match verified target rows")
                }
                rawTargets = computedTargets
            }
            if commitMargin == 0, (activeConstraint?.constraint.isUnconstrainedNow ?? true) {
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
        if let mtpCache, let mtpHandle = session.mtpKV {
            try mtpCache.validate(mtpHandle)
        }
        let basePairs = session.roundRecord.mtpPhysicalRows
        let usesManagedPairs = echoDrafter != nil || (mtpDraftEncoder != nil && dflashDrafter == nil
            && (lastRoundUsedPLD || lastDraftedPairs < drafts.count || (mtpQuadraticSeeding && !mtpStageProfiling)))
        let draftedPairs = (echoDrafter != nil || (mtpDraftEncoder != nil && dflashDrafter == nil))
            ? lastDraftedPairs : drafts.count
        let maxAddedPairs = echoDrafter != nil ? verifiedInputs.count
            : (usesManagedPairs && mtpDraftEncoder != nil
                ? min(verifiedInputs.count, max(drafts.count, mtpDraftEncoder!.positionCapacity - session.position))
                : drafts.count)
        guard kvCache.written(session.kv) == session.position + verifiedInputs.count else {
            throw QWEngineDriverError.state("greedy commit requires complete trunk and draft pair windows")
        }
        if let mtpCache, let mtpHandle = session.mtpKV {
            guard basePairs >= 0, basePairs <= QWKVCache.maxTokensPerSeq - maxAddedPairs,
                  mtpCache.written(mtpHandle) == basePairs + draftedPairs else {
                throw QWEngineDriverError.state("greedy commit requires complete trunk and draft pair windows")
            }
        }
        let retained = decision.retainedRows
        let retainedPairs: Int
        if echoDrafter != nil {
            retainedPairs = retained
        } else if usesManagedPairs, let mtpDraftEncoder {
            retainedPairs = min(retained, max(0, mtpDraftEncoder.positionCapacity - session.position))
        } else if mtpCache != nil {
            retainedPairs = try decision.retainedPairRows(drafts: drafts.count)
        } else {
            retainedPairs = 0
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
            if let mtpCache, let mtpHandle = session.mtpKV {
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
            if !mtpStageProfiling && !usedPLDThisRound {
                let roundMs = session.dflashRoundStartNs.map {
                    Double(DispatchTime.now().uptimeNanoseconds - $0) / 1_000_000
                }
                session.dflashPolicy.observeRound(
                    draftCount: drafts.count, accepted: decision.acceptedDrafts, roundMs: roundMs)
            }
            session.dflashRoundStartNs = nil
        }
    

        if stochastic, (activeConstraint?.constraint.isUnconstrainedNow ?? true) {
            for tok in decision.tokens {
                if let prev1 = session.tokenHistory.last {
                    let prev2 = session.tokenHistory.count >= 2 ? session.tokenHistory[session.tokenHistory.count - 2] : nil
                    session.tokenRecycler.recordTransition(previous: prev2, current: prev1, next: tok)
                }
                session.tokenHistory.append(tok)
            }
        } else {
            session.tokenHistory.append(contentsOf: decision.tokens)
        }
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
