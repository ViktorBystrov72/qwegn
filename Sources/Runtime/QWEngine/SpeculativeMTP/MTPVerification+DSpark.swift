// MTPVerification+DSpark.swift — DSpark tap capture, PLD/Markov/confidence proposal, and async round integration.
import Foundation
import Metal
import QWAlloc
import QWCore
import QWHAL
import QWMTP

extension QWEngineDriver {
    func encodeDSparkTargetTapIfNeeded(
        _ enc: any MTLComputeCommandEncoder,
        physicalLayer p: Int,
        nextPhysicalLayer nextP: Int?,
        mhcStream: QWTensorValue?,
        mhcControls: QWTensorValue?,
        branchDn16: QWTensorValue,
        residualX16: QWTensorValue,
        srcRowOffset: Int,
        dstRowOffset: Int,
        rows: Int
    ) {
        guard let ds = dsparkDrafter, p < ds.targetLayerSlotMap.count else { return }
        let slot = ds.targetLayerSlotMap[p]
        guard slot >= 0 else { return }
        enc.useResource(ds.tapBuf.region.buffer, usage: [.read, .write])
        if let _ = layersW[p].mhc, let mhcCfg = geometry.mhcConfig,
           let mhcStream, let mhcControls {
            let isLeave = (p == mhcCfg.end - 1)
            let nextIsMHC = nextP.map { $0 >= mhcCfg.start && $0 < mhcCfg.end } ?? false
            let deferPostFFN = !isLeave && nextIsMHC && (nextP != mhcCfg.start)
            if deferPostFFN { enc.memoryBarrier(scope: .buffers) }
            ds.encodeCaptureTargetTap(
                enc, base: mhcStream,
                srcRowOffset: srcRowOffset, dstRowOffset: dstRowOffset, rows: rows,
                slot: slot, hc: mhcCfg.streams,
                deferredInject: deferPostFFN ? (src: branchDn16, inj: mhcControls, isMHC: true) : nil,
                singleStream: false
            )
        } else {
            ds.encodeCaptureTargetTap(
                enc, base: residualX16,
                srcRowOffset: srcRowOffset, dstRowOffset: dstRowOffset, rows: rows,
                slot: slot, hc: 1,
                deferredInject: nil,
                singleStream: true
            )
        }
        enc.memoryBarrier(scope: .buffers)
    }

    func selectDSparkPLDCandidates(
        session: QWEngineSession,
        pending: Int,
        count: Int,
        resolvedStrategy: QWDraftStrategy
    ) -> [Int] {
        let ctx = pendingHistoryContext(session: session, pending: pending)
        let capped = min(count, 15)
        let minPldTokens = resolvedStrategy == .ngram ? min(4, capped) : min(6, capped)
        var cand: [Int] = []
        if let pld = QWContextCopyDrafter.periodicDrafts(
            history: session.tokenHistory, extraToken: ctx.extraPending,
            maxCount: capped, minRepeats: 3
        ), pld.count >= minPldTokens {
            cand = Array(pld.prefix(capped))
        } else if let match = QWContextCopyDrafter.drafts(
            history: session.tokenHistory, extraToken: ctx.extraPending,
            maxCount: capped, minMatchLen: 6
        ), !match.tokens.isEmpty {
            var extended = Array(match.tokens.prefix(capped))
            if extended.count < capped {
                session.tokenRecycler.appendDrafts(
                    into: &extended, maxCount: capped,
                    previousToken: ctx.previousToken, pendingToken: pending,
                    maxUnigramFallbacks: 0
                )
            }
            if extended.count >= minPldTokens {
                cand = extended
            }
        }
        return cand
    }

    /// Appends up to `min(requestedDrafts, 15) - drafts.count` context PLD/recycler suffix tokens after a full DSpark block.
    static func extendDSparkDraftsWithHybridSuffix(
        drafts: [Int],
        tokenHistory: [Int],
        extraPending: Int?,
        previousBeforePending: Int?,
        pending: Int,
        tokenRecycler: QWTokenRecycler,
        requestedDrafts: Int,
        maxNeuralDrafts: Int,
        strategy: QWDraftStrategy,
        isDeterministic: Bool,
        maxTotalDrafts: Int = 15
    ) -> [Int] {
        let targetCap = min(min(requestedDrafts, 15), maxTotalDrafts)
        guard isDeterministic,
              strategy != .mtp,
              targetCap > drafts.count,
              drafts.count == maxNeuralDrafts,
              !drafts.isEmpty else {
            return drafts
        }
        let remaining = targetCap - drafts.count
        var out = drafts
        let tailLen = min(tokenHistory.count, 64)
        var periodicTail: [Int] = []
        periodicTail.reserveCapacity(tailLen + (extraPending != nil ? 1 : 0) + drafts.count)
        periodicTail.append(contentsOf: tokenHistory.suffix(tailLen))
        if let extraPending { periodicTail.append(extraPending) }
        periodicTail.append(contentsOf: drafts)
        if let pld = QWContextCopyDrafter.periodicDrafts(
            history: periodicTail, extraToken: nil, maxCount: remaining, minRepeats: 3
        ), !pld.isEmpty {
            out.append(contentsOf: pld.prefix(remaining))
            return out
        }
        if let match = QWContextCopyDrafter.drafts(
            history: tokenHistory, extraToken: extraPending, trailingTokens: drafts,
            maxCount: remaining, minMatchLen: 5
        ), !match.tokens.isEmpty {
            out.append(contentsOf: match.tokens.prefix(remaining))
        }
        if out.count < targetCap {
            _ = tokenRecycler.appendDrafts(
                into: &out, maxCount: targetCap,
                previousToken: previousBeforePending, pendingToken: pending,
                maxUnigramFallbacks: 0
            )
        }
        return out
    }

    func prepareAsyncDSparkOrPLDRound(
        session: QWEngineSession,
        dsparkDrafter: QWDSparkDrafter,
        dsSlot: QWDSparkDrafter.SessionSlot,
        pending: Int,
        count: Int,
        table: QWMTPEmbeddingTable
    ) throws -> AsyncNeuralDraftPlan {
        let resolvedStrategy = (try? QWDraftStrategy.resolve(mtpDraftStrategy)) ?? .hybrid
        if let cand = proposeCanonicalOrFilteredDrafts(
            constraint: activeConstraint?.constraint,
            maxCount: count,
            fallbackCandidates: {
                guard resolvedStrategy != .mtp else { return nil }
                let raw = selectDSparkPLDCandidates(
                    session: session, pending: pending, count: count, resolvedStrategy: resolvedStrategy
                )
                return raw.isEmpty ? nil : raw
            }
        ), !cand.isEmpty {
            session.dsparkRoundStartNs = nil
            return .pld(cand)
        }
        let activeCount = min(count, dsparkDrafter.config.maxDrafts)
        let ids = table.tokens.contents().assumingMemoryBound(to: Int32.self)
        ids[0] = Int32(pending)
        let proposed: (cb: any MTLCommandBuffer, maxDrafts: Int)
        do {
            session.dsparkRoundStartNs = DispatchTime.now().uptimeNanoseconds
            proposed = try dsparkDrafter.enqueueGreedyPropose(
                driver: self,
                slot: dsSlot,
                pos0: session.dsparkCacheOffset,
                anchorToken: pending,
                requestedDrafts: activeCount,
                journalBuffer: table.tokens,
                copyEmbedRow: { tok, dst in try self.copyEmbedRow(tok, to: dst) }
            )
        } catch {
            admission.commandBufferErrors += 1
            throw error
        }
        return .asyncNeural(draftCBs: [proposed.cb], draftedPairs: proposed.maxDrafts)
    }

    func proposeDSparkOrPLDRound(
        session: QWEngineSession,
        dsparkDrafter: QWDSparkDrafter,
        dsSlot: QWDSparkDrafter.SessionSlot,
        pending: Int,
        count: Int,
        owner: QWStochasticSession?,
        draftCompleted: ((any MTLCommandBuffer) throws -> Void)? = nil
    ) throws -> MTPRoundProposal {
        let resolvedStrategy = (try? QWDraftStrategy.resolve(mtpDraftStrategy)) ?? .hybrid
        let canUseCanonical = (owner == nil || owner?.config.isGreedy == true) && !mtpStageProfiling
        let canUsePLD = canUseCanonical && resolvedStrategy != .mtp
        if canUseCanonical,
           let cand = proposeCanonicalOrFilteredDrafts(
               constraint: activeConstraint?.constraint,
               maxCount: count,
               fallbackCandidates: {
                   canUsePLD ? selectDSparkPLDCandidates(
                       session: session, pending: pending, count: count, resolvedStrategy: resolvedStrategy
                   ) : nil
               }),
           !cand.isEmpty {
            lastRoundUsedPLD = true
            mtpDraftProposals += cand.count
            session.dsparkRoundStartNs = nil
            return MTPRoundProposal(drafts: cand, proposals: [], draftedPairs: cand.count)
        }
        let activeCount = min(count, dsparkDrafter.config.maxDrafts)
        let startNs = DispatchTime.now().uptimeNanoseconds
        session.dsparkRoundStartNs = startNs

        if let owner {
            let res: (drafts: [Int], proposals: [QWMTPGumbelProposal])
            do {
                res = try dsparkDrafter.proposeStochastic(
                    driver: self,
                    session: session,
                    slot: dsSlot,
                    pos0: session.dsparkCacheOffset,
                    anchorToken: pending,
                    requestedDrafts: activeCount,
                    owner: owner,
                    copyEmbedRow: { tok, dst in try self.copyEmbedRow(tok, to: dst) },
                    completed: draftCompleted
                )
            } catch {
                admission.commandBufferErrors += 1
                throw error
            }
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - startNs) / 1e9
            dsparkDrafter.confidenceScheduler.recordDraft(seconds: elapsed)
            mtpDraftProposals += res.drafts.count
            return MTPRoundProposal(drafts: res.drafts, proposals: res.proposals, draftedPairs: res.drafts.count)
        }

        lastRoundUsedPLD = false
        let drafts: [Int]
        do {
            let enq = try dsparkDrafter.enqueueGreedyPropose(
                driver: self,
                slot: dsSlot,
                pos0: session.dsparkCacheOffset,
                anchorToken: pending,
                requestedDrafts: activeCount,
                copyEmbedRow: { tok, dst in try self.copyEmbedRow(tok, to: dst) }
            )
            let neuralDrafts = try dsparkDrafter.readGreedyDraftsAndSelectCount(
                completedCB: enq.cb, maxDrafts: enq.maxDrafts, requestedDrafts: activeCount
            )
            try draftCompleted?(enq.cb)
            let ctx = pendingHistoryContext(session: session, pending: pending)
            let safeCap = min(15, max(dsparkDrafter.config.maxDrafts, dsparkDrafter.maxCaptureRows - 1))
            drafts = Self.extendDSparkDraftsWithHybridSuffix(
                drafts: neuralDrafts,
                tokenHistory: session.tokenHistory,
                extraPending: ctx.extraPending,
                previousBeforePending: ctx.previousToken,
                pending: pending,
                tokenRecycler: session.tokenRecycler,
                requestedDrafts: count,
                maxNeuralDrafts: dsparkDrafter.config.maxDrafts,
                strategy: resolvedStrategy,
                isDeterministic: (activeConstraint?.constraint.isUnconstrainedNow ?? true),
                maxTotalDrafts: safeCap
            )
        } catch {
            admission.commandBufferErrors += 1
            throw error
        }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - startNs) / 1e9
        dsparkDrafter.confidenceScheduler.recordDraft(seconds: elapsed)
        mtpDraftProposals += drafts.count
        return MTPRoundProposal(drafts: drafts, proposals: [], draftedPairs: drafts.count)
    }

    func recordDSparkRoundVerifyLatency(_ session: QWEngineSession, verifyRows: Int) {
        guard let ds = dsparkDrafter, let startNs = session.dsparkRoundStartNs else { return }
        session.dsparkRoundStartNs = nil
        guard !lastRoundUsedPLD else { return }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - startNs) / 1e9
        ds.confidenceScheduler.recordVerify(verifyRows: verifyRows, seconds: elapsed)
    }

    static func shouldDrainPendingCommitInSyncBranch(
        hasOwner: Bool,
        adaptiveDepth: Bool,
        hasDeviceEmbedding: Bool,
        hasColdEvents: Bool,
        hasDSpark: Bool
    ) -> Bool {
        hasOwner || adaptiveDepth || !hasDeviceEmbedding || hasColdEvents || hasDSpark
    }

    func verifyAsyncSlotDraftPlan(
        _ draftPlan: AsyncNeuralDraftPlan,
        session: QWEngineSession,
        pending: Int,
        table: QWMTPEmbeddingTable,
        priorCommitCB: (any MTLCommandBuffer)?
    ) throws -> (window: QWVerifyWindow, drafts: [Int], draftedPairs: Int) {
        switch draftPlan {
        case .pld(let chosen):
            let window = try verifyTokensCore(
                session, explicitTokens: [pending] + chosen, rowCount: 1 + chosen.count,
                pendingToken: nil, useDeviceEmbed: table, priorCommitCB: priorCommitCB,
                inFlightDraftCBs: []).window
            lastRoundUsedPLD = true
            mtpDraftProposals += chosen.count
            return (window, chosen, chosen.count)
        case .asyncNeural(let draftCBs, let neuralCount):
            let (verifiedWindow, resolvedTokens) = try verifyTokensCore(
                session, explicitTokens: nil, rowCount: 1 + neuralCount,
                pendingToken: pending, useDeviceEmbed: table,
                priorCommitCB: priorCommitCB, inFlightDraftCBs: draftCBs)
            let drafts = Array(resolvedTokens.dropFirst())
            lastRoundUsedPLD = false
            mtpDraftProposals += drafts.count
            return (verifiedWindow, drafts, neuralCount)
        }
    }
}
