// MTPVerification+Constrained.swift — Row-sequential constrained speculative verification
// and canonical structural draft helpers on QWEngineDriver.

import Foundation
import QWAlloc
import QWCore
import QWHead
import QWMTP

extension QWEngineDriver {
    func canSpeculateConstraint(_ constraint: (any QWTokenConstraint)?) -> Bool {
        guard let constraint else { return true }
        if constraint.isUnconstrainedNow { return true }
        return QWConstrainedSpeculationGate.isEnabled
            && constraint.allowsSpeculation
            && constraint.makeSpeculativeProbe() != nil
    }

    var canSpeculateActiveConstraint: Bool {
        canSpeculateConstraint(activeConstraint?.constraint)
    }

    func proposeCanonicalOrFilteredDrafts(
        constraint: (any QWTokenConstraint)?,
        maxCount: Int,
        fallbackCandidates: () -> [Int]?
    ) -> [Int]? {
        guard maxCount > 0 else { return nil }
        if let constraint, !constraint.isUnconstrainedNow {
            let canonical = constraint.proposeCanonicalDrafts(maxCount: maxCount)
            if !canonical.isEmpty { return canonical }
        }
        guard let raw = fallbackCandidates(), !raw.isEmpty else { return nil }
        guard let constraint, !constraint.isUnconstrainedNow else {
            return Array(raw.prefix(maxCount))
        }
        guard let probe = constraint.makeSpeculativeProbe() else { return nil }
        var filtered: [Int] = []
        filtered.reserveCapacity(min(maxCount, raw.count))
        for tok in raw.prefix(maxCount) {
            guard (try? probe.accept(tok)) != nil else { break }
            filtered.append(tok)
        }
        return filtered.isEmpty ? nil : filtered
    }

    func resolveConstrainedGreedyRound(
        window: QWVerifyWindow,
        session: QWEngineSession,
        pending: Int,
        drafts: [Int],
        maxNew: Int,
        stopIDs: Set<Int>,
        active: (constraint: any QWTokenConstraint, sampler: QWSampler),
        margin: Float,
        minProb: Float
    ) throws -> (decision: QWGreedyRoundDecision, targets: [Int]) {
        let count = drafts.count + 1
        guard let probe = active.constraint.makeSpeculativeProbe() else {
            throw QWEngineDriverError.state("constraint does not provide a speculative probe")
        }
        let p = window.logits.region.pointer.advanced(by: window.logits.offset).assumingMemoryBound(to: Float.self)
        var rawTargets: [Int]
        if window.targetTokens != nil {
            rawTargets = try window.completedTargets(activeRows: count)
        } else {
            rawTargets = []
            rawTargets.reserveCapacity(count)
            for row in 0..<count {
                rawTargets.append(try QWVerifyWindow.greedyToken(
                    UnsafeBufferPointer(start: p + row * geometry.vocab, count: geometry.vocab),
                    ordering: .shippingTarget))
            }
        }
        var acceptedTargets = rawTargets
        let ctx = pendingHistoryContext(session: session, pending: pending)
        let useRepPenalty = active.sampler.config.repetitionPenalty != 1.0
        var extraHistory: [Int] = []
        if useRepPenalty { extraHistory.reserveCapacity(count) }

        for row in 0..<count {
            let rowBuf = UnsafeBufferPointer(start: p + row * geometry.vocab, count: geometry.vocab)
            let repHistory: [Int]
            if useRepPenalty {
                extraHistory.removeAll(keepingCapacity: true)
                if let extra = ctx.extraPending { extraHistory.append(extra) }
                if row > 0 { extraHistory.append(contentsOf: drafts.prefix(row)) }
                repHistory = session.activeRepetitionHistory(extraTokens: extraHistory)
            } else {
                repHistory = []
            }
            let hasMask = try probe.writeAllowedTokens(into: &active.sampler.allowedScratch)
            let targetId: Int
            if !hasMask {
                targetId = repHistory.isEmpty
                    ? rawTargets[row]
                    : try active.sampler.penalisedGreedyToken(logits: rowBuf, repetitionHistory: repHistory)
            } else if repHistory.isEmpty
                        && QWTokenMaskScratch.sortedTokens(active.sampler.allowedScratch, contains: rawTargets[row]) {
                targetId = rawTargets[row]
            } else {
                targetId = try active.sampler.constrainedTokenFromScratch(
                    logits: rowBuf, repetitionHistory: repHistory)
            }
            var chosen = targetId
            if row < drafts.count {
                let draftId = drafts[row]
                if draftId != targetId && margin > 0 && (0..<geometry.vocab).contains(draftId) {
                    if !hasMask {
                        let draftScore = rowBuf[draftId], targetScore = rowBuf[targetId]
                        if repHistory.isEmpty && draftScore.isFinite && targetScore.isFinite
                            && draftScore >= targetScore - margin
                            && QWSpeculativeTokenResolver.clearsProbabilityFloor(
                                rowBuf, draftScore: draftScore, maxScore: targetScore, minProbability: minProb) {
                            chosen = draftId
                        }
                    } else if active.sampler.clearsConstrainedVariationalGateFromScratch(
                        logits: rowBuf, draftToken: draftId, targetToken: targetId,
                        repetitionHistory: repHistory, margin: margin, minProbability: minProb
                    ) {
                        chosen = draftId
                    }
                }
            }
            acceptedTargets[row] = chosen
            if row < drafts.count {
                if stopIDs.contains(chosen) || chosen != drafts[row] { break }
                try probe.accept(chosen)
            }
        }
        let decision = try QWGreedyRoundDecision.make(
            geometry: geometry, drafts: drafts, targetTokens: acceptedTargets,
            maxNew: maxNew, stopIDs: stopIDs)
        return (decision, acceptedTargets)
    }

    func resolveConstrainedStochasticRound(
        window: QWVerifyWindow,
        session: QWEngineSession,
        owner: QWStochasticSession,
        pending: Int,
        drafts: [Int],
        proposals: [QWMTPGumbelProposal],
        maxNew: Int,
        stopIDs: Set<Int>,
        active: (constraint: any QWTokenConstraint, sampler: QWSampler),
        margin: Float,
        drawStart: UInt64
    ) throws -> (decision: QWGreedyRoundDecision, witness: QWStochasticRoundWitness, evidence: QWStochasticRoundEvidence?) {
        guard let probe = active.constraint.makeSpeculativeProbe() else {
            throw QWEngineDriverError.state("constraint does not provide a speculative probe")
        }
        let count = drafts.count + 1
        let p = window.logits.region.pointer.advanced(by: window.logits.offset).assumingMemoryBound(to: Float.self)
        let ctx = pendingHistoryContext(session: session, pending: pending)
        var extraHistory: [Int] = []
        extraHistory.reserveCapacity(count)

        func rowDistribution(_ row: Int) throws -> QWSamplingDistribution {
            extraHistory.removeAll(keepingCapacity: true)
            if let extra = ctx.extraPending { extraHistory.append(extra) }
            if row > 0 { extraHistory.append(contentsOf: drafts.prefix(row)) }
            let history = session.activeRepetitionHistory(extraTokens: extraHistory)
            let hasMask = try probe.writeAllowedTokens(into: &active.sampler.allowedScratch)
            if hasMask {
                let rowBuf = UnsafeBufferPointer(start: p + row * geometry.vocab, count: geometry.vocab)
                return try active.sampler.constrainedDistributionFromScratch(
                    logits: rowBuf, repetitionHistory: history)
            }
            return try owner.distribution(logits: window.logitRow(row), repetitionHistory: history)
        }

        if proposals.isEmpty {
            var targetTokens = [Int](repeating: 0, count: count)
            for row in 0..<count {
                let dist = try rowDistribution(row)
                let sampled = try owner.stream.sampleTarget(dist)
                targetTokens[row] = sampled
                if row < drafts.count {
                    if sampled != drafts[row] || stopIDs.contains(sampled) { break }
                    try probe.accept(sampled)
                }
            }
            let decision = try QWGreedyRoundDecision.make(
                geometry: geometry, drafts: drafts, targetTokens: targetTokens,
                maxNew: maxNew, stopIDs: stopIDs)
            let witness = QWStochasticRoundWitness(owner: owner, drawEnd: owner.stream.drawsConsumed)
            return (decision, witness, nil)
        }

        var targets: [QWSamplingDistribution] = []
        targets.reserveCapacity(count)
        var probeAlive = true
        for row in 0..<count {
            if probeAlive {
                let dist = try rowDistribution(row)
                targets.append(dist)
                if row < drafts.count, (try? probe.accept(drafts[row])) == nil {
                    probeAlive = false
                }
            } else if let fallback = targets.last {
                targets.append(fallback)
            }
        }
        let result = margin > 0
            ? try owner.stream.resolveRelaxedGumbelRound(
                proposals: proposals, targetDistributions: targets, margin: Double(margin))
            : try owner.stream.resolveGumbelRound(
                proposals: proposals, targetDistributions: targets)
        let decision = try QWGreedyRoundDecision.sampled(
            geometry: geometry, proposals: proposals, result: result,
            maxNew: maxNew, stopIDs: stopIDs, maxDrafts: 15)
        let witness = QWStochasticRoundWitness(owner: owner, drawEnd: owner.stream.drawsConsumed)
        let evidence = QWStochasticRoundEvidence(
            proposals: proposals, targets: targets, result: result,
            drawStart: drawStart, drawEnd: owner.stream.drawsConsumed)
        return (decision, witness, evidence)
    }
}
