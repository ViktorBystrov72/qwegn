// QWSpeculativeTokenResolver.swift — Shared L0 contract for speculative draft acceptance, bonus resolution, and rollback counts.

import Foundation

public enum QWSpeculativeGreedyOrdering: Sendable, Equatable {
    case numericalDraft
    case shippingTarget
}

public enum QWSpeculativeResolverError: Error, Equatable, CustomStringConvertible {
    case invalidVocabulary(Int)
    case invalidDraftCount(Int)
    case invalidTargetCount(expected: Int, actual: Int)
    case invalidTokenID(Int, vocabulary: Int)
    case emptyLogits
    case nonFiniteLogit
    case invalidBudget(Int)

    public var description: String {
        switch self {
        case .invalidVocabulary(let v):
            return "speculative resolver requires positive vocabulary, got \(v)"
        case .invalidDraftCount(let c):
            return "speculative resolver draft count \(c) out of range 1...15"
        case .invalidTargetCount(let expected, let actual):
            return "speculative resolver expected \(expected) target tokens, got \(actual)"
        case .invalidTokenID(let id, let vocab):
            return "speculative token \(id) outside vocabulary 0..<\(vocab)"
        case .emptyLogits:
            return "greedy logits are empty"
        case .nonFiniteLogit:
            return "non-finite greedy logits"
        case .invalidBudget(let n):
            return "invalid speculative output budget \(n)"
        }
    }
}

/// Canonical resolved outcome of one speculative verification round.
public struct QWSpeculativeResolution: Sendable, Equatable {
    public let acceptedDrafts: Int
    public let rawArgmaxAccepted: Int
    public let tokens: [Int]
    public let rollbackCount: Int
    public let stopped: Bool
    public let stoppedToken: Int?

    public init(
        acceptedDrafts: Int,
        rawArgmaxAccepted: Int,
        tokens: [Int],
        rollbackCount: Int,
        stopped: Bool,
        stoppedToken: Int? = nil
    ) {
        self.acceptedDrafts = acceptedDrafts
        self.rawArgmaxAccepted = rawArgmaxAccepted
        self.tokens = tokens
        self.rollbackCount = rollbackCount
        self.stopped = stopped
        self.stoppedToken = stoppedToken
    }

    public var firstToken: Int { tokens[0] }
    public var trailingTokens: [Int] { tokens.count > 1 ? Array(tokens.dropFirst()) : [] }
    public var pendingToken: Int? { tokens.last }

    /// True when the MTP chain/seed must re-sync to the last committed token instead of draft-0.
    public func needsResync(targetArgmaxes: [Int]) -> Bool {
        guard acceptedDrafts < targetArgmaxes.count else { return true }
        return rawArgmaxAccepted != acceptedDrafts || pendingToken != targetArgmaxes[acceptedDrafts]
    }
}

/// Shared speculative verification resolver for both QWEngine (`QWMTPGreedySelector`) and `QWFlashNext`.
public enum QWSpeculativeTokenResolver {
    @discardableResult
    public static func validateToken(_ id: Int, vocabulary: Int) throws -> Int {
        guard vocabulary > 0 else { throw QWSpeculativeResolverError.invalidVocabulary(vocabulary) }
        guard (0..<vocabulary).contains(id) else {
            throw QWSpeculativeResolverError.invalidTokenID(id, vocabulary: vocabulary)
        }
        return id
    }

    public static func validateTokens(_ ids: [Int], vocabulary: Int) throws {
        guard vocabulary > 0 else { throw QWSpeculativeResolverError.invalidVocabulary(vocabulary) }
        for id in ids where !(0..<vocabulary).contains(id) {
            throw QWSpeculativeResolverError.invalidTokenID(id, vocabulary: vocabulary)
        }
    }

    /// Checked bit-key argmax matching `qwg_mtp_argmax_checked` / `qwg_mtp_argmax_target_checked`.
    public static func greedyToken(
        _ scores: UnsafeBufferPointer<Float>,
        ordering: QWSpeculativeGreedyOrdering = .numericalDraft
    ) throws -> Int {
        guard !scores.isEmpty else { throw QWSpeculativeResolverError.emptyLogits }
        var bestKey: UInt32 = 0
        var token = 0
        for (id, score) in scores.enumerated() {
            guard score.isFinite else { throw QWSpeculativeResolverError.nonFiniteLogit }
            let bits = (ordering == .numericalDraft && score == 0) ? UInt32(0) : score.bitPattern
            let key = (bits & 0x8000_0000 != 0) ? ~bits : (bits ^ 0x8000_0000)
            if key > bestKey {
                bestKey = key
                token = id
            }
        }
        return token
    }

    /// Variational fidelity floor: ensures `softmax(logits)[draft] >= minProbability`.
    public static func clearsProbabilityFloor(
        _ logits: UnsafeBufferPointer<Float>,
        draftScore: Float,
        maxScore: Float,
        minProbability: Float
    ) -> Bool {
        guard minProbability > 0 else { return true }
        guard minProbability < 1 else { return false }
        var sum: Double = 0
        for score in logits {
            let diff = score - maxScore
            if diff > -45.0 { sum += exp(Double(diff)) }
        }
        let logProb = Double(draftScore - maxScore) - log(sum)
        return logProb >= log(Double(minProbability))
    }

    public static func rawArgmaxAcceptedCount(drafts: [Int], targetArgmaxes: [Int]) -> Int {
        let limit = min(drafts.count, targetArgmaxes.count)
        var sel = 0
        while sel < limit && targetArgmaxes[sel] == drafts[sel] {
            sel += 1
        }
        return sel
    }

    public static func rollbackCount(draftCount: Int, acceptedDrafts: Int) -> Int {
        max(0, draftCount - max(0, acceptedDrafts))
    }

    public static func retainedPairRows(retainedRows: Int, draftCount: Int, acceptedDrafts: Int) throws -> Int {
        guard (1...15).contains(draftCount), acceptedDrafts >= 0, acceptedDrafts <= draftCount else {
            throw QWSpeculativeResolverError.invalidDraftCount(draftCount)
        }
        return min(retainedRows, draftCount)
    }

    /// Resolves greedy / relaxed-variational acceptance, bonus token, stop truncation, and rollback count.
    public static func resolveGreedy(
        drafts: [Int],
        targetArgmaxes: [Int],
        vocabulary: Int,
        maxNew: Int = Int.max,
        stopIDs: Set<Int> = [],
        relaxedMargin: Float = 0,
        relaxedMinProb: Float = 0,
        logitRow: ((Int) -> UnsafeBufferPointer<Float>)? = nil
    ) throws -> QWSpeculativeResolution {
        guard vocabulary > 0 else { throw QWSpeculativeResolverError.invalidVocabulary(vocabulary) }
        guard (1...15).contains(drafts.count) else {
            throw QWSpeculativeResolverError.invalidDraftCount(drafts.count)
        }
        guard targetArgmaxes.count == drafts.count + 1 else {
            throw QWSpeculativeResolverError.invalidTargetCount(
                expected: drafts.count + 1, actual: targetArgmaxes.count
            )
        }
        guard maxNew >= 0 else { throw QWSpeculativeResolverError.invalidBudget(maxNew) }
        try validateTokens(drafts, vocabulary: vocabulary)
        try validateTokens(targetArgmaxes, vocabulary: vocabulary)
        for s in stopIDs { _ = try validateToken(s, vocabulary: vocabulary) }

        let rawAccepted = rawArgmaxAcceptedCount(drafts: drafts, targetArgmaxes: targetArgmaxes)
        var acceptedTargets = targetArgmaxes
        if relaxedMargin > 0, let logitRow {
            for row in 0..<drafts.count {
                let draftId = drafts[row]
                let targetId = acceptedTargets[row]
                if draftId != targetId {
                    let rowBuf = logitRow(row)
                    guard rowBuf.count >= vocabulary else {
                        throw QWSpeculativeResolverError.emptyLogits
                    }
                    let draftScore = rowBuf[draftId]
                    let targetScore = rowBuf[targetId]
                    if draftScore.isFinite && targetScore.isFinite
                        && draftScore >= targetScore - relaxedMargin
                        && clearsProbabilityFloor(
                            rowBuf, draftScore: draftScore, maxScore: targetScore,
                            minProbability: relaxedMinProb
                        ) {
                        acceptedTargets[row] = draftId
                    } else {
                        break
                    }
                }
            }
        }

        var accepted = 0
        while accepted < drafts.count && drafts[accepted] == acceptedTargets[accepted] {
            accepted += 1
        }
        let candidates = Array(drafts.prefix(accepted)) + [acceptedTargets[accepted]]
        var emitted: [Int] = []
        var stoppedToken: Int?
        for token in candidates.prefix(maxNew) {
            if stopIDs.contains(token) {
                stoppedToken = token
                break
            }
            emitted.append(token)
        }
        return QWSpeculativeResolution(
            acceptedDrafts: accepted,
            rawArgmaxAccepted: rawAccepted,
            tokens: emitted,
            rollbackCount: rollbackCount(draftCount: drafts.count, acceptedDrafts: accepted),
            stopped: stoppedToken != nil,
            stoppedToken: stoppedToken
        )
    }

    /// Resolves draft acceptance, constrained boundary stops, bonus token, and rollback count.
    public static func resolveVerifiedRound(
        drafts: [Int],
        targetArgmaxes: [Int],
        vocabulary: Int,
        isGreedy: Bool,
        isUnconstrainedNow: () -> Bool,
        boundaryToken: Int?,
        canSpeculateConstrained: Bool = false,
        isRepetitionBlocked: (Int) -> Bool,
        acceptConstrained: (Int) throws -> Void,
        sampleRow: (_ row: Int, _ argmaxToken: Int) throws -> Int,
        onCommittedToken: (Int) -> Void
    ) throws -> QWSpeculativeResolution {
        let dCount = drafts.count
        guard vocabulary > 0 else { throw QWSpeculativeResolverError.invalidVocabulary(vocabulary) }
        guard dCount >= 1 else { throw QWSpeculativeResolverError.invalidDraftCount(dCount) }
        guard targetArgmaxes.count >= dCount + 1 else {
            throw QWSpeculativeResolverError.invalidTargetCount(
                expected: dCount + 1, actual: targetArgmaxes.count
            )
        }
        let rawAccepted = rawArgmaxAcceptedCount(drafts: drafts, targetArgmaxes: targetArgmaxes)
        var committed: [Int] = []
        committed.reserveCapacity(dCount + 1)
        var accepted = 0

        for r in 0..<dCount {
            let draftTok = drafts[r]
            let wasUnconstrained = isUnconstrainedNow()
            let sampled: Int
            if targetArgmaxes[r] == draftTok && isGreedy && !isRepetitionBlocked(draftTok) && wasUnconstrained {
                try acceptConstrained(draftTok)
                sampled = draftTok
            } else {
                sampled = try sampleRow(r, targetArgmaxes[r])
            }
            committed.append(sampled)
            let hitBoundary = !canSpeculateConstrained
                && ((boundaryToken != nil && sampled == boundaryToken) || !isUnconstrainedNow())
            let matchesTarget = canSpeculateConstrained
                ? (sampled == draftTok)
                : (sampled == draftTok && targetArgmaxes[r] == draftTok)
            if matchesTarget && !hitBoundary {
                accepted += 1
                onCommittedToken(draftTok)
            } else {
                onCommittedToken(sampled)
                break
            }
        }
        if accepted == dCount {
            let bonusTok = try sampleRow(dCount, targetArgmaxes[dCount])
            onCommittedToken(bonusTok)
            committed.append(bonusTok)
        }
        return QWSpeculativeResolution(
            acceptedDrafts: accepted,
            rawArgmaxAccepted: rawAccepted,
            tokens: committed,
            rollbackCount: rollbackCount(draftCount: dCount, acceptedDrafts: accepted),
            stopped: false,
            stoppedToken: nil
        )
    }
}
