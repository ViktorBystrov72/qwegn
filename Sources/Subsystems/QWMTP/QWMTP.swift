// QWMTP.swift — Gumbel-max speculative-decode coupling protocol (and relaxed
// Gumbel coupling per arXiv:2610.26693).
import Foundation
import QWHead

public enum QWMTP {
    public enum ProtocolError: Error, CustomStringConvertible {
        case invalid(String)
        public var description: String {
            switch self { case .invalid(let message): return "QWMTP protocol: \(message)" }
        }
    }

    public struct GumbelRoundResult: Sendable, Equatable {
        public let accepted: Int
        public let finalToken: Int
        public let draftTokens: [Int]
    }

    /// Couples already-transformed sampler distributions. Using survivor
    /// probabilities preserves the configured temperature, top-k and top-p
    /// semantics while sharing token-indexed Philox Gumbel noise.
    public static func checkedGumbelCouplingRound(draftDistributions: [QWSamplingDistribution],
                                                  targetDistributions: [QWSamplingDistribution],
                                                  seed: UInt64,
                                                  offset: UInt64 = 0,
                                                  maxDrafts: Int = 7) throws -> GumbelRoundResult {
        let n = draftDistributions.count
        guard maxDrafts >= 1, (1...maxDrafts).contains(n), targetDistributions.count == n + 1,
              let vocabulary = targetDistributions.first?.vocabulary,
              vocabulary > 0,
              (draftDistributions + targetDistributions).allSatisfy({
                  $0.vocabulary == vocabulary && !$0.survivorIds.isEmpty
                      && $0.survivorIds.count == $0.survivorProbs.count
                      && zip($0.survivorIds, $0.survivorProbs).allSatisfy {
                          (0..<vocabulary).contains($0.0) && $0.1.isFinite && $0.1 > 0
                      }
              }) else {
            throw ProtocolError.invalid("Gumbel sampler distributions have invalid shapes or probabilities")
        }
        let width = UInt64(vocabulary)
        let (counterCount, countOverflow) = UInt64(n + 1).multipliedReportingOverflow(by: width)
        let (_, endOverflow) = offset.addingReportingOverflow(counterCount)
        guard !countOverflow, !endOverflow else {
            throw ProtocolError.invalid("Gumbel Philox counter overflow")
        }
        var winnerPairs: [[Int]] = []
        winnerPairs.reserveCapacity(n)
        for row in 0..<n {
            let (rowOffset, rowOverflow) = UInt64(row).multipliedReportingOverflow(by: width)
            let (base, baseOverflow) = offset.addingReportingOverflow(rowOffset)
            guard !rowOverflow, !baseOverflow else {
                throw ProtocolError.invalid("Gumbel Philox counter overflow")
            }
            winnerPairs.append([
                try gumbelArgmax(draftDistributions[row], seed: seed, base: base),
                try gumbelArgmax(targetDistributions[row], seed: seed, base: base)
            ])
        }
        let bonusOffset = UInt64(n) * width
        return try checkedGumbelCouplingRound(winnerPairs: winnerPairs,
            bonusTarget: gumbelArgmax(targetDistributions[n], seed: seed, base: offset + bonusOffset),
            vocabulary: vocabulary, maxDrafts: maxDrafts)
    }

    public static func checkedGumbelWinner(distribution: QWSamplingDistribution,
                                           seed: UInt64, offset: UInt64) throws -> Int {
        guard distribution.vocabulary > 0, !distribution.survivorIds.isEmpty,
              distribution.survivorIds.count == distribution.survivorProbs.count,
              zip(distribution.survivorIds, distribution.survivorProbs).allSatisfy({
                  (0..<distribution.vocabulary).contains($0.0) && $0.1.isFinite && $0.1 > 0
              }),
              offset <= UInt64.max - UInt64(distribution.vocabulary - 1) else {
            throw ProtocolError.invalid("Gumbel sampler distribution or counter is invalid")
        }
        return try gumbelArgmax(distribution, seed: seed, base: offset)
    }

    /// Resolves the integer winner pairs emitted by the GPU comparison kernel.
    public static func checkedGumbelCouplingRound(winnerPairs: [[Int]],
                                                  bonusTarget: Int,
                                                  vocabulary: Int,
                                                  maxDrafts: Int = 7) throws -> GumbelRoundResult {
        guard maxDrafts >= 1, (1...maxDrafts).contains(winnerPairs.count), vocabulary > 0,
              (0..<vocabulary).contains(bonusTarget),
              winnerPairs.allSatisfy({
                  $0.count == 2 && (0..<vocabulary).contains($0[0]) && (0..<vocabulary).contains($0[1])
              }) else {
            throw ProtocolError.invalid("Gumbel winner-pair shape or token ID")
        }
        var accepted = 0
        for (row, pair) in winnerPairs.enumerated() {
            if pair[0] != pair[1] {
                return GumbelRoundResult(accepted: row, finalToken: pair[1],
                                         draftTokens: winnerPairs.prefix(row + 1).map { $0[0] })
            }
            accepted += 1
        }
        return GumbelRoundResult(accepted: accepted, finalToken: bonusTarget,
                                 draftTokens: winnerPairs.map { $0[0] })
    }

    /// Relaxed Gumbel coupling round (arXiv:2610.26693): accepts draft winner `d_i` when
    /// `d_i == t_i` or `log p_target(d_i) >= log p_target(t_i) - margin`.
    public static func checkedRelaxedGumbelCouplingRound(draftDistributions: [QWSamplingDistribution],
                                                         targetDistributions: [QWSamplingDistribution],
                                                         seed: UInt64,
                                                         offset: UInt64 = 0,
                                                         margin: Double = 0,
                                                         maxDrafts: Int = 15) throws -> GumbelRoundResult {
        let n = draftDistributions.count
        guard maxDrafts >= 1, (1...maxDrafts).contains(n), targetDistributions.count == n + 1,
              margin.isFinite, margin >= 0,
              let vocabulary = targetDistributions.first?.vocabulary,
              vocabulary > 0,
              (draftDistributions + targetDistributions).allSatisfy({
                  $0.vocabulary == vocabulary && !$0.survivorIds.isEmpty
                      && $0.survivorIds.count == $0.survivorProbs.count
                      && zip($0.survivorIds, $0.survivorProbs).allSatisfy {
                          (0..<vocabulary).contains($0.0) && $0.1.isFinite && $0.1 > 0
                      }
              }) else {
            throw ProtocolError.invalid("relaxed Gumbel sampler distributions or margin invalid")
        }
        let width = UInt64(vocabulary)
        let (counterCount, countOverflow) = UInt64(n + 1).multipliedReportingOverflow(by: width)
        let (_, endOverflow) = offset.addingReportingOverflow(counterCount)
        guard !countOverflow, !endOverflow else {
            throw ProtocolError.invalid("Gumbel Philox counter overflow")
        }
        var winnerPairs: [[Int]] = []
        winnerPairs.reserveCapacity(n)
        for row in 0..<n {
            let (rowOffset, rowOverflow) = UInt64(row).multipliedReportingOverflow(by: width)
            let (base, baseOverflow) = offset.addingReportingOverflow(rowOffset)
            guard !rowOverflow, !baseOverflow else {
                throw ProtocolError.invalid("Gumbel Philox counter overflow")
            }
            let draftWinner = try gumbelArgmax(draftDistributions[row], seed: seed, base: base)
            var targetWinner = try gumbelArgmax(targetDistributions[row], seed: seed, base: base)
            if draftWinner != targetWinner, margin > 0 {
                let pDraft = probability(targetDistributions[row], token: draftWinner)
                let pTarget = probability(targetDistributions[row], token: targetWinner)
                if pDraft > 0, pTarget > 0, log(pDraft) >= log(pTarget) - margin {
                    targetWinner = draftWinner
                }
            }
            winnerPairs.append([draftWinner, targetWinner])
        }
        let bonusOffset = UInt64(n) * width
        return try checkedGumbelCouplingRound(
            winnerPairs: winnerPairs,
            bonusTarget: gumbelArgmax(targetDistributions[n], seed: seed, base: offset + bonusOffset),
            vocabulary: vocabulary,
            maxDrafts: maxDrafts
        )
    }

    private static func gumbelArgmax(_ distribution: QWSamplingDistribution,
                                     seed: UInt64, base: UInt64) throws -> Int {
        var bestToken: Int?
        var bestScore = -Double.infinity
        for (token, probability) in zip(distribution.survivorIds, distribution.survivorProbs) {
            let (counter, overflow) = base.addingReportingOverflow(UInt64(token))
            guard !overflow else { throw ProtocolError.invalid("Gumbel Philox counter overflow") }
            let uniform = max(Double(QWPhilox.u(counter, seed: seed, version: .upper24)),
                              1.0 / 16_777_216.0)
            let score = log(probability) - log(-log(uniform))
            if score > bestScore || (score == bestScore && bestToken.map { token < $0 } != false) {
                bestScore = score
                bestToken = token
            }
        }
        guard let bestToken else {
            throw ProtocolError.invalid("Gumbel distribution has no selectable token")
        }
        return bestToken
    }

    private static func probability(_ distribution: QWSamplingDistribution, token: Int) -> Double {
        let ids = distribution.survivorIds
        var low = 0, high = ids.count
        while low < high {
            let mid = low + (high - low) / 2
            if ids[mid] < token { low = mid + 1 } else { high = mid }
        }
        return low < ids.count && ids[low] == token ? distribution.survivorProbs[low] : 0
    }
}
