import XCTest
import QWHead
@testable import QWMTP

final class QWMTPSparseProtocolTests: XCTestCase {
    private func distribution(_ logits: [Float], topK: Int = 3) throws -> QWSamplingDistribution {
        try QWSampler(config: QWSamplerConfig(temperature: 1, topK: topK, topP: 1,
                                             uniformVersion: .upper24)).distribution(logits: logits)
    }

    func testGumbelCouplingWinnerPairsValidation() throws {
        XCTAssertEqual(try QWMTP.checkedGumbelCouplingRound(
            winnerPairs: [[1, 0], [2, 2]], bonusTarget: 1, vocabulary: 3),
            QWMTP.GumbelRoundResult(accepted: 0, finalToken: 0, draftTokens: [1]))
        XCTAssertThrowsError(try QWMTP.checkedGumbelCouplingRound(
            winnerPairs: [[1]], bonusTarget: 0, vocabulary: 3))
    }

    func testGumbelCouplingUsesFilteredSamplerSupports() throws {
        let drafts = try [
            distribution([4, -3, 1, -4], topK: 3),
            distribution([-2, 3, -1, 1], topK: 2),
            distribution([0, -2, 4, 1], topK: 4),
            distribution([3, -1, 0, 2], topK: 3),
            distribution([-2, 2, 0, 1], topK: 4)
        ]
        let targets = try [
            distribution([3, 0, 1, -2], topK: 3),
            distribution([0, 3, 1, -2], topK: 3),
            distribution([1, -1, 3, 0], topK: 2),
            distribution([2, 0, 1, 3], topK: 4),
            distribution([0, 1, 2, 3], topK: 2),
            distribution([2, 0, 3, 1], topK: 4)
        ]
        let seed: UInt64 = 713, offset: UInt64 = 41
        let result = try QWMTP.checkedGumbelCouplingRound(draftDistributions: drafts,
            targetDistributions: targets, seed: seed, offset: offset)

        func winner(_ row: QWSamplingDistribution, rowIndex: Int) -> Int {
            let base = offset + UInt64(rowIndex * row.vocabulary)
            return zip(row.survivorIds, row.survivorProbs).max { lhs, rhs in
                func score(_ item: (Int, Double)) -> Double {
                    let u = max(Double(QWPhilox.u(base + UInt64(item.0), seed: seed, version: .upper24)),
                                1.0 / 16_777_216.0)
                    return log(item.1) - log(-log(u))
                }
                let left = score(lhs), right = score(rhs)
                return left == right ? lhs.0 > rhs.0 : left < right
            }!.0
        }
        var pairs: [[Int]] = []
        for row in drafts.indices {
            pairs.append([winner(drafts[row], rowIndex: row), winner(targets[row], rowIndex: row)])
        }
        let expected = try QWMTP.checkedGumbelCouplingRound(winnerPairs: pairs,
            bonusTarget: winner(targets[5], rowIndex: 5), vocabulary: 4)
        XCTAssertEqual(result, expected)
        XCTAssertThrowsError(try QWMTP.checkedGumbelCouplingRound(
            draftDistributions: [drafts[0]], targetDistributions: [targets[0], targets[1]],
            seed: seed, offset: .max))
        let otherVocabulary = try distribution([3, 1], topK: 2)
        XCTAssertThrowsError(try QWMTP.checkedGumbelCouplingRound(
            draftDistributions: [otherVocabulary], targetDistributions: [targets[0], targets[1]],
            seed: seed))
    }

    func testGumbelWinnerPreservesFilteredCategoricalMarginal() throws {
        let row = try distribution([0.7, 0.2, -0.4, -2], topK: 4)
        var counts = [Int](repeating: 0, count: row.vocabulary)
        let samples = 4096
        for seed in 0..<samples {
            counts[try QWMTP.checkedGumbelWinner(distribution: row, seed: UInt64(seed), offset: 19)] += 1
        }
        var chiSquared = 0.0
        for (token, probability) in zip(row.survivorIds, row.survivorProbs) {
            let expected = Double(samples) * probability
            let difference = Double(counts[token]) - expected
            chiSquared += difference * difference / expected
        }
        XCTAssertLessThanOrEqual(chiSquared, 16.2663, "Gumbel-max marginal should match transformed sampler")
        XCTAssertEqual(counts.reduce(0, +), samples)
    }
}
