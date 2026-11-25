// QWConstrainedSpeculationTests.swift — Tests for masked constrained distributions and variational gates.

import XCTest
@testable import QWHead

final class QWConstrainedSpeculationTests: XCTestCase {
    func testConstrainedDistributionGreedyAndSingleton() throws {
        let greedy = QWSampler(config: QWSamplerConfig(temperature: 0, topK: 1, topP: 1, repetitionPenalty: 2.0))
        let logits: [Float] = [100.0, 6.0, 5.0, 4.0]
        let only = [3]
        let singleton = try logits.withUnsafeBufferPointer { lBuf in
            try only.withUnsafeBufferPointer { aBuf in
                try greedy.constrainedDistribution(logits: lBuf, allowed: aBuf, repetitionHistory: [1])
            }
        }
        XCTAssertEqual(singleton.survivorIds, [3])
        XCTAssertEqual(singleton.survivorProbs, [1.0])

        let multi = [1, 2, 3]
        let dist = try logits.withUnsafeBufferPointer { lBuf in
            try multi.withUnsafeBufferPointer { aBuf in
                try greedy.constrainedDistribution(logits: lBuf, allowed: aBuf, repetitionHistory: [1])
            }
        }
        XCTAssertEqual(dist.survivorIds, [2], "token 1 penalized from 6.0 -> 3.0, so token 2 (5.0) wins")
        XCTAssertEqual(dist.survivorProbs, [1.0])
    }

    func testConstrainedDistributionStochasticSumsToOneWithinAllowed() throws {
        let sampler = QWSampler(config: QWSamplerConfig(temperature: 0.7, topK: 3, topP: 0.9, seed: 9, repetitionPenalty: 1.2))
        var logits = [Float](repeating: -2.0, count: 16)
        logits[0] = 80.0 // Forbidden high-logit token
        logits[2] = 3.0
        logits[5] = 2.6
        logits[9] = 2.1
        logits[12] = 0.5
        let allowed = [2, 5, 9, 12]

        let dist = try logits.withUnsafeBufferPointer { lBuf in
            try allowed.withUnsafeBufferPointer { aBuf in
                try sampler.constrainedDistribution(logits: lBuf, allowed: aBuf, repetitionHistory: [2])
            }
        }
        XCTAssertFalse(dist.survivorIds.isEmpty)
        for idx in dist.survivorIds {
            XCTAssertTrue(allowed.contains(idx), "forbidden token \(idx) leaked into constrained distribution")
        }
        let sum = dist.survivorProbs.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 1e-5)

        sampler.allowedScratch = allowed
        let fromScratch = try logits.withUnsafeBufferPointer { lBuf in
            try sampler.constrainedDistributionFromScratch(logits: lBuf, repetitionHistory: [2])
        }
        XCTAssertEqual(fromScratch.survivorIds, dist.survivorIds)
        for (lhs, rhs) in zip(fromScratch.survivorProbs, dist.survivorProbs) {
            XCTAssertEqual(lhs, rhs, accuracy: 1e-6)
        }
    }

    func testClearsConstrainedVariationalGateRejectsInvalidAndRenormalizesOverAllowed() {
        let sampler = QWSampler(config: .greedy)
        // Token 0 is forbidden with huge logit +50. Allowed tokens are [1, 2] with logits [2.0, 1.9].
        let logits: [Float] = [50.0, 2.0, 1.9, -5.0]
        let allowed = [1, 2]

        let rejectedForbidden = logits.withUnsafeBufferPointer { lBuf in
            allowed.withUnsafeBufferPointer { aBuf in
                sampler.clearsConstrainedVariationalGate(
                    logits: lBuf, allowed: aBuf, draftToken: 0, targetToken: 1,
                    margin: 60.0, minProbability: 0.01
                )
            }
        }
        XCTAssertFalse(rejectedForbidden, "forbidden draft token must never pass variational gate")

        // Over allowed [1, 2], targetToken=1 (2.0), draftToken=2 (1.9), gap=0.1 <= 0.25, P_allowed(2) ≈ 0.475 >= 0.40.
        let acceptedValid = logits.withUnsafeBufferPointer { lBuf in
            allowed.withUnsafeBufferPointer { aBuf in
                sampler.clearsConstrainedVariationalGate(
                    logits: lBuf, allowed: aBuf, draftToken: 2, targetToken: 1,
                    margin: 0.25, minProbability: 0.40
                )
            }
        }
        XCTAssertTrue(acceptedValid, "masked renormalization over allowed must not be depressed by forbidden logit +50")

        let rejectedMargin = logits.withUnsafeBufferPointer { lBuf in
            allowed.withUnsafeBufferPointer { aBuf in
                sampler.clearsConstrainedVariationalGate(
                    logits: lBuf, allowed: aBuf, draftToken: 2, targetToken: 1,
                    margin: 0.05, minProbability: 0.10
                )
            }
        }
        XCTAssertFalse(rejectedMargin)

        sampler.allowedScratch = allowed
        XCTAssertTrue(
            logits.withUnsafeBufferPointer { lBuf in
                sampler.clearsConstrainedVariationalGateFromScratch(
                    logits: lBuf, draftToken: 2, targetToken: 1, margin: 0.25, minProbability: 0.40
                )
            }
        )
        XCTAssertFalse(
            logits.withUnsafeBufferPointer { lBuf in
                sampler.clearsConstrainedVariationalGateFromScratch(
                    logits: lBuf, draftToken: 0, targetToken: 1, margin: 60.0, minProbability: 0.01
                )
            }
        )
    }
}
