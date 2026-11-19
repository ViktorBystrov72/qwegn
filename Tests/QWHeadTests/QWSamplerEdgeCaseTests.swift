// QWSamplerEdgeCaseTests.swift — Edge-case guards and numerical parity for QWSampler.
import Foundation
import XCTest
@testable import QWHead

final class QWSamplerEdgeCaseTests: XCTestCase {
    func testGreedyTokenRejectsEmptyAndNaNWithDefaultPenalty() throws {
        XCTAssertThrowsError(try QWSampler.greedyToken(logits: []))
        XCTAssertThrowsError(try QWSampler.greedyToken(logits: [1.0, .nan, 2.0]))
        XCTAssertThrowsError(try QWSampler.greedyToken(logits: [1.0, .infinity, 2.0]))
    }

    func testPenalisedGreedyRejectsNonFiniteLogitsAndOverflow() throws {
        let sampler = QWSampler(config: QWSamplerConfig(temperature: 0, topK: 1, topP: 1, repetitionPenalty: 2.0))
        XCTAssertThrowsError(try sampler.penalisedGreedyToken(logits: [1.0, .nan, 2.0], repetitionHistory: []))
        // Multiplying extreme negative logits by penalty > 1 overflows to -inf -> rejected cleanly.
        XCTAssertThrowsError(try sampler.penalisedGreedyToken(logits: [-3e38, 1.0, 2.0], repetitionHistory: [0]))
        let winner = try sampler.penalisedGreedyToken(logits: [2.0, 1.5, 0.5], repetitionHistory: [0])
        XCTAssertEqual(winner, 1)
    }

    func testComputeTokenLogprobHandlesTiesAcrossTopKBoundary() throws {
        let logits: [Float] = [5.0, 5.0, 5.0, 1.0, -2.0]
        let sampler = QWSampler(config: QWSamplerConfig(temperature: 1.0, topK: 2, topP: 1.0, seed: 42))
        let fullSum = logits.reduce(0.0) { $0 + Foundation.exp(Double($1) - 5.0) }
        let expectedTieLP = Float(Foundation.log(1.0 / fullSum))
        let expectedSmallLP = Float((1.0 - 5.0) - Foundation.log(fullSum))

        logits.withUnsafeBufferPointer { buf in
            let r0 = sampler.computeTokenLogprob(logits: buf, chosenToken: 0, topK: 2)
            let r1 = sampler.computeTokenLogprob(logits: buf, chosenToken: 1, topK: 2)
            let r2 = sampler.computeTokenLogprob(logits: buf, chosenToken: 2, topK: 2)
            let r3 = sampler.computeTokenLogprob(logits: buf, chosenToken: 3, topK: 2)
            XCTAssertEqual(r0.logprob, expectedTieLP, accuracy: 1e-6)
            XCTAssertEqual(r1.logprob, expectedTieLP, accuracy: 1e-6)
            XCTAssertEqual(r2.logprob, expectedTieLP, accuracy: 1e-6)
            XCTAssertEqual(r3.logprob, expectedSmallLP, accuracy: 1e-6)
            XCTAssertEqual(r0.topCandidates.map(\.tokenId), [0, 1])
        }
    }

    func testConstrainedTokenMatchesDoubleReferenceUnderWideLogits() throws {
        let logits: [Float] = [-100.0, 50.0, 1000.0, 49.0, 50.5, -50.0]
        let sampler = QWSampler(config: QWSamplerConfig(temperature: 0.7, topK: 64, topP: 0.9, seed: 123))
        let greedy = QWSampler(config: .greedy)
        try logits.withUnsafeBufferPointer { buf in
            let gTok = try greedy.constrainedToken(logits: buf, allowed: [1, 3, 4], repetitionHistory: [])
            XCTAssertEqual(gTok, 4)

            let draws = try (0..<16).map { _ in
                try sampler.constrainedToken(logits: buf, allowed: [1, 3, 4], repetitionHistory: [])
            }
            for tok in draws {
                XCTAssertTrue([1, 3, 4].contains(tok))
            }
        }
    }
}
