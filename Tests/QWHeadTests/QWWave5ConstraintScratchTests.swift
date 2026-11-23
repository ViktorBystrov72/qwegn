// QWWave5ConstraintScratchTests.swift — Tests for zero-alloc QWTokenMaskScratch and QWSampler parity.
import XCTest
@testable import QWHead
import QWCore

final class QWWave5ConstraintScratchTests: XCTestCase {

    private final class MockConstraint: QWTokenConstraint, @unchecked Sendable {
        var nextAllowed: [Int]?
        var accepted: [Int] = []
        init(_ allowed: [Int]?) { self.nextAllowed = allowed }
        func allowedTokens() throws -> [Int]? { nextAllowed }
        func accept(_ token: Int) throws { accepted.append(token) }
    }

    func testTokenMaskScratchBinarySearchSparseClearAndValidation() throws {
        XCTAssertFalse(QWTokenMaskScratch.sortedTokens([], contains: 0))
        XCTAssertTrue(QWTokenMaskScratch.sortedTokens([5], contains: 5))
        XCTAssertFalse(QWTokenMaskScratch.sortedTokens([5], contains: 4))
        let sorted = [2, 9, 63, 64, 65, 191, 255]
        for id in sorted { XCTAssertTrue(QWTokenMaskScratch.sortedTokens(sorted, contains: id)) }
        for miss in [0, 1, 3, 62, 66, 256] {
            XCTAssertFalse(QWTokenMaskScratch.sortedTokens(sorted, contains: miss))
        }

        let scratch = QWTokenMaskScratch(vocabularyCapacity: 128)
        try scratch.markTokens(sorted, vocabularySize: 260)
        for id in sorted { XCTAssertTrue(scratch.contains(id)) }
        XCTAssertFalse(scratch.contains(66))
        XCTAssertFalse(scratch.contains(-1))
        XCTAssertFalse(scratch.contains(260))

        // Out-of-bounds token must throw and leave a clean cleared bitset.
        XCTAssertThrowsError(try scratch.markTokens([5, 260], vocabularySize: 260))
        XCTAssertFalse(scratch.contains(5))

        // Mark arbitrary sequence and clear sparsely.
        try scratch.markTokens([1, 130, 250], vocabularySize: 260)
        XCTAssertTrue(scratch.contains(130))
        scratch.clearBitset()
        XCTAssertFalse(scratch.contains(130))
    }

    func testPenalisedGreedyZeroAllocParityAcrossLargeVocabAndTies() throws {
        var logits = [Float](repeating: -3.5, count: 259)
        logits[0] = -0.0
        logits[63] = 4.0
        logits[64] = 4.0
        logits[129] = 4.5
        logits[258] = 4.5

        let configs = [Float(1.0), Float(1.35), Float(2.0)]
        let histories: [[Int]] = [[], [129], [0, 63, 64, 129], [64, 129, 258]]
        for penalty in configs {
            let sampler = QWSampler(config: QWSamplerConfig(temperature: 0, topK: 1, topP: 1, repetitionPenalty: penalty))
            for history in histories {
                let ref = try QWSampler.greedyToken(logits: logits, repetitionHistory: history, repetitionPenalty: penalty)
                let setTok = try logits.withUnsafeBufferPointer {
                    try sampler.penalisedGreedyToken(logits: $0, repetitionHistory: Set(history))
                }
                let arrTok = try logits.withUnsafeBufferPointer {
                    try sampler.penalisedGreedyToken(logits: $0, repetitionHistory: history)
                }
                XCTAssertEqual(setTok, ref)
                XCTAssertEqual(arrTok, ref)
            }
        }

        // Negative logits get multiplied by penalty; lowest-id ties preserved.
        let tieSampler = QWSampler(config: QWSamplerConfig(temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1.5))
        let negTieLogits: [Float] = [-1.0, -2.0, -1.0, -1.5]
        let tieBest = try tieSampler.penalisedGreedyToken(logits: negTieLogits, repetitionHistory: [0, 2])
        XCTAssertEqual(tieBest, 0)
        let negLogits: [Float] = [-1.2, -2.0, -1.2, -1.5]
        let best = try tieSampler.penalisedGreedyToken(logits: negLogits, repetitionHistory: [0, 2])
        XCTAssertEqual(best, 3)

        // Invalid range throws.
        XCTAssertThrowsError(try tieSampler.penalisedGreedyToken(logits: negLogits, repetitionHistory: [4]))
    }

    func testConstrainedTokenFromScratchMatchesReferenceAcrossConfigs() throws {
        var logits = [Float](repeating: -2.0, count: 160)
        for i in 0..<logits.count {
            logits[i] = Float(sin(Double(i) * 0.37) * 3.0)
        }
        let constraint = MockConstraint([1, 7, 33, 64, 65, 120, 159])
        let versions: [QWPhilox.UniformVersion] = [.upper24, .legacyRounded32]

        for version in versions {
            let cfg = QWSamplerConfig(
                temperature: 0.75, topK: 4, topP: 0.9, seed: 1337, uniformVersion: version, repetitionPenalty: 1.2
            )
            let refSampler = QWSampler(config: cfg)
            let scratchSampler = QWSampler(config: cfg)
            let supportSets: [[Int]] = [
                [1, 7, 33, 64, 65, 120, 159],
                [7, 65],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 64, 128, 159],
                [64]
            ]
            for (step, sup) in supportSets.enumerated() {
                constraint.nextAllowed = sup
                XCTAssertTrue(try constraint.writeAllowedTokens(into: &scratchSampler.allowedScratch))
                let rep = step % 2 == 0 ? [7, 65] : [64]
                let refTok = try logits.withUnsafeBufferPointer {
                    try refSampler.constrainedToken(logits: $0, allowed: sup, repetitionHistory: Set(rep))
                }
                let scratchTok = try logits.withUnsafeBufferPointer {
                    try scratchSampler.constrainedTokenFromScratch(logits: $0, repetitionHistory: rep)
                }
                XCTAssertEqual(scratchTok, refTok)
                XCTAssertEqual(scratchSampler.drawsConsumed, refSampler.drawsConsumed)
            }
        }

        // Empty support and out-of-order support must throw cleanly.
        let emptySampler = QWSampler(config: QWSamplerConfig(temperature: 0, topK: 1, topP: 1))
        XCTAssertThrowsError(try logits.withUnsafeBufferPointer {
            try emptySampler.constrainedToken(logits: $0, allowed: [])
        })
        XCTAssertThrowsError(try logits.withUnsafeBufferPointer {
            try emptySampler.constrainedToken(logits: $0, allowed: [5, 5])
        })
    }

    func testLoopRepetitionTokensIntoScratchMatchesAllocatingReference() {
        let cycle40 = Array(1000..<1040)
        let ngram8 = [501, 502, 503, 504, 505, 506, 507, 508]
        let twoCopyNonConsec = ngram8 + [901] + Array(600..<620) + ngram8
        let threeCopyNonConsec = ngram8 + [901] + Array(600..<612) + ngram8 + [902] + Array(700..<712) + ngram8
        let windows: [[Int]] = [
            [],
            [1, 2, 3],
            [9, 9, 9, 9],
            [1, 2, 1, 2, 1, 2],
            [10, 20, 30, 10, 20, 30, 10, 20, 30],
            [1, 2, 3, 4, 5, 1, 6, 1, 7, 1, 8, 1, 9, 1, 10, 1],
            Array(0..<32) + [7, 8, 7, 8, 7, 8],
            cycle40 + cycle40,
            twoCopyNonConsec,
            threeCopyNonConsec,
        ]
        var scratch: [Int] = [999, 888, 777]
        for w in windows {
            let expected = QWSampler.loopRepetitionTokens(in: w)
            QWSampler.loopRepetitionTokens(in: w, into: &scratch)
            XCTAssertEqual(scratch, expected)

            QWSampler.loopRepetitionTokens(in: w[...], into: &scratch)
            XCTAssertEqual(scratch, expected)
        }
        XCTAssertEqual(QWSampler.loopRepetitionTokens(in: cycle40 + cycle40), [1000])
        XCTAssertEqual(QWSampler.loopRepetitionTokens(in: twoCopyNonConsec), [])
        XCTAssertEqual(QWSampler.loopRepetitionTokens(in: threeCopyNonConsec), [901, 902])
    }
}
