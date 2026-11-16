import XCTest
import QWHead
@testable import QWMTP

final class QWMTPRandomStreamTests: XCTestCase {
    func testGumbelRoundSharesRowCountersAndBurnsCompleteWindow() throws {
        let rows = try [
            QWSampler(config: QWSamplerConfig(temperature: 1, topK: 3, topP: 1,
                                               uniformVersion: .upper24)).distribution(logits: [0, 1, 2]),
            QWSampler(config: QWSamplerConfig(temperature: 1, topK: 2, topP: 1,
                                               uniformVersion: .upper24)).distribution(logits: [2, 0, 1]),
            QWSampler(config: QWSamplerConfig(temperature: 1, topK: 3, topP: 1,
                                               uniformVersion: .upper24)).distribution(logits: [1, 2, 0])
        ]
        let stream = QWMTPRandomStream(seed: 137, drawIndex: 11)
        try stream.beginGumbelRound(drafts: 2, vocabulary: 3)
        let proposals = try [stream.sampleGumbelProposal(rows[0]), stream.sampleGumbelProposal(rows[1])]
        XCTAssertEqual(proposals.map(\.token), [
            try QWMTP.checkedGumbelWinner(distribution: rows[0], seed: 137, offset: 11),
            try QWMTP.checkedGumbelWinner(distribution: rows[1], seed: 137, offset: 14)
        ])
        XCTAssertEqual(stream.drawsConsumed, 17)
        let targets = [rows[2], rows[0], rows[1]]
        let expected = try QWMTP.checkedGumbelCouplingRound(draftDistributions: rows.prefix(2).map { $0 },
            targetDistributions: targets, seed: 137, offset: 11)
        let result = try stream.resolveGumbelRound(proposals: proposals, targetDistributions: targets)
        XCTAssertEqual(result, expected)
        XCTAssertEqual(stream.drawsConsumed, 20)
        XCTAssertFalse(stream.isRoundActive)
        XCTAssertThrowsError(try stream.resolveGumbelRound(proposals: proposals, targetDistributions: targets))

        try stream.beginGumbelRound(drafts: 7, vocabulary: 3)
        let partial = try stream.sampleGumbelProposal(rows[0])
        stream.cancelActiveRound()
        XCTAssertEqual(stream.drawsConsumed, 23, "successfully sampled row counters stay burned on cancellation")
        XCTAssertThrowsError(try stream.beginGumbelRound(drafts: 8, vocabulary: 3))
        XCTAssertThrowsError(try stream.beginGumbelRound(drafts: 1, vocabulary: 0))
        XCTAssertEqual(partial.token, try QWMTP.checkedGumbelWinner(
            distribution: rows[0], seed: 137, offset: 20))
    }

    func testTargetDrawFrozenCloneAndActiveRefusals() throws {
        let q = try QWSampler(config: QWSamplerConfig(temperature: 1, topK: 3, topP: 1))
            .distribution(logits: [0, 1, 2])
        let stream = QWMTPRandomStream(seed: 117)
        XCTAssertEqual(try stream.sampleTarget(q), try q.sampleToken(
            uniform: Double(QWPhilox.u(0, seed: 117, version: .upper24))))
        let frozen = try stream.frozenState()
        let clone = QWMTPRandomStream(cloning: frozen)
        XCTAssertEqual(try stream.sampleTarget(q), try clone.sampleTarget(q))
        XCTAssertEqual(stream.drawsConsumed, 2)
        try stream.beginGumbelRound(drafts: 1, vocabulary: 3)
        XCTAssertThrowsError(try stream.sampleTarget(q))
        XCTAssertThrowsError(try stream.frozenState())
        _ = try stream.sampleGumbelProposal(q)
        stream.cancelActiveRound()
        stream.cancelActiveRound()
        XCTAssertEqual(stream.drawsConsumed, 5)
        XCTAssertFalse(stream.isRoundActive)
        XCTAssertEqual(frozen.drawsConsumed, 1)
        XCTAssertThrowsError(try QWMTPRandomStream(seed: 0, drawIndex: UInt64.max).sampleTarget(q))
        let nearMax = QWMTPRandomStream(seed: 0, drawIndex: UInt64.max - 5)
        XCTAssertThrowsError(try nearMax.beginGumbelRound(drafts: 1, vocabulary: 3))
        XCTAssertEqual(nearMax.drawsConsumed, UInt64.max - 5)
        XCTAssertFalse(nearMax.isRoundActive)
    }
}
