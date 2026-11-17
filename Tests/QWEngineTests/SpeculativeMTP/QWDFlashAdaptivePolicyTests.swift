import XCTest
@testable import QWEngine

final class QWDFlashAdaptivePolicyTests: XCTestCase {
    func testInitialSelectionCapsAtSevenAndDropsOnNovelText() {
        var policy = QWDFlashAdaptivePolicy()
        XCTAssertEqual(policy.selectDraftCount(maxDrafts: 15), 7)
        policy.observeRound(draftCount: 7, accepted: 1, roundMs: 58.4)
        XCTAssertEqual(policy.selectDraftCount(maxDrafts: 15), 3)
        for _ in 0..<4 {
            policy.observeRound(draftCount: 3, accepted: 1, roundMs: 47.0)
            _ = policy.selectDraftCount(maxDrafts: 15)
        }
        XCTAssertEqual(policy.selectDraftCount(maxDrafts: 15), 3)
    }

    func testPromotesImmediateFullAcceptToFifteenAndSustainsHighAccept() {
        var policy = QWDFlashAdaptivePolicy()
        XCTAssertEqual(policy.selectDraftCount(maxDrafts: 15), 7)
        policy.observeRound(draftCount: 7, accepted: 7, roundMs: 58.0)
        XCTAssertEqual(policy.selectDraftCount(maxDrafts: 15), 15)
        for _ in 0..<6 {
            policy.observeRound(draftCount: 15, accepted: 15, roundMs: 105.0)
            XCTAssertEqual(policy.selectDraftCount(maxDrafts: 15), 15)
        }
    }

    func testPeriodicExploreProbesLargerWindowEverySixteenRounds() {
        var policy = QWDFlashAdaptivePolicy()
        _ = policy.selectDraftCount(maxDrafts: 15)
        policy.observeRound(draftCount: 7, accepted: 0, roundMs: 58.0)
        for _ in 2..<16 {
            let d = policy.selectDraftCount(maxDrafts: 15)
            XCTAssertEqual(d, 3)
            policy.observeRound(draftCount: d, accepted: 1, roundMs: 47.0)
        }
        XCTAssertEqual(policy.selectDraftCount(maxDrafts: 15), 7)
    }

    func testSevenDraftPackSelectsThreeOrSevenWithoutExceedingCap() {
        var policy = QWDFlashAdaptivePolicy()
        XCTAssertEqual(policy.selectDraftCount(maxDrafts: 7), 7)
        policy.observeRound(draftCount: 7, accepted: 0, roundMs: 58.0)
        XCTAssertEqual(policy.selectDraftCount(maxDrafts: 7), 3)
        policy.observeRound(draftCount: 3, accepted: 3, roundMs: 45.0)
        _ = policy.selectDraftCount(maxDrafts: 7)
        policy.observeRound(draftCount: 3, accepted: 3, roundMs: 45.0)
        XCTAssertEqual(policy.selectDraftCount(maxDrafts: 7), 7)
    }

    func testSparseMoEStartsAtThreeAndDropsToTwoOnLowAcceptance() {
        var policy = QWDFlashAdaptivePolicy(isSparseMoE: true)
        XCTAssertEqual(policy.selectDraftCount(maxDrafts: 7), 3)
        policy.observeRound(draftCount: 3, accepted: 0, roundMs: 31.0)
        XCTAssertEqual(policy.selectDraftCount(maxDrafts: 7), 2)
    }

    func testSparseMoEStepUpAndExploreGating() {
        var policy = QWDFlashAdaptivePolicy(isSparseMoE: true)
        XCTAssertEqual(policy.selectDraftCount(maxDrafts: 7), 3)
        policy.observeRound(draftCount: 3, accepted: 3, roundMs: 30.5)
        XCTAssertEqual(policy.selectDraftCount(maxDrafts: 7), 3)
        policy.observeRound(draftCount: 3, accepted: 3, roundMs: 30.5)
        XCTAssertEqual(policy.selectDraftCount(maxDrafts: 7), 5)
        for _ in 0..<5 {
            policy.observeRound(draftCount: 5, accepted: 5, roundMs: 39.5)
            _ = policy.selectDraftCount(maxDrafts: 7)
        }
        XCTAssertEqual(policy.selectDraftCount(maxDrafts: 7), 7)

        var cautious = QWDFlashAdaptivePolicy(isSparseMoE: true)
        XCTAssertEqual(cautious.selectDraftCount(maxDrafts: 7), 3)
        cautious.observeRound(draftCount: 3, accepted: 0, roundMs: 31.0)
        for _ in 2...15 {
            let d = cautious.selectDraftCount(maxDrafts: 7)
            cautious.observeRound(draftCount: d, accepted: 0, roundMs: 25.5)
        }
        XCTAssertEqual(cautious.selectDraftCount(maxDrafts: 7), 3)
        cautious.observeRound(draftCount: 3, accepted: 1, roundMs: 31.0)
        for _ in 17...31 {
            let d = cautious.selectDraftCount(maxDrafts: 7)
            cautious.observeRound(draftCount: d, accepted: 1, roundMs: 26.0)
        }
        XCTAssertLessThanOrEqual(cautious.selectDraftCount(maxDrafts: 7), 3)
    }
}
