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
}
