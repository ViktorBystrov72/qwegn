// QWTokenRecyclerTests.swift — Tests for QWTokenRecycler and zero-copy
// QWContextCopyDrafter sequence matching.
import XCTest
@testable import QWCore

final class QWTokenRecyclerTests: XCTestCase {

    func testIngestSequenceAndBigramChaining() {
        let recycler = QWTokenRecycler()
        recycler.ingestSequence([10, 20, 30, 40, 50, 60])

        var drafts: [Int] = []
        let added = recycler.appendDrafts(
            into: &drafts, maxCount: 4, previousToken: 10, pendingToken: 20)
        XCTAssertEqual(added, 4)
        XCTAssertEqual(drafts, [30, 40, 50, 60])
    }

    func testSliceIngestionWithPreviousTokenAndPartialDraftExtension() {
        let recycler = QWTokenRecycler()
        let full = [5, 11, 22, 33, 44, 55]
        recycler.ingestSequence(full[1...], previousToken: full[0])

        var single = [22]
        let addedFromOne = recycler.appendDrafts(
            into: &single, maxCount: 4, previousToken: 5, pendingToken: 11)
        XCTAssertEqual(addedFromOne, 3)
        XCTAssertEqual(single, [22, 33, 44, 55])

        var multi = [22, 33]
        let addedFromTwo = recycler.appendDrafts(
            into: &multi, maxCount: 4, previousToken: 5, pendingToken: 11)
        XCTAssertEqual(addedFromTwo, 2)
        XCTAssertEqual(multi, [22, 33, 44, 55])
    }

    func testReverseOrderMRUPriorityAcceptedOverRejected() {
        let recycler = QWTokenRecycler()
        // Caller records rejected rows first and accepted rows last in reverse order.
        recycler.recordTransition(previous: 7, current: 8, next: 99)
        recycler.recordTransition(previous: 7, current: 8, next: 42)

        var drafts: [Int] = []
        recycler.appendDrafts(
            into: &drafts, maxCount: 1, previousToken: 7, pendingToken: 8)
        XCTAssertEqual(drafts, [42])
    }

    func testBigramPreferenceAndMaxUnigramFallbacksCap() {
        let recycler = QWTokenRecycler()
        recycler.recordTransition(previous: nil, current: 100, next: 101)
        recycler.recordTransition(previous: nil, current: 101, next: 102)
        recycler.recordTransition(previous: nil, current: 102, next: 103)
        // Bigram overrides unigram when (prev, cur) is known.
        recycler.recordTransition(previous: 99, current: 100, next: 201)
        recycler.recordTransition(previous: 100, current: 201, next: 202)

        var cappedZero: [Int] = []
        recycler.appendDrafts(
            into: &cappedZero, maxCount: 5, previousToken: nil, pendingToken: 100,
            maxUnigramFallbacks: 0)
        XCTAssertEqual(cappedZero, [])

        var cappedTwo: [Int] = []
        recycler.appendDrafts(
            into: &cappedTwo, maxCount: 5, previousToken: 99, pendingToken: 100,
            maxUnigramFallbacks: 2)
        // Expected path: bigram (99, 100)->201, bigram (100, 201)->202; no further match.
        XCTAssertEqual(cappedTwo, [201, 202])

        recycler.recordTransition(previous: nil, current: 202, next: 203)
        recycler.recordTransition(previous: nil, current: 203, next: 204)
        recycler.recordTransition(previous: nil, current: 204, next: 205)

        var cappedAtOne: [Int] = []
        recycler.appendDrafts(
            into: &cappedAtOne, maxCount: 6, previousToken: 99, pendingToken: 100,
            maxUnigramFallbacks: 1)
        XCTAssertEqual(cappedAtOne, [201, 202, 203])

        var cappedDefaultTwo: [Int] = []
        recycler.appendDrafts(
            into: &cappedDefaultTwo, maxCount: 6, previousToken: 99, pendingToken: 100,
            maxUnigramFallbacks: 2)
        XCTAssertEqual(cappedDefaultTwo, [201, 202, 203, 204])
    }

    func testCycleGuardsStopRepeatingPairsAndSelfLoops() {
        let recycler = QWTokenRecycler()
        recycler.ingestSequence([1, 2, 3, 1, 2, 3])

        var biDrafts: [Int] = []
        recycler.appendDrafts(
            into: &biDrafts, maxCount: 8, previousToken: 1, pendingToken: 2,
            maxUnigramFallbacks: 0)
        XCTAssertEqual(biDrafts, [3, 1, 2])

        let uniRecycler = QWTokenRecycler()
        uniRecycler.recordTransition(previous: nil, current: 50, next: 51)
        uniRecycler.recordTransition(previous: nil, current: 51, next: 51)

        var uniDrafts: [Int] = []
        uniRecycler.appendDrafts(
            into: &uniDrafts, maxCount: 6, previousToken: nil, pendingToken: 50,
            maxUnigramFallbacks: 5)
        XCTAssertEqual(uniDrafts, [51])
    }

    func testResetAndBoundaryInputs() {
        let recycler = QWTokenRecycler()
        recycler.ingestSequence([10, 20, 30])
        recycler.recordTransition(previous: -1, current: 9, next: 12)

        var negDrafts: [Int] = []
        let negAdded = recycler.appendDrafts(
            into: &negDrafts, maxCount: 4, previousToken: 10, pendingToken: -1)
        XCTAssertEqual(negAdded, 0)

        var full = [1, 2]
        let alreadyFull = recycler.appendDrafts(
            into: &full, maxCount: 2, previousToken: 10, pendingToken: 20)
        XCTAssertEqual(alreadyFull, 0)

        recycler.reset()
        var afterReset: [Int] = []
        let postResetAdded = recycler.appendDrafts(
            into: &afterReset, maxCount: 4, previousToken: 10, pendingToken: 20)
        XCTAssertEqual(postResetAdded, 0)
        XCTAssertTrue(afterReset.isEmpty)
    }

    func testContextCopyDrafterZeroCopyParityWithInt64Reference() {
        let historyInt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3]
        let extra = 4
        let trailing = [5]

        let combinedInt64 = (historyInt + [extra] + trailing).map(Int64.init)
        let refMatch = QWContextCopyDrafter.drafts(
            history: combinedInt64, maxCount: 4, minMatchLen: 3)
        let zeroCopyMatch = QWContextCopyDrafter.drafts(
            history: historyInt, extraToken: extra, trailingTokens: trailing,
            maxCount: 4, minMatchLen: 3)

        XCTAssertEqual(zeroCopyMatch?.tokens, refMatch?.tokens)
        XCTAssertEqual(zeroCopyMatch?.matchLen, refMatch?.matchLen)
        XCTAssertEqual(zeroCopyMatch?.tokens, [6, 7, 8, 9])

        let periodicHistory = [10, 11, 12, 10, 11, 12, 10, 11]
        let periodicExtra = 12
        let refPeriodic = QWContextCopyDrafter.periodicDrafts(
            history: (periodicHistory + [periodicExtra]).map(Int64.init),
            maxCount: 4, maxPeriod: 6, minRepeats: 2)
        let zeroCopyPeriodic = QWContextCopyDrafter.periodicDrafts(
            history: periodicHistory, extraToken: periodicExtra,
            maxCount: 4, maxPeriod: 6, minRepeats: 2)

        XCTAssertEqual(zeroCopyPeriodic, refPeriodic)
        XCTAssertEqual(zeroCopyPeriodic, [10, 11, 12, 10])
    }
}
