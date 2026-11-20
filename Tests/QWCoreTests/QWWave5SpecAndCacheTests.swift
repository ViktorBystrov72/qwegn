// QWWave5SpecAndCacheTests.swift — Tests for zero-alloc QWTokenRecycler Int64 ingestion and QWSpeculativeTokenResolver.
import XCTest
@testable import QWCore

final class QWWave5SpecAndCacheTests: XCTestCase {

    func testRecyclerInt64IngestionMatchesIntSequenceBitExact() {
        let sequences: [[Int]] = [
            [11, 22, 33, 44, 55, 66, 77],
            [11, 22, 88, 99, 101, 22, 33, 102],
            [5, -1, 6, 7, Int(Int32.max) + 1, 8, 9, 10],
            [42]
        ]
        for seq in sequences {
            let ref = QWTokenRecycler()
            let opt = QWTokenRecycler()
            let seq64 = seq.map(Int64.init)
            ref.ingestSequence(seq)
            opt.ingestSequence(seq64)

            for p in seq {
                var dRef: [Int] = []
                var dOpt: [Int] = []
                let nRef = ref.appendDrafts(into: &dRef, maxCount: 6, previousToken: 11, pendingToken: p)
                let nOpt = opt.appendDrafts(into: &dOpt, maxCount: 6, previousToken: 11, pendingToken: p)
                XCTAssertEqual(nRef, nOpt)
                XCTAssertEqual(dRef, dOpt)
            }
        }

        // Verify slice overload with explicit previousToken matches ref slice ingestion.
        let full = [3, 7, 13, 19, 29, 37, 41]
        let full64 = full.map(Int64.init)
        let ref = QWTokenRecycler()
        let opt = QWTokenRecycler()
        ref.ingestSequence(full[1...], previousToken: full[0])
        opt.ingestSequence(full64[1...], previousToken: full[0])
        var dRef: [Int] = [13]
        var dOpt: [Int] = [13]
        XCTAssertEqual(
            ref.appendDrafts(into: &dRef, maxCount: 5, previousToken: 3, pendingToken: 7),
            opt.appendDrafts(into: &dOpt, maxCount: 5, previousToken: 3, pendingToken: 7)
        )
        XCTAssertEqual(dRef, dOpt)
        XCTAssertEqual(dOpt, [13, 19, 29, 37, 41])
    }

    func testRecyclerAppendDraftsPreservesExistingPrefixAndGuardsCycles() {
        let recycler = QWTokenRecycler()
        recycler.ingestSequence([1, 2, 3, 2, 3, 2, 3, 4])

        var drafts = [3]
        let added = recycler.appendDrafts(
            into: &drafts, maxCount: 6, previousToken: 1, pendingToken: 2, maxUnigramFallbacks: 1
        )
        XCTAssertGreaterThanOrEqual(added, 0)
        XCTAssertLessThanOrEqual(drafts.count, 6)

        // Compare incremental extension from empty vs seeded prefix.
        let recycler2 = QWTokenRecycler()
        recycler2.ingestSequence([100, 200, 300, 400, 500])
        var full: [Int] = []
        var partial: [Int] = [300, 400]
        _ = recycler2.appendDrafts(into: &full, maxCount: 3, previousToken: 100, pendingToken: 200)
        _ = recycler2.appendDrafts(into: &partial, maxCount: 3, previousToken: 100, pendingToken: 200)
        XCTAssertEqual(full, [300, 400, 500])
        XCTAssertEqual(partial, [300, 400, 500])
    }

    func testSpeculativeResolverGreedyAndVerifiedRoundBoundaries() throws {
        let drafts = [10, 20, 30]
        let targetsMatchTwo = [10, 20, 99, 77]

        var acceptedLog: [Int] = []
        var committedLog: [Int] = []
        let res = try QWSpeculativeTokenResolver.resolveVerifiedRound(
            drafts: drafts,
            targetArgmaxes: targetsMatchTwo,
            vocabulary: 128,
            isGreedy: true,
            isUnconstrainedNow: { true },
            boundaryToken: nil,
            isRepetitionBlocked: { _ in false },
            acceptConstrained: { acceptedLog.append($0) },
            sampleRow: { row, argmaxTok in
                XCTAssertEqual(row, 2)
                return argmaxTok
            },
            onCommittedToken: { committedLog.append($0) }
        )
        XCTAssertEqual(res.acceptedDrafts, 2)
        XCTAssertEqual(res.rollbackCount, 1)
        XCTAssertEqual(res.tokens, [10, 20, 99])
        XCTAssertEqual(acceptedLog, [10, 20])
        XCTAssertEqual(committedLog, [10, 20, 99])
        XCTAssertFalse(res.needsResync(targetArgmaxes: targetsMatchTwo))

        // Unblock greedy acceptance checks: repetition block forces sampleRow fallback.
        let blockedRes = try QWSpeculativeTokenResolver.resolveVerifiedRound(
            drafts: drafts,
            targetArgmaxes: [10, 20, 30, 40],
            vocabulary: 128,
            isGreedy: true,
            isUnconstrainedNow: { true },
            boundaryToken: nil,
            isRepetitionBlocked: { $0 == 20 },
            acceptConstrained: { _ in },
            sampleRow: { row, _ in row == 1 ? 88 : 99 },
            onCommittedToken: { _ in }
        )
        XCTAssertEqual(blockedRes.acceptedDrafts, 1)
        XCTAssertEqual(blockedRes.rollbackCount, 2)
        XCTAssertEqual(blockedRes.tokens, [10, 88])
        XCTAssertTrue(blockedRes.needsResync(targetArgmaxes: [10, 20, 30, 40]))
    }
}
