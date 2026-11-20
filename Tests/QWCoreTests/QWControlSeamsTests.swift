// QWControlSeamsTests.swift — Unit tests for QWCore L0 batch, cache, and speculative control seams.

import Dispatch
import Foundation
import XCTest
@testable import QWCore

private struct SeamTestKey: Hashable, Sendable {
    let tokens: [Int]
    let mode: String
}

private final class SeamTestJob: @unchecked Sendable {
    let id: Int
    init(_ id: Int) { self.id = id }
}

private final class LockedFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    var value: Bool {
        get { lock.withLock { flag } }
        set { lock.withLock { flag = newValue } }
    }
}

final class QWControlSeamsTests: XCTestCase {
    func testBatchAdmissionCounterAndPromiseBehavior() throws {
        while QWBatchAdmissionCounter.preparingCount > 0 { QWBatchAdmissionCounter.endPreparing() }
        XCTAssertEqual(QWBatchAdmissionCounter.preparingCount, 0)
        QWBatchAdmissionCounter.beginPreparing()
        XCTAssertEqual(QWBatchAdmissionCounter.preparingCount, 1)
        QWBatchAdmissionCounter.coalescePreparingBurst(
            maxSlots: 4, maxWaitUs: 1_000, pollIntervalUs: 250, queuedCount: { 4 }
        )
        QWBatchAdmissionCounter.endPreparing()
        QWBatchAdmissionCounter.endPreparing()
        XCTAssertEqual(QWBatchAdmissionCounter.preparingCount, 0)

        let promise = QWBatchJobPromise<Int>()
        promise.complete(.success(42))
        promise.complete(.failure(QWBatchSchedulerError.missingJobOutcome))
        XCTAssertEqual(try promise.wait(), 42)

        let errPromise = QWBatchJobPromise<Int>()
        errPromise.complete(.failure(QWBatchSchedulerError.missingJobOutcome))
        XCTAssertThrowsError(try errPromise.wait())
    }

    func testBatchPartitionSelectionAndPlanAdmissions() {
        XCTAssertEqual(QWBatchPartitionSelection.slotCapacity(fullCapacity: 4096, totalSlots: 4, alignment: 16), 1024)
        XCTAssertEqual(QWBatchPartitionSelection.slotCapacity(fullCapacity: 4096, totalSlots: 1, alignment: 16), 4096)
        XCTAssertEqual(QWBatchPartitionSelection.clampedMaxNew(requested: 100, promptCount: 90, activeCapacity: 128), 38)
        XCTAssertEqual(QWBatchPartitionSelection.clampedMaxNew(requested: -5, promptCount: 90, activeCapacity: 128), 0)
        XCTAssertTrue(QWBatchPartitionSelection.validateRequestBounds(prompt: [1, 2], maxNew: 0, stopIDs: [3], vocab: 10))
        XCTAssertFalse(QWBatchPartitionSelection.validateRequestBounds(prompt: [], maxNew: 1, stopIDs: [], vocab: 10))
        XCTAssertFalse(QWBatchPartitionSelection.validateRequestBounds(prompt: [10], maxNew: 1, stopIDs: [], vocab: 10))

        let eligible = QWBatchPartitionSelection.eligibleSlotIndices(count: 5, maxRows: 2) { $0 % 2 == 1 }
        XCTAssertEqual(eligible, [1, 3])
        XCTAssertEqual(
            QWBatchPartitionSelection.nextRoundRobinIndex(current: 0, stepIndices: [1, 3], activeCount: 5, mixedShortAndLong: true),
            4
        )

        let singleSolo = QWBatchPartitionSelection.planAdmissions(
            maxSlots: 4, slotCapacity: 256, slot0ActiveInBatchMode: false, slot0FitsInBatch: true,
            occupiedSlots: [false, false, false, false],
            pendingDemands: [.init(promptCount: 16, maxNew: 16)]
        )
        XCTAssertEqual(singleSolo, [.init(slot: 0, queueIndex: 0, inBatchMode: false)])

        let multi = QWBatchPartitionSelection.planAdmissions(
            maxSlots: 4, slotCapacity: 256, slot0ActiveInBatchMode: true, slot0FitsInBatch: true,
            occupiedSlots: [false, false, false, false],
            pendingDemands: [.init(promptCount: 32, maxNew: 32), .init(promptCount: 64, maxNew: 32)]
        )
        XCTAssertEqual(multi, [.init(slot: 0, queueIndex: 0, inBatchMode: true), .init(slot: 1, queueIndex: 1, inBatchMode: true)])

        let oversizedSolo = QWBatchPartitionSelection.planAdmissions(
            maxSlots: 4, slotCapacity: 256, slot0ActiveInBatchMode: false, slot0FitsInBatch: false,
            occupiedSlots: [false, false, false, false],
            pendingDemands: [.init(promptCount: 300, maxNew: 16), .init(promptCount: 16, maxNew: 16)]
        )
        XCTAssertEqual(oversizedSolo, [.init(slot: 0, queueIndex: 0, inBatchMode: false)])

        let blockedBySlot0 = QWBatchPartitionSelection.planAdmissions(
            maxSlots: 4, slotCapacity: 256, slot0ActiveInBatchMode: false, slot0FitsInBatch: false,
            occupiedSlots: [true, false, false, false],
            pendingDemands: [.init(promptCount: 16, maxNew: 16)]
        )
        XCTAssertTrue(blockedBySlot0.isEmpty)
    }

    func testBatchSlotCoordinatorWaveAndOutputState() {
        let coord = QWBatchSlotCoordinator<SeamTestJob>(maxSlots: 2)
        let j1 = SeamTestJob(1), j2 = SeamTestJob(2), j3 = SeamTestJob(3)
        XCTAssertTrue(coord.submit(j1))
        XCTAssertFalse(coord.submit(j2))
        XCTAssertEqual(coord.queuedCount, 2)
        let polled = coord.pollPending(maxCount: 1)
        XCTAssertEqual(polled.map(\.id), [1])
        XCTAssertEqual(coord.queuedCount, 1)
        XCTAssertFalse(coord.tryParkDriverIfIdle { _ = coord.submit(j3) })
        _ = coord.pollPending(maxCount: 2)
        XCTAssertTrue(coord.tryParkDriverIfIdle {})

        let ranExclusive = LockedFlag()
        coord.runExclusive { ranExclusive.value = true }
        XCTAssertTrue(ranExclusive.value)

        var wave = QWBatchWaveState()
        wave.markBlocked(freeWatermark: 10)
        XCTAssertFalse(wave.canPollPending(openSlots: 1))
        wave.refreshBlock(currentFree: 11)
        XCTAssertTrue(wave.canPollPending(openSlots: 1))

        var out = QWBatchSlotOutputState()
        out.markDecodeStartedIfNeeded()
        XCTAssertTrue(out.beginPredictedToken(5, stopIDs: [99], stepMsValue: 1.5))
        out.finishPredictedToken(maxNew: 2, accepted: true)
        XCTAssertFalse(out.finished)
        XCTAssertFalse(out.beginPredictedToken(99, stopIDs: [99], stepMsValue: 2.0))
        XCTAssertTrue(out.finished)
        XCTAssertEqual(out.stoppedToken, 99)
        out.recordSpeculativeRound(tokens: [7, 8], stepMsValue: 3.0, maxNew: 10, stopped: false, stoppedToken: nil, onToken: nil)
        XCTAssertEqual(out.speculativeSummary(enabled: true, proposals: 4, accepted: 2), "rounds=1, mean=2.00 tok/rnd")
    }

    func testAsyncColdWriteControllerAndColdCacheIndex() throws {
        let ctrl = QWAsyncColdWriteController()
        XCTAssertFalse(ctrl.reserve(bytes: 600 << 20))
        XCTAssertTrue(ctrl.reserve(bytes: 64))
        let ran = LockedFlag()
        ctrl.submitReserved(bytes: 64) { ran.value = true }
        ctrl.waitUntilIdle()
        XCTAssertTrue(ran.value)
        ctrl.closeAndDrain()
        XCTAssertFalse(ctrl.reserve(bytes: 1))
        ctrl.reopenAndDrain()
        XCTAssertTrue(ctrl.reserve(bytes: 1))
        ctrl.cancelReservation(bytes: 1)

        let idx = QWColdCacheIndex<String>(ssdByteLimit: 100, entryLimit: 2)
        let u1 = URL(fileURLWithPath: "/tmp/a.qwcache")
        let u2 = URL(fileURLWithPath: "/tmp/b.qwcache")
        let u3 = URL(fileURLWithPath: "/tmp/c.qwcache")
        var evicted: [URL] = []
        XCTAssertTrue(try idx.record(key: "k1", metadata: "m1", fileURL: u1, bytes: 60) { } onEvict: { evicted.append($0) })
        XCTAssertTrue(try idx.record(key: "k2", metadata: "m2", fileURL: u2, bytes: 60) { } onEvict: { evicted.append($0) })
        XCTAssertEqual(evicted, [u1])
        XCTAssertEqual(idx.chargedBytes, 60)
        XCTAssertEqual(idx.lookupExact("k2", verify: { $0 == "m2" })?.fileURL, u2)
        XCTAssertTrue(try idx.record(key: "k3", metadata: "m3", fileURL: u3, bytes: 40) { } onEvict: { evicted.append($0) })
        XCTAssertEqual(idx.entryCount, 2)
        XCTAssertEqual(idx.lookupBestPrefix { k, _ in k == "k3" ? 40 : 10 }?.key, "k3")
        idx.purgeAll { evicted.append($0) }
        XCTAssertEqual(idx.entryCount, 0)
        XCTAssertEqual(idx.chargedBytes, 0)
    }

    func testTwoTierPrefixCacheLookupPinningAndBudgets() {
        let store = QWSlotPinnedLRUStore<SeamTestKey, String>(byteLimit: 4096, entryLimit: 8, maxSlots: 2, rootPrefixTokenCount: 4)
        let baseKey = SeamTestKey(tokens: [1, 2, 3, 4], mode: "unsalted")
        let exactKey = SeamTestKey(tokens: [1, 2, 3, 4, 5, 6], mode: "salted")
        let saltedBase = SeamTestKey(tokens: [1, 2, 3, 4, 100], mode: "salted")
        XCTAssertTrue(QWTwoTierPrefixCache.storeWithSlotPinning(
            in: store, key: baseKey, value: "base", charge: 128, tokenCount: 4,
            allowsSuffix: true, rootHash: 99, history: [1, 2, 3, 4], isSharedBase: true,
            matchesPrefix: { _, _ in true }
        ))
        let exactHit = QWTwoTierPrefixCache.lookupWarmPrefix(
            in: store, exactKey: baseKey, saltedBaseKey: nil, tokenCount: 4,
            queryRootHash: 99, allowExact: true, allowPrefix: true,
            isCandidate: { _, _, _ in true }, matchesPrefix: { _, _, _ in true }
        )
        XCTAssertEqual(exactHit?.value, "base")

        let noExact = QWTwoTierPrefixCache.lookupWarmPrefix(
            in: store, exactKey: baseKey, saltedBaseKey: nil, tokenCount: 4,
            queryRootHash: 99, allowExact: false, allowPrefix: true,
            isCandidate: { _, _, _ in true }, matchesPrefix: { _, _, _ in true }
        )
        XCTAssertNil(noExact)

        let sharedFallback = QWTwoTierPrefixCache.lookupWarmPrefix(
            in: store, exactKey: exactKey, saltedBaseKey: saltedBase, tokenCount: 6,
            queryRootHash: nil, allowExact: true, allowPrefix: true,
            isCandidate: { k, _, sharedOnly in sharedOnly ? (k.mode == "unsalted") : (k.mode == "salted") },
            matchesPrefix: { k, _, _ in k == baseKey }
        )
        XCTAssertEqual(sharedFallback?.matchedTokens, 4)
        XCTAssertEqual(sharedFallback?.isUnsaltedSharedBase, true)

        XCTAssertTrue(QWTwoTierPrefixCache.shouldSpillToCold(
            isSharedBase: false, warmHasSharedBase: false, ssdByteLimit: 1024, entryBytes: 512, maxSpillBytes: 1024
        ))
        XCTAssertFalse(QWTwoTierPrefixCache.shouldSpillToCold(
            isSharedBase: false, warmHasSharedBase: true, ssdByteLimit: 1024, entryBytes: 512, maxSpillBytes: 1024
        ))
        XCTAssertEqual(QWTwoTierPrefixCache.incrementalCharge(fullBytes: 500, sharedBaseBytes: 200), 300)
        XCTAssertEqual(
            QWTwoTierPrefixCache.computeRetainPoints(
                explicitRetainLength: 20, matchedTokens: 0, firstVisualTokenOffset: 100,
                promptCount: 120, sharedBranchPrefixLength: 50
            ),
            [20, 96]
        )
    }

    func testSpeculativeTokenResolverParityAndRollbacks() throws {
        XCTAssertEqual(try QWSpeculativeTokenResolver.validateToken(2, vocabulary: 4), 2)
        XCTAssertThrowsError(try QWSpeculativeTokenResolver.validateToken(4, vocabulary: 4))
        XCTAssertThrowsError(try QWSpeculativeTokenResolver.validateTokens([1], vocabulary: 0))

        // Numerical draft canonicalizes -0/+0 while shippingTarget treats +0 as greater bit key.
        let zeros: [Float] = [-0.0, 0.0]
        try zeros.withUnsafeBufferPointer { buf in
            XCTAssertEqual(try QWSpeculativeTokenResolver.greedyToken(buf, ordering: .numericalDraft), 0)
            XCTAssertEqual(try QWSpeculativeTokenResolver.greedyToken(buf, ordering: .shippingTarget), 1)
        }
        let nonFinite: [Float] = [1.0, .nan]
        XCTAssertThrowsError(try nonFinite.withUnsafeBufferPointer { buf in
            try QWSpeculativeTokenResolver.greedyToken(buf, ordering: .numericalDraft)
        })

        let relaxedRow: [Float] = [0.0, 4.9, 5.0, -5.0]
        let resRelaxed = try relaxedRow.withUnsafeBufferPointer { buf in
            try QWSpeculativeTokenResolver.resolveGreedy(
                drafts: [1], targetArgmaxes: [2, 3], vocabulary: 4, maxNew: 5,
                relaxedMargin: 0.25, relaxedMinProb: 0.1, logitRow: { _ in buf }
            )
        }
        XCTAssertEqual(resRelaxed.acceptedDrafts, 1)
        XCTAssertEqual(resRelaxed.rawArgmaxAccepted, 0)
        XCTAssertEqual(resRelaxed.tokens, [1, 3])
        XCTAssertEqual(resRelaxed.rollbackCount, 0)
        XCTAssertTrue(resRelaxed.needsResync(targetArgmaxes: [2, 3]))

        let resStopped = try QWSpeculativeTokenResolver.resolveGreedy(
            drafts: [1, 2], targetArgmaxes: [1, 2, 9], vocabulary: 16, maxNew: 4, stopIDs: [2]
        )
        XCTAssertEqual(resStopped.acceptedDrafts, 2)
        XCTAssertEqual(resStopped.tokens, [1])
        XCTAssertTrue(resStopped.stopped)
        XCTAssertEqual(resStopped.stoppedToken, 2)

        var boundaryHit = false
        var committed: [Int] = []
        let verified = try QWSpeculativeTokenResolver.resolveVerifiedRound(
            drafts: [3, 4], targetArgmaxes: [3, 4, 5], vocabulary: 16,
            isGreedy: true, isUnconstrainedNow: { !boundaryHit }, boundaryToken: 4,
            isRepetitionBlocked: { _ in false },
            acceptConstrained: { tok in if tok == 4 { boundaryHit = true } },
            sampleRow: { _, tok in tok },
            onCommittedToken: { committed.append($0) }
        )
        XCTAssertEqual(verified.acceptedDrafts, 1)
        XCTAssertEqual(verified.rollbackCount, 1)
        XCTAssertEqual(committed, [3, 4])
        XCTAssertEqual(try QWSpeculativeTokenResolver.retainedPairRows(retainedRows: 3, draftCount: 2, acceptedDrafts: 1), 2)
    }
}
