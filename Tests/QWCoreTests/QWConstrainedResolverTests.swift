// QWConstrainedResolverTests.swift — Tests for constrained speculative verification and batch slot selection.

import XCTest
@testable import QWCore

final class QWConstrainedResolverTests: XCTestCase {
    func testBatchPartitionAllowsConstrainedSpeculationWhenFlagSet() {
        XCTAssertFalse(
            QWBatchPartitionSelection.isSlotSpeculatable(
                speculativeEnabled: true,
                remaining: 8,
                hasLogprobsSampler: false,
                isUnconstrainedNow: false,
                hasConstraint: true,
                hasConstraintSampler: true,
                canSpeculateConstrained: false
            )
        )
        XCTAssertTrue(
            QWBatchPartitionSelection.isSlotSpeculatable(
                speculativeEnabled: true,
                remaining: 8,
                hasLogprobsSampler: false,
                isUnconstrainedNow: false,
                hasConstraint: true,
                hasConstraintSampler: true,
                canSpeculateConstrained: true
            )
        )
    }

    func testConstrainedResolverVerifiesViaMaskedSampleRowAndTracksMTPResync() throws {
        // Drafts [10, 20, 30] are canonical constrained tokens, while raw unconstrained argmaxes are [99, 98, 97, 96]
        // (forbidden tokens with higher raw logits). Masked sampleRow returns [10, 20, 44] so 2 drafts accept!
        let drafts = [10, 20, 30]
        let rawArgmaxes = [99, 98, 97, 96]
        let maskedWinners = [10, 20, 44, 55]
        var sampledRows: [Int] = []
        var committed: [Int] = []
        var constraintAcceptedCalls = 0

        let res = try QWSpeculativeTokenResolver.resolveVerifiedRound(
            drafts: drafts,
            targetArgmaxes: rawArgmaxes,
            vocabulary: 128,
            isGreedy: true,
            isUnconstrainedNow: { false },
            boundaryToken: nil,
            canSpeculateConstrained: true,
            isRepetitionBlocked: { _ in false },
            acceptConstrained: { _ in constraintAcceptedCalls += 1 },
            sampleRow: { row, _ in
                sampledRows.append(row)
                return maskedWinners[row]
            },
            onCommittedToken: { committed.append($0) }
        )

        XCTAssertEqual(sampledRows, [0, 1, 2])
        XCTAssertEqual(res.acceptedDrafts, 2)
        XCTAssertEqual(res.rollbackCount, 1)
        XCTAssertEqual(res.tokens, [10, 20, 44])
        XCTAssertEqual(committed, [10, 20, 44])
        XCTAssertEqual(constraintAcceptedCalls, 0, "sampleRow advances live constraint when constrained")
        XCTAssertTrue(
            res.needsResync(targetArgmaxes: rawArgmaxes),
            "when accepted constrained drafts differ from raw unconstrained argmaxes, MTP hidden seed must resync"
        )
    }
}
