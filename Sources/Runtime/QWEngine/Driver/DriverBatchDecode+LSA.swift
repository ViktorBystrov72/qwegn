// DriverBatchDecode+LSA.swift — Short-context LSA eligibility & state handoff for batched decode.

import Foundation
import QWCore
import QWAttn

extension QWEngineDriver {
    /// Returns true when a session with `remaining` tokens stays strictly within the dense LSA window.
    func isSlotEligibleForLSABatchedDecode(position: Int, remaining: Int) -> Bool {
        guard lsaOperator != nil else { return true }
        return position + max(1, remaining) <= lsaConfig.denseBypassThreshold
    }

    /// Selects up to `maxRows` batch-eligible active indices, honoring the LSA dense window.
    func eligibleBatchSlotIndices(slots: [(position: Int, remaining: Int)], maxRows: Int) -> [Int] {
        QWBatchPartitionSelection.eligibleSlotIndices(count: slots.count, maxRows: maxRows) { idx in
            isSlotEligibleForLSABatchedDecode(position: slots[idx].position, remaining: slots[idx].remaining)
        }
    }

    /// Stashes any prior non-batch LSA state owner and clears ownership prior to batched dense decode.
    func prepareLSASessionsForBatchStep(_ sessions: [QWEngineSession]) throws {
        guard let lsa = lsaOperator else { return }
        for sess in sessions {
            guard sess.position < lsaConfig.denseBypassThreshold else {
                throw QWEngineDriverError.state(
                    "LSA batched decode requires session position < \(lsaConfig.denseBypassThreshold)")
            }
        }
        if let prev = lsa.stateOwner as? QWEngineSession,
           !prev.sessionFinished, !prev.batchedDecodeTouched, prev.position > 0 {
            prev.lastCommitCB?.waitUntilCompleted()
            prev.mlaIndexerStash = (lsa.captureIndexerState(position: prev.position), prev.position)
        }
        lsa.stateOwner = nil
    }
}
