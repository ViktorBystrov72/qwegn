// QWDFlashAdaptivePolicy.swift — Online throughput-optimal window selector
// for block-diffusion drafters (DFlash 2 / DSpine) on M = 1 + D verification.

import Foundation

/// Picks the DFlash/DSpine draft window `D` in `{3, 7, maxDrafts}` that
/// maximizes expected committed tokens per millisecond (`E[tokens(D)] / costMs(D)`).
public struct QWDFlashAdaptivePolicy: Sendable {
    public static let fixedWindow: Bool =
        ProcessInfo.processInfo.environment["QWG_DFLASH_FIXED_WINDOW"] == "1"

    static let maxSupportedDrafts = 15
    private static let exploreEvery = 16

    private var acc: [Double]
    private var costMsBucket: (b3: Double, b7: Double, b15: Double)
    private var rounds: Int = 0
    private var lastDraftCount: Int = 0
    private var lastAcceptedAll: Bool = false
    private var consecutiveFullAccepts: Int = 0

    public init() {
        self.acc = [
            0.78, 0.70, 0.65, 0.62, 0.60,
            0.58, 0.56, 0.55, 0.54, 0.53,
            0.52, 0.51, 0.50, 0.50, 0.50,
        ]
        // Priors calibrated from M5 Max single-pass (M=4, M=8) vs 2x8 split (M=16) round times.
        self.costMsBucket = (47.0, 58.4, 107.0)
    }

    public mutating func selectDraftCount(maxDrafts: Int) -> Int {
        guard maxDrafts > 1 && !Self.fixedWindow else { return maxDrafts }
        let cap = min(maxDrafts, Self.maxSupportedDrafts)
        rounds += 1
        if rounds == 1 {
            let initial = min(7, cap)
            lastDraftCount = initial
            return initial
        }
        if lastAcceptedAll {
            if lastDraftCount == cap {
                return cap
            }
            if lastDraftCount < 7 || consecutiveFullAccepts >= 2 || (rounds == 2 && lastDraftCount == 7) {
                let promoted = (lastDraftCount < 7) ? min(7, cap) : cap
                lastDraftCount = promoted
                return promoted
            }
        }
        if rounds % Self.exploreEvery == 0 {
            let mid = min(7, cap)
            if lastDraftCount < mid {
                lastDraftCount = mid
                return mid
            }
            if lastDraftCount < cap && reach(drafts: mid) >= 0.35 {
                lastDraftCount = cap
                return cap
            }
        }
        let w0 = min(3, cap)
        let w1 = min(7, cap)
        let w2 = cap
        var best = w0
        var bestRate = expectedTokens(drafts: w0) / bucketCost(drafts: w0)
        if w1 != w0 {
            let r1 = expectedTokens(drafts: w1) / bucketCost(drafts: w1)
            if r1 > bestRate { bestRate = r1; best = w1 }
        }
        if w2 != w1 && w2 != w0 {
            let r2 = expectedTokens(drafts: w2) / bucketCost(drafts: w2)
            if r2 > bestRate { bestRate = r2; best = w2 }
        }
        lastDraftCount = best
        return best
    }

    public mutating func observeRound(draftCount: Int, accepted: Int, roundMs: Double?) {
        guard draftCount > 0 else { return }
        lastAcceptedAll = (accepted >= draftCount)
        consecutiveFullAccepts = lastAcceptedAll ? (consecutiveFullAccepts + 1) : 0
        let alpha = (rounds <= 5 && accepted <= 1) ? 0.55 : 0.18
        let limit = min(draftCount, acc.count)
        for i in 0..<limit where accepted >= i {
            let target = accepted > i ? 1.0 : 0.0
            acc[i] += alpha * (target - acc[i])
        }
        guard let ms = roundMs, ms > 0 else { return }
        let msAlpha = 0.20
        if draftCount <= 3 {
            costMsBucket.b3 += msAlpha * (ms - costMsBucket.b3)
        } else if draftCount <= 7 {
            costMsBucket.b7 += msAlpha * (ms - costMsBucket.b7)
        } else {
            costMsBucket.b15 += msAlpha * (ms - costMsBucket.b15)
        }
    }

    func expectedTokens(drafts d: Int) -> Double {
        let limit = min(max(0, d), acc.count)
        var total = 1.0
        var prob = 1.0
        for i in 0..<limit {
            prob *= acc[i]
            total += prob
        }
        return total
    }

    private func reach(drafts d: Int) -> Double {
        let limit = min(max(0, d), acc.count)
        var prob = 1.0
        for i in 0..<limit { prob *= acc[i] }
        return prob
    }

    private func bucketCost(drafts d: Int) -> Double {
        if d <= 3 { return costMsBucket.b3 }
        if d <= 7 { return costMsBucket.b7 }
        return costMsBucket.b15
    }
}
