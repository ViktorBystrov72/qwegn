// QWHyperScheduleAndRing.swift — 4-slot rollback ring parity & layer stack schedule.

/// O(1) 8-slot ring-buffer parity arithmetic for speculative decode verification rollback.
public enum QWFourSlotRollbackRing {
    public static let ringMask = 7
    public static let maxVerifyRows = 8

    public static func supportsWindow(_ rows: Int) -> Bool {
        rows >= 1 && rows <= maxVerifyRows
    }

    @inline(__always)
    public static func rollback(_ parity: Int, by rejectedCount: Int) -> Int {
        (parity - rejectedCount) & ringMask
    }
}

/// Load-time command-buffer cut and PLE staging schedule for compiled layer stacks.
public enum QWLayerStackSchedule {
    public enum Step: Equatable, Sendable {
        case waitForPLE
        case commit
    }

    public static func decodeCuts(slotCount: Int, hasPLE: Bool = true) -> [Int] {
        (hasPLE && slotCount > 8) ? [0, 1, 8, slotCount] : [0, slotCount]
    }

    public static func stepOrder(chunkHasPLE: [Bool]) -> [[Step]] {
        var waited = false
        return chunkHasPLE.map { hasPLE in
            var steps: [Step] = []
            if hasPLE, !waited {
                steps.append(.waitForPLE)
                waited = true
            }
            steps.append(.commit)
            return steps
        }
    }

    public static func prefillChunkEnd(start: Int, count: Int, layersPerCB: Int, firstPLE: Int?) -> Int {
        let span = min(layersPerCB, max(1, start))
        let end = min(start + span, count)
        if let firstPLE, start < firstPLE && firstPLE < end { return firstPLE }
        return end
    }
}
