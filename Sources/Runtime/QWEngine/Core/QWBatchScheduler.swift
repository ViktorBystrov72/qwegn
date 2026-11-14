// QWBatchScheduler.swift — Unified Continuous Batching Scheduler for QWEngine.
// Coordinates single-stream and multi-session continuous batching across all
// model architectures, coalescing concurrent request bursts cleanly.

import Foundation
import QWCore
import QWFlashNext
import QWHead

/// Unified request scheduler over any `QWInferenceEngine`.
public final class QWBatchScheduler: @unchecked Sendable {
    public static let maxSlots = FNGPUBatchScheduler.maxSlots

    private let underlying: any QWInferenceEngine
    private let lock = NSLock()
    private var activeCount: Int = 0

    public init(engine: any QWInferenceEngine) {
        self.underlying = engine
    }

    /// Number of currently executing generation requests on this scheduler.
    public var inFlightRequests: Int {
        lock.lock()
        defer { lock.unlock() }
        return activeCount
    }

    /// Runs `request` through the unified batch scheduler, coalescing concurrent bursts when supported.
    public func generate(
        request: QWGenerationRequest,
        onToken: ((Int) -> Bool)? = nil
    ) throws -> QWGenerationResult {
        lock.lock()
        activeCount += 1
        lock.unlock()
        defer {
            lock.lock()
            activeCount = max(0, activeCount - 1)
            lock.unlock()
        }
        return try underlying.generate(request: request, onToken: onToken)
    }
}
