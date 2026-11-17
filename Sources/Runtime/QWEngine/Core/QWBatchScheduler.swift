// QWBatchScheduler.swift — Unified Continuous Batching Scheduler for QWEngine.
// Coordinates single-stream passthrough and multi-session continuous batching
// across QWEngineDriver modules, coalescing concurrent bursts cleanly.

import Foundation
import QWCore
import QWFlashNext
import QWHead

/// Unified request scheduler over any `QWInferenceEngine`.
public final class QWBatchScheduler: @unchecked Sendable {
    public static let maxSlots = FNGPUBatchScheduler.maxSlots

    private let underlying: any QWInferenceEngine
    private let driver: QWEngineDriver?
    private let lock = NSLock()
    private let driverWorker = QWDriverWorkerThread()
    private var activeCount: Int = 0
    private var pendingQueue: [QWBatchScheduledJob] = []
    private var isDriving: Bool = false
    private var cachedHeadScratch: QWEngineDriver.BatchScratchBox?
    private var recordedMaxWaveSize: Int = 0

    public init(engine: any QWInferenceEngine) {
        self.underlying = engine
        self.driver = Self.resolveDriver(engine)
    }

    deinit {
        driverWorker.shutdown()
    }

    static func resolveDriver(_ engine: any QWInferenceEngine) -> QWEngineDriver? {
        var current: any QWInferenceEngine = engine
        while let unified = current as? QWEngine {
            current = unified.underlying
        }
        if let direct = current as? QWEngineDriver {
            return direct
        }
        if let dense = current as? DenseModule {
            return dense.engine
        }
        return nil
    }

    /// Number of currently executing generation requests on this scheduler.
    public var inFlightRequests: Int {
        lock.withLock { activeCount }
    }

    var peakCoalescedWaveSize: Int {
        lock.withLock { recordedMaxWaveSize }
    }

    /// Runs `request` through single-stream fast path or batched continuous scheduling.
    public func generate(
        request: QWGenerationRequest,
        onToken: ((Int) -> Bool)? = nil
    ) throws -> QWGenerationResult {
        lock.withLock { activeCount += 1 }
        defer { lock.withLock { activeCount = max(0, activeCount - 1) } }

        guard let driver else {
            return try underlying.generate(request: request, onToken: onToken)
        }

        let job = QWBatchScheduledJob(request: request, onToken: onToken)
        let becameLeader = lock.withLock { () -> Bool in
            pendingQueue.append(job)
            guard !isDriving else { return false }
            isDriving = true
            return true
        }
        guard becameLeader else {
            return try job.wait()
        }

        coalescePreparingBurst(maxSlots: min(Self.maxSlots, driver.maxBatchSlots))
        enum FastDecision {
            case solo
            case dispatchBatch
        }
        let decision = lock.withLock { () -> FastDecision in
            if pendingQueue.count == 1 && pendingQueue[0] === job {
                pendingQueue.removeFirst()
                recordedMaxWaveSize = max(recordedMaxWaveSize, 1)
                return .solo
            }
            return .dispatchBatch
        }
        if decision == .solo {
            defer {
                let needsDrive = lock.withLock { () -> Bool in
                    if pendingQueue.isEmpty {
                        isDriving = false
                        return false
                    }
                    return true
                }
                if needsDrive {
                    driverWorker.async { [weak self] in self?.driveLoop() }
                }
            }
            return try underlying.generate(request: request, onToken: onToken)
        }

        driverWorker.async { [weak self] in self?.driveLoop() }
        return try job.wait()
    }

    private func coalescePreparingBurst(maxSlots: Int) {
        guard FNBatchAdmissionTracker.preparingCount > 0 else { return }
        var waitedUs = 0
        while waitedUs < 10_000 {
            let queued = lock.withLock { pendingQueue.count }
            if queued >= maxSlots { break }
            if FNBatchAdmissionTracker.preparingCount == 0 && waitedUs >= 1_000 { break }
            usleep(250)
            waitedUs += 250
        }
    }

    private func driveLoop() {
        while true {
            let maxWave = driver.map { min(Self.maxSlots, $0.maxBatchSlots) } ?? 1
            if FNBatchAdmissionTracker.preparingCount > 0 {
                coalescePreparingBurst(maxSlots: maxWave)
            }
            enum Wave {
                case idle
                case solo(QWBatchScheduledJob)
                case batch([QWBatchScheduledJob])
            }
            let wave = lock.withLock { () -> Wave in
                if pendingQueue.isEmpty {
                    isDriving = false
                    return .idle
                }
                if pendingQueue.count == 1 {
                    recordedMaxWaveSize = max(recordedMaxWaveSize, 1)
                    return .solo(pendingQueue.removeFirst())
                }
                let take = min(maxWave, pendingQueue.count)
                let slice = Array(pendingQueue.prefix(take))
                pendingQueue.removeFirst(take)
                recordedMaxWaveSize = max(recordedMaxWaveSize, slice.count)
                return .batch(slice)
            }

            switch wave {
            case .idle:
                return
            case .solo(let solo):
                do {
                    let result = try underlying.generate(request: solo.request, onToken: solo.onToken)
                    solo.complete(.success(result))
                } catch {
                    solo.complete(.failure(error))
                }
            case .batch(let jobs):
                guard let driver else {
                    for job in jobs {
                        job.complete(.failure(QWEngineDriverError.state("batch scheduler driver missing")))
                    }
                    continue
                }
                let scratch = ensureHeadScratch(for: driver)
                driver.runScheduledBatch(
                    initialJobs: jobs,
                    headScratch: scratch,
                    pollPending: { freeSlots in
                        guard freeSlots > 0 else { return [] }
                        return self.lock.withLock {
                            let cap = min(freeSlots, maxWave)
                            let n = min(cap, self.pendingQueue.count)
                            guard n > 0 else { return [] }
                            let polled = Array(self.pendingQueue.prefix(n))
                            self.pendingQueue.removeFirst(n)
                            return polled
                        }
                    },
                    requeueDelayed: { delayed in
                        guard !delayed.isEmpty else { return }
                        self.lock.withLock {
                            self.pendingQueue.insert(contentsOf: delayed, at: 0)
                        }
                    }
                )
            }
        }
    }

    private func ensureHeadScratch(for driver: QWEngineDriver) -> QWBatchHeadScratch? {
        if let box = cachedHeadScratch { return box.scratch }
        guard let scratch = try? QWBatchHeadScratch(hal: driver.hal, vocab: driver.geometry.vocab) else {
            return nil
        }
        cachedHeadScratch = QWEngineDriver.BatchScratchBox(scratch: scratch)
        return scratch
    }
}

extension QWEngineDriver {
    final class BatchScratchBox {
        let scratch: QWBatchHeadScratch
        init(scratch: QWBatchHeadScratch) { self.scratch = scratch }
    }
}

private final class QWDriverWorkerThread: @unchecked Sendable {
    private let condition = NSCondition()
    private var queue: [@Sendable () -> Void] = []
    private var started = false
    private var stopped = false

    func async(_ block: @escaping @Sendable () -> Void) {
        condition.lock()
        if stopped {
            condition.unlock()
            return
        }
        queue.append(block)
        let needStart = !started
        if needStart { started = true }
        condition.signal()
        condition.unlock()
        if needStart {
            let thread = Thread { self.runLoop() }
            thread.name = "qwegn.qwengine.batch.driver"
            thread.qualityOfService = .userInitiated
            thread.stackSize = 8 * 1024 * 1024
            thread.start()
        }
    }

    func shutdown() {
        condition.lock()
        stopped = true
        condition.broadcast()
        condition.unlock()
    }

    private func runLoop() {
        while true {
            condition.lock()
            while queue.isEmpty && !stopped {
                condition.wait()
            }
            if queue.isEmpty && stopped {
                condition.unlock()
                return
            }
            let work = queue.removeFirst()
            condition.unlock()
            autoreleasepool {
                work()
            }
        }
    }
}
