// QWServerMetrics.swift — Thread-safe telemetry registry for inference & cache state.

import Foundation
import Darwin
import Synchronization
import QWEngine

public final class QWEngineTelemetryRegistry: @unchecked Sendable {
    public static let shared = QWEngineTelemetryRegistry()

    let lock = NSLock()
    private let liveDecodeCountAtomic = Atomic<Int>(0)
    private let liveDecodeLastNsAtomic = Atomic<Int>(0)
    private var lastDrainedLiveCount: Int = 0
    private var lastDrainedLiveNs: Int = 0

    // Cache Counters
    /// The cache bytes the engine reported. Nil means the engine cannot
    /// measure them: absent, never zero.
    var warmCacheBytes: Int?
    var warmCacheLimitBytes: Int = 0
    var ssdCacheBytes: Int?
    var ssdCacheLimitBytes: Int = 0
    var cacheLookupsTotal: Int = 0
    var cacheHitsWarmTotal: Int = 0
    var cacheHitsSSDTotal: Int = 0
    var cacheHitsUnclassifiedTotal: Int = 0
    var cacheMissesTotal: Int = 0
    var cacheTokensReusedTotal: Int = 0
    var cacheEvictionsWarmTotal: Int = 0
    var cacheEvictionsSSDTotal: Int = 0
    /// When the cache was last written.
    var cacheSampledAt: Date?

    // Token Throughput & Phase Trackers
    var phase: EngineInferencePhase = .idle
    /// The decode rate the engine's own inter-token latencies give:
    /// 1000 / p50(ITL) over the last `itlWindow` tokens, coalescing burst emissions
    /// into their round's per-token latency in a bounded FIFO window.
    var itlDecodeTokPerSec: Double = 0.0
    /// FIFO ring buffer of the last `itlWindow` per-token latencies in milliseconds.
    var itlFifo: [Double] = []
    var itlFifoHead: Int = 0
    var currentRoundMs: Double = 0.0
    var currentRoundTokens: Int = 0
    static let speculativeBurstThresholdMs: Double = 0.5
    /// The last `itlWindow` inter-token latencies in milliseconds, kept sorted so the
    /// median needs no copy.
    var itlSorted: [Double] = []
    /// When the last token arrived, for the next inter-token latency.
    var lastTokenAt: Date?

    var tokensInTotal: Int = 0
    var tokensOutTotal: Int = 0
    var lastPrefillTokPerSec: Double = 0
    var lastDecodeTokPerSec: Double = 0
    /// When the last prefill rate was measured.
    var lastPrefillSampledAt: Date?
    /// The published trace of the two rates. A renderer draws this; it
    /// does not keep its own.
    var prefillRateSeries: [Double?] = []
    var decodeRateSeries: [Double?] = []
    /// When the last decode rate was measured.
    var lastDecodeSampledAt: Date?
    var ema1mPrefillTokPerSec: Double = 0
    var ema1mDecodeTokPerSec: Double = 0

    // Live chunked prefill rate and 1-second trace sampling state.
    private var livePrefillTokPerSec: Double = 0.0
    private var livePrefillSampledAt: Date?
    private var prefillChunkAnchorTokens: Int = 0
    private var prefillChunkAnchorAt: Date?
    private var nextTraceTickAt: Date?
    private var seriesTotalAppended: Int = 0
    private var turnStartSeriesSeq: Int = 0
    private var turnPrefillSamplesCommitted: Int = 0
    private var turnDecodeSamplesCommitted: Int = 0
    private static let traceWindowCapacity = 60
    private static let traceTickIntervalSec: TimeInterval = 1.0
    private static let minLivePrefillWindowSec: TimeInterval = 0.05

    // Per-turn prefill progress: the tokens the engine must compute this turn,
    // excluding tokens served from the prefix cache.
    var prefillComputedDone = 0
    var prefillComputedTotal = 0
    /// Whether the last completed turn's prompt came entirely from the
    /// prefix cache. Such a turn ran no prefill phase at all.
    var lastTurnPrefillFromCache = false

    // Cumulative counters for the idle-time overall averages.
    var totalPrefillComputedTokens: Double = 0
    var totalPrefillSeconds: Double = 0
    var totalDecodeTokens: Double = 0
    var totalDecodeSeconds: Double = 0
    /// When the last completed turn finished; the averages describe the work
    /// done up to then.
    var lastTurnSampledAt: Date?

    // Latency Trackers
    var lastTtftMs: Double = 0
    var totalTtftMs: Double = 0
    var totalRequestsWithTtft: Int = 0
    var lastTtftSampledAt: Date?

    // Context & Memory
    /// The tokens the last turn used. Nil means no turn has run: absent,
    /// never zero.
    var contextTokensActive: Int?
    var contextTokensMax: Int = 4096
    var contextSampledAt: Date?
    var arenaReservedBytes: Int = 0

    /// How long a rate stays current. Past this the display says the
    /// measurement is stale instead of presenting it as the engine's speed.
    let rateStaleAfter: TimeInterval

    public init(rateStaleAfter: TimeInterval = QWTelemetryFreshness.rateStaleAfter) {
        self.rateStaleAfter = rateStaleAfter
        itlFifo.reserveCapacity(Self.itlWindow)
        itlSorted.reserveCapacity(Self.itlWindow)
    }

    public func recordCacheLookup(hitTier: String?, matchedTokens: Int) {
        lock.lock()
        defer { lock.unlock() }
        cacheLookupsTotal += 1
        guard let tier = hitTier else {
            cacheMissesTotal += 1
            return
        }
        // A tier this build does not name is still a hit: the cache
        // served the prompt. It is counted as a hit of an unnamed
        // tier — never as a miss, which would be a claim the
        // measurement does not support.
        if tier == "warm_ram" {
            cacheHitsWarmTotal += 1
        } else if tier == "cold_ssd" || tier == "ssd" {
            cacheHitsSSDTotal += 1
        } else {
            cacheHitsUnclassifiedTotal += 1
        }
        cacheTokensReusedTotal += matchedTokens
    }

    public func updateCacheUsage(warmBytes: Int?, warmLimit: Int, ssdBytes: Int?, ssdLimit: Int) {
        lock.lock()
        defer { lock.unlock() }
        self.warmCacheBytes = warmBytes
        self.warmCacheLimitBytes = warmLimit
        self.ssdCacheBytes = ssdBytes
        self.ssdCacheLimitBytes = ssdLimit
        // When the cache was last measured: a card that shows a size must
        // say how old the size is. A report with no bytes in it measured
        // nothing, so there is no moment to remember.
        self.cacheSampledAt = (warmBytes == nil && ssdBytes == nil) ? nil : Date()
    }

    // MARK: - Live Phase Transitions

    public func beginPrefill(promptTokens: Int, at date: Date? = nil) {
        lock.withLock {
            let now = date ?? Date()
            phase = .prefill
            itlDecodeTokPerSec = 0.0
            itlFifo.removeAll(keepingCapacity: true)
            itlFifoHead = 0
            currentRoundMs = 0.0
            currentRoundTokens = 0
            itlSorted.removeAll(keepingCapacity: true)
            lastTokenAt = nil
            prefillComputedDone = 0
            prefillComputedTotal = promptTokens
            livePrefillTokPerSec = 0.0
            livePrefillSampledAt = nil
            prefillChunkAnchorTokens = 0
            prefillChunkAnchorAt = now
            nextTraceTickAt = now.addingTimeInterval(Self.traceTickIntervalSec)
            turnStartSeriesSeq = seriesTotalAppended
            turnPrefillSamplesCommitted = 0
            turnDecodeSamplesCommitted = 0
        }
        if self === Self.shared {
            QWFanController.shared.updateInferencePhase("prefill")
        }
    }

    /// Updates incremental prefill progress for the turn currently in prefill.
    public func updatePrefillProgress(completedTokens: Int, totalTokens: Int, at date: Date? = nil) {
        lock.lock()
        defer { lock.unlock() }
        guard phase == .prefill else { return }
        let now = date ?? Date()
        let safeTotal = max(0, totalTokens)
        prefillComputedTotal = safeTotal
        prefillComputedDone = min(safeTotal, max(0, completedTokens))
        if prefillComputedDone == 0 {
            prefillChunkAnchorTokens = 0
            prefillChunkAnchorAt = now
            nextTraceTickAt = now.addingTimeInterval(Self.traceTickIntervalSec)
        } else if prefillComputedDone > prefillChunkAnchorTokens, let anchorAt = prefillChunkAnchorAt {
            let elapsed = now.timeIntervalSince(anchorAt)
            if elapsed >= Self.minLivePrefillWindowSec {
                livePrefillTokPerSec = Double(prefillComputedDone - prefillChunkAnchorTokens) / elapsed
                livePrefillSampledAt = now
                prefillChunkAnchorTokens = prefillComputedDone
                prefillChunkAnchorAt = now
                advanceActiveTraceIfNeeded(now: now)
            }
        }
    }

    /// The decode phase has begun. The prefill's own duration is not known
    /// here: the time to the first token is not the prefill's duration, and
    /// a fully cached prompt has no prefill phase at all.
    public func beginDecode(promptTokens: Int, at date: Date? = nil) {
        lock.withLock {
            let now = date ?? Date()
            if phase == .prefill && livePrefillTokPerSec > 0 {
                advanceActiveTraceIfNeeded(now: now)
            } else if phase == .idle {
                turnStartSeriesSeq = seriesTotalAppended
                turnPrefillSamplesCommitted = 0
                turnDecodeSamplesCommitted = 0
            }
            phase = .decode
            prefillComputedDone = promptTokens
            prefillComputedTotal = promptTokens
            livePrefillTokPerSec = 0.0
            livePrefillSampledAt = nil
            prefillChunkAnchorTokens = 0
            prefillChunkAnchorAt = nil
            itlDecodeTokPerSec = 0.0
            itlFifo.removeAll(keepingCapacity: true)
            itlFifoHead = 0
            currentRoundMs = 0.0
            currentRoundTokens = 0
            itlSorted.removeAll(keepingCapacity: true)
            // The first token has arrived, so the gap to the next one is a
            // measurement: the rate exists once the turn has two tokens.
            lastTokenAt = now
            let monoNow = Self.monotonicNsNow()
            liveDecodeCountAtomic.store(0, ordering: .releasing)
            liveDecodeLastNsAtomic.store(monoNow, ordering: .releasing)
            lastDrainedLiveCount = 0
            lastDrainedLiveNs = monoNow
            nextTraceTickAt = now.addingTimeInterval(Self.traceTickIntervalSec)
        }
        if self === Self.shared {
            QWFanController.shared.updateInferencePhase("decode")
        }
    }

    private static func monotonicNsNow() -> Int {
        Int(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) & 0x7FFF_FFFF_FFFF_FFFF)
    }

    private func flushLiveDecodeAtomicsLocked(now: Date) {
        let count = liveDecodeCountAtomic.load(ordering: .acquiring)
        let lastNs = liveDecodeLastNsAtomic.load(ordering: .acquiring)
        let deltaTokens = count - lastDrainedLiveCount
        let deltaNs = lastNs - lastDrainedLiveNs
        if deltaTokens > 0 && deltaNs > 0 {
            let perTokenMs = (Double(deltaNs) / 1_000_000.0) / Double(deltaTokens)
            itlDecodeTokPerSec = 1000.0 / perTokenMs
            lastDecodeSampledAt = now
            lastDrainedLiveCount = count
            lastDrainedLiveNs = lastNs
        }
    }

    /// `at` exists for tests only: macOS Thread.sleep overshoots by 3-5x under
    /// scheduler load, so timing behavior is verified with injected timestamps
    /// instead of wall-clock sleeps. Production always passes nil for lock-free updates.
    public func recordDecodeToken(at date: Date? = nil) {
        guard let date else {
            let nowNs = Self.monotonicNsNow()
            liveDecodeLastNsAtomic.store(nowNs, ordering: .releasing)
            liveDecodeCountAtomic.wrappingAdd(1, ordering: .releasing)
            return
        }
        lock.lock()
        defer { lock.unlock() }
        let now = date
        if let last = lastTokenAt {
            let latency = now.timeIntervalSince(last) * 1000.0
            if latency > 0 {
                let perTokenMs: Double
                if latency < Self.speculativeBurstThresholdMs && currentRoundTokens > 0 && !itlFifo.isEmpty {
                    currentRoundMs += latency
                    currentRoundTokens += 1
                    perTokenMs = currentRoundMs / Double(currentRoundTokens)
                    let updateCount = min(currentRoundTokens - 1, itlFifo.count)
                    for offset in 1...updateCount {
                        let idx = (itlFifoHead - offset + itlFifo.count) % itlFifo.count
                        let oldVal = itlFifo[idx]
                        itlFifo[idx] = perTokenMs
                        Self.insertSorted(perTokenMs, into: &itlSorted, capacity: Self.itlWindow, evicting: oldVal)
                    }
                } else {
                    currentRoundMs = latency
                    currentRoundTokens = 1
                    perTokenMs = latency
                }
                if itlFifo.count < Self.itlWindow {
                    itlFifo.append(perTokenMs)
                    itlFifoHead = itlFifo.count % Self.itlWindow
                    Self.insertSorted(perTokenMs, into: &itlSorted, capacity: Self.itlWindow)
                } else {
                    let evicted = itlFifo[itlFifoHead]
                    itlFifo[itlFifoHead] = perTokenMs
                    itlFifoHead = (itlFifoHead + 1) % Self.itlWindow
                    Self.insertSorted(perTokenMs, into: &itlSorted, capacity: Self.itlWindow, evicting: evicted)
                }
                if !itlSorted.isEmpty {
                    itlDecodeTokPerSec = 1000.0 / itlSorted[itlSorted.count / 2]
                }
            }
        }
        lastTokenAt = now
        if itlDecodeTokPerSec > 0 {
            lastDecodeSampledAt = now
        }
        advanceActiveTraceIfNeeded(now: now)
    }

    /// How many inter-token latencies the live rate looks at.
    public static let itlWindow = 32

    static func removeSorted(_ value: Double, from samples: inout [Double]) {
        guard !samples.isEmpty else { return }
        var low = 0
        var high = samples.count
        while low < high {
            let mid = (low + high) / 2
            if samples[mid] < value { low = mid + 1 } else { high = mid }
        }
        if low < samples.count {
            samples.remove(at: low)
        } else {
            samples.removeLast()
        }
    }

    /// Inserts a sample keeping the array sorted, and drops the oldest when
    /// the window is full. The buffer never grows past its reserved
    /// capacity, so this allocates nothing in the steady state.
    static func insertSorted(_ value: Double, into samples: inout [Double], capacity: Int, evicting evicted: Double? = nil) {
        if let evicted {
            removeSorted(evicted, from: &samples)
        } else if samples.count >= capacity {
            samples.removeFirst()
        }
        var low = 0
        var high = samples.count
        while low < high {
            let mid = (low + high) / 2
            if samples[mid] <= value { low = mid + 1 } else { high = mid }
        }
        samples.insert(value, at: low)
    }

    public func endRequest(tokensIn: Int, tokensOut: Int, cachedTokens: Int = 0,
                           prefillSec: Double?, decodeSec: Double?, ttftSec: Double?,
                           at date: Date? = nil) {
        lock.withLock {
            let now = date ?? Date()
            flushLiveDecodeAtomicsLocked(now: now)
            if phase == .prefill || phase == .decode {
                advanceActiveTraceIfNeeded(now: now)
            }
            let hadLiveDecodeTail = phase == .decode && itlDecodeTokPerSec > 0
            let trailingDecodeRate = itlDecodeTokPerSec
            let committedPrefill = turnPrefillSamplesCommitted
            let committedDecode = turnDecodeSamplesCommitted
            let startSeq = turnStartSeriesSeq

            phase = .idle
            itlDecodeTokPerSec = 0.0
            itlFifo.removeAll(keepingCapacity: true)
            itlFifoHead = 0
            currentRoundMs = 0.0
            currentRoundTokens = 0
            itlSorted.removeAll(keepingCapacity: true)
            lastTokenAt = nil
            prefillComputedDone = 0
            prefillComputedTotal = 0
            livePrefillTokPerSec = 0.0
            livePrefillSampledAt = nil
            prefillChunkAnchorTokens = 0
            prefillChunkAnchorAt = nil
            nextTraceTickAt = nil
            turnPrefillSamplesCommitted = 0
            turnDecodeSamplesCommitted = 0

            tokensInTotal += tokensIn
            tokensOutTotal += tokensOut

            // Prefill speed is computed tokens / time; cache-served tokens cost nothing.
            let computed = max(0, tokensIn - cachedTokens)
            lastTurnPrefillFromCache = tokensIn > 0 && computed == 0
            var instPrefill: Double? = nil
            if let prefillSec, prefillSec > 0 && computed > 0 {
                let rate = Double(computed) / prefillSec
                instPrefill = rate
                lastPrefillTokPerSec = rate
                totalPrefillComputedTokens += Double(computed)
                totalPrefillSeconds += prefillSec
                ema1mPrefillTokPerSec = ema1mPrefillTokPerSec == 0 ? rate : (0.2 * rate + 0.8 * ema1mPrefillTokPerSec)
                lastPrefillSampledAt = now
            }
            // The first generated token comes from prefill, so decode counts remaining tokens.
            let decoded = tokensOut > 1 ? tokensOut - 1 : tokensOut
            var instDecode: Double? = nil
            if let decodeSec, decodeSec > 0 && decoded > 0 {
                let rate = Double(decoded) / decodeSec
                instDecode = rate
                lastDecodeTokPerSec = rate
                totalDecodeTokens += Double(decoded)
                totalDecodeSeconds += decodeSec
                ema1mDecodeTokPerSec = ema1mDecodeTokPerSec == 0 ? rate : (0.2 * rate + 0.8 * ema1mDecodeTokPerSec)
                lastDecodeSampledAt = now
            }

            if committedPrefill == 0 && committedDecode == 0 {
                appendTraceSample(prefill: instPrefill, decode: instDecode)
            } else {
                if hadLiveDecodeTail || (committedDecode == 0 && instDecode != nil) {
                    appendTraceSample(prefill: nil, decode: instDecode ?? (hadLiveDecodeTail ? trailingDecodeRate : nil))
                }
                if committedPrefill == 0, let instPrefill {
                    let ageSlots = seriesTotalAppended - startSeq
                    if ageSlots >= 1 && ageSlots <= prefillRateSeries.count {
                        prefillRateSeries[prefillRateSeries.count - ageSlots] = instPrefill
                    }
                }
            }
            // The time to the first token, when the engine reported one.
            if let ttftSec {
                let ttft = ttftSec * 1000.0
                lastTtftMs = ttft
                totalTtftMs += ttft
                totalRequestsWithTtft += 1
                lastTtftSampledAt = now
            }
            lastTurnSampledAt = now
        }
        if self === Self.shared {
            QWFanController.shared.updateInferencePhase("idle")
        }
    }

    /// Appends paired trace samples to the rolling 60-second series while advancing sequence count.
    private func appendTraceSample(prefill: Double?, decode: Double?) {
        Self.append(prefill, to: &prefillRateSeries)
        Self.append(decode, to: &decodeRateSeries)
        seriesTotalAppended += 1
    }

    /// Advances active prefill/decode trace slots once per elapsed second without drifting when idle.
    private func advanceActiveTraceIfNeeded(now: Date) {
        guard var tickAt = nextTraceTickAt else { return }
        if now.timeIntervalSince(tickAt) > Double(Self.traceWindowCapacity) {
            tickAt = now.addingTimeInterval(-Double(Self.traceWindowCapacity - 1))
        }
        switch phase {
        case .prefill:
            guard livePrefillTokPerSec > 0 else { return }
            while now >= tickAt {
                appendTraceSample(prefill: livePrefillTokPerSec, decode: nil)
                turnPrefillSamplesCommitted += 1
                tickAt = tickAt.addingTimeInterval(Self.traceTickIntervalSec)
            }
            nextTraceTickAt = tickAt
        case .decode:
            guard itlDecodeTokPerSec > 0 else { return }
            while now >= tickAt {
                appendTraceSample(prefill: nil, decode: itlDecodeTokPerSec)
                turnDecodeSamplesCommitted += 1
                tickAt = tickAt.addingTimeInterval(Self.traceTickIntervalSec)
            }
            nextTraceTickAt = tickAt
        case .idle:
            break
        }
    }
    /// Keeps a bounded trace: a gap is a nil, never a zero.
    static func append(_ value: Double?, to series: inout [Double?]) {
        series.append(value)
        if series.count > traceWindowCapacity { series.removeFirst(series.count - traceWindowCapacity) }
    }

    public func resetToIdle(resetCacheCounters: Bool = false) {
        lock.withLock {
            phase = .idle
            // Preserve lifetime cache hit/lookup counters across per-request completion/errors;
            // only clear them explicitly when releasing/reloading the resident model.
            if resetCacheCounters {
                cacheLookupsTotal = 0
                cacheHitsWarmTotal = 0
                cacheHitsSSDTotal = 0
                cacheHitsUnclassifiedTotal = 0
                cacheMissesTotal = 0
                cacheTokensReusedTotal = 0
                cacheEvictionsWarmTotal = 0
                cacheEvictionsSSDTotal = 0
            }
            liveDecodeCountAtomic.store(0, ordering: .releasing)
            liveDecodeLastNsAtomic.store(0, ordering: .releasing)
            lastDrainedLiveCount = 0
            lastDrainedLiveNs = 0
            itlDecodeTokPerSec = 0.0
            itlFifo.removeAll(keepingCapacity: true)
            itlFifoHead = 0
            currentRoundMs = 0.0
            currentRoundTokens = 0
            itlSorted.removeAll(keepingCapacity: true)
            lastTokenAt = nil
            prefillComputedDone = 0
            prefillComputedTotal = 0
            livePrefillTokPerSec = 0.0
            livePrefillSampledAt = nil
            prefillChunkAnchorTokens = 0
            prefillChunkAnchorAt = nil
            nextTraceTickAt = nil
            seriesTotalAppended = 0
            turnStartSeriesSeq = 0
            turnPrefillSamplesCommitted = 0
            turnDecodeSamplesCommitted = 0
            // Clear trace series on reset; last rate stays with its measurement timestamp.
            prefillRateSeries.removeAll()
            decodeRateSeries.removeAll()
            // The state belongs to the model that ran the turn: a new model
            // starts with its own.
            lastTurnPrefillFromCache = false
        }
        if self === Self.shared {
            QWFanController.shared.updateInferencePhase("idle")
        }
    }

    public func updateContext(active: Int?, capacity: Int) {
        lock.lock()
        defer { lock.unlock() }
        self.contextTokensActive = active
        self.contextTokensMax = capacity
        // A session with no turn in it measured nothing: there is no
        // moment to remember.
        self.contextSampledAt = active == nil ? nil : Date()
    }

    public func updateArena(reserved: Int) {
        lock.lock()
        defer { lock.unlock() }
        self.arenaReservedBytes = reserved
    }

    public static func getResidentMemoryBytes() -> Int {
        var vmInfo = task_vm_info_data_t()
        var vmCount = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let vmErr = withUnsafeMutablePointer(to: &vmInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(vmCount)) {
                task_info(mach_task_self_, mach_task_flavor_t(TASK_VM_INFO), $0, &vmCount)
            }
        }
        if vmErr == KERN_SUCCESS && vmInfo.phys_footprint > 0 {
            return Int(vmInfo.phys_footprint)
        }
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, mach_task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return kerr == KERN_SUCCESS ? Int(info.resident_size) : 0
    }

    public func snapshot(at date: Date? = nil) -> MetricsSnapshot {
        let hw = QWSystemMetricsSampler.shared.current()
        let busy = QWTelemetryRegistry.shared.value("gpu.busy_ratio")
        lock.lock()
        defer { lock.unlock() }
        let now = date ?? Date()
        flushLiveDecodeAtomicsLocked(now: now)
        advanceActiveTraceIfNeeded(now: now)

        let livePrefill: Double? = phase == .prefill && livePrefillTokPerSec > 0 ? livePrefillTokPerSec : nil
        let liveDecode: Double? = phase == .decode && itlDecodeTokPerSec > 0 ? itlDecodeTokPerSec : nil
        let lastPrefill: Double? = lastPrefillTokPerSec > 0 ? lastPrefillTokPerSec : nil
        let lastDecode: Double? = lastDecodeTokPerSec > 0 ? lastDecodeTokPerSec : nil
        let emaPrefill: Double? = ema1mPrefillTokPerSec > 0 ? ema1mPrefillTokPerSec : nil
        let emaDecode: Double? = ema1mDecodeTokPerSec > 0 ? ema1mDecodeTokPerSec : nil
        let avgPrefill: Double? = totalPrefillSeconds > 0 ? totalPrefillComputedTokens / totalPrefillSeconds : nil
        let avgDecode: Double? = totalDecodeSeconds > 0 ? totalDecodeTokens / totalDecodeSeconds : nil

        // What the UI shows: live rate while running, overall average when idle, or last/stale rate.
        let (displayPrefill, prefillMode) = Self.displayRate(
            live: livePrefill, average: phase == .idle ? avgPrefill : nil, last: lastPrefill,
            sampledAt: lastPrefillSampledAt, now: now, staleAfter: rateStaleAfter)
        let (displayDecode, decodeMode) = Self.displayRate(
            live: liveDecode, average: phase == .idle ? avgDecode : nil, last: lastDecode,
            sampledAt: lastDecodeSampledAt, now: now, staleAfter: rateStaleAfter)

        let hwAge = now.timeIntervalSince(hw.sampledAt)
        let hwStale = hwAge > QWTelemetryFreshness.hardwareStaleAfter
        let busyRatio = busy?.value
        let busyStale = busy.map { $0.isStale(now: now, bound: QWTelemetryFreshness.busyRatioStaleAfter) } ?? false
        let pubPrefill: [Double?]
        let pubDecode: [Double?]
        if let livePrefill {
            pubPrefill = Array((prefillRateSeries + [livePrefill]).suffix(Self.traceWindowCapacity))
            pubDecode = Array((decodeRateSeries + [nil]).suffix(Self.traceWindowCapacity))
        } else if let liveDecode {
            pubPrefill = Array((prefillRateSeries + [nil]).suffix(Self.traceWindowCapacity))
            pubDecode = Array((decodeRateSeries + [liveDecode]).suffix(Self.traceWindowCapacity))
        } else {
            pubPrefill = prefillRateSeries
            pubDecode = decodeRateSeries
        }

        return MetricsSnapshot(
            cache: .init(
                warmBytes: warmCacheBytes, warmLimitBytes: warmCacheLimitBytes,
                ssdBytes: ssdCacheBytes, ssdLimitBytes: ssdCacheLimitBytes,
                lookupsTotal: cacheLookupsTotal, hitsWarmTotal: cacheHitsWarmTotal,
                hitsSSDTotal: cacheHitsSSDTotal, hitsUnclassifiedTotal: cacheHitsUnclassifiedTotal,
                missesTotal: cacheMissesTotal, tokensReusedTotal: cacheTokensReusedTotal,
                evictionsWarmTotal: cacheEvictionsWarmTotal, evictionsSSDTotal: cacheEvictionsSSDTotal,
                sampledAt: cacheSampledAt),
            throughput: .init(
                phase: phase.rawValue, tokensInTotal: tokensInTotal, tokensOutTotal: tokensOutTotal,
                livePrefillTokensPerSec: livePrefill, liveDecodeTokensPerSec: liveDecode,
                lastPrefillTokensPerSec: lastPrefill, lastDecodeTokensPerSec: lastDecode,
                ema1mPrefillTokensPerSec: emaPrefill, ema1mDecodeTokensPerSec: emaDecode,
                avgPrefillTokensPerSec: avgPrefill, avgDecodeTokensPerSec: avgDecode,
                displayPrefillTokensPerSec: displayPrefill, displayDecodeTokensPerSec: displayDecode,
                prefillDisplayMode: prefillMode, decodeDisplayMode: decodeMode,
                prefillProgressTokens: prefillComputedDone, prefillTotalTokens: prefillComputedTotal,
                prefillRateSeries: pubPrefill, decodeRateSeries: pubDecode,
                prefillFromCache: lastTurnPrefillFromCache),
            latency: .init(
                lastTtftMs: totalRequestsWithTtft > 0 ? lastTtftMs : nil,
                avgTtftMs: totalRequestsWithTtft > 0 ? totalTtftMs / Double(totalRequestsWithTtft) : nil),
            context: .init(activeTokens: contextTokensActive, maxTokens: contextTokensMax, sampledAt: contextSampledAt),
            memory: .init(residentBytes: Self.getResidentMemoryBytes(), arenaReservedBytes: arenaReservedBytes),
            hardware: hw, gpuBusyRatio: busyRatio,
            provenance: Self.provenanceMap(
                livePrefill: livePrefill, liveDecode: liveDecode,
                lastPrefill: lastPrefill, lastDecode: lastDecode,
                emaPrefill: emaPrefill, emaDecode: emaDecode,
                avgPrefill: avgPrefill, avgDecode: avgDecode,
                displayPrefill: displayPrefill, displayDecode: displayDecode,
                prefillMode: prefillMode, decodeMode: decodeMode,
                ttft: totalRequestsWithTtft > 0, hw: hw, hwStale: hwStale,
                busyRatio: busyRatio, busyStale: busyStale,
                prefillFromCache: lastTurnPrefillFromCache),
            sampledAt: Self.sampledAtMap(
                livePrefill: livePrefill, liveDecode: liveDecode,
                lastPrefill: lastPrefill, lastDecode: lastDecode,
                emaPrefill: emaPrefill, emaDecode: emaDecode,
                avgPrefill: avgPrefill, avgDecode: avgDecode,
                displayPrefill: displayPrefill, displayDecode: displayDecode,
                ttft: totalRequestsWithTtft > 0, hw: hw, busy: busy,
                livePrefillAt: livePrefillSampledAt,
                lastPrefillAt: lastPrefillSampledAt, lastDecodeAt: lastDecodeSampledAt,
                lastTurnAt: lastTurnSampledAt, lastTtftAt: lastTtftSampledAt,
                prefillFromCache: lastTurnPrefillFromCache),
            unavailableReasons: Self.reasonMap(
                livePrefill: livePrefill, liveDecode: liveDecode,
                lastPrefill: lastPrefill, lastDecode: lastDecode,
                emaPrefill: emaPrefill, emaDecode: emaDecode,
                avgPrefill: avgPrefill, avgDecode: avgDecode,
                displayPrefill: displayPrefill, displayDecode: displayDecode,
                prefillMode: prefillMode, decodeMode: decodeMode,
                ttft: totalRequestsWithTtft > 0, hw: hw, hwStale: hwStale,
                busyRatio: busyRatio, busyStale: busyStale, busyReason: busy?.reason,
                prefillFromCache: lastTurnPrefillFromCache)
        )
    }

}
