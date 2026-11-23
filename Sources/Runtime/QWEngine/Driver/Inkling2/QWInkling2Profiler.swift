// QWInkling2Profiler.swift — Zero-overhead stage GPU profiler for Inkling-2.
// Gated on QW_PREFILL_PROFILE=1 and QW_DECODE_PROFILE=1; allocates nothing
// and adds zero Metal commands when profiling is disabled.

import Foundation
import Metal
import QWHAL
import os

enum Ink2Stage: String, CaseIterable, Hashable, Sendable {
    case embed
    case loopResCtrlNorm
    case denseAttnProj
    case sconv
    case qkvrKVWrite
    case flashAttn
    case moeRouteBucket
    case moeExperts
    case moeCombine
    case finalHead
    case dsparkIngest
    case dsparkDraft
    case dsparkMarkov

    var index: Int {
        switch self {
        case .embed: return 0
        case .loopResCtrlNorm: return 1
        case .denseAttnProj: return 2
        case .sconv: return 3
        case .qkvrKVWrite: return 4
        case .flashAttn: return 5
        case .moeRouteBucket: return 6
        case .moeExperts: return 7
        case .moeCombine: return 8
        case .finalHead: return 9
        case .dsparkIngest: return 10
        case .dsparkDraft: return 11
        case .dsparkMarkov: return 12
        }
    }
}

enum Ink2ProfileMode: String, Sendable {
    case prefill
    case decode
}

final class QWInkling2Profiler: @unchecked Sendable {
    static let prefillEnabled: Bool =
        ProcessInfo.processInfo.environment["QW_PREFILL_PROFILE"] == "1"
    static let decodeEnabled: Bool =
        ProcessInfo.processInfo.environment["QW_DECODE_PROFILE"] == "1"
    static var isAnyEnabled: Bool { prefillEnabled || decodeEnabled }

    static func isEnabled(for mode: Ink2ProfileMode) -> Bool {
        switch mode {
        case .prefill: return prefillEnabled
        case .decode: return decodeEnabled
        }
    }

    static func makeIfEnabled(
        mode: Ink2ProfileMode,
        device: (any MTLDevice)? = nil,
        segmentCapacity: Int = 2048
    ) -> QWInkling2Profiler? {
        guard isEnabled(for: mode) else { return nil }
        return QWInkling2Profiler(mode: mode, enabled: true, device: device, segmentCapacity: segmentCapacity)
    }

    struct StageToken: Sendable {
        let stage: Ink2Stage
        let startNs: UInt64
    }

    struct StageRow: Sendable {
        let stage: Ink2Stage
        let totalMs: Double
        let percentGPU: Double
        let calls: Int
        let avgMs: Double
    }

    struct Summary: Sendable {
        let mode: Ink2ProfileMode
        let steps: Int
        let tokens: Int
        let totalStageGpuMs: Double
        let totalCBGpuMs: Double
        let cpuHostMs: Double
        let blitCount: Int
        let rows: [StageRow]
    }

    private struct Bucket {
        var totalMs: Double = 0.0
        var calls: Int = 0
    }

    private final class StageProfilerBox: @unchecked Sendable {
        let profiler: QWGPUStageProfiler<Ink2Stage>
        init(_ profiler: QWGPUStageProfiler<Ink2Stage>) { self.profiler = profiler }
    }

    private final class Storage: @unchecked Sendable {
        var buckets: [Bucket]
        var stageBox: StageProfilerBox?
        var activeStage: Ink2Stage?
        var totalCBGpuMs: Double = 0.0
        var cpuHostMs: Double = 0.0
        var blitCount: Int = 0
        var steps: Int = 0
        var tokens: Int = 0

        init(stageBox: StageProfilerBox?) {
            self.buckets = [Bucket](repeating: Bucket(), count: Ink2Stage.allCases.count)
            self.stageBox = stageBox
        }
    }

    let mode: Ink2ProfileMode
    let isEnabled: Bool
    private let lock: OSAllocatedUnfairLock<Storage?>

    init(
        mode: Ink2ProfileMode = .decode,
        enabled: Bool? = nil,
        device: (any MTLDevice)? = nil,
        segmentCapacity: Int = 2048
    ) {
        let resolved = enabled ?? Self.isEnabled(for: mode)
        self.mode = mode
        self.isEnabled = resolved
        guard resolved else {
            self.lock = OSAllocatedUnfairLock(initialState: nil)
            return
        }
        let sp: StageProfilerBox? = if let device,
            QWGPUStageProfiler<Ink2Stage>.isSupported(on: device),
            let raw = try? QWGPUStageProfiler<Ink2Stage>(device: device, segmentCapacity: segmentCapacity) {
            StageProfilerBox(raw)
        } else {
            nil
        }
        self.lock = OSAllocatedUnfairLock(initialState: Storage(stageBox: sp))
    }

    var hasHardwareStageCounters: Bool {
        guard isEnabled else { return false }
        return lock.withLock { $0?.stageBox != nil }
    }

    /// Switches the active compute encoder to a timestamped stage segment when supported.
    func beginStage(
        _ stage: Ink2Stage,
        on cb: any MTLCommandBuffer,
        encoder enc: inout (any MTLComputeCommandEncoder)?,
        hal: QWHal
    ) {
        guard isEnabled else { return }
        let (sameStage, box) = lock.withLock { st -> (Bool, StageProfilerBox?) in
            guard let st else { return (false, nil) }
            return (st.activeStage == stage, st.stageBox)
        }
        if enc != nil && sameStage { return }
        if let box, box.profiler.hasSegmentCapacity {
            enc?.endEncoding()
            enc = nil
            if let next = try? box.profiler.makeEncoder(commandBuffer: cb, stage: stage) {
                hal.touchKeepAlive(next)
                enc = next
                lock.withLock { $0?.activeStage = stage }
                return
            }
        }
        if enc == nil, let plain = cb.makeComputeCommandEncoder() {
            hal.touchKeepAlive(plain)
            enc = plain
        }
        lock.withLock { $0?.activeStage = stage }
    }

    func endActiveStage(encoder enc: inout (any MTLComputeCommandEncoder)?) {
        enc?.endEncoding()
        enc = nil
        guard isEnabled else { return }
        lock.withLock { $0?.activeStage = nil }
    }

    /// Begins a host-timed stage scope (zero cost and returns nil when disabled).
    func begin(_ stage: Ink2Stage) -> StageToken? {
        guard isEnabled else { return nil }
        return StageToken(stage: stage, startNs: DispatchTime.now().uptimeNanoseconds)
    }

    func end(_ token: StageToken?, commandBuffer cb: (any MTLCommandBuffer)? = nil, countHost: Bool = false) {
        guard isEnabled, let token else { return }
        if countHost {
            let now = DispatchTime.now().uptimeNanoseconds
            if now >= token.startNs {
                recordHostOverhead(ms: Double(now - token.startNs) * 1e-6)
            }
        }
        if let cb {
            record(stage: token.stage, commandBuffer: cb)
        }
    }

    func record(stage: Ink2Stage, cbDurationMs: Double) {
        guard isEnabled else { return }
        let clamped = max(0.0, cbDurationMs)
        lock.withLock { st in
            guard let st else { return }
            let idx = stage.index
            st.buckets[idx].totalMs += clamped
            st.buckets[idx].calls += 1
        }
    }

    func record(stage: Ink2Stage, commandBuffer cb: any MTLCommandBuffer) {
        guard isEnabled else { return }
        let start = cb.gpuStartTime
        let end = cb.gpuEndTime
        guard start > 0, end > start else { return }
        record(stage: stage, cbDurationMs: (end - start) * 1000.0)
    }

    func recordBlit(count: Int = 1) {
        guard isEnabled, count > 0 else { return }
        lock.withLock { st in
            guard let st else { return }
            st.blitCount += count
            st.activeStage = nil
        }
    }

    func beginHostTimer() -> UInt64 {
        guard isEnabled else { return 0 }
        return DispatchTime.now().uptimeNanoseconds
    }

    func endHostTimer(_ startNs: UInt64) {
        guard isEnabled, startNs > 0 else { return }
        let now = DispatchTime.now().uptimeNanoseconds
        guard now >= startNs else { return }
        recordHostOverhead(ms: Double(now - startNs) * 1e-6)
    }

    func recordHostOverhead(ms: Double) {
        guard isEnabled, ms > 0 else { return }
        lock.withLock { st in
            st?.cpuHostMs += ms
        }
    }

    /// Resolves completed CB GPU timing + stage-boundary counters and resets counter slots.
    func finishCommandBuffer(
        _ cb: any MTLCommandBuffer,
        tokens: Int = 1,
        hostOverheadMs: Double = 0.0
    ) {
        guard isEnabled else { return }
        let start = cb.gpuStartTime
        let end = cb.gpuEndTime
        let cbMs = (start > 0 && end > start) ? (end - start) * 1000.0 : 0.0
        lock.withLock { st in
            guard let st else { return }
            st.activeStage = nil
            st.steps += 1
            st.tokens += max(0, tokens)
            st.totalCBGpuMs += cbMs
            if hostOverheadMs > 0 { st.cpuHostMs += hostOverheadMs }
            if let sp = st.stageBox?.profiler {
                defer { sp.reset() }
                if let pairs = try? sp.resolveByStage() {
                    for (stage, micros) in pairs {
                        let idx = stage.index
                        st.buckets[idx].totalMs += max(0.0, micros * 0.001)
                        st.buckets[idx].calls += 1
                    }
                }
            }
        }
    }

    func snapshot() -> Summary? {
        guard isEnabled else { return nil }
        return lock.withLock { st -> Summary? in
            guard let st else { return nil }
            let stageSum = st.buckets.reduce(0.0) { $0 + $1.totalMs }
            let gpuDenom = max(stageSum, st.totalCBGpuMs, 1e-9)
            var rows: [StageRow] = []
            rows.reserveCapacity(Ink2Stage.allCases.count)
            for stage in Ink2Stage.allCases {
                let b = st.buckets[stage.index]
                guard b.calls > 0 else { continue }
                let avg = b.totalMs / Double(b.calls)
                let pct = 100.0 * b.totalMs / gpuDenom
                rows.append(StageRow(stage: stage, totalMs: b.totalMs, percentGPU: pct, calls: b.calls, avgMs: avg))
            }
            rows.sort { a, b in
                if a.totalMs != b.totalMs { return a.totalMs > b.totalMs }
                return a.stage.rawValue < b.stage.rawValue
            }
            return Summary(
                mode: mode,
                steps: st.steps,
                tokens: st.tokens,
                totalStageGpuMs: stageSum,
                totalCBGpuMs: st.totalCBGpuMs,
                cpuHostMs: st.cpuHostMs,
                blitCount: st.blitCount,
                rows: rows
            )
        }
    }

    func formattedTable() -> String {
        guard let sum = snapshot() else { return "" }
        let cbGpu = sum.totalCBGpuMs > 0 ? sum.totalCBGpuMs : sum.totalStageGpuMs
        var out = String(
            format: "[ink2-profile:%@] steps=%d tokens=%d stage_gpu=%.3fms cb_gpu=%.3fms cpu_host=%.3fms blits=%d\n",
            sum.mode.rawValue, sum.steps, sum.tokens, sum.totalStageGpuMs, cbGpu, sum.cpuHostMs, sum.blitCount
        )
        out += "stage            |   total ms | % of GPU |   calls |   avg ms\n"
        out += "-----------------+------------+----------+---------+---------\n"
        for row in sum.rows {
            let name = row.stage.rawValue.padding(toLength: 16, withPad: " ", startingAt: 0)
            out += String(
                format: "%@ | %10.3f | %7.1f%% | %7d | %8.3f\n",
                name, row.totalMs, row.percentGPU, row.calls, row.avgMs
            )
        }
        return out
    }

    func printSummary(to handle: FileHandle = .standardError) {
        guard isEnabled else { return }
        let table = formattedTable()
        guard !table.isEmpty else { return }
        handle.write(Data(table.utf8))
    }

    func reset() {
        guard isEnabled else { return }
        lock.withLock { st in
            guard let st else { return }
            for i in st.buckets.indices {
                st.buckets[i] = Bucket()
            }
            st.stageBox?.profiler.reset()
            st.activeStage = nil
            st.totalCBGpuMs = 0.0
            st.cpuHostMs = 0.0
            st.blitCount = 0
            st.steps = 0
            st.tokens = 0
        }
    }
}
