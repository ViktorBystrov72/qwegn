// QWCommandFrame.swift — Command-buffer, encoder, profiling, and cold-unload lifecycle.
import Foundation
import Metal
import QWCore
import QWHAL

/// Owns the active command buffer/encoder pair, split step CB list, cold-event
/// pipeline, pending cold unloads, and stage profiling telemetry for one GPU pass.
final class QWCommandFrame: @unchecked Sendable {
    var activeCB: (any MTLCommandBuffer)?
    var activeEnc: (any MTLComputeCommandEncoder)?
    var stepCBs: [any MTLCommandBuffer]?
    var decodeTel = QWDecodeTelemetry()
    var coldEvents: ColdEventPipeline?
    var pendingColdUnloads: [(streamer: QWEngineDriver.ColdExpertStreamer, stagedCount: Int)] = []

    func resetColdTelemetry() {
        decodeTel.coldStageUs = 0
        decodeTel.coldWaitUs = 0
        decodeTel.coldIOUs = 0
        decodeTel.coldStaged = 0
    }

    /// Starts a command frame by acquiring an initial `(cb, enc)` pair and tracking `stepCBs`.
    func begin(hal: QWHal) throws -> (cb: any MTLCommandBuffer, enc: any MTLComputeCommandEncoder) {
        guard let pair = hal.beginUnretained() else { throw QWEngineDriverError.state("cmdbuffer") }
        activeCB = pair.0
        activeEnc = pair.1
        stepCBs = []
        return pair
    }

    func closeActiveEncoder() {
        if let openEnc = activeEnc {
            activeEnc = nil
            openEnc.endEncoding()
        }
    }

    func slicePrefillCB(
        hal: QWHal,
        committedCbs: inout [any MTLCommandBuffer]
    ) throws -> (cb: any MTLCommandBuffer, enc: any MTLComputeCommandEncoder) {
        guard let cb = activeCB, let enc = activeEnc else {
            throw QWEngineDriverError.state("slicePrefillCB requires activeCB/activeEnc")
        }
        activeEnc = nil
        activeCB = nil
        hal.end(cb, enc)
        committedCbs.append(cb)
        stepCBs?.append(cb)
        if !pendingColdUnloads.isEmpty {
            cb.waitUntilCompleted()
            flushPendingColdUnloads()
        }
        guard let nextPair = hal.beginUnretained() else {
            throw QWEngineDriverError.state("cmdbuffer slice")
        }
        activeCB = nextPair.0
        activeEnc = nextPair.1
        return nextPair
    }

    func flushPendingColdUnloads() {
        guard !pendingColdUnloads.isEmpty else { return }
        var maxByPtr: [UnsafeMutableRawPointer: (streamer: QWEngineDriver.ColdExpertStreamer, maxCount: Int)] = [:]
        for item in pendingColdUnloads {
            let key = item.streamer.guBasePtr
            if let cur = maxByPtr[key] {
                if item.stagedCount > cur.maxCount {
                    maxByPtr[key] = (item.streamer, item.stagedCount)
                }
            } else {
                maxByPtr[key] = (item.streamer, item.stagedCount)
            }
        }
        for (_, entry) in maxByPtr {
            entry.streamer.unload(stagedCount: entry.maxCount)
        }
        pendingColdUnloads.removeAll(keepingCapacity: true)
    }

    @discardableResult
    func drainColdEvents(waitUntilIdle: Bool) -> (any Error)? {
        guard let pipe = coldEvents else { return nil }
        if waitUntilIdle { pipe.waitUntilIdle() }
        let r = pipe.takeStepResult()
        pipe.tuneBridge()
        decodeTel.coldWaitUs += r.waitUs
        decodeTel.coldIOUs += r.ioUs
        decodeTel.coldStaged += r.staged
        return r.failure
    }

    /// Drains any unawaited command buffers, cold events, and cold unloads when leaving a frame early.
    func abortIncomplete(
        admission: QWSessionAdmissionManager,
        session: QWEngineSession?,
        fallbackCBs: [any MTLCommandBuffer] = [],
        includeActiveCB: Bool = false,
        markSessionFailed: Bool = false,
        resetDecodeProfiler: Bool = false,
        clearPrefillProfiler: Bool = false,
        onDrainUpstream: () -> Void = {}
    ) {
        closeActiveEncoder()
        let pending = stepCBs ?? fallbackCBs
        for cb in pending where cb.status != .notEnqueued {
            cb.waitUntilCompleted()
            if cb.status != .completed || cb.error != nil {
                if markSessionFailed {
                    admission.recordCommandBufferError(for: cb)
                    session?.stateFailed = true
                } else {
                    admission.commandBufferErrors += 1
                }
            }
        }
        if includeActiveCB, let curCB = activeCB, curCB.status != .notEnqueued {
            curCB.waitUntilCompleted()
            if curCB.status != .completed || curCB.error != nil {
                if markSessionFailed {
                    admission.recordCommandBufferError(for: curCB)
                    session?.stateFailed = true
                } else {
                    admission.commandBufferErrors += 1
                }
            }
        }
        onDrainUpstream()
        _ = drainColdEvents(waitUntilIdle: true)
        flushPendingColdUnloads()
        if resetDecodeProfiler, let p = decodeTel.profiler { p.reset() }
        if clearPrefillProfiler { decodeTel.profiler = nil }
        clearActiveBindings()
    }

    func clearActiveBindings() {
        activeCB = nil
        activeEnc = nil
        stepCBs = nil
    }

    func profileStage(_ stage: String, hal: QWHal, enabled: Bool) {
        guard enabled,
              let p = decodeTel.profiler, p.hasSegmentCapacity,
              let cb = activeCB, let cur = activeEnc else { return }
        cur.endEncoding()
        guard let enc = try? p.makeEncoder(commandBuffer: cb, stage: stage) else {
            fatalError("QW_DECODE_PROFILE: encoder creation failed after ending the active encoder")
        }
        hal.touchKeepAlive(enc)
        activeEnc = enc
    }

    func rotateActiveEncoder(
        _ enc: inout any MTLComputeCommandEncoder,
        stage: String,
        hal: QWHal,
        profilingEnabled: Bool
    ) throws {
        if profilingEnabled,
           let p = decodeTel.profiler, p.hasSegmentCapacity,
           let cb = activeCB {
            activeEnc = nil
            enc.endEncoding()
            guard let next = try? p.makeEncoder(commandBuffer: cb, stage: stage) else {
                throw QWEngineDriverError.state("profile encoder rotation failed")
            }
            hal.touchKeepAlive(next)
            activeEnc = next
            enc = next
            return
        }
        guard let cb = activeCB else {
            throw QWEngineDriverError.state("rotateActiveEncoder requires activeCB")
        }
        activeEnc = nil
        try hal.rotateEncoder(cb, &enc)
        activeEnc = enc
    }

    func beginDecodeProfileStep(device: any MTLDevice, enabled: Bool) {
        guard enabled, decodeTel.profiler == nil else { return }
        guard QWGPUStageProfiler<String>.isSupported(on: device) else {
            FileHandle.standardError.write(Data("[decode-profile] stage sampling unsupported\n".utf8))
            return
        }
        decodeTel.profiler = try? QWGPUStageProfiler<String>(device: device, segmentCapacity: 4096)
    }

    func endDecodeProfileStep(enabled: Bool) {
        guard enabled, let p = decodeTel.profiler else { return }
        defer { p.reset() }
        guard let pairs = try? p.resolveByStage() else { return }
        var stepTotal: [String: Double] = [:]
        for (stage, us) in pairs { stepTotal[stage, default: 0] += us }
        for (stage, us) in stepTotal {
            let e = decodeTel.stageTotals[stage] ?? (0, 0)
            decodeTel.stageTotals[stage] = (e.us + us, e.n + 1)
        }
        decodeTel.profiledSteps += 1
        guard decodeTel.profiledSteps % 32 == 0 else { return }
        let n = Double(decodeTel.profiledSteps)
        let total = decodeTel.stageTotals.values.reduce(0) { $0 + $1.us } / n
        var s = String(format: "[decode-profile] steps=%d sampled_gpu=%.2fms/token\n", decodeTel.profiledSteps, total / 1000)
        for (k, v) in decodeTel.stageTotals.sorted(by: { $0.value.us > $1.value.us }) {
            s += String(format: "[decode-profile]   %-28@ %7.3f ms/token  %5.1f%%\n",
                        k as NSString, v.us / n / 1000, 100 * v.us / n / total)
        }
        FileHandle.standardError.write(Data(s.utf8))
    }

    func beginPrefillProfileChunk(device: any MTLDevice, enabled: Bool) {
        guard enabled, decodeTel.profiler == nil,
              QWGPUStageProfiler<String>.isSupported(on: device) else { return }
        decodeTel.profiler = try? QWGPUStageProfiler<String>(device: device, segmentCapacity: 4096)
        decodeTel.prefillChunkStart = (decodeTel.coldStageUs, decodeTel.coldWaitUs, decodeTel.coldIOUs,
                                       decodeTel.coldStaged, Date())
    }

    func endPrefillProfileChunk(tokens: Int, enabled: Bool) {
        guard enabled, let p = decodeTel.profiler else { return }
        defer { decodeTel.profiler = nil }
        guard let pairs = try? p.resolveByStage() else { return }
        var totals: [String: Double] = [:]
        for (stage, us) in pairs { totals[stage, default: 0] += us }
        let total = totals.values.reduce(0, +)
        var s = String(format: "[prefill-profile] tokens=%d sampled_gpu=%.2fms (%.1f tok/s)\n",
                       tokens, total / 1000, Double(tokens) / max(total, 1) * 1e6)
        if let c = decodeTel.prefillChunkStart {
            s += String(format: "[prefill-profile]   wall=%.2fms cold: staged=%d stage=%.2fms (gpu-wait %.2fms, io %.2fms)\n",
                        Date().timeIntervalSince(c.t0) * 1000, decodeTel.coldStaged - c.staged,
                        (decodeTel.coldStageUs - c.stageUs) / 1000, (decodeTel.coldWaitUs - c.waitUs) / 1000,
                        (decodeTel.coldIOUs - c.ioUs) / 1000)
        }
        for (k, us) in totals.sorted(by: { $0.value > $1.value }) {
            s += String(format: "[prefill-profile]   %-28@ %8.2f ms  %5.1f%%\n",
                        k as NSString, us / 1000, 100 * us / max(total, 1))
        }
        FileHandle.standardError.write(Data(s.utf8))
    }
}
