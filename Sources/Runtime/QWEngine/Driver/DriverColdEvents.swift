// DriverColdEvents.swift — event-pipelined cold-expert staging for 1-row decode.
// Default on (tuning.coldEvents; QW_COLD_EVENTS=0 disables): each MoE route CB signals an MTLSharedEvent and commits without
// a host wait; a worker remaps ids, loads misses, writes promotions and signals the expert CB's wait value.
// A tuned delay CB (QW_COLD_EVENT_BRIDGE=0 disables) lets that signal land before the GPU waits; resident
// experts are mapped to bank slots on the GPU and run inside it, overlapping the remap and SSD reads
// (tuning.coldEventOverlap; QW_COLD_EVENT_OVERLAP=0 disables).
import Foundation
import QuartzCore
import Metal
import QWAlloc
import QWHAL
import QWGEMM

final class ColdEventPipeline: @unchecked Sendable {
    private final class WorkerState: @unchecked Sendable {
        struct Job {
            let wait: UInt64
            let run: () throws -> Bool
            let abort: () -> Void
        }
        let cond = NSCondition()
        var jobs: [Job] = []
        var isStopped = false
        var workerExited = false
        var failure: (any Error)?
        var gpuTimedOut = false
        var ioUs: Double = 0
        var waitUs: Double = 0
        var staged = 0
        var activeJobs = 0
        var signalTimes: [CFTimeInterval?] = []
    }

    let event: any MTLSharedEvent
    let promoList: any MTLBuffer
    let promoCapacity: Int
    let bridgeTick: any MTLBuffer
    /// GPU-mapped resident ids ([rows · K] Int32, -1 for a miss) for the overlapped resident pass.
    let residentIds: any MTLBuffer
    /// Delay length in dependent FMAs; tuned by `tuneBridge` from the step's timings.
    private(set) var bridgeIters: UInt32 = 3000
    /// This step's bridge CBs, in route order (encode thread only).
    var bridgeCBs: [any MTLCommandBuffer] = []
    static let bridgeEnabled = ProcessInfo.processInfo.environment["QW_COLD_EVENT_BRIDGE"] != "0"
    private var nextValue: UInt64 = 0
    private let state = WorkerState()
    static let spinTimeout: TimeInterval = 30

    var hasFailure: Bool {
        state.cond.lock()
        defer { state.cond.unlock() }
        return state.failure != nil
    }

    init?(device: any MTLDevice, promoCapacity: Int = 64) {
        guard let event = device.makeSharedEvent(),
              let list = device.makeBuffer(length: 4 * (1 + 2 * promoCapacity), options: .storageModeShared),
              let tick = device.makeBuffer(length: 16, options: .storageModePrivate),
              let resIds = device.makeBuffer(length: 4 * promoCapacity, options: .storageModePrivate)
        else { return nil }
        self.residentIds = resIds
        self.bridgeTick = tick
        self.event = event
        self.promoList = list
        self.promoCapacity = promoCapacity
        let workerState = self.state
        let t = Thread { [weak self, workerState] in
            Self.workerLoop(owner: { self }, state: workerState)
        }
        t.qualityOfService = .userInteractive
        t.name = "qwg.cold-events"
        t.start()
    }

    /// Stops the background cold-event worker thread and waits for it to exit cleanly.
    func shutdown() {
        state.cond.lock()
        if !state.isStopped {
            state.isStopped = true
            state.cond.broadcast()
        }
        while !state.workerExited {
            state.cond.wait()
        }
        state.cond.unlock()
    }

    deinit {
        shutdown()
    }

    /// Waits until all queued and active cold-event jobs have completed.
    func waitUntilIdle() {
        state.cond.lock()
        while !state.jobs.isEmpty || state.activeJobs > 0 {
            state.cond.wait()
        }
        state.cond.unlock()
    }

    /// Reserves (ready, done): the GPU signals `ready`, the worker signals `done`.
    func reserve() -> (ready: UInt64, done: UInt64) {
        nextValue += 2
        return (nextValue - 1, nextValue)
    }

    /// `run` returns whether it read from SSD; `onAbort` zeroes GPU inputs if skipped/failed.
    func enqueue(
        ready: UInt64,
        onAbort: @escaping () -> Void = {},
        _ run: @escaping () throws -> Bool
    ) {
        state.cond.lock()
        state.jobs.append(WorkerState.Job(wait: ready, run: run, abort: onAbort))
        state.cond.signal()
        state.cond.unlock()
    }

    func addStats(ioUs: Double, staged: Int) {
        state.cond.lock()
        state.ioUs += ioUs
        state.staged += staged
        state.cond.unlock()
    }

    /// Drains failure and counters; call only after the step's last CB completed.
    func takeStepResult() -> (failure: (any Error)?, ioUs: Double, waitUs: Double, staged: Int) {
        state.cond.lock()
        defer { state.cond.unlock() }
        let r = (state.failure, state.ioUs, state.waitUs, state.staged)
        state.failure = nil
        state.gpuTimedOut = false
        state.ioUs = 0
        state.waitUs = 0
        state.staged = 0
        return r
    }

    /// Retunes the delay from the step's bridge CBs and the host signal times: grow while the
    /// host signals after the bridge ends, shrink on spare slack. Call after the step completed.
    func tuneBridge() {
        state.cond.lock()
        let signals = state.signalTimes
        state.signalTimes.removeAll(keepingCapacity: true)
        state.cond.unlock()
        let bridges = bridgeCBs
        bridgeCBs.removeAll(keepingCapacity: true)
        guard Self.bridgeEnabled, signals.count == bridges.count, !signals.isEmpty else { return }
        var late = 0
        var slack: [Double] = []
        for (cb, signal) in zip(bridges, signals) {
            guard let t = signal else { continue }
            let end = cb.gpuEndTime
            guard end > 0 else { return }
            if t > end { late += 1 } else { slack.append(end - t) }
        }
        // A late layer costs a ~115 us wake, slack only its own length: shrink only when none was late.
        let iters = Double(bridgeIters)
        if late > 1 {
            bridgeIters = UInt32(min(iters * 1.2 + 100, 12_000))
        } else if late == 0, let minSlack = slack.min(), minSlack > 8e-6 {
            bridgeIters = UInt32(max(iters * 0.97, 500))
        }
    }

    private static func workerLoop(owner: () -> ColdEventPipeline?, state: WorkerState) {
        while true {
            state.cond.lock()
            while state.jobs.isEmpty && !state.isStopped { state.cond.wait() }
            if state.isStopped && state.jobs.isEmpty {
                state.workerExited = true
                state.cond.broadcast()
                state.cond.unlock()
                return
            }
            let job = state.jobs.removeFirst()
            state.activeJobs += 1
            let failed = state.failure != nil
            let earlierTimedOut = state.gpuTimedOut
            state.cond.unlock()

            guard let pipeline = owner(), !state.isStopped else {
                job.abort()
                state.cond.lock()
                state.activeJobs -= 1
                state.cond.broadcast()
                state.cond.unlock()
                continue
            }

            let t0 = Date()
            var ready = earlierTimedOut ? false : (pipeline.event.signaledValue >= job.wait)
            while !ready && !earlierTimedOut && !state.isStopped && Date().timeIntervalSince(t0) < Self.spinTimeout {
                ready = pipeline.event.signaledValue >= job.wait
            }
            let waited = Date().timeIntervalSince(t0) * 1e6
            var err: (any Error)?
            var didIO = false
            if !ready {
                err = QWEngineDriverError.state("cold event \(job.wait) never signalled")
            } else if !failed {
                do { didIO = try job.run() } catch { err = error }
            }
            if err != nil || failed {
                pipeline.promoList.contents().storeBytes(of: UInt32(0), as: UInt32.self)
                if ready { job.abort() }
            }
            // Only unblock GPU wait when route CB signaled ready and ids were safely invalidated on error.
            if ready {
                pipeline.event.signaledValue = job.wait + 1
            }
            let signalled = CACurrentMediaTime()
            state.cond.lock()
            state.waitUs += waited
            if !ready { state.gpuTimedOut = true }
            if let err, state.failure == nil { state.failure = err }
            state.signalTimes.append(didIO ? nil : signalled)
            state.activeJobs -= 1
            state.cond.broadcast()
            state.cond.unlock()
        }
    }
}

extension QWEngineDriver {
    func coldEventsUsable(_ moe: MoEW, rows: Int, isVerifyWindow: Bool = false) -> Bool {
        let totalRoutes = rows * moe.expertsPerTok
        guard tuning.coldEvents, !Self.decodeProfileEnabled, !QWMoEChain.coldTimingProxy,
              (1...4).contains(rows), stepCBs != nil,
              let st = moe.coldStreamer, st.isDynamic,
              moe.coldGateUp != nil, moe.coldDown != nil,
              st.guExpertBytes % 16 == 0, st.downExpertBytes % 16 == 0 else { return false }
        let effectiveSlots = st.effectiveColdBudget(isVerifyWindow: isVerifyWindow).slots
        guard effectiveSlots >= totalRoutes else { return false }
        if coldEvents == nil { coldEvents = ColdEventPipeline(device: hal.device) }
        return coldEvents != nil && totalRoutes <= coldEvents!.promoCapacity
    }

    /// Ends the route CB with a signal, hands the id rewrite + IO to the worker,
    /// and opens the expert CB behind the worker's completion value. With `resident`, the route CB
    /// also maps the ids to resident bank slots on the GPU and the resident experts run in the
    /// bridge CB, overlapping the host remap and SSD reads; the expert CB then needs only the misses.
    /// Returns whether `resident` was encoded.
    @discardableResult
    func stageColdExpertsEvent(moe: MoEW, s: MoEScratch, rows: Int = 1,
                               isVerifyWindow: Bool = false,
                               resident: ((any MTLComputeCommandEncoder, QWTensorValue) throws -> Void)? = nil) throws -> Bool {
        guard let pipe = coldEvents, let streamer = moe.coldStreamer,
              let curCB = activeCB, let curEnc = activeEnc else {
            throw QWEngineDriverError.state("cold event staging without pipeline or active CB")
        }
        if pipe.hasFailure {
            throw QWEngineDriverError.state("cold event staging aborted due to earlier layer failure")
        }
        let routes = moe.expertsPerTok
        let count = rows * routes
        var residentIds: QWTensorValue?
        if resident != nil, tuning.coldEventOverlap, count <= pipe.promoCapacity,
           let map = streamer.attachGPUSlotMap(device: hal.device) {
            let ids = QWTensorValue(region: QWMRegion(buffer: pipe.residentIds, base: 0, length: 4 * pipe.promoCapacity),
                                    offset: 0, shape: QWTensorShape([count], .i32))
            curEnc.setComputePipelineState(try hal.pipeline("qwg_moe_resident_ids"))
            s.idx.bind(curEnc, index: 0)
            curEnc.setBuffer(map, offset: 0, index: 1)
            ids.bind(curEnc, index: 2)
            var cnt = UInt32(count), experts = UInt32(streamer.totalExperts)
            curEnc.setBytes(&cnt, length: 4, index: 3)
            curEnc.setBytes(&experts, length: 4, index: 4)
            curEnc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
            residentIds = ids
        }
        let v = pipe.reserve()
        curEnc.endEncoding()
        activeEnc = nil
        curCB.encodeSignalEvent(pipe.event, value: v.ready)
        curCB.commit()
        activeCB = nil
        stepCBs?.append(curCB)
        // The signal fires only when its CB completes, so the delay is a CB of its own.
        if let residentIds, let resident {
            guard let (resCB, resEnc) = hal.begin(.serial) else {
                throw QWEngineDriverError.state("cold event resident cmdbuffer")
            }
            var resOpen = true
            defer { if resOpen { resEnc.endEncoding() } }
            try resident(resEnc, residentIds)
            if ColdEventPipeline.bridgeEnabled {
                resEnc.setComputePipelineState(try hal.pipeline("qwg_cold_event_bridge"))
                resEnc.setBuffer(pipe.bridgeTick, offset: 0, index: 0)
                var iters = pipe.bridgeIters
                resEnc.setBytes(&iters, length: 4, index: 1)
                resEnc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                resOpen = false
                hal.end(resCB, resEnc)
                pipe.bridgeCBs.append(resCB)
            } else {
                resOpen = false
                hal.end(resCB, resEnc)
            }
            stepCBs?.append(resCB)
        } else if ColdEventPipeline.bridgeEnabled, let (bridgeCB, bridge) = hal.begin(.serial) {
            var bridgeOpen = true
            defer { if bridgeOpen { bridge.endEncoding() } }
            bridge.setComputePipelineState(try hal.pipeline("qwg_cold_event_bridge"))
            bridge.setBuffer(pipe.bridgeTick, offset: 0, index: 0)
            var iters = pipe.bridgeIters
            bridge.setBytes(&iters, length: 4, index: 1)
            bridge.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            bridgeOpen = false
            hal.end(bridgeCB, bridge)
            stepCBs?.append(bridgeCB)
            pipe.bridgeCBs.append(bridgeCB)
        }
        guard let next = hal.begin(.serial, waitingFor: pipe.event, value: v.done) else {
            throw QWEngineDriverError.state("cold event cmdbuffer")
        }
        struct SendablePointers: @unchecked Sendable {
            let idxPtr: UnsafeMutableRawPointer
            let listPtr: UnsafeMutablePointer<UInt32>
        }
        let ptrs = SendablePointers(
            idxPtr: s.idx.region.buffer.contents().advanced(by: s.idx.region.base + s.idx.offset),
            listPtr: pipe.promoList.contents().assumingMemoryBound(to: UInt32.self)
        )
        let cap = pipe.promoCapacity
        pipe.enqueue(
            ready: v.ready,
            onAbort: {
                ptrs.listPtr[0] = 0
                let ids = ptrs.idxPtr.assumingMemoryBound(to: UInt32.self)
                for i in 0..<count { ids[i] = 0xFFFF_FFFF }
            }
        ) {
            let ids = ptrs.idxPtr.assumingMemoryBound(to: UInt32.self)
            let prep = try streamer.prepareSpan(
                idxPtr: ids, startRow: 0, totalRows: rows, routes: routes, isVerifyWindow: isVerifyWindow)
            guard prep.nextRow == rows, prep.promotions.count <= cap else {
                throw QWEngineDriverError.state("cold event span did not fit staging")
            }
            let t0 = Date()
            if prep.stagedCount > 0 {
                try streamer.executeBatch(guBatch: prep.guBatch, downBatch: prep.downBatch)
            }
            for (i, p) in prep.promotions.enumerated() {
                ptrs.listPtr[1 + 2 * i] = UInt32(p.staging)
                ptrs.listPtr[2 + 2 * i] = UInt32(p.bank)
            }
            ptrs.listPtr[0] = UInt32(prep.promotions.count)
            streamer.commitPromotion(prep.transaction, recordPendingPromotions: false)
            pipe.addStats(ioUs: Date().timeIntervalSince(t0) * 1e6, staged: prep.stagedCount)
            return prep.stagedCount > 0
        }
        activeCB = next.0
        activeEnc = next.1
        return residentIds != nil
    }

    /// Staging→bank copies for the promotion list the worker wrote (count may be 0).
    func encodeEventPromotions(moe: MoEW) throws {
        guard let pipe = coldEvents, let enc = activeEnc, let streamer = moe.coldStreamer,
              let coldGU = moe.coldGateUp, let coldDown = moe.coldDown else {
            throw QWEngineDriverError.state("cold event promotion without encoder or cold bank")
        }
        enc.setComputePipelineState(try hal.pipeline("qwg_moe_promote_copy"))
        for (src, dst, bytes) in [(coldGU.blocks, moe.gateUp.blocks, streamer.guExpertBytes),
                                  (coldDown.blocks, moe.down.blocks, streamer.downExpertBytes)] {
            src.bind(enc, index: 0)
            dst.bind(enc, index: 1)
            enc.setBuffer(pipe.promoList, offset: 0, index: 2)
            var vec = UInt32(bytes / 16)
            enc.setBytes(&vec, length: 4, index: 3)
            enc.dispatchThreads(MTLSize(width: min(Int(vec), 65_536), height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        enc.memoryBarrier(scope: .buffers)
    }
}
