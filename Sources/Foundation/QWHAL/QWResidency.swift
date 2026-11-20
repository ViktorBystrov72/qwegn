// QWResidency.swift — Keeps long-lived GPU allocations wired.
// The iogpu collector unwires allocations no command buffer referenced recently once the process wires
// a large share of RAM (MoE slabs: ~90 GiB), so idle layers re-wire at ~190 GB/s on the critical path
// (hundreds of ms in early decode and prefill). Referencing every long-lived allocation from each encoder
// keeps them warm (a queue-attached MTLResidencySet does not: the set is reclaimed whole). Between requests
// an idle heartbeat commits an empty encoder referencing the same set; otherwise the first request after
// a few idle seconds re-wires it all (~650 ms for ~90 GiB) before its first kernel.

import Foundation
import Metal
import Synchronization

/// Zero-copy (mmap-backed) buffers created by loaders that have no HAL at
/// hand. The owning engine adopts them via `QWHal.adoptPendingResidency()`.
public enum QWResidencyRegistry {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var pending = NSHashTable<AnyObject>.weakObjects()

    public static func note(_ buffer: any MTLBuffer) {
        lock.lock(); defer { lock.unlock() }
        pending.add(buffer)
    }

    static func drain(device: any MTLDevice) -> [any MTLBuffer] {
        lock.lock(); defer { lock.unlock() }
        var out: [any MTLBuffer] = []
        var keep: [AnyObject] = []
        for obj in pending.allObjects {
            guard let b = obj as? any MTLBuffer else { continue }
            if b.device === device { out.append(b) } else { keep.append(obj) }
        }
        pending.removeAllObjects()
        for k in keep { pending.add(k) }
        return out
    }
}

/// Env `QWG_KEEPALIVE=0` disables keep-alive (A/B and diagnostics).
let keepAliveEnabled = ProcessInfo.processInfo.environment["QWG_KEEPALIVE"] != "0"
/// Idle heartbeat period; `QWG_KEEPALIVE_IDLE_MS=0` disables the heartbeat.
let keepAliveIdleMs: Int = Int(ProcessInfo.processInfo.environment["QWG_KEEPALIVE_IDLE_MS"] ?? "") ?? 250
/// After this long without real GPU work the heartbeat lets the collector reclaim the memory.
let keepAliveIdleMaxSec: Double = Double(ProcessInfo.processInfo.environment["QWG_KEEPALIVE_IDLE_MAX_S"] ?? "") ?? 60

private struct UncheckedBufferList: @unchecked Sendable {
    let items: [any MTLBuffer]
}

extension QWHal {
    @inline(__always)
    static func monotonicNowNs() -> UInt64 {
        max(1, DispatchTime.now().uptimeNanoseconds)
    }

    private func publishResidencySnapshotLocked(_ state: inout MutableResidencyState) {
        let newPtr: UnsafeMutableRawPointer? = state.resources.isEmpty
            ? nil
            : Unmanaged.passRetained(FrozenResidencySnapshot(state.resources)).toOpaque()
        if let prev = frozenResidencyPtr.exchange(newPtr, ordering: .sequentiallyConsistent) {
            state.retiredSnapshots.append(
                Unmanaged<FrozenResidencySnapshot>.fromOpaque(prev).takeRetainedValue()
            )
        }
    }

    @inline(__always)
    func acquireResidencySnapshot() -> FrozenResidencySnapshot? {
        residencyActiveReaders.add(1, ordering: .sequentiallyConsistent)
        let raw = frozenResidencyPtr.load(ordering: .sequentiallyConsistent)
        let snap = raw.map { Unmanaged<FrozenResidencySnapshot>.fromOpaque($0).takeUnretainedValue() }
        residencyActiveReaders.subtract(1, ordering: .sequentiallyConsistent)
        return snap
    }

    /// Registers `buffers` as long-lived; every encoder begun by this HAL references them.
    public func makeResident(_ buffers: [any MTLBuffer]) {
        guard !buffers.isEmpty else { return }
        let box = UncheckedBufferList(items: buffers)
        residencyState.withLock { state in
            var added = false
            for b in box.items where state.ids.insert(ObjectIdentifier(b)).inserted {
                state.resources.append(b)
                added = true
            }
            if added {
                publishResidencySnapshotLocked(&state)
            }
            keepAliveLastTouchNs.store(Self.monotonicNowNs(), ordering: .releasing)
            startKeepAliveHeartbeatLocked(&state)
        }
    }

    private func startKeepAliveHeartbeatLocked(_ state: inout MutableResidencyState) {
        guard keepAliveEnabled, keepAliveIdleMs > 0, state.heartbeat == nil else { return }
        // An encoder with no dispatch is dropped before it reaches the GPU, so the heartbeat runs a no-op kernel.
        guard let lib = try? device.makeLibrary(source: "kernel void qw_keepalive_nop() {}", options: nil),
              let fn = lib.makeFunction(name: "qw_keepalive_nop"),
              let pso = try? device.makeComputePipelineState(function: fn) else { return }
        state.nopPSO = pso
        let period = DispatchTimeInterval.milliseconds(keepAliveIdleMs)
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        timer.schedule(deadline: .now() + period, repeating: period, leeway: .milliseconds(keepAliveIdleMs / 4))
        timer.setEventHandler { [weak self] in self?.keepAliveIdleTick() }
        state.heartbeat = timer
        timer.resume()
    }

    private func keepAliveIdleTick() {
        guard keepAliveEnabled else { return }
        let lastTouchNs = keepAliveLastTouchNs.load(ordering: .acquiring)
        guard lastTouchNs > 0 else { return }
        let nowNs = Self.monotonicNowNs()
        guard nowNs >= lastTouchNs else { return }
        let idleSec = Double(nowNs - lastTouchNs) * 1e-9
        guard idleSec * 1000 >= Double(keepAliveIdleMs), idleSec < keepAliveIdleMaxSec,
              let snap = acquireResidencySnapshot(), !snap.resources.isEmpty else { return }
        guard let pso = residencyState.withLock({ $0.nopPSO }),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return }
        enc.useResources(snap.resources, usage: .read)
        enc.setComputePipelineState(pso)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
    }

    /// Unregisters `buffers` so their memory can be released.
    public func evictResident(_ buffers: [any MTLBuffer]) {
        guard !buffers.isEmpty else { return }
        let gone = Set(buffers.map { ObjectIdentifier($0) })
        residencyState.withLock { state in
            guard !state.ids.isDisjoint(with: gone) else { return }
            state.ids.subtract(gone)
            state.resources.removeAll { gone.contains(ObjectIdentifier($0)) }
            publishResidencySnapshotLocked(&state)
            for _ in 0..<10_000 {
                if residencyActiveReaders.load(ordering: .sequentiallyConsistent) == 0 {
                    state.retiredSnapshots.removeAll(keepingCapacity: false)
                    break
                }
                sched_yield()
            }
        }
    }

    /// Adopts every zero-copy loader buffer created on this device since the last call.
    public func adoptPendingResidency() {
        makeResident(QWResidencyRegistry.drain(device: device))
    }

    /// References every long-lived allocation from `enc` (lock-free atomic snapshot).
    @inline(__always)
    public func touchKeepAlive(_ enc: any MTLComputeCommandEncoder) {
        guard keepAliveEnabled else { return }
        keepAliveLastTouchNs.store(Self.monotonicNowNs(), ordering: .relaxed)
        if let snap = acquireResidencySnapshot(), !snap.resources.isEmpty {
            enc.useResources(snap.resources, usage: .read)
        }
    }
}
