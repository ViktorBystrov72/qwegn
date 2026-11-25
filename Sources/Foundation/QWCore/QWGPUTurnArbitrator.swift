// QWGPUTurnArbitrator.swift — Lock-safe step-boundary Metal arbitration (ADR-002).
import Foundation
import os

/// Coordinates exclusive GPU ownership between multi-step primary decode and high-priority
/// Metal decision sidecar evaluations at clean step boundaries (`waitUntilCompleted()`).
public final class QWGPUTurnArbitrator: @unchecked Sendable {
    private struct State {
        var activePrimaryCount: Int = 0
        var sidecarActive: Bool = false
        var waitingSidecars: [CheckedContinuation<Void, Never>] = []
        var waitingPrimaries: [CheckedContinuation<Void, Never>] = []
        var syncWaitingPrimaries: Int = 0
        var totalPreemptionsServed: Int = 0
    }

    private let lock = OSAllocatedUnfairLock(initialState: State())
    private let syncCondition = NSCondition()

    public init() {}

    public var hasPendingSidecarWaiters: Bool {
        lock.withLock { !$0.waitingSidecars.isEmpty || $0.sidecarActive }
    }

    public var activePrimarySteps: Int {
        lock.withLock { $0.activePrimaryCount }
    }

    public var isSidecarActive: Bool {
        lock.withLock { $0.sidecarActive }
    }

    public var totalPreemptionsServed: Int {
        lock.withLock { $0.totalPreemptionsServed }
    }

    /// Acquires permission to submit one primary prefill/decode chunk synchronously on a worker thread.
    /// Fast-paths immediately when no decision sidecar holds or awaits the GPU.
    public func acquirePrimaryStepSync() {
        let fastAcquired = lock.withLock { s -> Bool in
            guard !s.sidecarActive && s.waitingSidecars.isEmpty else { return false }
            s.activePrimaryCount += 1
            return true
        }
        if fastAcquired { return }
        syncCondition.lock()
        var registered = false
        while true {
            let wasRegistered = registered
            let acquired = lock.withLock { s -> Bool in
                if !s.sidecarActive && s.waitingSidecars.isEmpty {
                    if wasRegistered {
                        s.syncWaitingPrimaries = max(0, s.syncWaitingPrimaries - 1)
                    }
                    s.activePrimaryCount += 1
                    return true
                }
                if !wasRegistered {
                    s.syncWaitingPrimaries += 1
                }
                return false
            }
            if acquired { break }
            registered = true
            syncCondition.wait()
        }
        syncCondition.unlock()
    }

    /// Runs one synchronous primary GPU step under the arbitrator, guaranteeing release on throw.
    public func withPrimaryStepSync<T>(_ body: () throws -> T) rethrows -> T {
        acquirePrimaryStepSync()
        do {
            let result = try body()
            releasePrimaryStep()
            return result
        } catch {
            releasePrimaryStep()
            throw error
        }
    }

    /// Acquires permission to submit one primary prefill/decode chunk. Fast-paths synchronously
    /// when no decision sidecar holds or awaits the GPU.
    public func acquirePrimaryStep() async {
        let fastAcquired = lock.withLock { s -> Bool in
            guard !s.sidecarActive && s.waitingSidecars.isEmpty else { return false }
            s.activePrimaryCount += 1
            return true
        }
        if fastAcquired { return }
        await withCheckedContinuation { cont in
            let resumeNow = lock.withLock { s -> Bool in
                if !s.sidecarActive && s.waitingSidecars.isEmpty {
                    s.activePrimaryCount += 1
                    return true
                }
                s.waitingPrimaries.append(cont)
                return false
            }
            if resumeNow { cont.resume() }
        }
    }

    /// Releases a primary step turn, immediately waking the next queued sidecar if all
    /// primary GPU steps have completed.
    public func releasePrimaryStep() {
        let (nextSidecar, primariesToWake, shouldBroadcastSync): (
            CheckedContinuation<Void, Never>?,
            [CheckedContinuation<Void, Never>],
            Bool
        ) = lock.withLock { s in
            s.activePrimaryCount = max(0, s.activePrimaryCount - 1)
            if s.activePrimaryCount == 0 && !s.sidecarActive {
                if !s.waitingSidecars.isEmpty {
                    s.sidecarActive = true
                    s.totalPreemptionsServed += 1
                    return (s.waitingSidecars.removeFirst(), [], false)
                } else {
                    var woken: [CheckedContinuation<Void, Never>] = []
                    if !s.waitingPrimaries.isEmpty {
                        s.activePrimaryCount += s.waitingPrimaries.count
                        woken = s.waitingPrimaries
                        s.waitingPrimaries.removeAll()
                    }
                    return (nil, woken, s.syncWaitingPrimaries > 0)
                }
            }
            return (nil, [], false)
        }
        if let nextSidecar {
            nextSidecar.resume()
        }
        for cont in primariesToWake { cont.resume() }
        if shouldBroadcastSync {
            syncCondition.lock()
            syncCondition.broadcast()
            syncCondition.unlock()
        }
    }

    /// Runs one primary GPU step under the arbitrator, guaranteeing release on throw.
    public func withPrimaryStep<T>(
        isolation: isolated (any Actor)? = #isolation,
        _ body: () async throws -> T
    ) async rethrows -> T {
        await acquirePrimaryStep()
        do {
            let result = try await body()
            releasePrimaryStep()
            return result
        } catch {
            releasePrimaryStep()
            throw error
        }
    }

    /// Acquires high-priority sidecar GPU access as soon as in-flight primary step finishes.
    public func acquireSidecarTurn() async {
        let immediate = lock.withLock { s -> Bool in
            if !s.sidecarActive && s.activePrimaryCount == 0 {
                s.sidecarActive = true
                return true
            }
            return false
        }
        if immediate { return }
        await withCheckedContinuation { cont in
            let resumeNow = lock.withLock { s -> Bool in
                if !s.sidecarActive && s.activePrimaryCount == 0 {
                    s.sidecarActive = true
                    s.totalPreemptionsServed += 1
                    return true
                }
                s.waitingSidecars.append(cont)
                return false
            }
            if resumeNow { cont.resume() }
        }
    }

    /// Releases sidecar ownership, handing FIFO to the next sidecar or waking paused primary steps.
    public func releaseSidecarTurn() {
        let (nextSidecar, primariesToWake, shouldBroadcastSync): (
            CheckedContinuation<Void, Never>?,
            [CheckedContinuation<Void, Never>],
            Bool
        ) = lock.withLock { s in
            if !s.waitingSidecars.isEmpty {
                s.totalPreemptionsServed += 1
                return (s.waitingSidecars.removeFirst(), [], false)
            }
            s.sidecarActive = false
            var woken: [CheckedContinuation<Void, Never>] = []
            if !s.waitingPrimaries.isEmpty {
                s.activePrimaryCount += s.waitingPrimaries.count
                woken = s.waitingPrimaries
                s.waitingPrimaries.removeAll()
            }
            return (nil, woken, s.syncWaitingPrimaries > 0)
        }
        if let nextSidecar {
            nextSidecar.resume()
        }
        for cont in primariesToWake { cont.resume() }
        if shouldBroadcastSync {
            syncCondition.lock()
            syncCondition.broadcast()
            syncCondition.unlock()
        }
    }

    /// Executes a high-priority Metal decision sidecar pass at the next clean step boundary.
    public func runHighPrioritySidecar<T>(
        isolation: isolated (any Actor)? = #isolation,
        _ body: () async throws -> T
    ) async rethrows -> T {
        await acquireSidecarTurn()
        do {
            let result = try await body()
            releaseSidecarTurn()
            return result
        } catch {
            releaseSidecarTurn()
            throw error
        }
    }
}
