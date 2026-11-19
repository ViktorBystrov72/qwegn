// QWSessionAdmissionManager.swift — Engine admission locks, GDN slots, and CB failure tracking.
import Foundation
import Metal
import os
import Synchronization
import QWCore

/// Owns request/generation concurrency gates, pre-carved GDN session slots,
/// and per-session command-buffer failure tracking for `QWEngineDriver`.
final class QWSessionAdmissionManager: @unchecked Sendable {
    let flight = NSLock()
    let generateLock = NSRecursiveLock()
    let requestGate = QWGenerationRequestGate()
    let slotLock = NSLock()
    var freeSlots: [QWEngineDriver.GdnSlot] = []

    private let _commandBufferErrors = Atomic<Int>(0)
    private let recordedFailedCBsLock = OSAllocatedUnfairLock<Set<ObjectIdentifier>>(initialState: [])
    private let failedCommitSessionsLock = OSAllocatedUnfairLock<Set<ObjectIdentifier>>(initialState: [])

    var commandBufferErrors: Int {
        get { _commandBufferErrors.load(ordering: .acquiring) }
        set { _commandBufferErrors.store(newValue, ordering: .releasing) }
    }

    func withFlight<T>(_ body: () throws -> T) throws -> T {
        guard flight.try() else { throw QWEngineDriverError.busy }
        defer { flight.unlock() }
        return try body()
    }

    func acquireRequest() throws -> QWGenerationPermit {
        try requestGate.acquire(flight: flight)
    }

    func releaseRequest(_ permit: QWGenerationPermit) {
        requestGate.release(permit)
    }

    func withRequest<T>(_ body: (QWGenerationPermit) throws -> T) throws -> T {
        try requestGate.withRequest(flight: flight, body)
    }

    func acquireSlot(dflashPreferred: Bool) throws -> QWEngineDriver.GdnSlot {
        try slotLock.withLock {
            guard !freeSlots.isEmpty else { throw QWEngineDriverError.busy }
            let slotIndex = dflashPreferred
                ? (freeSlots.firstIndex(where: { $0.dflashSlot != nil }) ?? 0)
                : 0
            return freeSlots.remove(at: slotIndex)
        }
    }

    func releaseSlot(_ slot: QWEngineDriver.GdnSlot) {
        slotLock.withLock {
            freeSlots.append(slot)
        }
    }

    func recordCommandBufferError(for cb: (any MTLCommandBuffer)?) {
        let shouldAdd: Bool
        if let cb {
            let oid = ObjectIdentifier(cb as AnyObject)
            shouldAdd = recordedFailedCBsLock.withLock { $0.insert(oid).inserted }
        } else {
            shouldAdd = true
        }
        if shouldAdd {
            _commandBufferErrors.add(1, ordering: .releasing)
        }
    }

    func recordAsyncCommandBufferFailure(_ cb: any MTLCommandBuffer, session: QWEngineSession?) {
        recordCommandBufferError(for: cb)
        if let session {
            let oid = ObjectIdentifier(session)
            failedCommitSessionsLock.withLock { _ = $0.insert(oid) }
        }
    }

    @discardableResult
    func syncCommitFailure(for session: QWEngineSession, cb: (any MTLCommandBuffer)? = nil) -> Bool {
        if let cb, cb.status != .completed {
            recordCommandBufferError(for: cb)
            session.stateFailed = true
        }
        let oid = ObjectIdentifier(session)
        let asyncFailed = failedCommitSessionsLock.withLock { $0.remove(oid) != nil }
        if asyncFailed {
            session.stateFailed = true
        }
        return session.stateFailed
    }
}
