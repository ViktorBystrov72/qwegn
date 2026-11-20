// QWWave6HALResidencyTests.swift — Stage 1 lock-free residency and unretained command buffers.
import Foundation
import Metal
import QWCore
@testable import QWHAL
import Synchronization
import XCTest

private struct SendableBuffers: @unchecked Sendable {
    let items: [any MTLBuffer]
}

final class QWWave6HALResidencyTests: XCTestCase {
    private func makeTestHAL(envOverrides: [String: String] = [:]) throws -> QWHal {
        var previous: [String: String?] = [:]
        for (key, val) in envOverrides {
            previous[key] = ProcessInfo.processInfo.environment[key]
            setenv(key, val, 1)
        }
        defer {
            for (key, oldVal) in previous {
                if let oldVal {
                    setenv(key, oldVal, 1)
                } else {
                    unsetenv(key)
                }
            }
        }
        return try QWHal(metallibPath: QWHALTests.anyMetallib(), geometry: .qwen38_27b)
    }

    private func makeSharedBuffer(_ device: any MTLDevice, bytes: Int = 256) throws -> any MTLBuffer {
        try XCTUnwrap(device.makeBuffer(length: bytes, options: .storageModeShared))
    }

    func testLockFreeResidencySnapshotPublishTouchAndEvictReclaim() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true",
                      "Metal command execution skipped on virtualized CI runner")
        let hal = try makeTestHAL()
        XCTAssertNil(hal.frozenResidencyPtr.load(ordering: .acquiring))
        XCTAssertEqual(hal.keepAliveLastTouchNs.load(ordering: .acquiring), 0)

        let b1 = try makeSharedBuffer(hal.device)
        let b2 = try makeSharedBuffer(hal.device)
        hal.makeResident([b1, b2])
        hal.makeResident([b1])

        let snap1 = try XCTUnwrap(hal.acquireResidencySnapshot())
        XCTAssertEqual(snap1.resources.count, 2)
        let t1 = hal.keepAliveLastTouchNs.load(ordering: .acquiring)
        XCTAssertGreaterThan(t1, 0)

        guard let (cb, enc) = hal.beginUnretained(.serial) else {
            return XCTFail("expected unretained command buffer and encoder")
        }
        XCTAssertFalse(cb.retainedReferences)
        hal.touchKeepAlive(enc)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed)

        let t2 = hal.keepAliveLastTouchNs.load(ordering: .acquiring)
        XCTAssertGreaterThanOrEqual(t2, t1)

        hal.evictResident([b1])
        let snap2 = try XCTUnwrap(hal.acquireResidencySnapshot())
        XCTAssertEqual(snap2.resources.count, 1)
        let retiredAfterEvict = hal.residencyState.withLock { $0.retiredSnapshots.count }
        XCTAssertEqual(retiredAfterEvict, 0)

        weak var ephemeralWeak: AnyObject?
        autoreleasepool {
            if let tmp = hal.device.makeBuffer(length: 4096, options: .storageModeShared) {
                ephemeralWeak = tmp as AnyObject
                hal.makeResident([tmp])
                hal.evictResident([tmp])
            }
        }
        XCTAssertNil(ephemeralWeak)

        hal.evictResident([b2])
        XCTAssertNil(hal.frozenResidencyPtr.load(ordering: .acquiring))
        XCTAssertNil(hal.acquireResidencySnapshot())
    }

    func testConcurrentTouchKeepAliveWithPublishAndEvict() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true",
                      "Metal command execution skipped on virtualized CI runner")
        let hal = try makeTestHAL()
        let anchor = try makeSharedBuffer(hal.device)
        hal.makeResident([anchor])

        var transient: [any MTLBuffer] = []
        for _ in 0..<8 {
            transient.append(try makeSharedBuffer(hal.device, bytes: 512))
        }
        let box = SendableBuffers(items: transient)

        let group = DispatchGroup()
        let q = DispatchQueue.global(qos: .userInitiated)
        let failures = Atomic<Int>(0)

        for worker in 0..<4 {
            group.enter()
            q.async {
                defer { group.leave() }
                for iter in 0..<24 {
                    if let (cb, enc) = hal.beginUnretained(.concurrent) {
                        hal.touchKeepAlive(enc)
                        hal.end(cb, enc)
                        cb.waitUntilCompleted()
                        if cb.status != .completed {
                            _ = failures.add(1, ordering: .relaxed)
                        }
                    } else {
                        _ = failures.add(1, ordering: .relaxed)
                    }
                    if worker == 0 {
                        let buf = box.items[iter % box.items.count]
                        if (iter & 1) == 0 {
                            hal.makeResident([buf])
                        } else {
                            hal.evictResident([buf])
                        }
                    }
                }
            }
        }
        group.wait()
        XCTAssertEqual(failures.load(ordering: .acquiring), 0)
        hal.evictResident(transient)
        let finalSnap = try XCTUnwrap(hal.acquireResidencySnapshot())
        XCTAssertEqual(finalSnap.resources.count, 1)
    }

    func testUnretainedCommandBufferDefaultAndOptOut() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true",
                      "Metal command execution skipped on virtualized CI runner")
        let halDefault = try makeTestHAL()
        XCTAssertFalse(halDefault.prefersRetainedCommandBuffers)

        let cbRaw = try XCTUnwrap(halDefault.makeUnretainedCommandBuffer())
        XCTAssertFalse(cbRaw.retainedReferences)

        guard let (retainedCB, retainedEnc) = halDefault.begin() else {
            return XCTFail("begin() failed")
        }
        XCTAssertTrue(retainedCB.retainedReferences)
        halDefault.end(retainedCB, retainedEnc)
        retainedCB.waitUntilCompleted()

        guard let (unretainedCB, unretainedEnc) = halDefault.beginUnretained(.concurrent) else {
            return XCTFail("beginUnretained(.concurrent) failed")
        }
        XCTAssertFalse(unretainedCB.retainedReferences)
        halDefault.end(unretainedCB, unretainedEnc)
        unretainedCB.waitUntilCompleted()

        let sharedEvent = try XCTUnwrap(halDefault.device.makeSharedEvent())
        guard let (waitCB, waitEnc) = halDefault.beginUnretained(.serial, waitingFor: sharedEvent, value: 1) else {
            return XCTFail("beginUnretained(_:waitingFor:value:) failed")
        }
        XCTAssertFalse(waitCB.retainedReferences)
        halDefault.end(waitCB, waitEnc)
        sharedEvent.signaledValue = 1
        waitCB.waitUntilCompleted()
        XCTAssertEqual(waitCB.status, .completed)

        let halOptOut = try makeTestHAL(envOverrides: ["QWG_RETAINED_CB": "1"])
        XCTAssertTrue(halOptOut.prefersRetainedCommandBuffers)
        let optOutCB = try XCTUnwrap(halOptOut.makeUnretainedCommandBuffer())
        XCTAssertTrue(optOutCB.retainedReferences)
        guard let (optOutPairCB, optOutPairEnc) = halOptOut.beginUnretained(.concurrent) else {
            return XCTFail("beginUnretained opt-out failed")
        }
        XCTAssertTrue(optOutPairCB.retainedReferences)
        halOptOut.end(optOutPairCB, optOutPairEnc)
        optOutPairCB.waitUntilCompleted()
    }

    func testAdoptPendingResidency() throws {
        let hal = try makeTestHAL()
        let pending = try makeSharedBuffer(hal.device)
        QWResidencyRegistry.note(pending)
        hal.adoptPendingResidency()
        let snap = try XCTUnwrap(hal.acquireResidencySnapshot())
        XCTAssertTrue(snap.resources.contains { ($0 as AnyObject) === (pending as AnyObject) })
    }
}
