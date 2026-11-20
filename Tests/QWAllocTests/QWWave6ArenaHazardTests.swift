// QWWave6ArenaHazardTests.swift — Tests arena hazard tracking policies and env overrides.
import Foundation
import Metal
@testable import QWAlloc
import QWCore
import QWHAL
import XCTest

final class QWWave6ArenaHazardTests: XCTestCase {
    private func withEnv(_ overrides: [String: String?], body: () throws -> Void) throws {
        var previous: [String: String?] = [:]
        for (key, value) in overrides {
            previous[key] = ProcessInfo.processInfo.environment[key]
            if let value {
                setenv(key, value, 1)
            } else {
                unsetenv(key)
            }
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
        try body()
    }

    func testHazardPolicyEnvironmentResolution() {
        XCTAssertEqual(QWArena.resolveHazardPolicy(environment: [:]), .safeDefault)
        XCTAssertEqual(QWArena.resolveHazardPolicy(environment: ["QWG_UNTRACKED_ARENAS": "1"]), .forceUntracked)
        XCTAssertEqual(QWArena.resolveHazardPolicy(environment: ["QWG_TRACKED_ARENAS": "1"]), .forceTracked)
        XCTAssertEqual(
            QWArena.resolveHazardPolicy(environment: ["QWG_TRACKED_ARENAS": "1", "QWG_UNTRACKED_ARENAS": "1"]),
            .forceTracked
        )
    }

    func testResourceOptionsAndRealMetalBufferModes() throws {
        let dev = QWAllocTests.hal.device

        let weightDefault = QWArena.resourceOptions(forWeightSlab: true, policy: .safeDefault)
        XCTAssertTrue(weightDefault.contains(.hazardTrackingModeUntracked))
        let weightBuf = try XCTUnwrap(dev.makeBuffer(length: 256, options: weightDefault))
        XCTAssertEqual(weightBuf.hazardTrackingMode, .untracked)

        let mutableDefault = QWArena.resourceOptions(forWeightSlab: false, policy: .safeDefault)
        XCTAssertFalse(mutableDefault.contains(.hazardTrackingModeUntracked))
        let mutableBuf = try XCTUnwrap(dev.makeBuffer(length: 256, options: mutableDefault))
        XCTAssertEqual(mutableBuf.hazardTrackingMode, .tracked)

        let forcedTracked = QWArena.resourceOptions(forWeightSlab: true, policy: .forceTracked)
        let trackedBuf = try XCTUnwrap(dev.makeBuffer(length: 256, options: forcedTracked))
        XCTAssertEqual(trackedBuf.hazardTrackingMode, .tracked)

        let forcedUntracked = QWArena.resourceOptions(forWeightSlab: false, policy: .forceUntracked)
        let untrackedBuf = try XCTUnwrap(dev.makeBuffer(length: 256, options: forcedUntracked))
        XCTAssertEqual(untrackedBuf.hazardTrackingMode, .untracked)
    }

    func testDefaultArenaTracksSharedBufferAndPreservesLayout() throws {
        try withEnv(["QWG_TRACKED_ARENAS": nil, "QWG_UNTRACKED_ARENAS": nil]) {
            let arena = try QWArena(
                hal: QWAllocTests.hal,
                weights: 1024,
                activations: 1024,
                kv: 1024,
                gdnStates: 1024,
                keepAlive: false
            )
            XCTAssertEqual(arena.hazardPolicy, .safeDefault)
            XCTAssertTrue(arena.allBuffers.allSatisfy { $0.hazardTrackingMode == .tracked })
            XCTAssertEqual(arena.activationsRegion().region.base, 1024)
            XCTAssertEqual(arena.kvRegion().region.base, 2048)
            XCTAssertEqual(arena.gdnRegion().region.base, 3072)
        }
    }

    func testUntrackedAndTrackedEnvTogglesOnArenaBuffers() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true",
                      "Metal execution skipped on virtualized CI runner")

        try withEnv(["QWG_TRACKED_ARENAS": nil, "QWG_UNTRACKED_ARENAS": "1"]) {
            let arena = try QWArena(
                hal: QWAllocTests.hal,
                weights: 512,
                activations: 512,
                kv: 512,
                gdnStates: 512,
                keepAlive: true
            )
            XCTAssertEqual(arena.hazardPolicy, .forceUntracked)
            XCTAssertTrue(arena.allBuffers.allSatisfy { $0.hazardTrackingMode == .untracked })

            let act = try arena.activationsRegion().allocate(QWTensorShape([16], .f32))
            arena.write(act, from: [Float](repeating: 3.5, count: 16))
            guard let cb = QWAllocTests.hal.makeUnretainedCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else {
                return XCTFail("failed to begin unretained CB")
            }
            QWAllocTests.hal.touchKeepAlive(enc)
            enc.memoryBarrier(scope: .buffers)
            QWAllocTests.hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertEqual(cb.status, .completed)
            XCTAssertEqual(arena.read(act, as: Float.self, count: 16), [Float](repeating: 3.5, count: 16))
        }

        try withEnv(["QWG_TRACKED_ARENAS": "1", "QWG_UNTRACKED_ARENAS": "1"]) {
            let arena = try QWArena(
                hal: QWAllocTests.hal,
                weights: 512,
                activations: 512,
                kv: 512,
                gdnStates: 512,
                keepAlive: false
            )
            XCTAssertEqual(arena.hazardPolicy, .forceTracked)
            XCTAssertTrue(arena.allBuffers.allSatisfy { $0.hazardTrackingMode == .tracked })
        }
    }
}
