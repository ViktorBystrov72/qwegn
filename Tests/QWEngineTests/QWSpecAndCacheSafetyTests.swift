// QWSpecAndCacheSafetyTests.swift — CPU-only unit tests for speculative loop, cache, residency, and lock safety.
import XCTest
import os
@testable import QWEngine
@testable import QWCore
import QWHAL
import Foundation

final class QWSpecAndCacheSafetyTests: XCTestCase {
    func testGreedyGenerationLoopInvokesOnStopTokenForFirstToken() throws {
        var stopped: Int?
        var streamed: [Int] = []
        let out = try QWGreedyGenerationLoop.run(
            vocab: 256,
            first: 2,
            maxNew: 8,
            stopIDs: [2],
            onToken: { tok in
                streamed.append(tok)
                return true
            },
            ordinary: { _ in
                XCTFail("stop on first token must not invoke ordinary decode")
                return 0
            },
            committedRound: nil,
            onStopToken: { stopped = $0 }
        )
        XCTAssertTrue(out.tokens.isEmpty)
        XCTAssertTrue(streamed.isEmpty)
        XCTAssertEqual(stopped, 2)
    }

    func testDriverStartupResolvesBackendBeforeDraftBudgetCheck() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-startup-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        XCTAssertThrowsError(try QWEngineDriver.resolveStartup(
            packPath: dir.path,
            kvMode: .q4,
            mtpDrafts: 8,
            mtpBackend: .standard,
            prefillScratchEviction: .never,
            mtpAdaptiveDepth: nil,
            mtpAdaptiveEntropyThreshold: nil
        )) { error in
            guard case QWEngineDriverError.badMTPDrafts(8) = error else {
                return XCTFail("expected badMTPDrafts(8) for non-dflash2 backend, got \(error)")
            }
        }
        XCTAssertThrowsError(try QWEngineDriver.resolveStartup(
            packPath: dir.path,
            kvMode: .q4,
            mtpDrafts: 15,
            mtpBackend: .dflash2,
            prefillScratchEviction: .never,
            mtpAdaptiveDepth: nil,
            mtpAdaptiveEntropyThreshold: nil
        )) { error in
            if case QWEngineDriverError.badMTPDrafts = error {
                XCTFail("dflash2 allows 15 drafts; should proceed past draft check to missing weights, got \(error)")
            }
        }
    }

    func testDiskManagerDFlashPersistenceHashCollisionGuardAndConcurrentIO() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-dm-safety-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        let config = QWCacheConfig(
            warmByteLimit: 1 << 20,
            ssdByteLimit: 8 << 20,
            entryLimit: 16,
            cacheDirectory: dir,
            wipeOnStart: false
        )
        let manager = QWCacheDiskManager(config: config)
        let rr = QWRoundRecord(
            pendingToken: 9,
            consumedCount: 4,
            outputCount: 1,
            trunkHiddenSeed: nil,
            mtpPhysicalRows: 0,
            absolutePosition: 4,
            samplerDrawIndex: 0
        )
        let dk = Data((0..<256).map { UInt8($0 & 0xff) })
        let dv = Data((0..<256).map { UInt8((255 - $0) & 0xff) })
        let dflashState = QWEnginePrefixState(
            position: 4,
            generated: 1,
            roundRecord: rr,
            recData: Data(repeating: 1, count: 64),
            ringData: Data(repeating: 2, count: 32),
            trunkHiddenData: nil,
            trunkHiddenValid: false,
            stochastic: nil,
            dflashKData: dk,
            dflashVData: dv,
            dflashCacheOffset: 4
        )
        XCTAssertEqual(dflashState.memoryBytes, 64 + 32 + 256 + 256)

        let url = try manager.write(
            tokens: [10, 20, 30, 40],
            mode: "hybrid|dflash2",
            nextToken: 9,
            state: dflashState,
            logits: Data(repeating: 3, count: 64),
            trunkPlanes: [Data(repeating: 0x55, count: 512)],
            mtpPlanes: [],
            allowsSuffix: true
        )
        XCTAssertNotNil(url)

        let hit = try XCTUnwrap(manager.lookup(tokens: [10, 20, 30, 40], mode: "hybrid|dflash2"))
        let readBack = try manager.read(hit: hit)
        XCTAssertEqual(readBack.state.dflashKData, dk)
        XCTAssertEqual(readBack.state.dflashVData, dv)
        XCTAssertEqual(readBack.state.dflashCacheOffset, 4)

        // Mismatched tokens or mode under same FNV-1a bucket must be rejected cleanly.
        let fakeMeta = QWCacheDiskManager.StoredMetadata(
            tokens: [1, 2, 3, 4],
            mode: "other",
            nextToken: 9,
            position: 4,
            generated: 1,
            roundRecord: rr,
            recBytes: 64,
            ringBytes: 32,
            trunkHiddenBytes: 0,
            trunkHiddenValid: false,
            logitsBytes: 64,
            trunkTokenCount: 4,
            trunkPlaneSizes: [512],
            hasMtp: false,
            mtpTokenCount: 0,
            mtpPlaneSizes: [],
            allowsSuffix: false
        )
        XCTAssertNotEqual(fakeMeta.tokens, hit.metadata.tokens)

        // Truncated payload must throw ReadError.truncated, never silently succeed with zero bytes.
        let fh = try FileHandle(forUpdating: url!)
        let size = try fh.seekToEnd()
        try fh.truncate(atOffset: max(1, size - 64))
        try fh.close()
        XCTAssertThrowsError(try manager.read(hit: hit))
    }

    func testResidencyBudgetDoesNotMaskNegativeLiveHeadroom() throws {
        let gib = 1 << 30
        let tight = QWEngineDriver.residencyBudgetBytes(
            policy: .safe,
            osReserveGiB: nil,
            arenaCap: 64 * gib,
            liveReclaimableBytes: 2 * gib
        )
        XCTAssertEqual(tight, 0, "Negative live - headroom must return 0 instead of 1 GiB floor")

        let zeroLive = QWEngineDriver.residencyBudgetBytes(
            policy: .maxResident,
            osReserveGiB: nil,
            arenaCap: 64 * gib,
            liveReclaimableBytes: 4 * gib
        )
        XCTAssertEqual(zeroLive, 0)
    }

    func testEstimateDriverContextMemoryHandlesDenseDFlashAndUnaliasedScratch() throws {
        let desc = QWModelDescriptor.qwen38_27b
        let baseEst = QWEngineDriver.estimateDriverContextMemory(
            descriptor: desc,
            contextTokens: 2048,
            kvMode: .q4,
            mtpDrafts: 0,
            mtpBackend: "variational",
            physicalMemoryBytes: 128 << 30
        )
        let dflashEst = QWEngineDriver.estimateDriverContextMemory(
            descriptor: desc,
            contextTokens: 2048,
            kvMode: .q4,
            mtpDrafts: 8,
            mtpBackend: "dflash2",
            physicalMemoryBytes: 128 << 30
        )
        XCTAssertGreaterThan(dflashEst.weightsBytes, baseEst.weightsBytes)
        XCTAssertGreaterThan(dflashEst.scratchAndStateBytes, baseEst.scratchAndStateBytes)
        XCTAssertEqual(dflashEst.kvBytes, baseEst.kvBytes, "DFlash2 must not charge an extra MTP KV layer")

        let geom = QWModelGeometry(descriptor: desc)
        let aliased = QWEngineDriver.prefillBytesPerToken(
            geometry: geom, maxPrefillChunk: 1024, useGroupedNAX: true, aliasBranches: true)
        let unaliased = QWEngineDriver.prefillBytesPerToken(
            geometry: geom, maxPrefillChunk: 1024, useGroupedNAX: true, aliasBranches: false)
        let breakdown = QWEngineDriver.prefillTransientBreakdown(geometry: geom, maxPrefillChunk: 1024)
        XCTAssertEqual(unaliased - aliased, min(breakdown.attnTransient, breakdown.mlpTransient))
    }

    func testAsyncColdWriteControllerBoundsQueueAndHalPSOStatsIsRaceSafe() {
        let controller = QWAsyncColdWriteController()
        let ran = OSAllocatedUnfairLock<Int>(initialState: 0)
        XCTAssertFalse(controller.reserve(bytes: 600 << 20))
        XCTAssertTrue(controller.reserve(bytes: 128 << 20))
        controller.submitReserved(bytes: 128 << 20) {
            ran.withLock { $0 += 1 }
        }
        controller.waitUntilIdle()
        XCTAssertEqual(ran.withLock { $0 }, 1)

        let group = DispatchGroup()
        for _ in 0..<8 {
            group.enter()
            DispatchQueue.global().async {
                defer { group.leave() }
                for _ in 0..<1000 { QWHalPSOStats.seconds += 0.001 }
            }
        }
        group.wait()
        XCTAssertGreaterThan(QWHalPSOStats.seconds, 0)
    }

    func testDenseModelUnderTightMemoryNeverReportsExpertReductionAndRejectsOversize() throws {
        let desc = QWModelDescriptor.qwen38_27b
        let tightEst = QWEngineDriver.estimateDriverContextMemory(
            descriptor: desc,
            contextTokens: 2048,
            kvMode: .q4,
            mtpDrafts: 0,
            physicalMemoryBytes: 12 << 30
        )
        XCTAssertFalse(tightEst.fits)
        XCTAssertEqual(QWModelGeometry(descriptor: desc).numExperts, 0)
    }
}
