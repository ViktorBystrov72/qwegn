// QWHybridSWABudgetAndCacheTests.swift — Tests for SWA driver budget, proportional pools, and compacted .qwcache IO.
import XCTest
import Foundation
@testable import QWEngine
@testable import QWCore
import QWHAL
import QWAlloc
import QWKV

final class QWHybridSWABudgetAndCacheTests: XCTestCase {
    private func makeHybridGeometry(layers: Int = 6, kvHeads: Int = 2, headDim: Int = 64,
                                    maxPos: Int = 131_072) throws -> QWModelGeometry {
        var types = [String](repeating: "sliding_attention", count: layers - 1)
        types.append("full_attention")
        let cfg: [String: Any] = [
            "model_type": "qwen3",
            "hidden_size": max(512, kvHeads * headDim),
            "vocab_size": 1024,
            "num_hidden_layers": layers,
            "layer_types": types,
            "sliding_window": 512,
            "num_attention_heads": max(8, kvHeads * 2),
            "num_key_value_heads": kvHeads,
            "head_dim": headDim,
            "intermediate_size": 1024,
            "max_position_embeddings": maxPos,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "eos_token_id": 2,
        ]
        let data = try JSONSerialization.data(withJSONObject: cfg)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    func testHybridKVBudgetAndContextMemoryEstimateScaleOnlyFullLayers() throws {
        let hybrid = try makeHybridGeometry(layers: 32, kvHeads: 8, headDim: 256, maxPos: 131_072)
        let fullCfg: [String: Any] = [
            "model_type": "qwen3",
            "hidden_size": 2048, "vocab_size": 1024, "num_hidden_layers": 32,
            "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
            "intermediate_size": 1024, "max_position_embeddings": 131_072,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "eos_token_id": 2,
        ]
        let full = QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: try JSONSerialization.data(withJSONObject: fullCfg)))

        let ctx = 65_536
        let fullKV = QWEngineDriver.kvBudgetBytes(maxTokens: ctx, mtpDrafts: 0, kvMode: .f16, geometry: full, prefillChunk: 512)
        let hybridKV = QWEngineDriver.kvBudgetBytes(maxTokens: ctx, mtpDrafts: 0, kvMode: .f16, geometry: hybrid, prefillChunk: 512)

        // 31 SWA layers stay bounded by w + prefillChunk + 512 = 1536 tokens instead of 65536.
        let swaCap = 512 + 512 + 512
        let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: hybrid)
        let expectedPages = 31 * QWEngineDriver.kvPagesPerPool(maxTokens: swaCap) + QWEngineDriver.kvPagesPerPool(maxTokens: ctx)
        let tables = QWKVCache.tableSlotBytes * 2 * 32 * QWKVCache.maxSeqs
        XCTAssertEqual(hybridKV, expectedPages * 2 * pageBytes + tables)
        XCTAssertLessThan(hybridKV, fullKV / 4)

        let tuning = QWEngineTuning(values: [:], environment: [:])
        let fullEst = QWEngineDriver.estimateDriverContextMemory(
            descriptor: full.descriptor, packPath: nil, contextTokens: ctx, kvMode: .f16,
            mtpDrafts: 0, mtpBackend: "standard", prefillChunkSize: 512, tuning: tuning,
            physicalMemoryBytes: 24 << 30)
        let hybridEst = QWEngineDriver.estimateDriverContextMemory(
            descriptor: hybrid.descriptor, packPath: nil, contextTokens: ctx, kvMode: .f16,
            mtpDrafts: 0, mtpBackend: "standard", prefillChunkSize: 512, tuning: tuning,
            physicalMemoryBytes: 24 << 30)
        XCTAssertGreaterThan(fullEst.maxFittingContext, 0)
        XCTAssertGreaterThan(hybridEst.maxFittingContext, fullEst.maxFittingContext * 4)
    }

    func testDriverInitMakeKVCachesSizesProportionalPoolsAndCompactedDiskIO() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true", "Requires Metal HAL")
        let geom = try makeHybridGeometry(layers: 4, maxPos: 16384)
        let hal = try QWHal(geometry: geom)
        let maxTokens = 8192
        let trunkKVBytes = QWEngineDriver.kvBudgetBytes(
            maxTokens: maxTokens, mtpDrafts: 0, kvMode: .f16, geometry: geom, prefillChunk: 512)
        let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: trunkKVBytes, gdnStates: 0)
        let caches = try QWEngineDriver.makeKVCaches(
            hal: hal, arena: arena, attnLayers: [0, 1, 2, 3], kvMode: .f16,
            geometry: geom, mtpDrafts: 0, trunkKVBytes: trunkKVBytes, mtpKVBytes: 0)
        XCTAssertEqual(caches.kvCache.layerSlidingWindows, [512, 512, 512, 0])
        XCTAssertNil(caches.mtpCache)

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-swa-disk-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let initialFree = caches.kvCache.freePageCount
        let tokenCount = 2048
        let h = try caches.kvCache.allocate(promptTokens: tokenCount)
        let swViews = try caches.kvCache.layerViews(h, layer: 0)
        let fullViews = try caches.kvCache.layerViews(h, layer: 3)
        let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: geom)
        XCTAssertEqual(swViews.k.pool.length / pageBytes, 7)
        XCTAssertEqual(fullViews.k.pool.length / pageBytes, 33)

        let rowElems = geom.kvChannels
        for chunkStart in stride(from: 0, to: tokenCount, by: 512) {
            let chunkEnd = min(tokenCount, chunkStart + 512)
            let count = chunkEnd - chunkStart
            for l in 0..<4 {
                let kRows = (0..<(count * rowElems)).map {
                    UInt16(truncatingIfNeeded: (chunkStart * rowElems + $0 + l * 13) & 0x7BFF)
                }
                let vRows = (0..<(count * rowElems)).map {
                    UInt16(truncatingIfNeeded: (chunkStart * rowElems + $0 + l * 29 + 7) & 0x7BFF)
                }
                try caches.kvCache.append(h, positions: chunkStart..<chunkEnd, layer: l, kind: .k, src: kRows)
                try caches.kvCache.append(h, positions: chunkStart..<chunkEnd, layer: l, kind: .v, src: vRows)
            }
            try caches.kvCache.recycleExpiredWindowPages(
                h, currentPosition: chunkEnd, safetyMargin: QWKVCache.pageTokens)
        }

        let shared = try caches.kvCache.captureSharedPrefix(h)
        var originalLivePages: [Data] = []
        try caches.kvCache.withSharedPrefixPagePointers(shared) { _, _, ptr, byteCount in
            originalLivePages.append(Data(bytes: ptr, count: byteCount))
        }
        let fileURL = dir.appendingPathComponent("hybrid_swa.qwcache")
        let rr = QWRoundRecord(pendingToken: 5, consumedCount: tokenCount, outputCount: 1,
                               trunkHiddenSeed: nil, mtpPhysicalRows: 0, absolutePosition: tokenCount, samplerDrawIndex: 0)
        let state = QWEnginePrefixState(position: tokenCount, generated: 1, roundRecord: rr,
                                        recData: Data([1, 2, 3, 4]), ringData: Data([5, 6, 7, 8]),
                                        trunkHiddenData: nil, trunkHiddenValid: false, stochastic: nil)
        let written = try QWCacheDiskManager.writeStandaloneFile(
            to: fileURL, tokens: Array(0..<tokenCount), mode: "hybrid|swa", nextToken: 5,
            state: state, logits: Data(repeating: 0x11, count: 64), trunk: shared, kvCache: caches.kvCache,
            mtp: nil, mtpCache: nil, allowsSuffix: true)
        let meta = try XCTUnwrap(written?.metadata)
        let swaPlaneBytes = 2 * pageBytes
        let fullPlaneBytes = 8 * pageBytes
        XCTAssertEqual(meta.trunkPlaneSizes, [swaPlaneBytes, swaPlaneBytes, swaPlaneBytes, swaPlaneBytes,
                                              swaPlaneBytes, swaPlaneBytes, fullPlaneBytes, fullPlaneBytes])

        caches.kvCache.finish(h)
        caches.kvCache.releaseSharedPrefix(shared)

        let restored = try QWCacheDiskManager.readStandaloneFile(
            at: fileURL, expectedMetadata: meta, kvCache: caches.kvCache, mtpCache: nil)
        XCTAssertEqual(restored.trunk.pageCount, 28) // 3 * (2 K + 2 V) + 1 * (8 K + 8 V) = 12 + 16
        XCTAssertEqual(restored.trunk.memoryBytes, 28 * pageBytes)

        var restoredLivePages: [Data] = []
        try caches.kvCache.withSharedPrefixPagePointers(restored.trunk) { _, _, ptr, byteCount in
            restoredLivePages.append(Data(bytes: ptr, count: byteCount))
        }
        XCTAssertEqual(restoredLivePages, originalLivePages)

        let restoredH = try caches.kvCache.allocate(adopting: restored.trunk, promptTokens: tokenCount)
        XCTAssertEqual(caches.kvCache.allocatedPhysicalPages(restoredH, layer: 0), 2)
        XCTAssertEqual(caches.kvCache.allocatedPhysicalPages(restoredH, layer: 3), 8)
        caches.kvCache.finish(restoredH)
        caches.kvCache.releaseSharedPrefix(restored.trunk)
        XCTAssertEqual(caches.kvCache.freePageCount, initialFree)
    }
}
