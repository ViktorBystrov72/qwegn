// QWKVSWRingTests.swift — Tests for lazy SWA page backing, proportional pools, recycling, and prefix compaction.
import XCTest
import Metal
import Foundation
@testable import QWKV
import QWCore
import QWHAL
import QWAlloc

final class QWKVSWRingTests: XCTestCase {
    private static func makeHybridGeometry(windows: [Int], maxPos: Int = 16384) throws -> QWModelGeometry {
        let sw = windows.first(where: { $0 > 0 }) ?? 512
        let layerTypes = windows.map { $0 > 0 ? "sliding_attention" : "full_attention" }
        let cfg: [String: Any] = [
            "model_type": "qwen3",
            "hidden_size": 512,
            "vocab_size": 1024,
            "num_hidden_layers": windows.count,
            "layer_types": layerTypes,
            "sliding_window": sw,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "intermediate_size": 1024,
            "max_position_embeddings": maxPos,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "eos_token_id": 2,
        ]
        let data = try JSONSerialization.data(withJSONObject: cfg)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    private static func kvRows(seed: Int, count: Int, rowElems: Int) -> [UInt16] {
        (0..<(count * rowElems)).map { idx in
            UInt16(truncatingIfNeeded: (seed &* 131 + idx) & 0x7BFF)
        }
    }

    private static func readDeviceTable(_ view: QWKVLayerView, count: Int) -> [Int32] {
        let ptr = view.table.region.pointer.advanced(by: view.table.offset).assumingMemoryBound(to: Int32.self)
        return (0..<count).map { ptr[$0] }
    }

    func testLazySWABackingProportionalPoolsAndRingRecyclingAcrossKVModes() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true", "Requires Metal HAL")
        let geom = try Self.makeHybridGeometry(windows: [512, 0], maxPos: 16384)
        let hal = try QWHal(geometry: geom)
        let rowElems = geom.kvChannels
        let swaCapPages = 6
        let fullCapPages = 20

        for mode in [QWKVMode.f16, .q8, .q4] {
            let pageBytes = QWKVCache.pageBytes(mode: mode, geometry: geom)
            let tables = QWKVCache.tableSlotBytes * 2 * 2 * QWKVCache.maxSeqs
            let kvBytes = (swaCapPages + fullCapPages) * 2 * pageBytes + tables
            let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: kvBytes, gdnStates: 0)
            let cache = try QWKVCache(
                hal: hal, arena: arena, layers: [0, 1], mode: mode, geometry: geom,
                layerSlidingWindows: [512, 0], layerPoolWeights: [swaCapPages, fullCapPages]
            )

            // Verify proportional pool sizes.
            let h = try cache.allocate(promptTokens: 1024)
            let v0 = try cache.layerViews(h, layer: 0)
            let v1 = try cache.layerViews(h, layer: 1)
            XCTAssertEqual(v0.k.pool.length / pageBytes, swaCapPages, "mode=\(mode)")
            XCTAssertEqual(v1.k.pool.length / pageBytes, fullCapPages, "mode=\(mode)")

            // SWA layer starts completely unallocated (-1); Full layer eagerly allocates 4 pages.
            XCTAssertEqual(cache.allocatedPhysicalPages(h, layer: 0), 0)
            XCTAssertEqual(cache.allocatedPhysicalPages(h, layer: 1), 4)
            XCTAssertEqual(cache.pageIDs(h, layer: 0, kind: .k), [Int32](repeating: -1, count: 4))
            XCTAssertEqual(Self.readDeviceTable(v0.k, count: 4), [Int32](repeating: -1, count: 4))

            // Grow extends SWA lazily and Full eagerly.
            try cache.grow(h, to: 1536)
            XCTAssertEqual(cache.allocatedPhysicalPages(h, layer: 0), 0)
            XCTAssertEqual(cache.allocatedPhysicalPages(h, layer: 1), 6)

            // Stream 4096 tokens in 256-token chunks with ring recycling and rollback checks.
            let safetyMargin = 7 + QWKVCache.pageTokens
            let maxSWACap = ((512 + 256 + safetyMargin + QWKVCache.pageTokens - 1) / QWKVCache.pageTokens) + 1
            for chunkIdx in 0..<16 {
                let start = chunkIdx * 256
                let end = start + 256
                if h.tokenCapacity < end { try cache.grow(h, to: end) }
                let kSrc = Self.kvRows(seed: chunkIdx + 1, count: 256, rowElems: rowElems)
                let vSrc = Self.kvRows(seed: chunkIdx + 101, count: 256, rowElems: rowElems)
                for layer in [0, 1] {
                    try cache.append(h, positions: start..<end, layer: layer, kind: .k, src: kSrc)
                    try cache.append(h, positions: start..<end, layer: layer, kind: .v, src: vSrc)
                }
                let recycled = try cache.recycleExpiredWindowPages(h, currentPosition: end, safetyMargin: safetyMargin)
                let liveSW = cache.allocatedPhysicalPages(h, layer: 0)
                XCTAssertLessThanOrEqual(liveSW, maxSWACap, "mode=\(mode) chunk=\(chunkIdx)")
                XCTAssertEqual(cache.allocatedPhysicalPages(h, layer: 1), max(6, chunkIdx + 1))
                if end > 512 + safetyMargin + 256 {
                    XCTAssertGreaterThanOrEqual(recycled, 0)
                }
            }

            // Speculative draft beyond 4096 and rollback by 7 tokens cleanly restores capacity.
            try cache.grow(h, to: 4103)
            let draftRows = Self.kvRows(seed: 999, count: 7, rowElems: rowElems)
            for layer in [0, 1] {
                try cache.append(h, positions: 4096..<4103, layer: layer, kind: .k, src: draftRows)
                try cache.append(h, positions: 4096..<4103, layer: layer, kind: .v, src: draftRows)
            }
            try cache.rollback(h, by: 7)
            XCTAssertEqual(cache.written(h), 4096)
            let snap = try cache.snapshot(h)
            try cache.recycleExpiredWindowPages(h, validStart: 3840, safetyMargin: 0)
            let freeDuringSnap = cache.freePageCount
            cache.release(snap)
            XCTAssertGreaterThanOrEqual(cache.freePageCount, freeDuringSnap)

            cache.finish(h)
            XCTAssertEqual(cache.freePageCount, cache.totalPages, "mode=\(mode) must leak 0 pages")
        }
    }

    func testSharedAndPrivatePrefixCompactionAndAdoptionRoundtrip() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true", "Requires Metal HAL")
        let geom = try Self.makeHybridGeometry(windows: [512, 0], maxPos: 8192)
        let hal = try QWHal(geometry: geom)
        let rowElems = geom.kvChannels
        let tokenCount = 2048

        for mode in [QWKVMode.f16, .q8, .q4] {
            let pageBytes = QWKVCache.pageBytes(mode: mode, geometry: geom)
            let tables = QWKVCache.tableSlotBytes * 2 * 2 * QWKVCache.maxSeqs
            let poolBytes = pageBytes * 24 * 2 * 2
            let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: poolBytes + tables, gdnStates: 0)
            let cache = try QWKVCache(hal: hal, arena: arena, layers: [0, 1], mode: mode, geometry: geom)

            let donor = try cache.allocate(promptTokens: tokenCount)
            for layer in [0, 1] {
                let srcK = Self.kvRows(seed: 10 + layer, count: tokenCount, rowElems: rowElems)
                let srcV = Self.kvRows(seed: 50 + layer, count: tokenCount, rowElems: rowElems)
                try cache.append(donor, positions: 0..<tokenCount, layer: layer, kind: .k, src: srcK)
                try cache.append(donor, positions: 0..<tokenCount, layer: layer, kind: .v, src: srcV)
            }
            try cache.recycleExpiredWindowPages(donor, currentPosition: tokenCount, safetyMargin: QWKVCache.pageTokens)

            // Layer 0 (W=512, tokens=2048): liveStartPage = 6, so 2 live pages per stream = 4 pages.
            // Layer 1 (W=0): 8 live pages per stream = 16 pages. Total = 20 pages instead of 32.
            XCTAssertEqual(cache.liveStartPage(layerIndex: 0, tokenCount: tokenCount), 6)
            XCTAssertEqual(cache.liveStartPage(layerIndex: 1, tokenCount: tokenCount), 0)

            let shared = try cache.captureSharedPrefix(donor)
            XCTAssertEqual(shared.pageCount, 20)
            XCTAssertEqual(shared.memoryBytes, 20 * pageBytes)

            let adopter = try cache.allocate(adopting: shared, promptTokens: tokenCount + 256)
            XCTAssertEqual(Array(cache.pageIDs(adopter, layer: 0, kind: .k).prefix(6)), [Int32](repeating: -1, count: 6))
            XCTAssertEqual(cache.pageIDs(adopter, layer: 0, kind: .k)[8], -1)
            XCTAssertGreaterThanOrEqual(cache.pageIDs(adopter, layer: 1, kind: .k)[8], 0)

            let expectedPrivBytes = 2 * cache.serializedPlaneBytes(layerIndex: 0, tokenCount: tokenCount)
                + 2 * cache.serializedPlaneBytes(layerIndex: 1, tokenCount: tokenCount)
            XCTAssertEqual(expectedPrivBytes, cache.sharedPrefixBytes(tokenCount: tokenCount))
            let priv = try cache.capturePrivatePrefix(donor, maxBytes: expectedPrivBytes)
            XCTAssertEqual(priv.memoryBytes, expectedPrivBytes)

            let privAdopter = try cache.allocate(promptTokens: tokenCount)
            try cache.adoptPrivatePrefix(priv, into: privAdopter)
            XCTAssertEqual(Array(cache.pageIDs(privAdopter, layer: 0, kind: .k).prefix(6)), [Int32](repeating: -1, count: 6))
            XCTAssertEqual(cache.allocatedPhysicalPages(privAdopter, layer: 0), 2)
            XCTAssertEqual(cache.allocatedPhysicalPages(privAdopter, layer: 1), 8)

            cache.finish(donor)
            cache.finish(adopter)
            cache.finish(privAdopter)
            cache.releaseSharedPrefix(shared)
            XCTAssertEqual(cache.freePageCount, cache.totalPages, "mode=\(mode) must return all pages")
        }
    }
}
