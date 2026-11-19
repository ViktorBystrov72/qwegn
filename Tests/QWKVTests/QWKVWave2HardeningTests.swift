// QWKVWave2HardeningTests.swift — Tests FP16 scale-zero quant, aux overflow, seq ID guards, and finish order.
import Foundation
import Metal
import XCTest
@testable import QWKV
import QWCore
import QWHAL
import QWAlloc

final class QWKVWave2HardeningTests: XCTestCase {
    private static let layers = [0, 1, 2, 3]

    func testQuantizeQ8AndQ4ZeroAndSubnormalInputNeverProducesNaN() throws {
        let kvHeads = 4
        let headDim = 256
        let total = kvHeads * headDim
        let zeros = [Float](repeating: 0.0, count: total)
        let tiny = (0..<total).map { ($0 % 2 == 0) ? Float(1e-9) : Float(-1e-9) }
        let constant = [Float](repeating: 3.25, count: total)

        for input in [zeros, tiny, constant] {
            var q8 = [Int8](repeating: 127, count: total)
            var s8 = [Float16](repeating: 0, count: kvHeads)
            try QWKVCompressedLayout.quantizeQ8(
                input: input, qOut: &q8, scalesOut: &s8, kvHeads: kvHeads, headDim: headDim
            )
            var deq8 = [Float](repeating: .nan, count: total)
            try QWKVCompressedLayout.dequantizeQ8(
                qIn: q8, scalesIn: s8, out: &deq8, kvHeads: kvHeads, headDim: headDim
            )
            for s in s8 {
                XCTAssertTrue(Float(s).isFinite && Float(s) >= 6.103515625e-5)
            }
            for v in deq8 { XCTAssertTrue(v.isFinite) }

            let groupsPerHead = (headDim + 127) / 128
            var q4 = [UInt8](repeating: 0xFF, count: total / 2)
            var s4 = [Float16](repeating: 0, count: kvHeads * groupsPerHead)
            var m4 = [Float16](repeating: 0, count: kvHeads * groupsPerHead)
            try QWKVCompressedLayout.quantizeQ4(
                input: input, qOut: &q4, scalesOut: &s4, minsOut: &m4, kvHeads: kvHeads, headDim: headDim
            )
            var deq4 = [Float](repeating: .nan, count: total)
            try QWKVCompressedLayout.dequantizeQ4(
                qIn: q4, scalesIn: s4, minsIn: m4, out: &deq4, kvHeads: kvHeads, headDim: headDim
            )
            for s in s4 {
                XCTAssertTrue(Float(s).isFinite && Float(s) >= 6.103515625e-5)
            }
            for v in deq4 { XCTAssertTrue(v.isFinite) }
        }

        var q8Zero = [Int8](repeating: 127, count: total)
        var s8Zero = [Float16](repeating: 0, count: kvHeads)
        try QWKVCompressedLayout.quantizeQ8(
            input: zeros, qOut: &q8Zero, scalesOut: &s8Zero, kvHeads: kvHeads, headDim: headDim
        )
        XCTAssertTrue(q8Zero.allSatisfy { $0 == 0 })

        var q4Const = [UInt8](repeating: 0xFF, count: total / 2)
        var s4Const = [Float16](repeating: 0, count: kvHeads * 2)
        var m4Const = [Float16](repeating: 0, count: kvHeads * 2)
        try QWKVCompressedLayout.quantizeQ4(
            input: zeros, qOut: &q4Const, scalesOut: &s4Const, minsOut: &m4Const, kvHeads: kvHeads, headDim: headDim
        )
        XCTAssertTrue(q4Const.allSatisfy { $0 == 0 })
    }

    func testInvalidSeqIDRejectedBeforeMutatingNextSeqOrFreeList() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true", "Requires Metal HAL")
        let hal = try QWHal(geometry: .qwen38_27b)
        let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: .qwen38_27b)
        let tables = QWKVCache.tableSlotBytes * 2 * Self.layers.count * QWKVCache.maxSeqs
        let poolBytes = pageBytes * QWKVCache.maxSeqs * Self.layers.count * 2
        let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: poolBytes + tables, gdnStates: 0)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: Self.layers, mode: .f16, geometry: .qwen38_27b)

        let live = try cache.allocate(promptTokens: 1)
        XCTAssertThrowsError(try cache.allocate(QWKVSeqID(QWKVCache.maxSeqs), promptTokens: 1))
        XCTAssertThrowsError(try cache.allocate(live.seq, promptTokens: 1))

        // nextSeq starts at 1 (IDs 1..<maxSeqs auto-assigned), plus explicit ID 0 gives full maxSeqs slots.
        var others: [QWKVHandle] = [try cache.allocate(0, promptTokens: 1)]
        for _ in 0..<(QWKVCache.maxSeqs - 2) {
            let h = try cache.allocate(promptTokens: 1)
            XCTAssertNotEqual(h.seq, live.seq)
            others.append(h)
        }
        XCTAssertThrowsError(try cache.allocate(promptTokens: 1))
        for h in others { cache.finish(h) }
        cache.finish(live)
        XCTAssertEqual(cache.freePageCount, cache.totalPages)
    }

    func testMultiBufferAuxSubAllocatorReceivesOverflowPools() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true", "Requires Metal HAL")
        let hal = try QWHal(geometry: .qwen38_27b)
        let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: .qwen38_27b)
        let poolBytes = qwAlignUp(pageBytes * 2, 256)
        let tables = QWKVCache.tableSlotBytes * 2 * Self.layers.count * QWKVCache.maxSeqs
        let kBuf = try XCTUnwrap(hal.device.makeBuffer(length: poolBytes * 3, options: .storageModeShared))
        let vBuf = try XCTUnwrap(hal.device.makeBuffer(length: poolBytes * 3, options: .storageModeShared))
        let auxBuf = try XCTUnwrap(hal.device.makeBuffer(length: poolBytes * 2 + tables, options: .storageModeShared))
        let kSub = QWSubAllocator(region: QWMRegion(buffer: kBuf, base: 0, length: kBuf.length))
        let vSub = QWSubAllocator(region: QWMRegion(buffer: vBuf, base: 0, length: vBuf.length))
        let auxSub = QWSubAllocator(region: QWMRegion(buffer: auxBuf, base: 0, length: auxBuf.length))

        let cache = try QWKVCache(
            hal: hal, subAllocator: kSub, vSubAllocator: vSub, auxSubAllocator: auxSub,
            layers: Self.layers, mode: .f16, geometry: .qwen38_27b
        )
        let h = try cache.allocate(promptTokens: 256)
        defer { cache.finish(h) }
        for l in [0, 1, 2] {
            let views = try cache.layerViews(h, layer: l)
            XCTAssertTrue(views.k.pool.buffer === kBuf)
            XCTAssertTrue(views.v.pool.buffer === vBuf)
        }
        let overflowViews = try cache.layerViews(h, layer: 3)
        XCTAssertTrue(overflowViews.k.pool.buffer === auxBuf)
        XCTAssertTrue(overflowViews.v.pool.buffer === auxBuf)
    }

    func testFinishReleasesRolledBackPinsAndSharedPagesWithoutLeak() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true", "Requires Metal HAL")
        let hal = try QWHal(geometry: .qwen38_27b)
        let pageBytes = QWKVCache.pageBytes(mode: .f16, geometry: .qwen38_27b)
        let tables = QWKVCache.tableSlotBytes * 2 * Self.layers.count * QWKVCache.maxSeqs
        let poolBytes = pageBytes * 6 * Self.layers.count * 2
        let arena = try QWArena(hal: hal, weights: 0, activations: 0, kv: poolBytes + tables, gdnStates: 0)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: Self.layers, mode: .f16, geometry: .qwen38_27b)

        let donor = try cache.allocate(promptTokens: 256)
        cache.markWritten(donor, upTo: 256)
        let prefix = try cache.captureSharedPrefix(donor)

        let adopter = try cache.allocate(adopting: prefix, promptTokens: 512)
        cache.markWritten(adopter, upTo: 512)
        let snap = try cache.snapshot(adopter)
        try cache.rollback(adopter, by: 256)

        cache.finish(adopter)
        cache.release(snap)
        cache.finish(donor)
        cache.releaseSharedPrefix(prefix)
        XCTAssertEqual(cache.freePageCount, cache.totalPages)
    }
}
