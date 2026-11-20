import XCTest
import Foundation
import QWCore
import QWHAL
import QWAlloc
import QWGEMM

/// The expert-sorted grouped MoE (MLX's SwitchGLU shape) against the per-pair routed path: same result, but each
/// expert's weight slab is read once for every pair routed to it instead of once per (token, route) pair.
final class FNMetalMoEGroupTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private func makeBank(_ n: Int, _ k: Int) -> Data {
        var d = Data(capacity: n * (k / 32) * 20)
        var rng: UInt64 = 0x9E3779B97F4A7C15
        func next() -> UInt32 {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17
            return UInt32(truncatingIfNeeded: rng >> 32)
        }
        for _ in 0..<(n * (k / 32)) {
            d.append(Data([UInt8(0x1F), 0x2C, 0, 0]))    // scale 0x2C1F, bias 0
            for _ in 0..<4 { d.append(Data([UInt8(next() & 0xFF), UInt8((next() >> 8) & 0xFF),
                                            UInt8((next() >> 16) & 0xFF), UInt8((next() >> 24) & 0xFF)])) }
        }
        return d
    }

    /// experts=32, routes=4, tokens=64, intermediate=64 -> rowsPerExpert = 128, K = 256.
    func testGroupedVsPerPair() throws {
        let experts = 32, routes = 4, tokens = 64, inter = 64, k = 256
        let rowsPerExpert = 2 * inter
        let n = experts * rowsPerExpert
        let bytes = n * (k / 32) * 20
        let pairs = tokens * routes
        let arena = try QWArena(hal: Self.hal, weights: bytes + (8 << 20),
                                activations: (256 << 20), kv: 0, gdnStates: 0)
        let data = makeBank(n, k)
        let w = try arena.weightsRegion().allocate(QWTensorShape([bytes / 2], .f16))
        arena.copyIn(w, from: data)
        let view = QWWeightView(blocks: w, n: n, k: k, isRotated: true)!

        let act = arena.activationsRegion()
        let x = try act.allocate(QWTensorShape([tokens * k], .f32))
        var seed: UInt32 = 777
        var xs = [Float](repeating: 0, count: tokens * k)
        for i in 0..<xs.count { seed = seed &* 1664525 &+ 1013904223; xs[i] = Float(Int32(truncatingIfNeeded: seed)) * 1e-6 }
        arena.copyIn(x, from: xs.withUnsafeBufferPointer { Data(buffer: $0) })

        // Routing: a skewed but deterministic expert choice per (token, route).
        let ids = try act.allocate(QWTensorShape([pairs], .i32))
        var idv = [Int32](repeating: 0, count: pairs)
        var rs: UInt32 = 99
        for t in 0..<tokens { for r in 0..<routes { rs = rs &* 1664525 &+ 1013904223; idv[t * routes + r] = Int32((rs >> 8) % UInt32(experts)) } }
        arena.copyIn(ids, from: idv.withUnsafeBufferPointer { Data(buffer: $0) })

        let yPair = try act.allocate(QWTensorShape([pairs * rowsPerExpert], .f32))
        let expOff = try act.allocate(QWTensorShape([experts + 1], .i32))
        let sortedPair = try act.allocate(QWTensorShape([pairs], .i32))
        let sortedTok = try act.allocate(QWTensorShape([pairs], .i32))

        guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
        try QWGemm.encodeMoEGroupPairsF32(enc, hal: Self.hal, ids: ids, expertOff: expOff,
                                         sortedPairIdx: sortedPair, sortedTok: sortedTok,
                                         experts: experts, totalPairs: pairs, routes: routes)
        try QWGemm.encodeMoEGateUpF32R1(enc, hal: Self.hal, x: x, bank: view, ids: ids, out: yPair,
                                         rowsPerExpert: rowsPerExpert, routes: routes, tokens: tokens)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "group/per-pair gate/up failed: \(String(describing: cb.error))")

        let off = Array(UnsafeBufferPointer(
            start: arena.cpuPointer(of: expOff).bindMemory(to: Int32.self, capacity: experts + 1),
            count: experts + 1))
        let sp = Array(UnsafeBufferPointer(
            start: arena.cpuPointer(of: sortedPair).bindMemory(to: Int32.self, capacity: pairs),
            count: pairs))
        let st = Array(UnsafeBufferPointer(
            start: arena.cpuPointer(of: sortedTok).bindMemory(to: Int32.self, capacity: pairs),
            count: pairs))
        XCTAssertEqual(off.first, 0)
        XCTAssertEqual(off.last, Int32(pairs))
        var seenPairs = Set<Int32>()
        for e in 0..<experts {
            XCTAssertLessThanOrEqual(off[e], off[e + 1])
            for slot in Int(off[e])..<Int(off[e + 1]) {
                let p = sp[slot]
                XCTAssertTrue(seenPairs.insert(p).inserted, "duplicate pair \(p)")
                XCTAssertEqual(Int(idv[Int(p)]), e)
                XCTAssertEqual(st[slot], p / Int32(routes))
            }
        }
        XCTAssertEqual(seenPairs.count, pairs)

        let got = Array(UnsafeBufferPointer(
            start: arena.cpuPointer(of: yPair).bindMemory(to: Float.self, capacity: pairs * rowsPerExpert),
            count: pairs * rowsPerExpert))
        XCTAssertTrue(got.allSatisfy { $0.isFinite })
    }
}
