// QWHeadArgmaxEdgeTests.swift — Sortable-key canonicalization, -Inf ranking, and fast-math finite checks.
import Foundation
import Metal
import XCTest
@testable import QWHead
import QWAlloc
import QWCore
import QWGEMM
import QWHAL

final class QWHeadArgmaxEdgeTests: XCTestCase {
    private func makeHal() throws -> QWHal {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("no Metal device") }
        return try QWHal(geometry: .qwen38_27b)
    }

    private func allocF32(_ arena: QWArena, _ vals: [Float]) throws -> QWTensorValue {
        let t = try arena.activationsRegion().allocate(QWTensorShape([vals.count], .f32))
        arena.write(t, from: vals)
        return t
    }

    func testArgmaxPreservesTotalOrderNegInfAndNaNRanking() throws {
        let hal = try makeHal()
        let head = try QWHead(hal: hal, geometry: .qwen38_27b)

        // TotalOrder: NaN < -Inf < -greatestFiniteMagnitude < -0.0 < +0.0 (shippingTarget).
        let cases: [([Float], UInt32)] = [
            ([-.infinity, -Float.greatestFiniteMagnitude, -.infinity, -.infinity], 1),
            ([-0.0, 0.0, -1.0, -.infinity], 1),
            ([0.0, -0.0, -1.0, -.infinity], 0),
            ([-0.0, -0.0, -1.0, -.infinity], 0),
            ([-.infinity, -120.5, -.infinity, .nan], 1),
            ([.nan, -.infinity, .nan, -.infinity], 1),
            ([-.infinity, -.infinity, -.infinity, -50.0], 3),
        ]
        for (idx, (logits, expected)) in cases.enumerated() {
            let n = logits.count
            let nTG = QWHead.argmaxSlots(n)
            let arena = try QWArena(hal: hal, weights: 0, activations: 64 << 10, kv: 0, gdnStates: 0)
            let act = arena.activationsRegion()
            let y = try allocF32(arena, logits)
            let kh = try act.allocate(QWTensorShape([nTG], .i32))
            let kl = try act.allocate(QWTensorShape([nTG], .i32))
            let tok = try act.allocate(QWTensorShape([1], .i32))

            guard let (cb, enc) = hal.begin() else { throw XCTSkip("no cmdbuffer") }
            try head.encodeArgmax(enc, y: y, keysHi: kh, keysLo: kl, token: tok, N: n)
            hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error, "case \(idx) GPU error")

            let got = arena.cpuPointer(of: tok).load(as: UInt32.self)
            XCTAssertEqual(got, expected, "case \(idx) expected \(expected), got \(got)")
        }
    }

    func testVariationalArgmaxFinDisablesMinLogProbWhenNegInf() throws {
        let hal = try makeHal()
        let vocab = 128
        let rows = 2
        let nTG = QWHead.argmaxSlots(vocab)
        let arena = try QWArena(hal: hal, weights: 0, activations: 256 << 10, kv: 0, gdnStates: 0)
        let act = arena.activationsRegion()
        var logits = [Float](repeating: -100.0, count: rows * vocab)
        // Row 0 checks draftTokens[1] (token 5 at -0.5, within margin=1.0 of max token 10 at 0.0);
        // -inf minLogProb must disable the probability floor and accept draft 5 on row 0.
        logits[10] = 0.0
        logits[5] = -0.5
        logits[vocab + 7] = 2.0

        let logitsT = try allocF32(arena, logits)
        let kh = try act.allocate(QWTensorShape([rows * nTG], .i32))
        let kl = try act.allocate(QWTensorShape([rows * nTG], .i32))
        let tok = try act.allocate(QWTensorShape([rows], .i32))
        let draft = try act.allocate(QWTensorShape([rows], .i32))
        arena.write(draft, from: [Int32(0), Int32(5)])

        let psoAmax = try hal.pipeline("qwg_head_argmax")
        guard let (cb, enc) = hal.begin() else { throw XCTSkip("no cmdbuffer") }
        enc.setComputePipelineState(psoAmax)
        for r in 0..<rows {
            let rY = logitsT.slice(byteOffset: r * vocab * 4, count: vocab, dtype: .f32)
            let rKh = kh.slice(byteOffset: r * nTG * 4, count: nTG, dtype: .i32)
            let rKl = kl.slice(byteOffset: r * nTG * 4, count: nTG, dtype: .i32)
            enc.bindTensors(rY, rKh, rKl, startIndex: 0)
            enc.setUInt32s(vocab, startIndex: 3)
            enc.dispatchThreadgroups(MTLSize(width: nTG, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        enc.memoryBarrier(scope: .buffers)
        let varCfg = QWVariationalAcceptance(logits: logitsT, draftTokens: draft,
                                             vocab: vocab, totalRows: rows,
                                             margin: 1.0, minLogProb: -.infinity)
        try QWGemm.encodeArgmaxFin(enc, hal: hal, keysHi: kh, keysLo: kl,
                                   token: tok, slots: nTG, rows: rows, variational: varCfg)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        let outIds = arena.read(tok, as: Int32.self, count: rows)
        XCTAssertEqual(outIds[0], 5)
        XCTAssertEqual(outIds[1], 7)
    }

    func testMtpHeadGemvArgmaxZeroWeightsPrefersLowestIndex() throws {
        let hal = try makeHal()
        let n = 20, k = 32
        let arena = try QWArena(hal: hal, weights: n * 20 + 256, activations: 64 << 10, kv: 0, gdnStates: 0)
        let blk = try arena.weightsRegion().allocate(QWTensorShape([n * 10], .f16))
        memset(arena.cpuPointer(of: blk), 0, n * 20)
        let act = arena.activationsRegion()
        let x = try act.allocate(QWTensorShape([k], .f16))
        arena.write(x, from: [Float16](repeating: Float16(1.0), count: k))
        let y = try act.allocate(QWTensorShape([n], .f32))
        let nTG = QWHead.gemvArgmaxSlots(n)
        let kh = try act.allocate(QWTensorShape([nTG], .i32))
        let kl = try act.allocate(QWTensorShape([nTG], .i32))
        let tok = try act.allocate(QWTensorShape([1], .i32))

        let pso = try hal.pipeline("qwg_mtp_head_gemv_q4g_argmax")
        guard let (cb, enc) = hal.begin() else { throw XCTSkip("no cmdbuffer") }
        enc.setComputePipelineState(pso)
        enc.bindTensors(blk, x, startIndex: 0)
        y.bind(enc, index: 3)
        enc.setUInt32s(n, k, startIndex: 4)
        enc.bindTensors(kh, kl, startIndex: 6)
        enc.setUInt32s(1, startIndex: 8)
        enc.dispatchThreadgroups(MTLSize(width: nTG, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)
        try QWGemm.encodeArgmaxFin(enc, hal: hal, keysHi: kh, keysLo: kl, token: tok, slots: nTG)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
        XCTAssertEqual(arena.cpuPointer(of: tok).load(as: UInt32.self), 0)
    }
}
