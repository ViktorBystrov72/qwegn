// QWUnionMoESpecAndInlineFwhtTests.swift — Tests Exp 2A union MoE-Spec pruning (T=2..16).
import Foundation
import Metal
import XCTest
@testable import QWGEMM
import QWAlloc
import QWCore
import QWHAL

private struct Ink2RouterParamsPOD {
    var t: UInt32
    var h: UInt32
    var numExperts: UInt32
    var topK: UInt32
    var routeScale: Float
    var moeSpecBudget: UInt32
}

final class QWUnionMoESpecAndInlineFwhtTests: XCTestCase {
    private static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private func runStep(_ encode: (any MTLComputeCommandEncoder) throws -> Void) throws {
        guard let (cb, enc) = Self.hal.begin() else { throw XCTSkip("no command buffer") }
        try encode(enc)
        Self.hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error, "GPU step failed: \(String(describing: cb.error))")
    }

    private func makeFloats(count: Int, seedInit: UInt32, scale: Float = 0.25) -> [Float] {
        var seed = seedInit
        return (0..<count).map { _ in
            seed = seed &* 1664525 &+ 1013904223
            return (Float(Int32(bitPattern: seed) >> 16) / 32768.0) * scale
        }
    }

    private func verifyUnionInvariants(
        rows: Int, topK: Int, budget: Int,
        fullIds: [Int32], fullWts: [Float],
        prunedIds: [Int32], prunedWts: [Float],
        label: String
    ) {
        XCTAssertEqual(Array(prunedIds[0..<topK]), Array(fullIds[0..<topK]), "\(label): t=0 ids must be lossless")
        for k in 0..<topK {
            XCTAssertEqual(prunedWts[k], fullWts[k], accuracy: 1e-6, "\(label): t=0 w[\(k)] must be lossless")
        }
        let rootMask = Set(fullIds[0..<topK].filter { $0 >= 0 }.map { Int($0) & 0xFFFF })
        let uniquePruned = Set(prunedIds.filter { $0 >= 0 }.map { Int($0) & 0xFFFF })
        XCTAssertLessThanOrEqual(uniquePruned.count, max(topK, budget), "\(label): exceeded expertBudget")
        for r in 1..<rows {
            var anyPruned = false
            var prunedSum: Float = 0
            var fullSum: Float = 0
            for k in 0..<topK {
                let idx = r * topK + k
                let fid = fullIds[idx]
                let pid = prunedIds[idx]
                if fullWts[idx] > 0 { fullSum += fullWts[idx] }
                if fid >= 0 && rootMask.contains(Int(fid) & 0xFFFF) {
                    XCTAssertEqual(pid, fid, "\(label): row \(r) freeMask hit k=\(k) must stay resident")
                }
                if pid < 0 {
                    anyPruned = true
                    XCTAssertEqual(prunedWts[idx], 0.0, "\(label): pruned slot wt must be 0")
                } else {
                    prunedSum += prunedWts[idx]
                }
            }
            if anyPruned && prunedSum > 0 {
                XCTAssertEqual(prunedSum, fullSum, accuracy: 1e-4, "\(label): row \(r) must renormalize sum")
            } else if !anyPruned {
                for k in 0..<topK {
                    let idx = r * topK + k
                    XCTAssertEqual(prunedWts[idx].bitPattern, fullWts[idx].bitPattern,
                                   "\(label): unpruned row \(r) must stay bit-exact")
                }
            }
        }
    }

    func testUnionMoESpecRouterRows4_8_16() throws {
        let cases: [(rows: Int, E: Int, topK: Int, budget: Int)] = [
            (4, 128, 8, 12),
            (8, 512, 8, 18),
            (16, 128, 16, 28),
            (16, 512, 8, 22),
        ]
        for (cIdx, cfg) in cases.enumerated() {
            let arena = try QWArena(hal: Self.hal, weights: 65536, activations: 2 << 20, kv: 0, gdnStates: 0)
            let act = arena.activationsRegion()
            let logitsTV = try act.allocate(QWTensorShape([cfg.rows * cfg.E], .f32))
            let biasTV = try act.allocate(QWTensorShape([cfg.E], .f32))
            let fullIdsTV = try act.allocate(QWTensorShape([cfg.rows * cfg.topK], .i32))
            let fullWtsTV = try act.allocate(QWTensorShape([cfg.rows * cfg.topK], .f32))
            let budIdsTV = try act.allocate(QWTensorShape([cfg.rows * cfg.topK], .i32))
            let budWtsTV = try act.allocate(QWTensorShape([cfg.rows * cfg.topK], .f32))

            var logits = makeFloats(count: cfg.rows * cfg.E, seedInit: UInt32(0x4100 + cIdx), scale: 3.0)
            for k in 0..<min(4, cfg.topK) {
                logits[k] += 8.0
                for r in 1..<cfg.rows where (r + k) % 2 == 0 {
                    logits[r * cfg.E + k] += 7.5
                }
            }
            arena.write(logitsTV, from: logits)
            arena.write(biasTV, from: makeFloats(count: cfg.E, seedInit: UInt32(0x5100 + cIdx), scale: 0.25))

            try runStep { enc in
                try QWGemm.encodeFnMoeRoute(
                    enc, hal: Self.hal, logits: logitsTV, idx: fullIdsTV, wts: fullWtsTV,
                    rows: cfg.rows, experts: cfg.E, topK: cfg.topK, normTopK: true, bias: biasTV)
                try QWGemm.encodeFnMoeRoute(
                    enc, hal: Self.hal, logits: logitsTV, idx: budIdsTV, wts: budWtsTV,
                    rows: cfg.rows, experts: cfg.E, topK: cfg.topK, normTopK: true, bias: biasTV,
                    expertBudget: cfg.budget)
            }
            let total = cfg.rows * cfg.topK
            verifyUnionInvariants(
                rows: cfg.rows, topK: cfg.topK, budget: cfg.budget,
                fullIds: arena.read(fullIdsTV, as: Int32.self, count: total),
                fullWts: arena.read(fullWtsTV, as: Float.self, count: total),
                prunedIds: arena.read(budIdsTV, as: Int32.self, count: total),
                prunedWts: arena.read(budWtsTV, as: Float.self, count: total),
                label: "fnMoeRoute M=\(cfg.rows) E=\(cfg.E) K=\(cfg.topK) B=\(cfg.budget)")
        }
    }

    func testInkling2RouterUnionPruneRows4_8_16() throws {
        let pSelect = try Self.hal.pipeline("qwg_ink2_router_select_topk")
        let pUnion = try Self.hal.pipeline("qwg_ink2_router_union_prune")
        let E = 128, topK = 8, H = 2048
        let routeScale: Float = 2.5
        for (idx, cfg) in [(4, 14), (8, 20), (16, 26)].enumerated() {
            let (rows, budget) = cfg
            let arena = try QWArena(hal: Self.hal, weights: 65536, activations: 1 << 20, kv: 0, gdnStates: 0)
            let act = arena.activationsRegion()
            let logitsTV = try act.allocate(QWTensorShape([rows * E], .f32))
            let biasTV = try act.allocate(QWTensorShape([E], .f32))
            let fullIdsTV = try act.allocate(QWTensorShape([rows * topK], .i32))
            let fullWtsTV = try act.allocate(QWTensorShape([rows * topK], .f32))
            let budIdsTV = try act.allocate(QWTensorShape([rows * topK], .i32))
            let budWtsTV = try act.allocate(QWTensorShape([rows * topK], .f32))

            var logits = makeFloats(count: rows * E, seedInit: UInt32(0x6100 + idx), scale: 2.5)
            for k in 0..<4 {
                logits[k] += 7.0
                for r in 1..<rows where r % 2 == 1 { logits[r * E + k] += 7.0 }
            }
            arena.write(logitsTV, from: logits)
            arena.write(biasTV, from: makeFloats(count: E, seedInit: UInt32(0x7100 + idx), scale: 0.25))

            var pZero = Ink2RouterParamsPOD(
                t: UInt32(rows), h: UInt32(H), numExperts: UInt32(E),
                topK: UInt32(topK), routeScale: routeScale, moeSpecBudget: 0)
            var pUnionArgs = pZero
            pUnionArgs.moeSpecBudget = UInt32(budget)

            try runStep { enc in
                enc.setComputePipelineState(pSelect)
                logitsTV.bind(enc, index: 0)
                biasTV.bind(enc, index: 1)
                fullIdsTV.bind(enc, index: 2)
                fullWtsTV.bind(enc, index: 3)
                enc.setBytes(&pZero, length: MemoryLayout<Ink2RouterParamsPOD>.stride, index: 4)
                enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

                enc.setComputePipelineState(pSelect)
                logitsTV.bind(enc, index: 0)
                biasTV.bind(enc, index: 1)
                budIdsTV.bind(enc, index: 2)
                budWtsTV.bind(enc, index: 3)
                enc.setBytes(&pZero, length: MemoryLayout<Ink2RouterParamsPOD>.stride, index: 4)
                enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(pUnion)
                budIdsTV.bind(enc, index: 0)
                budWtsTV.bind(enc, index: 1)
                enc.setBytes(&pUnionArgs, length: MemoryLayout<Ink2RouterParamsPOD>.stride, index: 2)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            let total = rows * topK
            verifyUnionInvariants(
                rows: rows, topK: topK, budget: budget,
                fullIds: arena.read(fullIdsTV, as: Int32.self, count: total),
                fullWts: arena.read(fullWtsTV, as: Float.self, count: total),
                prunedIds: arena.read(budIdsTV, as: Int32.self, count: total),
                prunedWts: arena.read(budWtsTV, as: Float.self, count: total),
                label: "ink2Union T=\(rows) B=\(budget)")
        }
    }
}
