// QWMoEGroupRouterAndClampTests.swift — GPU parity for group-limited top-k routing (E<=1024) and SwiGLU clamp modes.
import Foundation
import Metal
import XCTest
@testable import QWAlloc
@testable import QWCore
@testable import QWGEMM
@testable import QWHAL
@testable import QWQuant

final class QWMoEGroupRouterAndClampTests: XCTestCase {
    private func makeGeometry(
        experts: Int,
        topK: Int = 8,
        nGroup: Int = 8,
        topkGroup: Int = 4,
        routedScaling: Float = 2.5,
        scoreFunction: String = "sigmoid",
        normTopK: Bool = true
    ) throws -> QWModelGeometry {
        let cfg: [String: Any] = [
            "model_type": "bailing_moe_v3",
            "num_hidden_layers": 4,
            "hidden_size": 256,
            "intermediate_size": 512,
            "moe_intermediate_size": 128,
            "moe_shared_expert_intermediate_size": 64,
            "first_k_dense_replace": 1,
            "layer_group_size": 2,
            "num_experts": experts,
            "num_experts_per_tok": topK,
            "n_group": nGroup,
            "topk_group": topkGroup,
            "routed_scaling_factor": routedScaling,
            "score_function": scoreFunction,
            "moe_router_enable_expert_bias": true,
            "norm_topk_prob": normTopK,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "qk_nope_head_dim": 32,
            "qk_rope_head_dim": 32,
            "v_head_dim": 64,
            "kv_lora_rank": 64,
            "partial_rotary_factor": 0.5,
            "max_position_embeddings": 4096,
            "eos_token_id": 2,
            "vocab_size": 1024,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
        ]
        let data = try JSONSerialization.data(withJSONObject: cfg)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    private func requireGPUHal(
        experts: Int = 512,
        routedScaling: Float = 2.5,
        scoreFunction: String = "sigmoid"
    ) throws -> QWHal {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && MTLCreateSystemDefaultDevice() != nil,
            "Requires real Apple Silicon Metal GPU"
        )
        let geom = try makeGeometry(experts: experts, routedScaling: routedScaling, scoreFunction: scoreFunction)
        return try QWHal(geometry: geom, weightBasis: .rotated)
    }

    private static func sigmoid(_ x: Float) -> Float { 1.0 / (1.0 + exp(-x)) }
    private static func silu(_ x: Float) -> Float { x * sigmoid(x) }

    private static func fwht32(_ v: inout [Float]) {
        v.withUnsafeMutableBufferPointer { buf in
            for b in stride(from: 0, to: buf.count, by: 32) {
                QWQuant.fwht32OrthoBlock(buf.baseAddress! + b)
            }
        }
    }

    private static func makeDetFloats(count: Int, seedInit: UInt64, scale: Float) -> [Float] {
        var s = seedInit
        return (0..<count).map { _ in
            s = s &* 6364136223846793005 &+ 1442695040888963407
            let u = Float(Int32(truncatingIfNeeded: s >> 33)) / Float(1 << 30)
            return u * scale
        }
    }

    private static func cpuGroupTopKRouteRow(
        logits: ArraySlice<Float>,
        bias: [Float],
        experts: Int,
        topK: Int,
        nGroup: Int,
        topkGroup: Int,
        groupTopM: Int,
        routedScale: Float,
        normTopK: Bool
    ) -> (indices: [Int32], weights: [Float]) {
        let sig = logits.map(sigmoid)
        let biased = zip(sig, bias).map(+)
        let gSize = experts / max(1, nGroup)
        var allowedGroup = [Bool](repeating: nGroup <= 1, count: max(1, nGroup))
        if nGroup > 1 {
            var groupScores = [Float](repeating: 0, count: nGroup)
            for g in 0..<nGroup {
                let start = g * gSize
                let topVals = biased[start..<(start + gSize)].sorted(by: >)
                groupScores[g] = topVals.prefix(groupTopM).reduce(0, +)
            }
            let pickedGroups = (0..<nGroup).sorted { a, b in
                groupScores[a] == groupScores[b] ? a < b : groupScores[a] > groupScores[b]
            }.prefix(topkGroup)
            for g in pickedGroups { allowedGroup[g] = true }
        }
        var candidates = [(idx: Int, sel: Float)]()
        candidates.reserveCapacity(experts)
        for e in 0..<experts {
            let g = nGroup > 1 ? (e / gSize) : 0
            let sel = allowedGroup[g] ? biased[e] : -Float.greatestFiniteMagnitude
            candidates.append((e, sel))
        }
        let picked = candidates.sorted { a, b in
            a.sel == b.sel ? a.idx < b.idx : a.sel > b.sel
        }.prefix(topK)
        let rawWeights = picked.map { sig[$0.idx] }
        let denom = normTopK ? (rawWeights.reduce(0, +) + 1e-20) : 1.0
        let finalWeights = rawWeights.map { ($0 / denom) * routedScale }
        return (picked.map { Int32($0.idx) }, finalWeights)
    }

    func testGroupLimitedTopKDecodePrefillAndTieBreaking() throws {
        let routedScale: Float = 2.5
        for experts in [256, 512, 1024] {
            let hal = try requireGPUHal(experts: experts, routedScaling: routedScale, scoreFunction: "sigmoid")
            let topK = 8, nGroup = 8, topkGroup = 4, groupTopM = 2
            let bias = Self.makeDetFloats(count: experts, seedInit: UInt64(experts) &+ 17, scale: 0.4)
            let arena = try QWArena(hal: hal, weights: 0, activations: 2 << 20, kv: 0, gdnStates: 0)
            let act = arena.activationsRegion()
            let biasTV = try act.allocate(QWTensorShape([experts], .f32))
            arena.write(biasTV, from: bias)

            for rows in [1, 4, 5, 8, 19] {
                var logits = Self.makeDetFloats(
                    count: rows * experts, seedInit: UInt64(experts * 31 + rows), scale: 2.0
                )
                // Force exact ties on row 0 across groups 0..7 so lower group/expert IDs must win deterministically.
                if rows == 1 {
                    let gSize = experts / nGroup
                    for e in 0..<experts { logits[e] = Float((e % gSize == 0) ? 1.0 : -1.0) }
                }
                let effBias = (rows == 1) ? [Float](repeating: 0.125, count: experts) : bias
                arena.write(biasTV, from: effBias)

                let logitsTV = try act.allocate(QWTensorShape([rows * experts], .f32))
                let idxTV = try act.allocate(QWTensorShape([rows * topK], .i32))
                let wtsTV = try act.allocate(QWTensorShape([rows * topK], .f32))
                arena.write(logitsTV, from: logits)

                guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
                try QWGemm.encodeFnMoeRoute(
                    enc, hal: hal, logits: logitsTV, idx: idxTV, wts: wtsTV,
                    rows: rows, experts: experts, topK: topK, normTopK: true,
                    bias: biasTV, useSigmoid: true, nGroup: nGroup, topkGroup: topkGroup, groupTopM: groupTopM
                )
                hal.end(cb, enc)
                cb.waitUntilCompleted()
                XCTAssertNil(cb.error)

                let gotIdx = arena.read(idxTV, as: Int32.self, count: rows * topK)
                let gotWts = arena.read(wtsTV, as: Float.self, count: rows * topK)
                for t in 0..<rows {
                    let slice = logits[(t * experts)..<((t + 1) * experts)]
                    let want = Self.cpuGroupTopKRouteRow(
                        logits: slice, bias: effBias, experts: experts, topK: topK,
                        nGroup: nGroup, topkGroup: topkGroup, groupTopM: groupTopM,
                        routedScale: routedScale, normTopK: true
                    )
                    let rowGotIdx = Array(gotIdx[(t * topK)..<((t + 1) * topK)])
                    let rowGotWts = Array(gotWts[(t * topK)..<((t + 1) * topK)])
                    XCTAssertEqual(rowGotIdx, want.indices, "E=\(experts) T=\(rows) t=\(t) idx mismatch")
                    for k in 0..<topK {
                        XCTAssertEqual(rowGotWts[k], want.weights[k], accuracy: 1e-4,
                                       "E=\(experts) T=\(rows) t=\(t) k=\(k) weight mismatch")
                    }
                    let sumW = rowGotWts.reduce(0, +)
                    XCTAssertEqual(sumW, routedScale, accuracy: 1e-3, "unbiased scaled sum mismatch")
                }
            }
        }
    }

    func testSoftmaxRoutingUpTo1024Experts() throws {
        let experts = 1024, topK = 8
        let hal = try requireGPUHal(experts: experts, routedScaling: 1.0, scoreFunction: "softmax")
        let arena = try QWArena(hal: hal, weights: 0, activations: 2 << 20, kv: 0, gdnStates: 0)
        let act = arena.activationsRegion()
        for rows in [1, 7] {
            let logits = Self.makeDetFloats(count: rows * experts, seedInit: UInt64(9001 + rows), scale: 2.5)
            let logitsTV = try act.allocate(QWTensorShape([rows * experts], .f32))
            let idxTV = try act.allocate(QWTensorShape([rows * topK], .i32))
            let wtsTV = try act.allocate(QWTensorShape([rows * topK], .f32))
            arena.write(logitsTV, from: logits)

            guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
            try QWGemm.encodeFnMoeRoute(
                enc, hal: hal, logits: logitsTV, idx: idxTV, wts: wtsTV,
                rows: rows, experts: experts, topK: topK, normTopK: true, useSigmoid: false
            )
            hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)

            let gotIdx = arena.read(idxTV, as: Int32.self, count: rows * topK)
            let gotWts = arena.read(wtsTV, as: Float.self, count: rows * topK)
            for t in 0..<rows {
                let row = Array(logits[(t * experts)..<((t + 1) * experts)])
                let maxL = row.max() ?? 0
                let exps = row.map { exp($0 - maxL) }
                let invS = 1.0 / exps.reduce(0, +)
                let probs = exps.map { $0 * invS }
                let picked = (0..<experts).sorted { probs[$0] > probs[$1] }.prefix(topK)
                let pSum = picked.reduce(Float(0)) { $0 + probs[$1] }
                XCTAssertEqual(Array(gotIdx[(t * topK)..<((t + 1) * topK)]), picked.map(Int32.init))
                for (k, e) in picked.enumerated() {
                    XCTAssertEqual(gotWts[t * topK + k], probs[e] / pSum, accuracy: 1e-4)
                }
            }
        }
    }

    func testSwiGLUPostAndPreSiluClampAcrossAllKernels() throws {
        let hal = try requireGPUHal(experts: 32, routedScaling: 1.0, scoreFunction: "sigmoid")
        let rows = 1, inter = 64, experts = 32, topK = 4
        let arena = try QWArena(hal: hal, weights: 0, activations: 1 << 20, kv: 0, gdnStates: 0)
        let act = arena.activationsRegion()
        let gu32 = try act.allocate(QWTensorShape([rows * 2 * inter], .f32))
        let out32 = try act.allocate(QWTensorShape([rows * inter], .f32))
        let gu16 = try act.allocate(QWTensorShape([rows * 2 * inter], .f16))
        let outHi = try act.allocate(QWTensorShape([rows * inter], .f16))
        let logits = try act.allocate(QWTensorShape([rows * experts], .f32))
        let rIdx = try act.allocate(QWTensorShape([rows * topK], .i32))
        let rWts = try act.allocate(QWTensorShape([rows * topK], .f32))
        let guSh = try act.allocate(QWTensorShape([rows * 2 * inter], .f32))
        let actSh = try act.allocate(QWTensorShape([rows * inter], .f32))

        var gates = [Float](repeating: 0, count: inter)
        var ups = [Float](repeating: 0, count: inter)
        let samples: [(Float, Float)] = [
            (5.0, 1.0), (5.0, -9.0), (10.0, 2.0), (2.0, 8.0),
            (-3.0, -8.0), (4.5, 4.5), (-6.0, 1.5), (0.5, -6.5),
        ]
        for i in 0..<inter {
            let (g, u) = samples[i % samples.count]
            gates[i] = g
            ups[i] = u
        }
        let guFloat = gates + ups
        arena.write(gu32, from: guFloat)
        arena.write(gu16, from: guFloat.map(Float16.init))
        arena.write(guSh, from: guFloat)
        arena.write(logits, from: (0..<experts).map { Float($0) * 0.1 })

        for limit in [Float(4.0), Float(5.0), Float(7.0)] {
            for postSilu in [true, false] {
                let spec = QWSwiGLUClampSpec(limit: limit, postSilu: postSilu)
                let ref: [Float] = (0..<inter).map { i in
                    let g = gates[i]
                    let u = min(max(ups[i], -limit), limit)
                    let gVal: Float = postSilu
                        ? min(Self.silu(g), limit)
                        : Self.silu(min(g, limit))
                    return gVal * u
                }
                if limit == 4.0 {
                    // Explicit check that gate=5.0, up=1.0 separates post-SiLU (4.0) from pre-SiLU (~3.9281).
                    XCTAssertEqual(ref[0], postSilu ? 4.0 : Float(4.0 / (1.0 + exp(-4.0))), accuracy: 1e-4)
                }

                guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
                try QWGemm.encodeSiluMulF32(enc, hal: hal, gu: gu32, act: out32, rows: rows,
                                            intermediate: inter, swigluClamp: spec)
                try QWGemm.encodeSiluMulSplitF16(enc, hal: hal, gu: gu16, outHi: outHi, outLo: outHi,
                                                 rows: rows, intermediate: inter, swigluClamp: spec)
                try QWGemm.encodeFnMoeRoute(enc, hal: hal, logits: logits, idx: rIdx, wts: rWts,
                                            rows: 1, experts: experts, topK: topK, normTopK: true,
                                            guSh: guSh, actSh: actSh, sharedIntermediate: inter,
                                            useSigmoid: true, sharedSwigluClamp: spec)
                hal.end(cb, enc)
                cb.waitUntilCompleted()
                XCTAssertNil(cb.error)

                var got32 = arena.read(out32, as: Float.self, count: inter)
                var got16 = arena.read(outHi, as: Float16.self, count: inter).map(Float.init)
                var gotSh = arena.read(actSh, as: Float.self, count: inter)
                Self.fwht32(&got32)
                Self.fwht32(&got16)
                Self.fwht32(&gotSh)
                for i in 0..<inter {
                    XCTAssertEqual(got32[i], ref[i], accuracy: 2e-3, "f32 limit=\(limit) post=\(postSilu) i=\(i)")
                    XCTAssertEqual(got16[i], ref[i], accuracy: 2e-2, "f16 limit=\(limit) post=\(postSilu) i=\(i)")
                    XCTAssertEqual(gotSh[i], ref[i], accuracy: 2e-3, "sh limit=\(limit) post=\(postSilu) i=\(i)")
                }
            }
        }
    }
}
