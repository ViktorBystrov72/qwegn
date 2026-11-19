// QWRopeWave2Tests.swift — Regression tests for Wave 2 QWRope geometry bounds,
// non-64 rotaryDims routing, buffer(7) normOut parity, and non-gated extract guards.

import Foundation
import Metal
import XCTest
@testable import QWRope
import QWAlloc
import QWCore
import QWHAL
import QWQuant

final class QWRopeWave2Tests: XCTestCase {
    private static func makeGeometry(
        headDim: Int,
        qHeads: Int = 4,
        kvHeads: Int = 2,
        rotaryDims: Int,
        hasGate: Bool
    ) throws -> QWModelGeometry {
        let partial = Double(rotaryDims) / Double(headDim)
        let cfg: [String: Any] = [
            "model_type": "qwen3_5_text",
            "hidden_size": qHeads * headDim,
            "num_hidden_layers": 1,
            "num_attention_heads": qHeads,
            "num_key_value_heads": kvHeads,
            "head_dim": headDim,
            "intermediate_size": qHeads * headDim * 2,
            "max_position_embeddings": 1024,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "partial_rotary_factor": partial,
            "attn_output_gate": hasGate,
            "eos_token_id": 2,
            "vocab_size": 64,
        ]
        let data = try JSONSerialization.data(withJSONObject: cfg)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    private func f16(_ bits: UInt16) -> Float { QWQuant.f16tof32Bits(bits) }
    private func h16(_ val: Float) -> UInt16 { Float16(val).bitPattern }

    private func cpuRMSNorm(_ src: [Float], w: [Float], eps: Float = 1e-6) -> [Float] {
        var sq: Float = 0
        for v in src { sq += v * v }
        let rstd = 1.0 / sqrtf(sq / Float(src.count) + eps)
        return (0..<src.count).map { i in Float(Float16(src[i] * rstd * (1.0 + w[i]))) }
    }

    func testInvalidGeometryRejected() throws {
        let hal = try QWHal(geometry: .qwen38_27b)
        let tableBuf = try XCTUnwrap(hal.device.makeBuffer(length: 4096, options: .storageModeShared))
        let slab = try XCTUnwrap(hal.device.makeBuffer(length: 64, options: .storageModeShared))
        let table = QWTensorValue(
            region: QWMRegion(buffer: tableBuf, base: 0, length: tableBuf.length),
            offset: 0, shape: QWTensorShape([4, 64], .f32))

        let badHeadDim = try Self.makeGeometry(headDim: 512, qHeads: 2, kvHeads: 1, rotaryDims: 64, hasGate: false)
        XCTAssertThrowsError(
            try QWRopeApply(tableTensor: table, hal: hal, nullSlab: slab, geometry: badHeadDim)
        ) { err in
            guard case QWRopeError.badShape = err else {
                return XCTFail("Expected QWRopeError.badShape for headDim=512, got \(err)")
            }
        }

        let badRotary = try Self.makeGeometry(headDim: 128, qHeads: 4, kvHeads: 2, rotaryDims: 8, hasGate: false)
        XCTAssertThrowsError(
            try QWRopeApply(tableTensor: table, hal: hal, nullSlab: slab, geometry: badRotary)
        ) { err in
            guard case QWRopeError.badShape = err else {
                return XCTFail("Expected QWRopeError.badShape for rotaryDims=8, got \(err)")
            }
        }

        // Latent/compressed attention configs (e.g. ERNIE-5.5 headDim=512, rotaryDims=512)
        // must construct QWRopeApply cleanly in init(...) and only reject if encode* is called.
        let ernieCfg = """
        {
          "architectures": ["Ernie5_5_MoeForCausalLM"], "model_type": "ernie5_5_moe",
          "vocab_size": 1024, "hidden_size": 4096, "intermediate_size": 4096,
          "num_hidden_layers": 2, "max_position_embeddings": 1048576, "rms_norm_eps": 1e-5,
          "num_attention_heads": 64, "num_key_value_heads": 1, "head_dim": 512,
          "compress_ratios": [16, -2], "default_compress_ratio": 16,
          "q_lora_rank": 1536, "kv_lora_rank": 512,
          "compress_qk_nope_head_dim": 448, "compress_qk_rope_head_dim": 64, "compress_v_head_dim": 512,
          "o_groups": 4, "o_lora_rank": 512, "sliding_window": 512,
          "compress_rope_theta": 160000.0, "hca_rope_type": "yarn",
          "rope_scaling": { "type": "yarn", "factor": 16, "original_max_position_embeddings": 65536, "beta_fast": 32, "beta_slow": 1 },
          "rope_theta": 10000.0, "rotary_percent": 1.0, "rope_interleave": true,
          "hybrid_mla_attention": "mqa_dsa", "hybrid_mla_num_attention_heads": 64, "hybrid_mla_num_key_value_heads": 64,
          "hybrid_mla_q_lora_rank": 1536, "hybrid_mla_kv_lora_rank": 512,
          "hybrid_mla_qk_nope_head_dim": 192, "hybrid_mla_qk_rope_head_dim": 64, "hybrid_mla_v_head_dim": 256,
          "mqa_split_kv_b_proj": true, "index_n_heads": 32, "index_head_dim": 128, "index_topk": 2048,
          "use_fast_hadamard": true, "use_vha_attention": true, "add_full_attention_sink_bias": true,
          "use_gated_attn": true, "use_attn_merger": true, "gated_attn_use_q_lora": true, "eos_token_id": 2
        }
        """.data(using: .utf8)!
        let ernieGeo = QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: ernieCfg))
        XCTAssertTrue(ernieGeo.usesLatentOrCompressedAttention)
        XCTAssertGreaterThan(ernieGeo.headDim, 256)

        let ernieArena = try QWArena(hal: hal, weights: 64 << 10, activations: 4096, kv: 0, gdnStates: 0)
        let ernieTables = QWRopeTables(rows: 4, yarn: nil, geometry: ernieGeo)
        let ropeLatent = try QWRopeApply(
            tables: ernieTables, hal: hal, weights: ernieArena.weightsRegion(),
            arena: ernieArena, geometry: ernieGeo)
        let dummy = QWTensorValue(
            region: QWMRegion(buffer: tableBuf, base: 0, length: tableBuf.length),
            offset: 0, shape: QWTensorShape([1], .f16))
        guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
        defer { enc.endEncoding(); _ = cb }
        XCTAssertThrowsError(
            try ropeLatent.encodeQK(
                enc, hal: hal, q: dummy, k: dummy, wq: nil, wk: nil,
                qOut: dummy, kOut: dummy, positionStart: 0, count: 1)
        ) { err in
            guard case QWRopeError.badShape = err else {
                return XCTFail("Expected QWRopeError.badShape on encodeQK for latent headDim>256, got \(err)")
            }
        }
        XCTAssertThrowsError(
            try ropeLatent.encodeExtractQK(
                enc, hal: hal, attnY: dummy, qOut: dummy, kOut: dummy, vOut: dummy,
                wq: nil, wk: nil, positionStart: 0, count: 1)
        ) { err in
            guard case QWRopeError.badShape = err else {
                return XCTFail("Expected QWRopeError.badShape on encodeExtractQK for latent headDim>256, got \(err)")
            }
        }
    }

    func testGeneralKernelRotary32AndNormOutBuffer7Parity() throws {
        let geo = try Self.makeGeometry(headDim: 128, qHeads: 4, kvHeads: 2, rotaryDims: 32, hasGate: true)
        XCTAssertEqual(geo.headDim, 128)
        XCTAssertEqual(geo.rotaryDims, 32)
        XCTAssertEqual(geo.qProjOut, 1024)

        let hal = try QWHal(geometry: geo)
        let S = 3
        let pos0 = 5
        let qCols = geo.qProjOut
        let kCols = geo.kvHeads * geo.headDim
        let actBytes = (S * qCols * 2 + S * kCols * 2 + S * (qCols + kCols) + geo.headDim + kCols) * 2
            + (pos0 + S) * geo.rotaryDims * 4 + 4096
        let weightBytes = ((pos0 + S + 4) * geo.rotaryDims * 4) + 4096
        let arena = try QWArena(hal: hal, weights: weightBytes, activations: actBytes, kv: 0, gdnStates: 0)
        let a = arena.activationsRegion()

        let qIn = try a.allocate(QWTensorShape([S * qCols], .f16))
        let kIn = try a.allocate(QWTensorShape([S * kCols], .f16))
        let wq = try a.allocate(QWTensorShape([geo.headDim], .f16))
        let wk = try a.allocate(QWTensorShape([kCols], .f16))
        let qOut = try a.allocate(QWTensorShape([S * qCols], .f16))
        let kOut = try a.allocate(QWTensorShape([S * kCols], .f16))
        let normOut = try a.allocate(QWTensorShape([S * (qCols + kCols)], .f16))

        let qHost = (0..<(S * qCols)).map { h16(sin(Float($0 % 97) * 0.07) * 0.6) }
        let kHost = (0..<(S * kCols)).map { h16(cos(Float($0 % 89) * 0.09) * 0.5) }
        let wqHost = (0..<geo.headDim).map { h16(Float($0 % 7) * 0.05) }
        let wkHost = (0..<kCols).map { h16(Float($0 % 5) * 0.04) }
        arena.write(qIn, from: qHost)
        arena.write(kIn, from: kHost)
        arena.write(wq, from: wqHost)
        arena.write(wk, from: wkHost)

        let tables = QWRopeTables(rows: pos0 + S + 4, yarn: nil, geometry: geo)
        let rope = try QWRopeApply(tables: tables, hal: hal, weights: arena.weightsRegion(), arena: arena, geometry: geo)

        guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
        var ended = false
        defer { if !ended { enc.endEncoding() } }
        try rope.encodeQK(
            enc, hal: hal, q: qIn, k: kIn, wq: wq, wk: wk,
            qOut: qOut, kOut: kOut, normOut: normOut, positionStart: pos0, count: S)
        ended = true
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let gotQ = arena.read(qOut, as: UInt16.self, count: S * qCols).map(f16)
        let gotK = arena.read(kOut, as: UInt16.self, count: S * kCols).map(f16)
        let gotNorm = arena.read(normOut, as: UInt16.self, count: S * (qCols + kCols)).map(f16)

        let wqF = wqHost.map(f16)
        let wkF = wkHost.map(f16)
        let qF = qHost.map(f16)
        let kF = kHost.map(f16)
        let pairs = geo.rotaryDims / 2

        for s in 0..<S {
            let cs = tables.cosSinRow(pos0 + s)
            for h in 0..<geo.qHeads {
                let qBase = s * qCols + h * 2 * geo.headDim
                let x = Array(qF[qBase..<(qBase + geo.headDim)])
                let xn = cpuRMSNorm(x, w: wqF)
                for d in 0..<geo.headDim {
                    XCTAssertLessThanOrEqual(abs(gotNorm[qBase + d] - xn[d]), 3e-3, "normOut Q s=\(s) h=\(h) d=\(d)")
                    XCTAssertEqual(gotQ[qBase + geo.headDim + d], qF[qBase + geo.headDim + d], accuracy: 1e-4)
                }
                for p in 0..<pairs {
                    let c = cs[p], sn = cs[pairs + p]
                    let r0 = Float(Float16(xn[p] * c - xn[p + pairs] * sn))
                    let r1 = Float(Float16(xn[p + pairs] * c + xn[p] * sn))
                    XCTAssertLessThanOrEqual(abs(gotQ[qBase + p] - r0), 4e-3)
                    XCTAssertLessThanOrEqual(abs(gotQ[qBase + p + pairs] - r1), 4e-3)
                }
            }
            for h in 0..<geo.kvHeads {
                let kBase = s * kCols + h * geo.headDim
                let x = Array(kF[kBase..<(kBase + geo.headDim)])
                let wkSlice = Array(wkF[(h * geo.headDim)..<((h + 1) * geo.headDim)])
                let xn = cpuRMSNorm(x, w: wkSlice)
                let knOff = S * qCols + kBase
                for d in 0..<geo.headDim {
                    XCTAssertLessThanOrEqual(abs(gotNorm[knOff + d] - xn[d]), 3e-3, "normOut K s=\(s) h=\(h) d=\(d)")
                }
                for p in 0..<pairs {
                    let c = cs[p], sn = cs[pairs + p]
                    let r0 = Float(Float16(xn[p] * c - xn[p + pairs] * sn))
                    let r1 = Float(Float16(xn[p + pairs] * c + xn[p] * sn))
                    XCTAssertLessThanOrEqual(abs(gotK[kBase + p] - r0), 4e-3)
                    XCTAssertLessThanOrEqual(abs(gotK[kBase + p + pairs] - r1), 4e-3)
                }
            }
        }
    }

    func testFastExtractWithoutQGateDoesNotWritePastQOut() throws {
        let geo = try Self.makeGeometry(headDim: 256, qHeads: 2, kvHeads: 1, rotaryDims: 64, hasGate: false)
        XCTAssertEqual(geo.qProjOut, 512)
        XCTAssertFalse(geo.attnOutputGate)

        let hal = try QWHal(geometry: geo)
        let S = 2
        let inCols = geo.qProjOut + geo.kvHeads * geo.headDim * 2
        let qElems = S * geo.qProjOut
        let kElems = S * geo.kvHeads * geo.headDim
        let canaryCount = 256
        let actBytes = (S * inCols + qElems + canaryCount + kElems * 2) * 2 + 64 * 4 * 8 + 4096
        let arena = try QWArena(hal: hal, weights: 64 << 10, activations: actBytes, kv: 0, gdnStates: 0)
        let a = arena.activationsRegion()

        let attnY = try a.allocate(QWTensorShape([S * inCols], .f16))
        let qWithCanary = try a.allocate(QWTensorShape([qElems + canaryCount], .f16))
        let qOut = qWithCanary.slice(byteOffset: 0, count: qElems, dtype: .f16)
        let kOut = try a.allocate(QWTensorShape([kElems], .f16))
        let vOut = try a.allocate(QWTensorShape([kElems], .f16))

        arena.write(attnY, from: (0..<(S * inCols)).map { h16(sin(Float($0 % 53) * 0.11) * 0.4) })
        let sentinel: UInt16 = 0x5A5A
        arena.write(qWithCanary, from: [UInt16](repeating: sentinel, count: qElems + canaryCount))

        let tables = QWRopeTables(rows: 8, yarn: nil, geometry: geo)
        let rope = try QWRopeApply(tables: tables, hal: hal, weights: arena.weightsRegion(), arena: arena, geometry: geo)
        guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin"); return }
        try rope.encodeExtractQK(
            enc, hal: hal, attnY: attnY, qOut: qOut, kOut: kOut, vOut: vOut,
            wq: nil, wk: nil, positionStart: 1, count: S)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let readAll = arena.read(qWithCanary, as: UInt16.self, count: qElems + canaryCount)
        for i in qElems..<(qElems + canaryCount) {
            XCTAssertEqual(readAll[i], sentinel, "Fast extract wrote past ungated qOut at offset \(i - qElems)")
        }
    }
}
