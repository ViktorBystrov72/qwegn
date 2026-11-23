// QWAttentionBiasTests.swift — SafeTensors bias loading, startup detection, and E2E Metal parity.

import Foundation
import Metal
import XCTest
import QWAlloc
import QWCore
import QWHAL
import QWQuant
import QWRope
@testable import QWEngine

final class QWAttentionBiasTests: XCTestCase {
    override func setUpWithError() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "Attention bias GPU parity tests require bare-metal Apple Silicon GPU"
        )
    }

    private enum TensorSpec {
        case f16(shape: [Int], values: [Float])
        case bf16(shape: [Int], values: [Float])
        case f32(shape: [Int], values: [Float])
        case affineMLX(rows: Int, cols: Int, bits: Int, group: Int, seed: UInt32, scaleAmp: Float)
    }

    private struct BuiltCheckpoint {
        let dir: URL
        let weights: [String: [Float]]
        let vocab: Int
        let hidden: Int
        let intermediate: Int
        let qHeads: Int
        let kvHeads: Int
        let headDim: Int
        let rmsEps: Float
        let ropeTheta: Double
        let hasQKNorm: Bool
        let hasQGate: Bool
        let plusOneNorm: Bool
    }

    private static func nextRand(_ s: inout UInt32) -> UInt32 {
        s = s &* 1664525 &+ 1013904223
        return s
    }

    private static func detFloats(count: Int, seedInit: UInt32, amp: Float, offset: Float = 0) -> [Float] {
        var s = seedInit
        return (0..<count).map { _ in
            let r = nextRand(&s)
            let centered = Float(Int((r >> 16) & 0xFF) - 128) / 128.0
            let bf = UInt16(truncatingIfNeeded: (offset + centered * amp).bitPattern >> 16)
            return Float(bitPattern: UInt32(bf) << 16)
        }
    }

    private static func packAffineMLX(
        rows: Int, cols: Int, bits: Int, group: Int, seed: UInt32, scaleAmp: Float
    ) -> (packed: Data, scales: Data, biases: Data, dequant: [Float]) {
        var s = seed
        let groups = cols / group
        let packCols = (cols * bits) / 32
        var wWords = [UInt32](repeating: 0, count: rows * packCols)
        var sHalf = [UInt16](repeating: 0, count: rows * groups)
        var bHalf = [UInt16](repeating: 0, count: rows * groups)
        var dequant = [Float](repeating: 0, count: rows * cols)
        let mask = UInt32((1 << bits) - 1)
        let codesPerWord = 32 / bits
        let halfMax = Float(1 << (bits - 1)) - 0.5
        for r in 0..<rows {
            for g in 0..<groups {
                _ = nextRand(&s)
                let sf16 = Float16(scaleAmp * (0.5 + Float((s >> 16) & 0x1F) / 31.0))
                let bf16 = Float16(-halfMax * Float(sf16))
                let sv = Float(sf16), bv = Float(bf16)
                sHalf[r * groups + g] = sf16.bitPattern
                bHalf[r * groups + g] = bf16.bitPattern
                for cInG in 0..<group {
                    let c = g * group + cInG
                    _ = nextRand(&s)
                    let code = UInt32((s >> 20) & mask)
                    dequant[r * cols + c] = Float(code) * sv + bv
                    let wordIdx = r * packCols + (c / codesPerWord)
                    wWords[wordIdx] |= (code << ((c % codesPerWord) * bits))
                }
            }
        }
        return (
            wWords.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in Data(raw) },
            sHalf.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in Data(raw) },
            bHalf.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in Data(raw) },
            dequant
        )
    }

    private static func buildCheckpoint(
        at dir: URL,
        modelType: String,
        configBiasField: String?,
        configBiasValue: Bool = true,
        includeQKVBiases: Bool = true,
        omitKProjBias: Bool = false,
        includeOBias: Bool = true,
        includeMTP: Bool = false,
        mixedBiasDtypes: Bool = false
    ) throws -> BuiltCheckpoint {
        let h = 256, v = 256, inter = 256, qHeads = 4, kvHeads = 2, headDim = 64
        let rmsEps: Float = 1e-6, ropeTheta = 1_000_000.0
        let trunkBits = 4, trunkGroup = 64
        let hasQKNorm = modelType != "qwen2"
        let hasQGate = modelType == "qwen3_5"
        let plusOneNorm = false
        let arch: String = switch modelType {
        case "qwen2": "Qwen2ForCausalLM"
        case "qwen3_5": "Qwen3_5ForCausalLM"
        default: "Qwen3ForCausalLM"
        }

        var cfg: [String: Any] = [
            "architectures": [arch],
            "model_type": modelType,
            "vocab_size": v,
            "hidden_size": h,
            "intermediate_size": inter,
            "num_hidden_layers": 1,
            "num_attention_heads": qHeads,
            "num_key_value_heads": kvHeads,
            "head_dim": headDim,
            "max_position_embeddings": 2048,
            "rms_norm_eps": rmsEps,
            "rope_theta": ropeTheta,
            "partial_rotary_factor": 1.0,
            "tie_word_embeddings": true,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "quantization": ["group_size": trunkGroup, "bits": trunkBits, "mode": "affine"],
        ]
        if modelType == "qwen3_5" {
            cfg["layer_types"] = ["full_attention"]
            cfg["attn_output_gate"] = true
        }
        if includeMTP { cfg["mtp_num_hidden_layers"] = 1 }
        if let field = configBiasField { cfg[field] = configBiasValue }

        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: cfg).write(to: dir.appendingPathComponent("config.json"))
        try Data("{\"version\":\"1.0\"}".utf8).write(to: dir.appendingPathComponent("tokenizer.json"))

        var specs: [(String, TensorSpec)] = [
            ("model.embed_tokens.weight", .bf16(
                shape: [v, h], values: detFloats(count: v * h, seedInit: 11, amp: 0.35))),
            ("model.norm.weight", .bf16(
                shape: [h], values: detFloats(count: h, seedInit: 12, amp: 0.05, offset: plusOneNorm ? 0.0 : 1.0))),
        ]

        func appendLayer(_ pfx: String, seed: UInt32) {
            let qOut = (hasQGate ? 2 : 1) * qHeads * headDim
            let kvOut = kvHeads * headDim
            let normOffset: Float = plusOneNorm ? 0.0 : 1.0
            specs.append((pfx + "input_layernorm.weight", .bf16(
                shape: [h], values: detFloats(count: h, seedInit: seed, amp: 0.05, offset: normOffset))))
            specs.append((pfx + "post_attention_layernorm.weight", .bf16(
                shape: [h], values: detFloats(count: h, seedInit: seed + 1, amp: 0.05, offset: normOffset))))
            specs.append((pfx + "self_attn.q_proj.weight", .affineMLX(
                rows: qOut, cols: h, bits: trunkBits, group: trunkGroup, seed: seed + 2, scaleAmp: 0.015)))
            specs.append((pfx + "self_attn.k_proj.weight", .affineMLX(
                rows: kvOut, cols: h, bits: trunkBits, group: trunkGroup, seed: seed + 3, scaleAmp: 0.015)))
            specs.append((pfx + "self_attn.v_proj.weight", .affineMLX(
                rows: kvOut, cols: h, bits: trunkBits, group: trunkGroup, seed: seed + 4, scaleAmp: 0.015)))
            specs.append((pfx + "self_attn.o_proj.weight", .affineMLX(
                rows: h, cols: qHeads * headDim, bits: trunkBits, group: trunkGroup, seed: seed + 5, scaleAmp: 0.015)))
            if hasQKNorm {
                specs.append((pfx + "self_attn.q_norm.weight", .bf16(
                    shape: [headDim], values: detFloats(count: headDim, seedInit: seed + 6, amp: 0.05, offset: normOffset))))
                specs.append((pfx + "self_attn.k_norm.weight", .bf16(
                    shape: [headDim], values: detFloats(count: headDim, seedInit: seed + 7, amp: 0.05, offset: normOffset))))
            }
            if includeQKVBiases {
                let qb = detFloats(count: qOut, seedInit: seed + 21, amp: 0.35, offset: 0.12)
                let kb = detFloats(count: kvOut, seedInit: seed + 22, amp: 0.35, offset: -0.10)
                let vb = detFloats(count: kvOut, seedInit: seed + 23, amp: 0.40, offset: 0.18)
                specs.append((pfx + "self_attn.q_proj.bias",
                              mixedBiasDtypes ? .f16(shape: [qOut], values: qb) : .bf16(shape: [qOut], values: qb)))
                if !omitKProjBias {
                    specs.append((pfx + "self_attn.k_proj.bias", .bf16(shape: [kvOut], values: kb)))
                }
                specs.append((pfx + "self_attn.v_proj.bias",
                              mixedBiasDtypes ? .f32(shape: [kvOut], values: vb) : .bf16(shape: [kvOut], values: vb)))
            }
            if includeOBias {
                let ob = detFloats(count: h, seedInit: seed + 24, amp: 0.45, offset: 0.15)
                specs.append((pfx + "self_attn.o_proj.bias",
                              mixedBiasDtypes ? .f32(shape: [h], values: ob) : .bf16(shape: [h], values: ob)))
            }
            specs.append((pfx + "mlp.gate_proj.weight", .affineMLX(
                rows: inter, cols: h, bits: trunkBits, group: trunkGroup, seed: seed + 8, scaleAmp: 0.015)))
            specs.append((pfx + "mlp.up_proj.weight", .affineMLX(
                rows: inter, cols: h, bits: trunkBits, group: trunkGroup, seed: seed + 9, scaleAmp: 0.015)))
            specs.append((pfx + "mlp.down_proj.weight", .affineMLX(
                rows: h, cols: inter, bits: trunkBits, group: trunkGroup, seed: seed + 10, scaleAmp: 0.015)))
        }

        appendLayer("model.layers.0.", seed: 100)
        if includeMTP {
            let normOff: Float = plusOneNorm ? 0.0 : 1.0
            specs.append(("mtp.pre_fc_norm_embedding.weight", .bf16(
                shape: [h], values: detFloats(count: h, seedInit: 201, amp: 0.05, offset: normOff))))
            specs.append(("mtp.pre_fc_norm_hidden.weight", .bf16(
                shape: [h], values: detFloats(count: h, seedInit: 202, amp: 0.05, offset: normOff))))
            specs.append(("mtp.fc.weight", .affineMLX(
                rows: h, cols: 2 * h, bits: trunkBits, group: trunkGroup, seed: 203, scaleAmp: 0.012)))
            specs.append(("mtp.norm.weight", .bf16(
                shape: [h], values: detFloats(count: h, seedInit: 204, amp: 0.05, offset: normOff))))
            appendLayer("mtp.layers.0.", seed: 300)
        }

        var header: [String: Any] = [:]
        var payload = Data()
        var weights: [String: [Float]] = [:]

        func appendRaw(_ name: String, dtype: String, shape: [Int], data: Data) {
            let start = payload.count
            payload.append(data)
            header[name] = ["dtype": dtype, "shape": shape, "data_offsets": [start, payload.count]]
        }

        for (name, spec) in specs {
            switch spec {
            case .f16(let shape, let vals):
                let halves = vals.map { Float16($0) }
                weights[name] = halves.map { Float($0) }
                let bits = halves.map(\.bitPattern)
                appendRaw(name, dtype: "F16", shape: shape,
                          data: bits.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in Data(raw) })
            case .bf16(let shape, let vals):
                let bf = vals.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
                weights[name] = bf.map { Float(bitPattern: UInt32($0) << 16) }
                appendRaw(name, dtype: "BF16", shape: shape,
                          data: bf.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in Data(raw) })
            case .f32(let shape, let vals):
                weights[name] = vals
                appendRaw(name, dtype: "F32", shape: shape,
                          data: vals.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in Data(raw) })
            case .affineMLX(let rows, let cols, let bits, let group, let seed, let scaleAmp):
                let packed = packAffineMLX(
                    rows: rows, cols: cols, bits: bits, group: group, seed: seed, scaleAmp: scaleAmp)
                weights[name] = packed.dequant
                let stem = String(name.dropLast(".weight".count))
                appendRaw(name, dtype: "U32", shape: [rows, (cols * bits) / 32], data: packed.packed)
                appendRaw(stem + ".scales", dtype: "F16", shape: [rows, cols / group], data: packed.scales)
                appendRaw(stem + ".biases", dtype: "F16", shape: [rows, cols / group], data: packed.biases)
            }
        }

        var headerJSON = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        let padBytes = (8 - (headerJSON.count % 8)) % 8
        if padBytes > 0 { headerJSON.append(Data(repeating: 0x20, count: padBytes)) }
        var fileData = withUnsafeBytes(of: UInt64(headerJSON.count).littleEndian) {
            (raw: UnsafeRawBufferPointer) in Data(raw)
        }
        fileData.append(headerJSON)
        fileData.append(payload)
        try fileData.write(to: dir.appendingPathComponent("model.safetensors"))

        return BuiltCheckpoint(
            dir: dir, weights: weights, vocab: v, hidden: h, intermediate: inter,
            qHeads: qHeads, kvHeads: kvHeads, headDim: headDim, rmsEps: rmsEps, ropeTheta: ropeTheta,
            hasQKNorm: hasQKNorm, hasQGate: hasQGate, plusOneNorm: plusOneNorm
        )
    }

    private static func fp32RMSNorm(_ x: [Float], gamma: [Float], eps: Float, plusOne: Bool) -> [Float] {
        var sq: Float = 0
        for v in x { sq += v * v }
        let inv = 1.0 / sqrtf(sq / Float(x.count) + eps)
        return (0..<x.count).map { i in x[i] * inv * (plusOne ? gamma[i] + 1.0 : gamma[i]) }
    }

    private static func fp32MatVec(_ w: [Float], rows: Int, cols: Int, x: [Float]) -> [Float] {
        var y = [Float](repeating: 0, count: rows)
        for r in 0..<rows {
            var acc: Float = 0
            let base = r * cols
            for c in 0..<cols { acc += w[base + c] * x[c] }
            y[r] = acc
        }
        return y
    }

    private static func fp32ReferenceForward(
        checkpoint ckpt: BuiltCheckpoint,
        tokens: [Int],
        decodeSteps: Int,
        includeBiases: Bool = true
    ) throws -> (prefillLogits: [Float], generatedTokens: [Int]) {
        let h = ckpt.hidden, qH = ckpt.qHeads, kvH = ckpt.kvHeads, d = ckpt.headDim
        let rope = QWRopeCPU(rotaryDim: d, theta: ckpt.ropeTheta)
        let ropeTables = rope.tables(count: tokens.count + decodeSteps)
        let w = ckpt.weights
        let emb = try XCTUnwrap(w["model.embed_tokens.weight"])
        let ln1G = try XCTUnwrap(w["model.layers.0.input_layernorm.weight"])
        let ln2G = try XCTUnwrap(w["model.layers.0.post_attention_layernorm.weight"])
        let wQ = try XCTUnwrap(w["model.layers.0.self_attn.q_proj.weight"])
        let wK = try XCTUnwrap(w["model.layers.0.self_attn.k_proj.weight"])
        let wV = try XCTUnwrap(w["model.layers.0.self_attn.v_proj.weight"])
        let wO = try XCTUnwrap(w["model.layers.0.self_attn.o_proj.weight"])
        let bQ = includeBiases ? w["model.layers.0.self_attn.q_proj.bias"] : nil
        let bK = includeBiases ? w["model.layers.0.self_attn.k_proj.bias"] : nil
        let bV = includeBiases ? w["model.layers.0.self_attn.v_proj.bias"] : nil
        let bO = includeBiases ? w["model.layers.0.self_attn.o_proj.bias"] : nil
        let wQn = w["model.layers.0.self_attn.q_norm.weight"]
        let wKn = w["model.layers.0.self_attn.k_norm.weight"]
        let wG = try XCTUnwrap(w["model.layers.0.mlp.gate_proj.weight"])
        let wU = try XCTUnwrap(w["model.layers.0.mlp.up_proj.weight"])
        let wD = try XCTUnwrap(w["model.layers.0.mlp.down_proj.weight"])
        let fNorm = try XCTUnwrap(w["model.norm.weight"])

        var kCache: [[Float]] = [], vCache: [[Float]] = []
        let gqa = qH / kvH, scale = 1.0 / sqrtf(Float(d))
        let qStride = ckpt.hasQGate ? 2 : 1

        func stepToken(_ tok: Int, pos: Int) -> [Float] {
            var x = Array(emb[(tok * h)..<((tok + 1) * h)])
            let ln1 = fp32RMSNorm(x, gamma: ln1G, eps: ckpt.rmsEps, plusOne: ckpt.plusOneNorm)
            var qRaw = fp32MatVec(wQ, rows: qStride * qH * d, cols: h, x: ln1)
            var k = fp32MatVec(wK, rows: kvH * d, cols: h, x: ln1)
            var v = fp32MatVec(wV, rows: kvH * d, cols: h, x: ln1)
            if let bQ { for i in 0..<qRaw.count { qRaw[i] += bQ[i] } }
            if let bK { for i in 0..<k.count { k[i] += bK[i] } }
            if let bV { for i in 0..<v.count { v[i] += bV[i] } }

            var q = [Float](repeating: 0, count: qH * d)
            var gates = [Float](repeating: 0, count: qH * d)
            for head in 0..<qH {
                let base = head * qStride * d
                var vec = Array(qRaw[base..<(base + d)])
                if ckpt.hasQKNorm, let wQn {
                    vec = fp32RMSNorm(vec, gamma: wQn, eps: ckpt.rmsEps, plusOne: ckpt.plusOneNorm)
                }
                rope.apply(&vec, base: 0, dim: d, cos: ropeTables.cos, sin: ropeTables.sin, row: pos)
                q.replaceSubrange((head * d)..<((head + 1) * d), with: vec)
                if ckpt.hasQGate {
                    gates.replaceSubrange((head * d)..<((head + 1) * d),
                                          with: qRaw[(base + d)..<(base + 2 * d)])
                }
            }
            for head in 0..<kvH {
                var vec = Array(k[(head * d)..<((head + 1) * d)])
                if ckpt.hasQKNorm, let wKn {
                    vec = fp32RMSNorm(vec, gamma: wKn, eps: ckpt.rmsEps, plusOne: ckpt.plusOneNorm)
                }
                rope.apply(&vec, base: 0, dim: d, cos: ropeTables.cos, sin: ropeTables.sin, row: pos)
                k.replaceSubrange((head * d)..<((head + 1) * d), with: vec)
            }
            kCache.append(k); vCache.append(v)

            var attOut = [Float](repeating: 0, count: qH * d)
            for head in 0..<qH {
                let kh = head / gqa
                var scores = [Float](repeating: 0, count: kCache.count)
                var maxS: Float = -.infinity
                for t in 0..<kCache.count {
                    var dot: Float = 0
                    for i in 0..<d { dot += q[head * d + i] * kCache[t][kh * d + i] }
                    let s = dot * scale
                    scores[t] = s
                    if s > maxS { maxS = s }
                }
                var sumExp: Float = 0
                for t in 0..<scores.count {
                    scores[t] = expf(scores[t] - maxS)
                    sumExp += scores[t]
                }
                let invSum = 1.0 / sumExp
                for t in 0..<scores.count {
                    let p = scores[t] * invSum
                    for i in 0..<d { attOut[head * d + i] += p * vCache[t][kh * d + i] }
                }
                if ckpt.hasQGate {
                    for i in 0..<d {
                        attOut[head * d + i] *= 1.0 / (1.0 + expf(-gates[head * d + i]))
                    }
                }
            }
            var oOut = fp32MatVec(wO, rows: h, cols: qH * d, x: attOut)
            if let bO { for i in 0..<h { oOut[i] += bO[i] } }
            for i in 0..<h { x[i] += oOut[i] }
            let ln2 = fp32RMSNorm(x, gamma: ln2G, eps: ckpt.rmsEps, plusOne: ckpt.plusOneNorm)
            let gate = fp32MatVec(wG, rows: ckpt.intermediate, cols: h, x: ln2)
            let up = fp32MatVec(wU, rows: ckpt.intermediate, cols: h, x: ln2)
            let swiglu = (0..<ckpt.intermediate).map { i in
                (gate[i] / (1.0 + expf(-gate[i]))) * up[i]
            }
            let down = fp32MatVec(wD, rows: h, cols: ckpt.intermediate, x: swiglu)
            for i in 0..<h { x[i] += down[i] }
            let post = fp32RMSNorm(x, gamma: fNorm, eps: ckpt.rmsEps, plusOne: ckpt.plusOneNorm)
            return fp32MatVec(emb, rows: ckpt.vocab, cols: h, x: post)
        }

        var lastLogits: [Float] = []
        for (idx, tok) in tokens.enumerated() { lastLogits = stepToken(tok, pos: idx) }
        let prefillLogits = lastLogits
        func argmax(_ vals: [Float]) -> Int {
            var best = 0
            for i in 1..<vals.count where vals[i] > vals[best] { best = i }
            return best
        }
        var generated: [Int] = []
        var cur = argmax(prefillLogits)
        generated.append(cur)
        for p in 1..<decodeSteps {
            cur = argmax(stepToken(cur, pos: tokens.count + p - 1))
            generated.append(cur)
        }
        return (prefillLogits, generated)
    }

    private func assertLogitsClose(driver: QWEngineDriver, fp32Logits: [Float], label: String) {
        let ptr = driver.sc.logits.region.pointer.advanced(by: driver.sc.logits.offset)
            .assumingMemoryBound(to: Float.self)
        let got = Array(UnsafeBufferPointer(start: ptr, count: fp32Logits.count))
        var maxDiff: Float = 0
        var gotArg = 0, refArg = 0
        for i in 0..<got.count {
            XCTAssertTrue(got[i].isFinite, "\(label) non-finite logit at \(i)")
            maxDiff = max(maxDiff, abs(got[i] - fp32Logits[i]))
            if got[i] > got[gotArg] { gotArg = i }
            if fp32Logits[i] > fp32Logits[refArg] { refArg = i }
        }
        XCTAssertLessThan(maxDiff, 0.06, "\(label) diverged from FP32 reference (maxDiff=\(maxDiff))")
        XCTAssertEqual(gotArg, refArg, "\(label) argmax diverged")
    }

    private func verifyDriverAgainstFP32(
        driver: QWEngineDriver,
        checkpoint: BuiltCheckpoint,
        prompt: [Int],
        decodeSteps: Int,
        label: String,
        includeMTP: Bool = false
    ) throws {
        let fp32 = try Self.fp32ReferenceForward(checkpoint: checkpoint, tokens: prompt, decodeSteps: decodeSteps)
        let unbiased = try Self.fp32ReferenceForward(
            checkpoint: checkpoint, tokens: prompt, decodeSteps: 1, includeBiases: false)
        let maxShift = zip(fp32.prefillLogits, unbiased.prefillLogits).map { abs($0 - $1) }.max() ?? 0
        XCTAssertGreaterThan(maxShift, 0.2, "\(label) synthetic biases must shift logits measurably")

        let ps = try driver.makeSession(promptTokens: prompt.count)
        let first = try driver.prefill(ps, tokens: prompt)
        assertLogitsClose(driver: driver, fp32Logits: fp32.prefillLogits, label: "\(label) prefill")
        XCTAssertEqual(first, fp32.generatedTokens[0])
        driver.finishSession(ps)

        let ds = try driver.makeSession(promptTokens: prompt.count)
        let decoded = try driver.generate(session: ds, prompt: prompt, maxNew: decodeSteps)
        driver.finishSession(ds)
        XCTAssertEqual(decoded, fp32.generatedTokens, "\(label) decode tokens diverged")

        let prompts = [[1, 7, 13, 29], [2, 11, 23, 37, 53], [5, 19, 31], [8, 14, 27, 43, 61, 3]]
        let fp32Batch = try prompts.map {
            try Self.fp32ReferenceForward(checkpoint: checkpoint, tokens: $0, decodeSteps: 4).generatedTokens
        }
        var sessions: [QWEngineSession] = []
        var pending: [Int] = []
        defer { for s in sessions where !s.sessionFinished { driver.finishSession(s) } }
        for p in prompts {
            let sess = try driver.makeSession(promptTokens: p.count)
            sessions.append(sess)
            pending.append(try driver.prefill(sess, tokens: p))
        }
        var batched = pending.map { [$0] }
        for _ in 1..<4 {
            let (nextIds, _, scratch) = try driver.forwardBatchStep(sessions, tokens: pending)
            for b in 0..<prompts.count {
                XCTAssertTrue(scratch.logitBuffer(b).allSatisfy(\.isFinite))
                batched[b].append(nextIds[b])
            }
            pending = nextIds
        }
        XCTAssertEqual(batched, fp32Batch, "\(label) B=4 batched decode diverged")
        for s in sessions where !s.sessionFinished { driver.finishSession(s) }

        if includeMTP {
            let ms = try driver.makeSession(promptTokens: prompt.count)
            let mtpToks = try driver.generateExperimentalGreedy(session: ms, prompt: prompt, maxNew: decodeSteps)
            XCTAssertGreaterThan(driver.mtpDraftProposals, 0, "\(label) MTP must propose draft tokens")
            driver.finishSession(ms)
            XCTAssertEqual(mtpToks, fp32.generatedTokens, "\(label) MTP greedy tokens diverged")
        }
    }

    func testQwen2OmittedConfigBiasesAndMixedDtypesMatchFP32PrefillDecodeBatch() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-bias-qwen2-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let ckpt = try Self.buildCheckpoint(
            at: dir, modelType: "qwen2", configBiasField: nil,
            includeQKVBiases: true, includeOBias: false, mixedBiasDtypes: true)

        let driver = try QWEngineDriver(
            packPath: dir.path, maxTokens: 2048, ropeRows: 2048,
            kvMode: .f16, mtpDrafts: 0, prefillChunkSize: 32)
        XCTAssertTrue(driver.geometry.descriptor.hasQKVBias)
        XCTAssertFalse(driver.geometry.descriptor.hasAttnOutputBias)
        XCTAssertNotNil(driver.layersW[0].attn?.qkvBias)
        XCTAssertNil(driver.layersW[0].attn?.oBias)

        try verifyDriverAgainstFP32(
            driver: driver, checkpoint: ckpt, prompt: [1, 7, 13, 29, 7, 13, 41],
            decodeSteps: 6, label: "Qwen2 omitted-config F16/BF16/F32 QKV bias")
    }

    func testAttentionBiasTrueWithoutOBiasResolvesCleanlyAndMatchesFP32() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-bias-no-obias-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let ckpt = try Self.buildCheckpoint(
            at: dir, modelType: "qwen2", configBiasField: "attention_bias", configBiasValue: true,
            includeQKVBiases: true, includeOBias: false)

        let driver = try QWEngineDriver(
            packPath: dir.path, maxTokens: 2048, ropeRows: 2048,
            kvMode: .f16, mtpDrafts: 0, prefillChunkSize: 32)
        XCTAssertTrue(driver.geometry.descriptor.hasQKVBias)
        XCTAssertFalse(driver.geometry.descriptor.hasAttnOutputBias)
        XCTAssertNotNil(driver.layersW[0].attn?.qkvBias)
        XCTAssertNil(driver.layersW[0].attn?.oBias)

        try verifyDriverAgainstFP32(
            driver: driver, checkpoint: ckpt, prompt: [3, 9, 17, 25, 41],
            decodeSteps: 5, label: "attention_bias without o_proj.bias")
    }

    func testQwen3AndGatedQwen35WithFullQKVOBiasMatchFP32AcrossPrefillDecodeBatchAndMTP() throws {
        for modelType in ["qwen3", "qwen3_5"] {
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("qw-bias-\(modelType)-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: dir) }
            let ckpt = try Self.buildCheckpoint(
                at: dir, modelType: modelType, configBiasField: "attention_bias", configBiasValue: true,
                includeQKVBiases: true, includeOBias: true, includeMTP: true, mixedBiasDtypes: true)

            let driver = try QWEngineDriver(
                packPath: dir.path, maxTokens: 2048, ropeRows: 2048,
                kvMode: .f16, mtpDrafts: 3, mtpAdaptiveDepth: false,
                mtpDraftStrategy: "mtp", mtpQuadraticSeeding: true, prefillChunkSize: 32)
            XCTAssertTrue(driver.geometry.descriptor.hasQKVBias)
            XCTAssertTrue(driver.geometry.descriptor.hasAttnOutputBias)
            XCTAssertNotNil(driver.layersW[0].attn?.qkvBias)
            XCTAssertNotNil(driver.layersW[0].attn?.oBias)
            XCTAssertNotNil(driver.mtpWeights?.qkvBias)
            XCTAssertNotNil(driver.mtpWeights?.oBias)

            try verifyDriverAgainstFP32(
                driver: driver, checkpoint: ckpt, prompt: [2, 11, 23, 37, 11, 23, 53],
                decodeSteps: 6, label: "\(modelType) QKV+O bias", includeMTP: true)
        }
    }

    func testMissingKProjBiasFailsClosedOnDriverStartup() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-bias-missing-k-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try Self.buildCheckpoint(
            at: dir, modelType: "qwen2", configBiasField: "qkv_bias", configBiasValue: true,
            includeQKVBiases: true, omitKProjBias: true, includeOBias: false)

        XCTAssertThrowsError(try QWEngineDriver(
            packPath: dir.path, maxTokens: 512, ropeRows: 2048,
            kvMode: .f16, mtpDrafts: 0, prefillChunkSize: 32)) { error in
            XCTAssertTrue("\(error)".contains("k_proj.bias"), "Unexpected error: \(error)")
        }
    }
}
