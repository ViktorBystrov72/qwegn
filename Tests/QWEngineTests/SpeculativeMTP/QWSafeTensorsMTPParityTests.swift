// QWSafeTensorsMTPParityTests.swift — Synthetic MLX SafeTensors tied-head & MTP Metal parity.

import Foundation
import Metal
import XCTest
import QWAlloc
import QWCore
import QWHAL
import QWQuant
import QWRope
@testable import QWEngine

final class QWSafeTensorsMTPParityTests: XCTestCase {
    override func setUpWithError() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "SafeTensors MTP GPU parity tests require bare-metal Apple Silicon GPU"
        )
    }

    private enum STTensorSpec {
        case bf16(shape: [Int], values: [Float])
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
        precondition(bits == 4 || bits == 8, "helper supports 4-bit or 8-bit MLX codes")
        precondition(cols % group == 0 && group % 32 == 0)
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
                let rawS = scaleAmp * (0.5 + Float((s >> 16) & 0x1F) / 31.0)
                let sf16 = Float16(rawS)
                let bf16 = Float16(-halfMax * Float(sf16))
                let sv = Float(sf16)
                let bv = Float(bf16)
                sHalf[r * groups + g] = sf16.bitPattern
                bHalf[r * groups + g] = bf16.bitPattern
                for cInG in 0..<group {
                    let c = g * group + cInG
                    _ = nextRand(&s)
                    let code = UInt32((s >> 20) & mask)
                    dequant[r * cols + c] = Float(code) * sv + bv
                    let wordIdx = r * packCols + (c / codesPerWord)
                    let shift = (c % codesPerWord) * bits
                    wWords[wordIdx] |= (code << shift)
                }
            }
        }
        let wData = wWords.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in Data(raw) }
        let sData = sHalf.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in Data(raw) }
        let bData = bHalf.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in Data(raw) }
        return (wData, sData, bData, dequant)
    }

    private static func buildSyntheticSafeTensors(
        at dir: URL,
        modelType: String = "qwen3",
        trunkBits: Int = 4,
        trunkGroup: Int = 64,
        tiedHeadBits: Int? = 8,
        includeMTP: Bool = true
    ) throws -> BuiltCheckpoint {
        let h = 256, v = 256, inter = 256, qHeads = 4, kvHeads = 2, headDim = 64
        let rmsEps: Float = 1e-6, ropeTheta = 1_000_000.0
        let layerTypes = modelType == "qwen3_5" ? "\"layer_types\": [\"full_attention\"]," : ""
        let cfgJSON = """
        {
          "architectures": ["\(modelType == "qwen3_5" ? "Qwen3_5ForCausalLM" : "Qwen3ForCausalLM")"],
          "model_type": "\(modelType)",
          "vocab_size": \(v),
          "hidden_size": \(h),
          "intermediate_size": \(inter),
          "num_hidden_layers": 1,
          "mtp_num_hidden_layers": \(includeMTP ? 1 : 0),
          "num_attention_heads": \(qHeads),
          "num_key_value_heads": \(kvHeads),
          "head_dim": \(headDim),
          "max_position_embeddings": 2048,
          "rms_norm_eps": \(rmsEps),
          "rope_theta": \(ropeTheta),
          "partial_rotary_factor": 1.0,
          \(layerTypes)
          "tie_word_embeddings": true,
          "bos_token_id": 1,
          "eos_token_id": 2,
          "quantization": {
            "group_size": \(trunkGroup),
            "bits": \(trunkBits),
            "mode": "affine"
          }
        }
        """.data(using: .utf8)!

        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try cfgJSON.write(to: dir.appendingPathComponent("config.json"))
        try Data("{\"version\":\"1.0\"}".utf8).write(to: dir.appendingPathComponent("tokenizer.json"))

        var specs: [(String, STTensorSpec)] = []
        if let headBits = tiedHeadBits {
            let embScale: Float = headBits == 8 ? 0.004 : 0.04
            specs.append(("model.embed_tokens.weight", .affineMLX(
                rows: v, cols: h, bits: headBits, group: trunkGroup, seed: 11, scaleAmp: embScale)))
        } else {
            specs.append(("model.embed_tokens.weight", .bf16(
                shape: [v, h], values: detFloats(count: v * h, seedInit: 11, amp: 0.35))))
        }
        specs.append(("model.norm.weight", .bf16(
            shape: [h], values: detFloats(count: h, seedInit: 12, amp: 0.05, offset: 1.0))))

        func appendLayer(_ pfx: String, seed: UInt32) {
            let qOut = (modelType == "qwen3_5" ? 2 : 1) * qHeads * headDim
            let kvOut = kvHeads * headDim
            specs.append((pfx + "input_layernorm.weight", .bf16(
                shape: [h], values: detFloats(count: h, seedInit: seed, amp: 0.05, offset: 1.0))))
            specs.append((pfx + "post_attention_layernorm.weight", .bf16(
                shape: [h], values: detFloats(count: h, seedInit: seed + 1, amp: 0.05, offset: 1.0))))
            specs.append((pfx + "self_attn.q_proj.weight", .affineMLX(
                rows: qOut, cols: h, bits: trunkBits, group: trunkGroup, seed: seed + 2, scaleAmp: 0.015)))
            specs.append((pfx + "self_attn.k_proj.weight", .affineMLX(
                rows: kvOut, cols: h, bits: trunkBits, group: trunkGroup, seed: seed + 3, scaleAmp: 0.015)))
            specs.append((pfx + "self_attn.v_proj.weight", .affineMLX(
                rows: kvOut, cols: h, bits: trunkBits, group: trunkGroup, seed: seed + 4, scaleAmp: 0.015)))
            specs.append((pfx + "self_attn.o_proj.weight", .affineMLX(
                rows: h, cols: qHeads * headDim, bits: trunkBits, group: trunkGroup, seed: seed + 5, scaleAmp: 0.015)))
            specs.append((pfx + "self_attn.q_norm.weight", .bf16(
                shape: [headDim], values: detFloats(count: headDim, seedInit: seed + 6, amp: 0.05, offset: 1.0))))
            specs.append((pfx + "self_attn.k_norm.weight", .bf16(
                shape: [headDim], values: detFloats(count: headDim, seedInit: seed + 7, amp: 0.05, offset: 1.0))))
            specs.append((pfx + "mlp.gate_proj.weight", .affineMLX(
                rows: inter, cols: h, bits: trunkBits, group: trunkGroup, seed: seed + 8, scaleAmp: 0.015)))
            specs.append((pfx + "mlp.up_proj.weight", .affineMLX(
                rows: inter, cols: h, bits: trunkBits, group: trunkGroup, seed: seed + 9, scaleAmp: 0.015)))
            specs.append((pfx + "mlp.down_proj.weight", .affineMLX(
                rows: h, cols: inter, bits: trunkBits, group: trunkGroup, seed: seed + 10, scaleAmp: 0.015)))
        }

        appendLayer("model.layers.0.", seed: 100)
        if includeMTP {
            specs.append(("mtp.pre_fc_norm_embedding.weight", .bf16(
                shape: [h], values: detFloats(count: h, seedInit: 201, amp: 0.05, offset: 1.0))))
            specs.append(("mtp.pre_fc_norm_hidden.weight", .bf16(
                shape: [h], values: detFloats(count: h, seedInit: 202, amp: 0.05, offset: 1.0))))
            specs.append(("mtp.fc.weight", .affineMLX(
                rows: h, cols: 2 * h, bits: trunkBits, group: trunkGroup, seed: 203, scaleAmp: 0.012)))
            specs.append(("mtp.norm.weight", .bf16(
                shape: [h], values: detFloats(count: h, seedInit: 204, amp: 0.05, offset: 1.0))))
            appendLayer("mtp.layers.0.", seed: 300)
        }

        var header: [String: Any] = [:]
        var payload = Data()
        var weights: [String: [Float]] = [:]

        func appendRaw(_ name: String, dtype: String, shape: [Int], data: Data) {
            let start = payload.count
            payload.append(data)
            let end = payload.count
            header[name] = ["dtype": dtype, "shape": shape, "data_offsets": [start, end]]
        }

        for (name, spec) in specs {
            switch spec {
            case .bf16(let shape, let vals):
                weights[name] = vals
                var bf = [UInt16](repeating: 0, count: vals.count)
                for i in 0..<vals.count {
                    bf[i] = UInt16(truncatingIfNeeded: vals[i].bitPattern >> 16)
                }
                let data = bf.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in Data(raw) }
                appendRaw(name, dtype: "BF16", shape: shape, data: data)
            case .affineMLX(let rows, let cols, let bits, let group, let seed, let scaleAmp):
                let packed = packAffineMLX(
                    rows: rows, cols: cols, bits: bits, group: group, seed: seed, scaleAmp: scaleAmp)
                weights[name] = packed.dequant
                let stem = String(name.dropLast(".weight".count))
                let packCols = (cols * bits) / 32
                let groups = cols / group
                appendRaw(name, dtype: "U32", shape: [rows, packCols], data: packed.packed)
                appendRaw(stem + ".scales", dtype: "F16", shape: [rows, groups], data: packed.scales)
                appendRaw(stem + ".biases", dtype: "F16", shape: [rows, groups], data: packed.biases)
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
            qHeads: qHeads, kvHeads: kvHeads, headDim: headDim, rmsEps: rmsEps, ropeTheta: ropeTheta
        )
    }

    private static func fp32RMSNorm(_ x: [Float], gamma: [Float], eps: Float) -> [Float] {
        var sq: Float = 0
        for v in x { sq += v * v }
        let inv = 1.0 / sqrtf(sq / Float(x.count) + eps)
        var out = [Float](repeating: 0, count: x.count)
        for i in 0..<x.count { out[i] = x[i] * inv * gamma[i] }
        return out
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
        decodeSteps: Int
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
        let wQn = try XCTUnwrap(w["model.layers.0.self_attn.q_norm.weight"])
        let wKn = try XCTUnwrap(w["model.layers.0.self_attn.k_norm.weight"])
        let wG = try XCTUnwrap(w["model.layers.0.mlp.gate_proj.weight"])
        let wU = try XCTUnwrap(w["model.layers.0.mlp.up_proj.weight"])
        let wD = try XCTUnwrap(w["model.layers.0.mlp.down_proj.weight"])
        let fNorm = try XCTUnwrap(w["model.norm.weight"])

        var kCache: [[Float]] = []
        var vCache: [[Float]] = []
        let gqa = qH / kvH
        let scale = 1.0 / sqrtf(Float(d))

        func stepToken(_ tok: Int, pos: Int) throws -> [Float] {
            var x = Array(emb[(tok * h)..<((tok + 1) * h)])
            let ln1 = fp32RMSNorm(x, gamma: ln1G, eps: ckpt.rmsEps)
            var q = fp32MatVec(wQ, rows: qH * d, cols: h, x: ln1)
            var k = fp32MatVec(wK, rows: kvH * d, cols: h, x: ln1)
            let v = fp32MatVec(wV, rows: kvH * d, cols: h, x: ln1)
            for head in 0..<qH {
                let slice = Array(q[(head * d)..<((head + 1) * d)])
                var normed = fp32RMSNorm(slice, gamma: wQn, eps: ckpt.rmsEps)
                rope.apply(&normed, base: 0, dim: d, cos: ropeTables.cos, sin: ropeTables.sin, row: pos)
                q.replaceSubrange((head * d)..<((head + 1) * d), with: normed)
            }
            for head in 0..<kvH {
                let slice = Array(k[(head * d)..<((head + 1) * d)])
                var normed = fp32RMSNorm(slice, gamma: wKn, eps: ckpt.rmsEps)
                rope.apply(&normed, base: 0, dim: d, cos: ropeTables.cos, sin: ropeTables.sin, row: pos)
                k.replaceSubrange((head * d)..<((head + 1) * d), with: normed)
            }
            kCache.append(k)
            vCache.append(v)
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
            }
            let oOut = fp32MatVec(wO, rows: h, cols: qH * d, x: attOut)
            for i in 0..<h { x[i] += oOut[i] }
            let ln2 = fp32RMSNorm(x, gamma: ln2G, eps: ckpt.rmsEps)
            let gate = fp32MatVec(wG, rows: ckpt.intermediate, cols: h, x: ln2)
            let up = fp32MatVec(wU, rows: ckpt.intermediate, cols: h, x: ln2)
            var swiglu = [Float](repeating: 0, count: ckpt.intermediate)
            for i in 0..<ckpt.intermediate {
                let sig = 1.0 / (1.0 + expf(-gate[i]))
                swiglu[i] = (gate[i] * sig) * up[i]
            }
            let down = fp32MatVec(wD, rows: h, cols: ckpt.intermediate, x: swiglu)
            for i in 0..<h { x[i] += down[i] }
            let post = fp32RMSNorm(x, gamma: fNorm, eps: ckpt.rmsEps)
            return fp32MatVec(emb, rows: ckpt.vocab, cols: h, x: post)
        }

        var lastLogits: [Float] = []
        for (idx, tok) in tokens.enumerated() {
            lastLogits = try stepToken(tok, pos: idx)
        }
        let prefillLogits = lastLogits
        func argmax(_ v: [Float]) -> Int {
            var bestIdx = 0, bestVal = v[0]
            for i in 1..<v.count where v[i] > bestVal { bestVal = v[i]; bestIdx = i }
            return bestIdx
        }
        var generated: [Int] = []
        var cur = argmax(prefillLogits)
        generated.append(cur)
        var pos = tokens.count
        for _ in 1..<decodeSteps {
            let logits = try stepToken(cur, pos: pos)
            cur = argmax(logits)
            generated.append(cur)
            pos += 1
        }
        return (prefillLogits, generated)
    }

    private func assertLogitsClose(
        driver: QWEngineDriver,
        fp32Logits: [Float],
        label: String
    ) {
        let ptr = driver.sc.logits.region.pointer.advanced(by: driver.sc.logits.offset)
            .assumingMemoryBound(to: Float.self)
        let got = Array(UnsafeBufferPointer(start: ptr, count: fp32Logits.count))
        var maxDiff: Float = 0
        var gotArgmax = 0, refArgmax = 0
        for i in 0..<got.count {
            XCTAssertTrue(got[i].isFinite, "\(label) non-finite logit at \(i)")
            maxDiff = max(maxDiff, abs(got[i] - fp32Logits[i]))
            if got[i] > got[gotArgmax] { gotArgmax = i }
            if fp32Logits[i] > fp32Logits[refArgmax] { refArgmax = i }
        }
        XCTAssertLessThan(maxDiff, 0.05, "\(label) logits diverged from FP32 reference (maxDiff=\(maxDiff))")
        XCTAssertEqual(gotArgmax, refArgmax, "\(label) greedy argmax diverged from FP32 reference")
    }

    func testTiedAQ8HeadAndAQ4TrunkMatchesFP32ReferenceAndKeepsNativeAffineMTP() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-st-aq8-tied-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let ckpt = try Self.buildSyntheticSafeTensors(
            at: dir, modelType: "qwen3", trunkBits: 4, trunkGroup: 64, tiedHeadBits: 8, includeMTP: true)

        let prompt = [1, 7, 13, 29, 7, 13, 41]
        let steps = 8
        let fp32 = try Self.fp32ReferenceForward(checkpoint: ckpt, tokens: prompt, decodeSteps: steps)

        let prefillDriver = try QWEngineDriver(
            packPath: dir.path, maxTokens: 512, ropeRows: 2048,
            kvMode: .f16, mtpDrafts: 0, prefillChunkSize: 32)
        XCTAssertEqual(prefillDriver.weightFormat, .nativeAffine(bits: 4, groupSize: 64))
        XCTAssertNil(prefillDriver.weightFormatDeclineReason)
        if case .q4g(let hw) = prefillDriver.headStorage {
            XCTAssertEqual(hw.view.bits, 8)
            XCTAssertEqual(hw.view.groupSize, 64)
            XCTAssertFalse(hw.view.isRotated)
        } else {
            XCTFail("Expected quantized .q4g(AQ8_G64) tied headStorage")
        }

        let ps = try prefillDriver.makeSession(promptTokens: prompt.count)
        let firstTok = try prefillDriver.prefill(ps, tokens: prompt)
        assertLogitsClose(driver: prefillDriver, fp32Logits: fp32.prefillLogits, label: "AQ8 tied prefill")
        XCTAssertEqual(firstTok, fp32.generatedTokens[0])
        prefillDriver.finishSession(ps)

        let genBaseSess = try prefillDriver.makeSession(promptTokens: prompt.count)
        let baseGen = try prefillDriver.generate(session: genBaseSess, prompt: prompt, maxNew: steps)
        prefillDriver.finishSession(genBaseSess)
        XCTAssertEqual(baseGen, fp32.generatedTokens, "Baseline nativeAffine greedy tokens must match FP32")

        for strategy in ["mtp", "hybrid", "ngram"] {
            let mtpDriver = try QWEngineDriver(
                packPath: dir.path, maxTokens: 512, ropeRows: 2048,
                kvMode: .f16, mtpDrafts: 3, mtpAdaptiveDepth: false,
                mtpDraftStrategy: strategy, mtpQuadraticSeeding: true, prefillChunkSize: 32)
            XCTAssertEqual(mtpDriver.weightFormat, .nativeAffine(bits: 4, groupSize: 64),
                           "MTP nativeAffine must not fall back to HQ4_G32 for strategy \(strategy)")
            XCTAssertNil(mtpDriver.weightFormatDeclineReason)
            let s = try mtpDriver.makeSession(promptTokens: prompt.count)
            let tokens = try mtpDriver.generateExperimentalGreedy(session: s, prompt: prompt, maxNew: steps)
            XCTAssertEqual(s.roundRecord.mtpPhysicalRows, s.position - 1)
            if strategy != "ngram" {
                XCTAssertGreaterThan(mtpDriver.mtpDraftProposals, 0,
                                     "Strategy \(strategy) must actually invoke MTP draft proposals on .nativeAffine")
            }
            mtpDriver.finishSession(s)
            XCTAssertEqual(tokens, fp32.generatedTokens,
                           "MTP \(strategy) nativeAffine greedy tokens must match FP32 bit-exactly")
        }

        let batchDriver = try QWEngineDriver(
            packPath: dir.path, maxTokens: 512, ropeRows: 2048,
            kvMode: .f16, mtpDrafts: 2, mtpAdaptiveDepth: false, prefillChunkSize: 32)
        XCTAssertTrue(batchDriver.canRunBatchedDecode)
        XCTAssertEqual(batchDriver.weightFormat, .nativeAffine(bits: 4, groupSize: 64))
        XCTAssertFalse(batchDriver.capabilities.contains(.vision))
        let prompts = [[1, 7, 13, 29], [2, 11, 23, 37, 53]]
        let fp32Batch = try prompts.map {
            try Self.fp32ReferenceForward(checkpoint: ckpt, tokens: $0, decodeSteps: 4).generatedTokens
        }
        var sessions: [QWEngineSession] = []
        var pending: [Int] = []
        defer { for s in sessions where !s.sessionFinished { batchDriver.finishSession(s) } }
        for p in prompts {
            let sess = try batchDriver.makeSession(promptTokens: p.count)
            sessions.append(sess)
            pending.append(try batchDriver.prefill(sess, tokens: p))
        }
        var batched = pending.map { [$0] }
        for _ in 1..<4 {
            let (nextIds, _, scratch) = try batchDriver.forwardBatchStep(sessions, tokens: pending)
            for b in 0..<prompts.count {
                XCTAssertTrue(scratch.logitBuffer(b).allSatisfy(\.isFinite))
                batched[b].append(nextIds[b])
            }
            pending = nextIds
        }
        XCTAssertEqual(batched, fp32Batch, "Batched decode with nativeAffine tied AQ8 head must match FP32")
    }

    func testTiedBF16EmbedWithAQ4TrunkMatchesFP32ReferenceAndNativeAffineMTP() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-st-bf16-tied-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let ckpt = try Self.buildSyntheticSafeTensors(
            at: dir, modelType: "qwen3", trunkBits: 4, trunkGroup: 64, tiedHeadBits: nil, includeMTP: true)

        let prompt = [2, 11, 23, 37, 11, 23, 53]
        let steps = 8
        let fp32 = try Self.fp32ReferenceForward(checkpoint: ckpt, tokens: prompt, decodeSteps: steps)

        let driver = try QWEngineDriver(
            packPath: dir.path, maxTokens: 512, ropeRows: 2048,
            kvMode: .f16, mtpDrafts: 3, mtpAdaptiveDepth: false,
            mtpDraftStrategy: "hybrid", mtpQuadraticSeeding: true, prefillChunkSize: 32)
        XCTAssertEqual(driver.weightFormat, .nativeAffine(bits: 4, groupSize: 64))
        XCTAssertNil(driver.weightFormatDeclineReason)
        if case .f16(let headTensor) = driver.headStorage {
            XCTAssertEqual(headTensor.shape.elementCount, ckpt.vocab * ckpt.hidden)
        } else {
            XCTFail("Expected unquantized BF16 tied embed to serve .f16 headStorage on .nativeAffine")
        }

        let ps = try driver.makeSession(promptTokens: prompt.count)
        let firstTok = try driver.prefill(ps, tokens: prompt)
        assertLogitsClose(driver: driver, fp32Logits: fp32.prefillLogits, label: "BF16 tied prefill")
        XCTAssertEqual(firstTok, fp32.generatedTokens[0])
        driver.finishSession(ps)

        let gs = try driver.makeSession(promptTokens: prompt.count)
        let mtpTokens = try driver.generateExperimentalGreedy(session: gs, prompt: prompt, maxNew: steps)
        XCTAssertEqual(gs.roundRecord.mtpPhysicalRows, gs.position - 1)
        driver.finishSession(gs)
        XCTAssertEqual(mtpTokens, fp32.generatedTokens,
                       "Tied BF16 + AQ4_G64 trunk MTP greedy tokens must match FP32 bit-exactly")

        let req = QWGenerationRequest(prompt: prompt, maxNew: steps, sampling: .greedy)
        let reqRes = try driver.generate(request: req, onToken: nil)
        XCTAssertEqual(reqRes.tokens, fp32.generatedTokens,
                       "Standard request generation with nativeAffine MTP must match FP32 bit-exactly")
    }

    func testPlusOneNormTiedBF16SafeTensorsMatchesIncumbentBaseline() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-st-qwen35-bf16-tied-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try Self.buildSyntheticSafeTensors(
            at: dir, modelType: "qwen3_5", trunkBits: 4, trunkGroup: 64, tiedHeadBits: nil, includeMTP: true)

        let prompt = [3, 9, 17, 25, 9, 17, 31]
        let steps = 6

        let nativeDriver = try QWEngineDriver(
            packPath: dir.path, maxTokens: 512, ropeRows: 2048,
            kvMode: .q4, mtpDrafts: 3, mtpAdaptiveDepth: false,
            mtpDraftStrategy: "mtp", mtpQuadraticSeeding: true, prefillChunkSize: 32)
        XCTAssertEqual(nativeDriver.weightFormat, .nativeAffine(bits: 4, groupSize: 64))
        XCTAssertNil(nativeDriver.weightFormatDeclineReason)
        if case .f16 = nativeDriver.headStorage {} else {
            XCTFail("Expected .f16 headStorage for BF16 tied embed on qwen3_5 nativeAffine")
        }
        let sNative = try nativeDriver.makeSession(promptTokens: prompt.count)
        let nativeToks = try nativeDriver.generateExperimentalGreedy(session: sNative, prompt: prompt, maxNew: steps)
        nativeDriver.finishSession(sNative)

        setenv("QW_WEIGHT_FORMAT", "incumbent", 1)
        defer { unsetenv("QW_WEIGHT_FORMAT") }
        let incDriver = try QWEngineDriver(
            packPath: dir.path, maxTokens: 512, ropeRows: 2048,
            kvMode: .q4, mtpDrafts: 0, prefillChunkSize: 32)
        XCTAssertEqual(incDriver.weightFormat, .hq4G32)
        let sInc = try incDriver.makeSession(promptTokens: prompt.count)
        let incToks = try incDriver.generate(session: sInc, prompt: prompt, maxNew: steps)
        incDriver.finishSession(sInc)
        XCTAssertEqual(nativeToks, incToks,
                       "Plus-one RMSNorm MLX checkpoint with BF16 tied embed must preserve greedy tokens across nativeAffine MTP vs incumbent")
    }

    func testDriverResidencyEstimateMatchesPerTensorSafeTensorsHeadPrecision() throws {
        let dirAQ8 = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-est-aq8-\(UUID().uuidString)")
        let dirBF16 = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-est-bf16-\(UUID().uuidString)")
        defer {
            try? FileManager.default.removeItem(at: dirAQ8)
            try? FileManager.default.removeItem(at: dirBF16)
        }
        _ = try Self.buildSyntheticSafeTensors(
            at: dirAQ8, modelType: "qwen3", trunkBits: 4, trunkGroup: 64, tiedHeadBits: 8, includeMTP: true)
        _ = try Self.buildSyntheticSafeTensors(
            at: dirBF16, modelType: "qwen3", trunkBits: 4, trunkGroup: 64, tiedHeadBits: nil, includeMTP: true)

        let descAQ8 = try QWModelDescriptor.decode(
            configJSON: Data(contentsOf: dirAQ8.appendingPathComponent("config.json")))
        let descBF16 = try QWModelDescriptor.decode(
            configJSON: Data(contentsOf: dirBF16.appendingPathComponent("config.json")))
        let estAQ8 = QWEngineDriver.estimateDriverContextMemory(
            descriptor: descAQ8, packPath: dirAQ8.path, contextTokens: 2048, kvMode: .q4, mtpDrafts: 3)
        let estBF16 = QWEngineDriver.estimateDriverContextMemory(
            descriptor: descBF16, packPath: dirBF16.path, contextTokens: 2048, kvMode: .q4, mtpDrafts: 3)

        let h = descAQ8.hidden, v = descAQ8.vocab
        let aq8HeadBytes = v * QWWeightFormat.nativeAffine(bits: 8, groupSize: 64).rowBytes(k: h)
        let bf16HeadBytes = v * h * 2
        XCTAssertEqual(estBF16.weightsBytes - estAQ8.weightsBytes, bf16HeadBytes - aq8HeadBytes)
    }

    func testTiedAQ4G64MatchesFP32LogitsAndNativeAffineMTP() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-st-aq4-tied-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let ckpt = try Self.buildSyntheticSafeTensors(
            at: dir, modelType: "qwen3", trunkBits: 4, trunkGroup: 64, tiedHeadBits: 4, includeMTP: true)

        let prompt = [4, 15, 22, 39, 15, 22, 48]
        let steps = 6
        let fp32 = try Self.fp32ReferenceForward(checkpoint: ckpt, tokens: prompt, decodeSteps: steps)

        let driver = try QWEngineDriver(
            packPath: dir.path, maxTokens: 512, ropeRows: 2048,
            kvMode: .f16, mtpDrafts: 3, mtpAdaptiveDepth: false,
            mtpDraftStrategy: "mtp", mtpQuadraticSeeding: true, prefillChunkSize: 32)
        XCTAssertEqual(driver.weightFormat, .nativeAffine(bits: 4, groupSize: 64))
        XCTAssertNil(driver.weightFormatDeclineReason)
        if case .q4g(let hw) = driver.headStorage {
            XCTAssertEqual(hw.view.bits, 4)
            XCTAssertEqual(hw.view.groupSize, 64)
            XCTAssertFalse(hw.view.isRotated)
        } else {
            XCTFail("Expected .q4g(AQ4_G64) tied headStorage")
        }

        let ps = try driver.makeSession(promptTokens: prompt.count)
        let firstTok = try driver.prefill(ps, tokens: prompt)
        assertLogitsClose(driver: driver, fp32Logits: fp32.prefillLogits, label: "AQ4_G64 tied prefill")
        XCTAssertEqual(firstTok, fp32.generatedTokens[0])
        driver.finishSession(ps)

        let gs = try driver.makeSession(promptTokens: prompt.count)
        let tokens = try driver.generateExperimentalGreedy(session: gs, prompt: prompt, maxNew: steps)
        XCTAssertGreaterThan(driver.mtpDraftProposals, 0)
        driver.finishSession(gs)
        XCTAssertEqual(tokens, fp32.generatedTokens)
    }
}
