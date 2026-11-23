// QWInkling2VisionTests.swift — Tests Inkling-2 vision config, preprocessing, 23L ViT math, and 0 B toggle.
import Foundation
import XCTest
import QWCore
@testable import QWMM
@testable import QWVision

final class QWInkling2VisionTests: XCTestCase {
    private static func bf16Bits(_ v: Float) -> UInt16 {
        var bits = v.bitPattern
        if (bits & 0x7F80_0000) == 0x7F80_0000 { return UInt16(truncatingIfNeeded: bits >> 16) }
        bits = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: bits >> 16)
    }

    private static func f32FromBF16(_ u: UInt16) -> Float { Float(bitPattern: UInt32(u) << 16) }

    private static func seededFloats(_ count: Int, seed: UInt64, scale: Double = 0.25) -> [Float] {
        var s = seed
        return (0..<count).map { _ in
            s = s &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let v = Int64(bitPattern: s) >> 40
            let f = Float(Double(v) / Double(1 << 23) * scale)
            return f32FromBF16(bf16Bits(f))
        }
    }

    private static func writeSafetensors(tensors: [(name: String, shape: [Int], data: [Float])], to url: URL) throws {
        var body = Data()
        var header: [String: Any] = [:]
        var offset = 0
        for item in tensors {
            let byteLen = item.data.count * 2
            header[item.name] = [
                "dtype": "BF16",
                "shape": item.shape,
                "data_offsets": [offset, offset + byteLen],
            ]
            for v in item.data {
                var u = bf16Bits(v).littleEndian
                withUnsafeBytes(of: &u) { body.append(contentsOf: $0) }
            }
            offset += byteLen
        }
        let jsonData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var hdrBytes = UInt64(jsonData.count).littleEndian
        var file = Data()
        withUnsafeBytes(of: &hdrBytes) { file.append(contentsOf: $0) }
        file.append(jsonData)
        file.append(body)
        try file.write(to: url)
    }

    func testDescriptorAndConfigDecodeImageToken200054() throws {
        let cfgObj: [String: Any] = [
            "model_type": "inkling2",
            "architectures": ["Inkling2ForConditionalGeneration"],
            "eos_token_id": 200006,
            "text_config": [
                "model_type": "inkling2_loop_text",
                "hidden_size": 2304,
                "intermediate_size": 1152,
                "num_hidden_layers": 24,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 1048576,
                "vocab_size": 201024,
                "unpadded_vocab_size": 200058,
                "n_routed_experts": 128,
                "num_experts_per_tok": 8,
                "loop_config": ["core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": false],
            ],
            "vision_config": [
                "model_type": "inkling2_vision",
                "text_hidden_size": 2304,
                "temporal_patch_size": 2,
                "native_patch_size": 20,
                "adapter_merge_size": 2,
                "projection_hidden_dim": 4096,
                "projector_bias": false,
                "hidden_size": 1024,
                "num_hidden_layers": 23,
                "num_attention_heads": 16,
                "mlp_ratio": 4.0,
                "posemb_grid_size": 32,
                "layer_norm_eps": 1e-5,
            ],
        ]
        let desc = try QWModelDescriptor.decode(configJSON: try JSONSerialization.data(withJSONObject: cfgObj))
        XCTAssertEqual(desc.imageTokenId, 200054)
        XCTAssertEqual(desc.videoTokenId, -1)
        let vCfg = QWVisionConfig(descriptor: desc)
        XCTAssertEqual(vCfg.architecture, .inkling2)
        XCTAssertEqual(vCfg.temporal, 1)
        XCTAssertEqual(vCfg.patch, 20)
        XCTAssertEqual(vCfg.merge, 2)
        XCTAssertEqual(vCfg.patchChannels, 1200)
        XCTAssertEqual(vCfg.projectionHiddenDim, 4096)
        XCTAssertFalse(vCfg.projectorBias)
        XCTAssertEqual(vCfg.imageTokenId, 200054)
        let spec = QWMMSpec(visionConfig: vCfg)
        XCTAssertTrue(spec.isInkling2)
        XCTAssertEqual(spec.factor, 40)
    }

    func testAlignCornersFalseInterpTapsRopeCoordsAndHWCPatchSlab() throws {
        let vCfg = QWVisionConfig(
            dim: 16, heads: 2, headDim: 8, depth: 1, mlpInner: 32,
            patch: 2, temporal: 1, merge: 2, numPosEmbeds: 16, outDim: 8,
            imageTokenId: 200054, videoTokenId: -1, ropeTheta: 10_000, eps: 1e-5,
            architecture: .inkling2, projectionHiddenDim: 32, projectorBias: false
        )
        let spec = QWMMSpec(visionConfig: vCfg)
        let grid = try QWVisionGrid(t: 1, h: 2, w: 2, spec: spec)
        XCTAssertEqual(grid.patches, 4)
        XCTAssertEqual(grid.tokens, 1)

        // With side=4 and grid.h=2, index 0 -> (0.5)*2 - 0.5 = 0.5 -> lo=0, hi=1, frac=0.5 (indices 0, 1)
        // index 1 -> (1.5)*2 - 0.5 = 2.5 -> lo=2, hi=3, frac=0.5 (indices 2, 3)
        let taps = qwInterpTaps(grid: grid)
        XCTAssertEqual(Array(taps.indices.prefix(4)), [0, 1, 4, 5])
        for i in 0..<4 { XCTAssertEqual(taps.weights[i], 0.25, accuracy: 1e-6) }

        // Coords are (col+1, row+1) in merge-block order: (0,0)->(1,1), (0,1)->(2,1), (1,0)->(1,2), (1,1)->(2,2)
        XCTAssertEqual(qwVisionRopePositions(grid: grid), [1, 1, 2, 1, 1, 2, 2, 2])

        // 4x4 RGB image in HWC with unique values per pixel (y * 4 + x)
        var rgb = [Float](repeating: 0, count: 4 * 4 * 3)
        for y in 0..<4 {
            for x in 0..<4 {
                let p = (y * 4 + x) * 3
                rgb[p] = Float(y * 10 + x)
                rgb[p + 1] = Float(y * 10 + x) + 0.1
                rgb[p + 2] = Float(y * 10 + x) + 0.2
            }
        }
        let slab = try qwBuildPatchSlab(normalizedRGB: rgb, height: 4, width: 4, grid: grid)
        XCTAssertEqual(slab.count, 4 * 12)
        // Patch 1 (mr=0, mc=1) top-left pixel is (y=0, x=2)
        XCTAssertEqual(slab[12], 2.0, accuracy: 1e-6)
        // Patch 2 (mr=1, mc=0) bottom-right pixel is (y=3, x=1)
        XCTAssertEqual(slab[2 * 12 + 3 * 3], 31.0, accuracy: 1e-6)
    }

    func testSyntheticViTForwardParityAndResidencyToggle() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ink2_vis_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let d = 16, heads = 2, hd = 8, mlp = 32, patch = 2, merge = 2
        let posSide = 4, numPos = 16, projH = 24, outDim = 64, eps: Float = 1e-5
        let patchChan = 3 * patch * patch
        let merged = merge * merge * d

        let patchW = Self.seededFloats(d * patchChan, seed: 11)
        let posEmbed = Self.seededFloats(numPos * d, seed: 12)
        let classEmbed = Self.seededFloats(d, seed: 13)
        let clsPosEmbed = Self.seededFloats(d, seed: 14)
        let preG = Self.seededFloats(d, seed: 15, scale: 1.0)
        let preB = Self.seededFloats(d, seed: 16, scale: 0.1)
        let ln1G = Self.seededFloats(d, seed: 17, scale: 1.0)
        let ln1B = Self.seededFloats(d, seed: 18, scale: 0.1)
        let ln2G = Self.seededFloats(d, seed: 19, scale: 1.0)
        let ln2B = Self.seededFloats(d, seed: 20, scale: 0.1)
        let lam1 = Self.seededFloats(d, seed: 21, scale: 0.5)
        let lam2 = Self.seededFloats(d, seed: 22, scale: 0.5)
        let qW = Self.seededFloats(d * d, seed: 23), qB = Self.seededFloats(d, seed: 24)
        let kW = Self.seededFloats(d * d, seed: 25), kB = Self.seededFloats(d, seed: 26)
        let vW = Self.seededFloats(d * d, seed: 27), vB = Self.seededFloats(d, seed: 28)
        let oW = Self.seededFloats(d * d, seed: 29), oB = Self.seededFloats(d, seed: 30)
        let fc1W = Self.seededFloats(mlp * d, seed: 31), fc1B = Self.seededFloats(mlp, seed: 32)
        let fc2W = Self.seededFloats(d * mlp, seed: 33), fc2B = Self.seededFloats(d, seed: 34)
        let projInW = Self.seededFloats(projH * merged, seed: 35)
        let projOutW = Self.seededFloats(outDim * projH, seed: 36)

        let tensors: [(String, [Int], [Float])] = [
            ("vision_tower.embeddings.patch_embedding.weight", [d, patchChan], patchW),
            ("vision_tower.embeddings.position_embedding", [numPos, d], posEmbed),
            ("vision_tower.embeddings.class_embedding", [d], classEmbed),
            ("vision_tower.embeddings.cls_position_embedding", [d], clsPosEmbed),
            ("vision_tower.pre_layernorm.weight", [d], preG),
            ("vision_tower.pre_layernorm.bias", [d], preB),
            ("vision_tower.layers.0.layernorm_before.weight", [d], ln1G),
            ("vision_tower.layers.0.layernorm_before.bias", [d], ln1B),
            ("vision_tower.layers.0.layernorm_after.weight", [d], ln2G),
            ("vision_tower.layers.0.layernorm_after.bias", [d], ln2B),
            ("vision_tower.layers.0.lambda_1", [d], lam1),
            ("vision_tower.layers.0.lambda_2", [d], lam2),
            ("vision_tower.layers.0.self_attn.q_proj.weight", [d, d], qW),
            ("vision_tower.layers.0.self_attn.q_proj.bias", [d], qB),
            ("vision_tower.layers.0.self_attn.k_proj.weight", [d, d], kW),
            ("vision_tower.layers.0.self_attn.k_proj.bias", [d], kB),
            ("vision_tower.layers.0.self_attn.v_proj.weight", [d, d], vW),
            ("vision_tower.layers.0.self_attn.v_proj.bias", [d], vB),
            ("vision_tower.layers.0.self_attn.out_proj.weight", [d, d], oW),
            ("vision_tower.layers.0.self_attn.out_proj.bias", [d], oB),
            ("vision_tower.layers.0.mlp.fc1.weight", [mlp, d], fc1W),
            ("vision_tower.layers.0.mlp.fc1.bias", [mlp], fc1B),
            ("vision_tower.layers.0.mlp.fc2.weight", [d, mlp], fc2W),
            ("vision_tower.layers.0.mlp.fc2.bias", [d], fc2B),
            ("vision_tower.proj_in.weight", [projH, merged], projInW),
            ("vision_tower.proj_out.weight", [outDim, projH], projOutW),
        ]
        try Self.writeSafetensors(tensors: tensors, to: dir.appendingPathComponent("model.safetensors"))

        let cfgObj: [String: Any] = [
            "model_type": "inkling2",
            "architectures": ["Inkling2ForConditionalGeneration"],
            "eos_token_id": 200006,
            "text_config": [
                "model_type": "inkling2_loop_text",
                "hidden_size": outDim,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 1024,
                "vocab_size": 100,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "loop_config": ["core_start": 0, "core_end": 1, "num_passes": 1, "gate_rank": 4, "is_varloop": false],
            ],
            "vision_config": [
                "model_type": "inkling2_vision", "text_hidden_size": outDim,
                "native_patch_size": patch, "adapter_merge_size": merge,
                "projection_hidden_dim": projH, "projector_bias": false,
                "hidden_size": d, "num_hidden_layers": 1, "num_attention_heads": heads,
                "mlp_ratio": Double(mlp) / Double(d), "posemb_grid_size": posSide,
                "layer_norm_eps": Double(eps),
            ],
        ]
        let cfgData = try JSONSerialization.data(withJSONObject: cfgObj)
        _ = try QWModelDescriptor.decode(configJSON: cfgData)
        try cfgData.write(to: dir.appendingPathComponent("config.json"))

        let backend = try QWVisionOnDemandBackend(modelPath: dir.path, keepResident: false, allowOnDemand: true)
        XCTAssertEqual(backend.visionConfig.architecture, .inkling2)
        XCTAssertEqual(backend.visionConfig.heads, heads)
        XCTAssertEqual(backend.visionConfig.headDim, hd)
        XCTAssertFalse(backend.isLoaded)
        XCTAssertEqual(backend.residentBytes, 0)

        try backend.setVisionLoaded(true)
        XCTAssertTrue(backend.isLoaded)
        XCTAssertGreaterThan(backend.residentBytes, 0)

        let spec = QWMMSpec(visionConfig: backend.visionConfig)
        let grid = try QWVisionGrid(t: 1, h: 2, w: 2, spec: spec)
        let patches = Self.seededFloats(grid.patches * patchChan, seed: 99)
        let got = try backend.encodeSync(patches: patches, grid: grid.plan, bins: grid.interpBins, posIds: grid.ropePositions)
        XCTAssertEqual(got.count, grid.tokens * outDim)

        // Direct Float64 reference check
        let nPatch = grid.patches, nSeq = nPatch + 1
        func ln(_ x: [[Double]], _ g: [Float], _ b: [Float]) -> [[Double]] {
            x.map { row in
                let m = row.reduce(0, +) / Double(d)
                let v = row.reduce(0.0) { $0 + ($1 - m) * ($1 - m) } / Double(d)
                let inv = 1.0 / sqrt(v + Double(eps))
                return (0..<d).map { ((row[$0] - m) * inv) * Double(g[$0]) + Double(b[$0]) }
            }
        }
        func linear(_ x: [[Double]], _ w: [Float], _ b: [Float]?, nIn: Int, nOut: Int) -> [[Double]] {
            x.map { row in
                (0..<nOut).map { o in
                    var acc = b.map { Double($0[o]) } ?? 0.0
                    for k in 0..<nIn { acc += row[k] * Double(w[o * nIn + k]) }
                    return acc
                }
            }
        }
        var xRows: [[Double]] = [(0..<d).map { Double(classEmbed[$0]) + Double(clsPosEmbed[$0]) }]
        let peOut = linear((0..<nPatch).map { p in (0..<patchChan).map { Double(patches[p * patchChan + $0]) } },
                           patchW, nil, nIn: patchChan, nOut: d)
        for p in 0..<nPatch {
            var r = peOut[p]
            for c in 0..<4 {
                let idx = Int(grid.interpBins.idx[p * 4 + c])
                let wt = Double(grid.interpBins.wgt[p * 4 + c])
                for k in 0..<d { r[k] += wt * Double(posEmbed[idx * d + k]) }
            }
            xRows.append(r)
        }
        xRows = ln(xRows, preG, preB)
        let spatial = hd / 2
        let invFreq = (0..<(spatial / 2)).map { i in pow(10_000.0, -Double(2 * i) / Double(spatial)) }
        let coords: [(Double, Double)] = [(0, 0)] + (0..<nPatch).map {
            (Double(grid.ropePositions[$0 * 2]), Double(grid.ropePositions[$0 * 2 + 1]))
        }
        let n1 = ln(xRows, ln1G, ln1B)
        let q = linear(n1, qW, qB, nIn: d, nOut: d)
        let k = linear(n1, kW, kB, nIn: d, nOut: d)
        let v = linear(n1, vW, vB, nIn: d, nOut: d)
        func rope(_ vec: [Double], head: Int, coord: (Double, Double)) -> [Double] {
            var res = Array(vec[(head * hd)..<((head + 1) * hd)])
            for axis in 0..<2 {
                let cVal = axis == 0 ? coord.0 : coord.1
                for i in 0..<(spatial / 2) {
                    let ang = cVal * invFreq[i]
                    let cc = cos(ang), ss = sin(ang)
                    let j0 = axis * spatial + 2 * i, j1 = j0 + 1
                    let r0 = res[j0], r1 = res[j1]
                    res[j0] = r0 * cc - r1 * ss
                    res[j1] = r1 * cc + r0 * ss
                }
            }
            return res
        }
        var attnOut = Array(repeating: Array(repeating: 0.0, count: d), count: nSeq)
        let scale = 1.0 / sqrt(Double(hd))
        for h in 0..<heads {
            let qr = (0..<nSeq).map { rope(q[$0], head: h, coord: coords[$0]) }
            let kr = (0..<nSeq).map { rope(k[$0], head: h, coord: coords[$0]) }
            for i in 0..<nSeq {
                let sc = (0..<nSeq).map { j in zip(qr[i], kr[j]).reduce(0.0) { $0 + $1.0 * $1.1 } * scale }
                let mx = sc.max() ?? 0
                let ex = sc.map { exp($0 - mx) }
                let sm = ex.reduce(0, +)
                for cIdx in 0..<hd {
                    attnOut[i][h * hd + cIdx] = (0..<nSeq).reduce(0.0) { $0 + (ex[$1] / sm) * v[$1][h * hd + cIdx] }
                }
            }
        }
        let proj = linear(attnOut, oW, oB, nIn: d, nOut: d)
        for i in 0..<nSeq { for k in 0..<d { xRows[i][k] += Double(lam1[k]) * proj[i][k] } }
        let n2 = ln(xRows, ln2G, ln2B)
        let h1 = linear(n2, fc1W, fc1B, nIn: d, nOut: mlp).map { row in
            row.map { 0.5 * $0 * (1.0 + erf($0 * 0.7071067811865476)) }
        }
        let h2 = linear(h1, fc2W, fc2B, nIn: mlp, nOut: d)
        for i in 0..<nSeq { for k in 0..<d { xRows[i][k] += Double(lam2[k]) * h2[i][k] } }
        let mergedRow = [Array(xRows[1...].flatMap { $0 })]
        let pIn = linear(mergedRow, projInW, nil, nIn: merged, nOut: projH).map { row in
            row.map { 0.5 * $0 * (1.0 + erf($0 * 0.7071067811865476)) }
        }
        let want = linear(pIn, projOutW, nil, nIn: projH, nOut: outDim)[0]
        for i in 0..<outDim { XCTAssertEqual(Double(got[i]), want[i], accuracy: 1e-4) }

        try backend.setVisionLoaded(false)
        XCTAssertFalse(backend.isLoaded)
        XCTAssertEqual(backend.residentBytes, 0)
    }
}
