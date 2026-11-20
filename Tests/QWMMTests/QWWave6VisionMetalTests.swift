// QWWave6VisionMetalTests.swift — Parity, O(N) FlashAttention, Law 4/4b/15
// residency, CPU fallback, and vDSP patch-slab tests for Wave 6 Metal ViT.
import Accelerate
import Foundation
import Metal
import XCTest
@testable import QWMM
@testable import QWVision

final class QWWave6VisionMetalTests: XCTestCase {
    private static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()

    private static func skipUnlessMetalAvailable() throws -> (device: any MTLDevice, library: any MTLLibrary) {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true",
            "Metal ViT kernel tests run on bare-metal Apple Silicon"
        )
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device available")
        }
        try XCTSkipIf(
            dev.name.localizedCaseInsensitiveContains("paravirtual"),
            "Skipping Metal ViT compute test on Paravirtual GPU"
        )
        let libURL = repoRoot.appendingPathComponent("Sources/QWMetal/kernel.metallib")
        guard FileManager.default.fileExists(atPath: libURL.path) else {
            throw XCTSkip("Missing kernel.metallib at \(libURL.path)")
        }
        let lib = try dev.makeLibrary(URL: libURL)
        return (dev, lib)
    }

    private struct LCG {
        var state: UInt64
        mutating func nextUniform() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let u = Double((state >> 11) & 0x1FFFFF) / Double(1 << 21)
            return Float(u * 2.0 - 1.0)
        }
    }

    private static func errorStats(_ a: [Float], _ b: [Float]) -> (maxAbs: Float, cosSim: Double) {
        precondition(a.count == b.count && !a.isEmpty)
        var maxAbs: Float = 0
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0..<a.count {
            let da = Double(a[i]), db = Double(b[i])
            maxAbs = max(maxAbs, abs(a[i] - b[i]))
            dot += da * db
            na += da * da
            nb += db * db
        }
        let denom = (na * nb).squareRoot()
        let cosSim = denom > 0 ? (dot / denom) : (na == 0 && nb == 0 ? 1.0 : 0.0)
        return (maxAbs, cosSim)
    }

    private static func makeF16Buffer(_ dev: any MTLDevice, _ values: [Float]) throws -> any MTLBuffer {
        let buf = try XCTUnwrap(dev.makeBuffer(length: max(256, values.count * 2), options: .storageModeShared))
        let ptr = buf.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<values.count { ptr[i] = Float16(values[i]) }
        return buf
    }

    private static func readF16Buffer(_ buf: any MTLBuffer, count: Int) -> [Float] {
        let ptr = buf.contents().assumingMemoryBound(to: Float16.self)
        return (0..<count).map { Float(ptr[$0]) }
    }

    private static func setU32(_ enc: any MTLComputeCommandEncoder, _ v: Int, index: Int) {
        var u = UInt32(v)
        enc.setBytes(&u, length: 4, index: index)
    }

    private static func setF32(_ enc: any MTLComputeCommandEncoder, _ v: Float, index: Int) {
        var f = v
        enc.setBytes(&f, length: 4, index: index)
    }

    private static func writeSyntheticVisionDir(config: QWVisionConfig, seed: UInt64) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-w6-vis-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let cfgObj: [String: Any] = [
            "model_type": "qwen3_5",
            "text_config": [
                "model_type": "qwen3_5_text",
                "hidden_size": max(64, config.outDim),
                "vocab_size": 1024,
                "num_hidden_layers": 1,
                "rms_norm_eps": Double(config.eps),
                "max_position_embeddings": 2048,
                "num_attention_heads": max(1, config.heads),
                "num_key_value_heads": max(1, config.heads),
                "intermediate_size": max(64, config.mlpInner),
                "eos_token_id": 0
            ],
            "vision_config": [
                "depth": config.depth,
                "hidden_size": config.dim,
                "num_heads": config.heads,
                "intermediate_size": config.mlpInner,
                "patch_size": config.patch,
                "temporal_patch_size": config.temporal,
                "spatial_merge_size": config.merge,
                "num_position_embeddings": config.numPosEmbeds,
                "out_hidden_size": config.outDim
            ]
        ]
        try JSONSerialization.data(withJSONObject: cfgObj)
            .write(to: dir.appendingPathComponent("config.json"))

        let d = config.dim, mlp = config.mlpInner, qkvD = config.qkvDim
        let merged = config.merged, outD = config.outDim
        var specs: [(String, [Int], Float)] = [
            ("model.visual.pos_embed.weight", [config.numPosEmbeds, d], 0.05),
            ("model.visual.patch_embed.proj.weight", [d, 3, config.temporal, config.patch, config.patch], 0.05),
            ("model.visual.patch_embed.proj.bias", [d], 0.02),
            ("model.visual.merger.norm.weight", [d], 1.0),
            ("model.visual.merger.norm.bias", [d], 0.02),
            ("model.visual.merger.linear_fc1.weight", [merged, merged], 0.03),
            ("model.visual.merger.linear_fc1.bias", [merged], 0.02),
            ("model.visual.merger.linear_fc2.weight", [outD, merged], 0.04),
            ("model.visual.merger.linear_fc2.bias", [outD], 0.02)
        ]
        for b in 0..<config.depth {
            let blkSpecs: [(String, [Int], Float)] = [
                ("model.visual.blocks.\(b).norm1.weight", [d], 1.0),
                ("model.visual.blocks.\(b).norm1.bias", [d], 0.02),
                ("model.visual.blocks.\(b).norm2.weight", [d], 1.0),
                ("model.visual.blocks.\(b).norm2.bias", [d], 0.02),
                ("model.visual.blocks.\(b).attn.qkv.weight", [qkvD, d], 0.05),
                ("model.visual.blocks.\(b).attn.qkv.bias", [qkvD], 0.02),
                ("model.visual.blocks.\(b).attn.proj.weight", [d, d], 0.05),
                ("model.visual.blocks.\(b).attn.proj.bias", [d], 0.02),
                ("model.visual.blocks.\(b).mlp.linear_fc1.weight", [mlp, d], 0.04),
                ("model.visual.blocks.\(b).mlp.linear_fc1.bias", [mlp], 0.02),
                ("model.visual.blocks.\(b).mlp.linear_fc2.weight", [d, mlp], 0.04),
                ("model.visual.blocks.\(b).mlp.linear_fc2.bias", [d], 0.02)
            ]
            specs.append(contentsOf: blkSpecs)
        }
        specs.sort { $0.0 < $1.0 }

        var rng = LCG(state: seed)
        var headerObj: [String: Any] = [:]
        var payload = Data()
        for (name, shape, scale) in specs {
            let count = shape.reduce(1, *)
            let start = payload.count
            var vals = [UInt16](repeating: 0, count: count)
            let isGamma = name.hasSuffix("norm1.weight") || name.hasSuffix("norm2.weight") || name.hasSuffix("merger.norm.weight")
            for i in 0..<count {
                let v = isGamma ? (1.0 + 0.1 * rng.nextUniform()) : (scale * rng.nextUniform())
                vals[i] = Float16(v).bitPattern
            }
            vals.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                payload.append(raw.bindMemory(to: UInt8.self))
            }
            headerObj[name] = ["dtype": "F16", "shape": shape, "data_offsets": [start, payload.count]]
        }

        let hdrData = try JSONSerialization.data(withJSONObject: headerObj)
        var fileData = Data(count: 8 + hdrData.count + payload.count)
        var hdrLen = UInt64(hdrData.count).littleEndian
        withUnsafeBytes(of: &hdrLen) { (raw: UnsafeRawBufferPointer) in fileData.replaceSubrange(0..<8, with: raw) }
        fileData.replaceSubrange(8..<(8 + hdrData.count), with: hdrData)
        fileData.replaceSubrange((8 + hdrData.count)..<fileData.count, with: payload)
        try fileData.write(to: dir.appendingPathComponent("model.safetensors"))
        return dir
    }

    private static func makeGridInputs(
        config: QWVisionConfig, t: Int, h: Int, w: Int, seed: UInt64
    ) throws -> (grid: QWVisionGrid, patches: [Float], bins: QWVisionRef.InterpBins, posIds: [Int32]) {
        let spec = QWMMSpec(visionConfig: config)
        let grid = try QWVisionGrid(t: t, h: h, w: w, spec: spec)
        var rng = LCG(state: seed)
        let patches = (0..<(grid.patches * config.patchChannels)).map { _ in 0.5 * rng.nextUniform() }
        return (grid, patches, grid.interpBins, grid.ropePositions)
    }

    func testMetalVsRefParityMultiSegmentAndConfigs() throws {
        _ = try Self.skipUnlessMetalAvailable()
        let cases: [(QWVisionConfig, Int, Int, Int)] = [
            (
                QWVisionConfig(
                    dim: 256, heads: 4, headDim: 64, depth: 2, mlpInner: 512,
                    patch: 4, temporal: 2, merge: 2, numPosEmbeds: 16, outDim: 128,
                    imageTokenId: 248056, videoTokenId: 248057
                ),
                2, 6, 6 // 2 segments of length 36 (> 32 tile)
            ),
            (
                QWVisionConfig(
                    dim: 144, heads: 2, headDim: 72, depth: 2, mlpInner: 288,
                    patch: 4, temporal: 2, merge: 2, numPosEmbeds: 16, outDim: 96,
                    imageTokenId: 248056, videoTokenId: 248057
                ),
                3, 2, 6 // 3 segments of length 12 (non-multiple of 32)
            )
        ]

        for (idx, (cfg, t, h, w)) in cases.enumerated() {
            let dir = try Self.writeSyntheticVisionDir(config: cfg, seed: UInt64(1001 + idx))
            defer { try? FileManager.default.removeItem(at: dir) }

            let weights = try QWVisionWeights(dir: dir.path)
            let ref = QWVisionRef(weights: weights)
            let metal = try QWVisionMetalBackend(weights: weights)

            let inputs = try Self.makeGridInputs(config: cfg, t: t, h: h, w: w, seed: UInt64(2002 + idx))
            let expected = ref.encode(
                patches: inputs.patches, plan: inputs.grid.plan,
                bins: inputs.bins, posIds: inputs.posIds
            )
            let actual = try metal.encodeSync(
                patches: inputs.patches, grid: inputs.grid.plan,
                bins: inputs.bins, posIds: inputs.posIds
            )

            XCTAssertEqual(actual.count, expected.count)
            let stats = Self.errorStats(actual, expected)
            XCTAssertLessThan(stats.maxAbs, 5e-3, "case \(idx) maxAbs=\(stats.maxAbs)")
            XCTAssertGreaterThan(stats.cosSim, 0.9999, "case \(idx) cosSim=\(stats.cosSim)")
        }
    }

    func testTiledFlashAttentionParityAcrossHeadDimsAndHeterogeneousSegments() throws {
        let (dev, lib) = try Self.skipUnlessMetalAvailable()
        let queue = try XCTUnwrap(dev.makeCommandQueue())
        let fn = try XCTUnwrap(lib.makeFunction(name: "qwg_vis_flash_attn_f16"))
        let pso = try dev.makeComputePipelineState(function: fn)

        let cuSeqlens: [Int32] = [0, 7, 25, 38, 105]
        let nPatch = Int(cuSeqlens.last!)
        let heads = 4

        for headDim in [64, 72, 80, 96, 128] {
            let dim = heads * headDim
            let qkvStride = 3 * dim
            var rng = LCG(state: UInt64(headDim * 17 + 9))
            let qkvF32 = (0..<(nPatch * qkvStride)).map { _ in Float(Float16(0.5 * rng.nextUniform())) }

            let scale = Float(1.0 / Double(headDim).squareRoot())
            var cpuOut = [Float](repeating: 0, count: nPatch * dim)
            for seg in 0..<(cuSeqlens.count - 1) {
                let lo = Int(cuSeqlens[seg]), hi = Int(cuSeqlens[seg + 1])
                for h in 0..<heads {
                    for qi in lo..<hi {
                        var scores = [Float](repeating: 0, count: hi - lo)
                        var mx: Float = -.greatestFiniteMagnitude
                        for ki in lo..<hi {
                            var dot: Float = 0
                            for d in 0..<headDim {
                                dot += qkvF32[qi * qkvStride + h * headDim + d]
                                     * qkvF32[ki * qkvStride + dim + h * headDim + d]
                            }
                            let s = dot * scale
                            scores[ki - lo] = s
                            mx = max(mx, s)
                        }
                        var sum: Float = 0
                        for idx in 0..<scores.count {
                            let e = exp(scores[idx] - mx)
                            scores[idx] = e
                            sum += e
                        }
                        let inv = 1.0 / sum
                        for d in 0..<headDim {
                            var acc: Float = 0
                            for ki in lo..<hi {
                                acc += (scores[ki - lo] * inv)
                                     * qkvF32[ki * qkvStride + 2 * dim + h * headDim + d]
                            }
                            cpuOut[qi * dim + h * headDim + d] = acc
                        }
                    }
                }
            }

            let qkvBuf = try Self.makeF16Buffer(dev, qkvF32)
            let outBuf = try XCTUnwrap(dev.makeBuffer(length: max(256, nPatch * dim * 2), options: .storageModeShared))
            let cuBuf = try XCTUnwrap(dev.makeBuffer(length: cuSeqlens.count * 4, options: .storageModeShared))
            cuBuf.contents().copyMemory(from: cuSeqlens, byteCount: cuSeqlens.count * 4)

            let cb = try XCTUnwrap(queue.makeCommandBuffer())
            let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(pso)
            enc.setBuffer(qkvBuf, offset: 0, index: 0)
            enc.setBuffer(outBuf, offset: 0, index: 1)
            enc.setBuffer(cuBuf, offset: 0, index: 2)
            Self.setU32(enc, cuSeqlens.count - 1, index: 3)
            Self.setU32(enc, heads, index: 4)
            Self.setU32(enc, headDim, index: 5)
            Self.setU32(enc, dim, index: 6)
            Self.setF32(enc, scale, index: 7)
            let grid = MTLSize(width: (67 + 31) / 32, height: heads, depth: cuSeqlens.count - 1)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)

            let gpuOut = Self.readF16Buffer(outBuf, count: nPatch * dim)
            let stats = Self.errorStats(gpuOut, cpuOut)
            XCTAssertLessThan(stats.maxAbs, 5e-3, "headDim=\(headDim) maxAbs=\(stats.maxAbs)")
            XCTAssertGreaterThan(stats.cosSim, 0.9999, "headDim=\(headDim) cosSim=\(stats.cosSim)")
        }
    }

    func testLaw4LayerNormGeluSplitAndOutOfBoundsSafety() throws {
        let (dev, lib) = try Self.skipUnlessMetalAvailable()
        let queue = try XCTUnwrap(dev.makeCommandQueue())
        let psoTanh = try dev.makeComputePipelineState(function: XCTUnwrap(lib.makeFunction(name: "qwg_vis_gelu_tanh_f16")))
        let psoErf = try dev.makeComputePipelineState(function: XCTUnwrap(lib.makeFunction(name: "qwg_vis_gelu_erf_f16")))
        let psoLN = try dev.makeComputePipelineState(function: XCTUnwrap(lib.makeFunction(name: "qwg_vis_layernorm_f16")))
        let psoPos = try dev.makeComputePipelineState(function: XCTUnwrap(lib.makeFunction(name: "qwg_vis_pos_embed_f16")))
        let psoGemm = try dev.makeComputePipelineState(function: XCTUnwrap(lib.makeFunction(name: "qwg_vis_gemm_f16")))

        let count = 256
        let xs = (0..<count).map { Float($0 - 128) * 0.05 }
        let bufTanh = try Self.makeF16Buffer(dev, xs)
        let bufErf = try Self.makeF16Buffer(dev, xs)

        let lnDim = 48, lnRows = 2
        var lnIn = [Float](repeating: 3.0, count: lnRows * lnDim)
        for j in 0..<lnDim { lnIn[lnDim + j] = Float(j % 7) - 3.0 }
        let lnG = (0..<lnDim).map { Float($0 % 5) * 0.25 + 0.5 }
        let lnB = (0..<lnDim).map { Float($0 % 3) * 0.1 - 0.1 }
        let lnXBuf = try Self.makeF16Buffer(dev, lnIn)
        let lnGBuf = try Self.makeF16Buffer(dev, lnG)
        let lnBBuf = try Self.makeF16Buffer(dev, lnB)
        let lnOBuf = try XCTUnwrap(dev.makeBuffer(length: lnRows * lnDim * 2, options: .storageModeShared))

        let posS = try Self.makeF16Buffer(dev, [1, 2, 3, 4])
        let posE = try Self.makeF16Buffer(dev, [0.5, 0.5, 0.5, 0.5])
        let posI = try XCTUnwrap(dev.makeBuffer(length: 16, options: .storageModeShared))
        let posW = try XCTUnwrap(dev.makeBuffer(length: 16, options: .storageModeShared))
        let oobIdx: [Int32] = [-1, 999, 0, -5]
        let oobWgt: [Float] = [0.5, 0.5, 0.5, 0.5]
        posI.contents().copyMemory(from: oobIdx, byteCount: 16)
        posW.contents().copyMemory(from: oobWgt, byteCount: 16)

        let cb = try XCTUnwrap(queue.makeCommandBuffer())
        let enc = try XCTUnwrap(cb.makeComputeCommandEncoder())
        enc.setComputePipelineState(psoTanh)
        enc.setBuffer(bufTanh, offset: 0, index: 0)
        Self.setU32(enc, count, index: 1)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        enc.setComputePipelineState(psoErf)
        enc.setBuffer(bufErf, offset: 0, index: 0)
        Self.setU32(enc, count, index: 1)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        enc.setComputePipelineState(psoLN)
        enc.setBuffer(lnXBuf, offset: 0, index: 0)
        enc.setBuffer(lnGBuf, offset: 0, index: 1)
        enc.setBuffer(lnBBuf, offset: 0, index: 2)
        enc.setBuffer(lnOBuf, offset: 0, index: 3)
        Self.setU32(enc, lnRows, index: 4)
        Self.setU32(enc, lnDim, index: 5)
        Self.setF32(enc, 0, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: lnRows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

        enc.setComputePipelineState(psoPos)
        enc.setBuffer(posS, offset: 0, index: 0)
        enc.setBuffer(posE, offset: 0, index: 1)
        enc.setBuffer(posI, offset: 0, index: 2)
        enc.setBuffer(posW, offset: 0, index: 3)
        Self.setU32(enc, 1, index: 4)
        Self.setU32(enc, 4, index: 5)
        Self.setU32(enc, 1, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let outTanh = Self.readF16Buffer(bufTanh, count: count)
        let outErf = Self.readF16Buffer(bufErf, count: count)
        let refTanh = xs.map { x in Float(0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))) }
        let refErf = xs.map { qwGeluErf($0) }
        XCTAssertLessThan(Self.errorStats(outTanh, refTanh).maxAbs, 2e-3)
        XCTAssertLessThan(Self.errorStats(outErf, refErf).maxAbs, 2e-3)
        XCTAssertGreaterThan(Self.errorStats(outTanh, outErf).maxAbs, 1e-4, "gelu_tanh and gelu_erf must stay distinct")

        let lnActual = Self.readF16Buffer(lnOBuf, count: lnRows * lnDim)
        var lnRef = [Float](repeating: 0, count: lnRows * lnDim)
        qwLayerNorm(lnIn, into: &lnRef, gamma: lnG, beta: lnB, rows: lnRows, dim: lnDim, eps: 0)
        XCTAssertLessThan(Self.errorStats(lnActual, lnRef).maxAbs, 2e-3)

        XCTAssertEqual(Self.readF16Buffer(posS, count: 4), [1.25, 2.25, 3.25, 4.25])

        // 4. Unaligned GEMM tail bounds (M=13, N=67, K=37) with bias + in-place residual (y == residual)
        let gm = 13, gn = 67, gk = 37
        var rng = LCG(state: 8008)
        let gx = (0..<(gm * gk)).map { _ in Float(Float16(0.5 * rng.nextUniform())) }
        let gw = (0..<(gn * gk)).map { _ in Float(Float16(0.5 * rng.nextUniform())) }
        let gb = (0..<gn).map { _ in Float(Float16(0.2 * rng.nextUniform())) }
        let gres = (0..<(gm * gn)).map { _ in Float(Float16(0.3 * rng.nextUniform())) }
        let bx = try Self.makeF16Buffer(dev, gx)
        let bw = try Self.makeF16Buffer(dev, gw)
        let bb = try Self.makeF16Buffer(dev, gb)
        let by = try Self.makeF16Buffer(dev, gres)

        let cb2 = try XCTUnwrap(queue.makeCommandBuffer())
        let enc2 = try XCTUnwrap(cb2.makeComputeCommandEncoder())
        enc2.setComputePipelineState(psoGemm)
        enc2.setBuffer(bx, offset: 0, index: 0)
        enc2.setBuffer(bw, offset: 0, index: 1)
        enc2.setBuffer(bb, offset: 0, index: 2)
        enc2.setBuffer(by, offset: 0, index: 3)
        enc2.setBuffer(by, offset: 0, index: 4)
        Self.setU32(enc2, gm, index: 5)
        Self.setU32(enc2, gn, index: 6)
        Self.setU32(enc2, gk, index: 7)
        Self.setU32(enc2, 3, index: 8)
        enc2.dispatchThreadgroups(
            MTLSize(width: (gn + 63) / 64, height: (gm + 31) / 32, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        enc2.endEncoding()
        cb2.commit()
        cb2.waitUntilCompleted()
        XCTAssertNil(cb2.error)

        let gemmGot = Self.readF16Buffer(by, count: gm * gn)
        var gemmRef = [Float](repeating: 0, count: gm * gn)
        for m in 0..<gm {
            for n in 0..<gn {
                var acc: Float = gb[n] + gres[m * gn + n]
                for k in 0..<gk { acc += gx[m * gk + k] * gw[n * gk + k] }
                gemmRef[m * gn + n] = acc
            }
        }
        let gStats = Self.errorStats(gemmGot, gemmRef)
        XCTAssertLessThan(gStats.maxAbs, 5e-3)
        XCTAssertGreaterThan(gStats.cosSim, 0.9999)
    }

    func testLaw15ResidencyCompactF16AndCPUFallback() throws {
        _ = try Self.skipUnlessMetalAvailable()
        let prevEnv = ProcessInfo.processInfo.environment["QWG_VISION_CPU"]
        defer {
            if let prevEnv { setenv("QWG_VISION_CPU", prevEnv, 1) } else { unsetenv("QWG_VISION_CPU") }
        }
        unsetenv("QWG_VISION_CPU")

        let cfg = QWVisionConfig(
            dim: 128, heads: 2, headDim: 64, depth: 2, mlpInner: 256,
            patch: 4, temporal: 2, merge: 2, numPosEmbeds: 16, outDim: 64,
            imageTokenId: 248056, videoTokenId: 248057
        )
        let dir = try Self.writeSyntheticVisionDir(config: cfg, seed: 5005)
        defer { try? FileManager.default.removeItem(at: dir) }

        let fp32Bytes = try QWVisionWeights(dir: dir.path).residentBytes
        let backend = try QWVisionOnDemandBackend(modelPath: dir.path, keepResident: false, allowOnDemand: true)
        XCTAssertFalse(backend.isLoaded)
        XCTAssertEqual(backend.residentBytes, 0)

        try backend.prepareForPrompt(hasImages: true)
        XCTAssertTrue(backend.isLoaded)
        let loadedWeightBytes = backend.residentBytes
        XCTAssertGreaterThan(loadedWeightBytes, 0)
        XCTAssertLessThanOrEqual(loadedWeightBytes, fp32Bytes / 2 + 16_384, "FP16 Metal weights must be ~2x smaller than FP32")

        let inputs = try Self.makeGridInputs(config: cfg, t: 1, h: 4, w: 4, seed: 6006)
        let metalOut = try backend.encodeSync(
            patches: inputs.patches, grid: inputs.grid.plan, bins: inputs.bins, posIds: inputs.posIds
        )
        let ref = QWVisionRef(weights: try QWVisionWeights(dir: dir.path))
        let refOut = ref.encode(
            patches: inputs.patches, plan: inputs.grid.plan, bins: inputs.bins, posIds: inputs.posIds
        )
        let stats = Self.errorStats(metalOut, refOut)
        XCTAssertLessThan(stats.maxAbs, 5e-3)
        XCTAssertGreaterThan(stats.cosSim, 0.9999)

        try backend.prepareForPrompt(hasImages: false)
        XCTAssertFalse(backend.isLoaded)
        XCTAssertEqual(backend.residentBytes, 0)

        try backend.setVisionLoaded(true)
        XCTAssertTrue(backend.isLoaded)
        try backend.prepareForPrompt(hasImages: false)
        XCTAssertTrue(backend.isLoaded, "keepResident vision must remain loaded on text turns")
        try backend.setVisionLoaded(false)
        XCTAssertFalse(backend.isLoaded)
        XCTAssertEqual(backend.residentBytes, 0)

        let disabled = try QWVisionOnDemandBackend(modelPath: dir.path, keepResident: false, allowOnDemand: false)
        XCTAssertThrowsError(try disabled.encodeSync(
            patches: inputs.patches, grid: inputs.grid.plan, bins: inputs.bins, posIds: inputs.posIds
        ))

        setenv("QWG_VISION_CPU", "1", 1)
        let cpuBackend = try QWVisionOnDemandBackend(modelPath: dir.path, keepResident: true, allowOnDemand: false)
        XCTAssertTrue(cpuBackend.isLoaded)
        XCTAssertEqual(cpuBackend.residentBytes, fp32Bytes, "CPU fallback must keep full FP32 weights resident")
        let cpuOut = try cpuBackend.encodeSync(
            patches: inputs.patches, grid: inputs.grid.plan, bins: inputs.bins, posIds: inputs.posIds
        )
        XCTAssertEqual(cpuOut, refOut)
    }

    func testVDSPPatchSlabParityAgainstScalarReference() throws {
        func scalarPatchSlab(
            normalized: [Float]?, u8: [UInt8]?, height: Int, width: Int, grid: QWVisionGrid
        ) -> [Float] {
            let s = grid.spec, cells = s.merge * s.merge, blocksW = grid.w / s.merge
            var out = [Float](repeating: 0, count: grid.patches * s.patchChannels)
            for pi in 0..<grid.patches {
                let hb = pi / (cells * blocksW), rem = pi % (cells * blocksW)
                let wb = rem / cells, p = rem % cells
                let i = p / s.merge, j = p % s.merge
                let tp = hb * s.factor + i * s.patch, wp = wb * s.factor + j * s.patch
                let rowOff = pi * s.patchChannels
                for c in 0..<3 {
                    let chOff = rowOff + c * s.temporalPatchArea
                    for r in 0..<s.patch {
                        let srcRow = (tp + r) * width * 3 + wp * 3 + c
                        let dstOff = chOff + r * s.patch
                        for col in 0..<s.patch {
                            let idx = srcRow + col * 3
                            let val = normalized?[idx] ?? ((Float(u8![idx]) / 255.0 - 0.5) / 0.5)
                            for t in 0..<s.temporal { out[dstOff + t * s.patchArea + col] = val }
                        }
                    }
                }
            }
            return out
        }

        let specs: [(QWMMSpec, Int, Int)] = [
            (.qwen3VLReference, 2, 4),
            (QWMMSpec(patch: 8, temporal: 1, merge: 2, posSide: 16, imageToken: 248056, videoToken: 248057), 4, 6)
        ]
        var rng = LCG(state: 7007)
        for (spec, gh, gw) in specs {
            let grid = try QWVisionGrid(t: 1, h: gh, w: gw, spec: spec)
            let h = gh * spec.patch, w = gw * spec.patch
            let count = h * w * 3
            let norm = (0..<count).map { _ in rng.nextUniform() }
            let u8 = (0..<count).map { UInt8(($0 * 37) % 256) }

            let gotNorm = try qwBuildPatchSlab(normalizedRGB: norm, height: h, width: w, grid: grid)
            let expNorm = scalarPatchSlab(normalized: norm, u8: nil, height: h, width: w, grid: grid)
            XCTAssertEqual(gotNorm, expNorm)

            let gotU8 = try qwBuildPatchSlab(imageRGB: u8, height: h, width: w, grid: grid)
            let expU8 = scalarPatchSlab(normalized: nil, u8: u8, height: h, width: w, grid: grid)
            XCTAssertLessThan(Self.errorStats(gotU8, expU8).maxAbs, 1e-6)
        }
    }
}
