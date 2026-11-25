// QWDSparkTests.swift — Config, confidence scheduler, 6-kernel Metal parity, and E2E DSpark parity tests.

import CryptoKit
import Foundation
import Metal
import XCTest
import QWAlloc
import QWCore
import QWHAL
import QWMTP
import QWQuant
@testable import QWEngine

final class QWDSparkTests: XCTestCase {
    static func bf16Bits(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let bias = UInt32(0x7FFF) &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: (bits &+ bias) >> 16)
    }

    static func bf16ToF32(_ b: UInt16) -> Float {
        Float(bitPattern: UInt32(b) << 16)
    }

    static func detFloats(_ count: Int, seedInit: UInt32, amp: Float, offset: Float = 0) -> [Float] {
        var s = seedInit
        return (0..<count).map { _ in
            s = s &* 1664525 &+ 1013904223
            let u = Float(Int32(bitPattern: s)) / Float(Int32.max)
            return offset + u * amp
        }
    }

    static func makeHQ8G32Data(n: Int, k: Int, seedInit: UInt32, scale: Float) -> Data {
        let group = 32
        let groups = k / group
        let gBytes = QWAQBlock.bytes(bits: 8, group: group)
        var seed = seedInit
        var out = Data(count: n * groups * gBytes)
        out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let base = (r * groups + g) * gBytes
                    let scVal = scale * (1.0 + 0.125 * Float((r + g) % 5))
                    let sc = Float16(scVal).bitPattern
                    let bi = Float16(-127.5 * scVal).bitPattern
                    ptr[base] = UInt8(sc & 0xFF)
                    ptr[base + 1] = UInt8(sc >> 8)
                    ptr[base + 2] = UInt8(bi & 0xFF)
                    ptr[base + 3] = UInt8(bi >> 8)
                    for b in 0..<group {
                        seed = seed &* 1664525 &+ 1013904223
                        ptr[base + 4 + b] = UInt8((seed >> 16) & 0xFF)
                    }
                }
            }
        }
        return out
    }

    static func makeSHVQData(bits: Int, group: Int, n: Int, k: Int, seedInit: UInt32, scale: Float) -> Data {
        let groups = k / group
        let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
        var seed = seedInit
        var out = Data(count: n * groups * gBytes)
        out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let base = (r * groups + g) * gBytes
                    let sc = Float16(scale * (1.0 + 0.125 * Float((r + g) % 5))).bitPattern
                    ptr[base] = UInt8(sc & 0xFF)
                    ptr[base + 1] = UInt8(sc >> 8)
                    for b in 2..<gBytes {
                        seed = seed &* 1664525 &+ 1013904223
                        ptr[base + b] = UInt8((seed >> 16) & 0xFF)
                    }
                }
            }
        }
        return out
    }

    private static func dequantHQ8G32Row(data: Data, row: Int, k: Int) -> [Float] {
        let group = 32
        let groups = k / group
        let gBytes = QWAQBlock.bytes(bits: 8, group: group)
        var out = [Float](repeating: 0, count: k)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for g in 0..<groups {
                let base = (row * groups + g) * gBytes
                let scBits = UInt16(ptr[base]) | (UInt16(ptr[base + 1]) << 8)
                let biBits = UInt16(ptr[base + 2]) | (UInt16(ptr[base + 3]) << 8)
                let sc = Float(Float16(bitPattern: scBits))
                let bi = Float(Float16(bitPattern: biBits))
                for l in 0..<group {
                    out[g * group + l] = Float(ptr[base + 4 + l]) * sc + bi
                }
            }
        }
        return out
    }

    private static func fwht32(_ v: inout [Float]) {
        let scale = 1.0 / sqrtf(32.0)
        for off in stride(from: 0, to: v.count, by: 32) {
            var strideLen = 1
            while strideLen < 32 {
                var i = 0
                while i < 32 {
                    for j in 0..<strideLen {
                        let a = v[off + i + j]
                        let b = v[off + i + j + strideLen]
                        v[off + i + j] = a + b
                        v[off + i + j + strideLen] = a - b
                    }
                    i += strideLen * 2
                }
                strideLen *= 2
            }
            for j in 0..<32 { v[off + j] *= scale }
        }
    }

    private static func allocShared(_ device: any MTLDevice, bytes: Int) throws -> any MTLBuffer {
        guard let buf = device.makeBuffer(length: max(256, bytes), options: .storageModeShared) else {
            throw QWEngineDriverError.state("Failed to allocate shared Metal test buffer (\(bytes) B)")
        }
        memset(buf.contents(), 0, max(256, bytes))
        return buf
    }

    // MARK: - CPU Config, Confidence Scheduler, Backend Contract Tests

    func testDSparkConfigDecodingDefaultsAndDiscovery() throws {
        let layerTypesJSON = Array(repeating: "\"full_attention\"", count: 43).joined(separator: ", ")
        let cfgJSON = """
        {
          "architectures": ["Ernie5_5_MoeForCausalLM"],
          "model_type": "ernie5_5_moe",
          "vocab_size": 32768,
          "hidden_size": 512,
          "intermediate_size": 1024,
          "num_hidden_layers": 43,
          "layer_types": [\(layerTypesJSON)],
          "max_position_embeddings": 16384,
          "num_attention_heads": 8,
          "num_key_value_heads": 2,
          "head_dim": 64,
          "partial_rotary_factor": 1.0,
          "bos_token_id": 1,
          "eos_token_id": 2,
          "rms_norm_eps": 1e-6,
          "rope_theta": 8000000.0,
          "dspark_block_size": 7,
          "dspark_markov_rank": 256,
          "dspark_noise_token_id": 1024,
          "dspark_target_layer_ids": [40, 41, 42],
          "dspark_sliding_window": 128,
          "enable_confidence_head": true
        }
        """.data(using: .utf8)!
        let desc = try QWModelDescriptor.decode(configJSON: cfgJSON)
        let ds = try XCTUnwrap(desc.dsparkConfig)
        XCTAssertEqual(ds.blockSize, 7)
        XCTAssertEqual(ds.markovRank, 256)
        XCTAssertEqual(ds.noiseTokenId, 1024)
        XCTAssertEqual(ds.targetZeroBasedLayers, [40, 41, 42])
        XCTAssertEqual(ds.numHiddenLayers, 3)
        XCTAssertEqual(ds.slidingWindow, 128)
        XCTAssertEqual(ds.ropeTheta, 8_000_000.0, accuracy: 1e-6)
        XCTAssertTrue(ds.sampleFromAnchor)
        XCTAssertEqual(ds.sampleOffset, 0)
        XCTAssertEqual(ds.maxDrafts, 7)
        XCTAssertEqual(ds.enableConfidenceHead, true)

        // Reject out-of-range layer index (1-based 43 on 43 layers).
        let badJSON = String(decoding: cfgJSON, as: UTF8.self)
            .replacingOccurrences(of: "[40, 41, 42]", with: "[41, 42, 43]")
            .data(using: .utf8)!
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: badJSON))

        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-dspark-discover-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tmpDir) }
        let assetsDir = tmpDir.appendingPathComponent("assets")
        try FileManager.default.createDirectory(at: assetsDir, withIntermediateDirectories: true)
        XCTAssertNil(QWModelDescriptor.DSparkConfig.resolvePackURL(near: tmpDir.path))
        let dsFile = tmpDir.appendingPathComponent("dspark.qwp")
        let headFile = assetsDir.appendingPathComponent("draft_head.qwp")
        try Data("x".utf8).write(to: dsFile)
        try Data("y".utf8).write(to: headFile)
        XCTAssertEqual(QWModelDescriptor.DSparkConfig.resolvePackURL(near: tmpDir.path)?.path, dsFile.path)
        XCTAssertEqual(QWModelDescriptor.DSparkConfig.resolveDraftHeadURL(near: tmpDir.path)?.path, headFile.path)
    }

    func testConfidenceSchedulerBudgetFormulaAndEMA() {
        var sched = QWDSparkConfidenceScheduler(maxDrafts: 6, emaAlpha: 0.5)
        sched.recordDraft(seconds: 0.004)
        for m in 2...7 {
            // Moderate verify growth makes short k* win on early decay and k*=6 win on sustained confidence.
            sched.recordVerify(verifyRows: m, seconds: 0.004 + 0.0015 * Double(m))
        }
        let lowConf: [Float] = [0.8, 0.15, 0.1, 0.05, 0.05, 0.05]
        let highConf: [Float] = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94]
        let kLow = sched.optimalDraftCount(confidences: lowConf)
        let kHigh = sched.optimalDraftCount(confidences: highConf)
        XCTAssertEqual(kLow, 1)
        XCTAssertGreaterThan(kHigh, kLow)
        XCTAssertEqual(kHigh, 6)
    }

    func testQWMTPBackendDSparkContractAndParameters() throws {
        XCTAssertEqual(try QWMTPBackend.resolve(name: "dspark"), .dspark)
        XCTAssertEqual(try QWMTPBackend.resolveSelection(.dspark, drafts: 7), .dspark)
        XCTAssertThrowsError(try QWMTPBackend.resolveSelection(.dspark, drafts: 0))
        XCTAssertTrue(QWMTPBackend.dspark.route.contains("dspark"))
        XCTAssertNotEqual(QWMTPBackend.dspark.qualificationProfile, QWMTPBackend.standard.qualificationProfile)
        XCTAssertEqual(QWEngineDriver.prefixPolicyKey(backend: .dspark, drafts: 4, mode: .chunked), "dspark+chunked")

        let encoded = try JSONEncoder().encode(QWMTPBackend.dspark)
        XCTAssertEqual(String(decoding: encoded, as: UTF8.self), "\"dspark\"")
        XCTAssertEqual(try JSONDecoder().decode(QWMTPBackend.self, from: encoded), .dspark)

        let spec = QWSpeculativeParameters(mtpDrafts: 7, mtpBackend: "dspark", dsparkConfidenceSchedule: true)
        XCTAssertEqual(spec.mtpBackend, "dspark")
        XCTAssertEqual(spec.dsparkConfidenceSchedule, true)
    }

    // MARK: - Metal 6-Kernel Parity Tests

    func testMetalDSparkTapMeanAndPlainRoPESWAWrite() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && MTLCreateSystemDefaultDevice() != nil,
            "Requires Apple Silicon Metal device"
        )
        let hal = try QWHal(geometry: .qwen38_27b)
        let psoTap = try hal.pipeline("qwg_dspark_mhc_tap_mean")
        let psoKV = try hal.pipeline("qwg_dspark_swa_ctx_kv_write")

        // 1. Test `qwg_dspark_mhc_tap_mean` with deferred mHC branch+controls inject + FWHT32 (`flags = 10`).
        let T = 2, C = 4, H = 32, totalSlots = 3, slot = 1
        let mhcRows = C + C * C
        let baseData = Self.detFloats(T * C * H, seedInit: 11, amp: 0.5)
        let srcHalf = Self.detFloats(T * H, seedInit: 22, amp: 0.25).map { Float16($0) }
        let injData = Self.detFloats(T * mhcRows, seedInit: 33, amp: 0.2)

        let baseBuf = try Self.allocShared(hal.device, bytes: baseData.count * 4)
        let srcBuf = try Self.allocShared(hal.device, bytes: srcHalf.count * 2)
        let injBuf = try Self.allocShared(hal.device, bytes: injData.count * 4)
        let dstBuf = try Self.allocShared(hal.device, bytes: T * totalSlots * H * 2)
        memcpy(baseBuf.contents(), baseData, baseData.count * 4)
        memcpy(srcBuf.contents(), srcHalf, srcHalf.count * 2)
        memcpy(injBuf.contents(), injData, injData.count * 4)

        guard let cb = hal.makeUnretainedCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            return XCTFail("Metal command encoder failed")
        }
        enc.setComputePipelineState(psoTap)
        enc.setBuffer(baseBuf, offset: 0, index: 0)
        enc.setBuffer(srcBuf, offset: 0, index: 1)
        enc.setBuffer(injBuf, offset: 0, index: 2)
        enc.setBuffer(dstBuf, offset: 0, index: 3)
        enc.setUInt32s(0, 0, slot, H / 8, totalSlots, C, 10, startIndex: 4)
        enc.dispatchThreadgroups(MTLSize(width: T, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let tapPtr = dstBuf.contents().assumingMemoryBound(to: UInt16.self)
        for t in 0..<T {
            var cpu = [Float](repeating: 0, count: H)
            for d in 0..<H {
                var sum: Float = 0
                let branch = Float(srcHalf[t * H + d])
                for i in 0..<C {
                    var ns = injData[t * mhcRows + i] * branch
                    for j in 0..<C {
                        ns += injData[t * mhcRows + C + i * C + j] * baseData[(t * C + j) * H + d]
                    }
                    sum += ns
                }
                cpu[d] = sum / Float(C)
            }
            Self.fwht32(&cpu)
            for d in 0..<H {
                let got = Float(Float16(bitPattern: tapPtr[(t * totalSlots + slot) * H + d]))
                XCTAssertEqual(got, cpu[d], accuracy: 2e-3)
            }
        }

        // 2. Test `qwg_dspark_swa_ctx_kv_write` wraparound ring `(pos0 + t) % ringSize` + plain RoPE.
        let dim = 64, nopeDim = 32, ropeDim = 32, pos0 = 3, ringSize = 4, kvT = 4
        let rawIn = Self.detFloats(kvT * dim, seedInit: 44, amp: 0.6).map { Float16($0) }
        let normW = Self.detFloats(dim, seedInit: 55, amp: 0.2, offset: 1.0).map { Float16($0) }
        let (invFreq, _) = QWModelDescriptor.computeRopeInvFreq(rotaryDim: ropeDim, base: 8_000_000.0, yarn: nil)
        let rawBuf = try Self.allocShared(hal.device, bytes: rawIn.count * 2)
        let nwBuf = try Self.allocShared(hal.device, bytes: normW.count * 2)
        let invBuf = try Self.allocShared(hal.device, bytes: invFreq.count * 4)
        let ringBuf = try Self.allocShared(hal.device, bytes: ringSize * dim * 2)
        memcpy(rawBuf.contents(), rawIn, rawIn.count * 2)
        memcpy(nwBuf.contents(), normW, normW.count * 2)
        memcpy(invBuf.contents(), invFreq, invFreq.count * 4)

        guard let cb2 = hal.makeUnretainedCommandBuffer(), let enc2 = cb2.makeComputeCommandEncoder() else {
            return XCTFail("Metal command encoder failed")
        }
        var kp = QWDSparkDrafter.DSparkCtxKVParams(
            T: UInt32(kvT), dim: UInt32(dim), nopeDim: UInt32(nopeDim), ropeDim: UInt32(ropeDim),
            pos0: UInt32(pos0), ringSize: UInt32(ringSize), flags: 0, rmsEps: 1e-6
        )
        enc2.setComputePipelineState(psoKV)
        enc2.setBuffer(rawBuf, offset: 0, index: 0)
        enc2.setBuffer(nwBuf, offset: 0, index: 1)
        enc2.setBuffer(invBuf, offset: 0, index: 2)
        enc2.setBuffer(ringBuf, offset: 0, index: 3)
        enc2.setBytes(&kp, length: MemoryLayout<QWDSparkDrafter.DSparkCtxKVParams>.stride, index: 4)
        enc2.dispatchThreadgroups(MTLSize(width: kvT, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc2.endEncoding()
        cb2.commit()
        cb2.waitUntilCompleted()

        let ringPtr = ringBuf.contents().assumingMemoryBound(to: UInt16.self)
        for t in 0..<kvT {
            let sq = (0..<dim).reduce(Float(0)) { $0 + Float(rawIn[t * dim + $1]) * Float(rawIn[t * dim + $1]) }
            let invRms = 1.0 / sqrtf(sq / Float(dim) + 1e-6)
            var cpu = (0..<dim).map { Float(rawIn[t * dim + $0]) * invRms * Float(normW[$0]) }
            let pos = pos0 + t
            for i in 0..<(ropeDim / 2) {
                let b = nopeDim + 2 * i
                let ang = Float(pos) * invFreq[i]
                let c = cosf(ang), s = sinf(ang)
                let even = cpu[b], odd = cpu[b + 1]
                cpu[b] = even * c - odd * s
                cpu[b + 1] = odd * c + even * s
            }
            let dstRow = pos % ringSize
            for d in 0..<dim {
                let got = Float(Float16(bitPattern: ringPtr[dstRow * dim + d]))
                XCTAssertEqual(got, cpu[d], accuracy: 2e-3)
            }
        }
    }

    func testMetalDSparkBlockHCAAttention() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && MTLCreateSystemDefaultDevice() != nil,
            "Requires Apple Silicon Metal device"
        )
        let hal = try QWHal(geometry: .qwen38_27b)
        let pso = try hal.pipeline("qwg_dspark_block_hca_attn")

        let B = 4, qHeads = 2, headDim = 64, qUpStride = 192
        let pos0 = 3, ringSize = 8, slidingWindow = 8
        let qUp = Self.detFloats(B * qUpStride, seedInit: 61, amp: 0.4).map { Float16($0) }
        let swaKV = Self.detFloats(ringSize * headDim, seedInit: 62, amp: 0.4).map { Float16($0) }
        let blkKV = Self.detFloats(B * headDim, seedInit: 63, amp: 0.4).map { Float16($0) }
        let sink: [Float] = [-1.2, 0.7]

        let qBuf = try Self.allocShared(hal.device, bytes: qUp.count * 2)
        let swaBuf = try Self.allocShared(hal.device, bytes: swaKV.count * 2)
        let blkBuf = try Self.allocShared(hal.device, bytes: blkKV.count * 2)
        let sinkBuf = try Self.allocShared(hal.device, bytes: sink.count * 4)
        let outBuf = try Self.allocShared(hal.device, bytes: B * qHeads * headDim * 2)
        memcpy(qBuf.contents(), qUp, qUp.count * 2)
        memcpy(swaBuf.contents(), swaKV, swaKV.count * 2)
        memcpy(blkBuf.contents(), blkKV, blkKV.count * 2)
        memcpy(sinkBuf.contents(), sink, sink.count * 4)

        guard let cb = hal.makeUnretainedCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            return XCTFail("Metal command encoder failed")
        }
        var params = QWDSparkDrafter.DSparkBlockAttnParams(
            B: UInt32(B), qHeads: UInt32(qHeads), headDim: UInt32(headDim), qUpStride: UInt32(qUpStride),
            pos0: UInt32(pos0), ringSize: UInt32(ringSize), slidingWindow: UInt32(slidingWindow),
            hasSink: 1, swaMaskMode: 1
        )
        enc.setComputePipelineState(pso)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(swaBuf, offset: 0, index: 1)
        enc.setBuffer(blkBuf, offset: 0, index: 2)
        enc.setBuffer(sinkBuf, offset: 0, index: 3)
        enc.setBuffer(outBuf, offset: 0, index: 4)
        enc.setBytes(&params, length: MemoryLayout<QWDSparkDrafter.DSparkBlockAttnParams>.stride, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: B, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let scale = 1.0 / sqrtf(Float(headDim))
        let outPtr = outBuf.contents().assumingMemoryBound(to: UInt16.self)
        for q in 0..<B {
            for h in 0..<qHeads {
                var scores: [Float] = []
                var vals: [[Float]] = []
                for p in max(0, pos0 - slidingWindow)..<pos0 {
                    let row = p % ringSize
                    var dot: Float = 0
                    var kv = [Float](repeating: 0, count: headDim)
                    for d in 0..<headDim {
                        kv[d] = Float(swaKV[row * headDim + d])
                        dot += Float(qUp[q * qUpStride + h * headDim + d]) * kv[d]
                    }
                    scores.append(dot * scale)
                    vals.append(kv)
                }
                for b in 0..<B {
                    var dot: Float = 0
                    var kv = [Float](repeating: 0, count: headDim)
                    for d in 0..<headDim {
                        kv[d] = Float(blkKV[b * headDim + d])
                        dot += Float(qUp[q * qUpStride + h * headDim + d]) * kv[d]
                    }
                    scores.append(dot * scale)
                    vals.append(kv)
                }
                let maxS = max(sink[h], scores.max() ?? -Float.infinity)
                let sinkExp = expf(sink[h] - maxS)
                let exps = scores.map { expf($0 - maxS) }
                let denom = sinkExp + exps.reduce(0, +)
                for d in 0..<headDim {
                    var acc: Float = 0
                    for i in 0..<exps.count { acc += (exps[i] / denom) * vals[i][d] }
                    let got = Float(Float16(bitPattern: outPtr[(q * qHeads + h) * headDim + d]))
                    XCTAssertEqual(got, acc, accuracy: 2e-3)
                }
            }
        }
    }

    func testMetalDSparkMarkovArgmaxLogitsAndConfidence() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && MTLCreateSystemDefaultDevice() != nil,
            "Requires Apple Silicon Metal device"
        )
        let hal = try QWHal(geometry: .qwen38_27b)
        let psoArgmax = try hal.pipeline("qwg_dspark_markov_aq8_argmax")
        let psoFin = try hal.pipeline("qwg_head_argmax_fin")
        let psoLogits = try hal.pipeline("qwg_dspark_markov_aq8_logits")
        let psoConf = try hal.pipeline("qwg_dspark_confidence_dot")

        let V = 256, R = 64, H = 64, prevTok = 19, stepIdx = 0, hiddenRowIdx = 1
        let baseLogits = Self.detFloats(V, seedInit: 71, amp: 1.5)
        let w1Floats = Self.detFloats(V * R, seedInit: 72, amp: 0.4)
        let w1Bits = w1Floats.map(Self.bf16Bits)
        let w2Data = Self.makeHQ8G32Data(n: V, k: R, seedInit: 73, scale: 0.02)
        let headHidden = Self.detFloats(2 * H, seedInit: 74, amp: 0.5).map { Float16($0) }
        let confW = Self.detFloats(H + R, seedInit: 75, amp: 0.25).map(Self.bf16Bits)

        let baseBuf = try Self.allocShared(hal.device, bytes: V * 4)
        let w1Buf = try Self.allocShared(hal.device, bytes: V * R * 2)
        let w2Buf = try Self.allocShared(hal.device, bytes: w2Data.count)
        let seqBuf = try Self.allocShared(hal.device, bytes: 16)
        let outLogitsBuf = try Self.allocShared(hal.device, bytes: V * 4)
        let keysHiBuf = try Self.allocShared(hal.device, bytes: 16)
        let keysLoBuf = try Self.allocShared(hal.device, bytes: 16)
        let hiddenBuf = try Self.allocShared(hal.device, bytes: 2 * H * 2)
        let confWBuf = try Self.allocShared(hal.device, bytes: (H + R) * 2)
        let confOutBuf = try Self.allocShared(hal.device, bytes: 16)
        memcpy(baseBuf.contents(), baseLogits, V * 4)
        memcpy(w1Buf.contents(), w1Bits, V * R * 2)
        _ = w2Data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in memcpy(w2Buf.contents(), raw.baseAddress, w2Data.count) }
        let seqPtr = seqBuf.contents().assumingMemoryBound(to: Int32.self)
        seqPtr[0] = Int32(prevTok)
        memcpy(hiddenBuf.contents(), headHidden, 2 * H * 2)
        memcpy(confWBuf.contents(), confW, (H + R) * 2)

        for rotFlag in [UInt32(0), UInt32(1)] {
            guard let cb = hal.makeUnretainedCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
                return XCTFail("Metal command encoder failed")
            }
            let partCount = UInt32((V + 255) / 256)
            var lp = QWDSparkDrafter.DSparkMarkovParams(
                stepIdx: UInt32(stepIdx), V: UInt32(V), rank: UInt32(R),
                flags: rotFlag, partCount: UInt32((V + 7) / 8)
            )
            enc.setComputePipelineState(psoLogits)
            enc.setBuffer(baseBuf, offset: 0, index: 0)
            enc.setBuffer(w1Buf, offset: 0, index: 1)
            enc.setBuffer(w2Buf, offset: 0, index: 2)
            enc.setBuffer(seqBuf, offset: 0, index: 3)
            enc.setBuffer(outLogitsBuf, offset: 0, index: 4)
            enc.setBytes(&lp, length: MemoryLayout<QWDSparkDrafter.DSparkMarkovParams>.stride, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: (V + 7) / 8, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            var ap = QWDSparkDrafter.DSparkMarkovParams(
                stepIdx: UInt32(stepIdx), V: UInt32(V), rank: UInt32(R),
                flags: rotFlag, partCount: partCount
            )
            enc.setComputePipelineState(psoArgmax)
            enc.setBuffer(baseBuf, offset: 0, index: 0)
            enc.setBuffer(w1Buf, offset: 0, index: 1)
            enc.setBuffer(w2Buf, offset: 0, index: 2)
            enc.setBuffer(seqBuf, offset: 0, index: 3)
            enc.setBuffer(keysHiBuf, offset: 0, index: 4)
            enc.setBuffer(keysLoBuf, offset: 0, index: 5)
            enc.setBytes(&ap, length: MemoryLayout<QWDSparkDrafter.DSparkMarkovParams>.stride, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: Int(partCount), height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            var nTG32 = partCount
            enc.setComputePipelineState(psoFin)
            enc.setBuffer(keysHiBuf, offset: 0, index: 0)
            enc.setBuffer(keysLoBuf, offset: 0, index: 1)
            enc.setBuffer(seqBuf, offset: 4, index: 2)
            enc.setBytes(&nTG32, length: 4, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))

            if rotFlag == 0 {
                var cp = QWDSparkDrafter.DSparkConfParams(
                    stepIdx: UInt32(stepIdx), H: UInt32(H), R: UInt32(R),
                    hiddenRowIdx: UInt32(hiddenRowIdx), V: UInt32(V), flags: 0
                )
                enc.setComputePipelineState(psoConf)
                enc.setBuffer(hiddenBuf, offset: 0, index: 0)
                enc.setBuffer(w1Buf, offset: 0, index: 1)
                enc.setBuffer(confWBuf, offset: 0, index: 2)
                enc.setBuffer(seqBuf, offset: 0, index: 3)
                enc.setBuffer(confOutBuf, offset: 0, index: 4)
                enc.setBytes(&cp, length: MemoryLayout<QWDSparkDrafter.DSparkConfParams>.stride, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()

            var eVec = (0..<R).map { Self.bf16ToF32(w1Bits[prevTok * R + $0]) }
            if rotFlag == 1 { Self.fwht32(&eVec) }
            let outPtr = outLogitsBuf.contents().assumingMemoryBound(to: Float.self)
            var bestIdx = 0
            var bestVal = -Float.infinity
            for v in 0..<V {
                let row = Self.dequantHQ8G32Row(data: w2Data, row: v, k: R)
                let dot = zip(eVec, row).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                let expected = baseLogits[v] + dot
                XCTAssertEqual(outPtr[v], expected, accuracy: 2e-3)
                if expected > bestVal { bestVal = expected; bestIdx = v }
            }
            XCTAssertEqual(Int(seqPtr[1]), bestIdx)

            if rotFlag == 0 {
                var confSum: Float = 0
                for i in 0..<H { confSum += Float(headHidden[hiddenRowIdx * H + i]) * Self.bf16ToF32(confW[i]) }
                for r in 0..<R { confSum += Self.bf16ToF32(w1Bits[prevTok * R + r]) * Self.bf16ToF32(confW[H + r]) }
                let expectedProb = 1.0 / (1.0 + expf(-confSum))
                let gotProb = confOutBuf.contents().assumingMemoryBound(to: Float.self)[0]
                XCTAssertEqual(gotProb, expectedProb, accuracy: 1e-4)
            }
        }
    }

    // MARK: - Synthetic 2-Target + 1-Draft Driver E2E Bit-Exact Parity Test

    func testDSparkSyntheticDriverBitExactGreedyParity() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "DSpark full driver E2E requires bare-metal Metal 4 Apple Silicon GPU"
        )
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-dspark-e2e-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticTargetAndDSparkPack(at: rootDir)

        let prompt = [1, 7, 19, 42, 7, 19, 42, 88, 103, 7, 19, 42, 155, 201]
        let maxNew = 10

        func run(_ driver: QWEngineDriver, prompt: [Int], maxNew: Int, speculative: Bool) throws -> [Int] {
            let s = try driver.makeSession(promptTokens: prompt.count)
            defer { driver.finishSession(s) }
            return speculative
                ? try driver.generateExperimentalGreedy(session: s, prompt: prompt, maxNew: maxNew)
                : try driver.generate(session: s, prompt: prompt, maxNew: maxNew)
        }

        let baseDriver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 512, ropeRows: 512,
            kvMode: .q4, mtpDrafts: 0, prefillChunkSize: 32
        )
        let baseline = try run(baseDriver, prompt: prompt, maxNew: maxNew, speculative: false)
        XCTAssertEqual(baseline.count, maxNew)

        let autoDriver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 512, ropeRows: 512,
            kvMode: .q4, mtpDrafts: 4, mtpBackend: nil,
            dsparkConfidenceSchedule: true, prefillChunkSize: 32
        )
        XCTAssertEqual(autoDriver.mtpBackend, .dspark)
        XCTAssertNotNil(autoDriver.dsparkDrafter)
        let autoTokens = try run(autoDriver, prompt: prompt, maxNew: maxNew, speculative: true)
        XCTAssertEqual(autoTokens, baseline, "Auto-selected DSpark + confidence schedule must be bit-exact")

        let repPrompt = [1, 12, 24, 36, 48, 60, 72, 12, 24, 36, 48, 60]
        let repBase = try run(baseDriver, prompt: repPrompt, maxNew: 8, speculative: false)
        let repDSpark = try run(autoDriver, prompt: repPrompt, maxNew: 8, speculative: true)
        XCTAssertEqual(repDSpark, repBase, "Repetitive prompt DSpark Context PLD + Markov must be bit-exact")
    }

    // MARK: - Phase 5d CPU Verification Tests

    func testDSparkConfidenceSchedulerMoECalibrationAndInterpolation() {
        let f1 = QWDSparkConfidenceScheduler.verifyCostFactor(verifyRows: 1, numExperts: 512, expertsPerTok: 10)
        let f2 = QWDSparkConfidenceScheduler.verifyCostFactor(verifyRows: 2, numExperts: 512, expertsPerTok: 10)
        let f8 = QWDSparkConfidenceScheduler.verifyCostFactor(verifyRows: 8, numExperts: 512, expertsPerTok: 10)
        let fDense8 = QWDSparkConfidenceScheduler.verifyCostFactor(verifyRows: 8, numExperts: 0, expertsPerTok: 0)
        XCTAssertEqual(f1, 1.0, accuracy: 1e-9)
        XCTAssertGreaterThan(f8, f2)
        XCTAssertGreaterThan(f8, fDense8)

        var sched = QWDSparkConfidenceScheduler(maxDrafts: 7, emaAlpha: 0.5, numExperts: 512, expertsPerTok: 10)
        sched.recordDraft(seconds: 0.009)
        let collapse: [Float] = [0.90, 0.20, 0.10, 0.10, 0.10, 0.10, 0.10]
        let sustained: [Float] = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93]
        XCTAssertEqual(sched.optimalDraftCount(confidences: collapse), 1)
        XCTAssertEqual(sched.optimalDraftCount(confidences: sustained), 7)

        sched.recordVerify(verifyRows: 8, seconds: 0.068)
        XCTAssertEqual(sched.optimalDraftCount(confidences: collapse), 1)
        XCTAssertEqual(sched.optimalDraftCount(confidences: sustained), 7)
    }

    func testDSparkRequestedDraftsOverBlockSizePLDSuffixContinuation() {
        let n = [20, 30, 40, 50, 60, 70, 80]
        let h = [5, 10, 20, 30, 40, 50, 60, 70, 80, 91, 92, 93, 94, 95, 96, 97, 98, 1, 2, 3, 4]
        XCTAssertEqual(QWDSparkDrafter.appendPLDSuffixIfEligible(
            drafts: n, history: h, extraPending: 10, requestedDrafts: 11,
            neuralMaxDrafts: 7, schedulerEnabled: false, lastConfidence: nil), n + [91, 92, 93, 94])
        XCTAssertEqual(QWDSparkDrafter.appendPLDSuffixIfEligible(
            drafts: n, history: h, extraPending: 10, requestedDrafts: 15,
            neuralMaxDrafts: 7, schedulerEnabled: true, lastConfidence: 0.72), n + [91, 92, 93, 94, 95, 96, 97, 98])
        XCTAssertEqual(QWDSparkDrafter.appendPLDSuffixIfEligible(
            drafts: n, history: h, extraPending: 10, requestedDrafts: 11,
            neuralMaxDrafts: 7, schedulerEnabled: true, lastConfidence: 0.69), n)
        XCTAssertEqual(QWDSparkDrafter.appendPLDSuffixIfEligible(
            drafts: Array(n.prefix(5)), history: h, extraPending: 10, requestedDrafts: 11,
            neuralMaxDrafts: 7, schedulerEnabled: true, lastConfidence: 0.95), Array(n.prefix(5)))
        XCTAssertEqual(QWDSparkDrafter.appendPLDSuffixIfEligible(
            drafts: [3, 1, 2, 3, 1, 2, 3], history: [1, 2, 3, 1, 2, 3, 1], extraPending: 2,
            requestedDrafts: 11, neuralMaxDrafts: 7, schedulerEnabled: false, lastConfidence: nil),
            [3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1])
    }

    func testPostDSparkHybridSuffixExtension() {
        let neural7 = [20, 30, 40, 50, 60, 70, 80]
        // 5-gram match copies into [extraPending] + drafts; isolated recycler runs bigram only.
        let history = [40, 50, 60, 70, 80, 91, 92, 93, 94, 95]
        let savedHistory = history
        let recycler = QWTokenRecycler()
        recycler.ingestSequence([60, 70, 80, 96, 97])
        recycler.ingestSequence([97, 999])
        let ext15 = QWEngineDriver.extendDSparkDraftsWithHybridSuffix(
            drafts: neural7, tokenHistory: history, extraPending: 10, previousBeforePending: 3, pending: 10,
            tokenRecycler: recycler, requestedDrafts: 15, maxNeuralDrafts: 7, strategy: .hybrid, isDeterministic: true)
        XCTAssertEqual(ext15, neural7 + [91, 92, 93, 94, 95, 10, 20, 30])
        XCTAssertEqual(history, savedHistory)
        let extRec = QWEngineDriver.extendDSparkDraftsWithHybridSuffix(
            drafts: neural7, tokenHistory: [1, 2, 3], extraPending: nil, previousBeforePending: nil, pending: 10,
            tokenRecycler: recycler, requestedDrafts: 15, maxNeuralDrafts: 7, strategy: .hybrid, isDeterministic: true)
        XCTAssertEqual(extRec, neural7 + [96, 97])

        let extPeriodic = QWEngineDriver.extendDSparkDraftsWithHybridSuffix(
            drafts: [3, 1, 2, 3, 1, 2, 3], tokenHistory: [1, 2, 3, 1, 2, 3, 1], extraPending: 2,
            previousBeforePending: 1, pending: 2, tokenRecycler: recycler, requestedDrafts: 11,
            maxNeuralDrafts: 7, strategy: .hybrid, isDeterministic: true
        )
        XCTAssertEqual(extPeriodic, [3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1])

        // Pure .mtp (neural), stochastic, and scheduler-truncated (< 7) rounds must not extend.
        func extend(_ d: [Int], strat: QWDraftStrategy = .hybrid, det: Bool = true) -> [Int] {
            QWEngineDriver.extendDSparkDraftsWithHybridSuffix(
                drafts: d, tokenHistory: history, extraPending: 10, previousBeforePending: 3, pending: 10,
                tokenRecycler: recycler, requestedDrafts: 15, maxNeuralDrafts: 7, strategy: strat, isDeterministic: det)
        }
        XCTAssertEqual(extend(neural7, strat: .mtp), neural7)
        XCTAssertEqual(extend(neural7, det: false), neural7)
        let short5 = Array(neural7.prefix(5))
        XCTAssertEqual(extend(short5), short5)
    }

    func testDSparkPrefillCaptureSliceChunkBoundaries() {
        let win = 128, end16k = 15_912
        XCTAssertNil(QWDSparkDrafter.prefillCaptureSlice(startPos: 0, rows: 4096, windowEnd: end16k, slidingWindow: win))
        XCTAssertNil(QWDSparkDrafter.prefillCaptureSlice(startPos: 4096, rows: 4096, windowEnd: end16k, slidingWindow: win))
        XCTAssertNil(QWDSparkDrafter.prefillCaptureSlice(startPos: 8192, rows: 4096, windowEnd: end16k, slidingWindow: win))
        let last = QWDSparkDrafter.prefillCaptureSlice(startPos: 12_288, rows: 3624, windowEnd: end16k, slidingWindow: win)
        XCTAssertEqual(last?.ctxStartPos, 15_784)
        XCTAssertEqual(last?.localStart, 3496)
        XCTAssertEqual(last?.captureRows, 128)
        let short = QWDSparkDrafter.prefillCaptureSlice(startPos: 0, rows: 24, windowEnd: 24, slidingWindow: win)
        XCTAssertEqual(short?.ctxStartPos, 0)
        XCTAssertEqual(short?.localStart, 0)
        XCTAssertEqual(short?.captureRows, 24)
        XCTAssertNil(QWDSparkDrafter.prefillCaptureSlice(startPos: 0, rows: 0, windowEnd: 24, slidingWindow: win))
    }

    // MARK: - Phase 5e Tests (Overlap Gate & Bundled Draft Head)

    func testDSparkPriorCommitOverlapGateAndSyncDrainSafety() {
        XCTAssertTrue(QWDSparkDrafter.allowsPriorCommitOverlap(schedulerEnabled: false, verifyRows: 8, maxDrafts: 7))
        XCTAssertFalse(QWDSparkDrafter.allowsPriorCommitOverlap(schedulerEnabled: false, verifyRows: 12, maxDrafts: 7))
        XCTAssertFalse(QWDSparkDrafter.allowsPriorCommitOverlap(schedulerEnabled: false, verifyRows: 16, maxDrafts: 7))
        XCTAssertFalse(QWDSparkDrafter.allowsPriorCommitOverlap(schedulerEnabled: true, verifyRows: 8, maxDrafts: 7))
        XCTAssertTrue(QWEngineDriver.shouldDrainPendingCommitInSyncBranch(
            hasOwner: false, adaptiveDepth: false, hasDeviceEmbedding: true, hasColdEvents: false, hasDSpark: true))
        XCTAssertFalse(QWEngineDriver.shouldDrainPendingCommitInSyncBranch(
            hasOwner: false, adaptiveDepth: false, hasDeviceEmbedding: true, hasColdEvents: false, hasDSpark: false))
    }

    func testDSparkWeightsLoadsBundledDraftHeadFromPrimaryPack() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && MTLCreateSystemDefaultDevice() != nil,
            "Requires Apple Silicon Metal device"
        )
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-dspark-bundled-head-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticTargetAndDSparkPack(at: rootDir)

        let cfgData = try Data(contentsOf: rootDir.appendingPathComponent("assets/config.json"))
        let descriptor = try QWModelDescriptor.decode(configJSON: cfgData)
        let geometry = QWModelGeometry(descriptor: descriptor)
        let dsConfig = try XCTUnwrap(descriptor.dsparkConfig)
        let hal = try QWHal(geometry: geometry)
        let dsURL = rootDir.appendingPathComponent("dspark.qwp")
        let baseData = try Data(contentsOf: dsURL)
        let V = geometry.vocab, H = geometry.hidden, rot = hal.weightBasis.rotatesActivations

        func writeWithBundledTensor(name: String, bytes tensorBytes: Data, isRaw: Bool, extraMeta: [String: Any]) throws {
            let hdrLen = baseData.subdata(in: 4..<12).withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                Int(UInt64(littleEndian: raw.load(as: UInt64.self)))
            }
            var payload = Data(baseData.prefix(baseData.count - hdrLen))
            let rem = payload.count % 16_384
            if rem != 0 { payload.append(Data(repeating: 0, count: 16_384 - rem)) }
            let off = payload.count
            payload.append(tensorBytes)
            let tailRem = payload.count % 16_384
            if tailRem != 0 { payload.append(Data(repeating: 0, count: 16_384 - tailRem)) }
            var hdrObj = try XCTUnwrap(
                try JSONSerialization.jsonObject(with: baseData.suffix(hdrLen)) as? [String: Any]
            )
            var tensors = try XCTUnwrap(hdrObj["tensors"] as? [String: Any])
            var entry = extraMeta
            entry["shape"] = [V, H]
            entry[isRaw ? "raw_off" : "hq4_off"] = off
            tensors[name] = entry
            hdrObj["tensors"] = tensors
            let newHdr = try JSONSerialization.data(withJSONObject: hdrObj, options: [.sortedKeys])
            var newHdrLen = UInt64(newHdr.count).littleEndian
            withUnsafeBytes(of: &newHdrLen) { (buf: UnsafeRawBufferPointer) in payload.replaceSubrange(4..<12, with: buf) }
            payload.append(newHdr)
            try payload.write(to: dsURL)
        }

        try writeWithBundledTensor(
            name: "draft_head.weight",
            bytes: Self.makeSHVQData(bits: 2, group: 64, n: V, k: H, seedInit: 99, scale: 0.02),
            isRaw: false,
            extraMeta: ["hq": ["bits": 2, "group": 64, "vq": "shvq1"]]
        )
        let wShvq = try QWDSparkWeights(geometry: geometry, hal: hal, config: dsConfig, packURL: dsURL, draftHeadURL: nil)
        guard case .quantized(let vShvq) = wShvq.draftHead else {
            return XCTFail("Expected quantized bundled shvq2_g64 draftHead")
        }
        XCTAssertTrue(vShvq.bits == 2 && vShvq.groupSize == 64 && vShvq.isSHVQ && !vShvq.isIncumbentContainer && vShvq.isRotated == rot)

        try writeWithBundledTensor(
            name: "draft_head.weight", bytes: Data(repeating: 0, count: V * (H / 32) * 20), isRaw: false, extraMeta: [:]
        )
        let wInc = try QWDSparkWeights(geometry: geometry, hal: hal, config: dsConfig, packURL: dsURL, draftHeadURL: nil)
        guard case .quantized(let vInc) = wInc.draftHead else {
            return XCTFail("Expected quantized bundled hq4_g32_m5 draftHead")
        }
        XCTAssertTrue(vInc.bits == 4 && vInc.groupSize == 32 && !vInc.isSHVQ && vInc.isIncumbentContainer == rot && vInc.isRotated == rot)

        try writeWithBundledTensor(
            name: "dspark.draft_head.weight", bytes: Data(count: V * H * 2), isRaw: true, extraMeta: ["kind": "bf16"]
        )
        let wBf16 = try QWDSparkWeights(geometry: geometry, hal: hal, config: dsConfig, packURL: dsURL, draftHeadURL: nil)
        guard case .bf16(_, let nBf16, let kBf16) = wBf16.draftHead else {
            return XCTFail("Expected bf16 bundled dspark.draft_head.weight")
        }
        XCTAssertEqual(nBf16, V)
        XCTAssertEqual(kBf16, H)
    }
}

