// QWAttnSinkRelSparseTests.swift — Sparse prefill/decode parity and LSA Ward/e-means end-to-end tests.
import Foundation
import Metal
import XCTest
@testable import QWAttn
import QWAlloc
import QWCore
import QWHAL
import QWKV
import QWQuant

final class QWAttnSinkRelSparseTests: XCTestCase {
    private struct SplitMix64 {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &+ 0x9E3779B97F4A7C15
            var z = (state ^ (state >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        mutating func nextSigned(_ scale: Double = 1.0) -> Float {
            Float((Double(next() >> 11) / Double(1 << 53) * 2.0 - 1.0) * scale)
        }
    }

    private static func makeGeometry(qHeads: Int = 12, kvHeads: Int = 4, headDim: Int = 128, maxPos: Int = 8192) throws -> QWModelGeometry {
        let cfg: [String: Any] = [
            "model_type": "qwen3", "hidden_size": qHeads * headDim, "num_hidden_layers": 1,
            "num_attention_heads": qHeads, "num_key_value_heads": kvHeads, "head_dim": headDim,
            "intermediate_size": 512, "max_position_embeddings": maxPos, "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0, "eos_token_id": 2, "vocab_size": 128,
        ]
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: JSONSerialization.data(withJSONObject: cfg)))
    }

    private static func f32(_ bits: UInt16) -> Float { Float(bitPattern: UInt32(bits) << 16) }
    private static func synth(_ count: Int, seed: UInt64, scale: Double) -> [Float] {
        var rng = SplitMix64(state: seed)
        return (0..<count).map { _ in f32(QWBF16.fromFloatRNE(rng.nextSigned(scale))) }
    }

    private static func bf16Tensor(_ hal: QWHal, _ vals: [Float]? = nil, dims: [Int]) -> QWTensorValue {
        let count = dims.reduce(1, *)
        let bytes = max(16, count * 2)
        let buf = hal.device.makeBuffer(length: bytes, options: .storageModeShared)!
        memset(buf.contents(), 0, bytes)
        if let vals {
            vals.map(QWBF16.fromFloatRNE).withUnsafeBytes { raw in
                if let src = raw.baseAddress, !raw.isEmpty { memcpy(buf.contents(), src, raw.count) }
            }
        }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: bytes), offset: 0, shape: QWTensorShape(dims, .bf16))
    }

    private static func readBF16(_ tv: QWTensorValue, count: Int) -> [Float] {
        let ptr = tv.region.buffer.contents().advanced(by: tv.region.base + tv.offset).assumingMemoryBound(to: UInt16.self)
        return (0..<count).map { f32(ptr[$0]) }
    }

    private static func intTensor(_ hal: QWHal, _ vals: [Int32], dims: [Int]) -> QWTensorValue {
        let bytes = max(16, vals.count * 4)
        let buf = hal.device.makeBuffer(length: bytes, options: .storageModeShared)!
        memset(buf.contents(), 0, bytes)
        vals.withUnsafeBytes { raw in if let src = raw.baseAddress, !raw.isEmpty { memcpy(buf.contents(), src, raw.count) } }
        return QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: bytes), offset: 0, shape: QWTensorShape(dims, .i32))
    }

    private static func makeFullBlocks(_ hal: QWHal, rows: Int, numBlocks: Int, listStride: Int) -> (list: QWTensorValue, count: QWTensorValue) {
        var list = [Int32](repeating: 0, count: rows * listStride)
        for r in 0..<rows { for b in 0..<numBlocks { list[r * listStride + b] = Int32(b) } }
        return (intTensor(hal, list, dims: [rows, listStride]), intTensor(hal, [Int32](repeating: Int32(numBlocks), count: rows), dims: [rows]))
    }

    private static func buildKVCache(
        hal: QWHal, mode: QWKVMode, kRaw: [Float], vRaw: [Float], totalTokens: Int, nKV: Int, headDim: Int
    ) -> QWSinkRelKVCache {
        let rowBytes = QWAttnSinkRel.kvRowBytes(mode: mode, nKV: nKV, headDim: headDim)
        let payBytes = max(16, totalTokens * rowBytes.payload)
        let metaBytes = max(16, totalTokens * max(4, rowBytes.meta))
        let kBuf = hal.device.makeBuffer(length: payBytes, options: .storageModeShared)!
        let vBuf = hal.device.makeBuffer(length: payBytes, options: .storageModeShared)!
        let kmBuf = hal.device.makeBuffer(length: metaBytes, options: .storageModeShared)!
        let vmBuf = hal.device.makeBuffer(length: metaBytes, options: .storageModeShared)!
        memset(kBuf.contents(), 0, payBytes); memset(vBuf.contents(), 0, payBytes)
        memset(kmBuf.contents(), 0, metaBytes); memset(vmBuf.contents(), 0, metaBytes)

        for (src, payBuf, metaBuf) in [(kRaw, kBuf, kmBuf), (vRaw, vBuf, vmBuf)] {
            for pos in 0..<totalTokens {
                for kvh in 0..<nKV {
                    let srcOff = (pos * nKV + kvh) * headDim, rowHead = pos * nKV + kvh
                    let vals = (0..<headDim).map { Float(Float16(src[srcOff + $0])) }
                    switch mode {
                    case .f16:
                        let dst = payBuf.contents().assumingMemoryBound(to: UInt16.self)
                        for d in 0..<headDim { dst[rowHead * headDim + d] = Float16(vals[d]).bitPattern }
                    case .q8:
                        let sc16 = Float16(max((vals.map(abs).max() ?? 0) / 127.0, 6.103515625e-5)), invSc = 1.0 / Float(sc16)
                        let dst = payBuf.contents().assumingMemoryBound(to: Int8.self)
                        metaBuf.contents().assumingMemoryBound(to: UInt16.self)[rowHead] = sc16.bitPattern
                        for d in 0..<headDim { dst[rowHead * headDim + d] = Int8(max(-127, min(127, Int((vals[d] * invSc).rounded(.toNearestOrEven))))) }
                    case .q4:
                        let mn16 = Float16(vals.min() ?? 0), sc16 = Float16(max(((vals.max() ?? 0) - Float(mn16)) / 15.0, 6.103515625e-5))
                        let invSc = 1.0 / max(Float(sc16), 1e-7)
                        let dst = payBuf.contents().assumingMemoryBound(to: UInt8.self)
                        let meta = metaBuf.contents().assumingMemoryBound(to: UInt16.self)
                        meta[rowHead * 2] = sc16.bitPattern; meta[rowHead * 2 + 1] = mn16.bitPattern
                        for p in 0..<(headDim / 2) {
                            let q0 = max(0, min(15, Int(((vals[p * 2] - Float(mn16)) * invSc).rounded(.toNearestOrEven))))
                            let q1 = max(0, min(15, Int(((vals[p * 2 + 1] - Float(mn16)) * invSc).rounded(.toNearestOrEven))))
                            dst[rowHead * (headDim / 2) + p] = UInt8(q0 | (q1 << 4))
                        }
                    }
                }
            }
        }
        return QWSinkRelKVCache(
            kCache: QWMRegion(buffer: kBuf, base: 0, length: payBytes),
            vCache: QWMRegion(buffer: vBuf, base: 0, length: payBytes),
            kMeta: mode == .f16 ? nil : QWMRegion(buffer: kmBuf, base: 0, length: metaBytes),
            vMeta: mode == .f16 ? nil : QWMRegion(buffer: vmBuf, base: 0, length: metaBytes),
            ringCap: totalTokens, mode: mode
        )
    }

    private static func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        XCTAssertEqual(a.count, b.count)
        var worst: Float = 0
        for i in 0..<a.count {
            XCTAssertTrue(a[i].isFinite && b[i].isFinite, "non-finite at \(i)")
            worst = max(worst, abs(a[i] - b[i]))
        }
        return worst
    }

    func testSparsePrefillMatchesDenseAcrossKVModes() throws {
        let geom = try Self.makeGeometry(qHeads: 12, kvHeads: 4, headDim: 128)
        let hal = try QWHal(geometry: geom)
        try XCTSkipUnless(hal.caps.isMetal4TensorCapable, "NAX sparse sink/rel prefill requires Apple10 Metal 4 hardware")
        let attn = try QWAttnSinkRel(hal: hal)
        let nH = 12, nKV = 4, headDim = 128, T = 65, posOffset = 35, dRel = 16, relExtent = 48
        let kvLen = posOffset + T, numTiles = (T + 31) / 32, totalBlocks = (kvLen + 3) / 4, listStride = 32

        for (idx, mode) in [QWKVMode.f16, .q8, .q4].enumerated() {
            let s = UInt64(1000 + idx * 17)
            let cfg = QWSinkRelConfig(nH: nH, nKV: nKV, headDim: headDim, dRel: dRel, relExtent: relExtent, hasSink: true, hasRelBias: true, hasOutGate: true, kvMode: mode)
            let qTV = Self.bf16Tensor(hal, Self.synth(T * nH * headDim, seed: s + 1, scale: 1.5), dims: [T, nH, headDim])
            let rTV = Self.bf16Tensor(hal, Self.synth(T * nH * dRel, seed: s + 2, scale: 1.2), dims: [T, nH, dRel])
            let rpTV = Self.bf16Tensor(hal, Self.synth(dRel * relExtent, seed: s + 3, scale: 0.25), dims: [dRel, relExtent])
            let sTV = Self.bf16Tensor(hal, Self.synth(T * nH, seed: s + 4, scale: 1.0), dims: [T, nH])
            let gTV = Self.bf16Tensor(hal, Self.synth(T * nH * headDim, seed: s + 5, scale: 1.5), dims: [T, nH, headDim])
            let kv = Self.buildKVCache(hal: hal, mode: mode, kRaw: Self.synth(kvLen * nKV * headDim, seed: s + 6, scale: 1.5), vRaw: Self.synth(kvLen * nKV * headDim, seed: s + 7, scale: 0.8), totalTokens: kvLen, nKV: nKV, headDim: headDim)
            let denseOut = Self.bf16Tensor(hal, dims: [T, nH, headDim])
            let sparseOut = Self.bf16Tensor(hal, dims: [T, nH, headDim])
            let blocks = Self.makeFullBlocks(hal, rows: numTiles, numBlocks: totalBlocks, listStride: listStride)

            guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin failed"); return }
            try attn.encodePrefill(enc, config: cfg, q: qTV, r: rTV, relProj: rpTV, sink: sTV, gate: gTV, kv: kv, out: denseOut, T: T, posOffset: posOffset)
            enc.memoryBarrier(scope: .buffers)
            try attn.encodePrefillSparse(enc, config: cfg, q: qTV, r: rTV, relProj: rpTV, sink: sTV, gate: gTV, kv: kv, out: sparseOut, blockList: blocks.list, blockCount: blocks.count, listStride: listStride, T: T, posOffset: posOffset, perTileBlockList: true)
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            XCTAssertEqual(cb.status, .completed, "prefill sparse failed mode=\(mode): \(String(describing: cb.error))")
            XCTAssertLessThanOrEqual(Self.maxAbsDiff(Self.readBF16(sparseOut, count: T * nH * headDim), Self.readBF16(denseOut, count: T * nH * headDim)), 1.5e-2, "mode=\(mode)")
        }
    }

    func testSparseDecodeMatchesDenseAndT4BitExact() throws {
        let nH = 16, nKV = 4, headDim = 128, posOffset = 259, dRel = 16, relExtent = 64, totalKV = posOffset + 4
        let hal = try QWHal(geometry: try Self.makeGeometry(qHeads: nH, kvHeads: nKV, headDim: headDim))
        let attn = try QWAttnSinkRel(hal: hal)

        for (mIdx, mode) in [QWKVMode.f16, .q8, .q4].enumerated() {
            let s = UInt64(3000 + mIdx * 31)
            let cfg = QWSinkRelConfig(nH: nH, nKV: nKV, headDim: headDim, dRel: dRel, relExtent: relExtent, hasSink: true, hasRelBias: true, hasOutGate: true, kvMode: mode)
            let qAll = Self.synth(4 * nH * headDim, seed: s + 1, scale: 1.5)
            let rAll = Self.synth(4 * nH * dRel, seed: s + 2, scale: 1.2)
            let rpTV = Self.bf16Tensor(hal, Self.synth(dRel * relExtent, seed: s + 3, scale: 0.25), dims: [dRel, relExtent])
            let sinkAll = Self.synth(4 * nH, seed: s + 4, scale: 1.0)
            let gateAll = Self.synth(4 * nH * headDim, seed: s + 5, scale: 1.5)
            let kv = Self.buildKVCache(hal: hal, mode: mode, kRaw: Self.synth(totalKV * nKV * headDim, seed: s + 6, scale: 1.5), vRaw: Self.synth(totalKV * nKV * headDim, seed: s + 7, scale: 0.8), totalTokens: totalKV, nKV: nKV, headDim: headDim)

            var refT1Concat = [Float]()
            for t in 0..<4 {
                let q1 = Self.bf16Tensor(hal, Array(qAll[(t * nH * headDim)..<((t + 1) * nH * headDim)]), dims: [1, nH, headDim])
                let r1 = Self.bf16Tensor(hal, Array(rAll[(t * nH * dRel)..<((t + 1) * nH * dRel)]), dims: [1, nH, dRel])
                let s1 = Self.bf16Tensor(hal, Array(sinkAll[(t * nH)..<((t + 1) * nH)]), dims: [1, nH])
                let g1 = Self.bf16Tensor(hal, Array(gateAll[(t * nH * headDim)..<((t + 1) * nH * headDim)]), dims: [1, nH, headDim])
                let out1 = Self.bf16Tensor(hal, dims: [1, nH, headDim])
                guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin failed"); return }
                try attn.encodeDecode(enc, config: cfg, q: q1, r: r1, relProj: rpTV, sink: s1, gate: g1, kv: kv, out: out1, T: 1, posOffset: posOffset + t, splits: 1)
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                XCTAssertEqual(cb.status, .completed)
                refT1Concat.append(contentsOf: Self.readBF16(out1, count: nH * headDim))
            }

            let qTV = Self.bf16Tensor(hal, qAll, dims: [4, nH, headDim]), rTV = Self.bf16Tensor(hal, rAll, dims: [4, nH, dRel])
            let sTV = Self.bf16Tensor(hal, sinkAll, dims: [4, nH]), gTV = Self.bf16Tensor(hal, gateAll, dims: [4, nH, headDim])
            let blocks = Self.makeFullBlocks(hal, rows: 1, numBlocks: (totalKV + 3) / 4, listStride: 80)

            for (T, splits) in [(1, 1), (1, 4), (4, 1), (4, 4)] {
                let kvLen = posOffset + T
                let need = QWAttnSinkRel.partialsBytes(tokens: T, qHeads: nH, headDim: headDim, splits: splits)
                let pO = hal.device.makeBuffer(length: max(16, need.partOBytes), options: .storageModeShared)!
                let pML = hal.device.makeBuffer(length: max(16, need.partMLBytes), options: .storageModeShared)!
                let partials = splits > 1 ? QWSinkRelPartials(partO: QWMRegion(buffer: pO, base: 0, length: pO.length), partML: QWMRegion(buffer: pML, base: 0, length: pML.length)) : nil
                let denseOut = Self.bf16Tensor(hal, dims: [T, nH, headDim]), sparseOut = Self.bf16Tensor(hal, dims: [T, nH, headDim])
                guard let (cb, enc) = hal.begin() else { XCTFail("hal.begin failed"); return }
                try attn.encodeDecode(enc, config: cfg, q: qTV, r: rTV, relProj: rpTV, sink: sTV, gate: gTV, kv: kv, partials: partials, out: denseOut, T: T, posOffset: posOffset, kvLen: kvLen, splits: splits)
                enc.memoryBarrier(scope: .buffers)
                try attn.encodeDecodeSparse(enc, config: cfg, q: qTV, r: rTV, relProj: rpTV, sink: sTV, gate: gTV, kv: kv, partials: partials, out: sparseOut, blockList: blocks.list, blockCount: blocks.count, listStride: 80, T: T, posOffset: posOffset, kvLen: kvLen, splits: splits)
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                XCTAssertEqual(cb.status, .completed, "mode=\(mode) T=\(T) sp=\(splits)")
                let gotDense = Self.readBF16(denseOut, count: T * nH * headDim), gotSparse = Self.readBF16(sparseOut, count: T * nH * headDim)
                XCTAssertLessThanOrEqual(Self.maxAbsDiff(gotSparse, gotDense), 1.5e-2, "mode=\(mode) T=\(T) sp=\(splits)")
                if T == 4 && splits == 1 { XCTAssertEqual(Self.maxAbsDiff(gotDense, refT1Concat), 0.0, "T=4 fastT4 0-ULP parity failed mode=\(mode)") }
            }
        }
    }

    func testLSAWardAndEmeansBF16SparseEndToEndAt4096() throws {
        let nH = 12, nKV = 4, headDim = 128, kvLen = 4096
        let geom = try Self.makeGeometry(qHeads: nH, kvHeads: nKV, headDim: headDim, maxPos: 8192)
        let hal = try QWHal(geometry: geom), attn = try QWAttnSinkRel(hal: hal)
        let qTV = Self.bf16Tensor(hal, Self.synth(kvLen * nH * headDim, seed: 81, scale: 1.0), dims: [kvLen, nH, headDim])
        let kRaw = Self.synth(kvLen * nKV * headDim, seed: 82, scale: 1.0), vRaw = Self.synth(kvLen * nKV * headDim, seed: 83, scale: 0.8)
        let kTV = Self.bf16Tensor(hal, kRaw, dims: [kvLen, nKV, headDim])
        let kv = Self.buildKVCache(hal: hal, mode: .f16, kRaw: kRaw, vRaw: vRaw, totalTokens: kvLen, nKV: nKV, headDim: headDim)
        let sinkTV = Self.bf16Tensor(hal, Self.synth(64 * nH, seed: 84, scale: 0.5), dims: [64, nH])
        let gateTV = Self.bf16Tensor(hal, Self.synth(64 * nH * headDim, seed: 85, scale: 1.0), dims: [64, nH, headDim])
        let basis = QWLSASidecar.synthesizeDeterministicOrthogonalBasis(inDim: geom.lsaInputDim, indexDim: 64, seed: 777)
        let cfg = QWSinkRelConfig(nH: nH, nKV: nKV, headDim: headDim, hasSink: true, hasOutGate: true, kvMode: .f16)

        for clusterBlock in [512, 4096] {
            let lsaCfg = QWLSAConfig(mode: .svd, budget: 512, clusterBlock: clusterBlock)
            let lsa = try QWLSAOperator(hal: hal, geometry: geom, config: lsaCfg, indexDim: 64, layerWeights: [.init(physicalLayer: 0, wqIdxF16: basis, wkIdxF16: basis)], maxContext: kvLen, maxPrefillChunk: 512)
            let cntPtr = lsa.region.pointer.advanced(by: lsa.blockCountScratch.offset).assumingMemoryBound(to: Int32.self)
            guard let (cb1, enc1) = hal.begin() else { XCTFail("hal.begin failed"); return }
            for pos0 in stride(from: 0, to: 4032, by: 512) {
                let rows = min(512, 4032 - pos0)
                let qS = QWTensorValue(region: qTV.region, offset: pos0 * nH * headDim * 2, shape: QWTensorShape([rows, nH, headDim], .bf16))
                let kS = QWTensorValue(region: kTV.region, offset: pos0 * nKV * headDim * 2, shape: QWTensorShape([rows, nKV, headDim], .bf16))
                lsa.encodeIndexAndClusterBF16(enc1, qIn: qS, kIn: kS, fullAttnLayer: 0, pos0: pos0, rows: rows, qRowStride: nH * headDim, qMode: .none)
                enc1.memoryBarrier(scope: .buffers)
            }
            let q64 = QWTensorValue(region: qTV.region, offset: 4032 * nH * headDim * 2, shape: QWTensorShape([64, nH, headDim], .bf16))
            let k64 = QWTensorValue(region: kTV.region, offset: 4032 * nKV * headDim * 2, shape: QWTensorShape([64, nKV, headDim], .bf16))
            lsa.encodeIndexAndClusterBF16(enc1, qIn: q64, kIn: k64, fullAttnLayer: 0, pos0: 4032, rows: 64, qRowStride: nH * headDim, qMode: .tiled32)
            enc1.memoryBarrier(scope: .buffers)
            lsa.encodeSelectBlocks(enc1, fullAttnLayer: 0, pos0: 4032, rows: 64, tiled32: true)
            let preOut = Self.bf16Tensor(hal, dims: [64, nH, headDim])
            if hal.caps.isMetal4TensorCapable {
                enc1.memoryBarrier(scope: .buffers)
                try attn.encodePrefillSparse(enc1, config: cfg, q: q64, sink: sinkTV, gate: gateTV, kv: kv, out: preOut, blockList: lsa.blockListScratch, blockCount: lsa.blockCountScratch, listStride: lsa.listStride, T: 64, posOffset: 4032, kvLen: kvLen)
            }
            enc1.endEncoding(); cb1.commit(); cb1.waitUntilCompleted()
            XCTAssertEqual(cb1.status, .completed)
            XCTAssertTrue((1...(1 + lsaCfg.budgetBlocks + 128)).contains(Int(cntPtr[63])), "selected=\(cntPtr[63])")
            if hal.caps.isMetal4TensorCapable { XCTAssertTrue(Self.readBF16(preOut, count: 64 * nH * headDim).allSatisfy(\.isFinite)) }

            let q4 = QWTensorValue(region: qTV.region, offset: 4092 * nH * headDim * 2, shape: QWTensorShape([4, nH, headDim], .bf16))
            let k4 = QWTensorValue(region: kTV.region, offset: 4092 * nKV * headDim * 2, shape: QWTensorShape([4, nKV, headDim], .bf16))
            let decOut = Self.bf16Tensor(hal, dims: [4, nH, headDim])
            guard let (cb2, enc2) = hal.begin() else { XCTFail("hal.begin failed"); return }
            lsa.encodeIndexAndClusterBF16(enc2, qIn: q4, kIn: k4, fullAttnLayer: 0, pos0: 4092, rows: 4, qRowStride: nH * headDim, qMode: .allRows)
            enc2.memoryBarrier(scope: .buffers)
            lsa.encodeSelectBlocks(enc2, fullAttnLayer: 0, pos0: 4092, rows: 4, tiled32: false)
            enc2.memoryBarrier(scope: .buffers)
            try attn.encodeDecodeSparse(enc2, config: cfg, q: q4, sink: sinkTV, gate: gateTV, kv: kv, out: decOut, blockList: lsa.blockListScratch, blockCount: lsa.blockCountScratch, listStride: lsa.listStride, T: 4, posOffset: 4092, kvLen: kvLen, splits: 1)
            enc2.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()
            XCTAssertEqual(cb2.status, .completed)
            XCTAssertTrue((1...(1 + lsaCfg.budgetBlocks + 128)).contains(Int(cntPtr[0])), "decode selected=\(cntPtr[0])")
            XCTAssertTrue(Self.readBF16(decOut, count: 4 * nH * headDim).allSatisfy(\.isFinite))
        }
    }
}
