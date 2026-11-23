// QWAttnSlidingWindowTests.swift — GPU vs CPU parity for sliding_window GQA decode and paged prefill.
import XCTest
import Metal
import Foundation
@testable import QWAttn
import QWCore
import QWHAL
import QWAlloc
import QWKV
import QWQuant

final class QWAttnSlidingWindowTests: XCTestCase {

    private static func makeGeometry(
        qHeads: Int = 8,
        kvHeads: Int = 2,
        headDim: Int = 128,
        slidingWindow: Int?,
        maxPos: Int = 2048,
        gated: Bool = false
    ) throws -> QWModelGeometry {
        var cfg: [String: Any] = [
            "model_type": "mistral",
            "hidden_size": qHeads * headDim,
            "num_hidden_layers": 1,
            "num_attention_heads": qHeads,
            "num_key_value_heads": kvHeads,
            "head_dim": headDim,
            "intermediate_size": 512,
            "max_position_embeddings": maxPos,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "eos_token_id": 2,
            "vocab_size": 256,
        ]
        if let slidingWindow { cfg["sliding_window"] = slidingWindow }
        if gated { cfg["attn_output_gate"] = true }
        let data = try JSONSerialization.data(withJSONObject: cfg)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    private static func f16Bits(_ v: Float) -> UInt16 { QWQuant.f32tof16Bits(v) }
    private static func f16(_ bits: UInt16) -> Float { QWQuant.f16tof32Bits(bits) }

    private static func det(_ seed: Int) -> Float {
        let m = (seed &* 2654435761) & 0xFFFF
        return Float(Int(m % 2001) - 1000) / 1000.0
    }

    private static func synthKV(tokens: Int, geometry: QWModelGeometry) -> (k: [UInt16], v: [UInt16]) {
        let rowElems = geometry.kvHeads * geometry.headDim
        var k = [UInt16](repeating: 0, count: tokens * rowElems)
        var v = [UInt16](repeating: 0, count: tokens * rowElems)
        for t in 0..<tokens {
            for h in 0..<geometry.kvHeads {
                for d in 0..<geometry.headDim {
                    let e = h * geometry.headDim + d
                    // Keep quantized values well-conditioned across F16/Q8/Q4.
                    k[t * rowElems + e] = f16Bits(sin(Float(t * 29 + e * 7)) * 0.45)
                    v[t * rowElems + e] = f16Bits(cos(Float(t * 17 + e * 11)) * 0.45)
                }
            }
        }
        return (k, v)
    }

    private static func synthQGate(tokens: Int, geometry: QWModelGeometry) -> [UInt16] {
        var out = [UInt16](repeating: 0, count: tokens * geometry.qProjOut)
        let stride = (geometry.attnOutputGate ? 2 : 1) * geometry.headDim
        for t in 0..<tokens {
            for h in 0..<geometry.qHeads {
                let base = t * geometry.qProjOut + h * stride
                for d in 0..<geometry.headDim {
                    let val: Float = (d == 0) ? 1.5 : det(t * 131 + h * 37 + d * 13) * 0.35
                    out[base + d] = f16Bits(val)
                }
                if geometry.attnOutputGate {
                    for d in 0..<geometry.headDim {
                        out[base + geometry.headDim + d] = f16Bits(det(t * 73 + h * 19 + d * 5) * 0.5)
                    }
                }
            }
        }
        return out
    }

    private static func referenceRow(
        q: [Float],
        kFlat: [Float],
        vFlat: [Float],
        kvRowElems: Int,
        kvh: Int,
        headDim: Int,
        kvStart: Int,
        kvEnd: Int,
        gate: [Float]?
    ) -> [Float] {
        let scale = Float(1.0 / Double(headDim).squareRoot())
        var scores = [Float](repeating: 0, count: kvEnd - kvStart)
        for j in kvStart..<kvEnd {
            let kBase = j * kvRowElems + kvh * headDim
            var dot: Float = 0
            for d in 0..<headDim { dot += q[d] * kFlat[kBase + d] }
            scores[j - kvStart] = dot * scale
        }
        let m = scores.max() ?? 0
        var l: Float = 0
        var o = [Float](repeating: 0, count: headDim)
        for j in kvStart..<kvEnd {
            let p = exp(scores[j - kvStart] - m)
            l += p
            let vBase = j * kvRowElems + kvh * headDim
            for d in 0..<headDim { o[d] += p * vFlat[vBase + d] }
        }
        return QWAttnPagedPrefillGeneralTests.rotated(o, l: l, headDim: headDim, gate: gate)
    }

    private func writePagedKV(
        hal: QWHal,
        geometry: QWModelGeometry,
        mode: QWKVMode,
        kBits: [UInt16],
        vBits: [UInt16],
        totalTokens: Int
    ) throws -> (arena: QWArena, views: (k: QWKVLayerView, v: QWKVLayerView), kDeq: [Float], vDeq: [Float]) {
        let ceiling = ((totalTokens + 255) / 256) * 256
        let pageBytes = QWKVCache.pageBytes(mode: mode, geometry: geometry)
        let poolBytes = pageBytes * (ceiling / 256) * 2
        let tables = QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs
        let arena = try QWArena(hal: hal, weights: 0, activations: 0,
                                kv: poolBytes + tables,
                                gdnStates: QWAttnDecode.decodePartialsBytes(geometry: geometry))
        let cache = try QWKVCache(hal: hal, arena: arena, layers: [0], mode: mode, geometry: geometry)
        let handle = try cache.allocate(promptTokens: ceiling)
        try cache.append(handle, positions: 0..<totalTokens, layer: 0, kind: .k, src: kBits)
        try cache.append(handle, positions: 0..<totalTokens, layer: 0, kind: .v, src: vBits)
        let views = try cache.layerViews(handle, layer: 0)
        // Quantized CPU reference follows Q8/Q4 rounding per 128-half group matching writer.
        let kDeq: [Float]
        let vDeq: [Float]
        if mode == .f16 {
            kDeq = kBits.map { Self.f16($0) }
            vDeq = vBits.map { Self.f16($0) }
        } else {
            let rowElems = geometry.kvHeads * geometry.headDim
            kDeq = Self.quantizeDeq(kBits, rows: totalTokens, width: rowElems, headDim: geometry.headDim, mode: mode)
            vDeq = Self.quantizeDeq(vBits, rows: totalTokens, width: rowElems, headDim: geometry.headDim, mode: mode)
        }
        return (arena, views, kDeq, vDeq)
    }

    private static func quantizeDeq(
        _ data: [UInt16],
        rows: Int,
        width: Int,
        headDim: Int,
        mode: QWKVMode
    ) -> [Float] {
        var out = [Float](repeating: 0, count: rows * width)
        let group = (mode == .q8) ? headDim : 128
        for r in 0..<rows {
            for start in stride(from: 0, to: width, by: group) {
                let end = min(start + group, width)
                let vals = (start..<end).map { f16(data[r * width + $0]) }
                let lo = vals.min() ?? 0
                let hi = vals.max() ?? 0
                let minVal: Float = (mode == .q8) ? 0 : Float(Float16(lo))
                let absMax = vals.map(abs).max() ?? 0
                let rawScale = (mode == .q8) ? (absMax / 127.0) : ((hi - lo) / 15.0)
                let scale = Float(Float16(max(rawScale, 1e-8)))
                for i in start..<end {
                    let q = ((f16(data[r * width + i]) - minVal) / scale).rounded()
                    let c = (mode == .q8) ? max(-127, min(127, q)) : max(0, min(15, q))
                    out[r * width + i] = Float(Float16(c * scale + minVal))
                }
            }
        }
        return out
    }

    private func runDecodeParity(mode: QWKVMode, C: Int, slidingWindow: Int, splits: Int) throws {
        let geometry = try Self.makeGeometry(slidingWindow: slidingWindow > 0 ? slidingWindow : nil)
        let hal = try QWHal(geometry: geometry)
        let rowElems = geometry.kvHeads * geometry.headDim
        var (kBits, vBits) = Self.synthKV(tokens: C + 1, geometry: geometry)
        // Poison pre-window entries with opposing V so any mask leak breaks parity immediately.
        let kvStart = (slidingWindow > 0 && C + 1 > slidingWindow) ? (C + 1 - slidingWindow) : 0
        for t in 0..<kvStart {
            for h in 0..<geometry.kvHeads {
                for d in 0..<geometry.headDim {
                    let e = h * geometry.headDim + d
                    kBits[t * rowElems + e] = Self.f16Bits(d == 0 ? 8.0 : 0.25)
                    vBits[t * rowElems + e] = Self.f16Bits(-3.0)
                }
            }
        }
        let qBits = Self.synthQGate(tokens: 1, geometry: geometry)
        let prefixK = Array(kBits.prefix(C * rowElems))
        let prefixV = Array(vBits.prefix(C * rowElems))
        let (arena, views, kDeqPrefix, vDeqPrefix) = try writePagedKV(
            hal: hal, geometry: geometry, mode: mode, kBits: prefixK, vBits: prefixV, totalTokens: C)
        _ = arena

        let knewBits = Array(kBits[(C * rowElems)..<((C + 1) * rowElems)])
        let vnewBits = Array(vBits[(C * rowElems)..<((C + 1) * rowElems)])
        let qBuf = hal.device.makeBuffer(bytes: qBits, length: qBits.count * 2, options: .storageModeShared)!
        let kNewBuf = hal.device.makeBuffer(bytes: knewBits, length: knewBits.count * 2, options: .storageModeShared)!
        let vNewBuf = hal.device.makeBuffer(bytes: vnewBits, length: vnewBits.count * 2, options: .storageModeShared)!
        let outHalves = geometry.qHeads * geometry.headDim
        let outBuf = hal.device.makeBuffer(length: outHalves * 2, options: .storageModeShared)!
        let pBytes = QWAttnDecode.decodePartialsBytes(geometry: geometry, maxSplits: max(QWAttnDecode.maxSplits, splits))
        let pBuf = hal.device.makeBuffer(length: pBytes, options: .storageModeShared)!

        let dec = try QWAttnDecode(hal: hal, kvMode: mode, geometry: geometry)
        guard let (cb, enc) = hal.begin() else { throw NSError(domain: "hal", code: 1) }
        try dec.encode(
            enc,
            qgate: QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length),
                                 offset: 0, shape: QWTensorShape([geometry.qProjOut], .f16)),
            kv: views,
            knew: QWTensorValue(region: QWMRegion(buffer: kNewBuf, base: 0, length: kNewBuf.length),
                                offset: 0, shape: QWTensorShape([rowElems], .f16)),
            vnew: QWTensorValue(region: QWMRegion(buffer: vNewBuf, base: 0, length: vNewBuf.length),
                                offset: 0, shape: QWTensorShape([rowElems], .f16)),
            partials: QWMRegion(buffer: pBuf, base: 0, length: pBytes),
            out: QWTensorValue(region: QWMRegion(buffer: outBuf, base: 0, length: outBuf.length),
                               offset: 0, shape: QWTensorShape([outHalves], .f16)),
            C: C, splits: splits)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        XCTAssertEqual(cb.status, .completed, "decode failed: \(String(describing: cb.error))")

        let kAll = kDeqPrefix + knewBits.map { Self.f16($0) }
        let vAll = vDeqPrefix + vnewBits.map { Self.f16($0) }
        let outPtr = outBuf.contents().assumingMemoryBound(to: UInt16.self)
        let tol: Float = (mode == .f16) ? 2e-3 : 4e-3
        for h in 0..<geometry.qHeads {
            let q = (0..<geometry.headDim).map { Self.f16(qBits[h * geometry.headDim + $0]) }
            let want = Self.referenceRow(
                q: q, kFlat: kAll, vFlat: vAll, kvRowElems: rowElems,
                kvh: h / geometry.gqaRatio, headDim: geometry.headDim,
                kvStart: kvStart, kvEnd: C + 1, gate: nil)
            for d in 0..<geometry.headDim {
                let got = Self.f16(outPtr[h * geometry.headDim + d])
                XCTAssertEqual(got, want[d], accuracy: tol,
                               "decode mode=\(mode) C=\(C) sw=\(slidingWindow) splits=\(splits) h=\(h) d=\(d)")
            }
        }
    }

    func testDecodeGeneralSlidingWindowParityAcrossKVModes() throws {
        for mode in [QWKVMode.f16, .q8, .q4] {
            try runDecodeParity(mode: mode, C: 48, slidingWindow: 16, splits: 1)
            try runDecodeParity(mode: mode, C: 280, slidingWindow: 36, splits: 12)
            // Full causal (slidingWindow == nil -> 0) regression parity.
            try runDecodeParity(mode: mode, C: 48, slidingWindow: 0, splits: 8)
        }
    }

    func testPagedPrefillGeneralAndNAXSlidingWindowParityAcrossKVModes() throws {
        let pos = 250
        let T = 36
        let slidingWindow = 24
        let total = pos + T
        let geometry = try Self.makeGeometry(slidingWindow: slidingWindow)
        let hal = try QWHal(geometry: geometry)
        let rowElems = geometry.kvHeads * geometry.headDim
        let sp = (T + 31) / 32 * 32
        var (kBits, vBits) = Self.synthKV(tokens: total, geometry: geometry)
        // Poison earliest chunk tokens strictly outside every query's window [0..<pos+1-sw].
        let globalDead = max(0, (pos + 1) - slidingWindow)
        for t in 0..<globalDead {
            for h in 0..<geometry.kvHeads {
                for d in 0..<geometry.headDim {
                    let e = h * geometry.headDim + d
                    kBits[t * rowElems + e] = Self.f16Bits(d == 0 ? 8.0 : 0.25)
                    vBits[t * rowElems + e] = Self.f16Bits(-3.0)
                }
            }
        }
        let qBits = Self.synthQGate(tokens: sp, geometry: geometry)
        let qBuf = hal.device.makeBuffer(bytes: qBits, length: qBits.count * 2, options: .storageModeShared)!
        let qTV = QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length),
                                offset: 0, shape: QWTensorShape([sp * geometry.qProjOut], .f16))
        let outElems = T * geometry.qHeads * geometry.headDim

        for mode in [QWKVMode.f16, .q8, .q4] {
            let (arena, views, kDeq, vDeq) = try writePagedKV(
                hal: hal, geometry: geometry, mode: mode, kBits: kBits, vBits: vBits, totalTokens: total)
            _ = arena

            // 1. General scalar prefill
            let genOp = try QWAttnPagedPrefillGeneral(hal: hal, geometry: geometry)
            let genOut = hal.device.makeBuffer(length: outElems * 2, options: .storageModeShared)!
            guard let (cbG, encG) = hal.begin() else { throw NSError(domain: "hal", code: 2) }
            try genOp.encode(
                encG, qgate: qTV, kv: views,
                out: QWTensorValue(region: QWMRegion(buffer: genOut, base: 0, length: genOut.length),
                                   offset: 0, shape: QWTensorShape([outElems], .f16)),
                pos: pos, T: T)
            encG.endEncoding()
            cbG.commit()
            cbG.waitUntilCompleted()
            XCTAssertEqual(cbG.status, .completed, "general prefill failed: \(String(describing: cbG.error))")

            let genPtr = genOut.contents().assumingMemoryBound(to: UInt16.self)
            let genTol: Float = (mode == .f16) ? 2e-3 : 4e-3
            let qStride = (geometry.attnOutputGate ? 2 : 1) * geometry.headDim
            for qi in 0..<T {
                let kvEnd = pos + qi + 1
                let kvStart = max(0, kvEnd - slidingWindow)
                for h in 0..<geometry.qHeads {
                    let qBase = qi * geometry.qProjOut + h * qStride
                    let q = (0..<geometry.headDim).map { Self.f16(qBits[qBase + $0]) }
                    let want = Self.referenceRow(
                        q: q, kFlat: kDeq, vFlat: vDeq, kvRowElems: rowElems,
                        kvh: h / geometry.gqaRatio, headDim: geometry.headDim,
                        kvStart: kvStart, kvEnd: kvEnd, gate: nil)
                    for d in 0..<geometry.headDim {
                        let got = Self.f16(genPtr[(qi * geometry.qHeads + h) * geometry.headDim + d])
                        XCTAssertEqual(got, want[d], accuracy: genTol,
                                       "general prefill mode=\(mode) qi=\(qi) h=\(h) d=\(d)")
                    }
                }
            }

            // 2. NAX paged prefill (when hardware supports Metal 4 tensors)
            if hal.caps.isMetal4TensorCapable {
                let naxOp = try QWAttnPagedPrefill(hal: hal, geometry: geometry)
                let naxOut = hal.device.makeBuffer(length: outElems * 2, options: .storageModeShared)!
                guard let (cbN, encN) = hal.begin() else { throw NSError(domain: "hal", code: 3) }
                try naxOp.encode(
                    encN, qgate: qTV, kv: views,
                    out: QWTensorValue(region: QWMRegion(buffer: naxOut, base: 0, length: naxOut.length),
                                       offset: 0, shape: QWTensorShape([outElems], .f16)),
                    pos: pos, T: T)
                encN.endEncoding()
                cbN.commit()
                cbN.waitUntilCompleted()
                XCTAssertEqual(cbN.status, .completed, "NAX prefill failed: \(String(describing: cbN.error))")

                let naxPtr = naxOut.contents().assumingMemoryBound(to: UInt16.self)
                let naxTol: Float = (mode == .f16) ? 3e-3 : 8e-3
                for i in 0..<outElems {
                    let actual = Self.f16(naxPtr[i])
                    let expected = Self.f16(genPtr[i])
                    XCTAssertTrue(actual.isFinite, "NAX non-finite at \(i)")
                    XCTAssertEqual(actual, expected, accuracy: naxTol, "NAX vs general mode=\(mode) idx=\(i)")
                }
            }
        }
    }
}
