// QWAttnGeneralEquivalenceTests.swift — Equivalence gates for general vs specialized
// GQA decode, multi-row sharedPrefix (B=2...8), and paged prefill across f16/q8/q4.
import Foundation
import Metal
import XCTest
@testable import QWAttn
import QWAlloc
import QWCore
import QWHAL
import QWKV
import QWQuant

final class QWAttnGeneralEquivalenceTests: XCTestCase {
    private static let layer = 0

    private static func makeGeometry(
        qHeads: Int,
        kvHeads: Int,
        headDim: Int,
        gated: Bool,
        slidingWindow: Int? = nil
    ) throws -> QWModelGeometry {
        var cfg: [String: Any] = [
            "model_type": slidingWindow == nil ? "llama" : "mistral",
            "hidden_size": qHeads * headDim,
            "num_hidden_layers": 1,
            "num_attention_heads": qHeads,
            "num_key_value_heads": kvHeads,
            "head_dim": headDim,
            "intermediate_size": 256,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "eos_token_id": 2,
            "vocab_size": 128,
        ]
        if gated { cfg["attn_output_gate"] = true }
        if let slidingWindow { cfg["sliding_window"] = slidingWindow }
        let data = try JSONSerialization.data(withJSONObject: cfg)
        return QWModelGeometry(descriptor: try QWModelDescriptor.decode(configJSON: data))
    }

    private static func f16Bits(_ v: Float) -> UInt16 { QWQuant.f32tof16Bits(v) }
    private static func f16(_ bits: UInt16) -> Float { QWQuant.f16tof32Bits(bits) }
    private static func f16Ordered(_ bits: UInt16) -> Int {
        (bits & 0x8000) != 0 ? -Int(bits & 0x7FFF) : Int(bits & 0x7FFF)
    }
    private static func f16ULP(_ a: UInt16, _ b: UInt16) -> Int {
        abs(f16Ordered(a) - f16Ordered(b))
    }

    private static func det(_ seed: Int) -> Float {
        let m = (seed &* 2654435761) & 0xFFFF
        return Float(Int(m % 2001) - 1000) / 1000.0
    }

    private static func makeBuf(_ hal: QWHal, _ vals: [UInt16]) throws -> any MTLBuffer {
        let b = try XCTUnwrap(hal.device.makeBuffer(length: max(16, vals.count * 2), options: .storageModeShared))
        vals.withUnsafeBytes { raw in
            if let base = raw.baseAddress { memcpy(b.contents(), base, raw.count) }
        }
        return b
    }

    private static func tv(_ buf: any MTLBuffer, _ count: Int) -> QWTensorValue {
        QWTensorValue(region: QWMRegion(buffer: buf, base: 0, length: buf.length),
                      offset: 0, shape: QWTensorShape([count], .f16))
    }

    private static func synthKV(rows: Int, geometry: QWModelGeometry, seed: Int) -> (k: [UInt16], v: [UInt16]) {
        let w = geometry.kvHeads * geometry.headDim
        var k = [UInt16](repeating: 0, count: rows * w)
        var v = [UInt16](repeating: 0, count: rows * w)
        for r in 0..<rows {
            for h in 0..<geometry.kvHeads {
                for d in 0..<geometry.headDim {
                    let e = h * geometry.headDim + d
                    let kv = d == 0 ? 1.5 + Float(r) * 0.02 : det(seed &+ r &* 131 &+ e &* 7) * 0.25
                    k[r * w + e] = f16Bits(kv)
                    v[r * w + e] = f16Bits(det(seed &+ r &* 197 &+ e &* 11) * 0.4)
                }
            }
        }
        return (k, v)
    }

    private static func synthQGate(rows: Int, geometry: QWModelGeometry, seed: Int) -> [UInt16] {
        let stride = (geometry.attnOutputGate ? 2 : 1) * geometry.headDim
        var out = [UInt16](repeating: 0, count: rows * geometry.qProjOut)
        for r in 0..<rows {
            for h in 0..<geometry.qHeads {
                let base = r * geometry.qProjOut + h * stride
                for d in 0..<geometry.headDim {
                    out[base + d] = f16Bits(d == 0 ? 0.8 : det(seed &+ r &* 313 &+ h &* 37 &+ d &* 13) * 0.25)
                    if geometry.attnOutputGate {
                        out[base + geometry.headDim + d] = f16Bits(det(seed &+ r &* 577 &+ h &* 19 &+ d) * 0.4)
                    }
                }
            }
        }
        return out
    }

    private static func makeEnv(
        hal: QWHal,
        geometry: QWModelGeometry,
        mode: QWKVMode,
        tokensCeiling: Int,
        batchRows: Int,
        maxSplits: Int = QWAttnDecode.maxSplits
    ) throws -> (cache: QWKVCache, arena: QWArena, partials: QWMRegion) {
        let pageBytes = QWKVCache.pageBytes(mode: mode, geometry: geometry)
        let pages = (tokensCeiling + 255) / 256
        let tables = QWKVCache.tableSlotBytes * 2 * QWKVCache.maxSeqs
        let pBytes = QWAttnDecode.decodePartialsBytesBatched(T: max(1, batchRows), geometry: geometry, maxSplits: maxSplits)
        let arena = try QWArena(hal: hal, weights: 0, activations: 0,
                                kv: pageBytes * pages * 2 + tables, gdnStates: pBytes + 4096)
        let cache = try QWKVCache(hal: hal, arena: arena, layers: [layer], mode: mode, geometry: geometry)
        let g = arena.gdnRegion()
        let off = try g.allocate(bytes: pBytes)
        return (cache, arena, QWMRegion(buffer: g.region.buffer, base: g.region.base + off, length: pBytes))
    }

    private static func writePagedKV(
        hal: QWHal,
        views: (k: QWKVLayerView, v: QWKVLayerView),
        k: [UInt16],
        v: [UInt16],
        startPos: Int,
        count: Int,
        geometry: QWModelGeometry
    ) throws {
        let pso = try hal.pipeline("qwg_kv_paged_write_general")
        let kBuf = try makeBuf(hal, k)
        let vBuf = try makeBuf(hal, v)
        guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
        enc.setComputePipelineState(pso)
        enc.setBuffer(views.k.pool.buffer, offset: views.k.pool.base, index: 0)
        enc.setBuffer(views.v.pool.buffer, offset: views.v.pool.base, index: 1)
        views.k.table.bind(enc, index: 2)
        views.v.table.bind(enc, index: 3)
        enc.setBuffer(kBuf, offset: 0, index: 4)
        enc.setBuffer(vBuf, offset: 0, index: 5)
        var modeU: UInt32 = views.k.mode == .f16 ? 0 : (views.k.mode == .q8 ? 1 : 2)
        enc.setUInt32s(startPos, count, startIndex: 6)
        enc.setBytes(&modeU, length: 4, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: geometry.kvHeads, height: count, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)
    }

    private static func quantizeDeq(
        _ data: [UInt16],
        rows: Int,
        geometry: QWModelGeometry,
        mode: QWKVMode
    ) -> [Float] {
        let hd = geometry.headDim
        let width = geometry.kvHeads * hd
        if mode == .f16 { return data.map { f16($0) } }
        var out = [Float](repeating: 0, count: rows * width)
        let group = (mode == .q8) ? hd : min(128, hd)
        for r in 0..<rows {
            for kvh in 0..<geometry.kvHeads {
                let hBase = r * width + kvh * hd
                for start in stride(from: 0, to: hd, by: group) {
                    let end = min(start + group, hd)
                    let vals = (start..<end).map { f16(data[hBase + $0]) }
                    let lo = vals.min() ?? 0
                    let hi = vals.max() ?? 0
                    let minVal: Float = (mode == .q8) ? 0 : Float(Float16(lo))
                    let absMax = vals.map(abs).max() ?? 0
                    let rawScale = (mode == .q8) ? (absMax / 127.0) : ((hi - lo) / 15.0)
                    let scale = Float(Float16(max(rawScale, 6.103515625e-5)))
                    let inv = scale > 0 ? 1.0 / scale : 0.0
                    for i in start..<end {
                        let q = ((f16(data[hBase + i]) - minVal) * inv).rounded()
                        let c = (mode == .q8) ? max(-127, min(127, q)) : max(0, min(15, q))
                        out[hBase + i] = Float(Float16(c * scale + minVal))
                    }
                }
            }
        }
        return out
    }

    private static func decodeRefRow(
        qBits: [UInt16],
        kDeq: [Float],
        vDeq: [Float],
        geometry: QWModelGeometry,
        head: Int,
        kvStart: Int,
        kvEnd: Int
    ) -> [Float] {
        let hd = geometry.headDim
        let kvh = head / geometry.gqaRatio
        let kvRow = geometry.kvHeads * hd
        let qStride = (geometry.attnOutputGate ? 2 : 1) * hd
        let q = (0..<hd).map { f16(qBits[head * qStride + $0]) }
        let gate: [Float]? = geometry.attnOutputGate
            ? (0..<hd).map { 1.0 / (1.0 + exp(-f16(qBits[head * qStride + hd + $0]))) }
            : nil
        let scale = Float(1.0 / Double(hd).squareRoot())
        var scores = [Float](repeating: 0, count: kvEnd - kvStart)
        for j in kvStart..<kvEnd {
            var dot: Float = 0
            for d in 0..<hd { dot += q[d] * kDeq[j * kvRow + kvh * hd + d] }
            scores[j - kvStart] = dot * scale
        }
        let m = scores.max() ?? 0
        var l: Float = 0
        var o = [Float](repeating: 0, count: hd)
        for j in kvStart..<kvEnd {
            let p = exp(scores[j - kvStart] - m)
            l += p
            for d in 0..<hd { o[d] += p * vDeq[j * kvRow + kvh * hd + d] }
        }
        return QWAttnPagedPrefillGeneralTests.rotated(o, l: l, headDim: hd, gate: gate)
    }

    func testGeneralVsSpecializedDecodeParityAcrossKVModes() throws {
        let g = QWModelGeometry.qwen38_27b
        XCTAssertTrue(g.matchesHead256GQA6Kernel)
        let hal = try QWHal(geometry: g)
        let C = 48
        let kv = Self.synthKV(rows: C + 1, geometry: g, seed: 0x101)
        let qBits = Self.synthQGate(rows: 1, geometry: g, seed: 0x202)
        let rowElems = g.kvHeads * g.headDim
        let outElems = g.qHeads * g.headDim
        let kPre = Array(kv.k.prefix(C * rowElems)), vPre = Array(kv.v.prefix(C * rowElems))
        let kNew = Array(kv.k.suffix(rowElems)), vNew = Array(kv.v.suffix(rowElems))

        for mode in [QWKVMode.f16, .q8, .q4] {
            for splits in [1, 8] {
                var results: [[Float]] = []
                for forceGeneral in [false, true] {
                    let (cache, _, partials) = try Self.makeEnv(hal: hal, geometry: g, mode: mode, tokensCeiling: C + 4, batchRows: 1)
                    let h = try cache.allocate(promptTokens: C + 4)
                    try cache.append(h, positions: 0..<C, layer: Self.layer, kind: .k, src: kPre)
                    try cache.append(h, positions: 0..<C, layer: Self.layer, kind: .v, src: vPre)
                    let views = try cache.layerViews(h, layer: Self.layer)
                    let outBuf = try XCTUnwrap(hal.device.makeBuffer(length: outElems * 2, options: .storageModeShared))
                    let dec = try QWAttnDecode(hal: hal, kvMode: mode, geometry: g, forceGeneral: forceGeneral)
                    guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
                    try dec.encode(enc,
                                   qgate: Self.tv(try Self.makeBuf(hal, qBits), g.qProjOut),
                                   kv: views,
                                   knew: Self.tv(try Self.makeBuf(hal, kNew), rowElems),
                                   vnew: Self.tv(try Self.makeBuf(hal, vNew), rowElems),
                                   partials: partials,
                                   out: Self.tv(outBuf, outElems),
                                   C: C, splits: splits)
                    hal.end(cb, enc)
                    cb.waitUntilCompleted()
                    XCTAssertNil(cb.error)
                    let ptr = outBuf.contents().assumingMemoryBound(to: UInt16.self)
                    results.append((0..<outElems).map { Self.f16(ptr[$0]) })
                    cache.finish(h)
                }
                let maxDiff = zip(results[0], results[1]).map { abs($0 - $1) }.max() ?? 0
                let tol: Float = mode == .f16 ? 2e-3 : (mode == .q8 ? 2e-2 : 8e-2)
                XCTAssertLessThan(maxDiff, tol, "mode=\(mode) splits=\(splits) specialized vs general maxDiff=\(maxDiff)")
            }
        }
    }

    private static func assertRowMatch(
        _ batchPtr: UnsafePointer<UInt16>,
        _ seqPtr: UnsafePointer<UInt16>,
        row: Int,
        outElems: Int,
        mode: QWKVMode,
        sw: Int,
        context: String,
        cpuRef: (() -> [Float])? = nil
    ) {
        var worstNormalULP = 0
        var worstAbs: Float = 0
        var multiULPIndices: [Int] = []
        for i in 0..<outElems {
            let aBits = batchPtr[row * outElems + i]
            let bBits = seqPtr[i]
            let a = f16(aBits)
            let b = f16(bBits)
            XCTAssertTrue(a.isFinite && b.isFinite, "non-finite output (\(context))")
            if aBits != bBits {
                worstAbs = max(worstAbs, abs(a - b))
                if max(abs(a), abs(b)) >= 1e-3 {
                    let ulp = f16ULP(aBits, bBits)
                    worstNormalULP = max(worstNormalULP, ulp)
                    if ulp > 1 { multiULPIndices.append(i) }
                }
            }
        }
        let tol: Float = (mode == .f16 && sw == 0) ? 2e-5 : 1e-4
        XCTAssertLessThan(worstAbs, tol, "\(context) worstNormalULP=\(worstNormalULP)")
        if mode == .f16 && sw == 0 && !multiULPIndices.isEmpty {
            if let ref = cpuRef?() {
                for i in multiULPIndices {
                    let a = f16(batchPtr[row * outElems + i])
                    let b = f16(seqPtr[i])
                    XCTAssertLessThan(abs(a - ref[i]), tol, "\(context) idx=\(i) batch=\(a) ref=\(ref[i])")
                    XCTAssertLessThan(abs(b - ref[i]), tol, "\(context) idx=\(i) seq=\(b) ref=\(ref[i])")
                }
            } else {
                XCTAssertLessThanOrEqual(worstNormalULP, 1, "\(context) worstAbs=\(worstAbs)")
            }
        }
    }

    func testGeneralSingleTokenDecodeParityAcrossGQAAndHeadDims() throws {
        let shapes: [(q: Int, kv: Int, hd: Int, gated: Bool, sw: Int?)] = [
            (2, 2, 64, false, nil),
            (12, 1, 64, true, nil),
            (16, 1, 64, false, 64),
            (16, 4, 128, true, nil),
            (24, 2, 128, false, 64),
            (32, 2, 128, true, nil),
            (16, 2, 256, false, nil),
            (24, 2, 256, true, 64),
            (16, 1, 256, false, nil),
        ]
        let C = 96, splits = 4
        for shape in shapes {
            let g = try Self.makeGeometry(
                qHeads: shape.q, kvHeads: shape.kv, headDim: shape.hd,
                gated: shape.gated, slidingWindow: shape.sw)
            let hal = try QWHal(geometry: g)
            let rowElems = g.kvHeads * g.headDim
            let outElems = g.qHeads * g.headDim
            let kv = Self.synthKV(rows: C + 1, geometry: g, seed: shape.hd &+ shape.q &* 19)
            let qBits = Self.synthQGate(rows: 1, geometry: g, seed: shape.q &+ shape.hd &* 23)
            let kPre = Array(kv.k.prefix(C * rowElems)), vPre = Array(kv.v.prefix(C * rowElems))
            let kNew = Array(kv.k.suffix(rowElems)), vNew = Array(kv.v.suffix(rowElems))
            let sw = shape.sw ?? 0
            let kvStart = sw > 0 ? max(0, C + 1 - sw) : 0

            for mode in [QWKVMode.f16, .q8, .q4] {
                let (cache, _, partials) = try Self.makeEnv(hal: hal, geometry: g, mode: mode, tokensCeiling: C + 4, batchRows: 1)
                let h = try cache.allocate(promptTokens: C + 4)
                try cache.append(h, positions: 0..<C, layer: Self.layer, kind: .k, src: kPre)
                try cache.append(h, positions: 0..<C, layer: Self.layer, kind: .v, src: vPre)
                let views = try cache.layerViews(h, layer: Self.layer)
                let outBuf = try XCTUnwrap(hal.device.makeBuffer(length: outElems * 2, options: .storageModeShared))
                let dec = try QWAttnDecode(hal: hal, kvMode: mode, geometry: g)
                guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
                try dec.encode(enc, qgate: Self.tv(try Self.makeBuf(hal, qBits), g.qProjOut),
                               kv: views, knew: Self.tv(try Self.makeBuf(hal, kNew), rowElems),
                               vnew: Self.tv(try Self.makeBuf(hal, vNew), rowElems),
                               partials: partials, out: Self.tv(outBuf, outElems),
                               C: C, splits: splits, slidingWindow: sw)
                hal.end(cb, enc); cb.waitUntilCompleted()
                XCTAssertNil(cb.error)

                let kAll = Self.quantizeDeq(kPre, rows: C, geometry: g, mode: mode) + kNew.map { Self.f16($0) }
                let vAll = Self.quantizeDeq(vPre, rows: C, geometry: g, mode: mode) + vNew.map { Self.f16($0) }
                let ptr = outBuf.contents().assumingMemoryBound(to: UInt16.self)
                for head in 0..<g.qHeads {
                    let ref = Self.decodeRefRow(qBits: qBits, kDeq: kAll, vDeq: vAll, geometry: g,
                                                head: head, kvStart: kvStart, kvEnd: C + 1)
                    for d in 0..<g.headDim {
                        let got = Self.f16(ptr[head * g.headDim + d])
                        XCTAssertEqual(got, ref[d], accuracy: mode == .f16 ? 3e-3 : 5e-3,
                                       "decode hd=\(shape.hd) gqa=\(g.gqaRatio) mode=\(mode) sw=\(sw) h=\(head) d=\(d)")
                    }
                }
                cache.finish(h)
            }
        }
    }

    func testGeneralDecodeAndSharedPrefixMultiRowParityAcrossGeometries() throws {
        var configs: [(hd: Int, gqa: Int, kv: Int, gated: Bool, sw: Int?)] = []
        for hd in [64, 128, 256] {
            for (idx, gqa) in [1, 4, 6, 8, 12, 16].enumerated() {
                let kv = (gqa >= 12) ? 1 : 2
                configs.append((hd, gqa, kv, (idx + hd / 64) % 2 == 0, nil))
                configs.append((hd, gqa, kv, (idx + hd / 64) % 2 == 1, 64))
            }
        }
        for cfg in configs {
            let qHeads = cfg.gqa * cfg.kv
            let g = try Self.makeGeometry(
                qHeads: qHeads, kvHeads: cfg.kv, headDim: cfg.hd,
                gated: cfg.gated, slidingWindow: cfg.sw)
            let hal = try QWHal(geometry: g)
            let rowElems = g.kvHeads * g.headDim
            let outElems = g.qHeads * g.headDim
            let sw = cfg.sw ?? 0
            let cBase = sw > 0 ? 96 : 24
            let splits = sw > 0 ? 4 : 8
            let maxB = 8
            let qBuf = try XCTUnwrap(hal.device.makeBuffer(length: maxB * g.qProjOut * 2, options: .storageModeShared))
            let kBuf = try XCTUnwrap(hal.device.makeBuffer(length: maxB * rowElems * 2, options: .storageModeShared))
            let vBuf = try XCTUnwrap(hal.device.makeBuffer(length: maxB * rowElems * 2, options: .storageModeShared))
            let outBatch = try XCTUnwrap(hal.device.makeBuffer(length: maxB * outElems * 2, options: .storageModeShared))
            let outSeq = try XCTUnwrap(hal.device.makeBuffer(length: outElems * 2, options: .storageModeShared))

            for mode in [QWKVMode.f16, .q8, .q4] {
                let decB = try QWAttnDecode(hal: hal, sharedPrefixRows: maxB, kvMode: mode, geometry: g)
                let decS = try QWAttnDecode(hal: hal, kvMode: mode, geometry: g)
                let (cacheB, _, partB) = try Self.makeEnv(hal: hal, geometry: g, mode: mode, tokensCeiling: cBase + maxB + 8, batchRows: maxB)
                let hB = try cacheB.allocate(promptTokens: cBase + maxB + 8)
                let (cacheS, _, partS) = try Self.makeEnv(hal: hal, geometry: g, mode: mode, tokensCeiling: cBase + maxB + 8, batchRows: 1)
                let hS = try cacheS.allocate(promptTokens: cBase + maxB + 8)
                defer { cacheB.finish(hB); cacheS.finish(hS) }

                let kv = Self.synthKV(rows: cBase + maxB, geometry: g, seed: cfg.hd &+ cBase &+ sw &+ 17)
                let qBits = Self.synthQGate(rows: maxB, geometry: g, seed: qHeads &+ cBase &+ sw &+ 31)
                let kPre = Array(kv.k.prefix(cBase * rowElems)), vPre = Array(kv.v.prefix(cBase * rowElems))
                let kNewAll = Array(kv.k.suffix(maxB * rowElems)), vNewAll = Array(kv.v.suffix(maxB * rowElems))
                try cacheB.append(hB, positions: 0..<cBase, layer: Self.layer, kind: .k, src: kPre)
                try cacheB.append(hB, positions: 0..<cBase, layer: Self.layer, kind: .v, src: vPre)
                let viewsB = try cacheB.layerViews(hB, layer: Self.layer)
                try cacheS.append(hS, positions: 0..<cBase, layer: Self.layer, kind: .k, src: kPre)
                try cacheS.append(hS, positions: 0..<cBase, layer: Self.layer, kind: .v, src: vPre)
                let viewsS = try cacheS.layerViews(hS, layer: Self.layer)

                qBits.withUnsafeBytes { if let b = $0.baseAddress { _ = memcpy(qBuf.contents(), b, $0.count) } }
                kNewAll.withUnsafeBytes { if let b = $0.baseAddress { _ = memcpy(kBuf.contents(), b, $0.count) } }
                vNewAll.withUnsafeBytes { if let b = $0.baseAddress { _ = memcpy(vBuf.contents(), b, $0.count) } }

                var seqRows = [[UInt16]](repeating: [], count: maxB)
                for r in 0..<maxB {
                    guard let (cbS, encS) = hal.begin() else { throw XCTSkip("no command buffer") }
                    try decS.encode(encS,
                                    qgate: QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length), offset: r * g.qProjOut * 2, shape: QWTensorShape([g.qProjOut], .f16)),
                                    kv: viewsS,
                                    knew: QWTensorValue(region: QWMRegion(buffer: kBuf, base: 0, length: kBuf.length), offset: r * rowElems * 2, shape: QWTensorShape([rowElems], .f16)),
                                    vnew: QWTensorValue(region: QWMRegion(buffer: vBuf, base: 0, length: vBuf.length), offset: r * rowElems * 2, shape: QWTensorShape([rowElems], .f16)),
                                    partials: partS,
                                    out: Self.tv(outSeq, outElems),
                                    C: cBase + r, splits: splits, slidingWindow: sw)
                    hal.end(cbS, encS); cbS.waitUntilCompleted()
                    XCTAssertNil(cbS.error)
                    let seqPtr = outSeq.contents().assumingMemoryBound(to: UInt16.self)
                    seqRows[r] = Array(UnsafeBufferPointer(start: seqPtr, count: outElems))
                }

                for B in 2...maxB {
                    let kNew = Array(kNewAll.prefix(B * rowElems))
                    let vNew = Array(vNewAll.prefix(B * rowElems))
                    try Self.writePagedKV(hal: hal, views: viewsB, k: kNew, v: vNew, startPos: cBase, count: B, geometry: g)

                    guard let (cbB, encB) = hal.begin() else { throw XCTSkip("no command buffer") }
                    try decB.encodeBatched(
                        encB,
                        qgate: QWTensorValue(region: QWMRegion(buffer: qBuf, base: 0, length: qBuf.length), offset: 0, shape: QWTensorShape([B * g.qProjOut], .f16)),
                        kv: viewsB,
                        knew: QWTensorValue(region: QWMRegion(buffer: kBuf, base: 0, length: kBuf.length), offset: 0, shape: QWTensorShape([B * rowElems], .f16)),
                        vnew: QWTensorValue(region: QWMRegion(buffer: vBuf, base: 0, length: vBuf.length), offset: 0, shape: QWTensorShape([B * rowElems], .f16)),
                        partials: partB,
                        out: QWTensorValue(region: QWMRegion(buffer: outBatch, base: 0, length: outBatch.length), offset: 0, shape: QWTensorShape([B * outElems], .f16)),
                        cBase: cBase, T: B, splits: splits, sharedPrefix: true, slidingWindow: sw)
                    hal.end(cbB, encB); cbB.waitUntilCompleted()
                    XCTAssertNil(cbB.error)

                    let batchPtr = outBatch.contents().assumingMemoryBound(to: UInt16.self)
                    for r in 0..<B {
                        let refClosure: () -> [Float] = {
                            let kDeq = Self.quantizeDeq(kPre, rows: cBase, geometry: g, mode: mode)
                                + Self.quantizeDeq(Array(kNew.prefix(r * rowElems)), rows: r, geometry: g, mode: mode)
                                + Array(kNew[(r * rowElems)..<((r + 1) * rowElems)]).map { Self.f16($0) }
                            let vDeq = Self.quantizeDeq(vPre, rows: cBase, geometry: g, mode: mode)
                                + Self.quantizeDeq(Array(vNew.prefix(r * rowElems)), rows: r, geometry: g, mode: mode)
                                + Array(vNew[(r * rowElems)..<((r + 1) * rowElems)]).map { Self.f16($0) }
                            let qRow = Array(qBits[(r * g.qProjOut)..<((r + 1) * g.qProjOut)])
                            return (0..<g.qHeads).flatMap {
                                Self.decodeRefRow(qBits: qRow, kDeq: kDeq, vDeq: vDeq, geometry: g,
                                                  head: $0, kvStart: 0, kvEnd: cBase + r + 1)
                            }
                        }
                        seqRows[r].withUnsafeBufferPointer { sBuf in
                            Self.assertRowMatch(
                                batchPtr, sBuf.baseAddress!,
                                row: r, outElems: outElems, mode: mode, sw: sw,
                                context: "cBase=\(cBase) sw=\(sw) hd=\(cfg.hd) gqa=\(cfg.gqa) gated=\(cfg.gated) mode=\(mode) B=\(B) row=\(r)",
                                cpuRef: refClosure)
                        }
                    }
                    if sw > 0 && B == maxB {
                        let kAll = Self.quantizeDeq(kPre, rows: cBase, geometry: g, mode: mode)
                            + kNew.prefix(rowElems).map { Self.f16($0) }
                        let vAll = Self.quantizeDeq(vPre, rows: cBase, geometry: g, mode: mode)
                            + vNew.prefix(rowElems).map { Self.f16($0) }
                        let qRow = Array(qBits[0..<g.qProjOut])
                        for h in 0..<g.qHeads {
                            let want = Self.decodeRefRow(qBits: qRow, kDeq: kAll, vDeq: vAll, geometry: g,
                                                         head: h, kvStart: cBase + 1 - sw, kvEnd: cBase + 1)
                            for d in 0..<g.headDim {
                                XCTAssertEqual(Self.f16(batchPtr[h * g.headDim + d]), want[d],
                                               accuracy: mode == .f16 ? 3e-3 : 5e-3,
                                               "SWA ref hd=\(cfg.hd) gqa=\(cfg.gqa) mode=\(mode) h=\(h) d=\(d)")
                            }
                        }
                    }
                }
            }
        }
    }

    func testGeneralBatchedBoundariesAndGuards() throws {
        // Verify PSO threadgroup capacity & compact TG staging across B=2...8 at gqa=16.
        for hd in [128, 256] {
            let g16 = try Self.makeGeometry(qHeads: 16, kvHeads: 1, headDim: hd, gated: true)
            let hal16 = try QWHal(geometry: g16)
            for mode in [QWKVMode.f16, .q8, .q4] {
                for B in 2...8 {
                    let name = QWAttnDecode.generalSharedKernelName(mode, B, headDim: hd)
                    let pso = try hal16.pipeline(name)
                    XCTAssertGreaterThanOrEqual(pso.maxTotalThreadsPerThreadgroup, 512, "\(name) maxTPTG=\(pso.maxTotalThreadsPerThreadgroup)")
                    let expectedTG = 16 * (hd <= 128 ? 128 : 256) * 2 * 2
                    XCTAssertEqual(pso.staticThreadgroupMemoryLength, expectedTG, "\(name) staticTG=\(pso.staticThreadgroupMemoryLength)")
                }
            }
        }

        let gHighGQA = try Self.makeGeometry(qHeads: 34, kvHeads: 2, headDim: 64, gated: false)
        let halHigh = try QWHal(geometry: gHighGQA)
        XCTAssertThrowsError(try QWAttnDecode(hal: halHigh, geometry: gHighGQA)) { err in
            guard case QWAttnDecodeError.badShape = err else { return XCTFail("expected badShape, got \(err)") }
        }
        let gGQA16 = try Self.makeGeometry(qHeads: 32, kvHeads: 2, headDim: 64, gated: false)
        XCTAssertNoThrow(try QWAttnDecode(hal: try QWHal(geometry: gGQA16), sharedPrefixRows: 8, geometry: gGQA16))

        let gSW = try Self.makeGeometry(qHeads: 8, kvHeads: 2, headDim: 128, gated: true, slidingWindow: 64)
        let halSW = try QWHal(geometry: gSW)
        let cBase = 340, B = 6, splits = 8
        let rowElems = gSW.kvHeads * gSW.headDim, outElems = gSW.qHeads * gSW.headDim
        let kv = Self.synthKV(rows: cBase + B, geometry: gSW, seed: 0x555)
        let qBits = Self.synthQGate(rows: B, geometry: gSW, seed: 0x666)
        let kPre = Array(kv.k.prefix(cBase * rowElems)), vPre = Array(kv.v.prefix(cBase * rowElems))
        let kNew = Array(kv.k.suffix(B * rowElems)), vNew = Array(kv.v.suffix(B * rowElems))

        let (cacheB, _, partB) = try Self.makeEnv(hal: halSW, geometry: gSW, mode: .f16, tokensCeiling: cBase + B + 8, batchRows: B)
        let hB = try cacheB.allocate(promptTokens: cBase + B + 8)
        try cacheB.append(hB, positions: 0..<cBase, layer: Self.layer, kind: .k, src: kPre)
        try cacheB.append(hB, positions: 0..<cBase, layer: Self.layer, kind: .v, src: vPre)
        try cacheB.recycleExpiredWindowPages(hB, currentPosition: cBase, safetyMargin: 8)
        XCTAssertEqual(cacheB.allocatedPhysicalPages(hB, layer: Self.layer), 1, "page 0 should be recycled to -1")
        let viewsB = try cacheB.layerViews(hB, layer: Self.layer)
        try Self.writePagedKV(hal: halSW, views: viewsB, k: kNew, v: vNew, startPos: cBase, count: B, geometry: gSW)
        let dec = try QWAttnDecode(hal: halSW, sharedPrefixRows: B, kvMode: .f16, geometry: gSW)
        let outBuf = try XCTUnwrap(halSW.device.makeBuffer(length: B * outElems * 2, options: .storageModeShared))
        guard let (cb, enc) = halSW.begin() else { throw XCTSkip("no command buffer") }
        try dec.encodeBatched(
            enc,
            qgate: Self.tv(try Self.makeBuf(halSW, qBits), B * gSW.qProjOut),
            kv: viewsB,
            knew: Self.tv(try Self.makeBuf(halSW, kNew), B * rowElems),
            vnew: Self.tv(try Self.makeBuf(halSW, vNew), B * rowElems),
            partials: partB,
            out: Self.tv(outBuf, B * outElems),
            cBase: cBase, T: B, splits: splits, sharedPrefix: true, slidingWindow: 64)
        halSW.end(cb, enc); cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let kAll = kv.k.map { Self.f16($0) }, vAll = kv.v.map { Self.f16($0) }
        let outPtr = outBuf.contents().assumingMemoryBound(to: UInt16.self)
        for r in 0..<B {
            let kvEnd = cBase + r + 1
            let kvStart = max(0, kvEnd - 64)
            let qRow = Array(qBits[(r * gSW.qProjOut)..<((r + 1) * gSW.qProjOut)])
            for head in 0..<gSW.qHeads {
                let want = Self.decodeRefRow(qBits: qRow, kDeq: kAll, vDeq: vAll, geometry: gSW,
                                             head: head, kvStart: kvStart, kvEnd: kvEnd)
                for d in 0..<gSW.headDim {
                    let got = Self.f16(outPtr[r * outElems + head * gSW.headDim + d])
                    XCTAssertEqual(got, want[d], accuracy: 3e-3, "SWA sharedPrefix recycled r=\(r) h=\(head) d=\(d)")
                }
            }
        }
        cacheB.finish(hB)

        // Stride-mismatch sequential fallback parity (cBase=23, B=4, splits=8 => active0/8=3 != activeLast/8=4).
        let gFull = try Self.makeGeometry(qHeads: 8, kvHeads: 2, headDim: 128, gated: false)
        let halFull = try QWHal(geometry: gFull)
        let cMis = 23, bMis = 4, sMis = 8
        let kvM = Self.synthKV(rows: cMis + bMis, geometry: gFull, seed: 0x777)
        let qM = Self.synthQGate(rows: bMis, geometry: gFull, seed: 0x888)
        let kPreM = Array(kvM.k.prefix(cMis * rowElems)), vPreM = Array(kvM.v.prefix(cMis * rowElems))
        let kNewM = Array(kvM.k.suffix(bMis * rowElems)), vNewM = Array(kvM.v.suffix(bMis * rowElems))
        let (cacheM, _, partM) = try Self.makeEnv(hal: halFull, geometry: gFull, mode: .f16, tokensCeiling: cMis + bMis + 8, batchRows: bMis)
        let hM = try cacheM.allocate(promptTokens: cMis + bMis + 8)
        try cacheM.append(hM, positions: 0..<cMis, layer: Self.layer, kind: .k, src: kPreM)
        try cacheM.append(hM, positions: 0..<cMis, layer: Self.layer, kind: .v, src: vPreM)
        let viewsM = try cacheM.layerViews(hM, layer: Self.layer)
        let decM = try QWAttnDecode(hal: halFull, sharedPrefixRows: bMis, kvMode: .f16, geometry: gFull)
        let outM = try XCTUnwrap(halFull.device.makeBuffer(length: bMis * outElems * 2, options: .storageModeShared))
        guard let (cbM, encM) = halFull.begin() else { throw XCTSkip("no command buffer") }
        try decM.encodeBatched(
            encM, qgate: Self.tv(try Self.makeBuf(halFull, qM), bMis * gFull.qProjOut), kv: viewsM,
            knew: Self.tv(try Self.makeBuf(halFull, kNewM), bMis * rowElems),
            vnew: Self.tv(try Self.makeBuf(halFull, vNewM), bMis * rowElems),
            partials: partM, out: Self.tv(outM, bMis * outElems),
            cBase: cMis, T: bMis, splits: sMis, sharedPrefix: true)
        halFull.end(cbM, encM); cbM.waitUntilCompleted()
        XCTAssertNil(cbM.error)
        let kAllM = kvM.k.map { Self.f16($0) }, vAllM = kvM.v.map { Self.f16($0) }
        let mPtr = outM.contents().assumingMemoryBound(to: UInt16.self)
        for r in 0..<bMis {
            let qRow = Array(qM[(r * gFull.qProjOut)..<((r + 1) * gFull.qProjOut)])
            for head in 0..<gFull.qHeads {
                let want = Self.decodeRefRow(qBits: qRow, kDeq: kAllM, vDeq: vAllM, geometry: gFull,
                                             head: head, kvStart: 0, kvEnd: cMis + r + 1)
                for d in 0..<gFull.headDim {
                    XCTAssertEqual(Self.f16(mPtr[r * outElems + head * gFull.headDim + d]), want[d], accuracy: 3e-3)
                }
            }
        }
        cacheM.finish(hM)
    }

    func testGeneralVsNAXPagedPrefillParityAcrossGeometriesAndKVModes() throws {
        let probeHal = try QWHal(geometry: .qwen38_27b)
        try XCTSkipUnless(probeHal.caps.isMetal4TensorCapable, "Paged NAX prefill requires Apple10+ NAX")

        let shapes: [(q: Int, kv: Int, hd: Int, gated: Bool)] = [
            (4, 4, 64, false),
            (8, 2, 64, true),
            (8, 2, 128, false),
            (16, 2, 128, true),
            (12, 2, 256, false),
            (16, 2, 256, true),
        ]
        let pos = 32, T = 20, sp = 32
        for shape in shapes {
            let g = try Self.makeGeometry(qHeads: shape.q, kvHeads: shape.kv, headDim: shape.hd, gated: shape.gated)
            let hal = try QWHal(geometry: g)
            let kv = Self.synthKV(rows: pos + T, geometry: g, seed: shape.hd &+ shape.q)
            let qBits = Self.synthQGate(rows: sp, geometry: g, seed: shape.hd &+ 99)
            let outElems = T * g.qHeads * g.headDim

            for mode in [QWKVMode.f16, .q8, .q4] {
                let (cache, _, _) = try Self.makeEnv(hal: hal, geometry: g, mode: mode, tokensCeiling: pos + T + 8, batchRows: 1)
                let h = try cache.allocate(promptTokens: pos + T + 8)
                try cache.append(h, positions: 0..<(pos + T), layer: Self.layer, kind: .k, src: kv.k)
                try cache.append(h, positions: 0..<(pos + T), layer: Self.layer, kind: .v, src: kv.v)
                let views = try cache.layerViews(h, layer: Self.layer)

                let qBuf = try Self.makeBuf(hal, qBits)
                let outGen = try XCTUnwrap(hal.device.makeBuffer(length: outElems * 2, options: .storageModeShared))
                let outNAX = try XCTUnwrap(hal.device.makeBuffer(length: outElems * 2, options: .storageModeShared))
                let genOp = try QWAttnPagedPrefillGeneral(hal: hal, geometry: g)
                let naxOp = try QWAttnPagedPrefill(hal: hal, geometry: g)

                guard let (cb1, enc1) = hal.begin() else { throw XCTSkip("no command buffer") }
                try genOp.encode(enc1, qgate: Self.tv(qBuf, T * g.qProjOut), kv: views,
                                 out: Self.tv(outGen, outElems), pos: pos, T: T)
                hal.end(cb1, enc1); cb1.waitUntilCompleted()
                XCTAssertNil(cb1.error)

                guard let (cb2, enc2) = hal.begin() else { throw XCTSkip("no command buffer") }
                try naxOp.encode(enc2, qgate: Self.tv(qBuf, sp * g.qProjOut), kv: views,
                                 out: Self.tv(outNAX, outElems), pos: pos, T: T)
                hal.end(cb2, enc2); cb2.waitUntilCompleted()
                XCTAssertNil(cb2.error)

                let pGen = outGen.contents().assumingMemoryBound(to: UInt16.self)
                let pNAX = outNAX.contents().assumingMemoryBound(to: UInt16.self)
                let maxDiff = (0..<outElems).map { abs(Self.f16(pGen[$0]) - Self.f16(pNAX[$0])) }.max() ?? 0
                let tol: Float = mode == .f16 ? 5e-3 : 1.5e-2
                XCTAssertLessThan(maxDiff, tol, "prefill hd=\(shape.hd) gqa=\(shape.q / shape.kv) mode=\(mode) maxDiff=\(maxDiff)")
                cache.finish(h)
            }
        }
    }
}
