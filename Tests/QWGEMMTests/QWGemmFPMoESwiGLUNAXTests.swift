// QWGemmFPMoESwiGLUNAXTests.swift — Parity tests for fused NVFP4/MXFP8 NAX SwiGLU prefill and zero-rotation T<=8 decode.
import Foundation
import Metal
import XCTest
@testable import QWGEMM
import QWAlloc
import QWCore
import QWHAL

final class QWGemmFPMoESwiGLUNAXTests: XCTestCase {
    private func makeHal() throws -> QWHal {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "FP MoE NAX SwiGLU tests require bare-metal Metal 4 tensor GPU"
        )
        return try QWHal(geometry: .qwen38_27b)
    }

    private func sharedSlice<T>(_ hal: QWHal, values: [T]) throws -> QWFPBufferSlice {
        let len = max(values.count * MemoryLayout<T>.stride, 64)
        guard let buf = hal.device.makeBuffer(length: len, options: .storageModeShared) else {
            throw XCTSkip("MTLBuffer allocation failed")
        }
        values.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            if let src = raw.baseAddress, raw.count > 0 { buf.contents().copyMemory(from: src, byteCount: raw.count) }
        }
        return QWFPBufferSlice(buffer: buf, offset: 0, bytes: len)
    }

    private func bf16Bits(_ v: Float) -> UInt16 {
        var bits = v.bitPattern
        if (bits & 0x7F80_0000) == 0x7F80_0000 { return UInt16(truncatingIfNeeded: bits >> 16) }
        bits = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: bits >> 16)
    }

    private func f32FromBF16(_ u: UInt16) -> Float { Float(bitPattern: UInt32(u) << 16) }

    private func roundBF16(_ v: Double) -> Double { Double(f32FromBF16(bf16Bits(Float(v)))) }

    private func roundInter(_ v: Double, _ dt: QWFPElemDtype) -> Double {
        dt == .bf16 ? roundBF16(v) : Double(Float16(v))
    }

    private func fp8E4M3(_ b: UInt8) -> Double {
        let s: Double = (b & 0x80) != 0 ? -1.0 : 1.0
        let e = Int((b >> 3) & 0xF), m = Int(b & 0x7)
        if e == 15 && m == 7 { return .nan }
        if e == 0 { return s * pow(2.0, -6.0) * (Double(m) / 8.0) }
        return s * pow(2.0, Double(e - 7)) * (1.0 + Double(m) / 8.0)
    }

    private func ue8m0(_ b: UInt8) -> Double {
        b == 0 ? pow(2.0, -127.0) : pow(2.0, Double(Int(b) - 127))
    }

    private let fp4Vals: [Double] = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6]

    private func packNVFP4(experts: Int, rows: Int, k: Int, seed: UInt64)
        -> (w: [UInt32], sc: [UInt8], s2: [Float], ref: [Double]) {
        var rng = seed
        func nextByte() -> UInt8 { rng = rng &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407; return UInt8((rng >> 56) & 0xFF) }
        let scPool: [UInt8] = [0x30, 0x34, 0x38, 0x3C, 0x40, 0xB4, 0xBC, 0x28]
        let groups = k / 16
        var w = [UInt32](repeating: 0, count: experts * rows * groups * 2)
        var sc = [UInt8](repeating: 0, count: experts * rows * groups)
        var s2 = [Float](repeating: 1, count: experts)
        var ref = [Double](repeating: 0, count: experts * rows * k)
        for e in 0..<experts {
            let es2 = 0.25 + 0.125 * Double((e % 4) + 1)
            s2[e] = Float(es2)
            for r in 0..<rows {
                for g in 0..<groups {
                    let sb = scPool[Int(nextByte()) % scPool.count]
                    sc[(e * rows + r) * groups + g] = sb
                    let sv = fp8E4M3(sb)
                    var w0: UInt32 = 0, w1: UInt32 = 0
                    for j in 0..<16 {
                        let nib = Int(nextByte() & 0xF)
                        if j < 8 { w0 |= UInt32(nib) << (j * 4) } else { w1 |= UInt32(nib) << ((j - 8) * 4) }
                        ref[((e * rows + r) * k) + g * 16 + j] = fp4Vals[nib] * sv * es2
                    }
                    let base = ((e * rows + r) * groups + g) * 2
                    w[base] = w0; w[base + 1] = w1
                }
            }
        }
        return (w, sc, s2, ref)
    }

    private func packMXFP8(experts: Int, rows: Int, k: Int, seed: UInt64)
        -> (w: [UInt32], sc: [UInt8], ref: [Double]) {
        var rng = seed
        func nextByte() -> UInt8 { rng = rng &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407; return UInt8((rng >> 56) & 0xFF) }
        let wPool: [UInt8] = [0x00, 0x04, 0x20, 0x30, 0x38, 0x3C, 0x40, 0x44, 0xB8, 0xBC, 0xC0, 0x84]
        let e8Pool: [UInt8] = [125, 126, 127, 128, 129]
        let groups = k / 32
        var w = [UInt32](repeating: 0, count: experts * rows * groups * 8)
        var sc = [UInt8](repeating: 0, count: experts * rows * groups)
        var ref = [Double](repeating: 0, count: experts * rows * k)
        for e in 0..<experts {
            for r in 0..<rows {
                for g in 0..<groups {
                    let sb = e8Pool[Int(nextByte()) % e8Pool.count]
                    sc[(e * rows + r) * groups + g] = sb
                    let sv = ue8m0(sb)
                    for wi in 0..<8 {
                        var word: UInt32 = 0
                        for b in 0..<4 {
                            let fb = wPool[Int(nextByte()) % wPool.count]
                            word |= UInt32(fb) << (b * 8)
                            ref[((e * rows + r) * k) + g * 32 + wi * 4 + b] = fp8E4M3(fb) * sv
                        }
                        w[((e * rows + r) * groups + g) * 8 + wi] = word
                    }
                }
            }
        }
        return (w, sc, ref)
    }

    func testFusedFPMoENaxSwiGLUAndZeroRotationCompactDecodeParity() throws {
        let hal = try makeHal()
        let experts = 8, topK = 2, hidden = 64
        for inter in [64, 128] {
            let maxT = 10, maxPairs = maxT * topK
            let arena = try QWArena(hal: hal, weights: 1 << 16, activations: 16 << 20, kv: 0, gdnStates: 0)
            let ar = arena.activationsRegion()
            let gP = packNVFP4(experts: experts, rows: inter, k: hidden, seed: UInt64(0x1100 + inter))
            let uP = packNVFP4(experts: experts, rows: inter, k: hidden, seed: UInt64(0x2200 + inter))
            let dP = packNVFP4(experts: experts, rows: hidden, k: inter, seed: UInt64(0x3300 + inter))
            var guW: [UInt32] = [], guSc: [UInt8] = []
            let wExp = inter * (hidden / 8), sExp = inter * (hidden / 16)
            for e in 0..<experts {
                guW.append(contentsOf: gP.w[(e * wExp)..<((e + 1) * wExp)])
                guW.append(contentsOf: uP.w[(e * wExp)..<((e + 1) * wExp)])
                guSc.append(contentsOf: gP.sc[(e * sExp)..<((e + 1) * sExp)])
                guSc.append(contentsOf: uP.sc[(e * sExp)..<((e + 1) * sExp)])
            }
            let nvGU = QWMoEExpertBank.nvfp4(QWGemmFP.NVFP4Bank(
                weights: try sharedSlice(hal, values: guW), scales: try sharedSlice(hal, values: guSc),
                scales2: try sharedSlice(hal, values: gP.s2 + uP.s2.map { $0 * 1.5 }),
                experts: experts, rowsPerExpert: 2 * inter, k: hidden, splitGateUpScales2: true))
            let nvDN = QWMoEExpertBank.nvfp4(QWGemmFP.NVFP4Bank(
                weights: try sharedSlice(hal, values: dP.w), scales: try sharedSlice(hal, values: dP.sc),
                scales2: try sharedSlice(hal, values: dP.s2), experts: experts, rowsPerExpert: hidden, k: inter))

            let mxG = packMXFP8(experts: experts, rows: 2 * inter, k: hidden, seed: UInt64(0x4400 + inter))
            let mxD = packMXFP8(experts: experts, rows: hidden, k: inter, seed: UInt64(0x5500 + inter))
            let mxGU = QWMoEExpertBank.mxfp8(QWGemmFP.MXFP8Bank(
                weights: try sharedSlice(hal, values: mxG.w), scales: try sharedSlice(hal, values: mxG.sc),
                experts: experts, rowsPerExpert: 2 * inter, k: hidden))
            let mxDN = QWMoEExpertBank.mxfp8(QWGemmFP.MXFP8Bank(
                weights: try sharedSlice(hal, values: mxD.w), scales: try sharedSlice(hal, values: mxD.sc),
                experts: experts, rowsPerExpert: hidden, k: inter))

            let xF16 = (0..<(maxT * hidden)).map { Float16(sin(Double($0) * 0.19) * 0.45) }
            let xBF16 = xF16.map { bf16Bits(Float($0)) }
            let idVals = (0..<maxPairs).map { Int32(($0 * 3 + $0 / topK) % experts) }
            let wtVals = (0..<maxPairs).map { Float($0 % 2 == 0 ? 0.6 : 0.4) }
            let xF16TV = try ar.allocate(QWTensorShape([maxT, hidden], .f16)); arena.write(xF16TV, from: xF16.map(\.bitPattern))
            let xBF16TV = try ar.allocate(QWTensorShape([maxT, hidden], .bf16)); arena.write(xBF16TV, from: xBF16)
            let idsTV = try ar.allocate(QWTensorShape([maxPairs], .i32)); arena.write(idsTV, from: idVals)
            let wtsTV = try ar.allocate(QWTensorShape([maxPairs], .f32)); arena.write(wtsTV, from: wtVals)
            let expOffTV = try ar.allocate(QWTensorShape([experts + 1], .i32))
            let sortedPairTV = try ar.allocate(QWTensorShape([maxPairs], .i32))
            let sortedTokTV = try ar.allocate(QWTensorShape([maxPairs], .i32))
            let invPermTV = try ar.allocate(QWTensorShape([maxPairs], .i32))
            let tilePlan = QWGemm.MoETilePlan(experts: experts, pairs: maxPairs, gateUpRows: 2 * inter, downRows: hidden)
            let tileETV = try ar.allocate(QWTensorShape([max(1, tilePlan.maxTiles)], .i32))
            let tileRTV = try ar.allocate(QWTensorShape([max(1, tilePlan.maxTiles)], .i32))
            let tileCountTV = try ar.allocate(QWTensorShape([tilePlan.scratchInts], .i32))
            let actExpTV = try ar.allocate(QWTensorShape([experts], .i32))
            let actArgsTV = try ar.allocate(QWTensorShape([QWGemmFP.MoEActiveBucketPlan.indirectBufferBytes / 4], .i32))
            let guTV = try ar.allocate(QWTensorShape([maxPairs, 2 * inter], .f16))
            let actTV = try ar.allocate(QWTensorShape([maxPairs, inter], .f16))
            let dnAccTV = try ar.allocate(QWTensorShape([maxPairs, hidden], .f16))
            let outTV = try ar.allocate(QWTensorShape([maxT, hidden], .f32))

            for (isMX, gBank, dBank) in [(false, nvGU, nvDN), (true, mxGU, mxDN)] {
                for xDt in [QWFPElemDtype.f16, .bf16] {
                    var t4Prefix: [Float] = []
                    var t4Tol: [Double] = []
                    for curT in [4, maxT] {
                        guard let (cb, initEnc) = hal.begin() else { XCTFail("begin failed"); return }
                        var enc = initEnc
                        var rotations = 0
                        let rotator: (inout any MTLComputeCommandEncoder) throws -> Void = { cur in
                            rotations += 1
                            try hal.rotateEncoder(cb, &cur)
                        }
                        try QWGemm.encodeMoEBankRoutedCombine(
                            &enc, hal: hal, x: QWFPBufferSlice(xDt == .bf16 ? xBF16TV : xF16TV),
                            xDtype: xDt, interDtype: xDt,
                            gateUp: gBank, up: nil, down: dBank, idx: idsTV, wts: wtsTV,
                            shared: nil, sharedDtype: .f16, gate: nil, out: QWFPBufferSlice(outTV), outDtype: .f32,
                            expOff: expOffTV, sortedPair: sortedPairTV, sortedTok: sortedTokTV, invPerm: invPermTV,
                            tileE: curT > 8 ? tileETV : nil, tileR: curT > 8 ? tileRTV : nil,
                            tileCount: curT > 8 ? tileCountTV : nil,
                            activeExperts: actExpTV, activeIndirect: actArgsTV,
                            guBuf: guTV, actBuf: actTV, dnAccBuf: dnAccTV,
                            hidden: hidden, experts: experts, topK: topK, intermediate: inter, rows: curT,
                            gateSoftcap: 4.0, upSoftcap: 25.0, supportsGroupedNaxPrefill: true,
                            concurrentBarriers: true, rotateEncoder: rotator)
                        hal.end(cb, enc)
                        cb.waitUntilCompleted()
                        XCTAssertNil(cb.error)
                        if curT <= 8 {
                            XCTAssertEqual(rotations, 0, "T=\(curT) must run with zero encoder rotations")
                        }

                        let got = arena.read(outTV, as: Float.self, count: curT * hidden)
                        if curT == 4 {
                            t4Prefix = got
                            if !isMX {
                                for tok in 0..<4 {
                                    let t1Ids = try ar.allocate(QWTensorShape([topK], .i32))
                                    arena.write(t1Ids, from: Array(idVals[(tok * topK)..<((tok + 1) * topK)]))
                                    let t1Wts = try ar.allocate(QWTensorShape([topK], .f32))
                                    arena.write(t1Wts, from: Array(wtVals[(tok * topK)..<((tok + 1) * topK)]))
                                    let t1Out = try ar.allocate(QWTensorShape([hidden], .f32))
                                    guard let (cb1, enc0) = hal.begin() else { XCTFail("begin failed"); return }
                                    var enc1 = enc0
                                    let xTokSlice = QWFPBufferSlice(
                                        (xDt == .bf16 ? xBF16TV : xF16TV)
                                            .slice(byteOffset: tok * hidden * 2, count: hidden,
                                                   dtype: xDt == .bf16 ? .bf16 : .f16))
                                    try QWGemm.encodeMoEBankRoutedCombine(
                                        &enc1, hal: hal, x: xTokSlice,
                                        xDtype: xDt, interDtype: xDt,
                                        gateUp: gBank, up: nil, down: dBank, idx: t1Ids, wts: t1Wts,
                                        shared: nil, sharedDtype: .f16, gate: nil,
                                        out: QWFPBufferSlice(t1Out), outDtype: .f32,
                                        expOff: expOffTV, sortedPair: sortedPairTV,
                                        sortedTok: sortedTokTV, invPerm: invPermTV,
                                        tileE: nil, tileR: nil, tileCount: nil,
                                        activeExperts: actExpTV, activeIndirect: actArgsTV,
                                        guBuf: guTV, actBuf: actTV, dnAccBuf: dnAccTV,
                                        hidden: hidden, experts: experts, topK: topK, intermediate: inter, rows: 1,
                                        gateSoftcap: 4.0, upSoftcap: 25.0, supportsGroupedNaxPrefill: true,
                                        concurrentBarriers: true, rotateEncoder: nil)
                                    hal.end(cb1, enc1)
                                    cb1.waitUntilCompleted()
                                    XCTAssertNil(cb1.error)
                                    let got1 = arena.read(t1Out, as: Float.self, count: hidden)
                                    for h in 0..<hidden {
                                        XCTAssertEqual(got[tok * hidden + h], got1[h], accuracy: 1e-5,
                                                       "NVFP4 compact T=4 vs T=1 mismatch I=\(inter) x=\(xDt) [\(tok),\(h)]")
                                    }
                                }
                            }
                        } else {
                            for idx in 0..<t4Prefix.count {
                                let prefixTol = (!isMX && xDt == .bf16)
                                    ? 2.0 * t4Tol[idx]
                                    : max(0.08, abs(Double(t4Prefix[idx])) * 1.5e-2)
                                XCTAssertEqual(Double(got[idx]), Double(t4Prefix[idx]),
                                               accuracy: prefixTol,
                                               "T=10 vs T=4 mismatch isMX=\(isMX) I=\(inter) x=\(xDt) idx=\(idx)")
                            }
                        }
                        for t in 0..<curT {
                            var ref = [Double](repeating: 0, count: hidden)
                            var mag = [Double](repeating: 0, count: hidden)
                            var l1 = [Double](repeating: 0, count: hidden)
                            for r in 0..<topK {
                                let e = Int(idVals[t * topK + r]), w = Double(wtVals[t * topK + r])
                                let roundW = isMX || (curT > 8 && xDt == .bf16)
                                let act = (0..<inter).map { i -> Double in
                                    var gs = 0.0, us = 0.0
                                    for k in 0..<hidden {
                                        let xv = xDt == .bf16 ? Double(f32FromBF16(xBF16[t * hidden + k])) : Double(xF16[t * hidden + k])
                                        let rawG = isMX ? mxG.ref[(e * 2 * inter + i) * hidden + k] : gP.ref[(e * inter + i) * hidden + k]
                                        let rawU = isMX ? mxG.ref[(e * 2 * inter + inter + i) * hidden + k] : uP.ref[(e * inter + i) * hidden + k] * 1.5
                                        gs += xv * (roundW ? roundBF16(rawG) : rawG)
                                        us += xv * (roundW ? roundBF16(rawU) : rawU)
                                    }
                                    let sg = tanh(gs / 4.0) * 4.0 * (1.0 / (1.0 + exp(-gs)))
                                    let su = tanh(us / 25.0) * 25.0
                                    return roundInter(sg * su, xDt)
                                }
                                for h in 0..<hidden {
                                    var dn = 0.0, dl = 0.0
                                    for i in 0..<inter {
                                        let rawD = isMX ? mxD.ref[(e * hidden + h) * inter + i] : dP.ref[(e * hidden + h) * inter + i]
                                        let prod = act[i] * (roundW ? roundBF16(rawD) : rawD)
                                        dn += prod; dl += abs(prod)
                                    }
                                    let term = w * roundInter(dn, xDt)
                                    ref[h] += term; mag[h] += abs(term); l1[h] += w * dl
                                }
                            }
                            for h in 0..<hidden {
                                let tol = xDt == .bf16
                                    ? max(0.25, max(mag[h] * 3e-2, l1[h] * 1.5e-2))
                                    : max(0.08, mag[h] * 2.5e-2)
                                if curT == 4 { t4Tol.append(tol) }
                                XCTAssertEqual(Double(got[t * hidden + h]), ref[h],
                                               accuracy: tol,
                                               "isMX=\(isMX) I=\(inter) T=\(curT) x=\(xDt) [\(t),\(h)]")
                            }
                        }
                    }
                }
            }
        }
    }

    func testFusedFPMoENaxSwiGLUHeterogeneousBM16_32_64DirectParity() throws {
        let hal = try makeHal()
        let counts = [0, 1, 16, 17, 32, 66]
        let experts = counts.count, topK = 3, hidden = 64, inter = 64
        let totalPairs = counts.reduce(0, +), tokens = totalPairs / topK
        var idVals = [Int32]()
        for (e, c) in counts.enumerated() { for _ in 0..<c { idVals.append(Int32(e)) } }
        let xF16 = (0..<(tokens * hidden)).map { Float16(sin(Double($0) * 0.13) * 0.35) }
        let xBF16 = xF16.map { bf16Bits(Float($0)) }

        let gP = packNVFP4(experts: experts, rows: inter, k: hidden, seed: 0x6100)
        let uP = packNVFP4(experts: experts, rows: inter, k: hidden, seed: 0x6200)
        var guW: [UInt32] = [], guSc: [UInt8] = []
        let wExp = inter * (hidden / 8), sExp = inter * (hidden / 16)
        for e in 0..<experts {
            guW.append(contentsOf: gP.w[(e * wExp)..<((e + 1) * wExp)])
            guW.append(contentsOf: uP.w[(e * wExp)..<((e + 1) * wExp)])
            guSc.append(contentsOf: gP.sc[(e * sExp)..<((e + 1) * sExp)])
            guSc.append(contentsOf: uP.sc[(e * sExp)..<((e + 1) * sExp)])
        }
        let nvBank = QWGemmFP.NVFP4Bank(
            weights: try sharedSlice(hal, values: guW), scales: try sharedSlice(hal, values: guSc),
            scales2: try sharedSlice(hal, values: gP.s2 + uP.s2.map { $0 * 1.5 }),
            experts: experts, rowsPerExpert: 2 * inter, k: hidden, splitGateUpScales2: true)
        let mxP = packMXFP8(experts: experts, rows: 2 * inter, k: hidden, seed: 0x6300)
        let mxBank = QWGemmFP.MXFP8Bank(
            weights: try sharedSlice(hal, values: mxP.w), scales: try sharedSlice(hal, values: mxP.sc),
            experts: experts, rowsPerExpert: 2 * inter, k: hidden)

        let arena = try QWArena(hal: hal, weights: 1 << 16, activations: 8 << 20, kv: 0, gdnStates: 0)
        let ar = arena.activationsRegion()
        let xF16TV = try ar.allocate(QWTensorShape([tokens, hidden], .f16)); arena.write(xF16TV, from: xF16.map(\.bitPattern))
        let xBF16TV = try ar.allocate(QWTensorShape([tokens, hidden], .bf16)); arena.write(xBF16TV, from: xBF16)
        let idsTV = try ar.allocate(QWTensorShape([totalPairs], .i32)); arena.write(idsTV, from: idVals)
        let expOffTV = try ar.allocate(QWTensorShape([experts + 1], .i32))
        let sortedPairTV = try ar.allocate(QWTensorShape([totalPairs], .i32))
        let sortedTokTV = try ar.allocate(QWTensorShape([totalPairs], .i32))
        let invPermTV = try ar.allocate(QWTensorShape([totalPairs], .i32))
        let tilePlan = QWGemm.MoETilePlan(experts: experts, pairs: totalPairs, gateUpRows: 2 * inter, downRows: hidden)
        let tileETV = try ar.allocate(QWTensorShape([tilePlan.maxTiles], .i32))
        let tileRTV = try ar.allocate(QWTensorShape([tilePlan.maxTiles], .i32))
        let tileCountTV = try ar.allocate(QWTensorShape([tilePlan.scratchInts], .i32))
        let actTV = try ar.allocate(QWTensorShape([totalPairs, inter], .f16))

        for isMX in [false, true] {
            for xDt in [QWFPElemDtype.f16, .bf16] {
                guard let (cb, initEnc) = hal.begin() else { XCTFail("begin failed"); return }
                var enc = initEnc
                try QWGemm.encodeMoEGroupPairsF32(
                    enc, hal: hal, ids: idsTV, expertOff: expOffTV,
                    sortedPairIdx: sortedPairTV, sortedTok: sortedTokTV,
                    experts: experts, totalPairs: totalPairs, routes: topK,
                    invPerm: invPermTV, tileE: tileETV, tileR: tileRTV,
                    tileCount: tileCountTV, tilePlan: tilePlan)
                try hal.rotateEncoder(cb, &enc)
                let xSl = QWFPBufferSlice(xDt == .bf16 ? xBF16TV : xF16TV)
                let actSl = QWFPBufferSlice(actTV)
                if isMX {
                    try QWGemmFP.encodeMoENaxMXFP8SwiGLU(
                        enc, hal: hal, bank: mxBank, x: xSl, actOut: actSl,
                        expertOff: expOffTV, tileE: tileETV, tileR: tileRTV,
                        tileCount: tileCountTV, tilePlan: tilePlan, intermediate: inter,
                        inDtype: xDt, outDtype: xDt, gateCap: 4.0, upCap: 25.0, sortedTok: sortedTokTV)
                } else {
                    try QWGemmFP.encodeMoENaxNVFP4SwiGLU(
                        enc, hal: hal, bank: nvBank, x: xSl, actOut: actSl,
                        expertOff: expOffTV, tileE: tileETV, tileR: tileRTV,
                        tileCount: tileCountTV, tilePlan: tilePlan, intermediate: inter,
                        inDtype: xDt, outDtype: xDt, gateCap: 4.0, upCap: 25.0, sortedTok: sortedTokTV)
                }
                hal.end(cb, enc)
                cb.waitUntilCompleted()
                XCTAssertNil(cb.error)

                let expOff = arena.read(expOffTV, as: Int32.self, count: experts + 1)
                let sTok = arena.read(sortedTokTV, as: Int32.self, count: totalPairs)
                let rawAct = arena.read(actTV, as: UInt16.self, count: totalPairs * inter)
                for e in 0..<experts {
                    for s in Int(expOff[e])..<Int(expOff[e + 1]) {
                        let t = Int(sTok[s])
                        for i in 0..<inter {
                            var gs = 0.0, us = 0.0
                            for k in 0..<hidden {
                                let xv = xDt == .bf16 ? Double(f32FromBF16(xBF16[t * hidden + k])) : Double(xF16[t * hidden + k])
                                let rG = isMX ? mxP.ref[(e * 2 * inter + i) * hidden + k] : gP.ref[(e * inter + i) * hidden + k]
                                let rU = isMX ? mxP.ref[(e * 2 * inter + inter + i) * hidden + k] : uP.ref[(e * inter + i) * hidden + k] * 1.5
                                let wG = (isMX || xDt == .bf16) ? roundBF16(rG) : rG
                                let wU = (isMX || xDt == .bf16) ? roundBF16(rU) : rU
                                gs += xv * wG; us += xv * wU
                            }
                            let sg = tanh(gs / 4.0) * 4.0 * (1.0 / (1.0 + exp(-gs)))
                            let su = tanh(us / 25.0) * 25.0
                            let want = roundInter(sg * su, xDt)
                            let raw = rawAct[s * inter + i]
                            let got = xDt == .bf16 ? Double(f32FromBF16(raw)) : Double(Float(Float16(bitPattern: raw)))
                            XCTAssertEqual(got, want, accuracy: max(0.03, abs(want) * 2e-2),
                                           "direct SwiGLU BM16/32/64 isMX=\(isMX) x=\(xDt) e=\(e) s=\(s) i=\(i)")
                        }
                    }
                }
            }
        }
    }
}
