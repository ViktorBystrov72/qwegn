// QWFPBranchlessDequantTests.swift — 0-ULP bitPattern verification for branchless FP4/FP8
// dequantization and BF16 BM64 / BN64 SwiGLU NAX / T=1 compact MoE paths.
import Foundation
import Metal
import XCTest
@testable import QWGEMM
import QWAlloc
import QWCore
import QWHAL

final class QWFPBranchlessDequantTests: XCTestCase {
    private func makeHal() throws -> QWHal {
        do {
            return try QWHal(geometry: .qwen38_27b)
        } catch {
            throw XCTSkip("Metal HAL unavailable: \(error)")
        }
    }

    private func makeSlice<T>(_ hal: QWHal, values: [T]) throws -> (any MTLBuffer, QWFPBufferSlice) {
        let len = max(values.count * MemoryLayout<T>.stride, 64)
        guard let buf = hal.device.makeBuffer(length: len, options: .storageModeShared) else {
            throw XCTSkip("MTLBuffer allocation failed")
        }
        values.withUnsafeBytes { raw in
            if let src = raw.baseAddress, raw.count > 0 {
                buf.contents().copyMemory(from: src, byteCount: raw.count)
            }
        }
        return (buf, QWFPBufferSlice(buffer: buf, offset: 0, bytes: len))
    }

    private func referenceFP8E4M3FN(_ b: UInt8) -> Float {
        let sign: UInt32 = (b & 0x80) != 0 ? 0x8000_0000 : 0
        let e = Int((b >> 3) & 0xF)
        let m = Int(b & 0x7)
        if e == 15 && m == 7 { return Float(bitPattern: sign | 0x7FC0_0000) }
        if e == 0 {
            if m == 0 { return Float(bitPattern: sign) }
            return (sign != 0 ? -1.0 : 1.0) * Float(m) * (1.0 / 512.0)
        }
        let mag = ldexpf(1.0 + Float(m) / 8.0, Int32(e - 7))
        return sign != 0 ? -mag : mag
    }

    func testAll256FP8BytesAndAll16FP4NibblesExactBitPattern() throws {
        let hal = try makeHal()
        let arena = try QWArena(hal: hal, weights: 1 << 16, activations: 2 << 20, kv: 0, gdnStates: 0)
        let ar = arena.activationsRegion()

        // 1) Test all 256 E4M3FN bytes via qwg_moe_compact_mxfp8_down_t1 (x[0] = 1.0, scale = 127 => 1.0).
        let mxRows = 256, mxK = 32
        var mxW = [UInt32](repeating: 0, count: mxRows * (mxK / 4))
        for r in 0..<mxRows { mxW[r * 8] = UInt32(r) }
        let mxSc = [UInt8](repeating: 127, count: mxRows)
        let (_, mxWSlice) = try makeSlice(hal, values: mxW)
        let (_, mxScSlice) = try makeSlice(hal, values: mxSc)
        let mxBank = QWGemmFP.MXFP8Bank(weights: mxWSlice, scales: mxScSlice, experts: 1, rowsPerExpert: mxRows, k: mxK)

        var xMx = [Float](repeating: 0, count: mxK); xMx[0] = 1.0
        let (_, xMxSlice) = try makeSlice(hal, values: xMx)
        let (_, expZeroSlice) = try makeSlice(hal, values: [Int32](repeating: 0, count: 2))
        let (_, idsZeroSlice) = try makeSlice(hal, values: [Int32(0)])
        let mxOutTV = try ar.allocate(QWTensorShape([mxRows], .f32))

        guard let (cb1, enc1) = hal.begin() else { XCTFail("begin failed"); return }
        try QWGemmFP.encodeMoECompactMXFP8Down(
            enc1, hal: hal, actIn: xMxSlice, down: mxBank,
            expertOff: expZeroSlice, activeExperts: idsZeroSlice,
            activeArgs: expZeroSlice, sortedPairIdx: idsZeroSlice,
            slotOut: QWFPBufferSlice(mxOutTV), topK: 1,
            inDtype: .f32, outDtype: .f32, sortedOrder: false,
            useIndirectDispatch: false, directT1: true)
        hal.end(cb1, enc1)
        cb1.waitUntilCompleted()
        XCTAssertNil(cb1.error)
        let gotFP8 = arena.read(mxOutTV, as: UInt32.self, count: mxRows)
        for b in 0..<256 {
            let ref = referenceFP8E4M3FN(UInt8(b))
            let gotF = Float(bitPattern: gotFP8[b])
            if ref.isNaN {
                XCTAssertTrue(gotF.isNaN, "FP8 0x\(String(b, radix: 16)) must decode to NaN")
            } else if ref == 0 {
                XCTAssertEqual(gotF, 0, "FP8 0x\(String(b, radix: 16)) must decode to zero")
            } else {
                XCTAssertEqual(gotFP8[b], ref.bitPattern,
                               "FP8 0x\(String(b, radix: 16)) bitPattern mismatch: got \(gotF) vs \(ref)")
            }
        }

        // 2) Test all 16 FP4 nibbles (scale = 0x38 => 1.0) and all 256 FP8 scale bytes (w[0] = 0x2 => 1.0) via NVFP4 T1.
        let fp4Ref: [Float] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        let nvRows = 256 + 16, nvK = 16
        var nvW = [UInt32](repeating: 0, count: nvRows * 2)
        var nvSc = [UInt8](repeating: 0x38, count: nvRows)
        for r in 0..<16 { nvW[r * 2] = UInt32(r) }
        for b in 0..<256 {
            nvW[(16 + b) * 2] = 0x0000_0002
            nvSc[16 + b] = UInt8(b)
        }
        let (_, nvWSlice) = try makeSlice(hal, values: nvW)
        let (_, nvScSlice) = try makeSlice(hal, values: nvSc)
        let (_, nvS2Slice) = try makeSlice(hal, values: [Float(1.0)])
        let nvBank = QWGemmFP.NVFP4Bank(
            weights: nvWSlice, scales: nvScSlice, scales2: nvS2Slice,
            experts: 1, rowsPerExpert: nvRows, k: nvK)
        var xNv = [Float](repeating: 0, count: nvK); xNv[0] = 1.0
        let (_, xNvSlice) = try makeSlice(hal, values: xNv)
        let nvOutTV = try ar.allocate(QWTensorShape([nvRows], .f32))

        guard let (cb2, enc2) = hal.begin() else { XCTFail("begin failed"); return }
        try QWGemmFP.encodeMoECompactNVFP4Down(
            enc2, hal: hal, actIn: xNvSlice, down: nvBank,
            expertOff: expZeroSlice, activeExperts: idsZeroSlice,
            activeArgs: expZeroSlice, sortedPairIdx: idsZeroSlice,
            slotOut: QWFPBufferSlice(nvOutTV), topK: 1,
            inDtype: .f32, outDtype: .f32, sortedOrder: false,
            useIndirectDispatch: false, directT1: true)
        hal.end(cb2, enc2)
        cb2.waitUntilCompleted()
        XCTAssertNil(cb2.error)
        let gotNV = arena.read(nvOutTV, as: UInt32.self, count: nvRows)
        for n in 0..<16 {
            if fp4Ref[n] == 0 {
                XCTAssertEqual(Float(bitPattern: gotNV[n]), 0, "FP4 nibble \(n) zero mismatch")
            } else {
                XCTAssertEqual(gotNV[n], fp4Ref[n].bitPattern, "FP4 nibble \(n) bitPattern mismatch")
            }
        }
        for b in 0..<256 {
            let ref = referenceFP8E4M3FN(UInt8(b))
            let gotF = Float(bitPattern: gotNV[16 + b])
            if ref.isNaN {
                XCTAssertTrue(gotF.isNaN, "NVFP4 scale 0x\(String(b, radix: 16)) must decode to NaN")
            } else if ref == 0 {
                XCTAssertEqual(gotF, 0, "NVFP4 scale 0x\(String(b, radix: 16)) zero mismatch")
            } else {
                XCTAssertEqual(gotNV[16 + b], ref.bitPattern,
                               "NVFP4 scale 0x\(String(b, radix: 16)) bitPattern mismatch")
            }
        }
    }

    func testBF16BM64AndRows4GemvParity() throws {
        let hal = try makeHal()
        guard hal.caps.isMetal4TensorCapable else { throw XCTSkip("Requires Metal 4") }
        let m = 72, n = 256, k = 128
        let arena = try QWArena(hal: hal, weights: n * k * 2 + 4096, activations: m * (k + n) * 4 + 4096, kv: 0, gdnStates: 0)
        let ar = arena.activationsRegion()
        var wBits = [UInt16](repeating: 0, count: n * k)
        for i in 0..<wBits.count {
            let v = Float(sin(Double(i) * 0.03) * 0.5)
            var u = v.bitPattern
            u = u &+ 0x7FFF &+ ((u >> 16) & 1)
            wBits[i] = UInt16(truncatingIfNeeded: u >> 16)
        }
        let (_, wSlice) = try makeSlice(hal, values: wBits)
        let wBank = QWGemmFP.BF16Bank(weights: wSlice, n: n, k: k)

        var xBits = [UInt16](repeating: 0, count: m * k)
        for i in 0..<xBits.count {
            let v = Float(cos(Double(i) * 0.07) * 0.4)
            var u = v.bitPattern
            u = u &+ 0x7FFF &+ ((u >> 16) & 1)
            xBits[i] = UInt16(truncatingIfNeeded: u >> 16)
        }
        let xTV = try ar.allocate(QWTensorShape([m * k], .bf16)); arena.write(xTV, from: xBits)
        let yNaxTV = try ar.allocate(QWTensorShape([m * n], .f32))
        let yRefTV = try ar.allocate(QWTensorShape([4 * n], .f32))

        guard let (cb, enc) = hal.begin() else { XCTFail("begin failed"); return }
        try QWGemmFP.encodeBF16NAX(enc, hal: hal, weight: wBank, x: QWFPBufferSlice(xTV), y: QWFPBufferSlice(yNaxTV),
                                   m: m, inDtype: .bf16, outDtype: .f32)
        let x4 = QWFPBufferSlice(xTV).slice(byteOffset: 0, bytes: 4 * k * 2)
        try QWGemmFP.encodeBF16(enc, hal: hal, weight: wBank, x: x4, y: QWFPBufferSlice(yRefTV),
                                m: 4, inDtype: .bf16, outDtype: .f32, minNaxRows: 9)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let nax = arena.read(yNaxTV, as: Float.self, count: m * n)
        let gemv4 = arena.read(yRefTV, as: Float.self, count: 4 * n)
        for r in 0..<m {
            for c in 0..<n {
                let ref = (0..<k).reduce(0.0) { acc, i in
                    let wv = Float(bitPattern: UInt32(wBits[c * k + i]) << 16)
                    let xv = Float(bitPattern: UInt32(xBits[r * k + i]) << 16)
                    return acc + Double(wv) * Double(xv)
                }
                XCTAssertEqual(Double(nax[r * n + c]), ref, accuracy: 0.06, "BM64 [\(r),\(c)]")
                if r < 4 {
                    XCTAssertEqual(Double(gemv4[r * n + c]), ref, accuracy: 1e-4, "rows4 [\(r),\(c)]")
                }
            }
        }
    }
}
