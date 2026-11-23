// QWGemmSwiGLUNAXTests.swift — 0-ULP GPU equivalence tests for fused NAX gate_up + SwiGLU + FWHT-32.
import Foundation
import Metal
import XCTest
@testable import QWGEMM
import QWAlloc
import QWCore
import QWHAL
import QWQuant

final class QWGemmSwiGLUNAXTests: XCTestCase {
    private enum FormatSpec {
        case hq4g32
        case aq(bits: Int, group: Int)
        case shvq(bits: Int, group: Int)

        var label: String {
            switch self {
            case .hq4g32: return "HQ4_G32"
            case .aq(let b, let g): return "AQ\(b)_G\(g)"
            case .shvq(let b, let g): return "SHVQ\(b)_G\(g)"
            }
        }

        var gemmKernel: String {
            switch self {
            case .hq4g32: return "qwg_gemm_q4g_h16"
            case .aq(let b, let g): return QWGemmAQ.gemmKernelName(bits: b, group: g, isSHVQ: false)!
            case .shvq(let b, let g): return QWGemmAQ.gemmKernelName(bits: b, group: g, isSHVQ: true)!
            }
        }

        var swigluKernel: String {
            switch self {
            case .hq4g32: return "qwg_gemm_q4g_h16_swiglu"
            case .aq(let b, let g): return QWGemmAQ.swigluGemmKernelName(bits: b, group: g, isSHVQ: false)!
            case .shvq(let b, let g): return QWGemmAQ.swigluGemmKernelName(bits: b, group: g, isSHVQ: true)!
            }
        }
    }

    private func makeWeights(_ spec: FormatSpec, rows: Int, k: Int) throws -> Data {
        var seed: UInt32 = 0x51A7_1234
        func nextRand() -> UInt32 {
            seed = seed &* 1664525 &+ 1013904223
            return seed
        }
        switch spec {
        case .hq4g32:
            let groups = k / 32
            let gBytes = 20
            var out = Data(count: rows * groups * gBytes)
            out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                for r in 0..<rows {
                    for g in 0..<groups {
                        let base = (r * groups + g) * gBytes
                        let scVal = Float(Float16(0.015 * Float(1 + (r + g) % 5)))
                        let sc = Float16(scVal).bitPattern
                        let bi = Float16(-1031.5 * scVal).bitPattern
                        ptr[base] = UInt8(sc & 0xFF)
                        ptr[base + 1] = UInt8(sc >> 8)
                        ptr[base + 2] = UInt8(bi & 0xFF)
                        ptr[base + 3] = UInt8(bi >> 8)
                        for b in 0..<16 {
                            ptr[base + 4 + b] = UInt8((nextRand() >> 16) & 0xFF)
                        }
                    }
                }
            }
            return out
        case .aq(let bits, let group):
            let groups = k / group
            let maxCode = UInt32((1 << bits) - 1)
            var out = Data()
            out.reserveCapacity(rows * groups * (group * bits / 8 + 4))
            for r in 0..<rows {
                for g in 0..<groups {
                    let vals = (0..<group).map { _ in (nextRand() >> 12) % (maxCode + 1) }
                    let sc: Float = 0.015625 * Float(1 + (r + g) % 4)
                    let bi: Float = -0.25
                    out.append(try QWAQBlock.pack(bits: bits, group: group, values: vals, scale: sc, bias: bi))
                }
            }
            return out
        case .shvq(let bits, let group):
            let groups = k / group
            let gBytes = QWSHVQBlock.bytes(bits: bits, group: group)
            var out = Data(count: rows * groups * gBytes)
            out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
                guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
                for r in 0..<rows {
                    for g in 0..<groups {
                        let base = (r * groups + g) * gBytes
                        let sc = Float16(0.025 * Float(1 + (r + g) % 5)).bitPattern
                        ptr[base] = UInt8(sc & 0xFF)
                        ptr[base + 1] = UInt8(sc >> 8)
                        for b in 2..<gBytes {
                            ptr[base + b] = UInt8((nextRand() >> 16) & 0xFF)
                        }
                    }
                }
            }
            return out
        }
    }

    func testFusedSwiGLUNAXMatchesReferenceZeroULP() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "Fused NAX SwiGLU tests require bare-metal Metal 4 tensor GPU"
        )
        let formats: [FormatSpec] = [.hq4g32, .aq(bits: 4, group: 32), .aq(bits: 8, group: 32),
                                     .shvq(bits: 4, group: 32), .shvq(bits: 3, group: 64)]
        let mValues = [1, 7, 16, 17, 32, 33, 64, 96]
        let iValues = [64, 96, 128, 256]
        let k = 128
        let maxM = mValues.max()!

        for basis in [QWWeightBasis.rotated, .standard] {
            let hal = try QWHal(geometry: .qwen38_27b, weightBasis: basis)
            let psoSilu = try hal.pipeline("qwg_silu_mul_f16")
            var actSeed: UInt32 = 0x1357_9BDF
            var xVals = [Float16](repeating: 0, count: maxM * k)
            for idx in xVals.indices {
                actSeed = actSeed &* 1664525 &+ 1013904223
                let u = Float(Int32(bitPattern: actSeed)) / Float(Int32.max)
                xVals[idx] = Float16(u * 0.75)
            }

            for spec in formats {
                let psoGemm = try hal.pipeline(spec.gemmKernel)
                let psoFused = try hal.pipeline(spec.swigluKernel)
                for iDim in iValues {
                    let wData = try makeWeights(spec, rows: 2 * iDim, k: k)
                    let arena = try QWArena(hal: hal, weights: wData.count + (1 << 20),
                                            activations: 16 << 20, kv: 0, gdnStates: 0)
                    let wBuf = try arena.weightsRegion().allocate(QWTensorShape([(wData.count + 1) / 2], .f16))
                    arena.copyIn(wBuf, from: wData)
                    let act = arena.activationsRegion()
                    let xBuf = try act.allocate(QWTensorShape([maxM * k], .f16))
                    let guRaw = try act.allocate(QWTensorShape([maxM * 2 * iDim], .f16))
                    let refOut = try act.allocate(QWTensorShape([maxM * iDim], .f16))
                    let fusedOut = try act.allocate(QWTensorShape([maxM * iDim], .f16))
                    xBuf.region.buffer.contents().advanced(by: xBuf.region.base + xBuf.offset)
                        .copyMemory(from: xVals, byteCount: xVals.count * 2)

                    for m in mValues {
                        let refBM = m <= 16 ? 16 : (m <= 32 ? 32 : 64)
                        let refBN = m <= 16 ? 128 : 64
                        let refMode = m <= 16 ? 0x2000_0000 : (m <= 32 ? 0x4000_0000 : 0)

                        let use64 = m <= 16 && (iDim & 63) == 0
                        let fusBM = use64 ? 16 : (m <= 32 ? 32 : 64)
                        let fusBN = use64 ? 64 : 32
                        let fusMode = use64 ? 0x2000_0000 : (m <= 32 ? 0x4000_0000 : 0)

                        guard let (cb, enc) = hal.begin() else {
                            XCTFail("failed to create Metal command buffer")
                            return
                        }
                        // Reference path: qwg_gemm_*_h16 + qwg_silu_mul_f16
                        enc.setComputePipelineState(psoGemm)
                        wBuf.bind(enc, index: 0); xBuf.bind(enc, index: 1)
                        guRaw.bind(enc, index: 2); guRaw.bind(enc, index: 3)
                        enc.setUInt32s(m | refMode, 2 * iDim, k, startIndex: 4)
                        enc.dispatchThreadgroups(
                            MTLSize(width: (2 * iDim + refBN - 1) / refBN, height: (m + refBM - 1) / refBM, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)

                        enc.setComputePipelineState(psoSilu)
                        guRaw.bind(enc, index: 0); refOut.bind(enc, index: 1)
                        enc.setUInt32s(m, iDim, startIndex: 2)
                        enc.dispatchThreadgroups(
                            MTLSize(width: max(1, (((iDim + 7) / 8) + 127) / 128), height: m, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)

                        // Fused path: qwg_gemm_*_h16_swiglu
                        enc.setComputePipelineState(psoFused)
                        wBuf.bind(enc, index: 0); xBuf.bind(enc, index: 1); fusedOut.bind(enc, index: 2)
                        enc.setUInt32s(m | fusMode, iDim, k, k, iDim, startIndex: 3)
                        enc.dispatchThreadgroups(
                            MTLSize(width: iDim / fusBN, height: (m + fusBM - 1) / fusBM, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

                        hal.end(cb, enc)
                        cb.waitUntilCompleted()
                        XCTAssertNil(cb.error)

                        let count = m * iDim
                        let refBits = Self.readUInt16(refOut, count: count)
                        let fusBits = Self.readUInt16(fusedOut, count: count)
                        XCTAssertEqual(fusBits, refBits,
                                       "\(basis) \(spec.label) M=\(m) I=\(iDim) K=\(k) not 0-ULP exact")
                    }
                }
            }
        }
    }

    private static func readUInt16(_ t: QWTensorValue, count: Int) -> [UInt16] {
        let ptr = t.region.buffer.contents().advanced(by: t.region.base + t.offset).assumingMemoryBound(to: UInt16.self)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    func testSwiGLUSoftcapF32AndFWHT32Parity() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current != nil,
            "SwiGLU softcap FWHT-32 parity requires Metal GPU"
        )
        let rows = 4
        let iDim = 96
        let gateCap: Float = 4.0
        let upCap: Float = 25.0
        var seed: UInt64 = 0x51A9_2048_1337
        var guF32 = [Float](repeating: 0, count: rows * 2 * iDim)
        var guF16 = [Float16](repeating: 0, count: rows * 2 * iDim)
        for i in guF32.indices {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            let v = Float(Int32(truncatingIfNeeded: seed >> 32)) / Float(Int32.max) * 3.5
            guF32[i] = v
            guF16[i] = Float16(v)
        }

        func cpuRef(_ vals: [Float], rotate: Bool) -> [Float] {
            var out = [Float](repeating: 0, count: rows * iDim)
            for r in 0..<rows {
                for j in 0..<iDim {
                    let g = vals[r * 2 * iDim + j]
                    let u = vals[r * 2 * iDim + iDim + j]
                    let sig = 1.0 / (1.0 + expf(-g))
                    let gs = tanhf(min(15.0, max(-15.0, g / gateCap))) * gateCap * sig
                    let us = tanhf(min(15.0, max(-15.0, u / upCap))) * upCap
                    out[r * iDim + j] = gs * us
                }
                if rotate {
                    out.withUnsafeMutableBufferPointer { buf in
                        var off = 0
                        while off < iDim {
                            QWQuant.fwht32OrthoBlock(buf.baseAddress!.advanced(by: r * iDim + off))
                            off += 32
                        }
                    }
                }
            }
            return out
        }

        for basis in [QWWeightBasis.rotated, .standard] {
            let hal = try QWHal(geometry: .qwen38_27b, weightBasis: basis)
            guard let gu32Buf = hal.device.makeBuffer(bytes: &guF32, length: guF32.count * 4, options: .storageModeShared),
                  let act32Buf = hal.device.makeBuffer(length: rows * iDim * 4, options: .storageModeShared),
                  let gu16Buf = hal.device.makeBuffer(bytes: &guF16, length: guF16.count * 2, options: .storageModeShared),
                  let act16Buf = hal.device.makeBuffer(length: rows * iDim * 2, options: .storageModeShared) else {
                XCTFail("buffer allocation failed")
                return
            }
            guard let (cb, enc) = hal.begin() else {
                XCTFail("failed to begin command buffer")
                return
            }
            try QWGemmFP.encodeSwiGLUSoftcap(
                enc, hal: hal,
                gu: QWFPBufferSlice(buffer: gu32Buf, offset: 0, bytes: gu32Buf.length),
                act: QWFPBufferSlice(buffer: act32Buf, offset: 0, bytes: act32Buf.length),
                rows: rows, intermediate: iDim,
                gateCap: gateCap, upCap: upCap,
                inDtype: .f32, outDtype: .f32, rotateFwht32: true
            )
            try QWGemmFP.encodeSwiGLUSoftcap(
                enc, hal: hal,
                gu: QWFPBufferSlice(buffer: gu16Buf, offset: 0, bytes: gu16Buf.length),
                act: QWFPBufferSlice(buffer: act16Buf, offset: 0, bytes: act16Buf.length),
                rows: rows, intermediate: iDim,
                gateCap: gateCap, upCap: upCap,
                inDtype: .f16, outDtype: .f16, rotateFwht32: true
            )
            hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)

            let rotateExpected = (basis == .rotated)
            let ref32 = cpuRef(guF32, rotate: rotateExpected)
            let got32 = Array(UnsafeBufferPointer(
                start: act32Buf.contents().assumingMemoryBound(to: Float.self), count: rows * iDim
            ))
            for idx in got32.indices {
                XCTAssertEqual(got32[idx], ref32[idx], accuracy: 1e-4, "F32 softcap+FWHT mismatch at \(idx)")
            }

            let ref16 = cpuRef(guF16.map { Float($0) }, rotate: rotateExpected)
            let got16 = Array(UnsafeBufferPointer(
                start: act16Buf.contents().assumingMemoryBound(to: Float16.self), count: rows * iDim
            ))
            for idx in got16.indices {
                XCTAssertEqual(Float(got16[idx]), ref16[idx], accuracy: 1e-2, "F16 softcap+FWHT mismatch at \(idx)")
            }
        }
    }

    func testSHVQ4UnalignedN32NAXAndDirectBF16DecodeParity() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "SHVQ4 unaligned-N NAX & BF16 decode tests require Metal 4 tensor GPU"
        )
        let k = 128
        let maxRows = 48
        let hal = try QWHal(geometry: .qwen38_27b, weightBasis: .rotated)

        var seed: UInt32 = 0xCAFE_4321
        var xVals = [Float16](repeating: 0, count: maxRows * k)
        for idx in xVals.indices {
            seed = seed &* 1664525 &+ 1013904223
            let u = Float(Int32(bitPattern: seed)) / Float(Int32.max)
            xVals[idx] = Float16(u * 0.5)
        }

        func unpackSHVQ4Rows(n: Int, from data: Data) -> [[Float]] {
            let groups = k / 32
            let gBytes = QWSHVQBlock.bytes(bits: 4, group: 32)
            return (0..<n).map { r in
                var row = [Float]()
                row.reserveCapacity(k)
                var tmp = [Float]()
                for g in 0..<groups {
                    let off = (r * groups + g) * gBytes
                    let sub = data.subdata(in: off..<(off + gBytes))
                    sub.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                        QWSHVQBlock.unpack(group: raw, bits: 4, group: 32, into: &tmp)
                    }
                    row.append(contentsOf: tmp)
                }
                return row
            }
        }

        func dotRef(_ rowW: [Float], rowIdx: Int) -> Float {
            var acc: Float = 0
            let base = rowIdx * k
            for j in 0..<k { acc += rowW[j] * Float(xVals[base + j]) }
            return acc
        }

        // 1. Unaligned N=32 NAX F32 / F16 / BF16 parity (n % 64 != 0).
        let n32 = 32
        let w32Data = try makeWeights(.shvq(bits: 4, group: 32), rows: n32, k: k)
        let w32Ref = unpackSHVQ4Rows(n: n32, from: w32Data)
        let arena = try QWArena(hal: hal, weights: w32Data.count + (1 << 20),
                                activations: 8 << 20, kv: 0, gdnStates: 0)
        let w32Buf = try arena.weightsRegion().allocate(QWTensorShape([(w32Data.count + 1) / 2], .f16))
        arena.copyIn(w32Buf, from: w32Data)
        guard let w32View = QWWeightView(blocks: w32Buf, n: n32, k: k, bits: 4, groupSize: 32,
                                         isRotated: true, isIncumbentContainer: false, isSHVQ: true) else {
            XCTFail("failed to construct SHVQ4 N=32 QWWeightView")
            return
        }
        let act = arena.activationsRegion()
        let x16 = try act.allocate(QWTensorShape([maxRows * k], .f16))
        x16.region.buffer.contents().advanced(by: x16.region.base + x16.offset)
            .copyMemory(from: xVals, byteCount: xVals.count * 2)
        let yF32 = try act.allocate(QWTensorShape([maxRows * n32], .f32))
        let yF16 = try act.allocate(QWTensorShape([maxRows * n32], .f16))
        let yBF16 = try act.allocate(QWTensorShape([maxRows * n32], .bf16))

        guard let (cbNax, encNax) = hal.begin() else {
            XCTFail("failed to create Metal command buffer")
            return
        }
        try QWGemm.encodeNaxF32(encNax, hal: hal, weight: w32View, x16: x16, y16: x16, y32: yF32, rows: maxRows)
        encNax.memoryBarrier(scope: .buffers)
        try QWGemm.encodeNaxF32(encNax, hal: hal, weight: w32View, x16: x16, y16: x16, y32: yF16, rows: maxRows)
        encNax.memoryBarrier(scope: .buffers)
        try QWGemm.encodeNaxF32(encNax, hal: hal, weight: w32View, x16: x16, y16: x16, y32: yBF16, rows: maxRows)
        hal.end(cbNax, encNax)
        cbNax.waitUntilCompleted()
        XCTAssertNil(cbNax.error)

        let f32Ptr = yF32.region.buffer.contents().advanced(by: yF32.region.base + yF32.offset)
            .assumingMemoryBound(to: Float.self)
        let gotF32 = Array(UnsafeBufferPointer(start: f32Ptr, count: maxRows * n32))
        let gotF16 = Self.readUInt16(yF16, count: maxRows * n32)
        let gotBF16 = Self.readUInt16(yBF16, count: maxRows * n32)
        for m in 0..<maxRows {
            for col in 0..<n32 {
                let ref = dotRef(w32Ref[col], rowIdx: m)
                let idx = m * n32 + col
                XCTAssertEqual(gotF32[idx], ref, accuracy: 5e-3, "NAX N=32 F32 mismatch at (\(m),\(col))")
                XCTAssertEqual(Float(Float16(bitPattern: gotF16[idx])), ref, accuracy: 1e-2,
                               "NAX N=32 F16 mismatch at (\(m),\(col))")
                XCTAssertEqual(QWBF16.toFloat(gotBF16[idx]), ref, accuracy: max(2e-2, abs(ref) * 2e-2),
                               "NAX N=32 BF16 mismatch at (\(m),\(col))")
            }
        }

        // 2. Direct BF16 decode GEMV on N=96 and N=32: batched rows=4,8 must match rows=1 bit-for-bit.
        for n in [96, 32] {
            let wData = try makeWeights(.shvq(bits: 4, group: 32), rows: n, k: k)
            let wRef = unpackSHVQ4Rows(n: n, from: wData)
            let a = try QWArena(hal: hal, weights: wData.count + (1 << 20),
                                activations: 4 << 20, kv: 0, gdnStates: 0)
            let wBuf = try a.weightsRegion().allocate(QWTensorShape([(wData.count + 1) / 2], .f16))
            a.copyIn(wBuf, from: wData)
            let reg = a.activationsRegion()
            let xBuf = try reg.allocate(QWTensorShape([8 * k], .f16))
            xBuf.region.buffer.contents().advanced(by: xBuf.region.base + xBuf.offset)
                .copyMemory(from: xVals, byteCount: 8 * k * 2)
            let ySeq = try reg.allocate(QWTensorShape([8 * n], .bf16))
            let yR4 = try reg.allocate(QWTensorShape([4 * n], .bf16))
            let yR8 = try reg.allocate(QWTensorShape([8 * n], .bf16))

            guard let (cb, enc) = hal.begin() else {
                XCTFail("failed to create Metal command buffer")
                return
            }
            for r in 0..<8 {
                let xSlice = xBuf.slice(byteOffset: r * k * 2, count: k, dtype: .f16)
                let ySlice = ySeq.slice(byteOffset: r * n * 2, count: n, dtype: .bf16)
                try QWGemmAQ.encodeDecodeBF16(enc, hal: hal, blocks: wBuf, x: xSlice, yBF16: ySlice,
                                              n: n, k: k, bits: 4, group: 32, rows: 1, isSHVQ: true)
            }
            let x4 = xBuf.slice(byteOffset: 0, count: 4 * k, dtype: .f16)
            try QWGemmAQ.encodeDecodeBF16(enc, hal: hal, blocks: wBuf, x: x4, yBF16: yR4,
                                          n: n, k: k, bits: 4, group: 32, rows: 4, isSHVQ: true)
            try QWGemmAQ.encodeDecodeBF16(enc, hal: hal, blocks: wBuf, x: xBuf, yBF16: yR8,
                                          n: n, k: k, bits: 4, group: 32, rows: 8, isSHVQ: true)
            hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)

            let bitsSeq = Self.readUInt16(ySeq, count: 8 * n)
            let bitsR4 = Self.readUInt16(yR4, count: 4 * n)
            let bitsR8 = Self.readUInt16(yR8, count: 8 * n)
            XCTAssertEqual(bitsR4, Array(bitsSeq.prefix(4 * n)),
                           "SHVQ4 encodeDecodeBF16 N=\(n) rows=4 not 0-ULP identical to rows=1")
            func bf16Ordered(_ u: UInt16) -> Int32 {
                (u & 0x7FFF) == 0 ? 0 : ((u & 0x8000) != 0 ? -Int32(u & 0x7FFF) : Int32(u))
            }
            XCTAssertEqual(bitsR8.count, bitsSeq.count)
            for idx in 0..<bitsSeq.count {
                let ulp = abs(bf16Ordered(bitsR8[idx]) - bf16Ordered(bitsSeq[idx]))
                XCTAssertLessThanOrEqual(
                    ulp, 1,
                    "SHVQ4 encodeDecodeBF16 N=\(n) rows=8 ULP=\(ulp) (\(bitsR8[idx]) vs \(bitsSeq[idx])) at \(idx)")
            }

            for r in 0..<8 {
                for col in 0..<n {
                    let ref = dotRef(wRef[col], rowIdx: r)
                    let got = QWBF16.toFloat(bitsR8[r * n + col])
                    XCTAssertEqual(got, ref, accuracy: max(2e-2, abs(ref) * 2e-2),
                                   "SHVQ4 encodeDecodeBF16 N=\(n) mismatch at (\(r),\(col))")
                }
            }
        }
    }
}
