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
}
