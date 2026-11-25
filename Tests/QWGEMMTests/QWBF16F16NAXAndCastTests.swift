// QWBF16F16NAXAndCastTests.swift — Parity gate for F16-input BF16 NAX GEMM & GPU f16->f32 cast.
import XCTest
import Foundation
import Metal
@testable import QWGEMM
import QWCore
import QWHAL
import QWAlloc

final class QWBF16F16NAXAndCastTests: XCTestCase {
    static let hal: QWHal = { try! QWHal(geometry: .qwen38_27b) }()

    private struct SplitMix64 {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &+ 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        mutating func nextSigned(_ scale: Float = 0.35) -> Float {
            let u = Float(next() >> 11) * (1.0 / 9007199254740992.0)
            return (u * 2.0 - 1.0) * scale
        }
    }

    @inline(__always)
    private static func f32ToBF16(_ v: Float) -> UInt16 {
        var bits = v.bitPattern
        if (bits & 0x7F80_0000) == 0x7F80_0000 { return UInt16(truncatingIfNeeded: bits >> 16) }
        bits = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: bits >> 16)
    }

    @inline(__always)
    private static func bf16ToF32(_ b: UInt16) -> Float {
        Float(bitPattern: UInt32(b) << 16)
    }

    func testF16ToF32CastParity() throws {
        for count in [64, 2048, 8192] {
            var rng = SplitMix64(state: UInt64(count))
            let vals = (0..<count).map { _ in Float16(rng.nextSigned(4.0)) }
            let inBytes = max(64, qwAlignUp(count * 2, 64))
            let outBytes = max(64, qwAlignUp(count * 4, 64))
            let inBuf = try XCTUnwrap(Self.hal.device.makeBuffer(length: inBytes, options: .storageModeShared))
            let outBuf = try XCTUnwrap(Self.hal.device.makeBuffer(length: outBytes, options: .storageModeShared))
            let inPtr = inBuf.contents().assumingMemoryBound(to: Float16.self)
            for (i, h) in vals.enumerated() { inPtr[i] = h }

            let x16 = QWTensorValue(region: QWMRegion(buffer: inBuf, base: 0, length: inBytes), offset: 0, shape: QWTensorShape([count], .f16))
            let y32 = QWTensorValue(region: QWMRegion(buffer: outBuf, base: 0, length: outBytes), offset: 0, shape: QWTensorShape([count], .f32))
            let (cb, enc) = try XCTUnwrap(Self.hal.begin())
            try QWGemm.encodeCastF16ToF32(enc, hal: Self.hal, x16: x16, y32: y32, count: count)
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertEqual(cb.status, .completed)

            let outPtr = outBuf.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<count {
                XCTAssertEqual(outPtr[i].bitPattern, Float(vals[i]).bitPattern)
            }
        }
    }

    func testBF16CoopFastNAXWithF16InputsAndF16OutputsParity() throws {
        try XCTSkipUnless(Self.hal.caps.isMetal4TensorCapable, "BF16 NAX requires Metal 4 tensor hardware")
        var rng = SplitMix64(state: 909)
        for (m, n, k) in [(17, 128, 64), (32, 192, 128), (48, 128, 256), (65, 256, 128), (131, 192, 256)] {
            let wFloats = (0..<(n * k)).map { _ in rng.nextSigned(0.25) }
            let wBF16 = wFloats.map(Self.f32ToBF16)
            let xFloats = (0..<(m * k)).map { _ in rng.nextSigned(0.5) }
            let xF16 = xFloats.map { Float16($0) }

            let wBytes = max(64, qwAlignUp(n * k * 2, 64))
            let xBytes = max(64, qwAlignUp(m * k * 2, 64))
            let yBytes = max(64, qwAlignUp(m * n * 2, 64))
            let wBuf = try XCTUnwrap(Self.hal.device.makeBuffer(length: wBytes, options: .storageModeShared))
            let xBuf = try XCTUnwrap(Self.hal.device.makeBuffer(length: xBytes, options: .storageModeShared))
            let yBuf = try XCTUnwrap(Self.hal.device.makeBuffer(length: yBytes, options: .storageModeShared))
            memcpy(wBuf.contents(), wBF16, n * k * 2)
            memcpy(xBuf.contents(), xF16, m * k * 2)

            let bank = QWGemmFP.BF16Bank(weights: QWFPBufferSlice(buffer: wBuf, offset: 0, bytes: n * k * 2), n: n, k: k)
            let xSlice = QWFPBufferSlice(buffer: xBuf, offset: 0, bytes: m * k * 2)
            let ySlice = QWFPBufferSlice(buffer: yBuf, offset: 0, bytes: m * n * 2)
            let (cb, enc) = try XCTUnwrap(Self.hal.begin())
            try QWGemmFP.encodeBF16NAX(enc, hal: Self.hal, weight: bank, x: xSlice, y: ySlice, m: m, inDtype: .f16, outDtype: .f16)
            Self.hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertEqual(cb.status, .completed)

            let yPtr = yBuf.contents().assumingMemoryBound(to: Float16.self)
            var maxDiff: Float = 0
            for row in 0..<m {
                for col in 0..<n {
                    var acc: Float = 0
                    for kk in 0..<k {
                        let xb = Self.bf16ToF32(Self.f32ToBF16(Float(xF16[row * k + kk])))
                        let wb = Self.bf16ToF32(wBF16[col * k + kk])
                        acc += xb * wb
                    }
                    let got = Float(yPtr[row * n + col])
                    let exp = Float(Float16(acc))
                    maxDiff = max(maxDiff, abs(got - exp))
                }
            }
            XCTAssertLessThan(maxDiff, 2e-2, "BF16 NAX f16->f16 M=\(m) N=\(n) K=\(k) maxDiff=\(maxDiff)")
        }
    }
}
