// QWRouterNaxTests.swift — Parity and bounds tests for F32/BF16 qwg_router_nax_f32 and qwg_router_nax.
import Foundation
import Metal
import XCTest
@testable import QWGEMM
import QWAlloc
import QWCore
import QWHAL
import QWQuant

final class QWRouterNaxTests: XCTestCase {
    private func makeHal(basis: QWWeightBasis = .rotated) throws -> QWHal {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("no Metal device") }
        return try QWHal(geometry: .qwen38_27b, weightBasis: basis)
    }

    func testRouterNaxF32AndF16InputParityAndSharedGate() throws {
        let hal = try makeHal()
        try XCTSkipUnless(hal.caps.isMetal4TensorCapable, "qwg_router_nax_f32 requires Metal 4")

        for rows in [64, 100] {
            let n = 128, k = 256
            let wF32 = (0..<(n * k)).map { sin(Float($0) * 0.017) * 0.2 }
            let wBF16 = wF32.map(QWBF16.fromFloatRNE)
            let xF32 = (0..<(rows * k)).map { cos(Float($0) * 0.011) * 0.5 }
            let xF16 = xF32.map { Float(Float16($0)) }
            let shW = (0..<k).map { sin(Float($0) * 0.031) * 0.1 }

            let arena = try QWArena(hal: hal, weights: n * k * 8 + 4096,
                                    activations: rows * (n + k) * 16 + 4096, kv: 0, gdnStates: 0)
            let wr = arena.weightsRegion(), ar = arena.activationsRegion()
            let wF32T = try wr.allocate(QWTensorShape([n, k], .f32))
            let wBF16T = try wr.allocate(QWTensorShape([n, k], .bf16))
            let shWT = try wr.allocate(QWTensorShape([k], .f32))
            arena.write(wF32T, from: wF32)
            arena.write(wBF16T, from: wBF16)
            arena.write(shWT, from: shW)

            let xF32T = try ar.allocate(QWTensorShape([rows, k], .f32))
            let xF16T = try ar.allocate(QWTensorShape([rows, k], .f16))
            arena.write(xF32T, from: xF32)
            arena.write(xF16T, from: xF16.map { Float16($0) })

            let yF32F32 = try ar.allocate(QWTensorShape([rows, n], .f32))
            let yF32F16 = try ar.allocate(QWTensorShape([rows, n], .f32))
            let yBF16F32 = try ar.allocate(QWTensorShape([rows, n], .f32))
            let yBF16F16 = try ar.allocate(QWTensorShape([rows, n], .f32))
            let gateF32 = try ar.allocate(QWTensorShape([rows], .f32))
            let gateF16 = try ar.allocate(QWTensorShape([rows], .f32))

            guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
            try QWGemm.encodeFnRouterGemm(enc, hal: hal, weight: wF32T, x: xF32T, y: yF32F32,
                                          rows: rows, n: n, k: k, shGateWeight: shWT, shGateOut: gateF32)
            try QWGemm.encodeFnRouterGemm(enc, hal: hal, weight: wF32T, x: xF16T, y: yF32F16,
                                          rows: rows, n: n, k: k, shGateWeight: shWT, shGateOut: gateF16)
            try QWGemm.encodeFnRouterGemm(enc, hal: hal, weight: wBF16T, x: xF32T, y: yBF16F32,
                                          rows: rows, n: n, k: k)
            try QWGemm.encodeFnRouterGemm(enc, hal: hal, weight: wBF16T, x: xF16T, y: yBF16F16,
                                          rows: rows, n: n, k: k)
            hal.end(cb, enc)
            cb.waitUntilCompleted()
            XCTAssertNil(cb.error)

            let gotF32F32 = arena.read(yF32F32, as: Float.self, count: rows * n)
            let gotF32F16 = arena.read(yF32F16, as: Float.self, count: rows * n)
            let gotBF16F32 = arena.read(yBF16F32, as: Float.self, count: rows * n)
            let gotBF16F16 = arena.read(yBF16F16, as: Float.self, count: rows * n)
            let gotGateF32 = arena.read(gateF32, as: Float.self, count: rows)
            let gotGateF16 = arena.read(gateF16, as: Float.self, count: rows)

            for r in 0..<rows {
                var shRefF32: Float = 0
                var shRefF16: Float = 0
                for c in 0..<k {
                    shRefF32 += shW[c] * xF32[r * k + c]
                    shRefF16 += shW[c] * xF16[r * k + c]
                }
                XCTAssertEqual(gotGateF32[r], 1.0 / (1.0 + exp(-shRefF32)), accuracy: 1e-4)
                XCTAssertEqual(gotGateF16[r], 1.0 / (1.0 + exp(-shRefF16)), accuracy: 1e-4)
                for col in 0..<n {
                    var refF32: Float = 0
                    var refF16: Float = 0
                    var refBF16F32: Float = 0
                    var refBF16F16: Float = 0
                    for c in 0..<k {
                        let wq = Float(bitPattern: UInt32(wBF16[col * k + c]) << 16)
                        refF32 += wF32[col * k + c] * xF32[r * k + c]
                        refF16 += wF32[col * k + c] * xF16[r * k + c]
                        refBF16F32 += wq * xF32[r * k + c]
                        refBF16F16 += wq * xF16[r * k + c]
                    }
                    XCTAssertEqual(gotF32F32[r * n + col], refF32, accuracy: 2e-4, "f32/f32 (\(r),\(col))")
                    XCTAssertEqual(gotF32F16[r * n + col], refF16, accuracy: 2e-4, "f32/f16 (\(r),\(col))")
                    XCTAssertEqual(gotBF16F32[r * n + col], refBF16F32, accuracy: 2e-4, "bf16/f32 (\(r),\(col))")
                    XCTAssertEqual(gotBF16F16[r * n + col], refBF16F16, accuracy: 2e-4, "bf16/f16 (\(r),\(col))")
                }
            }
        }
    }

    func testRouterNaxF32LargeKAccumulationPrecision() throws {
        let hal = try makeHal()
        try XCTSkipUnless(hal.caps.isMetal4TensorCapable, "qwg_router_nax_f32 requires Metal 4")

        let rows = 64, n = 64, k = 4096
        let wF32 = (0..<(n * k)).map { sin(Float($0) * 0.0073) * 0.1 }
        let xF32 = (0..<(rows * k)).map { cos(Float($0) * 0.0041) * 0.25 }

        let arena = try QWArena(hal: hal, weights: n * k * 4 + 4096,
                                activations: rows * (n + k) * 4 + 4096, kv: 0, gdnStates: 0)
        let wT = try arena.weightsRegion().allocate(QWTensorShape([n, k], .f32))
        arena.write(wT, from: wF32)
        let xT = try arena.activationsRegion().allocate(QWTensorShape([rows, k], .f32))
        arena.write(xT, from: xF32)
        let yT = try arena.activationsRegion().allocate(QWTensorShape([rows, n], .f32))

        guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
        try QWGemm.encodeFnRouterGemm(enc, hal: hal, weight: wT, x: xT, y: yT, rows: rows, n: n, k: k)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let got = arena.read(yT, as: Float.self, count: rows * n)
        for r in [0, 31, 63] {
            for col in [0, 31, 63] {
                var ref: Double = 0
                for c in 0..<k { ref += Double(wF32[col * k + c]) * Double(xF32[r * k + c]) }
                XCTAssertEqual(got[r * n + col], Float(ref), accuracy: 2e-3, "largeK (\(r),\(col))")
            }
        }
    }

    func testRouterNaxF32ColumnBoundsGuardOnNonMultipleOf64() throws {
        let hal = try makeHal()
        try XCTSkipUnless(hal.caps.isMetal4TensorCapable, "qwg_router_nax_f32 requires Metal 4")

        let rows = 64, n = 72, paddedN = 128, k = 64
        let arena = try QWArena(hal: hal, weights: n * k * 4 + 4096,
                                activations: (rows * (k + paddedN)) * 4 + 4096, kv: 0, gdnStates: 0)
        let wVals = (0..<(n * k)).map { Float(($0 * 11) % 23 - 11) * 0.05 }
        let xVals = (0..<(rows * k)).map { Float(($0 * 5) % 17 - 8) * 0.1 }
        let wT = try arena.weightsRegion().allocate(QWTensorShape([n, k], .f32))
        arena.write(wT, from: wVals)
        let xT = try arena.activationsRegion().allocate(QWTensorShape([rows, k], .f32))
        arena.write(xT, from: xVals)
        let yT = try arena.activationsRegion().allocate(QWTensorShape([rows * paddedN], .f32))
        let sentinel: Float = -9999.0
        arena.write(yT, from: [Float](repeating: sentinel, count: rows * paddedN))

        let pso = try hal.pipeline("qwg_router_nax_f32")
        guard let (cb, enc) = hal.begin() else { throw XCTSkip("no command buffer") }
        enc.setComputePipelineState(pso)
        enc.bindTensors(wT, xT, yT, startIndex: 0)
        enc.setUInt32s(rows, n, k, startIndex: 3)
        wT.bind(enc, index: 7)
        yT.bind(enc, index: 8)
        enc.setUInt32s(0, startIndex: 9)
        enc.dispatchThreadgroups(MTLSize(width: 2, height: (rows + 63) / 64, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let got = arena.read(yT, as: Float.self, count: rows * n)
        for row in 0..<rows {
            for col in 0..<n {
                var ref: Float = 0
                for c in 0..<k { ref += wVals[col * k + c] * xVals[row * k + c] }
                XCTAssertEqual(got[row * n + col], ref, accuracy: 2e-4, "row=\(row) col=\(col)")
            }
        }
        let tail = arena.read(yT.slice(byteOffset: rows * n * 4, count: rows * (paddedN - n), dtype: .f32),
                              as: Float.self, count: rows * (paddedN - n))
        XCTAssertEqual(tail.first, sentinel)
        XCTAssertEqual(tail.last, sentinel)
    }
}
