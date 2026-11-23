// QWNormAddBiasTests.swift — Metal vs CPU FP32 parity and guard tests for qwg_add_bias_f16.

import Metal
import XCTest
@testable import QWNorm
import QWAlloc
import QWCore
import QWHAL
import QWQuant

final class QWNormAddBiasTests: XCTestCase {
    private func f16(_ bits: UInt16) -> Float { QWQuant.f16tof32Bits(bits) }
    private func h16(_ val: Float) -> UInt16 { Float16(val).bitPattern }

    func testAddBiasF16MultiRowMultiBlockMatchesCPUReference() throws {
        let geometry = try QWNormGeometryTests.standard(hidden: 512)
        let hal = try QWHal(geometry: geometry)
        let rows = 5
        let cols = 512 // 64 uint4 elements = 2 threadgroups in X
        let total = rows * cols
        let arena = try QWArena(
            hal: hal, weights: cols * 4 + 4096,
            activations: total * 4 + 4096, kv: 0, gdnStates: 0)
        let xTensor = try arena.activationsRegion().allocate(QWTensorShape([total], .f16))
        let biasTensor = try arena.weightsRegion().allocate(QWTensorShape([cols], .f16))

        let xRaw = (0..<total).map { h16(Float(($0 % 37) - 18) * 0.05) }
        let bRaw = (0..<cols).map { h16(Float(($0 % 29) - 14) * 0.03) }
        arena.write(xTensor, from: xRaw)
        arena.write(biasTensor, from: bRaw)

        let (cb, enc) = try XCTUnwrap(hal.begin())
        try QWNorm.encodeAddBiasF16(enc, hal: hal, x: xTensor, bias: biasTensor, rows: rows, cols: cols)
        hal.end(cb, enc)
        cb.waitUntilCompleted()
        XCTAssertNil(cb.error)

        let got = arena.read(xTensor, as: UInt16.self, count: total)
        for r in 0..<rows {
            for c in 0..<cols {
                let idx = r * cols + c
                let want = Float(Float16(f16(xRaw[idx]) + f16(bRaw[c])))
                let diff = abs(f16(got[idx]) - want)
                XCTAssertLessThanOrEqual(diff, 1e-3, "row \(r) col \(c) mismatch")
            }
        }
    }

    func testAddBiasF16RejectsBadShapesAndAlignment() throws {
        let geometry = try QWNormGeometryTests.standard(hidden: 256)
        let hal = try QWHal(geometry: geometry)
        let arena = try QWArena(hal: hal, weights: 4096, activations: 8192, kv: 0, gdnStates: 0)
        let x = try arena.activationsRegion().allocate(QWTensorShape([2, 256], .f16))
        let bias = try arena.weightsRegion().allocate(QWTensorShape([256], .f16))
        let wrongBias = try arena.weightsRegion().allocate(QWTensorShape([128], .f16))
        let wrongX = try arena.activationsRegion().allocate(QWTensorShape([2, 256], .f32))
        let badSlice = bias.slice(byteOffset: 2, count: 256, dtype: .f16)

        func expectBadShape(_ block: (any MTLComputeCommandEncoder) throws -> Void) throws {
            let (_, enc) = try XCTUnwrap(hal.begin())
            defer { enc.endEncoding() }
            do {
                try block(enc)
                XCTFail("Expected QWNormError.badShape")
            } catch QWNormError.badShape {
                // expected
            }
        }

        try expectBadShape { try QWNorm.encodeAddBiasF16($0, hal: hal, x: x, bias: bias, rows: 2, cols: 252) }
        try expectBadShape { try QWNorm.encodeAddBiasF16($0, hal: hal, x: x, bias: wrongBias, rows: 2, cols: 256) }
        try expectBadShape { try QWNorm.encodeAddBiasF16($0, hal: hal, x: wrongX, bias: bias, rows: 2, cols: 256) }
        try expectBadShape { try QWNorm.encodeAddBiasF16($0, hal: hal, x: x, bias: badSlice, rows: 2, cols: 256) }
        _ = try arena.weightsRegion().allocate(QWTensorShape([1], .f16))
    }
}
