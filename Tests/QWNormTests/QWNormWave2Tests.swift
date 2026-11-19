// QWNormWave2Tests.swift — Tests shape/dtype/row guards in QWNorm.encode.
import Metal
import XCTest
@testable import QWNorm
import QWCore
import QWHAL
import QWAlloc

final class QWNormWave2Tests: XCTestCase {
    func testEncodeRejectsZeroRowsMismatchedOutAndWrongDTypes() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true", "Requires Metal HAL")
        let geometry = QWModelGeometry.qwen38_27b
        let hal = try QWHal(geometry: geometry)
        let hidden = geometry.hidden
        let arena = try QWArena(hal: hal, weights: 0, activations: hidden * 16 + 4096, kv: 0, gdnStates: 0)
        let a = arena.activationsRegion()
        let xF16 = try a.allocate(QWTensorShape([hidden], .f16))
        let gammaF16 = try a.allocate(QWTensorShape([hidden], .f16))
        let outF16 = try a.allocate(QWTensorShape([hidden], .f16))
        let shortOut = try a.allocate(QWTensorShape([hidden / 2], .f16))
        let zeroX = QWTensorValue(region: xF16.region, offset: xF16.offset, shape: QWTensorShape([0], .f16))
        let xF32 = try a.allocate(QWTensorShape([hidden], .f32))
        let scalesF16 = try a.allocate(QWTensorShape([1], .f16))

        func expectBadShape(_ block: (any MTLComputeCommandEncoder) throws -> Void) throws {
            let (cb, enc) = try XCTUnwrap(hal.begin())
            do {
                try block(enc)
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                XCTFail("Expected QWNormError.badShape")
            } catch let error as QWNormError {
                enc.endEncoding()
                guard case .badShape = error else {
                    XCTFail("Unexpected QWNormError: \(error)")
                    return
                }
            } catch {
                enc.endEncoding()
                XCTFail("Unexpected error: \(error)")
            }
        }

        try expectBadShape { enc in
            try QWNorm.encode(enc, hal: hal, variant: .rmsnorm, x: zeroX, gamma: gammaF16, out: outF16, geometry: geometry)
        }
        try expectBadShape { enc in
            try QWNorm.encode(enc, hal: hal, variant: .rmsnorm, x: xF16, gamma: gammaF16, out: shortOut, geometry: geometry)
        }
        try expectBadShape { enc in
            try QWNorm.encode(enc, hal: hal, variant: .rmsnorm, x: xF32, gamma: gammaF16, out: outF16, geometry: geometry)
        }
        try expectBadShape { enc in
            try QWNorm.encode(
                enc, hal: hal, variant: .rmsnormPrologueScale, x: xF16, gamma: gammaF16,
                out: outF16, rowScalesOut: scalesF16, geometry: geometry
            )
        }
    }
}
