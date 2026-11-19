// QWMMHardeningTests.swift — Regression tests for Wave 2 QWMM safety:
// enum QWMMBlock, grid dim validation, smartResize bounds, QWX0Plan consumption,
// buildPatchSlab temporal==1, and qwInterleaveX0 underrun errors.
import XCTest
@testable import QWMM

final class QWMMHardeningTests: XCTestCase {
    func testVisionGridRejectsNonPositiveDims() {
        XCTAssertThrowsError(try QWVisionGrid(t: 0, h: 4, w: 4)) { e in
            XCTAssertEqual(e as? QWGridError, .mergeShape(t: 0, h: 4, w: 4))
        }
        XCTAssertThrowsError(try QWVisionGrid(t: 1, h: 0, w: 4)) { e in
            XCTAssertEqual(e as? QWGridError, .mergeShape(t: 1, h: 0, w: 4))
        }
        XCTAssertThrowsError(try QWVisionGrid(t: 1, h: 4, w: -2)) { e in
            XCTAssertEqual(e as? QWGridError, .mergeShape(t: 1, h: 4, w: -2))
        }
    }

    func testSmartResizeRejectsDegenerateDimsAndClampsExtremeAspectToBudget() throws {
        XCTAssertThrowsError(try QWMMConfig.smartResize(height: 0, width: 100)) { e in
            guard case .pixelDimensions = e as? QWGridError else { return XCTFail("wrong error \(e)") }
        }
        XCTAssertThrowsError(try QWMMConfig.smartResize(height: -10, width: -10)) { e in
            guard case .pixelDimensions = e as? QWGridError else { return XCTFail("wrong error \(e)") }
        }
        XCTAssertThrowsError(try QWMMConfig.smartResize(height: 100, width: 100, maxPixels: 500)) { e in
            guard case .pixelDimensions = e as? QWGridError else { return XCTFail("wrong error \(e)") }
        }

        // Aspect 198.8 < 200 with tight maxPixels must not exceed maxPixels or produce 0 height.
        let tight = try QWMMConfig.smartResize(height: 17, width: 3380, maxPixels: 100_000)
        XCTAssertGreaterThan(tight.h, 0)
        XCTAssertGreaterThan(tight.w, 0)
        XCTAssertEqual(tight.h % 32, 0)
        XCTAssertEqual(tight.w % 32, 0)
        XCTAssertLessThanOrEqual(tight.h * tight.w, 100_000)

        let zeroMinSpec = QWMMSpec(patch: 16, temporal: 2, merge: 2, posSide: 48,
                                   imageToken: 248056, videoToken: 248057, minPixels: 0, maxPixels: 1_048_576)
        let smallH = try QWMMConfig.smartResize(height: 1, width: 199, spec: zeroMinSpec)
        XCTAssertGreaterThan(smallH.h, 0)
        XCTAssertGreaterThan(smallH.w, 0)
    }

    func testBuildPatchSlabRequiresTemporalOne() throws {
        let gridT2 = try QWVisionGrid(t: 2, h: 2, w: 2)
        let rgb = [UInt8](repeating: 128, count: 32 * 32 * 3)
        XCTAssertThrowsError(try qwBuildPatchSlab(imageRGB: rgb, height: 32, width: 32, grid: gridT2)) { e in
            XCTAssertEqual(e as? QWGridError, .mergeShape(t: 2, h: 2, w: 2))
        }
    }

    func testMRopeEnumBlocksAndEmptyBlocks() throws {
        XCTAssertEqual(qwMRopePositions(blocks: []).pos, [])
        XCTAssertEqual(qwMRopePositions(blocks: []).delta, 0)

        let g = try QWVisionGrid(t: 1, h: 2, w: 2)
        let (pos, delta) = qwMRopePositions(blocks: [
            .text([1, 2]),
            .image(tokens: [248056], grid: g),
            .video(tokens: [248057], grid: g)
        ])
        XCTAssertEqual(pos.count, 3 * 4)
        XCTAssertEqual(delta, 0)
    }

    func testQWX0PlanRequiresFullImagesAndVideosConsumption() throws {
        let g = try QWVisionGrid(t: 1, h: 2, w: 2)
        XCTAssertThrowsError(try QWX0Plan(tokens: [1, 2, 3], images: [g], videos: [])) { e in
            XCTAssertEqual(e as? QWGridError, .placeholderOverflow)
        }
        XCTAssertThrowsError(try QWX0Plan(tokens: [1, QWMMConfig.imageToken, 2], images: [g, g], videos: [])) { e in
            XCTAssertEqual(e as? QWGridError, .placeholderOverflow)
        }
        XCTAssertThrowsError(try QWX0Plan(tokens: [1, 2], images: [], videos: [g])) { e in
            XCTAssertEqual(e as? QWGridError, .placeholderOverflow)
        }
    }

    func testInterleaveX0ThrowsBufferLengthOnUnderrun() throws {
        let g = try QWVisionGrid(t: 1, h: 2, w: 2)
        let plan = try QWX0Plan(tokens: [10, QWMMConfig.imageToken, 20], images: [g], videos: [])
        let hidden = 8

        XCTAssertThrowsError(try qwInterleaveX0(
            textEmbeddings: [Float](repeating: 1.0, count: hidden),
            visualTokens: [Float](repeating: 2.0, count: hidden),
            plan: plan,
            hiddenDim: hidden
        )) { e in
            XCTAssertEqual(e as? QWGridError, .bufferLength(expected: 2 * hidden, actual: hidden))
        }

        XCTAssertThrowsError(try qwInterleaveX0(
            textEmbeddings: [Float](repeating: 1.0, count: 2 * hidden),
            visualTokens: [Float](repeating: 2.0, count: hidden - 1),
            plan: plan,
            hiddenDim: hidden
        )) { e in
            XCTAssertEqual(e as? QWGridError, .bufferLength(expected: hidden, actual: hidden - 1))
        }
    }
}
