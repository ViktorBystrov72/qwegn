// QWVideoPatchSlabTests.swift — Unit tests for multi-frame temporal patch slabs,
// video smart-resize budgeting, X0 interleaved video+image ordering, and 3D MRoPE.

import Foundation
import QWCore
@testable import QWMM
import QWToken
import QWVision
import XCTest

final class QWVideoPatchSlabTests: XCTestCase {
    func testVideoSmartResizePadsTubeletsAndRespectsTotalPixelBudget() throws {
        let spec = QWMMSpec.qwen3VLReference // patch=14, temporal=2, merge=2, factor=28
        let resOdd = try qwVideoSmartResize(
            height: 720,
            width: 1280,
            numFrames: 5,
            spec: spec,
            maxVideoTotalPixels: 3 * 224 * 224
        )
        XCTAssertEqual(resOdd.paddedFrames, 6)
        XCTAssertEqual(resOdd.t, 3)
        XCTAssertEqual(resOdd.pixelH % spec.factor, 0)
        XCTAssertEqual(resOdd.pixelW % spec.factor, 0)
        XCTAssertLessThanOrEqual(resOdd.t * resOdd.pixelH * resOdd.pixelW, 3 * 224 * 224)
    }

    func testSingleTubeletIdenticalFramesBitExactWithImagePatchSlab() throws {
        let spec = QWMMSpec.qwen3VLReference
        let grid = try QWVisionGrid(t: 1, h: 4, w: 4, spec: spec)
        let pixelH = grid.h * spec.patch
        let pixelW = grid.w * spec.patch
        var rgb = [UInt8](repeating: 0, count: pixelH * pixelW * 3)
        for idx in rgb.indices {
            rgb[idx] = UInt8((idx * 37 + 19) & 0xFF)
        }
        let imageSlab = try qwBuildPatchSlab(imageRGB: rgb, height: pixelH, width: pixelW, grid: grid)
        let videoSlab = try qwBuildVideoPatchSlab(
            framesRGB: [rgb, rgb], height: pixelH, width: pixelW, grid: grid
        )
        XCTAssertEqual(imageSlab.count, videoSlab.count)
        for i in imageSlab.indices {
            XCTAssertEqual(imageSlab[i].bitPattern, videoSlab[i].bitPattern, "Mismatch at index \(i)")
        }
    }

    func testMultiTubeletTemporalFramesPreserveDistinctSubTemporalSlices() throws {
        let spec = QWMMSpec(
            patch: 2, temporal: 2, merge: 2, posSide: 4,
            imageToken: 100, videoToken: 101,
            imageMean: [0, 0, 0], imageStd: [1, 1, 1]
        )
        let grid = try QWVisionGrid(t: 2, h: 2, w: 2, spec: spec)
        let pixelH = 4
        let pixelW = 4
        let count = pixelH * pixelW * 3
        let f0 = [UInt8](repeating: 10, count: count)
        let f1 = [UInt8](repeating: 60, count: count)
        let f2 = [UInt8](repeating: 120, count: count)
        let f3 = [UInt8](repeating: 240, count: count)

        let slab = try qwBuildVideoPatchSlab(
            framesRGB: [f0, f1, f2, f3], height: pixelH, width: pixelW, grid: grid
        )
        let patchArea = spec.patch * spec.patch // 4
        let temporalArea = spec.temporal * patchArea // 8
        let patchChannels = 3 * temporalArea // 24
        XCTAssertEqual(slab.count, grid.patches * patchChannels)

        // Tubelet 0, patch 0, channel 0: subT=0 should be 10/255, subT=1 should be 60/255.
        XCTAssertEqual(slab[0], 10.0 / 255.0, accuracy: 1e-6)
        XCTAssertEqual(slab[patchArea], 60.0 / 255.0, accuracy: 1e-6)

        // Tubelet 1 (`row = 4`), patch 0, channel 0: subT=0 should be 120/255, subT=1 should be 240/255.
        let tubelet1Base = 4 * patchChannels
        XCTAssertEqual(slab[tubelet1Base], 120.0 / 255.0, accuracy: 1e-6)
        XCTAssertEqual(slab[tubelet1Base + patchArea], 240.0 / 255.0, accuracy: 1e-6)
    }

    func testX0PlanMixedVideoBeforeImageTracksOrderAnd3DMRoPE() throws {
        let spec = QWMMSpec(
            patch: 14, temporal: 2, merge: 2, posSide: 48,
            imageToken: 248056, videoToken: 248057,
            mropeLayout: .qwenInterleaved
        )
        let vidGrid = try QWVisionGrid(t: 3, h: 4, w: 4, spec: spec) // 3 * 2 * 2 = 12 tokens
        let imgGrid = try QWVisionGrid(t: 1, h: 2, w: 4, spec: spec) // 1 * 1 * 2 = 2 tokens
        let rawTokens = [10, 248057, 20, 248056, 30]
        let plan = try QWX0Plan(tokens: rawTokens, images: [imgGrid], videos: [vidGrid], spec: spec)

        XCTAssertEqual(plan.visionSpanIsVideo, [true, false])
        XCTAssertEqual(plan.tokens.count, 1 + 12 + 1 + 2 + 1)

        // Verify Qwen3-VL temporal coordinates across the 3 video tubelets (`max(4,4)/2 = 2` step per tubelet).
        let n = plan.tokens.count
        XCTAssertEqual(plan.positions[1], 1)     // tubelet 0: t = 1
        XCTAssertEqual(plan.positions[1 + 4], 3) // tubelet 1: t = 3
        XCTAssertEqual(plan.positions[1 + 8], 5) // tubelet 2: t = 5

        // text[0] (cur 1) + 3 video tubelets * 2 (cur 7) + text[1] (cur 8) + image max(2,4)/2=2 (cur 10) + text[2] (cur 11):
        // delta = 11 - 17 = -6.
        XCTAssertEqual(plan.mropeDelta, -6)
        XCTAssertEqual(plan.positions[n - 1], 10)
        XCTAssertEqual(plan.positions[2 * n - 1], 10)
        XCTAssertEqual(plan.positions[3 * n - 1], 10)
    }
}
