// QWMM+Inkling2.swift — Inkling-2 align_corners=False pos_embed, (x+1,y+1) RoPE & HWC patchify.
import Foundation
import QWVision

/// PyTorch/MLX `align_corners=False` 1D bilinear sampling taps into `0..<side`.
func axisTapsAlignCornersFalse(_ index: Int, size: Int, side: Int) -> (taps: [Int], weights: [Float]) {
    guard side > 0, size > 0 else { return ([0, 0], [1, 0]) }
    let raw = (Float(index) + 0.5) * Float(side) / Float(size) - 0.5
    let clamped = max(raw, 0)
    let lo = min(Int(clamped.rounded(.down)), side - 1)
    let hi = min(lo + 1, side - 1)
    let frac = clamped - Float(lo)
    return ([lo, hi], [1 - frac, frac])
}

/// Inkling-2 bilinear interpolation table (`align_corners=False`) in merge-block order.
public func qwInkling2InterpTaps(grid: QWVisionGrid) -> (indices: [Int32], weights: [Float]) {
    let m = grid.spec.merge
    let side = grid.spec.posSide
    let blocksW = grid.w / m
    var indices: [Int32] = []
    var weights: [Float] = []
    indices.reserveCapacity(grid.patches * 4)
    weights.reserveCapacity(grid.patches * 4)
    for _ in 0..<grid.t {
        for within in 0..<(grid.h * grid.w) {
            let mc = within % m
            let mr = (within / m) % m
            let blockCol = (within / (m * m)) % blocksW
            let blockRow = within / (m * m * blocksW)
            let row = blockRow * m + mr
            let col = blockCol * m + mc
            let (ht, hw) = axisTapsAlignCornersFalse(row, size: grid.h, side: side)
            let (wt, ww) = axisTapsAlignCornersFalse(col, size: grid.w, side: side)
            for hh in 0..<2 {
                for wc in 0..<2 {
                    indices.append(Int32(ht[hh] * side + wt[wc]))
                    weights.append(hw[hh] * ww[wc])
                }
            }
        }
    }
    return (indices, weights)
}

/// Inkling-2 2D RoPE coordinates `(col + 1, row + 1)` in merge-block order (CLS `(0,0)` added by ViT).
public func qwInkling2RopePositions(grid: QWVisionGrid) -> [Int32] {
    let m = grid.spec.merge
    var out: [Int32] = []
    out.reserveCapacity(grid.patches * 2)
    for _ in 0..<grid.t {
        for hb in 0..<(grid.h / m) {
            for wb in 0..<(grid.w / m) {
                for i in 0..<m {
                    for j in 0..<m {
                        out.append(Int32(wb * m + j + 1))
                        out.append(Int32(hb * m + i + 1))
                    }
                }
            }
        }
    }
    return out
}

/// Builds Inkling-2 `[nPatch, patch * patch * 3]` HWC subpatch slab from normalized `[paddedH, paddedW, 3]` RGB.
public func qwBuildInkling2PatchSlab(
    normalizedPaddedRGB: [Float],
    paddedH: Int,
    paddedW: Int,
    grid: QWVisionGrid
) throws -> [Float] {
    let spec = grid.spec
    let expectedH = grid.h * spec.patch
    let expectedW = grid.w * spec.patch
    guard grid.t == 1, paddedH == expectedH, paddedW == expectedW else {
        throw QWGridError.pixelDimensions(
            expectedH: expectedH, expectedW: expectedW, actualH: paddedH, actualW: paddedW
        )
    }
    let expectedCount = paddedH * paddedW * 3
    guard normalizedPaddedRGB.count == expectedCount else {
        throw QWGridError.bufferLength(expected: expectedCount, actual: normalizedPaddedRGB.count)
    }
    let m = spec.merge
    let patch = spec.patch
    let rows = grid.h / m
    let cols = grid.w / m
    let rowFloats = patch * 3
    let patchFloats = patch * rowFloats
    let rowBytes = rowFloats * MemoryLayout<Float>.stride
    var out = [Float](repeating: 0, count: grid.patches * patchFloats)

    normalizedPaddedRGB.withUnsafeBufferPointer { srcBuf in
        out.withUnsafeMutableBufferPointer { dstBuf in
            guard let srcBase = srcBuf.baseAddress, let dstBase = dstBuf.baseAddress else { return }
            var pi = 0
            for blockRow in 0..<rows {
                for blockCol in 0..<cols {
                    for mr in 0..<m {
                        for mc in 0..<m {
                            let y0 = (blockRow * m + mr) * patch
                            let x0 = (blockCol * m + mc) * patch
                            let dstPatch = dstBase + pi * patchFloats
                            for pr in 0..<patch {
                                let srcOff = ((y0 + pr) * paddedW + x0) * 3
                                memcpy(dstPatch + pr * rowFloats, srcBase + srcOff, rowBytes)
                            }
                            pi += 1
                        }
                    }
                }
            }
        }
    }
    return out
}
