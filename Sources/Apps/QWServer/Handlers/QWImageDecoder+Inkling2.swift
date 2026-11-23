// QWImageDecoder+Inkling2.swift — Inkling-2 checkerboard RGBA composite & 40x40 -1 pad preprocessing.
import CoreGraphics
import Foundation
import ImageIO
import QWMM

extension QWImageDecoder {
    public static func prepareInkling2Patches(
        urlOrDataUri: String,
        maxPixels: Int = QWImageDecoder.liveMaxPixels,
        policy: QWImageDecodePolicy = .serverDefault,
        spec: QWMMSpec
    ) throws -> (patches: [Float], grid: QWVisionGrid) {
        let data = try resolveSourceData(urlOrDataUri: urlOrDataUri, policy: policy)
        return try prepareInkling2PatchesData(data, maxPixels: maxPixels, policy: policy, spec: spec)
    }

    static func prepareInkling2PatchesAsync(
        urlOrDataUri: String,
        maxPixels: Int = QWImageDecoder.liveMaxPixels,
        policy: QWImageDecodePolicy = .serverDefault,
        spec: QWMMSpec
    ) async throws -> (patches: [Float], grid: QWVisionGrid) {
        let data = try await resolveSourceDataAsync(urlOrDataUri: urlOrDataUri, policy: policy)
        return try await runOffCooperativePool {
            try prepareInkling2PatchesData(data, maxPixels: maxPixels, policy: policy, spec: spec)
        }
    }

    public static func prepareInkling2PatchesData(
        _ data: Data,
        maxPixels: Int,
        policy: QWImageDecodePolicy = .serverDefault,
        spec: QWMMSpec
    ) throws -> (patches: [Float], grid: QWVisionGrid) {
        let (source, orientedW, orientedH, orientation) = try inspectSource(data: data, policy: policy)
        var targetW = orientedW
        var targetH = orientedH
        if targetW * targetH > maxPixels {
            let scale = (Double(maxPixels) / Double(targetW * targetH)).squareRoot()
            targetW = max(1, Int(Double(targetW) * scale))
            targetH = max(1, Int(Double(targetH) * scale))
        }

        let cgImage: CGImage
        let renderOrientation: UInt32
        if targetW < orientedW || targetH < orientedH {
            let thumbOpts: [CFString: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceShouldCacheImmediately: false,
                kCGImageSourceThumbnailMaxPixelSize: max(targetW, targetH),
            ]
            guard let thumb = CGImageSourceCreateThumbnailAtIndex(source, 0, thumbOpts as CFDictionary) else {
                throw QWImageDecoderError.imageDecodeFailed("CGImageSourceCreateThumbnailAtIndex failed")
            }
            cgImage = thumb
            targetW = thumb.width
            targetH = thumb.height
            renderOrientation = 1
        } else {
            guard let full = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                throw QWImageDecoderError.imageDecodeFailed("CGImageSourceCreateImageAtIndex failed")
            }
            cgImage = full
            renderOrientation = orientation
        }

        let rgb = try renderCheckerboardRGB(
            cgImage: cgImage, targetWidth: targetW, targetHeight: targetH, orientation: renderOrientation
        )
        return try normalizePadAndPatchifyInkling2(rgb: rgb, height: targetH, width: targetW, spec: spec)
    }

    /// Checkerboard composite `(y/8 + x/8) % 2 == 0 ? 204 : 153`, `-1` edge pad to factor `40`, then normalize.
    public static func normalizePadAndPatchifyInkling2(
        rgb: [UInt8],
        height: Int,
        width: Int,
        spec: QWMMSpec
    ) throws -> (patches: [Float], grid: QWVisionGrid) {
        let factor = max(1, spec.factor)
        let rows = max(1, (height + factor - 1) / factor)
        let cols = max(1, (width + factor - 1) / factor)
        let paddedH = rows * factor
        let paddedW = cols * factor
        let rescale: Float = 1.0 / 255.0
        let padNorm: Float = (-1.0 * rescale - 0.5) / 0.5
        var norm = [Float](repeating: padNorm, count: paddedH * paddedW * 3)
        for y in 0..<height {
            let srcRow = y * width * 3
            let dstRow = y * paddedW * 3
            for i in 0..<(width * 3) {
                norm[dstRow + i] = (Float(rgb[srcRow + i]) * rescale - 0.5) / 0.5
            }
        }
        let grid = try QWVisionGrid(t: 1, h: rows * spec.merge, w: cols * spec.merge, spec: spec)
        let patches = try qwBuildInkling2PatchSlab(
            normalizedPaddedRGB: norm, paddedH: paddedH, paddedW: paddedW, grid: grid
        )
        return (patches, grid)
    }

    private static func renderCheckerboardRGB(
        cgImage: CGImage,
        targetWidth: Int,
        targetHeight: Int,
        orientation: UInt32
    ) throws -> [UInt8] {
        guard targetWidth > 0, targetHeight > 0 else {
            throw QWImageDecoderError.emptyImage
        }
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw QWImageDecoderError.imageDecodeFailed("Failed to create sRGB colorspace")
        }
        var rgba = [UInt8](repeating: 0, count: targetWidth * targetHeight * 4)
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        let drew = rgba.withUnsafeMutableBytes { (ptr: UnsafeMutableRawBufferPointer) -> Bool in
            guard let context = CGContext(
                data: ptr.baseAddress, width: targetWidth, height: targetHeight,
                bitsPerComponent: 8, bytesPerRow: targetWidth * 4,
                space: colorSpace, bitmapInfo: bitmapInfo
            ) else { return false }
            let swapsAxes = (5...8).contains(orientation)
            let drawW = swapsAxes ? targetHeight : targetWidth
            let drawH = swapsAxes ? targetWidth : targetHeight
            let w = CGFloat(targetWidth), h = CGFloat(targetHeight)
            let transform: CGAffineTransform
            switch orientation {
            case 2: transform = CGAffineTransform(a: -1, b: 0, c: 0, d: 1, tx: w, ty: 0)
            case 3: transform = CGAffineTransform(a: -1, b: 0, c: 0, d: -1, tx: w, ty: h)
            case 4: transform = CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: h)
            case 5: transform = CGAffineTransform(a: 0, b: -1, c: -1, d: 0, tx: w, ty: h)
            case 6: transform = CGAffineTransform(a: 0, b: -1, c: 1, d: 0, tx: 0, ty: h)
            case 7: transform = CGAffineTransform(a: 0, b: 1, c: 1, d: 0, tx: 0, ty: 0)
            case 8: transform = CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: w, ty: 0)
            default: transform = .identity
            }
            context.concatenate(transform)
            context.interpolationQuality = .high
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: drawW, height: drawH))
            return true
        }
        guard drew else {
            throw QWImageDecoderError.imageDecodeFailed("CGContext creation failed")
        }
        var rgb = [UInt8](repeating: 0, count: targetWidth * targetHeight * 3)
        for y in 0..<targetHeight {
            for x in 0..<targetWidth {
                let s = (y * targetWidth + x) * 4
                let d = (y * targetWidth + x) * 3
                let alpha = Int(rgba[s + 3])
                if alpha == 255 {
                    rgb[d] = rgba[s]; rgb[d + 1] = rgba[s + 1]; rgb[d + 2] = rgba[s + 2]
                } else {
                    let bg = ((y / 8 + x / 8) % 2 == 0) ? 204 : 153
                    let bgContrib = (bg * (255 - alpha) + 127) / 255
                    rgb[d] = UInt8(min(255, Int(rgba[s]) + bgContrib))
                    rgb[d + 1] = UInt8(min(255, Int(rgba[s + 1]) + bgContrib))
                    rgb[d + 2] = UInt8(min(255, Int(rgba[s + 2]) + bgContrib))
                }
            }
        }
        return rgb
    }
}
