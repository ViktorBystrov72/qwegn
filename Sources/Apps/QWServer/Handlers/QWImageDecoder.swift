import Foundation
import CoreGraphics
import CryptoKit
import ImageIO
import QWCore
import QWEngine
import QWMM
import QWVision

public enum QWImageDecoderError: Error, CustomStringConvertible, Equatable {
    case invalidDataUrl(String)
    case disallowedImageSource(String)
    case imageTooLarge(String)
    case base64DecodeFailed
    case fileNotFound(String)
    case imageDecodeFailed(String)
    case emptyImage

    public var description: String {
        switch self {
        case .invalidDataUrl(let str):
            return "QWImageDecoder: invalid data URI scheme in '\(str.prefix(60))...'"
        case .disallowedImageSource(let reason):
            return "QWImageDecoder: disallowed image source: \(reason)"
        case .imageTooLarge(let reason):
            return "QWImageDecoder: image too large: \(reason)"
        case .base64DecodeFailed:
            return "QWImageDecoder: failed to decode base64 image data"
        case .fileNotFound(let path):
            return "QWImageDecoder: image file not found at '\(path)'"
        case .imageDecodeFailed(let reason):
            return "QWImageDecoder: failed to decode image: \(reason)"
        case .emptyImage:
            return "QWImageDecoder: image has 0 width or height"
        }
    }
}

public enum QWImageDecoder {
    public struct DecodedImage: Sendable {
        public let rgb: [UInt8]
        public let width: Int
        public let height: Int

        public init(rgb: [UInt8], width: Int, height: Int) {
            self.rgb = rgb
            self.width = width
            self.height = height
        }
    }

    /// Resolves and decodes an image source subject to `policy` (server HTTP default allows `data:image/*;base64` only).
    public static func decode(
        urlOrDataUri: String,
        policy: QWImageDecodePolicy = .serverDefault
    ) throws -> DecodedImage {
        let data = try resolveSourceData(urlOrDataUri: urlOrDataUri, policy: policy)
        return try decode(data: data, policy: policy)
    }

    private static func resolveSourceData(urlOrDataUri: String, policy: QWImageDecodePolicy) throws -> Data {
        let trimmed = urlOrDataUri.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = trimmed.lowercased()

        if lower.hasPrefix("data:") {
            guard policy.allowDataURLs else {
                throw QWImageDecoderError.disallowedImageSource("data URIs are disabled by policy")
            }
            guard let commaIdx = trimmed.firstIndex(of: ",") else {
                throw QWImageDecoderError.invalidDataUrl(trimmed)
            }
            let header = lower[..<commaIdx]
            guard header.hasPrefix("data:image/"), header.contains(";base64") else {
                throw QWImageDecoderError.disallowedImageSource("Only base64 data:image/* URIs are allowed")
            }
            let base64Part = String(trimmed[trimmed.index(after: commaIdx)...])
            guard base64Part.utf8.count <= (policy.maxSourceBytes / 3 + 2) * 4 else {
                throw QWImageDecoderError.imageTooLarge("Base64 payload exceeds \(policy.maxSourceBytes) bytes")
            }
            guard let decoded = Data(base64Encoded: base64Part, options: .ignoreUnknownCharacters) else {
                throw QWImageDecoderError.base64DecodeFailed
            }
            guard decoded.count <= policy.maxSourceBytes else {
                throw QWImageDecoderError.imageTooLarge("Decoded image exceeds \(policy.maxSourceBytes) bytes")
            }
            return decoded
        }

        if lower.hasPrefix("http://") || lower.hasPrefix("https://") {
            guard policy.allowRemoteImageURLs else {
                throw QWImageDecoderError.disallowedImageSource("Remote HTTP(S) image URLs are disabled by policy")
            }
            guard let url = URL(string: trimmed) else {
                throw QWImageDecoderError.invalidDataUrl(trimmed)
            }
            return try QWImageDecodePolicy.fetchBoundedRemoteData(url: url, maxBytes: policy.maxSourceBytes)
        }

        if lower.hasPrefix("file://") || trimmed.hasPrefix("/") || trimmed.hasPrefix("~") || trimmed.hasPrefix("./") || trimmed.hasPrefix("../") {
            guard policy.allowLocalImagePaths else {
                throw QWImageDecoderError.disallowedImageSource("Local filesystem image paths are disabled by policy")
            }
            let path = lower.hasPrefix("file://")
                ? (URL(string: trimmed)?.path ?? "")
                : trimmed.expandingTildePath
            return try readBoundedLocalFile(atPath: path, displayPath: trimmed, maxBytes: policy.maxSourceBytes)
        }

        if policy.allowLocalImagePaths, FileManager.default.fileExists(atPath: trimmed.expandingTildePath) {
            return try readBoundedLocalFile(atPath: trimmed.expandingTildePath, displayPath: trimmed, maxBytes: policy.maxSourceBytes)
        }

        if policy.allowDataURLs,
           trimmed.utf8.count <= (policy.maxSourceBytes / 3 + 2) * 4,
           let decoded = Data(base64Encoded: trimmed, options: .ignoreUnknownCharacters),
           !decoded.isEmpty {
            guard decoded.count <= policy.maxSourceBytes else {
                throw QWImageDecoderError.imageTooLarge("Decoded image exceeds \(policy.maxSourceBytes) bytes")
            }
            return decoded
        }

        throw QWImageDecoderError.disallowedImageSource("Rejected image URL '\(trimmed.prefix(48))'")
    }

    private static func readBoundedLocalFile(atPath path: String, displayPath: String, maxBytes: Int) throws -> Data {
        guard !path.isEmpty,
              FileManager.default.fileExists(atPath: path),
              let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let fileSize = attrs[.size] as? NSNumber else {
            throw QWImageDecoderError.fileNotFound(path.isEmpty ? displayPath : path)
        }
        guard fileSize.int64Value <= Int64(maxBytes) else {
            throw QWImageDecoderError.imageTooLarge("Local image file exceeds \(maxBytes) bytes")
        }
        guard let fileData = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
            throw QWImageDecoderError.fileNotFound(path)
        }
        return fileData
    }

    /// Default live pixel budget (1024×1024 = 1024 patches = 256 visual tokens); overridable via QW_VISION_MAX_PIXELS.
    public static var liveMaxPixels: Int {
        if let raw = ProcessInfo.processInfo.environment["QW_VISION_MAX_PIXELS"],
           let parsed = Int(raw), parsed >= QWMMConfig.minPixels {
            return min(parsed, QWMMConfig.maxPixels)
        }
        return 1_048_576
    }

    /// Decodes an image and resizes it via ImageIO thumbnailing before full-pixel allocation when downsampling.
    public static func decodeSmartResized(
        urlOrDataUri: String,
        maxPixels: Int = QWImageDecoder.liveMaxPixels,
        policy: QWImageDecodePolicy = .serverDefault,
        spec: QWMMSpec = .qwen3VLReference
    ) throws -> (image: DecodedImage, grid: QWVisionGrid) {
        let data = try resolveSourceData(urlOrDataUri: urlOrDataUri, policy: policy)
        let (source, rawWidth, rawHeight) = try inspectSource(data: data, policy: policy)
        let resized = try QWMMConfig.smartResize(height: rawHeight, width: rawWidth, maxPixels: maxPixels, spec: spec)
        let grid = try QWVisionGrid(t: 1, h: resized.h / spec.patch, w: resized.w / spec.patch, spec: spec)

        let cgImage: CGImage
        if resized.w < rawWidth || resized.h < rawHeight {
            let thumbOpts: [CFString: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceShouldCacheImmediately: false,
                kCGImageSourceThumbnailMaxPixelSize: max(resized.w, resized.h),
            ]
            guard let thumb = CGImageSourceCreateThumbnailAtIndex(source, 0, thumbOpts as CFDictionary) else {
                throw QWImageDecoderError.imageDecodeFailed("CGImageSourceCreateThumbnailAtIndex failed")
            }
            cgImage = thumb
        } else {
            guard let full = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                throw QWImageDecoderError.imageDecodeFailed("CGImageSourceCreateImageAtIndex failed")
            }
            cgImage = full
        }
        let decoded = try renderRGB(cgImage: cgImage, targetWidth: resized.w, targetHeight: resized.h)
        return (decoded, grid)
    }

    struct EncodedCacheKey: Hashable, Sendable {
        let digest: String
        let hiddenDim: Int
        let maxPixels: Int
        let patch: Int
        let merge: Int
        let posSide: Int
    }

    final class EncodedImageLRUCache: @unchecked Sendable {
        private let lock = NSLock()
        private let maxEntries = 8
        private var order: [EncodedCacheKey] = []
        private var entries: [EncodedCacheKey: (grid: QWVisionGrid, features: [Float])] = [:]

        func get(_ key: EncodedCacheKey) -> (grid: QWVisionGrid, features: [Float])? {
            lock.lock()
            defer { lock.unlock() }
            guard let val = entries[key] else { return nil }
            if let idx = order.firstIndex(of: key) {
                order.remove(at: idx)
                order.append(key)
            }
            return val
        }

        func put(_ key: EncodedCacheKey, value: (grid: QWVisionGrid, features: [Float])) {
            lock.lock()
            defer { lock.unlock() }
            if entries[key] != nil, let idx = order.firstIndex(of: key) {
                order.remove(at: idx)
            } else if order.count >= maxEntries, let oldest = order.first {
                order.removeFirst()
                entries.removeValue(forKey: oldest)
            }
            order.append(key)
            entries[key] = value
        }

        func clear() {
            lock.lock()
            defer { lock.unlock() }
            order.removeAll()
            entries.removeAll()
        }
    }

    static let encodedCache = EncodedImageLRUCache()

    private static func cacheKey(
        urlOrDataUri: String,
        hiddenDim: Int,
        maxPixels: Int,
        spec: QWMMSpec
    ) -> EncodedCacheKey {
        let trimmed = urlOrDataUri.trimmingCharacters(in: .whitespacesAndNewlines)
        var hasher = SHA256()
        hasher.update(data: Data(trimmed.utf8))
        if (trimmed.starts(with: "/") || trimmed.starts(with: "~")) && trimmed.count < 1024 {
            let path = trimmed.expandingTildePath
            if let attrs = try? FileManager.default.attributesOfItem(atPath: path) {
                if let size = attrs[.size] as? NSNumber {
                    hasher.update(data: Data("s:\(size.uint64Value)".utf8))
                }
                if let mtime = attrs[.modificationDate] as? Date {
                    hasher.update(data: Data("m:\(mtime.timeIntervalSince1970)".utf8))
                }
            }
        }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        return EncodedCacheKey(
            digest: digest,
            hiddenDim: hiddenDim,
            maxPixels: maxPixels,
            patch: spec.patch,
            merge: spec.merge,
            posSide: spec.posSide
        )
    }

    fileprivate static func cachedEncodedImage(
        urlOrDataUri: String, hiddenDim: Int, maxPixels: Int = QWImageDecoder.liveMaxPixels,
        spec: QWMMSpec = .qwen3VLReference
    ) -> (grid: QWVisionGrid, features: [Float])? {
        encodedCache.get(cacheKey(urlOrDataUri: urlOrDataUri, hiddenDim: hiddenDim, maxPixels: maxPixels, spec: spec))
    }

    fileprivate static func storeEncodedImage(
        urlOrDataUri: String, hiddenDim: Int, maxPixels: Int = QWImageDecoder.liveMaxPixels,
        spec: QWMMSpec = .qwen3VLReference,
        grid: QWVisionGrid, features: [Float]
    ) {
        encodedCache.put(
            cacheKey(urlOrDataUri: urlOrDataUri, hiddenDim: hiddenDim, maxPixels: maxPixels, spec: spec),
            value: (grid: grid, features: features)
        )
    }

    /// Loads vision on demand when `imageUrls` has uncached images, encodes all images into `QWVisualSpan`s,
    /// and unloads the vision head when `imageUrls` is empty to free RAM.
    public static func prepareMultimodalTurn(
        promptTokens: [Int],
        imageUrls: [String],
        hiddenDim: Int,
        visionBackend: (any QWVisionBackend)?,
        policy: QWImageDecodePolicy = .serverDefault,
        onImageEncoded: ((Int, Int, Int, Int, Double) -> Void)? = nil
    ) async throws -> (tokens: [Int], visualSpans: [QWVisualSpan], loadedOnDemand: Bool, unloadedOnText: Bool) {
        if imageUrls.isEmpty {
            if let visionBackend, visionBackend.isLoaded {
                try visionBackend.prepareForPrompt(hasImages: false)
                return (promptTokens, [], false, true)
            }
            return (promptTokens, [], false, false)
        }
        guard let visionBackend else {
            throw QWMultimodalError.badArgument("Model pack does not have an active vision tower")
        }
        let maxPixels = liveMaxPixels
        let spec = QWMMSpec(visionConfig: visionBackend.visionConfig, maxPixels: maxPixels)
        let wasLoaded = visionBackend.isLoaded
        var loadedNow = false
        var images: [(grid: QWVisionGrid, features: [Float])] = []
        let clock = ContinuousClock()
        for (idx, urlStr) in imageUrls.enumerated() {
            if let cached = cachedEncodedImage(
                urlOrDataUri: urlStr, hiddenDim: hiddenDim, maxPixels: maxPixels, spec: spec
            ) {
                onImageEncoded?(idx + 1, imageUrls.count, cached.grid.h, cached.grid.w, 0.0)
                images.append(cached)
                continue
            }
            do {
                if !visionBackend.isLoaded {
                    try visionBackend.prepareForPrompt(hasImages: true)
                    loadedNow = !wasLoaded && visionBackend.isLoaded
                }
                let (decoded, grid) = try decodeSmartResized(
                    urlOrDataUri: urlStr, maxPixels: maxPixels, policy: policy, spec: spec
                )
                let t0 = clock.now
                let visualTokens = try await qwEncodeVision(
                    imageRGB: decoded.rgb, height: decoded.height, width: decoded.width,
                    grid: grid, backend: visionBackend
                )
                let d = t0.duration(to: clock.now)
                let ms = Double(d.components.attoseconds) / 1e15 + Double(d.components.seconds) * 1e3
                storeEncodedImage(urlOrDataUri: urlStr, hiddenDim: hiddenDim, maxPixels: maxPixels,
                                  spec: spec, grid: grid, features: visualTokens)
                onImageEncoded?(idx + 1, imageUrls.count, grid.h, grid.w, ms)
                images.append((grid: grid, features: visualTokens))
            } catch let mmError as QWMultimodalError {
                throw mmError
            } catch {
                throw QWMultimodalError.badArgument("\(error)")
            }
        }
        let mmPrompt = try QWMultimodalPrompt(
            promptTokens: promptTokens, images: images, hiddenDim: hiddenDim, spec: spec
        )
        return (mmPrompt.expandedTokens, mmPrompt.visualSpans, loadedNow, false)
    }

    /// Synchronous CLI helper: loads vision on demand for `imagePath`, encodes into `QWVisualSpan`s, and unloads immediately.
    public static func prepareOneShotMultimodalTurn(
        promptTokens: [Int],
        imagePath: String,
        modelPath: String,
        hiddenDim: Int
    ) throws -> (tokens: [Int], visualSpans: [QWVisualSpan]) {
        let controller = try QWCLIVisionController(modelPath: modelPath)
        let out = try controller.prepareTurn(
            promptTokens: promptTokens, imagePaths: [imagePath],
            currentTurnHasImage: true, hiddenDim: hiddenDim
        )
        controller.unload()
        return out
    }

    /// Decodes raw image data (JPEG, PNG, WebP, TIFF) via ImageIO into RGB8 bytes.
    public static func decode(data: Data, policy: QWImageDecodePolicy = .serverDefault) throws -> DecodedImage {
        let (source, _, _) = try inspectSource(data: data, policy: policy)
        guard let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw QWImageDecoderError.imageDecodeFailed("CGImageSourceCreateImageAtIndex failed")
        }
        return try renderRGB(cgImage: cgImage, targetWidth: cgImage.width, targetHeight: cgImage.height)
    }

    private static func inspectSource(data: Data, policy: QWImageDecodePolicy) throws -> (CGImageSource, Int, Int) {
        guard data.count <= policy.maxSourceBytes else {
            throw QWImageDecoderError.imageTooLarge("Image buffer exceeds \(policy.maxSourceBytes) bytes")
        }
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            throw QWImageDecoderError.imageDecodeFailed("CGImageSource creation failed")
        }
        if let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
           let w = (props[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
           let h = (props[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue {
            guard w > 0, h > 0 else {
                throw QWImageDecoderError.emptyImage
            }
            guard w <= policy.maxPixelDimension && h <= policy.maxPixelDimension else {
                throw QWImageDecoderError.imageTooLarge("Image dimensions \(w)x\(h) exceed max \(policy.maxPixelDimension)")
            }
            return (source, w, h)
        }
        guard let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw QWImageDecoderError.imageDecodeFailed("CGImageSourceCreateImageAtIndex failed")
        }
        let w = cgImage.width
        let h = cgImage.height
        guard w > 0, h > 0 else { throw QWImageDecoderError.emptyImage }
        guard w <= policy.maxPixelDimension && h <= policy.maxPixelDimension else {
            throw QWImageDecoderError.imageTooLarge("Image dimensions \(w)x\(h) exceed max \(policy.maxPixelDimension)")
        }
        return (source, w, h)
    }

    private static func renderRGB(cgImage: CGImage, targetWidth: Int, targetHeight: Int) throws -> DecodedImage {
        guard targetWidth > 0, targetHeight > 0 else {
            throw QWImageDecoderError.emptyImage
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var rgba = [UInt8](repeating: 0, count: targetWidth * targetHeight * 4)
        let drew = rgba.withUnsafeMutableBytes { (ptr: UnsafeMutableRawBufferPointer) -> Bool in
            guard let context = CGContext(
                data: ptr.baseAddress,
                width: targetWidth,
                height: targetHeight,
                bitsPerComponent: 8,
                bytesPerRow: targetWidth * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else {
                return false
            }
            context.interpolationQuality = .high
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))
            return true
        }
        guard drew else {
            throw QWImageDecoderError.imageDecodeFailed("CGContext creation failed")
        }

        var rgb = [UInt8](repeating: 0, count: targetWidth * targetHeight * 3)
        for i in 0..<(targetWidth * targetHeight) {
            rgb[i * 3 + 0] = rgba[i * 4 + 0]
            rgb[i * 3 + 1] = rgba[i * 4 + 1]
            rgb[i * 3 + 2] = rgba[i * 4 + 2]
        }
        return DecodedImage(rgb: rgb, width: targetWidth, height: targetHeight)
    }
}

/// Prompt-scoped vision residency controller for interactive CLI (`qwg repl`) and one-shot CLI (`qwg run`).
public final class QWCLIVisionController: @unchecked Sendable {
    private let backend: QWVisionOnDemandBackend

    public init(modelPath: String) throws {
        self.backend = try QWVisionOnDemandBackend(modelPath: modelPath)
    }

    public var isLoaded: Bool { backend.isLoaded }

    public func unload() {
        backend.unload()
    }

    /// Encodes `imagePaths` when present, loading vision on demand and unloading when `currentTurnHasImage == false`.
    public func prepareTurn(
        promptTokens: [Int],
        imagePaths: [String],
        currentTurnHasImage: Bool,
        hiddenDim: Int
    ) throws -> (tokens: [Int], visualSpans: [QWVisualSpan]) {
        if imagePaths.isEmpty {
            try backend.prepareForPrompt(hasImages: false)
            return (promptTokens, [])
        }
        let maxPixels = QWImageDecoder.liveMaxPixels
        let spec = QWMMSpec(visionConfig: backend.visionConfig, maxPixels: maxPixels)
        var images: [(grid: QWVisionGrid, features: [Float])] = []
        for path in imagePaths {
            if let cached = QWImageDecoder.cachedEncodedImage(
                urlOrDataUri: path, hiddenDim: hiddenDim, maxPixels: maxPixels, spec: spec
            ) {
                images.append(cached)
                continue
            }
            if !backend.isLoaded {
                try backend.prepareForPrompt(hasImages: true)
            }
            let (decoded, grid) = try QWImageDecoder.decodeSmartResized(
                urlOrDataUri: path, maxPixels: maxPixels, policy: .cliLocal, spec: spec
            )
            let patches = try qwBuildPatchSlab(
                imageRGB: decoded.rgb, height: decoded.height, width: decoded.width, grid: grid
            )
            let visualTokens = try backend.encodeSync(
                patches: patches, grid: grid.plan, bins: grid.interpBins, posIds: grid.ropePositions
            )
            QWImageDecoder.storeEncodedImage(
                urlOrDataUri: path, hiddenDim: hiddenDim, maxPixels: maxPixels,
                spec: spec, grid: grid, features: visualTokens
            )
            images.append((grid: grid, features: visualTokens))
        }
        if !currentTurnHasImage {
            try backend.prepareForPrompt(hasImages: false)
        }
        let mmPrompt = try QWMultimodalPrompt(
            promptTokens: promptTokens, images: images, hiddenDim: hiddenDim, spec: spec
        )
        return (mmPrompt.expandedTokens, mmPrompt.visualSpans)
    }
}
