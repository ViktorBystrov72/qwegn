import Accelerate
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

    private enum ClassifiedImageSource: Sendable {
        case dataURI(String)
        case remoteURL(URL)
        case localFilePath(String)
        case rawBase64(String)
    }

    private static func validateSourcePolicy(
        urlOrDataUri: String,
        policy: QWImageDecodePolicy
    ) throws -> ClassifiedImageSource {
        let trimmed = urlOrDataUri.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw QWImageDecoderError.disallowedImageSource("Empty image source")
        }
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
            return .dataURI(trimmed)
        }

        if lower.hasPrefix("http://") || lower.hasPrefix("https://") {
            guard policy.allowRemoteImageURLs else {
                throw QWImageDecoderError.disallowedImageSource("Remote HTTP(S) image URLs are disabled by policy")
            }
            guard let url = URL(string: trimmed) else {
                throw QWImageDecoderError.invalidDataUrl(trimmed)
            }
            return .remoteURL(url)
        }

        if lower.hasPrefix("file://") || trimmed.hasPrefix("/") || trimmed.hasPrefix("~")
            || trimmed.hasPrefix("./") || trimmed.hasPrefix("../") {
            guard policy.allowLocalImagePaths else {
                throw QWImageDecoderError.disallowedImageSource("Local filesystem image paths are disabled by policy")
            }
            let path = lower.hasPrefix("file://")
                ? (URL(string: trimmed)?.path ?? "")
                : trimmed.expandingTildePath
            return .localFilePath(path)
        }

        if lower.contains("://") {
            throw QWImageDecoderError.disallowedImageSource("Rejected image URL '\(trimmed.prefix(48))'")
        }

        if policy.allowLocalImagePaths, FileManager.default.fileExists(atPath: trimmed.expandingTildePath) {
            return .localFilePath(trimmed.expandingTildePath)
        }

        if policy.allowDataURLs {
            return .rawBase64(trimmed)
        }

        throw QWImageDecoderError.disallowedImageSource("Rejected image URL '\(trimmed.prefix(48))'")
    }

    static func resolveSourceData(urlOrDataUri: String, policy: QWImageDecodePolicy) throws -> Data {
        let classified = try validateSourcePolicy(urlOrDataUri: urlOrDataUri, policy: policy)
        switch classified {
        case .dataURI(let trimmed):
            return try decodeBase64DataURI(trimmed, maxBytes: policy.maxSourceBytes)
        case .remoteURL(let url):
            return try QWImageDecodePolicy.fetchBoundedRemoteData(url: url, maxBytes: policy.maxSourceBytes)
        case .localFilePath(let path):
            let displayPath = urlOrDataUri.trimmingCharacters(in: .whitespacesAndNewlines)
            return try readBoundedLocalFile(atPath: path, displayPath: displayPath, maxBytes: policy.maxSourceBytes)
        case .rawBase64(let trimmed):
            return try decodeRawBase64(trimmed, maxBytes: policy.maxSourceBytes)
        }
    }

    static func resolveSourceDataAsync(urlOrDataUri: String, policy: QWImageDecodePolicy) async throws -> Data {
        let classified = try validateSourcePolicy(urlOrDataUri: urlOrDataUri, policy: policy)
        switch classified {
        case .remoteURL(let url):
            return try await QWImageDecodePolicy.fetchBoundedRemoteDataAsync(url: url, maxBytes: policy.maxSourceBytes)
        case .dataURI(let trimmed):
            return try await runOffCooperativePool {
                try decodeBase64DataURI(trimmed, maxBytes: policy.maxSourceBytes)
            }
        case .localFilePath(let path):
            let displayPath = urlOrDataUri.trimmingCharacters(in: .whitespacesAndNewlines)
            return try await runOffCooperativePool {
                try readBoundedLocalFile(atPath: path, displayPath: displayPath, maxBytes: policy.maxSourceBytes)
            }
        case .rawBase64(let trimmed):
            return try await runOffCooperativePool {
                try decodeRawBase64(trimmed, maxBytes: policy.maxSourceBytes)
            }
        }
    }

    private static func decodeBase64DataURI(_ trimmed: String, maxBytes: Int) throws -> Data {
        guard let commaIdx = trimmed.firstIndex(of: ",") else {
            throw QWImageDecoderError.invalidDataUrl(trimmed)
        }
        let base64Part = String(trimmed[trimmed.index(after: commaIdx)...])
        guard base64Part.utf8.count <= (maxBytes / 3 + 2) * 4 else {
            throw QWImageDecoderError.imageTooLarge("Base64 payload exceeds \(maxBytes) bytes")
        }
        guard let decoded = Data(base64Encoded: base64Part, options: .ignoreUnknownCharacters) else {
            throw QWImageDecoderError.base64DecodeFailed
        }
        guard decoded.count <= maxBytes else {
            throw QWImageDecoderError.imageTooLarge("Decoded image exceeds \(maxBytes) bytes")
        }
        return decoded
    }

    private static func decodeRawBase64(_ trimmed: String, maxBytes: Int) throws -> Data {
        guard trimmed.utf8.count <= (maxBytes / 3 + 2) * 4,
              let decoded = Data(base64Encoded: trimmed, options: .ignoreUnknownCharacters),
              !decoded.isEmpty else {
            throw QWImageDecoderError.disallowedImageSource("Rejected image URL '\(trimmed.prefix(48))'")
        }
        guard decoded.count <= maxBytes else {
            throw QWImageDecoderError.imageTooLarge("Decoded image exceeds \(maxBytes) bytes")
        }
        return decoded
    }

    static func runOffCooperativePool<T: Sendable>(
        _ work: @escaping @Sendable () throws -> T
    ) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    continuation.resume(returning: try work())
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
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
        return try decodeSmartResizedData(data, maxPixels: maxPixels, policy: policy, spec: spec)
    }

    private static func decodeSmartResizedAsync(
        urlOrDataUri: String,
        maxPixels: Int = QWImageDecoder.liveMaxPixels,
        policy: QWImageDecodePolicy = .serverDefault,
        spec: QWMMSpec = .qwen3VLReference
    ) async throws -> (image: DecodedImage, grid: QWVisionGrid) {
        let data = try await resolveSourceDataAsync(urlOrDataUri: urlOrDataUri, policy: policy)
        return try await runOffCooperativePool {
            try decodeSmartResizedData(data, maxPixels: maxPixels, policy: policy, spec: spec)
        }
    }

    private static func decodeSmartResizedData(
        _ data: Data,
        maxPixels: Int,
        policy: QWImageDecodePolicy,
        spec: QWMMSpec
    ) throws -> (image: DecodedImage, grid: QWVisionGrid) {
        let (source, orientedWidth, orientedHeight, orientation) = try inspectSource(data: data, policy: policy)
        let resized = try QWMMConfig.smartResize(
            height: orientedHeight, width: orientedWidth, maxPixels: maxPixels, spec: spec
        )
        let grid = try QWVisionGrid(t: 1, h: resized.h / spec.patch, w: resized.w / spec.patch, spec: spec)

        let cgImage: CGImage
        let renderOrientation: UInt32
        if resized.w < orientedWidth || resized.h < orientedHeight {
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
            renderOrientation = 1
        } else {
            guard let full = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                throw QWImageDecoderError.imageDecodeFailed("CGImageSourceCreateImageAtIndex failed")
            }
            cgImage = full
            renderOrientation = orientation
        }
        let decoded = try renderRGB(
            cgImage: cgImage, targetWidth: resized.w, targetHeight: resized.h, orientation: renderOrientation
        )
        return (decoded, grid)
    }

    struct EncodedCacheKey: Hashable, Sendable {
        let digest: String
        let hiddenDim: Int
        let maxPixels: Int
        let patch: Int
        let merge: Int
        let posSide: Int
        let usesUnfoldedPatchEmbed: Bool
        let imageMean: [Float]
        let imageStd: [Float]
        let allowDataURLs: Bool
        let allowLocalImagePaths: Bool
        let allowRemoteImageURLs: Bool
    }

    final class EncodedImageLRUCache: @unchecked Sendable {
        private let lock = NSLock()
        private let maxEntries = 8
        private var order: [EncodedCacheKey] = []
        private var entries: [EncodedCacheKey: (grid: QWVisionGrid, features: [Float])] = [:]

        func get(_ key: EncodedCacheKey) -> (grid: QWVisionGrid, features: [Float])? {
            lock.withLock {
                guard let val = entries[key] else { return nil }
                if let idx = order.firstIndex(of: key) {
                    order.remove(at: idx)
                    order.append(key)
                }
                return val
            }
        }

        func put(_ key: EncodedCacheKey, value: (grid: QWVisionGrid, features: [Float])) {
            lock.withLock {
                if entries[key] != nil, let idx = order.firstIndex(of: key) {
                    order.remove(at: idx)
                } else if order.count >= maxEntries, let oldest = order.first {
                    order.removeFirst()
                    entries.removeValue(forKey: oldest)
                }
                order.append(key)
                entries[key] = value
            }
        }

        func clear() {
            lock.withLock {
                order.removeAll()
                entries.removeAll()
            }
        }
    }

    static let encodedCache = EncodedImageLRUCache()

    private static func cacheKey(
        urlOrDataUri: String,
        hiddenDim: Int,
        maxPixels: Int,
        spec: QWMMSpec,
        policy: QWImageDecodePolicy
    ) -> EncodedCacheKey {
        let trimmed = urlOrDataUri.trimmingCharacters(in: .whitespacesAndNewlines)
        var hasher = SHA256()
        hasher.update(data: Data(trimmed.utf8))
        let lower = trimmed.lowercased()
        let isLocalPathCandidate = lower.hasPrefix("file://") || trimmed.starts(with: "/")
            || trimmed.starts(with: "~") || trimmed.starts(with: "./") || trimmed.starts(with: "../")
        if policy.allowLocalImagePaths && isLocalPathCandidate && trimmed.count < 1024 {
            let path = lower.hasPrefix("file://")
                ? (URL(string: trimmed)?.path ?? "")
                : trimmed.expandingTildePath
            if !path.isEmpty, let attrs = try? FileManager.default.attributesOfItem(atPath: path) {
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
            posSide: spec.posSide,
            usesUnfoldedPatchEmbed: spec.usesUnfoldedPatchEmbed,
            imageMean: spec.imageMean,
            imageStd: spec.imageStd,
            allowDataURLs: policy.allowDataURLs,
            allowLocalImagePaths: policy.allowLocalImagePaths,
            allowRemoteImageURLs: policy.allowRemoteImageURLs
        )
    }

    fileprivate static func cachedEncodedImage(
        urlOrDataUri: String, hiddenDim: Int, maxPixels: Int = QWImageDecoder.liveMaxPixels,
        spec: QWMMSpec = .qwen3VLReference, policy: QWImageDecodePolicy = .serverDefault
    ) -> (grid: QWVisionGrid, features: [Float])? {
        encodedCache.get(cacheKey(
            urlOrDataUri: urlOrDataUri, hiddenDim: hiddenDim, maxPixels: maxPixels, spec: spec, policy: policy
        ))
    }

    fileprivate static func storeEncodedImage(
        urlOrDataUri: String, hiddenDim: Int, maxPixels: Int = QWImageDecoder.liveMaxPixels,
        spec: QWMMSpec = .qwen3VLReference, policy: QWImageDecodePolicy = .serverDefault,
        grid: QWVisionGrid, features: [Float]
    ) {
        encodedCache.put(
            cacheKey(urlOrDataUri: urlOrDataUri, hiddenDim: hiddenDim, maxPixels: maxPixels, spec: spec, policy: policy),
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
                return (promptTokens, [], false, !visionBackend.isLoaded)
            }
            return (promptTokens, [], false, false)
        }
        guard imageUrls.count <= QWImageDecodePolicy.defaultMaxImagesPerTurn else {
            throw QWMultimodalError.badArgument(
                "Too many images in turn (\(imageUrls.count)); maximum is \(QWImageDecodePolicy.defaultMaxImagesPerTurn)"
            )
        }
        guard let visionBackend else {
            throw QWMultimodalError.badArgument("Model pack does not have an active vision tower")
        }
        guard visionBackend.isVisionEnabled else {
            throw QWMultimodalError.badArgument(
                "Vision tower is disabled (--no-load-vision). Enable --load-vision to process image inputs."
            )
        }
        let maxPixels = liveMaxPixels
        let spec = QWMMSpec(visionConfig: visionBackend.visionConfig, maxPixels: maxPixels)
        let wasLoaded = visionBackend.isLoaded
        var loadedNow = false
        var images: [(grid: QWVisionGrid, features: [Float])] = []
        let clock = ContinuousClock()
        for (idx, urlStr) in imageUrls.enumerated() {
            do {
                _ = try validateSourcePolicy(urlOrDataUri: urlStr, policy: policy)
                if let cached = cachedEncodedImage(
                    urlOrDataUri: urlStr, hiddenDim: hiddenDim, maxPixels: maxPixels, spec: spec, policy: policy
                ) {
                    onImageEncoded?(idx + 1, imageUrls.count, cached.grid.h, cached.grid.w, 0.0)
                    images.append(cached)
                    continue
                }
                if !visionBackend.isLoaded {
                    try visionBackend.prepareForPrompt(hasImages: true)
                    loadedNow = !wasLoaded && visionBackend.isLoaded
                }
                let patches: [Float]
                let grid: QWVisionGrid
                if spec.isInkling2 {
                    let prep = try await prepareInkling2PatchesAsync(
                        urlOrDataUri: urlStr, maxPixels: maxPixels, policy: policy, spec: spec
                    )
                    patches = prep.patches
                    grid = prep.grid
                } else {
                    let (decoded, decodedGrid) = try await decodeSmartResizedAsync(
                        urlOrDataUri: urlStr, maxPixels: maxPixels, policy: policy, spec: spec
                    )
                    grid = decodedGrid
                    patches = try qwBuildPatchSlab(
                        imageRGB: decoded.rgb, height: decoded.height, width: decoded.width, grid: grid
                    )
                }
                let t0 = clock.now
                let visualTokens = try await visionBackend.encode(
                    patches: patches, grid: grid.plan, bins: grid.interpBins, posIds: grid.ropePositions
                )
                let d = t0.duration(to: clock.now)
                let ms = Double(d.components.attoseconds) / 1e15 + Double(d.components.seconds) * 1e3
                storeEncodedImage(urlOrDataUri: urlStr, hiddenDim: hiddenDim, maxPixels: maxPixels,
                                  spec: spec, policy: policy, grid: grid, features: visualTokens)
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
        let (source, orientedW, orientedH, orientation) = try inspectSource(data: data, policy: policy)
        guard let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw QWImageDecoderError.imageDecodeFailed("CGImageSourceCreateImageAtIndex failed")
        }
        return try renderRGB(
            cgImage: cgImage, targetWidth: orientedW, targetHeight: orientedH, orientation: orientation
        )
    }

    static func inspectSource(
        data: Data,
        policy: QWImageDecodePolicy
    ) throws -> (source: CGImageSource, width: Int, height: Int, orientation: UInt32) {
        guard data.count <= policy.maxSourceBytes else {
            throw QWImageDecoderError.imageTooLarge("Image buffer exceeds \(policy.maxSourceBytes) bytes")
        }
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            throw QWImageDecoderError.imageDecodeFailed("CGImageSource creation failed")
        }
        guard let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let w = (props[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
              let h = (props[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue else {
            throw QWImageDecoderError.imageDecodeFailed("Missing pixel dimensions in image header")
        }
        guard w > 0, h > 0 else {
            throw QWImageDecoderError.emptyImage
        }
        let rawOrientation = (props[kCGImagePropertyOrientation] as? NSNumber)?.uint32Value ?? 1
        let orientation: UInt32 = (1...8).contains(rawOrientation) ? rawOrientation : 1
        let swapsAxes = (5...8).contains(orientation)
        let orientedW = swapsAxes ? h : w
        let orientedH = swapsAxes ? w : h
        guard orientedW <= policy.maxPixelDimension && orientedH <= policy.maxPixelDimension else {
            throw QWImageDecoderError.imageTooLarge(
                "Image dimensions \(orientedW)x\(orientedH) exceed max \(policy.maxPixelDimension)"
            )
        }
        let (area, overflow) = Int64(orientedW).multipliedReportingOverflow(by: Int64(orientedH))
        let maxArea = min(Int64(policy.maxPixelArea), Int64(QWImageDecodePolicy.defaultMaxPixelArea))
        guard !overflow, area <= maxArea else {
            throw QWImageDecoderError.imageTooLarge("Image pixel area \(orientedW)x\(orientedH) exceeds max \(maxArea)")
        }
        return (source, orientedW, orientedH, orientation)
    }

    private static func renderRGB(
        cgImage: CGImage,
        targetWidth: Int,
        targetHeight: Int,
        orientation: UInt32 = 1
    ) throws -> DecodedImage {
        guard targetWidth > 0, targetHeight > 0 else {
            throw QWImageDecoderError.emptyImage
        }
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw QWImageDecoderError.imageDecodeFailed("Failed to create sRGB colorspace")
        }
        var rgba = [UInt8](repeating: 255, count: targetWidth * targetHeight * 4)
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        let drew = rgba.withUnsafeMutableBytes { (ptr: UnsafeMutableRawBufferPointer) -> Bool in
            guard let context = CGContext(
                data: ptr.baseAddress,
                width: targetWidth,
                height: targetHeight,
                bitsPerComponent: 8,
                bytesPerRow: targetWidth * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            ) else {
                return false
            }
            context.setFillColor(CGColor(srgbRed: 1.0, green: 1.0, blue: 1.0, alpha: 1.0))
            context.fill(CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))
            let swapsAxes = (5...8).contains(orientation)
            let drawW = swapsAxes ? targetHeight : targetWidth
            let drawH = swapsAxes ? targetWidth : targetHeight
            let w = CGFloat(targetWidth)
            let h = CGFloat(targetHeight)
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
        rgba.withUnsafeBytes { srcRaw in
            rgb.withUnsafeMutableBytes { dstRaw in
                guard let sBase = srcRaw.baseAddress, let dBase = dstRaw.baseAddress else { return }
                var srcBuf = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: sBase),
                    height: vImagePixelCount(targetHeight),
                    width: vImagePixelCount(targetWidth),
                    rowBytes: targetWidth * 4
                )
                var dstBuf = vImage_Buffer(
                    data: dBase,
                    height: vImagePixelCount(targetHeight),
                    width: vImagePixelCount(targetWidth),
                    rowBytes: targetWidth * 3
                )
                if vImageConvert_RGBA8888toRGB888(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags)) != kvImageNoError {
                    let sPtr = sBase.assumingMemoryBound(to: UInt8.self)
                    let dPtr = dBase.assumingMemoryBound(to: UInt8.self)
                    for i in 0..<(targetWidth * targetHeight) {
                        dPtr[i * 3 + 0] = sPtr[i * 4 + 0]
                        dPtr[i * 3 + 1] = sPtr[i * 4 + 1]
                        dPtr[i * 3 + 2] = sPtr[i * 4 + 2]
                    }
                }
            }
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
        guard backend.isVisionEnabled else {
            throw QWMultimodalError.badArgument(
                "Vision tower is disabled (--no-load-vision). Enable --load-vision to process image inputs."
            )
        }
        let maxPixels = QWImageDecoder.liveMaxPixels
        let spec = QWMMSpec(visionConfig: backend.visionConfig, maxPixels: maxPixels)
        var images: [(grid: QWVisionGrid, features: [Float])] = []
        for path in imagePaths {
            if let cached = QWImageDecoder.cachedEncodedImage(
                urlOrDataUri: path, hiddenDim: hiddenDim, maxPixels: maxPixels, spec: spec, policy: .cliLocal
            ) {
                images.append(cached)
                continue
            }
            if !backend.isLoaded {
                try backend.prepareForPrompt(hasImages: true)
            }
            let patches: [Float]
            let grid: QWVisionGrid
            if spec.isInkling2 {
                let prep = try QWImageDecoder.prepareInkling2Patches(
                    urlOrDataUri: path, maxPixels: maxPixels, policy: .cliLocal, spec: spec
                )
                patches = prep.patches
                grid = prep.grid
            } else {
                let (decoded, decodedGrid) = try QWImageDecoder.decodeSmartResized(
                    urlOrDataUri: path, maxPixels: maxPixels, policy: .cliLocal, spec: spec
                )
                grid = decodedGrid
                patches = try qwBuildPatchSlab(
                    imageRGB: decoded.rgb, height: decoded.height, width: decoded.width, grid: grid
                )
            }
            let visualTokens = try backend.encodeSync(
                patches: patches, grid: grid.plan, bins: grid.interpBins, posIds: grid.ropePositions
            )
            QWImageDecoder.storeEncodedImage(
                urlOrDataUri: path, hiddenDim: hiddenDim, maxPixels: maxPixels,
                spec: spec, policy: .cliLocal, grid: grid, features: visualTokens
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
