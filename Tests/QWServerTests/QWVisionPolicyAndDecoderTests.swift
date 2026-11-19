// QWVisionPolicyAndDecoderTests.swift — Regression tests for Wave 2 vision
// residency enforcement, safetensors/ViT bounds, Jinja quoting, and EXIF/sRGB/alpha decoding.
import CoreGraphics
import Foundation
import ImageIO
import QWCore
@testable import QWEngine
import QWHAL
import QWMM
@testable import QWServer
import QWToken
@testable import QWVision
import UniformTypeIdentifiers
import XCTest

private final class PolicyTestVisionBackend: QWVisionBackend, @unchecked Sendable {
    var isLoaded: Bool
    let keepResident: Bool
    let allowOnDemand: Bool
    let outDim: Int

    init(isLoaded: Bool, keepResident: Bool, allowOnDemand: Bool, outDim: Int = 16) {
        self.isLoaded = isLoaded
        self.keepResident = keepResident
        self.allowOnDemand = allowOnDemand
        self.outDim = outDim
    }

    var isVisionEnabled: Bool { isLoaded || keepResident || allowOnDemand }

    func prepareForPrompt(hasImages: Bool) throws {
        if hasImages {
            guard isVisionEnabled else {
                throw QWVisionError.badLayout("vision disabled")
            }
            isLoaded = true
        } else if !keepResident {
            isLoaded = false
        }
    }

    func encode(patches: [Float], grid: QWVisionGridPlan,
                bins: QWVisionRef.InterpBins, posIds: [Int32]) async throws -> [Float] {
        try prepareForPrompt(hasImages: true)
        return [Float](repeating: 0.5, count: grid.nToken * outDim)
    }
}

private final class PolicyStubModelBackend: QWModelBackend, @unchecked Sendable {
    let family: QWModelDescriptor.Family = .qwen38Dense
    let descriptor: QWModelDescriptor
    let parametersInfo: QWEngineParametersInfo
    let visionBackend: (any QWVisionBackend)? = nil
    let hardwareCapabilities: QWDeviceCaps? = nil
    var prefixCache: QWPrefixCacheState {
        QWPrefixCacheState(isConfigured: false, warmBytes: 0, ssdBytes: 0, warmEntries: 0, ssdEntries: 0)
    }
    var capabilities: Set<QWKnobCapability> { [] }
    var supportsTokenConstraints: Bool { false }

    init(descriptor: QWModelDescriptor, packPath: String) {
        self.descriptor = descriptor
        self.parametersInfo = QWEngineParametersInfo(
            modelPackPath: packPath, modelName: "stub", modelDescriptor: descriptor,
            headStorage: "BF16", visionStatus: "disabled", contextBudget: 4096,
            kvMode: .q4, mtpDrafts: 0, mtpBackend: "none", gdnPrefillMode: "chunked",
            prefillScratchEviction: "auto", prefillChunkRows: 256, weightContainerNote: nil
        )
    }

    func generate(request: QWModelGenerationRequest, config: QWEngineLifecycleActor.Config,
                  onToken: ((Int) -> Bool)?) throws -> QWModelCompletionMetrics {
        QWModelCompletionMetrics(tokens: [descriptor.eosTokenId], prefillSec: 0.01,
                                 decodeSec: 0.01, firstTokenSec: 0.01, activeContextCapacity: 4096)
    }
}

final class QWVisionPolicyAndDecoderTests: XCTestCase {
    private func makeMinimalTokenizer() throws -> QWTokenizer {
        var vocab: [String: Int] = [:]
        var nextId = 0
        var n = 0
        for b in 0..<256 {
            let printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
            let sc = printable ? Unicode.Scalar(UInt8(b)) : Unicode.Scalar(0x100 + n)!
            if !printable { n += 1 }
            if b == 0xC0 || b == 0xC1 || b >= 0xF5 { continue }
            vocab[String(sc)] = nextId
            nextId += 1
        }
        let added: [[String: Any]] = [
            ["id": 248044, "content": "<|endoftext|>", "special": true],
            ["id": 248045, "content": "<|im_start|>", "special": true],
            ["id": 248046, "content": "<|im_end|>", "special": true],
            ["id": 248053, "content": "<|vision_start|>", "special": true],
            ["id": 248054, "content": "<|vision_end|>", "special": true],
            ["id": 248056, "content": "<|image_pad|>", "special": true]
        ]
        let preRegex = "[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|\\s+"
        let root: [String: Any] = [
            "added_tokens": added,
            "model": ["type": "BPE", "vocab": vocab, "merges": []],
            "pre_tokenizer": [
                "type": "Sequence",
                "pretokenizers": [
                    ["type": "Split", "pattern": ["Regex": preRegex], "behavior": "Isolated", "invert": false],
                    ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false]
                ]
            ]
        ]
        return try QWTokenizer(json: try JSONSerialization.data(withJSONObject: root))
    }

    private func makePngDataUri(width: Int, height: Int, rgbaBytes: [UInt8]) throws -> String {
        let cs = try XCTUnwrap(CGColorSpace(name: CGColorSpace.sRGB))
        var buf = rgbaBytes
        let ctx = try XCTUnwrap(CGContext(
            data: &buf, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
            space: cs, bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        ))
        let cg = try XCTUnwrap(ctx.makeImage())
        let data = NSMutableData()
        let dest = try XCTUnwrap(CGImageDestinationCreateWithData(data as CFMutableData, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(dest, cg, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
        return "data:image/png;base64,\((data as Data).base64EncodedString())"
    }

    func testEncodeSyncRefusesWhenVisionDisabledWithoutInvokingLoader() throws {
        final class Flag: @unchecked Sendable {
            private let lock = NSLock()
            private var value = false
            func mark() { lock.withLock { value = true } }
            var called: Bool { lock.withLock { value } }
        }
        let flag = Flag()
        let backend = try QWVisionOnDemandBackend(
            source: .custom {
                flag.mark()
                throw QWVisionError.badLayout("should not load")
            },
            keepResident: false,
            allowOnDemand: false
        )
        let grid = try QWVisionGrid(t: 1, h: 2, w: 2)
        let patches = [Float](repeating: 0, count: grid.patches * QWVisionConfig.qwen3VLReference.patchChannels)
        XCTAssertThrowsError(try backend.encodeSync(
            patches: patches, grid: grid.plan, bins: grid.interpBins, posIds: grid.ropePositions
        )) { error in
            XCTAssertTrue(error is QWVisionError)
        }
        XCTAssertFalse(flag.called)
    }

    func testDisabledVisionBypassesEncodedCacheAndUnloadedOnTextReflectsActualTransition() async throws {
        QWImageDecoder.encodedCache.clear()
        let uri = try makePngDataUri(width: 32, height: 32, rgbaBytes: [UInt8](repeating: 255, count: 32 * 32 * 4))
        let tokens = [10, 248056, 20]

        let enabled = PolicyTestVisionBackend(isLoaded: false, keepResident: true, allowOnDemand: false)
        _ = try await QWImageDecoder.prepareMultimodalTurn(
            promptTokens: tokens, imageUrls: [uri], hiddenDim: 16, visionBackend: enabled
        )
        // Resident backend stays loaded on text turn -> unloadedOnText must be false.
        let residentTextTurn = try await QWImageDecoder.prepareMultimodalTurn(
            promptTokens: [10, 20], imageUrls: [], hiddenDim: 16, visionBackend: enabled
        )
        XCTAssertFalse(residentTextTurn.unloadedOnText)

        // On-demand backend actually unloads on text turn -> unloadedOnText must be true.
        let onDemand = PolicyTestVisionBackend(isLoaded: true, keepResident: false, allowOnDemand: true)
        let onDemandTextTurn = try await QWImageDecoder.prepareMultimodalTurn(
            promptTokens: [10, 20], imageUrls: [], hiddenDim: 16, visionBackend: onDemand
        )
        XCTAssertTrue(onDemandTextTurn.unloadedOnText)

        // Disabled backend must NOT serve cached visual encodings for the previously cached image.
        let disabled = PolicyTestVisionBackend(isLoaded: false, keepResident: false, allowOnDemand: false)
        do {
            _ = try await QWImageDecoder.prepareMultimodalTurn(
                promptTokens: tokens, imageUrls: [uri], hiddenDim: 16, visionBackend: disabled
            )
            XCTFail("Expected disabled vision backend to reject cached image")
        } catch let err as QWMultimodalError {
            XCTAssertTrue("\(err)".contains("--no-load-vision"))
        }
    }

    func testCountChatPromptTokensAndEnvEagerRespectNoLoadVision() async throws {
        setenv("QW_VISION_EAGER", "1", 1)
        defer { unsetenv("QW_VISION_EAGER") }

        let resolved = try QWServerConfigurationResolver.resolve(
            cliPack: "/dummy/model.qwp",
            cliLoadVision: false,
            cliIgnoreProfile: true,
            environment: ["QW_VISION_EAGER": "1"]
        )
        XCTAssertFalse(resolved.loadVision)

        let tok = try makeMinimalTokenizer()
        let tpl = try QWChatTemplate(source: "{% for m in messages %}{{ m.content }}{% endfor %}")
        let desc = QWModelDescriptor(
            family: .qwen38Dense, modelType: "qwen3_5", capabilities: [.vision],
            hidden: 256, vocab: 248320, denseIntermediate: 512, numLayers: 1,
            layerTypes: ["full_attention"], fullAttentionInterval: 1, rmsEps: 1e-6,
            maxPositionEmbeddings: 4096, qHeads: 4, kvHeads: 2, headDim: 64,
            attnOutputGate: false, partialRotaryFactor: 1.0, ropeTheta: 1e6,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil, numExperts: 0,
            expertsPerTok: 0, moeIntermediate: 0, sharedExpertIntermediate: 0,
            expertGateType: nil, hcCount: 1, hcLowrank: 0, gdnQKHeads: 0,
            gdnVHeads: 0, gdnHeadDim: 0, gdnQKVChannels: 0, gdnZChannels: 0,
            gdnConvKernel: 0, pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil, mtpLayers: 0,
            mtpUseDedicatedEmbeddings: false, eosTokenId: 248044, visDepth: 1
        )
        let cfg = QWEngineLifecycleActor.Config(
            packPath: "/dummy/model.qwp", modelName: "stub", host: "127.0.0.1", port: 0, loadVision: false
        )
        let backend = PolicyStubModelBackend(descriptor: desc, packPath: "/dummy/model.qwp")
        let actor = QWEngineLifecycleActor(
            config: cfg,
            logger: QWServerLogger(level: .none),
            load: { _, _, _ in QWEngineLifecycleActor.LoadedState(modelBackend: backend, tokenizer: tok, chatTemplate: tpl) }
        )
        let imgJson = """
        {"model":"stub","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}},{"type":"text","text":"hi"}]}]}
        """
        let req = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(imgJson.utf8))
        XCTAssertThrowsError(try actor.validateRequestCapabilities(req)) { err in
            XCTAssertTrue(err is QWMultimodalError)
        }
        do {
            _ = try await actor.countChatPromptTokens(request: req)
            XCTFail("countChatPromptTokens must enforce --no-load-vision")
        } catch let err as QWMultimodalError {
            XCTAssertTrue("\(err)".contains("--no-load-vision"))
        }
    }

    func testChatTemplateNormalizesOutsideQuotedStringsOnly() throws {
        let src = "{# skip.0 #}API v1.0 {{ \"v1.0\" ~ ' foo.0 ' ~ \"esc \\\"a.0\\\"\" ~ messages.0.content.0.text }}"
        let tpl = try QWChatTemplate(source: src)
        let msgs: [QWJson] = [
            .object([
                ("role", .string("user")),
                ("content", .array([.object([("text", .string("OK"))])]))
            ])
        ]
        let rendered = try tpl.render(messages: msgs)
        XCTAssertEqual(rendered, "API v1.0 v1.0 foo.0 esc \"a.0\"OK")
    }

    func testSafetensorsBoundsAndVisionNaNSafety() throws {
        let fm = FileManager.default
        let tmp = fm.temporaryDirectory.appendingPathComponent("qwg-st-test-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tmp) }

        // 1. Unsupported dtype must throw badLayout without fatalError.
        let hdrBadType = try JSONSerialization.data(withJSONObject: [
            "w": ["dtype": "I32", "shape": [2], "data_offsets": [0, 8]]
        ])
        var lenBad = UInt64(hdrBadType.count).littleEndian
        var blobBadType = Data(bytes: &lenBad, count: 8)
        blobBadType.append(hdrBadType)
        blobBadType.append(Data(count: 8))
        let pathBadType = tmp.appendingPathComponent("badtype.safetensors").path
        try blobBadType.write(to: URL(fileURLWithPath: pathBadType))
        let stBadType = try QWSafetensors(path: pathBadType)
        XCTAssertThrowsError(try stBadType.f32("w", want: [2])) { err in
            guard case .badLayout = err as? QWVisionError else { return XCTFail("wrong error \(err)") }
        }

        // 2. Truncated tensor offsets must throw badLayout instead of OOB read.
        let hdrOOB = try JSONSerialization.data(withJSONObject: [
            "w": ["dtype": "F32", "shape": [4], "data_offsets": [0, 16]]
        ])
        var lenOOB = UInt64(hdrOOB.count).littleEndian
        var blobOOB = Data(bytes: &lenOOB, count: 8)
        blobOOB.append(hdrOOB)
        blobOOB.append(Data(count: 4))
        let pathOOB = tmp.appendingPathComponent("oob.safetensors").path
        try blobOOB.write(to: URL(fileURLWithPath: pathOOB))
        let stOOB = try QWSafetensors(path: pathOOB)
        XCTAssertThrowsError(try stOOB.f32("w", want: [4])) { err in
            guard case .badLayout = err as? QWVisionError else { return XCTFail("wrong error \(err)") }
        }

        // 3. Negative header offset must fail at parse.
        let hdrNeg = try JSONSerialization.data(withJSONObject: [
            "w": ["dtype": "F32", "shape": [1], "data_offsets": [-4, 0]]
        ])
        var lenNeg = UInt64(hdrNeg.count).littleEndian
        var blobNeg = Data(bytes: &lenNeg, count: 8)
        blobNeg.append(hdrNeg)
        let pathNeg = tmp.appendingPathComponent("neg.safetensors").path
        try blobNeg.write(to: URL(fileURLWithPath: pathNeg))
        XCTAssertThrowsError(try QWSafetensors(path: pathNeg))

        // 4. Empty/zero-var/NaN qwLayerNorm + extreme QKV overflow softmax + out-of-bounds bins.idx must not crash or emit NaN.
        var emptyOut: [Float] = []
        qwLayerNorm([], into: &emptyOut, gamma: [], beta: [], rows: 0, dim: 0, eps: 1e-6)
        XCTAssertTrue(emptyOut.isEmpty)

        var zeroVarOut = [Float](repeating: -1, count: 4)
        qwLayerNorm([3, 3, 3, 3], into: &zeroVarOut, gamma: [1, 1, 1, 1], beta: [0.5, 0.5, 0.5, 0.5], rows: 1, dim: 4, eps: 0)
        XCTAssertEqual(zeroVarOut, [0.5, 0.5, 0.5, 0.5])

        var nanNormOut = [Float](repeating: -1, count: 4)
        qwLayerNorm([Float.nan, 1, 2, 3], into: &nanNormOut, gamma: [1, 1, 1, 1], beta: [0.25, 0.25, 0.25, 0.25], rows: 1, dim: 4, eps: 1e-6)
        XCTAssertEqual(nanNormOut, [0.25, 0.25, 0.25, 0.25])

        let d = QWVisionConfig.qwen3VLReference.dim
        let mlp = QWVisionConfig.qwen3VLReference.mlpInner
        let specs: [(String, [Int])] = [
            ("model.visual.pos_embed.weight", [2304, d]),
            ("model.visual.patch_embed.proj.weight", [d, 3, 2, 16, 16]),
            ("model.visual.patch_embed.proj.bias", [d]),
            ("model.visual.blocks.0.norm1.weight", [d]),
            ("model.visual.blocks.0.norm1.bias", [d]),
            ("model.visual.blocks.0.norm2.weight", [d]),
            ("model.visual.blocks.0.norm2.bias", [d]),
            ("model.visual.blocks.0.attn.qkv.weight", [3 * d, d]),
            ("model.visual.blocks.0.attn.qkv.bias", [3 * d]),
            ("model.visual.blocks.0.attn.proj.weight", [d, d]),
            ("model.visual.blocks.0.attn.proj.bias", [d]),
            ("model.visual.blocks.0.mlp.linear_fc1.weight", [mlp, d]),
            ("model.visual.blocks.0.mlp.linear_fc1.bias", [mlp]),
            ("model.visual.blocks.0.mlp.linear_fc2.weight", [d, mlp]),
            ("model.visual.blocks.0.mlp.linear_fc2.bias", [d]),
            ("model.visual.merger.norm.weight", [d]),
            ("model.visual.merger.norm.bias", [d]),
            ("model.visual.merger.linear_fc1.weight", [4608, 4608]),
            ("model.visual.merger.linear_fc1.bias", [4608]),
            ("model.visual.merger.linear_fc2.weight", [16, 4608]),
            ("model.visual.merger.linear_fc2.bias", [16]),
        ]
        var hdrObj: [String: Any] = [:]
        var stOff = 0
        var qkvStart = 0
        for (name, shape) in specs {
            if name == "model.visual.blocks.0.attn.qkv.weight" { qkvStart = stOff }
            let bytes = shape.reduce(1, *) * 2
            hdrObj[name] = ["dtype": "BF16", "shape": shape, "data_offsets": [stOff, stOff + bytes]]
            stOff += bytes
        }
        let stHdr = try JSONSerialization.data(withJSONObject: hdrObj)
        var stLen = UInt64(stHdr.count).littleEndian
        var stData = Data(bytes: &stLen, count: 8)
        stData.append(stHdr)
        // Fill weights with finite 1.0 (0x3C00), and Q/K slices of qkv.weight with 2^65 (0x6000) so Q*K^T overflows softmax.
        var payload = Data(count: stOff)
        let qkCount = 2 * d * d
        payload.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt16.self) else { return }
            for i in 0..<(stOff / 2) {
                ptr[i] = 0x3C00
            }
            let qkvElemStart = qkvStart / 2
            for i in 0..<qkCount {
                ptr[qkvElemStart + i] = 0x6000
            }
        }
        stData.append(payload)
        let stDir = tmp.appendingPathComponent("vit", isDirectory: true)
        try fm.createDirectory(at: stDir, withIntermediateDirectories: true)
        try stData.write(to: stDir.appendingPathComponent("model.safetensors"))
        let weights = try QWVisionWeights(dir: stDir.path)
        let ref = QWVisionRef(weights: weights)
        let g = try QWVisionGrid(t: 1, h: 2, w: 2)
        let oobBins = QWVisionRef.InterpBins(
            idx: [-1, 99_999, 0, 1, -5, 2304, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0],
            wgt: [Float](repeating: 0.25, count: 16)
        )
        let patches = [Float](repeating: 0.1, count: g.patches * QWVisionConfig.qwen3VLReference.patchChannels)
        let out = ref.encode(patches: patches, plan: g.plan, bins: oobBins, posIds: g.ropePositions)
        XCTAssertEqual(out.count, g.tokens * 16)
        XCTAssertTrue(out.allSatisfy { $0.isFinite })
    }

    func testImageDecoderEXIFOrientationAndWhiteAlphaCompositing() throws {
        // 4x2 asymmetric quadrant image: TL=Red, TR=Green, BL=Blue, BR=Yellow.
        let rawW = 4, rawH = 2
        var rgba = [UInt8](repeating: 255, count: rawW * rawH * 4)
        func setPx(_ x: Int, _ y: Int, _ r: UInt8, _ g: UInt8, _ b: UInt8) {
            let idx = (y * rawW + x) * 4
            rgba[idx] = r; rgba[idx + 1] = g; rgba[idx + 2] = b; rgba[idx + 3] = 255
        }
        for y in 0..<rawH {
            for x in 0..<rawW {
                if y == 0 && x < 2 { setPx(x, y, 255, 0, 0) }
                else if y == 0 && x >= 2 { setPx(x, y, 0, 255, 0) }
                else if y == 1 && x < 2 { setPx(x, y, 0, 0, 255) }
                else { setPx(x, y, 255, 255, 0) }
            }
        }
        let cs = try XCTUnwrap(CGColorSpace(name: CGColorSpace.sRGB))
        let ctx = try XCTUnwrap(CGContext(
            data: &rgba, width: rawW, height: rawH, bitsPerComponent: 8, bytesPerRow: rawW * 4,
            space: cs, bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        ))
        let baseCG = try XCTUnwrap(ctx.makeImage())

        for orientation: UInt32 in 1...8 {
            let tiffData = NSMutableData()
            let dest = try XCTUnwrap(CGImageDestinationCreateWithData(
                tiffData as CFMutableData, UTType.tiff.identifier as CFString, 1, nil
            ))
            CGImageDestinationAddImage(dest, baseCG, [kCGImagePropertyOrientation: orientation] as CFDictionary)
            XCTAssertTrue(CGImageDestinationFinalize(dest))
            let data = tiffData as Data

            let decoded = try QWImageDecoder.decode(data: data)
            let swaps = (5...8).contains(orientation)
            XCTAssertEqual(decoded.width, swaps ? rawH : rawW)
            XCTAssertEqual(decoded.height, swaps ? rawW : rawH)

            // Reference upright image via ImageIO thumbnail-with-transform.
            let src = try XCTUnwrap(CGImageSourceCreateWithData(data as CFData, nil))
            let refThumb = try XCTUnwrap(CGImageSourceCreateThumbnailAtIndex(src, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: max(rawW, rawH)
            ] as CFDictionary))
            XCTAssertEqual(decoded.width, refThumb.width)
            XCTAssertEqual(decoded.height, refThumb.height)

            // Compare top-left pixel against ImageIO upright reference.
            var refRGBA = [UInt8](repeating: 255, count: decoded.width * decoded.height * 4)
            let refCtx = try XCTUnwrap(CGContext(
                data: &refRGBA, width: decoded.width, height: decoded.height,
                bitsPerComponent: 8, bytesPerRow: decoded.width * 4, space: cs,
                bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
            ))
            refCtx.draw(refThumb, in: CGRect(x: 0, y: 0, width: decoded.width, height: decoded.height))
            for p in 0..<(decoded.width * decoded.height) {
                XCTAssertEqual(decoded.rgb[p * 3 + 0], refRGBA[p * 4 + 0], "orientation \(orientation) R at \(p)")
                XCTAssertEqual(decoded.rgb[p * 3 + 1], refRGBA[p * 4 + 1], "orientation \(orientation) G at \(p)")
                XCTAssertEqual(decoded.rgb[p * 3 + 2], refRGBA[p * 4 + 2], "orientation \(orientation) B at \(p)")
            }

            // Smart-resized non-thumbnailed path (upscaled) and thumbnailed path (downscaled) must agree on orientation.
            let smart = try QWImageDecoder.decodeSmartResized(urlOrDataUri: "data:image/tiff;base64,\(data.base64EncodedString())")
            if swaps {
                XCTAssertGreaterThan(smart.image.height, smart.image.width, "orientation \(orientation) must preserve portrait")
            } else {
                XCTAssertGreaterThan(smart.image.width, smart.image.height, "orientation \(orientation) must preserve landscape")
            }
        }

        // Thumbnailed EXIF orientation 6 downscale (128x64 raw -> oriented 64x128 -> smartResize 32x64).
        let bigW = 128, bigH = 64
        var bigRGBA = [UInt8](repeating: 255, count: bigW * bigH * 4)
        for y in 0..<bigH {
            for x in 0..<bigW {
                let i = (y * bigW + x) * 4
                bigRGBA[i] = x < bigW / 2 ? 255 : 0
                bigRGBA[i + 1] = 0
                bigRGBA[i + 2] = x < bigW / 2 ? 0 : 255
                bigRGBA[i + 3] = 255
            }
        }
        let bigCtx = try XCTUnwrap(CGContext(
            data: &bigRGBA, width: bigW, height: bigH, bitsPerComponent: 8, bytesPerRow: bigW * 4,
            space: cs, bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        ))
        let bigCG = try XCTUnwrap(bigCtx.makeImage())
        let bigTiff = NSMutableData()
        let bigDest = try XCTUnwrap(CGImageDestinationCreateWithData(bigTiff as CFMutableData, UTType.tiff.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(bigDest, bigCG, [kCGImagePropertyOrientation: 6] as CFDictionary)
        XCTAssertTrue(CGImageDestinationFinalize(bigDest))
        let bigThumb = try QWImageDecoder.decodeSmartResized(
            urlOrDataUri: "data:image/tiff;base64,\((bigTiff as Data).base64EncodedString())",
            maxPixels: 2048
        )
        XCTAssertEqual(bigThumb.image.width, 32)
        XCTAssertEqual(bigThumb.image.height, 64)
        // Orientation 6 rotates left half (Red) to top half, right half (Blue) to bottom half.
        XCTAssertGreaterThan(bigThumb.image.rgb[0], 200)
        XCTAssertLessThan(bigThumb.image.rgb[2], 50)
        let bottomIdx = (63 * 32 + 16) * 3
        XCTAssertLessThan(bigThumb.image.rgb[bottomIdx], 50)
        XCTAssertGreaterThan(bigThumb.image.rgb[bottomIdx + 2], 200)

        // Transparent RGBA pixel (0,0,0,0) must composite to white (255,255,255), not black (0,0,0).
        var alphaBuf = [UInt8](repeating: 0, count: 4 * 4 * 4)
        // Pixel (0,0): fully transparent black -> white. Pixel (1,0): 50% red premultiplied (128,0,0,128).
        alphaBuf[4] = 128; alphaBuf[7] = 128
        let alphaUri = try makePngDataUri(width: 4, height: 4, rgbaBytes: alphaBuf)
        let alphaDecoded = try QWImageDecoder.decode(urlOrDataUri: alphaUri)
        XCTAssertEqual(alphaDecoded.rgb[0], 255)
        XCTAssertEqual(alphaDecoded.rgb[1], 255)
        XCTAssertEqual(alphaDecoded.rgb[2], 255)
        XCTAssertEqual(alphaDecoded.rgb[3], 255)
        XCTAssertGreaterThan(alphaDecoded.rgb[4], 100)
        XCTAssertGreaterThan(alphaDecoded.rgb[5], 100)
    }
}
