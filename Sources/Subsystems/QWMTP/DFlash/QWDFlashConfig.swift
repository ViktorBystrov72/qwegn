// QWDFlashConfig.swift — DFlash 2 / DSpine configuration, error type,
// session KV slot layout, and model-matched pack discovery.
import Foundation
import QWAlloc
import QWCore
import QWQuant

public enum QWDFlashError: Error, CustomStringConvertible, Sendable {
    case packMissing(String)
    case weightMissing(String, String)
    case state(String)

    public var description: String {
        switch self {
        case .packMissing(let p): return "QWDFlash: pack not found at \(p)"
        case .weightMissing(let name, let detail): return "QWDFlash: weight \(name) invalid (\(detail))"
        case .state(let msg): return msg
        }
    }
}

public struct QWDFlashConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let numLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let slidingWindow: Int
    public let maxCacheSize: Int
    public let blockSize: Int
    public let maskTokenId: Int
    public let targetLayerIds: [Int]
    public let convKernelSize: Int
    public let convGroupSize: Int
    public let selectorRank: Int
    public let selectorTopK: Int
    public let isDSpine: Bool
    public let draftBlockCausal: Bool
    public let bridgeCodeSize: Int
    public let bridgeRank: Int
    public let bridgeEps: Float
    public let virtualLayerCapture: Bool

    public init(
        hiddenSize: Int = 5120, numLayers: Int = 5,
        numAttentionHeads: Int = 32, numKeyValueHeads: Int = 8, headDim: Int = 128,
        intermediateSize: Int = 17408, vocabSize: Int = 248320,
        rmsNormEps: Float = 1e-6, ropeTheta: Float = 10_000_000.0,
        slidingWindow: Int = 2048, blockSize: Int = 8, maskTokenId: Int = 248070,
        targetLayerIds: [Int] = [5, 19, 33, 47, 61],
        convKernelSize: Int = 2, convGroupSize: Int = 16,
        selectorRank: Int = 256, selectorTopK: Int = 16,
        isDSpine: Bool = false, draftBlockCausal: Bool = false,
        bridgeCodeSize: Int = 1024, bridgeRank: Int = 512, bridgeEps: Float = 1e-6,
        virtualLayerCapture: Bool = false
    ) {
        self.hiddenSize = hiddenSize; self.numLayers = numLayers
        self.numAttentionHeads = numAttentionHeads; self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim; self.intermediateSize = intermediateSize; self.vocabSize = vocabSize
        self.rmsNormEps = rmsNormEps; self.ropeTheta = ropeTheta; self.slidingWindow = slidingWindow
        self.maxCacheSize = max(1, slidingWindow - 1); self.blockSize = blockSize
        self.maskTokenId = maskTokenId; self.targetLayerIds = targetLayerIds
        self.convKernelSize = convKernelSize; self.convGroupSize = convGroupSize
        self.selectorRank = selectorRank; self.selectorTopK = min(max(1, selectorTopK), 16)
        self.isDSpine = isDSpine; self.draftBlockCausal = draftBlockCausal
        self.bridgeCodeSize = bridgeCodeSize; self.bridgeRank = bridgeRank; self.bridgeEps = bridgeEps
        self.virtualLayerCapture = virtualLayerCapture
    }

    public struct PrefillCaptureSlice: Sendable, Equatable {
        public let localStart: Int
        public let captureRows: Int
        public let ctxStartPos: Int

        public init(localStart: Int, captureRows: Int, ctxStartPos: Int) {
            self.localStart = localStart
            self.captureRows = captureRows
            self.ctxStartPos = ctxStartPos
        }
    }

    /// Computes the live `[localStart..<localStart+captureRows]` slice intersecting `[windowEnd-maxCacheSize..<windowEnd]`.
    public static func prefillCaptureSlice(
        startPos: Int, rows: Int, windowEnd: Int, maxCacheSize: Int
    ) -> PrefillCaptureSlice? {
        guard rows > 0, maxCacheSize > 0 else { return nil }
        let windowStart = max(0, max(windowEnd, startPos + rows) - maxCacheSize)
        guard startPos + rows > windowStart else { return nil }
        let rawLocal = max(0, windowStart - startPos)
        let rawCount = rows - rawLocal
        guard rawCount > 0 else { return nil }
        let captureRows = min(rawCount, maxCacheSize)
        let localStart = rawLocal + (rawCount - captureRows)
        return PrefillCaptureSlice(
            localStart: localStart, captureRows: captureRows, ctxStartPos: startPos + localStart)
    }

    /// True when capture indices follow virtual `executionPlan` layers (`0..<executionPlan.count`) on looped targets.
    public func usesVirtualCapture(for target: QWModelDescriptor) -> Bool {
        virtualLayerCapture || (target.executionPlan.count > target.numLayers
            && targetLayerIds.contains { $0 >= target.numLayers })
    }

    public static let defaultDSpinePackPath: String =
        "\(FileManager.default.homeDirectoryForCurrentUser.path)/.qwg/models/Qwen3.8-27B-DSpine-b16-hq4g32"

    public static let defaultPackPath: String = {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let dflash = "\(home)/.qwg/models/Qwen3.8-27B-DFlash2-hq4g32"
        let dspine = "\(home)/.qwg/models/Qwen3.8-27B-DSpine-b16-hq4g32"
        if !FileManager.default.fileExists(atPath: dflash) && FileManager.default.fileExists(atPath: dspine) {
            return dspine
        }
        return dflash
    }()

    /// Scans `~/.qwg/models` (or `defaultPackPath` if nil) for a compatible DFlash2/DSpine pack.
    public static func defaultPackPath(for target: QWModelDescriptor?) -> String {
        defaultPackPath(for: target, modelsDirectory: URL(fileURLWithPath: QWPaths.modelsDirectoryPath))
    }

    public static func defaultPackPath(for target: QWModelDescriptor?, modelsDirectory: URL) -> String {
        guard let target else { return defaultPackPath }
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: modelsDirectory, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]
        ) else {
            return defaultPackPath
        }
        var dflashMatches: [String] = []
        var dspineMatches: [String] = []
        for url in entries.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let isDir = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
            guard isDir || url.pathExtension == "qwp", isDrafterPack(at: url) else { continue }
            let cfg = load(from: url)
            let layerBound = cfg.usesVirtualCapture(for: target)
                ? max(target.numLayers, target.executionPlan.count)
                : target.numLayers
            let compatible = cfg.hiddenSize == target.hidden
                && cfg.vocabSize == target.vocab
                && cfg.targetLayerIds.count == cfg.numLayers
                && cfg.targetLayerIds.allSatisfy { (0..<layerBound).contains($0) }
            guard compatible else { continue }
            if cfg.isDSpine {
                dspineMatches.append(url.path)
            } else {
                dflashMatches.append(url.path)
            }
        }
        if dflashMatches.contains(defaultPackPath) { return defaultPackPath }
        if let first = dflashMatches.sorted(by: { lhs, rhs in
            let lr = Self.drafterQuantPreferenceRank(lhs)
            let rr = Self.drafterQuantPreferenceRank(rhs)
            return lr != rr ? lr > rr : lhs < rhs
        }).first { return first }
        if let first = dspineMatches.first { return first }
        return defaultPackPath
    }

    /// Ranks compatible drafter packs so pure 4-bit SHVQ (`-0.8 GiB` UMA, top code tok/s) precedes HQ4 and mixed sub-4-bit packs.
    public static func drafterQuantPreferenceRank(_ packPath: String) -> Int {
        let lower = URL(fileURLWithPath: packPath).lastPathComponent.lowercased()
        let hasSub4Bit = lower.contains("shvq3") || lower.contains("shvq2") || lower.contains("shvq15")
        if lower.contains("shvq4") && !hasSub4Bit { return 2 }
        return hasSub4Bit ? 0 : 1
    }

    /// Validates that `targetDescriptor` is compatible with DFlash 2 / DSpine and that the drafter pack
    /// at `packPath` exists, has valid configuration, and contains the required weight tensors.
    public static func validateDrafter(
        targetDescriptor: QWModelDescriptor,
        packPath: String?,
        drafts: Int = 7,
        selectorTopK: Int? = nil
    ) throws {
        let rawPath = (packPath ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let defaultForTarget = Self.defaultPackPath(for: targetDescriptor)
        let effectivePath = (rawPath.isEmpty ? defaultForTarget : rawPath).expandingTildePath
        let resolvedURL = effectivePath.canonicalFileURL
        guard FileManager.default.fileExists(atPath: resolvedURL.path) else {
            throw QWDFlashError.packMissing(resolvedURL.path)
        }
        let cfg = Self.load(from: resolvedURL, topKOverride: selectorTopK)
        let maxDrafts = max(7, cfg.blockSize - 1)
        guard (1...maxDrafts).contains(drafts) else {
            throw QWDFlashError.state(
                "DFlash2 drafter requires mtpDrafts in 1...\(maxDrafts) (got \(drafts))"
            )
        }
        guard cfg.hiddenSize == targetDescriptor.hidden else {
            throw QWDFlashError.state(
                "DFlash2 drafter hiddenSize (\(cfg.hiddenSize)) does not match target (\(targetDescriptor.hidden))"
            )
        }
        guard cfg.vocabSize == targetDescriptor.vocab else {
            throw QWDFlashError.state(
                "DFlash2 drafter vocabSize (\(cfg.vocabSize)) does not match target (\(targetDescriptor.vocab))"
            )
        }
        let layerBound = cfg.usesVirtualCapture(for: targetDescriptor)
            ? max(targetDescriptor.numLayers, targetDescriptor.executionPlan.count)
            : targetDescriptor.numLayers
        guard cfg.targetLayerIds.count == cfg.numLayers,
              cfg.targetLayerIds.allSatisfy({ (0..<layerBound).contains($0) }) else {
            throw QWDFlashError.state(
                "DFlash2 drafter targetLayerIds \(cfg.targetLayerIds) out of range for \(layerBound)-layer target"
            )
        }
        guard cfg.headDim == 128 else {
            throw QWDFlashError.state("DFlash2 drafter headDim must be 128 (got \(cfg.headDim))")
        }
        guard cfg.vocabSize <= 393_216 else {
            throw QWDFlashError.state("DFlash2 drafter vocabSize \(cfg.vocabSize) exceeds 393216")
        }
        guard let cfgURL = QWModelRoot(url: resolvedURL).configURL,
              let cfgData = try? Data(contentsOf: cfgURL),
              let drafterDesc = try? QWModelDescriptor.decodeDrafter(configJSON: cfgData) else {
            throw QWDFlashError.state("DFlash2 drafter missing valid config.json at \(resolvedURL.path)")
        }
        let pack = try QWQuant.Pack(url: resolvedURL, descriptor: drafterDesc, skipIdentityCheck: true)
        var required = ["fc.weight", "hidden_norm.weight", "norm.weight", "layers.0.self_attn.q_proj.weight"]
        if cfg.isDSpine {
            required += ["selected_prefix_bridge.codebook", "selected_prefix_bridge.read_weight",
                         "selected_prefix_bridge.feature_write_weights"]
        }
        for name in required {
            _ = try pack.entry(name)
        }
    }

    /// True when `packURL` holds a DFlash 2 or DSpine drafter `config.json`.
    public static func isDrafterPack(at packURL: URL) -> Bool {
        let root = QWModelRoot(url: packURL)
        guard let cfgURL = root.configURL,
              let data = try? Data(contentsOf: cfgURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return false
        }
        if json["dflash_config"] != nil || json["selected_prefix_code_bridge"] != nil {
            return true
        }
        if let mt = (json["model_type"] as? String)?.lowercased(),
           mt.contains("dflash") || mt.contains("dspine") {
            return true
        }
        return false
    }

    /// Infers `"dspine"` vs `"dflash2"` and maximum draft count (`blockSize - 1`) from the drafter pack's `config.json`.
    public static func inferBackendAndMaxDrafts(packPath: String?) -> (backend: String, maxDrafts: Int) {
        let rawPath = (packPath ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let effectivePath = (rawPath.isEmpty ? Self.defaultPackPath : rawPath).expandingTildePath
        let url = effectivePath.canonicalFileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            return ("dflash2", 7)
        }
        let cfg = Self.load(from: url)
        return (cfg.isDSpine ? "dspine" : "dflash2", max(7, min(15, cfg.blockSize - 1)))
    }

    public static func load(from packURL: URL, topKOverride: Int? = nil) -> QWDFlashConfig {
        let dir = packURL.pathExtension == "qwp" ? packURL.deletingLastPathComponent() : packURL
        let cfgURL = dir.appendingPathComponent("assets/config.json")
        guard let data = try? Data(contentsOf: cfgURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return QWDFlashConfig(selectorTopK: topKOverride ?? 16)
        }
        let dflash = json["dflash_config"] as? [String: Any] ?? [:]
        let rope = (json["rope_parameters"] as? [String: Any]) ?? (json["rope_scaling"] as? [String: Any]) ?? [:]
        let bridgeDict = (dflash["selected_prefix_code_bridge"] as? [String: Any])
            ?? (json["selected_prefix_code_bridge"] as? [String: Any])
        let isDSpine = bridgeDict != nil || (json["selected_prefix_code_bridge"] as? Bool == true)
        let refineDict = dflash["selected_prefix_last_write_refine"] as? [String: Any]
        let secondPass = json["code_bridge_second_pass"] as? [String: Any]
        let topK = topKOverride
            ?? (refineDict?["head_topk"] as? Int)
            ?? (secondPass?["head_topk"] as? Int)
            ?? (dflash["selector_top_k"] as? Int)
            ?? 16
        return QWDFlashConfig(
            hiddenSize: (json["hidden_size"] as? Int) ?? 5120,
            numLayers: (json["num_hidden_layers"] as? Int) ?? 5,
            numAttentionHeads: (json["num_attention_heads"] as? Int) ?? 32,
            numKeyValueHeads: (json["num_key_value_heads"] as? Int) ?? 8,
            headDim: (json["head_dim"] as? Int) ?? 128,
            intermediateSize: (json["intermediate_size"] as? Int) ?? 17408,
            vocabSize: (json["vocab_size"] as? Int) ?? 248320,
            rmsNormEps: Float((json["rms_norm_eps"] as? Double) ?? 1e-6),
            ropeTheta: Float((json["rope_theta"] as? Double) ?? (rope["rope_theta"] as? Double) ?? 10_000_000.0),
            slidingWindow: (json["sliding_window"] as? Int) ?? (isDSpine ? 8192 : 2048),
            blockSize: (dflash["block_size"] as? Int) ?? (json["block_size"] as? Int) ?? (isDSpine ? 16 : 8),
            maskTokenId: (dflash["mask_token_id"] as? Int) ?? 248070,
            targetLayerIds: (dflash["target_layer_ids"] as? [Int]) ?? (isDSpine ? [1, 16, 31, 46, 61] : [5, 19, 33, 47, 61]),
            convKernelSize: (dflash["conv_kernel_size"] as? Int) ?? (isDSpine ? 0 : 2),
            convGroupSize: (dflash["conv_group_size"] as? Int) ?? 16,
            selectorRank: (dflash["selector_rank"] as? Int) ?? 256,
            selectorTopK: topK,
            isDSpine: isDSpine,
            draftBlockCausal: (dflash["draft_block_causal"] as? Bool) ?? isDSpine,
            bridgeCodeSize: (bridgeDict?["code_size"] as? Int) ?? (json["selected_prefix_code_size"] as? Int) ?? 1024,
            bridgeRank: (bridgeDict?["rank"] as? Int) ?? (json["selected_prefix_rank"] as? Int) ?? 512,
            bridgeEps: Float((bridgeDict?["eps"] as? Double) ?? 1e-6),
            virtualLayerCapture: (dflash["virtual_layer_capture"] as? Bool)
                ?? (json["virtual_layer_capture"] as? Bool) ?? false
        )
    }
}

public struct QWDFlashSessionSlot: Sendable {
    public let kSlab: QWTensorValue
    public let vSlab: QWTensorValue
    public let kLayers: [QWTensorValue]
    public let vLayers: [QWTensorValue]

    public init(kSlab: QWTensorValue, vSlab: QWTensorValue,
                kLayers: [QWTensorValue], vLayers: [QWTensorValue]) {
        self.kSlab = kSlab
        self.vSlab = vSlab
        self.kLayers = kLayers
        self.vLayers = vLayers
    }
}
