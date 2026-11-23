// QWVision+Ephemeral.swift — Vision backend protocol, reference adapter, and
// prompt-scoped on-demand residency controller (0 B steady-state text RAM).

import Foundation
import QWCore
import QWHAL
import QWQuant

/// Unified vision-tower contract: C19 patch slab in, merged visual tokens out.
public protocol QWVisionBackend: Sendable {
    func encode(patches: [Float], grid: QWVisionGridPlan,
                bins: QWVisionRef.InterpBins, posIds: [Int32]) async throws -> [Float]
    /// Prepares vision residency for the upcoming prompt: loads on demand when
    /// `hasImages == true` and unloads resident weights when `hasImages == false`.
    func prepareForPrompt(hasImages: Bool) throws
    /// Dynamically loads or unloads the vision tower weights in memory without restarting the LLM.
    func setVisionLoaded(_ loaded: Bool) throws
    /// True when vision tower weights are currently resident in memory.
    var isLoaded: Bool { get }
    /// True when vision tower weights are configured to stay resident across prompts.
    var keepResident: Bool { get }
    /// True when on-demand vision loading is permitted for image turns.
    var allowOnDemand: Bool { get }
    /// True when the backend is permitted to serve image requests (resident, loaded, or on-demand).
    var isVisionEnabled: Bool { get }
    /// Config-sourced vision tower geometry served by this backend.
    var visionConfig: QWVisionConfig { get }
}

extension QWVisionBackend {
    public func prepareForPrompt(hasImages: Bool) throws {}
    public func setVisionLoaded(_ loaded: Bool) throws {}
    public var isLoaded: Bool { true }
    public var keepResident: Bool { isLoaded }
    public var allowOnDemand: Bool { true }
    public var isVisionEnabled: Bool { isLoaded || keepResident || allowOnDemand }
    public var visionConfig: QWVisionConfig { .qwen3VLReference }
}

extension QWVisionWeights {
    /// Total bytes occupied by the widened fp32 vision tower arrays in RAM.
    public var residentBytes: Int {
        let blockFloats = blocks.reduce(0) { acc, b in
            acc + b.norm1G.count + b.norm1B.count + b.norm2G.count + b.norm2B.count
                + b.qkvW.count + b.qkvB.count + b.projW.count + b.projB.count
                + b.l1W.count + b.l1B.count + b.l2W.count + b.l2B.count
        }
        let topFloats = posEmbed.count + patchW.count + patchB.count
            + mergerGamma.count + mergerBeta.count
            + fc1W.count + fc1B.count + fc2W.count + fc2B.count + invFreq.count
        return (blockFloats + topFloats) * MemoryLayout<Float>.stride
    }
}

/// Prompt-scoped on-demand vision backend: unloaded by default (0 B resident),
/// loaded once when a prompt has images, and unloaded when a subsequent prompt has no images.
public final class QWVisionOnDemandBackend: QWVisionBackend, @unchecked Sendable {
    public enum Source: Sendable {
        case pack(String)
        case directory(String)
        case custom(@Sendable () throws -> QWVisionWeights)
    }

    public let source: Source
    private let lock = NSLock()
    private let preferredHal: QWHal?
    private var resolvedConfig: QWVisionConfig
    private var metalBackend: QWVisionMetalBackend?
    private var residentWeights: QWVisionWeights?
    private var inkling2Weights: QWInkling2VisionWeights?
    private var keepResidentFlag: Bool = false
    private var allowOnDemandFlag: Bool = true

    public var visionConfig: QWVisionConfig {
        lock.withLock { resolvedConfig }
    }

    public var isLoaded: Bool {
        lock.withLock { metalBackend != nil || residentWeights != nil || inkling2Weights != nil }
    }

    public var keepResident: Bool {
        lock.withLock { keepResidentFlag }
    }

    public var allowOnDemand: Bool {
        lock.withLock { allowOnDemandFlag }
    }

    public var isVisionEnabled: Bool {
        lock.withLock {
            keepResidentFlag || allowOnDemandFlag || metalBackend != nil || residentWeights != nil || inkling2Weights != nil
        }
    }

    public var residentBytes: Int {
        lock.withLock {
            (metalBackend?.residentBytes ?? 0)
                + (residentWeights?.residentBytes ?? 0)
                + (inkling2Weights?.residentBytes ?? 0)
        }
    }

    public init(
        source: Source,
        keepResident: Bool = false,
        allowOnDemand: Bool = true,
        hal: QWHal? = nil
    ) throws {
        self.source = source
        self.preferredHal = hal
        self.keepResidentFlag = keepResident
        self.allowOnDemandFlag = allowOnDemand
        switch source {
        case .pack(let packPath):
            let ck = try QWVisionCheckpoint(pack: packPath)
            try Self.validate(checkpoint: ck)
            self.resolvedConfig = ck.config
        case .directory(let dir):
            let ck = try QWVisionCheckpoint(dir: dir)
            try Self.validate(checkpoint: ck)
            self.resolvedConfig = ck.config
        case .custom:
            self.resolvedConfig = .qwen3VLReference
        }
        if keepResident {
            try loadResidentLocked()
        }
    }

    /// Resolves a model path (`.qwp` pack or `.safetensors` directory/file) into an on-demand backend.
    public convenience init(
        modelPath: String,
        keepResident: Bool = false,
        allowOnDemand: Bool = true,
        hal: QWHal? = nil
    ) throws {
        let url = URL(fileURLWithPath: modelPath)
        let root = QWModelRoot(url: url)
        switch root.kind {
        case .qwp1:
            try self.init(source: .pack(root.weights.path), keepResident: keepResident, allowOnDemand: allowOnDemand, hal: hal)
        case .safetensorsSingle, .safetensorsSharded:
            let dir = root.weights.hasDirectoryPath
                ? root.weights.path
                : root.weights.deletingLastPathComponent().path
            try self.init(source: .directory(dir), keepResident: keepResident, allowOnDemand: allowOnDemand, hal: hal)
        case nil:
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: modelPath, isDirectory: &isDir), isDir.boolValue {
                try self.init(source: .directory(modelPath), keepResident: keepResident, allowOnDemand: allowOnDemand, hal: hal)
            } else {
                try self.init(source: .pack(modelPath), keepResident: keepResident, allowOnDemand: allowOnDemand, hal: hal)
            }
        }
    }

    /// True when the checkpoint at `modelPath` carries valid vision tower weights.
    public static func hasVisionWeights(modelPath: String) -> Bool {
        (try? QWVisionOnDemandBackend(modelPath: modelPath)) != nil
    }

    private static func shouldUseMetal(hal: QWHal?) -> Bool {
        if let hal {
            return !hal.caps.name.localizedCaseInsensitiveContains("paravirtual")
        }
        if ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] == "true" {
            return false
        }
        guard let caps = QWDeviceCaps.current else { return false }
        return !caps.name.localizedCaseInsensitiveContains("paravirtual")
    }

    private static func validate(checkpoint ck: QWVisionCheckpoint) throws {
        defer { ck.releaseBacking() }
        let cfg = ck.config
        guard cfg.dim > 0, cfg.heads > 0, cfg.headDim * cfg.heads == cfg.dim,
              cfg.numPosEmbeds > 0, cfg.posSide * cfg.posSide == cfg.numPosEmbeds else {
            throw QWVisionError.badLayout("vision config incomplete or invalid")
        }
        if cfg.architecture == .inkling2 {
            guard ck.has("embeddings.patch_embedding.weight"),
                  ck.has("embeddings.position_embedding"),
                  ck.has("embeddings.class_embedding"),
                  ck.has("embeddings.cls_position_embedding"),
                  ck.has("pre_layernorm.weight"),
                  ck.has("layers.0.layernorm_before.weight"),
                  ck.has("proj_in.weight"),
                  ck.has("proj_out.weight") else {
                throw QWVisionError.missingTensor("inkling2 vision tensors missing")
            }
            return
        }
        guard ck.shape("pos_embed.weight") == [cfg.numPosEmbeds, cfg.dim] else {
            throw QWVisionError.missingTensor("pos_embed.weight")
        }
        let pw = ck.shape("patch_embed.proj.weight")
        let baseShape = [cfg.dim, 3, cfg.temporal, cfg.patch, cfg.patch]
        let legacyShape = [cfg.dim, cfg.temporal, cfg.patch, cfg.patch, 3]
        guard pw == baseShape || pw == legacyShape else {
            throw QWVisionError.badLayout("patch_embed.proj.weight \(pw.debugDescription) invalid")
        }
        guard ck.has("blocks.0.norm1.weight"),
              ck.has("merger.norm.weight"),
              ck.has("merger.norm.bias"),
              ck.has("merger.linear_fc1.weight"),
              ck.has("merger.linear_fc2.weight") else {
            throw QWVisionError.missingTensor("vision blocks/merger missing")
        }
    }

    private func makeInkling2WeightsLocked() throws -> QWInkling2VisionWeights {
        switch source {
        case .pack(let packPath):
            return try QWInkling2VisionWeights(checkpoint: try QWVisionCheckpoint(pack: packPath))
        case .directory(let dir):
            return try QWInkling2VisionWeights(checkpoint: try QWVisionCheckpoint(dir: dir))
        case .custom:
            throw QWVisionError.badLayout("custom builder unsupported for inkling2 vision")
        }
    }

    private func loadCPUWeightsFromSource() throws -> QWVisionWeights {
        switch source {
        case .pack(let packPath):
            return try QWVisionWeights(pack: packPath)
        case .directory(let dir):
            return try QWVisionWeights(dir: dir)
        case .custom(let builder):
            return try builder()
        }
    }

    private func loadResidentLocked() throws {
        if metalBackend != nil || residentWeights != nil || inkling2Weights != nil {
            return
        }
        if resolvedConfig.architecture == .inkling2 {
            let loaded = try makeInkling2WeightsLocked()
            resolvedConfig = loaded.config
            inkling2Weights = loaded
            return
        }
        let forceCPU = ProcessInfo.processInfo.environment["QWG_VISION_CPU"] == "1"
        let canTryMetal = !forceCPU && Self.shouldUseMetal(hal: preferredHal)
        if canTryMetal {
            switch source {
            case .pack(let packPath):
                let ck = try QWVisionCheckpoint(pack: packPath)
                resolvedConfig = ck.config
                do {
                    metalBackend = try QWVisionMetalBackend(checkpoint: ck, hal: preferredHal)
                    return
                } catch is QWHalError {
                    residentWeights = try QWVisionWeights(checkpoint: ck)
                    return
                }
            case .directory(let dir):
                let ck = try QWVisionCheckpoint(dir: dir)
                resolvedConfig = ck.config
                do {
                    metalBackend = try QWVisionMetalBackend(checkpoint: ck, hal: preferredHal)
                    return
                } catch is QWHalError {
                    residentWeights = try QWVisionWeights(checkpoint: ck)
                    return
                }
            case .custom(let builder):
                let loaded = try builder()
                resolvedConfig = loaded.config
                do {
                    metalBackend = try QWVisionMetalBackend(weights: loaded, hal: preferredHal)
                    return
                } catch is QWHalError {
                    residentWeights = loaded
                    return
                }
            }
        }
        let loaded = try loadCPUWeightsFromSource()
        resolvedConfig = loaded.config
        residentWeights = loaded
    }

    private func unloadLocked() {
        metalBackend = nil
        residentWeights = nil
        inkling2Weights = nil
    }

    public func setVisionLoaded(_ loaded: Bool) throws {
        try lock.withLock {
            keepResidentFlag = loaded
            if loaded {
                try loadResidentLocked()
            } else {
                unloadLocked()
            }
        }
    }

    public func prepareForPrompt(hasImages: Bool) throws {
        try lock.withLock {
            if hasImages {
                guard keepResidentFlag || allowOnDemandFlag || metalBackend != nil || residentWeights != nil || inkling2Weights != nil else {
                    throw QWVisionError.badLayout(
                        "Vision tower is disabled in Model Settings (enable 'Load Vision Tower' in Settings → Model)"
                    )
                }
                try loadResidentLocked()
            } else if !keepResidentFlag {
                unloadLocked()
            } else {
                metalBackend?.releaseScratch()
            }
        }
    }

    public func unload() {
        lock.withLock {
            keepResidentFlag = false
            unloadLocked()
        }
    }

    public func encodeSync(
        patches: [Float],
        grid: QWVisionGridPlan,
        bins: QWVisionRef.InterpBins,
        posIds: [Int32]
    ) throws -> [Float] {
        let forceCPU = ProcessInfo.processInfo.environment["QWG_VISION_CPU"] == "1"
        enum EngineSelection {
            case metal(QWVisionMetalBackend)
            case cpu(QWVisionWeights)
            case inkling2(QWInkling2VisionWeights)
        }
        let selection: EngineSelection = try lock.withLock {
            guard keepResidentFlag || allowOnDemandFlag || metalBackend != nil || residentWeights != nil || inkling2Weights != nil else {
                throw QWVisionError.badLayout(
                    "Vision tower is disabled in Model Settings (enable 'Load Vision Tower' in Settings → Model)"
                )
            }
            if resolvedConfig.architecture == .inkling2 {
                if inkling2Weights == nil {
                    inkling2Weights = try makeInkling2WeightsLocked()
                }
                guard let iw = inkling2Weights else {
                    throw QWVisionError.badLayout("Inkling-2 vision weights failed to load")
                }
                return .inkling2(iw)
            }
            if forceCPU {
                if let existing = residentWeights {
                    return .cpu(existing)
                }
                let cpuW = try loadCPUWeightsFromSource()
                resolvedConfig = cpuW.config
                if metalBackend == nil {
                    residentWeights = cpuW
                }
                return .cpu(cpuW)
            }
            try loadResidentLocked()
            if let metal = metalBackend {
                return .metal(metal)
            }
            guard let weights = residentWeights else {
                throw QWVisionError.badLayout("Vision weights failed to load")
            }
            return .cpu(weights)
        }
        switch selection {
        case .metal(let metal):
            return try metal.encodeSync(patches: patches, grid: grid, bins: bins, posIds: posIds)
        case .cpu(let weights):
            return QWVisionRef(weights: weights).encode(patches: patches, plan: grid, bins: bins, posIds: posIds)
        case .inkling2(let iw):
            return QWInkling2VisionRef(weights: iw).encode(patches: patches, plan: grid, bins: bins, posIds: posIds)
        }
    }

    public func encode(
        patches: [Float],
        grid: QWVisionGridPlan,
        bins: QWVisionRef.InterpBins,
        posIds: [Int32]
    ) async throws -> [Float] {
        try encodeSync(patches: patches, grid: grid, bins: bins, posIds: posIds)
    }
}
