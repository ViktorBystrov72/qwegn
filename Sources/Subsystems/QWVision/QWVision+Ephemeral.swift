// QWVision+Ephemeral.swift — Vision backend protocol, reference adapter, and
// prompt-scoped on-demand residency controller (0 B steady-state text RAM).

import Foundation
import QWCore
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
    /// Config-sourced vision tower geometry served by this backend.
    var visionConfig: QWVisionConfig { get }
}

extension QWVisionBackend {
    public func prepareForPrompt(hasImages: Bool) throws {}
    public func setVisionLoaded(_ loaded: Bool) throws {}
    public var isLoaded: Bool { true }
    public var keepResident: Bool { isLoaded }
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
    private var resolvedConfig: QWVisionConfig
    private var residentWeights: QWVisionWeights?
    private var keepResidentFlag: Bool = false
    private var allowOnDemandFlag: Bool = true

    public var visionConfig: QWVisionConfig {
        lock.lock()
        defer { lock.unlock() }
        return resolvedConfig
    }

    public var isLoaded: Bool {
        lock.lock()
        defer { lock.unlock() }
        return residentWeights != nil
    }

    public var keepResident: Bool {
        lock.lock()
        defer { lock.unlock() }
        return keepResidentFlag
    }

    public var residentBytes: Int {
        lock.lock()
        defer { lock.unlock() }
        return residentWeights?.residentBytes ?? 0
    }

    public init(source: Source, keepResident: Bool = false, allowOnDemand: Bool = true) throws {
        self.source = source
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
            _ = try loadWeightsLocked()
        }
    }

    /// Resolves a model path (`.qwp` pack or `.safetensors` directory/file) into an on-demand backend.
    public convenience init(modelPath: String, keepResident: Bool = false, allowOnDemand: Bool = true) throws {
        let url = URL(fileURLWithPath: modelPath)
        let root = QWModelRoot(url: url)
        switch root.kind {
        case .qwp1:
            try self.init(source: .pack(root.weights.path), keepResident: keepResident, allowOnDemand: allowOnDemand)
        case .safetensorsSingle, .safetensorsSharded:
            let dir = root.weights.hasDirectoryPath
                ? root.weights.path
                : root.weights.deletingLastPathComponent().path
            try self.init(source: .directory(dir), keepResident: keepResident, allowOnDemand: allowOnDemand)
        case nil:
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: modelPath, isDirectory: &isDir), isDir.boolValue {
                try self.init(source: .directory(modelPath), keepResident: keepResident, allowOnDemand: allowOnDemand)
            } else {
                try self.init(source: .pack(modelPath), keepResident: keepResident, allowOnDemand: allowOnDemand)
            }
        }
    }

    /// True when the checkpoint at `modelPath` carries valid vision tower weights.
    public static func hasVisionWeights(modelPath: String) -> Bool {
        (try? QWVisionOnDemandBackend(modelPath: modelPath)) != nil
    }

    private static func validate(checkpoint ck: QWVisionCheckpoint) throws {
        defer {
            ck.shards.removeAll()
            ck.pack?.adviseDontNeed()
            ck.pack = nil
        }
        let cfg = ck.config
        guard cfg.dim > 0, cfg.heads > 0, cfg.headDim * cfg.heads == cfg.dim,
              cfg.numPosEmbeds > 0, cfg.posSide * cfg.posSide == cfg.numPosEmbeds else {
            throw QWVisionError.badLayout("vision config incomplete or invalid")
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

    private func loadWeightsLocked() throws -> QWVisionWeights {
        if let existing = residentWeights {
            return existing
        }
        let loaded: QWVisionWeights
        switch source {
        case .pack(let packPath):
            loaded = try QWVisionWeights(pack: packPath)
        case .directory(let dir):
            loaded = try QWVisionWeights(dir: dir)
        case .custom(let builder):
            loaded = try builder()
        }
        resolvedConfig = loaded.config
        residentWeights = loaded
        return loaded
    }

    public func setVisionLoaded(_ loaded: Bool) throws {
        lock.lock()
        defer { lock.unlock() }
        keepResidentFlag = loaded
        if loaded {
            _ = try loadWeightsLocked()
        } else if residentWeights != nil {
            residentWeights = nil
        }
    }

    public func prepareForPrompt(hasImages: Bool) throws {
        lock.lock()
        defer { lock.unlock() }
        if hasImages {
            guard keepResidentFlag || allowOnDemandFlag || residentWeights != nil else {
                throw QWVisionError.badLayout("Vision tower is disabled in Model Settings (enable 'Load Vision Tower' in Settings → Model)")
            }
            _ = try loadWeightsLocked()
        } else if !keepResidentFlag, residentWeights != nil {
            residentWeights = nil
        }
    }

    public func unload() {
        lock.lock()
        keepResidentFlag = false
        if residentWeights != nil {
            residentWeights = nil
        }
        lock.unlock()
    }

    public func encodeSync(
        patches: [Float],
        grid: QWVisionGridPlan,
        bins: QWVisionRef.InterpBins,
        posIds: [Int32]
    ) throws -> [Float] {
        let weights: QWVisionWeights = try {
            lock.lock()
            defer { lock.unlock() }
            return try loadWeightsLocked()
        }()
        let ref = QWVisionRef(weights: weights)
        return ref.encode(patches: patches, plan: grid, bins: bins, posIds: posIds)
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
