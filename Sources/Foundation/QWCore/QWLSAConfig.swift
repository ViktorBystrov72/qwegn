// QWLSAConfig.swift — Latent Sparse Attention (LSA, arXiv:2610.42147) configuration & geometry math.
// Pure: no Metal, no heavy I/O. Shared across QWCore, QWQuant, QWEngine, QWFlashNext, QWServer, and qwg.

import Foundation

/// User-facing Latent Sparse Attention execution mode (`lsaMode` knob).
public enum QWLSAMode: String, Sendable, Codable, CaseIterable, Equatable, Hashable {
    /// Disable LSA (use native dense attention or native QSA).
    case off = "off"
    /// Automatically select `.qlsa` sidecar if present, `qsa-cluster` on QSA models, or `svd` fallback.
    case auto = "auto"
    /// Require and use a `.qlsa` sidecar (`model.qlsa` or `--lsa-sidecar-path`).
    case sidecar = "sidecar"
    /// Use RoPE-free Weight-SVD index projections (from `.qlsa` sidecar if present, else synthesized at load).
    case svd = "svd"
    /// Cluster QSA 4-token pooled index keys directly via Ward clustering (Flash-Next native indexer).
    case qsaCluster = "qsa-cluster"

    public static func resolve(_ raw: String) -> QWLSAMode? {
        QWLSAMode(rawValue: raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
    }
}

/// Summary metadata of a discovered `.qlsa` sidecar file (`QLSA0001`).
public struct QWLSASidecarInfo: Sendable, Codable, Equatable, Hashable {
    public let path: String
    public let modelId: String
    public let geometryDigest: String
    public let method: String
    public let indexDim: Int
    public let poolSize: Int
    public let clusterBlockSize: Int
    public let bucketsPerBlock: Int
    public let maxBucketKeys: Int
    public let recommendedBudget: Int
    public let geometryMatches: Bool

    public init(
        path: String,
        modelId: String,
        geometryDigest: String,
        method: String,
        indexDim: Int = QWLSAConfig.defaultIndexDim,
        poolSize: Int = QWLSAConfig.poolSize,
        clusterBlockSize: Int = QWLSAConfig.defaultClusterBlock,
        bucketsPerBlock: Int = QWLSAConfig.bucketsPerBlock,
        maxBucketKeys: Int = QWLSAConfig.maxBucketKeys,
        recommendedBudget: Int = QWLSAConfig.defaultBudget,
        geometryMatches: Bool = true
    ) {
        self.path = path
        self.modelId = modelId
        self.geometryDigest = geometryDigest
        self.method = method
        self.indexDim = indexDim
        self.poolSize = poolSize
        self.clusterBlockSize = clusterBlockSize
        self.bucketsPerBlock = bucketsPerBlock
        self.maxBucketKeys = maxBucketKeys
        self.recommendedBudget = recommendedBudget
        self.geometryMatches = geometryMatches
    }
}

/// Complete Latent Sparse Attention runtime configuration.
public struct QWLSAConfig: Sendable, Codable, Equatable, Hashable {
    /// Fixed number of sink tokens always attended ($n_{\mathrm{sink}} = 4$, 1 four-token block).
    public static let sinkTokens: Int = 4
    /// Fixed local sliding window tokens always attended ($n_w = 512$, 128 four-token blocks).
    public static let windowTokens: Int = 512
    /// Initial pooling granularity ($n_{\mathrm{pool}} = 4$ consecutive tokens per pool block).
    public static let poolSize: Int = 4
    /// Default Ward clustering block size ($n_{\mathrm{cluster}} = 512$ tokens = 128 pool blocks).
    public static let defaultClusterBlock: Int = 512
    /// Capacity-balanced `e-means` macro-chunk size ($n_{\mathrm{cluster}} = 4096$ tokens = 1024 pool blocks).
    public static let emeansClusterBlock: Int = 4096
    /// Number of Ward buckets produced per cluster block ($m = 64$, average 8 keys / 2 pool blocks per bucket).
    public static let bucketsPerBlock: Int = 64
    /// Hard cap on keys per Ward bucket ($n_b^{\max} = 16$ keys = 4 four-token pool blocks -> fits in `uchar4`).
    public static let maxBucketKeys: Int = 16
    /// Default low-rank latent index dimension ($d_{\mathrm{idx}} = 64$).
    public static let defaultIndexDim: Int = 64
    /// Default sparse token budget ($k = 2048$ tokens = 512 four-token blocks).
    public static let defaultBudget: Int = 2048

    /// True for training-free SVD sidecar methods superseded by Full-Slice Q4 Direct when eligible.
    public static func isTrainingFreeSidecarMethod(_ method: String) -> Bool {
        ["weight_svd", "weight_svd_v1", "svd", "head_svd"].contains(
            method.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        )
    }

    /// True when geometry supports Full-Slice Q4 Multi-Head Direct LSA (2-stage p16->p4 + all_sum).
    public static func supportsFullSliceQ4Direct(qHeads: Int, kvHeads: Int, sliceDim: Int) -> Bool {
        guard kvHeads >= 1, qHeads >= kvHeads, qHeads <= 64, qHeads % kvHeads == 0,
              sliceDim >= 4, sliceDim <= 256, sliceDim % 4 == 0 else { return false }
        return qHeads * sliceDim <= 12288
    }

    public var mode: QWLSAMode
    public var budget: Int
    public var clusterBlock: Int
    public var sidecarPath: String
    /// Opt-in cross-loop LSA selection sharing across recurrent visits to the same physical layer.
    public var crossLoopReuse: Bool
    /// Recompute interval across visits to the same physical layer (`0` = reuse first visit across all passes).
    public var loopShareStride: Int
    /// Whether `crossLoopReuse` was explicitly provided (`false` overrides `QW_LSA_LOOP_SHARE`).
    public var hasExplicitLoopShareOverride: Bool
    /// Whether `budget` was explicitly pinned rather than defaulting to `defaultBudget`.
    public var hasExplicitBudgetOverride: Bool

    public static let loopShareEnvKey = "QW_LSA_LOOP_SHARE"

    public var isEnabled: Bool { mode != .off }

    public init(
        mode: QWLSAMode = .off,
        budget: Int = QWLSAConfig.defaultBudget,
        clusterBlock: Int = QWLSAConfig.defaultClusterBlock,
        sidecarPath: String = "",
        crossLoopReuse: Bool? = nil,
        loopShareStride: Int = 0,
        explicitBudget: Bool? = nil
    ) {
        self.mode = mode
        self.budget = budget
        self.clusterBlock = clusterBlock
        self.sidecarPath = sidecarPath
        self.crossLoopReuse = crossLoopReuse ?? false
        self.loopShareStride = max(0, loopShareStride)
        self.hasExplicitLoopShareOverride = (crossLoopReuse != nil)
        self.hasExplicitBudgetOverride = explicitBudget ?? (budget != Self.defaultBudget)
    }

    public static func == (lhs: QWLSAConfig, rhs: QWLSAConfig) -> Bool {
        lhs.mode == rhs.mode && lhs.budget == rhs.budget
            && lhs.clusterBlock == rhs.clusterBlock && lhs.sidecarPath == rhs.sidecarPath
            && lhs.crossLoopReuse == rhs.crossLoopReuse && lhs.loopShareStride == rhs.loopShareStride
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(mode)
        hasher.combine(budget)
        hasher.combine(clusterBlock)
        hasher.combine(sidecarPath)
        hasher.combine(crossLoopReuse)
        hasher.combine(loopShareStride)
    }

    private enum CodingKeys: String, CodingKey {
        case mode, budget, clusterBlock, sidecarPath, crossLoopReuse, loopShareStride
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.mode = try c.decodeIfPresent(QWLSAMode.self, forKey: .mode) ?? .off
        let decodedBudget = try c.decodeIfPresent(Int.self, forKey: .budget)
        self.budget = decodedBudget ?? Self.defaultBudget
        self.clusterBlock = try c.decodeIfPresent(Int.self, forKey: .clusterBlock) ?? Self.defaultClusterBlock
        self.sidecarPath = try c.decodeIfPresent(String.self, forKey: .sidecarPath) ?? ""
        let decodedReuse = try c.decodeIfPresent(Bool.self, forKey: .crossLoopReuse)
        self.crossLoopReuse = decodedReuse ?? false
        self.loopShareStride = max(0, try c.decodeIfPresent(Int.self, forKey: .loopShareStride) ?? 0)
        self.hasExplicitLoopShareOverride = (decodedReuse != nil)
        self.hasExplicitBudgetOverride = (decodedBudget != nil && decodedBudget != Self.defaultBudget)
    }

    /// Context-scaled default budget maintaining >= 3.1% floor up to 256K and >= 1.56% at 512K..1M.
    public static func adaptiveBudget(forMaxContext maxContext: Int) -> Int {
        if maxContext >= 786_432 { return 16_384 }
        if maxContext >= 196_608 { return 8192 }
        if maxContext >= 98_304 { return 4096 }
        return defaultBudget
    }

    /// Scales default budget for `maxContext` unless the user pinned an explicit budget.
    public func resolvingAdaptiveBudget(forMaxContext maxContext: Int) -> QWLSAConfig {
        guard isEnabled, !hasExplicitBudgetOverride else { return self }
        let target = Self.adaptiveBudget(forMaxContext: maxContext)
        guard target != budget else { return self }
        var copy = self
        copy.budget = target
        return copy
    }

    public static func parseLoopShareEnv(_ raw: String?) -> (enabled: Bool, stride: Int)? {
        guard let raw else { return nil }
        let norm = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !norm.isEmpty else { return nil }
        if ["0", "false", "no", "off"].contains(norm) { return (false, 0) }
        if ["1", "true", "yes", "on", "all"].contains(norm) { return (true, 0) }
        let strideText = norm.hasPrefix("stride:") ? String(norm.dropFirst(7)) : norm
        if let n = Int(strideText), n >= 1 {
            return (true, n >= 2 ? n : 0)
        }
        return nil
    }

    public func resolvingLoopShareEnvironment(
        _ env: [String: String] = ProcessInfo.processInfo.environment
    ) -> QWLSAConfig {
        guard !hasExplicitLoopShareOverride, !crossLoopReuse,
              let parsed = Self.parseLoopShareEnv(env[Self.loopShareEnvKey]) else {
            return self
        }
        var copy = self
        copy.crossLoopReuse = parsed.enabled
        copy.loopShareStride = parsed.stride
        return copy
    }

    /// Selected token budget $k$, clamped to `512...16384` and aligned to a 4-token block boundary.
    public var alignedBudget: Int {
        let clamped = min(16_384, max(512, budget))
        return (clamped / Self.poolSize) * Self.poolSize
    }

    /// Number of 4-token blocks in the selected budget ($k / 4$).
    public var budgetBlocks: Int {
        alignedBudget / Self.poolSize
    }

    /// Cluster block size $n_{\mathrm{cluster}}$ aligned to a supported power-of-two (`256`, `512`, `1024`, or `4096`).
    public var alignedClusterBlock: Int {
        if clusterBlock <= 256 { return 256 }
        if clusterBlock >= 4096 { return 4096 }
        if clusterBlock >= 1024 { return 1024 }
        return 512
    }

    /// Maximum 4-token blocks emitted in `lsaBlockList` for one query token:
    /// 1 sink block + $k / 4$ selected middle blocks + $(n_w / 4 + 1)$ local window blocks.
    public var maxTotalSelectedBlocks: Int {
        1 + budgetBlocks + (Self.windowTokens / Self.poolSize) + 1
    }

    /// Sequence length below which all tokens are covered by sink + budget + window,
    /// so attention bypasses sparse selection and runs bit-exact dense attention.
    public var denseBypassThreshold: Int {
        Self.sinkTokens + alignedBudget + Self.windowTokens
    }

    /// Context threshold above which QSA models automatically engage `e-means-4096` coarse clustering in `auto` mode.
    public static let qsaAutoClusterThresholdTokens: Int = 262_144

    /// Resolves the `.qlsa` sidecar file URL for a model pack, checking `explicitSidecarPath` first,
    /// then `<modelRoot>/model.qlsa` and `<modelRoot>/assets/model.qlsa`.
    public static func resolveSidecarURL(
        packPath: String,
        explicitSidecarPath: String? = nil,
        fileManager fm: FileManager = .default
    ) -> URL? {
        if let explicit = explicitSidecarPath?.trimmingCharacters(in: .whitespacesAndNewlines),
           !explicit.isEmpty {
            let expanded = (explicit as NSString).expandingTildeInPath
            if fm.fileExists(atPath: expanded) {
                return URL(fileURLWithPath: expanded)
            }
        }
        let trimmedPack = packPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPack.isEmpty else { return nil }
        let expandedPack = (trimmedPack as NSString).expandingTildeInPath
        let root = QWModelRoot(url: URL(fileURLWithPath: expandedPack))
        let candidates = [
            root.directory.appendingPathComponent("model.qlsa"),
            root.assets.appendingPathComponent("model.qlsa"),
        ]
        for candidate in candidates where fm.fileExists(atPath: candidate.path) {
            return candidate
        }
        return nil
    }
}
