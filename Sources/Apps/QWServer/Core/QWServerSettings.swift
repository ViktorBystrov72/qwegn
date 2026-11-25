import Foundation
import QWCore
import QWEngine

public typealias DiscoveredModel = QWServerSettings.DiscoveredModel

/// Persistent server settings model for QWEGN.
public struct QWServerSettings: Codable, Sendable, Equatable {
    // MARK: - System & App
    public var runAtStartup: Bool
    public var autoStartServer: Bool
    public var keepServerRunningAfterQuit: Bool

    // MARK: - Network & Server
    public var host: String
    public var port: Int
    public var apiKey: String = ""
    public var corsAllowedOrigins: [String] = []
    public var allowLocalImagePaths: Bool = false
    public var allowRemoteImageURLs: Bool = false
    public var telemetryPollIntervalMs: Int

    // MARK: - Model & Architecture
    public var modelPackPath: String
    public var modelName: String
    public var modelAlias: String // User assigned alias for the active model (e.g. "qwen-27b", "chat")
    public var modelAliases: [String: String] // Per-model aliases persisted by modelName
    public var maxContext: Int
    public var kvMode: String // "f16", "q8", "q4"
    public var ropeScalingType: String // "none", "yarn"
    public var ropeScalingFactor: Double

    // MARK: - Sampling & Generation Defaults
    public var temperature: Double
    public var topP: Double
    public var topK: Int
    public var maxTokens: Int
    public var repetitionPenalty: Double
    public var seed: Int // -1 = random

    // MARK: - Reasoning & Thinking
    public var enableThinking: Bool
    public var reasoningEffort: String // "low", "medium", "xhigh"
    public var systemPrompt: String
    public var lazyLoad: Bool
    public var loadVision: Bool

    // MARK: - Compute & Speculation
    public var speculativeDecodingEnabled: Bool
    public var mtpDrafts: Int // 0...15
    public var mtpBackend: String
    public var mtpRelaxedMargin: Double
    /// Variational acceptance probability floor (`0` disables it).
    public var mtpRelaxedMinProb: Double
    public var mtpDraftStrategy: String
    public var mtpQuadraticSeeding: Bool
    public var mtpAdaptiveDepth: Bool
    public var mtpEntropyThreshold: Double
    public var dflashDraftPackPath: String
    public var dflashSelectorTopK: Int
    public var dsparkConfidenceSchedule: Bool
    public var gdnPrefill: String
    public var evictPrefillScratch: String
    public var prefillChunkSize: Int
    public var maxConcurrentSessions: Int
    public var lsaMode: String
    public var lsaBudget: Int
    public var lsaClusterBlock: Int
    public var lsaSidecarPath: String
    public var lsaCrossLoopReuse: Bool = false
    public var engineTuning: [String: QWKnobValue] = [:]  // QWEngineTuning knobs; absent = default

    // MARK: - Two-Tier KV Cache
    public var warmRAMLimitGB: Double
    public var ssdCacheEnabled: Bool
    public var ssdCacheLimitGB: Double
    public var cacheDir: String

    // MARK: - Logging & Diagnostics
    public var logDir: String
    public var logFileName: String
    public var logLevel: String // "debug", "info", "warn", "error"

    // MARK: - Thermal & Cooling
    public var coolingPolicy: String // "auto", "proactive", "max"

    // MARK: - Software Updates
    public var updateCheckSchedule: String // "onLaunchAndDaily", "daily", "onLaunch", "manual"

    // MARK: - Decision Sidecar (Dual-Slot ADR-001)
    public var decisionModelPackPath: String = ""
    public var decisionComputeUnit: String = "auto"
    public var decisionLoadVision: Bool = true
    public var decisionDefaultScoreReadout: String = "mean"
    public var decisionDefaultTruncation: String = "error"
    public var decisionAlwaysEmitBox: Bool = false

    public struct DiscoveredModel: Identifiable, Hashable, Codable, Sendable {
        public var id: String { name }
        public let name: String
        public let packPath: String
        public let family: String?
        public let nativeWindow: Int?
        public let supported: Bool
        public let reason: String?
        public let capabilities: [String]
        public let maxSpeculativeDrafts: Int
        public let hasLSASidecar: Bool
        public let lsaSidecarPath: String?
        public let isDecisionCapable: Bool
        public let isCoreAIBundle: Bool
        public let decisionArchetype: String?
        public let hasVision: Bool

        public init(name: String, packPath: String, family: String? = nil, nativeWindow: Int? = nil,
                    supported: Bool = true, reason: String? = nil,
                    capabilities: [String] = [], maxSpeculativeDrafts: Int = 15,
                    hasLSASidecar: Bool = false, lsaSidecarPath: String? = nil,
                    isDecisionCapable: Bool = false, isCoreAIBundle: Bool = false,
                    decisionArchetype: String? = nil, hasVision: Bool = false) {
            self.name = name
            self.packPath = packPath
            self.family = family
            self.nativeWindow = nativeWindow
            self.supported = supported
            self.reason = reason
            self.capabilities = capabilities
            self.maxSpeculativeDrafts = maxSpeculativeDrafts
            self.hasLSASidecar = hasLSASidecar
            self.lsaSidecarPath = lsaSidecarPath
            self.isDecisionCapable = isDecisionCapable
            self.isCoreAIBundle = isCoreAIBundle
            self.decisionArchetype = decisionArchetype
            self.hasVision = hasVision
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.name = try c.decode(String.self, forKey: .name)
            self.packPath = try c.decode(String.self, forKey: .packPath)
            self.family = try c.decodeIfPresent(String.self, forKey: .family)
            self.nativeWindow = try c.decodeIfPresent(Int.self, forKey: .nativeWindow)
            self.supported = try c.decodeIfPresent(Bool.self, forKey: .supported) ?? true
            self.reason = try c.decodeIfPresent(String.self, forKey: .reason)
            self.capabilities = try c.decodeIfPresent([String].self, forKey: .capabilities) ?? []
            self.maxSpeculativeDrafts = try c.decodeIfPresent(Int.self, forKey: .maxSpeculativeDrafts) ?? 15
            self.hasLSASidecar = try c.decodeIfPresent(Bool.self, forKey: .hasLSASidecar) ?? false
            self.lsaSidecarPath = try c.decodeIfPresent(String.self, forKey: .lsaSidecarPath)
            self.isDecisionCapable = try c.decodeIfPresent(Bool.self, forKey: .isDecisionCapable) ?? false
            self.isCoreAIBundle = try c.decodeIfPresent(Bool.self, forKey: .isCoreAIBundle) ?? false
            self.decisionArchetype = try c.decodeIfPresent(String.self, forKey: .decisionArchetype)
            self.hasVision = try c.decodeIfPresent(Bool.self, forKey: .hasVision) ?? false
        }
    }

    public init(
        runAtStartup: Bool = false,
        autoStartServer: Bool = true,
        keepServerRunningAfterQuit: Bool = false,
        host: String = "127.0.0.1",
        port: Int = 8080,
        apiKey: String = "",
        corsAllowedOrigins: [String] = [],
        allowLocalImagePaths: Bool = false,
        allowRemoteImageURLs: Bool = false,
        telemetryPollIntervalMs: Int = 500,
        modelPackPath: String = "",
        modelName: String = "Qwen3.8-27B",
        modelAlias: String = "",
        modelAliases: [String: String] = [:],
        maxContext: Int = 4096,
        kvMode: String = "q4",
        ropeScalingType: String = "none",
        ropeScalingFactor: Double = 1.0,
        temperature: Double = 0.7,
        topP: Double = 0.9,
        topK: Int = 40,
        maxTokens: Int = 512,
        repetitionPenalty: Double = 1.05,
        seed: Int = -1,
        enableThinking: Bool = true,
        reasoningEffort: String = "xhigh",
        systemPrompt: String = "",
        lazyLoad: Bool = true,
        loadVision: Bool = false,
        speculativeDecodingEnabled: Bool = true,
        mtpDrafts: Int = 3,
        mtpBackend: String = "variational",
        mtpRelaxedMargin: Double = 1.5,
        mtpRelaxedMinProb: Double = QWSpeculativeParameters.defaultRelaxedMinProb,
        mtpDraftStrategy: String = "hybrid",
        mtpQuadraticSeeding: Bool = true,
        mtpAdaptiveDepth: Bool = false,
        mtpEntropyThreshold: Double = 0.7,
        dflashDraftPackPath: String = "",
        dflashSelectorTopK: Int = 16,
        dsparkConfidenceSchedule: Bool = false,
        gdnPrefill: String = "chunked",
        evictPrefillScratch: String = "auto",
        prefillChunkSize: Int = QWPrefillLimits.chunkRows,
        maxConcurrentSessions: Int = QWCacheConfig.defaultMaxConcurrentSessions,
        lsaMode: String = "auto",
        lsaBudget: Int = QWLSAConfig.defaultBudget,
        lsaClusterBlock: Int = QWLSAConfig.defaultClusterBlock,
        lsaSidecarPath: String = "",
        lsaCrossLoopReuse: Bool = false,
        warmRAMLimitGB: Double = 4.0,
        ssdCacheEnabled: Bool = false,
        ssdCacheLimitGB: Double = 10.0,
        cacheDir: String = "~/.qwg/cache",
        logDir: String = "~/.qwg/logs",
        logFileName: String = "qwg.log",
        logLevel: String = "info",
        coolingPolicy: String = "auto",
        updateCheckSchedule: String = QWUpdateCheckSchedule.onLaunchAndDaily.rawValue,
        decisionModelPackPath: String = "",
        decisionComputeUnit: String = "auto",
        decisionLoadVision: Bool = true,
        decisionDefaultScoreReadout: String = "mean",
        decisionDefaultTruncation: String = "error",
        decisionAlwaysEmitBox: Bool = false
    ) {
        self.runAtStartup = runAtStartup
        self.autoStartServer = autoStartServer
        self.keepServerRunningAfterQuit = keepServerRunningAfterQuit
        self.host = host
        self.port = port
        self.apiKey = apiKey
        self.corsAllowedOrigins = corsAllowedOrigins
        self.allowLocalImagePaths = allowLocalImagePaths
        self.allowRemoteImageURLs = allowRemoteImageURLs
        self.telemetryPollIntervalMs = telemetryPollIntervalMs
        self.modelPackPath = modelPackPath
        self.modelName = modelName
        self.modelAlias = modelAlias
        var mergedAliases = modelAliases
        let trimmedAlias = modelAlias.trimmingCharacters(in: .whitespacesAndNewlines)
        if !modelName.isEmpty && !trimmedAlias.isEmpty {
            mergedAliases[modelName] = trimmedAlias
        }
        self.modelAliases = mergedAliases
        self.maxContext = maxContext
        self.kvMode = kvMode
        self.ropeScalingType = ropeScalingType
        self.ropeScalingFactor = ropeScalingFactor
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.maxTokens = maxTokens
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.enableThinking = enableThinking
        self.reasoningEffort = reasoningEffort
        self.systemPrompt = systemPrompt
        self.lazyLoad = lazyLoad
        self.loadVision = loadVision
        self.speculativeDecodingEnabled = speculativeDecodingEnabled
        self.mtpDrafts = mtpDrafts
        self.mtpBackend = mtpBackend
        self.mtpRelaxedMargin = mtpRelaxedMargin
        self.mtpRelaxedMinProb = mtpRelaxedMinProb
        self.mtpDraftStrategy = mtpDraftStrategy
        self.mtpQuadraticSeeding = mtpQuadraticSeeding
        self.mtpAdaptiveDepth = mtpAdaptiveDepth
        self.mtpEntropyThreshold = mtpEntropyThreshold
        self.dflashDraftPackPath = dflashDraftPackPath
        self.dflashSelectorTopK = dflashSelectorTopK
        self.dsparkConfidenceSchedule = dsparkConfidenceSchedule
        self.gdnPrefill = gdnPrefill
        self.evictPrefillScratch = evictPrefillScratch
        self.prefillChunkSize = prefillChunkSize
        self.maxConcurrentSessions = max(1, min(16, maxConcurrentSessions))
        self.lsaMode = lsaMode
        self.lsaBudget = lsaBudget
        self.lsaClusterBlock = lsaClusterBlock
        self.lsaSidecarPath = lsaSidecarPath
        self.lsaCrossLoopReuse = lsaCrossLoopReuse
        self.warmRAMLimitGB = warmRAMLimitGB
        self.ssdCacheEnabled = ssdCacheEnabled
        self.ssdCacheLimitGB = ssdCacheLimitGB
        self.cacheDir = cacheDir
        self.logDir = logDir
        self.logFileName = logFileName
        self.logLevel = logLevel
        self.coolingPolicy = QWCoolingPolicy(rawValue: coolingPolicy)?.rawValue ?? "auto"
        self.updateCheckSchedule = QWUpdateCheckSchedule.resolve(updateCheckSchedule).rawValue
        self.decisionModelPackPath = decisionModelPackPath
        self.decisionComputeUnit = decisionComputeUnit
        self.decisionLoadVision = decisionLoadVision
        self.decisionDefaultScoreReadout = decisionDefaultScoreReadout
        self.decisionDefaultTruncation = decisionDefaultTruncation
        self.decisionAlwaysEmitBox = decisionAlwaysEmitBox
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.runAtStartup = try c.decodeIfPresent(Bool.self, forKey: .runAtStartup) ?? false
        self.autoStartServer = try c.decodeIfPresent(Bool.self, forKey: .autoStartServer) ?? true
        self.keepServerRunningAfterQuit = try c.decodeIfPresent(Bool.self, forKey: .keepServerRunningAfterQuit) ?? false
        self.host = try c.decodeIfPresent(String.self, forKey: .host) ?? "127.0.0.1"
        self.port = try c.decodeIfPresent(Int.self, forKey: .port) ?? 8080
        self.apiKey = try c.decodeIfPresent(String.self, forKey: .apiKey) ?? ""
        self.corsAllowedOrigins = try c.decodeIfPresent([String].self, forKey: .corsAllowedOrigins) ?? []
        self.allowLocalImagePaths = try c.decodeIfPresent(Bool.self, forKey: .allowLocalImagePaths) ?? false
        self.allowRemoteImageURLs = try c.decodeIfPresent(Bool.self, forKey: .allowRemoteImageURLs) ?? false
        self.telemetryPollIntervalMs = try c.decodeIfPresent(Int.self, forKey: .telemetryPollIntervalMs) ?? 500
        self.modelPackPath = try c.decodeIfPresent(String.self, forKey: .modelPackPath) ?? ""
        self.modelName = try c.decodeIfPresent(String.self, forKey: .modelName) ?? "Qwen3.8-27B"
        let decodedAlias = try c.decodeIfPresent(String.self, forKey: .modelAlias) ?? ""
        var decodedAliases = try c.decodeIfPresent([String: String].self, forKey: .modelAliases) ?? [:]
        let trimmedDecoded = decodedAlias.trimmingCharacters(in: .whitespacesAndNewlines)
        if !self.modelName.isEmpty && !trimmedDecoded.isEmpty && decodedAliases[self.modelName] == nil {
            decodedAliases[self.modelName] = trimmedDecoded
        }
        self.modelAliases = decodedAliases
        self.modelAlias = decodedAliases[self.modelName] ?? trimmedDecoded
        self.maxContext = try c.decodeIfPresent(Int.self, forKey: .maxContext) ?? 4096
        self.kvMode = try c.decodeIfPresent(String.self, forKey: .kvMode) ?? "q4"
        func decodeDouble(forKey key: CodingKeys, defaultVal: Double) throws -> Double {
            try c.decodeIfPresent(Double.self, forKey: key) ?? defaultVal
        }
        self.ropeScalingType = try c.decodeIfPresent(String.self, forKey: .ropeScalingType) ?? "none"
        self.ropeScalingFactor = try decodeDouble(forKey: .ropeScalingFactor, defaultVal: 1.0)
        self.temperature = try decodeDouble(forKey: .temperature, defaultVal: 0.7)
        self.topP = try decodeDouble(forKey: .topP, defaultVal: 0.9)
        self.topK = try c.decodeIfPresent(Int.self, forKey: .topK) ?? 40
        self.maxTokens = try c.decodeIfPresent(Int.self, forKey: .maxTokens) ?? 512
        self.repetitionPenalty = try decodeDouble(forKey: .repetitionPenalty, defaultVal: 1.05)
        self.seed = try c.decodeIfPresent(Int.self, forKey: .seed) ?? -1
        self.enableThinking = try c.decodeIfPresent(Bool.self, forKey: .enableThinking) ?? true
        self.reasoningEffort = try c.decodeIfPresent(String.self, forKey: .reasoningEffort) ?? "xhigh"
        self.systemPrompt = try c.decodeIfPresent(String.self, forKey: .systemPrompt) ?? ""
        self.lazyLoad = try c.decodeIfPresent(Bool.self, forKey: .lazyLoad) ?? true
        self.loadVision = try c.decodeIfPresent(Bool.self, forKey: .loadVision) ?? false

        self.speculativeDecodingEnabled = try c.decodeIfPresent(Bool.self, forKey: .speculativeDecodingEnabled) ?? true
        self.mtpDrafts = try c.decodeIfPresent(Int.self, forKey: .mtpDrafts) ?? 3
        let rawBackend = try c.decodeIfPresent(String.self, forKey: .mtpBackend) ?? "variational"
        self.mtpBackend = (rawBackend == "speculative" || rawBackend == "lightening") ? "standard" : rawBackend
        self.mtpRelaxedMargin = try decodeDouble(forKey: .mtpRelaxedMargin, defaultVal: 1.5)
        self.mtpRelaxedMinProb = try decodeDouble(forKey: .mtpRelaxedMinProb, defaultVal: QWSpeculativeParameters.defaultRelaxedMinProb)
        self.mtpDraftStrategy = try c.decodeIfPresent(String.self, forKey: .mtpDraftStrategy) ?? "hybrid"
        self.mtpQuadraticSeeding = try c.decodeIfPresent(Bool.self, forKey: .mtpQuadraticSeeding) ?? true
        self.mtpAdaptiveDepth = try c.decodeIfPresent(Bool.self, forKey: .mtpAdaptiveDepth) ?? false
        self.mtpEntropyThreshold = try decodeDouble(forKey: .mtpEntropyThreshold, defaultVal: 0.7)
        self.dflashDraftPackPath = try c.decodeIfPresent(String.self, forKey: .dflashDraftPackPath) ?? ""
        self.dflashSelectorTopK = try c.decodeIfPresent(Int.self, forKey: .dflashSelectorTopK) ?? 16
        self.dsparkConfidenceSchedule = try c.decodeIfPresent(Bool.self, forKey: .dsparkConfidenceSchedule) ?? false
        self.gdnPrefill = try c.decodeIfPresent(String.self, forKey: .gdnPrefill) ?? "chunked"
        self.evictPrefillScratch = try c.decodeIfPresent(String.self, forKey: .evictPrefillScratch) ?? "auto"
        self.prefillChunkSize = try c.decodeIfPresent(Int.self, forKey: .prefillChunkSize) ?? QWPrefillLimits.chunkRows
        let decodedSessions = try c.decodeIfPresent(Int.self, forKey: .maxConcurrentSessions) ?? QWCacheConfig.defaultMaxConcurrentSessions
        self.maxConcurrentSessions = max(1, min(16, decodedSessions))
        self.lsaMode = try c.decodeIfPresent(String.self, forKey: .lsaMode) ?? "auto"
        self.lsaBudget = try c.decodeIfPresent(Int.self, forKey: .lsaBudget) ?? QWLSAConfig.defaultBudget
        self.lsaClusterBlock = try c.decodeIfPresent(Int.self, forKey: .lsaClusterBlock) ?? QWLSAConfig.defaultClusterBlock
        self.lsaSidecarPath = try c.decodeIfPresent(String.self, forKey: .lsaSidecarPath) ?? ""
        self.lsaCrossLoopReuse = try c.decodeIfPresent(Bool.self, forKey: .lsaCrossLoopReuse) ?? false
        self.engineTuning = try c.decodeIfPresent([String: QWKnobValue].self, forKey: .engineTuning) ?? [:]
        self.warmRAMLimitGB = try decodeDouble(forKey: .warmRAMLimitGB, defaultVal: 4.0)
        self.ssdCacheEnabled = try c.decodeIfPresent(Bool.self, forKey: .ssdCacheEnabled) ?? false
        self.ssdCacheLimitGB = try decodeDouble(forKey: .ssdCacheLimitGB, defaultVal: 10.0)
        self.cacheDir = try c.decodeIfPresent(String.self, forKey: .cacheDir) ?? "~/.qwg/cache"
        self.logDir = try c.decodeIfPresent(String.self, forKey: .logDir) ?? "~/.qwg/logs"
        self.logFileName = try c.decodeIfPresent(String.self, forKey: .logFileName) ?? "qwg.log"
        self.logLevel = try c.decodeIfPresent(String.self, forKey: .logLevel) ?? "info"
        let rawCooling = try c.decodeIfPresent(String.self, forKey: .coolingPolicy) ?? "auto"
        self.coolingPolicy = QWCoolingPolicy(rawValue: rawCooling)?.rawValue ?? "auto"
        let rawSchedule = try c.decodeIfPresent(String.self, forKey: .updateCheckSchedule)
        self.updateCheckSchedule = QWUpdateCheckSchedule.resolve(rawSchedule).rawValue
        self.decisionModelPackPath = try c.decodeIfPresent(String.self, forKey: .decisionModelPackPath) ?? ""
        self.decisionComputeUnit = try c.decodeIfPresent(String.self, forKey: .decisionComputeUnit) ?? "auto"
        self.decisionLoadVision = try c.decodeIfPresent(Bool.self, forKey: .decisionLoadVision) ?? true
        self.decisionDefaultScoreReadout = try c.decodeIfPresent(String.self, forKey: .decisionDefaultScoreReadout) ?? "mean"
        self.decisionDefaultTruncation = try c.decodeIfPresent(String.self, forKey: .decisionDefaultTruncation) ?? "error"
        self.decisionAlwaysEmitBox = try c.decodeIfPresent(Bool.self, forKey: .decisionAlwaysEmitBox) ?? false
    }

    /// Full expanded path to the log file
    public var expandedLogFilePath: String {
        let expandedDir = logDir.expandingTildePath
        return (expandedDir as NSString).appendingPathComponent(logFileName)
    }

    /// Cohesive sampling parameters view of these settings.
    public var samplingParameters: QWSamplingParameters {
        get {
            QWSamplingParameters(
                temperature: temperature,
                topP: topP,
                topK: topK,
                maxTokens: maxTokens,
                repetitionPenalty: repetitionPenalty,
                seed: seed
            )
        }
        set {
            temperature = newValue.temperature
            topP = newValue.topP
            topK = newValue.topK
            maxTokens = newValue.maxTokens
            repetitionPenalty = newValue.repetitionPenalty
            seed = newValue.seed
        }
    }

    /// Cohesive speculative decoding parameters view of these settings.
    public var speculativeParameters: QWSpeculativeParameters {
        get {
            QWSpeculativeParameters(
                enabled: speculativeDecodingEnabled,
                mtpDrafts: mtpDrafts,
                mtpBackend: mtpBackend,
                mtpRelaxedMargin: mtpRelaxedMargin,
                mtpRelaxedMinProb: mtpRelaxedMinProb,
                mtpDraftStrategy: mtpDraftStrategy,
                mtpQuadraticSeeding: mtpQuadraticSeeding,
                adaptiveDepth: mtpAdaptiveDepth,
                entropyThreshold: mtpEntropyThreshold,
                dflashDraftPackPath: dflashDraftPackPath.isEmpty ? nil : dflashDraftPackPath,
                dflashSelectorTopK: dflashSelectorTopK,
                dsparkConfidenceSchedule: dsparkConfidenceSchedule
            )
        }
        set {
            speculativeDecodingEnabled = newValue.enabled
            mtpDrafts = newValue.mtpDrafts
            mtpBackend = newValue.mtpBackend ?? "variational"
            mtpRelaxedMargin = newValue.mtpRelaxedMargin
            mtpRelaxedMinProb = newValue.mtpRelaxedMinProb
            mtpDraftStrategy = newValue.mtpDraftStrategy
            mtpQuadraticSeeding = newValue.mtpQuadraticSeeding
            mtpAdaptiveDepth = newValue.adaptiveDepth
            mtpEntropyThreshold = newValue.entropyThreshold
            dflashDraftPackPath = newValue.dflashDraftPackPath ?? ""
            dflashSelectorTopK = newValue.dflashSelectorTopK
            if let ds = newValue.dsparkConfidenceSchedule {
                dsparkConfidenceSchedule = ds
            }
        }
    }

    /// Cohesive Latent Sparse Attention configuration view of these settings.
    public var lsaConfig: QWLSAConfig {
        get {
            QWLSAConfig(
                mode: QWLSAMode.resolve(lsaMode) ?? .off,
                budget: lsaBudget,
                clusterBlock: lsaClusterBlock,
                sidecarPath: lsaSidecarPath,
                crossLoopReuse: lsaCrossLoopReuse
            )
        }
        set {
            lsaMode = newValue.mode.rawValue
            lsaBudget = newValue.budget
            lsaClusterBlock = newValue.clusterBlock
            lsaSidecarPath = newValue.sidecarPath
            lsaCrossLoopReuse = newValue.crossLoopReuse
        }
    }

    /// Returns the hardware-scaled default prefill chunk size for a model descriptor.
    public static func recommendedPrefillChunkSize(
        for descriptor: QWModelDescriptor?,
        physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) -> Int {
        guard let descriptor else { return QWPrefillLimits.chunkRows }
        let isMoE = (descriptor.numExperts ?? 0) > 0
            || descriptor.capabilities.contains(.moe)
        let ramGB = Double(physicalMemory) / 1_000_000_000.0
        if isMoE && ramGB > 36.5 {
            return 4096
        }
        if ramGB > 90.0 {
            return 2048
        }
        return QWPrefillLimits.chunkRows
    }

    /// Reinstates the engine's top-performance default settings while preserving
    /// the currently selected model, model aliases, and host/port configuration.
    public mutating func resetToTopPerformanceDefaults(for descriptor: QWModelDescriptor? = nil) {
        let resolvedDescriptor = descriptor ?? (try? validatePrimaryModel())
        let specDefaults = QWSpeculativeParameters.topPerformanceDefaults(for: resolvedDescriptor)
        speculativeParameters = specDefaults
        maxContext = resolvedDescriptor?.maxPositionEmbeddings ?? 4096
        kvMode = "q4"
        ropeScalingType = "none"
        ropeScalingFactor = 1.0
        temperature = 0.7
        topP = 0.9
        topK = 40
        maxTokens = 512
        repetitionPenalty = 1.05
        seed = -1
        enableThinking = true
        reasoningEffort = "xhigh"
        gdnPrefill = "chunked"
        evictPrefillScratch = "auto"
        lsaMode = "auto"
        lsaBudget = QWLSAConfig.defaultBudget
        lsaClusterBlock = QWLSAConfig.defaultClusterBlock
        lsaSidecarPath = ""
        lsaCrossLoopReuse = false
        engineTuning = [:]
        prefillChunkSize = Self.recommendedPrefillChunkSize(for: resolvedDescriptor)
        maxConcurrentSessions = QWCacheConfig.defaultMaxConcurrentSessions
        warmRAMLimitGB = 4.0
    }

}
