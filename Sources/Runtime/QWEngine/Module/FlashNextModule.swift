// FlashNextModule.swift — The Flash-Next (qwen4_exp) family module.
// The adapter: it owns no numerics, it drives the Flash-Next engine through
// the seam the core loops over.

import Foundation
import QWCore
import QWFlashNext
import QWHead
import QWHAL
import QWKV

/// One Flash-Next session behind the seam. The engine owns the numerics;
/// this owns the request.
final class FNGPUSessionState: QWModuleSessionState, @unchecked Sendable {
    let engine: FNGPUEngine
    /// The session's own state: history, sampler, prefill cursors.
    var data: FNGPUSessionData
    /// How many tokens the request may generate. The engine's state wipe is
    /// scoped to the prompt plus these tokens.
    var maxNew: Int = 0

    init(engine: FNGPUEngine, data: FNGPUSessionData) {
        self.engine = engine
        self.data = data
    }

    var activeTokens: Int { data.activeTokens }

    func reset(sampling: QWSamplerConfig, speculative: QWSpeculativeParameters) throws {
        let useSpec = FNGPUSessionState.useSpec(sampling: sampling, speculative: speculative, engine: engine)
        let draftMode = try FNGPUSessionState.draftMode(for: speculative)
        data = FNGPUSessionData(history: [], sampler: QWSampler(config: sampling),
                                useSpec: useSpec)
        data.draftMode = draftMode
        let specCap = engine.dflash != nil
            ? min(FNGPUDFlash.maxDrafts, engine.dflashMaxDrafts)
            : FNGPUSessionData.defaultMTPDrafts
        data.maxDrafts = max(1, min(specCap, speculative.effectiveDrafts))
        data.relaxedMargin = sampling.hasExplicitSeed ? 0 : speculative.effectiveRelaxedMargin
        data.relaxedMinProb = speculative.effectiveRelaxedMinProb
        data.quadraticSeeding = speculative.mtpQuadraticSeeding
        maxNew = 0
    }

    /// Whether the session runs the speculative path. Depends only on whether
    /// speculative decoding is enabled (`effectiveDrafts > 0`) and supported by
    /// the engine, independent of `temperature`, `topK`, or `topP`.
    static func useSpec(sampling: QWSamplerConfig, speculative: QWSpeculativeParameters,
                        engine: FNGPUEngine) -> Bool {
        guard !(sampling.hasExplicitSeed && !sampling.isGreedy) else { return false }
        return speculative.effectiveDrafts > 0 && engine.supportsSpeculativeDecode
    }

    /// Which drafter the session runs: the context-copy drafter in hybrid
    /// mode, the ngram drafter, or the neural MTP drafter.
    static func draftMode(for speculative: QWSpeculativeParameters,
                          environment: [String: String] = ProcessInfo.processInfo.environment) throws -> QWDraftStrategy {
        if let backend = speculative.mtpBackend, !backend.isEmpty {
            _ = try QWMTPBackend.resolve(name: backend.lowercased())
        }
        if speculative.draftStrategyIsResolved != true,
           let envDraft = environment["FN_DRAFT"], !envDraft.isEmpty {
            return try QWDraftStrategy.resolve(envDraft)
        }
        return try QWDraftStrategy.resolve(speculative.mtpDraftStrategy)
    }
}

/// The Flash-Next family, served by `FNGPUEngine`.
public final class FlashNextModule: QWInferenceModule, @unchecked Sendable {
    public static let family: QWModelDescriptor.Family = .qwen38FlashNext

    public static func resolveSpeculation(
        _ requested: QWSpeculativeParameters, descriptor: QWModelDescriptor,
        environment: [String: String], modelSwitch: Bool
    ) throws -> QWResolvedSpeculation {
        var parameters = requested
        let envPack = environment["QW_DFLASH_DRAFT_PACK"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        let reqPack = parameters.dflashDraftPackPath?.trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedPack = (reqPack?.isEmpty == false ? reqPack : nil)
            ?? (envPack?.isEmpty == false ? envPack : nil)
            ?? QWDFlashConfig.defaultPackPath(for: descriptor)
        let inferred = QWDFlashConfig.inferBackendAndMaxDrafts(packPath: resolvedPack)
        let dflashCap = min(FNGPUDFlash.maxDrafts, max(1, inferred.maxDrafts))
        if modelSwitch {
            if FNGPUDFlash.isDFlashName(parameters.mtpBackend) {
                let compatible = (try? QWDFlashConfig.validateDrafter(
                    targetDescriptor: descriptor, packPath: resolvedPack,
                    drafts: 1, selectorTopK: parameters.dflashSelectorTopK
                )) != nil
                if compatible {
                    parameters.mtpBackend = inferred.backend
                    parameters.mtpDrafts = max(1, min(dflashCap, parameters.mtpDrafts))
                } else {
                    parameters.mtpBackend = "variational"
                    parameters.mtpDrafts = min(parameters.mtpDrafts, max(1, descriptor.maxSpeculativeDrafts))
                }
            } else {
                parameters.mtpDrafts = min(parameters.mtpDrafts, max(1, descriptor.maxSpeculativeDrafts))
            }
        }
        let mode = try FNGPUSessionState.draftMode(for: parameters, environment: environment)
        parameters.mtpDraftStrategy = mode.rawValue
        parameters.draftStrategyIsResolved = true
        if FNGPUDFlash.isDFlashName(parameters.mtpBackend) {
            let drafts = parameters.enabled ? max(0, min(dflashCap, parameters.mtpDrafts)) : 0
            let backendName = parameters.mtpBackend?.lowercased() == "dspine" ? "dspine" : "dflash2"
            return QWResolvedSpeculation(parameters: parameters, engineDrafts: drafts,
                                         engineBackend: drafts > 0 ? .dflash2 : nil,
                                         reportedDrafts: drafts,
                                         reportedBackend: backendName)
        }
        let engineDrafts = parameters.enabled && descriptor.capabilities.contains(.mtp) ? parameters.mtpDrafts : 0
        return QWResolvedSpeculation(parameters: parameters, engineDrafts: engineDrafts,
                                     engineBackend: nil,
                                     reportedDrafts: parameters.enabled ? max(0, min(FNGPUSessionData.defaultMTPDrafts, parameters.mtpDrafts)) : 0,
                                     reportedBackend: mode.rawValue)
    }

    public static func estimateContextMemory(
        descriptor: QWModelDescriptor, packPath: String?, contextTokens: Int, kvMode: QWKVMode,
        mtpDrafts: Int, mtpBackend: String, prefillChunkSize: Int, tuning: QWEngineTuning,
        physicalMemoryBytes: Int?
    ) -> QWContextMemoryEstimate {
        let clampedContext = max(64, min(1_048_576, contextTokens))
        let isDFlash = mtpDrafts > 0 && FNGPUDFlash.isDFlashName(mtpBackend)
        let rawPerTokBytes: Int = switch kvMode {
        case .f16: 31_296
        case .q8:  18_096
        case .q4:  11_752
        }
        let perTokBytes = FNGPUDFlash.adjustedPerTokenKVBytes(rawPerTokBytes, usesDFlash: isDFlash)
        let hiBit = (descriptor.quantization?.bits ?? 4) > 4
        let minWeights = (hiBit ? 3_408 : 2_384) << 20
        let onDisk = QWResidencyAdmission.flashNextOnDiskFootprint(packPath: packPath, minWeightsBytes: minWeights)
        let dflashDelta = FNGPUDFlash.estimateFootprint(
            targetDescriptor: descriptor, targetPackPath: packPath, dflashPackPath: nil, isDFlash: isDFlash)
        let weightsBytes = max(512 << 20, onDisk.weightsBytes - dflashDelta.nativeMTPWeightBytes + dflashDelta.dflashWeightBytes) + onDisk.pleBytes
        let scratchBytes = (1408 << 20) + (836 << 20) + (isDFlash ? max(0, dflashDelta.dflashScratchBytes - (112 << 20)) : 0)
        let fixedBytes = weightsBytes + scratchBytes
        let alignedCap = QWCapacityPlan.steppedCapacity(clampedContext)
        let kvCap = alignedCap >= 32_768 ? (alignedCap + 1024) : alignedCap
        let kvBytes = kvCap * perTokBytes
        let requiredBytes = fixedBytes + kvBytes
        let reserveGiB = tuning.osReserveGiB ?? Double(QWCapacityBudget.osReserveBytes >> 30)
        let reserveBytes = max(0, Int((reserveGiB * Double(1 << 30)).rounded()))
        let availBytes = max(0, (physicalMemoryBytes ?? 0) - reserveBytes)
        let maxFitting = availBytes > fixedBytes ? min(1_048_576, (availBytes - fixedBytes) / perTokBytes) : 0
        return QWContextMemoryEstimate(
            contextTokens: clampedContext, weightsBytes: weightsBytes, kvBytes: kvBytes,
            scratchAndStateBytes: scratchBytes, requiredBytes: requiredBytes,
            availableBytes: availBytes, maxFittingContext: maxFitting,
            fits: availBytes == 0 || requiredBytes <= availBytes)
    }

    /// Capabilities supported by Flash-Next (`qwen4_exp`), including the 27-block ViT vision tower and DFlash 2.
    public static let capabilities: Set<QWKnobCapability> = [
        .vision, .prefixCache, .kvQuantQ8, .kvQuantQ4, .speculative, .lsa, .dflash,
    ]

    /// `FNKVStorageMode` keeps the legacy f32 path alongside f16/q8/q4.
    public static let supportedKVStorageModes: Set<String> = ["f32", "f16", "q8", "q4"]

    /// Every kernel door the Flash-Next engine dispatches.
    public static let requiredDoors = QWKernelInventory.flash

    /// The Flash-Next engine. Nil before the engine is loaded.
    public let engine: FNGPUEngine?
    /// The frozen numeric plan of the loaded model.
    public let plan: QWEnginePlan
    private let stateLock = NSLock()
    private var _onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)?
    private var _prefixCacheHit: Bool = false
    private var _prefixCacheMatchedTokens: Int = 0
    private var _prefixCacheTier: FNPrefixCacheTier?
    private var _naxLogMessage: String = ""
    private var _lastResult: QWGenerationResult?
    private var _lastSpecProposals: Int = 0
    private var _lastSpecAccepted: Int = 0

    /// Where the module reports what its prefix cache did.
    public var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)? {
        get { stateLock.withLock { _onCacheEvent } }
        set { stateLock.withLock { _onCacheEvent = newValue } }
    }
    /// Whether the prefix cache served the last prompt this module ran.
    public var prefixCacheHit: Bool { stateLock.withLock { _prefixCacheHit } }
    /// How many tokens the last restore reused.
    public var prefixCacheMatchedTokens: Int { stateLock.withLock { _prefixCacheMatchedTokens } }
    /// The tier that served the last restore: the RAM or the disk.
    public var prefixCacheTier: FNPrefixCacheTier? { stateLock.withLock { _prefixCacheTier } }
    /// The NAX mode the last request chose, in the engine's own words.
    public internal(set) var naxLogMessage: String {
        get { stateLock.withLock { _naxLogMessage } }
        set { stateLock.withLock { _naxLogMessage = newValue } }
    }
    /// The result of the last turn this module ran.
    public internal(set) var lastResult: QWGenerationResult? {
        get { stateLock.withLock { _lastResult } }
        set { stateLock.withLock { _lastResult = newValue } }
    }
    /// Speculative draft proposals in the last turn.
    public internal(set) var lastSpecProposals: Int {
        get { stateLock.withLock { _lastSpecProposals } }
        set { stateLock.withLock { _lastSpecProposals = newValue } }
    }
    /// Speculative accepted drafts in the last turn.
    public internal(set) var lastSpecAccepted: Int {
        get { stateLock.withLock { _lastSpecAccepted } }
        set { stateLock.withLock { _lastSpecAccepted = newValue } }
    }

    public init(engine: FNGPUEngine) {
        self.engine = engine
        self.plan = engine.plan
        engine.batchScheduler.onCacheEvaluated = { [weak self, weak engine] sessionData in
            guard let self, let engine else { return }
            self.reportCacheEvent(engine: engine, sessionData: sessionData)
        }
        engine.batchScheduler.onSlot0CacheEvaluated = nil
        engine.batchScheduler.onStepTiming = { gpuMs, wallMs in
            if gpuMs > 0 && wallMs > 0 {
                QWTelemetryRegistry.shared.record("gpu.busy_ratio", gpuMs / wallMs, provenance: .measured)
            }
        }
        engine.onCacheRetained = { [weak self] tokens, chargedBytes, storeTotalBytes, warmLimitBytes in
            self?.onCacheEvent?(.retained(tokens: tokens, chargedBytes: chargedBytes,
                                          storeTotalBytes: storeTotalBytes, warmLimitBytes: warmLimitBytes))
        }
        engine.onCacheSkippedOversize = { [weak self] tokens, charge, warmLimitBytes in
            self?.onCacheEvent?(.skippedOversize(tokens: tokens, charge: charge, warmLimitBytes: warmLimitBytes))
        }
        engine.onCacheEvicted = { [weak self] tokens, freedBytes in
            self?.onCacheEvent?(.evicted(tokens: tokens, freedBytes: freedBytes))
        }
    }

    /// A module with no engine yet. Every call that needs the engine fails
    /// loud — it never pretends to serve.
    public init(plan: QWEnginePlan) {
        self.engine = nil
        self.plan = plan
    }

    public static func make(plan: QWEnginePlan,
                            engine: (any QWLoadedEngine)?) throws -> any QWInferenceModule {
        guard let engine else { return FlashNextModule(plan: plan) }
        guard let flashNext = engine as? FNGPUEngine else {
            throw QWModuleError.unsupportedFamily(.qwen38FlashNext)
        }
        return FlashNextModule(engine: flashNext)
    }

    /// The KV storage mode the plan was frozen with.
    public var kvStorageMode: String { plan.kvStorageMode }

    /// The Flash-Next engine owns its whole loop, so there is no
    /// conversation to have: the module returns none.
    public func makeConversation() -> QWConversation? { nil }

    /// The last turn's measurements, in the shape every family reports.
    public var lastTurn: QWGenerationResult? { lastResult }

    /// What the Flash-Next engine says about itself. The frame belongs
    /// to the consumer; the NAX line is this engine's own story.
    public var report: QWEngineReport {
        guard let engine else {
            return QWEngineReport(capacityLog: "", arenaBytes: 0,
                                  weightUsedBytes: 0, weightReservedBytes: 0)
        }
        return QWEngineReport(
            capacityLog: engine.capacityLogMessage,
            arenaBytes: engine.arenaBytes,
            weightUsedBytes: engine.weightUsedBytes,
            weightReservedBytes: engine.weightReservedBytes,
            steadyAllocations: engine.steadyStateAllocations,
            pipelineSeconds: QWHalPSOStats.seconds,
            detailLines: naxLogMessage + engine.prefillStackNote + engine.dflashReportLine,
            contextCapacity: engine.capacity,
            weightFormat: engine.headWeightFormatLabel
        )
    }

    /// Builds the Flash-Next engine for a configuration. The module owns
    /// how its engine is made, so no consumer names the concrete engine.
    public static func makeEngine(options: QWEngineBuildOptions) throws -> any QWInferenceEngine {
        let resolvedTuning = options.tuning.overlaid(environment: ProcessInfo.processInfo.environment)
        let engine = try autoreleasepool {
            try FNGPUEngine(
                packPath: options.packPath,
                maxTokens: options.maxTokens,
                kvMode: options.kvMode,
                osReserveGiB: options.osReserveGiB ?? resolvedTuning.osReserveGiB,
                prefillChunkSize: options.prefillChunkSize,
                prefixCacheDirectory: options.prefixCacheDirectory,
                prefixCacheWarmByteLimit: options.cacheConfig?.warmByteLimit ?? 0,
                prefixCacheSSDByteLimit: options.cacheConfig?.ssdByteLimit ?? 0,
                maxConcurrentSessions: options.maxConcurrentSessions,
                evictPrefillScratch: options.prefillScratchEviction.shouldEvict(),
                prewarmDoors: FlashNextModule.requiredDoors,
                lsaConfig: options.lsaConfig,
                streamNgrams: resolvedTuning.streamNgrams,
                ropeYarnFactor: options.ropeYarnFactor,
                mtpBackend: options.mtpBackend?.rawValue,
                mtpDrafts: options.mtpDrafts,
                dflashDraftPackPath: options.dflashDraftPackPath,
                dflashSelectorTopK: options.dflashSelectorTopK,
                mtpDraftStrategy: options.mtpDraftStrategy
            )
        }
        return FlashNextModule(engine: engine)
    }

    /// Whether a prefix cache is configured on this module right now.
    public var hasPrefixCache: Bool { engine?.hasPrefixCacheTier == true }

    public func configurePrefixCache(_ config: QWCacheConfig) throws {
        throw QWModuleError.notImplemented("FlashNextModule.configurePrefixCache")
    }

    public func makeSessionState() throws -> any QWModuleSessionState {
        guard let engine else {
            throw QWModuleError.notImplemented("FlashNextModule has no engine yet")
        }
        let data = FNGPUSessionData(history: [], sampler: QWSampler(config: .greedy))
        return FNGPUSessionState(engine: engine, data: data)
    }

    /// Prepares the engine for this prompt: the prefill stack is carved
    /// here, not inside the measured prefill.
    public func prepare(_ state: any QWModuleSessionState, prompt: [Int], maxNew: Int) throws {
        guard let engine else {
            throw QWModuleError.notImplemented("FlashNextModule has no engine yet")
        }
        try engine.preparePrefill(forTokenCount: prompt.count)
        // The session knows how many tokens this request may generate, so the
        // engine's state wipe is scoped to the prompt plus those tokens.
        let session = try unwrap(state)
        session.maxNew = maxNew
        session.data.history = prompt.map { Int64($0) }
        engine.beginSession(&session.data, maxNew: maxNew)
    }

    /// The request is over. The Flash-Next engine holds nothing across a
    /// request that the session state does not already own.
    public func finish(_ state: any QWModuleSessionState) {
        if let session = state as? FNGPUSessionState {
            engine?.flushPendingPrefixCache(&session.data)
            stateLock.withLock {
                _lastSpecProposals = session.data.specProposals
                _lastSpecAccepted = session.data.specAccepted
            }
        }
    }

    public func encodePrefill(_ state: any QWModuleSessionState, tokens: [Int],
                               chunk: Range<Int>, runsHead: Bool) throws -> Int {
        guard let engine else {
            throw QWModuleError.notImplemented("FlashNextModule has no engine yet")
        }
        let session = try unwrap(state)
        // The prompt is fed exactly once: the session takes it whole and the
        // engine's own chunking walks it.
        let first = chunk.lowerBound == 0
        if session.data.history.isEmpty {
            session.data.history = tokens.map { Int64($0) }
            engine.beginSession(&session.data, maxNew: session.maxNew)
        }
        let token = try engine.encodePrefillChunk(&session.data, chunk: chunk, runsHead: runsHead)
        // The cache answered once per prompt, not once per chunk.
        if first { reportCacheEvent(engine: engine, session: session) }
        return token
    }

    public func encodeDecodeStep(_ state: any QWModuleSessionState, token: Int,
                                 position: Int) throws -> Int {
        guard let engine else {
            throw QWModuleError.notImplemented("FlashNextModule has no engine yet")
        }
        let session = try unwrap(state)
        let next = try engine.encodeDecodeToken(&session.data, token: token, position: position)
        // The engine's own GPU busy ratio: the command buffers' GPU time over
        // the wall time the step took. Measured, never estimated. Both
        // numbers are this session's: another session's step has its own.
        let gpuMs = session.data.lastStepGPUMillis
        let wallMs = session.data.stepMs
        if gpuMs > 0, wallMs > 0 {
            QWTelemetryRegistry.shared.record("gpu.busy_ratio", gpuMs / wallMs, provenance: .measured)
        }
        return next
    }

    /// The module attributes the step's time to the token it produced: a
    /// speculative round splits its time across the tokens it committed.
    public func lastStepMilliseconds(_ state: any QWModuleSessionState) -> Double {
        (state as? FNGPUSessionState)?.data.stepMs ?? 0
    }

    /// The GPU seconds the prefill of this session spent.
    public func prefillGPUSeconds(_ state: any QWModuleSessionState) -> Double {
        (state as? FNGPUSessionState)?.data.prefillGPUSec ?? 0
    }

    /// Tokens committed per speculative round in this session.
    public func roundTokenCounts(_ state: any QWModuleSessionState) -> [Int] {
        (state as? FNGPUSessionState)?.data.roundTokenCounts ?? []
    }

    /// The concrete session behind the seam. A state this module did not make
    /// is an error, never a silently mis-typed session.
    private func unwrap(_ state: any QWModuleSessionState) throws -> FNGPUSessionState {
        guard let session = state as? FNGPUSessionState else {
            throw QWModuleError.unsupportedFamily(.qwen38FlashNext)
        }
        return session
    }

    /// Reports what the prefix cache did with this prompt. The event fires
    /// once per prefill: a hit says how many tokens were reused, a miss says
    /// the prompt will be saved.
    private func reportCacheEvent(engine: FNGPUEngine, session: FNGPUSessionState) {
        reportCacheEvent(engine: engine, sessionData: session.data)
    }

    func recordGenerationResult(_ result: QWGenerationResult, specProposals: Int, specAccepted: Int) {
        stateLock.withLock {
            _lastSpecProposals = specProposals
            _lastSpecAccepted = specAccepted
            _lastResult = result
        }
    }

    func reportCacheEvent(engine: FNGPUEngine, sessionData: FNGPUSessionData) {
        let (hit, matched, tier, sink) = stateLock.withLock {
            _prefixCacheHit = sessionData.prefixCacheHit
            _prefixCacheMatchedTokens = sessionData.prefixCacheMatchedTokens
            _prefixCacheTier = sessionData.prefixCacheTier
            return (_prefixCacheHit, _prefixCacheMatchedTokens, _prefixCacheTier, _onCacheEvent)
        }
        guard let sink else { return }
        if hit {
            // The tier that served the prefix is the one it came from; a
            // tier the engine does not name is left unnamed, not guessed.
            sink(.hit(tier: tier?.rawValue ?? "",
                      matchedTokens: matched,
                      totalTokens: sessionData.history.count))
        } else if engine.hasPrefixCacheTier {
            sink(.miss(totalTokens: sessionData.history.count))
        }
    }
}

/// The Flash-Next engine is a loaded engine the registry can hand over.
extension FNGPUEngine: QWLoadedEngine {}

/// The Flash-Next module, seen through the neutral protocol: the same
/// surface the dense engine offers, with no family in the signature.
extension FlashNextModule: QWInferenceEngine {
    /// What this engine can do. A drafter the engine has no head for is
    /// absent: the capability is not advertised and not served.
    public var capabilities: Set<QWKnobCapability> {
        guard let engine else { return FlashNextModule.capabilities }
        var caps = FlashNextModule.capabilities
        if !engine.supportsSpeculativeDecode { caps.remove(.speculative) }
        if engine.dflash == nil { caps.remove(.dflash) }
        return caps
    }

    /// The cache the module holds: the warm tier in the RAM and the cold
    /// one on disk. Their sizes can be counted, so they are reported; a
    /// tier that is not there is absent, not zero.
    public var prefixCache: QWPrefixCacheState {
        guard hasPrefixCache else { return QWPrefixCacheState(isConfigured: false) }
        let disk = engine?.prefixCacheDiskState
        let warm = engine?.prefixCacheWarmState
        return QWPrefixCacheState(
            isConfigured: true,
            warmBytes: warm?.bytes,
            ssdBytes: disk?.bytes,
            warmEntries: warm?.entries,
            ssdEntries: disk?.entries,
            warmLimitBytes: engine?.prefixCacheWarmLimitBytes)
    }

    public var supportsTokenConstraints: Bool { true }

    /// Runs one generation through the shared core loop every family
    /// uses. A request that demands a cache this engine does not have
    /// is refused, never served without one.
    public func generate(
        request: QWGenerationRequest,
        onToken: ((Int) -> Bool)?
    ) throws -> QWGenerationResult {
        guard engine != nil else {
            throw QWEngineError.state("FlashNextModule has no engine yet")
        }
        if request.usePrefixCache && !hasPrefixCache {
            throw QWEngineError.state("the request wants a prefix cache this engine does not have")
        }
        return try runGeneration(request: request, onToken: onToken, onStopToken: nil)
    }

    public func buildStandalonePrefixCache(
        tokens: [Int],
        outputURL: URL,
        pinAsSharedBase: Bool = true,
        onPrefillProgress: (@Sendable (Int, Int) -> Void)?
    ) throws -> Int {
        guard let engine else {
            throw QWEngineError.state("FlashNextModule has no engine yet")
        }
        return try engine.buildStandalonePrefixCache(
            tokens: tokens,
            outputURL: outputURL,
            pinAsSharedBase: pinAsSharedBase,
            onPrefillProgress: onPrefillProgress
        )
    }

    public func mountStandalonePrefixCache(from fileURL: URL) throws -> (tokens: [Int], byteCount: Int) {
        guard let engine else {
            throw QWEngineError.state("FlashNextModule has no engine yet")
        }
        return try engine.mountStandalonePrefixCache(from: fileURL)
    }

    public func unmountSharedBasePrefixes() {
        engine?.unmountSharedBasePrefixes()
    }
}
