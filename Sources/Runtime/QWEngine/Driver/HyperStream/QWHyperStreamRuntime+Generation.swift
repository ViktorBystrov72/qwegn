// QWHyperStreamRuntime+Generation.swift — Generation, speculation policy, and telemetry on QWHyperStreamRuntime.

import Foundation
import QWCore
import QWGEMM
import QWHAL
import QWHead
import QWMTP

/// HyperStream speculative-decode policy helpers shared across resolution and generation.
enum FNGPUSessionState {
    /// Whether the session runs the speculative path (`effectiveDrafts > 0` and supported).
    static func useSpec(sampling: QWSamplerConfig, speculative: QWSpeculativeParameters,
                        runtime: QWHyperStreamRuntime) -> Bool {
        guard !(sampling.hasExplicitSeed && !sampling.isGreedy) else { return false }
        return speculative.effectiveDrafts > 0 && runtime.supportsSpeculativeDecode
    }

    /// Which drafter the session runs: hybrid, ngram, or neural MTP.
    static func draftMode(
        for speculative: QWSpeculativeParameters,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> QWDraftStrategy {
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

extension QWHyperStreamRuntime {
    public var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)? {
        get { stateLock.withLock { _onCacheEvent } }
        set { stateLock.withLock { _onCacheEvent = newValue } }
    }

    public var prefixCacheHit: Bool { stateLock.withLock { _prefixCacheHit } }
    public var prefixCacheMatchedTokens: Int { stateLock.withLock { _prefixCacheMatchedTokens } }
    public var prefixCacheTier: FNPrefixCacheTier? { stateLock.withLock { _prefixCacheTier } }

    public internal(set) var naxLogMessage: String {
        get { stateLock.withLock { _naxLogMessage } }
        set { stateLock.withLock { _naxLogMessage = newValue } }
    }

    public internal(set) var lastResult: QWGenerationResult? {
        get { stateLock.withLock { _lastResult } }
        set { stateLock.withLock { _lastResult = newValue } }
    }

    func wireRuntimeCallbacks() {
        batchScheduler.onCacheEvaluated = { [weak self] sessionData in
            self?.recordCacheEvaluation(sessionData)
        }
        batchScheduler.onSlot0CacheEvaluated = nil
        batchScheduler.onStepTiming = { gpuMs, wallMs in
            if gpuMs > 0 && wallMs > 0 {
                QWTelemetryRegistry.shared.record("gpu.busy_ratio", gpuMs / wallMs, provenance: .measured)
            }
        }
        onCacheRetained = { [weak self] tokens, chargedBytes, storeTotalBytes, warmLimitBytes in
            self?.onCacheEvent?(.retained(tokens: tokens, chargedBytes: chargedBytes,
                                          storeTotalBytes: storeTotalBytes, warmLimitBytes: warmLimitBytes))
        }
        onCacheSkippedOversize = { [weak self] tokens, charge, warmLimitBytes in
            self?.onCacheEvent?(.skippedOversize(tokens: tokens, charge: charge, warmLimitBytes: warmLimitBytes))
        }
        onCacheEvicted = { [weak self] tokens, freedBytes in
            self?.onCacheEvent?(.evicted(tokens: tokens, freedBytes: freedBytes))
        }
        wireWarmStoreCallbacks()
    }

    func wireWarmStoreCallbacks() {
        prefixCacheWarmStore?.onEvict = { [weak self] tokCount, freedBytes in
            self?.onCacheEvicted?(tokCount, freedBytes)
        }
    }

    func recordCacheEvaluation(_ sessionData: FNGPUSessionData) {
        let (hit, matched, tier, sink) = stateLock.withLock {
            _prefixCacheHit = sessionData.prefixCacheHit
            _prefixCacheMatchedTokens = sessionData.prefixCacheMatchedTokens
            _prefixCacheTier = sessionData.prefixCacheTier
            return (_prefixCacheHit, _prefixCacheMatchedTokens, _prefixCacheTier, _onCacheEvent)
        }
        guard let sink else { return }
        if hit {
            sink(.hit(tier: tier?.rawValue ?? "",
                      matchedTokens: matched,
                      totalTokens: sessionData.history.count))
        } else if hasPrefixCacheTier {
            sink(.miss(totalTokens: sessionData.history.count))
        }
    }

    func recordGenerationResult(_ result: QWGenerationResult, specProposals: Int, specAccepted: Int) {
        stateLock.withLock {
            _lastResult = result
            _lastSpecProposals = specProposals
            _lastSpecAccepted = specAccepted
        }
    }

    /// Resolves HyperStream speculation parameters consistently for CLI, server, and runtime.
    public static func resolveSpeculation(
        _ requested: QWSpeculativeParameters,
        descriptor: QWModelDescriptor,
        environment: [String: String],
        modelSwitch: Bool
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
            return QWResolvedSpeculation(
                parameters: parameters, engineDrafts: drafts,
                engineBackend: drafts > 0 ? .dflash2 : nil,
                reportedDrafts: drafts, reportedBackend: backendName)
        }
        let engineDrafts = parameters.enabled && descriptor.capabilities.contains(.mtp) ? parameters.mtpDrafts : 0
        return QWResolvedSpeculation(
            parameters: parameters, engineDrafts: engineDrafts,
            engineBackend: nil,
            reportedDrafts: parameters.enabled ? max(0, min(FNGPUSessionData.defaultMTPDrafts, parameters.mtpDrafts)) : 0,
            reportedBackend: mode.rawValue)
    }

    func buildCapabilities() -> Set<QWKnobCapability> {
        var caps = QWModelCapabilities.resolve(descriptor: descriptor, hasVisionWeights: hasVisionWeights)
        if !supportsSpeculativeDecode { caps.remove(.speculative) }
        return caps
    }

    var prefixCacheState: QWPrefixCacheState {
        guard hasPrefixCacheTier else { return QWPrefixCacheState(isConfigured: false) }
        let disk = prefixCacheDiskState
        let warm = prefixCacheWarmState
        return QWPrefixCacheState(
            isConfigured: true, warmBytes: warm?.bytes, ssdBytes: disk?.bytes,
            warmEntries: warm?.entries, ssdEntries: disk?.entries, warmLimitBytes: prefixCacheWarmLimitBytes)
    }

    func buildReport() -> QWEngineReport {
        QWEngineReport(
            capacityLog: capacityLogMessage,
            arenaBytes: arenaBytes,
            weightUsedBytes: weightUsedBytes,
            weightReservedBytes: weightReservedBytes,
            steadyAllocations: steadyStateAllocations,
            pipelineSeconds: QWHalPSOStats.seconds,
            detailLines: naxLogMessage + prefillStackNote + dflashReportLine,
            contextCapacity: capacity,
            weightFormat: headWeightFormatLabel
        )
    }

    func configurePrefixCache(_ config: QWCacheConfig) throws {
        try batchScheduler.runExclusive {
            let warmBytes = max(0, config.warmByteLimit)
            let ssdBytes = max(0, config.ssdByteLimit)
            let entryLimit = max(1, config.entryLimit)
            self.prefixCacheSSDByteLimit = ssdBytes
            if warmBytes > 0 {
                self.prefixCacheWarmStore = FNPrefixWarmStore(byteLimit: warmBytes, maxSlots: entryLimit)
            } else {
                self.prefixCacheWarmStore = nil
            }
            if ssdBytes > 0 {
                let dir = config.cacheDirectory
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                if config.wipeOnStart, let items = try? FileManager.default.contentsOfDirectory(
                    at: dir, includingPropertiesForKeys: nil) {
                    for url in items where url.lastPathComponent.hasPrefix("fn_entry_") && url.pathExtension == "qwcache" {
                        try? FileManager.default.removeItem(at: url)
                    }
                }
                self.prefixCacheDirectory = dir
            } else {
                self.prefixCacheDirectory = nil
            }
            wireWarmStoreCallbacks()
        }
    }

    /// Generates tokens for a request through `QWHyperStreamRuntime`'s continuous batch scheduler.
    func runGeneration(
        request: QWGenerationRequest,
        onToken: ((Int) -> Bool)?,
        onStopToken: ((Int) -> Void)?
    ) throws -> QWGenerationResult {
        guard !request.prompt.isEmpty else {
            return QWGenerationResult(tokens: [], stepMs: [], stepTokenCounts: [])
        }
        for token in request.prompt where token < 0 || token >= plan.vocab {
            throw QWModuleError.invalidToken(token)
        }
        let maxCap = capacity
        guard request.prompt.count < maxCap else {
            throw FNGPUError.contextDoesNotFit(
                requested: request.prompt.count + 1,
                maxFitting: max(0, maxCap - 1)
            )
        }
        let effectiveMaxNew = min(max(0, request.maxNew), max(0, maxCap - request.prompt.count))
        naxLogMessage = QWGemm.configureDenseNax(
            temperature: Double(request.sampling.temperature),
            topP: Double(request.sampling.topP),
            topK: request.sampling.topK
        ).logMessage
        let useSpec = FNGPUSessionState.useSpec(
            sampling: request.sampling, speculative: request.speculative, runtime: self)
        let draftMode = try FNGPUSessionState.draftMode(for: request.speculative)
        let specCap = dflash != nil
            ? min(FNGPUDFlash.maxDrafts, dflashMaxDrafts)
            : FNGPUSessionData.defaultMTPDrafts
        let job = FNBatchJob(
            prompt: request.prompt,
            maxNew: effectiveMaxNew,
            stopIDs: request.stopIDs,
            sampling: request.sampling,
            useSpec: useSpec && !request.collectLogprobs,
            draftMode: draftMode,
            maxDrafts: max(1, min(specCap, request.speculative.effectiveDrafts)),
            relaxedMargin: request.sampling.hasExplicitSeed ? 0 : request.speculative.effectiveRelaxedMargin,
            relaxedMinProb: request.speculative.effectiveRelaxedMinProb,
            quadraticSeeding: request.speculative.mtpQuadraticSeeding,
            collectLogprobs: request.collectLogprobs,
            topLogprobsCount: request.topLogprobsCount,
            onPrefillProgress: request.onPrefillProgress,
            onToken: onToken,
            onStopToken: onStopToken,
            onTokenLogprob: request.onTokenLogprob,
            constraint: request.constraint,
            retainPrefixLength: request.retainPrefixLength,
            visualSpans: request.visualSpans,
            usePrefixCache: request.usePrefixCache,
            cacheSalt: request.cacheSalt
        )
        let out: FNBatchJobOutput
        do {
            out = try batchScheduler.submit(job)
        } catch FNGPUError.invalidToken(let tok) {
            throw QWModuleError.invalidToken(tok)
        } catch let error as QWEngineError {
            throw QWModuleError.from(error)
        }
        let cache = hasPrefixCacheTier
            ? QWCacheAnswer(hit: out.prefixCacheHit, matchedTokens: out.prefixCacheMatchedTokens,
                            totalTokens: request.prompt.count, tier: out.prefixCacheTier?.rawValue)
            : nil
        var specSummary: String? = nil
        if out.specProposals > 0 {
            let rounds = max(1, out.stepTokenCounts.count)
            let totalTok = out.stepTokenCounts.reduce(0, +)
            let specLabel = dflash.map { $0.config.isDSpine ? "dspine" : "dflash2" } ?? "mtp"
            specSummary = String(format: "%@=%d/%d, rounds=%d, mean=%.2f tok/rnd",
                                 specLabel, out.specAccepted, out.specProposals,
                                 rounds, Double(totalTok) / Double(rounds))
        }
        let finalResult = QWGenerationResult(
            tokens: out.tokens, stepMs: out.stepMs, stepTokenCounts: out.stepTokenCounts,
            prefillSeconds: out.prefillSeconds, firstTokenSeconds: out.firstTokenSeconds,
            decodeSeconds: out.decodeSeconds, prefillGPUSeconds: out.prefillGPUSeconds,
            stopTokenID: out.stopTokenID, cache: cache, speculativeSummary: specSummary,
            tokenLogprobs: out.tokenLogprobs)
        recordGenerationResult(finalResult, specProposals: out.specProposals, specAccepted: out.specAccepted)
        return finalResult
    }

    /// Direct generation façade preserving `QWHyperStreamRuntime.generate(...)`.
    public func generate(
        prompt: [Int],
        maxNew: Int,
        sampling: QWSamplingParameters = QWSamplingParameters(temperature: 0, topP: 1, topK: 1, repetitionPenalty: 1.0),
        speculative: QWSpeculativeParameters = QWSpeculativeParameters(),
        stopIds: Set<Int> = [],
        visualSpans: [QWVisualSpan] = [],
        onToken: ((Int) -> Bool)? = nil,
        onStopToken: ((Int) -> Void)? = nil
    ) throws -> GenerationResult {
        guard !prompt.isEmpty else {
            return GenerationResult(tokens: [], prefillSec: nil, prefillGPUSec: nil,
                                    decodeSec: nil, firstTokenSec: nil,
                                    stepMs: [], denseNaxLogMessage: "")
        }
        let contract = QWSamplerConfig(sampling, vocabulary: plan.vocab)
        let result = try runGeneration(
            request: QWGenerationRequest(prompt: prompt, maxNew: maxNew, stopIDs: stopIds,
                                         sampling: contract, speculative: speculative,
                                         visualSpans: visualSpans),
            onToken: onToken, onStopToken: onStopToken)
        return GenerationResult(tokens: result.tokens,
                                prefillSec: result.prefillSeconds,
                                prefillGPUSec: result.prefillGPUSeconds,
                                decodeSec: result.decodeSeconds,
                                firstTokenSec: result.firstTokenSeconds,
                                stepMs: result.stepMs,
                                denseNaxLogMessage: naxLogMessage,
                                prefixCacheHit: prefixCacheHit,
                                prefixCacheMatchedTokens: prefixCacheMatchedTokens)
    }
}
