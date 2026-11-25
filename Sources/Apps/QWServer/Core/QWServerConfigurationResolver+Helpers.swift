// QWServerConfigurationResolver+Helpers.swift — Speculative, cache, and tuning config resolution.

import Foundation
import QWCore
import QWEngine
import QWQuant


extension QWServerConfigurationResolver {
    struct ExtraResolvedConfig {
        let dflashDraftPackPath: String
        let mtpBackend: String?
        let dflashSelectorTopK: Int
        let dsparkConfidenceSchedule: Bool?
        let gdnPrefill: String
        let evictPrefillScratch: String
        let prefillChunkSize: Int
        let maxConcurrentSessions: Int
        let lsaMode: String
        let lsaBudget: Int
        let lsaClusterBlock: Int
        let lsaSidecarPath: String
        let lsaCrossLoopReuse: Bool
        let lsaLoopShareStride: Int
        let warmRAM: Double
        let ssdCacheEnabled: Bool
        let ssdCache: Double
        let cacheDir: String
        let logDir: String
        let logFile: String
        let logLevel: String
        let lazy: Bool
        let mtpAdaptiveDepth: Bool
        let mtpEntropyThreshold: Double
        let mtpRelaxedMargin: Double
        let mtpRelaxedMinProb: Double
        let mtpDraftStrategy: String
        let mtpQuadraticSeeding: Bool
        let loadVision: Bool
        let ropeType: String
        let ropeFactor: Double
        let speculativeEnabled: Bool
        let decisionModelPackPath: String
        let decisionComputeUnit: String
        let decisionLoadVision: Bool
        let decisionDefaultScoreReadout: String
        let decisionDefaultTruncation: String
        let decisionAlwaysEmitBox: Bool
        let tuning: QWEngineTuning
    }

    static func resolveSpeculativeAndTuning(
        fileSettings: QWServerSettings, environment: [String: String], sources: inout [String: String],
        resolvedPack: String, resolvedMaxContext: Int,
        cliMTPDrafts: Int? = nil,
        cliMTPBackend: String?, cliDFlashDraftPack: String?,
        cliDFlashBlockSize: Int?, cliDFlashTopK: Int?,
        cliDSparkConfidenceSchedule: Bool? = nil,
        cliGDNPrefill: String?, cliEvictScratch: String?,
        cliPrefillChunkSize: Int?, cliMaxConcurrentSessions: Int?,
        cliLSAMode: String?, cliLSABudget: Int?,
        cliLSAClusterBlock: Int?, cliLSASidecarPath: String?,
        cliLSACrossLoopReuse: Bool? = nil,
        cliWarmRAMLimitGB: Double?, cliSSDCacheLimitGB: Double?,
        cliCacheDir: String?, cliLogDir: String?, cliLogFileName: String?,
        cliLogLevel: String?, cliLazy: Bool?,
        cliMTPAdaptiveDepth: Bool?, cliMTPEntropyThreshold: Double?,
        cliMTPRelaxedMargin: Double?, cliMTPRelaxedMinProb: Double?,
        cliMTPDraftStrategy: String?, cliMTPQuadraticSeeding: Bool?,
        cliLoadVision: Bool?, cliMemoryPolicy: String?,
        cliOSReserveGiB: Double?, cliResidentExpertsPerLayer: Int?,
        cliMoESpecBudget: Int?, cliStreamNgrams: Bool?,
        cliDecisionModelPackPath: String? = nil, cliDecisionComputeUnit: String? = nil,
        cliDecisionLoadVision: Bool? = nil, cliDecisionDefaultScoreReadout: String? = nil,
        cliDecisionDefaultTruncation: String? = nil, cliDecisionAlwaysEmitBox: Bool? = nil,
        drafters: [QWServerSettings.DiscoveredModel]? = nil
    ) -> ExtraResolvedConfig {
        let requestedMTPBackend: String? = {
            if let b = cliMTPBackend, !b.isEmpty { return b }
            if let b = environment["QWG_MTP_BACKEND"], !b.isEmpty { return b }
            if !fileSettings.mtpBackend.isEmpty { return fileSettings.mtpBackend }
            return nil
        }()
        let targetDescriptor = QWServerSettings.loadPrimaryDescriptor(packPath: resolvedPack)
        let resolvedDFlashDraftPackPath: String = {
            if let p = cliDFlashDraftPack, !p.isEmpty { sources["dflashDraftPackPath"] = "cli"; return p.canonicalPath }
            if let p = environment["QW_DFLASH_DRAFT_PACK"] ?? environment["QWG_DFLASH_DRAFT_PACK"], !p.isEmpty {
                sources["dflashDraftPackPath"] = "env"; return p.canonicalPath
            }
            if !fileSettings.dflashDraftPackPath.isEmpty {
                sources["dflashDraftPackPath"] = "config-file"; return fileSettings.dflashDraftPackPath.canonicalPath
            }
            let availableDrafters = drafters ?? QWServerSettings.scanDrafterModelsDirectory()
            if let match = QWServerSettings.resolveDefaultDrafterPack(
                for: targetDescriptor,
                preferredBackend: requestedMTPBackend,
                drafters: availableDrafters,
                targetModelName: fileSettings.modelName
            ) {
                sources["dflashDraftPackPath"] = "discovered (~/.qwg/models/\(match.name))"
                return match.packPath
            }
            sources["dflashDraftPackPath"] = "default"
            if targetDescriptor != nil {
                return ""
            }
            return QWServerSettings.normalizeExternalDrafterBackend(requestedMTPBackend) == "dspine"
                ? QWDFlashConfig.defaultDSpinePackPath
                : QWDFlashConfig.defaultPackPath
        }()

        let resolvedMTPBackend: String? = {
            if let b = cliMTPBackend, !b.isEmpty { sources["mtpBackend"] = "cli"; return b }
            if let bs = cliDFlashBlockSize, (2...16).contains(bs) {
                sources["mtpBackend"] = "cli (--dflash-window)"
                return QWServerSettings.inferDrafterBackendAndMaxDrafts(packPath: resolvedDFlashDraftPackPath).backend
            }
            if let b = environment["QWG_MTP_BACKEND"], !b.isEmpty { sources["mtpBackend"] = "env"; return b }
            if !resolvedPack.isEmpty, let desc = targetDescriptor, desc.dsparkConfig != nil,
               QWModelDescriptor.DSparkConfig.resolvePackURL(near: resolvedPack) != nil,
               !desc.supportsNativeMTP, fileSettings.mtpBackend == "variational" {
                sources["mtpBackend"] = "auto (dspark.qwp)"
                return "dspark"
            }
            if !fileSettings.mtpBackend.isEmpty { sources["mtpBackend"] = "config-file"; return fileSettings.mtpBackend }
            return nil
        }()

        let resolvedDSparkConfidenceSchedule: Bool? = {
            if let ds = cliDSparkConfidenceSchedule {
                sources["dsparkConfidenceSchedule"] = "cli"
                return ds
            }
            if let envStr = environment["QWG_DSPARK_CONFIDENCE_SCHEDULE"] ?? environment["QW_DSPARK_CONFIDENCE_SCHEDULE"] {
                sources["dsparkConfidenceSchedule"] = "env"
                return ["1", "true", "yes"].contains(envStr.lowercased())
            }
            if fileSettings.dsparkConfidenceSchedule {
                sources["dsparkConfidenceSchedule"] = "config-file"
                return true
            }
            return nil
        }()

        let resolvedDFlashSelectorTopK: Int = {
            if let tk = cliDFlashTopK, (1...16).contains(tk) { sources["dflashSelectorTopK"] = "cli"; return tk }
            if let envStr = environment["QW_DFLASH_TOP_K"] ?? environment["QWG_DFLASH_TOP_K"],
               let tk = Int(envStr), (1...16).contains(tk) { sources["dflashSelectorTopK"] = "env"; return tk }
            sources["dflashSelectorTopK"] = "config-file"; return max(1, min(16, fileSettings.dflashSelectorTopK))
        }()

        // 11. GDN Prefill & Eviction
        let resolvedGDNPrefill: String = {
            if let g = cliGDNPrefill, !g.isEmpty { sources["gdnPrefill"] = "cli"; return g }
            if let g = environment["QWG_GDN_PREFILL"], !g.isEmpty { sources["gdnPrefill"] = "env"; return g }
            sources["gdnPrefill"] = "config-file"; return fileSettings.gdnPrefill
        }()

        let resolvedEvictScratch: String = {
            if let e = cliEvictScratch, !e.isEmpty { sources["evictPrefillScratch"] = "cli"; return e }
            if let e = environment["QWG_EVICT_PREFILL_SCRATCH"], !e.isEmpty { sources["evictPrefillScratch"] = "env"; return e }
            sources["evictPrefillScratch"] = "config-file"; return fileSettings.evictPrefillScratch
        }()

        let resolvedPrefillChunkSize: Int = {
            if let c = cliPrefillChunkSize, c >= QWPrefillLimits.minRows {
                sources["prefillChunkSize"] = "cli"; return QWPrefillLimits.clampedRows(c)
            }
            if let envStr = environment["QWG_PREFILL_CHUNK_SIZE"], let c = Int(envStr), c >= QWPrefillLimits.minRows {
                sources["prefillChunkSize"] = "env"; return QWPrefillLimits.clampedRows(c)
            }
            sources["prefillChunkSize"] = "config-file"; return QWPrefillLimits.clampedRows(fileSettings.prefillChunkSize)
        }()

        let resolvedMaxConcurrentSessions: Int = {
            if let s = cliMaxConcurrentSessions, (1...16).contains(s) {
                sources["maxConcurrentSessions"] = "cli"; return s
            }
            if let envStr = environment["QWG_MAX_CONCURRENT_SESSIONS"] ?? environment["QW_MAX_CONCURRENT_SESSIONS"],
               let s = Int(envStr), (1...16).contains(s) {
                sources["maxConcurrentSessions"] = "env"; return s
            }
            sources["maxConcurrentSessions"] = "config-file"; return max(1, min(16, fileSettings.maxConcurrentSessions))
        }()

        let resolvedLSAMode: String = {
            if let m = cliLSAMode, !m.isEmpty { sources["lsaMode"] = "cli"; return m.lowercased() }
            if let m = environment["QWG_LSA_MODE"] ?? environment["QW_LSA_MODE"], !m.isEmpty {
                sources["lsaMode"] = "env"; return m.lowercased()
            }
            sources["lsaMode"] = "config-file"; return fileSettings.lsaMode.lowercased()
        }()

        let resolvedLSABudget: Int = {
            if let b = cliLSABudget { sources["lsaBudget"] = "cli"; return b }
            if let envStr = environment["QWG_LSA_BUDGET"] ?? environment["QW_LSA_BUDGET"], let b = Int(envStr) {
                sources["lsaBudget"] = "env"; return b
            }
            sources["lsaBudget"] = "config-file"; return fileSettings.lsaBudget
        }()

        let resolvedLSAClusterBlock: Int = {
            if let cb = cliLSAClusterBlock { sources["lsaClusterBlock"] = "cli"; return cb }
            if let envStr = environment["QWG_LSA_CLUSTER_BLOCK"] ?? environment["QW_LSA_CLUSTER_BLOCK"], let cb = Int(envStr) {
                sources["lsaClusterBlock"] = "env"; return cb
            }
            sources["lsaClusterBlock"] = "config-file"; return fileSettings.lsaClusterBlock
        }()

        let resolvedLSASidecarPath: String = {
            if let sp = cliLSASidecarPath, !sp.isEmpty { sources["lsaSidecarPath"] = "cli"; return sp }
            if let sp = environment["QWG_LSA_SIDECAR"] ?? environment["QW_LSA_SIDECAR"], !sp.isEmpty {
                sources["lsaSidecarPath"] = "env"; return sp
            }
            sources["lsaSidecarPath"] = "config-file"; return fileSettings.lsaSidecarPath
        }()

        let resolvedLSALoopShare: (enabled: Bool, stride: Int) = {
            let envShare = QWLSAConfig.parseLoopShareEnv(environment[QWLSAConfig.loopShareEnvKey])
            if let cli = cliLSACrossLoopReuse {
                sources["lsaCrossLoopReuse"] = "cli"
                return (cli, cli ? (envShare?.stride ?? 0) : 0)
            }
            if let envShare {
                sources["lsaCrossLoopReuse"] = "env"
                return envShare
            }
            sources["lsaCrossLoopReuse"] = "config-file"
            return (fileSettings.lsaCrossLoopReuse, 0)
        }()

        // 12. Cache
        let resolvedWarmRAM: Double = {
            if let w = cliWarmRAMLimitGB { sources["warmRAM"] = "cli"; return w }
            if let envStr = environment["QWG_WARM_CACHE_LIMIT"], let w = Double(envStr) { sources["warmRAM"] = "env"; return w }
            sources["warmRAM"] = "config-file"; return fileSettings.warmRAMLimitGB
        }()

        let resolvedSSDCacheEnabled: Bool = {
            if let s = cliSSDCacheLimitGB { sources["ssdCacheEnabled"] = "cli"; return s > 0 }
            if let envStr = environment["QWG_SSD_CACHE_ENABLED"] ?? environment["QW_SSD_CACHE_ENABLED"] {
                sources["ssdCacheEnabled"] = "env"; return envStr == "1" || envStr.lowercased() == "true"
            }
            if let envStr = environment["QWG_SSD_CACHE_LIMIT"], let s = Double(envStr) { sources["ssdCacheEnabled"] = "env"; return s > 0 }
            sources["ssdCacheEnabled"] = "config-file"; return fileSettings.ssdCacheEnabled
        }()

        let resolvedSSDCache: Double = {
            if let s = cliSSDCacheLimitGB { sources["ssdCache"] = "cli"; return s }
            if let envStr = environment["QWG_SSD_CACHE_LIMIT"], let s = Double(envStr) { sources["ssdCache"] = "env"; return s }
            sources["ssdCache"] = "config-file"; return resolvedSSDCacheEnabled ? fileSettings.ssdCacheLimitGB : 0.0
        }()

        let resolvedCacheDir: String = {
            if let c = cliCacheDir, !c.isEmpty { sources["cacheDir"] = "cli"; return c }
            if let c = environment["QWG_CACHE_DIR"], !c.isEmpty { sources["cacheDir"] = "env"; return c }
            sources["cacheDir"] = "config-file"; return fileSettings.cacheDir
        }()

        // 13. Logging & Lazy
        let resolvedLogLevel: String = {
            if let l = cliLogLevel, !l.isEmpty { sources["logLevel"] = "cli"; return l }
            if let l = environment["QWG_LOG_LEVEL"], !l.isEmpty { sources["logLevel"] = "env"; return l }
            sources["logLevel"] = "config-file"; return fileSettings.logLevel
        }()

        let resolvedLogDir: String = {
            if let d = cliLogDir, !d.isEmpty { sources["logDir"] = "cli"; return d }
            if let d = environment["QWG_LOG_DIR"], !d.isEmpty { sources["logDir"] = "env"; return d }
            sources["logDir"] = "config-file"; return fileSettings.logDir
        }()

        let resolvedLogFile: String = {
            if let f = cliLogFileName, !f.isEmpty { sources["logFileName"] = "cli"; return f }
            if let f = environment["QWG_LOG_FILE"], !f.isEmpty { sources["logFileName"] = "env"; return f }
            sources["logFileName"] = "config-file"; return fileSettings.logFileName
        }()

        let resolvedLazy = cliLazy ?? fileSettings.lazyLoad

        let (resolvedRopeType, resolvedRopeFactor): (String, Double) = {
            if let envStr = environment["QW_ROPE_YARN_FACTOR"], let factor = Double(envStr), factor > 0 {
                sources["ropeScaling"] = "env (QW_ROPE_YARN_FACTOR=\(factor))"
                return (factor > 1.0 ? "yarn" : "none", factor)
            }
            let geom: QWModelGeometry? = (!resolvedPack.isEmpty
                ? (try? QWModelGeometry.load(packPath: resolvedPack))
                : nil)
            if let geom, resolvedMaxContext > geom.maxPositionEmbeddings {
                let autoFactor = Double(resolvedMaxContext) / Double(geom.originalMaxPositionEmbeddings)
                sources["ropeScaling"] = "auto-derived (context \(resolvedMaxContext) > native \(geom.maxPositionEmbeddings))"
                return ("yarn", autoFactor)
            }
            if fileSettings.ropeScalingType.lowercased() == "none", let nativeYarn = geom?.yarnRopeConfig, nativeYarn.factor > 1.0 {
                sources["ropeScaling"] = "model-config (yarn \(nativeYarn.factor)x)"
                return ("yarn", nativeYarn.factor)
            }
            sources["ropeScaling"] = "config-file"
            return (fileSettings.ropeScalingType, fileSettings.ropeScalingFactor)
        }()

        let resolvedSpeculativeEnabled: Bool = {
            if let d = cliMTPDrafts {
                sources["speculativeDecodingEnabled"] = "cli (--mtp-drafts)"
                return d > 0
            }
            if let bs = cliDFlashBlockSize, (2...16).contains(bs) {
                sources["speculativeDecodingEnabled"] = "cli (--dflash-window)"
                return true
            }
            if let b = cliMTPBackend, !b.isEmpty {
                sources["speculativeDecodingEnabled"] = "cli (--mtp-backend)"
                return true
            }
            if let envStr = environment["QWG_SPECULATIVE_DECODING"] ?? environment["QW_SPECULATIVE"] {
                sources["speculativeDecodingEnabled"] = "env"
                return envStr == "1" || envStr.lowercased() == "true"
            }
            sources["speculativeDecodingEnabled"] = "config-file"; return fileSettings.speculativeDecodingEnabled
        }()

        let resolvedMTPAdaptiveDepth: Bool = {
            if let ad = cliMTPAdaptiveDepth { sources["mtpAdaptiveDepth"] = "cli"; return ad }
            if let envStr = environment["QW_MTP_ADAPTIVE_DEPTH"] ?? environment["QWG_MTP_ADAPTIVE_DEPTH"] {
                sources["mtpAdaptiveDepth"] = "env"; return envStr == "1" || envStr.lowercased() == "true"
            }
            sources["mtpAdaptiveDepth"] = "config-file"; return fileSettings.mtpAdaptiveDepth
        }()

        let resolvedMTPEntropyThreshold: Double = {
            if let et = cliMTPEntropyThreshold { sources["mtpEntropyThreshold"] = "cli"; return et }
            if let envStr = environment["QW_MTP_ADAPTIVE_ENTROPY_THRESHOLD"] ?? environment["QWG_MTP_ADAPTIVE_ENTROPY_THRESHOLD"],
               let val = Double(envStr) { sources["mtpEntropyThreshold"] = "env"; return val }
            sources["mtpEntropyThreshold"] = "config-file"; return fileSettings.mtpEntropyThreshold
        }()

        let resolvedMTPRelaxedMargin: Double = {
            if let rm = cliMTPRelaxedMargin { sources["mtpRelaxedMargin"] = "cli"; return max(0.0, rm) }
            if let envStr = environment["QW_MTP_RELAXED_MARGIN"] ?? environment["QWG_MTP_RELAXED_MARGIN"],
               let val = Double(envStr) { sources["mtpRelaxedMargin"] = "env"; return max(0.0, val) }
            sources["mtpRelaxedMargin"] = "config-file"; return max(0.0, fileSettings.mtpRelaxedMargin)
        }()
        let resolvedMTPRelaxedMinProb = resolveRelaxedMinProb(cli: cliMTPRelaxedMinProb, environment: environment,
                                                              file: fileSettings.mtpRelaxedMinProb, sources: &sources)

        let resolvedMTPDraftStrategy: String = {
            if let ds = cliMTPDraftStrategy, !ds.isEmpty { sources["mtpDraftStrategy"] = "cli"; return ds }
            if let envStr = environment["QW_MTP_DRAFT_STRATEGY"] ?? environment["QWG_MTP_DRAFT_STRATEGY"], !envStr.isEmpty {
                sources["mtpDraftStrategy"] = "env"; return envStr
            }
            sources["mtpDraftStrategy"] = "config-file"; return fileSettings.mtpDraftStrategy
        }()

        let resolvedMTPQuadraticSeeding: Bool = {
            if let qs = cliMTPQuadraticSeeding { sources["mtpQuadraticSeeding"] = "cli"; return qs }
            if let envStr = environment["QW_MTP_QUADRATIC_SEEDING"] ?? environment["QWG_MTP_QUADRATIC_SEEDING"] {
                sources["mtpQuadraticSeeding"] = "env"; return envStr == "1" || envStr.lowercased() == "true"
            }
            sources["mtpQuadraticSeeding"] = "config-file"; return fileSettings.mtpQuadraticSeeding
        }()

        let resolvedLoadVision: Bool = {
            if let cli = cliLoadVision {
                sources["loadVision"] = "cli (--load-vision)"; return cli
            }
            if let envStr = environment["QWG_LOAD_VISION"] ?? environment["QW_LOAD_VISION"] ?? environment["QW_VISION_EAGER"] {
                sources["loadVision"] = "env"; return envStr == "1" || envStr.lowercased() == "true"
            }
            sources["loadVision"] = "config-file"; return fileSettings.loadVision
        }()

        let resolvedDecisionModelPackPath: String = {
            func resolvePath(_ raw: String) -> String {
                let ref = fileSettings.resolveModelReference(raw, allowArbitraryFilesystemPaths: true)
                return ref.packPath.isEmpty ? raw.expandingTildePath : ref.packPath
            }
            if let p = cliDecisionModelPackPath, !p.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                sources["decisionModelPackPath"] = "cli"; return resolvePath(p)
            }
            if let p = environment["QWG_DECISION_MODEL"] ?? environment["QW_DECISION_MODEL"],
               !p.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                sources["decisionModelPackPath"] = "env"; return resolvePath(p)
            }
            let fileP = fileSettings.decisionModelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
            if !fileP.isEmpty {
                sources["decisionModelPackPath"] = "config-file"; return resolvePath(fileP)
            }
            sources["decisionModelPackPath"] = "none"
            return ""
        }()

        let resolvedDecisionComputeUnit: String = {
            if let u = cliDecisionComputeUnit, !u.isEmpty { sources["decisionComputeUnit"] = "cli"; return u.lowercased() }
            if let u = environment["QWG_DECISION_COMPUTE_UNIT"], !u.isEmpty { sources["decisionComputeUnit"] = "env"; return u.lowercased() }
            sources["decisionComputeUnit"] = "config-file"; return fileSettings.decisionComputeUnit.lowercased()
        }()

        let resolvedDecisionLoadVision: Bool = {
            if let v = cliDecisionLoadVision { sources["decisionLoadVision"] = "cli"; return v }
            if let envStr = environment["QWG_DECISION_LOAD_VISION"] {
                sources["decisionLoadVision"] = "env"; return envStr == "1" || envStr.lowercased() == "true"
            }
            sources["decisionLoadVision"] = "config-file"; return fileSettings.decisionLoadVision
        }()

        let resolvedDecisionScoreReadout: String = {
            if let r = cliDecisionDefaultScoreReadout, !r.isEmpty { sources["decisionDefaultScoreReadout"] = "cli"; return r.lowercased() }
            if let r = environment["QWG_DECISION_SCORE_READOUT"], !r.isEmpty { sources["decisionDefaultScoreReadout"] = "env"; return r.lowercased() }
            sources["decisionDefaultScoreReadout"] = "config-file"; return fileSettings.decisionDefaultScoreReadout.lowercased()
        }()

        let resolvedDecisionTruncation: String = {
            if let t = cliDecisionDefaultTruncation, !t.isEmpty { sources["decisionDefaultTruncation"] = "cli"; return t.lowercased() }
            if let t = environment["QWG_DECISION_TRUNCATION"], !t.isEmpty { sources["decisionDefaultTruncation"] = "env"; return t.lowercased() }
            sources["decisionDefaultTruncation"] = "config-file"; return fileSettings.decisionDefaultTruncation.lowercased()
        }()

        let resolvedDecisionAlwaysEmitBox: Bool = {
            if let b = cliDecisionAlwaysEmitBox { sources["decisionAlwaysEmitBox"] = "cli"; return b }
            if let envStr = environment["QWG_DECISION_ALWAYS_EMIT_BOX"] {
                sources["decisionAlwaysEmitBox"] = "env"; return envStr == "1" || envStr.lowercased() == "true"
            }
            sources["decisionAlwaysEmitBox"] = "config-file"; return fileSettings.decisionAlwaysEmitBox
        }()

        var resolvedTuning = fileSettings.resolvedEngineTuning
        if let cliPol = cliMemoryPolicy, let parsed = QWMemoryPolicy(rawValue: cliPol) {
            resolvedTuning.memoryPolicy = parsed
            sources["memoryPolicy"] = "cli (--memory-policy)"
        }
        if let cliOS = cliOSReserveGiB {
            resolvedTuning.osReserveGiB = cliOS
            sources["osReserveGiB"] = "cli (--os-reserve)"
        }
        if let cliRes = cliResidentExpertsPerLayer {
            resolvedTuning.residentExpertsPerLayer = cliRes > 0 ? cliRes : nil
            sources["residentExpertsPerLayer"] = "cli (--resident-experts-per-layer)"
        }
        if let cliBudget = cliMoESpecBudget {
            resolvedTuning.moeSpecBudget = cliBudget
            sources["moeSpecBudget"] = "cli (--moe-spec-budget)"
        }
        if let cliStream = cliStreamNgrams {
            resolvedTuning.streamNgrams = cliStream
            sources["streamNgrams"] = "cli (--stream-ngrams)"
        }

        for id in QWEngineTuning.knobIds where sources[id] == nil {
            if let env = QWEngineTuning.environmentNames[id], environment[env] != nil {
                sources[id] = "env"
            } else if fileSettings.engineTuning[id] != nil {
                sources[id] = "config-file"
            }
        }
        return ExtraResolvedConfig(
            dflashDraftPackPath: resolvedDFlashDraftPackPath,
            mtpBackend: resolvedMTPBackend,
            dflashSelectorTopK: resolvedDFlashSelectorTopK,
            dsparkConfidenceSchedule: resolvedDSparkConfidenceSchedule,
            gdnPrefill: resolvedGDNPrefill,
            evictPrefillScratch: resolvedEvictScratch,
            prefillChunkSize: resolvedPrefillChunkSize,
            maxConcurrentSessions: resolvedMaxConcurrentSessions,
            lsaMode: resolvedLSAMode,
            lsaBudget: resolvedLSABudget,
            lsaClusterBlock: resolvedLSAClusterBlock,
            lsaSidecarPath: resolvedLSASidecarPath,
            lsaCrossLoopReuse: resolvedLSALoopShare.enabled,
            lsaLoopShareStride: resolvedLSALoopShare.stride,
            warmRAM: resolvedWarmRAM,
            ssdCacheEnabled: resolvedSSDCacheEnabled,
            ssdCache: resolvedSSDCache,
            cacheDir: resolvedCacheDir,
            logDir: resolvedLogDir,
            logFile: resolvedLogFile,
            logLevel: resolvedLogLevel,
            lazy: resolvedLazy,
            mtpAdaptiveDepth: resolvedMTPAdaptiveDepth,
            mtpEntropyThreshold: resolvedMTPEntropyThreshold,
            mtpRelaxedMargin: resolvedMTPRelaxedMargin,
            mtpRelaxedMinProb: resolvedMTPRelaxedMinProb,
            mtpDraftStrategy: resolvedMTPDraftStrategy,
            mtpQuadraticSeeding: resolvedMTPQuadraticSeeding,
            loadVision: resolvedLoadVision,
            ropeType: resolvedRopeType,
            ropeFactor: resolvedRopeFactor,
            speculativeEnabled: resolvedSpeculativeEnabled,
            decisionModelPackPath: resolvedDecisionModelPackPath,
            decisionComputeUnit: resolvedDecisionComputeUnit,
            decisionLoadVision: resolvedDecisionLoadVision,
            decisionDefaultScoreReadout: resolvedDecisionScoreReadout,
            decisionDefaultTruncation: resolvedDecisionTruncation,
            decisionAlwaysEmitBox: resolvedDecisionAlwaysEmitBox,
            tuning: resolvedTuning)
    }
}
