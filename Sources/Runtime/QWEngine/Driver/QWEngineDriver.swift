// QWEngineDriver.swift — C23: the kernel-encoding integration leg (spec-21
// "leg 2/3": the plans QWEngine.swift binds become REAL launches here).
//
// Zero new math law: every compute step rides a gated door exactly as its
// parity test drove it. The ONLY new kernels are the integration epilogues
// (qwengine.metal): the q̂/k̂ l2-norm and the β/γ gate folds the PYTHON
// oracle used to compute between doors — no component ever needed a device
import Foundation
import Darwin
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWMLP
import QWHead
import QWMTP
import QWQuant

public final class QWEngineDriver: @unchecked Sendable {
    static let P = "model.language_model."
    /// Geometry of the loaded model, read from its own config.json.
    public let geometry: QWModelGeometry
    /// The container this model's weights are held in, and so the basis its
    /// activations are in.
    public let weightFormat: QWWeightFormat
    /// Why the size-preserving family was declined, when it was.
    public let weightFormatDeclineReason: String?

    let hal: QWHal
    let pack: QWQuant.Pack?
    let embedReader: QWSafeTensorsReader?
    let embedScalesReader: QWSafeTensorsReader?
    let embedBiasesReader: QWSafeTensorsReader?
    let arena: QWArena
    let kvCache: QWKVCache
    let ropeApply: QWRopeApply
    public let ropeRows: Int
    let attnLayers: [Int]
    let hcaLayers: [Int]
    let gdnLayers: [Int]

    let attnDecode: QWAttnDecode?
    let pagedAttnPrefill: QWAttnPagedPrefill
    /// The standard path's paged prefill: one scalar kernel over the pool.
    let pagedAttnPrefillGeneral: QWAttnPagedPrefillGeneral
    let gdnConv: QWGDNConv
    let gdnDecode: QWGDNDecode
    let gdnPrefill: QWGDNPrefill
    let head: QWHead
    let psoBf16F16: any MTLComputePipelineState
    let psoGEMVDxH16T4: any MTLComputePipelineState
    let psoGEMVDxH16DualT4: any MTLComputePipelineState
    let psoGEMVDxH16GdnInProjT4: any MTLComputePipelineState
    let psoGEMVSiluPairH16: any MTLComputePipelineState
    let psoFwht32: any MTLComputePipelineState
    let psoRMSNorm: any MTLComputePipelineState
    let psoRMSNormResidual: any MTLComputePipelineState
    let psoRMSNormResidualWide: any MTLComputePipelineState
    let psoHeadF16: any MTLComputePipelineState
    let psoGemmBatchedH16: any MTLComputePipelineState
    let psoGemmBatchedH16GdnInProj: any MTLComputePipelineState
    let psoSiluMulF16: any MTLComputePipelineState
    let psoKVPagedWrite: any MTLComputePipelineState
    /// The general writer: a head narrower than 256 halves (C103).
    let psoKVPagedWriteGeneral: any MTLComputePipelineState
    let psoZeroPadAttn: any MTLComputePipelineState
    let psoMHCEnter: (any MTLComputePipelineState)?
    let psoMHCRouterNorm: (any MTLComputePipelineState)?
    let psoMHCRouterNormRows: (any MTLComputePipelineState)?
    let psoMHCMerge: (any MTLComputePipelineState)?
    let psoMHCExitHead: (any MTLComputePipelineState)?
    let mhcSplit: QWMHCSplitRouter?
    let attnPrefill: QWAttnPrefill
    let gdnIndexMap: [Int]
    let attnIndexMap: [Int]
    let hcaIndexMap: [Int]
    let prefillModeLock = NSLock()
    var storedPrefillMode: GDNPrefillMode = .chunked
    public var gdnPrefillMode: GDNPrefillMode {
        get { prefillModeLock.withLock { storedPrefillMode } }
        set { prefillModeLock.withLock { storedPrefillMode = newValue } }
    }
    public var gdnSolveBackend: QWGDNPrefill.SolveBackend {
        get { gdnPrefill.solveBackend }
        set { gdnPrefill.solveBackend = newValue }
    }

    // Weight views are module-internal: an internal property may not expose a private type.
    let layersW: [LayerW]
    /// The kernel each layer runs. The plan resolved the kinds when the
    /// model was loaded, so the hot loop looks a kernel up instead of
    /// switching on the kind.
    let layerKernels: [any QWLayerKernel]

    /// 15 MTP Layer weights (Phase 4, section 5A.4 / 5A.5).
    typealias MTPWeights = QWMTPDraftWeights
    let mtpWeights: MTPWeights?
    let echoWeights: EchoDraftWeights?
    let mhcExitHead: MHCExitHeadWeights?
    let multimaxParams: QWTensorValue?
    // P6 FIX: the 2.5 GB f16 embed table is GONE from GPU memory — it
    // existed only so copyEmbedRow could memcpy a row. A decode step now
    // pulls its ONE 10 KiB row straight from the pack's mmap'd bf16 plane
    let embedPlane: String                       // pack name of the bf16 embed plane
    let embedRawOff: Int                         // cached byte offset in pack.data
    let mtpDeviceEmbedding: QWMTPEmbeddingTable?
    public let mtpAdaptiveDepth: Bool
    public let mtpAdaptiveEntropyThreshold: Double
    public var mtpRelaxedMargin: Double
    public var mtpDraftStrategy: String
    public var mtpQuadraticSeeding: Bool
    var activeRelaxedMarginOverride: Float?
    var activeCollectLogprobs: Bool = false
    var activeConstraint: (constraint: any QWTokenConstraint, sampler: QWSampler)? = nil
    var activeVisualSpans: [QWVisualSpan] = []
    var activeCacheSalt: String? = nil
    public var mtpRelaxedMinProb: Double = QWSpeculativeParameters.defaultRelaxedMinProb

    /// Effective log-probability / logit margin for Variational MTP verification.
    public var effectiveRelaxedMargin: Float {
        if let override = activeRelaxedMarginOverride {
            return override
        }
        guard mtpBackend == .variational else { return 0 }
        return Float(max(0.0, mtpRelaxedMargin))
    }
    /// EXP-12: batched q4g verify head (reuses mtpDraftHead's planes).
    let mtpVerifyHeadQ4: Bool
    let mtpStageProfiling: Bool
    let mtpVerificationProfiling: Bool
    var verificationGpuMsByStage: [String: Double]?
    let mtpBatchedMLP: QWMTPBatchedMLP?
    let mtpBatchedTrunk: QWMTPBatchedTrunk?
    let mtpSharedAttentionPartials: QWMRegion?
    var pendingMTPProfile: (draft: QWMTPStageTiming, verification: QWMTPStageTiming)?
    var pendingMTPAdaptiveEntropy: [Double]?
    var lastMTPRoundProfile: QWMTPRoundProfile?
    let embedStaging: any MTLBuffer              // one [H] f16 host-shared row buffer (init-time)
    let headStorage: QWHeadStorage
    let finalNorm: QWTensorValue
    /// L1 (P4-AD-60): the draft-only q4g head (nil = the bf16 head serves
    /// both draft and verify — the pre-L1 behavior). File-backed via mmap;
    /// the weight arena never carries it.
    let mtpDraftHead: QWMTPTDraftHead?

    public internal(set) var mtpDraftProposals: Int = 0
    public internal(set) var mtpDraftAccepted: Int = 0


    /// Dedicated MTP draft scratch buffers (Phase 4, section 5A.6).
    typealias MTPScratch = QWMTPDraftScratch
    let mtpScratch: MTPScratch?
    let mtpDraftEncoder: QWMTPDraftEncoder?
    let echoDrafter: QWEchoDraftEncoder?
    let dflashDrafter: QWDFlashDrafter?
    var lastRoundUsedPLD: Bool = false
    var lastDraftedPairs: Int = 0
    public var dflashDraftPackPath: String? { dflashDrafter?.packPath }
    public var dflashSelectorTopK: Int? { dflashDrafter?.config.selectorTopK }
    let gdnReplayCheckpoint: QWGDNReplayCheckpoint?
    let mtpPromptEmbeddings: QWTensorValue?
    let verifyWindow: QWVerifyWindow?
    var verifiedSession: QWEngineSession?
    var verifiedInputs: [Int] = []
    var preparedGreedyLease: QWGreedyRoundLease?
    var preparedStochasticWitness: QWStochasticRoundWitness?
    let requestGate = QWGenerationRequestGate()
    var prefixStore: QWEnginePrefixStore<QWEngineCachedPrefix>?
    var cacheDiskManager: QWCacheDiskManager?
    public internal(set) var lastPrefixCacheStats: QWPrefixCacheStats?
    let mtpCache: QWKVCache?
    public let mtpDrafts: Int
    /// Performance and memory tunables, environment overrides applied.
    public let tuning: QWEngineTuning
    public let kvMode: QWKVMode
    public let capacityPlan: QWCapacityPlan
    public let capacityLogMessage: String
    /// The MTP execution backend (LighteningMTP L1). The selection is
    /// immutable for the driver lifetime and is part of the session and
    /// prefix-cache identity; an explicitly selected backend cannot be
    public let mtpBackend: QWMTPBackend

    /// The prefix-cache policy key (LighteningMTP L1). It carries the
    /// backend's numerical identity: a Standard and a Lightening run are
    /// never admitted to each other's cached prefixes, even while the L1
    static func prefixPolicyKey(backend: QWMTPBackend, drafts: Int,
                                mode: GDNPrefillMode, cacheSalt: String? = nil) -> String {
        let base = drafts > 0 ? "\(backend.rawValue)+\(mode.rawValue)" : mode.rawValue
        if let salt = cacheSalt?.trimmingCharacters(in: .whitespacesAndNewlines), !salt.isEmpty {
            return "\(base)|salt=\(salt)"
        }
        return base
    }

    let sc: Scratch
    let psc: PrefillScratch
    public let prefillScratchEviction: QWPrefillScratchEvictionPolicy
    public let prefillChunkSize: Int
    /// Latent Sparse Attention configuration for this driver instance.
    public let lsaConfig: QWLSAConfig
    /// Latent Sparse Attention GPU operator (nil when `lsaConfig.mode == .off` or no full-attention layers).
    public let lsaOperator: QWLSAOperator?
    /// Native NoPE / Decoupled-RoPE MLA + Key-Pool Indexer operator.
    public let mlaOperator: QWMLAAttnOperator?
    /// Native Hybrid Compressed Attention (HCA) operator.
    public let hcaOperator: QWHCAAttnOperator?
    let hcaLsaBuffer: (any MTLBuffer)?
    /// The frozen numeric plan of the loaded model.
    public let plan: QWEnginePlan
    let pscBuffer: any MTLBuffer
    let pscByteOffset: Int
    let pscByteLength: Int
    let flight = NSLock()          // single-flight admission
    var activeCB: (any MTLCommandBuffer)?
    var activeEnc: (any MTLComputeCommandEncoder)?
    var pendingColdUnloads: [(streamer: ColdExpertStreamer, stagedCount: Int)] = []
    /// Every command buffer a decode step splits into (cold staging ends and
    /// reopens the CB per MoE layer). Non-nil only inside `forwardStep`, so the
    /// step's GPU time covers all of them, not just the last.
    var stepCBs: [any MTLCommandBuffer]?
    var decodeTel = QWDecodeTelemetry()
    var coldEvents: ColdEventPipeline?
    var spanMapIds: (any MTLBuffer)?  // qwg_moe_span_ids output of the current prefill span (host-read)

    // Pre-carved GDN session slots (two: the engine holds at most two live
    // sequences — one computing, one being admitted; more would share the
    // step scratch anyway under the single-flight law). Each slot carries
    let slotLock = NSLock()
    var freeSlots: [GdnSlot] = []

    public init(packPath: String, maxTokens: Int = 4096, ropeRows: Int = 8192,
                kvMode: QWKVMode = .f16,
                gdnPrefillMode: GDNPrefillMode = .chunked,
                weightArenaSizing: QWWeightArenaSizing = .planned,
                mtpDrafts: Int = 0,
                mtpBackend: QWMTPBackend? = nil,
                prefillScratchEviction: QWPrefillScratchEvictionPolicy = .auto,
                mtpAdaptiveDepth: Bool? = nil,
                mtpAdaptiveEntropyThreshold: Double? = nil,
                mtpRelaxedMargin: Double = 1.5,
                mtpDraftStrategy: String = "hybrid",
                mtpQuadraticSeeding: Bool = true,
                ropeYarnFactor: Double? = nil,
                dflashDraftPackPath: String? = nil,
                dflashSelectorTopK: Int? = nil,
                osReserveGiB: Double? = nil,
                prefillChunkSize: Int = QWPrefillLimits.chunkRows,
                prewarmDoors: [String] = DenseModule.requiredDoors,
                lsaConfig: QWLSAConfig = QWLSAConfig(),
                residentExpertsPerLayer: Int? = nil,
                tuning: QWEngineTuning = .default) throws {
        self.kvMode = kvMode
        self.lsaConfig = lsaConfig
        let cfg = try Self.resolveStartup(packPath: packPath, kvMode: kvMode,
                                          mtpDrafts: mtpDrafts, mtpBackend: mtpBackend,
                                          prefillScratchEviction: prefillScratchEviction,
                                          mtpAdaptiveDepth: mtpAdaptiveDepth,
                                          mtpAdaptiveEntropyThreshold: mtpAdaptiveEntropyThreshold,
                                          mtpRelaxedMargin: mtpRelaxedMargin,
                                          mtpDraftStrategy: mtpDraftStrategy,
                                          mtpQuadraticSeeding: mtpQuadraticSeeding,
                                          dflashDraftPackPath: dflashDraftPackPath,
                                          dflashSelectorTopK: dflashSelectorTopK)
        let startup = try Self.prepareContainersAndGeometry(
            packPath: packPath, maxTokens: maxTokens, ropeRows: ropeRows, kvMode: kvMode,
            prefillChunkSize: prefillChunkSize, mtpDrafts: mtpDrafts,
            residentExpertsPerLayer: residentExpertsPerLayer, osReserveGiB: osReserveGiB,
            cfg: cfg, baseTuning: tuning)
        let chunkRows = startup.prefillChunkSize
        let geometry = startup.geometry
        let prelimHal = startup.hal
        let stContainer = startup.stContainer
        self.pack = startup.pack
        self.mtpDrafts = startup.mtpDrafts
        self.weightFormat = startup.weightFormat
        self.weightFormatDeclineReason = startup.weightFormatDeclineReason
        self.tuning = startup.tuning
        self.geometry = geometry
        self.hal = prelimHal
        self.mtpBackend = cfg.mtpBackend
        self.prefillChunkSize = chunkRows
        self.plan = try QWEngineBootstrap.makePlan(
            geometry: geometry, packPath: packPath,
            prefillChunkRows: chunkRows, kvStorageMode: String(describing: kvMode))
        self.prefillScratchEviction = cfg.prefillScratchEviction
        self.mtpAdaptiveDepth = cfg.mtpAdaptiveDepth
        self.mtpAdaptiveEntropyThreshold = cfg.mtpAdaptiveEntropyThreshold
        self.mtpRelaxedMargin = cfg.mtpRelaxedMargin
        self.mtpDraftStrategy = cfg.mtpDraftStrategy
        self.mtpQuadraticSeeding = cfg.mtpQuadraticSeeding
        self.mtpStageProfiling = cfg.mtpStageProfiling
        self.mtpVerificationProfiling = cfg.mtpVerificationProfiling
        self.mtpVerifyHeadQ4 = cfg.verifyHeadQ4Enabled
        self.storedPrefillMode = gdnPrefillMode
        let environment = cfg.environment
        let numV = geometry.numVirtualLayers
        let attnL = (0..<numV).filter { geometry.isVirtualLayerFullAttention($0) }
        let hcaL = (0..<numV).filter { geometry.isVirtualLayerCompressedAttention($0) }
        let gdnL = (0..<numV).filter { geometry.isVirtualLayerGDN($0) }
        self.attnLayers = attnL; self.hcaLayers = hcaL; self.gdnLayers = gdnL

        let pipes = try Self.resolveCorePipelines(
            hal: prelimHal, weightFormat: startup.weightFormat, mhcConfig: geometry.mhcConfig)
        self.psoBf16F16 = pipes.psoBf16F16; self.psoGEMVDxH16T4 = pipes.psoGEMVDxH16T4
        self.psoGEMVDxH16DualT4 = pipes.psoGEMVDxH16DualT4
        self.psoGEMVDxH16GdnInProjT4 = pipes.psoGEMVDxH16GdnInProjT4
        self.psoGEMVSiluPairH16 = pipes.psoGEMVSiluPairH16; self.psoFwht32 = pipes.psoFwht32
        self.psoRMSNorm = pipes.psoRMSNorm; self.psoRMSNormResidual = pipes.psoRMSNormResidual
        self.psoRMSNormResidualWide = pipes.psoRMSNormResidualWide; self.psoHeadF16 = pipes.psoHeadF16
        self.psoGemmBatchedH16 = pipes.psoGemmBatchedH16
        self.psoGemmBatchedH16GdnInProj = pipes.psoGemmBatchedH16GdnInProj
        self.psoSiluMulF16 = pipes.psoSiluMulF16; self.psoKVPagedWrite = pipes.psoKVPagedWrite
        self.psoKVPagedWriteGeneral = pipes.psoKVPagedWriteGeneral
        self.psoZeroPadAttn = pipes.psoZeroPadAttn
        self.psoMHCEnter = pipes.psoMHCEnter; self.psoMHCRouterNorm = pipes.psoMHCRouterNorm
        self.psoMHCRouterNormRows = pipes.psoMHCRouterNormRows
        self.psoMHCMerge = pipes.psoMHCMerge; self.psoMHCExitHead = pipes.psoMHCExitHead
        self.mhcSplit = try QWMHCSplitRouter(hal: prelimHal, geometry: geometry)
        self.attnPrefill = try QWAttnPrefill(hal: prelimHal, backend: .naxX64, geometry: geometry)
        self.pagedAttnPrefill = try QWAttnPagedPrefill(hal: prelimHal, geometry: geometry)
        self.pagedAttnPrefillGeneral = try QWAttnPagedPrefillGeneral(hal: prelimHal, geometry: geometry)

        var gMap = [Int](repeating: -1, count: numV); for (i, gl) in gdnL.enumerated() { gMap[gl] = i }
        var aMap = [Int](repeating: -1, count: numV); for (i, al) in attnL.enumerated() { aMap[al] = i }
        var hMap = [Int](repeating: -1, count: numV); for (i, hl) in hcaL.enumerated() { hMap[hl] = i }
        self.gdnIndexMap = gMap; self.attnIndexMap = aMap; self.hcaIndexMap = hMap
        self.head = try QWHead(hal: prelimHal, geometry: geometry)
        self.gdnPrefill = try QWGDNPrefill(hal: prelimHal, scanBackend: .hybrid, solveBackend: .scalar)

        let budget = try Self.planArenaAndBudget(
            geometry: geometry, hal: prelimHal, weightFormat: startup.weightFormat,
            weightFormatDeclineReason: startup.weightFormatDeclineReason, tuning: startup.tuning,
            osReserveGiB: startup.osReserveGiB, prefillChunkSize: chunkRows, maxTokens: maxTokens,
            ropeRows: ropeRows, kvMode: kvMode, attnLayerCount: attnL.count, gdnLayerCount: gdnL.count,
            cfg: cfg, mtpDrafts: startup.mtpDrafts, mtpWeightDrafts: startup.mtpWeightDrafts)
        self.capacityPlan = budget.capacityPlan
        self.capacityLogMessage = budget.capacityLogMessage
        self.arena = budget.arena

        let wsub = budget.arena.weightsRegion()
        var L: WeightLoader
        if let pk = startup.pack {
            L = WeightLoader(pk: pk, hl: prelimHal, psoCvt: pipes.psoBf16F16,
                             wsub: wsub, plan: budget.weightPlan, geometry: geometry, format: startup.weightFormat)
        } else if let st = stContainer {
            let psoQuant = try prelimHal.pipeline("qwg_quantize_bf16_to_hq4_g32_m5")
            L = WeightLoader(st: st, hl: prelimHal, psoCvt: pipes.psoBf16F16, psoQuantBF16: psoQuant,
                             wsub: wsub, plan: budget.weightPlan, geometry: geometry, format: startup.weightFormat)
        } else {
            throw QWEngineDriverError.packMissing(packPath)
        }

        let embHead = try Self.loadEmbeddingsAndHead(
            loader: &L, pack: startup.pack, stContainer: stContainer, packPath: packPath,
            hal: prelimHal, geometry: geometry, cfg: cfg, mtpDrafts: startup.mtpDrafts, budget: budget)
        self.embedPlane = embHead.embedPlane; self.embedRawOff = embHead.embedRawOff
        self.embedReader = embHead.embedReader; self.embedScalesReader = embHead.embedScalesReader
        self.embedBiasesReader = embHead.embedBiasesReader
        self.mtpDeviceEmbedding = embHead.mtpDeviceEmbedding; self.embedStaging = embHead.embedStaging
        self.mtpDraftHead = embHead.mtpDraftHead; self.headStorage = embHead.headStorage

        let hcaSidecar = Self.loadHCALSASidecar(
            hal: prelimHal, geometry: geometry, lsaConfig: lsaConfig,
            packPath: packPath, pack: startup.pack, safetensors: stContainer, hcaLayers: hcaL)
        self.hcaLsaBuffer = hcaSidecar?.buffer
        L.hcaLsaByLayer = hcaSidecar?.byLayer ?? [:]
        let decoder = try L.loadDecoderWeights(geometry: geometry)
        self.layersW = decoder.layers
        if startup.tuning.expertCache {
            for lw in decoder.layers { lw.moe?.coldStreamer?.enableDynamicResidency() }
        }
        self.layerKernels = try QWLayerKernelRegistry.resolve(layers: decoder.layers, plan: plan)
        self.finalNorm = decoder.finalNorm
        self.mhcExitHead = try L.loadMHCExitHeadIfNeeded()
        self.multimaxParams = try L.loadMultimaxIfNeeded()

        let yarnFactor = try Self.resolveYaRNFactor(environment["QW_ROPE_YARN_FACTOR"] ?? ropeYarnFactor.map { String($0) })
        let yarnConfig: QWYaRNConfig? = yarnFactor.map {
            QWYaRNConfig(factor: $0, originalMaxPosition: geometry.maxPositionEmbeddings)
        } ?? (budget.effectiveMaxTokens > geometry.maxPositionEmbeddings
              ? QWYaRNConfig.config1M(targetContext: budget.effectiveMaxTokens, originalMaxPosition: geometry.maxPositionEmbeddings)
              : nil)
        let tableRopeRows = geometry.usesLatentOrCompressedAttention ? 1 : budget.effectiveRopeRows
        let tables = QWRopeTables(rows: tableRopeRows, yarn: yarnConfig, geometry: geometry)
        self.ropeRows = budget.effectiveRopeRows
        try budget.weightPlan.check(QWTensorShape([tableRopeRows, geometry.rotaryDims], .f32), at: L.weightRequest)
        let ropeApp = try QWRopeApply(tables: tables, hal: prelimHal, weights: wsub, arena: budget.arena, geometry: geometry)
        self.ropeApply = ropeApp
        L.weightRequest += 1
        let eWeights = try L.loadEchoDraftWeightsIfNeeded(drafts: startup.mtpWeightDrafts)
        let mWeights = try Self.loadMTPWeights(&L, drafts: startup.mtpWeightDrafts, geometry: geometry)
        self.echoWeights = eWeights; self.mtpWeights = mWeights
        try budget.weightPlan.finish(requests: L.weightRequest, usedBytes: wsub.usedBytes)
        wsub.markSteadyState()
        startup.pack?.adviseDontNeed()
        stContainer?.adviseDontNeed()

        let aDec = geometry.usesLatentOrCompressedAttention ? nil
            : try QWAttnDecode(hal: prelimHal, sharedPrefixRows: startup.mtpDrafts > 0 ? min(8, startup.mtpDrafts + 1) : nil,
                               kvMode: kvMode, geometry: geometry)
        self.attnDecode = aDec
        self.gdnConv = try QWGDNConv(hal: prelimHal, geometry: geometry)
        self.gdnDecode = try QWGDNDecode(hal: prelimHal)
        let caches = try Self.makeKVCaches(
            hal: prelimHal, arena: budget.arena, attnLayers: attnL, kvMode: kvMode,
            geometry: geometry, mtpDrafts: startup.mtpDrafts,
            trunkKVBytes: budget.trunkKVBytes, mtpKVBytes: budget.mtpKVBytes)
        self.kvCache = caches.kvCache; self.mtpCache = caches.mtpCache

        let asub = budget.arena.activationsRegion(); let gsub = budget.arena.gdnRegion()
        let pOff = try gsub.allocate(bytes: budget.pCarveBytes)
        let mtpPartialsOff: Int = startup.mtpDrafts > 0 ? try gsub.allocate(bytes: budget.pCarveBytes) : 0
        let builtScratch = try Self.makeDecodeAndPrefillScratch(
            asub: asub, gsub: gsub, geometry: geometry, maxPrefillChunk: chunkRows,
            pOff: pOff, pCarveBytes: budget.pCarveBytes, useGroupedNAX: budget.useGroupedNAX)
        self.sc = builtScratch.sc; self.psc = builtScratch.psc
        self.pscBuffer = asub.region.buffer
        self.pscByteOffset = asub.region.base + builtScratch.gsOff
        self.pscByteLength = asub.usedBytes - builtScratch.gsOff
        let dflash: QWDFlashDrafter? = (cfg.mtpBackend == .dflash2 && startup.mtpDrafts > 0)
            ? try QWDFlashDrafter(geometry: geometry, hal: prelimHal, packPath: cfg.dflashDraftPackPath,
                                  drafts: startup.mtpDrafts, selectorTopK: cfg.dflashSelectorTopK,
                                  sessionSlotCount: Self.gdnSlotCount()) : nil
        self.dflashDrafter = dflash
        self.freeSlots = try Self.buildGDNSlots(
            gsub: gsub, geometry: geometry, gdnLayerCount: gdnL.count,
            mtpDrafts: startup.mtpDrafts, dflashDrafter: dflash)

        self.lsaOperator = try QWLSAOperator.makeIfEnabled(
            hal: prelimHal, geometry: geometry, lsaConfig: lsaConfig, packPath: packPath,
            pack: startup.pack, safetensors: stContainer, attnLayers: attnL,
            maxContext: budget.effectiveMaxTokens, maxPrefillChunk: chunkRows)
        let mlaOp = try QWMLAAttnOperator.makeIfEnabled(
            hal: prelimHal, geometry: geometry, attnLayers: attnL,
            maxContext: budget.effectiveMaxTokens, maxPrefillChunk: chunkRows,
            lsaConfig: lsaConfig, extraAttnLayers: startup.mtpDrafts > 0 ? 1 : 0)
        self.mlaOperator = mlaOp
        self.hcaOperator = try QWHCAAttnOperator.makeIfEnabled(
            hal: prelimHal, geometry: geometry, hcaLayers: hcaL,
            maxContext: budget.effectiveMaxTokens, maxPrefillChunk: chunkRows,
            lsaConfig: lsaConfig, indexDim: hcaSidecar?.indexDim ?? QWLSAConfig.defaultIndexDim)

        let mtpRes = try Self.buildMTPVerificationResources(
            geometry: geometry, hal: prelimHal, asub: asub, gsub: gsub,
            attnDecode: aDec, attnLayers: attnL, gdnLayerCount: gdnL.count,
            mlaOperator: mlaOp, layersW: decoder.layers,
            mtpWeights: mWeights, echoWeights: eWeights, dflashDrafter: dflash,
            mtpDrafts: startup.mtpDrafts, effectiveMaxTokens: budget.effectiveMaxTokens,
            maxT: chunkRows, mtpPartialsOff: mtpPartialsOff, pCarveBytes: budget.pCarveBytes,
            headIsQ4g: cfg.headIsQ4g, verifyHeadQ4Enabled: cfg.verifyHeadQ4Enabled,
            useGroupedNAX: budget.useGroupedNAX, scratch: builtScratch.sc,
            ropeApply: ropeApp, effectiveRopeRows: budget.effectiveRopeRows, kvMode: kvMode)
        self.gdnReplayCheckpoint = mtpRes.gdnReplayCheckpoint; self.verifyWindow = mtpRes.verifyWindow
        self.mtpBatchedMLP = mtpRes.mtpBatchedMLP; self.mtpBatchedTrunk = mtpRes.mtpBatchedTrunk
        self.mtpSharedAttentionPartials = mtpRes.mtpSharedAttentionPartials
        self.mtpScratch = mtpRes.mtpScratch; self.mtpPromptEmbeddings = mtpRes.mtpPromptEmbeddings
        self.mtpDraftEncoder = mtpRes.mtpDraftEncoder; self.echoDrafter = mtpRes.echoDrafter

        asub.markSteadyState(); gsub.markSteadyState()
        prelimHal.adoptPendingResidency()
        prelimHal.makeResident(allResidentBuffers())
        try Self.prewarmPipelines(hal: prelimHal, doors: prewarmDoors)
        if let (cb, enc) = prelimHal.begin() {
            enc.useResource(wsub.region.buffer, usage: .read)
            enc.useResource(asub.region.buffer, usage: [.read, .write])
            enc.useResource(gsub.region.buffer, usage: [.read, .write])
            bindResidentResources(enc)
            if let pso = try? prelimHal.pipeline("qwg_cast_f32_to_f16") {
                enc.setComputePipelineState(pso); builtScratch.sc.gs.bind(enc, index: 0); builtScratch.sc.x.bind(enc, index: 1)
                var one: UInt32 = 1; enc.setBytes(&one, length: 4, index: 2)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            }
            prelimHal.end(cb, enc); cb.waitUntilCompleted()
        }
    }

    public var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)?
    var activePrefillProgress: (@Sendable (Int, Int) -> Void)?
    var activePrefillTotalTokens: Int?
    var activePrefillBasePosition: Int = 0
    public internal(set) var lastExperimentalGeneration: QWExperimentalGenerationStats?
    public internal(set) var lastGeneration: QWGenerationStats?
    public internal(set) var lastRequestedBackend: String?
    public internal(set) var lastPromptTokens: Int = 0
    public internal(set) var lastStopToken: Int?
    public let cpuEncodeLedger = QWEncodeLedger()
    public let prefillTimingLedger = QWPrefillTimingLedger()
    public internal(set) var commandBufferErrors: Int = 0
    public let decodeGpuLedger = QWDecodeGPULedger()

    deinit {
        if cacheDiskManager?.config.wipeOnStart == true {
            cacheDiskManager?.purgeAll()
        }
    }
}
