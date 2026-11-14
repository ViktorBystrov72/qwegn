// QWEngineDriver+Init.swift — Startup resource builders for QWEngineDriver initialization.

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

extension QWEngineDriver {
    struct StartupContainers {
        let tuning: QWEngineTuning
        let osReserveGiB: Double?
        let geometry: QWModelGeometry
        let hal: QWHal
        let pack: QWQuant.Pack?
        let stContainer: QWSafeTensorsContainer?
        let weightFormat: QWWeightFormat
        let weightFormatDeclineReason: String?
        let mtpDrafts: Int
        let mtpWeightDrafts: Int
    }

    struct CorePipelines {
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
        let psoKVPagedWriteGeneral: any MTLComputePipelineState
        let psoZeroPadAttn: any MTLComputePipelineState
        let psoMHCEnter: (any MTLComputePipelineState)?
        let psoMHCRouterNorm: (any MTLComputePipelineState)?
        let psoMHCRouterNormRows: (any MTLComputePipelineState)?
        let psoMHCMerge: (any MTLComputePipelineState)?
        let psoMHCExitHead: (any MTLComputePipelineState)?
    }

    struct ArenaBudgetPlan {
        let effectiveMaxTokens: Int
        let effectiveRopeRows: Int
        let weightPlan: QWWeightAllocationPlan
        let trunkKVBytes: Int
        let mtpKVBytes: Int
        let actBytes: Int
        let gdnBytes: Int
        let fixedBytes: Int
        let mtpPartialsOff: Int?
        let pCarveBytes: Int
        let useGroupedNAX: Bool
        let capacityPlan: QWCapacityPlan
        let capacityLogMessage: String
        let arena: QWArena
        let availBytes: Int
    }

    struct EmbeddingAndHeadSetup {
        let embedPlane: String
        let embedRawOff: Int
        let embedReader: QWSafeTensorsReader?
        let embedScalesReader: QWSafeTensorsReader?
        let embedBiasesReader: QWSafeTensorsReader?
        let mtpDeviceEmbedding: QWMTPEmbeddingTable?
        let embedStaging: any MTLBuffer
        let mtpDraftHead: QWMTPTDraftHead?
        let headStorage: QWHeadStorage
    }

    struct MTPVerificationResources {
        let gdnReplayCheckpoint: QWGDNReplayCheckpoint?
        let verifyWindow: QWVerifyWindow?
        let mtpBatchedMLP: QWMTPBatchedMLP?
        let mtpBatchedTrunk: QWMTPBatchedTrunk?
        let mtpSharedAttentionPartials: QWMRegion?
        let mtpScratch: MTPScratch?
        let mtpPromptEmbeddings: QWTensorValue?
        let mtpDraftEncoder: QWMTPDraftEncoder?
        let echoDrafter: QWEchoDraftEncoder?
    }

    func prefixPolicyKey(mode: GDNPrefillMode) -> String {
        Self.prefixPolicyKey(backend: mtpBackend, drafts: mtpDrafts, mode: mode)
    }

    func flushPendingColdUnloads() {
        guard !pendingColdUnloads.isEmpty else { return }
        var maxByPtr: [UnsafeMutableRawPointer: (streamer: ColdExpertStreamer, maxCount: Int)] = [:]
        for item in pendingColdUnloads {
            let key = item.streamer.guBasePtr
            if let cur = maxByPtr[key] {
                if item.stagedCount > cur.maxCount {
                    maxByPtr[key] = (item.streamer, item.stagedCount)
                }
            } else {
                maxByPtr[key] = (item.streamer, item.stagedCount)
            }
        }
        for (_, entry) in maxByPtr {
            entry.streamer.unload(stagedCount: entry.maxCount)
        }
        pendingColdUnloads.removeAll(keepingCapacity: true)
    }

    static func prepareContainersAndGeometry(
        packPath: String, maxTokens: Int, ropeRows: Int, kvMode: QWKVMode,
        prefillChunkSize: Int, mtpDrafts requestedMTPDrafts: Int,
        residentExpertsPerLayer: Int?, osReserveGiB explicitOSReserveGiB: Double?,
        cfg: StartupConfig, baseTuning: QWEngineTuning
    ) throws -> StartupContainers {
        let tuning = baseTuning.overlaid(environment: cfg.environment)
        let osReserveGiB = explicitOSReserveGiB ?? tuning.osReserveGiB
        let explicitResE = residentExpertsPerLayer ?? tuning.residentExpertsPerLayer
        let rawGeometry = try QWModelGeometry.load(packPath: packPath).withResidentExperts(explicitResE)
        let root = QWModelRoot(url: URL(fileURLWithPath: packPath))
        let isSafetensors = (root.kind == .safetensorsSingle || root.kind == .safetensorsSharded)
        let stContainer = isSafetensors ? try QWSafeTensorsContainer(url: root.weights) : nil
        let pack = isSafetensors ? nil : try QWQuant.Pack(url: URL(fileURLWithPath: packPath))
        let nextNMTPName = "model.language_model.layers.\(rawGeometry.numLayers).enorm.weight"
        let nextNMTPAltName = "model.layers.\(rawGeometry.numLayers).enorm.weight"
        let hasMTPWeights: Bool
        if let st = stContainer {
            hasMTPWeights = st.tensor("mtp.pre_fc_norm_embedding.weight") != nil
                || st.tensor(nextNMTPName) != nil
                || st.tensor(nextNMTPAltName) != nil
        } else if let pk = pack {
            hasMTPWeights = (try? pk.entry("mtp.pre_fc_norm_embedding.weight")) != nil
                || (try? pk.entry(nextNMTPName)) != nil
                || (try? pk.entry(nextNMTPAltName)) != nil
        } else {
            hasMTPWeights = false
        }
        let mtpDrafts = (cfg.mtpBackend == .dflash2 || hasMTPWeights) ? requestedMTPDrafts : 0
        let mtpWeightDrafts = hasMTPWeights ? mtpDrafts : 0
        try Self.validateGeometry(rawGeometry, kvMode: kvMode)
        let choice = try Self.weightFormatChoice(for: root, quantization: rawGeometry.descriptor.quantization,
                                                 geometry: rawGeometry, mtpDrafts: mtpDrafts, pack: pack)
        let prelimHal = try QWHal(geometry: rawGeometry, weightBasis: choice.format.basis)
        let geometry = explicitResE != nil || rawGeometry.numExperts == 0 ? rawGeometry
            : Self.planResidentGeometry(
                raw: rawGeometry, cfg: cfg, hal: prelimHal, format: choice.format, tuning: tuning,
                osReserveGiB: osReserveGiB, prefillChunkSize: prefillChunkSize, maxTokens: maxTokens,
                ropeRows: ropeRows, mtpDrafts: mtpDrafts, mtpWeightDrafts: mtpWeightDrafts, kvMode: kvMode)
        return StartupContainers(
            tuning: tuning, osReserveGiB: osReserveGiB, geometry: geometry, hal: prelimHal,
            pack: pack, stContainer: stContainer, weightFormat: choice.format,
            weightFormatDeclineReason: choice.declineReason,
            mtpDrafts: mtpDrafts, mtpWeightDrafts: mtpWeightDrafts)
    }

    static func resolveCorePipelines(
        hal: QWHal, weightFormat: QWWeightFormat, mhcConfig: QWModelDescriptor.MHCConfig?
    ) throws -> CorePipelines {
        let psoBf16F16 = try hal.pipeline("qwg_bf16_to_f16")
        let psoGEMVDxH16T4 = try hal.pipeline(Self.decodeKernelName(format: weightFormat))
        let psoGEMVDxH16DualT4 = try hal.pipeline("qwg_gemv_q4g_dx_h16_dual_t4")
        let psoGEMVDxH16GdnInProjT4 = try hal.pipeline(Self.gdnInProjDecodeKernelName(format: weightFormat))
        let psoGEMVSiluPairH16 = try hal.pipeline(Self.siluPairKernelName(format: weightFormat))
        let psoFwht32 = try hal.pipeline("qwg_fwht32_f16")
        let psoRMSNorm = try hal.pipeline("qwg_rmsnorm")
        let psoRMSNormResidual = try hal.pipeline("qwg_rmsnorm_residual")
        let psoRMSNormResidualWide = try hal.pipeline("qwg_rmsnorm_residual_wide")
        let psoHeadF16 = try hal.pipeline("qwg_gemv_f16_v4")
        let psoGemmBatchedH16 = try hal.pipeline(Self.gemmKernelName(format: weightFormat, bm128: false))
        let psoGemmBatchedH16GdnInProj = try hal.pipeline(Self.gdnInProjPrefillKernelName(format: weightFormat))
        let psoSiluMulF16 = try hal.pipeline("qwg_silu_mul_f16")
        let psoKVPagedWrite = try hal.pipeline("qwg_kv_paged_write")
        let psoKVPagedWriteGeneral = try hal.pipeline("qwg_kv_paged_write_general")
        let psoZeroPadAttn = try hal.pipeline("qwg_zero_pad_attn_f16")
        let psoMHCEnter: (any MTLComputePipelineState)?
        let psoMHCRouterNorm: (any MTLComputePipelineState)?
        let psoMHCRouterNormRows: (any MTLComputePipelineState)?
        let psoMHCMerge: (any MTLComputePipelineState)?
        let psoMHCExitHead: (any MTLComputePipelineState)?
        if let mhcCfg = mhcConfig {
            psoMHCEnter = try hal.pipeline("qwg_mhc_enter_f16")
            psoMHCRouterNorm = try hal.pipeline("qwg_mhc_router_norm_f16")
            psoMHCRouterNormRows = ProcessInfo.processInfo.environment["QW_MHC_ROWS_KERNEL"] == "0"
                ? nil : try hal.pipeline("qwg_mhc_router_norm_rows_f16")
            psoMHCMerge = try hal.pipeline("qwg_mhc_merge_f16")
            psoMHCExitHead = mhcCfg.hasExitHead ? try hal.pipeline("qwg_mhc_exit_head_f16") : nil
        } else {
            psoMHCEnter = nil; psoMHCRouterNorm = nil; psoMHCRouterNormRows = nil
            psoMHCMerge = nil; psoMHCExitHead = nil
        }
        return CorePipelines(
            psoBf16F16: psoBf16F16, psoGEMVDxH16T4: psoGEMVDxH16T4,
            psoGEMVDxH16DualT4: psoGEMVDxH16DualT4,
            psoGEMVDxH16GdnInProjT4: psoGEMVDxH16GdnInProjT4,
            psoGEMVSiluPairH16: psoGEMVSiluPairH16, psoFwht32: psoFwht32,
            psoRMSNorm: psoRMSNorm, psoRMSNormResidual: psoRMSNormResidual,
            psoRMSNormResidualWide: psoRMSNormResidualWide, psoHeadF16: psoHeadF16,
            psoGemmBatchedH16: psoGemmBatchedH16,
            psoGemmBatchedH16GdnInProj: psoGemmBatchedH16GdnInProj,
            psoSiluMulF16: psoSiluMulF16, psoKVPagedWrite: psoKVPagedWrite,
            psoKVPagedWriteGeneral: psoKVPagedWriteGeneral, psoZeroPadAttn: psoZeroPadAttn,
            psoMHCEnter: psoMHCEnter, psoMHCRouterNorm: psoMHCRouterNorm,
            psoMHCRouterNormRows: psoMHCRouterNormRows, psoMHCMerge: psoMHCMerge,
            psoMHCExitHead: psoMHCExitHead)
    }

    static func planArenaAndBudget(
        geometry: QWModelGeometry, hal: QWHal, weightFormat: QWWeightFormat,
        weightFormatDeclineReason: String?, tuning: QWEngineTuning, osReserveGiB: Double?,
        prefillChunkSize: Int, maxTokens: Int, ropeRows: Int, kvMode: QWKVMode,
        attnLayerCount: Int, gdnLayerCount: Int, cfg: StartupConfig,
        mtpDrafts: Int, mtpWeightDrafts: Int
    ) throws -> ArenaBudgetPlan {
        let useGroupedNAX = hal.caps.isMetal4TensorCapable
        let bytesPerToken = Self.prefillBytesPerToken(geometry: geometry, useGroupedNAX: useGroupedNAX)
        let pscBytes = prefillChunkSize * bytesPerToken
        let decodeActBytes = Self.decodeAndMTPScratchBytes(
            geometry: geometry, prefillChunkRows: prefillChunkSize,
            mtpDrafts: mtpDrafts, maxTokens: maxTokens, useGroupedNAX: useGroupedNAX)
        let actBytes = pscBytes + decodeActBytes
        let recB = QWSlotGeometry.recElems(geometry: geometry) * 4
        let ringB = QWSlotGeometry.ringElems(geometry: geometry) * 2
        let slotBytes = gdnLayerCount * (recB + ringB)
        let trunkHidBytes = (geometry.mtpSharedLastLayer && geometry.mhcConfig != nil)
            ? geometry.mhcConfig!.streams * geometry.hidden * 4 : geometry.hidden * 2
        let gdnBytes = 2 * slotBytes + Self.partialsScratchBytes(geometry: geometry, mtpDrafts: mtpDrafts, maxTokens: maxTokens)
            + (mtpDrafts > 0 ? QWGDNReplayCheckpoint.allocationBytes(geometry: geometry, layers: gdnLayerCount, rows: mtpDrafts + 1) : 0)
            + (mtpDrafts > 0 ? 2 * trunkHidBytes : 0) + (1 << 20)
        let activeAttnLayers = attnLayerCount + (mtpDrafts > 0 ? 1 : 0)
        let mlaPerTokBytes: Int
        if geometry.usesNoPEMLA {
            mlaPerTokBytes = activeAttnLayers * ((geometry.indexerHeadDim * 2) / 4)
        } else if geometry.usesDecoupledRopeMLA {
            let ropeDim = geometry.hybridMLAConfig?.qkRopeHeadDim ?? 64
            mlaPerTokBytes = activeAttnLayers * ((geometry.indexerHeadDim + ropeDim) * 2)
        } else {
            mlaPerTokBytes = 0
        }
        let hcaPerTokBytes = Self.hcaPerTokenBytes(geometry: geometry)
        let mlaFixedBytes = Self.mlaFixedScratchBytes(
            geometry: geometry, prefillChunkRows: prefillChunkSize, activeAttnLayers: activeAttnLayers)
        let extraFixedBytes = mlaFixedBytes
            + Self.hcaFixedScratchBytes(geometry: geometry, prefillChunkRows: prefillChunkSize)
            + (mtpDrafts > 0 && geometry.mtpSharedLastLayer ? (112 << 20) : 0)
        let perTokBytes = 2 * activeAttnLayers * (QWKVCache.pageBytes(mode: kvMode, geometry: geometry) / QWKVCache.pageTokens)
            + mlaPerTokBytes + hcaPerTokBytes
        let maxMTP = cfg.mtpBackend == .dflash2 ? 15 : 5
        let prelimPlan = try QWWeightAllocationPlan(geometry: geometry, ropeRows: max(ropeRows, maxTokens),
                                                    mtpDrafts: mtpWeightDrafts, headQ4g: cfg.headIsQ4g, maxMTPDrafts: maxMTP,
                                                    format: weightFormat)
        let arenaCap = max(1 << 30, hal.caps.maxWorkingSetBytes - (64 << 20))
        let isMoE = geometry.numExperts > 0
        let isColdStreamingMoE = isMoE && geometry.effectiveResidentExperts < geometry.numExperts
        let effectiveReserveGiB = isMoE
            ? Self.effectiveOSReserveGiB(policy: tuning.memoryPolicy, explicitReserveGiB: osReserveGiB) : osReserveGiB
        let availBytes = isMoE
            ? Self.residencyBudgetBytes(policy: tuning.memoryPolicy, osReserveGiB: osReserveGiB, arenaCap: arenaCap)
            : min(QWCapacityBudget.availableBytes(osReserveGiB: osReserveGiB), arenaCap)
        let requestedMaxTokens = (isColdStreamingMoE && maxTokens == geometry.maxPositionEmbeddings)
            ? max(16_384, min(maxTokens, 32_768)) : maxTokens
        let prelimCap = try QWCapacityPlan.plan(
            nativeWindow: geometry.maxPositionEmbeddings, requested: requestedMaxTokens,
            perTokenBytes: perTokBytes, fixedBytes: prelimPlan.bytes + actBytes + gdnBytes + extraFixedBytes,
            availableBytes: availBytes, yarnAllowed: true)
        let effectiveMaxTokens = (!prelimCap.fits && requestedMaxTokens == geometry.maxPositionEmbeddings && prelimCap.maxFittingContext >= QWCapacityPlan.kvStepTokens)
            ? (prelimCap.maxFittingContext / QWCapacityPlan.kvStepTokens) * QWCapacityPlan.kvStepTokens : requestedMaxTokens
        let effectiveRopeRows = max(ropeRows, effectiveMaxTokens)
        let weightPlan = try QWWeightAllocationPlan(geometry: geometry, ropeRows: effectiveRopeRows, mtpDrafts: mtpWeightDrafts,
                                                    headQ4g: cfg.headIsQ4g, maxMTPDrafts: maxMTP, format: weightFormat)
        let weightsBytes = weightPlan.bytes
        let trunkKVBytes = Self.kvBudgetBytes(maxTokens: effectiveMaxTokens, mtpDrafts: 0, kvMode: kvMode, geometry: geometry)
        let mtpKVBytes = Self.kvBudgetBytes(maxTokens: effectiveMaxTokens, mtpDrafts: mtpDrafts, kvMode: kvMode, geometry: geometry) - trunkKVBytes
        let kvBytes = trunkKVBytes + mtpKVBytes
        let fixedBytes = weightsBytes + actBytes + gdnBytes + extraFixedBytes
        let capPlan = try QWCapacityPlan.plan(
            nativeWindow: geometry.maxPositionEmbeddings, requested: effectiveMaxTokens,
            perTokenBytes: perTokBytes, fixedBytes: fixedBytes, availableBytes: availBytes, yarnAllowed: true)
        let formatSuffix = " weights_format=\(weightFormat.name) weights_bpw="
            + String(format: "%.2f", weightFormat.effectiveBitsPerWeight(geometry: geometry))
            + (weightFormatDeclineReason.map { " weights_declined=\($0)" } ?? "")
        let capLog = try QWEngineBootstrap.admit(
            weightsBytes: weightsBytes, fixedScratchBytes: actBytes + gdnBytes + extraFixedBytes,
            kvMode: String(describing: kvMode), perTokenBytes: perTokBytes,
            maxTokens: effectiveMaxTokens, totalPlannedBytes: fixedBytes + kvBytes,
            availableBytes: availBytes, osReserveGiB: effectiveReserveGiB, logSuffix: formatSuffix)
        let arena = try QWArena(hal: hal, weights: weightsBytes, activations: actBytes,
                                kv: kvBytes, gdnStates: gdnBytes)
        return ArenaBudgetPlan(
            effectiveMaxTokens: effectiveMaxTokens, effectiveRopeRows: effectiveRopeRows,
            weightPlan: weightPlan, trunkKVBytes: trunkKVBytes, mtpKVBytes: mtpKVBytes,
            actBytes: actBytes, gdnBytes: gdnBytes, fixedBytes: fixedBytes, mtpPartialsOff: nil,
            pCarveBytes: Self.partialsCarveBytes(geometry: geometry, maxTokens: effectiveMaxTokens),
            useGroupedNAX: useGroupedNAX, capacityPlan: capPlan, capacityLogMessage: capLog,
            arena: arena, availBytes: availBytes)
    }

    static func loadEmbeddingsAndHead(
        loader L: inout WeightLoader, pack: QWQuant.Pack?, stContainer: QWSafeTensorsContainer?,
        packPath: String, hal: QWHal, geometry: QWModelGeometry, cfg: StartupConfig,
        mtpDrafts: Int, budget: ArenaBudgetPlan
    ) throws -> EmbeddingAndHeadSetup {
        let embedName = L.resolveName(Self.P + "embed_tokens.weight")
        let embedBytes = geometry.vocab * geometry.hidden * 2
        let devEmbedEnv = cfg.environment["QW_MTP_DEVICE_EMBEDDING"] ?? cfg.environment["QW_MTP_DEVICE_EMBED"]
        let fixedBytes = budget.fixedBytes
        let allowDeviceEmbed = devEmbedEnv != "0" && mtpDrafts > 0 && (
            devEmbedEnv == "1"
            || (mtpDrafts > 1
                && geometry.effectiveResidentExperts == geometry.numExperts
                && fixedBytes + budget.trunkKVBytes + budget.mtpKVBytes + embedBytes <= budget.availBytes)
        )
        let embedRawOff: Int
        let embedReader: QWSafeTensorsReader?
        let embedScalesReader: QWSafeTensorsReader?
        let embedBiasesReader: QWSafeTensorsReader?
        let mtpDeviceEmbedding: QWMTPEmbeddingTable?
        if let pk = pack {
            guard let rawOff = try pk.entry(embedName).rawOff else {
                throw QWEngineDriverError.weightMissing(embedName, "not a bf16 raw plane")
            }
            embedRawOff = rawOff; embedReader = nil; embedScalesReader = nil; embedBiasesReader = nil
            mtpDeviceEmbedding = allowDeviceEmbed
                ? try QWMTPEmbeddingTable(hal: hal, path: pk.url.path, offset: rawOff,
                                         vocabulary: geometry.vocab, hidden: geometry.hidden) : nil
        } else if let st = stContainer {
            embedRawOff = 0
            let eReader = try st.reader(of: embedName)
            embedReader = eReader
            if eReader.tensorInfo.dtype == .u32, embedName.hasSuffix(".weight") {
                let stem = String(embedName.dropLast(".weight".count))
                embedScalesReader = try st.reader(of: stem + ".scales")
                embedBiasesReader = try st.reader(of: stem + ".biases")
            } else {
                embedScalesReader = nil; embedBiasesReader = nil
            }
            if allowDeviceEmbed, eReader.tensorInfo.dtype != .u32,
               let sURL = st.shardURL(of: embedName), let rng = st.byteRange(of: embedName) {
                mtpDeviceEmbedding = try QWMTPEmbeddingTable(hal: hal, path: sURL.path, offset: rng.lowerBound,
                                                             vocabulary: geometry.vocab, hidden: geometry.hidden)
            } else {
                mtpDeviceEmbedding = nil
            }
        } else {
            throw QWEngineDriverError.packMissing(packPath)
        }
        guard let embedStaging = hal.device.makeBuffer(length: geometry.hidden * 2, options: .storageModeShared) else {
            throw QWEngineDriverError.state("failed to allocate embed staging buffer")
        }
        let mtpDraftHead: QWMTPTDraftHead?
        if cfg.draftHeadEnabled, let sidecarPath = cfg.environment["QW_MTP_DRAFT_HEAD_SIDECAR"] {
            let sidecar = try QWQuant.Pack(url: URL(fileURLWithPath: sidecarPath),
                                           descriptor: geometry.descriptor, skipIdentityCheck: true)
            mtpDraftHead = try QWMTPTDraftHead(geometry: geometry, hal: hal, sidecar: sidecar)
        } else if cfg.draftHeadEnabled, cfg.headIsQ4g, let pk = pack {
            mtpDraftHead = try QWMTPTDraftHead(geometry: geometry, hal: hal, sidecar: pk)
        } else {
            if cfg.draftHeadEnabled && pack != nil {
                throw QWEngineDriverError.state("QW_MTP_DRAFT_HEAD=1 requires QW_MTP_DRAFT_HEAD_SIDECAR")
            }
            mtpDraftHead = nil
        }
        let headStorage = try Self.loadHead(loader: &L, headIsQ4g: cfg.headIsQ4g, geometry: geometry)
        if case .q4g(let hw) = headStorage, !hw.view.isIncumbentContainer {
            for name in [QWGemmAQ.kernelName(bits: hw.view.bits, group: hw.view.groupSize, isSHVQ: hw.view.isSHVQ),
                         QWGemmAQ.decodeKernelName(bits: hw.view.bits, group: hw.view.groupSize, isSHVQ: hw.view.isSHVQ),
                         QWGemmAQ.headArgmaxKernelName(bits: hw.view.bits, group: hw.view.groupSize, isSHVQ: hw.view.isSHVQ)].compactMap({ $0 }) {
                _ = try hal.pipeline(name)
            }
        }
        return EmbeddingAndHeadSetup(
            embedPlane: embedName, embedRawOff: embedRawOff, embedReader: embedReader,
            embedScalesReader: embedScalesReader, embedBiasesReader: embedBiasesReader,
            mtpDeviceEmbedding: mtpDeviceEmbedding, embedStaging: embedStaging,
            mtpDraftHead: mtpDraftHead, headStorage: headStorage)
    }

    static func buildGDNSlots(
        gsub: QWSubAllocator, geometry: QWModelGeometry, gdnLayerCount: Int,
        mtpDrafts: Int, dflashDrafter: QWDFlashDrafter?
    ) throws -> [GdnSlot] {
        var slots: [GdnSlot] = []
        let H = geometry.hidden
        for slotIdx in 0..<2 {
            let recSlab = try gsub.allocate(QWTensorShape([max(1, gdnLayerCount * QWSlotGeometry.recElems(geometry: geometry))], .f32))
            let ringSlab = try gsub.allocate(QWTensorShape([max(1, gdnLayerCount * QWSlotGeometry.ringElems(geometry: geometry))], .f16))
            let recViews = Self.slotViews(recSlab, elem: QWSlotGeometry.recElems(geometry: geometry), dt: .f32, layers: gdnLayerCount, geometry: geometry)
            let ringViews = Self.slotViews(ringSlab, elem: QWSlotGeometry.ringElems(geometry: geometry), dt: .f16, layers: gdnLayerCount, geometry: geometry)
            func disjoint(_ views: [QWTensorValue]) -> Bool {
                guard views.count > 1 else { return true }
                var spans = views.map { ($0.region.base + $0.offset, $0.bytes) }
                spans.sort { $0.0 < $1.0 }
                for i in 1..<spans.count where spans[i].0 < spans[i - 1].0 + spans[i - 1].1 { return false }
                return true
            }
            precondition(recViews.count == gdnLayerCount && disjoint(recViews),
                         "QWEngineDriver: GDN rec state views ALIAS (per-layer state law)")
            precondition(ringViews.count == gdnLayerCount && disjoint(ringViews),
                         "QWEngineDriver: GDN ring views ALIAS (per-layer state law)")
            let trunkHidShape = (geometry.mtpSharedLastLayer && geometry.mhcConfig != nil)
                ? QWTensorShape([geometry.mhcConfig!.streams * H], .f32) : QWTensorShape([H], .f16)
            slots.append(GdnSlot(recSlab: recSlab, ringSlab: ringSlab,
                                 rec: recViews, ring: ringViews,
                                 trunkHidden: mtpDrafts > 0 ? try gsub.allocate(trunkHidShape) : nil,
                                 dflashSlot: dflashDrafter?.sessionSlots[slotIdx]))
        }
        return slots
    }

    static func buildMTPVerificationResources(
        geometry: QWModelGeometry, hal: QWHal, asub: QWSubAllocator, gsub: QWSubAllocator,
        attnDecode: QWAttnDecode?, attnLayers: [Int], gdnLayerCount: Int,
        mlaOperator: QWMLAAttnOperator?, layersW: [LayerW],
        mtpWeights: MTPWeights?, echoWeights: EchoDraftWeights?, dflashDrafter: QWDFlashDrafter?,
        mtpDrafts: Int, effectiveMaxTokens: Int, maxT: Int, mtpPartialsOff: Int, pCarveBytes: Int,
        headIsQ4g: Bool, verifyHeadQ4Enabled: Bool, useGroupedNAX: Bool,
        scratch: Scratch, ropeApply: QWRopeApply, effectiveRopeRows: Int, kvMode: QWKVMode
    ) throws -> MTPVerificationResources {
        guard mtpDrafts > 0 && (mtpWeights != nil || echoWeights != nil || dflashDrafter != nil) else {
            return MTPVerificationResources(
                gdnReplayCheckpoint: nil, verifyWindow: nil, mtpBatchedMLP: nil,
                mtpBatchedTrunk: nil, mtpSharedAttentionPartials: nil, mtpScratch: nil,
                mtpPromptEmbeddings: nil, mtpDraftEncoder: nil, echoDrafter: nil)
        }
        let gdnReplayCheckpoint = try QWGDNReplayCheckpoint(
            geometry: geometry, allocator: gsub, layers: gdnLayerCount, rows: mtpDrafts + 1)
        _ = try hal.pipeline("qwg_gdn_replay_state")
        _ = try hal.pipeline("qwg_verify_copy")
        let verifyWindow = try QWVerifyWindow(
            geometry: geometry, allocator: asub, rows: mtpDrafts + 1, layerMajor: true,
            gpuTargets: true, q4Head: verifyHeadQ4Enabled || headIsQ4g, useGroupedNAX: useGroupedNAX)
        let mtpBatchedMLP = try QWMTPBatchedMLP(geometry: geometry, hal: hal, allocator: asub, rows: mtpDrafts + 1)
        let mtpBatchedTrunk = try QWMTPBatchedTrunk(geometry: geometry, hal: hal, allocator: asub, rows: mtpDrafts + 1)
        let mtpSharedAttentionPartials: QWMRegion?
        if attnDecode != nil {
            let maxDecodeSplits = effectiveMaxTokens > 262_144 ? QWAttnDecode.maxLongContextSplits : QWAttnDecode.maxSplits
            let bytes = QWAttnDecode.decodePartialsBytesBatched(T: min(8, mtpDrafts + 1), geometry: geometry, maxSplits: maxDecodeSplits)
            let offset = try asub.allocate(bytes: bytes)
            mtpSharedAttentionPartials = QWMRegion(buffer: asub.region.buffer, base: asub.region.base + offset, length: bytes)
        } else {
            mtpSharedAttentionPartials = nil
        }
        _ = try hal.pipeline("qwg_mtp_argmax_target_checked")
        let partials = QWMRegion(buffer: gsub.region.buffer, base: gsub.region.base + mtpPartialsOff, length: pCarveBytes)
        let mtpScratch = try MTPScratch(geometry: geometry, allocator: asub, partials: partials)
        let mtpPromptEmbeddings = try asub.allocate(QWTensorShape([maxT * geometry.hidden], .f16))
        let mtpDraftEncoder = try mtpWeights.map {
            try QWMTPDraftEncoder(geometry: geometry, hal: hal, weights: $0, scratch: mtpScratch,
                                  rope: ropeApply, ropeRows: effectiveRopeRows, kvMode: kvMode,
                                  mlaOperator: mlaOperator, mlaLayerIndex: attnLayers.count)
        }
        let echoDrafter: QWEchoDraftEncoder?
        if let ew = echoWeights, let mla = mlaOperator, let lastLayer = layersW.last {
            echoDrafter = try QWEchoDraftEncoder(
                geometry: geometry, hal: hal, weights: ew, trunkLayer: lastLayer,
                mlaOperator: mla, mlaLayerIndex: attnLayers.count, maxSeedRows: maxT)
        } else {
            echoDrafter = nil
        }
        for m in ["m2", "m3", "m5", "m6", "m7", "m8"] { _ = try hal.pipeline("qwg_head_f16_\(m)") }
        for m in 2...8 {
            _ = try hal.pipeline("qwg_head_q4g_m\(m)")
            if m >= 4 { _ = try hal.pipeline("qwg_head_q4g_rp_m\(m)") }
        }
        return MTPVerificationResources(
            gdnReplayCheckpoint: gdnReplayCheckpoint, verifyWindow: verifyWindow,
            mtpBatchedMLP: mtpBatchedMLP, mtpBatchedTrunk: mtpBatchedTrunk,
            mtpSharedAttentionPartials: mtpSharedAttentionPartials, mtpScratch: mtpScratch,
            mtpPromptEmbeddings: mtpPromptEmbeddings, mtpDraftEncoder: mtpDraftEncoder,
            echoDrafter: echoDrafter)
    }
}
