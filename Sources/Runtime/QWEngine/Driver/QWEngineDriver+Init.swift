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
        let prefillChunkSize: Int
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
        let psoRMSNormResidualMoE: any MTLComputePipelineState
        let psoRMSNormResidualMoEWide: any MTLComputePipelineState
        let psoHeadF16: any MTLComputePipelineState
        let psoGemmBatchedH16: any MTLComputePipelineState
        let psoGemmBatchedH16SwiGLU: any MTLComputePipelineState
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
        Self.prefixPolicyKey(backend: mtpBackend, drafts: mtpDrafts, mode: mode, cacheSalt: activeCacheSalt)
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
        let maxwellMTPName = "mtp_layers.0.enorm.weight"
        let maxwellMTPAltName = "model.mtp_layers.0.enorm.weight"
        let hasMTPWeights: Bool
        if let st = stContainer {
            hasMTPWeights = st.tensor("mtp.pre_fc_norm_embedding.weight") != nil
                || st.tensor(nextNMTPName) != nil
                || st.tensor(nextNMTPAltName) != nil
                || st.tensor(maxwellMTPName) != nil
                || st.tensor(maxwellMTPAltName) != nil
        } else if let pk = pack {
            hasMTPWeights = (try? pk.entry("mtp.pre_fc_norm_embedding.weight")) != nil
                || (try? pk.entry(nextNMTPName)) != nil
                || (try? pk.entry(nextNMTPAltName)) != nil
                || (try? pk.entry(maxwellMTPName)) != nil
                || (try? pk.entry(maxwellMTPAltName)) != nil
        } else {
            hasMTPWeights = false
        }
        let resolvedDraftStrategy = (try? QWDraftStrategy.resolve(cfg.mtpDraftStrategy)) ?? .hybrid
        let allowWeightFreeMTP = !hasMTPWeights && cfg.mtpBackend != .dflash2 && resolvedDraftStrategy == .ngram
        let mtpDrafts = (cfg.mtpBackend == .dflash2 || hasMTPWeights)
            ? requestedMTPDrafts
            : (allowWeightFreeMTP ? min(requestedMTPDrafts, 7) : 0)
        let mtpWeightDrafts = (cfg.mtpBackend == .dflash2) ? 0 : (hasMTPWeights ? mtpDrafts : 0)
        try Self.validateGeometry(rawGeometry, kvMode: kvMode)
        let choice = try Self.weightFormatChoice(for: root, quantization: rawGeometry.descriptor.quantization,
                                                 geometry: rawGeometry, mtpDrafts: mtpWeightDrafts, pack: pack)
        let prelimHal = try QWHal(geometry: rawGeometry, weightBasis: choice.format.basis)
        let clampedChunk = max(32, min(QWPrefillLimits.maxRows, (prefillChunkSize / 32) * 32))
        let (geometry, plannedChunk) = (explicitResE != nil || rawGeometry.numExperts == 0)
            ? (rawGeometry, clampedChunk)
            : Self.planResidencyAndChunk(
                raw: rawGeometry, cfg: cfg, hal: prelimHal, format: choice.format, tuning: tuning,
                osReserveGiB: osReserveGiB, prefillChunkSize: prefillChunkSize, maxTokens: maxTokens,
                ropeRows: ropeRows, mtpDrafts: mtpDrafts, mtpWeightDrafts: mtpWeightDrafts, kvMode: kvMode)
        return StartupContainers(
            tuning: tuning, osReserveGiB: osReserveGiB, geometry: geometry, hal: prelimHal,
            pack: pack, stContainer: stContainer, weightFormat: choice.format,
            weightFormatDeclineReason: choice.declineReason,
            mtpDrafts: mtpDrafts, mtpWeightDrafts: mtpWeightDrafts,
            prefillChunkSize: plannedChunk)
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
        let psoRMSNormResidualMoE = try hal.pipeline("qwg_rmsnorm_residual_moe")
        let psoRMSNormResidualMoEWide = try hal.pipeline("qwg_rmsnorm_residual_moe_wide")
        let psoHeadF16 = try hal.pipeline("qwg_gemv_f16_v4")
        let psoGemmBatchedH16 = try hal.pipeline(Self.gemmKernelName(format: weightFormat, bm128: false))
        let psoGemmBatchedH16SwiGLU = try hal.pipeline(Self.swigluGemmKernelName(format: weightFormat))
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
            psoRMSNormResidualWide: psoRMSNormResidualWide,
            psoRMSNormResidualMoE: psoRMSNormResidualMoE,
            psoRMSNormResidualMoEWide: psoRMSNormResidualMoEWide,
            psoHeadF16: psoHeadF16,
            psoGemmBatchedH16: psoGemmBatchedH16,
            psoGemmBatchedH16SwiGLU: psoGemmBatchedH16SwiGLU,
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
        let bytesPerToken = Self.prefillBytesPerToken(
            geometry: geometry, maxPrefillChunk: prefillChunkSize, useGroupedNAX: useGroupedNAX)
        let pscBytes = prefillChunkSize * bytesPerToken
        let decodeActBytes = Self.decodeAndMTPScratchBytes(
            geometry: geometry, prefillChunkRows: prefillChunkSize,
            mtpDrafts: mtpDrafts, maxTokens: maxTokens, useGroupedNAX: useGroupedNAX)
        let actBytes = pscBytes + decodeActBytes
        let slotCount = Self.gdnSlotCount()
        let recB = QWSlotGeometry.recElems(geometry: geometry) * 4
        let ringB = QWSlotGeometry.ringElems(geometry: geometry) * 2
        let slotBytes = gdnLayerCount * (recB + ringB)
        let trunkHidBytes = geometry.usesMultiStreamMTPHidden
            ? geometry.mhcConfig!.streams * geometry.hidden * 4 : geometry.hidden * 2
        let needsMTPCache = (mtpWeightDrafts > 0 || cfg.mtpBackend == .dflash2) && mtpDrafts > 0
        let gdnBytes = slotCount * slotBytes + Self.partialsScratchBytes(geometry: geometry, mtpDrafts: mtpWeightDrafts > 0 ? mtpDrafts : 0, maxTokens: maxTokens)
            + (mtpDrafts > 0 ? QWGDNReplayCheckpoint.allocationBytes(geometry: geometry, layers: gdnLayerCount, rows: mtpDrafts + 1) : 0)
            + (mtpDrafts > 0 ? slotCount * trunkHidBytes : 0) + (1 << 20)
        let activeAttnLayers = attnLayerCount + (needsMTPCache ? 1 : 0)
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
        let mbsaPerTokBytes = Self.mbsaPerTokenBytes(geometry: geometry, attnLayerCount: attnLayerCount)
        let mlaFixedBytes = Self.mlaFixedScratchBytes(
            geometry: geometry, prefillChunkRows: prefillChunkSize, activeAttnLayers: activeAttnLayers)
        let extraFixedBytes = mlaFixedBytes
            + Self.hcaFixedScratchBytes(geometry: geometry, prefillChunkRows: prefillChunkSize)
            + Self.mbsaFixedScratchBytes(geometry: geometry, prefillChunkRows: prefillChunkSize, attnLayerCount: attnLayerCount)
            + (mtpWeightDrafts > 0 && (geometry.mtpSharedLastLayer || geometry.hasDedicatedMHCMTP) ? (112 << 20) : 0)
        let perTokBytes = 2 * activeAttnLayers * (QWKVCache.pageBytes(mode: kvMode, geometry: geometry) / QWKVCache.pageTokens)
            + mlaPerTokBytes + hcaPerTokBytes + mbsaPerTokBytes
        let maxMTP = cfg.mtpBackend == .dflash2 ? 15 : 7
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
        let mtpKVBytes = needsMTPCache
            ? (Self.kvBudgetBytes(maxTokens: effectiveMaxTokens, mtpDrafts: mtpDrafts, kvMode: kvMode, geometry: geometry) - trunkKVBytes)
            : 0
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
        mtpDrafts: Int, hasMTPDrafter: Bool, budget: ArenaBudgetPlan
    ) throws -> EmbeddingAndHeadSetup {
        let embedName = L.resolveName(Self.P + "embed_tokens.weight")
        let embedBytes = geometry.vocab * geometry.hidden * 2
        let devEmbedEnv = cfg.environment["QW_MTP_DEVICE_EMBEDDING"] ?? cfg.environment["QW_MTP_DEVICE_EMBED"]
        let fixedBytes = budget.fixedBytes
        let allowDeviceEmbed = devEmbedEnv != "0" && hasMTPDrafter && mtpDrafts > 0 && (
            devEmbedEnv == "1"
            || (geometry.effectiveResidentExperts == geometry.numExperts
                && fixedBytes + budget.trunkKVBytes + budget.mtpKVBytes + embedBytes + (1 << 30) <= budget.availBytes)
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
        let slotCount = Self.gdnSlotCount()
        for slotIdx in 0..<slotCount {
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
            let trunkHidShape = geometry.usesMultiStreamMTPHidden
                ? QWTensorShape([geometry.mhcConfig!.streams * H], .f32) : QWTensorShape([H], .f16)
            let dflashSlot = (dflashDrafter.map { slotIdx < $0.sessionSlots.count ? $0.sessionSlots[slotIdx] : nil }) ?? nil
            slots.append(GdnSlot(recSlab: recSlab, ringSlab: ringSlab,
                                 rec: recViews, ring: ringViews,
                                 trunkHidden: mtpDrafts > 0 ? try gsub.allocate(trunkHidShape) : nil,
                                 dflashSlot: dflashSlot))
        }
        return slots
    }

    private static func prewarmBatchedDecodePipelines(
        hal: QWHal, geometry: QWModelGeometry, layersW: [LayerW], headAQ4: Bool, hasMLA: Bool
    ) throws {
        // Always prewarm B=2..4 batched head pipelines before hal.freezePipelines().
        for m in 2...4 {
            _ = try hal.pipeline("qwg_head_f16_m\(m)")
            _ = try hal.pipeline("qwg_head_q4g_m\(m)", isAQ4: headAQ4)
        }
        _ = try hal.pipeline("qwg_head_q4g_rp_m4", isAQ4: headAQ4)
        guard !hasMLA && geometry.hcaConfig == nil && geometry.hybridMLAConfig == nil
                && geometry.blockSparseGQAConfig == nil else {
            return
        }
        for m in 2...4 {
            _ = try hal.pipeline("qwm_mlp_gate_up_m\(m)")
            _ = try hal.pipeline("qwm_mlp_gate_up_fwht32_m\(m)")
            _ = try hal.pipeline("qwm_mlp_down_m\(m)")
            if m == 4 {
                _ = try hal.pipeline("qwm_mlp_gate_up_rp_m\(m)")
                _ = try hal.pipeline("qwm_mlp_down_rp_m\(m)")
            }
        }
        var mlpAQ4Prewarmed = false
        var seenGemms = Set<String>()
        var seenAqDecode = Set<String>()
        var seenAqSilu = Set<String>()
        func prewarmWeight(_ w: QWWeightView) throws {
            guard !w.isIncumbentContainer,
                  let name = QWGemmAQ.gemmKernelName(bits: w.bits, group: w.groupSize, bm128: false, isSHVQ: w.isSHVQ),
                  seenGemms.insert(name).inserted else { return }
            _ = try hal.pipeline(name)
        }
        func prewarmBatchedProj(_ w: QWWeightView) throws {
            if QWMTPBatchedMLP.usesQ4GMKernel(w) {
                if !w.isIncumbentContainer && !mlpAQ4Prewarmed {
                    for m in 2...4 {
                        _ = try hal.pipeline("qwm_mlp_gate_up_m\(m)", isAQ4: true)
                        _ = try hal.pipeline("qwm_mlp_gate_up_fwht32_m\(m)", isAQ4: true)
                        _ = try hal.pipeline("qwm_mlp_down_m\(m)", isAQ4: true)
                        if m == 4 {
                            _ = try hal.pipeline("qwm_mlp_gate_up_rp_m\(m)", isAQ4: true)
                            _ = try hal.pipeline("qwm_mlp_down_rp_m\(m)", isAQ4: true)
                        }
                    }
                    mlpAQ4Prewarmed = true
                }
            } else if let name = QWGemmAQ.decodeKernelName(bits: w.bits, group: w.groupSize, isSHVQ: w.isSHVQ),
                      seenAqDecode.insert(name).inserted {
                _ = try hal.pipeline(name)
            }
        }
        var gdnHQ4Prewarmed = false
        var gdnAQ4Prewarmed = false
        var seenGdnInProj = Set<String>()
        for lw in layersW {
            try prewarmWeight(lw.gu.view)
            try prewarmWeight(lw.down)
            if lw.moe == nil {
                let gu = lw.gu.view
                if QWMTPBatchedMLP.usesQ4GMKernel(gu) {
                    try prewarmBatchedProj(gu)
                } else if let name = QWGemmAQ.siluPairKernelName(bits: gu.bits, group: gu.groupSize, isSHVQ: gu.isSHVQ),
                          seenAqSilu.insert(name).inserted {
                    _ = try hal.pipeline(name)
                }
                try prewarmBatchedProj(lw.down)
            }
            if let a = lw.attn {
                try prewarmWeight(a.qkvProj)
                try prewarmWeight(a.o)
                try prewarmBatchedProj(a.qkvProj)
                try prewarmBatchedProj(a.o)
            }
            if let g = lw.gdn {
                try prewarmWeight(g.inProj)
                if geometry.gdnVectorDecay { try prewarmBatchedProj(g.inProj) }
                if let fgb = g.fgbProj {
                    try prewarmWeight(fgb)
                    try prewarmBatchedProj(fgb)
                }
                try prewarmWeight(g.out)
                try prewarmBatchedProj(g.out)
                if !geometry.gdnVectorDecay {
                    let v = g.inProj
                    if v.isIncumbentContainer || (v.bits == 4 && v.groupSize == 32 && !v.isSHVQ) {
                        let isAQ4 = !v.isIncumbentContainer
                        if (isAQ4 && !gdnAQ4Prewarmed) || (!isAQ4 && !gdnHQ4Prewarmed) {
                            for m in 2...4 {
                                _ = try hal.pipeline("qwm_gdn_projection_m\(m)", isAQ4: isAQ4)
                                if m == 4 { _ = try hal.pipeline("qwm_gdn_projection_rp_m\(m)", isAQ4: isAQ4) }
                            }
                            if isAQ4 { gdnAQ4Prewarmed = true } else { gdnHQ4Prewarmed = true }
                        }
                    } else if let name = QWGemmAQ.gdnInProjDecodeKernelName(bits: v.bits, group: v.groupSize, isSHVQ: v.isSHVQ),
                              seenGdnInProj.insert(name).inserted {
                        _ = try hal.pipeline(name)
                    }
                }
            }
        }
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
        let headAQ4 = (hal.weightBasis == .standard) || (layersW.first.map { !$0.gu.view.isIncumbentContainer } ?? false)
        try Self.prewarmBatchedDecodePipelines(hal: hal, geometry: geometry, layersW: layersW,
                                               headAQ4: headAQ4, hasMLA: mlaOperator != nil)
        guard mtpDrafts > 0 else {
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
        let needsNeuralScratch = mtpWeights != nil || echoWeights != nil
        let partials = QWMRegion(buffer: gsub.region.buffer, base: gsub.region.base + mtpPartialsOff, length: pCarveBytes)
        let mtpScratch = needsNeuralScratch
            ? try MTPScratch(geometry: geometry, allocator: asub, partials: partials)
            : nil
        let mtpPromptEmbeddings = (needsNeuralScratch || dflashDrafter != nil)
            ? try asub.allocate(QWTensorShape([maxT * geometry.hidden], .f16))
            : nil
        let mtpDraftEncoder: QWMTPDraftEncoder? = try mtpWeights.flatMap { w in
            try mtpScratch.map { scratch in
                try QWMTPDraftEncoder(geometry: geometry, hal: hal, weights: w, scratch: scratch,
                                      rope: ropeApply, ropeRows: effectiveRopeRows, kvMode: kvMode,
                                      mlaOperator: mlaOperator, mlaLayerIndex: attnLayers.count)
            }
        }
        let echoDrafter: QWEchoDraftEncoder?
        if let ew = echoWeights, let draftLayer = ew.dedicatedLayer ?? layersW.last,
           (mlaOperator != nil || ew.dedicatedLayer != nil) {
            echoDrafter = try QWEchoDraftEncoder(
                geometry: geometry, hal: hal, weights: ew, trunkLayer: draftLayer,
                mlaOperator: mlaOperator, mlaLayerIndex: attnLayers.count,
                maxSeedRows: maxT, kvMode: kvMode)
        } else {
            echoDrafter = nil
        }
        for m in ["m5", "m6", "m7", "m8"] { _ = try hal.pipeline("qwg_head_f16_\(m)") }
        for m in 5...8 {
            _ = try hal.pipeline("qwg_head_q4g_m\(m)", isAQ4: headAQ4)
            _ = try hal.pipeline("qwg_head_q4g_rp_m\(m)", isAQ4: headAQ4)
        }
        return MTPVerificationResources(
            gdnReplayCheckpoint: gdnReplayCheckpoint, verifyWindow: verifyWindow,
            mtpBatchedMLP: mtpBatchedMLP, mtpBatchedTrunk: mtpBatchedTrunk,
            mtpSharedAttentionPartials: mtpSharedAttentionPartials, mtpScratch: mtpScratch,
            mtpPromptEmbeddings: mtpPromptEmbeddings, mtpDraftEncoder: mtpDraftEncoder,
            echoDrafter: echoDrafter)
    }
}
