// DriverResidency.swift — How many experts per MoE layer stay resident.
// The planner sizes the resident bank to the memory budget the tuning's
// memory policy names; the rest of each layer's experts stream from SSD.
import Foundation
import Metal
import os
import QWCore
import QWHAL
import QWHead
import QWGDN
import QWKV
import QWMTP
import QWQuant

extension QWEngineDriver {
    private static let cachedPackFormats = OSAllocatedUnfairLock<[String: (format: QWWeightFormat, headQ4g: Bool, headFormat: QWWeightFormat?)]>(initialState: [:])

    /// Default OS RAM reserve (GiB) per memory policy when `osReserveGiB` is not explicitly set:
    /// 16 GiB under `.maxResident` and 24 GiB under `.safe` (matching `docs/benchmarks.md`).
    static func effectiveOSReserveGiB(policy: QWMemoryPolicy, explicitReserveGiB: Double?) -> Double {
        if let explicitReserveGiB { return explicitReserveGiB }
        return policy == .maxResident ? 16.0 : 24.0
    }

    /// The bytes the resident plan may use: physical RAM less the policy's OS reserve, and
    /// never more than currently free/reclaimable host RAM less safety headroom (`6 GiB` for
    /// `.safe`, `4 GiB` for `.maxResident`) so wired Metal arenas cannot starve macOS.
    static func residencyBudgetBytes(
        policy: QWMemoryPolicy,
        osReserveGiB: Double?,
        arenaCap: Int,
        liveReclaimableBytes: Int? = QWAdmission.reclaimableMemoryBytes()
    ) -> Int {
        let reserveGiB = effectiveOSReserveGiB(policy: policy, explicitReserveGiB: osReserveGiB)
        let physical = min(QWCapacityBudget.availableBytes(osReserveGiB: reserveGiB), arenaCap)
        guard let live = liveReclaimableBytes else { return physical }
        let headroom = policy == .safe ? QWEngineTuning.safeHeadroomBytes : (4 << 30)
        return min(physical, max(0, live - headroom))
    }

    static func prefillTransientBreakdown(
        geometry: QWModelGeometry,
        maxPrefillChunk: Int = 4096,
        useGroupedNAX: Bool = true
    ) -> (persistent: Int, attnTransient: Int, mlpTransient: Int) {
        QWScratchArenaBuilder.prefillTransientBreakdown(
            geometry: geometry, maxPrefillChunk: maxPrefillChunk, useGroupedNAX: useGroupedNAX)
    }

    static func prefillBytesPerToken(
        geometry: QWModelGeometry,
        maxPrefillChunk: Int = 4096,
        useGroupedNAX: Bool = true,
        aliasBranches: Bool
    ) -> Int {
        QWScratchArenaBuilder.prefillBytesPerToken(
            geometry: geometry, maxPrefillChunk: maxPrefillChunk,
            useGroupedNAX: useGroupedNAX, aliasBranches: aliasBranches)
    }

    static func mhcSplitBufferBytes(geometry: QWModelGeometry) -> Int {
        guard let cfg = geometry.mhcConfig,
              ProcessInfo.processInfo.environment["QW_MHC_SPLIT"] != "0" else { return 0 }
        let parts = (geometry.hidden + 63) / 64
        let outs = 2 * cfg.streams + cfg.streams * cfg.streams + 1 + cfg.streams * (cfg.streams + 1) / 2
        return QWMHCSplitRouter.maxRows * parts * outs * 4
    }

    static func externalMoEDecodeBufferBytes(
        geometry: QWModelGeometry,
        tuning: QWEngineTuning,
        prefillChunkSize: Int
    ) -> Int {
        guard geometry.numExperts > 0, geometry.effectiveResidentExperts < geometry.numExperts else { return 0 }
        var total = 0
        if tuning.coldEvents { total += 4096 }
        if tuning.expertCache {
            total += geometry.numVirtualLayers * 4 * geometry.numExperts
            total += max(4 * prefillChunkSize * geometry.expertsPerTok, 4096)
        }
        return total
    }

    static func sharedBatchMLPFixedBytes(geometry: QWModelGeometry, mtpDrafts: Int) -> Int {
        guard mtpDrafts == 0,
              !geometry.usesLatentOrCompressedAttention,
              geometry.hcaConfig == nil,
              geometry.hybridMLAConfig == nil,
              geometry.blockSparseGQAConfig == nil else {
            return 0
        }
        return QWMTPBatchedMLP.allocationBytes(geometry: geometry, rows: QWBatchHeadScratch.maxBatchRows) + 1024
    }

    private static func dflashScratchEstimate(config: QWDFlashConfig, sessionSlotCount: Int) -> Int {
        func aligned(_ bytes: Int) -> Int { ((bytes + 4095) / 4096) * 4096 }
        let maxCtx = config.maxCacheSize
        let maxB = max(8, config.blockSize), maxD = maxB - 1
        let H = config.hiddenSize, V = config.vocabSize, I = config.intermediateSize, R = config.selectorRank
        let qDim = config.numAttentionHeads * config.headDim
        let kvDim = config.numKeyValueHeads * config.headDim
        let hasConv = config.convKernelSize > 0
        let convProjDim = (2 * config.convKernelSize * H) / max(1, config.convGroupSize)
        var total = aligned(maxCtx * config.targetLayerIds.count * H * 2) + aligned(maxCtx * H * 2)
        total += aligned(maxCtx * 2 * kvDim * 2) + aligned(maxB * H * 4) * 2 + aligned(maxB * H * 2) * 3
        total += aligned(maxB * 4) * 2 + (hasConv ? aligned(maxB * convProjDim * 4) : 0)
        total += aligned(maxB * (qDim + 2 * kvDim) * 2) + aligned(maxB * qDim * 2) + aligned(maxB * I * 2)
        total += aligned(maxD * V * 4) + aligned(maxD * 256 * 16 * 4) * 2 + aligned(maxD * 16 * 4) * 2
        if !config.isDSpine { total += aligned(maxD * R * 4) }
        if config.isDSpine {
            let bc = config.bridgeCodeSize, br = config.bridgeRank
            total += aligned(maxB * 4) + aligned(maxD * bc * 4) + aligned(maxB * bc * 2)
            total += aligned(maxB * 3 * br * 4) + aligned(maxD * br * 2) + aligned(maxD * H * 4)
            total += aligned(maxD * 16 * br * 2) + aligned(maxD * 16 * H * 4)
            total += aligned(maxD * 16 * H * 2) * 2 + aligned(maxD * 16 * 16 * 4)
        }
        total += (2 * max(1, sessionSlotCount)) * aligned(config.numLayers * maxCtx * kvDim * 2)
        return max(32 * 1024 * 1024, total + 16 * 1024 * 1024)
    }

    static func dflashFootprintBytes(
        draftPackPath: String?,
        descriptor: QWModelDescriptor,
        mtpDrafts: Int,
        isDFlash: Bool,
        isDSpineBackend: Bool,
        selectorTopK: Int? = nil,
        physicalMemoryBytes: Int?
    ) -> (weights: Int, scratch: Int) {
        guard isDFlash, mtpDrafts > 0 else { return (0, 0) }
        let slots = gdnSlotCount(physicalMemoryBytes: physicalMemoryBytes)
        let rawPath = draftPackPath?.isEmpty == false ? draftPackPath! : QWDFlashConfig.defaultPackPath(for: descriptor)
        let url = URL(fileURLWithPath: rawPath)
        if !rawPath.isEmpty, FileManager.default.fileExists(atPath: url.path),
           let cfgURL = QWModelRoot(url: url).configURL,
           let cfgData = try? Data(contentsOf: cfgURL),
           let dDesc = try? QWModelDescriptor.decodeDrafter(configJSON: cfgData) {
            let dCfg = QWDFlashConfig.load(from: url, topKOverride: selectorTopK)
            let weights = (try? QWQuant.Pack(url: url, descriptor: dDesc))
                .map { QWDFlashDrafter.computeWeightCapacity(pack: $0, isDSpine: dCfg.isDSpine) }
                ?? (dCfg.isDSpine ? 1_920_000_000 : 1_480_000_000)
            return (weights, dflashScratchEstimate(config: dCfg, sessionSlotCount: slots))
        }
        let defaultWeights = isDSpineBackend ? 1_920_000_000 : 1_480_000_000
        return (defaultWeights, 256 << 20)
    }

    /// The geometry and prefill chunk size fitting resident weights inside the UMA budget.
    static func planResidencyAndChunk(
        raw: QWModelGeometry, cfg: StartupConfig, hal: QWHal, format: QWWeightFormat, tuning: QWEngineTuning,
        osReserveGiB: Double?, prefillChunkSize: Int, maxTokens: Int, ropeRows: Int,
        mtpDrafts: Int, mtpWeightDrafts: Int, kvMode: QWKVMode,
        allowExpertReduction: Bool = true
    ) throws -> (geometry: QWModelGeometry, prefillChunkSize: Int) {
        try QWScratchArenaBuilder.planResidencyAndChunk(
            raw: raw, cfg: cfg, hal: hal, format: format, tuning: tuning,
            osReserveGiB: osReserveGiB, prefillChunkSize: prefillChunkSize, maxTokens: maxTokens,
            ropeRows: ropeRows, mtpDrafts: mtpDrafts, mtpWeightDrafts: mtpWeightDrafts,
            kvMode: kvMode, allowExpertReduction: allowExpertReduction)
    }

    /// Predicts the GPU memory footprint (weights, KV cache, activations/GDN scratch) and
    /// checks whether `contextTokens` fits within the physical memory budget after OS reserve.
    public static func estimateContextMemory(
        descriptor: QWModelDescriptor,
        packPath: String? = nil,
        contextTokens: Int,
        kvMode: QWKVMode = .f16,
        mtpDrafts: Int = 0,
        mtpBackend: String = "variational",
        prefillChunkSize: Int = QWPrefillLimits.chunkRows,
        tuning: QWEngineTuning = .default,
        physicalMemoryBytes: Int? = QWAdmission.physicalMemoryBytes()
    ) -> QWContextMemoryEstimate {
        QWModuleRegistry.estimateContextMemory(
            descriptor: descriptor, packPath: packPath, contextTokens: contextTokens,
            kvMode: kvMode, mtpDrafts: mtpDrafts, mtpBackend: mtpBackend,
            prefillChunkSize: prefillChunkSize, tuning: tuning,
            physicalMemoryBytes: physicalMemoryBytes
        )
    }

    /// Concurrent GDN state slots sized to host physical RAM (4 on >=32 GiB, 2 on 16 GiB CI VMs).
    static func gdnSlotCount(physicalMemoryBytes: Int? = QWAdmission.physicalMemoryBytes()) -> Int {
        (physicalMemoryBytes ?? 0) >= (32 << 30) ? 4 : 2
    }

    static func estimateDriverContextMemory(
        descriptor: QWModelDescriptor,
        packPath: String? = nil,
        contextTokens: Int,
        kvMode: QWKVMode = .f16,
        mtpDrafts: Int = 0,
        mtpBackend: String = "variational",
        prefillChunkSize: Int = QWPrefillLimits.chunkRows,
        tuning: QWEngineTuning = .default,
        physicalMemoryBytes: Int? = QWAdmission.physicalMemoryBytes()
    ) -> QWContextMemoryEstimate {
        if case .recurrentLoop(_, let ink) = descriptor.executionTopology, ink != nil {
            return QWInkling2Runtime.estimateContextMemory(
                descriptor: descriptor, packPath: packPath, contextTokens: contextTokens,
                kvMode: kvMode, mtpDrafts: mtpDrafts, prefillChunkSize: prefillChunkSize,
                tuning: tuning, physicalMemoryBytes: physicalMemoryBytes)
        }
        if (descriptor.hcCount ?? 0) > 1 {
            return estimateHyperStreamContextMemory(
                descriptor: descriptor, packPath: packPath, contextTokens: contextTokens,
                kvMode: kvMode, mtpDrafts: mtpDrafts, mtpBackend: mtpBackend,
                prefillChunkSize: prefillChunkSize, tuning: tuning,
                physicalMemoryBytes: physicalMemoryBytes)
        }
        return QWScratchArenaBuilder.estimateDriverContextMemory(
            descriptor: descriptor, packPath: packPath, contextTokens: contextTokens,
            kvMode: kvMode, mtpDrafts: mtpDrafts, mtpBackend: mtpBackend,
            prefillChunkSize: prefillChunkSize, tuning: tuning,
            physicalMemoryBytes: physicalMemoryBytes,
            resolveFormat: { path, desc, rawGeo, weightDrafts in
                resolvePackFormatForEstimate(
                    packPath: path, descriptor: desc, rawGeometry: rawGeo, mtpWeightDrafts: weightDrafts)
            })
    }

    private static func estimateHyperStreamContextMemory(
        descriptor: QWModelDescriptor,
        packPath: String?,
        contextTokens: Int,
        kvMode: QWKVMode,
        mtpDrafts: Int,
        mtpBackend: String,
        prefillChunkSize: Int,
        tuning: QWEngineTuning,
        physicalMemoryBytes: Int?
    ) -> QWContextMemoryEstimate {
        let clampedContext = max(64, min(1_048_576, contextTokens))
        let isDFlash = mtpDrafts > 0 && FNGPUDFlash.isDFlashName(mtpBackend)
        let qsaBaseLayers = max(1, descriptor.numLayers - descriptor.linearAttentionLayers.count)
        let qsaLayers = qsaBaseLayers + max(1, descriptor.mtpLayers)
        let kvPerLayer: Int = switch kvMode {
        case .f16: 2 * descriptor.kvHeads * descriptor.headDim * 2
        case .q8:  2 * descriptor.kvHeads * descriptor.headDim + 2 * descriptor.kvHeads * 2
        case .q4:  descriptor.kvHeads * descriptor.headDim + 2 * descriptor.kvHeads * 8
        }
        let idxDim = descriptor.indexerHeadDim ?? 0
        let compress = max(1, descriptor.indexerCompressRatio ?? 4)
        let idxPerLayer = idxDim > 0 ? (idxDim * 2 + (idxDim * 2) / compress) : 0
        let rawPerTokBytes = qsaLayers * (kvPerLayer + idxPerLayer) + descriptor.rotaryDims * 8
        let perTokBytes = FNGPUDFlash.adjustedPerTokenKVBytes(rawPerTokBytes, usesDFlash: isDFlash)

        let bits = max(2, descriptor.quantization?.bits ?? 4)
        let groupSize = max(16, descriptor.quantization?.groupSize ?? 32)
        let bytesPerParam = Double(bits) / 8.0 + 4.0 / Double(groupSize)
        let embedBytes = descriptor.vocab * descriptor.hidden * 2
        let headBytes = Int(Double(descriptor.vocab * descriptor.hidden) * bytesPerParam)
        let moeCount = descriptor.numExperts ?? 0
        let denseInter = descriptor.denseIntermediate ?? 0
        let moeInter = descriptor.moeIntermediate ?? denseInter
        let sharedInter = descriptor.sharedExpertIntermediate ?? 0
        let mlpParams = (moeCount > 0 ? moeCount * 3 * descriptor.hidden * moeInter : 3 * descriptor.hidden * denseInter)
            + (sharedInter > 0 ? 3 * descriptor.hidden * sharedInter : 0)
        let gdnCount = descriptor.linearAttentionLayers.count
        let attnCount = max(0, descriptor.numLayers - gdnCount)
        let gdnProjParams = (descriptor.gdnQKVChannels + 2 * descriptor.gdnZChannels
            + descriptor.gdnVHeads * descriptor.gdnHeadDim) * descriptor.hidden
        let qsaProjParams = (descriptor.qHeads * descriptor.headDim + 2 * descriptor.kvHeads * descriptor.headDim
            + descriptor.qHeads * descriptor.headDim
            + (descriptor.indexerHeads ?? 0) * idxDim) * descriptor.hidden
        let hcCount = max(1, descriptor.hcCount ?? 1)
        let hcLowrank = descriptor.hcLowrank ?? 0
        let hcParams = 2 * descriptor.numLayers * (2 * hcCount * hcLowrank * hcCount * descriptor.hidden)
        let trunkWeightBytes = Int(Double(gdnCount * (mlpParams + gdnProjParams)
            + attnCount * (mlpParams + qsaProjParams) + hcParams) * bytesPerParam)
        let minWeights = max(512 << 20, embedBytes + headBytes + trunkWeightBytes)

        let onDisk = QWResidencyAdmission.flashNextOnDiskFootprint(packPath: packPath, minWeightsBytes: minWeights)
        let dflashDelta = FNGPUDFlash.estimateFootprint(
            targetDescriptor: descriptor, targetPackPath: packPath, dflashPackPath: nil, isDFlash: isDFlash)
        let weightsBytes = max(512 << 20, onDisk.weightsBytes - dflashDelta.nativeMTPWeightBytes + dflashDelta.dflashWeightBytes)
            + onDisk.pleBytes

        let alignedCap = QWCapacityPlan.steppedCapacity(clampedContext)
        let gdnStatePerSlot = gdnCount * (descriptor.gdnVHeads * descriptor.gdnHeadDim * descriptor.gdnHeadDim * 4 * 8
            + max(0, descriptor.gdnConvKernel - 1) * descriptor.gdnQKVChannels * 2 * 8)
        let pleHistoryPerSlot = max(0, descriptor.pleLayerIds?.count ?? 0) * descriptor.hidden * 2 * 8
        let prefillScratch = QWHyperStreamRuntime.prefillActivationBytes(
            maxRows: min(max(256, prefillChunkSize), 1024), capacity: alignedCap)
        let decodeScratch = max(32 << 20, descriptor.vocab * 4 * 8 + hcCount * descriptor.hidden * 4 * 64
            + max(moeInter, denseInter) * 4 * 32)
        let lsaClusterBytes = idxDim > 0 ? (qsaLayers * max(1, (alignedCap + 2047) / 2048) * 64 * idxDim * 2) : 0
        let scratchBytes = FNGPUBatchScheduler.maxSlots * (gdnStatePerSlot + pleHistoryPerSlot)
            + prefillScratch + decodeScratch + lsaClusterBytes
            + (isDFlash ? max(0, dflashDelta.dflashScratchBytes - (112 << 20)) : 0)

        let fixedBytes = weightsBytes + scratchBytes
        let reserveGiB = tuning.osReserveGiB ?? Double(QWCapacityBudget.osReserveBytes >> 30)
        let reserveBytes = max(0, Int((reserveGiB * Double(1 << 30)).rounded()))
        let availBytes = max(0, (physicalMemoryBytes ?? 0) - reserveBytes)
        let kvCap = availBytes > 0
            ? QWHyperStreamRuntime.resolveKVCapacity(
                alignedCap: alignedCap,
                fixedBytes: fixedBytes,
                perTokenBytes: perTokBytes,
                availableBytes: availBytes
            )
            : (alignedCap >= 4096 ? (alignedCap + 1024) : alignedCap)
        let kvBytes = kvCap * perTokBytes
        let requiredBytes = fixedBytes + kvBytes
        let maxFitting = availBytes > fixedBytes ? min(1_048_576, (availBytes - fixedBytes) / perTokBytes) : 0
        return QWContextMemoryEstimate(
            contextTokens: clampedContext, weightsBytes: weightsBytes, kvBytes: kvBytes,
            scratchAndStateBytes: scratchBytes, requiredBytes: requiredBytes,
            availableBytes: availBytes, maxFittingContext: maxFitting,
            fits: availBytes == 0 || requiredBytes <= availBytes)
    }

    private static func resolvePackFormatForEstimate(
        packPath: String?,
        descriptor: QWModelDescriptor,
        rawGeometry: QWModelGeometry,
        mtpWeightDrafts: Int
    ) -> (QWWeightFormat, Bool, QWWeightFormat?) {
        if let path = packPath, !path.isEmpty {
            let cacheKey = "\(path)|\(mtpWeightDrafts > 0)|\(descriptor.geometryDigestV2)"
            if let hit = cachedPackFormats.withLock({ $0[cacheKey] }) { return hit }
            let root = QWModelRoot(url: URL(fileURLWithPath: path))
            let isSafetensors = (root.kind == .safetensorsSingle || root.kind == .safetensorsSharded)
            let pk = (root.kind == .qwp1 || root.kind == nil)
                ? (try? QWQuant.Pack(url: URL(fileURLWithPath: path), descriptor: descriptor))
                : nil
            let st = isSafetensors ? (try? QWSafeTensorsContainer(url: root.weights)) : nil
            if let choice = try? Self.weightFormatChoice(
                for: root, quantization: descriptor.quantization, geometry: rawGeometry,
                mtpDrafts: mtpWeightDrafts, pack: pk) {
                let headSpec = Self.resolveHeadSpec(st: st, pack: pk, trunkFormat: choice.format,
                                                    geometry: rawGeometry)
                let res = (choice.format, headSpec.isQ4g, headSpec.format)
                cachedPackFormats.withLock { $0[cacheKey] = res }
                return res
            }
            if isSafetensors {
                if let q = descriptor.quantization, let b = q.bits, let g = q.groupSize,
                   QWAQBlock.isSupported(bits: b, group: g) {
                    let trunkFmt = QWWeightFormat.nativeAffine(bits: b, groupSize: g)
                    let headSpec = Self.resolveHeadSpec(st: st, pack: nil, trunkFormat: trunkFmt,
                                                        geometry: rawGeometry)
                    return (trunkFmt, headSpec.isQ4g, headSpec.format)
                }
                return (.hq4G32, true, .hq4G32)
            }
        }
        if let q = descriptor.quantization, let b = q.bits, let g = q.groupSize,
           QWAQBlock.isSupported(bits: b, group: g) {
            return (.nativeAffine(bits: b, groupSize: g), false, nil)
        }
        return (.hq4G32, false, nil)
    }
}
