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
    private static let cachedPackFormats = OSAllocatedUnfairLock<[String: (format: QWWeightFormat, headQ4g: Bool)]>(initialState: [:])

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
        QWScratchArenaBuilder.estimateDriverContextMemory(
            descriptor: descriptor, packPath: packPath, contextTokens: contextTokens,
            kvMode: kvMode, mtpDrafts: mtpDrafts, mtpBackend: mtpBackend,
            prefillChunkSize: prefillChunkSize, tuning: tuning,
            physicalMemoryBytes: physicalMemoryBytes,
            resolveFormat: { path, desc, rawGeo, weightDrafts in
                resolvePackFormatForEstimate(
                    packPath: path, descriptor: desc, rawGeometry: rawGeo, mtpWeightDrafts: weightDrafts)
            })
    }

    private static func resolvePackFormatForEstimate(
        packPath: String?,
        descriptor: QWModelDescriptor,
        rawGeometry: QWModelGeometry,
        mtpWeightDrafts: Int
    ) -> (QWWeightFormat, Bool) {
        if let path = packPath, !path.isEmpty {
            if let hit = cachedPackFormats.withLock({ $0[path] }) { return hit }
            let root = QWModelRoot(url: URL(fileURLWithPath: path))
            let isSafetensors = (root.kind == .safetensorsSingle || root.kind == .safetensorsSharded)
            let pk = (root.kind == .qwp1 || root.kind == nil)
                ? (try? QWQuant.Pack(url: URL(fileURLWithPath: path), descriptor: descriptor))
                : nil
            let resolvedHeadQ4g = isSafetensors
                ? true
                : ((try? pk?.entry("lm_head.weight").quantized) ?? false)
            if let choice = try? Self.weightFormatChoice(
                for: root, quantization: descriptor.quantization, geometry: rawGeometry,
                mtpDrafts: mtpWeightDrafts, pack: pk) {
                cachedPackFormats.withLock { $0[path] = (choice.format, resolvedHeadQ4g) }
                return (choice.format, resolvedHeadQ4g)
            }
            if isSafetensors {
                if let q = descriptor.quantization, let b = q.bits, let g = q.groupSize,
                   QWAQBlock.isSupported(bits: b, group: g) {
                    return (.nativeAffine(bits: b, groupSize: g), true)
                }
                return (.hq4G32, true)
            }
        }
        if let q = descriptor.quantization, let b = q.bits, let g = q.groupSize,
           QWAQBlock.isSupported(bits: b, group: g) {
            return (.nativeAffine(bits: b, groupSize: g), false)
        }
        return (.hq4G32, false)
    }
}
