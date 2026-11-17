// DriverResidency.swift — How many experts per MoE layer stay resident.
// The planner sizes the resident bank to the memory budget the tuning's
// memory policy names; the rest of each layer's experts stream from SSD.
import Foundation
import os
import QWCore
import QWHAL
import QWHead
import QWGDN
import QWKV
import QWQuant

extension QWEngineDriver {
    private static let cachedPackFormats = OSAllocatedUnfairLock<[String: QWWeightFormat]>(initialState: [:])

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
        return min(physical, max(1 << 30, live - headroom))
    }

    /// The geometry and prefill chunk size fitting resident weights inside the UMA budget.
    static func planResidencyAndChunk(
        raw: QWModelGeometry, cfg: StartupConfig, hal: QWHal, format: QWWeightFormat, tuning: QWEngineTuning,
        osReserveGiB: Double?, prefillChunkSize: Int, maxTokens: Int, ropeRows: Int,
        mtpDrafts: Int, mtpWeightDrafts: Int, kvMode: QWKVMode
    ) -> (geometry: QWModelGeometry, prefillChunkSize: Int) {
        let maxMTP0 = cfg.mtpBackend == .dflash2 ? 15 : 7
        let arenaCap0 = max(1 << 30, hal.caps.maxWorkingSetBytes - (64 << 20))
        let avail0 = Self.residencyBudgetBytes(policy: tuning.memoryPolicy, osReserveGiB: osReserveGiB, arenaCap: arenaCap0)
        let chunk0 = max(32, min(QWPrefillLimits.maxRows, (prefillChunkSize / 32) * 32))
        let useNax0 = hal.caps.isMetal4TensorCapable
        let numV0 = raw.numVirtualLayers
        let attnCount0 = (0..<numV0).filter { raw.isVirtualLayerFullAttention($0) }.count
        let gdnCount0 = raw.gdnLayerCount
        let slotCount = Self.gdnSlotCount()
        let slotB0 = gdnCount0 * (QWSlotGeometry.recElems(geometry: raw) * 4 + QWSlotGeometry.ringElems(geometry: raw) * 2)
        let trunkHidB0 = (raw.mtpSharedLastLayer && raw.mhcConfig != nil)
            ? raw.mhcConfig!.streams * raw.hidden * 4
            : raw.hidden * 2
        let gdn0 = slotCount * slotB0 + Self.partialsScratchBytes(geometry: raw, mtpDrafts: mtpDrafts, maxTokens: maxTokens)
            + (mtpDrafts > 0 ? QWGDNReplayCheckpoint.allocationBytes(geometry: raw, layers: gdnCount0, rows: mtpDrafts + 1) : 0)
            + (mtpDrafts > 0 ? slotCount * trunkHidB0 : 0)
            + (1 << 20)
        let activeAttn0 = attnCount0 + (mtpDrafts > 0 ? 1 : 0)
        let modelCap0 = raw.maxPositionEmbeddings > 0 ? min(maxTokens, raw.maxPositionEmbeddings) : maxTokens
        let minFitTokens0 = min(modelCap0, max(16_384, min(maxTokens, 32_768)))
        let kv0 = Self.kvBudgetBytes(maxTokens: minFitTokens0, mtpDrafts: mtpDrafts, kvMode: kvMode, geometry: raw)
            + Self.hcaPerTokenBytes(geometry: raw) * minFitTokens0
        let devEmbedEnv = cfg.environment["QW_MTP_DEVICE_EMBEDDING"] ?? cfg.environment["QW_MTP_DEVICE_EMBED"]
        let forceDeviceEmbed = devEmbedEnv == "1"
        let disableDeviceEmbed = devEmbedEnv == "0"
        let fullWPlan = try? QWWeightAllocationPlan(
            geometry: raw, ropeRows: max(ropeRows, maxTokens),
            mtpDrafts: mtpWeightDrafts, headQ4g: cfg.headIsQ4g, maxMTPDrafts: maxMTP0,
            format: format)
        let fullBytes0 = fullWPlan?.bytes ?? 0
        let embedBytes0 = raw.vocab * raw.hidden * 2

        func safeWeightCap(forChunk chunkRows: Int) -> Int {
            let bpt = Self.prefillBytesPerToken(geometry: raw, maxPrefillChunk: chunkRows, useGroupedNAX: useNax0)
            let act = chunkRows * bpt + Self.decodeAndMTPScratchBytes(
                geometry: raw, prefillChunkRows: chunkRows, mtpDrafts: mtpDrafts,
                maxTokens: maxTokens, useGroupedNAX: useNax0)
            let mlaFixed = Self.mlaFixedScratchBytes(geometry: raw, prefillChunkRows: chunkRows, activeAttnLayers: activeAttn0)
                + Self.hcaFixedScratchBytes(geometry: raw, prefillChunkRows: chunkRows)
                + (mtpDrafts > 0 && raw.mtpSharedLastLayer ? (112 << 20) : 0)
            let baseReserve = act + gdn0 + mlaFixed + kv0 + (256 << 20)
            let safeCapWithoutEmbed = max(1 << 30, avail0 - baseReserve)
            let canFitDeviceEmbed = fullBytes0 > 0 && (fullBytes0 + embedBytes0 + (1 << 30) <= safeCapWithoutEmbed)
            let mtpEmbed = (!disableDeviceEmbed && mtpDrafts > 0
                && (forceDeviceEmbed || (mtpDrafts > 1 && canFitDeviceEmbed))) ? embedBytes0 : 0
            return max(1 << 30, avail0 - (baseReserve + mtpEmbed))
        }

        if fullBytes0 <= safeWeightCap(forChunk: chunk0) {
            return (raw, chunk0)
        }
        for candChunk in [3584, 3072, 2560, 2048, 1536, 1024, 512] where candChunk < chunk0 {
            if fullBytes0 <= safeWeightCap(forChunk: candChunk) {
                return (raw, candChunk)
            }
        }

        let fallbackChunk = min(chunk0, 1024)
        let fallbackCap = safeWeightCap(forChunk: fallbackChunk)
        let minE = max(8, raw.expertsPerTok)
        var chosenE = minE
        var cand = ((raw.numExperts - 2) / 2) * 2
        while cand >= minE {
            let candGeo = raw.withResidentExperts(cand)
            if let p = try? QWWeightAllocationPlan(
                geometry: candGeo, ropeRows: max(ropeRows, maxTokens),
                mtpDrafts: mtpWeightDrafts, headQ4g: cfg.headIsQ4g, maxMTPDrafts: maxMTP0,
                format: format),
               p.bytes <= fallbackCap {
                chosenE = cand
                break
            }
            cand -= 2
        }
        return (raw.withResidentExperts(chosenE), fallbackChunk)
    }

    static func planResidentGeometry(
        raw: QWModelGeometry, cfg: StartupConfig, hal: QWHal, format: QWWeightFormat, tuning: QWEngineTuning,
        osReserveGiB: Double?, prefillChunkSize: Int, maxTokens: Int, ropeRows: Int,
        mtpDrafts: Int, mtpWeightDrafts: Int, kvMode: QWKVMode
    ) -> QWModelGeometry {
        planResidencyAndChunk(
            raw: raw, cfg: cfg, hal: hal, format: format, tuning: tuning,
            osReserveGiB: osReserveGiB, prefillChunkSize: prefillChunkSize, maxTokens: maxTokens,
            ropeRows: ropeRows, mtpDrafts: mtpDrafts, mtpWeightDrafts: mtpWeightDrafts, kvMode: kvMode
        ).geometry
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
        let clampedContext = max(64, min(1_048_576, contextTokens))
        let rawGeometry = QWModelGeometry(descriptor: descriptor).withResidentExperts(tuning.residentExpertsPerLayer)
        let effDrafts = max(0, mtpDrafts)
        let isDFlash = ["dflash2", "dflash", "dspine"].contains(mtpBackend.lowercased())
        let maxMTP = isDFlash ? 15 : 7
        let mtpWeightDrafts = (descriptor.mtpLayers > 0 && !isDFlash) ? min(effDrafts, maxMTP) : 0
        let format: QWWeightFormat = {
            if let path = packPath, !path.isEmpty {
                if let hit = cachedPackFormats.withLock({ $0[path] }) { return hit }
                let root = QWModelRoot(url: URL(fileURLWithPath: path))
                let pk = (root.kind == .qwp1 || root.kind == nil)
                    ? (try? QWQuant.Pack(url: URL(fileURLWithPath: path), descriptor: descriptor))
                    : nil
                if let choice = try? Self.weightFormatChoice(
                    for: root, quantization: descriptor.quantization, geometry: rawGeometry,
                    mtpDrafts: effDrafts, pack: pk) {
                    cachedPackFormats.withLock { $0[path] = choice.format }
                    return choice.format
                }
            }
            if let q = descriptor.quantization, let b = q.bits, let g = q.groupSize, QWAQBlock.isSupported(bits: b, group: g) {
                return .nativeAffine(bits: b, groupSize: g)
            }
            return .hq4G32
        }()
        let chunkRows = max(32, min(QWPrefillLimits.maxRows, (prefillChunkSize / 32) * 32))
        let bpt = Self.prefillBytesPerToken(geometry: rawGeometry, maxPrefillChunk: chunkRows)
        let actBytes = chunkRows * bpt + Self.decodeAndMTPScratchBytes(
            geometry: rawGeometry, prefillChunkRows: chunkRows, mtpDrafts: effDrafts, maxTokens: clampedContext)
        let numV = rawGeometry.numVirtualLayers
        let attnCount = (0..<numV).filter { rawGeometry.isVirtualLayerFullAttention($0) }.count
        let slotCount = Self.gdnSlotCount(physicalMemoryBytes: physicalMemoryBytes)
        let gdnCount = rawGeometry.gdnLayerCount
        let slotBytes = gdnCount * (QWSlotGeometry.recElems(geometry: rawGeometry) * 4
            + QWSlotGeometry.ringElems(geometry: rawGeometry) * 2)
        let trunkHidBytes = (rawGeometry.mtpSharedLastLayer && rawGeometry.mhcConfig != nil)
            ? rawGeometry.mhcConfig!.streams * rawGeometry.hidden * 4
            : rawGeometry.hidden * 2
        let gdnBytes = slotCount * slotBytes + Self.partialsScratchBytes(geometry: rawGeometry, mtpDrafts: effDrafts, maxTokens: clampedContext)
            + (effDrafts > 0 ? QWGDNReplayCheckpoint.allocationBytes(geometry: rawGeometry, layers: gdnCount, rows: effDrafts + 1) : 0)
            + (effDrafts > 0 ? slotCount * trunkHidBytes : 0) + (1 << 20)
        let activeAttnLayers = attnCount + (effDrafts > 0 ? 1 : 0)
        let mlaFixedBytes = Self.mlaFixedScratchBytes(
            geometry: rawGeometry, prefillChunkRows: chunkRows, activeAttnLayers: activeAttnLayers)
            + Self.hcaFixedScratchBytes(geometry: rawGeometry, prefillChunkRows: chunkRows)
            + (effDrafts > 0 && rawGeometry.mtpSharedLastLayer ? (112 << 20) : 0)
        let ropeRows = max(rawGeometry.maxPositionEmbeddings, clampedContext)
        let isMoE = rawGeometry.numExperts > 0
        let reserveGiB = isMoE
            ? Self.effectiveOSReserveGiB(policy: tuning.memoryPolicy, explicitReserveGiB: tuning.osReserveGiB)
            : (tuning.osReserveGiB ?? Double(QWCapacityBudget.osReserveBytes >> 30))
        let availBytes = max(0, (physicalMemoryBytes ?? 0) - max(0, Int((reserveGiB * Double(1 << 30)).rounded())))
        let geometry = { () -> QWModelGeometry in
            guard isMoE, tuning.residentExpertsPerLayer == nil, availBytes > 0 else { return rawGeometry }
            let modelCap = rawGeometry.maxPositionEmbeddings > 0
                ? min(clampedContext, rawGeometry.maxPositionEmbeddings)
                : clampedContext
            let minFitTokens = min(modelCap, max(16_384, min(clampedContext, 32_768)))
            let minFitKV = Self.kvBudgetBytes(maxTokens: minFitTokens,
                                              mtpDrafts: effDrafts, kvMode: kvMode, geometry: rawGeometry)
                + Self.hcaPerTokenBytes(geometry: rawGeometry) * minFitTokens
            let safeWeightCap = max(1 << 30, availBytes - (actBytes + gdnBytes + mlaFixedBytes + minFitKV + (256 << 20)))
            let fullPlan = try? QWWeightAllocationPlan(geometry: rawGeometry, ropeRows: ropeRows, mtpDrafts: mtpWeightDrafts,
                                                       headQ4g: true, maxMTPDrafts: maxMTP, format: format)
            guard let fullBytes = fullPlan?.bytes, fullBytes > safeWeightCap else { return rawGeometry }
            let minE = max(8, rawGeometry.expertsPerTok)
            var chosenE = minE
            var cand = ((rawGeometry.numExperts - 2) / 2) * 2
            while cand >= minE {
                let candGeo = rawGeometry.withResidentExperts(cand)
                if let p = try? QWWeightAllocationPlan(geometry: candGeo, ropeRows: ropeRows, mtpDrafts: mtpWeightDrafts,
                                                       headQ4g: true, maxMTPDrafts: maxMTP, format: format),
                   p.bytes <= safeWeightCap {
                    chosenE = cand; break
                }
                cand -= 2
            }
            return rawGeometry.withResidentExperts(chosenE)
        }()
        let weightsBytes = (try? QWWeightAllocationPlan(
            geometry: geometry, ropeRows: ropeRows, mtpDrafts: mtpWeightDrafts,
            headQ4g: true, maxMTPDrafts: maxMTP, format: format).bytes) ?? 0
        let mlaPerTokBytes = geometry.usesAnyMLA
            ? activeAttnLayers * ((geometry.indexerHeadDim * 2) / max(1, geometry.indexerCompressRatio))
            : 0
        let hcaPerTokBytes = Self.hcaPerTokenBytes(geometry: geometry)
        let perTokBytes = 2 * activeAttnLayers * (QWKVCache.pageBytes(mode: kvMode, geometry: geometry) / QWKVCache.pageTokens)
            + mlaPerTokBytes + hcaPerTokBytes
        let kvBytes = Self.kvBudgetBytes(maxTokens: clampedContext, mtpDrafts: effDrafts, kvMode: kvMode, geometry: geometry)
            + (mlaPerTokBytes + hcaPerTokBytes) * clampedContext
        let scratchBytes = actBytes + gdnBytes + mlaFixedBytes
        let fixedBytes = weightsBytes + scratchBytes
        let requiredBytes = fixedBytes + kvBytes
        let maxFitting = availBytes > fixedBytes ? min(1_048_576, (availBytes - fixedBytes) / max(1, perTokBytes)) : 0
        return QWContextMemoryEstimate(
            contextTokens: clampedContext, weightsBytes: weightsBytes, kvBytes: kvBytes,
            scratchAndStateBytes: scratchBytes, requiredBytes: requiredBytes,
            availableBytes: availBytes, maxFittingContext: maxFitting,
            fits: availBytes == 0 || requiredBytes <= availBytes)
    }
}
