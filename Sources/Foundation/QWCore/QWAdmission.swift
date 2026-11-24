// QWAdmission.swift — Dense working-set admission planner (Qwen3.8-27B, qwen3_5).
// Computes explicit working-set budgets from QWModelDescriptor fields
// (model resident bytes, KV, GDN state, and scratch) with deterministic safe failure.

import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// HQ4_G32_M5 plane geometry shared by admission and the pack reader (QWQuant aliases it).
public enum QWHQ4Layout {
    public static let group = 32
    public static let blockBytes = 20
}

/// KV storage dtype widths (`f16` = 2 B, `q8` = 1 B, `q4` = 0.5 B).
public enum QWAdmissionDtype: Sendable {
    case f16, q8, q4
    public var bytesPerParam: Int { switch self { case .f16: 2; case .q8, .q4: 1 } }
    public func kvBytes(elements: Int) -> Int {
        switch self { case .f16: elements * 2; case .q8: elements; case .q4: elements / 2 }
    }
}

public enum QWAdmissionError: Error, CustomStringConvertible, Equatable {
    case bad(String)
    public var description: String {
        switch self { case .bad(let s): return "QWAdmission: \(s)" }
    }
}

/// Deterministic working-set admission plan for one or more sequences.
public struct QWAdmissionPlan: Sendable, Equatable {
    public struct Components: Sendable, Equatable, CustomStringConvertible {
        public let modelResidentBytes: Int
        public let kvBytes: Int
        public let gdnStateBytes: Int
        public let scratchBytes: Int
        public var total: Int { modelResidentBytes + kvBytes + gdnStateBytes + scratchBytes }
        public var description: String {
            "model \(modelResidentBytes) kv \(kvBytes) gdn \(gdnStateBytes) scratch \(scratchBytes) → TOTAL \(total)"
        }
    }

    public let descriptor: QWModelDescriptor
    public let components: Components
    public let workingSetCapBytes: Int
    public let contextTokens: Int
    public let kvDtype: QWAdmissionDtype
    public let scratchPerSeq: Int

    public var headroomBytes: Int { workingSetCapBytes - components.total }

    /// Binary admission for the current configuration: fit iff headroom >= 0.
    public func admit() -> QWAdmissionVerdict {
        headroomBytes >= 0 ? .admitted(headroomBytes: headroomBytes)
                           : .refused(shortByBytes: -headroomBytes)
    }

    /// Marginal admission: checks whether one additional sequence fits in headroom.
    public func admitAnotherSequence() -> QWAdmissionVerdict {
        let s = Self.perSequenceStateBytes(descriptor: descriptor,
                                           kvDtype: kvDtype,
                                           contextTokens: contextTokens,
                                           scratchPerSeq: scratchPerSeq)
        let per = s.kv + s.gdn + s.scratch
        return per <= headroomBytes ? .admitted(headroomBytes: headroomBytes - per)
                                    : .refused(shortByBytes: per - headroomBytes)
    }

    /// Build the admission plan from descriptor-derived component formulas.
    public init(descriptor: QWModelDescriptor,
                contextTokens: Int,
                kvDtype: QWAdmissionDtype = .q4,
                scratchPerSeq: Int,
                workingSetCapBytes: Int) throws {
        guard contextTokens > 0, scratchPerSeq >= 0, workingSetCapBytes > 0 else {
            throw QWAdmissionError.bad("admission: contextTokens/scratch/cap must be positive (scratch ≥ 0)")
        }
        let model = try Self.modelResidentBytes(descriptor: descriptor)
        let perSeq = Self.perSequenceStateBytes(descriptor: descriptor,
                                                kvDtype: kvDtype,
                                                contextTokens: contextTokens,
                                                scratchPerSeq: scratchPerSeq)
        self.descriptor = descriptor
        self.contextTokens = contextTokens
        self.kvDtype = kvDtype
        self.scratchPerSeq = scratchPerSeq
        self.workingSetCapBytes = workingSetCapBytes
        self.components = Components(
            modelResidentBytes: model,
            kvBytes: perSeq.kv, gdnStateBytes: perSeq.gdn, scratchBytes: perSeq.scratch)
    }

    /// Stable report dictionary for CLI and ledger comparison.
    public func report() -> [String: Any] {
        [
            "model": descriptor.family.rawValue,
            "components": [
                "model_resident_bytes": components.modelResidentBytes,
                "kv_bytes": components.kvBytes,
                "gdn_state_bytes": components.gdnStateBytes,
                "scratch_bytes": components.scratchBytes,
                "total_bytes": components.total,
            ],
            "working_set_cap_bytes": workingSetCapBytes,
            "headroom_bytes": headroomBytes,
            "admit": admit() == .admitted(headroomBytes: headroomBytes),
        ]
    }

    // MARK: - Component Formulas

    /// Resident weights of the model: HQ4_G32_M5 for standard dense/MoE models, and
    /// config-derived mixed-precision (`HQ8_G32` trunk + `SHVQ` routed experts) for hybrid HCA / Latent-MoE models.
    public static func modelResidentBytes(descriptor d: QWModelDescriptor) throws -> Int {
        if case .recurrentLoop(_, let ink) = d.executionTopology, ink != nil {
            return inkling2ModelResidentBytes(descriptor: d)
        }
        let hcaLayers = d.compressedAttentionLayers.count
        if hcaLayers > 0 || d.moeLatentSize != nil || d.hybridMLAConfig != nil
            || (d.mlaConfig != nil && !d.mlaUseNoPE) {
            return try hybridModelResidentBytes(descriptor: d)
        }
        let hq4 = { (elems: Int) in elems / QWHQ4Layout.group * QWHQ4Layout.blockBytes }
        var t = 0
        t += hq4(d.vocab * d.hidden) * 2
        let attnLayers = (0..<d.numLayers).filter { d.isFullAttention($0) }
        let gdnLayers = d.linearAttentionLayers.count
        t += hq4(attnLayers.count * (d.qProjOut * d.hidden
                                     + 2 * d.kvHeads * d.headDim * d.hidden
                                     + d.qHeads * d.headDim * d.hidden))
        t += hq4(gdnLayers * gdnLayerWeightElems(d))
        let mlpPerLayerElems: Int
        if let numE = d.numExperts, let moeI = d.moeIntermediate {
            let denseLayers = min(d.numLayers, max(0, d.firstKDenseReplace))
            let moeLayers = d.numLayers - denseLayers
            let denseI = d.denseIntermediate ?? moeI
            let sharedI = d.sharedExpertIntermediate ?? 0
            t += hq4(denseLayers * 3 * (denseI * d.hidden))
            t += hq4(moeLayers * 3 * ((numE * moeI + sharedI) * d.hidden))
            mlpPerLayerElems = 3 * ((numE * moeI + sharedI) * d.hidden)
        } else if let mlpI = d.denseIntermediate {
            t += hq4(d.numLayers * 3 * (mlpI * d.hidden))
            mlpPerLayerElems = 3 * (mlpI * d.hidden)
        } else {
            throw QWAdmissionError.bad("admission: descriptor has neither dense nor MoE MLP intermediate size")
        }
        let mtpBlock = d.qProjOut * d.hidden + 2 * d.kvHeads * d.headDim * d.hidden
            + d.qHeads * d.headDim * d.hidden + mlpPerLayerElems + 2 * d.hidden * d.hidden
        t += hq4(d.mtpLayers * mtpBlock)
        return t
    }

    private static func gdnLayerWeightElems(_ d: QWModelDescriptor) -> Int {
        let base = d.gdnQKVChannels * d.hidden
            + d.gdnConvKernel * d.gdnQKVChannels
            + d.hidden * d.gdnVHeads * d.gdnHeadDim
        if d.gdnVectorDecay {
            if let kda = d.kdaConfig, kda.isFullRank {
                return base + (2 * d.gdnZChannels + d.gdnVHeads) * d.hidden
            }
            let lr = d.gdnLowRankDim ?? d.gdnHeadDim
            return base + (2 * lr + d.gdnVHeads) * d.hidden + 2 * d.gdnZChannels * lr
        }
        return base + (d.gdnZChannels + 2 * d.gdnVHeads) * d.hidden
    }

    private static func hybridModelResidentBytes(descriptor d: QWModelDescriptor) throws -> Int {
        let hq8 = { (elems: Int) in (elems / 32) * 34 }
        let parseSHVQ = { (tag: String?, fallbackBits: Int) -> (group: Int, blockBytes: Int) in
            let fallbackPayload = fallbackBits == 15 ? (64 * 3) / 16 : (64 * fallbackBits) / 8
            guard let raw = tag?.lowercased(), raw.hasPrefix("shvq") else {
                return (64, 2 + fallbackPayload)
            }
            let suffix = raw.dropFirst(4).replacingOccurrences(of: "_", with: "")
            let parts = suffix.split(separator: "g")
            guard parts.count == 2, let bits = Int(parts[0]), let group = Int(parts[1]),
                  group > 0, group % 32 == 0,
                  bits == 1 || bits == 15 || (2...4).contains(bits) else {
                return (64, 2 + fallbackPayload)
            }
            let payload = bits == 15 ? (group * 3) / 16 : (group * bits) / 8
            return (group, 2 + payload)
        }
        let guSpec = parseSHVQ(d.quantization?.expertGateUp, 15)
        let dnSpec = parseSHVQ(d.quantization?.expertDown, 3)

        var bytes = d.vocab * d.hidden * 2 + hq8(d.vocab * d.hidden)
        var trunkElems = 0
        var rawBytes = 0

        if let hca = d.hcaConfig {
            let hcaCount = d.compressedAttentionLayers.count
            let perHCA = d.hidden * hca.qLoraRank
                + hca.qLoraRank * hca.qHeads * hca.headDim
                + (hca.gatedAttnUseQLora ? hca.qLoraRank : d.hidden) * (hca.oGroups * hca.oLoraRank)
                + d.hidden * (hca.kvHeads * hca.headDim)
                + 2 * d.hidden * hca.headDim
                + hca.oGroups * hca.oLoraRank * ((hca.qHeads / max(1, hca.oGroups)) * hca.vHeadDim)
                + d.hidden * (hca.oGroups * hca.oLoraRank)
            trunkElems += hcaCount * perHCA
            rawBytes += hcaCount * (hca.slidingWindow * hca.headDim * 4 + hca.qHeads * hca.qHeads * 4)
        }

        let mlaCount = d.fullAttentionLayers.count
        if let mla = d.mlaConfig, mlaCount > 0 {
            let idxH = d.indexerHeads ?? 0
            let idxD = d.indexerHeadDim ?? 0
            let qElems = mla.qLoraRank > 0
                ? (d.hidden * mla.qLoraRank + mla.qLoraRank * mla.qHeads * (mla.qkNopeHeadDim + mla.qkRopeHeadDim))
                : (d.hidden * mla.qHeads * (mla.qkNopeHeadDim + mla.qkRopeHeadDim))
            let gateElems: Int = {
                switch mla.outputGateMode {
                case .none: return 0
                case .elementWiseFromQLora: return max(0, mla.qLoraRank) * mla.qHeads * mla.vHeadDim
                case .elementWiseFromHidden: return d.hidden * mla.qHeads * mla.vHeadDim
                case .headWiseFromHidden: return d.hidden * mla.qHeads
                }
            }()
            let idxElems = (idxH > 0 && idxD > 0) ? (max(d.hidden, mla.qLoraRank) * idxH * idxD + d.hidden * idxD + d.hidden * idxH) : 0
            let perMLA = qElems + gateElems
                + d.hidden * (mla.kvLoraRank + mla.qkRopeHeadDim)
                + mla.kvLoraRank * mla.qHeads * (mla.qkNopeHeadDim + mla.vHeadDim)
                + d.hidden * (mla.qHeads * mla.vHeadDim)
                + idxElems
            trunkElems += mlaCount * perMLA
            rawBytes += mlaCount * (mla.qHeads * mla.qHeads * 4)
        }

        let gdnCount = d.linearAttentionLayers.count
        if gdnCount > 0 {
            trunkElems += gdnCount * gdnLayerWeightElems(d)
        }

        if let numE = d.numExperts, let moeI = d.moeIntermediate {
            let denseCount = min(d.numLayers, max(0, d.firstKDenseReplace))
            let moeCount = d.numLayers - denseCount
            let denseI = d.denseIntermediate ?? moeI
            let sharedI = d.sharedExpertIntermediate ?? 0
            let routedDim = d.moeLatentSize ?? d.hidden
            trunkElems += denseCount * 3 * denseI * d.hidden
            if routedDim != d.hidden {
                trunkElems += moeCount * 2 * d.hidden * routedDim
            }
            trunkElems += moeCount * 3 * sharedI * d.hidden
            let gateUpElems = moeCount * numE * 2 * moeI * routedDim
            let downElems = moeCount * numE * moeI * routedDim
            bytes += (gateUpElems / guSpec.group) * guSpec.blockBytes + (downElems / dnSpec.group) * dnSpec.blockBytes
            let routerSplits = d.moeSplitFeatureRouting ? 2 : 1
            rawBytes += moeCount * routerSplits * numE * d.hidden * 4
        } else if let mlpI = d.denseIntermediate {
            trunkElems += d.numLayers * 3 * mlpI * d.hidden
        } else {
            throw QWAdmissionError.bad("admission: descriptor has neither dense nor MoE MLP intermediate size")
        }

        if let mhc = d.mhcConfig {
            let mixRows = (2 + mhc.streams) * mhc.streams
            rawBytes += (mhc.end - mhc.start) * 2 * mixRows * (mhc.streams * d.hidden) * 4
        }
        if d.mtpLayers > 0 {
            trunkElems += d.mtpLayers * 2 * d.hidden * d.hidden
        }
        bytes += hq8(trunkElems) + rawBytes
        return bytes
    }

    private static func inkling2ModelResidentBytes(descriptor d: QWModelDescriptor) -> Int {
        let ink = d.inkling2Config
        let qDim = d.qHeads * d.headDim
        let kvDim = d.kvHeads * d.headDim
        let dRel = ink?.dRel ?? 16
        let relExt = ink?.relExtent ?? 1024
        let convK = ink?.sconvKernelSize ?? 4
        let nExp = d.numExperts ?? 0
        let moeI = d.moeIntermediate ?? 0
        var total = 2 * d.vocab * d.hidden * 2 // BF16 embed + lm_head
        let perLayerDenseElems = (2 * qDim + 2 * kvDim + qDim) * d.hidden
            + dRel * d.qHeads * d.hidden + d.qHeads * d.hidden
            + (2 * d.hidden + 2 * kvDim) * convK + dRel * relExt + d.hidden * 4 + nExp * d.hidden
        total += d.numLayers * perLayerDenseElems * 2
        if let ink {
            let loopLayers = max(0, ink.loopCoreEnd - ink.loopCoreStart)
            total += loopLayers * 4 * (ink.loopGateRank * 2 * d.hidden + 2 * d.hidden * ink.loopGateRank) * 2
        }
        let expPerLayerElems = 3 * nExp * moeI * d.hidden
        for layer in 0..<d.numLayers {
            let mode = ink?.expertProjSpec(layer: layer, projection: "gate_proj").mode ?? .bf16
            switch mode {
            case .nvfp4: total += (expPerLayerElems / 16) * 9 + 3 * nExp * 4
            case .mxfp8: total += (expPerLayerElems / 32) * 33
            case .bf16: total += expPerLayerElems * 2
            }
        }
        if let vis = ink?.vision {
            let vh = vis.hiddenSize
            let mlpH = Int(Double(vh) * vis.mlpRatio)
            let visElems = vis.numHiddenLayers * (4 * vh * vh + 2 * vh * mlpH)
                + vis.projectionHiddenDim * (vis.projectionHiddenDim + vis.textHiddenSize)
            total += visElems * 2
        }
        if let ds = ink?.dspark {
            let dsKV = ds.kvHeads * ds.headDim
            let dsQ = ds.qHeads * ds.headDim
            let perL = (dsQ + 4 * dsKV + dsQ) * ds.hiddenSize + 3 * ds.hiddenSize * ds.intermediateSize
            total += (ds.numLayers * perL + ds.hiddenSize * d.vocab) * 2
        }
        return total
    }

    private static func effectiveGQAKeyTokens(descriptor d: QWModelDescriptor,
                                              contextTokens: Int,
                                              prefillChunk: Int) -> Int {
        d.executionPlan.reduce(0) { acc, l in
            guard d.isFullAttention(l) else { return acc }
            let w = d.layerSlidingWindow(l)
            return acc + (w > 0 ? min(contextTokens, w + prefillChunk + 256) : contextTokens)
        }
    }

    /// Per-sequence state bytes: KV cache, GDN recurrent + conv state, and scratch.
    public static func perSequenceStateBytes(descriptor d: QWModelDescriptor,
                                             kvDtype: QWAdmissionDtype,
                                             contextTokens: Int,
                                             scratchPerSeq: Int,
                                             prefillChunk: Int = 4096)
        -> (kv: Int, gdn: Int, scratch: Int) {
        let execPlan = d.executionPlan
        let attnLayers = execPlan.filter { d.isFullAttention($0) }.count
        let hcaLayers = execPlan.filter { d.isCompressedAttention($0) }.count
        let kv: Int
        if hcaLayers > 0 || d.hybridMLAConfig != nil || (d.mlaConfig != nil && !d.mlaUseNoPE) {
            var totalKV = 0
            if let hca = d.hcaConfig, hcaLayers > 0 {
                let ratio = max(1, hca.defaultCompressRatio)
                let compTokens = (contextTokens + ratio - 1) / ratio
                let ringTokens = min(contextTokens, hca.slidingWindow)
                totalKV += hcaLayers * (compTokens + ringTokens) * hca.kvHeads * hca.headDim * 2
            }
            if let mla = d.mlaConfig ?? d.hybridMLAConfig.map({
                QWModelDescriptor.MLAConfig(qHeads: $0.qHeads, qProjection: $0.qLoraRank > 0 ? .factored(rank: $0.qLoraRank) : .direct, kvLoraRank: $0.kvLoraRank, qkNopeHeadDim: $0.qkNopeHeadDim, qkRopeHeadDim: $0.qkRopeHeadDim, vHeadDim: $0.vHeadDim)
            }), attnLayers > 0 {
                let ckvBytes = kvDtype.kvBytes(elements: attnLayers * contextTokens * mla.kvLoraRank)
                let ropeBytes = attnLayers * contextTokens * mla.qkRopeHeadDim * 2
                let idxComp = max(1, d.indexerCompressRatio ?? 1)
                let idxDim = d.indexerHeadDim ?? 0
                let idxBytes = idxDim > 0 ? attnLayers * ((contextTokens + idxComp - 1) / idxComp) * idxDim * 2 : 0
                totalKV += ckvBytes + ropeBytes + idxBytes
            }
            kv = totalKV
        } else {
            let kvElems = d.mlaUseNoPE
                ? attnLayers * contextTokens * ((d.kvLoraRank ?? 512) + (d.qkRopeHeadDim ?? 64))
                : 2 * effectiveGQAKeyTokens(descriptor: d, contextTokens: contextTokens, prefillChunk: prefillChunk)
                    * d.kvHeads * d.headDim
            kv = kvDtype.kvBytes(elements: kvElems)
        }
        let gdnLayers = execPlan.filter { d.isLinearAttention($0) }.count
        let rec = d.gdnVHeads * d.gdnHeadDim * d.gdnHeadDim
        let ring = max(0, d.gdnConvKernel - 1) * d.gdnQKVChannels
        var gdn = gdnLayers * (rec * 4 + ring * 2)
        if case .recurrentLoop(_, let ink) = d.executionTopology, let ink {
            let sconvSlots = max(0, ink.sconvKernelSize - 1)
            let convElems = 2 * d.hidden + 2 * d.kvHeads * d.headDim
            gdn += execPlan.count * sconvSlots * convElems * 4
        }
        return (kv, gdn, scratchPerSeq)
    }

    /// Legacy byte-width overload retained for callers passing explicit `kvBytesPerParam`.
    public static func perSequenceStateBytes(descriptor d: QWModelDescriptor,
                                             kvBytesPerParam: Int,
                                             contextTokens: Int,
                                             scratchPerSeq: Int,
                                             prefillChunk: Int = 4096)
        -> (kv: Int, gdn: Int, scratch: Int) {
        let dtype: QWAdmissionDtype = kvBytesPerParam >= 2 ? .f16 : .q8
        var res = perSequenceStateBytes(descriptor: d, kvDtype: dtype, contextTokens: contextTokens,
                                        scratchPerSeq: scratchPerSeq, prefillChunk: prefillChunk)
        if kvBytesPerParam > 0 && kvBytesPerParam != dtype.bytesPerParam {
            let effTokens = d.hcaConfig == nil && d.hybridMLAConfig == nil && d.mlaConfig == nil && !d.mlaUseNoPE
                ? effectiveGQAKeyTokens(descriptor: d, contextTokens: contextTokens, prefillChunk: prefillChunk)
                : d.executionPlan.filter { d.isFullAttention($0) }.count * contextTokens
            res.kv = 2 * effTokens * d.kvHeads * d.headDim * kvBytesPerParam
        }
        return res
    }
}

public enum QWAdmissionVerdict: Equatable, Sendable {
    case admitted(headroomBytes: Int)
    case refused(shortByBytes: Int)
}

/// Physical RAM query via `sysctlbyname("hw.memsize")` and live reclaimable RAM via `host_statistics64`.
public enum QWAdmission {
    public static func physicalMemoryBytes() -> Int? {
        #if canImport(Darwin)
        var size: Int64 = 0
        var len = MemoryLayout<Int64>.size
        guard sysctlbyname("hw.memsize", &size, &len, nil, 0) == 0 else { return nil }
        return Int(size)
        #else
        return nil
        #endif
    }

    /// Returns currently free + file-cache reclaimable + purgeable physical memory in bytes.
    /// Excludes wired memory, VM compressor pages, and active anonymous allocations held by running processes.
    public static func reclaimableMemoryBytes() -> Int? {
        #if canImport(Darwin)
        var vmStat = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &vmStat) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return nil }
        let pageSize = UInt64(sysconf(_SC_PAGESIZE))
        let reclaimablePages = UInt64(vmStat.free_count)
            + UInt64(vmStat.external_page_count)
            + UInt64(vmStat.purgeable_count)
        return Int(reclaimablePages * pageSize)
        #else
        return nil
        #endif
    }
}
