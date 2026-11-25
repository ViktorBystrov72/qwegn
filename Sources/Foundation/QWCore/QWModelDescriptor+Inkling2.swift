// QWModelDescriptor+Inkling2.swift — Inkling-2 looped-MoE geometry & mixed-FP quant metadata.
import CoreFoundation
import Foundation

extension QWModelDescriptor {
    /// Per-layer expert quantization mode parsed from HF `quantization` JSON.
    public struct Inkling2ExpertQuantSpec: Sendable, Hashable, Codable {
        public enum Mode: String, Sendable, Hashable, Codable {
            case nvfp4
            case mxfp8
            case bf16
        }

        public struct ProjSpec: Sendable, Hashable, Codable {
            public let mode: Mode
            public let bits: Int
            public let groupSize: Int

            public init(mode: Mode, bits: Int, groupSize: Int) {
                self.mode = mode
                self.bits = bits
                self.groupSize = groupSize
            }
        }

        public let gateProj: ProjSpec
        public let upProj: ProjSpec
        public let downProj: ProjSpec

        public init(gateProj: ProjSpec, upProj: ProjSpec, downProj: ProjSpec) {
            self.gateProj = gateProj
            self.upProj = upProj
            self.downProj = downProj
        }

        public func spec(for projection: String) -> ProjSpec {
            switch projection {
            case "up_proj": return upProj
            case "down_proj": return downProj
            default: return gateProj
            }
        }
    }

    /// Config for Inkling-2 DSpark (`mtp.safetensors`) speculative drafter.
    public struct Inkling2DSparkConfig: Sendable, Hashable, Codable {
        public let hiddenSize: Int
        public let numLayers: Int
        public let qHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let intermediateSize: Int
        public let maxPositionEmbeddings: Int
        public let rmsEps: Float
        public let ropeTheta: Double
        public let qBias: Bool
        public let oBias: Bool
        public let localLayerIds: [Int]
        public let slidingWindowSize: Int
        public let blockSize: Int
        public let markovRank: Int
        public let markovHeadType: String
        public let targetLayerIds: [Int]

        public init(hiddenSize: Int, numLayers: Int, qHeads: Int, kvHeads: Int,
                    headDim: Int, intermediateSize: Int, maxPositionEmbeddings: Int,
                    rmsEps: Float, ropeTheta: Double, qBias: Bool, oBias: Bool,
                    localLayerIds: [Int], slidingWindowSize: Int, blockSize: Int,
                    markovRank: Int, markovHeadType: String, targetLayerIds: [Int]) {
            self.hiddenSize = hiddenSize
            self.numLayers = numLayers
            self.qHeads = qHeads
            self.kvHeads = kvHeads
            self.headDim = headDim
            self.intermediateSize = intermediateSize
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.rmsEps = rmsEps
            self.ropeTheta = ropeTheta
            self.qBias = qBias
            self.oBias = oBias
            self.localLayerIds = localLayerIds
            self.slidingWindowSize = slidingWindowSize
            self.blockSize = blockSize
            self.markovRank = markovRank
            self.markovHeadType = markovHeadType
            self.targetLayerIds = targetLayerIds
        }
    }

    /// Config for Inkling-2 vision tower (`inkling2_vision`).
    public struct Inkling2VisionConfig: Sendable, Hashable, Codable {
        public let textHiddenSize: Int
        public let temporalPatchSize: Int
        public let nativePatchSize: Int
        public let adapterMergeSize: Int
        public let projectionHiddenDim: Int
        public let projectorBias: Bool
        public let hiddenSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let mlpRatio: Double
        public let posembGridSize: Int
        public let layerNormEps: Double

        public init(textHiddenSize: Int, temporalPatchSize: Int, nativePatchSize: Int,
                    adapterMergeSize: Int, projectionHiddenDim: Int, projectorBias: Bool,
                    hiddenSize: Int, numHiddenLayers: Int, numAttentionHeads: Int,
                    mlpRatio: Double, posembGridSize: Int, layerNormEps: Double) {
            self.textHiddenSize = textHiddenSize
            self.temporalPatchSize = temporalPatchSize
            self.nativePatchSize = nativePatchSize
            self.adapterMergeSize = adapterMergeSize
            self.projectionHiddenDim = projectionHiddenDim
            self.projectorBias = projectorBias
            self.hiddenSize = hiddenSize
            self.numHiddenLayers = numHiddenLayers
            self.numAttentionHeads = numAttentionHeads
            self.mlpRatio = mlpRatio
            self.posembGridSize = posembGridSize
            self.layerNormEps = layerNormEps
        }
    }

    /// Full config-derived architecture specification for `inkling2_loop`.
    public struct Inkling2Config: Sendable, Hashable, Codable {
        public let unpaddedVocabSize: Int
        public let dRel: Int
        public let relExtent: Int
        public let qBias: Bool
        public let oBias: Bool
        public let logScalingNFloor: Int
        public let logScalingAlpha: Float
        public let useValueNormNoGamma: Bool
        public let qkL2ClipThreshold: Float
        public let relL2ClipThreshold: Float
        public let localLayerIds: Set<Int>
        public let sconvKernelSize: Int
        public let finalLogitSoftcapping: Float
        public let swaHeadDim: Int
        public let swaQHeads: Int
        public let swaKVHeads: Int
        public let slidingWindowSize: Int
        public let layersPerAttnResBlock: Int
        public let attnResNormalizeValues: Bool
        public let mlpGateSoftcap: Float
        public let mlpUpSoftcap: Float
        public let routeScale: Float
        public let useGateBias: Bool
        public let loopCoreStart: Int
        public let loopCoreEnd: Int
        public let loopNumPasses: Int
        public let loopGateRank: Int
        public let isVarloop: Bool
        public let defaultQuantMode: Inkling2ExpertQuantSpec.Mode
        public let defaultQuantBits: Int
        public let defaultQuantGroupSize: Int
        public let perLayerExpertQuant: [Int: Inkling2ExpertQuantSpec]
        public let dspark: Inkling2DSparkConfig?
        public let vision: Inkling2VisionConfig?
        public let padTokenId: Int

        public init(unpaddedVocabSize: Int, dRel: Int, relExtent: Int,
                    qBias: Bool, oBias: Bool, logScalingNFloor: Int, logScalingAlpha: Float,
                    useValueNormNoGamma: Bool, qkL2ClipThreshold: Float, relL2ClipThreshold: Float,
                    localLayerIds: Set<Int>, sconvKernelSize: Int, finalLogitSoftcapping: Float,
                    swaHeadDim: Int, swaQHeads: Int, swaKVHeads: Int, slidingWindowSize: Int,
                    layersPerAttnResBlock: Int, attnResNormalizeValues: Bool,
                    mlpGateSoftcap: Float, mlpUpSoftcap: Float, routeScale: Float, useGateBias: Bool,
                    loopCoreStart: Int, loopCoreEnd: Int, loopNumPasses: Int,
                    loopGateRank: Int, isVarloop: Bool,
                    defaultQuantMode: Inkling2ExpertQuantSpec.Mode,
                    defaultQuantBits: Int, defaultQuantGroupSize: Int,
                    perLayerExpertQuant: [Int: Inkling2ExpertQuantSpec],
                    dspark: Inkling2DSparkConfig?, vision: Inkling2VisionConfig?,
                    padTokenId: Int) {
            self.unpaddedVocabSize = unpaddedVocabSize
            self.dRel = dRel
            self.relExtent = relExtent
            self.qBias = qBias
            self.oBias = oBias
            self.logScalingNFloor = logScalingNFloor
            self.logScalingAlpha = logScalingAlpha
            self.useValueNormNoGamma = useValueNormNoGamma
            self.qkL2ClipThreshold = qkL2ClipThreshold
            self.relL2ClipThreshold = relL2ClipThreshold
            self.localLayerIds = localLayerIds
            self.sconvKernelSize = sconvKernelSize
            self.finalLogitSoftcapping = finalLogitSoftcapping
            self.swaHeadDim = swaHeadDim
            self.swaQHeads = swaQHeads
            self.swaKVHeads = swaKVHeads
            self.slidingWindowSize = slidingWindowSize
            self.layersPerAttnResBlock = layersPerAttnResBlock
            self.attnResNormalizeValues = attnResNormalizeValues
            self.mlpGateSoftcap = mlpGateSoftcap
            self.mlpUpSoftcap = mlpUpSoftcap
            self.routeScale = routeScale
            self.useGateBias = useGateBias
            self.loopCoreStart = loopCoreStart
            self.loopCoreEnd = loopCoreEnd
            self.loopNumPasses = loopNumPasses
            self.loopGateRank = loopGateRank
            self.isVarloop = isVarloop
            self.defaultQuantMode = defaultQuantMode
            self.defaultQuantBits = defaultQuantBits
            self.defaultQuantGroupSize = defaultQuantGroupSize
            self.perLayerExpertQuant = perLayerExpertQuant
            self.dspark = dspark
            self.vision = vision
            self.padTokenId = padTokenId
        }

        public func isSlidingLayer(_ layer: Int) -> Bool {
            localLayerIds.contains(layer)
        }

        public func isAttnResBlockBoundary(_ layer: Int) -> Bool {
            layersPerAttnResBlock > 0 && layer % layersPerAttnResBlock == 0
        }

        public func expertProjSpec(layer: Int, projection: String) -> Inkling2ExpertQuantSpec.ProjSpec {
            if let layerSpec = perLayerExpertQuant[layer] {
                return layerSpec.spec(for: projection)
            }
            return Inkling2ExpertQuantSpec.ProjSpec(
                mode: defaultQuantMode, bits: defaultQuantBits, groupSize: defaultQuantGroupSize
            )
        }

        public func expertModesCount(numLayers: Int) -> (nvfp4: Int, mxfp8: Int, bf16: Int) {
            var nv = 0, mx = 0, bf = 0
            for l in 0..<numLayers {
                switch expertProjSpec(layer: l, projection: "gate_proj").mode {
                case .nvfp4: nv += 1
                case .mxfp8: mx += 1
                case .bf16: bf += 1
                }
            }
            return (nv, mx, bf)
        }
    }

    internal static func dictInt(_ d: [String: Any]?, _ k: String) -> Int? {
        guard let n = d?[k] as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else { return nil }
        return n.intValue
    }

    internal static func dictDouble(_ d: [String: Any]?, _ k: String) -> Double? {
        guard let n = d?[k] as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else { return nil }
        return n.doubleValue
    }

    internal static func dictFloat(_ d: [String: Any]?, _ k: String) -> Float? {
        dictDouble(d, k).map(Float.init)
    }

    internal static func dictBool(_ d: [String: Any]?, _ k: String) -> Bool? {
        guard let n = d?[k] as? NSNumber, CFGetTypeID(n) == CFBooleanGetTypeID() else { return nil }
        return n.boolValue
    }

    internal static func decodeInkling2Config(
        text t: ConfigEnvelope.Text,
        env: ConfigEnvelope,
        rawRoot: [String: Any]?,
        rawText: [String: Any],
        inv: (Bool, String) throws -> Void
    ) throws -> Inkling2Config {
        let localArr = (rawText["local_layer_ids"] as? [Int]) ?? []
        let localSet = Set(localArr)
        for idx in localSet {
            try inv(idx >= 0 && idx < t.num_hidden_layers, "inkling2 local_layer_ids index \(idx) out of range")
        }
        let swSize = dictInt(rawText, "sliding_window_size") ?? t.sliding_window ?? 1024
        let dRel = dictInt(rawText, "d_rel") ?? 16
        let relExtent = dictInt(rawText, "rel_extent") ?? swSize
        try inv(dRel > 0 && relExtent > 0 && swSize > 0, "inkling2 rel/swa dims must be positive")
        let sconvK = dictInt(rawText, "sconv_kernel_size") ?? 4
        try inv(sconvK == 4, "inkling2 sconv_kernel_size \(sconvK) != 4")
        let loopCfg = rawText["loop_config"] as? [String: Any]
        let coreStart = dictInt(loopCfg, "core_start") ?? 6
        let coreEnd = dictInt(loopCfg, "core_end") ?? 18
        let numPasses = dictInt(loopCfg, "num_passes") ?? 3
        let gateRank = dictInt(loopCfg, "gate_rank") ?? 64
        let isVarloop = dictBool(loopCfg, "is_varloop") ?? false
        try inv(coreStart >= 0 && coreEnd > coreStart && coreEnd <= t.num_hidden_layers && numPasses >= 1,
                "inkling2 loop_config out of range")
        try inv(isVarloop == false, "inkling2 varloop is not supported")

        let qRoot = (rawRoot?["quantization"] as? [String: Any])
            ?? (rawRoot?["quantization_config"] as? [String: Any])
        let defModeStr = (qRoot?["mode"] as? String ?? "bf16").lowercased()
        let defMode = Inkling2ExpertQuantSpec.Mode(rawValue: defModeStr) ?? .bf16
        let defBits = dictInt(qRoot, "bits") ?? (defMode == .nvfp4 ? 4 : (defMode == .mxfp8 ? 8 : 16))
        let defGroup = dictInt(qRoot, "group_size") ?? (defMode == .nvfp4 ? 16 : (defMode == .mxfp8 ? 32 : 1))

        var layerProjs: [Int: [String: Inkling2ExpertQuantSpec.ProjSpec]] = [:]
        if let qRoot {
            for (key, val) in qRoot {
                guard let dict = val as? [String: Any],
                      let layersRange = key.range(of: "layers."),
                      let modeStr = (dict["mode"] as? String)?.lowercased(),
                      let mode = Inkling2ExpertQuantSpec.Mode(rawValue: modeStr) else {
                    continue
                }
                let afterLayers = key[layersRange.upperBound...]
                guard let dot = afterLayers.firstIndex(of: "."),
                      let layerIdx = Int(afterLayers[..<dot]) else {
                    continue
                }
                let projName = String(key.split(separator: ".").last ?? "")
                let bits = dictInt(dict, "bits") ?? defBits
                let grp = dictInt(dict, "group_size") ?? defGroup
                layerProjs[layerIdx, default: [:]][projName] = Inkling2ExpertQuantSpec.ProjSpec(
                    mode: mode, bits: bits, groupSize: grp
                )
            }
        }
        var perLayerQuant: [Int: Inkling2ExpertQuantSpec] = [:]
        let defProj = Inkling2ExpertQuantSpec.ProjSpec(mode: defMode, bits: defBits, groupSize: defGroup)
        for (layerIdx, map) in layerProjs {
            perLayerQuant[layerIdx] = Inkling2ExpertQuantSpec(
                gateProj: map["gate_proj"] ?? defProj,
                upProj: map["up_proj"] ?? defProj,
                downProj: map["down_proj"] ?? defProj
            )
        }

        let mtpDict = rawRoot?["mtp_config"] as? [String: Any]
        let dspark: Inkling2DSparkConfig? = mtpDict.map { mc in
            Inkling2DSparkConfig(
                hiddenSize: dictInt(mc, "hidden_size") ?? t.hidden_size,
                numLayers: dictInt(mc, "num_hidden_layers") ?? 3,
                qHeads: dictInt(mc, "num_attention_heads") ?? t.num_attention_heads,
                kvHeads: dictInt(mc, "num_key_value_heads") ?? (t.num_key_value_heads ?? t.num_attention_heads),
                headDim: dictInt(mc, "head_dim") ?? (t.head_dim ?? 128),
                intermediateSize: dictInt(mc, "intermediate_size") ?? (t.hidden_size * 4),
                maxPositionEmbeddings: dictInt(mc, "max_position_embeddings") ?? 1048576,
                rmsEps: dictFloat(mc, "rms_norm_eps") ?? Float(t.rms_norm_eps ?? t.norm_eps ?? 1e-6),
                ropeTheta: dictDouble(mc, "rope_theta") ?? 1000000.0,
                qBias: dictBool(mc, "q_bias") ?? false,
                oBias: dictBool(mc, "o_bias") ?? false,
                localLayerIds: (mc["local_layer_ids"] as? [Int]) ?? [0, 2],
                slidingWindowSize: dictInt(mc, "sliding_window_size") ?? 1024,
                blockSize: dictInt(mc, "block_size") ?? 8,
                markovRank: dictInt(mc, "markov_rank") ?? 256,
                markovHeadType: (mc["markov_head_type"] as? String) ?? "rnn_hidden",
                targetLayerIds: (mc["target_layer_ids"] as? [Int]) ?? [5, 23, 41, 48]
            )
        }

        let rawVis = rawRoot?["vision_config"] as? [String: Any]
        let visCfg: Inkling2VisionConfig? = env.vision_config.map { vc in
            Inkling2VisionConfig(
                textHiddenSize: dictInt(rawVis, "text_hidden_size") ?? t.hidden_size,
                temporalPatchSize: vc.temporal_patch_size ?? 2,
                nativePatchSize: dictInt(rawVis, "native_patch_size") ?? vc.patch_size ?? 20,
                adapterMergeSize: dictInt(rawVis, "adapter_merge_size") ?? vc.spatial_merge_size ?? 2,
                projectionHiddenDim: dictInt(rawVis, "projection_hidden_dim") ?? 4096,
                projectorBias: dictBool(rawVis, "projector_bias") ?? false,
                hiddenSize: vc.hidden_size ?? 1024,
                numHiddenLayers: dictInt(rawVis, "num_hidden_layers") ?? vc.depth ?? 23,
                numAttentionHeads: dictInt(rawVis, "num_attention_heads") ?? vc.num_heads ?? 16,
                mlpRatio: dictDouble(rawVis, "mlp_ratio") ?? 4.0,
                posembGridSize: dictInt(rawVis, "posemb_grid_size") ?? 32,
                layerNormEps: dictDouble(rawVis, "layer_norm_eps") ?? 1e-5
            )
        }

        let padTok = env.generation_config?.pad_token_id
            ?? dictInt(rawRoot, "pad_token_id")
            ?? dictInt(rawText, "pad_token_id")
            ?? 199999

        return Inkling2Config(
            unpaddedVocabSize: dictInt(rawText, "unpadded_vocab_size") ?? t.vocab_size,
            dRel: dRel,
            relExtent: relExtent,
            qBias: dictBool(rawText, "q_bias") ?? false,
            oBias: dictBool(rawText, "o_bias") ?? false,
            logScalingNFloor: dictInt(rawText, "log_scaling_n_floor") ?? 8192,
            logScalingAlpha: dictFloat(rawText, "log_scaling_alpha") ?? 0.1,
            useValueNormNoGamma: dictBool(rawText, "use_value_norm_no_gamma") ?? true,
            qkL2ClipThreshold: dictFloat(rawText, "qk_l2_clip_threshold") ?? 22.627416997969522,
            relL2ClipThreshold: dictFloat(rawText, "rel_l2_clip_threshold") ?? 8.0,
            localLayerIds: localSet,
            sconvKernelSize: sconvK,
            finalLogitSoftcapping: dictFloat(rawText, "final_logit_softcapping") ?? 30.0,
            swaHeadDim: dictInt(rawText, "swa_head_dim") ?? (t.head_dim ?? 128),
            swaQHeads: dictInt(rawText, "swa_num_attention_heads") ?? t.num_attention_heads,
            swaKVHeads: dictInt(rawText, "swa_num_key_value_heads") ?? (t.num_key_value_heads ?? t.num_attention_heads),
            slidingWindowSize: swSize,
            layersPerAttnResBlock: dictInt(rawText, "n_layers_per_attn_res_block") ?? 6,
            attnResNormalizeValues: dictBool(rawText, "attn_res_normalize_values") ?? true,
            mlpGateSoftcap: dictFloat(rawText, "mlp_gate_softcap") ?? 4.0,
            mlpUpSoftcap: dictFloat(rawText, "mlp_up_softcap") ?? 25.0,
            routeScale: dictFloat(rawText, "route_scale") ?? 8.0,
            useGateBias: dictBool(rawText, "use_gate_bias") ?? true,
            loopCoreStart: coreStart,
            loopCoreEnd: coreEnd,
            loopNumPasses: numPasses,
            loopGateRank: gateRank,
            isVarloop: isVarloop,
            defaultQuantMode: defMode,
            defaultQuantBits: defBits,
            defaultQuantGroupSize: defGroup,
            perLayerExpertQuant: perLayerQuant,
            dspark: dspark,
            vision: visCfg,
            padTokenId: padTok
        )
    }

    internal static func inkling2LoopSettings(rawText: [String: Any], numLayers: Int) -> LoopSettings? {
        guard let lc = rawText["loop_config"] as? [String: Any],
              let s = dictInt(lc, "core_start"), let e = dictInt(lc, "core_end"),
              s >= 0, e > s, e <= numLayers else {
            return nil
        }
        let passes = max(1, dictInt(lc, "num_passes") ?? 1)
        return LoopSettings(start: s, end: e, evalCount: passes, counts: [passes])
    }

    internal static func resolveInkling2VisionDims(
        vc: ConfigEnvelope.VisionConfig?,
        rawVision: [String: Any]?,
        hidden: Int
    ) throws -> (depth: Int, hidden: Int, heads: Int, inter: Int, patch: Int, merge: Int, pos: Int, out: Int) {
        guard let vc else { return (0, 0, 0, 0, 0, 0, 0, 0) }
        let patch = dictInt(rawVision, "native_patch_size") ?? vc.patch_size ?? 16
        let pos: Int
        if let explicit = vc.num_position_embeddings {
            pos = explicit
        } else if let grid = dictInt(rawVision, "posemb_grid_size"), grid > 0 {
            pos = grid * grid
        } else if let img = vc.image_size, patch > 0 {
            let side = max(1, img / patch)
            pos = side * side
        } else {
            throw DescriptorError.invariant("vision_config missing num_position_embeddings and image_size")
        }
        let vh = vc.hidden_size ?? 0
        let mlpRatio = dictDouble(rawVision, "mlp_ratio") ?? 4.0
        let inter = vc.intermediate_size ?? (vh > 0 ? Int(Double(vh) * mlpRatio) : 0)
        return (
            depth: dictInt(rawVision, "num_hidden_layers") ?? vc.depth ?? 0,
            hidden: vh,
            heads: dictInt(rawVision, "num_attention_heads") ?? vc.num_heads ?? 0,
            inter: inter,
            patch: patch,
            merge: dictInt(rawVision, "adapter_merge_size") ?? vc.spatial_merge_size ?? 2,
            pos: pos,
            out: dictInt(rawVision, "text_hidden_size") ?? vc.out_hidden_size ?? hidden
        )
    }

    internal static func resolveVisionDims(
        vc: ConfigEnvelope.VisionConfig?,
        rawVision: [String: Any]?,
        hidden: Int,
        isInkling2: Bool
    ) throws -> (depth: Int, hidden: Int, heads: Int, inter: Int, patch: Int, temporal: Int, merge: Int, pos: Int, out: Int) {
        guard let vc else { return (0, 0, 0, 0, 0, 0, 0, 0, 0) }
        if isInkling2 || dictInt(rawVision, "native_patch_size") != nil {
            let v = try resolveInkling2VisionDims(vc: vc, rawVision: rawVision, hidden: hidden)
            return (v.depth, v.hidden, v.heads, v.inter, v.patch, vc.temporal_patch_size ?? 2, v.merge, v.pos, v.out)
        }
        let patch = vc.patch_size ?? 16
        let pos: Int
        if let explicit = vc.num_position_embeddings ?? vc.num_patches {
            pos = explicit
        } else if let img = vc.image_size, patch > 0 {
            let side = max(1, img / patch)
            pos = side * side
        } else if (rawVision?["rope_parameters"] as? [String: Any]) != nil || vc.model_type == "lfm3_vl_vision" {
            // 2D Axial RoPE vision towers (e.g., LFM3-VL) do not use learned absolute pos embeddings.
            pos = 0
        } else {
            throw DescriptorError.invariant("vision_config missing num_position_embeddings and image_size")
        }
        let defaultMerge = (vc.model_type == "siglip2_vision_model" || vc.model_type == "siglip_vision_model") ? 1 : 2
        let merge = vc.spatial_merge_size ?? vc.downsample_factor ?? defaultMerge
        let temporalDefault = (vc.model_type == "lfm3_vl_vision" || vc.model_type == "lfm2_vl_vision"
                               || vc.model_type == "siglip2_vision_model" || vc.model_type == "siglip_vision_model") ? 1 : 2
        return (vc.depth ?? vc.num_hidden_layers ?? 0,
                vc.hidden_size ?? 0,
                vc.num_heads ?? vc.num_attention_heads ?? 0,
                vc.intermediate_size ?? 0,
                patch, vc.temporal_patch_size ?? temporalDefault, merge, pos,
                vc.out_dim ?? vc.out_hidden_size ?? hidden)
    }

    internal static func appendInkling2DigestFields(_ d: QWModelDescriptor, into f: inout [String]) {
        guard let ink = d.inkling2Config else { return }
        let m = ink.expertModesCount(numLayers: d.numLayers)
        f.append("ink2=\(ink.unpaddedVocabSize),\(ink.dRel),\(ink.relExtent),\(ink.sconvKernelSize),\(ink.layersPerAttnResBlock),\(ink.loopGateRank),\(m.nvfp4),\(m.mxfp8),\(m.bf16),\(ink.dspark?.blockSize ?? 0)")
    }

    internal func inkling2SpecialTokens(stops: [Int]) -> QWModelSpecialTokens {
        QWModelSpecialTokens(
            eosTokenId: eosTokenId,
            extraStopTokenIds: stops,
            usesChatTemplateIds: false
        )
    }
}
