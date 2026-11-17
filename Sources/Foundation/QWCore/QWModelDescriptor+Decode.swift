import Foundation

extension QWModelDescriptor {

    // MARK: - Fail-closed decode from validated config JSON

    public enum DescriptorError: Error, CustomStringConvertible, Equatable {
        case missingTextConfig
        case invariant(String)
        case drafterModel(String)
        public var description: String {
            switch self {
            case .missingTextConfig: return "QWCore: config JSON has neither text_config nor top-level geometry"
            case .invariant(let m): return "QWCore: descriptor invariant violated: \(m)"
            case .drafterModel(let m): return m
            }
        }
    }

    internal static func isDrafterJSON(_ obj: [String: Any]) -> Bool {
        if obj["dflash_config"] != nil || obj["selected_prefix_code_bridge"] != nil { return true }
        if let archs = obj["architectures"] as? [String],
           archs.contains(where: { $0.localizedCaseInsensitiveContains("DraftModel") || $0.localizedCaseInsensitiveContains("DFlash") }) {
            return true
        }
        if let mt = (obj["model_type"] as? String)?.lowercased(),
           mt.contains("dflash") || mt.contains("dspine") {
            return true
        }
        return false
    }

    /// Decode the pinned HF config.json into a descriptor, re-deriving every
    /// channel count and refusing on a violated identity or cadence.
    public static func decode(configJSON data: Data) throws -> QWModelDescriptor {
        if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           isDrafterJSON(obj) {
            throw DescriptorError.drafterModel("DFlash2 drafter model (cannot be loaded as a primary model)")
        }
        if QWStandardConfigDecoder.isStandardSchema(data) {
            return try decodeStandard(data)
        }
        if let feature = QWStandardConfigDecoder.standardFamilyRefusal(data) {
            throw QWStandardConfigDecoder.Error.unsupported(feature)
        }
        let env = try JSONDecoder().decode(ConfigEnvelope.self, from: data)
        let t: ConfigEnvelope.Text
        if let tc = env.text_config {
            t = tc
        } else {
            do {
                t = try JSONDecoder().decode(ConfigEnvelope.Text.self, from: data)
            } catch DecodingError.keyNotFound(let key, _) {
                throw DescriptorError.invariant("flat config missing \(key.stringValue)")
            } catch DecodingError.valueNotFound(_, let ctx) {
                let name = ctx.codingPath.last?.stringValue ?? "required field"
                throw DescriptorError.invariant("flat config missing \(name)")
            }
        }
        func inv(_ cond: Bool, _ msg: String) throws {
            if !cond { throw DescriptorError.invariant(msg) }
        }
        let rawLayerTypes: [String]
        if let explicitTypes = t.layer_types {
            rawLayerTypes = explicitTypes
        } else if let ratios = t.compress_ratios {
            try inv(ratios.count >= t.num_hidden_layers,
                    "compress_ratios count \(ratios.count) < num_hidden_layers \(t.num_hidden_layers)")
            rawLayerTypes = (0..<t.num_hidden_layers).map { l in
                ratios[l] > 0 ? "compressed_attention" : "full_attention"
            }
        } else if t.attention_variant == "block_sparse_gqa" || t.model_type == "maxwell_moe" {
            rawLayerTypes = Array(repeating: "full_attention", count: t.num_hidden_layers)
        } else {
            throw DescriptorError.invariant("flat config missing layer_types")
        }
        let normalizedLayerTypes = rawLayerTypes.map { kind -> String in
            kind == "deepseek_sparse_attention" ? "full_attention" : kind
        }
        try inv(normalizedLayerTypes.count == t.num_hidden_layers,
                "layer_types count \(normalizedLayerTypes.count) != num_hidden_layers \(t.num_hidden_layers)")
        let kindsOK = normalizedLayerTypes.allSatisfy {
            $0 == "full_attention" || $0 == "linear_attention" || $0 == "compressed_attention"
        }
        try inv(kindsOK, "layer_types contain unknown kind")

        let resolvedFAInterval: Int
        if let explicitInterval = t.full_attention_interval {
            resolvedFAInterval = explicitInterval
        } else if let firstFA = normalizedLayerTypes.firstIndex(of: "full_attention") {
            let candidate = firstFA + 1
            let fullIndices = normalizedLayerTypes.indices.filter { normalizedLayerTypes[$0] == "full_attention" }
            let uniformCadence = fullIndices.allSatisfy { ($0 + 1) % candidate == 0 }
            resolvedFAInterval = uniformCadence ? candidate : 0
        } else {
            resolvedFAInterval = 1
        }

        let resolvedHeadDim: Int
        if let hd = t.head_dim ?? t.attention_head_size, hd > 0 {
            resolvedHeadDim = hd
        } else if let qkHd = t.qk_head_dim, qkHd > 0 {
            resolvedHeadDim = qkHd
        } else if let nope = t.qk_nope_head_dim, let rope = t.qk_rope_head_dim, nope + rope > 0 {
            resolvedHeadDim = nope + rope
        } else {
            throw DescriptorError.invariant("flat config missing head_dim")
        }

        let hasHybridMLA = t.hybrid_mla_kv_lora_rank != nil || t.hybrid_mla_attention != nil
        let isNoPEMLA = t.mla_use_nope ?? (t.kv_lora_rank != nil && t.use_mla != false && t.compress_ratios == nil && !hasHybridMLA)
        let resolvedPartialRotary: Double
        if let prf = t.partial_rotary_factor ?? t.rope_parameters?.partial_rotary_factor {
            resolvedPartialRotary = prf
        } else if let rp = t.rotary_percent {
            resolvedPartialRotary = rp
        } else if isNoPEMLA || hasHybridMLA || t.compress_ratios != nil
                    || t.attention_variant == "block_sparse_gqa" || t.model_type == "maxwell_moe" {
            resolvedPartialRotary = 1.0
        } else {
            throw DescriptorError.invariant("flat config missing partial_rotary_factor")
        }

        let ropeTheta = t.rope_theta ?? t.rope_parameters?.rope_theta ?? 10000.0
        let mropeSec: (w: Int, h: Int, t: Int)
        let hasMrope: Bool
        if let sec = t.rope_parameters?.mrope_section {
            try inv(sec.count == 3, "mrope_section must have 3 elements")
            mropeSec = (sec[0], sec[1], sec[2])
            hasMrope = true
        } else {
            mropeSec = (11, 11, 10)
            let rotHalf = Int(Double(resolvedHeadDim) * resolvedPartialRotary) / 2
            hasMrope = (rotHalf == 32)
        }

        let resolvedNumExperts = t.num_experts ?? t.n_routed_experts
        if resolvedNumExperts == nil && resolvedFAInterval > 0 {
            for l in 0..<t.num_hidden_layers {
                let want = (l + 1) % resolvedFAInterval == 0
                try inv((normalizedLayerTypes[l] == "full_attention") == want, "layer \(l) cadence")
            }
        }
        let hasLinearLayers = normalizedLayerTypes.contains("linear_attention")
        let linCfg = t.linear_attn_config
        let qkHeads = hasLinearLayers ? (t.linear_num_key_heads ?? linCfg?.num_heads ?? 0) : 0
        let vHeads = hasLinearLayers ? (t.linear_num_value_heads ?? linCfg?.num_heads ?? 0) : 0
        let gHeadDim = hasLinearLayers ? (t.linear_key_head_dim ?? linCfg?.head_dim ?? 0) : 0
        let gConv = hasLinearLayers ? (t.linear_conv_kernel_dim ?? linCfg?.short_conv_kernel_size ?? 0) : 0
        if hasLinearLayers {
            try inv(qkHeads > 0 && vHeads > 0 && gHeadDim > 0, "GDN heads/dim missing")
            try inv((t.linear_value_head_dim ?? gHeadDim) == gHeadDim, "GDN v-head dim != qk head dim")
        }
        let qkv = (2 * qkHeads + vHeads) * gHeadDim
        let z = vHeads * gHeadDim

        let linType = (linCfg?.Linear_type ?? linCfg?.linear_type ?? "").lowercased()
        let isKDA = hasLinearLayers && (linType == "kda" || linCfg?.gate_lower_bound != nil)
        let gdnLowRank = linCfg?.low_rank_dim ?? (isKDA ? gHeadDim : nil)

        if let n = resolvedNumExperts, let k = t.num_experts_per_tok {
            try inv(n > 0 && k > 0 && k <= n, "MoE top-k \(k) > experts \(n)")
        }
        let resolvedIdxHeads = t.indexer_n_heads ?? t.index_n_heads
        let resolvedIdxHeadDim = t.indexer_head_dim ?? t.index_head_dim
        let resolvedIdxBudget = t.indexer_budget ?? t.index_topk
        let resolvedIdxKvHeads = t.indexer_kv_heads ?? (resolvedIdxHeads != nil ? 1 : nil)
        let resolvedIdxCompress = t.indexer_compress_ratio
            ?? t.index_kpool
            ?? (t.index_kpool_compress == true ? 4 : (resolvedIdxHeads != nil ? 1 : nil))
        if let idx = resolvedIdxHeads {
            try inv(resolvedIdxKvHeads != nil && resolvedIdxHeadDim != nil
                    && resolvedIdxBudget != nil && resolvedIdxCompress != nil,
                    "indexer partially specified")
            _ = idx
        }
        if let hc = t.hc_count { try inv(hc >= 2, "hc_count \(hc) < 2") }
        if let pleIds = t.ple_layer_ids, !pleIds.isEmpty {
            try inv((t.ngram_size ?? 0) >= 2, "PLE without ngram_size")
            let heads = (t.ngram_size ?? 0) > 0 ? ((t.ngram_size ?? 0) - 1) * (t.heads_per_ngram ?? 0) : -1
            try inv(heads > 0 && (t.ple_embed_dim ?? 0) % heads == 0,
                    "ple_embed_dim \(t.ple_embed_dim ?? -1) % heads \(heads) != 0")
            for id in pleIds {
                try inv(id >= 1 && id <= t.num_hidden_layers, "ple_layer_ids one-indexed violation: \(id)")
                try inv(normalizedLayerTypes[id - 1] == "linear_attention",
                        "PLE id \(id) must land on linear_attention (got \(normalizedLayerTypes[id-1]))")
            }
        }
        let eos = t.eos_token_id.value
        let eosList = t.eos_token_id.values

        let resolvedSharedIntermediate: Int?
        if let explicitShared = t.shared_expert_intermediate_size {
            resolvedSharedIntermediate = explicitShared
        } else if let nShared = t.n_shared_experts, nShared > 0, let moeInt = t.moe_intermediate_size {
            resolvedSharedIntermediate = nShared * moeInt
        } else {
            resolvedSharedIntermediate = nil
        }

        var caps: Set<Capability> = []
        if env.vision_config != nil { caps.insert(.vision) }
        if resolvedNumExperts != nil { caps.insert(.moe) }
        if (t.ple_layer_ids?.isEmpty == false) { caps.insert(.ple) }
        if resolvedIdxHeads != nil { caps.insert(.qsaIndexer) }
        let rawMHCVal = t.mhc ?? env.mhc ?? (t.enable_hyper_connections == true ? .flag(true) : nil)
        let parsedMHC: MHCConfig?
        switch rawMHCVal {
        case .config(let rm):
            if let s = rm.start, let e = rm.end, s >= 0, e > s, e <= t.num_hidden_layers {
                parsedMHC = MHCConfig(start: s, end: e, streams: max(2, rm.streams ?? 2),
                                      iterations: max(1, rm.iterations ?? 20),
                                      fixedIdentity: rm.fixed_identity ?? false, preScale: rm.pre_scale ?? 1.0,
                                      hcEps: t.hc_eps, hasExitHead: t.enable_hyper_connections == true)
            } else {
                parsedMHC = nil
            }
        case .flag(true):
            parsedMHC = MHCConfig(start: 0, end: t.num_hidden_layers,
                                  streams: max(2, t.num_residual_streams ?? t.hc_mult ?? 4),
                                  iterations: max(1, t.mhc_sinkhorn_iterations ?? t.hc_sinkhorn_iters ?? 20),
                                  fixedIdentity: false, preScale: 1.0,
                                  hcEps: t.hc_eps ?? (t.model_type == "maxwell_moe" ? Float(t.rms_norm_eps) : nil),
                                  hasExitHead: t.enable_hyper_connections == true)
        case .flag(false), nil:
            parsedMHC = nil
        }
        if t.hc_count != nil || parsedMHC != nil { caps.insert(.hyperConnections) }
        let resolvedMtpLayers = max(0, t.mtp_num_hidden_layers ?? t.num_nextn_predict_layers ?? t.mtp_num_layers ?? 0)
        if resolvedMtpLayers > 0 { caps.insert(.mtp) }

        let parsedYarn: YarnRopeConfig? = {
            guard let rs = t.rope_scaling else { return nil }
            let kind = (rs.type ?? rs.rope_type ?? "").lowercased()
            guard kind == "yarn", let factor = rs.factor, let orig = rs.original_max_position_embeddings else {
                return nil
            }
            return YarnRopeConfig(factor: factor, originalMaxPositionEmbeddings: orig,
                                  betaFast: rs.beta_fast ?? 32.0, betaSlow: rs.beta_slow ?? 1.0,
                                  mscale: rs.mscale ?? 0.0, mscaleAllDim: rs.mscale_all_dim ?? 0.0)
        }()

        let parsedHCA: HCAConfig? = {
            guard normalizedLayerTypes.contains("compressed_attention"),
                  let ratios = t.compress_ratios else { return nil }
            let defRatio = ratios.first(where: { $0 > 0 }) ?? 128
            return HCAConfig(
                compressRatios: ratios,
                defaultCompressRatio: defRatio,
                qHeads: t.num_attention_heads,
                kvHeads: t.num_key_value_heads,
                headDim: resolvedHeadDim,
                qLoraRank: t.q_lora_rank ?? 1024,
                kvLoraRank: t.kv_lora_rank ?? 512,
                qkNopeHeadDim: t.qk_nope_head_dim ?? 448,
                qkRopeHeadDim: t.qk_rope_head_dim ?? 64,
                vHeadDim: t.v_head_dim ?? resolvedHeadDim,
                oGroups: t.o_groups ?? 8,
                oLoraRank: t.o_lora_rank ?? 1024,
                slidingWindow: t.sliding_window ?? 128,
                compressRopeTheta: t.compress_rope_theta ?? ropeTheta,
                yarn: (t.hca_rope_type == nil || t.hca_rope_type?.lowercased() == "yarn") ? parsedYarn : nil,
                useAttnMerger: t.use_attn_merger ?? true,
                useVHA: t.use_vha_attention ?? false,
                hasAttnSinkBias: t.add_full_attention_sink_bias ?? false,
                gatedAttnUseQLora: t.gated_attn_use_q_lora ?? false
            )
        }()

        let parsedHybridMLA: HybridMLAConfig? = {
            guard hasHybridMLA else { return nil }
            let nope = t.hybrid_mla_qk_nope_head_dim ?? t.qk_nope_head_dim ?? 192
            let rope = t.hybrid_mla_qk_rope_head_dim ?? t.qk_rope_head_dim ?? 64
            let mlaYarn: YarnRopeConfig? = (t.rope_type?.lowercased() == "yarn") ? parsedYarn : nil
            return HybridMLAConfig(
                qHeads: t.hybrid_mla_num_attention_heads ?? t.num_attention_heads,
                qLoraRank: t.hybrid_mla_q_lora_rank ?? t.q_lora_rank ?? 1024,
                kvLoraRank: t.hybrid_mla_kv_lora_rank ?? t.kv_lora_rank ?? 512,
                qkNopeHeadDim: nope,
                qkRopeHeadDim: rope,
                vHeadDim: t.hybrid_mla_v_head_dim ?? t.v_head_dim ?? 256,
                decoupledRope: !isNoPEMLA && rope > 0,
                ropeInterleave: t.rope_interleave ?? t.indexer_rope_interleave ?? false,
                slidingWindow: t.sliding_window ?? 0,
                hasAttnSinkBias: t.add_full_attention_sink_bias ?? false,
                useVHA: t.use_vha_attention ?? false,
                indexerUseFastHadamard: t.use_fast_hadamard ?? false,
                splitKvBProj: t.mqa_split_kv_b_proj ?? false,
                yarn: mlaYarn
            )
        }()

        let parsedAct: ActivationKind
        if let act = t.hidden_act?.lowercased(), act == "situ" {
            parsedAct = .situ(beta: t.activation_situ_beta ?? 4.0,
                              linearBeta: t.activation_situ_linear_beta ?? 25.0)
        } else {
            parsedAct = .silu
        }

        let rawLoop = t.loop_settings ?? env.loop_settings
        let parsedLoop: LoopSettings? = (rawLoop.flatMap { rl in
            guard let s = rl.start, let e = rl.end, s >= 0, e > s, e <= t.num_hidden_layers else { return nil }
            return LoopSettings(start: s, end: e, evalCount: max(1, rl.eval_count ?? rl.counts?.first ?? 1), counts: rl.counts)
        })

        let resolvedFirstKDenseReplace: Int
        if let fk = t.first_k_dense_replace {
            resolvedFirstKDenseReplace = max(0, fk)
        } else if let denseLayers = t.mlp_only_layers, !denseLayers.isEmpty {
            try inv(Set(denseLayers) == Set(0..<denseLayers.count),
                    "mlp_only_layers \(denseLayers) must be contiguous from 0")
            resolvedFirstKDenseReplace = denseLayers.count
        } else {
            resolvedFirstKDenseReplace = 0
        }

        let parsedBlockSparseGQA: BlockSparseGQAConfig?
        if t.attention_variant == "block_sparse_gqa" {
            guard let blk = t.block_sparse_gqa_block_size,
                  let pfxWin = t.block_sparse_gqa_prefix_window_size,
                  let locWin = t.block_sparse_gqa_local_window_size,
                  let remote = t.block_sparse_gqa_num_remote_blocks,
                  let idxDim = t.block_sparse_gqa_indexer_head_dim,
                  let idxHeads = t.block_sparse_gqa_indexer_n_heads,
                  blk > 0, pfxWin > 0, locWin > 0, remote > 0,
                  pfxWin % blk == 0, locWin % blk == 0,
                  idxHeads == t.num_key_value_heads, idxDim == resolvedHeadDim else {
                throw DescriptorError.invariant("block_sparse_gqa parameters missing or invalid")
            }
            parsedBlockSparseGQA = BlockSparseGQAConfig(
                blockSize: blk,
                prefixBlocks: pfxWin / blk,
                localBlocks: locWin / blk,
                remoteBlocks: remote,
                indexerHeadDim: idxDim
            )
        } else {
            parsedBlockSparseGQA = nil
        }

        guard let familySpec = QWModelFamilyRegistry.shared.spec(forModelType: t.model_type) else {
            throw QWModelDescriptorError.unsupportedModelType(t.model_type)
        }
        let family = familySpec.family
        let gateDeclared: Bool = t.output_gate_type != nil
        let attnGate: Bool = t.attn_output_gate ?? t.use_gated_attn ?? (gateDeclared || familySpec.defaultAttnOutputGate)
        let resolvedGateType = t.scoring_func ?? t.router_score_function ?? familySpec.defaultExpertGateType
        let vc = env.vision_config
        let resolvedVisPatch = vc?.patch_size ?? (vc != nil ? 16 : 0)
        let resolvedVisPosEmbeds: Int = {
            if let explicit = vc?.num_position_embeddings { return explicit }
            if let img = vc?.image_size, resolvedVisPatch > 0 {
                let side = max(1, img / resolvedVisPatch)
                return side * side
            }
            return vc != nil ? 2304 : 0
        }()
        return QWModelDescriptor(
            family: family, modelType: t.model_type, capabilities: caps,
            hidden: t.hidden_size, vocab: t.vocab_size, denseIntermediate: t.intermediate_size,
            numLayers: t.num_hidden_layers, layerTypes: normalizedLayerTypes, fullAttentionInterval: resolvedFAInterval,
            rmsEps: Float(t.rms_norm_eps), maxPositionEmbeddings: t.max_position_embeddings,
            qHeads: t.num_attention_heads, kvHeads: t.num_key_value_heads, headDim: resolvedHeadDim,
            attnOutputGate: attnGate, partialRotaryFactor: resolvedPartialRotary, ropeTheta: ropeTheta,
            indexerHeads: resolvedIdxHeads, indexerKvHeads: resolvedIdxKvHeads,
            indexerHeadDim: resolvedIdxHeadDim, indexerBudget: resolvedIdxBudget,
            indexerCompressRatio: resolvedIdxCompress,
            numExperts: resolvedNumExperts, expertsPerTok: t.num_experts_per_tok, moeIntermediate: t.moe_intermediate_size,
            sharedExpertIntermediate: resolvedSharedIntermediate, expertGateType: resolvedGateType,
            hcCount: t.hc_count, hcLowrank: t.hc_lowrank,
            gdnQKHeads: qkHeads, gdnVHeads: vHeads, gdnHeadDim: gHeadDim,
            gdnQKVChannels: qkv, gdnZChannels: z, gdnConvKernel: gConv,
            pleLayerIds: t.ple_layer_ids, pleEmbedDim: t.ple_embed_dim, pleConvKernel: t.ple_conv_kernel_size,
            ngramSize: t.ngram_size, headsPerNgram: t.heads_per_ngram, ngramVocabSizeBase: t.ngram_vocab_size_base,
            makeNgramVocabDivisibleBy: t.make_ngram_vocab_size_divisible_by, splitNgramParts: t.split_ngram_parts,
            mtpLayers: resolvedMtpLayers, mtpUseDedicatedEmbeddings: t.mtp_use_dedicated_embeddings ?? false,
            eosTokenId: eos, mropeSection: mropeSec,
            visDepth: vc?.depth ?? 0, visHidden: vc?.hidden_size ?? 0,
            visHeads: vc?.num_heads ?? 0, visIntermediate: vc?.intermediate_size ?? 0,
            visPatch: resolvedVisPatch, visTemporal: vc?.temporal_patch_size ?? (vc != nil ? 2 : 0),
            visMerge: vc?.spatial_merge_size ?? (vc != nil ? 2 : 0), visPosEmbeds: resolvedVisPosEmbeds,
            visOut: vc?.out_hidden_size ?? (vc != nil ? t.hidden_size : 0),
            imageTokenId: env.image_token_id ?? (vc != nil ? 248056 : -1),
            videoTokenId: env.video_token_id ?? (vc != nil ? 248057 : -1),
            visionStartTokenId: env.vision_start_token_id ?? (vc != nil ? 248053 : -1),
            visionEndTokenId: env.vision_end_token_id ?? (vc != nil ? 248054 : -1),
            hasQKNorm: t.qk_layernorm ?? true,
            usesPlusOneRMSNorm: familySpec.usesPlusOneRMSNorm,
            quantization: t.quantization ?? t.quantization_config ?? env.quantization ?? env.quantization_config,
            hasMropeSection: hasMrope,
            tieWordEmbeddings: t.tie_word_embeddings ?? false,
            normTopKProb: t.norm_topk_prob ?? (resolvedNumExperts != nil),
            eosTokenIds: eosList,
            envelopeModelType: env.model_type,
            loopSettings: parsedLoop,
            mhcConfig: parsedMHC,
            firstKDenseReplace: resolvedFirstKDenseReplace,
            moeLayerFreq: max(1, t.moe_layer_freq ?? 1),
            routedScalingFactor: t.routed_scaling_factor ?? 1.0,
            swigluLimit: (t.mlp_clamp == false) ? nil : (t.swiglu_limit ?? t.swiglu_limits),
            topkMethod: t.topk_method,
            sharedExpertHasGate: t.moe_shared_expert_gate ?? t.use_shared_expert_gate ?? (family == .maxwellMoE ? false : (t.n_shared_experts == nil)),
            mlaUseNoPE: isNoPEMLA,
            kvLoraRank: parsedHybridMLA?.kvLoraRank ?? t.kv_lora_rank,
            qLoraRank: parsedHybridMLA?.qLoraRank ?? t.q_lora_rank,
            qkNopeHeadDim: parsedHybridMLA?.qkNopeHeadDim ?? t.qk_nope_head_dim,
            qkRopeHeadDim: parsedHybridMLA?.qkRopeHeadDim ?? t.qk_rope_head_dim,
            vHeadDim: parsedHybridMLA?.vHeadDim ?? t.v_head_dim,
            indexKPoolCompress: t.index_kpool_compress ?? false,
            indexKPoolAlwaysSelectTail: t.index_kpool_always_select_tail ?? false,
            indexShareForMTP: t.index_share_for_mtp_iteration ?? false,
            indexerRopeInterleave: t.indexer_rope_interleave ?? t.rope_interleave ?? false,
            gdnVectorDecay: isKDA,
            gdnLowRankDim: gdnLowRank,
            gdnGateLowerBound: linCfg?.gate_lower_bound,
            gdnSeparateConv: isKDA,
            hcaConfig: parsedHCA,
            hybridMLAConfig: parsedHybridMLA,
            blockSparseGQAConfig: parsedBlockSparseGQA,
            activationKind: parsedAct,
            moeLatentSize: t.moe_latent_size,
            latentMoeUseNorm: t.latent_moe_use_norm ?? false,
            moeSplitFeatureRouting: t.moe_split_feature_routing ?? false,
            moeRouterEnableExpertBias: t.moe_router_enable_expert_bias ?? false,
            routedScalingFactorLearnable: t.routed_scaling_factor_learnable ?? false,
            multimaxHead: (t.multimax?.lowercased() == "lm_head"),
            mtpSharedLastLayer: t.mtp_shared_last_layer ?? false,
            role: caps.contains(.vision) ? .multimodalConditionalLM : .causalLM)
    }

}
