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
        let rawRoot = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        let rawText = (rawRoot?["text_config"] as? [String: Any]) ?? rawRoot ?? [:]
        let env = try JSONDecoder().decode(ConfigEnvelope.self, from: data)
        let t: ConfigEnvelope.Text
        if let tc = env.text_config {
            t = tc
        } else {
            do {
                t = try JSONDecoder().decode(ConfigEnvelope.Text.self, from: data)
            } catch DecodingError.keyNotFound(let key, _) {
                throw DescriptorError.invariant("flat config missing '\(key.stringValue)'")
            } catch DecodingError.valueNotFound(_, let ctx) {
                let name = ctx.codingPath.last?.stringValue ?? "required field"
                throw DescriptorError.invariant("flat config missing '\(name)'")
            }
        }
        func inv(_ cond: Bool, _ msg: String) throws {
            if !cond { throw DescriptorError.invariant(msg) }
        }
        guard let textFamilySpec = QWModelFamilyRegistry.shared.spec(forModelType: t.model_type) else {
            throw QWModelDescriptorError.unsupportedModelType(t.model_type)
        }
        let envFamilySpec = env.model_type.flatMap { QWModelFamilyRegistry.shared.spec(forModelType: $0) }
        let familySpec = envFamilySpec ?? textFamilySpec
        let family = familySpec.family

        try validateUnsupportedFeatures(text: t, env: env, familySpec: familySpec,
                                        rawRoot: rawRoot, rawText: rawText, inv: inv)

        try inv(t.hidden_size > 0, "hidden_size \(t.hidden_size) <= 0")
        try inv(t.vocab_size > 0, "vocab_size \(t.vocab_size) <= 0")
        try inv(t.num_hidden_layers > 0, "num_hidden_layers \(t.num_hidden_layers) <= 0")
        try inv(t.num_attention_heads > 0, "num_attention_heads \(t.num_attention_heads) <= 0")
        let resolvedKVHeads = t.num_key_value_heads ?? t.num_attention_heads
        try inv(resolvedKVHeads > 0 && t.num_attention_heads % resolvedKVHeads == 0,
                "num_attention_heads \(t.num_attention_heads) not divisible by num_key_value_heads \(resolvedKVHeads)")
        let resolvedMaxPos = t.max_position_embeddings ?? t.model_max_length
            ?? env.max_position_embeddings ?? env.model_max_length ?? 0
        try inv(resolvedMaxPos > 0, "max_position_embeddings \(resolvedMaxPos) <= 0")
        try inv(t.rms_norm_eps > 0 && t.rms_norm_eps.isFinite, "rms_norm_eps \(t.rms_norm_eps) invalid")
        let rawLayerTypes: [String]
        if let explicitTypes = t.layer_types {
            if familySpec.defaultAllFullAttention,
               let odd = explicitTypes.first(where: { $0 != "full_attention" && $0 != "sliding_attention" }) {
                throw DescriptorError.invariant("unsupported layer_types '\(odd)'")
            }
            rawLayerTypes = explicitTypes
        } else if let ratios = t.compress_ratios {
            try inv(ratios.count >= t.num_hidden_layers,
                    "compress_ratios count \(ratios.count) < num_hidden_layers \(t.num_hidden_layers)")
            rawLayerTypes = (0..<t.num_hidden_layers).map { l in
                ratios[l] > 0 ? "compressed_attention" : "full_attention"
            }
        } else if family == .inkling2Loop {
            let localSet = Set((rawText["local_layer_ids"] as? [Int]) ?? [])
            rawLayerTypes = (0..<t.num_hidden_layers).map { localSet.contains($0) ? "sliding_attention" : "full_attention" }
        } else if t.attention_variant == "block_sparse_gqa" || t.model_type == "maxwell_moe"
                    || familySpec.defaultAllFullAttention {
            rawLayerTypes = Array(repeating: "full_attention", count: t.num_hidden_layers)
        } else {
            throw DescriptorError.invariant("flat config missing 'layer_types'")
        }
        let normalizedLayerTypes = rawLayerTypes.map { kind -> String in
            kind == "deepseek_sparse_attention" ? "full_attention" : kind
        }
        try inv(normalizedLayerTypes.count == t.num_hidden_layers,
                "layer_types count \(normalizedLayerTypes.count) != num_hidden_layers \(t.num_hidden_layers)")
        let isGQAKind: (String) -> Bool = { $0 == "full_attention" || $0 == "sliding_attention" }
        if let badKind = normalizedLayerTypes.first(where: {
            !isGQAKind($0) && $0 != "linear_attention" && $0 != "compressed_attention"
        }) {
            throw DescriptorError.invariant("layer_types contain unknown kind '\(badKind)'")
        }

        let hasNonGQALayers = normalizedLayerTypes.contains("linear_attention")
            || normalizedLayerTypes.contains("compressed_attention")
        let resolvedFAInterval: Int
        if let explicitInterval = t.full_attention_interval ?? (family == .inkling2Loop ? dictInt(rawText, "n_layers_per_attn_res_block") : nil) {
            resolvedFAInterval = explicitInterval
        } else if let firstFA = normalizedLayerTypes.firstIndex(where: isGQAKind) {
            let candidate = firstFA + 1
            let fullIndices = normalizedLayerTypes.indices.filter { isGQAKind(normalizedLayerTypes[$0]) }
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
        } else if t.head_dim == nil && t.attention_head_size == nil && familySpec.defaultAllFullAttention {
            try inv(t.hidden_size % t.num_attention_heads == 0,
                    "hidden_size \(t.hidden_size) is not divisible by num_attention_heads \(t.num_attention_heads)")
            resolvedHeadDim = t.hidden_size / t.num_attention_heads
        } else if let badHd = t.head_dim ?? t.attention_head_size {
            throw DescriptorError.invariant("head_dim \(badHd) <= 0")
        } else {
            throw DescriptorError.invariant("flat config missing 'head_dim'")
        }

        let hasHybridMLA = t.hybrid_mla_kv_lora_rank != nil || t.hybrid_mla_attention != nil
        let isNoPEMLA = t.mla_use_nope ?? (t.kv_lora_rank != nil && t.use_mla != false && t.compress_ratios == nil && !hasHybridMLA)
        if !isNoPEMLA && !hasHybridMLA && !normalizedLayerTypes.contains("compressed_attention") {
            try inv(resolvedHeadDim % 32 == 0 && resolvedHeadDim <= 256,
                    "head_dim \(resolvedHeadDim) must be a multiple of 32 in 32...256")
        }
        let resolvedPartialRotary: Double
        if let prf = t.partial_rotary_factor ?? t.rope_parameters?.partial_rotary_factor {
            resolvedPartialRotary = prf
        } else if let rp = t.rotary_percent {
            resolvedPartialRotary = rp
        } else if isNoPEMLA || hasHybridMLA || t.compress_ratios != nil
                    || t.attention_variant == "block_sparse_gqa" || t.model_type == "maxwell_moe"
                    || familySpec.defaultAllFullAttention || family == .inkling2Loop {
            resolvedPartialRotary = 1.0
        } else {
            throw DescriptorError.invariant("flat config missing 'partial_rotary_factor'")
        }

        // Minimal synthetic hybrid unit fixtures omit `rope_theta`; standard full-attention configs require it.
        let ropeTheta: Double
        if let explicitTheta = t.rope_theta ?? t.rope_parameters?.rope_theta ?? (family == .inkling2Loop ? dictDouble(rawRoot?["mtp_config"] as? [String: Any], "rope_theta") : nil) {
            ropeTheta = explicitTheta
        } else if !familySpec.defaultAllFullAttention {
            ropeTheta = 10000.0
        } else {
            throw DescriptorError.invariant("flat config missing 'rope_theta'")
        }
        try inv(resolvedPartialRotary > 0 && resolvedPartialRotary <= 1.0 && resolvedPartialRotary.isFinite,
                "partial_rotary_factor \(resolvedPartialRotary) invalid")
        try inv(ropeTheta > 0 && ropeTheta.isFinite, "rope_theta \(ropeTheta) invalid")
        if familySpec.defaultAllFullAttention && t.rope_parameters?.mrope_section != nil {
            throw DescriptorError.invariant("unsupported 'mrope_section'")
        }
        let mropeSec: (w: Int, h: Int, t: Int)
        let hasMrope: Bool
        if let sec = t.rope_parameters?.mrope_section {
            try inv(sec.count == 3, "mrope_section must have 3 elements")
            mropeSec = (sec[0], sec[1], sec[2])
            hasMrope = true
        } else if familySpec.defaultAllFullAttention || family == .inkling2Loop {
            mropeSec = (0, 0, 0)
            hasMrope = false
        } else {
            mropeSec = (11, 11, 10)
            let rotHalf = Int(Double(resolvedHeadDim) * resolvedPartialRotary) / 2
            hasMrope = (rotHalf == 32)
        }

        let moeConfig = try validateAndResolveMoE(text: t, familySpec: familySpec, inv: inv)
        let resolvedNumExperts = moeConfig.numExperts
        if resolvedNumExperts == nil && resolvedFAInterval > 0 && hasNonGQALayers {
            for l in 0..<t.num_hidden_layers {
                let want = (l + 1) % resolvedFAInterval == 0
                try inv(isGQAKind(normalizedLayerTypes[l]) == want, "layer \(l) cadence")
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
        guard let eosVal = t.eos_token_id ?? env.eos_token_id ?? env.generation_config?.eos_token_id else {
            throw DescriptorError.invariant("flat config missing 'eos_token_id'")
        }
        let eos = eosVal.value
        let eosList = eosVal.values

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
        let mtpCfgLayers = family == .inkling2Loop ? dictInt(rawRoot?["mtp_config"] as? [String: Any], "num_hidden_layers") : nil
        let resolvedMtpLayers = max(0, t.mtp_num_hidden_layers ?? t.num_nextn_predict_layers ?? t.mtp_num_layers ?? mtpCfgLayers ?? 0)
        if resolvedMtpLayers > 0 { caps.insert(.mtp) }

        let (resolvedRopeScaling, parsedYarn) = try parseRoPEScaling(text: t, env: env)

        let parsedHCA: HCAConfig? = try {
            guard normalizedLayerTypes.contains("compressed_attention"),
                  let ratios = t.compress_ratios else { return nil }
            guard let defRatio = ratios.first(where: { $0 > 0 }) else {
                throw DescriptorError.invariant("compressed_attention missing positive compress_ratios entry")
            }
            guard let qLoraRank = t.q_lora_rank else {
                throw DescriptorError.invariant("HCA missing q_lora_rank")
            }
            guard let kvLoraRank = t.kv_lora_rank else {
                throw DescriptorError.invariant("HCA missing kv_lora_rank")
            }
            guard let qkNope = t.qk_nope_head_dim ?? (rawText["compress_qk_nope_head_dim"] as? Int) else {
                throw DescriptorError.invariant("HCA missing qk_nope_head_dim")
            }
            guard let qkRope = t.qk_rope_head_dim ?? (rawText["compress_qk_rope_head_dim"] as? Int) else {
                throw DescriptorError.invariant("HCA missing qk_rope_head_dim")
            }
            guard let slidingWindow = t.sliding_window else {
                throw DescriptorError.invariant("HCA missing sliding_window")
            }
            let hcaVDim = t.v_head_dim ?? (rawText["compress_v_head_dim"] as? Int) ?? resolvedHeadDim
            // Documented test fallback: minimal HCA unit fixtures omit `o_groups`/`o_lora_rank`.
            return HCAConfig(
                compressRatios: ratios,
                defaultCompressRatio: defRatio,
                qHeads: t.num_attention_heads,
                kvHeads: resolvedKVHeads,
                headDim: resolvedHeadDim,
                qLoraRank: qLoraRank,
                kvLoraRank: kvLoraRank,
                qkNopeHeadDim: qkNope,
                qkRopeHeadDim: qkRope,
                vHeadDim: hcaVDim,
                oGroups: t.o_groups ?? 8,
                oLoraRank: t.o_lora_rank ?? 1024,
                slidingWindow: slidingWindow,
                compressRopeTheta: t.compress_rope_theta ?? ropeTheta,
                yarn: (t.hca_rope_type == nil || t.hca_rope_type?.lowercased() == "yarn") ? parsedYarn : nil,
                useAttnMerger: t.use_attn_merger ?? true,
                useVHA: t.use_vha_attention ?? false,
                hasAttnSinkBias: t.add_full_attention_sink_bias ?? false,
                gatedAttnUseQLora: t.gated_attn_use_q_lora ?? false
            )
        }()

        let parsedHybridMLA: HybridMLAConfig? = try {
            guard hasHybridMLA else { return nil }
            guard let qLora = t.hybrid_mla_q_lora_rank ?? t.q_lora_rank else {
                throw DescriptorError.invariant("HybridMLA missing q_lora_rank")
            }
            guard let kvLora = t.hybrid_mla_kv_lora_rank ?? t.kv_lora_rank else {
                throw DescriptorError.invariant("HybridMLA missing kv_lora_rank")
            }
            guard let nope = t.hybrid_mla_qk_nope_head_dim ?? t.qk_nope_head_dim else {
                throw DescriptorError.invariant("HybridMLA missing qk_nope_head_dim")
            }
            guard let rope = t.hybrid_mla_qk_rope_head_dim ?? t.qk_rope_head_dim else {
                throw DescriptorError.invariant("HybridMLA missing qk_rope_head_dim")
            }
            guard let vDim = t.hybrid_mla_v_head_dim ?? t.v_head_dim else {
                throw DescriptorError.invariant("HybridMLA missing v_head_dim")
            }
            let mlaYarn: YarnRopeConfig? = (t.rope_type?.lowercased() == "yarn") ? parsedYarn : nil
            return HybridMLAConfig(
                qHeads: t.hybrid_mla_num_attention_heads ?? t.num_attention_heads,
                qLoraRank: qLora,
                kvLoraRank: kvLora,
                qkNopeHeadDim: nope,
                qkRopeHeadDim: rope,
                vHeadDim: vDim,
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
        if let rawAct = t.hidden_act {
            let act = rawAct.lowercased()
            if act == "situ" {
                parsedAct = .situ(beta: t.activation_situ_beta ?? 4.0,
                                  linearBeta: t.activation_situ_linear_beta ?? 25.0)
            } else if act == "silu" {
                parsedAct = .silu
            } else {
                throw DescriptorError.invariant("unsupported hidden_act '\(rawAct)'")
            }
        } else {
            parsedAct = .silu
        }

        let rawLoop = t.loop_settings ?? env.loop_settings
        let parsedLoop: LoopSettings? = rawLoop.flatMap { rl in
            guard let s = rl.start, let e = rl.end, s >= 0, e > s, e <= t.num_hidden_layers else { return nil }
            return LoopSettings(start: s, end: e, evalCount: max(1, rl.eval_count ?? rl.counts?.first ?? 1), counts: rl.counts)
        } ?? (family == .inkling2Loop ? inkling2LoopSettings(rawText: rawText, numLayers: t.num_hidden_layers) : nil)
        let parsedInk2 = family == .inkling2Loop
            ? try decodeInkling2Config(text: t, env: env, rawRoot: rawRoot, rawText: rawText, inv: inv)
            : nil

        let resolvedSlidingWindow: Int?
        let resolvedLayerSlidingWindows: [Int]?
        let effectiveSW = t.sliding_window ?? dictInt(rawText, "sliding_window_size")
        if !normalizedLayerTypes.contains("compressed_attention"), !hasHybridMLA {
            if let sw = effectiveSW { try inv(sw >= 0, "sliding_window \(sw) < 0") }
            if let mwl = t.max_window_layers { try inv(mwl >= 0, "max_window_layers \(mwl) < 0") }
            if let pat = t.sliding_window_pattern { try inv(pat >= 1, "sliding_window_pattern \(pat) < 1") }
            if normalizedLayerTypes.contains("sliding_attention") {
                guard let sw = effectiveSW, sw > 0 else {
                    throw DescriptorError.invariant("layer_types 'sliding_attention' requires positive sliding_window")
                }
                let effective = sw < resolvedMaxPos ? sw : 0
                resolvedSlidingWindow = nil
                resolvedLayerSlidingWindows = normalizedLayerTypes.map { $0 == "sliding_attention" ? effective : 0 }
            } else if (t.use_sliding_window ?? true),
                      let sw = effectiveSW, sw > 0, sw < resolvedMaxPos {
                if let mwl = t.max_window_layers {
                    resolvedSlidingWindow = nil
                    resolvedLayerSlidingWindows = (0..<t.num_hidden_layers).map { $0 < mwl ? 0 : sw }
                } else if let pat = t.sliding_window_pattern {
                    resolvedSlidingWindow = nil
                    resolvedLayerSlidingWindows = pat == 1 ? nil
                        : (0..<t.num_hidden_layers).map { ($0 + 1) % pat == 0 ? 0 : sw }
                } else {
                    resolvedSlidingWindow = sw
                    resolvedLayerSlidingWindows = nil
                }
            } else {
                resolvedSlidingWindow = nil
                resolvedLayerSlidingWindows = nil
            }
        } else {
            resolvedSlidingWindow = nil
            resolvedLayerSlidingWindows = nil
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
                  idxHeads == resolvedKVHeads, idxDim == resolvedHeadDim else {
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

        let gateDeclared: Bool = t.output_gate_type != nil
        let attnGate: Bool = t.attn_output_gate ?? t.use_gated_attn ?? (gateDeclared || familySpec.defaultAttnOutputGate)
        let resolvedGateType = t.scoring_func ?? t.router_score_function ?? familySpec.defaultExpertGateType
        let vd = try resolveVisionDims(vc: env.vision_config, rawVision: rawRoot?["vision_config"] as? [String: Any], hidden: t.hidden_size, isInkling2: family == .inkling2Loop)
        let hasVis = env.vision_config != nil
        let descriptor = QWModelDescriptor(
            family: family, modelType: t.model_type, capabilities: caps,
            hidden: t.hidden_size, vocab: t.vocab_size,
            denseIntermediate: family == .inkling2Loop ? nil : t.intermediate_size,
            numLayers: t.num_hidden_layers, layerTypes: normalizedLayerTypes, fullAttentionInterval: resolvedFAInterval,
            rmsEps: Float(t.rms_norm_eps), maxPositionEmbeddings: resolvedMaxPos,
            qHeads: t.num_attention_heads, kvHeads: resolvedKVHeads, headDim: resolvedHeadDim,
            attnOutputGate: attnGate, partialRotaryFactor: resolvedPartialRotary, ropeTheta: ropeTheta,
            indexerHeads: resolvedIdxHeads, indexerKvHeads: resolvedIdxKvHeads,
            indexerHeadDim: resolvedIdxHeadDim, indexerBudget: resolvedIdxBudget,
            indexerCompressRatio: resolvedIdxCompress,
            numExperts: resolvedNumExperts, expertsPerTok: t.num_experts_per_tok,
            moeIntermediate: t.moe_intermediate_size ?? (family == .inkling2Loop && resolvedNumExperts != nil ? t.intermediate_size : nil),
            sharedExpertIntermediate: resolvedSharedIntermediate, expertGateType: resolvedGateType,
            hcCount: t.hc_count, hcLowrank: t.hc_lowrank,
            gdnQKHeads: qkHeads, gdnVHeads: vHeads, gdnHeadDim: gHeadDim,
            gdnQKVChannels: qkv, gdnZChannels: z, gdnConvKernel: gConv,
            pleLayerIds: t.ple_layer_ids, pleEmbedDim: t.ple_embed_dim, pleConvKernel: t.ple_conv_kernel_size,
            ngramSize: t.ngram_size, headsPerNgram: t.heads_per_ngram, ngramVocabSizeBase: t.ngram_vocab_size_base,
            makeNgramVocabDivisibleBy: t.make_ngram_vocab_size_divisible_by, splitNgramParts: t.split_ngram_parts,
            mtpLayers: resolvedMtpLayers, mtpUseDedicatedEmbeddings: t.mtp_use_dedicated_embeddings ?? false,
            eosTokenId: eos, mropeSection: mropeSec,
            visDepth: vd.depth, visHidden: vd.hidden, visHeads: vd.heads, visIntermediate: vd.inter,
            visPatch: vd.patch, visTemporal: vd.temporal, visMerge: vd.merge, visPosEmbeds: vd.pos, visOut: vd.out,
            imageTokenId: env.image_token_id ?? (hasVis ? (family == .inkling2Loop ? 200054 : 248056) : -1),
            videoTokenId: env.video_token_id ?? (hasVis ? (family == .inkling2Loop ? -1 : 248057) : -1),
            visionStartTokenId: env.vision_start_token_id ?? (hasVis ? 248053 : -1),
            visionEndTokenId: env.vision_end_token_id ?? (hasVis ? 248054 : -1),
            hasQKNorm: t.use_qk_norm ?? t.qk_layernorm ?? t.qk_norm ?? familySpec.defaultHasQKNorm,
            hasQKVBias: (t.qkv_bias == true) || (t.attention_bias == true),
            hasAttnOutputBias: (t.attention_bias == true),
            usesPlusOneRMSNorm: familySpec.usesPlusOneRMSNorm,
            quantization: t.quantization ?? t.quantization_config ?? env.quantization ?? env.quantization_config,
            hasMropeSection: hasMrope,
            tieWordEmbeddings: t.tie_word_embeddings ?? env.tie_word_embeddings ?? false,
            normTopKProb: t.norm_topk_prob ?? (resolvedNumExperts != nil ? familySpec.defaultNormTopKProb : false),
            eosTokenIds: eosList,
            envelopeModelType: env.model_type,
            loopSettings: parsedLoop,
            mhcConfig: parsedMHC,
            firstKDenseReplace: moeConfig.firstKDenseReplace,
            mlpOnlyLayers: moeConfig.mlpOnlyLayers,
            decoderSparseStep: moeConfig.decoderSparseStep,
            moeLayerFreq: moeConfig.moeLayerFreq,
            routedScalingFactor: t.routed_scaling_factor ?? dictFloat(rawText, "route_scale") ?? 1.0,
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
            yarnRopeConfig: (!hasHybridMLA && parsedHCA == nil) ? parsedYarn : nil,
            ropeScaling: resolvedRopeScaling,
            slidingWindow: resolvedSlidingWindow,
            layerSlidingWindows: resolvedLayerSlidingWindows,
            activationKind: parsedAct,
            moeLatentSize: t.moe_latent_size,
            latentMoeUseNorm: t.latent_moe_use_norm ?? false,
            moeSplitFeatureRouting: t.moe_split_feature_routing ?? false,
            moeRouterEnableExpertBias: t.moe_router_enable_expert_bias ?? (dictBool(rawText, "use_gate_bias") ?? false),
            routedScalingFactorLearnable: t.routed_scaling_factor_learnable ?? false,
            multimaxHead: (t.multimax?.lowercased() == "lm_head"),
            mtpSharedLastLayer: t.mtp_shared_last_layer ?? false,
            inkling2Config: parsedInk2,
            role: caps.contains(.vision) ? .multimodalConditionalLM : .causalLM)
        let issues = descriptor.verifyInternalConsistency()
        if !issues.isEmpty {
            throw DescriptorError.invariant(issues.joined(separator: "; "))
        }
        return descriptor
    }

    private struct ResolvedMoESchedule {
        let numExperts: Int?
        let firstKDenseReplace: Int
        let mlpOnlyLayers: Set<Int>
        let decoderSparseStep: Int
        let moeLayerFreq: Int
    }

    private static func rawKeyPresent(_ dict: [String: Any]?, _ key: String) -> Bool {
        guard let val = dict?[key] else { return false }
        return !(val is NSNull)
    }

    private static func validateUnsupportedFeatures(
        text t: ConfigEnvelope.Text,
        env: ConfigEnvelope,
        familySpec: QWModelFamilySpec,
        rawRoot: [String: Any]?,
        rawText: [String: Any],
        inv: (Bool, String) throws -> Void
    ) throws {
        let archs = t.architectures ?? env.architectures
        if let archs, !archs.isEmpty,
           !archs.contains(where: { $0.hasSuffix("ForCausalLM") || $0.hasSuffix("ForConditionalGeneration") }) {
            throw DescriptorError.invariant("unsupported architecture '\(archs[0])'")
        }
        if familySpec.defaultAllFullAttention,
           rawKeyPresent(rawText, "auto_map") || rawKeyPresent(rawRoot, "auto_map") {
            throw DescriptorError.invariant("unsupported 'auto_map'")
        }
        try inv(t.mlp_bias != true, "unsupported 'mlp_bias'")
        try inv(t.output_router_logits != true, "unsupported 'output_router_logits'")
    }

    private static func validateAndResolveMoE(
        text t: ConfigEnvelope.Text,
        familySpec: QWModelFamilySpec,
        inv: (Bool, String) throws -> Void
    ) throws -> ResolvedMoESchedule {
        let family = familySpec.family
        let resolvedMLPOnly = Set(t.mlp_only_layers ?? [])
        if let only = t.mlp_only_layers {
            for idx in only {
                try inv(idx >= 0 && idx < t.num_hidden_layers, "mlp_only_layers index \(idx) out of range")
            }
        }
        let resolvedFirstKDenseReplace: Int
        if let fk = t.first_k_dense_replace {
            try inv(fk >= 0, "first_k_dense_replace \(fk) < 0")
            resolvedFirstKDenseReplace = fk
        } else if let denseLayers = t.mlp_only_layers, !denseLayers.isEmpty,
                  Set(denseLayers) == Set(0..<denseLayers.count) {
            resolvedFirstKDenseReplace = denseLayers.count
        } else {
            resolvedFirstKDenseReplace = 0
        }
        let resolvedDecoderSparseStep = t.decoder_sparse_step ?? 1
        try inv(resolvedDecoderSparseStep >= 1, "decoder_sparse_step \(resolvedDecoderSparseStep) < 1")
        let resolvedMoELayerFreq = t.moe_layer_freq ?? 1
        try inv(resolvedMoELayerFreq >= 1, "moe_layer_freq \(resolvedMoELayerFreq) < 1")

        let rawNumExperts = t.num_experts ?? t.n_routed_experts
        if let rawNumExperts {
            try inv(rawNumExperts >= 0, "num_experts \(rawNumExperts) < 0")
        }
        if familySpec.disallowsMoEConfig {
            try inv(rawNumExperts == nil,
                    "num_experts \(rawNumExperts ?? 0) on the dense family '\(family.rawValue)'")
            if let moeW = t.moe_intermediate_size, moeW > 0 {
                throw DescriptorError.invariant("moe_intermediate_size \(moeW) on the dense family '\(family.rawValue)'")
            }
            if let sharedW = t.shared_expert_intermediate_size, sharedW > 0 {
                throw DescriptorError.invariant("shared_expert_intermediate_size \(sharedW) on the dense family '\(family.rawValue)'")
            }
        }
        if familySpec.requiresSparseMoEConfig {
            guard let experts = rawNumExperts else {
                throw DescriptorError.invariant("flat config missing 'num_experts' on sparse family '\(family.rawValue)'")
            }
            try inv(experts >= 32 && experts % 32 == 0 && experts <= 512,
                    "num_experts \(experts) must be a multiple of 32 in 32...512")
            guard let topK = t.num_experts_per_tok else {
                throw DescriptorError.invariant("flat config missing 'num_experts_per_tok'")
            }
            try inv(topK > 0 && topK <= 16 && topK <= experts,
                    "num_experts_per_tok \(topK) must be in 1...min(16, \(experts))")
            guard let moeW = t.moe_intermediate_size else {
                throw DescriptorError.invariant("flat config missing 'moe_intermediate_size'")
            }
            try inv(moeW > 0, "moe_intermediate_size \(moeW) <= 0")
        }
        if let sw = t.shared_expert_intermediate_size {
            try inv(sw >= 0, "shared_expert_intermediate_size \(sw) < 0")
        }
        let resolvedNumExperts = rawNumExperts.flatMap { $0 > 0 ? $0 : nil }
        if let n = rawNumExperts, let k = t.num_experts_per_tok {
            try inv(n > 0 && k > 0 && k <= n, "MoE top-k \(k) > experts \(n)")
        }
        let hasDenseLayer: Bool = {
            guard resolvedNumExperts != nil else { return true }
            return (0..<t.num_hidden_layers).contains { layer in
                if resolvedMLPOnly.contains(layer) || layer < resolvedFirstKDenseReplace { return true }
                if resolvedDecoderSparseStep > 1 && (layer + 1) % resolvedDecoderSparseStep != 0 { return true }
                if resolvedMoELayerFreq > 1 && (layer - resolvedFirstKDenseReplace) % resolvedMoELayerFreq != 0 { return true }
                return false
            }
        }()
        if hasDenseLayer || familySpec.disallowsMoEConfig {
            guard let inter = t.intermediate_size else {
                throw DescriptorError.invariant("flat config missing 'intermediate_size'")
            }
            try inv(inter > 0, "intermediate_size \(inter) <= 0")
        }
        return ResolvedMoESchedule(
            numExperts: resolvedNumExperts,
            firstKDenseReplace: resolvedFirstKDenseReplace,
            mlpOnlyLayers: resolvedMLPOnly,
            decoderSparseStep: resolvedDecoderSparseStep,
            moeLayerFreq: resolvedMoELayerFreq
        )
    }

    private static func parseRoPEScaling(
        text t: ConfigEnvelope.Text,
        env: ConfigEnvelope
    ) throws -> (RoPEScaling?, YarnRopeConfig?) {
        let candidates: [(type: String?, ropeType: String?, factor: Double?, orig: Int?,
                          betaFast: Double?, betaSlow: Double?, mscale: Double?, mscaleAllDim: Double?,
                          defaultMscale: Double)] = [
            t.rope_scaling.map {
                ($0.type, $0.rope_type, $0.factor, $0.original_max_position_embeddings,
                 $0.beta_fast, $0.beta_slow, $0.mscale, $0.mscale_all_dim, 0.0)
            },
            t.rope_parameters.map {
                ($0.type, $0.rope_type, $0.factor, $0.original_max_position_embeddings,
                 $0.beta_fast, $0.beta_slow, $0.mscale, $0.mscale_all_dim, 1.0)
            },
        ].compactMap { $0 }

        for c in candidates {
            let rawKind = c.ropeType ?? c.type
            guard let kind = rawKind?.lowercased() else {
                if c.factor != nil {
                    throw DescriptorError.invariant("unsupported rope_scaling without a type")
                }
                continue
            }
            if kind == "default" { continue }
            guard let factor = c.factor, factor > 0, factor.isFinite else {
                throw DescriptorError.invariant("rope_scaling '\(kind)' missing positive factor")
            }
            let orig = c.orig ?? t.original_max_position_embeddings ?? env.original_max_position_embeddings
            switch kind {
            case "yarn":
                guard let orig, orig > 0 else {
                    throw DescriptorError.invariant("rope_scaling 'yarn' missing original_max_position_embeddings")
                }
                let yarn = YarnRopeConfig(
                    factor: factor,
                    originalMaxPositionEmbeddings: orig,
                    betaFast: c.betaFast ?? 32.0,
                    betaSlow: c.betaSlow ?? 1.0,
                    mscale: c.mscale ?? c.defaultMscale,
                    mscaleAllDim: c.mscaleAllDim ?? 0.0
                )
                return (.yarn(yarn), yarn)
            case "linear":
                return (.linear(factor: factor), nil)
            case "dynamic":
                return (.dynamic(factor: factor, originalMaxPositionEmbeddings: orig), nil)
            default:
                throw DescriptorError.invariant("unsupported rope_scaling '\(rawKind ?? kind)'")
            }
        }
        return (nil, nil)
    }

    /// Return a copy of this descriptor with updated attention bias flags.
    public func withAttentionBias(qkv: Bool? = nil, output: Bool? = nil) -> QWModelDescriptor {
        let nextQKV = qkv ?? hasQKVBias
        let nextOutput = output ?? hasAttnOutputBias
        if nextQKV == hasQKVBias && nextOutput == hasAttnOutputBias { return self }
        return QWModelDescriptor(
            family: family, modelType: modelType, capabilities: capabilities,
            hidden: hidden, vocab: vocab, denseIntermediate: denseIntermediate,
            numLayers: numLayers, layerTypes: layerTypes, fullAttentionInterval: fullAttentionInterval,
            rmsEps: rmsEps, maxPositionEmbeddings: maxPositionEmbeddings,
            qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
            attnOutputGate: attnOutputGate, partialRotaryFactor: partialRotaryFactor, ropeTheta: ropeTheta,
            indexerHeads: indexerHeads, indexerKvHeads: indexerKvHeads, indexerHeadDim: indexerHeadDim,
            indexerBudget: indexerBudget, indexerCompressRatio: indexerCompressRatio,
            numExperts: numExperts, expertsPerTok: expertsPerTok, moeIntermediate: moeIntermediate,
            sharedExpertIntermediate: sharedExpertIntermediate, expertGateType: expertGateType,
            hcCount: hcCount, hcLowrank: hcLowrank,
            gdnQKHeads: gdnQKHeads, gdnVHeads: gdnVHeads, gdnHeadDim: gdnHeadDim,
            gdnQKVChannels: gdnQKVChannels, gdnZChannels: gdnZChannels, gdnConvKernel: gdnConvKernel,
            pleLayerIds: pleLayerIds, pleEmbedDim: pleEmbedDim, pleConvKernel: pleConvKernel,
            ngramSize: ngramSize, headsPerNgram: headsPerNgram, ngramVocabSizeBase: ngramVocabSizeBase,
            makeNgramVocabDivisibleBy: makeNgramVocabDivisibleBy, splitNgramParts: splitNgramParts,
            mtpLayers: mtpLayers, mtpUseDedicatedEmbeddings: mtpUseDedicatedEmbeddings, eosTokenId: eosTokenId,
            mropeSection: mropeSection,
            visDepth: visDepth, visHidden: visHidden, visHeads: visHeads,
            visIntermediate: visIntermediate, visPatch: visPatch, visTemporal: visTemporal,
            visMerge: visMerge, visPosEmbeds: visPosEmbeds, visOut: visOut,
            imageTokenId: imageTokenId, videoTokenId: videoTokenId,
            visionStartTokenId: visionStartTokenId, visionEndTokenId: visionEndTokenId,
            hasQKNorm: hasQKNorm, hasQKVBias: nextQKV, hasAttnOutputBias: nextOutput,
            usesPlusOneRMSNorm: usesPlusOneRMSNorm,
            quantization: quantization, hasMropeSection: hasMropeSection,
            tieWordEmbeddings: tieWordEmbeddings, normTopKProb: normTopKProb,
            eosTokenIds: eosTokenIds, envelopeModelType: envelopeModelType,
            loopSettings: loopSettings, mhcConfig: mhcConfig,
            firstKDenseReplace: firstKDenseReplace,
            mlpOnlyLayers: mlpOnlyLayers,
            decoderSparseStep: decoderSparseStep,
            moeLayerFreq: moeLayerFreq,
            routedScalingFactor: routedScalingFactor, swigluLimit: swigluLimit,
            topkMethod: topkMethod, residentExpertsPerLayer: residentExpertsPerLayer,
            sharedExpertHasGate: sharedExpertHasGate,
            mlaUseNoPE: mlaUseNoPE, kvLoraRank: kvLoraRank, qLoraRank: qLoraRank,
            qkNopeHeadDim: qkNopeHeadDim, qkRopeHeadDim: qkRopeHeadDim, vHeadDim: vHeadDim,
            indexKPoolCompress: indexKPoolCompress, indexKPoolAlwaysSelectTail: indexKPoolAlwaysSelectTail,
            indexShareForMTP: indexShareForMTP, indexerRopeInterleave: indexerRopeInterleave,
            gdnVectorDecay: gdnVectorDecay, gdnLowRankDim: gdnLowRankDim,
            gdnGateLowerBound: gdnGateLowerBound, gdnSeparateConv: gdnSeparateConv,
            hcaConfig: hcaConfig, hybridMLAConfig: hybridMLAConfig,
            blockSparseGQAConfig: blockSparseGQAConfig,
            yarnRopeConfig: yarnRopeConfig,
            ropeScaling: ropeScaling,
            slidingWindow: slidingWindow,
            layerSlidingWindows: layerSlidingWindows,
            activationKind: activationKind, moeLatentSize: moeLatentSize,
            latentMoeUseNorm: latentMoeUseNorm,
            moeSplitFeatureRouting: moeSplitFeatureRouting,
            moeRouterEnableExpertBias: moeRouterEnableExpertBias,
            routedScalingFactorLearnable: routedScalingFactorLearnable,
            multimaxHead: multimaxHead, mtpSharedLastLayer: mtpSharedLastLayer,
            inkling2Config: inkling2Config, role: role
        )
    }

}
