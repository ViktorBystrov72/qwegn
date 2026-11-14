// QWModelDescriptor+ConfigEnvelope.swift — Decodable HF/Qwen config JSON schema.
import Foundation

extension QWModelDescriptor {
    internal struct ConfigEnvelope: Decodable {
        struct RopeParameters: Decodable {
            let mrope_section: [Int]?
            let rope_theta: Double?
            let partial_rotary_factor: Double?
        }
        struct RopeScaling: Decodable {
            let type: String?
            let rope_type: String?
            let factor: Double?
            let original_max_position_embeddings: Int?
            let beta_fast: Double?
            let beta_slow: Double?
            let mscale: Double?
            let mscale_all_dim: Double?
        }
        struct VisionConfig: Decodable {
            let depth: Int?
            let hidden_size: Int?
            let num_heads: Int?
            let intermediate_size: Int?
            let patch_size: Int?
            let temporal_patch_size: Int?
            let spatial_merge_size: Int?
            let num_position_embeddings: Int?
            let out_hidden_size: Int?
            let image_size: Int?
            let rms_norm_eps: Double?
            let hidden_act: String?
        }
        struct RawLoopSettings: Decodable {
            let start: Int?
            let end: Int?
            let eval_count: Int?
            let counts: [Int]?
        }
        struct RawMHCConfig: Decodable {
            let start: Int?
            let end: Int?
            let streams: Int?
            let iterations: Int?
            let fixed_identity: Bool?
            let pre_scale: Float?
        }
        enum MHCValue: Decodable {
            case flag(Bool)
            case config(RawMHCConfig)

            init(from decoder: Decoder) throws {
                let c = try decoder.singleValueContainer()
                if let b = try? c.decode(Bool.self) {
                    self = .flag(b)
                    return
                }
                self = .config(try c.decode(RawMHCConfig.self))
            }
        }
        struct RawLinearAttnConfig: Decodable {
            let Linear_type: String?
            let linear_type: String?
            let num_heads: Int?
            let head_dim: Int?
            let short_conv_kernel_size: Int?
            let gate_lower_bound: Float?
            let low_rank_dim: Int?
        }
        struct Text: Decodable {
            let model_type: String
            let hidden_size: Int
            let vocab_size: Int
            let num_hidden_layers: Int
            var layer_types: [String]? = nil
            var compress_ratios: [Int]? = nil
            var full_attention_interval: Int? = nil
            let rms_norm_eps: Double
            let max_position_embeddings: Int
            let num_attention_heads: Int
            let num_key_value_heads: Int
            var head_dim: Int? = nil
            var qk_head_dim: Int? = nil
            var qk_nope_head_dim: Int? = nil
            var qk_rope_head_dim: Int? = nil
            var v_head_dim: Int? = nil
            var kv_lora_rank: Int? = nil
            var q_lora_rank: Int? = nil
            var o_groups: Int? = nil
            var o_lora_rank: Int? = nil
            var sliding_window: Int? = nil
            var mla_use_nope: Bool? = nil
            var use_mla: Bool? = nil
            var partial_rotary_factor: Double? = nil
            var rotary_percent: Double? = nil
            let rope_theta: Double?
            var compress_rope_theta: Double? = nil
            var rope_parameters: RopeParameters? = nil
            var rope_scaling: RopeScaling? = nil
            var rope_type: String? = nil
            var hca_rope_type: String? = nil
            var rope_interleave: Bool? = nil
            let output_gate_type: String?
            var attn_output_gate: Bool? = nil
            var use_gated_attn: Bool? = nil
            var use_attn_merger: Bool? = nil
            var use_vha_attention: Bool? = nil
            var add_full_attention_sink_bias: Bool? = nil
            var gated_attn_use_q_lora: Bool? = nil
            var hybrid_mla_attention: String? = nil
            var hybrid_mla_num_attention_heads: Int? = nil
            var hybrid_mla_q_lora_rank: Int? = nil
            var hybrid_mla_kv_lora_rank: Int? = nil
            var hybrid_mla_qk_nope_head_dim: Int? = nil
            var hybrid_mla_qk_rope_head_dim: Int? = nil
            var hybrid_mla_v_head_dim: Int? = nil
            var use_fast_hadamard: Bool? = nil
            var mqa_split_kv_b_proj: Bool? = nil
            let linear_key_head_dim: Int?
            let linear_num_key_heads: Int?
            let linear_num_value_heads: Int?
            let linear_value_head_dim: Int?
            let linear_conv_kernel_dim: Int?
            var linear_attn_config: RawLinearAttnConfig? = nil
            let num_experts: Int?
            var n_routed_experts: Int? = nil
            let num_experts_per_tok: Int?
            let moe_intermediate_size: Int?
            var moe_latent_size: Int? = nil
            var latent_moe_use_norm: Bool? = nil
            let shared_expert_intermediate_size: Int?
            var n_shared_experts: Int? = nil
            var moe_shared_expert_gate: Bool? = nil
            var moe_split_feature_routing: Bool? = nil
            var first_k_dense_replace: Int? = nil
            var moe_layer_freq: Int? = nil
            var routed_scaling_factor: Float? = nil
            var routed_scaling_factor_learnable: Bool? = nil
            var scoring_func: String? = nil
            var topk_method: String? = nil
            var mlp_clamp: Bool? = nil
            var swiglu_limit: Float? = nil
            var swiglu_limits: Float? = nil
            var norm_topk_prob: Bool? = nil
            var hidden_act: String? = nil
            var activation_situ_beta: Float? = nil
            var activation_situ_linear_beta: Float? = nil
            let indexer_budget: Int?
            var index_topk: Int? = nil
            let indexer_compress_ratio: Int?
            var index_kpool: Int? = nil
            let indexer_head_dim: Int?
            var index_head_dim: Int? = nil
            let indexer_kv_heads: Int?
            let indexer_n_heads: Int?
            var index_n_heads: Int? = nil
            var index_kpool_compress: Bool? = nil
            var index_kpool_always_select_tail: Bool? = nil
            var index_share_for_mtp_iteration: Bool? = nil
            var indexer_rope_interleave: Bool? = nil
            let hc_count: Int?
            let hc_lowrank: Int?
            var enable_hyper_connections: Bool? = nil
            var hc_mult: Int? = nil
            var hc_sinkhorn_iters: Int? = nil
            var hc_eps: Float? = nil
            let ple_layer_ids: [Int]?
            let ple_embed_dim: Int?
            let ple_conv_kernel_size: Int?
            let ngram_size: Int?
            let heads_per_ngram: Int?
            let ngram_vocab_size_base: Int?
            let make_ngram_vocab_size_divisible_by: Int?
            let split_ngram_parts: Int?
            let mtp_num_hidden_layers: Int?
            var num_nextn_predict_layers: Int? = nil
            let mtp_use_dedicated_embeddings: Bool?
            var mtp_shared_last_layer: Bool? = nil
            var multimax: String? = nil
            var tie_word_embeddings: Bool? = nil
            let eos_token_id: JSONIntOrList
            let intermediate_size: Int?           // dense models / first_k_dense_replace
            var quantization: QWQuantizationSpec? = nil
            var quantization_config: QWQuantizationSpec? = nil
            var loop_settings: RawLoopSettings? = nil
            var mhc: MHCValue? = nil
        }
        let text_config: Text?
        let vision_config: VisionConfig?
        let quantization: QWQuantizationSpec?
        let quantization_config: QWQuantizationSpec?
        let image_token_id: Int?
        let video_token_id: Int?
        let vision_start_token_id: Int?
        let vision_end_token_id: Int?
        let model_type: String?
        let loop_settings: RawLoopSettings?
        let mhc: MHCValue?
    }

}
