import XCTest
import Foundation
@testable import QWCore

/// A flat HuggingFace / MLX config must decode into the same descriptor the
/// hybrid schema produces, with the values the flat schema omits synthesised.
final class QWStandardConfigTests: XCTestCase {
    private func config(_ extra: [String: Any]) -> Data {
        var base: [String: Any] = [
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 4096,
            "vocab_size": 151936,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 12288,
            "max_position_embeddings": 40960,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "eos_token_id": 151645,
            "tie_word_embeddings": false,
        ]
        for (k, v) in extra { base[k] = v }
        return try! JSONSerialization.data(withJSONObject: base)
    }

    /// The config with one key removed.
    private func config(without key: String) throws -> Data {
        let c = config([:])
        var obj = try JSONSerialization.jsonObject(with: c) as! [String: Any]
        obj.removeValue(forKey: key)
        return try JSONSerialization.data(withJSONObject: obj)
    }

    // MARK: - The synthesised values

    func testStandardQwen3DecodesWithSynthesisedLayerTypes() throws {
        let d = try QWModelDescriptor.decode(configJSON: config([:]))
        XCTAssertEqual(d.numLayers, 36)
        XCTAssertEqual(d.hidden, 4096)
        XCTAssertEqual(d.qHeads, 32)
        XCTAssertEqual(d.kvHeads, 8)
        XCTAssertEqual(d.headDim, 128)              // hidden_size / num_attention_heads
        XCTAssertEqual(d.layerTypes.count, 36)
        XCTAssertTrue(d.layerTypes.allSatisfy { $0 == "full_attention" })
        XCTAssertEqual(d.fullAttentionLayers.count, 36)
        // Every layer is full attention, so the full-attention stride is 1.
        XCTAssertEqual(d.fullAttentionInterval, 1)
        XCTAssertEqual(d.partialRotaryFactor, 1.0)
        XCTAssertNil(d.numExperts)
    }

    func testExplicitHeadDimWins() throws {
        let d = try QWModelDescriptor.decode(configJSON: config(["head_dim": 96]))
        XCTAssertEqual(d.headDim, 96)
    }

    func testKeyValueHeadsDefaultToTheQueryHeads() throws {
        var c = config([:])
        var obj = try JSONSerialization.jsonObject(with: c) as! [String: Any]
        obj.removeValue(forKey: "num_key_value_heads")
        c = try JSONSerialization.data(withJSONObject: obj)
        let d = try QWModelDescriptor.decode(configJSON: c)
        XCTAssertEqual(d.kvHeads, d.qHeads)
    }

    func testAModelWithoutTheOptionalKeysStillDecodes() throws {
        // The optional keys carry a defensible default. The required ones
        // (rms_norm_eps, rope_theta, intermediate_size, the head counts)
        // change the model's numerics or geometry, so they are never guessed.
        var c = config([:])
        var obj = try JSONSerialization.jsonObject(with: c) as! [String: Any]
        for key in ["num_key_value_heads", "head_dim", "partial_rotary_factor"] {
            obj.removeValue(forKey: key)
        }
        c = try JSONSerialization.data(withJSONObject: obj)
        let d = try QWModelDescriptor.decode(configJSON: c)
        XCTAssertEqual(d.numLayers, 36)
        XCTAssertEqual(d.kvHeads, d.qHeads)
        XCTAssertEqual(d.headDim, 128)
        XCTAssertEqual(d.ropeTheta, 1_000_000.0, accuracy: 1.0)
        XCTAssertEqual(d.rmsEps, 1e-6, accuracy: 1e-9)
    }

    func testAModelMissingANumericsKeyFailsLoud() throws {
        // These keys change the model's output or its shape. A config without
        // one is refused, never silently defaulted: a fabricated vocab size or
        // context window is baked into the geometry digest.
        for key in ["rope_theta", "rms_norm_eps", "vocab_size",
                    "max_position_embeddings", "eos_token_id"] {
            XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: try config(without: key)), key)
        }
    }

    func testAModelMissingANumericsKeySaysWhichOne() throws {
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: try config(without: "vocab_size"))) { error in
            XCTAssertTrue("\(error)".contains("vocab_size"), "unexpected error: \(error)")
        }
    }

    func testTheRoPEBaseIsReadFromTheV5RoPEParametersBlock() throws {
        // transformers v5 moved `rope_theta` into `rope_parameters`.
        var c = config([:])
        var obj = try JSONSerialization.jsonObject(with: c) as! [String: Any]
        obj.removeValue(forKey: "rope_theta")
        obj["rope_parameters"] = ["rope_type": "default", "rope_theta": 500000.0]
        c = try JSONSerialization.data(withJSONObject: obj)
        let d = try QWModelDescriptor.decode(configJSON: c)
        XCTAssertEqual(d.ropeTheta, 500000.0, accuracy: 1.0)
    }

    func testUnsupportedRoPEScalingCurveIsRefusedWhileLinearYarnDynamicDecode() throws {
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: config([
            "rope_scaling": ["rope_type": "llama3", "factor": 8.0],
        ])))
        let linear = try QWModelDescriptor.decode(configJSON: config([
            "rope_scaling": ["type": "linear", "factor": 2.0],
        ]))
        XCTAssertEqual(linear.ropeScaling, .linear(factor: 2.0))
        let dynamic = try QWModelDescriptor.decode(configJSON: config([
            "rope_scaling": ["rope_type": "dynamic", "factor": 4.0, "original_max_position_embeddings": 32768],
        ]))
        XCTAssertEqual(dynamic.ropeScaling, .dynamic(factor: 4.0, originalMaxPositionEmbeddings: 32768))
        let yarn = try QWModelDescriptor.decode(configJSON: config([
            "rope_scaling": ["rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768,
                             "beta_fast": 32.0, "beta_slow": 1.0],
        ]))
        XCTAssertEqual(yarn.yarnRopeConfig?.factor, 4.0)
        if case .yarn(let yc)? = yarn.ropeScaling {
            XCTAssertEqual(yc.originalMaxPositionEmbeddings, 32768)
        } else {
            XCTFail("expected RoPEScaling.yarn")
        }
    }

    func testADefaultRoPEScalingIsAccepted() throws {
        let d = try QWModelDescriptor.decode(configJSON: config([
            "rope_scaling": ["rope_type": "default"],
        ]))
        XCTAssertEqual(d.ropeTheta, 1_000_000.0, accuracy: 1.0)
        XCTAssertNil(d.ropeScaling)
    }

    func testAttentionBiasDecodesQKVAndOutputFlags() throws {
        let base = try QWModelDescriptor.decode(configJSON: config([:]))
        XCTAssertFalse(base.hasQKVBias)
        XCTAssertFalse(base.hasAttnOutputBias)

        let biased = try QWModelDescriptor.decode(configJSON: config([
            "attention_bias": true,
        ]))
        XCTAssertTrue(biased.hasQKVBias)
        XCTAssertTrue(biased.hasAttnOutputBias)
        XCTAssertEqual(biased.geometryDigestV2Hex, base.geometryDigestV2Hex)
        XCTAssertNotEqual(biased.geometryDigestV3Hex, base.geometryDigestV3Hex)
    }

    func testMixedDenseMoESchedulesAndSharedExpertsDecode() throws {
        let d = try QWModelDescriptor.decode(configJSON: config([
            "model_type": "qwen3_moe",
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 1536,
            "shared_expert_intermediate_size": 768,
            "mlp_only_layers": [0, 3, 7],
            "first_k_dense_replace": 2,
            "decoder_sparse_step": 2,
        ]))
        XCTAssertEqual(d.sharedExpertIntermediate, 768)
        XCTAssertEqual(d.firstKDenseReplace, 2)
        XCTAssertEqual(d.mlpOnlyLayers, [0, 3, 7])
        XCTAssertEqual(d.decoderSparseStep, 2)
        XCTAssertFalse(d.isMoELayer(0))
        XCTAssertFalse(d.isMoELayer(1))
        XCTAssertFalse(d.isMoELayer(2))
        XCTAssertFalse(d.isMoELayer(3))
        XCTAssertFalse(d.isMoELayer(4))
        XCTAssertTrue(d.isMoELayer(5))
        XCTAssertFalse(d.isMoELayer(7))
    }

    func testActiveSlidingWindowDecodesAcrossStandardModels() throws {
        let mistral = try QWModelDescriptor.decode(configJSON: config([
            "model_type": "mistral",
            "sliding_window": 4096,
        ]))
        XCTAssertEqual(mistral.slidingWindow, 4096)
    }

    func testASlidingWindowTheFamilyTurnsOffIsAccepted() throws {
        let d = try QWModelDescriptor.decode(configJSON: config([
            "model_type": "qwen2",
            "sliding_window": 4096,
            "use_sliding_window": false,
        ]))
        XCTAssertEqual(d.family, .qwen2Dense)
        XCTAssertNil(d.slidingWindow)
    }

    func testASlidingWindowTheFamilyTurnsOnDecodes() throws {
        let d = try QWModelDescriptor.decode(configJSON: config([
            "model_type": "qwen2",
            "sliding_window": 4096,
            "use_sliding_window": true,
        ]))
        XCTAssertEqual(d.slidingWindow, 4096)
        let fullCtx = try QWModelDescriptor.decode(configJSON: config([
            "model_type": "qwen2",
            "sliding_window": 40960,
            "use_sliding_window": true,
        ]))
        XCTAssertNil(fullCtx.slidingWindow)
    }

    func testANullSlidingWindowIsAbsent() throws {
        let d = try QWModelDescriptor.decode(configJSON: config([
            "model_type": "mistral",
            "sliding_window": NSNull(),
        ]))
        XCTAssertEqual(d.family, .mistralDense)
    }

    func testAnActivationTheEngineDoesNotHaveIsRefused() throws {
        // The engine's text MLP is SwiGLU. Another activation would change
        // every number, and a silently ignored key is a model that runs wrong.
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: config([
            "model_type": "llama",
            "hidden_act": "gelu",
        ]))) { error in
            XCTAssertTrue("\(error)".contains("hidden_act"), "the refusal names the feature: \(error)")
        }
        // silu is the activation the engine has.
        XCTAssertNoThrow(try QWModelDescriptor.decode(configJSON: config([
            "model_type": "llama",
            "hidden_act": "silu",
        ])))
    }

    func testAnMLPBiasIsRefused() throws {
        // The sibling of attention_bias, which was already policed.
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: config([
            "model_type": "llama",
            "mlp_bias": true,
        ])))
    }

    func testTiedWordEmbeddingsDecodeAcrossStandardFamilies() throws {
        let cases: [(String, [String: Any])] = [
            ("qwen2", [:]),
            ("qwen3", [:]),
            ("qwen3_moe", ["num_experts": 64, "num_experts_per_tok": 8, "moe_intermediate_size": 768]),
        ]
        for (type, extra) in cases {
            var fields = extra
            fields["model_type"] = type
            fields["tie_word_embeddings"] = true
            let d = try QWModelDescriptor.decode(configJSON: config(fields))
            XCTAssertTrue(d.tieWordEmbeddings, type)
        }
    }

    func testARopeScalingWithoutATypeIsRefused() throws {
        // A factor IS a scaling curve, and the engine runs plain RoPE.
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: config([
            "model_type": "llama",
            "rope_scaling": ["factor": 8.0],
        ])))
    }

    func testACheckpointThatIsNotACausalLMIsRefusedByName() throws {
        // The architecture names the model's HEAD. An embedding, a reranker
        // or a guard checkpoint has another output, and the engine would
        // decode it as a plain LM.
        for arch in ["Qwen3ForEmbedding", "Qwen3ForReranking", "Qwen3ForGuardModel"] {
            XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: config([
                "architectures": [arch],
            ]))) { error in
                XCTAssertTrue("\(error)".contains(arch), "the refusal names it: \(error)")
            }
        }
    }

    func testACausalLMArchitectureIsAccepted() throws {
        for arch in ["Qwen3ForCausalLM", "Qwen2ForCausalLM", "LlamaForCausalLM"] {
            XCTAssertNoThrow(try QWModelDescriptor.decode(configJSON: config([
                "architectures": [arch],
            ])), arch)
        }
    }

    func testACheckpointThatNeedsCustomCodeIsRefused() throws {
        // `auto_map` means the checkpoint runs its own Python. The engine
        // runs the kernels it has.
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: config([
            "auto_map": ["AutoModelForCausalLM": "modeling_qwen3.GuardModel"],
        ]))) { error in
            XCTAssertTrue("\(error)".contains("auto_map"), "the refusal names it: \(error)")
        }
    }

    func testExplicitQKNormTraitOverridesFamilyDefault() throws {
        let off = try QWModelDescriptor.decode(configJSON: config([
            "use_qk_norm": false,
        ]))
        XCTAssertFalse(off.hasQKNorm)
        let on = try QWModelDescriptor.decode(configJSON: config([
            "use_qk_norm": true,
        ]))
        XCTAssertTrue(on.hasQKNorm)
        let llamaOn = try QWModelDescriptor.decode(configJSON: config([
            "model_type": "llama",
            "use_qk_norm": true,
        ]))
        XCTAssertTrue(llamaOn.hasQKNorm)
    }

    func testABiasOnTheQKVProjectionsDecodesQKVBiasOnly() throws {
        let base = try QWModelDescriptor.decode(configJSON: config([:]))
        let biased = try QWModelDescriptor.decode(configJSON: config([
            "qkv_bias": true,
        ]))
        XCTAssertTrue(biased.hasQKVBias)
        XCTAssertFalse(biased.hasAttnOutputBias)
        XCTAssertEqual(biased.geometryDigestV2Hex, base.geometryDigestV2Hex)
        XCTAssertNotEqual(biased.geometryDigestV3Hex, base.geometryDigestV3Hex)

        let unBiased = try QWModelDescriptor.decode(configJSON: config([
            "qkv_bias": false,
        ]))
        XCTAssertFalse(unBiased.hasQKVBias)
        XCTAssertFalse(unBiased.hasAttnOutputBias)
    }

    func testAModelThatAsksForRouterLogitsIsRefused() throws {
        // The engine returns no router logits, so it cannot serve a model
        // that asks for them.
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: config([
            "model_type": "qwen3_moe",
            "moe_intermediate_size": 768,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "output_router_logits": true,
        ]))) { error in
            XCTAssertTrue("\(error)".contains("output_router_logits"),
                          "the refusal names it: \(error)")
        }
    }

    func testALayerTypesListOfFullAttentionIsStandard() throws {
        // A recent transformers re-saves every qwen2 config with the list it
        // computes. All full attention is not a hybrid key.
        let d = try QWModelDescriptor.decode(configJSON: config([
            "model_type": "qwen2",
            "layer_types": [String](repeating: "full_attention", count: 36),
        ]))
        XCTAssertEqual(d.family, .qwen2Dense)
        XCTAssertTrue(d.layerTypes.allSatisfy { $0 == "full_attention" })
    }

    func testASlidingLayerIsRefusedByName() throws {
        // The hybrid decoder would ask for a key the user never wrote; the
        // refusal must name the feature instead.
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: config([
            "model_type": "qwen2",
            "layer_types": [String](repeating: "sliding_attention", count: 36),
        ]))) { error in
            XCTAssertTrue("\(error)".contains("layer_types"),
                          "the refusal names the feature: \(error)")
        }
    }

    func testAnExpertCountOnADenseFamilyIsRefused() throws {
        // The plan would name the layer MoE and refuse it much later.
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: config([
            "model_type": "llama",
            "num_experts": 8,
        ])))
    }

    func testAnEosTokenIDListKeepsEveryEntry() throws {
        // llama-3.1 ships `eos_token_id` as a LIST, and the chat turn ends
        // with the LAST one — dropping it would never stop the generation.
        let d = try QWModelDescriptor.decode(configJSON: config([
            "eos_token_id": [128001, 128008, 128009],
        ]))
        XCTAssertEqual(d.eosTokenId, 128001)
        XCTAssertEqual(d.eosTokenIds, [128001, 128008, 128009])
        XCTAssertTrue(d.specialTokens.defaultStopTokenIds.contains(128009))
    }

    // MARK: - The gate and the norms

    func testQwen3HasQKNormAndNoOutputGate() throws {
        let d = try QWModelDescriptor.decode(configJSON: config([:]))
        // Qwen3 carries q_norm/k_norm and no attention output gate.
        XCTAssertTrue(d.hasQKNorm)
        XCTAssertFalse(d.attnOutputGate)
    }

    func testALlamaConfigHasNeitherQKNormNorAnOutputGate() throws {
        let d = try QWModelDescriptor.decode(configJSON: config(["model_type": "llama"]))
        XCTAssertFalse(d.hasQKNorm)
        XCTAssertFalse(d.attnOutputGate)
    }

    func testAnExplicitOutputGateIsHonoured() throws {
        let d = try QWModelDescriptor.decode(configJSON: config(["output_gate_type": "sigmoid"]))
        XCTAssertTrue(d.attnOutputGate)
    }

    func testTheFamilyDefaultTurnsTheGateOnWhenTheConfigIsSilent() throws {
        // The same rule as the hybrid path: the family's default applies when
        // the config says nothing about the gate.
        let registry = QWModelFamilyRegistry.shared
        registry.register(QWModelFamilySpec(modelType: "qwen3_gated",
                                            family: .qwen3Dense,
                                            defaultAttnOutputGate: true))
        defer { registry.unregister(modelType: "qwen3_gated") }
        let d = try QWModelDescriptor.decode(configJSON: config([
            "model_type": "qwen3_gated",
        ]))
        XCTAssertTrue(d.attnOutputGate)
        XCTAssertEqual(d.qProjOut, 2 * d.qHeads * d.headDim)
    }

    func testAnExplicitFalseGateWinsOverTheFamilyDefault() throws {
        let registry = QWModelFamilyRegistry.shared
        registry.register(QWModelFamilySpec(modelType: "qwen3_ungated",
                                            family: .qwen3Dense,
                                            defaultAttnOutputGate: true))
        defer { registry.unregister(modelType: "qwen3_ungated") }
        let d = try QWModelDescriptor.decode(configJSON: config([
            "model_type": "qwen3_ungated",
            "attn_output_gate": false,
        ]))
        XCTAssertFalse(d.attnOutputGate)
        XCTAssertEqual(d.qProjOut, d.qHeads * d.headDim)
    }

    // MARK: - MoE

    func testStandardMoEConfigCarriesItsExpertGeometry() throws {
        let d = try QWModelDescriptor.decode(configJSON: config([
            "model_type": "qwen3_moe",
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 1536,
            "norm_topk_prob": true,
        ]))
        XCTAssertEqual(d.numExperts, 128)
        XCTAssertEqual(d.expertsPerTok, 8)
        XCTAssertEqual(d.moeIntermediate, 1536)
        XCTAssertTrue(d.capabilities.contains(.moe))
    }

    // MARK: - The facts the decoder must not drop

    func testTheDecoderReadsEveryKeyThatChangesTheModel() throws {
        // These are implemented later (Task 4.3 owns the module), but the
        // decoder must still REPORT them and reach the descriptor: a
        // silently dropped key is a model that runs with the wrong numerics.
        let d = try QWModelDescriptor.decode(configJSON: config([
            "model_type": "qwen3_moe",
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 1536,
            "norm_topk_prob": true,
        ]))
        XCTAssertTrue(d.normTopKProb)
    }

    func testAConfigWithoutThoseKeysReportsThemAbsent() throws {
        let d = try QWModelDescriptor.decode(configJSON: config([:]))
        XCTAssertFalse(d.normTopKProb)
        XCTAssertFalse(d.tieWordEmbeddings)
        XCTAssertNil(d.quantization)
    }

    // MARK: - The quantisation block

    func testQuantizationConfigIsParsed() throws {
        let d = try QWModelDescriptor.decode(configJSON: config([
            "quantization": ["quant_method": "mlx", "bits": 4, "group_size": 64],
        ]))
        XCTAssertEqual(d.quantization?.method, "mlx")
        XCTAssertEqual(d.quantization?.bits, 4)
        XCTAssertEqual(d.quantization?.groupSize, 64)
    }

    func testTheQuantizationConfigAliasIsAlsoAccepted() throws {
        let d = try QWModelDescriptor.decode(configJSON: config([
            "quantization_config": ["quant_method": "mlx", "bits": 4, "group_size": 64],
        ]))
        XCTAssertEqual(d.quantization?.bits, 4)
    }

    func testACheckpointWithoutAQuantizationBlockHasNone() throws {
        let d = try QWModelDescriptor.decode(configJSON: config([:]))
        XCTAssertNil(d.quantization)
    }

    // MARK: - The mrope invariant

    func testStandardConfigHasNoMropeSection() throws {
        let d = try QWModelDescriptor.decode(configJSON: config([:]))
        XCTAssertEqual(d.mropeW + d.mropeH + d.mropeT, 0)
        XCTAssertEqual(d.verifyInternalConsistency(), [])
    }

    // MARK: - Failure

    func testUnknownModelTypeFailsLoud() throws {
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: config([
            "model_type": "definitely_not_a_model",
        ]))) { error in
            XCTAssertEqual(error as? QWModelDescriptorError,
                           .unsupportedModelType("definitely_not_a_model"))
        }
    }

    func testAGeometryThatCannotBeDerivedFailsLoud() throws {
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: config([
            "hidden_size": 4001,        // not divisible by 32 heads
            "num_attention_heads": 32,
        ]))) { error in
            XCTAssertTrue("\(error)".contains("hidden_size 4001 is not divisible by num_attention_heads 32"),
                          "unexpected error: \(error)")
        }
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: config([
            "head_dim": 512,
        ]))) { error in
            XCTAssertTrue("\(error)".contains("512"), "expected >256 refusal: \(error)")
        }
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: config([
            "head_dim": 100,
        ]))) { error in
            XCTAssertTrue("\(error)".contains("32"), "expected %32 refusal: \(error)")
        }
    }

    func testAnExplicitHeadDimIsNotOverriddenByTheDivisibilityGuard() throws {
        // Some models declare a head_dim that is not hidden / qHeads.
        let d = try QWModelDescriptor.decode(configJSON: config([
            "hidden_size": 2880,
            "num_attention_heads": 64,
            "head_dim": 64,
        ]))
        XCTAssertEqual(d.headDim, 64)
        XCTAssertEqual(d.qProjOut, 64 * 64)
    }

    // MARK: - The hybrid schema is untouched

    func testTheHybridSchemaStillTakesItsOwnPath() throws {
        // A hybrid config carries `text_config`; it must NOT be routed to the
        // standard decoder. A DecodingError proves the HYBRID envelope ran.
        let hybrid = try JSONSerialization.data(withJSONObject: [
            "model_type": "qwen3_5",
            "text_config": ["model_type": "qwen3_5"],
        ])
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: hybrid)) { error in
            XCTAssertTrue(error is DecodingError, "expected the hybrid envelope, got \(error)")
        }
    }

    func testAHybridConfigThatLostItsEnvelopeIsNotReadAsStandard() throws {
        // The hybrid keys are the marker. A config that kept them but lost
        // its `text_config` must fail in the hybrid decoder, not be read as
        // a dense standard model with no GDN, no MTP and no mrope.
        let hybrid = try JSONSerialization.data(withJSONObject: [
            "model_type": "qwen3_5",
            "hidden_size": 5120,
            "vocab_size": 248320,
            "num_hidden_layers": 64,
            "full_attention_interval": 4,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 17408,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262144,
            "eos_token_id": 248044,
            "rope_parameters": ["rope_type": "default", "rope_theta": 1e7,
                                "partial_rotary_factor": 0.25,
                                "mrope_section": [11, 11, 10]],
        ])
        XCTAssertFalse(QWStandardConfigDecoder.isStandardSchema(hybrid))
        XCTAssertThrowsError(try QWModelDescriptor.decode(configJSON: hybrid)) { error in
            // The hybrid decoder's own refusal — not the standard decoder's.
            XCTAssertTrue("\(error)".contains("layer_types"),
                          "expected the hybrid envelope, got \(error)")
        }
    }

    func testAJSONNullTextConfigIsTreatedAsAbsent() throws {
        let obj = try JSONSerialization.jsonObject(with: config([:])) as! [String: Any]
        var withNull = obj
        withNull["text_config"] = NSNull()
        let data = try JSONSerialization.data(withJSONObject: withNull)
        XCTAssertTrue(QWStandardConfigDecoder.isStandardSchema(data))
    }
}
