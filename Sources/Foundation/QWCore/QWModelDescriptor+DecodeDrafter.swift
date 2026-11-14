// QWModelDescriptor+DecodeDrafter.swift — Drafter and standard HF config decoders.
import Foundation

extension QWModelDescriptor {
    /// Decodes a DFlash2 or DSpine speculative drafter `config.json` with `.speculativeDrafter` role.
    public static func decodeDrafter(configJSON data: Data) throws -> QWModelDescriptor {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw DescriptorError.invariant("drafter config JSON is not an object")
        }
        guard isDrafterJSON(root) else {
            throw DescriptorError.invariant("config JSON is not a speculative drafter model")
        }
        let tc = (root["text_config"] as? [String: Any]) ?? root
        let modelType = (tc["model_type"] as? String) ?? (root["model_type"] as? String) ?? ""
        guard let spec = QWModelFamilyRegistry.shared.spec(forModelType: modelType) else {
            throw QWModelDescriptorError.unsupportedModelType(modelType)
        }
        guard let hidden = tc["hidden_size"] as? Int, hidden > 0,
              let vocab = tc["vocab_size"] as? Int, vocab > 0,
              let numLayers = tc["num_hidden_layers"] as? Int, numLayers > 0,
              let qHeads = tc["num_attention_heads"] as? Int, qHeads > 0 else {
            throw DescriptorError.invariant("drafter config missing core dimensions")
        }
        let kvHeads = max(1, (tc["num_key_value_heads"] as? Int) ?? qHeads)
        let headDim = max(1, (tc["head_dim"] as? Int) ?? (hidden / qHeads))
        let intermediate = (tc["intermediate_size"] as? Int) ?? (hidden * 4)
        let rmsEps = Float((tc["rms_norm_eps"] as? Double) ?? 1e-6)
        let maxPos = (tc["max_position_embeddings"] as? Int) ?? 32768
        let ropeDict = (tc["rope_parameters"] as? [String: Any]) ?? (tc["rope_scaling"] as? [String: Any])
        let ropeTheta = (tc["rope_theta"] as? Double) ?? (ropeDict?["rope_theta"] as? Double) ?? 10000.0
        let partialRotary = (tc["partial_rotary_factor"] as? Double) ?? (ropeDict?["partial_rotary_factor"] as? Double) ?? 1.0
        let eosScalar = (tc["eos_token_id"] as? Int) ?? (tc["eos_token_id"] as? [Int])?.first ?? 0
        let eosList = (tc["eos_token_id"] as? [Int]) ?? [eosScalar]
        let layerTypes = (tc["layer_types"] as? [String]).flatMap { $0.count == numLayers ? $0 : nil }
            ?? [String](repeating: "full_attention", count: numLayers)
        return QWModelDescriptor(
            family: spec.family, modelType: modelType, capabilities: [],
            hidden: hidden, vocab: vocab, denseIntermediate: intermediate,
            numLayers: numLayers, layerTypes: layerTypes, fullAttentionInterval: 1,
            rmsEps: rmsEps, maxPositionEmbeddings: maxPos,
            qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
            attnOutputGate: false, partialRotaryFactor: partialRotary, ropeTheta: ropeTheta,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: nil, expertsPerTok: nil, moeIntermediate: nil,
            sharedExpertIntermediate: nil, expertGateType: nil, hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 0, gdnVHeads: 0, gdnHeadDim: 0, gdnQKVChannels: 0, gdnZChannels: 0, gdnConvKernel: 0,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 0, mtpUseDedicatedEmbeddings: false, eosTokenId: eosScalar,
            mropeSection: (0, 0, 0),
            visDepth: 0, visHidden: 0, visHeads: 0, visIntermediate: 0,
            visPatch: 0, visTemporal: 0, visMerge: 0, visPosEmbeds: 0, visOut: 0,
            imageTokenId: -1, videoTokenId: -1, visionStartTokenId: -1, visionEndTokenId: -1,
            hasQKNorm: (tc["qk_norm"] as? Bool) ?? true,
            usesPlusOneRMSNorm: spec.usesPlusOneRMSNorm,
            hasMropeSection: false, eosTokenIds: eosList, role: .speculativeDrafter)
    }

    /// Decodes a flat HF/MLX config into a descriptor.
    internal static func decodeStandard(_ data: Data) throws -> QWModelDescriptor {
        let c = try QWStandardConfigDecoder.decode(data)
        let layerTypes = [String](repeating: "full_attention", count: c.numLayers)
        var caps: Set<Capability> = []
        if c.numExperts != nil { caps.insert(.moe) }
        if c.mtpLayers > 0 { caps.insert(.mtp) }
        let usesPlusOne = QWModelFamilyRegistry.shared.spec(forModelType: c.modelType)?.usesPlusOneRMSNorm ?? false
        return QWModelDescriptor(
            family: c.family, modelType: c.modelType, capabilities: caps,
            hidden: c.hidden, vocab: c.vocab, denseIntermediate: c.intermediate,
            numLayers: c.numLayers, layerTypes: layerTypes,
            fullAttentionInterval: 1, rmsEps: c.rmsEps,
            maxPositionEmbeddings: c.maxPositionEmbeddings,
            qHeads: c.qHeads, kvHeads: c.kvHeads, headDim: c.headDim,
            attnOutputGate: c.attnOutputGate, partialRotaryFactor: c.partialRotaryFactor, ropeTheta: c.ropeTheta,
            indexerHeads: nil, indexerKvHeads: nil, indexerHeadDim: nil,
            indexerBudget: nil, indexerCompressRatio: nil,
            numExperts: c.numExperts, expertsPerTok: c.expertsPerTok, moeIntermediate: c.moeIntermediate,
            sharedExpertIntermediate: c.sharedExpertIntermediate, expertGateType: nil, hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 0, gdnVHeads: 0, gdnHeadDim: 0, gdnQKVChannels: 0, gdnZChannels: 0, gdnConvKernel: 0,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: c.mtpLayers, mtpUseDedicatedEmbeddings: c.mtpUseDedicatedEmbeddings,
            eosTokenId: c.eosTokenId, mropeSection: (0, 0, 0),
            visDepth: 0, visHidden: 0, visHeads: 0, visIntermediate: 0,
            visPatch: 0, visTemporal: 0, visMerge: 0, visPosEmbeds: 0, visOut: 0,
            imageTokenId: -1, videoTokenId: -1, visionStartTokenId: -1, visionEndTokenId: -1,
            hasQKNorm: c.hasQKNorm,
            usesPlusOneRMSNorm: usesPlusOne,
            quantization: c.quantization, hasMropeSection: false,
            tieWordEmbeddings: c.tieWordEmbeddings, normTopKProb: c.normTopKProb,
            eosTokenIds: c.eosTokenIds, role: .causalLM)
    }
}

/// HF configs sometimes carry eos as a scalar, sometimes as a list.
internal struct JSONIntOrList: Decodable {
    enum E: Error { case shape }
    let value: Int
    let values: [Int]
    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let i = try? c.decode(Int.self) {
            value = i
            values = [i]
            return
        }
        if let a = try? c.decode([Int].self), let f = a.first {
            value = f
            values = a
            return
        }
        throw E.shape
    }
}
