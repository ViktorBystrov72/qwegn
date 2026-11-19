// FNPackLoader+Reference.swift — CPU-only FlashNext model builder for tests.

import Foundation
import QWCore
import QWRope
@testable import QWQuant
@testable import QWFlashNext

// mmap-backed and read-only, so concurrent reads are safe; a dequant failure here is a loader bug, not input.
public final class FNPackExperts: FNLatentExpertSource, @unchecked Sendable {
    let pack: QWQuant.Pack
    let gateUpName: String
    let downName: String
    let latentDName: String?
    let latentUName: String?

    public init(pack: QWQuant.Pack, gateUpName: String, downName: String,
                latentDName: String? = nil, latentUName: String? = nil) {
        self.pack = pack; self.gateUpName = gateUpName; self.downName = downName
        self.latentDName = latentDName; self.latentUName = latentUName
    }

    private func dequantChecked(_ name: String?, expert: Int, label: String) -> [Float] {
        guard let name else { preconditionFailure("FNPackExperts: missing \(label) tensor name") }
        do {
            return try pack.dequantExpertQ4g(name, expert: expert)
        } catch {
            preconditionFailure("FNPackExperts: failed \(label)[\(expert)] from '\(name)': \(error)")
        }
    }

    public func gateUp(_ e: Int) -> [Float] {
        // Geometry is validated when the pack is opened; a failure here indicates pack corruption.
        dequantChecked(gateUpName, expert: e, label: "gateUp")
    }

    public func down(_ e: Int) -> [Float] {
        dequantChecked(downName, expert: e, label: "down")
    }

    public func latentD(_ g: Int) -> [Float] {
        dequantChecked(latentDName, expert: g, label: "latentD")
    }

    public func latentU(_ g: Int) -> [Float] {
        dequantChecked(latentUName, expert: g, label: "latentU")
    }
}

// Loads FlashNext decoder layers from a QWP1 HQ4 pack or a SafeTensors container.

extension FNPackLoader {
    private func requireField<T>(_ value: T?, _ field: String) throws -> T {
        guard let value else {
            throw QWSafeTensorsError.badHeader("FNPackLoader: missing \(field) in model descriptor")
        }
        return value
    }

    public func hyperConnection(prefix: String, withInjection: Bool) throws -> FNHyperConnection {
        let hc = try requireField(descriptor.hcCount, "hc_count")
        let lowrank = try requireField(descriptor.hcLowrank, "hc_lowrank")
        return FNHyperConnection(
            hc: hc, hidden: descriptor.hidden, lowrank: lowrank, eps: descriptor.rmsEps,
            normWeight: try tensor(prefix + "hc_norm.weight"),
            downWeight: try tensor(prefix + "input_mix_weight_down.weight"),
            upWeight: try tensor(prefix + "input_mix_weight_up.weight"),
            injectWeight: withInjection ? try tensor(prefix + "block_inject_weight.weight") : [])
    }

    public func gatedDeltaNet(layer: Int) throws -> FNGatedDeltaNet {
        let p = layerPrefix(layer) + "linear_attn."
        let d = descriptor
        return FNGatedDeltaNet(
            hidden: d.hidden, kHeads: d.gdnQKHeads, vHeads: d.gdnVHeads, kDim: d.gdnHeadDim, vDim: d.gdnHeadDim,
            convKernel: d.gdnConvKernel, eps: d.rmsEps,
            inProjQKV: try tensor(p + "in_proj_qkv.weight"), inProjZ: try tensor(p + "in_proj_z.weight"),
            inProjB: try tensor(p + "in_proj_b.weight"), inProjA: try tensor(p + "in_proj_a.weight"),
            convWeight: try tensor(p + "conv1d.weight"), dtBias: try tensor(p + "dt_bias"),
            aLog: try tensor(p + "A_log"), normWeight: try tensor(p + "norm.weight"),
            outProj: try tensor(p + "out_proj.weight"))
    }

    /// `indexerBudget` overrides the descriptor's budget (real-weight tests only; nil = descriptor value).
    public func fullAttention(layer: Int, indexerBudget: Int? = nil, prefix: String? = nil) throws -> FNAttention {
        let p = (prefix ?? layerPrefix(layer)) + "self_attn."
        let ix = p + "indexer."
        let d = descriptor
        let indexer = FNQSAIndexer(
            heads: try requireField(d.indexerHeads, "indexer_heads"),
            kvHeads: try requireField(d.indexerKvHeads, "indexer_kv_heads"),
            headDim: try requireField(d.indexerHeadDim, "indexer_head_dim"),
            budget: try indexerBudget ?? requireField(d.indexerBudget, "indexer_budget"),
            compress: try requireField(d.indexerCompressRatio, "indexer_compress_ratio"),
            eps: d.rmsEps,
            qkProj: try tensor(ix + "index_qk_proj.weight"),
            qNorm: try tensor(ix + "q_layernorm.weight"), kNorm: try tensor(ix + "k_layernorm.weight"))
        return FNAttention(
            hidden: d.hidden, heads: d.qHeads, kvHeads: d.kvHeads, headDim: d.headDim, eps: d.rmsEps,
            qProj: try tensor(p + "q_proj.weight"), kProj: try tensor(p + "k_proj.weight"),
            vProj: try tensor(p + "v_proj.weight"), oProj: try tensor(p + "o_proj.weight"),
            qNorm: try tensor(p + "q_norm.weight"), kNorm: try tensor(p + "k_norm.weight"),
            indexer: indexer, rope: QWRopeCPU(rotaryDim: d.rotaryDims, theta: d.ropeTheta))
    }

    public func moe(layer: Int, normTopK: Bool, prefix: String? = nil) throws -> FNSparseMoE {
        guard let pack else {
            throw QWSafeTensorsError.badHeader("FNPackLoader: MoE experts require a quantized pack")
        }
        let m = (prefix ?? layerPrefix(layer)) + "mlp."
        let d = descriptor
        var latentGroups = 0, latentDim = 0, latentLabels: [Int] = []
        var latentDName: String? = nil, latentUName: String? = nil
        if let dEntry = try? pack.entry(m + "experts.latent_d"),
           dEntry.shape.count == 3 {
            latentGroups = dEntry.shape[0]
            latentDim = dEntry.shape[1]
            latentLabels = try tensor(m + "latent_labels").map { Int($0.rounded()) }
            latentDName = m + "experts.latent_d"
            latentUName = m + "experts.latent_u"
        }
        let src = FNPackExperts(pack: pack, gateUpName: m + "experts.gate_up_proj",
                                downName: m + "experts.down_proj",
                                latentDName: latentDName, latentUName: latentUName)
        return FNSparseMoE(
            hidden: d.hidden,
            experts: try requireField(d.numExperts, "num_experts"),
            topK: try requireField(d.expertsPerTok, "experts_per_tok"),
            intermediate: try requireField(d.moeIntermediate, "moe_intermediate_size"),
            sharedIntermediate: try requireField(d.sharedExpertIntermediate, "shared_expert_intermediate_size"),
            normTopK: normTopK,
            gate: try tensor(m + "gate.weight"),
            source: src,
            sharedGate: try tensor(m + "shared_expert.gate_proj.weight"),
            sharedUp: try tensor(m + "shared_expert.up_proj.weight"),
            sharedDown: try tensor(m + "shared_expert.down_proj.weight"),
            sharedGateScalar: try tensor(m + "shared_expert_gate.weight"),
            latentGroups: latentGroups, latentDim: latentDim, latentLabels: latentLabels,
            latentSource: latentDName != nil ? src : nil)
    }

    /// PLE layer attached to decoder layer `layer` (dims from the descriptor; dilation = ngram size, as in HF).
    public func ple(layer: Int) throws -> FNPLE {
        let p = layerPrefix(layer) + "ple."
        let d = descriptor
        return FNPLE(
            hidden: d.hidden,
            hc: try requireField(d.hcCount, "hc_count"),
            embedDim: try requireField(d.pleEmbedDim, "ple_embed_dim"),
            convKernel: try requireField(d.pleConvKernel, "ple_conv_kernel"),
            dilation: try requireField(d.ngramSize, "ngram_size"),
            eps: d.rmsEps,
            keyProj: try tensor(p + "key_proj.weight"), valueProj: try tensor(p + "value_proj.weight"),
            normKey: try tensor(p + "norm_key.weight"), normQuery: try tensor(p + "norm_query.weight"),
            normConv: try tensor(p + "norm_conv.weight"), convWeight: try tensor(p + "conv1d.weight"))
    }

    /// Text decoder layer `layer` (or MTP layer when `prefix` is supplied).
    public func decoderLayer(_ layer: Int, normTopK: Bool, pleRows: (any FNNgramRows)? = nil,
                             prefix: String? = nil) throws -> FNDecoderLayer {
        let isFull = (prefix != nil) || descriptor.isFullAttention(layer)
        let mixer: FNMixer = isFull
            ? .full(try fullAttention(layer: layer, prefix: prefix))
            : .linear(try gatedDeltaNet(layer: layer))
        let p = prefix ?? layerPrefix(layer)
        var pleLayer: FNPLE? = nil
        var pleEmb: FNNgramEmbedding? = nil
        if prefix == nil, let emb = pleEmbedding(layer, pleRows: pleRows) {
            pleLayer = try ple(layer: layer)
            pleEmb = emb
        }
        return FNDecoderLayer(
            mixer: mixer,
            attnHC: try hyperConnection(prefix: p + "attn_hyper_connection.", withInjection: true),
            mlpHC: try hyperConnection(prefix: p + "mlp_hyper_connection.", withInjection: true),
            moe: try moe(layer: layer, normTopK: normTopK, prefix: prefix),
            ple: pleLayer, pleEmbedding: pleEmb)
    }

    /// N-gram seed of the PLE hashing: absent from the assets config; 1234 is what the pack's router constants use.

    /// Full text model: embedding, every decoder layer, the final mixer and lm_head.
    public func model(normTopK: Bool, pleRows: any FNNgramRows) throws -> FNModel {
        let d = descriptor
        var layers: [FNDecoderLayer] = []
        for i in 0..<d.numLayers {
            layers.append(try decoderLayer(i, normTopK: normTopK, pleRows: pleRows))
        }
        return FNModel(
            vocab: d.vocab, hidden: d.hidden,
            hc: try requireField(d.hcCount, "hc_count"),
            embedding: try tensor(Self.prefix + "embed_tokens.weight"), layers: layers,
            finalMixer: try hyperConnection(prefix: Self.prefix + "hyper_connection_mixer.", withInjection: false),
            lmHead: try tensor("lm_head.weight"))
    }
}

private extension QWQuant.Pack {
    func dequantExpertQ4g(_ name: String, expert: Int) throws -> [Float] {
        let e = try entry(name)
        guard let hq4 = e.hq4Off, let count = e.experts, let per = e.eHq4, e.shape.count == 3 else {
            throw QWQuantError.notQuantPlane(name)
        }
        let rows = e.shape[1], cols = e.shape[2]
        let grp = e.hqGroup, blkBytes = e.hqBlockBytes
        guard (0..<count).contains(expert), cols % grp == 0,
              per == rows * (cols / grp) * blkBytes else {
            throw QWQuantError.notQuantPlane(name)
        }
        let start = hq4 + expert * per
        guard start + per <= rawCount else { throw QWQuantError.truncated }
        let ptr = UnsafeRawPointer(rawBase).advanced(by: start)
        if e.isIncumbentHQ4 {
            return Self.dequantHQ4Slab(base: ptr, nRows: rows, kCols: cols)
        }
        return Self.dequantHQBlockSlab(
            base: ptr, nRows: rows, kCols: cols, bits: e.hqBits, group: grp, isSHVQ: e.isSHVQ
        )
    }
}
