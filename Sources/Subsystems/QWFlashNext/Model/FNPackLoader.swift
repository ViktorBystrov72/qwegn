import Foundation
import QWCore
import QWQuant

// Pack-backed expert source: each expert is dequantized on demand from the HQ4 expert stacks. The pack is
// mmap-backed and read-only, so concurrent reads are safe; a dequant failure here is a loader bug, not input.
// Names follow the pack: `model.language_model.layers.N.*`.
public struct FNPackLoader {
    public let pack: QWQuant.Pack?
    public let st: QWSafeTensorsContainer?
    public let descriptor: QWModelDescriptor
    public let isMLXCheckpoint: Bool

    static let prefix = "model.language_model."

    public init(pack: QWQuant.Pack, descriptor: QWModelDescriptor) {
        self.pack = pack
        self.st = nil
        self.descriptor = descriptor
        self.isMLXCheckpoint = false
    }

    public init(st: QWSafeTensorsContainer, descriptor: QWModelDescriptor) {
        self.pack = nil
        self.st = st
        self.descriptor = descriptor
        self.isMLXCheckpoint = st.isMLXCheckpoint
    }


    public func layerPrefix(_ layer: Int) -> String { "\(Self.prefix)layers.\(layer)." }

    /// Dense fp32 tensor: bf16 raw entries verbatim, HQ4 2-D entries or SafeTensors entries dequantized.
    public func tensor(_ name: String) throws -> [Float] {
        if let pack {
            if try pack.entry(name).isRawBf16 { return try pack.rawBf16(name) }
            return try pack.dequantQ4g(name)
        }
        guard let st else { throw QWSafeTensorsError.missingTensor(name) }
        var vals = try st.readFloats(name)
        if isMLXCheckpoint && name.contains("norm") && !name.hasSuffix("linear_attn.norm.weight") {
            for i in 0..<vals.count { vals[i] -= 1.0 }
        }
        return vals
    }


    public func pleEmbedding(_ layer: Int, pleRows: (any FNNgramRows)?) -> FNNgramEmbedding? {
        guard let rows = pleRows,
              let index = descriptor.pleModuleIndices.firstIndex(of: layer),
              let ngram = descriptor.ngramSize,
              let headsPerNgram = descriptor.headsPerNgram,
              let baseVocab = descriptor.ngramVocabSizeBase else { return nil }
        return FNNgramEmbedding(
            vocab: descriptor.vocab, ngram: ngram, headsPerNgram: headsPerNgram,
            baseVocab: baseVocab, seed: Self.ngramSeed,
            eos: Int64(descriptor.eosTokenId), pleLayerIndex: index, rows: rows)
    }


    /// N-gram seed of the PLE hashing: absent from the assets config; 1234 is what the pack's router constants use.
    public static let ngramSeed = 1234

}
