// QWDecisionModelConfig.swift — 5-axis capability config for generalized decision models.
import Foundation

/// Short 1D depthwise gated convolution capability used across LFM2, LFM3, and LFM2-Encoder blocks.
public struct ShortConvConfig: Sendable, Hashable, Codable, Equatable {
    public let kernelSize: Int
    public let hasBias: Bool
    public let isCentered: Bool
    public let stateKey: String?

    public init(kernelSize: Int, hasBias: Bool = false, isCentered: Bool = false, stateKey: String? = nil) {
        self.kernelSize = kernelSize
        self.hasBias = hasBias
        self.isCentered = isCentered
        self.stateKey = stateKey
    }
}

/// Normalization functional form decoupled from model family names.
public enum NormKind: Sendable, Hashable, Codable, Equatable {
    case rmsNorm(eps: Float, hasUnitOffset: Bool)
    case layerNorm(eps: Float, hasBias: Bool)

    public var eps: Float {
        switch self {
        case .rmsNorm(let eps, _): return eps
        case .layerNorm(let eps, _): return eps
        }
    }
}

/// Per-layer-kind rotary embedding geometry.
public struct PerKindRoPEEntry: Sendable, Hashable, Codable, Equatable {
    public let ropeTheta: Double
    public let partialRotaryFactor: Double
    public let ropeType: String

    public init(ropeTheta: Double, partialRotaryFactor: Double = 1.0, ropeType: String = "default") {
        self.ropeTheta = ropeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.ropeType = ropeType
    }
}

/// Layer-kind-indexed RoPE schedule (e.g., full_attention vs sliding_attention).
public struct PerKindRoPEConfig: Sendable, Hashable, Codable, Equatable {
    public let byLayerKind: [String: PerKindRoPEEntry]
    public let zeroTailPartialRotation: Bool

    public init(byLayerKind: [String: PerKindRoPEEntry], zeroTailPartialRotation: Bool = false) {
        self.byLayerKind = byLayerKind
        self.zeroTailPartialRotation = zeroTailPartialRotation
    }

    public func entry(forLayerKind kind: String) -> PerKindRoPEEntry? {
        byLayerKind[kind]
    }
}

/// Attention output gating projection layout.
public enum AttnGateMode: String, Sendable, Hashable, Codable, Equatable {
    case none
    case packedElementWiseQProj
    case separateHeadWiseSigmoid
}

/// 5-axis capability descriptor for System-1 decision models (`specs/07`).
public struct QWDecisionModelConfig: Sendable, Hashable, Codable, Equatable {
    public enum MarkerLayout: String, Sendable, Hashable, Codable, Equatable {
        case questionOptionsState
        case stateQuestionOptions
    }

    public enum SequenceFraming: Sendable, Hashable, Codable, Equatable {
        case turnVerbalizer(
            codesPerOption: Int,
            numbers: Bool,
            packSet: Bool,
            maxSetSize: Int,
            colonSlots: Bool,
            stateSeparator: String,
            assistantLead: String = "",
            chatML: Bool = false
        )
        case optionMarkers(
            layout: MarkerLayout,
            markerToken: String,
            bosToken: String?,
            maxLength: Int,
            imageTextLength: Int?,
            audioTextLength: Int?
        )
        case sequencePair(maxLength: Int)
    }

    public enum AttentionTopology: Sendable, Hashable, Codable, Equatable {
        case causal(supportsPrefixCache: Bool)
        case prefixBidirectional
        case promptBidirectional
        case packedSet(maxSetSize: Int)
        case bidirectionalEncoder(isolateMediaPrefix: Bool)
    }

    public enum PoolingKind: String, Sendable, Hashable, Codable, Equatable {
        case firstToken
        case lastToken
        case meanNonPad
    }

    public enum HeadKind: Sendable, Hashable, Codable, Equatable {
        case vocabVerbalizer(vocabSize: Int, tieEmbeddings: Bool, sparseGatherThreshold: Int)
        case markerScorer(hiddenSize: Int, headLayers: Int, chains: [String])
        case pooledClassifier(pooling: PoolingKind, numLabels: Int, id2label: [Int: String])
        case jointOptionProjector(hiddenSize: Int)
    }

    public let framing: SequenceFraming
    public let topology: AttentionTopology
    public let head: HeadKind
    public let hasStatePointer: Bool
    public let hasGridReadout: Bool
    public let usesNumbersThousandthsGrid: Bool
    public let boxEdgesMode: String
    public let visionMaxPixels: Int?
    public let inlineTemperatures: [String: Double]
    public let defaultScoreReadout: String

    public init(
        framing: SequenceFraming,
        topology: AttentionTopology,
        head: HeadKind,
        hasStatePointer: Bool = false,
        hasGridReadout: Bool = false,
        usesNumbersThousandthsGrid: Bool = false,
        boxEdgesMode: String = "median",
        visionMaxPixels: Int? = nil,
        inlineTemperatures: [String: Double] = [:],
        defaultScoreReadout: String = "mean"
    ) {
        self.framing = framing
        self.topology = topology
        self.head = head
        self.hasStatePointer = hasStatePointer
        self.hasGridReadout = hasGridReadout
        self.usesNumbersThousandthsGrid = usesNumbersThousandthsGrid
        self.boxEdgesMode = boxEdgesMode
        self.visionMaxPixels = visionMaxPixels
        self.inlineTemperatures = inlineTemperatures
        self.defaultScoreReadout = defaultScoreReadout
    }

    /// Synthesizes a standard causal verbalizer capability bundle for any generative LLM pack.
    public static func synthesizedVerbalizer(
        vocabSize: Int,
        tieWordEmbeddings: Bool = false,
        supportsPrefixCache: Bool = true
    ) -> QWDecisionModelConfig {
        QWDecisionModelConfig(
            framing: .turnVerbalizer(
                codesPerOption: 1,
                numbers: false,
                packSet: false,
                maxSetSize: 1,
                colonSlots: false,
                stateSeparator: "\n\n",
                assistantLead: "",
                chatML: false
            ),
            topology: .causal(supportsPrefixCache: supportsPrefixCache),
            head: .vocabVerbalizer(vocabSize: vocabSize, tieEmbeddings: tieWordEmbeddings, sparseGatherThreshold: 256),
            hasStatePointer: false,
            hasGridReadout: false,
            usesNumbersThousandthsGrid: false,
            boxEdgesMode: "median",
            visionMaxPixels: nil,
            inlineTemperatures: [:],
            defaultScoreReadout: "mean"
        )
    }

    public var topologyDigestToken: String {
        switch topology {
        case .causal(let cached): return "causal(\(cached ? 1 : 0))"
        case .prefixBidirectional: return "prefix_bi"
        case .promptBidirectional: return "prompt_bi"
        case .packedSet(let n): return "packed_set(\(n))"
        case .bidirectionalEncoder(let iso): return "bi_enc(\(iso ? 1 : 0))"
        }
    }

    public var headDigestToken: String {
        let base: String
        switch head {
        case .vocabVerbalizer(let v, let tie, _):
            base = "vocab(\(v),\(tie ? 1 : 0))"
        case .markerScorer(let h, let l, let chains):
            base = "marker(\(h),\(l),\(chains.sorted().joined(separator: "+")))"
        case .pooledClassifier(let p, let n, _):
            base = "pooled(\(p.rawValue),\(n))"
        case .jointOptionProjector(let h):
            base = "joint(\(h))"
        }
        return "\(base),ptr=\(hasStatePointer ? 1 : 0),grid=\(hasGridReadout ? 1 : 0),num=\(usesNumbersThousandthsGrid ? 1 : 0)"
    }
}
