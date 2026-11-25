// QWModelDescriptor+Capabilities.swift — Config-driven capability structs for L1-L7.
import Foundation

/// Per-layer SwiGLU clamp configuration specialized via HAL function constants.
public struct QWSwiGLUClampSpec: Sendable, Hashable, Codable, Equatable {
    public let limit: Float
    public let postSilu: Bool

    public init(limit: Float, postSilu: Bool = false) {
        self.limit = limit
        self.postSilu = postSilu
    }
}

extension QWModelDescriptor {

    /// Spatial/temporal MRoPE rotary axis layout (`rope_parameters` / model config).
    public enum MRoPELayout: String, Sendable, Hashable, Codable, Equatable {
        case none1D
        case qwenInterleaved
        case bailingVideoRope
    }

    /// Config-sourced GDN / KDA linear-attention parameters.
    public struct KDAConfig: Sendable, Hashable, Codable, Equatable {
        public enum GateProjection: Sendable, Hashable, Codable, Equatable {
            case lowRank(dim: Int)
            case fullRank
        }

        public enum NormActivation: String, Sendable, Hashable, Codable, Equatable {
            case silu
            case sigmoid
        }

        public let qkHeads: Int
        public let vHeads: Int
        public let headDim: Int
        public let convKernel: Int
        public let gateProjection: GateProjection
        public let safeGate: Bool
        public let gateLowerBound: Float?
        public let outputNormAct: NormActivation
        public let separateConv: Bool

        public init(
            qkHeads: Int,
            vHeads: Int,
            headDim: Int,
            convKernel: Int,
            gateProjection: GateProjection,
            safeGate: Bool = false,
            gateLowerBound: Float? = nil,
            outputNormAct: NormActivation = .sigmoid,
            separateConv: Bool = true
        ) {
            self.qkHeads = qkHeads
            self.vHeads = vHeads
            self.headDim = headDim
            self.convKernel = convKernel
            self.gateProjection = gateProjection
            self.safeGate = safeGate
            self.gateLowerBound = gateLowerBound
            self.outputNormAct = outputNormAct
            self.separateConv = separateConv
        }

        public var lowRankDim: Int? {
            if case .lowRank(let dim) = gateProjection { return dim }
            return nil
        }

        public var isFullRank: Bool {
            if case .fullRank = gateProjection { return true }
            return false
        }
    }

    /// Rotary parameters bound to an MLA attention block.
    public struct MLARopeConfig: Sendable, Hashable, Codable, Equatable {
        public let dim: Int
        public let theta: Double
        public let interleave: Bool
        public let mropeLayout: MRoPELayout
        public let yarn: YarnRopeConfig?

        public init(
            dim: Int,
            theta: Double,
            interleave: Bool = false,
            mropeLayout: MRoPELayout = .none1D,
            yarn: YarnRopeConfig? = nil
        ) {
            self.dim = dim
            self.theta = theta
            self.interleave = interleave
            self.mropeLayout = mropeLayout
            self.yarn = yarn
        }
    }

    /// Unified config-sourced Multi-head Latent Attention traits.
    public struct MLAConfig: Sendable, Hashable, Codable, Equatable {
        public enum QProjection: Sendable, Hashable, Codable, Equatable {
            case factored(rank: Int)
            case direct
        }

        public enum Sparsity: Sendable, Hashable, Codable, Equatable {
            case dense(slidingWindow: Int)
            case keyPool
            case dsa(slidingWindow: Int)
        }

        public enum OutputGateMode: String, Sendable, Hashable, Codable, Equatable {
            case none
            case elementWiseFromQLora
            case headWiseFromHidden
            case elementWiseFromHidden
        }

        public let qHeads: Int
        public let qProjection: QProjection
        public let kvLoraRank: Int
        public let qkNopeHeadDim: Int
        public let qkRopeHeadDim: Int
        public let vHeadDim: Int
        public let rope: MLARopeConfig?
        public let sparsity: Sparsity
        public let outputGateMode: OutputGateMode
        public let useVHA: Bool
        public let hasAttnSinkBias: Bool
        public let splitKvBProj: Bool
        public let indexerUseFastHadamard: Bool

        public init(
            qHeads: Int,
            qProjection: QProjection,
            kvLoraRank: Int,
            qkNopeHeadDim: Int,
            qkRopeHeadDim: Int,
            vHeadDim: Int,
            rope: MLARopeConfig? = nil,
            sparsity: Sparsity = .dense(slidingWindow: 0),
            outputGateMode: OutputGateMode = .none,
            useVHA: Bool = false,
            hasAttnSinkBias: Bool = false,
            splitKvBProj: Bool = false,
            indexerUseFastHadamard: Bool = false
        ) {
            self.qHeads = qHeads
            self.qProjection = qProjection
            self.kvLoraRank = kvLoraRank
            self.qkNopeHeadDim = qkNopeHeadDim
            self.qkRopeHeadDim = qkRopeHeadDim
            self.vHeadDim = vHeadDim
            self.rope = rope
            self.sparsity = sparsity
            self.outputGateMode = outputGateMode
            self.useVHA = useVHA
            self.hasAttnSinkBias = hasAttnSinkBias
            self.splitKvBProj = splitKvBProj
            self.indexerUseFastHadamard = indexerUseFastHadamard
        }

        public var qLoraRank: Int {
            if case .factored(let rank) = qProjection { return rank }
            return 0
        }

        public var isDirectQ: Bool {
            if case .direct = qProjection { return true }
            return false
        }

        public var slidingWindow: Int {
            switch sparsity {
            case .dense(let sw), .dsa(let sw): return sw
            case .keyPool: return 0
            }
        }

        public var hasIndexer: Bool {
            switch sparsity {
            case .dense: return false
            case .keyPool, .dsa: return true
            }
        }
    }

    /// Attention sinks, relative biases, per-layer SWA geometry, and log-length scaling traits.
    public struct AttentionSinkRelConfig: Sendable, Hashable, Codable, Equatable {
        public let hasSink: Bool
        public let dRel: Int
        public let relExtent: Int
        public let slidingWindowSize: Int?
        public let swaQHeads: Int?
        public let swaKVHeads: Int?
        public let swaHeadDim: Int?
        public let logScalingFloor: Int?
        public let logScalingAlpha: Float?
        public let qkL2ClipThreshold: Float?
        public let relL2ClipThreshold: Float?
        public let valueNormNoGamma: Bool
        public let finalLogitSoftcapping: Float?
        public let localLayerIds: Set<Int>
        public let sconvKernelSize: Int
        public let layersPerAttnResBlock: Int?
        public let attnResNormalizeValues: Bool

        public init(
            hasSink: Bool = false,
            dRel: Int = 0,
            relExtent: Int = 0,
            slidingWindowSize: Int? = nil,
            swaQHeads: Int? = nil,
            swaKVHeads: Int? = nil,
            swaHeadDim: Int? = nil,
            logScalingFloor: Int? = nil,
            logScalingAlpha: Float? = nil,
            qkL2ClipThreshold: Float? = nil,
            relL2ClipThreshold: Float? = nil,
            valueNormNoGamma: Bool = false,
            finalLogitSoftcapping: Float? = nil,
            localLayerIds: Set<Int> = [],
            sconvKernelSize: Int = 4,
            layersPerAttnResBlock: Int? = nil,
            attnResNormalizeValues: Bool = false
        ) {
            self.hasSink = hasSink
            self.dRel = dRel
            self.relExtent = relExtent
            self.slidingWindowSize = slidingWindowSize
            self.swaQHeads = swaQHeads
            self.swaKVHeads = swaKVHeads
            self.swaHeadDim = swaHeadDim
            self.logScalingFloor = logScalingFloor
            self.logScalingAlpha = logScalingAlpha
            self.qkL2ClipThreshold = qkL2ClipThreshold
            self.relL2ClipThreshold = relL2ClipThreshold
            self.valueNormNoGamma = valueNormNoGamma
            self.finalLogitSoftcapping = finalLogitSoftcapping
            self.localLayerIds = localLayerIds
            self.sconvKernelSize = sconvKernelSize
            self.layersPerAttnResBlock = layersPerAttnResBlock
            self.attnResNormalizeValues = attnResNormalizeValues
        }

        public func isSlidingLayer(_ layer: Int) -> Bool {
            localLayerIds.contains(layer)
        }
    }

    /// Group-limited MoE routing and shared-expert gating configuration.
    public struct MoERoutingConfig: Sendable, Hashable, Codable, Equatable {
        public let nGroup: Int
        public let topkGroup: Int
        public let groupTopM: Int
        public let sharedExpertHasGate: Bool?
        public let scoreFunction: String
        public let enableExpertBias: Bool
        public let routedScalingFactor: Float
        public let routedScalingFactorLearnable: Bool
        public let normTopKProb: Bool

        public init(
            nGroup: Int = 1,
            topkGroup: Int = 1,
            groupTopM: Int = 2,
            sharedExpertHasGate: Bool? = nil,
            scoreFunction: String = "softmax",
            enableExpertBias: Bool = false,
            routedScalingFactor: Float = 1.0,
            routedScalingFactorLearnable: Bool = false,
            normTopKProb: Bool = false
        ) {
            self.nGroup = max(1, nGroup)
            self.topkGroup = max(1, min(self.nGroup, topkGroup))
            self.groupTopM = max(1, min(2, groupTopM))
            self.sharedExpertHasGate = sharedExpertHasGate
            self.scoreFunction = scoreFunction
            self.enableExpertBias = enableExpertBias
            self.routedScalingFactor = routedScalingFactor
            self.routedScalingFactorLearnable = routedScalingFactorLearnable
            self.normTopKProb = normTopKProb
        }
    }

    /// Per-layer and dense SwiGLU clamping configuration.
    public struct SwiGLUClampConfig: Sendable, Hashable, Codable, Equatable {
        public enum Mode: String, Sendable, Hashable, Codable, Equatable {
            case preSilu
            case postSilu
        }

        public let mode: Mode
        public let denseLimit: Float?
        public let layerRoutedLimits: [Float]
        public let layerSharedLimits: [Float]

        public init(
            mode: Mode = .preSilu,
            denseLimit: Float? = nil,
            layerRoutedLimits: [Float] = [],
            layerSharedLimits: [Float] = []
        ) {
            self.mode = mode
            self.denseLimit = denseLimit
            self.layerRoutedLimits = layerRoutedLimits
            self.layerSharedLimits = layerSharedLimits
        }

        public func routedSpec(layer: Int) -> QWSwiGLUClampSpec? {
            let lim = (0..<layerRoutedLimits.count).contains(layer) ? layerRoutedLimits[layer] : (denseLimit ?? 0)
            guard lim > 0 else { return nil }
            return QWSwiGLUClampSpec(limit: lim, postSilu: mode == .postSilu)
        }

        public func sharedSpec(layer: Int) -> QWSwiGLUClampSpec? {
            let lim = (0..<layerSharedLimits.count).contains(layer) ? layerSharedLimits[layer] : (denseLimit ?? 0)
            guard lim > 0 else { return nil }
            return QWSwiGLUClampSpec(limit: lim, postSilu: mode == .postSilu)
        }

        public func distinctSpecs() -> [QWSwiGLUClampSpec] {
            var set = Set<QWSwiGLUClampSpec>()
            let isPost = (mode == .postSilu)
            if let d = denseLimit, d > 0 {
                set.insert(QWSwiGLUClampSpec(limit: d, postSilu: isPost))
            }
            for v in layerRoutedLimits where v > 0 {
                set.insert(QWSwiGLUClampSpec(limit: v, postSilu: isPost))
            }
            for v in layerSharedLimits where v > 0 {
                set.insert(QWSwiGLUClampSpec(limit: v, postSilu: isPost))
            }
            return set.sorted { ($0.limit, $0.postSilu ? 1 : 0) < ($1.limit, $1.postSilu ? 1 : 0) }
        }
    }

    /// Native Multi-Token Prediction (MTP) fusion and layer layout configuration.
    public struct MTPConfig: Sendable, Hashable, Codable, Equatable {
        public enum FusionMode: String, Sendable, Hashable, Codable, Equatable {
            case concatEHProj
            case splitEProjHProj
            case sharedLastLayerEcho
        }

        public let layers: Int
        public let fusionMode: FusionMode
        public let useDedicatedEmbeddings: Bool
        public let hasFinalNorm: Bool

        public init(
            layers: Int,
            fusionMode: FusionMode,
            useDedicatedEmbeddings: Bool = false,
            hasFinalNorm: Bool = false
        ) {
            self.layers = layers
            self.fusionMode = fusionMode
            self.useDedicatedEmbeddings = useDedicatedEmbeddings
            self.hasFinalNorm = hasFinalNorm
        }
    }

    /// Vision tower and multimodal projector configuration.
    public struct VisionTowerConfig: Sendable, Hashable, Codable, Equatable {
        public enum PatchEmbedKind: String, Sendable, Hashable, Codable, Equatable {
            case conv3d
            case unfoldedPatchProj
            case lfmHybridConv
            case siglip2NaFlex
        }

        public let depth: Int
        public let hidden: Int
        public let heads: Int
        public let intermediate: Int
        public let patch: Int
        public let temporal: Int
        public let merge: Int
        public let numPosEmbeds: Int
        public let outDim: Int
        public let mergerIntermediateDim: Int?
        public let projectorBias: Bool
        public let normEps: Float
        public let ropeTheta: Double
        public let mropeLayout: MRoPELayout
        public let patchEmbedKind: PatchEmbedKind
        public let imageMean: [Float]
        public let imageStd: [Float]
        public let disableInternalMergerProj: Bool
        public let kvHeads: Int?
        public let layerTypes: [String]?
        public let convKernelSize: Int?
        public let projectorNormEps: Float

        public init(
            depth: Int,
            hidden: Int,
            heads: Int,
            intermediate: Int,
            patch: Int,
            temporal: Int,
            merge: Int,
            numPosEmbeds: Int,
            outDim: Int,
            mergerIntermediateDim: Int? = nil,
            projectorBias: Bool = true,
            normEps: Float = 1e-6,
            ropeTheta: Double = 10_000.0,
            mropeLayout: MRoPELayout = .qwenInterleaved,
            patchEmbedKind: PatchEmbedKind = .conv3d,
            imageMean: [Float] = [0.5, 0.5, 0.5],
            imageStd: [Float] = [0.5, 0.5, 0.5],
            disableInternalMergerProj: Bool = false,
            kvHeads: Int? = nil,
            layerTypes: [String]? = nil,
            convKernelSize: Int? = nil,
            projectorNormEps: Float = 1e-6
        ) {
            self.depth = depth
            self.hidden = hidden
            self.heads = heads
            self.intermediate = intermediate
            self.patch = patch
            self.temporal = temporal
            self.merge = merge
            self.numPosEmbeds = numPosEmbeds
            self.outDim = outDim
            self.mergerIntermediateDim = mergerIntermediateDim
            self.projectorBias = projectorBias
            self.normEps = normEps
            self.ropeTheta = ropeTheta
            self.mropeLayout = mropeLayout
            self.patchEmbedKind = patchEmbedKind
            self.imageMean = imageMean
            self.imageStd = imageStd
            self.disableInternalMergerProj = disableInternalMergerProj
            self.kvHeads = kvHeads
            self.layerTypes = layerTypes
            self.convKernelSize = convKernelSize
            self.projectorNormEps = projectorNormEps
        }
    }

    /// Top-level runtime stream and loop topology derived from config.
    public enum ExecutionTopology: Sendable, Hashable, Equatable {
        case standardSingleStream
        case sinkhornMHC(MHCConfig)
        case hyperStreamPLE(hcCount: Int, hcLowrank: Int?)
        case recurrentLoop(LoopSettings, inkling2: Inkling2Config?)
    }

    /// Resolved execution topology inferred strictly from decoded traits.
    public var executionTopology: ExecutionTopology {
        if let loop = loopSettings {
            return .recurrentLoop(loop, inkling2: inkling2Config)
        }
        if let mhc = mhcConfig {
            return .sinkhornMHC(mhc)
        }
        if let hc = hcCount, hc > 0 {
            return .hyperStreamPLE(hcCount: hc, hcLowrank: hcLowrank)
        }
        return .standardSingleStream
    }

    /// Resolved special token IDs for chat template framing, reasoning tags, and stop sets.
    public var specialTokens: QWModelSpecialTokens {
        var stops = eosTokenIds.filter { $0 >= 0 && $0 < vocab }
        if let spec = QWModelFamilyRegistry.shared.spec(forModelType: modelType),
           let defaults = spec.defaultSpecialTokens {
            if defaults.usesChatTemplateIds, defaults.imEndTokenId >= 0, defaults.imEndTokenId < vocab,
               !stops.contains(defaults.imEndTokenId) {
                stops.append(defaults.imEndTokenId)
            }
            return QWModelSpecialTokens(
                eosTokenId: eosTokenId,
                imStartTokenId: defaults.imStartTokenId,
                imEndTokenId: defaults.imEndTokenId,
                thinkTokenId: defaults.thinkTokenId,
                endThinkTokenId: defaults.endThinkTokenId,
                extraStopTokenIds: stops,
                usesChatTemplateIds: defaults.usesChatTemplateIds
            )
        }
        return QWModelSpecialTokens(
            eosTokenId: eosTokenId,
            extraStopTokenIds: stops,
            usesChatTemplateIds: usesChatTemplateIds
        )
    }

    /// Returns a copy with vision mean/std overridden (e.g. from `preprocessor_config.json`).
    public func withVisionNormalization(mean: [Float]?, std: [Float]?) -> QWModelDescriptor {
        guard let vt = visionTowerConfig else { return self }
        let nextMean = (mean?.count == 3) ? mean! : vt.imageMean
        let nextStd = (std?.count == 3) ? std! : vt.imageStd
        if nextMean == vt.imageMean && nextStd == vt.imageStd { return self }
        let updatedVT = VisionTowerConfig(
            depth: vt.depth, hidden: vt.hidden, heads: vt.heads, intermediate: vt.intermediate,
            patch: vt.patch, temporal: vt.temporal, merge: vt.merge, numPosEmbeds: vt.numPosEmbeds,
            outDim: vt.outDim, mergerIntermediateDim: vt.mergerIntermediateDim,
            projectorBias: vt.projectorBias, normEps: vt.normEps, ropeTheta: vt.ropeTheta,
            mropeLayout: vt.mropeLayout, patchEmbedKind: vt.patchEmbedKind,
            imageMean: nextMean, imageStd: nextStd,
            disableInternalMergerProj: vt.disableInternalMergerProj,
            kvHeads: vt.kvHeads,
            layerTypes: vt.layerTypes,
            convKernelSize: vt.convKernelSize
        )
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
            hasQKNorm: hasQKNorm, hasQKVBias: hasQKVBias, hasAttnOutputBias: hasAttnOutputBias,
            usesPlusOneRMSNorm: usesPlusOneRMSNorm, quantization: quantization,
            hasMropeSection: hasMropeSection, tieWordEmbeddings: tieWordEmbeddings,
            normTopKProb: normTopKProb, eosTokenIds: eosTokenIds, envelopeModelType: envelopeModelType,
            loopSettings: loopSettings, mhcConfig: mhcConfig, firstKDenseReplace: firstKDenseReplace,
            mlpOnlyLayers: mlpOnlyLayers, decoderSparseStep: decoderSparseStep, moeLayerFreq: moeLayerFreq,
            routedScalingFactor: routedScalingFactor, swigluLimit: swigluLimit, topkMethod: topkMethod,
            residentExpertsPerLayer: residentExpertsPerLayer, sharedExpertHasGate: sharedExpertHasGate,
            mlaUseNoPE: mlaUseNoPE, kvLoraRank: kvLoraRank, qLoraRank: qLoraRank,
            qkNopeHeadDim: qkNopeHeadDim, qkRopeHeadDim: qkRopeHeadDim, vHeadDim: vHeadDim,
            indexKPoolCompress: indexKPoolCompress, indexKPoolAlwaysSelectTail: indexKPoolAlwaysSelectTail,
            indexShareForMTP: indexShareForMTP, indexerRopeInterleave: indexerRopeInterleave,
            gdnVectorDecay: gdnVectorDecay, gdnLowRankDim: gdnLowRankDim,
            gdnGateLowerBound: gdnGateLowerBound, gdnSeparateConv: gdnSeparateConv,
            hcaConfig: hcaConfig, hybridMLAConfig: hybridMLAConfig, blockSparseGQAConfig: blockSparseGQAConfig,
            yarnRopeConfig: yarnRopeConfig, ropeScaling: ropeScaling, slidingWindow: slidingWindow,
            layerSlidingWindows: layerSlidingWindows, activationKind: activationKind,
            moeLatentSize: moeLatentSize, latentMoeUseNorm: latentMoeUseNorm,
            moeSplitFeatureRouting: moeSplitFeatureRouting,
            moeRouterEnableExpertBias: moeRouterEnableExpertBias,
            routedScalingFactorLearnable: routedScalingFactorLearnable,
            multimaxHead: multimaxHead, mtpSharedLastLayer: mtpSharedLastLayer,
            inkling2Config: inkling2Config, dsparkConfig: dsparkConfig,
            mlaConfig: mlaConfig, kdaConfig: kdaConfig,
            attentionSinkRelConfig: attentionSinkRelConfig, moeRoutingConfig: moeRoutingConfig,
            swigluClampConfig: swigluClampConfig, mtpConfig: mtpConfig,
            visionTowerConfig: updatedVT, ropeInterleave: ropeInterleave, role: role,
            shortConvConfig: shortConvConfig, normKind: normKind,
            useSandwichNorm: useSandwichNorm, perKindRoPEConfig: perKindRoPEConfig,
            attnGateMode: attnGateMode, decisionConfig: decisionConfig
        )
    }

    /// Whether the pack carries Qwen3.8 chat template special IDs inside vocabulary range.
    public var usesChatTemplateIds: Bool {
        if let spec = QWModelFamilyRegistry.shared.spec(forModelType: modelType) {
            return spec.usesQwenChatTemplateIds && vocab > 248046
        }
        return [.qwen38Dense, .qwen38MoE, .qwen38FlashNext, .senseNovaLoop].contains(family) && vocab > 248046
    }

    internal static func synthesizeCapabilityBundle(
        family: Family, modelType: String, capabilities: Set<Capability>,
        qHeads: Int, headDim: Int, attnOutputGate: Bool, ropeTheta: Double, ropeInterleave: Bool,
        indexerHeads: Int?, indexKPoolCompress: Bool, mlaUseNoPE: Bool,
        kvLoraRank: Int?, qLoraRank: Int?, qkNopeHeadDim: Int?, qkRopeHeadDim: Int?, vHeadDim: Int?,
        hybridMLAConfig: HybridMLAConfig?, numExperts: Int?, sharedExpertHasGate: Bool,
        expertGateType: String?, topkMethod: String?, moeRouterEnableExpertBias: Bool,
        routedScalingFactor: Float, routedScalingFactorLearnable: Bool, normTopKProb: Bool,
        swigluLimit: Float?, mtpLayers: Int, mtpUseDedicatedEmbeddings: Bool, mtpSharedLastLayer: Bool,
        mhcConfig: MHCConfig?, gdnQKHeads: Int, gdnVHeads: Int, gdnHeadDim: Int, gdnConvKernel: Int,
        gdnVectorDecay: Bool, gdnLowRankDim: Int?, gdnGateLowerBound: Float?, gdnSeparateConv: Bool,
        visDepth: Int, visHidden: Int, visHeads: Int, visIntermediate: Int, visPatch: Int,
        visTemporal: Int, visMerge: Int, visPosEmbeds: Int, visOut: Int,
        hasMropeSection: Bool, slidingWindow: Int?, inkling2Config: Inkling2Config?
    ) -> (
        kda: KDAConfig?, mla: MLAConfig?, sinkRel: AttentionSinkRelConfig?, moe: MoERoutingConfig?,
        swiglu: SwiGLUClampConfig?, mtp: MTPConfig?, vision: VisionTowerConfig?
    ) {
        let kda: KDAConfig? = (gdnVectorDecay && gdnQKHeads > 0) ? KDAConfig(
            qkHeads: gdnQKHeads, vHeads: gdnVHeads, headDim: gdnHeadDim, convKernel: gdnConvKernel,
            gateProjection: gdnLowRankDim.map { .lowRank(dim: $0) } ?? .fullRank,
            safeGate: false, gateLowerBound: gdnGateLowerBound, outputNormAct: .sigmoid,
            separateConv: gdnSeparateConv
        ) : nil
        let sparsity: MLAConfig.Sparsity = (indexerHeads ?? 0) > 0
            ? (indexKPoolCompress ? .keyPool : .dsa(slidingWindow: hybridMLAConfig?.slidingWindow ?? slidingWindow ?? 0))
            : .dense(slidingWindow: hybridMLAConfig?.slidingWindow ?? slidingWindow ?? 0)
        let mla: MLAConfig? = {
            if let h = hybridMLAConfig {
                let qProj: MLAConfig.QProjection = h.qLoraRank > 0 ? .factored(rank: h.qLoraRank) : .direct
                let outMode: MLAConfig.OutputGateMode = attnOutputGate
                    ? (qProj == .direct ? .headWiseFromHidden : .elementWiseFromQLora) : .none
                let ropeCfg: MLARopeConfig? = h.decoupledRope ? MLARopeConfig(
                    dim: h.qkRopeHeadDim, theta: ropeTheta, interleave: h.ropeInterleave,
                    mropeLayout: hasMropeSection ? .qwenInterleaved : .none1D, yarn: h.yarn
                ) : nil
                return MLAConfig(
                    qHeads: h.qHeads, qProjection: qProj, kvLoraRank: h.kvLoraRank,
                    qkNopeHeadDim: h.qkNopeHeadDim, qkRopeHeadDim: h.qkRopeHeadDim, vHeadDim: h.vHeadDim,
                    rope: ropeCfg, sparsity: sparsity, outputGateMode: outMode,
                    useVHA: h.useVHA, hasAttnSinkBias: h.hasAttnSinkBias,
                    splitKvBProj: h.splitKvBProj, indexerUseFastHadamard: h.indexerUseFastHadamard
                )
            }
            if mlaUseNoPE, let kvR = kvLoraRank, kvR > 0 {
                let qProj: MLAConfig.QProjection = (qLoraRank ?? 0) > 0 ? .factored(rank: qLoraRank!) : .direct
                return MLAConfig(
                    qHeads: qHeads, qProjection: qProj, kvLoraRank: kvR,
                    qkNopeHeadDim: qkNopeHeadDim ?? headDim, qkRopeHeadDim: 0, vHeadDim: vHeadDim ?? headDim,
                    rope: nil, sparsity: sparsity, outputGateMode: attnOutputGate ? .elementWiseFromHidden : .none
                )
            }
            if let kvR = kvLoraRank, kvR > 0, let nope = qkNopeHeadDim, let ropeDim = qkRopeHeadDim {
                let qProj: MLAConfig.QProjection = (qLoraRank ?? 0) > 0 ? .factored(rank: qLoraRank!) : .direct
                let spec = QWModelFamilyRegistry.shared.spec(forModelType: modelType)
                let layout: MRoPELayout = hasMropeSection
                    ? ((spec?.defaultBailingVideoRope == true) ? .bailingVideoRope : .qwenInterleaved)
                    : .none1D
                return MLAConfig(
                    qHeads: qHeads, qProjection: qProj, kvLoraRank: kvR,
                    qkNopeHeadDim: nope, qkRopeHeadDim: ropeDim, vHeadDim: vHeadDim ?? headDim,
                    rope: ropeDim > 0 ? MLARopeConfig(dim: ropeDim, theta: ropeTheta, interleave: ropeInterleave, mropeLayout: layout) : nil,
                    sparsity: sparsity, outputGateMode: attnOutputGate ? .headWiseFromHidden : .none
                )
            }
            return nil
        }()
        let sinkRel: AttentionSinkRelConfig? = inkling2Config.map { ink in
            AttentionSinkRelConfig(
                hasSink: true, dRel: ink.dRel, relExtent: ink.relExtent,
                slidingWindowSize: ink.slidingWindowSize, swaQHeads: ink.swaQHeads,
                swaKVHeads: ink.swaKVHeads, swaHeadDim: ink.swaHeadDim,
                logScalingFloor: ink.logScalingNFloor, logScalingAlpha: ink.logScalingAlpha,
                qkL2ClipThreshold: ink.qkL2ClipThreshold, relL2ClipThreshold: ink.relL2ClipThreshold,
                valueNormNoGamma: ink.useValueNormNoGamma, finalLogitSoftcapping: ink.finalLogitSoftcapping,
                localLayerIds: ink.localLayerIds, sconvKernelSize: ink.sconvKernelSize,
                layersPerAttnResBlock: ink.layersPerAttnResBlock,
                attnResNormalizeValues: ink.attnResNormalizeValues
            )
        }
        let moe: MoERoutingConfig? = (numExperts ?? 0) > 0 ? MoERoutingConfig(
            nGroup: 1, topkGroup: 1, groupTopM: 2, sharedExpertHasGate: sharedExpertHasGate,
            scoreFunction: expertGateType ?? "softmax",
            enableExpertBias: topkMethod == "noaux_tc" || moeRouterEnableExpertBias,
            routedScalingFactor: routedScalingFactor,
            routedScalingFactorLearnable: routedScalingFactorLearnable,
            normTopKProb: normTopKProb
        ) : nil
        let swiglu: SwiGLUClampConfig? = swigluLimit.map {
            SwiGLUClampConfig(mode: .preSilu, denseLimit: $0)
        }
        let mtp: MTPConfig? = mtpLayers > 0 ? MTPConfig(
            layers: mtpLayers,
            fusionMode: mtpSharedLastLayer ? .sharedLastLayerEcho
                : ((family == .maxwellMoE && mhcConfig != nil) ? .splitEProjHProj : .concatEHProj),
            useDedicatedEmbeddings: mtpUseDedicatedEmbeddings,
            hasFinalNorm: false
        ) : nil
        let spec = QWModelFamilyRegistry.shared.spec(forModelType: modelType)
        let vision: VisionTowerConfig? = capabilities.contains(.vision) ? VisionTowerConfig(
            depth: visDepth, hidden: visHidden, heads: visHeads, intermediate: visIntermediate,
            patch: visPatch, temporal: visTemporal, merge: visMerge, numPosEmbeds: visPosEmbeds,
            outDim: visOut, mergerIntermediateDim: inkling2Config?.vision?.projectionHiddenDim,
            projectorBias: inkling2Config?.vision?.projectorBias ?? true,
            normEps: Float(inkling2Config?.vision?.layerNormEps ?? 1e-6),
            mropeLayout: (spec?.defaultBailingVideoRope == true) ? .bailingVideoRope
                : (hasMropeSection ? .qwenInterleaved : .none1D),
            patchEmbedKind: inkling2Config?.vision != nil ? .unfoldedPatchProj : .conv3d,
            imageMean: spec?.defaultVisionMean ?? [0.5, 0.5, 0.5],
            imageStd: spec?.defaultVisionStd ?? [0.5, 0.5, 0.5]
        ) : nil
        return (kda, mla, sinkRel, moe, swiglu, mtp, vision)
    }

    internal static func appendCapabilityV3DigestFields(_ d: QWModelDescriptor, into f: inout [String]) {
        if let kda = d.kdaConfig, kda.isFullRank || kda.safeGate {
            f.append("kda_cfg=\(kda.isFullRank ? "full" : "lora\(kda.lowRankDim ?? 0)"),\(kda.safeGate ? 1 : 0),\(kda.outputNormAct.rawValue)")
        }
        if let mla = d.mlaConfig, mla.isDirectQ || mla.outputGateMode == .headWiseFromHidden || (d.hybridMLAConfig == nil && !d.mlaUseNoPE) {
            f.append("mla_cfg=\(mla.isDirectQ ? 1 : 0),\(mla.outputGateMode.rawValue),\(mla.hasIndexer ? 1 : 0),\(mla.rope?.interleave == true ? 1 : 0)")
        }
        if let moe = d.moeRoutingConfig, moe.nGroup > 1 {
            f.append("moe_grp=\(moe.nGroup),\(moe.topkGroup),\(moe.groupTopM),\(moe.scoreFunction)")
        }
        if let sw = d.swigluClampConfig, sw.mode == .postSilu || !sw.layerRoutedLimits.isEmpty || !sw.layerSharedLimits.isEmpty {
            let rHex = sw.layerRoutedLimits.map { String($0.bitPattern, radix: 16) }.joined(separator: ",")
            let sHex = sw.layerSharedLimits.map { String($0.bitPattern, radix: 16) }.joined(separator: ",")
            f.append("swiglu_layers=\(sw.mode.rawValue)|r=\(rHex)|s=\(sHex)")
        }
        if let vis = d.visionTowerConfig, vis.disableInternalMergerProj {
            f.append("vis_merger=\(vis.mergerIntermediateDim ?? 0),\(vis.outDim),1")
        }
        if d.ropeInterleave {
            f.append("rope_ilv=1")
        }
    }

    internal static func decodeSwiGLUClampConfig(
        text t: ConfigEnvelope.Text,
        routedLimits: [Float]?,
        sharedLimits: [Float]?,
        inv: (Bool, String) throws -> Void
    ) throws -> SwiGLUClampConfig? {
        let singleLimit: Float? = (t.mlp_clamp == false) ? nil : (t.swiglu_limit ?? t.swiglu_limits)
        let rList = routedLimits ?? []
        let sList = sharedLimits ?? []
        if let lim = singleLimit {
            try inv(lim > 0 && lim.isFinite, "swiglu_limit \(lim) invalid")
        }
        for v in rList { try inv(v >= 0 && v.isFinite, "expert_swiglu_limit_list value \(v) invalid") }
        for v in sList { try inv(v >= 0 && v.isFinite, "share_expert_swiglu_limit_list value \(v) invalid") }
        let hasPerLayer = rList.contains(where: { $0 > 0 }) || sList.contains(where: { $0 > 0 })
        if hasPerLayer {
            return SwiGLUClampConfig(
                mode: .postSilu,
                denseLimit: singleLimit,
                layerRoutedLimits: rList,
                layerSharedLimits: sList
            )
        }
        if let lim = singleLimit {
            return SwiGLUClampConfig(mode: .preSilu, denseLimit: lim)
        }
        return nil
    }

    internal static func parseRoPEScaling(
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
}
