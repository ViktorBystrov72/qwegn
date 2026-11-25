// QWVisionConfig.swift — Config-sourced vision tower geometry (no model defaults).
import Foundation
import QWCore

/// Vision tower architecture parameters derived from a model's `config.json`.
public struct QWVisionConfig: Sendable, Equatable, Hashable {
    public enum Architecture: String, Sendable, Equatable, Hashable {
        case qwen3VL
        case inkling2
        case lfm3VL
        case siglip2
    }

    public let dim: Int
    public let heads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let depth: Int
    public let mlpInner: Int
    public let patch: Int
    public let temporal: Int
    public let merge: Int
    public let numPosEmbeds: Int
    public let outDim: Int
    public let imageTokenId: Int
    public let videoTokenId: Int
    public let ropeTheta: Double
    public let eps: Float
    public let projectorNormEps: Float
    public let architecture: Architecture
    public let projectionHiddenDim: Int
    public let projectorBias: Bool
    public let mergerIntermediateDim: Int?
    public let imageMean: [Float]
    public let imageStd: [Float]
    public let mropeLayout: QWModelDescriptor.MRoPELayout
    public let disableInternalMergerProj: Bool
    public let visionLayerTypes: [String]
    public let convKernelSize: Int

    public var effectiveMergerIntermediateDim: Int {
        if projectionHiddenDim > 0 { return projectionHiddenDim }
        if let mid = mergerIntermediateDim, mid > 0 { return mid }
        return merged
    }

    public var spatial: Int { headDim / 2 }
    public var invFreqCount: Int { spatial / 2 }
    public var qkvDim: Int { 3 * dim }
    public var patchArea: Int { patch * patch }
    public var temporalPatchArea: Int { temporal * patchArea }
    public var patchChannels: Int { 3 * temporalPatchArea }
    public var merged: Int { merge * merge * dim }
    public var posSide: Int {
        numPosEmbeds > 0 ? Int(Double(numPosEmbeds).squareRoot().rounded()) : 0
    }

    public init(
        dim: Int,
        heads: Int,
        headDim: Int,
        depth: Int,
        mlpInner: Int,
        patch: Int,
        temporal: Int,
        merge: Int,
        numPosEmbeds: Int,
        outDim: Int,
        imageTokenId: Int,
        videoTokenId: Int,
        ropeTheta: Double = 10_000,
        eps: Float = 1e-6,
        projectorNormEps: Float? = nil,
        architecture: Architecture = .qwen3VL,
        projectionHiddenDim: Int = 0,
        projectorBias: Bool = true,
        mergerIntermediateDim: Int? = nil,
        imageMean: [Float] = [0.5, 0.5, 0.5],
        imageStd: [Float] = [0.5, 0.5, 0.5],
        mropeLayout: QWModelDescriptor.MRoPELayout = .qwenInterleaved,
        disableInternalMergerProj: Bool = false,
        kvHeads: Int? = nil,
        visionLayerTypes: [String] = [],
        convKernelSize: Int = 3
    ) {
        self.dim = dim
        self.heads = heads
        self.kvHeads = kvHeads ?? heads
        self.headDim = headDim
        self.depth = depth
        self.mlpInner = mlpInner
        self.patch = patch
        self.temporal = temporal
        self.merge = merge
        self.numPosEmbeds = numPosEmbeds
        self.outDim = outDim
        self.imageTokenId = imageTokenId
        self.videoTokenId = videoTokenId
        self.ropeTheta = ropeTheta
        self.eps = eps
        self.projectorNormEps = projectorNormEps ?? eps
        self.architecture = architecture
        self.projectionHiddenDim = projectionHiddenDim
        self.projectorBias = projectorBias
        self.mergerIntermediateDim = mergerIntermediateDim
        self.imageMean = imageMean
        self.imageStd = imageStd
        self.mropeLayout = mropeLayout
        self.disableInternalMergerProj = disableInternalMergerProj
        self.visionLayerTypes = visionLayerTypes
        self.convKernelSize = convKernelSize
    }

    public init(descriptor: QWModelDescriptor) {
        if let vt = descriptor.visionTowerConfig {
            let computedHeadDim = vt.heads > 0 ? vt.hidden / vt.heads : 0
            let arch: Architecture = switch vt.patchEmbedKind {
            case .unfoldedPatchProj: .inkling2
            case .lfmHybridConv: .lfm3VL
            case .siglip2NaFlex: .siglip2
            case .conv3d: .qwen3VL
            }
            let isFixedTemporal = (vt.patchEmbedKind == .unfoldedPatchProj || vt.patchEmbedKind == .siglip2NaFlex)
            let projH = vt.mergerIntermediateDim ?? (vt.merge * vt.merge * vt.hidden)
            self.init(
                dim: vt.hidden,
                heads: vt.heads,
                headDim: computedHeadDim,
                depth: vt.depth,
                mlpInner: vt.intermediate,
                patch: vt.patch,
                temporal: isFixedTemporal ? 1 : vt.temporal,
                merge: vt.merge,
                numPosEmbeds: vt.numPosEmbeds,
                outDim: vt.outDim,
                imageTokenId: descriptor.imageTokenId,
                videoTokenId: descriptor.videoTokenId,
                ropeTheta: vt.ropeTheta,
                eps: vt.normEps,
                projectorNormEps: vt.projectorNormEps,
                architecture: arch,
                projectionHiddenDim: projH,
                projectorBias: vt.projectorBias,
                mergerIntermediateDim: vt.mergerIntermediateDim,
                imageMean: vt.imageMean,
                imageStd: vt.imageStd,
                mropeLayout: vt.mropeLayout,
                disableInternalMergerProj: vt.disableInternalMergerProj,
                kvHeads: vt.kvHeads ?? vt.heads,
                visionLayerTypes: vt.layerTypes ?? [],
                convKernelSize: vt.convKernelSize ?? 3
            )
            return
        }
        let computedHeadDim = descriptor.visHeads > 0 ? descriptor.visHidden / descriptor.visHeads : 0
        self.init(
            dim: descriptor.visHidden,
            heads: descriptor.visHeads,
            headDim: computedHeadDim,
            depth: descriptor.visDepth,
            mlpInner: descriptor.visIntermediate,
            patch: descriptor.visPatch,
            temporal: descriptor.visTemporal,
            merge: descriptor.visMerge,
            numPosEmbeds: descriptor.visPosEmbeds,
            outDim: descriptor.visOut,
            imageTokenId: descriptor.imageTokenId,
            videoTokenId: descriptor.videoTokenId
        )
    }

    /// Reference Qwen3-VL vision tower geometry for synthetic fixtures without config.json.
    public static let qwen3VLReference = QWVisionConfig(
        dim: 1152,
        heads: 16,
        headDim: 72,
        depth: 27,
        mlpInner: 4304,
        patch: 16,
        temporal: 2,
        merge: 2,
        numPosEmbeds: 2304,
        outDim: 5120,
        imageTokenId: 248056,
        videoTokenId: 248057
    )
}
