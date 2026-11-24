// QWVisionConfig.swift — Config-sourced vision tower geometry (no model defaults).
import Foundation
import QWCore

/// Vision tower architecture parameters derived from a model's `config.json`.
public struct QWVisionConfig: Sendable, Equatable, Hashable {
    public enum Architecture: String, Sendable, Equatable, Hashable {
        case qwen3VL
        case inkling2
    }

    public let dim: Int
    public let heads: Int
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
    public let architecture: Architecture
    public let projectionHiddenDim: Int
    public let projectorBias: Bool
    public let mergerIntermediateDim: Int?
    public let imageMean: [Float]
    public let imageStd: [Float]
    public let mropeLayout: QWModelDescriptor.MRoPELayout
    public let disableInternalMergerProj: Bool

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
        architecture: Architecture = .qwen3VL,
        projectionHiddenDim: Int = 0,
        projectorBias: Bool = true,
        mergerIntermediateDim: Int? = nil,
        imageMean: [Float] = [0.5, 0.5, 0.5],
        imageStd: [Float] = [0.5, 0.5, 0.5],
        mropeLayout: QWModelDescriptor.MRoPELayout = .qwenInterleaved,
        disableInternalMergerProj: Bool = false
    ) {
        self.dim = dim
        self.heads = heads
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
        self.architecture = architecture
        self.projectionHiddenDim = projectionHiddenDim
        self.projectorBias = projectorBias
        self.mergerIntermediateDim = mergerIntermediateDim
        self.imageMean = imageMean
        self.imageStd = imageStd
        self.mropeLayout = mropeLayout
        self.disableInternalMergerProj = disableInternalMergerProj
    }

    public init(descriptor: QWModelDescriptor) {
        if let vt = descriptor.visionTowerConfig {
            let computedHeadDim = vt.heads > 0 ? vt.hidden / vt.heads : 0
            let isUnfolded = (vt.patchEmbedKind == .unfoldedPatchProj)
            let projH = vt.mergerIntermediateDim ?? (vt.merge * vt.merge * vt.hidden)
            self.init(
                dim: vt.hidden,
                heads: vt.heads,
                headDim: computedHeadDim,
                depth: vt.depth,
                mlpInner: vt.intermediate,
                patch: vt.patch,
                temporal: isUnfolded ? 1 : vt.temporal,
                merge: vt.merge,
                numPosEmbeds: vt.numPosEmbeds,
                outDim: vt.outDim,
                imageTokenId: descriptor.imageTokenId,
                videoTokenId: descriptor.videoTokenId,
                ropeTheta: vt.ropeTheta,
                eps: vt.normEps,
                architecture: isUnfolded ? .inkling2 : .qwen3VL,
                projectionHiddenDim: projH,
                projectorBias: vt.projectorBias,
                mergerIntermediateDim: vt.mergerIntermediateDim,
                imageMean: vt.imageMean,
                imageStd: vt.imageStd,
                mropeLayout: vt.mropeLayout,
                disableInternalMergerProj: vt.disableInternalMergerProj
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
