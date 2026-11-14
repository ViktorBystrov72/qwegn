// QWVisionConfig.swift — Config-sourced vision tower geometry (no model defaults).
import Foundation
import QWCore

/// Vision tower architecture parameters derived from a model's `config.json`.
public struct QWVisionConfig: Sendable, Equatable, Hashable {
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
        eps: Float = 1e-6
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
    }

    public init(descriptor: QWModelDescriptor) {
        let computedHeadDim = descriptor.visHeads > 0
            ? descriptor.visHidden / descriptor.visHeads
            : 0
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
