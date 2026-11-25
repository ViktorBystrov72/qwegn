// QWMultimodalPrompt.swift — Multimodal prompt preparation and X0 activation interleaving.
import Foundation
import QWCore
import QWMM
import QWVision

public enum QWMultimodalError: Error, CustomStringConvertible {
    case featureDimensionMismatch(expected: Int, actual: Int)
    case gridCountMismatch(expected: Int, actual: Int)
    case badArgument(String)

    public var description: String {
        switch self {
        case .featureDimensionMismatch(let exp, let act):
            return "QWMultimodal: visual feature count mismatch (expected \(exp), got \(act))"
        case .gridCountMismatch(let exp, let act):
            return "QWMultimodal: grid count mismatch (expected \(exp), got \(act))"
        case .badArgument(let msg):
            return "QWMultimodal: \(msg)"
        }
    }
}

public struct QWMultimodalPrompt: Sendable {
    public let originalTokens: [Int]
    public let expandedTokens: [Int]
    public let plan: QWX0Plan
    public let visualTokens: [Float]
    public let hiddenDim: Int

    public init(
        promptTokens: [Int],
        images: [(grid: QWVisionGrid, features: [Float])],
        videos: [(grid: QWVisionGrid, features: [Float])] = [],
        hiddenDim: Int,
        spec: QWMMSpec = .qwen3VLReference
    ) throws {
        self.originalTokens = promptTokens
        self.hiddenDim = hiddenDim

        let imageGrids = images.map { $0.grid }
        let videoGrids = videos.map { $0.grid }

        for (grid, feat) in images {
            let expectedCount = grid.tokens * hiddenDim
            guard feat.count == expectedCount else {
                throw QWMultimodalError.featureDimensionMismatch(expected: expectedCount, actual: feat.count)
            }
        }

        for (grid, feat) in videos {
            let expectedCount = grid.tokens * hiddenDim
            guard feat.count == expectedCount else {
                throw QWMultimodalError.featureDimensionMismatch(expected: expectedCount, actual: feat.count)
            }
        }

        let builtPlan = try QWX0Plan(tokens: promptTokens, images: imageGrids, videos: videoGrids, spec: spec)
        var allVisTokens: [Float] = []
        var ii = 0
        var vi = 0
        for isVid in builtPlan.visionSpanIsVideo {
            if isVid {
                allVisTokens.append(contentsOf: videos[vi].features)
                vi += 1
            } else {
                allVisTokens.append(contentsOf: images[ii].features)
                ii += 1
            }
        }

        self.visualTokens = allVisTokens
        self.plan = builtPlan
        self.expandedTokens = builtPlan.tokens
    }

    /// Extracts the contiguous visual feature spans (`[tokenCount * hiddenDim]`) in expanded-prompt token coordinates.
    public var visualSpans: [QWVisualSpan] {
        var out: [QWVisualSpan] = []
        var visOffset = 0
        var visIdx = 0
        for span in plan.spans {
            if case .vision(let grid, _, let rows) = span.content {
                let count = rows * hiddenDim
                let slice = Array(visualTokens[visOffset..<(visOffset + count)])
                let isVid = visIdx < plan.visionSpanIsVideo.count ? plan.visionSpanIsVideo[visIdx] : (grid.t > 1)
                out.append(QWVisualSpan(
                    tokenOffset: span.rowStart,
                    tokenCount: rows,
                    embeddings: slice,
                    gridT: grid.t,
                    gridH: grid.h,
                    gridW: grid.w,
                    spatialMergeSize: grid.spec.merge,
                    isVideo: isVid
                ))
                visOffset += count
                visIdx += 1
            }
        }
        return out
    }
}
