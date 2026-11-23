// QWGenerationResult.swift — Result payload returned by unified engine generation.
import Foundation
import QWCore
import QWHead

/// What one generation produced.
public struct QWGenerationResult: Sendable {
    /// The generated tokens, in order.
    public let tokens: [Int]
    /// Per-step latency in milliseconds.
    public let stepMs: [Double]
    /// How many tokens each step committed.
    public let stepTokenCounts: [Int]
    /// Wall time the prefill took. Absent when nothing measured it.
    public let prefillSeconds: Double?
    /// Wall time from the request to the first generated token.
    public let firstTokenSeconds: Double?
    /// Wall time the decode took.
    public let decodeSeconds: Double?
    /// GPU time the prefill spent, as the module measured it.
    public let prefillGPUSeconds: Double?
    /// The stop token that ended the turn, when one did. A turn that
    /// ran to its limit has no stop token: absence is not a zero.
    public let stopTokenID: Int?
    /// What the prefix cache did for this turn. Absent when no cache
    /// answered the prompt.
    public let cache: QWCacheAnswer?
    /// Speculative decoding telemetry summary, when speculative rounds ran.
    public let speculativeSummary: String?
    /// Per-token log-probabilities when `collectLogprobs` was requested.
    public let tokenLogprobs: [QWTokenLogprob]

    public init(tokens: [Int], stepMs: [Double], stepTokenCounts: [Int],
                prefillSeconds: Double? = nil, firstTokenSeconds: Double? = nil,
                decodeSeconds: Double? = nil, prefillGPUSeconds: Double? = nil,
                stopTokenID: Int? = nil, cache: QWCacheAnswer? = nil,
                speculativeSummary: String? = nil,
                tokenLogprobs: [QWTokenLogprob] = []) {
        self.tokens = tokens
        self.stepMs = stepMs
        self.stepTokenCounts = stepTokenCounts
        self.prefillSeconds = prefillSeconds
        self.firstTokenSeconds = firstTokenSeconds
        self.decodeSeconds = decodeSeconds
        self.prefillGPUSeconds = prefillGPUSeconds
        self.stopTokenID = stopTokenID
        self.cache = cache
        self.speculativeSummary = speculativeSummary
        self.tokenLogprobs = tokenLogprobs
    }
}
