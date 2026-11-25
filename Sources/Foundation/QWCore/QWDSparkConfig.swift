// QWDSparkConfig.swift — Capability config, sibling pack discovery, and budget scheduler for DSpark.

import Foundation

extension QWModelDescriptor {
    /// Target-coupled DSpark draft-model capability configuration decoded from `config.json`.
    public struct DSparkConfig: Sendable, Codable, Equatable, Hashable {
        public let blockSize: Int
        public let markovRank: Int
        public let noiseTokenId: Int
        public let targetLayerIds: [Int]
        public let numHiddenLayers: Int
        public let slidingWindow: Int
        public let ropeTheta: Double
        public let sampleFromAnchor: Bool
        public let enableConfidenceHead: Bool?

        /// Maximum draft tokens per speculative block (`B` when `sampleFromAnchor`, else `B - 1`).
        public var maxDrafts: Int { sampleFromAnchor ? blockSize : max(1, blockSize - 1) }

        /// Starting token index inside the `[B]` draft block for sampling.
        public var sampleOffset: Int { sampleFromAnchor ? 0 : 1 }

        /// 0-based target physical layer indices (`dspark_target_layer_ids` are 0-based capture-after indices).
        public var targetZeroBasedLayers: [Int] { targetLayerIds }

        public init(
            blockSize: Int,
            markovRank: Int,
            noiseTokenId: Int,
            targetLayerIds: [Int],
            numHiddenLayers: Int? = nil,
            slidingWindow: Int = 128,
            ropeTheta: Double = 8_000_000.0,
            sampleFromAnchor: Bool = true,
            enableConfidenceHead: Bool? = nil
        ) {
            self.blockSize = blockSize
            self.markovRank = markovRank
            self.noiseTokenId = noiseTokenId
            self.targetLayerIds = targetLayerIds
            self.numHiddenLayers = max(1, numHiddenLayers ?? targetLayerIds.count)
            self.slidingWindow = max(1, slidingWindow)
            self.ropeTheta = ropeTheta
            self.sampleFromAnchor = sampleFromAnchor
            self.enableConfidenceHead = enableConfidenceHead
        }

        /// Resolves `<pack_dir>/dspark.qwp` or `<pack_dir>/assets/dspark.qwp` if present.
        public static func resolvePackURL(near modelPath: String) -> URL? {
            let trimmed = modelPath.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return nil }
            let url = URL(fileURLWithPath: trimmed.expandingTildePath)
            return resolveSiblingFile("dspark.qwp", near: url)
        }

        /// Resolves optional `<pack_dir>/draft_head.qwp` or `<pack_dir>/assets/draft_head.qwp` if present.
        public static func resolveDraftHeadURL(near modelPath: String) -> URL? {
            let trimmed = modelPath.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return nil }
            let url = URL(fileURLWithPath: trimmed.expandingTildePath)
            return resolveSiblingFile("draft_head.qwp", near: url)
        }

        private static func resolveSiblingFile(_ filename: String, near url: URL) -> URL? {
            var isDir: ObjCBool = false
            let fm = FileManager.default
            let baseDir: URL = (fm.fileExists(atPath: url.path, isDirectory: &isDir) && isDir.boolValue)
                ? url
                : url.deletingLastPathComponent()
            let candidates = [
                baseDir.appendingPathComponent(filename),
                baseDir.appendingPathComponent("assets").appendingPathComponent(filename),
            ]
            return candidates.first { fm.fileExists(atPath: $0.path) }
        }
    }

    internal static func decodeDSparkConfig(
        text t: ConfigEnvelope.Text,
        rawRoot: [String: Any]?,
        rawText: [String: Any],
        ropeTheta: Double,
        vocab: Int,
        numLayers: Int,
        inv: (Bool, String) throws -> Void
    ) throws -> DSparkConfig? {
        let nested = (rawText["dspark_config"] as? [String: Any])
            ?? (rawRoot?["dspark_config"] as? [String: Any])
        let targetIds = t.dspark_target_layer_ids
            ?? (rawRoot?["dspark_target_layer_ids"] as? [Int])
            ?? (nested?["target_layer_ids"] as? [Int])
        let blockSize = t.dspark_block_size
            ?? (rawRoot?["dspark_block_size"] as? Int)
            ?? (nested?["block_size"] as? Int)
        let markovRank = t.dspark_markov_rank
            ?? (rawRoot?["dspark_markov_rank"] as? Int)
            ?? (nested?["markov_rank"] as? Int)
        let noiseTok = t.dspark_noise_token_id
            ?? (rawRoot?["dspark_noise_token_id"] as? Int)
            ?? (nested?["noise_token_id"] as? Int)
        let draftLayers = t.dspark_num_hidden_layers
            ?? (rawRoot?["dspark_num_hidden_layers"] as? Int)
            ?? (nested?["num_hidden_layers"] as? Int)
        let draftWindow = t.dspark_sliding_window
            ?? (rawRoot?["dspark_sliding_window"] as? Int)
            ?? (nested?["sliding_window"] as? Int)
        let draftTheta = t.dspark_rope_theta
            ?? (rawRoot?["dspark_rope_theta"] as? Double)
            ?? (nested?["rope_theta"] as? Double)
        let enableConf = t.enable_confidence_head
            ?? (rawRoot?["enable_confidence_head"] as? Bool)
            ?? (nested?["enable_confidence_head"] as? Bool)
        let sampleFromAnchor = (rawText["dspark_sample_from_anchor"] as? Bool)
            ?? (rawRoot?["dspark_sample_from_anchor"] as? Bool)
            ?? (nested?["sample_from_anchor"] as? Bool)
            ?? true

        let hasAny = targetIds != nil || blockSize != nil || markovRank != nil
            || noiseTok != nil || nested != nil
        guard hasAny else { return nil }
        guard let ids = targetIds, !ids.isEmpty else {
            throw DescriptorError.invariant("DSpark config requires non-empty dspark_target_layer_ids")
        }
        let resolvedBlock = blockSize ?? 7
        let resolvedRank = markovRank ?? 256
        let resolvedNoise = noiseTok ?? 1024
        try inv((2...16).contains(resolvedBlock), "dspark_block_size \(resolvedBlock) must be in 2...16")
        try inv(resolvedRank > 0 && resolvedRank % 32 == 0, "dspark_markov_rank \(resolvedRank) must be a positive multiple of 32")
        try inv((0..<vocab).contains(resolvedNoise), "dspark_noise_token_id \(resolvedNoise) out of vocab range 0..<\(vocab)")
        try inv(ids.allSatisfy { (0..<numLayers).contains($0) },
                "dspark_target_layer_ids \(ids) must contain 0-based layer indices in 0..<\(numLayers)")
        try inv(Set(ids).count == ids.count, "dspark_target_layer_ids \(ids) must not contain duplicates")

        let resolvedDraftLayers = draftLayers ?? ids.count
        let resolvedWindow = draftWindow ?? t.sliding_window ?? 128
        let resolvedTheta = draftTheta ?? ropeTheta
        try inv(resolvedDraftLayers > 0, "dspark_num_hidden_layers must be positive")
        try inv(resolvedWindow > 0, "dspark_sliding_window must be positive")
        try inv(resolvedTheta > 0 && resolvedTheta.isFinite, "dspark_rope_theta must be positive and finite")

        return DSparkConfig(
            blockSize: resolvedBlock,
            markovRank: resolvedRank,
            noiseTokenId: resolvedNoise,
            targetLayerIds: ids,
            numHiddenLayers: resolvedDraftLayers,
            slidingWindow: resolvedWindow,
            ropeTheta: resolvedTheta,
            sampleFromAnchor: sampleFromAnchor,
            enableConfidenceHead: enableConf
        )
    }
}

/// Online confidence-head adaptive verification budget scheduler for DSpark.
public struct QWDSparkConfidenceScheduler: Sendable {
    public private(set) var draftSecondsEMA: Double = 0
    private var verifySecondsEMA: [Double]
    private var hasDraftSample = false
    private var hasVerifySample: [Bool]
    public let maxDrafts: Int
    public let emaAlpha: Double
    public let numExperts: Int
    public let expertsPerTok: Int

    public init(
        maxDrafts: Int,
        emaAlpha: Double = 0.25,
        numExperts: Int = 0,
        expertsPerTok: Int = 0
    ) {
        let cap = max(1, maxDrafts)
        self.maxDrafts = cap
        self.emaAlpha = emaAlpha
        self.numExperts = max(0, numExperts)
        self.expertsPerTok = max(0, expertsPerTok)
        self.verifySecondsEMA = Array(repeating: 0, count: cap + 2)
        self.hasVerifySample = Array(repeating: false, count: cap + 2)
    }

    /// Relative verify cost prior at batch size `r = 1 + k` using unique active MoE experts `E*(1-(1-K/E)^r)`.
    public static func verifyCostFactor(
        verifyRows r: Int,
        numExperts: Int,
        expertsPerTok: Int
    ) -> Double {
        let rr = Double(max(1, r))
        guard numExperts > 0, expertsPerTok > 0 else {
            return 1.0 + 0.03 * (rr - 1.0)
        }
        let e = Double(numExperts)
        let k = min(e, max(1.0, Double(expertsPerTok)))
        let unique = e * (1.0 - pow(1.0 - k / e, rr))
        return 1.0 + 0.28 * (unique / k - 1.0)
    }

    /// Updates online EMA of per-step draft latency.
    public mutating func recordDraft(seconds: Double) {
        guard seconds > 0, seconds.isFinite else { return }
        if !hasDraftSample {
            draftSecondsEMA = seconds
            hasDraftSample = true
        } else {
            draftSecondsEMA = emaAlpha * seconds + (1.0 - emaAlpha) * draftSecondsEMA
        }
    }

    /// Updates online EMA of verifier latency for batch size `verifyRows = 1 + k`.
    public mutating func recordVerify(verifyRows m: Int, seconds: Double) {
        guard m >= 1, m < verifySecondsEMA.count, seconds > 0, seconds.isFinite else { return }
        if !hasVerifySample[m] {
            verifySecondsEMA[m] = seconds
            hasVerifySample[m] = true
        } else {
            verifySecondsEMA[m] = emaAlpha * seconds + (1.0 - emaAlpha) * verifySecondsEMA[m]
        }
    }

    /// Computes optimal draft budget `k* in 1...probs.count` maximizing expected tokens/sec:
    /// `E[accepted(k)] / (t_draft + t_verify(1 + k))` where `E[accepted(k)] = 1 + sum_{m=1..k} prod_{i<m} p_i`.
    public func optimalDraftCount(confidences probs: [Float], fallbackMax: Int? = nil) -> Int {
        let maxK = min(maxDrafts, min(probs.count, fallbackMax ?? probs.count))
        guard maxK >= 1 else { return 0 }
        let tDraft = hasDraftSample ? draftSecondsEMA : 1.0
        var bestK = maxK
        var bestScore = -Double.infinity
        var survival = 1.0
        var expectedTokens = 1.0
        for k in 1...maxK {
            let pPrev = Double(min(1.0, max(0.0, probs[k - 1])))
            survival *= pPrev
            expectedTokens += survival
            let m = 1 + k
            let tVer = hasVerifySample[m] ? verifySecondsEMA[m] : Self.interpolatedVerifyTime(
                m: m, emAs: verifySecondsEMA, valid: hasVerifySample, tDraft: tDraft,
                numExperts: numExperts, expertsPerTok: expertsPerTok)
            let denom = max(1e-9, tDraft + tVer)
            let score = expectedTokens / denom
            if score > bestScore {
                bestScore = score
                bestK = k
            }
        }
        return bestK
    }

    private static func interpolatedVerifyTime(
        m: Int,
        emAs: [Double],
        valid: [Bool],
        tDraft: Double,
        numExperts: Int,
        expertsPerTok: Int
    ) -> Double {
        var lo = -1
        for i in stride(from: m - 1, through: 1, by: -1) where valid[i] { lo = i; break }
        var hi = -1
        for i in (m + 1)..<valid.count where valid[i] { hi = i; break }
        let fM = verifyCostFactor(verifyRows: m, numExperts: numExperts, expertsPerTok: expertsPerTok)
        if lo >= 1 && hi > lo {
            let fLo = verifyCostFactor(verifyRows: lo, numExperts: numExperts, expertsPerTok: expertsPerTok)
            let fHi = verifyCostFactor(verifyRows: hi, numExperts: numExperts, expertsPerTok: expertsPerTok)
            let span = fHi - fLo
            let frac = span > 1e-9 ? (fM - fLo) / span : Double(m - lo) / Double(hi - lo)
            return emAs[lo] + frac * (emAs[hi] - emAs[lo])
        }
        if lo >= 1 {
            let fLo = verifyCostFactor(verifyRows: lo, numExperts: numExperts, expertsPerTok: expertsPerTok)
            return emAs[lo] * (fM / max(1e-9, fLo))
        }
        if hi >= 1 {
            let fHi = verifyCostFactor(verifyRows: hi, numExperts: numExperts, expertsPerTok: expertsPerTok)
            return emAs[hi] * (fM / max(1e-9, fHi))
        }
        let baseRatio = (numExperts > 0 && expertsPerTok > 0) ? 2.5 : 2.0
        return max(tDraft * baseRatio, 1e-6) * fM
    }
}
