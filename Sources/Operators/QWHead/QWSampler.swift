// QWSampler.swift — C15 sampler: the distribution path (T>0), CPU side.
//
// WHY CPU (the spec's race; DECISIONS): Apple Silicon UMA maps
// storageModeShared CPU-side (QWMRegion.pointer), so the logits hand-off
// is a memory read, not a PCIe trip — but ONLY with the exp pass on
// vecLib vexpf: a scalar fp64 exp over 61440 logits measured 3.0 ms
// (bench_head v1, attribution probe) — 12× the ≤250 µs gate. vexpf is the
import Foundation
import Accelerate
import QWCore

public final class QWSampler {
    public var config: QWSamplerConfig
    var callIndex: UInt64 = 0
    // Reused scratch (per-token call; allocator churn is C09 steady-state
    // law territory): e = exp((v − dmax)/T) fp32.
    var eScratch: [Float] = []
    var shiftedScratch: [Float] = []
    var penaltyScratch: [Float] = []
    var heapScratch: [Int] = []
    var topKScratch: [Int] = []
    var seenEpoch: [UInt32] = []
    var currentEpoch: UInt32 = 0

    public init(config: QWSamplerConfig) { self.config = config }
    public var drawsConsumed: UInt64 { callIndex }

    @inline(__always)
    static func allFiniteSIMD(_ buf: UnsafeBufferPointer<Float>) -> Bool {
        let n = buf.count
        guard n > 0, let base = buf.baseAddress else { return true }
        var maxMag: Float = 0
        var rawSum: Float = 0
        vDSP_maxmgv(base, 1, &maxMag, vDSP_Length(n))
        vDSP_sve(base, 1, &rawSum, vDSP_Length(n))
        if maxMag < 1e30 && rawSum.isFinite { return true }
        return buf.allSatisfy(\.isFinite)
    }

    @inline(__always)
    static func allFiniteNonNegSIMD(_ buf: UnsafeBufferPointer<Float>, sum: Float) -> Bool {
        let n = buf.count
        guard n > 0, let base = buf.baseAddress else { return true }
        var minVal: Float = 0
        var maxMag: Float = 0
        vDSP_minv(base, 1, &minVal, vDSP_Length(n))
        vDSP_maxmgv(base, 1, &maxMag, vDSP_Length(n))
        if minVal >= 0 && maxMag < 1e30 && sum.isFinite { return true }
        return buf.allSatisfy { $0.isFinite && $0 >= 0 }
    }

    public static func adjustedLogits(_ logits: [Float], repetitionHistory: [Int],
                                      repetitionPenalty: Float) throws -> [Float] {
        guard repetitionPenalty.isFinite, repetitionPenalty > 0,
              logits.allSatisfy(\.isFinite) else {
            throw QWSamplerError.invalid("invalid repetition penalty or logit row")
        }
        guard repetitionHistory.allSatisfy({ (0..<logits.count).contains($0) }) else {
            throw QWSamplerError.invalid("repetition history contains token outside the logit vocabulary")
        }
        var adjusted = logits
        for token in Set(repetitionHistory) {
            let value = adjusted[token]
            adjusted[token] = value < 0 ? value * repetitionPenalty : value / repetitionPenalty
        }
        guard adjusted.allSatisfy(\.isFinite) else {
            throw QWSamplerError.invalid("repetition penalty produced non-finite logits")
        }
        return adjusted
    }

    /// Sample one token from one LM-head logit row. O(N) SIMD exp pass +
    /// O(N log K) bounded selection; only the K survivors are sorted.
    public func sample(logits: [Float], repetitionHistory: [Int] = []) throws -> QWSampleResult {
        guard callIndex < UInt64.max else { throw QWSamplerError.invalid("draw counter exhausted") }
        let distribution = try distribution(logits: logits, repetitionHistory: repetitionHistory)
        let uniform = QWPhilox.u(callIndex, seed: config.seed, version: config.uniformVersion)
        let result = try distribution.sample(uniform: Double(uniform), version: config.uniformVersion)
        callIndex += 1
        return result
    }

    /// The shared entry point: one logit row in, one token id out. The draw
    /// counter lives in the sampler, so a family that owns its own sampler
    /// gets the same sequence as any other family with the same seed.
    public func sampleToken(logits: [Float], repetitionHistory: [Int] = []) throws -> Int {
        try sample(logits: logits, repetitionHistory: repetitionHistory).token
    }

    /// The same entry point over a borrowed row: the caller keeps the buffer
    /// and the sampler copies nothing.
    public func sampleToken(logits: UnsafeBufferPointer<Float>,
                             repetitionHistory: [Int] = []) throws -> Int {
        try sample(logits: logits, repetitionHistory: repetitionHistory).token
    }

    /// The greedy token of a borrowed logit row, with the repetition penalty
    /// applied. The penalised row lives in the sampler's own scratch, so the
    /// caller does not allocate one per token.
    public func penalisedGreedyToken(logits: UnsafeBufferPointer<Float>,
                                     repetitionHistory: Set<Int>) throws -> Int {
        let n = logits.count
        guard n > 0 else { throw QWSamplerError.invalid("empty logit row") }
        guard config.repetitionPenalty.isFinite, config.repetitionPenalty > 0 else {
            throw QWSamplerError.invalid("invalid repetition penalty")
        }
        guard logits.allSatisfy(\.isFinite) else {
            throw QWSamplerError.invalid("logit row contains a non-finite value")
        }
        guard repetitionHistory.allSatisfy({ (0..<n).contains($0) }) else {
            throw QWSamplerError.invalid("repetition history contains token outside the logit vocabulary")
        }
        if penaltyScratch.count != n { penaltyScratch = [Float](repeating: 0, count: n) }
        penaltyScratch.withUnsafeMutableBufferPointer { dst in
            dst.baseAddress!.update(from: logits.baseAddress!, count: n)
            for token in repetitionHistory {
                let value = dst[token]
                dst[token] = value < 0 ? value * config.repetitionPenalty
                                       : value / config.repetitionPenalty
            }
        }
        return try QWSampler.greedyToken(logits: penaltyScratch)
    }

    /// The same over an array.
    public func penalisedGreedyToken(logits: [Float],
                                     repetitionHistory: Set<Int>) throws -> Int {
        try logits.withUnsafeBufferPointer {
            try penalisedGreedyToken(logits: $0, repetitionHistory: repetitionHistory)
        }
    }

    /// Samples one token from a borrowed logit row. No copy: the row stays
    /// where the caller put it.
    public func sample(logits: UnsafeBufferPointer<Float>,
                        repetitionHistory: [Int] = []) throws -> QWSampleResult {
        guard callIndex < UInt64.max else { throw QWSamplerError.invalid("draw counter exhausted") }
        let distribution = try distribution(logits: logits, repetitionHistory: repetitionHistory)
        let uniform = QWPhilox.u(callIndex, seed: config.seed, version: config.uniformVersion)
        let result = try distribution.sample(uniform: Double(uniform), version: config.uniformVersion)
        callIndex += 1
        return result
    }

    /// Selects one token from `allowed` only: greedy takes the largest penalised logit
    /// without a draw; sampled configs apply temperature/top-k/top-p over `allowed` and
    /// consume one draw. Disallowed logits are never read.
    public func constrainedToken(logits: UnsafeBufferPointer<Float>, allowed: [Int],
                                 repetitionHistory: Set<Int> = []) throws -> Int {
        let n = logits.count
        guard n > 0 else { throw QWSamplerError.invalid("empty logit row") }
        guard !allowed.isEmpty else { throw QWTokenConstraintError.emptySupport }
        guard config.repetitionPenalty.isFinite, config.repetitionPenalty > 0 else {
            throw QWSamplerError.invalid("invalid repetition penalty")
        }
        var values = [Float](repeating: 0, count: allowed.count)
        var previous = -1
        for (index, token) in allowed.enumerated() {
            guard token > previous, token < n else {
                throw QWSamplerError.invalid("constrained support must be ascending token ids inside the vocabulary")
            }
            previous = token
            var value = logits[token]
            if config.repetitionPenalty != 1, repetitionHistory.contains(token) {
                value = value < 0 ? value * config.repetitionPenalty : value / config.repetitionPenalty
            }
            guard value.isFinite else {
                throw QWSamplerError.invalid("constrained logit is non-finite")
            }
            values[index] = value
        }
        if config.isGreedy {
            var best = 0
            for index in 1..<values.count where values[index] > values[best] { best = index }
            return allowed[best]
        }
        let temperature = config.temperature
        guard temperature.isFinite, temperature > 0, config.topK > 0,
              config.topP.isFinite, config.topP > 0, config.topP <= 1 else {
            throw QWSamplerError.invalid("invalid temperature, top-k or top-p")
        }
        guard callIndex < UInt64.max else { throw QWSamplerError.invalid("draw counter exhausted") }
        let maxValue = values.max()!
        let invT = 1 / Double(temperature)
        var weights = [Double](repeating: 0, count: values.count)
        var fullMass = 0.0
        for index in values.indices {
            let w = Foundation.exp(Double(values[index] - maxValue) * invT)
            weights[index] = w
            fullMass += w
        }
        guard fullMass.isFinite, fullMass > 0 else {
            throw QWSamplerError.invalid("invalid constrained softmax mass")
        }
        let order = Self.topIndices(weights, count: min(config.topK, weights.count))
        let topP = Double(config.topP)
        var keep: [Int] = []
        var cumulative = 0.0
        for index in order {
            if cumulative >= topP { break }
            keep.append(index)
            cumulative += weights[index] / fullMass
        }
        keep.sort()
        let ids = keep.map { allowed[$0] }
        let kept = keep.map { weights[$0] }
        let mass = kept.reduce(0, +)
        guard mass.isFinite, mass > 0 else {
            throw QWSamplerError.invalid("invalid constrained survivor mass")
        }
        let distribution = QWSamplingDistribution(vocabulary: n, ids: ids, weights: kept,
                                                  mass: mass, fullMass: fullMass)
        let uniform = QWPhilox.u(callIndex, seed: config.seed, version: config.uniformVersion)
        let token = try distribution.sampleToken(uniform: Double(uniform), version: config.uniformVersion)
        callIndex += 1
        return token
    }

    /// Indices of the `count` largest weights, descending, ties to the lower index.
}
