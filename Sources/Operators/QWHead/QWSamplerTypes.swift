// QWSamplerTypes.swift — Sampler configuration, sample result, error, and sparse distribution.
import Foundation
import QWCore

public struct QWSamplerConfig: Sendable, Hashable {
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var seed: UInt64
    public var uniformVersion: QWPhilox.UniformVersion
    public var repetitionPenalty: Float
    public var hasExplicitSeed: Bool
    public init(temperature: Float = 0.7, topK: Int = 64, topP: Float = 0.9,
                seed: UInt64 = 0, uniformVersion: QWPhilox.UniformVersion = .legacyRounded32,
                repetitionPenalty: Float = 1.0, hasExplicitSeed: Bool = false) {
        self.temperature = temperature; self.topK = topK
        self.topP = topP; self.seed = seed
        self.uniformVersion = uniformVersion
        self.repetitionPenalty = repetitionPenalty
        self.hasExplicitSeed = hasExplicitSeed
    }

    public static func == (lhs: QWSamplerConfig, rhs: QWSamplerConfig) -> Bool {
        lhs.temperature == rhs.temperature && lhs.topK == rhs.topK && lhs.topP == rhs.topP
            && lhs.seed == rhs.seed && lhs.uniformVersion == rhs.uniformVersion
            && lhs.repetitionPenalty == rhs.repetitionPenalty
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(temperature); hasher.combine(topK); hasher.combine(topP)
        hasher.combine(seed); hasher.combine(uniformVersion); hasher.combine(repetitionPenalty)
    }

    /// True when the configuration asks for deterministic greedy argmax decoding.
    /// The repo's canonical rule, the same one `QWInferenceParameters.isGreedy`
    /// uses: never greedy on one path and stochastic on another.
    public var isGreedy: Bool {
        temperature <= 0 || topK == 1 || (topP > 0 && topP <= 1e-5)
    }

    /// Deterministic argmax decoding.
    public static let greedy = QWSamplerConfig(temperature: 0, topK: 1, topP: 1,
                                               seed: 0, repetitionPenalty: 1.0)

    /// Builds the sampler's contract from the request's sampling parameters.
    /// It mirrors the dense path's own normalisation exactly, so a request
    /// means the same thing on either family.
    public init(_ parameters: QWSamplingParameters, vocabulary: Int,
                uniformVersion: QWPhilox.UniformVersion = .upper24) {
        let greedyWithPenalty = parameters.isGreedyWithRepetitionPenalty
        self.init(temperature: greedyWithPenalty ? 1 : Float(parameters.temperature),
                  topK: greedyWithPenalty ? 1 : (parameters.topK > 0 ? parameters.topK : vocabulary),
                  topP: greedyWithPenalty ? 1 : Float(parameters.topP),
                  seed: parameters.resolvedSeed(),
                  uniformVersion: uniformVersion,
                  repetitionPenalty: Float(parameters.repetitionPenalty),
                  hasExplicitSeed: parameters.hasExplicitSeed)
    }
}

public struct QWSampleResult: Sendable {
    public let token: Int
    public let logprob: Float
    /// The renormalized nucleus (ids ascending, probabilities sum to 1) —
    /// C16's verify path consumes the distribution directly.
    public let survivorIds: [Int]
    public let survivorProbs: [Float]
}

public enum QWSamplerError: Error, CustomStringConvertible {
    case invalid(String)
    public var description: String {
        switch self { case .invalid(let reason): return "QWSampler: \(reason)" }
    }
}

/// Immutable transformed support, ascending by token ID. Construction owns no
/// draws; speculative acceptance can retain the exact Double probabilities.
public struct QWSamplingDistribution: Sendable {
    public let vocabulary: Int
    public let survivorIds: [Int]
    public let survivorProbs: [Double]
    private let weights: [Double]
    private let mass: Double
    private let fullMass: Double

    init(vocabulary: Int, ids: [Int], weights: [Double], mass: Double, fullMass: Double) {
        self.vocabulary = vocabulary
        survivorIds = ids
        survivorProbs = weights.map { $0 / mass }
        self.weights = weights
        self.mass = mass
        self.fullMass = fullMass
    }

    public static func sparse(vocabulary: Int, ids: [Int], weights: [Double]) throws -> QWSamplingDistribution {
        guard ids.count == weights.count, !ids.isEmpty, vocabulary > 0 else {
            throw QWSamplerError.invalid("sparse distribution ids and weights count mismatch")
        }
        var merged: [Int: Double] = [:]
        for (id, w) in zip(ids, weights) {
            guard (0..<vocabulary).contains(id), w.isFinite, w > 0 else {
                throw QWSamplerError.invalid("sparse distribution token ID or weight invalid")
            }
            merged[id, default: 0] += w
        }
        let sorted = merged.sorted { $0.key < $1.key }
        let sortedIds = sorted.map(\.key)
        let sortedWeights = sorted.map(\.value)
        let mass = sortedWeights.reduce(0, +)
        guard mass.isFinite, mass > 0 else {
            throw QWSamplerError.invalid("sparse distribution mass invalid")
        }
        return QWSamplingDistribution(vocabulary: vocabulary, ids: sortedIds,
                                      weights: sortedWeights, mass: mass, fullMass: mass)
    }

    public func sample(uniform: Double,
                       version: QWPhilox.UniformVersion = .upper24) throws -> QWSampleResult {
        let selected = try selectedIndex(uniform: uniform, version: version)
        return QWSampleResult(token: survivorIds[selected],
            logprob: Float(Foundation.log(weights[selected]) - Foundation.log(fullMass)),
            survivorIds: survivorIds, survivorProbs: survivorProbs.map(Float.init))
    }

    public func sampleToken(uniform: Double,
                            version: QWPhilox.UniformVersion = .upper24) throws -> Int {
        survivorIds[try selectedIndex(uniform: uniform, version: version)]
    }

    private func selectedIndex(uniform: Double, version: QWPhilox.UniformVersion) throws -> Int {
        guard uniform.isFinite, uniform >= 0,
              version == .legacyRounded32 ? uniform <= 1 : uniform < 1 else {
            throw QWSamplerError.invalid("uniform outside versioned interval")
        }
        var selected = weights.count - 1
        var cumulative = 0.0
        for index in weights.indices {
            cumulative += version == .legacyRounded32 ? weights[index] : survivorProbs[index]
            let boundary = version == .legacyRounded32 ? uniform * mass : uniform
            if version == .legacyRounded32 ? cumulative >= boundary : cumulative > boundary {
                selected = index
                break
            }
        }
        return selected
    }
}

