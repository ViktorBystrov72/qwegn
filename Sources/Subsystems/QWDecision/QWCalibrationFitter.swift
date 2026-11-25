// QWCalibrationFitter.swift — Temperature scaling NLL minimizer, PAV isotonic regression, and 15-bin ECE/Brier diagnostics.
import Foundation

public struct QWCalibrationSample: Sendable, Equatable {
    public let logits: [Double]
    public let labelIndex: Int
    public let bucket: String

    public init(logits: [Double], labelIndex: Int, bucket: String = "default") {
        self.logits = logits
        self.labelIndex = labelIndex
        self.bucket = bucket
    }
}

public struct QWCalibrationFitMetrics: Sendable, Equatable {
    public let eceBefore: Double
    public let eceAfter: Double
    public let brierBefore: Double
    public let brierAfter: Double
    public let nllBefore: Double
    public let nllAfter: Double
    public let accuracyBefore: Double
    public let accuracyAfter: Double

    public init(
        eceBefore: Double,
        eceAfter: Double,
        brierBefore: Double,
        brierAfter: Double,
        nllBefore: Double,
        nllAfter: Double,
        accuracyBefore: Double,
        accuracyAfter: Double
    ) {
        self.eceBefore = eceBefore
        self.eceAfter = eceAfter
        self.brierBefore = brierBefore
        self.brierAfter = brierAfter
        self.nllBefore = nllBefore
        self.nllAfter = nllAfter
        self.accuracyBefore = accuracyBefore
        self.accuracyAfter = accuracyAfter
    }
}

public enum QWCalibrationFitter {
    public static func negativeLogLikelihood(samples: [QWCalibrationSample], temperature: Double) -> Double {
        guard !samples.isEmpty else { return 0 }
        let t = max(1e-4, temperature)
        var total = 0.0
        for s in samples where (0..<s.logits.count).contains(s.labelIndex) {
            let probs = QWCalibrationEngine.calibratedSoftmax(logits: s.logits, temperature: t)
            let p = max(1e-15, probs[s.labelIndex])
            total -= log(p)
        }
        return total / Double(samples.count)
    }

    public static func expectedCalibrationError(
        probs: [[Double]],
        labels: [Int],
        bins: Int = 15
    ) -> Double {
        guard !probs.isEmpty, probs.count == labels.count, bins > 0 else { return 0 }
        var binCount = [Int](repeating: 0, count: bins)
        var binConf = [Double](repeating: 0, count: bins)
        var binAcc = [Double](repeating: 0, count: bins)
        for i in probs.indices {
            let p = probs[i]
            guard !p.isEmpty else { continue }
            let pred = QWCalibrationEngine.argmaxIndex(p)
            let conf = p[pred]
            let b = min(bins - 1, max(0, Int(conf * Double(bins))))
            binCount[b] += 1
            binConf[b] += conf
            binAcc[b] += (pred == labels[i]) ? 1.0 : 0.0
        }
        let n = Double(probs.count)
        var ece = 0.0
        for b in 0..<bins where binCount[b] > 0 {
            let cnt = Double(binCount[b])
            ece += (cnt / n) * abs((binAcc[b] / cnt) - (binConf[b] / cnt))
        }
        return ece
    }

    public static func brierScore(probs: [[Double]], labels: [Int]) -> Double {
        guard !probs.isEmpty, probs.count == labels.count else { return 0 }
        var sum = 0.0
        for i in probs.indices {
            let p = probs[i]
            for j in p.indices {
                let target = (j == labels[i]) ? 1.0 : 0.0
                let diff = p[j] - target
                sum += diff * diff
            }
        }
        return sum / Double(probs.count)
    }

    /// Fits an optimal scalar temperature `T > 0` minimizing NLL via golden-section search.
    public static func fitTemperature(
        samples: [QWCalibrationSample],
        minTemp: Double = 0.05,
        maxTemp: Double = 20.0
    ) -> Double {
        let valid = samples.filter { (0..<$0.logits.count).contains($0.labelIndex) }
        guard !valid.isEmpty else { return 1.0 }
        // Optimize in log-temperature space so scaling sensitivity is symmetric across < 1 and > 1.
        var a = log(max(1e-3, minTemp))
        var b = log(max(minTemp + 0.01, maxTemp))
        let phi = (sqrt(5.0) - 1.0) / 2.0
        var c = b - phi * (b - a)
        var d = a + phi * (b - a)
        var fc = negativeLogLikelihood(samples: valid, temperature: exp(c))
        var fd = negativeLogLikelihood(samples: valid, temperature: exp(d))
        for _ in 0..<64 {
            if fc < fd {
                b = d
                d = c
                fd = fc
                c = b - phi * (b - a)
                fc = negativeLogLikelihood(samples: valid, temperature: exp(c))
            } else {
                a = c
                c = d
                fc = fd
                d = a + phi * (b - a)
                fd = negativeLogLikelihood(samples: valid, temperature: exp(d))
            }
        }
        let bestTemp = exp(0.5 * (a + b))
        return abs(bestTemp - 1.0) < 1e-4 ? 1.0 : bestTemp
    }

    /// Pool-Adjacent-Violators (PAV) monotone isotonic calibration returning sorted `[raw, calibrated]` knots.
    public static func fitPAVIsotonicKnots(
        observations: [(raw: Double, target: Double)]
    ) -> [[Double]] {
        let sorted = observations
            .filter { $0.raw.isFinite && $0.target.isFinite }
            .sorted { lhs, rhs in
                if lhs.raw != rhs.raw { return lhs.raw < rhs.raw }
                return lhs.target < rhs.target
            }
        guard !sorted.isEmpty else { return [] }

        struct Block {
            var minRaw: Double
            var maxRaw: Double
            var sumTarget: Double
            var count: Double
            var mean: Double { sumTarget / count }
        }

        var blocks: [Block] = []
        blocks.reserveCapacity(sorted.count)
        for obs in sorted {
            blocks.append(Block(minRaw: obs.raw, maxRaw: obs.raw, sumTarget: obs.target, count: 1.0))
            while blocks.count >= 2 && blocks[blocks.count - 2].mean > blocks[blocks.count - 1].mean {
                let right = blocks.removeLast()
                var left = blocks.removeLast()
                left.maxRaw = right.maxRaw
                left.sumTarget += right.sumTarget
                left.count += right.count
                blocks.append(left)
            }
        }
        var knots: [[Double]] = []
        knots.reserveCapacity(blocks.count * 2)
        for blk in blocks {
            let y = min(1.0, max(0.0, blk.mean))
            if knots.isEmpty || abs(knots[knots.count - 1][0] - blk.minRaw) > 1e-12 {
                knots.append([blk.minRaw, y])
            }
            if abs(blk.maxRaw - blk.minRaw) > 1e-12 {
                knots.append([blk.maxRaw, y])
            }
        }
        return knots
    }

    /// Fits per-bucket temperatures (`temperatures` map) and returns full pre/post metrics.
    public static func fitProfile(
        samples: [QWCalibrationSample],
        binaryTables: [String: [(raw: Double, target: Double)]] = [:]
    ) -> (profile: QWCalibrationProfile, metrics: QWCalibrationFitMetrics) {
        let grouped = Dictionary(grouping: samples, by: \.bucket)
        var temps: [String: Double] = [:]
        for (bucket, bucketSamples) in grouped {
            temps[bucket] = fitTemperature(samples: bucketSamples)
        }
        let defaultTemp = fitTemperature(samples: samples)
        var fittedTables: [String: [[Double]]] = [:]
        for (name, obs) in binaryTables {
            fittedTables[name] = fitPAVIsotonicKnots(observations: obs)
        }
        let profile = QWCalibrationProfile(
            temperatures: temps,
            defaultTemperature: defaultTemp,
            tables: fittedTables
        )

        let rawProbs = samples.map { QWCalibrationEngine.calibratedSoftmax(logits: $0.logits, temperature: 1.0) }
        let calProbs = samples.map { s in
            let t = temps[s.bucket] ?? defaultTemp
            return QWCalibrationEngine.calibratedSoftmax(logits: s.logits, temperature: t)
        }
        let labels = samples.map(\.labelIndex)
        let nllBefore = negativeLogLikelihood(samples: samples, temperature: 1.0)
        var nllAfterSum = 0.0
        for i in samples.indices {
            let p = max(1e-15, calProbs[i][samples[i].labelIndex])
            nllAfterSum -= log(p)
        }
        let nllAfter = samples.isEmpty ? 0 : nllAfterSum / Double(samples.count)
        let accBefore = samples.isEmpty ? 0 : Double(samples.indices.filter {
            QWCalibrationEngine.argmaxIndex(rawProbs[$0]) == labels[$0]
        }.count) / Double(samples.count)
        let accAfter = samples.isEmpty ? 0 : Double(samples.indices.filter {
            QWCalibrationEngine.argmaxIndex(calProbs[$0]) == labels[$0]
        }.count) / Double(samples.count)

        let metrics = QWCalibrationFitMetrics(
            eceBefore: expectedCalibrationError(probs: rawProbs, labels: labels),
            eceAfter: expectedCalibrationError(probs: calProbs, labels: labels),
            brierBefore: brierScore(probs: rawProbs, labels: labels),
            brierAfter: brierScore(probs: calProbs, labels: labels),
            nllBefore: nllBefore,
            nllAfter: nllAfter,
            accuracyBefore: accBefore,
            accuracyAfter: accAfter
        )
        return (profile, metrics)
    }
}
