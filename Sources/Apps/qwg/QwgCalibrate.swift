// QwgCalibrate.swift — `qwg calibrate` offline NLL temperature + PAV isotonic calibrator.

import ArgumentParser
import Foundation
import QWDecision

struct Calibrate: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "calibrate",
        abstract: "Fit NLL temperatures and optional PAV isotonic tables from labeled JSONL logs."
    )

    @Option(name: .customLong("input"), help: "Input JSONL file of {\"logits\":[...],\"label\":0,\"bucket\":\"choice:2\"} or {\"score\":0.8,\"target\":1}.")
    var inputPath: String

    @Option(name: .customLong("output"), help: "Destination calibration.json path.")
    var outputPath: String

    @Option(name: .customLong("chain"), help: "Chain key written into chain_temperatures (default: 'default').")
    var chain: String = "default"

    @Flag(name: .customLong("isotonic"), help: "Also fit PAV monotone isotonic regression tables for binary scores.")
    var includeIsotonic: Bool = false

    struct FitReport: Equatable {
        let profile: QWCalibrationProfile
        let metrics: QWCalibrationFitMetrics
        let sampleCount: Int
    }

    func run() throws {
        let report = try Self.fitCalibrationFile(
            inputURL: URL(fileURLWithPath: inputPath.expandingTildePath),
            outputURL: URL(fileURLWithPath: outputPath.expandingTildePath),
            chain: chain,
            includeIsotonic: includeIsotonic
        )
        print(String(
            format: "calibrated %d samples (default_T=%.4f): ECE %.4f -> %.4f | Brier %.4f -> %.4f | NLL %.4f -> %.4f -> %@",
            report.sampleCount,
            report.profile.defaultTemperature,
            report.metrics.eceBefore,
            report.metrics.eceAfter,
            report.metrics.brierBefore,
            report.metrics.brierAfter,
            report.metrics.nllBefore,
            report.metrics.nllAfter,
            outputPath
        ))
    }

    static func fitCalibrationFile(
        inputURL: URL,
        outputURL: URL,
        chain: String = "default",
        includeIsotonic: Bool = false
    ) throws -> FitReport {
        let raw = try String(contentsOf: inputURL, encoding: .utf8)
        var samples: [QWCalibrationSample] = []
        var isotonicObs: [String: [(raw: Double, target: Double)]] = [:]

        for rawLine in raw.split(whereSeparator: \.isNewline) {
            let trimmed = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty,
                  let obj = try? JSONSerialization.jsonObject(with: Data(trimmed.utf8)) as? [String: Any] else {
                continue
            }
            let bucket = (obj["bucket"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
            let resolvedBucket = (bucket?.isEmpty == false) ? bucket! : "default"

            if let rawLogits = obj["logits"] as? [NSNumber], !rawLogits.isEmpty,
               let label = (obj["label"] as? NSNumber)?.intValue {
                let logits = rawLogits.map(\.doubleValue)
                samples.append(QWCalibrationSample(logits: logits, labelIndex: label, bucket: resolvedBucket))
                if includeIsotonic, logits.count == 2 {
                    let probs = QWCalibrationEngine.calibratedSoftmax(logits: logits, temperature: 1.0)
                    let target = (label == 1) ? 1.0 : 0.0
                    isotonicObs[resolvedBucket, default: []].append((probs[1], target))
                }
            } else if let scoreNum = obj["score"] as? NSNumber,
                      let targetNum = obj["target"] as? NSNumber {
                let p = min(1.0 - 1e-6, max(1e-6, scoreNum.doubleValue))
                let logit = log(p / (1.0 - p))
                let targetInt = targetNum.doubleValue >= 0.5 ? 1 : 0
                samples.append(QWCalibrationSample(logits: [0.0, logit], labelIndex: targetInt, bucket: resolvedBucket))
                if includeIsotonic {
                    isotonicObs[resolvedBucket, default: []].append((p, Double(targetInt)))
                }
            }
        }

        guard !samples.isEmpty else {
            throw ValidationError("No valid calibration records found in '\(inputURL.path)'")
        }

        let fitted = QWCalibrationFitter.fitProfile(
            samples: samples,
            binaryTables: includeIsotonic ? isotonicObs : [:]
        )

        var chainMap: [String: [String: Double]] = [:]
        if !fitted.profile.temperatures.isEmpty {
            chainMap[chain] = fitted.profile.temperatures
        }
        let doc: [String: Any] = [
            "default_temperature": fitted.profile.defaultTemperature,
            "temperatures": fitted.profile.temperatures,
            "chain_temperatures": chainMap,
            "tables": fitted.profile.tables,
            "metrics": [
                "samples": samples.count,
                "ece_before": fitted.metrics.eceBefore,
                "ece_after": fitted.metrics.eceAfter,
                "brier_before": fitted.metrics.brierBefore,
                "brier_after": fitted.metrics.brierAfter,
                "nll_before": fitted.metrics.nllBefore,
                "nll_after": fitted.metrics.nllAfter,
                "accuracy_before": fitted.metrics.accuracyBefore,
                "accuracy_after": fitted.metrics.accuracyAfter,
            ],
        ]
        let encoded = try JSONSerialization.data(withJSONObject: doc, options: [.prettyPrinted, .sortedKeys])
        try encoded.write(to: outputURL, options: .atomic)
        return FitReport(profile: fitted.profile, metrics: fitted.metrics, sampleCount: samples.count)
    }
}
