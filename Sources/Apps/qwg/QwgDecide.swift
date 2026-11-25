// QwgDecide.swift — `qwg decide` CLI subcommand for single-pass System-1 decisions.

import ArgumentParser
import Foundation
import QWCli
import QWCore
import QWDecision
import QWEngine
import QWServer

struct Decide: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "decide",
        abstract: "Evaluate a structured System-1 decision query locally or via a running qwg server."
    )

    @Option(name: [.customLong("model"), .customShort("m")], help: "Model directory, ~/.qwg/models name, alias, or .aimodel bundle.")
    var model: String?

    @Option(name: .customLong("state"), help: "Inline text or JSON state to evaluate.")
    var state: String?

    @Option(name: .customLong("state-file"), help: "Path to a UTF-8 file supplying the state.")
    var stateFile: String?

    @Flag(name: .customLong("json-state"), help: "Parse state or --state-file as structured JSON.")
    var jsonState: Bool = false

    @Option(name: [.customLong("question"), .customShort("q")], help: "Decision question instructions.")
    var question: String

    @Option(name: .customLong("kind"), help: "Decision kind (binary|choice|multiple|score|read|estimate|box|boxes).")
    var kind: String?

    @Option(name: .customLong("options"), help: "Repeatable option/level keys for choice, multiple, or score.")
    var options: [String] = []

    @Option(name: .customLong("labels"), help: "Comma-separated option/label list.")
    var labels: String?

    @Option(name: .customLong("min"), help: "Minimum value for estimate questions.")
    var minValue: Double?

    @Option(name: .customLong("max"), help: "Maximum value for estimate questions.")
    var maxValue: Double?

    @Option(name: .customLong("step"), help: "Step size for estimate questions.")
    var stepValue: Double?

    @Option(name: .customLong("unit"), help: "Optional unit label for read/estimate questions.")
    var unit: String?

    @Option(name: .customLong("read-kind"), help: "Read kind override (text|number|date|path).")
    var readKind: String?

    @Option(name: .customLong("max-boxes"), help: "Maximum instances for boxes questions (default 16).")
    var maxBoxes: Int?

    @Option(name: .customLong("score-readout"), help: "Score readout mode: mean (default), median, or argmax.")
    var scoreReadout: String = "mean"

    @Option(name: .customLong("truncation"), help: "Truncation policy: error, head, tail, middle, or auto_state.")
    var truncation: String?

    @Option(name: .customLong("chain"), help: "Decision head chain name (default: 'default').")
    var chain: String = "default"

    @Option(name: .customLong("calibration"), help: "Optional calibration.json path overriding pack calibration.")
    var calibrationPath: String?

    @Option(name: .customLong("compute-unit"), help: "Compute unit preference: auto, metal, or ane.")
    var computeUnit: String = "auto"

    @Option(name: .customLong("server"), help: "Optional running QWServer base URL (e.g. http://127.0.0.1:8080).")
    var serverURL: String?

    @Option(name: .customLong("image"), help: "Repeatable path to an image file for multimodal vision decisions.")
    var images: [String] = []

    @Flag(name: .customLong("load-vision"), inversion: .prefixedNo, help: "Load or disable vision tower for multimodal decisions.")
    var loadVision: Bool?

    @Flag(name: .customLong("no-profile"), help: "Bypass ~/.qwg/settings.json defaults (hermetic mode).")
    var noProfile: Bool = false

    @Flag(name: .customLong("pretty"), help: "Pretty-print JSON output.")
    var pretty: Bool = false

    func run() async throws {
        let rawStateText: String
        if let sf = stateFile {
            rawStateText = try String(contentsOfFile: sf.expandingTildePath, encoding: .utf8)
        } else {
            rawStateText = state ?? ""
        }

        var collectedOptions = options
        if let labels, !labels.isEmpty {
            let split = labels.split(separator: ",").map {
                String($0).trimmingCharacters(in: .whitespacesAndNewlines)
            }.filter { !$0.isEmpty }
            collectedOptions.append(contentsOf: split)
        }

        let resolvedKindStr = (kind ?? (collectedOptions.isEmpty ? "binary" : "choice"))
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let loadedImages = try images.map { path -> (data: Data, mimeType: String) in
            let expanded = path.expandingTildePath
            let url = URL(fileURLWithPath: expanded)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("Image file not found: '\(path)'")
            }
            let data = try Data(contentsOf: url)
            guard !data.isEmpty else {
                throw ValidationError("Image file is empty: '\(path)'")
            }
            return (data, Self.mimeType(forPath: expanded))
        }

        if let serverURL, !serverURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            try await runRemote(
                baseURL: serverURL,
                rawStateText: rawStateText,
                resolvedKindStr: resolvedKindStr,
                collectedOptions: collectedOptions,
                encodedImages: loadedImages.map { "data:\($0.mimeType);base64,\($0.data.base64EncodedString())" }
            )
            return
        }

        let settings = noProfile ? nil : QWSettingsManager.shared.loadSettings()
        let rawRef: String = if let model {
            model
        } else if let settings {
            settings.decisionModelPackPath.isEmpty ? settings.modelName : settings.decisionModelPackPath
        } else {
            ""
        }
        let resolvedPath: String
        if let settings, !rawRef.isEmpty {
            let ref = settings.resolveModelReference(rawRef, allowArbitraryFilesystemPaths: true)
            resolvedPath = ref.packPath.isEmpty ? rawRef.expandingTildePath : ref.packPath
        } else {
            resolvedPath = rawRef.expandingTildePath
        }
        guard !resolvedPath.isEmpty else {
            throw ValidationError("Pass --model <path-or-name> or --server <url>.")
        }

        let effectiveLoadVision = loadVision ?? (settings?.decisionLoadVision ?? true)
        if !loadedImages.isEmpty && !effectiveLoadVision {
            throw ValidationError("Image input provided via --image while vision is disabled (--no-load-vision).")
        }

        let resolvedComputeUnit = QWDecisionComputeUnit(rawValue: computeUnit.lowercased()) ?? .auto
        let sidecar = try await QWEngineLifecycleActor.defaultDecisionLoader(
            resolvedPath,
            resolvedComputeUnit,
            effectiveLoadVision,
            QWGPUTurnArbitrator(),
            QWServerLogger(level: .none)
        )

        var activeCalibration = sidecar.runner.calibration
        if let calPath = calibrationPath {
            let data = try Data(contentsOf: URL(fileURLWithPath: calPath.expandingTildePath))
            activeCalibration = try QWCalibrationProfile.load(fromData: data)
        }
        let runner = QWDecisionRunner(
            tokenizer: sidecar.runner.tokenizer,
            backend: sidecar.runner.backend,
            calibration: activeCalibration
        )

        let httpQ = QWDecisionHTTPQuestion(
            name: "q0",
            kind: resolvedKindStr,
            instructions: question,
            options: collectedOptions.isEmpty ? nil : collectedOptions,
            scoreReadout: scoreReadout,
            readKind: readKind,
            unit: unit,
            minValue: minValue,
            maxValue: maxValue,
            stepValue: stepValue,
            maxBoxes: maxBoxes,
            chain: chain
        )
        let domainQ = try httpQ.toDecisionQuestion(
            fallbackName: "q0",
            defaultScoreReadout: QWScoreReadout(rawValue: scoreReadout.lowercased()) ?? .mean
        )
        let domainState: QWDecisionState
        if rawStateText.isEmpty {
            domainState = .empty
        } else if jsonState {
            domainState = .json(Data(rawStateText.utf8))
        } else {
            domainState = .text(rawStateText)
        }
        let truncPolicy = truncation.flatMap { QWTruncationPolicy(rawValue: $0.lowercased()) }
        let request = QWDecisionRequest(
            state: domainState,
            questions: [domainQ],
            truncation: truncPolicy,
            chain: chain,
            images: loadedImages.map(\.data)
        )
        let response = try await runner.decide(request)
        let enc = JSONEncoder()
        enc.outputFormatting = pretty ? [.prettyPrinted, .sortedKeys] : [.sortedKeys]
        let outData = try enc.encode(response)
        print(String(decoding: outData, as: UTF8.self))
    }

    private func runRemote(
        baseURL: String,
        rawStateText: String,
        resolvedKindStr: String,
        collectedOptions: [String],
        encodedImages: [String]
    ) async throws {
        guard let url = URL(string: baseURL.trimmingCharacters(in: CharacterSet(charactersIn: "/")) + "/v1/decisions") else {
            throw ValidationError("Invalid --server URL: '\(baseURL)'")
        }
        var qObj: [String: Any] = [
            "name": "q0",
            "kind": resolvedKindStr,
            "instructions": question,
            "score_readout": scoreReadout,
            "chain": chain,
        ]
        if !collectedOptions.isEmpty { qObj["options"] = collectedOptions }
        if let minValue { qObj["min"] = minValue }
        if let maxValue { qObj["max"] = maxValue }
        if let stepValue { qObj["step"] = stepValue }
        if let unit { qObj["unit"] = unit }
        if let readKind { qObj["read_kind"] = readKind }
        if let maxBoxes { qObj["max_boxes"] = maxBoxes }

        var payload: [String: Any] = [
            "questions": [qObj],
            "chain": chain,
            "score_readout": scoreReadout,
        ]
        if let model { payload["model"] = model }
        if let truncation { payload["truncation"] = truncation }
        if !encodedImages.isEmpty { payload["images"] = encodedImages }
        if jsonState, let parsed = try? JSONSerialization.jsonObject(with: Data(rawStateText.utf8)) {
            payload["state"] = parsed
        } else if !rawStateText.isEmpty {
            payload["state"] = rawStateText
        }

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: payload)
        let (data, resp) = try await URLSession.shared.data(for: req)
        if let http = resp as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            let body = String(decoding: data, as: UTF8.self)
            FileHandle.standardError.write(Data("HTTP \(http.statusCode): \(body)\n".utf8))
            throw ExitCode(1)
        }
        if pretty,
           let obj = try? JSONSerialization.jsonObject(with: data),
           let formatted = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys]) {
            print(String(decoding: formatted, as: UTF8.self))
        } else {
            print(String(decoding: data, as: UTF8.self))
        }
    }

    private static func mimeType(forPath path: String) -> String {
        switch URL(fileURLWithPath: path).pathExtension.lowercased() {
        case "jpg", "jpeg": return "image/jpeg"
        case "webp": return "image/webp"
        case "gif": return "image/gif"
        case "tiff", "tif": return "image/tiff"
        default: return "image/png"
        }
    }
}
