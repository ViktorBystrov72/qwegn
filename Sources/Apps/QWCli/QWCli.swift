// QWCli.swift — Pure CLI library logic for pack inspection, admission planning, and I/O.
// Implements `inspectReport`, `admissionPlanReport`, `dequantToNpy`,
// `quantizePlan`, `downloadPlan`, NumPy `.npy` serialization, and terminal
// ANSI color styling for the `qwg` command-line executable.

import Foundation
import QWCore
import QWKV
import QWQuant

/// Pure CLI logic library backing the `qwg` executable.
public enum QWCli {

    // MARK: - CLI Options & Validation

    public enum CLIError: Error, CustomStringConvertible, Equatable {
        case invalidKVMode(String)
        case unpriceableKVMode(String)
        case invalidContextBudget(Int, max: Int)
        case inputFileNotFound(String)

        public var description: String {
            switch self {
            case .invalidKVMode(let mode):
                return "Invalid KV cache mode '\(mode)' (supported: f16, q8, q4)"
            case .unpriceableKVMode(let mode):
                return "Invalid KV cache mode '\(mode)' for the admission plan (it prices f16, q8, and q4)"
            case .invalidContextBudget(let tokens, let max):
                return "Invalid context budget \(tokens) tokens (must be positive and <= \(max))"
            case .inputFileNotFound(let path):
                return "Multimodal input file not found: \(path)"
            }
        }
    }

    /// Resolves string representation of KV cache mode ("f16", "q8", "q4") to `QWKVMode`.
    /// There is no "auto": a knob value is either one the registry declares or
    /// it is not a value at all.
    public static func resolveKVMode(_ string: String) throws -> QWKVMode {
        switch string.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "f16":
            return .f16
        case "q8":
            return .q8
        case "q4":
            return .q4
        default:
            throw CLIError.invalidKVMode(string)
        }
    }

    /// Validates context budget within valid positive range up to 1M tokens.
    public static func validateContextBudget(tokens: Int, maxTokens: Int = QWKVCache.maxTokensPerSeq) throws {
        guard tokens > 0 && tokens <= maxTokens else {
            throw CLIError.invalidContextBudget(tokens, max: maxTokens)
        }
    }

    /// Multimodal command options for image and video inputs.
    public struct MultimodalInputOptions: Equatable, Sendable {
        public var imagePath: String?
        public var videoPath: String?

        public init(imagePath: String? = nil, videoPath: String? = nil) {
            self.imagePath = imagePath
            self.videoPath = videoPath
        }

        public func validate(fileManager: FileManager = .default) throws {
            if let img = imagePath, !img.isEmpty {
                guard fileManager.fileExists(atPath: img) else {
                    throw CLIError.inputFileNotFound(img)
                }
            }
            if let vid = videoPath, !vid.isEmpty {
                guard fileManager.fileExists(atPath: vid) else {
                    throw CLIError.inputFileNotFound(vid)
                }
            }
        }
    }

    // MARK: - Inspect & Admission Plan

    /// Generates the `qwg inspect <pack>` JSON report with provenance, geometry guard, and tensor table.
    public static func inspectReport(packURL: URL) throws -> String {
        let peek = try QWQuant.Pack.peekProvenance(url: packURL)
        var obj: [String: Any] = [
            "pack": packURL.lastPathComponent,
        ]
        if let mid = peek.modelId {
            obj["modelId"] = mid
            obj["geometryDigest"] = peek.geometryDigest ?? "MISSING"
        }
        let declared = peek.modelId
        let isSupportedDeclared = declared.map {
            QWModelDescriptor.Family(rawValue: $0) != nil
                || QWModelFamilyRegistry.shared.spec(forModelType: $0) != nil
        } ?? true
        if !isSupportedDeclared {
            obj["modelStatus"] = "declared '\(declared ?? "?")' — unsupported model family for tensor inspection"
            obj["tensors"] = peek.tensorCount
            obj["sourceModel"] = peek.source ?? "unknown"
            obj["sourceSha256"] = peek.sha256Source ?? "unknown"
            let data = try JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
            return String(decoding: data, as: UTF8.self)
        }
        let pack = try openPackForInspection(packURL: packURL, declaredModelId: declared)
        let prov = pack.provenance
        obj["tensors"] = pack.tensorNames.count
        obj["geometryHash"] = prov.geometryHash ?? "MISSING"
        obj["geometryGuard"] = prov.geometryHash == QWQuant.geometryHash ? "ok" : "MISMATCH"
        obj["sourceModel"] = prov.source ?? "unknown"
        obj["sourceSha256"] = prov.sha256Source ?? "unknown"
        var rows: [[String: Any]] = []
        for name in pack.tensorNames {
            let e = try pack.entry(name)
            var row: [String: Any] = ["name": name, "shape": e.shape]
            if e.hq8Quantized {
                row["kind"] = "hq8_g32_m5"
                row["rows"] = e.nRows ?? 0
                row["cols"] = e.kCols ?? 0
                row["groups"] = e.groups ?? 0
            } else if e.quantized {
                row["kind"] = "q4g"
                row["rows"] = e.nRows ?? 0
                row["cols"] = e.kCols ?? 0
                row["groups"] = e.groups ?? 0
            } else {
                row["kind"] = "bf16"
            }
            rows.append(row)
        }
        obj["table"] = rows
        let data = try JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }

    /// Generates the `qwg plan` JSON admission report pricing every resident component against the cap.
    public static func admissionPlanReport(configURL: URL,
                                           contextTokens: Int,
                                           kvMode: String = "q4",
                                           scratchMB: Int,
                                           capGB: Double?) throws -> String {
        let kv: QWAdmissionDtype
        switch kvMode.lowercased() {
        case "f16": kv = .f16
        case "q8":  kv = .q8
        case "q4":  kv = .q4
        default: throw CLIError.unpriceableKVMode(kvMode)
        }
        let descriptor = try QWModelDescriptor.decode(configJSON: try Data(contentsOf: configURL))
        let ram = QWAdmission.physicalMemoryBytes()
        let cap = capGB.map { Int64(($0 * 1_073_741_824).rounded()) } ?? Int64(ram ?? 0)
        guard cap > 0 else {
            throw CLIError.inputFileNotFound("no working-set cap (hw.memsize unreadable; pass --cap-gb)")
        }
        let plan = try QWAdmissionPlan(
            descriptor: descriptor,
            contextTokens: contextTokens,
            kvDtype: kv,
            scratchPerSeq: scratchMB * 1_048_576,
            workingSetCapBytes: Int(cap))
        var obj = plan.report()
        obj["ram_bytes"] = ram ?? NSNull()
        switch plan.admit() {
        case .admitted(let headroom):
            obj["verdict"] = "ADMIT"
            obj["headroom_bytes"] = headroom
        case .refused(let short):
            obj["verdict"] = "REFUSE"
            obj["short_by_bytes"] = short
        }
        if case .admitted = plan.admitAnotherSequence() {
            obj["marginal_verdict"] = "one_more_seq_fits"
        } else {
            obj["marginal_verdict"] = "one_more_seq_refused"
        }
        let data = try JSONSerialization.data(withJSONObject: obj,
                                              options: [.prettyPrinted, .sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }

    private static func openPackForInspection(packURL: URL, declaredModelId: String?) throws -> QWQuant.Pack {
        if declaredModelId == nil, QWModelRoot(url: packURL).configURL == nil {
            return try QWQuant.Pack(url: packURL, descriptor: .qwen38_27b)
        }
        return try QWQuant.Pack(url: packURL)
    }

    /// Dequantizes a stored pack tensor to a little-endian float32 `.npy` file.
    public static func dequantToNpy(packURL: URL, tensor: String, out: URL) throws {
        let peek = try QWQuant.Pack.peekProvenance(url: packURL)
        let pack = try openPackForInspection(packURL: packURL, declaredModelId: peek.modelId)
        let f32: [Float]
        if try pack.entry(tensor).q8Off == nil {
            f32 = try pack.dequantQ4g(tensor)
        } else {
            f32 = try pack.dequantQ8(tensor)
        }
        try writeNpy(f32, shape: try pack.entry(tensor).shape, to: out)
    }

    // MARK: - Quantize & Pull

    /// Resolved command and provenance summary for `qwg quantize`.
    public struct QuantizePlan: Equatable {
        public var command: [String]
        public var provenanceLine: String
        public init(command: [String], provenanceLine: String) {
            self.command = command; self.provenanceLine = provenanceLine
        }
    }

    public static func quantizePlan(repo: String, outPack: URL,
                                    quant: String, scriptsDir: URL,
                                    pythonURL: URL? = nil,
                                    denseQuant: String? = nil,
                                    expertQuant: String? = nil,
                                    expertDownQuant: String? = nil,
                                    ngramQuant: String? = nil) -> QuantizePlan {
        let py = pythonURL ?? scriptsDir.deletingLastPathComponent().appendingPathComponent(".venv-oracle/bin/python")
        let packPy = scriptsDir.appendingPathComponent("quantize/pack.py")
        var cmd = [py.path, packPy.path, "--model", repo, "--out", outPack.path, "--quant", quant]
        if let dq = denseQuant { cmd += ["--dense-quant", dq] }
        if let eq = expertQuant { cmd += ["--expert-quant", eq] }
        if let edq = expertDownQuant { cmd += ["--expert-down-quant", edq] }
        if let nq = ngramQuant { cmd += ["--ngram-quant", nq] }
        return QuantizePlan(
            command: cmd,
            provenanceLine: "source=\(repo) quant=\(quant) → \(outPack.lastPathComponent) (packer embeds source sha256)")
    }

    // MARK: - NumPy Writer

    /// Writes `f32` in little-endian C-order `.npy` format byte-identical to `numpy.save`.
    static func writeNpy(_ f32: [Float], shape: [Int], to url: URL) throws {
        var desc = "{'descr': '<f4', 'fortran_order': False, 'shape': ("
        for (i, d) in shape.enumerated() { desc += "\(i == 0 ? "" : ", ")\(d)" }
        if shape.count == 1 { desc += "," }
        desc += "), }"
        var hdr = Data([0x93] + Array("NUMPY".utf8) + [1, 0])
        let L = 11 + desc.utf8.count
        let pad = (64 - L % 64) % 64
        let lenField = desc.utf8.count + pad + 1
        hdr.append(UInt8(lenField & 0xff)); hdr.append(UInt8(lenField >> 8))
        hdr.append(contentsOf: desc.utf8)
        hdr.append(contentsOf: [UInt8](repeating: 0x20, count: pad))
        hdr.append(0x0a)
        var body = Data(capacity: f32.count * 4)
        f32.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in body.append(contentsOf: raw) }
        try (hdr + body).write(to: url)
    }

    // MARK: - Terminal Styling

    /// Terminal styling with automatic TTY and `NO_COLOR` environment detection.
    public struct Color: Sendable {
        public let enabled: Bool

        public init(enabled: Bool) {
            self.enabled = enabled
        }

        public static func auto(force: Bool? = nil) -> Color {
            if let f = force { return Color(enabled: f) }
            if ProcessInfo.processInfo.environment["NO_COLOR"] != nil { return Color(enabled: false) }
            if let term = ProcessInfo.processInfo.environment["TERM"], term == "dumb" { return Color(enabled: false) }
            return Color(enabled: isatty(FileHandle.standardOutput.fileDescriptor) != 0)
        }

        public var thinking: String { enabled ? "\u{001B}[2;90m" : "" }
        public var reset: String { enabled ? "\u{001B}[0m" : "" }
        public var prompt: String { enabled ? "\u{001B}[1;36m" : "" }
        public var systemNotice: String { enabled ? "\u{001B}[33m" : "" }
        public var dim: String { enabled ? "\u{001B}[2m" : "" }
        public var bold: String { enabled ? "\u{001B}[1m" : "" }
    }
}
