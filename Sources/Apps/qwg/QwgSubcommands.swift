// QwgSubcommands.swift — Init, Inspect, Quantize, Pull, Bench, and LsaConvert subcommands.

import ArgumentParser
import Foundation
import QWCli
import QWCore
import QWEngine
import QWHead
import QWKV
import QWQuant
import QWServer
import QWToken

// MARK: - Init

struct Init: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "init",
        abstract: "Create or rebuild the Python environment in ~/Library/Application Support/Qwegn/venv/.")
    @Flag(name: .customLong("python"), help: "Create or rebuild the Python environment for quantize.") var python: Bool = false

    func validate() throws {
        guard python else {
            throw ValidationError("Pass --python to create or rebuild the Python environment.")
        }
    }

    func run() throws {
        _ = try QWCli.ensurePythonEnvironment(supportRoot: Quantize.repoRoot(), rebuild: true)
    }
}


// MARK: - Inspect

struct Inspect: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Dump pack provenance + tensor table; optionally dequant one tensor to .npy.")
    @Argument(help: "Model directory or legacy .qwp pack.") var pack: String
    @Option(name: .customLong("dequant"), help: "Tensor name → write dequantized f32 .npy.") var dequant: String?
    @Option(name: .customLong("out"), help: "Override the .npy output path (with --dequant).") var out: String?

    func run() throws {
        let url = URL(fileURLWithPath: pack)
        if let t = dequant {
            let dst = out.map { URL(fileURLWithPath: $0) }
                ?? url.deletingPathExtension().appendingPathExtension("npy")
            try QWCli.dequantToNpy(packURL: url, tensor: t, out: dst)
            print("wrote \(dst.path)")
            return
        }
        print(try QWCli.inspectReport(packURL: url))
    }
}

// MARK: - Quantize

struct Quantize: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Shell out to the proven Python packer (Scripts/quantize/pack.py) — Swift never reimplements the quantizer.")
    @Argument(help: "Source model directory (holds *.safetensors).") var model: String
    @Option(name: .customLong("out"), help: "New output model directory (model.qwp and assets/).") var outPack: String
    @Option(name: .customLong("quant"), help: "Quantization format (q4g, hq{2,3,4,6,8}_g{32,64,128,256}, shvq{1,15,2,3,4}_g{32,64,128,256}).") var quant: String = "q4g"
    @Option(name: .customLong("dense-quant"), help: "Optional override for 2D dense/shared tensors.") var denseQuant: String?
    @Option(name: .customLong("expert-quant"), help: "Optional override for routed MoE gate/up tensors.") var expertQuant: String?
    @Option(name: .customLong("expert-down-quant"), help: "Optional override for routed MoE down_proj tensors.") var expertDownQuant: String?
    @Option(name: .customLong("ngram-quant"), help: "Optional PLE ngram table format (q8_g32, vq2_g32, or vq4_g32).") var ngramQuant: String?

    func run() throws {
        let lowerQ = quant.lowercased()
        guard lowerQ == "q4g" || lowerQ == "hq4_g32_m5"
            || lowerQ.hasPrefix("hq") || lowerQ.hasPrefix("shvq") else {
            throw ValidationError("Unsupported quant '\(quant)'; Q8 mirrors are not used by the engine.")
        }
        let repoRoot = Self.repoRoot()
        let py = try QWCli.ensurePythonEnvironment(supportRoot: repoRoot)
        let scripts = repoRoot.appendingPathComponent("Scripts")
        let plan = QWCli.quantizePlan(repo: model, outPack: URL(fileURLWithPath: outPack),
                                      quant: quant, scriptsDir: scripts, pythonURL: py,
                                      denseQuant: denseQuant, expertQuant: expertQuant,
                                      expertDownQuant: expertDownQuant, ngramQuant: ngramQuant)
        print("provenance: \(plan.provenanceLine)")
        let p = Process()
        p.executableURL = URL(fileURLWithPath: plan.command[0])
        p.arguments = Array(plan.command.dropFirst())
        let pipe = Pipe(); p.standardOutput = pipe; p.standardError = pipe
        try p.run()
        let h = pipe.fileHandleForReading
        while true {
            let d = h.availableData
            if d.isEmpty { break }
            FileHandle.standardOutput.write(d)
        }
        p.waitUntilExit()
        if p.terminationStatus != 0 { throw ExitCode(p.terminationStatus) }
    }

    static func repoRoot() -> URL {
        QWPaths.supportDirectoryURL
    }
}

// MARK: - Pull

struct Pull: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Parallel ranged download of a checkpoint from the HF hub (plan math is QWCli-downloadPlan-tested).")
    @Argument(help: "hf repo id, e.g. org/model.") var model: String
    @Option(name: .customLong("folder"), help: "Local destination directory (default: ~/.qwg/models).") var folder: String = QWPaths.modelsDirectoryPath
    @Option(name: .customLong("parts"), help: "Parallel range chunks per file.") var parts: Int = 4

    func run() throws {
        let dest = URL(fileURLWithPath: (folder as NSString).expandingTildeInPath, isDirectory: true)
        let saved = try QWCli.pullCheckpoint(repo: model, folder: dest, parts: parts)
        print("ready: \(saved.path)")
    }
}


// MARK: - Bench

struct Bench: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Ratchet gate: re-verify the m1 soak artifact + baselines (m1 --check); other names race Scripts/bench binaries.")
    @Argument(help: "Bench name: 'm1' (the M1 soak/ratchet gate) or a Scripts/bench/* component name.") var component: String = "m1"
    @Flag(name: .customLong("check"), help: "m1: re-verify newest soak log + ratchet vs baselines (CI merge gate; exit 1 on FAIL). For components: 3%-regression check after the run.") var check: Bool = false
    @Option(name: .customLong("soak-dir"), help: "Directory holding m1_soak_*.jsonl (default: cwd).") var soakDir: String?
    @Option(name: .customLong("baselines"), help: "Baseline JSON (default: Scripts/bench/baselines.json).") var baselines: String?
    @Flag(name: .customLong("run"), help: "Component benches: execute Scripts/bench/run_sweep.sh first (authoritative, GPU-quiet); default is re-verify only.") var runSweep: Bool = false

    func run() throws {
        guard component != "m1", component != "full" else {
            try runM1Gate()
            return
        }
        let fm = FileManager.default
        let repo = fm.currentDirectoryPath
        let script = repo + "/Scripts/bench/run_sweep.sh"
        guard fm.fileExists(atPath: script) else {
            FileHandle.standardError.write("qwg bench \(component): no run_sweep.sh at \(script)\n".data(using: .utf8)!)
            throw ExitCode(2)
        }
        if !runSweep {
            FileHandle.standardError.write("qwg bench \(component): component benches run via `Scripts/bench/run_sweep.sh \(component)` (GPU-quiet, operator; add --run to invoke from here)\n".data(using: .utf8)!)
            throw ExitCode(0)
        }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/bash")
        p.arguments = [script, component]
        p.standardOutput = FileHandle.standardOutput
        p.standardError = FileHandle.standardError
        try p.run(); p.waitUntilExit()
        throw ExitCode(p.terminationStatus)
    }

    func runM1Gate() throws {
        let dir = URL(fileURLWithPath: soakDir ?? FileManager.default.currentDirectoryPath)
        let baselineURL = URL(fileURLWithPath: baselines
            ?? FileManager.default.currentDirectoryPath + "/Scripts/bench/baselines.json")
        guard let log = QWCli.newestSoakLog(in: dir) else {
            FileHandle.standardError.write("[bench m1] FAIL no m1_soak_*.jsonl in \(dir.path) — run `make m1` first; the artifact IS the gate\n".data(using: .utf8)!)
            throw ExitCode(1)
        }
        let samples = try QWCli.SoakSample.load(log)
        let verdict = QWCli.verifySoak(samples: samples)
        for l in verdict.lines { print(l) }
        var pass = verdict.allPass
        if FileManager.default.fileExists(atPath: baselineURL.path) {
            let base = try QWCli.Baselines.load(baselineURL)
            let rows = QWCli.ratchetCheck(current: verdict.metrics, baseline: base)
            for l in QWCli.ratchetLines(rows) { print(l) }
            if !rows.allSatisfy({ $0.ok }) { pass = false }
        } else {
            print("[bench m1] SKIP ratchet (no \(baselineURL.lastPathComponent); m1 --ratchet-update writes it on the first authoritative pass)")
        }
        print(pass ? "[bench m1] VERDICT: PASS — soak verified, ratchet green"
                   : "[bench m1] VERDICT: FAIL")
        if !pass { throw ExitCode(1) }
    }
}


// MARK: - LsaConvert

struct LsaConvert: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "lsa-convert",
        abstract: "Generate or import a .qlsa (QLSA0001) Latent Sparse Attention projection sidecar for a model directory.")

    @Argument(help: "Model directory, name in ~/.qwg/models, alias, or .qwp pack.") var model: String
    @Option(name: .customLong("method"), help: "Conversion method: svd (zero-data weight SVD), pca (activation PCA), distill (ridge distillation), or import.") var method: String = "svd"
    @Option(name: .customLong("out"), help: "Output .qlsa sidecar file path (default: <model_dir>/model.qlsa).") var out: String?
    @Option(name: .customLong("calibration-file"), help: "Text file for activation calibration (with --method pca or --method distill).") var calibrationFile: String?
    @Option(name: .customLong("import-path"), help: "External safetensors/npy file to import (with --method import).") var importPath: String?
    @Option(name: .customLong("index-dim"), help: "Latent index dimension d_idx (default: 64).") var indexDim: Int = QWLSAConfig.defaultIndexDim
    @Option(name: .customLong("recommended-budget"), help: "Recommended LSA key budget k (default: 2048).") var recommendedBudget: Int = QWLSAConfig.defaultBudget

    func run() throws {
        let settings = QWSettingsManager.shared.loadSettings()
        let resolvedPack = settings.resolveModelReference(model).packPath
        let targetPath = resolvedPack.isEmpty ? model.expandingTildePath : resolvedPack
        let supportRoot = Quantize.repoRoot()
        let scriptPath = supportRoot.appendingPathComponent("Scripts/convert_lsa_sidecar.py").path
        if FileManager.default.fileExists(atPath: scriptPath) && method.lowercased() != "svd-native" {
            let py = try QWCli.ensurePythonEnvironment(supportRoot: supportRoot)
            var args = [scriptPath, "--model", targetPath, "--method", method.lowercased(),
                        "--index-dim", "\(indexDim)", "--recommended-budget", "\(recommendedBudget)"]
            if let out { args += ["--out", out.expandingTildePath] }
            if let calibrationFile { args += ["--calibration-file", calibrationFile.expandingTildePath] }
            if let importPath { args += ["--import-path", importPath.expandingTildePath] }
            let p = Process()
            p.executableURL = py
            p.arguments = args
            p.standardOutput = FileHandle.standardOutput
            p.standardError = FileHandle.standardError
            try p.run()
            p.waitUntilExit()
            if p.terminationStatus != 0 { throw ExitCode(p.terminationStatus) }
            return
        }
        // Native Accelerate LAPACK SVD synthesis fallback for .qwp packs and .safetensors checkpoints
        let assets = try QWModelRuntimeAssets.load(packPath: targetPath)
        let desc = assets.descriptor
        let layerTensors = try QWLSASidecar.synthesizeFromModel(packPath: targetPath, descriptor: desc, indexDim: indexDim)
        let root = QWModelRoot(url: URL(fileURLWithPath: targetPath))
        let outURL = out.map { URL(fileURLWithPath: $0.expandingTildePath) }
            ?? root.directory.appendingPathComponent("model.qlsa")
        let prov = QWLSASidecar.Header.Provenance(
            modelId: root.directory.lastPathComponent,
            geometryDigest: desc.geometryDigestV3Hex,
            method: "weight_svd",
            indexDim: indexDim,
            poolSize: QWLSAConfig.poolSize,
            clusterBlockSize: QWLSAConfig.defaultClusterBlock,
            bucketsPerBlock: QWLSAConfig.bucketsPerBlock,
            maxBucketKeys: QWLSAConfig.maxBucketKeys,
            recommendedBudget: recommendedBudget
        )
        let shape = [indexDim, desc.lsaInputDim]
        try QWLSASidecar.write(to: outURL, provenance: prov, layerTensors: layerTensors, wqShape: shape, wkShape: shape)
        print("wrote \(outURL.path) (\(layerTensors.count) full-attention layers, shape=\(shape), method=weight_svd)")
    }
}
