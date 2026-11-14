// QWServerSettings+Discovery.swift — Model aliases, directory scanning, and pre-launch validation.

import Foundation
import QWCore
import QWEngine


extension QWServerSettings {
    /// Returns the persisted alias for `modelName` (or empty string if none).
    public func alias(for modelName: String) -> String {
        if let mapped = modelAliases[modelName], !mapped.isEmpty {
            return mapped
        }
        if modelName == self.modelName {
            return modelAlias.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return ""
    }

    /// Updates or removes the persisted alias for `modelName`.
    public mutating func setAlias(_ alias: String, for modelName: String) {
        let trimmed = alias.trimmingCharacters(in: .whitespacesAndNewlines)
        if !modelName.isEmpty {
            if trimmed.isEmpty {
                modelAliases.removeValue(forKey: modelName)
            } else {
                modelAliases[modelName] = trimmed
            }
        }
        if modelName == self.modelName {
            self.modelAlias = trimmed
        }
    }

    /// Synchronizes `modelAlias` and `modelAliases` for the currently active `modelName`.
    public mutating func syncActiveModelAlias() {
        let trimmed = modelAlias.trimmingCharacters(in: .whitespacesAndNewlines)
        if !modelName.isEmpty {
            if !trimmed.isEmpty {
                modelAliases[modelName] = trimmed
            } else if let mapped = modelAliases[modelName], !mapped.isEmpty {
                self.modelAlias = mapped
                return
            } else {
                modelAliases.removeValue(forKey: modelName)
            }
        }
        self.modelAlias = trimmed
    }

    /// Removes aliases of models that are no longer present in `~/.qwg/models`.
    /// Returns true when at least one orphaned alias was removed.
    @discardableResult
    public mutating func pruneRemovedModels(discovered: [DiscoveredModel] = Self.scanModelsDirectory()) -> Bool {
        let installed = Set(discovered.filter { !$0.packPath.isEmpty }.map(\.name))
        guard !installed.isEmpty else { return false }
        // The active model may be served under a name of the user's
        // choosing; the pack that backs it says whether it is still
        // there. An alias whose model is installed is not an orphan.
        let installedPacks = Set(discovered.filter { !$0.packPath.isEmpty }
            .map { $0.packPath.expandingTildePath })
        let activePack = modelPackPath.expandingTildePath
        let activeInstalled = installedPacks.contains(activePack)
        var changed = false
        for key in Array(modelAliases.keys)
        where !installed.contains(key) && !(key == modelName && activeInstalled) {
            modelAliases.removeValue(forKey: key)
            changed = true
        }
        if !installed.contains(modelName) {
            let modelsRoot = QWPaths.modelsDirectoryPath
            if !activeInstalled && (activePack.isEmpty || activePack.hasPrefix(modelsRoot)) {
                if !modelAlias.isEmpty {
                    modelAlias = ""
                    changed = true
                }
            }
        } else if let mapped = modelAliases[modelName], modelAlias != mapped {
            modelAlias = mapped
            changed = true
        }
        return changed
    }

    /// Resolves a model reference (CLI `--model` argument, model name in `~/.qwg/models`, alias, or pack path)
    /// to `(packPath, modelName, alias)`.
    public func resolveModelReference(
        _ ref: String,
        discovered: [DiscoveredModel] = Self.scanModelsDirectory()
    ) -> (packPath: String, modelName: String, alias: String) {
        let trimmed = ref.trimmingCharacters(in: .whitespacesAndNewlines)
        let activeAlias = alias(for: modelName)
        guard !trimmed.isEmpty else { return (modelPackPath.expandingTildePath, modelName, activeAlias) }
        // 0. Prefer the active model when `ref` matches its name or alias case-insensitively.
        if !modelPackPath.isEmpty && (trimmed.caseInsensitiveCompare(modelName) == .orderedSame
            || (!activeAlias.isEmpty && trimmed.caseInsensitiveCompare(activeAlias) == .orderedSame)) {
            return (modelPackPath.expandingTildePath, modelName, activeAlias)
        }
        // 1. Match by persisted alias in modelAliases
        if let (targetModel, targetAlias) = modelAliases.first(where: { $0.value == trimmed })
            ?? modelAliases.first(where: { $0.value.caseInsensitiveCompare(trimmed) == .orderedSame }) {
            if let disc = discovered.first(where: { $0.name == targetModel }), !disc.packPath.isEmpty {
                return (disc.packPath, disc.name, targetAlias)
            }
            let candidate = (QWPaths.modelsDirectoryPath as NSString).appendingPathComponent(targetModel)
            if FileManager.default.fileExists(atPath: candidate) { return (candidate, targetModel, targetAlias) }
            if targetModel == modelName && !modelPackPath.isEmpty {
                return (modelPackPath.expandingTildePath, modelName, targetAlias)
            }
        }
        if !modelAlias.isEmpty && modelAlias.caseInsensitiveCompare(trimmed) == .orderedSame {
            return (modelPackPath.expandingTildePath, modelName, modelAlias)
        }
        // 2. Match by discovered model name in ~/.qwg/models
        if let disc = discovered.first(where: { $0.name == trimmed })
            ?? discovered.first(where: { $0.name.caseInsensitiveCompare(trimmed) == .orderedSame }),
           !disc.packPath.isEmpty {
            return (disc.packPath, disc.name, alias(for: disc.name))
        }
        // 3. Direct path or relative to ~/.qwg/models or HuggingFace / MLX cache
        let fm = FileManager.default
        func resolvedTuple(_ path: String) -> (String, String, String) {
            let resolvedURL = QWModelRoot.resolveSnapshotDirectory(URL(fileURLWithPath: path), fileManager: fm)
            let name = QWModelRoot.canonicalModelName(for: resolvedURL)
            return (resolvedURL.path, name, alias(for: name))
        }
        let inModelsDir = (QWPaths.modelsDirectoryPath as NSString).appendingPathComponent(trimmed)
        if !trimmed.contains("/") && !trimmed.hasPrefix("~") && fm.fileExists(atPath: inModelsDir) {
            return resolvedTuple(inModelsDir)
        }
        let expanded = trimmed.expandingTildePath
        if !fm.fileExists(atPath: expanded) {
            let repoLeaf = trimmed.split(separator: "/").last.map(String.init) ?? trimmed
            for rootDir in ["./models", QWPaths.modelsDirectoryPath] {
                for sub in [trimmed, repoLeaf] {
                    let cand = (rootDir as NSString).appendingPathComponent(sub)
                    if fm.fileExists(atPath: cand) { return resolvedTuple(cand) }
                }
            }
            // Check HuggingFace hub cache (~/.cache/huggingface/hub/models--<org>--<repo>) or ~/.mlx-serve/models/<org>/<repo>
            let hfHub = QWPaths.hfHubCacheDirectoryURL.path
            let sanitized = trimmed.replacingOccurrences(of: "/", with: "--")
            let hfRepoCandidate = (hfHub as NSString).appendingPathComponent(
                sanitized.hasPrefix("models--") ? sanitized : "models--\(sanitized)"
            )
            if fm.fileExists(atPath: hfRepoCandidate) { return resolvedTuple(hfRepoCandidate) }
            let mlxRoot = ("~/.mlx-serve/models" as NSString).expandingTildeInPath
            let mlxCandidate = (mlxRoot as NSString).appendingPathComponent(trimmed)
            if fm.fileExists(atPath: mlxCandidate) { return resolvedTuple(mlxCandidate) }
        }
        let resolvedURL = QWModelRoot.resolveSnapshotDirectory(URL(fileURLWithPath: expanded), fileManager: fm)
        let resolvedPath = resolvedURL.path
        let canonical = QWModelRoot.canonicalModelName(for: resolvedURL)
        let derivedName = canonical.isEmpty ? modelName : canonical
        if let disc = discovered.first(where: { $0.packPath == resolvedPath || $0.packPath == expanded }) {
            return (disc.packPath, disc.name, alias(for: disc.name))
        }
        return (resolvedPath, derivedName, alias(for: derivedName))
    }

    /// Scans ~/.qwg/models and the Hugging Face hub cache for installed primary model packs.
    public static func scanModelsDirectory() -> [DiscoveredModel] {
        let dir = QWPaths.modelsDirectoryPath
        let fm = FileManager.default
        if !fm.fileExists(atPath: dir) { try? fm.createDirectory(atPath: dir, withIntermediateDirectories: true) }
        let local = QWModelCatalog.scan(directory: dir)
        let hub = QWModelCatalog.scanHuggingFaceHub(directory: QWPaths.hfHubCacheDirectoryURL.path)
        let localNames = Set(local.map(\.name))
        return (local + hub.filter { !localNames.contains($0.name) }).map { e in
            DiscoveredModel(name: e.name, packPath: e.path, family: e.family?.rawValue,
                            nativeWindow: e.nativeWindow, supported: e.supported, reason: e.reason,
                            capabilities: e.capabilities.map(\.rawValue).sorted(),
                            maxSpeculativeDrafts: e.maxSpeculativeDrafts,
                            hasLSASidecar: e.hasLSASidecar, lsaSidecarPath: e.lsaSidecarPath)
        }
    }

    /// Scans ~/.qwg/models for installed DFlash 2 and DSpine drafter model packs via `config.json`.
    public static func scanDrafterModelsDirectory() -> [DiscoveredModel] {
        let dir = QWPaths.modelsDirectoryPath
        var models: [DiscoveredModel] = []
        if let items = try? FileManager.default.contentsOfDirectory(atPath: dir) {
            for item in items.sorted() where !item.hasPrefix(".") {
                let itemPath = (dir as NSString).appendingPathComponent(item)
                let url = URL(fileURLWithPath: itemPath)
                guard QWDFlashConfig.isDrafterPack(at: url) else { continue }
                models.append(DiscoveredModel(name: QWModelRoot.canonicalModelName(for: url), packPath: itemPath))
            }
        }
        return models
    }

    /// Infers `"dspine"` vs `"dflash2"` and maximum draft count (`blockSize - 1`) from the drafter pack's `config.json`.
    public static func inferDrafterBackendAndMaxDrafts(packPath: String?) -> (backend: String, maxDrafts: Int) {
        QWDFlashConfig.inferBackendAndMaxDrafts(packPath: packPath)
    }

    /// Validates the currently selected primary model pack (fail-fast check before server launch or upon UI selection).
    public func validatePrimaryModel() throws -> QWModelDescriptor {
        let rawPath = modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let ref = resolveModelReference(rawPath.isEmpty ? modelName : rawPath)
        let candidatePath = ref.packPath.isEmpty ? modelName : ref.packPath
        let assets = try QWModelRuntimeAssets.load(packPath: candidatePath)
        return assets.descriptor
    }

    /// Validates the DFlash 2 / DSpine drafter model configuration against the selected primary model.
    public func validateDFlashConfiguration() throws {
        let descriptor = try validatePrimaryModel()
        let maxD = QWDFlashConfig.inferBackendAndMaxDrafts(packPath: dflashDraftPackPath).maxDrafts
        let drafts = max(1, min(maxD, mtpDrafts))
        try QWDFlashConfig.validateDrafter(
            targetDescriptor: descriptor,
            packPath: dflashDraftPackPath,
            drafts: drafts,
            selectorTopK: dflashSelectorTopK
        )
    }

    /// Validates both the primary model and, when DFlash 2 / DSpine speculative decoding is enabled, the drafter pack.
    public func validateBeforeLaunch() throws {
        guard !modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
              !modelName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return
        }
        let descriptor = try validatePrimaryModel()
        let isExternalDrafter = ["dflash2", "dflash", "dspine"].contains(mtpBackend.lowercased())
        if speculativeDecodingEnabled && isExternalDrafter && mtpDrafts > 0 {
            try QWDFlashConfig.validateDrafter(targetDescriptor: descriptor, packPath: dflashDraftPackPath,
                                               drafts: mtpDrafts, selectorTopK: dflashSelectorTopK)
        }
        guard QWLSAMode.resolve(lsaMode) != nil else {
            throw QWEngineError.state("Unknown LSA mode '\(lsaMode)' (expected off, auto, sidecar, svd, or qsa-cluster)")
        }
        if lsaMode.lowercased() == "sidecar" {
            let rawPath = modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
            let ref = resolveModelReference(rawPath.isEmpty ? modelName : rawPath)
            guard QWLSAConfig.resolveSidecarURL(packPath: ref.packPath,
                                                explicitSidecarPath: lsaSidecarPath.isEmpty ? nil : lsaSidecarPath) != nil else {
                throw QWEngineError.state("LSA mode 'sidecar' requires a .qlsa sidecar file in '\(ref.packPath)' or via lsaSidecarPath")
            }
        }
    }
}
