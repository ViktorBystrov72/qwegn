// QWSettingsManager.swift — Persistent atomic settings & per-model profile coordinator.

import Foundation
import QWCore
import QWEngine

/// Manager for loading and saving settings to ~/.qwg/settings.json
public final class QWSettingsManager: @unchecked Sendable {
    public static let shared = QWSettingsManager()
    let lock = NSLock()
    var cachedSettings: QWServerSettings?
    var cachedBytes: Data?
    var cachedURL: URL?

    public static var defaultSettingsFileURL: URL { QWPaths.settingsFileURL }
    public init() {}

    /// Returns the per-model `QWProfileStore` associated with a settings file URL.
    public func profileStore(forSettingsURL url: URL = defaultSettingsFileURL) -> QWProfileStore {
        QWProfileStore(root: url.deletingLastPathComponent().appendingPathComponent("profiles", isDirectory: true))
    }

    /// Loads the dedicated per-model knob profile for `modelName`, if one exists on disk.
    public func loadProfile(forModel modelName: String, from url: URL = defaultSettingsFileURL) -> QWKnobProfile? {
        let trimmed = modelName.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : (try? profileStore(forSettingsURL: url).load(modelId: trimmed))
    }

    /// Removes the dedicated per-model knob profile for `modelName`, if present.
    public func removeProfile(forModel modelName: String, from url: URL = defaultSettingsFileURL) {
        let trimmed = modelName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        try? FileManager.default.removeItem(at: profileStore(forSettingsURL: url).profileURL(for: trimmed))
    }

    /// Resolves effective settings for `modelName` / `packPath`: starts from the model's config files + engine defaults,
    /// and overlays the dedicated per-model profile from `profiles/` if one exists.
    public func effectiveSettings(
        forModel modelName: String, packPath: String, base: QWServerSettings,
        from url: URL = defaultSettingsFileURL, ignoreProfile: Bool = false
    ) -> (settings: QWServerSettings, profile: QWKnobProfile?) {
        var resolved = base
        resolved.modelName = modelName
        if !packPath.isEmpty { resolved.modelPackPath = packPath }
        resolved.modelAlias = resolved.alias(for: modelName)
        resolved.applyModelConfigDefaults(packPath: packPath)
        let appliedProfile: QWKnobProfile?
        if !ignoreProfile, let profile = loadProfile(forModel: modelName, from: url) {
            resolved.apply(knobProfile: profile)
            appliedProfile = profile
        } else {
            appliedProfile = nil
        }
        let descPack = resolved.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let targetDescriptor = QWServerSettings.loadPrimaryDescriptor(
            packPath: descPack.isEmpty ? packPath : descPack
        )
        resolved.reconcileStaleDrafterSelection(for: targetDescriptor)
        return (resolved, appliedProfile)
    }

    /// Persists customized per-model knob values for `settings.modelName` into `profiles/`.
    private func saveModelProfile(for settings: QWServerSettings, to url: URL) throws {
        let modelId = settings.modelName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !modelId.isEmpty else { return }
        let rawPath = settings.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let ref = settings.resolveModelReference(rawPath.isEmpty ? modelId : rawPath)
        var baseline = settings
        baseline.applyModelConfigDefaults(packPath: ref.packPath)
        let baseKnobs = baseline.validatedKnobValues
        let currentKnobs = settings.validatedKnobValues
        let store = profileStore(forSettingsURL: url)
        let existing = try? store.load(modelId: modelId)
        var diff: [String: QWKnobValue] = [:]
        for (id, value) in currentKnobs {
            if baseKnobs[id] != value || existing?.values[id] != nil {
                diff[id] = value
            }
        }
        let profile = QWKnobProfile(modelId: modelId, values: diff)
        try store.save(profile)
    }

    /// Loads settings from ~/.qwg/settings.json, falling back to defaults.
    public func loadSettings(from url: URL = defaultSettingsFileURL) -> QWServerSettings {
        do {
            return try loadSettingsSnapshot(from: url).settings
        } catch {
            QWServerLogger.shared.log(.warn, "Failed to load settings from \(url.path): \(error)")
            return QWServerSettings()
        }
    }

    func loadSettingsSnapshot(from url: URL) throws -> (settings: QWServerSettings, keys: Set<String>) {
        lock.lock()
        defer { lock.unlock() }

        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch let error as CocoaError where error.code == .fileReadNoSuchFile {
            cachedSettings = nil
            cachedBytes = nil
            cachedURL = nil
            return (QWServerSettings(), [])
        }
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw QWServerConfigurationResolver.Error.invalidSettingsFile(url.path)
        }
        let keys = Set(object.keys)
        if let cached = cachedSettings, cachedBytes == data, cachedURL == url.standardizedFileURL {
            return (cached, keys)
        }

        var settings = try JSONDecoder().decode(QWServerSettings.self, from: data)
        var persistedBytes = data
        if url.standardizedFileURL == Self.defaultSettingsFileURL.standardizedFileURL && settings.pruneRemovedModels() {
            do {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                let prunedData = try encoder.encode(settings)
                try prunedData.write(to: url, options: .atomic)
                QWPaths.setPrivateFilePermissions(atPath: url.path)
                persistedBytes = prunedData
            } catch {
                QWServerLogger.shared.log(.warn, "Failed to prune settings at \(url.path): \(error)")
            }
        }
        cachedSettings = settings
        cachedBytes = persistedBytes
        cachedURL = url.standardizedFileURL
        return (settings, keys)
    }

    /// Saves settings atomically to ~/.qwg/settings.json, enforcing `0o700` dirs and `0o600` file permissions.
    public func saveSettings(_ settings: QWServerSettings, to url: URL = defaultSettingsFileURL) throws {
        lock.lock()
        defer { lock.unlock() }

        var normalized = settings
        normalized.syncActiveModelAlias()
        if url == Self.defaultSettingsFileURL {
            normalized.pruneRemovedModels()
        }

        try QWPaths.ensurePrivateDirectory(url.deletingLastPathComponent())
        try? QWPaths.ensurePrivateDirectory(normalized.logDir.canonicalFileURL)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(normalized)
        try data.write(to: url, options: .atomic)
        QWPaths.setPrivateFilePermissions(atPath: url.path)
        try? saveModelProfile(for: normalized, to: url)
        cachedSettings = normalized
        cachedBytes = data
        cachedURL = url.standardizedFileURL
    }
}
