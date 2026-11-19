// QWSettingsManager.swift — Persistent atomic settings & per-model profile coordinator.

import Foundation
import Darwin
import QWCore
import QWEngine

struct SettingsStatFingerprint: Equatable {
    let dev: dev_t
    let ino: UInt64
    let size: off_t
    let sec: Int
    let nsec: Int

    static func probe(path: String) -> SettingsStatFingerprint? {
        var st = Darwin.stat()
        let rc: Int32 = stat(path, &st)
        guard rc == 0 else { return nil }
        return SettingsStatFingerprint(
            dev: st.st_dev,
            ino: UInt64(st.st_ino),
            size: st.st_size,
            sec: st.st_mtimespec.tv_sec,
            nsec: st.st_mtimespec.tv_nsec
        )
    }
}

/// Manager for loading and saving settings to ~/.qwg/settings.json
public final class QWSettingsManager: @unchecked Sendable {
    public static let shared = QWSettingsManager()
    let lock = NSLock()
    var cachedSettings: QWServerSettings?
    var cachedKeys: Set<String>?
    var cachedStat: SettingsStatFingerprint?
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

        let stdURL = url.standardizedFileURL
        guard let fp = SettingsStatFingerprint.probe(path: stdURL.path) else {
            if cachedURL == stdURL {
                cachedSettings = nil
                cachedKeys = nil
                cachedStat = nil
                cachedURL = nil
            }
            return (QWServerSettings(), [])
        }
        if cachedURL == stdURL, cachedStat == fp, let cachedSettings, let cachedKeys {
            return (cachedSettings, cachedKeys)
        }

        let data: Data
        do {
            data = try Data(contentsOf: stdURL)
        } catch let error as CocoaError where error.code == .fileReadNoSuchFile {
            if cachedURL == stdURL {
                cachedSettings = nil
                cachedKeys = nil
                cachedStat = nil
                cachedURL = nil
            }
            return (QWServerSettings(), [])
        }
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw QWServerConfigurationResolver.Error.invalidSettingsFile(stdURL.path)
        }
        let keys = Set(object.keys)
        var settings = try JSONDecoder().decode(QWServerSettings.self, from: data)
        if stdURL == Self.defaultSettingsFileURL.standardizedFileURL && settings.pruneRemovedModels() {
            do {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                let prunedData = try encoder.encode(settings)
                try prunedData.write(to: stdURL, options: .atomic)
                QWPaths.setPrivateFilePermissions(atPath: stdURL.path)
            } catch {
                QWServerLogger.shared.log(.warn, "Failed to prune settings at \(stdURL.path): \(error)")
            }
        }
        cachedSettings = settings
        cachedKeys = keys
        cachedStat = SettingsStatFingerprint.probe(path: stdURL.path) ?? fp
        cachedURL = stdURL
        return (settings, keys)
    }

    /// Saves settings atomically to ~/.qwg/settings.json, enforcing `0o700` dirs and `0o600` file permissions.
    public func saveSettings(_ settings: QWServerSettings, to url: URL = defaultSettingsFileURL) throws {
        lock.lock()
        defer { lock.unlock() }

        var normalized = settings
        normalized.syncActiveModelAlias()
        let stdURL = url.standardizedFileURL
        if stdURL == Self.defaultSettingsFileURL.standardizedFileURL {
            normalized.pruneRemovedModels()
        }

        try QWPaths.ensurePrivateDirectory(stdURL.deletingLastPathComponent())
        try? QWPaths.ensurePrivateDirectory(normalized.logDir.canonicalFileURL)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(normalized)
        try data.write(to: stdURL, options: .atomic)
        QWPaths.setPrivateFilePermissions(atPath: stdURL.path)
        try? saveModelProfile(for: normalized, to: stdURL)
        let keys = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]).map { Set($0.keys) }
        cachedSettings = normalized
        cachedKeys = keys
        cachedStat = SettingsStatFingerprint.probe(path: stdURL.path)
        cachedURL = stdURL
    }
}
