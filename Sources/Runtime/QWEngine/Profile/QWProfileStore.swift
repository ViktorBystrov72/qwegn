// QWProfileStore.swift — Per-model knob values on disk.
// The store is the only place a knob value is chosen: the explicit value,
// then the profile, then the built-in default. A knob the model cannot
// honour is absent from the result — never a value that reads like a
// setting the user made.

import Foundation
import CryptoKit
import QWCore

/// A profile could not be read or written.
public enum QWProfileStoreError: Error, Equatable, CustomStringConvertible {
    case unreadableProfile(model: String, reason: String)
    case unwritableProfile(model: String, reason: String)
    case profileFromNewerSchema(model: String, schemaVersion: Int)

    public var description: String {
        switch self {
        case .unreadableProfile(let model, let reason):
            return "QWProfileStore: the profile for '\(model)' could not be read (\(reason))"
        case .unwritableProfile(let model, let reason):
            return "QWProfileStore: the profile for '\(model)' could not be written (\(reason))"
        case .profileFromNewerSchema(let model, let version):
            return "QWProfileStore: the profile for '\(model)' was written against schema "
                + "\(version), which this build does not understand"
        }
    }
}

/// One model's knob values, read from and written to disk. A profile bound
/// to another geometry is stale: its values are refused, not applied.
public final class QWProfileStore {
    /// The directory the profiles live in.
    public let root: URL
    private let lock = NSLock()

    public init(root: URL) {
        self.root = root
    }

    /// The store the app uses: one profile per model under `~/.qwg/profiles`.
    public static func standard() -> QWProfileStore {
        QWProfileStore(root: QWPaths.profilesDirectoryURL)
    }

    /// The file one model's profile lives in. A model id may hold
    /// characters a file name cannot, so the id is reduced to a safe stem
    /// and disambiguated by a digest of the whole id.
    public func profileURL(for modelId: String) -> URL {
        root.appendingPathComponent(Self.fileName(for: modelId), isDirectory: false)
    }

    /// A file name that is safe and stable for a model id.
    static func fileName(for modelId: String) -> String {
        var stem = ""
        for scalar in modelId.unicodeScalars {
            let safe = CharacterSet.alphanumerics.contains(scalar)
                || scalar == "-" || scalar == "_" || scalar == "."
            stem.append(safe ? Character(scalar) : "_")
        }
        while stem.contains("..") {
            stem = stem.replacingOccurrences(of: "..", with: "_")
        }
        if stem.isEmpty { stem = "model" }
        // A file name has a length limit. The digest keeps the name unique,
        // so a long stem can be cut without two models colliding.
        if stem.count > 120 { stem = String(stem.prefix(120)) }
        let digest = SHA256.hash(data: Data(modelId.utf8)).prefix(4)
            .map { String(format: "%02x", $0) }.joined()
        return "\(stem)-\(digest).json"
    }

    /// The profile for a model, or nil when there is none.
    public func load(modelId: String) throws -> QWKnobProfile? {
        lock.lock()
        defer { lock.unlock() }
        let url = profileURL(for: modelId)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw QWProfileStoreError.unreadableProfile(
                model: modelId,
                reason: "\(error.localizedDescription) (at \(url.path))")
        }
        let profile: QWKnobProfile
        do {
            profile = try JSONDecoder().decode(QWKnobProfile.self, from: data)
        } catch {
            throw QWProfileStoreError.unreadableProfile(
                model: modelId,
                reason: "\(error.localizedDescription) (at \(url.path))")
        }
        guard profile.schemaVersion <= QWKnobProfile.currentSchemaVersion else {
            throw QWProfileStoreError.profileFromNewerSchema(model: modelId,
                                                             schemaVersion: profile.schemaVersion)
        }
        guard profile.modelId == modelId else {
            throw QWProfileStoreError.unreadableProfile(
                model: modelId,
                reason: "the file holds the profile of '\(profile.modelId)'")
        }
        for (id, value) in profile.values {
            guard let descriptor = QWKnobRegistry.descriptor(id) else {
                throw QWProfileStoreError.unreadableProfile(model: modelId,
                                                            reason: "unknown knob '\(id)'")
            }
            do {
                try descriptor.validate(value)
            } catch {
                throw QWProfileStoreError.unreadableProfile(model: modelId,
                                                            reason: String(describing: error))
            }
        }
        return profile
    }

    /// Writes a model's profile.
    public func save(_ profile: QWKnobProfile) throws {
        // The timestamp is stored as a number, not as a formatted date, so a
        // profile that is written and read back is the same profile: a
        // format that rounded to the second would quietly move the moment
        // it names.
        lock.lock()
        defer { lock.unlock() }
        // A value the registry refuses must not reach the disk: the profile
        // would be unreadable the moment it was read back.
        for (id, value) in profile.values {
            guard let descriptor = QWKnobRegistry.descriptor(id) else {
                throw QWKnobError.unknownKnob(id)
            }
            try descriptor.validate(value)
        }
        try QWPaths.ensurePrivateDirectory(root)
        var stamped = profile
        stamped.schemaVersion = QWKnobProfile.currentSchemaVersion
        // The store owns the timestamp: the moment the file reached the disk
        // is the one the profile names.
        stamped.updatedAt = Date()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(stamped)
        let profileURL = profileURL(for: profile.modelId)
        do {
            try data.write(to: profileURL, options: .atomic)
            QWPaths.setPrivateFilePermissions(atPath: profileURL.path)
        } catch {
            throw QWProfileStoreError.unwritableProfile(model: profile.modelId,
                                                        reason: error.localizedDescription)
        }
    }

    /// The knob values for a model: the explicit value, else the profile's,
    /// else the built-in default. A knob the model cannot honour is absent
    /// from the result.
    public func resolve(modelId: String,
                        explicit: [String: QWKnobValue],
                        capabilities: Set<QWKnobCapability>) throws -> [String: QWKnobValue] {
        try resolve(modelId: modelId, explicit: explicit, capabilities: capabilities,
                    expectedDigest: nil)
    }

    /// The knob values for a model, with the geometry they must apply to.
    public func resolve(modelId: String,
                        explicit: [String: QWKnobValue],
                        capabilities: Set<QWKnobCapability>,
                        expectedDigest: String?) throws -> [String: QWKnobValue] {
        // `expectedDigest` is the **geometry** digest — the model's shape,
        // never the plan's own digest, which covers the knobs a profile
        // holds and would make a profile stale against itself.
        for id in explicit.keys where QWKnobRegistry.descriptor(id) == nil {
            throw QWKnobError.unknownKnob(id)
        }
        let profile = try load(modelId: modelId)
        if let expectedDigest, let profile,
           !profile.geometryDigest.isEmpty,
           profile.geometryDigest != expectedDigest {
            throw QWKnobError.staleProfile(model: modelId, digest: profile.geometryDigest)
        }
        var out: [String: QWKnobValue] = [:]
        for descriptor in QWKnobRegistry.all {
            if let capability = descriptor.requiredCapability, !capabilities.contains(capability) {
                if explicit[descriptor.id] != nil {
                    throw QWKnobError.capabilityUnavailable(knob: descriptor.id, model: modelId,
                                                            required: capability)
                }
                continue   // absent, not zero
            }
            let value = explicit[descriptor.id]
                ?? profile?.values[descriptor.id]
                ?? descriptor.defaultValue
            do {
                try descriptor.validate(value, capabilities: capabilities, model: modelId)
            } catch QWKnobError.capabilityUnavailable where explicit[descriptor.id] == nil {
                // A value the model cannot honour is absent, not a failure
                // of the whole profile: one knob must not cost the user
                // every other knob. An explicitly requested value is
                // refused above, because that is the user asking.
                continue
            }
            out[descriptor.id] = value
        }
        return out
    }

    /// Removes a model's profile, so the model falls back to its defaults.
    /// This is the way back from a profile that cannot be read: the store
    /// refuses it loudly, and this is the way the user says "forget it".
    /// Returns whether there was a profile to remove.
    @discardableResult
    public func forget(modelId: String) throws -> Bool {
        lock.lock()
        defer { lock.unlock() }
        let url = profileURL(for: modelId)
        guard FileManager.default.fileExists(atPath: url.path) else { return false }
        do {
            try FileManager.default.removeItem(at: url)
        } catch {
            throw QWProfileStoreError.unwritableProfile(model: modelId,
                                                        reason: error.localizedDescription)
        }
        return true
    }
}
