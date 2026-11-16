// QWMemoryManifest.swift — Manifest and error types for QWMemoryPack.
import CryptoKit
import Foundation
import QWCore

public enum QWMemoryPackError: LocalizedError, Sendable {
    case directoryNotFound(String)
    case noMatchingCacheArtifact(modelName: String, kvMode: String?)
    case cacheFileMissing(String)

    public var errorDescription: String? {
        switch self {
        case .directoryNotFound(let path):
            return "Directory not found: '\(path)'"
        case .noMatchingCacheArtifact(let modelName, let kvMode):
            return "No matching .qwcache artifact in .qwg/manifest.json for model '\(modelName)' (kvMode=\(kvMode ?? "any"))"
        case .cacheFileMissing(let path):
            return "Memory pack .qwcache file not found at '\(path)'"
        }
    }
}

public struct QWMemoryManifest: Codable, Sendable, Equatable {
    public static let currentSchemaVersion = 1

    public struct FileEntry: Codable, Sendable, Equatable {
        public let relativePath: String
        public let tier: Int
        public let sha256: String
        public let mtimeEpochSec: Int64
        public let byteCount: Int
        public let tokenOffset: Int
        public let tokenCount: Int
        public let inCortex: Bool
        public let symbols: [String]

        public init(
            relativePath: String,
            tier: Int,
            sha256: String,
            mtimeEpochSec: Int64,
            byteCount: Int,
            tokenOffset: Int,
            tokenCount: Int,
            inCortex: Bool = true,
            symbols: [String] = []
        ) {
            self.relativePath = relativePath
            self.tier = tier
            self.sha256 = sha256
            self.mtimeEpochSec = mtimeEpochSec
            self.byteCount = byteCount
            self.tokenOffset = tokenOffset
            self.tokenCount = tokenCount
            self.inCortex = inCortex
            self.symbols = symbols
        }

        private enum CodingKeys: String, CodingKey {
            case relativePath, tier, sha256, mtimeEpochSec, byteCount, tokenOffset, tokenCount, inCortex, symbols
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.relativePath = try c.decode(String.self, forKey: .relativePath)
            self.tier = try c.decode(Int.self, forKey: .tier)
            self.sha256 = try c.decode(String.self, forKey: .sha256)
            self.mtimeEpochSec = try c.decode(Int64.self, forKey: .mtimeEpochSec)
            self.byteCount = try c.decode(Int.self, forKey: .byteCount)
            self.tokenOffset = try c.decode(Int.self, forKey: .tokenOffset)
            self.tokenCount = try c.decode(Int.self, forKey: .tokenCount)
            self.inCortex = try c.decodeIfPresent(Bool.self, forKey: .inCortex) ?? true
            self.symbols = try c.decodeIfPresent([String].self, forKey: .symbols) ?? []
        }
    }

    public struct CacheArtifact: Codable, Sendable, Equatable {
        public let modelName: String
        public let geometryDigest: String
        public let kvMode: String
        public let lsaMode: String
        public let lsaSidecarDigest: String?
        public let tokenCount: Int
        public let cacheFileName: String
        public let cacheFileBytes: Int
        public let createdAtEpochSec: Int64

        public init(
            modelName: String,
            geometryDigest: String,
            kvMode: String,
            lsaMode: String,
            lsaSidecarDigest: String? = nil,
            tokenCount: Int,
            cacheFileName: String,
            cacheFileBytes: Int,
            createdAtEpochSec: Int64 = Int64(Date().timeIntervalSince1970)
        ) {
            self.modelName = modelName
            self.geometryDigest = geometryDigest
            self.kvMode = kvMode
            self.lsaMode = lsaMode
            self.lsaSidecarDigest = lsaSidecarDigest
            self.tokenCount = tokenCount
            self.cacheFileName = cacheFileName
            self.cacheFileBytes = cacheFileBytes
            self.createdAtEpochSec = createdAtEpochSec
        }
    }

    public var schemaVersion: Int
    public var packName: String
    public var rootPath: String
    public var gitHeadSHA: String?
    public var clusterBlockTokens: Int
    public var includeTests: Bool
    public var files: [FileEntry]
    public var caches: [CacheArtifact]
    public var semanticIndex: QWMemorySemanticIndex?

    public init(
        schemaVersion: Int = Self.currentSchemaVersion,
        packName: String,
        rootPath: String,
        gitHeadSHA: String? = nil,
        clusterBlockTokens: Int = 512,
        includeTests: Bool = false,
        files: [FileEntry] = [],
        caches: [CacheArtifact] = [],
        semanticIndex: QWMemorySemanticIndex? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.packName = packName
        self.rootPath = rootPath
        self.gitHeadSHA = gitHeadSHA
        self.clusterBlockTokens = clusterBlockTokens
        self.includeTests = includeTests
        self.files = files
        self.caches = caches
        self.semanticIndex = semanticIndex
    }

    public static func qwgDirectoryURL(in rootURL: URL) -> URL {
        rootURL.standardizedFileURL.appendingPathComponent(".qwg", isDirectory: true)
    }

    public static func manifestURL(in rootURL: URL) -> URL {
        qwgDirectoryURL(in: rootURL).appendingPathComponent("manifest.json", isDirectory: false)
    }

    public static func load(from rootURL: URL) throws -> QWMemoryManifest {
        let url = manifestURL(in: rootURL)
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(QWMemoryManifest.self, from: data)
    }

    public func save(to rootURL: URL) throws {
        let dir = Self.qwgDirectoryURL(in: rootURL)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: Self.manifestURL(in: rootURL), options: .atomic)
    }

    public func cacheFileURL(for artifact: CacheArtifact, rootURL: URL? = nil) -> URL {
        let base = rootURL ?? URL(fileURLWithPath: rootPath, isDirectory: true)
        return Self.qwgDirectoryURL(in: base).appendingPathComponent(artifact.cacheFileName, isDirectory: false)
    }

    /// Selects the matching `.qwcache` artifact for the given model identity or geometry digest.
    public func matchingCache(
        modelName: String? = nil,
        geometryDigest: String? = nil,
        kvMode: String? = nil,
        allowFallbackToAnyModel: Bool = false
    ) -> CacheArtifact? {
        let trimmedDigest = geometryDigest?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let trimmedName = modelName?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let hasIdentityConstraint = !trimmedDigest.isEmpty || !trimmedName.isEmpty

        var candidates: [CacheArtifact] = []
        if !trimmedDigest.isEmpty {
            candidates = caches.filter {
                !$0.geometryDigest.isEmpty &&
                    $0.geometryDigest.caseInsensitiveCompare(trimmedDigest) == .orderedSame
            }
        }
        if candidates.isEmpty, !trimmedName.isEmpty {
            let targetSlug = QWMemoryPack.safeSlug(trimmedName)
            candidates = caches.filter {
                $0.modelName.caseInsensitiveCompare(trimmedName) == .orderedSame ||
                    (!targetSlug.isEmpty && QWMemoryPack.safeSlug($0.modelName) == targetSlug)
            }
        }
        if candidates.isEmpty {
            guard !hasIdentityConstraint || allowFallbackToAnyModel else {
                return nil
            }
            candidates = caches
        }
        if let kv = kvMode?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(), !kv.isEmpty {
            return candidates.first(where: { $0.kvMode.lowercased() == kv }) ?? candidates.first
        }
        return candidates.first
    }
}
