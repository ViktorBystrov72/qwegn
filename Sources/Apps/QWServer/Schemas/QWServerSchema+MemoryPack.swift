// QWServerSchema+MemoryPack.swift — Inspection and live indexing DTOs for `.qwcache` Memory Packs.
import Foundation
import QWEngine

/// Detailed repository `.qwg/manifest.json` + `.qwcache` inspection payload for UI and CLI preflight.
public struct QWMemoryInspectResponse: Codable, Sendable, Equatable {
    public struct ArtifactAvailability: Codable, Sendable, Equatable {
        public let artifact: QWMemoryManifest.CacheArtifact
        public let cacheFileExists: Bool
        public let modelInstalled: Bool
        public let installedModelName: String?
        public let matchesActiveModel: Bool

        public init(
            artifact: QWMemoryManifest.CacheArtifact,
            cacheFileExists: Bool,
            modelInstalled: Bool,
            installedModelName: String?,
            matchesActiveModel: Bool
        ) {
            self.artifact = artifact
            self.cacheFileExists = cacheFileExists
            self.modelInstalled = modelInstalled
            self.installedModelName = installedModelName
            self.matchesActiveModel = matchesActiveModel
        }
    }

    public struct RecommendedIndexOptions: Codable, Sendable, Equatable {
        public let kvMode: String
        public let lsaMode: String
        public let scope: String
        public let includeTests: Bool

        public init(kvMode: String, lsaMode: String, scope: String, includeTests: Bool) {
            self.kvMode = kvMode
            self.lsaMode = lsaMode
            self.scope = scope
            self.includeTests = includeTests
        }
    }

    public let status: String
    public let rootPath: String
    public let packName: String
    public let manifestExists: Bool
    public let activeModelName: String
    public let activeGeometryDigest: String?
    public let activeKVMode: String
    public let activeLSAMode: String
    public let matchingArtifact: QWMemoryManifest.CacheArtifact?
    public let availableArtifacts: [ArtifactAvailability]
    public let gitHeadSHA: String?
    public let manifestGitHeadSHA: String?
    public let commitsBehind: Int
    public let gitDiverged: Bool
    public let modifiedFiles: [String]
    public let deletedFiles: [String]
    public let candidateFileCount: Int
    public let recommendedOptions: RecommendedIndexOptions

    public init(
        status: String,
        rootPath: String,
        packName: String,
        manifestExists: Bool,
        activeModelName: String,
        activeGeometryDigest: String?,
        activeKVMode: String,
        activeLSAMode: String,
        matchingArtifact: QWMemoryManifest.CacheArtifact?,
        availableArtifacts: [ArtifactAvailability],
        gitHeadSHA: String?,
        manifestGitHeadSHA: String?,
        commitsBehind: Int,
        gitDiverged: Bool,
        modifiedFiles: [String],
        deletedFiles: [String],
        candidateFileCount: Int,
        recommendedOptions: RecommendedIndexOptions
    ) {
        self.status = status
        self.rootPath = rootPath
        self.packName = packName
        self.manifestExists = manifestExists
        self.activeModelName = activeModelName
        self.activeGeometryDigest = activeGeometryDigest
        self.activeKVMode = activeKVMode
        self.activeLSAMode = activeLSAMode
        self.matchingArtifact = matchingArtifact
        self.availableArtifacts = availableArtifacts
        self.gitHeadSHA = gitHeadSHA
        self.manifestGitHeadSHA = manifestGitHeadSHA
        self.commitsBehind = commitsBehind
        self.gitDiverged = gitDiverged
        self.modifiedFiles = modifiedFiles
        self.deletedFiles = deletedFiles
        self.candidateFileCount = candidateFileCount
        self.recommendedOptions = recommendedOptions
    }
}

/// Request payload for `POST /v1/memory/index` and `/api/memory/index`.
public struct QWMemoryIndexRequest: Codable, Sendable, Equatable {
    public let path: String
    public let packName: String?
    public let model: String?
    public let kvMode: String?
    public let lsaMode: String?
    public let scope: String?
    public let maxCortexTokens: Int?
    public let includeTests: Bool?
    public let mountAfterIndex: Bool?
    public let stream: Bool?
    public let mtpBackend: String?
    public let dflashDraftPackPath: String?

    public init(
        path: String,
        packName: String? = nil,
        model: String? = nil,
        kvMode: String? = nil,
        lsaMode: String? = nil,
        scope: String? = nil,
        maxCortexTokens: Int? = nil,
        includeTests: Bool? = nil,
        mountAfterIndex: Bool? = nil,
        stream: Bool? = nil,
        mtpBackend: String? = nil,
        dflashDraftPackPath: String? = nil
    ) {
        self.path = path
        self.packName = packName
        self.model = model
        self.kvMode = kvMode
        self.lsaMode = lsaMode
        self.scope = scope
        self.maxCortexTokens = maxCortexTokens
        self.includeTests = includeTests
        self.mountAfterIndex = mountAfterIndex
        self.stream = stream
        self.mtpBackend = mtpBackend
        self.dflashDraftPackPath = dflashDraftPackPath
    }
}

/// Final completion summary returned by `POST /v1/memory/index` and emitted in the terminal SSE event.
public struct QWMemoryIndexResult: Codable, Sendable, Equatable {
    public let packName: String
    public let rootPath: String
    public let modelName: String
    public let kvMode: String
    public let lsaMode: String
    public let fileCount: Int
    public let cortexFileCount: Int
    public let vaultFileCount: Int
    public let tokenCount: Int
    public let cacheFilePath: String
    public let cacheBytes: Int
    public let elapsedSeconds: Double
    public let mounted: QWEngineInfoResponse.MountedMemoryPackInfo?

    public init(
        packName: String,
        rootPath: String,
        modelName: String,
        kvMode: String,
        lsaMode: String,
        fileCount: Int,
        cortexFileCount: Int,
        vaultFileCount: Int,
        tokenCount: Int,
        cacheFilePath: String,
        cacheBytes: Int,
        elapsedSeconds: Double,
        mounted: QWEngineInfoResponse.MountedMemoryPackInfo? = nil
    ) {
        self.packName = packName
        self.rootPath = rootPath
        self.modelName = modelName
        self.kvMode = kvMode
        self.lsaMode = lsaMode
        self.fileCount = fileCount
        self.cortexFileCount = cortexFileCount
        self.vaultFileCount = vaultFileCount
        self.tokenCount = tokenCount
        self.cacheFilePath = cacheFilePath
        self.cacheBytes = cacheBytes
        self.elapsedSeconds = elapsedSeconds
        self.mounted = mounted
    }
}

/// Streaming progress event emitted as `data: <json>\n\n` during `POST /v1/memory/index`.
public struct QWMemoryIndexProgressEvent: Codable, Sendable, Equatable {
    public let phase: String
    public let processedFiles: Int
    public let totalFiles: Int
    public let currentFile: String?
    public let cortexFiles: Int
    public let vaultFiles: Int
    public let cortexTokens: Int
    public let prefillProcessedTokens: Int
    public let prefillTotalTokens: Int
    public let progressFraction: Double
    public let message: String
    public let result: QWMemoryIndexResult?
    public let error: String?

    public init(
        phase: String,
        processedFiles: Int = 0,
        totalFiles: Int = 0,
        currentFile: String? = nil,
        cortexFiles: Int = 0,
        vaultFiles: Int = 0,
        cortexTokens: Int = 0,
        prefillProcessedTokens: Int = 0,
        prefillTotalTokens: Int = 0,
        progressFraction: Double = 0,
        message: String,
        result: QWMemoryIndexResult? = nil,
        error: String? = nil
    ) {
        self.phase = phase
        self.processedFiles = processedFiles
        self.totalFiles = totalFiles
        self.currentFile = currentFile
        self.cortexFiles = cortexFiles
        self.vaultFiles = vaultFiles
        self.cortexTokens = cortexTokens
        self.prefillProcessedTokens = prefillProcessedTokens
        self.prefillTotalTokens = prefillTotalTokens
        self.progressFraction = progressFraction
        self.message = message
        self.result = result
        self.error = error
    }
}
