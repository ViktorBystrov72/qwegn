// QWUpdateChecker.swift — GitHub Releases checker with ETag cache, redirect fallback, and schedule rules.

import Foundation

/// Transport abstraction allowing deterministic offline mocking of HTTP update checks and downloads.
public protocol QWUpdateHTTPTransport: Sendable {
    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse)
    func resolveRedirectLocation(for url: URL, userAgent: String) async throws -> URL?
    func downloadToFile(
        for request: URLRequest,
        destinationPartURL: URL,
        maxBytes: Int64,
        allowedHosts: Set<String>,
        allowLoopbackHTTP: Bool,
        onProgress: @escaping @Sendable (Int64, Int64?) -> Void
    ) async throws -> Int64
}

/// Default `URLSession`-backed transport with redirect allowlisting and native streaming progress.
public final class QWURLSessionUpdateTransport: NSObject, QWUpdateHTTPTransport, @unchecked Sendable {
    public static let shared = QWURLSessionUpdateTransport()

    private let session: URLSession

    public override init() {
        let config = URLSessionConfiguration.ephemeral
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        config.timeoutIntervalForRequest = 15
        config.timeoutIntervalForResource = 600
        self.session = URLSession(configuration: config)
        super.init()
    }

    public func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw QWUpdateError.invalidResponse("Non-HTTP response")
        }
        return (data, http)
    }

    public func resolveRedirectLocation(for url: URL, userAgent: String) async throws -> URL? {
        var req = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 15)
        req.httpMethod = "HEAD"
        req.setValue(userAgent, forHTTPHeaderField: "User-Agent")

        let delegate = SingleHopRedirectDelegate()
        let hopSession = URLSession(configuration: .ephemeral, delegate: delegate, delegateQueue: nil)
        defer { hopSession.invalidateAndCancel() }
        let (_, response) = try await hopSession.data(for: req)
        if let redirected = delegate.capturedLocation() {
            return redirected
        }
        guard let http = response as? HTTPURLResponse else { return nil }
        if let loc = http.value(forHTTPHeaderField: "Location"), let locURL = URL(string: loc, relativeTo: url)?.absoluteURL {
            return locURL
        }
        return http.url
    }

    public func downloadToFile(
        for request: URLRequest,
        destinationPartURL: URL,
        maxBytes: Int64,
        allowedHosts: Set<String>,
        allowLoopbackHTTP: Bool,
        onProgress: @escaping @Sendable (Int64, Int64?) -> Void
    ) async throws -> Int64 {
        let delegate = StreamingDownloadDelegate(
            destinationPartURL: destinationPartURL,
            maxBytes: maxBytes,
            allowedHosts: allowedHosts,
            allowLoopbackHTTP: allowLoopbackHTTP,
            onProgress: onProgress
        )
        let dlSession = URLSession(configuration: .ephemeral, delegate: delegate, delegateQueue: nil)
        defer { dlSession.finishTasksAndInvalidate() }
        return try await delegate.run(request: request, on: dlSession)
    }
}

private final class SingleHopRedirectDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var location: URL?

    func capturedLocation() -> URL? {
        lock.withLock { location }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        lock.withLock { location = request.url }
        completionHandler(nil)
    }
}

private final class StreamingDownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let destinationPartURL: URL
    private let maxBytes: Int64
    private let allowedHosts: Set<String>
    private let allowLoopbackHTTP: Bool
    private let onProgress: @Sendable (Int64, Int64?) -> Void
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Int64, Error>?
    private var task: URLSessionDownloadTask?
    private var writtenBytes: Int64 = 0
    private var moveError: Error?
    private var sizeExceeded = false
    private var disallowedRedirectHost: String?

    init(
        destinationPartURL: URL,
        maxBytes: Int64,
        allowedHosts: Set<String>,
        allowLoopbackHTTP: Bool,
        onProgress: @escaping @Sendable (Int64, Int64?) -> Void
    ) {
        self.destinationPartURL = destinationPartURL
        self.maxBytes = maxBytes
        self.allowedHosts = allowedHosts
        self.allowLoopbackHTTP = allowLoopbackHTTP
        self.onProgress = onProgress
    }

    func run(request: URLRequest, on session: URLSession) async throws -> Int64 {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { cont in
                let dlTask = session.downloadTask(with: request)
                lock.withLock {
                    self.continuation = cont
                    self.task = dlTask
                }
                dlTask.resume()
            }
        } onCancel: { [weak self] in
            self?.cancelActiveTask()
        }
    }

    private func cancelActiveTask() {
        let active = lock.withLock { task }
        active?.cancel()
    }

    private func resumeOnce(with result: Result<Int64, Error>) {
        let cont: CheckedContinuation<Int64, Error>? = lock.withLock {
            let current = continuation
            continuation = nil
            task = nil
            return current
        }
        cont?.resume(with: result)
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        guard let url = request.url,
              QWUpdateChecker.isAllowedUpdateURL(
                  url,
                  extraAllowedHosts: allowedHosts,
                  allowLoopbackHTTP: allowLoopbackHTTP
              ) else {
            let host = request.url?.host ?? "(unknown)"
            lock.withLock { disallowedRedirectHost = host }
            completionHandler(nil)
            task.cancel()
            return
        }
        completionHandler(request)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        let expected: Int64? = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : nil
        if totalBytesWritten > maxBytes || (expected ?? 0) > maxBytes {
            lock.withLock { sizeExceeded = true }
            downloadTask.cancel()
            return
        }
        lock.withLock { writtenBytes = totalBytesWritten }
        onProgress(totalBytesWritten, expected)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        if let http = downloadTask.response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            lock.withLock {
                moveError = QWUpdateError.networkFailure("Download HTTP \(http.statusCode)")
            }
            return
        }
        let fm = FileManager.default
        do {
            try fm.createDirectory(at: destinationPartURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            if fm.fileExists(atPath: destinationPartURL.path) {
                try fm.removeItem(at: destinationPartURL)
            }
            try fm.moveItem(at: location, to: destinationPartURL)
            let attrs = try fm.attributesOfItem(atPath: destinationPartURL.path)
            let fileSize = (attrs[.size] as? NSNumber)?.int64Value ?? 0
            lock.withLock { writtenBytes = fileSize }
        } catch {
            lock.withLock { moveError = error }
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let (finalBytes, mErr, exceeded, badHost) = lock.withLock {
            (writtenBytes, moveError, sizeExceeded, disallowedRedirectHost)
        }
        if let badHost {
            resumeOnce(with: .failure(QWUpdateError.disallowedHost(badHost)))
        } else if exceeded {
            resumeOnce(with: .failure(QWUpdateError.fileTooLarge(limitBytes: maxBytes)))
        } else if let mErr {
            resumeOnce(with: .failure(mErr))
        } else if let error {
            resumeOnce(with: .failure(error))
        } else {
            resumeOnce(with: .success(finalBytes))
        }
    }
}

/// Checks GitHub Releases for newer Qwegn versions using ETag caching and rate-limit fallback.
public struct QWUpdateChecker: Sendable {
    public static let defaultRepo = "qwegnai/qwegn"
    public static let defaultAllowedHosts: Set<String> = [
        "api.github.com",
        "github.com",
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
        "github-releases.githubusercontent.com",
    ]
    public static let defaultAPIURL = URL(string: "https://api.github.com/repos/qwegnai/qwegn/releases/latest")
        ?? URL(fileURLWithPath: "/")
    public static let defaultLatestPageURL = URL(string: "https://github.com/qwegnai/qwegn/releases/latest")
        ?? URL(fileURLWithPath: "/")

    public let apiURL: URL
    public let latestPageURL: URL
    public let extraAllowedHosts: Set<String>
    public let allowLoopbackHTTP: Bool
    public let transport: any QWUpdateHTTPTransport

    public init(
        apiURL: URL? = nil,
        latestPageURL: URL = defaultLatestPageURL,
        extraAllowedHosts: Set<String> = [],
        allowLoopbackHTTP: Bool? = nil,
        transport: any QWUpdateHTTPTransport = QWURLSessionUpdateTransport.shared
    ) {
        let envAPI = ProcessInfo.processInfo.environment["QWEGN_UPDATE_API_URL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let envURL = envAPI.flatMap { $0.isEmpty ? nil : URL(string: $0) }
        let resolvedAPI = apiURL ?? envURL ?? Self.defaultAPIURL
        self.apiURL = resolvedAPI
        self.latestPageURL = latestPageURL

        var hosts = extraAllowedHosts
        if let h = resolvedAPI.host?.lowercased() {
            hosts.insert(h)
        }
        if let h = latestPageURL.host?.lowercased() {
            hosts.insert(h)
        }
        self.extraAllowedHosts = hosts
        let isCustomAPI = apiURL != nil || envURL != nil
        self.allowLoopbackHTTP = allowLoopbackHTTP ?? isCustomAPI
        self.transport = transport
    }

    public static func isDisabledByEnvironment(
        _ env: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        guard let raw = env["QWEGN_NO_UPDATE_CHECK"]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() else {
            return false
        }
        return raw == "1" || raw == "true" || raw == "yes"
    }

    public static func isAllowedUpdateURL(
        _ url: URL,
        extraAllowedHosts: Set<String> = [],
        allowLoopbackHTTP: Bool = false
    ) -> Bool {
        guard let scheme = url.scheme?.lowercased(),
              let host = url.host?.lowercased() else {
            return false
        }
        let isLoopback = host == "127.0.0.1" || host == "localhost" || host == "::1" || host == "[::1]"
        if scheme == "http" {
            guard allowLoopbackHTTP && isLoopback else { return false }
        } else if scheme != "https" {
            return false
        }
        return defaultAllowedHosts.contains(host) || extraAllowedHosts.contains(host)
    }

    public static func shouldPerformAutomaticCheck(
        schedule: QWUpdateCheckSchedule,
        trigger: QWUpdateCheckTrigger,
        state: QWUpdateState,
        now: Date = Date(),
        env: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        guard !isDisabledByEnvironment(env) else { return false }
        if trigger == .manual { return true }
        guard schedule != .manual else { return false }

        switch trigger {
        case .launch:
            switch schedule {
            case .manual:
                return false
            case .daily:
                guard let last = state.lastCheckedAt else { return true }
                return now.timeIntervalSince(last) >= QWUpdateCheckSchedule.dailyIntervalSeconds
            case .onLaunchAndDaily, .onLaunch:
                guard let last = state.lastCheckedAt else { return true }
                return now.timeIntervalSince(last) >= QWUpdateCheckSchedule.launchBurstCooldownSeconds
            }
        case .dailyTimer:
            guard schedule == .onLaunchAndDaily || schedule == .daily else { return false }
            guard let last = state.lastCheckedAt else { return true }
            return now.timeIntervalSince(last) >= QWUpdateCheckSchedule.dailyIntervalSeconds
        case .manual:
            return true
        }
    }

    public func check(
        currentVersion: String = QWVersion.current,
        stateURL: URL = QWPaths.updateStateFileURL,
        force: Bool = false,
        now: Date = Date()
    ) async throws -> (state: QWUpdateState, release: QWReleaseInfo?) {
        var state = QWUpdateStateStore.load(from: stateURL)
        guard Self.isAllowedUpdateURL(
            apiURL,
            extraAllowedHosts: extraAllowedHosts,
            allowLoopbackHTTP: allowLoopbackHTTP
        ) else {
            throw QWUpdateError.disallowedHost(apiURL.host ?? apiURL.absoluteString)
        }

        let userAgent = "Qwegn/\(currentVersion)"
        var request = URLRequest(url: apiURL, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 15)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        if !force, let etag = state.etag, !etag.isEmpty {
            request.setValue(etag, forHTTPHeaderField: "If-None-Match")
        }

        do {
            let (data, http) = try await transport.data(for: request)
            if http.statusCode == 304 {
                state.lastCheckedAt = now
                state.lastCheckSucceededAt = now
                state.lastError = nil
                try QWUpdateStateStore.save(state, to: stateURL)
                return (state, state.cachedRelease)
            }
            if http.statusCode == 403 || http.statusCode == 429 {
                let fallback = try await resolveViaRedirectFallback(userAgent: userAgent)
                state.lastCheckedAt = now
                state.lastCheckSucceededAt = now
                state.cachedRelease = fallback
                state.lastError = nil
                try QWUpdateStateStore.save(state, to: stateURL)
                return (state, fallback)
            }
            guard (200...299).contains(http.statusCode) else {
                throw QWUpdateError.networkFailure("HTTP \(http.statusCode)")
            }

            let parsed = try parseGitHubReleaseJSON(data)
            state.lastCheckedAt = now
            state.lastCheckSucceededAt = now
            state.etag = http.value(forHTTPHeaderField: "ETag") ?? state.etag
            state.cachedRelease = parsed
            state.lastError = nil
            try QWUpdateStateStore.save(state, to: stateURL)
            return (state, parsed)
        } catch {
            state.lastCheckedAt = now
            state.lastError = error.localizedDescription
            try? QWUpdateStateStore.save(state, to: stateURL)
            throw error
        }
    }

    public static func isUpdateAvailable(currentVersion: String, latest: QWReleaseInfo?) -> Bool {
        guard let latest, let current = QWSemVer(currentVersion) else { return false }
        guard !latest.version.isPrerelease else { return false }
        return latest.version > current
    }

    public static func buildSnapshot(
        currentVersion: String = QWVersion.current,
        state: QWUpdateState,
        installOrigin: QWInstallOrigin,
        schedule: QWUpdateCheckSchedule,
        phase: QWUpdatePhase,
        downloadedBytes: Int64? = nil,
        totalBytes: Int64? = nil,
        signatureSummary: String? = nil,
        errorMessage: String? = nil,
        canAutoCheck: Bool = !isDisabledByEnvironment()
    ) -> QWUpdateStatusSnapshot {
        let release = state.cachedRelease
        let available = isUpdateAvailable(currentVersion: currentVersion, latest: release)
        let isSkipped = available && (release?.version.description == state.skippedVersion)
        let verifiedMatch: Bool = {
            guard let release,
                  let ver = state.verifiedPkgVersion,
                  let path = state.verifiedPkgPath,
                  ver == release.version.description else {
                return false
            }
            return FileManager.default.fileExists(atPath: path)
        }()

        return QWUpdateStatusSnapshot(
            currentVersion: currentVersion,
            latestRelease: release,
            updateAvailable: available,
            isSkipped: isSkipped,
            installOrigin: installOrigin,
            schedule: schedule,
            phase: phase,
            downloadedBytes: downloadedBytes,
            totalBytes: totalBytes,
            verifiedSHA256: verifiedMatch ? state.verifiedPkgSHA256 : nil,
            downloadedPkgPath: verifiedMatch ? state.verifiedPkgPath : nil,
            signatureSummary: verifiedMatch ? signatureSummary : nil,
            lastCheckedAt: state.lastCheckedAt,
            errorMessage: errorMessage ?? state.lastError,
            canAutoCheck: canAutoCheck,
            canInstallPkg: installOrigin.canInstallPkg
        )
    }

    private func resolveViaRedirectFallback(userAgent: String) async throws -> QWReleaseInfo {
        guard let redirected = try await transport.resolveRedirectLocation(for: latestPageURL, userAgent: userAgent),
              Self.isAllowedUpdateURL(
                  redirected,
                  extraAllowedHosts: extraAllowedHosts,
                  allowLoopbackHTTP: allowLoopbackHTTP
              ) else {
            throw QWUpdateError.rateLimitedWithoutFallback
        }
        let comps = redirected.pathComponents
        guard let tagIdx = comps.lastIndex(of: "tag"),
              tagIdx + 1 < comps.count else {
            throw QWUpdateError.rateLimitedWithoutFallback
        }
        let tag = comps[tagIdx + 1]
        guard let version = QWSemVer(tag), !version.isPrerelease else {
            throw QWUpdateError.rateLimitedWithoutFallback
        }
        let tagPath = redirected.path
        let downloadBase: String
        if let tagRange = tagPath.range(of: "/releases/tag/"),
           var urlComps = URLComponents(url: redirected, resolvingAgainstBaseURL: false) {
            urlComps.path = String(tagPath[..<tagRange.lowerBound]) + "/releases/download/\(tag)"
            urlComps.query = nil
            urlComps.fragment = nil
            downloadBase = urlComps.url?.absoluteString ?? "https://github.com/\(Self.defaultRepo)/releases/download/\(tag)"
        } else {
            downloadBase = "https://github.com/\(Self.defaultRepo)/releases/download/\(tag)"
        }
        let pkgName = "Qwegn-\(version.description)-macos-arm64.pkg"
        let tarName = "qwegn-\(tag)-macos-arm64.tar.gz"
        let sumsName = "SHA256SUMS.txt"
        guard let pkgURL = URL(string: "\(downloadBase)/\(pkgName)"),
              let tarURL = URL(string: "\(downloadBase)/\(tarName)"),
              let sumsURL = URL(string: "\(downloadBase)/\(sumsName)") else {
            throw QWUpdateError.rateLimitedWithoutFallback
        }
        return QWReleaseInfo(
            version: version,
            tagName: tag,
            title: "Qwegn \(tag)",
            notes: "Release notes unavailable while GitHub API is rate-limited. Open the GitHub Release page for details.",
            publishedAt: nil,
            htmlURL: redirected,
            pkgAsset: QWReleaseAsset(name: pkgName, downloadURL: pkgURL),
            tarballAsset: QWReleaseAsset(name: tarName, downloadURL: tarURL),
            checksumsAsset: QWReleaseAsset(name: sumsName, downloadURL: sumsURL)
        )
    }

    private func parseGitHubReleaseJSON(_ data: Data) throws -> QWReleaseInfo? {
        let dto: GitHubReleaseDTO
        do {
            dto = try JSONDecoder().decode(GitHubReleaseDTO.self, from: data)
        } catch {
            throw QWUpdateError.invalidResponse(error.localizedDescription)
        }
        if dto.draft == true || dto.prerelease == true {
            return nil
        }
        guard let version = QWSemVer(dto.tagName), !version.isPrerelease else {
            return nil
        }
        guard let htmlURL = URL(string: dto.htmlURL),
              Self.isAllowedUpdateURL(
                  htmlURL,
                  extraAllowedHosts: extraAllowedHosts,
                  allowLoopbackHTTP: allowLoopbackHTTP
              ) else {
            throw QWUpdateError.disallowedHost(dto.htmlURL)
        }

        let assets = try dto.assets.compactMap { asset -> QWReleaseAsset? in
            guard let url = URL(string: asset.browserDownloadURL) else { return nil }
            guard Self.isAllowedUpdateURL(
                url,
                extraAllowedHosts: extraAllowedHosts,
                allowLoopbackHTTP: allowLoopbackHTTP
            ) else {
                throw QWUpdateError.disallowedHost(url.host ?? asset.browserDownloadURL)
            }
            return QWReleaseAsset(name: asset.name, downloadURL: url, sizeBytes: asset.size)
        }

        let pkgAsset = assets.first { $0.name.hasSuffix("-macos-arm64.pkg") }
            ?? assets.first { $0.name.hasSuffix(".pkg") }
        let tarAsset = assets.first { $0.name.hasSuffix("-macos-arm64.tar.gz") }
            ?? assets.first { $0.name.hasSuffix(".tar.gz") }
        let sumsAsset = assets.first { $0.name == "SHA256SUMS.txt" }

        let published = dto.publishedAt.flatMap { Self.parseISO8601($0) }
        let title = (dto.name?.trimmingCharacters(in: .whitespacesAndNewlines)).flatMap { $0.isEmpty ? nil : $0 }
            ?? "Qwegn \(dto.tagName)"
        return QWReleaseInfo(
            version: version,
            tagName: dto.tagName,
            title: title,
            notes: dto.body ?? "",
            publishedAt: published,
            htmlURL: htmlURL,
            pkgAsset: pkgAsset,
            tarballAsset: tarAsset,
            checksumsAsset: sumsAsset
        )
    }

    private static func parseISO8601(_ raw: String) -> Date? {
        let withFrac = ISO8601DateFormatter()
        withFrac.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let d = withFrac.date(from: raw) { return d }
        let plain = ISO8601DateFormatter()
        plain.formatOptions = [.withInternetDateTime]
        return plain.date(from: raw)
    }
}

private struct GitHubReleaseDTO: Decodable {
    let tagName: String
    let name: String?
    let body: String?
    let draft: Bool?
    let prerelease: Bool?
    let publishedAt: String?
    let htmlURL: String
    let assets: [GitHubAssetDTO]

    enum CodingKeys: String, CodingKey {
        case tagName = "tag_name"
        case name
        case body
        case draft
        case prerelease
        case publishedAt = "published_at"
        case htmlURL = "html_url"
        case assets
    }
}

private struct GitHubAssetDTO: Decodable {
    let name: String
    let browserDownloadURL: String
    let size: Int64?

    enum CodingKeys: String, CodingKey {
        case name
        case browserDownloadURL = "browser_download_url"
        case size
    }
}
