// QWHttpSecuritySupport.swift — Loopback/CORS/CSRF/path guards and cancellation hub.

import Darwin
import Foundation
import Network
import QWCore
import Synchronization

public enum QWHttpServerError: Error, LocalizedError, Equatable, Sendable {
    case unauthenticatedNonLoopbackBind(host: String)
    case invalidListenHost(String)
    case invalidLogDirectory(String)

    public var errorDescription: String? {
        switch self {
        case .unauthenticatedNonLoopbackBind(let host):
            return "QWHttpServer: refusing unauthenticated bind on non-loopback host '\(host)'; configure --api-key / QWG_API_KEY first"
        case .invalidListenHost(let host):
            return "QWHttpServer: invalid listen host '\(host)'"
        case .invalidLogDirectory(let reason):
            return "QWHttpServer: invalid log directory: \(reason)"
        }
    }
}

public struct QWMemoryWorkspaceAccessError: Error, LocalizedError, Equatable, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

final class QWWorkspacePolicyRegistry: @unchecked Sendable {
    static let shared = QWWorkspacePolicyRegistry()
    private let lock = NSLock()
    private var entries: [ObjectIdentifier: Entry] = [:]

    private struct Entry {
        weak var actor: AnyObject?
        var roots: [String]
    }

    func setAllowedRoots(_ roots: [String], for actor: AnyObject) {
        let id = ObjectIdentifier(actor)
        lock.withLock {
            entries = entries.filter { $0.value.actor != nil }
            if roots.isEmpty {
                entries.removeValue(forKey: id)
            } else {
                entries[id] = Entry(actor: actor, roots: roots)
            }
        }
    }

    func allowedRoots(for actor: AnyObject) -> [String] {
        let id = ObjectIdentifier(actor)
        return lock.withLock {
            entries = entries.filter { $0.value.actor != nil }
            return entries[id]?.roots ?? []
        }
    }
}

public enum QWHttpSecurity {
    private static let blockedSystemLogPrefixes = [
        "/etc", "/private/etc", "/System", "/Library",
        "/Applications", "/usr", "/bin", "/sbin", "/private/var/root",
    ]

    public static func isLoopbackHost(_ host: String) -> Bool {
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if trimmed == "localhost" || trimmed == "::1" || trimmed == "[::1]" {
            return true
        }
        var addr = in_addr()
        if inet_pton(AF_INET, trimmed, &addr) == 1 {
            return (UInt32(bigEndian: addr.s_addr) >> 24) == 127
        }
        return false
    }

    public static func isJSONContentType(_ headers: [String: String]) -> Bool {
        guard let raw = headers["content-type"] else { return false }
        let mediaType = raw.split(separator: ";", maxSplits: 1).first?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() ?? ""
        return mediaType == "application/json"
    }

    public static func isMultipartFormDataContentType(_ headers: [String: String]) -> Bool {
        guard let raw = headers["content-type"] else { return false }
        let parts = raw.split(separator: ";").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard let media = parts.first?.lowercased(), media == "multipart/form-data" else {
            return false
        }
        return parts.dropFirst().contains { param in
            let low = param.lowercased()
            guard low.hasPrefix("boundary=") else { return false }
            let val = param.dropFirst("boundary=".count).trimmingCharacters(in: CharacterSet(charactersIn: "\" "))
            return !val.isEmpty
        }
    }

    public static func isValidMutationOrigin(
        headers: [String: String],
        allowlist: [String],
        requireJSONContentType: Bool = false,
        allowMultipartContentType: Bool = false
    ) -> Bool {
        if let rawOrigin = headers["origin"] {
            guard resolvedAllowedOrigin(rawOrigin, allowlist: allowlist) != nil else {
                return false
            }
        } else if let secFetchSite = headers["sec-fetch-site"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased(),
                  secFetchSite == "cross-site" {
            return false
        }
        if allowMultipartContentType && isMultipartFormDataContentType(headers) {
            return true
        }
        if requireJSONContentType {
            return isJSONContentType(headers)
        }
        if headers["content-type"] != nil {
            return isJSONContentType(headers)
        }
        return true
    }

    public static func isInsideAppBundle(_ url: URL) -> Bool {
        url.pathComponents.contains { $0.lowercased().hasSuffix(".app") }
    }

    public static func canonicalDirectoryContainment(_ candidate: URL, inside root: URL) -> Bool {
        let candPath = candidate.resolvingSymlinksInPath().standardizedFileURL.path
        let rootPath = root.resolvingSymlinksInPath().standardizedFileURL.path
        let prefix = rootPath.hasSuffix("/") ? rootPath : (rootPath + "/")
        return candPath == rootPath || candPath.hasPrefix(prefix)
    }

    public static func isUserOwnedDirectory(_ url: URL) -> Bool {
        var st = stat()
        guard stat(url.path, &st) == 0 else { return false }
        let isDir = (st.st_mode & S_IFMT) == S_IFDIR
        return isDir && st.st_uid == getuid()
    }

    public static func validateSafeLogFileName(_ rawName: String) throws {
        let trimmed = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              trimmed != ".",
              trimmed != "..",
              !trimmed.contains("/"),
              !trimmed.contains("\\"),
              !trimmed.contains("\0") else {
            throw QWHttpServerError.invalidLogDirectory("logFileName must be a valid relative filename")
        }
    }

    public static func validateSafeLogDirectory(
        _ rawDir: String,
        createIfInQwgRoot: Bool = false,
        fileManager: FileManager = .default
    ) throws -> URL {
        let trimmed = rawDir.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !trimmed.contains("\0") else {
            throw QWHttpServerError.invalidLogDirectory("logDir must not be empty")
        }
        let expanded = trimmed.expandingTildePath
        let rawURL = URL(fileURLWithPath: expanded, isDirectory: true)
        guard !isInsideAppBundle(rawURL) else {
            throw QWHttpServerError.invalidLogDirectory("logDir must not reside inside an application bundle")
        }
        let resolvedURL = rawURL.resolvingSymlinksInPath().standardizedFileURL
        guard resolvedURL.path != "/", !isInsideAppBundle(resolvedURL) else {
            throw QWHttpServerError.invalidLogDirectory("logDir '\(resolvedURL.path)' is not permitted")
        }
        for blocked in blockedSystemLogPrefixes {
            if resolvedURL.path == blocked || resolvedURL.path.hasPrefix(blocked + "/") {
                throw QWHttpServerError.invalidLogDirectory("logDir '\(resolvedURL.path)' is inside a protected system path")
            }
        }

        let qwgRoot = QWPaths.rootDirectoryURL.resolvingSymlinksInPath().standardizedFileURL
        let insideQwgRoot = canonicalDirectoryContainment(resolvedURL, inside: qwgRoot)

        var isDir: ObjCBool = false
        if fileManager.fileExists(atPath: resolvedURL.path, isDirectory: &isDir) {
            guard isDir.boolValue else {
                throw QWHttpServerError.invalidLogDirectory("logDir '\(resolvedURL.path)' exists and is not a directory")
            }
            guard isUserOwnedDirectory(resolvedURL) else {
                throw QWHttpServerError.invalidLogDirectory("logDir '\(resolvedURL.path)' must be owned by the current user")
            }
            return resolvedURL
        }

        guard insideQwgRoot else {
            throw QWHttpServerError.invalidLogDirectory(
                "logDir '\(resolvedURL.path)' must resolve inside ~/.qwg or an existing user-owned directory"
            )
        }

        var probe = resolvedURL
        while probe.path != "/" {
            var parentIsDir: ObjCBool = false
            if fileManager.fileExists(atPath: probe.path, isDirectory: &parentIsDir) {
                guard parentIsDir.boolValue, isUserOwnedDirectory(probe) else {
                    throw QWHttpServerError.invalidLogDirectory("ancestor '\(probe.path)' is not a user-owned directory")
                }
                break
            }
            probe = probe.deletingLastPathComponent()
        }

        if createIfInQwgRoot {
            try fileManager.createDirectory(
                at: resolvedURL,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
            let postURL = resolvedURL.resolvingSymlinksInPath().standardizedFileURL
            var postIsDir: ObjCBool = false
            guard canonicalDirectoryContainment(postURL, inside: qwgRoot),
                  fileManager.fileExists(atPath: postURL.path, isDirectory: &postIsDir),
                  postIsDir.boolValue,
                  isUserOwnedDirectory(postURL) else {
                throw QWHttpServerError.invalidLogDirectory("created logDir failed post-creation safety checks")
            }
            return postURL
        }
        return resolvedURL
    }

    public static func resolvedAllowedWorkspaceRootURLs(
        configuredRoots: [String],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> [URL] {
        var candidates: [String] = [
            QWPaths.rootDirectoryURL.appendingPathComponent("workspaces", isDirectory: true).path
        ]
        candidates.append(contentsOf: configuredRoots)
        if let envRaw = environment["QWG_ALLOWED_WORKSPACE_ROOTS"], !envRaw.isEmpty {
            let parts = envRaw.split(separator: ":").map {
                String($0).trimmingCharacters(in: .whitespacesAndNewlines)
            }
            candidates.append(contentsOf: parts)
        }
        var seen = Set<String>()
        var out: [URL] = []
        for raw in candidates {
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, !trimmed.contains("\0") else { continue }
            let url = URL(fileURLWithPath: trimmed.expandingTildePath, isDirectory: true)
                .resolvingSymlinksInPath().standardizedFileURL
            guard url.path != "/", !isInsideAppBundle(url) else { continue }
            if seen.insert(url.path).inserted {
                out.append(url)
            }
        }
        return out
    }

    public static func validateAllowedWorkspacePath(
        _ rawPath: String,
        configuredRoots: [String],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> URL {
        let trimmed = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !trimmed.contains("\0") else {
            throw QWMemoryWorkspaceAccessError("Workspace path must not be empty")
        }
        let candidate = URL(fileURLWithPath: trimmed.expandingTildePath, isDirectory: true)
            .resolvingSymlinksInPath().standardizedFileURL
        guard candidate.path != "/", !isInsideAppBundle(candidate) else {
            throw QWMemoryWorkspaceAccessError("Workspace path '\(candidate.path)' is not permitted")
        }
        let allowedRoots = resolvedAllowedWorkspaceRootURLs(configuredRoots: configuredRoots, environment: environment)
        guard allowedRoots.contains(where: { canonicalDirectoryContainment(candidate, inside: $0) }) else {
            throw QWMemoryWorkspaceAccessError(
                "Workspace path '\(candidate.path)' is outside allowed workspace roots; configure QWG_ALLOWED_WORKSPACE_ROOTS or ~/.qwg/workspaces"
            )
        }
        return candidate
    }

    public static func constantTimeEquals(_ lhs: String, _ rhs: String) -> Bool {
        let a = Array(lhs.utf8)
        let b = Array(rhs.utf8)
        var diff: UInt8 = a.count == b.count ? 0 : 1
        let n = max(a.count, b.count)
        for i in 0..<n {
            let ca = i < a.count ? a[i] : 0
            let cb = i < b.count ? b[i] : 0
            diff |= (ca ^ cb)
        }
        return diff == 0 && a.count == b.count
    }

    public static func extractPresentedToken(headers: [String: String]) -> String? {
        if let auth = headers["authorization"] {
            let trimmed = auth.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.lowercased().hasPrefix("bearer ") {
                let token = String(trimmed.dropFirst(7)).trimmingCharacters(in: .whitespacesAndNewlines)
                if !token.isEmpty { return token }
            }
        }
        if let key = headers["x-api-key"]?.trimmingCharacters(in: .whitespacesAndNewlines), !key.isEmpty {
            return key
        }
        return nil
    }

    public static func resolvedAllowedOrigin(_ rawOrigin: String?, allowlist: [String]) -> String? {
        guard let rawOrigin else { return nil }
        let origin = rawOrigin.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !origin.isEmpty else { return nil }
        if allowlist.contains(where: { $0.trimmingCharacters(in: .whitespacesAndNewlines) == origin }) {
            return origin
        }
        guard let comps = URLComponents(string: origin),
              let scheme = comps.scheme?.lowercased(),
              (scheme == "http" || scheme == "https"),
              let host = comps.host?.lowercased(),
              comps.user == nil, comps.password == nil,
              (comps.path.isEmpty || comps.path == "/"),
              comps.query == nil, comps.fragment == nil,
              origin.hasSuffix("/") == false else {
            return nil
        }
        return isLoopbackHost(host) ? origin : nil
    }

    public static func escapeHTML(_ text: String) -> String {
        var out = ""
        out.reserveCapacity(text.count)
        for ch in text {
            switch ch {
            case "&": out.append("&amp;")
            case "<": out.append("&lt;")
            case ">": out.append("&gt;")
            case "\"": out.append("&quot;")
            case "'": out.append("&#39;")
            default: out.append(ch)
            }
        }
        return out
    }
}

/// Multiplexes `NWConnection.stateUpdateHandler` notifications onto per-connection teardown callbacks.
final class QWConnectionCancellationHub: @unchecked Sendable {
    private let lock = NSLock()
    private var terminated = false
    private var handlers: [@Sendable () -> Void] = []
    let cancelFlag = QWGenerationCancelFlag()

    var isTerminated: Bool {
        cancelFlag.isCancelled
    }

    func attach(to connection: NWConnection) {
        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .cancelled, .failed:
                self?.fire()
            default:
                break
            }
        }
    }

    func register(_ handler: @escaping @Sendable () -> Void) {
        let alreadyTerminated = lock.withLock { () -> Bool in
            if terminated { return true }
            handlers.append(handler)
            return false
        }
        if alreadyTerminated {
            handler()
        }
    }

    func fire() {
        cancelFlag.cancel()
        let pending = lock.withLock { () -> [@Sendable () -> Void]? in
            guard !terminated else { return nil }
            terminated = true
            let list = handlers
            handlers.removeAll()
            return list
        }
        guard let pending else { return }
        for handler in pending {
            handler()
        }
    }
}

/// Bounded-backpressure `.contentProcessed` SSE/NDJSON frame writer reusing a single encoder and scratch buffer.
final class QWSSEWriter: @unchecked Sendable {
    private static let ssePrefix = Data("data: ".utf8)
    private static let sseSuffix = Data("\n\n".utf8)
    private let lock = NSLock()
    private let connection: NWConnection
    private let cancelFlag: QWGenerationCancelFlag?
    private let encoder: JSONEncoder
    private let onError: (@Sendable () -> Void)?
    private let maxInFlightFrames: Int
    private let maxInFlightBytes: Int
    private let maxPendingBytes: Int
    private var scratch = Data()
    private var inFlightFrames = 0
    private var inFlightBytes = 0
    private var pendingBytes = 0
    private var pendingQueue: [(data: Data, isTerminal: Bool)] = []
    private var closed = false
    private var finishQueued = false

    init(
        connection: NWConnection,
        cancelFlag: QWGenerationCancelFlag? = nil,
        dateEncodingStrategy: JSONEncoder.DateEncodingStrategy? = nil,
        maxInFlightFrames: Int = 32,
        maxInFlightBytes: Int = 512 * 1024,
        maxPendingBytes: Int = 8 * 1024 * 1024,
        onError: (@Sendable () -> Void)? = nil
    ) {
        self.connection = connection
        self.cancelFlag = cancelFlag
        let enc = JSONEncoder()
        if let dateEncodingStrategy {
            enc.dateEncodingStrategy = dateEncodingStrategy
        }
        self.encoder = enc
        self.maxInFlightFrames = max(1, maxInFlightFrames)
        self.maxInFlightBytes = max(4096, maxInFlightBytes)
        self.maxPendingBytes = max(self.maxInFlightBytes, maxPendingBytes)
        self.onError = onError
        self.scratch.reserveCapacity(2048)
    }

    var isClosed: Bool {
        lock.withLock { closed }
    }

    func sendSSEDataFrame<T: Encodable>(_ object: T) {
        let payload: Data? = lock.withLock {
            guard !closed, !finishQueued, !(cancelFlag?.isCancelled ?? false) else { return nil }
            guard let json = try? encoder.encode(object) else { return nil }
            scratch.removeAll(keepingCapacity: true)
            scratch.append(Self.ssePrefix)
            scratch.append(json)
            scratch.append(Self.sseSuffix)
            return Data(scratch)
        }
        if let payload { enqueueFrame(payload, isTerminal: false) }
    }

    func sendSSENamedEvent<T: Encodable>(event: String, object: T) {
        let payload: Data? = lock.withLock {
            guard !closed, !finishQueued, !(cancelFlag?.isCancelled ?? false) else { return nil }
            guard let json = try? encoder.encode(object) else { return nil }
            scratch.removeAll(keepingCapacity: true)
            scratch.append(contentsOf: "event: ".utf8)
            scratch.append(contentsOf: event.utf8)
            scratch.append(contentsOf: "\ndata: ".utf8)
            scratch.append(json)
            scratch.append(Self.sseSuffix)
            return Data(scratch)
        }
        if let payload { enqueueFrame(payload, isTerminal: false) }
    }

    func sendSSEJsonObject(event: String? = nil, fields: [String: Any]) {
        guard let json = try? JSONSerialization.data(withJSONObject: fields) else { return }
        let payload: Data? = lock.withLock {
            guard !closed, !finishQueued, !(cancelFlag?.isCancelled ?? false) else { return nil }
            scratch.removeAll(keepingCapacity: true)
            if let event {
                scratch.append(contentsOf: "event: ".utf8)
                scratch.append(contentsOf: event.utf8)
                scratch.append(contentsOf: "\n".utf8)
            }
            scratch.append(Self.ssePrefix)
            scratch.append(json)
            scratch.append(Self.sseSuffix)
            return Data(scratch)
        }
        if let payload { enqueueFrame(payload, isTerminal: false) }
    }

    func sendNDJSONLine<T: Encodable>(_ object: T) {
        let payload: Data? = lock.withLock {
            guard !closed, !finishQueued, !(cancelFlag?.isCancelled ?? false) else { return nil }
            guard let json = try? encoder.encode(object) else { return nil }
            scratch.removeAll(keepingCapacity: true)
            scratch.append(json)
            scratch.append(0x0A)
            return Data(scratch)
        }
        if let payload { enqueueFrame(payload, isTerminal: false) }
    }

    func sendRaw(_ data: Data) {
        enqueueFrame(data, isTerminal: false)
    }

    func finish(with finalData: Data? = nil) {
        let body = finalData ?? Data()
        enqueueFrame(body, isTerminal: true)
    }

    private func enqueueFrame(_ data: Data, isTerminal: Bool) {
        enum Action {
            case skip
            case sendNow(Data, Bool)
            case abort
        }
        let action = lock.withLock { () -> Action in
            if closed || finishQueued { return .skip }
            if !isTerminal && (cancelFlag?.isCancelled ?? false) {
                closed = true
                pendingQueue.removeAll(keepingCapacity: false)
                pendingBytes = 0
                return .abort
            }
            if isTerminal { finishQueued = true }
            if !isTerminal && pendingQueue.isEmpty
                && inFlightFrames < maxInFlightFrames
                && (inFlightBytes + data.count) <= maxInFlightBytes {
                inFlightFrames += 1
                inFlightBytes += data.count
                return .sendNow(data, false)
            }
            if !isTerminal && pendingBytes + data.count > maxPendingBytes {
                closed = true
                pendingQueue.removeAll(keepingCapacity: false)
                pendingBytes = 0
                return .abort
            }
            if pendingQueue.isEmpty && inFlightFrames == 0 && isTerminal {
                closed = true
                inFlightFrames += 1
                inFlightBytes += data.count
                return .sendNow(data, true)
            }
            pendingQueue.append((data, isTerminal))
            pendingBytes += data.count
            return .skip
        }
        switch action {
        case .skip:
            break
        case .sendNow(let frame, let terminal):
            dispatchSend(frame, isTerminal: terminal)
        case .abort:
            cancelFlag?.cancel()
            connection.cancel()
            onError?()
        }
    }

    private func dispatchSend(_ data: Data, isTerminal: Bool) {
        let byteLen = data.count
        connection.send(content: data, isComplete: isTerminal, completion: .contentProcessed { error in
            if error != nil {
                self.handleCompletedSend(bytes: byteLen, isTerminal: isTerminal, failed: true)
            } else {
                self.handleCompletedSend(bytes: byteLen, isTerminal: isTerminal, failed: false)
            }
        })
    }

    private func handleCompletedSend(bytes: Int, isTerminal: Bool, failed: Bool) {
        enum Next {
            case abort
            case drain([(data: Data, isTerminal: Bool)], closeAfter: Bool)
        }
        let next = lock.withLock { () -> Next in
            inFlightFrames = max(0, inFlightFrames - 1)
            inFlightBytes = max(0, inFlightBytes - bytes)
            if failed {
                closed = true
                pendingQueue.removeAll(keepingCapacity: false)
                pendingBytes = 0
                return .abort
            }
            if isTerminal {
                closed = true
                pendingQueue.removeAll(keepingCapacity: false)
                pendingBytes = 0
                return .drain([], closeAfter: true)
            }
            var toSend: [(data: Data, isTerminal: Bool)] = []
            while let head = pendingQueue.first {
                let fitsWindow = inFlightFrames < maxInFlightFrames && (inFlightFrames == 0 || inFlightBytes + head.data.count <= maxInFlightBytes)
                guard fitsWindow else { break }
                pendingQueue.removeFirst()
                pendingBytes = max(0, pendingBytes - head.data.count)
                inFlightFrames += 1
                inFlightBytes += head.data.count
                if head.isTerminal { closed = true }
                toSend.append(head)
                if head.isTerminal { break }
            }
            return .drain(toSend, closeAfter: false)
        }
        switch next {
        case .abort:
            cancelFlag?.cancel()
            connection.cancel()
            onError?()
        case .drain(let frames, let closeAfter):
            for frame in frames {
                dispatchSend(frame.data, isTerminal: frame.isTerminal)
            }
            if closeAfter {
                connection.cancel()
            }
        }
    }
}

/// Lock-free atomic cancellation flag checked on every generated token.
public final class QWGenerationCancelFlag: @unchecked Sendable {
    private let flag = Atomic<Bool>(false)
    public init() {}
    public var isCancelled: Bool { flag.load(ordering: .acquiring) }
    public func cancel() { flag.store(true, ordering: .releasing) }
}

final class QWRequestReadTimeout: @unchecked Sendable {
    private let lock = NSLock()
    private var timer: DispatchSourceTimer?
    private var cancelled = false

    init(seconds: TimeInterval, queue: DispatchQueue, onTimeout: @escaping @Sendable () -> Void) {
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + seconds)
        timer.setEventHandler { [weak self] in
            let shouldFire = self?.lock.withLock { () -> Bool in
                guard let self, !self.cancelled else { return false }
                self.cancelled = true
                self.timer = nil
                return true
            } ?? false
            timer.cancel()
            if shouldFire {
                onTimeout()
            }
        }
        self.timer = timer
        timer.resume()
    }

    func cancel() {
        let pending = lock.withLock { () -> DispatchSourceTimer? in
            guard !cancelled else { return nil }
            cancelled = true
            let current = timer
            timer = nil
            return current
        }
        pending?.cancel()
    }
}
