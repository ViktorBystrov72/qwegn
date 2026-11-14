// QWHttpSecuritySupport.swift — Loopback/CORS/auth guards and NWConnection cancellation hub.

import Foundation
import Network
import Synchronization

public enum QWHttpServerError: Error, LocalizedError, Equatable, Sendable {
    case unauthenticatedNonLoopbackBind(host: String)
    case invalidListenHost(String)

    public var errorDescription: String? {
        switch self {
        case .unauthenticatedNonLoopbackBind(let host):
            return "QWHttpServer: refusing unauthenticated bind on non-loopback host '\(host)'; configure --api-key / QWG_API_KEY first"
        case .invalidListenHost(let host):
            return "QWHttpServer: invalid listen host '\(host)'"
        }
    }
}

public enum QWHttpSecurity {
    public static func isLoopbackHost(_ host: String) -> Bool {
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if trimmed == "localhost" || trimmed == "::1" || trimmed == "[::1]" || trimmed == "127.0.0.1" {
            return true
        }
        return trimmed.hasPrefix("127.")
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

    var isTerminated: Bool {
        lock.lock()
        defer { lock.unlock() }
        return terminated
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
        lock.lock()
        let alreadyTerminated = terminated
        if !alreadyTerminated {
            handlers.append(handler)
        }
        lock.unlock()
        if alreadyTerminated {
            handler()
        }
    }

    func fire() {
        lock.lock()
        guard !terminated else {
            lock.unlock()
            return
        }
        terminated = true
        let pending = handlers
        handlers.removeAll()
        lock.unlock()
        for handler in pending {
            handler()
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
