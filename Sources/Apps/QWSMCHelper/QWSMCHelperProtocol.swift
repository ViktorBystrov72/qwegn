// QWSMCHelperProtocol.swift — Unix socket protocol, RPM safety clamping, and watchdog server.
// Runs inside the minimal root LaunchDaemon (qwg-smc-helper) so unprivileged qwg and
// Qwegn.app processes can request safe fan overrides without elevating the engine or UI.

import Darwin
import Foundation

/// Canonical paths, identifiers, and safety invariants for the SMC helper daemon.
public enum QWSMCHelperConstants {
    public static let socketPath = "/var/run/qwg-smc.sock"
    public static let daemonLabel = "ai.qwegn.smc-helper"
    public static let plistFileName = "ai.qwegn.smc-helper.plist"
    public static let systemPlistPath = "/Library/LaunchDaemons/ai.qwegn.smc-helper.plist"
    public static let installedBinaryPath = "/usr/local/libexec/qwg-smc-helper"
    public static let defaultWatchdogSeconds: TimeInterval = 15.0
    public static let minSafeFanRPM: Double = 1000.0
}

/// Error payload returned by a `QWSMCBackend` implementation.
public struct QWSMCBackendError: Error, Sendable, Equatable {
    public let message: String
    public init(_ message: String) { self.message = message }
}

/// Hardware abstraction backing `QWSMCHelperServer` for hermetic testing.
public protocol QWSMCBackend: Sendable {
    func fanCount() -> Int
    func fanBounds(index: Int) -> (minRPM: Double, maxRPM: Double)
    func setFanTarget(index: Int, rpm: Double) -> Result<Double, QWSMCBackendError>
    func restoreAuto(fanCount: Int) -> Result<Void, QWSMCBackendError>
}

/// Live IOKit AppleSMC backend used by the privileged daemon.
public final class QWAppleSMCBackend: QWSMCBackend, @unchecked Sendable {
    private let client = QWSMCClient()

    public init() {}

    public func fanCount() -> Int {
        client.fanCount()
    }

    public func fanBounds(index: Int) -> (minRPM: Double, maxRPM: Double) {
        client.fanBounds(index: index)
    }

    public func setFanTarget(index: Int, rpm: Double) -> Result<Double, QWSMCBackendError> {
        let modeRes = client.writeUInt8("F\(index)md", value: 1)
        let tgRes = client.writeFloat("F\(index)Tg", value: Float(rpm))
        guard modeRes == KERN_SUCCESS, tgRes == KERN_SUCCESS else {
            return .failure(QWSMCBackendError("smc_write_failed(\(modeRes),\(tgRes))"))
        }
        return .success(rpm)
    }

    public func restoreAuto(fanCount: Int) -> Result<Void, QWSMCBackendError> {
        let count = max(1, min(fanCount, 4))
        for idx in 0..<count {
            let res = client.writeUInt8("F\(idx)md", value: 0)
            if res != KERN_SUCCESS {
                return .failure(QWSMCBackendError("smc_auto_failed(\(idx),\(res))"))
            }
        }
        return .success(())
    }
}

/// Parsed response returned by `qwg-smc-helper` over `/var/run/qwg-smc.sock`.
public struct QWSMCHelperResponse: Sendable, Equatable {
    public let ok: Bool
    public let message: String

    public init(ok: Bool, message: String) {
        self.ok = ok
        self.message = message
    }
}

/// Unix domain socket client used by `QWFanController` and `qwg fan-helper`.
public enum QWSMCHelperClient {
    public static func send(
        _ command: String,
        socketPath: String = QWSMCHelperConstants.socketPath,
        timeoutSeconds: Int = 1
    ) -> QWSMCHelperResponse? {
        let fd = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { return nil }
        defer { Darwin.close(fd) }

        var noSigPipe: Int32 = 1
        _ = Darwin.setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &noSigPipe, socklen_t(MemoryLayout<Int32>.size))
        var tv = timeval(tv_sec: max(1, timeoutSeconds), tv_usec: 0)
        _ = Darwin.setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        _ = Darwin.setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = Array(socketPath.utf8CString)
        let maxLen = MemoryLayout.size(ofValue: addr.sun_path)
        guard pathBytes.count <= maxLen else { return nil }
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: maxLen) { dst in
                for (i, ch) in pathBytes.enumerated() { dst[i] = ch }
            }
        }
        let len = socklen_t(MemoryLayout<sockaddr_un>.size)
        let connRes = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.connect(fd, $0, len) }
        }
        guard connRes == 0 else { return nil }

        let line = command.hasSuffix("\n") ? command : command + "\n"
        let payload = Array(line.utf8)
        var sent = 0
        while sent < payload.count {
            let written = payload.withUnsafeBytes { (buf: UnsafeRawBufferPointer) -> Int in
                guard let base = buf.baseAddress else { return -1 }
                return Darwin.write(fd, base.advanced(by: sent), payload.count - sent)
            }
            if written <= 0 { return nil }
            sent += written
        }

        var buffer = [UInt8](repeating: 0, count: 256)
        var received = Data()
        while received.count < 256 {
            let n = buffer.withUnsafeMutableBytes { (buf: UnsafeMutableRawBufferPointer) -> Int in
                guard let base = buf.baseAddress else { return -1 }
                return Darwin.read(fd, base, buf.count)
            }
            if n <= 0 { break }
            received.append(contentsOf: buffer[0..<n])
            if buffer[0..<n].contains(0x0A) { break }
        }
        guard let rawReply = String(data: received, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !rawReply.isEmpty else {
            return nil
        }
        if rawReply == "OK" || rawReply.hasPrefix("OK ") {
            let msg = rawReply == "OK" ? "" : String(rawReply.dropFirst(3))
            return QWSMCHelperResponse(ok: true, message: msg)
        }
        if rawReply == "ERR" || rawReply.hasPrefix("ERR ") {
            let msg = rawReply == "ERR" ? "error" : String(rawReply.dropFirst(4))
            return QWSMCHelperResponse(ok: false, message: msg)
        }
        return QWSMCHelperResponse(ok: false, message: rawReply)
    }

    public static func ping(socketPath: String = QWSMCHelperConstants.socketPath) -> Bool {
        send("PING\n", socketPath: socketPath)?.ok == true
    }
}

/// Privileged Unix domain socket server with RPM clamping, peer auth, and dead-man auto-restore.
public final class QWSMCHelperServer: @unchecked Sendable {
    public enum ServerError: Error, LocalizedError {
        case pathTooLong(String)
        case nonSocketExists(String)
        case socketFailed(String, Int32)

        public var errorDescription: String? {
            switch self {
            case .pathTooLong(let p): return "Socket path exceeds sun_path capacity: \(p)"
            case .nonSocketExists(let p): return "Refusing to overwrite non-socket file at \(p)"
            case .socketFailed(let op, let code): return "\(op) failed: \(String(cString: strerror(code)))"
            }
        }
    }

    public let socketPath: String
    public let watchdogTimeout: TimeInterval
    private let backend: any QWSMCBackend
    private let allowedUIDs: Set<uid_t>?
    private let stateLock = NSLock()
    private var manualOverrideActive = false
    private var lastSetUptime: TimeInterval = 0
    private var listenFD: Int32 = -1
    private var stopRequested = false

    public init(
        socketPath: String = QWSMCHelperConstants.socketPath,
        watchdogTimeout: TimeInterval = QWSMCHelperConstants.defaultWatchdogSeconds,
        backend: any QWSMCBackend = QWAppleSMCBackend(),
        allowedUIDs: Set<uid_t>? = nil
    ) {
        self.socketPath = socketPath
        self.watchdogTimeout = watchdogTimeout
        self.backend = backend
        self.allowedUIDs = allowedUIDs
    }

    deinit {
        stop()
    }

    /// Evaluates one protocol line (`PING`, `AUTO`, `SET <fanIdx> <rpm>`) and returns `OK`/`ERR`.
    public func handleCommand(
        _ rawLine: String,
        now: TimeInterval = ProcessInfo.processInfo.systemUptime
    ) -> String {
        let trimmed = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "ERR empty_command\n" }
        let parts = trimmed.split(whereSeparator: { $0 == " " || $0 == "\t" }).map(String.init)
        guard let verb = parts.first else { return "ERR empty_command\n" }

        return stateLock.withLock {
            switch verb {
            case "PING":
                return "OK PONG\n"
            case "AUTO":
                let count = max(1, backend.fanCount())
                switch backend.restoreAuto(fanCount: count) {
                case .success:
                    manualOverrideActive = false
                    lastSetUptime = 0
                    return "OK AUTO\n"
                case .failure(let err):
                    return "ERR \(err.message)\n"
                }
            case "SET":
                guard parts.count == 3,
                      let fanIndex = Int(parts[1]),
                      let requestedRPM = Double(parts[2]),
                      requestedRPM.isFinite,
                      requestedRPM > 0 else {
                    return "ERR invalid_set_syntax\n"
                }
                let count = min(max(backend.fanCount(), 0), 4)
                guard fanIndex >= 0 && fanIndex < count else {
                    return "ERR invalid_fan_index\n"
                }
                let bounds = backend.fanBounds(index: fanIndex)
                let floorRPM = max(bounds.minRPM, QWSMCHelperConstants.minSafeFanRPM)
                let ceilingRPM = max(floorRPM, bounds.maxRPM)
                let clamped = min(ceilingRPM, max(floorRPM, requestedRPM.rounded()))
                switch backend.setFanTarget(index: fanIndex, rpm: clamped) {
                case .success(let applied):
                    manualOverrideActive = true
                    lastSetUptime = now
                    return "OK SET \(fanIndex) \(Int(applied.rounded()))\n"
                case .failure(let err):
                    return "ERR \(err.message)\n"
                }
            default:
                return "ERR unknown_command\n"
            }
        }
    }

    /// Restores macOS System Auto if manual override has not been refreshed within `watchdogTimeout`.
    @discardableResult
    public func checkWatchdog(now: TimeInterval = ProcessInfo.processInfo.systemUptime) -> Bool {
        stateLock.withLock {
            guard manualOverrideActive, (now - lastSetUptime) >= watchdogTimeout else {
                return false
            }
            let count = max(1, backend.fanCount())
            if case .success = backend.restoreAuto(fanCount: count) {
                manualOverrideActive = false
                lastSetUptime = 0
                return true
            }
            return false
        }
    }

    /// Verifies Unix socket peer UID via `LOCAL_PEERCRED`.
    public static func isPeerAuthorized(fd: Int32, allowedUIDs: Set<uid_t>? = nil) -> Bool {
        var cred = xucred()
        var len = socklen_t(MemoryLayout<xucred>.size)
        let rc = withUnsafeMutablePointer(to: &cred) { ptr in
            Darwin.getsockopt(fd, SOL_LOCAL, LOCAL_PEERCRED, ptr, &len)
        }
        guard rc == 0, cred.cr_version == XUCRED_VERSION else { return false }
        if let allowedUIDs {
            return cred.cr_uid == 0 || allowedUIDs.contains(cred.cr_uid)
        }
        return cred.cr_uid == 0 || cred.cr_uid == Darwin.geteuid() || (cred.cr_uid >= 500 && cred.cr_uid < 65534)
    }

    /// Binds and starts listening on `socketPath`; safe to call before spawning `serve`.
    @discardableResult
    public func startListening() throws -> Int32 {
        if let existing = stateLock.withLock({ listenFD >= 0 ? listenFD : nil }) {
            return existing
        }
        var st = stat()
        if Darwin.lstat(socketPath, &st) == 0 {
            if (st.st_mode & S_IFMT) == S_IFSOCK {
                Darwin.unlink(socketPath)
            } else {
                throw ServerError.nonSocketExists(socketPath)
            }
        }
        let fd = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw ServerError.socketFailed("socket", errno) }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = Array(socketPath.utf8CString)
        let maxLen = MemoryLayout.size(ofValue: addr.sun_path)
        guard pathBytes.count <= maxLen else {
            Darwin.close(fd)
            throw ServerError.pathTooLong(socketPath)
        }
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: maxLen) { dst in
                for (i, ch) in pathBytes.enumerated() { dst[i] = ch }
            }
        }
        let len = socklen_t(MemoryLayout<sockaddr_un>.size)
        let bindRes = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.bind(fd, $0, len) }
        }
        guard bindRes == 0 else {
            let err = errno
            Darwin.close(fd)
            throw ServerError.socketFailed("bind", err)
        }
        guard Darwin.listen(fd, 16) == 0 else {
            let err = errno
            Darwin.close(fd)
            Darwin.unlink(socketPath)
            throw ServerError.socketFailed("listen", err)
        }
        _ = Darwin.chmod(socketPath, 0o660)
        if Darwin.geteuid() == 0 {
            _ = Darwin.chown(socketPath, 0, 20)
        }
        stateLock.withLock {
            listenFD = fd
            stopRequested = false
        }
        return fd
    }

    public func stop() {
        stateLock.withLock {
            stopRequested = true
        }
    }

    /// Runs the accept/watchdog loop until `stop()` or `stopCheck()` returns true.
    public func run(stopCheck: () -> Bool = { false }) throws {
        _ = Darwin.signal(SIGPIPE, SIG_IGN)
        let fd = try startListening()
        defer { cleanupSocket(fd: fd) }

        while !stopCheck() && !stateLock.withLock({ stopRequested }) {
            var pfd = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
            let ready = Darwin.poll(&pfd, 1, 200)
            if ready > 0 && (pfd.revents & Int16(POLLIN)) != 0 {
                let clientFD = Darwin.accept(fd, nil, nil)
                if clientFD >= 0 {
                    handleClientConnection(clientFD)
                }
            }
            _ = checkWatchdog()
        }
        stateLock.withLock {
            if manualOverrideActive {
                let count = max(1, backend.fanCount())
                _ = backend.restoreAuto(fanCount: count)
                manualOverrideActive = false
                lastSetUptime = 0
            }
        }
    }

    private func handleClientConnection(_ clientFD: Int32) {
        defer { Darwin.close(clientFD) }
        var noSigPipe: Int32 = 1
        _ = Darwin.setsockopt(clientFD, SOL_SOCKET, SO_NOSIGPIPE, &noSigPipe, socklen_t(MemoryLayout<Int32>.size))
        var tv = timeval(tv_sec: 2, tv_usec: 0)
        _ = Darwin.setsockopt(clientFD, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        _ = Darwin.setsockopt(clientFD, SOL_SOCKET, SO_SNDTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))

        guard Self.isPeerAuthorized(fd: clientFD, allowedUIDs: allowedUIDs) else {
            writeReply("ERR unauthorized\n", to: clientFD)
            return
        }
        var buf = [UInt8](repeating: 0, count: 256)
        var req = Data()
        while req.count < 256 {
            let n = buf.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) -> Int in
                guard let base = raw.baseAddress else { return -1 }
                return Darwin.read(clientFD, base, raw.count)
            }
            if n <= 0 { break }
            req.append(contentsOf: buf[0..<n])
            if buf[0..<n].contains(0x0A) { break }
        }
        let line = String(data: req, encoding: .utf8) ?? ""
        let reply = handleCommand(line)
        writeReply(reply, to: clientFD)
    }

    private func writeReply(_ reply: String, to fd: Int32) {
        let bytes = Array(reply.utf8)
        var sent = 0
        while sent < bytes.count {
            let w = bytes.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Int in
                guard let base = raw.baseAddress else { return -1 }
                return Darwin.write(fd, base.advanced(by: sent), bytes.count - sent)
            }
            if w <= 0 { break }
            sent += w
        }
    }

    private func cleanupSocket(fd: Int32) {
        Darwin.close(fd)
        stateLock.withLock {
            if listenFD == fd { listenFD = -1 }
        }
        var st = stat()
        if Darwin.lstat(socketPath, &st) == 0 && (st.st_mode & S_IFMT) == S_IFSOCK {
            Darwin.unlink(socketPath)
        }
    }
}
