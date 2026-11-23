import Foundation

/// Centralized path expansion, canonicalization, and `~/.qwg` workspace layout.
public enum QWPaths {
    public static var rootDirectoryURL: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".qwg", isDirectory: true)
    }

    public static var settingsFileURL: URL {
        rootDirectoryURL.appendingPathComponent("settings.json", isDirectory: false)
    }

    public static var modelsDirectoryURL: URL {
        rootDirectoryURL.appendingPathComponent("models", isDirectory: true)
    }

    public static var modelsDirectoryPath: String { modelsDirectoryURL.path }

    /// The Hugging Face hub cache the `hf` CLI writes to:
    /// `HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`.
    public static var hfHubCacheDirectoryURL: URL {
        let env = ProcessInfo.processInfo.environment
        if let cache = env["HF_HUB_CACHE"], !cache.isEmpty {
            return URL(fileURLWithPath: cache.expandingTildePath, isDirectory: true)
        }
        if let home = env["HF_HOME"], !home.isEmpty {
            return URL(fileURLWithPath: home.expandingTildePath, isDirectory: true)
                .appendingPathComponent("hub", isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub", isDirectory: true)
    }

    public static var defaultKVCacheDirectoryURL: URL {
        rootDirectoryURL.appendingPathComponent("kv-cache", isDirectory: true)
    }

    /// The directory holding one knob profile file per model.
    public static var profilesDirectoryURL: URL {
        rootDirectoryURL.appendingPathComponent("profiles", isDirectory: true)
    }

    public static var pidFilePath: String {
        rootDirectoryURL.appendingPathComponent("qwg.pid", isDirectory: false).path
    }

    public static var engineLockFilePath: String {
        rootDirectoryURL.appendingPathComponent("qwg-engine.lock", isDirectory: false).path
    }

    public static var menuLockFilePath: String {
        rootDirectoryURL.appendingPathComponent("qwg-menu.lock", isDirectory: false).path
    }

    /// Persisted update checker state (`~/.qwg/update-state.json`).
    public static var updateStateFileURL: URL {
        let env = ProcessInfo.processInfo.environment
        if let override = env["QWEGN_UPDATE_STATE_PATH"], !override.isEmpty {
            return URL(fileURLWithPath: override.expandingTildePath, isDirectory: false)
        }
        return rootDirectoryURL.appendingPathComponent("update-state.json", isDirectory: false)
    }

    /// Staging cache for downloaded & SHA-256 verified `.pkg` installers.
    public static var updatesCacheDirectoryURL: URL {
        let env = ProcessInfo.processInfo.environment
        if let override = env["QWEGN_UPDATES_CACHE_DIR"], !override.isEmpty {
            return URL(fileURLWithPath: override.expandingTildePath, isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Caches/Qwegn/updates", isDirectory: true)
    }

    /// Per-user Application Support folder (`~/Library/Application Support/Qwegn`).
    public static var applicationSupportDirectoryURL: URL {
        if let env = ProcessInfo.processInfo.environment["QWEGN_APP_SUPPORT_DIR"], !env.isEmpty {
            return URL(fileURLWithPath: env.expandingTildePath, isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/Qwegn", isDirectory: true)
    }

    /// Lazy Python virtual environment for `qwg quantize` (`~/Library/Application Support/Qwegn/venv`).
    public static var pythonVenvDirectoryURL: URL {
        if let env = ProcessInfo.processInfo.environment["QWEGN_VENV_DIR"], !env.isEmpty {
            return URL(fileURLWithPath: env.expandingTildePath, isDirectory: true)
        }
        return applicationSupportDirectoryURL.appendingPathComponent("venv", isDirectory: true)
    }

    /// Support root holding `Scripts/quantize/pack.py` (repo root in dev, `/usr/local/share/qwegn` in `.pkg`).
    public static var supportDirectoryURL: URL {
        let fm = FileManager.default
        if let env = ProcessInfo.processInfo.environment["QWEGINE_ROOT"], !env.isEmpty {
            return URL(fileURLWithPath: env.expandingTildePath, isDirectory: true)
        }
        let execDir = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().deletingLastPathComponent()
        var d = execDir
        for _ in 0..<8 {
            if fm.fileExists(atPath: d.appendingPathComponent("Scripts/quantize/pack.py").path) { return d }
            d = d.deletingLastPathComponent()
        }
        let candidates = [
            execDir.deletingLastPathComponent().appendingPathComponent("share/qwegn", isDirectory: true),
            URL(fileURLWithPath: "/usr/local/share/qwegn", isDirectory: true),
            URL(fileURLWithPath: "/Applications/Qwegn.app/Contents/Resources", isDirectory: true),
        ]
        for cand in candidates where fm.fileExists(atPath: cand.appendingPathComponent("Scripts/quantize/pack.py").path) {
            return cand
        }
        return URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true)
    }

    /// Ensures `url` exists with owner-only `0o700` directory permissions.
    public static func ensurePrivateDirectory(_ url: URL) throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: url.path) {
            try fm.createDirectory(at: url, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        }
        try? fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: url.path)
    }

    /// Sets owner-only `0o600` permissions on the file at `path`.
    public static func setPrivateFilePermissions(atPath path: String) {
        _ = chmod(path, 0o600)
    }
}

public enum QWEngineInstanceError: Error, LocalizedError, CustomStringConvertible, Equatable {
    case engineAlreadyRunning(pid: Int32)
    case menuAlreadyRunning(pid: Int32)
    case lockOpenFailed(path: String, errno: Int32)

    public var description: String {
        switch self {
        case .engineAlreadyRunning(let pid):
            return "Refusing to start a second engine instance: another qwg engine is already running (PID \(pid)). Stop it first before launching a new engine."
        case .menuAlreadyRunning(let pid):
            return "Another qwg-menu instance is already running (PID \(pid))."
        case .lockOpenFailed(let path, let err):
            return "Failed to acquire instance lock at '\(path)': \(String(cString: strerror(err)))"
        }
    }

    public var errorDescription: String? { description }
}

/// Process-level single-instance lock using kernel `flock(LOCK_EX | LOCK_NB)`.
/// Automatically released by the OS kernel if the holding process exits or crashes.
public enum QWEngineInstanceLock {
    private static let lock = NSLock()
    private nonisolated(unsafe) static var engineLockFD: Int32 = -1
    private nonisolated(unsafe) static var menuLockFD: Int32 = -1

    /// Acquires an exclusive kernel lock for the `qwg` engine process (`serve`, `run`, `repl`).
    /// Idempotent within the same process. Throws `QWEngineInstanceError.engineAlreadyRunning`
    /// if another OS process currently holds the engine lock.
    public static func acquireEngineLock() throws {
        try lock.withLock {
            if engineLockFD >= 0 { return }
            try QWPaths.ensurePrivateDirectory(QWPaths.rootDirectoryURL)
            let fd = try acquireLockFile(
                at: QWPaths.engineLockFilePath,
                fallbackPIDPath: QWPaths.pidFilePath,
                alreadyRunning: { QWEngineInstanceError.engineAlreadyRunning(pid: $0) }
            )
            try? "\(ProcessInfo.processInfo.processIdentifier)".write(
                toFile: QWPaths.pidFilePath, atomically: true, encoding: .utf8
            )
            QWPaths.setPrivateFilePermissions(atPath: QWPaths.pidFilePath)
            engineLockFD = fd
        }
    }

    /// Releases the engine lock held by this process, if any.
    public static func releaseEngineLock() {
        lock.withLock {
            guard engineLockFD >= 0 else { return }
            _ = ftruncate(engineLockFD, 0)
            _ = flock(engineLockFD, LOCK_UN)
            close(engineLockFD)
            engineLockFD = -1
            try? FileManager.default.removeItem(atPath: QWPaths.pidFilePath)
        }
    }

    /// Returns the PID of a currently running `qwg` engine process if another process is running, or `nil`.
    public static func runningEnginePID() -> Int32? {
        let heldHere = lock.withLock { engineLockFD >= 0 }
        if heldHere {
            return ProcessInfo.processInfo.processIdentifier
        }
        return runningEnginePID(
            lockFilePath: QWPaths.engineLockFilePath,
            pidFilePath: QWPaths.pidFilePath
        )
    }

    /// Acquires an exclusive kernel lock for `qwg-menu` so two menu bar apps never run simultaneously.
    public static func acquireMenuLock() throws {
        try lock.withLock {
            if menuLockFD >= 0 { return }
            try QWPaths.ensurePrivateDirectory(QWPaths.rootDirectoryURL)
            menuLockFD = try acquireLockFile(
                at: QWPaths.menuLockFilePath,
                fallbackPIDPath: nil,
                alreadyRunning: { QWEngineInstanceError.menuAlreadyRunning(pid: $0) }
            )
        }
    }

    static func acquireLockFile(
        at path: String,
        fallbackPIDPath: String?,
        alreadyRunning: (Int32) -> QWEngineInstanceError
    ) throws -> Int32 {
        let fd = open(path, O_CREAT | O_RDWR | O_CLOEXEC, 0o600)
        guard fd >= 0 else {
            throw QWEngineInstanceError.lockOpenFailed(path: path, errno: errno)
        }
        _ = fchmod(fd, 0o600)
        if flock(fd, LOCK_EX | LOCK_NB) != 0 {
            let lockErr = errno
            let fallbackPid = fallbackPIDPath.flatMap { readPID(fromPath: $0) }
            let existingPid = readPID(fromPath: path) ?? fallbackPid ?? -1
            close(fd)
            if lockErr == EWOULDBLOCK || lockErr == EAGAIN {
                throw alreadyRunning(existingPid)
            }
            throw QWEngineInstanceError.lockOpenFailed(path: path, errno: lockErr)
        }
        let pidStr = "\(ProcessInfo.processInfo.processIdentifier)\n"
        _ = ftruncate(fd, 0)
        _ = lseek(fd, 0, SEEK_SET)
        pidStr.withCString { ptr in
            var remaining = strlen(ptr)
            var offset = 0
            while remaining > 0 {
                let wrote = write(fd, ptr.advanced(by: offset), remaining)
                if wrote < 0 {
                    if errno == EINTR { continue }
                    break
                }
                if wrote == 0 { break }
                offset += wrote
                remaining -= wrote
            }
        }
        return fd
    }

    static func runningEnginePID(
        lockFilePath: String,
        pidFilePath: String,
        processVerifier: (Int32) -> Bool = isVerifiedEngineProcess(pid:)
    ) -> Int32? {
        for path in [lockFilePath, pidFilePath] {
            if let pid = readPID(fromPath: path),
               pid > 0,
               kill(pid, 0) == 0,
               processVerifier(pid) {
                return pid
            }
        }
        return nil
    }

    static func isVerifiedEngineProcess(pid: Int32) -> Bool {
        var buffer = [UInt8](repeating: 0, count: 4096)
        let length = proc_pidpath(pid, &buffer, UInt32(buffer.count))
        guard length > 0 else { return false }
        let path = String(decoding: buffer.prefix(Int(length)), as: UTF8.self)
        return URL(fileURLWithPath: path).lastPathComponent == "qwg"
    }

    private static func readPID(fromPath path: String) -> Int32? {
        guard let str = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
        return Int32(str.trimmingCharacters(in: .whitespacesAndNewlines))
    }
}

extension String {
    /// Expands a leading `~` or `~user` to the full filesystem path.
    @inline(__always)
    public var expandingTildePath: String {
        (self as NSString).expandingTildeInPath
    }

    /// Canonicalized filesystem path expanding leading tilde (`~`) if present.
    @inline(__always)
    public var canonicalPath: String {
        expandingTildePath
    }

    /// Expands `~` and returns a file `URL`.
    @inline(__always)
    public var canonicalFileURL: URL {
        URL(fileURLWithPath: expandingTildePath)
    }
}
