// QWSMCHelperManager.swift — LaunchDaemon plist generation, status, and install orchestration.
// Supports SMAppService.daemon for Developer ID app bundles and LaunchDaemon installation
// via admin authorization or CLI root install for ad-hoc builds and .pkg packages.

import Darwin
import Foundation
import Security
import ServiceManagement

/// Installation and runtime health state of `qwg-smc-helper`.
public enum QWSMCHelperStatus: String, Sendable, Equatable {
    case active
    case requiresApproval
    case installedNotRunning
    case notInstalled
}

/// Manages LaunchDaemon installation, status inspection, and removal for `qwg-smc-helper`.
public enum QWSMCHelperManager {
    public enum ManagerError: Error, LocalizedError {
        case rootRequired(String)
        case helperBinaryMissing
        case commandFailed(String)
        case daemonUnreachable

        public var errorDescription: String? {
            switch self {
            case .rootRequired(let cmd): return "Root privileges required. Run: \(cmd)"
            case .helperBinaryMissing: return "qwg-smc-helper binary not found next to Qwegn.app or qwg"
            case .commandFailed(let msg): return msg
            case .daemonUnreachable: return "qwg-smc-helper installed, but /var/run/qwg-smc.sock did not respond"
            }
        }
    }

    /// LaunchDaemon XML plist for `/Library/LaunchDaemons/ai.qwegn.smc-helper.plist`.
    public static func systemLaunchDaemonPlistXML(
        binaryPath: String = QWSMCHelperConstants.installedBinaryPath
    ) -> String {
        let escaped = xmlEscaped(binaryPath)
        return """
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>\(QWSMCHelperConstants.daemonLabel)</string>
            <key>ProgramArguments</key>
            <array>
                <string>\(escaped)</string>
            </array>
            <key>RunAtLoad</key>
            <true/>
            <key>KeepAlive</key>
            <true/>
            <key>StandardErrorPath</key>
            <string>/var/log/qwg-smc-helper.log</string>
        </dict>
        </plist>
        """
    }

    /// LaunchDaemon XML plist bundled inside `Qwegn.app/Contents/Library/LaunchDaemons/`.
    public static func bundledSMAppServicePlistXML() -> String {
        """
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>\(QWSMCHelperConstants.daemonLabel)</string>
            <key>BundleProgram</key>
            <string>Contents/MacOS/qwg-smc-helper</string>
            <key>RunAtLoad</key>
            <true/>
            <key>KeepAlive</key>
            <true/>
        </dict>
        </plist>
        """
    }

    /// Finds an executable `qwg-smc-helper` across bundle, CLI sibling, `/Applications`, and `/usr/local/libexec`.
    public static func locateHelperBinary(
        bundleURL: URL? = Bundle.main.bundleURL,
        cliExecutablePath: String? = CommandLine.arguments.first
    ) -> String? {
        var candidates: [String] = []
        if let bundleURL {
            candidates.append(bundleURL.appendingPathComponent("Contents/MacOS/qwg-smc-helper").path)
            candidates.append(bundleURL.appendingPathComponent("qwg-smc-helper").path)
        }
        if let cliPath = cliExecutablePath, !cliPath.isEmpty {
            let resolved = URL(fileURLWithPath: cliPath).resolvingSymlinksInPath().deletingLastPathComponent()
            candidates.append(resolved.appendingPathComponent("qwg-smc-helper").path)
        }
        candidates.append("/Applications/Qwegn.app/Contents/MacOS/qwg-smc-helper")
        candidates.append(QWSMCHelperConstants.installedBinaryPath)
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
    }

    /// Probes the live helper socket and `launchd`/`SMAppService` registration state.
    public static func currentStatus(socketPath: String = QWSMCHelperConstants.socketPath) -> QWSMCHelperStatus {
        if QWSMCHelperClient.ping(socketPath: socketPath) {
            return .active
        }
        let svcStatus = SMAppService.daemon(plistName: QWSMCHelperConstants.plistFileName).status
        if svcStatus == .requiresApproval {
            return .requiresApproval
        }
        if FileManager.default.fileExists(atPath: QWSMCHelperConstants.systemPlistPath) || svcStatus == .enabled {
            return .installedNotRunning
        }
        return .notInstalled
    }

    /// Installs `/usr/local/libexec/qwg-smc-helper` and `/Library/LaunchDaemons/ai.qwegn.smc-helper.plist` (root only).
    public static func installSystemDaemon(sourceBinaryPath: String) throws {
        guard Darwin.geteuid() == 0 else {
            throw ManagerError.rootRequired("sudo qwg fan-helper install")
        }
        guard FileManager.default.isExecutableFile(atPath: sourceBinaryPath) else {
            throw ManagerError.helperBinaryMissing
        }
        let fm = FileManager.default
        let libexecDir = URL(fileURLWithPath: QWSMCHelperConstants.installedBinaryPath).deletingLastPathComponent().path
        try fm.createDirectory(atPath: libexecDir, withIntermediateDirectories: true)
        _ = Darwin.chown(libexecDir, 0, 0)
        _ = Darwin.chmod(libexecDir, 0o755)

        let srcURL = URL(fileURLWithPath: sourceBinaryPath).resolvingSymlinksInPath()
        let dstURL = URL(fileURLWithPath: QWSMCHelperConstants.installedBinaryPath)
        if srcURL.path != dstURL.resolvingSymlinksInPath().path {
            let stageURL = URL(fileURLWithPath: QWSMCHelperConstants.installedBinaryPath + ".staging.\(getpid())")
            try? fm.removeItem(at: stageURL)
            try fm.copyItem(at: srcURL, to: stageURL)
            _ = Darwin.chown(stageURL.path, 0, 0)
            _ = Darwin.chmod(stageURL.path, 0o755)
            _ = try fm.replaceItemAt(dstURL, withItemAt: stageURL)
        }
        _ = Darwin.chown(dstURL.path, 0, 0)
        _ = Darwin.chmod(dstURL.path, 0o755)

        let plistURL = URL(fileURLWithPath: QWSMCHelperConstants.systemPlistPath)
        let launchDaemonsDir = plistURL.deletingLastPathComponent().path
        try fm.createDirectory(atPath: launchDaemonsDir, withIntermediateDirectories: true)
        try systemLaunchDaemonPlistXML().write(to: plistURL, atomically: true, encoding: .utf8)
        _ = Darwin.chown(plistURL.path, 0, 0)
        _ = Darwin.chmod(plistURL.path, 0o644)

        let serviceTarget = "system/\(QWSMCHelperConstants.daemonLabel)"
        _ = runProcess("/bin/launchctl", args: ["bootout", serviceTarget])
        let bootRes = runProcess("/bin/launchctl", args: ["bootstrap", "system", plistURL.path])
        if bootRes.status != 0 {
            throw ManagerError.commandFailed("launchctl bootstrap failed: \(bootRes.output)")
        }
        _ = runProcess("/bin/launchctl", args: ["kickstart", "-k", serviceTarget])

        for _ in 0..<30 {
            if QWSMCHelperClient.ping() { return }
            Thread.sleep(forTimeInterval: 0.1)
        }
        throw ManagerError.daemonUnreachable
    }

    /// Stops `ai.qwegn.smc-helper`, restores System Auto fans, and removes daemon files (root only).
    public static func uninstallSystemDaemon() throws {
        guard Darwin.geteuid() == 0 else {
            throw ManagerError.rootRequired("sudo qwg fan-helper uninstall")
        }
        if QWSMCHelperClient.ping() {
            _ = QWSMCHelperClient.send("AUTO\n")
        }
        _ = runProcess("/bin/launchctl", args: ["bootout", "system/\(QWSMCHelperConstants.daemonLabel)"])
        let fm = FileManager.default
        try? fm.removeItem(atPath: QWSMCHelperConstants.systemPlistPath)
        var st = stat()
        if Darwin.lstat(QWSMCHelperConstants.socketPath, &st) == 0 && (st.st_mode & S_IFMT) == S_IFSOCK {
            Darwin.unlink(QWSMCHelperConstants.socketPath)
        }
        try? fm.removeItem(atPath: QWSMCHelperConstants.installedBinaryPath)
    }

    /// Enables the privileged helper from `Qwegn.app` (via `SMAppService` if signed, else admin prompt).
    public static func enableFromApp(bundleURL: URL = Bundle.main.bundleURL) throws -> QWSMCHelperStatus {
        let current = currentStatus()
        if current == .active { return .active }

        let embeddedPlist = bundleURL.appendingPathComponent("Contents/Library/LaunchDaemons/\(QWSMCHelperConstants.plistFileName)").path
        if FileManager.default.fileExists(atPath: embeddedPlist), hasDeveloperIDTeamIdentifier(at: bundleURL) {
            let service = SMAppService.daemon(plistName: QWSMCHelperConstants.plistFileName)
            try? service.register()
            if service.status == .requiresApproval {
                return .requiresApproval
            }
            for _ in 0..<20 {
                if QWSMCHelperClient.ping() { return .active }
                Thread.sleep(forTimeInterval: 0.1)
            }
        }

        guard let helperPath = locateHelperBinary(bundleURL: bundleURL) else {
            throw ManagerError.helperBinaryMissing
        }
        let cmd = "\(shellSingleQuoted(helperPath)) --install"
        let res = runProcess("/usr/bin/osascript", args: ["-e", "do shell script \"\(appleScriptQuoted(cmd))\" with administrator privileges"])
        guard res.status == 0 else {
            throw ManagerError.commandFailed(res.output.isEmpty ? "Administrator authorization was canceled" : res.output)
        }
        let status = currentStatus()
        guard status == .active else {
            throw ManagerError.daemonUnreachable
        }
        return status
    }

    /// Disables and removes the privileged helper from `Qwegn.app`.
    public static func disableFromApp(bundleURL: URL = Bundle.main.bundleURL) throws -> QWSMCHelperStatus {
        let service = SMAppService.daemon(plistName: QWSMCHelperConstants.plistFileName)
        if service.status == .enabled || service.status == .requiresApproval {
            try? service.unregister()
        }
        if FileManager.default.fileExists(atPath: QWSMCHelperConstants.systemPlistPath) || QWSMCHelperClient.ping() {
            let helperPath = locateHelperBinary(bundleURL: bundleURL) ?? QWSMCHelperConstants.installedBinaryPath
            guard FileManager.default.isExecutableFile(atPath: helperPath) else {
                throw ManagerError.helperBinaryMissing
            }
            let cmd = "\(shellSingleQuoted(helperPath)) --uninstall"
            let res = runProcess("/usr/bin/osascript", args: ["-e", "do shell script \"\(appleScriptQuoted(cmd))\" with administrator privileges"])
            guard res.status == 0 else {
                throw ManagerError.commandFailed(res.output.isEmpty ? "Administrator authorization was canceled" : res.output)
            }
        }
        return currentStatus()
    }

    public static func shellSingleQuoted(_ arg: String) -> String {
        "'" + arg.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    public static func appleScriptQuoted(_ script: String) -> String {
        script
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }

    private static func xmlEscaped(_ value: String) -> String {
        value
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
    }

    private static func hasDeveloperIDTeamIdentifier(at bundleURL: URL) -> Bool {
        var code: SecStaticCode?
        guard SecStaticCodeCreateWithPath(bundleURL as CFURL, [], &code) == errSecSuccess,
              let staticCode = code else {
            return false
        }
        var info: CFDictionary?
        let flags = SecCSFlags(rawValue: kSecCSSigningInformation)
        guard SecCodeCopySigningInformation(staticCode, flags, &info) == errSecSuccess,
              let dict = info as? [String: Any],
              let team = dict[kSecCodeInfoTeamIdentifier as String] as? String else {
            return false
        }
        return !team.isEmpty
    }

    private static func runProcess(_ executable: String, args: [String]) -> (status: Int32, output: String) {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: executable)
        proc.arguments = args
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        do {
            try proc.run()
        } catch {
            return (1, error.localizedDescription)
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()
        let text = (String(data: data, encoding: .utf8) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        return (proc.terminationStatus, text)
    }
}
