// main.swift — Minimal privileged AppleSMC fan helper daemon (`qwg-smc-helper`).
// Accepts bounded SET/AUTO/PING commands over /var/run/qwg-smc.sock with
// hardware [minRPM, maxRPM] clamping and 15 s dead-man auto-restore.

import Darwin
import Foundation
import QWSMCHelper

private nonisolated(unsafe) var qwgHelperStopSignal: sig_atomic_t = 0

private func handleStopSignal(_ sig: Int32) {
    let _ = sig
    qwgHelperStopSignal = 1
}

private func printUsage() {
    print("Usage: qwg-smc-helper [--install | --uninstall | --socket PATH --watchdog-sec SEC]")
}

let args = Array(CommandLine.arguments.dropFirst())
var socketPath = QWSMCHelperConstants.socketPath
var watchdogSeconds = QWSMCHelperConstants.defaultWatchdogSeconds
var idx = 0

while idx < args.count {
    switch args[idx] {
    case "--help", "-h":
        printUsage()
        exit(0)
    case "--install":
        do {
            let exePath = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().path
            try QWSMCHelperManager.installSystemDaemon(sourceBinaryPath: exePath)
            print("qwg-smc-helper installed and active at \(QWSMCHelperConstants.socketPath)")
            exit(0)
        } catch {
            fputs("qwg-smc-helper install failed: \(error.localizedDescription)\n", stderr)
            exit(1)
        }
    case "--uninstall":
        do {
            try QWSMCHelperManager.uninstallSystemDaemon()
            print("qwg-smc-helper uninstalled")
            exit(0)
        } catch {
            fputs("qwg-smc-helper uninstall failed: \(error.localizedDescription)\n", stderr)
            exit(1)
        }
    case "--socket":
        guard idx + 1 < args.count else {
            fputs("qwg-smc-helper: --socket requires a path\n", stderr)
            exit(2)
        }
        socketPath = args[idx + 1]
        idx += 2
    case "--watchdog-sec":
        guard idx + 1 < args.count, let sec = Double(args[idx + 1]), sec >= 1.0 else {
            fputs("qwg-smc-helper: --watchdog-sec requires >= 1.0\n", stderr)
            exit(2)
        }
        watchdogSeconds = sec
        idx += 2
    default:
        fputs("qwg-smc-helper: unknown argument '\(args[idx])'\n", stderr)
        exit(2)
    }
}

_ = Darwin.signal(SIGTERM, handleStopSignal)
_ = Darwin.signal(SIGINT, handleStopSignal)

let server = QWSMCHelperServer(socketPath: socketPath, watchdogTimeout: watchdogSeconds)
do {
    try server.run { qwgHelperStopSignal != 0 }
} catch {
    fputs("qwg-smc-helper error: \(error.localizedDescription)\n", stderr)
    exit(1)
}
