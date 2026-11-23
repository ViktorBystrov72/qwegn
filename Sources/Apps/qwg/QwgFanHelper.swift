// QwgFanHelper.swift — `qwg fan-helper status|install|uninstall` subcommands.
// Lets CLI users inspect or install the privileged qwg-smc-helper LaunchDaemon
// without running inference or the HTTP server as root.

import ArgumentParser
import Darwin
import Foundation
import QWSMCHelper

struct FanHelper: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "fan-helper",
        abstract: "Inspect, install, or uninstall the privileged qwg-smc-helper LaunchDaemon for non-root fan control.",
        subcommands: [Status.self, Install.self, Uninstall.self],
        defaultSubcommand: Status.self
    )

    struct Status: ParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Show qwg-smc-helper status and AppleSMC fan capabilities.")

        func run() throws {
            let status = QWSMCHelperManager.currentStatus()
            let binaryPath = QWSMCHelperManager.locateHelperBinary() ?? "(not found)"
            let client = QWSMCClient()
            let fans = client.fanCount()
            print("status: \(status.rawValue)")
            print("socket: \(QWSMCHelperConstants.socketPath)")
            print("plist: \(QWSMCHelperConstants.systemPlistPath)")
            print("helperBinary: \(binaryPath)")
            print("fans: \(fans)")
            for idx in 0..<fans {
                let bounds = client.fanBounds(index: idx)
                let cur = client.readValue("F\(idx)Ac").map { String(format: "%.0f", $0) } ?? "—"
                print("fan[\(idx)]: actual=\(cur) rpm bounds=[\(Int(bounds.minRPM)), \(Int(bounds.maxRPM))] rpm")
            }
        }
    }

    struct Install: ParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Install qwg-smc-helper LaunchDaemon (requires sudo/root).")
        @Option(name: .customLong("helper-bin"), help: "Path to qwg-smc-helper binary.") var helperBin: String?

        func run() throws {
            guard Darwin.geteuid() == 0 else {
                FileHandle.standardError.write(Data("qwg fan-helper install: run as root: sudo qwg fan-helper install\n".utf8))
                throw ExitCode(1)
            }
            guard let bin = helperBin ?? QWSMCHelperManager.locateHelperBinary() else {
                FileHandle.standardError.write(Data("qwg fan-helper install: qwg-smc-helper binary not found\n".utf8))
                throw ExitCode(1)
            }
            try QWSMCHelperManager.installSystemDaemon(sourceBinaryPath: bin)
            print("qwg-smc-helper installed and active at \(QWSMCHelperConstants.socketPath)")
        }
    }

    struct Uninstall: ParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Stop and remove the qwg-smc-helper LaunchDaemon (requires sudo/root).")

        func run() throws {
            guard Darwin.geteuid() == 0 else {
                FileHandle.standardError.write(Data("qwg fan-helper uninstall: run as root: sudo qwg fan-helper uninstall\n".utf8))
                throw ExitCode(1)
            }
            try QWSMCHelperManager.uninstallSystemDaemon()
            print("qwg-smc-helper uninstalled")
        }
    }
}
