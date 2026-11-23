// QWSMCHelperTests.swift — Tests for AppleSMC helper protocol, clamping, watchdog, and socket client.

import Darwin
import Foundation
@testable import QWServer
import QWSMCHelper
import XCTest

private final class MockSMCBackend: QWSMCBackend, @unchecked Sendable {
    private let lock = NSLock()
    let count: Int
    let minRPM: Double
    let maxRPM: Double
    var failWrites = false
    var appliedTargets: [Int: Double] = [:]
    var autoRestoreCount = 0
    var setCallCount = 0

    init(count: Int = 2, minRPM: Double = 1200, maxRPM: Double = 5400) {
        self.count = count
        self.minRPM = minRPM
        self.maxRPM = maxRPM
    }

    func fanCount() -> Int { count }

    func fanBounds(index: Int) -> (minRPM: Double, maxRPM: Double) {
        let _ = index
        return (minRPM, maxRPM)
    }

    func setFanTarget(index: Int, rpm: Double) -> Result<Double, QWSMCBackendError> {
        lock.withLock {
            setCallCount += 1
            if failWrites { return .failure(QWSMCBackendError("mock_write_denied")) }
            appliedTargets[index] = rpm
            return .success(rpm)
        }
    }

    func restoreAuto(fanCount: Int) -> Result<Void, QWSMCBackendError> {
        let _ = fanCount
        return lock.withLock {
            if failWrites { return .failure(QWSMCBackendError("mock_auto_denied")) }
            autoRestoreCount += 1
            return .success(())
        }
    }

    var snapshotState: (targets: [Int: Double], autoRestores: Int, sets: Int) {
        lock.withLock { (appliedTargets, autoRestoreCount, setCallCount) }
    }
}

final class QWSMCHelperTests: XCTestCase {
    func testProtocolCommandParsingClampingAndSafetyFloors() {
        let backend = MockSMCBackend(count: 2, minRPM: 1200, maxRPM: 5400)
        let server = QWSMCHelperServer(socketPath: "/tmp/unused.sock", backend: backend)

        XCTAssertEqual(server.handleCommand("PING\n"), "OK PONG\n")
        XCTAssertEqual(server.handleCommand("SET 0 3200\n", now: 10), "OK SET 0 3200\n")
        XCTAssertEqual(backend.snapshotState.targets[0], 3200)

        // Never allow sub-minimum RPM (fan-stop protection).
        XCTAssertEqual(server.handleCommand("SET 0 500\n", now: 11), "OK SET 0 1200\n")
        XCTAssertEqual(backend.snapshotState.targets[0], 1200)

        // Clamp above hardware max to maxRPM.
        XCTAssertEqual(server.handleCommand("SET 1 9999\n", now: 12), "OK SET 1 5400\n")
        XCTAssertEqual(backend.snapshotState.targets[1], 5400)

        // Reject invalid index, negative RPM, and unknown verbs.
        XCTAssertTrue(server.handleCommand("SET 9 3000\n").hasPrefix("ERR "))
        XCTAssertTrue(server.handleCommand("SET 0 -10\n").hasPrefix("ERR "))
        XCTAssertTrue(server.handleCommand("SET 0 0\n").hasPrefix("ERR "))
        XCTAssertTrue(server.handleCommand("BOGUS\n").hasPrefix("ERR "))

        XCTAssertEqual(server.handleCommand("AUTO\n"), "OK AUTO\n")
        XCTAssertEqual(backend.snapshotState.autoRestores, 1)

        // Enforce global 1000 RPM floor even if a host sensor reports lower minRPM.
        let lowFloorBackend = MockSMCBackend(count: 1, minRPM: 400, maxRPM: 3000)
        let lowFloorServer = QWSMCHelperServer(socketPath: "/tmp/unused2.sock", backend: lowFloorBackend)
        XCTAssertEqual(lowFloorServer.handleCommand("SET 0 450\n", now: 1), "OK SET 0 1000\n")
    }

    func testDeadManWatchdogRestoresAutoAfterTimeout() {
        let backend = MockSMCBackend(count: 2, minRPM: 1200, maxRPM: 5400)
        let server = QWSMCHelperServer(socketPath: "/tmp/unused3.sock", watchdogTimeout: 15.0, backend: backend)

        XCTAssertFalse(server.checkWatchdog(now: 50.0))
        XCTAssertEqual(server.handleCommand("SET 0 4000\n", now: 100.0), "OK SET 0 4000\n")
        XCTAssertFalse(server.checkWatchdog(now: 114.0))
        XCTAssertEqual(backend.snapshotState.autoRestores, 0)

        XCTAssertTrue(server.checkWatchdog(now: 115.1))
        XCTAssertEqual(backend.snapshotState.autoRestores, 1)
        XCTAssertFalse(server.checkWatchdog(now: 130.0))
        XCTAssertEqual(backend.snapshotState.autoRestores, 1)
    }

    func testEndToEndUnixSocketAndQWFanControllerReplyVerification() throws {
        let sockPath = "/tmp/qwg-smc-\(UUID().uuidString.prefix(8)).sock"
        let backend = MockSMCBackend(count: 2, minRPM: 1200, maxRPM: 5400)
        let server = QWSMCHelperServer(socketPath: sockPath, watchdogTimeout: 60.0, backend: backend)
        try server.startListening()

        let serverDone = DispatchSemaphore(value: 0)
        DispatchQueue.global(qos: .userInitiated).async {
            try? server.run()
            serverDone.signal()
        }
        defer {
            server.stop()
            _ = serverDone.wait(timeout: .now() + 2.0)
            Darwin.unlink(sockPath)
        }

        XCTAssertTrue(QWSMCHelperClient.ping(socketPath: sockPath))
        XCTAssertEqual(QWSMCHelperManager.currentStatus(socketPath: sockPath), .active)

        let setResp = try XCTUnwrap(QWSMCHelperClient.send("SET 0 3500\n", socketPath: sockPath))
        XCTAssertTrue(setResp.ok)
        XCTAssertEqual(setResp.message, "SET 0 3500")
        XCTAssertEqual(backend.snapshotState.targets[0], 3500)

        let autoResp = try XCTUnwrap(QWSMCHelperClient.send("AUTO\n", socketPath: sockPath))
        XCTAssertTrue(autoResp.ok)
        XCTAssertEqual(backend.snapshotState.autoRestores, 1)

        if ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true",
           QWSMCClient().fanCount() > 0 {
            let fan = QWFanController(socketPath: sockPath, registerAtExit: false)
            fan.setPolicy(.max)
            let maxSnap = fan.sampleAndReconcile()
            XCTAssertTrue(maxSnap.fanControlStatus.hasPrefix("Max Blast"), "got: \(maxSnap.fanControlStatus)")
            XCTAssertGreaterThan(backend.snapshotState.sets, 1)

            let beforeAuto = backend.snapshotState.autoRestores
            fan.setPolicy(.auto)
            let autoSnap = fan.sampleAndReconcile()
            XCTAssertEqual(autoSnap.fanControlStatus, "System Auto")
            XCTAssertGreaterThan(backend.snapshotState.autoRestores, beforeAuto)

            backend.failWrites = true
            fan.setPolicy(.max)
            let deniedSnap = fan.sampleAndReconcile()
            XCTAssertTrue(deniedSnap.fanControlStatus.contains("requires root"),
                          "ERR reply must report unauthorized, got: \(deniedSnap.fanControlStatus)")
        } else {
            backend.failWrites = true
        }

        let errResp = try XCTUnwrap(QWSMCHelperClient.send("SET 0 3000\n", socketPath: sockPath))
        XCTAssertFalse(errResp.ok)
        XCTAssertTrue(errResp.message.contains("mock_write_denied"))

        server.stop()
        XCTAssertEqual(serverDone.wait(timeout: .now() + 2.0), .success)
        XCTAssertFalse(FileManager.default.fileExists(atPath: sockPath))
        XCTAssertFalse(QWSMCHelperClient.ping(socketPath: sockPath))
    }

    func testLaunchDaemonPlistGenerationAndShellEscaping() throws {
        let sysXML = QWSMCHelperManager.systemLaunchDaemonPlistXML(binaryPath: "/usr/local/libexec/qwg-smc-helper")
        let sysPlist = try XCTUnwrap(
            PropertyListSerialization.propertyList(from: Data(sysXML.utf8), format: nil) as? [String: Any]
        )
        XCTAssertEqual(sysPlist["Label"] as? String, QWSMCHelperConstants.daemonLabel)
        XCTAssertEqual(sysPlist["ProgramArguments"] as? [String], ["/usr/local/libexec/qwg-smc-helper"])
        XCTAssertEqual(sysPlist["RunAtLoad"] as? Bool, true)
        XCTAssertEqual(sysPlist["KeepAlive"] as? Bool, true)

        let bundleXML = QWSMCHelperManager.bundledSMAppServicePlistXML()
        let bundlePlist = try XCTUnwrap(
            PropertyListSerialization.propertyList(from: Data(bundleXML.utf8), format: nil) as? [String: Any]
        )
        XCTAssertEqual(bundlePlist["Label"] as? String, QWSMCHelperConstants.daemonLabel)
        XCTAssertEqual(bundlePlist["BundleProgram"] as? String, "Contents/MacOS/qwg-smc-helper")
        XCTAssertEqual(bundlePlist["RunAtLoad"] as? Bool, true)
        XCTAssertEqual(bundlePlist["KeepAlive"] as? Bool, true)

        XCTAssertEqual(
            QWSMCHelperManager.shellSingleQuoted("/Users/a b/O'Reilly/qwg-smc-helper"),
            "'/Users/a b/O'\\''Reilly/qwg-smc-helper'"
        )
        XCTAssertEqual(
            QWSMCHelperManager.appleScriptQuoted("'/a\\b/\"c\"' --install"),
            "'/a\\\\b/\\\"c\\\"' --install"
        )
    }
}
