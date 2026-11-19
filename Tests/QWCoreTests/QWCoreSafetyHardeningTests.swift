// QWCoreSafetyHardeningTests.swift — Regression tests for QWPaths, QWTokenRecycler, and QWContextCopyDrafter.
import Darwin
import Foundation
import XCTest
@testable import QWCore

final class QWCoreSafetyHardeningTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwcore-sec-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    func testLockFileSetsCLOEXECAndFailsClosedOnUnopenablePath() throws {
        let lockPath = dir.appendingPathComponent("test.lock").path
        let fd = try QWEngineInstanceLock.acquireLockFile(
            at: lockPath,
            fallbackPIDPath: nil,
            alreadyRunning: { QWEngineInstanceError.engineAlreadyRunning(pid: $0) }
        )
        defer {
            _ = flock(fd, LOCK_UN)
            close(fd)
        }
        let flags = fcntl(fd, F_GETFD)
        XCTAssertNotEqual(flags, -1)
        XCTAssertNotEqual(flags & FD_CLOEXEC, 0, "Lock FD must have FD_CLOEXEC")

        // A second open cannot acquire LOCK_EX and must throw engineAlreadyRunning.
        XCTAssertThrowsError(try QWEngineInstanceLock.acquireLockFile(
            at: lockPath,
            fallbackPIDPath: nil,
            alreadyRunning: { QWEngineInstanceError.engineAlreadyRunning(pid: $0) }
        )) { err in
            guard case QWEngineInstanceError.engineAlreadyRunning = err else {
                return XCTFail("Expected engineAlreadyRunning, got \(err)")
            }
        }

        // Unopenable parent path (file used as directory) must fail closed, never return silently.
        let badPath = lockPath + "/nested.lock"
        XCTAssertThrowsError(try QWEngineInstanceLock.acquireLockFile(
            at: badPath,
            fallbackPIDPath: nil,
            alreadyRunning: { QWEngineInstanceError.engineAlreadyRunning(pid: $0) }
        )) { err in
            guard case QWEngineInstanceError.lockOpenFailed = err else {
                return XCTFail("Expected lockOpenFailed, got \(err)")
            }
        }
    }

    func testRunningEnginePIDRejectsNonQWGProcessViaProcPidPath() throws {
        let lockPath = dir.appendingPathComponent("engine.lock").path
        let pidPath = dir.appendingPathComponent("engine.pid").path
        let selfPID = ProcessInfo.processInfo.processIdentifier
        try "\(selfPID)\n".write(toFile: lockPath, atomically: true, encoding: .utf8)
        try "\(selfPID)\n".write(toFile: pidPath, atomically: true, encoding: .utf8)

        XCTAssertFalse(QWEngineInstanceLock.isVerifiedEngineProcess(pid: selfPID))
        XCTAssertFalse(QWEngineInstanceLock.isVerifiedEngineProcess(pid: -1))
        XCTAssertNil(QWEngineInstanceLock.runningEnginePID(lockFilePath: lockPath, pidFilePath: pidPath))
        XCTAssertEqual(
            QWEngineInstanceLock.runningEnginePID(
                lockFilePath: lockPath,
                pidFilePath: pidPath,
                processVerifier: { $0 == selfPID }
            ),
            selfPID
        )
    }

    func testTokenRecyclerRejectsOutOfRangeInt32Tokens() {
        let recycler = QWTokenRecycler()
        recycler.recordTransition(previous: 1, current: 2, next: 3)
        recycler.recordTransition(previous: Int.max, current: 2, next: 3)
        recycler.recordTransition(previous: 1, current: Int(Int32.max) + 1, next: 3)

        var drafts = [Int(Int32.max) + 1]
        XCTAssertEqual(recycler.appendDrafts(
            into: &drafts, maxCount: 4, previousToken: 1, pendingToken: 2
        ), 0)

        var empty: [Int] = []
        XCTAssertEqual(recycler.appendDrafts(
            into: &empty, maxCount: 4, previousToken: Int.max, pendingToken: Int(Int32.max) + 1
        ), 0)

        var sentinel = [-1]
        XCTAssertEqual(recycler.appendDrafts(
            into: &sentinel, maxCount: 4, previousToken: nil, pendingToken: 1
        ), 0)
    }

    func testContextCopyDrafterGuardsInvalidPeriodRepeatsAndMatchLens() {
        let history: [Int64] = [1, 2, 3, 1, 2, 3]
        XCTAssertNil(QWContextCopyDrafter.periodicDrafts(history: history, maxPeriod: 0, minRepeats: 2))
        XCTAssertNil(QWContextCopyDrafter.periodicDrafts(history: history, maxPeriod: -3, minRepeats: 2))
        XCTAssertNil(QWContextCopyDrafter.periodicDrafts(history: history, maxPeriod: 3, minRepeats: 0))
        XCTAssertNil(QWContextCopyDrafter.periodicDrafts(history: history, maxPeriod: 3, minRepeats: -1))
        XCTAssertNil(QWContextCopyDrafter.periodicDrafts(history: [1, 2], maxPeriod: Int.max, minRepeats: 2))
        XCTAssertNil(QWContextCopyDrafter.drafts(history: history, minMatchLen: 0, maxMatchLen: 5))
        XCTAssertNil(QWContextCopyDrafter.drafts(history: history, minMatchLen: 5, maxMatchLen: 3))
    }
}
