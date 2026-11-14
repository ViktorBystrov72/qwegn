// QWMenuBarUITests.swift — Smoke tests validating QWMenuBarUI library linkage and state defaults.
import Observation
import os
@testable import QWMenuBarUI
import QWServer
import XCTest

final class QWMenuBarUITests: XCTestCase {
    func testEngineAndConnectionStateDefaults() {
        XCTAssertFalse(EngineState.stopped.isRunningOrUp)
        XCTAssertFalse(EngineState.stopped.isTransitioning)
        XCTAssertTrue(EngineState.starting.isTransitioning)
        XCTAssertTrue(EngineState.stopping.isTransitioning)
        XCTAssertTrue(EngineState.serverUp(isLoaded: true).isRunningOrUp)
        XCTAssertEqual(ConnectionState.allCases.count, 3)
        XCTAssertEqual(String(describing: TelemetryObservable.self), "TelemetryObservable")
    }

    @MainActor
    func testTelemetryObservableTracksStateMutations() {
        let telemetry = TelemetryObservable(startServices: false, initialSettings: QWServerSettings())
        let stateChanged = OSAllocatedUnfairLock(initialState: false)
        withObservationTracking {
            _ = telemetry.engineState
        } onChange: {
            stateChanged.withLock { $0 = true }
        }

        // Unread tracked property and @ObservationIgnored property do not trigger observers.
        telemetry.recentLogs = ["ignored"]
        telemetry.isPopoverVisible = true
        XCTAssertFalse(stateChanged.withLock { $0 })

        // Tracked property mutation fires fine-grained observation.
        telemetry.engineState = .starting
        XCTAssertTrue(stateChanged.withLock { $0 })
    }
}
