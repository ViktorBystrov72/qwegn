// QWMenuBarUITests.swift — Smoke tests validating QWMenuBarUI library linkage and state defaults.
import Observation
import os
import QWCore
@testable import QWMenuBarUI
import QWServer
import SwiftUI
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

    func testMarqueeMathHoldsThenScrollsAndStaysZeroWhenFitting() {
        XCTAssertEqual(MarqueeMath.offset(elapsed: 3.0, textWidth: 160, containerWidth: 200), 0)
        XCTAssertEqual(MarqueeMath.offset(elapsed: 0.0, textWidth: 340, containerWidth: 200), 0)
        XCTAssertEqual(MarqueeMath.offset(elapsed: 1.39, textWidth: 340, containerWidth: 200, holdSeconds: 1.4), 0)

        let moving = MarqueeMath.offset(
            elapsed: 2.4,
            textWidth: 340,
            containerWidth: 200,
            gap: 28,
            holdSeconds: 1.4,
            speedPtPerSec: 28
        )
        XCTAssertEqual(moving, -28.0, accuracy: 0.01)

        // Wraps cleanly back to leading hold once (textWidth + gap) completes.
        let period = 1.4 + Double(340 + 28) / 28.0
        XCTAssertEqual(
            MarqueeMath.offset(
                elapsed: period + 0.5,
                textWidth: 340,
                containerWidth: 200,
                gap: 28,
                holdSeconds: 1.4,
                speedPtPerSec: 28
            ),
            0
        )
    }

    @MainActor
    func testCupertinoTelemetryViewStays440PtWideWithLongModelAndQuantization() {
        let telemetry = TelemetryObservable(
            startServices: false,
            enableAutoLifecycle: false,
            initialSettings: QWServerSettings()
        )
        let modelName = "ERNIE-5.5-Flash-Base-hq8g32-shvq15g64-shvq3g64"
        let dashboard = QWDashboardViewModel.build(
            lifecycle: .init(phase: .loaded, since: Date(), model: modelName, detail: nil),
            modelId: modelName,
            capabilities: [.prefixCache],
            knobs: [
                "kvMode": .string("q4"),
                "contextTokens": .int(1_048_576),
                "speculativeEnabled": .bool(true),
                "mtpDrafts": .int(3),
            ],
            facts: QWDashboardFacts(
                quantization: "HQ8_G32+ExpSHVQ15_G64/SHVQ3_G64 (native, GPU)",
                hostLogicalCores: 18,
                gdnPrefillMode: "chunked",
                gpuName: "Apple M5 Max",
                metalFamily: "Metal 4",
                physicalMemoryGB: 128,
                lsaMode: "auto",
                mountedMemoryPack: "qwegine (Q4)"
            )
        )
        telemetry.connection = .connected
        telemetry.engineState = .serverUp(isLoaded: true)
        telemetry.isDaemonTransitioning = false
        telemetry.dashboard = dashboard

        let hosting = NSHostingView(rootView: CupertinoTelemetryView(telemetry: telemetry))
        hosting.frame = NSRect(x: 0, y: 0, width: 440, height: 640)
        hosting.layoutSubtreeIfNeeded()

        XCTAssertEqual(hosting.fittingSize.width, 440, accuracy: 0.5)
        XCTAssertGreaterThan(hosting.fittingSize.height, 300)
    }

    @MainActor
    func testCupertinoTelemetryViewUsesConcentric10PtOuterMargin() throws {
        let telemetry = TelemetryObservable(
            startServices: false,
            enableAutoLifecycle: false,
            initialSettings: QWServerSettings()
        )
        let modelName = "Qwen3.8-Flash-Next-hq8g32-hq4g64"
        telemetry.connection = .connected
        telemetry.engineState = .serverUp(isLoaded: true)
        telemetry.isDaemonTransitioning = false
        telemetry.dashboard = QWDashboardViewModel.build(
            lifecycle: .init(phase: .loaded, since: Date(), model: modelName, detail: nil),
            modelId: modelName,
            capabilities: [.prefixCache],
            knobs: ["kvMode": .string("q4"), "contextTokens": .int(1_048_576)],
            facts: QWDashboardFacts(
                quantization: "HQ8_G32 (native, GPU)",
                hostLogicalCores: 18,
                gdnPrefillMode: "chunked",
                gpuName: "Apple M5 Max",
                metalFamily: "Metal 4",
                physicalMemoryGB: 128,
                lsaMode: "auto",
                mountedMemoryPack: nil
            )
        )

        let hosting = NSHostingView(
            rootView: CupertinoTelemetryView(telemetry: telemetry).background(TelemetryTheme.surface)
        )
        hosting.appearance = NSAppearance(named: .darkAqua)
        hosting.wantsLayer = true
        let fitting = hosting.fittingSize
        hosting.frame = NSRect(origin: .zero, size: fitting)
        hosting.layoutSubtreeIfNeeded()

        let pixelW = Int((fitting.width * 2).rounded())
        let pixelH = Int((fitting.height * 2).rounded())
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: pixelW,
            pixelsHigh: pixelH,
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0
        ) else {
            throw XCTSkip("Unable to allocate offscreen 2x bitmap representation")
        }
        rep.size = fitting
        hosting.cacheDisplay(in: hosting.bounds, to: rep)

        let scanY = min(pixelH - 1, 140)
        guard let bg = rep.colorAt(x: 2, y: scanY)?.usingColorSpace(.deviceRGB) else {
            throw XCTSkip("Offscreen bitmap background pixel unavailable")
        }
        func differs(_ x: Int) -> Bool {
            guard let c = rep.colorAt(x: x, y: scanY)?.usingColorSpace(.deviceRGB) else { return false }
            return abs(c.redComponent - bg.redComponent) > 0.015
                || abs(c.greenComponent - bg.greenComponent) > 0.015
                || abs(c.blueComponent - bg.blueComponent) > 0.015
        }
        guard let leftEdge = (0..<(pixelW / 2)).first(where: differs),
              let rightEdge = ((pixelW / 2)..<pixelW).reversed().first(where: differs) else {
            throw XCTSkip("Offscreen host did not rasterize card border contrast")
        }
        let leftPt = Double(leftEdge) / 2.0
        let rightPt = Double(pixelW - 1 - rightEdge) / 2.0
        XCTAssertGreaterThanOrEqual(leftPt, 8.5, "Card left gutter must preserve ~10pt concentric outer inset")
        XCTAssertLessThanOrEqual(leftPt, 13.0, "Card left gutter must not double beyond 10pt outer inset")
        XCTAssertGreaterThanOrEqual(rightPt, 8.5, "Card right gutter must preserve ~10pt concentric outer inset")
        XCTAssertLessThanOrEqual(rightPt, 13.0, "Card right gutter must not double beyond 10pt outer inset")
        XCTAssertLessThanOrEqual(abs(leftPt - rightPt), 1.5, "Card left/right outer gutters must stay symmetric")
    }
}
