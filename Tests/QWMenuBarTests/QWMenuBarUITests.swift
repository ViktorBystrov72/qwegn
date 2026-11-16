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

    @MainActor
    func testThermalCardsAndPerformanceSettingsRenderWithoutOverflowAt2x() throws {
        let telemetry = TelemetryObservable(
            startServices: false,
            enableAutoLifecycle: false,
            initialSettings: QWServerSettings(coolingPolicy: "proactive")
        )
        let now = Date()
        func card(_ id: String, _ title: String, _ val: String, _ unit: String?, _ num: Double?) -> QWDashboardCard {
            QWDashboardCard(
                id: id, title: title, value: val, unit: unit, number: num,
                provenance: .measured, sampledAt: now, age: 0.2, isStale: false, reason: nil, series: []
            )
        }
        telemetry.connection = .connected
        telemetry.engineState = .serverUp(isLoaded: true)
        telemetry.isDaemonTransitioning = false
        telemetry.dashboard = QWDashboardViewModel(
            sampledAt: now,
            lifecycle: .init(phase: .loaded, since: now, model: "Qwen3.8-27B-q4", detail: nil),
            modelId: "Qwen3.8-27B-q4",
            capabilities: ["prefixCache"],
            generation: .decode,
            facts: QWDashboardFacts(
                quantization: "HQ4_G32 (native, GPU)",
                hostLogicalCores: 18,
                gdnPrefillMode: "chunked",
                gpuName: "Apple M5 Max",
                metalFamily: "Metal 4",
                physicalMemoryGB: 128,
                lsaMode: "auto",
                coolingPolicy: "proactive",
                fanControlStatus: "System Auto (SMC write requires root)"
            ),
            cards: [
                card("throughput.display_decode_tokens_per_sec", "Decode", "92.4", "tok/s", 92.4),
                card("throughput.display_prefill_tokens_per_sec", "Prefill", "610.0", "tok/s", 610.0),
                card("gpu.busy_ratio", "Engine GPU", "94.1", "%", 94.1),
                card("gpu.device", "Device GPU", "95.0", "%", 95.0),
                card("hardware.cpu_percent", "CPU", "18.5", "%", 18.5),
                card("hardware.memory_percent", "Memory", "64.2", "%", 64.2),
                card("hardware.memory_pressure_level", "Memory pressure", "Normal", nil, nil),
                card("hardware.gpu_temp_celsius", "GPU Temp", "68.4", "°C", 68.4),
                card("hardware.fan_rpm", "Fans", "2450", "rpm", 2450),
                card("hardware.system_power_watts", "System Power", "78.2", "W", 78.2),
                card("hardware.thermal_state", "Thermal State", "Nominal", nil, nil),
                card("memory.arena_reserved_bytes", "Metal arena", "24.10", "GiB", 24.10),
                card("memory.resident_bytes", "Resident weights", "22.50", "GiB", 22.50),
                card("context.active_tokens", "Last turn tokens", "4096", "tok", 4096),
            ],
            actions: [],
            knobs: ["kvMode": .string("q4"), "contextTokens": .int(131_072)],
            knobSources: [:]
        )

        let popoverHost = NSHostingView(
            rootView: CupertinoTelemetryView(telemetry: telemetry).background(TelemetryTheme.surface)
        )
        popoverHost.appearance = NSAppearance(named: .darkAqua)
        popoverHost.wantsLayer = true
        let popoverSize = popoverHost.fittingSize
        popoverHost.frame = NSRect(origin: .zero, size: popoverSize)
        popoverHost.layoutSubtreeIfNeeded()
        XCTAssertEqual(popoverSize.width, 440, accuracy: 0.5)
        XCTAssertGreaterThan(popoverSize.height, 340)

        let settingsHost = NSHostingView(
            rootView: SettingsView(telemetry: telemetry, initialTab: .performance)
                .frame(width: 780, height: 960)
                .background(TelemetryTheme.surface)
        )
        settingsHost.appearance = NSAppearance(named: .darkAqua)
        settingsHost.wantsLayer = true
        let settingsWindow = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 780, height: 960),
            styleMask: [.titled],
            backing: .buffered,
            defer: false
        )
        settingsWindow.appearance = NSAppearance(named: .darkAqua)
        settingsWindow.isReleasedWhenClosed = false
        settingsWindow.contentView = settingsHost
        settingsWindow.layoutIfNeeded()
        settingsHost.layoutSubtreeIfNeeded()
        XCTAssertEqual(settingsHost.bounds.width, 780, accuracy: 2.0)

        if let dir = ProcessInfo.processInfo.environment["QWG_SNAPSHOT_DIR"], !dir.isEmpty {
            let dirURL = URL(fileURLWithPath: dir, isDirectory: true)
            try FileManager.default.createDirectory(at: dirURL, withIntermediateDirectories: true)
            try write2xPNG(of: popoverHost, size: popoverSize, to: dirURL.appendingPathComponent("menubar_popover_2x.png"))
            try write2xPNG(of: settingsHost, size: NSSize(width: 780, height: 960), to: dirURL.appendingPathComponent("menubar_settings_perf_2x.png"))
        }
    }

    @MainActor
    private func write2xPNG(of view: NSView, size: NSSize, to url: URL) throws {
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: Int((size.width * 2).rounded()),
            pixelsHigh: Int((size.height * 2).rounded()),
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0
        ) else { return }
        rep.size = size
        view.cacheDisplay(in: view.bounds, to: rep)
        if let png = rep.representation(using: .png, properties: [:]) {
            try png.write(to: url)
        }
    }
}
