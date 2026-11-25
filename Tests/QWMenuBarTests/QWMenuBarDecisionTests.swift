// QWMenuBarDecisionTests.swift — Tests & 2x Retina snapshots for MenuBar decision sidecar & settings parity.

import AppKit
import Foundation
import QWCore
@testable import QWMenuBarUI
import QWServer
import SwiftUI
import XCTest

final class QWMenuBarDecisionTests: XCTestCase {
    @MainActor
    func testCupertinoTelemetryViewWithLoadedDecisionSidecarFits440PtWidth() throws {
        _ = NSApplication.shared
        var settings = QWServerSettings()
        settings.decisionModelPackPath = "/tmp/d1-2B-sidecar"
        settings.decisionComputeUnit = "metal"
        settings.decisionLoadVision = true

        let telemetry = TelemetryObservable(
            startServices: false,
            enableAutoLifecycle: false,
            initialSettings: settings
        )
        let modelName = "Qwen3.8-27B-Instruct-hq4g32"
        telemetry.connection = .connected
        telemetry.engineState = .serverUp(isLoaded: true)
        telemetry.isDaemonTransitioning = false
        telemetry.dashboard = QWDashboardViewModel.build(
            lifecycle: .init(phase: .loaded, since: Date(), model: modelName, detail: nil),
            modelId: modelName,
            capabilities: [.prefixCache],
            knobs: ["kvMode": .string("q4"), "contextTokens": .int(65_536)],
            facts: QWDashboardFacts(
                quantization: "HQ4_G32 (native, GPU)",
                hostLogicalCores: 18,
                gdnPrefillMode: "chunked",
                gpuName: "Apple M5 Max",
                metalFamily: "Metal 4",
                physicalMemoryGB: 128,
                lsaMode: "auto",
                decisionSidecarLoaded: true,
                decisionSidecarModel: "d1-2B-System1-Decision-hq8g32",
                decisionSidecarBackend: "metal4",
                decisionSidecarComputeUnit: "metal",
                decisionSidecarBytes: 1_920_000_000,
                decisionSidecarTotalRequests: 42,
                decisionSidecarMeanLatencyMs: 6.4,
                decisionSidecarVision: true,
                decisionSidecarArchetype: "causal_verbalizer",
                decisionSidecarPromptTokens: 2048
            )
        )

        let hosting = NSHostingView(
            rootView: CupertinoTelemetryView(telemetry: telemetry).background(TelemetryTheme.surface)
        )
        hosting.appearance = NSAppearance(named: .darkAqua)
        hosting.wantsLayer = true
        let fit = hosting.fittingSize
        hosting.frame = NSRect(origin: .zero, size: fit)
        hosting.layoutSubtreeIfNeeded()

        XCTAssertEqual(fit.width, 440, accuracy: 0.5)
        XCTAssertGreaterThan(fit.height, 360)

        if let snapDir = ProcessInfo.processInfo.environment["QWG_SNAPSHOT_DIR"], !snapDir.isEmpty {
            let dirURL = URL(fileURLWithPath: snapDir, isDirectory: true)
            try FileManager.default.createDirectory(at: dirURL, withIntermediateDirectories: true)
            let window = NSWindow(
                contentRect: NSRect(origin: .zero, size: fit),
                styleMask: [.borderless],
                backing: .buffered,
                defer: false
            )
            window.appearance = NSAppearance(named: .darkAqua)
            window.isReleasedWhenClosed = false
            window.contentView = hosting
            window.layoutIfNeeded()
            hosting.layoutSubtreeIfNeeded()
            try write2xPNG(of: hosting, size: fit, to: dirURL.appendingPathComponent("menubar_decision_sidecar.png"))
            window.close()
        }
    }

    @MainActor
    func testSettingsViewIncludesDecisionSidecarAndAutoReasoningEffort() throws {
        _ = NSApplication.shared
        var settings = QWServerSettings()
        settings.reasoningEffort = "auto"
        settings.decisionModelPackPath = "/tmp/d1-omni-600M.aimodel"
        settings.decisionComputeUnit = "ane"
        settings.decisionLoadVision = false
        settings.decisionDefaultScoreReadout = "mode"
        settings.decisionDefaultTruncation = "tail"
        settings.decisionAlwaysEmitBox = true

        let telemetry = TelemetryObservable(
            startServices: false,
            enableAutoLifecycle: false,
            initialSettings: settings
        )

        for (tab, slug) in [(SettingsView.SettingsTab.model, "settings_model_decision.png"),
                            (SettingsView.SettingsTab.sampling, "settings_sampling_auto.png")] {
            let size = NSSize(width: 780, height: 920)
            let host = NSHostingView(
                rootView: SettingsView(telemetry: telemetry, initialTab: tab)
                    .frame(width: size.width, height: size.height)
                    .background(TelemetryTheme.surface)
            )
            host.appearance = NSAppearance(named: .darkAqua)
            host.wantsLayer = true
            let window = NSWindow(
                contentRect: NSRect(origin: .zero, size: size),
                styleMask: [.titled],
                backing: .buffered,
                defer: false
            )
            window.appearance = NSAppearance(named: .darkAqua)
            window.isReleasedWhenClosed = false
            window.contentView = host
            window.layoutIfNeeded()
            host.layoutSubtreeIfNeeded()
            XCTAssertGreaterThan(host.fittingSize.width, 400)

            if let snapDir = ProcessInfo.processInfo.environment["QWG_SNAPSHOT_DIR"], !snapDir.isEmpty {
                let dirURL = URL(fileURLWithPath: snapDir, isDirectory: true)
                try FileManager.default.createDirectory(at: dirURL, withIntermediateDirectories: true)
                try write2xPNG(of: host, size: size, to: dirURL.appendingPathComponent(slug))
            }
            window.close()
        }

        let menuBarSource = try String(contentsOf: URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/Apps/QWMenuBar/SettingsView.swift"), encoding: .utf8)
        XCTAssertTrue(menuBarSource.contains("decisionModelPackPath"))
        XCTAssertTrue(menuBarSource.contains("decisionComputeUnit"))
        XCTAssertTrue(menuBarSource.contains("decisionLoadVision"))
        XCTAssertTrue(menuBarSource.contains("decisionDefaultScoreReadout"))
        XCTAssertTrue(menuBarSource.contains("decisionDefaultTruncation"))
        XCTAssertTrue(menuBarSource.contains("decisionAlwaysEmitBox"))
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
