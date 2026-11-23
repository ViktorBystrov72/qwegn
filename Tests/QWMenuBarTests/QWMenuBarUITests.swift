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
    func testSelectedModelSupportsDFlash2AndDrafterSelection() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-menubar-dflash-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        func writeJSON(_ obj: [String: Any], to url: URL) throws {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]).write(to: url)
        }

        let fnLayerTypes = (0..<48).map { ($0 + 1) % 4 == 0 ? "full_attention" : "linear_attention" }
        let fnConfig: [String: Any] = [
            "model_type": "qwen4_exp",
            "text_config": [
                "model_type": "qwen4_exp_text",
                "hidden_size": 2560, "vocab_size": 248320, "num_hidden_layers": 48,
                "layer_types": fnLayerTypes, "full_attention_interval": 4,
                "rms_norm_eps": 1e-6, "max_position_embeddings": 262144,
                "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
                "partial_rotary_factor": 0.25,
                "rope_parameters": ["rope_type": "default", "rope_theta": 1e7, "mrope_section": [11, 11, 10]],
                "output_gate_type": "sigmoid",
                "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
                "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
                "shared_expert_intermediate_size": 640, "hc_count": 4, "hc_lowrank": 320,
                "mtp_num_hidden_layers": 1, "eos_token_id": 248044,
            ] as [String: Any],
        ]
        let q27LayerTypes = (0..<64).map { ($0 + 1) % 4 == 0 ? "full_attention" : "linear_attention" }
        let q27Config: [String: Any] = [
            "model_type": "qwen3_5",
            "text_config": [
                "model_type": "qwen3_5_text",
                "hidden_size": 5120, "intermediate_size": 17408, "vocab_size": 248320, "num_hidden_layers": 64,
                "layer_types": q27LayerTypes, "full_attention_interval": 4,
                "rms_norm_eps": 1e-6, "max_position_embeddings": 262144,
                "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
                "partial_rotary_factor": 0.25,
                "rope_parameters": ["rope_type": "default", "rope_theta": 1e7, "mrope_section": [11, 11, 10]],
                "attn_output_gate": true,
                "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
                "mtp_num_hidden_layers": 1, "eos_token_id": 248044,
            ] as [String: Any],
        ]

        let fnDir = root.appendingPathComponent("Qwen3.8-Flash-Next-hq8g32-hq4g64")
        let q27Dir = root.appendingPathComponent("Qwen3.8-27B-hq4g32")
        let dfn = root.appendingPathComponent("Qwen3.8-Flash-Next-DFlash2-hq4g32")
        let d27 = root.appendingPathComponent("Qwen3.8-27B-DFlash2-hq4g32")
        let s27 = root.appendingPathComponent("Qwen3.8-27B-DSpine-b16-hq4g32")

        try writeJSON(fnConfig, to: fnDir.appendingPathComponent("assets/config.json"))
        try writeJSON(q27Config, to: q27Dir.appendingPathComponent("assets/config.json"))
        try writeJSON([
            "model_type": "qwen3", "hidden_size": 2560, "vocab_size": 248320, "num_hidden_layers": 5,
            "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
            "dflash_config": ["block_size": 8, "target_layer_ids": [5, 15, 25, 35, 45]],
        ], to: dfn.appendingPathComponent("config.json"))
        try writeJSON([
            "model_type": "qwen3", "hidden_size": 5120, "vocab_size": 248320, "num_hidden_layers": 5,
            "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
            "dflash_config": ["block_size": 8, "target_layer_ids": [5, 19, 33, 47, 61]],
        ], to: d27.appendingPathComponent("assets/config.json"))
        try writeJSON([
            "model_type": "qwen3", "hidden_size": 5120, "vocab_size": 248320, "num_hidden_layers": 5,
            "block_size": 16, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
            "dflash_config": [
                "target_layer_ids": [1, 16, 31, 46, 61],
                "selected_prefix_code_bridge": ["code_size": 1024, "rank": 512, "eps": 1e-6],
            ],
        ], to: s27.appendingPathComponent("assets/config.json"))

        let fnDesc = try XCTUnwrap(QWServerSettings.loadPrimaryDescriptor(packPath: fnDir.path))
        let q27Desc = try XCTUnwrap(QWServerSettings.loadPrimaryDescriptor(packPath: q27Dir.path))
        let drafters = QWServerSettings.scanDrafterModelsDirectory(in: root.path)

        var fnSettings = QWServerSettings()
        fnSettings.modelName = "Qwen3.8-Flash-Next-hq8g32-hq4g64"
        fnSettings.modelPackPath = fnDir.path
        fnSettings.dflashDraftPackPath = d27.path // stale incompatible path from 27B

        XCTAssertTrue(SettingsView.supportsDFlash2(
            for: fnSettings, descriptor: fnDesc, discoveredModels: [], discoveredDrafters: drafters
        ))
        XCTAssertTrue(SettingsView.supportsDFlash2(
            for: QWServerSettings(modelName: "CapModel"),
            descriptor: nil,
            discoveredModels: [DiscoveredModel(name: "CapModel", packPath: "/tmp/c", capabilities: ["dflash"])],
            discoveredDrafters: []
        ))
        XCTAssertFalse(SettingsView.supportsDFlash2(
            for: QWServerSettings(modelName: "Llama"),
            descriptor: nil,
            discoveredModels: [DiscoveredModel(name: "Llama", packPath: "/tmp/l", family: "llama-dense", capabilities: [])],
            discoveredDrafters: []
        ))

        let selectableFN = SettingsView.selectableDrafterModels(
            for: fnSettings, descriptor: fnDesc, discoveredDrafters: drafters
        )
        XCTAssertEqual(Set(selectableFN.map(\.name)), ["Qwen3.8-27B-DFlash2-hq4g32", "Qwen3.8-Flash-Next-DFlash2-hq4g32"])
        fnSettings.dflashDraftPackPath = ""
        let cleanSelectableFN = SettingsView.selectableDrafterModels(
            for: fnSettings, descriptor: fnDesc, discoveredDrafters: drafters
        )
        XCTAssertEqual(cleanSelectableFN.map(\.name), ["Qwen3.8-Flash-Next-DFlash2-hq4g32"])

        SettingsView.stageExternalBackendSelection(
            "dflash2", settings: &fnSettings, targetDescriptor: fnDesc, discoveredDrafters: drafters
        )
        XCTAssertEqual(fnSettings.dflashDraftPackPath, dfn.path)
        XCTAssertEqual(fnSettings.mtpDrafts, 7)

        var q27Settings = QWServerSettings()
        q27Settings.modelName = "Qwen3.8-27B-hq4g32"
        q27Settings.modelPackPath = q27Dir.path
        SettingsView.stageExternalBackendSelection(
            "dspine", settings: &q27Settings, targetDescriptor: q27Desc, discoveredDrafters: drafters
        )
        XCTAssertEqual(q27Settings.dflashDraftPackPath, s27.path)
        XCTAssertEqual(q27Settings.mtpDrafts, 15)
    }

    @MainActor
    func testSpeculativeSettingsTabRendersAndOmitsFalseDenseTag() throws {
        let speculativePath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/Apps/QWMenuBar/SettingsView+Speculative.swift")
        let speculativeSource = try String(contentsOf: speculativePath, encoding: .utf8)
        XCTAssertFalse(speculativeSource.contains("(Dense)"))

        let telemetry = TelemetryObservable(
            startServices: false,
            enableAutoLifecycle: false,
            initialSettings: QWServerSettings(
                modelName: "Qwen3.8-Flash-Next",
                mtpDrafts: 7,
                mtpBackend: "dflash2",
                dflashDraftPackPath: "/tmp/Qwen3.8-Flash-Next-DFlash2-hq4g32"
            )
        )
        let host = NSHostingView(
            rootView: SettingsView(telemetry: telemetry, initialTab: .speculative)
                .frame(width: 780, height: 900)
                .background(TelemetryTheme.surface)
        )
        host.appearance = NSAppearance(named: .darkAqua)
        host.wantsLayer = true
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 780, height: 900),
            styleMask: [.titled],
            backing: .buffered,
            defer: false
        )
        window.appearance = NSAppearance(named: .darkAqua)
        window.isReleasedWhenClosed = false
        window.contentView = host
        window.layoutIfNeeded()
        host.layoutSubtreeIfNeeded()
        XCTAssertEqual(host.bounds.width, 780, accuracy: 2.0)

        if let dir = ProcessInfo.processInfo.environment["QWG_SNAPSHOT_DIR"], !dir.isEmpty {
            let dirURL = URL(fileURLWithPath: dir, isDirectory: true)
            try FileManager.default.createDirectory(at: dirURL, withIntermediateDirectories: true)
            try write2xPNG(of: host, size: NSSize(width: 780, height: 900), to: dirURL.appendingPathComponent("menubar_settings_speculative_2x.png"))
        }
    }

    @MainActor
    func testAllSevenSettingsTabsRenderCleanlyAndGateBackends() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-menubar-settings-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let fnDisc = DiscoveredModel(
            name: "Qwen3.8-Flash-Next-hq8g32-hq4g64",
            packPath: "/tmp/fn",
            family: QWModelDescriptor.Family.qwen38FlashNext.rawValue,
            capabilities: ["dflash", "moe", "mtp", "ple", "qsaIndexer", "vision"]
        )
        let dfnURL = root.appendingPathComponent("Qwen3.8-Flash-Next-DFlash2-shvq4g32/assets/config.json")
        try FileManager.default.createDirectory(at: dfnURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let drafterObj: [String: Any] = [
            "model_type": "qwen3",
            "hidden_size": 2560,
            "vocab_size": 248320,
            "num_hidden_layers": 5,
            "num_target_layers": 48,
            "dflash_config": ["target_layer_ids": [5, 15, 25, 35, 45], "block_size": 8],
        ]
        try JSONSerialization.data(withJSONObject: drafterObj, options: [.sortedKeys]).write(to: dfnURL)
        let drafters = QWServerSettings.scanDrafterModelsDirectory(in: root.path)
        var fnSettings = QWServerSettings(
            modelPackPath: "/tmp/fn",
            modelName: fnDisc.name,
            mtpDrafts: 7,
            mtpBackend: "dflash2",
            dflashDraftPackPath: dfnURL.deletingLastPathComponent().deletingLastPathComponent().path
        )
        XCTAssertTrue(SettingsView.supportsExternalDrafter(
            backend: "dflash2", for: fnSettings, descriptor: nil, discoveredModels: [fnDisc], discoveredDrafters: drafters
        ))
        XCTAssertFalse(SettingsView.supportsExternalDrafter(
            backend: "dspine", for: fnSettings, descriptor: nil, discoveredModels: [fnDisc], discoveredDrafters: drafters
        ))

        let snapDir = ProcessInfo.processInfo.environment["QWG_SNAPSHOT_DIR"]
        func renderTab(_ tab: SettingsView.SettingsTab, settings: QWServerSettings, size: NSSize, slug: String) throws {
            let telemetry = TelemetryObservable(startServices: false, enableAutoLifecycle: false, initialSettings: settings)
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
            XCTAssertEqual(host.bounds.width, size.width, accuracy: 2.0)
            if let snapDir, !snapDir.isEmpty {
                let dirURL = URL(fileURLWithPath: snapDir, isDirectory: true)
                try FileManager.default.createDirectory(at: dirURL, withIntermediateDirectories: true)
                try write2xPNG(of: host, size: size, to: dirURL.appendingPathComponent("\(slug)_2x.png"))
            }
        }

        let stdSize = NSSize(width: 740, height: 640)
        for tab in SettingsView.SettingsTab.allCases {
            try renderTab(tab, settings: fnSettings, size: stdSize, slug: "settings_\(tab.id.lowercased().replacingOccurrences(of: " ", with: "_"))")
        }
        try renderTab(.general, settings: fnSettings, size: NSSize(width: 740, height: 860), slug: "settings_general_tall")
        try renderTab(.speculative, settings: fnSettings, size: NSSize(width: 660, height: 640), slug: "settings_speculative_narrow")
        try renderTab(.performance, settings: fnSettings, size: NSSize(width: 740, height: 960), slug: "settings_performance_tall")
        try renderTab(.performance, settings: fnSettings, size: NSSize(width: 660, height: 960), slug: "settings_performance_narrow")
        fnSettings.mtpBackend = "variational"
        fnSettings.mtpDrafts = 3
        try renderTab(.speculative, settings: fnSettings, size: stdSize, slug: "settings_speculative_variational")

        let inkSettings = QWServerSettings(
            modelPackPath: "/tmp/Inkling-2-Mini-QWP-Fast",
            modelName: "Inkling-2-Mini-QWP-Fast",
            mtpDrafts: 15,
            mtpBackend: "variational"
        )
        XCTAssertEqual(QWModelDescriptor.qwen38_27b.maxSpeculativeDrafts, 15)
        XCTAssertEqual(QWModelFamilyRegistry.shared.spec(forModelType: "inkling2_loop")?.maxSpeculativeDrafts, 15)
        XCTAssertEqual(inkSettings.mtpDrafts, 15)
        try renderTab(.speculative, settings: inkSettings, size: stdSize, slug: "settings_speculative_15drafts")
    }

    @MainActor
    func testMenuBarStatusItemFontHasZeroPixelWidthDriftAcrossRegimes() throws {
        let _ = NSApplication.shared
        func makeCard(_ id: String, value: String?, unit: String?, number: Double? = nil) -> QWDashboardCard {
            QWDashboardCard(
                id: id, title: id, value: value, unit: unit, number: number,
                provenance: value == nil ? .unavailable : .measured,
                sampledAt: value == nil ? nil : Date(),
                age: value == nil ? nil : 0.1,
                isStale: false, reason: value == nil ? "not measured" : nil, series: []
            )
        }
        func makeDashboard(
            decode: Double?,
            prefill: Double?,
            prefillText: String? = nil,
            arena: Double?,
            ctx: Double?,
            temp: Double?
        ) -> QWDashboardViewModel {
            let prefillVal = prefillText ?? prefill.map { String(format: "%.1f", $0) }
            let prefillUnit: String? = prefillText == nil ? "tok/s" : nil
            return QWDashboardViewModel(
                sampledAt: Date(),
                lifecycle: .init(phase: .loaded, since: Date(), model: "m", detail: nil),
                modelId: "m",
                capabilities: [],
                generation: .decode,
                facts: nil,
                cards: [
                    makeCard("throughput.display_decode_tokens_per_sec", value: decode.map { String(format: "%.1f", $0) }, unit: "tok/s", number: decode),
                    makeCard("throughput.display_prefill_tokens_per_sec", value: prefillVal, unit: prefillUnit, number: prefill),
                    makeCard("memory.arena_reserved_bytes", value: arena.map { String(format: "%.2f", $0) }, unit: "GiB", number: arena),
                    makeCard("context.active_tokens", value: ctx.map { String(Int($0)) }, unit: "tok", number: ctx),
                    makeCard("hardware.gpu_temp_celsius", value: temp.map { String(format: "%.1f", $0) }, unit: "°C", number: temp),
                ],
                actions: [],
                knobs: ["contextTokens": .int(262_144)],
                knobSources: [:]
            )
        }

        let regimes: [QWDashboardViewModel] = [
            makeDashboard(decode: 9.5, prefill: 9.5, arena: 8.40, ctx: 38, temp: 48.2),
            makeDashboard(decode: 91.9, prefill: 93.3, arena: 25.01, ctx: 512, temp: 54.2),
            makeDashboard(decode: 248.3, prefill: 2450.0, arena: 108.42, ctx: 4096, temp: 102.0),
            makeDashboard(decode: 1240.0, prefill: 12500.0, arena: 1024.0, ctx: 128_000, temp: 115.0),
            makeDashboard(decode: 84.0, prefill: nil, prefillText: "from cache", arena: 25.01, ctx: 10_240, temp: 64.0),
            makeDashboard(decode: nil, prefill: nil, arena: nil, ctx: nil, temp: nil),
        ]

        let statusImage = QGProcessorLogoView.statusBarImage(pointSize: 18, statusDotColor: .systemBlue)
        let metricCombos: [Set<QWMenuBarMetric>] = [
            [.decodeSpeed, .prefillSpeed],
            Set(QWMenuBarMetric.allCases),
        ]

        var allButtonsForSnapshot: [NSButton] = []
        for metrics in metricCombos {
            var baselineAttrW: CGFloat?
            var baselineBtnW: CGFloat?
            for dash in regimes {
                let title = try XCTUnwrap(QWMenuBarStatusFormatter.statusTitle(
                    dashboard: dash, isConnected: true, enabledMetrics: metrics, hideSpeedsWhenIdle: false
                ))
                let attributed = QWMenuBarStatusStyle.attributedTitle(for: title)
                let attrW = attributed.size().width

                let button = NSButton()
                button.isBordered = false
                button.appearance = NSAppearance(named: .darkAqua)
                button.image = statusImage
                button.imagePosition = .imageLeading
                button.alignment = .left
                button.font = QWMenuBarStatusStyle.font
                button.attributedTitle = attributed
                button.sizeToFit()
                let btnW = button.fittingSize.width

                if let refAttr = baselineAttrW, let refBtn = baselineBtnW {
                    XCTAssertEqual(attrW, refAttr, accuracy: 0.01, "Attributed width drifted for '\(title)'")
                    XCTAssertEqual(btnW, refBtn, accuracy: 0.01, "NSButton width drifted for '\(title)'")
                } else {
                    baselineAttrW = attrW
                    baselineBtnW = btnW
                }
                if metrics.count == QWMenuBarMetric.allCases.count {
                    allButtonsForSnapshot.append(button)
                }
            }
        }

        if let dir = ProcessInfo.processInfo.environment["QWG_SNAPSHOT_DIR"], !dir.isEmpty {
            let dirURL = URL(fileURLWithPath: dir, isDirectory: true)
            try FileManager.default.createDirectory(at: dirURL, withIntermediateDirectories: true)
            let sheetWidth = max(460, ceil((allButtonsForSnapshot.map { $0.fittingSize.width }.max() ?? 432) + 24))
            let rowHeight: CGFloat = 24
            let sheetHeight = CGFloat(allButtonsForSnapshot.count) * rowHeight + 24
            let container = NSView(frame: NSRect(x: 0, y: 0, width: sheetWidth, height: sheetHeight))
            container.appearance = NSAppearance(named: .darkAqua)
            container.wantsLayer = true
            container.layer?.backgroundColor = NSColor(srgbRed: 0.067, green: 0.075, blue: 0.094, alpha: 1.0).cgColor
            for (index, button) in allButtonsForSnapshot.enumerated() {
                let y = sheetHeight - 12 - CGFloat(index + 1) * rowHeight
                button.frame = NSRect(x: 12, y: y, width: button.fittingSize.width, height: rowHeight)
                container.addSubview(button)
            }
            container.layoutSubtreeIfNeeded()
            try write2xPNG(of: container, size: container.bounds.size, to: dirURL.appendingPathComponent("menubar_status_alignment_2x.png"))
        }
    }

    func testShortcutRouterHandlesAllHotkeysCapsLockAndNonLatinLayouts() {
        XCTAssertEqual(
            QWMenuBarShortcutRouter.resolve(
                charactersIgnoringModifiers: ",", keyCode: 43, modifierFlags: [.command],
                isEditingText: false, isPopoverKey: false
            ),
            .openSettings
        )
        XCTAssertEqual(
            QWMenuBarShortcutRouter.resolve(
                charactersIgnoringModifiers: "R", keyCode: 15, modifierFlags: [.command, .capsLock],
                isEditingText: false, isPopoverKey: false
            ),
            .restartServer
        )
        XCTAssertEqual(
            QWMenuBarShortcutRouter.resolve(
                charactersIgnoringModifiers: ",", keyCode: 43, modifierFlags: [.command, .function],
                isEditingText: false, isPopoverKey: false
            ),
            .openSettings
        )
        XCTAssertEqual(
            QWMenuBarShortcutRouter.resolve(
                charactersIgnoringModifiers: "б", keyCode: 43, modifierFlags: [.command],
                isEditingText: false, isPopoverKey: false
            ),
            .openSettings
        )
        XCTAssertEqual(
            QWMenuBarShortcutRouter.resolve(
                charactersIgnoringModifiers: "ц", keyCode: 13, modifierFlags: [.command],
                isEditingText: false, isPopoverKey: false
            ),
            .closeFrontWindow
        )
        XCTAssertNil(
            QWMenuBarShortcutRouter.resolve(
                charactersIgnoringModifiers: "a", keyCode: 12, modifierFlags: [.command],
                isEditingText: true, isPopoverKey: false
            )
        )
        XCTAssertNil(
            QWMenuBarShortcutRouter.resolve(
                charactersIgnoringModifiers: "r", keyCode: 15, modifierFlags: [.command],
                isEditingText: true, isPopoverKey: false
            )
        )
        XCTAssertEqual(
            QWMenuBarShortcutRouter.resolve(
                charactersIgnoringModifiers: ",", keyCode: 43, modifierFlags: [.command],
                isEditingText: true, isPopoverKey: false
            ),
            .openSettings
        )
        XCTAssertNil(
            QWMenuBarShortcutRouter.resolve(
                charactersIgnoringModifiers: ",", keyCode: 43, modifierFlags: [.command, .shift],
                isEditingText: false, isPopoverKey: false
            )
        )
        XCTAssertEqual(
            QWMenuBarShortcutRouter.resolve(
                charactersIgnoringModifiers: "\u{1B}", keyCode: 53, modifierFlags: [.capsLock],
                isEditingText: false, isPopoverKey: true
            ),
            .dismissPopover
        )
        XCTAssertNil(
            QWMenuBarShortcutRouter.resolve(
                charactersIgnoringModifiers: "\u{1B}", keyCode: 53, modifierFlags: [],
                isEditingText: false, isPopoverKey: false
            )
        )
    }

    @MainActor
    func testAppDelegateOpensSingleSettingsWindowAndInstallsMainMenu() throws {
        _ = NSApplication.shared
        let telemetry = TelemetryObservable(startServices: false, enableAutoLifecycle: false, initialSettings: QWServerSettings())
        let delegate = AppDelegate(telemetry: telemetry)

        let repeatComma = try XCTUnwrap(
            NSEvent.keyEvent(
                with: .keyDown, location: .zero, modifierFlags: [.command],
                timestamp: 0, windowNumber: 0, context: nil,
                characters: ",", charactersIgnoringModifiers: ",", isARepeat: true, keyCode: 43
            )
        )
        XCTAssertNil(delegate.handleShortcutEvent(repeatComma))
        XCTAssertNil(delegate.currentSettingsWindow)

        delegate.openSettingsWindow()
        let first = try XCTUnwrap(delegate.currentSettingsWindow)
        delegate.openSettingsWindow()
        let second = try XCTUnwrap(delegate.currentSettingsWindow)
        XCTAssertTrue(first === second)
        XCTAssertEqual(first.title, "Qwegn Settings")
        XCTAssertEqual(first.identifier?.rawValue, "qwg.settings")
        XCTAssertTrue(first.contentViewController is NSHostingController<SettingsView>)
        first.close()

        let menu = delegate.buildMainMenu()
        let titles = menu.items.compactMap { $0.submenu?.items.map(\.title) }.joined()
        XCTAssertEqual(menu.items.count, 3)
        XCTAssertTrue(titles.contains("Settings…"))
        XCTAssertTrue(titles.contains("Copy"))
        XCTAssertTrue(titles.contains("Close"))
        XCTAssertTrue(titles.contains("Quit Qwegn"))
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
