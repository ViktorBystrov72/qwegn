// QWCliGuiWebParityTests.swift — Verifies CLI, GUI (QWMenuBar), and WebUI (/ui) knob parity.
import JavaScriptCore
import XCTest
@testable import QWCore
@testable import QWServer

final class QWCliGuiWebParityTests: XCTestCase {
    private static func repoRootURL() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static let cliKnobToSettingKeys: [String: [String]] = [
        "contextTokens": ["maxContext"],
        "maxTokens": ["maxTokens"],
        "temperature": ["temperature"],
        "topP": ["topP"],
        "topK": ["topK"],
        "repetitionPenalty": ["repetitionPenalty"],
        "seed": ["seed"],
        "enableThinking": ["enableThinking"],
        "reasoningEffort": ["reasoningEffort"],
        "systemPrompt": ["systemPrompt"],
        "mtpDrafts": ["mtpDrafts"],
        "mtpBackend": ["mtpBackend"],
        "mtpRelaxedMargin": ["mtpRelaxedMargin"],
        "mtpRelaxedMinProb": ["mtpRelaxedMinProb"],
        "mtpDraftStrategy": ["mtpDraftStrategy"],
        "mtpQuadraticSeeding": ["mtpQuadraticSeeding"],
        "prefillChunkRows": ["prefillChunkSize"],
        "maxConcurrentSessions": ["maxConcurrentSessions"],
        "prefixCacheEnabled": ["warmRAMLimitGB"],
        "warmCacheLimitGiB": ["warmRAMLimitGB"],
        "ssdCacheLimitGiB": ["ssdCacheEnabled", "ssdCacheLimitGB"],
        "cacheDir": ["cacheDir"],
        "evictPrefillScratch": ["evictPrefillScratch"],
        "lsaMode": ["lsaMode"],
        "lsaBudget": ["lsaBudget"],
        "lsaClusterBlock": ["lsaClusterBlock"],
        "lsaSidecarPath": ["lsaSidecarPath"],
        "lsaCrossLoopReuse": ["lsaCrossLoopReuse"],
        "lazyLoad": ["lazyLoad"],
        "loadVision": ["loadVision"],
        "logLevel": ["logLevel"],
        "kvMode": ["kvMode"],
        "gdnPrefill": ["gdnPrefill"],
        "mtpAdaptiveDepth": ["mtpAdaptiveDepth"],
        "mtpEntropyThreshold": ["mtpEntropyThreshold"],
        "dflashDraftPackPath": ["dflashDraftPackPath"],
        "dflashSelectorTopK": ["dflashSelectorTopK"],
        "decisionModelPackPath": ["decisionModelPackPath"],
        "decisionComputeUnit": ["decisionComputeUnit"],
        "decisionLoadVision": ["decisionLoadVision"],
        "decisionDefaultScoreReadout": ["decisionDefaultScoreReadout"],
        "decisionDefaultTruncation": ["decisionDefaultTruncation"],
        "decisionAlwaysEmitBox": ["decisionAlwaysEmitBox"],
    ]

    func testAllCLIKnobsAndEngineTuningAreExposedInGUIAndWebUI() throws {
        let root = Self.repoRootURL()
        let guiSources = try [
            "Sources/Apps/QWMenuBar/SettingsView.swift",
            "Sources/Apps/QWMenuBar/SettingsView+Sections.swift",
            "Sources/Apps/QWMenuBar/SettingsView+Speculative.swift",
            "Sources/Apps/QWMenuBar/SettingsView+Performance.swift",
        ].map {
            try String(contentsOf: root.appendingPathComponent($0), encoding: .utf8)
        }.joined(separator: "\n")

        let settingsJS = try String(
            contentsOf: root.appendingPathComponent("Sources/Apps/QWServer/Resources/js/settings.js"),
            encoding: .utf8
        )

        let cliKnobs = QWKnobRegistry.all.filter { $0.surfaces.contains(.cli) }
        XCTAssertGreaterThanOrEqual(cliKnobs.count, 42)

        for knob in cliKnobs {
            if QWEngineTuning.knobIds.contains(knob.id) {
                XCTAssertTrue(
                    guiSources.contains("\"\(knob.id)\""),
                    "GUI SettingsView missing engineTuning control for CLI knob '\(knob.id)'"
                )
                XCTAssertTrue(
                    settingsJS.contains("=\"\(knob.id)\""),
                    "WebUI settings.js missing engineTuning control for CLI knob '\(knob.id)'"
                )
                continue
            }
            let expectedKeys = try XCTUnwrap(
                Self.cliKnobToSettingKeys[knob.id],
                "Unmapped CLI knob '\(knob.id)' in parity test"
            )
            for key in expectedKeys {
                XCTAssertTrue(
                    guiSources.contains("draftSettings.\(key)"),
                    "GUI SettingsView missing control for CLI knob '\(knob.id)' -> '\(key)'"
                )
                XCTAssertTrue(
                    settingsJS.contains("=\"\(key)\""),
                    "WebUI settings.js missing control for CLI knob '\(knob.id)' -> '\(key)'"
                )
            }
        }

        // Verify "none" log level option is offered in both GUI and WebUI.
        XCTAssertTrue(guiSources.contains(".tag(\"none\")"))
        XCTAssertTrue(settingsJS.contains("value=\"none\""))
    }

    func testWebUISettingsFormRendersAndRoundTripsEngineTuningAndLSA() throws {
        let root = Self.repoRootURL()
        let ctx = try XCTUnwrap(JSContext())
        var jsError: String?
        ctx.exceptionHandler = { _, ex in jsError = ex?.toString() }

        ctx.evaluateScript("""
        var elements = {};
        function stubEl() {
          return {
            hidden: false, innerHTML: "", textContent: "", value: "", disabled: false,
            oninput: null, onchange: null, style: {},
            classList: { add: function(){}, remove: function(){} },
            setAttribute: function(){}, querySelector: function(){ return null; },
            querySelectorAll: function(){ return []; }
          };
        }
        var API_BASE = "";
        var loadedSettingsObj = null;
        var discoveredModelsList = [];
        var discoveredDraftersList = [];
        var publishedModel = null;
        function escapeHtml(s) {
          return String(s === null || s === undefined ? "" : s)
            .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
        }
        function showToast() {}
        function fetchDashboard() {}
        var queryHandlers = {};
        var document = {
          getElementById: function(id) { if (!elements[id]) elements[id] = stubEl(); return elements[id]; },
          querySelectorAll: function(sel) { return queryHandlers[sel] || []; },
          querySelector: function() { return null; },
          addEventListener: function() {}
        };
        """)

        let settingsJS = try String(
            contentsOf: root.appendingPathComponent("Sources/Apps/QWServer/Resources/js/settings.js"),
            encoding: .utf8
        )
        ctx.evaluateScript(settingsJS)
        XCTAssertNil(jsError, "settings.js failed to evaluate: \(jsError ?? "")")

        var seedSettings = QWServerSettings()
        seedSettings.lazyLoad = false
        seedSettings.lsaMode = "sidecar"
        seedSettings.lsaBudget = 16384
        seedSettings.lsaClusterBlock = 4096
        seedSettings.lsaSidecarPath = "/tmp/model.qlsa"
        seedSettings.lsaCrossLoopReuse = true
        seedSettings.gdnPrefill = "regres"
        seedSettings.evictPrefillScratch = "always"
        seedSettings.prefillChunkSize = 2048
        seedSettings.maxConcurrentSessions = 4
        seedSettings.ssdCacheEnabled = true
        seedSettings.ssdCacheLimitGB = 24.0
        seedSettings.cacheDir = "/tmp/qwg-cache"
        seedSettings.logLevel = "none"
        seedSettings.mtpBackend = "variational"
        seedSettings.mtpAdaptiveDepth = true
        seedSettings.mtpEntropyThreshold = 0.45
        seedSettings.setEngineTuning("memoryPolicy", .string("safe"))
        seedSettings.setEngineTuning("osReserveGiB", .double(12))
        seedSettings.setEngineTuning("moeSpecBudget", .int(4))
        seedSettings.setEngineTuning("streamNgrams", .bool(true))
        seedSettings.setEngineTuning("coldEvents", .bool(false))

        let seedJSON = try XCTUnwrap(String(data: JSONEncoder().encode(seedSettings), encoding: .utf8))
        ctx.setObject(seedJSON, forKeyedSubscript: "seedSettingsJson" as NSString)
        let rendered = ctx.evaluateScript("""
        (function() {
          loadedSettingsObj = JSON.parse(seedSettingsJson);
          renderSettingsForm(loadedSettingsObj);
          return elements["settings-form-host"].innerHTML;
        })();
        """)?.toString() ?? ""
        XCTAssertNil(jsError)

        for expectedAttr in [
            "data-setting-bool=\"lazyLoad\"",
            "data-setting=\"lsaMode\"",
            "data-setting-int=\"lsaBudget\"",
            "data-setting-int=\"lsaClusterBlock\"",
            "data-setting=\"lsaSidecarPath\"",
            "data-setting-bool=\"lsaCrossLoopReuse\"",
            "data-setting=\"gdnPrefill\"",
            "data-setting=\"evictPrefillScratch\"",
            "data-setting-int=\"prefillChunkSize\"",
            "data-setting-int=\"maxConcurrentSessions\"",
            "data-setting-bool=\"ssdCacheEnabled\"",
            "data-setting-float=\"ssdCacheLimitGB\"",
            "data-setting=\"cacheDir\"",
            "data-setting-float=\"mtpRelaxedMargin\"",
            "data-setting-float=\"mtpRelaxedMinProb\"",
            "data-setting-bool=\"mtpQuadraticSeeding\"",
            "data-setting-bool=\"mtpAdaptiveDepth\"",
            "data-setting-float=\"mtpEntropyThreshold\"",
            "data-tuning-string=\"memoryPolicy\"",
            "data-tuning-double=\"osReserveGiB\"",
            "data-tuning-int=\"residentExpertsPerLayer\"",
            "data-tuning-int=\"moeSpecBudget\"",
            "data-tuning-bool=\"streamNgrams\"",
            "data-tuning-bool=\"expertCache\"",
            "data-tuning-bool=\"coldEvents\"",
            "data-tuning-bool=\"coldEventOverlap\"",
            "data-tuning-bool=\"prefillSpanMap\"",
        ] {
            XCTAssertTrue(rendered.contains(expectedAttr), "Rendered WebUI settings HTML missing \(expectedAttr)")
        }

        let roundTrippedJSON = ctx.evaluateScript("""
        (function() {
          function attrEl(attrs, value, readOnly) {
            return {
              value: String(value),
              readOnly: Boolean(readOnly),
              hasAttribute: function(k) { return Object.prototype.hasOwnProperty.call(attrs, k); },
              getAttribute: function(k) { return attrs[k]; }
            };
          }
          queryHandlers["[data-setting]"] = [
            attrEl({"data-setting": "lsaSidecarPath"}, "/tmp/custom.qlsa"),
            attrEl({"data-setting": "gdnPrefill"}, "naive"),
            attrEl({"data-setting": "logLevel"}, "none")
          ];
          queryHandlers["[data-setting-int]"] = [
            attrEl({"data-setting-int": "lsaBudget"}, "8192"),
            attrEl({"data-setting-int": "prefillChunkSize"}, "1024")
          ];
          queryHandlers["[data-setting-float]"] = [
            attrEl({"data-setting-float": "mtpEntropyThreshold"}, "0.55")
          ];
          queryHandlers["[data-setting-bool]"] = [
            attrEl({"data-setting-bool": "lazyLoad"}, "false"),
            attrEl({"data-setting-bool": "lsaCrossLoopReuse"}, "true"),
            attrEl({"data-setting-bool": "ssdCacheEnabled"}, "true")
          ];
          queryHandlers["[data-tuning-string]"] = [
            attrEl({"data-tuning-string": "memoryPolicy", "data-tuning-default": "maxResident"}, "safe")
          ];
          queryHandlers["[data-tuning-double]"] = [
            attrEl({"data-tuning-double": "osReserveGiB", "data-tuning-default": "8"}, "16")
          ];
          queryHandlers["[data-tuning-int]"] = [
            attrEl({"data-tuning-int": "moeSpecBudget", "data-tuning-default": "0"}, "6")
          ];
          queryHandlers["[data-tuning-bool]"] = [
            attrEl({"data-tuning-bool": "streamNgrams", "data-tuning-default": "false"}, "true")
          ];
          return JSON.stringify(collectSettingsDraftPayload());
        })();
        """)?.toString() ?? ""
        XCTAssertNil(jsError)

        let decoded = try JSONDecoder().decode(QWServerSettings.self, from: Data(roundTrippedJSON.utf8))
        XCTAssertEqual(decoded.lsaSidecarPath, "/tmp/custom.qlsa")
        XCTAssertEqual(decoded.gdnPrefill, "naive")
        XCTAssertEqual(decoded.logLevel, "none")
        XCTAssertEqual(decoded.lsaBudget, 8192)
        XCTAssertEqual(decoded.prefillChunkSize, 1024)
        XCTAssertEqual(decoded.mtpEntropyThreshold, 0.55, accuracy: 1e-6)
        XCTAssertFalse(decoded.lazyLoad)
        XCTAssertTrue(decoded.lsaCrossLoopReuse)
        XCTAssertTrue(decoded.ssdCacheEnabled)
        XCTAssertEqual(decoded.resolvedEngineTuning.memoryPolicy, .safe)
        XCTAssertEqual(decoded.resolvedEngineTuning.osReserveGiB, 16)
        XCTAssertEqual(decoded.resolvedEngineTuning.moeSpecBudget, 6)
        XCTAssertTrue(decoded.resolvedEngineTuning.streamNgrams)

        let cfg = QWEngineLifecycleActor.Config(
            packPath: "/tmp/model.qwp",
            modelName: "test",
            lsaConfig: QWLSAConfig(mode: .sidecar, budget: 8192, clusterBlock: 512, crossLoopReuse: true)
        )
        XCTAssertEqual(cfg.knobValues["lsaCrossLoopReuse"], .bool(true))
    }
}
