// QWDecisionsWebUIParityTests.swift — WebUI /ui?tab=decisions workbench, XSS, JSContext & HTTP parity tests.

import Foundation
import JavaScriptCore
import XCTest
import QWCore
import QWDecision
import QWToken
@testable import QWServer

private final class MockWebUIDecisionBackend: QWDecisionExecutionBackend, @unchecked Sendable {
    let decisionConfig: QWDecisionModelConfig
    let maxContextLength: Int = 2048

    init(vocabSize: Int) {
        self.decisionConfig = QWDecisionModelConfig.synthesizedVerbalizer(vocabSize: vocabSize)
    }

    func evaluate(_ request: QWDecisionBackendBatchRequest) async throws -> QWDecisionBackendOutput {
        let outs = request.slots.map { slot in
            var map: [Int: Float] = [:]
            for grp in slot.candidateTokenGroups {
                if let tid = grp.first { map[tid] = Float(tid % 5) }
            }
            return QWDecisionSlotOutput(tokenLogits: map)
        }
        return QWDecisionBackendOutput(slotOutputs: outs, promptTokens: 16, latencyMs: 4.2)
    }
}

final class QWDecisionsWebUIParityTests: XCTestCase {
    private static func repoRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static func readResource(_ rel: String) throws -> String {
        try String(contentsOf: repoRoot().appendingPathComponent("Sources/Apps/QWServer/Resources/\(rel)"), encoding: .utf8)
    }

    func testWebUIResourcesContainDecisionsTabAndStayWithinLimits() throws {
        let html = try Self.readResource("controlbar.html")
        for token in [
            "id=\"nav-tab-decisions\"",
            "id=\"view-decisions\"",
            "id=\"decisions-workbench-host\"",
            "/ui/css/decisions.css",
            "/ui/js/decisions.js",
        ] {
            XCTAssertTrue(html.contains(token), "controlbar.html missing \(token)")
        }

        let files = [
            "controlbar.html",
            "css/decisions.css",
            "js/decisions.js",
            "js/state.js",
            "js/settings.js",
            "js/dashboard.js",
        ]
        for f in files {
            let text = try Self.readResource(f)
            let lines = text.split(separator: "\n", omittingEmptySubsequences: false).count
            XCTAssertLessThanOrEqual(lines, 800, "\(f) exceeds 800-line ceiling (\(lines))")
            XCTAssertFalse(text.contains("https" + "://"), "\(f) must remain 100% offline (no external CDN URLs)")
        }

        let settingsJS = try Self.readResource("js/settings.js")
        for settingKey in [
            "value=\"auto\"",
            "data-setting=\"decisionModelPackPath\"",
            "data-setting=\"decisionComputeUnit\"",
            "data-setting-bool=\"decisionLoadVision\"",
        ] {
            XCTAssertTrue(settingsJS.contains(settingKey), "settings.js missing \(settingKey)")
        }
    }

    func testDecisionsJSContextSwitchTabXSSBarsSpansAndBoxes() throws {
        let ctx = try XCTUnwrap(JSContext())
        var jsError: String?
        ctx.exceptionHandler = { _, exc in jsError = exc?.toString() ?? "JS error" }

        ctx.evaluateScript("""
        var elems = {};
        function mkEl(id) {
          var cls = new Set(id === "view-decisions" ? ["hidden"] : []);
          return {
            id: id,
            innerHTML: "",
            textContent: "",
            classList: {
              add: function(c){ cls.add(c); },
              remove: function(c){ cls.delete(c); },
              toggle: function(c, f){ if (f) cls.add(c); else cls.delete(c); },
              contains: function(c){ return cls.has(c); }
            },
            _cls: cls
          };
        }
        elems["nav-tab-monitor"] = mkEl("nav-tab-monitor");
        elems["nav-tab-decisions"] = mkEl("nav-tab-decisions");
        elems["view-monitor"] = mkEl("view-monitor");
        elems["view-decisions"] = mkEl("view-decisions");
        var replacedUrl = "";
        var window = {
          location: { origin: "http://127.0.0.1:8080", href: "http://127.0.0.1:8080/ui" },
          history: { replaceState: function(_a, _b, u){ replacedUrl = u; } }
        };
        function URL(u) {
          this.searchParams = { set: function(k, v){ replacedUrl = u.split("?")[0] + "?" + k + "=" + v; } };
          this.toString = function(){ return replacedUrl || u; };
        }
        var document = {
          getElementById: function(id){ return elems[id] || null; },
          querySelectorAll: function(){ return [elems["view-monitor"], elems["view-decisions"]]; },
          addEventListener: function(){}
        };
        function loadDiscoveredModels(){}
        function initLogStream(){}
        function closeLogStream(){}
        function loadLiveSettings(){}
        function fetch(){ return Promise.resolve({ ok: false }); }
        function escapeHtml(raw) {
          return String(raw === undefined || raw === null ? "" : raw)
            .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
            .replace(/\\"/g, "&quot;").replace(/'/g, "&#39;");
        }
        """)
        XCTAssertNil(jsError, "JS stub setup error: \(jsError ?? "")")

        let stateJS = try Self.readResource("js/state.js")
        let decJS = try Self.readResource("js/decisions.js")
        ctx.evaluateScript(stateJS)
        ctx.evaluateScript(decJS)
        XCTAssertNil(jsError, "JS load error: \(jsError ?? "")")

        let check = ctx.evaluateScript("""
        (() => {
          switchTab("decisions");
          const tabActive = elems["nav-tab-decisions"].classList.contains("is-active");
          const paneVisible = !elems["view-decisions"].classList.contains("hidden");
          const stateText = "Patient has <script>bad()</script> fever now.";
          const resp = {
            model: "d1-2B<img onerror=alert(1)>",
            backend: "metal4",
            usage: { latencyMs: 5.25, promptTokens: 42 },
            answers: [
              {
                name: "tier<xss>",
                kind: "choice",
                choice: "urgent",
                confidence: 0.88,
                probs: [{ key: "urgent", value: 0.88 }, { key: "routine", value: 0.12 }]
              },
              {
                name: "symptom",
                kind: "read",
                text: "fever",
                confidence: 0.55,
                span: { lowerBound: 35, upperBound: 40 }
              },
              {
                name: "bbox",
                kind: "boxes",
                confidence: 0.91,
                boxes: [{ left: 0.1, top: 0.2, right: 0.5, bottom: 0.8, confidence: 0.91, found: 0.95 }]
              }
            ]
          };
          const imgUri = "data:image/png;base64,iVBORw0KGgo=";
          const html = renderDecisionResultsHTML(resp, stateText, [imgUri]);
          const payload = buildDecisionRequestPayload(decisionWorkbenchState);
          const sn = buildDecisionSnippets(payload, "http://127.0.0.1:8080");
          return { tabActive, paneVisible, replacedUrl, html, curlDecisions: sn.curlDecisions, hermesMcp: sn.hermesMcp };
        })()
        """)?.toDictionary()
        XCTAssertNil(jsError, "JS eval error: \(jsError ?? "")")

        XCTAssertEqual(check?["tabActive"] as? Bool, true)
        XCTAssertEqual(check?["paneVisible"] as? Bool, true)
        XCTAssertEqual(check?["replacedUrl"] as? String, "http://127.0.0.1:8080/ui?tab=decisions")

        let rendered = try XCTUnwrap(check?["html"] as? String)
        XCTAssertFalse(rendered.contains("<script>"), "raw <script> must be escaped")
        XCTAssertTrue(rendered.contains("&lt;script&gt;bad()&lt;/script&gt;"))
        XCTAssertTrue(rendered.contains("Acting"))
        XCTAssertTrue(rendered.contains("Fail-Open"))
        XCTAssertTrue(rendered.contains("dec-prob-row"))
        XCTAssertTrue(rendered.contains("<mark class=\"dec-span-mark\">fever</mark>"))
        XCTAssertTrue(rendered.contains("<svg class=\"dec-box-svg\""))
        XCTAssertTrue(rendered.contains("<rect x=\"100.0\" y=\"200.0\""))

        let curl = try XCTUnwrap(check?["curlDecisions"] as? String)
        let mcp = try XCTUnwrap(check?["hermesMcp"] as? String)
        XCTAssertTrue(curl.contains("/v1/decisions"))
        XCTAssertTrue(mcp.contains("hermes mcp add"))
        XCTAssertFalse(curl.contains("https" + "://"))
        XCTAssertFalse(mcp.contains("https" + "://"))
    }

    func testLiveHTTPServerServesDecisionsAssetsAndDashboardFacts() async throws {
        var vocab: [String: Int] = [:]
        var nextId = 0
        var n = 0
        for b in 0..<256 {
            let printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
            let sc = printable ? Unicode.Scalar(UInt8(b)) : Unicode.Scalar(0x100 + n)!
            if !printable { n += 1 }
            if b == 0xC0 || b == 0xC1 || b >= 0xF5 { continue }
            vocab[String(sc)] = nextId
            nextId += 1
        }
        for w in ["yes", "no"] where vocab[w] == nil {
            vocab[w] = nextId
            nextId += 1
        }
        let backendVocabSize = max(nextId, 4)
        let preRegex = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|\\p{N}{1,3}| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
        let root: [String: Any] = [
            "model": [
                "type": "BPE",
                "vocab": vocab,
                "merges": [] as [[String]],
            ],
            "pre_tokenizer": [
                "type": "Sequence",
                "pretokenizers": [
                    ["type": "Split", "pattern": ["Regex": preRegex], "behavior": "Isolated", "invert": false],
                    ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false],
                ],
            ],
            "normalizer": ["type": "NFC"],
        ]
        let tok = try QWTokenizer(json: JSONSerialization.data(withJSONObject: root))
        let loader: QWEngineLifecycleActor.DecisionRunnerLoader = { packPath, computeUnit, _, _, _ in
            let backend = MockWebUIDecisionBackend(vocabSize: backendVocabSize)
            return QWLoadedDecisionSidecar(
                runner: QWDecisionRunner(tokenizer: tok, backend: backend),
                modelName: "d1-mock-sidecar",
                packPath: packPath,
                backendKind: "metal4",
                computeUnit: computeUnit.rawValue,
                estimatedBytes: 256 << 20
            )
        }

        let port = 19148
        let actor = QWEngineLifecycleActor(
            config: QWEngineLifecycleActor.Config(packPath: "/dummy/primary.qwp", modelName: "primary", host: "127.0.0.1", port: port),
            logger: QWServerLogger(level: .none),
            decisionLoader: loader
        )
        let server = QWHttpServer(actor: actor, host: "127.0.0.1", port: port, logger: QWServerLogger(level: .none))
        try server.start()
        defer { server.stop() }
        try await Task.sleep(nanoseconds: 100_000_000)

        let base = URL(string: "http://127.0.0.1:\(port)")!
        for (path, expectedSubstr) in [
            ("/ui", "nav-tab-decisions"),
            ("/ui/css/decisions.css", ".dec-box-stage"),
            ("/ui/js/decisions.js", "renderDecisionResultsHTML"),
        ] {
            let (data, resp) = try await URLSession.shared.data(from: base.appendingPathComponent(path))
            XCTAssertEqual(try XCTUnwrap(resp as? HTTPURLResponse).statusCode, 200)
            let body = String(decoding: data, as: UTF8.self)
            XCTAssertTrue(body.contains(expectedSubstr), "\(path) missing \(expectedSubstr)")
        }

        setenv("QWG_ALLOW_UNREGISTERED_MODEL_PATHS", "1", 1)
        defer { unsetenv("QWG_ALLOW_UNREGISTERED_MODEL_PATHS") }
        var loadReq = URLRequest(url: base.appendingPathComponent("/v1/decisions/model/load"))
        loadReq.httpMethod = "POST"
        loadReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        loadReq.httpBody = try JSONSerialization.data(withJSONObject: ["pack_path": "/tmp/mock-sidecar", "compute_unit": "metal"])
        let (_, loadResp) = try await URLSession.shared.data(for: loadReq)
        XCTAssertEqual(try XCTUnwrap(loadResp as? HTTPURLResponse).statusCode, 200)

        let (dashData, dashResp) = try await URLSession.shared.data(from: base.appendingPathComponent("/v1/dashboard"))
        XCTAssertEqual(try XCTUnwrap(dashResp as? HTTPURLResponse).statusCode, 200)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let vm = try decoder.decode(QWDashboardViewModel.self, from: dashData)
        XCTAssertEqual(vm.facts?.decisionSidecarLoaded, true)
        XCTAssertEqual(vm.facts?.decisionSidecarModel, "d1-mock-sidecar")
        XCTAssertEqual(vm.facts?.decisionSidecarBackend, "metal4")
        XCTAssertEqual(vm.facts?.decisionSidecarBytes, 256 << 20)
    }
}
