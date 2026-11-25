// QWChatStreamingUXTests.swift — Verify WebUI chat free-scroll streaming, collapsed reasoning, and quips.

import Foundation
import JavaScriptCore
import XCTest

final class QWChatStreamingUXTests: XCTestCase {
    private func read(_ relativePath: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(contentsOf: root.appendingPathComponent(relativePath), encoding: .utf8)
    }

    private func makeJSContextWithDOMStubs() throws -> JSContext {
        let ctx = try XCTUnwrap(JSContext())
        var jsError: String?
        ctx.exceptionHandler = { _, exc in
            jsError = exc?.toString() ?? "unknown JS error"
        }
        ctx.evaluateScript("""
        var _dom = {};
        var _docListeners = {};
        function _makeEl(id) {
          return {
            id: id, hidden: false, innerHTML: "", textContent: "", value: "", className: "",
            scrollTop: 0, scrollHeight: 0, clientHeight: 0, _listeners: {},
            setAttribute: function(k, v) { this[k] = v; },
            getAttribute: function(k) { return this[k] || ""; },
            addEventListener: function(t, fn) { this._listeners[t] = fn; },
            querySelector: function() { return null; }
          };
        }
        var document = {
          getElementById: function(id) { if (!_dom[id]) _dom[id] = _makeEl(id); return _dom[id]; },
          addEventListener: function(t, fn) { _docListeners[t] = fn; }
        };
        var window = { location: { origin: "http://127.0.0.1:8080" } };
        function escapeHtml(s) {
          return String(s == null ? "" : s)
            .replace(/&/g, "&amp;").replace(/</g, "&lt;")
            .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
        }
        function formatTokenCount(n) { return String(n); }
        var smokeProbeState = { status: "empty", mode: "rendered", content: "", reasoning: "", rawText: "", isJsonFallback: false };
        function showToast() {}
        function setInterval() { return 42; }
        function clearInterval() {}
        """)
        XCTAssertNil(jsError)

        for rel in [
            "Sources/Apps/QWServer/Resources/js/markdown.js",
            "Sources/Apps/QWServer/Resources/js/chat-presets.js",
            "Sources/Apps/QWServer/Resources/js/chat-media.js",
            "Sources/Apps/QWServer/Resources/js/chat.js",
        ] {
            ctx.evaluateScript(try read(rel))
            XCTAssertNil(jsError, "JS error evaluating \(rel): \(jsError ?? "")")
        }
        return ctx
    }

    func testReasoningQuipsCatalogIsRichNonRepeatingAndHonest() throws {
        let ctx = try makeJSContextWithDOMStubs()
        let result = ctx.evaluateScript("""
        (() => {
          const forbidden = ["tok/s", "TTFT:", "Apple Silicon", "https://", "Misses", "CACHE MISS"];
          const list = CHAT_REASONING_QUIPS;
          const invalid = list.filter((q) => typeof q !== "string" || !q.trim() || q.trim() !== q);
          const badHit = list.filter((q) => forbidden.some((f) => q.includes(f)));
          let prev = -1;
          let repeated = false;
          for (let i = 0; i < 50; i += 1) {
            const pick = pickRandomReasoningQuip(prev, () => ((i * 7 + 3) % 100) / 100);
            if (pick.index === prev) repeated = true;
            prev = pick.index;
          }
          return { count: list.length, invalidCount: invalid.length, badCount: badHit.length, repeated };
        })()
        """)?.toDictionary()

        XCTAssertGreaterThanOrEqual(result?["count"] as? Int ?? 0, 20)
        XCTAssertEqual(result?["invalidCount"] as? Int, 0)
        XCTAssertEqual(result?["badCount"] as? Int, 0)
        XCTAssertEqual(result?["repeated"] as? Bool, false)
    }

    func testReasoningDisclosureIsCollapsedByDefaultShowsDimmedQuipAndHonorsUserToggle() throws {
        let ctx = try makeJSContextWithDOMStubs()
        let check = ctx.evaluateScript("""
        (() => {
          const session = createBlankChatSession("Reasoning UX", "server-default", "");
          chatState.sessions = [session];
          chatState.activeSessionId = session.id;
          const msg = {
            id: "msg-r1", role: "assistant", content: "",
            reasoning: "Deriving invariant", toolCalls: [], usage: null,
            isStreaming: true, isRaw: false, createdAt: 1
          };
          session.messages.push(msg);

          renderChatTranscript();
          const transcript = document.getElementById("chat-transcript");
          const collapsedStreamingHtml = transcript.innerHTML;
          const activeQuipText = chatState.reasoningQuip.text;

          handleChatThinkingToggle({
            target: { tagName: "DETAILS", open: true, getAttribute: () => "msg-r1" }
          });
          msg.reasoning += " and testing boundary";
          renderChatTranscript();
          const userOpenedStreamingHtml = transcript.innerHTML;

          msg.isStreaming = false;
          msg.content = "Final answer.";
          renderChatTranscript();
          const finishedHtml = transcript.innerHTML;

          smokeProbeState = {
            status: "done", mode: "rendered", content: "",
            reasoning: "Probe reasoning trace", rawText: "[reasoning]", isJsonFallback: false
          };
          renderSmokeProbeOutput();
          const probeHtml = document.getElementById("probe-output").innerHTML;

          return { collapsedStreamingHtml, activeQuipText, userOpenedStreamingHtml, finishedHtml, probeHtml };
        })()
        """)?.toDictionary()

        let collapsed = try XCTUnwrap(check?["collapsedStreamingHtml"] as? String)
        let quip = try XCTUnwrap(check?["activeQuipText"] as? String)
        let openedStreaming = try XCTUnwrap(check?["userOpenedStreamingHtml"] as? String)
        let finished = try XCTUnwrap(check?["finishedHtml"] as? String)
        let probe = try XCTUnwrap(check?["probeHtml"] as? String)

        XCTAssertTrue(collapsed.contains(#"<details class="md-thinking is-reasoning" data-thinking-msg-id="msg-r1">"#))
        XCTAssertFalse(collapsed.contains(#"data-thinking-msg-id="msg-r1" open"#))
        XCTAssertTrue(collapsed.contains("md-thinking-quip"))
        XCTAssertFalse(quip.isEmpty)
        XCTAssertTrue(collapsed.contains(quip))

        XCTAssertTrue(openedStreaming.contains(#"data-thinking-msg-id="msg-r1" open"#))
        XCTAssertTrue(finished.contains(#"data-thinking-msg-id="msg-r1" open"#))
        XCTAssertFalse(finished.contains("md-thinking-quip"))
        XCTAssertFalse(finished.contains("is-reasoning"))

        XCTAssertTrue(probe.contains(#"<details class="md-thinking">"#))
        XCTAssertFalse(probe.contains(#"<details class="md-thinking" open"#))
    }

    func testChatTranscriptScrollAutoFollowDisengagesOnUserScrollUpAndResumesAtBottom() throws {
        let ctx = try makeJSContextWithDOMStubs()
        let scrollCheck = ctx.evaluateScript("""
        (() => {
          const nearExact = isTranscriptNearBottom({ scrollHeight: 2000, scrollTop: 1500, clientHeight: 500 }, 48);
          const nearWithin = isTranscriptNearBottom({ scrollHeight: 2000, scrollTop: 1460, clientHeight: 500 }, 48);
          const nearOutside = isTranscriptNearBottom({ scrollHeight: 2000, scrollTop: 1400, clientHeight: 500 }, 48);

          const transcript = document.getElementById("chat-transcript");
          const jumpBtn = document.getElementById("chat-scroll-bottom-btn");
          transcript.scrollHeight = 2000;
          transcript.clientHeight = 500;
          transcript.scrollTop = 1500;
          scrollChatTranscriptToBottom(true);
          const btnHiddenInitially = jumpBtn.hidden;

          handleChatTranscriptWheel({ deltaY: -40 });
          const afterWheelFollow = chatState.autoFollowScroll;
          transcript.scrollTop = 400;
          scrollChatTranscriptToBottom(false);
          const preservedAfterScrollFalse = transcript.scrollTop;
          const btnShownAfterScrollUp = !jumpBtn.hidden;

          scrollChatTranscriptToBottom(true);
          const snapFollow = chatState.autoFollowScroll;
          const snappedTop = transcript.scrollTop;
          const btnHiddenAfterSnap = jumpBtn.hidden;

          transcript.scrollTop = 900;
          handleChatTranscriptScroll();
          const disengagedInsideIgnoreWindow = !chatState.autoFollowScroll;

          chatState.ignoreScrollUntil = 0;
          transcript.scrollTop = 1470;
          handleChatTranscriptScroll();
          const reengagedAtBottom = chatState.autoFollowScroll;
          const btnHiddenAtBottom = jumpBtn.hidden;

          return {
            nearExact, nearWithin, nearOutside,
            btnHiddenInitially, afterWheelFollow, preservedAfterScrollFalse, btnShownAfterScrollUp,
            snapFollow, snappedTop, btnHiddenAfterSnap,
            disengagedInsideIgnoreWindow, reengagedAtBottom, btnHiddenAtBottom
          };
        })()
        """)?.toDictionary()

        XCTAssertEqual(scrollCheck?["nearExact"] as? Bool, true)
        XCTAssertEqual(scrollCheck?["nearWithin"] as? Bool, true)
        XCTAssertEqual(scrollCheck?["nearOutside"] as? Bool, false)
        XCTAssertEqual(scrollCheck?["btnHiddenInitially"] as? Bool, true)
        XCTAssertEqual(scrollCheck?["afterWheelFollow"] as? Bool, false)
        XCTAssertEqual(scrollCheck?["preservedAfterScrollFalse"] as? Int, 400)
        XCTAssertEqual(scrollCheck?["btnShownAfterScrollUp"] as? Bool, true)
        XCTAssertEqual(scrollCheck?["snapFollow"] as? Bool, true)
        XCTAssertEqual(scrollCheck?["snappedTop"] as? Int, 2000)
        XCTAssertEqual(scrollCheck?["btnHiddenAfterSnap"] as? Bool, true)
        XCTAssertEqual(scrollCheck?["disengagedInsideIgnoreWindow"] as? Bool, true)
        XCTAssertEqual(scrollCheck?["reengagedAtBottom"] as? Bool, true)
        XCTAssertEqual(scrollCheck?["btnHiddenAtBottom"] as? Bool, true)
    }

    func testControlbarMarkupAndStylesSupportFloatingScrollButtonAndDimmedQuips() throws {
        let html = try read("Sources/Apps/QWServer/Resources/controlbar.html")
        let panels = try read("Sources/Apps/QWServer/Resources/css/panels.css")
        let chatCss = try read("Sources/Apps/QWServer/Resources/css/chat.css")

        XCTAssertTrue(html.contains("class=\"chat-transcript-wrap\""))
        XCTAssertTrue(html.contains("id=\"chat-scroll-bottom-btn\""))
        XCTAssertTrue(panels.contains(".md-thinking-quip"))
        XCTAssertTrue(panels.contains("@keyframes qwg-quip-pulse"))
        XCTAssertTrue(chatCss.contains(".chat-scroll-bottom-btn"))

        for rel in [
            "Sources/Apps/QWServer/Resources/controlbar.html",
            "Sources/Apps/QWServer/Resources/css/base.css",
            "Sources/Apps/QWServer/Resources/css/panels.css",
            "Sources/Apps/QWServer/Resources/css/chat.css",
            "Sources/Apps/QWServer/Resources/js/state.js",
            "Sources/Apps/QWServer/Resources/js/dashboard.js",
            "Sources/Apps/QWServer/Resources/js/markdown.js",
            "Sources/Apps/QWServer/Resources/js/chat-presets.js",
            "Sources/Apps/QWServer/Resources/js/chat-media.js",
            "Sources/Apps/QWServer/Resources/js/chat.js",
            "Sources/Apps/QWServer/Resources/js/actions.js",
            "Sources/Apps/QWServer/Resources/js/settings.js",
        ] {
            let lineCount = try read(rel).components(separatedBy: "\n").count
            XCTAssertLessThanOrEqual(lineCount, 800, "\(rel) exceeded 800 lines")
        }
    }
}
