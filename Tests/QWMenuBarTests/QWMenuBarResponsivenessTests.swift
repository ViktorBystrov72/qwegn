// QWMenuBarResponsivenessTests.swift — Tests for off-main metadata caching,
// removal of spurious local-registry polling, and LogViewerWindow filter memoization.

import Foundation
import QWCore
@testable import QWMenuBarUI
@testable import QWServer
import SwiftUI
import XCTest

final class QWMenuBarResponsivenessTests: XCTestCase {
    @MainActor
    func testFetchDirectOrHttpIgnoresLocalTelemetryRegistry() {
        let localReg = QWEngineTelemetryRegistry.shared
        let priorPhase = localReg.snapshot().throughput.phase
        localReg.beginPrefill(promptTokens: 64)
        localReg.recordCacheLookup(hitTier: "warm_ram", matchedTokens: 32)
        defer { localReg.resetToIdle(resetCacheCounters: true) }

        var settings = QWServerSettings()
        settings.port = 0
        let telemetry = TelemetryObservable(
            startServices: false, enableAutoLifecycle: false, initialSettings: settings
        )
        telemetry.engineState = .stopped
        telemetry.fetchDirectOrHttp()

        XCTAssertEqual(
            telemetry.engineState, .stopped,
            "fetchDirectOrHttp() must never flip stopped child-process state using in-process registry snapshot"
        )
        let _ = priorPhase
    }

    @MainActor
    func testTelemetryAsyncMetadataCachingPopulatesOffMainActor() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-menubar-cache-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("assets"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let configObj: [String: Any] = [
            "model_type": "qwen3",
            "hidden_size": 1024,
            "intermediate_size": 2816,
            "vocab_size": 32000,
            "num_hidden_layers": 16,
            "max_position_embeddings": 32768,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "eos_token_id": 248044
        ]
        try JSONSerialization.data(withJSONObject: configObj).write(
            to: dir.appendingPathComponent("assets/config.json")
        )

        let telemetry = TelemetryObservable(
            startServices: false, enableAutoLifecycle: false, initialSettings: QWServerSettings()
        )
        XCTAssertNil(
            telemetry.cachedPrimaryDescriptor(packPath: dir.path),
            "cache miss must return nil synchronously without blocking @MainActor on disk parsing"
        )

        var loadedDesc: QWModelDescriptor?
        for _ in 0..<60 {
            try await Task.sleep(nanoseconds: 25_000_000)
            if let hit = telemetry.cachedPrimaryDescriptor(packPath: dir.path) {
                loadedDesc = hit
                break
            }
        }
        XCTAssertNotNil(loadedDesc, "background metadata task must populate cachedPrimaryDescriptors asynchronously")
        XCTAssertEqual(loadedDesc?.hidden, 1024)
    }

    @MainActor
    func testLogFilterCacheMemoizesAndAppendsIncrementally() {
        let cache = QWLogFilterCache()
        func entry(_ id: UInt64, line: String) -> QWLogEntry {
            QWLogLineParser.parse(id: id, line: line)
        }

        let base = [
            entry(1, line: "[2026-11-19 10:00:00] INFO  Booting engine"),
            entry(2, line: "[2026-11-19 10:00:01] ERROR Out of budget"),
            entry(3, line: "[2026-11-19 10:00:02] WARN  Slow prefill")
        ]

        let pass1 = cache.entries(from: base, filter: .all, query: "")
        XCTAssertEqual(pass1.count, 3)
        XCTAssertEqual(cache.fullRecomputeCount, 1)

        // Identical inputs within same body render pass hit cache in O(1).
        for _ in 0..<4 {
            let hit = cache.entries(from: base, filter: .all, query: "")
            XCTAssertEqual(hit.count, 3)
        }
        XCTAssertEqual(cache.fullRecomputeCount, 1)

        // Append-only update increments count cleanly without full recompute.
        let appended = base + [
            entry(4, line: "[2026-11-19 10:00:03] ERROR Timeout hit")
        ]
        let pass2 = cache.entries(from: appended, filter: .all, query: "")
        XCTAssertEqual(pass2.count, 4)
        XCTAssertEqual(cache.fullRecomputeCount, 1)

        // Changing filter invalidates and recomputes once.
        let errs1 = cache.entries(from: appended, filter: .error, query: "")
        let errs2 = cache.entries(from: appended, filter: .error, query: "")
        XCTAssertEqual(errs1.count, 2)
        XCTAssertEqual(errs2.count, 2)
        XCTAssertEqual(cache.fullRecomputeCount, 2)

        // Appending matching + non-matching rows with active filter filters only tail delta.
        let appendedWithWarn = appended + [
            entry(5, line: "[2026-11-19 10:00:04] WARN  Transient warning"),
            entry(6, line: "[2026-11-19 10:00:05] ERROR Second failure")
        ]
        let errs3 = cache.entries(from: appendedWithWarn, filter: .error, query: "")
        XCTAssertEqual(errs3.map(\.id), [2, 4, 6])
        XCTAssertEqual(cache.fullRecomputeCount, 2)
    }
}
