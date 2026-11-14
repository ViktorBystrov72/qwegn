import XCTest
@testable import QWServer
@testable import QWEngine
import Foundation

final class QWServerMetricsTests: XCTestCase {
    /// A tier this build does not name is still a hit. Counting it as
    /// a miss would assert that the cache did not serve a prompt it
    /// served, and would drop the tokens it reused.
    func testAnUnnamedTierIsAHitAndNeverAMiss() {
        let registry = QWEngineTelemetryRegistry()
        registry.recordCacheLookup(hitTier: "cold_ssd", matchedTokens: 64)
        registry.recordCacheLookup(hitTier: "a_tier_this_build_does_not_name", matchedTokens: 32)
        registry.recordCacheLookup(hitTier: nil, matchedTokens: 0)
        let snap = registry.snapshot()
        XCTAssertEqual(snap.cache.lookupsTotal, 3)
        XCTAssertEqual(snap.cache.hitsSSDTotal, 1)
        XCTAssertEqual(snap.cache.hitsUnclassifiedTotal, 1)
        XCTAssertEqual(snap.cache.missesTotal, 1)
        // the reused tokens of both hits are counted
        XCTAssertEqual(snap.cache.tokensReusedTotal, 96)
        // the server knows its own counter, so it reports the tier
        XCTAssertTrue(registry.formatPrometheus()
            .contains("qwegine_cache_hits_total{tier=\"unclassified\"} 1"))
    }

    func testTelemetryRegistryAndPrometheusOutput() {
        let registry = QWEngineTelemetryRegistry()

        // 1. Initial State
        let snap0 = registry.snapshot()
        XCTAssertEqual(snap0.cache.lookupsTotal, 0)
        XCTAssertEqual(snap0.cache.hitsWarmTotal, 0)
        XCTAssertEqual(snap0.cache.hitsSSDTotal, 0)
        XCTAssertEqual(snap0.cache.missesTotal, 0)

        // 2. Record lookups
        registry.recordCacheLookup(hitTier: "warm_ram", matchedTokens: 32)
        registry.recordCacheLookup(hitTier: "cold_ssd", matchedTokens: 64)
        registry.recordCacheLookup(hitTier: nil, matchedTokens: 0)

        registry.updateCacheUsage(warmBytes: 1024 * 1024 * 50, warmLimit: 1024 * 1024 * 100,
                                  ssdBytes: 1024 * 1024 * 200, ssdLimit: 1024 * 1024 * 500)

        // 3. Record generation
        registry.endRequest(tokensIn: 100, tokensOut: 50, prefillSec: 0.1, decodeSec: 0.5, ttftSec: 0.08)
        registry.updateContext(active: 150, capacity: 4096)
        registry.updateArena(reserved: 1024 * 1024 * 1024)

        let snap1 = registry.snapshot()
        XCTAssertEqual(snap1.cache.lookupsTotal, 3)
        XCTAssertEqual(snap1.cache.hitsWarmTotal, 1)
        XCTAssertEqual(snap1.cache.hitsSSDTotal, 1)
        XCTAssertEqual(snap1.cache.missesTotal, 1)
        XCTAssertEqual(snap1.cache.tokensReusedTotal, 96)
        XCTAssertEqual(snap1.cache.evictionsWarmTotal, 0)
        XCTAssertEqual(snap1.cache.evictionsSSDTotal, 0)
        XCTAssertEqual(snap1.throughput.tokensInTotal, 100)
        XCTAssertEqual(snap1.throughput.tokensOutTotal, 50)
        XCTAssertEqual(snap1.context.activeTokens, 150)
        XCTAssertEqual(snap1.context.maxTokens, 4096)

        // 4. Test Prometheus text rendering
        let promText = registry.formatPrometheus()
        XCTAssertTrue(promText.contains("qwegine_cache_hits_total{tier=\"warm_ram\"} 1"))
        XCTAssertTrue(promText.contains("qwegine_cache_hits_total{tier=\"cold_ssd\"} 1"))
        XCTAssertTrue(promText.contains("qwegine_cache_hits_total{tier=\"miss\"} 1"))
        XCTAssertTrue(promText.contains("qwegine_cache_tokens_reused_total 96"))
        XCTAssertTrue(promText.contains("qwegine_tokens_total{direction=\"in\"} 100"))
        XCTAssertTrue(promText.contains("qwegine_tokens_total{direction=\"out\"} 50"))
        XCTAssertTrue(promText.contains("qwegine_context_tokens{state=\"active\"} 150"))
        XCTAssertTrue(promText.contains("qwegine_context_tokens{state=\"capacity\"} 4096"))
    }

    func testHttpServerMetricsEndpoints() async throws {
        let port = 19097
        let config = QWEngineLifecycleActor.Config(
            packPath: "/dummy/model.qwp",
            modelName: "test-metrics-model",
            host: "127.0.0.1",
            port: port
        )
        let logger = QWServerLogger(level: .none)
        let actor = QWEngineLifecycleActor(config: config, logger: logger)
        let server = QWHttpServer(actor: actor, host: "127.0.0.1", port: port, logger: logger)

        try server.start()
        defer { server.stop() }

        // Allow listener to bind
        try await Task.sleep(nanoseconds: 100_000_000)

        let session = URLSession.shared
        let baseURL = URL(string: "http://127.0.0.1:\(port)")!

        // 1. Test GET /metrics (Prometheus format)
        let promURL = baseURL.appendingPathComponent("metrics")
        let (promData, promResp) = try await session.data(from: promURL)
        let httpProm = promResp as! HTTPURLResponse
        XCTAssertEqual(httpProm.statusCode, 200)
        XCTAssertTrue((httpProm.value(forHTTPHeaderField: "Content-Type") ?? "").contains("text/plain"))
        let promString = String(data: promData, encoding: .utf8) ?? ""
        XCTAssertTrue(promString.contains("qwegine_cache_hits_total"))
        XCTAssertTrue(promString.contains("qwegine_tokens_total"))

        // 2. Test GET /v1/metrics (JSON format)
        let jsonURL = baseURL.appendingPathComponent("v1/metrics")
        let (jsonData, jsonResp) = try await session.data(from: jsonURL)
        let httpJson = jsonResp as! HTTPURLResponse
        XCTAssertEqual(httpJson.statusCode, 200)
        XCTAssertTrue((httpJson.value(forHTTPHeaderField: "Content-Type") ?? "").contains("application/json"))
        let parsed = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any]
        XCTAssertNotNil(parsed?["cache"])
        XCTAssertNotNil(parsed?["throughput"])
        XCTAssertNotNil(parsed?["latency"])
        XCTAssertNotNil(parsed?["context"])
        XCTAssertNotNil(parsed?["memory"])
        XCTAssertNotNil(parsed?["hardware"])
    }

    func testEngineInferencePhaseAndLiveRates() {
        let registry = QWEngineTelemetryRegistry()

        // 1. Initial state must be idle with no live rates: a rate that was
        // not measured is absent, not zero.
        var snap = registry.snapshot()
        XCTAssertEqual(snap.throughput.phase, "idle")
        XCTAssertNil(snap.throughput.livePrefillTokensPerSec)
        XCTAssertNil(snap.throughput.liveDecodeTokensPerSec)

        // 2. Prefill phase
        registry.beginPrefill(promptTokens: 100)
        snap = registry.snapshot()
        XCTAssertEqual(snap.throughput.phase, "prefill")

        // 3. Decode phase transition. The prefill's own duration is not
        // known here: the time to the first token is not the prefill's
        // duration, and a fully cached prompt has no prefill phase at all.
        registry.beginDecode(promptTokens: 100)
        snap = registry.snapshot()
        XCTAssertEqual(snap.throughput.phase, "decode")
        XCTAssertNil(snap.throughput.lastPrefillTokensPerSec)
        XCTAssertEqual(snap.provenance[QWTelemetryMetric.lastPrefillTokensPerSec.rawValue], .unavailable)

        // 4. Decode token generation
        registry.recordDecodeToken()
        registry.recordDecodeToken()

        // 5. End request: the engine's prefill duration is the only one that
        // may stand for the prefill phase (100 tok / 0.2 s = 500 tok/s).
        registry.endRequest(tokensIn: 100, tokensOut: 10, prefillSec: 0.2, decodeSec: 0.3, ttftSec: 0.2)
        snap = registry.snapshot()
        XCTAssertEqual(snap.throughput.phase, "idle")
        XCTAssertNil(snap.throughput.livePrefillTokensPerSec)
        XCTAssertNil(snap.throughput.liveDecodeTokensPerSec)
        XCTAssertEqual(snap.throughput.lastPrefillTokensPerSec ?? 0, 500.0)
        XCTAssertGreaterThan(snap.throughput.lastDecodeTokensPerSec ?? 0, 0.0)

        // 6. Reset to idle preserves the last recorded rates while the live
        // rates stay absent.
        registry.resetToIdle()
        snap = registry.snapshot()
        XCTAssertEqual(snap.throughput.phase, "idle")
        XCTAssertNil(snap.throughput.livePrefillTokensPerSec)
        XCTAssertNil(snap.throughput.liveDecodeTokensPerSec)
        XCTAssertEqual(snap.throughput.lastPrefillTokensPerSec ?? 0, 500.0)
        XCTAssertNotNil(snap.hardware)
    }

    func testUpdatePrefillProgressAdvancesAndClampsDuringPrefill() {
        let registry = QWEngineTelemetryRegistry()

        registry.beginPrefill(promptTokens: 65_536)
        var snap = registry.snapshot()
        XCTAssertEqual(snap.throughput.phase, "prefill")
        XCTAssertEqual(snap.throughput.prefillProgressTokens, 0)
        XCTAssertEqual(snap.throughput.prefillTotalTokens, 65_536)

        // Prefix-cache partial hit adjusts computed denominator and advances incrementally.
        registry.updatePrefillProgress(completedTokens: 16_384, totalTokens: 32_768)
        snap = registry.snapshot()
        XCTAssertEqual(snap.throughput.prefillProgressTokens, 16_384)
        XCTAssertEqual(snap.throughput.prefillTotalTokens, 32_768)

        // Overshoot and negative values clamp cleanly within [0, totalTokens].
        registry.updatePrefillProgress(completedTokens: 50_000, totalTokens: 32_768)
        snap = registry.snapshot()
        XCTAssertEqual(snap.throughput.prefillProgressTokens, 32_768)
        XCTAssertEqual(snap.throughput.prefillTotalTokens, 32_768)

        // Outside prefill phase, progress updates are ignored.
        registry.beginDecode(promptTokens: 65_536)
        registry.updatePrefillProgress(completedTokens: 100, totalTokens: 200)
        snap = registry.snapshot()
        XCTAssertEqual(snap.throughput.phase, "decode")
        XCTAssertEqual(snap.throughput.prefillProgressTokens, 65_536)
        XCTAssertEqual(snap.throughput.prefillTotalTokens, 65_536)
    }

    func testActiveDecodeAdvancesThroughputTraceEverySecond() {
        let registry = QWEngineTelemetryRegistry()
        let t0 = Date(timeIntervalSince1970: 1_700_000_000)

        registry.beginPrefill(promptTokens: 2_000, at: t0)
        registry.updatePrefillProgress(
            completedTokens: 2_000,
            totalTokens: 2_000,
            at: t0.addingTimeInterval(0.4)
        )
        registry.beginDecode(promptTokens: 2_000, at: t0.addingTimeInterval(0.4))

        for step in 1...4 {
            let tokenAt = t0.addingTimeInterval(0.4 + Double(step) * 0.25)
            registry.recordDecodeToken(at: tokenAt)
        }
        let s1 = registry.snapshot(at: t0.addingTimeInterval(1.5))
        XCTAssertEqual(s1.throughput.phase, "decode")
        XCTAssertGreaterThan(s1.throughput.liveDecodeTokensPerSec ?? 0, 0)
        XCTAssertEqual(s1.throughput.prefillRateSeries.count, s1.throughput.decodeRateSeries.count)
        let countAt1s = s1.throughput.decodeRateSeries.count
        XCTAssertGreaterThanOrEqual(countAt1s, 2)

        for step in 5...8 {
            let tokenAt = t0.addingTimeInterval(0.4 + Double(step) * 0.25)
            registry.recordDecodeToken(at: tokenAt)
        }
        let s2 = registry.snapshot(at: t0.addingTimeInterval(2.5))
        XCTAssertEqual(s2.throughput.decodeRateSeries.count, countAt1s + 1)
        XCTAssertEqual(s2.throughput.prefillRateSeries.count, s2.throughput.decodeRateSeries.count)

        registry.endRequest(
            tokensIn: 2_000,
            tokensOut: 8,
            prefillSec: 0.4,
            decodeSec: 2.2,
            ttftSec: 0.4,
            at: t0.addingTimeInterval(2.6)
        )
        let afterIdle = registry.snapshot(at: t0.addingTimeInterval(3.0))
        let laterIdle = registry.snapshot(at: t0.addingTimeInterval(12.0))
        XCTAssertEqual(afterIdle.throughput.phase, "idle")
        XCTAssertEqual(afterIdle.throughput.decodeRateSeries.count, laterIdle.throughput.decodeRateSeries.count)
    }

    func testActivePrefillAdvancesLiveRateAndTraceEverySecond() {
        let registry = QWEngineTelemetryRegistry()
        let t0 = Date(timeIntervalSince1970: 1_700_000_100)

        registry.beginPrefill(promptTokens: 100_000, at: t0)
        registry.updatePrefillProgress(completedTokens: 25_000, totalTokens: 100_000, at: t0.addingTimeInterval(1.1))
        let s1 = registry.snapshot(at: t0.addingTimeInterval(1.2))
        XCTAssertEqual(s1.throughput.phase, "prefill")
        XCTAssertGreaterThan(s1.throughput.livePrefillTokensPerSec ?? 0, 0)
        XCTAssertEqual(s1.throughput.displayPrefillTokensPerSec, s1.throughput.livePrefillTokensPerSec)
        let count1 = s1.throughput.prefillRateSeries.count
        XCTAssertGreaterThanOrEqual(count1, 2)
        XCTAssertEqual(s1.throughput.prefillRateSeries.count, s1.throughput.decodeRateSeries.count)

        registry.updatePrefillProgress(completedTokens: 55_000, totalTokens: 100_000, at: t0.addingTimeInterval(2.2))
        let s2 = registry.snapshot(at: t0.addingTimeInterval(2.3))
        XCTAssertEqual(s2.throughput.prefillRateSeries.count, count1 + 1)
        XCTAssertEqual(s2.throughput.prefillRateSeries.count, s2.throughput.decodeRateSeries.count)
        XCTAssertGreaterThan(s2.throughput.prefillRateSeries.compactMap { $0 }.count, 0)
    }

    func testSixtySecondTraceWindowScrollsLeftEverySecond() {
        let registry = QWEngineTelemetryRegistry()
        let t0 = Date(timeIntervalSince1970: 1_700_000_200)

        registry.beginPrefill(promptTokens: 1_000, at: t0)
        registry.beginDecode(promptTokens: 1_000, at: t0.addingTimeInterval(0.2))

        for sec in 1...66 {
            registry.recordDecodeToken(at: t0.addingTimeInterval(Double(sec) + 0.1))
            if sec == 65 {
                let s65 = registry.snapshot(at: t0.addingTimeInterval(65.5))
                XCTAssertEqual(s65.throughput.decodeRateSeries.count, 60)
                registry.recordDecodeToken(at: t0.addingTimeInterval(66.2))
                let s66 = registry.snapshot(at: t0.addingTimeInterval(66.5))
                XCTAssertEqual(s66.throughput.decodeRateSeries.count, 60)
                XCTAssertEqual(
                    Array(s65.throughput.decodeRateSeries.dropFirst().dropLast()),
                    Array(s66.throughput.decodeRateSeries.dropLast(2))
                )
                break
            }
        }
    }
}

