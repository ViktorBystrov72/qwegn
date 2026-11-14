import XCTest
@testable import QWEngine

/// The engine-owned telemetry registry: provenance, age, and honest gaps.
final class QWTelemetryRegistryTests: XCTestCase {
    func testRecordedValueCarriesProvenance() {
        let registry = QWTelemetryRegistry()
        registry.record("decode.tps", 42.0)
        XCTAssertEqual(registry.value("decode.tps")?.provenance, .measured)
        XCTAssertEqual(registry.value("decode.tps")?.value, 42.0)
        registry.record("prefill.tps", 100.0, provenance: .derived)
        XCTAssertEqual(registry.value("prefill.tps")?.provenance, .derived)
    }

    func testValueCarriesAgeAndStaleness() {
        let registry = QWTelemetryRegistry()
        let past = Date().addingTimeInterval(-120)
        registry.record("decode.tps", 42.0, provenance: .measured)
        let value = registry.value("decode.tps")
        XCTAssertNotNil(value?.age)
        XCTAssertFalse(value!.isStale(now: Date(), bound: 30))
        XCTAssertTrue(QWTelemetryValue(value: 1.0, provenance: .measured, sampledAt: past, reason: nil)
                        .isStale(now: Date(), bound: 30))
    }

    func testUnavailableValueIsAlwaysStale() {
        let value = QWTelemetryValue.unavailable("no device")
        XCTAssertTrue(value.isStale(now: Date(), bound: 3600))
        XCTAssertNil(value.age)
    }

    func testSnapshotReturnsEveryRecordedValue() {
        let registry = QWTelemetryRegistry()
        registry.record("a", 1.0)
        registry.lock.withLock {
            registry.values["b"] = .unavailable("nope")
        }
        let snapshot = registry.snapshot()
        XCTAssertEqual(snapshot.count, 2)
        XCTAssertEqual(snapshot["b"]?.provenance, .unavailable)
    }
}
