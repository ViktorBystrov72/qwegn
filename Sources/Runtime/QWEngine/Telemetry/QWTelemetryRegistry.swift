// QWTelemetryRegistry.swift — The engine's own record of what it measured.
// Pure: the registry stores what the engine reported, with its provenance and
// age. It never invents a reading and never turns a gap into a zero.

import Foundation

/// The engine-side registry of telemetry readings.
public final class QWTelemetryRegistry: @unchecked Sendable {
    /// The process-wide registry the engine writes to.
    public static let shared = QWTelemetryRegistry()

    let lock = NSLock()
    var values: [String: QWTelemetryValue] = [:]

    public init() {}

    /// Records a reading with its provenance.
    public func record(_ name: String, _ value: Double, provenance: QWTelemetryProvenance = .measured) {
        lock.lock(); defer { lock.unlock() }
        values[name] = .init(value: value, provenance: provenance, sampledAt: Date(), reason: nil)
    }

    /// The last reading for a name, with its provenance and age.
    public func value(_ name: String) -> QWTelemetryValue? {
        lock.lock(); defer { lock.unlock() }
        return values[name]
    }

    /// Every recorded reading, by name.
    public func snapshot() -> [String: QWTelemetryValue] {
        lock.lock(); defer { lock.unlock() }
        return values
    }

    /// Forgets every reading.
    public func reset() {
        lock.lock(); defer { lock.unlock() }
        values.removeAll()
    }
}
