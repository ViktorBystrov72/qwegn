// QWTelemetryValue.swift — One telemetry reading, with its provenance.
// Pure: a value never travels without saying how it was obtained.

import Foundation

/// How a telemetry value came to be.
public enum QWTelemetryProvenance: String, Sendable, Codable, CaseIterable {
    /// Read straight from the hardware or the engine's own counters.
    case measured
    /// Computed from measured values.
    case derived
    /// Approximated; not a measurement.
    case estimated
    /// Measured, but so long ago that it no longer describes the present.
    /// A stale reading is never presented as a current one.
    case stale
    /// Not available. The reason says why. Never a number.
    case unavailable
}

/// A telemetry reading plus where it came from and when it was taken.
public struct QWTelemetryValue: Sendable, Codable, Equatable {
    /// The reading; nil when the value is unavailable.
    public let value: Double?
    /// How the reading was obtained.
    public let provenance: QWTelemetryProvenance
    /// When the reading was taken; nil when unavailable.
    public let sampledAt: Date?
    /// Why the value is unavailable, when it is.
    public let reason: String?

    public init(value: Double?, provenance: QWTelemetryProvenance, sampledAt: Date?, reason: String?) {
        self.value = value
        self.provenance = provenance
        self.sampledAt = sampledAt
        self.reason = reason
    }

    /// Seconds since the reading was taken; nil when it has no timestamp.
    public var age: TimeInterval? { sampledAt.map { Date().timeIntervalSince($0) } }

    /// Whether the reading is older than `bound` seconds.
    public func isStale(now: Date, bound: TimeInterval) -> Bool {
        guard let sampledAt else { return true }
        return now.timeIntervalSince(sampledAt) > bound
    }

    /// No reading, and the reason there is none.
    public static func unavailable(_ reason: String) -> QWTelemetryValue {
        .init(value: nil, provenance: .unavailable, sampledAt: nil, reason: reason)
    }
}
