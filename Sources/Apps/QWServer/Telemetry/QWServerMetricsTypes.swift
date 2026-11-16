// QWServerMetricsTypes.swift — Metric keys, display modes, and snapshot DTOs.

import Foundation
import Darwin
import QWEngine

public enum QWTelemetryMetric: String, Sendable, CaseIterable {
    case livePrefillTokensPerSec = "throughput.live_prefill_tokens_per_sec"
    case liveDecodeTokensPerSec = "throughput.live_decode_tokens_per_sec"
    case lastPrefillTokensPerSec = "throughput.last_prefill_tokens_per_sec"
    case lastDecodeTokensPerSec = "throughput.last_decode_tokens_per_sec"
    case ema1mPrefillTokensPerSec = "throughput.ema_1m_prefill_tokens_per_sec"
    case ema1mDecodeTokensPerSec = "throughput.ema_1m_decode_tokens_per_sec"
    case avgPrefillTokensPerSec = "throughput.avg_prefill_tokens_per_sec"
    case avgDecodeTokensPerSec = "throughput.avg_decode_tokens_per_sec"
    case displayPrefillTokensPerSec = "throughput.display_prefill_tokens_per_sec"
    case displayDecodeTokensPerSec = "throughput.display_decode_tokens_per_sec"
    case ttftMs = "latency.ttft_ms"
    case cpuPercent = "hardware.cpu_percent"
    case gpuPercent = "hardware.gpu_percent"
    case memoryPercent = "hardware.memory_percent"
    case memoryPressure = "hardware.memory_pressure"
    case gpuTemperatureCelsius = "hardware.gpu_temp_celsius"
    case fanSpeedRPM = "hardware.fan_rpm"
    case systemPowerWatts = "hardware.system_power_watts"
    case thermalState = "hardware.thermal_state"
    case gpuBusyRatio = "gpu.busy_ratio"
    /// The last turn's prompt was served entirely from the prefix
    /// cache: the engine ran no prefill phase, so it has no prefill speed.
    case prefillFromCache = "throughput.prefill_from_cache"
}

/// How long a reading stays current. Past its bound a reading is stale: it
/// was measured, but it no longer describes the present.
public enum QWTelemetryFreshness {
    /// A host reading is refreshed every couple of seconds.
    public static let hardwareStaleAfter: TimeInterval = 10
    /// A rate measured this long ago is no longer the engine's speed.
    public static let rateStaleAfter: TimeInterval = 30
    /// The engine's busy ratio describes its last measured step.
    public static let busyRatioStaleAfter: TimeInterval = 30
    /// The cache changes only when a cached request runs, so it stays
    /// current far longer than a host reading.
    public static let cacheStaleAfter: TimeInterval = 300
    /// A latency describes the turn it came from.
    public static let latencyStaleAfter: TimeInterval = 60
}

/// What a displayed rate is, so the UI can label it.
public enum QWRateDisplayMode: String, Sendable, Codable, CaseIterable {
    /// The current phase's live rate.
    case live
    /// The average over every completed turn.
    case average = "avg"
    /// The most recent measurement, still current.
    case last
    /// The most recent measurement, too old to present as current.
    case stale

    /// A payload may name a mode this build does not know. The honest
    /// default is the least committal one.
    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = QWRateDisplayMode(rawValue: raw) ?? .last
    }
}

extension QWEngineTelemetryRegistry {
    public enum EngineInferencePhase: String, Codable, Sendable {
        case idle = "idle"
        case prefill = "prefill"
        case decode = "decode"
    }

    public struct MetricsSnapshot: Codable, Sendable {
        public let cache: CacheMetrics
        public let throughput: ThroughputMetrics
        public let latency: LatencyMetrics
        public let context: ContextMetrics
        public let memory: MemoryMetrics
        public let hardware: QWSystemHardwareUtilization?
        /// The engine's own GPU busy ratio for its last measured step.
        public let gpuBusyRatio: Double?
        /// How each metric was obtained. A metric that is not listed was
        /// never measured, and the UI says so instead of showing a zero.
        public let provenance: [String: QWTelemetryProvenance]
        /// When each metric was taken. A cached metric says how old it is.
        public let sampledAt: [String: Date]
        /// Why a metric is unavailable or stale, when it is.
        public let unavailableReasons: [String: String]

        public struct CacheMetrics: Codable, Sendable {
            /// The warm cache's bytes. Nil means the engine cannot
            /// measure them — absent, never zero.
            public let warmBytes: Int?
            public let warmLimitBytes: Int
            /// The SSD cache's bytes. Nil means the engine cannot measure
            /// them — absent, never zero.
            public let ssdBytes: Int?
            public let ssdLimitBytes: Int
            public let lookupsTotal: Int
            public let hitsWarmTotal: Int
            public let hitsSSDTotal: Int
            /// Hits whose tier this build does not name. The tokens were
            /// reused; the tier is not in the vocabulary. Nil means this
            /// server does not report it — absent, never zero.
            public let hitsUnclassifiedTotal: Int?
            public let missesTotal: Int
            public let tokensReusedTotal: Int
            public let evictionsWarmTotal: Int
            public let evictionsSSDTotal: Int
            /// When the cache was last written. Nil means it never was.
            public var sampledAt: Date? = nil
        }

        public struct ThroughputMetrics: Codable, Sendable {
            public let phase: String
            public let tokensInTotal: Int
            public let tokensOutTotal: Int
            /// The live rates. Nil means "not measured", never zero.
            public let livePrefillTokensPerSec: Double?
            public let liveDecodeTokensPerSec: Double?
            public let lastPrefillTokensPerSec: Double?
            public let lastDecodeTokensPerSec: Double?
            public let ema1mPrefillTokensPerSec: Double?
            public let ema1mDecodeTokensPerSec: Double?
            /// Overall averages across every completed turn since start (token-weighted:
            /// computed tokens / time spent computing them). Prefill counts ONLY tokens the
            /// engine actually computed — tokens served from the prefix cache are excluded.
            public let avgPrefillTokensPerSec: Double?
            public let avgDecodeTokensPerSec: Double?
            /// What a UI should show right now. The mode says which number it is.
            public let displayPrefillTokensPerSec: Double?
            public let displayDecodeTokensPerSec: Double?
            public let prefillDisplayMode: QWRateDisplayMode
            public let decodeDisplayMode: QWRateDisplayMode
            /// The published trace of the prefill rate. A nil is a gap.
            public var prefillRateSeries: [Double?] = []
            /// The published trace of the decode rate. A nil is a gap.
            public var decodeRateSeries: [Double?] = []
            /// Progress of the turn in flight: the tokens the engine had to compute.
            public let prefillProgressTokens: Int
            public let prefillTotalTokens: Int
            /// True when the last turn's prompt was served entirely from
            /// the prefix cache: the engine computed no prefill tokens,
            /// so that turn had no prefill speed at all.
            public let prefillFromCache: Bool

            public init(phase: String, tokensInTotal: Int, tokensOutTotal: Int,
                        livePrefillTokensPerSec: Double?, liveDecodeTokensPerSec: Double?,
                        lastPrefillTokensPerSec: Double?, lastDecodeTokensPerSec: Double?,
                        ema1mPrefillTokensPerSec: Double?, ema1mDecodeTokensPerSec: Double?,
                        avgPrefillTokensPerSec: Double?, avgDecodeTokensPerSec: Double?,
                        displayPrefillTokensPerSec: Double?, displayDecodeTokensPerSec: Double?,
                        prefillDisplayMode: QWRateDisplayMode, decodeDisplayMode: QWRateDisplayMode,
                        prefillProgressTokens: Int, prefillTotalTokens: Int,
                        prefillRateSeries: [Double?] = [], decodeRateSeries: [Double?] = [],
                        prefillFromCache: Bool = false) {
                self.phase = phase
                self.tokensInTotal = tokensInTotal
                self.tokensOutTotal = tokensOutTotal
                self.livePrefillTokensPerSec = livePrefillTokensPerSec
                self.liveDecodeTokensPerSec = liveDecodeTokensPerSec
                self.lastPrefillTokensPerSec = lastPrefillTokensPerSec
                self.lastDecodeTokensPerSec = lastDecodeTokensPerSec
                self.ema1mPrefillTokensPerSec = ema1mPrefillTokensPerSec
                self.ema1mDecodeTokensPerSec = ema1mDecodeTokensPerSec
                self.avgPrefillTokensPerSec = avgPrefillTokensPerSec
                self.avgDecodeTokensPerSec = avgDecodeTokensPerSec
                self.displayPrefillTokensPerSec = displayPrefillTokensPerSec
                self.displayDecodeTokensPerSec = displayDecodeTokensPerSec
                self.prefillDisplayMode = prefillDisplayMode
                self.decodeDisplayMode = decodeDisplayMode
                self.prefillProgressTokens = prefillProgressTokens
                self.prefillTotalTokens = prefillTotalTokens
                self.prefillFromCache = prefillFromCache
                self.prefillRateSeries = prefillRateSeries
                self.decodeRateSeries = decodeRateSeries
            }

            /// Decodes a payload from an older server: a field that is not
            /// there was never measured, not zero.
            public init(from decoder: Decoder) throws {
                let c = try decoder.container(keyedBy: CodingKeys.self)
                self.init(
                    phase: try c.decodeIfPresent(String.self, forKey: .phase) ?? "idle",
                    tokensInTotal: try c.decodeIfPresent(Int.self, forKey: .tokensInTotal) ?? 0,
                    tokensOutTotal: try c.decodeIfPresent(Int.self, forKey: .tokensOutTotal) ?? 0,
                    livePrefillTokensPerSec: try c.decodeIfPresent(Double.self, forKey: .livePrefillTokensPerSec),
                    liveDecodeTokensPerSec: try c.decodeIfPresent(Double.self, forKey: .liveDecodeTokensPerSec),
                    lastPrefillTokensPerSec: try c.decodeIfPresent(Double.self, forKey: .lastPrefillTokensPerSec),
                    lastDecodeTokensPerSec: try c.decodeIfPresent(Double.self, forKey: .lastDecodeTokensPerSec),
                    ema1mPrefillTokensPerSec: try c.decodeIfPresent(Double.self, forKey: .ema1mPrefillTokensPerSec),
                    ema1mDecodeTokensPerSec: try c.decodeIfPresent(Double.self, forKey: .ema1mDecodeTokensPerSec),
                    avgPrefillTokensPerSec: try c.decodeIfPresent(Double.self, forKey: .avgPrefillTokensPerSec),
                    avgDecodeTokensPerSec: try c.decodeIfPresent(Double.self, forKey: .avgDecodeTokensPerSec),
                    displayPrefillTokensPerSec: try c.decodeIfPresent(Double.self, forKey: .displayPrefillTokensPerSec),
                    displayDecodeTokensPerSec: try c.decodeIfPresent(Double.self, forKey: .displayDecodeTokensPerSec),
                    prefillDisplayMode: try c.decodeIfPresent(QWRateDisplayMode.self, forKey: .prefillDisplayMode) ?? .last,
                    decodeDisplayMode: try c.decodeIfPresent(QWRateDisplayMode.self, forKey: .decodeDisplayMode) ?? .last,
                    prefillProgressTokens: try c.decodeIfPresent(Int.self, forKey: .prefillProgressTokens) ?? 0,
                    prefillTotalTokens: try c.decodeIfPresent(Int.self, forKey: .prefillTotalTokens) ?? 0,
                    prefillRateSeries: try c.decodeIfPresent([Double?].self, forKey: .prefillRateSeries) ?? [],
                    decodeRateSeries: try c.decodeIfPresent([Double?].self, forKey: .decodeRateSeries) ?? [],
                    prefillFromCache: try c.decodeIfPresent(Bool.self, forKey: .prefillFromCache) ?? false)
            }
        }

        public struct LatencyMetrics: Codable, Sendable {
            public let lastTtftMs: Double?
            public let avgTtftMs: Double?
        }

        public struct ContextMetrics: Codable, Sendable {
            /// The tokens the last turn used. Nil means no turn has run:
            /// absent, never zero.
            public var activeTokens: Int?
            public let maxTokens: Int
            /// When the context was last updated. Nil means never.
            public var sampledAt: Date? = nil
        }

        public struct MemoryMetrics: Codable, Sendable {
            public let residentBytes: Int
            public let arenaReservedBytes: Int
        }

        public init(cache: CacheMetrics, throughput: ThroughputMetrics, latency: LatencyMetrics,
                    context: ContextMetrics, memory: MemoryMetrics,
                    hardware: QWSystemHardwareUtilization?, gpuBusyRatio: Double?,
                    provenance: [String: QWTelemetryProvenance], sampledAt: [String: Date],
                    unavailableReasons: [String: String]) {
            self.cache = cache
            self.throughput = throughput
            self.latency = latency
            self.context = context
            self.memory = memory
            self.hardware = hardware
            self.gpuBusyRatio = gpuBusyRatio
            self.provenance = provenance
            self.sampledAt = sampledAt
            self.unavailableReasons = unavailableReasons
        }

        /// Decodes a payload from an older server: a field that is not there
        /// is unavailable, not a zero that reads like an idle engine.
        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.init(
                cache: try c.decodeIfPresent(CacheMetrics.self, forKey: .cache)
                    ?? .init(warmBytes: 0, warmLimitBytes: 0, ssdBytes: 0, ssdLimitBytes: 0,
                             lookupsTotal: 0, hitsWarmTotal: 0, hitsSSDTotal: 0,
                             hitsUnclassifiedTotal: nil, missesTotal: 0,
                             tokensReusedTotal: 0, evictionsWarmTotal: 0, evictionsSSDTotal: 0),
                throughput: try c.decodeIfPresent(ThroughputMetrics.self, forKey: .throughput)
                    ?? .init(phase: "idle", tokensInTotal: 0, tokensOutTotal: 0,
                             livePrefillTokensPerSec: nil, liveDecodeTokensPerSec: nil,
                             lastPrefillTokensPerSec: nil, lastDecodeTokensPerSec: nil,
                             ema1mPrefillTokensPerSec: nil, ema1mDecodeTokensPerSec: nil,
                             avgPrefillTokensPerSec: nil, avgDecodeTokensPerSec: nil,
                             displayPrefillTokensPerSec: nil, displayDecodeTokensPerSec: nil,
                             prefillDisplayMode: .last, decodeDisplayMode: .last,
                             prefillProgressTokens: 0, prefillTotalTokens: 0),
                latency: try c.decodeIfPresent(LatencyMetrics.self, forKey: .latency)
                    ?? .init(lastTtftMs: nil, avgTtftMs: nil),
                context: try c.decodeIfPresent(ContextMetrics.self, forKey: .context)
                    ?? .init(activeTokens: 0, maxTokens: 0),
                memory: try c.decodeIfPresent(MemoryMetrics.self, forKey: .memory)
                    ?? .init(residentBytes: 0, arenaReservedBytes: 0),
                hardware: try c.decodeIfPresent(QWSystemHardwareUtilization.self, forKey: .hardware),
                gpuBusyRatio: try c.decodeIfPresent(Double.self, forKey: .gpuBusyRatio),
                provenance: try c.decodeIfPresent([String: QWTelemetryProvenance].self, forKey: .provenance) ?? [:],
                sampledAt: try c.decodeIfPresent([String: Date].self, forKey: .sampledAt) ?? [:],
                unavailableReasons: try c.decodeIfPresent([String: String].self, forKey: .unavailableReasons) ?? [:])
        }
    }
}
