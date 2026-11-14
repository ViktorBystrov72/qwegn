// QWMenuBarStatusFormatter.swift — Selectable menu bar telemetry metrics and formatter

import Foundation
import QWCore

/// Telemetry dimensions users can surface directly in the macOS menu bar item.
public enum QWMenuBarMetric: String, CaseIterable, Codable, Sendable, Hashable, Identifiable {
    case decodeSpeed
    case prefillSpeed
    case engineMemory
    case contextSize

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .decodeSpeed: return "Decode speed"
        case .prefillSpeed: return "Prefill speed"
        case .engineMemory: return "Engine memory"
        case .contextSize: return "Context size"
        }
    }

    public var subtitle: String {
        switch self {
        case .decodeSpeed: return "Generation speed (tok/s)"
        case .prefillSpeed: return "Prompt rate or cache hit"
        case .engineMemory: return "Arena / resident RAM"
        case .contextSize: return "Active / context limit"
        }
    }

    public var icon: String {
        switch self {
        case .decodeSpeed: return "gauge.with.dots.needle.67percent"
        case .prefillSpeed: return "bolt.horizontal.fill"
        case .engineMemory: return "memorychip"
        case .contextSize: return "text.alignleft"
        }
    }
}

/// Pure formatter turning published `QWDashboardViewModel` cards into a compact menu bar title.
public enum QWMenuBarStatusFormatter {
    /// Formats a token budget compactly (e.g. `262144` -> `"256k"`, `1048576` -> `"1M"`).
    public static func formatCompactBudget(_ tokens: Int) -> String {
        guard tokens > 0 else { return "0" }
        if tokens >= 1_048_576 {
            if tokens % 1_048_576 == 0 {
                return "\(tokens / 1_048_576)M"
            }
            return String(format: "%.1fM", Double(tokens) / 1_048_576.0)
        }
        if tokens >= 1024 {
            if tokens % 1024 == 0 {
                return "\(tokens / 1024)k"
            }
            return String(format: "%.1fk", Double(tokens) / 1024.0)
        }
        return "\(tokens)"
    }

    /// Formats a compact status-bar string from published cards and user metric selections.
    public static func statusTitle(
        dashboard: QWDashboardViewModel?,
        isConnected: Bool,
        enabledMetrics: Set<QWMenuBarMetric>,
        hideSpeedsWhenIdle: Bool
    ) -> String? {
        guard isConnected,
              let dashboard,
              dashboard.lifecycle.phase == .loaded,
              !enabledMetrics.isEmpty else {
            return nil
        }

        var parts: [String] = []
        let gen = dashboard.generation
        let allowSpeeds = !hideSpeedsWhenIdle || gen == .prefill || gen == .decode

        if allowSpeeds, let speedSegment = formatSpeedSegment(
            dashboard: dashboard,
            wantDecode: enabledMetrics.contains(.decodeSpeed),
            wantPrefill: enabledMetrics.contains(.prefillSpeed)
        ) {
            parts.append(speedSegment)
        }

        if enabledMetrics.contains(.engineMemory),
           let memSegment = formatMemorySegment(dashboard: dashboard) {
            parts.append(memSegment)
        }

        if enabledMetrics.contains(.contextSize),
           let ctxSegment = formatContextSegment(dashboard: dashboard) {
            parts.append(ctxSegment)
        }

        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    /// Representative preview string used in Settings when the engine is offline or idle.
    public static func samplePreviewTitle(
        enabledMetrics: Set<QWMenuBarMetric>,
        hideSpeedsWhenIdle: Bool
    ) -> String? {
        guard !enabledMetrics.isEmpty else { return nil }
        var parts: [String] = []
        let wantDecode = enabledMetrics.contains(.decodeSpeed)
        let wantPrefill = enabledMetrics.contains(.prefillSpeed)
        if !hideSpeedsWhenIdle {
            if wantDecode && wantPrefill {
                parts.append("↓91.9 · ↑93.3 tok/s")
            } else if wantDecode {
                parts.append("91.9 tok/s")
            } else if wantPrefill {
                parts.append("↑93.3 tok/s")
            }
        }
        if enabledMetrics.contains(.engineMemory) {
            parts.append("25.01 GiB")
        }
        if enabledMetrics.contains(.contextSize) {
            parts.append("38/256k")
        }
        if parts.isEmpty && hideSpeedsWhenIdle && (wantDecode || wantPrefill) {
            return wantDecode && wantPrefill ? "↓91.9 · ↑93.3 tok/s (active)" : "91.9 tok/s (active)"
        }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    private static func formatSpeedSegment(
        dashboard: QWDashboardViewModel,
        wantDecode: Bool,
        wantPrefill: Bool
    ) -> String? {
        let decodeCard = wantDecode ? card("throughput.display_decode_tokens_per_sec", in: dashboard) : nil
        let prefillCard = wantPrefill ? card("throughput.display_prefill_tokens_per_sec", in: dashboard) : nil

        let decodeVal = (decodeCard?.unit == "tok/s") ? decodeCard?.value : nil
        let prefillRate = (prefillCard?.unit == "tok/s") ? prefillCard?.value : nil
        let prefillCached = prefillCard?.value == "from cache"

        if let d = decodeVal, let p = prefillRate {
            return "↓\(d) · ↑\(p) tok/s"
        }
        if let d = decodeVal, prefillCached {
            return "↓\(d) tok/s · ↑cache"
        }
        if let d = decodeVal {
            return wantPrefill ? "↓\(d) tok/s" : "\(d) tok/s"
        }
        if let p = prefillRate {
            return "↑\(p) tok/s"
        }
        if prefillCached {
            return "↑cache"
        }
        return nil
    }

    private static func formatMemorySegment(dashboard: QWDashboardViewModel) -> String? {
        let arena = card("memory.arena_reserved_bytes", in: dashboard)
        let resident = card("memory.resident_bytes", in: dashboard)
        guard let mem = (arena?.value != nil ? arena : resident),
              let val = mem.value else {
            return nil
        }
        let unit = mem.unit.map { " \($0)" } ?? ""
        return "\(val)\(unit)"
    }

    private static func formatContextSegment(dashboard: QWDashboardViewModel) -> String? {
        let budget = dashboard.knobs["contextTokens"]?.intValue
        let activeCard = card("context.active_tokens", in: dashboard)
        if let active = compactActiveTokens(activeCard) {
            if let budget, budget > 0 {
                return "\(active)/\(formatCompactBudget(budget))"
            }
            return "\(active) tok"
        }
        if let budget, budget > 0 {
            return "\(formatCompactBudget(budget)) ctx"
        }
        return nil
    }

    private static func compactActiveTokens(_ card: QWDashboardCard?) -> String? {
        guard let card, card.value != nil else { return nil }
        if let n = card.number {
            let rounded = Int(n.rounded())
            return rounded >= 10_000 ? formatCompactBudget(rounded) : QWTokenCountFormatter.format(rounded)
        }
        return card.value
    }

    private static func card(_ id: String, in dashboard: QWDashboardViewModel) -> QWDashboardCard? {
        dashboard.cards.first { $0.id == id }
    }
}
