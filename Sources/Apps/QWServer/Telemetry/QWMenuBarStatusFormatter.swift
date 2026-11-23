// QWMenuBarStatusFormatter.swift — Selectable menu bar telemetry metrics and formatter

import Foundation
import QWCore

/// Telemetry dimensions users can surface directly in the macOS menu bar item.
public enum QWMenuBarMetric: String, CaseIterable, Codable, Sendable, Hashable, Identifiable {
    case decodeSpeed
    case prefillSpeed
    case engineMemory
    case contextSize
    case gpuTemperature

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .decodeSpeed: return "Decode speed"
        case .prefillSpeed: return "Prefill speed"
        case .engineMemory: return "Engine memory"
        case .contextSize: return "Context size"
        case .gpuTemperature: return "GPU temp"
        }
    }

    public var subtitle: String {
        switch self {
        case .decodeSpeed: return "Generation speed (tok/s)"
        case .prefillSpeed: return "Prompt rate or cache hit"
        case .engineMemory: return "Arena / resident RAM"
        case .contextSize: return "Active / context limit"
        case .gpuTemperature: return "Die hotspot (°C)"
        }
    }

    public var icon: String {
        switch self {
        case .decodeSpeed: return "gauge.with.dots.needle.67percent"
        case .prefillSpeed: return "bolt.horizontal.fill"
        case .engineMemory: return "memorychip"
        case .contextSize: return "text.alignleft"
        case .gpuTemperature: return "thermometer.medium"
        }
    }
}

/// Pure formatter turning published `QWDashboardViewModel` cards into a fixed-width menu bar title.
public enum QWMenuBarStatusFormatter {
    public static let decodeFieldWidth = 5
    public static let prefillFieldWidth = 6
    public static let memoryFieldWidth = 6
    public static let activeContextFieldWidth = 5
    public static let temperatureDigitsWidth = 3

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

    /// Formats a rate into an exact character width (`maxIntegerDigits + 2`, e.g. `" 91.9"` or `"  93.3"`).
    public static func formatFixedRate(_ value: Double?, maxIntegerDigits: Int) -> String {
        let width = maxIntegerDigits + 2
        guard let raw = value, raw.isFinite else {
            return padLeft("—", to: width)
        }
        let v = max(0.0, raw)
        let decimalThreshold = pow(10.0, Double(maxIntegerDigits)) - 0.05
        if v < decimalThreshold {
            return padLeft(String(format: "%.1f", v), to: width)
        }
        let k = v / 1000.0
        let kDecimalThreshold = pow(10.0, Double(max(1, width - 3))) - 0.05
        if k < kDecimalThreshold {
            return padLeft(String(format: "%.1fk", k), to: width)
        }
        let maxIntK = Int(pow(10.0, Double(max(1, width - 1))) - 1)
        let roundedK = Int(min(Double(maxIntK), k.rounded()))
        return padLeft("\(roundedK)k", to: width)
    }

    /// Formats an active context count into a fixed 5-character slot (`"   38"`, `"4,096"`, `" 128k"`).
    public static func formatFixedActiveTokens(_ tokens: Int?) -> String {
        guard let tokens else {
            return padLeft("—", to: activeContextFieldWidth)
        }
        let n = max(0, tokens)
        if n < 1_000 {
            return padLeft("\(n)", to: activeContextFieldWidth)
        }
        if n < 10_000 {
            return padLeft(QWTokenCountFormatter.groupedThousands(n), to: activeContextFieldWidth)
        }
        let k = Double(n) / 1024.0
        if k < 99.95 {
            return padLeft(String(format: "%.1fk", k), to: activeContextFieldWidth)
        }
        let roundedK = Int(k.rounded())
        if roundedK < 1_000 {
            return padLeft("\(roundedK)k", to: activeContextFieldWidth)
        }
        let m = Double(n) / 1_048_576.0
        if m < 99.95 {
            return padLeft(String(format: "%.1fM", m), to: activeContextFieldWidth)
        }
        let roundedM = Int(min(9999.0, m.rounded()))
        return padLeft("\(roundedM)M", to: activeContextFieldWidth)
    }

    /// Formats a compact fixed-width status-bar string from published cards and user metric selections.
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

        if enabledMetrics.contains(.engineMemory) {
            parts.append(formatMemorySegment(dashboard: dashboard))
        }

        if enabledMetrics.contains(.contextSize) {
            parts.append(formatContextSegment(dashboard: dashboard))
        }

        if enabledMetrics.contains(.gpuTemperature) {
            parts.append(formatTemperatureSegment(dashboard: dashboard))
        }

        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    /// Representative fixed-width preview string used in Settings when the engine is offline or idle.
    public static func samplePreviewTitle(
        enabledMetrics: Set<QWMenuBarMetric>,
        hideSpeedsWhenIdle: Bool
    ) -> String? {
        guard !enabledMetrics.isEmpty else { return nil }
        var parts: [String] = []
        let wantDecode = enabledMetrics.contains(.decodeSpeed)
        let wantPrefill = enabledMetrics.contains(.prefillSpeed)
        let sampleDecode = formatFixedRate(91.9, maxIntegerDigits: 3)
        let samplePrefill = formatFixedRate(93.3, maxIntegerDigits: 4)
        let speedPreview: String? = {
            if wantDecode && wantPrefill {
                return "↓\(sampleDecode) t/s · ↑\(samplePrefill) t/s"
            }
            if wantDecode {
                return "\(sampleDecode) tok/s"
            }
            if wantPrefill {
                return "↑\(samplePrefill) tok/s"
            }
            return nil
        }()
        if !hideSpeedsWhenIdle, let speedPreview {
            parts.append(speedPreview)
        }
        if enabledMetrics.contains(.engineMemory) {
            parts.append("\(formatFixedMemory(25.01)) GiB")
        }
        if enabledMetrics.contains(.contextSize) {
            parts.append("\(formatFixedActiveTokens(38))/\(formatCompactBudget(262_144))")
        }
        if enabledMetrics.contains(.gpuTemperature) {
            parts.append(formatFixedTemperature(54))
        }
        if parts.isEmpty, hideSpeedsWhenIdle, let speedPreview {
            return "\(speedPreview) (active)"
        }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    private static func formatSpeedSegment(
        dashboard: QWDashboardViewModel,
        wantDecode: Bool,
        wantPrefill: Bool
    ) -> String? {
        guard wantDecode || wantPrefill else { return nil }
        let decodeCard = wantDecode ? card("throughput.display_decode_tokens_per_sec", in: dashboard) : nil
        let prefillCard = wantPrefill ? card("throughput.display_prefill_tokens_per_sec", in: dashboard) : nil

        let decodeVal: Double? = (decodeCard?.unit == "tok/s") ? numericValue(of: decodeCard) : nil
        let prefillRate: Double? = (prefillCard?.unit == "tok/s") ? numericValue(of: prefillCard) : nil
        let prefillCached = prefillCard?.value == "from cache"

        let decodeSlot = formatFixedRate(decodeVal, maxIntegerDigits: 3)
        let prefillSlot = formatFixedRate(prefillRate, maxIntegerDigits: 4)

        if wantDecode && wantPrefill {
            let prefillPart = prefillCached ? "↑from cache" : "↑\(prefillSlot) t/s"
            return "↓\(decodeSlot) t/s · \(prefillPart)"
        }
        if wantDecode {
            return "\(decodeSlot) tok/s"
        }
        return prefillCached ? "↑  from cache" : "↑\(prefillSlot) tok/s"
    }

    private static func formatMemorySegment(dashboard: QWDashboardViewModel) -> String {
        let arena = card("memory.arena_reserved_bytes", in: dashboard)
        let resident = card("memory.resident_bytes", in: dashboard)
        let mem = (arena?.value != nil ? arena : resident)
        let rawUnit = (mem?.unit?.isEmpty == false ? mem!.unit! : "GiB")
        let unit = padRight(rawUnit, to: 3)
        return "\(formatFixedMemory(numericValue(of: mem))) \(unit)"
    }

    private static func formatContextSegment(dashboard: QWDashboardViewModel) -> String {
        let budget = dashboard.knobs["contextTokens"]?.intValue
        let activeCard = card("context.active_tokens", in: dashboard)
        let activeNum = numericValue(of: activeCard).map { Int($0.rounded()) }
        let activeSlot = formatFixedActiveTokens(activeNum)
        if let budget, budget > 0 {
            return "\(activeSlot)/\(formatCompactBudget(budget))"
        }
        return "\(activeSlot) tok"
    }

    private static func card(_ id: String, in dashboard: QWDashboardViewModel) -> QWDashboardCard? {
        dashboard.cards.first { $0.id == id }
    }

    private static func formatTemperatureSegment(dashboard: QWDashboardViewModel) -> String {
        formatFixedTemperature(numericValue(of: card("hardware.gpu_temp_celsius", in: dashboard)))
    }

    private static func formatFixedMemory(_ value: Double?) -> String {
        guard let raw = value, raw.isFinite else {
            return padLeft("—", to: memoryFieldWidth)
        }
        let v = max(0.0, raw)
        if v < 999.995 {
            return padLeft(String(format: "%.2f", v), to: memoryFieldWidth)
        }
        if v < 9999.95 {
            return padLeft(String(format: "%.1f", v), to: memoryFieldWidth)
        }
        let rounded = Int(min(999_999.0, v.rounded()))
        return padLeft("\(rounded)", to: memoryFieldWidth)
    }

    private static func formatFixedTemperature(_ value: Double?) -> String {
        guard let raw = value, raw.isFinite, raw > 0 else {
            return padLeft("—", to: temperatureDigitsWidth) + "°C"
        }
        let rounded = min(999, max(0, Int(min(999.0, raw.rounded()))))
        return padLeft("\(rounded)", to: temperatureDigitsWidth) + "°C"
    }

    private static func numericValue(of card: QWDashboardCard?) -> Double? {
        guard let card, let val = card.value else { return nil }
        return card.number ?? Double(val.replacingOccurrences(of: ",", with: ""))
    }

    private static func padLeft(_ text: String, to width: Int) -> String {
        text.count >= width ? text : String(repeating: " ", count: width - text.count) + text
    }

    private static func padRight(_ text: String, to width: Int) -> String {
        text.count >= width ? text : text + String(repeating: " ", count: width - text.count)
    }
}
