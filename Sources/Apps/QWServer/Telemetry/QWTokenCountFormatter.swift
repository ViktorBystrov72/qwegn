// QWTokenCountFormatter.swift — Locale-independent formatting for engine token counters.
// Applies comma thousands separators below 1,000,000 and switches to compact M notation at 1,000,000+.

import Foundation

public enum QWTokenCountFormatter {
    /// Formats a discrete token count (`< 1,000,000` with ASCII comma separators, `>= 1,000,000` as `M`).
    public static func format(_ count: Int) -> String {
        format(Double(count))
    }

    /// Formats a token-count measurement with comma grouping below `1,000,000` and `M` at `1,000,000+`.
    public static func format(_ value: Double) -> String {
        guard value.isFinite else { return "—" }
        let rounded = value.rounded()
        let sign = rounded < 0 ? "-" : ""
        let absVal = abs(rounded)
        if absVal >= 99_995_000 {
            return String(format: "%@%.1fM", sign, absVal / 1_000_000.0)
        }
        if absVal >= 1_000_000 {
            return String(format: "%@%.2fM", sign, absVal / 1_000_000.0)
        }
        return "\(sign)\(groupedThousands(Int(absVal)))"
    }

    /// Formats a non-negative integer with ASCII comma thousands separators deterministically across locales.
    public static func groupedThousands(_ nonNegativeValue: Int) -> String {
        let clamped = max(0, nonNegativeValue)
        if clamped < 1_000 { return "\(clamped)" }
        let digits = Array(String(clamped))
        var out = ""
        out.reserveCapacity(digits.count + digits.count / 3)
        for (index, ch) in digits.enumerated() {
            if index > 0 && (digits.count - index) % 3 == 0 {
                out.append(",")
            }
            out.append(ch)
        }
        return out
    }
}
