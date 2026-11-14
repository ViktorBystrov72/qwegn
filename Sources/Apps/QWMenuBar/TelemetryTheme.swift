// TelemetryTheme.swift — Adaptive Light/Dark design tokens for QWMenuBar

import AppKit
import SwiftUI

/// Adaptive semantic colors, typography, spacing, and radii for Light and Dark Mode.
public enum TelemetryTheme {
    private static func adaptive(light: NSColor, dark: NSColor) -> Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            switch appearance.bestMatch(from: [.darkAqua, .aqua, .vibrantDark, .vibrantLight]) {
            case .darkAqua, .vibrantDark:
                return dark
            default:
                return light
            }
        })
    }

    public static let surface = adaptive(
        light: NSColor.windowBackgroundColor,
        dark: NSColor(srgbRed: 0.067, green: 0.075, blue: 0.094, alpha: 1.0)
    )
    public static let accent = adaptive(
        light: NSColor(srgbRed: 0.0, green: 0.439, blue: 0.902, alpha: 1.0),
        dark: NSColor(srgbRed: 0.039, green: 0.518, blue: 1.0, alpha: 1.0)
    )
    public static let prefill = adaptive(
        light: NSColor(srgbRed: 0.722, green: 0.420, blue: 0.0, alpha: 1.0),
        dark: NSColor(srgbRed: 1.0, green: 0.839, blue: 0.039, alpha: 1.0)
    )
    public static let cool = adaptive(
        light: NSColor(srgbRed: 0.0, green: 0.455, blue: 0.678, alpha: 1.0),
        dark: NSColor(srgbRed: 0.392, green: 0.824, blue: 1.0, alpha: 1.0)
    )
    public static let neutral = Color.secondary
    public static let highlight = adaptive(
        light: NSColor(srgbRed: 0.04, green: 0.38, blue: 0.82, alpha: 1.0),
        dark: NSColor(srgbRed: 0.667, green: 0.780, blue: 1.0, alpha: 1.0)
    )
    public static let logoInk = Color.primary
    public static let active = adaptive(
        light: NSColor(srgbRed: 0.086, green: 0.545, blue: 0.227, alpha: 1.0),
        dark: NSColor(srgbRed: 0.188, green: 0.820, blue: 0.345, alpha: 1.0)
    )
    public static let warning = adaptive(
        light: NSColor(srgbRed: 0.780, green: 0.400, blue: 0.0, alpha: 1.0),
        dark: NSColor(srgbRed: 1.0, green: 0.624, blue: 0.039, alpha: 1.0)
    )
    public static let failure = adaptive(
        light: NSColor(srgbRed: 0.800, green: 0.125, blue: 0.145, alpha: 1.0),
        dark: NSColor(srgbRed: 1.0, green: 0.271, blue: 0.227, alpha: 1.0)
    )
    public static let unknown = Color.secondary.opacity(0.75)

    public static let label = Font.system(size: 11, weight: .medium)
    public static let body = Font.system(size: 12, weight: .regular)
    public static let metric = Font.system(size: 12, weight: .medium, design: .monospaced)

    public static func ui(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        Font.system(size: size, weight: weight)
    }

    public static func numeric(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        Font.system(size: size, weight: weight, design: .monospaced).monospacedDigit()
    }

    public enum Radius {
        public static let small: CGFloat = 5
        public static let medium: CGFloat = 8
        public static let large: CGFloat = 10
    }

    public static let hairline = adaptive(
        light: NSColor.black.withAlphaComponent(0.10),
        dark: NSColor.white.withAlphaComponent(0.11)
    )
    public static let substrate = adaptive(
        light: NSColor.black.withAlphaComponent(0.045),
        dark: NSColor.white.withAlphaComponent(0.06)
    )
    public static let rowSubstrate = adaptive(
        light: NSColor.black.withAlphaComponent(0.03),
        dark: NSColor.white.withAlphaComponent(0.035)
    )

    public static func hairline(increasedContrast: Bool) -> Color {
        adaptive(
            light: NSColor.black.withAlphaComponent(increasedContrast ? 0.26 : 0.10),
            dark: NSColor.white.withAlphaComponent(increasedContrast ? 0.28 : 0.11)
        )
    }

    public static func secondary(increasedContrast: Bool) -> Color {
        increasedContrast ? Color.primary.opacity(0.88) : Color.secondary
    }

    public static func quiet(increasedContrast: Bool) -> Color {
        increasedContrast ? Color.primary.opacity(0.72) : Color.secondary.opacity(0.80)
    }

    public static func separator(increasedContrast: Bool) -> Color {
        adaptive(
            light: NSColor.black.withAlphaComponent(increasedContrast ? 0.28 : 0.12),
            dark: NSColor.white.withAlphaComponent(increasedContrast ? 0.30 : 0.12)
        )
    }

    public static func primary(increasedContrast: Bool) -> Color {
        let _ = increasedContrast
        return Color.primary
    }
}
