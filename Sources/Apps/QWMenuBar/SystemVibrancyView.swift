// SystemVibrancyView.swift — Adaptive Liquid Glass / Reduce Transparency backdrop

import SwiftUI

/// Adaptive translucency surface that falls back cleanly under Reduce Transparency.
public struct SystemVibrancyView: View {
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    public init() {}

    public var body: some View {
        Group {
            if reduceTransparency {
                TelemetryTheme.surface
            } else {
                Rectangle()
                    .fill(.ultraThinMaterial)
                    .glassEffect(.regular, in: .rect(cornerRadius: 0))
            }
        }
        .ignoresSafeArea()
    }
}
