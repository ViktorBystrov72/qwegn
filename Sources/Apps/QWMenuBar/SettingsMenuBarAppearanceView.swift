// SettingsMenuBarAppearanceView.swift — Menu bar status item metric toggles and live preview

import Observation
import QWServer
import SwiftUI

/// Lets the user pick which telemetry metrics appear next to the menu bar icon.
struct SettingsMenuBarAppearanceView: View {
    @Bindable var telemetry: TelemetryObservable
    @Environment(\.colorSchemeContrast) private var contrast: ColorSchemeContrast

    private var increasedContrast: Bool {
        contrast == .increased
    }

    private var hasAnySpeedEnabled: Bool {
        telemetry.isMenuBarMetricEnabled(.decodeSpeed) || telemetry.isMenuBarMetricEnabled(.prefillSpeed)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: "menubar.rectangle")
                    .font(.system(size: 11))
                    .foregroundStyle(TelemetryTheme.accent)
                Text("Menu Bar Appearance")
                    .font(TelemetryTheme.ui(12, .semibold))
                    .foregroundStyle(TelemetryTheme.primary(increasedContrast: increasedContrast))
            }

            VStack(alignment: .leading, spacing: 10) {
                previewStrip

                LazyVGrid(columns: [GridItem(.flexible(), spacing: 8), GridItem(.flexible(), spacing: 8)], spacing: 8) {
                    ForEach(QWMenuBarMetric.allCases) { metric in
                        metricToggleTile(metric)
                    }
                }

                Divider()

                HStack(alignment: .center, spacing: 8) {
                    VStack(alignment: .leading, spacing: 1) {
                        Text("Only show speeds during active generation")
                            .font(TelemetryTheme.ui(11.5, .medium))
                            .foregroundStyle(TelemetryTheme.secondary(increasedContrast: increasedContrast))
                        Text("Hides decode/prefill rates when the engine is idle; keeps memory or context visible.")
                            .font(TelemetryTheme.ui(10.5))
                            .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
                    }
                    Spacer(minLength: 8)
                    Toggle("", isOn: $telemetry.hideMenuBarSpeedsWhenIdle)
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .controlSize(.small)
                        .disabled(!hasAnySpeedEnabled)
                }
            }
            .padding(12)
            .background(TelemetryTheme.substrate)
            .clipShape(RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium))
            .overlay(
                RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium)
                    .strokeBorder(TelemetryTheme.hairline(increasedContrast: increasedContrast), lineWidth: 1)
            )
        }
    }

    private var previewStrip: some View {
        let liveText = telemetry.menuBarStatusTitle
        let sampleText = QWMenuBarStatusFormatter.samplePreviewTitle(
            enabledMetrics: telemetry.enabledMenuBarMetrics,
            hideSpeedsWhenIdle: telemetry.hideMenuBarSpeedsWhenIdle
        )
        let displayText = liveText ?? sampleText
        let isSample = liveText == nil && sampleText != nil

        return HStack(spacing: 8) {
            Text("Preview")
                .font(TelemetryTheme.ui(10.5, .medium))
                .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))

            HStack(spacing: 5) {
                ZStack(alignment: .bottomTrailing) {
                    QGProcessorLogoView()
                        .frame(width: 15, height: 15)
                        .foregroundStyle(TelemetryTheme.logoInk)
                    Circle()
                        .fill(TelemetryTheme.active)
                        .frame(width: 5.5, height: 5.5)
                        .offset(x: 2, y: 1.5)
                }
                if let displayText {
                    Text(displayText)
                        .font(TelemetryTheme.numeric(11, .medium))
                        .foregroundStyle(TelemetryTheme.primary(increasedContrast: increasedContrast))
                        .lineLimit(1)
                        .minimumScaleFactor(0.82)
                } else {
                    Text("Icon only")
                        .font(TelemetryTheme.ui(10.5))
                        .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(TelemetryTheme.rowSubstrate)
            .clipShape(Capsule())
            .overlay(
                Capsule()
                    .strokeBorder(TelemetryTheme.hairline(increasedContrast: increasedContrast), lineWidth: 0.75)
            )

            if isSample {
                Text("(sample)")
                    .font(TelemetryTheme.ui(10))
                    .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
            }

            Spacer(minLength: 0)
        }
    }

    private func metricToggleTile(_ metric: QWMenuBarMetric) -> some View {
        let isOn = telemetry.isMenuBarMetricEnabled(metric)
        let tint = isOn ? TelemetryTheme.accent : TelemetryTheme.hairline(increasedContrast: increasedContrast)

        return Button {
            telemetry.setMenuBarMetric(metric, enabled: !isOn)
        } label: {
            HStack(alignment: .center, spacing: 8) {
                Image(systemName: metric.icon)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(isOn ? TelemetryTheme.accent : TelemetryTheme.quiet(increasedContrast: increasedContrast))
                    .frame(width: 18)

                VStack(alignment: .leading, spacing: 1) {
                    Text(metric.title)
                        .font(TelemetryTheme.ui(11.5, .medium))
                        .foregroundStyle(TelemetryTheme.primary(increasedContrast: increasedContrast))
                        .lineLimit(1)
                    Text(metric.subtitle)
                        .font(TelemetryTheme.ui(10))
                        .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
                        .lineLimit(1)
                }

                Spacer(minLength: 4)

                Image(systemName: isOn ? "checkmark.circle.fill" : "circle")
                    .font(.system(size: 14))
                    .foregroundStyle(isOn ? TelemetryTheme.accent : TelemetryTheme.quiet(increasedContrast: increasedContrast))
            }
            .padding(.horizontal, 9)
            .padding(.vertical, 7)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(isOn ? TelemetryTheme.accent.opacity(0.08) : TelemetryTheme.rowSubstrate)
            .clipShape(RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium))
            .overlay(
                RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium)
                    .strokeBorder(isOn ? TelemetryTheme.accent.opacity(0.35) : tint, lineWidth: 0.75)
            )
        }
        .buttonStyle(.plain)
    }
}
