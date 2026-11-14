// CupertinoTelemetryView.swift — Menu bar popover chrome and glass hierarchy

import AppKit
import QWServer
import SwiftUI

/// Menu bar control panel popover hosting adaptive vibrancy behind card chrome.
public struct CupertinoTelemetryView: View {
    public var telemetry: TelemetryObservable
    public var onOpenWebUI: () -> Void
    public var onOpenLogs: () -> Void
    public var onOpenSettings: () -> Void
    public var onQuit: () -> Void

    @State var switcherAnchor: NSView?

    @Environment(\.accessibilityReduceTransparency) internal var reduceTransparency: Bool
    @Environment(\.accessibilityReduceMotion) internal var reduceMotion: Bool
    @Environment(\.colorSchemeContrast) internal var contrast: ColorSchemeContrast

    public init(
        telemetry: TelemetryObservable,
        onOpenWebUI: (() -> Void)? = nil,
        onOpenLogs: @escaping () -> Void = {},
        onOpenSettings: @escaping () -> Void = {},
        onQuit: @escaping () -> Void = { NSApplication.shared.terminate(nil) }
    ) {
        self.telemetry = telemetry
        self.onOpenWebUI = onOpenWebUI ?? { telemetry.openWebDashboard() }
        self.onOpenLogs = onOpenLogs
        self.onOpenSettings = onOpenSettings
        self.onQuit = onQuit
    }

    public var body: some View {
        ZStack {
            SystemVibrancyView()
                .ignoresSafeArea()

            GlassEffectContainer(spacing: 10) {
                VStack(spacing: 10) {
                    headerSection
                    failureBanner
                    loadedWeightsCard
                    inferenceTelemetryCard
                    memoryAndSystemCard
                    footerDock
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
                .frame(maxWidth: .infinity)
                .tint(.clear)
            }
        }
        .frame(width: 440)
        .fixedSize(horizontal: false, vertical: true)
    }

    @ViewBuilder
    var failureBanner: some View {
        if let error = telemetry.displayedError {
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 11))
                    .foregroundStyle(TelemetryTheme.failure)
                Text(error)
                    .font(TelemetryTheme.ui(11))
                    .foregroundStyle(TelemetryTheme.primary(increasedContrast: contrast == .increased))
                    .lineLimit(2)
                Spacer(minLength: 0)
            }
            .padding(8)
            .background(TelemetryTheme.failure.opacity(0.12))
            .clipShape(.rect(cornerRadius: TelemetryTheme.Radius.large))
            .overlay(
                RoundedRectangle(cornerRadius: TelemetryTheme.Radius.large)
                    .stroke(TelemetryTheme.failure.opacity(0.35), lineWidth: 0.5)
            )
        }
    }

    var statusDotColor: Color {
        switch telemetry.menuBarStatusState {
        case .offline: return TelemetryTheme.unknown
        case .standby, .transitioning: return TelemetryTheme.warning
        case .ready: return TelemetryTheme.active
        case .prefilling: return TelemetryTheme.prefill
        case .decoding: return TelemetryTheme.accent
        case .failed: return TelemetryTheme.failure
        }
    }

    var isActivelyGenerating: Bool {
        guard let phase = telemetry.dashboard?.generation else { return false }
        return phase == .prefill || phase == .decode
    }
}
