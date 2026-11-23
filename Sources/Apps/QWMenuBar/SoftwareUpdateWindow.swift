// SoftwareUpdateWindow.swift — Dedicated Cupertino release notes and verified installer window.

import AppKit
import QWCore
import SwiftUI

/// Cupertino-styled Software Update window implementing Option A UX.
public struct SoftwareUpdateView: View {
    public var telemetry: TelemetryObservable
    public var onClose: () -> Void

    @Environment(\.colorSchemeContrast) private var contrast: ColorSchemeContrast

    public init(telemetry: TelemetryObservable, onClose: @escaping () -> Void = {}) {
        self.telemetry = telemetry
        self.onClose = onClose
    }

    private var increasedContrast: Bool { contrast == .increased }
    private var updater: QWUpdateController { telemetry.updater }
    private var snap: QWUpdateStatusSnapshot { updater.status }
    private var isBusyInference: Bool {
        let phase = telemetry.dashboard?.generation
        return phase == .prefill || phase == .decode || telemetry.isModelTransitioning
    }

    public var body: some View {
        ZStack {
            SystemVibrancyView()
                .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 14) {
                headerRow

                if snap.phase == .readyToInstall, let sha = snap.verifiedSHA256 {
                    verifiedDigestBanner(sha: sha, signature: snap.signatureSummary)
                }

                if snap.updateAvailable && isBusyInference {
                    activeInferenceWarning
                }

                if snap.installOrigin != .pkgReceipt && snap.updateAvailable {
                    portableOriginNotice
                }

                if let rel = snap.latestRelease, snap.updateAvailable || !rel.notes.isEmpty {
                    releaseNotesCard(release: rel)
                } else if snap.phase == .error, let err = snap.errorMessage {
                    errorDetailCard(err)
                } else {
                    Spacer(minLength: 0)
                }

                if updater.isDownloading || snap.phase == .verifying {
                    downloadProgressSection
                }

                actionBar
            }
            .padding(18)
        }
        .frame(minWidth: 460, minHeight: 340)
    }

    private var headerRow: some View {
        HStack(alignment: .top, spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: TelemetryTheme.Radius.large)
                    .fill(TelemetryTheme.substrate)
                    .frame(width: 44, height: 44)
                    .overlay(
                        RoundedRectangle(cornerRadius: TelemetryTheme.Radius.large)
                            .strokeBorder(TelemetryTheme.hairline(increasedContrast: increasedContrast), lineWidth: 0.75)
                    )
                QGProcessorLogoView()
                    .frame(width: 26, height: 26)
                    .foregroundStyle(TelemetryTheme.logoInk)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(headlineText)
                    .font(TelemetryTheme.ui(14, .semibold))
                    .foregroundStyle(TelemetryTheme.primary(increasedContrast: increasedContrast))
                Text(subtitleText)
                    .font(TelemetryTheme.ui(11.5))
                    .foregroundStyle(TelemetryTheme.secondary(increasedContrast: increasedContrast))
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 0)
        }
    }

    private var headlineText: String {
        switch snap.phase {
        case .checking:
            return "Checking for updates…"
        case .error:
            return "Unable to check for updates"
        case .readyToInstall:
            return "Qwegn \(snap.latestRelease?.version.description ?? "") is ready to install"
        default:
            if snap.updateAvailable, let ver = snap.latestRelease?.version.description {
                return "Qwegn \(ver) is available"
            }
            return "Qwegn is up to date"
        }
    }

    private var subtitleText: String {
        if snap.phase == .checking {
            return "Contacting GitHub Releases for \(snap.currentVersion)…"
        }
        if snap.phase == .error {
            return snap.errorMessage ?? "Network error while checking releases."
        }
        if snap.updateAvailable, let rel = snap.latestRelease {
            let sizePart = rel.pkgAsset?.sizeBytes.map { " · \(formatBytes($0)) .pkg" } ?? ""
            return "Installed: v\(snap.currentVersion) → Latest: v\(rel.version.description)\(sizePart) (\(snap.installOrigin.displayTitle))."
        }
        let checked = snap.lastCheckedAt.map { "Last checked \(relativeTimeString(from: $0))." } ?? "Automatic checks: \(snap.schedule.displayTitle)."
        return "You are running Qwegn v\(snap.currentVersion). \(checked)"
    }

    private func verifiedDigestBanner(sha: String, signature: String?) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "checkmark.shield.fill")
                .font(.system(size: 13))
                .foregroundStyle(TelemetryTheme.active)
            VStack(alignment: .leading, spacing: 2) {
                Text("SHA-256 verified against SHA256SUMS.txt")
                    .font(TelemetryTheme.ui(11, .semibold))
                    .foregroundStyle(TelemetryTheme.active)
                Text(signature ?? String(sha.prefix(24)) + "…")
                    .font(TelemetryTheme.numeric(10))
                    .foregroundStyle(TelemetryTheme.secondary(increasedContrast: increasedContrast))
                    .lineLimit(1)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(TelemetryTheme.active.opacity(0.11))
        .clipShape(RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium))
        .overlay(
            RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium)
                .strokeBorder(TelemetryTheme.active.opacity(0.30), lineWidth: 0.75)
        )
        .help("SHA-256: \(sha)")
    }

    private var activeInferenceWarning: some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 12))
                .foregroundStyle(TelemetryTheme.warning)
            Text("Active inference or model load in progress — finish before opening Installer.app (which stops Qwegn.app).")
                .font(TelemetryTheme.ui(11))
                .foregroundStyle(TelemetryTheme.primary(increasedContrast: increasedContrast))
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(TelemetryTheme.warning.opacity(0.12))
        .clipShape(RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium))
        .overlay(
            RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium)
                .strokeBorder(TelemetryTheme.warning.opacity(0.32), lineWidth: 0.75)
        )
    }

    private var portableOriginNotice: some View {
        HStack(spacing: 8) {
            Image(systemName: "shippingbox")
                .font(.system(size: 12))
                .foregroundStyle(TelemetryTheme.cool)
            Text("Detected \(snap.installOrigin.displayTitle). Open the GitHub Release to update manually, or download the verified .pkg.")
                .font(TelemetryTheme.ui(11))
                .foregroundStyle(TelemetryTheme.secondary(increasedContrast: increasedContrast))
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(TelemetryTheme.rowSubstrate)
        .clipShape(RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium))
    }

    private func releaseNotesCard(release: QWReleaseInfo) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Release Notes · \(release.tagName)")
                    .font(TelemetryTheme.ui(11, .semibold))
                    .foregroundStyle(TelemetryTheme.secondary(increasedContrast: increasedContrast))
                Spacer()
                if let date = release.publishedAt {
                    Text(date.formatted(date: .abbreviated, time: .omitted))
                        .font(TelemetryTheme.numeric(10.5))
                        .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
                }
            }

            ScrollView(.vertical, showsIndicators: true) {
                Text(renderReleaseMarkdown(release.notes))
                    .font(TelemetryTheme.ui(11.5))
                    .foregroundStyle(TelemetryTheme.primary(increasedContrast: increasedContrast))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(10)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(TelemetryTheme.substrate)
            .clipShape(RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium))
            .overlay(
                RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium)
                    .strokeBorder(TelemetryTheme.hairline(increasedContrast: increasedContrast), lineWidth: 0.75)
            )
        }
    }

    private func errorDetailCard(_ message: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(message)
                .font(TelemetryTheme.ui(11.5))
                .foregroundStyle(TelemetryTheme.failure)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .padding(12)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(TelemetryTheme.failure.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium))
    }

    private var downloadProgressSection: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(snap.phase == .verifying ? "Verifying CryptoKit SHA-256…" : "Downloading verified .pkg…")
                    .font(TelemetryTheme.ui(11, .medium))
                    .foregroundStyle(TelemetryTheme.secondary(increasedContrast: increasedContrast))
                Spacer()
                Text(progressByteLabel)
                    .font(TelemetryTheme.numeric(10.5))
                    .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
            }
            if let frac = updater.downloadFraction {
                ProgressView(value: frac)
                    .progressViewStyle(.linear)
            } else {
                ProgressView()
                    .progressViewStyle(.linear)
            }
        }
    }

    private var actionBar: some View {
        HStack(spacing: 8) {
            if updater.isDownloading {
                Spacer()
                Button("Cancel Download") { updater.cancelDownload() }
                    .buttonStyle(.bordered)
            } else if snap.updateAvailable {
                if snap.isSkipped {
                    Button("Unskip") { updater.clearSkippedVersion() }
                        .buttonStyle(.bordered)
                        .fixedSize(horizontal: true, vertical: false)
                } else {
                    Button("Skip Version") {
                        updater.skipCurrentUpdate()
                        onClose()
                    }
                    .buttonStyle(.plain)
                    .font(TelemetryTheme.ui(11.5))
                    .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
                    .fixedSize(horizontal: true, vertical: false)
                }

                Spacer(minLength: 4)

                Button("Later") { onClose() }
                    .buttonStyle(.bordered)
                    .fixedSize(horizontal: true, vertical: false)

                if snap.phase == .readyToInstall {
                    Button("Reveal .pkg") { updater.revealDownloadedPkg() }
                        .buttonStyle(.bordered)
                        .fixedSize(horizontal: true, vertical: false)
                    Button("Open Installer…") { updater.openInstaller() }
                        .buttonStyle(.borderedProminent)
                        .fixedSize(horizontal: true, vertical: false)
                } else if snap.installOrigin == .pkgReceipt {
                    Button("Download & Install…") { updater.startDownload() }
                        .buttonStyle(.borderedProminent)
                        .fixedSize(horizontal: true, vertical: false)
                } else {
                    Button("Download .pkg") { updater.startDownload() }
                        .buttonStyle(.bordered)
                        .fixedSize(horizontal: true, vertical: false)
                    Button("Open Release ↗") { updater.openReleasePage() }
                        .buttonStyle(.borderedProminent)
                        .fixedSize(horizontal: true, vertical: false)
                }
            } else {
                if snap.latestRelease != nil {
                    Button("Release Page ↗") { updater.openReleasePage() }
                        .buttonStyle(.bordered)
                }
                Spacer()
                Button("Close") { onClose() }
                    .buttonStyle(.bordered)
                Button(updater.isChecking ? "Checking…" : "Check Again") {
                    updater.checkNow(force: true)
                }
                .buttonStyle(.borderedProminent)
                .disabled(updater.isChecking)
            }
        }
    }

    private var progressByteLabel: String {
        let done = updater.downloadedBytes ?? 0
        if let total = updater.totalBytes, total > 0 {
            let pct = Int((Double(done) / Double(total) * 100.0).rounded())
            return "\(formatBytes(done)) / \(formatBytes(total)) · \(pct)%"
        }
        return formatBytes(done)
    }

    private func renderReleaseMarkdown(_ notes: String) -> AttributedString {
        let trimmed = notes.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return AttributedString("No release notes provided.") }
        let opts = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        if let parsed = try? AttributedString(markdown: trimmed, options: opts) {
            return parsed
        }
        return AttributedString(trimmed)
    }

    private func formatBytes(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    private func relativeTimeString(from date: Date) -> String {
        if abs(Date().timeIntervalSince(date)) < 5 {
            return "just now"
        }
        let fmt = RelativeDateTimeFormatter()
        fmt.unitsStyle = .short
        return fmt.localizedString(for: date, relativeTo: Date())
            .trimmingCharacters(in: CharacterSet(charactersIn: "."))
    }
}
