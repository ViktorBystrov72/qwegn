// SettingsView+Updates.swift — Software Updates group inside Settings -> General.

import QWCore
import QWServer
import SwiftUI

extension SettingsView {
    var softwareUpdatesGroup: some View {
        let snap = telemetry.updater.status
        return settingsGroup(title: "Software Updates", icon: "arrow.triangle.2.circlepath") {
            formRow(label: "Check Automatically:", key: "updateCheckSchedule") {
                Picker("", selection: $draftSettings.updateCheckSchedule) {
                    ForEach(QWUpdateCheckSchedule.allCases, id: \.rawValue) { item in
                        Text(item.displayTitle).tag(item.rawValue)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            if let note = QWSettingsCoherence.effectNote(for: "updateCheckSchedule") {
                Text(note)
                    .font(TelemetryTheme.ui(11))
                    .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(updateSummaryLine(snap))
                        .font(TelemetryTheme.ui(11, .medium))
                        .foregroundStyle(TelemetryTheme.secondary(increasedContrast: increasedContrast))
                    Text("Install source: \(snap.installOrigin.displayTitle)")
                        .font(TelemetryTheme.ui(10.5))
                        .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
                }
                Spacer(minLength: 8)
                if snap.isSkipped, let ver = snap.latestRelease?.version.description {
                    Button("Unskip v\(ver)") {
                        telemetry.updater.clearSkippedVersion()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
                Button(telemetry.updater.isChecking ? "Checking…" : "Check Now") {
                    onOpenUpdates(true)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(telemetry.updater.isChecking)
            }
        }
    }

    private func updateSummaryLine(_ snap: QWUpdateStatusSnapshot) -> String {
        let checked = snap.lastCheckedAt.map { date -> String in
            if abs(Date().timeIntervalSince(date)) < 5 {
                return "checked just now"
            }
            let fmt = RelativeDateTimeFormatter()
            fmt.unitsStyle = .short
            let rel = fmt.localizedString(for: date, relativeTo: Date())
                .trimmingCharacters(in: CharacterSet(charactersIn: "."))
            return "checked \(rel)"
        } ?? "never checked"
        if snap.updateAvailable, let rel = snap.latestRelease {
            return "Installed v\(snap.currentVersion) · v\(rel.version.description) available · \(checked)"
        }
        return "Installed v\(snap.currentVersion) · \(checked)"
    }
}
