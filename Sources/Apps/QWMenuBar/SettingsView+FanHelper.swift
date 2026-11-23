// SettingsView+FanHelper.swift — Fan Helper LaunchDaemon status and setup actions.
// Drives one-click elevation/SMAppService setup from Settings -> Performance
// so users can unlock Pre-Spin and Max Blast without running Qwegn as root.

import QWSMCHelper
import ServiceManagement
import SwiftUI

extension SettingsView {
    func refreshFanHelperStatus() {
        Task { @MainActor in
            let status = await Task.detached(priority: .utility) {
                QWSMCHelperManager.currentStatus()
            }.value
            fanHelperStatus = status
        }
    }

    func enableFanHelper() {
        guard !isUpdatingFanHelper else { return }
        isUpdatingFanHelper = true
        fanHelperMessage = nil
        Task { @MainActor in
            defer { isUpdatingFanHelper = false }
            do {
                let status = try await Task.detached(priority: .userInitiated) {
                    try QWSMCHelperManager.enableFromApp()
                }.value
                fanHelperStatus = status
                if status == .requiresApproval {
                    fanHelperMessage = "Approve Qwegn in System Settings -> Login Items to finish setup."
                } else {
                    telemetry.fetchDashboard()
                }
            } catch {
                fanHelperStatus = await Task.detached(priority: .utility) {
                    QWSMCHelperManager.currentStatus()
                }.value
                fanHelperMessage = error.localizedDescription
            }
        }
    }

    func disableFanHelper() {
        guard !isUpdatingFanHelper else { return }
        isUpdatingFanHelper = true
        fanHelperMessage = nil
        Task { @MainActor in
            defer { isUpdatingFanHelper = false }
            do {
                let status = try await Task.detached(priority: .userInitiated) {
                    try QWSMCHelperManager.disableFromApp()
                }.value
                fanHelperStatus = status
                telemetry.fetchDashboard()
            } catch {
                fanHelperStatus = await Task.detached(priority: .utility) {
                    QWSMCHelperManager.currentStatus()
                }.value
                fanHelperMessage = error.localizedDescription
            }
        }
    }

    func openLoginItemsSettings() {
        SMAppService.openSystemSettingsLoginItems()
    }

    var fanHelperDotColor: Color {
        switch fanHelperStatus {
        case .active: return TelemetryTheme.active
        case .requiresApproval, .installedNotRunning: return TelemetryTheme.warning
        case .notInstalled: return TelemetryTheme.unknown
        }
    }

    var fanHelperStatusLabel: String {
        switch fanHelperStatus {
        case .active: return "Fan Helper: Active (/var/run/qwg-smc.sock)"
        case .requiresApproval: return "Fan Helper: Approval needed in Login Items"
        case .installedNotRunning: return "Fan Helper: Installed (not running)"
        case .notInstalled: return "Fan Helper: Not installed"
        }
    }
}
