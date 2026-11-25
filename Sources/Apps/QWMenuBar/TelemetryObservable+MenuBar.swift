// TelemetryObservable+MenuBar.swift — Customizable menu bar telemetry persistence and title

import Foundation
import QWServer

extension TelemetryObservable {
    private static let menuBarMetricsDefaultsKey = "qwg.ui.menuBarMetrics"
    private static let hideSpeedsWhenIdleDefaultsKey = "qwg.ui.hideMenuBarSpeedsWhenIdle"
    private static let legacyShowThroughputDefaultsKey = "qwg.ui.showMenuBarThroughput"

    static func loadEnabledMenuBarMetrics() -> Set<QWMenuBarMetric> {
        let defaults = UserDefaults.standard
        if let rawList = defaults.stringArray(forKey: menuBarMetricsDefaultsKey) {
            return Set(rawList.compactMap(QWMenuBarMetric.init(rawValue:)))
        }
        if let legacy = defaults.object(forKey: legacyShowThroughputDefaultsKey) as? Bool {
            return legacy ? [.decodeSpeed] : []
        }
        return [.decodeSpeed]
    }

    static func loadHideMenuBarSpeedsWhenIdle() -> Bool {
        UserDefaults.standard.object(forKey: hideSpeedsWhenIdleDefaultsKey) as? Bool ?? false
    }

    func persistMenuBarMetrics() {
        let ordered = QWMenuBarMetric.allCases
            .filter { enabledMenuBarMetrics.contains($0) }
            .map(\.rawValue)
        UserDefaults.standard.set(ordered, forKey: Self.menuBarMetricsDefaultsKey)
        UserDefaults.standard.set(
            enabledMenuBarMetrics.contains(.decodeSpeed) || enabledMenuBarMetrics.contains(.prefillSpeed),
            forKey: Self.legacyShowThroughputDefaultsKey
        )
    }

    func persistHideMenuBarSpeedsWhenIdle() {
        UserDefaults.standard.set(hideMenuBarSpeedsWhenIdle, forKey: Self.hideSpeedsWhenIdleDefaultsKey)
    }

    /// Full status item readout matching the user's selected menu bar metrics.
    public var menuBarStatusTitle: String? {
        QWMenuBarStatusFormatter.statusTitle(
            dashboard: dashboard,
            isConnected: isConnected,
            enabledMetrics: enabledMenuBarMetrics,
            hideSpeedsWhenIdle: hideMenuBarSpeedsWhenIdle
        )
    }

    public func isMenuBarMetricEnabled(_ metric: QWMenuBarMetric) -> Bool {
        enabledMenuBarMetrics.contains(metric)
    }

    public func setMenuBarMetric(_ metric: QWMenuBarMetric, enabled: Bool) {
        if enabled {
            enabledMenuBarMetrics.insert(metric)
        } else {
            enabledMenuBarMetrics.remove(metric)
        }
    }

    public func unloadDecisionSidecar() {
        guard let req = authorizedRequest(path: "/v1/decisions/model/unload", method: "POST", timeoutInterval: 15.0) else { return }
        Task {
            do {
                let (data, res) = try await URLSession.shared.data(for: req)
                guard let http = res as? HTTPURLResponse, http.statusCode == 200 else {
                    let code = (res as? HTTPURLResponse)?.statusCode ?? 0
                    self.fail("Sidecar unload failed (HTTP \(code)): \(Self.responseDetail(data))")
                    return
                }
                self.fetchDashboard()
                QWServerLogger.shared.log(.info, "Decision sidecar unloaded via API.")
            } catch {
                self.fail("Sidecar unload failed: \(error.localizedDescription)")
            }
        }
    }
}
