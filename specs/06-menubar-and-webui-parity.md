# 06 — Menu Bar (`@Observable`) & WebUI Telemetry Parity

> Normative specification for `QWMenuBarUI` (`Sources/Apps/QWMenuBar/`), `Qwegn.app` bundling, `QWSettingsCoherence`, and `/ui` parity.

## 1. Observation Architecture (`@Observable`)
- `TelemetryObservable` (`Sources/Apps/QWMenuBar/TelemetryObservable.swift`) is `@MainActor @Observable` (no `ObservableObject` or `@Published`), importing `Observation`.
- Internal timers, `Task` handles, `Process` handles, and stale-fetch tokens (`pollTimer`, `loggerSubscriptionId`, `serverProcess`, `lastKnownServerPid`, `nextLogSequence`, `isPopoverVisible`, `isLogWindowVisible`, `dashboardStreamTask`, `logStreamTask`, `lastDashboardStreamAt`, `dashboardGeneration`, `engineInfoGeneration`, `lastErrorPhase`) are annotated `@ObservationIgnored` so non-UI housekeeping never triggers SwiftUI invalidation.
- `init(startServices: Bool = true, enableAutoLifecycle: Bool = true, initialSettings: QWServerSettings? = nil)` supports hermetic unit testing (`startServices: false` / `enableAutoLifecycle: false`) while preserving zero-argument production startup (`TelemetryObservable()`).
- Views accept plain `var telemetry: TelemetryObservable` (`CupertinoTelemetryView`, `SettingsView`, `LogViewerWindow`) or `@Bindable var telemetry: TelemetryObservable` where `$telemetry.*` bindings are required (`SettingsMenuBarAppearanceView`).
- `QWMenuBarApp` updates `NSStatusItem` through `withObservationTracking` (`observeStatusItem()`), re-evaluating only when properties read by `menuBarStatusState` or `menuBarStatusTitle` mutate.

## 2. Single-Source Telemetry & Honesty Invariants
Both `Qwegn.app` and `QWServer` (`/ui` + `controlbar.html`) render from `QWDashboardViewModel` (`QWDashboardBuilder`):
- **Catalogue lockstep:** Every card ID read in `QWMenuBar` (`TelemetryCards.swift`) belongs to `MenuBarCardCatalogue.ids`; `/ui` cards belong to `QWDashboardBuilder.cardSpecs`.
- **Zero fabricated state:** Missing metrics render `—` with `card.reason` tooltips; UI code must never read `engineInfo` / `settings.modelName` for active model titles or synthesize `"Idle"` / `"Stopped"` fallbacks when disconnected (`MenuBarRendererParityTests` & `QWTelemetryHonestyTests`).
- **Lifecycle boundary & restart coherence:** `QWSettingsCoherence` classifies every setting into `.live` vs `.serverRestart`. Server header actions use `restartServerProcess()` / `toggleDaemon()`; model actions stay inside `RESIDENT MODEL`.

## 3. Co-Bundled Binary & Packaging
- `Package.swift` splits reusable views/state into `QWMenuBarUI` (`L8.5`, unit-tested by `QWMenuBarTests`) and keeps `qwg-menu` (`L9.0`, `Sources/Apps/QWMenuBarEntry`) as the thin `@main` executable.
- `Scripts/build_app.sh` embeds and code-signs release `qwg` inside `Qwegn.app/Contents/MacOS/qwg` so `TelemetryObservable` launches the matching signed engine sibling binary regardless of working directory.
