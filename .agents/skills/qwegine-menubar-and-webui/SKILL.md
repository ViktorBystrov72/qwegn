---
name: qwegine-menubar-and-webui
description: >-
  Use when designing, modifying, or visually validating Qwegn.app (QWMenuBar:
  CupertinoTelemetryView, TelemetryCards, SettingsView, LogViewerWindow,
  QGProcessorLogoView), the offline QWServer WebUI (/ui, controlbar.html,
  Chat, #modal-memory-pack, QWDashboardViewModel), or menu bar / dashboard parity
  and honesty tests.
---

# Qwegn Menu Bar (`Qwegn.app`) & Offline WebUI (`/ui`) Runbook

## 1. Control-Panel Hierarchy & Lifecycle Separation

1. **Dynamic State & Model Traits Only:** Show only live telemetry and active-model traits (`quantization`, `KV`, `ctx`, `Spec`, `MemPack`) on primary popover and `/ui` hero cards. Omit constant host specs (`Apple M5 Max`, Metal family, core count), duplicate `modelAlias`, and internal dispatch tags (`GDN chunked`) from the `440pt` popover; keep full runtime knob tables inside a closed-by-default disclosure drawer in `/ui`. Keep the popover inner stack clamped to `width: 420` followed by `.padding(10)` and `spacing: 8` inside `.frame(width: 440).clipped()`, give header/model button clusters `.fixedSize(horizontal: true, vertical: false).layoutPriority(1)`, render long resident model titles and compressible quantization pills via `MarqueeText` (and `/ui` `.hero-model-marquee` guarded by `title.dataset.sig` across `1 Hz` SSE ticks), and never apply unconstrained `.fixedSize(horizontal: true)` across overflowing pill rows (`QWMenuBarUITests`, `MenuBarRendererParityTests`, `DashboardRendererParityTests`).
2. **Truncation-Free Metric Cells & Auto-Collapsing Sparkline:** Render unavailable metrics as `—` / `— tok/s` and attach `card.reason` via `.help(...)` / `title` tooltips rather than inline subtitle text. Hide `THROUGHPUT TRACE` whenever both `prefillSeries` and `decodeSeries` contain zero non-`nil` samples; keep `nil` intact in `QWEngineTelemetryRegistry` while bridging active companion-phase `nil` slots cleanly in `QWSparklineGeometry` and `js/state.js` (`MenuBarRendererParityTests`, `DashboardRendererParityTests`).
3. **Daemon vs. Model Lifecycle Boundary:** Place `qwg` server lifecycle controls (`Restart` → `restartServerProcess()` / `⌘R`, `Stop`/`Start` → `toggleDaemon()`) in the popover header, and keep model weight lifecycle (`Switch`, `Unload`/`Load`) inside `RESIDENT MODEL`.

## 2. macOS 26 AppKit / SwiftUI Glass & Bundle Invariants

1. **AppKit `NSMenu` Bridge Inside `GlassEffectContainer`:** On macOS 26, SwiftUI `Menu { ... }.menuStyle(.borderlessButton)` collapses invisible inside `GlassEffectContainer` + `.tint(.clear)`. Render dropdown triggers as `.buttonStyle(.plain)` SwiftUI `Button` views backed by an `NSViewRepresentable` anchor (`MenuAnchorBridge`) that calls `NSMenu.popUp(positioning:at:in:)` with `objc_setAssociatedObject`-retained action targets.
2. **Single-Control Transition Spinners:** Pass `busy: false` (while keeping `disabled: daemonBusy || modelBusy`) on secondary header buttons such as `Restart` so daemon transitions animate only the primary `Stop`/`Start` button.
3. **Co-Bundled `qwg` Sibling Binary (`Scripts/build_app.sh`):** Bundle and codesign release `qwg` at `Qwegn.app/Contents/MacOS/qwg` and resolve `Bundle.main` sibling `qwg` first in `TelemetryObservable.findQWGBinary()` so Finder launches (`cwd = "/"`) execute the exact fresh release binary (`AGENTS.md` Law 7).

## 3. Telemetry Catalogue & Honesty Gates (`QWServerTests`)

1. **Catalogue Registration:** Register every `card("<id>")` read in `QWMenuBar` inside `MenuBarCardCatalogue.ids` (`QWDashboardViewModel.swift`), and every `/ui` `data-card="<id>"` inside `QWDashboardBuilder.specs`.
2. **Zero Client-Invented Telemetry & Emblem Parity:** Derive all UI values from `QWDashboardViewModel` (`QWMenuBarStatusFormatter` for selectable menu bar metrics). Keep `TelemetryCards.swift`, `TelemetryObservable.swift`, and the modular `/ui` assets (`Sources/Apps/QWServer/Resources/{controlbar.html,css/{base,panels,chat}.css},js/{state,dashboard,markdown,chat-presets,chat,actions,settings,updates}.js}`) free of `engineInfo` / `settings.modelName` UI reads, hardcoded chip/model/bandwidth literals, `"https://"` CDN references, or `"Idle"` / `"Stopped"` fallback strings (`MenuBarRendererParityTests`, `DashboardRendererParityTests`, `QWTelemetryHonestyTests`). Keep all WebUI files `<= 800` lines (including parity test suites in `Tests/`), copy nested `css/` and `js/` directories via `Package.swift`, and keep all 4 `QWG` vector emblem files (`QGProcessorLogoView.swift`, `gen_app_icon.swift`, `logo.svg`, `controlbar.html`) synchronized (`x2="106"` left bus pins).
3. **CLI–GUI–WebUI Knob Parity Gate:** When adding or modifying a CLI runtime/tuning knob, declare its `.menuBar`/`.api`/`.profile`/`.dashboard` surfaces in `Sources/Foundation/QWCore/QWKnobRegistry.swift`, verify `Scripts/gen_knob_cli.py --stdout` matches `Sources/Apps/qwg/Generated/KnobOptions.swift`, wire `QWServerConfigKnobs.swift` + `QWResolvedServerConfig.swift`, expose controls in both `QWMenuBar` (`SettingsView+Performance.swift` / `SettingsView+Sections.swift`) and `/ui` (`Sources/Apps/QWServer/Resources/js/settings.js`), and verify `Tests/QWServerTests/QWCliGuiWebParityTests.swift` passes.

## 4. Mandatory `2×` Retina Offscreen & Live `/ui` Verification

1. **TCC-Safe `2×` Retina `NSHostingView` Snapshots:** Because macOS TCC blocks headless `screencapture`, compile a scratch Swift AppKit renderer against `.build/out/Intermediates.noindex/qwegine.build/Debug/qwg-menu-p.build/Objects-normal/arm64/qwg-menu.LinkFileList` (filtering out `QWMenuBarMain.o`, expanding `${=OBJ_LIST}` in `zsh`), set `telemetry.isDaemonTransitioning = false` after `TelemetryObservable()`, rasterize idle + active `CupertinoTelemetryView` (`880px` wide) and touched `SettingsView` / `LogViewerWindow` tabs at `2×` Retina via `bitmapImageRepForCachingDisplay` + `cacheDisplay`, inspect every PNG with `read_file` for invisible controls, `...` truncation, or hairline gaps, and remove scratch artifacts before finishing.
2. **Live `/ui` & `.qwcache` Modal Verification:** When `/ui` assets (`controlbar.html`, `css/*`, `js/*`) or `QWServer` endpoints change, validate `http://127.0.0.1:8080/ui` with the `browser` subagent across all 5 tabs (`Monitor`, `Chat`, `Models`, `Live Logs`, `Settings`) at desktop (`1440×900`) and narrow (`900×800`) viewports, confirming zero console/page/HTTP>=400 errors and `scrollWidth <= clientWidth`.
3. **Deterministic `.qwcache` Fixtures & Rendered State Checks:** When testing `computeDelta` or `#modal-memory-pack` (`missing`, `wrong_model`, `stale`, `ready`):
   - Change file byte count and advance `modificationDate` by `>= 2 s` in staleness tests so same-second `(mtimeEpochSec, byteCount)` fast-paths never flake.
   - Seed scratch `manifest.json` `CacheArtifact` entries with `createdAtEpochSec` (`Int64`, not ISO-8601 strings) so `QWMemoryManifest.load` succeeds.
   - Verify every modal branch by checking the live rendered DOM badge (`No .qwcache`, `Different Model`, `Stale Index`, `Ready`) and `BROWSER_HOST_FACTS` screenshots rather than accepting source-inspection fallback claims.
