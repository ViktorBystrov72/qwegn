// SettingsView.swift — Sidebar-based native macOS settings window

import AppKit
import QWCore
import QWSMCHelper
import QWServer
import ServiceManagement
import SwiftUI

/// Sidebar-navigated Settings window covering general lifecycle, models, sampling, speculative, performance, memory, and logging.
public struct SettingsView: View {
    public var telemetry: TelemetryObservable
    /// Increased Contrast asks for stronger text. L-M7: it is honoured.
    @Environment(\.colorSchemeContrast) internal var contrast: ColorSchemeContrast
    @State var draftSettings: QWServerSettings
    @State var baselineSettings: QWServerSettings
    @State var activeTab: SettingsTab = .general
    @State var saveConfirmationMessage: String?
    @State var validationErrorMessage: String?
    @State var discoveredModels: [DiscoveredModel] = []
    @State var discoveredDrafterModels: [DiscoveredModel] = []
    @State var pendingProfileReset: Bool = false
    @State var isGeneratingLSASidecar: Bool = false
    @State var lsaSidecarTask: Task<Void, Never>?
    @State var isSaving: Bool = false
    @State var fanHelperStatus: QWSMCHelperStatus = .notInstalled
    @State var isUpdatingFanHelper: Bool = false
    @State var fanHelperMessage: String?
    public var onClose: () -> Void
    public var onOpenUpdates: (Bool) -> Void

    public enum SettingsTab: String, CaseIterable, Identifiable {
        case general = "General"
        case model = "Model"
        case sampling = "Sampling"
        case speculative = "Speculative"
        case performance = "Performance"
        case cache = "Memory & Cache"
        case logging = "Logging"

        public var id: String { rawValue }

        public var icon: String {
            switch self {
            case .general: return "gearshape"
            case .model: return "cpu"
            case .sampling: return "slider.horizontal.below.rectangle"
            case .speculative: return "bolt.badge.clock"
            case .performance: return "gauge.with.dots.needle.67percent"
            case .cache: return "internaldrive"
            case .logging: return "doc.text"
            }
        }
    }

    public init(
        telemetry: TelemetryObservable,
        initialTab: SettingsTab = .general,
        onClose: @escaping () -> Void = {},
        onOpenUpdates: @escaping (Bool) -> Void = { _ in }
    ) {
        self.telemetry = telemetry
        self._draftSettings = State(initialValue: telemetry.settings)
        self._baselineSettings = State(initialValue: telemetry.settings)
        self._activeTab = State(initialValue: initialTab)
        self.onClose = onClose
        self.onOpenUpdates = onOpenUpdates
    }

    var isDirty: Bool {
        draftSettings != baselineSettings || pendingProfileReset
    }

    var increasedContrast: Bool {
        contrast == .increased
    }

    public var body: some View {
        ZStack {
            SystemVibrancyView()
                .edgesIgnoringSafeArea(.all)

            HStack(spacing: 0) {
                sidebarPane
                Divider()
                detailPane
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            openOrRefreshIfClean()
        }
        .onDisappear {
            lsaSidecarTask?.cancel()
            lsaSidecarTask = nil
        }
        .onReceive(NotificationCenter.default.publisher(for: .qwgSettingsWindowDidOpen)) { _ in
            openOrRefreshIfClean()
        }
    }

    private var sidebarPane: some View {
        VStack(alignment: .leading, spacing: 0) {
            List(SettingsTab.allCases, selection: $activeTab) { tab in
                Label(tab.rawValue, systemImage: tab.icon)
                    .font(TelemetryTheme.ui(12, activeTab == tab ? .semibold : .regular))
                    .tag(tab)
            }
            .listStyle(.sidebar)
            .scrollContentBackground(.hidden)

            Divider()

            VStack(alignment: .leading, spacing: 4) {
                Text("~/.qwg/settings.json")
                    .font(TelemetryTheme.numeric(10))
                    .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Button(action: revealSettingsInFinder) {
                    Label("Reveal in Finder", systemImage: "folder")
                        .font(TelemetryTheme.ui(11))
                }
                .buttonStyle(.plain)
                .foregroundStyle(TelemetryTheme.secondary(increasedContrast: increasedContrast))
            }
            .padding(12)
        }
        .frame(width: 185)
        .frame(maxHeight: .infinity)
        .background(TelemetryTheme.rowSubstrate)
    }

    private var detailPane: some View {
        VStack(spacing: 0) {
            detailHeader
            Divider()

            ScrollView(.vertical, showsIndicators: true) {
                VStack(alignment: .leading, spacing: 14) {
                    pendingRestartPanel
                    switch activeTab {
                    case .general:
                        generalSection
                    case .model:
                        modelSection
                    case .sampling:
                        samplingSection
                    case .speculative:
                        speculativeSection
                    case .performance:
                        performanceSection
                    case .cache:
                        cacheSection
                    case .logging:
                        loggingSection
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(16)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            Divider()
            bottomActionBar
        }
        .frame(minWidth: 0, maxWidth: .infinity, maxHeight: .infinity)
        .clipped()
    }

    private var detailHeader: some View {
        HStack(spacing: 8) {
            Image(systemName: activeTab.icon)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(TelemetryTheme.accent)
            Text(activeTab.rawValue)
                .font(TelemetryTheme.ui(13, .semibold))
                .foregroundStyle(TelemetryTheme.primary(increasedContrast: increasedContrast))
            Spacer()
            if isDirty {
                Text("Unsaved Changes")
                    .font(TelemetryTheme.ui(10, .semibold))
                    .foregroundStyle(TelemetryTheme.warning)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 2)
                    .background(TelemetryTheme.warning.opacity(0.14))
                    .clipShape(Capsule())
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private var bottomActionBar: some View {
        let needsRestart = !telemetry.pendingServerRestartKeys(for: draftSettings).isEmpty
        return HStack(spacing: 10) {
            Button("Reset to Defaults") {
                resetToTopPerformanceDefaults()
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .help("Stage top-performance default parameters for the selected model")

            if let err = validationErrorMessage {
                Text(err)
                    .font(TelemetryTheme.ui(11, .medium))
                    .foregroundStyle(TelemetryTheme.failure)
                    .lineLimit(2)
                    .help(err)
            } else if let msg = saveConfirmationMessage {
                Text(msg)
                    .font(TelemetryTheme.ui(11, .medium))
                    .foregroundStyle(TelemetryTheme.active)
                    .lineLimit(1)
            }

            Spacer(minLength: 8)

            Button("Discard Changes") {
                discardChanges()
            }
            .buttonStyle(.bordered)
            .controlSize(.regular)
            .disabled(!isDirty || isSaving)

            if isDirty && needsRestart && telemetry.isConnected {
                Button(isSaving ? "Saving…" : "Save") {
                    applyAndSave()
                }
                .buttonStyle(.bordered)
                .controlSize(.regular)
                .disabled(validationErrorMessage != nil || isSaving)

                Button("Save & Restart Server") {
                    saveAndRestartServer()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.regular)
                .disabled(validationErrorMessage != nil || isSaving)
            } else {
                Button(isSaving ? "Saving…" : "Save & Apply") {
                    applyAndSave()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.regular)
                .disabled(!isDirty || validationErrorMessage != nil || isSaving)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(TelemetryTheme.rowSubstrate)
    }

    private func openOrRefreshIfClean() {
        let wasClean = !isDirty
        if wasClean {
            draftSettings = telemetry.settings
        }
        refreshDiscoveredModels()
        if wasClean {
            baselineSettings = draftSettings
        }
    }

    // MARK: - 1. General & System Section
    private var generalSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            settingsGroup(title: "System Integration", icon: "macwindow") {
                settingsToggleRow("Run at startup", isOn: $draftSettings.runAtStartup)
                settingsToggleRow("Auto-start inference server on launch", isOn: $draftSettings.autoStartServer)
            }

            SettingsMenuBarAppearanceView(telemetry: telemetry)

            settingsGroup(title: "Server & App Lifecycle", icon: "server.rack") {
                settingsToggleRow(
                    "Keep the server running after the app quits",
                    key: "keepServerRunningAfterQuit",
                    isOn: $draftSettings.keepServerRunningAfterQuit
                )
                Text(QWSettingsCoherence.effectNote(for: "keepServerRunningAfterQuit") ?? "")
                    .font(TelemetryTheme.ui(11))
                    .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
            }

            softwareUpdatesGroup

            settingsGroup(title: "Network & Endpoints", icon: "network") {
                formRow(label: "Listen Host:", key: "host") {
                    TextField("127.0.0.1", text: $draftSettings.host)
                        .textFieldStyle(.roundedBorder)
                }
                formRow(label: "Listen Port:", key: "port") {
                    TextField("8080", value: $draftSettings.port, formatter: plainNumberFormatter)
                        .textFieldStyle(.roundedBorder)
                }
                formRow(label: "Poll Interval (ms):", key: "telemetryPollIntervalMs") {
                    TextField("500", value: $draftSettings.telemetryPollIntervalMs, formatter: plainNumberFormatter)
                        .textFieldStyle(.roundedBorder)
                }
            }
        }
    }

    // MARK: - 2. Model & Weights Section
    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            settingsGroup(title: "Model Selection (~/.qwg/models)", icon: "cpu") {
                formRow(label: "Installed Model:", key: "modelName") {
                    HStack(spacing: 8) {
                        Picker("", selection: $draftSettings.modelName) {
                            ForEach(discoveredModels.filter(\.supported)) { m in
                                Text(m.name).tag(m.name)
                            }
                        }
                        .labelsHidden()
                        .pickerStyle(.menu)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .onChange(of: draftSettings.modelName) { _, newName in
                            if let match = discoveredModels.first(where: { $0.name == newName }), !match.packPath.isEmpty {
                                draftSettings.modelPackPath = match.packPath
                            }
                            let (effective, _) = QWSettingsManager.shared.effectiveSettings(
                                forModel: newName,
                                packPath: draftSettings.modelPackPath,
                                base: draftSettings
                            )
                            draftSettings = effective
                            validateModelAndDrafterOnSelection()
                        }

                        Button(action: refreshDiscoveredModels) {
                            Image(systemName: "arrow.clockwise")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .accessibilityLabel("Refresh installed models")
                        .help("Refresh models in ~/.qwg/models")

                        Button(action: revealModelsFolderInFinder) {
                            Image(systemName: "folder")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .accessibilityLabel("Open models folder in Finder")
                        .help("Open ~/.qwg/models in Finder")
                    }
                }

                VStack(alignment: .leading, spacing: 4) {
                    formRow(label: "Model Alias:", key: "modelAlias") {
                        TextField("e.g. qwen-27b (optional)", text: Binding(
                            get: { draftSettings.modelAlias },
                            set: { draftSettings.setAlias($0, for: draftSettings.modelName) }
                        ))
                        .textFieldStyle(.roundedBorder)
                    }
                    Text("Persisted per model in ~/.qwg/settings.json and returned by /v1/models.")
                        .font(TelemetryTheme.ui(11))
                        .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
                        .fixedSize(horizontal: false, vertical: true)
                }

                VStack(alignment: .leading, spacing: 4) {
                    SettingsRowLabel(text: "Model Pack (.qwp path):", key: "modelPackPath")
                    HStack(spacing: 8) {
                        TextField("/path/to/model.qwp", text: $draftSettings.modelPackPath)
                            .textFieldStyle(.roundedBorder)
                        Button("Browse...") {
                            selectModelPackFile()
                        }
                    }
                }

                settingsToggleRow(
                    "Lazy Load Model Weights on First Request",
                    key: "lazyLoad",
                    isOn: $draftSettings.lazyLoad
                )

                VStack(alignment: .leading, spacing: 4) {
                    settingsToggleRow(
                        "Load Vision Tower (Multimodal Image Input)",
                        key: "loadVision",
                        isOn: $draftSettings.loadVision
                    )
                    .disabled(!selectedModelSupportsVision)
                    Text(selectedModelSupportsVision
                        ? (QWSettingsCoherence.effectNote(for: "loadVision")
                            ?? "Loads or unloads the vision tower in place without restarting the engine.")
                        : "Selected model does not include a vision tower.")
                        .font(TelemetryTheme.ui(11))
                        .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
                }
            }

            settingsGroup(title: "Decision Sidecar (Dual-Slot)", icon: "bolt.circle") {
                VStack(alignment: .leading, spacing: 4) {
                    SettingsRowLabel(text: "Decision Model Pack / .aimodel:", key: "decisionModelPackPath")
                    TextField("~/.qwg/models/<decision-pack> or .aimodel", text: $draftSettings.decisionModelPackPath)
                        .textFieldStyle(.roundedBorder)
                    if let note = QWSettingsCoherence.effectNote(for: "decisionModelPackPath") {
                        Text(note)
                            .font(TelemetryTheme.ui(11))
                            .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
                    }
                }

                formRow(label: "Compute Unit:", key: "decisionComputeUnit") {
                    Picker("", selection: $draftSettings.decisionComputeUnit) {
                        Text("Auto").tag("auto")
                        Text("Metal 4").tag("metal")
                        Text("ANE").tag("ane")
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                    .frame(maxWidth: .infinity)
                }

                settingsToggleRow(
                    "Load Decision Vision Tower",
                    key: "decisionLoadVision",
                    isOn: $draftSettings.decisionLoadVision
                )
            }

            settingsGroup(title: "Context & KV Format", icon: "slider.horizontal.3") {
                ContextLimitSliderRow(
                    maxContext: $draftSettings.maxContext,
                    estimate: selectedModelContextMemoryEstimate,
                    modelName: draftSettings.modelName,
                    kvMode: mountedMemoryPack?.enforcedKVMode ?? draftSettings.kvMode,
                    contrast: contrast
                )

                VStack(alignment: .leading, spacing: 4) {
                    formRow(label: "KV Cache Mode:", key: "kvMode") {
                        Picker("", selection: Binding(
                            get: { mountedMemoryPack?.enforcedKVMode ?? draftSettings.kvMode },
                            set: { newVal in
                                if !isKVModeLockedByMemoryPack {
                                    draftSettings.kvMode = newVal
                                }
                            }
                        )) {
                            Text("FP16").tag("f16")
                            Text("Q8_0 (8-bit)").tag("q8")
                            Text("Q4_0 (4-bit)").tag("q4")
                        }
                        .labelsHidden()
                        .pickerStyle(.segmented)
                        .frame(maxWidth: .infinity)
                        .disabled(isKVModeLockedByMemoryPack)
                    }

                    if let mounted = mountedMemoryPack, mounted.enforcesKVMode {
                        HStack(spacing: 6) {
                            Image(systemName: "lock.fill")
                                .font(.system(size: 10))
                                .foregroundStyle(TelemetryTheme.warning)
                            Text("Locked by Memory Pack (\(mounted.packName) · \(mounted.enforcedKVMode.uppercased()))")
                                .font(TelemetryTheme.ui(10, .medium))
                                .foregroundStyle(TelemetryTheme.warning)
                            Spacer()
                            Button("Unmount") {
                                unmountActiveMemoryPack()
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.mini)
                        }
                    }
                }
            }
        }
    }
}
