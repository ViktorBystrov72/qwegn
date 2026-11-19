// SettingsView+Actions.swift — the SettingsView actions.
import AppKit
import ServiceManagement
import SwiftUI
import QWCore
import QWQuant
import QWServer

extension SettingsView {

    // MARK: - Actions

    func refreshDiscoveredModels() {
        telemetry.refreshDiscoveredModelsAsync { scanned, drafters in
            applyDiscoveredModels(scanned: scanned, drafters: drafters)
        }
    }

    private func applyDiscoveredModels(scanned: [DiscoveredModel], drafters: [DiscoveredModel]) {
        discoveredModels = scanned
        // The configured pack is the model the engine runs. The name the
        // server serves it under can be an id of the user's choosing, not
        // a directory name, so a name that matches no installed pack does
        // not mean the model is gone.
        if !discoveredModels.contains(where: { $0.name == draftSettings.modelName }) {
            let expandedPack = draftSettings.modelPackPath.expandingTildePath
            let inModelsFolder = expandedPack.isEmpty || expandedPack.hasPrefix(QWPaths.modelsDirectoryPath)
            if let configured = discoveredModels.first(where: { $0.packPath == expandedPack }) {
                draftSettings.modelAlias = draftSettings.alias(for: draftSettings.modelName)
                discoveredModels.insert(DiscoveredModel(name: draftSettings.modelName, packPath: configured.packPath), at: 0)
            } else if inModelsFolder, let first = discoveredModels.first(where: \.supported) {
                draftSettings.modelName = first.name
                draftSettings.modelPackPath = first.packPath
                draftSettings.modelAlias = draftSettings.alias(for: first.name)
            } else if !draftSettings.modelName.isEmpty {
                discoveredModels.insert(DiscoveredModel(name: draftSettings.modelName, packPath: draftSettings.modelPackPath), at: 0)
            }
        } else {
            draftSettings.modelAlias = draftSettings.alias(for: draftSettings.modelName)
        }
        discoveredDrafterModels = drafters
        let trimmedDraft = draftSettings.dflashDraftPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedDraft.isEmpty
            && !discoveredDrafterModels.contains(where: { $0.packPath == trimmedDraft }) {
            let name = QWModelRoot.canonicalModelName(for: URL(fileURLWithPath: trimmedDraft))
            discoveredDrafterModels.insert(DiscoveredModel(name: name, packPath: trimmedDraft), at: 0)
        } else if isExternalDrafterSelected, trimmedDraft.isEmpty,
                  let match = QWServerSettings.resolveDefaultDrafterPack(
                      for: selectedPrimaryDescriptor,
                      preferredBackend: draftSettings.mtpBackend,
                      drafters: discoveredDrafterModels,
                      targetModelName: draftSettings.modelName
                  ) {
            draftSettings.dflashDraftPackPath = match.packPath
        }
    }

    private func chooseFilePanel(
        canChooseFiles: Bool,
        canChooseDirectories: Bool,
        message: String? = nil
    ) -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseFiles = canChooseFiles
        panel.canChooseDirectories = canChooseDirectories
        panel.allowsMultipleSelection = false
        if let message { panel.message = message }
        return panel.runModal() == .OK ? panel.url : nil
    }

    private func resolvedSelectedModelReference() -> (packPath: String, modelName: String, alias: String) {
        let rawPath = draftSettings.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        return draftSettings.resolveModelReference(
            rawPath.isEmpty ? draftSettings.modelName : rawPath,
            discovered: discoveredModels
        )
    }

    func selectDrafterPackFile() {
        guard let url = chooseFilePanel(canChooseFiles: true, canChooseDirectories: true) else { return }
        let path = url.path
        let name = QWModelRoot.canonicalModelName(for: url)
        Task { @MainActor in
            let isValidDrafter = await Task.detached(priority: .userInitiated) {
                QWModelCatalog.isDrafterPack(at: path)
            }.value
            guard isValidDrafter else {
                validationErrorMessage = "Cannot select '\(name)': not a DFlash2 or DSpine drafter pack"
                return
            }
            draftSettings.dflashDraftPackPath = path
            telemetry.invalidateModelMetadataCache()
            refreshDiscoveredModels()
            validateModelAndDrafterOnSelection()
        }
    }

    func revealModelsFolderInFinder() {
        let url = QWPaths.modelsDirectoryURL
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: url.path)
    }

    func selectModelPackFile() {
        guard let url = chooseFilePanel(canChooseFiles: true, canChooseDirectories: true) else { return }
        let path = url.path
        let baseName = QWModelRoot.canonicalModelName(for: url)
        Task { @MainActor in
            let entry = await Task.detached(priority: .userInitiated) {
                QWModelCatalog.classify(name: baseName, path: path)
            }.value
            guard entry.supported else {
                validationErrorMessage = "Cannot select '\(baseName)': \(entry.reason ?? "not runnable")"
                return
            }
            draftSettings.modelPackPath = path
            if !baseName.isEmpty {
                draftSettings.modelName = baseName
            }
            telemetry.invalidateModelMetadataCache()
            refreshDiscoveredModels()
            validateModelAndDrafterOnSelection()
        }
    }

    var selectedModelSupportsVision: Bool {
        if let disc = discoveredModels.first(where: { $0.name == draftSettings.modelName }),
           !disc.capabilities.isEmpty {
            return disc.capabilities.contains("vision")
        }
        if let desc = selectedPrimaryDescriptor {
            return desc.visDepth > 0 && desc.capabilities.contains(.vision)
        }
        return true
    }

    var selectedPrimaryDescriptor: QWModelDescriptor? {
        let rawPath = draftSettings.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let ref = resolvedSelectedModelReference()
        let candidate = ref.packPath.isEmpty ? rawPath : ref.packPath
        return telemetry.cachedPrimaryDescriptor(packPath: candidate)
    }

    var selectedModelSupportsDFlash2: Bool {
        Self.supportsDFlash2(
            for: draftSettings,
            descriptor: selectedPrimaryDescriptor,
            discoveredModels: discoveredModels,
            discoveredDrafters: discoveredDrafterModels
        )
    }

    var selectedModelSupportsDFlash2Backend: Bool {
        Self.supportsExternalDrafter(
            backend: "dflash2",
            for: draftSettings,
            descriptor: selectedPrimaryDescriptor,
            discoveredModels: discoveredModels,
            discoveredDrafters: discoveredDrafterModels
        )
    }

    var selectedModelSupportsDSpineBackend: Bool {
        Self.supportsExternalDrafter(
            backend: "dspine",
            for: draftSettings,
            descriptor: selectedPrimaryDescriptor,
            discoveredModels: discoveredModels,
            discoveredDrafters: discoveredDrafterModels
        )
    }

    static func supportsDFlash2(
        for settings: QWServerSettings,
        descriptor: QWModelDescriptor?,
        discoveredModels: [DiscoveredModel],
        discoveredDrafters: [DiscoveredModel]
    ) -> Bool {
        let disc = discoveredModels.first(where: { $0.name == settings.modelName })
        return QWServerSettings.supportsDFlash2(
            descriptor: descriptor,
            discoveredModel: disc,
            drafters: discoveredDrafters
        )
    }

    static func supportsExternalDrafter(
        backend: String,
        for settings: QWServerSettings,
        descriptor: QWModelDescriptor?,
        discoveredModels: [DiscoveredModel],
        discoveredDrafters: [DiscoveredModel]
    ) -> Bool {
        let disc = discoveredModels.first(where: { $0.name == settings.modelName })
        return QWServerSettings.supportsExternalDrafter(
            backend: backend,
            descriptor: descriptor,
            discoveredModel: disc,
            drafters: discoveredDrafters,
            configuredPackPath: settings.dflashDraftPackPath
        )
    }

    var selectableDrafterModels: [DiscoveredModel] {
        Self.selectableDrafterModels(
            for: draftSettings,
            descriptor: selectedPrimaryDescriptor,
            discoveredDrafters: discoveredDrafterModels
        )
    }

    static func selectableDrafterModels(
        for settings: QWServerSettings,
        descriptor: QWModelDescriptor?,
        discoveredDrafters: [DiscoveredModel]
    ) -> [DiscoveredModel] {
        guard let descriptor else { return discoveredDrafters }
        let compatible = QWServerSettings.compatibleDrafterModels(for: descriptor, in: discoveredDrafters)
        guard !compatible.isEmpty else { return discoveredDrafters }
        let current = settings.dflashDraftPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if !current.isEmpty,
           !compatible.contains(where: { $0.packPath == current }),
           let custom = discoveredDrafters.first(where: { $0.packPath == current }) {
            return [custom] + compatible
        }
        return compatible
    }

    var isExternalDrafterSelected: Bool {
        let lower = draftSettings.mtpBackend.lowercased()
        return lower == "dflash2" || lower == "dspine"
    }

    var selectedModelMaxDrafts: Int {
        if isExternalDrafterSelected {
            let effPath = QWServerSettings.effectiveDrafterPackPath(
                configuredPath: draftSettings.dflashDraftPackPath,
                for: selectedPrimaryDescriptor,
                preferredBackend: draftSettings.mtpBackend,
                drafters: discoveredDrafterModels,
                targetModelName: draftSettings.modelName
            ) ?? draftSettings.dflashDraftPackPath
            if let cached = telemetry.cachedMaxDrafts(forDrafterPackPath: effPath) {
                return draftSettings.mtpBackend.lowercased() == "dspine" ? max(15, cached) : cached
            }
            return draftSettings.mtpBackend.lowercased() == "dspine" ? 15 : 7
        }
        if let desc = selectedPrimaryDescriptor,
           desc.maxSpeculativeDrafts > 0 {
            return desc.maxSpeculativeDrafts
        }
        if let disc = discoveredModels.first(where: { $0.name == draftSettings.modelName }), disc.maxSpeculativeDrafts > 0 {
            return disc.maxSpeculativeDrafts
        }
        return 7
    }

    private func resolvedLSASidecarPaths() -> (packPath: String, explicitPath: String) {
        let ref = resolvedSelectedModelReference()
        let explicit = draftSettings.lsaSidecarPath.trimmingCharacters(in: .whitespacesAndNewlines)
        return (ref.packPath, explicit)
    }

    var selectedModelHasLSASidecar: Bool {
        let paths = resolvedLSASidecarPaths()
        return telemetry.cachedLSASidecarStatus(packPath: paths.packPath, explicitPath: paths.explicitPath)?.hasSidecar ?? false
    }

    var selectedModelLSASidecarStatusText: String {
        let paths = resolvedLSASidecarPaths()
        return telemetry.cachedLSASidecarStatus(packPath: paths.packPath, explicitPath: paths.explicitPath)?.statusText
            ?? "No .qlsa sidecar detected (Auto uses in-memory Weight-SVD)"
    }

    func selectLSASidecarFile() {
        guard let url = chooseFilePanel(canChooseFiles: true, canChooseDirectories: false) else { return }
        draftSettings.lsaSidecarPath = url.path
        telemetry.invalidateLSASidecarCache()
        validateModelAndDrafterOnSelection()
    }

    func generateLSASidecarForSelectedModel() {
        guard !isGeneratingLSASidecar else { return }
        lsaSidecarTask?.cancel()
        validationErrorMessage = nil

        let packPath = resolvedSelectedModelReference().packPath
        let clusterBlock = draftSettings.lsaClusterBlock
        let budget = draftSettings.lsaBudget
        let snapshot = draftSettings
        let cachedDesc = selectedPrimaryDescriptor
        isGeneratingLSASidecar = true

        lsaSidecarTask = Task { @MainActor in
            defer {
                isGeneratingLSASidecar = false
                lsaSidecarTask = nil
            }
            do {
                let outcome: (URL, Int) = try await Task.detached(priority: .userInitiated) {
                    let desc = try cachedDesc ?? snapshot.validatePrimaryModel()
                    let indexDim = QWLSAConfig.defaultIndexDim
                    let tensors = try QWLSASidecar.synthesizeFromModel(
                        packPath: packPath,
                        descriptor: desc,
                        indexDim: indexDim
                    )
                    try Task.checkCancellation()
                    let root = QWModelRoot(url: URL(fileURLWithPath: packPath))
                    let outURL = root.directory.appendingPathComponent("model.qlsa")
                    let prov = QWLSASidecar.Header.Provenance(
                        modelId: root.directory.lastPathComponent,
                        geometryDigest: desc.geometryDigestV3Hex,
                        method: "weight_svd",
                        indexDim: indexDim,
                        poolSize: QWLSAConfig.poolSize,
                        clusterBlockSize: clusterBlock,
                        bucketsPerBlock: QWLSAConfig.bucketsPerBlock,
                        maxBucketKeys: QWLSAConfig.maxBucketKeys,
                        recommendedBudget: budget
                    )
                    let shape = [indexDim, desc.lsaInputDim]
                    try QWLSASidecar.write(
                        to: outURL,
                        provenance: prov,
                        layerTensors: tensors,
                        wqShape: shape,
                        wkShape: shape
                    )
                    return (outURL, tensors.count)
                }.value

                guard !Task.isCancelled else { return }
                telemetry.invalidateLSASidecarCache()
                refreshDiscoveredModels()
                validationErrorMessage = nil
                showTimedConfirmation("Built \(outcome.0.lastPathComponent) (\(outcome.1) layers)")
            } catch is CancellationError {
                validationErrorMessage = nil
            } catch {
                validationErrorMessage = "LSA sidecar build failed: \(error.localizedDescription)"
            }
        }
    }

    func cancelLSASidecarGeneration() {
        lsaSidecarTask?.cancel()
        lsaSidecarTask = nil
        isGeneratingLSASidecar = false
    }

    func resetToTopPerformanceDefaults() {
        draftSettings.applyModelConfigDefaults(packPath: resolvedSelectedModelReference().packPath)
        pendingProfileReset = true
        validationErrorMessage = nil
        showTimedConfirmation("Staged model & engine defaults — click Save to apply")
    }

    func resetSpeculativeDefaults() {
        var baseline = draftSettings
        baseline.applyModelConfigDefaults(packPath: resolvedSelectedModelReference().packPath)
        draftSettings.speculativeDecodingEnabled = baseline.speculativeDecodingEnabled
        draftSettings.mtpBackend = baseline.mtpBackend
        draftSettings.mtpDrafts = baseline.mtpDrafts
        draftSettings.mtpDraftStrategy = baseline.mtpDraftStrategy
        draftSettings.mtpRelaxedMargin = baseline.mtpRelaxedMargin
        draftSettings.mtpRelaxedMinProb = baseline.mtpRelaxedMinProb
        draftSettings.mtpQuadraticSeeding = baseline.mtpQuadraticSeeding
        draftSettings.mtpAdaptiveDepth = baseline.mtpAdaptiveDepth
        draftSettings.mtpEntropyThreshold = baseline.mtpEntropyThreshold
        draftSettings.dflashDraftPackPath = baseline.dflashDraftPackPath
        draftSettings.dflashSelectorTopK = baseline.dflashSelectorTopK
        validationErrorMessage = nil
        showTimedConfirmation("Staged speculative defaults — click Save to apply")
    }

    private func showTimedConfirmation(_ message: String) {
        saveConfirmationMessage = message
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 2_500_000_000)
            if saveConfirmationMessage == message {
                saveConfirmationMessage = nil
            }
        }
    }

    func validateModelAndDrafterOnSelection() {
        let snapshot = draftSettings
        let models = discoveredModels
        if let desc = selectedPrimaryDescriptor {
            applyValidation(for: desc)
            return
        }
        Task { @MainActor in
            let result = await Task.detached(priority: .userInitiated) { () -> Result<(String, QWModelDescriptor), any Error> in
                let rawPath = snapshot.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
                let ref = snapshot.resolveModelReference(rawPath.isEmpty ? snapshot.modelName : rawPath, discovered: models)
                let key = ref.packPath.isEmpty ? rawPath : ref.packPath
                do {
                    return .success((key, try snapshot.validatePrimaryModel()))
                } catch {
                    return .failure(error)
                }
            }.value
            guard draftSettings.modelName == snapshot.modelName,
                  draftSettings.modelPackPath == snapshot.modelPackPath else { return }
            switch result {
            case .success(let (key, desc)):
                if !key.isEmpty { telemetry.cachedPrimaryDescriptors[key] = desc }
                applyValidation(for: desc)
            case .failure(let error):
                validationErrorMessage = "\(error)"
            }
        }
    }

    private func applyValidation(for desc: QWModelDescriptor) {
        do {
            if !(desc.visDepth > 0 && desc.capabilities.contains(.vision)) {
                draftSettings.loadVision = false
            }
            let supportsDFlash = Self.supportsDFlash2(
                for: draftSettings,
                descriptor: desc,
                discoveredModels: discoveredModels,
                discoveredDrafters: discoveredDrafterModels
            )
            if !supportsDFlash && isExternalDrafterSelected {
                draftSettings.mtpBackend = "variational"
            } else if isExternalDrafterSelected,
                      !Self.supportsExternalDrafter(
                          backend: draftSettings.mtpBackend,
                          for: draftSettings,
                          descriptor: desc,
                          discoveredModels: discoveredModels,
                          discoveredDrafters: discoveredDrafterModels
                      ),
                      draftSettings.dflashDraftPackPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                draftSettings.reconcileStaleDrafterSelection(for: desc, drafters: discoveredDrafterModels)
            }
            if isExternalDrafterSelected,
               draftSettings.dflashDraftPackPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
               let match = QWServerSettings.resolveDefaultDrafterPack(
                   for: desc,
                   preferredBackend: draftSettings.mtpBackend,
                   drafters: discoveredDrafterModels,
                   targetModelName: draftSettings.modelName
               ) {
                draftSettings.dflashDraftPackPath = match.packPath
            }
            let effPath = QWServerSettings.effectiveDrafterPackPath(
                configuredPath: draftSettings.dflashDraftPackPath,
                for: desc,
                preferredBackend: draftSettings.mtpBackend,
                drafters: discoveredDrafterModels,
                targetModelName: draftSettings.modelName
            ) ?? draftSettings.dflashDraftPackPath
            let maxD = isExternalDrafterSelected
                ? (telemetry.cachedMaxDrafts(forDrafterPackPath: effPath) ?? 7)
                : max(1, desc.maxSpeculativeDrafts)
            if draftSettings.mtpDrafts > maxD {
                draftSettings.mtpDrafts = maxD
            }
            if isExternalDrafterSelected {
                try draftSettings.validateDFlashConfiguration()
            }
            if draftSettings.lsaMode.lowercased() == "sidecar" && !selectedModelHasLSASidecar {
                validationErrorMessage = "LSA mode 'sidecar' requires a model.qlsa file (click 'Build .qlsa (SVD)' or choose 'Auto')"
                return
            }
            validationErrorMessage = nil
        } catch {
            validationErrorMessage = "\(error)"
        }
    }

    func selectCacheDirectory() {
        guard let url = chooseFilePanel(canChooseFiles: false, canChooseDirectories: true) else { return }
        draftSettings.cacheDir = url.path
    }

    var mountedMemoryPack: QWEngineInfoResponse.MountedMemoryPackInfo? {
        telemetry.engineInfo?.mountedMemoryPack
    }

    var isKVModeLockedByMemoryPack: Bool {
        mountedMemoryPack?.enforcesKVMode == true
    }

    var selectedModelContextMemoryEstimate: QWContextMemoryEstimate? {
        var effective = draftSettings
        if let enforced = mountedMemoryPack?.enforcedKVMode {
            effective.kvMode = enforced
        }
        return telemetry.cachedContextEstimate(for: effective, descriptor: selectedPrimaryDescriptor)
    }

    func selectAndMountMemoryPackDirectory() {
        guard let url = chooseFilePanel(
            canChooseFiles: false,
            canChooseDirectories: true,
            message: "Select a repository or knowledge directory containing .qwg/manifest.json"
        ) else { return }
        telemetry.mountMemoryPack(path: url.path) { result in
            switch result {
            case .success(let info):
                draftSettings.kvMode = info.enforcedKVMode
                validationErrorMessage = nil
                showTimedConfirmation("Mounted '\(info.packName)' (KV \(info.enforcedKVMode.uppercased()) locked)")
            case .failure(let error):
                validationErrorMessage = "Memory pack mount failed: \(error.localizedDescription)"
            }
        }
    }

    func unmountActiveMemoryPack() {
        telemetry.unmountMemoryPack {
            validationErrorMessage = nil
            showTimedConfirmation("Unmounted memory pack (KV Cache Mode unlocked)")
        }
    }

    func revealLogFolderInFinder() {
        let url = draftSettings.logDir.canonicalFileURL
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: url.path)
    }

    func revealSettingsInFinder() {
        let url = QWPaths.settingsFileURL
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    func updateRunAtStartup(_ enabled: Bool) {
        if #available(macOS 13.0, *) {
            do {
                if enabled {
                    if SMAppService.mainApp.status != .enabled {
                        try SMAppService.mainApp.register()
                    }
                } else {
                    if SMAppService.mainApp.status == .enabled {
                        try SMAppService.mainApp.unregister()
                    }
                }
            } catch {
                print("Failed to update SMAppService runAtStartup: \(error)")
            }
        }
    }

    func discardChanges() {
        draftSettings = baselineSettings
        pendingProfileReset = false
        validationErrorMessage = nil
        saveConfirmationMessage = nil
    }

    func applyAndSave() {
        Task { @MainActor in
            _ = await persistDraftSettings()
        }
    }

    func saveAndRestartServer() {
        Task { @MainActor in
            if await persistDraftSettings() {
                telemetry.restartServerProcess()
            }
        }
    }

    private func persistDraftSettings() async -> Bool {
        guard !isSaving else { return false }
        isSaving = true
        defer { isSaving = false }
        do {
            var toSave = draftSettings
            let scanned = await Task.detached(priority: .userInitiated) {
                QWServerSettings.scanModelsDirectory()
            }.value
            _ = toSave.pruneRemovedModels(discovered: scanned)
            try toSave.validateBeforeLaunch()
            updateRunAtStartup(toSave.runAtStartup)
            if pendingProfileReset {
                QWSettingsManager.shared.removeProfile(forModel: toSave.modelName)
            }
            try await telemetry.saveSettingsAndValidateLive(toSave)
            baselineSettings = telemetry.settings
            draftSettings = telemetry.settings
            pendingProfileReset = false
            validationErrorMessage = nil
            showTimedConfirmation("Settings saved & applied")
            return true
        } catch {
            validationErrorMessage = "Save failed: \(error.localizedDescription)"
            return false
        }
    }

    // MARK: - Pending Server Restart & UI Building Blocks

    var pendingRestartPanel: some View {
        let pending = telemetry.pendingServerRestartKeys(for: draftSettings)
        return Group {
            if pending.isEmpty {
                EmptyView()
            } else {
                VStack(alignment: .leading, spacing: 6) {
                    Text("\(pending.count) changed setting\(pending.count == 1 ? "" : "s") need a server restart")
                        .font(TelemetryTheme.ui(11, .semibold))
                        .foregroundStyle(TelemetryTheme.warning)
                    Text(pending.map { QWSettingsCoherence.displayName(for: $0) }.joined(separator: ", "))
                        .font(TelemetryTheme.ui(10))
                        .foregroundStyle(TelemetryTheme.secondary(increasedContrast: contrast == .increased))
                    HStack(spacing: 8) {
                        Button("Restart Server") { telemetry.restartServerProcess() }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.small)
                            .disabled(validationErrorMessage != nil || isSaving)
                        Text("Stops the server process and starts it again.")
                            .font(TelemetryTheme.ui(10))
                            .foregroundStyle(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                    }
                }
                .padding(10)
                .background(TelemetryTheme.warning.opacity(0.10))
                .clipShape(RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium))
                .overlay(
                    RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium)
                        .strokeBorder(TelemetryTheme.warning.opacity(0.30), lineWidth: 1)
                )
            }
        }
    }

    func formRow<Content: View>(label: String, key: String = "",
                                @ViewBuilder content: () -> Content) -> some View {
        HStack(alignment: .center, spacing: 12) {
            SettingsRowLabel(text: label, key: key)
                .frame(width: 195, alignment: .leading)
            content()
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    func settingsToggleRow(_ title: String, key: String = "", isOn: Binding<Bool>) -> some View {
        HStack(alignment: .center, spacing: 8) {
            Text(title)
                .font(TelemetryTheme.ui(11.5, .medium))
                .foregroundStyle(TelemetryTheme.secondary(increasedContrast: contrast == .increased))
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
            SettingsRestartBadge(key: key)
            Spacer(minLength: 8)
            Toggle("", isOn: isOn)
                .labelsHidden()
                .toggleStyle(.switch)
                .controlSize(.small)
        }
    }

    @ViewBuilder
    func steppedSlider(value: Binding<Double>, in range: ClosedRange<Double>, step: Double) -> some View {
        let stepCount = step > 0 ? Int(((range.upperBound - range.lowerBound) / step).rounded()) : 0
        let snapped = Binding<Double>(
            get: { value.wrappedValue },
            set: { raw in
                guard step > 0 else {
                    value.wrappedValue = min(range.upperBound, max(range.lowerBound, raw))
                    return
                }
                let rounded = ((raw - range.lowerBound) / step).rounded() * step + range.lowerBound
                let clamped = min(range.upperBound, max(range.lowerBound, rounded))
                // Round floating-point noise for fractional step increments.
                value.wrappedValue = (clamped * 1000).rounded() / 1000
            }
        )
        if step > 0 && stepCount <= 12 {
            Slider(value: snapped, in: range, step: step)
        } else {
            Slider(value: snapped, in: range)
        }
    }

    func intSliderRow(label: String, key: String, range: ClosedRange<Double>, step: Double,
                      readout: @escaping (Int) -> String = { "\($0)" },
                      tint: @escaping (Int) -> Color = { _ in TelemetryTheme.accent },
                      value: Binding<Int>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                SettingsRowLabel(text: label, key: key)
                Spacer()
                Text(readout(value.wrappedValue))
                    .font(TelemetryTheme.numeric(11, .semibold))
                    .foregroundStyle(tint(value.wrappedValue))
            }
            steppedSlider(
                value: Binding(get: { Double(value.wrappedValue) },
                               set: { value.wrappedValue = Int($0.rounded()) }),
                in: range,
                step: step
            )
        }
    }

    func settingsGroup<Content: View>(title: String, icon: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 11))
                    .foregroundStyle(TelemetryTheme.accent)
                Text(title)
                    .font(TelemetryTheme.ui(12, .semibold))
                    .foregroundStyle(TelemetryTheme.primary(increasedContrast: contrast == .increased))
            }
            VStack(alignment: .leading, spacing: 10) {
                content()
            }
            .padding(12)
            .background(TelemetryTheme.substrate)
            .clipShape(RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium))
            .overlay(
                RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium)
                    .strokeBorder(TelemetryTheme.hairline(increasedContrast: contrast == .increased), lineWidth: 1)
            )
        }
    }

    var plainNumberFormatter: NumberFormatter {
        let f = NumberFormatter()
        f.numberStyle = .none
        f.usesGroupingSeparator = false
        return f
    }
}
