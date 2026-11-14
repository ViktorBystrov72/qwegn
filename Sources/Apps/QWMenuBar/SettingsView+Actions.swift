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
        let scanned = QWServerSettings.scanModelsDirectory()
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
        discoveredDrafterModels = QWServerSettings.scanDrafterModelsDirectory()
        if !draftSettings.dflashDraftPackPath.isEmpty
            && !discoveredDrafterModels.contains(where: { $0.packPath == draftSettings.dflashDraftPackPath }) {
            let name = QWModelRoot.canonicalModelName(for: URL(fileURLWithPath: draftSettings.dflashDraftPackPath))
            discoveredDrafterModels.insert(DiscoveredModel(name: name, packPath: draftSettings.dflashDraftPackPath), at: 0)
        } else if isExternalDrafterSelected, draftSettings.dflashDraftPackPath.isEmpty, let first = discoveredDrafterModels.first {
            draftSettings.dflashDraftPackPath = first.packPath
        }
    }

    func selectDrafterPackFile() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            draftSettings.dflashDraftPackPath = url.path
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
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            let baseName = QWModelRoot.canonicalModelName(for: url)
            let entry = QWModelCatalog.classify(name: baseName, path: url.path)
            guard entry.supported else {
                validationErrorMessage = "Cannot select '\(baseName)': \(entry.reason ?? "not runnable")"
                return
            }
            draftSettings.modelPackPath = url.path
            if !baseName.isEmpty {
                draftSettings.modelName = baseName
            }
            refreshDiscoveredModels()
            validateModelAndDrafterOnSelection()
        }
    }

    var selectedModelSupportsVision: Bool {
        if let disc = discoveredModels.first(where: { $0.name == draftSettings.modelName }),
           !disc.capabilities.isEmpty {
            return disc.capabilities.contains("vision")
        }
        if let desc = try? draftSettings.validatePrimaryModel() {
            return desc.visDepth > 0 && desc.capabilities.contains(.vision)
        }
        return true
    }

    var selectedModelSupportsDFlash2: Bool {
        if let desc = try? draftSettings.validatePrimaryModel() {
            return desc.hcCount == nil && desc.numExperts == nil
        }
        if let disc = discoveredModels.first(where: { $0.name == draftSettings.modelName }) {
            return !disc.capabilities.contains("moe") && !disc.capabilities.contains("hyperConnection")
        }
        return true
    }

    var isExternalDrafterSelected: Bool {
        let lower = draftSettings.mtpBackend.lowercased()
        return lower == "dflash2" || lower == "dspine"
    }

    var selectedModelMaxDrafts: Int {
        if isExternalDrafterSelected {
            let inferred = QWServerSettings.inferDrafterBackendAndMaxDrafts(packPath: draftSettings.dflashDraftPackPath)
            return draftSettings.mtpBackend.lowercased() == "dspine" ? max(15, inferred.maxDrafts) : inferred.maxDrafts
        }
        if let desc = try? draftSettings.validatePrimaryModel(), desc.maxSpeculativeDrafts > 0 {
            return desc.maxSpeculativeDrafts
        }
        if let disc = discoveredModels.first(where: { $0.name == draftSettings.modelName }), disc.maxSpeculativeDrafts > 0 {
            return disc.maxSpeculativeDrafts
        }
        return 5
    }

    var selectedModelHasLSASidecar: Bool {
        let rawPath = draftSettings.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let ref = draftSettings.resolveModelReference(rawPath.isEmpty ? draftSettings.modelName : rawPath, discovered: discoveredModels)
        let explicit = draftSettings.lsaSidecarPath.trimmingCharacters(in: .whitespacesAndNewlines)
        return QWLSAConfig.resolveSidecarURL(packPath: ref.packPath, explicitSidecarPath: explicit.isEmpty ? nil : explicit) != nil
    }

    var selectedModelLSASidecarStatusText: String {
        let rawPath = draftSettings.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let ref = draftSettings.resolveModelReference(rawPath.isEmpty ? draftSettings.modelName : rawPath, discovered: discoveredModels)
        let explicit = draftSettings.lsaSidecarPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = QWLSAConfig.resolveSidecarURL(packPath: ref.packPath, explicitSidecarPath: explicit.isEmpty ? nil : explicit) else {
            return "No .qlsa sidecar detected (Auto uses in-memory Weight-SVD)"
        }
        if let desc = try? draftSettings.validatePrimaryModel(),
           let info = try? QWLSASidecar.peekInfo(url: url, descriptor: desc) {
            let matchNote = info.geometryMatches ? "verified" : "digest mismatch"
            return "Detected \(url.lastPathComponent) [\(info.method), d=\(info.indexDim), \(matchNote)]"
        }
        return "Detected \(url.lastPathComponent)"
    }

    func selectLSASidecarFile() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            draftSettings.lsaSidecarPath = url.path
            validateModelAndDrafterOnSelection()
        }
    }

    func generateLSASidecarForSelectedModel() {
        guard !isGeneratingLSASidecar else { return }
        lsaSidecarTask?.cancel()
        validationErrorMessage = nil

        let rawPath = draftSettings.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let ref = draftSettings.resolveModelReference(
            rawPath.isEmpty ? draftSettings.modelName : rawPath,
            discovered: discoveredModels
        )
        let desc: QWModelDescriptor
        do {
            desc = try draftSettings.validatePrimaryModel()
        } catch {
            validationErrorMessage = "LSA sidecar build failed: \(error)"
            return
        }

        let packPath = ref.packPath
        let clusterBlock = draftSettings.lsaClusterBlock
        let budget = draftSettings.lsaBudget
        isGeneratingLSASidecar = true

        lsaSidecarTask = Task { @MainActor in
            defer {
                isGeneratingLSASidecar = false
                lsaSidecarTask = nil
            }
            do {
                let outcome: (URL, Int) = try await Task.detached(priority: .userInitiated) {
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
        let rawPath = draftSettings.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let ref = draftSettings.resolveModelReference(rawPath.isEmpty ? draftSettings.modelName : rawPath)
        draftSettings.applyModelConfigDefaults(packPath: ref.packPath)
        pendingProfileReset = true
        validationErrorMessage = nil
        showTimedConfirmation("Staged model & engine defaults — click Save to apply")
    }

    func resetSpeculativeDefaults() {
        let rawPath = draftSettings.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let ref = draftSettings.resolveModelReference(rawPath.isEmpty ? draftSettings.modelName : rawPath)
        var baseline = draftSettings
        baseline.applyModelConfigDefaults(packPath: ref.packPath)
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
        do {
            let desc = try draftSettings.validatePrimaryModel()
            if !(desc.visDepth > 0 && desc.capabilities.contains(.vision)) {
                draftSettings.loadVision = false
            }
            let supportsDFlash = desc.hcCount == nil && desc.numExperts == nil
            if !supportsDFlash && isExternalDrafterSelected {
                draftSettings.mtpBackend = "variational"
            }
            let maxD = isExternalDrafterSelected
                ? QWServerSettings.inferDrafterBackendAndMaxDrafts(packPath: draftSettings.dflashDraftPackPath).maxDrafts
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
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            draftSettings.cacheDir = url.path
        }
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
        return effective.estimateContextMemory()
    }

    func selectAndMountMemoryPackDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "Select a repository or knowledge directory containing .qwg/manifest.json"
        if panel.runModal() == .OK, let url = panel.url {
            telemetry.mountMemoryPack(path: url.path) { result in
                switch result {
                case .success(let info):
                    draftSettings.kvMode = info.enforcedKVMode
                    validationErrorMessage = nil
                    saveConfirmationMessage = "Mounted '\(info.packName)' (KV \(info.enforcedKVMode.uppercased()) locked)"
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                        saveConfirmationMessage = nil
                    }
                case .failure(let error):
                    validationErrorMessage = "Memory pack mount failed: \(error.localizedDescription)"
                }
            }
        }
    }

    func unmountActiveMemoryPack() {
        telemetry.unmountMemoryPack {
            validationErrorMessage = nil
            saveConfirmationMessage = "Unmounted memory pack (KV Cache Mode unlocked)"
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                saveConfirmationMessage = nil
            }
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
            let scanned = QWServerSettings.scanModelsDirectory()
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
                .frame(width: 210, alignment: .leading)
            content()
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    func settingsToggleRow(_ title: String, key: String = "", isOn: Binding<Bool>) -> some View {
        HStack(alignment: .center, spacing: 8) {
            Text(title)
                .font(TelemetryTheme.ui(11.5, .medium))
                .foregroundStyle(TelemetryTheme.secondary(increasedContrast: contrast == .increased))
                .lineLimit(1)
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
