// SettingsView+Speculative.swift — The Speculative tab: Standard, Variational,
// DFlash 2, and DSpine block-drafter controls with config.json auto-detection.

import SwiftUI
import QWCore
import QWServer

extension SettingsView {
    // MARK: - 4. Speculative Decoding Section
    var speculativeSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            settingsGroup(title: "Speculative Multi-Token Prediction (MTP)", icon: "bolt.badge.clock") {
                HStack(spacing: 8) {
                    Text("Stored per model profile")
                        .font(TelemetryTheme.ui(11))
                        .foregroundStyle(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                    Spacer()
                    Button("Reset Speculative Defaults") {
                        resetSpeculativeDefaults()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .help("Stage default speculative parameters for the selected model")
                }

                settingsToggleRow(
                    "Enable Speculative Decoding",
                    key: "speculativeDecodingEnabled",
                    isOn: $draftSettings.speculativeDecodingEnabled
                )

                formRow(label: "MTP Backend:", key: "mtpBackend") {
                    Picker("", selection: $draftSettings.mtpBackend) {
                        Text("Standard").tag("standard")
                        Text("Variational").tag("variational")
                        Text(selectedModelSupportsDFlash2
                             ? (selectedModelSupportsDFlash2Backend ? "DFlash 2" : "DFlash 2 (No Pack Installed)")
                             : "DFlash 2 (Unsupported)")
                            .tag("dflash2")
                            .selectionDisabled(!selectedModelSupportsDFlash2Backend)
                        Text(selectedModelSupportsDFlash2
                             ? (selectedModelSupportsDSpineBackend ? "DSpine" : "DSpine (No Pack Installed)")
                             : "DSpine (Unsupported)")
                            .tag("dspine")
                            .selectionDisabled(!selectedModelSupportsDSpineBackend)
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .onChange(of: draftSettings.mtpBackend) { _, newBackend in
                        handleMTPBackendSelectionChange(newBackend)
                    }
                }

                intSliderRow(label: "MTP Draft Tokens:", key: "mtpDrafts",
                             range: 0...Double(selectedModelMaxDrafts), step: 1,
                             readout: { isExternalDrafterSelected ? "\($0) drafts (B=\($0 + 1))" : "\($0) drafts" },
                             value: $draftSettings.mtpDrafts)
                if isExternalDrafterSelected {
                    Text("Block size B = \(draftSettings.mtpDrafts + 1); sparse-MoE adaptive window trims automatically on high entropy.")
                        .font(TelemetryTheme.ui(11))
                        .foregroundStyle(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                        .fixedSize(horizontal: false, vertical: true)
                }

                if !isExternalDrafterSelected {
                    formRow(label: "Draft Strategy:", key: "mtpDraftStrategy") {
                        Picker("", selection: $draftSettings.mtpDraftStrategy) {
                            Text("Hybrid (MTP + N-Gram)").tag("hybrid")
                            Text("MTP Only").tag("mtp")
                            Text("N-Gram Only").tag("ngram")
                        }
                        .labelsHidden()
                        .pickerStyle(.menu)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }

                if draftSettings.mtpBackend == "variational" {
                    Divider()

                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            SettingsRowLabel(text: "Relaxed Margin (ε):", key: "mtpRelaxedMargin")
                            Spacer()
                            Text(draftSettings.mtpRelaxedMargin == 0
                                 ? "0.00 (Strict Argmax)"
                                 : String(format: "%.2f nats", draftSettings.mtpRelaxedMargin))
                                .font(TelemetryTheme.numeric(11, .semibold))
                                .foregroundStyle(draftSettings.mtpRelaxedMargin == 0 ? TelemetryTheme.warning : TelemetryTheme.accent)
                        }
                        steppedSlider(value: $draftSettings.mtpRelaxedMargin, in: 0.0...5.0, step: 0.1)
                        Text("Accepts drafted tokens whose verifier logit is within ε of the target argmax (arXiv:2610.26693)")
                            .font(TelemetryTheme.ui(11))
                            .foregroundStyle(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    VariationalFloorRow(value: $draftSettings.mtpRelaxedMinProb, contrast: contrast)

                    settingsToggleRow(
                        "Quadratic Prefix Seeding (Batch Verify Seed)",
                        key: "mtpQuadraticSeeding",
                        isOn: $draftSettings.mtpQuadraticSeeding
                    )
                }

                if !isExternalDrafterSelected {
                    Divider()

                    settingsToggleRow(
                        "Adaptive Draft Depth (Entropy Stop)",
                        key: "mtpAdaptiveDepth",
                        isOn: $draftSettings.mtpAdaptiveDepth
                    )

                    if draftSettings.mtpAdaptiveDepth {
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                SettingsRowLabel(text: "Entropy Threshold:", key: "mtpEntropyThreshold")
                                Spacer()
                                Text(String(format: "%.2f", draftSettings.mtpEntropyThreshold))
                                    .font(TelemetryTheme.numeric(11, .semibold))
                                    .foregroundStyle(TelemetryTheme.accent)
                            }
                            steppedSlider(value: $draftSettings.mtpEntropyThreshold, in: 0.05...1.0, step: 0.05)
                        }
                    }
                }

                if isExternalDrafterSelected {
                    Divider()

                    formRow(label: "Drafter Model:", key: "dflashDraftPackPath") {
                        HStack(spacing: 8) {
                            Picker("", selection: $draftSettings.dflashDraftPackPath) {
                                ForEach(selectableDrafterModels) { m in
                                    Text(m.name).tag(m.packPath)
                                }
                            }
                            .labelsHidden()
                            .pickerStyle(.menu)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .onChange(of: draftSettings.dflashDraftPackPath) { _, newPath in
                                syncDrafterBackendFromPack(newPath)
                                validateModelAndDrafterOnSelection()
                            }

                            Button(action: refreshDiscoveredModels) {
                                Image(systemName: "arrow.clockwise")
                                    .font(.system(size: 11))
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .accessibilityLabel("Refresh DFlash 2 and DSpine drafter models")
                            .help("Refresh DFlash 2 / DSpine drafter models in ~/.qwg/models")
                        }
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        SettingsRowLabel(text: "Drafter Pack Path:", key: "dflashDraftPackPath")
                        HStack(spacing: 8) {
                            TextField("~/.qwg/models/<drafter-pack>", text: $draftSettings.dflashDraftPackPath)
                                .textFieldStyle(.roundedBorder)
                            Button("Browse...") {
                                selectDrafterPackFile()
                            }
                        }
                    }

                    formRow(label: "Selector Top-K:", key: "dflashSelectorTopK") {
                        Picker("", selection: $draftSettings.dflashSelectorTopK) {
                            Text("1 (Argmax)").tag(1)
                            Text("4").tag(4)
                            Text("8").tag(8)
                            Text("16 (Default)").tag(16)
                        }
                        .labelsHidden()
                        .pickerStyle(.menu)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
        }
    }

    static func stageExternalBackendSelection(
        _ newBackend: String,
        settings: inout QWServerSettings,
        targetDescriptor: QWModelDescriptor?,
        discoveredDrafters: [DiscoveredModel]
    ) {
        let lower = newBackend.lowercased()
        let current = settings.dflashDraftPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let currentCompatible = !current.isEmpty
            && (targetDescriptor.map { QWServerSettings.isDrafterCompatible(packPath: current, with: $0) } ?? true)
            && QWServerSettings.inferDrafterBackendAndMaxDrafts(packPath: current).backend == lower
        if !currentCompatible {
            if let match = QWServerSettings.resolveDefaultDrafterPack(
                for: targetDescriptor,
                preferredBackend: lower,
                drafters: discoveredDrafters,
                targetModelName: settings.modelName
            ) {
                settings.dflashDraftPackPath = match.packPath
            } else if let targetDescriptor, !current.isEmpty,
                      !QWServerSettings.isDrafterCompatible(packPath: current, with: targetDescriptor) {
                settings.dflashDraftPackPath = ""
            }
        }
        let inferred = QWServerSettings.inferDrafterBackendAndMaxDrafts(packPath: settings.dflashDraftPackPath)
        settings.mtpDrafts = inferred.maxDrafts
    }

    private func handleMTPBackendSelectionChange(_ newBackend: String) {
        let lower = newBackend.lowercased()
        let isExt = (lower == "dflash2" || lower == "dspine")
        if !isExt {
            let maxD = selectedModelMaxDrafts
            if draftSettings.mtpDrafts > maxD {
                draftSettings.mtpDrafts = maxD
            }
            validationErrorMessage = nil
            return
        }
        Self.stageExternalBackendSelection(
            newBackend,
            settings: &draftSettings,
            targetDescriptor: selectedPrimaryDescriptor,
            discoveredDrafters: discoveredDrafterModels
        )
        do {
            try draftSettings.validateDFlashConfiguration()
            validationErrorMessage = nil
        } catch {
            validationErrorMessage = "\(newBackend.uppercased()) fail-fast: \(error)"
            draftSettings.mtpBackend = "variational"
            let fallbackMax = selectedModelMaxDrafts
            if draftSettings.mtpDrafts > fallbackMax { draftSettings.mtpDrafts = min(3, fallbackMax) }
        }
    }

    private func syncDrafterBackendFromPack(_ path: String) {
        guard !path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        let inferred = QWServerSettings.inferDrafterBackendAndMaxDrafts(packPath: path)
        if isExternalDrafterSelected {
            if draftSettings.mtpBackend != inferred.backend {
                draftSettings.mtpBackend = inferred.backend
            }
            draftSettings.mtpDrafts = inferred.maxDrafts
        }
    }
}
