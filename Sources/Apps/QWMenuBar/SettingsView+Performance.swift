// SettingsView+Performance.swift — The Performance tab: memory policy,
// expert residency and streaming, prefill shape, and sparse attention.
// Every value is stored per model profile; an absent value is the default.

import SwiftUI
import QWCore
import QWSMCHelper
import QWServer

extension SettingsView {
    // MARK: - 5. Performance & Memory Section
    var performanceSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            settingsGroup(title: "Memory & Residency", icon: "memorychip") {
                HStack(spacing: 6) {
                    Text("Stored per model · QW_* env vars override")
                        .font(TelemetryTheme.ui(11))
                        .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 6)
                    Button("Reset Performance Defaults") { resetPerformanceDefaults() }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .help("Reinstate the measured-best performance settings for the selected model")
                }

                formRow(label: "Memory Policy:", key: "engineTuning") {
                    Picker("", selection: tuningString("memoryPolicy")) {
                        Text("Max Resident (Fastest)").tag(QWMemoryPolicy.maxResident.rawValue)
                        Text("Safe (No Swap)").tag(QWMemoryPolicy.safe.rawValue)
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                Text(tuningString("memoryPolicy").wrappedValue == QWMemoryPolicy.maxResident.rawValue
                     ? "Sizes resident weights to physical RAM less the OS reserve. Fastest; other apps' memory may be swapped out."
                     : "Sizes resident weights to memory that is free or reclaimable at load, less 6 GiB headroom. Never swaps.")
                    .font(TelemetryTheme.ui(11))
                    .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                    .fixedSize(horizontal: false, vertical: true)

                doubleSliderRow(label: "OS Reserve:", id: "osReserveGiB", range: 2...32, step: 1,
                                readout: { String(format: "%.0f GiB", $0) })

                formRow(label: "Scratch Eviction:", key: "evictPrefillScratch") {
                    Picker("", selection: $draftSettings.evictPrefillScratch) {
                        Text("Auto").tag("auto")
                        Text("Always").tag("always")
                        Text("Never").tag("never")
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                    .frame(maxWidth: .infinity)
                }

                if selectedModelSupportsPLE {
                    Divider()
                    tuningToggle("Stream PLE N-gram Table from SSD (Saves ~27.3 GiB GPU RAM)", id: "streamNgrams")
                    Text("Streams external ngram.bin via a 64 MiB L1 cache instead of wiring into Metal RAM. Auto-engages when context exceeds UMA headroom.")
                        .font(TelemetryTheme.ui(11))
                        .foregroundStyle(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            if selectedModelStreamsExperts {
                settingsGroup(title: "MoE Expert Streaming", icon: "square.stack.3d.down.right") {
                    intSliderRow(label: "Resident Experts / Layer:", key: "engineTuning",
                                 range: 0...Double(selectedModelExpertCount), step: 1,
                                 readout: { $0 == 0 ? "Auto (memory policy)" : "\($0) of \(selectedModelExpertCount)" },
                                 value: tuningInt("residentExpertsPerLayer"))
                    intSliderRow(label: "MoE-Spec Verify Budget:", key: "engineTuning",
                                 range: 0...64, step: 1,
                                 readout: { $0 == 0 ? "0 (Lossless)" : "\($0) experts / layer" },
                                 value: tuningInt("moeSpecBudget"))
                    tuningToggle("Cross-Token Expert Cache", id: "expertCache")
                    tuningToggle("Event-Pipelined Cold Expert Staging (Decode)", id: "coldEvents")
                    tuningToggle("Overlap Resident Experts with Cold Staging", id: "coldEventOverlap")
                        .disabled(!tuningBool("coldEvents").wrappedValue)
                    tuningToggle("Mapped Expert Spans per Prefill Chunk", id: "prefillSpanMap")
                    Text("Experts that do not fit in memory stream from SSD. These keep the GPU busy while they load.")
                        .font(TelemetryTheme.ui(11))
                        .foregroundStyle(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                }
            }

            settingsGroup(title: "Prefill Shaping", icon: "cpu") {
                // Bounds are the resolver's clamp bounds (QWPrefillLimits).
                intSliderRow(label: "Prefill Chunk Rows:", key: "prefillChunkSize",
                             range: Double(QWPrefillLimits.minRows)...Double(QWPrefillLimits.maxRows),
                             step: Double(QWPrefillLimits.rowStep), readout: { "\($0) rows" },
                             value: $draftSettings.prefillChunkSize)

                formRow(label: "GDN Prefill:", key: "gdnPrefill") {
                    Picker("", selection: $draftSettings.gdnPrefill) {
                        Text("Chunked (Default)").tag("chunked")
                        Text("Regression (Parallel)").tag("regres")
                        Text("Naive (Stepwise)").tag("naive")
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }

            settingsGroup(title: "Thermal & Cooling", icon: "fanblades") {
                formRow(label: "Cooling Policy:", key: "coolingPolicy") {
                    Picker("", selection: $draftSettings.coolingPolicy) {
                        Text("Auto").tag("auto")
                        Text("Pre-Spin").tag("proactive")
                        Text("Max Blast").tag("max")
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                    .frame(maxWidth: .infinity)
                }
                if let fanStatus = telemetry.dashboard?.facts?.fanControlStatus, !fanStatus.isEmpty {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(fanStatus.contains("requires root") ? TelemetryTheme.warning : TelemetryTheme.active)
                            .frame(width: 6, height: 6)
                        Text("Hardware Status: \(fanStatus)")
                            .font(TelemetryTheme.numeric(11))
                            .foregroundStyle(TelemetryTheme.secondary(increasedContrast: contrast == .increased))
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                HStack(spacing: 8) {
                    Circle()
                        .fill(fanHelperDotColor)
                        .frame(width: 7, height: 7)
                    Text(fanHelperStatusLabel)
                        .font(TelemetryTheme.numeric(11))
                        .foregroundStyle(fanHelperStatus == .active
                            ? TelemetryTheme.active
                            : TelemetryTheme.secondary(increasedContrast: contrast == .increased))
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 6)
                    if isUpdatingFanHelper {
                        ProgressView()
                            .controlSize(.small)
                    } else if fanHelperStatus == .active {
                        Button("Remove") {
                            disableFanHelper()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                        .help("Stop and remove the privileged Fan Helper LaunchDaemon")
                    } else if fanHelperStatus == .requiresApproval {
                        Button("Open Login Items") {
                            openLoginItemsSettings()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                    } else {
                        Button("Enable Fan Control…") {
                            enableFanHelper()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                        .help("Install the minimal privileged Fan Helper daemon for non-root SMC fan writes")
                    }
                }
                .onAppear {
                    refreshFanHelperStatus()
                }
                if let fanHelperMessage, !fanHelperMessage.isEmpty {
                    Text(fanHelperMessage)
                        .font(TelemetryTheme.ui(10.5))
                        .foregroundStyle(TelemetryTheme.warning)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Text("Auto defers to macOS SMC. Pre-Spin ramps fans on prefill/decode; Max Blast targets 100% RPM. Enabling the privileged Fan Helper allows non-root SMC fan writes with a 15 s auto-restore safety watchdog.")
                    .font(TelemetryTheme.ui(11))
                    .foregroundStyle(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                    .fixedSize(horizontal: false, vertical: true)
            }

            settingsGroup(title: "Latent Sparse Attention (LSA — arXiv:2610.42147)", icon: "point.3.connected.trianglepath.dotted") {
                formRow(label: "LSA Mode:", key: "lsaMode") {
                    Picker("", selection: $draftSettings.lsaMode) {
                        Text("Off").tag("off")
                        Text("Auto").tag("auto")
                        Text("Sidecar (.qlsa)").tag("sidecar")
                        Text("Weight-SVD").tag("svd")
                        Text("QSA-Cluster").tag("qsa-cluster")
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .onChange(of: draftSettings.lsaMode) { _, _ in
                        validateModelAndDrafterOnSelection()
                    }
                }

                HStack(spacing: 8) {
                    Circle()
                        .fill(selectedModelHasLSASidecar ? TelemetryTheme.active : TelemetryTheme.unknown)
                        .frame(width: 7, height: 7)
                    Text(isGeneratingLSASidecar ? "Synthesizing model.qlsa via Weight-SVD…" : selectedModelLSASidecarStatusText)
                        .font(TelemetryTheme.numeric(11))
                        .foregroundStyle(selectedModelHasLSASidecar
                            ? TelemetryTheme.active
                            : TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                    Spacer()
                    if isGeneratingLSASidecar {
                        ProgressView()
                            .controlSize(.small)
                        Button("Cancel") {
                            cancelLSASidecarGeneration()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                    } else {
                        Button("Build .qlsa (SVD)") {
                            generateLSASidecarForSelectedModel()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                        .help("Synthesize a model.qlsa projection sidecar for the selected model using Weight-SVD")
                    }
                }

                if draftSettings.lsaMode != "off" {
                    Divider()

                    formRow(label: "Key Budget (k):", key: "lsaBudget") {
                        Picker("", selection: $draftSettings.lsaBudget) {
                            Text("512 tokens").tag(512)
                            Text("1,024 tokens").tag(1024)
                            Text("Auto / 2,048 (Adaptive ≤16K)").tag(2048)
                            Text("4,096 tokens").tag(4096)
                            Text("8,192 tokens").tag(8192)
                            Text("16,384 tokens").tag(16384)
                        }
                        .labelsHidden()
                        .pickerStyle(.menu)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    formRow(label: "Cluster Block (L):", key: "lsaClusterBlock") {
                        Picker("", selection: $draftSettings.lsaClusterBlock) {
                            Text("256 tokens").tag(256)
                            Text("512 tokens (Default)").tag(512)
                            Text("1,024 tokens").tag(1024)
                            Text("4,096 (e-means)").tag(4096)
                        }
                        .labelsHidden()
                        .pickerStyle(.menu)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        SettingsRowLabel(text: "Custom .qlsa Sidecar Path (Optional):", key: "lsaSidecarPath")
                        HStack(spacing: 8) {
                            TextField("Auto-detected from <model_dir>/model.qlsa", text: $draftSettings.lsaSidecarPath)
                                .textFieldStyle(.roundedBorder)
                            Button("Browse...") {
                                selectLSASidecarFile()
                            }
                        }
                    }

                    settingsToggleRow(
                        "Cross-Loop Block Reuse (Looped Attention)",
                        key: "lsaCrossLoopReuse",
                        isOn: $draftSettings.lsaCrossLoopReuse
                    )

                    Text("Auto selects .qlsa / Weight-SVD on GQA (<= k+516 runs exact dense; k=2,048 scales adaptively from 4K at 98K+ up to 16K at 786K+), native flat QSA <= 262K + e-means-4096 > 262K on QSA, and < 41K flat bypass + >= 41K e-means coarse pruning on MLA. Keep Cross-Loop Reuse off for full-slice recall.")
                        .font(TelemetryTheme.ui(11))
                        .foregroundColor(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    /// True when the selected model carries Per-Layer Embedding (PLE) N-gram modules (`ngram.bin`).
    var selectedModelSupportsPLE: Bool {
        if let desc = try? draftSettings.validatePrimaryModel() {
            return !desc.pleModuleIndices.isEmpty
        }
        if let disc = discoveredModels.first(where: { $0.name == draftSettings.modelName }),
           !disc.capabilities.isEmpty {
            return disc.capabilities.contains("ple")
        }
        return false
    }

    /// The selected model has routed experts the engine can stream (MoE outside FlashNext's own engine).
    var selectedModelStreamsExperts: Bool {
        if let desc = try? draftSettings.validatePrimaryModel() {
            return desc.supportsExpertStreaming
        }
        if let disc = discoveredModels.first(where: { $0.name == draftSettings.modelName }),
           let familyRaw = disc.family,
           let family = QWModelDescriptor.Family(rawValue: familyRaw),
           let spec = QWModelFamilyRegistry.shared.spec(for: family) {
            return disc.capabilities.contains("moe") && spec.supportsExpertStreaming
        }
        return false
    }

    var selectedModelExpertCount: Int {
        max(1, (try? draftSettings.validatePrimaryModel())?.numExperts ?? 256)
    }

    func resetPerformanceDefaults() {
        let rawPath = draftSettings.modelPackPath.trimmingCharacters(in: .whitespacesAndNewlines)
        var baseline = draftSettings
        baseline.applyModelConfigDefaults(packPath: draftSettings.resolveModelReference(
            rawPath.isEmpty ? draftSettings.modelName : rawPath).packPath)
        draftSettings.engineTuning = baseline.engineTuning
        draftSettings.evictPrefillScratch = baseline.evictPrefillScratch
        draftSettings.prefillChunkSize = baseline.prefillChunkSize
        draftSettings.maxConcurrentSessions = baseline.maxConcurrentSessions
        draftSettings.gdnPrefill = baseline.gdnPrefill
        draftSettings.lsaCrossLoopReuse = baseline.lsaCrossLoopReuse
    }

    private func tuningValue(_ id: String) -> QWKnobValue? {
        draftSettings.engineTuning[id] ?? QWKnobRegistry.descriptor(id)?.defaultValue
    }

    func tuningBool(_ id: String) -> Binding<Bool> {
        Binding(get: { tuningValue(id)?.boolValue ?? true },
                set: { draftSettings.setEngineTuning(id, .bool($0)) })
    }

    func tuningInt(_ id: String) -> Binding<Int> {
        Binding(get: { tuningValue(id)?.intValue ?? 0 },
                set: { draftSettings.setEngineTuning(id, .int($0)) })
    }

    func tuningString(_ id: String) -> Binding<String> {
        Binding(get: { tuningValue(id)?.stringValue ?? "" },
                set: { draftSettings.setEngineTuning(id, .string($0)) })
    }

    private func tuningToggle(_ title: String, id: String) -> some View {
        settingsToggleRow(title, key: "engineTuning", isOn: tuningBool(id))
    }

    private func doubleSliderRow(label: String, id: String, range: ClosedRange<Double>, step: Double,
                                 readout: @escaping (Double) -> String) -> some View {
        let value = Binding<Double>(get: { tuningValue(id)?.doubleValue ?? range.lowerBound },
                                    set: { draftSettings.setEngineTuning(id, .double($0)) })
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                SettingsRowLabel(text: label, key: "engineTuning")
                Spacer()
                Text(readout(value.wrappedValue))
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundColor(TelemetryTheme.accent)
            }
            steppedSlider(value: value, in: range, step: step)
        }
    }
}
