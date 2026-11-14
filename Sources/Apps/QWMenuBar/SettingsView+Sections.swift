// SettingsView+Sections.swift — Sampling, cache, and logging tabs for SettingsView

import QWCore
import QWServer
import SwiftUI

extension SettingsView {
    // MARK: - 3. Sampling & Reasoning Section
    var samplingSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            settingsGroup(title: "Sampling & Generation Hyperparameters", icon: "slider.horizontal.3") {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        SettingsRowLabel(text: "Temperature:", key: "temperature")
                        Spacer()
                        Text(draftSettings.temperature == 0 ? "0.00 (Greedy / Argmax)" : String(format: "%.2f", draftSettings.temperature))
                            .font(TelemetryTheme.numeric(11, .semibold))
                            .foregroundStyle(draftSettings.temperature == 0 ? TelemetryTheme.warning : TelemetryTheme.accent)
                    }
                    steppedSlider(value: $draftSettings.temperature, in: 0.0...2.0, step: 0.05)
                }

                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        SettingsRowLabel(text: "Top-P (Nucleus):", key: "topP")
                        Spacer()
                        Text(String(format: "%.2f", draftSettings.topP))
                            .font(TelemetryTheme.numeric(11, .semibold))
                            .foregroundStyle(TelemetryTheme.accent)
                    }
                    steppedSlider(value: $draftSettings.topP, in: 0.05...1.0, step: 0.05)
                }

                intSliderRow(label: "Top-K Filtering:", key: "topK", range: 0...100, step: 1,
                             readout: { $0 <= 0 ? "0 (Disabled)" : ($0 == 1 ? "1 (Greedy / Argmax)" : "\($0)") },
                             tint: { $0 <= 0 ? TelemetryTheme.unknown : ($0 == 1 ? TelemetryTheme.warning : TelemetryTheme.accent) },
                             value: $draftSettings.topK)

                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        SettingsRowLabel(text: "Repetition Penalty:", key: "repetitionPenalty")
                        Spacer()
                        Text(String(format: "%.2f", draftSettings.repetitionPenalty))
                            .font(TelemetryTheme.numeric(11, .semibold))
                            .foregroundStyle(TelemetryTheme.accent)
                    }
                    steppedSlider(value: $draftSettings.repetitionPenalty, in: 1.0...1.5, step: 0.01)
                }

                formRow(label: "Max Tokens:", key: "maxTokens") {
                    let presets = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
                    Picker("", selection: $draftSettings.maxTokens) {
                        if !presets.contains(draftSettings.maxTokens) {
                            Text("\(draftSettings.maxTokens.formatted()) tokens (Custom)")
                                .tag(draftSettings.maxTokens)
                        }
                        Text("256 tokens").tag(256)
                        Text("512 tokens (Default)").tag(512)
                        Text("1,024 tokens").tag(1024)
                        Text("2,048 tokens").tag(2048)
                        Text("4,096 tokens").tag(4096)
                        Text("8,192 tokens").tag(8192)
                        Text("16,384 tokens").tag(16384)
                        Text("32,768 tokens").tag(32768)
                        Text("65,536 tokens").tag(65536)
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                }

                formRow(label: "Random Seed:", key: "seed") {
                    VStack(alignment: .leading, spacing: 3) {
                        TextField("-1 (Random)", value: $draftSettings.seed, formatter: plainNumberFormatter)
                            .textFieldStyle(.roundedBorder)
                        Text("Set -1 for non-deterministic pseudorandom sampling")
                            .font(TelemetryTheme.ui(11))
                            .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
                    }
                }
            }

            settingsGroup(title: "Reasoning & Template Conditioning", icon: "sparkles") {
                settingsToggleRow(
                    "Enable Thinking Mode (<think>…</think>)",
                    key: "enableThinking",
                    isOn: $draftSettings.enableThinking
                )

                formRow(label: "Reasoning Effort:", key: "reasoningEffort") {
                    Picker("", selection: $draftSettings.reasoningEffort) {
                        Text("Low").tag("low")
                        Text("Medium").tag("medium")
                        Text("Extra High (Default)").tag("xhigh")
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                }

                VStack(alignment: .leading, spacing: 4) {
                    SettingsRowLabel(text: "Default System Prompt:", key: "systemPrompt")
                    TextField("e.g. You are a helpful AI assistant...", text: $draftSettings.systemPrompt)
                        .textFieldStyle(.roundedBorder)
                    Text("Prepended if incoming request does not provide a system role message")
                        .font(TelemetryTheme.ui(11))
                        .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
                }
            }
        }
    }

    // MARK: - 6. Memory & Cache Section
    var cacheSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            settingsGroup(title: "Codebase & Knowledge Memory Pack (.qwg)", icon: "memorychip") {
                if let mounted = mountedMemoryPack {
                    HStack(spacing: 8) {
                        Image(systemName: "lock.shield.fill")
                            .font(.system(size: 12))
                            .foregroundStyle(TelemetryTheme.active)
                        VStack(alignment: .leading, spacing: 2) {
                            Text("\(mounted.packName) — \(mounted.fileCount) files · \(mounted.tokenCount) tokens · KV \(mounted.enforcedKVMode.uppercased()) (enforced)")
                                .font(TelemetryTheme.ui(11, .semibold))
                                .foregroundStyle(TelemetryTheme.primary(increasedContrast: increasedContrast))
                            Text("\(mounted.rootPath) · LSA \(mounted.lsaMode) · Live delta: \(mounted.modifiedFileCount) file(s)")
                                .font(TelemetryTheme.numeric(10))
                                .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
                        }
                        Spacer()
                        Button("Mount...") {
                            selectAndMountMemoryPackDirectory()
                        }
                        .controlSize(.small)
                        Button("Unmount") {
                            unmountActiveMemoryPack()
                        }
                        .controlSize(.small)
                    }
                } else {
                    HStack(spacing: 8) {
                        Text("Auto-discovers <repo>/.qwg/manifest.json from coding agent system prompts, or mount manually:")
                            .font(TelemetryTheme.ui(11))
                            .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
                        Spacer()
                        Button("Mount Repository...") {
                            selectAndMountMemoryPackDirectory()
                        }
                        .controlSize(.small)
                    }
                }
            }

            settingsGroup(title: "Two-Tier Hierarchical Prefix Cache", icon: "internaldrive") {
                formRow(label: "Warm RAM Limit (GB):", key: "warmRAMLimitGB") {
                    TextField("4.0", value: $draftSettings.warmRAMLimitGB, formatter: plainNumberFormatter)
                        .textFieldStyle(.roundedBorder)
                }

                intSliderRow(label: "Concurrent Session Slots:", key: "maxConcurrentSessions",
                             range: 1...16, step: 1,
                             readout: { $0 == 1 ? "1 slot (Single-stream)" : "\($0) slots" },
                             value: $draftSettings.maxConcurrentSessions)
                Text("Protects concurrent agent/subagent KV cache slots in Warm RAM without mutual eviction when requests fit within per-slot context.")
                    .font(TelemetryTheme.ui(11))
                    .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))

                settingsToggleRow(
                    "Persist Prefix Cache to Cold SSD",
                    key: "ssdCacheEnabled",
                    isOn: $draftSettings.ssdCacheEnabled
                )

                formRow(label: "Cold SSD Limit (GB):", key: "ssdCacheLimitGB") {
                    TextField("10.0", value: $draftSettings.ssdCacheLimitGB, formatter: plainNumberFormatter)
                        .textFieldStyle(.roundedBorder)
                        .disabled(!draftSettings.ssdCacheEnabled)
                }

                VStack(alignment: .leading, spacing: 4) {
                    SettingsRowLabel(text: "Cache Directory:", key: "cacheDir")
                    HStack(spacing: 8) {
                        TextField("~/.qwg/cache", text: $draftSettings.cacheDir)
                            .textFieldStyle(.roundedBorder)
                            .disabled(!draftSettings.ssdCacheEnabled)
                        Button("Browse...") {
                            selectCacheDirectory()
                        }
                        .disabled(!draftSettings.ssdCacheEnabled)
                    }
                }
            }
        }
    }

    // MARK: - 7. Logging & Diagnostics Section
    var loggingSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            settingsGroup(title: "Logging Configuration", icon: "doc.text") {
                formRow(label: "Log Level:", key: "logLevel") {
                    Picker("", selection: $draftSettings.logLevel) {
                        Text("Debug").tag("debug")
                        Text("Info").tag("info")
                        Text("Warn").tag("warn")
                        Text("Error").tag("error")
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                }

                formRow(label: "Log File Name:", key: "logFileName") {
                    TextField("qwg.log", text: $draftSettings.logFileName)
                        .textFieldStyle(.roundedBorder)
                }

                VStack(alignment: .leading, spacing: 4) {
                    SettingsRowLabel(text: "Log Directory:", key: "logDir")
                    HStack(spacing: 8) {
                        TextField("~/.qwg/logs", text: $draftSettings.logDir)
                            .textFieldStyle(.roundedBorder)
                        Button("Reveal in Finder") {
                            revealLogFolderInFinder()
                        }
                    }
                }
            }
        }
    }
}
