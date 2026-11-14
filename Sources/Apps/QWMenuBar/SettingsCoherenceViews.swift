// SettingsCoherenceViews.swift — The marks that say when a setting takes
// effect. A setting the server reads only when it starts is marked, so a
// change that waits for a restart is never silent.

import SwiftUI
import QWCore
import QWServer

/// The marker on a setting the server reads when it starts.
struct SettingsRestartBadge: View {
    let key: String

    var body: some View {
        if QWSettingsCoherence.needsServerRestart(key) {
            HStack(spacing: 3) {
                Image(systemName: "arrow.clockwise")
                    .font(.system(size: 9.5, weight: .semibold))
                Text("Restart")
                    .font(.system(size: 10, weight: .semibold))
            }
            .foregroundColor(TelemetryTheme.warning)
            .padding(.horizontal, 6)
            .padding(.vertical, 1.5)
            .background(TelemetryTheme.warning.opacity(0.14))
            .clipShape(Capsule())
            .fixedSize()
            .help("The server reads this setting when it starts. A change to it takes effect after the server restarts.")
        }
    }
}

/// A settings label with the marker that says when the value takes
/// effect.
struct SettingsRowLabel: View {
    let text: String
    let key: String
    @Environment(\.colorSchemeContrast) private var contrast: ColorSchemeContrast

    var body: some View {
        HStack(spacing: 6) {
            Text(text)
                .font(TelemetryTheme.ui(11.5, .medium))
                .foregroundStyle(TelemetryTheme.secondary(increasedContrast: contrast == .increased))
                .lineLimit(1)
            SettingsRestartBadge(key: key)
        }
    }
}

/// The Variational MTP fidelity floor (τ): a relaxed draft must also hold `p >= τ`.
struct VariationalFloorRow: View {
    @Binding var value: Double
    let contrast: ColorSchemeContrast

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                SettingsRowLabel(text: "Fidelity Floor (τ):", key: "mtpRelaxedMinProb")
                Spacer()
                Text(value == 0 ? "0.00 (Off)" : String(format: "p ≥ %.2f", value))
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundColor(value == 0 ? TelemetryTheme.warning : TelemetryTheme.accent)
            }
            Slider(value: $value, in: 0.0...0.5, step: 0.05)
            Text("A relaxed draft must also hold at least τ of the verifier's probability mass.")
                .font(TelemetryTheme.ui(11))
                .foregroundStyle(TelemetryTheme.quiet(increasedContrast: contrast == .increased))
        }
    }
}

/// Power-of-2 Context Limit slider (64 to 1,048,576 tokens) with live UMA memory bar and inline clamp.
struct ContextLimitSliderRow: View {
    @Binding var maxContext: Int
    let estimate: QWContextMemoryEstimate?
    let modelName: String
    let kvMode: String
    let contrast: ColorSchemeContrast

    private var isOverBudget: Bool {
        guard let estimate else { return false }
        return !estimate.fits
    }

    private var maxFittingStep: Int {
        guard let estimate else { return 64 }
        let fitting = QWSettingsCoherence.contextSliderSteps.filter { $0 <= estimate.maxFittingContext }
        return fitting.last ?? 64
    }

    private var stepBinding: Binding<Double> {
        Binding(
            get: { Double(QWSettingsCoherence.nearestContextStepIndex(for: maxContext)) },
            set: { newIndex in
                maxContext = QWSettingsCoherence.contextTokens(forStepIndex: Int(newIndex.rounded()))
            }
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                SettingsRowLabel(text: "Context Limit:", key: "maxContext")
                Spacer()
                Text(QWSettingsCoherence.formatContextTokens(maxContext))
                    .font(TelemetryTheme.numeric(11, .semibold))
                    .foregroundStyle(isOverBudget ? TelemetryTheme.failure : TelemetryTheme.accent)
            }
            Slider(
                value: stepBinding,
                in: 0...Double(QWSettingsCoherence.contextSliderSteps.count - 1),
                step: 1
            )
            HStack {
                Text("64")
                Spacer()
                Text("1k")
                Spacer()
                Text("16k")
                Spacer()
                Text("256k")
                Spacer()
                Text("1M")
            }
            .font(TelemetryTheme.numeric(10))
            .foregroundStyle(TelemetryTheme.quiet(increasedContrast: contrast == .increased))

            if let est = estimate {
                memorySummarySection(est)
            }
        }
    }

    @ViewBuilder
    private func memorySummarySection(_ est: QWContextMemoryEstimate) -> some View {
        let gib = 1_073_741_824.0
        let reqGiB = Double(est.requiredBytes) / gib
        let wGiB = Double(est.weightsBytes) / gib
        let kvGiB = Double(est.kvBytes) / gib
        let scrGiB = Double(est.scratchAndStateBytes) / gib
        let availGiB = Double(est.availableBytes) / gib

        VStack(alignment: .leading, spacing: 6) {
            proportionalUMABar(est)
            Text(String(
                format: "Predicted Memory: %.1f GiB (Weights %.1f GiB · KV %.1f GiB · Scratch %.1f GiB) / %.1f GiB budget",
                reqGiB, wGiB, kvGiB, scrGiB, availGiB
            ))
            .font(TelemetryTheme.numeric(11))
            .foregroundStyle(est.fits
                ? TelemetryTheme.quiet(increasedContrast: contrast == .increased)
                : TelemetryTheme.failure)

            if !est.fits {
                let overGiB = Double(est.requiredBytes - est.availableBytes) / gib
                HStack(alignment: .center, spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 11))
                        .foregroundStyle(TelemetryTheme.failure)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Context Limit Exceeds Available Memory")
                            .font(TelemetryTheme.ui(11, .semibold))
                            .foregroundStyle(TelemetryTheme.failure)
                        Text(String(
                            format: "%@ (%@ KV) exceeds the %.1f GiB budget by %.1f GiB for %@.",
                            QWSettingsCoherence.formatContextTokens(maxContext),
                            kvMode.uppercased(),
                            availGiB,
                            overGiB,
                            modelName.isEmpty ? "the selected model" : modelName
                        ))
                        .font(TelemetryTheme.ui(11))
                        .foregroundStyle(TelemetryTheme.secondary(increasedContrast: contrast == .increased))
                    }
                    Spacer(minLength: 6)
                    if maxFittingStep < maxContext {
                        Button("Clamp to \(QWSettingsCoherence.formatContextTokens(maxFittingStep))") {
                            maxContext = maxFittingStep
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(TelemetryTheme.failure)
                        .controlSize(.small)
                    }
                }
                .padding(8)
                .background(TelemetryTheme.failure.opacity(0.10))
                .clipShape(RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium))
                .overlay(
                    RoundedRectangle(cornerRadius: TelemetryTheme.Radius.medium)
                        .strokeBorder(TelemetryTheme.failure.opacity(0.35), lineWidth: 1)
                )
            }
        }
    }

    private func proportionalUMABar(_ est: QWContextMemoryEstimate) -> some View {
        let totalDenominator = max(Double(max(est.availableBytes, est.requiredBytes)), 1.0)
        let weightsFrac = CGFloat(max(0, Double(est.weightsBytes)) / totalDenominator)
        let kvFrac = CGFloat(max(0, Double(est.kvBytes)) / totalDenominator)
        let scratchFrac = CGFloat(max(0, Double(est.scratchAndStateBytes)) / totalDenominator)
        let budgetFrac = CGFloat(max(0, Double(est.availableBytes)) / totalDenominator)

        return GeometryReader { geo in
            let width = geo.size.width
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 3)
                    .fill(TelemetryTheme.substrate)
                HStack(spacing: 1) {
                    if weightsFrac > 0 {
                        Rectangle()
                            .fill(TelemetryTheme.accent)
                            .frame(width: max(2, width * weightsFrac))
                    }
                    if kvFrac > 0 {
                        Rectangle()
                            .fill(est.fits ? TelemetryTheme.cool : TelemetryTheme.failure)
                            .frame(width: max(2, width * kvFrac))
                    }
                    if scratchFrac > 0 {
                        Rectangle()
                            .fill(TelemetryTheme.prefill)
                            .frame(width: max(2, width * scratchFrac))
                    }
                }
                .frame(height: 6)
                .clipShape(RoundedRectangle(cornerRadius: 3))

                if !est.fits && budgetFrac > 0 && budgetFrac < 1 {
                    Rectangle()
                        .fill(TelemetryTheme.primary(increasedContrast: true))
                        .frame(width: 1.5, height: 10)
                        .offset(x: min(width - 2, width * budgetFrac))
                }
            }
        }
        .frame(height: 10)
    }
}
