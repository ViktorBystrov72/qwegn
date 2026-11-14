// LogViewerWindow.swift — Structured engine & HTTP request log console

import AppKit
import QWServer
import SwiftUI

/// Streaming engine & HTTP request log window with chip filters, wrap toggle, and copy feedback.
public struct LogViewerWindow: View {
    public var telemetry: TelemetryObservable
    /// Increased Contrast asks for stronger text. L-M7: it is honoured.
    @Environment(\.colorSchemeContrast) internal var contrast: ColorSchemeContrast
    @State private var filterLevel: QWLogFilterLevel = .all
    @State private var searchText: String = ""
    @State private var autoScroll: Bool = true
    @State private var wrapLines: Bool = true
    @State private var showCopiedFeedback: Bool = false

    public init(
        telemetry: TelemetryObservable,
        initialFilterLevel: QWLogFilterLevel = .all,
        initialSearchText: String = "",
        initialWrapLines: Bool = true,
        onClose: @escaping () -> Void = {}
    ) {
        self.telemetry = telemetry
        self._filterLevel = State(initialValue: initialFilterLevel)
        self._searchText = State(initialValue: initialSearchText)
        self._wrapLines = State(initialValue: initialWrapLines)
        let _ = onClose
    }

    private var increasedContrast: Bool {
        contrast == .increased
    }

    private var filteredEntries: [QWLogEntry] {
        telemetry.logEntries.filter { $0.matches(filter: filterLevel, query: searchText) }
    }

    public var body: some View {
        ZStack {
            SystemVibrancyView()
                .edgesIgnoringSafeArea(.all)

            VStack(spacing: 0) {
                toolbarHeader
                Divider()
                filterBar
                Divider()
                logScrollBody
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var toolbarHeader: some View {
        HStack(spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "terminal")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(TelemetryTheme.accent)
                Text("Engine Logs")
                    .font(TelemetryTheme.ui(13, .semibold))
                    .foregroundStyle(TelemetryTheme.primary(increasedContrast: increasedContrast))
                Text("\(filteredEntries.count)")
                    .font(TelemetryTheme.numeric(11, .medium))
                    .foregroundStyle(TelemetryTheme.secondary(increasedContrast: increasedContrast))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(TelemetryTheme.substrate)
                    .clipShape(Capsule())
                sourceBadge
            }

            Spacer(minLength: 8)

            toolbarActions
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(TelemetryTheme.rowSubstrate)
    }

    private var sourceBadge: some View {
        let isSSE = telemetry.isLogStreamConnected
        let label = isSSE ? "SSE Stream" : (telemetry.isConnected ? "Local Process" : "Offline")
        let tint: Color = isSSE ? TelemetryTheme.active : (telemetry.isConnected ? TelemetryTheme.cool : TelemetryTheme.unknown)
        return HStack(spacing: 4) {
            Circle()
                .fill(tint)
                .frame(width: 6, height: 6)
            Text(label)
                .font(TelemetryTheme.ui(10, .medium))
                .foregroundStyle(TelemetryTheme.secondary(increasedContrast: increasedContrast))
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(TelemetryTheme.substrate)
        .clipShape(Capsule())
    }

    private var searchField: some View {
        HStack(spacing: 5) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 10))
                .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
            TextField("Filter logs…", text: $searchText)
                .textFieldStyle(.plain)
                .font(TelemetryTheme.ui(11))
            if !searchText.isEmpty {
                Button(action: { searchText = "" }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 10))
                        .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Clear search")
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .frame(minWidth: 150, maxWidth: 220)
        .background(TelemetryTheme.substrate)
        .clipShape(RoundedRectangle(cornerRadius: TelemetryTheme.Radius.small))
        .overlay(
            RoundedRectangle(cornerRadius: TelemetryTheme.Radius.small)
                .strokeBorder(TelemetryTheme.hairline(increasedContrast: increasedContrast), lineWidth: 1)
        )
    }

    private var toolbarActions: some View {
        HStack(spacing: 5) {
            Toggle("Wrap", isOn: $wrapLines)
                .toggleStyle(.button)
                .controlSize(.small)
                .help("Toggle soft line wrapping")

            Toggle("Follow", isOn: $autoScroll)
                .toggleStyle(.button)
                .controlSize(.small)
                .help(autoScroll ? "Auto-scroll enabled (click to pause)" : "Auto-scroll paused")

            Button(action: revealInFinder) {
                Label("Finder", systemImage: "folder")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .help("Reveal log file in Finder")

            Button(action: copyLogs) {
                Label(showCopiedFeedback ? "Copied" : "Copy", systemImage: showCopiedFeedback ? "checkmark" : "doc.on.doc")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .help("Copy visible log lines to clipboard")

            Button(action: { telemetry.clearLogs() }) {
                Image(systemName: "trash")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .accessibilityLabel("Clear log buffer")
            .help("Clear log viewer buffer")
        }
    }

    private var filterBar: some View {
        HStack(spacing: 6) {
            ForEach(QWLogFilterLevel.allCases) { chip in
                let isSelected = filterLevel == chip
                Button(action: { filterLevel = chip }) {
                    Text(chip.rawValue)
                        .font(TelemetryTheme.ui(10, isSelected ? .semibold : .medium))
                        .foregroundStyle(isSelected
                            ? TelemetryTheme.accent
                            : TelemetryTheme.secondary(increasedContrast: increasedContrast))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(isSelected ? TelemetryTheme.accent.opacity(0.14) : TelemetryTheme.substrate)
                        .clipShape(Capsule())
                        .overlay(
                            Capsule()
                                .strokeBorder(
                                    isSelected
                                        ? TelemetryTheme.accent.opacity(0.45)
                                        : TelemetryTheme.hairline(increasedContrast: increasedContrast),
                                    lineWidth: 1
                                )
                        )
                }
                .buttonStyle(.plain)
            }
            Spacer(minLength: 8)
            searchField
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(TelemetryTheme.rowSubstrate)
    }

    private var logScrollBody: some View {
        ScrollViewReader { proxy in
            Group {
                if filteredEntries.isEmpty {
                    emptyLogView
                } else {
                    ScrollView(wrapLines ? .vertical : [.vertical, .horizontal], showsIndicators: true) {
                        LazyVStack(alignment: .leading, spacing: 3) {
                            ForEach(filteredEntries) { entry in
                                logRowView(entry)
                                    .id(entry.id)
                            }
                        }
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .onChange(of: filteredEntries.last?.id) { _, lastId in
                        guard autoScroll, let lastId else { return }
                        withAnimation(.easeOut(duration: 0.1)) {
                            proxy.scrollTo(lastId, anchor: .bottom)
                        }
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(TelemetryTheme.surface.opacity(0.65))
        }
    }

    private var emptyLogView: some View {
        let hasActiveFilter = filterLevel != .all || !searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        return VStack(spacing: 8) {
            Image(systemName: hasActiveFilter ? "line.3.horizontal.decrease.circle" : "terminal")
                .font(.system(size: 22, weight: .light))
                .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
            Text(hasActiveFilter ? "No log entries match the active filter" : "Waiting for server log output…")
                .font(TelemetryTheme.ui(12, .medium))
                .foregroundStyle(TelemetryTheme.secondary(increasedContrast: increasedContrast))
            if hasActiveFilter {
                Button("Clear Filter") {
                    filterLevel = .all
                    searchText = ""
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func logRowView(_ entry: QWLogEntry) -> some View {
        let color = entryTint(entry)
        return HStack(alignment: .firstTextBaseline, spacing: 8) {
            if let ts = entry.timestamp {
                Text(ts)
                    .font(TelemetryTheme.numeric(11))
                    .foregroundStyle(TelemetryTheme.quiet(increasedContrast: increasedContrast))
                    .lineLimit(1)
                    .fixedSize()
            }
            Text(entry.kind == .http ? "HTTP" : levelLabel(entry.level))
                .font(TelemetryTheme.numeric(10.5, .semibold))
                .foregroundStyle(color)
                .padding(.horizontal, 5)
                .padding(.vertical, 1)
                .frame(minWidth: 48, alignment: .center)
                .background(color.opacity(0.12))
                .clipShape(RoundedRectangle(cornerRadius: 4))
                .lineLimit(1)
                .fixedSize()
            Text(entry.body.isEmpty ? entry.line : entry.body)
                .font(TelemetryTheme.numeric(11))
                .foregroundStyle(entry.kind == .http ? color : TelemetryTheme.primary(increasedContrast: increasedContrast))
                .lineLimit(wrapLines ? nil : 1)
                .fixedSize(horizontal: !wrapLines, vertical: wrapLines)
                .textSelection(.enabled)
        }
        .fixedSize(horizontal: !wrapLines, vertical: false)
    }

    private func levelLabel(_ level: QWLogLevel) -> String {
        switch level {
        case .debug: return "DEBUG"
        case .info: return "INFO"
        case .warn: return "WARN"
        case .error: return "ERROR"
        case .none: return "LOG"
        }
    }

    private func entryTint(_ entry: QWLogEntry) -> Color {
        if entry.kind == .http {
            return entry.level == .error ? TelemetryTheme.failure : TelemetryTheme.active
        }
        switch entry.level {
        case .error: return TelemetryTheme.failure
        case .warn: return TelemetryTheme.warning
        case .debug: return TelemetryTheme.unknown
        case .info, .none: return TelemetryTheme.accent
        }
    }

    private func revealInFinder() {
        let path = telemetry.settings.expandedLogFilePath
        let url = URL(fileURLWithPath: path)
        let dir = url.deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: url.path) {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            FileManager.default.createFile(atPath: url.path, contents: Data())
        }
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    private func copyLogs() {
        let text = filteredEntries.map(\.line).joined(separator: "\n")
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(text, forType: .string)
        showCopiedFeedback = true
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            showCopiedFeedback = false
        }
    }
}
