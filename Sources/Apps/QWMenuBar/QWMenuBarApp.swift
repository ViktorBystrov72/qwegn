// QWMenuBarApp.swift — Menu bar status item, 6-state icon, shortcuts, and windows

import AppKit
import Observation
import QWCore
import QWServer
import SwiftUI

public extension Notification.Name {
    static let qwgSettingsWindowDidOpen = Notification.Name("qwgSettingsWindowDidOpen")
}

public struct QWMenuBarApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    public init() {}

    public var body: some Scene {
        Settings {
            EmptyView()
        }
    }
}

@MainActor
public final class AppDelegate: NSObject, NSApplicationDelegate, NSPopoverDelegate, NSWindowDelegate {
    private var statusItem: NSStatusItem!
    private var popover: NSPopover!
    private let telemetry = TelemetryObservable()

    private var settingsWindow: NSWindow?
    private var logsWindow: NSWindow?
    private var isTerminating = false

    public func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if isTerminating { return .terminateNow }

        let decision = QWServerLifecycleRules.quitDecision(
            keepServerRunningAfterQuit: telemetry.settings.keepServerRunningAfterQuit,
            parentWatchdogArmed: telemetry.parentPidWatchdogArmed
        )
        if decision == .leaveServerRunning {
            QWServerLogger.shared.log(.info, "Menu bar app quitting: the server keeps running.")
            return .terminateNow
        }
        if !telemetry.isAnyServerRunning {
            return .terminateNow
        }
        if telemetry.settings.keepServerRunningAfterQuit {
            QWServerLogger.shared.log(
                .warn,
                "Menu bar app quitting: server has parent-pid watchdog so it stops now. Restart server to apply Keep Running."
            )
        }

        isTerminating = true
        QWServerLogger.shared.log(.info, "Menu bar app quitting: stopping engine server…")
        Task { @MainActor in
            await telemetry.stopEngineAsync()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }

    public func applicationDidFinishLaunching(_ notification: Notification) {
        do {
            try QWEngineInstanceLock.acquireMenuLock()
        } catch {
            QWServerLogger.shared.log(.warn, "\(error.localizedDescription)")
            NSApp.terminate(nil)
            return
        }

        NSApp.setActivationPolicy(.accessory)
        let logPath = telemetry.settings.expandedLogFilePath
        QWServerLogger.shared.configureFileSink(path: logPath)
        QWServerLogger.shared.log(.info, "Qwegn Menu Bar Controller initialized. Logging to \(logPath)")

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = statusItem.button {
            button.target = self
            button.action = #selector(togglePopover)
        }
        observeStatusItem()

        popover = NSPopover()
        popover.contentSize = NSSize(width: 440, height: 520)
        popover.behavior = .transient
        popover.animates = true
        popover.delegate = self

        let contentView = CupertinoTelemetryView(
            telemetry: telemetry,
            onOpenWebUI: { [weak self] in self?.openWebUI() },
            onOpenLogs: { [weak self] in self?.openLogsWindow() },
            onOpenSettings: { [weak self] in self?.openSettingsWindow() },
            onQuit: { NSApp.terminate(nil) }
        )
        let hostingController = NSHostingController(rootView: contentView)
        hostingController.sizingOptions = [.preferredContentSize]
        hostingController.view.wantsLayer = true
        hostingController.view.layer?.backgroundColor = .clear
        popover.contentViewController = hostingController

        if ProcessInfo.processInfo.environment["QWG_OPEN_POPOVER"] == "1" {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
                guard let self, let button = self.statusItem.button else { return }
                self.popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
                self.telemetry.setPopoverVisible(true)
            }
        }

        NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            self?.handleShortcutEvent(event) ?? event
        }
    }

    @MainActor
    private func observeStatusItem() {
        withObservationTracking {
            updateStatusItem()
        } onChange: { [weak self] in
            Task { @MainActor [weak self] in
                self?.observeStatusItem()
            }
        }
    }

    private func updateStatusItem() {
        guard let button = statusItem?.button else { return }
        let state = telemetry.menuBarStatusState
        let dotColor: NSColor? = switch state {
        case .offline: nil
        case .standby, .transitioning: .systemOrange
        case .ready: .systemGreen
        case .prefilling: .systemYellow
        case .decoding: .systemBlue
        case .failed: .systemRed
        }
        button.image = QGProcessorLogoView.statusBarImage(
            pointSize: 18,
            statusDotColor: dotColor,
            dimmed: state == .offline
        )
        if let statusTitle = telemetry.menuBarStatusTitle {
            button.imagePosition = .imageLeading
            button.font = .monospacedDigitSystemFont(ofSize: 11, weight: .medium)
            button.title = " \(statusTitle)"
        } else {
            button.imagePosition = .imageOnly
            button.title = ""
        }
    }

    private func handleShortcutEvent(_ event: NSEvent) -> NSEvent? {
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        guard flags == [.command],
              let key = event.charactersIgnoringModifiers?.lowercased() else {
            return event
        }
        let isEditingText = NSApp.keyWindow?.firstResponder is NSTextView
        switch key {
        case "r":
            if isEditingText { return event }
            telemetry.restartServerProcess()
            return nil
        case "d":
            if isEditingText { return event }
            openWebUI()
            return nil
        case "l":
            if isEditingText { return event }
            openLogsWindow()
            return nil
        case ",":
            openSettingsWindow()
            return nil
        case "q":
            NSApp.terminate(nil)
            return nil
        default:
            return event
        }
    }

    public func popoverDidShow(_ notification: Notification) {
        telemetry.setPopoverVisible(true)
    }

    public func popoverDidClose(_ notification: Notification) {
        telemetry.setPopoverVisible(false)
    }

    public func windowWillClose(_ notification: Notification) {
        guard let closed = notification.object as? NSWindow else { return }
        if closed === logsWindow {
            telemetry.setLogWindowVisible(false)
        }
    }

    public func openWebUI() {
        popover.performClose(nil)
        telemetry.openWebDashboard()
    }

    @objc private func togglePopover() {
        guard let button = statusItem.button else { return }
        if popover.isShown {
            popover.performClose(nil)
        } else {
            popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
            popover.contentViewController?.view.window?.makeKey()
            telemetry.setPopoverVisible(true)
        }
    }

    public func openSettingsWindow() {
        popover.performClose(nil)
        NotificationCenter.default.post(name: .qwgSettingsWindowDidOpen, object: nil)
        if let window = settingsWindow {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let settingsView = SettingsView(
            telemetry: telemetry,
            onClose: { [weak self] in self?.settingsWindow?.close() }
        )
        let hosting = NSHostingController(rootView: settingsView)
        let window = NSWindow(contentViewController: hosting)
        window.title = "Qwegn Settings"
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable]
        window.setContentSize(NSSize(width: 740, height: 540))
        window.minSize = NSSize(width: 660, height: 480)
        window.isMovableByWindowBackground = false
        window.isReleasedWhenClosed = false
        window.center()
        self.settingsWindow = window

        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    public func openLogsWindow() {
        popover.performClose(nil)
        telemetry.setLogWindowVisible(true)
        if let window = logsWindow {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let logsView = LogViewerWindow(telemetry: telemetry)
        let hosting = NSHostingController(rootView: logsView)
        let window = NSWindow(contentViewController: hosting)
        window.title = "Qwegn Logs"
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable]
        window.setContentSize(NSSize(width: 760, height: 500))
        window.minSize = NSSize(width: 600, height: 380)
        window.isMovableByWindowBackground = false
        window.isReleasedWhenClosed = false
        window.delegate = self
        window.center()
        self.logsWindow = window

        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}
