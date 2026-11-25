// QWMenuBarApp.swift — Menu bar status item, 6-state icon, shortcuts, and windows

import AppKit
import Observation
import QWCore
import QWServer
import SwiftUI

/// Monospaced status item typography and attributed title builder shared by AppKit UI and tests.
@MainActor
public enum QWMenuBarStatusStyle {
    public static let font: NSFont = .monospacedSystemFont(ofSize: 11, weight: .medium)

    public static func attributedTitle(for statusTitle: String) -> NSAttributedString {
        NSAttributedString(
            string: " \(statusTitle)",
            attributes: [.font: font, .kern: 0]
        )
    }
}

public extension Notification.Name {
    static let qwgSettingsWindowDidOpen = Notification.Name("qwgSettingsWindowDidOpen")
}

/// Pure AppKit menu bar entrypoint without a SwiftUI `Settings` scene, preventing a blank duplicate settings window on ⌘,.
@MainActor
public enum QWMenuBarApp {
    private static var retainedDelegate: AppDelegate?

    public static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        retainedDelegate = delegate
        app.delegate = delegate
        app.run()
    }
}

/// Semantic actions bound to QWMenuBar keyboard shortcuts.
public enum QWMenuBarShortcutAction: Equatable, Sendable {
    case restartServer
    case openWebUI
    case openLogs
    case openSettings
    case quitApp
    case closeFrontWindow
    case dismissPopover
}

/// Resolves key-down inputs into QWMenuBar shortcut actions independent of Caps Lock, layout script, or transient flags.
public enum QWMenuBarShortcutRouter {
    private static let relevantModifiers: NSEvent.ModifierFlags = [.command, .option, .control, .shift]
    private static let escapeKeyCode: UInt16 = 53
    private static let ansiKeyMap: [UInt16: String] = [
        43: ",",
        15: "r",
        2: "d",
        37: "l",
        12: "q",
        13: "w",
    ]

    public static func resolve(
        charactersIgnoringModifiers: String?,
        keyCode: UInt16,
        modifierFlags: NSEvent.ModifierFlags,
        isEditingText: Bool,
        isPopoverKey: Bool
    ) -> QWMenuBarShortcutAction? {
        let masked = modifierFlags.intersection(relevantModifiers)
        if keyCode == escapeKeyCode, masked.isEmpty, isPopoverKey {
            return .dismissPopover
        }
        guard masked == [.command] else {
            return nil
        }

        switch normalizedCommandKey(charactersIgnoringModifiers: charactersIgnoringModifiers, keyCode: keyCode) {
        case "r":
            return isEditingText ? nil : .restartServer
        case "d":
            return isEditingText ? nil : .openWebUI
        case "l":
            return isEditingText ? nil : .openLogs
        case ",":
            return .openSettings
        case "q":
            return .quitApp
        case "w":
            return .closeFrontWindow
        default:
            return nil
        }
    }

    private static func normalizedCommandKey(charactersIgnoringModifiers: String?, keyCode: UInt16) -> String? {
        if let raw = charactersIgnoringModifiers?.lowercased(), !raw.isEmpty,
           raw.unicodeScalars.allSatisfy(\.isASCII) {
            return raw
        }
        return ansiKeyMap[keyCode]
    }
}

@MainActor
public final class AppDelegate: NSObject, NSApplicationDelegate, NSPopoverDelegate, NSWindowDelegate, NSMenuItemValidation {
    private var statusItem: NSStatusItem?
    private var popover: NSPopover?
    private let telemetry: TelemetryObservable
    private var keyMonitor: Any?

    private var settingsWindow: NSWindow?
    private var logsWindow: NSWindow?
    private var updateWindow: NSWindow?
    private var isTerminating = false
    private var lastStatusState: MenuBarStatusState?
    private var lastStatusTitle: String?
    private var hasRenderedStatusItem = false

    internal var currentSettingsWindow: NSWindow? { settingsWindow }
    internal var currentUpdateWindow: NSWindow? { updateWindow }

    public init(telemetry: TelemetryObservable = TelemetryObservable()) {
        self.telemetry = telemetry
        super.init()
    }

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
        NSApp.mainMenu = buildMainMenu()
        let logPath = telemetry.settings.expandedLogFilePath
        QWServerLogger.shared.configureFileSink(path: logPath)
        QWServerLogger.shared.log(.info, "Qwegn Menu Bar Controller initialized. Logging to \(logPath)")

        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem = item
        if let button = item.button {
            button.target = self
            button.action = #selector(togglePopover)
        }
        observeStatusItem()

        let createdPopover = NSPopover()
        createdPopover.contentSize = NSSize(width: 440, height: 520)
        createdPopover.behavior = .transient
        createdPopover.animates = true
        createdPopover.delegate = self

        let contentView = CupertinoTelemetryView(
            telemetry: telemetry,
            onOpenWebUI: { [weak self] in self?.openWebUI() },
            onOpenLogs: { [weak self] in self?.openLogsWindow() },
            onOpenSettings: { [weak self] in self?.openSettingsWindow() },
            onQuit: { NSApp.terminate(nil) },
            onOpenUpdates: { [weak self] trigger in self?.openUpdatesWindow(triggerCheck: trigger) }
        )
        let hostingController = NSHostingController(rootView: contentView)
        hostingController.sizingOptions = [.preferredContentSize]
        hostingController.view.wantsLayer = true
        hostingController.view.layer?.backgroundColor = .clear
        createdPopover.contentViewController = hostingController
        popover = createdPopover

        if ProcessInfo.processInfo.environment["QWG_OPEN_POPOVER"] == "1" {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
                guard let self, let button = self.statusItem?.button else { return }
                self.popover?.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
                self.popover?.contentViewController?.view.window?.makeKey()
                NSApp.activate(ignoringOtherApps: true)
                self.telemetry.setPopoverVisible(true)
            }
        }

        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self else { return event }
            return self.handleShortcutEvent(event)
        }
    }

    public func applicationWillTerminate(_ notification: Notification) {
        if let keyMonitor {
            NSEvent.removeMonitor(keyMonitor)
            self.keyMonitor = nil
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
        let statusTitle = telemetry.menuBarStatusTitle
        if hasRenderedStatusItem && state == lastStatusState && statusTitle == lastStatusTitle {
            return
        }

        if !hasRenderedStatusItem || state != lastStatusState {
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
        }

        if !hasRenderedStatusItem || statusTitle != lastStatusTitle {
            if let statusTitle {
                button.imagePosition = .imageLeading
                button.alignment = .left
                button.font = QWMenuBarStatusStyle.font
                button.attributedTitle = QWMenuBarStatusStyle.attributedTitle(for: statusTitle)
            } else {
                button.imagePosition = .imageOnly
                button.attributedTitle = NSAttributedString()
                button.title = ""
            }
        }

        lastStatusState = state
        lastStatusTitle = statusTitle
        hasRenderedStatusItem = true
    }

    internal func handleShortcutEvent(_ event: NSEvent) -> NSEvent? {
        let isPopoverKey = (popover?.isShown == true)
            && (NSApp.keyWindow === popover?.contentViewController?.view.window)
        guard let action = QWMenuBarShortcutRouter.resolve(
            charactersIgnoringModifiers: event.charactersIgnoringModifiers,
            keyCode: event.keyCode,
            modifierFlags: event.modifierFlags,
            isEditingText: isEditingTextResponder(),
            isPopoverKey: isPopoverKey
        ) else {
            return event
        }
        if event.isARepeat {
            return nil
        }
        performShortcutAction(action)
        return nil
    }

    internal func buildMainMenu() -> NSMenu {
        let mainMenu = NSMenu()

        let appMenuItem = NSMenuItem()
        let appMenu = NSMenu(title: "Qwegn")
        appMenu.addItem(makeTargetedItem(title: "Settings…", action: #selector(menuOpenSettings(_:)), keyEquivalent: ","))
        appMenu.addItem(makeTargetedItem(title: "Check for Updates…", action: #selector(menuCheckForUpdates(_:)), keyEquivalent: ""))
        appMenu.addItem(.separator())
        appMenu.addItem(makeTargetedItem(title: "Restart Server", action: #selector(menuRestartServer(_:)), keyEquivalent: "r"))
        appMenu.addItem(makeTargetedItem(title: "Open Web UI", action: #selector(menuOpenWebUI(_:)), keyEquivalent: "d"))
        appMenu.addItem(makeTargetedItem(title: "Capture Screen to Chat…", action: #selector(menuCaptureScreenToChat(_:)), keyEquivalent: ""))
        appMenu.addItem(makeTargetedItem(title: "Show Logs", action: #selector(menuOpenLogs(_:)), keyEquivalent: "l"))
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Quit Qwegn", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appMenuItem.submenu = appMenu
        mainMenu.addItem(appMenuItem)

        let editMenuItem = NSMenuItem()
        let editMenu = NSMenu(title: "Edit")
        editMenu.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        let redoItem = NSMenuItem(title: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        redoItem.keyEquivalentModifierMask = [.command, .shift]
        editMenu.addItem(redoItem)
        editMenu.addItem(.separator())
        editMenu.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        editMenu.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        editMenu.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        editMenu.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editMenuItem.submenu = editMenu
        mainMenu.addItem(editMenuItem)

        let windowMenuItem = NSMenuItem()
        let windowMenu = NSMenu(title: "Window")
        windowMenu.addItem(makeTargetedItem(title: "Close", action: #selector(menuCloseFrontWindow(_:)), keyEquivalent: "w"))
        windowMenu.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        windowMenuItem.submenu = windowMenu
        mainMenu.addItem(windowMenuItem)

        return mainMenu
    }

    public func validateMenuItem(_ menuItem: NSMenuItem) -> Bool {
        let isEditingText = isEditingTextResponder()
        switch menuItem.action {
        case #selector(menuRestartServer(_:)),
             #selector(menuOpenWebUI(_:)),
             #selector(menuOpenLogs(_:)):
            return !isEditingText
        case #selector(menuCloseFrontWindow(_:)):
            return (popover?.isShown == true)
                || settingsWindow?.isVisible == true
                || logsWindow?.isVisible == true
                || updateWindow?.isVisible == true
                || NSApp.keyWindow != nil
        default:
            return true
        }
    }

    private func makeTargetedItem(title: String, action: Selector, keyEquivalent: String) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: action, keyEquivalent: keyEquivalent)
        item.target = self
        return item
    }

    private func isEditingTextResponder() -> Bool {
        let responder = NSApp.keyWindow?.firstResponder ?? popover?.contentViewController?.view.window?.firstResponder
        guard let textResponder = responder as? NSText else { return false }
        return textResponder.isEditable
    }

    private func performShortcutAction(_ action: QWMenuBarShortcutAction) {
        switch action {
        case .restartServer:
            telemetry.restartServerProcess()
        case .openWebUI:
            openWebUI()
        case .openLogs:
            openLogsWindow()
        case .openSettings:
            openSettingsWindow()
        case .quitApp:
            NSApp.terminate(nil)
        case .closeFrontWindow:
            closeFrontWindow()
        case .dismissPopover:
            popover?.performClose(nil)
        }
    }

    private func closeFrontWindow() {
        if let popover, popover.isShown,
           NSApp.keyWindow === popover.contentViewController?.view.window || NSApp.keyWindow == nil {
            popover.performClose(nil)
            return
        }
        if let keyWindow = NSApp.keyWindow {
            keyWindow.performClose(nil)
            return
        }
        if let window = settingsWindow, window.isVisible {
            window.performClose(nil)
        } else if let window = logsWindow, window.isVisible {
            window.performClose(nil)
        } else if let window = updateWindow, window.isVisible {
            window.performClose(nil)
        }
    }

    @objc private func menuOpenSettings(_ sender: Any?) { openSettingsWindow() }
    @objc private func menuCheckForUpdates(_ sender: Any?) { openUpdatesWindow(triggerCheck: true) }
    @objc private func menuRestartServer(_ sender: Any?) { telemetry.restartServerProcess() }
    @objc private func menuOpenWebUI(_ sender: Any?) { openWebUI() }
    @objc private func menuCaptureScreenToChat(_ sender: Any?) { captureScreenToChat() }
    @objc private func menuOpenLogs(_ sender: Any?) { openLogsWindow() }
    @objc private func menuCloseFrontWindow(_ sender: Any?) { closeFrontWindow() }

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
        popover?.performClose(nil)
        telemetry.openWebDashboard()
    }

    public func captureScreenToChat() {
        popover?.performClose(nil)
        let port = telemetry.serverPortString
        Task { @MainActor in
            do {
                let snap = try await QWScreenCaptureHelper.captureActiveDisplay(promptForPermissionIfNeeded: true)
                let pb = NSPasteboard.general
                pb.clearContents()
                pb.setData(snap.pngData, forType: .png)
                if let url = URL(string: "http://127.0.0.1:\(port)/ui?tab=chat") {
                    NSWorkspace.shared.open(url)
                }
            } catch {
                let alert = NSAlert()
                alert.alertStyle = .informational
                alert.messageText = "Screen Recording Permission Needed"
                alert.informativeText = "\(error)"
                alert.addButton(withTitle: "OK")
                alert.runModal()
            }
        }
    }

    @objc private func togglePopover() {
        guard let button = statusItem?.button else { return }
        if popover?.isShown == true {
            popover?.performClose(nil)
        } else {
            popover?.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
            popover?.contentViewController?.view.window?.makeKey()
            NSApp.activate(ignoringOtherApps: true)
            telemetry.setPopoverVisible(true)
        }
    }

    public func openSettingsWindow() {
        popover?.performClose(nil)
        NotificationCenter.default.post(name: .qwgSettingsWindowDidOpen, object: nil)
        if let window = settingsWindow {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let settingsView = SettingsView(
            telemetry: telemetry,
            onClose: { [weak self] in self?.settingsWindow?.close() },
            onOpenUpdates: { [weak self] trigger in self?.openUpdatesWindow(triggerCheck: trigger) }
        )
        let hosting = NSHostingController(rootView: settingsView)
        let window = NSWindow(contentViewController: hosting)
        window.identifier = NSUserInterfaceItemIdentifier("qwg.settings")
        window.title = "Qwegn Settings"
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable]
        window.setContentSize(NSSize(width: 740, height: 540))
        window.minSize = NSSize(width: 660, height: 480)
        window.isMovableByWindowBackground = false
        window.isReleasedWhenClosed = false
        window.delegate = self
        window.center()
        self.settingsWindow = window

        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    public func openUpdatesWindow(triggerCheck: Bool = false) {
        popover?.performClose(nil)
        if triggerCheck {
            telemetry.updater.checkNow(force: true)
        }
        if let window = updateWindow {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let updateView = SoftwareUpdateView(
            telemetry: telemetry,
            onClose: { [weak self] in self?.updateWindow?.close() }
        )
        let hosting = NSHostingController(rootView: updateView)
        let window = NSWindow(contentViewController: hosting)
        window.identifier = NSUserInterfaceItemIdentifier("qwg.updates")
        window.title = "Software Update"
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable]
        window.setContentSize(NSSize(width: 540, height: 460))
        window.minSize = NSSize(width: 460, height: 340)
        window.isMovableByWindowBackground = false
        window.isReleasedWhenClosed = false
        window.delegate = self
        window.center()
        self.updateWindow = window

        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    public func openLogsWindow() {
        popover?.performClose(nil)
        telemetry.setLogWindowVisible(true)
        if let window = logsWindow {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let logsView = LogViewerWindow(telemetry: telemetry)
        let hosting = NSHostingController(rootView: logsView)
        let window = NSWindow(contentViewController: hosting)
        window.identifier = NSUserInterfaceItemIdentifier("qwg.logs")
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
