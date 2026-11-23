// QWUpdateUITests.swift — Hermetic tests for QWUpdateController, SoftwareUpdateView, and AppDelegate integration.

import AppKit
import CryptoKit
import Foundation
import SwiftUI
import XCTest
@testable import QWCore
@testable import QWMenuBarUI
@testable import QWServer

private final class MockUIUpdateTransport: QWUpdateHTTPTransport, @unchecked Sendable {
    let payload = Data("ui-pkg-installer-payload".utf8)
    let releaseJSON: Data
    let checksumsText: String

    init(tag: String = "v99.0.0") {
        let sum = SHA256.hash(data: payload).map { String(format: "%02x", $0) }.joined()
        self.checksumsText = "\(sum)  Qwegn-99.0.0-macos-arm64.pkg\n"
        let obj: [String: Any] = [
            "tag_name": tag,
            "name": "Qwegn \(tag)",
            "body": "## What's New\n- Cupertino Software Update window",
            "draft": false,
            "prerelease": false,
            "published_at": "2026-11-23T12:00:00Z",
            "html_url": "https://github.com/qwegnai/qwegn/releases/tag/\(tag)",
            "assets": [
                [
                    "name": "Qwegn-99.0.0-macos-arm64.pkg",
                    "browser_download_url": "https://github.com/qwegnai/qwegn/releases/download/\(tag)/Qwegn-99.0.0-macos-arm64.pkg",
                    "size": payload.count,
                ],
                [
                    "name": "SHA256SUMS.txt",
                    "browser_download_url": "https://github.com/qwegnai/qwegn/releases/download/\(tag)/SHA256SUMS.txt",
                    "size": checksumsText.utf8.count,
                ],
            ],
        ]
        self.releaseJSON = try! JSONSerialization.data(withJSONObject: obj)
    }

    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let url = request.url!
        if url.lastPathComponent == "SHA256SUMS.txt" {
            let resp = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: nil)!
            return (Data(checksumsText.utf8), resp)
        }
        let resp = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: nil)!
        return (releaseJSON, resp)
    }

    func resolveRedirectLocation(for url: URL, userAgent: String) async throws -> URL? { nil }

    func downloadToFile(
        for request: URLRequest,
        destinationPartURL: URL,
        maxBytes: Int64,
        allowedHosts: Set<String>,
        allowLoopbackHTTP: Bool,
        onProgress: @escaping @Sendable (Int64, Int64?) -> Void
    ) async throws -> Int64 {
        try FileManager.default.createDirectory(at: destinationPartURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try payload.write(to: destinationPartURL, options: .atomic)
        onProgress(Int64(payload.count), Int64(payload.count))
        return Int64(payload.count)
    }
}

final class QWUpdateUITests: XCTestCase {
    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-ui-upd-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return dir
    }

    @MainActor
    func testControllerLifecycleAndSkipBehavior() async throws {
        let dir = try makeTempDir()
        let stateURL = dir.appendingPathComponent("update-state.json")
        let cacheDir = dir.appendingPathComponent("cache", isDirectory: true)
        let transport = MockUIUpdateTransport()
        var openedPkg: URL?

        let controller = QWUpdateController(
            schedule: .onLaunchAndDaily,
            currentVersion: "0.32.2",
            installOrigin: .pkgReceipt,
            stateURL: stateURL,
            cacheDirectory: cacheDir,
            checker: QWUpdateChecker(transport: transport),
            openInstallerHandler: { url in
                await MainActor.run { openedPkg = url }
                return true
            }
        )

        await controller.checkNowAndAwait(force: true)
        XCTAssertTrue(controller.status.updateAvailable)
        XCTAssertTrue(controller.shouldShowBadge)
        XCTAssertEqual(controller.badgeText, "↑ v99.0.0")

        controller.skipCurrentUpdate()
        XCTAssertTrue(controller.status.isSkipped)
        XCTAssertFalse(controller.shouldShowBadge)

        controller.clearSkippedVersion()
        XCTAssertFalse(controller.status.isSkipped)
        XCTAssertTrue(controller.shouldShowBadge)

        await controller.startDownloadAndAwait()
        XCTAssertEqual(controller.status.phase, .readyToInstall)
        XCTAssertEqual(controller.badgeText, "✓ v99.0.0")
        XCTAssertNotNil(controller.status.verifiedSHA256)

        controller.openInstaller()
        for _ in 0..<20 where openedPkg == nil {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertEqual(openedPkg?.lastPathComponent, "Qwegn-99.0.0-macos-arm64.pkg")
    }

    @MainActor
    func testAppDelegateMainMenuAndSingletonUpdateWindow() throws {
        _ = NSApplication.shared
        let telemetry = TelemetryObservable(startServices: false, enableAutoLifecycle: false, initialSettings: QWServerSettings())
        let delegate = AppDelegate(telemetry: telemetry)

        let menu = delegate.buildMainMenu()
        let appItems = menu.items.first?.submenu?.items.map(\.title) ?? []
        XCTAssertTrue(appItems.contains("Check for Updates…"))

        delegate.openUpdatesWindow(triggerCheck: false)
        let first = try XCTUnwrap(delegate.currentUpdateWindow)
        delegate.openUpdatesWindow(triggerCheck: false)
        let second = try XCTUnwrap(delegate.currentUpdateWindow)
        XCTAssertTrue(first === second)
        XCTAssertEqual(first.identifier?.rawValue, "qwg.updates")
        XCTAssertEqual(first.title, "Software Update")
        XCTAssertTrue(first.contentViewController is NSHostingController<SoftwareUpdateView>)
        first.makeKeyAndOrderFront(nil)
        XCTAssertTrue(delegate.validateMenuItem(NSMenuItem(title: "Close", action: NSSelectorFromString("menuCloseFrontWindow:"), keyEquivalent: "w")))
        first.close()
    }

    @MainActor
    func testPopoverWithUpdatePillAndSoftwareUpdateViewsFitBounds() async throws {
        let dir = try makeTempDir()
        let stateURL = dir.appendingPathComponent("update-state.json")
        let cacheDir = dir.appendingPathComponent("cache", isDirectory: true)
        let controller = QWUpdateController(
            schedule: .onLaunchAndDaily,
            currentVersion: "0.32.2",
            installOrigin: .pkgReceipt,
            stateURL: stateURL,
            cacheDirectory: cacheDir,
            checker: QWUpdateChecker(transport: MockUIUpdateTransport())
        )
        await controller.checkNowAndAwait(force: true)
        XCTAssertTrue(controller.shouldShowBadge)

        let telemetry = TelemetryObservable(
            startServices: false,
            enableAutoLifecycle: false,
            initialSettings: QWServerSettings(),
            updateController: controller
        )

        let popHost = NSHostingView(rootView: CupertinoTelemetryView(telemetry: telemetry))
        let popSize = popHost.fittingSize
        XCTAssertLessThanOrEqual(popSize.width, 440.5)

        let updHost = NSHostingView(rootView: SoftwareUpdateView(telemetry: telemetry))
        updHost.frame = NSRect(x: 0, y: 0, width: 500, height: 420)
        updHost.layoutSubtreeIfNeeded()
        XCTAssertGreaterThanOrEqual(updHost.fittingSize.width, 460)

        let settingsHost = NSHostingView(rootView: SettingsView(telemetry: telemetry, initialTab: .general))
        settingsHost.frame = NSRect(x: 0, y: 0, width: 740, height: 540)
        settingsHost.layoutSubtreeIfNeeded()
        XCTAssertGreaterThan(settingsHost.bounds.width, 700)
    }
}
