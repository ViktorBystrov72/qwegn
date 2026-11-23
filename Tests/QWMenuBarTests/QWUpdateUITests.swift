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

    init(tag: String = "v99.0.0", body: String = "## What's New\n- Cupertino Software Update window") {
        let sum = SHA256.hash(data: payload).map { String(format: "%02x", $0) }.joined()
        self.checksumsText = "\(sum)  Qwegn-99.0.0-macos-arm64.pkg\n"
        let obj: [String: Any] = [
            "tag_name": tag,
            "name": "Qwegn \(tag)",
            "body": body,
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
        updHost.frame = NSRect(x: 0, y: 0, width: 540, height: 460)
        updHost.layoutSubtreeIfNeeded()
        XCTAssertGreaterThanOrEqual(updHost.fittingSize.width, 460)

        let settingsHost = NSHostingView(rootView: SettingsView(telemetry: telemetry, initialTab: .general))
        settingsHost.frame = NSRect(x: 0, y: 0, width: 740, height: 540)
        settingsHost.layoutSubtreeIfNeeded()
        XCTAssertGreaterThan(settingsHost.bounds.width, 700)
    }

    func testReleaseMarkdownParserCoversReleaseNotesBlocks() {
        XCTAssertTrue(ReleaseMarkdownParser.parse("   \n\t  ").isEmpty)

        let sample = """
        > **Supersedes `v0.34.0`** for macOS 26 release binaries.

        ### Fixed
        - **ABI Fix**: Hoisted `MTLBuffer.contents()` lookups.
          - Sub-detail with `x21` preservation
        - [x] Verified CI gate
        - [ ] Tag next release

        1. Build release binary
        2. Run ABI check

        ```swift
        let ver = QWVersion.current
        ```

        | Component | Status |
        | :--- | ---: |
        | `QWMenuBar` | OK |

        ---

        Summary paragraph line 1
        line 2.
        """
        let blocks = ReleaseMarkdownParser.parse(sample)
        XCTAssertEqual(blocks.count, 8)
        guard blocks.count == 8 else { return }
        XCTAssertEqual(blocks[0], .blockquote(blocks: [.paragraph(text: "**Supersedes `v0.34.0`** for macOS 26 release binaries.")]))
        XCTAssertEqual(blocks[1], .heading(level: 3, text: "Fixed"))
        guard case let .list(firstList) = blocks[2] else {
            return XCTFail("Expected bullet/task list block")
        }
        XCTAssertEqual(firstList, [
            ReleaseMarkdownListItem(depth: 0, marker: .unordered, text: "**ABI Fix**: Hoisted `MTLBuffer.contents()` lookups."),
            ReleaseMarkdownListItem(depth: 1, marker: .unordered, text: "Sub-detail with `x21` preservation"),
            ReleaseMarkdownListItem(depth: 0, marker: .task(checked: true), text: "Verified CI gate"),
            ReleaseMarkdownListItem(depth: 0, marker: .task(checked: false), text: "Tag next release"),
        ])
        XCTAssertEqual(blocks[3], .list(items: [
            ReleaseMarkdownListItem(depth: 0, marker: .ordered(1), text: "Build release binary"),
            ReleaseMarkdownListItem(depth: 0, marker: .ordered(2), text: "Run ABI check"),
        ]))
        XCTAssertEqual(blocks[4], .codeBlock(language: "swift", code: "let ver = QWVersion.current"))
        XCTAssertEqual(blocks[5], .table(
            headers: ["Component", "Status"],
            alignments: [.leading, .trailing],
            rows: [["`QWMenuBar`", "OK"]]
        ))
        XCTAssertEqual(blocks[6], .thematicBreak)
        XCTAssertEqual(blocks[7], .paragraph(text: "Summary paragraph line 1 line 2."))
    }

    func testReleaseMarkdownInlineStylesCodeAndSanitizesLinks() {
        let md = "Use `qwg update`, [Release](https://github.com/qwegnai/qwegn), [JS](javascript:alert(1)), and [Local](file:///etc/passwd)."
        let attr = ReleaseMarkdownInline.attributedString(for: md, baseSize: 11.5)
        let runs = Array(attr.runs)
        let codeRun = runs.first { String(attr[$0.range].characters) == "qwg update" }
        XCTAssertTrue(codeRun?.inlinePresentationIntent?.contains(.code) == true)
        XCTAssertNotNil(codeRun?.swiftUI.backgroundColor)

        let httpsRun = runs.first { String(attr[$0.range].characters) == "Release" }
        XCTAssertEqual(httpsRun?.link?.absoluteString, "https://github.com/qwegnai/qwegn")

        let jsRun = runs.first { String(attr[$0.range].characters) == "JS" }
        XCTAssertNil(jsRun?.link)

        let localRun = runs.first { String(attr[$0.range].characters) == "Local" }
        XCTAssertNil(localRun?.link)
    }

    @MainActor
    func testSoftwareUpdateViewWithRichMarkdownNotesFitsBounds() async throws {
        let dir = try makeTempDir()
        let richNotes = """
        > **Supersedes `v0.34.0`**. Fixes codegen crash.

        ### Fixed
        - **ABI Fix**: Hoisted `MTLBuffer.contents()`.
          - Eliminates sibling tail-call clobbers.

        | Binary | Gate |
        | :--- | ---: |
        | `qwg` | PASS |
        """
        let transport = MockUIUpdateTransport(tag: "v99.1.0", body: richNotes)
        let controller = QWUpdateController(
            schedule: .onLaunchAndDaily,
            currentVersion: "0.34.1",
            installOrigin: .pkgReceipt,
            stateURL: dir.appendingPathComponent("update-state.json"),
            cacheDirectory: dir.appendingPathComponent("cache", isDirectory: true),
            checker: QWUpdateChecker(transport: transport)
        )
        await controller.checkNowAndAwait(force: true)

        let telemetry = TelemetryObservable(
            startServices: false,
            enableAutoLifecycle: false,
            initialSettings: QWServerSettings(),
            updateController: controller
        )
        for size in [NSSize(width: 540, height: 460), NSSize(width: 460, height: 340)] {
            let host = NSHostingView(rootView: SoftwareUpdateView(telemetry: telemetry))
            host.frame = NSRect(origin: .zero, size: size)
            host.layoutSubtreeIfNeeded()
            XCTAssertGreaterThanOrEqual(host.bounds.width, 460)
            XCTAssertGreaterThanOrEqual(host.bounds.height, 340)
        }
    }
}
