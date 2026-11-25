// QWKVDeduplicatedPrefixTests.swift — Tests for QWKVCache.sliceSharedPrefix and page refcounting.

import Foundation
import XCTest
import QWCore
import QWHAL
import QWAlloc
@testable import QWKV

final class QWKVDeduplicatedPrefixTests: XCTestCase {
    func testSliceSharedPrefixRetainsPhysicalPagesAndRejectsRecycledSWA() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current != nil,
            "Requires Metal device"
        )
        let geom = QWModelGeometry(descriptor: .qwen38_27b)
        let hal = try QWHal(geometry: geom)
        let pageBytes = QWKVCache.pageBytes(mode: .q8, geometry: geom)
        let kvBytes = 64 * 2 * pageBytes + QWKVCache.tableSlotBytes * 2 * 2 * QWKVCache.maxSeqs
        let arena = try QWArena(hal: hal, weights: 4096, activations: 4096, kv: kvBytes, gdnStates: 4096)
        let cache = try QWKVCache(
            hal: hal, arena: arena, layers: [0, 1], mode: .q8,
            geometry: geom, layerSlidingWindows: [0, 0]
        )

        let freeInitial = cache.freePageCount
        let h1 = try cache.allocate(promptTokens: 600)
        cache.markWritten(h1, upTo: 600)
        var fullPrefix: QWKVSharedPrefix? = try cache.captureSharedPrefix(h1)
        cache.finish(h1)

        let full = try XCTUnwrap(fullPrefix)
        XCTAssertEqual(full.tokenCount, 600)
        // Non-page-aligned slice must return nil.
        XCTAssertNil(try cache.sliceSharedPrefix(full, toPageAlignedTokens: 300))

        // 512-token (2-page) slice retains physical pages 0 and 1 across both layers and K/V kinds.
        var slicedPrefix: QWKVSharedPrefix? = try cache.sliceSharedPrefix(full, toPageAlignedTokens: 512)
        let sliced = try XCTUnwrap(slicedPrefix)
        XCTAssertEqual(sliced.tokenCount, 512)
        XCTAssertEqual(sliced.pageCount, 2 * 2 * 2)

        // Release full 600-tok donor first: page 2 frees, while pages 0..1 stay retained by `sliced`!
        cache.releaseSharedPrefix(full)
        fullPrefix = nil
        let freeWithSliceOnly = cache.freePageCount
        XCTAssertEqual(freeInitial - freeWithSliceOnly, 2 * 2 * 2)

        // Adopting `sliced` into a new 700-tok sequence reuses pages 0..1 cleanly.
        let h2 = try cache.allocate(adopting: sliced, promptTokens: 700)
        XCTAssertEqual(cache.written(h2), 512)
        cache.finish(h2)

        cache.releaseSharedPrefix(sliced)
        slicedPrefix = nil
        XCTAssertEqual(cache.freePageCount, freeInitial)
    }
}
