// FNContextHeadroomAndBoundsTests.swift — Tests < 32K KV headroom and LRU protected eviction.
import Foundation
import XCTest
import QWAttn
import QWCore
@testable import QWEngine

final class FNContextHeadroomAndBoundsTests: XCTestCase {
    func testResolveKVCapacityAdds1024HeadroomFrom4KWhenRAMAllows() {
        let fixed = 20 * (1 << 30)
        let perTok = FNKVStorageMode.f16.perTokenBytes
        let roomy = 64 * (1 << 30)

        // Contexts < 4096 keep exact aligned capacity.
        XCTAssertEqual(
            QWHyperStreamRuntime.resolveKVCapacity(alignedCap: 2048, fixedBytes: fixed, perTokenBytes: perTok, availableBytes: roomy),
            2048
        )
        // Contexts >= 4096 (including < 32K) gain +1024 decode headroom when RAM permits.
        for cap in [4096, 8192, 16384, 32768] {
            XCTAssertEqual(
                QWHyperStreamRuntime.resolveKVCapacity(alignedCap: cap, fixedBytes: fixed, perTokenBytes: perTok, availableBytes: roomy),
                cap + 1024
            )
        }
    }

    func testResolveKVCapacityFallsBackToAlignedCapWhenTightOnRAM() {
        let fixed = 20 * (1 << 30)
        let perTok = FNKVStorageMode.f16.perTokenBytes
        for cap in [4096, 16384, 32768] {
            let exactAvail = fixed + cap * perTok
            XCTAssertEqual(
                QWHyperStreamRuntime.resolveKVCapacity(alignedCap: cap, fixedBytes: fixed, perTokenBytes: perTok, availableBytes: exactAvail),
                cap
            )
            XCTAssertEqual(
                QWHyperStreamRuntime.resolveKVCapacity(alignedCap: cap, fixedBytes: fixed, perTokenBytes: perTok, availableBytes: exactAvail + 1024 * perTok),
                cap + 1024
            )
        }
    }

    func testWarmStoreEvictOldestEvictsProtectedNonSharedAndCleansSlotMetadata() {
        let store = QWSlotPinnedLRUStore<String, Int>(byteLimit: 1024, maxSlots: 4, rootPrefixTokenCount: 64)
        let h0 = (0..<150).map { Int64($0) }
        let h1 = (0..<150).map { Int64(10_000 + $0) }
        let sharedHist = (0..<128).map { Int64(90_000 + $0) }

        XCTAssertTrue(store.store(key: "shared", value: 99, charge: 100, tokenCount: 128, history: sharedHist, isSharedBase: true))
        XCTAssertTrue(store.store(key: "s0_base", value: 1, charge: 100, tokenCount: 64, slot: 0, rootHash: 1, history: Array(h0.prefix(64))))
        XCTAssertTrue(store.store(key: "s0_last", value: 2, charge: 100, tokenCount: 150, slot: 0, rootHash: 1, history: h0))
        XCTAssertTrue(store.store(key: "s1_last", value: 3, charge: 100, tokenCount: 150, slot: 1, rootHash: 2, history: h1))

        // Unprotected non-shared entry goes first.
        XCTAssertTrue(store.evictOldest())
        XCTAssertFalse(store.contains("s0_base"))
        XCTAssertTrue(store.contains("s0_last"))
        XCTAssertTrue(store.contains("s1_last"))

        // Next, protected non-shared entries evict cleanly in LRU order and clear stale slot metadata.
        XCTAssertTrue(store.evictOldest())
        XCTAssertFalse(store.contains("s0_last"))
        XCTAssertNil(store.sharedBranchPrefixLength(for: h0 + [999]))
        XCTAssertTrue(store.evictOldest())
        XCTAssertFalse(store.contains("s1_last"))
        XCTAssertNil(store.sharedBranchPrefixLength(for: h1 + [999]))

        // Shared base remains pinned and evictOldest returns false once no non-shared entry remains.
        XCTAssertFalse(store.evictOldest())
        XCTAssertTrue(store.contains("shared"))
        XCTAssertEqual(store.chargedBytes, 100)
    }
}
