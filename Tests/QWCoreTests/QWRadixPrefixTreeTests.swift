// QWRadixPrefixTreeTests.swift — Unit tests for 256-token QWRadixPrefixTree and retain checkpoints.

import Foundation
import Testing
@testable import QWCore

@Suite("QWRadixPrefixTree & Retain Points")
struct QWRadixPrefixTreeTests {
    @Test("Page-chunked Radix tree resolves longest ancestor checkpoint, salted base, and page donors")
    func testRadixLookupAndRemoval() {
        let tree = QWRadixPrefixTree<String>(pageTokens: 256)
        let base512 = (0..<512).map { ($0 * 7 + 3) % 251 }
        let ext768 = base512 + (0..<256).map { ($0 * 11 + 5) % 251 }
        let siblingA = ext768 + (0..<80).map { 100 + ($0 % 50) }
        let siblingB = ext768 + (0..<95).map { 150 + ($0 % 50) }

        tree.insert(key: "base_512", tokens: base512, mode: "chunked", allowsSuffix: true, isSharedBase: true)
        tree.insert(key: "sib_a_848", tokens: siblingA, mode: "chunked|salt=u1", allowsSuffix: true, isSharedBase: false)

        // Salted query falls back to unsalted shared base when no checkpoint exists at 768 yet.
        let hitBefore768 = tree.lookupLongestCheckpoint(
            tokens: siblingB, mode: "chunked|salt=u1", fallbackSharedBaseMode: "chunked", allowExact: true
        )
        #expect(hitBefore768?.key == "base_512")
        #expect(hitBefore768?.matchedTokens == 512)
        #expect(hitBefore768?.isSharedBase == true)

        // Pure-attention donor lookup finds the 768-token (3-page) shared boundary from sib_a_848!
        let donorHit = tree.lookupDeepestSharedPage(
            tokens: siblingB, mode: "chunked|salt=u1", fallbackSharedBaseMode: "chunked", minExclusiveTokens: 512
        )
        #expect(donorHit?.donorKey == "sib_a_848")
        #expect(donorHit?.pageAlignedTokens == 768)

        // Branch prefix length detects exact fork at 768 tokens.
        let branchLen = tree.longestBranchPrefixLength(tokens: siblingB, mode: "chunked|salt=u1", minTokens: 128)
        #expect(branchLen == 768)

        // Inserting explicit 768 checkpoint upgrades checkpoint lookup to 768.
        tree.insert(key: "cp_768", tokens: ext768, mode: "chunked|salt=u1", allowsSuffix: true, isSharedBase: false)
        let hitAfter768 = tree.lookupLongestCheckpoint(
            tokens: siblingB, mode: "chunked|salt=u1", fallbackSharedBaseMode: "chunked", allowExact: true
        )
        #expect(hitAfter768?.key == "cp_768")
        #expect(hitAfter768?.matchedTokens == 768)

        // Removing cp_768 and sib_a_848 cleans up subtree cleanly and falls back to base_512.
        tree.remove(key: "cp_768")
        tree.remove(key: "sib_a_848")
        let donorAfterRemove = tree.lookupDeepestSharedPage(
            tokens: siblingB, mode: "chunked|salt=u1", minExclusiveTokens: 512
        )
        #expect(donorAfterRemove == nil)
        let fallbackHit = tree.lookupLongestCheckpoint(
            tokens: siblingB, mode: "chunked|salt=u1", fallbackSharedBaseMode: "chunked", allowExact: true
        )
        #expect(fallbackHit?.key == "base_512")
    }

    @Test("computeRetainPoints preserves sharedBranchPrefixLength, wave prefix, and 4K periodic checkpoints")
    func testComputeRetainPointsOrderingAndPeriodic() {
        let pts = QWTwoTierPrefixCache.computeRetainPoints(
            explicitRetainLength: 1024,
            matchedTokens: 512,
            firstVisualTokenOffset: 10_000,
            promptCount: 10_000,
            sharedBranchPrefixLength: 2300,
            commonWavePrefixLength: 6000,
            periodicCheckpointInterval: 4096
        )
        // Must include 1024 (explicit), 2300 (branch), 4096 (periodic), 6000 (wave), 8192 (periodic), and 9996 (auto).
        #expect(pts == [1024, 2300, 4096, 6000, 8192, 9996])
    }
}
