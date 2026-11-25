// QWConstrainedDraftSeedingTests.swift — Tests for speculative constraint probes and canonical draft seeding.

import XCTest
@testable import QWServer
import QWHead
import QWToken

final class QWConstrainedDraftSeedingTests: XCTestCase {
    private func grammar(_ schema: String) throws -> QWJSONGrammar {
        try QWJSONGrammar.compile(schema: try QWJson.parse(schema))
    }

    func testStructuredOutputProbeCloneDoesNotMutateLiveConstraint() throws {
        let pieces: [String?] = ["{", "}", "\"", "status", "\":\"", "ok", "\",\"", "n", "\":", "1", "<eos>"]
        let table = QWJSONTokenTable(tokenBytes: pieces.map { $0.map { Array($0.utf8) } }, vocabulary: pieces.count)
        let g = try grammar("""
        {"type":"object","properties":{"status":{"enum":["ok"]},"n":{"type":"integer"}},
         "required":["status","n"],"additionalProperties":false}
        """)
        let live = QWStructuredOutputConstraint(grammar: g, table: table, stopTokens: [10])
        XCTAssertTrue(live.allowsSpeculation)
        let beforeAllowed = try live.allowedTokens()

        guard let probe = live.makeSpeculativeProbe() else {
            return XCTFail("expected speculative probe")
        }
        let drafts = probe.proposeCanonicalDrafts(maxCount: 6)
        XCTAssertFalse(drafts.isEmpty)
        XCTAssertFalse(drafts.contains(10), "stop token must never be proposed as a speculative draft")
        for tok in drafts {
            let stepAllowed = try XCTUnwrap(try probe.allowedTokens())
            XCTAssertTrue(stepAllowed.contains(tok), "proposed canonical draft \(tok) must be valid in grammar")
            try probe.accept(tok)
        }
        XCTAssertEqual(try live.allowedTokens(), beforeAllowed, "advancing probe must not mutate live constraint")
        XCTAssertFalse(live.isComplete)
    }

    func testCanonicalDraftsPreserveAlternativeKeyOrderInLiveAllowedSet() throws {
        let pieces: [String?] = ["{\"", "\"", "alpha", "beta", "\":", "true", ",", "}", "{", "<eos>"]
        let table = QWJSONTokenTable(tokenBytes: pieces.map { $0.map { Array($0.utf8) } }, vocabulary: pieces.count)
        let g = try grammar("""
        {"type":"object","properties":{"alpha":{"type":"boolean"},"beta":{"type":"boolean"}},
         "required":["alpha","beta"],"additionalProperties":false}
        """)
        let live = QWStructuredOutputConstraint(grammar: g, table: table, stopTokens: [9])
        try live.accept(0) // {"
        let allowedKeys = try XCTUnwrap(try live.allowedTokens())
        XCTAssertTrue(allowedKeys.contains(2) && allowedKeys.contains(3), "both alpha and beta remain legal")

        let drafts = live.proposeCanonicalDrafts(maxCount: 4)
        XCTAssertEqual(drafts, [2, 4], "canonical seeder proposes first unvisited key + colon without forcing it")
        // Verify model can still pick beta (token 3) on verification mismatch.
        try live.accept(3)
        try live.accept(4)
        try live.accept(5)
        let afterBeta = live.proposeCanonicalDrafts(maxCount: 4)
        XCTAssertEqual(afterBeta, [6, 1, 2, 4], "after beta:true, canonical seeder proposes ,\"alpha\":")
    }

    func testToolCallProbeAndCanonicalLCPSeeding() throws {
        let pieces: [String?] = [
            "<tool_call>", "</tool_call>", "\n", "<function", "=", "search_", "code>", "docs>",
            "<parameter", "=query>", "hello"
        ]
        let table = QWJSONTokenTable(tokenBytes: pieces.map { $0.map { Array($0.utf8) } }, vocabulary: pieces.count)
        let t1 = try QWJson.parse("""
        {"type":"function","function":{"name":"search_code","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}
        """)
        let t2 = try QWJson.parse("""
        {"type":"function","function":{"name":"search_docs","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}
        """)
        let live = try XCTUnwrap(QWToolCallConstraint(
            tools: [t1, t2], forcedToolName: nil, table: table, toolCallToken: 0, toolCallEndToken: 1
        ))
        XCTAssertTrue(live.allowsSpeculation)
        try live.accept(0) // <tool_call>
        try live.accept(2) // \n
        try live.accept(3) // <function

        let before = try live.allowedTokens()
        guard let probe = live.makeSpeculativeProbe() else {
            return XCTFail("expected tool speculative probe")
        }
        let lcpDrafts = probe.proposeCanonicalDrafts(maxCount: 4)
        XCTAssertEqual(lcpDrafts, [4, 5], "must seed longest common prefix '=search_' across candidate tools")
        for tok in lcpDrafts { try probe.accept(tok) }
        XCTAssertEqual(try live.allowedTokens(), before, "tool probe must not mutate live constraint")

        try live.accept(4)
        try live.accept(5)
        try live.accept(6) // code>
        try live.accept(2)
        try live.accept(8) // <parameter
        XCTAssertEqual(live.proposeCanonicalDrafts(maxCount: 4), [9], "single parameter seeds '=query>'")
    }

    func testThinkingBudgetChainsEndThinkIntoInnerCanonicalDrafts() throws {
        let pieces: [String?] = ["think", "</think>", "{\"", "answer", "\":\"", "yes", "\"}", "<eos>"]
        let table = QWJSONTokenTable(tokenBytes: pieces.map { $0.map { Array($0.utf8) } }, vocabulary: pieces.count)
        let g = try grammar("""
        {"type":"object","properties":{"answer":{"enum":["yes"]}},"required":["answer"],"additionalProperties":false}
        """)
        let inner = QWStructuredOutputConstraint(
            grammar: g, table: table, stopTokens: [7], startsInThinking: true, endThinkToken: 1
        )
        let budget = QWThinkingBudgetConstraint(
            maxThinkingTokens: 2,
            endThinkTokenId: 1,
            vocabularySize: pieces.count,
            table: table,
            inner: inner
        )
        try budget.accept(0)
        try budget.accept(0)
        XCTAssertEqual(try budget.allowedTokens(), [1], "at budget ceiling only </think> is allowed")

        let chained = budget.proposeCanonicalDrafts(maxCount: 6)
        XCTAssertEqual(chained, [1, 2, 3, 4, 5, 6], "must chain </think> + full canonical JSON span without <eos>")
        XCTAssertEqual(try budget.allowedTokens(), [1], "proposeCanonicalDrafts must not mutate budget or inner state")

        let heal = QWPrefixHealingConstraint(prefixBytes: Array("{\"answer\":\"".utf8), table: table)
        guard let healProbe = heal.makeSpeculativeProbe() else {
            return XCTFail("expected healing speculative probe")
        }
        let healDrafts = healProbe.proposeCanonicalDrafts(maxCount: 4)
        XCTAssertEqual(healDrafts, [2, 3, 4])
        for tok in healDrafts { try healProbe.accept(tok) }
        XCTAssertFalse(heal.isUnconstrainedNow, "healing probe must not mutate original healing constraint")
    }

    func testPrettyPrintedIndentationStyleLearningAndNestedArraySeeding() throws {
        // Real BPE vocabulary where closing quote + comma + newline is fused into "\",\n"
        let pieces: [String?] = [
            "{\n  \"", "svc", "\": \"", "gw", "\",\n", "  \"", "items", "\": [\n    {\n      \"",
            "ok", "\": ", "true", "\n    }", ",\n    {\n      \"", "false", "\n  ]\n}", "<eos>"
        ]
        let table = QWJSONTokenTable(tokenBytes: pieces.map { $0.map { Array($0.utf8) } }, vocabulary: pieces.count)
        let g = try grammar("""
        {"type":"object","properties":{
            "svc":{"type":"string"},
            "items":{"type":"array","minItems":2,"maxItems":2,"items":{
                "type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false
            }}
        },"required":["svc","items"],"additionalProperties":false}
        """)
        let live = QWStructuredOutputConstraint(grammar: g, table: table, stopTokens: [15])
        XCTAssertTrue(live.proposeCanonicalDrafts(maxCount: 4).isEmpty,
                      "before observing root formatting style when newline tokens exist, yield to neural drafter")
        try live.accept(0) // {\n  "
        try live.accept(1) // svc
        try live.accept(2) // ": "
        try live.accept(3) // gw
        try live.accept(4) // ",\n  (fused closing quote + comma + newline!)
        let drafts = live.proposeCanonicalDrafts(maxCount: 5)
        XCTAssertEqual(drafts, [5, 6, 7, 8, 9], "must deduct consumed newline and seed 2-space pretty nested item 0")
        for tok in drafts { try live.accept(tok) }
        try live.accept(10) // true (boolean completes into .afterValue)
        let item2Drafts = live.proposeCanonicalDrafts(maxCount: 4)
        XCTAssertEqual(item2Drafts, [11, 12, 8, 9], "minItems=2 must close object 0 and seed pretty object 1")
    }
}
