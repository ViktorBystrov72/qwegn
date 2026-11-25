// QWErnie55ChatTests.swift — Tests ERNIE chatv2p3 template, <ernie:tool_call> stream parsing, and tool constraints.
import XCTest
import QWCore
import QWHead
@testable import QWServer
import QWToken

final class QWErnie55ChatTests: XCTestCase {
    private static let minimalErnieJinja = """
    {# <|ernie:im_start|><ernie:global_setting><ernie:think> #}
    {{ messages[0].content }}
    """

    func testErnieChatTemplateEffortThinkingAndToolsRender() throws {
        let tpl = try QWChatTemplate(source: Self.minimalErnieJinja)
        XCTAssertEqual(tpl.toolCallEnvelope, QWToolCallEnvelope(openTag: "<ernie:tool_call>", closeTag: "</ernie:tool_call>"))
        XCTAssertEqual(tpl.thinkOpenTag, "<ernie:think>")
        XCTAssertEqual(tpl.thinkCloseTag, "</ernie:think>")
        XCTAssertFalse(tpl.usesStandardToolCallHint)
        XCTAssertFalse(tpl.isInkling2Template)
        XCTAssertTrue(tpl.supportsThinkingTags)

        let weatherTool: QWJson = .object([
            ("type", .string("function")),
            ("function", .object([
                ("name", .string("get_weather")),
                ("description", .string("Fetch weather")),
                ("parameters", .object([
                    ("type", .string("object")),
                    ("properties", .object([
                        ("city", .object([("type", .string("string"))])),
                        ("days", .object([("type", .string("integer"))])),
                    ])),
                ])),
            ])),
        ])
        let msgs: [QWJson] = [
            .object([("role", .string("system")), ("content", .string("Be concise."))]),
            .object([("role", .string("user")), ("content", .string("Paris 2 days?"))]),
            .object([
                ("role", .string("assistant")),
                ("reasoning_content", .string("Need weather tool.")),
                ("content", .string("")),
                ("tool_calls", .array([
                    .object([
                        ("id", .string("c1")),
                        ("type", .string("function")),
                        ("function", .object([
                            ("name", .string("get_weather")),
                            ("arguments", .string("{\"days\":2,\"city\":\"Paris\"}")),
                        ])),
                    ]),
                ])),
            ]),
            .object([("role", .string("tool")), ("tool_call_id", .string("c1")), ("content", .string("18C"))]),
            .object([("role", .string("tool")), ("tool_call_id", .string("c2")), ("content", .string("Sunny"))]),
        ]

        var opts = QWChatTemplateOptions()
        opts.tools = .array([weatherTool])
        opts.enableThinking = .bool(true)
        opts.reasoningEffort = .string("xhigh")
        let wireOn = try tpl.render(messages: msgs, options: opts)
        XCTAssertTrue(wireOn.hasPrefix("<|ERNIE_MASK|><|ernie:im_start|>system\n<ernie:system_setting>\nBe concise.\n</ernie:system_setting>\n<ernie:global_setting>\nreasoning_effort=high\nparallel_tool_calls=True\n</ernie:global_setting>\n<ernie:tool_list>\n["))
        XCTAssertTrue(wireOn.contains("<|ernie:im_start|>assistant\n<ernie:think>Need weather tool.</ernie:think><ernie:tool_call>\n<invoke name=\"get_weather\"><parameter name=\"city\">Paris</parameter><parameter name=\"days\">2</parameter></invoke>\n</ernie:tool_call><|ernie:im_end|>"))
        XCTAssertTrue(wireOn.contains("<|ernie:im_start|>tool\n<ernie:tool_output>18C</ernie:tool_output>\n<ernie:tool_output>Sunny</ernie:tool_output><|ernie:im_end|>"))
        XCTAssertTrue(wireOn.hasSuffix("\n<|ernie:im_start|>assistant\n<ernie:think>"))

        for (effort, want) in [("low", "low"), ("medium", "medium"), ("high", "high"), ("xhigh", "high")] {
            var eOpts = QWChatTemplateOptions()
            eOpts.enableThinking = .bool(true)
            eOpts.reasoningEffort = .string(effort)
            let w = try tpl.render(messages: [.object([("role", .string("user")), ("content", .string("Hi"))])], options: eOpts)
            XCTAssertTrue(w.contains("<ernie:global_setting>\nreasoning_effort=\(want)\n</ernie:global_setting>"))
        }

        var offOpts = QWChatTemplateOptions()
        offOpts.enableThinking = .bool(false)
        offOpts.reasoningEffort = .string("ignored_when_off")
        let wireOff = try tpl.render(messages: [.object([("role", .string("user")), ("content", .string("Hi"))])], options: offOpts)
        XCTAssertTrue(wireOff.contains("<ernie:global_setting>\n</ernie:global_setting>"))
        XCTAssertTrue(wireOff.hasSuffix("\n<|ernie:im_start|>assistant\n<ernie:think></ernie:think>"))
    }

    func testStreamParserParallelErnieInvokesAndSchemaCoercion() throws {
        let tool1: QWJson = .object([
            ("type", .string("function")),
            ("function", .object([
                ("name", .string("get_weather")),
                ("parameters", .object([
                    ("type", .string("object")),
                    ("properties", .object([
                        ("city", .object([("type", .string("string"))])),
                        ("days", .object([("type", .string("integer"))])),
                    ])),
                ])),
            ])),
        ])
        let tool2: QWJson = .object([
            ("type", .string("function")),
            ("function", .object([
                ("name", .string("flag_check")),
                ("parameters", .object([
                    ("type", .string("object")),
                    ("properties", .object([
                        ("code", .object([("type", .string("string"))])),
                        ("strict", .object([("type", .string("boolean"))])),
                    ])),
                ])),
            ])),
        ])
        let sample = "<ernie:think>Plan both calls.</ernie:think>Checking now."
            + "<ernie:tool_call>\n"
            + "<invoke name=\"get_weather\"><parameter name=\"city\">123</parameter><parameter name=\"days\">2</parameter></invoke>\n"
            + "<invoke name=\"flag_check\"><parameter name=\"code\">true</parameter><parameter name=\"strict\">true</parameter></invoke>\n"
            + "</ernie:tool_call>"

        let parser = QWChatStreamParser(startsInReasoning: true, allowToolCalls: true, allowReasoningTags: true)
        var deltas: [QWChatStreamParser.Delta] = []
        var idx = sample.startIndex
        while idx < sample.endIndex {
            let next = sample.index(idx, offsetBy: 3, limitedBy: sample.endIndex) ?? sample.endIndex
            deltas.append(contentsOf: parser.feed(String(sample[idx..<next])))
            idx = next
        }
        deltas.append(contentsOf: parser.finish())

        let res = parser.result(tools: [tool1, tool2], truncated: false)
        XCTAssertEqual(res.reasoningContent, "Plan both calls.")
        XCTAssertEqual(res.content, "Checking now.")
        XCTAssertEqual(res.finishReason, "tool_calls")
        XCTAssertEqual(res.toolCalls?.count, 2)
        XCTAssertEqual(res.toolCalls?[0].function.name, "get_weather")
        XCTAssertEqual(res.toolCalls?[0].function.arguments, "{\"city\":\"123\",\"days\":2}")
        XCTAssertEqual(res.toolCalls?[1].function.name, "flag_check")
        XCTAssertEqual(res.toolCalls?[1].function.arguments, "{\"code\":\"true\",\"strict\":true}")
        XCTAssertEqual(QWToolCallParser.firstCompleteToolCallEnd(in: sample), sample.endIndex)
    }

    func testToolCallConstraintEnforcesErnieInvokeAndParameterNames() throws {
        let pieces: [String?] = [
            "<ernie:tool_call>", "</ernie:tool_call>", "\n",
            "<invoke", " name=\"get_weather\">", " name=\"bad_fn\">",
            "<parameter", " name=\"city\">", " name=\"bogus\">",
            "Paris", "</parameter>", "</invoke>",
        ]
        let table = QWJSONTokenTable(tokenBytes: pieces.map { $0.map { Array($0.utf8) } }, vocabulary: pieces.count)
        let toolJson = try QWJson.parse("""
        {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}
        """)
        let c = try XCTUnwrap(QWToolCallConstraint(
            tools: [toolJson], forcedToolName: nil, table: table, toolCallToken: 0, toolCallEndToken: 1
        ))
        try c.accept(0) // <ernie:tool_call>
        try c.accept(2) // \n
        try c.accept(3) // <invoke
        XCTAssertEqual(try c.allowedTokens(), [4], "must allow get_weather, block bad_fn")
        try c.accept(4) //  name="get_weather">
        XCTAssertNil(try c.allowedTokens())
        try c.accept(6) // <parameter
        XCTAssertEqual(try c.allowedTokens(), [7], "must allow city, block bogus")
        try c.accept(7) //  name="city">
        try c.accept(9) // Paris
        try c.accept(10) // </parameter>
        try c.accept(11) // </invoke>
        try c.accept(1) // </ernie:tool_call>
        XCTAssertTrue(c.isUnconstrainedNow)
    }

    func testToolCallConstraintClosesOnSplitErnieEndTag() throws {
        let pieces: [String?] = [
            "<ernie:tool_call>", "\n",
            "<invoke", " name=\"get_weather\">",
            "<parameter", " name=\"city\">",
            "Paris", "</parameter>", "</invoke>",
            "</ernie:", "tool_call>",
        ]
        let table = QWJSONTokenTable(tokenBytes: pieces.map { $0.map { Array($0.utf8) } }, vocabulary: pieces.count)
        let toolJson = try QWJson.parse("""
        {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}
        """)
        let c = try XCTUnwrap(QWToolCallConstraint(
            tools: [toolJson], forcedToolName: nil, table: table, toolCallToken: 0, toolCallEndToken: nil
        ))
        try c.accept(0) // <ernie:tool_call>
        try c.accept(1) // \n
        try c.accept(2) // <invoke
        try c.accept(3) //  name="get_weather">
        try c.accept(4) // <parameter
        try c.accept(5) //  name="city">
        try c.accept(6) // Paris
        try c.accept(7) // </parameter>
        try c.accept(8) // </invoke>
        try c.accept(1) // \n
        XCTAssertFalse(c.isUnconstrainedNow)
        try c.accept(9) // </ernie:
        XCTAssertFalse(c.isUnconstrainedNow)
        try c.accept(10) // tool_call>
        XCTAssertTrue(c.isUnconstrainedNow)
    }

    func testErnieWireMatchesUpstreamEncodingErniePyByteForByte() throws {
        let tpl = try QWChatTemplate(source: Self.minimalErnieJinja)
        let weatherTool = try QWJson.parse("""
        {"type":"function","function":{"name":"get_weather","description":"Fetch weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}
        """)
        let msgs: [QWJson] = [
            .object([("role", .string("system")), ("content", .string("Be concise."))]),
            .object([("role", .string("user")), ("content", .string("Paris and Tokyo weather?"))]),
            .object([
                ("role", .string("assistant")),
                ("reasoning_content", .string("Need two calls.")),
                ("content", .string("")),
                ("tool_calls", .array([
                    .object([
                        ("id", .string("c1")), ("type", .string("function")),
                        ("function", .object([("name", .string("get_weather")), ("arguments", .string("{\"days\":2,\"city\":\"Paris\"}"))])),
                    ]),
                    .object([
                        ("id", .string("c2")), ("type", .string("function")),
                        ("function", .object([("name", .string("get_weather")), ("arguments", .string("{\"days\":1,\"city\":\"Tokyo\"}"))])),
                    ]),
                ])),
            ]),
            .object([("role", .string("tool")), ("tool_call_id", .string("c1")), ("content", .string("18C"))]),
            .object([("role", .string("tool")), ("tool_call_id", .string("c2")), ("content", .string("22C"))]),
        ]
        var opts = QWChatTemplateOptions()
        opts.tools = .array([weatherTool])
        opts.enableThinking = .bool(true)
        opts.reasoningEffort = .string("high")
        let actual = try tpl.render(messages: msgs, options: opts)
        let expected = "<|ERNIE_MASK|><|ernie:im_start|>system\n"
            + "<ernie:system_setting>\nBe concise.\n</ernie:system_setting>\n"
            + "<ernie:global_setting>\nreasoning_effort=high\nparallel_tool_calls=True\n</ernie:global_setting>\n"
            + "<ernie:tool_list>\n[{\"function\": {\"description\": \"Fetch weather\", \"name\": \"get_weather\", \"parameters\": {\"properties\": {\"city\": {\"type\": \"string\"}, \"days\": {\"type\": \"integer\"}}, \"type\": \"object\"}}, \"type\": \"function\"}]\n</ernie:tool_list><|ernie:im_end|>\n"
            + "<|ernie:im_start|>user\nParis and Tokyo weather?<|ernie:im_end|>\n"
            + "<|ernie:im_start|>assistant\n<ernie:think>Need two calls.</ernie:think><ernie:tool_call>\n"
            + "<invoke name=\"get_weather\"><parameter name=\"city\">Paris</parameter><parameter name=\"days\">2</parameter></invoke>\n"
            + "<invoke name=\"get_weather\"><parameter name=\"city\">Tokyo</parameter><parameter name=\"days\">1</parameter></invoke>\n"
            + "</ernie:tool_call><|ernie:im_end|>\n"
            + "<|ernie:im_start|>tool\n<ernie:tool_output>18C</ernie:tool_output>\n<ernie:tool_output>22C</ernie:tool_output><|ernie:im_end|>\n"
            + "<|ernie:im_start|>assistant\n<ernie:think>"
        XCTAssertEqual(actual, expected)
    }

    func testToolCallConstraintHandlesDeepWhitespaceAndJsonTokensOnQoSStack() throws {
        let pieces: [String?] = [
            "<ernie:tool_call>",
            String(repeating: " ", count: 4096),
            String(repeating: "\n", count: 2048),
            "{" + String(repeating: "a", count: 4096),
            "<invoke name=\"get_weather\">",
            "<invoke name=\"bad_fn\">",
            " ",
        ]
        let table = QWJSONTokenTable(tokenBytes: pieces.map { $0.map { Array($0.utf8) } }, vocabulary: pieces.count)
        let toolJson = try QWJson.parse("""
        {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}
        """)
        let constraint = try XCTUnwrap(QWToolCallConstraint(
            tools: [toolJson], forcedToolName: nil, table: table, toolCallToken: 0, toolCallEndToken: nil
        ))
        let exp = expectation(description: "QoS worker allowedTokens completes without stack overflow")
        final class Box: @unchecked Sendable {
            let lock = NSLock()
            var result: Result<[Int]?, any Error>?
        }
        let box = Box()
        DispatchQueue.global(qos: .userInitiated).async {
            let outcome = Result {
                try constraint.accept(0)
                return try constraint.allowedTokens()
            }
            box.lock.withLock { box.result = outcome }
            exp.fulfill()
        }
        wait(for: [exp], timeout: 10.0)
        let allowed = try XCTUnwrap(try box.lock.withLock { box.result }?.get())
        XCTAssertTrue(allowed.contains(1), "4096-space token must remain allowed")
        XCTAssertTrue(allowed.contains(2), "2048-newline token must remain allowed")
        XCTAssertTrue(allowed.contains(3), "long json header token must remain allowed")
        XCTAssertTrue(allowed.contains(4), "<invoke name=\"get_weather\"> must be allowed")
        XCTAssertTrue(allowed.contains(6), "single space must remain allowed")
        XCTAssertFalse(allowed.contains(5), "<invoke name=\"bad_fn\"> must be blocked")
    }

    func testStreamParserIgnoresInk2MarkersInConstrainedNonInklingJSON() throws {
        let markerJSON = #"{"note":"<think>kept</think> </think:opensource> <|channel>thought <|channel|>final <|content_thinking|> to=self<|message|> <tool_call>literal</tool_call>"}"#
        let parsed = QWToolCallParser.parse(
            text: markerJSON, tools: nil, startsInThinking: false,
            truncated: false, allowToolCalls: false, allowReasoningTags: false,
            allowInk2Markers: false
        )
        XCTAssertEqual(parsed.content, markerJSON)
        XCTAssertNil(parsed.reasoningContent)
        XCTAssertEqual(parsed.finishReason, "stop")

        let stream = QWChatStreamParser(
            startsInReasoning: false, allowToolCalls: false,
            allowReasoningTags: false, allowInk2Markers: false
        )
        var streamed = ""
        for idx in stride(from: 0, to: markerJSON.count, by: 5) {
            let start = markerJSON.index(markerJSON.startIndex, offsetBy: idx)
            let end = markerJSON.index(start, offsetBy: 5, limitedBy: markerJSON.endIndex) ?? markerJSON.endIndex
            for delta in stream.feed(String(markerJSON[start..<end])) {
                if case .content(let s) = delta { streamed += s }
                if case .reasoning(let s) = delta { XCTFail("Unexpected reasoning delta: \(s)") }
            }
        }
        for delta in stream.finish() {
            if case .content(let s) = delta { streamed += s }
        }
        XCTAssertEqual(streamed, markerJSON)
    }

    func testToolCallConstraintForcesToolCallTokenWhenRequired() throws {
        let pieces: [String?] = [
            "<ernie:tool_call>", "</ernie:tool_call>", "</ernie:think>",
            "\n", "<invoke", " name=\"get_weather\">",
            "<parameter", " name=\"city\">", "Paris", "</parameter>", "</invoke>",
            "thinking step",
        ]
        let table = QWJSONTokenTable(tokenBytes: pieces.map { $0.map { Array($0.utf8) } }, vocabulary: pieces.count)
        let toolJson = try QWJson.parse("""
        {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}
        """)

        let cNoThink = try XCTUnwrap(QWToolCallConstraint(
            tools: [toolJson], forcedToolName: "", table: table,
            toolCallToken: 0, toolCallEndToken: 1,
            requireToolCall: true, startsInThinking: false, endThinkToken: 2
        ))
        XCTAssertEqual(try cNoThink.allowedTokens(), [0], "must force <ernie:tool_call> immediately when thinking is off")
        XCTAssertTrue(cNoThink.allowsSpeculation)
        XCTAssertFalse(cNoThink.isUnconstrainedNow)
        try cNoThink.accept(0) // <ernie:tool_call>
        try cNoThink.accept(3) // \n
        try cNoThink.accept(4) // <invoke
        XCTAssertEqual(try cNoThink.allowedTokens(), [5])
        try cNoThink.accept(5) //  name="get_weather">
        try cNoThink.accept(1) // </ernie:tool_call>
        XCTAssertTrue(cNoThink.isUnconstrainedNow)

        let cThink = try XCTUnwrap(QWToolCallConstraint(
            tools: [toolJson], forcedToolName: "", table: table,
            toolCallToken: 0, toolCallEndToken: 1,
            requireToolCall: true, startsInThinking: true, endThinkToken: 2
        ))
        XCTAssertNil(try cThink.allowedTokens(), "free during thinking phase")
        try cThink.accept(11) // thinking step
        XCTAssertNil(try cThink.allowedTokens())
        try cThink.accept(2) // </ernie:think>
        XCTAssertEqual(try cThink.allowedTokens(), [0], "must force <ernie:tool_call> once thinking ends")
    }

    func testCustomNamespacedTemplateEnvelopeThinkAndXMLInvokeEndToEnd() throws {
        let acmeJinja = """
        {%- for m in messages -%}{{ m['role'] }}: {{ m['content'] }}\n{%- endfor -%}
        {%- if tools -%}{# <acme:tool_call></acme:tool_call> #}{%- endif -%}
        <acme:think>
        """
        let tpl = try QWChatTemplate(source: acmeJinja)
        XCTAssertEqual(tpl.toolCallEnvelope, QWToolCallEnvelope(openTag: "<acme:tool_call>", closeTag: "</acme:tool_call>"))
        XCTAssertEqual(tpl.thinkOpenTag, "<acme:think>")
        XCTAssertEqual(tpl.thinkCloseTag, "</acme:think>")
        XCTAssertFalse(tpl.usesStandardToolCallHint)
        XCTAssertTrue(tpl.supportsThinkingTags)

        var offOpts = QWChatTemplateOptions()
        offOpts.enableThinking = .bool(false)
        let wireOff = try tpl.render(messages: [.object([("role", .string("user")), ("content", .string("Hi"))])], options: offOpts)
        XCTAssertTrue(wireOff.hasSuffix("<acme:think></acme:think>"))

        let pieces: [String?] = [
            "<acme:tool_call>", "</acme:", "tool_call>", "</acme:think>",
            "\n", "<invoke", " name=\"run_cmd\">", " name=\"bad_cmd\">",
            "<parameter", " name=\"cmd\">", " name=\"bogus\">",
            "ls", "</parameter>", "</invoke>", "plan",
        ]
        let table = QWJSONTokenTable(tokenBytes: pieces.map { $0.map { Array($0.utf8) } }, vocabulary: pieces.count)
        let toolJson = try QWJson.parse("""
        {"type":"function","function":{"name":"run_cmd","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}
        """)
        let tc = try XCTUnwrap(QWToolCallConstraint(
            tools: [toolJson], forcedToolName: nil, table: table,
            toolCallToken: 0, toolCallEndToken: nil,
            toolCallCloseTag: tpl.toolCallEnvelope.closeTag,
            requireToolCall: true, startsInThinking: true,
            endThinkToken: 3, endThinkCloseTag: tpl.thinkCloseTag
        ))
        let budget = QWThinkingBudgetConstraint(
            maxThinkingTokens: 1, endThinkTokenId: 3,
            endThinkCloseTag: tpl.thinkCloseTag, vocabularySize: pieces.count,
            table: table, inner: tc
        )
        try budget.accept(14) // plan (1 thinking token)
        XCTAssertEqual(try budget.allowedTokens(), [3], "budget forces </acme:think>")
        try budget.accept(3) // </acme:think> forwarded to tc -> triggers tool call requirement
        XCTAssertEqual(try budget.allowedTokens(), [0], "tool constraint forces <acme:tool_call>")
        try budget.accept(0) // <acme:tool_call>
        try budget.accept(4) // \n
        try budget.accept(5) // <invoke
        XCTAssertEqual(try budget.allowedTokens(), [6], "must allow run_cmd only")
        try budget.accept(6) //  name="run_cmd">
        try budget.accept(8) // <parameter
        XCTAssertEqual(try budget.allowedTokens(), [9], "must allow cmd only")
        try budget.accept(9) //  name="cmd">
        try budget.accept(11) // ls
        try budget.accept(12) // </parameter>
        try budget.accept(13) // </invoke>
        try budget.accept(1) // </acme:
        XCTAssertFalse(tc.isUnconstrainedNow)
        try budget.accept(2) // tool_call>
        XCTAssertTrue(tc.isUnconstrainedNow)

        let raw = "checking</acme:think>Now.\n<acme:tool_call>\n<invoke name=\"run_cmd\"><parameter name=\"cmd\">ls</parameter></invoke>\n</acme:tool_call>"
        let parser = QWChatStreamParser(
            startsInReasoning: true, allowToolCalls: true, allowReasoningTags: true,
            toolEnvelopes: [tpl.toolCallEnvelope, .standard],
            thinkTags: [("<acme:think>", "</acme:think>")]
        )
        for ch in raw {
            _ = parser.feed(String(ch))
        }
        _ = parser.finish()
        let parsed = parser.result(tools: [toolJson], truncated: false)
        XCTAssertEqual(parsed.reasoningContent, "checking")
        XCTAssertEqual(parsed.content, "Now.")
        XCTAssertEqual(parsed.finishReason, "tool_calls")
        XCTAssertEqual(parsed.toolCalls?.count, 1)
        XCTAssertEqual(parsed.toolCalls?[0].function.name, "run_cmd")
        XCTAssertEqual(parsed.toolCalls?[0].function.arguments, "{\"cmd\":\"ls\"}")
    }

    func testStandardToolEnvelopeParsesParallelXMLInvokes() throws {
        let tool = try QWJson.parse("""
        {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}
        """)
        let raw = "<think>Plan.</think>Calling.\n<tool_call>\n"
            + "<invoke name=\"get_weather\"><parameter name=\"city\">Paris</parameter><parameter name=\"days\">2</parameter></invoke>\n"
            + "<invoke name=\"get_weather\"><parameter name=\"city\">Tokyo</parameter><parameter name=\"days\">3</parameter></invoke>\n"
            + "</tool_call>"
        let parsed = QWToolCallParser.parse(text: raw, tools: [tool], startsInThinking: false)
        XCTAssertEqual(parsed.reasoningContent, "Plan.")
        XCTAssertEqual(parsed.content, "Calling.")
        XCTAssertEqual(parsed.finishReason, "tool_calls")
        XCTAssertEqual(parsed.toolCalls?.count, 2)
        XCTAssertEqual(parsed.toolCalls?[0].function.arguments, "{\"city\":\"Paris\",\"days\":2}")
        XCTAssertEqual(parsed.toolCalls?[1].function.arguments, "{\"city\":\"Tokyo\",\"days\":3}")
    }

    func testJsonToolCallWithLiteralInvokeStringDoesNotMisparseAsXMLInvoke() throws {
        let tool = try QWJson.parse("""
        {"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"content":{"type":"string"}}}}}
        """)
        let raw = #"<tool_call>{"name":"edit","arguments":{"content":"<invoke name=\"x\"><parameter name=\"y\">1</parameter></invoke>"}}</tool_call>"#
        let parsed = QWToolCallParser.parse(text: raw, tools: [tool], startsInThinking: false)
        XCTAssertEqual(parsed.finishReason, "tool_calls")
        XCTAssertEqual(parsed.toolCalls?.count, 1)
        XCTAssertEqual(parsed.toolCalls?[0].function.name, "edit")
        XCTAssertEqual(
            parsed.toolCalls?[0].function.arguments,
            #"{"content":"<invoke name=\"x\"><parameter name=\"y\">1</parameter></invoke>"}"#
        )
    }
}
