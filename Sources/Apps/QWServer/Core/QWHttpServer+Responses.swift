// QWHttpServer+Responses.swift — OpenAI POST /v1/responses, GET /v1/responses/:id,
// DELETE /v1/responses/:id, previous_response_id chaining, background mode, and SSE events.

import Foundation
import Network
import QWCore
import QWEngine
import QWToken
import QWVision

extension QWHttpServer {

    func handleCreateResponse(connection: NWConnection, headers: [String: String] = [:], body: Data) async throws {
        let req: OpenAIResponseCreateRequest
        do {
            req = try JSONDecoder().decode(OpenAIResponseCreateRequest.self, from: body)
        } catch {
            sendOpenAIError(connection, status: 400, message: "Malformed JSON payload: \(error.localizedDescription)")
            return
        }

        if let maxOut = req.maxOutputTokens, maxOut <= 0 {
            sendOpenAIError(connection, status: 400, message: "max_output_tokens must be greater than 0")
            return
        }
        if (req.background ?? false) && (req.store == false) {
            sendOpenAIError(connection, status: 400, message: "background mode requires store to be true")
            return
        }

        let prevEntry: QWResponsesStore.StoredResponse?
        if let prevId = req.previousResponseId?.trimmingCharacters(in: .whitespacesAndNewlines), !prevId.isEmpty {
            guard let found = responsesStore.get(prevId) else {
                sendOpenAIError(connection, status: 400,
                                message: "Previous response '\(prevId)' was not found.",
                                code: "previous_response_not_found")
                return
            }
            guard found.resource.status == "completed" || found.resource.status == "incomplete" else {
                sendOpenAIError(connection, status: 400,
                                message: "Previous response '\(prevId)' is in status '\(found.resource.status)'.")
                return
            }
            prevEntry = found
        } else {
            prevEntry = nil
        }

        let inputMessages: [ChatMessage]
        do {
            inputMessages = try OpenAIResponsesMapper.parseInput(req.input, hasPrevious: prevEntry != nil)
        } catch {
            sendOpenAIError(connection, status: 400, message: "\(error)")
            return
        }

        var conversationMessages = prevEntry?.historyMessages ?? []
        conversationMessages.append(contentsOf: inputMessages)
        guard !conversationMessages.isEmpty else {
            sendOpenAIError(connection, status: 400, message: "input must not be empty")
            return
        }

        var fullMessages = conversationMessages
        if let instructions = req.instructions, !instructions.isEmpty {
            fullMessages.insert(ChatMessage(role: "system", text: instructions), at: 0)
        }

        let (normalizedTools, resourceTools, normalizedToolChoice, resourceToolChoice) =
            OpenAIResponsesMapper.normalizeTools(tools: req.tools, toolChoice: req.toolChoice, maxToolCalls: req.maxToolCalls)

        let chatReq: ChatCompletionRequest
        let thinkingExplicitlyDisabled: Bool
        do {
            (chatReq, thinkingExplicitlyDisabled) = try OpenAIResponsesMapper.buildChatRequest(
                from: req, messages: fullMessages, tools: normalizedTools,
                toolChoice: normalizedToolChoice, headers: headers
            )
            try chatReq.validateMetadata()
            _ = try QWStructuredOutput.grammar(for: chatReq)
            try actor.validateRequestCapabilities(chatReq)
        } catch {
            sendOpenAIError(connection, status: 400, message: "\(error)")
            return
        }

        let respId = chatReq.effectiveRequestId ?? ("resp_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))
        let replyModel = resolveReplyModelName(requested: req.model, strictNotFound: false) ?? actor.getModelName()
        var extraHeaders = ["x-request-id: \(respId)"]
        if let sid = chatReq.effectiveSessionId { extraHeaders.append("x-session-id: \(sid)") }

        let serverCfg = actor.getConfig()
        let template = OpenAIResponseTemplate(
            id: respId, createdAt: Int(Date().timeIntervalSince1970), model: replyModel, req: req,
            tools: resourceTools, toolChoice: resourceToolChoice,
            resolvedTemperature: req.temperature ?? serverCfg.temperature,
            resolvedTopP: req.topP ?? serverCfg.topP,
            textFormat: OpenAIResponsesMapper.resourceTextConfig(from: req.text)
        )
        let shouldStore = req.store ?? true
        let hasActiveTools = !(normalizedTools?.isEmpty ?? true) && !chatReq.isToolChoiceNone
        let isConstrainedJson = (chatReq.responseFormatType == "json_schema" || chatReq.responseFormatType == "json_object") && !hasActiveTools
        let (policyThink, _) = chatReq.resolvedThinkingPolicy(
            defaultEnableThinking: serverCfg.enableThinking,
            defaultMaxTokens: serverCfg.maxTokens,
            modelMaxLength: serverCfg.modelMaxLength
        )
        let shouldThink = policyThink && actor.supportsThinkingTags()
        let allowReasoningTags = (!isConstrainedJson || shouldThink) && !thinkingExplicitlyDisabled

        if (req.background ?? false) && !(req.stream ?? false) {
            let queued = template.makeResource(status: "queued", completedAt: nil, output: [], usage: nil)
            responsesStore.put(QWResponsesStore.StoredResponse(resource: queued, historyMessages: conversationMessages))
            let bgTask = Task { [weak self] in
                guard let self else { return }
                await self.runBackgroundResponse(
                    template: template, chatReq: chatReq, baseHistory: conversationMessages,
                    hasActiveTools: hasActiveTools, allowReasoningTags: allowReasoningTags,
                    thinkingExplicitlyDisabled: thinkingExplicitlyDisabled
                )
            }
            responsesStore.attachTask(id: respId, task: bgTask)
            sendJsonResponse(connection, status: 200, object: queued, extraHeaders: extraHeaders)
            return
        }

        if req.stream ?? false {
            try await runStreamingResponse(
                connection: connection, extraHeaders: extraHeaders, template: template,
                chatReq: chatReq, baseHistory: conversationMessages, shouldStore: shouldStore,
                hasActiveTools: hasActiveTools, allowReasoningTags: allowReasoningTags,
                thinkingExplicitlyDisabled: thinkingExplicitlyDisabled
            )
            return
        }

        let shouldStop = generationShouldStop(for: connection)
        let result: ChatCompletionExecutionResult
        do {
            result = try await actor.handleChatCompletion(request: chatReq, onToken: nil, shouldStop: shouldStop)
        } catch {
            if shouldStop() { return }
            sendChatError(connection, error: error, path: "/v1/responses")
            return
        }
        guard !shouldStop() else { return }

        let (finalResource, nextHistory) = Self.finalizeResponse(
            template: template, req: req, chatReq: chatReq, result: result, baseHistory: conversationMessages,
            hasActiveTools: hasActiveTools, allowReasoningTags: allowReasoningTags,
            thinkingExplicitlyDisabled: thinkingExplicitlyDisabled
        )
        if shouldStore {
            responsesStore.put(QWResponsesStore.StoredResponse(resource: finalResource, historyMessages: nextHistory))
        }
        sendJsonResponse(connection, status: 200, object: finalResource, extraHeaders: extraHeaders)
    }

    func handleGetResponse(connection: NWConnection, responseId: String) {
        guard let stored = responsesStore.get(responseId) else {
            sendOpenAIError(connection, status: 404, message: "Response with id '\(responseId)' not found.", code: "not_found")
            return
        }
        sendJsonResponse(connection, status: 200, object: stored.resource)
    }

    func handleDeleteResponse(connection: NWConnection, responseId: String) {
        guard responsesStore.delete(responseId) else {
            sendOpenAIError(connection, status: 404, message: "Response with id '\(responseId)' not found.", code: "not_found")
            return
        }
        let payload: [String: Any] = ["id": responseId, "object": "response", "deleted": true]
        if let data = try? JSONSerialization.data(withJSONObject: payload) {
            sendRawData(connection, status: 200, contentType: "application/json", data: data)
        } else {
            sendSimpleResponse(connection, status: 200, body: "{\"id\":\"\(responseId)\",\"object\":\"response\",\"deleted\":true}")
        }
    }

    private func runBackgroundResponse(
        template: OpenAIResponseTemplate,
        chatReq: ChatCompletionRequest,
        baseHistory: [ChatMessage],
        hasActiveTools: Bool,
        allowReasoningTags: Bool,
        thinkingExplicitlyDisabled: Bool
    ) async {
        do {
            let result = try await actor.handleChatCompletion(
                request: chatReq, onToken: nil, shouldStop: { Task.isCancelled }
            )
            if Task.isCancelled { return }
            let (finalResource, nextHistory) = Self.finalizeResponse(
                template: template, req: template.req, chatReq: chatReq, result: result, baseHistory: baseHistory,
                hasActiveTools: hasActiveTools, allowReasoningTags: allowReasoningTags,
                thinkingExplicitlyDisabled: thinkingExplicitlyDisabled
            )
            responsesStore.update(id: template.id, resource: finalResource, historyMessages: nextHistory)
        } catch {
            if Task.isCancelled { return }
            let code = (error is QWServerContextLengthError) ? "context_length_exceeded" : "server_error"
            let failed = template.makeResource(
                status: "failed", completedAt: nil, output: [], usage: nil,
                error: OpenAIResponseResource.ErrorInfo(code: code, message: "\(error)")
            )
            responsesStore.update(id: template.id, resource: failed)
        }
    }

    private func runStreamingResponse(
        connection: NWConnection,
        extraHeaders: [String],
        template: OpenAIResponseTemplate,
        chatReq: ChatCompletionRequest,
        baseHistory: [ChatMessage],
        shouldStore: Bool,
        hasActiveTools: Bool,
        allowReasoningTags: Bool,
        thinkingExplicitlyDisabled: Bool
    ) async throws {
        let shouldStop = generationShouldStop(for: connection)
        let serverConfig = actor.getConfig()
        let isConstrainedJson = (chatReq.responseFormatType == "json_schema" || chatReq.responseFormatType == "json_object") && !hasActiveTools
        let (shouldThink, _) = chatReq.resolvedThinkingPolicy(
            defaultEnableThinking: serverConfig.enableThinking,
            defaultMaxTokens: serverConfig.maxTokens,
            modelMaxLength: serverConfig.modelMaxLength
        )
        let initialShouldThink = shouldThink && actor.supportsThinkingTags() && !thinkingExplicitlyDisabled
        let streamParser = QWChatStreamParser(
            startsInReasoning: initialShouldThink,
            allowToolCalls: hasActiveTools,
            allowReasoningTags: allowReasoningTags || (initialShouldThink && !thinkingExplicitlyDisabled)
        )
        let emitter = ResponsesSSECoordinator(connection: connection, template: template)
        defer { emitter.stopAll() }

        let result: ChatCompletionExecutionResult
        do {
            result = try await actor.handleChatCompletion(
                request: chatReq,
                onStreamReady: { [weak self] in
                    guard let self else { return }
                    let readyShouldThink = shouldThink && self.actor.supportsThinkingTags() && !thinkingExplicitlyDisabled
                    let readyAllowTags = (!isConstrainedJson || readyShouldThink) && !thinkingExplicitlyDisabled
                    streamParser.setStartsInReasoning(readyShouldThink, allowReasoningTags: readyAllowTags)
                    self.sendSSEHeaders(connection, extraHeaders: extraHeaders)
                    emitter.emitLifecycleEvent(type: "response.created", status: "in_progress")
                    emitter.emitLifecycleEvent(type: "response.in_progress", status: "in_progress")
                    let (gate, task) = self.startSSEKeepalive(connection: connection)
                    emitter.markStarted(gate: gate, task: task)
                },
                onToken: { piece in
                    emitter.stopKeepaliveIfRunning()
                    for delta in streamParser.feed(piece) {
                        emitter.handleDelta(delta, allowReasoning: !thinkingExplicitlyDisabled)
                    }
                },
                shouldStop: shouldStop
            )
        } catch {
            if shouldStop() { return }
            if !emitter.isStarted {
                sendChatError(connection, error: error, path: "/v1/responses")
                return
            }
            let code = (error is QWServerContextLengthError) ? "context_length_exceeded" : "server_error"
            let failed = template.makeResource(
                status: "failed", completedAt: nil, output: [], usage: nil,
                error: OpenAIResponseResource.ErrorInfo(code: code, message: "\(error)")
            )
            emitter.emitTerminalEvent(type: "response.failed", resource: failed)
            emitter.sendDoneAndClose()
            return
        }

        guard !shouldStop() else { return }
        for delta in streamParser.finish() {
            emitter.handleDelta(delta, allowReasoning: !thinkingExplicitlyDisabled)
        }
        let parsed = streamParser.result(tools: hasActiveTools ? chatReq.tools : nil, truncated: result.truncated)
        let cappedCalls = OpenAIResponsesMapper.capToolCalls(
            parsed.toolCalls, parallelToolCalls: template.req.parallelToolCalls, maxToolCalls: template.req.maxToolCalls
        )
        let itemStatus = result.truncated ? "incomplete" : "completed"
        emitter.finishContentItems(status: itemStatus)
        if let calls = cappedCalls {
            emitter.emitFunctionCalls(calls, status: itemStatus)
        }
        emitter.ensureAtLeastOneItem(status: itemStatus)

        let (finalResource, nextHistory) = Self.buildFinalizedResource(
            template: template, parsed: parsed, effectiveCalls: cappedCalls, result: result,
            baseHistory: baseHistory, thinkingExplicitlyDisabled: thinkingExplicitlyDisabled,
            streamedItemIDs: emitter.snapshotItemIDs
        )
        if shouldStore {
            responsesStore.put(QWResponsesStore.StoredResponse(resource: finalResource, historyMessages: nextHistory))
        }
        let terminalType = result.truncated ? "response.incomplete" : "response.completed"
        emitter.emitTerminalEvent(type: terminalType, resource: finalResource)
        emitter.sendDoneAndClose()
    }

    static func finalizeResponse(
        template: OpenAIResponseTemplate,
        req: OpenAIResponseCreateRequest,
        chatReq: ChatCompletionRequest,
        result: ChatCompletionExecutionResult,
        baseHistory: [ChatMessage],
        hasActiveTools: Bool,
        allowReasoningTags: Bool,
        thinkingExplicitlyDisabled: Bool
    ) -> (OpenAIResponseResource, [ChatMessage]) {
        let isConstrainedJson = (chatReq.responseFormatType == "json_schema" || chatReq.responseFormatType == "json_object") && !hasActiveTools
        let effectiveStartsInThinking = result.startsInThinking && !thinkingExplicitlyDisabled
        let effectiveAllowTags = allowReasoningTags || ((!isConstrainedJson || effectiveStartsInThinking) && !thinkingExplicitlyDisabled)
        let parsed = QWToolCallParser.parse(
            text: result.content, tools: chatReq.tools,
            startsInThinking: effectiveStartsInThinking,
            truncated: result.truncated, allowToolCalls: hasActiveTools, allowReasoningTags: effectiveAllowTags
        )
        let cappedCalls = OpenAIResponsesMapper.capToolCalls(
            parsed.toolCalls, parallelToolCalls: req.parallelToolCalls, maxToolCalls: req.maxToolCalls
        )
        return buildFinalizedResource(
            template: template, parsed: parsed, effectiveCalls: cappedCalls, result: result,
            baseHistory: baseHistory, thinkingExplicitlyDisabled: thinkingExplicitlyDisabled
        )
    }

    struct StreamedItemIDs: Sendable {
        var reasoningId: String? = nil
        var messageId: String? = nil
        var functionCallIds: [String] = []
    }

    static func buildFinalizedResource(
        template: OpenAIResponseTemplate,
        parsed: QWToolCallParser.ParseResult,
        effectiveCalls: [ToolCall]?,
        result: ChatCompletionExecutionResult,
        baseHistory: [ChatMessage],
        thinkingExplicitlyDisabled: Bool,
        streamedItemIDs: StreamedItemIDs? = nil
    ) -> (OpenAIResponseResource, [ChatMessage]) {
        let itemStatus = result.truncated ? "incomplete" : "completed"
        var outputItems: [QWJson] = []
        let reasoningText = thinkingExplicitlyDisabled ? nil : parsed.reasoningContent
        if let rText = reasoningText, !rText.isEmpty {
            let rid = streamedItemIDs?.reasoningId ?? ("rs_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(16))
            outputItems.append(.object([
                ("type", .string("reasoning")),
                ("id", .string(rid)),
                ("summary", .array([.object([("type", .string("summary_text")), ("text", .string(rText))])]))
            ]))
        }
        let msgId = streamedItemIDs?.messageId ?? ("msg_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(16))
        if let text = parsed.content, !text.isEmpty {
            outputItems.append(makeMessageOutputItem(id: msgId, text: text, status: itemStatus))
        } else if effectiveCalls?.isEmpty ?? true {
            outputItems.append(makeMessageOutputItem(id: msgId, text: parsed.content ?? "", status: itemStatus))
        }
        for (idx, call) in (effectiveCalls ?? []).enumerated() {
            let fcid = (streamedItemIDs?.functionCallIds.indices.contains(idx) == true
                ? streamedItemIDs?.functionCallIds[idx] : nil)
                ?? ("fc_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(16))
            outputItems.append(.object([
                ("type", .string("function_call")),
                ("id", .string(fcid)),
                ("call_id", .string(call.id)),
                ("name", .string(call.function.name)),
                ("arguments", .string(call.function.arguments)),
                ("status", .string(itemStatus))
            ]))
        }

        let rToks = thinkingExplicitlyDisabled ? 0 : (result.reasoningTokens ?? 0)
        let usage = OpenAIResponseResource.UsageInfo(
            inputTokens: result.promptTokens, outputTokens: result.completionTokens,
            cachedTokens: result.cachedTokens, reasoningTokens: rToks
        )
        let incomplete = result.truncated
            ? OpenAIResponseResource.IncompleteDetails(reason: "max_output_tokens")
            : nil
        let resource = template.makeResource(
            status: result.truncated ? "incomplete" : "completed",
            completedAt: Int(Date().timeIntervalSince1970),
            output: outputItems,
            usage: usage,
            incompleteDetails: incomplete
        )

        var nextHistory = baseHistory
        nextHistory.append(ChatMessage(
            role: "assistant",
            content: parsed.content.map { ChatMessageContent(string: $0) },
            toolCalls: effectiveCalls,
            reasoningContent: reasoningText
        ))
        return (resource, nextHistory)
    }

    private static func makeMessageOutputItem(id: String, text: String, status: String) -> QWJson {
        .object([
            ("type", .string("message")),
            ("id", .string(id)),
            ("status", .string(status)),
            ("role", .string("assistant")),
            ("content", .array([.object([
                ("type", .string("output_text")), ("text", .string(text)), ("annotations", .array([]))
            ])]))
        ])
    }
}

private final class ResponsesSSECoordinator: @unchecked Sendable {
    private struct LifecyclePayload: Encodable {
        let type: String
        let sequenceNumber: Int
        let response: OpenAIResponseResource
        enum CodingKeys: String, CodingKey {
            case type, response
            case sequenceNumber = "sequence_number"
        }
    }

    private let lock = NSLock()
    private let connection: NWConnection
    private let template: OpenAIResponseTemplate
    private var started = false
    private var keepaliveGate: QWStreamKeepaliveGate?
    private var keepaliveTask: Task<Void, Never>?
    private var seq = 0
    private var nextOutputIndex = 0
    private var reasoningId: String? = nil
    private var reasoningIndex: Int? = nil
    private var reasoningText = ""
    private var reasoningClosed = false
    private var messageId: String? = nil
    private var messageIndex: Int? = nil
    private var messageText = ""
    private var messageClosed = false
    private var functionCallIds: [String] = []

    init(connection: NWConnection, template: OpenAIResponseTemplate) {
        self.connection = connection
        self.template = template
    }

    var isStarted: Bool { lock.withLock { started } }

    var snapshotItemIDs: QWHttpServer.StreamedItemIDs {
        lock.withLock {
            QWHttpServer.StreamedItemIDs(
                reasoningId: reasoningId,
                messageId: messageId,
                functionCallIds: functionCallIds
            )
        }
    }

    func markStarted(gate: QWStreamKeepaliveGate, task: Task<Void, Never>) {
        lock.withLock {
            started = true
            keepaliveGate = gate
            keepaliveTask = task
        }
    }

    func stopKeepaliveIfRunning() {
        let t = lock.withLock { () -> Task<Void, Never>? in
            guard let g = keepaliveGate, g.stopIfRunning() else { return nil }
            return keepaliveTask
        }
        t?.cancel()
    }

    func stopAll() {
        let (g, t) = lock.withLock { (keepaliveGate, keepaliveTask) }
        g?.stop()
        t?.cancel()
    }

    func emitLifecycleEvent(type: String, status: String) {
        let resource = template.makeResource(status: status, completedAt: nil, output: [], usage: nil)
        let s = lock.withLock { () -> Int in let v = seq; seq += 1; return v }
        sendEncoded(type: type, object: LifecyclePayload(type: type, sequenceNumber: s, response: resource))
    }

    func emitTerminalEvent(type: String, resource: OpenAIResponseResource) {
        let s = lock.withLock { () -> Int in let v = seq; seq += 1; return v }
        sendEncoded(type: type, object: LifecyclePayload(type: type, sequenceNumber: s, response: resource))
    }

    func handleDelta(_ delta: QWChatStreamParser.Delta, allowReasoning: Bool) {
        switch delta {
        case .reasoning(let text):
            guard allowReasoning, !text.isEmpty else { return }
            let openInfo = lock.withLock { () -> (id: String, idx: Int, newlyOpened: Bool) in
                if let rid = reasoningId, let ridx = reasoningIndex {
                    reasoningText += text
                    return (rid, ridx, false)
                }
                let rid = "rs_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(16)
                let ridx = nextOutputIndex
                nextOutputIndex += 1
                reasoningId = rid
                reasoningIndex = ridx
                reasoningText = text
                return (rid, ridx, true)
            }
            if openInfo.newlyOpened {
                emitJsonEvent("response.output_item.added", [
                    "output_index": openInfo.idx,
                    "item": ["type": "reasoning", "id": openInfo.id, "summary": [Any]()] as [String: Any]
                ])
                emitJsonEvent("response.reasoning_summary_part.added", [
                    "item_id": openInfo.id, "output_index": openInfo.idx, "summary_index": 0,
                    "part": ["type": "summary_text", "text": ""]
                ])
            }
            emitJsonEvent("response.reasoning_summary_text.delta", [
                "item_id": openInfo.id, "output_index": openInfo.idx, "summary_index": 0, "delta": text
            ])
        case .content(let text):
            guard !text.isEmpty else { return }
            closeReasoningIfNeeded()
            let openInfo = lock.withLock { () -> (id: String, idx: Int, newlyOpened: Bool) in
                if let mid = messageId, let midx = messageIndex {
                    messageText += text
                    return (mid, midx, false)
                }
                let mid = "msg_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(16)
                let midx = nextOutputIndex
                nextOutputIndex += 1
                messageId = mid
                messageIndex = midx
                messageText = text
                return (mid, midx, true)
            }
            if openInfo.newlyOpened {
                emitJsonEvent("response.output_item.added", [
                    "output_index": openInfo.idx,
                    "item": [
                        "type": "message", "id": openInfo.id, "status": "in_progress",
                        "role": "assistant", "content": [Any]()
                    ] as [String: Any]
                ])
                emitJsonEvent("response.content_part.added", [
                    "item_id": openInfo.id, "output_index": openInfo.idx, "content_index": 0,
                    "part": ["type": "output_text", "text": "", "annotations": [Any]()] as [String: Any]
                ])
            }
            emitJsonEvent("response.output_text.delta", [
                "item_id": openInfo.id, "output_index": openInfo.idx, "content_index": 0, "delta": text
            ])
        }
    }

    func finishContentItems(status: String) {
        closeReasoningIfNeeded()
        closeMessageIfNeeded(status: status)
    }

    func ensureAtLeastOneItem(status: String) {
        let count = lock.withLock { nextOutputIndex }
        guard count == 0 else { return }
        let (mid, midx) = lock.withLock { () -> (String, Int) in
            let id = "msg_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(16)
            let idx = nextOutputIndex
            nextOutputIndex += 1
            messageId = id
            return (id, idx)
        }
        let part: [String: Any] = ["type": "output_text", "text": "", "annotations": [Any]()]
        emitJsonEvent("response.output_item.added", [
            "output_index": midx,
            "item": ["type": "message", "id": mid, "status": "in_progress", "role": "assistant", "content": [Any]()] as [String: Any]
        ])
        emitJsonEvent("response.content_part.added", [
            "item_id": mid, "output_index": midx, "content_index": 0, "part": part
        ])
        emitJsonEvent("response.output_text.done", [
            "item_id": mid, "output_index": midx, "content_index": 0, "text": ""
        ])
        emitJsonEvent("response.content_part.done", [
            "item_id": mid, "output_index": midx, "content_index": 0, "part": part
        ])
        emitJsonEvent("response.output_item.done", [
            "output_index": midx,
            "item": ["type": "message", "id": mid, "status": status, "role": "assistant", "content": [part]] as [String: Any]
        ])
    }

    func emitFunctionCalls(_ calls: [ToolCall], status: String) {
        for call in calls {
            let fcid = "fc_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(16)
            let idx = lock.withLock { () -> Int in
                let v = nextOutputIndex
                nextOutputIndex += 1
                functionCallIds.append(fcid)
                return v
            }
            emitJsonEvent("response.output_item.added", [
                "output_index": idx,
                "item": [
                    "type": "function_call", "id": fcid, "call_id": call.id,
                    "name": call.function.name, "arguments": "", "status": "in_progress"
                ] as [String: Any]
            ])
            emitJsonEvent("response.function_call_arguments.delta", [
                "item_id": fcid, "output_index": idx, "delta": call.function.arguments
            ])
            emitJsonEvent("response.function_call_arguments.done", [
                "item_id": fcid, "output_index": idx, "arguments": call.function.arguments
            ])
            emitJsonEvent("response.output_item.done", [
                "output_index": idx,
                "item": [
                    "type": "function_call", "id": fcid, "call_id": call.id,
                    "name": call.function.name, "arguments": call.function.arguments, "status": status
                ] as [String: Any]
            ])
        }
    }

    func sendDoneAndClose() {
        connection.send(content: Data("data: [DONE]\n\n".utf8), isComplete: true, completion: .contentProcessed({ _ in
            self.connection.cancel()
        }))
    }

    private func closeReasoningIfNeeded() {
        let info = lock.withLock { () -> (id: String, idx: Int, text: String)? in
            guard let id = reasoningId, let idx = reasoningIndex, !reasoningClosed else { return nil }
            reasoningClosed = true
            return (id, idx, reasoningText)
        }
        guard let info else { return }
        emitJsonEvent("response.reasoning_summary_text.done", [
            "item_id": info.id, "output_index": info.idx, "summary_index": 0, "text": info.text
        ])
        emitJsonEvent("response.reasoning_summary_part.done", [
            "item_id": info.id, "output_index": info.idx, "summary_index": 0,
            "part": ["type": "summary_text", "text": info.text]
        ])
        emitJsonEvent("response.output_item.done", [
            "output_index": info.idx,
            "item": [
                "type": "reasoning", "id": info.id,
                "summary": [["type": "summary_text", "text": info.text]]
            ] as [String: Any]
        ])
    }

    private func closeMessageIfNeeded(status: String) {
        let info = lock.withLock { () -> (id: String, idx: Int, text: String)? in
            guard let id = messageId, let idx = messageIndex, !messageClosed else { return nil }
            messageClosed = true
            return (id, idx, messageText)
        }
        guard let info else { return }
        let part: [String: Any] = ["type": "output_text", "text": info.text, "annotations": [Any]()]
        emitJsonEvent("response.output_text.done", [
            "item_id": info.id, "output_index": info.idx, "content_index": 0, "text": info.text
        ])
        emitJsonEvent("response.content_part.done", [
            "item_id": info.id, "output_index": info.idx, "content_index": 0, "part": part
        ])
        emitJsonEvent("response.output_item.done", [
            "output_index": info.idx,
            "item": [
                "type": "message", "id": info.id, "status": status,
                "role": "assistant", "content": [part]
            ] as [String: Any]
        ])
    }

    private func emitJsonEvent(_ type: String, _ fields: [String: Any]) {
        let s = lock.withLock { () -> Int in let v = seq; seq += 1; return v }
        var body = fields
        body["type"] = type
        body["sequence_number"] = s
        guard let data = try? JSONSerialization.data(withJSONObject: body) else { return }
        var out = Data("event: \(type)\ndata: ".utf8)
        out.append(data)
        out.append(Data("\n\n".utf8))
        connection.send(content: out, isComplete: false, completion: .idempotent)
    }

    private func sendEncoded<T: Encodable>(type: String, object: T) {
        guard let data = try? JSONEncoder().encode(object) else { return }
        var out = Data("event: \(type)\ndata: ".utf8)
        out.append(data)
        out.append(Data("\n\n".utf8))
        connection.send(content: out, isComplete: false, completion: .idempotent)
    }
}
