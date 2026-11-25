// QWHttpServer+ChatCompletions.swift — OpenAI POST /v1/chat/completions handler
// powered by QWSSESession and QWChatPolicyResolver.

import Foundation
import Network
import QWCore
import QWDecision
import QWEngine
import QWVision

extension QWHttpServer {

    func handleChatCompletions(connection: NWConnection, headers: [String: String] = [:], body: Data) async throws {
        let req: ChatCompletionRequest
        do {
            var parsedReq = try QWServerRequestDecoder.decode(ChatCompletionRequest.self, from: body)
            parsedReq.httpHeaders = headers
            req = parsedReq
        } catch {
            let err = ServerErrorPayload(
                message: "Malformed JSON payload: \(error.localizedDescription)",
                type: "invalid_request_error",
                code: 400
            )
            sendJsonResponse(connection, status: 400, object: err)
            return
        }

        if let n = req.n, n > 1 {
            let err = ServerErrorPayload(
                message: "n > 1 is not supported (got n=\(n))",
                type: "invalid_request_error",
                code: 400
            )
            sendJsonResponse(connection, status: 400, object: err)
            return
        }

        if req.continueFinalMessage == true && req.addGenerationPrompt == true {
            let err = ServerErrorPayload(
                message: "continue_final_message and add_generation_prompt cannot both be true",
                type: "invalid_request_error",
                code: 400
            )
            sendJsonResponse(connection, status: 400, object: err)
            return
        }

        do {
            try req.validateMetadata()
            _ = try QWStructuredOutput.grammar(for: req)
            try actor.validateRequestCapabilities(req)
        } catch {
            let err = ServerErrorPayload(message: "\(error)", type: "invalid_request_error", code: 400)
            sendJsonResponse(connection, status: 400, object: err)
            return
        }

        let rawReq = req
        let effectiveReq = await resolveAutoReasoningEffortIfNeeded(rawReq)

        let returnModelName = resolveReplyModelName(requested: effectiveReq.model, strictNotFound: false) ?? actor.getModelName()
        let reqId = effectiveReq.effectiveRequestId ?? "chatcmpl-\(UUID().uuidString.prefix(12))"
        var resolvedHeaders = ["x-request-id: \(reqId)"]
        if let sid = effectiveReq.effectiveSessionId {
            resolvedHeaders.append("x-session-id: \(sid)")
        }
        let extraHeaders = resolvedHeaders

        let serverConfig = actor.getConfig()
        let streamPolicy = QWChatPolicyResolver.resolveStreamPolicy(
            for: effectiveReq,
            defaultEnableThinking: serverConfig.enableThinking,
            defaultMaxTokens: serverConfig.maxTokens,
            modelMaxLength: serverConfig.modelMaxLength,
            supportsThinkingTags: actor.supportsThinkingTags()
        )
        let shouldStop = generationShouldStop(for: connection)

        if effectiveReq.isStreaming {
            let session = makeSSESession(for: connection, format: .sse, extraHeaders: extraHeaders)
            defer { session.stopAll() }
            @Sendable func sendChunk(_ chunk: ChatCompletionChunk) {
                session.sendSSEDataFrame(chunk)
            }

            let streamParser = QWChatStreamParser(
                startsInReasoning: streamPolicy.shouldThink,
                allowToolCalls: streamPolicy.allowToolCalls,
                allowReasoningTags: streamPolicy.allowReasoningTags,
                allowInk2Markers: actor.isInkling2Template() && (!streamPolicy.isJsonResponseFormat || streamPolicy.shouldThink)
            )

            final class StreamLogprobBox: @unchecked Sendable {
                var pending: [ChatLogprobTokenEntry] = []
                func drain() -> ChoiceLogprobs? {
                    guard !pending.isEmpty else { return nil }
                    let out = ChoiceLogprobs(content: pending)
                    pending.removeAll(keepingCapacity: true)
                    return out
                }
            }
            let lpBox = StreamLogprobBox()

            let result: ChatCompletionExecutionResult
            do {
                result = try await actor.handleChatCompletion(
                    request: effectiveReq,
                    onStreamReady: { [weak self] in
                        guard let self else { return }
                        let readyPolicy = QWChatPolicyResolver.resolveStreamPolicy(
                            for: effectiveReq,
                            defaultEnableThinking: serverConfig.enableThinking,
                            defaultMaxTokens: serverConfig.maxTokens,
                            modelMaxLength: serverConfig.modelMaxLength,
                            supportsThinkingTags: self.actor.supportsThinkingTags()
                        )
                        streamParser.setStartsInReasoning(
                            readyPolicy.shouldThink,
                            allowReasoningTags: readyPolicy.allowReasoningTags,
                            allowInk2Markers: self.actor.isInkling2Template() && (!readyPolicy.isJsonResponseFormat || readyPolicy.shouldThink)
                        )
                        session.begin()
                        sendChunk(ChatCompletionChunk(id: reqId, model: returnModelName, role: "assistant"))
                    },
                    onToken: { tokenPiece in
                        session.stopKeepaliveIfRunning()
                        for delta in streamParser.feed(tokenPiece) {
                            let chunkLP = lpBox.drain()
                            switch delta {
                            case .reasoning(let text):
                                sendChunk(ChatCompletionChunk(
                                    id: reqId, model: returnModelName,
                                    reasoningContent: text, logprobs: chunkLP
                                ))
                            case .content(let text):
                                sendChunk(ChatCompletionChunk(
                                    id: reqId, model: returnModelName,
                                    content: text, logprobs: chunkLP
                                ))
                            }
                        }
                    },
                    onTokenLogprob: (effectiveReq.logprobs == true) ? { lp in
                        lpBox.pending.append(lp)
                    } : nil,
                    shouldStop: shouldStop
                )
            } catch {
                if shouldStop() { return }
                if !session.isStarted {
                    sendChatError(connection, error: error, path: "/v1/chat/completions")
                    return
                }
                logger.log(.error, "HTTP POST /v1/chat/completions (stream) failed: \(error)")
                session.finishWithOpenAIError(error)
                return
            }

            guard !shouldStop() else { return }
            for delta in streamParser.finish() {
                let chunkLP = lpBox.drain()
                switch delta {
                case .reasoning(let text):
                    sendChunk(ChatCompletionChunk(
                        id: reqId, model: returnModelName,
                        reasoningContent: text, logprobs: chunkLP
                    ))
                case .content(let text):
                    sendChunk(ChatCompletionChunk(
                        id: reqId, model: returnModelName,
                        content: text, logprobs: chunkLP
                    ))
                }
            }
            if let remainingLP = lpBox.drain() {
                sendChunk(ChatCompletionChunk(id: reqId, model: returnModelName, logprobs: remainingLP))
            }

            let parsed = streamParser.result(
                tools: streamPolicy.allowToolCalls ? effectiveReq.tools : nil,
                truncated: result.truncated
            )
            let effectiveCalls = QWChatPolicyResolver.capToolCalls(
                parsed.toolCalls,
                parallelToolCalls: effectiveReq.parallelToolCalls
            )
            if let toolCalls = effectiveCalls {
                for (idx, call) in toolCalls.enumerated() {
                    let fn = ToolCallChunk.FunctionChunk(name: call.function.name, arguments: call.function.arguments)
                    let callChunk = ToolCallChunk(index: idx, id: call.id, type: "function", function: fn)
                    sendChunk(ChatCompletionChunk(id: reqId, model: returnModelName, toolCall: callChunk))
                }
            }

            let usage = ChatCompletionResponse.Usage(
                promptTokens: result.promptTokens,
                completionTokens: result.completionTokens,
                totalTokens: result.promptTokens + result.completionTokens,
                cachedTokens: result.cachedTokens,
                reasoningTokens: result.reasoningTokens
            )
            sendChunk(ChatCompletionChunk(id: reqId, model: returnModelName, finishReason: parsed.finishReason, usage: usage))
            session.finish(doneMarker: true)
        } else {
            let result: ChatCompletionExecutionResult
            do {
                result = try await actor.handleChatCompletion(request: effectiveReq, onToken: nil, shouldStop: shouldStop)
            } catch {
                if shouldStop() { return }
                sendChatError(connection, error: error, path: "/v1/chat/completions")
                return
            }
            guard !shouldStop() else { return }
            let parsed = QWToolCallParser.parse(
                text: result.content,
                tools: effectiveReq.tools,
                startsInThinking: result.startsInThinking,
                truncated: result.truncated,
                allowToolCalls: streamPolicy.allowToolCalls,
                allowReasoningTags: !streamPolicy.isJsonResponseFormat || result.startsInThinking,
                allowInk2Markers: actor.isInkling2Template() && (!streamPolicy.isJsonResponseFormat || result.startsInThinking)
            )
            let effectiveCalls = QWChatPolicyResolver.capToolCalls(
                parsed.toolCalls,
                parallelToolCalls: effectiveReq.parallelToolCalls
            )
            let resp = ChatCompletionResponse(
                id: reqId,
                model: returnModelName,
                content: parsed.content,
                toolCalls: effectiveCalls,
                reasoningContent: parsed.reasoningContent,
                logprobs: result.logprobs.map { ChoiceLogprobs(content: $0) },
                promptTokens: result.promptTokens,
                completionTokens: result.completionTokens,
                cachedTokens: result.cachedTokens,
                reasoningTokens: result.reasoningTokens,
                finishReason: parsed.finishReason
            )
            sendJsonResponse(connection, status: 200, object: resp, extraHeaders: extraHeaders)
        }
    }

    private func resolveAutoReasoningEffortIfNeeded(_ req: ChatCompletionRequest) async -> ChatCompletionRequest {
        guard QWChatPolicyResolver.isAutoReasoningEffort(req.rawReasoningEffort),
              actor.isDecisionSidecarLoaded(),
              req.enableThinking != false,
              req.reasoning?.enabled != false,
              req.chatTemplateKwargs?.enableThinking != false,
              let promptText = req.messages.last(where: { $0.role.lowercased() == "user" })?.extractTextContent(),
              !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return req
        }
        let complexityReq = QWDecisionRequest(
            state: .text(promptText),
            questions: [
                QWDecisionQuestion(
                    name: "complexity",
                    kind: .choice,
                    instructions: "How much reasoning effort does this request require before answering?",
                    criteria: [
                        QWDecisionCriterion(key: "trivial", description: "direct fact lookup or trivial rewrite"),
                        QWDecisionCriterion(key: "moderate", description: "single-step explanation or routine coding"),
                        QWDecisionCriterion(key: "deep", description: "multi-step planning, debugging, or complex reasoning"),
                    ]
                ),
            ]
        )
        guard let env = try? await actor.decide(complexityReq),
              let answer = env.response.answers.first,
              let tier = answer.choice,
              let resolved = QWChatPolicyResolver.resolveAutoReasoningEffort(tier: tier, confidence: answer.confidence) else {
            return req
        }
        return req.withResolvedReasoningEffort(resolved)
    }

    func sendChatError(_ connection: NWConnection, error: any Error, path: String) {
        if let ctxErr = error as? QWServerContextLengthError {
            logger.log(.warn, "HTTP POST \(path) context length exceeded: \(ctxErr.description)")
            sendOpenAIError(connection, status: 400, message: ctxErr.description,
                            type: "invalid_request_error", code: "context_length_exceeded")
        } else if let mmErr = error as? QWMultimodalError {
            logger.log(.warn, "HTTP POST \(path) bad multimodal request: \(mmErr)")
            sendOpenAIError(connection, status: 400, message: "\(mmErr)", type: "invalid_request_error")
        } else if error is QWJSONSchemaError || error is QWMetadataValidationError || error is QWChatTemplateError {
            logger.log(.warn, "HTTP POST \(path) invalid request: \(error)")
            sendOpenAIError(connection, status: 400, message: "\(error)", type: "invalid_request_error")
        } else {
            logger.log(.error, "HTTP POST \(path) failed: \(error)")
            sendOpenAIError(connection, status: 500, message: "\(error)", type: "server_error")
        }
    }
}
