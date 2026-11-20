// QWHttpServer+OllamaGenerate.swift — Ollama /api/generate endpoint using
// QWSSESession, QWChatPolicyResolver, and direct OllamaGenerateRequest mapping.

import Foundation
import Network

/// The Ollama text-generation answer: the fields Ollama clients read.
/// Durations are in nanoseconds, as Ollama reports them. A field the
/// engine did not measure is left out, never filled in.
struct OllamaGenerateResponse: Codable {
    let model: String
    let createdAt: String
    let response: String
    let done: Bool
    let doneReason: String?
    let promptEvalCount: Int
    let evalCount: Int
    let totalDuration: Int64
    let promptEvalDuration: Int64
    let evalDuration: Int64

    enum CodingKeys: String, CodingKey {
        case model, response, done
        case createdAt = "created_at"
        case doneReason = "done_reason"
        case promptEvalCount = "prompt_eval_count"
        case evalCount = "eval_count"
        case totalDuration = "total_duration"
        case promptEvalDuration = "prompt_eval_duration"
        case evalDuration = "eval_duration"
    }
}

extension QWHttpServer {

    /// POST /api/generate — routes through the shared chat engine path as one user message.
    func handleOllamaGenerate(connection: NWConnection, body: Data) async throws {
        let parsed: OllamaGenerateRequest
        do {
            parsed = try QWServerRequestDecoder.decode(OllamaGenerateRequest.self, from: body)
        } catch {
            sendJsonResponse(connection, status: 400, object: ["error": "invalid request: malformed JSON body"])
            return
        }
        guard let prompt = parsed.prompt else {
            sendJsonResponse(connection, status: 400, object: ["error": "prompt: required field is missing"])
            return
        }

        let requested = (parsed.model ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard let replyModel = resolveReplyModelName(requested: requested, strictNotFound: true) else {
            sendJsonResponse(connection, status: 404, object: ["error": "model \"\(requested)\" not found, try pulling it first"])
            return
        }

        let req = parsed.toChatCompletionRequest(promptText: prompt)
        do {
            try req.validateMetadata()
            _ = try QWStructuredOutput.grammar(for: req)
            try actor.validateRequestCapabilities(req)
        } catch {
            let status = Self.ollamaErrorStatus(for: error)
            sendJsonResponse(connection, status: status == 500 ? 400 : status,
                             object: ["error": "invalid request: \(error)"])
            return
        }

        let shouldStop = generationShouldStop(for: connection)
        let startedAt = Date()
        if req.isStreaming {
            let session = makeSSESession(for: connection, format: .ndjson, enableKeepalive: false)
            defer { session.stopAll() }
            let cfg = actor.getConfig()
            let policy = QWChatPolicyResolver.resolveStreamPolicy(
                for: req,
                defaultEnableThinking: cfg.enableThinking,
                defaultMaxTokens: cfg.maxTokens,
                modelMaxLength: cfg.modelMaxLength,
                supportsThinkingTags: actor.supportsThinkingTags()
            )
            let streamParser = QWChatStreamParser(
                startsInReasoning: policy.shouldThink,
                allowToolCalls: false,
                allowReasoningTags: policy.allowReasoningTags
            )
            let emitChunk: @Sendable (String) -> Void = { text in
                guard !text.isEmpty else { return }
                session.sendNDJSONLine(OllamaGenerateResponse(
                    model: replyModel, createdAt: OllamaChatResponse.now(), response: text,
                    done: false, doneReason: nil,
                    promptEvalCount: 0, evalCount: 0, totalDuration: 0, promptEvalDuration: 0, evalDuration: 0
                ))
            }
            let result: ChatCompletionExecutionResult
            do {
                result = try await actor.handleChatCompletion(
                    request: req,
                    onStreamReady: { [weak self] in
                        guard let self else { return }
                        let readyPolicy = QWChatPolicyResolver.resolveStreamPolicy(
                            for: req,
                            defaultEnableThinking: cfg.enableThinking,
                            defaultMaxTokens: cfg.maxTokens,
                            modelMaxLength: cfg.modelMaxLength,
                            supportsThinkingTags: self.actor.supportsThinkingTags()
                        )
                        streamParser.setStartsInReasoning(
                            readyPolicy.shouldThink,
                            allowReasoningTags: readyPolicy.allowReasoningTags
                        )
                        session.begin(startKeepalive: false)
                    },
                    onToken: { piece in
                        for delta in streamParser.feed(piece) {
                            if case .content(let t) = delta { emitChunk(t) }
                        }
                    },
                    shouldStop: shouldStop
                )
            } catch {
                if shouldStop() { return }
                if !session.isStarted {
                    sendJsonResponse(connection, status: Self.ollamaErrorStatus(for: error), object: ["error": "\(error)"])
                    return
                }
                session.finishWithNDJSONError(error)
                return
            }
            guard !shouldStop() else { return }
            for delta in streamParser.finish() {
                if case .content(let t) = delta { emitChunk(t) }
            }
            let elapsedNs = Int64(Date().timeIntervalSince(startedAt) * 1e9)
            session.sendNDJSONLine(OllamaGenerateResponse(
                model: replyModel,
                createdAt: OllamaChatResponse.now(),
                response: "",
                done: true,
                doneReason: result.truncated ? "length" : "stop",
                promptEvalCount: result.promptTokens,
                evalCount: result.completionTokens,
                totalDuration: elapsedNs,
                promptEvalDuration: Int64((result.prefillSec ?? 0) * 1e9),
                evalDuration: Int64((result.decodeSec ?? 0) * 1e9)
            ))
            session.finish(doneMarker: false)
            return
        }

        let result: ChatCompletionExecutionResult
        do {
            result = try await actor.handleChatCompletion(request: req, onToken: nil, shouldStop: shouldStop)
        } catch {
            if shouldStop() { return }
            sendJsonResponse(connection, status: Self.ollamaErrorStatus(for: error), object: ["error": "\(error)"])
            return
        }
        guard !shouldStop() else { return }
        let elapsedNs = Int64(Date().timeIntervalSince(startedAt) * 1e9)
        let parsedOut = QWToolCallParser.parse(
            text: result.content, tools: nil, startsInThinking: result.startsInThinking,
            truncated: result.truncated, allowToolCalls: false
        )
        let resp = OllamaGenerateResponse(
            model: replyModel,
            createdAt: OllamaChatResponse.now(),
            response: parsedOut.content ?? "",
            done: true,
            doneReason: result.truncated ? "length" : "stop",
            promptEvalCount: result.promptTokens,
            evalCount: result.completionTokens,
            totalDuration: elapsedNs,
            promptEvalDuration: Int64((result.prefillSec ?? 0) * 1e9),
            evalDuration: Int64((result.decodeSec ?? 0) * 1e9)
        )
        sendJsonResponse(connection, status: 200, object: resp)
    }
}
