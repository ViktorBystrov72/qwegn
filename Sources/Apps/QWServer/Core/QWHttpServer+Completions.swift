// QWHttpServer+Completions.swift — Legacy OpenAI POST /v1/completions handler.

import Foundation
import Network
import QWCore
import QWEngine

extension QWHttpServer {

    func handleCompletions(connection: NWConnection, headers: [String: String] = [:], body: Data) async throws {
        var req: CompletionRequest
        do {
            req = try JSONDecoder().decode(CompletionRequest.self, from: body)
            req.httpHeaders = headers
        } catch {
            sendOpenAIError(connection, status: 400, message: "Malformed JSON payload: \(error.localizedDescription)")
            return
        }

        if let n = req.n, n > 1 {
            sendOpenAIError(connection, status: 400, message: "n > 1 is not supported (got n=\(n))")
            return
        }
        if let maxT = req.maxTokens, maxT < 0 {
            sendOpenAIError(connection, status: 400, message: "max_tokens must be >= 0")
            return
        }
        if let lp = req.logprobs, lp < 0 {
            sendOpenAIError(connection, status: 400, message: "logprobs must be >= 0")
            return
        }

        let replyModel = resolveReplyModelName(requested: req.model, strictNotFound: false) ?? actor.getModelName()
        let reqId = "cmpl-\(UUID().uuidString.prefix(12))"
        let shouldStop = generationShouldStop(for: connection)
        let created = Int(Date().timeIntervalSince1970)
        let extraHeaders = ["x-request-id: \(reqId)"]

        if req.stream ?? false {
            final class CompletionSSEBox: @unchecked Sendable {
                private let lock = NSLock()
                private var started = false
                private var gate: QWStreamKeepaliveGate?
                private var task: Task<Void, Never>?
                var isStarted: Bool { lock.withLock { started } }
                func markStarted(gate: QWStreamKeepaliveGate, task: Task<Void, Never>) {
                    lock.withLock { started = true; self.gate = gate; self.task = task }
                }
                func stopKeepaliveIfRunning() {
                    let t = lock.withLock { () -> Task<Void, Never>? in
                        guard let g = gate, g.stopIfRunning() else { return nil }
                        return task
                    }
                    t?.cancel()
                }
                func stopAll() {
                    let (g, t) = lock.withLock { (gate, task) }
                    g?.stop()
                    t?.cancel()
                }
            }

            let box = CompletionSSEBox()
            defer { box.stopAll() }
            let ssePrefix = Data("data: ".utf8)
            let sseSuffix = Data("\n\n".utf8)
            @Sendable func sendChunk(_ chunk: CompletionResponse) {
                guard let data = try? JSONEncoder().encode(chunk) else { return }
                var frame = Data(capacity: ssePrefix.count + data.count + sseSuffix.count)
                frame.append(ssePrefix)
                frame.append(data)
                frame.append(sseSuffix)
                connection.send(content: frame, isComplete: false, completion: .idempotent)
            }

            let result: CompletionExecutionResult
            do {
                result = try await actor.handleTextCompletion(
                    request: req,
                    onStreamReady: { [weak self] in
                        guard let self else { return }
                        self.sendSSEHeaders(connection, extraHeaders: extraHeaders)
                        let (gate, task) = self.startSSEKeepalive(connection: connection)
                        box.markStarted(gate: gate, task: task)
                    },
                    onEchoPrompt: { idx, text, logprobs in
                        box.stopKeepaliveIfRunning()
                        let choice = CompletionChoice(text: text, index: idx, logprobs: logprobs, finishReason: nil)
                        sendChunk(CompletionResponse(id: reqId, model: replyModel, choices: [choice], created: created))
                    },
                    onChoiceToken: { idx, piece, logprobs in
                        box.stopKeepaliveIfRunning()
                        let choice = CompletionChoice(text: piece, index: idx, logprobs: logprobs, finishReason: nil)
                        sendChunk(CompletionResponse(id: reqId, model: replyModel, choices: [choice], created: created))
                    },
                    shouldStop: shouldStop
                )
            } catch {
                if shouldStop() { return }
                if !box.isStarted {
                    sendCompletionError(connection, error: error)
                    return
                }
                let errPayload = ServerErrorPayload(message: "\(error)", type: "server_error", code: 500)
                if let errData = try? JSONEncoder().encode(errPayload) {
                    var frame = Data(capacity: ssePrefix.count + errData.count + sseSuffix.count)
                    frame.append(ssePrefix)
                    frame.append(errData)
                    frame.append(sseSuffix)
                    connection.send(content: frame, isComplete: false, completion: .idempotent)
                }
                connection.send(content: Data("data: [DONE]\n\n".utf8), isComplete: true, completion: .contentProcessed({ _ in
                    connection.cancel()
                }))
                return
            }

            guard !shouldStop() else { return }
            let finalChoices = result.choices.map {
                CompletionChoice(text: "", index: $0.index, logprobs: nil, finishReason: $0.finishReason)
            }
            let includeUsage = req.streamOptions?.includeUsage ?? true
            let usage = includeUsage
                ? CompletionUsage(promptTokens: result.totalPromptTokens, completionTokens: result.totalCompletionTokens)
                : nil
            sendChunk(CompletionResponse(
                id: reqId, model: replyModel,
                choices: finalChoices.isEmpty ? [CompletionChoice(text: "", index: 0, finishReason: "stop")] : finalChoices,
                usage: usage, created: created
            ))
            connection.send(content: Data("data: [DONE]\n\n".utf8), isComplete: true, completion: .contentProcessed({ _ in
                connection.cancel()
            }))
        } else {
            let result: CompletionExecutionResult
            do {
                result = try await actor.handleTextCompletion(request: req, shouldStop: shouldStop)
            } catch {
                if shouldStop() { return }
                sendCompletionError(connection, error: error)
                return
            }
            guard !shouldStop() else { return }
            let choices = result.choices.map {
                CompletionChoice(text: $0.text, index: $0.index, logprobs: $0.logprobs, finishReason: $0.finishReason)
            }
            let resp = CompletionResponse(
                id: reqId,
                model: replyModel,
                choices: choices,
                usage: CompletionUsage(promptTokens: result.totalPromptTokens, completionTokens: result.totalCompletionTokens),
                created: created
            )
            sendJsonResponse(connection, status: 200, object: resp, extraHeaders: extraHeaders)
        }
    }

    func handleTokenizerInfo(connection: NWConnection, model: String?) async throws {
        do {
            let info = try await actor.handleTokenizerInfo(model: model)
            sendJsonResponse(connection, status: 200, object: info)
        } catch {
            sendCompletionError(connection, error: error)
        }
    }

    func handleTokenize(connection: NWConnection, body: Data) async throws {
        let req: TokenizeRequest
        do {
            req = try JSONDecoder().decode(TokenizeRequest.self, from: body)
        } catch {
            sendOpenAIError(connection, status: 400, message: "Malformed JSON payload: \(error.localizedDescription)")
            return
        }
        do {
            let resp = try await actor.handleTokenize(request: req)
            sendJsonResponse(connection, status: 200, object: resp)
        } catch {
            sendCompletionError(connection, error: error)
        }
    }

    func handleDetokenize(connection: NWConnection, body: Data) async throws {
        let req: DetokenizeRequest
        do {
            req = try JSONDecoder().decode(DetokenizeRequest.self, from: body)
        } catch {
            sendOpenAIError(connection, status: 400, message: "Malformed JSON payload: \(error.localizedDescription)")
            return
        }
        do {
            let resp = try await actor.handleDetokenize(request: req)
            sendJsonResponse(connection, status: 200, object: resp)
        } catch {
            sendCompletionError(connection, error: error)
        }
    }

    private func sendCompletionError(_ connection: NWConnection, error: any Error) {
        if let ctxErr = error as? QWServerContextLengthError {
            logger.log(.warn, "HTTP POST /v1/completions context length exceeded: \(ctxErr.description)")
            sendOpenAIError(connection, status: 400, message: ctxErr.description,
                            type: "invalid_request_error", code: "context_length_exceeded")
        } else if let inputErr = error as? CompletionInvalidInputError {
            logger.log(.warn, "HTTP POST /v1/completions invalid request: \(inputErr.message)")
            sendOpenAIError(connection, status: 400, message: inputErr.message, type: "invalid_request_error")
        } else {
            logger.log(.error, "HTTP POST /v1/completions failed: \(error)")
            sendOpenAIError(connection, status: 500, message: "\(error)", type: "server_error")
        }
    }
}
