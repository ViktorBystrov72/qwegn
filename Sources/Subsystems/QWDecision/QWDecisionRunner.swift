// QWDecisionRunner.swift — Pure-host single-pass orchestrator combining framing, backend, calibration, and decoding.
import Foundation
import QWCore
import QWToken

public final class QWDecisionRunner: Sendable {
    public let tokenizer: QWTokenizer
    public let backend: any QWDecisionExecutionBackend
    public let calibration: QWCalibrationProfile

    public init(
        tokenizer: QWTokenizer,
        backend: any QWDecisionExecutionBackend,
        calibration: QWCalibrationProfile = QWCalibrationProfile()
    ) {
        self.tokenizer = tokenizer
        self.backend = backend
        self.calibration = calibration.mergingInlineTemperatures(backend.decisionConfig.inlineTemperatures)
    }

    public func decide(_ request: QWDecisionRequest) async throws -> QWDecisionResponse {
        guard !request.questions.isEmpty else {
            throw QWDecisionError.invalidQuestion("at least one question is required")
        }
        let built = try QWSequenceFramingEngine.buildBatch(
            request: request,
            config: backend.decisionConfig,
            tokenizer: tokenizer,
            maxContextLength: backend.maxContextLength
        )
        let backendOut = try await backend.evaluate(built.request)
        guard backendOut.slotOutputs.count == built.request.slots.count else {
            throw QWDecisionError.backendFailed(
                "backend emitted \(backendOut.slotOutputs.count) slot outputs for \(built.request.slots.count) requested slots"
            )
        }
        let answers = try QWTypedAnswerDecoder.decode(
            built: built,
            output: backendOut,
            calibration: calibration,
            chain: request.chain
        )
        return QWDecisionResponse(
            answers: answers,
            usage: QWDecisionUsage(
                promptTokens: max(built.promptTokens, backendOut.promptTokens),
                outputTokens: 0,
                truncatedTokens: built.truncatedTokens,
                latencyMs: backendOut.latencyMs
            ),
            chain: request.chain
        )
    }
}
