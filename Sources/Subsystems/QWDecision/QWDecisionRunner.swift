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

    public func decideBatch(_ requests: [QWDecisionRequest]) async throws -> [QWDecisionResponse] {
        guard !requests.isEmpty else { return [] }
        if requests.count == 1 {
            return [try await decide(requests[0])]
        }
        var builtList: [QWBuiltDecisionBatch] = []
        builtList.reserveCapacity(requests.count)
        var allSequences: [QWDecisionSequenceInput] = []
        var allSlots: [QWDecisionSlotTarget] = []
        var candSet = Set<Int>()
        var slotCounts: [Int] = []
        slotCounts.reserveCapacity(requests.count)

        for req in requests {
            guard !req.questions.isEmpty else {
                throw QWDecisionError.invalidQuestion("at least one question is required")
            }
            let built = try QWSequenceFramingEngine.buildBatch(
                request: req,
                config: backend.decisionConfig,
                tokenizer: tokenizer,
                maxContextLength: backend.maxContextLength
            )
            let seqOffset = allSequences.count
            allSequences.append(contentsOf: built.request.sequences)
            for slot in built.request.slots {
                allSlots.append(QWDecisionSlotTarget(
                    sequenceIndex: slot.sequenceIndex + seqOffset,
                    tokenPosition: slot.tokenPosition,
                    candidateTokenGroups: slot.candidateTokenGroups,
                    isPointerSlot: slot.isPointerSlot,
                    pointerEndPosition: slot.pointerEndPosition
                ))
            }
            candSet.formUnion(built.request.uniqueCandidateTokenIds)
            slotCounts.append(built.request.slots.count)
            builtList.append(built)
        }

        let combinedReq = QWDecisionBackendBatchRequest(
            sequences: allSequences,
            slots: allSlots,
            uniqueCandidateTokenIds: candSet.sorted()
        )
        let combinedOut = try await backend.evaluate(combinedReq)
        guard combinedOut.slotOutputs.count == allSlots.count else {
            throw QWDecisionError.backendFailed(
                "backend emitted \(combinedOut.slotOutputs.count) slot outputs for \(allSlots.count) requested slots"
            )
        }

        var responses: [QWDecisionResponse] = []
        responses.reserveCapacity(requests.count)
        var slotCursor = 0
        for (idx, built) in builtList.enumerated() {
            let count = slotCounts[idx]
            let slice = Array(combinedOut.slotOutputs[slotCursor..<(slotCursor + count)])
            slotCursor += count
            let subOut = QWDecisionBackendOutput(
                slotOutputs: slice,
                promptTokens: built.promptTokens,
                latencyMs: combinedOut.latencyMs
            )
            let answers = try QWTypedAnswerDecoder.decode(
                built: built,
                output: subOut,
                calibration: calibration,
                chain: requests[idx].chain
            )
            responses.append(QWDecisionResponse(
                answers: answers,
                usage: QWDecisionUsage(
                    promptTokens: built.promptTokens,
                    outputTokens: 0,
                    truncatedTokens: built.truncatedTokens,
                    latencyMs: combinedOut.latencyMs
                ),
                chain: requests[idx].chain
            ))
        }
        return responses
    }
}
