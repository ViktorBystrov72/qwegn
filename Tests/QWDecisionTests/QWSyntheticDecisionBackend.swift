// QWSyntheticDecisionBackend.swift — Test-only deterministic QWDecisionExecutionBackend fixture (Law 18: Tests/ only).
import Foundation
import QWCore
@testable import QWDecision

final class QWSyntheticDecisionBackend: QWDecisionExecutionBackend, @unchecked Sendable {
    let decisionConfig: QWDecisionModelConfig
    let maxContextLength: Int
    var lastRequest: QWDecisionBackendBatchRequest?
    var handler: @Sendable (QWDecisionBackendBatchRequest) throws -> QWDecisionBackendOutput

    init(
        decisionConfig: QWDecisionModelConfig,
        maxContextLength: Int = 4096,
        handler: @escaping @Sendable (QWDecisionBackendBatchRequest) throws -> QWDecisionBackendOutput
    ) {
        self.decisionConfig = decisionConfig
        self.maxContextLength = maxContextLength
        self.handler = handler
    }

    func evaluate(_ request: QWDecisionBackendBatchRequest) async throws -> QWDecisionBackendOutput {
        lastRequest = request
        return try handler(request)
    }
}
