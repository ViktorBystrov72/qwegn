// QWHttpServer+Decisions.swift — HTTP handlers for /v1/decisions, /v1/systemone, and /v1/laya.

import Foundation
import Network
import QWCore
import QWDecision
import QWEngine

extension QWHttpServer {
    func handleDecisionRoute(connection: NWConnection, body: Data) async {
        do {
            let httpReq = try QWServerRequestDecoder.decode(QWDecisionHTTPRequest.self, from: body)
            let domainReq = try httpReq.toDecisionRequest()
            let envelope = try await actor.decide(domainReq, preferredModel: httpReq.model)
            sendJsonResponse(connection, status: 200, object: QWDecisionHTTPResponse(envelope: envelope))
        } catch {
            sendDecisionError(connection, error: error)
        }
    }

    func handleDecisionBatchRoute(connection: NWConnection, body: Data) async {
        do {
            let batchReq = try QWServerRequestDecoder.decode(QWDecisionBatchHTTPRequest.self, from: body)
            let items = batchReq.resolvedItems
            guard !items.isEmpty else {
                throw QWDecisionError.invalidQuestion("batch request requires non-empty 'items'")
            }
            var results: [QWDecisionHTTPResponse] = []
            results.reserveCapacity(items.count)
            for item in items {
                let domainReq = try item.toDecisionRequest()
                let envelope = try await actor.decide(domainReq, preferredModel: item.model)
                results.append(QWDecisionHTTPResponse(envelope: envelope))
            }
            sendJsonResponse(connection, status: 200, object: QWDecisionBatchHTTPResponse(results: results))
        } catch {
            sendDecisionError(connection, error: error)
        }
    }

    func handleLayaRoute(connection: NWConnection, body: Data) async {
        do {
            let layaReq = try QWServerRequestDecoder.decode(QWLayaHTTPRequest.self, from: body)
            let domainReq = try layaReq.toDecisionRequest()
            let envelope = try await actor.decide(domainReq, preferredModel: layaReq.model)
            sendJsonResponse(connection, status: 200, object: QWLayaHTTPResponse(envelope: envelope))
        } catch {
            sendDecisionError(connection, error: error)
        }
    }

    func handleDecisionLoadRoute(connection: NWConnection, body: Data) async {
        do {
            let req = try QWServerRequestDecoder.decode(QWDecisionLoadHTTPRequest.self, from: body)
            let rawRef = req.packPath ?? req.model ?? ""
            guard !rawRef.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw QWDecisionError.invalidQuestion("decision model load requires 'pack_path' or 'model'")
            }
            let allowUnregistered = ProcessInfo.processInfo.environment["QWG_ALLOW_UNREGISTERED_MODEL_PATHS"] == "1"
            let settings = QWSettingsManager.shared.loadSettings()
            let ref = settings.resolveModelReference(rawRef, allowArbitraryFilesystemPaths: true)
            let resolvedPath = ref.packPath.isEmpty ? rawRef.expandingTildePath : ref.packPath
            if !allowUnregistered {
                let candURL = URL(fileURLWithPath: resolvedPath)
                let inRegistered = QWHttpSecurity.canonicalDirectoryContainment(candURL, inside: QWPaths.modelsDirectoryURL)
                    || QWHttpSecurity.canonicalDirectoryContainment(candURL, inside: QWPaths.hfHubCacheDirectoryURL)
                guard inRegistered else {
                    throw QWMetadataValidationError("Decision model pack must reside under ~/.qwg/models or HuggingFace cache")
                }
            }
            let unit = QWDecisionComputeUnit(rawValue: (req.computeUnit ?? settings.decisionComputeUnit).lowercased()) ?? .auto
            let loadVision = req.loadVision ?? settings.decisionLoadVision
            let status = try await actor.loadDecisionSidecar(
                packPath: resolvedPath,
                computeUnit: unit,
                loadVision: loadVision
            )
            sendJsonResponse(connection, status: 200, object: status)
        } catch {
            sendDecisionError(connection, error: error)
        }
    }

    func handleDecisionUnloadRoute(connection: NWConnection) async {
        let status = await actor.unloadDecisionSidecar()
        sendJsonResponse(connection, status: 200, object: status)
    }

    func handleDecisionStatusRoute(connection: NWConnection) {
        let status = actor.decisionSidecarStatus()
        sendJsonResponse(connection, status: 200, object: status)
    }

    func sendDecisionError(_ connection: NWConnection, error: any Error) {
        if let decErr = error as? QWDecisionError {
            switch decErr {
            case .unsupportedLabelSubset(_, let allowed):
                let payload = QWDecisionErrorPayload(
                    message: decErr.description,
                    type: "invalid_request_error",
                    code: 400,
                    allowedLabels: allowed
                )
                sendJsonResponse(connection, status: 400, object: payload)
            case .invalidQuestion, .contextOverflow, .missingSingleTokenAlias, .calibrationFailed:
                let payload = QWDecisionErrorPayload(
                    message: decErr.description,
                    type: "invalid_request_error",
                    code: 400
                )
                sendJsonResponse(connection, status: 400, object: payload)
            case .backendFailed(let msg):
                let isUnavailable = msg.localizedCaseInsensitiveContains("no decision sidecar")
                let status = isUnavailable ? 503 : 500
                let payload = QWDecisionErrorPayload(
                    message: decErr.description,
                    type: isUnavailable ? "service_unavailable" : "server_error",
                    code: status
                )
                sendJsonResponse(connection, status: status, object: payload)
            }
            return
        }
        if let metaErr = error as? QWMetadataValidationError {
            let payload = QWDecisionErrorPayload(message: metaErr.description, type: "invalid_request_error", code: 403)
            sendJsonResponse(connection, status: 403, object: payload)
            return
        }
        let payload = QWDecisionErrorPayload(
            message: "Malformed decision payload: \(error.localizedDescription)",
            type: "invalid_request_error",
            code: 400
        )
        sendJsonResponse(connection, status: 400, object: payload)
    }
}
