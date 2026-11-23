// QWHttpServer+Updates.swift — HTTP endpoints proxying local update checks and verified installs.

import Foundation
import Network
import QWCore
import QWEngine

private struct QWUpdateCheckRequest: Decodable {
    let force: Bool?
}

private struct QWUpdateSkipRequest: Decodable {
    let version: String?
}

private struct QWUpdateInstallRequest: Decodable {
    let force: Bool?
}

extension QWHttpServer {
    var loopbackDesktopActionsAllowed: Bool {
        QWHttpSecurity.isLoopbackHost(host)
    }

    func isInferenceCurrentlyActive() -> Bool {
        let snap = QWEngineTelemetryRegistry.shared.snapshot()
        let gen = snap.throughput.phase.lowercased()
        if gen == "prefill" || gen == "decode" {
            return true
        }
        let phase = actor.engineState().lifecycle.phase
        return phase == .loading || phase == .unloading
    }

    func handleUpdateStatusRoute(connection: NWConnection) async {
        let snapshot = await updateService.currentSnapshot()
        let resp = QWUpdateServerResponse(
            snapshot: snapshot,
            inferenceActive: isInferenceCurrentlyActive(),
            loopbackDesktopActionsAllowed: loopbackDesktopActionsAllowed
        )
        sendIsoJsonResponse(connection, status: 200, object: resp)
    }

    func handleUpdateCheckRoute(connection: NWConnection, body: Data) async {
        let req = body.isEmpty ? nil : try? JSONDecoder().decode(QWUpdateCheckRequest.self, from: body)
        let snapshot = await updateService.checkNow(force: req?.force ?? true)
        let resp = QWUpdateServerResponse(
            snapshot: snapshot,
            inferenceActive: isInferenceCurrentlyActive(),
            loopbackDesktopActionsAllowed: loopbackDesktopActionsAllowed
        )
        sendIsoJsonResponse(connection, status: 200, object: resp)
    }

    func handleUpdateSkipRoute(connection: NWConnection, body: Data) async {
        guard !body.isEmpty,
              let req = try? JSONDecoder().decode(QWUpdateSkipRequest.self, from: body) else {
            let err = ServerErrorPayload(message: "Invalid JSON payload for update skip.", code: 400)
            sendJsonResponse(connection, status: 400, object: err)
            return
        }
        do {
            let snapshot = try await updateService.setSkippedVersion(req.version)
            let resp = QWUpdateServerResponse(
                snapshot: snapshot,
                inferenceActive: isInferenceCurrentlyActive(),
                loopbackDesktopActionsAllowed: loopbackDesktopActionsAllowed
            )
            sendIsoJsonResponse(connection, status: 200, object: resp)
        } catch {
            let err = ServerErrorPayload(message: error.localizedDescription, code: 500)
            sendJsonResponse(connection, status: 500, object: err)
        }
    }

    func handleUpdateDownloadRoute(connection: NWConnection) async {
        guard loopbackDesktopActionsAllowed else {
            let err = ServerErrorPayload(
                message: "Desktop installer downloads are restricted to loopback servers.",
                code: 403
            )
            sendJsonResponse(connection, status: 403, object: err)
            return
        }
        do {
            let snapshot = try await updateService.startDownload()
            let resp = QWUpdateServerResponse(
                snapshot: snapshot,
                inferenceActive: isInferenceCurrentlyActive(),
                loopbackDesktopActionsAllowed: loopbackDesktopActionsAllowed
            )
            sendIsoJsonResponse(connection, status: 200, object: resp)
        } catch let updateErr as QWUpdateError {
            let err = ServerErrorPayload(message: updateErr.localizedDescription, code: 400)
            sendJsonResponse(connection, status: 400, object: err)
        } catch {
            let err = ServerErrorPayload(message: error.localizedDescription, code: 500)
            sendJsonResponse(connection, status: 500, object: err)
        }
    }

    func handleUpdateCancelDownloadRoute(connection: NWConnection) async {
        guard loopbackDesktopActionsAllowed else {
            let err = ServerErrorPayload(
                message: "Desktop update cancellation is restricted to loopback servers.",
                code: 403
            )
            sendJsonResponse(connection, status: 403, object: err)
            return
        }
        await updateService.cancelDownload()
        let snapshot = await updateService.currentSnapshot()
        let resp = QWUpdateServerResponse(
            snapshot: snapshot,
            inferenceActive: isInferenceCurrentlyActive(),
            loopbackDesktopActionsAllowed: loopbackDesktopActionsAllowed
        )
        sendIsoJsonResponse(connection, status: 200, object: resp)
    }

    func handleUpdateInstallRoute(connection: NWConnection, body: Data) async {
        guard loopbackDesktopActionsAllowed else {
            let err = ServerErrorPayload(
                message: "Launching macOS Installer is restricted to loopback servers.",
                code: 403
            )
            sendJsonResponse(connection, status: 403, object: err)
            return
        }
        let req = body.isEmpty ? nil : try? JSONDecoder().decode(QWUpdateInstallRequest.self, from: body)
        do {
            let snapshot = try await updateService.installVerifiedPackage(
                inferenceActive: isInferenceCurrentlyActive(),
                force: req?.force ?? false
            )
            let resp = QWUpdateServerResponse(
                snapshot: snapshot,
                inferenceActive: isInferenceCurrentlyActive(),
                loopbackDesktopActionsAllowed: loopbackDesktopActionsAllowed
            )
            sendIsoJsonResponse(connection, status: 200, object: resp)
        } catch let activeErr as QWServerUpdateInstallError {
            let err = ServerErrorPayload(message: activeErr.localizedDescription, code: 409)
            sendJsonResponse(connection, status: 409, object: err)
        } catch let updateErr as QWUpdateError {
            let err = ServerErrorPayload(message: updateErr.localizedDescription, code: 400)
            sendJsonResponse(connection, status: 400, object: err)
        } catch {
            let err = ServerErrorPayload(message: error.localizedDescription, code: 500)
            sendJsonResponse(connection, status: 500, object: err)
        }
    }
}
