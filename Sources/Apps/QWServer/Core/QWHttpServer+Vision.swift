// QWHttpServer+Vision.swift — Loopback-only privacy-gated ScreenCaptureKit snapshot endpoint.

import Foundation
import Network

extension QWHttpServer {
    private struct ScreenCaptureRequestBody: Decodable {
        let prompt: Bool?
        let maxDimension: Int?

        enum CodingKeys: String, CodingKey {
            case prompt
            case maxDimension = "max_dimension"
        }
    }

    private struct ScreenCaptureResponseBody: Encodable {
        let status: String
        let dataUrl: String
        let width: Int
        let height: Int
        let displayId: UInt32

        enum CodingKeys: String, CodingKey {
            case status
            case dataUrl = "data_url"
            case width
            case height
            case displayId = "display_id"
        }
    }

    static func isLoopbackClientEndpoint(_ endpoint: NWEndpoint) -> Bool {
        switch endpoint {
        case .unix:
            return true
        case .hostPort(let host, _):
            switch host {
            case .ipv4(let addr):
                return addr.rawValue.first == 127
            case .ipv6(let addr):
                let b = Array(addr.rawValue)
                let isV6Loopback = b.count == 16 && b.dropLast().allSatisfy { $0 == 0 } && b.last == 1
                let isMappedV4Loopback = b.count == 16
                    && b[0..<10].allSatisfy { $0 == 0 }
                    && b[10] == 0xFF && b[11] == 0xFF && b[12] == 127
                return isV6Loopback || isMappedV4Loopback
            case .name(let name, _):
                return name.lowercased() == "localhost"
            @unknown default:
                return false
            }
        default:
            return false
        }
    }

    func handleScreenCaptureRoute(connection: NWConnection, body: Data) async {
        guard Self.isLoopbackClientEndpoint(connection.endpoint) else {
            sendOpenAIError(
                connection,
                status: 403,
                message: "Screen capture is restricted to local loopback clients (127.0.0.1 / ::1).",
                type: "permission_error",
                code: "loopback_required"
            )
            return
        }
        let parsed = body.isEmpty ? nil : (try? JSONDecoder().decode(ScreenCaptureRequestBody.self, from: body))
        let prompt = parsed?.prompt ?? false
        let maxDim = min(max(parsed?.maxDimension ?? 1920, 256), 3840)
        do {
            let snap = try await QWScreenCaptureHelper.captureActiveDisplay(
                promptForPermissionIfNeeded: prompt,
                maxDimension: maxDim
            )
            sendJsonResponse(
                connection,
                status: 200,
                object: ScreenCaptureResponseBody(
                    status: "ok",
                    dataUrl: snap.dataURL,
                    width: snap.width,
                    height: snap.height,
                    displayId: snap.displayID
                )
            )
        } catch let capErr as QWScreenCaptureError {
            let status: Int = if case .permissionDenied = capErr { 403 } else { 500 }
            sendOpenAIError(
                connection,
                status: status,
                message: capErr.description,
                type: status == 403 ? "permission_error" : "server_error",
                code: status == 403 ? "screen_capture_denied" : "screen_capture_failed"
            )
        } catch {
            sendOpenAIError(
                connection,
                status: 500,
                message: error.localizedDescription,
                type: "server_error",
                code: "screen_capture_failed"
            )
        }
    }
}
