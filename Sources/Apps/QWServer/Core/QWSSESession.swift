// QWSSESession.swift — Unified SSE and NDJSON streaming session managing
// deferred 200 headers, keepalive timers, passive EOF cancellation, and framing.

import Foundation
import Network
import QWCore

/// Thread-safe gate that stops SSE keepalive comments once the first token arrives.
public final class QWStreamKeepaliveGate: @unchecked Sendable {
    private let lock = NSLock()
    private var stopped = false

    public init() {}

    public func stop() {
        lock.withLock { stopped = true }
    }

    @discardableResult
    public func stopIfRunning() -> Bool {
        lock.withLock {
            if stopped { return false }
            stopped = true
            return true
        }
    }

    public func isStopped() -> Bool {
        lock.withLock { stopped }
    }

    @discardableResult
    public func runIfActive(_ body: () -> Bool) -> Bool {
        lock.withLock {
            guard !stopped else { return false }
            return body()
        }
    }
}

protocol QWSSEFrameTransport: Sendable {
    var isClosed: Bool { get }
    func sendSSEDataFrame<T: Encodable>(_ object: T)
    func sendSSENamedEvent<T: Encodable>(event: String, object: T)
    func sendSSEJsonObject(event: String?, fields: [String: Any])
    func sendNDJSONLine<T: Encodable>(_ object: T)
    func sendRaw(_ data: Data)
    func finish(with finalData: Data?)
}

extension QWSSEWriter: QWSSEFrameTransport {}

/// Cohesive SSE/NDJSON session owning deferred headers, keepalive lifecycle,
/// cancellation checks, backpressured frame emission, and terminal error/done framing.
final class QWSSESession: @unchecked Sendable {
    enum WireFormat: Sendable {
        case sse
        case ndjson

        var contentType: String {
            switch self {
            case .sse: return "text/event-stream"
            case .ndjson: return "application/x-ndjson"
            }
        }
    }

    private static let sseDoneMarker = Data("data: [DONE]\n\n".utf8)
    private static let keepaliveComment = Data(": keepalive\n\n".utf8)
    static let defaultKeepaliveIntervalNs: UInt64 = 5_000_000_000

    private let lock = NSLock()
    private let format: WireFormat
    private let headersPacket: Data
    private let keepaliveIntervalNs: UInt64?
    private let transport: any QWSSEFrameTransport
    private let stopChecker: @Sendable () -> Bool
    private let keepaliveGate = QWStreamKeepaliveGate()
    private var keepaliveTask: Task<Void, Never>?
    private var started = false
    private var finished = false

    init(
        format: WireFormat = .sse,
        extraHeaders: [String] = [],
        corsHeaders: String = "",
        keepaliveIntervalNs: UInt64? = nil,
        transport: any QWSSEFrameTransport,
        stopChecker: @escaping @Sendable () -> Bool
    ) {
        self.format = format
        self.headersPacket = Self.buildResponseHeadersData(
            format: format,
            extraHeaders: extraHeaders,
            corsHeaders: corsHeaders
        )
        self.keepaliveIntervalNs = keepaliveIntervalNs
        self.transport = transport
        self.stopChecker = stopChecker
    }

    var isStarted: Bool {
        lock.withLock { started }
    }

    var isClosed: Bool {
        transport.isClosed || stopChecker()
    }

    var shouldStop: Bool {
        stopChecker()
    }

    static func buildResponseHeadersData(
        format: WireFormat,
        extraHeaders: [String],
        corsHeaders: String
    ) -> Data {
        var raw = "HTTP/1.1 200 OK\r\nContent-Type: \(format.contentType)\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n"
        for line in extraHeaders {
            raw += "\(line)\r\n"
        }
        raw += "\(corsHeaders)\r\n"
        return Data(raw.utf8)
    }

    /// Commits deferred HTTP 200 headers and optionally launches the keepalive timer loop.
    func begin(startKeepalive: Bool? = nil) {
        let shouldStart = lock.withLock { () -> Bool in
            guard !started, !finished else { return false }
            started = true
            return true
        }
        guard shouldStart else { return }
        transport.sendRaw(headersPacket)

        let wantKeepalive = startKeepalive ?? (format == .sse && keepaliveIntervalNs != nil)
        guard wantKeepalive, let interval = keepaliveIntervalNs, interval > 0 else {
            return
        }
        let task = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: interval)
                guard let self, !Task.isCancelled else { break }
                let keepRunning = self.keepaliveGate.runIfActive {
                    guard !self.stopChecker(), !self.transport.isClosed else { return false }
                    self.transport.sendRaw(Self.keepaliveComment)
                    return true
                }
                if !keepRunning { break }
            }
        }
        let shouldCancelImmediately = lock.withLock { () -> Bool in
            if finished || keepaliveGate.isStopped() {
                return true
            }
            keepaliveTask = task
            return false
        }
        if shouldCancelImmediately {
            task.cancel()
        }
    }

    func stopKeepaliveIfRunning() {
        let taskToCancel = lock.withLock { () -> Task<Void, Never>? in
            guard keepaliveGate.stopIfRunning() else { return nil }
            let t = keepaliveTask
            keepaliveTask = nil
            return t
        }
        taskToCancel?.cancel()
    }

    func stopAll() {
        keepaliveGate.stop()
        let taskToCancel = lock.withLock { () -> Task<Void, Never>? in
            let t = keepaliveTask
            keepaliveTask = nil
            return t
        }
        taskToCancel?.cancel()
    }

    func sendSSEDataFrame<T: Encodable>(_ object: T) {
        guard !isClosed else { return }
        transport.sendSSEDataFrame(object)
    }

    func sendSSENamedEvent<T: Encodable>(event: String, object: T) {
        guard !isClosed else { return }
        transport.sendSSENamedEvent(event: event, object: object)
    }

    func sendSSEJsonObject(event: String? = nil, fields: [String: Any]) {
        guard !isClosed else { return }
        transport.sendSSEJsonObject(event: event, fields: fields)
    }

    func sendNDJSONLine<T: Encodable>(_ object: T) {
        guard !isClosed else { return }
        transport.sendNDJSONLine(object)
    }

    func sendRaw(_ data: Data) {
        guard !isClosed else { return }
        transport.sendRaw(data)
    }

    func finish(doneMarker: Bool = false) {
        guard markFinishedAndStopKeepalive() else { return }
        transport.finish(with: doneMarker ? Self.sseDoneMarker : nil)
    }

    func finishWithOpenAIError(_ error: any Error) {
        guard markFinishedAndStopKeepalive() else { return }
        let payload = ServerErrorPayload(message: "\(error)", type: "server_error", code: 500)
        transport.sendSSEDataFrame(payload)
        transport.finish(with: Self.sseDoneMarker)
    }

    func finishWithNamedSSEError(
        event: String = "error",
        fields: [String: Any],
        doneMarker: Bool = false
    ) {
        guard markFinishedAndStopKeepalive() else { return }
        transport.sendSSEJsonObject(event: event, fields: fields)
        transport.finish(with: doneMarker ? Self.sseDoneMarker : nil)
    }

    func finishWithNamedSSEErrorObject<T: Encodable>(
        event: String,
        object: T,
        doneMarker: Bool = false
    ) {
        guard markFinishedAndStopKeepalive() else { return }
        transport.sendSSENamedEvent(event: event, object: object)
        transport.finish(with: doneMarker ? Self.sseDoneMarker : nil)
    }

    func finishWithNDJSONError(_ error: any Error) {
        guard markFinishedAndStopKeepalive() else { return }
        transport.sendNDJSONLine(["error": "\(error)"])
        transport.finish(with: nil)
    }

    private func markFinishedAndStopKeepalive() -> Bool {
        let (shouldFinish, taskToCancel) = lock.withLock { () -> (Bool, Task<Void, Never>?) in
            keepaliveGate.stop()
            let t = keepaliveTask
            keepaliveTask = nil
            guard !finished else { return (false, t) }
            finished = true
            return (true, t)
        }
        taskToCancel?.cancel()
        return shouldFinish
    }
}

extension QWHttpServer {
    func makeSSESession(
        for connection: NWConnection,
        format: QWSSESession.WireFormat = .sse,
        extraHeaders: [String] = [],
        enableKeepalive: Bool? = nil,
        keepaliveIntervalNs: UInt64 = QWSSESession.defaultKeepaliveIntervalNs,
        dateEncodingStrategy: JSONEncoder.DateEncodingStrategy? = nil,
        onError: (@Sendable () -> Void)? = nil
    ) -> QWSSESession {
        let writer = makeSSEWriter(
            for: connection,
            dateEncodingStrategy: dateEncodingStrategy,
            onError: onError
        )
        let shouldKeepalive = enableKeepalive ?? (format == .sse)
        return QWSSESession(
            format: format,
            extraHeaders: extraHeaders,
            corsHeaders: corsHeaderLines(for: connection),
            keepaliveIntervalNs: shouldKeepalive ? keepaliveIntervalNs : nil,
            transport: writer,
            stopChecker: generationShouldStop(for: connection)
        )
    }
}
