// QWImageDecodePolicy.swift — Image source policy, SSRF IP/DNS guard, and pinned remote fetch.

import Darwin
import Foundation
import Network

public struct QWImageDecodePolicy: Sendable, Equatable {
    public static let defaultMaxPixelArea = 16_777_216
    public static let defaultMaxImagesPerTurn = 16
    public static let serverDefault = QWImageDecodePolicy()
    public static let cliLocal = QWImageDecodePolicy(allowLocalImagePaths: true)

    public let allowDataURLs: Bool
    public let allowLocalImagePaths: Bool
    public let allowRemoteImageURLs: Bool
    public let maxSourceBytes: Int
    public let maxPixelDimension: Int
    public let maxPixelArea: Int

    public init(
        allowDataURLs: Bool = true,
        allowLocalImagePaths: Bool = false,
        allowRemoteImageURLs: Bool = false,
        maxSourceBytes: Int = 20 * 1024 * 1024,
        maxPixelDimension: Int = 8192,
        maxPixelArea: Int = QWImageDecodePolicy.defaultMaxPixelArea
    ) {
        self.allowDataURLs = allowDataURLs
        self.allowLocalImagePaths = allowLocalImagePaths
        self.allowRemoteImageURLs = allowRemoteImageURLs
        self.maxSourceBytes = maxSourceBytes
        self.maxPixelDimension = maxPixelDimension
        self.maxPixelArea = maxPixelArea
    }

    struct ResolvedRemoteTarget: Sendable, Equatable {
        let hostHeader: String
        let sniHost: String
        let port: UInt16
        let useTLS: Bool
        let requestTarget: String
        let resolvedIPs: [String]
    }

    public static func validateRemoteURL(_ url: URL) throws {
        _ = try validateAndResolveRemoteURL(url)
    }

    static func validateAndResolveRemoteURLAsync(_ url: URL) async throws -> ResolvedRemoteTarget {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    continuation.resume(returning: try validateAndResolveRemoteURL(url))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    static func validateAndResolveRemoteURL(_ url: URL) throws -> ResolvedRemoteTarget {
        guard let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https" else {
            throw QWImageDecoderError.disallowedImageSource("Unsupported remote image scheme '\(url.scheme ?? "")'")
        }
        guard url.user == nil && url.password == nil else {
            throw QWImageDecoderError.disallowedImageSource("Remote image URL must not contain credentials")
        }
        guard let rawHost = url.host, !rawHost.isEmpty else {
            throw QWImageDecoderError.disallowedImageSource("Remote image URL is missing host")
        }
        guard !rawHost.contains("%"), !rawHost.contains("\r"), !rawHost.contains("\n") else {
            throw QWImageDecoderError.disallowedImageSource("Invalid characters or zone ID in remote host '\(rawHost)'")
        }
        let stripped = rawHost.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        let normalized = stripped.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: "."))
        if normalized == "localhost" || normalized.hasSuffix(".localhost") {
            throw QWImageDecoderError.disallowedImageSource("Blocked private/loopback remote host '\(rawHost)'")
        }

        let useTLS = scheme == "https"
        let port: UInt16
        if let explicitPort = url.port {
            guard (1...65535).contains(explicitPort) else {
                throw QWImageDecoderError.disallowedImageSource("Invalid remote port '\(explicitPort)'")
            }
            port = UInt16(explicitPort)
        } else {
            port = useTLS ? 443 : 80
        }

        let components = URLComponents(url: url, resolvingAgainstBaseURL: false)
        let rawPath = components?.percentEncodedPath ?? ""
        let pathPart = rawPath.isEmpty ? "/" : rawPath
        let queryPart = components?.percentEncodedQuery.map { "?\($0)" } ?? ""
        let requestTarget = pathPart + queryPart
        guard !requestTarget.contains("\r"), !requestTarget.contains("\n") else {
            throw QWImageDecoderError.disallowedImageSource("Invalid control character in remote URL path")
        }

        let isDefaultPort = (useTLS && port == 443) || (!useTLS && port == 80)
        let isLiteralV6 = IPv6AddressBytes(stripped) != nil
        let formattedHost = isLiteralV6 ? "[\(stripped)]" : stripped
        let hostHeader = isDefaultPort ? formattedHost : "\(formattedHost):\(port)"

        if let v4 = IPv4AddressBytes(stripped) {
            guard !isBlockedIPv4(v4) else {
                throw QWImageDecoderError.disallowedImageSource("Blocked private/loopback IPv4 host '\(rawHost)'")
            }
            return ResolvedRemoteTarget(
                hostHeader: hostHeader, sniHost: stripped, port: port, useTLS: useTLS,
                requestTarget: requestTarget, resolvedIPs: [stripped]
            )
        }
        if let v6 = IPv6AddressBytes(stripped) {
            guard !isBlockedIPv6(v6) else {
                throw QWImageDecoderError.disallowedImageSource("Blocked private/loopback IPv6 host '\(rawHost)'")
            }
            return ResolvedRemoteTarget(
                hostHeader: hostHeader, sniHost: stripped, port: port, useTLS: useTLS,
                requestTarget: requestTarget, resolvedIPs: [stripped]
            )
        }

        let resolvedIPs = try resolveAndValidateHostname(stripped)
        return ResolvedRemoteTarget(
            hostHeader: hostHeader, sniHost: stripped, port: port, useTLS: useTLS,
            requestTarget: requestTarget, resolvedIPs: resolvedIPs
        )
    }

    private static func resolveAndValidateHostname(_ host: String) throws -> [String] {
        // Omit AI_ADDRCONFIG so IPv6/NAT64 records are inspected regardless of local route state.
        var hints = addrinfo(
            ai_flags: 0,
            ai_family: AF_UNSPEC,
            ai_socktype: SOCK_STREAM,
            ai_protocol: 0,
            ai_addrlen: 0,
            ai_canonname: nil,
            ai_addr: nil,
            ai_next: nil
        )
        var res: UnsafeMutablePointer<addrinfo>?
        let status = getaddrinfo(host, nil, &hints, &res)
        guard status == 0, let first = res else {
            throw QWImageDecoderError.imageDecodeFailed("Failed to resolve remote image host '\(host)'")
        }
        defer { freeaddrinfo(first) }

        var resolvedIPs: [String] = []
        var cur: UnsafeMutablePointer<addrinfo>? = first
        while let node = cur {
            if let sa = node.pointee.ai_addr {
                if sa.pointee.sa_family == sa_family_t(AF_INET) {
                    let sinAddr = UnsafeRawPointer(sa).assumingMemoryBound(to: sockaddr_in.self).pointee.sin_addr
                    let ipHost = UInt32(bigEndian: sinAddr.s_addr)
                    if isBlockedIPv4(ipHost) {
                        throw QWImageDecoderError.disallowedImageSource("Blocked resolved private IPv4 address for '\(host)'")
                    }
                    if let ipStr = formatIPv4Address(sinAddr), !resolvedIPs.contains(ipStr) {
                        resolvedIPs.append(ipStr)
                    }
                } else if sa.pointee.sa_family == sa_family_t(AF_INET6) {
                    var in6 = UnsafeRawPointer(sa).assumingMemoryBound(to: sockaddr_in6.self).pointee.sin6_addr
                    let bytes = withUnsafeBytes(of: &in6) { (raw: UnsafeRawBufferPointer) in
                        Array(raw)
                    }
                    if isBlockedIPv6(bytes) {
                        throw QWImageDecoderError.disallowedImageSource("Blocked resolved private IPv6 address for '\(host)'")
                    }
                    if let ipStr = formatIPv6Address(in6), !resolvedIPs.contains(ipStr) {
                        resolvedIPs.append(ipStr)
                    }
                }
            }
            cur = node.pointee.ai_next
        }
        if resolvedIPs.isEmpty {
            throw QWImageDecoderError.imageDecodeFailed("No usable addresses resolved for '\(host)'")
        }
        return resolvedIPs
    }

    static func fetchBoundedRemoteDataAsync(url: URL, maxBytes: Int) async throws -> Data {
        let target = try await validateAndResolveRemoteURLAsync(url)
        var lastTransportError: Error?
        for ip in target.resolvedIPs {
            do {
                return try await fetchPinnedRemoteDataAsync(
                    hostHeader: target.hostHeader,
                    sniHost: target.sniHost,
                    resolvedIP: ip,
                    port: target.port,
                    useTLS: target.useTLS,
                    requestTarget: target.requestTarget,
                    maxBytes: maxBytes
                )
            } catch let pinnedError as QWPinnedFetchTransportError {
                lastTransportError = pinnedError.underlying
            }
        }
        if let lastTransportError {
            throw lastTransportError
        }
        throw QWImageDecoderError.imageDecodeFailed("Failed to fetch remote image '\(url.absoluteString)'")
    }

    static func fetchPinnedRemoteDataAsync(
        hostHeader: String,
        sniHost: String,
        resolvedIP: String,
        port: UInt16,
        useTLS: Bool,
        requestTarget: String,
        maxBytes: Int,
        timeoutSeconds: TimeInterval = 15
    ) async throws -> Data {
        try await withCheckedThrowingContinuation { continuation in
            let fetcher = QWRemoteImageFetcher(maxBytes: maxBytes)
            fetcher.start(
                hostHeader: hostHeader,
                sniHost: sniHost,
                resolvedIP: resolvedIP,
                port: port,
                useTLS: useTLS,
                requestTarget: requestTarget,
                timeoutSeconds: timeoutSeconds
            ) { result in
                _ = fetcher
                continuation.resume(with: result)
            }
        }
    }

    static func fetchBoundedRemoteData(url: URL, maxBytes: Int) throws -> Data {
        let target = try validateAndResolveRemoteURL(url)
        var lastError: Error?
        for ip in target.resolvedIPs {
            let sem = DispatchSemaphore(value: 0)
            let box = SyncFetchResultBox()
            let fetcher = QWRemoteImageFetcher(maxBytes: maxBytes)
            fetcher.start(
                hostHeader: target.hostHeader,
                sniHost: target.sniHost,
                resolvedIP: ip,
                port: target.port,
                useTLS: target.useTLS,
                requestTarget: target.requestTarget,
                timeoutSeconds: 15
            ) { result in
                _ = fetcher
                box.set(result)
                sem.signal()
            }
            if sem.wait(timeout: .now() + 16) == .timedOut {
                fetcher.cancel()
                lastError = QWImageDecoderError.imageDecodeFailed("Timed out fetching remote image '\(url.absoluteString)'")
                continue
            }
            switch box.get() {
            case .success(let data):
                return data
            case .failure(let error as QWPinnedFetchTransportError):
                lastError = error.underlying
            case .failure(let error):
                throw error
            case nil:
                lastError = QWImageDecoderError.imageDecodeFailed("Remote image fetch terminated without result")
            }
        }
        throw lastError ?? QWImageDecoderError.imageDecodeFailed("Failed to fetch remote image '\(url.absoluteString)'")
    }

    static func IPv4AddressBytes(_ host: String) -> UInt32? {
        var addr = in_addr()
        guard inet_pton(AF_INET, host, &addr) == 1 else { return nil }
        return UInt32(bigEndian: addr.s_addr)
    }

    static func IPv6AddressBytes(_ host: String) -> [UInt8]? {
        let stripped = host.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        var addr = in6_addr()
        guard inet_pton(AF_INET6, stripped, &addr) == 1 else { return nil }
        return withUnsafeBytes(of: &addr) { (raw: UnsafeRawBufferPointer) in Array(raw) }
    }

    private static func formatIPv4Address(_ addr: in_addr) -> String? {
        var copy = addr
        var buffer = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
        return buffer.withUnsafeMutableBufferPointer { buf -> String? in
            guard let base = buf.baseAddress else { return nil }
            guard inet_ntop(AF_INET, &copy, base, socklen_t(INET_ADDRSTRLEN)) != nil else { return nil }
            return String(cString: base)
        }
    }

    private static func formatIPv6Address(_ addr: in6_addr) -> String? {
        var copy = addr
        var buffer = [CChar](repeating: 0, count: Int(INET6_ADDRSTRLEN))
        return buffer.withUnsafeMutableBufferPointer { buf -> String? in
            guard let base = buf.baseAddress else { return nil }
            guard inet_ntop(AF_INET6, &copy, base, socklen_t(INET6_ADDRSTRLEN)) != nil else { return nil }
            return String(cString: base)
        }
    }

    static func isBlockedIPv4(_ ip: UInt32) -> Bool {
        let b0 = (ip >> 24) & 0xFF
        let b1 = (ip >> 16) & 0xFF
        if b0 == 0 || b0 == 10 || b0 == 127 { return true }
        if b0 == 100 && (b1 >= 64 && b1 <= 127) { return true }
        if b0 == 169 && b1 == 254 { return true }
        if b0 == 172 && (b1 >= 16 && b1 <= 31) { return true }
        if b0 == 192 && b1 == 168 { return true }
        if b0 >= 224 { return true }
        return false
    }

    static func isBlockedIPv6(_ bytes: [UInt8]) -> Bool {
        guard bytes.count == 16 else { return true }
        // ::/96 (unspecified ::, loopback ::1, and deprecated IPv4-compatible IPv6)
        if bytes[0..<12].allSatisfy({ $0 == 0 }) { return true }
        // ::ffff:0:0/96 (IPv4-mapped IPv6)
        if bytes[0..<10].allSatisfy({ $0 == 0 }) && bytes[10] == 0xFF && bytes[11] == 0xFF { return true }
        // 64:ff9b::/96 (NAT64 well-known prefix, RFC 6052)
        if bytes[0] == 0x00 && bytes[1] == 0x64 && bytes[2] == 0xFF && bytes[3] == 0x9B
            && bytes[4..<12].allSatisfy({ $0 == 0 }) { return true }
        // 64:ff9b:1::/48 (NAT64 local-use prefix, RFC 8215)
        if bytes[0] == 0x00 && bytes[1] == 0x64 && bytes[2] == 0xFF && bytes[3] == 0x9B
            && bytes[4] == 0x00 && bytes[5] == 0x01 { return true }
        // 100::/64 (discard-only prefix, RFC 6666)
        if bytes[0] == 0x01 && bytes[1..<8].allSatisfy({ $0 == 0 }) { return true }
        // 2002::/16 (6to4 tunneling)
        if bytes[0] == 0x20 && bytes[1] == 0x02 { return true }
        if (bytes[0] & 0xFE) == 0xFC { return true }
        if bytes[0] == 0xFE && (bytes[1] & 0xC0) == 0x80 { return true }
        if bytes[0] == 0xFE && (bytes[1] & 0xC0) == 0xC0 { return true }
        if bytes[0] == 0xFF { return true }
        return false
    }
}

private struct QWPinnedFetchTransportError: Error {
    let underlying: Error
}

private final class SyncFetchResultBox: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Result<Data, Error>?

    func set(_ result: Result<Data, Error>) {
        lock.withLock { value = result }
    }

    func get() -> Result<Data, Error>? {
        lock.withLock { value }
    }
}

private final class QWRemoteImageFetcher: @unchecked Sendable {
    private let maxBytes: Int
    private let queue = DispatchQueue(label: "qwegine.pinned-remote-image-fetch")
    private let lock = NSLock()
    private var connection: NWConnection?
    private var timeoutWorkItem: DispatchWorkItem?
    private var completion: ((Result<Data, Error>) -> Void)?
    private var finished = false
    private var parser: QWPinnedHTTPResponseParser

    init(maxBytes: Int) {
        self.maxBytes = maxBytes
        self.parser = QWPinnedHTTPResponseParser(maxBytes: maxBytes)
    }

    func start(
        hostHeader: String,
        sniHost: String,
        resolvedIP: String,
        port: UInt16,
        useTLS: Bool,
        requestTarget: String,
        timeoutSeconds: TimeInterval,
        completion: @escaping (Result<Data, Error>) -> Void
    ) {
        guard let nwHost = Self.makePinnedIPHost(resolvedIP),
              let nwPort = NWEndpoint.Port(rawValue: port) else {
            completion(.failure(QWImageDecoderError.imageDecodeFailed("Invalid pinned IP endpoint '\(resolvedIP)'")))
            return
        }

        let tcpOptions = NWProtocolTCP.Options()
        tcpOptions.connectionTimeout = Int(max(1, min(10, timeoutSeconds)))
        let parameters: NWParameters
        if useTLS {
            let tlsOptions = NWProtocolTLS.Options()
            sniHost.withCString { cHost in
                sec_protocol_options_set_tls_server_name(tlsOptions.securityProtocolOptions, cHost)
            }
            parameters = NWParameters(tls: tlsOptions, tcp: tcpOptions)
        } else {
            parameters = NWParameters(tls: nil, tcp: tcpOptions)
        }

        let conn = NWConnection(to: .hostPort(host: nwHost, port: nwPort), using: parameters)
        let workItem = DispatchWorkItem { [weak self] in
            self?.finish(with: .failure(
                QWPinnedFetchTransportError(
                    underlying: QWImageDecoderError.imageDecodeFailed("Timed out fetching remote image from '\(sniHost)'")
                )
            ))
        }

        lock.withLock {
            self.connection = conn
            self.timeoutWorkItem = workItem
            self.completion = completion
        }
        queue.asyncAfter(deadline: .now() + timeoutSeconds, execute: workItem)

        let requestText =
            "GET \(requestTarget) HTTP/1.1\r\n" +
            "Host: \(hostHeader)\r\n" +
            "Accept: image/*,*/*;q=0.8\r\n" +
            "Accept-Encoding: identity\r\n" +
            "Connection: close\r\n\r\n"

        conn.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                self.sendRequestAndReceive(requestData: Data(requestText.utf8), over: conn)
            case .waiting(let err):
                self.finish(with: .failure(QWPinnedFetchTransportError(
                    underlying: QWImageDecoderError.imageDecodeFailed("Failed to connect to '\(sniHost)': \(err.localizedDescription)")
                )))
            case .failed(let err):
                self.finish(with: .failure(QWPinnedFetchTransportError(
                    underlying: QWImageDecoderError.imageDecodeFailed("Failed to fetch remote image: \(err.localizedDescription)")
                )))
            case .cancelled:
                self.finish(with: .failure(QWPinnedFetchTransportError(
                    underlying: QWImageDecoderError.imageDecodeFailed("Remote image fetch cancelled")
                )))
            default:
                break
            }
        }
        conn.start(queue: queue)
    }

    func cancel() {
        finish(with: .failure(QWImageDecoderError.imageDecodeFailed("Remote image fetch cancelled")))
    }

    private func sendRequestAndReceive(requestData: Data, over conn: NWConnection) {
        conn.send(content: requestData, completion: .contentProcessed { [weak self] sendError in
            guard let self else { return }
            if let sendError {
                self.finish(with: .failure(QWPinnedFetchTransportError(
                    underlying: QWImageDecoderError.imageDecodeFailed("Failed sending remote image request: \(sendError.localizedDescription)")
                )))
                return
            }
            self.receiveNextChunk(over: conn)
        })
    }

    private func receiveNextChunk(over conn: NWConnection) {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            if let data, !data.isEmpty {
                do {
                    if let finishedBody = try self.parser.feed(data) {
                        self.finish(with: .success(finishedBody))
                        return
                    }
                } catch {
                    self.finish(with: .failure(error))
                    return
                }
            }
            if let error {
                if !self.parser.hasReceivedHeaders {
                    self.finish(with: .failure(QWPinnedFetchTransportError(
                        underlying: QWImageDecoderError.imageDecodeFailed("Failed reading remote image: \(error.localizedDescription)")
                    )))
                } else {
                    self.finish(with: .failure(
                        QWImageDecoderError.imageDecodeFailed("Failed reading remote image: \(error.localizedDescription)")
                    ))
                }
                return
            }
            if isComplete {
                do {
                    let body = try self.parser.finishAtEOF()
                    self.finish(with: .success(body))
                } catch {
                    self.finish(with: .failure(error))
                }
                return
            }
            if data == nil || data?.isEmpty == true {
                self.finish(with: .failure(QWImageDecoderError.imageDecodeFailed("Remote image connection closed unexpectedly")))
                return
            }
            self.receiveNextChunk(over: conn)
        }
    }

    private func finish(with result: Result<Data, Error>) {
        let work: (NWConnection?, DispatchWorkItem?, ((Result<Data, Error>) -> Void)?)? = lock.withLock {
            guard !finished else { return nil }
            finished = true
            let out = (connection, timeoutWorkItem, completion)
            connection = nil
            timeoutWorkItem = nil
            completion = nil
            return out
        }
        guard let (conn, item, cb) = work else { return }
        item?.cancel()
        conn?.stateUpdateHandler = nil
        conn?.cancel()
        cb?(result)
    }

    private static func makePinnedIPHost(_ ip: String) -> NWEndpoint.Host? {
        if let v4 = IPv4Address(ip) {
            return .ipv4(v4)
        }
        if let v6 = IPv6Address(ip) {
            return .ipv6(v6)
        }
        return nil
    }
}

private struct QWPinnedHTTPResponseParser {
    private static let maxHeaderBytes = 64 * 1024
    private static let headerTerminator = Data("\r\n\r\n".utf8)
    private static let lineTerminator = Data("\r\n".utf8)

    let maxBytes: Int
    private(set) var hasReceivedHeaders = false
    private var buffer = Data()
    private var isChunked = false
    private var expectedContentLength: Int?
    private var body = Data()
    private var chunkPendingSize: Int?
    private var expectingChunkCRLF = false

    init(maxBytes: Int) {
        self.maxBytes = maxBytes
    }

    mutating func feed(_ chunk: Data) throws -> Data? {
        buffer.append(chunk)
        if !hasReceivedHeaders {
            guard let range = buffer.range(of: Self.headerTerminator) else {
                if buffer.count > Self.maxHeaderBytes {
                    throw QWImageDecoderError.imageDecodeFailed("Remote image HTTP response headers too large")
                }
                return nil
            }
            let headerData = Data(buffer[..<range.lowerBound])
            buffer.removeSubrange(..<range.upperBound)
            try parseHeaders(headerData)
            hasReceivedHeaders = true
        }
        return try consumeBodyBuffer()
    }

    mutating func finishAtEOF() throws -> Data {
        guard hasReceivedHeaders else {
            throw QWImageDecoderError.imageDecodeFailed("Empty HTTP response from remote image host")
        }
        if let done = try consumeBodyBuffer() {
            return done
        }
        if isChunked {
            throw QWImageDecoderError.imageDecodeFailed("Truncated chunked HTTP response from remote image host")
        }
        if let expected = expectedContentLength, body.count < expected {
            throw QWImageDecoderError.imageDecodeFailed("Truncated HTTP response body (\(body.count) < \(expected) bytes)")
        }
        return body
    }

    private mutating func parseHeaders(_ data: Data) throws {
        guard let text = String(data: data, encoding: .isoLatin1) else {
            throw QWImageDecoderError.imageDecodeFailed("Invalid HTTP response header encoding")
        }
        let lines = text.components(separatedBy: "\r\n")
        guard let statusLine = lines.first else {
            throw QWImageDecoderError.imageDecodeFailed("Missing HTTP status line")
        }
        let statusParts = statusLine.split(separator: " ", maxSplits: 2, omittingEmptySubsequences: true)
        guard statusParts.count >= 2,
              statusParts[0].uppercased().hasPrefix("HTTP/"),
              let statusCode = Int(statusParts[1]) else {
            throw QWImageDecoderError.imageDecodeFailed("Malformed HTTP status line '\(statusLine)'")
        }
        if (300...399).contains(statusCode) {
            throw QWImageDecoderError.disallowedImageSource("Redirects are blocked for remote image fetches")
        }
        guard (200...299).contains(statusCode) else {
            throw QWImageDecoderError.imageDecodeFailed("Remote image returned HTTP \(statusCode)")
        }

        var parsedContentLength: Int?
        for line in lines.dropFirst() {
            if line.isEmpty { continue }
            guard let colonIdx = line.firstIndex(of: ":") else { continue }
            let name = line[..<colonIdx].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: colonIdx)...].trimmingCharacters(in: .whitespaces)
            switch name {
            case "content-encoding":
                let lower = value.lowercased()
                if !lower.isEmpty && lower != "identity" {
                    throw QWImageDecoderError.imageDecodeFailed("Unsupported remote image Content-Encoding '\(value)'")
                }
            case "transfer-encoding":
                let tokens = value.lowercased().split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
                if tokens.contains("chunked") {
                    isChunked = true
                }
            case "content-length":
                guard !value.isEmpty, value.utf8.allSatisfy({ $0 >= 0x30 && $0 <= 0x39 }) else {
                    throw QWImageDecoderError.imageDecodeFailed("Invalid remote image Content-Length '\(value)'")
                }
                guard let len = Int(value) else {
                    throw QWImageDecoderError.imageTooLarge("Remote image Content-Length exceeds \(maxBytes) bytes")
                }
                if let prev = parsedContentLength, prev != len {
                    throw QWImageDecoderError.imageDecodeFailed("Conflicting Content-Length headers from remote image host")
                }
                if len > maxBytes {
                    throw QWImageDecoderError.imageTooLarge("Remote image Content-Length exceeds \(maxBytes) bytes")
                }
                parsedContentLength = len
            default:
                break
            }
        }
        if !isChunked {
            expectedContentLength = parsedContentLength
        }
    }

    private mutating func consumeBodyBuffer() throws -> Data? {
        if isChunked {
            return try consumeChunkedBuffer()
        }
        if let expected = expectedContentLength {
            let needed = expected - body.count
            if needed > 0 && !buffer.isEmpty {
                let take = min(needed, buffer.count)
                guard body.count + take <= maxBytes else {
                    throw QWImageDecoderError.imageTooLarge("Remote image exceeds \(maxBytes) bytes")
                }
                body.append(buffer.prefix(take))
                buffer.removeSubrange(..<take)
            }
            return body.count >= expected ? body : nil
        }
        if !buffer.isEmpty {
            guard body.count + buffer.count <= maxBytes else {
                throw QWImageDecoderError.imageTooLarge("Remote image exceeds \(maxBytes) bytes")
            }
            body.append(buffer)
            buffer.removeAll(keepingCapacity: true)
        }
        return nil
    }

    private mutating func consumeChunkedBuffer() throws -> Data? {
        while true {
            if expectingChunkCRLF {
                guard buffer.count >= 2 else { return nil }
                guard buffer[buffer.startIndex] == 0x0D, buffer[buffer.index(after: buffer.startIndex)] == 0x0A else {
                    throw QWImageDecoderError.imageDecodeFailed("Malformed chunk terminator in remote image response")
                }
                buffer.removeFirst(2)
                expectingChunkCRLF = false
            }
            if let size = chunkPendingSize {
                guard buffer.count >= size else { return nil }
                body.append(buffer.prefix(size))
                buffer.removeFirst(size)
                chunkPendingSize = nil
                expectingChunkCRLF = true
                continue
            }
            guard let crlf = buffer.range(of: Self.lineTerminator) else {
                if buffer.count > 1024 {
                    throw QWImageDecoderError.imageDecodeFailed("Malformed chunk header in remote image response")
                }
                return nil
            }
            let sizeLineData = Data(buffer[..<crlf.lowerBound])
            buffer.removeSubrange(..<crlf.upperBound)
            guard let sizeLine = String(data: sizeLineData, encoding: .ascii) else {
                throw QWImageDecoderError.imageDecodeFailed("Invalid chunk size encoding in remote image response")
            }
            let hexPart = sizeLine.split(separator: ";", maxSplits: 1).first.map {
                $0.trimmingCharacters(in: .whitespaces)
            } ?? ""
            guard !hexPart.isEmpty, let size = UInt64(hexPart, radix: 16) else {
                throw QWImageDecoderError.imageDecodeFailed("Invalid chunk size '\(sizeLine)' in remote image response")
            }
            if size == 0 {
                return body
            }
            let remainingBudget = max(0, maxBytes - body.count)
            guard size <= UInt64(remainingBudget) else {
                throw QWImageDecoderError.imageTooLarge("Remote image exceeds \(maxBytes) bytes")
            }
            chunkPendingSize = Int(size)
        }
    }
}
