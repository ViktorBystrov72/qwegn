// QWImageDecodePolicy.swift — Image source policy, SSRF IP/DNS guard, and bounded remote fetch.

import Darwin
import Foundation

public struct QWImageDecodePolicy: Sendable, Equatable {
    public static let serverDefault = QWImageDecodePolicy()
    public static let cliLocal = QWImageDecodePolicy(allowLocalImagePaths: true)

    public let allowDataURLs: Bool
    public let allowLocalImagePaths: Bool
    public let allowRemoteImageURLs: Bool
    public let maxSourceBytes: Int
    public let maxPixelDimension: Int

    public init(
        allowDataURLs: Bool = true,
        allowLocalImagePaths: Bool = false,
        allowRemoteImageURLs: Bool = false,
        maxSourceBytes: Int = 20 * 1024 * 1024,
        maxPixelDimension: Int = 8192
    ) {
        self.allowDataURLs = allowDataURLs
        self.allowLocalImagePaths = allowLocalImagePaths
        self.allowRemoteImageURLs = allowRemoteImageURLs
        self.maxSourceBytes = maxSourceBytes
        self.maxPixelDimension = maxPixelDimension
    }

    public static func validateRemoteURL(_ url: URL) throws {
        guard let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https" else {
            throw QWImageDecoderError.disallowedImageSource("Unsupported remote image scheme '\(url.scheme ?? "")'")
        }
        guard let host = url.host, !host.isEmpty else {
            throw QWImageDecoderError.disallowedImageSource("Remote image URL is missing host")
        }
        if host.lowercased() == "localhost" {
            throw QWImageDecoderError.disallowedImageSource("Blocked private/loopback remote host '\(host)'")
        }
        if let v4 = IPv4AddressBytes(host) {
            guard !isBlockedIPv4(v4) else {
                throw QWImageDecoderError.disallowedImageSource("Blocked private/loopback IPv4 host '\(host)'")
            }
            return
        }
        if let v6 = IPv6AddressBytes(host) {
            guard !isBlockedIPv6(v6) else {
                throw QWImageDecoderError.disallowedImageSource("Blocked private/loopback IPv6 host '\(host)'")
            }
            return
        }
        var hints = addrinfo(
            ai_flags: AI_ADDRCONFIG,
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
        var cur: UnsafeMutablePointer<addrinfo>? = first
        var sawAddress = false
        while let node = cur {
            if let sa = node.pointee.ai_addr {
                if sa.pointee.sa_family == sa_family_t(AF_INET) {
                    sawAddress = true
                    let addr = UnsafeRawPointer(sa).assumingMemoryBound(to: sockaddr_in.self).pointee.sin_addr.s_addr
                    let ipHost = UInt32(bigEndian: addr)
                    if isBlockedIPv4(ipHost) {
                        throw QWImageDecoderError.disallowedImageSource("Blocked resolved private IPv4 address for '\(host)'")
                    }
                } else if sa.pointee.sa_family == sa_family_t(AF_INET6) {
                    sawAddress = true
                    var in6 = UnsafeRawPointer(sa).assumingMemoryBound(to: sockaddr_in6.self).pointee.sin6_addr
                    let bytes = withUnsafeBytes(of: &in6) { (raw: UnsafeRawBufferPointer) in
                        Array(raw)
                    }
                    if isBlockedIPv6(bytes) {
                        throw QWImageDecoderError.disallowedImageSource("Blocked resolved private IPv6 address for '\(host)'")
                    }
                }
            }
            cur = node.pointee.ai_next
        }
        if !sawAddress {
            throw QWImageDecoderError.imageDecodeFailed("No usable addresses resolved for '\(host)'")
        }
    }

    static func fetchBoundedRemoteData(url: URL, maxBytes: Int) throws -> Data {
        try validateRemoteURL(url)
        let fetcher = QWRemoteImageFetcher(maxBytes: maxBytes)
        return try fetcher.fetch(url: url)
    }

    private static func IPv4AddressBytes(_ host: String) -> UInt32? {
        var addr = in_addr()
        guard inet_pton(AF_INET, host, &addr) == 1 else { return nil }
        return UInt32(bigEndian: addr.s_addr)
    }

    private static func IPv6AddressBytes(_ host: String) -> [UInt8]? {
        let stripped = host.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        var addr = in6_addr()
        guard inet_pton(AF_INET6, stripped, &addr) == 1 else { return nil }
        return withUnsafeBytes(of: &addr) { (raw: UnsafeRawBufferPointer) in Array(raw) }
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
        if bytes.allSatisfy({ $0 == 0 }) { return true }
        if bytes[0..<15].allSatisfy({ $0 == 0 }) && bytes[15] == 1 { return true }
        if bytes[0..<10].allSatisfy({ $0 == 0 }) && bytes[10] == 0xFF && bytes[11] == 0xFF {
            let v4 = (UInt32(bytes[12]) << 24) | (UInt32(bytes[13]) << 16) | (UInt32(bytes[14]) << 8) | UInt32(bytes[15])
            return isBlockedIPv4(v4)
        }
        if (bytes[0] & 0xFE) == 0xFC { return true }
        if bytes[0] == 0xFE && (bytes[1] & 0xC0) == 0x80 { return true }
        if bytes[0] == 0xFF { return true }
        return false
    }
}

private final class QWRemoteImageFetcher: NSObject, URLSessionDataDelegate, URLSessionTaskDelegate, @unchecked Sendable {
    private let maxBytes: Int
    private let lock = NSLock()
    private let sem = DispatchSemaphore(value: 0)
    private var accumulated = Data()
    private var failure: Error?
    private var finished = false

    init(maxBytes: Int) {
        self.maxBytes = maxBytes
    }

    func fetch(url: URL) throws -> Data {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 10
        config.timeoutIntervalForResource = 15
        config.httpShouldSetCookies = false
        let session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        session.dataTask(with: req).resume()
        _ = sem.wait(timeout: .now() + 16)
        lock.lock()
        defer { lock.unlock() }
        if let failure { throw failure }
        if !finished {
            throw QWImageDecoderError.imageDecodeFailed("Timed out fetching remote image '\(url.absoluteString)'")
        }
        return accumulated
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        lock.lock()
        failure = QWImageDecoderError.disallowedImageSource("Redirects are blocked for remote image fetches")
        lock.unlock()
        completionHandler(nil)
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            lock.lock()
            failure = QWImageDecoderError.imageDecodeFailed("Remote image returned HTTP \(http.statusCode)")
            lock.unlock()
            completionHandler(.cancel)
            return
        }
        if response.expectedContentLength > Int64(maxBytes) {
            lock.lock()
            failure = QWImageDecoderError.imageTooLarge("Remote image Content-Length exceeds \(maxBytes) bytes")
            lock.unlock()
            completionHandler(.cancel)
            return
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        lock.lock()
        let exceeded = accumulated.count + data.count > maxBytes
        if exceeded {
            failure = QWImageDecoderError.imageTooLarge("Remote image exceeds \(maxBytes) bytes")
        } else {
            accumulated.append(data)
        }
        lock.unlock()
        if exceeded { dataTask.cancel() }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        lock.lock()
        if failure == nil, let error {
            failure = QWImageDecoderError.imageDecodeFailed("Failed to fetch remote image: \(error.localizedDescription)")
        }
        finished = true
        lock.unlock()
        sem.signal()
    }
}
