# 05 — HTTP Server Security, Framing, Image SSRF & Stream Cancellation

> Normative specification for `QWServer` (`Sources/Apps/QWServer/`), `QWCore/QWPaths.swift`, and `QWMenuBar` PID hygiene.

## 1. `O(1)` Bounded HTTP/1.1 Framing (`QWHttpIncrementalParser`)
`QWHttpIncrementalParser` (`Sources/Apps/QWServer/Core/QWHttpIncrementalParser.swift`) avoids quadratic `\r\n\r\n` rescanning over fragmented TCP reads via a monotonic `scannedOffset` cursor (`searchStart = max(0, scannedOffset - 3)`):
- Enforces `defaultMaxHeaderBytes = 64 KiB` (`431`) and `defaultMaxBodyBytes = 32 MiB` (`413`).
- Validates `Content-Length` without integer overflow (`400`) and rejects unsupported `Transfer-Encoding` (`501`).

## 2. Bind, Auth & CORS Controls (`QWHttpSecuritySupport`)
- **Loopback default:** HTTP listener defaults to `127.0.0.1`. `QWHttpSecurity.isLoopbackHost` validates `127.*`, `localhost`, `::1`, `[::1]`.
- **Non-loopback bind gate:** Starting `QWServer` on any non-loopback host without a configured `apiKey` throws `QWHttpServerError.unauthenticatedNonLoopbackBind` before socket bind.
- **Constant-time authentication:** `QWHttpSecurity.extractPresentedToken` accepts `Authorization: Bearer <token>` or `x-api-key`, validated with `QWHttpSecurity.constantTimeEquals`.
- **CORS & XSS safety:** `QWHttpSecurity.resolvedAllowedOrigin` permits loopback HTTP/HTTPS origins (with empty credentials/path/query/fragment) or explicit `corsAllowedOrigins` matches; `QWHttpSecurity.escapeHTML` sanitizes error/HTML responses.

## 3. Connection & Inference Cancellation Propagation
- `QWConnectionCancellationHub` hooks `NWConnection.stateUpdateHandler` to cancel both:
  1. Per-request `Task` (`routeTask.cancel()`) in `QWHttpServer.dispatchRoute`, and
  2. Lock-free `QWGenerationCancelFlag` (`Synchronization.Atomic<Bool>`), checked inside `QWEngineLifecycleActor+Chat` generation loops and before post-generation response emission across `/v1/chat/completions`, `/v1/messages`, `/api/chat`, and `/api/generate`.

## 4. Image Source & SSRF Policy (`QWImageDecodePolicy`)
- **`serverDefault`:** Permits `data:image/*;base64` only (`allowLocalImagePaths = false`, `allowRemoteImageURLs = false`); local paths and remote HTTP(S) require explicit server opt-in; CLI uses `QWImageDecodePolicy.cliLocal`.
- **SSRF IP/DNS/redirect filter:** `validateRemoteURL` rejects non-HTTP(S) schemes, missing hosts, `localhost`, private/loopback/link-local IPv4 & IPv6 literals, and DNS resolutions (`getaddrinfo`) mapping to blocked ranges; downloads are byte-capped (`20 MiB`) and downsampled via `ImageIO` thumbnailing (`maxPixelDimension = 8192`).

## 5. Filesystem Permissions & PID Verification
- Sensitive config, profile, and PID/lock state directories under `~/.qwg` (`QWPaths`, `QWSettingsManager`, `QWProfileStore`, `QWCli`) enforce restrictive POSIX modes (`0o700` dirs / `0o600` files) and path-traversal canonicalization.
- Process checks in `TelemetryObservable` verify live PIDs against executable identity via `proc_pidpath` rather than trusting stale PID numbers alone.
