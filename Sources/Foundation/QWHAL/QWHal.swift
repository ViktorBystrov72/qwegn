// QWHal.swift — Metal 4.0 hardware abstraction layer for Qwen3.8 engines.
// Owns MTLDevice, command queues, GPU family capability probes, and metallib
// pipeline caching (with post-init lock-free freeze).
// Nothing above QWHAL calls MTLCreateSystemDefaultDevice directly.

import Foundation
import Metal
import os
import QuartzCore
import QWCore
import Synchronization

public struct QWDeviceCaps: Sendable {
    public let name: String
    public let familyApple9: Bool
    public let familyApple10: Bool
    public let familyApple11: Bool
    public let unifiedMemory: Bool
    public let maxWorkingSetBytes: Int
    public let threadsPerTG: Int
    /// Measured DRAM read bandwidth, GB/s (S3 fair run; 585 on this machine).
    public let measuredReadBWBps: Double
    /// Measured L2-resident read bandwidth, GB/s (~950 at 32 MB).
    public let measuredL2BWBps: Double

    public var isMetal4TensorCapable: Bool { familyApple10 }
}

extension QWDeviceCaps {
    /// The device's capabilities, read from the device itself.
    public static func probe(_ dev: MTLDevice) -> QWDeviceCaps {
        var apple9 = false, apple10 = false, apple11 = false
        if #available(macOS 14.0, *) { apple9 = dev.supportsFamily(.apple9) }
        if #available(macOS 15.0, *) { apple10 = dev.supportsFamily(.apple10) }
        if #available(macOS 26.0, *), let f11 = MTLGPUFamily(rawValue: 1011) {
            apple11 = dev.supportsFamily(f11)
        }
        return QWDeviceCaps(
            name: dev.name,
            familyApple9: apple9,
            familyApple10: apple10,
            familyApple11: apple11,
            unifiedMemory: dev.hasUnifiedMemory,
            maxWorkingSetBytes: Int(dev.recommendedMaxWorkingSetSize),
            threadsPerTG: 1024,
            measuredReadBWBps: 585e9,
            measuredL2BWBps: 950e9
        )
    }

    /// The capabilities of this machine, found once. They do not change
    /// while the process runs, and finding them means creating a Metal
    /// device: a request must not pay for that every time.
    public static let current: QWDeviceCaps? = {
        guard let dev = MTLCreateSystemDefaultDevice() else { return nil }
        return probe(dev)
    }()
}

public enum QWHalError: Error, CustomStringConvertible {
    case noDevice
    case metallibNotFound(String)
    case metallibLoadFailed(String)
    case metallibBasisMismatch(path: String, expected: QWWeightBasis, found: QWWeightBasis)
    case kernelMissing(String)

    public var description: String {
        switch self {
        case .noDevice: return "QWHAL: no Metal device"
        case .metallibNotFound(let p): return "QWHAL: metallib not found at \(p)"
        case .metallibLoadFailed(let p): return "QWHAL: metallib load failed: \(p)"
        case .metallibBasisMismatch(let p, let expected, let found):
            return "QWHAL: metallib at \(p) is in the \(found) basis, but the model needs \(expected)"
        case .kernelMissing(let n): return "QWHAL: kernel '\(n)' missing from metallib"
        }
    }
}

/// One HAL instance per process. Sendable through immutable fields; the
/// queue is internally synchronized by Metal itself.
public final class QWHal: @unchecked Sendable {
    public enum ComputeDispatchMode: Sendable {
        case serial
        case concurrent
    }

    public let device: any MTLDevice
    public let caps: QWDeviceCaps
    public let queue: any MTLCommandQueue
    /// Long-lived allocations every encoder references (see QWResidency.swift).
    let residencyLock = NSLock()
    var keepAlive: [any MTLResource] = []
    var keepAliveIDs = Set<ObjectIdentifier>()
    var keepAliveLastTouch: CFAbsoluteTime = 0
    var keepAliveHeartbeat: (any DispatchSourceTimer)?
    var keepAliveNop: (any MTLComputePipelineState)?
    public let metallib: any MTLLibrary
    public let geometry: QWModelGeometry
    private let functionConstants: MTLFunctionConstantValues

    /// Immutable snapshot published atomically so post-freeze hits take no lock.
    final class FrozenPSOSnapshot: @unchecked Sendable {
        let map: [String: any MTLComputePipelineState]
        init(_ map: [String: any MTLComputePipelineState]) { self.map = map }
    }

    struct MutablePSOState: @unchecked Sendable {
        var map: [String: any MTLComputePipelineState] = [:]
        var frozen = false
        var lockCount = 0
        var retiredSnapshots: [FrozenPSOSnapshot] = []
    }

    let psoState = OSAllocatedUnfairLock(initialState: MutablePSOState())
    private let frozenSnapshotPtr = Atomic<UnsafeMutableRawPointer?>(nil)

    private func publishFrozenSnapshotLocked(_ state: inout MutablePSOState) {
        let ptr = Unmanaged.passRetained(FrozenPSOSnapshot(state.map)).toOpaque()
        if let prev = frozenSnapshotPtr.exchange(ptr, ordering: .releasing) {
            state.retiredSnapshots.append(
                Unmanaged<FrozenPSOSnapshot>.fromOpaque(prev).takeRetainedValue()
            )
        }
    }

    /// Freeze the pipeline cache so post-freeze hits take no lock. Idempotent.
    public func freezePipelines() {
        psoState.withLock { state in
            state.frozen = true
            publishFrozenSnapshotLocked(&state)
        }
    }

    /// The constants the standard path added, as values. The builder is
    /// write-only, so this is the readable seam over the same numbers.
    public static func standardConstants(geometry: QWModelGeometry)
        -> (hasQGate: Bool, qStride: UInt32, attnScale: Float) {
        (geometry.attnOutputGate,
         geometry.attnOutputGate ? 2 : 1,
         Float(1.0 / Double(geometry.headDim).squareRoot()))
    }

    /// The basis this HAL's kernels assume the activations are in.
    public let weightBasis: QWWeightBasis

    /// The marker kernels the two libraries carry. Only the names are used:
    /// the HAL asks the library which basis it is rather than trusting the
    /// path it was given (the standard name is spelled in two pieces).
    static let standardBasisMarker = "qwg_basis_marker" + "_standard"
    static let rotatedBasisMarker = "qwg_basis_marker"

    /// The basis a metallib was compiled for. A library with no marker at
    /// all predates the marker, and that is the incumbent's — the rotated
    /// basis.
    static func basis(of library: any MTLLibrary) -> QWWeightBasis {
        let names = library.functionNames
        if names.contains(standardBasisMarker) { return .standard }
        if names.contains(rotatedBasisMarker) { return .rotated }
        // Neither marker: a library built before the marker existed. That is
        // the incumbent's own, and the incumbent's basis is the rotated one.
        return .rotated
    }

    deinit {
        keepAliveHeartbeat?.cancel()
        if let raw = frozenSnapshotPtr.exchange(nil, ordering: .acquiring) {
            Unmanaged<FrozenPSOSnapshot>.fromOpaque(raw).release()
        }
    }

    /// - Parameter weightBasis: the basis the model's weights are quantised
    ///   in. It picks the metallib and so decides whether the kernels rotate
    ///   the activations; it must be the basis the loader actually produced.
    public init(metallibPath: String? = nil, geometry: QWModelGeometry,
                weightBasis: QWWeightBasis = .rotated) throws {
        self.geometry = geometry
        self.weightBasis = weightBasis
        self.functionConstants = Self.makeFunctionConstants(geometry: geometry)
        guard let dev = MTLCreateSystemDefaultDevice() else { throw QWHalError.noDevice }
        device = dev
        // Cold-expert decode commits ~2 CBs per MoE layer; the default cap of 64 would block encode.
        guard let q = dev.makeCommandQueue(maxCommandBufferCount: 256) else { throw QWHalError.noDevice }
        queue = q

        let mtlPath: String
        if let p = metallibPath {
            mtlPath = p
        } else {
            mtlPath = QWHal.defaultMetallibPath(basis: weightBasis)
        }
        guard FileManager.default.fileExists(atPath: mtlPath) else {
            throw QWHalError.metallibNotFound(mtlPath)
        }
        do {
            metallib = try dev.makeLibrary(URL: URL(fileURLWithPath: mtlPath))
        } catch {
            throw QWHalError.metallibLoadFailed(mtlPath)
        }
        // The library says which basis it is. If it is not the basis the
        // model's weights are in, the run would produce numbers that mean
        // nothing and no error anywhere — so it is refused here instead.
        let libBasis = Self.basis(of: metallib)
        guard libBasis == weightBasis else {
            throw QWHalError.metallibBasisMismatch(path: mtlPath,
                                                    expected: weightBasis,
                                                    found: libBasis)
        }

        caps = QWDeviceCaps.probe(dev)
    }

    /// Bundled metallib lookup: env override -> bundle -> build dir walk-up -> installed Qwegn.app.
    /// The basis picks the file: two builds of the same sources, one per basis.
    public static func defaultMetallibPath(basis: QWWeightBasis,
                                           appBundleURL: URL? = nil,
                                           searchDirs: [URL]? = nil) -> String {
        if let env = ProcessInfo.processInfo.environment["QW_METALLIB"] { return env }
        // A bundle that ships one library per basis names them after the
        // basis; the incumbent's generic "default" is the last resort, and a
        // bundle holding the wrong one is refused by the basis check below.
        let resource = basis.metallibFileName.replacingOccurrences(of: ".metallib", with: "")
        if searchDirs == nil, appBundleURL == nil {
            if let b = Bundle.main.url(forResource: resource, withExtension: "metallib") {
                return b.path
            }
            if let b = Bundle.main.url(forResource: "default", withExtension: "metallib") {
                return b.path
            }
        }
        let name = basis.metallibFileName
        let execDir = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().deletingLastPathComponent()
        let startDirs = searchDirs ?? [execDir, URL(fileURLWithPath: FileManager.default.currentDirectoryPath)]
        for rel in ["Sources/QWMetal/\(name)",
                    ".build/release/\(name)",
                    ".build/debug/\(name)",
                    name] {
            for startDir in startDirs {
                var dir = startDir
                for _ in 0..<6 {
                    let cand = dir.appendingPathComponent(rel)
                    if FileManager.default.fileExists(atPath: cand.path) { return cand.path }
                    dir = dir.deletingLastPathComponent()
                }
            }
        }
        var bundleDirs: [URL] = []
        if let explicitApp = appBundleURL {
            bundleDirs.append(explicitApp.appendingPathComponent("Contents/Resources", isDirectory: true))
        } else if let envApp = ProcessInfo.processInfo.environment["QWEGN_APP_BUNDLE"], !envApp.isEmpty {
            bundleDirs.append(URL(fileURLWithPath: envApp.expandingTildePath).appendingPathComponent("Contents/Resources", isDirectory: true))
        }
        bundleDirs.append(contentsOf: [
            URL(fileURLWithPath: "/Applications/Qwegn.app/Contents/Resources", isDirectory: true),
            FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Applications/Qwegn.app/Contents/Resources", isDirectory: true),
            execDir.deletingLastPathComponent().appendingPathComponent("share/qwegn", isDirectory: true),
            URL(fileURLWithPath: "/usr/local/share/qwegn", isDirectory: true),
        ])
        for dir in bundleDirs {
            let cand = dir.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: cand.path) { return cand.path }
        }
        return name
    }

    /// Cached pipeline state. First use pays creation; post-freeze hits take no lock.
    public func pipeline(_ name: String) throws -> any MTLComputePipelineState {
        if let raw = frozenSnapshotPtr.load(ordering: .acquiring),
           let hit = Unmanaged<FrozenPSOSnapshot>.fromOpaque(raw).takeUnretainedValue().map[name] {
            return hit
        }
        return try psoState.withLock { state in
            state.lockCount += 1
            if let p = state.map[name] {
                if state.frozen { publishFrozenSnapshotLocked(&state) }
                return p
            }
            let fn: any MTLFunction
            do {
                fn = try metallib.makeFunction(name: name, constantValues: functionConstants)
            } catch {
                throw QWHalError.kernelMissing(name)
            }
            let psoT0 = CFAbsoluteTimeGetCurrent()
            let p = try device.makeComputePipelineState(function: fn)
            QWHalPSOStats.add(CFAbsoluteTimeGetCurrent() - psoT0)
            state.map[name] = p
            if state.frozen { publishFrozenSnapshotLocked(&state) }
            return p
        }
    }

    /// Begin a command buffer + encoder as a pair.
    public func begin() -> (any MTLCommandBuffer, any MTLComputeCommandEncoder)? {
        guard let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return nil }
        touchKeepAlive(enc)
        return (cb, enc)
    }

    /// Begin with explicit compute dispatch type.
    public func begin(_ mode: ComputeDispatchMode) -> (any MTLCommandBuffer, any MTLComputeCommandEncoder)? {
        guard let cb = queue.makeCommandBuffer() else { return nil }
        switch mode {
        case .serial:
            guard let enc = cb.makeComputeCommandEncoder() else { return nil }
            touchKeepAlive(enc)
            return (cb, enc)
        case .concurrent:
            guard let enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent) else { return nil }
            touchKeepAlive(enc)
            return (cb, enc)
        }
    }

    public func end(_ cb: any MTLCommandBuffer, _ enc: any MTLComputeCommandEncoder) {
        enc.endEncoding()
        cb.commit()
    }

    /// Ends `enc` and opens a fresh compute command encoder on `cb` with matching `dispatchType`
    /// so GPU-written indirect dispatch counts are visible to the AGX command processor.
    public func rotateEncoder(_ cb: any MTLCommandBuffer,
                              _ enc: inout any MTLComputeCommandEncoder) throws {
        let mode: ComputeDispatchMode = (enc.dispatchType == .concurrent) ? .concurrent : .serial
        enc.endEncoding()
        let next: (any MTLComputeCommandEncoder)?
        switch mode {
        case .serial: next = cb.makeComputeCommandEncoder()
        case .concurrent: next = cb.makeComputeCommandEncoder(dispatchType: .concurrent)
        }
        guard let next else { throw QWHalError.noDevice }
        touchKeepAlive(next)
        enc = next
    }

    /// Like `begin(_:)`, but the command buffer first waits on the GPU for `event` to reach `value`,
    /// so host-side staging can finish while later command buffers are already committed.
    public func begin(_ mode: ComputeDispatchMode, waitingFor event: any MTLSharedEvent,
                      value: UInt64) -> (any MTLCommandBuffer, any MTLComputeCommandEncoder)? {
        guard let cb = queue.makeCommandBuffer() else { return nil }
        cb.encodeWaitForEvent(event, value: value)
        let enc: (any MTLComputeCommandEncoder)?
        switch mode {
        case .serial: enc = cb.makeComputeCommandEncoder()
        case .concurrent: enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent)
        }
        guard let enc else { return nil }
        touchKeepAlive(enc)
        return (cb, enc)
    }
}

/// Diagnostic (QW_PREFILL_TIMER=1 prints it): total time spent creating compute pipelines (cold start).
public enum QWHalPSOStats {
    nonisolated(unsafe) public static var seconds: Double = 0
    static func add(_ s: Double) { seconds += s }
}
