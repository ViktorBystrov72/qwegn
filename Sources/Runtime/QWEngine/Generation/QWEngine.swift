// QWEngine — C21 leg 1: the orchestrator's host-side laws (spec-21).
//
// Zero kernels, ever — the composition root is pure encoding order, graph
// submission and scheduling policy ("if a clever scheduling idea lives
// here it must be behind a measured flag"). This file is the SKELETON the
// doors hang on: the 3:1 cadence interleaver (C01's classifier is the only
// architecture it knows), the chunked-prefill plan (C13's ADR'd L=64), the
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWMM

// MARK: - Config (the whole contract — the CLI maps flags 1:1, spec-22) ─────

public enum GDNPrefillMode: String, Sendable, CaseIterable, Equatable {
    case chunked
    case naive
    /// EXP-9 sub-item 1 (OPT-IN, NOT the shipping default): register-
    /// resident single-launch recurrence (`qwg_gdn_prefill_regres`). Local
    /// MLX-derived work-distribution idea (see docs/ACCELERATOR_LEDGER.md
    /// EXP-9); parity-gated against `.chunked`/`.naive`, SMALL-T measured
    /// win only (component bench, T<=1024) — not certified for T>1024.
    case regres
}

/// Typed per-session round record (Phase 4, section 5A.3).
/// Tracks the exact state boundaries across speculative rounds and non-speculative steps:
/// - pendingToken: token emitted to output/user, but whose trunk forward pass has not yet executed (tok0).
public struct QWRoundRecord: Sendable, Equatable, Codable {
    public var pendingToken: Int?
    public var consumedCount: Int
    public var outputCount: Int
    public var trunkHiddenSeed: [Float]?
    public var mtpPhysicalRows: Int
    public var absolutePosition: Int
    public var samplerDrawIndex: UInt64
    public var mropeDelta: Int

    enum CodingKeys: String, CodingKey {
        case pendingToken, consumedCount, outputCount, trunkHiddenSeed
        case mtpPhysicalRows, absolutePosition, samplerDrawIndex, mropeDelta
    }

    public init(
        pendingToken: Int? = nil,
        consumedCount: Int = 0,
        outputCount: Int = 0,
        trunkHiddenSeed: [Float]? = nil,
        mtpPhysicalRows: Int = 0,
        absolutePosition: Int = 0,
        samplerDrawIndex: UInt64 = 0,
        mropeDelta: Int = 0
    ) {
        self.pendingToken = pendingToken
        self.consumedCount = consumedCount
        self.outputCount = outputCount
        self.trunkHiddenSeed = trunkHiddenSeed
        self.mtpPhysicalRows = mtpPhysicalRows
        self.absolutePosition = absolutePosition
        self.samplerDrawIndex = samplerDrawIndex
        self.mropeDelta = mropeDelta
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        pendingToken = try c.decodeIfPresent(Int.self, forKey: .pendingToken)
        consumedCount = try c.decode(Int.self, forKey: .consumedCount)
        outputCount = try c.decode(Int.self, forKey: .outputCount)
        trunkHiddenSeed = try c.decodeIfPresent([Float].self, forKey: .trunkHiddenSeed)
        mtpPhysicalRows = try c.decode(Int.self, forKey: .mtpPhysicalRows)
        absolutePosition = try c.decode(Int.self, forKey: .absolutePosition)
        samplerDrawIndex = try c.decode(UInt64.self, forKey: .samplerDrawIndex)
        mropeDelta = try c.decodeIfPresent(Int.self, forKey: .mropeDelta) ?? 0
    }
}

// MARK: - L5: CPU-tail ledger (encode time = the ≤300 µs/step gate feed) ───

public final class QWEncodeLedger: @unchecked Sendable {
    public let budgetMicros: Double
    private let lock = NSLock()
    private var samples: [Double] = []                  // µs per step

    public init(budgetMicros: Double = 300) { self.budgetMicros = budgetMicros }

    public func record(micros: Double) { lock.lock(); samples.append(micros); lock.unlock() }
    public func reset() { lock.lock(); samples.removeAll(); lock.unlock() }
    public var count: Int { lock.lock(); defer { lock.unlock() }; return samples.count }
    /// The raw sample list (the m1 harness computes p50/p95/mean over the
    /// FULL ledger — summary()'s p50/max alone can't feed the ≤1% gate).
    public func values() -> [Double] { lock.lock(); defer { lock.unlock() }; return samples }
    public func summary() -> (n: Int, p50: Double, max: Double, overBudget: Int) {
        lock.lock(); defer { lock.unlock() }
        guard !samples.isEmpty else { return (0, 0, 0, 0) }
        let s = samples.sorted()
        let over = samples.filter { $0 > budgetMicros }.count
        return (s.count, s[s.count / 2], s.last!, over)
    }
}

// MARK: - Decode GPU-step ledger (submit-schedule telemetry, observe-only) ─

public final class QWDecodeGPULedger: @unchecked Sendable {
    private let lock = NSLock()
    private var samples: [Double] = [] // µs per decode command buffer on GPU

    public init() {}

    public func record(micros: Double) { lock.lock(); samples.append(micros); lock.unlock() }
    public func reset() { lock.lock(); samples.removeAll(); lock.unlock() }
    public var count: Int { lock.lock(); defer { lock.unlock() }; return samples.count }
    public func values() -> [Double] { lock.lock(); defer { lock.unlock() }; return samples }
}

public struct QWPrefillChunkTiming: Sendable {
    public let encodeMicros: Double
    public let gpuMicros: Double?
}

public final class QWPrefillTimingLedger: @unchecked Sendable {
    private let lock = NSLock()
    private var samples: [QWPrefillChunkTiming] = []
    private var enabled = false

    public init() {}

    public var isEnabled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return enabled
    }

    public func setEnabled(_ enabled: Bool) {
        lock.lock()
        self.enabled = enabled
        lock.unlock()
    }

    public func record(encodeMicros: Double, gpuMicros: Double?) {
        lock.lock()
        if enabled {
            samples.append(QWPrefillChunkTiming(encodeMicros: encodeMicros, gpuMicros: gpuMicros))
        }
        lock.unlock()
    }

    public func reset() {
        lock.lock()
        samples.removeAll()
        lock.unlock()
    }

    public func values() -> [QWPrefillChunkTiming] {
        lock.lock()
        defer { lock.unlock() }
        return samples
    }
}
