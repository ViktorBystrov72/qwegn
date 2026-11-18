// EngineSession.swift — per-sequence session state (pre-carved slots,
// zero steady-state allocations; split from QWEngineDriver.swift).
import Foundation
import Darwin
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWNorm
import QWRope
import QWKV
import QWAttn
import QWGDN
import QWMLP
import QWHead
import QWMTP
import QWQuant

/// THE PER-LAYER STATE LAW (C23 fix — the single-slot version conflated
public final class QWEngineSession: @unchecked Sendable {
    public static let repetitionPenaltyWindow = 64
    public internal(set) var position: Int = 0        // next token's absolute position
    public internal(set) var promptTokenCount: Int = 0
    public internal(set) var generated: Int = 0
    public internal(set) var roundRecord: QWRoundRecord
    public var tokenHistory: [Int] = []
    public let tokenRecycler = QWTokenRecycler()
    let kv: QWKVHandle
    let mtpKV: QWKVHandle?
    var rec: [QWTensorValue]                  // [gdnLayer] → [48·128·128] f32 (zeroed)
    let ring: [QWTensorValue]                 // [gdnLayer] → [3·10240] f16 (zeroed)
    let recSlab: QWTensorValue                // whole slab (memset at admission)
    let ringSlab: QWTensorValue
    let trunkHidden: QWTensorValue?
    let dflashSlot: QWDFlashSessionSlot?
    public internal(set) var dflashCacheOffset: Int = 0
    var dflashPolicy: QWDFlashAdaptivePolicy
    var dflashRoundStartNs: UInt64?
    var trunkHiddenValid = false
    /// finishSession idempotence witness (static-analysis H1): the driver
    /// checks-and-sets it under the flight lock, so a double finish can
    /// never push one GDN slot into the free list twice.
    var sessionFinished = false
    /// MLA indexer pools captured while another session owned the operator.
    var mlaIndexerStash: (data: Data, position: Int)?
    var mlaLiveStash: (data: Data, trunkPosition: Int, mtpPosition: Int)?
    var batchedDecodeTouched = false
    var stateFailed = false
    var verificationPending = false
    var lastCommitCB: (any MTLCommandBuffer)?
    var stochastic: QWStochasticSession?
    var attnKVViews: [(k: QWKVLayerView, v: QWKVLayerView)] = []
    var mtpKVViews: [(k: QWKVLayerView, v: QWKVLayerView)] = []
    public var mtpPhysicalRows: Int { roundRecord.mtpPhysicalRows }

    init(kv: QWKVHandle, mtpKV: QWKVHandle? = nil, rec: [QWTensorValue],
         ring: [QWTensorValue], recSlab: QWTensorValue, ringSlab: QWTensorValue,
         trunkHidden: QWTensorValue? = nil, dflashSlot: QWDFlashSessionSlot? = nil,
         isSparseMoE: Bool = false) {
        self.kv = kv; self.mtpKV = mtpKV; self.rec = rec; self.ring = ring
        self.recSlab = recSlab; self.ringSlab = ringSlab
        self.trunkHidden = trunkHidden
        self.dflashSlot = dflashSlot
        self.dflashPolicy = QWDFlashAdaptivePolicy(isSparseMoE: isSparseMoE)
        self.roundRecord = QWRoundRecord()
    }

    public func snapshot(using driver: QWEngineDriver) throws -> QWEngineSessionSnapshot {
        try driver.snapshotSession(self)
    }

    public func rollback(using driver: QWEngineDriver, to snapshot: QWEngineSessionSnapshot) throws {
        try driver.rollbackSession(self, to: snapshot)
    }

    /// Identifies active repetition-loop tokens in a trailing generated window.
    public static func loopRepetitionTokens(in window: [Int]) -> [Int] {
        QWSampler.loopRepetitionTokens(in: window)
    }

    /// Returns loop-targeted tokens from the trailing `repetitionPenaltyWindow` current-turn generated tokens.
    public func activeRepetitionHistory(extraTokens: [Int] = []) -> [Int] {
        let baseStart = min(max(0, promptTokenCount), tokenHistory.count)
        let generatedCount = (tokenHistory.count - baseStart) + extraTokens.count
        guard generatedCount >= 2 else { return [] }
        let dropCount = max(0, generatedCount - Self.repetitionPenaltyWindow)
        let histStart = min(tokenHistory.count, baseStart + dropCount)
        let extraStart = max(0, dropCount - (tokenHistory.count - baseStart))
        var out: [Int] = []
        out.reserveCapacity(min(generatedCount, Self.repetitionPenaltyWindow))
        if histStart < tokenHistory.count {
            out.append(contentsOf: tokenHistory[histStart..<tokenHistory.count])
        }
        if extraStart < extraTokens.count {
            out.append(contentsOf: extraTokens[extraStart..<extraTokens.count])
        }
        return Self.loopRepetitionTokens(in: out)
    }
}
