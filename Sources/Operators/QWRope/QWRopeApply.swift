import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc

/// C07 apply path: the ONE fused norm+rope launch (`qwg_rope_norm_qk`,
/// Sources/QWMetal/qwrope.metal). Table rows live in a device buffer uploaded
/// once per QWRopeTables build; the engine keeps it weight-class (immutable,
public final class QWRopeApply: @unchecked Sendable {
    public let tables: QWRopeTables?
    public let geometry: QWModelGeometry
    /// 16 B scratch (never written) bound when normOut == nil — Metal requires
    /// every declared buffer index bound; the kernel null-checks the pointer.
    let nullSlab: MTLBuffer
    /// device copy of tables.cosSin ([rows][rotaryDims] f32).
    public let tableTensor: QWTensorValue
    private let pso: any MTLComputePipelineState
    private let psoExtract: any MTLComputePipelineState
    /// The general kernels: a head narrower than 256 halves leaves the upper
    /// lanes idle, and a model with no q/k norm skips the normalisation.
    /// The 27B's kernels are frozen — their arithmetic is the golden's.
    private let psoGeneral: any MTLComputePipelineState
    private let psoExtractGeneral: any MTLComputePipelineState

    enum Buffer: Int { case qIn = 0, kIn, qOut, kOut, wq, wk, table, normOut, headU4, eps, qkvBias }

    /// Floats per table row: the model's rotary dims.
    private var tableWidth: Int { geometry.rotaryDims }

    /// True when the frozen 27B kernels cannot serve this call: a 256-half
    /// head, q/k norm, and hardcoded `simd_shuffle_xor(lo, 4)` (`rotaryDims == 64`).
    private func needsGeneralKernel(hasNorm: Bool, hasBias: Bool = false) -> Bool {
        hasBias || geometry.headDim != 256 || !hasNorm || geometry.rotaryDims != 64
    }

    private static func validateGeometry(_ g: QWModelGeometry) throws {
        guard g.headDim > 0, g.headDim <= 256, g.headDim % 8 == 0,
              g.rotaryDims > 0, g.rotaryDims <= g.headDim, g.rotaryDims % 16 == 0 else {
            throw QWRopeError.badShape(
                "unsupported geometry headDim=\(g.headDim) rotaryDims=\(g.rotaryDims)"
            )
        }
    }

    /// Test path: bind a PRE-ALLOCATED table tensor (the caller's own arena —
    /// re-bumping activationsRegion() would restart the cursor at 0 and stomp
    /// earlier tensors; measured bug). Caller copyIns the fixture bytes.
    init(tableTensor t: QWTensorValue, hal: QWHal, nullSlab slab: MTLBuffer,
         geometry: QWModelGeometry) throws {
        if !geometry.usesLatentOrCompressedAttention {
            try Self.validateGeometry(geometry)
        }
        self.tables = nil
        self.geometry = geometry
        self.tableTensor = t
        self.nullSlab = slab
        self.pso = try hal.pipeline("qwg_rope_norm_qk")
        self.psoExtract = try hal.pipeline("qwg_extract_rope_norm_qk")
        self.psoGeneral = try hal.pipeline("qwg_rope_qk_general")
        self.psoExtractGeneral = try hal.pipeline("qwg_extract_rope_qk_general")
    }

    public init(tables: QWRopeTables, hal: QWHal, weights: QWSubAllocator, arena: QWArena,
                geometry: QWModelGeometry? = nil) throws {
        let g = geometry ?? tables.geometry
        if !g.usesLatentOrCompressedAttention {
            try Self.validateGeometry(g)
        }
        self.tables = tables
        self.geometry = g
        let t = try weights.allocate(QWTensorShape([tables.rows, tables.width], .f32))
        tables.cosSin.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in arena.copyIn(t, from: Data(raw)) }
        self.tableTensor = t
        guard let slab = hal.device.makeBuffer(length: 16, options: .storageModeShared)
        else { throw QWRopeError.badShape("null slab alloc failed") }
        self.nullSlab = slab
        self.pso = try hal.pipeline("qwg_rope_norm_qk")
        self.psoExtract = try hal.pipeline("qwg_extract_rope_norm_qk")
        self.psoGeneral = try hal.pipeline("qwg_rope_qk_general")
        self.psoExtractGeneral = try hal.pipeline("qwg_extract_rope_qk_general")
    }

    /// Encode the fused q‖k norm+rope (f16). q [S·qProjOut] (Q‖gate
    /// interleaved when the model has a gate), k [S·kvWidth], weights wq
    /// [headDim] and wk [kvHeads·headDim] — both optional; normOut/qkvBias optional.
    public func encodeQK(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                         q: QWTensorValue, k: QWTensorValue,
                         wq: QWTensorValue?, wk: QWTensorValue?,
                         qOut: QWTensorValue, kOut: QWTensorValue,
                         normOut: QWTensorValue? = nil,
                         qkvBias: QWTensorValue? = nil,
                         positions: [Int]) throws {
        try Self.validateGeometry(geometry)
        let S = positions.count
        let nrows = tables?.rows ?? (tableTensor.shape.dims.first ?? 0)
        if let p = positions.max(), let q0 = positions.min(),
           (0..<nrows).contains(p), (0..<nrows).contains(q0),
           p - q0 + 1 == S, zip(positions, positions.dropFirst()).allSatisfy({ $1 == $0 + 1 }) {
            try encodeQKContiguous(enc, q: q, k: k, wq: wq, wk: wk,
                                   qOut: qOut, kOut: kOut, normOut: normOut, qkvBias: qkvBias,
                                   rowBase: q0, rows: S)
        } else if positions.count == S, Set(positions).count == S, S <= 4096 {
            // sparse positions (spec 07 allows; phase-0 engine path is
            // contiguous-only). Gather device-side? Not yet: fail loud rather
            // than silently mis-rope. The engine's scheduler keeps positions
            // contiguous; a violation here is a bug in the caller, not data.
            throw QWRopeError.badShape("non-contiguous positions (\(positions.prefix(4))…) — phase-0 contiguous-only")
        } else {
            throw QWRopeError.posBeyondTable(positions.max() ?? -1, nrows)
        }
    }

    /// Fast path for contiguous text positions [positionStart ..< positionStart+count]
    /// without host-side position array allocation.
    public func encodeQK(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                         q: QWTensorValue, k: QWTensorValue,
                         wq: QWTensorValue?, wk: QWTensorValue?,
                         qOut: QWTensorValue, kOut: QWTensorValue,
                         normOut: QWTensorValue? = nil,
                         qkvBias: QWTensorValue? = nil,
                         positionStart: Int, count: Int) throws {
        try Self.validateGeometry(geometry)
        let nrows = tables?.rows ?? (tableTensor.shape.dims.first ?? 0)
        guard count > 0 else { throw QWRopeError.badShape("q/k/weights/positions mismatch: S=\(count)") }
        guard positionStart >= 0, positionStart + count <= nrows else {
            throw QWRopeError.posBeyondTable(positionStart + count - 1, nrows)
        }
        try encodeQKContiguous(enc, q: q, k: k, wq: wq, wk: wk,
                               qOut: qOut, kOut: kOut, normOut: normOut, qkvBias: qkvBias,
                               rowBase: positionStart, rows: count)
    }

    private func encodeQKContiguous(_ enc: any MTLComputeCommandEncoder,
                                    q: QWTensorValue, k: QWTensorValue,
                                    wq: QWTensorValue?, wk: QWTensorValue?,
                                    qOut: QWTensorValue, kOut: QWTensorValue,
                                    normOut: QWTensorValue?,
                                    qkvBias: QWTensorValue?,
                                    rowBase: Int, rows: Int) throws {
        let qSlot = geometry.qProjOut                             // Q‖gate or Q only
        let kAll = geometry.kvHeads * geometry.headDim
        let S = rows
        guard q.shape.dtype == .f16, k.shape.dtype == .f16,
              q.shape.elementCount == S * qSlot, k.shape.elementCount == S * kAll,
              qOut.shape == q.shape, kOut.shape == k.shape,
              wq.map({ $0.shape.elementCount == geometry.headDim }) ?? true,
              wk.map({ $0.shape.elementCount == geometry.kvHeads * geometry.headDim }) ?? true,
              (wq == nil) == (wk == nil),
              normOut.map({ $0.shape.dtype == .f16 && $0.shape.elementCount == S * (qSlot + kAll) }) ?? true,
              qkvBias.map({
                  $0.shape.dtype == .f16 && $0.shape.elementCount == geometry.attnTotalN
                      && ($0.region.base + $0.offset) % 16 == 0
              }) ?? true
        else { throw QWRopeError.badShape("q/k/weights/positions mismatch: S=\(S)") }
        let general = needsGeneralKernel(hasNorm: wq != nil, hasBias: qkvBias != nil)
        enc.setComputePipelineState(general ? psoGeneral : pso)
        q.bind(enc, index: Buffer.qIn.rawValue)
        k.bind(enc, index: Buffer.kIn.rawValue)
        qOut.bind(enc, index: Buffer.qOut.rawValue)
        kOut.bind(enc, index: Buffer.kOut.rawValue)
        if let w = wq { w.bind(enc, index: Buffer.wq.rawValue) }
        else { enc.setBuffer(nullSlab, offset: 0, index: Buffer.wq.rawValue) }
        if let w = wk { w.bind(enc, index: Buffer.wk.rawValue) }
        else { enc.setBuffer(nullSlab, offset: 0, index: Buffer.wk.rawValue) }
        if let n = normOut {
            n.bind(enc, index: Buffer.normOut.rawValue)
        } else {
            enc.setBuffer(nullSlab, offset: 0, index: Buffer.normOut.rawValue)
        }
        if let b = qkvBias {
            b.bind(enc, index: Buffer.qkvBias.rawValue)
        } else {
            enc.setBuffer(nullSlab, offset: 0, index: Buffer.qkvBias.rawValue)
        }
        struct RopeFlags { var headU4, writeNorm, kNormBaseU4, normOn: UInt32 }   // metal uint4
        // staged-norm layout law (mirrored in tests): q heads at [0, S·qProjOut/8),
        // k heads at the tail [S·qProjOut/8, S·(qProjOut+kvWidth)/8). uint4 units.
        let flagsW: UInt32 = (wq != nil ? 1 : 0) | (qkvBias != nil ? 2 : 0)
        var flags = RopeFlags(headU4: UInt32(geometry.headDim / 8),      // 32
                              writeNorm: normOut == nil ? 0 : 1,         // ptr never null → gate
                              kNormBaseU4: UInt32(S * qSlot / 8),
                              normOn: flagsW)
        var eps = geometry.rmsEps
        withUnsafeBytes(of: &flags) { (raw: UnsafeRawBufferPointer) in enc.setBytes(raw.baseAddress!, length: 16, index: Buffer.headU4.rawValue) }
        enc.setBytes(&eps, length: 4, index: Buffer.eps.rawValue)
        // fold rowBase into the table bind: +rowBase·width f32 = rowBase·width·4 B
        enc.setBuffer(tableTensor.region.buffer,
                      offset: tableTensor.region.base + tableTensor.offset + rowBase * tableWidth * 4,
                      index: Buffer.table.rawValue)
        enc.dispatchThreadgroups(MTLSize(width: S, height: geometry.qHeads + geometry.kvHeads, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    /// Fused QKV extraction + RoPE + RMSNorm + V copy for Qwen3.8 Attention (EXP-8 landing #2).
    /// Reads directly from packed `attnY` [S, attnTotalN] f16 without separate extract passes.
    /// Emits `qOut` [S, qProjOut], `kOut` [S, kvWidth], and `vOut` [S, kvWidth].
    public func encodeExtractQK(_ enc: any MTLComputeCommandEncoder, hal: QWHal,
                                attnY: QWTensorValue,
                                qOut: QWTensorValue, kOut: QWTensorValue, vOut: QWTensorValue,
                                wq: QWTensorValue?, wk: QWTensorValue?,
                                qkvBias: QWTensorValue? = nil,
                                positionStart: Int, count: Int,
                                inRowCols: Int? = nil) throws {
        try Self.validateGeometry(geometry)
        let nrows = tables?.rows ?? (tableTensor.shape.dims.first ?? 0)
        guard count > 0 else { throw QWRopeError.badShape("count mismatch: S=\(count)") }
        guard positionStart >= 0, positionStart + count <= nrows else {
            throw QWRopeError.posBeyondTable(positionStart + count - 1, nrows)
        }
        let S = count
        let totalN = geometry.attnTotalN
        let rowCols = inRowCols ?? totalN
        let qSlot = geometry.qProjOut
        let kAll = geometry.kvHeads * geometry.headDim
        let vAll = geometry.kvHeads * geometry.headDim
        guard attnY.shape.dtype == .f16, qOut.shape.dtype == .f16,
              kOut.shape.dtype == .f16, vOut.shape.dtype == .f16,
              attnY.shape.elementCount == S * rowCols,
              rowCols >= totalN, rowCols % 8 == 0,
              qOut.shape.elementCount == S * qSlot,
              kOut.shape.elementCount == S * kAll,
              vOut.shape.elementCount == S * vAll,
              wq.map({ $0.shape.elementCount == geometry.headDim }) ?? true,
              wk.map({ $0.shape.elementCount == geometry.kvHeads * geometry.headDim }) ?? true,
              (wq == nil) == (wk == nil),
              qkvBias.map({
                  $0.shape.dtype == .f16 && $0.shape.elementCount == totalN
                      && ($0.region.base + $0.offset) % 16 == 0
              }) ?? true
        else { throw QWRopeError.badShape("extractQK tensor shapes mismatch: S=\(S)") }

        let general = needsGeneralKernel(hasNorm: wq != nil, hasBias: qkvBias != nil) || rowCols != totalN
        enc.setComputePipelineState(general ? psoExtractGeneral : psoExtract)
        attnY.bind(enc, index: 0)
        qOut.bind(enc, index: 1)
        kOut.bind(enc, index: 2)
        vOut.bind(enc, index: 3)
        if let w = wq { w.bind(enc, index: 4) }
        else { enc.setBuffer(nullSlab, offset: 0, index: 4) }
        if let w = wk { w.bind(enc, index: 5) }
        else { enc.setBuffer(nullSlab, offset: 0, index: 5) }
        enc.setBuffer(tableTensor.region.buffer,
                      offset: tableTensor.region.base + tableTensor.offset + positionStart * tableWidth * 4,
                      index: 6)
        var eps = geometry.rmsEps
        enc.setBytes(&eps, length: 4, index: 7)
        struct RopeFlags { var headU4, inRowU4, kNormBaseU4, normOn: UInt32 }
        let flagsW: UInt32 = (wq != nil ? 1 : 0) | (qkvBias != nil ? 2 : 0)
        var flags = RopeFlags(headU4: UInt32(geometry.headDim / 8),
                              inRowU4: rowCols == totalN ? 0 : UInt32(rowCols / 8),
                              kNormBaseU4: UInt32(S * (geometry.qHeads * geometry.headDim) / 8),
                              normOn: flagsW)
        withUnsafeBytes(of: &flags) { (raw: UnsafeRawBufferPointer) in enc.setBytes(raw.baseAddress!, length: 16, index: 8) }
        if let b = qkvBias {
            b.bind(enc, index: 9)
        } else {
            enc.setBuffer(nullSlab, offset: 0, index: 9)
        }
        enc.dispatchThreadgroups(MTLSize(width: S, height: geometry.qHeads + geometry.kvHeads, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }
}

