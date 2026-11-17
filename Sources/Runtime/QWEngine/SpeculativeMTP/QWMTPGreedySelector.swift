import Metal
import QWAlloc
import QWCore
import QWHAL

/// Checked two-stage argmax. Non-finite scores produce token -1. Draft zeros
/// compare equal; target zeros retain shipping bit-key order. ID ties are low.
struct QWMTPGreedySelector {
    let geometry: QWModelGeometry
    let vocabulary: Int
    let maxRows: Int
    let keysHi: QWTensorValue
    let keysLo: QWTensorValue
    let token: QWTensorValue

    static func allocationBytes(geometry: QWModelGeometry, vocabulary: Int, maxRows: Int = 1) -> Int {
        precondition((1...geometry.vocab).contains(vocabulary) && maxRows >= 1)
        let slots = (vocabulary + 255) / 256
        return 2 * ((maxRows * slots * 4 + 255) & ~255) + 256
    }

    init(geometry: QWModelGeometry, allocator: QWSubAllocator, vocabulary: Int, maxRows: Int = 1) throws {
        self.geometry = geometry
        guard (1...geometry.vocab).contains(vocabulary), maxRows >= 1 else {
            throw QWEngineDriverError.state("draft argmax vocabulary or row capacity is invalid")
        }
        self.vocabulary = vocabulary
        self.maxRows = maxRows
        let slots = (vocabulary + 255) / 256
        keysHi = try allocator.allocate(QWTensorShape([maxRows * slots], .i32))
        keysLo = try allocator.allocate(QWTensorShape([maxRows * slots], .i32))
        token = try allocator.allocate(QWTensorShape([1], .i32))
    }

    func encode(_ enc: any MTLComputeCommandEncoder, hal: QWHal, logits: QWTensorValue,
                ordering: QWVerifyWindow.GreedyOrdering = .numericalDraft,
                destination: QWTensorValue? = nil) throws {
        try encodeBatched(enc, hal: hal, logits: logits, rows: 1,
                          ordering: ordering, destination: destination ?? token)
    }

    func encodeBatched(_ enc: any MTLComputeCommandEncoder, hal: QWHal, logits: QWTensorValue,
                       rows: Int, ordering: QWVerifyWindow.GreedyOrdering = .shippingTarget,
                       destination: QWTensorValue) throws {
        guard (1...maxRows).contains(rows),
              logits.shape.dtype == .f32, logits.shape.elementCount == rows * vocabulary,
              logits.offset >= 0, logits.region.base >= 0,
              logits.offset <= logits.region.length,
              logits.bytes <= logits.region.length - logits.offset,
              logits.region.base <= logits.region.buffer.length,
              logits.region.length <= logits.region.buffer.length - logits.region.base,
              (logits.region.base + logits.offset) % 4 == 0,
              destination.shape.dtype == .i32, destination.shape.elementCount >= rows,
              destination.offset >= 0, destination.region.base >= 0,
              destination.offset <= destination.region.length,
              destination.bytes <= destination.region.length - destination.offset,
              destination.region.base <= destination.region.buffer.length,
              destination.region.length <= destination.region.buffer.length - destination.region.base,
              (destination.region.base + destination.offset) % 4 == 0 else {
            throw QWEngineDriverError.state("draft argmax logits geometry or extent is invalid")
        }
        for dest in [keysHi, keysLo, destination] where logits.region.buffer === dest.region.buffer {
            let a = logits.region.base + logits.offset, b = dest.region.base + dest.offset
            guard a + logits.bytes <= b || b + dest.bytes <= a else {
                throw QWEngineDriverError.state("draft argmax logits overlap reduction scratch")
            }
        }
        let slots = (vocabulary + 255) / 256
        let first = try hal.pipeline(ordering == .numericalDraft
            ? "qwg_mtp_argmax_checked" : "qwg_mtp_argmax_target_checked")
        let final = try hal.pipeline("qwg_head_argmax_fin")
        var n = UInt32(vocabulary)
        enc.setComputePipelineState(first)
        logits.bind(enc, index: 0)
        keysHi.bind(enc, index: 1)
        keysLo.bind(enc, index: 2)
        enc.setBytes(&n, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: slots, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        n = UInt32(slots)
        enc.setComputePipelineState(final)
        keysHi.bind(enc, index: 0)
        keysLo.bind(enc, index: 1)
        destination.bind(enc, index: 2)
        enc.setBytes(&n, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    /// Call only after successful command-buffer completion.
    func completedToken() throws -> Int {
        let id = Int(token.region.pointer.advanced(by: token.offset).assumingMemoryBound(to: Int32.self).pointee)
        guard (0..<vocabulary).contains(id) else {
            throw QWEngineDriverError.state("draft argmax rejected non-finite logits or invalid token \(id)")
        }
        return id
    }
}
