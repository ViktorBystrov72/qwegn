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
              Self.hasValidWordExtent(logits),
              destination.shape.dtype == .i32, destination.shape.elementCount >= rows,
              Self.hasValidWordExtent(destination) else {
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

    private static func hasValidWordExtent(_ tensor: QWTensorValue) -> Bool {
        tensor.offset >= 0 && tensor.region.base >= 0
            && tensor.offset <= tensor.region.length
            && tensor.bytes <= tensor.region.length - tensor.offset
            && tensor.region.base <= tensor.region.buffer.length
            && tensor.region.length <= tensor.region.buffer.length - tensor.region.base
            && (tensor.region.base + tensor.offset) % 4 == 0
    }

    /// Call only after successful command-buffer completion.
    func completedToken() throws -> Int {
        let id = Int(token.region.pointer.advanced(by: token.offset).assumingMemoryBound(to: Int32.self).pointee)
        do {
            return try QWSpeculativeTokenResolver.validateToken(id, vocabulary: vocabulary)
        } catch {
            throw QWEngineDriverError.state("draft argmax rejected non-finite logits or invalid token \(id)")
        }
    }

    /// Reads and validates `count` completed target tokens written by `encodeBatched`.
    func completedTokens(from destination: QWTensorValue, count: Int) throws -> [Int] {
        guard count > 0, count <= destination.shape.elementCount else {
            throw QWEngineDriverError.state("GPU target active row count is invalid")
        }
        let ids = destination.region.pointer.advanced(by: destination.offset).assumingMemoryBound(to: Int32.self)
        var out: [Int] = []
        out.reserveCapacity(count)
        for row in 0..<count {
            let id = Int(ids[row])
            do {
                out.append(try QWSpeculativeTokenResolver.validateToken(id, vocabulary: vocabulary))
            } catch {
                throw QWEngineDriverError.state("GPU target row \(row) rejected non-finite logits or invalid token \(id)")
            }
        }
        return out
    }
}
