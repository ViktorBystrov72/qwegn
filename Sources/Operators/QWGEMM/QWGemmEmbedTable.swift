// QWGemmEmbedTable.swift — GPU gather dispatch extension for QWDeviceEmbedTable.
import Metal
import QWAlloc
import QWHAL
import QWLoad

extension QWDeviceEmbedTable {
    /// Gathers the rows of `token` into `out` on the device.
    public func gather(enc: any MTLComputeCommandEncoder, hal: QWHal,
                       token: QWTensorValue, out: QWTensorValue, rows: Int, hc: Int) throws {
        switch storage {
        case .bf16(let table):
            try QWGemm.encodeEmbedBf16F32(enc, hal: hal, table: table, token: token, out: out,
                                          hidden: hidden, hc: hc, rows: rows)
        case .nativeAffine(let blocks, let bits, let group):
            try QWGemmAQ.encodeEmbedAQ(enc, hal: hal, blocks: blocks, token: token, out: out,
                                       hidden: hidden, hc: hc, bits: bits, group: group, rows: rows)
        }
    }
}
