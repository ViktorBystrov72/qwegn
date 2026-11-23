// QWQSALayerKernels.swift — HyperStream PLE-enabled QSA, GDN, KDA, and full-attention layer kernels.
// Resolves `ple: true` keys in `QWLayerKernelRegistry` while `QWHyperStreamRuntime` drives execution.

import Foundation
import QWCore

/// Shared base for PLE-enabled HyperStream layer kernels resolved by `QWLayerKernelRegistry`.
class HyperStreamPLELayerKernelBase: @unchecked Sendable {
    func encodeDecodeHyperStream(_ key: QWEnginePlan.LayerKernelKey) throws -> Never {
        throw QWEngineDriverError.state(
            "PLE HyperStream kernel (\(key.attention.rawValue)/\(key.mlp.rawValue)) requires QWHyperStreamRuntime")
    }

    func encodePrefillHyperStream(_ key: QWEnginePlan.LayerKernelKey) throws -> Never {
        throw QWEngineDriverError.state(
            "PLE HyperStream kernel (\(key.attention.rawValue)/\(key.mlp.rawValue)) requires QWHyperStreamRuntime")
    }
}

final class GDNPLELayerKernel: HyperStreamPLELayerKernelBase, QWLayerKernel, @unchecked Sendable {
    static let key = QWEnginePlan.LayerKernelKey(attention: .gdn, mlp: .dense, ple: true)
    static func make() -> any QWLayerKernel { GDNPLELayerKernel() }
    func encodeDecode(_ context: QWLayerDecodeContext) throws { try encodeDecodeHyperStream(Self.key) }
    func encodePrefillChunk(_ context: QWLayerPrefillContext) throws { try encodePrefillHyperStream(Self.key) }
}

final class GDNPLEMoELayerKernel: HyperStreamPLELayerKernelBase, QWLayerKernel, @unchecked Sendable {
    static let key = QWEnginePlan.LayerKernelKey(attention: .gdn, mlp: .moe, ple: true)
    static func make() -> any QWLayerKernel { GDNPLEMoELayerKernel() }
    func encodeDecode(_ context: QWLayerDecodeContext) throws { try encodeDecodeHyperStream(Self.key) }
    func encodePrefillChunk(_ context: QWLayerPrefillContext) throws { try encodePrefillHyperStream(Self.key) }
}

final class KDAPLELayerKernel: HyperStreamPLELayerKernelBase, QWLayerKernel, @unchecked Sendable {
    static let key = QWEnginePlan.LayerKernelKey(attention: .kda, mlp: .dense, ple: true)
    static func make() -> any QWLayerKernel { KDAPLELayerKernel() }
    func encodeDecode(_ context: QWLayerDecodeContext) throws { try encodeDecodeHyperStream(Self.key) }
    func encodePrefillChunk(_ context: QWLayerPrefillContext) throws { try encodePrefillHyperStream(Self.key) }
}

final class KDAPLEMoELayerKernel: HyperStreamPLELayerKernelBase, QWLayerKernel, @unchecked Sendable {
    static let key = QWEnginePlan.LayerKernelKey(attention: .kda, mlp: .moe, ple: true)
    static func make() -> any QWLayerKernel { KDAPLEMoELayerKernel() }
    func encodeDecode(_ context: QWLayerDecodeContext) throws { try encodeDecodeHyperStream(Self.key) }
    func encodePrefillChunk(_ context: QWLayerPrefillContext) throws { try encodePrefillHyperStream(Self.key) }
}

final class QSAPLELayerKernel: HyperStreamPLELayerKernelBase, QWLayerKernel, @unchecked Sendable {
    static let key = QWEnginePlan.LayerKernelKey(attention: .qsaSparse, mlp: .dense, ple: true)
    static func make() -> any QWLayerKernel { QSAPLELayerKernel() }
    func encodeDecode(_ context: QWLayerDecodeContext) throws { try encodeDecodeHyperStream(Self.key) }
    func encodePrefillChunk(_ context: QWLayerPrefillContext) throws { try encodePrefillHyperStream(Self.key) }
}

final class QSAPLEMoELayerKernel: HyperStreamPLELayerKernelBase, QWLayerKernel, @unchecked Sendable {
    static let key = QWEnginePlan.LayerKernelKey(attention: .qsaSparse, mlp: .moe, ple: true)
    static func make() -> any QWLayerKernel { QSAPLEMoELayerKernel() }
    func encodeDecode(_ context: QWLayerDecodeContext) throws { try encodeDecodeHyperStream(Self.key) }
    func encodePrefillChunk(_ context: QWLayerPrefillContext) throws { try encodePrefillHyperStream(Self.key) }
}

final class FullAttentionPLELayerKernel: HyperStreamPLELayerKernelBase, QWLayerKernel, @unchecked Sendable {
    static let key = QWEnginePlan.LayerKernelKey(attention: .fullAttention, mlp: .dense, ple: true)
    static func make() -> any QWLayerKernel { FullAttentionPLELayerKernel() }
    func encodeDecode(_ context: QWLayerDecodeContext) throws { try encodeDecodeHyperStream(Self.key) }
    func encodePrefillChunk(_ context: QWLayerPrefillContext) throws { try encodePrefillHyperStream(Self.key) }
}

final class MoEPLELayerKernel: HyperStreamPLELayerKernelBase, QWLayerKernel, @unchecked Sendable {
    static let key = QWEnginePlan.LayerKernelKey(attention: .fullAttention, mlp: .moe, ple: true)
    static func make() -> any QWLayerKernel { MoEPLELayerKernel() }
    func encodeDecode(_ context: QWLayerDecodeContext) throws { try encodeDecodeHyperStream(Self.key) }
    func encodePrefillChunk(_ context: QWLayerPrefillContext) throws { try encodePrefillHyperStream(Self.key) }
}
