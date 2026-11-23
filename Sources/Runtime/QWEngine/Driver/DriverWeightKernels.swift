// DriverWeightKernels.swift — the weight kernels a container chooses, and
// the head's own load. The dense path's weight GEMVs are the same kernels
// in both containers; only the bank they read differs. specs/04-qwquant.md.
import Foundation
import Metal
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWHead
import QWLoad
import QWQuant

extension QWEngineDriver {

    /// The decode GEMV for a container: the incumbent's over `Block_HQ4_G32`,
    /// the family's twin over the family's blocks.
    static func decodeKernelName(format: QWWeightFormat) throws -> String {
        guard format.usesDenseAQKernels else { return "qwg_gemv_q4g_dx_h16_t4" }
        guard let name = QWGemmAQ.decodeKernelName(bits: format.bits, group: format.groupSize,
                                                   isSHVQ: format.denseIsSHVQ) else {
            throw QWEngineDriverError.state("no decode kernel for \(format.bits)-bit group "
                                            + "\(format.groupSize)")
        }
        return name
    }

    /// The MLP gate/up GEMV for a container.
    static func siluPairKernelName(format: QWWeightFormat) throws -> String {
        guard format.usesDenseAQKernels else { return "qwg_gemv_q4g_silu_pair_dx_h16" }
        guard let name = QWGemmAQ.siluPairKernelName(bits: format.bits, group: format.groupSize,
                                                     isSHVQ: format.denseIsSHVQ) else {
            throw QWEngineDriverError.state("no MLP kernel for \(format.bits)-bit group "
                                            + "\(format.groupSize)")
        }
        return name
    }

    /// The prefill GEMM for a container.
    static func gemmKernelName(format: QWWeightFormat) throws -> String {
        guard format.usesDenseAQKernels else { return "qwg_gemm_q4g_h16" }
        guard let name = QWGemmAQ.gemmKernelName(bits: format.bits, group: format.groupSize,
                                                 isSHVQ: format.denseIsSHVQ) else {
            throw QWEngineDriverError.state("no prefill kernel for \(format.bits)-bit group "
                                            + "\(format.groupSize)")
        }
        return name
    }

    /// The fused gate_up + SwiGLU + FWHT-32 prefill GEMM for a container.
    static func swigluGemmKernelName(format: QWWeightFormat) throws -> String {
        guard format.usesDenseAQKernels else { return "qwg_gemm_q4g_h16_swiglu" }
        guard let name = QWGemmAQ.swigluGemmKernelName(bits: format.bits, group: format.groupSize,
                                                       isSHVQ: format.denseIsSHVQ) else {
            throw QWEngineDriverError.state("no fused SwiGLU kernel for \(format.bits)-bit group "
                                            + "\(format.groupSize)")
        }
        return name
    }

    /// The GDN in-projection decode kernel for a container: the incumbent's
    /// over `Block_HQ4_G32`, the family's twin over the family's blocks.
    static func gdnInProjDecodeKernelName(format: QWWeightFormat) throws -> String {
        guard format.usesDenseAQKernels else { return "qwg_gemv_q4g_dx_h16_gdn_inproj_t4" }
        guard let name = QWGemmAQ.gdnInProjDecodeKernelName(bits: format.bits,
                                                            group: format.groupSize,
                                                            isSHVQ: format.denseIsSHVQ) else {
            throw QWEngineDriverError.state("no GDN decode kernel for \(format.bits)-bit group "
                                            + "\(format.groupSize)")
        }
        return name
    }

    /// The GDN in-projection prefill kernel for a container.
    static func gdnInProjPrefillKernelName(format: QWWeightFormat) throws -> String {
        guard format.usesDenseAQKernels else { return "qwg_gemm_q4g_h16_gdn_inproj" }
        guard let name = QWGemmAQ.gdnInProjPrefillKernelName(bits: format.bits,
                                                            group: format.groupSize,
                                                            isSHVQ: format.denseIsSHVQ) else {
            throw QWEngineDriverError.state("no GDN prefill kernel for \(format.bits)-bit group "
                                            + "\(format.groupSize)")
        }
        return name
    }

    struct HeadWeightSpec: Sendable, Equatable {
        let isQ4g: Bool
        let format: QWWeightFormat?
    }

    /// Resolves head precision and format against tied embeddings or explicit `lm_head.weight`.
    static func resolveHeadSpec(
        st: QWSafeTensorsContainer?,
        pack: QWQuant.Pack?,
        trunkFormat: QWWeightFormat,
        geometry: QWModelGeometry
    ) -> HeadWeightSpec {
        let tie = geometry.descriptor.tieWordEmbeddings
        let rawExists: (String) -> Bool = { name in
            if let pack { return (try? pack.entry(name)) != nil }
            if let st { return st.tensor(name) != nil }
            return false
        }
        let resolvedName = QWCheckpointTensorLoader.resolveName(
            "lm_head.weight",
            defaultPrefix: P,
            tieWordEmbeddings: tie,
            rawExists: rawExists
        )
        if let pack {
            let isQ4g = (try? pack.entry(resolvedName).quantized) ?? false
            return HeadWeightSpec(isQ4g: isQ4g, format: nil)
        }
        guard let st else {
            return HeadWeightSpec(isQ4g: false, format: nil)
        }
        guard trunkFormat.isNativeAffine else {
            return HeadWeightSpec(isQ4g: true, format: .hq4G32)
        }
        if resolvedName.hasSuffix(".weight") {
            let stem = String(resolvedName.dropLast(".weight".count))
            if st.tensor(stem + ".scales") != nil,
               let layout = try? QWAQRepacker.layout(container: st, name: resolvedName,
                                                     rows: geometry.vocab, cols: geometry.hidden),
               QWAQBlock.isSupported(bits: layout.bits, group: layout.group) {
                return HeadWeightSpec(
                    isQ4g: true,
                    format: .nativeAffine(bits: layout.bits, groupSize: layout.group)
                )
            }
        }
        return HeadWeightSpec(isQ4g: false, format: nil)
    }

    /// The head's bank: the family's own container, or the incumbent's.
    static func loadHead(loader: inout WeightLoader, headIsQ4g: Bool,
                         headFormat: QWWeightFormat? = nil,
                         geometry: QWModelGeometry) throws -> QWHeadStorage {
        guard headIsQ4g else { return .f16(try loader.convertBF16("lm_head.weight")) }
        let view = try loader.q4("lm_head.weight", n: geometry.vocab, k: geometry.hidden,
                                 formatOverride: headFormat)
        guard let weight = QWHeadWeight(blocks: view.blocks, n: view.n, k: view.k,
                                        bits: view.bits, groupSize: view.groupSize,
                                        isRotated: view.isRotated,
                                        isIncumbentContainer: view.isIncumbentContainer,
                                        isSHVQ: view.isSHVQ) else {
            throw QWEngineDriverError.weightMissing("lm_head.weight", "QWHeadWeight layout invalid")
        }
        return .q4g(weight)
    }
}
