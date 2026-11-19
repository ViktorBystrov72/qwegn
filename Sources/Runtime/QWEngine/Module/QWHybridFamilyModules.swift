// QWHybridFamilyModules.swift — Dedicated QWInferenceModule subclasses for
// distinct hybrid dense/MoE/KDA/MLA/HCA families served by QWEngineDriver.

import QWCore

/// Shared base for hybrid sparse-MoE families; capabilities follow `QWModelFamilyRegistry`.
open class HybridMoEModule: DenseModule, @unchecked Sendable {
    open override class var capabilities: Set<QWKnobCapability> {
        let supportsStreaming = QWModelFamilyRegistry.shared.spec(for: family)?.supportsExpertStreaming ?? true
        return supportsStreaming
            ? DenseModule.capabilities.union([.expertStreaming])
            : DenseModule.capabilities
    }
}

/// The Qwen3.8 sparse-MoE hybrid GDN family (`qwen3_5_moe`, e.g., Qwen3.6-35B-A3B).
public final class Qwen38MoEModule: HybridMoEModule, @unchecked Sendable {
    public override class var family: QWModelDescriptor.Family { .qwen38MoE }
}

/// The looped hybrid GDN family (`sensenova_loop`).
public final class SenseNovaLoopModule: DenseModule, @unchecked Sendable {
    public override class var family: QWModelDescriptor.Family { .senseNovaLoop }
}

/// The GLM-5.3 KDA + NoPE MLA + Key-Pool MoE family (`glm5_next`).
public final class Glm5NextModule: HybridMoEModule, @unchecked Sendable {
    public override class var family: QWModelDescriptor.Family { .glm5Next }
}

/// The ERNIE-5.5 HCA + Hybrid MLA + Latent-MoE + mHC family (`ernie5_5_moe`).
public final class Ernie55MoEModule: HybridMoEModule, @unchecked Sendable {
    public override class var family: QWModelDescriptor.Family { .ernie55MoE }
}

/// The Maxwell-MoE Block-Sparse GQA + mHC + dedicated mHC-MTP family (`maxwell_moe`).
public final class MaxwellMoEModule: HybridMoEModule, @unchecked Sendable {
    public override class var family: QWModelDescriptor.Family { .maxwellMoE }
}
