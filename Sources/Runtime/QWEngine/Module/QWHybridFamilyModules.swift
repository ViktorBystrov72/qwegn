// QWHybridFamilyModules.swift — Dedicated QWInferenceModule subclasses for
// distinct hybrid dense/MoE/KDA/MLA/HCA families served by QWEngineDriver.

import QWCore

/// The Qwen3.8 sparse-MoE hybrid GDN family (`qwen3_5_moe`, e.g., Qwen3.6-35B-A3B).
public final class Qwen38MoEModule: DenseModule, @unchecked Sendable {
    public override class var family: QWModelDescriptor.Family { .qwen38MoE }
    public override class var capabilities: Set<QWKnobCapability> {
        DenseModule.capabilities.union([.expertStreaming])
    }
}

/// The looped hybrid GDN family (`sensenova_loop`).
public final class SenseNovaLoopModule: DenseModule, @unchecked Sendable {
    public override class var family: QWModelDescriptor.Family { .senseNovaLoop }
}

/// The GLM-5.3 KDA + NoPE MLA + Key-Pool MoE family (`glm5_next`).
public final class Glm5NextModule: DenseModule, @unchecked Sendable {
    public override class var family: QWModelDescriptor.Family { .glm5Next }
    public override class var capabilities: Set<QWKnobCapability> {
        DenseModule.capabilities.union([.expertStreaming])
    }
}

/// The ERNIE-5.5 HCA + Hybrid MLA + Latent-MoE + mHC family (`ernie5_5_moe`).
public final class Ernie55MoEModule: DenseModule, @unchecked Sendable {
    public override class var family: QWModelDescriptor.Family { .ernie55MoE }
    public override class var capabilities: Set<QWKnobCapability> {
        DenseModule.capabilities.union([.expertStreaming])
    }
}

/// The Maxwell-MoE Block-Sparse GQA + mHC + dedicated mHC-MTP family (`maxwell_moe`).
public final class MaxwellMoEModule: DenseModule, @unchecked Sendable {
    public override class var family: QWModelDescriptor.Family { .maxwellMoE }
    public override class var capabilities: Set<QWKnobCapability> {
        DenseModule.capabilities.union([.expertStreaming])
    }
}
