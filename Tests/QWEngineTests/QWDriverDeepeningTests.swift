// QWDriverDeepeningTests.swift — Unit tests for deepened Driver components and data-driven registry.
import XCTest
import Metal
@testable import QWEngine
import QWCore
import QWMTP

final class QWDriverDeepeningTests: XCTestCase {
    private static func makeDescriptor(
        family: QWModelDescriptor.Family,
        modelType: String,
        numExperts: Int? = nil,
        expertsPerTok: Int? = nil,
        residentExperts: Int? = nil,
        mlaUseNoPE: Bool = false,
        indexerHeadDim: Int? = nil,
        indexerCompressRatio: Int? = nil,
        hybridMLAConfig: QWModelDescriptor.HybridMLAConfig? = nil
    ) -> QWModelDescriptor {
        let isMoE = (numExperts ?? 0) > 0
        return QWModelDescriptor(
            family: family, modelType: modelType, capabilities: [],
            hidden: 2048, vocab: 32768, denseIntermediate: isMoE ? nil : 6144,
            numLayers: 4,
            layerTypes: ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            fullAttentionInterval: 4, rmsEps: 1e-6, maxPositionEmbeddings: 131072,
            qHeads: 16, kvHeads: 4, headDim: 128, attnOutputGate: true,
            partialRotaryFactor: 0.25, ropeTheta: 1e6,
            indexerHeads: indexerHeadDim != nil ? 32 : nil,
            indexerKvHeads: indexerHeadDim != nil ? 1 : nil,
            indexerHeadDim: indexerHeadDim,
            indexerBudget: indexerHeadDim != nil ? 2048 : nil,
            indexerCompressRatio: indexerCompressRatio,
            numExperts: numExperts, expertsPerTok: expertsPerTok,
            moeIntermediate: isMoE ? 512 : nil,
            sharedExpertIntermediate: isMoE ? 512 : nil,
            expertGateType: isMoE ? "sigmoid" : nil,
            hcCount: nil, hcLowrank: nil,
            gdnQKHeads: 16, gdnVHeads: 32, gdnHeadDim: 64,
            gdnQKVChannels: 6144, gdnZChannels: 2048, gdnConvKernel: 4,
            pleLayerIds: nil, pleEmbedDim: nil, pleConvKernel: nil,
            ngramSize: nil, headsPerNgram: nil, ngramVocabSizeBase: nil,
            makeNgramVocabDivisibleBy: nil, splitNgramParts: nil,
            mtpLayers: 1, mtpUseDedicatedEmbeddings: false, eosTokenId: 2,
            residentExpertsPerLayer: residentExperts,
            mlaUseNoPE: mlaUseNoPE, kvLoraRank: mlaUseNoPE ? 512 : nil, qLoraRank: mlaUseNoPE ? 1024 : nil,
            hybridMLAConfig: hybridMLAConfig
        )
    }

    func testModuleRegistryAndHybridTraitsAreDrivenByFamilyRegistry() throws {
        XCTAssertEqual(Set(QWModuleRegistry.supportedFamilies), Set(QWModelDescriptor.Family.allCases))
        for family in QWModuleRegistry.supportedFamilies {
            XCTAssertNotNil(QWModuleRegistry.familySpec(for: family), family.rawValue)
            XCTAssertEqual(QWModuleRegistry.capabilities(for: family),
                           QWModuleRegistry.moduleType(for: family)?.capabilities ?? [])
            let expectedModule: any QWInferenceModule.Type = DenseModule.self
            XCTAssertEqual(ObjectIdentifier(try XCTUnwrap(QWModuleRegistry.moduleType(for: family))),
                           ObjectIdentifier(expectedModule), family.rawValue)
        }
        for modelType in ["qwen3_5_moe", "glm5_next", "ernie5_5_moe", "maxwell_moe"] {
            let fam = try XCTUnwrap(QWModuleRegistry.family(forPackModelId: modelType))
            let mod = try XCTUnwrap(QWModuleRegistry.moduleType(for: fam))
            XCTAssertEqual(ObjectIdentifier(mod), ObjectIdentifier(DenseModule.self), modelType)
            let moeDesc = Self.makeDescriptor(
                family: fam, modelType: modelType,
                numExperts: 32, expertsPerTok: 4, residentExperts: 32)
            XCTAssertTrue(QWModelCapabilities.resolve(descriptor: moeDesc).contains(.expertStreaming), modelType)
        }
        let loopDesc = Self.makeDescriptor(
            family: .senseNovaLoop, modelType: "sensenova_loop",
            numExperts: nil, expertsPerTok: nil, residentExperts: nil)
        XCTAssertFalse(QWModelCapabilities.resolve(descriptor: loopDesc).contains(.expertStreaming))
        XCTAssertFalse(DenseModule.capabilities.contains(.expertStreaming))
        XCTAssertEqual(QWModuleRegistry.family(forPackModelId: "qwen3_moe"), .qwen3MoE)
        XCTAssertEqual(QWModuleRegistry.family(forPackModelId: "sensenova_loop"), .senseNovaLoop)
    }

    func testScratchArenaBuilderConsistencyAcrossPrefillAndMoEPlans() {
        let moeDesc = Self.makeDescriptor(
            family: .qwen38MoE, modelType: "qwen3_5_moe",
            numExperts: 64, expertsPerTok: 8, residentExperts: 64)
        let moeGeo = QWModelGeometry(descriptor: moeDesc)

        let aliased = QWScratchArenaBuilder.prefillBytesPerToken(
            geometry: moeGeo, maxPrefillChunk: 512, useGroupedNAX: true, aliasBranches: true)
        let unaliased = QWScratchArenaBuilder.prefillBytesPerToken(
            geometry: moeGeo, maxPrefillChunk: 512, useGroupedNAX: true, aliasBranches: false)
        let breakdown = QWScratchArenaBuilder.prefillTransientBreakdown(
            geometry: moeGeo, maxPrefillChunk: 512, useGroupedNAX: true)
        XCTAssertEqual(aliased, breakdown.persistent + max(breakdown.attnTransient, breakdown.mlpTransient))
        XCTAssertEqual(unaliased, breakdown.persistent + breakdown.attnTransient + breakdown.mlpTransient)

        let decodePlan = QWScratchArenaBuilder.MoEScratchPlan(
            rows: 1, geometry: moeGeo, useGroupedNAX: true)
        XCTAssertFalse(decodePlan.isPrefill)
        XCTAssertEqual(decodePlan.allocationBytes,
                       QWEngineDriver.moeScratchAllocationBytes(rows: 1, geometry: moeGeo, useGroupedNAX: true))

        let prefillPlan = QWScratchArenaBuilder.MoEScratchPlan(
            rows: 256, geometry: moeGeo, useGroupedNAX: true)
        XCTAssertEqual(prefillPlan.allocationBytes,
                       QWEngineDriver.moeScratchAllocationBytes(rows: 256, geometry: moeGeo, useGroupedNAX: true))

        let nopeDesc = Self.makeDescriptor(
            family: .glm5Next, modelType: "glm5_next",
            numExperts: 32, expertsPerTok: 4, residentExperts: 32,
            mlaUseNoPE: true, indexerHeadDim: 128, indexerCompressRatio: 4)
        let nopeGeo = QWModelGeometry(descriptor: nopeDesc)
        XCTAssertEqual(QWScratchArenaBuilder.mlaPerTokenBytes(geometry: nopeGeo, activeAttnLayers: 2),
                       2 * ((128 * 2) / 4))

        let hybridCfg = QWModelDescriptor.HybridMLAConfig(
            qHeads: 16, qLoraRank: 1024, kvLoraRank: 512,
            qkNopeHeadDim: 128, qkRopeHeadDim: 64, vHeadDim: 128, slidingWindow: 128)
        let dsaDesc = Self.makeDescriptor(
            family: .ernie55MoE, modelType: "ernie5_5_moe",
            numExperts: 32, expertsPerTok: 4, residentExperts: 32,
            mlaUseNoPE: false, indexerHeadDim: 128, indexerCompressRatio: 128,
            hybridMLAConfig: hybridCfg)
        let dsaGeo = QWModelGeometry(descriptor: dsaDesc)
        XCTAssertEqual(QWScratchArenaBuilder.mlaPerTokenBytes(geometry: dsaGeo, activeAttnLayers: 2),
                       2 * ((128 + 64) * 2))
    }

    func testAdmissionManagerAndSpeculativeCoordinatorLifecycle() throws {
        let admission = QWSessionAdmissionManager()
        let executed = try admission.withFlight {
            XCTAssertThrowsError(try admission.withFlight { 1 }) { err in
                guard case QWEngineDriverError.busy = err else {
                    XCTFail("expected busy, got \(err)")
                    return
                }
            }
            return 42
        }
        XCTAssertEqual(executed, 42)
        XCTAssertTrue(admission.flight.try())
        admission.flight.unlock()

        admission.recordCommandBufferError(for: nil)
        admission.recordCommandBufferError(for: nil)
        XCTAssertEqual(admission.commandBufferErrors, 2)

        let permit = try admission.acquireRequest()
        try admission.requestGate.validate(permit)
        admission.releaseRequest(permit)

        let coord = QWSpeculativeSessionCoordinator()
        coord.addMTPStats(proposals: 5, accepted: 3)
        XCTAssertEqual(coord.mtpDraftProposals, 5)
        XCTAssertEqual(coord.mtpDraftAccepted, 3)

        coord.beginRoundPreparation(collectAdaptiveEntropy: true)
        XCTAssertEqual(coord.pendingMTPAdaptiveEntropy, [])
        coord.lastRoundUsedPLD = true
        coord.lastDraftedPairs = 4
        coord.verifiedInputs = [1, 2, 3]
        coord.finishCommit(profile: nil)
        XCTAssertFalse(coord.lastRoundUsedPLD)
        XCTAssertEqual(coord.lastDraftedPairs, 0)
        XCTAssertTrue(coord.verifiedInputs.isEmpty)
        XCTAssertNil(coord.pendingMTPAdaptiveEntropy)

        let frame = QWCommandFrame()
        frame.decodeTel.coldStaged = 9
        frame.resetColdTelemetry()
        XCTAssertEqual(frame.decodeTel.coldStaged, 0)
        frame.abortIncomplete(admission: admission, session: nil, clearPrefillProfiler: true)
        XCTAssertNil(frame.activeCB)
        XCTAssertNil(frame.activeEnc)
        XCTAssertNil(frame.stepCBs)
    }
}
