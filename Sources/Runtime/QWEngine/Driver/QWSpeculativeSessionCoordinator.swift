// QWSpeculativeSessionCoordinator.swift — Speculative verification round/lease state coordinator.
import Foundation
import Synchronization
import QWCore
import QWMTP

/// Coordinates in-flight speculative verification session state, prepared round
/// leases, stochastic witnesses, stage timing profiles, and draft counters.
final class QWSpeculativeSessionCoordinator: @unchecked Sendable {
    var verifiedSession: QWEngineSession?
    var verifiedInputs: [Int] = []
    var preparedGreedyLease: QWGreedyRoundLease?
    var preparedStochasticWitness: QWStochasticRoundWitness?
    var pendingMTPProfile: (draft: QWMTPStageTiming, verification: QWMTPStageTiming)?
    var pendingMTPAdaptiveEntropy: [Double]?
    var lastMTPRoundProfile: QWMTPRoundProfile?
    var lastRoundUsedPLD: Bool = false
    var lastDraftedPairs: Int = 0
    var verificationGpuMsByStage: [String: Double]?

    private let _mtpDraftProposals = Atomic<Int>(0)
    private let _mtpDraftAccepted = Atomic<Int>(0)

    var mtpDraftProposals: Int {
        get { _mtpDraftProposals.load(ordering: .acquiring) }
        set { _mtpDraftProposals.store(newValue, ordering: .releasing) }
    }

    var mtpDraftAccepted: Int {
        get { _mtpDraftAccepted.load(ordering: .acquiring) }
        set { _mtpDraftAccepted.store(newValue, ordering: .releasing) }
    }

    func beginRoundPreparation(collectAdaptiveEntropy: Bool) {
        lastMTPRoundProfile = nil
        pendingMTPProfile = nil
        pendingMTPAdaptiveEntropy = collectAdaptiveEntropy ? [] : nil
        preparedStochasticWitness = nil
        lastRoundUsedPLD = false
        lastDraftedPairs = 0
    }

    func clearActiveVerification(matching session: QWEngineSession? = nil) {
        if session == nil || verifiedSession === session {
            verifiedSession = nil
            verifiedInputs.removeAll(keepingCapacity: true)
            preparedGreedyLease = nil
            preparedStochasticWitness = nil
            pendingMTPProfile = nil
            pendingMTPAdaptiveEntropy = nil
        }
        lastRoundUsedPLD = false
        lastDraftedPairs = 0
    }

    func finishCommit(profile: QWMTPRoundProfile?) {
        clearActiveVerification()
        lastMTPRoundProfile = profile
    }

    func recordPendingProfile(
        profileStart: UInt64?,
        verifyStart: UInt64?,
        draftAccumulator: QWMTPStageAccumulator?,
        verifyAccumulator: QWMTPStageAccumulator?,
        isWeightFree: Bool
    ) throws {
        guard let profileStart, let verifyStart,
              let draftAccumulator, let verifyAccumulator else { return }
        let rawDraftHostMs = Double(verifyStart - profileStart) / 1_000_000
        let rawVerifyHostMs = Double(DispatchTime.now().uptimeNanoseconds - verifyStart) / 1_000_000
        let draftTiming: QWMTPStageTiming
        if isWeightFree && draftAccumulator.commandBuffers == 0 {
            draftTiming = try QWMTPStageTiming.hostOnly(hostMs: rawDraftHostMs)
        } else {
            draftTiming = try QWMTPStageTiming(
                hostMs: max(draftAccumulator.gpuMs, rawDraftHostMs),
                gpuMs: draftAccumulator.gpuMs, commandBuffers: draftAccumulator.commandBuffers)
        }
        pendingMTPProfile = (
            draftTiming,
            try QWMTPStageTiming(hostMs: max(verifyAccumulator.gpuMs, rawVerifyHostMs),
                                 gpuMs: verifyAccumulator.gpuMs, commandBuffers: verifyAccumulator.commandBuffers))
    }

    func buildCommittedProfile(
        profileStart: UInt64?,
        resolutionGPU: Double?,
        retainedRows: Int,
        drafts: [Int],
        acceptedDrafts: Int,
        backend: String
    ) throws -> QWMTPRoundProfile? {
        guard let profileStart, let resolutionGPU else { return nil }
        guard let pendingMTPProfile else {
            throw QWEngineDriverError.state("MTP profile is missing its prepared stages")
        }
        let rawResHostMs = Double(DispatchTime.now().uptimeNanoseconds - profileStart) / 1_000_000
        return QWMTPRoundProfile(
            drafts: drafts.count, retainedRows: retainedRows,
            draft: pendingMTPProfile.draft, verification: pendingMTPProfile.verification,
            resolution: try QWMTPStageTiming(
                hostMs: max(resolutionGPU, rawResHostMs),
                gpuMs: resolutionGPU, commandBuffers: 1),
            draftTokens: drafts, acceptedDrafts: acceptedDrafts,
            backend: backend,
            verificationGpuMsByStage: verificationGpuMsByStage,
            adaptiveEntropyByDraft: pendingMTPAdaptiveEntropy)
    }

    func addMTPStats(proposals: Int = 0, accepted: Int = 0) {
        if proposals != 0 { _mtpDraftProposals.add(proposals, ordering: .relaxed) }
        if accepted != 0 { _mtpDraftAccepted.add(accepted, ordering: .relaxed) }
    }
}
