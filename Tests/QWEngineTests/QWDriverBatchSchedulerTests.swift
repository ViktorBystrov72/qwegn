// QWDriverBatchSchedulerTests.swift — Tests for 4-slot GDN capacity, prewarm, and QWBatchScheduler.

import CryptoKit
import Foundation
import Metal
import XCTest
import QWCore
import QWHAL
import QWHead
import QWFlashNext
@testable import QWEngine

private final class MockBatchEngine: QWInferenceEngine, @unchecked Sendable {
    var capabilities: Set<QWKnobCapability> { [] }
    var prefixCache: QWPrefixCacheState { QWPrefixCacheState(isConfigured: false) }
    var hasPrefixCache: Bool { false }
    var onCacheEvent: (@Sendable (QWPrefixCacheEvent) -> Void)?
    var supportsTokenConstraints: Bool { false }
    var lastTurn: QWGenerationResult? { nil }

    private let delayUs: useconds_t
    private let lock = NSLock()
    private var calls: Int = 0

    init(delayUs: useconds_t = 0) {
        self.delayUs = delayUs
    }

    var callCount: Int { lock.withLock { calls } }

    func generate(request: QWGenerationRequest, onToken: ((Int) -> Bool)?) throws -> QWGenerationResult {
        lock.withLock { calls += 1 }
        if delayUs > 0 { usleep(delayUs) }
        var out: [Int] = []
        let base = request.prompt.first ?? 1
        for i in 0..<request.maxNew {
            let tok = base + i + 1
            if request.stopIDs.contains(tok) {
                return QWGenerationResult(tokens: out, stepMs: Array(repeating: 0.1, count: out.count),
                                          stepTokenCounts: [], stopTokenID: tok)
            }
            out.append(tok)
            if let onToken, !onToken(tok) { break }
        }
        return QWGenerationResult(tokens: out, stepMs: Array(repeating: 0.1, count: out.count), stepTokenCounts: [])
    }

    func makeConversation() -> QWConversation? { nil }
    var report: QWEngineReport {
        QWEngineReport(capacityLog: "mock", arenaBytes: 0, weightUsedBytes: 0, weightReservedBytes: 0)
    }
}

private final class FixedSetConstraint: QWTokenConstraint, @unchecked Sendable {
    let allowed: [Int]
    let allowsSpeculation: Bool
    private let lock = NSLock()
    private var acceptedTokens: [Int] = []

    init(allowed: [Int], allowsSpeculation: Bool = true) {
        self.allowed = allowed.sorted()
        self.allowsSpeculation = allowsSpeculation
    }

    func allowedTokens() throws -> [Int]? { allowed }
    func accept(_ token: Int) throws { lock.withLock { acceptedTokens.append(token) } }
    var accepted: [Int] { lock.withLock { acceptedTokens } }
}

final class QWDriverBatchSchedulerTests: XCTestCase {
    private enum PlaneSpec {
        case f32(shape: [Int], values: [Float])
        case bf16(shape: [Int], values: [Float])
        case hq4(n: Int, k: Int, seed: UInt32, scale: Float)
    }

    func testGDNSlotCountByPhysicalMemory() {
        XCTAssertEqual(QWEngineDriver.gdnSlotCount(physicalMemoryBytes: nil), 2)
        XCTAssertEqual(QWEngineDriver.gdnSlotCount(physicalMemoryBytes: 0), 2)
        XCTAssertEqual(QWEngineDriver.gdnSlotCount(physicalMemoryBytes: 16 << 30), 2)
        XCTAssertEqual(QWEngineDriver.gdnSlotCount(physicalMemoryBytes: (32 << 30) - 1), 2)
        XCTAssertEqual(QWEngineDriver.gdnSlotCount(physicalMemoryBytes: 32 << 30), 4)
        XCTAssertEqual(QWEngineDriver.gdnSlotCount(physicalMemoryBytes: 128 << 30), 4)
    }

    func testEstimateDriverContextMemoryScalesWithSlotCount() throws {
        let cfgJSON = """
        {
          "architectures": ["Qwen3_5ForCausalLM"],
          "model_type": "qwen3_5",
          "vocab_size": 256,
          "hidden_size": 256,
          "intermediate_size": 256,
          "num_hidden_layers": 2,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 64,
          "linear_num_key_heads": 2,
          "linear_num_value_heads": 4,
          "linear_key_head_dim": 128,
          "linear_value_head_dim": 128,
          "linear_conv_kernel_dim": 4,
          "layer_types": ["linear_attention", "full_attention"],
          "max_position_embeddings": 2048,
          "rms_norm_eps": 1e-6,
          "rope_theta": 1000000.0,
          "partial_rotary_factor": 1.0,
          "eos_token_id": 2
        }
        """.data(using: .utf8)!
        let desc = try QWModelDescriptor.decode(configJSON: cfgJSON)
        let geom = QWModelGeometry(descriptor: desc)
        let mem16 = QWEngineDriver.estimateDriverContextMemory(
            descriptor: desc, contextTokens: 1024, kvMode: .q4,
            mtpDrafts: 0, physicalMemoryBytes: 16 << 30)
        let mem128 = QWEngineDriver.estimateDriverContextMemory(
            descriptor: desc, contextTokens: 1024, kvMode: .q4,
            mtpDrafts: 0, physicalMemoryBytes: 128 << 30)
        let expectedSlotDelta = 2 * geom.gdnLayerCount * (
            QWSlotGeometry.recElems(geometry: geom) * 4 + QWSlotGeometry.ringElems(geometry: geom) * 2)
        XCTAssertEqual(mem128.scratchAndStateBytes - mem16.scratchAndStateBytes, expectedSlotDelta)
    }

    func testBatchSchedulerUnwrapsModulesAndRunsConcurrentPassthrough() throws {
        let mock = MockBatchEngine(delayUs: 1000)
        let scheduler = QWBatchScheduler(engine: mock)
        XCTAssertNil(QWBatchScheduler.resolveDriver(mock))
        let plan = try QWEngineBootstrap.makePlan(
            geometry: QWModelGeometry(descriptor: .qwen38_27b), packPath: "test",
            prefillChunkRows: QWPrefillLimits.chunkRows, kvStorageMode: "f16")
        let denseNoEngine = DenseModule(plan: plan)
        XCTAssertNil(QWBatchScheduler.resolveDriver(denseNoEngine))
        let unified = try QWEngine(engine: mock, descriptor: .qwen38_27b, plan: plan)
        XCTAssertNil(QWBatchScheduler.resolveDriver(unified))

        let results = LockedBox<[Int: [Int]]>([:])
        let group = DispatchGroup()
        for idx in 0..<4 {
            group.enter()
            DispatchQueue.global().async {
                defer { group.leave() }
                let req = QWGenerationRequest(prompt: [10 * (idx + 1)], maxNew: 5, sampling: .greedy)
                if let res = try? scheduler.generate(request: req) {
                    results.mutate { $0[idx] = res.tokens }
                }
            }
        }
        group.wait()
        XCTAssertEqual(mock.callCount, 4)
        XCTAssertEqual(scheduler.inFlightRequests, 0)
        for idx in 0..<4 {
            let base = 10 * (idx + 1)
            XCTAssertEqual(results.value[idx], [base + 1, base + 2, base + 3, base + 4, base + 5])
        }
    }

    func testBatchedDecodeParityAndConcurrentSchedulerOnSyntheticHybridPack() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "GPU batch decode test requires bare-metal Apple Silicon GPU"
        )
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-batch-sched-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticHybridPack(at: rootDir)

        let driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 2048, ropeRows: 2048,
            kvMode: .q4, mtpDrafts: 0, prefillChunkSize: 32)
        let expectedSlots = QWEngineDriver.gdnSlotCount()
        XCTAssertEqual(driver.maxBatchSlots, expectedSlots)
        XCTAssertEqual(driver.freeSlots.count, expectedSlots)
        XCTAssertTrue(driver.canRunBatchedDecode)

        for m in 2...4 {
            _ = try driver.hal.pipeline("qwg_head_f16_m\(m)")
            _ = try driver.hal.pipeline("qwg_head_q4g_m\(m)")
            _ = try driver.hal.pipeline("qwm_gdn_projection_m\(m)")
        }
        _ = try driver.hal.pipeline("qwg_head_q4g_rp_m4")
        _ = try driver.hal.pipeline("qwm_gdn_projection_rp_m4")

        let prompts = [
            [1, 10, 20, 30, 40],
            [2, 15, 25, 35, 45, 55],
            [5, 19, 31, 47, 61],
            [4, 18, 32, 44],
        ]
        let steps = 6
        var serialOutputs: [[Int]] = []
        for p in prompts {
            let res = try driver.generate(request: QWGenerationRequest(prompt: p, maxNew: steps, sampling: .greedy), onToken: nil)
            serialOutputs.append(res.tokens)
        }
        let repeat0 = try driver.generate(request: QWGenerationRequest(prompt: prompts[0], maxNew: steps, sampling: .greedy), onToken: nil)
        XCTAssertEqual(repeat0.tokens, serialOutputs[0])
        XCTAssertEqual(driver.freeSlots.count, expectedSlots)

        // Direct forwardBatchStep check at B=2..activeB (slot recycling, logits parity, and tokens).
        let activeB = min(expectedSlots, 4)
        for batchSize in 2...activeB {
            var sessions: [QWEngineSession] = []
            var pending: [Int] = []
            defer { for s in sessions where !s.sessionFinished { driver.finishSession(s) } }
            for b in 0..<batchSize {
                let sess = try driver.makeSession(promptTokens: prompts[b].count)
                sessions.append(sess)
                pending.append(try driver.prefill(sess, tokens: prompts[b]))
            }
            var batched = pending.map { [$0] }
            for stepIdx in 1..<steps {
                let (nextIds, stepMs, scratch) = try driver.forwardBatchStep(sessions, tokens: pending)
                XCTAssertGreaterThan(stepMs, 0)
                if stepIdx == 1 && batchSize == min(3, activeB) {
                    for b in 0..<batchSize {
                        let lb = Array(scratch.logitBuffer(b))
                        XCTAssertTrue(lb.allSatisfy(\.isFinite), "Non-finite logits on batch row \(b)")
                        let serialSess = try driver.makeSession(promptTokens: prompts[b].count)
                        let prevLP = driver.activeCollectLogprobs
                        driver.activeCollectLogprobs = true
                        defer {
                            driver.activeCollectLogprobs = prevLP
                            driver.finishSession(serialSess)
                        }
                        let f = try driver.prefill(serialSess, tokens: prompts[b])
                        _ = try driver.decodeStep(serialSess, token: f)
                        let serPtr = driver.sc.logits.region.pointer
                            .advanced(by: driver.sc.logits.offset).assumingMemoryBound(to: Float.self)
                        var maxDiff: Float = 0
                        for vIdx in 0..<256 { maxDiff = max(maxDiff, abs(lb[vIdx] - serPtr[vIdx])) }
                        XCTAssertLessThan(maxDiff, 0.05, "Batch vs serial logit drift too high on row \(b)")
                    }
                } else if stepIdx == 1 {
                    for b in 0..<batchSize {
                        XCTAssertTrue(scratch.logitBuffer(b).allSatisfy(\.isFinite))
                    }
                }
                pending = nextIds
                for b in 0..<batchSize { batched[b].append(nextIds[b]) }
            }
            for b in 0..<batchSize {
                XCTAssertEqual(batched[b].count, steps)
                XCTAssertEqual(batched[b], serialOutputs[b],
                               "Batched B=\(batchSize) stream \(b) diverged from serial decodeStep")
            }
        }
        XCTAssertEqual(driver.freeSlots.count, expectedSlots)

        // Concurrent scheduler burst via DenseModule + QWEngine unwrapping.
        let module = DenseModule(engine: driver)
        let unified = try QWEngine(engine: module, descriptor: driver.geometry.descriptor, plan: driver.plan)
        XCTAssertTrue(QWBatchScheduler.resolveDriver(unified) === driver)
        let scheduler = unified.batchScheduler

        let singleRes = try scheduler.generate(request: QWGenerationRequest(prompt: prompts[0], maxNew: steps, sampling: .greedy))
        XCTAssertEqual(singleRes.tokens, serialOutputs[0])

        let burstOutputs = LockedBox<[Int: QWGenerationResult]>([:])
        let burstGroup = DispatchGroup()
        let maxNewPerReq = [0, 2, 4, 6]
        FNBatchAdmissionTracker.beginPreparing()
        for idx in 0..<4 {
            burstGroup.enter()
            DispatchQueue.global().async {
                defer { burstGroup.leave() }
                let req = QWGenerationRequest(
                    prompt: prompts[idx],
                    maxNew: maxNewPerReq[idx],
                    sampling: .greedy,
                    collectLogprobs: idx == 1,
                    topLogprobsCount: idx == 1 ? 3 : 0
                )
                if let res = try? scheduler.generate(request: req) {
                    burstOutputs.mutate { $0[idx] = res }
                }
            }
        }
        usleep(2000)
        FNBatchAdmissionTracker.endPreparing()
        burstGroup.wait()

        XCTAssertGreaterThanOrEqual(scheduler.peakCoalescedWaveSize, 2)
        XCTAssertEqual(scheduler.inFlightRequests, 0)
        XCTAssertEqual(driver.freeSlots.count, expectedSlots)
        XCTAssertEqual(burstOutputs.value[0]?.tokens, [])
        for idx in 1..<4 {
            XCTAssertEqual(burstOutputs.value[idx]?.tokens, Array(serialOutputs[idx].prefix(maxNewPerReq[idx])))
        }
        XCTAssertEqual(burstOutputs.value[1]?.tokenLogprobs.count, maxNewPerReq[1])

        // Constrained greedy (token 0 + batched + B=1 stepSingleSlot tail) + logprobs parity vs serial.
        let serialConstraint = FixedSetConstraint(allowed: [11, 22, 33])
        let serialConstrained = try driver.generate(
            request: QWGenerationRequest(prompt: prompts[0], maxNew: 5, sampling: .greedy,
                                         collectLogprobs: true, topLogprobsCount: 3, constraint: serialConstraint),
            onToken: nil)

        let batchConstraint = FixedSetConstraint(allowed: [11, 22, 33])
        let stochasticCfg = QWSamplerConfig(
            temperature: 0.7, topK: 16, topP: 0.9,
            seed: 42, uniformVersion: .upper24, repetitionPenalty: 1.0, hasExplicitSeed: true
        )
        let mixedOutputs = LockedBox<[Int: QWGenerationResult]>([:])
        let mixedGroup = DispatchGroup()
        let cancelledCount = LockedBox<Int>(0)
        FNBatchAdmissionTracker.beginPreparing()
        for idx in 0..<3 {
            mixedGroup.enter()
            DispatchQueue.global().async {
                defer { mixedGroup.leave() }
                let req: QWGenerationRequest
                if idx == 0 {
                    // maxNew=5 while peers end after 2 tokens, forcing transitions through prefill -> B=3 -> B=1 stepSingleSlot.
                    req = QWGenerationRequest(prompt: prompts[0], maxNew: 5, sampling: .greedy,
                                              collectLogprobs: true, topLogprobsCount: 3, constraint: batchConstraint)
                } else if idx == 1 {
                    req = QWGenerationRequest(prompt: prompts[1], maxNew: 2, sampling: stochasticCfg,
                                              collectLogprobs: true, topLogprobsCount: 2)
                } else {
                    req = QWGenerationRequest(prompt: prompts[2], maxNew: 6, sampling: .greedy)
                }
                let cb: ((Int) -> Bool)? = idx == 2 ? { _ in
                    cancelledCount.mutate { $0 += 1 }
                    return cancelledCount.value < 2
                } : nil
                if let res = try? scheduler.generate(request: req, onToken: cb) {
                    mixedOutputs.mutate { $0[idx] = res }
                }
            }
        }
        usleep(2000)
        FNBatchAdmissionTracker.endPreparing()
        mixedGroup.wait()

        XCTAssertEqual(driver.freeSlots.count, expectedSlots)
        let constrainedRes = mixedOutputs.value[0]
        XCTAssertEqual(constrainedRes?.tokens.count, 5)
        XCTAssertEqual(constrainedRes?.tokens, serialConstrained.tokens)
        XCTAssertEqual(constrainedRes?.tokenLogprobs.count, 5)
        for stepIdx in 0..<5 {
            if let b = constrainedRes?.tokenLogprobs[stepIdx] {
                XCTAssertTrue(b.logprob.isFinite)
                let s = serialConstrained.tokenLogprobs[stepIdx]
                XCTAssertEqual(b.logprob, s.logprob, accuracy: 0.05, "logprob drift at constrained step \(stepIdx)")
                XCTAssertEqual(b.topCandidates.count, 3)
            } else {
                XCTFail("missing logprob at constrained step \(stepIdx)")
            }
        }
        XCTAssertTrue(constrainedRes?.tokens.allSatisfy { batchConstraint.allowed.contains($0) } == true)
        XCTAssertEqual(batchConstraint.accepted, constrainedRes?.tokens)
        XCTAssertEqual(mixedOutputs.value[1]?.tokens.count, 2)
        XCTAssertEqual(mixedOutputs.value[1]?.tokenLogprobs.count, 2)
        XCTAssertEqual(mixedOutputs.value[2]?.tokens.count, 2)

        // Ensure prompt.count >= ropeRows fails cleanly without poisoning subsequent requests.
        let oversizedGroup = DispatchGroup()
        let oversizedErr = LockedBox<Error?>(nil)
        let validAfter = LockedBox<QWGenerationResult?>(nil)
        FNBatchAdmissionTracker.beginPreparing()
        oversizedGroup.enter()
        DispatchQueue.global().async {
            defer { oversizedGroup.leave() }
            let tooLong = [Int](repeating: 1, count: driver.ropeRows)
            do {
                _ = try scheduler.generate(request: QWGenerationRequest(prompt: tooLong, maxNew: 2, sampling: .greedy))
            } catch {
                oversizedErr.mutate { $0 = error }
            }
        }
        oversizedGroup.enter()
        DispatchQueue.global().async {
            defer { oversizedGroup.leave() }
            validAfter.mutate { $0 = try? scheduler.generate(request: QWGenerationRequest(prompt: prompts[1], maxNew: 3, sampling: .greedy)) }
        }
        usleep(2000)
        FNBatchAdmissionTracker.endPreparing()
        oversizedGroup.wait()
        XCTAssertEqual(oversizedErr.value as? QWEngineDriverError, .posBeyondTable(driver.ropeRows))
        XCTAssertEqual(validAfter.value?.tokens, Array(serialOutputs[1].prefix(3)))
        XCTAssertEqual(driver.freeSlots.count, expectedSlots)
    }

    func testBatchSchedulerSpeculativeOptOutConstraintAllowsSpecAndSummary() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.isMetal4TensorCapable == true,
            "Scheduler speculative parity tests require bare-metal Apple Silicon GPU"
        )
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-batch-spec-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticHybridPack(at: rootDir, includeMTP: true)

        let driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 2048, ropeRows: 2048,
            kvMode: .q4, mtpDrafts: 2, mtpAdaptiveDepth: false, prefillChunkSize: 32)
        try driver.configurePrefixCache(config: QWCacheConfig(warmByteLimit: 8 << 20, ssdByteLimit: 0, entryLimit: 8))
        let module = DenseModule(engine: driver)
        let engine = try QWEngine(engine: module, descriptor: driver.geometry.descriptor, plan: driver.plan)
        let scheduler = engine.batchScheduler

        func runViaBatchDriver(_ req: QWGenerationRequest) throws -> QWGenerationResult {
            let job = QWBatchScheduledJob(request: req, onToken: nil)
            driver.runScheduledBatch(
                initialJobs: [job], headScratch: nil,
                pollPending: { _ in [] }, requeueDelayed: { _ in })
            return try job.wait()
        }

        let prompt = [1, 10, 20, 30, 40, 10, 20, 30, 40]
        let serial = try driver.generate(
            request: QWGenerationRequest(prompt: prompt, maxNew: 6, sampling: .greedy), onToken: nil)

        // 1. Enabled speculation in DriverBatchDecode emits speculativeSummary ("rounds=...") and bit-exact tokens.
        let specRes = try runViaBatchDriver(QWGenerationRequest(
            prompt: prompt, maxNew: 6, sampling: .greedy,
            speculative: QWSpeculativeParameters(enabled: true, mtpDrafts: 2)))
        XCTAssertEqual(specRes.tokens, serial.tokens)
        XCTAssertTrue(specRes.speculativeSummary?.hasPrefix("rounds=") == true,
                      "Expected speculative rounds summary, got \(String(describing: specRes.speculativeSummary))")

        // 2. Per-request opt-out (`speculative.enabled == false`) runs single-step with nil summary.
        let optOutRes = try runViaBatchDriver(QWGenerationRequest(
            prompt: prompt, maxNew: 5, sampling: .greedy,
            speculative: QWSpeculativeParameters(enabled: false, mtpDrafts: 0)))
        XCTAssertEqual(optOutRes.tokens, Array(serial.tokens.prefix(5)))
        XCTAssertNil(optOutRes.speculativeSummary)
        XCTAssertTrue(optOutRes.stepTokenCounts.allSatisfy { $0 == 1 })

        // 3. Constraint `allowsSpeculation == false` disables speculative rounds.
        let noSpecConstraint = FixedSetConstraint(allowed: [11, 22, 33], allowsSpeculation: false)
        let constrainedRes = try runViaBatchDriver(QWGenerationRequest(
            prompt: prompt, maxNew: 5, sampling: .greedy,
            speculative: QWSpeculativeParameters(enabled: true, mtpDrafts: 2),
            constraint: noSpecConstraint))
        XCTAssertEqual(constrainedRes.tokens.count, 5)
        XCTAssertNil(constrainedRes.speculativeSummary)
        XCTAssertTrue(constrainedRes.stepTokenCounts.allSatisfy { $0 == 1 })

        // 4. Explicit stochastic seed disables speculative rounds on both batch driver and solo scheduler path.
        let seededCfg = QWSamplerConfig(
            temperature: 0.7, topK: 16, topP: 0.9,
            seed: 42, uniformVersion: .upper24, repetitionPenalty: 1.0, hasExplicitSeed: true)
        let seededBatchRes = try runViaBatchDriver(QWGenerationRequest(
            prompt: prompt, maxNew: 5, sampling: seededCfg,
            speculative: QWSpeculativeParameters(enabled: true, mtpDrafts: 2)))
        XCTAssertEqual(seededBatchRes.tokens.count, 5)
        XCTAssertNil(seededBatchRes.speculativeSummary)
        XCTAssertTrue(seededBatchRes.stepTokenCounts.allSatisfy { $0 == 1 })

        let seededSoloRes = try scheduler.generate(request: QWGenerationRequest(
            prompt: prompt, maxNew: 5, sampling: seededCfg,
            speculative: QWSpeculativeParameters(enabled: true, mtpDrafts: 2)))
        XCTAssertEqual(seededSoloRes.tokens.count, 5)
        XCTAssertNil(seededSoloRes.speculativeSummary)
        XCTAssertTrue(seededSoloRes.stepTokenCounts.allSatisfy { $0 == 1 })
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func detVec(_ count: Int, seedInit: UInt32, amp: Float, offset: Float = 0) -> [Float] {
        var s = seedInit
        return (0..<count).map { _ in
            s = s &* 1664525 &+ 1013904223
            let u = Float(Int32(bitPattern: s)) / Float(Int32.max)
            return offset + u * amp
        }
    }

    private static func makeIncumbentHQ4Data(n: Int, k: Int, seedInit: UInt32, scale: Float) -> Data {
        let groups = k / 32
        let gBytes = 20
        var seed = seedInit
        var out = Data(count: n * groups * gBytes)
        out.withUnsafeMutableBytes { (raw: UnsafeMutableRawBufferPointer) in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            for r in 0..<n {
                for g in 0..<groups {
                    let base = (r * groups + g) * gBytes
                    let scVal = Float(Float16(scale * (1.0 + 0.125 * Float((r + g) % 5))))
                    let sc = Float16(scVal).bitPattern
                    let bi = Float16(-1031.5 * scVal).bitPattern
                    ptr[base] = UInt8(sc & 0xFF)
                    ptr[base + 1] = UInt8(sc >> 8)
                    ptr[base + 2] = UInt8(bi & 0xFF)
                    ptr[base + 3] = UInt8(bi >> 8)
                    for b in 0..<16 {
                        seed = seed &* 1664525 &+ 1013904223
                        ptr[base + 4 + b] = UInt8((seed >> 16) & 0xFF)
                    }
                }
            }
        }
        return out
    }

    private static func buildSyntheticHybridPack(at rootDir: URL, includeMTP: Bool = false) throws {
        let h = 256, v = 256, inter = 256, qHeads = 4, kvHeads = 2, headDim = 64
        let gdnKHeads = 2, gdnVHeads = 4, gdnKDim = 128, gdnVDim = 128, convK = 4
        let mtpField = includeMTP ? "\"mtp_num_hidden_layers\": 1," : ""

        let cfgJSON = """
        {
          "architectures": ["Qwen3_5ForCausalLM"],
          "model_type": "qwen3_5",
          "vocab_size": \(v),
          "hidden_size": \(h),
          "intermediate_size": \(inter),
          "num_hidden_layers": 2,
          \(mtpField)
          "num_attention_heads": \(qHeads),
          "num_key_value_heads": \(kvHeads),
          "head_dim": \(headDim),
          "linear_num_key_heads": \(gdnKHeads),
          "linear_num_value_heads": \(gdnVHeads),
          "linear_key_head_dim": \(gdnKDim),
          "linear_value_head_dim": \(gdnVDim),
          "linear_conv_kernel_dim": \(convK),
          "layer_types": ["linear_attention", "full_attention"],
          "max_position_embeddings": 2048,
          "rms_norm_eps": 1e-6,
          "rope_theta": 1000000.0,
          "partial_rotary_factor": 1.0,
          "tie_word_embeddings": false,
          "bos_token_id": 1,
          "eos_token_id": 2
        }
        """.data(using: .utf8)!

        let descriptor = try QWModelDescriptor.decode(configJSON: cfgJSON)
        let g = QWModelGeometry(descriptor: descriptor)
        let assetsDir = rootDir.appendingPathComponent("assets")
        try FileManager.default.createDirectory(at: assetsDir, withIntermediateDirectories: true)
        let cfgURL = assetsDir.appendingPathComponent("config.json")
        let tokURL = assetsDir.appendingPathComponent("tokenizer.json")
        let tplURL = assetsDir.appendingPathComponent("chat_template.jinja")
        try cfgJSON.write(to: cfgURL)
        try Data("{\"version\":\"1.0\"}".utf8).write(to: tokURL)
        try Data("bos".utf8).write(to: tplURL)

        var manifest: [String: [String: Any]] = [:]
        for u in [cfgURL, tokURL, tplURL] {
            let d = try Data(contentsOf: u)
            manifest[u.lastPathComponent] = ["sha256": sha256Hex(d), "bytes": d.count]
        }
        let mfData = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
        try mfData.write(to: assetsDir.appendingPathComponent("manifest.json"))

        var planes: [(String, PlaneSpec)] = [
            ("model.embed_tokens.weight", .bf16(shape: [v, h], values: detVec(v * h, seedInit: 11, amp: 0.25))),
            ("model.norm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 12, amp: 0.05, offset: 1.0))),
            ("lm_head.weight", .hq4(n: v, k: h, seed: 13, scale: 0.02)),
        ]

        let b0 = "model.layers.0."
        planes.append((b0 + "input_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 100, amp: 0.05, offset: 1.0))))
        planes.append((b0 + "post_attention_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 101, amp: 0.05, offset: 1.0))))
        planes.append((b0 + "linear_attn.in_proj_qkv.weight", .hq4(n: g.gdnQKVChannels, k: h, seed: 102, scale: 0.008)))
        planes.append((b0 + "linear_attn.in_proj_z.weight", .hq4(n: g.gdnZChannels, k: h, seed: 103, scale: 0.008)))
        planes.append((b0 + "linear_attn.in_proj_a.weight", .hq4(n: g.gdnVHeads, k: h, seed: 104, scale: 0.008)))
        planes.append((b0 + "linear_attn.in_proj_b.weight", .hq4(n: g.gdnVHeads, k: h, seed: 105, scale: 0.008)))
        planes.append((b0 + "linear_attn.conv1d.weight", .bf16(shape: [g.gdnQKVChannels, 1, convK],
                                                               values: detVec(g.gdnQKVChannels * convK, seedInit: 106, amp: 0.05))))
        planes.append((b0 + "linear_attn.norm.weight", .bf16(shape: [g.gdnHeadDim], values: [Float](repeating: 1.0, count: g.gdnHeadDim))))
        planes.append((b0 + "linear_attn.A_log", .f32(shape: [g.gdnVHeads], values: [Float](repeating: -1.0, count: g.gdnVHeads))))
        planes.append((b0 + "linear_attn.dt_bias", .f32(shape: [g.gdnVHeads], values: [Float](repeating: 0.0, count: g.gdnVHeads))))
        planes.append((b0 + "linear_attn.out_proj.weight", .hq4(n: h, k: g.gdnZChannels, seed: 107, scale: 0.008)))
        planes.append((b0 + "mlp.gate_proj.weight", .hq4(n: inter, k: h, seed: 108, scale: 0.008)))
        planes.append((b0 + "mlp.up_proj.weight", .hq4(n: inter, k: h, seed: 109, scale: 0.008)))
        planes.append((b0 + "mlp.down_proj.weight", .hq4(n: h, k: inter, seed: 110, scale: 0.008)))

        let b1 = "model.layers.1."
        let qRows = g.qProjOut, kvRows = g.kvHeads * g.headDim, oCols = g.qHeads * g.headDim
        planes.append((b1 + "input_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 200, amp: 0.05, offset: 1.0))))
        planes.append((b1 + "post_attention_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 201, amp: 0.05, offset: 1.0))))
        planes.append((b1 + "self_attn.q_proj.weight", .hq4(n: qRows, k: h, seed: 202, scale: 0.008)))
        planes.append((b1 + "self_attn.k_proj.weight", .hq4(n: kvRows, k: h, seed: 203, scale: 0.008)))
        planes.append((b1 + "self_attn.v_proj.weight", .hq4(n: kvRows, k: h, seed: 204, scale: 0.008)))
        planes.append((b1 + "self_attn.o_proj.weight", .hq4(n: h, k: oCols, seed: 205, scale: 0.008)))
        planes.append((b1 + "self_attn.q_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))))
        planes.append((b1 + "self_attn.k_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))))
        planes.append((b1 + "mlp.gate_proj.weight", .hq4(n: inter, k: h, seed: 206, scale: 0.008)))
        planes.append((b1 + "mlp.up_proj.weight", .hq4(n: inter, k: h, seed: 207, scale: 0.008)))
        planes.append((b1 + "mlp.down_proj.weight", .hq4(n: h, k: inter, seed: 208, scale: 0.008)))
        if includeMTP {
            planes.append(("mtp.pre_fc_norm_embedding.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
            planes.append(("mtp.pre_fc_norm_hidden.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
            planes.append(("mtp.fc.weight", .hq4(n: h, k: 2 * h, seed: 300, scale: 0.006)))
            planes.append(("mtp.norm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
            let m0 = "mtp.layers.0."
            planes.append((m0 + "input_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 301, amp: 0.05, offset: 1.0))))
            planes.append((m0 + "post_attention_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: 302, amp: 0.05, offset: 1.0))))
            planes.append((m0 + "self_attn.q_proj.weight", .hq4(n: qRows, k: h, seed: 303, scale: 0.008)))
            planes.append((m0 + "self_attn.k_proj.weight", .hq4(n: kvRows, k: h, seed: 304, scale: 0.008)))
            planes.append((m0 + "self_attn.v_proj.weight", .hq4(n: kvRows, k: h, seed: 305, scale: 0.008)))
            planes.append((m0 + "self_attn.o_proj.weight", .hq4(n: h, k: oCols, seed: 306, scale: 0.008)))
            planes.append((m0 + "self_attn.q_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))))
            planes.append((m0 + "self_attn.k_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))))
            planes.append((m0 + "mlp.gate_proj.weight", .hq4(n: inter, k: h, seed: 307, scale: 0.008)))
            planes.append((m0 + "mlp.up_proj.weight", .hq4(n: inter, k: h, seed: 308, scale: 0.008)))
            planes.append((m0 + "mlp.down_proj.weight", .hq4(n: h, k: inter, seed: 309, scale: 0.008)))
        }

        var file = Data()
        file.append(Data("QWP1".utf8))
        file.append(Data(repeating: 0, count: 8))
        var entries: [String: [String: Any]] = [:]
        func padTo(_ align: Int) {
            let rem = file.count % align
            if rem != 0 { file.append(Data(repeating: 0, count: align - rem)) }
        }
        padTo(256)

        for (name, spec) in planes {
            switch spec {
            case .f32(let shape, let vals):
                padTo(256)
                let off = file.count
                var raw = Data(capacity: vals.count * 4)
                for val in vals {
                    var le = val.bitPattern.littleEndian
                    withUnsafeBytes(of: &le) { (rawBuf: UnsafeRawBufferPointer) in raw.append(contentsOf: rawBuf) }
                }
                file.append(raw)
                padTo(256)
                entries[name] = ["shape": shape, "kind": "f32", "raw_off": off]
            case .bf16(let shape, let vals):
                padTo(256)
                let off = file.count
                var raw = Data(capacity: vals.count * 2)
                for val in vals {
                    let bits = val.bitPattern
                    let roundBias = UInt32(0x7FFF) &+ ((bits >> 16) & 1)
                    let bf = UInt16(truncatingIfNeeded: (bits &+ roundBias) >> 16).littleEndian
                    withUnsafeBytes(of: bf) { (rawBuf: UnsafeRawBufferPointer) in raw.append(contentsOf: rawBuf) }
                }
                file.append(raw)
                padTo(256)
                entries[name] = ["shape": shape, "kind": "bf16", "raw_off": off]
            case .hq4(let n, let k, let seed, let scale):
                padTo(16_384)
                let off = file.count
                file.append(makeIncumbentHQ4Data(n: n, k: k, seedInit: seed, scale: scale))
                padTo(16_384)
                entries[name] = ["shape": [n, k], "hq4_off": off]
            }
        }
        let headerObj: [String: Any] = [
            "provenance": [
                "model_id": "qwen3_5",
                "geometry_digest": descriptor.geometryDigestV2Hex,
                "source": "synthetic-batch-parity",
                "sha256_source": "synthetic",
                "quant_recorded": "q4g",
                "assets_dir": "assets",
                "assets_manifest_sha256": sha256Hex(mfData),
            ],
            "tensors": entries,
        ]
        let hdrData = try JSONSerialization.data(withJSONObject: headerObj, options: [.sortedKeys])
        let hdrLen = UInt64(hdrData.count).littleEndian
        withUnsafeBytes(of: hdrLen) { (raw: UnsafeRawBufferPointer) in file.replaceSubrange(4..<12, with: raw) }
        file.append(hdrData)
        try file.write(to: rootDir.appendingPathComponent("model.qwp"))
    }
}

private final class LockedBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: T
    init(_ value: T) { self.storage = value }
    var value: T { lock.withLock { storage } }
    func mutate(_ body: (inout T) -> Void) { lock.withLock { body(&storage) } }
}
