// QWMTPNextNParityTests.swift — Bit-exact parity for NextN hybrid PLD + commit pair seeding.

import CryptoKit
import Foundation
import XCTest
import QWCore
import QWFlashNext
import QWHAL
@testable import QWEngine

final class QWMTPNextNParityTests: XCTestCase {
    override func setUpWithError() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["GITHUB_ACTIONS"] != "true"
                && QWDeviceCaps.current?.familyApple9 == true,
            "MTP NextN GPU parity tests require bare-metal Apple Silicon GPU"
        )
    }

    private enum PlaneSpec {
        case bf16(shape: [Int], values: [Float])
        case hq4(n: Int, k: Int, seed: UInt32, scale: Float)
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
        let group = 32
        let groups = k / group
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

    private static func buildSyntheticNextNPack(at rootDir: URL) throws {
        let h = 256
        let v = 256
        let inter = 256
        let qHeads = 4
        let kvHeads = 2
        let headDim = 64

        let cfgJSON = """
        {
          "architectures": ["Qwen3_5ForCausalLM"],
          "model_type": "qwen3_5",
          "vocab_size": \(v),
          "hidden_size": \(h),
          "intermediate_size": \(inter),
          "num_hidden_layers": 1,
          "mtp_num_hidden_layers": 1,
          "num_attention_heads": \(qHeads),
          "num_key_value_heads": \(kvHeads),
          "head_dim": \(headDim),
          "max_position_embeddings": 2048,
          "rms_norm_eps": 1e-6,
          "rope_theta": 1000000.0,
          "partial_rotary_factor": 1.0,
          "layer_types": ["full_attention"],
          "tie_word_embeddings": false,
          "bos_token_id": 1,
          "eos_token_id": 2
        }
        """.data(using: .utf8)!

        let descriptor = try QWModelDescriptor.decode(configJSON: cfgJSON)
        let g = QWModelGeometry(descriptor: descriptor)
        let qRows = g.qProjOut
        let kvRows = g.kvHeads * g.headDim
        let oCols = g.qHeads * g.headDim

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

        func appendAttnAndMLP(_ blk: String, seedBase: UInt32) {
            planes.append((blk + "input_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: seedBase, amp: 0.05, offset: 1.0))))
            planes.append((blk + "post_attention_layernorm.weight", .bf16(shape: [h], values: detVec(h, seedInit: seedBase + 1, amp: 0.05, offset: 1.0))))
            planes.append((blk + "self_attn.q_proj.weight", .hq4(n: qRows, k: h, seed: seedBase + 2, scale: 0.008)))
            planes.append((blk + "self_attn.k_proj.weight", .hq4(n: kvRows, k: h, seed: seedBase + 3, scale: 0.008)))
            planes.append((blk + "self_attn.v_proj.weight", .hq4(n: kvRows, k: h, seed: seedBase + 4, scale: 0.008)))
            planes.append((blk + "self_attn.o_proj.weight", .hq4(n: h, k: oCols, seed: seedBase + 5, scale: 0.008)))
            planes.append((blk + "self_attn.q_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))))
            planes.append((blk + "self_attn.k_norm.weight", .bf16(shape: [headDim], values: [Float](repeating: 1.0, count: headDim))))
            planes.append((blk + "mlp.gate_proj.weight", .hq4(n: inter, k: h, seed: seedBase + 6, scale: 0.008)))
            planes.append((blk + "mlp.up_proj.weight", .hq4(n: inter, k: h, seed: seedBase + 7, scale: 0.008)))
            planes.append((blk + "mlp.down_proj.weight", .hq4(n: h, k: inter, seed: seedBase + 8, scale: 0.008)))
        }

        appendAttnAndMLP("model.layers.0.", seedBase: 100)
        planes.append(("mtp.pre_fc_norm_embedding.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
        planes.append(("mtp.pre_fc_norm_hidden.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
        planes.append(("mtp.fc.weight", .hq4(n: h, k: 2 * h, seed: 200, scale: 0.006)))
        planes.append(("mtp.norm.weight", .bf16(shape: [h], values: [Float](repeating: 1.0, count: h))))
        appendAttnAndMLP("mtp.layers.0.", seedBase: 300)

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
                "source": "synthetic-nextn-parity",
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

    func testNextNHybridNgramAndCommitSeedingBitExactParity() throws {
        try XCTSkipUnless(QWDeviceCaps.current?.isMetal4TensorCapable == true,
                          "NextN driver parity requires bare-metal Apple Silicon GPU")
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-nextn-parity-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticNextNPack(at: rootDir)

        let prompt = [1, 10, 20, 30, 40, 50, 60, 10, 20, 30, 40]
        let maxNew = 12

        let baseDriver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 512, ropeRows: 2048,
            kvMode: .q4, mtpDrafts: 0, prefillChunkSize: 32)
        let baseSession = try baseDriver.makeSession(promptTokens: prompt.count)
        let baseline = try baseDriver.generate(session: baseSession, prompt: prompt, maxNew: maxNew)
        baseDriver.finishSession(baseSession)
        XCTAssertEqual(baseline.count, maxNew)

        for strategy in ["mtp", "hybrid", "ngram"] {
            let driver = try QWEngineDriver(
                packPath: rootDir.path, maxTokens: 512, ropeRows: 2048,
                kvMode: .q4, mtpDrafts: 3, mtpAdaptiveDepth: false,
                mtpDraftStrategy: strategy, mtpQuadraticSeeding: true, prefillChunkSize: 32)
            let session = try driver.makeSession(promptTokens: prompt.count)
            let tokens = try driver.generateExperimentalGreedy(session: session, prompt: prompt, maxNew: maxNew)
            XCTAssertEqual(session.roundRecord.mtpPhysicalRows, session.position - 1,
                           "MTP physical pair rows must track trunk position - 1 for strategy \(strategy)")
            driver.finishSession(session)
            XCTAssertEqual(tokens, baseline,
                           "NextN strategy \(strategy) must be bit-exact with greedy baseline")
        }
    }

    func testBatchedDecodeParityWithMTPDrafts2AndPostBatchScheduler() throws {
        try XCTSkipUnless(QWDeviceCaps.current?.isMetal4TensorCapable == true,
                          "Batched MTP parity requires bare-metal Apple Silicon GPU")
        let rootDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qw-nextn-batch-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: rootDir) }
        try Self.buildSyntheticNextNPack(at: rootDir)

        // mtpDrafts=2 allocates a 3-row mtpBatchedMLP (< 4), exercising lazy 4-row sharedBatchMLP upgrade.
        let driver = try QWEngineDriver(
            packPath: rootDir.path, maxTokens: 2048, ropeRows: 2048,
            kvMode: .q4, mtpDrafts: 2, mtpAdaptiveDepth: false, prefillChunkSize: 32)
        XCTAssertTrue(driver.canRunBatchedDecode)
        XCTAssertEqual(driver.mtpBatchedMLP?.rows, 3)

        let prompts = [
            [1, 10, 20, 30, 40],
            [2, 15, 25, 35, 45, 55],
            [5, 19, 31, 47, 61],
            [4, 18, 32, 44],
        ]
        let steps = 5
        var serialOutputs: [[Int]] = []
        for p in prompts {
            let res = try driver.generate(request: QWGenerationRequest(prompt: p, maxNew: steps, sampling: .greedy), onToken: nil)
            serialOutputs.append(res.tokens)
        }

        let activeB = min(driver.freeSlots.count, 4)
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
            for _ in 1..<steps {
                let (nextIds, _, scratch) = try driver.forwardBatchStep(sessions, tokens: pending)
                for b in 0..<batchSize {
                    XCTAssertTrue(scratch.logitBuffer(b).allSatisfy(\.isFinite))
                    XCTAssertTrue(sessions[b].batchedDecodeTouched)
                    XCTAssertFalse(driver.canRunSpeculativeRound(sessions[b]))
                    batched[b].append(nextIds[b])
                }
                pending = nextIds
            }
            for b in 0..<batchSize {
                XCTAssertEqual(batched[b], serialOutputs[b], "B=\(batchSize) row \(b) diverged with mtpDrafts=2")
            }
        }

        // Concurrent burst forcing B=3 -> B=1 stepSingleSlot tail (batchedDecodeTouched == true), plus 1-stream speculative.
        let module = DenseModule(engine: driver)
        let engine = try QWEngine(engine: module, descriptor: driver.geometry.descriptor, plan: driver.plan)
        let scheduler = engine.batchScheduler
        let singleSpec = try scheduler.generate(request: QWGenerationRequest(prompt: [1, 10, 20, 30, 40, 10, 20, 30, 40], maxNew: 8, sampling: .greedy))
        let serialLong = try driver.generate(request: QWGenerationRequest(prompt: [1, 10, 20, 30, 40, 10, 20, 30, 40], maxNew: 8, sampling: .greedy), onToken: nil)
        XCTAssertEqual(singleSpec.tokens, serialLong.tokens)
        XCTAssertFalse(singleSpec.stepTokenCounts.isEmpty)

        final class Box: @unchecked Sendable {
            private let lock = NSLock()
            var values: [Int: QWGenerationResult] = [:]
            func set(_ idx: Int, _ val: QWGenerationResult) { lock.withLock { values[idx] = val } }
        }
        let box = Box()
        let group = DispatchGroup()
        let maxNewPerReq = [2, 2, 5]
        FNBatchAdmissionTracker.beginPreparing()
        for idx in 0..<3 {
            group.enter()
            DispatchQueue.global().async {
                defer { group.leave() }
                if let res = try? scheduler.generate(request: QWGenerationRequest(prompt: prompts[idx], maxNew: maxNewPerReq[idx], sampling: .greedy)) {
                    box.set(idx, res)
                }
            }
        }
        usleep(2000)
        FNBatchAdmissionTracker.endPreparing()
        group.wait()
        for idx in 0..<3 {
            XCTAssertEqual(box.values[idx]?.tokens, Array(serialOutputs[idx].prefix(maxNewPerReq[idx])))
        }
    }
}
