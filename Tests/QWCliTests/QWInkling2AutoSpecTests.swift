// QWInkling2AutoSpecTests.swift — Synthetic CLI auto-DSpark resolution & chunk clamping tests.
import ArgumentParser
import Foundation
import Testing
@testable import qwg
import QWCore
@testable import QWEngine
import QWServer

@Suite("Inkling-2 auto DSpark and prefill chunk clamping")
struct QWInkling2AutoSpecTests {
    private static func makeSyntheticTokenizerData() throws -> Data {
        var vocab: [String: Int] = ["<|endoftext|>": 200006]
        var nextId = 0
        var n = 0
        for b in 0..<256 {
            let idc = Unicode.Scalar(UInt8(b))
            let printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
            let sc: Unicode.Scalar = printable ? idc : Unicode.Scalar(0x100 + n)!
            if !printable { n += 1 }
            vocab[String(sc)] = nextId
            nextId += 1
        }
        let root: [String: Any] = [
            "added_tokens": [["id": 200006, "content": "<|endoftext|>", "special": true]],
            "model": ["type": "BPE", "vocab": vocab, "merges": [String]()],
            "pre_tokenizer": [
                "type": "Sequence",
                "pretokenizers": [
                    ["type": "Split", "pattern": ["Regex": "\\s+"], "behavior": "Isolated", "invert": false],
                    ["type": "ByteLevel", "use_regex": false]
                ]
            ]
        ]
        return try JSONSerialization.data(withJSONObject: root)
    }

    private static func makeSyntheticPack(draftFilename: String?) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg_ink2_autospec_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let localLayers = (0..<24).filter { ($0 + 1) % 6 != 0 }
        let obj: [String: Any] = [
            "architectures": ["Inkling2LoopForConditionalGeneration"],
            "eos_token_id": 200006,
            "model_type": "inkling2_loop",
            "generation_config": ["eos_token_id": 200006, "pad_token_id": 199999],
            "mtp_config": [
                "hidden_size": 2304, "num_hidden_layers": 3, "num_attention_heads": 24,
                "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 9216,
                "max_position_embeddings": 1048576, "rms_norm_eps": 1e-6, "rope_theta": 1_000_000.0,
                "q_bias": false, "o_bias": false, "local_layer_ids": [0, 2],
                "sliding_window_size": 1024, "block_size": 8, "markov_rank": 256,
                "markov_head_type": "rnn_hidden", "target_layer_ids": [5, 23, 41, 48]
            ],
            "text_config": [
                "model_type": "inkling2_loop_text",
                "model_max_length": 1048576, "hidden_size": 2304, "num_hidden_layers": 24,
                "vocab_size": 201024, "unpadded_vocab_size": 200058,
                "num_attention_heads": 24, "num_key_value_heads": 8, "head_dim": 128,
                "d_rel": 16, "rel_extent": 1024, "q_bias": false, "o_bias": false,
                "log_scaling_n_floor": 8192, "log_scaling_alpha": 0.1,
                "rms_norm_eps": 1e-6, "use_value_norm_no_gamma": true,
                "qk_l2_clip_threshold": 22.627416997969522, "rel_l2_clip_threshold": 8.0,
                "local_layer_ids": localLayers, "sliding_window_size": 1024, "sconv_kernel_size": 4,
                "final_logit_softcapping": 30.0, "n_layers_per_attn_res_block": 6,
                "attn_res_normalize_values": true, "n_routed_experts": 128,
                "num_experts_per_tok": 8, "n_shared_experts": 0, "intermediate_size": 1152,
                "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 8.0,
                "use_gate_bias": true,
                "loop_config": ["core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": false]
            ]
        ]
        let cfgData = try JSONSerialization.data(withJSONObject: obj)
        try cfgData.write(to: dir.appendingPathComponent("config.json"))
        try Self.makeSyntheticTokenizerData().write(to: dir.appendingPathComponent("tokenizer.json"))
        try Data("{% for m in messages %}{{ m.content }}{% endfor %}".utf8)
            .write(to: dir.appendingPathComponent("chat_template.jinja"))
        try Data().write(to: dir.appendingPathComponent("model.qwp"))
        if let draftFilename {
            try Data().write(to: dir.appendingPathComponent(draftFilename))
        }
        return dir
    }

    private static func resolveConfig(packURL: URL, args: [String]) throws -> (QWResolvedCliEngineConfig, QWGKnobOptions) {
        let cmd = try Run.parse(args + [packURL.path])
        let settings = QWServerSettings()
        let mtp = try QWResolvedCliEngineConfig.resolvePackAndMTP(
            verb: "qwg run", modelOption: cmd.modelOption, pack: cmd.pack,
            knobs: cmd.knobs, dflashWindow: nil, settings: settings
        )
        let cfg = try QWResolvedCliEngineConfig.resolveAssetsAndKnobs(
            verb: "qwg run", resolvedPack: mtp.resolvedPack, knobs: cmd.knobs, dflashWindow: nil,
            initialMtpDrafts: mtp.effectiveMtpDrafts, effectiveMtpBackend: mtp.effectiveMtpBackend,
            resolvedBackend: mtp.resolvedBackend, image: nil, video: nil, settings: settings,
            validateDefaultContextImmediately: false, ignoreProfile: true
        )
        return (cfg, cmd.knobs)
    }

    @Test("Auto-DSpark defaults to 3 drafts when dspark.qwp or mtp.safetensors exists, honors --mtp-drafts 0")
    func autoDSparkDefaultsAndExplicitZero() throws {
        let dirQwp = try Self.makeSyntheticPack(draftFilename: "dspark.qwp")
        defer { try? FileManager.default.removeItem(at: dirQwp) }

        let (cfgDefault, knobsDefault) = try Self.resolveConfig(packURL: dirQwp, args: [])
        #expect(cfgDefault.effectiveMtpDrafts == 3)
        #expect(cfgDefault.makeSpeculativeParameters(knobs: knobsDefault).mtpDrafts == 3)
        #expect(try QWMTPBackend.resolveSelection(cfgDefault.resolvedBackend, drafts: cfgDefault.effectiveMtpDrafts) == .standard)

        let (cfgZero, knobsZero) = try Self.resolveConfig(packURL: dirQwp, args: ["--mtp-drafts", "0"])
        #expect(cfgZero.effectiveMtpDrafts == 0)
        #expect(!cfgZero.makeSpeculativeParameters(knobs: knobsZero).enabled)

        let (cfg15, knobs15) = try Self.resolveConfig(packURL: dirQwp, args: ["--mtp-drafts", "15"])
        #expect(cfg15.effectiveMtpDrafts == 15)
        #expect(cfg15.makeSpeculativeParameters(knobs: knobs15).mtpDrafts == 15)
        #expect(try QWMTPBackend.resolveSelection(cfg15.resolvedBackend, drafts: cfg15.effectiveMtpDrafts) == .standard)

        let dirMtp = try Self.makeSyntheticPack(draftFilename: "mtp.safetensors")
        defer { try? FileManager.default.removeItem(at: dirMtp) }
        let (cfgMtp, _) = try Self.resolveConfig(packURL: dirMtp, args: [])
        #expect(cfgMtp.effectiveMtpDrafts == 3)
    }

    @Test("Prefill chunk size clamps safely to 32...min(8192, capacity)")
    func prefillChunkSizeClamping() throws {
        #expect(throws: (any Error).self) { _ = try Run.parse(["--prefill-chunk-size", "16384", "dummy"]) }
        #expect(throws: (any Error).self) { _ = try Run.parse(["--prefill-chunk-size", "16", "dummy"]) }
        let high = try Run.parse(["--prefill-chunk-size", "8192", "dummy"])
        let low = try Run.parse(["--prefill-chunk-size", "32", "dummy"])
        #expect(high.knobs.prefillChunkSize == 8192)
        #expect(low.knobs.prefillChunkSize == 32)
        #expect(QWPrefillLimits.clampedRows(16384) == 8192)
        #expect(QWPrefillLimits.clampedRows(16) == 32)
        #expect(QWPrefillLimits.clampedRows(4096) == 4096)
        #expect(QWPrefillLimits.clampedRows(500) == 480)
        #expect(QWInkling2Session.resolveMaxChunk(prefillChunkSize: 16384, capacity: 32768) == 8192)
        #expect(QWInkling2Session.resolveMaxChunk(prefillChunkSize: 8192, capacity: 4096) == 4096)
        #expect(QWInkling2Session.resolveMaxChunk(prefillChunkSize: 8, capacity: 4096) == 32)
    }
}
