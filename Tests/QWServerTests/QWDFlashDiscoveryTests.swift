// QWDFlashDiscoveryTests.swift — Target-matched DFlash 2 / DSpine discovery and resolution tests.
import Foundation
import QWCore
import QWEngine
@testable import QWServer
import XCTest

final class QWDFlashDiscoveryTests: XCTestCase {
    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwg-dflash-disc-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func writeJSON(_ object: [String: Any], to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        try data.write(to: url)
    }

    private func flashNextConfig(layers: Int = 48) -> [String: Any] {
        let layerTypes = (0..<layers).map { ($0 + 1) % 4 == 0 ? "full_attention" : "linear_attention" }
        let text: [String: Any] = [
            "model_type": "qwen4_exp_text",
            "hidden_size": 2560,
            "vocab_size": 248320,
            "num_hidden_layers": layers,
            "layer_types": layerTypes,
            "full_attention_interval": 4,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262144,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "rope_parameters": ["rope_type": "default", "rope_theta": 1e7, "mrope_section": [11, 11, 10]],
            "output_gate_type": "sigmoid",
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "hc_count": 4,
            "hc_lowrank": 320,
            "mtp_num_hidden_layers": 1,
            "eos_token_id": 248044,
        ]
        return ["model_type": "qwen4_exp", "text_config": text]
    }

    private func dense27BConfig(layers: Int = 64) -> [String: Any] {
        let layerTypes = (0..<layers).map { ($0 + 1) % 4 == 0 ? "full_attention" : "linear_attention" }
        let text: [String: Any] = [
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "num_hidden_layers": layers,
            "layer_types": layerTypes,
            "full_attention_interval": 4,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262144,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "rope_parameters": ["rope_type": "default", "rope_theta": 1e7, "mrope_section": [11, 11, 10]],
            "attn_output_gate": true,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mtp_num_hidden_layers": 1,
            "eos_token_id": 248044,
        ]
        return ["model_type": "qwen3_5", "text_config": text]
    }

    private func llamaConfig() -> [String: Any] {
        [
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "vocab_size": 128256,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "eos_token_id": 128001,
        ]
    }

    private func drafterConfig(
        hidden: Int,
        vocab: Int = 248320,
        targetLayerIds: [Int],
        numTargetLayers: Int? = nil,
        blockSize: Int = 8,
        isDSpine: Bool = false
    ) -> [String: Any] {
        var dflash: [String: Any] = [
            "block_size": blockSize,
            "mask_token_id": 248077,
            "target_layer_ids": targetLayerIds,
        ]
        if isDSpine {
            dflash["selected_prefix_code_bridge"] = ["code_size": 1024, "rank": 512, "eps": 1e-6]
        }
        var out: [String: Any] = [
            "architectures": [isDSpine ? "DFlashDraftModel" : "DFlash2DraftModel"],
            "model_type": "qwen3",
            "hidden_size": hidden,
            "intermediate_size": hidden * 3,
            "num_hidden_layers": targetLayerIds.count,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": vocab,
            "block_size": blockSize,
            "dflash_config": dflash,
        ]
        if let numTargetLayers {
            out["num_target_layers"] = numTargetLayers
        }
        return out
    }

    private func writeMockPrimaryPack(at dir: URL, config: [String: Any]) throws {
        let assets = dir.appendingPathComponent("assets", isDirectory: true)
        try FileManager.default.createDirectory(at: assets, withIntermediateDirectories: true)
        try Data().write(to: dir.appendingPathComponent("model.qwp"))
        try writeJSON(config, to: assets.appendingPathComponent("config.json"))
        try Data("{}".utf8).write(to: assets.appendingPathComponent("tokenizer.json"))
        try Data("{% for m in messages %}{{ m.content }}{% endfor %}".utf8)
            .write(to: assets.appendingPathComponent("chat_template.jinja"))
    }

    func testScanDrafterModelsAndGeometryMatchingAcrossFlashNextAnd27B() throws {
        let root = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }

        let d27 = root.appendingPathComponent("Qwen3.8-27B-DFlash2-hq4g32")
        let s27 = root.appendingPathComponent("Qwen3.8-27B-DSpine-b16-hq4g32")
        let dfn = root.appendingPathComponent("Qwen3.8-Flash-Next-DFlash2-hq4g32")
        let dfnShvq = root.appendingPathComponent("Qwen3.8-Flash-Next-DFlash2-shvq3g64-shvq15g64")
        let badVocab = root.appendingPathComponent("BadVocab-DFlash2")
        let badLayers = root.appendingPathComponent("BadLayers-DFlash2")
        let primaryFN = root.appendingPathComponent("Qwen3.8-Flash-Next-hq8g32-hq4g64")

        try writeJSON(
            drafterConfig(hidden: 5120, targetLayerIds: [5, 19, 33, 47, 61], numTargetLayers: 64, blockSize: 8),
            to: d27.appendingPathComponent("assets/config.json")
        )
        try writeJSON(
            drafterConfig(hidden: 5120, targetLayerIds: [1, 16, 31, 46, 61], numTargetLayers: 64, blockSize: 16, isDSpine: true),
            to: s27.appendingPathComponent("assets/config.json")
        )
        try writeJSON(
            drafterConfig(hidden: 2560, targetLayerIds: [5, 15, 25, 35, 45], numTargetLayers: 48, blockSize: 8),
            to: dfn.appendingPathComponent("config.json")
        )
        try writeJSON(
            drafterConfig(hidden: 2560, targetLayerIds: [5, 15, 25, 35, 45], numTargetLayers: 48, blockSize: 8),
            to: dfnShvq.appendingPathComponent("assets/config.json")
        )
        try writeJSON(
            drafterConfig(hidden: 2560, vocab: 151936, targetLayerIds: [5, 15, 25, 35, 45], numTargetLayers: 48),
            to: badVocab.appendingPathComponent("assets/config.json")
        )
        try writeJSON(
            drafterConfig(hidden: 2560, targetLayerIds: [5, 15, 25, 35, 61], numTargetLayers: 48),
            to: badLayers.appendingPathComponent("assets/config.json")
        )
        try writeMockPrimaryPack(at: primaryFN, config: flashNextConfig())

        let scanned = QWServerSettings.scanDrafterModelsDirectory(in: root.path)
        let scannedNames = Set(scanned.map(\.name))
        XCTAssertEqual(scannedNames.count, 6)
        XCTAssertFalse(scannedNames.contains("Qwen3.8-Flash-Next-hq8g32-hq4g64"))
        XCTAssertEqual(QWServerSettings.inferDrafterBackendAndMaxDrafts(packPath: dfn.path).backend, "dflash2")
        XCTAssertEqual(QWServerSettings.inferDrafterBackendAndMaxDrafts(packPath: dfn.path).maxDrafts, 7)
        XCTAssertEqual(QWServerSettings.inferDrafterBackendAndMaxDrafts(packPath: s27.path).backend, "dspine")
        XCTAssertEqual(QWServerSettings.inferDrafterBackendAndMaxDrafts(packPath: s27.path).maxDrafts, 15)

        let fnDesc = try QWModelDescriptor.decode(configJSON: JSONSerialization.data(withJSONObject: flashNextConfig()))
        let q27Desc = try QWModelDescriptor.decode(configJSON: JSONSerialization.data(withJSONObject: dense27BConfig()))
        let llamaDesc = try QWModelDescriptor.decode(configJSON: JSONSerialization.data(withJSONObject: llamaConfig()))

        let fnCompatible = QWServerSettings.compatibleDrafterModels(for: fnDesc, in: scanned)
        XCTAssertEqual(Set(fnCompatible.map(\.name)), [
            "Qwen3.8-Flash-Next-DFlash2-hq4g32",
            "Qwen3.8-Flash-Next-DFlash2-shvq3g64-shvq15g64",
        ])
        let fnBest = QWServerSettings.resolveDefaultDrafterPack(
            for: fnDesc,
            preferredBackend: "dflash2",
            drafters: scanned,
            targetModelName: "Qwen3.8-Flash-Next-hq8g32-hq4g64"
        )
        XCTAssertEqual(fnBest?.name, "Qwen3.8-Flash-Next-DFlash2-hq4g32")

        let q27DFlash = QWServerSettings.resolveDefaultDrafterPack(
            for: q27Desc,
            preferredBackend: "dflash2",
            drafters: scanned,
            targetModelName: "Qwen3.8-27B-hq4g32"
        )
        XCTAssertEqual(q27DFlash?.name, "Qwen3.8-27B-DFlash2-hq4g32")

        let q27DSpine = QWServerSettings.resolveDefaultDrafterPack(
            for: q27Desc,
            preferredBackend: "dspine",
            drafters: scanned,
            targetModelName: "Qwen3.8-27B-hq4g32"
        )
        XCTAssertEqual(q27DSpine?.name, "Qwen3.8-27B-DSpine-b16-hq4g32")

        XCTAssertTrue(QWServerSettings.compatibleDrafterModels(for: llamaDesc, in: scanned).isEmpty)
        XCTAssertNil(QWServerSettings.resolveDefaultDrafterPack(for: llamaDesc, preferredBackend: "dflash2", drafters: scanned))
    }

    func testConfigurationResolverAndLifecycleLoadSelectTargetMatchedDrafter() throws {
        let root = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }

        let fnPrimary = root.appendingPathComponent("Qwen3.8-Flash-Next-hq8g32-hq4g64")
        let q27Primary = root.appendingPathComponent("Qwen3.8-27B-hq4g32")
        let llamaPrimary = root.appendingPathComponent("Llama-3-8B")
        let d27 = root.appendingPathComponent("Qwen3.8-27B-DFlash2-hq4g32")
        let s27 = root.appendingPathComponent("Qwen3.8-27B-DSpine-b16-hq4g32")
        let dfn = root.appendingPathComponent("Qwen3.8-Flash-Next-DFlash2-hq4g32")

        try writeMockPrimaryPack(at: fnPrimary, config: flashNextConfig())
        try writeMockPrimaryPack(at: q27Primary, config: dense27BConfig())
        try writeMockPrimaryPack(at: llamaPrimary, config: llamaConfig())
        try writeJSON(
            drafterConfig(hidden: 5120, targetLayerIds: [5, 19, 33, 47, 61], numTargetLayers: 64, blockSize: 8),
            to: d27.appendingPathComponent("assets/config.json")
        )
        try writeJSON(
            drafterConfig(hidden: 5120, targetLayerIds: [1, 16, 31, 46, 61], numTargetLayers: 64, blockSize: 16, isDSpine: true),
            to: s27.appendingPathComponent("assets/config.json")
        )
        try writeJSON(
            drafterConfig(hidden: 2560, targetLayerIds: [5, 15, 25, 35, 45], numTargetLayers: 48, blockSize: 8),
            to: dfn.appendingPathComponent("assets/config.json")
        )

        let drafters = QWServerSettings.scanDrafterModelsDirectory(in: root.path)
        var fnSources: [String: String] = [:]
        var fnSettings = QWServerSettings()
        fnSettings.modelName = "Qwen3.8-Flash-Next-hq8g32-hq4g64"
        let fnResolved = QWServerConfigurationResolver.resolveSpeculativeAndTuning(
            fileSettings: fnSettings, environment: [:], sources: &fnSources,
            resolvedPack: fnPrimary.path, resolvedMaxContext: 4096,
            cliMTPBackend: nil, cliDFlashDraftPack: nil, cliDFlashBlockSize: 8, cliDFlashTopK: nil,
            cliGDNPrefill: nil, cliEvictScratch: nil, cliPrefillChunkSize: nil, cliMaxConcurrentSessions: nil,
            cliLSAMode: nil, cliLSABudget: nil, cliLSAClusterBlock: nil, cliLSASidecarPath: nil,
            cliWarmRAMLimitGB: nil, cliSSDCacheLimitGB: nil, cliCacheDir: nil, cliLogDir: nil,
            cliLogFileName: nil, cliLogLevel: nil, cliLazy: nil, cliMTPAdaptiveDepth: nil,
            cliMTPEntropyThreshold: nil, cliMTPRelaxedMargin: nil, cliMTPRelaxedMinProb: nil,
            cliMTPDraftStrategy: nil, cliMTPQuadraticSeeding: nil, cliLoadVision: nil,
            cliMemoryPolicy: nil, cliOSReserveGiB: nil, cliResidentExpertsPerLayer: nil,
            cliMoESpecBudget: nil, cliStreamNgrams: nil, drafters: drafters
        )
        XCTAssertEqual(fnResolved.dflashDraftPackPath, dfn.path)
        XCTAssertEqual(fnResolved.mtpBackend, "dflash2")
        XCTAssertEqual(fnSources["dflashDraftPackPath"], "discovered (~/.qwg/models/Qwen3.8-Flash-Next-DFlash2-hq4g32)")

        var dspineSources: [String: String] = [:]
        var q27Settings = QWServerSettings()
        q27Settings.modelName = "Qwen3.8-27B-hq4g32"
        let dspineResolved = QWServerConfigurationResolver.resolveSpeculativeAndTuning(
            fileSettings: q27Settings, environment: [:], sources: &dspineSources,
            resolvedPack: q27Primary.path, resolvedMaxContext: 4096,
            cliMTPBackend: "dspine", cliDFlashDraftPack: nil, cliDFlashBlockSize: nil, cliDFlashTopK: nil,
            cliGDNPrefill: nil, cliEvictScratch: nil, cliPrefillChunkSize: nil, cliMaxConcurrentSessions: nil,
            cliLSAMode: nil, cliLSABudget: nil, cliLSAClusterBlock: nil, cliLSASidecarPath: nil,
            cliWarmRAMLimitGB: nil, cliSSDCacheLimitGB: nil, cliCacheDir: nil, cliLogDir: nil,
            cliLogFileName: nil, cliLogLevel: nil, cliLazy: nil, cliMTPAdaptiveDepth: nil,
            cliMTPEntropyThreshold: nil, cliMTPRelaxedMargin: nil, cliMTPRelaxedMinProb: nil,
            cliMTPDraftStrategy: nil, cliMTPQuadraticSeeding: nil, cliLoadVision: nil,
            cliMemoryPolicy: nil, cliOSReserveGiB: nil, cliResidentExpertsPerLayer: nil,
            cliMoESpecBudget: nil, cliStreamNgrams: nil, drafters: drafters
        )
        XCTAssertEqual(dspineResolved.dflashDraftPackPath, s27.path)
        XCTAssertEqual(dspineResolved.mtpBackend, "dspine")

        var llamaSources: [String: String] = [:]
        var llamaSettings = QWServerSettings()
        llamaSettings.modelName = "Llama-3-8B"
        let llamaResolved = QWServerConfigurationResolver.resolveSpeculativeAndTuning(
            fileSettings: llamaSettings, environment: [:], sources: &llamaSources,
            resolvedPack: llamaPrimary.path, resolvedMaxContext: 4096,
            cliMTPBackend: nil, cliDFlashDraftPack: nil, cliDFlashBlockSize: nil, cliDFlashTopK: nil,
            cliGDNPrefill: nil, cliEvictScratch: nil, cliPrefillChunkSize: nil, cliMaxConcurrentSessions: nil,
            cliLSAMode: nil, cliLSABudget: nil, cliLSAClusterBlock: nil, cliLSASidecarPath: nil,
            cliWarmRAMLimitGB: nil, cliSSDCacheLimitGB: nil, cliCacheDir: nil, cliLogDir: nil,
            cliLogFileName: nil, cliLogLevel: nil, cliLazy: nil, cliMTPAdaptiveDepth: nil,
            cliMTPEntropyThreshold: nil, cliMTPRelaxedMargin: nil, cliMTPRelaxedMinProb: nil,
            cliMTPDraftStrategy: nil, cliMTPQuadraticSeeding: nil, cliLoadVision: nil,
            cliMemoryPolicy: nil, cliOSReserveGiB: nil, cliResidentExpertsPerLayer: nil,
            cliMoESpecBudget: nil, cliStreamNgrams: nil, drafters: drafters
        )
        XCTAssertEqual(llamaResolved.dflashDraftPackPath, "")

        let fnDesc = try XCTUnwrap(QWServerSettings.loadPrimaryDescriptor(packPath: fnPrimary.path))
        let q27Desc = try XCTUnwrap(QWServerSettings.loadPrimaryDescriptor(packPath: q27Primary.path))
        let fnLoadCfg = QWEngineLifecycleActor.Config(
            packPath: fnPrimary.path, modelName: "Qwen3.8-Flash-Next",
            mtpDrafts: 7, mtpBackend: "dflash2", dflashDraftPackPath: nil
        )
        XCTAssertEqual(
            QWEngineLifecycleActor.resolveEffectiveDrafterPackPath(for: fnLoadCfg, targetDescriptor: fnDesc, drafters: drafters),
            dfn.path
        )
        let q27DSpineLoadCfg = QWEngineLifecycleActor.Config(
            packPath: q27Primary.path, modelName: "Qwen3.8-27B",
            mtpDrafts: 15, mtpBackend: "dspine", dflashDraftPackPath: ""
        )
        XCTAssertEqual(
            QWEngineLifecycleActor.resolveEffectiveDrafterPackPath(for: q27DSpineLoadCfg, targetDescriptor: q27Desc, drafters: drafters),
            s27.path
        )
        let emptyDSpineLoadCfg = QWEngineLifecycleActor.Config(
            packPath: q27Primary.path, modelName: "Qwen3.8-27B",
            mtpDrafts: 15, mtpBackend: "dspine", dflashDraftPackPath: nil
        )
        XCTAssertEqual(
            QWEngineLifecycleActor.resolveEffectiveDrafterPackPath(for: emptyDSpineLoadCfg, targetDescriptor: q27Desc, drafters: []),
            QWDFlashConfig.defaultDSpinePackPath
        )
    }

    func testSupportsDFlash2AndValidationGuardsFailFastSafely() throws {
        let root = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }

        let fnDesc = try QWModelDescriptor.decode(configJSON: JSONSerialization.data(withJSONObject: flashNextConfig()))
        let llamaDesc = try QWModelDescriptor.decode(configJSON: JSONSerialization.data(withJSONObject: llamaConfig()))
        let fnDisc = DiscoveredModel(
            name: "Qwen3.8-Flash-Next", packPath: "/tmp/fn",
            family: QWModelDescriptor.Family.qwen38FlashNext.rawValue,
            capabilities: ["hyperConnections", "moe", "mtp", "ple", "qsaIndexer", "vision"]
        )
        let capDFlashDisc = DiscoveredModel(name: "Custom", packPath: "/tmp/c", capabilities: ["dflash"])
        let llamaDisc = DiscoveredModel(
            name: "Llama-3-8B", packPath: "/tmp/llama",
            family: QWModelDescriptor.Family.llamaDense.rawValue,
            capabilities: []
        )

        XCTAssertTrue(QWServerSettings.supportsDFlash2(descriptor: fnDesc, discoveredModel: fnDisc, drafters: []))
        XCTAssertTrue(QWServerSettings.supportsDFlash2(descriptor: nil, discoveredModel: fnDisc, drafters: []))
        XCTAssertTrue(QWServerSettings.supportsDFlash2(descriptor: nil, discoveredModel: capDFlashDisc, drafters: []))
        XCTAssertFalse(QWServerSettings.supportsDFlash2(descriptor: llamaDesc, discoveredModel: llamaDisc, drafters: []))

        let densePrimary = root.appendingPathComponent("Dense27B")
        try writeMockPrimaryPack(at: densePrimary, config: dense27BConfig())
        var settings = QWServerSettings()
        settings.modelName = "Dense27B"
        settings.modelPackPath = densePrimary.path
        settings.speculativeDecodingEnabled = true
        settings.mtpBackend = "variational"
        XCTAssertNoThrow(try settings.validateBeforeLaunch())

        settings.mtpBackend = "dflash2"
        settings.mtpDrafts = 7
        settings.dflashDraftPackPath = root.appendingPathComponent("missing-drafter").path
        XCTAssertThrowsError(try settings.validateDFlashConfiguration())
        XCTAssertThrowsError(try settings.validateBeforeLaunch())

        let mismatchedDrafter = root.appendingPathComponent("Qwen3.8-Flash-Next-DFlash2-hq4g32")
        try writeJSON(
            drafterConfig(hidden: 2560, targetLayerIds: [5, 15, 25, 35, 45], numTargetLayers: 48, blockSize: 8),
            to: mismatchedDrafter.appendingPathComponent("assets/config.json")
        )
        settings.dflashDraftPackPath = mismatchedDrafter.path
        XCTAssertThrowsError(try settings.validateDFlashConfiguration()) { err in
            XCTAssertTrue("\(err)".contains("hiddenSize"))
        }
        XCTAssertThrowsError(try settings.validateBeforeLaunch()) { err in
            XCTAssertTrue("\(err)".contains("hiddenSize"))
        }
    }

    func testDrafterQuantPreferencePrefersPureSHVQ4OverHQ4AndMixedSub4Bit() {
        let shvq4 = QWDFlashConfig.drafterQuantPreferenceRank("/tmp/Qwen3.8-Flash-Next-DFlash2-shvq4g32")
        let hq4 = QWDFlashConfig.drafterQuantPreferenceRank("/tmp/Qwen3.8-Flash-Next-DFlash2-hq4g32")
        let mixed = QWDFlashConfig.drafterQuantPreferenceRank("/tmp/Qwen3.8-Flash-Next-DFlash2-shvq4g32-shvq3g64")
        XCTAssertGreaterThan(shvq4, hq4)
        XCTAssertGreaterThan(hq4, mixed)
    }

    func testSupportsExternalDrafterScanSummariesAndReconcileStaleProfiles() throws {
        let root = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }

        let fnPrimary = root.appendingPathComponent("Qwen3.8-Flash-Next-hq8g32-hq4g64")
        let q27Primary = root.appendingPathComponent("Qwen3.8-27B-hq4g32")
        let llamaPrimary = root.appendingPathComponent("Llama-3-8B")
        let dfn = root.appendingPathComponent("Qwen3.8-Flash-Next-DFlash2-shvq4g32")
        let d27 = root.appendingPathComponent("Qwen3.8-27B-DFlash2-hq4g32")
        let s27 = root.appendingPathComponent("Qwen3.8-27B-DSpine-b16-hq4g32")

        try writeMockPrimaryPack(at: fnPrimary, config: flashNextConfig())
        try writeMockPrimaryPack(at: q27Primary, config: dense27BConfig())
        try writeMockPrimaryPack(at: llamaPrimary, config: llamaConfig())
        try writeJSON(
            drafterConfig(hidden: 2560, targetLayerIds: [5, 15, 25, 35, 45], numTargetLayers: 48, blockSize: 8),
            to: dfn.appendingPathComponent("assets/config.json")
        )
        try writeJSON(
            drafterConfig(hidden: 5120, targetLayerIds: [5, 19, 33, 47, 61], numTargetLayers: 64, blockSize: 8),
            to: d27.appendingPathComponent("assets/config.json")
        )
        try writeJSON(
            drafterConfig(hidden: 5120, targetLayerIds: [1, 16, 31, 46, 61], numTargetLayers: 64, blockSize: 16, isDSpine: true),
            to: s27.appendingPathComponent("assets/config.json")
        )

        let fnDesc = try XCTUnwrap(QWServerSettings.loadPrimaryDescriptor(packPath: fnPrimary.path))
        let q27Desc = try XCTUnwrap(QWServerSettings.loadPrimaryDescriptor(packPath: q27Primary.path))
        let llamaDesc = try XCTUnwrap(QWServerSettings.loadPrimaryDescriptor(packPath: llamaPrimary.path))
        let drafters = QWServerSettings.scanDrafterModelsDirectory(in: root.path)

        XCTAssertTrue(QWServerSettings.supportsExternalDrafter(backend: "dflash2", descriptor: fnDesc, drafters: drafters))
        XCTAssertFalse(QWServerSettings.supportsExternalDrafter(backend: "dspine", descriptor: fnDesc, drafters: drafters))
        XCTAssertTrue(QWServerSettings.supportsExternalDrafter(backend: "dflash2", descriptor: q27Desc, drafters: drafters))
        XCTAssertTrue(QWServerSettings.supportsExternalDrafter(backend: "dspine", descriptor: q27Desc, drafters: drafters))
        XCTAssertFalse(QWServerSettings.supportsExternalDrafter(backend: "dflash2", descriptor: llamaDesc, drafters: drafters))
        XCTAssertFalse(QWServerSettings.supportsExternalDrafter(backend: "dspine", descriptor: llamaDesc, drafters: drafters))

        let primaries: [QWServerSettings.DiscoveredModel] = [
            .init(name: "Qwen3.8-Flash-Next-hq8g32-hq4g64", packPath: fnPrimary.path, supported: true),
            .init(name: "Qwen3.8-27B-hq4g32", packPath: q27Primary.path, supported: true),
            .init(name: "Llama-3-8B", packPath: llamaPrimary.path, supported: true),
        ]
        let summaries = QWServerSettings.scanDrafterSummaries(primaryModels: primaries, drafters: drafters)
        let byName = Dictionary(uniqueKeysWithValues: summaries.map { ($0.name, $0) })
        XCTAssertEqual(byName["Qwen3.8-Flash-Next-DFlash2-shvq4g32"]?.backend, "dflash2")
        XCTAssertEqual(byName["Qwen3.8-Flash-Next-DFlash2-shvq4g32"]?.maxDrafts, 7)
        XCTAssertEqual(byName["Qwen3.8-Flash-Next-DFlash2-shvq4g32"]?.compatibleModels, ["Qwen3.8-Flash-Next-hq8g32-hq4g64"])
        XCTAssertEqual(byName["Qwen3.8-27B-DSpine-b16-hq4g32"]?.backend, "dspine")
        XCTAssertEqual(byName["Qwen3.8-27B-DSpine-b16-hq4g32"]?.maxDrafts, 15)
        XCTAssertEqual(byName["Qwen3.8-27B-DSpine-b16-hq4g32"]?.compatibleModels, ["Qwen3.8-27B-hq4g32"])

        // Heal stale cross-model path + unsupported DSpine selection on Flash-Next.
        var fnStale = QWServerSettings(modelPackPath: fnPrimary.path, modelName: "Qwen3.8-Flash-Next-hq8g32-hq4g64")
        fnStale.mtpBackend = "dspine"
        fnStale.mtpDrafts = 15
        fnStale.dflashDraftPackPath = s27.path
        XCTAssertTrue(fnStale.reconcileStaleDrafterSelection(for: fnDesc, drafters: drafters))
        XCTAssertEqual(fnStale.dflashDraftPackPath, dfn.path)
        XCTAssertEqual(fnStale.mtpBackend, "dflash2")
        XCTAssertEqual(fnStale.mtpDrafts, 7)

        // Heal stale cross-model path stored even while variational backend is active.
        var fnVarStale = QWServerSettings(modelPackPath: fnPrimary.path, modelName: "Qwen3.8-Flash-Next-hq8g32-hq4g64")
        fnVarStale.mtpBackend = "variational"
        fnVarStale.dflashDraftPackPath = root.appendingPathComponent("deleted-dflash-pack").path
        XCTAssertTrue(fnVarStale.reconcileStaleDrafterSelection(for: fnDesc, drafters: drafters))
        XCTAssertEqual(fnVarStale.dflashDraftPackPath, dfn.path)
        XCTAssertEqual(fnVarStale.mtpBackend, "variational")

        // Heal unsupported primary model selection cleanly back to variational with empty path.
        var llamaStale = QWServerSettings(modelPackPath: llamaPrimary.path, modelName: "Llama-3-8B")
        llamaStale.mtpBackend = "dflash2"
        llamaStale.mtpDrafts = 7
        llamaStale.dflashDraftPackPath = d27.path
        XCTAssertTrue(llamaStale.reconcileStaleDrafterSelection(for: llamaDesc, drafters: drafters))
        XCTAssertEqual(llamaStale.mtpBackend, "variational")
        XCTAssertEqual(llamaStale.dflashDraftPackPath, "")

        // Verify QWSettingsManager.effectiveSettings heals stale profile without disturbing strict validation.
        let settingsURL = root.appendingPathComponent("settings.json")
        let mgr = QWSettingsManager()
        let store = mgr.profileStore(forSettingsURL: settingsURL)
        try store.save(QWKnobProfile(
            modelId: "Qwen3.8-Flash-Next-hq8g32-hq4g64",
            values: ["dflashDraftPackPath": .string(root.appendingPathComponent("deleted-pack").path)]
        ))
        let (eff, _) = mgr.effectiveSettings(
            forModel: "Qwen3.8-Flash-Next-hq8g32-hq4g64",
            packPath: fnPrimary.path,
            base: QWServerSettings(),
            from: settingsURL
        )
        XCTAssertNotEqual(eff.dflashDraftPackPath, root.appendingPathComponent("deleted-pack").path)
    }

    func testWebUIActionsExposesDFlash2AndDSpineControls() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let actionsJS = try String(
            contentsOf: repoRoot.appendingPathComponent("Sources/Apps/QWServer/Resources/js/actions.js"),
            encoding: .utf8
        )
        let stateJS = try String(
            contentsOf: repoRoot.appendingPathComponent("Sources/Apps/QWServer/Resources/js/state.js"),
            encoding: .utf8
        )
        XCTAssertTrue(stateJS.contains("discoveredDraftersList"))
        for needle in [
            "dflashDraftPackPath",
            "dflashSelectorTopK",
            "mtpDrafts",
            "mtpDraftStrategy",
            "handleWebMTPBackendChange",
            "handleWebDrafterSelectChange",
            "discoveredDraftersList",
        ] {
            XCTAssertTrue(actionsJS.contains(needle), "Missing expected WebUI settings symbol '\(needle)' in actions.js")
        }
    }
}
