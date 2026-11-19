// QWModuleRegistry.swift — Data-driven registry mapping model families & traits to modules.
import Foundation
import QWCore
import QWHAL
import QWKV

/// Maps a model family and `QWModelFamilyRegistry` traits to the module that serves it.
public enum QWModuleRegistry {
    private static let registeredModules: [any QWInferenceModule.Type] = [
        DenseModule.self,
        Qwen38MoEModule.self,
        FlashNextModule.self,
        SenseNovaLoopModule.self,
        Glm5NextModule.self,
        Ernie55MoEModule.self,
        MaxwellMoEModule.self,
        Qwen2DenseModule.self,
        LlamaModule.self,
        MistralModule.self,
        Qwen3DenseModule.self,
        Qwen3MoEModule.self,
    ]

    private static let modulesByFamily: [QWModelDescriptor.Family: any QWInferenceModule.Type] =
        Dictionary(uniqueKeysWithValues: registeredModules.map { ($0.family, $0) })

    /// The module type for a family, or nil when no module serves it.
    public static func moduleType(for family: QWModelDescriptor.Family) -> (any QWInferenceModule.Type)? {
        modulesByFamily[family]
    }

    /// Family specification from `QWModelFamilyRegistry`, when registered.
    public static func familySpec(for family: QWModelDescriptor.Family) -> QWModelFamilySpec? {
        QWModelFamilyRegistry.shared.spec(for: family)
    }

    /// Every family a module serves. Derived from `Family.allCases` and registered modules.
    public static var supportedFamilies: [QWModelDescriptor.Family] {
        QWModelDescriptor.Family.allCases.filter { modulesByFamily[$0] != nil }
    }

    /// The family a pack declares, resolved directly or via `QWModelFamilyRegistry`.
    public static func family(forPackModelId modelId: String) -> QWModelDescriptor.Family? {
        if let direct = supportedFamilies.first(where: { $0.rawValue == modelId }) {
            return direct
        }
        if let spec = QWModelFamilyRegistry.shared.spec(forModelType: modelId),
           modulesByFamily[spec.family] != nil {
            return spec.family
        }
        return nil
    }

    /// The module that serves a pack's declared model id or HuggingFace `model_type`.
    public static func moduleType(forPackModelId modelId: String) -> (any QWInferenceModule.Type)? {
        family(forPackModelId: modelId).flatMap(moduleType(for:))
    }

    /// The capabilities the module that serves a family offers.
    public static func capabilities(for family: QWModelDescriptor.Family) -> Set<QWKnobCapability> {
        moduleType(for: family)?.capabilities ?? []
    }

    /// Builds the engine that serves a family without naming a concrete engine type.
    public static func makeEngine(for family: QWModelDescriptor.Family,
                                  options: QWEngineBuildOptions) throws -> any QWInferenceEngine {
        guard let type = moduleType(for: family) else {
            throw QWModuleError.unsupportedFamily(family)
        }
        return try type.makeEngine(options: options)
    }

    /// Predicts total GPU memory required (weights + KV + scratch) for a model descriptor and context length.
    public static func estimateContextMemory(
        descriptor: QWModelDescriptor,
        packPath: String? = nil,
        contextTokens: Int,
        kvMode: QWKVMode = .f16,
        mtpDrafts: Int = 0,
        mtpBackend: String = "variational",
        prefillChunkSize: Int = QWPrefillLimits.chunkRows,
        tuning: QWEngineTuning = .default,
        physicalMemoryBytes: Int? = QWAdmission.physicalMemoryBytes()
    ) -> QWContextMemoryEstimate {
        let module = moduleType(for: descriptor.family) ?? DenseModule.self
        return module.estimateContextMemory(
            descriptor: descriptor, packPath: packPath, contextTokens: contextTokens,
            kvMode: kvMode, mtpDrafts: mtpDrafts, mtpBackend: mtpBackend,
            prefillChunkSize: prefillChunkSize, tuning: tuning,
            physicalMemoryBytes: physicalMemoryBytes
        )
    }
}
