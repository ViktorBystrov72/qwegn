// LoadPolicy.swift — one place turning the model descriptor (config.json)
// plus env kill-switches into load behavior. Defaults are config-derived;
// env vars may only DISABLE (ship law), never invent a default.
import Foundation
import QWCore

public struct LoadPolicy: Sendable {
    public var directIO: Bool
    public var zeroCopyMaps: Bool
    public var wireByDefault: Bool
    public var wireForced: Bool
    public var wireSkipped: Bool
    public var wireHeadroomMarginBytes: Int
    /// Preserve BF16/HQ8 precision on safetensors decision packs <= 8B parameters (`ADR-009`).
    public var preserveBF16DecisionWeights: Bool

    public var preferredSafetensorsPrecision: String {
        preserveBF16DecisionWeights ? "bf16" : "hq4_g32"
    }

    public init(descriptor: QWModelDescriptor,
                env: [String: String] = ProcessInfo.processInfo.environment) {
        directIO = env["QW_PLANE_BUFFERED"] != "1"
        zeroCopyMaps = env["QW_PLANE_MAP_DISABLE"] != "1"
        wireByDefault = false
        wireForced = env["QW_WIRE_FORCE"] == "1" || env["QW_FN_WIRE_FORCE"] == "1"
        wireSkipped = env["QW_WIRE_SKIP"] == "1" || env["QW_FN_WIRE_SKIP"] == "1"
        wireHeadroomMarginBytes = 6 << 30
        let isDecision = descriptor.decisionConfig != nil || descriptor.capabilities.contains(.decision)
        let paramCount = Self.estimatedParamCount(descriptor)
        preserveBF16DecisionWeights = isDecision
            && paramCount <= 8_000_000_000
            && env["QW_DECISION_FORCE_HQ4"] != "1"
    }

    private static func estimatedParamCount(_ d: QWModelDescriptor) -> Int64 {
        let h = Int64(max(1, d.hidden))
        let inter = Int64(max(Int(h), d.denseIntermediate ?? Int(h)))
        let layers = Int64(max(1, d.numLayers))
        let vocab = Int64(max(0, d.vocab))
        let moe = Int64(max(0, d.numExperts ?? 0)) * 3 * h * Int64(max(0, d.moeIntermediate ?? 0))
        let perLayer = 4 * h * h + 3 * h * inter + moe
        let embed = (d.tieWordEmbeddings ? 1 : 2) * vocab * h
        return embed + layers * perLayer
    }
}
