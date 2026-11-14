// QWSlotGeometry.swift — Recurrent/conv slot sizing derived from geometry.
import Foundation

/// Recurrent state and conv-ring footprint math derived from `QWModelGeometry`.
public enum QWSlotGeometry {
    /// Linear-attention (GDN/KDA) layer count from the canonical execution plan.
    public static func gdnLayers(geometry: QWModelGeometry) -> Int {
        geometry.gdnLayerCount
    }

    /// fp32 elements of one layer's recurrent state (`gdnVHeads * gdnHeadDim^2`).
    public static func recElems(geometry: QWModelGeometry) -> Int {
        geometry.gdnVHeads * geometry.gdnHeadDim * geometry.gdnHeadDim
    }

    /// elements of one layer's conv ring (`(gdnConvKernel - 1) * gdnQKVChannels`).
    public static func ringElems(geometry: QWModelGeometry) -> Int {
        (geometry.gdnConvKernel - 1) * geometry.gdnQKVChannels
    }

    /// Bytes of one sequence's full fp32 recurrent + conv-ring state.
    public static func stateBytes(geometry: QWModelGeometry) -> Int {
        gdnLayers(geometry: geometry) * (recElems(geometry: geometry) + ringElems(geometry: geometry)) * 4
    }
}
