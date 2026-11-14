// QWMLP.swift — Packed weight descriptor for SwiGLU / dense MLP projections.

import Foundation
import QWCore
import QWAlloc
import QWGEMM

/// Packed HQ4_G32_M5 weight plane for one projection (contiguous 20-byte Block_HQ4_G32),
/// `n` rows × `k` inner — layout-validated at init by QWWeightView rules.
public struct QWMLPWeight: Sendable {
    public let view: QWWeightView
    public init?(blocks: QWTensorValue, n: Int, k: Int, bits: Int = 4, groupSize: Int = 32,
                 isRotated: Bool, isIncumbentContainer: Bool? = nil, isSHVQ: Bool = false) {
        guard let v = QWWeightView(blocks: blocks, n: n, k: k, bits: bits,
                                   groupSize: groupSize, isRotated: isRotated,
                                   isIncumbentContainer: isIncumbentContainer, isSHVQ: isSHVQ) else { return nil }
        self.view = v
    }
    public init?(nibbles: QWTensorValue, scales: QWTensorValue? = nil, mins: QWTensorValue? = nil,
                 n: Int, k: Int, isRotated: Bool, isIncumbentContainer: Bool? = nil, isSHVQ: Bool = false) {
        self.init(blocks: nibbles, n: n, k: k, isRotated: isRotated,
                  isIncumbentContainer: isIncumbentContainer, isSHVQ: isSHVQ)
    }
}
