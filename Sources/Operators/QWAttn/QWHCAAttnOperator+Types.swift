// QWHCAAttnOperator+Types.swift — Weight and ring-state descriptors for HCA attention.
import QWCore
import QWAlloc
import QWGEMM

extension QWHCAAttnOperator {
    public struct LayerWeights: Sendable {
        public let wFusedIn: QWWeightView?
        public let qNormW: QWTensorValue
        public let kvNormW: QWTensorValue
        public let compNormW: QWTensorValue
        public let ape: QWTensorValue
        public let wQUpGate: QWWeightView?
        public let woA: QWWeightView?
        public let woARawF16: QWTensorValue?
        public let woB: QWWeightView?
        public let vhaMix: QWTensorValue?
        public let attnSink: QWTensorValue?
        public let lsaWQIdx: QWTensorValue?
        public let lsaWKIdx: QWTensorValue?

        public init(wFusedIn: QWWeightView? = nil,
                    qNormW: QWTensorValue,
                    kvNormW: QWTensorValue,
                    compNormW: QWTensorValue,
                    ape: QWTensorValue,
                    wQUpGate: QWWeightView? = nil,
                    woA: QWWeightView? = nil,
                    woARawF16: QWTensorValue? = nil,
                    woB: QWWeightView? = nil,
                    vhaMix: QWTensorValue? = nil,
                    attnSink: QWTensorValue? = nil,
                    lsaWQIdx: QWTensorValue? = nil,
                    lsaWKIdx: QWTensorValue? = nil) {
            self.wFusedIn = wFusedIn
            self.qNormW = qNormW
            self.kvNormW = kvNormW
            self.compNormW = compNormW
            self.ape = ape
            self.wQUpGate = wQUpGate
            self.woA = woA
            self.woARawF16 = woARawF16
            self.woB = woB
            self.vhaMix = vhaMix
            self.attnSink = attnSink
            self.lsaWQIdx = lsaWQIdx
            self.lsaWKIdx = lsaWKIdx
        }
    }

    public struct LayerState: Sendable {
        public let hcaLayerIndex: Int
        public let swaKV: QWTensorValue
        public let compKvPend: QWTensorValue
        public let compScorePend: QWTensorValue
        public let compKV: QWTensorValue
        public let pooledKeys: QWTensorValue?
        public let coarseCSR: QWCoarsePoolIndex.LayerCSR?
    }

    public static let splitMaxRows = 15
    public static let maxSelectBatchRows = 32
    public static let emeansProbeClusters = 128
    public static let emeansCandCap = 16384

}
