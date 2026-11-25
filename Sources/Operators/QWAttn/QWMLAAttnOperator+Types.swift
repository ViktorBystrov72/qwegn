// QWMLAAttnOperator+Types.swift — Weight and KV state descriptors for MLA sparse attention.
import QWCore
import QWHAL
import QWAlloc
import QWGEMM
import QWKV

extension QWMLAAttnOperator {
    public struct LayerWeights: Sendable {
        public let qANormW: QWTensorValue
        public let kvANormW: QWTensorValue
        public let kNormW: QWTensorValue
        public let kNormB: QWTensorValue
        public let wGateWeights: QWTensorValue
        public let ape: QWTensorValue?
        public let qBProj: QWWeightView
        public let kvBProj: QWWeightView
        public let vhaMix: QWTensorValue?
        public let attnSink: QWTensorValue?
        public let hasOGate: Bool
        public let oGateF16: QWTensorValue?
        public let oGateProj: QWWeightView?
        public let flashGateW: QWTensorValue?
        public let flashBias: QWTensorValue?

        public init(qANormW: QWTensorValue,
                    kvANormW: QWTensorValue,
                    kNormW: QWTensorValue,
                    kNormB: QWTensorValue,
                    wGateWeights: QWTensorValue,
                    ape: QWTensorValue?,
                    qBProj: QWWeightView,
                    kvBProj: QWWeightView,
                    vhaMix: QWTensorValue? = nil,
                    attnSink: QWTensorValue? = nil,
                    hasOGate: Bool = false,
                    oGateF16: QWTensorValue? = nil,
                    oGateProj: QWWeightView? = nil,
                    flashGateW: QWTensorValue? = nil,
                    flashBias: QWTensorValue? = nil) {
            self.qANormW = qANormW
            self.kvANormW = kvANormW
            self.kNormW = kNormW
            self.kNormB = kNormB
            self.wGateWeights = wGateWeights
            self.ape = ape
            self.qBProj = qBProj
            self.kvBProj = kvBProj
            self.vhaMix = vhaMix
            self.attnSink = attnSink
            self.hasOGate = hasOGate
            self.oGateF16 = oGateF16
            self.oGateProj = oGateProj
            self.flashGateW = flashGateW
            self.flashBias = flashBias
        }
    }

    public struct LayerState: Sendable {
        public let fullAttentionIndex: Int
        public let kPending: QWTensorValue
        public let gatePending: QWTensorValue
        public let pooledKeys: QWTensorValue
        public let kPeCache: QWTensorValue?
        public let coarseCSR: QWCoarsePoolIndex.LayerCSR?
    }

    public static let minSelectBatchRows = 32
    public static let emeansProbeClusters = 320
    public static let emeansCandCap = 10240
    public static let dsaEmeansProbeClusters = 480
    public static let dsaEmeansCandCap = 16384
    public static let selectScratchBudgetBytes = 16 << 20
    public static let flashIndexerSelHeads = 8
}
