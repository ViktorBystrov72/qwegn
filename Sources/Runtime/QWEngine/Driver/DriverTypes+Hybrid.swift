// DriverTypes+Hybrid.swift — Hybrid mHC exit-head and shared-last-layer ECHO draft
// weight descriptors for QWEngineDriver.

import Foundation
import Metal
import QWAlloc
import QWCore
import QWGEMM
import QWQuant

extension QWEngineDriver {
    struct MHCExitHeadWeights: Sendable {
        let headW: QWTensorValue
        let headB: QWTensorValue
    }

    struct EchoDraftWeights: @unchecked Sendable {
        let eNorm: QWTensorValue
        let hNorm: QWTensorValue
        let eProj: QWWeightView
        let hProj: QWWeightView
        let hcHeadW: QWTensorValue
        let hcHeadB: QWTensorValue
        let sharedHeadNorm: QWTensorValue
        let routerBias: QWTensorValue?
        let dedicatedLayer: LayerW?

        init(eNorm: QWTensorValue, hNorm: QWTensorValue,
             eProj: QWWeightView, hProj: QWWeightView,
             hcHeadW: QWTensorValue, hcHeadB: QWTensorValue,
             sharedHeadNorm: QWTensorValue, routerBias: QWTensorValue? = nil,
             dedicatedLayer: LayerW? = nil) {
            self.eNorm = eNorm; self.hNorm = hNorm; self.eProj = eProj; self.hProj = hProj
            self.hcHeadW = hcHeadW; self.hcHeadB = hcHeadB; self.sharedHeadNorm = sharedHeadNorm
            self.routerBias = routerBias; self.dedicatedLayer = dedicatedLayer
        }

        var residentBuffers: [any MTLBuffer] {
            var out: [any MTLBuffer] = [
                eNorm.region.buffer,
                hNorm.region.buffer,
                eProj.blocks.region.buffer,
                hProj.blocks.region.buffer,
                hcHeadW.region.buffer,
                hcHeadB.region.buffer,
                sharedHeadNorm.region.buffer
            ]
            if let b = routerBias {
                out.append(b.region.buffer)
            }
            if let dl = dedicatedLayer {
                out.append(contentsOf: [
                    dl.ln1.region.buffer, dl.ln2.region.buffer,
                    dl.gu.view.blocks.region.buffer, dl.down.blocks.region.buffer
                ])
                if let a = dl.attn {
                    out.append(contentsOf: [a.qkvProj.blocks.region.buffer, a.o.blocks.region.buffer])
                    if let qb = a.qkvBias { out.append(qb.region.buffer) }
                    if let ob = a.oBias { out.append(ob.region.buffer) }
                }
                if let m = dl.moe {
                    out.append(m.router.region.buffer)
                    if let rb = m.routerBias { out.append(rb.region.buffer) }
                    if let sgu = m.sharedGateUp { out.append(sgu.blocks.region.buffer) }
                    if let sdn = m.sharedDown { out.append(sdn.blocks.region.buffer) }
                    if let cgu = m.coldGateUp { out.append(cgu.blocks.region.buffer) }
                    if let cdn = m.coldDown { out.append(cdn.blocks.region.buffer) }
                }
                if let mh = dl.mhc {
                    out.append(contentsOf: [
                        mh.attnW.region.buffer, mh.attnB.region.buffer,
                        mh.ffnW.region.buffer, mh.ffnB.region.buffer
                    ])
                }
            }
            return out
        }
    }
}
