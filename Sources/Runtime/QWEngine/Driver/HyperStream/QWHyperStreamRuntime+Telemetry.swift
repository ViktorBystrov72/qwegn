// QWHyperStreamRuntime+Telemetry.swift — Deduplicated weight and PLE residency accounting for Flash-Next.

import Foundation
import Metal
import QWAlloc
import QWHyperPLE

extension QWHyperStreamRuntime {
    /// Sums unique non-arena `MTLBuffer` lengths owned by Flash-Next weights, embeddings, PLE tables, and MTP.
    static func standaloneWeightBytes(
        arenaBuffers: [any MTLBuffer],
        layers: [FNGPULayerWeights],
        head: FNGPUHead,
        embedTable: FNGPUEmbedTable,
        gpuPLETable: FNGPUPLETable?,
        mtp: FNGPUMTP?
    ) -> Int {
        var candidates: [any MTLBuffer] = []

        func collectTensor(_ tensor: QWTensorValue) {
            candidates.append(tensor.region.buffer)
        }

        func collectWeight(_ weight: QWWeightView) {
            collectTensor(weight.blocks)
        }

        func collectHC(_ hc: FNGPUHC) {
            collectTensor(hc.norm)
            collectTensor(hc.injectBF16)
            collectWeight(hc.down)
            collectWeight(hc.up)
        }

        func collectHead(_ headBlock: FNGPUHead) {
            collectWeight(headBlock.weight)
            collectWeight(headBlock.mixerDown)
            collectWeight(headBlock.mixerUp)
            collectTensor(headBlock.normWeight)
        }

        func collectLayer(_ layer: FNGPULayerWeights) {
            switch layer.mixer {
            case .gdn(let gdn):
                collectWeight(gdn.qkv)
                collectWeight(gdn.z)
                if let qkvZ = gdn.qkvZ { collectWeight(qkvZ) }
                collectWeight(gdn.out)
                collectWeight(gdn.inProjBA)
                collectTensor(gdn.conv)
                collectTensor(gdn.normWeight)
                collectTensor(gdn.aLog)
                collectTensor(gdn.dtBias)
                collectTensor(gdn.inProjB)
                collectTensor(gdn.inProjA)
                if let ba32 = gdn.inProjBA32 { collectTensor(ba32) }
            case .attention(let attn):
                collectWeight(attn.bank)
                collectWeight(attn.oProj)
                collectTensor(attn.qNorm)
                collectTensor(attn.kNorm)
                collectTensor(attn.iqNorm)
                collectTensor(attn.ikNorm)
            }
            collectHC(layer.attnHC)
            collectHC(layer.mlpHC)
            collectTensor(layer.moe.gateAndSharedBF16)
            collectWeight(layer.moe.gateUp)
            collectWeight(layer.moe.down)
            collectWeight(layer.moe.sharedGateUp)
            collectWeight(layer.moe.sharedDown)
            collectTensor(layer.moe.sharedGateScalar)
            if let latentD = layer.moe.latentD { collectWeight(latentD) }
            if let latentU = layer.moe.latentU { collectWeight(latentU) }
            if let latentLabels = layer.moe.latentLabels { collectTensor(latentLabels) }
            if let ple = layer.ple {
                collectWeight(ple.keyValueBank)
                collectWeight(ple.keyBank)
                collectWeight(ple.valueBank)
                collectTensor(ple.normKey)
                collectTensor(ple.normQuery)
                collectTensor(ple.normConv)
                collectTensor(ple.convW)
            }
        }

        for layer in layers { collectLayer(layer) }
        collectHead(head)
        switch embedTable.storage {
        case .nativeAffine(let blocks, _, _): collectTensor(blocks)
        case .bf16(let table): collectTensor(table)
        }
        if let pleTable = gpuPLETable { candidates.append(pleTable.buffer) }
        if let mtp {
            collectTensor(mtp.preNormEmb)
            collectTensor(mtp.preNormHid)
            collectWeight(mtp.fcEmbedding)
            collectWeight(mtp.fcHidden)
            collectLayer(mtp.layer)
            collectHead(mtp.mtpHead)
        }
        return standaloneUniqueBufferBytes(arenaBuffers: arenaBuffers, candidateBuffers: candidates)
    }

    static func standaloneUniqueBufferBytes(
        arenaBuffers: [any MTLBuffer],
        candidateBuffers: [any MTLBuffer]
    ) -> Int {
        var seen = Set(arenaBuffers.map(ObjectIdentifier.init))
        var total = 0
        for buffer in candidateBuffers {
            if seen.insert(ObjectIdentifier(buffer)).inserted {
                total += buffer.length
            }
        }
        return total
    }
}
