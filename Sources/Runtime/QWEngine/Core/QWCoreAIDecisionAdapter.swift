// QWCoreAIDecisionAdapter.swift — CoreAI ANE QWDecisionExecutionBackend adapter (ADR-004).

import Foundation
import QWCore
import QWDecision

#if canImport(CoreAI)
import CoreAI
#endif

/// Requested compute unit preference for loading a decision sidecar.
public enum QWDecisionComputeUnit: String, Sendable, Codable, CaseIterable {
    case auto
    case metal
    case ane
}

/// Evaluates `.aimodel` decision bundles on Apple Neural Engine / CoreAI without GPU preemption.
public final class QWCoreAIDecisionAdapter: QWDecisionExecutionBackend, @unchecked Sendable {
    public let decisionConfig: QWDecisionModelConfig
    public let maxContextLength: Int
    public let modelPath: String
    public let activeComputeUnit: String
    public let chainOrder: [String]
    private let runnerBox: any Sendable

    private init(
        decisionConfig: QWDecisionModelConfig,
        maxContextLength: Int,
        modelPath: String,
        activeComputeUnit: String,
        chainOrder: [String],
        runnerBox: any Sendable
    ) {
        self.decisionConfig = decisionConfig
        self.maxContextLength = maxContextLength
        self.modelPath = modelPath
        self.activeComputeUnit = activeComputeUnit
        self.chainOrder = chainOrder
        self.runnerBox = runnerBox
    }

    public static func load(
        bundleURL: URL,
        decisionConfig: QWDecisionModelConfig? = nil,
        computeUnit: QWDecisionComputeUnit = .ane
    ) async throws -> QWCoreAIDecisionAdapter {
        #if canImport(CoreAI)
        if #available(macOS 27.0, *) {
            let meta = try loadBundleMetadata(bundleURL: bundleURL, overrideConfig: decisionConfig)
            let worker = try await CoreAIRunner(
                assetURL: meta.assetURL,
                lMax: meta.lMax,
                kMax: meta.kMax,
                padId: meta.padId,
                chainRoutes: meta.chainRoutes,
                computeUnit: computeUnit
            )
            return QWCoreAIDecisionAdapter(
                decisionConfig: meta.config,
                maxContextLength: meta.lMax,
                modelPath: bundleURL.path,
                activeComputeUnit: computeUnit == .metal ? "gpu" : "ane",
                chainOrder: meta.chainOrder,
                runnerBox: worker
            )
        }
        #endif
        throw QWDecisionError.backendFailed("CoreAI ANE adapter requires macOS 27.0+ with CoreAI.framework")
    }

    public func evaluate(_ request: QWDecisionBackendBatchRequest) async throws -> QWDecisionBackendOutput {
        #if canImport(CoreAI)
        if #available(macOS 27.0, *), let worker = runnerBox as? CoreAIRunner {
            return try await worker.evaluate(request)
        }
        #endif
        throw QWDecisionError.backendFailed("CoreAI ANE adapter requires macOS 27.0+ with CoreAI.framework")
    }
}

private struct CoreAIBundleMeta {
    let assetURL: URL
    let config: QWDecisionModelConfig
    let lMax: Int
    let kMax: Int
    let padId: Int32
    let chainOrder: [String]
    let chainRoutes: [String: Int32]
}

private func loadBundleMetadata(
    bundleURL: URL,
    overrideConfig: QWDecisionModelConfig?
) throws -> CoreAIBundleMeta {
    let fm = FileManager.default
    var resolvedAsset = bundleURL
    if bundleURL.pathExtension.lowercased() != "aimodel" {
        if let items = try? fm.contentsOfDirectory(at: bundleURL, includingPropertiesForKeys: nil),
           let first = items.first(where: { $0.pathExtension.lowercased() == "aimodel" }) {
            resolvedAsset = first
        } else {
            let nested = bundleURL.appendingPathComponent("model.aimodel")
            if fm.fileExists(atPath: nested.path) { resolvedAsset = nested }
        }
    }
    guard fm.fileExists(atPath: resolvedAsset.path) else {
        throw QWDecisionError.backendFailed("CoreAI .aimodel bundle not found under \(bundleURL.path)")
    }

    let parentDir = resolvedAsset.deletingLastPathComponent()
    let provCandidates = [
        parentDir.appendingPathComponent("combined_provenance.json"),
        bundleURL.appendingPathComponent("combined_provenance.json"),
    ]
    var lMax = 1024
    var kMax = 128
    var chainOrder: [String] = ["default"]
    var chainRoutes: [String: Int32] = ["default": 0, "base": 0]

    for cand in provCandidates {
        if let data = try? Data(contentsOf: cand),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let shape = json["shape"] as? [String: Any] {
                if let l = shape["L_max"] as? Int, l > 0 { lMax = l }
                if let k = shape["K_max"] as? Int, k > 0 { kMax = k }
            }
            if let heads = json["heads"] as? [String: Any], !heads.isEmpty {
                let orderedKeys = Array(heads.keys).sorted()
                chainOrder = orderedKeys
                for (idx, hKey) in orderedKeys.enumerated() {
                    chainRoutes[hKey] = Int32(idx)
                    if let hObj = heads[hKey] as? [String: Any],
                       let tasks = hObj["tasks"] as? [String] {
                        for task in tasks { chainRoutes[task] = Int32(idx) }
                    }
                }
            }
            break
        }
    }

    let config: QWDecisionModelConfig
    if let overrideConfig {
        config = overrideConfig
    } else {
        config = QWDecisionModelConfig(
            framing: .optionMarkers(
                layout: .questionOptionsState,
                markerToken: "[MASK]",
                bosToken: "[CLS]",
                maxLength: lMax,
                imageTextLength: nil,
                audioTextLength: nil
            ),
            topology: .bidirectionalEncoder(isolateMediaPrefix: false),
            head: .markerScorer(hiddenSize: 1024, headLayers: 2, chains: chainOrder),
            defaultScoreReadout: "mean"
        )
    }
    return CoreAIBundleMeta(
        assetURL: resolvedAsset,
        config: config,
        lMax: lMax,
        kMax: kMax,
        padId: 0,
        chainOrder: chainOrder,
        chainRoutes: chainRoutes
    )
}

#if canImport(CoreAI)
@available(macOS 27.0, *)
private actor CoreAIRunner {
    private let fn: InferenceFunction
    private let lMax: Int
    private let kMax: Int
    private let padId: Int32
    private let chainRoutes: [String: Int32]
    private var idsBuf: [Int32]
    private var attBuf: [Int32]
    private var mposBuf: [Int32]
    private var mmaskBuf: [Bool]

    init(
        assetURL: URL,
        lMax: Int,
        kMax: Int,
        padId: Int32,
        chainRoutes: [String: Int32],
        computeUnit: QWDecisionComputeUnit
    ) async throws {
        self.lMax = lMax
        self.kMax = kMax
        self.padId = padId
        self.chainRoutes = chainRoutes
        let opts: SpecializationOptions
        switch computeUnit {
        case .metal:
            opts = SpecializationOptions(preferredComputeUnitKind: .gpu)
        case .auto, .ane:
            opts = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        }
        let model = try await AIModel.specialize(
            contentsOf: assetURL,
            options: opts,
            cache: .default,
            cachePolicy: .persistent
        )
        guard let fnName = model.functionNames.first,
              let loadedFn = try model.loadFunction(named: fnName) else {
            throw QWDecisionError.backendFailed("CoreAI bundle \(assetURL.lastPathComponent) has no inference function")
        }
        self.fn = loadedFn
        self.idsBuf = [Int32](repeating: padId, count: lMax)
        self.attBuf = [Int32](repeating: 0, count: lMax)
        self.mposBuf = [Int32](repeating: 0, count: kMax)
        self.mmaskBuf = [Bool](repeating: false, count: kMax)
    }

    func evaluate(_ request: QWDecisionBackendBatchRequest) async throws -> QWDecisionBackendOutput {
        let t0 = ContinuousClock.now
        var slotOutputs = [QWDecisionSlotOutput](repeating: QWDecisionSlotOutput(), count: request.slots.count)
        var slotsBySeq: [Int: [Int]] = [:]
        for (idx, slot) in request.slots.enumerated() {
            slotsBySeq[slot.sequenceIndex, default: []].append(idx)
        }

        for seqIdx in slotsBySeq.keys.sorted() {
            guard seqIdx >= 0 && seqIdx < request.sequences.count else {
                throw QWDecisionError.invalidQuestion("sequence index \(seqIdx) out of bounds")
            }
            let seq = request.sequences[seqIdx]
            guard seq.tokenIds.count <= lMax else {
                throw QWDecisionError.contextOverflow(tokens: seq.tokenIds.count, limit: lMax)
            }
            guard seq.markerPositions.count <= kMax else {
                throw QWDecisionError.invalidQuestion("marker count \(seq.markerPositions.count) exceeds CoreAI K_max=\(kMax)")
            }

            let n = seq.tokenIds.count
            for i in 0..<n {
                idsBuf[i] = Int32(truncatingIfNeeded: seq.tokenIds[i])
                attBuf[i] = 1
            }
            if n < lMax {
                for i in n..<lMax {
                    idsBuf[i] = padId
                    attBuf[i] = 0
                }
            }
            let k = seq.markerPositions.count
            for i in 0..<k {
                mposBuf[i] = Int32(seq.markerPositions[i])
                mmaskBuf[i] = true
            }
            if k < kMax {
                for i in k..<kMax {
                    mposBuf[i] = 0
                    mmaskBuf[i] = false
                }
            }

            let qtype: Int32 = (k == 2) ? 2 : 0
            let headIdx: Int32 = chainRoutes[seq.chain] ?? 0
            let logits = try await runOnePass(qtype: qtype, headIdx: headIdx)
            let markerCount = max(0, min(k, logits.count))
            let markerSlice = Array(logits.prefix(markerCount))
            for slotIdx in (slotsBySeq[seqIdx] ?? []) {
                slotOutputs[slotIdx] = QWDecisionSlotOutput(markerLogits: markerSlice)
            }
        }

        let elapsed = t0.duration(to: ContinuousClock.now)
        let latencyMs = Double(elapsed.components.seconds) * 1000.0
            + Double(elapsed.components.attoseconds) / 1e15
        let totalPromptTokens = request.sequences.reduce(0) { $0 + $1.tokenIds.count }
        return QWDecisionBackendOutput(
            slotOutputs: slotOutputs,
            promptTokens: totalPromptTokens,
            latencyMs: (latencyMs * 100.0).rounded() / 100.0
        )
    }

    private func runOnePass(qtype: Int32, headIdx: Int32) async throws -> [Float] {
        let ids = idsBuf
        let att = attBuf
        let mpos = mposBuf
        let mmask = mmaskBuf
        let qtArr = [qtype]
        let hiArr = [headIdx]

        var inputs = InferenceFunction.Inputs()
        inputs.insert(NDArray.View(span: ids.span, shape: [1, lMax]), for: "input_ids")
        inputs.insert(NDArray.View(span: att.span, shape: [1, lMax]), for: "attention_mask")
        inputs.insert(NDArray.View(span: mpos.span, shape: [1, kMax]), for: "marker_pos")
        inputs.insert(NDArray.View(span: mmask.span, shape: [1, kMax]), for: "marker_mask")
        inputs.insert(NDArray.View(span: qtArr.span, shape: [1]), for: "qtype")
        inputs.insert(NDArray.View(span: hiArr.span, shape: [1]), for: "head_idx")

        var outputs = try await fn.run(inputs: inputs)
        guard let nd = outputs.remove("logits")?.ndArray else {
            throw QWDecisionError.backendFailed("CoreAI function did not return 'logits'")
        }
        var total = 1
        for dim in nd.shape { total *= dim }
        guard total > 0 else { return [] }
        switch nd.scalarType {
        case .float32:
            var out = [Float](repeating: 0, count: total)
            nd.view(as: Float.self).withUnsafePointer { ptr, _, _ in
                for i in 0..<total { out[i] = ptr[i] }
            }
            return out
        case .float16:
            var out = [Float](repeating: 0, count: total)
            nd.view(as: Float16.self).withUnsafePointer { ptr, _, _ in
                for i in 0..<total { out[i] = Float(ptr[i]) }
            }
            return out
        default:
            throw QWDecisionError.backendFailed("unsupported CoreAI logits scalar type \(nd.scalarType)")
        }
    }
}
#endif
