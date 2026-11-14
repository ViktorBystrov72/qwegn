// FNPLE+Reference.swift — CPU-only PLE & in-memory array rows for tests.

import Foundation
@testable import QWFlashNext


public struct FNPLE: Sendable {
    public let hidden: Int
    public let hc: Int
    public let embedDim: Int
    public let convKernel: Int
    public let dilation: Int
    public let eps: Float
    public let keyProj: [Float]     // (hc*hidden) x embedDim
    public let valueProj: [Float]   // hidden x embedDim
    public let normKey: [Float]     // hc*hidden
    public let normQuery: [Float]   // hc*hidden
    public let normConv: [Float]    // hc*hidden
    public let convWeight: [Float]  // (hc*hidden) x K  (index c*K + j)

    public init(hidden: Int, hc: Int, embedDim: Int, convKernel: Int, dilation: Int, eps: Float,
                keyProj: [Float], valueProj: [Float], normKey: [Float], normQuery: [Float], normConv: [Float],
                convWeight: [Float]) {
        self.hidden = hidden; self.hc = hc; self.embedDim = embedDim; self.convKernel = convKernel
        self.dilation = dilation; self.eps = eps; self.keyProj = keyProj; self.valueProj = valueProj
        self.normKey = normKey; self.normQuery = normQuery; self.normConv = normConv; self.convWeight = convWeight
    }

    /// `embedding`: rows x embedDim (N-gram lookup output for this sequence).
    /// `hiddenState`: rows x (hc*hidden). Returns rows x (hc*hidden).
    public func forward(embedding e: [Float], hiddenState h: [Float], rows T: Int) -> [Float] {
        let width = hc * hidden
        var kPre = [Float](repeating: 0, count: T * width)
        var vals = [Float](repeating: 0, count: T * hidden)
        for t in 0..<T {
            let et = e[(t * embedDim)..<((t + 1) * embedDim)]
            let kp = FNMath.linear(et, weight: keyProj, out: width, inDim: embedDim)
            let vp = FNMath.linear(et, weight: valueProj, out: hidden, inDim: embedDim)
            for c in 0..<width { kPre[t * width + c] = kp[c] }
            for c in 0..<hidden { vals[t * hidden + c] = vp[c] }
        }
        let keyNorm = FNMath.groupRMSNorm(kPre, rows: T, width: width, group: hidden, weight: normKey, eps: eps)
        let queryNorm = FNMath.groupRMSNorm(h, rows: T, width: width, group: hidden, weight: normQuery, eps: eps)
        // Gate per (token, stream): sign(g)*sqrt(max(|g|,1e-6)), g = <k,q>/sqrt(H).
        var gv = [Float](repeating: 0, count: T * width)
        let scale = 1 / Float(hidden).squareRoot()
        for t in 0..<T {
            for c in 0..<hc {
                var g: Float = 0
                for d in 0..<hidden { g += keyNorm[t * width + c * hidden + d] * queryNorm[t * width + c * hidden + d] }
                g *= scale
                let gate = (max(abs(g), 1e-6)).squareRoot() * (g > 0 ? 1 : (g < 0 ? -1 : 0))
                let s = FNMath.sigmoid(gate)
                for d in 0..<hidden { gv[t * width + c * hidden + d] = s * vals[t * hidden + d] }
            }
        }
        let gvNorm = FNMath.groupRMSNorm(gv, rows: T, width: width, group: hidden, weight: normConv, eps: eps)
        // Causal depthwise conv with dilation (left pad (K-1)*dilation), then residual + SiLU.
        var out = [Float](repeating: 0, count: T * width)
        for t in 0..<T {
            for c in 0..<width {
                var acc: Float = 0
                for j in 0..<convKernel {
                    let src = t - (convKernel - 1 - j) * dilation
                    if src >= 0 { acc += convWeight[c * convKernel + j] * gvNorm[src * width + c] }
                }
                out[t * width + c] = gv[t * width + c] + FNMath.silu(acc)
            }
        }
        return out
    }
}

/// Row source of the PLE N-gram table (`[totalRows, headDim]`). Rows are read on demand: the real table is
/// ~100 GB and never materialized. Reads are reference-only; an I/O failure is a fatal environment error.

/// In-memory table (tiny goldens): row-major `[totalRows, headDim]`.
public struct FNArrayRows: FNNgramRows {
    public let headDim: Int
    let table: [Float]

    public init(table: [Float], headDim: Int) {
        self.table = table; self.headDim = headDim
    }

    public func row(_ id: Int) -> [Float] {
        Array(table[(id * headDim)..<((id + 1) * headDim)])
    }
}

/// PLE N-gram embedding: per token, the rows of every hashed head concatenated (HF Qwen4ExpTextNGramEmbedding).
