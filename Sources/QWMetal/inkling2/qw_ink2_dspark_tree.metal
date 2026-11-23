// qw_ink2_dspark_tree.metal — Strided DSpark ops, KV-shared attention & top-K tree Markov kernels.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "inkling2/qw_ink2_common.metalh"

using namespace metal;

struct QWInk2DSparkMixMapParams {
    uint32_t T;
    uint32_t H;
    uint32_t tapPlaneStride;
    uint32_t useRowMap;
    float eps;
    float a0;
    float a1;
    float a2;
    float a3;
    uint32_t rowMap[16];
};

struct QWInk2DSparkCtxKVFusedParams {
    uint32_t T;
    uint32_t nKV;
    uint32_t headDim;
    uint32_t isSliding;
    uint32_t posOff;
    uint32_t ringCap;
    float eps;
    float ropeTheta;
};

struct QWInk2DSparkPropQKRStridedParams {
    uint32_t B;
    uint32_t nH;
    uint32_t nKV;
    uint32_t headDim;
    uint32_t posOff;
    uint32_t rowStride;
    float eps;
    float ropeTheta;
};

struct QWInk2DSparkAttnBatchParams {
    uint32_t B;
    uint32_t nH;
    uint32_t nKV;
    uint32_t headDim;
    uint32_t ctxStart;
    uint32_t ctxEnd;
    uint32_t sliding;
    uint32_t ringCap;
    uint32_t qStride;
    uint32_t kvPropStride;
    float scale;
};

struct QWInk2DSparkMarkovStepBatchParams {
    uint32_t P;
    uint32_t H;
    uint32_t projRows;
    uint32_t rank;
    uint32_t ctxStepIdx;
    uint32_t resetState;
    uint32_t parentStateSlots[4];
    uint32_t prevTokNodes[4];
};

struct QWInk2DSparkTopKParams {
    uint32_t stepIdx;
    uint32_t V;
    uint32_t rank;
    uint32_t partCount;
    uint32_t numParents;
    float softcap;
    uint32_t keepK[4];
    uint32_t outNodeBase[4];
};

kernel void qwg_ink2_dspark_mix_norm_map(device const bfloat* taps [[buffer(0)]],
                                         device const bfloat* normW [[buffer(1)]],
                                         device bfloat* outBF16 [[buffer(2)]],
                                         constant QWInk2DSparkMixMapParams& p [[buffer(3)]],
                                         uint tg [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint simd_id [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    const uint t = tg;
    if (t >= p.T) return;
    const uint srcT = (p.useRowMap != 0u && t < 16u) ? p.rowMap[t] : t;
    const size_t p0 = size_t(srcT) * p.H;
    const size_t ps = size_t(p.tapPlaneStride);
    const size_t p1 = ps + p0;
    const size_t p2 = 2u * ps + p0;
    const size_t p3 = 3u * ps + p0;

    float vals[16];
    uint k = 0u;
    float sq = 0.0f;
    for (uint d = tid; d < p.H; d += 256u) {
        float m = float(taps[p0 + d]) * p.a0 + float(taps[p1 + d]) * p.a1
                + float(taps[p2 + d]) * p.a2 + float(taps[p3 + d]) * p.a3;
        if (k < 16u) vals[k++] = m;
        sq += m * m;
    }
    sq = simd_sum(sq);
    threadgroup float sgSums[8];
    if (lane == 0u) sgSums[simd_id] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        float v = (lane < 8u) ? sgSums[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) sgSums[0] = rsqrt(v / float(p.H) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invRms = sgSums[0];
    device bfloat* dst = outBF16 + size_t(t) * p.H;
    k = 0u;
    for (uint d = tid; d < p.H; d += 256u) {
        float m = (k < 16u) ? vals[k++]
            : (float(taps[p0 + d]) * p.a0 + float(taps[p1 + d]) * p.a1
             + float(taps[p2 + d]) * p.a2 + float(taps[p3 + d]) * p.a3);
        dst[d] = qw_ink2_bf16((m * invRms) * float(normW[d]));
    }
}

kernel void qwg_ink2_dspark_ctx_kv_write_fused(device const bfloat* kvCtxProj [[buffer(0)]],
                                                device const bfloat* kNormCtxW [[buffer(1)]],
                                                device bfloat* kCache [[buffer(2)]],
                                                device bfloat* vCache [[buffer(3)]],
                                                constant QWInk2DSparkCtxKVFusedParams& p [[buffer(4)]],
                                                uint2 tg [[threadgroup_position_in_grid]],
                                                uint tid [[thread_index_in_threadgroup]],
                                                uint simd_id [[simdgroup_index_in_threadgroup]],
                                                uint lane [[thread_index_in_simdgroup]]) {
    const uint kvh = tg.x;
    const uint t = tg.y;
    if (kvh >= p.nKV || t >= p.T) return;
    const uint kvDim = p.nKV * p.headDim;
    const uint halfDim = p.headDim / 2u;
    const size_t srcBase = size_t(t) * (2u * kvDim) + size_t(kvh) * p.headDim;
    const uint absPos = p.posOff + t;
    const uint slot = (p.isSliding != 0u) ? (absPos % p.ringCap) : absPos;
    const size_t dstOff = size_t(slot) * kvDim + size_t(kvh) * p.headDim;

    device const bfloat* kSrc = kvCtxProj + srcBase;
    device const bfloat* vSrc = kSrc + kvDim;
    device bfloat* kDst = kCache + dstOff;
    device bfloat* vDst = vCache + dstOff;

    float sq = 0.0f;
    for (uint d = tid; d < halfDim; d += 64u) {
        float a = float(kSrc[d]);
        float b = float(kSrc[d + halfDim]);
        sq += a * a + b * b;
    }
    sq = simd_sum(sq);
    threadgroup float sgSq[2];
    if (lane == 0u) sgSq[simd_id] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        sgSq[0] = rsqrt((sgSq[0] + sgSq[1]) / float(p.headDim) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invRms = sgSq[0];
    const float fpos = float(absPos);

    for (uint d = tid; d < halfDim; d += 64u) {
        float k0 = float(qw_ink2_bf16((float(kSrc[d]) * invRms) * float(kNormCtxW[d])));
        float k1 = float(qw_ink2_bf16((float(kSrc[d + halfDim]) * invRms) * float(kNormCtxW[d + halfDim])));
        float freq = fpos * metal::pow(p.ropeTheta, -float(2u * d) / float(p.headDim));
        float c = metal::cos(freq), s = metal::sin(freq);
        kDst[d] = qw_ink2_bf16(k0 * c - k1 * s);
        kDst[d + halfDim] = qw_ink2_bf16(k1 * c + k0 * s);
        vDst[d] = vSrc[d];
        vDst[d + halfDim] = vSrc[d + halfDim];
    }
}

kernel void qwg_ink2_dspark_prop_qk_rope_strided(device bfloat* qkvBuf [[buffer(0)]],
                                                 device const bfloat* qNormW [[buffer(1)]],
                                                 device const bfloat* kNormW [[buffer(2)]],
                                                 constant QWInk2DSparkPropQKRStridedParams& p [[buffer(3)]],
                                                 uint2 tg [[threadgroup_position_in_grid]],
                                                 uint tid [[thread_index_in_threadgroup]],
                                                 uint simd_id [[simdgroup_index_in_threadgroup]],
                                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint hIdx = tg.x;
    const uint t = tg.y;
    if (hIdx >= p.nH + p.nKV || t >= p.B) return;
    const bool isQ = (hIdx < p.nH);
    const uint h = isQ ? hIdx : (hIdx - p.nH);
    const uint qDim = p.nH * p.headDim;
    const size_t rowBase = size_t(t) * p.rowStride + (isQ ? 0u : size_t(qDim));
    device bfloat* row = qkvBuf + rowBase + size_t(h) * p.headDim;
    device const bfloat* nw = isQ ? qNormW : kNormW;
    const uint halfDim = p.headDim / 2u;

    float sq = 0.0f;
    for (uint d = tid; d < halfDim; d += 64u) {
        float a = float(row[d]);
        float b = float(row[d + halfDim]);
        sq += a * a + b * b;
    }
    sq = simd_sum(sq);
    threadgroup float sgSq[2];
    if (lane == 0u) sgSq[simd_id] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        sgSq[0] = rsqrt((sgSq[0] + sgSq[1]) / float(p.headDim) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invRms = sgSq[0];
    const float fpos = float(p.posOff + t);

    for (uint d = tid; d < halfDim; d += 64u) {
        float v0 = float(qw_ink2_bf16((float(row[d]) * invRms) * float(nw[d])));
        float v1 = float(qw_ink2_bf16((float(row[d + halfDim]) * invRms) * float(nw[d + halfDim])));
        float freq = fpos * metal::pow(p.ropeTheta, -float(2u * d) / float(p.headDim));
        float c = metal::cos(freq), s = metal::sin(freq);
        row[d] = qw_ink2_bf16(v0 * c - v1 * s);
        row[d + halfDim] = qw_ink2_bf16(v1 * c + v0 * s);
    }
}

/// 4-query KV-shared 8-SG bidirectional GQA attention for DSpark drafting (`qStride`/`kvPropStride` support fused QKV).
kernel void qwg_ink2_dspark_attn_gqa_batch(device const bfloat* qProp [[buffer(0)]],
                                           device const bfloat* kProp [[buffer(1)]],
                                           device const bfloat* vProp [[buffer(2)]],
                                           device const bfloat* kCache [[buffer(3)]],
                                           device const bfloat* vCache [[buffer(4)]],
                                           device bfloat* out [[buffer(5)]],
                                           constant QWInk2DSparkAttnBatchParams& p [[buffer(6)]],
                                           uint3 tg [[threadgroup_position_in_grid]],
                                           uint simd_id [[simdgroup_index_in_threadgroup]],
                                           uint lane [[thread_index_in_simdgroup]]) {
    const uint kvh = tg.x;
    const uint bStart = tg.y * 4u;
    const uint gqa = (p.nKV > 0u) ? (p.nH / p.nKV) : 0u;
    const uint gStart = tg.z * 2u;
    const uint chunks = (p.headDim + 31u) / 32u;
    if (kvh >= p.nKV || bStart >= p.B || gStart >= gqa || chunks > 4u) return;

    const uint bCnt = min(4u, p.B - bStart);
    const uint gCnt = min(2u, gqa - gStart);
    const uint hBase = kvh * gqa + gStart;
    const uint qDim = p.nH * p.headDim;
    const uint kvDim = p.nKV * p.headDim;
    const uint qStride = (p.qStride > 0u) ? p.qStride : qDim;
    const uint kvPStride = (p.kvPropStride > 0u) ? p.kvPropStride : kvDim;

    float qVec[4][2][4];
    #pragma unroll
    for (uint bb = 0u; bb < 4u; ++bb) {
        if (bb >= bCnt) break;
        const size_t qTok = size_t(bStart + bb) * qStride;
        #pragma unroll
        for (uint g = 0u; g < 2u; ++g) {
            if (g >= gCnt) break;
            device const bfloat* qRow = qProp + qTok + size_t(hBase + g) * p.headDim;
            for (uint c = 0u; c < chunks; ++c) {
                const uint d = lane + c * 32u;
                qVec[bb][g][c] = (d < p.headDim) ? float(qRow[d]) : 0.0f;
            }
        }
    }

    threadgroup float vSh[8][132];
    threadgroup float scoreSh[8][8];

    const uint qB = simd_id >> 1u;
    const uint qG = simd_id & 1u;
    const bool qActive = (qB < bCnt && qG < gCnt);
    float mReg = -INFINITY;
    float lReg = 0.0f;
    float oReg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const uint ctxCount = (p.ctxEnd > p.ctxStart) ? (p.ctxEnd - p.ctxStart) : 0u;
    const uint totalKeys = ctxCount + p.B;
    for (uint tileBase = 0u; tileBase < totalKeys; tileBase += 8u) {
        const uint tileCnt = min(8u, totalKeys - tileBase);
        if (simd_id < tileCnt) {
            const uint keyIdx = tileBase + simd_id;
            device const bfloat* kRow;
            device const bfloat* vRow;
            if (keyIdx < ctxCount) {
                const uint j = p.ctxStart + keyIdx;
                const uint slot = (p.sliding != 0u) ? (j % p.ringCap) : j;
                const size_t kvOff = size_t(slot) * kvDim + size_t(kvh) * p.headDim;
                kRow = kCache + kvOff;
                vRow = vCache + kvOff;
            } else {
                const uint bp = keyIdx - ctxCount;
                const size_t pOff = size_t(bp) * kvPStride + size_t(kvh) * p.headDim;
                kRow = kProp + pOff;
                vRow = vProp + pOff;
            }
            float kVec[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint c = 0u; c < chunks; ++c) {
                const uint d = lane + c * 32u;
                float kv = 0.0f, vv = 0.0f;
                if (d < p.headDim) {
                    kv = float(kRow[d]);
                    vv = float(vRow[d]);
                }
                kVec[c] = kv;
                vSh[simd_id][lane + c * 32u] = vv;
            }
            #pragma unroll
            for (uint qb = 0u; qb < 4u; ++qb) {
                if (qb >= bCnt) break;
                #pragma unroll
                for (uint qg = 0u; qg < 2u; ++qg) {
                    if (qg >= gCnt) break;
                    float dotAcc = 0.0f;
                    for (uint c = 0u; c < chunks; ++c) dotAcc += qVec[qb][qg][c] * kVec[c];
                    const float s = simd_sum(dotAcc) * p.scale;
                    if (lane == 0u) scoreSh[simd_id][qb * 2u + qg] = s;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (qActive) {
            for (uint s = 0u; s < tileCnt; ++s) {
                const float sVal = scoreSh[s][simd_id];
                const float mNew = max(mReg, sVal);
                const float expPrev = (lReg > 0.0f) ? metal::exp(mReg - mNew) : 0.0f;
                const float w = metal::exp(sVal - mNew);
                lReg = lReg * expPrev + w;
                for (uint c = 0u; c < chunks; ++c) oReg[c] = oReg[c] * expPrev + w * vSh[s][lane + c * 32u];
                mReg = mNew;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (qActive) {
        const size_t oTok = size_t(bStart + qB) * qDim;
        device bfloat* oRow = out + oTok + size_t(hBase + qG) * p.headDim;
        const float invL = (lReg > 0.0f) ? (1.0f / lReg) : 0.0f;
        for (uint c = 0u; c < chunks; ++c) {
            const uint d = lane + c * 32u;
            if (d < p.headDim) oRow[d] = qw_ink2_bf16(oReg[c] * invL);
        }
    }
}

kernel void qwg_ink2_dspark_markov_step_batch(device const int32_t* treeToks [[buffer(0)]],
                                              device const bfloat* markovW1 [[buffer(1)]],
                                              device const bfloat* rnnProjW [[buffer(2)]],
                                              device const float* ctxF32 [[buffer(3)]],
                                              device const bfloat* inStatesBF16 [[buffer(4)]],
                                              device bfloat* outStatesBF16 [[buffer(5)]],
                                              device bfloat* outLatentsBF16 [[buffer(6)]],
                                              constant QWInk2DSparkMarkovStepBatchParams& p [[buffer(7)]],
                                              uint tg [[threadgroup_position_in_grid]],
                                              uint tid [[thread_index_in_threadgroup]],
                                              uint simd_id [[simdgroup_index_in_threadgroup]],
                                              uint lane [[thread_index_in_simdgroup]]) {
    if (tg >= p.P || tg >= 4u) return;
    const uint rank = min(p.rank, 256u);
    const uint prevTok = uint(max(0, treeToks[p.prevTokNodes[tg]]));
    const size_t wOff = size_t(prevTok) * rank;
    const size_t inStOff = size_t(p.parentStateSlots[tg]) * rank;
    threadgroup float sSh[256];
    threadgroup float wSh[256];
    threadgroup float rawSh[768];

    if (tid < rank) {
        sSh[tid] = (p.resetState != 0u) ? 0.0f : float(inStatesBF16[inStOff + tid]);
        wSh[tid] = float(markovW1[wOff + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint wStride = 2u * rank + 2u * p.H;
    const size_t ctxOff = size_t(p.ctxStepIdx) * p.projRows;
    for (uint baseR = 0u; baseR < p.projRows; baseR += 8u) {
        const uint r = baseR + simd_id;
        if (r < p.projRows) {
            device const bfloat* wr = rnnProjW + size_t(r) * wStride;
            float acc = 0.0f;
            for (uint k = lane; k < 2u * rank; k += 32u) {
                float inp = (k < rank) ? sSh[k] : wSh[k - rank];
                acc += inp * float(wr[k]);
            }
            float val = simd_sum(acc) + ctxF32[ctxOff + r];
            if (lane == 0u && r < 768u) rawSh[r] = float(qw_ink2_bf16(val));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < rank) {
        const float gSig = float(qw_ink2_bf16(1.0f / (1.0f + metal::exp(-rawSh[tid]))));
        const float cTanh = float(qw_ink2_bf16(qw_ink2_tanh_safe(rawSh[tid + rank])));
        const float oTanh = float(qw_ink2_bf16(qw_ink2_tanh_safe(rawSh[tid + 2u * rank])));
        const size_t outOff = size_t(tg) * rank + tid;
        outStatesBF16[outOff] = qw_ink2_bf16(gSig * sSh[tid] + (1.0f - gSig) * cTanh);
        outLatentsBF16[outOff] = qw_ink2_bf16(oTanh);
    }
}

static inline void qw_insert_topk3(thread float (&vals)[3], thread int32_t (&idxs)[3],
                                   float v, int32_t idx, uint K) {
    if (idx == 0x7fffffff) return;
    if (K >= 1u && (v > vals[0] || (v == vals[0] && idx < idxs[0]))) {
        if (K >= 3u) { vals[2] = vals[1]; idxs[2] = idxs[1]; }
        if (K >= 2u) { vals[1] = vals[0]; idxs[1] = idxs[0]; }
        vals[0] = v; idxs[0] = idx;
    } else if (K >= 2u && (v > vals[1] || (v == vals[1] && idx < idxs[1]))) {
        if (K >= 3u) { vals[2] = vals[1]; idxs[2] = idxs[1]; }
        vals[1] = v; idxs[1] = idx;
    } else if (K >= 3u && (v > vals[2] || (v == vals[2] && idx < idxs[2]))) {
        vals[2] = v; idxs[2] = idx;
    }
}

static inline void qw_topk3_reduce_tg(
        thread float (&bestV)[4][3], thread int32_t (&bestI)[4][3],
        uint numP, constant const uint32_t* keepK,
        device float* partVals, device int32_t* partIdxs,
        threadgroup float* tgV, threadgroup int32_t* tgI,
        uint tg, uint simd_id, uint lane) {
    if (lane == 0u) {
        #pragma unroll
        for (uint pIdx = 0u; pIdx < 4u; ++pIdx) {
            if (pIdx >= numP) break;
            #pragma unroll
            for (uint k = 0u; k < 3u; ++k) {
                const uint sIdx = (simd_id * 4u + pIdx) * 3u + k;
                tgV[sIdx] = bestV[pIdx][k];
                tgI[sIdx] = bestI[pIdx][k];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id < numP && lane == 0u) {
        const uint pIdx = simd_id;
        const uint K = min(keepK[pIdx], 3u);
        float wV[3] = {-INFINITY, -INFINITY, -INFINITY};
        int32_t wI[3] = {0x7fffffff, 0x7fffffff, 0x7fffffff};
        for (uint sg = 0u; sg < 8u; ++sg) {
            const uint base = (sg * 4u + pIdx) * 3u;
            for (uint k = 0u; k < K; ++k) qw_insert_topk3(wV, wI, tgV[base + k], tgI[base + k], K);
        }
        const size_t dstBase = (size_t(tg) * 4u + pIdx) * 3u;
        #pragma unroll
        for (uint k = 0u; k < 3u; ++k) {
            partVals[dstBase + k] = (k < K) ? wV[k] : -INFINITY;
            partIdxs[dstBase + k] = (k < K) ? wI[k] : 0x7fffffff;
        }
    }
}

/// Evaluates up to P<=4 sibling branches in one scan over AQ8 foldW + baseLogits, keeping top-K<=3 per branch.
kernel void qwg_ink2_dspark_markov_logits_aq8_topk_stage1(
        device const bfloat* baseLogits [[buffer(0)]],
        device const bfloat* latentBatchBF16 [[buffer(1)]],
        device const char* foldW [[buffer(2)]],
        device const bfloat* foldB [[buffer(3)]],
        device float* partVals [[buffer(4)]],
        device int32_t* partIdxs [[buffer(5)]],
        constant QWInk2DSparkTopKParams& p [[buffer(6)]],
        uint tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint simd_id [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (tg >= p.partCount || p.numParents == 0u || p.numParents > 4u) return;
    const uint nGroups = p.rank >> 5u;
    const uint rank = min(p.rank, 256u);
    threadgroup float latSh[4][256];
    threadgroup float latSumSh[4][8];

    for (uint b = 0u; b < p.numParents; ++b) {
        if (tid < 256u) {
            latSh[b][tid] = (tid < rank) ? float(latentBatchBF16[size_t(b) * p.rank + tid]) : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id < 8u) {
        for (uint b = 0u; b < p.numParents; ++b) {
            const float ls = simd_sum((simd_id * 32u + lane < rank && p.rank <= 256u)
                                      ? latSh[b][simd_id * 32u + lane] : 0.0f);
            if (lane == 0u) latSumSh[b][simd_id] = ls;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float bestV[4][3];
    int32_t bestI[4][3];
    float rejectCeil[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    #pragma unroll
    for (uint b = 0u; b < 4u; ++b) {
        #pragma unroll
        for (uint k = 0u; k < 3u; ++k) { bestV[b][k] = -INFINITY; bestI[b][k] = 0x7fffffff; }
    }

    const size_t baseOff = size_t(p.stepIdx) * p.V;
    const size_t rowBytes = size_t(nGroups) * 36u;
    const uint gridRows = p.partCount * 8u;
    const bool fastSmall = (p.rank <= 256u && (p.rank & 31u) == 0u);
    const uint g = lane >> 2u;
    const uint sub = lane & 3u;
    const bool activeLane = fastSmall && (g < nGroups);
    const uint b0 = activeLane ? (g * 32u + sub * 4u) : 0u;
    const uint b1 = activeLane ? (b0 + 16u) : 0u;

    float4 lv0[4], lv1[4];
    float latSum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (uint b = 0u; b < 4u; ++b) {
        if (b < p.numParents && activeLane) {
            lv0[b] = float4(latSh[b][b0], latSh[b][b0 + 1u], latSh[b][b0 + 2u], latSh[b][b0 + 3u]);
            lv1[b] = float4(latSh[b][b1], latSh[b][b1 + 1u], latSh[b][b1 + 2u], latSh[b][b1 + 3u]);
            if (sub == 0u) latSum[b] = latSumSh[b][g];
        } else {
            lv0[b] = float4(0.0f);
            lv1[b] = float4(0.0f);
        }
    }

    for (uint v = tg * 8u + simd_id; v < p.V; v += gridRows) {
        device const char* rowPtr = foldW + size_t(v) * rowBytes;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (fastSmall) {
            if (g < nGroups) {
                device const char* gp = rowPtr + size_t(g) * 36u;
                const half2 sb = *(device const half2*)gp;
                device const uint* qp = (device const uint*)(gp + 4u);
                const float4 f0 = float4(as_type<uchar4>(qp[sub]));
                const float4 f1 = float4(as_type<uchar4>(qp[sub + 4u]));
                const float sc = float(sb.x);
                const float mn = float(sb.y);
                #pragma unroll
                for (uint b = 0u; b < 4u; ++b) {
                    if (b >= p.numParents) break;
                    acc[b] = sc * (dot(f0, lv0[b]) + dot(f1, lv1[b]));
                    if (sub == 0u) acc[b] += mn * latSum[b];
                }
            }
        } else {
            for (uint gg = 0u; gg < nGroups; ++gg) {
                device const char* gp = rowPtr + size_t(gg) * 36u;
                const half2 sb = *(device const half2*)gp;
                const float wv = float(sb.x) * float(((device const uchar*)(gp + 4u))[lane]) + float(sb.y);
                #pragma unroll
                for (uint b = 0u; b < 4u; ++b) {
                    if (b >= p.numParents) break;
                    acc[b] += wv * float(latentBatchBF16[size_t(b) * p.rank + gg * 32u + lane]);
                }
            }
        }
        float dots[4];
        #pragma unroll
        for (uint b = 0u; b < 4u; ++b) dots[b] = (b < p.numParents) ? simd_sum(acc[b]) : 0.0f;
        if (lane == 0u) {
            const float fBias = float(foldB[v]);
            const float baseL = float(baseLogits[baseOff + v]);
            const int32_t iv = int32_t(v);
            #pragma unroll
            for (uint b = 0u; b < 4u; ++b) {
                if (b >= p.numParents) break;
                const uint K = min(p.keepK[b], 3u);
                if (K == 0u) continue;
                const float bias = float(qw_ink2_bf16(dots[b] + fBias));
                float comb = float(qw_ink2_bf16(baseL + bias));
                if (p.softcap > 0.0f) {
                    if (comb < rejectCeil[b]) continue;
                    const float capped = float(qw_ink2_bf16(qw_ink2_tanh_safe(comb / p.softcap) * p.softcap));
                    if (capped < bestV[b][K - 1u] && comb > rejectCeil[b]) rejectCeil[b] = comb;
                    comb = capped;
                }
                qw_insert_topk3(bestV[b], bestI[b], comb, iv, K);
            }
        }
    }
    threadgroup float tgV[8 * 4 * 3];
    threadgroup int32_t tgI[8 * 4 * 3];
    qw_topk3_reduce_tg(bestV, bestI, p.numParents, p.keepK, partVals, partIdxs, tgV, tgI, tg, simd_id, lane);
}

/// BF16 foldW fallback for generalized top-K tree drafting when rank % 32 != 0.
kernel void qwg_ink2_dspark_markov_logits_bf16_topk_stage1(
        device const bfloat* baseLogits [[buffer(0)]],
        device const bfloat* latentBatchBF16 [[buffer(1)]],
        device const bfloat* foldW [[buffer(2)]],
        device const bfloat* foldB [[buffer(3)]],
        device float* partVals [[buffer(4)]],
        device int32_t* partIdxs [[buffer(5)]],
        constant QWInk2DSparkTopKParams& p [[buffer(6)]],
        uint tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint simd_id [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (tg >= p.partCount || p.numParents == 0u || p.numParents > 4u) return;
    const uint rank = min(p.rank, 256u);
    threadgroup float latSh[4][256];
    for (uint b = 0u; b < p.numParents; ++b) {
        if (tid < rank && tid < 256u) latSh[b][tid] = float(latentBatchBF16[size_t(b) * p.rank + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float bestV[4][3];
    int32_t bestI[4][3];
    float rejectCeil[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    #pragma unroll
    for (uint b = 0u; b < 4u; ++b) {
        #pragma unroll
        for (uint k = 0u; k < 3u; ++k) { bestV[b][k] = -INFINITY; bestI[b][k] = 0x7fffffff; }
    }
    const size_t baseOff = size_t(p.stepIdx) * p.V;
    const uint gridRows = p.partCount * 8u;
    for (uint v = tg * 8u + simd_id; v < p.V; v += gridRows) {
        device const bfloat* fw = foldW + size_t(v) * p.rank;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if ((rank & 7u) == 0u) {
            for (uint k = lane * 8u; k < rank; k += 256u) {
                const ushort4 w0 = *(device const ushort4*)(fw + k);
                const ushort4 w1 = *(device const ushort4*)(fw + k + 4u);
                const float4 wf0 = float4(as_type<bfloat>(w0.x), as_type<bfloat>(w0.y),
                                          as_type<bfloat>(w0.z), as_type<bfloat>(w0.w));
                const float4 wf1 = float4(as_type<bfloat>(w1.x), as_type<bfloat>(w1.y),
                                          as_type<bfloat>(w1.z), as_type<bfloat>(w1.w));
                #pragma unroll
                for (uint b = 0u; b < 4u; ++b) {
                    if (b >= p.numParents) break;
                    const float4 lv0 = float4(latSh[b][k], latSh[b][k + 1u], latSh[b][k + 2u], latSh[b][k + 3u]);
                    const float4 lv1 = float4(latSh[b][k + 4u], latSh[b][k + 5u], latSh[b][k + 6u], latSh[b][k + 7u]);
                    acc[b] += dot(wf0, lv0) + dot(wf1, lv1);
                }
            }
        } else {
            for (uint k = lane; k < rank; k += 32u) {
                const float wv = float(fw[k]);
                #pragma unroll
                for (uint b = 0u; b < 4u; ++b) {
                    if (b >= p.numParents) break;
                    acc[b] += wv * latSh[b][k];
                }
            }
        }
        float dots[4];
        #pragma unroll
        for (uint b = 0u; b < 4u; ++b) dots[b] = (b < p.numParents) ? simd_sum(acc[b]) : 0.0f;
        if (lane == 0u) {
            const float fBias = float(foldB[v]);
            const float baseL = float(baseLogits[baseOff + v]);
            const int32_t iv = int32_t(v);
            #pragma unroll
            for (uint b = 0u; b < 4u; ++b) {
                if (b >= p.numParents) break;
                const uint K = min(p.keepK[b], 3u);
                if (K == 0u) continue;
                const float bias = float(qw_ink2_bf16(dots[b] + fBias));
                float comb = float(qw_ink2_bf16(baseL + bias));
                if (p.softcap > 0.0f) {
                    if (comb < rejectCeil[b]) continue;
                    const float capped = float(qw_ink2_bf16(qw_ink2_tanh_safe(comb / p.softcap) * p.softcap));
                    if (capped < bestV[b][K - 1u] && comb > rejectCeil[b]) rejectCeil[b] = comb;
                    comb = capped;
                }
                qw_insert_topk3(bestV[b], bestI[b], comb, iv, K);
            }
        }
    }
    threadgroup float tgV[8 * 4 * 3];
    threadgroup int32_t tgI[8 * 4 * 3];
    qw_topk3_reduce_tg(bestV, bestI, p.numParents, p.keepK, partVals, partIdxs, tgV, tgI, tg, simd_id, lane);
}

kernel void qwg_ink2_dspark_markov_topk_final(device const float* partVals [[buffer(0)]],
                                              device const int32_t* partIdxs [[buffer(1)]],
                                              device int32_t* treeToks [[buffer(2)]],
                                              constant QWInk2DSparkTopKParams& p [[buffer(3)]],
                                              uint tg [[threadgroup_position_in_grid]],
                                              uint tid [[thread_index_in_threadgroup]],
                                              uint simd_id [[simdgroup_index_in_threadgroup]],
                                              uint lane [[thread_index_in_simdgroup]]) {
    const uint b = tg;
    if (b >= p.numParents || b >= 4u) return;
    const uint K = min(p.keepK[b], 3u);
    float bV[3] = {-INFINITY, -INFINITY, -INFINITY};
    int32_t bI[3] = {0x7fffffff, 0x7fffffff, 0x7fffffff};
    for (uint i = tid; i < p.partCount; i += 256u) {
        const size_t base = (size_t(i) * 4u + b) * 3u;
        for (uint k = 0u; k < K; ++k) qw_insert_topk3(bV, bI, partVals[base + k], partIdxs[base + k], K);
    }

    threadgroup float tgV[256 * 3];
    threadgroup int32_t tgI[256 * 3];
    for (uint k = 0u; k < 3u; ++k) { tgV[tid * 3u + k] = bV[k]; tgI[tid * 3u + k] = bI[k]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0u && lane < 32u) {
        float sgV[3] = {-INFINITY, -INFINITY, -INFINITY};
        int32_t sgI[3] = {0x7fffffff, 0x7fffffff, 0x7fffffff};
        for (uint src = lane; src < 256u; src += 32u) {
            for (uint k = 0u; k < K; ++k) qw_insert_topk3(sgV, sgI, tgV[src * 3u + k], tgI[src * 3u + k], K);
        }
        for (uint k = 0u; k < 3u; ++k) { tgV[lane * 3u + k] = sgV[k]; tgI[lane * 3u + k] = sgI[k]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float winV[3] = {-INFINITY, -INFINITY, -INFINITY};
        int32_t winI[3] = {0x7fffffff, 0x7fffffff, 0x7fffffff};
        for (uint s = 0u; s < 32u; ++s) {
            for (uint k = 0u; k < K; ++k) qw_insert_topk3(winV, winI, tgV[s * 3u + k], tgI[s * 3u + k], K);
        }
        const uint dstBase = p.outNodeBase[b];
        for (uint k = 0u; k < K; ++k) {
            treeToks[dstBase + k] = (winI[k] == 0x7fffffff) ? 0 : winI[k];
        }
    }
}
