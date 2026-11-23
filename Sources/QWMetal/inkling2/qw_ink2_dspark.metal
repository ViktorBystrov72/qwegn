// qw_ink2_dspark.metal — Inkling-2 DSpark (DFlare mix, bidirectional draft GQA, RNN Markov head).
#include <metal_stdlib>
#include <metal_simdgroup>
#include "inkling2/qw_ink2_common.metalh"

using namespace metal;

struct QWInk2DSparkMixParams {
    uint32_t T;
    uint32_t H;
    uint32_t tapPlaneStride;
    float eps;
    float a0;
    float a1;
    float a2;
    float a3;
};

struct QWInk2DSparkCtxKVParams {
    uint32_t T;
    uint32_t nKV;
    uint32_t headDim;
    uint32_t isSliding;
    uint32_t posOff;
    uint32_t ringCap;
    float eps;
    float ropeTheta;
};

struct QWInk2DSparkAddNormParams {
    uint32_t rows;
    uint32_t dim;
    uint32_t hasRes;
    uint32_t writeRes;
    float eps;
};

struct QWInk2DSparkPropQKRParams {
    uint32_t B;
    uint32_t nH;
    uint32_t nKV;
    uint32_t headDim;
    uint32_t posOff;
    float eps;
    float ropeTheta;
};

struct QWInk2DSparkAttnParams {
    uint32_t B;
    uint32_t nH;
    uint32_t nKV;
    uint32_t headDim;
    uint32_t ctxStart;
    uint32_t ctxEnd;
    uint32_t sliding;
    uint32_t ringCap;
    float scale;
};

struct QWInk2DSparkMarkovCtxParams {
    uint32_t count;
    uint32_t H;
    uint32_t projRows;
    uint32_t rank;
};

struct QWInk2DSparkMarkovStepParams {
    uint32_t stepIdx;
    uint32_t H;
    uint32_t projRows;
    uint32_t rank;
    uint32_t resetState;
};

struct QWInk2DSparkLogitsParams {
    uint32_t stepIdx;
    uint32_t V;
    uint32_t rank;
    uint32_t partCount;
    float softcap;
};

kernel void qwg_ink2_dspark_mix_norm(device const bfloat* taps [[buffer(0)]],
                                     device const bfloat* normW [[buffer(1)]],
                                     device bfloat* outBF16 [[buffer(2)]],
                                     constant QWInk2DSparkMixParams& p [[buffer(3)]],
                                     uint tg [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint simd_id [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    const uint t = tg;
    if (t >= p.T) return;
    const size_t p0 = size_t(t) * p.H;
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

kernel void qwg_ink2_dspark_ctx_kv_write(device const bfloat* kCtxProj [[buffer(0)]],
                                         device const bfloat* vCtxProj [[buffer(1)]],
                                         device const bfloat* kNormCtxW [[buffer(2)]],
                                         device bfloat* kCache [[buffer(3)]],
                                         device bfloat* vCache [[buffer(4)]],
                                         constant QWInk2DSparkCtxKVParams& p [[buffer(5)]],
                                         uint2 tg [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint simd_id [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    const uint kvh = tg.x;
    const uint t = tg.y;
    if (kvh >= p.nKV || t >= p.T) return;
    const uint kvDim = p.nKV * p.headDim;
    const uint halfDim = p.headDim / 2u;
    const size_t srcOff = size_t(t) * kvDim + size_t(kvh) * p.headDim;
    const uint absPos = p.posOff + t;
    const uint slot = (p.isSliding != 0u) ? (absPos % p.ringCap) : absPos;
    const size_t dstOff = size_t(slot) * kvDim + size_t(kvh) * p.headDim;

    device const bfloat* kSrc = kCtxProj + srcOff;
    device const bfloat* vSrc = vCtxProj + srcOff;
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

kernel void qwg_ink2_dspark_add_norm(device const bfloat* x [[buffer(0)]],
                                     device bfloat* res [[buffer(1)]],
                                     device const bfloat* normW [[buffer(2)]],
                                     device bfloat* normOut [[buffer(3)]],
                                     constant QWInk2DSparkAddNormParams& p [[buffer(4)]],
                                     uint tg [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint simd_id [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    const uint r = tg;
    if (r >= p.rows) return;
    const size_t base = size_t(r) * p.dim;
    float vals[16];
    uint k = 0u;
    float sq = 0.0f;
    for (uint d = tid; d < p.dim; d += 256u) {
        float v = float(x[base + d]);
        if (p.hasRes != 0u) v += float(res[base + d]);
        if (k < 16u) vals[k++] = v;
        if (p.writeRes != 0u) res[base + d] = qw_ink2_bf16(v);
        sq += v * v;
    }
    sq = simd_sum(sq);
    threadgroup float sgSums[8];
    if (lane == 0u) sgSums[simd_id] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        float val = (lane < 8u) ? sgSums[lane] : 0.0f;
        val = simd_sum(val);
        if (lane == 0u) sgSums[0] = rsqrt(val / float(p.dim) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invRms = sgSums[0];
    k = 0u;
    for (uint d = tid; d < p.dim; d += 256u) {
        float v = (k < 16u) ? vals[k++]
            : (float(x[base + d]) + (p.hasRes != 0u ? float(res[base + d]) : 0.0f));
        normOut[base + d] = qw_ink2_bf16((v * invRms) * float(normW[d]));
    }
}

kernel void qwg_ink2_dspark_prop_qk_rope(device bfloat* qProp [[buffer(0)]],
                                         device const bfloat* qNormW [[buffer(1)]],
                                         device bfloat* kProp [[buffer(2)]],
                                         device const bfloat* kNormW [[buffer(3)]],
                                         constant QWInk2DSparkPropQKRParams& p [[buffer(4)]],
                                         uint2 tg [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint simd_id [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    const uint hIdx = tg.x;
    const uint t = tg.y;
    if (hIdx >= p.nH + p.nKV || t >= p.B) return;
    const bool isQ = (hIdx < p.nH);
    const uint h = isQ ? hIdx : (hIdx - p.nH);
    const uint totalH = isQ ? p.nH : p.nKV;
    device bfloat* row = (isQ ? qProp : kProp) + (size_t(t) * totalH + h) * p.headDim;
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

kernel void qwg_ink2_dspark_markov_ctx(device const bfloat* x [[buffer(0)]],
                                       device const bfloat* fRnn [[buffer(1)]],
                                       device const bfloat* rnnProjW [[buffer(2)]],
                                       device const bfloat* rnnProjB [[buffer(3)]],
                                       device float* ctxOut [[buffer(4)]],
                                       constant QWInk2DSparkMarkovCtxParams& p [[buffer(5)]],
                                       uint tg [[threadgroup_position_in_grid]],
                                       uint simd_id [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    const uint r = tg * 8u + simd_id;
    if (r >= p.projRows) return;
    const uint colBase = 2u * p.rank;
    const uint wStride = colBase + 2u * p.H;
    device const bfloat* wX = rnnProjW + size_t(r) * wStride + colBase;
    device const bfloat* wF = wX + p.H;
    const uint cnt = min(p.count, 8u);

    float accF = 0.0f;
    float accX[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if ((p.H & 7u) == 0u && (colBase & 3u) == 0u) {
        for (uint k = lane * 8u; k < p.H; k += 256u) {
            const ushort4 wfU0 = *(device const ushort4*)(wF + k);
            const ushort4 wfU1 = *(device const ushort4*)(wF + k + 4u);
            const ushort4 wxU0 = *(device const ushort4*)(wX + k);
            const ushort4 wxU1 = *(device const ushort4*)(wX + k + 4u);
            const ushort4 fU0 = *(device const ushort4*)(fRnn + k);
            const ushort4 fU1 = *(device const ushort4*)(fRnn + k + 4u);
            const float4 wf0 = float4(as_type<bfloat>(wfU0.x), as_type<bfloat>(wfU0.y), as_type<bfloat>(wfU0.z), as_type<bfloat>(wfU0.w));
            const float4 wf1 = float4(as_type<bfloat>(wfU1.x), as_type<bfloat>(wfU1.y), as_type<bfloat>(wfU1.z), as_type<bfloat>(wfU1.w));
            const float4 wx0 = float4(as_type<bfloat>(wxU0.x), as_type<bfloat>(wxU0.y), as_type<bfloat>(wxU0.z), as_type<bfloat>(wxU0.w));
            const float4 wx1 = float4(as_type<bfloat>(wxU1.x), as_type<bfloat>(wxU1.y), as_type<bfloat>(wxU1.z), as_type<bfloat>(wxU1.w));
            const float4 ff0 = float4(as_type<bfloat>(fU0.x), as_type<bfloat>(fU0.y), as_type<bfloat>(fU0.z), as_type<bfloat>(fU0.w));
            const float4 ff1 = float4(as_type<bfloat>(fU1.x), as_type<bfloat>(fU1.y), as_type<bfloat>(fU1.z), as_type<bfloat>(fU1.w));
            accF += dot(wf0, ff0) + dot(wf1, ff1);
            #pragma unroll
            for (uint i = 0u; i < 8u; ++i) {
                if (i >= cnt) break;
                device const bfloat* xi = x + size_t(i) * p.H + k;
                const ushort4 xU0 = *(device const ushort4*)xi;
                const ushort4 xU1 = *(device const ushort4*)(xi + 4u);
                const float4 xv0 = float4(as_type<bfloat>(xU0.x), as_type<bfloat>(xU0.y), as_type<bfloat>(xU0.z), as_type<bfloat>(xU0.w));
                const float4 xv1 = float4(as_type<bfloat>(xU1.x), as_type<bfloat>(xU1.y), as_type<bfloat>(xU1.z), as_type<bfloat>(xU1.w));
                accX[i] += dot(wx0, xv0) + dot(wx1, xv1);
            }
        }
    } else {
        for (uint k = lane; k < p.H; k += 32u) {
            float wf = float(wF[k]);
            float wx = float(wX[k]);
            accF += float(fRnn[k]) * wf;
            #pragma unroll
            for (uint i = 0u; i < 8u; ++i) {
                if (i >= cnt) break;
                accX[i] += float(x[size_t(i) * p.H + k]) * wx;
            }
        }
    }
    const float fTerm = simd_sum(accF) + float(rnnProjB[r]);
    for (uint i = 0u; i < cnt; ++i) {
        float xTerm = simd_sum(accX[i]);
        if (lane == 0u) {
            ctxOut[size_t(i) * p.projRows + r] = xTerm + fTerm;
        }
    }
}

kernel void qwg_ink2_dspark_markov_step_state(device const int32_t* draftToks [[buffer(0)]],
                                              device const bfloat* markovW1 [[buffer(1)]],
                                              device const bfloat* rnnProjW [[buffer(2)]],
                                              device const float* ctxF32 [[buffer(3)]],
                                              device bfloat* stateBF16 [[buffer(4)]],
                                              device bfloat* latentBF16 [[buffer(5)]],
                                              constant QWInk2DSparkMarkovStepParams& p [[buffer(6)]],
                                              uint tid [[thread_index_in_threadgroup]],
                                              uint simd_id [[simdgroup_index_in_threadgroup]],
                                              uint lane [[thread_index_in_simdgroup]]) {
    const uint rank = min(p.rank, 256u);
    const uint prevTok = uint(max(0, draftToks[p.stepIdx]));
    const size_t wOff = size_t(prevTok) * rank;
    threadgroup float sSh[256];
    threadgroup float wSh[256];
    threadgroup float rawSh[768];

    if (tid < rank) {
        sSh[tid] = (p.resetState != 0u) ? 0.0f : float(stateBF16[tid]);
        wSh[tid] = float(markovW1[wOff + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint wStride = 2u * rank + 2u * p.H;
    const size_t ctxOff = size_t(p.stepIdx) * p.projRows;
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
            if (lane == 0u && r < 768u) {
                rawSh[r] = float(qw_ink2_bf16(val));
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < rank) {
        float gSig = float(qw_ink2_bf16(1.0f / (1.0f + metal::exp(-rawSh[tid]))));
        float cTanh = float(qw_ink2_bf16(qw_ink2_tanh_safe(rawSh[tid + rank])));
        float oTanh = float(qw_ink2_bf16(qw_ink2_tanh_safe(rawSh[tid + 2u * rank])));
        stateBF16[tid] = qw_ink2_bf16(gSig * sSh[tid] + (1.0f - gSig) * cTanh);
        latentBF16[tid] = qw_ink2_bf16(oTanh);
    }
}

kernel void qwg_ink2_dspark_markov_logits_stage1(device const bfloat* baseLogits [[buffer(0)]],
                                                 device const bfloat* latentBF16 [[buffer(1)]],
                                                 device const bfloat* foldW [[buffer(2)]],
                                                 device const bfloat* foldB [[buffer(3)]],
                                                 device float* partVals [[buffer(4)]],
                                                 device int32_t* partIdxs [[buffer(5)]],
                                                 constant QWInk2DSparkLogitsParams& p [[buffer(6)]],
                                                 uint tg [[threadgroup_position_in_grid]],
                                                 uint tid [[thread_index_in_threadgroup]],
                                                 uint simd_id [[simdgroup_index_in_threadgroup]],
                                                 uint lane [[thread_index_in_simdgroup]]) {
    if (tg >= p.partCount) return;
    threadgroup float latSh[256];
    if (tid < p.rank && tid < 256u) {
        latSh[tid] = float(latentBF16[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float bestVal = -INFINITY;
    float rejectCeil = -INFINITY;
    int32_t bestIdx = 0x7fffffff;
    const size_t baseOff = size_t(p.stepIdx) * p.V;
    const uint rowsPerTG = ((p.V + p.partCount - 1u) / p.partCount + 7u) & ~7u;
    const uint vStart = tg * rowsPerTG;
    const uint vEnd = min(p.V, vStart + rowsPerTG);

    for (uint v = vStart + simd_id; v < vEnd; v += 8u) {
        device const bfloat* fw = foldW + size_t(v) * p.rank;
        float acc = 0.0f;
        if ((p.rank & 7u) == 0u) {
            for (uint k = lane * 8u; k < p.rank; k += 256u) {
                ushort4 w0 = *(device const ushort4*)(fw + k);
                ushort4 w1 = *(device const ushort4*)(fw + k + 4u);
                float4 wf0 = float4(as_type<bfloat>(w0.x), as_type<bfloat>(w0.y),
                                    as_type<bfloat>(w0.z), as_type<bfloat>(w0.w));
                float4 wf1 = float4(as_type<bfloat>(w1.x), as_type<bfloat>(w1.y),
                                    as_type<bfloat>(w1.z), as_type<bfloat>(w1.w));
                float4 lv0 = float4(latSh[k], latSh[k + 1u], latSh[k + 2u], latSh[k + 3u]);
                float4 lv1 = float4(latSh[k + 4u], latSh[k + 5u], latSh[k + 6u], latSh[k + 7u]);
                acc += dot(wf0, lv0) + dot(wf1, lv1);
            }
        } else {
            for (uint k = lane; k < p.rank; k += 32u) {
                acc += float(fw[k]) * latSh[k];
            }
        }
        float dotVal = simd_sum(acc);
        if (lane == 0u) {
            float bias = float(qw_ink2_bf16(dotVal + float(foldB[v])));
            float combined = float(qw_ink2_bf16(float(baseLogits[baseOff + v]) + bias));
            bool skip = false;
            if (p.softcap > 0.0f) {
                if (combined < rejectCeil) {
                    skip = true;
                } else {
                    float capped = float(qw_ink2_bf16(qw_ink2_tanh_safe(combined / p.softcap) * p.softcap));
                    if (capped < bestVal && combined > rejectCeil) rejectCeil = combined;
                    combined = capped;
                }
            }
            if (!skip) {
                int32_t iv = int32_t(v);
                if (combined > bestVal || (combined == bestVal && iv < bestIdx)) {
                    bestVal = combined;
                    bestIdx = iv;
                }
            }
        }
    }

    threadgroup float tgV[8];
    threadgroup int32_t tgI[8];
    if (lane == 0u) {
        tgV[simd_id] = bestVal;
        tgI[simd_id] = bestIdx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float winV = -INFINITY;
        int32_t winI = 0x7fffffff;
        for (uint s = 0u; s < 8u; ++s) {
            float cv = tgV[s];
            int32_t ci = tgI[s];
            if (cv > winV || (cv == winV && ci < winI)) {
                winV = cv;
                winI = ci;
            }
        }
        partVals[tg] = winV;
        partIdxs[tg] = winI;
    }
}

kernel void qwg_ink2_dspark_markov_argmax_final(device const float* partVals [[buffer(0)]],
                                                device const int32_t* partIdxs [[buffer(1)]],
                                                device int32_t* draftToks [[buffer(2)]],
                                                constant QWInk2DSparkLogitsParams& p [[buffer(3)]],
                                                uint tid [[thread_index_in_threadgroup]],
                                                uint simd_id [[simdgroup_index_in_threadgroup]],
                                                uint lane [[thread_index_in_simdgroup]]) {
    float bestV = -INFINITY;
    int32_t bestI = 0x7fffffff;
    for (uint i = tid; i < p.partCount; i += 256u) {
        float v = partVals[i];
        int32_t idx = partIdxs[i];
        if (v > bestV || (v == bestV && idx < bestI)) {
            bestV = v;
            bestI = idx;
        }
    }
    for (uint off = 16u; off > 0u; off >>= 1u) {
        float ov = simd_shuffle_down(bestV, off);
        int32_t oi = simd_shuffle_down(bestI, off);
        if (ov > bestV || (ov == bestV && oi < bestI)) {
            bestV = ov;
            bestI = oi;
        }
    }
    threadgroup float sgV[8];
    threadgroup int32_t sgI[8];
    if (lane == 0u) {
        sgV[simd_id] = bestV;
        sgI[simd_id] = bestI;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float winV = -INFINITY;
        int32_t winI = 0x7fffffff;
        for (uint s = 0u; s < 8u; ++s) {
            float cv = sgV[s];
            int32_t ci = sgI[s];
            if (cv > winV || (cv == winV && ci < winI)) {
                winV = cv;
                winI = ci;
            }
        }
        draftToks[p.stepIdx + 1u] = (winI == 0x7fffffff) ? 0 : winI;
    }
}

static inline void qw_ink2_dspark_attn_gqa_step(
        thread float qVec[4][4], thread float oVec[4][4],
        thread float* m, thread float* lSum,
        device const bfloat* kRow, device const bfloat* vRow,
        uint headDim, uint chunks, uint gCnt, float scale, uint lane) {
    float kVec[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float vVec[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint c = 0u; c < chunks; ++c) {
        const uint d = lane + c * 32u;
        if (d < headDim) {
            kVec[c] = float(kRow[d]);
            vVec[c] = float(vRow[d]);
        }
    }
    for (uint g = 0u; g < gCnt; ++g) {
        float dot = 0.0f;
        for (uint c = 0u; c < chunks; ++c) {
            dot += qVec[g][c] * kVec[c];
        }
        const float s = simd_sum(dot) * scale;
        const float mNew = max(m[g], s);
        const float expPrev = (lSum[g] > 0.0f) ? metal::exp(m[g] - mNew) : 0.0f;
        const float w = metal::exp(s - mNew);
        lSum[g] = lSum[g] * expPrev + w;
        for (uint c = 0u; c < chunks; ++c) {
            oVec[g][c] = oVec[g][c] * expPrev + w * vVec[c];
        }
        m[g] = mNew;
    }
}

kernel void qwg_ink2_dspark_attn_gqa(device const bfloat* qProp [[buffer(0)]],
                                     device const bfloat* kProp [[buffer(1)]],
                                     device const bfloat* vProp [[buffer(2)]],
                                     device const bfloat* kCache [[buffer(3)]],
                                     device const bfloat* vCache [[buffer(4)]],
                                     device bfloat* out [[buffer(5)]],
                                     constant QWInk2DSparkAttnParams& p [[buffer(6)]],
                                     uint3 tg [[threadgroup_position_in_grid]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    const uint kvh = tg.x;
    const uint t = tg.y;
    const uint gqa = (p.nKV > 0u) ? (p.nH / p.nKV) : 0u;
    const uint gStart = tg.z * 4u;
    const uint chunks = (p.headDim + 31u) / 32u;
    if (kvh >= p.nKV || t >= p.B || gStart >= gqa || chunks > 4u) return;

    const uint gCnt = min(4u, gqa - gStart);
    const uint hBase = kvh * gqa + gStart;
    const uint qDim = p.nH * p.headDim;
    const uint kvDim = p.nKV * p.headDim;

    float qVec[4][4];
    float oVec[4][4];
    float m[4];
    float lSum[4];
    for (uint g = 0u; g < gCnt; ++g) {
        device const bfloat* qRow = qProp + size_t(t) * qDim + size_t(hBase + g) * p.headDim;
        m[g] = -INFINITY;
        lSum[g] = 0.0f;
        for (uint c = 0u; c < chunks; ++c) {
            const uint d = lane + c * 32u;
            qVec[g][c] = (d < p.headDim) ? float(qRow[d]) : 0.0f;
            oVec[g][c] = 0.0f;
        }
    }

    for (uint j = p.ctxStart; j < p.ctxEnd; ++j) {
        const uint slot = (p.sliding != 0u) ? (j % p.ringCap) : j;
        const size_t kvOff = size_t(slot) * kvDim + size_t(kvh) * p.headDim;
        qw_ink2_dspark_attn_gqa_step(qVec, oVec, m, lSum,
                                     kCache + kvOff, vCache + kvOff,
                                     p.headDim, chunks, gCnt, p.scale, lane);
    }

    for (uint bp = 0u; bp < p.B; ++bp) {
        const size_t pOff = size_t(bp) * kvDim + size_t(kvh) * p.headDim;
        qw_ink2_dspark_attn_gqa_step(qVec, oVec, m, lSum,
                                     kProp + pOff, vProp + pOff,
                                     p.headDim, chunks, gCnt, p.scale, lane);
    }

    for (uint g = 0u; g < gCnt; ++g) {
        device bfloat* oRow = out + size_t(t) * qDim + size_t(hBase + g) * p.headDim;
        const float invL = (lSum[g] > 0.0f) ? (1.0f / lSum[g]) : 0.0f;
        for (uint c = 0u; c < chunks; ++c) {
            const uint d = lane + c * 32u;
            if (d < p.headDim) {
                oRow[d] = qw_ink2_bf16(oVec[g][c] * invL);
            }
        }
    }
}

kernel void qwg_ink2_dspark_markov_logits_aq8_stage1(device const bfloat* baseLogits [[buffer(0)]],
                                                     device const bfloat* latentBF16 [[buffer(1)]],
                                                     device const char* foldW [[buffer(2)]],
                                                     device const bfloat* foldB [[buffer(3)]],
                                                     device float* partVals [[buffer(4)]],
                                                     device int32_t* partIdxs [[buffer(5)]],
                                                     constant QWInk2DSparkLogitsParams& p [[buffer(6)]],
                                                     uint tg [[threadgroup_position_in_grid]],
                                                     uint tid [[thread_index_in_threadgroup]],
                                                     uint simd_id [[simdgroup_index_in_threadgroup]],
                                                     uint lane [[thread_index_in_simdgroup]]) {
    if (tg >= p.partCount) return;
    threadgroup float latSh[256];
    threadgroup float latSumSh[8];
    const uint nGroups = p.rank >> 5u;
    if (tid < 256u) {
        latSh[tid] = (tid < p.rank) ? float(latentBF16[tid]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id < 8u) {
        const float ls = simd_sum((simd_id * 32u + lane < p.rank && p.rank <= 256u)
                                  ? latSh[simd_id * 32u + lane] : 0.0f);
        if (lane == 0u) latSumSh[simd_id] = ls;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float bestVal = -INFINITY;
    float rejectCeil = -INFINITY;
    int32_t bestIdx = 0x7fffffff;
    const size_t baseOff = size_t(p.stepIdx) * p.V;
    const size_t rowBytes = size_t(nGroups) * 36u;
    const uint rowsPerTG = ((p.V + p.partCount - 1u) / p.partCount + 7u) & ~7u;
    const uint vStart = tg * rowsPerTG;
    const uint vEnd = min(p.V, vStart + rowsPerTG);
    const bool fastSmallRank = (p.rank <= 256u && (p.rank & 31u) == 0u);
    const uint g = lane >> 2u;
    const uint sub = lane & 3u;
    const bool laneActive = fastSmallRank && (g < nGroups);
    const uint b0 = laneActive ? (g * 32u + sub * 8u) : 0u;
    const uint b1 = laneActive ? (b0 + 4u) : 0u;
    const float4 lv0 = laneActive ? float4(latSh[b0], latSh[b0 + 1u], latSh[b0 + 2u], latSh[b0 + 3u]) : float4(0.0f);
    const float4 lv1 = laneActive ? float4(latSh[b1], latSh[b1 + 1u], latSh[b1 + 2u], latSh[b1 + 3u]) : float4(0.0f);
    const float latSum = (laneActive && sub == 0u) ? latSumSh[g] : 0.0f;

    for (uint v = vStart + simd_id; v < vEnd; v += 8u) {
        device const char* rowPtr = foldW + size_t(v) * rowBytes;
        float acc = 0.0f;
        if (fastSmallRank) {
            if (g < nGroups) {
                device const char* gp = rowPtr + size_t(g) * 36u;
                const half2 sb = *(device const half2*)gp;
                const packed_uint2 q2 = *(device const packed_uint2*)(gp + 4u + sub * 8u);
                const float4 f0 = float4(as_type<uchar4>(q2.x));
                const float4 f1 = float4(as_type<uchar4>(q2.y));
                acc = float(sb.x) * (dot(f0, lv0) + dot(f1, lv1));
                if (sub == 0u) acc += float(sb.y) * latSum;
            }
        } else {
            for (uint g = 0u; g < nGroups; ++g) {
                device const char* gp = rowPtr + size_t(g) * 36u;
                const half2 sb = *(device const half2*)gp;
                const uint code = ((device const uchar*)(gp + 4u))[lane];
                acc += (float(sb.x) * float(code) + float(sb.y)) * float(latentBF16[g * 32u + lane]);
            }
        }
        const float dotVal = simd_sum(acc);
        if (lane == 0u) {
            float bias = float(qw_ink2_bf16(dotVal + float(foldB[v])));
            float combined = float(qw_ink2_bf16(float(baseLogits[baseOff + v]) + bias));
            bool skip = false;
            if (p.softcap > 0.0f) {
                if (combined < rejectCeil) {
                    skip = true;
                } else {
                    float capped = float(qw_ink2_bf16(qw_ink2_tanh_safe(combined / p.softcap) * p.softcap));
                    if (capped < bestVal && combined > rejectCeil) rejectCeil = combined;
                    combined = capped;
                }
            }
            if (!skip) {
                const int32_t iv = int32_t(v);
                if (combined > bestVal || (combined == bestVal && iv < bestIdx)) {
                    bestVal = combined;
                    bestIdx = iv;
                }
            }
        }
    }

    threadgroup float tgV[8];
    threadgroup int32_t tgI[8];
    if (lane == 0u) {
        tgV[simd_id] = bestVal;
        tgI[simd_id] = bestIdx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float winV = -INFINITY;
        int32_t winI = 0x7fffffff;
        for (uint s = 0u; s < 8u; ++s) {
            const float cv = tgV[s];
            const int32_t ci = tgI[s];
            if (cv > winV || (cv == winV && ci < winI)) {
                winV = cv;
                winI = ci;
            }
        }
        partVals[tg] = winV;
        partIdxs[tg] = winI;
    }
}
