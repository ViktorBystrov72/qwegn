// qwattn_decode_general_shared.metal — Shared-prefix multi-row decode (T=2..8)
// for general GQA geometries (headDim 32..256, gqaRatio <= 16, optional SWA).
#include <metal_stdlib>
#include "include/qw_geometry_constants.metalh"
#include "include/qw_dequant.metalh"
#include "include/qw_kv_paged.metalh"
using namespace metal;

#define QW_MAXSPLITS 64u
#define QW_MAXKEYS   16u

static inline void qwg_shared_merge_row(
        const thread float (&q)[8],
        const thread half (&kk)[8],
        const thread half (&vv)[8],
        thread float& m,
        thread float& l,
        thread float (&acc)[8]) {
    float pd = 0.0f;
    for (uint i = 0; i < 8u; ++i) pd += q[i] * float(kk[i]);
    const float score = simd_sum(pd) * QW_ATTN_SCALE;
    const float mn = max(m, score);
    const float correction = score > m ? metal::exp(m - score) : 1.0f;
    const float p = metal::exp(score - mn);
    l = l * correction + p;
    for (uint i = 0; i < 8u; ++i) acc[i] = acc[i] * correction + p * float(vv[i]);
    m = mn;
}

template <uint KV>
static inline void qwg_shared_load_pool_row(
        const device half* kpool, const device half* vpool,
        const device int* ktab, const device int* vtab,
        uint row, uint kvh, uint d0, bool live,
        thread half (&kk)[8], thread half (&vv)[8]) {
    if (live) {
        if (KV == 0u) {
            const uint off = (row & 255u) * QW_ROW + kvh * QW_HEAD_DIM + d0;
            *(thread uint4*)kk = *(const device uint4*)(kpool + uint64_t(ktab[row >> 8]) * QW_PGEL + off);
            *(thread uint4*)vv = *(const device uint4*)(vpool + uint64_t(vtab[row >> 8]) * QW_PGEL + off);
        } else {
            QW_KV_LOAD8(kk, kpool, ktab[row >> 8], row & 255u, kvh, d0, KV);
            QW_KV_LOAD8(vv, vpool, vtab[row >> 8], row & 255u, kvh, d0, KV);
        }
    } else {
        for (uint i = 0; i < 8u; ++i) { kk[i] = half(0.0h); vv[i] = half(0.0h); }
    }
}

template <uint B, uint KV, uint STAGE_DIM>
static inline void qwg_decode_general_shared_body(
        const device half* qgate, const device half* kpool, const device half* vpool,
        const device int* ktab, const device int* vtab,
        const device half* knew, const device half* vnew, device float* partials,
        uint C, uint splits, uint slidingWindow, uint tid, uint3 tg,
        threadgroup half* stageK, threadgroup half* stageV) {
    const uint lane = tid & 31u, head = tg.x * QW_GQA_RATIO + (tid >> 5);
    const uint kvh = tg.x, d0 = lane * 8u;
    const bool live = d0 < QW_HEAD_DIM;
    const uint keysPerPhase = min(QW_MAXKEYS, 2u * QW_GQA_RATIO);
    const uint tptg = QW_GQA_RATIO * 32u;
    const uint vecsPerKey = QW_HEAD_DIM >> 3u;
    const uint totalVecs = keysPerPhase * vecsPerKey;
    const uint active0 = (slidingWindow > 0u) ? min(C + 1u, slidingWindow) : (C + 1u);
    const uint per = (active0 + splits - 1u) / splits;
    float q[B][8], m[B], l[B], acc[B][8];
    uint k0[B], cEnd[B], tiledEnd[B];
    #pragma unroll
    for (uint b = 0; b < B; ++b) {
        const uint kvEnd = C + b + 1u;
        const uint kvStart = (slidingWindow > 0u && kvEnd > slidingWindow) ? (kvEnd - slidingWindow) : 0u;
        k0[b] = min(kvStart + tg.y * per, kvEnd);
        const uint k1b = min(k0[b] + per, kvEnd);
        cEnd[b] = min(k1b, C + b);
        m[b] = -1e30f; l[b] = 0.0f;
        if (live) {
            const device half* qp = qgate + uint64_t(b) * QW_QPROJ_OUT + head * QW_Q_STRIDE * QW_HEAD_DIM + d0;
            for (uint i = 0; i < 8u; ++i) q[b][i] = float(qp[i]);
        } else {
            for (uint i = 0; i < 8u; ++i) q[b][i] = 0.0f;
        }
        #pragma unroll
        for (uint i = 0; i < 8u; ++i) acc[b][i] = 0.0f;
    }
    const uint tiledStart = k0[B - 1u];
    uint sharedEnd = tiledStart;
    #pragma unroll
    for (uint b = 0; b < B; ++b) {
        for (uint r = k0[b]; r < min(tiledStart, cEnd[b]); ++r) {
            half kk[8], vv[8];
            qwg_shared_load_pool_row<KV>(kpool, vpool, ktab, vtab, r, kvh, d0, live, kk, vv);
            qwg_shared_merge_row(q[b], kk, vv, m[b], l[b], acc[b]);
        }
        tiledEnd[b] = tiledStart + ((cEnd[b] > tiledStart) ? ((cEnd[b] - tiledStart) / keysPerPhase) * keysPerPhase : 0u);
        sharedEnd = max(sharedEnd, tiledEnd[b]);
    }
    for (uint r = tiledStart; r < sharedEnd; r += keysPerPhase) {
        for (uint v = tid; v < totalVecs; v += tptg) {
            const uint keyIdx = v / vecsPerKey;
            const uint col = (v % vecsPerKey) << 3u;
            const uint row = r + keyIdx;
            const uint pk = (uint)ktab[row >> 8], pv = (uint)vtab[row >> 8];
            const uint e = keyIdx * STAGE_DIM + col;
            if (KV == 0u) {
                const uint off = (row & 255u) * QW_ROW + kvh * QW_HEAD_DIM + col;
                *(threadgroup uint4*)&stageK[e] = *(const device uint4*)(kpool + uint64_t(pk) * QW_PGEL + off);
                *(threadgroup uint4*)&stageV[e] = *(const device uint4*)(vpool + uint64_t(pv) * QW_PGEL + off);
            } else {
                QW_KV_LOAD8S(stageK, kpool, pk, row & 255u, kvh, col, e, KV);
                QW_KV_LOAD8S(stageV, vpool, pv, row & 255u, kvh, col, e, KV);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint b = 0; b < B; ++b) {
            if (r < tiledEnd[b]) {
                float scores[QW_MAXKEYS];
                for (uint j = 0; j < keysPerPhase; ++j) {
                    half kk[8];
                    if (live) {
                        *(thread uint4*)kk = *(threadgroup const uint4*)&stageK[j * STAGE_DIM + d0];
                    } else {
                        #pragma unroll
                        for (uint i = 0; i < 8u; ++i) kk[i] = half(0.0h);
                    }
                    float pd = 0.0f;
                    for (uint i = 0; i < 8u; ++i) pd += q[b][i] * float(kk[i]);
                    scores[j] = simd_sum(pd) * QW_ATTN_SCALE;
                }
                float bm = scores[0];
                for (uint j = 1; j < keysPerPhase; ++j) bm = max(bm, scores[j]);
                const float mb = max(m[b], bm);
                float lp = 0.0f, op[8];
                #pragma unroll
                for (uint i = 0; i < 8u; ++i) op[i] = 0.0f;
                for (uint j = 0; j < keysPerPhase; ++j) {
                    const float pj = metal::exp(scores[j] - mb);
                    lp += pj;
                    half vv[8];
                    if (live) {
                        *(thread uint4*)vv = *(threadgroup const uint4*)&stageV[j * STAGE_DIM + d0];
                    } else {
                        #pragma unroll
                        for (uint i = 0; i < 8u; ++i) vv[i] = half(0.0h);
                    }
                    #pragma unroll
                    for (uint i = 0; i < 8u; ++i) op[i] += pj * float(vv[i]);
                }
                const float correction = metal::exp(m[b] - mb);
                l[b] = l[b] * correction + lp;
                #pragma unroll
                for (uint i = 0; i < 8u; ++i) acc[b][i] = acc[b][i] * correction + op[i];
                m[b] = mb;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for (uint b = 0; b < B; ++b) {
        for (uint r = max(k0[b], tiledEnd[b]); r < cEnd[b]; ++r) {
            half kk[8], vv[8];
            qwg_shared_load_pool_row<KV>(kpool, vpool, ktab, vtab, r, kvh, d0, live, kk, vv);
            qwg_shared_merge_row(q[b], kk, vv, m[b], l[b], acc[b]);
        }
        if (C + b >= k0[b] && (C + b - k0[b]) < per) {
            half kk[8], vv[8];
            if (live) {
                *(thread uint4*)kk = *(const device uint4*)(knew + uint64_t(b) * QW_ROW + kvh * QW_HEAD_DIM + d0);
                *(thread uint4*)vv = *(const device uint4*)(vnew + uint64_t(b) * QW_ROW + kvh * QW_HEAD_DIM + d0);
            } else {
                for (uint i = 0; i < 8u; ++i) { kk[i] = half(0.0h); vv[i] = half(0.0h); }
            }
            qwg_shared_merge_row(q[b], kk, vv, m[b], l[b], acc[b]);
        }
        const uint splitStride = max(QW_MAXSPLITS, splits);
        device float* dst = partials + (uint64_t(b) * QW_Q_HEADS * splitStride + head * splitStride + tg.y) * (QW_HEAD_DIM + 2u);
        if (live) {
            for (uint i = 0; i < 8u; ++i) dst[d0 + i] = acc[b][i];
        }
        if (lane == 0u) { dst[QW_HEAD_DIM] = m[b]; dst[QW_HEAD_DIM + 1u] = l[b]; }
    }
}

#define QW_GEN_SHARED_ENTRY(SFX_DIM, STAGE_DIM, B, KV, SFX_MODE) \
kernel void qwg_attn_decode_general_shared##SFX_DIM##_m##B##SFX_MODE( \
        const device half* qgate [[buffer(0)]], const device half* kpool [[buffer(1)]], \
        const device half* vpool [[buffer(2)]], const device int* ktab [[buffer(3)]], \
        const device int* vtab [[buffer(4)]], const device half* knew [[buffer(5)]], \
        const device half* vnew [[buffer(6)]], device float* partials [[buffer(7)]], \
        constant uint& C [[buffer(8)]], constant uint& splits [[buffer(9)]], \
        constant uint& slidingWindow [[buffer(10)]], \
        uint tid [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) { \
    threadgroup half stageK[QW_MAXKEYS * STAGE_DIM], stageV[QW_MAXKEYS * STAGE_DIM]; \
    qwg_decode_general_shared_body<B, KV, STAGE_DIM>(qgate, kpool, vpool, ktab, vtab, knew, vnew, \
                                                     partials, C, splits, slidingWindow, tid, tg, \
                                                     &stageK[0], &stageV[0]); \
}

#define QW_GEN_SHARED_FAMILY(SFX_DIM, STAGE_DIM, KV, SFX_MODE) \
QW_GEN_SHARED_ENTRY(SFX_DIM, STAGE_DIM, 2, KV, SFX_MODE) \
QW_GEN_SHARED_ENTRY(SFX_DIM, STAGE_DIM, 3, KV, SFX_MODE) \
QW_GEN_SHARED_ENTRY(SFX_DIM, STAGE_DIM, 4, KV, SFX_MODE) \
QW_GEN_SHARED_ENTRY(SFX_DIM, STAGE_DIM, 5, KV, SFX_MODE) \
QW_GEN_SHARED_ENTRY(SFX_DIM, STAGE_DIM, 6, KV, SFX_MODE) \
QW_GEN_SHARED_ENTRY(SFX_DIM, STAGE_DIM, 7, KV, SFX_MODE) \
QW_GEN_SHARED_ENTRY(SFX_DIM, STAGE_DIM, 8, KV, SFX_MODE)

QW_GEN_SHARED_FAMILY(, 256u, 0u,)
QW_GEN_SHARED_FAMILY(, 256u, 1u, _q8)
QW_GEN_SHARED_FAMILY(, 256u, 2u, _q4)
QW_GEN_SHARED_FAMILY(_hd128, 128u, 0u,)
QW_GEN_SHARED_FAMILY(_hd128, 128u, 1u, _q8)
QW_GEN_SHARED_FAMILY(_hd128, 128u, 2u, _q4)
#undef QW_GEN_SHARED_FAMILY
#undef QW_GEN_SHARED_ENTRY
