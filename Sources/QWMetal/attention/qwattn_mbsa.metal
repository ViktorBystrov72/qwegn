// qwattn_mbsa.metal — Maxwell Block-Sparse GQA indexer prep, top-k selector & scalar fallback attention.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "include/qw_geometry_constants.metalh"
#include "include/qw_dequant.metalh"
#include "include/qw_kv_paged.metalh"
using namespace metal;

struct MBSAH8 { half v[8]; } __attribute__((__packed__, __aligned__(16)));
static inline MBSAH8 mbsa_asH8(uint4 w) { return *(thread MBSAH8*)&w; }

// Fused per-head RMSNorm (1+w law) + NeoX RoPE on indexer QKV [rows, (kvHeads+1)*headDim].
// Writes rotated idxQOut [rows, kvHeads*headDim] and appends idxKCache[pos0.., headDim].
kernel void qwg_mbsa_index_prep_f16(const device uint4* qkvRaw [[buffer(0)]],
                                    device uint4*       qOut   [[buffer(1)]],
                                    device uint4*       kCache [[buffer(2)]],
                                    const device uint4* wq     [[buffer(3)]],
                                    const device uint4* wk     [[buffer(4)]],
                                    const device float* table  [[buffer(5)]],
                                    constant uint4&     dims   [[buffer(6)]], // (pos0, kvHeads, headU4, _)
                                    constant float&     eps    [[buffer(7)]],
                                    uint2 tg [[threadgroup_position_in_grid]],
                                    uint  ti [[thread_index_in_threadgroup]]) {
    const uint pos0 = dims.x;
    const uint kvHeads = dims.y;
    const uint headU4 = dims.z;
    const uint r = tg.y;
    const bool isK = (tg.x == kvHeads);
    const uint head = isK ? kvHeads : tg.x;
    const uint totalHeads = kvHeads + 1u;
    const uint srcStrideU4 = (dims.w != 0u) ? dims.w : (totalHeads * headU4);

    const device uint4* src = qkvRaw + (size_t)r * srcStrideU4 + (size_t)head * headU4;
    device uint4* dst = isK
        ? (kCache + (size_t)(pos0 + r) * headU4)
        : (qOut + ((size_t)r * kvHeads + head) * headU4);
    const device uint4* w = isK ? wk : wq;
    const bool active = ti < headU4;

    uint4 xw = active ? src[ti] : uint4(0u);
    MBSAH8 xv = mbsa_asH8(xw);
    MBSAH8 wv = active ? mbsa_asH8(w[ti]) : mbsa_asH8(uint4(0u));
    float acc = 0.0f;
    for (int i = 0; i < 8; i++) { float v = float(xv.v[i]); acc += v * v; }
    float rstd = rsqrt(simd_sum(acc) / float(headU4 * 8u) + eps);
    float xn[8];
    for (int i = 0; i < 8; i++) xn[i] = float(xv.v[i]) * rstd * (float(wv.v[i]) + 1.0f);

    const uint ropeSlots = QW_ROPE_DIMS / 8u;
    const uint partnerHalf = QW_ROPE_DIMS / 16u;
    const uint partner = (ti < partnerHalf) ? (ti + partnerHalf) : (ti - partnerHalf);

    float pf[8];
    if (partnerHalf > 0u && ropeSlots <= 32u) {
        const ushort srcLane = ushort(partner & 31u);
        #pragma unroll
        for (int i = 0; i < 8; i++) pf[i] = simd_shuffle(xn[i], srcLane);
    } else if (ti < ropeSlots) {
        MBSAH8 xvP = mbsa_asH8((active && partner < headU4) ? src[partner] : uint4(0u));
        MBSAH8 wvP = mbsa_asH8((active && partner < headU4) ? w[partner] : uint4(0u));
        for (int i = 0; i < 8; i++) pf[i] = float(xvP.v[i]) * rstd * (float(wvP.v[i]) + 1.0f);
    }

    if (active) {
        MBSAH8 out8;
        if (ti < ropeSlots) {
            const device float* cs = table + (size_t)(pos0 + r) * QW_ROPE_DIMS;
            const bool firstHalf = ti < partnerHalf;
            const uint pBase = (firstHalf ? ti : (ti - partnerHalf)) * 8u;
            for (int i = 0; i < 8; i++) {
                uint p = pBase + uint(i);
                float c = cs[p], sn = cs[QW_ROPE_DIMS / 2u + p];
                float rotv = firstHalf ? -pf[i] : pf[i];
                out8.v[i] = half(xn[i] * c + rotv * sn);
            }
        } else {
            for (int i = 0; i < 8; i++) out8.v[i] = half(xn[i]);
        }
        dst[ti] = *(thread uint4*)&out8;
    }
}

// Per-(group, query) MBSA block-max scorer & selector. Grid = (kvHeads, rows, 1), TG = 256.
kernel void qwg_mbsa_select_blocks(const device half* idxQ       [[buffer(0)]],
                                   const device half* idxKCache  [[buffer(1)]],
                                   device int*        blockList  [[buffer(2)]],
                                   device int*        blockCount [[buffer(3)]],
                                   constant uint4&    p0         [[buffer(4)]], // (pos0, kvHeads, headDim, blockSize)
                                   constant uint4&    p1         [[buffer(5)]], // (prefixBlks, localBlks, remoteBlks, listStride)
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
    const uint kvh = tg.x, qi = tg.y;
    const uint pos = p0.x + qi, kvHeads = p0.y, headDim = p0.z, blockSize = p0.w;
    const uint prefixBlks = p1.x, localBlks = p1.y, remoteBlks = p1.z, listStride = p1.w;
    const uint curBlk = pos / blockSize;
    const uint localStartBlk = (curBlk + 1u > localBlks) ? (curBlk + 1u - localBlks) : 0u;
    const size_t slotBase = ((size_t)qi * kvHeads + kvh) * listStride;

    if (curBlk < prefixBlks + localBlks + remoteBlks || localStartBlk <= prefixBlks + remoteBlks) {
        const uint totalBlks = min(curBlk + 1u, listStride);
        for (uint b = tid; b < totalBlks; b += 256u) blockList[slotBase + b] = int(b);
        if (tid == 0u) blockCount[(size_t)qi * kvHeads + kvh] = int(totalBlks);
        return;
    }

    threadgroup float sharedQ[256];
    threadgroup float sgMax[8];
    const float idxScale = rsqrt(float(headDim));
    const device half* qRow = idxQ + ((size_t)qi * kvHeads + kvh) * headDim;
    for (uint d = tid; d < headDim; d += 256u) sharedQ[d] = float(qRow[d]) * idxScale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint sg = tid >> 5u, lane = tid & 31u;
    const uint rKeep = min(remoteBlks, 32u);
    float topScores[32];
    int   topIds[32];
    for (uint r = 0u; r < rKeep; r++) { topScores[r] = -1e30f; topIds[r] = -1; }
    uint worstSlot = 0u;
    float worstScore = -1e30f;

    if (headDim == 128u) {
        const uint hp = lane >> 4u, d8 = (lane & 15u) << 3u;
        const float4 qLo = *(threadgroup const float4*)(sharedQ + d8);
        const float4 qHi = *(threadgroup const float4*)(sharedQ + d8 + 4u);
        for (uint bBase = prefixBlks; bBase < localStartBlk; bBase += 8u) {
            const uint b = bBase + sg;
            float tokMax = -1e30f;
            if (b < localStartBlk) {
                const uint tStart = b * blockSize;
                for (uint off = 0u; off < blockSize; off += 2u) {
                    const uint kOff = off + hp;
                    float dotv = -1e30f;
                    if (kOff < blockSize) {
                        const uint4 raw = *(const device uint4*)(idxKCache + (size_t)(tStart + kOff) * 128u + d8);
                        const half4 kh0 = as_type<half4>(raw.xy), kh1 = as_type<half4>(raw.zw);
                        float d = dot(qLo, float4(kh0)) + dot(qHi, float4(kh1));
                        d += simd_shuffle_xor(d, ushort(1));
                        d += simd_shuffle_xor(d, ushort(2));
                        d += simd_shuffle_xor(d, ushort(4));
                        d += simd_shuffle_xor(d, ushort(8));
                        dotv = d;
                    }
                    tokMax = max(tokMax, dotv);
                }
                const float peer = simd_shuffle_xor(tokMax, ushort(16));
                tokMax = max(tokMax, peer);
            }
            if (lane == 0u) sgMax[sg] = tokMax;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u && rKeep > 0u) {
                for (uint s = 0u; s < 8u && bBase + s < localStartBlk; s++) {
                    const float bScore = sgMax[s];
                    if (topIds[worstSlot] < 0 || bScore > worstScore) {
                        topScores[worstSlot] = bScore;
                        topIds[worstSlot] = int(bBase + s);
                        uint ws = 0u;
                        for (uint i = 1u; i < rKeep; i++) {
                            if (topIds[ws] >= 0 && (topIds[i] < 0 || topScores[i] < topScores[ws] || (topScores[i] == topScores[ws] && topIds[i] > topIds[ws]))) {
                                ws = i;
                            }
                        }
                        worstSlot = ws;
                        worstScore = topScores[ws];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (uint b = prefixBlks; b < localStartBlk; b++) {
            const uint tStart = b * blockSize;
            float tokMax = -1e38f;
            for (uint off = sg; off < blockSize; off += 8u) {
                const device half* kRow = idxKCache + (size_t)(tStart + off) * headDim;
                float dotv = 0.0f;
                for (uint d = lane; d < headDim; d += 32u) dotv += sharedQ[d] * float(kRow[d]);
                tokMax = max(tokMax, simd_sum(dotv));
            }
            if (lane == 0u) sgMax[sg] = tokMax;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u && rKeep > 0u) {
                float bScore = sgMax[0];
                for (uint i = 1u; i < 8u; i++) bScore = max(bScore, sgMax[i]);
                if (topIds[worstSlot] < 0 || bScore > worstScore) {
                    topScores[worstSlot] = bScore;
                    topIds[worstSlot] = int(b);
                    uint ws = 0u;
                    for (uint i = 1u; i < rKeep; i++) {
                        if (topIds[ws] >= 0 && (topIds[i] < 0 || topScores[i] < topScores[ws] || (topScores[i] == topScores[ws] && topIds[i] > topIds[ws]))) {
                            ws = i;
                        }
                    }
                    worstSlot = ws;
                    worstScore = topScores[ws];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid == 0u) {
        for (uint i = 0u; i < rKeep; i++) {
            for (uint j = i + 1u; j < rKeep; j++) {
                if (topIds[j] >= 0 && (topIds[i] < 0 || topIds[j] < topIds[i])) {
                    int tmp = topIds[i]; topIds[i] = topIds[j]; topIds[j] = tmp;
                }
            }
        }
        uint outIdx = 0u;
        const uint prefixEnd = min(prefixBlks, localStartBlk);
        for (uint p = 0u; p < prefixEnd && outIdx < listStride; p++) blockList[slotBase + outIdx++] = int(p);
        for (uint r = 0u; r < rKeep && outIdx < listStride; r++) {
            if (topIds[r] >= 0) blockList[slotBase + outIdx++] = topIds[r];
        }
        for (uint l = localStartBlk; l <= curBlk && outIdx < listStride; l++) blockList[slotBase + outIdx++] = int(l);
        blockCount[(size_t)qi * kvHeads + kvh] = int(outIdx);
    }
}

template <uint MODE>
static inline void qwg_mbsa_attn_body(const device half* qgate, device half* kpool, device half* vpool,
                                      const device int* ktab, const device int* vtab,
                                      const device int* blockList, const device int* blockCount,
                                      device half* outp, uint4 params, uint tid, uint3 tg) {
    const uint head = tg.x, qi = tg.y, lane = tid & 31u, d0 = lane * 8u;
    const bool live = d0 < QW_HEAD_DIM;
    const uint kvh = head / QW_GQA_RATIO;
    const uint pos = params.x + qi, blockSize = params.y, listStride = params.z;

    const device half* qrow = qgate + (size_t)qi * QW_QPROJ_OUT + (size_t)head * QW_Q_STRIDE * QW_HEAD_DIM;
    float q8[8];
    for (uint i = 0; i < 8u; i++) q8[i] = (live ? float(qrow[d0 + i]) : 0.0f);
    float gf[8];
    for (uint i = 0; i < 8u; i++) gf[i] = 1.0f;
    if (QW_HAS_Q_GATE && live) {
        const device half* grow = qrow + QW_HEAD_DIM;
        for (uint i = 0; i < 8u; i++) gf[i] = 1.0f / (1.0f + metal::exp(-float(grow[d0 + i])));
    }

    const size_t slotBase = ((size_t)qi * QW_KV_HEADS + kvh) * listStride;
    const uint nBlks = (uint)blockCount[(size_t)qi * QW_KV_HEADS + kvh];
    const uint kvRow = QW_FC_KV_ROW, kvPgel = QW_FC_KV_PGEL;
    float m = -1e30f, lsum = 0.0f;
    float acc[8];
    for (uint i = 0; i < 8u; i++) acc[i] = 0.0f;

    for (uint bi = 0u; bi < nBlks; bi++) {
        const int rawBlk = blockList[slotBase + bi];
        if (rawBlk < 0) continue;
        const uint t0 = uint(rawBlk) * blockSize;
        if (t0 > pos) continue;
        const uint tEnd = min(pos + 1u, t0 + blockSize);
        for (uint j = t0; j < tEnd; j++) {
            const uint page = j >> 8, slot = j & 255u;
            const uint pk = (uint)ktab[page], pv = (uint)vtab[page];
            float k8[8], v8[8];
            for (uint i = 0; i < 8u; i++) { k8[i] = 0.0f; v8[i] = 0.0f; }
            if (live) {
                if (MODE == 0u) {
                    const uint64_t rowOff = (uint64_t)slot * kvRow + (uint64_t)kvh * QW_HEAD_DIM;
                    const device half* sk = kpool + (uint64_t)pk * kvPgel + rowOff;
                    const device half* sv = vpool + (uint64_t)pv * kvPgel + rowOff;
                    for (uint i = 0; i < 8u; i++) { k8[i] = float(sk[d0 + i]); v8[i] = float(sv[d0 + i]); }
                } else {
                    half kh[8], vh[8];
                    QW_KV_LOAD8(kh, kpool, pk, slot, kvh, d0, MODE)
                    QW_KV_LOAD8(vh, vpool, pv, slot, kvh, d0, MODE)
                    for (uint i = 0; i < 8u; i++) { k8[i] = float(kh[i]); v8[i] = float(vh[i]); }
                }
            }
            float dotv = 0.0f;
            for (uint i = 0; i < 8u; i++) dotv += q8[i] * k8[i];
            const float s = simd_sum(dotv) * QW_ATTN_SCALE;
            const float mnew = max(m, s);
            const float corr = (m > -1e29f) ? metal::exp(m - mnew) : 1.0f;
            const float p = metal::exp(s - mnew);
            for (uint i = 0; i < 8u; i++) acc[i] = acc[i] * corr + p * v8[i];
            lsum = lsum * corr + p;
            m = mnew;
        }
    }

    half hv[8];
    const float invL = (lsum > 0.0f) ? (1.0f / lsum) : 0.0f;
    for (uint i = 0; i < 8u; i++) hv[i] = live ? half(acc[i] * invL * gf[i]) : half(0.0f);
    fwht32_inplace(hv, lane);
    if (live) {
        device half* o = outp + (size_t)qi * (QW_Q_HEADS * QW_HEAD_DIM) + (size_t)head * QW_HEAD_DIM + d0;
        for (uint i = 0; i < 8u; i++) o[i] = hv[i];
    }
}

#define QW_MBSA_ATTN_ENTRY(SFX, MODE) \
kernel void qwg_mbsa_attn##SFX(const device half* qgate [[buffer(0)]], \
                               device half* kpool [[buffer(1)]], \
                               device half* vpool [[buffer(2)]], \
                               const device int* ktab [[buffer(3)]], \
                               const device int* vtab [[buffer(4)]], \
                               const device int* blockList [[buffer(5)]], \
                               const device int* blockCount [[buffer(6)]], \
                               device half* outp [[buffer(7)]], \
                               constant uint4& params [[buffer(8)]], \
                               uint tid [[thread_index_in_threadgroup]], \
                               uint3 tg [[threadgroup_position_in_grid]]) { \
    qwg_mbsa_attn_body<MODE>(qgate, kpool, vpool, ktab, vtab, blockList, blockCount, outp, params, tid, tg); \
}
QW_MBSA_ATTN_ENTRY(_f16, 0u)
QW_MBSA_ATTN_ENTRY(_q8,  1u)
QW_MBSA_ATTN_ENTRY(_q4,  2u)
#undef QW_MBSA_ATTN_ENTRY

// Prefill 32-query-tile MBSA block-max selector using simdgroup MMA + union bitmask compaction.
kernel void qwg_mbsa_select_blocks_tiled32(const device half* idxQ           [[buffer(0)]],
                                           const device half* idxKCache      [[buffer(1)]],
                                           device int*        tileBlockList  [[buffer(2)]],
                                           device uint*       tileBlockMask  [[buffer(3)]],
                                           device uint*       tileBlockCount [[buffer(4)]],
                                           constant uint4&    p0             [[buffer(5)]],
                                           constant uint4&    p1             [[buffer(6)]],
                                           constant uint4&    p2             [[buffer(7)]],
                                           uint2 tg  [[threadgroup_position_in_grid]],
                                           uint  tid [[thread_index_in_threadgroup]]) {
    const uint kvh = tg.x, tileIdx = tg.y;
    const uint rows = p2.x, listStride = p2.y;
    const uint tile0 = tileIdx * 32u;
    if (tile0 >= rows) return;
    const uint R = min(32u, rows - tile0);
    const uint pos0 = p0.x, kvHeads = p0.y, headDim = p0.z, blockSize = p0.w;
    const uint prefixBlks = p1.x, localBlks = p1.y, remoteBlks = p1.z, tileListStride = p1.w;
    const uint rKeep = min(remoteBlks, 32u);
    const uint kDim = min(headDim, 128u);
    const uint kPadded = min(128u, (kDim + 7u) & ~7u);
    const float idxScale = rsqrt(float(headDim));

    threadgroup half sQ[32][136];
    threadgroup half sK[32][136];
    threadgroup float sScores[32][34];
    threadgroup int sRowBlks[32 * 32];
    threadgroup atomic_uint maskWin[512];
    threadgroup float sBlkMax[32];
    threadgroup uchar sIsDense[32];
    threadgroup uint sCurBlk[32];
    threadgroup uint sLocalStart[32];
    threadgroup uint sSgCounts[8];
    threadgroup uint sSgBases[8];
    threadgroup uint sNextOutCount;

    const uint sg = tid >> 5u, lane = tid & 31u;
    bool qDenseSelf = false;
    uint qCurBlkSelf = 0u, qLocalStartSelf = 0u;
    if (tid < R) {
        const uint qPos = pos0 + tile0 + tid;
        qCurBlkSelf = qPos / blockSize;
        qLocalStartSelf = (qCurBlkSelf + 1u > localBlks) ? (qCurBlkSelf + 1u - localBlks) : 0u;
        qDenseSelf = (qCurBlkSelf < prefixBlks + localBlks + remoteBlks) || (qLocalStartSelf <= prefixBlks + remoteBlks);
        sIsDense[tid] = qDenseSelf ? 1 : 0;
        sCurBlk[tid] = qCurBlkSelf;
        sLocalStart[tid] = qLocalStartSelf;
        sBlkMax[tid] = -1e30f;
    }
    for (uint e = tid; e < 32u * kPadded; e += 256u) {
        const uint q = e / kPadded;
        const uint k = e % kPadded;
        sQ[q][k] = (q < R && k < kDim) ? idxQ[((size_t)(tile0 + q) * kvHeads + kvh) * headDim + k] : half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint maxCurBlk = sCurBlk[R - 1u];
    const uint maxLocalStart = sLocalStart[R - 1u];
    const bool anySparse = (sIsDense[R - 1u] == 0);

    float topScores[32];
    int   topIds[32];
    for (uint r = 0u; r < rKeep; ++r) { topScores[r] = -1e30f; topIds[r] = -1; }
    uint worstSlot = 0u;
    float worstScore = -1e30f;

    if (anySparse && maxLocalStart > prefixBlks && rKeep > 0u) {
        const uint sgRow = (sg >> 1u) * 8u;
        const uint sgCol = (sg & 1u) * 16u;
        const uint redRow = tid >> 3u;
        const uint lane8 = tid & 7u;
        const bool fast128 = (headDim == 128u);

        for (uint b = prefixBlks; b < maxLocalStart; ++b) {
            const uint tStart = b * blockSize;
            for (uint off = 0u; off < blockSize; off += 32u) {
                const uint subLen = min(32u, blockSize - off);
                if (fast128 && subLen == 32u) {
                    const uint kr = tid >> 3u;
                    const uint d16 = (tid & 7u) << 4u;
                    const device uint4* src = (const device uint4*)(idxKCache + (size_t)(tStart + off + kr) * 128u + d16);
                    *(threadgroup uint4*)(&sK[kr][d16])      = src[0];
                    *(threadgroup uint4*)(&sK[kr][d16 + 8u]) = src[1];
                } else {
                    for (uint e = tid; e < 32u * kPadded; e += 256u) {
                        const uint kr = e / kPadded;
                        const uint k = e % kPadded;
                        sK[kr][k] = (kr < subLen && k < kDim)
                            ? idxKCache[(size_t)(tStart + off + kr) * headDim + k]
                            : half(0.0h);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_float8x8 acc0 = simdgroup_float8x8(0.0f);
                simdgroup_float8x8 acc1 = simdgroup_float8x8(0.0f);
                for (uint k = 0u; k < kPadded; k += 8u) {
                    simdgroup_half8x8 a, b0, b1;
                    simdgroup_load(a,  &sQ[sgRow][k],        136, ulong2(0, 0), false);
                    simdgroup_load(b0, &sK[sgCol + 0u][k],   136, ulong2(0, 0), true);
                    simdgroup_load(b1, &sK[sgCol + 8u][k],   136, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(acc0, a, b0, acc0);
                    simdgroup_multiply_accumulate(acc1, a, b1, acc1);
                }
                simdgroup_store(acc0, &sScores[sgRow][sgCol + 0u], 34, ulong2(0, 0), false);
                simdgroup_store(acc1, &sScores[sgRow][sgCol + 8u], 34, ulong2(0, 0), false);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float subMax = -1e30f;
                if (redRow < R) {
                    const uint c0 = lane8 * 4u;
                    for (uint c = 0u; c < 4u; ++c) {
                        if (c0 + c < subLen) subMax = max(subMax, sScores[redRow][c0 + c] * idxScale);
                    }
                    subMax = max(subMax, simd_shuffle_xor(subMax, ushort(1)));
                    subMax = max(subMax, simd_shuffle_xor(subMax, ushort(2)));
                    subMax = max(subMax, simd_shuffle_xor(subMax, ushort(4)));
                    if (lane8 == 0u) {
                        sBlkMax[redRow] = (off == 0u) ? subMax : max(sBlkMax[redRow], subMax);
                    }
                }
                if (off + 32u >= blockSize) {
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (tid < R && !qDenseSelf && b >= prefixBlks && b < qLocalStartSelf) {
                        const float bScore = sBlkMax[tid];
                        if (topIds[worstSlot] < 0 || bScore > worstScore) {
                            topScores[worstSlot] = bScore;
                            topIds[worstSlot] = int(b);
                            uint ws = 0u;
                            for (uint i = 1u; i < rKeep; ++i) {
                                if (topIds[ws] >= 0 && (topIds[i] < 0 || topScores[i] < topScores[ws]
                                        || (topScores[i] == topScores[ws] && topIds[i] > topIds[ws]))) {
                                    ws = i;
                                }
                            }
                            worstSlot = ws;
                            worstScore = topScores[ws];
                        }
                    }
                }
            }
        }
    }
    if (tid < R) {
        for (uint r = 0u; r < rKeep; ++r) {
            sRowBlks[tid * 32u + r] = qDenseSelf ? -1 : topIds[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const size_t slotBase = ((size_t)tileIdx * kvHeads + kvh) * tileListStride;
    uint outCount = 0u;
    for (uint winBase = 0u; winBase <= maxCurBlk; winBase += 512u) {
        const uint winLen = min(512u, maxCurBlk + 1u - winBase);
        for (uint k = 0u; k < 2u; ++k) {
            const uint idx = tid + k * 256u;
            uint detMask = 0u;
            if (idx < winLen) {
                const uint b = winBase + idx;
                for (uint q = 0u; q < R; ++q) {
                    const bool hit = (sIsDense[q] != 0)
                        ? (b <= sCurBlk[q] && b < listStride)
                        : (b < min(prefixBlks, sLocalStart[q]) || (b >= sLocalStart[q] && b <= sCurBlk[q]));
                    if (hit) detMask |= (1u << q);
                }
            }
            atomic_store_explicit(&maskWin[idx], detMask, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (anySparse && rKeep > 0u) {
            const uint totalRemoteSlots = R * rKeep;
            for (uint e = tid; e < totalRemoteSlots; e += 256u) {
                const uint q = e / rKeep;
                const uint j = e % rKeep;
                if (sIsDense[q] == 0) {
                    const int rb = sRowBlks[q * 32u + j];
                    if (rb >= int(winBase) && rb < int(winBase + winLen)) {
                        atomic_fetch_or_explicit(&maskWin[uint(rb) - winBase], 1u << q, memory_order_relaxed);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint idx0 = 2u * tid;
        const uint idx1 = idx0 + 1u;
        const uint m0 = (idx0 < winLen) ? atomic_load_explicit(&maskWin[idx0], memory_order_relaxed) : 0u;
        const uint m1 = (idx1 < winLen) ? atomic_load_explicit(&maskWin[idx1], memory_order_relaxed) : 0u;
        const uint c0 = (m0 != 0u) ? 1u : 0u;
        const uint c1 = (m1 != 0u) ? 1u : 0u;
        const uint c = c0 + c1;
        const uint sgPrefix = simd_prefix_exclusive_sum(c);
        const uint sgSum = simd_sum(c);
        if (lane == 0u) sSgCounts[sg] = sgSum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint run = outCount;
            for (uint s = 0u; s < 8u; ++s) {
                sSgBases[s] = run;
                run += sSgCounts[s];
            }
            sNextOutCount = run;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint localOut = sSgBases[sg] + sgPrefix;
        if (c0 != 0u && localOut < tileListStride) {
            tileBlockList[slotBase + localOut] = int(winBase + idx0);
            tileBlockMask[slotBase + localOut] = m0;
            localOut += 1u;
        }
        if (c1 != 0u && localOut < tileListStride) {
            tileBlockList[slotBase + localOut] = int(winBase + idx1);
            tileBlockMask[slotBase + localOut] = m1;
        }
        outCount = min(sNextOutCount, tileListStride);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) tileBlockCount[(size_t)tileIdx * kvHeads + kvh] = outCount;
}
