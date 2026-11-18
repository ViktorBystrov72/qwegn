// qwattn_paged_sparse.metal — Sparse paged KV decode and prefill kernels for LSA.
// Attends over selected 4-token blocks `blockList[0..<blockCount]` plus the
// unaligned tail `[4 * ((pos + 1) / 4) .. pos]`, supporting f16, q8, and q4 KV pools.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "include/qw_attn_sparse_decode.metalh"

using namespace metal;
using namespace mlx::steel;

#define QW_LSA_KU           16u
#define QW_LSA_BK           32u
#define QW_LSA_NAX_HDIM     256u

// 1. Sparse Split-K Decode & Batched Decode over Paged KV Pool (f16, q8, q4)
#define QW_DECODE_SPARSE_ENTRY(SFX, MODE) \
kernel void qwg_attn_decode_sparse##SFX( \
        const device half* qgate         [[buffer(0)]], \
        device half*       kpool         [[buffer(1)]], \
        device half*       vpool         [[buffer(2)]], \
        const device int*  ktab          [[buffer(3)]], \
        const device int*  vtab          [[buffer(4)]], \
        const device half* knew          [[buffer(5)]], \
        const device half* vnew          [[buffer(6)]], \
        device float*      partials      [[buffer(7)]], \
        constant uint&     C             [[buffer(8)]], \
        constant uint&     splits        [[buffer(9)]], \
        device half*       outp          [[buffer(10)]], \
        const device int*  blockList     [[buffer(11)]], \
        const device uint* blockCountBuf [[buffer(12)]], \
        uint tid [[thread_index_in_threadgroup]], \
        uint3 tg [[threadgroup_position_in_grid]]) { \
    threadgroup half stageK[16u * QW_LSA_HEAD_DIM_MAX], stageV[16u * QW_LSA_HEAD_DIM_MAX]; \
    lsa_decode_sparse_core<MODE, false>(qgate, kpool, vpool, ktab, vtab, knew, vnew, partials, outp, \
                                        blockList, blockCountBuf, C, splits, 0u, true, tid, tg, stageK, stageV); \
} \
kernel void qwg_attn_decode_sparse_batched##SFX( \
        const device half* qgate         [[buffer(0)]], \
        device half*       kpool         [[buffer(1)]], \
        device half*       vpool         [[buffer(2)]], \
        const device int*  ktab          [[buffer(3)]], \
        const device int*  vtab          [[buffer(4)]], \
        const device half* knew          [[buffer(5)]], \
        const device half* vnew          [[buffer(6)]], \
        device float*      partials      [[buffer(7)]], \
        constant uint&     cBase         [[buffer(8)]], \
        constant uint&     splits        [[buffer(9)]], \
        device half*       outp          [[buffer(10)]], \
        const device int*  blockList     [[buffer(11)]], \
        const device uint* blockCountBuf [[buffer(12)]], \
        constant uint&     listStride    [[buffer(13)]], \
        constant uint&     writeKV       [[buffer(14)]], \
        uint tid [[thread_index_in_threadgroup]], \
        uint3 tg [[threadgroup_position_in_grid]]) { \
    threadgroup half stageK[16u * QW_LSA_HEAD_DIM_MAX], stageV[16u * QW_LSA_HEAD_DIM_MAX]; \
    lsa_decode_sparse_core<MODE, true>(qgate, kpool, vpool, ktab, vtab, knew, vnew, partials, outp, \
                                       blockList, blockCountBuf, cBase, splits, listStride, writeKV != 0u, tid, tg, stageK, stageV); \
}
QW_DECODE_SPARSE_ENTRY(, 0u)
QW_DECODE_SPARSE_ENTRY(_q8, 1u)
QW_DECODE_SPARSE_ENTRY(_q4, 2u)
#undef QW_DECODE_SPARSE_ENTRY

// -----------------------------------------------------------------------------
// 2. General Sparse Paged Prefill (scalar simdgroup per (head, query), f16/q8/q4)
// -----------------------------------------------------------------------------
kernel void qwg_attn_prefill_paged_general_sparse(
        const device half* qgate         [[buffer(0)]],
        device half*       kpool         [[buffer(1)]],
        device half*       vpool         [[buffer(2)]],
        const device int*  ktab          [[buffer(3)]],
        const device int*  vtab          [[buffer(4)]],
        device half*       outp          [[buffer(5)]],
        constant uint&     pos           [[buffer(6)]],
        constant uint&     q8mode        [[buffer(7)]],
        const device int*  blockList     [[buffer(8)]],
        const device uint* blockCountBuf [[buffer(9)]],
        constant uint&     listStride    [[buffer(10)]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 tg [[threadgroup_position_in_grid]]) {
    const uint head = tg.x;
    const uint qi   = tg.y;
    const uint lane = tid & 31u;
    const uint d0   = lane * 8u;
    const bool live = d0 < QW_HEAD_DIM;
    const uint kvh  = head / QW_GQA_RATIO;

    const device half* qrow = qgate + (uint64_t)qi * QW_QPROJ_OUT
                            + (uint64_t)head * QW_Q_STRIDE * QW_HEAD_DIM;
    float q8[8];
    for (uint i = 0u; i < 8u; i++) q8[i] = 0.0f;
    if (live) {
        for (uint i = 0u; i < 8u; i++) q8[i] = float(qrow[d0 + i]);
    }
    float gf[8];
    for (uint i = 0u; i < 8u; i++) gf[i] = 1.0f;
    if (QW_HAS_Q_GATE && live) {
        const device half* grow = qrow + QW_HEAD_DIM;
        for (uint i = 0u; i < 8u; i++) {
            gf[i] = 1.0f / (1.0f + metal::exp(-float(grow[d0 + i])));
        }
    }

    const uint kvRow  = QW_FC_KV_ROW;
    const uint kvPgel = QW_FC_KV_PGEL;
    const uint kvEnd  = pos + qi + 1u;
    const uint totalCompleteBlocks = kvEnd >> 2u;
    const uint uCount = min(blockCountBuf[qi], totalCompleteBlocks);
    const uint tailStart = totalCompleteBlocks * 4u;
    const uint nTail = (kvEnd > tailStart) ? (kvEnd - tailStart) : 0u;
    const uint totalSlots = uCount * 4u + nTail;
    const device int* myList = blockList + uint64_t(qi) * listStride;

    float m = -1e30f, lsum = 0.0f;
    float acc[8];
    for (uint i = 0u; i < 8u; i++) acc[i] = 0.0f;

    for (uint slotIdx = 0u; slotIdx < totalSlots; slotIdx++) {
        const uint j = (slotIdx < uCount * 4u)
            ? (uint(myList[slotIdx >> 2u]) * 4u + (slotIdx & 3u))
            : (tailStart + (slotIdx - uCount * 4u));
        if (j >= kvEnd) continue;
        const uint page = j >> 8u, slot = j & 255u;
        const uint pk = (uint)ktab[page];
        const uint pv = (uint)vtab[page];
        float k8[8], v8[8];
        for (uint i = 0u; i < 8u; i++) { k8[i] = 0.0f; v8[i] = 0.0f; }
        if (live) {
            if (q8mode == 0u) {
                const uint64_t rowOff = (uint64_t)slot * kvRow + (uint64_t)kvh * QW_HEAD_DIM;
                device const half* sk = kpool + (uint64_t)pk * kvPgel + rowOff;
                device const half* sv = vpool + (uint64_t)pv * kvPgel + rowOff;
                for (uint i = 0u; i < 8u; i++) { k8[i] = float(sk[d0 + i]); v8[i] = float(sv[d0 + i]); }
            } else {
                half kh[8], vh[8];
                QW_KV_LOAD8(kh, kpool, pk, slot, kvh, d0, q8mode)
                QW_KV_LOAD8(vh, vpool, pv, slot, kvh, d0, q8mode)
                for (uint i = 0u; i < 8u; i++) { k8[i] = float(kh[i]); v8[i] = float(vh[i]); }
            }
        }
        float dot = 0.0f;
        for (uint i = 0u; i < 8u; i++) dot += q8[i] * k8[i];
        const float s = simd_sum(dot) * QW_ATTN_SCALE;
        const float mnew = max(m, s);
        const float corr = (m > -1e29f) ? metal::exp(m - mnew) : 1.0f;
        const float p = metal::exp(s - mnew);
        for (uint i = 0u; i < 8u; i++) acc[i] = acc[i] * corr + p * v8[i];
        lsum = lsum * corr + p;
        m = mnew;
    }

    half hv[8];
    const float linv = (lsum > 0.0f) ? (1.0f / lsum) : 0.0f;
    for (uint i = 0u; i < 8u; i++) {
        hv[i] = live ? half(acc[i] * linv * gf[i]) : half(0.0f);
    }
    fwht32_inplace(hv, lane);
    if (live) {
        device half* o = outp + (uint64_t)qi * (QW_Q_HEADS * QW_HEAD_DIM)
                       + (uint64_t)head * QW_HEAD_DIM + d0;
        for (uint i = 0u; i < 8u; i++) o[i] = hv[i];
    }
}

// 3. M5 NAX Cooperative Tensor Sparse Paged Prefill (f16, q8, q4)
// Uses the tile's last query row `qRef = min(tile0 + 31, T - 1)` block list on 16x32 NAX MMA tiles.
template <uint MODE, uint STAGE_HDIM = QW_LSA_NAX_HDIM>
static inline void lsa_prefill_paged_nax_sparse_body(
        const device half* qgate, const device half* kpool, const device half* vpool,
        const device int* ktab, const device int* vtab, device half* outp,
        constant uint& pos, constant uint& T,
        const device int* blockList, const device uint* blockCountBuf, constant uint& listStride,
        uint tid, uint sgi, uint3 tg, threadgroup half* kvblk, threadgroup half* vblk) {
    const uint kThreads   = 64u * QW_GQA_RATIO;
    const uint num_groups = QW_KV_HEADS;
    const uint kvh        = tg.x % num_groups;
    const uint tile0      = (tg.x / num_groups) * 32u;
    const uint headBase   = kvh * QW_GQA_RATIO;
    const uint qRef       = min(tile0 + 31u, T - 1u);
    const uint kEnd       = pos + min(tile0 + 32u, T);

    const uint totalCompleteBlocks = kEnd >> 2u;
    const uint uCount     = min(blockCountBuf[qRef], totalCompleteBlocks);
    const uint tailStart  = totalCompleteBlocks * 4u;
    const uint nTail      = (kEnd > tailStart) ? (kEnd - tailStart) : 0u;
    const uint totalSlots = uCount * 4u + nTail;
    const device int* myList = blockList + uint64_t(qRef) * listStride;

    const uint sgi_h     = sgi >> 1u;
    const uint sgi_local = sgi & 1u;
    const uint head      = headBase + sgi_h;
    const uint rowBase   = tile0 + sgi_local * QW_LSA_KU;

    const float scale2 = QW_ATTN_SCALE * 1.4426950408889634f;
    constexpr short kU = 16;
    constexpr short kMaxNq = short(STAGE_HDIM >> 4u);
    constexpr short kMaxPair = short(STAGE_HDIM >> 5u);
    const short nq = short(QW_HEAD_DIM >> 4u);
    const short npair = short(QW_HEAD_DIM >> 5u);

    const device half* qdev = qgate + (uint64_t)rowBase * QW_QPROJ_OUT
                              + (uint64_t)head * QW_Q_STRIDE * QW_HEAD_DIM;

    const short2 sc = BaseNAXFrag::get_coord();
    const short sm = sc.y, sn = sc.x;

    NAXTile<half, 1, 1> Qf[kMaxNq];
    const short validRows = short(clamp(int(T) - int(rowBase), 0, 16));
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < kMaxNq; id++) {
        if (id < nq) {
            if (validRows == 16)
                Qf[id].load(qdev + id * kU, int(QW_QPROJ_OUT));
            else
                Qf[id].load_safe(qdev + id * kU, int(QW_QPROJ_OUT), short2(kU, validRows));
        }
    }

    NAXTile<float, 1, kMaxNq> Otile;
    Otile.clear();

    float mrow[2], lrow[2];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) { mrow[i] = -1e30f; lrow[i] = 0.0f; }

    const uint kLoadVec8 = 4u * QW_HEAD_DIM;
    const uint kStageStride = STAGE_HDIM;

    for (uint s0 = 0u; s0 < totalSlots; s0 += QW_LSA_BK) {
        NAXTile<half, 1, 2> Pt;

        for (uint e = tid; e < kLoadVec8; e += kThreads) {
            const uint VecPerRow = QW_HEAD_DIM >> 3u;
            const uint kr = (QW_HEAD_DIM == 256u) ? (e >> 5u) : (e / VecPerRow);
            const uint dim0 = (QW_HEAD_DIM == 256u) ? ((e & 31u) << 3u) : ((e % VecPerRow) << 3u);
            const uint slotIdx = s0 + kr;
            const uint absKey = (slotIdx < uCount * 4u)
                ? (uint(myList[slotIdx >> 2u]) * 4u + (slotIdx & 3u))
                : (tailStart + (slotIdx - uCount * 4u));
            const uint ob = kr * kStageStride + dim0;
            if (slotIdx < totalSlots && absKey < kEnd) {
                const uint page = absKey >> 8u;
                const uint slot = absKey & 255u;
                const uint pk = (uint)ktab[page];
                const uint pv = (uint)vtab[page];
                if (MODE == 0u) {
                    const uint64_t rowOff = (uint64_t)slot * QW_ROW + (uint64_t)kvh * QW_HEAD_DIM + dim0;
                    device const half* sk = kpool + (uint64_t)pk * QW_PGEL + rowOff;
                    device const half* sv = vpool + (uint64_t)pv * QW_PGEL + rowOff;
                    *(threadgroup half4*)(kvblk + ob)      = *(device const half4*)sk;
                    *(threadgroup half4*)(kvblk + ob + 4u) = *(device const half4*)(sk + 4u);
                    *(threadgroup half4*)(vblk + ob)       = *(device const half4*)sv;
                    *(threadgroup half4*)(vblk + ob + 4u)  = *(device const half4*)(sv + 4u);
                } else if (MODE == 1u) {
                    qwg_q8_load8((const device char*)kpool, (uint64_t)pk * QW_Q8_PGE, slot, kvh, dim0, kvblk, ob);
                    qwg_q8_load8((const device char*)vpool, (uint64_t)pv * QW_Q8_PGE, slot, kvh, dim0, vblk, ob);
                } else {
                    qwg_q4_load8((const device uchar*)kpool, (uint64_t)pk * QW_Q4_PGE, slot, kvh, dim0, kvblk, ob);
                    qwg_q4_load8((const device uchar*)vpool, (uint64_t)pv * QW_Q4_PGE, slot, kvh, dim0, vblk, ob);
                }
            } else {
                *(threadgroup half4*)(kvblk + ob)      = half4(0.0h);
                *(threadgroup half4*)(kvblk + ob + 4u) = half4(0.0h);
                *(threadgroup half4*)(vblk + ob)       = half4(0.0h);
                *(threadgroup half4*)(vblk + ob + 4u)  = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        NAXTile<float, 1, 2> Stile;
        Stile.clear();
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < kMaxNq; id++) {
            if (id < nq) {
                NAXTile<half, 2, 1> Kt;
                Kt.template load<half, STAGE_HDIM, 1>(kvblk + id * kU);
                BaseNAXFrag::mma(Stile.frag_at(0, 0),
                                 Stile.frag_at(0, 1),
                                 Qf[id].frag_at(0, 0), metal::false_type{},
                                 Kt.frag_at(0, 0), Kt.frag_at(1, 0),
                                 metal::true_type{});
            }
        }

        const uint lastSlotInStep = s0 + QW_LSA_BK - 1u;
        const uint maxAbsKeyInStep = (lastSlotInStep < uCount * 4u)
            ? (uint(myList[lastSlotInStep >> 2u]) * 4u + (lastSlotInStep & 3u))
            : (tailStart + (lastSlotInStep - uCount * 4u));
        const bool isUnmasked = (s0 + QW_LSA_BK <= totalSlots) && (maxAbsKeyInStep <= pos + rowBase);

        if (isUnmasked) {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < 2; ik++) {
                thread auto& fg = Stile.frag_at(0, ik);
                STEEL_PRAGMA_UNROLL
                for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                    fg[loc] *= scale2;
                }
            }
        } else {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < 2; ik++) {
                thread auto& fg = Stile.frag_at(0, ik);
                const uint col0 = s0 + (uint)(ik * kU) + (uint)sn;
                STEEL_PRAGMA_UNROLL
                for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                    const uint i = (uint)(loc >> 2), j = (uint)(loc & 3);
                    const uint slotC = col0 + j;
                    const uint abs_cc = (slotC < uCount * 4u)
                        ? (uint(myList[slotC >> 2u]) * 4u + (slotC & 3u))
                        : (tailStart + (slotC - uCount * 4u));
                    const uint abs_rr = pos + rowBase + (uint)sm + i * 8u;
                    fg[loc] = (slotC < totalSlots && abs_cc <= abs_rr && abs_cc < (pos + T))
                        ? (fg[loc] * scale2)
                        : (-1e30f);
                }
            }
        }

        float mnew[2];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) mnew[i] = mrow[i];
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < 2; ik++) {
            thread auto& fg = Stile.frag_at(0, ik);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < BaseNAXFrag::kElemRows; i++) {
                float v = fg[i * 4 + 0] > fg[i * 4 + 1] ? fg[i * 4 + 0] : fg[i * 4 + 1];
                v = v > fg[i * 4 + 2] ? v : fg[i * 4 + 2];
                v = v > fg[i * 4 + 3] ? v : fg[i * 4 + 3];
                mnew[i] = mnew[i] > v ? mnew[i] : v;
            }
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            mnew[i] = max(mnew[i], simd_shuffle_xor(mnew[i], ushort(1)));
            mnew[i] = max(mnew[i], simd_shuffle_xor(mnew[i], ushort(8)));
        }

        float psum[2] = {0.0f, 0.0f};
        if (isUnmasked) {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < 2; ik++) {
                thread auto& sf = Stile.frag_at(0, ik);
                thread auto& pf = Pt.frag_at(0, ik);
                STEEL_PRAGMA_UNROLL
                for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                    const short i = loc >> 2;
                    const float x = fast::exp2(sf[loc] - mnew[i]);
                    pf[loc] = half(x);
                    psum[i] += x;
                }
            }
        } else {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < 2; ik++) {
                thread auto& sf = Stile.frag_at(0, ik);
                thread auto& pf = Pt.frag_at(0, ik);
                STEEL_PRAGMA_UNROLL
                for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                    const short i = loc >> 2;
                    const float x = (sf[loc] > -1e29f) ? fast::exp2(sf[loc] - mnew[i]) : 0.0f;
                    pf[loc] = half(x);
                    psum[i] += x;
                }
            }
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            psum[i] += simd_shuffle_xor(psum[i], ushort(1));
            psum[i] += simd_shuffle_xor(psum[i], ushort(8));
        }

        float corr[2];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            corr[i] = (mrow[i] > -1e29f) ? fast::exp2(mrow[i] - mnew[i]) : 0.0f;
            lrow[i] = lrow[i] * corr[i] + psum[i];
            mrow[i] = mnew[i];
        }

        if (corr[0] != 1.0f || corr[1] != 1.0f) {
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < kMaxNq; id++) {
                if (id < nq) {
                    thread auto& of = Otile.frag_at(0, id);
                    STEEL_PRAGMA_UNROLL
                    for (short loc = 0; loc < BaseNAXFrag::kElemsPerFrag; loc++) {
                        of[loc] *= corr[loc >> 2];
                    }
                }
            }
        }

        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < 2; ik++) {
            STEEL_PRAGMA_UNROLL
            for (short idp = 0; idp < kMaxPair; idp++) {
                if (idp < npair) {
                    const short id0 = idp * 2;
                    NAXTile<half, 1, 2> Vt;
                    Vt.template load<half, STAGE_HDIM, 1>(vblk + (ik * kU) * kStageStride + id0 * kU);
                    BaseNAXFrag::mma(Otile.frag_at(0, id0),
                                     Otile.frag_at(0, id0 + 1),
                                     Pt.frag_at(0, ik), metal::false_type{},
                                     Vt.frag_at(0, 0), Vt.frag_at(0, 1),
                                     metal::false_type{});
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float linv[2];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) linv[i] = (lrow[i] > 0.0f) ? (1.0f / lrow[i]) : 0.0f;

    const device half* gdev = qdev + QW_HEAD_DIM;
    device half* odev = outp + (uint64_t)rowBase * (QW_Q_HEADS * QW_HEAD_DIM)
                        + (uint64_t)head * QW_HEAD_DIM;

    STEEL_PRAGMA_UNROLL
    for (short idp = 0; idp < kMaxPair; idp++) {
        if (idp < npair) {
            const short id0 = idp * 2;
            const short id1 = id0 + 1;
            NAXTile<half, 1, 1> G0, G1;
            if (QW_HAS_Q_GATE) {
                if (validRows == 16) {
                    G0.load(gdev + id0 * kU, int(QW_QPROJ_OUT));
                    G1.load(gdev + id1 * kU, int(QW_QPROJ_OUT));
                } else {
                    G0.load_safe(gdev + id0 * kU, int(QW_QPROJ_OUT), short2(kU, validRows));
                    G1.load_safe(gdev + id1 * kU, int(QW_QPROJ_OUT), short2(kU, validRows));
                }
            }
            thread auto& of0 = Otile.frag_at(0, id0);
            thread auto& of1 = Otile.frag_at(0, id1);
            thread auto& gf0 = G0.frag_at(0, 0);
            thread auto& gf1 = G1.frag_at(0, 0);

            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                float w[8];
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; j++) {
                    const short loc = i * 4 + j;
                    const float g0 = QW_HAS_Q_GATE ? (1.0f / (1.0f + fast::exp(-float(gf0[loc])))) : 1.0f;
                    const float g1 = QW_HAS_Q_GATE ? (1.0f / (1.0f + fast::exp(-float(gf1[loc])))) : 1.0f;
                    w[j]     = of0[loc] * linv[i] * g0;
                    w[4 + j] = of1[loc] * linv[i] * g1;
                }
                if (QW_FWHT_ROTATED) {
                    const float a0 = w[0] + w[1], a1 = w[0] - w[1];
                    const float a2 = w[2] + w[3], a3 = w[2] - w[3];
                    const float b0 = a0 + a2,     b2 = a0 - a2;
                    const float b1 = a1 + a3,     b3 = a1 - a3;
                    const float c0 = w[4] + w[5], c1 = w[4] - w[5];
                    const float c2 = w[6] + w[7], c3 = w[6] - w[7];
                    const float d0 = c0 + c2,     d2 = c0 - c2;
                    const float d1 = c1 + c3,     d3 = c1 - c3;
                    w[0] = b0; w[1] = b1; w[2] = b2; w[3] = b3;
                    w[4] = d0; w[5] = d1; w[6] = d2; w[7] = d3;
                    const bool hi4  = (sn & 4) != 0;
                    const bool hi8  = (sn & 8) != 0;
                    STEEL_PRAGMA_UNROLL
                    for (short k = 0; k < 8; k++) {
                        const float p4 = simd_shuffle_xor(w[k], ushort(1));
                        w[k] = hi4 ? (p4 - w[k]) : (w[k] + p4);
                    }
                    STEEL_PRAGMA_UNROLL
                    for (short k = 0; k < 8; k++) {
                        const float p8 = simd_shuffle_xor(w[k], ushort(8));
                        w[k] = hi8 ? (p8 - w[k]) : (w[k] + p8);
                    }
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; j++) {
                        const float u = w[j], v = w[4 + j];
                        w[j]     = (u + v) * 0.17677669529663689f;
                        w[4 + j] = (u - v) * 0.17677669529663689f;
                    }
                }
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; j++) {
                    const short loc = i * 4 + j;
                    gf0[loc] = half(w[j]);
                    gf1[loc] = half(w[4 + j]);
                }
            }
            if (validRows == 16) {
                G0.store(odev + id0 * kU, int(QW_Q_HEADS * QW_HEAD_DIM));
                G1.store(odev + id1 * kU, int(QW_Q_HEADS * QW_HEAD_DIM));
            } else if (validRows > 0) {
                G0.store_safe(odev + id0 * kU, int(QW_Q_HEADS * QW_HEAD_DIM), short2(kU, validRows));
                G1.store_safe(odev + id1 * kU, int(QW_Q_HEADS * QW_HEAD_DIM), short2(kU, validRows));
            }
        }
    }
}

#define QW_NAX_SPARSE_PREFILL_ENTRY(SFX, MODE, STAGE_HDIM) \
kernel void qwg_attn_prefill_paged_nax_sparse_pair##SFX( \
        const device half*  qgate         [[buffer(0)]], \
        const device half*  kpool         [[buffer(1)]], \
        const device half*  vpool         [[buffer(2)]], \
        const device int*   ktab          [[buffer(3)]], \
        const device int*   vtab          [[buffer(4)]], \
        device half*        outp          [[buffer(5)]], \
        constant uint&      pos           [[buffer(6)]], \
        constant uint&      T             [[buffer(7)]], \
        const device int*   blockList     [[buffer(8)]], \
        const device uint*  blockCountBuf [[buffer(9)]], \
        constant uint&      listStride    [[buffer(10)]], \
        uint tid [[thread_index_in_threadgroup]], \
        uint sgi [[simdgroup_index_in_threadgroup]], \
        uint3 tg [[threadgroup_position_in_grid]]) { \
    threadgroup half kvblk[QW_LSA_BK * STAGE_HDIM]; \
    threadgroup half vblk[QW_LSA_BK * STAGE_HDIM]; \
    lsa_prefill_paged_nax_sparse_body<MODE, STAGE_HDIM>(qgate, kpool, vpool, ktab, vtab, outp, pos, T, \
                                                        blockList, blockCountBuf, listStride, tid, sgi, tg, kvblk, vblk); \
}
QW_NAX_SPARSE_PREFILL_ENTRY(, 0u, 256u)
QW_NAX_SPARSE_PREFILL_ENTRY(_q8, 1u, 256u)
QW_NAX_SPARSE_PREFILL_ENTRY(_q4, 2u, 256u)
QW_NAX_SPARSE_PREFILL_ENTRY(_hd128, 0u, 128u)
QW_NAX_SPARSE_PREFILL_ENTRY(_hd128_q8, 1u, 128u)
QW_NAX_SPARSE_PREFILL_ENTRY(_hd128_q4, 2u, 128u)
#undef QW_NAX_SPARSE_PREFILL_ENTRY
