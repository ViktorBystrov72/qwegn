// MQA-shared sparse latent attention on NAX tensor MMA (Apple10+, f16/q8/q4 KV);
// 1 token x 16 heads per 128T TG; decode/verify use split-K merged by combine.
#include "qw_mla_nax_common.metalh"

// Grid: (T, qHeads / 16, 1), Threadgroup: (128, 1, 1).
kernel void qwg_mla_sparse_attn_nax(
    device const half*           qAbs      [[buffer(0)]],
    device const half*           kPool     [[buffer(1)]],
    device const half*           vPool     [[buffer(2)]],
    device const int*            kTable    [[buffer(3)]],
    device const int*            vTable    [[buffer(4)]],
    device const int*            blockList [[buffer(5)]],
    device half*                 uLatOut   [[buffer(6)]],
    constant MLASparseAttnParams& p        [[buffer(7)]],
    uint2                        tg        [[threadgroup_position_in_grid]],
    uint                         tid       [[thread_index_in_threadgroup]],
    uint                         sg        [[simdgroup_index_in_threadgroup]],
    uint                         lane      [[thread_index_in_simdgroup]]
) {
    if (tg.x >= p.T) return;
    threadgroup float sPart[2 * 4 * 16 * 32];
    threadgroup uint  blkIdx[1024];
    if (p.kvMode == 0u) {
        qwg_mla_nax_attend<false, 0u, 1u>(qAbs, kPool, vPool, kTable, vTable, blockList, uLatOut, nullptr, p,
                                          tg.x, tg.y * MLA_NAX_HG, 0u, 1u, tid, 0u, sg, lane, sPart, blkIdx);
    } else if (p.kvMode == 1u) {
        qwg_mla_nax_attend<false, 1u, 1u>(qAbs, kPool, vPool, kTable, vTable, blockList, uLatOut, nullptr, p,
                                          tg.x, tg.y * MLA_NAX_HG, 0u, 1u, tid, 0u, sg, lane, sPart, blkIdx);
    } else {
        qwg_mla_nax_attend<false, 2u, 1u>(qAbs, kPool, vPool, kTable, vTable, blockList, uLatOut, nullptr, p,
                                          tg.x, tg.y * MLA_NAX_HG, 0u, 1u, tid, 0u, sg, lane, sPart, blkIdx);
    }
}

// Split-K partials: partO[(t*nSplit+z)*qHeads+h] and partML = (log2 max, sum).
// Grid: (T, qHeads / 16, nSplit), Threadgroup: (128, 1, 1).
kernel void qwg_mla_sparse_attn_nax_split(
    device const half*           qAbs      [[buffer(0)]],
    device const half*           kPool     [[buffer(1)]],
    device const half*           vPool     [[buffer(2)]],
    device const int*            kTable    [[buffer(3)]],
    device const int*            vTable    [[buffer(4)]],
    device const int*            blockList [[buffer(5)]],
    device half*                 partO     [[buffer(6)]],
    constant MLASparseAttnParams& p        [[buffer(7)]],
    device float2*               partML    [[buffer(8)]],
    uint3                        tg        [[threadgroup_position_in_grid]],
    uint3                        ntg       [[threadgroups_per_grid]],
    uint                         tid       [[thread_index_in_threadgroup]],
    uint                         sg        [[simdgroup_index_in_threadgroup]],
    uint                         lane      [[thread_index_in_simdgroup]]
) {
    if (tg.x >= p.T) return;
    threadgroup float sPart[2 * 4 * 16 * 32];
    threadgroup uint  blkIdx[1024];
    if (p.kvMode == 0u) {
        qwg_mla_nax_attend<true, 0u, 1u>(qAbs, kPool, vPool, kTable, vTable, blockList, partO, partML, p,
                                         tg.x, tg.y * MLA_NAX_HG, tg.z, ntg.z, tid, 0u, sg, lane, sPart, blkIdx);
    } else if (p.kvMode == 1u) {
        qwg_mla_nax_attend<true, 1u, 1u>(qAbs, kPool, vPool, kTable, vTable, blockList, partO, partML, p,
                                         tg.x, tg.y * MLA_NAX_HG, tg.z, ntg.z, tid, 0u, sg, lane, sPart, blkIdx);
    } else {
        qwg_mla_nax_attend<true, 2u, 1u>(qAbs, kPool, vPool, kTable, vTable, blockList, partO, partML, p,
                                         tg.x, tg.y * MLA_NAX_HG, tg.z, ntg.z, tid, 0u, sg, lane, sPart, blkIdx);
    }
}

// Grid: (T * qHeads, 1, 1), Threadgroup: (128, 1, 1).
kernel void qwg_mla_sparse_attn_combine(
    device const half*           partO     [[buffer(0)]],
    device const float2*         partML    [[buffer(1)]],
    device half*                 uLatOut   [[buffer(2)]],
    constant MLASparseAttnParams& p        [[buffer(3)]],
    constant uint&               nSplit    [[buffer(4)]],
    uint                         tg        [[threadgroup_position_in_grid]],
    uint                         tid       [[thread_index_in_threadgroup]]
) {
    const uint t = tg / p.qHeads;
    const uint h = tg - t * p.qHeads;
    if (t >= p.T) return;
    const size_t base = (size_t)t * nSplit * p.qHeads + h;
    float mMax = -1e30f;
    for (uint z = 0u; z < nSplit; ++z) {
        const float2 ml = partML[base + (size_t)z * p.qHeads];
        if (ml.y > 0.0f) mMax = max(mMax, ml.x);
    }
    float4 acc = float4(0.0f);
    float wSum = 0.0f;
    for (uint z = 0u; z < nSplit; ++z) {
        const size_t row = base + (size_t)z * p.qHeads;
        const float2 ml = partML[row];
        if (ml.y <= 0.0f) continue;
        const float w = ml.y * fast::exp2(ml.x - mMax);
        wSum += w;
        acc += w * float4(*(device const half4*)(partO + row * MLA_NAX_RANK + tid * 4u));
    }
    const float inv = (wSum > 0.0f) ? (1.0f / wSum) : 0.0f;
    *(device half4*)(uLatOut + ((size_t)t * p.qHeads + h) * MLA_NAX_RANK + tid * 4u) = half4(acc * inv);
}

kernel void qwg_mla_dsa_sparse_attn_nax(
    device const half*              qAbs      [[buffer(0)]],
    device const half*              kPool     [[buffer(1)]],
    device const half*              vPool     [[buffer(2)]],
    device const int*               kTable    [[buffer(3)]],
    device const int*               vTable    [[buffer(4)]],
    device const int*               blockList [[buffer(5)]],
    device half*                    uLatOut   [[buffer(6)]],
    constant MLADSASparseAttnParams& p        [[buffer(7)]],
    device const half*              mlaQFull  [[buffer(8)]],
    device const half*              kPeCache  [[buffer(9)]],
    device const float*             attnSink  [[buffer(10)]],
    uint2                           tg        [[threadgroup_position_in_grid]],
    uint                            tid       [[thread_index_in_threadgroup]],
    uint                            sg        [[simdgroup_index_in_threadgroup]],
    uint                            lane      [[thread_index_in_simdgroup]]
) {
    if (tg.x >= p.T) return;
    threadgroup float sPart[2 * 4 * 16 * 32];
    threadgroup uint  blkIdx[2560];
    if (p.kvMode == 0u) {
        qw_mla_dsa_nax_attend<false, 0u, 1u>(qAbs, kPool, vPool, kTable, vTable, blockList, uLatOut, nullptr, p,
                                             mlaQFull, kPeCache, attnSink, tg.x, tg.y * MLA_NAX_HG, 0u, 1u, tid, 0u, sg, lane, sPart, blkIdx);
    } else if (p.kvMode == 1u) {
        qw_mla_dsa_nax_attend<false, 1u, 1u>(qAbs, kPool, vPool, kTable, vTable, blockList, uLatOut, nullptr, p,
                                             mlaQFull, kPeCache, attnSink, tg.x, tg.y * MLA_NAX_HG, 0u, 1u, tid, 0u, sg, lane, sPart, blkIdx);
    } else {
        qw_mla_dsa_nax_attend<false, 2u, 1u>(qAbs, kPool, vPool, kTable, vTable, blockList, uLatOut, nullptr, p,
                                             mlaQFull, kPeCache, attnSink, tg.x, tg.y * MLA_NAX_HG, 0u, 1u, tid, 0u, sg, lane, sPart, blkIdx);
    }
}

kernel void qwg_mla_dsa_sparse_attn_nax_split(
    device const half*              qAbs      [[buffer(0)]],
    device const half*              kPool     [[buffer(1)]],
    device const half*              vPool     [[buffer(2)]],
    device const int*               kTable    [[buffer(3)]],
    device const int*               vTable    [[buffer(4)]],
    device const int*               blockList [[buffer(5)]],
    device half*                    partO     [[buffer(6)]],
    constant MLADSASparseAttnParams& p        [[buffer(7)]],
    device float2*                  partML    [[buffer(8)]],
    device const half*              mlaQFull  [[buffer(9)]],
    device const half*              kPeCache  [[buffer(10)]],
    device const float*             attnSink  [[buffer(11)]],
    uint3                           tg        [[threadgroup_position_in_grid]],
    uint3                           ntg       [[threadgroups_per_grid]],
    uint                            tid       [[thread_index_in_threadgroup]],
    uint                            sg        [[simdgroup_index_in_threadgroup]],
    uint                            lane      [[thread_index_in_simdgroup]]
) {
    if (tg.x >= p.T) return;
    threadgroup float sPart[2 * 4 * 16 * 32];
    threadgroup uint  blkIdx[2560];
    if (p.kvMode == 0u) {
        qw_mla_dsa_nax_attend<true, 0u, 1u>(qAbs, kPool, vPool, kTable, vTable, blockList, partO, partML, p,
                                            mlaQFull, kPeCache, attnSink, tg.x, tg.y * MLA_NAX_HG, tg.z, ntg.z, tid, 0u, sg, lane, sPart, blkIdx);
    } else if (p.kvMode == 1u) {
        qw_mla_dsa_nax_attend<true, 1u, 1u>(qAbs, kPool, vPool, kTable, vTable, blockList, partO, partML, p,
                                            mlaQFull, kPeCache, attnSink, tg.x, tg.y * MLA_NAX_HG, tg.z, ntg.z, tid, 0u, sg, lane, sPart, blkIdx);
    } else {
        qw_mla_dsa_nax_attend<true, 2u, 1u>(qAbs, kPool, vPool, kTable, vTable, blockList, partO, partML, p,
                                            mlaQFull, kPeCache, attnSink, tg.x, tg.y * MLA_NAX_HG, tg.z, ntg.z, tid, 0u, sg, lane, sPart, blkIdx);
    }
}

kernel void qwg_mla_dsa_sparse_attn_combine(
    device const half*              partO     [[buffer(0)]],
    device const float2*            partML    [[buffer(1)]],
    device half*                    uLatOut   [[buffer(2)]],
    constant MLADSASparseAttnParams& p        [[buffer(3)]],
    constant uint&                  nSplit    [[buffer(4)]],
    device const float*             attnSink  [[buffer(5)]],
    uint                            tg        [[threadgroup_position_in_grid]],
    uint                            tid       [[thread_index_in_threadgroup]]
) {
    const uint t = tg / p.qHeads;
    const uint h = tg - t * p.qHeads;
    if (t >= p.T) return;
    const size_t base = (size_t)t * nSplit * p.qHeads + h;
    float sink2 = -1e30f;
    bool validSink = false;
    if (p.hasSink != 0u) {
        const float sk = attnSink[h] * 1.4426950408889634f;
        if (sk > -1e29f) { sink2 = sk; validSink = true; }
    }
    float mMax = sink2;
    for (uint z = 0u; z < nSplit; ++z) {
        const float2 ml = partML[base + (size_t)z * p.qHeads];
        if (ml.y > 0.0f) mMax = max(mMax, ml.x);
    }
    float4 acc = float4(0.0f);
    float wSum = validSink ? fast::exp2(sink2 - mMax) : 0.0f;
    for (uint z = 0u; z < nSplit; ++z) {
        const size_t row = base + (size_t)z * p.qHeads;
        const float2 ml = partML[row];
        if (ml.y <= 0.0f) continue;
        const float w = ml.y * fast::exp2(ml.x - mMax);
        wSum += w;
        acc += w * float4(*(device const half4*)(partO + row * MLA_NAX_RANK + tid * 4u));
    }
    const float inv = (wSum > 0.0f) ? (1.0f / wSum) : 0.0f;
    *(device half4*)(uLatOut + ((size_t)t * p.qHeads + h) * MLA_NAX_RANK + tid * 4u) = half4(acc * inv);
}
