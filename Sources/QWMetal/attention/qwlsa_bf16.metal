// qwlsa_bf16.metal — BF16 LSA index projection kernel (`qwg_lsa_index_prep_bf16`).
// Projects BF16 post-prep Q/K slices into L2-normalized F16 qIdxOut and raw F16 kPending.
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

static inline float lsa_dot_row_tg_bf16(threadgroup const float* x,
                                        device const half* row,
                                        uint inputDim,
                                        uint lane) {
    float acc = 0.0f;
    if ((inputDim & 3u) == 0u) {
        const uint n4 = inputDim >> 2u;
        device const half4* row4 = (device const half4*)row;
        for (uint j4 = lane; j4 < n4; j4 += 32u) {
            const float4 w = float4(row4[j4]);
            const uint j = j4 << 2u;
            acc = fma(x[j], w.x, fma(x[j + 1u], w.y, fma(x[j + 2u], w.z, fma(x[j + 3u], w.w, acc))));
        }
    } else {
        for (uint j = lane; j < inputDim; j += 32u) {
            acc = fma(x[j], float(row[j]), acc);
        }
    }
    return simd_sum(acc);
}

kernel void qwg_lsa_index_prep_bf16(device const bfloat* qIn        [[buffer(0)]],
                                    device const bfloat* kIn        [[buffer(1)]],
                                    device const half*   wqIdx      [[buffer(2)]],
                                    device const half*   wkIdx      [[buffer(3)]],
                                    device half*         qIdxOut    [[buffer(4)]],
                                    device half*         kPending   [[buffer(5)]],
                                    constant uint&       T          [[buffer(6)]],
                                    constant uint&       pos0       [[buffer(7)]],
                                    constant uint&       qHeads     [[buffer(8)]],
                                    constant uint&       kvHeads    [[buffer(9)]],
                                    constant uint&       headDim    [[buffer(10)]],
                                    constant uint&       qRowStride [[buffer(11)]],
                                    constant uint&       sliceStart [[buffer(12)]],
                                    constant uint&       sliceDim   [[buffer(13)]],
                                    constant uint&       indexDim   [[buffer(14)]],
                                    constant uint&       ringMask   [[buffer(15)]],
                                    constant uint&       qMode      [[buffer(16)]],
                                    uint t   [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
    if (t >= T) return;
    threadgroup float sliceBuf[2048];
    threadgroup float qProj[128];
    threadgroup float partial[4];

    const uint sg = tid >> 5u;
    const uint lane = tid & 31u;
    const uint inputDim = min(kvHeads * sliceDim, 2048u);
    const uint dCap = min(indexDim, 128u);
    const uint gqaRatio = max(1u, qHeads / max(1u, kvHeads));
    const float invGqa = 1.0f / float(gqaRatio);
    device const bfloat* kRow = kIn + uint64_t(t) * (kvHeads * headDim);

    for (uint idx = tid; idx < inputDim; idx += 128u) {
        const uint kvh = idx / sliceDim;
        const uint d = sliceStart + (idx - kvh * sliceDim);
        sliceBuf[idx] = float(kRow[kvh * headDim + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint ringSlot = (pos0 + t) & ringMask;
    for (uint ch = sg; ch < dCap; ch += 4u) {
        const float accK = lsa_dot_row_tg_bf16(sliceBuf, wkIdx + uint64_t(ch) * inputDim, inputDim, lane);
        if (lane == 0u) {
            kPending[uint64_t(ringSlot) * indexDim + ch] = half(accK);
        }
    }

    const bool needQ = (qMode == 1u) || (qMode == 2u && (((t & 31u) == 31u) || (t + 1u == T)));
    if (!needQ) return;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const bfloat* qRow = qIn + uint64_t(t) * qRowStride;
    for (uint idx = tid; idx < inputDim; idx += 128u) {
        const uint kvh = idx / sliceDim;
        const uint d = sliceStart + (idx - kvh * sliceDim);
        float qSum = 0.0f;
        const uint hBase = kvh * gqaRatio;
        for (uint r = 0u; r < gqaRatio; r++) {
            qSum += float(qRow[(hBase + r) * headDim + d]);
        }
        sliceBuf[idx] = qSum * invGqa;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint ch = sg; ch < dCap; ch += 4u) {
        const float accQ = lsa_dot_row_tg_bf16(sliceBuf, wqIdx + uint64_t(ch) * inputDim, inputDim, lane);
        if (lane == 0u) {
            qProj[ch] = accQ;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float qVal = (tid < dCap) ? qProj[tid] : 0.0f;
    const float ss = simd_sum(qVal * qVal);
    if (lane == 0u) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float qNormSq = partial[0] + partial[1] + partial[2] + partial[3];
    const float qInv = rsqrt(max(qNormSq, 1e-12f));
    if (tid < dCap) {
        qIdxOut[uint64_t(t) * indexDim + tid] = half(qVal * qInv);
    }
}
