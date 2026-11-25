// qw_shortconv.metal — Depthwise causal & centered isolated 1D ShortConv (K=3..7, FP32 acc).

#include <metal_stdlib>
using namespace metal;

struct QWShortConvParams {
    uint seqLen;
    uint channels;
    uint kernelSize;
    uint hasBias;
    uint mediaPrefixLen;
    uint hasStateIn;
    uint hasStateOut;
};

kernel void qwg_shortconv_causal_f16(
    device const half*     x       [[buffer(0)]],
    device const half*     weight  [[buffer(1)]],
    device const half*     bias    [[buffer(2)]],
    device const half*     stateIn [[buffer(3)]],
    device half*           out     [[buffer(4)]],
    device half*           stateOut[[buffer(5)]],
    constant QWShortConvParams& p  [[buffer(6)]],
    uint2 tid                      [[thread_position_in_grid]]
) {
    uint d = tid.x;
    uint t = tid.y;
    if (d >= p.channels || t >= p.seqLen) return;

    uint k = p.kernelSize;
    uint stateRows = (k > 1u) ? (k - 1u) : 0u;
    float acc = (p.hasBias != 0u) ? float(bias[d]) : 0.0f;

    device const half* wd = weight + (size_t)d * k;
    for (uint j = 0u; j < k; j++) {
        int srcT = int(t) - int(k - 1u - j);
        float v = 0.0f;
        if (srcT >= 0) {
            v = float(x[(size_t)srcT * p.channels + d]);
        } else if (stateRows > 0u && p.hasStateIn != 0u) {
            int sIdx = int(stateRows) + srcT;
            if (sIdx >= 0 && sIdx < int(stateRows)) {
                v = float(stateIn[(size_t)sIdx * p.channels + d]);
            }
        }
        acc += v * float(wd[j]);
    }
    out[(size_t)t * p.channels + d] = half(acc);

    // Emit final (K - 1) input rows into stateOut on the last token row for each channel.
    if (t + 1u == p.seqLen && stateRows > 0u && p.hasStateOut != 0u) {
        for (uint s = 0u; s < stateRows; s++) {
            int srcT = int(p.seqLen) - int(stateRows) + int(s);
            half val = 0.0h;
            if (srcT >= 0) {
                val = x[(size_t)srcT * p.channels + d];
            } else if (p.hasStateIn != 0u) {
                int inIdx = int(stateRows) + srcT;
                if (inIdx >= 0 && inIdx < int(stateRows)) {
                    val = stateIn[(size_t)inIdx * p.channels + d];
                }
            }
            stateOut[(size_t)s * p.channels + d] = val;
        }
    }
}

kernel void qwg_shortconv_centered_f16(
    device const half*     x       [[buffer(0)]],
    device const half*     weight  [[buffer(1)]],
    device const half*     bias    [[buffer(2)]],
    device half*           out     [[buffer(3)]],
    constant QWShortConvParams& p  [[buffer(4)]],
    uint2 tid                      [[thread_position_in_grid]]
) {
    uint d = tid.x;
    uint t = tid.y;
    if (d >= p.channels || t >= p.seqLen) return;

    uint k = p.kernelSize;
    int halfK = int((k > 0u ? (k - 1u) : 0u) / 2u);
    bool inMedia = (p.mediaPrefixLen > 0u) && (t < p.mediaPrefixLen);
    int winLo = inMedia ? 0 : int(p.mediaPrefixLen);
    int winHi = inMedia ? int(p.mediaPrefixLen) : int(p.seqLen);

    float acc = (p.hasBias != 0u) ? float(bias[d]) : 0.0f;
    device const half* wd = weight + (size_t)d * k;
    for (uint j = 0u; j < k; j++) {
        int srcT = int(t) + int(j) - halfK;
        if (srcT >= winLo && srcT < winHi) {
            acc += float(x[(size_t)srcT * p.channels + d]) * float(wd[j]);
        }
    }
    out[(size_t)t * p.channels + d] = half(acc);
}

struct QWShortConvGatedParams {
    uint seqLen;
    uint channels;
    uint kernelSize;
    uint hasBias;
};

// Gated Lfm3ShortConv: bcx[t, B|C|x] -> conv(B * x) * C -> out[t, d].
kernel void qwg_shortconv_gated_causal_f16(
    device const half*               bcx    [[buffer(0)]],
    device const half*               weight [[buffer(1)]],
    device const half*               bias   [[buffer(2)]],
    device half*                     out    [[buffer(3)]],
    constant QWShortConvGatedParams& p      [[buffer(4)]],
    uint2 tid                               [[thread_position_in_grid]]
) {
    uint d = tid.x;
    uint t = tid.y;
    if (d >= p.channels || t >= p.seqLen) return;

    uint k = p.kernelSize;
    size_t rowStride = (size_t)p.channels * 3u;
    float acc = (p.hasBias != 0u) ? float(bias[d]) : 0.0f;
    device const half* wd = weight + (size_t)d * k;
    for (uint j = 0u; j < k; j++) {
        int srcT = int(t) - int(k - 1u - j);
        if (srcT >= 0) {
            size_t base = (size_t)srcT * rowStride + d;
            float bVal = float(bcx[base]);
            float xVal = float(bcx[base + (size_t)p.channels * 2u]);
            acc += (bVal * xVal) * float(wd[j]);
        }
    }
    float cVal = float(bcx[(size_t)t * rowStride + p.channels + d]);
    out[(size_t)t * p.channels + d] = half(acc * cVal);
}
