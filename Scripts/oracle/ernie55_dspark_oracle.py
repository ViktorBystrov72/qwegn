"""2-layer PyTorch parity oracle for ERNIE-5.5 target taps, DSpark draft, and FlashIndexer.
Matches vLLM-ERNIE (dspark.py, model.py, attention.py, flashindexer.py, mhc.py) and QWP1 packs
on deterministic synthetic checkpoints, comparing HF BF16/MXFP4 vs packed/repacked QWP1 weights.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "quantize"))

from ernie55_adapter import ShardReader, quantize_ernie55_model  # noqa: E402
from hq_blocks import dequant_hq8_g32_m5, hq_block_bytes  # noqa: E402
from shvq import any_block_bytes, dequant_any_blocks  # noqa: E402
from verify import read_header  # noqa: E402


def _rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return y if weight is None else y * weight.float()


def _situ_and_mul(gate: torch.Tensor, up: torch.Tensor, beta: float = 4.0, lin_beta: float = 25.0) -> torch.Tensor:
    gf, uf = gate.float(), up.float()
    return (beta * torch.tanh(gf / beta) * torch.sigmoid(gf)) * (lin_beta * torch.tanh(uf / lin_beta))


def _multimax(logits: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    r, t = params[:4].float(), params[4:8].float()
    xf = logits.float()
    return (xf + t[0] * F.relu(r[0] - xf) + t[1] * F.relu(xf - r[1])
            + t[2] * F.relu(r[2] - xf).pow(2) + t[3] * F.relu(xf - r[3]).pow(2))


def _rope_gptj(x: torch.Tensor, positions: torch.Tensor, rope_dim: int, theta: float, *,
               inverse: bool = False, yarn_scale: dict | None = None) -> torch.Tensor:
    if rope_dim <= 0:
        return x
    xf = x.float().clone()
    tail = xf[..., -rope_dim:]
    half = rope_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    mscale = 1.0
    if yarn_scale is not None and str(yarn_scale.get("type", "")).lower() == "yarn":
        factor = float(yarn_scale.get("factor", 1.0))
        orig_max = float(yarn_scale.get("original_max_position_embeddings", 65536))
        beta_fast = float(yarn_scale.get("beta_fast", 32.0))
        beta_slow = float(yarn_scale.get("beta_slow", 1.0))
        low = max(0, math.floor(rope_dim * math.log(orig_max / (beta_fast * 2.0 * math.pi)) / (2.0 * math.log(theta))))
        high = min(half - 1, math.ceil(rope_dim * math.log(orig_max / (beta_slow * 2.0 * math.pi)) / (2.0 * math.log(theta))))
        ramp = ((torch.arange(half, dtype=torch.float32) - low) / max(1, high - low)).clamp(0.0, 1.0)
        smooth = 1.0 - ramp
        inv_freq = (inv_freq / factor) * (1.0 - smooth) + inv_freq * smooth
        mscale = 0.1 * math.log(factor) + 1.0 if factor > 1.0 else 1.0
    ang = positions.float().unsqueeze(-1) * inv_freq
    c = torch.cos(ang) * mscale
    s = (torch.sin(ang) * mscale) * (-1.0 if inverse else 1.0)
    while c.ndim < tail.ndim:
        c, s = c.unsqueeze(-2), s.unsqueeze(-2)
    even, odd = tail[..., 0::2], tail[..., 1::2]
    xf[..., -rope_dim::2] = even * c - odd * s
    xf[..., -rope_dim + 1::2] = odd * c + even * s
    return xf


def _mhc_pre(streams: torch.Tensor, fn_w: torch.Tensor, bias: torch.Tensor, norm_w: torch.Tensor,
             rms_eps: float, hc_eps: float, sinkhorn_iters: int = 20) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t, c, h = streams.shape
    flat = streams.float().reshape(t, c * h)
    rstd = 1.0 / (torch.sqrt(flat.pow(2).mean(dim=-1, keepdim=True)) + rms_eps)
    logits = (flat @ fn_w.float().T) * rstd + bias.float()
    pre = torch.sigmoid(logits[:, :c]) + hc_eps
    post = torch.sigmoid(logits[:, c:2 * c]) * 2.0
    comb = torch.softmax(logits[:, 2 * c:].reshape(t, c, c), dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(max(0, sinkhorn_iters - 1)):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    layer_in = torch.einsum("tc,tch->th", pre, streams.float())
    return pre, post, comb, _rms_norm(layer_in, norm_w, rms_eps)


def _mhc_post(streams: torch.Tensor, branch_out: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    return torch.einsum("tij,tjh->tih", comb.float(), streams.float()) + post.unsqueeze(-1) * branch_out.float().unsqueeze(1)


def _mhc_exit_contract(streams: torch.Tensor, head_w: torch.Tensor, head_b: torch.Tensor,
                       rms_eps: float, hc_eps: float) -> torch.Tensor:
    t, c, h = streams.shape
    flat = streams.float().reshape(t, c * h)
    rstd = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + rms_eps)
    pre = torch.sigmoid((flat @ head_w.float().T) * rstd + head_b.float()) + hc_eps
    return torch.einsum("tc,tch->th", pre, streams.float())


def _sink_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float,
                    attn_mask: torch.Tensor, sink: torch.Tensor | None) -> torch.Tensor:
    # q: [Tq, Hq, D], k: [Tk, Hkv, D], v: [Tk, Hkv, Dv]
    tq, hq, _ = q.shape
    hkv = k.shape[1]
    if hkv == 1 and hq > 1:
        k, v = k.expand(-1, hq, -1), v.expand(-1, hq, -1)
    elif hkv != hq:
        k = k.repeat_interleave(hq // hkv, dim=1)
        v = v.repeat_interleave(hq // hkv, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    scores = scores.masked_fill(~attn_mask.unsqueeze(0), float("-inf"))
    if sink is None:
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum("hqk,khd->qhd", probs, v.float())
    s_col = sink.float().view(hq, 1, 1).expand(hq, tq, 1)
    all_s = torch.cat([scores, s_col], dim=-1)
    m = all_s.amax(dim=-1, keepdim=True)
    exp_s = torch.exp(scores - m)
    denom = exp_s.sum(dim=-1, keepdim=True) + torch.exp(s_col - m)
    probs = exp_s / denom
    return torch.einsum("hqk,khd->qhd", probs, v.float())


def _run_moe(x: torch.Tensor, w: dict[str, torch.Tensor], blk: str, tc: dict, *, stacked: bool) -> torch.Tensor:
    t, _ = x.shape
    e_cnt = int(tc.get("n_routed_experts") or tc.get("num_experts"))
    topk = int(tc.get("num_experts_per_tok", 2))
    beta, lbeta = float(tc.get("activation_situ_beta", 4.0)), float(tc.get("activation_situ_linear_beta", 25.0))
    split_f = bool(tc.get("moe_split_feature_routing", False))
    norm_topk = bool(tc.get("norm_topk_prob", True))
    logits = x.float() @ w[blk + "mlp.gate.weight"].float().T
    s = torch.sigmoid(logits[:, :e_cnt]) + (torch.sigmoid(logits[:, e_cnt:]) if split_f else 0.0)
    bias = w.get(blk + "mlp.gate.e_score_correction_bias")
    score = s + (bias.float().unsqueeze(0) if bias is not None else 0.0)
    top_idx = torch.topk(score, k=topk, dim=-1).indices
    wts = s.gather(1, top_idx)
    if norm_topk:
        wts = wts / (wts.sum(dim=-1, keepdim=True) + 1e-20)
    sc_param = w.get(blk + "mlp.routed_scaling_factor_param")
    if sc_param is not None:
        wts = wts * sc_param.float()[top_idx]
    else:
        wts = wts * float(tc.get("routed_scaling_factor", 1.0))

    h_lat = x.float() @ w[blk + "mlp.fc1.weight"].float().T
    if (ln := w.get(blk + "mlp.latent_norm.weight")) is not None:
        h_lat = _rms_norm(h_lat, ln, float(tc.get("rms_norm_eps", 1e-6)))

    routed_lat = torch.zeros_like(h_lat)
    for ti in range(t):
        for r in range(topk):
            eid = int(top_idx[ti, r].item())
            if stacked:
                gu = w[blk + "mlp.experts.gate_up_proj"].float()[eid]
                half = gu.shape[0] // 2
                g_w, u_w = gu[:half], gu[half:]
                d_w = w[blk + "mlp.experts.down_proj"].float()[eid]
            else:
                g_w = w[blk + f"mlp.experts.{eid}.gate_proj.weight"].float()
                u_w = w[blk + f"mlp.experts.{eid}.up_proj.weight"].float()
                d_w = w[blk + f"mlp.experts.{eid}.down_proj.weight"].float()
            act = _situ_and_mul(h_lat[ti] @ g_w.T, h_lat[ti] @ u_w.T, beta=beta, lin_beta=lbeta)
            routed_lat[ti] += wts[ti, r] * (act @ d_w.T)

    out = routed_lat @ w[blk + "mlp.fc2.weight"].float().T
    if (blk + "mlp.shared_experts.gate_proj.weight") in w:
        sh_act = _situ_and_mul(
            x.float() @ w[blk + "mlp.shared_experts.gate_proj.weight"].float().T,
            x.float() @ w[blk + "mlp.shared_experts.up_proj.weight"].float().T,
            beta=beta, lin_beta=lbeta,
        )
        out = out + sh_act @ w[blk + "mlp.shared_experts.down_proj.weight"].float().T
    return out


def _run_hca_common_out(attn_ctx_out: torch.Tensor, qr: torch.Tensor, o_gate_logits: torch.Tensor, positions: torch.Tensor,
                        w: dict[str, torch.Tensor], blk: str, nope_d: int, rope_d: int, theta: float,
                        o_groups: int, yarn_scale: dict | None) -> torch.Tensor:
    t, q_heads, head_dim = attn_ctx_out.shape
    o = attn_ctx_out
    if (vha := w.get(blk + "self_attn.vha_mix.weight")) is not None:
        o = torch.einsum("hg,tgd->thd", vha.float(), o)
    o = _rope_gptj(o, positions, rope_d, theta, inverse=True, yarn_scale=yarn_scale)
    o_grp = o.reshape(t, o_groups, (q_heads // o_groups) * head_dim)
    wo_a = w[blk + "self_attn.wo_a.weight"].float().reshape(o_groups, -1, (q_heads // o_groups) * head_dim)
    z = torch.einsum("goi,tgi->tgo", wo_a, o_grp).reshape(t, -1)
    z = z * torch.sigmoid(o_gate_logits.float())
    return z @ w[blk + "self_attn.o_proj.weight"].float().T


def flashindexer_select_heads_ref(hidden: torch.Tensor, q_idx: torch.Tensor, weights: torch.Tensor,
                                  gate_weight: torch.Tensor, bias: torch.Tensor, topk_heads: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gates = hidden.float() @ gate_weight.float().T
    scores = gates + bias.float().unsqueeze(0)
    heads = torch.topk(scores, k=topk_heads, dim=-1).indices
    q_sel = torch.gather(q_idx.float(), 1, heads.unsqueeze(-1).expand(-1, -1, q_idx.shape[-1]))
    w_sel = weights.float().gather(1, heads) * gates.gather(1, heads)
    return q_sel, w_sel, heads


def run_target_and_capture(input_ids: torch.Tensor, w: dict[str, torch.Tensor], tc: dict) -> dict[str, torch.Tensor]:
    t = input_ids.shape[0]
    positions = torch.arange(t, dtype=torch.long)
    rms_eps, hc_eps = float(tc.get("rms_norm_eps", 1e-6)), float(tc.get("hc_eps", 1e-6))
    streams_n, sinkhorn = int(tc.get("hc_mult", 4)), int(tc.get("hc_sinkhorn_iters", 20))
    theta, comp_theta = float(tc.get("rope_theta", 8000000.0)), float(tc.get("compress_rope_theta", 160000.0))
    yarn = tc.get("rope_scaling")
    streams = w["model.embed_tokens.weight"].float()[input_ids].unsqueeze(1).repeat(1, streams_n, 1)
    taps: dict[int, torch.Tensor] = {}
    fi_out: dict[str, torch.Tensor] = {}
    target_ids = {int(i) for i in tc.get("dspark_target_layer_ids", [])}

    for l in range(int(tc["num_hidden_layers"])):
        blk = f"model.layers.{l}."
        _, post_a, comb_a, x = _mhc_pre(streams, w[blk + "mhc_attn.weight"], w[blk + "mhc_attn.bias"],
                                        w[blk + "input_layernorm.weight"], rms_eps, hc_eps, sinkhorn)
        c_ratios = tc.get("compress_ratios") or []
        is_comp = l < len(c_ratios) and int(c_ratios[l]) > 0
        if is_comp:
            comp_r = int(c_ratios[l])
            qh, hd = int(tc["num_attention_heads"]), int(tc["head_dim"])
            qr_rank, o_groups = int(tc["q_lora_rank"]), max(1, int(tc.get("o_groups", 1)))
            nope_d, rope_d = int(tc["qk_nope_head_dim"]), int(tc["qk_rope_head_dim"])
            fused = x @ w[blk + "self_attn.fused_in_proj.weight"].float().T
            qr, kv_r, c_kv, c_sc = fused.split([qr_rank, hd, hd, hd], dim=-1)
            qr = _rms_norm(qr, w[blk + "self_attn.q_norm.weight"], rms_eps)
            kv_r = _rope_gptj(_rms_norm(kv_r, w[blk + "self_attn.kv_norm.weight"], rms_eps),
                              positions, rope_d, comp_theta, yarn_scale=yarn)
            qupg = qr @ w[blk + "self_attn.q_up_gate_proj.weight"].float().T
            qup, ogate = qupg.split([qh * hd, qh // qh * o_groups * int(tc["o_lora_rank"])], dim=-1)
            qup = _rope_gptj(_rms_norm(qup.view(t, qh, hd), None, rms_eps), positions, rope_d, comp_theta, yarn_scale=yarn)
            ape = w[blk + "self_attn.compressor.ape"].float()
            n_blks = t // comp_r
            comp_kvs = []
            for b in range(n_blks):
                s0, s1 = b * comp_r, (b + 1) * comp_r
                wts_b = torch.softmax(c_sc[s0:s1] + ape[:comp_r], dim=0)
                c_b = _rms_norm((wts_b * c_kv[s0:s1]).sum(dim=0, keepdim=True), w[blk + "self_attn.comp_norm.weight"], rms_eps)
                comp_kvs.append(_rope_gptj(c_b, torch.tensor([s0]), rope_d, comp_theta, yarn_scale=yarn)[0])
            win = int(tc.get("sliding_window", 128))
            k_all = torch.stack(comp_kvs, dim=0) if comp_kvs else torch.zeros((0, hd))
            k_cat = torch.cat([k_all, kv_r], dim=0).unsqueeze(1)
            pos_k = positions.unsqueeze(0)
            pos_q = positions.unsqueeze(1)
            swa_mask = (pos_k <= pos_q) & (pos_k >= (pos_q - win + 1))
            if n_blks > 0:
                blk_end = torch.tensor([(b + 1) * comp_r for b in range(n_blks)]).unsqueeze(0)
                comp_mask = blk_end <= (pos_q - win + 1).clamp(min=0)
                full_mask = torch.cat([comp_mask, swa_mask], dim=1)
            else:
                full_mask = swa_mask
            att = _sink_attention(qup, k_cat, k_cat, hd ** -0.5, full_mask, w.get(blk + "self_attn.attn_sink"))
            attn_out = _run_hca_common_out(att, qr, ogate, positions, w, blk, nope_d, rope_d, comp_theta, o_groups, yarn)
        else:
            qh = int(tc["hybrid_mla_num_attention_heads"])
            qr_rank, kvr = int(tc["hybrid_mla_q_lora_rank"]), int(tc["hybrid_mla_kv_lora_rank"])
            nope_d, rope_d, vd = int(tc["hybrid_mla_qk_nope_head_dim"]), int(tc["hybrid_mla_qk_rope_head_dim"]), int(tc["hybrid_mla_v_head_dim"])
            idx_h, idx_d = int(tc["index_n_heads"]), int(tc["index_head_dim"])
            qa = _rms_norm(x @ w[blk + "self_attn.q_a_proj.weight"].float().T, w[blk + "self_attn.q_a_layernorm.weight"], rms_eps)
            kva = x @ w[blk + "self_attn.kv_a_proj_with_mqa.weight"].float().T
            kv_c, k_pe = kva.split([kvr, rope_d], dim=-1)
            kv_c = _rms_norm(kv_c, w[blk + "self_attn.kv_a_layernorm.weight"], rms_eps)
            k_pe = _rope_gptj(k_pe, positions, rope_d, theta, yarn_scale=None)
            qb_full = qa @ w[blk + "self_attn.q_b_gate_idx_proj.weight"].float().T
            qn, qr_p, ogate, q_idx = qb_full.split([qh * nope_d, qh * rope_d, qh * vd, idx_h * idx_d], dim=-1)
            qr_p = _rope_gptj(qr_p.view(t, qh, rope_d), positions, rope_d, theta, yarn_scale=None)
            q_full = torch.cat([qn.view(t, qh, nope_d), qr_p], dim=-1)
            kv_b = w[blk + "self_attn.kv_b_proj.weight"].float().view(qh, nope_d + vd, kvr)
            k_nope = torch.einsum("tk,hnk->thn", kv_c, kv_b[:, :nope_d, :])
            v_lat = torch.einsum("tk,hvk->thv", kv_c, kv_b[:, nope_d:, :])
            k_full = torch.cat([k_nope, k_pe.unsqueeze(1).expand(-1, qh, -1)], dim=-1)
            mask = positions.unsqueeze(0) <= positions.unsqueeze(1)
            att = _sink_attention(q_full, k_full, v_lat, (nope_d + rope_d) ** -0.5, mask, w.get(blk + "self_attn.attn_sink"))
            if (vha := w.get(blk + "self_attn.vha_mix.weight")) is not None:
                att = torch.einsum("hg,tgd->thd", vha.float(), att)
            attn_out = (att.reshape(t, qh * vd) * torch.sigmoid(ogate)) @ w[blk + "self_attn.o_proj.weight"].float().T
            if (blk + "self_attn.indexer.linear_gate.weight") in w:
                iw = x @ w[blk + "self_attn.indexer.weights_proj.weight"].float().T
                q_sel, w_sel, fi_heads = flashindexer_select_heads_ref(
                    x, q_idx.view(t, idx_h, idx_d), iw,
                    w[blk + "self_attn.indexer.linear_gate.weight"], w[blk + "self_attn.indexer.flashindexer_bias"],
                    topk_heads=min(8, max(1, idx_h // 2)),
                )
                fi_out[f"fi_q_sel_{l}"], fi_out[f"fi_w_sel_{l}"], fi_out[f"fi_heads_{l}"] = q_sel, w_sel, fi_heads

        streams = _mhc_post(streams, attn_out, post_a, comb_a)
        _, post_f, comb_f, xf = _mhc_pre(streams, w[blk + "mhc_ffn.weight"], w[blk + "mhc_ffn.bias"],
                                         w[blk + "post_attention_layernorm.weight"], rms_eps, hc_eps, sinkhorn)
        if f"{blk}mlp.experts.0.gate_proj.weight" in w:
            ffn_out = _run_moe(xf, w, blk, tc, stacked=False)
        else:
            b, lb = float(tc.get("activation_situ_beta", 4.0)), float(tc.get("activation_situ_linear_beta", 25.0))
            ffn_out = _situ_and_mul(xf @ w[blk + "mlp.gate_proj.weight"].float().T,
                                    xf @ w[blk + "mlp.up_proj.weight"].float().T, beta=b, lin_beta=lb) @ w[blk + "mlp.down_proj.weight"].float().T
        streams = _mhc_post(streams, ffn_out, post_f, comb_f)
        if l in target_ids:
            taps[l] = streams.mean(dim=1)

    ordered = [taps[int(i)] for i in tc["dspark_target_layer_ids"]]
    head_in = _mhc_exit_contract(streams, w["model.mhc_contract.weight"], w["model.mhc_contract.bias"], rms_eps, hc_eps)
    target_logits = _rms_norm(head_in, w["model.norm.weight"], rms_eps) @ w["lm_head.weight"].float().T
    if "lm_head.multimax" in w:
        target_logits = _multimax(target_logits, w["lm_head.multimax"])
    return {"concat_taps": torch.cat(ordered, dim=-1), "target_logits": target_logits, **fi_out}


def dspark_ingest_context(concat_taps: torch.Tensor, positions: torch.Tensor, w: dict[str, torch.Tensor], tc: dict) -> list[torch.Tensor]:
    main_x = _rms_norm(concat_taps.float() @ w["dspark.main_proj.weight"].float().T,
                       w["dspark.main_norm.weight"], float(tc.get("rms_norm_eps", 1e-6)))
    rope_d, theta = int(tc["qk_rope_head_dim"]), float(tc.get("rope_theta", 8000000.0))
    n_ds = int(tc.get("dspark_num_hidden_layers") or len(tc["dspark_target_layer_ids"]))
    ctx_kvs: list[torch.Tensor] = []
    for i in range(n_ds):
        dblk = f"dspark.layers.{i}."
        kv = main_x @ w[dblk + "self_attn.linear_kv_proj.weight"].float().T
        kv = _rms_norm(kv, w[dblk + "self_attn.kv_norm.weight"], float(tc.get("rms_norm_eps", 1e-6)))
        kv = _rope_gptj(kv, positions, rope_d, theta, yarn_scale=None)
        ctx_kvs.append(kv)
    return ctx_kvs


def dspark_draft_block(anchor_tok: int, ctx_kvs: list[torch.Tensor], pos0: int,
                       w: dict[str, torch.Tensor], tc: dict) -> dict[str, torch.Tensor]:
    blk_sz = int(tc.get("dspark_block_size", 7))
    noise_id = int(tc.get("dspark_noise_token_id", 1024))
    rms_eps, hc_eps = float(tc.get("rms_norm_eps", 1e-6)), float(tc.get("hc_eps", 1e-6))
    streams_n, sinkhorn = int(tc.get("hc_mult", 4)), int(tc.get("hc_sinkhorn_iters", 20))
    qh, hd = int(tc["num_attention_heads"]), int(tc["head_dim"])
    o_groups, o_lora = max(1, int(tc.get("o_groups", 1))), int(tc["o_lora_rank"])
    nope_d, rope_d, theta = int(tc["qk_nope_head_dim"]), int(tc["qk_rope_head_dim"]), float(tc.get("rope_theta", 8000000.0))
    win = int(tc.get("sliding_window", 128))
    n_ds = len(ctx_kvs)

    blk_toks = torch.tensor([anchor_tok] + [noise_id] * (blk_sz - 1), dtype=torch.long)
    blk_pos = torch.arange(pos0, pos0 + blk_sz, dtype=torch.long)
    streams = w["model.embed_tokens.weight"].float()[blk_toks].unsqueeze(1).repeat(1, streams_n, 1)

    for i in range(n_ds):
        dblk = f"dspark.layers.{i}."
        _, post_a, comb_a, x = _mhc_pre(streams, w[dblk + "mhc_attn.weight"], w[dblk + "mhc_attn.bias"],
                                        w[dblk + "input_layernorm.weight"], rms_eps, hc_eps, sinkhorn)
        qr = _rms_norm(x @ w[dblk + "self_attn.linear_q_down_proj.weight"].float().T, w[dblk + "self_attn.q_norm.weight"], rms_eps)
        kv_prop = _rope_gptj(_rms_norm(x @ w[dblk + "self_attn.linear_kv_proj.weight"].float().T,
                                       w[dblk + "self_attn.kv_norm.weight"], rms_eps), blk_pos, rope_d, theta, yarn_scale=None)
        qupg = qr @ w[dblk + "self_attn.q_up_gate_proj.weight"].float().T
        qup, ogate = qupg.split([qh * hd, o_groups * o_lora], dim=-1)
        qup = _rope_gptj(_rms_norm(qup.view(blk_sz, qh, hd), None, rms_eps), blk_pos, rope_d, theta, yarn_scale=None)
        ctx_start = max(0, pos0 - win)
        k_cat = torch.cat([ctx_kvs[i][ctx_start:pos0], kv_prop], dim=0).unsqueeze(1)
        mask = torch.ones((blk_sz, k_cat.shape[0]), dtype=torch.bool)
        att = _sink_attention(qup, k_cat, k_cat, hd ** -0.5, mask, w.get(dblk + "self_attn.attn_sink"))
        attn_out = _run_hca_common_out(att, qr, ogate, blk_pos, w, dblk, nope_d, rope_d, theta, o_groups, None)
        streams = _mhc_post(streams, attn_out, post_a, comb_a)
        _, post_f, comb_f, xf = _mhc_pre(streams, w[dblk + "mhc_ffn.weight"], w[dblk + "mhc_ffn.bias"],
                                         w[dblk + "post_attention_layernorm.weight"], rms_eps, hc_eps, sinkhorn)
        streams = _mhc_post(streams, _run_moe(xf, w, dblk, tc, stacked=True), post_f, comb_f)

    # Pre-norm head_hidden feeds confidence_head; post-dspark.norm feeds lm_head (coordinator #14).
    head_hidden = _mhc_exit_contract(streams, w["dspark.mhc_contract.weight"], w["dspark.mhc_contract.bias"], rms_eps, hc_eps)
    norm_hidden = _rms_norm(head_hidden, w["dspark.norm.weight"], rms_eps)
    base_logits = norm_hidden @ w["lm_head.weight"].float().T

    prev_tok = anchor_tok
    draft_tokens: list[int] = []
    final_logits, confidences = [], []
    mw1, mw2 = w["dspark.markov_w1.weight"].float(), w["dspark.markov_w2.weight"].float()
    conf_w = w.get("dspark.confidence_head.weight")
    mm_p = w.get("lm_head.multimax")

    for j in range(blk_sz):
        e_j = mw1[prev_tok]
        logits_j = base_logits[j] + (e_j @ mw2.T)
        if mm_p is not None:
            logits_j = _multimax(logits_j, mm_p)
        tok_j = int(torch.argmax(logits_j).item())
        draft_tokens.append(tok_j)
        final_logits.append(logits_j)
        if conf_w is not None:
            conf_in = torch.cat([head_hidden[j], e_j], dim=-1)
            confidences.append(torch.sigmoid((conf_in.float() @ conf_w.float().T).squeeze(-1)))
        prev_tok = tok_j

    return {
        "head_hidden": head_hidden,
        "base_logits": base_logits,
        "final_logits": torch.stack(final_logits, dim=0),
        "draft_tokens": torch.tensor(draft_tokens, dtype=torch.long),
        "confidences": torch.stack(confidences, dim=0) if confidences else torch.empty(0),
    }


def _pack_mxfp4_weight(t_f32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Round into exact representable MXFP4 E2M1 * 2^(e-127) values so source HF == dequant.
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    *pfx, n, k = t_f32.shape
    blk = t_f32.reshape(*pfx, n, k // 32, 32)
    sc_val = (blk.abs().amax(dim=-1) / 6.0).clamp(min=2.0 ** -30)
    exp = torch.ceil(torch.log2(sc_val)).clamp(-127, 127).to(torch.int64)
    sc_f = (2.0 ** exp.float()).unsqueeze(-1)
    normed = (blk / sc_f).clamp(-6.0, 6.0)
    sign = (normed < 0).to(torch.uint8)
    idx = (normed.abs().unsqueeze(-1) - lut).abs().argmin(dim=-1).to(torch.uint8)
    nib = (sign << 3) | idx
    packed = (nib[..., 0::2] | (nib[..., 1::2] << 4)).reshape(*pfx, n, k // 2).contiguous().view(torch.int8)
    e8m0_dt = getattr(torch, "float8_e8m0fnu", None)
    u8_exp = (exp + 127).to(torch.uint8)
    scales = u8_exp.view(e8m0_dt) if e8m0_dt is not None else u8_exp
    return packed, scales


def build_synthetic_dspark_hf_checkpoint(out_dir: Path, *, seed: int = 42, bfloat16_router: bool = True,
                                         with_flashindexer: bool = True, num_draft_layers: int = 2) -> dict:
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    h, v, inter, num_layers = 128, 256, 192, 2
    qh, hd, qr, og, ol, nope, rope_d, win = 4, 64, 64, 2, 64, 48, 16, 4
    mqh, mqr, mkvr, mnope, mrope, mv, idxh, idxd = 4, 64, 64, 32, 16, 32, 4, 32
    e_cnt, moe_inter, lat, sh_inter, streams, rank = 4, 64, 64, 64, 4, 64
    tc = {
        "model_type": "ernie5_5_moe", "architectures": ["Ernie5_5_MoeForCausalLM"],
        "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
        "hidden_size": h, "vocab_size": v, "intermediate_size": inter, "num_hidden_layers": num_layers,
        "num_attention_heads": qh, "num_key_value_heads": 1, "head_dim": hd, "q_lora_rank": qr, "kv_lora_rank": hd,
        "o_groups": og, "o_lora_rank": ol, "qk_nope_head_dim": nope, "qk_rope_head_dim": rope_d, "v_head_dim": hd,
        "compress_ratios": [4, -2], "layer_types": ["compressed_attention", "full_attention"],
        "sliding_window": win, "max_position_embeddings": 256, "rope_theta": 10000.0, "compress_rope_theta": 10000.0,
        "rope_scaling": {"type": "yarn", "factor": 4, "original_max_position_embeddings": 64, "beta_fast": 32, "beta_slow": 1},
        "rope_interleave": True, "enable_hyper_connections": True, "hc_mult": streams, "hc_eps": 1e-6, "hc_sinkhorn_iters": 20,
        "use_vha_attention": True, "add_full_attention_sink_bias": True, "use_gated_attn": True,
        "first_k_dense_replace": 1, "moe_layer_freq": 1, "n_routed_experts": e_cnt, "num_experts_per_tok": 2,
        "moe_intermediate_size": moe_inter, "moe_latent_size": lat, "n_shared_experts": 1, "shared_expert_intermediate_size": sh_inter,
        "latent_moe_use_norm": True, "moe_split_feature_routing": True, "routed_scaling_factor": 3.0,
        "routed_scaling_factor_learnable": True, "norm_topk_prob": True, "scoring_func": "sigmoid",
        "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0, "rms_norm_eps": 1e-6, "multimax": "lm_head",
        "hybrid_mla_num_attention_heads": mqh, "hybrid_mla_q_lora_rank": mqr, "hybrid_mla_kv_lora_rank": mkvr,
        "hybrid_mla_qk_nope_head_dim": mnope, "hybrid_mla_qk_rope_head_dim": mrope, "hybrid_mla_v_head_dim": mv,
        "index_n_heads": idxh, "index_head_dim": idxd, "index_topk": 4, "tie_word_embeddings": False,
        "num_nextn_predict_layers": 0, "mtp_shared_last_layer": True, "dspark_block_size": 4,
        "dspark_markov_rank": rank, "dspark_noise_token_id": 17, "dspark_target_layer_ids": [0, 1],
        "dspark_num_hidden_layers": num_draft_layers, "torch_dtype": "bfloat16",
    }

    def bf(rows: int, *cols: int, sc: float = 0.08):
        return (torch.randn((rows, *cols), generator=g) * sc).to(torch.bfloat16)

    def ones(n: int):
        return torch.ones(n, dtype=torch.bfloat16) + 0.05 * torch.randn(n, generator=g, dtype=torch.float32).to(torch.bfloat16)

    def add_mhc(s: dict[str, torch.Tensor], pfx: str):
        rc = 2 * streams + streams * streams
        for sub in ("self_attention_hyper_connection", "mlp_hyper_connection"):
            s[f"{pfx}{sub}.mapping_proj.weight"] = torch.randn((streams * h, rc), generator=g) * 0.05
            s[f"{pfx}{sub}.bias"] = torch.randn(rc, generator=g) * 0.05
            for ak in ("alpha_pre", "alpha_post", "alpha_res"):
                s[f"{pfx}{sub}.{ak}"] = torch.tensor([0.5], dtype=torch.float32)

    def add_hca_common(s: dict[str, torch.Tensor], pfx: str, full: bool):
        s[pfx + "self_attn.linear_q_down_proj.weight"] = bf(qr, h)
        s[pfx + "self_attn.linear_kv_proj.weight"] = bf(hd, h)
        s[pfx + "self_attn.q_norm.weight"], s[pfx + "self_attn.kv_norm.weight"] = ones(qr), ones(hd)
        if full:
            s[pfx + "self_attn.core_attention.compressor.linear_wkv.weight"] = bf(hd, h)
            s[pfx + "self_attn.core_attention.compressor.linear_wgate.weight"] = bf(hd, h)
            s[pfx + "self_attn.core_attention.compressor.norm.weight"] = ones(hd)
            s[pfx + "self_attn.core_attention.compressor.ape"] = torch.randn((tc["compress_ratios"][0], hd), generator=g) * 0.05
        s[pfx + "self_attn.linear_q_up_proj.weight"] = bf(qh * hd, qr)
        s[pfx + "self_attn.gate_proj.weight"] = bf(og * ol, qr)
        s[pfx + "self_attn.wo_a.weight"] = bf(og * ol, (qh // og) * hd)
        s[pfx + "self_attn.o_proj.weight"] = bf(h, og * ol)
        s[pfx + "self_attn.vha_postmix_U"] = torch.randn((qh, 4), generator=g) * 0.05
        s[pfx + "self_attn.vha_postmix_V"] = torch.randn((qh, 4), generator=g) * 0.05
        s[pfx + "self_attn.core_attention.softmax_offset"] = torch.randn(qh, generator=g) * 0.1

    def add_moe(s: dict[str, torch.Tensor], pfx: str, *, mxfp4: bool):
        r_dt = torch.bfloat16 if bfloat16_router else torch.float32
        s[pfx + "block_sparse_moe.gate.weight"] = (torch.randn((e_cnt, h), generator=g) * 0.1).to(r_dt)
        s[pfx + "block_sparse_moe.gate.weight_1"] = (torch.randn((e_cnt, h), generator=g) * 0.1).to(r_dt)
        s[pfx + "block_sparse_moe.e_score_correction_bias"] = torch.randn(e_cnt, generator=g) * 0.02
        s[pfx + "block_sparse_moe.gate.routed_scaling_factor_param"] = torch.full((e_cnt,), 1.5, dtype=torch.float32)
        s[pfx + "block_sparse_moe.fc1_latent_proj.weight"] = bf(lat, h)
        s[pfx + "block_sparse_moe.latent_norm.weight"] = ones(lat)
        s[pfx + "block_sparse_moe.fc2_latent_proj.weight"] = bf(h, lat)
        s[pfx + "block_sparse_moe.shared_experts.w1.weight"] = bf(sh_inter, h)
        s[pfx + "block_sparse_moe.shared_experts.w3.weight"] = bf(sh_inter, h)
        s[pfx + "block_sparse_moe.shared_experts.w2.weight"] = bf(h, sh_inter)
        for e in range(e_cnt):
            for k, r, c in (("w1", moe_inter, lat), ("w3", moe_inter, lat), ("w2", lat, moe_inter)):
                w0 = torch.randn((r, c), generator=g) * 0.1
                if mxfp4:
                    p, sc = _pack_mxfp4_weight(w0)
                    s[f"{pfx}block_sparse_moe.experts.{e}.{k}.weight"] = p
                    s[f"{pfx}block_sparse_moe.experts.{e}.{k}.weight_scale_inv"] = sc
                else:
                    s[f"{pfx}block_sparse_moe.experts.{e}.{k}.weight"] = w0.to(torch.bfloat16)

    shard0: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": bf(v, h), "model.norm.weight": ones(h), "lm_head.weight": bf(v, h),
        "model.mhc_contract.hc_head_fn": torch.randn((streams * h, streams), generator=g) * 0.05,
        "model.mhc_contract.hc_head_base": torch.randn(streams, generator=g) * 0.05,
        "model.mhc_contract.hc_head_scale": torch.tensor([0.8], dtype=torch.float32),
        "model.lm_head.multimax_ranges": torch.tensor([-5.0, 5.0, -8.0, 8.0], dtype=torch.float32),
        "model.lm_head.multimax_ts": torch.tensor([0.1, 0.1, 0.02, 0.02], dtype=torch.float32),
    }
    b0 = "model.layers.0."
    shard0[b0 + "input_layernorm.weight"], shard0[b0 + "post_attention_layernorm.weight"] = ones(h), ones(h)
    add_mhc(shard0, b0)
    add_hca_common(shard0, b0, full=True)
    shard0[b0 + "mlp.gate_proj.weight"], shard0[b0 + "mlp.up_proj.weight"] = bf(inter, h), bf(inter, h)
    shard0[b0 + "mlp.down_proj.weight"] = bf(h, inter)

    b1 = "model.layers.1."
    shard0[b1 + "input_layernorm.weight"], shard0[b1 + "post_attention_layernorm.weight"] = ones(h), ones(h)
    add_mhc(shard0, b1)
    shard0[b1 + "self_attn.q_a_proj.weight"] = bf(mqr, h)
    shard0[b1 + "self_attn.kv_a_proj_with_mqa.weight"] = bf(mkvr + mrope, h)
    shard0[b1 + "self_attn.q_a_layernorm.weight"], shard0[b1 + "self_attn.kv_a_layernorm.weight"] = ones(mqr), ones(mkvr)
    shard0[b1 + "self_attn.q_b_proj.weight"] = bf(mqh * (mnope + mrope), mqr)
    shard0[b1 + "self_attn.gate_proj.weight"] = bf(mqh * mv, mqr)
    shard0[b1 + "self_attn.kv_b_proj.weight"] = bf(mqh * (mnope + mv), mkvr)
    shard0[b1 + "self_attn.o_proj.weight"] = bf(h, mqh * mv)
    shard0[b1 + "self_attn.core_attention.indexer.wk.weight"] = bf(idxd, h)
    shard0[b1 + "self_attn.core_attention.indexer.wq_b.weight"] = bf(idxh * idxd, mqr)
    shard0[b1 + "self_attn.core_attention.indexer.k_norm.weight"] = ones(idxd)
    shard0[b1 + "self_attn.core_attention.indexer.k_norm.bias"] = torch.randn(idxd, generator=g).to(torch.bfloat16) * 0.05
    shard0[b1 + "self_attn.core_attention.indexer.weights_proj.weight"] = torch.randn((idxh, h), generator=g) * 0.05
    shard0[b1 + "self_attn.vha_postmix_U"] = torch.randn((mqh, 4), generator=g) * 0.05
    shard0[b1 + "self_attn.vha_postmix_V"] = torch.randn((mqh, 4), generator=g) * 0.05
    shard0[b1 + "self_attn.core_attention.softmax_offset"] = torch.randn(mqh, generator=g) * 0.1
    add_moe(shard0, b1, mxfp4=False)

    shard1: dict[str, torch.Tensor] = {
        "model.mtp.feature_projection.weight": bf(h, len(tc["dspark_target_layer_ids"]) * h),
        "model.mtp.feature_norm.weight": ones(h), "model.mtp.norm.weight": ones(h),
        "model.mtp.mhc_contract.hc_head_fn": torch.randn((streams * h, streams), generator=g) * 0.05,
        "model.mtp.mhc_contract.hc_head_base": torch.randn(streams, generator=g) * 0.05,
        "model.mtp.mhc_contract.hc_head_scale": torch.tensor([0.75], dtype=torch.float32),
        "model.mtp.markov_head.markov_w1.weight": bf(rank, v),
        "model.mtp.markov_head.markov_w2.weight": bf(v, rank),
        "model.mtp.confidence_head.projection.weight": bf(1, h + rank),
    }
    for i in range(num_draft_layers):
        dpfx = f"model.mtp.{i}."
        shard1[dpfx + "input_layernorm.weight"], shard1[dpfx + "post_attention_layernorm.weight"] = ones(h), ones(h)
        add_mhc(shard1, dpfx)
        add_hca_common(shard1, dpfx, full=False)
        add_moe(shard1, dpfx, mxfp4=True)

    f0, f1 = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    save_file(shard0, str(out_dir / f0))
    save_file(shard1, str(out_dir / f1))
    wmap = {**{k: f0 for k in shard0}, **{k: f1 for k in shard1}}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wmap}, indent=2, sort_keys=True))
    (out_dir / "config.json").write_text(json.dumps(tc, indent=2, sort_keys=True))
    (out_dir / "tokenizer.json").write_text('{"version":"1.0","model":{"type":"BPE"}}')
    (out_dir / "tokenizer_config.json").write_text('{"bos_token":"<s>","eos_token":"</s>"}')

    if with_flashindexer:
        fi_dir = out_dir / "flashindexer"
        fi_dir.mkdir(exist_ok=True)
        save_file({
            "model.layers.1.self_attn.indexer.linear_gate.weight": bf(idxh, h),
            "model.layers.1.self_attn.indexer.flashindexer_bias": torch.randn(idxh, generator=g) * 0.1,
        }, str(fi_dir / "flashindexer.safetensors"))
    return tc


def load_hf_canonical_weights(hf_dir: Path, tc: dict) -> dict[str, torch.Tensor]:
    files = sorted(hf_dir.glob("*.safetensors"))
    reader = ShardReader(hf_dir, files, partial_index_ok=True)
    w: dict[str, torch.Tensor] = {}
    try:
        h, v = int(tc["hidden_size"]), int(tc["vocab_size"])
        streams, rank = int(tc.get("hc_mult", 4)), int(tc.get("dspark_markov_rank", 256))
        for k in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
            w[k] = reader.get(k).float()
        w["model.mhc_contract.weight"], w["model.mhc_contract.bias"] = _fold_exit(reader, "model.mhc_contract", "model.mhc_contract.", streams, h)
        w["lm_head.multimax"] = torch.cat([reader.get("model.lm_head.multimax_ranges").float(), reader.get("model.lm_head.multimax_ts").float()])
        for l in range(int(tc["num_hidden_layers"])):
            blk = f"model.layers.{l}."
            w[blk + "input_layernorm.weight"], w[blk + "post_attention_layernorm.weight"] = reader.get(blk + "input_layernorm.weight").float(), reader.get(blk + "post_attention_layernorm.weight").float()
            _read_mhc(reader, blk, blk, streams, h, w)
            if l < len(tc["compress_ratios"]) and int(tc["compress_ratios"][l]) > 0:
                w[blk + "self_attn.fused_in_proj.weight"] = torch.cat([
                    reader.get(blk + "self_attn.linear_q_down_proj.weight").float(), reader.get(blk + "self_attn.linear_kv_proj.weight").float(),
                    reader.get(blk + "self_attn.core_attention.compressor.linear_wkv.weight").float(), reader.get(blk + "self_attn.core_attention.compressor.linear_wgate.weight").float(),
                ], dim=0)
                w[blk + "self_attn.comp_norm.weight"] = reader.get(blk + "self_attn.core_attention.compressor.norm.weight").float()
                w[blk + "self_attn.compressor.ape"] = reader.get(blk + "self_attn.core_attention.compressor.ape").float()
                _read_hca(reader, blk, blk, int(tc["num_attention_heads"]), w)
            else:
                qh, mqr, mnope, mrope = int(tc["hybrid_mla_num_attention_heads"]), int(tc["hybrid_mla_q_lora_rank"]), int(tc["hybrid_mla_qk_nope_head_dim"]), int(tc["hybrid_mla_qk_rope_head_dim"])
                for s in ("q_a_proj.weight", "kv_a_proj_with_mqa.weight", "q_a_layernorm.weight", "kv_a_layernorm.weight", "kv_b_proj.weight", "o_proj.weight"):
                    w[blk + "self_attn." + s] = reader.get(blk + "self_attn." + s).float()
                qb = reader.get(blk + "self_attn.q_b_proj.weight").float().view(qh, mnope + mrope, mqr)
                w[blk + "self_attn.q_b_gate_idx_proj.weight"] = torch.cat([
                    qb[:, :mnope].reshape(qh * mnope, mqr), qb[:, mnope:].reshape(qh * mrope, mqr),
                    reader.get(blk + "self_attn.gate_proj.weight").float(), reader.get(blk + "self_attn.core_attention.indexer.wq_b.weight").float(),
                ], dim=0)
                w[blk + "self_attn.indexer.weights_proj.weight"] = reader.get(blk + "self_attn.core_attention.indexer.weights_proj.weight").float()
                w[blk + "self_attn.vha_mix.weight"] = _fold_vha_ref(reader, blk, qh)
                w[blk + "self_attn.attn_sink"] = reader.get(blk + "self_attn.core_attention.softmax_offset").float()
            if l == 0:
                for k in ("gate_proj", "up_proj", "down_proj"):
                    w[blk + f"mlp.{k}.weight"] = reader.get(blk + f"mlp.{k}.weight").float()
            else:
                _read_moe_w(reader, blk, blk, tc, w, stacked=False)
        if reader.has("model.mtp.feature_projection.weight"):
            w["dspark.main_proj.weight"] = reader.get("model.mtp.feature_projection.weight").float()
            w["dspark.main_norm.weight"], w["dspark.norm.weight"] = reader.get("model.mtp.feature_norm.weight").float(), reader.get("model.mtp.norm.weight").float()
            w["dspark.mhc_contract.weight"], w["dspark.mhc_contract.bias"] = _fold_exit(reader, "model.mtp.mhc_contract", "model.mtp.mhc_contract.", streams, h)
            mw1 = reader.get("model.mtp.markov_head.markov_w1.weight").float()
            w["dspark.markov_w1.weight"] = mw1.t().contiguous() if list(mw1.shape) == [rank, v] else mw1
            w["dspark.markov_w2.weight"] = reader.get("model.mtp.markov_head.markov_w2.weight").float()
            w["dspark.confidence_head.weight"] = reader.get("model.mtp.confidence_head.projection.weight").float()
            for i in range(int(tc.get("dspark_num_hidden_layers") or len(tc["dspark_target_layer_ids"]))):
                sblk, dblk = f"model.mtp.{i}.", f"dspark.layers.{i}."
                w[dblk + "input_layernorm.weight"], w[dblk + "post_attention_layernorm.weight"] = reader.get(sblk + "input_layernorm.weight").float(), reader.get(sblk + "post_attention_layernorm.weight").float()
                _read_mhc(reader, sblk, dblk, streams, h, w)
                w[dblk + "self_attn.linear_q_down_proj.weight"] = reader.get(sblk + "self_attn.linear_q_down_proj.weight").float()
                w[dblk + "self_attn.linear_kv_proj.weight"] = reader.get(sblk + "self_attn.linear_kv_proj.weight").float()
                _read_hca(reader, sblk, dblk, int(tc["num_attention_heads"]), w)
                _read_moe_w(reader, sblk, dblk, tc, w, stacked=True)
    finally:
        reader.close()
    fi_p = hf_dir / "flashindexer" / "flashindexer.safetensors"
    if fi_p.is_file():
        from safetensors.torch import load_file
        for k, t in load_file(str(fi_p)).items():
            w[k] = t.float()
    return w


def _fold_exit(reader: ShardReader, pfx: str, alt: str, streams: int, h: int) -> tuple[torch.Tensor, torch.Tensor]:
    fn = reader.get(pfx + ".hc_head_fn").float()
    sc = float(reader.get(pfx + ".hc_head_scale").float().reshape(-1)[0].item())
    return fn.reshape(streams * h, streams).T.contiguous() * sc, reader.get(pfx + ".hc_head_base").float().reshape(streams)


def _fold_vha_ref(reader: ShardReader, blk: str, qh: int) -> torch.Tensor:
    return torch.eye(qh) + reader.get(blk + "self_attn.vha_postmix_V").float() @ reader.get(blk + "self_attn.vha_postmix_U").float().T


def _read_mhc(reader: ShardReader, sblk: str, dblk: str, streams: int, h: int, w: dict[str, torch.Tensor]) -> None:
    rows = 2 * streams + streams * streams
    for can, sub in (("mhc_attn", "self_attention_hyper_connection"), ("mhc_ffn", "mlp_hyper_connection")):
        mp = reader.get(sblk + sub + ".mapping_proj.weight").float().reshape(streams * h, rows).T.contiguous()
        for idx, ak in enumerate(("alpha_pre", "alpha_post", "alpha_res")):
            sc = float(reader.get(sblk + sub + "." + ak).float().reshape(-1)[0].item())
            mp[slice(idx * streams, (idx + 1) * streams) if idx < 2 else slice(2 * streams, None)] *= sc
        w[dblk + can + ".weight"], w[dblk + can + ".bias"] = mp, reader.get(sblk + sub + ".bias").float().reshape(rows)


def _read_hca(reader: ShardReader, sblk: str, dblk: str, qh: int, w: dict[str, torch.Tensor]) -> None:
    for s in ("q_norm.weight", "kv_norm.weight", "wo_a.weight", "o_proj.weight"):
        w[dblk + "self_attn." + s] = reader.get(sblk + "self_attn." + s).float()
    w[dblk + "self_attn.q_up_gate_proj.weight"] = torch.cat([
        reader.get(sblk + "self_attn.linear_q_up_proj.weight").float(), reader.get(sblk + "self_attn.gate_proj.weight").float(),
    ], dim=0)
    w[dblk + "self_attn.vha_mix.weight"] = _fold_vha_ref(reader, sblk, qh)
    w[dblk + "self_attn.attn_sink"] = reader.get(sblk + "self_attn.core_attention.softmax_offset").float()


def _read_moe_w(reader: ShardReader, sblk: str, dblk: str, tc: dict, w: dict[str, torch.Tensor], *, stacked: bool) -> None:
    w[dblk + "mlp.gate.weight"] = torch.cat([reader.get(sblk + "block_sparse_moe.gate.weight").float(),
                                             reader.get(sblk + "block_sparse_moe.gate.weight_1").float()], dim=0)
    for dk, sk in (("mlp.gate.e_score_correction_bias", "block_sparse_moe.e_score_correction_bias"),
                   ("mlp.routed_scaling_factor_param", "block_sparse_moe.gate.routed_scaling_factor_param"),
                   ("mlp.fc1.weight", "block_sparse_moe.fc1_latent_proj.weight"), ("mlp.latent_norm.weight", "block_sparse_moe.latent_norm.weight"),
                   ("mlp.fc2.weight", "block_sparse_moe.fc2_latent_proj.weight")):
        w[dblk + dk] = reader.get(sblk + sk).float()
    for dk, sk in (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")):
        w[dblk + f"mlp.shared_experts.{dk}.weight"] = reader.get(sblk + f"block_sparse_moe.shared_experts.{sk}.weight").float()
    e_cnt = int(tc["n_routed_experts"])
    if stacked:
        w[dblk + "mlp.experts.gate_up_proj"] = torch.stack([
            torch.cat([reader.get(sblk + f"block_sparse_moe.experts.{e}.w1.weight").float(),
                       reader.get(sblk + f"block_sparse_moe.experts.{e}.w3.weight").float()], dim=0)
            for e in range(e_cnt)], dim=0)
        w[dblk + "mlp.experts.down_proj"] = torch.stack([reader.get(sblk + f"block_sparse_moe.experts.{e}.w2.weight").float() for e in range(e_cnt)], dim=0)
    else:
        for e in range(e_cnt):
            for dk, sk in (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")):
                w[dblk + f"mlp.experts.{e}.{dk}.weight"] = reader.get(sblk + f"block_sparse_moe.experts.{e}.{sk}.weight").float()


def load_qwp_weights(pack_dir: Path) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for qname in ("model.qwp", "dspark.qwp", "flashindexer.qwp"):
        qp = pack_dir / qname
        if not qp.is_file():
            continue
        hdr, _ = read_header(qp)
        with qp.open("rb") as fp:
            for name, e in hdr["tensors"].items():
                shape = [int(d) for d in e["shape"]]
                if "raw_off" in e:
                    kind = e.get("kind") or e.get("dtype") or "f16"
                    bpe = 2 if kind in ("bf16", "f16") else (4 if kind == "f32" else 1)
                    n_bytes = int(e["n_bytes"]) if "n_bytes" in e else math.prod(shape) * bpe
                    fp.seek(int(e["raw_off"]))
                    buf = fp.read(n_bytes)
                    if kind == "bf16":
                        arr = torch.from_numpy(np.frombuffer(buf, dtype=np.uint16).reshape(shape).copy()).view(torch.bfloat16).float()
                    elif kind == "f32":
                        arr = torch.from_numpy(np.frombuffer(buf, dtype=np.float32).reshape(shape).copy())
                    else:
                        arr = torch.from_numpy(np.frombuffer(buf, dtype=np.float16).reshape(shape).copy()).float()
                elif "hq4_off" in e:
                    hq = e.get("hq")
                    bits, grp = (4, 32) if hq is None else (int(hq["bits"]), int(hq["group"]))
                    is_m5, is_shvq = (hq is None), bool(hq and hq.get("vq") == "shvq1")
                    bbytes = any_block_bytes(bits, grp, is_m5=is_m5, is_shvq=is_shvq)
                    k = shape[-1]
                    outer = int(np.prod(shape[:-1]))
                    fp.seek(int(e["hq4_off"]))
                    blocks = np.frombuffer(fp.read(outer * (k // grp) * bbytes), dtype=np.uint8).reshape(outer, k // grp, bbytes).copy()
                    arr = torch.from_numpy(dequant_any_blocks(blocks, bits, grp, is_m5=is_m5, is_shvq=is_shvq).reshape(shape))
                else:
                    n, k = shape
                    fp.seek(int(e["hq8_off"]))
                    qs = np.frombuffer(fp.read(n * k), dtype=np.int8).reshape(n, k).copy()
                    fp.seek(int(e["hq8_s_off"]))
                    sc = np.frombuffer(fp.read(n * (k // 32) * 2), dtype=np.float16).reshape(n, k // 32).copy()
                    arr = torch.from_numpy(dequant_hq8_g32_m5(qs, sc))
                out[name] = arr
    return out


def verify_synthetic_oracle(work_dir: Path | None = None, *, seed: int = 42) -> dict:
    from inkling2_adapter import maybe_force_cpu_quant
    from repack_dense_qwp import repack_dense_qwp
    maybe_force_cpu_quant("cpu")
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        root = Path(tmp)
        hf_dir, pk_hq8, pk_prod, pk_re = root / "hf", root / "pk_hq8", root / "pk_prod", root / "pk_repack"
        tc = build_synthetic_dspark_hf_checkpoint(hf_dir, seed=seed)
        quantize_ernie55_model(hf_dir, pk_hq8, device="cpu", dense_quant="hq8_g32", expert_quant="hq8_g32", expert_down_quant="hq8_g32",
                               dspark_dense_quant="hq8_g32", dspark_expert_quant="hq8_g32", dspark_expert_down_quant="hq8_g32")
        quantize_ernie55_model(hf_dir, pk_prod, device="cpu")
        repack_dense_qwp(pk_prod, pk_re, dense_quant="shvq4_g32", keep_hq8_head=True, bf16_routers=True, device="cpu")
        ctx = torch.tensor([1, 12, 45, 78, 33, 9, 102, 200], dtype=torch.long)

        def _eval(weights: dict[str, torch.Tensor]):
            res = run_target_and_capture(ctx, weights, tc)
            kv = dspark_ingest_context(res["concat_taps"], torch.arange(ctx.shape[0]), weights, tc)
            return res, kv, dspark_draft_block(int(ctx[-1].item()), kv, int(ctx.shape[0]), weights, tc)

        r_hf, kv_hf, d_hf = _eval(load_hf_canonical_weights(hf_dir, tc))
        r_hq8, kv_hq8, d_hq8 = _eval(load_qwp_weights(pk_hq8))
        r_prd, _, d_prd = _eval(load_qwp_weights(pk_prod))
        r_rep, _, d_rep = _eval(load_qwp_weights(pk_re))
        return {
            "draft_tokens_hf": d_hf["draft_tokens"].tolist(), "draft_tokens_hq8": d_hq8["draft_tokens"].tolist(),
            "draft_tokens_prod": d_prd["draft_tokens"].tolist(), "draft_tokens_repack": d_rep["draft_tokens"].tolist(),
            "greedy_match": bool(torch.equal(d_hf["draft_tokens"], d_hq8["draft_tokens"])),
            "fi_head_match": bool(torch.equal(r_hf["fi_heads_1"], r_hq8["fi_heads_1"]) and torch.equal(r_hf["fi_heads_1"], r_prd["fi_heads_1"])),
            "hq8_tap_err": float((r_hf["concat_taps"] - r_hq8["concat_taps"]).abs().max().item()),
            "hq8_kv_err": float(max((a - b).abs().max().item() for a, b in zip(kv_hf, kv_hq8))),
            "hq8_logit_err": float((d_hf["final_logits"] - d_hq8["final_logits"]).abs().max().item()),
            "hq8_conf_err": float((d_hf["confidences"] - d_hq8["confidences"]).abs().max().item()),
            "prod_tap_err": float((r_hf["concat_taps"] - r_prd["concat_taps"]).abs().max().item()),
            "repack_logit_err": float((r_prd["target_logits"] - r_rep["target_logits"]).abs().max().item()),
            "repack_dspark_logit_err": float((d_prd["final_logits"] - d_rep["final_logits"]).abs().max().item()),
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None, help="Optional directory to emit golden HF/QWP report")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.out is not None:
        a.out.mkdir(parents=True, exist_ok=True)
    res = verify_synthetic_oracle(a.out)
    if a.out is not None:
        (a.out / "golden.json").write_text(json.dumps(res, indent=2, sort_keys=True))
    print(json.dumps(res, indent=2, sort_keys=True))
    return 0 if (res["greedy_match"] and res["fi_head_match"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
