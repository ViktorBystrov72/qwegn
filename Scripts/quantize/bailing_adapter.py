"""Bailing-MoE-v3 / Ling-3.2-air QWP1 packer adapter.
Dequantizes blockwise FP8 checkpoints, de-interleaves direct MLA Q projections,
emits full-rank/LoRA KDA layers, batched Group-MoE slabs, NextN MTP, and ViT+merger.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from ernie55_adapter import ShardReader, _hq_meta
from hq_blocks import validate_source_quant_config
from pack import pack_file, sha256, synthesize_base_chat_template
from shvq import any_block_bytes, format_any_quant_spec, parse_any_quant_spec, quant_any_blocks
from verify import audit_pack, read_header

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oracle"))
from emit_pack_identity import MODEL_ID, geo_v2  # noqa: E402


def _is_mla_layer(tc: dict, layer: int, *, force_mla: bool = False) -> bool:
    if force_mla:
        return True
    ltypes = tc.get("layer_types")
    if isinstance(ltypes, list) and layer < len(ltypes):
        return ltypes[layer] in ("full_attention", "deepseek_sparse_attention")
    gs = int(tc.get("layer_group_size") or 0)
    num_layers = int(tc["num_hidden_layers"])
    if gs > 0:
        return (layer + 1) % gs == 0 or layer >= (num_layers // gs) * gs
    return True


def _is_dense_layer(tc: dict, layer: int) -> bool:
    dense_layers = tc.get("mlp_only_layers") or []
    if dense_layers:
        return layer in {int(x) for x in dense_layers}
    return layer < int(tc.get("first_k_dense_replace", 0))


def _attn_prefix(reader: ShardReader, blk: str) -> str:
    if reader.has(blk + "attention.kv_a_proj_with_mqa.weight") or reader.has(blk + "attention.q_proj.weight"):
        return blk + "attention."
    return blk + "self_attn."


def _deinterleave_mla_q(q_raw, q_heads: int, nope_d: int, rope_d: int, in_dim: int):
    return (
        q_raw.view(q_heads, nope_d + rope_d, in_dim)
        .split([nope_d, rope_d], dim=1)
    )


def _deinterleave_q_cat(q_raw, q_heads: int, nope_d: int, rope_d: int, in_dim: int):
    import torch

    q_nope, q_rope = _deinterleave_mla_q(q_raw, q_heads, nope_d, rope_d, in_dim)
    return torch.cat([
        q_nope.reshape(q_heads * nope_d, in_dim),
        q_rope.reshape(q_heads * rope_d, in_dim),
    ], dim=0).contiguous()


def _kda_dims(tc: dict) -> dict[str, int | bool]:
    lin = tc.get("linear_attn_config") or {}
    h = int(tc["hidden_size"])
    qh = int(tc["num_attention_heads"])
    qk_h = int(tc.get("num_kv_heads_for_linear_attn") or lin.get("num_heads") or qh)
    v_h = int(tc.get("linear_num_value_heads") or lin.get("num_heads") or qh)
    hd = int(tc.get("head_dim") or lin.get("head_dim") or (h // qh))
    no_lora = bool(tc.get("no_kda_lora", not tc.get("use_kda_lora", False)))
    return {
        "h": h,
        "qk_h": qk_h,
        "v_h": v_h,
        "hd": hd,
        "qk_ch": qk_h * hd,
        "z_ch": v_h * hd,
        "conv_k": int(tc.get("short_conv_kernel_size") or tc.get("linear_conv_kernel_dim") or 4),
        "no_lora": no_lora,
        "low_rank": int(tc.get("kda_lora_rank") or hd),
    }


def _vision_merger_remaps(reader: ShardReader) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for key in sorted(reader.tensor_to_file):
        if key.startswith("model.visual.") and not key.endswith("_scale_inv"):
            pairs.append((key, key))
    for src, dst in (
        ("linear_proj.0.weight", "model.visual.merger.linear_fc1.weight"),
        ("linear_proj.0.bias", "model.visual.merger.linear_fc1.bias"),
        ("linear_proj.2.weight", "model.visual.merger.linear_fc2.weight"),
        ("linear_proj.2.bias", "model.visual.merger.linear_fc2.bias"),
    ):
        if reader.has(src):
            pairs.append((src, dst))
    return pairs


def _append_layer_shapes(
    exp: dict[str, list[int]],
    reader: ShardReader,
    tc: dict,
    blk: str,
    *,
    is_mla: bool,
    is_moe: bool,
) -> None:
    h = int(tc["hidden_size"])
    inter = int(tc["intermediate_size"])
    q_heads = int(tc["num_attention_heads"])
    q_rank = int(tc.get("q_lora_rank") or 0)
    kv_rank = int(tc["kv_lora_rank"])
    nope_d = int(tc["qk_nope_head_dim"])
    rope_d = int(tc["qk_rope_head_dim"])
    v_d = int(tc["v_head_dim"])
    n_exp = int(tc.get("num_experts") or tc.get("n_routed_experts") or 0)
    moe_inter = int(tc.get("moe_intermediate_size") or inter)
    sh_raw = tc.get("moe_shared_expert_intermediate_size") or tc.get("shared_expert_intermediate_size") or 0
    n_sh = int(tc.get("num_shared_experts") or 1)
    sh_inter = int(sh_raw) * max(1, n_sh) if sh_raw else 0
    ap = _attn_prefix(reader, blk)

    exp[blk + "input_layernorm.weight"] = [h]
    exp[blk + "post_attention_layernorm.weight"] = [h]

    if is_mla:
        if q_rank > 0:
            exp[ap + "q_a_proj.weight"] = [q_rank, h]
            exp[ap + "q_a_layernorm.weight"] = [q_rank]
            if reader.has(ap + "q_b_proj.weight"):
                exp[ap + "q_b_proj.weight"] = [q_heads * (nope_d + rope_d), q_rank]
            else:
                exp[ap + "q_b_nope_proj.weight"] = [q_heads * nope_d, q_rank]
                exp[ap + "q_b_rope_proj.weight"] = [q_heads * rope_d, q_rank]
        else:
            q_name = ap + ("q_proj.weight" if reader.has(ap + "q_proj.weight") else "q_b_proj.weight")
            exp[q_name] = [q_heads * (nope_d + rope_d), h]
        exp[ap + "kv_a_proj_with_mqa.weight"] = [kv_rank + rope_d, h]
        exp[ap + "kv_a_layernorm.weight"] = [kv_rank]
        exp[ap + "kv_b_proj.weight"] = [q_heads * (nope_d + v_d), kv_rank]
        o_name = ap + ("dense.weight" if reader.has(ap + "dense.weight") else "o_proj.weight")
        exp[o_name] = [h, q_heads * v_d]
        if reader.has(ap + "g_proj.weight"):
            exp[ap + "g_proj.weight"] = reader.shapes[ap + "g_proj.weight"]
    else:
        kd = _kda_dims(tc)
        qk_ch, z_ch, v_h, hd = int(kd["qk_ch"]), int(kd["z_ch"]), int(kd["v_h"]), int(kd["hd"])
        exp[ap + "q_proj.weight"] = [qk_ch, h]
        exp[ap + "k_proj.weight"] = [qk_ch, h]
        exp[ap + "v_proj.weight"] = [z_ch, h]
        if kd["no_lora"]:
            exp[ap + "f_proj.weight"] = [z_ch, h]
            exp[ap + "g_proj.weight"] = [z_ch, h]
            exp[ap + "b_proj.weight"] = [v_h, h]
        else:
            lr = int(kd["low_rank"])
            exp[ap + "f_a_proj.weight"] = [lr, h]
            exp[ap + "g_a_proj.weight"] = [lr, h]
            exp[ap + "b_proj.weight"] = [v_h, h]
            exp[ap + "f_b_proj.weight"] = [z_ch, lr]
            exp[ap + "g_b_proj.weight"] = [z_ch, lr]
        if reader.has(ap + "q_conv1d.weight"):
            exp[ap + "q_conv1d.weight"] = reader.shapes[ap + "q_conv1d.weight"]
            exp[ap + "k_conv1d.weight"] = reader.shapes[ap + "k_conv1d.weight"]
            exp[ap + "v_conv1d.weight"] = reader.shapes[ap + "v_conv1d.weight"]
        else:
            exp[ap + "conv1d.weight"] = reader.shapes[ap + "conv1d.weight"]
        exp[ap + "o_norm.weight"] = [hd]
        exp[ap + "A_log"] = [v_h]
        exp[ap + "dt_bias"] = [z_ch]
        o_name = ap + ("o_proj.weight" if reader.has(ap + "o_proj.weight") else "dense.weight")
        exp[o_name] = [h, z_ch]

    if is_moe and n_exp > 0:
        exp[blk + "mlp.gate.weight"] = [n_exp, h]
        exp[blk + "mlp.gate.e_score_correction_bias"] = [n_exp]
        if sh_inter > 0:
            exp[blk + "mlp.shared_experts.gate_proj.weight"] = [sh_inter, h]
            exp[blk + "mlp.shared_experts.up_proj.weight"] = [sh_inter, h]
            exp[blk + "mlp.shared_experts.down_proj.weight"] = [h, sh_inter]
        for e in range(n_exp):
            exp[blk + f"mlp.experts.{e}.gate_proj.weight"] = [moe_inter, h]
            exp[blk + f"mlp.experts.{e}.up_proj.weight"] = [moe_inter, h]
            exp[blk + f"mlp.experts.{e}.down_proj.weight"] = [h, moe_inter]
    else:
        exp[blk + "mlp.gate_proj.weight"] = [inter, h]
        exp[blk + "mlp.up_proj.weight"] = [inter, h]
        exp[blk + "mlp.down_proj.weight"] = [h, inter]


def build_bailing_expected_shapes(config: dict, reader: ShardReader) -> dict[str, list[int]]:
    tc = config["text_config"] if "text_config" in config else config
    h = int(tc["hidden_size"])
    vocab = int(tc["vocab_size"])
    num_layers = int(tc["num_hidden_layers"])
    tie_emb = bool(tc.get("tie_word_embeddings", config.get("tie_word_embeddings", False)))
    exp: dict[str, list[int]] = {
        "model.embed_tokens.weight": [vocab, h],
        "model.norm.weight": [h],
    }
    if not tie_emb:
        exp["lm_head.weight"] = [vocab, h]
    for l in range(num_layers):
        _append_layer_shapes(
            exp, reader, tc, f"model.layers.{l}.",
            is_mla=_is_mla_layer(tc, l), is_moe=not _is_dense_layer(tc, l),
        )
    if int(tc.get("num_nextn_predict_layers") or 0) >= 1:
        mblk = f"model.layers.{num_layers}."
        exp[mblk + "enorm.weight"] = [h]
        exp[mblk + "hnorm.weight"] = [h]
        exp[mblk + "eh_proj.weight"] = [h, 2 * h]
        exp[mblk + "final_layernorm.weight"] = [h]
        _append_layer_shapes(
            exp, reader, tc, mblk,
            is_mla=not bool(tc.get("mtp_use_kda", False)),
            is_moe=int(tc.get("num_experts") or tc.get("n_routed_experts") or 0) > 0,
        )
    for src, dst in _vision_merger_remaps(reader):
        exp[dst] = reader.shapes[src]
    return exp


def _stream_decoder_layer(
    reader: ShardReader,
    tc: dict,
    blk: str,
    *,
    is_mla: bool,
    is_moe: bool,
    dense_spec: tuple[int, int, bool, bool],
    expert_spec: tuple[int, int, bool, bool],
    expert_down_spec: tuple[int, int, bool, bool],
    dense_fmt: str,
    expert_fmt: str,
    expert_down_fmt: str,
    bump,
    expert_batch_size: int,
):
    import torch

    h = int(tc["hidden_size"])
    inter = int(tc["intermediate_size"])
    q_heads = int(tc["num_attention_heads"])
    q_rank = int(tc.get("q_lora_rank") or 0)
    nope_d = int(tc["qk_nope_head_dim"])
    rope_d = int(tc["qk_rope_head_dim"])
    n_exp = int(tc.get("num_experts") or tc.get("n_routed_experts") or 0)
    moe_inter = int(tc.get("moe_intermediate_size") or inter)
    sh_raw = tc.get("moe_shared_expert_intermediate_size") or tc.get("shared_expert_intermediate_size") or 0
    n_sh = int(tc.get("num_shared_experts") or 1)
    sh_inter = int(sh_raw) * max(1, n_sh) if sh_raw else 0
    ap = _attn_prefix(reader, blk)

    bump("bf16", 2)
    yield blk + "input_layernorm.weight", reader.get(blk + "input_layernorm.weight"), "bf16"
    yield blk + "post_attention_layernorm.weight", reader.get(blk + "post_attention_layernorm.weight"), "bf16"

    if is_mla:
        if q_rank > 0:
            bump(dense_fmt)
            bump("bf16")
            yield ap + "q_a_proj.weight", reader.get(ap + "q_a_proj.weight"), dense_spec
            yield ap + "q_a_layernorm.weight", reader.get(ap + "q_a_layernorm.weight"), "bf16"
            if reader.has(ap + "q_b_proj.weight"):
                qb = _deinterleave_q_cat(reader.get(ap + "q_b_proj.weight"), q_heads, nope_d, rope_d, q_rank)
                bump(dense_fmt)
                yield ap + "q_b_proj.weight", qb, dense_spec
                del qb
            else:
                bump(dense_fmt, 2)
                yield ap + "q_b_nope_proj.weight", reader.get(ap + "q_b_nope_proj.weight"), dense_spec
                yield ap + "q_b_rope_proj.weight", reader.get(ap + "q_b_rope_proj.weight"), dense_spec
        else:
            q_name = ap + ("q_proj.weight" if reader.has(ap + "q_proj.weight") else "q_b_proj.weight")
            q_planar = _deinterleave_q_cat(reader.get(q_name), q_heads, nope_d, rope_d, h)
            bump(dense_fmt)
            yield q_name, q_planar, dense_spec
            del q_planar
        bump(dense_fmt, 3)
        bump("bf16")
        yield ap + "kv_a_proj_with_mqa.weight", reader.get(ap + "kv_a_proj_with_mqa.weight"), dense_spec
        yield ap + "kv_a_layernorm.weight", reader.get(ap + "kv_a_layernorm.weight"), "bf16"
        yield ap + "kv_b_proj.weight", reader.get(ap + "kv_b_proj.weight"), dense_spec
        o_name = ap + ("dense.weight" if reader.has(ap + "dense.weight") else "o_proj.weight")
        yield o_name, reader.get(o_name), dense_spec
        if reader.has(ap + "g_proj.weight"):
            bump("bf16")
            yield ap + "g_proj.weight", reader.get(ap + "g_proj.weight"), "bf16"
    else:
        kd = _kda_dims(tc)
        bump(dense_fmt, 4)
        yield ap + "q_proj.weight", reader.get(ap + "q_proj.weight"), dense_spec
        yield ap + "k_proj.weight", reader.get(ap + "k_proj.weight"), dense_spec
        yield ap + "v_proj.weight", reader.get(ap + "v_proj.weight"), dense_spec
        if kd["no_lora"]:
            bump(dense_fmt, 2)
            bump("bf16")
            yield ap + "f_proj.weight", reader.get(ap + "f_proj.weight"), dense_spec
            yield ap + "g_proj.weight", reader.get(ap + "g_proj.weight"), dense_spec
            yield ap + "b_proj.weight", reader.get(ap + "b_proj.weight"), "bf16"
        else:
            bump(dense_fmt, 3)
            bump("bf16", 2)
            yield ap + "f_a_proj.weight", reader.get(ap + "f_a_proj.weight"), dense_spec
            yield ap + "g_a_proj.weight", reader.get(ap + "g_a_proj.weight"), dense_spec
            yield ap + "b_proj.weight", reader.get(ap + "b_proj.weight"), dense_spec
            yield ap + "f_b_proj.weight", reader.get(ap + "f_b_proj.weight"), "bf16"
            yield ap + "g_b_proj.weight", reader.get(ap + "g_b_proj.weight"), "bf16"
        if reader.has(ap + "q_conv1d.weight"):
            bump("bf16", 3)
            yield ap + "q_conv1d.weight", reader.get(ap + "q_conv1d.weight"), "bf16"
            yield ap + "k_conv1d.weight", reader.get(ap + "k_conv1d.weight"), "bf16"
            yield ap + "v_conv1d.weight", reader.get(ap + "v_conv1d.weight"), "bf16"
        else:
            bump("bf16")
            yield ap + "conv1d.weight", reader.get(ap + "conv1d.weight"), "bf16"
        bump("bf16")
        bump("f32", 2)
        yield ap + "o_norm.weight", reader.get(ap + "o_norm.weight"), "bf16"
        yield ap + "A_log", reader.get(ap + "A_log").to(torch.float32), "f32"
        yield ap + "dt_bias", reader.get(ap + "dt_bias").to(torch.float32), "f32"
        o_name = ap + ("o_proj.weight" if reader.has(ap + "o_proj.weight") else "dense.weight")
        yield o_name, reader.get(o_name), dense_spec

    if is_moe and n_exp > 0:
        bump("f32", 2)
        yield blk + "mlp.gate.weight", reader.get(blk + "mlp.gate.weight").to(torch.float32), "f32"
        if reader.has(blk + "mlp.gate.e_score_correction_bias") or reader.has(blk + "mlp.gate.expert_bias"):
            bias_t = reader.get_first(
                blk + "mlp.gate.e_score_correction_bias",
                blk + "mlp.gate.expert_bias",
            ).to(torch.float32)
        else:
            bias_t = torch.zeros(n_exp, dtype=torch.float32)
        yield blk + "mlp.gate.e_score_correction_bias", bias_t, "f32"
        if sh_inter > 0:
            bump(dense_fmt, 3)
            yield blk + "mlp.shared_experts.gate_proj.weight", reader.get(blk + "mlp.shared_experts.gate_proj.weight"), dense_spec
            yield blk + "mlp.shared_experts.up_proj.weight", reader.get(blk + "mlp.shared_experts.up_proj.weight"), dense_spec
            yield blk + "mlp.shared_experts.down_proj.weight", reader.get(blk + "mlp.shared_experts.down_proj.weight"), dense_spec

        eb, eg, em5, eshvq = expert_spec
        edb, edg, edm5, edshvq = expert_down_spec
        step = max(1, expert_batch_size)
        for e0 in range(0, n_exp, step):
            e1 = min(n_exp, e0 + step)
            cnt = e1 - e0
            gu_list, dn_list = [], []
            for e in range(e0, e1):
                ep = blk + f"mlp.experts.{e}."
                gu_list.append(reader.get(ep + "gate_proj.weight"))
                gu_list.append(reader.get(ep + "up_proj.weight"))
                dn_list.append(reader.get(ep + "down_proj.weight"))
            gu_blocks, eff_eg = quant_any_blocks(
                torch.cat(gu_list, dim=0), bits=eb, group=eg, is_m5=em5, is_shvq=eshvq)
            del gu_list
            gu_meta = _hq_meta(eb, eff_eg, em5, eshvq)
            gu_bbytes = any_block_bytes(eb, eff_eg, is_m5=(gu_meta is None), is_shvq=eshvq)
            gu_blocks = gu_blocks.reshape(2 * cnt, moe_inter, h // eff_eg, gu_bbytes)

            dn_blocks, eff_edg = quant_any_blocks(
                torch.cat(dn_list, dim=0), bits=edb, group=edg, is_m5=edm5, is_shvq=edshvq)
            del dn_list
            dn_meta = _hq_meta(edb, eff_edg, edm5, edshvq)
            dn_bbytes = any_block_bytes(edb, eff_edg, is_m5=(dn_meta is None), is_shvq=edshvq)
            dn_blocks = dn_blocks.reshape(cnt, h, moe_inter // eff_edg, dn_bbytes)

            bump(expert_fmt, 2 * cnt)
            bump(expert_down_fmt, cnt)
            for i, e in enumerate(range(e0, e1)):
                ep = blk + f"mlp.experts.{e}."
                yield ep + "gate_proj.weight", gu_blocks[2 * i], ("prepacked_hq4", [moe_inter, h], gu_meta)
                yield ep + "up_proj.weight", gu_blocks[2 * i + 1], ("prepacked_hq4", [moe_inter, h], gu_meta)
                yield ep + "down_proj.weight", dn_blocks[i], ("prepacked_hq4", [h, moe_inter], dn_meta)
            del gu_blocks, dn_blocks
    else:
        bump(dense_fmt, 3)
        yield blk + "mlp.gate_proj.weight", reader.get(blk + "mlp.gate_proj.weight"), dense_spec
        yield blk + "mlp.up_proj.weight", reader.get(blk + "mlp.up_proj.weight"), dense_spec
        yield blk + "mlp.down_proj.weight", reader.get(blk + "mlp.down_proj.weight"), dense_spec


def quantize_bailing_model(
    source: Path,
    destination: Path,
    flat_root: bool = False,
    quant: str = "hq4_g32_m5",
    group_size: int | None = None,
    dense_quant: str | None = None,
    expert_quant: str | None = None,
    expert_down_quant: str | None = None,
    reuse_source_sha_from: Path | None = None,
    expert_batch_size: int = 64,
):
    import torch

    torch.set_num_threads(min(os.cpu_count() or 8, 16))
    source = source.resolve(strict=True)
    destination = destination.expanduser().absolute()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no .safetensors under {source}")
    config = json.loads((source / "config.json").read_text())
    tc = config["text_config"] if "text_config" in config else config
    parent_mt = config.get("model_type")
    mt = tc.get("model_type") or parent_mt or ""

    fp8_block_size = validate_source_quant_config(config)

    if not (source / "tokenizer.json").is_file():
        raise FileNotFoundError(f"required asset missing: {source / 'tokenizer.json'}")
    embedded_tpl = None
    if not (source / "chat_template.jinja").is_file():
        tok_cfg_p = source / "tokenizer_config.json"
        if tok_cfg_p.is_file():
            tok_cfg = json.loads(tok_cfg_p.read_text())
            if isinstance(tok_cfg.get("chat_template"), str):
                embedded_tpl = tok_cfg["chat_template"]
        if not embedded_tpl:
            embedded_tpl = synthesize_base_chat_template(source)

    dense_spec = parse_any_quant_spec(dense_quant or quant, default_group=group_size)
    expert_spec = parse_any_quant_spec(expert_quant or quant, default_group=group_size)
    expert_down_spec = parse_any_quant_spec(expert_down_quant or expert_quant or quant, default_group=group_size)
    dense_fmt = format_any_quant_spec(*dense_spec)
    expert_fmt = format_any_quant_spec(*expert_spec)
    expert_down_fmt = format_any_quant_spec(*expert_down_spec)
    if dense_fmt == expert_fmt == expert_down_fmt:
        quant_recorded = dense_fmt
    elif expert_fmt == expert_down_fmt:
        quant_recorded = f"dense_{dense_fmt}__expert_{expert_fmt}"
    else:
        quant_recorded = f"dense_{dense_fmt}__expert_{expert_fmt}__expdown_{expert_down_fmt}"

    reader = ShardReader(source, files, fp8_block_size=fp8_block_size)
    expected_shapes = build_bailing_expected_shapes(config, reader)

    sha256_source_hex = None
    if reuse_source_sha_from is not None:
        rq = Path(reuse_source_sha_from) / "model.qwp"
        if rq.is_file():
            rp = read_header(rq)[0].get("provenance", {})
            if rp.get("source") == str(source) and rp.get("sha256_source"):
                sha256_source_hex = rp["sha256_source"]
    if sha256_source_hex is None:
        h_src = hashlib.sha256()
        for idx, path in enumerate(files, 1):
            if idx == 1 or idx % 8 == 0 or idx == len(files):
                print(f"hashing shard {idx}/{len(files)} ({path.name})", flush=True)
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(16 << 20), b""):
                    h_src.update(chunk)
        sha256_source_hex = h_src.hexdigest()

    identity = {
        "model_id": MODEL_ID[mt],
        "geometry_digest": hashlib.sha256(geo_v2(tc, parent_mt=parent_mt).encode()).hexdigest(),
    }
    stats: dict[str, int] = {"bf16": 0, "f32": 0}

    def bump(k: str, n: int = 1) -> None:
        stats[k] = stats.get(k, 0) + n

    def stream():
        num_layers = int(tc["num_hidden_layers"])
        tie_emb = bool(tc.get("tie_word_embeddings", config.get("tie_word_embeddings", False)))
        bump("bf16", 2)
        yield "model.embed_tokens.weight", reader.get_first("model.embed_tokens.weight", "model.word_embeddings.weight"), "bf16"
        yield "model.norm.weight", reader.get("model.norm.weight"), "bf16"
        if not tie_emb:
            bump(dense_fmt)
            yield "lm_head.weight", reader.get("lm_head.weight"), dense_spec

        for l in range(num_layers):
            print(f"quantizing layer {l}/{num_layers - 1}", flush=True)
            yield from _stream_decoder_layer(
                reader, tc, f"model.layers.{l}.",
                is_mla=_is_mla_layer(tc, l), is_moe=not _is_dense_layer(tc, l),
                dense_spec=dense_spec, expert_spec=expert_spec, expert_down_spec=expert_down_spec,
                dense_fmt=dense_fmt, expert_fmt=expert_fmt, expert_down_fmt=expert_down_fmt,
                bump=bump, expert_batch_size=expert_batch_size,
            )

        if int(tc.get("num_nextn_predict_layers") or 0) >= 1:
            mblk = f"model.layers.{num_layers}."
            print(f"quantizing mtp layer {num_layers}", flush=True)
            bump("bf16", 3)
            bump(dense_fmt)
            yield mblk + "enorm.weight", reader.get(mblk + "enorm.weight"), "bf16"
            yield mblk + "hnorm.weight", reader.get(mblk + "hnorm.weight"), "bf16"
            yield mblk + "eh_proj.weight", reader.get(mblk + "eh_proj.weight"), dense_spec
            yield mblk + "final_layernorm.weight", reader.get_first(mblk + "final_layernorm.weight", mblk + "shared_head.norm.weight"), "bf16"
            yield from _stream_decoder_layer(
                reader, tc, mblk,
                is_mla=not bool(tc.get("mtp_use_kda", False)),
                is_moe=int(tc.get("num_experts") or tc.get("n_routed_experts") or 0) > 0,
                dense_spec=dense_spec, expert_spec=expert_spec, expert_down_spec=expert_down_spec,
                dense_fmt=dense_fmt, expert_fmt=expert_fmt, expert_down_fmt=expert_down_fmt,
                bump=bump, expert_batch_size=expert_batch_size,
            )

        for src, dst in _vision_merger_remaps(reader):
            bump("bf16")
            yield dst, reader.get(src), "bf16"

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temp:
            staging = Path(temp) / "model"
            assets = staging if flat_root else staging / "assets"
            assets.mkdir(parents=True, exist_ok=True)
            manifest: dict[str, dict] = {}
            for src in sorted(source.iterdir()):
                if (not src.is_file() or src.name.startswith(".")
                        or src.name.endswith((".safetensors", ".index.json"))
                        or src.name == "crc32.txt"):
                    continue
                shutil.copyfile(src, assets / src.name)
                manifest[src.name] = {"sha256": sha256(src), "bytes": src.stat().st_size}
            if embedded_tpl and not (assets / "chat_template.jinja").is_file():
                tpl_dst = assets / "chat_template.jinja"
                tpl_dst.write_text(embedded_tpl, encoding="utf-8")
                manifest["chat_template.jinja"] = {"sha256": sha256(tpl_dst), "bytes": tpl_dst.stat().st_size}
            manifest_path = assets / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
            weights = staging / "model.qwp"
            pack_file(stream(), weights, {
                **identity,
                "source": str(source),
                "sha256_source": sha256_source_hex,
                "source_revision": (source.name if source.parent.name == "snapshots"
                                    and len(source.name) == 40
                                    and all(c in "0123456789abcdef" for c in source.name)
                                    else None),
                "quant_recorded": quant_recorded,
                "hq8_reference": None,
                "assets_dir": "." if flat_root else "assets",
                "assets_manifest_sha256": sha256(manifest_path),
            })
            audit_pack(weights, expected_shapes=expected_shapes)
            (staging / "model.qwp.sha256").write_text(f"{sha256(weights)}  model.qwp\n")
            if destination.exists():
                raise FileExistsError(f"output appeared during quantization: {destination}")
            staging.rename(destination)
    finally:
        reader.close()
    print(f"packed quant={quant_recorded} stats={stats} -> {destination}", flush=True)
