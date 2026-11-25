"""ERNIE-5.5-Flash (ernie5_5_moe) canonical QWP1 packer, QLSA0001, DSpark & FlashIndexer builder.
Preserves native BF16/F32 routers, packs bundled model.mtp.* -> dspark.qwp (--dspark-only
supported on partial shards), and packs flashindexer/flashindexer.safetensors -> flashindexer.qwp.
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, shutil, sys, tempfile
from collections import OrderedDict
from pathlib import Path
import numpy as np

from ernie55_chat_template import is_ernie_chat_source, synthesize_ernie_chat_template
from hq_blocks import VALID_GROUPS, dequant_with_scale_inv, validate_source_quant_config
from pack import pack_file, resolve_assets_dir, sha256, synthesize_base_chat_template
from shvq import any_block_bytes, format_any_quant_spec, parse_any_quant_spec, quant_any_blocks
from verify import audit_pack, read_header

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oracle"))
from emit_pack_identity import MODEL_ID, geo_v2  # noqa: E402

QLSA_MAGIC, QLSA_ALIGN = b"QLSA0001", 64


class ShardReader:
    """Bounded-LRU safetensors slice reader with optional FP8/MXFP4 block dequant."""

    def __init__(self, source: Path, files: list[Path], fp8_block_size: int = 128,
                 max_open: int = 8, partial_index_ok: bool = False):
        from safetensors import safe_open
        self._safe_open, self.source, self.fp8_block_size, self.max_open = safe_open, source, fp8_block_size, max_open
        self._handles: OrderedDict[Path, object] = OrderedDict()
        self.tensor_to_file: dict[str, Path] = {}
        self.shapes: dict[str, list[int]] = {}
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as sf:
                for name in sf.keys():
                    if name in self.tensor_to_file:
                        raise ValueError(f"duplicate source tensor: {name}")
                    self.tensor_to_file[name], self.shapes[name] = path, list(sf.get_slice(name).get_shape())
        if (idx_path := source / "model.safetensors.index.json").is_file():
            wmap = json.loads(idx_path.read_text())["weight_map"]
            actual = {k: p.name for k, p in self.tensor_to_file.items()}
            if partial_index_ok:
                if bad := [k for k, fn in actual.items() if wmap.get(k) != fn]:
                    raise ValueError(f"partial shard mismatch vs index: {bad[:3]}")
            elif actual != wmap:
                raise ValueError("source shards do not match model.safetensors.index.json")

    def has(self, name: str) -> bool:
        return name in self.tensor_to_file

    def first_present(self, *names: str) -> str:
        for n in names:
            if n in self.tensor_to_file:
                return n
        raise KeyError(f"missing required tensor; tried {names}")

    def _handle(self, path: Path):
        if path in self._handles:
            self._handles.move_to_end(path)
            return self._handles[path]
        while len(self._handles) >= self.max_open:
            self._handles.popitem(last=False)
        self._handles[path] = (sf := self._safe_open(path, framework="pt", device="cpu"))
        return sf

    def get(self, name: str):
        t = self._handle(self.tensor_to_file[name]).get_tensor(name)
        if (sinv := f"{name}_scale_inv") in self.tensor_to_file:
            t = dequant_with_scale_inv(t, self._handle(self.tensor_to_file[sinv]).get_tensor(sinv), fp8_block_size=self.fp8_block_size)
        return t

    def get_first(self, *names: str):
        return self.get(self.first_present(*names))

    def close(self) -> None:
        self._handles.clear()


def _hq_meta(bits: int, group: int, is_m5: bool, is_shvq: bool) -> dict | None:
    if is_shvq:
        return {"bits": bits, "group": group, "vq": "shvq1"}
    return None if (bits == 4 and group == 32 and is_m5) else {"bits": bits, "group": group}


def normalize_quant_spec(spec: str) -> str:
    """Normalize shorthand specs (`shvq4g32`, `hq8g32`) into canonical underscore form."""
    s = spec.strip().lower()
    if s.startswith("aq"):
        raise ValueError(f"Unsupported DSpark quant {spec!r}: verify.py requires rotated hq8_g32")
    return re.sub(r"^([a-z]+\d+)(g\d+(?:_m5)?)$", r"\1_\2", s)


def _is_compressed_layer(tc: dict, layer: int) -> bool:
    ltypes, cratios = tc.get("layer_types"), tc.get("compress_ratios")
    if isinstance(ltypes, list) and layer < len(ltypes):
        return ltypes[layer] == "compressed_attention"
    return isinstance(cratios, list) and layer < len(cratios) and int(cratios[layer]) > 0


def _is_moe_layer(tc: dict, layer: int) -> bool:
    n_exp = int(tc.get("n_routed_experts") or tc.get("num_experts") or 0)
    first_k, freq = int(tc.get("first_k_dense_replace", 0)), max(1, int(tc.get("moe_layer_freq", 1)))
    return n_exp > 0 and layer >= first_k and (layer % freq == 0)


def _compute_hca_svd_f16(wkv, comp_norm, nope_dim: int, index_dim: int) -> np.ndarray:
    import torch
    w = wkv.detach().to(torch.float64)[:nope_dim]
    if comp_norm is not None:
        w = w * comp_norm.detach().to(torch.float64)[:nope_dim].unsqueeze(1)
    gram = (1.0 / float(max(1, w.shape[1]))) * (w @ w.T)
    _, eigvecs = torch.linalg.eigh(0.5 * (gram + gram.T))
    arr = eigvecs[:, -index_dim:].flip(dims=[1]).T.contiguous().to(torch.float32).cpu().numpy()
    signs = np.where(arr[np.arange(index_dim), np.argmax(np.abs(arr), axis=1)] < 0.0, -1.0, 1.0).astype(np.float32)
    return np.ascontiguousarray((arr * signs[:, None]).astype(np.float16).view("<u2"))


def write_hca_qlsa(reader: ShardReader, tc: dict, out_path: Path, model_id: str, geometry_digest: str) -> dict | None:
    num_layers = int(tc["num_hidden_layers"])
    hca_layers = [l for l in range(num_layers) if _is_compressed_layer(tc, l)]
    nope_dim = int(tc.get("qk_nope_head_dim", 0))
    if not hca_layers or nope_dim <= 0:
        return None
    index_dim = min(64, nope_dim)
    pos_ratios = [int(r) for r in (tc.get("compress_ratios") or [])[:num_layers] if int(r) > 0]
    pool_size = pos_ratios[0] if pos_ratios else 128
    tensors_meta: dict[str, dict] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(QLSA_MAGIC + b"\0" * 56)
        for l in hca_layers:
            blk = f"model.layers.{l}."
            if reader.has(blk + "self_attn.fused_in_proj.weight"):
                qr, hd = int(tc["q_lora_rank"]), int(tc["head_dim"])
                wkv = reader.get(blk + "self_attn.fused_in_proj.weight")[qr + hd: qr + 2 * hd]
            else:
                wkv = reader.get_first(blk + "self_attn.core_attention.compressor.linear_wkv.weight",
                                       blk + "self_attn.compress_kv.wkv.weight")
            cnorm = reader.get_first(blk + "self_attn.comp_norm.weight",
                                     blk + "self_attn.core_attention.compressor.norm.weight",
                                     blk + "self_attn.compress_kv.norm.weight")
            raw_bytes = _compute_hca_svd_f16(wkv, cnorm, nope_dim=nope_dim, index_dim=index_dim).tobytes()
            for suffix in ("wq_idx", "wk_idx"):
                if (pad := (-f.tell()) % QLSA_ALIGN):
                    f.write(b"\0" * pad)
                tensors_meta[f"layers.{l}.{suffix}"] = {
                    "shape": [index_dim, nope_dim], "dtype": "f16", "raw_off": f.tell(), "n_bytes": len(raw_bytes)}
                f.write(raw_bytes)
        if (pad := (-f.tell()) % QLSA_ALIGN):
            f.write(b"\0" * pad)
        hdr = json.dumps({
            "provenance": {
                "format": "QLSA0001", "model_id": model_id, "geometry_digest": geometry_digest,
                "method": "weight_svd_v1", "index_dim": index_dim, "pool_size": pool_size,
                "cluster_block_size": 1024, "buckets_per_block": 16, "max_bucket_keys": 64,
                "recommended_budget": int(tc.get("index_topk", 2048)),
                "sink_tokens": pool_size, "window_tokens": int(tc.get("sliding_window") or 128),
            },
            "tensors": tensors_meta,
        }, sort_keys=True).encode("utf-8")
        f.write(hdr)
        f.seek(8)
        f.write(len(hdr).to_bytes(8, "little"))
    return {"sha256": sha256(out_path), "bytes": out_path.stat().st_size}


def _fold_mhc_sub(reader: ShardReader, blk: str, canonical: str, hf_sub: str, streams: int, h: int):
    import torch
    rows, width = 2 * streams + streams * streams, streams * h
    if reader.has(blk + f"{canonical}.weight"):
        return (reader.get(blk + f"{canonical}.weight").to(torch.float32).reshape(rows, width).contiguous(),
                reader.get(blk + f"{canonical}.bias").to(torch.float32).reshape(rows).contiguous())
    pfx = blk + hf_sub + "."
    mp = reader.get(pfx + "mapping_proj.weight").to(torch.float32).reshape(width, rows).T.contiguous()
    for i, ak in enumerate(("alpha_pre", "alpha_post", "alpha_res")):
        sc = float(reader.get(pfx + ak).to(torch.float32).reshape(-1)[0].item()) if reader.has(pfx + ak) else 1.0
        mp[(slice(i * streams, (i + 1) * streams) if i < 2 else slice(2 * streams, None))] *= sc
    return mp, reader.get(pfx + "bias").to(torch.float32).reshape(rows).contiguous()


def _fold_exit_head(reader: ShardReader, prefix: str, alt_prefix: str, streams: int, h: int):
    import torch
    width = streams * h
    if reader.has(prefix + ".weight"):
        return (reader.get(prefix + ".weight").to(torch.float32).reshape(streams, width).contiguous(),
                reader.get(prefix + ".bias").to(torch.float32).reshape(streams).contiguous())
    fn = reader.get_first(prefix + ".hc_head_fn", alt_prefix + "hc_head_fn").to(torch.float32)
    sc_name = prefix + ".hc_head_scale" if reader.has(prefix + ".hc_head_scale") else (alt_prefix + "hc_head_scale")
    sc = float(reader.get(sc_name).to(torch.float32).reshape(-1)[0].item()) if reader.has(sc_name) else 1.0
    bs = reader.get_first(prefix + ".hc_head_base", alt_prefix + "hc_head_base").to(torch.float32).reshape(streams)
    return (fn.reshape(width, streams).T.contiguous() * sc), bs.contiguous()


def _fold_vha(reader: ShardReader, blk: str, q_heads: int):
    import torch
    if reader.has(blk + "self_attn.vha_mix.weight"):
        return reader.get(blk + "self_attn.vha_mix.weight").to(torch.float32).reshape(q_heads, q_heads).contiguous()
    u = reader.get(blk + "self_attn.vha_postmix_U").to(torch.float32)
    v = reader.get(blk + "self_attn.vha_postmix_V").to(torch.float32)
    return (torch.eye(q_heads, dtype=torch.float32) + (v @ u.T)).contiguous()


def _force_cpu_if_requested(device: str) -> None:
    if device == "cpu":
        from inkling2_adapter import maybe_force_cpu_quant
        maybe_force_cpu_quant("cpu")


def build_expected_shapes(tc: dict) -> dict[str, list[int]]:
    h, v, num_layers, inter = int(tc["hidden_size"]), int(tc["vocab_size"]), int(tc["num_hidden_layers"]), int(tc["intermediate_size"])
    n_exp = int(tc.get("n_routed_experts") or tc.get("num_experts") or 0)
    moe_inter, lat_size = int(tc.get("moe_intermediate_size", inter)), int(tc.get("moe_latent_size") or 0)
    routed_dim = lat_size if lat_size > 0 else h
    sh_inter = int(tc.get("shared_expert_intermediate_size") or (int(tc.get("n_shared_experts") or 0) * moe_inter))
    streams, mhc_rows = int(tc.get("hc_mult", 4)), 2 * int(tc.get("hc_mult", 4)) + int(tc.get("hc_mult", 4)) ** 2
    has_mhc, use_vha = bool(tc.get("enable_hyper_connections", False)), bool(tc.get("use_vha_attention", False))
    has_sink, has_ogate = bool(tc.get("add_full_attention_sink_bias", False)), bool(tc.get("use_gated_attn", False))
    split_router, learn_scale = bool(tc.get("moe_split_feature_routing", False)), bool(tc.get("routed_scaling_factor_learnable", False))
    exp: dict[str, list[int]] = {"model.embed_tokens.weight": [v, h], "model.norm.weight": [h]}
    if not bool(tc.get("tie_word_embeddings", False)):
        exp["lm_head.weight"] = [v, h]
    if has_mhc:
        exp["model.mhc_contract.weight"], exp["model.mhc_contract.bias"] = [streams, streams * h], [streams]
    if tc.get("multimax") == "lm_head":
        exp["lm_head.multimax"] = [8]
    for l in range(num_layers):
        blk = f"model.layers.{l}."
        exp[blk + "input_layernorm.weight"], exp[blk + "post_attention_layernorm.weight"] = [h], [h]
        if has_mhc:
            for k in ("mhc_attn", "mhc_ffn"):
                exp[blk + f"{k}.weight"], exp[blk + f"{k}.bias"] = [mhc_rows, streams * h], [mhc_rows]
        if _is_compressed_layer(tc, l):
            q_heads, head_dim, q_rank = int(tc["num_attention_heads"]), int(tc["head_dim"]), int(tc["q_lora_rank"])
            o_groups, o_lora = max(1, int(tc.get("o_groups", 1))), int(tc["o_lora_rank"])
            for k, sh in (("fused_in_proj.weight", [q_rank + 3 * head_dim, h]), ("q_norm.weight", [q_rank]),
                          ("kv_norm.weight", [head_dim]), ("comp_norm.weight", [head_dim]),
                          ("compressor.ape", [int(tc["compress_ratios"][l]), head_dim]),
                          ("q_up_gate_proj.weight", [q_heads * head_dim + o_groups * o_lora, q_rank]),
                          ("wo_a.weight", [o_groups * o_lora, (q_heads // o_groups) * head_dim]),
                          ("o_proj.weight", [h, o_groups * o_lora])):
                exp[blk + "self_attn." + k] = sh
        else:
            q_heads = int(tc.get("hybrid_mla_num_attention_heads", tc["num_attention_heads"]))
            q_rank, kv_rank = int(tc.get("hybrid_mla_q_lora_rank", tc["q_lora_rank"])), int(tc.get("hybrid_mla_kv_lora_rank", tc["kv_lora_rank"]))
            nope_d, rope_d = int(tc.get("hybrid_mla_qk_nope_head_dim", tc["qk_nope_head_dim"])), int(tc.get("hybrid_mla_qk_rope_head_dim", tc["qk_rope_head_dim"]))
            v_d, idx_h, idx_d = int(tc.get("hybrid_mla_v_head_dim", tc["v_head_dim"])), int(tc.get("index_n_heads", q_heads)), int(tc.get("index_head_dim", 128))
            q_ch = q_heads * (nope_d + rope_d + (v_d if has_ogate else 0))
            for k, sh in (("q_a_proj.weight", [q_rank, h]), ("kv_a_proj_with_mqa.weight", [kv_rank + rope_d, h]),
                          ("indexer.wk.weight", [idx_d, h]), ("q_a_layernorm.weight", [q_rank]),
                          ("kv_a_layernorm.weight", [kv_rank]), ("indexer.k_norm.weight", [idx_d]),
                          ("indexer.k_norm.bias", [idx_d]), ("indexer.weights_proj.weight", [idx_h, h]),
                          ("q_b_gate_idx_proj.weight", [q_ch + idx_h * idx_d, q_rank]),
                          ("kv_b_proj.weight", [q_heads * (nope_d + v_d), kv_rank]), ("o_proj.weight", [h, q_heads * v_d])):
                exp[blk + "self_attn." + k] = sh
        if use_vha:
            exp[blk + "self_attn.vha_mix.weight"] = [q_heads, q_heads]
        if has_sink:
            exp[blk + "self_attn.attn_sink"] = [q_heads]
        if _is_moe_layer(tc, l):
            exp[blk + "mlp.gate.weight"] = [(2 * n_exp) if split_router else n_exp, h]
            exp[blk + "mlp.gate.e_score_correction_bias"] = [n_exp]
            if learn_scale:
                exp[blk + "mlp.routed_scaling_factor_param"] = [n_exp]
            if lat_size > 0:
                exp[blk + "mlp.fc1.weight"], exp[blk + "mlp.latent_norm.weight"], exp[blk + "mlp.fc2.weight"] = [lat_size, h], [lat_size], [h, lat_size]
            if sh_inter > 0:
                for k, sh in (("gate_proj", [sh_inter, h]), ("up_proj", [sh_inter, h]), ("down_proj", [h, sh_inter])):
                    exp[blk + f"mlp.shared_experts.{k}.weight"] = sh
            for e in range(n_exp):
                exp[blk + f"mlp.experts.{e}.gate_proj.weight"] = [moe_inter, routed_dim]
                exp[blk + f"mlp.experts.{e}.up_proj.weight"] = [moe_inter, routed_dim]
                exp[blk + f"mlp.experts.{e}.down_proj.weight"] = [routed_dim, moe_inter]
        else:
            exp[blk + "mlp.gate_proj.weight"], exp[blk + "mlp.up_proj.weight"], exp[blk + "mlp.down_proj.weight"] = [inter, h], [inter, h], [h, inter]
    if int(tc.get("num_nextn_predict_layers") or 0) >= 1 and bool(tc.get("mtp_shared_last_layer", False)):
        eblk = f"model.layers.{num_layers}."
        for k, sh in (("enorm.weight", [h]), ("hnorm.weight", [h]), ("e_proj.weight", [h, h]), ("h_proj.weight", [h, h]),
                      ("mhc_contract.weight", [streams, streams * h]), ("mhc_contract.bias", [streams]), ("shared_head.norm.weight", [h])):
            exp[eblk + k] = sh
        if n_exp > 0:
            exp[eblk + "mlp.gate.e_score_correction_bias"] = [n_exp]
    return exp


def _make_emitter(stats: dict[str, int], expected: dict[str, list[int]] | None = None):
    def emit(name: str, arr, kind, *, shape: list[int] | None = None, fmt: str | None = None):
        if expected is not None:
            expected[name] = list(shape if shape is not None else arr.shape)
        key = fmt or ("bf16" if kind is True else (kind[0] if isinstance(kind, tuple) else str(kind)))
        stats[key] = stats.get(key, 0) + 1
        return name, arr, kind
    return emit


def _emit_moe(reader: ShardReader, tc: dict, src_blk: str, dst_blk: str,
              dense_spec, dense_fmt: str, expert_spec, expert_fmt: str,
              expert_down_spec, expert_down_fmt: str, emit, *,
              expert_batch_size: int = 64, stacked_experts: bool = False):
    import torch
    n_exp = int(tc.get("n_routed_experts") or tc.get("num_experts") or 0)
    moe_inter = int(tc.get("moe_intermediate_size", tc["intermediate_size"]))
    lat_size = int(tc.get("moe_latent_size") or 0)
    routed_dim = lat_size if lat_size > 0 else int(tc["hidden_size"])
    sh_inter = int(tc.get("shared_expert_intermediate_size") or (int(tc.get("n_shared_experts") or 0) * moe_inter))

    if reader.has(src_blk + "mlp.gate.weight"):
        r_w = reader.get(src_blk + "mlp.gate.weight")
    elif bool(tc.get("moe_split_feature_routing", False)):
        g0, g1 = reader.get(src_blk + "block_sparse_moe.gate.weight"), reader.get(src_blk + "block_sparse_moe.gate.weight_1")
        r_w = torch.cat([g0, g1], dim=0) if (g0.dtype == torch.bfloat16 and g1.dtype == torch.bfloat16) else torch.cat([g0.to(torch.float32), g1.to(torch.float32)], dim=0)
    else:
        r_w = reader.get(src_blk + "block_sparse_moe.gate.weight")
    yield emit(dst_blk + "mlp.gate.weight", r_w, "bf16" if r_w.dtype == torch.bfloat16 else "f32")
    yield emit(dst_blk + "mlp.gate.e_score_correction_bias", reader.get_first(src_blk + "mlp.gate.e_score_correction_bias", src_blk + "block_sparse_moe.e_score_correction_bias"), "f32")
    if bool(tc.get("routed_scaling_factor_learnable", False)):
        yield emit(dst_blk + "mlp.routed_scaling_factor_param", reader.get_first(src_blk + "mlp.routed_scaling_factor_param", src_blk + "block_sparse_moe.gate.routed_scaling_factor_param"), "f32")
    if lat_size > 0:
        yield emit(dst_blk + "mlp.fc1.weight", reader.get_first(src_blk + "mlp.fc1.weight", src_blk + "block_sparse_moe.fc1_latent_proj.weight"), dense_spec, fmt=dense_fmt)
        yield emit(dst_blk + "mlp.latent_norm.weight", reader.get_first(src_blk + "mlp.latent_norm.weight", src_blk + "block_sparse_moe.latent_norm.weight"), "bf16")
        yield emit(dst_blk + "mlp.fc2.weight", reader.get_first(src_blk + "mlp.fc2.weight", src_blk + "block_sparse_moe.fc2_latent_proj.weight"), dense_spec, fmt=dense_fmt)
    if sh_inter > 0:
        for dst_k, bs_k in (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")):
            yield emit(dst_blk + f"mlp.shared_experts.{dst_k}.weight", reader.get_first(src_blk + f"mlp.shared_experts.{dst_k}.weight", src_blk + f"block_sparse_moe.shared_experts.{bs_k}.weight"), dense_spec, fmt=dense_fmt)

    use_bs = not reader.has(src_blk + "mlp.experts.0.gate_proj.weight")
    w1k, w3k, w2k = ("w1.weight", "w3.weight", "w2.weight") if use_bs else ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
    epfx = lambda e: src_blk + (f"block_sparse_moe.experts.{e}." if use_bs else f"mlp.experts.{e}.")
    if stacked_experts:
        if reader.has(src_blk + "mlp.experts.gate_up_proj"):
            yield emit(dst_blk + "mlp.experts.gate_up_proj", reader.get(src_blk + "mlp.experts.gate_up_proj"), expert_spec, fmt=expert_fmt)
            yield emit(dst_blk + "mlp.experts.down_proj", reader.get(src_blk + "mlp.experts.down_proj"), expert_down_spec, fmt=expert_down_fmt)
            return
        w1_0 = reader.get(epfx(0) + w1k)
        gu_stack = torch.empty((n_exp, 2 * moe_inter, routed_dim), dtype=w1_0.dtype)
        gu_stack[0, :moe_inter], gu_stack[0, moe_inter:] = w1_0, reader.get(epfx(0) + w3k)
        del w1_0
        for e in range(1, n_exp):
            gu_stack[e, :moe_inter], gu_stack[e, moe_inter:] = reader.get(epfx(e) + w1k), reader.get(epfx(e) + w3k)
        yield emit(dst_blk + "mlp.experts.gate_up_proj", gu_stack, expert_spec, fmt=expert_fmt)
        del gu_stack
        w2_0 = reader.get(epfx(0) + w2k)
        dn_stack = torch.empty((n_exp, routed_dim, moe_inter), dtype=w2_0.dtype)
        dn_stack[0] = w2_0
        del w2_0
        for e in range(1, n_exp):
            dn_stack[e] = reader.get(epfx(e) + w2k)
        yield emit(dst_blk + "mlp.experts.down_proj", dn_stack, expert_down_spec, fmt=expert_down_fmt)
        del dn_stack
        return

    (eb, eg, em5, eshvq), (edb, edg, edm5, edshvq), step = expert_spec, expert_down_spec, max(1, expert_batch_size)
    for e0 in range(0, n_exp, step):
        e1, cnt = min(n_exp, e0 + step), min(n_exp, e0 + step) - e0
        gu_blocks, eff_eg = quant_any_blocks(torch.cat([t for e in range(e0, e1) for t in (reader.get(epfx(e) + w1k), reader.get(epfx(e) + w3k))], dim=0),
                                             bits=eb, group=eg, is_m5=em5, is_shvq=eshvq)
        gu_meta = _hq_meta(eb, eff_eg, em5, eshvq)
        gu_blocks = gu_blocks.reshape(2 * cnt, moe_inter, routed_dim // eff_eg, any_block_bytes(eb, eff_eg, is_m5=(gu_meta is None), is_shvq=eshvq))
        dn_blocks, eff_edg = quant_any_blocks(torch.cat([reader.get(epfx(e) + w2k) for e in range(e0, e1)], dim=0),
                                              bits=edb, group=edg, is_m5=edm5, is_shvq=edshvq)
        dn_meta = _hq_meta(edb, eff_edg, edm5, edshvq)
        dn_blocks = dn_blocks.reshape(cnt, routed_dim, moe_inter // eff_edg, any_block_bytes(edb, eff_edg, is_m5=(dn_meta is None), is_shvq=edshvq))
        for i, e in enumerate(range(e0, e1)):
            ep = dst_blk + f"mlp.experts.{e}."
            yield emit(ep + "gate_proj.weight", gu_blocks[2 * i], ("prepacked_hq4", [moe_inter, routed_dim], gu_meta), shape=[moe_inter, routed_dim], fmt=expert_fmt)
            yield emit(ep + "up_proj.weight", gu_blocks[2 * i + 1], ("prepacked_hq4", [moe_inter, routed_dim], gu_meta), shape=[moe_inter, routed_dim], fmt=expert_fmt)
            yield emit(ep + "down_proj.weight", dn_blocks[i], ("prepacked_hq4", [routed_dim, moe_inter], dn_meta), shape=[routed_dim, moe_inter], fmt=expert_down_fmt)
        del gu_blocks, dn_blocks


def _emit_hca_attn_common(reader: ShardReader, tc: dict, src_blk: str, dst_blk: str,
                          dense_spec, dense_fmt: str, emit):
    import torch
    q_heads = int(tc["num_attention_heads"])
    yield emit(dst_blk + "self_attn.q_norm.weight", reader.get_first(
        src_blk + "self_attn.q_norm.weight", src_blk + "self_attn.q_layernorm.weight"), "bf16")
    yield emit(dst_blk + "self_attn.kv_norm.weight", reader.get_first(
        src_blk + "self_attn.kv_norm.weight", src_blk + "self_attn.kv_layernorm.weight"), "bf16")
    qup_gate = (reader.get(src_blk + "self_attn.q_up_gate_proj.weight")
                if reader.has(src_blk + "self_attn.q_up_gate_proj.weight")
                else torch.cat([reader.get(src_blk + "self_attn.linear_q_up_proj.weight"),
                                reader.get(src_blk + "self_attn.gate_proj.weight")], dim=0))
    yield emit(dst_blk + "self_attn.q_up_gate_proj.weight", qup_gate, dense_spec, fmt=dense_fmt)
    del qup_gate
    yield emit(dst_blk + "self_attn.wo_a.weight", reader.get_first(
        src_blk + "self_attn.wo_a.weight", src_blk + "self_attn.linear_o_group_proj.weight",
        src_blk + "self_attn.linear_o_group_proj"), dense_spec, fmt=dense_fmt)
    yield emit(dst_blk + "self_attn.o_proj.weight", reader.get(src_blk + "self_attn.o_proj.weight"), dense_spec, fmt=dense_fmt)
    if bool(tc.get("use_vha_attention", False)):
        yield emit(dst_blk + "self_attn.vha_mix.weight", _fold_vha(reader, src_blk, q_heads), "f32")
    if bool(tc.get("add_full_attention_sink_bias", False)):
        yield emit(dst_blk + "self_attn.attn_sink", reader.get_first(
            src_blk + "self_attn.attn_sink", src_blk + "self_attn.core_attention.attn_sink",
            src_blk + "self_attn.core_attention.softmax_offset"), "f32")


def _emit_mhc_layer(reader: ShardReader, sblk: str, dblk: str, streams: int, h: int, emit):
    aw, ab = _fold_mhc_sub(reader, sblk, "mhc_attn", "self_attention_hyper_connection", streams, h)
    fw, fb = _fold_mhc_sub(reader, sblk, "mhc_ffn", "mlp_hyper_connection", streams, h)
    yield emit(dblk + "mhc_attn.weight", aw, "f32")
    yield emit(dblk + "mhc_attn.bias", ab, "f32")
    yield emit(dblk + "mhc_ffn.weight", fw, "f32")
    yield emit(dblk + "mhc_ffn.bias", fb, "f32")


def _dspark_layer_indices(reader: ShardReader, tc: dict) -> list[int]:
    if tc.get("dspark_num_hidden_layers") is not None:
        return list(range(int(tc["dspark_num_hidden_layers"])))
    idxs: list[int] = []
    i = 0
    while reader.has(f"model.mtp.{i}.input_layernorm.weight"):
        idxs.append(i)
        i += 1
    if not idxs:
        raise ValueError("missing model.mtp.{i}.input_layernorm.weight tensors for DSpark")
    return idxs


def _has_bundled_dspark(reader: ShardReader) -> bool:
    return reader.has("model.mtp.feature_projection.weight") and reader.has("model.mtp.0.self_attn.linear_kv_proj.weight")


def pack_dspark_qwp(reader: ShardReader, tc: dict, out_qwp: Path, *, source: Path, sha256_source_hex: str,
                    identity: dict, assets_dir_val: str, assets_manifest_sha256: str,
                    dspark_dense_quant: str = "shvq4_g32", dspark_expert_quant: str = "shvq15_g64",
                    dspark_expert_down_quant: str = "shvq15_g64", dspark_markov_w2_quant: str = "hq8_g32") -> dict:
    """Pack `model.mtp.*` DSpark draft model into QWP1 `dspark.qwp` + `.sha256`."""
    d_spec, e_spec = parse_any_quant_spec(normalize_quant_spec(dspark_dense_quant)), parse_any_quant_spec(normalize_quant_spec(dspark_expert_quant))
    ed_spec, mw2_spec = parse_any_quant_spec(normalize_quant_spec(dspark_expert_down_quant)), parse_any_quant_spec(normalize_quant_spec(dspark_markov_w2_quant))
    d_fmt, e_fmt, ed_fmt, mw2_fmt = (
        format_any_quant_spec(*d_spec), format_any_quant_spec(*e_spec), format_any_quant_spec(*ed_spec), format_any_quant_spec(*mw2_spec))
    layer_ids = _dspark_layer_indices(reader, tc)
    expected_shapes: dict[str, list[int]] = {}
    stats: dict[str, int] = {"bf16": 0, "f32": 0}
    emit = _make_emitter(stats, expected_shapes)

    def stream():
        h, v, rank, streams = int(tc["hidden_size"]), int(tc["vocab_size"]), int(tc.get("dspark_markov_rank", 256)), int(tc.get("hc_mult", 4))
        has_mhc = bool(tc.get("enable_hyper_connections", False))
        yield emit("dspark.main_proj.weight", reader.get("model.mtp.feature_projection.weight"), d_spec, fmt=d_fmt)
        yield emit("dspark.main_norm.weight", reader.get("model.mtp.feature_norm.weight"), "bf16")
        yield emit("dspark.norm.weight", reader.get("model.mtp.norm.weight"), "bf16")
        if has_mhc:
            mw, mb = _fold_exit_head(reader, "model.mtp.mhc_contract", "model.mtp.mhc_contract.", streams, h)
            yield emit("dspark.mhc_contract.weight", mw, "f32")
            yield emit("dspark.mhc_contract.bias", mb, "f32")
        mw1 = reader.get("model.mtp.markov_head.markov_w1.weight")
        yield emit("dspark.markov_w1.weight", mw1.t().contiguous() if list(mw1.shape) == [rank, v] else mw1, "bf16")
        yield emit("dspark.markov_w2.weight", reader.get("model.mtp.markov_head.markov_w2.weight"), mw2_spec, fmt=mw2_fmt)
        if reader.has("model.mtp.confidence_head.projection.weight") or reader.has("model.mtp.confidence_head.proj.weight"):
            yield emit("dspark.confidence_head.weight", reader.get_first(
                "model.mtp.confidence_head.projection.weight", "model.mtp.confidence_head.proj.weight"), "bf16")
        for i in layer_ids:
            print(f"packing dspark layer {i}/{len(layer_ids) - 1}", flush=True)
            sblk, dblk = f"model.mtp.{i}.", f"dspark.layers.{i}."
            yield emit(dblk + "input_layernorm.weight", reader.get(sblk + "input_layernorm.weight"), "bf16")
            yield emit(dblk + "post_attention_layernorm.weight", reader.get(sblk + "post_attention_layernorm.weight"), "bf16")
            if has_mhc:
                yield from _emit_mhc_layer(reader, sblk, dblk, streams, h, emit)
            yield emit(dblk + "self_attn.linear_q_down_proj.weight", reader.get(sblk + "self_attn.linear_q_down_proj.weight"), d_spec, fmt=d_fmt)
            yield emit(dblk + "self_attn.linear_kv_proj.weight", reader.get(sblk + "self_attn.linear_kv_proj.weight"), d_spec, fmt=d_fmt)
            yield from _emit_hca_attn_common(reader, tc, sblk, dblk, d_spec, d_fmt, emit)
            yield from _emit_moe(reader, tc, sblk, dblk, d_spec, d_fmt, e_spec, e_fmt, ed_spec, ed_fmt, emit, stacked_experts=True)

    dspark_geo = f"qwg-ernie55-dspark-v1:{tc['hidden_size']}|{len(layer_ids)}|{tc.get('dspark_block_size', 7)}|{tc.get('dspark_markov_rank', 256)}"
    pack_file(stream(), out_qwp, {
        **identity, "source": str(source), "sha256_source": sha256_source_hex,
        "dspark_geometry_hash": hashlib.sha256(dspark_geo.encode()).hexdigest(),
        "quant_recorded": f"dspark_dense_{d_fmt}__expert_{e_fmt}__expdown_{ed_fmt}__markov_{mw2_fmt}",
        "dspark_dense_quant": d_fmt, "dspark_expert_quant": e_fmt, "dspark_expert_down_quant": ed_fmt,
        "dspark_markov_w2_quant": mw2_fmt, "assets_dir": assets_dir_val, "assets_manifest_sha256": assets_manifest_sha256,
    })
    audit_pack(out_qwp, expected_shapes=expected_shapes)
    out_qwp.with_name(out_qwp.name + ".sha256").write_text(f"{sha256(out_qwp)}  {out_qwp.name}\n")
    return stats


def _canonical_flashindexer_name(raw_key: str) -> str:
    m = re.match(r"^model\.layers\.(\d+)\.(?:self_attn\.core_attention|self_attn|attn)\.indexer\.(linear_gate\.weight|flashindexer_bias)$", raw_key)
    if not m:
        raise ValueError(f"unexpected FlashIndexer tensor name: {raw_key!r}")
    return f"model.layers.{m.group(1)}.self_attn.indexer.{m.group(2)}"


def pack_flashindexer_qwp(source: Path, out_qwp: Path, *, identity: dict, assets_dir_val: str, assets_manifest_sha256: str) -> dict | None:
    """Pack optional `flashindexer/flashindexer.safetensors` into sibling QWP1 `flashindexer.qwp`."""
    fi_src = source / "flashindexer" / "flashindexer.safetensors"
    if not fi_src.is_file():
        fi_src = source / "flashindexer.safetensors"
    if not fi_src.is_file():
        return None
    import torch
    from safetensors.torch import load_file
    if not (raw := load_file(str(fi_src))):
        raise ValueError(f"empty FlashIndexer safetensors: {fi_src}")
    expected_shapes: dict[str, list[int]] = {}
    stats: dict[str, int] = {"bf16": 0, "f32": 0}
    emit = _make_emitter(stats, expected_shapes)
    items = [emit(_canonical_flashindexer_name(k), raw[k], "f32" if raw[k].dtype == torch.float32 else "bf16") for k in sorted(raw)]
    pack_file(items, out_qwp, {**identity, "source": str(fi_src), "sha256_source": sha256(fi_src),
                                "quant_recorded": "flashindexer_bf16", "assets_dir": assets_dir_val, "assets_manifest_sha256": assets_manifest_sha256})
    audit_pack(out_qwp, expected_shapes=expected_shapes)
    out_qwp.with_name(out_qwp.name + ".sha256").write_text(f"{sha256(out_qwp)}  {out_qwp.name}\n")
    return stats


def _stage_assets(source: Path, assets: Path, tc: dict, reader: ShardReader, *,
                  model_id: str, geo_digest: str, require_tokenizer: bool, write_qlsa: bool) -> str:
    assets.mkdir(parents=True, exist_ok=True)
    if require_tokenizer and not (source / "tokenizer.json").is_file():
        raise FileNotFoundError(f"required asset missing: {source / 'tokenizer.json'}")
    embedded_tpl = None
    if not (source / "chat_template.jinja").is_file() and (source / "tokenizer.json").is_file():
        if (tok_cfg_p := source / "tokenizer_config.json").is_file():
            embedded_tpl = json.loads(tok_cfg_p.read_text()).get("chat_template")
        if not isinstance(embedded_tpl, str) or not embedded_tpl:
            embedded_tpl = synthesize_ernie_chat_template() if is_ernie_chat_source(source) else synthesize_base_chat_template(source)
    manifest: dict[str, dict] = {}
    for src in sorted(source.iterdir()):
        if not src.is_file() or src.name.startswith(".") or src.name.endswith((".safetensors", ".index.json")) or src.name == "crc32.txt":
            continue
        shutil.copyfile(src, assets / src.name)
        manifest[src.name] = {"sha256": sha256(src), "bytes": src.stat().st_size}
    if embedded_tpl and not (assets / "chat_template.jinja").is_file():
        tpl_dst = assets / "chat_template.jinja"
        tpl_dst.write_text(embedded_tpl, encoding="utf-8")
        manifest["chat_template.jinja"] = {"sha256": sha256(tpl_dst), "bytes": tpl_dst.stat().st_size}
    if write_qlsa and (qlsa_meta := write_hca_qlsa(reader, tc, assets / "model.qlsa", model_id, geo_digest)) is not None:
        manifest["model.qlsa"] = qlsa_meta
    manifest_path = assets / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return sha256(manifest_path)


def _hash_files(files: list[Path]) -> str:
    h_src = hashlib.sha256()
    for idx, path in enumerate(files, 1):
        if idx == 1 or idx % 16 == 0 or idx == len(files):
            print(f"hashing shard {idx}/{len(files)} ({path.name})", flush=True)
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(16 << 20), b""):
                h_src.update(chunk)
    return h_src.hexdigest()


def _open_ernie_source(source: Path, *, partial_index_ok: bool):
    if not (files := sorted(source.glob("*.safetensors"))):
        raise ValueError(f"no .safetensors under {source}")
    config = json.loads((source / "config.json").read_text())
    tc = config["text_config"] if "text_config" in config else config
    reader = ShardReader(source, files, fp8_block_size=validate_source_quant_config(config), partial_index_ok=partial_index_ok)
    model_id, geo_digest = MODEL_ID[tc["model_type"]], hashlib.sha256(geo_v2(tc).encode()).hexdigest()
    return files, tc, reader, model_id, geo_digest, {"model_id": model_id, "geometry_digest": geo_digest}


def _pack_dspark_and_fi(
    source: Path, reader: ShardReader, tc: dict, out_dir: Path, *, sha256_source_hex: str, identity: dict,
    assets_dir_val: str, assets_manifest_sha256: str, dspark_dense_quant: str, dspark_expert_quant: str,
    dspark_expert_down_quant: str, dspark_markov_w2_quant: str,
) -> tuple[dict, dict | None]:
    ds_stats = pack_dspark_qwp(
        reader, tc, out_dir / "dspark.qwp", source=source, sha256_source_hex=sha256_source_hex, identity=identity,
        assets_dir_val=assets_dir_val, assets_manifest_sha256=assets_manifest_sha256,
        dspark_dense_quant=dspark_dense_quant, dspark_expert_quant=dspark_expert_quant,
        dspark_expert_down_quant=dspark_expert_down_quant, dspark_markov_w2_quant=dspark_markov_w2_quant)
    fi_stats = pack_flashindexer_qwp(source, out_dir / "flashindexer.qwp", identity=identity,
                                     assets_dir_val=assets_dir_val, assets_manifest_sha256=assets_manifest_sha256)
    return ds_stats, fi_stats


def quantize_ernie55_dspark_only(
    source: Path, destination: Path, *, flat_root: bool = False, device: str = "auto",
    dspark_dense_quant: str = "shvq4_g32", dspark_expert_quant: str = "shvq15_g64",
    dspark_expert_down_quant: str = "shvq15_g64", dspark_markov_w2_quant: str = "hq8_g32",
) -> dict:
    """Pack ONLY `dspark.qwp` (+ `flashindexer.qwp` if present) from a full or partial HF snapshot."""
    import torch
    _force_cpu_if_requested(device)
    torch.set_num_threads(min(os.cpu_count() or 8, 16))
    _, tc, reader, _, _, identity = _open_ernie_source(source, partial_index_ok=True)
    dspark_opts = dict(dspark_dense_quant=dspark_dense_quant, dspark_expert_quant=dspark_expert_quant,
                       dspark_expert_down_quant=dspark_expert_down_quant, dspark_markov_w2_quant=dspark_markov_w2_quant)
    try:
        if not (mtp_files := sorted({reader.tensor_to_file[k] for k in reader.tensor_to_file if k.startswith("model.mtp.")})):
            raise ValueError(f"no model.mtp.* tensors found under {source}")
        sha_src = _hash_files(mtp_files)
        if destination.exists():
            existing_assets = resolve_assets_dir(destination)
            if not (manifest_path := existing_assets / "manifest.json").is_file():
                raise FileExistsError(f"output exists without manifest.json: {destination}")
            assets_dir_val, manifest_sha = ("." if existing_assets == destination else "assets"), sha256(manifest_path)
            with tempfile.TemporaryDirectory(prefix=f".{destination.name}-ds-", dir=destination.parent) as temp:
                st_dir = Path(temp)
                ds_stats, fi_stats = _pack_dspark_and_fi(source, reader, tc, st_dir, sha256_source_hex=sha_src, identity=identity,
                                                         assets_dir_val=assets_dir_val, assets_manifest_sha256=manifest_sha, **dspark_opts)
                for fn in ("dspark.qwp", "dspark.qwp.sha256") + (("flashindexer.qwp", "flashindexer.qwp.sha256") if fi_stats is not None else ()):
                    os.replace(st_dir / fn, destination / fn)
            return {"dspark_stats": ds_stats, "flashindexer_stats": fi_stats}
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temp:
            staging, assets_dir_val = Path(temp) / "model", ("." if flat_root else "assets")
            manifest_sha = _stage_assets(source, staging if flat_root else staging / "assets", tc, reader,
                                         model_id=identity["model_id"], geo_digest=identity["geometry_digest"], require_tokenizer=False, write_qlsa=False)
            ds_stats, fi_stats = _pack_dspark_and_fi(source, reader, tc, staging, sha256_source_hex=sha_src, identity=identity,
                                                     assets_dir_val=assets_dir_val, assets_manifest_sha256=manifest_sha, **dspark_opts)
            if destination.exists():
                raise FileExistsError(f"output appeared during packing: {destination}")
            staging.rename(destination)
        return {"dspark_stats": ds_stats, "flashindexer_stats": fi_stats}
    finally:
        reader.close()


def quantize_ernie55_model(
    source: Path, destination: Path, flat_root: bool = False, quant: str = "hq4_g32_m5", group_size: int | None = None,
    dense_quant: str | None = None, expert_quant: str | None = None, expert_down_quant: str | None = None,
    reuse_source_sha_from: Path | None = None, expert_batch_size: int = 64, device: str = "auto",
    dspark_dense_quant: str = "shvq4_g32", dspark_expert_quant: str = "shvq15_g64",
    dspark_expert_down_quant: str = "shvq15_g64", dspark_markov_w2_quant: str = "hq8_g32",
):
    import torch
    _force_cpu_if_requested(device)
    torch.set_num_threads(min(os.cpu_count() or 8, 16))
    if dense_quant is None and expert_quant is None and expert_down_quant is None and quant == "hq4_g32_m5":
        dense_quant, expert_quant, expert_down_quant = "hq8_g32", "shvq15_g64", "shvq3_g64"
    dense_spec = parse_any_quant_spec(normalize_quant_spec(dense_quant or quant), default_group=group_size)
    expert_spec = parse_any_quant_spec(normalize_quant_spec(expert_quant or quant), default_group=group_size)
    expert_down_spec = parse_any_quant_spec(normalize_quant_spec(expert_down_quant or expert_quant or quant), default_group=group_size)
    dense_fmt, expert_fmt, expert_down_fmt = (
        format_any_quant_spec(*dense_spec), format_any_quant_spec(*expert_spec), format_any_quant_spec(*expert_down_spec))
    quant_recorded = (dense_fmt if dense_fmt == expert_fmt == expert_down_fmt
                      else (f"dense_{dense_fmt}__expert_{expert_fmt}" if expert_fmt == expert_down_fmt
                            else f"dense_{dense_fmt}__expert_{expert_fmt}__expdown_{expert_down_fmt}"))
    files, tc, reader, model_id, geo_digest, identity = _open_ernie_source(source, partial_index_ok=False)
    expected_shapes, sha256_source_hex = build_expected_shapes(tc), None
    if reuse_source_sha_from is not None and (rq := Path(reuse_source_sha_from) / "model.qwp").is_file():
        rp = read_header(rq)[0].get("provenance", {})
        if rp.get("source") == str(source) and rp.get("sha256_source"):
            sha256_source_hex = rp["sha256_source"]
    sha256_source_hex = sha256_source_hex or _hash_files(files)
    stats: dict[str, int] = {"bf16": 0, "f32": 0}
    emit = _make_emitter(stats)

    def stream():
        h, num_layers = int(tc["hidden_size"]), int(tc["num_hidden_layers"])
        n_exp = int(tc.get("n_routed_experts") or tc.get("num_experts") or 0)
        has_mhc, streams = bool(tc.get("enable_hyper_connections", False)), int(tc.get("hc_mult", 4))
        use_vha, has_sink = bool(tc.get("use_vha_attention", False)), bool(tc.get("add_full_attention_sink_bias", False))
        has_ogate = bool(tc.get("use_gated_attn", False))
        yield emit("model.embed_tokens.weight", reader.get("model.embed_tokens.weight"), "bf16")
        yield emit("model.norm.weight", reader.get("model.norm.weight"), "bf16")
        if not bool(tc.get("tie_word_embeddings", False)):
            yield emit("lm_head.weight", reader.get("lm_head.weight"), dense_spec, fmt=dense_fmt)
        if has_mhc:
            mw, mb = _fold_exit_head(reader, "model.mhc_contract", "model.mhc_contract.", streams, h)
            yield emit("model.mhc_contract.weight", mw, "f32")
            yield emit("model.mhc_contract.bias", mb, "f32")
        if tc.get("multimax") == "lm_head":
            mm = (reader.get("lm_head.multimax").to(torch.float32).reshape(8) if reader.has("lm_head.multimax")
                  else torch.cat([reader.get("model.lm_head.multimax_ranges").to(torch.float32).reshape(4),
                                  reader.get("model.lm_head.multimax_ts").to(torch.float32).reshape(4)], dim=0))
            yield emit("lm_head.multimax", mm, "f32")
        for l in range(num_layers):
            print(f"quantizing layer {l}/{num_layers - 1}", flush=True)
            blk = f"model.layers.{l}."
            yield emit(blk + "input_layernorm.weight", reader.get(blk + "input_layernorm.weight"), "bf16")
            yield emit(blk + "post_attention_layernorm.weight", reader.get(blk + "post_attention_layernorm.weight"), "bf16")
            if has_mhc:
                yield from _emit_mhc_layer(reader, blk, blk, streams, h, emit)
            if _is_compressed_layer(tc, l):
                fused_in = (reader.get(blk + "self_attn.fused_in_proj.weight") if reader.has(blk + "self_attn.fused_in_proj.weight")
                            else torch.cat([
                                reader.get(blk + "self_attn.linear_q_down_proj.weight"), reader.get(blk + "self_attn.linear_kv_proj.weight"),
                                reader.get_first(blk + "self_attn.core_attention.compressor.linear_wkv.weight", blk + "self_attn.compress_kv.wkv.weight"),
                                reader.get_first(blk + "self_attn.core_attention.compressor.linear_wgate.weight", blk + "self_attn.compress_kv.wgate.weight"),
                            ], dim=0))
                yield emit(blk + "self_attn.fused_in_proj.weight", fused_in, dense_spec, fmt=dense_fmt)
                del fused_in
                yield emit(blk + "self_attn.comp_norm.weight", reader.get_first(
                    blk + "self_attn.comp_norm.weight", blk + "self_attn.core_attention.compressor.norm.weight", blk + "self_attn.compress_kv.norm.weight"), "bf16")
                yield emit(blk + "self_attn.compressor.ape", reader.get_first(blk + "self_attn.compressor.ape", blk + "self_attn.core_attention.compressor.ape"), "f32")
                yield from _emit_hca_attn_common(reader, tc, blk, blk, dense_spec, dense_fmt, emit)
            else:
                q_heads = int(tc.get("hybrid_mla_num_attention_heads", tc["num_attention_heads"]))
                q_rank = int(tc.get("hybrid_mla_q_lora_rank", tc["q_lora_rank"]))
                nope_d, rope_d = int(tc.get("hybrid_mla_qk_nope_head_dim", tc["qk_nope_head_dim"])), int(tc.get("hybrid_mla_qk_rope_head_dim", tc["qk_rope_head_dim"]))
                idx_pfx = blk + ("self_attn.indexer." if reader.has(blk + "self_attn.indexer.wk.weight") else "self_attn.core_attention.indexer.")
                for dst, src in (("q_a_proj.weight", blk + "self_attn.q_a_proj.weight"), ("kv_a_proj_with_mqa.weight", blk + "self_attn.kv_a_proj_with_mqa.weight"),
                                 ("indexer.wk.weight", idx_pfx + "wk.weight")):
                    yield emit(blk + "self_attn." + dst, reader.get(src), dense_spec, fmt=dense_fmt)
                for dst, src, dt in (("q_a_layernorm.weight", blk + "self_attn.q_a_layernorm.weight", "bf16"), ("kv_a_layernorm.weight", blk + "self_attn.kv_a_layernorm.weight", "bf16"),
                                     ("indexer.k_norm.weight", idx_pfx + "k_norm.weight", "bf16"), ("indexer.k_norm.bias", idx_pfx + "k_norm.bias", "bf16"),
                                     ("indexer.weights_proj.weight", idx_pfx + "weights_proj.weight", "f32")):
                    yield emit(blk + "self_attn." + dst, reader.get(src), dt)
                if reader.has(blk + "self_attn.q_b_gate_idx_proj.weight"):
                    qb_fused = reader.get(blk + "self_attn.q_b_gate_idx_proj.weight")
                else:
                    if reader.has(blk + "self_attn.q_b_proj.weight"):
                        qb_raw = reader.get(blk + "self_attn.q_b_proj.weight").view(q_heads, nope_d + rope_d, q_rank)
                        parts = [qb_raw[:, :nope_d, :].reshape(q_heads * nope_d, q_rank), qb_raw[:, nope_d:, :].reshape(q_heads * rope_d, q_rank)]
                    else:
                        parts = [reader.get(blk + "self_attn.q_b_nope_proj.weight"), reader.get(blk + "self_attn.q_b_rope_proj.weight")]
                    if has_ogate:
                        parts.append(reader.get(blk + "self_attn.gate_proj.weight"))
                    parts.append(reader.get(idx_pfx + "wq_b.weight"))
                    qb_fused = torch.cat(parts, dim=0)
                yield emit(blk + "self_attn.q_b_gate_idx_proj.weight", qb_fused, dense_spec, fmt=dense_fmt)
                del qb_fused
                yield emit(blk + "self_attn.kv_b_proj.weight", reader.get(blk + "self_attn.kv_b_proj.weight"), dense_spec, fmt=dense_fmt)
                yield emit(blk + "self_attn.o_proj.weight", reader.get(blk + "self_attn.o_proj.weight"), dense_spec, fmt=dense_fmt)
                if use_vha:
                    yield emit(blk + "self_attn.vha_mix.weight", _fold_vha(reader, blk, q_heads), "f32")
                if has_sink:
                    yield emit(blk + "self_attn.attn_sink", reader.get_first(blk + "self_attn.attn_sink", blk + "self_attn.core_attention.softmax_offset"), "f32")
            if _is_moe_layer(tc, l):
                yield from _emit_moe(reader, tc, blk, blk, dense_spec, dense_fmt, expert_spec, expert_fmt,
                                     expert_down_spec, expert_down_fmt, emit, expert_batch_size=expert_batch_size, stacked_experts=False)
            else:
                for k in ("gate_proj", "up_proj", "down_proj"):
                    yield emit(blk + f"mlp.{k}.weight", reader.get(blk + f"mlp.{k}.weight"), dense_spec, fmt=dense_fmt)
        if int(tc.get("num_nextn_predict_layers") or 0) >= 1 and bool(tc.get("mtp_shared_last_layer", False)):
            eblk = f"model.layers.{num_layers}."
            for k, dt in (("enorm.weight", "bf16"), ("hnorm.weight", "bf16"), ("shared_head.norm.weight", "bf16")):
                yield emit(eblk + k, reader.get(eblk + k), dt)
            for k in ("e_proj.weight", "h_proj.weight"):
                yield emit(eblk + k, reader.get(eblk + k), dense_spec, fmt=dense_fmt)
            ew, eb_b = _fold_exit_head(reader, eblk + "mhc_contract", eblk, streams, h)
            yield emit(eblk + "mhc_contract.weight", ew, "f32")
            yield emit(eblk + "mhc_contract.bias", eb_b, "f32")
            if n_exp > 0:
                yield emit(eblk + "mlp.gate.e_score_correction_bias", reader.get_first(
                    eblk + "mlp.gate.e_score_correction_bias", eblk + "block_sparse_moe.e_score_correction_bias",
                    f"model.layers.{num_layers - 1}.mlp.gate.e_score_correction_bias",
                    f"model.layers.{num_layers - 1}.block_sparse_moe.e_score_correction_bias"), "f32")

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temp:
            staging, assets_dir_val = Path(temp) / "model", ("." if flat_root else "assets")
            manifest_sha = _stage_assets(source, staging if flat_root else staging / "assets", tc, reader,
                                         model_id=model_id, geo_digest=geo_digest, require_tokenizer=True, write_qlsa=True)
            weights = staging / "model.qwp"
            pack_file(stream(), weights, {
                **identity, "source": str(source), "sha256_source": sha256_source_hex,
                "source_revision": (source.name if source.parent.name == "snapshots" and len(source.name) == 40
                                    and all(c in "0123456789abcdef" for c in source.name) else None),
                "quant_recorded": quant_recorded, "hq8_reference": None,
                "assets_dir": assets_dir_val, "assets_manifest_sha256": manifest_sha,
            })
            audit_pack(weights, expected_shapes=expected_shapes)
            (staging / "model.qwp.sha256").write_text(f"{sha256(weights)}  model.qwp\n")
            if _has_bundled_dspark(reader):
                _pack_dspark_and_fi(source, reader, tc, staging, sha256_source_hex=sha256_source_hex, identity=identity,
                                    assets_dir_val=assets_dir_val, assets_manifest_sha256=manifest_sha,
                                    dspark_dense_quant=dspark_dense_quant, dspark_expert_quant=dspark_expert_quant,
                                    dspark_expert_down_quant=dspark_expert_down_quant, dspark_markov_w2_quant=dspark_markov_w2_quant)
            else:
                pack_flashindexer_qwp(source, staging / "flashindexer.qwp", identity=identity,
                                      assets_dir_val=assets_dir_val, assets_manifest_sha256=manifest_sha)
            if destination.exists():
                raise FileExistsError(f"output appeared during quantization: {destination}")
            staging.rename(destination)
    finally:
        reader.close()
    print(f"packed quant={quant_recorded} stats={stats} -> {destination}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dspark-only", action="store_true")
    ap.add_argument("--flat-root", action="store_true")
    ap.add_argument("--quant", default="hq4_g32_m5")
    ap.add_argument("--group-size", type=int, default=None, choices=list(VALID_GROUPS))
    for k in ("--dense-quant", "--expert-quant", "--expert-down-quant"):
        ap.add_argument(k, default=None)
    ap.add_argument("--reuse-source-sha-from", type=Path, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    for flag, dflt in (("--dspark-dense-quant", "shvq4_g32"), ("--dspark-expert-quant", "shvq15_g64"),
                       ("--dspark-expert-down-quant", "shvq15_g64"), ("--dspark-markov-w2-quant", "hq8_g32")):
        ap.add_argument(flag, default=dflt)
    a = ap.parse_args(argv)
    ds_kw = dict(device=a.device, dspark_dense_quant=a.dspark_dense_quant, dspark_expert_quant=a.dspark_expert_quant,
                 dspark_expert_down_quant=a.dspark_expert_down_quant, dspark_markov_w2_quant=a.dspark_markov_w2_quant)
    src, dst = a.model.expanduser().resolve(), a.out.expanduser().absolute()
    if a.dspark_only:
        print(json.dumps(quantize_ernie55_dspark_only(src, dst, flat_root=a.flat_root, **ds_kw), indent=2, sort_keys=True))
    else:
        quantize_ernie55_model(src, dst, flat_root=a.flat_root, quant=a.quant, group_size=a.group_size,
                               dense_quant=a.dense_quant, expert_quant=a.expert_quant,
                               expert_down_quant=a.expert_down_quant, reuse_source_sha_from=a.reuse_source_sha_from, **ds_kw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
