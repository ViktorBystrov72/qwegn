"""Maxwell-MoE (maxwell_moe) QWP1 packer adapter.
Folds PyTorch mHC/HCHead weights without transposing row-major projections,
emits canonical GQA + MBSA indexer, dense/sparse MoE, and dedicated mHC-MTP.
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


def _head_dim(tc: dict) -> int:
    return int(tc.get("attention_head_size") or tc.get("head_dim") or (int(tc["hidden_size"]) // int(tc["num_attention_heads"])))


def _is_dense_layer(tc: dict, layer: int) -> bool:
    dense_layers = tc.get("mlp_only_layers") or []
    if dense_layers:
        return layer in set(int(x) for x in dense_layers)
    return layer < int(tc.get("first_k_dense_replace", 0))


def _fold_mhc_pytorch(reader: ShardReader, blk: str, hf_sub: str, streams: int, h: int):
    import torch

    rows = 2 * streams + streams * streams
    width = streams * h
    pfx = blk + hf_sub + "."
    mp = reader.get(pfx + "mapping_proj.weight").to(torch.float32)
    if list(mp.shape) == [width, rows]:
        mp = mp.T.contiguous()
    elif list(mp.shape) == [rows, width]:
        mp = mp.contiguous()
    else:
        raise ValueError(f"{pfx}mapping_proj.weight unexpected shape {list(mp.shape)}")
    a_pre = float(reader.get(pfx + "alpha_pre").to(torch.float32).reshape(-1)[0].item()) if reader.has(pfx + "alpha_pre") else 1.0
    a_post = float(reader.get(pfx + "alpha_post").to(torch.float32).reshape(-1)[0].item()) if reader.has(pfx + "alpha_post") else 1.0
    a_res = float(reader.get(pfx + "alpha_res").to(torch.float32).reshape(-1)[0].item()) if reader.has(pfx + "alpha_res") else 1.0
    mp[:streams] *= a_pre
    mp[streams: 2 * streams] *= a_post
    mp[2 * streams:] *= a_res
    b = reader.get(pfx + "bias").to(torch.float32).reshape(rows).contiguous()
    return mp, b


def _fold_hc_head_pytorch(reader: ShardReader, prefix: str, streams: int, h: int):
    import torch

    width = streams * h
    fn = reader.get(prefix + ".hc_head_fn").to(torch.float32)
    if list(fn.shape) == [width, streams]:
        fn = fn.T.contiguous()
    elif list(fn.shape) == [streams, width]:
        fn = fn.contiguous()
    else:
        raise ValueError(f"{prefix}.hc_head_fn unexpected shape {list(fn.shape)}")
    sc_name = prefix + ".hc_head_scale"
    if reader.has(sc_name):
        sc_raw = reader.get(sc_name).to(torch.float32).reshape(-1)
        sc = sc_raw.reshape(streams, 1) if sc_raw.numel() == streams else sc_raw[0]
        fn = (fn * sc).contiguous()
    bs = reader.get(prefix + ".hc_head_base").to(torch.float32).reshape(streams).contiguous()
    return fn, bs


def _append_layer_shapes(exp: dict[str, list[int]], tc: dict, blk: str, *, is_moe: bool, include_indexer: bool) -> None:
    h = int(tc["hidden_size"])
    d = _head_dim(tc)
    qh = int(tc["num_attention_heads"])
    kvh = int(tc["num_key_value_heads"])
    inter = int(tc["intermediate_size"])
    n_exp = int(tc.get("num_experts") or 0)
    moe_inter = int(tc.get("moe_intermediate_size", inter))
    sh_inter = int(tc.get("shared_expert_intermediate_size") or 0)
    streams = int(tc.get("num_residual_streams", 4))
    mhc_rows = 2 * streams + streams * streams

    exp[blk + "input_layernorm.weight"] = [h]
    exp[blk + "post_attention_layernorm.weight"] = [h]
    if bool(tc.get("enable_hyper_connections", True)):
        exp[blk + "mhc_attn.weight"] = [mhc_rows, streams * h]
        exp[blk + "mhc_attn.bias"] = [mhc_rows]
        exp[blk + "mhc_ffn.weight"] = [mhc_rows, streams * h]
        exp[blk + "mhc_ffn.bias"] = [mhc_rows]
    exp[blk + "self_attn.q_proj.weight"] = [qh * d, h]
    exp[blk + "self_attn.k_proj.weight"] = [kvh * d, h]
    exp[blk + "self_attn.v_proj.weight"] = [kvh * d, h]
    exp[blk + "self_attn.o_proj.weight"] = [h, qh * d]
    exp[blk + "self_attn.q_norm.weight"] = [d]
    exp[blk + "self_attn.k_norm.weight"] = [d]
    if include_indexer:
        exp[blk + "self_attn.indexer.linear_q_proj.weight"] = [kvh * d, h]
        exp[blk + "self_attn.indexer.linear_k_proj.weight"] = [d, h]
        exp[blk + "self_attn.indexer.q_layernorm.weight"] = [d]
        exp[blk + "self_attn.indexer.k_layernorm.weight"] = [d]
    if is_moe:
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


def build_maxwell_expected_shapes(tc: dict) -> dict[str, list[int]]:
    h = int(tc["hidden_size"])
    v = int(tc["vocab_size"])
    num_layers = int(tc["num_hidden_layers"])
    streams = int(tc.get("num_residual_streams", 4))
    has_mhc = bool(tc.get("enable_hyper_connections", True))
    has_idx = tc.get("attention_variant") == "block_sparse_gqa"
    exp: dict[str, list[int]] = {
        "model.embed_tokens.weight": [v, h],
        "model.norm.weight": [h],
    }
    if not bool(tc.get("tie_word_embeddings", False)):
        exp["lm_head.weight"] = [v, h]
    if has_mhc:
        exp["model.mhc_contract.weight"] = [streams, streams * h]
        exp["model.mhc_contract.bias"] = [streams]
    for l in range(num_layers):
        _append_layer_shapes(
            exp, tc, f"model.layers.{l}.",
            is_moe=not _is_dense_layer(tc, l), include_indexer=has_idx)
    if int(tc.get("mtp_num_layers") or 0) >= 1:
        mpfx = "mtp_layers.0."
        exp[mpfx + "enorm.weight"] = [h]
        exp[mpfx + "hnorm.weight"] = [h]
        exp[mpfx + "e_proj.weight"] = [h, h]
        exp[mpfx + "h_proj.weight"] = [h, h]
        exp[mpfx + "final_layernorm.weight"] = [h]
        if has_mhc:
            exp[mpfx + "mhc_contract.weight"] = [streams, streams * h]
            exp[mpfx + "mhc_contract.bias"] = [streams]
        _append_layer_shapes(exp, tc, mpfx + "decoder_layer.", is_moe=True, include_indexer=False)
    return exp


def _stream_decoder_layer(
    reader: ShardReader, tc: dict, blk: str, *,
    is_moe: bool, include_indexer: bool, dense_spec, expert_spec, expert_down_spec,
    dense_fmt: str, expert_fmt: str, expert_down_fmt: str, bump, expert_batch_size: int
):
    import torch

    h = int(tc["hidden_size"])
    n_exp = int(tc.get("num_experts") or 0)
    moe_inter = int(tc.get("moe_intermediate_size", tc["intermediate_size"]))
    sh_inter = int(tc.get("shared_expert_intermediate_size") or 0)
    has_mhc = bool(tc.get("enable_hyper_connections", True))
    streams = int(tc.get("num_residual_streams", 4))

    bump("bf16", 2)
    yield blk + "input_layernorm.weight", reader.get(blk + "input_layernorm.weight"), "bf16"
    yield blk + "post_attention_layernorm.weight", reader.get(blk + "post_attention_layernorm.weight"), "bf16"

    if has_mhc:
        attn_sub = "self_attn_hyper_connection" if reader.has(blk + "self_attn_hyper_connection.bias") else "self_attention_hyper_connection"
        aw, ab = _fold_mhc_pytorch(reader, blk, attn_sub, streams, h)
        fw, fb = _fold_mhc_pytorch(reader, blk, "mlp_hyper_connection", streams, h)
        bump("f32", 4)
        yield blk + "mhc_attn.weight", aw, "f32"
        yield blk + "mhc_attn.bias", ab, "f32"
        yield blk + "mhc_ffn.weight", fw, "f32"
        yield blk + "mhc_ffn.bias", fb, "f32"

    bump(dense_fmt, 4)
    yield blk + "self_attn.q_proj.weight", reader.get(blk + "self_attn.q_proj.weight"), dense_spec
    yield blk + "self_attn.k_proj.weight", reader.get(blk + "self_attn.k_proj.weight"), dense_spec
    yield blk + "self_attn.v_proj.weight", reader.get(blk + "self_attn.v_proj.weight"), dense_spec
    yield blk + "self_attn.o_proj.weight", reader.get(blk + "self_attn.o_proj.weight"), dense_spec
    bump("bf16", 2)
    yield blk + "self_attn.q_norm.weight", reader.get_first(blk + "self_attn.q_layernorm.weight", blk + "self_attn.q_norm.weight"), "bf16"
    yield blk + "self_attn.k_norm.weight", reader.get_first(blk + "self_attn.k_layernorm.weight", blk + "self_attn.k_norm.weight"), "bf16"

    if include_indexer:
        bump(dense_fmt, 2)
        bump("bf16", 2)
        yield blk + "self_attn.indexer.linear_q_proj.weight", reader.get(blk + "self_attn.indexer.linear_q_proj.weight"), dense_spec
        yield blk + "self_attn.indexer.linear_k_proj.weight", reader.get(blk + "self_attn.indexer.linear_k_proj.weight"), dense_spec
        yield blk + "self_attn.indexer.q_layernorm.weight", reader.get(blk + "self_attn.indexer.q_layernorm.weight"), "bf16"
        yield blk + "self_attn.indexer.k_layernorm.weight", reader.get(blk + "self_attn.indexer.k_layernorm.weight"), "bf16"

    if is_moe:
        bump("f32", 2)
        yield blk + "mlp.gate.weight", reader.get(blk + "block_sparse_moe.gate.weight").to(torch.float32), "f32"
        yield blk + "mlp.gate.e_score_correction_bias", reader.get_first(
            blk + "block_sparse_moe.expert_bias", blk + "block_sparse_moe.e_score_correction_bias"
        ).to(torch.float32), "f32"
        if sh_inter > 0:
            bump(dense_fmt, 3)
            yield blk + "mlp.shared_experts.gate_proj.weight", reader.get(blk + "block_sparse_moe.shared_expert.w1.weight"), dense_spec
            yield blk + "mlp.shared_experts.up_proj.weight", reader.get(blk + "block_sparse_moe.shared_expert.w3.weight"), dense_spec
            yield blk + "mlp.shared_experts.down_proj.weight", reader.get(blk + "block_sparse_moe.shared_expert.w2.weight"), dense_spec
        eb, eg, em5, eshvq = expert_spec
        edb, edg, edm5, edshvq = expert_down_spec
        step = max(1, expert_batch_size)
        for e0 in range(0, n_exp, step):
            e1 = min(n_exp, e0 + step)
            cnt = e1 - e0
            gu_list, dn_list = [], []
            for e in range(e0, e1):
                epfx = blk + f"block_sparse_moe.experts.{e}."
                gu_list.append(reader.get(epfx + "w1.weight"))
                gu_list.append(reader.get(epfx + "w3.weight"))
                dn_list.append(reader.get(epfx + "w2.weight"))
            gu_cat = torch.cat(gu_list, dim=0)
            del gu_list
            gu_blocks, eff_eg = quant_any_blocks(gu_cat, bits=eb, group=eg, is_m5=em5, is_shvq=eshvq)
            del gu_cat
            gu_meta = _hq_meta(eb, eff_eg, em5, eshvq)
            gu_bbytes = any_block_bytes(eb, eff_eg, is_m5=(gu_meta is None), is_shvq=eshvq)
            gu_blocks = gu_blocks.reshape(2 * cnt, moe_inter, h // eff_eg, gu_bbytes)

            dn_cat = torch.cat(dn_list, dim=0)
            del dn_list
            dn_blocks, eff_edg = quant_any_blocks(dn_cat, bits=edb, group=edg, is_m5=edm5, is_shvq=edshvq)
            del dn_cat
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
        yield blk + "mlp.gate_proj.weight", reader.get(blk + "block_sparse_moe.w1.weight"), dense_spec
        yield blk + "mlp.up_proj.weight", reader.get(blk + "block_sparse_moe.w3.weight"), dense_spec
        yield blk + "mlp.down_proj.weight", reader.get(blk + "block_sparse_moe.w2.weight"), dense_spec


def quantize_maxwell_model(
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
    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no .safetensors under {source}")
    config = json.loads((source / "config.json").read_text())
    tc = config["text_config"] if "text_config" in config else config
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
    expected_shapes = build_maxwell_expected_shapes(tc)

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
        "model_id": MODEL_ID[tc["model_type"]],
        "geometry_digest": hashlib.sha256(geo_v2(tc).encode()).hexdigest(),
    }
    stats: dict[str, int] = {"bf16": 0, "f32": 0}

    def bump(k: str, n: int = 1) -> None:
        stats[k] = stats.get(k, 0) + n

    def stream():
        h = int(tc["hidden_size"])
        num_layers = int(tc["num_hidden_layers"])
        has_mhc = bool(tc.get("enable_hyper_connections", True))
        streams = int(tc.get("num_residual_streams", 4))
        has_idx = tc.get("attention_variant") == "block_sparse_gqa"

        bump("bf16", 2)
        yield "model.embed_tokens.weight", reader.get("model.embed_tokens.weight"), "bf16"
        yield "model.norm.weight", reader.get("model.norm.weight"), "bf16"
        if not bool(tc.get("tie_word_embeddings", False)):
            bump(dense_fmt)
            yield "lm_head.weight", reader.get("lm_head.weight"), dense_spec
        if has_mhc:
            mw, mb = _fold_hc_head_pytorch(reader, "model.hc_head", streams, h)
            bump("f32", 2)
            yield "model.mhc_contract.weight", mw, "f32"
            yield "model.mhc_contract.bias", mb, "f32"

        for l in range(num_layers):
            print(f"quantizing layer {l}/{num_layers - 1}", flush=True)
            yield from _stream_decoder_layer(
                reader, tc, f"model.layers.{l}.",
                is_moe=not _is_dense_layer(tc, l), include_indexer=has_idx,
                dense_spec=dense_spec, expert_spec=expert_spec, expert_down_spec=expert_down_spec,
                dense_fmt=dense_fmt, expert_fmt=expert_fmt, expert_down_fmt=expert_down_fmt,
                bump=bump, expert_batch_size=expert_batch_size,
            )

        if int(tc.get("mtp_num_layers") or 0) >= 1:
            mpfx = "mtp_layers.0."
            bump("bf16", 3)
            bump(dense_fmt, 2)
            yield mpfx + "enorm.weight", reader.get(mpfx + "enorm.weight"), "bf16"
            yield mpfx + "hnorm.weight", reader.get(mpfx + "hnorm.weight"), "bf16"
            yield mpfx + "e_proj.weight", reader.get(mpfx + "e_proj.weight"), dense_spec
            yield mpfx + "h_proj.weight", reader.get(mpfx + "h_proj.weight"), dense_spec
            yield mpfx + "final_layernorm.weight", reader.get(mpfx + "final_layernorm.weight"), "bf16"
            if has_mhc:
                mw, mb = _fold_hc_head_pytorch(reader, mpfx + "hc_head", streams, h)
                bump("f32", 2)
                yield mpfx + "mhc_contract.weight", mw, "f32"
                yield mpfx + "mhc_contract.bias", mb, "f32"
            yield from _stream_decoder_layer(
                reader, tc, mpfx + "decoder_layer.",
                is_moe=True, include_indexer=False,
                dense_spec=dense_spec, expert_spec=expert_spec, expert_down_spec=expert_down_spec,
                dense_fmt=dense_fmt, expert_fmt=expert_fmt, expert_down_fmt=expert_down_fmt,
                bump=bump, expert_batch_size=expert_batch_size,
            )

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
