"""Generalized Decision Model (encoder, marker scorer, reranker, LFM) QWP1 packer."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from ernie55_adapter import ShardReader
from hq_blocks import validate_source_quant_config
from pack import pack_file, sha256
from shvq import format_any_quant_spec, parse_any_quant_spec
from verify import audit_pack

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oracle"))
from emit_pack_identity import MODEL_ID, geo_v2  # noqa: E402

DECISION_RAW_PATTERNS = (
    "shortconv", "conv_l", ".conv.conv.", "conv.weight", "conv.bias",
    "marker_head", "classifier", "state_pointer", "grid_head", "binary_head",
    "score.", ".score", "type_emb", "temperatures", "temperature",
    "norm", "layernorm", "rmsnorm", "embed_tokens", "word_embeddings",
    "embeddings", "lm_head", "vision", "visual", "multi_modal_projector",
    "siglip", ".gate.", "self_attn.gate_proj", "g_proj", "a_log", "dt_bias",
)


def estimate_param_count(shapes: dict[str, list[int]]) -> int:
    """Computes total parameter count from source SafeTensors shapes."""
    total = 0
    for name, dims in shapes.items():
        if name.endswith("_scale_inv"):
            continue
        p = 1
        for d in dims:
            p *= int(d)
        total += p
    return total


def is_decision_raw_plane(name: str, shape: list[int], preserve_all_bf16: bool) -> bool:
    """Returns True if tensor must remain uncompressed BF16 in QWP1."""
    if preserve_all_bf16:
        return True
    if len(shape) != 2 or (shape[1] % 32) != 0:
        return True
    lower = name.lower()
    if any(pat in lower for pat in DECISION_RAW_PATTERNS):
        return True
    if any(lower.endswith(s) for s in (".bias", "_bias", "_emb", ".emb")):
        return True
    return False


def _ensure_assets_and_manifest(
    source: Path,
    assets: Path,
    has_chat_template: str | None,
) -> Path:
    manifest: dict[str, dict] = {}
    for src in sorted(source.iterdir()):
        if (not src.is_file() or src.name.startswith(".")
                or src.name.endswith((".safetensors", ".index.json"))
                or src.name == "crc32.txt"):
            continue
        shutil.copyfile(src, assets / src.name)
        manifest[src.name] = {"sha256": sha256(src), "bytes": src.stat().st_size}
    if has_chat_template and not (assets / "chat_template.jinja").is_file():
        tpl = assets / "chat_template.jinja"
        tpl.write_text(has_chat_template, encoding="utf-8")
        manifest["chat_template.jinja"] = {"sha256": sha256(tpl), "bytes": tpl.stat().st_size}
    manifest_path = assets / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest_path


def _resolve_chat_template(source: Path) -> str:
    if (source / "chat_template.jinja").is_file():
        return (source / "chat_template.jinja").read_text(encoding="utf-8")
    tok_cfg_p = source / "tokenizer_config.json"
    if tok_cfg_p.is_file():
        tok_cfg = json.loads(tok_cfg_p.read_text(encoding="utf-8"))
        if isinstance(tok_cfg.get("chat_template"), str):
            return tok_cfg["chat_template"]
    return "{% for m in messages %}{{ m['content'] }}{% endfor %}"


def _hash_sources(files: list[Path]) -> str:
    h_src = hashlib.sha256()
    for path in files:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(16 << 20), b""):
                h_src.update(chunk)
    return h_src.hexdigest()


def pack_decision_or_lfm_model(
    source: Path,
    destination: Path,
    *,
    flat_root: bool = False,
    quant: str = "hq4_g32_m5",
    group_size: int | None = None,
    dense_quant: str | None = None,
    squeeze_conv1d: bool = False,
    is_decision_hint: bool = True,
) -> Path:
    """Packs an LFM or Generalized Decision checkpoint into QWP1."""
    import torch

    torch.set_num_threads(min(os.cpu_count() or 8, 16))
    source = source.resolve(strict=True)
    destination = destination.expanduser().absolute()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no .safetensors under {source}")

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    tc = config["text_config"] if "text_config" in config else config
    parent_mt = config.get("model_type")
    mt = tc.get("model_type") or parent_mt or ""
    fp8_block_size = validate_source_quant_config(config)
    if not (source / "tokenizer.json").is_file():
        raise FileNotFoundError(f"required asset missing: {source / 'tokenizer.json'}")
    embedded_tpl = _resolve_chat_template(source)

    reader = ShardReader(source, files, fp8_block_size=fp8_block_size)
    est_params = estimate_param_count(reader.shapes)
    force_hq4 = os.environ.get("QW_DECISION_FORCE_HQ4") == "1"
    decision_flagged = is_decision_hint or ("decision_config" in config) or ("decision_config" in tc)
    preserve_small_bf16 = (
        decision_flagged and (est_params <= 8_000_000_000) and not force_hq4 and dense_quant is None
    )

    dense_spec = parse_any_quant_spec(dense_quant or quant, default_group=group_size)
    dense_fmt = format_any_quant_spec(*dense_spec)
    quant_recorded = "bf16" if preserve_small_bf16 else f"decision_{dense_fmt}"
    expected_shapes: dict[str, list[int]] = {}
    for name, shape in reader.shapes.items():
        if name.endswith("_scale_inv"):
            continue
        if squeeze_conv1d and len(shape) == 3 and shape[1] == 1 and "conv" in name.lower():
            expected_shapes[name] = [shape[0], shape[2]]
        else:
            expected_shapes[name] = list(shape)

    identity = {
        "model_id": MODEL_ID[mt],
        "geometry_digest": hashlib.sha256(geo_v2(tc, parent_mt=parent_mt).encode()).hexdigest(),
    }
    sha256_source_hex = _hash_sources(files)
    stats: dict[str, int] = {"bf16": 0, dense_fmt: 0}

    def stream():
        for name in sorted(reader.tensor_to_file):
            if name.endswith("_scale_inv"):
                continue
            t = reader.get(name)
            shape = list(t.shape)
            if squeeze_conv1d and len(shape) == 3 and shape[1] == 1 and "conv" in name.lower():
                t = t.reshape(shape[0], shape[2]).contiguous()
                shape = [shape[0], shape[2]]
            if is_decision_raw_plane(name, shape, preserve_small_bf16):
                stats["bf16"] += 1
                yield name, t, "bf16"
            else:
                stats[dense_fmt] = stats.get(dense_fmt, 0) + 1
                yield name, t, dense_spec

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temp:
            staging = Path(temp) / "model"
            assets = staging if flat_root else staging / "assets"
            assets.mkdir(parents=True, exist_ok=True)
            manifest_path = _ensure_assets_and_manifest(source, assets, embedded_tpl)
            weights = staging / "model.qwp"
            pack_file(stream(), weights, {
                **identity,
                "source": str(source),
                "sha256_source": sha256_source_hex,
                "source_revision": None,
                "quant_recorded": quant_recorded,
                "hq8_reference": None,
                "assets_dir": "." if flat_root else "assets",
                "assets_manifest_sha256": sha256(manifest_path),
            })
            audit_pack(weights, expected_shapes=expected_shapes)
            (staging / "model.qwp.sha256").write_text(f"{sha256(weights)}  model.qwp\n")
            staging.rename(destination)
    finally:
        reader.close()
    print(f"packed decision quant={quant_recorded} stats={stats} -> {destination}", flush=True)
    return destination


def quantize_decision_model(
    source: Path,
    destination: Path,
    flat_root: bool = False,
    quant: str = "hq4_g32_m5",
    group_size: int | None = None,
    dense_quant: str | None = None,
    expert_quant: str | None = None,
    expert_down_quant: str | None = None,
) -> Path:
    """Entrypoint for encoder/marker/reranker/omni decision checkpoints."""
    _ = expert_quant, expert_down_quant
    return pack_decision_or_lfm_model(
        source, destination, flat_root=flat_root, quant=quant, group_size=group_size,
        dense_quant=dense_quant, squeeze_conv1d=True, is_decision_hint=True,
    )
