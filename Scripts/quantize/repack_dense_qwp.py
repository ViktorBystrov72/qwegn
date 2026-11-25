"""Repack local QWP1 dense HQ8_G32 weights -> SHVQ4_G32 while preserving experts & raw planes verbatim.
Supports --keep-hq8-head (keep lm_head.weight HQ8_G32), --bf16-routers (cast F32 routers -> BF16),
and carries over assets/ + sibling dspark.qwp / draft_head.qwp / flashindexer.qwp packs atomically.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
import numpy as np

from ernie55_adapter import _hq_meta, normalize_quant_spec
from hq_blocks import dequant_hq8_g32_m5_rot, dequant_hq_blocks_rot, hq_block_bytes, quant_hq_blocks, resolve_group_size
from inkling2_adapter import maybe_force_cpu_quant
from pack import pack_file, resolve_assets_dir, sha256
from shvq import any_block_bytes, format_any_quant_spec, parse_any_quant_spec, quant_shvq_blocks
from verify import audit_pack, compare_planes, read_header

_SIBLING_PACKS = ("dspark.qwp", "draft_head.qwp", "flashindexer.qwp")
_RAW_NP_DTYPES = {"bf16": np.uint16, "f16": np.float16, "f32": np.float32, "f8_e4m3": np.uint8}


def _is_hq8_entry(entry: dict) -> bool:
    if "hq8_off" in entry:
        return True
    hq = entry.get("hq")
    return "hq4_off" in entry and isinstance(hq, dict) and hq.get("bits") == 8 and not hq.get("vq")


def _is_expert_entry(name: str, entry: dict) -> bool:
    return ".experts." in name or bool(entry.get("experts"))


def _raw_kind(entry: dict) -> str:
    return str(entry.get("kind") or entry.get("dtype") or "f16")


def _is_router_entry(name: str, entry: dict) -> bool:
    return (name.endswith("mlp.gate.weight") or name.endswith("block_sparse_moe.gate.weight")) and _raw_kind(entry) == "f32"


def _updated_quant_recorded(prev_quant: str | None, dst_fmt: str) -> str:
    if not prev_quant:
        return dst_fmt
    if re.search(r"(^|__)dense_hq8[^_]*(?:_g\d+)?", prev_quant):
        return re.sub(r"dense_hq8(?:_g\d+)?", f"dense_{dst_fmt}", prev_quant, count=1)
    if prev_quant.startswith("hq8"):
        return dst_fmt
    return f"dense_{dst_fmt}__{prev_quant}"


def _read_raw_tensor(fp, entry: dict, *, cast_to_bf16: bool):
    import torch
    shape = [int(d) for d in entry["shape"]]
    dtype = _raw_kind(entry)
    if dtype not in _RAW_NP_DTYPES:
        raise ValueError(f"unsupported raw dtype {dtype!r} for {shape}")
    np_dt = _RAW_NP_DTYPES[dtype]
    n_bytes = int(entry["n_bytes"]) if "n_bytes" in entry else math.prod(shape) * np.dtype(np_dt).itemsize
    fp.seek(int(entry["raw_off"]))
    arr = np.frombuffer(fp.read(n_bytes), dtype=np_dt).reshape(shape).copy()
    if cast_to_bf16:
        if dtype != "f32":
            raise ValueError(f"expected f32 router before bf16 cast, got {dtype!r}")
        return torch.from_numpy(arr).to(torch.bfloat16), "bf16"
    if dtype == "bf16":
        return torch.from_numpy(arr).view(torch.bfloat16), "bf16"
    return arr, dtype


def _requantize_hq8_entry(fp, name: str, entry: dict, dst_spec: tuple[int, int, bool, bool], *,
                          device: str | None, chunk_rows: int = 4096):
    shape = [int(d) for d in entry["shape"]]
    if len(shape) != 2:
        raise ValueError(f"expected 2D HQ8 weight for {name}, got {shape}")
    n, k = shape
    db, dg, dm5, dshvq = dst_spec
    eff_dg = resolve_group_size(k, dg)
    dst_meta = _hq_meta(db, eff_dg, dm5, dshvq)
    dst_bbytes = any_block_bytes(db, eff_dg, is_m5=(dst_meta is None), is_shvq=dshvq)
    out_blocks = np.empty((n, k // eff_dg, dst_bbytes), dtype=np.uint8)

    if "hq8_off" in entry:
        fp.seek(int(entry["hq8_off"]))
        qs = np.frombuffer(fp.read(n * k), dtype=np.int8).reshape(n, k)
        fp.seek(int(entry["hq8_s_off"]))
        scales = np.frombuffer(fp.read(n * (k // 32) * 2), dtype=np.float16).reshape(n, k // 32)
        for r0 in range(0, n, chunk_rows):
            r1 = min(r0 + chunk_rows, n)
            w_rot = dequant_hq8_g32_m5_rot(qs[r0:r1], scales[r0:r1])
            q_chunk, _ = (quant_shvq_blocks(w_rot, bits=db, group=eff_dg, rotate=False, device=device) if dshvq
                          else quant_hq_blocks(w_rot, bits=db, group=eff_dg, is_m5=dm5, rotate=False))
            out_blocks[r0:r1] = q_chunk
        del qs, scales
        return name, out_blocks, ("prepacked_hq4", [n, k], dst_meta)

    src_g = int((entry.get("hq") or {}).get("group", 32))
    src_nb = k // src_g
    src_bbytes = hq_block_bytes(8, src_g)
    src_row_bytes = src_nb * src_bbytes
    base_off = int(entry["hq4_off"])
    for r0 in range(0, n, chunk_rows):
        r1 = min(r0 + chunk_rows, n)
        fp.seek(base_off + r0 * src_row_bytes)
        buf = fp.read((r1 - r0) * src_row_bytes)
        src_chunk = np.frombuffer(buf, dtype=np.uint8).reshape(r1 - r0, src_nb, src_bbytes)
        w_rot = dequant_hq_blocks_rot(src_chunk, bits=8, group=src_g, is_m5=False)
        q_chunk, _ = (quant_shvq_blocks(w_rot, bits=db, group=eff_dg, rotate=False, device=device) if dshvq
                      else quant_hq_blocks(w_rot, bits=db, group=eff_dg, is_m5=dm5, rotate=False))
        out_blocks[r0:r1] = q_chunk
    return name, out_blocks, ("prepacked_hq4", [n, k], dst_meta)


def _copy_assets_and_siblings(src_dir: Path, staging: Path, src_assets: Path, *, flat_root: bool) -> None:
    dst_assets = staging if flat_root else (staging / "assets")
    dst_assets.mkdir(parents=True, exist_ok=True)
    skip = {"model.qwp", "model.qwp.sha256", "crc32.txt"}
    for fn in _SIBLING_PACKS:
        skip.add(fn)
        skip.add(fn + ".sha256")
    for p in sorted(src_assets.iterdir()):
        if not p.is_file() or p.name.startswith(".") or p.name in skip:
            continue
        shutil.copy2(p, dst_assets / p.name)
    for fn in _SIBLING_PACKS:
        for ext in ("", ".sha256"):
            sp = src_dir / (fn + ext)
            if sp.is_file():
                shutil.copy2(sp, staging / (fn + ext))


def repack_dense_qwp(
    src_dir: Path, out_dir: Path, *, dense_quant: str = "shvq4_g32",
    keep_hq8_head: bool = False, bf16_routers: bool = False,
    device: str = "auto", flat_root: bool | None = None, compare: bool = True,
) -> dict:
    """Repack `src_dir/model.qwp` HQ8 dense layers to `dense_quant` into `out_dir`."""
    maybe_force_cpu_quant(device)
    quant_dev = "cpu" if device == "cpu" else None
    src_dir = Path(src_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().absolute()
    src_qwp = src_dir / "model.qwp"
    if not src_qwp.is_file():
        raise FileNotFoundError(f"missing source pack: {src_qwp}")
    if out_dir.exists():
        raise FileExistsError(f"output already exists: {out_dir}")

    dst_spec = parse_any_quant_spec(normalize_quant_spec(dense_quant))
    dst_fmt = format_any_quant_spec(*dst_spec)
    src_hdr, _ = read_header(src_qwp)
    tensors = src_hdr["tensors"]
    src_prov = dict(src_hdr.get("provenance") or {})
    src_assets = resolve_assets_dir(src_dir)
    use_flat = (src_assets == src_dir) if flat_root is None else flat_root

    copy_refs: dict[str, dict] = {}
    requant_names: list[str] = []
    router_cast_names: list[str] = []
    raw_names: list[str] = []

    for name, entry in tensors.items():
        if "raw_off" in entry:
            raw_names.append(name)
            if bf16_routers and _is_router_entry(name, entry):
                router_cast_names.append(name)
        elif _is_hq8_entry(entry) and (not _is_expert_entry(name, entry)) and not (keep_hq8_head and name == "lm_head.weight"):
            requant_names.append(name)
        elif "hq4_off" in entry:
            copy_refs[name] = entry
        else:
            raise ValueError(f"unsupported tensor container for repack: {name}: {list(entry)}")

    if not requant_names:
        raise ValueError(f"no dense HQ8 tensors eligible for repack in {src_qwp}")

    prov = dict(src_prov)
    prov["quant_recorded"] = _updated_quant_recorded(src_prov.get("quant_recorded"), dst_fmt)
    prov["hq8_reference"] = str(src_qwp)
    prov["assets_dir"] = "." if use_flat else "assets"
    if keep_hq8_head:
        prov["dense_head_quant"] = "hq8_g32"
    if bf16_routers:
        prov["bf16_routers"] = True

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out_dir.name}-", dir=out_dir.parent) as temp:
        staging = Path(temp) / "model"
        _copy_assets_and_siblings(src_dir, staging, src_assets, flat_root=use_flat)
        out_qwp = staging / "model.qwp"

        with src_qwp.open("rb") as fp:
            def stream():
                cast_set = set(router_cast_names)
                for name in raw_names:
                    arr, kind = _read_raw_tensor(fp, tensors[name], cast_to_bf16=(name in cast_set))
                    yield name, arr, kind
                total = len(requant_names)
                for idx, name in enumerate(requant_names, 1):
                    if idx == 1 or idx % 32 == 0 or idx == total:
                        print(f"repacking dense {idx}/{total}: {name} -> {dst_fmt}", flush=True)
                    yield _requantize_hq8_entry(fp, name, tensors[name], dst_spec, device=quant_dev)

            pack_file(stream(), out_qwp, prov, copy_entries=(src_qwp, copy_refs))

        expected_shapes = {k: list(v["shape"]) for k, v in tensors.items()}
        audit_pack(out_qwp, expected_shapes=expected_shapes)
        (staging / "model.qwp.sha256").write_text(f"{sha256(out_qwp)}  model.qwp\n")
        compared_bytes, skipped = (0, [])
        if compare:
            allow_changed = set(requant_names) | set(router_cast_names)
            compared_bytes, skipped = compare_planes(out_qwp, src_qwp, allow_requantized=allow_changed)
        if out_dir.exists():
            raise FileExistsError(f"output appeared during repack: {out_dir}")
        staging.rename(out_dir)

    return {
        "out": str(out_dir),
        "dense_quant": dst_fmt,
        "copied_hq4_tensors": len(copy_refs),
        "requantized_dense_tensors": len(requant_names),
        "cast_bf16_routers": len(router_cast_names),
        "kept_hq8_head": keep_hq8_head and "lm_head.weight" in copy_refs,
        "compared_bytes": compared_bytes,
        "skipped_diff_planes": skipped,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path, help="Source QWP model directory containing model.qwp")
    ap.add_argument("--out", required=True, type=Path, help="Destination directory for repacked QWP model")
    ap.add_argument("--dense-quant", default="shvq4_g32")
    ap.add_argument("--keep-hq8-head", action="store_true")
    ap.add_argument("--bf16-routers", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--flat-root", action="store_true")
    ap.add_argument("--no-compare", action="store_true")
    a = ap.parse_args(argv)
    import json
    res = repack_dense_qwp(
        a.src, a.out, dense_quant=a.dense_quant, keep_hq8_head=a.keep_hq8_head,
        bf16_routers=a.bf16_routers, device=a.device,
        flat_root=(True if a.flat_root else None), compare=(not a.no_compare),
    )
    print(json.dumps(res, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
