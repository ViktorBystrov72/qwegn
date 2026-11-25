"""Selective QWP1 dense repacker: requantize wide dense projections while keeping experts,
lm_head, embed_tokens, routers, mhc, norms, and sensitive bottlenecks verbatim via copy_entries.
Supports both full model directories (preserving assets/ + sibling packs) and .qwp files.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile

from hq_blocks import resolve_group_size
from inkling2_adapter import maybe_force_cpu_quant
from pack import pack_file, resolve_assets_dir, sha256
from repack_dense_qwp import _copy_assets_and_siblings
from repack_dspark_qwp import (
    _entry_quant_spec,
    _parse_quant_or_keep,
    _read_full_raw,
    _requant_2d,
)
from shvq import format_any_quant_spec
from verify import audit_pack, compare_planes, planes, read_header, verify_assets

_WIDE_ATTN_SUFFIXES = (
    ".self_attn.q_up_gate_proj.weight",
    ".self_attn.linear_q_up_proj.weight",
    ".self_attn.q_up_proj.weight",
    ".self_attn.gate_proj.weight",
    ".self_attn.wo_a.weight",
    ".self_attn.linear_o_group_proj.weight",
    ".self_attn.linear_o_group_proj",
    ".self_attn.o_proj.weight",
    ".self_attn.q_b_gate_idx_proj.weight",
    ".self_attn.q_b_proj.weight",
    ".self_attn.q_b_nope_proj.weight",
    ".self_attn.q_b_rope_proj.weight",
)
_SHARED_SUFFIXES = (
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
    ".gate_up_proj.weight",
    ".w1.weight",
    ".w2.weight",
    ".w3.weight",
)
_DENSE_MLP_SUFFIXES = tuple(
    p + s for p in (".mlp", ".block_sparse_moe") for s in _SHARED_SUFFIXES
)
_PROTECTED_NAMES = {
    "lm_head.weight",
    "draft_head.weight",
    "dspark.draft_head.weight",
    "dspark.lm_head.weight",
    "model.embed_tokens.weight",
}


def is_wide_dense_projection(name: str, entry: dict) -> bool:
    """Return True only for wide dense projections; experts, heads, bottlenecks, and raw tensors stay protected."""
    if "raw_off" in entry or name in _PROTECTED_NAMES or name.startswith("dspark."):
        return False
    if ".experts." in name or "experts" in entry:
        return False
    if len(entry.get("shape") or ()) != 2:
        return False
    if name.endswith(_WIDE_ATTN_SUFFIXES):
        return True
    if (".shared_experts." in name or ".shared_expert." in name) and name.endswith(_SHARED_SUFFIXES):
        return True
    if ".shared_expert" not in name and name.endswith(_DENSE_MLP_SUFFIXES):
        return True
    return False


def _selective_quant_recorded(prev_quant: str | None, dst_fmt: str) -> str:
    if not prev_quant:
        return f"selective_{dst_fmt}"
    if re.search(r"(^|__)dense_hq8[^_]*(?:_g\d+)?", prev_quant):
        return re.sub(r"dense_hq8(?:_g\d+)?", f"dense_selective_{dst_fmt}", prev_quant, count=1)
    return f"selective_{dst_fmt}__{prev_quant}"


def _resolve_inputs(src: Path, out: Path) -> tuple[Path, Path, Path | None, bool]:
    src_p = Path(src).expanduser().resolve()
    if not src_p.exists():
        raise FileNotFoundError(f"missing source pack or directory: {src_p}")
    if src_p.is_dir():
        src_dir, src_qwp = src_p, src_p / "model.qwp"
    else:
        src_dir, src_qwp = src_p.parent, src_p
    if not src_qwp.is_file():
        raise FileNotFoundError(f"missing source QWP1 pack: {src_qwp}")

    out_p = Path(out).expanduser().absolute()
    file_mode = out_p.suffix == ".qwp"
    if file_mode:
        if out_p.resolve() == src_qwp:
            raise ValueError(f"--out cannot overwrite --src in-place: {out_p}")
        if out_p.exists() and out_p.is_dir():
            raise IsADirectoryError(f"--out .qwp path is a directory: {out_p}")
        return src_dir, src_qwp, out_p, True
    if out_p.resolve() == src_dir and src_qwp == src_dir / "model.qwp":
        raise ValueError(f"--out directory cannot equal --src directory: {out_p}")
    if out_p.exists():
        raise FileExistsError(f"output directory already exists: {out_p}")
    return src_dir, src_qwp, None, False


def _pack_selective_file(
    src_qwp: Path, out_qwp: Path, *,
    wide_spec: tuple[int, int, bool, bool] | None, dst_fmt: str,
    flat_root: bool | None, device: str, compare: bool,
) -> dict:
    with maybe_force_cpu_quant(device):
        quant_dev = "cpu" if device == "cpu" else ("mps" if device == "mps" else None)
        src_hdr, _ = read_header(src_qwp)
        src_tensors = src_hdr["tensors"]
        src_prov = dict(src_hdr.get("provenance") or {})

        raw_names: list[str] = []
        copy_refs: dict[str, dict] = {}
        requant_plan: list[tuple[str, tuple[int, int, bool, bool]]] = []
        preserved_quant_names: list[str] = []

        for name, entry in src_tensors.items():
            if "raw_off" in entry:
                raw_names.append(name)
                continue
            if is_wide_dense_projection(name, entry) and wide_spec is not None:
                k_dim = int(entry["shape"][-1])
                eff = (wide_spec[0], resolve_group_size(k_dim, wide_spec[1]), wide_spec[2], wide_spec[3])
                if "hq4_off" in entry and _entry_quant_spec(entry) == eff:
                    copy_refs[name] = entry
                    preserved_quant_names.append(name)
                else:
                    requant_plan.append((name, eff))
            elif "hq4_off" in entry:
                copy_refs[name] = entry
                preserved_quant_names.append(name)
            elif "hq8_off" in entry:
                requant_plan.append((name, (8, 32, False, False)))
                preserved_quant_names.append(name)
            else:
                raise ValueError(f"unsupported tensor container in pack: {name}: {list(entry)}")

        prov = dict(src_prov)
        prov["quant_recorded"] = _selective_quant_recorded(src_prov.get("quant_recorded"), dst_fmt)
        prov["selective_wide_dense_quant"] = dst_fmt
        prov["hq8_reference"] = str(src_qwp)
        if flat_root is not None:
            prov["assets_dir"] = "." if flat_root else "assets"

        expected_shapes = {n: list(e["shape"]) for n, e in src_tensors.items()}
        with src_qwp.open("rb") as fp:
            def stream():
                for name in raw_names:
                    arr, kind = _read_full_raw(fp, src_tensors[name])
                    yield name, arr, kind
                total = len(requant_plan)
                for idx, (name, spec) in enumerate(requant_plan, 1):
                    if idx == 1 or idx % 32 == 0 or idx == total:
                        print(f"repacking wide dense {idx}/{total}: {name} -> {format_any_quant_spec(*spec)}", flush=True)
                    yield _requant_2d(fp, name, src_tensors[name], spec, device=quant_dev)

            pack_file(stream(), out_qwp, prov, copy_entries=(src_qwp, copy_refs))

        out_hdr, _ = audit_pack(out_qwp, expected_shapes=expected_shapes)
        digest = sha256(out_qwp)
        out_qwp.with_name(out_qwp.name + ".sha256").write_text(f"{digest}  {out_qwp.name}\n", encoding="utf-8")
        compared_bytes, skipped = (0, [])
        if compare:
            allow_changed = {
                n for n, _ in requant_plan
                if planes(out_hdr["tensors"][n]) != planes(src_tensors[n])
            }
            compared_bytes, skipped = compare_planes(out_qwp, src_qwp, allow_requantized=allow_changed)
        return {
            "hdr": out_hdr,
            "sha256": digest,
            "copied_quant_tensors": len(copy_refs),
            "copied_raw_tensors": len(raw_names),
            "requantized_wide_tensors": len(requant_plan),
            "preserved_quant_tensors": len(preserved_quant_names),
            "compared_bytes": compared_bytes,
            "skipped_diff_planes": skipped,
        }


def repack_selective_qwp(
    src: Path,
    out: Path,
    *,
    wide_dense_quant: str = "hq4_g32_m5",
    device: str = "auto",
    flat_root: bool | None = None,
    compare: bool = True,
) -> dict:
    """Selectively requantize wide dense projections in `model.qwp` while keeping bottlenecks/experts verbatim."""
    wide_spec = _parse_quant_or_keep(wide_dense_quant)
    dst_fmt = "keep" if wide_spec is None else format_any_quant_spec(*wide_spec)
    src_dir, src_qwp, out_file_target, file_mode = _resolve_inputs(src, out)

    if file_mode:
        assert out_file_target is not None
        out_file_target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{out_file_target.name}-", dir=out_file_target.parent) as tmpdir:
            tmp_qwp = Path(tmpdir) / out_file_target.name
            info = _pack_selective_file(
                src_qwp, tmp_qwp, wide_spec=wide_spec, dst_fmt=dst_fmt,
                flat_root=None, device=device, compare=compare,
            )
            tmp_sha = tmp_qwp.with_name(tmp_qwp.name + ".sha256")
            os.replace(tmp_qwp, out_file_target)
            os.replace(tmp_sha, out_file_target.with_name(out_file_target.name + ".sha256"))
        out_qwp = out_file_target
        out_str = str(out_file_target)
        assets_verified = 0
    else:
        out_dir = Path(out).expanduser().absolute()
        try:
            src_assets = resolve_assets_dir(src_dir)
        except FileNotFoundError:
            src_assets = None
        use_flat = (src_assets == src_dir) if (flat_root is None and src_assets is not None) else bool(flat_root)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{out_dir.name}-", dir=out_dir.parent) as tmpdir:
            staging = Path(tmpdir) / "model"
            staging.mkdir(parents=True, exist_ok=True)
            if src_assets is not None:
                _copy_assets_and_siblings(src_dir, staging, src_assets, flat_root=use_flat)
            staged_qwp = staging / "model.qwp"
            info = _pack_selective_file(
                src_qwp, staged_qwp, wide_spec=wide_spec, dst_fmt=dst_fmt,
                flat_root=(use_flat if src_assets is not None else None), device=device, compare=compare,
            )
            has_siblings = any((staging / fn).exists() for fn in ("dspark.qwp", "draft_head.qwp", "flashindexer.qwp"))
            can_verify_assets = src_assets is not None and (src_assets / "manifest.json").is_file() and not (use_flat and has_siblings)
            assets_verified = verify_assets(staging, info["hdr"]) if can_verify_assets else 0
            if out_dir.exists():
                raise FileExistsError(f"output appeared during repack: {out_dir}")
            staging.rename(out_dir)
        out_qwp = out_dir / "model.qwp"
        out_str = str(out_dir)

    src_bytes = src_qwp.stat().st_size
    out_bytes = out_qwp.stat().st_size
    return {
        "src": str(src_qwp),
        "out": out_str,
        "sha256": info["sha256"],
        "src_bytes": src_bytes,
        "out_bytes": out_bytes,
        "saved_bytes": src_bytes - out_bytes,
        "wide_dense_quant": dst_fmt,
        "copied_quant_tensors": info["copied_quant_tensors"],
        "copied_raw_tensors": info["copied_raw_tensors"],
        "requantized_wide_tensors": info["requantized_wide_tensors"],
        "assets_verified": assets_verified,
        "compared_bytes": info["compared_bytes"],
        "skipped_diff_planes": info["skipped_diff_planes"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path, help="Source model dir or model.qwp file")
    ap.add_argument("--out", required=True, type=Path, help="Destination model dir or .qwp file path")
    ap.add_argument("--wide-dense-quant", default="hq4_g32_m5", help="Target quant for wide dense projections (e.g. hq4_g32_m5, hq6_g32, keep)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--flat-root", action="store_true")
    ap.add_argument("--no-compare", action="store_true")
    a = ap.parse_args(argv)
    res = repack_selective_qwp(
        a.src,
        a.out,
        wide_dense_quant=a.wide_dense_quant,
        device=a.device,
        flat_root=(True if a.flat_root else None),
        compare=(not a.no_compare),
    )
    print(json.dumps(res, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
