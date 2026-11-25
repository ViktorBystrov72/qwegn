"""Repack local dspark.qwp DSpark draft packs in-place of slow re-download from HF.
Default compresses dense projections to hq4_g32_m5 while preserving expert_down,
expert_gate_up, markov_w1/w2, confidence_head, norms, routers, and mhc verbatim,
plus optionally bundling rotated draft_head.weight from model.qwp.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from ernie55_adapter import _hq_meta, normalize_quant_spec
from hq4 import fwht32_ortho
from hq_blocks import (
    dequant_hq8_g32_m5_rot,
    dequant_hq_blocks_rot,
    quant_hq_blocks,
    resolve_group_size,
)
from inkling2_adapter import maybe_force_cpu_quant
from pack import pack_file, sha256
from shvq import (
    any_block_bytes,
    dequant_shvq_blocks_rot,
    format_any_quant_spec,
    parse_any_quant_spec,
    quant_shvq_blocks,
)
from verify import audit_pack, compare_planes, hash_region, planes, read_header

_EXPERT_DOWN_ITEM_RE = re.compile(r"\.mlp\.experts\.\d+\.down_proj\.weight$")
_RAW_NP_DTYPES = {"bf16": np.uint16, "f16": np.float16, "f32": np.float32}
_HEAD_CANDIDATES = ("draft_head.weight", "lm_head.weight", "dspark.draft_head.weight", "dspark.lm_head.weight")


def _resolve_pack_path(path: Path, default_name: str) -> Path:
    p = Path(path).expanduser()
    return (p / default_name) if p.is_dir() else p


def _parse_quant_or_keep(spec: str | None) -> tuple[int, int, bool, bool] | None:
    if spec is None or spec.strip().lower() == "keep":
        return None
    return parse_any_quant_spec(normalize_quant_spec(spec))


def _entry_quant_spec(entry: dict) -> tuple[int, int, bool, bool]:
    if "hq8_off" in entry:
        return 8, 32, True, False
    hq = entry.get("hq")
    if hq is None:
        return 4, 32, True, False
    return int(hq["bits"]), int(hq["group"]), False, (hq.get("vq") == "shvq1")


def _entry_fmt_label(entry: dict) -> str:
    if "raw_off" in entry:
        return str(entry.get("kind", "bf16"))
    return format_any_quant_spec(*_entry_quant_spec(entry))


def _classify_dspark_tensor(name: str, entry: dict) -> str:
    if name in _HEAD_CANDIDATES:
        return "draft_head"
    if name in ("dspark.markov_w2.weight", "markov_w2.weight") and "raw_off" not in entry:
        return "markov_w2"
    if "raw_off" in entry:
        return "raw"
    if ".experts." in name or "experts" in entry:
        if name.endswith(".mlp.experts.down_proj") or _EXPERT_DOWN_ITEM_RE.search(name):
            return "expert_down"
        return "expert_gate_up"
    if "hq4_off" in entry or "hq8_off" in entry:
        return "dense"
    raise ValueError(f"unsupported tensor container in dspark pack: {name}: {list(entry)}")


def _read_raw_rows(fp, entry: dict, total_rows: int, k: int, r0: int, r1: int) -> np.ndarray:
    kind = str(entry.get("kind") or "bf16")
    if kind not in _RAW_NP_DTYPES:
        raise ValueError(f"unsupported raw tensor kind {kind!r} for shape {entry['shape']}")
    dt = _RAW_NP_DTYPES[kind]
    row_bytes = k * np.dtype(dt).itemsize
    fp.seek(int(entry["raw_off"]) + r0 * row_bytes)
    arr = np.frombuffer(fp.read((r1 - r0) * row_bytes), dtype=dt).reshape(r1 - r0, k)
    if kind == "bf16":
        import torch
        return torch.from_numpy(arr.copy()).view(torch.bfloat16).to(torch.float32).numpy()
    return arr.astype(np.float32, copy=True)


def _dequant_rows(fp, entry: dict, total_rows: int, k: int, r0: int, r1: int, *, rotate_out: bool) -> np.ndarray:
    rn = r1 - r0
    if "raw_off" in entry:
        w = _read_raw_rows(fp, entry, total_rows, k, r0, r1)
        return w if rotate_out else fwht32_ortho(w.reshape(-1, 32)).reshape(rn, k)
    if "hq8_off" in entry:
        fp.seek(int(entry["hq8_off"]) + r0 * k)
        qs = np.frombuffer(fp.read(rn * k), dtype=np.int8).reshape(rn, k).copy()
        sc_per_row = k // 32
        fp.seek(int(entry["hq8_s_off"]) + r0 * sc_per_row * 2)
        sc = np.frombuffer(fp.read(rn * sc_per_row * 2), dtype=np.float16).reshape(rn, sc_per_row).copy()
        w_rot = dequant_hq8_g32_m5_rot(qs, sc)
    elif "hq4_off" in entry:
        sb, sg, sm5, sshvq = _entry_quant_spec(entry)
        nb = k // sg
        bbytes = any_block_bytes(sb, sg, is_m5=sm5, is_shvq=sshvq)
        row_bytes = nb * bbytes
        fp.seek(int(entry["hq4_off"]) + r0 * row_bytes)
        blk = np.frombuffer(fp.read(rn * row_bytes), dtype=np.uint8).reshape(rn, nb, bbytes).copy()
        w_rot = (
            dequant_shvq_blocks_rot(blk, bits=sb, group=sg)
            if sshvq
            else dequant_hq_blocks_rot(blk, bits=sb, group=sg, is_m5=sm5)
        )
    else:
        raise ValueError(f"unsupported source layout for {list(entry)}")
    if not rotate_out:
        return w_rot
    return fwht32_ortho(w_rot.reshape(-1, 32)).reshape(rn, k)


def _requant_2d(
    fp, name: str, entry: dict, dst_spec: tuple[int, int, bool, bool], *,
    out_name: str | None = None, device: str | None = None, chunk_rows: int = 4096,
):
    shape = [int(d) for d in entry["shape"]]
    if len(shape) != 2:
        raise ValueError(f"expected 2D shape for {name}, got {shape}")
    n, k = shape
    db, dg, dm5, dshvq = dst_spec
    eff_dg = resolve_group_size(k, dg)
    dst_meta = _hq_meta(db, eff_dg, dm5, dshvq)
    dst_bbytes = any_block_bytes(db, eff_dg, is_m5=(dst_meta is None), is_shvq=dshvq)
    out_blocks = np.empty((n, k // eff_dg, dst_bbytes), dtype=np.uint8)
    for r0 in range(0, n, chunk_rows):
        r1 = min(r0 + chunk_rows, n)
        w_rot = _dequant_rows(fp, entry, n, k, r0, r1, rotate_out=False)
        q_chunk, _ = (
            quant_shvq_blocks(w_rot, bits=db, group=eff_dg, rotate=False, device=device)
            if dshvq
            else quant_hq_blocks(w_rot, bits=db, group=eff_dg, is_m5=dm5, rotate=False)
        )
        out_blocks[r0:r1] = q_chunk
    return (out_name or name), out_blocks, ("prepacked_hq4", [n, k], dst_meta)


def _requant_3d_expert(fp, name: str, entry: dict, dst_spec: tuple[int, int, bool, bool], *, chunk_rows: int = 4096):
    shape = [int(d) for d in entry["shape"]]
    if len(shape) != 3:
        raise ValueError(f"expected 3D expert shape for {name}, got {shape}")
    e_cnt, n, k = shape
    total_rows = e_cnt * n
    w_unrot = np.empty((e_cnt, n, k), dtype=np.float32)
    flat = w_unrot.reshape(total_rows, k)
    for r0 in range(0, total_rows, chunk_rows):
        r1 = min(r0 + chunk_rows, total_rows)
        flat[r0:r1] = _dequant_rows(fp, entry, total_rows, k, r0, r1, rotate_out=True)
    return name, w_unrot, dst_spec


def _read_full_raw(fp, entry: dict):
    import torch
    shape = [int(d) for d in entry["shape"]]
    kind = str(entry.get("kind") or "bf16")
    if kind not in _RAW_NP_DTYPES:
        raise ValueError(f"unsupported raw kind {kind!r}")
    dt = _RAW_NP_DTYPES[kind]
    n_bytes = math.prod(shape) * np.dtype(dt).itemsize
    fp.seek(int(entry["raw_off"]))
    arr = np.frombuffer(fp.read(n_bytes), dtype=dt).reshape(shape).copy()
    if kind == "bf16":
        return torch.from_numpy(arr).view(torch.bfloat16), "bf16"
    return arr, kind


def _prepare_draft_head(dh_fp, src_name: str, entry: dict, quant_mode: str, *, device: str | None):
    import torch
    shape = [int(d) for d in entry["shape"]]
    if len(shape) != 2:
        raise ValueError(f"draft_head source {src_name} must be 2D, got {shape}")
    n, k = shape
    mode = quant_mode.strip().lower()
    if mode == "keep":
        if "raw_off" in entry:
            arr, kind = _read_full_raw(dh_fp, entry)
            return ("draft_head.weight", arr, kind), _entry_fmt_label(entry)
        if "hq8_off" in entry:
            dh_spec = (8, 32, False, False)
            return _requant_2d(dh_fp, src_name, entry, dh_spec, out_name="draft_head.weight", device=device), "hq8_g32"
        sb, sg, sm5, sshvq = _entry_quant_spec(entry)
        bbytes = any_block_bytes(sb, sg, is_m5=sm5, is_shvq=sshvq)
        nbytes = n * (k // sg) * bbytes
        dh_fp.seek(int(entry["hq4_off"]))
        blk = np.frombuffer(dh_fp.read(nbytes), dtype=np.uint8).reshape(n, k // sg, bbytes).copy()
        return ("draft_head.weight", blk, ("prepacked_hq4", [n, k], entry.get("hq"))), _entry_fmt_label(entry)
    if mode == "bf16":
        if "raw_off" in entry:
            arr, kind = _read_full_raw(dh_fp, entry)
            if kind == "bf16":
                return ("draft_head.weight", arr, "bf16"), "bf16"
            return ("draft_head.weight", torch.from_numpy(arr).to(torch.bfloat16), "bf16"), "bf16"
        w = np.empty((n, k), dtype=np.float32)
        for r0 in range(0, n, 4096):
            r1 = min(r0 + 4096, n)
            w[r0:r1] = _dequant_rows(dh_fp, entry, n, k, r0, r1, rotate_out=True)
        return ("draft_head.weight", torch.from_numpy(w).to(torch.bfloat16), "bf16"), "bf16"
    dh_spec = parse_any_quant_spec(normalize_quant_spec(quant_mode))
    eff_spec = (dh_spec[0], resolve_group_size(k, dh_spec[1]), dh_spec[2], dh_spec[3])
    return _requant_2d(dh_fp, src_name, entry, eff_spec, out_name="draft_head.weight", device=device), format_any_quant_spec(*eff_spec)


def _verify_against_source(out_qwp: Path, src_qwp: Path, *, allow_changed: set[str], draft_head_added: bool) -> tuple[int, list[str]]:
    out_hdr, _ = read_header(out_qwp)
    src_hdr, src_start = read_header(src_qwp)
    out_tensors, src_tensors = out_hdr["tensors"], src_hdr["tensors"]
    if (not draft_head_added) and set(out_tensors) == set(src_tensors):
        return compare_planes(out_qwp, src_qwp, allow_requantized=allow_changed)
    if not draft_head_added:
        raise ValueError("output tensor names differ unexpectedly from source")
    compared = 0
    skipped: list[str] = []
    with out_qwp.open("rb") as nf, src_qwp.open("rb") as of:
        for name, prev in src_tensors.items():
            if name in _HEAD_CANDIDATES:
                skipped.append(name)
                continue
            cur = out_tensors.get(name)
            if cur is None:
                raise ValueError(f"missing source tensor in output: {name}")
            if cur["shape"] != prev["shape"]:
                raise ValueError(f"{name}: shape changed {prev['shape']} -> {cur['shape']}")
            np_planes, op_planes = planes(cur), planes(prev)
            if name in allow_changed:
                if np_planes == op_planes:
                    raise ValueError(f"{name}: expected requantized plane schema change")
                skipped.append(name)
                continue
            if np_planes != op_planes:
                raise ValueError(f"{name}: unexpected plane schema change")
            for key, length, _ in np_planes:
                off = prev.get(key)
                if type(off) is not int or off < 12 or off + length > src_start:
                    raise ValueError(f"{name}: invalid source plane {key}")
                if hash_region(nf, cur[key], length) != hash_region(of, off, length):
                    raise ValueError(f"{name}: {key} bytes differ")
                compared += length
    return compared, sorted(skipped)


def repack_dspark_qwp(
    src_dspark: Path,
    out_dspark: Path,
    *,
    expert_down_quant: str = "keep",
    dense_quant: str = "hq4_g32_m5",
    expert_gate_up_quant: str = "keep",
    draft_head_from: Path | None = None,
    draft_head_quant: str = "shvq2_g64",
    device: str = "auto",
    compare: bool = True,
) -> dict:
    """Repack `dspark.qwp` into `out_dspark` with selective dense / expert_down quant + optional bundled draft_head."""
    src_qwp = _resolve_pack_path(src_dspark, "dspark.qwp").resolve()
    out_qwp = Path(out_dspark).expanduser().resolve()
    if not src_qwp.is_file():
        raise FileNotFoundError(f"missing source dspark pack: {src_qwp}")
    if out_qwp.is_dir():
        raise IsADirectoryError(f"--out must be a file path, got directory: {out_qwp}")
    if out_qwp == src_qwp:
        raise ValueError(f"--out cannot overwrite --src in-place: {out_qwp}")

    ed_spec = _parse_quant_or_keep(expert_down_quant)
    d_spec = _parse_quant_or_keep(dense_quant)
    eg_spec = _parse_quant_or_keep(expert_gate_up_quant)

    dh_path: Path | None = None
    dh_src_name: str | None = None
    dh_src_entry: dict | None = None
    if draft_head_from is not None:
        dh_path = _resolve_pack_path(draft_head_from, "model.qwp").resolve()
        if not dh_path.is_file():
            raise FileNotFoundError(f"missing --draft-head-from pack: {dh_path}")
        dh_hdr, _ = read_header(dh_path)
        for cand in _HEAD_CANDIDATES:
            if cand in dh_hdr["tensors"]:
                dh_src_name, dh_src_entry = cand, dh_hdr["tensors"][cand]
                break
        if dh_src_entry is None:
            raise ValueError(f"no draft/lm head tensor found in {dh_path}")

    with maybe_force_cpu_quant(device):
        quant_dev = "cpu" if device == "cpu" else ("mps" if device == "mps" else None)
        src_hdr, _ = read_header(src_qwp)
        src_tensors = src_hdr["tensors"]
        src_prov = dict(src_hdr.get("provenance") or {})

        copy_refs: dict[str, dict] = {}
        raw_names: list[str] = []
        requant_plan: list[tuple[str, str, tuple[int, int, bool, bool]]] = []
        skipped_head_names: set[str] = set()
        resolved_labels: dict[str, str] = {}

        for name, entry in src_tensors.items():
            cat = _classify_dspark_tensor(name, entry)
            if cat == "draft_head":
                if dh_src_entry is not None:
                    skipped_head_names.add(name)
                    continue
                if "raw_off" in entry:
                    raw_names.append(name)
                elif "hq4_off" in entry:
                    copy_refs[name] = entry
                else:
                    requant_plan.append((name, cat, (8, 32, False, False)))
                continue
            if cat == "raw":
                raw_names.append(name)
                continue
            if cat == "markov_w2":
                if "hq4_off" in entry:
                    copy_refs[name] = entry
                    resolved_labels["markov"] = _entry_fmt_label(entry)
                else:
                    requant_plan.append((name, cat, (8, 32, False, False)))
                    resolved_labels["markov"] = "hq8_g32"
                continue

            target = {"dense": d_spec, "expert_down": ed_spec, "expert_gate_up": eg_spec}[cat]
            k_dim = int(entry["shape"][-1])
            if target is None:
                if "hq4_off" in entry:
                    copy_refs[name] = entry
                    resolved_labels.setdefault(cat, _entry_fmt_label(entry))
                else:
                    requant_plan.append((name, cat, _entry_quant_spec(entry)))
                    resolved_labels.setdefault(cat, _entry_fmt_label(entry))
                continue

            eff_target = (target[0], resolve_group_size(k_dim, target[1]), target[2], target[3])
            if "hq4_off" in entry and _entry_quant_spec(entry) == eff_target:
                copy_refs[name] = entry
            else:
                requant_plan.append((name, cat, eff_target))
            resolved_labels.setdefault(cat, format_any_quant_spec(*target))

        dense_lbl = resolved_labels.get("dense", "keep")
        egu_lbl = resolved_labels.get("expert_gate_up", "keep")
        ed_lbl = resolved_labels.get("expert_down", "keep")
        mw2_lbl = resolved_labels.get("markov", src_prov.get("dspark_markov_w2_quant", "hq8_g32"))

        dh_item = None
        dh_lbl = None
        if dh_path is not None and dh_src_name is not None and dh_src_entry is not None:
            with dh_path.open("rb") as dh_fp:
                dh_item, dh_lbl = _prepare_draft_head(dh_fp, dh_src_name, dh_src_entry, draft_head_quant, device=quant_dev)

        prov = dict(src_prov)
        prov["dspark_reference"] = str(src_qwp)
        prov["dspark_dense_quant"] = dense_lbl
        prov["dspark_expert_quant"] = egu_lbl
        prov["dspark_expert_down_quant"] = ed_lbl
        prov["dspark_markov_w2_quant"] = mw2_lbl
        qrec = f"dspark_dense_{dense_lbl}__expert_{egu_lbl}__expdown_{ed_lbl}__markov_{mw2_lbl}"
        if dh_lbl is not None:
            prov["dspark_draft_head_quant"] = dh_lbl
            prov["dspark_draft_head_source"] = str(dh_path)
            qrec += f"__head_{dh_lbl}"
        prov["quant_recorded"] = qrec

        expected_shapes = {n: list(e["shape"]) for n, e in src_tensors.items() if n not in skipped_head_names}
        if dh_item is not None and dh_src_entry is not None:
            expected_shapes["draft_head.weight"] = [int(d) for d in dh_src_entry["shape"]]

        out_qwp.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{out_qwp.name}-", dir=out_qwp.parent) as tmpdir:
            tmp_qwp = Path(tmpdir) / out_qwp.name
            with src_qwp.open("rb") as fp:
                def stream():
                    for name in raw_names:
                        arr, kind = _read_full_raw(fp, src_tensors[name])
                        yield name, arr, kind
                    if dh_item is not None:
                        yield dh_item
                    total = len(requant_plan)
                    for idx, (name, cat, spec) in enumerate(requant_plan, 1):
                        entry = src_tensors[name]
                        fmt_s = format_any_quant_spec(*spec)
                        if idx == 1 or cat == "expert_down" or idx == total or idx % 8 == 0:
                            print(f"repacking dspark {idx}/{total} [{cat}]: {name} -> {fmt_s}", flush=True)
                        if "experts" in entry:
                            yield _requant_3d_expert(fp, name, entry, spec)
                            gc.collect()
                        else:
                            yield _requant_2d(fp, name, entry, spec, device=quant_dev)

                pack_file(stream(), tmp_qwp, prov, copy_entries=(src_qwp, copy_refs))

            out_hdr, _ = audit_pack(tmp_qwp, expected_shapes=expected_shapes)
            digest = sha256(tmp_qwp)
            compared_bytes, skipped = (0, [])
            if compare:
                allow_changed = {
                    n for n, _, _ in requant_plan
                    if planes(out_hdr["tensors"][n]) != planes(src_tensors[n])
                }
                if dh_item is not None and "draft_head.weight" in src_tensors:
                    if planes(out_hdr["tensors"]["draft_head.weight"]) != planes(src_tensors["draft_head.weight"]):
                        allow_changed.add("draft_head.weight")
                compared_bytes, skipped = _verify_against_source(
                    tmp_qwp, src_qwp, allow_changed=allow_changed, draft_head_added=(dh_item is not None),
                )
            sha_path = out_qwp.with_name(out_qwp.name + ".sha256")
            tmp_sha = Path(tmpdir) / sha_path.name
            tmp_sha.write_text(f"{digest}  {out_qwp.name}\n", encoding="utf-8")
            os.replace(tmp_qwp, out_qwp)
            os.replace(tmp_sha, sha_path)

    src_bytes = src_qwp.stat().st_size
    out_bytes = out_qwp.stat().st_size
    return {
        "src": str(src_qwp),
        "out": str(out_qwp),
        "sha256": digest,
        "src_bytes": src_bytes,
        "out_bytes": out_bytes,
        "saved_bytes": src_bytes - out_bytes,
        "dense_quant": dense_lbl,
        "expert_down_quant": ed_lbl,
        "expert_gate_up_quant": egu_lbl,
        "markov_w2_quant": mw2_lbl,
        "draft_head_quant": dh_lbl,
        "copied_tensors": len(copy_refs) + len(raw_names),
        "requantized_tensors": len(requant_plan),
        "compared_bytes": compared_bytes,
        "skipped_diff_planes": skipped,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path, help="Source dspark.qwp (or model dir containing dspark.qwp)")
    ap.add_argument("--out", required=True, type=Path, help="Destination repacked dspark.qwp file path")
    ap.add_argument("--expert-down-quant", default="keep", help="Target quant for routed experts down_proj (default keep)")
    ap.add_argument("--dense-quant", default="hq4_g32_m5", help="Target quant for dense projections (or keep)")
    ap.add_argument("--expert-gate-up-quant", default="keep", help="Target quant for routed experts gate_up_proj (default keep)")
    ap.add_argument("--draft-head-from", type=Path, default=None, help="Optional model.qwp/draft_head.qwp to bundle draft_head.weight")
    ap.add_argument("--draft-head-quant", default="shvq2_g64", help="Quant format for bundled draft_head.weight (quant spec, keep, or bf16)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--no-compare", action="store_true")
    a = ap.parse_args(argv)
    res = repack_dspark_qwp(
        a.src,
        a.out,
        expert_down_quant=a.expert_down_quant,
        dense_quant=a.dense_quant,
        expert_gate_up_quant=a.expert_gate_up_quant,
        draft_head_from=a.draft_head_from,
        draft_head_quant=a.draft_head_quant,
        device=a.device,
        compare=(not a.no_compare),
    )
    print(json.dumps(res, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
