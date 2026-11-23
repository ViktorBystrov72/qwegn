"""CPU-only unit tests for eval_ink2_quant_kl.py and inkling2_adapter.py (HF + MLX)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_ink2_quant_kl import (  # noqa: E402
    STAGE1_SPECS,
    _LayerWeightProvider,
    eval_per_tensor_quant_fidelity,
    forward_inkling2_48_visits,
    main as eval_main,
    make_synthetic_inkling2_checkpoint,
    run_eval,
)
from inkling2_adapter import (  # noqa: E402
    DSPARK_QUANT_CHOICES,
    E2M1_LUT,
    E4M3FN_LUT,
    RECIPES,
    Inkling2Source,
    dequant_bf16,
    dequant_mxfp8,
    dequant_nvfp4,
    inkling2_geo_v2,
    inkling2_pack_identity,
    main as adapter_main,
    open_inkling2_shards,
    pack_inkling2_model,
    parse_inkling2_config,
    reload_nvfp4_proj_f32,
    reload_qwp_tensor,
    verify_16k_alignment,
)
from score_teacher_logits import score as score_kl  # noqa: E402
from verify import audit_pack, read_header, verify_assets  # noqa: E402


def _scalar_e4m3fn(b: int) -> float:
    sign = -1.0 if (b & 0x80) else 1.0
    exp, mant = (b >> 3) & 0x0F, b & 0x07
    if exp == 15 and mant == 7:
        return float("nan")
    if exp == 0:
        return sign * (2.0 ** -6) * (mant / 8.0)
    return sign * (2.0 ** (exp - 7)) * (1.0 + mant / 8.0)


def _scalar_ue8m0(b: int) -> float:
    if b == 255:
        return float("nan")
    return 2.0 ** (-127 if b == 0 else (b - 127))


class TestInkling2AdapterAndOracle(unittest.TestCase):
    def test_e4m3fn_lut_matches_torch_float8(self):
        torch_vals = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32).numpy()
        for b in range(256):
            if b in (0x7F, 0xFF):
                self.assertTrue(np.isnan(E4M3FN_LUT[b]) and np.isnan(torch_vals[b]))
            else:
                self.assertEqual(float(E4M3FN_LUT[b]), float(torch_vals[b]), msg=f"byte {b}")

    def test_nvfp4_and_mxfp8_dequant_both_hf_and_mlx_layouts(self):
        rng = np.random.default_rng(99)
        valid_fp8 = np.array([0x00, 0x01, 0x38, 0x3C, 0x40, 0x44, 0xB8, 0xBC, 0x7E, 0xFE], dtype=np.uint8)
        valid_e8m0 = np.array([0, 125, 126, 127, 128, 130], dtype=np.uint8)

        e_cnt, rows, cols = 2, 4, 32
        w_u32 = rng.integers(0, 2**32, size=(e_cnt, rows, cols // 8), dtype=np.uint32)
        w_u8 = w_u32.view(np.uint8)
        sc_u8 = rng.choice(valid_fp8, size=(e_cnt, rows, cols // 16)).astype(np.uint8)
        s2 = np.array([0.25, -0.125], dtype=np.float32)

        ref_nv = np.empty((e_cnt, rows, cols), dtype=np.float32)
        for e in range(e_cnt):
            for r in range(rows):
                for g in range(cols // 16):
                    scale = np.float32(_scalar_e4m3fn(int(sc_u8[e, r, g]))) * np.float32(s2[e])
                    for wi in range(2):
                        word = int(w_u32[e, r, g * 2 + wi])
                        for j in range(8):
                            ref_nv[e, r, g * 16 + wi * 8 + j] = np.float32(E2M1_LUT[(word >> (j * 4)) & 0xF]) * scale

        # MLX u32 and HF u8+F8_E4M3FN must match scalar reference bit-for-bit
        np.testing.assert_allclose(dequant_nvfp4(w_u32, sc_u8, s2, as_torch=False), ref_nv, rtol=1e-6, atol=0.0)
        np.testing.assert_allclose(
            dequant_nvfp4(torch.from_numpy(w_u8), torch.from_numpy(sc_u8).view(torch.float8_e4m3fn), s2, as_torch=False),
            ref_nv,
            rtol=1e-6,
            atol=0.0,
        )
        np.testing.assert_allclose(dequant_nvfp4(w_u32[0], sc_u8[0], s2[0], as_torch=False), ref_nv[0], rtol=1e-6, atol=0.0)

        bad_sc = sc_u8.copy()
        bad_sc[0, 0, 0] = 0x7F
        with self.assertRaises(ValueError):
            dequant_nvfp4(w_u8, bad_sc, s2, as_torch=False)

        m_rows, m_cols = 3, 64
        mw_u8 = rng.choice(valid_fp8, size=(e_cnt, m_rows, m_cols)).astype(np.uint8)
        mw_u32 = mw_u8.view("<u4")
        msc_u8 = rng.choice(valid_e8m0, size=(e_cnt, m_rows, m_cols // 32)).astype(np.uint8)
        ref_mx = np.empty((e_cnt, m_rows, m_cols), dtype=np.float32)
        for e in range(e_cnt):
            for r in range(m_rows):
                for g in range(m_cols // 32):
                    scale = np.float32(_scalar_ue8m0(int(msc_u8[e, r, g])))
                    for j in range(32):
                        ref_mx[e, r, g * 32 + j] = np.float32(_scalar_e4m3fn(int(mw_u8[e, r, g * 32 + j]))) * scale

        # HF float8_e4m3fn and MLX uint32 must both match scalar reference
        np.testing.assert_allclose(dequant_mxfp8(torch.from_numpy(mw_u8).view(torch.float8_e4m3fn), msc_u8, as_torch=False), ref_mx, rtol=1e-6, atol=0.0)
        np.testing.assert_allclose(dequant_mxfp8(torch.from_numpy(mw_u32.copy()), msc_u8, as_torch=False), ref_mx, rtol=1e-6, atol=0.0)

        bad_msc = msc_u8.copy()
        bad_msc[1, 0, 0] = 0xFF
        with self.assertRaises(ValueError):
            dequant_mxfp8(mw_u8, bad_msc, as_torch=False)

    def test_hf_vs_mlx_exact_forward_equivalence_and_vision_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hf_dir = make_synthetic_inkling2_checkpoint(root / "hf", layout="hf", seed=1337)
            mlx_dir = make_synthetic_inkling2_checkpoint(root / "mlx", layout="mlx", seed=1337)
            hf_cfg = json.loads((hf_dir / "config.json").read_text())
            mlx_cfg = json.loads((mlx_dir / "config.json").read_text())
            self.assertEqual(inkling2_geo_v2(hf_cfg), inkling2_geo_v2(mlx_cfg))

            hf_r, _ = open_inkling2_shards(hf_dir)
            mlx_r, _ = open_inkling2_shards(mlx_dir)
            try:
                hf_spec = parse_inkling2_config(hf_cfg, source_dir=hf_dir, reader=hf_r)
                mlx_spec = parse_inkling2_config(mlx_cfg, source_dir=mlx_dir, reader=mlx_r)
                self.assertEqual(hf_spec.coda_start, mlx_spec.coda_start)
                self.assertEqual(hf_spec.execution_plan, mlx_spec.execution_plan)
                self.assertEqual(len(hf_spec.execution_plan), 48)

                hf_src, mlx_src = Inkling2Source(hf_r, hf_spec), Inkling2Source(mlx_r, mlx_spec)
                self.assertEqual(hf_src.layout, "hf")
                self.assertEqual(mlx_src.layout, "mlx")
                self.assertTrue(torch.equal(hf_src.get_global("message_queries"), mlx_src.get_global("message_queries")))
                for layer, proj in ((0, "gate_proj"), (0, "up_proj"), (0, "down_proj"), (20, "gate_proj"), (20, "up_proj"), (20, "down_proj")):
                    self.assertTrue(
                        torch.equal(hf_src.load_expert_proj_f32(layer, proj), mlx_src.load_expert_proj_f32(layer, proj)),
                        f"expert mismatch L{layer} {proj}",
                    )
                hf_vis = {k: v for k, v in hf_src.iter_vision_canonical()}
                mlx_vis = {k: v for k, v in mlx_src.iter_vision_canonical()}
                self.assertEqual(set(hf_vis.keys()), {"model.visual.embeddings.patch_embedding.weight", "model.visual.proj_out.weight"})
                self.assertEqual(set(hf_vis.keys()), set(mlx_vis.keys()))
                for k in hf_vis:
                    self.assertTrue(torch.equal(hf_vis[k], mlx_vis[k]))

                toks = torch.arange(0, 64, dtype=torch.int64) % hf_spec.unpadded_vocab_size
                hf_logits = forward_inkling2_48_visits(hf_spec, _LayerWeightProvider(hf_src, None), toks).numpy()
                mlx_logits = forward_inkling2_48_visits(mlx_spec, _LayerWeightProvider(mlx_src, None), toks).numpy()
                np.testing.assert_allclose(hf_logits, mlx_logits, rtol=0, atol=0)
            finally:
                hf_r.close()
                mlx_r.close()

    def test_48_visit_oracle_hf_teacher_vs_mlx_candidate_and_qwegn(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hf_dir = make_synthetic_inkling2_checkpoint(root / "hf_teacher", layout="hf", seed=2025)
            mlx_dir = make_synthetic_inkling2_checkpoint(root / "mlx_cand", layout="mlx", seed=2025, jitter_mlx_quant=True)

            hf_r, _ = open_inkling2_shards(hf_dir)
            try:
                hf_cfg = json.loads((hf_dir / "config.json").read_text())
                hf_spec = parse_inkling2_config(hf_cfg, source_dir=hf_dir, reader=hf_r)
                hf_src = Inkling2Source(hf_r, hf_spec)
                s1 = eval_per_tensor_quant_fidelity(hf_src, device="cpu")
                for tname, rows in s1["tensors"].items():
                    self.assertEqual(set(rows.keys()), set(STAGE1_SPECS))
                    for qspec, m in rows.items():
                        self.assertGreater(m["cosine"], 0.90, f"{tname}/{qspec}")
                    self.assertLess(rows["hq8_g32"]["rel_l2"], rows["hq6_g32"]["rel_l2"])
                    self.assertLess(rows["hq6_g32"]["rel_l2"], rows["hq4_g32_m5"]["rel_l2"])
                    self.assertLess(rows["shvq4_g64"]["rel_l2"], rows["shvq3_g64"]["rel_l2"])
            finally:
                hf_r.close()

            cap_dir, rep_path = root / "captures", root / "report.json"
            report = run_eval(hf_dir, mlx_candidate_dir=mlx_dir, dry_run=False, num_tokens=64, out_dir=cap_dir, device="cpu")
            self.assertEqual(report["teacher_layout"], "hf")
            self.assertEqual(report["visits"], 48)
            self.assertEqual(set(report["candidates"].keys()), {"mlx_candidate", *RECIPES.keys()})
            self.assertTrue((cap_dir / "teacher_logits.npy").is_file())
            self.assertTrue((cap_dir / "mlx_candidate_logits.npy").is_file())
            self.assertTrue((cap_dir / "contexts.npy").is_file())
            for cname, cr in report["candidates"].items():
                self.assertTrue((cap_dir / f"{cname}_logits.npy").is_file())
                self.assertGreaterEqual(cr["mean_kl"], 0.0)
                self.assertGreaterEqual(cr["p95_kl"], 0.0)
                self.assertGreaterEqual(cr["max_kl"], cr["mean_kl"])
                self.assertGreaterEqual(cr["top1_agreement"], 0.0)
                self.assertLessEqual(cr["top1_agreement"], 1.0)
                self.assertGreater(cr["estimated_qwp_gib"], 0.0)
                self.assertGreater(cr["active_dram_gb_per_tok"], 0.0)
            self.assertGreater(report["candidates"]["mlx_candidate"]["mean_kl"], 0.0)

            rc = eval_main(["--dry-run", "--num-tokens", "64", "--skip-stage1", "--candidates", "cand2_shvq4_uniform", "--output", str(rep_path)])
            self.assertEqual(rc, 0)
            parsed = json.loads(rep_path.read_text())
            self.assertEqual(set(parsed["candidates"].keys()), {"mlx_candidate", "cand2_shvq4_uniform"})

    def test_pack_hf_and_mlx_recipes_verify_16k_alignment_and_reload_parity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hf_ckpt = make_synthetic_inkling2_checkpoint(root / "synth_hf", layout="hf", seed=1337)
            mlx_ckpt = make_synthetic_inkling2_checkpoint(root / "synth_mlx", layout="mlx", seed=1337)
            cfg = json.loads((mlx_ckpt / "config.json").read_text())
            want_identity = inkling2_pack_identity(cfg)
            self.assertEqual(want_identity["model_id"], "inkling2_loop")
            self.assertTrue(inkling2_geo_v2(cfg).startswith("qwg-geo-v2:model_type=inkling2_loop_text|"))

            reader, _ = open_inkling2_shards(mlx_ckpt)
            try:
                spec = parse_inkling2_config(cfg, source_dir=mlx_ckpt, reader=reader)
                src = Inkling2Source(reader, spec)
                pfx = src.out_pfx
                mdl = f"{pfx}model."
                ref_embed = dequant_bf16(src.get_global("embed_tokens.weight"), as_torch=False)
                ref_q0 = dequant_bf16(src.get_layer(0, "self_attn.q_proj.weight"), as_torch=False)
                ref_ctrl0 = dequant_bf16(src.get_layer(6, "controllers.attn.down_weight"), as_torch=False)
                ref_head = dequant_bf16(src.get_global("lm_head.weight"), as_torch=False)
                ref_nv_g0 = src.load_expert_proj_f32(0, "gate_proj", as_torch=False)
                ref_nv_u0 = src.load_expert_proj_f32(0, "up_proj", as_torch=False)
                ref_nv_d0 = src.load_expert_proj_f32(0, "down_proj", as_torch=False)
                ref_mx_d20 = src.load_expert_proj_f32(20, "down_proj", as_torch=False)
                ref_vis = dequant_bf16(dict(src.iter_vision_canonical())["model.visual.embeddings.patch_embedding.weight"], as_torch=False)
            finally:
                reader.close()

            # Pack cand1, cand2, cand5 from HF + all 5 recipes across MLX + HF
            jobs = [
                (hf_ckpt, "hf", "cand1_hq8_shvq4", DSPARK_QUANT_CHOICES[0]),
                (mlx_ckpt, "mlx", "cand2_shvq4_uniform", DSPARK_QUANT_CHOICES[1]),
                (mlx_ckpt, "mlx", "cand3_maxwell_shvq3d4", DSPARK_QUANT_CHOICES[2]),
                (mlx_ckpt, "mlx", "cand4_hq4_scalar", DSPARK_QUANT_CHOICES[3]),
                (hf_ckpt, "hf", "cand5_hybrid_lossless_moe", DSPARK_QUANT_CHOICES[1]),
                (mlx_ckpt, "mlx", "cand5_hybrid_lossless_moe", DSPARK_QUANT_CHOICES[0]),
            ]
            for idx, (src_ckpt, exp_layout, rname, ds_q) in enumerate(jobs):
                out_dir = root / f"pack_{idx}_{exp_layout}_{rname}"
                summary = pack_inkling2_model(src_ckpt, out_dir, recipe_name=rname, quant_dspark=ds_q, include_vision=True, device="cpu")
                self.assertEqual(summary["source_layout"], exp_layout)
                self.assertEqual(summary["recipe"], rname)
                self.assertEqual(summary["quant_dspark"], ds_q)

                qwp_path, ds_path = out_dir / "model.qwp", out_dir / "dspark.qwp"
                self.assertEqual(verify_16k_alignment(qwp_path), [])
                self.assertEqual(verify_16k_alignment(ds_path), [])
                hdr, counts = audit_pack(qwp_path)
                self.assertGreater(sum(counts.values()), 0)
                self.assertEqual(hdr["provenance"]["model_id"], want_identity["model_id"])
                self.assertEqual(hdr["provenance"]["geometry_digest"], want_identity["geometry_digest"])
                self.assertGreater(verify_assets(out_dir, hdr), 0)

                ds_hdr, _ = audit_pack(ds_path)
                self.assertEqual(ds_hdr["provenance"]["quant_recorded"], ds_q)
                self.assertEqual((out_dir / "dspark.qwp.sha256").read_text().split()[0], hashlib.sha256(ds_path.read_bytes()).hexdigest())

                vres = subprocess.run([sys.executable, str(SCRIPT_DIR / "verify.py"), str(out_dir)], capture_output=True, text=True, check=False)
                self.assertEqual(vres.returncode, 0, vres.stderr + vres.stdout)
                self.assertEqual(json.loads(vres.stdout)["status"], "PASS")

                with qwp_path.open("rb") as f:
                    tensors = hdr["tensors"]
                    np.testing.assert_allclose(reload_qwp_tensor(f, tensors["model.visual.embeddings.patch_embedding.weight"]), ref_vis, rtol=0, atol=0)
                    np.testing.assert_allclose(reload_qwp_tensor(f, tensors[f"{mdl}embed_tokens.weight"]), ref_embed, rtol=0, atol=0)
                    q0_hat = reload_qwp_tensor(f, tensors[f"{mdl}layers.0.self_attn.q_proj.weight"])
                    c0_hat = reload_qwp_tensor(f, tensors[f"{mdl}layers.6.controllers.attn.down_weight"])
                    hd_hat = reload_qwp_tensor(f, tensors[f"{pfx}lm_head.weight"])
                    self.assertGreater(float(np.dot(q0_hat.ravel(), ref_q0.ravel()) / (np.linalg.norm(q0_hat) * np.linalg.norm(ref_q0))), 0.99)
                    self.assertGreater(float(np.dot(c0_hat.ravel(), ref_ctrl0.ravel()) / (np.linalg.norm(c0_hat) * np.linalg.norm(ref_ctrl0))), 0.99)
                    self.assertGreater(float(np.dot(hd_hat.ravel(), ref_head.ravel()) / (np.linalg.norm(hd_hat) * np.linalg.norm(ref_head))), 0.99)

                    if rname == "cand5_hybrid_lossless_moe":
                        sw0 = f"{mdl}layers.0.mlp.switch_mlp."
                        np.testing.assert_array_equal(reload_nvfp4_proj_f32(f, tensors, sw0, "gate_proj"), ref_nv_g0)
                        np.testing.assert_array_equal(reload_nvfp4_proj_f32(f, tensors, sw0, "up_proj"), ref_nv_u0)
                        np.testing.assert_array_equal(reload_nvfp4_proj_f32(f, tensors, sw0, "down_proj"), ref_nv_d0)
                    else:
                        gu0_hat = reload_qwp_tensor(f, tensors[f"{mdl}layers.0.mlp.experts.gate_up_proj"])
                        ref_gu0 = np.concatenate([ref_nv_g0, ref_nv_u0], axis=1)
                        self.assertGreater(float(np.dot(gu0_hat.ravel(), ref_gu0.ravel()) / (np.linalg.norm(gu0_hat) * np.linalg.norm(ref_gu0))), 0.97)
                        dn0_hat = reload_qwp_tensor(f, tensors[f"{mdl}layers.0.mlp.experts.down_proj"])
                        self.assertGreater(float(np.dot(dn0_hat.ravel(), ref_nv_d0.ravel()) / (np.linalg.norm(dn0_hat) * np.linalg.norm(ref_nv_d0))), 0.98)

                    dn20_hat = reload_qwp_tensor(f, tensors[f"{mdl}layers.20.mlp.experts.down_proj"])
                    self.assertGreater(float(np.dot(dn20_hat.ravel(), ref_mx_d20.ravel()) / (np.linalg.norm(dn20_hat) * np.linalg.norm(ref_mx_d20))), 0.98)

                with ds_path.open("rb") as df:
                    dt = ds_hdr["tensors"]
                    w1 = reload_qwp_tensor(df, dt["markov_head.markov_w1.weight"])
                    self.assertEqual(w1.shape, (spec.unpadded_vocab_size, spec.loop_gate_rank))
                    self.assertIn("raw_off", dt["markov_head.markov_w1.weight"])
                    if ds_q != "bf16":
                        self.assertIn("hq4_off", dt["layers.0.self_attn.q_proj.weight"])
                        self.assertIn("raw_off", dt["markov_head.hidden_out.weight"])
                        self.assertIn("raw_off", dt["markov_head.rnn_proj.weight"])

            no_vis_dir = root / "pack_novis"
            rc = adapter_main(["--model", str(hf_ckpt), "--out", str(no_vis_dir), "--recipe", "cand2_shvq4_uniform", "--no-vision", "--device", "cpu"])
            self.assertEqual(rc, 0)
            nv_hdr, _ = read_header(no_vis_dir / "model.qwp")
            self.assertFalse(any(k.startswith(("model.visual.", "vision_tower.")) for k in nv_hdr["tensors"]))


if __name__ == "__main__":
    unittest.main()
