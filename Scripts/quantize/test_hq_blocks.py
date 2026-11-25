"""Unit tests for HQ{4,6,8}_G{32,64,128,256} block quantizer, mixed pack, and verifier."""

from pathlib import Path
import tempfile
import unittest
import numpy as np

from hq4 import quant_hq4_blocks, dequant_hq4_g32_m5
from hq_blocks import (
    VALID_BITS,
    VALID_GROUPS,
    parse_quant_spec,
    format_quant_spec,
    resolve_group_size,
    hq_block_bytes,
    quant_hq_blocks,
    dequant_hq_blocks,
    dequant_mxfp4_blockwise,
    dequant_with_scale_inv,
    unpacked_safetensors_shape,
    validate_source_quant_config,
)
from pack import pack_file, quantize_model
from verify import audit_pack, read_header


class HQBlocksTests(unittest.TestCase):
    def test_parse_and_format_quant_spec(self) -> None:
        self.assertEqual(parse_quant_spec("hq4"), (4, 32, True))
        self.assertEqual(parse_quant_spec("hq4_g32_m5"), (4, 32, True))
        self.assertEqual(parse_quant_spec("hq4_g32"), (4, 32, False))
        self.assertEqual(parse_quant_spec("hq4_g64"), (4, 64, False))
        self.assertEqual(parse_quant_spec("hq6", default_group=128), (6, 128, False))
        self.assertEqual(parse_quant_spec("hq8_g32"), (8, 32, False))
        self.assertEqual(format_quant_spec(4, 32, is_m5=True), "hq4_g32_m5")
        self.assertEqual(format_quant_spec(4, 32, is_m5=False), "hq4_g32")
        self.assertEqual(format_quant_spec(8, 32), "hq8_g32")
        self.assertEqual(format_quant_spec(6, 128), "hq6_g128")

    def test_incumbent_hq4_g32_m5_is_byte_identical(self) -> None:
        rng = np.random.default_rng(42)
        w = rng.standard_normal((16, 256), dtype=np.float32) * 0.03
        raw_inc = quant_hq4_blocks(w)
        raw_new, eff_g = quant_hq_blocks(w, 4, 32, is_m5=True)
        self.assertEqual(eff_g, 32)
        self.assertEqual(raw_inc.tobytes(), raw_new.tobytes())
        w_inc = dequant_hq4_g32_m5(raw_inc)
        w_new = dequant_hq_blocks(raw_new, 4, 32, is_m5=True)
        np.testing.assert_allclose(w_inc, w_new, rtol=0, atol=0)

    def test_all_twelve_pairs_monotonic_fidelity(self) -> None:
        rng = np.random.default_rng(123)
        w = rng.standard_normal((32, 512), dtype=np.float32) * 0.02
        mse_table: dict[tuple[int, int], float] = {}
        for bits in VALID_BITS:
            for group in VALID_GROUPS:
                raw, eff_g = quant_hq_blocks(w, bits, group, is_m5=False)
                self.assertEqual(eff_g, group)
                expected_bytes = 32 * (512 // group) * hq_block_bytes(bits, group)
                self.assertEqual(raw.size, expected_bytes)
                w_hat = dequant_hq_blocks(raw, bits, group, is_m5=False)
                mse = float(np.mean((w - w_hat) ** 2))
                mse_table[(bits, group)] = mse
        for group in VALID_GROUPS:
            self.assertLess(mse_table[(3, group)], mse_table[(2, group)] * 0.45)
            self.assertLess(mse_table[(4, group)], mse_table[(3, group)] * 0.35)
            self.assertLess(mse_table[(6, group)], mse_table[(4, group)] * 0.1)
            self.assertLess(mse_table[(8, group)], mse_table[(6, group)] * 0.1)
        for bits in VALID_BITS:
            self.assertLessEqual(mse_table[(bits, 32)], mse_table[(bits, 64)])
            self.assertLessEqual(mse_table[(bits, 64)], mse_table[(bits, 128)])
            self.assertLessEqual(mse_table[(bits, 128)], mse_table[(bits, 256)])

    def test_gcd_group_size_clamping(self) -> None:
        self.assertEqual(resolve_group_size(320, 256), 64)
        self.assertEqual(resolve_group_size(320, 128), 64)
        self.assertEqual(resolve_group_size(320, 64), 64)
        self.assertEqual(resolve_group_size(320, 32), 32)

    def test_mixed_pack_and_verify_roundtrip(self) -> None:
        rng = np.random.default_rng(7)
        dense_f32 = rng.standard_normal((16, 320), dtype=np.float32) * 0.02
        exp_f32 = rng.standard_normal((4, 8, 256), dtype=np.float32) * 0.02
        norm_f32 = np.ones((256,), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            qwp_path = Path(tmp) / "model.qwp"
            pack_file(
                [
                    ("model.layers.0.dense.weight", dense_f32, "hq8_g128"),
                    ("model.layers.0.mlp.experts.down_proj", exp_f32, "hq4_g64"),
                    ("model.layers.0.input_layernorm.weight", norm_f32, True),
                ],
                qwp_path,
                provenance={
                    "source": "unit-test",
                    "quant": "dense_hq8_g128__expert_hq4_g64",
                    "geometry_hash": "0" * 64,
                },
            )
            ok, msg = audit_pack(
                qwp_path,
                expected_shapes={
                    "model.layers.0.dense.weight": [16, 320],
                    "model.layers.0.mlp.experts.down_proj": [4, 8, 256],
                    "model.layers.0.input_layernorm.weight": [256],
                },
            )
            self.assertTrue(ok, msg)
            hdr, _ = read_header(qwp_path)
            self.assertEqual(hdr["tensors"]["model.layers.0.dense.weight"]["hq"], {"bits": 8, "group": 64})
            self.assertEqual(hdr["tensors"]["model.layers.0.mlp.experts.down_proj"]["hq"], {"bits": 4, "group": 64})

    def test_dequant_mxfp4_blockwise_exact_and_nan_rejection(self) -> None:
        import torch
        e2m1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        # Pack all 16 nibble combinations across 16 bytes -> 32 elements on row 0, and negative bytes (-128..127) on row 1.
        row0_bytes = [((2 * j + 1) << 4) | (2 * j) for j in range(8)] * 2
        row1_bytes = [0x8F, 0xF8, 0x70, 0x07, 0xAB, 0xCD, 0xEF, 0x12] * 2
        packed_u8 = torch.tensor([row0_bytes, row1_bytes], dtype=torch.uint8)
        packed_i8 = packed_u8.view(torch.int8)
        scales_u8 = torch.tensor([[127], [128]], dtype=torch.uint8)  # 2^0=1.0, 2^1=2.0
        scales_e8 = scales_u8.view(torch.float8_e8m0fnu)

        out = dequant_mxfp4_blockwise(packed_i8, scales_e8)
        self.assertEqual(tuple(out.shape), (2, 32))
        self.assertEqual(out.dtype, torch.float32)
        for j in range(16):
            self.assertEqual(float(out[0, j]), e2m1[j])
        self.assertEqual(float(out[1, 0]), e2m1[0xF] * 2.0)
        self.assertEqual(float(out[1, 1]), e2m1[0x8] * 2.0)

        out_dispatch = dequant_with_scale_inv(packed_i8.unsqueeze(0), scales_e8.unsqueeze(0))
        self.assertEqual(tuple(out_dispatch.shape), (1, 2, 32))
        self.assertTrue(torch.equal(out_dispatch[0], out))

        nan_scales = torch.tensor([[127], [255]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
        with self.assertRaises(ValueError):
            dequant_mxfp4_blockwise(packed_i8, nan_scales)

    def test_validate_source_quant_config_and_unpacked_shapes(self) -> None:
        self.assertEqual(validate_source_quant_config({}), 128)
        self.assertEqual(validate_source_quant_config({"quantization_config": {"quant_method": "fp8", "weight_block_size": [64, 64]}}), 64)
        self.assertEqual(validate_source_quant_config({"text_config": {"quantization_config": {"quant_method": "mxfp4"}}}), 128)
        for bad in ({}, {"quant_method": "gptq"}, {"quant_method": "awq"}):
            with self.assertRaisesRegex(ValueError, "already quantized"):
                validate_source_quant_config({"quantization_config": bad})
        self.assertEqual(unpacked_safetensors_shape([8, 32], "I8", [8, 2], "F8_E8M0"), [8, 64])
        self.assertEqual(unpacked_safetensors_shape([2, 4, 16], "U8", [2, 4, 1], "U8"), [2, 4, 32])
        self.assertEqual(unpacked_safetensors_shape([8, 64], "F8_E4M3", [1, 1], "F32"), [8, 64])
        self.assertEqual(unpacked_safetensors_shape([64], "BF16", None, None), [64])

    def test_generic_pack_quantize_model_pure_mxfp4_two_shards(self) -> None:
        import json
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "hf_mxfp4"
            dst = Path(tmp) / "out_mxfp4"
            src.mkdir()
            (src / "tokenizer.json").write_text("{}")
            (src / "config.json").write_text(json.dumps({
                "model_type": "generic_test_moe",
                "quantization_config": {"quant_method": "mxfp4"},
            }))
            q_w = torch.randint(-128, 128, (8, 32), dtype=torch.int8)
            q_s = torch.full((8, 2), 126, dtype=torch.uint8).view(torch.float8_e8m0fnu)
            exp_w = torch.randint(-128, 128, (2, 4, 32), dtype=torch.int8)
            exp_s = torch.full((2, 4, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
            norm = torch.ones((64,), dtype=torch.bfloat16)
            save_file({
                "model.layers.0.self_attn.q_proj.weight": q_w,
                "model.layers.0.mlp.experts.gate_up_proj.weight": exp_w,
                "model.layers.0.input_layernorm.weight": norm,
            }, str(src / "model-00001-of-00002.safetensors"))
            save_file({
                "model.layers.0.self_attn.q_proj.weight_scale_inv": q_s,
                "model.layers.0.mlp.experts.gate_up_proj.weight_scale_inv": exp_s,
            }, str(src / "model-00002-of-00002.safetensors"))
            quantize_model(src, dst)
            hdr, _ = read_header(dst / "model.qwp")
            self.assertEqual(hdr["tensors"]["model.layers.0.self_attn.q_proj.weight"]["shape"], [8, 64])
            self.assertEqual(hdr["tensors"]["model.layers.0.mlp.experts.gate_up_proj.weight"]["shape"], [2, 4, 64])
            ok, msg = audit_pack(dst / "model.qwp", expected_shapes={
                "model.layers.0.self_attn.q_proj.weight": [8, 64],
                "model.layers.0.mlp.experts.gate_up_proj.weight": [2, 4, 64],
                "model.layers.0.input_layernorm.weight": [64],
            })
            self.assertTrue(ok, msg)


if __name__ == "__main__":
    unittest.main()
