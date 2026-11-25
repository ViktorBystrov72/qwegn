"""LiquidAI LFM2 / LFM3 / LFM3-VL QWP1 packer adapter."""
from __future__ import annotations

import json
from pathlib import Path

from decision_adapter import pack_decision_or_lfm_model


def quantize_lfm_model(
    source: Path,
    destination: Path,
    flat_root: bool = False,
    quant: str = "hq4_g32_m5",
    group_size: int | None = None,
    dense_quant: str | None = None,
    expert_quant: str | None = None,
    expert_down_quant: str | None = None,
) -> Path:
    """Packs LFM2/LFM3/LFM3-VL checkpoints preserving ShortConv & decision heads."""
    _ = expert_quant, expert_down_quant
    cfg = json.loads((Path(source) / "config.json").read_text(encoding="utf-8"))
    tc = cfg["text_config"] if "text_config" in cfg else cfg
    mt = tc.get("model_type") or cfg.get("model_type") or ""
    is_decision = (
        ("decision_config" in cfg)
        or ("decision_config" in tc)
        or (cfg.get("model_type") == "lfm3_vl")
        or (mt in ("lfm3", "lfm3_vl"))
        or any(k.startswith("d1_") for k in cfg)
        or any("D1Model" in str(v) for v in (cfg.get("auto_map") or {}).values())
    )
    return pack_decision_or_lfm_model(
        source, destination, flat_root=flat_root, quant=quant, group_size=group_size,
        dense_quant=dense_quant, squeeze_conv1d=True, is_decision_hint=is_decision,
    )
