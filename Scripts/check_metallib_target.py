#!/usr/bin/env python3
"""Verify compiled metallibs target macOS >= 26.2 and enforce the NAX guard."""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METALLIBS = (
    ROOT / "Sources" / "QWMetal" / "kernel.metallib",
    ROOT / "Sources" / "QWMetal" / "kernel_std.metallib",
)
MIN_TARGET = (26, 2)


def find_metal_tool(name: str) -> list[str]:
    probe = subprocess.run(["xcrun", "-sdk", "macosx", name, "--version"],
                           capture_output=True, text=True)
    if probe.returncode == 0:
        return ["xcrun", "-sdk", "macosx", name]
    cryptex = sorted(Path("/private/var/run/com.apple.security.cryptexd/mnt").glob(
        f"com.apple.MobileAsset.MetalToolchain-*/Metal.xctoolchain/usr/bin/{name}"))
    if cryptex:
        if name == "metal":
            sdk = subprocess.run(["xcrun", "--sdk", "macosx", "--show-sdk-path"],
                                 capture_output=True, text=True, check=True).stdout.strip()
            return [str(cryptex[-1]), "-isysroot", sdk]
        return [str(cryptex[-1])]
    return ["xcrun", "-sdk", "macosx", name]


def check_metallib_platform(readobj: list[str], path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"missing metallib: {path}")
    r = subprocess.run([*readobj, "--headers", str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"metal-readobj failed for {path}:\n{r.stderr[:1000]}")
    maj_m = re.search(r"PlatformMajor:\s*(\d+)", r.stdout)
    min_m = re.search(r"PlatformMinor:\s*(\d+)", r.stdout)
    if not maj_m or not min_m:
        raise RuntimeError(f"could not parse PlatformMajor/Minor from {path}")
    target = (int(maj_m.group(1)), int(min_m.group(1)))
    if target < MIN_TARGET:
        raise RuntimeError(
            f"{path} targets macOS {target[0]}.{target[1]}, must be >= {MIN_TARGET[0]}.{MIN_TARGET[1]}")
    return f"{path.name}: macOS {target[0]}.{target[1]}"


def check_negative_compile_guard(metal_cc: list[str]) -> None:
    src = '#include "mlx/backend/metal/kernels/steel/gemm/nax.h"\nkernel void k() {}\n'
    with tempfile.TemporaryDirectory(prefix="qwg-nax-guard-") as td:
        tmp = Path(td)
        m_path = tmp / "guard.metal"
        a_path = tmp / "guard.air"
        m_path.write_text(src, encoding="utf-8")
        r = subprocess.run(
            [*metal_cc, "-std=metal4.0", "-mmacosx-version-min=26.0",
             "-ISources/QWMetal", "-ISources/QWMetal/include", "-c", str(m_path), "-o", str(a_path)],
            cwd=ROOT, capture_output=True, text=True,
        )
        if r.returncode == 0:
            raise RuntimeError("NAX header compiled under -mmacosx-version-min=26.0 without failing")
        if "require -mmacosx-version-min=26.2" not in r.stderr:
            raise RuntimeError(f"unexpected NAX guard error under 26.0:\n{r.stderr[:1000]}")


def main() -> int:
    readobj = find_metal_tool("metal-readobj")
    metal_cc = find_metal_tool("metal")
    summaries = [check_metallib_platform(readobj, lib) for lib in METALLIBS]
    check_negative_compile_guard(metal_cc)
    print("metallib target: PASS (" + ", ".join(summaries) + "; 26.0 guard enforced)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
