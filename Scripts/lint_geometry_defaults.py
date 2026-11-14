#!/usr/bin/env python3
"""The geometry-default gate.

Enforces three invariants across `Sources`:
1. No `QWModelGeometry` or `QWModelDescriptor` parameter carries a default value.
2. No `?? ...qwen38_27b` nil-coalescing fallback exists.
3. No `.qwen38_27b` reference appears outside the 5 allowlisted files.
"""
import os
import re
import sys

PARAM_RE = re.compile(r"\b[A-Za-z_]\w*\s*:\s*QWModel(?:Geometry|Descriptor)\s*=\s*[^,)\n]+")
NIL_COALESCE_RE = re.compile(r"\?\?\s*(?:QWModel(?:Geometry|Descriptor)\s*)?\.qwen38_27b\b")
REF_RE = re.compile(r"\.qwen38_27b\b")

ALLOWED_QWEN38_27B_FILES = {
    "Sources/Foundation/QWCore/QWModelDescriptor.swift",
    "Sources/Foundation/QWCore/QWModelGeometry.swift",
    "Sources/Foundation/QWQuant/QWPack.swift",
    "Sources/Apps/QWCli/QWCli.swift",
    "Sources/Apps/QWBenchKernels/main.swift",
}

problems = 0
roots = sys.argv[1:] or ["Sources"]
for root in roots:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".swift"):
                continue
            path = os.path.normpath(os.path.join(dirpath, fn))
            text = open(path, encoding="utf-8", errors="replace").read()
            for m in PARAM_RE.finditer(text):
                if re.search(r"\b(?:let|var)\s+$", text[:m.start()]):
                    continue
                line = text.count("\n", 0, m.start()) + 1
                snippet = " ".join(m.group(0).split())
                print(f"{path}:{line}: model geometry/descriptor parameter carries a "
                      f"default — {snippet}")
                problems += 1
            for m in NIL_COALESCE_RE.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                snippet = " ".join(m.group(0).split())
                print(f"{path}:{line}: forbidden nil-coalescing .qwen38_27b fallback — {snippet}")
                problems += 1
            if path not in ALLOWED_QWEN38_27B_FILES:
                for m in REF_RE.finditer(text):
                    line = text.count("\n", 0, m.start()) + 1
                    print(f"{path}:{line}: unexpected .qwen38_27b reference outside allowlist")
                    problems += 1

if problems:
    print(f"geometry: {problems} violation(s)")
    sys.exit(1)
print("geometry: 0 violation(s)")
