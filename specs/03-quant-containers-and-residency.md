# 03 — Quant Containers, Plane Sizing, Checked POSIX I/O & Residency

> Normative specification for pack containers (`QWQuant`, `QWLoad`), `PlaneLayout`, `QWPOSIX`, and runtime residency flags.

## 1. Supported Weight Containers & Codebook Isolation
- **Containers:** `HQ*` (group-quantized Hadamard), `AQ*` (additive vector quantization), `SHVQ{1,15,2,3,4}_G{32,64,128,256}` (Signed Hadamard e-means Vector Quantization, including 1.5-bit `SHVQ15` sub-2-bit with 0-ULP CPU/Metal parity via `QWSHVQBlock`), `VQ*` (e.g., `VQ2` PLE tables), and `QNGRAM` / HyperPLE streams (`FNNgramMLXRows`, `FNNgramShards`, `FNNgramBinRows`).
- **Basis marker verification (`QWHal`):** Every compiled `.metallib` carries an explicit basis marker (`qwg_basis_marker` for `.rotated`, `qwg_basis_marker_standard` for `.standard`). `QWHal.init` throws `QWHalError.metallibBasisMismatch` if pack weight basis and metallib basis disagree.

## 2. `PlaneLayout` Mixed-Plane Byte Resolution
Mixed raw BF16/FP16 and quantized weight tensors (including `DFlash` / `DSpine` drafters) resolve byte counts exclusively via `PlaneLayout.resolve(entry)` (`Sources/Foundation/QWLoad/PlaneLayout.swift`):
1. Check `entry.rawOff != nil` **before** checking quantized plane offsets (`hq4Off`).
2. Compute `.raw` bytes via `shape.reduce(1, *) * rawElemBytes`; compute `.hq4` bytes from `experts * eHq4` (3D MoE banks) or `n * (k / hqGroup) * hqBlockBytes`.
3. `QWCheckpointTensorLoader.mapCachedMoESlab` enforces explicit `sysconf(_SC_PAGESIZE)` modulus alignment checks before `mmap`/zero-copy projection.

## 3. Zero-Copy Buffer Lifetimes & Checked POSIX I/O
- **Zero-copy retention (`PlaneIO.mapZeroCopyRange` & `DispatchData`):** Both `MTLBuffer(bytesNoCopy:)` and `DispatchData` mappings retain the backing `QWQuant.Pack` via `nonisolated(unsafe)` lifetime extension across deallocation (`withExtendedLifetime(...) {}`) without `@Sendable` capture warnings, preventing premature `munmap` under async consumers.
- **Checked `pread` loops (`Sources/Foundation/QWCore/QWPOSIX.swift`):** Direct `Darwin.pread` reads across `QWDispatchIO`, `FNNgramMLXRows`, `FNNgramShards`, and `FNNgramBinRows` route through `QWPOSIX.preadUpTo` and `QWPOSIX.preadExact`:
  - Reject negative `fd`, `count`, or `offset` with `QWPOSIXError.invalidArgument`.
  - Check `offset + Int64(totalRead)` with `addingReportingOverflow`.
  - Retry `EINTR` and partial short reads until `count` or EOF; `preadExact` throws `QWPOSIXError.shortRead` on premature EOF.

## 4. Runtime Residency Toggles & Hermetic Mode
All containers and model modules honor runtime residency controls without hidden overrides:
- `--load-vision` / `--no-load-vision`: Loads or omits the ViT encoder; image requests when disabled fail cleanly without lazy surprise allocations.
- `--stream-ngrams` / `--no-stream-ngrams`: Streams HyperPLE / `QNGRAM` tables from NVMe via a bounded cache rather than pinning full tables in GPU/RAM.
- `--no-profile`: Bypasses `~/.qwg/settings.json` and per-model profile overrides for deterministic, hermetic CLI/bench execution (enforced by `Scripts/paired_perf.py`).
