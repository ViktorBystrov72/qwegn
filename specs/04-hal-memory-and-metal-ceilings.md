# 04 — HAL Concurrency, Memory Discipline & Apple Silicon Ceilings

> Normative specification for `QWHal`, `ColdExpertStreamer`, UMA memory budgeting, and AGX hardware limits.

## 1. Lock-Free Post-Freeze PSO Cache (`QWHal`)
`QWHal` manages Metal `MTLComputePipelineState` (PSO) lookup with zero lock contention on the hot decode/prefill critical path (`Sources/Foundation/QWHAL/QWHal.swift`, verified by `QWHalDoorArenaTests.testFrozenPipelineHitsTakeNoLock`):
- Mutable pipeline compilation and table registration (`MutablePSOState`) are guarded by `OSAllocatedUnfairLock`.
- Publishing/freezing snapshots builds an immutable `FrozenPSOSnapshot` and swaps `frozenSnapshotPtr` (`Synchronization.Atomic<UnsafeMutableRawPointer?>`) with release/acquire ordering.
- Prior snapshots remain retained in `retiredSnapshots` until `QWHal.deinit` releases the active atomic pointer, guaranteeing safe lock-free reader dereference throughout process lifetime.

## 2. AGX 32 KiB Constant-Cache Ceiling
- Apple Silicon AGX spills `constant` address-space tables to device DRAM whenever a single PSO references `> 32 KiB` of constant arrays.
- Separate codebook families (e.g. `qw_shvq_tables.metalh`, `qw_shvq_sub2_tables.metalh`, `qw_shvq_routed_lut.metalh`, `qw_shvq_sub2_routed_lut.metalh`) MUST stay isolated in family-specific `.metalh` headers and PSOs so each PSO stays `<= 32 KiB` constant LUT footprint.

## 3. Unified Memory Discipline & Zero Steady-State Allocations
- **Host budget (`128 GiB` UMA):** `QWResidencyAdmission` / `QWAdmission` verifies `planned <= available` before loading weights or expanding context/KV arenas; refusals exit non-zero before allocation.
- **Zero steady-state allocations:** After warmup/pre-allocation (`QWArena`), decode and prefill steady state executes with `0` heap or Metal buffer allocations per step across dense, MoE, GDN, HCA, and MLA/QSA paths.
- **Sequential GPU ownership:** Only one heavyweight GPU inference process executes at a time; benchmarks (`Scripts/paired_perf.py`) run strictly sequentially with quiet-GPU preflight checks.

## 4. Safe `ColdExpertStreamer` Synchronization
- Cold-expert NVMe streaming (`Sources/Runtime/QWEngine/Driver/ColdExpertStreamer.swift`) eliminates unchecked `!` unwraps across `syncGPUSlotMap`, `verifyPromoted`, and `prepareSpanDynamic`, validating slot bounds and staging buffers explicitly before GPU promotion.
- `QWHal` command queue allocates `maxCommandBufferCount: 256` to accommodate ~2 CBs per MoE layer during cold-expert decode without command-buffer starvation.

## 5. Bare-Metal vs Virtualized CI Gating
- `QWDeviceCaps.isMetal4TensorCapable` (`MTLDevice.supportsFamily(.apple10)`) separates real Apple Silicon Metal 4 hardware from virtualized CI GPUs (`Apple Paravirtual device`, 16 GiB RAM, no `~/.qwg/models`).
- Production code and Metal kernels never weaken or branch around virtualized runner limits; instead, bare-metal Metal 4 tests (NAX cooperative tensors, 1024-thread GDN recurrent kernels, 32 KiB-TG / multi-TG atomic e-means selectors, full hybrid/speculative GPU driver E2E) gate in `Tests/` via `XCTSkipUnless(hal.caps.isMetal4TensorCapable, ...)`, and host-dependent checks (`ru_maxrss`, `~/.qwg/models`, Debug CPU timing budgets) gate via `GITHUB_ACTIONS` `XCTSkipIf`/`XCTSkipUnless`.

## 6. Multi-Session Continuous Batching, Recurrent Reset & Drafter Slot Sizing
- **Unified batch schedulers (`QWBatchScheduler`, `FNGPUBatchScheduler`):** Both standard `QWEngineDriver` (`DenseModule` + `HybridMoEModule`, via `QWBatchScheduler` / `DriverBatchDecode`) and `QWFlashNext` (`FNGPUEngine`, via `FNGPUBatchScheduler`) support continuous multi-slot batching (`maxSlots = 4`) alongside universal KV page-pressure eviction and RoPE boundary step-down across `>= 16K` contexts.
- **Unconditional recurrent state hygiene (`FNGPUSession`, `FNGPUBatchScheduler`):** Exiting batched waves back to single-slot mode (`cleanupExtraSlotsBeforeSingleMode`) and resetting slot-0 (`wipeSessionState(overwriteTokens:)`) unconditionally clear GDN recurrent states, convolution rings, PLE histories, and parities even when `highWaterPos == 0` or `used == 0`, preventing inter-request state leakage across concurrent HTTP bursts. After `syncSlotFromEngine(syncSlot)`, `slotStates[syncSlot].highWaterPos` is synchronized to `engine.highWaterPos`.
- **Drafter slot alignment & prefix cache pinning (`QWDFlashDrafter`, `PrefixCache`, `FNPrefixCache`):** `QWDFlashDrafter(..., sessionSlotCount:)` matches `DriverResidency.gdnSlotCount()` (`4` on `>= 32 GiB` UMA, `2` on `16 GiB`). Normal per-turn prefix caching stores entries with `isSharedBase: false` so LRU headroom reclaim works without evicting mounted `.qwcache` shared bases (`isSharedBase: true`).

## 7. Async Speculative Commit Retention & Cache-Line-Disjoint MTP Embeds
- **Async `Commit(r-1) -> Draft(r) -> Verify(r)` pipeline (`MTPVerification`, `QWHal.touchKeepAlive`):** Speculative verification retains ephemeral staging and input buffers used by asynchronous `pendingCommitCB` command buffers in `QWHal` via `touchKeepAlive(_:)` until GPU completion, overlapping prior-round `Commit` execution with next-round drafting and GPU MTP embedding inside a single `waitUntilCompleted()` boundary per speculative round.
- **Cache-line-disjoint `QWMTPEmbeddingTable.tokens` sub-buffers:** Across multi-draft `encodeRows` and `encodeBatchedMTPPairSeed` invocations, CPU-written `tokenIDs` slots and GPU-read embedding outputs are partitioned on 128-byte cache-line boundaries inside `QWMTPEmbeddingTable`, eliminating false sharing during pipelined BF16 GPU MTP seed construction.
