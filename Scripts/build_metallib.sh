#!/usr/bin/env bash
# Build the engine metallib (Sources/QWMetal/*.metal → kernel.metallib).
# QWHAL discovers Sources/QWMetal/kernel.metallib by cwd walk; CI checks
# freshness by rebuilding into a temp and comparing kernel symbol sets.
#
# TWO libraries are built from the same sources, one per activation basis:
#
#   kernel.metallib      ROTATED  — the engine's own weight format, where the
#                                   weights carry the rotation and the
#                                   activations are rotated to match.
#   kernel_std.metallib  STANDARD — an MLX affine checkpoint's planes, which
#                                   carry no rotation, so the activations are
#                                   left alone.
#
# The basis is a BUILD-time property, not a runtime one: the rotation is a
# chain of `simd_shuffle_xor` calls, and a runtime branch in front of them
# puts the shuffle in a divergent region — measured at -14.5% prefill
# throughput on the 8B dense model. A compile-time constant lets the branch
# vanish, so each library's rotation code is exactly what it should be.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-Sources/QWMetal/kernel.metallib}"
STD_OUT="${2:-Sources/QWMetal/kernel_std.metallib}"
METAL_MIN_MACOS="${QW_METAL_MIN_MACOS:-26.2}"

if xcrun -sdk macosx metal --version >/dev/null 2>&1; then
  METAL_CC=(xcrun -sdk macosx metal)
  METALLIB_LD=(xcrun -sdk macosx metallib)
  METAL_READOBJ=(xcrun -sdk macosx metal-readobj)
else
  CRYPTEX_BIN="$(ls -d /private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-*/Metal.xctoolchain/usr/bin 2>/dev/null | tail -n 1)"
  SDK_PATH="$(xcrun --sdk macosx --show-sdk-path)"
  METAL_CC=("$CRYPTEX_BIN/metal" -isysroot "$SDK_PATH")
  METALLIB_LD=("$CRYPTEX_BIN/metallib")
  METAL_READOBJ=("$CRYPTEX_BIN/metal-readobj")
fi

JOBS="${QW_METAL_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 8)}"

verify_metallib_target() {
  local lib="$1" maj="" min=""
  read -r maj min < <("${METAL_READOBJ[@]}" --headers "$lib" | awk '/PlatformMajor:/ {maj=$2} /PlatformMinor:/ {min=$2} END {print maj, min}')
  if [ -z "$maj" ] || [ -z "$min" ] || ! [[ "$maj" =~ ^[0-9]+$ ]] || ! [[ "$min" =~ ^[0-9]+$ ]]; then
    echo "FAIL: could not parse macOS platform target from $lib" >&2
    return 1
  fi
  if [ "$maj" -lt 26 ] || { [ "$maj" -eq 26 ] && [ "$min" -lt 2 ]; }; then
    echo "FAIL: $lib targets macOS ${maj}.${min}, but Apple MPP NAX matmul2d requires >= 26.2" >&2
    return 1
  fi
  echo "verified $lib target: macOS ${maj}.${min}"
}

# build_variant <output> [extra compiler flags...]
build_variant() {
  local out="$1"; shift
  local tmp; tmp="$(mktemp -d)"
  local airs=() pids=() errs=() sources=()
  local idx=0 m a err fail=0
  while IFS= read -r -d '' m; do
    a="$(printf '%s/%04d_%s.air' "$tmp" "$idx" "$(basename "${m%.metal}")")"
    err="$a.err"
    "${METAL_CC[@]}" -std=metal4.0 -mmacosx-version-min="$METAL_MIN_MACOS" -O2 "$@" \
      -ISources/QWMetal -ISources/QWMetal/include -ISources/QWMetal/gdn \
      -c "$m" -o "$a" 2>"$err" &
    pids+=($!)
    errs+=("$err")
    sources+=("$m")
    airs+=("$a")
    idx=$((idx + 1))
    if [ "${#pids[@]}" -ge "$JOBS" ]; then
      if ! wait "${pids[0]}"; then fail=1; fi
      pids=("${pids[@]:1}")
    fi
  done < <(find Sources/QWMetal -name '*.metal' -print0 | sort -z)
  for p in "${pids[@]}"; do
    if ! wait "$p"; then fail=1; fi
  done
  if [ "$fail" -ne 0 ]; then
    for ((i=0; i<${#sources[@]}; i++)); do
      if [ -s "${errs[$i]}" ]; then
        echo "metal compile failed for ${sources[$i]}:" >&2
        cat "${errs[$i]}" >&2
      fi
    done
    rm -rf "$tmp"
    return 1
  fi
  "${METALLIB_LD[@]}" "${airs[@]}" -o "$out"
  rm -rf "$tmp"
  verify_metallib_target "$out"
}

build_variant "$OUT"
build_variant "$STD_OUT" -DQW_BASIS_STANDARD=1

for d in .build/release .build/debug; do
  if [ -d "$d" ]; then
    cp "$OUT" "$d/kernel.metallib"
    cp "$STD_OUT" "$d/kernel_std.metallib"
  fi
done
echo "built $OUT"
echo "built $STD_OUT"
