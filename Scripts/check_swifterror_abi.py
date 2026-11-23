#!/usr/bin/env python3
"""Verify arm64 Mach-O binaries have no LLVM AArch64 swifterror (x21) tail-call clobbers."""
from __future__ import annotations

from collections import deque
import re
import subprocess
import sys

SYM_RE = re.compile(r"^([0-9a-fA-F]+)\s+<([^>]+)>:$")
INSN_RE = re.compile(r"^\s*([0-9a-fA-F]+):\s+[0-9a-fA-F]+\s+(\S+)\s*(.*)$")
STUB_RE = re.compile(r"^\s*0x([0-9a-fA-F]+)\s+\d+\s+(\S+)")
BRANCH_COND_RE = re.compile(r"^(?:b\.[a-z]+|cbn?z|tbn?z)$")
HEX_ADDR_RE = re.compile(r"^(0x[0-9a-fA-F]+)")
CALLEE_SAVED_MOV_SRC = re.compile(r"^x(19|20|22|23|24|25|26|27|28),\s*x21$")
CALLEE_SAVED_MOV_DST = re.compile(r"^x21,\s*x(19|20|22|23|24|25|26|27|28)$")
ZERO_X21_RE = re.compile(r"^[xw]21,\s*(?:[xw]zr|#0x0|#0)(?:\s*;.*)?$")
READ_X21_RE = re.compile(r"\bx21\b")
WRITE_ALU_OPS = frozenset({
    "mov", "add", "sub", "adr", "adrp", "orr", "and", "eor",
    "lsl", "lsr", "asr", "mul", "madd", "csel", "cset", "movz", "movn", "movk",
})
WRITE_MEM_OPS = frozenset({"ldr", "ldp", "ldur"})
SAFE_ERROR_BL = ("_swift_allocError", "_swift_willThrow", "_swift_unexpectedError", "_swift_errorRetain")
SKIP_TAIL_STUBS = ("_swift_willThrow", "_swift_unexpectedError", "___stack_chk_fail")
RULE_B_TAIL_STUBS = frozenset({
    "_memcpy", "_memmove", "_memset", "_bzero", "_free", "_objc_release", "_swift_release",
})


def load_stubs(path: str) -> dict[int, str]:
    out = subprocess.run(["otool", "-Iv", path], capture_output=True, text=True, check=True).stdout
    stubs: dict[int, str] = {}
    for line in out.splitlines():
        m = STUB_RE.match(line)
        if m:
            stubs[int(m.group(1), 16)] = m.group(2)
    return stubs


def branch_target(ops: str) -> int | None:
    m = HEX_ADDR_RE.search(ops.split(",")[-1].strip())
    return int(m.group(1), 16) if m else None


def find_frame_regs(insns: list[tuple[int, str, str]]) -> set[str]:
    regs = {"sp", "x29"}
    for _, op, ops in insns[:48]:
        if op == "mov" and re.match(r"^(x\d+),\s*sp$", ops):
            regs.add(ops.split(",")[0].strip())
        elif op in ("add", "sub") and re.match(r"^(x\d+),\s*(?:sp|x29),", ops):
            regs.add(ops.split(",")[0].strip())
    return regs


def is_frame_mem_op(ops: str, frame_regs: set[str]) -> bool:
    m = re.search(r"\[(x\d+|sp)\b", ops)
    return bool(m and m.group(1) in frame_regs)


def is_write_x21(op: str, ops: str) -> bool:
    if op in WRITE_ALU_OPS:
        dst = ops.split(",")[0].strip()
        if dst in ("x21", "w21"):
            return not (op == "mov" and ZERO_X21_RE.match(ops))
        return False
    if op in WRITE_MEM_OPS:
        return bool(READ_X21_RE.search(ops.split("[")[0]))
    return False


def scan_function(sym: str, insns: list[tuple[int, str, str]], stubs: dict[int, str]):
    n = len(insns)
    if n == 0:
        return None
    addr_to_idx = {addr: idx for idx, (addr, _, _) in enumerate(insns)}
    frame_regs = find_frame_regs(insns)

    saved_regs: set[str] = set()
    save_indices: set[int] = set()
    for i, (_, op, ops) in enumerate(insns):
        if op in ("stp", "str", "stur") and READ_X21_RE.search(ops.split("[")[0]):
            if is_frame_mem_op(ops, frame_regs):
                save_indices.add(i)
        elif op == "mov":
            m = CALLEE_SAVED_MOV_SRC.match(ops)
            if m:
                saved_regs.add(f"x{m.group(1)}")
                save_indices.add(i)

    def is_restore(op: str, ops: str) -> bool:
        if op in WRITE_MEM_OPS and READ_X21_RE.search(ops.split("[")[0]):
            return is_frame_mem_op(ops, frame_regs)
        if op == "mov":
            m = CALLEE_SAVED_MOV_DST.match(ops)
            return bool(m and f"x{m.group(1)}" in saved_regs)
        return False

    succ: list[list[int]] = [[] for _ in range(n)]
    ext_tail: dict[int, str] = {}
    is_ret: set[int] = set()
    prev_bl: list[str | None] = [None] * n
    cur_bl: str | None = None
    for i, (_, op, ops) in enumerate(insns):
        if op in ("ret", "retaa", "retab", "brk", "udf", "br"):
            if op.startswith("ret"):
                is_ret.add(i)
            cur_bl = None
            continue
        if op == "b":
            tgt = branch_target(ops)
            if tgt is not None and tgt in addr_to_idx:
                succ[i] = [addr_to_idx[tgt]]
            elif tgt is not None:
                ext_tail[i] = stubs.get(tgt, ops)
            cur_bl = None
            continue
        if BRANCH_COND_RE.match(op):
            tgt = branch_target(ops)
            if tgt is not None and tgt in addr_to_idx:
                succ[i].append(addr_to_idx[tgt])
            if i + 1 < n:
                succ[i].append(i + 1)
            continue
        if op in ("bl", "blr"):
            tgt = branch_target(ops)
            cur_bl = stubs.get(tgt, ops) if tgt is not None else ops
            if i + 1 < n:
                succ[i].append(i + 1)
            continue
        prev_bl[i] = cur_bl
        # Any write to x0 clears the direct ObjC return-value association, except copying x0 -> x21.
        if i + 1 < n:
            succ[i].append(i + 1)

    unsaved_nodes: set[int] = set()
    q: deque[int] = deque([0])
    while q:
        u = q.popleft()
        if u in unsaved_nodes:
            continue
        unsaved_nodes.add(u)
        if u in save_indices:
            continue
        for v in succ[u]:
            if v not in unsaved_nodes:
                q.append(v)

    for i in sorted(unsaved_nodes):
        if i in save_indices:
            continue
        addr, op, ops = insns[i]
        if not is_write_x21(op, ops):
            continue
        pbl = prev_bl[i] or ""
        if any(k in pbl for k in SAFE_ERROR_BL):
            continue
        from_objc = ("objc_msgSend" in pbl) and (op == "mov") and ops.startswith("x21, x0")
        bfs: deque[int] = deque(succ[i])
        seen: set[int] = {i}
        while bfs:
            u = bfs.popleft()
            if u in seen:
                continue
            seen.add(u)
            _, u_op, u_ops = insns[u]
            if is_write_x21(u_op, u_ops) or (u_op == "mov" and ZERO_X21_RE.match(u_ops)):
                continue
            if u_op in ("bl", "blr"):
                tgt = branch_target(u_ops)
                bname = stubs.get(tgt, u_ops) if tgt is not None else u_ops
                if "_swift_willThrow" in bname or "_swift_unexpectedError" in bname:
                    continue
            if u in ext_tail:
                tname = ext_tail[u]
                if not any(k in tname for k in SKIP_TAIL_STUBS):
                    return ("UNSAVED_X21_TAIL", sym, addr, f"{op} {ops}", tname)
                continue
            if u in is_ret and from_objc:
                return ("UNSAVED_X21_OBJC_RET", sym, addr, f"{op} {ops}", "ret")
            for v in succ[u]:
                if v not in seen:
                    bfs.append(v)

    if save_indices:
        for i, (addr, op, ops) in enumerate(insns):
            if is_restore(op, ops) or not is_write_x21(op, ops):
                continue
            bfs = deque(succ[i])
            seen = {i}
            while bfs:
                u = bfs.popleft()
                if u in seen:
                    continue
                seen.add(u)
                _, u_op, u_ops = insns[u]
                if is_restore(u_op, u_ops):
                    continue
                if u in ext_tail and ext_tail[u] in RULE_B_TAIL_STUBS:
                    return ("UNRESTORED_X21_TAIL", sym, addr, f"{op} {ops}", ext_tail[u])
                for v in succ[u]:
                    if v not in seen:
                        bfs.append(v)
    return None


def check_binary(path: str) -> tuple[int, list[tuple[str, str, int, str, str]]]:
    stubs = load_stubs(path)
    proc = subprocess.Popen(["objdump", "-d", path], stdout=subprocess.PIPE, text=True)
    cur_sym: str | None = None
    insns: list[tuple[int, str, str]] = []
    hits: list[tuple[str, str, int, str, str]] = []
    swift_funcs = 0

    assert proc.stdout is not None
    for line in proc.stdout:
        m = SYM_RE.match(line)
        if m:
            if cur_sym and cur_sym.startswith("_$s"):
                swift_funcs += 1
                hit = scan_function(cur_sym, insns, stubs)
                if hit:
                    hits.append(hit)
            cur_sym = m.group(2)
            insns = []
            continue
        m2 = INSN_RE.match(line)
        if m2 and cur_sym and cur_sym.startswith("_$s"):
            insns.append((int(m2.group(1), 16), m2.group(2), m2.group(3).strip()))
    if cur_sym and cur_sym.startswith("_$s"):
        swift_funcs += 1
        hit = scan_function(cur_sym, insns, stubs)
        if hit:
            hits.append(hit)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"objdump exited {rc} for {path}")
    return swift_funcs, hits


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: check_swifterror_abi.py <macho-binary> [...]", file=sys.stderr)
        return 2
    summaries: list[str] = []
    all_hits: list[tuple[str, str, str, int, str, str]] = []
    for path in argv[1:]:
        funcs, hits = check_binary(path)
        summaries.append(f"{path}: {funcs} funcs")
        for h in hits:
            all_hits.append((path, *h))
    if all_hits:
        dem = subprocess.run(
            ["xcrun", "swift-demangle"],
            input="\n".join(h[2] for h in all_hits),
            text=True, capture_output=True, check=True,
        ).stdout.splitlines()
        print(f"FAIL: {len(all_hits)} swifterror (x21) ABI violation(s):", file=sys.stderr)
        for (path, kind, sym, addr, ins, tail), d in zip(all_hits, dem):
            print(f"  [{kind}] {path} @ 0x{addr:x}: {d}", file=sys.stderr)
            print(f"      write={ins} -> tail={tail} ({sym})", file=sys.stderr)
        return 1
    print("swifterror ABI: PASS (" + ", ".join(summaries) + ")")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
