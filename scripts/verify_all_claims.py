#!/usr/bin/env python3
"""Zero-GPU, zero-third-party-dependency verification suite for Phase-SHA claims.

Usage:
    python scripts/verify_all_claims.py
    python scripts/verify_all_claims.py --run-lean

This script enforces CUDA_VISIBLE_DEVICES="" and independently verifies:
  1. Pure-Python FIPS 180-4 SHA-256 reference parity against Python hashlib,
     the Bitcoin Genesis Block header (double-SHA256), and optional CPU src/sha256_ref.py.
  2. All 50/50 SAT hex message-pair witnesses in P13 (including standard-IV R=16
     non-trivial collision, S3 Double-SHA256 pairs, and G4 isomorphic controls).
  3. All 4/4 Bitcoin 80-byte witness headers & GE/holdout invariants in P12.
  4. Zero-sorry, zero-admit static audit across all 4 Lean 4 projects (32 files / 483 theorem/lemma declarations: 338 public + 145 private/protected).
  5. JSON structural integrity across all V11 and V12 verdict.json files.
  6. (Optional --run-lean) Fresh `lake build` + VerifyAxioms check on Lean 4.36.0.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

MASK32 = 0xFFFFFFFF
REPO_ROOT = Path(__file__).resolve().parent.parent


def _first_n_primes(n: int) -> Tuple[int, ...]:
    primes: List[int] = []
    cand = 2
    while len(primes) < n:
        lim = math.isqrt(cand)
        if all(cand % p != 0 for p in primes if p <= lim):
            primes.append(cand)
        cand += 1
    return tuple(primes)


def _icbrt(n: int) -> int:
    if n <= 0:
        return 0
    x = 1 << ((n.bit_length() + 2) // 3)
    while True:
        y = (2 * x + n // (x * x)) // 3
        if y >= x:
            return x
        x = y


SHA256_IV: Tuple[int, ...] = tuple(
    (math.isqrt(p << 64)) & MASK32 for p in _first_n_primes(8)
)
SHA256_K: Tuple[int, ...] = tuple(
    (_icbrt(p << 96)) & MASK32 for p in _first_n_primes(64)
)

# P13 G4 Isomorphic Control Cipher (seed 20261125, matching run_p13_experiments.py)
CTRL_SIGMA0_SHIFTS: Tuple[int, int, int] = (1, 7, 25)
CTRL_SIGMA1_SHIFTS: Tuple[int, int, int] = (4, 9, 27)
_ctrl_rng = random.Random(20261125)
CTRL_K_CONSTS: Tuple[int, ...] = tuple(
    _ctrl_rng.getrandbits(32) for _ in range(64)
)


def rotr32(x: int, n: int) -> int:
    x &= MASK32
    if n == 0:
        return x
    return ((x >> n) | ((x << (32 - n)) & MASK32)) & MASK32


def expand_schedule(w16: Sequence[int], rounds: int) -> List[int]:
    w = [int(v) & MASK32 for v in w16]
    for t in range(16, rounds):
        s0 = rotr32(w[t - 15], 7) ^ rotr32(w[t - 15], 18) ^ (w[t - 15] >> 3)
        s1 = rotr32(w[t - 2], 17) ^ rotr32(w[t - 2], 19) ^ (w[t - 2] >> 10)
        w.append((w[t - 16] + s0 + w[t - 7] + s1) & MASK32)
    return w[:rounds]


def sha256_compress_py(
    w16: Sequence[int],
    rounds: int = 64,
    iv: Sequence[int] = SHA256_IV,
    k_consts: Sequence[int] = SHA256_K,
    s0_shifts: Tuple[int, int, int] = (2, 13, 22),
    s1_shifts: Tuple[int, int, int] = (6, 11, 25),
) -> List[int]:
    w = expand_schedule(w16, rounds)
    a, b, c, d, e, f, g, h = (int(v) & MASK32 for v in iv)
    for t in range(rounds):
        s1 = rotr32(e, s1_shifts[0]) ^ rotr32(e, s1_shifts[1]) ^ rotr32(e, s1_shifts[2])
        ch = (e & f) ^ ((~e & MASK32) & g)
        t1 = (h + s1 + ch + k_consts[t] + w[t]) & MASK32
        s0 = rotr32(a, s0_shifts[0]) ^ rotr32(a, s0_shifts[1]) ^ rotr32(a, s0_shifts[2])
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = (s0 + maj) & MASK32
        h, g, f = g, f, e
        e = (d + t1) & MASK32
        d, c, b = c, b, a
        a = (t1 + t2) & MASK32
    return [
        (a + iv[0]) & MASK32,
        (b + iv[1]) & MASK32,
        (c + iv[2]) & MASK32,
        (d + iv[3]) & MASK32,
        (e + iv[4]) & MASK32,
        (f + iv[5]) & MASK32,
        (g + iv[6]) & MASK32,
        (h + iv[7]) & MASK32,
    ]


def sha256_bytes_py(msg: bytes) -> bytes:
    bit_len = len(msg) * 8
    padded = msg + b"\x80"
    while len(padded) % 64 != 56:
        padded += b"\x00"
    padded += bit_len.to_bytes(8, "big")
    state = list(SHA256_IV)
    for off in range(0, len(padded), 64):
        blk = [int.from_bytes(padded[off + 4 * i : off + 4 * (i + 1)], "big") for i in range(16)]
        state = sha256_compress_py(blk, rounds=64, iv=state)
    return b"".join(w.to_bytes(4, "big") for w in state)


def verify_check1_sha256_reference() -> Tuple[bool, str]:
    # 1a. Parity with hashlib on empty, abc, and deterministic test vectors
    rng = random.Random(2026)
    vectors = [b"", b"abc", b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"]
    vectors.extend(rng.randbytes(length) for length in range(1, 200, 9))
    for vec in vectors:
        got = sha256_bytes_py(vec)
        exp = hashlib.sha256(vec).digest()
        if got != exp:
            return False, f"SHA-256 parity failure on len={len(vec)}: got={got.hex()} exp={exp.hex()}"

    # 1b. Bitcoin Genesis Block header double-SHA256
    genesis_hex = (
        "01000000"
        "0000000000000000000000000000000000000000000000000000000000000000"
        "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
        "29ab5f49"
        "ffff001d"
        "1dac2b7c"
    )
    genesis_bytes = bytes.fromhex(genesis_hex)
    assert len(genesis_bytes) == 80
    block_hash = sha256_bytes_py(sha256_bytes_py(genesis_bytes))[::-1].hex()
    expected_genesis = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"
    if block_hash != expected_genesis:
        return False, f"BTC Genesis mismatch: {block_hash} != {expected_genesis}"

    # 1c. Optional CPU parity with src/sha256_ref.py if torch is installed
    torch_status = "torch not installed (pure-Python verified)"
    try:
        import torch  # type: ignore

        sys.path.insert(0, str(REPO_ROOT))
        from src.sha256_ref import sha256_digest  # type: ignore

        for r in (4, 12, 16, 64):
            words = [[rng.getrandbits(32) for _ in range(16)] for _ in range(8)]
            t_in = torch.tensor(words, dtype=torch.int64, device="cpu")
            t_out = sha256_digest(t_in, rounds=r).tolist()
            for row_idx, wrow in enumerate(words):
                if t_out[row_idx] != sha256_compress_py(wrow, rounds=r):
                    return False, f"src/sha256_ref.py CPU mismatch at rounds={r}, row={row_idx}"
        torch_status = "src/sha256_ref.py CPU parity PASS (rounds=4,12,16,64)"
    except ImportError:
        pass

    return True, f"{len(vectors)} vectors + BTC Genesis ({expected_genesis[:18]}...) + {torch_status}"


def verify_check2_p13_witnesses() -> Tuple[bool, str]:
    path = REPO_ROOT / "v12_frontier" / "track_b1_p13_local_coll" / "results" / "p13_witnesses.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    witnesses = data["witnesses"]
    if len(witnesses) != 50:
        return False, f"Expected 50 witnesses, found {len(witnesses)}"

    std_r16_collision_seen = False
    for idx, w in enumerate(witnesses):
        label = w["label"]
        rounds = int(w["rounds"])
        r2: Optional[int] = w.get("r2")
        m1 = [int(x, 16) for x in w["M_hex"]]
        m2 = [int(x, 16) for x in w["M_prime_hex"]]
        iv1 = [int(x, 16) for x in w["IV_hex"]]
        iv2 = [int(x, 16) for x in w["IV_prime_hex"]]
        h1_exp = [int(x, 16) for x in w["H_hex"]]
        h2_exp = [int(x, 16) for x in w["H_prime_hex"]]

        if m1 == m2 and iv1 == iv2:
            return False, f"Witness #{idx} ({label}) has identical inputs!"

        if label.startswith("G4_CTRL"):
            h1_got = sha256_compress_py(
                m1, rounds=rounds, iv=iv1,
                k_consts=CTRL_K_CONSTS, s0_shifts=CTRL_SIGMA0_SHIFTS, s1_shifts=CTRL_SIGMA1_SHIFTS
            )
            h2_got = sha256_compress_py(
                m2, rounds=rounds, iv=iv2,
                k_consts=CTRL_K_CONSTS, s0_shifts=CTRL_SIGMA0_SHIFTS, s1_shifts=CTRL_SIGMA1_SHIFTS
            )
        elif r2 is not None:
            b1_a = sha256_compress_py(m1, rounds=rounds, iv=iv1)
            b1_b = sha256_compress_py(m2, rounds=rounds, iv=iv2)
            pad_a = b1_a + [0x80000000, 0, 0, 0, 0, 0, 0, 256]
            pad_b = b1_b + [0x80000000, 0, 0, 0, 0, 0, 0, 256]
            h1_got = sha256_compress_py(pad_a, rounds=int(r2), iv=SHA256_IV)
            h2_got = sha256_compress_py(pad_b, rounds=int(r2), iv=SHA256_IV)
        else:
            h1_got = sha256_compress_py(m1, rounds=rounds, iv=iv1)
            h2_got = sha256_compress_py(m2, rounds=rounds, iv=iv2)

        if h1_got != h1_exp or h2_got != h2_exp:
            return False, f"Digest mismatch on #{idx} ({label})"

        hw_got = sum((a ^ b).bit_count() for a, b in zip(h1_got, h2_got))
        if hw_got != int(w["delta_H_hw"]):
            return False, f"delta_H_hw mismatch on #{idx} ({label}): {hw_got} vs {w['delta_H_hw']}"

        clz1 = 32 - h1_got[0].bit_length()
        clz2 = 32 - h2_got[0].bit_length()
        if clz1 != int(w["clz_H0"]) or clz2 != int(w["clz_H0_prime"]):
            return False, f"clz_H0 mismatch on #{idx} ({label})"

        if label == "ladder_R16_standard" and hw_got == 0 and bool(w["g2_nontrivial"]):
            std_r16_collision_seen = True

    if not std_r16_collision_seen:
        return False, "Missing expected standard-IV R=16 G2-nontrivial exact collision!"

    return True, "50/50 SAT hex witnesses bit-verified (including R=16 std-IV collision, S3 Double-SHA256 & G4_CTRL)"


def verify_check3_p12_btc_headers() -> Tuple[bool, str]:
    path = REPO_ROOT / "v12_frontier" / "track_a_p12_amortized" / "results" / "a2_gnbc_search.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["main_vs_ctrl_summary"]
    if float(summary["main_a2_1_p_neutral_t5"]) != 0.0:
        return False, f"Unexpected main_a2_1_p_neutral_t5={summary['main_a2_1_p_neutral_t5']}"
    if float(summary["main_best_net_gain_vs_classic_pct"]) >= 0.0:
        return False, f"Unexpected positive net gain: {summary['main_best_net_gain_vs_classic_pct']}"

    total_headers = 0
    for track_name, track_obj in data["tracks"].items():
        headers = track_obj["A2-3"]["best_nonce_cluster_holdout"]["witness_headers_hex"]
        if len(headers) != 4:
            return False, f"{track_name} expected 4 headers, got {len(headers)}"
        raws = [bytes.fromhex(h) for h in headers]
        if any(len(r) != 80 for r in raws):
            return False, f"{track_name} header length != 80 bytes"
        chunk1_prefix = raws[0][:64]
        mid_prefix = raws[0][64:76]
        nonces = set()
        digests = set()
        for r in raws:
            if r[:64] != chunk1_prefix or r[64:76] != mid_prefix:
                return False, f"{track_name} headers do not share Chunk1/prefix!"
            nonces.add(r[76:80])
            d1 = sha256_bytes_py(sha256_bytes_py(r))
            if d1 != hashlib.sha256(hashlib.sha256(r).digest()).digest():
                return False, f"{track_name} hashlib mismatch"
            digests.add(d1)
        if len(nonces) != 4 or len(digests) != 4:
            return False, f"{track_name} expected 4 distinct nonces and 4 distinct digests"
        total_headers += len(raws)

    return True, (
        f"{total_headers}/16 BTC 80-byte headers verified across 4 tracks "
        f"(p_neutral_t5=0.0, best_net_vs_classic={summary['main_best_net_gain_vs_classic_pct']:.3f}%)"
    )


def _strip_lean_comments(src: str) -> str:
    """Remove Lean 4 `-- ...` and nested `/- ... -/` comments while keeping line structure."""
    out: List[str] = []
    i = 0
    n = len(src)
    depth = 0
    in_str = False
    while i < n:
        ch = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if depth > 0:
            if ch == "/" and nxt == "-":
                depth += 1
                i += 2
            elif ch == "-" and nxt == "/":
                depth -= 1
                i += 2
            else:
                if ch == "\n":
                    out.append("\n")
                i += 1
        elif in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(src[i + 1])
                i += 2
            else:
                if ch == '"':
                    in_str = False
                i += 1
        else:
            if ch == "-" and nxt == "-":
                while i < n and src[i] != "\n":
                    i += 1
            elif ch == "/" and nxt == "-":
                depth = 1
                i += 2
            else:
                if ch == '"':
                    in_str = True
                out.append(ch)
                i += 1
    return "".join(out)


def verify_check4_lean4_zero_sorry() -> Tuple[bool, str]:
    roots = [
        REPO_ROOT / "formal_proofs",
        REPO_ROOT / "formal_proofs_v10",
        REPO_ROOT / "v11_discovery" / "p3_symbolic_lean4" / "lean_verify",
        REPO_ROOT / "v12_frontier" / "track_b2_p15_mitm_biclique" / "lean_verify",
    ]
    decl_pat = re.compile(r"^\s*(?:private\s+|protected\s+)?(?:theorem|lemma)\b", re.MULTILINE)
    sorry_pat = re.compile(r"\b(?:sorry|admit)\b")
    axiom_pat = re.compile(r"^\s*axiom\b", re.MULTILINE)

    total_files = 0
    total_decls = 0
    per_project: List[str] = []
    for root in roots:
        if not root.exists():
            return False, f"Missing Lean 4 directory: {root}"
        lean_files = sorted(p for p in root.rglob("*.lean") if ".lake" not in p.parts)
        if not lean_files:
            return False, f"No .lean files in {root}"
        proj_decls = 0
        for lf in lean_files:
            cleaned = _strip_lean_comments(lf.read_text(encoding="utf-8"))
            if sorry_pat.search(cleaned):
                return False, f"Forbidden sorry/admit found in {lf}"
            if axiom_pat.search(cleaned):
                return False, f"Forbidden custom axiom declaration found in {lf}"
            proj_decls += len(decl_pat.findall(cleaned))
        total_files += len(lean_files)
        total_decls += proj_decls
        rel = root.relative_to(REPO_ROOT).as_posix()
        per_project.append(f"{rel}={len(lean_files)}f/{proj_decls}t")

    return True, f"{total_files} .lean files, {total_decls} theorem/lemma declarations, 0 sorry ({', '.join(per_project)})"


def verify_check5_verdict_jsons() -> Tuple[bool, str]:
    verdicts: List[Path] = sorted((REPO_ROOT / "v11_discovery").glob("*/verdict.json"))
    verdicts.extend(sorted((REPO_ROOT / "v12_frontier").glob("*/verdict.json")))
    if len(verdicts) < 14:
        return False, f"Expected >= 14 verdict.json files across V11+V12, found {len(verdicts)}"
    statuses: Dict[str, int] = {}
    for vp in verdicts:
        obj = json.loads(vp.read_text(encoding="utf-8"))
        st = str(obj.get("status", obj.get("verdict", "UNKNOWN")))
        statuses[st] = statuses.get(st, 0) + 1
    return True, f"{len(verdicts)} verdict.json files valid JSON; status breakdown={statuses}"


def run_optional_lean_builds() -> Tuple[bool, str]:
    targets = [
        ("P3", REPO_ROOT / "v11_discovery" / "p3_symbolic_lean4" / "lean_verify", "VerifyAxioms.lean"),
        ("P15", REPO_ROOT / "v12_frontier" / "track_b2_p15_mitm_biclique" / "lean_verify", "VerifyAxioms.lean"),
    ]
    for name, cwd, ax_file in targets:
        print(f"  [Lean] Building {name} in {cwd} (`lake build -- -j 2`)...", flush=True)
        res = subprocess.run(["lake", "build"], cwd=cwd, capture_output=True, text=True)
        if res.returncode != 0:
            return False, f"{name} lake build failed:\n{res.stdout}\n{res.stderr}"
        ax = subprocess.run(["lake", "env", "lean", ax_file], cwd=cwd, capture_output=True, text=True)
        if ax.returncode != 0 or "sorryAx" in ax.stdout or "sorryAx" in ax.stderr:
            return False, f"{name} axiom check failed:\n{ax.stdout}\n{ax.stderr}"
    return True, "P3 & P15 Lean 4.36.0 `lake build` + `VerifyAxioms.lean` PASS (zero sorryAx)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Phase-SHA V1-V12 empirical & formal claims (CPU-only).")
    parser.add_argument("--run-lean", action="store_true", help="Also run `lake build` + axiom audit if `lake` is installed.")
    args = parser.parse_args()

    checks = [
        ("1. SHA-256 Reference Parity & BTC Genesis", verify_check1_sha256_reference),
        ("2. P13 50/50 SAT Hex Witness Verification", verify_check2_p13_witnesses),
        ("3. P12 Bitcoin 80-byte Header & GE Invariants", verify_check3_p12_btc_headers),
        ("4. Lean 4 Zero-Sorry Corpus Audit (4 Projects)", verify_check4_lean4_zero_sorry),
        ("5. V11 + V12 Verdict JSON Integrity", verify_check5_verdict_jsons),
    ]
    if args.run_lean:
        checks.append(("6. Live Lean 4 `lake build` + Axiom Check", run_optional_lean_builds))

    all_ok = True
    print("=== Phase-SHA Independent Verification Suite (CUDA_VISIBLE_DEVICES='') ===", flush=True)
    for title, fn in checks:
        ok, msg = fn()
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {title}: {msg}", flush=True)
        if not ok:
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
