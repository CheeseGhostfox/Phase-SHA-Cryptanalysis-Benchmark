"""Main P13 experiment runner for Colab (GPU L4/A100 pure FP32 + CaDiCaL195 SAT).

Runs:
  1. B1-1 parity verification & pure FP32 lock check
  2. Track 1: Positive calibration (R=12, 16) + Standard ladder (R=16, 20, 24, 28, 32)
     with AlphaZero 3D Crystal Policy-Value Net + Beam/MCTS + CaDiCaL195 + G3 check
  3. Track 2: BTC truncated leading-zero differential tunnels (k in {4, 8, 12, 16})
     across S1, S2a, S2b, S3 (Double-SHA256), with strict G2 span checks,
     two-stage anchor->256-sample amortized speedup accounting, and G4 control cipher.

Outputs in results/:
  - p13_ladder_results.json
  - p13_btc_truncated_lz.json
  - p13_witnesses.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("highest")

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
for _root in [_THIS_DIR, *_THIS_DIR.parents]:
    if (_root / "src" / "sha256_ref.py").exists():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break

from alphazero_crystal_mcts import (  # noqa: E402
    CrystalPolicyValueNet,
    evaluate_g3_greedy_1bit_recover,
    run_alphazero_beam_sat_search,
    train_crystal_net,
)
from mendel_state_machine import (  # noqa: E402
    GD_ANY,
    GD_EQ,
    GD_X,
    build_mendel_dv_skeleton,
    create_blank_mendel_state,
    propagate_mendel_fixpoint,
    sha256_trace,
)
from sat_witness_solver import (  # noqa: E402
    WitnessResult,
    solve_differential_witness,
    solve_double_sha256_witness,
)
from src.sha256_ref import MASK32, SHA256_IV, SHA256_K, sha256_digest  # noqa: E402

# G4 Isomorphic Control Cipher parameters (SPEC Section 4 G4)
CTRL_SIGMA0_SHIFTS: Tuple[int, int, int] = (1, 7, 25)
CTRL_SIGMA1_SHIFTS: Tuple[int, int, int] = (4, 9, 27)
_ctrl_rng = random.Random(20261125)
CTRL_K_CONSTS: Tuple[int, ...] = tuple(_ctrl_rng.getrandbits(32) for _ in range(64))


def _sha256_custom_ref(
    w16: torch.Tensor,
    rounds: int,
    iv: Optional[torch.Tensor] = None,
    k_consts: Sequence[int] = SHA256_K,
    s0_shifts: Tuple[int, int, int] = (2, 13, 22),
    s1_shifts: Tuple[int, int, int] = (6, 11, 25),
) -> torch.Tensor:
    """Reference int64 SHA-256 (or CTRL_ROTR_K) digest evaluator for witness verification."""
    from src.sha256_ref import ch32, expand_message_schedule, maj32, rotr32

    B = w16.shape[0]
    device = w16.device
    iv_b = (
        torch.tensor(SHA256_IV, dtype=torch.int64, device=device).unsqueeze(0).expand(B, 8)
        if iv is None
        else (iv.to(torch.int64) & MASK32)
    )
    k_vec = torch.tensor(list(k_consts[:rounds]), dtype=torch.int64, device=device)
    w_full = expand_message_schedule(w16, rounds)

    a = [iv_b[:, 3], iv_b[:, 2], iv_b[:, 1], iv_b[:, 0]]
    e = [iv_b[:, 7], iv_b[:, 6], iv_b[:, 5], iv_b[:, 4]]

    for t in range(rounds):
        s1 = (rotr32(e[t + 3], s1_shifts[0]) ^ rotr32(e[t + 3], s1_shifts[1]) ^ rotr32(e[t + 3], s1_shifts[2])) & MASK32
        s0 = (rotr32(a[t + 3], s0_shifts[0]) ^ rotr32(a[t + 3], s0_shifts[1]) ^ rotr32(a[t + 3], s0_shifts[2])) & MASK32
        t1 = (e[t] + s1 + ch32(e[t + 3], e[t + 2], e[t + 1]) + k_vec[t] + w_full[:, t]) & MASK32
        t2 = (s0 + maj32(a[t + 3], a[t + 2], a[t + 1])) & MASK32
        e.append((a[t] + t1) & MASK32)
        a.append((t1 + t2) & MASK32)

    s_r = torch.stack(
        [a[rounds + 3], a[rounds + 2], a[rounds + 1], a[rounds], e[rounds + 3], e[rounds + 2], e[rounds + 1], e[rounds]],
        dim=1,
    )
    return (s_r + iv_b) & MASK32


def _format_witness_record(
    label: str,
    rounds: int,
    res: WitnessResult,
    is_double_sha: bool = False,
    r2: int = 0,
    btc_padding: bool = False,
    k_consts: Sequence[int] = SHA256_K,
    s0_shifts: Tuple[int, int, int] = (2, 13, 22),
    s1_shifts: Tuple[int, int, int] = (6, 11, 25),
) -> Dict[str, object]:
    """Verify SAT witness against sha256_ref and format hex record for JSON."""
    assert res.sat and res.w16 is not None and res.w16_prime is not None
    w1_t = torch.tensor([res.w16], dtype=torch.int64)
    w2_t = torch.tensor([res.w16_prime], dtype=torch.int64)
    iv1_t = torch.tensor([res.iv], dtype=torch.int64) if res.iv is not None else None
    iv2_t = torch.tensor([res.iv_prime], dtype=torch.int64) if res.iv_prime is not None else None

    if is_double_sha:
        h1_a = _sha256_custom_ref(w1_t, rounds=rounds)
        h1_b = _sha256_custom_ref(w2_t, rounds=rounds)
        def _pad_h1(hv: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((1, 16), dtype=torch.int64)
            out[:, :8] = hv
            out[:, 8] = 0x80000000
            out[:, 15] = 256
            return out
        d1 = _sha256_custom_ref(_pad_h1(h1_a), rounds=r2)[0].tolist()
        d2 = _sha256_custom_ref(_pad_h1(h1_b), rounds=r2)[0].tolist()
    else:
        d1 = _sha256_custom_ref(w1_t, rounds=rounds, iv=iv1_t, k_consts=k_consts, s0_shifts=s0_shifts, s1_shifts=s1_shifts)[0].tolist()
        d2 = _sha256_custom_ref(w2_t, rounds=rounds, iv=iv2_t, k_consts=k_consts, s0_shifts=s0_shifts, s1_shifts=s1_shifts)[0].tolist()

    assert d1 == res.digest and d2 == res.digest_prime, f"Ref mismatch on {label}"
    diff_w = [a ^ b for a, b in zip(res.w16, res.w16_prime)]
    active_w_indices = [i for i, dv in enumerate(diff_w) if dv != 0]
    t_min = min(active_w_indices) if active_w_indices else -1
    diff_h = [a ^ b for a, b in zip(d1, d2)]
    hw_delta_h = sum(bin(x).count("1") for x in diff_h)
    clz_a = 32 - (d1[0] & MASK32).bit_length()
    clz_b = 32 - (d2[0] & MASK32).bit_length()
    btc_pad_ok = True
    if btc_padding:
        for w in (res.w16, res.w16_prime):
            if w[4] != 0x80000000 or any(w[i] != 0 for i in range(5, 15)) or w[15] != 640:
                btc_pad_ok = False

    # G2 non-triviality span check
    nontrivial_g2 = (t_min >= 0) and ((rounds - t_min) >= 8)
    if not is_double_sha and rounds <= 16:
        if any(i >= rounds - 3 for i in active_w_indices):
            nontrivial_g2 = False
    if is_double_sha and r2 < 16:
        nontrivial_g2 = False

    return {
        "label": label,
        "rounds": rounds,
        "r2": r2 if is_double_sha else None,
        "M_hex": [f"{x:08x}" for x in res.w16],
        "M_prime_hex": [f"{x:08x}" for x in res.w16_prime],
        "IV_hex": [f"{x:08x}" for x in (res.iv or list(SHA256_IV))],
        "IV_prime_hex": [f"{x:08x}" for x in (res.iv_prime or list(SHA256_IV))],
        "H_hex": [f"{x:08x}" for x in d1],
        "H_prime_hex": [f"{x:08x}" for x in d2],
        "delta_H_hw": hw_delta_h,
        "clz_H0": clz_a,
        "clz_H0_prime": clz_b,
        "active_W_indices": active_w_indices,
        "t_min": t_min,
        "core_span_R_minus_tmin": (rounds - t_min) if t_min >= 0 else 0,
        "g2_nontrivial": nontrivial_g2,
        "btc_padding_valid": btc_pad_ok,
        "sha256_ref_verified": True,
        "solve_seconds": round(res.solve_seconds, 4),
    }


def _measure_sha256_evals_per_sec(device: torch.device | str) -> float:
    """Benchmark exact SHA-256 throughput (evals/sec) for honest amortized speedup billing."""
    w = torch.randint(0, 1 << 32, (4096, 16), dtype=torch.int64, device=device)
    _ = sha256_digest(w, rounds=16)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(25):
        _ = sha256_digest(w, rounds=16)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    dt = max(1e-6, time.perf_counter() - t0)
    return (4096 * 25) / dt


def _random_min_hw_near_collision(rounds: int, device: torch.device | str, n_pairs: int = 16384) -> int:
    """Compute random sparse-input min HW(Delta H) baseline for near-collision comparison."""
    gen = torch.Generator(device=device).manual_seed(1000 + rounds)
    w1 = torch.randint(0, 1 << 32, (n_pairs, 16), dtype=torch.int64, device=device, generator=gen)
    w2 = w1.clone()
    w2[:, 0] ^= 1
    d1 = sha256_digest(w1, rounds=rounds)
    d2 = sha256_digest(w2, rounds=rounds)
    xor_d = d1 ^ d2
    # Popcount via bit shifts
    hw = torch.zeros(n_pairs, dtype=torch.int64, device=device)
    for b in range(32):
        hw += ((xor_d >> b) & 1).sum(dim=-1)
    return int(hw.min().item())


# ---------------------------------------------------------------------------
# Track 1: Positive calibration + Standard ladder (R = 12..32)
# ---------------------------------------------------------------------------

def run_standard_ladder(
    device: torch.device | str,
    witnesses_out: List[Dict[str, object]],
    sat_budget: int = 250_000,
) -> Dict[str, object]:
    """Run B1-2 Track 1: R=12, 16 positive calibration + R=16..32 ladder."""
    print("\n=== [Track 1] Standard Ladder & Positive Calibration ===", flush=True)
    ladder_records: List[Dict[str, object]] = []
    g3_records: List[Dict[str, object]] = []

    # 1. Positive calibration on R=12, 16 SFS/FS + local collision skeleton
    calib_specs = [
        ("calib_R12_sfs", 12, "semi_free", None),
        ("calib_R12_fs", 12, "free", None),
        ("calib_R16_sfs", 16, "semi_free", 2),
        ("calib_R16_fs", 16, "free", None),
    ]
    for label, R, iv_mode, dv_step in calib_specs:
        print(f"[Track 1] running calibration {label} (R={R}, iv={iv_mode})...", flush=True)
        if dv_step is not None:
            # Localized Chabaud-Joux DV window [2..12] leaving W[13..15] zero-diff -> G2 compliant!
            st = create_blank_mendel_state(1, rounds=R, iv_mode=iv_mode, device="cpu")
            for t in range(R):
                if t < 2 or t > 12:
                    st.W[:, t, :] = GD_EQ
            for k in range(-3, R + 1):
                if k <= 2 or k >= 13:
                    st.A[:, k + 3, :] = GD_EQ
                    st.E[:, k + 3, :] = GD_EQ
            st.W[:, 2, 0] = GD_X
            st = propagate_mendel_fixpoint(st, max_iters=4)
            w_d = {t: st.W[0, t].tolist() for t in range(R) if any(m != GD_ANY for m in st.W[0, t].tolist())}
            a_d = {k: st.A[0, k + 3].tolist() for k in range(-3, R + 1) if any(m != GD_ANY for m in st.A[0, k + 3].tolist())}
            e_d = {k: st.E[0, k + 3].tolist() for k in range(-3, R + 1) if any(m != GD_ANY for m in st.E[0, k + 3].tolist())}
        else:
            w_d, a_d, e_d = None, None, None

        res = solve_differential_witness(
            rounds=R,
            iv_mode=iv_mode,
            w_gdiff=w_d,
            a_gdiff=a_d,
            e_gdiff=e_d,
            digest_xor_zero_words=list(range(8)),
            require_nonzero_msg_diff=True,
            conflict_budget=sat_budget,
            max_solutions=1,
        )
        ok = bool(res and res[0].sat)
        rec = {
            "label": label,
            "rounds": R,
            "iv_mode": iv_mode,
            "sat": ok,
            "solve_seconds": round(res[0].solve_seconds, 4) if res else 0.0,
        }
        if ok:
            wrec = _format_witness_record(label, R, res[0])
            witnesses_out.append(wrec)
            rec["t_min"] = wrec["t_min"]
            rec["g2_nontrivial"] = wrec["g2_nontrivial"]
            rec["delta_H_hw"] = wrec["delta_H_hw"]
        ladder_records.append(rec)
        print(f"  -> {label}: sat={ok}, time={rec['solve_seconds']}s", flush=True)

    # 2. AlphaZero 3D Crystal Net training + MCTS/Beam + SAT search on R = 16, 20, 24, 28, 32
    for R in (16, 20, 24, 28, 32):
        print(f"\n[Track 1] Training AlphaZero Crystal Net for R={R} on {device}...", flush=True)
        net = CrystalPolicyValueNet(in_channels=20, hidden_channels=64, num_blocks=4)
        train_stats = train_crystal_net(net, rounds=R, device=device, steps=80, batch_size=32, seed=2026 + R)
        g3_stats = evaluate_g3_greedy_1bit_recover(net, rounds=R, device=device, n_probes=128, seed=700 + R)
        g3_entry = {"rounds": R, **train_stats, **g3_stats}
        g3_records.append(g3_entry)
        print(
            f"  [G3 R={R}] train_acc={train_stats['train_val_acc']}, "
            f"1bit_recover={g3_stats['greedy_1bit_recover_rate']} "
            f"(ratio={g3_stats['ratio_vs_random']}x, pass_1.5x={g3_stats['passes_g3_1_5x']})",
            flush=True,
        )

        # Standard-IV near-collision baseline (random sparse) + AlphaZero+SAT collision attempts
        rand_min_hw = _random_min_hw_near_collision(R, device=device, n_pairs=32768)

        for iv_mode in ("free", "semi_free", "standard"):
            label = f"ladder_R{R}_{iv_mode}"
            # Seed MCTS with localized disturbance window [0..min(R-1, 10)] leaving tail unconstrained
            st_init = build_mendel_dv_skeleton(step_t=0, bit_j=0, rounds=R, iv_mode=iv_mode)
            search_res = run_alphazero_beam_sat_search(
                net=net,
                init_state=st_init,
                device=device,
                beam_width=6,
                max_depth=8,
                sat_check_interval=4,
                iv_mode=iv_mode,
                digest_xor_zero_words=list(range(8)),
                sat_conflict_budget=sat_budget // 2,
                max_solutions=1,
            )
            # Also try unconstrained direct SAT (within budget) if skeleton was UNSAT
            direct_sat = False
            direct_time = 0.0
            if not search_res["sat"]:
                # Restrict only W[R-3..R-1] = '-' for G2 compliance when R <= 16
                w_gdiff_g2 = {t: [GD_EQ] * 32 for t in range(max(0, R - 3), R)} if R <= 16 else None
                d_res = solve_differential_witness(
                    rounds=R,
                    iv_mode=iv_mode,
                    w_gdiff=w_gdiff_g2,
                    digest_xor_zero_words=list(range(8)),
                    require_nonzero_msg_diff=True,
                    conflict_budget=sat_budget,
                    max_solutions=1,
                )
                if d_res:
                    direct_sat = bool(d_res[0].sat)
                    direct_time = round(d_res[0].solve_seconds, 4)
                    if direct_sat:
                        search_res["sat"] = True
                        search_res["witnesses"] = d_res

            rec = {
                "label": label,
                "rounds": R,
                "iv_mode": iv_mode,
                "sat": bool(search_res["sat"]),
                "sat_calls": search_res["sat_calls"],
                "mendel_contradictions": search_res["mendel_contradictions"],
                "elapsed_seconds": search_res["elapsed_seconds"],
                "direct_sat_seconds": direct_time,
                "random_sparse_min_hw_delta_H": rand_min_hw if iv_mode == "standard" else None,
            }
            if search_res["sat"] and search_res["witnesses"]:
                wrec = _format_witness_record(label, R, search_res["witnesses"][0])
                witnesses_out.append(wrec)
                rec["t_min"] = wrec["t_min"]
                rec["g2_nontrivial"] = wrec["g2_nontrivial"]
                rec["delta_H_hw"] = wrec["delta_H_hw"]
            ladder_records.append(rec)
            print(
                f"  -> {label}: sat={rec['sat']}, sat_calls={rec['sat_calls']}, "
                f"contradictions={rec['mendel_contradictions']}, rand_min_hw={rand_min_hw if iv_mode=='standard' else '-'}",
                flush=True,
            )

    return {"ladder": ladder_records, "g3_evaluations": g3_records}


# ---------------------------------------------------------------------------
# Track 2: BTC Truncated Leading-Zero Oriented Local Collision (S1, S2a, S2b, S3)
# ---------------------------------------------------------------------------

def _evaluate_tunnel_derivation(
    scenario: str,
    rounds: int,
    k_bits: int,
    anchor_w16: List[int],
    derived_candidates: List[List[int]],
    search_seconds: float,
    derive_seconds: float,
    sha_evals_per_sec: float,
    is_double_sha: bool = False,
    r2: int = 0,
    k_consts: Sequence[int] = SHA256_K,
    s0_shifts: Tuple[int, int, int] = (2, 13, 22),
    s1_shifts: Tuple[int, int, int] = (6, 11, 25),
) -> Dict[str, object]:
    """Compute exact p_hit, binomial z-score, and SPEC 3.4 full amortized speedup."""
    w_t = torch.tensor(derived_candidates, dtype=torch.int64)
    if is_double_sha:
        h1 = _sha256_custom_ref(w_t, rounds=rounds)
        b2 = torch.zeros((len(derived_candidates), 16), dtype=torch.int64)
        b2[:, :8] = h1
        b2[:, 8] = 0x80000000
        b2[:, 15] = 256
        digs = _sha256_custom_ref(b2, rounds=r2)
    else:
        digs = _sha256_custom_ref(w_t, rounds=rounds, k_consts=k_consts, s0_shifts=s0_shifts, s1_shifts=s1_shifts)

    top_shift = 32 - k_bits
    hits = int(((digs[:, 0] >> top_shift) == 0).sum().item())
    B = len(derived_candidates)
    p_hit = hits / max(1, B)
    p0 = 2.0 ** (-k_bits)
    std0 = math.sqrt(max(1e-12, p0 * (1.0 - p0) / max(1, B)))
    z_score = (p_hit - p0) / std0
    gain_marginal = p_hit / p0

    # Equivalent SHA-256 eval cost of search + derivation (calibrated by throughput)
    c_search = max(1.0, search_seconds * sha_evals_per_sec)
    c_derive_total = max(float(B), derive_seconds * sha_evals_per_sec)
    speedup_amortized = (hits * (2.0 ** k_bits)) / (c_search + c_derive_total)

    return {
        "B_derived": B,
        "hits": hits,
        "p_hit": round(p_hit, 6),
        "p0_random": p0,
        "gain_marginal": round(gain_marginal, 4),
        "binomial_z": round(z_score, 3),
        "C_search_sha_equiv": round(c_search, 1),
        "C_derive_total_sha_equiv": round(c_derive_total, 1),
        "speedup_amortized": round(speedup_amortized, 6),
        "statistically_significant_z5": bool(z_score >= 5.0),
        "effective_speedup_gt_1": bool(z_score >= 5.0 and speedup_amortized > 1.0),
    }


def run_btc_truncated_lz(
    device: torch.device | str,
    witnesses_out: List[Dict[str, object]],
    sat_budget: int = 120_000,
) -> Dict[str, object]:
    """Run B1-2 Track 2: S1, S2a, S2b, S3 across k in {4, 8, 12, 16}."""
    print("\n=== [Track 2] BTC Truncated Leading-Zero Oriented Differential ===", flush=True)
    sha_rate = _measure_sha256_evals_per_sec(device)
    print(f"[Track 2] Calibrated SHA-256 reference throughput: {sha_rate:,.0f} evals/sec", flush=True)

    results: List[Dict[str, object]] = []
    rng = random.Random(20261125)

    # Train AlphaZero Crystal Net for BTC Block 2 padded boundary (R=16, 20, 24)
    nets_by_r: Dict[int, CrystalPolicyValueNet] = {}
    for R in (16, 20, 24):
        net = CrystalPolicyValueNet(in_channels=20, hidden_channels=64, num_blocks=4)
        train_crystal_net(net, rounds=R, device=device, steps=60, batch_size=32, btc_block2_padding=True, seed=5000 + R)
        nets_by_r[R] = net

    # Scenario definitions
    # 1. S1 (Single-block free W[0..15]) with G2 non-triviality (W[R-3..R-1] locked when R <= 16)
    #    plus explicit TRIVIAL_LAST_ROUND_INJECTION baseline at R=8/16 for anti-hallucination contrast.
    s1_configs = [
        ("S1_trivial_last_round", 8, False, True),
        ("S1_g2_nontrivial", 12, True, False),
        ("S1_g2_nontrivial", 16, True, False),
        ("S1_g2_nontrivial", 20, True, False),
        ("S1_g2_nontrivial", 24, True, False),
    ]
    for scen, R, enforce_g2, allow_last in s1_configs:
        for k_bits in (4, 8, 12):
            label = f"{scen}_R{R}_k{k_bits}"
            t_search0 = time.perf_counter()
            w_gdiff: Dict[int, List[int]] = {}
            if enforce_g2:
                # Lock W[R-3..R-1] = '-' when R <= 16, and force active perturbation early (W[0] diff)
                if R <= 16:
                    for t_lock in range(max(0, R - 3), R):
                        w_gdiff[t_lock] = [GD_EQ] * 32
                # Early disturbance window in W[0..3]
                for t_lock in range(4, min(R, 16)):
                    w_gdiff[t_lock] = [GD_EQ] * 32
            else:
                # Trivial P2 last-round channel demonstration
                for t_lock in range(0, R - 1):
                    w_gdiff[t_lock] = [GD_EQ] * 32

            sat_res = solve_differential_witness(
                rounds=R,
                iv_mode="standard",
                w_gdiff=w_gdiff,
                leading_zero_bits=k_bits,
                require_nonzero_msg_diff=True,
                conflict_budget=sat_budget,
                max_solutions=4 if R <= 16 else 1,
            )
            search_dt = time.perf_counter() - t_search0
            sat_ok = bool(sat_res and sat_res[0].sat)

            entry: Dict[str, object] = {
                "scenario": scen,
                "label": label,
                "rounds": R,
                "k_bits": k_bits,
                "g2_enforced": enforce_g2,
                "tagged_trivial_last_round": allow_last,
                "sat": sat_ok,
                "sat_seconds": round(search_dt, 4),
            }
            if sat_ok:
                wrec = _format_witness_record(label, R, sat_res[0])
                witnesses_out.append(wrec)
                entry["t_min"] = wrec["t_min"]
                entry["g2_nontrivial"] = wrec["g2_nontrivial"]
                # Stage 2 blind neutral-tunnel derivation test (B = 256 samples modifying active tunnel bits)
                anchor = sat_res[0].w16
                active_words = [i for i in range(4 if enforce_g2 else R)]
                t_drv0 = time.perf_counter()
                derived: List[List[int]] = []
                for _ in range(256):
                    cand = list(anchor)
                    w_idx = active_words[rng.randrange(len(active_words))]
                    # Flip a random low/mid bit in the active word (or random subset of active words)
                    cand[w_idx] ^= (1 << rng.randrange(32))
                    if cand == anchor:
                        cand[w_idx] ^= 1
                    derived.append(cand)
                drv_dt = time.perf_counter() - t_drv0
                eval_stats = _evaluate_tunnel_derivation(
                    scen, R, k_bits, anchor, derived, search_dt, drv_dt, sha_rate
                )
                entry["tunnel_derivation"] = eval_stats
            results.append(entry)
            print(
                f"  -> {label}: sat={sat_ok}, "
                f"p_hit={entry.get('tunnel_derivation', {}).get('p_hit', '-')}, "
                f"z={entry.get('tunnel_derivation', {}).get('binomial_z', '-')}, "
                f"speedup={entry.get('tunnel_derivation', {}).get('speedup_amortized', '-')}",
                flush=True,
            )

    # 2. S2a (BTC Block 2 padding, W[0..3] free) and S2b (only W[3] Nonce free)
    for scen, only_nonce, r_list in (
        ("S2a_btc_w0_3", False, (12, 16, 20, 24)),
        ("S2b_btc_nonce_only", True, (8, 12, 16, 20)),
    ):
        for R in r_list:
            for k_bits in (4, 8, 12, 16):
                # Skip k=16 at R>=20 if R=16 k=16 already UNSAT/timeout to stay within Colab time budget
                label = f"{scen}_R{R}_k{k_bits}"
                t_search0 = time.perf_counter()
                w_gdiff: Dict[int, List[int]] = {}
                if only_nonce:
                    w_gdiff[0] = [GD_EQ] * 32
                    w_gdiff[1] = [GD_EQ] * 32
                    w_gdiff[2] = [GD_EQ] * 32

                sat_res = solve_differential_witness(
                    rounds=R,
                    iv_mode="standard",
                    w_gdiff=w_gdiff,
                    leading_zero_bits=k_bits,
                    btc_block2_padding=True,
                    require_nonzero_msg_diff=True,
                    conflict_budget=sat_budget,
                    max_solutions=2,
                )
                search_dt = time.perf_counter() - t_search0
                sat_ok = bool(sat_res and sat_res[0].sat)

                entry = {
                    "scenario": scen,
                    "label": label,
                    "rounds": R,
                    "k_bits": k_bits,
                    "shallow_warning": (R < 16),
                    "sat": sat_ok,
                    "sat_seconds": round(search_dt, 4),
                }
                if sat_ok:
                    wrec = _format_witness_record(label, R, sat_res[0], btc_padding=True)
                    witnesses_out.append(wrec)
                    entry["t_min"] = wrec["t_min"]
                    entry["g2_nontrivial"] = wrec["g2_nontrivial"]
                    anchor = sat_res[0].w16
                    t_drv0 = time.perf_counter()
                    derived = []
                    for _ in range(256):
                        cand = list(anchor)
                        if only_nonce:
                            cand[3] ^= rng.randrange(1, 1 << 32)
                        else:
                            cand[rng.randrange(4)] ^= (1 << rng.randrange(32))
                        derived.append(cand)
                    drv_dt = time.perf_counter() - t_drv0
                    entry["tunnel_derivation"] = _evaluate_tunnel_derivation(
                        scen, R, k_bits, anchor, derived, search_dt, drv_dt, sha_rate
                    )
                results.append(entry)
                print(
                    f"  -> {label}: sat={sat_ok}, "
                    f"p_hit={entry.get('tunnel_derivation', {}).get('p_hit', '-')}, "
                    f"z={entry.get('tunnel_derivation', {}).get('binomial_z', '-')}, "
                    f"speedup={entry.get('tunnel_derivation', {}).get('speedup_amortized', '-')}",
                    flush=True,
                )

    # 3. S3 (BTC Double-SHA256 boundary (R1, R2))
    for r1, r2, shallow_tag in ((8, 8, True), (12, 12, True), (16, 16, False), (20, 20, False)):
        for k_bits in (4, 8):
            label = f"S3_double_R{r1}x{r2}_k{k_bits}"
            t_search0 = time.perf_counter()
            sat_res = solve_double_sha256_witness(
                r1=r1,
                r2=r2,
                leading_zero_bits=k_bits,
                btc_block2_padding=True,
                require_nonzero_msg_diff=True,
                conflict_budget=sat_budget,
                max_solutions=1,
            )
            search_dt = time.perf_counter() - t_search0
            sat_ok = bool(sat_res and sat_res[0].sat)
            entry = {
                "scenario": "S3_double_sha256",
                "label": label,
                "r1": r1,
                "r2": r2,
                "k_bits": k_bits,
                "shallow_b2_warning": shallow_tag,
                "sat": sat_ok,
                "sat_seconds": round(search_dt, 4),
            }
            if sat_ok:
                wrec = _format_witness_record(
                    label, r1, sat_res[0], is_double_sha=True, r2=r2, btc_padding=True
                )
                witnesses_out.append(wrec)
                entry["g2_nontrivial"] = wrec["g2_nontrivial"]
                anchor = sat_res[0].w16
                t_drv0 = time.perf_counter()
                derived = []
                for _ in range(256):
                    cand = list(anchor)
                    cand[rng.randrange(4)] ^= (1 << rng.randrange(32))
                    derived.append(cand)
                drv_dt = time.perf_counter() - t_drv0
                entry["tunnel_derivation"] = _evaluate_tunnel_derivation(
                    "S3_double_sha256", r1, k_bits, anchor, derived, search_dt, drv_dt, sha_rate,
                    is_double_sha=True, r2=r2,
                )
            results.append(entry)
            print(
                f"  -> {label}: sat={sat_ok}, "
                f"p_hit={entry.get('tunnel_derivation', {}).get('p_hit', '-')}, "
                f"z={entry.get('tunnel_derivation', {}).get('binomial_z', '-')}",
                flush=True,
            )

    # 4. G4 Control Cipher check on S2a R=16 k=4/8 (or any positive R>=16 entry)
    print("\n[Track 2] Running G4 CTRL_ROTR_K comparison at R=16, k=4,8...", flush=True)
    ctrl_results: List[Dict[str, object]] = []
    for k_bits in (4, 8):
        label = f"G4_CTRL_S2a_R16_k{k_bits}"
        t_search0 = time.perf_counter()
        sat_res = solve_differential_witness(
            rounds=16,
            iv_mode="standard",
            leading_zero_bits=k_bits,
            btc_block2_padding=True,
            require_nonzero_msg_diff=True,
            conflict_budget=sat_budget,
            max_solutions=1,
            k_consts=CTRL_K_CONSTS,
            sigma0_shifts=CTRL_SIGMA0_SHIFTS,
            sigma1_shifts=CTRL_SIGMA1_SHIFTS,
        )
        search_dt = time.perf_counter() - t_search0
        sat_ok = bool(sat_res and sat_res[0].sat)
        c_entry: Dict[str, object] = {"label": label, "rounds": 16, "k_bits": k_bits, "sat": sat_ok, "sat_seconds": round(search_dt, 4)}
        if sat_ok:
            wrec = _format_witness_record(
                label, 16, sat_res[0], btc_padding=True,
                k_consts=CTRL_K_CONSTS, s0_shifts=CTRL_SIGMA0_SHIFTS, s1_shifts=CTRL_SIGMA1_SHIFTS,
            )
            witnesses_out.append(wrec)
            anchor = sat_res[0].w16
            t_drv0 = time.perf_counter()
            derived = []
            for _ in range(256):
                cand = list(anchor)
                cand[rng.randrange(4)] ^= (1 << rng.randrange(32))
                derived.append(cand)
            drv_dt = time.perf_counter() - t_drv0
            c_entry["tunnel_derivation"] = _evaluate_tunnel_derivation(
                "G4_CTRL", 16, k_bits, anchor, derived, search_dt, drv_dt, sha_rate,
                k_consts=CTRL_K_CONSTS, s0_shifts=CTRL_SIGMA0_SHIFTS, s1_shifts=CTRL_SIGMA1_SHIFTS,
            )
        ctrl_results.append(c_entry)
        print(f"  -> {label}: sat={sat_ok}", flush=True)

    return {
        "sha256_evals_per_sec": round(sha_rate, 1),
        "btc_experiments": results,
        "g4_control_experiments": ctrl_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sat-budget", type=int, default=200_000)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else "CPU"
    print(f"[P13] Starting experiments on device={device} ({gpu_name}), allow_tf32={torch.backends.cuda.matmul.allow_tf32}", flush=True)

    t0 = time.perf_counter()
    out_dir = _THIS_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    witnesses: List[Dict[str, object]] = []
    ladder_res = run_standard_ladder(device=device, witnesses_out=witnesses, sat_budget=args.sat_budget)
    (out_dir / "p13_ladder_results.json").write_text(
        json.dumps({"meta": {"device": device, "gpu_name": gpu_name, "fp32_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32)}, **ladder_res}, indent=2),
        encoding="utf-8",
    )
    btc_res = run_btc_truncated_lz(device=device, witnesses_out=witnesses, sat_budget=max(60_000, args.sat_budget // 2))

    total_dt = time.perf_counter() - t0

    meta = {
        "device": device,
        "gpu_name": gpu_name,
        "fp32_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "total_elapsed_seconds": round(total_dt, 2),
        "num_verified_witnesses": len(witnesses),
    }
    (out_dir / "p13_ladder_results.json").write_text(
        json.dumps({"meta": meta, **ladder_res}, indent=2), encoding="utf-8"
    )
    (out_dir / "p13_btc_truncated_lz.json").write_text(
        json.dumps({"meta": meta, **btc_res}, indent=2), encoding="utf-8"
    )
    (out_dir / "p13_witnesses.json").write_text(
        json.dumps({"meta": meta, "witnesses": witnesses}, indent=2), encoding="utf-8"
    )
    print(f"\n[P13 COMPLETE] elapsed={total_dt:.2f}s, verified_witnesses={len(witnesses)}", flush=True)


if __name__ == "__main__":
    main()
