"""P11 full experiment runner (pure FP32, zero local GPU when --cpu passed, atomic JSON + --resume):
- Part 1: Exact chi=2 MPO ADD / chi=1 XOR verification & gate-by-gate MPS-TEBD ROTR entanglement isolation
- Part 2: Exact solution-manifold entanglement spectrum of REAL 256-bit SHA-256 (R=1..64) vs rand_ctrl vs Page curve
- Part 3: Neural Orthogonal Disentangler U_theta (3A target-conditioned local film + 3B amortized cross-midstate)
- Part 4: Lyapunov Energy Funnel ablation (endpoint-only V(x,y*) vs intermediate-state trajectory V(S_mid, y*))
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sha256_ref import MASK32, SHA256_IV
from v11_discovery.p11_tensor_network.disentangler_and_funnel import (
    LyapunovFunnelNet,
    NeuralDisentanglerCircuit,
    evaluate_mps_born_enrichment,
    train_neural_disentangler,
)
from v11_discovery.p11_tensor_network.mpo_arx import (
    analyze_state_entanglement,
    apply_mpo_to_mps,
    apply_rotr_y_mps,
    apply_swap_xy_local_mps,
    build_reversible_2reg_add_mpo,
    build_reversible_2reg_xor_mpo,
    build_sha256_solution_state,
    enforce_fp32_lock,
    make_cipher_profile,
    mps_norm,
    product_state_mps,
    sha256_run_rounds,
    verify_exact_mpo_primitives,
    verify_sha256_exact_64,
)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _words_to_bits(words: torch.Tensor, n_bits: int = 32) -> torch.Tensor:
    shifts = torch.arange(n_bits, dtype=torch.int64, device=words.device)
    bits = ((words.unsqueeze(-1) >> shifts) & 1).to(torch.float32)
    return bits.reshape(words.shape[0], -1)


# ==============================================================================
# PART 1B: Gate-by-Gate MPS-TEBD Isolation of ROTR-Driven Entanglement
# ==============================================================================

def run_mps_tebd_rotr_ablation(
    n_bits: int,
    rot_mode: str,
    rot_val: int,
    rounds: int,
    chi_max: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Run gate-by-gate MPS-TEBD on an n-bit 2-register reversible ARX Feistel network.

    Initial state: x in uniform superposition |+>^{otimes n}, y = |0>^{otimes n} (analogous to
    unknown nonce x and known midstate y; exact product state chi=1 at r=0).
    Each round: (x, y) -> (x, x + y) [chi=2 MPO] -> ROTR_y(rot_val) [SWAP chain]
                -> (x, x ^ y) [chi=1 MPO] -> local branch swap (x, y) -> (y, x) [chi=1].
    """
    t0 = time.perf_counter()
    # Build initial chi=1 MPS: x uniform superposition, y = 0
    v_init = torch.tensor([1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0), 0.0], dtype=torch.float32, device=device)
    mps = product_state_mps(n_bits, [v_init.clone() for _ in range(n_bits)])

    mpo_add = build_reversible_2reg_add_mpo(n_bits, device)
    mpo_xor = build_reversible_2reg_xor_mpo(n_bits, device)

    per_round: List[Dict[str, Any]] = []
    cum_trunc_err = 0.0
    for r in range(1, rounds + 1):
        mps, err_add, chi_add = apply_mpo_to_mps(mps, mpo_add, chi_max=chi_max)
        mps, err_rot = apply_rotr_y_mps(mps, rot=rot_val, chi_max=chi_max, device=device)
        mps, err_xor, chi_xor = apply_mpo_to_mps(mps, mpo_xor, chi_max=chi_max)
        mps = apply_swap_xy_local_mps(mps, device)
        cum_trunc_err += err_add + err_rot + err_xor
        max_bond = max(int(t.shape[2]) for t in mps[:-1]) if n_bits > 1 else 1
        norm_now = mps_norm(mps)
        per_round.append(
            {
                "round": r,
                "max_bond_dim": max_bond,
                "round_trunc_err_sq": round(err_add + err_rot + err_xor, 6),
                "cum_trunc_err_sq": round(cum_trunc_err, 6),
                "mps_norm": round(norm_now, 6),
            }
        )
    return {
        "n_bits": n_bits,
        "rot_mode": rot_mode,
        "rot_val": rot_val,
        "chi_max": chi_max,
        "rounds": rounds,
        "per_round": per_round,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


# ==============================================================================
# PART 2: Exact Solution-Manifold Entanglement of REAL 256-bit SHA-256
# ==============================================================================

def _sample_midstates(n_midstates: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    s0s = torch.randint(0, 1 << 32, (n_midstates, 8), dtype=torch.int64, generator=g)
    w16s = torch.randint(0, 1 << 32, (n_midstates, 16), dtype=torch.int64, generator=g)
    # First midstate is always standard SHA256_IV for canonical anchor
    s0s[0] = torch.tensor(SHA256_IV, dtype=torch.int64)
    return s0s, w16s


def eval_sha256_solution_entanglement(
    num_rounds: int,
    m_nonce_bits: int,
    k_zero_bits: int,
    cipher_kind: str,
    n_midstates: int,
    chi_list: Sequence[int],
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """Measure exact Schmidt spectrum & MPS fidelity of real SHA-256 BTC mining solution manifold."""
    t0 = time.perf_counter()
    s0s, w16s = _sample_midstates(n_midstates, seed=seed)
    prof = make_cipher_profile(ctrl=(cipher_kind == "rand_cipher_ctrl"), seed=seed + 101)
    g_oracle = torch.Generator(device=device).manual_seed(seed + num_rounds * 997 + k_zero_bits)

    mid_vns: List[float] = []
    max_vns: List[float] = []
    chi99s: List[int] = []
    n_valids: List[int] = []
    fid_accum: Dict[str, List[float]] = {str(c): [] for c in chi_list}

    for i in range(n_midstates):
        if cipher_kind == "random_oracle_page":
            # Exact cardinality-matched random Boolean function (theoretical Page-curve ceiling)
            n_states = 1 << m_nonce_bits
            n_target = max(1, n_states >> k_zero_bits)
            perm = torch.randperm(n_states, device=device, generator=g_oracle)[:n_target]
            valid = torch.zeros(n_states, dtype=torch.float32, device=device)
            valid[perm] = 1.0
            psi = valid / valid.norm()
            nv = n_target
        else:
            psi, nv = build_sha256_solution_state(
                s0s[i], w16s[i], num_rounds, m_nonce_bits, k_zero_bits, prof, device
            )
        ent = analyze_state_entanglement(psi, n_qubits=m_nonce_bits, chi_list=chi_list)
        mid_vns.append(ent["mid_vn_entropy_bits"])
        max_vns.append(ent["max_vn_entropy_bits"])
        chi99s.append(ent["mid_chi99"])
        n_valids.append(nv)
        for c in chi_list:
            fid_accum[str(c)].append(ent["fidelity_by_chi"][str(c)])

    def _mean(xs: Sequence[float]) -> float:
        return round(sum(xs) / max(1, len(xs)), 5)

    def _std(xs: Sequence[float]) -> float:
        m = sum(xs) / max(1, len(xs))
        return round(math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs))), 5)

    return {
        "num_rounds": num_rounds,
        "m_nonce_bits": m_nonce_bits,
        "k_zero_bits": k_zero_bits,
        "cipher_kind": cipher_kind,
        "n_midstates": n_midstates,
        "mean_n_valid": round(sum(n_valids) / len(n_valids), 2),
        "mid_vn_mean": _mean(mid_vns),
        "mid_vn_std": _std(mid_vns),
        "max_vn_mean": _mean(max_vns),
        "mid_chi99_mean": round(sum(chi99s) / len(chi99s), 2),
        "fidelity_mean_by_chi": {str(c): _mean(fid_accum[str(c)]) for c in chi_list},
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


# ==============================================================================
# PART 3: Neural Disentangler (3A Target-Conditioned Local + 3B Cross-Midstate)
# ==============================================================================

def run_target_conditioned_disentangler(
    num_rounds: int,
    m_nonce_bits: int,
    k_zero_bits: int,
    cipher_kind: str,
    n_layers: int,
    steps: int,
    chi_list: Sequence[int],
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """Route 1 + Route 3: Fit U_theta(. | y*) specifically to one BTC target midstate."""
    t0 = time.perf_counter()
    s0s, w16s = _sample_midstates(2, seed=seed + 500)
    prof = make_cipher_profile(ctrl=(cipher_kind == "rand_cipher_ctrl"), seed=seed + 101)
    g_oracle = torch.Generator(device=device).manual_seed(seed + num_rounds * 313 + 17)

    if cipher_kind == "random_oracle_page":
        n_states = 1 << m_nonce_bits
        n_target = max(1, n_states >> k_zero_bits)
        perm = torch.randperm(n_states, device=device, generator=g_oracle)[:n_target]
        valid = torch.zeros(n_states, dtype=torch.float32, device=device)
        valid[perm] = 1.0
        psi = valid / valid.norm()
        n_valid = n_target
    else:
        psi, n_valid = build_sha256_solution_state(
            s0s[0], w16s[0], num_rounds, m_nonce_bits, k_zero_bits, prof, device
        )
        valid = (psi > 0.0).to(torch.float32)

    before = evaluate_mps_born_enrichment(psi, valid, m_nonce_bits, chi_list, disentangler=None)
    circuit, train_info = train_neural_disentangler(
        psi.unsqueeze(0),
        n_qubits=m_nonce_bits,
        n_layers=n_layers,
        steps=steps,
        lr=2e-2,
        seed=seed + num_rounds * 13,
    )
    after = evaluate_mps_born_enrichment(psi, valid, m_nonce_bits, chi_list, disentangler=circuit)

    return {
        "mode": "target_conditioned_local",
        "num_rounds": num_rounds,
        "m_nonce_bits": m_nonce_bits,
        "k_zero_bits": k_zero_bits,
        "cipher_kind": cipher_kind,
        "n_valid": int(n_valid),
        "train_info": train_info,
        "before": before,
        "after": after,
        "delta_mid_vn_bits": round(
            after["entanglement"]["mid_vn_entropy_bits"] - before["entanglement"]["mid_vn_entropy_bits"], 5
        ),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def run_amortized_cross_midstate_disentangler(
    num_rounds: int,
    m_nonce_bits: int,
    k_zero_bits: int,
    cipher_kind: str,
    n_train: int,
    n_test: int,
    n_layers: int,
    steps: int,
    chi_list: Sequence[int],
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """Train ONE shared disentangler U_theta on n_train midstates; test on n_test UNSEEN midstates."""
    t0 = time.perf_counter()
    s0s, w16s = _sample_midstates(n_train + n_test, seed=seed + 900)
    prof = make_cipher_profile(ctrl=(cipher_kind == "rand_cipher_ctrl"), seed=seed + 101)

    train_psis = []
    for i in range(n_train):
        p, _ = build_sha256_solution_state(s0s[i], w16s[i], num_rounds, m_nonce_bits, k_zero_bits, prof, device)
        train_psis.append(p)
    psi_train_batch = torch.stack(train_psis, dim=0)

    circuit, train_info = train_neural_disentangler(
        psi_train_batch,
        n_qubits=m_nonce_bits,
        n_layers=n_layers,
        steps=steps,
        lr=1.5e-2,
        seed=seed + num_rounds * 29,
    )

    vn_before, vn_after = [], []
    chi99_before, chi99_after = [], []
    enrich8_before, enrich8_after = [], []
    fid8_before, fid8_after = [], []

    for j in range(n_train, n_train + n_test):
        p_test, _ = build_sha256_solution_state(s0s[j], w16s[j], num_rounds, m_nonce_bits, k_zero_bits, prof, device)
        v_mask = (p_test > 0.0).to(torch.float32)
        b_res = evaluate_mps_born_enrichment(p_test, v_mask, m_nonce_bits, chi_list, disentangler=None)
        a_res = evaluate_mps_born_enrichment(p_test, v_mask, m_nonce_bits, chi_list, disentangler=circuit)
        vn_before.append(b_res["entanglement"]["mid_vn_entropy_bits"])
        vn_after.append(a_res["entanglement"]["mid_vn_entropy_bits"])
        chi99_before.append(b_res["entanglement"]["mid_chi99"])
        chi99_after.append(a_res["entanglement"]["mid_chi99"])
        ck = str(8 if 8 in chi_list else chi_list[0])
        enrich8_before.append(b_res["enrichment_by_chi"][ck])
        enrich8_after.append(a_res["enrichment_by_chi"][ck])
        fid8_before.append(b_res["entanglement"]["fidelity_by_chi"][ck])
        fid8_after.append(a_res["entanglement"]["fidelity_by_chi"][ck])

    def _m(xs: Sequence[float]) -> float:
        return round(sum(xs) / max(1, len(xs)), 5)

    return {
        "mode": "amortized_cross_midstate",
        "num_rounds": num_rounds,
        "m_nonce_bits": m_nonce_bits,
        "k_zero_bits": k_zero_bits,
        "cipher_kind": cipher_kind,
        "n_train_midstates": n_train,
        "n_test_midstates": n_test,
        "train_info": train_info,
        "heldout_mid_vn_before": _m(vn_before),
        "heldout_mid_vn_after": _m(vn_after),
        "heldout_delta_vn": round(_m(vn_after) - _m(vn_before), 5),
        "heldout_chi99_before": _m(chi99_before),
        "heldout_chi99_after": _m(chi99_after),
        "heldout_fid_chi8_before": _m(fid8_before),
        "heldout_fid_chi8_after": _m(fid8_after),
        "heldout_enrichment_chi8_before": _m(enrich8_before),
        "heldout_enrichment_chi8_after": _m(enrich8_after),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


# ==============================================================================
# PART 4: Lyapunov Funnel Ablation (Endpoint-Only vs Intermediate-State Trajectory)
# ==============================================================================

def _flip_random_k_bits_words(
    words: torch.Tensor,
    n_flips: int,
    bit_span: int,
    gen: torch.Generator,
) -> torch.Tensor:
    """Flip exactly n_flips distinct bit positions within [0, bit_span) on (B, W) uint32 tensor."""
    B, W = words.shape
    out = words.clone()
    total_bits = min(bit_span, W * 32)
    # Sample random positions per row
    scores = torch.rand((B, total_bits), device=words.device, generator=gen)
    idx = torch.topk(scores, k=n_flips, dim=1).indices  # (B, n_flips)
    for f in range(n_flips):
        pos = idx[:, f]
        w_i = pos // 32
        b_i = pos % 32
        bit_mask = (torch.ones_like(b_i) << b_i) & MASK32
        out.scatter_add_(
            1,
            w_i.unsqueeze(1),
            torch.zeros((B, 1), dtype=torch.int64, device=words.device),
        )
        cur_w = out.gather(1, w_i.unsqueeze(1)).squeeze(1)
        out.scatter_(1, w_i.unsqueeze(1), (cur_w ^ bit_mask).unsqueeze(1))
    return out


def train_and_eval_lyapunov_funnel(
    num_rounds: int,
    funnel_mode: str,
    cipher_kind: str,
    steps: int,
    batch_size: int,
    eval_n: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """Compare Endpoint-Only V(x, y*) vs Intermediate-State Trajectory V(S_mid, y*) on real SHA-256.

    - endpoint_only: candidate is 32-bit nonce W_0 (flipped 1 bit vs 4 bits away from true x*); target is A_R (32 bits) + E_R (32 bits) = 64 bits.
    - intermediate_traj: candidate is 256-bit intermediate state S_{R//2} (flipped 1 bit vs 4 bits away from true S_{R//2}*); target is A_R + E_R (64 bits).
    """
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    gen_tr = torch.Generator(device=device).manual_seed(seed + 19)
    gen_ev = torch.Generator(device=device).manual_seed(seed + 7777)

    prof = make_cipher_profile(ctrl=(cipher_kind == "rand_cipher_ctrl"), seed=seed + 101)
    mid_r = max(1, num_rounds // 2)
    s0_fixed = torch.tensor(SHA256_IV, dtype=torch.int64, device=device)

    is_traj = funnel_mode == "intermediate_traj"
    cand_bits_dim = 256 if is_traj else 32
    target_bits_dim = 64  # A_R and E_R
    model = LyapunovFunnelNet(in_bits=cand_bits_dim + target_bits_dim, hidden=384, n_blocks=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=1e-4)

    def _sample_batch(B: int, gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s0 = s0_fixed.unsqueeze(0).expand(B, -1)
        w16 = torch.zeros((B, 16), dtype=torch.int64, device=device)
        w16[:, 0] = torch.randint(0, 1 << 32, (B,), dtype=torch.int64, device=device, generator=gen)
        s_end, s_mid = sha256_run_rounds(s0, w16, num_rounds=num_rounds, prof=prof, return_mid_round=mid_r)
        assert s_mid is not None
        tgt = torch.stack([s_end[:, 0], s_end[:, 4]], dim=1)  # (B, 2) -> 64 bits
        tgt_bits = _words_to_bits(tgt, 32)
        if is_traj:
            true_words = s_mid  # (B, 8) -> 256 bits
            span = 256
        else:
            true_words = w16[:, :1]  # (B, 1) -> 32 bits
            span = 32
        closer_words = _flip_random_k_bits_words(true_words, n_flips=1, bit_span=span, gen=gen)
        farther_words = _flip_random_k_bits_words(true_words, n_flips=4, bit_span=span, gen=gen)
        return (
            _words_to_bits(true_words, 32),
            _words_to_bits(closer_words, 32),
            _words_to_bits(farther_words, 32),
            tgt_bits,
        )

    model.train()
    for _ in range(steps):
        b_true, b_close, b_far, b_tgt = _sample_batch(batch_size, gen_tr)
        v_true = model(torch.cat([b_true, b_tgt], dim=-1))
        v_close = model(torch.cat([b_close, b_tgt], dim=-1))
        v_far = model(torch.cat([b_far, b_tgt], dim=-1))
        # Margin pairwise ranking loss: V(true) < V(1-flip) < V(4-flip)
        loss = (
            F.softplus(0.5 + v_true - v_close).mean()
            + F.softplus(0.5 + v_close - v_far).mean()
            + 0.5 * F.softplus(1.0 + v_true - v_far).mean()
            + 1e-3 * (v_true.pow(2).mean() + v_far.pow(2).mean())
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()

    # Fresh held-out evaluation
    model.eval()
    with torch.no_grad():
        b_true, b_close, b_far, b_tgt = _sample_batch(eval_n, gen_ev)
        v_true = model(torch.cat([b_true, b_tgt], dim=-1))
        v_close = model(torch.cat([b_close, b_tgt], dim=-1))
        v_far = model(torch.cat([b_far, b_tgt], dim=-1))

        acc_true_vs_close = float((v_true < v_close).to(torch.float32).mean().item())
        acc_close_vs_far = float((v_close < v_far).to(torch.float32).mean().item())
        acc_true_vs_far = float((v_true < v_far).to(torch.float32).mean().item())
        z_close_vs_far = (acc_close_vs_far - 0.5) / math.sqrt(0.25 / eval_n)

        # Greedy 1-bit flip descent test on subset (256 instances):
        # Start 1 bit away from true solution; test all 32 candidate single-bit flips (in first word)
        n_desc = min(256, eval_n)
        # Re-sample fresh 1-flip perturbations strictly inside the first 32 bits so greedy tests all 32 flips
        s0_d = s0_fixed.unsqueeze(0).expand(n_desc, -1)
        w16_d = torch.zeros((n_desc, 16), dtype=torch.int64, device=device)
        w16_d[:, 0] = torch.randint(0, 1 << 32, (n_desc,), dtype=torch.int64, device=device, generator=gen_ev)
        s_end_d, s_mid_d = sha256_run_rounds(s0_d, w16_d, num_rounds=num_rounds, prof=prof, return_mid_round=mid_r)
        assert s_mid_d is not None
        tgt_d = _words_to_bits(torch.stack([s_end_d[:, 0], s_end_d[:, 4]], dim=1), 32)
        base_bits = _words_to_bits(s_mid_d if is_traj else w16_d[:, :1], 32)
        # Flip 1 known random bit in [0, 32)
        flipped_bit = torch.randint(0, 32, (n_desc,), device=device, generator=gen_ev)
        start_bits = base_bits.clone()
        start_bits[torch.arange(n_desc, device=device), flipped_bit] = (
            1.0 - start_bits[torch.arange(n_desc, device=device), flipped_bit]
        )
        # Score all 32 single-bit neighbors
        neigh = start_bits.unsqueeze(1).expand(n_desc, 32, -1).clone()
        ar32 = torch.arange(32, device=device)
        neigh[:, ar32, ar32] = 1.0 - neigh[:, ar32, ar32]
        tgt_rep = tgt_d.unsqueeze(1).expand(n_desc, 32, -1)
        v_neigh = model(torch.cat([neigh, tgt_rep], dim=-1).reshape(n_desc * 32, -1)).reshape(n_desc, 32)
        best_flip = torch.argmin(v_neigh, dim=1)
        greedy_exact_recover_rate = float((best_flip == flipped_bit).to(torch.float32).mean().item())
        # Random guessing among 32 neighbors has expected recovery rate 1/32 = 0.03125
        greedy_enrichment = greedy_exact_recover_rate / (1.0 / 32.0)

    return {
        "num_rounds": num_rounds,
        "mid_round": mid_r if is_traj else 0,
        "funnel_mode": funnel_mode,
        "cipher_kind": cipher_kind,
        "n_params": sum(p.numel() for p in model.parameters()),
        "rank_acc_true_vs_1flip": round(acc_true_vs_close, 5),
        "rank_acc_1flip_vs_4flip": round(acc_close_vs_far, 5),
        "rank_acc_true_vs_4flip": round(acc_true_vs_far, 5),
        "z_1flip_vs_4flip": round(float(z_close_vs_far), 2),
        "greedy_1bit_recover_rate": round(greedy_exact_recover_rate, 5),
        "greedy_enrichment_vs_random": round(greedy_enrichment, 3),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-SHA V11 P11 Tensor Network & Lyapunov Funnel")
    p.add_argument("--out_dir", type=str, default="v11_discovery/p11_tensor_network/results")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=20261125)
    args = p.parse_args()

    fp32_info = enforce_fp32_lock()
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "p11_results.json"

    sha_diff = verify_sha256_exact_64(device)
    mpo_checks = verify_exact_mpo_primitives(device)
    # Verify disentangler orthogonality on tiny 8-qubit test
    test_circ = NeuralDisentanglerCircuit(n_qubits=8, n_layers=3).to(device)
    ortho_err = test_circ.max_orthogonality_error()
    assert sha_diff == 0, f"SHA-256 64-round mismatch: {sha_diff}"
    for nb, info in mpo_checks.items():
        assert info["bit_mismatches"] == 0, f"MPO mismatch at n={nb}: {info}"
        assert info["max_inverse_overlap_err"] < 1e-4, f"MPO inverse error at n={nb}: {info}"
    assert ortho_err < 1e-4, f"Disentangler ortho err too large: {ortho_err}"

    ref_checks = {
        "sha256_64r_ref_xor_diff": sha_diff,
        "mpo_exact_checks": mpo_checks,
        "disentangler_ortho_max_err": round(ortho_err, 7),
    }
    print(f"[P11 Init] device={device} fp32={fp32_info} sha_diff={sha_diff} ortho_err={ortho_err:.2e}", flush=True)

    payload: Dict[str, Any] = {
        "experiment": "P11_Tensor_Network_Disentangler_And_Lyapunov_Funnel",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "fp32_lock": fp32_info,
        "reference_verification": ref_checks,
        "status": "running",
        "part_1_mpo_tebd": [],
        "part_2_sha256_entanglement": [],
        "part_3a_local_disentangler": [],
        "part_3b_amortized_disentangler": [],
        "part_4_lyapunov_funnel": [],
    }
    if args.resume and res_path.exists():
        try:
            prev = json.loads(res_path.read_text(encoding="utf-8"))
            for key in (
                "part_1_mpo_tebd",
                "part_2_sha256_entanglement",
                "part_3a_local_disentangler",
                "part_3b_amortized_disentangler",
                "part_4_lyapunov_funnel",
            ):
                if key in prev and isinstance(prev[key], list):
                    payload[key] = prev[key]
            print(
                f"[P11 Resume] P1={len(payload['part_1_mpo_tebd'])} P2={len(payload['part_2_sha256_entanglement'])} "
                f"P3A={len(payload['part_3a_local_disentangler'])} P3B={len(payload['part_3b_amortized_disentangler'])} "
                f"P4={len(payload['part_4_lyapunov_funnel'])}",
                flush=True,
            )
        except Exception as exc:
            print(f"[P11 Resume] load failed ({exc}), starting fresh", flush=True)
    _atomic_write_json(res_path, payload)

    if args.smoke:
        r1 = run_mps_tebd_rotr_ablation(6, "rotr_sha_like", 2, 3, 8, device)
        r2 = eval_sha256_solution_entanglement(2, 10, 4, "sha256_real", 2, (2, 4, 8), device, args.seed)
        r3a = run_target_conditioned_disentangler(2, 10, 4, "sha256_real", 2, 15, (2, 4, 8), device, args.seed)
        r3b = run_amortized_cross_midstate_disentangler(2, 10, 4, "sha256_real", 4, 2, 2, 15, (2, 4, 8), device, args.seed)
        r4e = train_and_eval_lyapunov_funnel(2, "endpoint_only", "sha256_real", 15, 64, 64, device, args.seed)
        r4t = train_and_eval_lyapunov_funnel(2, "intermediate_traj", "sha256_real", 15, 64, 64, device, args.seed)
        payload["part_1_mpo_tebd"] = [r1]
        payload["part_2_sha256_entanglement"] = [r2]
        payload["part_3a_local_disentangler"] = [r3a]
        payload["part_3b_amortized_disentangler"] = [r3b]
        payload["part_4_lyapunov_funnel"] = [r4e, r4t]
        payload["status"] = "smoke_passed"
        _atomic_write_json(res_path, payload)
        print("[P11 Smoke] ALL PASSED", flush=True)
        return

    done_1 = {(r["n_bits"], r["rot_mode"], r["chi_max"]) for r in payload["part_1_mpo_tebd"]}
    done_2 = {(r["num_rounds"], r["k_zero_bits"], r["cipher_kind"]) for r in payload["part_2_sha256_entanglement"]}
    done_3a = {(r["num_rounds"], r["cipher_kind"]) for r in payload["part_3a_local_disentangler"]}
    done_3b = {(r["num_rounds"], r["cipher_kind"]) for r in payload["part_3b_amortized_disentangler"]}
    done_4 = {(r["num_rounds"], r["funnel_mode"], r["cipher_kind"]) for r in payload["part_4_lyapunov_funnel"]}

    # ================= PART 1: Gate-by-Gate MPS-TEBD ROTR Entanglement Isolation =================
    tebd_specs = [
        (8, "rotr_zero", 0, 6, 32),
        (8, "rotr_sha_like", 2, 6, 16),
        (8, "rotr_sha_like", 2, 6, 32),
        (8, "rotr_random_ctrl", 3, 6, 32),
        (12, "rotr_zero", 0, 6, 32),
        (12, "rotr_sha_like", 3, 6, 16),
        (12, "rotr_sha_like", 3, 6, 32),
        (12, "rotr_random_ctrl", 5, 6, 32),
    ]
    for nb, rmode, rval, rds, cmax in tebd_specs:
        if (nb, rmode, cmax) in done_1:
            continue
        row = run_mps_tebd_rotr_ablation(nb, rmode, rval, rds, cmax, device)
        payload["part_1_mpo_tebd"].append(row)
        done_1.add((nb, rmode, cmax))
        last = row["per_round"][-1]
        print(
            f"[Part 1 TEBD n={nb:2d} {rmode:16s} chi={cmax:2d}] R={rds}: "
            f"max_bond={last['max_bond_dim']:2d} cum_err={last['cum_trunc_err_sq']:.5f} "
            f"norm={last['mps_norm']:.5f} ({row['elapsed_sec']}s)",
            flush=True,
        )
        _atomic_write_json(res_path, payload)

    # ================= PART 2: Exact Solution-Manifold Entanglement of REAL 256-bit SHA-256 =================
    chi_list_p2 = (2, 4, 8, 16, 32, 64)
    p2_rounds = [1, 2, 3, 4, 5, 6, 8, 12, 16, 32, 64]
    for kz in (4, 8):
        for rds in p2_rounds:
            for ckind in ("sha256_real", "rand_cipher_ctrl", "random_oracle_page"):
                if (rds, kz, ckind) in done_2:
                    continue
                row = eval_sha256_solution_entanglement(
                    num_rounds=rds,
                    m_nonce_bits=16,
                    k_zero_bits=kz,
                    cipher_kind=ckind,
                    n_midstates=8,
                    chi_list=chi_list_p2,
                    device=device,
                    seed=args.seed + kz * 1000 + rds,
                )
                payload["part_2_sha256_entanglement"].append(row)
                done_2.add((rds, kz, ckind))
                print(
                    f"[Part 2 k={kz} R={rds:2d} {ckind:18s}] "
                    f"S_vN={row['mid_vn_mean']:.4f}+-{row['mid_vn_std']:.4f} "
                    f"chi99={row['mid_chi99_mean']:6.1f} F(8)={row['fidelity_mean_by_chi']['8']:.4f} "
                    f"F(32)={row['fidelity_mean_by_chi']['32']:.4f} ({row['elapsed_sec']}s)",
                    flush=True,
                )
                _atomic_write_json(res_path, payload)

    # ================= PART 3A: Target-Conditioned Local Neural Disentangler =================
    chi_list_p3 = (4, 8, 16, 32)
    for rds in (1, 2, 3, 4, 5, 6, 8, 16):
        for ckind in ("sha256_real", "rand_cipher_ctrl", "random_oracle_page"):
            if (rds, ckind) in done_3a:
                continue
            row = run_target_conditioned_disentangler(
                num_rounds=rds,
                m_nonce_bits=16,
                k_zero_bits=6,
                cipher_kind=ckind,
                n_layers=6,
                steps=250,
                chi_list=chi_list_p3,
                device=device,
                seed=args.seed + 30000 + rds,
            )
            payload["part_3a_local_disentangler"].append(row)
            done_3a.add((rds, ckind))
            b_s = row["before"]["entanglement"]["mid_vn_entropy_bits"]
            a_s = row["after"]["entanglement"]["mid_vn_entropy_bits"]
            b_e8 = row["before"]["enrichment_by_chi"]["8"]
            a_e8 = row["after"]["enrichment_by_chi"]["8"]
            print(
                f"[Part 3A Local R={rds:2d} {ckind:18s}] "
                f"S_vN: {b_s:.4f} -> {a_s:.4f} (d={row['delta_mid_vn_bits']:+.4f}) | "
                f"E(chi=8): {b_e8:.2f}x -> {a_e8:.2f}x ({row['elapsed_sec']}s)",
                flush=True,
            )
            _atomic_write_json(res_path, payload)

    # ================= PART 3B: Cross-Midstate Amortized Neural Disentangler =================
    for rds in (1, 2, 3, 4, 6, 8):
        for ckind in ("sha256_real", "rand_cipher_ctrl"):
            if (rds, ckind) in done_3b:
                continue
            row = run_amortized_cross_midstate_disentangler(
                num_rounds=rds,
                m_nonce_bits=16,
                k_zero_bits=6,
                cipher_kind=ckind,
                n_train=48,
                n_test=24,
                n_layers=6,
                steps=300,
                chi_list=chi_list_p3,
                device=device,
                seed=args.seed + 40000 + rds,
            )
            payload["part_3b_amortized_disentangler"].append(row)
            done_3b.add((rds, ckind))
            print(
                f"[Part 3B Amort R={rds:2d} {ckind:18s}] "
                f"HeldOut S_vN: {row['heldout_mid_vn_before']:.4f} -> {row['heldout_mid_vn_after']:.4f} "
                f"(d={row['heldout_delta_vn']:+.4f}) | E(8): {row['heldout_enrichment_chi8_before']:.2f}x -> "
                f"{row['heldout_enrichment_chi8_after']:.2f}x ({row['elapsed_sec']}s)",
                flush=True,
            )
            _atomic_write_json(res_path, payload)

    # ================= PART 4: Lyapunov Funnel Ablation (Endpoint vs Intermediate Trajectory) =================
    for rds in (1, 2, 3, 4, 5, 6, 8):
        for fmode in ("endpoint_only", "intermediate_traj"):
            for ckind in ("sha256_real", "rand_cipher_ctrl"):
                if (rds, fmode, ckind) in done_4:
                    continue
                row = train_and_eval_lyapunov_funnel(
                    num_rounds=rds,
                    funnel_mode=fmode,
                    cipher_kind=ckind,
                    steps=1800,
                    batch_size=512,
                    eval_n=2048,
                    device=device,
                    seed=args.seed + 50000 + rds * 10 + (1 if fmode == "intermediate_traj" else 0),
                )
                payload["part_4_lyapunov_funnel"].append(row)
                done_4.add((rds, fmode, ckind))
                print(
                    f"[Part 4 Funnel R={rds:2d} {fmode:17s} {ckind:16s}] "
                    f"1v4={row['rank_acc_1flip_vs_4flip']:.4f} (z={row['z_1flip_vs_4flip']:+.2f}) "
                    f"0v4={row['rank_acc_true_vs_4flip']:.4f} greedy={row['greedy_1bit_recover_rate']:.4f} "
                    f"({row['greedy_enrichment_vs_random']:.2f}x) ({row['elapsed_sec']}s)",
                    flush=True,
                )
                _atomic_write_json(res_path, payload)

    payload["status"] = "completed"
    _atomic_write_json(res_path, payload)
    print("[P11 Complete] All 4 parts finished cleanly.", flush=True)


if __name__ == "__main__":
    main()
