"""Milestone 1 hard acceptance tests for Ouroboros-64 physical core & CDCL bridge."""

from __future__ import annotations

import hashlib
import numpy as np
import pytest
import torch

from v10_ouroboros64.ouroboros_bench.throughput_bench import run_throughput_benchmark
from v10_ouroboros64.ouroboros_core.cnf_cdcl import CaDiCaLCDCLSolver
from v10_ouroboros64.ouroboros_core.engine import (
    C_PLANES,
    MASK32,
    SUPPORTED_ROUNDS,
    SYM_0,
    SYM_1,
    SYM_Q,
    TorchCrystalEngine,
    get_core,
    words_to_bits_np,
)


def _pad_block(msg: bytes) -> np.ndarray:
    buf = bytearray(msg) + b"\x80"
    while len(buf) < 56:
        buf.append(0)
    buf += (len(msg) * 8).to_bytes(8, "big")
    return np.array([[int.from_bytes(buf[i : i + 4], "big") for i in range(0, 64, 4)]], dtype=np.uint32)


def test_sha256_matches_hashlib() -> None:
    core = get_core()
    tc = TorchCrystalEngine("cuda" if torch.cuda.is_available() else "cpu")
    msgs = [b"", b"abc", b"Ouroboros-64 boundary welded crystal", b"X" * 55]
    for msg in msgs:
        w = _pad_block(msg)
        cr = core.compress_exact(w, rounds=64)
        got = b"".join(int(x).to_bytes(4, "big") for x in cr.H_target[0])
        exp = hashlib.sha256(msg).digest()
        assert got == exp, f"Hash mismatch for {msg!r}"

        tc_res = tc.compress_exact(torch.from_numpy(w.astype(np.int64)), rounds=64)
        got_t = b"".join(int(x).to_bytes(4, "big") for x in tc_res["H"][0].cpu().tolist())
        assert got_t == exp


@pytest.mark.parametrize("rounds", SUPPORTED_ROUNDS)
def test_defect_energy_iff_preimage(rounds: int) -> None:
    core = get_core()
    rng = np.random.default_rng(1000 + rounds)
    w0_15 = rng.integers(0, 1 << 32, size=(4, 16), dtype=np.uint32)
    cr = core.compress_exact(w0_15, rounds=rounds)

    # Exact preimage crystal has E_defect == 0
    ev0 = core.eval_crystal_batch(cr.H_target, cr.W, cr.S, cr.C, rounds=rounds)
    assert np.all(ev0["E"] == 0), f"Expected 0 defect energy, got {ev0['E']}"

    # Corrupt single bit in W, internal S, C, or H_target -> E > 0 strictly
    for kind in ("W", "S", "C", "S_bnd", "H", "wrong_preimage"):
        W, S, C, H = cr.W.copy(), cr.S.copy(), cr.C.copy(), cr.H_target.copy()
        if kind == "W":
            W[:, 0] ^= np.uint32(1 << 5)
        elif kind == "S":
            S[:, rounds // 2, 3] ^= np.uint32(1 << 17)
        elif kind == "C":
            C[:, min(17, rounds - 1), 6] ^= np.uint32(1 << 8)
        elif kind == "S_bnd":
            S[:, rounds, 7] ^= np.uint32(1)
        elif kind == "H":
            H[:, 2] ^= np.uint32(1 << 11)
        elif kind == "wrong_preimage":
            # Different valid preimage compressed without changing H -> boundary fails iff wrong digest
            w_other = w0_15.copy()
            w_other[:, 15] ^= np.uint32(1)
            cr2 = core.compress_exact(w_other, rounds=rounds)
            W, S, C = cr2.W, cr2.S, cr2.C
        ev = core.eval_crystal_batch(H, W, S, C, rounds=rounds)
        assert np.all(ev["E"] > 0), f"Kind={kind} at R={rounds} unexpectedly gave E=0: {ev['E']}"


def test_cpp_vs_torch_and_incremental_exact() -> None:
    core = get_core()
    tc = TorchCrystalEngine("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(777)
    R = 64
    B = 4
    w0_15 = rng.integers(0, 1 << 32, size=(B, 16), dtype=np.uint32)
    cr = core.compress_exact(w0_15, rounds=R)

    # Inject substantial multi-region defects while preserving welded boundaries
    W_b = cr.W.copy()
    S_b = cr.S.copy()
    C_b = cr.C.copy()
    W_b ^= rng.integers(0, 1 << 32, size=W_b.shape, dtype=np.uint32) & np.uint32(0x00100101)
    S_b[:, 1:R, :] ^= rng.integers(0, 1 << 32, size=(B, R - 1, 8), dtype=np.uint32) & np.uint32(0x01010001)
    C_b ^= rng.integers(0, 1 << 32, size=C_b.shape, dtype=np.uint32) & np.uint32(0x00000102)

    ev_cpp = core.eval_crystal_batch(cr.H_target, W_b, S_b, C_b, rounds=R, alpha=2, beta=3, gamma=1)
    ev_tc = tc.eval_crystal_batch(
        torch.from_numpy(cr.H_target.astype(np.int64)),
        torch.from_numpy(W_b.astype(np.int64)),
        torch.from_numpy(S_b.astype(np.int64)),
        torch.from_numpy(C_b.astype(np.int64)),
        rounds=R,
        alpha=2,
        beta=3,
        gamma=1,
    )
    assert np.array_equal(ev_cpp["E"], ev_tc["E"].cpu().numpy().astype(np.uint64))
    assert np.array_equal(ev_cpp["dS"], ev_tc["dS"].cpu().numpy().astype(np.uint64))
    assert np.array_equal(ev_cpp["dW"], ev_tc["dW"].cpu().numpy().astype(np.uint64))
    assert np.array_equal(ev_cpp["dC"], ev_tc["dC"].cpu().numpy().astype(np.uint64))

    # Compare incremental eval_cluster_flips_batch against full recomputation across 64 random multi-bit clusters
    K = 16
    M = 6
    clusters = np.full((B * K, M, 4), -1, dtype=np.int32)
    for bi in range(B):
        for ki in range(K):
            idx = bi * K + ki
            m_use = int(rng.integers(1, M + 1))
            # Pick unique sites so duplicate XOR doesn't self-cancel cleanly in naive manual check too
            seen = set()
            for mi in range(m_use):
                while True:
                    kind = int(rng.integers(0, 3))
                    t = int(rng.integers(1 if kind == 1 else 0, R))
                    word = int(rng.integers(0, 8)) if kind == 1 else (int(rng.integers(0, C_PLANES)) if kind == 2 else 0)
                    bit = int(rng.integers(0, 32))
                    key = (kind, t, word, bit)
                    if key not in seen:
                        seen.add(key)
                        clusters[idx, mi] = key
                        break

    dE, nE = core.eval_cluster_flips_batch(
        cr.H_target, W_b, S_b, C_b, clusters, rounds=R, alpha=2, beta=3, gamma=1
    )

    for bi in range(B):
        for ki in range(K):
            idx = bi * K + ki
            Wt, St, Ct = W_b[bi : bi + 1].copy(), S_b[bi : bi + 1].copy(), C_b[bi : bi + 1].copy()
            for mi in range(M):
                knd, tt, ww, bb = [int(x) for x in clusters[idx, mi]]
                if knd < 0:
                    continue
                if knd == 0:
                    Wt[0, tt] ^= np.uint32(1 << bb)
                elif knd == 1:
                    St[0, tt, ww] ^= np.uint32(1 << bb)
                else:
                    Ct[0, tt, ww] ^= np.uint32(1 << bb)
            ev_ref = core.eval_crystal_batch(cr.H_target[bi : bi + 1], Wt, St, Ct, rounds=R, alpha=2, beta=3, gamma=1)
            assert int(nE[idx]) == int(ev_ref["E"][0]), f"Inc mismatch at b={bi}, k={ki}"
            assert int(dE[idx]) == int(ev_ref["E"][0]) - int(ev_cpp["E"][bi])


def test_gf2_rref_and_bidirectional_prop() -> None:
    core = get_core()
    # 1. Standalone GF(2) 20-variable multi-step linear deduction + contradiction test
    eqs = [([i, i + 1], (i % 2)) for i in range(19)]
    eqs.append(([19], 1))
    res = core.solve_gf2_equations(num_vars=20, equations=eqs, return_degrees=True)
    assert not res["conflict"] and res["rank"] == 20 and len(res["units"]) == 20
    expected = {19: 1}
    for i in range(18, -1, -1):
        expected[i] = expected[i + 1] ^ (i % 2)
    assert res["units"] == expected

    eqs_bad = eqs + [([0, 19], res["units"][0] ^ res["units"][19] ^ 1)]
    res_bad = core.solve_gf2_equations(num_vars=20, equations=eqs_bad)
    assert res_bad["conflict"]

    # 2. Crystal Plane A + B coupled bidirectional propagation on R=20
    R = 20
    rng = np.random.default_rng(42)
    w0_15 = rng.integers(0, 1 << 32, size=(1, 16), dtype=np.uint32)
    cr = core.compress_exact(w0_15, rounds=R)

    W_gt = words_to_bits_np(cr.W[0])
    S_gt = words_to_bits_np(cr.S[0])
    C_gt = words_to_bits_np(cr.C[0])

    # Unknown everywhere except welded S[0], S[R] + LSB of W[0..15] + W[0..14] known (solving W[15] and W[16..19])
    W_sym = np.full((R, 32), SYM_Q, dtype=np.uint8)
    S_sym = np.full((R + 1, 8, 32), SYM_Q, dtype=np.uint8)
    C_sym = np.full((R, C_PLANES, 32), SYM_Q, dtype=np.uint8)

    W_sym[:15] = W_gt[:15]
    W_sym[15, 0] = W_gt[15, 0]  # Only LSB of W[15] given; R=16..19 must propagate + deduce W[16]'s LSB via GF2/ARX
    prop = core.propagate_symbols(cr.H_target[0], W_sym, S_sym, C_sym, rounds=R, use_gf2=True, max_iters=16)
    assert not prop["conflict"], "True partial assignment triggered false conflict!"
    assert prop["unit_prop_deductions"] > 0
    # Verify ALL deduced bits strictly equal ground truth
    for arr_sym, arr_gt in ((prop["W_sym"], W_gt), (prop["S_sym"], S_gt), (prop["C_sym"], C_gt)):
        det_mask = (arr_sym == SYM_0) | (arr_sym == SYM_1)
        assert np.all(arr_sym[det_mask] == arr_gt[det_mask]), "Propagator deduced an incorrect bit!"
    # Check W[16, 0] was deduced strictly via linear/full-adder LSB closure from W[0..15][0]
    assert prop["W_sym"][16, 0] == W_gt[16, 0]

    # Flip 1 bit against boundary S[R] (round R-1 transition contradiction) -> must flag conflict
    W_bad = W_gt.copy()
    W_bad[R - 1, 0] ^= 1
    S_bad = np.full((R + 1, 8, 32), SYM_Q, dtype=np.uint8)
    S_bad[R - 1] = S_gt[R - 1]
    C_bad = np.full((R, C_PLANES, 32), SYM_Q, dtype=np.uint8)
    prop_bad = core.propagate_symbols(cr.H_target[0], W_bad, S_bad, C_bad, rounds=R, use_gf2=True)
    assert prop_bad["conflict"], "Inconsistent round transition was not detected!"


def test_cnf_cdcl_round16_blind_preimage() -> None:
    core = get_core()
    rng = np.random.default_rng(999)
    w_true = rng.integers(0, 1 << 32, size=(1, 16), dtype=np.uint32)
    cr = core.compress_exact(w_true, rounds=16)

    solver = CaDiCaLCDCLSolver(solver_name="cadical195")
    res = solver.solve(H_target=cr.H_target[0], rounds=16, time_limit_sec=10.0)
    print(
        f"  [Test R=16 Blind CDCL] status={res['status']} valid={res['valid_preimage']} "
        f"confl={res['conflicts']} time={res['elapsed_sec']:.3f}s",
        flush=True,
    )
    assert res["status"] == "SAT" and res["valid_preimage"]


def test_cluster_throughput_over_100k() -> None:
    thr = run_throughput_benchmark(rounds=64, num_crystals=16, cands_per_crystal=10000, cluster_size=4, repeats=3)
    print(f"  [Test Throughput R=64] cpp_cluster_ops_per_sec={thr['cpp_cluster_ops_per_sec']:,.1f}", flush=True)
    assert thr["cpp_cluster_ops_per_sec"] >= 100_000.0
