"""Local CPU unit tests for P15 SHA-256 / BTC Double-SHA256 engine and 4-color skeleton."""

from __future__ import annotations

import hashlib
import os
import random
import sys

# Force CPU-only execution locally
os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import torch
from src.sha256_ref import sha256_trace as ref_sha256_trace

from mitm_biclique_skeleton import (
    MitMConfig,
    solve_four_color_mitm_pysat,
    z3_verify_backward_chunk_noninterference,
)
from sha256_engine import (
    MASK32,
    SHA256_IV,
    btc_double_sha256_trace,
    build_dependency_graph,
    clz_digest_be,
    clz_digest_btc_le,
    compress_trace,
    double_sha256_single_block_trace,
    pad_bytes_to_blocks,
    sha256_bytes,
    step_backward,
    step_forward,
    words_to_bytes,
)


def test_cross_check_sha256_ref() -> None:
    rng = random.Random(20261125)
    for rounds in (1, 8, 16, 24, 32, 64):
        batch = []
        for _ in range(8):
            batch.append([rng.getrandbits(32) for _ in range(16)])
        w16_t = torch.tensor(batch, dtype=torch.int64, device="cpu")
        ref_tr = ref_sha256_trace(w16_t, rounds=rounds)

        for i in range(8):
            tr = compress_trace(batch[i], iv=SHA256_IV, rounds=rounds)
            xor_w = sum(a ^ int(b) for a, b in zip(tr.W_full, ref_tr.W_full[i].tolist()))
            xor_a = sum(a ^ int(b) for a, b in zip(tr.A_full, ref_tr.A_full[i].tolist()))
            xor_e = sum(a ^ int(b) for a, b in zip(tr.E_full, ref_tr.E_full[i].tolist()))
            xor_d = sum(a ^ int(b) for a, b in zip(tr.digest, ref_tr.digest[i].tolist()))
            assert (xor_w | xor_a | xor_e | xor_d) == 0, f"Mismatch at rounds={rounds}"
    print("[PASS] Cross-check vs src/sha256_ref.py: xor_diff == 0 across R in {1,8,16,24,32,64}")


def test_cross_check_hashlib_and_btc_double() -> None:
    rng = random.Random(42)
    # 1. Single and multi-block SHA-256 vs hashlib
    for msg_len in (0, 3, 55, 56, 64, 80, 119):
        msg = bytes(rng.getrandbits(8) for _ in range(msg_len))
        got = words_to_bytes(sha256_bytes(msg))
        exp = hashlib.sha256(msg).digest()
        assert got == exp, f"sha256 mismatch at len={msg_len}"

    # 2. BTC 80-byte Double-SHA256 trace vs hashlib
    for _ in range(16):
        hdr = bytes(rng.getrandbits(8) for _ in range(80))
        tr = btc_double_sha256_trace(hdr)
        got = words_to_bytes(tr.final_digest)
        exp = hashlib.sha256(hashlib.sha256(hdr).digest()).digest()
        assert got == exp, "BTC Double-SHA256 mismatch"
        _ = clz_digest_be(tr.final_digest)
        _ = clz_digest_btc_le(tr.final_digest)

    # 3. Single-block 128-round Double-SHA256
    msg50 = b"phase-sha-v12-p15-double-sha256-test"
    blk = pad_bytes_to_blocks(msg50)[0]
    tr1, tr2 = double_sha256_single_block_trace(blk)
    assert words_to_bytes(tr2.digest) == hashlib.sha256(hashlib.sha256(msg50).digest()).digest()
    print("[PASS] Cross-check vs hashlib.sha256 & 80B BTC Double-SHA256: xor_diff == 0")


def test_step_backward_exact() -> None:
    rng = random.Random(999)
    for t in range(64):
        st = tuple(rng.getrandbits(32) for _ in range(8))
        wt = rng.getrandbits(32)
        nxt = step_forward(st, wt, t)
        rec = step_backward(nxt, wt, t)
        assert rec == st, f"Backward invertibility failed at round {t}"
    print("[PASS] step_backward exact round invertibility (t=0..63)")


def test_dependency_graph_and_constant_folding() -> None:
    g64 = build_dependency_graph("sha256_64")
    assert len(g64.constant_words) == 0

    g_btc2 = build_dependency_graph("btc_hdr2_64")
    # W[4..15] are constant; W[16] depends on W[14], W[9], W[1], W[0] (W[1],W[0] non-const)
    # Wait: W[19] = sigma1(W[17]) + W[12] + sigma0(W[4]) + W[3] (non-const)
    # Check W[20]? No, W[4..15] gives 12 constants. Let's check exact count & BTC 128-round graph.
    assert g_btc2.msg_nodes["W[4]"].const_val == 0x80000000
    assert g_btc2.msg_nodes["W[15]"].const_val == 640

    g128 = build_dependency_graph("btc_double_128")
    assert g128.num_rounds == 128
    assert g128.msg_nodes["W0[4]"].const_val == 0x80000000
    assert g128.msg_nodes["W1[8]"].const_val == 0x80000000
    assert g128.msg_nodes["W1[15]"].const_val == 256
    assert len(g128.splice_links) == 8
    print(
        f"[PASS] Dependency graphs built: btc_hdr2_64 consts={len(g_btc2.constant_words)}, "
        f"btc_double_128 nodes={len(g128.msg_nodes)}, consts={len(g128.constant_words)}"
    )


def test_pysat_and_z3_skeleton() -> None:
    # 1. PySAT 4-color skeleton smoke test on 24 rounds
    cfg = MitMConfig(rounds=24, mode="sha256_64", use_msg_schedule=True, biclique_rounds=4)
    sol = solve_four_color_mitm_pysat(cfg)
    assert sol.feasible, "PySAT 24-round skeleton should find a feasible 4-color assignment"

    # 2. Z3 BitVec(32) non-interference verification:
    # Known 4-round backSolveA: over rounds [20..23], A_20 (= D_21, etc.) is independent of W[21..23]
    # Specifically, reconstructing 'd' (A_20) at round 20 from S_24 is independent of W[20..23]?
    # Wait: step_backward recovers A_23, A_22, A_21, A_20 from S_24:
    # Round 23 backward recovers d_23 = A_20 using only S_24 (and no W[23] for A-track, wait: d_23 = ne - t1 where t1 = na - t2, so d_23 is independent of W[23]!).
    # And over 1 round [23..24], d_23 is independent of W[23]; over 4 rounds [20..24], a_20 (= A_20) is independent of W[20..23]?
    # Let's check: at round 23 backward, (a_23, b_23, c_23) = (na, nb, nc shifted) -> independent of W[23], and d_23 = ne - (na - t2) -> independent of W[23]!
    # So after 4 backward steps (rounds 23, 22, 21, 20), state_20.a, b, c, d are ALL independent of W[23], W[22], W[21], W[20]?
    # Wait: round 22 backward needs h_23! And h_23 depends on W[23]! Wait: h_23 becomes g_22 -> f_21 -> e_20! It only touches E-track, NOT A-track until 4 steps later!
    # So (a_20, b_20, c_20, d_20) IS strictly independent of W[20..23] (only 'a' at start=20 needs W[20]? No, d_20 = A_{17}!).
    # Wait: state_20 = (A_20, A_19, A_18, A_17, E_20, E_19, E_18, E_17).
    # Let's check Z3 on outputs ("a", "b", "c", "d") at round 20 from S_24 with W[20..23] perturbed!
    ok_4r, msg_4r = z3_verify_backward_chunk_noninterference(
        round_end=24,
        num_rounds=4,
        eliminated_w_indices=[20, 21, 22, 23],
        observed_outputs=("a", "b", "c", "d"),
    )
    assert ok_4r, f"Expected 4-round A-track independence of W[20..23], got: {msg_4r}"

    # Negative control: E-track ('e') at round 20 MUST depend on W[20..23]!
    ok_neg, msg_neg = z3_verify_backward_chunk_noninterference(
        round_end=24,
        num_rounds=4,
        eliminated_w_indices=[20, 21, 22, 23],
        observed_outputs=("e",),
    )
    assert not ok_neg, f"Negative control failed: E-track should depend on W, got: {msg_neg}"
    print(f"[PASS] PySAT 4-color skeleton feasible={sol.feasible}; Z3 4-round non-interference={ok_4r}, neg_ctrl={not ok_neg}")


if __name__ == "__main__":
    test_cross_check_sha256_ref()
    test_cross_check_hashlib_and_btc_double()
    test_step_backward_exact()
    test_dependency_graph_and_constant_folding()
    test_pysat_and_z3_skeleton()
    print("ALL STEP 1 ENGINE & SKELETON TESTS PASSED.")
