"""Seconds-level CPU-only local smoke test for P3 AST, Worlds, Z3 V1-V7 Auditor, and MCTS."""

from __future__ import annotations

import os
import sys
import time
import random

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from ast_spec import MASK32, Add, Ch, Const, Sub, Var, Xor, ast_stats
from neural_mcts_engine import (
    NeuralMCTSDiscoveryEngine,
    ast_backSolveA,
    ast_clamp4,
    ast_five_round_coupling,
    ast_state_vars,
    apply_mcts_action,
    evaluate_candidate_fast,
    ACTION_NAMES,
)
from sha_worlds import SHA256_K, RoundsConfig, build_world_spec
from z3_auditor import CandidateProposition, Z3Auditor


def main() -> None:
    t0 = time.perf_counter()
    auditor = Z3Auditor(timeout_ms=10000, require_min_active_count=5)
    rng = np.random.default_rng(123)

    print("=== 1. Negative Control Tests (Known <=4R & Tautologies must be REJECTED) ===")
    # Neg-1: 4-round clamp embedded inside 5-round window (must fail V6 active_span < 5)
    cfg5 = RoundsConfig(track="A1", round_start=0, num_rounds=5)
    spec5 = build_world_spec(cfg5)
    s5 = ast_state_vars("s5")
    _, _, _, a_m2 = ast_clamp4(s5)
    neg1 = CandidateProposition(
        track="A1",
        title="NEG1: 4-round clamp disguised in 5-round window",
        form="FormI_Reconstruction",
        config=cfg5,
        eliminated_vars=["w0", "w1", "w2", "w3", "w4", "s0.e", "s0.f", "s0.g", "s0.h"],
        condition_ast={"op": "True"},
        mask_ast=Const(MASK32),
        lhs_ast=a_m2,
        rhs_ast=Var("s0.c"),
    )
    res_neg1 = auditor.audit(neg1, spec5)
    print(f"[NEG1 4R clamp] passed={res_neg1.passed}, reason={res_neg1.reject_reason}")
    assert not res_neg1.passed and "v6_active" in res_neg1.reject_reason

    # Neg-2: 1-round r18_a10 tunneling in 5-round window (must fail V6 active_span=1 < 5)
    s1 = ast_state_vars("s1")
    neg2 = CandidateProposition(
        track="A1",
        title="NEG2: 1-round backSolveA",
        form="FormI_Reconstruction",
        config=cfg5,
        eliminated_vars=["w0", "s0.e", "s0.f", "s0.g", "s0.h"],
        condition_ast={"op": "True"},
        mask_ast=Const(MASK32),
        lhs_ast=ast_backSolveA(s1["e"], s1["a"], Var("s0.a"), Var("s0.b"), Var("s0.c")),
        rhs_ast=Var("s0.d"),
    )
    res_neg2 = auditor.audit(neg2, spec5)
    print(f"[NEG2 1R r18_a10] passed={res_neg2.passed}, reason={res_neg2.reject_reason}")
    assert not res_neg2.passed and "v6_active" in res_neg2.reject_reason

    # Neg-3: 1-round Ch masking shield (must fail V6)
    neg3 = CandidateProposition(
        track="A1",
        title="NEG3: 1-round Ch masking",
        form="FormI_Reconstruction",
        config=cfg5,
        eliminated_vars=["s0.g"],
        condition_ast={"op": "True"},
        mask_ast=Var("s0.e"),
        lhs_ast=Ch(Var("s0.e"), Var("s0.f"), Var("s0.g")),
        rhs_ast=Var("s0.f"),
    )
    res_neg3 = auditor.audit(neg3, spec5)
    print(f"[NEG3 1R Ch mask] passed={res_neg3.passed}, reason={res_neg3.reject_reason}")
    assert not res_neg3.passed

    # Neg-4: Pure algebraic tautology (x + y) - y == x (must fail V5)
    neg4 = CandidateProposition(
        track="A1",
        title="NEG4: algebraic tautology",
        form="FormI_Reconstruction",
        config=cfg5,
        eliminated_vars=["w0"],
        condition_ast={"op": "True"},
        mask_ast=Const(MASK32),
        lhs_ast=Sub(Add(Var("s5.a"), Var("s5.b")), Var("s5.b")),
        rhs_ast=Var("s5.a"),
    )
    res_neg4 = auditor.audit(neg4, spec5)
    print(f"[NEG4 tautology] passed={res_neg4.passed}, reason={res_neg4.reject_reason}")
    assert not res_neg4.passed and "v5_pure_tautology" in res_neg4.reject_reason

    print("\n=== 2. Positive >=5-Round Multi-Track Smoke Samples ===")
    # Test at least 1 hit from each Track (A1, A2, B1, B2)
    py_rng = random.Random(7)
    from neural_mcts_engine import build_seed_states
    seeds = build_seed_states(py_rng)
    positive_hits = 0
    hits_by_track = {}
    for s in seeds:
        if hits_by_track.get(s.track, 0) >= 2:
            continue
        for act_name in (
            "bridge_5r_coupling",
            "peel_inv_round_known_w",
            "sched_elim_w0",
            "sched_elim_w14",
            "btc_splice_recover_a",
            "btc_b2_sched_collapse",
            "form2_noninterference",
        ):
            a_idx = ACTION_NAMES.index(act_name)
            nxt = apply_mcts_action(s, a_idx, py_rng)
            if nxt is None:
                continue
            prop = CandidateProposition(
                track=nxt.track,
                title=nxt.title,
                form=nxt.form,
                config=nxt.config,
                eliminated_vars=nxt.eliminated_vars,
                condition_ast=nxt.cond_ast,
                mask_ast=nxt.mask_ast,
                lhs_ast=nxt.lhs_ast,
                rhs_ast=nxt.rhs_ast,
            )
            ok_fast, cprop, rew, _ = evaluate_candidate_fast(prop, nxt.spec, rng, n_samples=128)
            if ok_fast and cprop is not None:
                ares = auditor.audit(cprop, nxt.spec)
                if ares.passed:
                    positive_hits += 1
                    hits_by_track[nxt.track] = hits_by_track.get(nxt.track, 0) + 1
                    print(
                        f"[PASS #{positive_hits}] track={nxt.track} span={ares.v6_active_span} "
                        f"rounds={ares.v6_active_rounds} splice={ares.v6_splice_necessary} "
                        f"nodes={ares.ast_nodes} z3_ms={ares.z3_solve_ms:.1f} | {nxt.title}"
                    )
                    if hits_by_track.get(nxt.track, 0) >= 2:
                        break

    assert positive_hits >= 5, f"Expected >=5 positive smoke hits, got {positive_hits}"
    print(f"Tracks covered in smoke: {hits_by_track}")

    print("\n=== 3. Mini Neural-MCTS Search Smoke (6 seconds) ===")
    engine = NeuralMCTSDiscoveryEngine(seed=42, z3_timeout_ms=8000)
    recs = engine.run_search(budget_seconds=6.0, rollouts_per_root=8, max_depth=3, log_every_sec=3.0)
    print(f"Smoke search done in {time.perf_counter() - t0:.2f}s | stats={engine.stats} | accepted={len(recs)}")


if __name__ == "__main__":
    main()
