"""Industrial SAT (CaDiCaL-195) strict Blind Inversion baseline for R in {16, 20, 24}."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from v10_ouroboros64.ouroboros_core.cnf_cdcl import CaDiCaLCDCLSolver
from v10_ouroboros64.ouroboros_core.engine import get_core
from .throughput_bench import run_throughput_benchmark


def _run_suite_track(
    solver: CaDiCaLCDCLSolver,
    rounds: int,
    n_queries: int,
    fixed_prefix_words: int,
    time_limit_sec: float,
    seed: int,
) -> Dict[str, Any]:
    core = get_core()
    rng = np.random.default_rng(seed + rounds * 100 + fixed_prefix_words)
    w_all = rng.integers(0, 1 << 32, size=(n_queries, 16), dtype=np.uint32)
    cr = core.compress_exact(w_all, rounds=rounds)

    records: List[Dict[str, Any]] = []
    for qi in range(n_queries):
        fixed_map: Optional[Dict[int, int]] = None
        if fixed_prefix_words > 0:
            fixed_map = {idx: int(w_all[qi, idx]) for idx in range(fixed_prefix_words)}
        res = solver.solve(
            H_target=cr.H_target[qi],
            rounds=rounds,
            fixed_words=fixed_map,
            time_limit_sec=time_limit_sec,
        )
        records.append(res)
        print(
            f"  [Blind SAT R={rounds:2d} free={512 - 32*fixed_prefix_words:3d}b q={qi+1}/{n_queries}] "
            f"status={res['status']:17s} valid={res['valid_preimage']} "
            f"confl={res['conflicts']:7d} time={res['elapsed_sec']:.3f}s",
            flush=True,
        )

    solved = [r for r in records if r["valid_preimage"]]
    return {
        "rounds": rounds,
        "free_bits": 512 - 32 * fixed_prefix_words,
        "n_queries": n_queries,
        "solved_count": len(solved),
        "solve_rate": round(len(solved) / float(max(1, n_queries)), 4),
        "mean_conflicts": round(sum(r["conflicts"] for r in records) / float(max(1, n_queries)), 1),
        "mean_elapsed_sec": round(sum(r["elapsed_sec"] for r in records) / float(max(1, n_queries)), 4),
        "num_vars": records[0]["num_vars"] if records else 0,
        "num_clauses": records[0]["num_clauses"] if records else 0,
        "records": [
            {
                "status": r["status"],
                "valid_preimage": r["valid_preimage"],
                "conflicts": r["conflicts"],
                "decisions": r["decisions"],
                "propagations": r["propagations"],
                "elapsed_sec": round(r["elapsed_sec"], 4),
            }
            for r in records
        ],
    }


def run_milestone1_benchmarks(out_path: Path | None = None) -> Dict[str, Any]:
    """Run throughput + strict Blind Inversion CaDiCaL195 benchmarks for R=16, 20, 24."""
    print("[Milestone 1] Running throughput benchmark...", flush=True)
    thr = run_throughput_benchmark()
    print(f"  -> C++ cluster ΔE throughput: {thr['cpp_cluster_ops_per_sec']:,.1f} ops/sec", flush=True)

    solver = CaDiCaLCDCLSolver(solver_name="cadical195")
    t0 = time.perf_counter()

    # Suite 1: 100% unconstrained 512-bit blind preimage (R=16 fast, R=20/24 bounded 5s wall)
    print("\n[Milestone 1] Running Suite A: Unconstrained 512-bit Blind Inversion...", flush=True)
    unconstrained = [
        _run_suite_track(solver, rounds=16, n_queries=3, fixed_prefix_words=0, time_limit_sec=5.0, seed=101),
        _run_suite_track(solver, rounds=20, n_queries=2, fixed_prefix_words=0, time_limit_sec=5.0, seed=102),
        _run_suite_track(solver, rounds=24, n_queries=1, fixed_prefix_words=0, time_limit_sec=5.0, seed=103),
    ]

    # Suite 2: Reduced-DoF Blind Inversion (zero internal state leaked; 96/64/32-bit free W suffix)
    print("\n[Milestone 1] Running Suite B: Reduced-DoF Blind Inversion (zero internal leaks)...", flush=True)
    reduced_dof = [
        _run_suite_track(solver, rounds=20, n_queries=3, fixed_prefix_words=13, time_limit_sec=5.0, seed=201),
        _run_suite_track(solver, rounds=24, n_queries=2, fixed_prefix_words=14, time_limit_sec=5.0, seed=202),
        _run_suite_track(solver, rounds=24, n_queries=2, fixed_prefix_words=15, time_limit_sec=5.0, seed=203),
    ]

    payload = {
        "milestone": "Milestone 1",
        "elapsed_total_sec": round(time.perf_counter() - t0, 2),
        "throughput": thr,
        "blind_inversion_unconstrained_512b": unconstrained,
        "blind_inversion_reduced_dof": reduced_dof,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[Milestone 1] Saved benchmark report -> {out_path}", flush=True)
    return payload


if __name__ == "__main__":
    default_out = Path(__file__).resolve().parent.parent / "runs" / "milestone1_benchmark.json"
    run_milestone1_benchmarks(default_out)
