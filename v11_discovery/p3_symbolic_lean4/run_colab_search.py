"""High-throughput multi-worker CPU symbolic + Neural-MCTS + Z3 discovery runner for Colab v11-p3."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from ast_spec import MASK32
from neural_mcts_engine import NeuralMCTSDiscoveryEngine
from sha_worlds import build_world_spec
from z3_auditor import AuditResult, CandidateProposition, Z3Auditor


def _worker_task(args: tuple[int, float, int, int]) -> Dict[str, Any]:
    worker_id, budget_sec, z3_timeout_ms, min_active = args
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Spread CPU threads cleanly across workers
    import torch
    torch.set_num_threads(1)

    seed = 1000 + worker_id * 137
    engine = NeuralMCTSDiscoveryEngine(
        seed=seed,
        z3_timeout_ms=z3_timeout_ms,
        require_min_active_count=min_active,
    )
    recs = engine.run_search(
        budget_seconds=budget_sec,
        rollouts_per_root=28,
        max_depth=4,
        c_puct=1.45,
        log_every_sec=45.0,
    )
    serialized: List[Dict[str, Any]] = []
    for r in recs:
        assert r.proposition is not None
        p = r.proposition
        serialized.append(
            {
                "track": p.track,
                "title": p.title,
                "form": p.form,
                "config": {
                    "track": p.config.track,
                    "round_start": p.config.round_start,
                    "num_rounds": p.config.num_rounds,
                    "use_msg_schedule": p.config.use_msg_schedule,
                    "btc_splice": p.config.btc_splice,
                    "block1_rounds": p.config.block1_rounds,
                    "block2_rounds": p.config.block2_rounds,
                    "padded_mode": p.config.padded_mode,
                    "padded_free_words": p.config.padded_free_words,
                },
                "eliminated_vars": p.eliminated_vars,
                "condition_ast": p.condition_ast,
                "mask_ast": p.mask_ast,
                "lhs_ast": p.lhs_ast,
                "rhs_ast": p.rhs_ast,
                "seed_note": p.seed_note,
            }
        )
    return {"worker_id": worker_id, "stats": engine.stats, "candidates": serialized}


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 Multi-Worker Neural-MCTS + Z3 Discovery")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--budget_sec", type=float, default=1800.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--z3_timeout_ms", type=int, default=15000)
    parser.add_argument("--min_active", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "discovered_identities.json"

    n_cpus = os.cpu_count() or 2
    n_workers = args.workers if args.workers > 0 else max(1, min(8, n_cpus))
    print(
        f"[P3 Search] Starting {n_workers} workers (cpus={n_cpus}), "
        f"budget={args.budget_sec:.1f}s, z3_timeout={args.z3_timeout_ms}ms",
        flush=True,
    )

    t0 = time.time()
    # Run in rounds/chunks of <= 300s so intermediate discovered_identities.json is checkpointed!
    chunk_sec = min(300.0, args.budget_sec)
    global_auditor = Z3Auditor(timeout_ms=args.z3_timeout_ms, require_min_active_count=args.min_active)
    agg_stats = {
        "total_candidates_generated": 0,
        "passed_random_filter": 0,
        "z3_unsat_verified": 0,
        "passed_all_7_nontriviality": 0,
        "rejected_reasons": {},
    }

    chunk_idx = 0
    while (time.time() - t0) < args.budget_sec:
        rem = args.budget_sec - (time.time() - t0)
        cur_budget = min(chunk_sec, rem)
        if cur_budget < 3.0:
            break
        chunk_idx += 1
        worker_args = [
            (chunk_idx * 100 + wid, cur_budget, args.z3_timeout_ms, args.min_active)
            for wid in range(n_workers)
        ]
        if n_workers == 1:
            results = [_worker_task(worker_args[0])]
        else:
            with mp.Pool(processes=n_workers) as pool:
                results = pool.map(_worker_task, worker_args)

        for wres in results:
            st = wres["stats"]
            agg_stats["total_candidates_generated"] += st["total_candidates_generated"]
            agg_stats["passed_random_filter"] += st["passed_random_filter"]
            for rk, rv in st["rejected_reasons"].items():
                agg_stats["rejected_reasons"][rk] = agg_stats["rejected_reasons"].get(rk, 0) + rv
            for cand in wres["candidates"]:
                from sha_worlds import RoundsConfig
                cfg = RoundsConfig(**cand["config"])
                spec = build_world_spec(cfg)
                prop = CandidateProposition(
                    track=cand["track"],
                    title=cand["title"],
                    form=cand["form"],
                    config=cfg,
                    eliminated_vars=cand["eliminated_vars"],
                    condition_ast=cand["condition_ast"],
                    mask_ast=cand["mask_ast"],
                    lhs_ast=cand["lhs_ast"],
                    rhs_ast=cand["rhs_ast"],
                    seed_note=cand.get("seed_note", ""),
                )
                ares = global_auditor.audit(prop, spec)
                if ares.passed:
                    agg_stats["z3_unsat_verified"] += 1
                    agg_stats["passed_all_7_nontriviality"] += 1

        # Sort accepted by (track, v6_active_span desc, ast_nodes asc)
        sorted_recs = sorted(
            global_auditor.accepted_records,
            key=lambda r: (r.proposition.track if r.proposition else "", -r.v6_active_span, r.ast_nodes),
        )
        identities_out = [
            rec.to_identity_dict(f"p3_id_{i + 1:03d}")
            for i, rec in enumerate(sorted_recs)
        ]
        wall_sec = time.time() - t0
        payload = {
            "run_summary": {
                "colab_session": "v11-p3",
                "colab_stopped": False,
                "total_candidates_generated": agg_stats["total_candidates_generated"],
                "passed_random_filter": agg_stats["passed_random_filter"],
                "z3_unsat_verified": agg_stats["z3_unsat_verified"],
                "passed_all_7_nontriviality": len(identities_out),
                "wall_time_seconds": round(wall_sec, 2),
                "workers": n_workers,
                "rejected_reasons": agg_stats["rejected_reasons"],
            },
            "identities": identities_out,
        }
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"[P3 Chunk {chunk_idx}] wall={wall_sec:.1f}s gen={agg_stats['total_candidates_generated']} "
            f"fast={agg_stats['passed_random_filter']} unique_verified={len(identities_out)}",
            flush=True,
        )

    print(f"[P3 Search DONE] Saved {len(global_auditor.accepted_records)} identities to {out_json}", flush=True)


if __name__ == "__main__":
    main()
