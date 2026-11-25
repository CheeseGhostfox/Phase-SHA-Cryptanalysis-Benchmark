"""Resumable Colab A100 experimental queue orchestrator for Phase-SHA V8A."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from v8a import configure_strict_fp32
from v8a.torus_flow import TorusFlow
from v8a.train_v8a import run_training


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _make_args(
    run_name: str,
    run_dir: Path,
    mode: str,
    rounds: int,
    k_bits: int,
    steps: int,
    transport_steps: int,
    batch_size: int,
    num_layers: int,
    hidden_dim: int,
    ff_dim: int,
    eval_every: int = 250,
    probe_every: int = 1000,
) -> argparse.Namespace:
    return argparse.Namespace(
        mode=mode,
        rounds=rounds,
        k_bits=k_bits,
        steps=steps,
        transport_steps=transport_steps,
        batch_size=batch_size,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        ff_dim=ff_dim,
        num_blocks=2,
        no_attn=False,
        num_components=12,
        lr_phi=5e-4,
        lr_psi=0.03,
        weight_decay=1e-5,
        grad_clip=2.0,
        eval_every=eval_every,
        probe_every=probe_every,
        eval_size=4096,
        probe_train_size=32768,
        run_dir=str(run_dir),
        run_name=run_name,
        seed=42 + rounds,
    )


def _stage_b_arch(rounds: int) -> Dict[str, int]:
    if rounds <= 8:
        nl = min(36, 4 * rounds + 4)
        return {
            "num_layers": nl,
            "hidden_dim": 320,
            "ff_dim": 384,
            "transport_steps": 3000,
            "steps": 5000,
        }
    if rounds == 12:
        return {
            "num_layers": 36,
            "hidden_dim": 320,
            "ff_dim": 352,
            "transport_steps": 4000,
            "steps": 6000,
        }
    return {
        "num_layers": 54,
        "hidden_dim": 256,
        "ff_dim": 288,
        "transport_steps": 4500,
        "steps": 6500,
    }


def verify_all_param_budgets() -> Dict[str, int]:
    """Verify every Stage A and Stage B architecture strictly sits in [15M, 40M] params."""
    budgets: Dict[str, int] = {}
    flow_a = TorusFlow(num_layers=24, hidden_dim=384, ff_dim=512, num_blocks=2, use_attn=True)
    budgets["stage_a_24L_384D"] = flow_a.param_count()
    for r in (3, 4, 5, 6, 8, 12, 18):
        cfg = _stage_b_arch(r)
        flow_b = TorusFlow(
            num_layers=cfg["num_layers"],
            hidden_dim=cfg["hidden_dim"],
            ff_dim=cfg["ff_dim"],
            num_blocks=2,
            use_attn=True,
        )
        p = flow_b.param_count()
        budgets[f"stage_b_R{r}_{cfg['num_layers']}L_{cfg['hidden_dim']}D"] = p
        if not (15_000_000 <= p <= 40_000_000):
            raise AssertionError(f"Param budget out of [15M, 40M] for Stage B R={r}: {p:,}")
    return budgets


def _detect_critical_round(stage_a_results: List[Dict[str, Any]]) -> int:
    for res in sorted(stage_a_results, key=lambda x: x["rounds"]):
        if res["flow_gain_bits"] < 5.0 or res["exact_pass_64"] == 0.0:
            return max(3, min(8, int(res["rounds"])))
    return 4


def _load_completed(run_dir: Path) -> Dict[str, Any] | None:
    tel_path = run_dir / "telemetry.json"
    sum_path = run_dir / "summary.json"
    if tel_path.exists() and sum_path.exists():
        try:
            tel = json.loads(tel_path.read_text(encoding="utf-8"))
            if tel.get("status") == "completed":
                return json.loads(sum_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def main() -> None:
    configure_strict_fp32()
    parser = argparse.ArgumentParser(description="Phase-SHA V8A Colab A100 Sweep Queue")
    parser.add_argument("--out_dir", type=str, default="/content/v8a_runs")
    parser.add_argument("--k_bits", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "queue_status.json"
    summary_path = out_dir / "summary.json"

    budgets = verify_all_param_budgets()
    print("[V8A Queue] Verified parameter budgets:", flush=True)
    for k, v in budgets.items():
        print(f"  {k}: {v:,}", flush=True)

    stage_a_schedule = [
        (2, 2000),
        (3, 3000),
        (4, 5000),
        (5, 5000),
        (6, 4000),
        (8, 4000),
    ]

    all_results: List[Dict[str, Any]] = []
    stage_a_results: List[Dict[str, Any]] = []
    planned_names = [f"track1_r{r}_k{args.k_bits}" for r, _ in stage_a_schedule]

    def _update_queue_status(current_run: str, phase: str, critical_round: int | None = None) -> None:
        _atomic_write_json(
            status_path,
            {
                "status": phase,
                "current_run": current_run,
                "k_bits": args.k_bits,
                "batch_size": args.batch_size,
                "critical_round": critical_round,
                "planned_runs": planned_names,
                "completed_runs": [r["run_name"] for r in all_results],
                "results": all_results,
                "updated_at": time.time(),
            },
        )

    # --- Stage A: Track-1 Pure Unsupervised Phase-Transition Spectrum ---
    for r, steps in stage_a_schedule:
        run_name = f"track1_r{r}_k{args.k_bits}"
        run_dir = out_dir / run_name
        existing = _load_completed(run_dir)
        if existing is not None:
            print(f"[V8A Queue] Resuming completed {run_name}", flush=True)
            stage_a_results.append(existing)
            all_results.append(existing)
            _update_queue_status(run_name, "stage_a_running")
            continue

        _update_queue_status(run_name, "stage_a_running")
        run_args = _make_args(
            run_name=run_name,
            run_dir=run_dir,
            mode="unsupervised",
            rounds=r,
            k_bits=args.k_bits,
            steps=steps,
            transport_steps=0,
            batch_size=args.batch_size,
            num_layers=24,
            hidden_dim=384,
            ff_dim=512,
        )
        res = run_training(run_args)
        stage_a_results.append(res)
        all_results.append(res)
        _update_queue_status(run_name, "stage_a_running")

    r_crit = _detect_critical_round(stage_a_results)
    stage_b_rounds = sorted(set([r_crit, 8, 12, 18]))
    for r in stage_b_rounds:
        planned_names.append(f"track2_r{r}_k{args.k_bits}")
    print(f"[V8A Queue] Detected critical round R_c={r_crit}; Stage B rounds={stage_b_rounds}", flush=True)

    # --- Stage B: Track-2 Reversible Round-Transport + Unsupervised Fine-Tune ---
    for r in stage_b_rounds:
        run_name = f"track2_r{r}_k{args.k_bits}"
        run_dir = out_dir / run_name
        existing = _load_completed(run_dir)
        if existing is not None:
            print(f"[V8A Queue] Resuming completed {run_name}", flush=True)
            all_results.append(existing)
            _update_queue_status(run_name, "stage_b_running", critical_round=r_crit)
            continue

        cfg = _stage_b_arch(r)
        _update_queue_status(run_name, "stage_b_running", critical_round=r_crit)
        run_args = _make_args(
            run_name=run_name,
            run_dir=run_dir,
            mode="round_transport",
            rounds=r,
            k_bits=args.k_bits,
            steps=cfg["steps"],
            transport_steps=cfg["transport_steps"],
            batch_size=args.batch_size,
            num_layers=cfg["num_layers"],
            hidden_dim=cfg["hidden_dim"],
            ff_dim=cfg["ff_dim"],
        )
        res = run_training(run_args)
        all_results.append(res)
        _update_queue_status(run_name, "stage_b_running", critical_round=r_crit)

    _update_queue_status("", "completed", critical_round=r_crit)
    _atomic_write_json(
        summary_path,
        {
            "status": "completed",
            "critical_round": r_crit,
            "stage_b_rounds": stage_b_rounds,
            "param_budgets": budgets,
            "results": all_results,
            "finished_at": time.time(),
        },
    )

    print("\n=== Phase-SHA V8A Final Sweep Summary ===", flush=True)
    header = f"{'Run':<20} {'Mode':<15} {'R':>2} {'Base':>7} {'Eval':>7} {'Gain':>7} {'ch55':>5} {'w_acc':>7} {'p@1':>7} {'p@64':>7}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for res in all_results:
        print(
            f"{res['run_name']:<20} {res['mode']:<15} {res['rounds']:>2} "
            f"{res['baseline_input_bits']:>7.2f} {res['eval_extracted_bits']:>7.2f} "
            f"{res['flow_gain_bits']:>+7.2f} {res['collapsed_ch_55']:>5d} "
            f"{res['w_bit_acc']:>7.4f} {res['exact_pass_1']:>7.4f} {res['exact_pass_64']:>7.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
