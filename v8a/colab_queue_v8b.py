"""Resumable Colab A100 experimental queue orchestrator for Phase-SHA V8B (T1 + T2)."""

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
from v8a.torus_flow_v8b import V8BTorusFlow
from v8a.train_v8a import run_training


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _v8b_t1_cfg(rounds: int, steps: int) -> Dict[str, Any]:
    return {
        "arch": "v8b_t1",
        "mode": "unsupervised",
        "rounds": rounds,
        "steps": steps,
        "transport_steps": 0,
        "num_layers": 24,
        "c_bit": 128,
        "c_mid": 96,
        "d_glob": 360,
        "ff_glob": 480,
    }


def _v8b_t2_cfg(rounds: int) -> Dict[str, Any]:
    if rounds <= 2:
        return {
            "arch": "v8b_t2",
            "mode": "round_transport",
            "rounds": rounds,
            "steps": 2000,
            "transport_steps": 800,
            "num_layers": 15,
            "c_bit": 160,
            "c_mid": 128,
            "d_glob": 448,
            "ff_glob": 600,
        }
    if rounds == 5:
        return {
            "arch": "v8b_t2",
            "mode": "round_transport",
            "rounds": 5,
            "steps": 3500,
            "transport_steps": 2000,
            "num_layers": 30,
            "c_bit": 128,
            "c_mid": 96,
            "d_glob": 320,
            "ff_glob": 416,
        }
    if rounds == 6:
        return {
            "arch": "v8b_t2",
            "mode": "round_transport",
            "rounds": 6,
            "steps": 4000,
            "transport_steps": 2500,
            "num_layers": 35,
            "c_bit": 128,
            "c_mid": 96,
            "d_glob": 288,
            "ff_glob": 384,
        }
    if rounds == 8:
        return {
            "arch": "v8b_t2",
            "mode": "round_transport",
            "rounds": 8,
            "steps": 4500,
            "transport_steps": 3000,
            "num_layers": 43,
            "c_bit": 112,
            "c_mid": 80,
            "d_glob": 256,
            "ff_glob": 352,
        }
    if rounds == 12:
        return {
            "arch": "v8b_t2",
            "mode": "round_transport",
            "rounds": 12,
            "steps": 5500,
            "transport_steps": 4000,
            "num_layers": 52,
            "c_bit": 96,
            "c_mid": 64,
            "d_glob": 224,
            "ff_glob": 288,
        }
    if rounds == 16:
        return {
            "arch": "v8b_t2",
            "mode": "round_transport",
            "rounds": 16,
            "steps": 6500,
            "transport_steps": 5000,
            "num_layers": 68,
            "c_bit": 88,
            "c_mid": 64,
            "d_glob": 192,
            "ff_glob": 256,
        }
    return {
        "arch": "v8b_t2",
        "mode": "round_transport",
        "rounds": rounds,
        "steps": 7000,
        "transport_steps": 5500,
        "num_layers": 76,
        "c_bit": 80,
        "c_mid": 56,
        "d_glob": 176,
        "ff_glob": 240,
    }


def verify_v8b_param_budgets() -> Dict[str, int]:
    """Verify every V8B Track-1 & Track-2 architecture lies strictly within [20M, 40M]."""
    configs = [
        ("v8b_t1_r2", _v8b_t1_cfg(2, 2000)),
        ("v8b_t1_r5", _v8b_t1_cfg(5, 3500)),
        ("v8b_t2_r2", _v8b_t2_cfg(2)),
        ("v8b_t2_r5", _v8b_t2_cfg(5)),
        ("v8b_t2_r6", _v8b_t2_cfg(6)),
        ("v8b_t2_r8", _v8b_t2_cfg(8)),
        ("v8b_t2_r12", _v8b_t2_cfg(12)),
        ("v8b_t2_r16", _v8b_t2_cfg(16)),
        ("v8b_t2_r18", _v8b_t2_cfg(18)),
    ]
    budgets: Dict[str, int] = {}
    for name, cfg in configs:
        flow = V8BTorusFlow(
            flow_layout=cfg["arch"],
            rounds=cfg["rounds"],
            num_layers=cfg["num_layers"],
            c_bit=cfg["c_bit"],
            c_mid=cfg["c_mid"],
            d_glob=cfg["d_glob"],
            ff_glob=cfg["ff_glob"],
            num_arx_blocks=2,
            use_checkpoint=False,
        )
        p = flow.param_count()
        budgets[name] = p
        if not (20_000_000 <= p <= 40_000_000):
            raise AssertionError(f"{name} param_count={p:,} outside [20M, 40M]!")
    return budgets


def _make_args(
    run_name: str,
    run_dir: Path,
    cfg: Dict[str, Any],
    k_bits: int,
    batch_size: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        arch=cfg["arch"],
        mode=cfg["mode"],
        rounds=cfg["rounds"],
        k_bits=k_bits,
        steps=cfg["steps"],
        transport_steps=cfg["transport_steps"],
        batch_size=batch_size,
        num_layers=cfg["num_layers"],
        hidden_dim=cfg["d_glob"],
        ff_dim=cfg["ff_glob"],
        num_blocks=2,
        c_bit=cfg["c_bit"],
        c_mid=cfg["c_mid"],
        d_glob=cfg["d_glob"],
        ff_glob=cfg["ff_glob"],
        num_arx_blocks=2,
        lambda_lip=2e-3,
        lip_sigma=0.15,
        use_checkpoint=True,
        init_ckpt="",
        no_attn=False,
        num_components=12,
        lr_phi=5e-4,
        lr_psi=0.03,
        weight_decay=1e-5,
        grad_clip=2.0,
        eval_every=250,
        probe_every=1000,
        eval_size=4096,
        probe_train_size=32768,
        run_dir=str(run_dir),
        run_name=run_name,
        seed=42 + cfg["rounds"],
    )


def _load_completed(run_dir: Path) -> Dict[str, Any] | None:
    tel_p = run_dir / "telemetry.json"
    sum_p = run_dir / "summary.json"
    if tel_p.exists() and sum_p.exists():
        try:
            tel = json.loads(tel_p.read_text(encoding="utf-8"))
            if tel.get("status") == "completed":
                return json.loads(sum_p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def main() -> None:
    configure_strict_fp32()
    parser = argparse.ArgumentParser(description="Phase-SHA V8B Dual-Track Sweep Queue")
    parser.add_argument("--out_dir", type=str, default="/content/v8a_runs")
    parser.add_argument("--k_bits", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "queue_status.json"
    summary_path = out_dir / "summary_v8b.json"

    budgets = verify_v8b_param_budgets()
    print("[V8B Queue] Verified parameter budgets [20M-40M]:", flush=True)
    for k, v in budgets.items():
        print(f"  {k}: {v:,}", flush=True)

    # Preserve V8A-T1 baseline rows in results
    v8a_archive: List[Dict[str, Any]] = []
    arc_p = out_dir / "v8a_baseline_archive.json"
    if arc_p.exists():
        try:
            v8a_archive = json.loads(arc_p.read_text(encoding="utf-8")).get("v8a_t1_baseline", [])
        except Exception:
            v8a_archive = []

    schedule_list = [
        (f"v8b_t1_r2_k{args.k_bits}", _v8b_t1_cfg(2, 2000)),
        (f"v8b_t1_r5_k{args.k_bits}", _v8b_t1_cfg(5, 3500)),
        (f"v8b_t2_r2_k{args.k_bits}", _v8b_t2_cfg(2)),
        (f"v8b_t2_r5_k{args.k_bits}", _v8b_t2_cfg(5)),
        (f"v8b_t2_r6_k{args.k_bits}", _v8b_t2_cfg(6)),
        (f"v8b_t2_r8_k{args.k_bits}", _v8b_t2_cfg(8)),
        (f"v8b_t2_r12_k{args.k_bits}", _v8b_t2_cfg(12)),
    ]

    v8b_results: List[Dict[str, Any]] = []
    planned_names = [r["run_name"] for r in v8a_archive] + [name for name, _ in schedule_list]

    def _update_status(current_run: str, status_str: str) -> None:
        combined = list(v8a_archive) + list(v8b_results)
        _atomic_write_json(
            status_path,
            {
                "status": status_str,
                "current_run": current_run,
                "k_bits": args.k_bits,
                "batch_size": args.batch_size,
                "planned_runs": planned_names,
                "completed_runs": [r["run_name"] for r in combined],
                "results": combined,
                "updated_at": time.time(),
            },
        )

    def _run_item(run_name: str, cfg: Dict[str, Any], phase_tag: str) -> Dict[str, Any]:
        run_dir = out_dir / run_name
        existing = _load_completed(run_dir)
        if existing is not None:
            print(f"[V8B Queue] Resuming completed {run_name}", flush=True)
            v8b_results.append(existing)
            _update_status(run_name, phase_tag)
            return existing
        _update_status(run_name, phase_tag)
        run_args = _make_args(run_name, run_dir, cfg, args.k_bits, args.batch_size)
        res = run_training(run_args)
        v8b_results.append(res)
        _update_status(run_name, phase_tag)
        return res

    r12_res = None
    for run_name, cfg in schedule_list:
        tag = "v8b_t1_running" if cfg["arch"] == "v8b_t1" else "v8b_t2_running"
        res = _run_item(run_name, cfg, tag)
        if cfg["rounds"] == 12:
            r12_res = res

    # Dynamic bonus extension to R=16 and R=18 if R=12 achieves breakthrough
    if r12_res is not None and (
        r12_res.get("flow_gain_bits", 0.0) > 5.0
        or r12_res.get("w_bit_acc", 0.5) > 0.55
        or r12_res.get("exact_pass_64", 0.0) > 0.0
    ):
        print("[V8B Queue] R=12 breakthrough confirmed! Extending to R=16 and R=18!", flush=True)
        for bonus_r in (16, 18):
            b_name = f"v8b_t2_r{bonus_r}_k{args.k_bits}"
            if b_name not in planned_names:
                planned_names.append(b_name)
            _run_item(b_name, _v8b_t2_cfg(bonus_r), "v8b_t2_bonus_running")

    _update_status("", "v8b_completed")
    combined_all = list(v8a_archive) + list(v8b_results)
    _atomic_write_json(
        summary_path,
        {
            "status": "v8b_completed",
            "param_budgets": budgets,
            "results": combined_all,
            "finished_at": time.time(),
        },
    )

    print("\n=== Phase-SHA V8A vs V8B Final Summary ===", flush=True)
    header = (
        f"{'Run':<20} {'Arch':<8} {'Mode':<15} {'R':>2} {'Base':>7} "
        f"{'Eval':>7} {'Gain':>7} {'ch55':>5} {'w_acc':>7} {'p@1':>7} {'p@64':>7}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for res in combined_all:
        print(
            f"{res['run_name']:<20} {res.get('arch', 'v8a'):<8} {res['mode']:<15} {res['rounds']:>2} "
            f"{res['baseline_input_bits']:>7.2f} {res['eval_extracted_bits']:>7.2f} "
            f"{res['flow_gain_bits']:>+7.2f} {res['collapsed_ch_55']:>5d} "
            f"{res['w_bit_acc']:>7.4f} {res['exact_pass_1']:>7.4f} {res['exact_pass_64']:>7.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
