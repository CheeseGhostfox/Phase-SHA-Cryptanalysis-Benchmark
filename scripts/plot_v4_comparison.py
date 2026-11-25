"""Compare Phase-SHA V3 vs V4A vs V4B vs V4C (R=18 Full 32-bit) from unified metrics.csv."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_eval_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("eval_bce_loss", "") not in ("", None)]


def _f(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    v = row.get(key, "")
    if v in ("", None):
        return default
    return float(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot V3 vs V4A/V4B/V4C comparison.")
    parser.add_argument("--v3-csv", type=str, default="runs/r18_32b_v3/metrics.csv")
    parser.add_argument("--v4a-csv", type=str, default="runs/v4a_r18/metrics.csv")
    parser.add_argument("--v4b-csv", type=str, default="runs/v4b_r18/metrics.csv")
    parser.add_argument("--v4c-csv", type=str, default="runs/v4c_r18/metrics.csv")
    parser.add_argument("--out", type=str, default="runs/v4_comparison_r18.png")
    args = parser.parse_args()

    series_spec: List[Tuple[str, Path, str, str]] = [
        ("V3 Baseline (1D Spin)", Path(args.v3_csv), "#555555", "o--"),
        ("V4A (S^1 Complex)", Path(args.v4a_csv), "#1f77b4", "s-"),
        ("V4B (S^2 Bloch Mat2C)", Path(args.v4b_csv), "#2ca02c", "^-"),
        ("V4C (S^3 Quat H)", Path(args.v4c_csv), "#d62728", "D-"),
    ]

    loaded: List[Tuple[str, List[Dict[str, str]], str, str]] = []
    for label, path, color, style in series_spec:
        rows = _load_eval_rows(path)
        if rows:
            loaded.append((label, rows, color, style))

    if not loaded:
        raise FileNotFoundError("No valid eval rows found in any input metrics.csv.")

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.8), dpi=180)
    ln2 = math.log(2.0)

    # Panel 1: Best-of-K m=32 Physics Energy Trajectory
    ax = axes[0, 0]
    for label, rows, color, style in loaded:
        steps = [int(r["step"]) for r in rows]
        min_e32 = [_f(r, "min_phys_energy_m32") for r in rows]
        ax.plot(steps, min_e32, style, color=color, linewidth=2.0, markersize=4.5, label=f"{label} (min={min(min_e32):.2f})")
    ax.set_title("Closed-Loop Breakthrough: Best-of-K 2-adic m=32 Energy")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("min_phys_energy_m32 (lower = closer to exact)")
    ax.legend(loc="best", fontsize=8.3)

    # Panel 2: Holdout Calibration BCE vs Blind ln(2)
    ax = axes[0, 1]
    for label, rows, color, style in loaded:
        steps = [int(r["step"]) for r in rows]
        bces = [_f(r, "eval_bce_loss") for r in rows]
        ax.plot(steps, bces, style, color=color, linewidth=2.0, markersize=4.5, label=f"{label} (last={bces[-1]:.4f})")
    ax.axhline(ln2, color="black", linestyle="--", linewidth=1.2, label=f"ln(2)={ln2:.4f}")
    ax.set_title("Deterministic Holdout BCE Calibration")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("eval_bce_loss")
    ax.legend(loc="best", fontsize=8.3)

    # Panel 3: Per-Round Accuracy Profile r0..r17 (at best min_phys_energy_m32 step)
    ax = axes[1, 0]
    rounds_arr = np.arange(18)
    for label, rows, color, style in loaded:
        best_r = min(rows, key=lambda r: _f(r, "min_phys_energy_m32", 1e9))
        r_vals = [_f(best_r, f"per_round_acc_{t}", 0.5) for t in range(18)]
        ax.plot(rounds_arr, r_vals, style, color=color, linewidth=2.0, markersize=4.5, label=f"{label} @ step {best_r['step']}")
    ax.axhline(0.50, color="black", linestyle=":", linewidth=1.1)
    ax.axvline(15.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title("R=18 Per-Round Accuracy Sawtooth (Best-of-K Particle)")
    ax.set_xlabel("Round t (0=W0, 13=Wall, 15=Clamped, 16..17=Expanded W)")
    ax.set_ylabel("Per-Round Free-Bit Accuracy")
    ax.set_xticks(rounds_arr)
    ax.set_ylim(0.42, 1.05)
    ax.legend(loc="upper left", fontsize=8.2)

    # Panel 4: Key Quantitative Summary Bars
    ax = axes[1, 1]
    metric_keys = [
        ("anneal_latch_ratio", "Latch\nRatio"),
        ("w0_free_bit_acc", "W0 Free\nAcc (BestK)"),
        ("zero_w0_bit_acc", "Zero W0\nAcc"),
        ("exact_preimage_rate@256", "Exact\nPreimage"),
        ("unique_ratio@256", "Unique\nRatio"),
    ]
    n_m = len(metric_keys)
    n_s = len(loaded)
    x = np.arange(n_m)
    w = 0.78 / max(1, n_s)
    for idx, (label, rows, color, _) in enumerate(loaded):
        best_r = min(rows, key=lambda r: _f(r, "min_phys_energy_m32", 1e9))
        vals = [_f(best_r, k) for k, _ in metric_keys]
        offset = (idx - 0.5 * (n_s - 1)) * w
        bars = ax.bar(x + offset, vals, w, color=color, alpha=0.86, label=label.split(" ")[0])
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + 0.015,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                rotation=35,
            )
    ax.axhline(0.50, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metric_keys], fontsize=8.8)
    ax.set_ylim(0.0, 1.15)
    ax.set_title("R=18 Breakthrough Summary (V3 vs V4A / V4B / V4C)")
    ax.legend(loc="best", fontsize=8.2)

    fig.suptitle(
        "Phase-SHA V4 Manifold Ladder (R=18 Full 32-bit): V3 vs V4A (S^1) vs V4B (S^2 Bloch) vs V4C (S^3 Quat)",
        fontsize=12.8,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved V4 comparison figure to {out_path}", flush=True)


if __name__ == "__main__":
    main()
