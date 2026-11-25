"""Compare Phase-SHA V3/V4 baselines vs V5 suite (V5D_pure, V5D_neural, V5C, V5B, V5A).

Produces a dual-axis multi-panel diagnostic & scientific comparison figure:
  - By Training Step (equal iteration budget)
  - By Cumulative Compute Time wall_s (equal GPU wall-clock budget)
  - R=18 per-round dead-wall sawtooth (highlighting r13 & r16)
  - V5 orthogonal mechanism probes (RRR gap/drift, MPS entanglement entropy, BM rank1/eff_rank, Oct assoc)
"""

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


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def _series(rows: List[Dict[str, str]], x_col: str, y_col: str) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        xv = r.get(x_col, "")
        yv = r.get(y_col, "")
        if xv not in ("", None) and yv not in ("", None):
            try:
                xf, yf = float(xv), float(yv)
                if math.isfinite(xf) and math.isfinite(yf):
                    xs.append(xf)
                    ys.append(yf)
            except ValueError:
                pass
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _eval_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [r for r in rows if r.get("min_phys_energy_m32", "") not in ("", None)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Phase-SHA V3/V4 vs V5 dual-axis comparison.")
    parser.add_argument("--v3-csv", type=str, default="runs/r18_32b_v3/metrics.csv")
    parser.add_argument("--v4a-csv", type=str, default="runs/v4a_r18_colab/metrics.csv")
    parser.add_argument("--v4b-csv", type=str, default="runs/v4b_r18_colab/metrics.csv")
    parser.add_argument("--v4c-csv", type=str, default="runs/v4c_r18_colab/metrics.csv")
    parser.add_argument("--v5d-pure-csv", type=str, default="runs/v5d_pure_r18_colab/metrics.csv")
    parser.add_argument("--v5d-neural-csv", type=str, default="runs/v5d_neural_r18_colab/metrics.csv")
    parser.add_argument("--v5c-csv", type=str, default="runs/v5c_r18_colab/metrics.csv")
    parser.add_argument("--v5b-csv", type=str, default="runs/v5b_r18_colab/metrics.csv")
    parser.add_argument("--v5a-csv", type=str, default="runs/v5a_r18_colab/metrics.csv")
    parser.add_argument("--out", type=str, default="runs/v5_comparison_r18_colab.png")
    args = parser.parse_args()

    specs: List[Tuple[str, Path, str, str]] = [
        ("V3 Baseline (1D)", Path(args.v3_csv), "#6b7280", ":"),
        ("V4A (S^1)", Path(args.v4a_csv), "#60a5fa", "--"),
        ("V4B (S^2 Bloch)", Path(args.v4b_csv), "#34d399", "--"),
        ("V4C (S^3 Quat)", Path(args.v4c_csv), "#f87171", "--"),
        ("V5D-Pure (RRR)", Path(args.v5d_pure_csv), "#7c3aed", "-"),
        ("V5D-Neural (RRR+NN)", Path(args.v5d_neural_csv), "#a855f7", "-."),
        ("V5C (MPS Entangled)", Path(args.v5c_csv), "#0284c7", "-"),
        ("V5B (BM S^31)", Path(args.v5b_csv), "#059669", "-"),
        ("V5A (Octonion S^7)", Path(args.v5a_csv), "#dc2626", "-"),
    ]

    loaded: List[Tuple[str, List[Dict[str, str]], str, str]] = []
    for label, p, color, ls in specs:
        rows = _load_csv(p)
        if rows:
            loaded.append((label, rows, color, ls))

    if not loaded:
        raise FileNotFoundError("No valid metrics.csv found for V5 comparison.")

    plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 11.2), dpi=180)
    ln2 = math.log(2.0)

    # Panel 1: min_phys_energy_m32 vs Training Step
    ax = axes[0, 0]
    for label, rows, color, ls in loaded:
        xs, ys = _series(rows, "step", "min_phys_energy_m32")
        if len(xs) > 0:
            ax.plot(xs, ys, linestyle=ls, marker="o", markersize=3.2, color=color, linewidth=1.8, label=f"{label} ({ys.min():.2f})")
    ax.set_title("Min 2-adic m=32 Energy vs Step")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("min_phys_energy_m32 (lower = closer)")
    ax.legend(loc="best", fontsize=7.5)

    # Panel 2: min_phys_energy_m32 vs Cumulative Compute Wall Time (wall_s)
    ax = axes[0, 1]
    has_wall = False
    for label, rows, color, ls in loaded:
        xs, ys = _series(rows, "wall_s", "min_phys_energy_m32")
        if len(xs) > 0:
            has_wall = True
            ax.plot(xs / 60.0, ys, linestyle=ls, marker="s", markersize=3.2, color=color, linewidth=1.8, label=f"{label} ({xs[-1]/60.0:.1f}m)")
    if not has_wall:
        ax.text(0.5, 0.5, "wall_s not logged yet", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Min 2-adic m=32 Energy vs Wall Time (Equal Compute)")
    ax.set_xlabel("Cumulative GPU Wall Time (minutes)")
    ax.set_ylabel("min_phys_energy_m32")
    if has_wall:
        ax.legend(loc="best", fontsize=7.5)

    # Panel 3: best_global_m32_energy & eval_bce_loss vs Step
    ax = axes[0, 2]
    for label, rows, color, ls in loaded:
        xs, ys = _series(rows, "step", "best_global_m32_energy")
        if len(xs) > 0:
            ax.plot(xs, ys, linestyle=ls, marker="^", markersize=3.0, color=color, linewidth=1.6, label=f"{label} (glob={ys.min():.2f})")
    ax.set_title("Best Global Particle m=32 Energy vs Step")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("best_global_m32_energy")
    ax.legend(loc="best", fontsize=7.4)

    # Panel 4: R=18 Per-Round Accuracy Sawtooth (r0..r17) highlighting dead walls r13 & r16
    ax = axes[1, 0]
    rounds_arr = np.arange(18)
    ax.axvspan(12.6, 13.4, color="#fecaca", alpha=0.45, label="r13 Dead Wall")
    ax.axvspan(15.6, 16.4, color="#fed7aa", alpha=0.45, label="r16 Dead Wall")
    for label, rows, color, ls in loaded:
        evs = _eval_rows(rows)
        if not evs:
            continue
        best_r = min(evs, key=lambda r: float(r.get("min_phys_energy_m32", 1e9) or 1e9))
        vals = [float(best_r.get(f"zero_round_acc_{t}", best_r.get(f"per_round_acc_{t}", 0.5)) or 0.5) for t in range(18)]
        ax.plot(rounds_arr, vals, linestyle=ls, marker="o", markersize=3.2, color=color, linewidth=1.7, label=f"{label} @{best_r['step']}")
    ax.axhline(0.50, color="black", linestyle=":", linewidth=1.0)
    ax.set_xticks(rounds_arr)
    ax.set_ylim(0.42, 1.04)
    ax.set_title("R=18 Per-Round Deterministic Accuracy & Closed-Loop Walls")
    ax.set_xlabel("Round t")
    ax.set_ylabel("Free-Bit Accuracy")
    ax.legend(loc="upper left", fontsize=7.0, ncol=2)

    # Panel 5: Holdout Eval BCE & Continuous Collapse / Entanglement Diagnostics
    ax = axes[1, 1]
    for label, rows, color, ls in loaded:
        xs, ys = _series(rows, "step", "eval_bce_loss")
        if len(xs) > 0:
            ax.plot(xs, ys, linestyle=ls, color=color, linewidth=1.7, label=f"{label} (last={ys[-1]:.4f})")
    ax.axhline(ln2, color="black", linestyle="--", linewidth=1.2, label=f"ln(2)={ln2:.4f}")
    ax.set_title("Holdout Eval BCE Calibration vs ln(2)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("eval_bce_loss")
    ax.legend(loc="best", fontsize=7.3)

    # Panel 6: V5 Specific Orthogonal Probes (Align / Rank1 / Associator / Ent_Entropy / RRR Gap)
    ax = axes[1, 2]
    for label, rows, color, ls in loaded:
        for col, tag_suffix, mk in (
            ("zero_manifold_align", "align", "o"),
            ("rank1_ratio", "rank1", "s"),
            ("ent_entropy", "S_ent", "^"),
            ("rrr_gap", "rrr_gap", "d"),
            ("zero_assoc", "assoc", "x"),
        ):
            xs, ys = _series(rows, "step", col)
            if len(xs) > 0:
                ax.plot(xs, ys, linestyle=ls, marker=mk, markersize=2.8, color=color, linewidth=1.5, label=f"{label.split(' ')[0]}:{tag_suffix}={ys[-1]:.3f}")
    ax.set_title("V5 Mechanism Probes (Collapse / Entanglement / RRR Gap / Assoc)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Probe Value")
    ax.legend(loc="best", fontsize=6.8, ncol=2)

    fig.suptitle(
        "Phase-SHA V5 Single-Variable Ablation vs V3 / V4A / V4B / V4C (Dual-Axis: Step & Wall Time, Strict FP32)",
        fontsize=13.0,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved V5 dual-axis comparison figure to {out_p}", flush=True)


if __name__ == "__main__":
    main()
