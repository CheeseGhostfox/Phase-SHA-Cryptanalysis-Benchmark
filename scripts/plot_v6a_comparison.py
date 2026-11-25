"""Compare Phase-SHA V6A (Ouroboros) vs V3/V4/V5 baselines on R=18 (Strict Pure FP32).

Generates a 2x2 high-resolution diagnostic comparison figure:
  1) Top-Left: Instantaneous (`inst_E32`) & Historical-Min (`hist_min`) 2-adic m=32 Energy vs Step
  2) Top-Right: Same Energy Metrics vs Cumulative GPU Compute Time (`wall_s`, minutes)
  3) Bottom-Left: V6A 2D Homotopy Progress (`alpha`) & Equivalent Homotopy Residual (`inst_E_hom`)
  4) Bottom-Right: V6A Bottleneck-MPS Entanglement Entropy (`S_ent`, dashed `0.55`) & `gl_gap`
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

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
    xs: List[float] = []
    ys: List[float] = []
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Phase-SHA V6A vs V3/V4/V5 R=18 comparison.")
    parser.add_argument("--v6a-csv", type=str, default="runs/v6a_r18_colab/metrics.csv")
    parser.add_argument("--v3-csv", type=str, default="runs/r18_32b_v3/metrics.csv")
    parser.add_argument("--v4c-csv", type=str, default="runs/v4c_r18_colab/metrics.csv")
    parser.add_argument("--v5d-pure-csv", type=str, default="runs/v5d_pure_r18_colab/metrics.csv")
    parser.add_argument("--v5d-neural-csv", type=str, default="runs/v5d_neural_r18_colab/metrics.csv")
    parser.add_argument("--v5c-csv", type=str, default="runs/v5c_r18_colab/metrics.csv")
    parser.add_argument("--out", type=str, default="runs/v6a_comparison_r18_colab.png")
    args = parser.parse_args()

    baselines: List[Tuple[str, Path, str, str]] = [
        ("V3 Baseline", Path(args.v3_csv), "#6b7280", ":"),
        ("V4C (S^3 Quat)", Path(args.v4c_csv), "#f59e0b", "--"),
        ("V5D-Pure (RRR)", Path(args.v5d_pure_csv), "#7c3aed", "-"),
        ("V5D-Neural", Path(args.v5d_neural_csv), "#a855f7", "-."),
        ("V5C (MPS Old)", Path(args.v5c_csv), "#0284c7", "--"),
    ]
    v6a_rows = _load_csv(Path(args.v6a_csv))
    loaded_base = [(lbl, _load_csv(p), c, ls) for lbl, p, c, ls in baselines]
    loaded_base = [(lbl, rows, c, ls) for lbl, rows, c, ls in loaded_base if rows]

    if not v6a_rows and not loaded_base:
        raise FileNotFoundError(f"No metrics CSV found (checked {args.v6a_csv} and baselines).")

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.2), dpi=160)

    # Panel 1: Same Training Step Energy
    ax = axes[0, 0]
    for lbl, rows, c, ls in loaded_base:
        xs, ys = _series(rows, "step", "min_phys_energy_m32")
        if len(xs) > 0:
            ax.plot(xs, ys, linestyle=ls, color=c, linewidth=1.7, alpha=0.88, label=f"{lbl} (min={ys.min():.2f})")
    if v6a_rows:
        xs, y_inst = _series(v6a_rows, "step", "inst_E32")
        xs_h, y_hist = _series(v6a_rows, "step", "hist_min")
        if len(xs) > 0:
            ax.plot(xs, y_inst, color="#e11d48", linewidth=2.2, label=f"V6A inst_E32 (last={y_inst[-1]:.2f})")
        if len(xs_h) > 0:
            ax.plot(xs_h, y_hist, "--", color="#16a34a", linewidth=2.0, label=f"V6A hist_min ({y_hist.min():.2f})")
    ax.set_title("1. True 2-adic m=32 Energy vs Step (R=18)")
    ax.set_xlabel("Step / Tick")
    ax.set_ylabel("m=32 Energy (lower = closer)")
    ax.legend(loc="best", fontsize=8.2)

    # Panel 2: Same Compute Wall Time (minutes)
    ax = axes[0, 1]
    has_wall = False
    for lbl, rows, c, ls in loaded_base:
        xs, ys = _series(rows, "wall_s", "min_phys_energy_m32")
        if len(xs) > 0:
            has_wall = True
            ax.plot(xs / 60.0, ys, linestyle=ls, color=c, linewidth=1.7, alpha=0.88, label=f"{lbl} ({xs[-1]/60.0:.1f}m)")
    if v6a_rows:
        xs, y_inst = _series(v6a_rows, "wall_s", "inst_E32")
        xs_h, y_hist = _series(v6a_rows, "wall_s", "hist_min")
        if len(xs) > 0:
            has_wall = True
            ax.plot(xs / 60.0, y_inst, color="#e11d48", linewidth=2.2, label=f"V6A inst_E32 ({xs[-1]/60.0:.1f}m)")
        if len(xs_h) > 0:
            ax.plot(xs_h / 60.0, y_hist, "--", color="#16a34a", linewidth=2.0, label="V6A hist_min")
    if not has_wall:
        ax.text(0.5, 0.5, "No wall_s data available", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("2. True 2-adic m=32 Energy vs GPU Wall Time")
    ax.set_xlabel("Cumulative GPU Wall Time (min)")
    ax.set_ylabel("m=32 Energy")
    if has_wall:
        ax.legend(loc="best", fontsize=8.2)

    # Panel 3: V6A Homotopy Progression alpha & inst_E_hom
    ax = axes[1, 0]
    if v6a_rows:
        xs_a, ys_a = _series(v6a_rows, "step", "alpha")
        xs_h, ys_h = _series(v6a_rows, "step", "inst_E_hom")
        if len(xs_a) > 0:
            ax.plot(xs_a, ys_a, color="#2563eb", linewidth=2.2, label=f"Homotopy alpha (end={ys_a[-1]:.3f})")
        ax.set_ylim(-0.05, 1.12)
        ax.set_ylabel("Homotopy alpha in [0, 1]", color="#2563eb")
        ax_r = ax.twinx()
        if len(xs_h) > 0:
            ax_r.plot(xs_h, ys_h, "--", color="#ea580c", linewidth=1.9, label=f"inst_E_hom (last={ys_h[-1]:.4f})")
        ax_r.set_ylabel("Instantaneous Homotopy Residual inst_E_hom", color="#ea580c")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_r.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8.5)
    else:
        ax.text(0.5, 0.5, "Waiting for V6A metrics.csv", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("3. V6A Homotopy Continuation (alpha & inst_E_hom)")
    ax.set_xlabel("Step / Tick")

    # Panel 4: V6A Bottleneck-MPS Entanglement S_ent & GL-Gap
    ax = axes[1, 1]
    if v6a_rows:
        xs_s, ys_s = _series(v6a_rows, "step", "S_ent")
        xs_g, ys_g = _series(v6a_rows, "step", "gl_gap")
        if len(xs_s) > 0:
            ax.plot(xs_s, ys_s, color="#059669", linewidth=2.2, label=f"MPS S_ent (last={ys_s[-1]:.3f})")
        ax.axhline(0.55, color="#dc2626", linestyle=":", linewidth=1.4, label="Floor S_min = 0.55")
        ax.set_ylabel("MPS Entanglement Entropy S_ent", color="#059669")
        ax_r = ax.twinx()
        if len(xs_g) > 0:
            ax_r.plot(xs_g, ys_g, "-.", color="#7c3aed", linewidth=1.8, label=f"gl_gap (last={ys_g[-1]:.4f})")
        ax_r.set_ylabel("GL-RRR Fixed-Point Gap (gl_gap)", color="#7c3aed")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_r.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8.5)
    else:
        ax.text(0.5, 0.5, "Waiting for V6A metrics.csv", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("4. V6A Bottleneck-MPS Entanglement Guardrail & GL-Gap")
    ax.set_xlabel("Step / Tick")

    fig.suptitle(
        "Phase-SHA V6A (Ouroboros) vs V3 / V4C / V5 Suite on R=18 (Strict Pure FP32)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=160)
    plt.close(fig)
    print(f"Saved V6A comparison chart to {out_p}", flush=True)


if __name__ == "__main__":
    main()
