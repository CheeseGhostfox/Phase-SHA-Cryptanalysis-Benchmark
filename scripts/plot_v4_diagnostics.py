"""Generate comprehensive 6-panel Phase-SHA V4 post-mortem diagnostic chart from Colab A100 runs."""

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


def _load_csv(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def _series(rows: List[Dict[str, str]], col: str) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        v = r.get(col, "")
        if v not in ("", None):
            try:
                f = float(v)
                if math.isfinite(f):
                    xs.append(float(r["step"]))
                    ys.append(f)
            except ValueError:
                pass
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _eval_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [r for r in rows if r.get("eval_bce_loss", "") not in ("", None)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 6-panel Phase-SHA V4 diagnostic chart.")
    parser.add_argument("--v3-csv", type=str, default="runs/r18_32b_v3/metrics.csv")
    parser.add_argument("--v4a-csv", type=str, default="runs/v4a_r18_colab/metrics.csv")
    parser.add_argument("--v4b-csv", type=str, default="runs/v4b_r18_colab/metrics.csv")
    parser.add_argument("--v4c-csv", type=str, default="runs/v4c_r18_colab/metrics.csv")
    parser.add_argument("--out", type=str, default="runs/v4_diagnostics_r18_colab.png")
    args = parser.parse_args()

    specs = [
        ("V3 Baseline (1D)", Path(args.v3_csv), "#4b5563", "--"),
        ("V4A (S^1 Complex, stopped @6200)", Path(args.v4a_csv), "#2563eb", "-"),
        ("V4B (S^2 Bloch, probe @2000)", Path(args.v4b_csv), "#16a34a", "-"),
        ("V4C (S^3 Quat, full @10000)", Path(args.v4c_csv), "#dc2626", "-"),
    ]
    loaded = []
    for lbl, p, c, ls in specs:
        rows = _load_csv(p)
        if rows:
            loaded.append((lbl, rows, c, ls))
    if not loaded:
        raise FileNotFoundError("No V3/V4 CSV files loaded.")

    plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(2, 3, figsize=(19.0, 11.2), dpi=180)
    ln2 = math.log(2.0)

    # Panel (1,1): Training Total Loss & Physics Loss
    ax = axes[0, 0]
    for lbl, rows, c, ls in loaded:
        xs, ys = _series(rows, "train_loss")
        xp, yp = _series(rows, "train_phys_loss")
        if len(xs) > 0:
            ax.plot(xs, ys, linestyle=ls, color=c, linewidth=1.8, label=f"{lbl.split(' ')[0]} loss={ys[-1]:.4f}")
        if len(xp) > 0:
            ax.plot(xp, yp, linestyle=":", color=c, alpha=0.85, linewidth=1.5, label=f"{lbl.split(' ')[0]} phys={yp[-1]:.4f}")
    ax.set_title("Training Loss (solid) & Normalized Phys Loss (dotted)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.legend(loc="best", fontsize=7.5, ncol=2)

    # Panel (1,2): Manifold Comfort-Zone Diagnosis (xor, align, comm)
    ax = axes[0, 1]
    for lbl, rows, c, ls in loaded:
        xx, yx = _series(rows, "step") if False else _series(rows, "train_xor_loss")
        xa, ya = _series(rows, "zero_manifold_align")
        if len(ya) == 0:
            xa, ya = _series(rows, "train_align_loss")
        xc, yc = _series(rows, "zero_manifold_comm")
        if len(xx) > 0:
            ax.plot(xx, yx, linestyle=":", color=c, alpha=0.75, linewidth=1.5, label=f"{lbl.split(' ')[0]} xor={yx[-1]:.3f}")
        if len(xa) > 0:
            ax.plot(xa, ya, linestyle=ls, marker="o", markersize=3.0, color=c, linewidth=1.9, label=f"{lbl.split(' ')[0]} align={ya[-1]:.3f}")
        if len(xc) > 0 and float(yc.max()) > 1e-6:
            ax.plot(xc, yc, linestyle="--", marker="^", markersize=2.8, color=c, alpha=0.8, linewidth=1.4, label=f"{lbl.split(' ')[0]} comm={yc[-1]:.3f}")
    ax.set_title("Manifold Comfort-Zone Diagnosis: Align Plateau vs XOR / Comm")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Residual / Divergence")
    ax.legend(loc="best", fontsize=7.2, ncol=2)

    # Panel (1,3): Min m=32 & Global Min m=32 2-adic Energy
    ax = axes[0, 2]
    for lbl, rows, c, ls in loaded:
        xs, ym = _series(rows, "min_phys_energy_m32")
        xg, yg = _series(rows, "best_global_m32_energy")
        if len(xs) > 0:
            ax.plot(xs, ym, linestyle=ls, marker="o", markersize=3.2, color=c, linewidth=1.8, label=f"{lbl.split(' ')[0]} minE32={ym.min():.2f}")
        if len(xg) > 0:
            ax.plot(xg, yg, linestyle="--", marker="s", markersize=2.6, color=c, alpha=0.8, linewidth=1.3, label=f"{lbl.split(' ')[0]} glob={yg.min():.2f}")
    ax.set_title("Best-of-K (solid) & Global Min (dashed) 2-adic m=32 Energy")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Energy (lower = closer to exact trace)")
    ax.legend(loc="best", fontsize=7.4, ncol=2)

    # Panel (2,1): Holdout BCE Calibration & Zero W0 Acc
    ax = axes[1, 0]
    ax2 = ax.twinx()
    for lbl, rows, c, ls in loaded:
        xb, yb = _series(rows, "eval_bce_loss")
        xw, yw = _series(rows, "zero_w0_bit_acc")
        if len(xb) > 0:
            ax.plot(xb, yb, linestyle=ls, marker="o", markersize=3.0, color=c, linewidth=1.8, label=f"{lbl.split(' ')[0]} bce={yb[-1]:.4f}")
        if len(xw) > 0:
            ax2.plot(xw, yw, linestyle=":", color=c, alpha=0.75, linewidth=1.4)
    ax.axhline(ln2, color="black", linestyle="--", linewidth=1.2, label=f"ln(2)={ln2:.4f}")
    ax2.axhline(0.50, color="gray", linestyle=":", linewidth=1.0)
    ax.set_title("Holdout Eval BCE vs ln(2) & Zero W0 Acc (dotted)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("eval_bce_loss")
    ax2.set_ylabel("zero_w0_bit_acc")
    ax2.set_ylim(0.45, 0.56)
    ax2.grid(False)
    ax.legend(loc="best", fontsize=7.6)

    # Panel (2,2): R=18 Per-Round Accuracy Profile & r13/r16 Dead Walls
    ax = axes[1, 1]
    r_idx = np.arange(18)
    ax.axvspan(12.6, 13.4, color="#fecaca", alpha=0.5, label="r13 Wall")
    ax.axvspan(15.6, 16.4, color="#fed7aa", alpha=0.5, label="r16 Wall")
    for lbl, rows, c, ls in loaded:
        evs = _eval_rows(rows)
        if not evs:
            continue
        best_r = min(evs, key=lambda r: float(r.get("min_phys_energy_m32", 1e9) or 1e9))
        vals = [float(best_r.get(f"zero_round_acc_{t}", 0.5) or 0.5) for t in range(18)]
        ax.plot(r_idx, vals, linestyle=ls, marker="o", markersize=3.5, color=c, linewidth=1.8, label=f"{lbl.split(' ')[0]} @{best_r['step']} (r10={vals[10]:.2f}, r12={vals[12]:.2f})")
    ax.axhline(0.50, color="black", linestyle=":", linewidth=1.0)
    ax.set_xticks(r_idx)
    ax.set_ylim(0.45, 1.04)
    ax.set_title("R=18 Per-Round Deterministic Accuracy & r13/r16 Closed-Loop Walls")
    ax.set_xlabel("Round t (0=W0 .. 17=Expanded W)")
    ax.set_ylabel("Zero-Init Bit Accuracy")
    ax.legend(loc="upper left", fontsize=7.3)

    # Panel (2,3): W0 Bit-Position 0..31 Profile + Summary Table
    ax = axes[1, 2]
    b_idx = np.arange(32)
    summary_lines = ["Variant | Step | minE32 | globE32 | r10 | r12 | align | lsb03_z | pre@K"]
    for lbl, rows, c, ls in loaded:
        evs = _eval_rows(rows)
        if not evs:
            continue
        last_e = evs[-1]
        best_e = min(evs, key=lambda r: float(r.get("min_phys_energy_m32", 1e9) or 1e9))
        vals = [float(best_e.get(f"zero_w0_bit_acc_pos_{k}", 0.5) or 0.5) for k in range(32)]
        ax.plot(b_idx, vals, linestyle=ls, marker=".", markersize=4.5, color=c, linewidth=1.4, label=f"{lbl.split(' ')[0]} @{best_e['step']}")
        summary_lines.append(
            f"{lbl.split(' ')[0]:5s} | {int(last_e['step']):5d} | {float(best_e['min_phys_energy_m32']):5.2f} | "
            f"{float(best_e['best_global_m32_energy']):5.2f} | {float(last_e.get('zero_round_acc_10', 0.5)):.3f} | "
            f"{float(last_e.get('zero_round_acc_12', 0.5)):.3f} | {float(last_e.get('zero_manifold_align', 0.0) or 0.0):.3f} | "
            f"{float(last_e.get('zero_w0_lsb03_zscore', 0.0) or 0.0):+5.2f} | {float(last_e.get('exact_preimage_rate@256', 0.0) or 0.0):.3f}"
        )
    ax.axhline(0.50, color="black", linestyle=":", linewidth=1.0)
    ax.set_ylim(0.38, 0.62)
    ax.set_xticks(np.arange(0, 32, 4))
    ax.set_title("W0 Bit-Position 0..31 Accuracy & Post-Mortem Metrics")
    ax.set_xlabel("W0 Bit Index k (0=LSB, 31=MSB)")
    ax.set_ylabel("Zero-Init W0 Bit Acc")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.text(
        0.02,
        0.02,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        fontsize=6.8,
        family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.92),
    )

    fig.suptitle(
        "Phase-SHA V4 Colab A100 Strict-FP32 Post-Mortem Diagnostics: V3 vs V4A (S^1) vs V4B (S^2) vs V4C (S^3)",
        fontsize=12.8,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved V4 post-mortem diagnostic figure to {out_p}", flush=True)


if __name__ == "__main__":
    main()
