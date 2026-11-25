"""Generate Phase-SHA V2 vs V3 (R=18, Full 32-bit) comparison figure."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_eval_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("eval_bce_loss", "") not in ("", None)]


def main() -> None:
    v2_rows = _load_eval_rows(ROOT_DIR / "runs" / "r18_32b_v2" / "metrics.csv")
    v3_rows = _load_eval_rows(ROOT_DIR / "runs" / "r18_32b_v3" / "metrics.csv")

    v2_last = v2_rows[-1]  # step 8000
    v3_best = min(v3_rows, key=lambda r: float(r["eval_bce_loss"]))  # step 1500

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.0), dpi=180)

    # Panel 1: Holdout eval_bce_loss trajectory (Overconfident > ln2 vs Calibrated < ln2)
    ax = axes[0, 0]
    ax.plot(
        [int(r["step"]) for r in v2_rows],
        [float(r["eval_bce_loss"]) for r in v2_rows],
        "o-",
        color="#d62728",
        linewidth=2.0,
        label=f"V2 R=18 (steps 0..8000, best={min(float(r['eval_bce_loss']) for r in v2_rows):.4f})",
    )
    ax.plot(
        [int(r["step"]) for r in v3_rows],
        [float(r["eval_bce_loss"]) for r in v3_rows],
        "s-",
        color="#1f77b4",
        linewidth=2.3,
        markersize=5.5,
        label=f"V3 R=18 Closed-Loop (steps 0..1500, best={float(v3_best['eval_bce_loss']):.4f})",
    )
    ln2 = math.log(2.0)
    ax.axhline(ln2, color="black", linestyle="--", linewidth=1.4, label=f"blind baseline ln(2)={ln2:.4f}")
    ax.set_title("Holdout Calibration: Overconfident V2 vs Calibrated V3")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("eval_bce_loss (2048 Zero-Init)")
    ax.legend(loc="best", fontsize=8.3)

    # Panel 2: Per-Round Accuracy Profile r0..r17 (2048 Zero-Init)
    ax = axes[0, 1]
    rounds = np.arange(18)
    v2_r = [float(v2_last[f"zero_round_acc_{t}"]) for t in range(18)]
    v3_r = [float(v3_best[f"zero_round_acc_{t}"]) for t in range(18)]
    ax.plot(rounds, v2_r, "o--", color="#d62728", linewidth=1.8, markersize=4.5, label=f"V2 Final (step {v2_last['step']})")
    ax.plot(rounds, v3_r, "s-", color="#1f77b4", linewidth=2.2, markersize=5.0, label=f"V3 Best (step {v3_best['step']})")
    ax.axhline(0.50, color="black", linestyle=":", linewidth=1.2)
    ax.axvline(15.5, color="gray", linestyle="--", linewidth=1.1, alpha=0.75)
    ax.annotate("W17=1.00", xy=(17, 1.00), xytext=(15.1, 0.93), fontsize=8.2, color="#1f77b4")
    ax.annotate("E14=1.00", xy=(14, 1.00), xytext=(12.4, 0.95), fontsize=8.2, color="#1f77b4")
    ax.annotate("A10=1.00\n(r10=0.750)", xy=(10, 0.750), xytext=(7.2, 0.79), arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8.0, color="#1f77b4")
    ax.annotate("W16/E13\nWall ~0.50", xy=(13, 0.50), xytext=(11.2, 0.455), fontsize=8.0, color="#333333")
    ax.set_title("R=18 Per-Round Accuracy Sawtooth (2048 Zero-Init)")
    ax.set_xlabel("Round t (0=W0 Input, 15=Clamped, 16..17=Expanded W)")
    ax.set_ylabel("Free-Bit Accuracy")
    ax.set_xticks(rounds)
    ax.set_ylim(0.44, 1.05)
    ax.legend(loc="upper left", fontsize=8.3)

    # Panel 3: All-Round Bit-Position Profile (zero_bit_acc_pos_0..31)
    ax = axes[1, 0]
    bits = np.arange(32)
    v2_b = [float(v2_last[f"zero_bit_acc_pos_{k}"]) for k in range(32)]
    v3_b = [float(v3_best[f"zero_bit_acc_pos_{k}"]) for k in range(32)]
    ax.plot(bits, v2_b, "o--", color="#d62728", linewidth=1.6, markersize=3.8, label=f"V2 step {v2_last['step']} (mean={np.mean(v2_b):.4f})")
    ax.plot(bits, v3_b, "s-", color="#1f77b4", linewidth=2.0, markersize=4.0, label=f"V3 step {v3_best['step']} + BiGRU (mean={np.mean(v3_b):.4f})")
    ax.axhline(0.50, color="black", linestyle=":", linewidth=1.1)
    ax.set_title("All-Round Free-Bit Profile Across k=0..31 (2048 Holdout)")
    ax.set_xlabel("Bit Position k (0 = LSB, 31 = MSB)")
    ax.set_ylabel("Mean Free-Bit Accuracy")
    ax.set_xticks(np.arange(0, 32, 2))
    ax.legend(loc="best", fontsize=8.3)

    # Panel 4: Key Quantitative Summary Bar Chart
    ax = axes[1, 1]
    labels = ["eval_bce\n(lower better)", "zero_latch\nratio", "r14\n(E14)", "r10\n(A10+E10)", "r12\n(E12)", "r11\n(E11)"]
    v2_vals = [
        float(v2_last["eval_bce_loss"]),
        0.0,
        float(v2_last["zero_round_acc_14"]),
        float(v2_last["zero_round_acc_10"]),
        float(v2_last["zero_round_acc_12"]),
        float(v2_last["zero_round_acc_11"]),
    ]
    v3_vals = [
        float(v3_best["eval_bce_loss"]),
        float(v3_best["zero_latch_ratio"]),
        float(v3_best["zero_round_acc_14"]),
        float(v3_best["zero_round_acc_10"]),
        float(v3_best["zero_round_acc_12"]),
        float(v3_best["zero_round_acc_11"]),
    ]
    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, v2_vals, w, label=f"V2 (step {v2_last['step']})", color="#d62728", alpha=0.85)
    b2 = ax.bar(x + w / 2, v3_vals, w, label=f"V3 (step {v3_best['step']})", color="#1f77b4", alpha=0.88)
    for bar, val in list(zip(b1, v2_vals)) + list(zip(b2, v3_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=35)
    ax.axhline(ln2, color="black", linestyle="--", linewidth=1.1, label="ln(2)=0.693")
    ax.axhline(0.50, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.8)
    ax.set_ylim(0.0, 1.15)
    ax.set_title("R=18 V2 vs V3 Calibration & Algebraic Breakthrough Metrics")
    ax.legend(loc="best", fontsize=8.2)

    fig.suptitle(
        "Phase-SHA V2 vs V3 (R=18, Full 32-bit): Calibration, Latch & Carry Analysis",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path = ROOT_DIR / "runs" / "v2_vs_v3_r18_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved V2 vs V3 R=18 comparison plot to {out_path}", flush=True)


if __name__ == "__main__":
    main()
