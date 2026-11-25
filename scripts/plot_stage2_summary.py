"""Generate Stage 2 cross-round phase transition summary (R=12 -> R=16 -> R=18, 32-bit W0)."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT_DIR / "runs"


def _load_eval_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("eval_bce_loss", "") not in ("", None)]


def main() -> None:
    stages: List[Tuple[str, int, Path, str]] = [
        ("R=12 (32-bit W0)", 12, RUNS_DIR / "r12_32b_v2" / "metrics.csv", "#1f77b4"),
        ("R=16 (32-bit W0)", 16, RUNS_DIR / "r16_32b_v2" / "metrics.csv", "#ff7f0e"),
        ("R=18 (32-bit W0, MsgSched)", 18, RUNS_DIR / "r18_32b_v2" / "metrics.csv", "#d62728"),
    ]

    stage_rows = {}
    for label, r, path, _ in stages:
        if not path.exists():
            raise FileNotFoundError(f"Missing stage metrics file: {path}")
        rows = _load_eval_rows(path)
        if not rows:
            raise RuntimeError(f"No evaluation rows in {path}")
        stage_rows[label] = (r, rows)

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11), dpi=180)

    # Panel 1: Per-round accuracy decay curve aligned by distance from right boundary (R-1-t)
    ax = axes[0, 0]
    for label, r, _, color in stages:
        _, rows = stage_rows[label]
        last = rows[-1]
        dists = [r - 1 - t for t in range(r)]
        accs = [float(last[f"zero_round_acc_{t}"]) for t in range(r)]
        # Plot from right boundary (dist=0) toward W0 input end (dist=R-1)
        ax.plot(
            dists[::-1],
            accs[::-1],
            "o-",
            label=f"{label} (step {last['step']})",
            color=color,
            linewidth=2.0,
            markersize=4.5,
        )
    ax.axhline(0.50, color="black", linestyle=":", linewidth=1.2, label="random guess (0.50)")
    ax.axvline(
        2.5,
        color="gray",
        linestyle="--",
        linewidth=1.1,
        alpha=0.7,
        label="R<=16 clamped boundary (W_16+ free for R=18)",
    )
    ax.set_title("Per-Round Accuracy Decay from Right Boundary (2048 Zero-Init)")
    ax.set_xlabel("Distance Backwards from Output Round (R - 1 - t)")
    ax.set_ylabel("Free-Bit Accuracy")
    ax.set_ylim(0.45, 1.03)
    ax.legend(loc="best", fontsize=8.0)

    # Panel 2: Final Summary Bar Chart
    ax = axes[0, 1]
    metric_names = ["w0_free_bit_acc", "zero_w0_bit_acc", "exact_preimage_rate@256", "eval_bce_loss"]
    x_pos = np.arange(len(metric_names))
    width = 0.25
    for idx, (label, _, _, color) in enumerate(stages):
        _, rows = stage_rows[label]
        last = rows[-1]
        vals = [float(last[m]) for m in metric_names]
        bars = ax.bar(x_pos + (idx - 1) * width, vals, width, label=label, color=color, alpha=0.88)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                v + 0.015,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                rotation=45,
            )
    ax.axhline(0.50, color="gray", linestyle=":", linewidth=1.0)
    ax.axhline(math.log(2.0), color="black", linestyle="--", linewidth=1.0, label="ln(2)=0.693")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["w0_acc (K=256)", "zero_w0 (2048)", "preimage@256", "eval_bce"], fontsize=9)
    ax.set_ylim(0.0, 1.15)
    ax.set_title("Final Stage Metrics Comparison (Full 32-bit W0)")
    ax.legend(loc="best", fontsize=7.8)

    # Panel 3: Zero-init W0 per-bit accuracy (bits 0..31) on 2048 holdout
    ax = axes[1, 0]
    bits = np.arange(32)
    for label, _, _, color in stages:
        _, rows = stage_rows[label]
        last = rows[-1]
        w0_bits = [float(last[f"zero_w0_bit_acc_pos_{k}"]) for k in range(32)]
        ax.plot(bits, w0_bits, "o-", label=label, color=color, linewidth=1.6, markersize=3.5)
    # 95% binomial CI band around 0.50 for N=2048: std = sqrt(0.25/2048) = 0.01105
    ci_2s = 2.0 * math.sqrt(0.25 / 2048.0)
    ax.axhspan(0.50 - ci_2s, 0.50 + ci_2s, color="gray", alpha=0.18, label=f"95% noise band (0.50 +/- {ci_2s:.3f})")
    ax.axvspan(-0.5, 3.5, color="#ffd700", alpha=0.14, label="LSB 0..3 focus")
    ax.axhline(0.50, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("W0 Per-Bit Accuracy on 2048 Unseen Digests (LSB 0..3 Focus)")
    ax.set_xlabel("W0 Bit Position k (0 = LSB, 31 = MSB)")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(np.arange(0, 32, 2))
    ax.legend(loc="best", fontsize=8.0)

    # Panel 4: eval_bce_loss Trajectory Across Stages
    ax = axes[1, 1]
    for label, _, _, color in stages:
        _, rows = stage_rows[label]
        steps = [int(r["step"]) for r in rows]
        bces = [float(r["eval_bce_loss"]) for r in rows]
        ax.plot(steps, bces, "o-", label=label, color=color, linewidth=2.0, markersize=4.5)
    ax.axhline(math.log(2.0), color="black", linestyle="--", linewidth=1.3, label="blind baseline ln(2)=0.6931")
    ax.set_title("Holdout Zero-Init BCE Evolution Across Curriculum")
    ax.set_xlabel("Training Step within Stage")
    ax.set_ylabel("eval_bce_loss")
    ax.legend(loc="best", fontsize=8.2)

    fig.suptitle(
        "Phase-SHA V2 Stage 2: Full 32-bit Phase Transition (R=12 -> R=16 -> R=18)",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path = RUNS_DIR / "phase_transition_R12_to_R18.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Stage 2 summary plot to {out_path}", flush=True)


if __name__ == "__main__":
    main()
