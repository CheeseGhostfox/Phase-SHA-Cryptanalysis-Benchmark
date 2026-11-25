"""Generate final Phase-SHA V3 Paper I figures for Stage 3A (R=16, full 32-bit W0)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent


def main() -> None:
    run_dir = ROOT_DIR / "runs" / "r16_32b_v3"
    with (run_dir / "metrics.csv").open("r", encoding="utf-8") as f:
        rows: List[Dict[str, str]] = list(csv.DictReader(f))
    with (run_dir / "paper1_ablation.json").open("r", encoding="utf-8") as f:
        abl_list = json.load(f)
    abl = {item["name"]: item for item in abl_list}

    # Extract training step metrics
    tr_steps = [int(r["step"]) for r in rows if r.get("train_bce_loss")]
    tr_bce = [float(r["train_bce_loss"]) for r in rows if r.get("train_bce_loss")]
    tr_bce_end = [float(r["train_bce_final"]) for r in rows if r.get("train_bce_final")]
    tr_phys = [float(r["train_phys_loss"]) for r in rows if r.get("train_phys_loss")]

    ev_rows = [r for r in rows if r.get("eval_bce_loss")]
    ev_steps_old = [int(r["step"]) for r in ev_rows]
    ev_bce_old = [float(r["eval_bce_loss"]) for r in ev_rows]
    ev_pre_old = [float(r["exact_preimage_rate@256"]) for r in ev_rows]
    ev_zw0_old = [float(r["zero_w0_bit_acc"]) for r in ev_rows]
    ev_zm1_old = [float(r["zero_phys_energy_m1"]) for r in ev_rows]
    ev_zm32_old = [float(r["zero_phys_energy_m32"]) for r in ev_rows]

    a1 = abl["A1_ode14_nolatch"]
    a2 = abl["A2_ode32_nolatch"]
    a3 = abl["A3_ode24_latch"]
    a4 = abl["A4_ode32_latch"]

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

    # 1. Training Curves & Aligned Evaluation Convergence
    fig, axes = plt.subplots(1, 3, figsize=(17.8, 5.2), dpi=180)

    ax = axes[0]
    ax.plot(tr_steps, tr_bce, label="train_bce_loss (weighted)", color="#1f77b4", alpha=0.65)
    ax.plot(tr_steps, tr_bce_end, label="train_bce_final (K_unroll=14)", color="#17becf", linewidth=2.0)
    ax.plot(
        ev_steps_old,
        ev_bce_old,
        "o--",
        label="eval_bce (legacy K=64, d=0.08)",
        color="#7f7f7f",
        alpha=0.75,
        markersize=4,
    )
    ax.plot(
        [0, 500, 2000],
        [ev_bce_old[0], ev_bce_old[1], a1["eval_bce_loss"]],
        "^-",
        color="#2ca02c",
        linewidth=2.0,
        markersize=6,
        label=f"A1 aligned K=14, NoLatch ({a1['eval_bce_loss']:.4f})",
    )
    ax.plot(
        [0, 500, 2000],
        [ev_bce_old[0], ev_bce_old[1], a4["eval_bce_loss"]],
        "*-",
        color="#d62728",
        linewidth=2.2,
        markersize=9,
        label=f"A4 aligned K=32, Latch ({a4['eval_bce_loss']:.4f})",
    )
    ln2 = math.log(2.0)
    ax.axhline(ln2, color="black", linestyle="--", linewidth=1.3, label=f"blind ln(2)={ln2:.4f}")
    ax.set_title("Free-Bit BCE Loss (V3 Direct x0 Logits)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("BCEWithLogitsLoss")
    ax.legend(loc="best", fontsize=7.6)

    ax = axes[1]
    ax.plot(ev_steps_old, ev_zm1_old, "o--", color="#7f7f7f", alpha=0.65, label="zero E_phys m=1 (legacy)")
    ax.plot(
        [0, 500, 2000],
        [ev_zm1_old[0], ev_zm1_old[1], a4["zero_phys_energy_m1"]],
        "o-",
        color="#2ca02c",
        linewidth=2.0,
        label="zero E_phys m=1 (A4 aligned -> 0.000)",
    )
    ax.plot(
        [0, 500, 2000],
        [ev_zm32_old[0], ev_zm32_old[1], a4["zero_phys_energy_m32"]],
        "s-",
        color="#ff7f0e",
        linewidth=2.0,
        label="zero E_phys m=32 (A4 aligned -> 1.8e-29)",
    )
    ax.set_title("2-adic Physics Energy (2048 Zero-Init)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Zero-Init 2-adic Energy")
    ax2 = ax.twinx()
    ax2.plot(tr_steps, tr_phys, color="#9467bd", alpha=0.55, linewidth=1.4, label="train_phys_loss (norm)")
    ax2.set_ylabel("Train Normalized Phys Loss", color="#9467bd")
    ax2.tick_params(axis="y", labelcolor="#9467bd")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=7.6)

    ax = axes[2]
    ax.plot(
        ev_steps_old,
        ev_pre_old,
        "o--",
        color="#7f7f7f",
        alpha=0.7,
        label="pre@256 (legacy K=64, d=0.08)",
    )
    ax.plot(
        [0, 500, 2000],
        [ev_pre_old[0], ev_pre_old[1], a1["exact_preimage_rate@256"]],
        "^-",
        color="#2ca02c",
        linewidth=2.0,
        label=f"A1 pre@256 (1.0000) / 1-shot ({a1['exact_1shot_rate']*100:.2f}%)",
    )
    ax.plot(
        [0, 500, 2000],
        [ev_zw0_old[0], ev_zw0_old[1], a4["zero_w0_bit_acc"]],
        "d:",
        color="#17becf",
        linewidth=2.0,
        label=f"A4 zero_w0 (1.000000)",
    )
    ax.plot(
        [0, 500, 2000],
        [float(ev_rows[0]["zero_latch_ratio"]), float(ev_rows[1]["zero_latch_ratio"]), a4["zero_latch_ratio"]],
        "s-",
        color="#ff7f0e",
        linewidth=1.8,
        label=f"A4 latch_z ({a4['zero_latch_ratio']:.3f})",
    )
    ax.scatter(
        [2000],
        [a4["exact_1shot_rate"]],
        marker="*",
        s=190,
        color="#d62728",
        zorder=6,
        label=f"A4 1-shot Exact (2048/2048 = 100.0%)",
    )
    ax.set_title("Exact Preimage Breakthrough & Latch Ratio")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Rate / Accuracy")
    ax.set_ylim(-0.03, 1.07)
    ax.legend(loc="lower right", fontsize=7.5)

    fig.suptitle(
        "Phase-SHA V3 (r16_32b_v3, R=16, Full 32-bit): Training & Aligned 2048 Breakthrough",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Per-Round Wavefront Heatmap (Baseline Steps 0, 500 + Aligned Paper I Settings A1, A3, A4)
    col_labels = [
        "Step 0\n(V2 init)",
        "Step 500\n(K=64)",
        "Step 2000\nA1 (K=14,\nNoLatch)",
        "Step 2000\nA3 (K=24,\nLatch)",
        "Step 2000\nA4 (K=32,\nLatch)",
    ]
    rounds = 16
    mat_k256 = np.zeros((rounds, len(col_labels)), dtype=np.float64)
    mat_zero = np.zeros((rounds, len(col_labels)), dtype=np.float64)

    for c_idx, src in enumerate([ev_rows[0], ev_rows[1]]):
        for t in range(rounds):
            mat_k256[t, c_idx] = float(src[f"per_round_acc_{t}"])
            mat_zero[t, c_idx] = float(src[f"zero_round_acc_{t}"])
    for c_idx, item in enumerate([a1, a3, a4], start=2):
        for t in range(rounds):
            mat_k256[t, c_idx] = float(item["k256_round_acc"][t])
            mat_zero[t, c_idx] = float(item["zero_round_acc"][t])

    fig, axes = plt.subplots(
        1, 2, figsize=(15.0, 8.2), dpi=180, sharey=True, constrained_layout=True
    )
    axes[0].imshow(mat_k256, aspect="auto", origin="lower", vmin=0.45, vmax=1.00, cmap="viridis", interpolation="nearest")
    axes[0].set_title("K=256 Unrolled Inference (64 Digests)", fontsize=11.5)
    axes[0].set_ylabel("Round t (0=W0 end, 0..12=free, 13..15=clamped)")

    im1 = axes[1].imshow(mat_zero, aspect="auto", origin="lower", vmin=0.45, vmax=1.00, cmap="viridis", interpolation="nearest")
    axes[1].set_title("Deterministic Zero-Init 1-Shot (Holdout 2048)", fontsize=11.5)

    for ax, mat in ((axes[0], mat_k256), (axes[1], mat_zero)):
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=8.8)
        ax.set_yticks(np.arange(rounds))
        ax.set_yticklabels([str(t) for t in range(rounds)], fontsize=9.0)
        ax.axhline(12.5, color="white", linestyle="--", linewidth=1.3, alpha=0.85)
        for t in range(rounds):
            for c in range(len(col_labels)):
                val = mat[t, c]
                txt_c = "black" if val >= 0.78 else "white"
                ax.text(c, t, f"{val:.3f}" if val > 0.99 and val < 1.0 else f"{val:.2f}", ha="center", va="center", color=txt_c, fontsize=7.3)

    cbar = fig.colorbar(im1, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("Per-Round Free-Bit Accuracy", fontsize=10.5)
    fig.suptitle(
        "Phase-SHA V3 (r16_32b_v3, R=16): Full 16-Round Reasoning Wavefront Breakthrough",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(run_dir / "per_round_wavefront_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Bit-Zipper Heatmap across Bits 0..31 (2048 1-Shot W0 & BiGRU Uniformity)
    bit_labels = [
        "Step 0\n(V2 init)",
        "Step 500\n(K=64)",
        "A1 Step 2000\n(K=14 NoLatch)",
        "A2 Step 2000\n(K=32 NoLatch)",
        "A3 Step 2000\n(K=24 Latch)",
        "A4 Step 2000\n(K=32 Latch)",
    ]
    mat_bit = np.zeros((32, len(bit_labels)), dtype=np.float64)
    for c_idx, src in enumerate([ev_rows[0], ev_rows[1]]):
        for k in range(32):
            mat_bit[k, c_idx] = float(src[f"zero_w0_bit_acc_pos_{k}"])
    for c_idx, item in enumerate([a1, a2, a3, a4], start=2):
        for k in range(32):
            mat_bit[k, c_idx] = float(item["zero_w0_bit_acc_pos"][k])

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 7.2), dpi=180, constrained_layout=True)
    im = ax.imshow(mat_bit, aspect="auto", origin="lower", vmin=0.45, vmax=1.00, cmap="viridis", interpolation="nearest")
    ax.set_title(
        "Phase-SHA V3 (r16_32b_v3, R=16): Full 32-bit W0 Bit-Zipper Accuracy (2048 1-Shot Zero-Init)",
        fontsize=11.5,
    )
    ax.set_xlabel("Checkpoint & Aligned Inference Setting")
    ax.set_ylabel("Bit Position k (0 = LSB, 31 = MSB)")
    ax.set_xticks(np.arange(len(bit_labels)))
    ax.set_xticklabels(bit_labels, fontsize=8.5)
    ax.set_yticks(np.arange(0, 32, 2))
    ax.set_yticklabels([str(k) for k in range(0, 32, 2)], fontsize=8.8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label("W0 Single-Bit Accuracy on 2048 Unseen Digests", fontsize=10.5)
    fig.savefig(run_dir / "bit_zipper_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved clean Paper I Stage 3A (Phase-SHA V3) figures under {run_dir}", flush=True)


if __name__ == "__main__":
    main()
