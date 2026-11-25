"""Generate Stage 1 (r12_sanity) training curves and 2-adic bit-zipper heatmap."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT_DIR / "runs" / "r12_sanity"


def load_metrics_csv(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _extract_series(rows: List[Dict[str, str]], col: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        val = r.get(col, "")
        if val != "" and val is not None:
            xs.append(int(r["step"]))
            ys.append(float(val))
    return np.asarray(xs, dtype=np.int64), np.asarray(ys, dtype=np.float64)


def plot_training_curves(
    rows: List[Dict[str, str]], step0_data: dict, out_path: Path
) -> None:
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=180)

    # Panel 1: Flow Matching Loss
    ax = axes[0]
    s_tr_fm, v_tr_fm = _extract_series(rows, "train_fm_loss")
    s_ev_fm, v_ev_fm = _extract_series(rows, "eval_fm_loss")
    if len(s_tr_fm) > 0:
        ax.plot(s_tr_fm, v_tr_fm, label="train_fm_loss", color="#1f77b4", alpha=0.8)
    if len(s_ev_fm) > 0:
        ax.plot(
            s_ev_fm,
            v_ev_fm,
            "o-",
            label="eval_fm_loss (holdout 2048)",
            color="#d62728",
            linewidth=2.0,
            markersize=4.5,
        )
    ax.set_title("Flow Matching Loss ($L_{FM}$)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("MSE Loss")
    ax.legend(loc="best", fontsize=8.5)

    # Panel 2: 2-adic Physics Energy
    ax = axes[1]
    s_m1, v_m1 = _extract_series(rows, "phys_energy_m1")
    s_m32, v_m32 = _extract_series(rows, "phys_energy_m32")
    s_tp, v_tp = _extract_series(rows, "train_phys_loss")

    if len(s_m1) > 0:
        ax.plot(
            s_m1,
            v_m1,
            "o-",
            label="anneal E_phys (m=1, LSB)",
            color="#2ca02c",
            linewidth=2.0,
            markersize=4,
        )
    if len(s_m32) > 0:
        ax.plot(
            s_m32,
            v_m32,
            "s-",
            label="anneal E_phys (m=32, MSB)",
            color="#ff7f0e",
            linewidth=2.0,
            markersize=4,
        )

    phys0 = step0_data.get("physics_only", {})
    if "phys_energy_m1" in phys0:
        ax.axhline(
            float(phys0["phys_energy_m1"]),
            color="#2ca02c",
            linestyle="--",
            alpha=0.6,
            label=f"step0 physics-only m1 ({phys0['phys_energy_m1']:.2f})",
        )
    if "phys_energy_m32" in phys0:
        ax.axhline(
            float(phys0["phys_energy_m32"]),
            color="#ff7f0e",
            linestyle="--",
            alpha=0.6,
            label=f"step0 physics-only m32 ({phys0['phys_energy_m32']:.2f})",
        )

    ax.set_title("2-adic Physics Energy ($E_{phys}$)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Holdout Anneal Energy")

    if len(s_tp) > 0:
        ax2 = ax.twinx()
        ax2.plot(
            s_tp,
            v_tp,
            label="train_phys_loss (norm)",
            color="#9467bd",
            alpha=0.55,
            linewidth=1.3,
        )
        ax2.set_ylabel("Train Normalized Phys Loss", color="#9467bd")
        ax2.tick_params(axis="y", labelcolor="#9467bd")
        ax2.grid(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=7.5)
    else:
        ax.legend(loc="best", fontsize=7.5)

    # Panel 3: Preimage Rate@256 and W0 Accuracy
    ax = axes[2]
    s_pre, v_pre = _extract_series(rows, "exact_preimage_rate@256")
    s_w0, v_w0 = _extract_series(rows, "w0_free_bit_acc")
    s_b0, v_b0 = _extract_series(rows, "blind1_w0_bit_acc")

    if len(s_pre) > 0:
        ax.plot(
            s_pre,
            v_pre,
            "o-",
            label="exact_preimage_rate@256",
            color="#d62728",
            linewidth=2.2,
            markersize=5,
        )
    if len(s_w0) > 0:
        ax.plot(
            s_w0,
            v_w0,
            "^-",
            label="w0_free_bit_acc (annealed)",
            color="#1f77b4",
            linewidth=1.8,
            markersize=4,
        )
    if len(s_b0) > 0:
        ax.plot(
            s_b0,
            v_b0,
            "d--",
            label="blind1_w0_bit_acc (tau=0)",
            color="#17becf",
            linewidth=1.6,
            markersize=4,
        )
    if "exact_preimage_rate@256" in phys0:
        ax.axhline(
            float(phys0["exact_preimage_rate@256"]),
            color="#7f7f7f",
            linestyle=":",
            linewidth=1.4,
            label=f"step0 physics-only ({phys0['exact_preimage_rate@256']:.4f})",
        )
    rand_rate = step0_data.get("audit", {}).get(
        "theoretical_random_preimage_rate_at_K", 0.0039
    )
    ax.axhline(
        float(rand_rate),
        color="black",
        linestyle="-.",
        alpha=0.5,
        linewidth=1.1,
        label=f"random guess@256 ({rand_rate:.4f})",
    )

    ax.set_title("Preimage Recovery & W0 Bit Accuracy")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Rate / Accuracy")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="best", fontsize=8.0)

    fig.suptitle(
        "Phase-SHA Stage 1 (R=12, 16-bit W0, B=512, K=256) Training & Generalization",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_bit_zipper_heatmap(rows: List[Dict[str, str]], out_path: Path) -> None:
    eval_rows = [
        r for r in rows if r.get("bit_acc_pos_0", "") not in ("", None)
    ]
    if not eval_rows:
        raise RuntimeError("No evaluation rows found in metrics.csv for heatmap.")

    steps = [int(r["step"]) for r in eval_rows]
    mat_anneal = np.zeros((32, len(eval_rows)), dtype=np.float64)
    mat_blind1 = np.zeros((32, len(eval_rows)), dtype=np.float64)

    for col_idx, r in enumerate(eval_rows):
        for k in range(32):
            mat_anneal[k, col_idx] = float(r[f"bit_acc_pos_{k}"])
            b_val = r.get(f"blind1_bit_acc_pos_{k}", "")
            mat_blind1[k, col_idx] = float(b_val) if b_val != "" else 0.5

    fig, axes = plt.subplots(
        1, 2, figsize=(15, 7), dpi=180, sharey=True, constrained_layout=True
    )

    vmin = 0.45
    vmax = 1.00
    cmap = "viridis"

    im0 = axes[0].imshow(
        mat_anneal,
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )
    axes[0].set_title("Annealed Reconstruction (K=256, Holdout 64)", fontsize=11.5)
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Bit Position k (0 = LSB, 31 = MSB)")

    im1 = axes[1].imshow(
        mat_blind1,
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )
    axes[1].set_title("Blind 1-Step Prediction (tau=0, Holdout 2048)", fontsize=11.5)
    axes[1].set_xlabel("Training Step")

    xtick_positions = np.arange(len(steps))
    xtick_labels = [str(s) for s in steps]
    for ax in axes:
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticks(np.arange(0, 32, 2))
        ax.set_yticklabels([str(k) for k in range(0, 32, 2)], fontsize=8.5)

    cbar = fig.colorbar(im1, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("Free-Bit Accuracy on Unseen Digests", fontsize=10.5)

    fig.suptitle(
        "Phase-SHA Stage 1 (R=12): 2-adic Bit-Zipper Accuracy Evolution (LSB -> MSB)",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    csv_path = RUN_DIR / "metrics.csv"
    step0_path = RUN_DIR / "step0_baseline.json"
    rows = load_metrics_csv(csv_path)
    step0_data = {}
    if step0_path.exists():
        with step0_path.open("r", encoding="utf-8") as f:
            step0_data = json.load(f)

    curves_out = RUN_DIR / "training_curves.png"
    heatmap_out = RUN_DIR / "bit_zipper_heatmap.png"

    plot_training_curves(rows, step0_data, curves_out)
    plot_bit_zipper_heatmap(rows, heatmap_out)
    print(f"Saved {curves_out} and {heatmap_out}", flush=True)


if __name__ == "__main__":
    main()
