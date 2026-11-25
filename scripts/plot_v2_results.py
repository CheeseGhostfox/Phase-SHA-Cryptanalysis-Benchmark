"""Generate Phase-SHA V2 training curves, per-round wavefront, and bit-zipper heatmaps."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent


def load_metrics_csv(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _infer_rounds(rows: List[Dict[str, str]]) -> int:
    keys = rows[0].keys()
    r = 0
    while f"per_round_acc_{r}" in keys:
        r += 1
    if r == 0:
        raise RuntimeError("Could not infer rounds from metrics.csv header.")
    return r


def _extract_series(
    rows: List[Dict[str, str]], col: str
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        val = r.get(col, "")
        if val != "" and val is not None:
            xs.append(int(r["step"]))
            ys.append(float(val))
    return np.asarray(xs, dtype=np.int64), np.asarray(ys, dtype=np.float64)


def plot_training_curves(
    rows: List[Dict[str, str]], title_tag: str, out_path: Path, ver_tag: str = "V2"
) -> None:
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5), dpi=180)

    ax = axes[0]
    s_tr, v_tr = _extract_series(rows, "train_bce_loss")
    s_fin, v_fin = _extract_series(rows, "train_bce_final")
    s_ev, v_ev = _extract_series(rows, "eval_bce_loss")
    if len(s_tr) > 0:
        ax.plot(s_tr, v_tr, label="train_bce_loss (weighted)", color="#1f77b4", alpha=0.65)
    if len(s_fin) > 0:
        ax.plot(s_fin, v_fin, label="train_bce_final (step K)", color="#17becf", alpha=0.85)
    if len(s_ev) > 0:
        ax.plot(
            s_ev,
            v_ev,
            "o-",
            label="eval_bce_loss (zero-init 2048)",
            color="#d62728",
            linewidth=2.2,
            markersize=4.5,
        )
    ln2 = math.log(2.0)
    ax.axhline(
        ln2,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"blind baseline ln(2) = {ln2:.4f}",
    )
    ax.set_title(f"Free-Bit BCE Loss ({ver_tag} Direct x0 Logits)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("BCEWithLogitsLoss")
    ax.legend(loc="best", fontsize=8.0)

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
    ax.set_title("2-adic Physics Energy (All 32 Scales Active)")
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
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=7.8)
    else:
        ax.legend(loc="best", fontsize=8.0)

    ax = axes[2]
    s_pre, v_pre = _extract_series(rows, "exact_preimage_rate@256")
    s_uniq, v_uniq = _extract_series(rows, "unique_ratio@256")
    s_w0, v_w0 = _extract_series(rows, "w0_free_bit_acc")
    s_zw0, v_zw0 = _extract_series(rows, "zero_w0_bit_acc")

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
    if len(s_uniq) > 0:
        ax.plot(
            s_uniq,
            v_uniq,
            "s--",
            label="unique_ratio@256 (diversity)",
            color="#8c564b",
            linewidth=1.7,
            markersize=4,
        )
    if len(s_w0) > 0:
        ax.plot(
            s_w0,
            v_w0,
            "^-",
            label="w0_free_bit_acc (K=256)",
            color="#1f77b4",
            linewidth=1.8,
            markersize=4.5,
        )
    if len(s_zw0) > 0:
        ax.plot(
            s_zw0,
            v_zw0,
            "d:",
            label="zero_w0_bit_acc (determ.)",
            color="#17becf",
            linewidth=1.6,
            markersize=4,
        )
    ax.set_title("Preimage Rate, Diversity & W0 Accuracy")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Rate / Ratio / Accuracy")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="best", fontsize=7.8)

    fig.suptitle(f"Phase-SHA {ver_tag} ({title_tag}) Training & Generalization", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_per_round_wavefront_heatmap(
    rows: List[Dict[str, str]], rounds: int, title_tag: str, out_path: Path, ver_tag: str = "V2"
) -> None:
    eval_rows = [
        r for r in rows if r.get("per_round_acc_0", "") not in ("", None)
    ]
    if not eval_rows:
        raise RuntimeError("No per_round_acc rows found in metrics.csv.")

    steps = [int(r["step"]) for r in eval_rows]
    mat_anneal = np.zeros((rounds, len(eval_rows)), dtype=np.float64)
    mat_zero = np.zeros((rounds, len(eval_rows)), dtype=np.float64)

    for col_idx, r in enumerate(eval_rows):
        for t in range(rounds):
            mat_anneal[t, col_idx] = float(r[f"per_round_acc_{t}"])
            mat_zero[t, col_idx] = float(r[f"zero_round_acc_{t}"])

    fig_h = max(6.8, 0.44 * rounds + 1.5)
    fig, axes = plt.subplots(
        1, 2, figsize=(15.5, fig_h), dpi=180, sharey=True, constrained_layout=True
    )

    vmin, vmax = 0.45, 1.00
    cmap = "viridis"

    axes[0].imshow(
        mat_anneal,
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )
    axes[0].set_title("K=256 Unrolled Inference", fontsize=11.5)
    axes[0].set_xlabel("Training Step")
    free_end = rounds - 4
    if rounds > 16:
        ylabel = (
            f"Round t (0=W0, 0..{free_end}=free, {free_end+1}..15=clamped, "
            f"16..{rounds-1}=free W_t)"
        )
    else:
        ylabel = f"Round t (0=W0 end, 0..{free_end}=free, {free_end+1}..{rounds-1}=clamped)"
    axes[0].set_ylabel(ylabel)

    im1 = axes[1].imshow(
        mat_zero,
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
    )
    axes[1].set_title("Deterministic Zero-Init Inference (Holdout 2048)", fontsize=11.5)
    axes[1].set_xlabel("Training Step")

    for ax, mat in ((axes[0], mat_anneal), (axes[1], mat_zero)):
        ax.set_xticks(np.arange(len(steps)))
        ax.set_xticklabels([str(s) for s in steps], rotation=45, ha="right", fontsize=8.5)
        ax.set_yticks(np.arange(rounds))
        ax.set_yticklabels([str(t) for t in range(rounds)], fontsize=8.5)
        ax.axhline(rounds - 3.5, color="white", linestyle="--", linewidth=1.2, alpha=0.85)
        if rounds > 16:
            ax.axhline(15.5, color="cyan", linestyle=":", linewidth=1.3, alpha=0.9)
        for t in range(rounds):
            for c in range(len(steps)):
                val = mat[t, c]
                txt_color = "black" if val >= 0.78 else "white"
                ax.text(
                    c,
                    t,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontsize=6.5 if rounds <= 18 else 5.5,
                )

    cbar = fig.colorbar(im1, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("Per-Round Free-Bit Accuracy", fontsize=10.5)

    fig.suptitle(
        f"Phase-SHA {ver_tag} ({title_tag}): Per-Round Reasoning Wavefront",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_bit_zipper_heatmap(
    rows: List[Dict[str, str]], title_tag: str, out_path: Path, ver_tag: str = "V2"
) -> None:
    eval_rows = [
        r for r in rows if r.get("bit_acc_pos_0", "") not in ("", None)
    ]
    if not eval_rows:
        raise RuntimeError("No bit_acc_pos rows found in metrics.csv.")

    steps = [int(r["step"]) for r in eval_rows]
    mat = np.zeros((32, len(eval_rows)), dtype=np.float64)
    for col_idx, r in enumerate(eval_rows):
        for k in range(32):
            mat[k, col_idx] = float(r[f"bit_acc_pos_{k}"])

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 6.8), dpi=180, constrained_layout=True)
    im = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        vmin=0.45,
        vmax=1.00,
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_title(f"Phase-SHA {ver_tag} ({title_tag}): Bit-Zipper Accuracy (K=256)", fontsize=12)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Bit Position k (0 = LSB, 31 = MSB)")
    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(0, 32, 2))
    ax.set_yticklabels([str(k) for k in range(0, 32, 2)], fontsize=8.5)
    cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label("Free-Bit Accuracy on Unseen Digests", fontsize=10.5)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(ROOT_DIR / "runs" / "r12_v2"),
        help="Run directory containing metrics.csv",
    )
    parser.add_argument(
        "--version-tag",
        type=str,
        default=None,
        help="Explicit version tag override (e.g. V2 or V3).",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    rows = load_metrics_csv(run_dir / "metrics.csv")
    rounds = _infer_rounds(rows)
    title_tag = f"{run_dir.name}, R={rounds}"
    if args.version_tag is not None:
        ver_tag = args.version_tag
    elif "zero_manifold_align" in rows[0] or "v4" in run_dir.name.lower():
        if "v4a" in run_dir.name.lower():
            ver_tag = "V4A"
        elif "v4b" in run_dir.name.lower():
            ver_tag = "V4B"
        elif "v4c" in run_dir.name.lower():
            ver_tag = "V4C"
        else:
            ver_tag = "V4"
    elif "v3" in run_dir.name.lower() or "zero_latch_ratio" in rows[0]:
        ver_tag = "V3"
    else:
        ver_tag = "V2"

    plot_training_curves(
        rows, title_tag, run_dir / "training_curves.png", ver_tag=ver_tag
    )
    plot_per_round_wavefront_heatmap(
        rows,
        rounds=rounds,
        title_tag=title_tag,
        out_path=run_dir / "per_round_wavefront_heatmap.png",
        ver_tag=ver_tag,
    )
    plot_bit_zipper_heatmap(
        rows, title_tag, run_dir / "bit_zipper_heatmap.png", ver_tag=ver_tag
    )
    print(f"Saved {ver_tag} plots (R={rounds}) under {run_dir}", flush=True)


if __name__ == "__main__":
    main()
