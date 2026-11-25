"""Merge partial P6 sweep summaries into final p6_sweep_summary.json and p6_run_meta.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def merge_parts(part_paths: List[Path], out_dir: Path, colab_stopped: bool | None = None) -> None:
    parts = [json.load(open(p, "r", encoding="utf-8")) for p in part_paths]
    base_meta = dict(parts[0]["meta"])
    total_wall = sum(float(p["meta"]["total_wall_sec"]) for p in parts)

    merged: Dict[str, object] = {
        "meta": {
            "mode": "full",
            "gpu_name": ", ".join(sorted(set(str(p["meta"]["gpu_name"]) for p in parts))),
            "fp32_locked": all(bool(p["meta"]["fp32_locked"]) for p in parts),
            "tf32_disabled": all(bool(p["meta"]["tf32_disabled"]) for p in parts),
            "rounds": [4, 6, 8, 10, 12],
            "weight_decays": base_meta["weight_decays"],
            "variants": base_meta["variants"],
            "ctrl_configs": base_meta["ctrl_configs"],
            "steps_base": base_meta["steps_base"],
            "max_steps_auto_extend": base_meta["max_steps_auto_extend"],
            "eval_every": base_meta["eval_every"],
            "batch_size": base_meta["batch_size"],
            "sae_steps": base_meta["sae_steps"],
            "total_wall_sec": total_wall,
            "sessions_merged": len(parts),
        },
        "track_a": {},
        "track_b": {},
        "comparisons": {"track_a": {}, "track_b": {}},
        "zero_param_baselines": {"track_a": {}, "track_b": {}},
        "dissections": [],
    }

    for p in parts:
        for trk in ("track_a", "track_b"):
            merged[trk].update(p[trk])
            merged["comparisons"][trk].update(p["comparisons"][trk])
            merged["zero_param_baselines"][trk].update(p["zero_param_baselines"][trk])
        merged["dissections"].extend(p["dissections"])

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "p6_sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    run_meta = {
        "command": "merged from partial Colab runs: " + ", ".join(p.name for p in part_paths),
        "mode": "full",
        "gpu_name": merged["meta"]["gpu_name"],
        "fp32_locked": merged["meta"]["fp32_locked"],
        "tf32_disabled": merged["meta"]["tf32_disabled"],
        "steps_base": merged["meta"]["steps_base"],
        "seeds": {
            "SEED_SPLIT_A": 20261106,
            "SEED_B_TRAIN": 20261116,
            "SEED_B_HOLD": 998244353,
            "SEED_B_RETEST": 777666555,
            "SEED_CTRL_1": 1000003,
            "SEED_CTRL_2": 2000003,
        },
        "part_wall_secs": [float(p["meta"]["total_wall_sec"]) for p in parts],
        "total_wall_sec": total_wall,
        "colab_stopped": colab_stopped,
    }
    with open(out_dir / "p6_run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    print(
        f"Merged {len(parts)} parts: track_a_rounds={sorted(merged['track_a'].keys())}, "
        f"track_b_rounds={sorted(merged['track_b'].keys())}, "
        f"dissections={len(merged['dissections'])}, total_wall={total_wall:.1f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+", required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--colab-stopped", action="store_true")
    args = parser.parse_args()
    merge_parts([Path(p) for p in args.parts], Path(args.out_dir), colab_stopped=args.colab_stopped)


if __name__ == "__main__":
    main()
