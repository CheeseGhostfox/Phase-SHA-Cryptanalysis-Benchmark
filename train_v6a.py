"""Unified CLI Entry Point for Phase-SHA V6A-v2 (Ouroboros Full-Stack Inverse Scrambling Engine)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from src.sha256_ref import configure_cuda_memory
from v6a import enforce_v6a_fp32
from v6a.v6a_engine import OuroborosV6AEngine, run_frontier_push, run_phase0_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-SHA V6A-v2: Ouroboros (2D Homotopy x GL-RRR x Bottleneck-MPS x BM-Escape x Zero-Brake Nav)"
    )
    parser.add_argument("--mode", type=str, default="attack", choices=["calibrate", "frontier_push", "attack"])
    parser.add_argument("--rounds", type=int, default=18)
    parser.add_argument("--gpu_tier", "--gpu-tier", dest="gpu_tier", type=str, default="40g", choices=["40g", "80g", "local"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--save_dir", "--output-dir", dest="save_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num_digests", "--num-digests", dest="num_digests", type=int, default=None)
    parser.add_argument("--trajs_per_digest", "--trajs-per-digest", dest="trajs_per_digest", type=int, default=None)
    parser.add_argument("--start_mask_m", "--start-mask-m", dest="start_mask_m", type=int, default=None)
    parser.add_argument("--domain_mode", "--domain-mode", dest="domain_mode", type=str, default=None)
    parser.add_argument("--cal_max_sec", "--cal-max-sec", dest="cal_max_sec", type=float, default=120.0)
    parser.add_argument("--frontier_rounds", "--frontier-rounds", dest="frontier_rounds", type=str, default="6,7,8,9,10")
    parser.add_argument("--frontier_max_ticks", "--frontier-max-ticks", dest="frontier_max_ticks", type=int, default=1500)
    parser.add_argument("--frontier_max_sec", "--frontier-max-sec", dest="frontier_max_sec", type=float, default=300.0)
    parser.add_argument("--r20-stress", "--r20_stress", dest="r20_stress", action="store_true")
    parser.add_argument("--log_every", "--log-every", dest="log_every", type=int, default=None)
    parser.add_argument("--eval_every", "--eval-every", dest="eval_every", type=int, default=None)
    parser.add_argument("--checkpoint_every", "--checkpoint-every", dest="checkpoint_every", type=int, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    if args.config is not None:
        with Path(args.config).open("r", encoding="utf-8") as f:
            cfg: Dict[str, Any] = yaml.safe_load(f)
    else:
        tier_suffix = "80g" if args.gpu_tier == "80g" else "40g"
        default_p = Path("configs") / f"v6a_r18_colab_{tier_suffix}.yaml"
        if default_p.exists():
            with default_p.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {}

    cfg["rounds"] = int(args.rounds)
    cfg["gpu_tier"] = args.gpu_tier
    if args.gpu_tier == "local" and args.config is None:
        cfg["num_digests"] = 4
        cfg["trajs_per_digest"] = 16
        cfg["inner_steps_per_tick"] = 4
        cfg["osd_every"] = 16
        cfg["osd_top_k"] = 4
    if args.steps is not None:
        cfg["steps"] = int(args.steps)
    if args.num_digests is not None:
        cfg["num_digests"] = int(args.num_digests)
    if args.trajs_per_digest is not None:
        cfg["trajs_per_digest"] = int(args.trajs_per_digest)
    if args.start_mask_m is not None:
        cfg["start_mask_m"] = int(args.start_mask_m)
    if args.domain_mode is not None:
        cfg["domain_mode"] = str(args.domain_mode)
    if args.r20_stress:
        cfg["r20_stress"] = True
        cfg["rounds"] = 20
    if args.log_every is not None:
        cfg["log_every"] = int(args.log_every)
    if args.eval_every is not None:
        cfg["eval_every"] = int(args.eval_every)
    if args.checkpoint_every is not None:
        cfg["checkpoint_every"] = int(args.checkpoint_every)
    return cfg


def main() -> None:
    enforce_v6a_fp32()
    args = parse_args()
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        configure_cuda_memory(
            0.90 if args.gpu_tier != "local" else 0.85,
            device=0 if device.index is None else int(device.index),
        )

    if args.mode == "calibrate":
        out_dir = Path(args.save_dir) if args.save_dir is not None else Path("runs") / "v6a_calibrate_colab"
        report = run_phase0_calibration(
            output_dir=out_dir,
            rounds_list=(4, 5, 6, 8, 10, 12),
            max_sec_per_round=float(args.cal_max_sec),
            gpu_tier=args.gpu_tier,
            device=device,
        )
        print(f"[Phase-SHA V6A Calibrate] Critical R_c={report['critical_round_R_c']} saved to {out_dir}", flush=True)
        return

    if args.mode == "frontier_push":
        cfg: Dict[str, Any] = {}
        if args.config is not None and Path(args.config).exists():
            with Path(args.config).open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        if args.trajs_per_digest is not None:
            cfg["num_trajs"] = int(args.trajs_per_digest)
        if args.start_mask_m is not None:
            cfg["start_mask_m"] = int(args.start_mask_m)
        if args.frontier_max_ticks != 1500 or "max_ticks" not in cfg:
            cfg["max_ticks"] = int(args.frontier_max_ticks)
        if args.frontier_max_sec != 300.0 or "max_sec_per_round" not in cfg:
            cfg["max_sec_per_round"] = float(args.frontier_max_sec)
        r_tuple = tuple(int(x.strip()) for x in str(args.frontier_rounds).split(",") if x.strip())
        if "rounds_list" in cfg and args.frontier_rounds == "6,7,8,9,10":
            r_tuple = tuple(int(x) for x in cfg["rounds_list"])
        out_dir = Path(args.save_dir) if args.save_dir is not None else Path("runs") / "v6a_frontier_push"
        report = run_frontier_push(output_dir=out_dir, config=cfg, rounds_list=r_tuple, device=device)
        print(f"[Phase-SHA V6A-v2 Frontier Push] R_c={report['critical_round_R_c']} saved to {out_dir}", flush=True)
        return

    cfg = load_config(args)
    out_dir = (
        Path(args.save_dir)
        if args.save_dir is not None
        else Path("runs") / str(cfg.get("experiment_name", f"v6a_r{cfg['rounds']}_{args.gpu_tier}"))
    )
    engine = OuroborosV6AEngine(config=cfg, output_dir=out_dir, device=device)
    if args.resume is not None:
        engine.load_checkpoint(args.resume)
        print(f"[Phase-SHA V6A-v2] Resumed from {args.resume} at step={engine.step}", flush=True)
    engine.train()


if __name__ == "__main__":
    main()
