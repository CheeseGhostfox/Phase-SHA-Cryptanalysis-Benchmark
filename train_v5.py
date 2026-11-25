"""Unified CLI entry point for Phase-SHA V5 (v5d_pure, v5d_neural, v5c, v5b, v5a)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from src.sha256_ref import configure_cuda_memory
from src.v5_interfaces import build_r20_stress_holdout_w16, enforce_v5_fp32
from src.v5_trainer import UnifiedTrainerV5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-SHA V5: Single-Variable Orthogonal Solver Suite (V5D / V5C / V5B / V5A)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["v5d_pure", "v5d_neural", "v5c", "v5b", "v5a"],
        help="V5 ablation card variant.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (defaults to configs/<variant>_r18_colab.yaml).",
    )
    parser.add_argument("--save_dir", "--output-dir", dest="save_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--init-from", "--init_from", dest="init_from", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--num-particles", "--num_particles", dest="num_particles", type=int, default=None)
    parser.add_argument("--bond_dim", "--bond-dim", dest="bond_dim", type=int, default=None)
    parser.add_argument("--manifold_dim", "--manifold-dim", dest="manifold_dim", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-every", "--eval_every", dest="eval_every", type=int, default=None)
    parser.add_argument("--checkpoint-every", "--checkpoint_every", dest="checkpoint_every", type=int, default=None)
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--r20-stress", action="store_true", help="Evaluate against unified R=20 stress set.")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg_path = (
        Path(args.config)
        if args.config is not None
        else Path("configs") / f"{args.variant}_r18_colab.yaml"
    )
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    cfg["variant"] = str(args.variant).lower()
    if args.rounds is not None:
        cfg["rounds"] = int(args.rounds)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.num_particles is not None:
        cfg["num_particles"] = int(args.num_particles)
    if args.bond_dim is not None:
        cfg["bond_dim"] = int(args.bond_dim)
    if args.manifold_dim is not None:
        cfg["manifold_dim"] = int(args.manifold_dim)
    if args.steps is not None:
        cfg["steps"] = int(args.steps)
    if args.eval_every is not None:
        cfg["eval_every"] = int(args.eval_every)
    if args.checkpoint_every is not None:
        cfg["checkpoint_every"] = int(args.checkpoint_every)
    if args.log_every is not None:
        cfg["log_every"] = int(args.log_every)
    if args.no_compile:
        cfg["use_torch_compile"] = "false"
    return cfg


def main() -> None:
    enforce_v5_fp32()
    args = parse_args()
    cfg = load_config(args)

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        frac = float(cfg.get("cuda_memory_fraction", 0.90))
        dev_idx = 0 if device.index is None else int(device.index)
        configure_cuda_memory(frac, device=dev_idx)

    out_dir = (
        Path(args.save_dir)
        if args.save_dir is not None
        else Path("runs") / str(cfg.get("experiment_name", f"{args.variant}_r{cfg.get('rounds', 18)}"))
    )

    trainer = UnifiedTrainerV5(config=cfg, output_dir=out_dir, device=device)
    if args.resume is not None and args.init_from is not None:
        raise ValueError("Cannot pass both --resume and --init-from simultaneously.")
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
        print(
            f"[Phase-SHA {args.variant.upper()}] Resumed from {args.resume} at step={trainer.step}",
            flush=True,
        )
    elif args.init_from is not None:
        trainer.load_weights_only(args.init_from)

    if args.r20_stress:
        r20_w16, _ = build_r20_stress_holdout_w16(
            holdout_size=int(cfg.get("holdout_size", 2048)),
            holdout_seed=int(cfg.get("holdout_seed", 20261122)),
            domain_mode=trainer.domain_mode,
            num_bits=trainer.num_bits,
            msg_bytes_len=trainer.msg_bytes_len,
            device="cpu",
        )
        metrics = trainer.evaluate_holdout(holdout_w16_override=r20_w16)
        print(f"[R=20 Stress {args.variant.upper()}] {metrics}", flush=True)
        return

    if args.eval_only:
        metrics = trainer.evaluate_holdout()
        print(f"[Eval-Only {args.variant.upper()}] {metrics}", flush=True)
        return

    trainer.train()


if __name__ == "__main__":
    main()
