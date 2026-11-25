"""Unified CLI entry point for Phase-SHA V4 (V4A / V4B / V4C) strict-FP32 training & eval."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from src.sha256_ref import configure_cuda_memory, digest_words_to_bytes, sha256_digest
from src.v4_manifolds import enforce_fp32_precision
from src.v4_trainer import UnrolledTrainerV4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-SHA V4: High-Dimensional Manifold SHA-256 Solver (V4A / V4B / V4C)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["v4a", "v4b", "v4c"],
        help="Manifold variant: v4a (S^1), v4b (S^2 Bloch), v4c (S^3 Quat).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (defaults to configs/<variant>_r18.yaml).",
    )
    parser.add_argument(
        "--save_dir",
        "--output-dir",
        dest="save_dir",
        type=str,
        default=None,
        help="Directory for checkpoints and CSV metrics (supports Google Drive paths).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (.pt) to resume training losslessly.",
    )
    parser.add_argument(
        "--init-from",
        "--init_from",
        dest="init_from",
        type=str,
        default=None,
        help="Path to V3 or V4 checkpoint (.pt) for weight-only warm-start.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device override.")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--num-particles", "--num_particles", dest="num_particles", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-every", "--eval_every", dest="eval_every", type=int, default=None)
    parser.add_argument(
        "--checkpoint-every",
        "--checkpoint_every",
        dest="checkpoint_every",
        type=int,
        default=None,
    )
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=None)
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile even on Linux.")
    parser.add_argument("--eval-only", action="store_true", help="Run holdout evaluation only.")
    parser.add_argument("--solve-hex", type=str, default=None, help="64-hex-char target digest.")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg_path = (
        Path(args.config)
        if args.config is not None
        else Path("configs") / f"{args.variant}_r18.yaml"
    )
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    cfg["variant"] = str(args.variant).lower()
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.num_particles is not None:
        cfg["num_particles"] = args.num_particles
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.eval_every is not None:
        cfg["eval_every"] = args.eval_every
    if args.checkpoint_every is not None:
        cfg["checkpoint_every"] = args.checkpoint_every
    if args.log_every is not None:
        cfg["log_every"] = args.log_every
    if args.no_compile:
        cfg["use_torch_compile"] = "false"
    return cfg


def hex_to_digest_tensor(hex_str: str, device: torch.device) -> torch.Tensor:
    raw = bytes.fromhex(hex_str.strip().lower())
    if len(raw) != 32:
        raise ValueError(f"Expected 32-byte digest, got {len(raw)}")
    words = [int.from_bytes(raw[4 * i : 4 * (i + 1)], "big") for i in range(8)]
    return torch.tensor(words, dtype=torch.int64, device=device).unsqueeze(0)


def main() -> None:
    enforce_fp32_precision()
    args = parse_args()
    cfg = load_config(args)

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        frac = float(cfg.get("cuda_memory_fraction", 0.80))
        dev_idx = 0 if device.index is None else int(device.index)
        configure_cuda_memory(frac, device=dev_idx)

    out_dir = (
        Path(args.save_dir)
        if args.save_dir is not None
        else Path("runs") / str(cfg.get("experiment_name", f"{args.variant}_r18"))
    )

    trainer = UnrolledTrainerV4(config=cfg, output_dir=out_dir, device=device)
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

    if args.solve_hex is not None:
        td = hex_to_digest_tensor(args.solve_hex, device)
        res = trainer.solver.solve(
            td,
            domain_mode=trainer.domain_mode,
            num_bits=trainer.num_bits,
            msg_bytes_len=trainer.msg_bytes_len,
        )
        best_w16 = res.best_w16[0]
        verify_dig = sha256_digest(best_w16.unsqueeze(0), rounds=trainer.rounds)[0]
        print(
            f"[Solve {args.variant.upper()}] solved={bool(res.solved[0].item())} "
            f"hits={int(res.num_hits[0].item())} "
            f"best_energy={float(res.best_energy[0].item()):.4f} "
            f"W0=0x{int(best_w16[0].item()):08x} "
            f"digest={digest_words_to_bytes(verify_dig).hex()}",
            flush=True,
        )
        return

    if args.eval_only:
        metrics = trainer.evaluate_holdout(run_anneal=True)
        print(f"[Eval-Only {args.variant.upper()}] {metrics}", flush=True)
        return

    trainer.train()


if __name__ == "__main__":
    main()
