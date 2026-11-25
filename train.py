"""Unified CLI entry point for Phase-SHA training, evaluation, and preimage solving."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from src.flow_trainer import FlowTrainer
from src.sha256_ref import configure_cuda_memory, digest_words_to_bytes, sha256_digest
from src.v2_trainer import UnrolledTrainerV2
from src.v3_trainer import UnrolledTrainerV3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-SHA: 2-adic Differentiable SHA-256 Solver")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/r12_sanity.yaml",
        help="Path to YAML experiment config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for checkpoints and CSV telemetry (default: runs/<experiment_name>).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (.pt) to resume training or run evaluation/solving.",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Path to checkpoint (.pt) for curriculum weight-only warm start.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g. 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--steps", type=int, default=None, help="Override total training steps.")
    parser.add_argument("--eval-every", type=int, default=None, help="Override eval frequency.")
    parser.add_argument(
        "--checkpoint-every", type=int, default=None, help="Override checkpoint frequency."
    )
    parser.add_argument("--log-every", type=int, default=None, help="Override log frequency.")
    parser.add_argument(
        "--share-layers",
        action="store_true",
        help="Share weights across transformer blocks.",
    )
    parser.add_argument(
        "--grad-checkpointing",
        action="store_true",
        help="Enable activation gradient checkpointing.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate holdout set.",
    )
    parser.add_argument(
        "--solve-hex",
        type=str,
        default=None,
        help="64-hex-char target digest to solve via frequency-homotopy annealing.",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    with open(args.config, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.eval_every is not None:
        cfg["eval_every"] = args.eval_every
    if args.checkpoint_every is not None:
        cfg["checkpoint_every"] = args.checkpoint_every
    if args.log_every is not None:
        cfg["log_every"] = args.log_every
    if args.share_layers:
        cfg["share_layers"] = True
        cfg["share_blocks"] = True
    if args.grad_checkpointing:
        cfg["grad_checkpointing"] = True
    return cfg


def hex_to_digest_tensor(hex_str: str, device: torch.device) -> torch.Tensor:
    cleaned = hex_str.strip().lower()
    raw = bytes.fromhex(cleaned)
    if len(raw) != 32:
        raise ValueError(f"Expected 32-byte (64 hex chars) digest, got {len(raw)} bytes")
    words = [int.from_bytes(raw[4 * i : 4 * (i + 1)], "big") for i in range(8)]
    return torch.tensor(words, dtype=torch.int64, device=device).unsqueeze(0)


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        frac = float(cfg.get("cuda_memory_fraction", 0.80))
        dev_idx = 0 if device.index is None else int(device.index)
        configure_cuda_memory(frac, device=dev_idx)

    out_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("runs") / str(cfg.get("experiment_name", "phase_sha"))
    )

    ver = str(cfg.get("version", "v1")).lower()
    if ver == "v3":
        trainer_cls = UnrolledTrainerV3
    elif ver == "v2":
        trainer_cls = UnrolledTrainerV2
    else:
        trainer_cls = FlowTrainer
    trainer = trainer_cls(config=cfg, output_dir=out_dir, device=device)
    if args.resume is not None and args.init_from is not None:
        raise ValueError("Cannot pass both --resume and --init-from simultaneously.")
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
        print(f"[Phase-SHA] Resumed from {args.resume} at step={trainer.step}", flush=True)
    elif args.init_from is not None:
        if not hasattr(trainer, "load_weights_only"):
            raise ValueError("--init-from is only supported for V2 trainers.")
        trainer.load_weights_only(args.init_from)

    if args.solve_hex is not None:
        target_digest = hex_to_digest_tensor(args.solve_hex, device)
        res = trainer.solver.solve(
            target_digest,
            domain_mode=trainer.domain_mode,
            num_bits=trainer.num_bits,
            msg_bytes_len=trainer.msg_bytes_len,
        )
        best_w16 = res.best_w16[0]
        verify_digest = sha256_digest(best_w16.unsqueeze(0), rounds=trainer.rounds)[0]
        print(
            f"[Solve] solved={bool(res.solved[0].item())} hits={int(res.num_hits[0].item())} "
            f"best_energy={float(res.best_energy[0].item()):.4f} "
            f"W0=0x{int(best_w16[0].item()):08x} "
            f"digest={digest_words_to_bytes(verify_digest).hex()}",
            flush=True,
        )
        return

    if args.eval_only:
        metrics = trainer.evaluate_holdout(run_anneal=True)
        print(f"[Eval-Only] {metrics}", flush=True)
        return

    trainer.train()


if __name__ == "__main__":
    main()
