"""Unified Training & Pure-Neural Evaluation Harness for V7 (V7A / V7B / V7C)."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v7 import (
    V7_PROD_MAX_PARAMS,
    V7_PROD_MIN_PARAMS,
    assert_strict_fp32,
    configure_v7_vram_cap,
    count_parameters,
    enforce_v7_fp32,
)
from v7.v7_arena import (
    V7CSVLogger,
    build_v7_holdout,
    evaluate_pure_neural_model,
    sample_online_sha_batch,
)
from v7.v7a_autoregressive import V7AConfig, V7AReverseTraceTransformer, train_v7a_step
from v7.v7b_looped_rl import V7BConfig, V7BLoopedRLInverter, grpo_step
from v7.v7c_levin_meta import (
    OfflineV6ACorpus,
    V7CConfig,
    V7CLevinMetaInverter,
    pretrain_v7c_stage1_step,
    train_v7c_stage2_step,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate V7 pure-neural SHA-256 inverters")
    p.add_argument("--model", type=str, required=True, choices=("v7a", "v7b", "v7c"))
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--max_vram_gb", type=float, default=None)
    p.add_argument("--v6a_corpus", type=str, default=None, help="Optional offline V6A corpus JSON (V7C Stage 2 only)")
    p.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume from")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_model_from_cfg(model_type: str, model_cfg: Dict[str, Any]) -> torch.nn.Module:
    if model_type == "v7a":
        return V7AReverseTraceTransformer(V7AConfig(**model_cfg))
    if model_type == "v7b":
        return V7BLoopedRLInverter(V7BConfig(**model_cfg))
    if model_type == "v7c":
        return V7CLevinMetaInverter(V7CConfig(**model_cfg))
    raise ValueError(f"Unsupported model type: {model_type}")


def _cosine_lr(step_in_phase: int, total_steps: int, base_lr: float, min_lr: float, warmup: int) -> float:
    if warmup > 0 and step_in_phase < warmup:
        return base_lr * float(step_in_phase + 1) / float(warmup)
    if total_steps <= warmup:
        return base_lr
    prog = float(step_in_phase - warmup) / float(max(1, total_steps - warmup))
    prog = min(1.0, max(0.0, prog))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * prog))


def _set_optimizer_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for pg in opt.param_groups:
        pg["lr"] = lr


def _peak_vram_gb(dev: torch.device) -> float:
    if dev.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(dev)) / (1024.0 ** 3)


def _write_summary(save_dir: Path, payload: Dict[str, Any]) -> None:
    tmp = save_dir / "v7_summary.json.tmp"
    out = save_dir / "v7_summary.json"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(out)


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    round_idx: int,
    r_c_neural: int,
    summary: Dict[str, Any],
) -> None:
    tmp = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "global_step": global_step,
            "round_idx": round_idx,
            "r_c_neural": r_c_neural,
            "summary": summary,
        },
        tmp,
    )
    tmp.replace(path)


def run_training(args: argparse.Namespace) -> None:
    enforce_v7_fp32()
    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 7000))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dev = torch.device(args.device)
    max_vram_gb = float(
        args.max_vram_gb if args.max_vram_gb is not None else cfg.get("max_vram_gb", 11.0)
    )
    if dev.type == "cuda":
        frac = configure_v7_vram_cap(max_gb=max_vram_gb, device=dev.index or 0)
        torch.cuda.reset_peak_memory_stats(dev)
    else:
        frac = 0.0

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    model_type = str(args.model).lower()
    model = build_model_from_cfg(model_type, cfg.get("model", {})).to(dev)
    assert_strict_fp32(model)
    n_params = count_parameters(model)
    print(
        f"[V7:{model_type.upper()}] params={n_params:,} ({n_params/1e6:.2f}M) | "
        f"device={dev} | vram_cap={max_vram_gb:.2f}GB (frac={frac:.3f}) | strict_fp32=True",
        flush=True,
    )
    if not (V7_PROD_MIN_PARAMS <= n_params <= V7_PROD_MAX_PARAMS):
        print(
            f"[V7 WARNING] params={n_params:,} outside production range "
            f"[{V7_PROD_MIN_PARAMS:,}, {V7_PROD_MAX_PARAMS:,}]",
            flush=True,
        )

    tr = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr["lr"]),
        weight_decay=float(tr.get("weight_decay", 0.0)),
    )

    logger = V7CSVLogger(save_dir / "metrics.csv")
    rounds_schedule: List[int] = [int(r) for r in tr.get("rounds_schedule", [2, 3, 4, 5, 6, 8])]
    n_eval = int(tr.get("n_eval", 256))
    pass_k = int(tr.get("pass_k", 64))
    eval_batch_size = int(tr.get("eval_batch_size", 32))
    advance_pass64 = float(tr.get("advance_pass64", 0.50))

    summary: Dict[str, Any] = {
        "model": model_type,
        "params": n_params,
        "strict_fp32": True,
        "allow_peel_back": False,
        "domain_mode": "free_msg",
        "max_vram_gb_cap": max_vram_gb,
        "r_c_neural": 0,
        "r_c_neural_50": 0,
        "current_round": rounds_schedule[0],
        "stage": "init",
        "rounds": {},
    }

    global_step = 0
    start_round_idx = 0
    if args.resume is not None and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        global_step = int(ckpt.get("global_step", 0))
        start_round_idx = int(ckpt.get("round_idx", 0))
        summary = ckpt.get("summary", summary)
        print(f"[V7:{model_type.upper()}] resumed from {args.resume} at step={global_step} round_idx={start_round_idx}", flush=True)

    train_gen = torch.Generator(device="cpu")
    train_gen.manual_seed(seed + 1337)

    # Pre-build isolated holdout digest sets BEFORE any training
    holdouts = {
        r: build_v7_holdout(rounds=r, n_eval=n_eval, device=dev) for r in rounds_schedule
    }

    # --- Optional V7C Stage 1 ARX Universal Pretraining ---
    st1 = cfg.get("stage1", {})
    if model_type == "v7c" and bool(st1.get("enabled", False)) and start_round_idx == 0 and global_step == 0:
        assert isinstance(model, V7CLevinMetaInverter)
        st1_steps = int(st1.get("steps", 1600))
        st1_tiers: List[int] = [int(t) for t in st1.get("tiers", [0, 1, 2, 3])]
        st1_bs = int(st1.get("batch_size", 48))
        st1_lr = float(st1.get("lr", 3.0e-4))
        st1_min_lr = float(st1.get("min_lr", 5.0e-5))
        st1_max_sec = float(st1.get("max_seconds", 360.0))
        st1_n_in = int(st1.get("n_inputs", 2))
        st1_plen = int(st1.get("prog_len", 48))
        st1_log_every = int(st1.get("log_every", 40))

        summary["stage"] = "stage1_arx_pretrain"
        _write_summary(save_dir, summary)
        t_st1 = time.monotonic()
        print(f"[V7C Stage 1] Universal ARX curriculum steps={st1_steps} tiers={st1_tiers}", flush=True)

        for s in range(st1_steps):
            if (time.monotonic() - t_st1) > st1_max_sec:
                print(f"[V7C Stage 1] time budget {st1_max_sec}s reached at step {s}", flush=True)
                break
            cur_lr = _cosine_lr(s, st1_steps, st1_lr, st1_min_lr, warmup=min(100, st1_steps // 10))
            _set_optimizer_lr(optimizer, cur_lr)
            tier_idx = min(len(st1_tiers) - 1, (s * len(st1_tiers)) // max(1, st1_steps))
            tier = st1_tiers[tier_idx]

            t0 = time.monotonic()
            stats = pretrain_v7c_stage1_step(
                model=model,
                optimizer=optimizer,
                batch_size=st1_bs,
                tier=tier,
                n_inputs=st1_n_in,
                prog_len=st1_plen,
                device=dev,
                generator=train_gen,
                grad_clip=float(st1.get("grad_clip", 1.0)),
            )
            dt_ms = (time.monotonic() - t0) * 1000.0
            global_step += 1

            if (s + 1) % st1_log_every == 0 or (s + 1) == st1_steps:
                vram = _peak_vram_gb(dev)
                logger.log(
                    {
                        "step": global_step,
                        "model": model_type,
                        "stage": f"stage1_tier{tier}",
                        "round_r": 0,
                        "r_c_neural": summary["r_c_neural"],
                        "train_loss": stats["train_loss"],
                        "joint_nll": stats["joint_nll_total"],
                        "lr": cur_lr,
                        "peak_vram_gb": vram,
                        "step_time_ms": dt_ms,
                    }
                )
                print(
                    f"[V7C Stage1] step={global_step} tier={tier} loss={stats['train_loss']:.4f} "
                    f"tok_acc={stats['tok_acc']:.3f} vram={vram:.2f}GB",
                    flush=True,
                )

    v6a_corpus: Optional[OfflineV6ACorpus] = None
    if model_type == "v7c" and args.v6a_corpus:
        v6a_corpus = OfflineV6ACorpus.load(args.v6a_corpus)
        print(
            f"[V7C Stage 2] loaded offline V6A corpus rounds={ {k: int(v.shape[0]) for k, v in v6a_corpus.by_round.items()} }",
            flush=True,
        )

    # --- Progressive Round Curriculum Race: R = 2 -> 3 -> 4 -> 5 -> 6 -> 8 ---
    steps_per_round = int(tr.get("steps_per_round", 3500))
    max_sec_per_round = float(tr.get("max_sec_per_round", 540.0))
    eval_every = int(tr.get("eval_every", 400))
    log_every = int(tr.get("log_every", 25))
    bs = int(tr.get("batch_size", 32))

    def _make_sample_fn():
        if model_type == "v7a":
            assert isinstance(model, V7AReverseTraceTransformer)
            temp = float(tr.get("sample_temperature", 0.75))
            top_p = float(tr.get("sample_top_p", 0.95))

            def fn(d_chunk: torch.Tensor, r_val: int, k_val: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
                return model.sample_preimages(
                    d_chunk, rounds=r_val, num_samples=k_val, temperature=temp, top_p=top_p
                )

            return fn
        if model_type == "v7b":
            assert isinstance(model, V7BLoopedRLInverter)
            eval_temp = float(tr.get("eval_temperature", 0.85))

            def fn(d_chunk: torch.Tensor, r_val: int, k_val: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
                return model.sample_preimages(
                    d_chunk, rounds=r_val, num_samples=k_val, temperature=eval_temp
                )

            return fn
        assert isinstance(model, V7CLevinMetaInverter)
        temp = float(tr.get("sample_temperature", 0.75))
        top_p = float(tr.get("sample_top_p", 0.95))

        def fn(d_chunk: torch.Tensor, r_val: int, k_val: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            return model.sample_preimages(
                d_chunk, rounds=r_val, num_samples=k_val, temperature=temp, top_p=top_p
            )

        return fn

    sample_fn = _make_sample_fn()

    for r_idx in range(start_round_idx, len(rounds_schedule)):
        cur_r = rounds_schedule[r_idx]
        summary["current_round"] = cur_r
        summary["stage"] = f"round_{cur_r}"
        _write_summary(save_dir, summary)

        holdout_r = holdouts[cur_r]
        t_round = time.monotonic()
        best_pass64_r = -1.0
        best_metrics_r: Dict[str, Any] = {}

        print(f"\n=== [V7:{model_type.upper()}] Entering Round R={cur_r} (budget={steps_per_round} steps / {max_sec_per_round:.0f}s) ===", flush=True)

        for local_step in range(steps_per_round):
            if (time.monotonic() - t_round) > max_sec_per_round:
                print(f"[V7:{model_type.upper()}] R={cur_r} wall-time budget {max_sec_per_round:.0f}s hit at local_step={local_step}", flush=True)
                break

            cur_lr = _cosine_lr(
                local_step,
                steps_per_round,
                float(tr["lr"]),
                float(tr.get("min_lr", 2e-5)),
                warmup=int(tr.get("warmup_steps", 100)),
            )
            _set_optimizer_lr(optimizer, cur_lr)

            t0 = time.monotonic()
            if model_type == "v7a":
                assert isinstance(model, V7AReverseTraceTransformer)
                stats = train_v7a_step(
                    model=model,
                    optimizer=optimizer,
                    batch_size=bs,
                    rounds=cur_r,
                    device=dev,
                    generator=train_gen,
                    grad_clip=float(tr.get("grad_clip", 1.0)),
                )
            elif model_type == "v7b":
                assert isinstance(model, V7BLoopedRLInverter)
                # V7B consumes ONLY digests (ZERO W* labels!)
                _, _, dig_only = sample_online_sha_batch(
                    batch_size=bs, rounds=cur_r, device=dev, generator=train_gen
                )
                stats = grpo_step(
                    model=model,
                    optimizer=optimizer,
                    digest=dig_only,
                    rounds=cur_r,
                    group_size=int(tr.get("group_size", 32)),
                    temperature=float(tr.get("rl_temperature", 1.0)),
                    entropy_coef=float(tr.get("entropy_coef", 0.015)),
                    gfn_coef=float(tr.get("gfn_coef", 0.1)),
                    gfn_beta=float(tr.get("gfn_beta", 2.0)),
                    grad_clip=float(tr.get("grad_clip", 1.0)),
                )
            else:
                assert isinstance(model, V7CLevinMetaInverter)
                mix_rounds = set(int(x) for x in tr.get("v6a_mix_rounds", [4, 5, 6]))
                mix_prob = float(tr.get("v6a_mix_prob", 0.25)) if cur_r in mix_rounds else 0.0
                stats = train_v7c_stage2_step(
                    model=model,
                    optimizer=optimizer,
                    batch_size=bs,
                    rounds=cur_r,
                    device=dev,
                    generator=train_gen,
                    v6a_corpus=v6a_corpus,
                    v6a_mix_prob=mix_prob,
                    grad_clip=float(tr.get("grad_clip", 1.0)),
                )

            dt_ms = (time.monotonic() - t0) * 1000.0
            global_step += 1

            is_last_step = (local_step + 1) == steps_per_round
            if (local_step + 1) % log_every == 0:
                vram = _peak_vram_gb(dev)
                logger.log(
                    {
                        "step": global_step,
                        "model": model_type,
                        "stage": f"round_{cur_r}_train",
                        "round_r": cur_r,
                        "r_c_neural": summary["r_c_neural"],
                        "train_loss": stats["train_loss"],
                        "joint_nll": stats.get("joint_nll_total", ""),
                        "mean_reward": stats.get("mean_reward", ""),
                        "lr": cur_lr,
                        "peak_vram_gb": vram,
                        "step_time_ms": dt_ms,
                    }
                )
                extra = (
                    f"nll={stats['joint_nll_total']:.2f} w_acc={stats.get('w_nibble_acc', 0):.3f}"
                    if "joint_nll_total" in stats
                    else f"rew={stats.get('mean_reward', 0):.3f} lcp={stats.get('group_2adic_lcp', 0):.3f}"
                )
                print(
                    f"[V7:{model_type.upper()}] R={cur_r} step={global_step} ({local_step+1}/{steps_per_round}) "
                    f"loss={stats['train_loss']:.4f} {extra} vram={vram:.2f}GB {dt_ms:.1f}ms",
                    flush=True,
                )

            if (local_step + 1) % eval_every == 0 or is_last_step:
                ev = evaluate_pure_neural_model(
                    sample_fn=sample_fn,
                    holdout=holdout_r,
                    pass_k=pass_k,
                    eval_batch_size=eval_batch_size,
                )
                vram = _peak_vram_gb(dev)
                if ev["exact_hits"] > 0 and cur_r > int(summary["r_c_neural"]):
                    summary["r_c_neural"] = cur_r
                if ev["exact_pass_64"] >= advance_pass64 and cur_r > int(summary["r_c_neural_50"]):
                    summary["r_c_neural_50"] = cur_r

                if ev["exact_pass_64"] > best_pass64_r:
                    best_pass64_r = ev["exact_pass_64"]
                    best_metrics_r = dict(ev)
                    summary["rounds"][str(cur_r)] = best_metrics_r
                    _save_checkpoint(
                        save_dir / f"best_r{cur_r}.pt",
                        model,
                        optimizer,
                        global_step,
                        r_idx,
                        int(summary["r_c_neural"]),
                        summary,
                    )

                logger.log(
                    {
                        "step": global_step,
                        "model": model_type,
                        "stage": f"round_{cur_r}_eval",
                        "round_r": cur_r,
                        "r_c_neural": summary["r_c_neural"],
                        "train_loss": stats["train_loss"],
                        "joint_nll": ev.get("joint_nll", stats.get("joint_nll_total", "")),
                        "mean_reward": stats.get("mean_reward", ""),
                        "lr": cur_lr,
                        "peak_vram_gb": vram,
                        "exact_pass_1": ev["exact_pass_1"],
                        "exact_pass_64": ev["exact_pass_64"],
                        "exact_hits": int(ev["exact_hits"]),
                        "eval_n_digests": int(ev["eval_n_digests"]),
                        "hash_bit_acc": ev["hash_bit_acc"],
                        "hash_bit_acc_best64": ev["hash_bit_acc_best64"],
                        "w0_bit_acc": ev["w0_bit_acc"],
                        "w0_zscore": ev["w0_zscore"],
                    }
                )
                _write_summary(save_dir, summary)
                _save_checkpoint(
                    save_dir / "latest.pt",
                    model,
                    optimizer,
                    global_step,
                    r_idx,
                    int(summary["r_c_neural"]),
                    summary,
                )
                print(
                    f"  [EVAL R={cur_r}] pass@1={ev['exact_pass_1']*100:.2f}% "
                    f"pass@64={ev['exact_pass_64']*100:.2f}% ({int(ev['exact_hits'])}/{int(ev['eval_n_digests'])}) "
                    f"hash_bit={ev['hash_bit_acc']:.4f} best64={ev['hash_bit_acc_best64']:.4f} "
                    f"w0_z={ev['w0_zscore']:.2f} R_c={summary['r_c_neural']}",
                    flush=True,
                )
                if ev["exact_pass_64"] >= advance_pass64:
                    print(
                        f"  [ADVANCE] R={cur_r} reached pass@64={ev['exact_pass_64']*100:.1f}% >= {advance_pass64*100:.1f}% -> advancing!",
                        flush=True,
                    )
                    break

        # Final end-of-round eval if stopped by wall-clock before an eval_every boundary
        if str(cur_r) not in summary["rounds"]:
            ev = evaluate_pure_neural_model(
                sample_fn=sample_fn,
                holdout=holdout_r,
                pass_k=pass_k,
                eval_batch_size=eval_batch_size,
            )
            if ev["exact_hits"] > 0 and cur_r > int(summary["r_c_neural"]):
                summary["r_c_neural"] = cur_r
            if ev["exact_pass_64"] >= advance_pass64 and cur_r > int(summary["r_c_neural_50"]):
                summary["r_c_neural_50"] = cur_r
            summary["rounds"][str(cur_r)] = ev
            _write_summary(save_dir, summary)
            _save_checkpoint(
                save_dir / f"best_r{cur_r}.pt",
                model,
                optimizer,
                global_step,
                r_idx,
                int(summary["r_c_neural"]),
                summary,
            )

    summary["stage"] = "completed"
    _write_summary(save_dir, summary)
    print(f"\n=== [V7:{model_type.upper()} DONE] R_c_neural={summary['r_c_neural']} | summary={save_dir / 'v7_summary.json'} ===", flush=True)


if __name__ == "__main__":
    run_training(parse_args())
