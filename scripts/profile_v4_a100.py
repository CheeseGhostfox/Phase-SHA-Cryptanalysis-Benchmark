"""10-step CUDA-Event stage profiler, TORCH_LOGS graph-break auditor, and eval timer for Phase-SHA V4."""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.flow_trainer import prepare_ground_truth_nodes
from src.sha256_ref import configure_cuda_memory, sha256_digest
from src.v4_boundary import ManifoldBoundary, apply_clamp_prepared, build_manifold_boundary
from src.v4_manifolds import (
    ManifoldPhysicsEngine,
    apply_manifold_closed_loop_schedule,
    enforce_fp32_precision,
    normalize_manifold,
)
from src.v4_solver import anneal_schedule
from src.v4_trainer import (
    UnrolledTrainerV4,
    _step_loss_from_parts,
    build_v4_init,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Phase-SHA V4 step stages on A100.")
    parser.add_argument("--variant", type=str, default="v4c", choices=["v4a", "v4b", "v4c"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=["eager", "default", "reduce-overhead"],
    )
    parser.add_argument("--backward-group-chunks", type=int, default=1)
    parser.add_argument("--allow-rnn", action="store_true")
    parser.add_argument("--anneal-chunk-size", type=int, default=None)
    parser.add_argument("--check-graph-breaks", action="store_true")
    parser.add_argument("--profile-eval", action="store_true")
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--skip-torch-profiler", action="store_true")
    parser.add_argument("--internal-graph-break-worker", action="store_true")
    return parser.parse_args()


def _sample_gpu_telemetry(stop_event: threading.Event, samples: List[Dict[str, float]]) -> None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ]
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=2.0)
            first = out.strip().splitlines()[0]
            util_s, mem_s, pwr_s = [p.strip() for p in first.split(",")[:3]]
            samples.append(
                {
                    "util_pct": float(util_s),
                    "mem_mib": float(mem_s),
                    "power_w": float(pwr_s),
                }
            )
        except Exception:
            pass
        stop_event.wait(0.25)


def _measure_trainer_throughput(
    trainer: UnrolledTrainerV4,
    warmup: int,
    steps: int,
    use_prefetch: bool,
) -> Dict[str, Any]:
    device = trainer.device
    for _ in range(warmup):
        if use_prefetch and hasattr(trainer, "start_prefetch"):
            trainer.start_prefetch()
        trainer.train_one_step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    samples: List[Dict[str, float]] = []
    stop_evt = threading.Event()
    poller: Optional[threading.Thread] = None
    if device.type == "cuda":
        poller = threading.Thread(target=_sample_gpu_telemetry, args=(stop_evt, samples), daemon=True)
        poller.start()

    step_wall_ms: List[float] = []
    try:
        for _ in range(steps):
            t0 = time.perf_counter()
            if hasattr(trainer, "train_one_step"):
                try:
                    trainer.train_one_step(sync_metrics=False)
                except TypeError:
                    trainer.train_one_step()
            t1 = time.perf_counter()
            step_wall_ms.append((t1 - t0) * 1000.0)
        t_sync0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_sync1 = time.perf_counter()
    finally:
        stop_evt.set()
        if poller is not None:
            poller.join(timeout=2.0)
        if hasattr(trainer, "stop_prefetch"):
            trainer.stop_prefetch()

    total_wall_ms = sum(step_wall_ms) + (t_sync1 - t_sync0) * 1000.0
    mean_wall_ms = total_wall_ms / max(1, steps)
    peak_gb = (
        torch.cuda.max_memory_allocated(device) / (1024.0**3)
        if device.type == "cuda"
        else 0.0
    )
    utils = [s["util_pct"] for s in samples] or [0.0]
    pwrs = [s["power_w"] for s in samples] or [0.0]
    mems = [s["mem_mib"] for s in samples] or [0.0]
    return {
        "mean_step_wall_ms": mean_wall_ms,
        "end_sync_tail_ms": (t_sync1 - t_sync0) * 1000.0,
        "peak_vram_gb": peak_gb,
        "gpu_util_mean": statistics.mean(utils),
        "gpu_util_max": max(utils),
        "power_w_mean": statistics.mean(pwrs),
        "power_w_max": max(pwrs),
        "nvidia_mem_mib_max": max(mems),
    }


def _profile_stage_breakdown(
    trainer: UnrolledTrainerV4,
    steps: int = 5,
    use_prefetch: bool = False,
) -> Dict[str, float]:
    """Decompose each training step into GPU event time and CPU wall time per stage."""
    device = trainer.device
    is_cuda = device.type == "cuda"
    manifold = trainer.manifold
    physics = trainer.physics
    m = max(1, trainer.num_particles)
    k_unroll = trainer.k_unroll
    chunk_len = max(1, trainer.bptt_chunk_len)
    group_chunks = max(1, int(getattr(trainer, "backward_group_chunks", 1)))
    backward_every = chunk_len * group_chunks
    raw_weights = [float(s + 1) for s in range(k_unroll)]
    w_sum = sum(raw_weights)
    step_weights = [w / w_sum for w in raw_weights]
    denom = float(max(1, k_unroll - 1))
    full_lam = torch.ones(32, dtype=torch.float32, device=device)

    stage_keys = [
        "data_gen",
        "h2d",
        "radar",
        "backbone",
        "retract_sched",
        "loss",
        "backward",
        "optim_ema",
        "sync_or_log",
    ]
    wall_acc: Dict[str, List[float]] = {k: [] for k in stage_keys}
    gpu_acc: Dict[str, List[float]] = {k: [] for k in stage_keys}

    if use_prefetch and hasattr(trainer, "start_prefetch"):
        trainer.start_prefetch()

    try:
        for _ in range(steps):
            trainer.model.train()
            trainer.train_model.train()

            step_wall = {k: 0.0 for k in stage_keys}
            ev_pairs: Dict[str, List[tuple[Any, Any]]] = {k: [] for k in stage_keys}

            def begin(name: str):
                if is_cuda:
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev0.record()
                else:
                    ev0 = None
                return time.perf_counter(), ev0

            def end(name: str, tok):
                t0, ev0 = tok
                if is_cuda:
                    ev1 = torch.cuda.Event(enable_timing=True)
                    ev1.record()
                    ev_pairs[name].append((ev0, ev1))
                step_wall[name] += (time.perf_counter() - t0) * 1000.0

            t_all0 = time.perf_counter()
            if is_cuda:
                ev_all0 = torch.cuda.Event(enable_timing=True)
                ev_all0.record()

            # 1. Data generation (+ optional H2D)
            if use_prefetch and hasattr(trainer, "_prefetcher") and trainer._prefetcher is not None:
                tok = begin("h2d")
                x_star_base, cb_base = trainer._prefetcher.next_on_device()
                end("h2d", tok)
                tok_init = begin("data_gen")
                if m > 1:
                    x_star = x_star_base.repeat_interleave(m, dim=0)
                    boundary = ManifoldBoundary.from_clamped_boundary(cb_base).repeat_for_candidates(m)
                else:
                    x_star = x_star_base
                    boundary = ManifoldBoundary.from_clamped_boundary(cb_base)
                u_init = build_v4_init(
                    boundary=boundary,
                    manifold=manifold,
                    num_particles=m,
                    zero_prob=trainer.zero_init_prob,
                    generator=trainer.train_rng,
                )
                end("data_gen", tok_init)
            else:
                tok = begin("data_gen")
                w16 = trainer._sample_train_w16(trainer.batch_size)
                x_star_base, cb_base = prepare_ground_truth_nodes(
                    w16,
                    rounds=trainer.rounds,
                    domain_mode=trainer.domain_mode,
                    num_bits=trainer.num_bits,
                    msg_bytes_len=trainer.msg_bytes_len,
                )
                if m > 1:
                    x_star = x_star_base.repeat_interleave(m, dim=0)
                    boundary = ManifoldBoundary.from_clamped_boundary(cb_base).repeat_for_candidates(m)
                else:
                    x_star = x_star_base
                    boundary = ManifoldBoundary.from_clamped_boundary(cb_base)
                u_init = build_v4_init(
                    boundary=boundary,
                    manifold=manifold,
                    num_particles=m,
                    zero_prob=trainer.zero_init_prob,
                    generator=trainer.train_rng,
                )
                end("data_gen", tok)

            bm = u_init.shape[0]
            clamp_mask = boundary.clamp_mask.to(torch.float32)
            trainer.optimizer.zero_grad(set_to_none=True)

            rc_fn = getattr(trainer, "_compiled_retract_clamp", None) or physics.retract_and_clamp
            add32_fn = getattr(trainer, "_compiled_spin_add32", None)
            loss_head_fn = getattr(trainer, "_compiled_loss_head", None)
            energy_fn = getattr(trainer, "_compiled_energy_core", None)
            clone_cg = bool(getattr(trainer, "_clone_cudagraph_outputs", False))

            prep = boundary.prepare_for_manifold(manifold)
            u_cur = apply_clamp_prepared(u_init.detach(), prep)
            if trainer.closed_loop_schedule and trainer.rounds > 16:
                u_cur = apply_clamp_prepared(
                    apply_manifold_closed_loop_schedule(
                        u_cur, manifold, trainer.rounds, add32_fn=add32_fn
                    ),
                    prep,
                )
            h_prev = None
            grouped_loss = None
            damp = trainer.damp
            full_lam = torch.ones(32, dtype=torch.float32, device=device)
            free_count = prep.free_mask.sum().clamp_min(1.0)
            wall_mask = torch.zeros((1, trainer.rounds, 1, 1), dtype=torch.float32, device=device)
            wall_bce_mul = torch.ones((trainer.rounds,), dtype=torch.float32, device=device)
            for wr in trainer.closed_loop_wall_rounds:
                if 0 <= int(wr) < trainer.rounds:
                    wall_mask[:, int(wr)] = 1.0
                    wall_bce_mul[int(wr)] = float(trainer.closed_loop_bce_scale)
            wall_free = prep.free_mask * wall_mask
            wall_den = wall_free.sum().clamp_min(1.0)
            b_offsets = torch.arange(bm // m, device=device) * m
            damp_t = torch.tensor(float(damp), dtype=torch.float32, device=device)
            w_xor_t = torch.tensor(float(trainer.w_xor), dtype=torch.float32, device=device)
            frac_vals = [float(s) / denom for s in range(k_unroll)]
            frac_vec = torch.tensor(frac_vals, dtype=torch.float32, device=device)
            step_frac_mat = frac_vec.unsqueeze(1).expand(k_unroll, bm).contiguous()
            lam_comm_vec = torch.tensor(
                [anneal_schedule(f, trainer.lam_comm_start, trainer.lam_comm_end, trainer.anneal_mode) for f in frac_vals],
                dtype=torch.float32, device=device,
            )
            lam_align_vec = torch.tensor(
                [anneal_schedule(f, trainer.lam_align_start, trainer.lam_align_end, trainer.anneal_mode) for f in frac_vals],
                dtype=torch.float32, device=device,
            )
            eta_vec = torch.tensor(
                [float(trainer.eta_max) * (0.2 + 0.8 * f) for f in frac_vals],
                dtype=torch.float32, device=device,
            )
            beta_vec = torch.tensor(
                [float(trainer.beta_max) * (f * f) for f in frac_vals],
                dtype=torch.float32, device=device,
            )

            for s in range(k_unroll):
                tok = begin("radar")
                grad_radar, res_feat, _, grad_riem = physics.compute_radar_prepared(
                    u_nodes=u_cur,
                    prep=prep,
                    lam=full_lam,
                    grad_scale=trainer.grad_scale,
                    w_xor=w_xor_t,
                    lam_comm=lam_comm_vec[s],
                    lam_align=lam_align_vec[s],
                    energy_fn=energy_fn,
                    return_raw_grad=True,
                    clone_cudagraph_outputs=clone_cg,
                )
                end("radar", tok)

                tok = begin("backbone")
                if clone_cg and hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                logits_s, delta_u_ambient, h_tilde = trainer.train_model(
                    u_nodes=u_cur,
                    grad_radar=grad_radar,
                    clamp_mask=clamp_mask,
                    res_features=res_feat,
                    k_spins=physics.K_spins,
                    step_frac=step_frac_mat[s],
                    lam=full_lam,
                    h_prev=h_prev,
                )
                if clone_cg:
                    logits_s = logits_s.clone()
                    delta_u_ambient = delta_u_ambient.clone()
                    h_tilde = h_tilde.clone()
                end("backbone", tok)

                tok = begin("retract_sched")
                if trainer.sync_h_damp and h_prev is not None:
                    h_next = (1.0 - damp_t) * h_prev + damp_t * h_tilde
                else:
                    h_next = h_tilde
                if clone_cg and hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                u_next = rc_fn(
                    u_cur,
                    logits_s,
                    delta_u_ambient,
                    grad_riem,
                    prep.u_clamped_masked,
                    prep.free_mask_exp,
                    damp_t,
                    eta_vec[s],
                    beta_vec[s],
                )
                if clone_cg:
                    u_next = u_next.clone()
                if trainer.closed_loop_schedule and trainer.rounds > 16:
                    u_sched = apply_manifold_closed_loop_schedule(
                        u_next, manifold, trainer.rounds, add32_fn=add32_fn
                    )
                    u_next = normalize_manifold(
                        prep.u_clamped_masked + prep.free_mask_exp * u_sched.to(torch.float32)
                    )
                end("retract_sched", tok)

                tok = begin("loss")
                step_loss, *_ = _step_loss_from_parts(
                    logits_spin=logits_s,
                    u_step_end=u_next,
                    x_star=x_star,
                    clamp_mask=prep.clamp_mask,
                    free_mask=prep.free_mask,
                    free_count=free_count,
                    wall_free=wall_free,
                    wall_den=wall_den,
                    wall_bce_mul=wall_bce_mul,
                    b_offsets=b_offsets,
                    full_lam=full_lam,
                    u_a_left=prep.u_a_left,
                    u_e_left=prep.u_e_left,
                    u_a_right=prep.u_a_right,
                    u_e_right=prep.u_e_right,
                    a_right_mask=prep.a_right_mask,
                    e_right_mask=prep.e_right_mask,
                    a_right_clamped=prep.a_right_clamped,
                    e_right_clamped=prep.e_right_clamped,
                    k_last_spins=prep.k_last_spins,
                    digest_target_bits=prep.digest_target_bits,
                    digest_target_mask=prep.digest_target_mask,
                    iv_words=prep.iv_words,
                    physics=physics,
                    num_particles=m,
                    wta_winner_weight=trainer.wta_winner_weight,
                    alpha_phys=trainer.alpha_phys,
                    bce_weight=trainer.bce_weight,
                    anti_collapse_weight=trainer.anti_collapse_weight,
                    logit_reg_weight=trainer.logit_reg_weight,
                    wavefront_weight=trainer.wavefront_weight,
                    w_xor=w_xor_t,
                    lam_comm=lam_comm_vec[s],
                    lam_align=lam_align_vec[s],
                    frac=frac_vec[s],
                    has_free_right=prep.has_free_right,
                    has_digest_prefix=prep.has_digest_prefix,
                    energy_fn=energy_fn,
                    loss_head_fn=loss_head_fn,
                    add32_fn=add32_fn,
                    clone_cudagraph_outputs=clone_cg,
                )
                weighted_term = step_weights[s] * step_loss
                grouped_loss = weighted_term if grouped_loss is None else (grouped_loss + weighted_term)
                end("loss", tok)

                is_chunk_end = ((s + 1) % chunk_len == 0) or (s == k_unroll - 1)
                is_backward_end = ((s + 1) % backward_every == 0) or (s == k_unroll - 1)
                if is_backward_end and grouped_loss is not None:
                    tok = begin("backward")
                    grouped_loss.backward()
                    grouped_loss = None
                    end("backward", tok)

                if is_chunk_end:
                    u_cur = u_next.detach()
                    h_prev = h_next.detach()
                else:
                    u_cur = u_next
                    h_prev = h_next

            tok = begin("optim_ema")
            nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.grad_clip, foreach=True)
            trainer.optimizer.step()
            trainer.scheduler.step()
            trainer._update_ema()
            end("optim_ema", tok)

            tok_sync = time.perf_counter()
            if is_cuda:
                ev_all1 = torch.cuda.Event(enable_timing=True)
                ev_all1.record()
                torch.cuda.synchronize(device)
                total_gpu_span = ev_all0.elapsed_time(ev_all1)
            else:
                total_gpu_span = 0.0
            sync_ms = (time.perf_counter() - tok_sync) * 1000.0
            t_all1 = time.perf_counter()
            total_wall = (t_all1 - t_all0) * 1000.0

            gpu_sum = 0.0
            for k in stage_keys:
                if k == "sync_or_log":
                    continue
                g_ms = sum(e0.elapsed_time(e1) for e0, e1 in ev_pairs[k]) if is_cuda else step_wall[k]
                gpu_acc[k].append(g_ms)
                wall_acc[k].append(step_wall[k])
                gpu_sum += g_ms
            wall_acc["sync_or_log"].append(sync_ms)
            gpu_acc["sync_or_log"].append(max(0.0, total_wall - gpu_sum))
            wall_acc.setdefault("total_wall", []).append(total_wall)
            gpu_acc.setdefault("total_gpu_span", []).append(total_gpu_span)
    finally:
        if hasattr(trainer, "stop_prefetch"):
            trainer.stop_prefetch()

    summary: Dict[str, float] = {}
    for k in stage_keys:
        summary[f"{k}_wall_ms"] = statistics.mean(wall_acc[k])
        summary[f"{k}_gpu_ms"] = statistics.mean(gpu_acc[k])
    summary["total_wall_ms"] = statistics.mean(wall_acc["total_wall"])
    summary["total_gpu_span_ms"] = statistics.mean(gpu_acc["total_gpu_span"])
    return summary


def _run_graph_break_audit(args: argparse.Namespace) -> None:
    print("\n=== Dynamo Graph-Break Audit (TORCH_LOGS=graph_breaks) ===", flush=True)
    env = os.environ.copy()
    env["TORCH_LOGS"] = "graph_breaks"
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        sys.executable,
        "-u",
        __file__,
        "--variant",
        args.variant,
        "--config",
        args.config,
        "--compile-mode",
        args.compile_mode,
        "--backward-group-chunks",
        str(args.backward_group_chunks),
        "--internal-graph-break-worker",
    ]
    if args.allow_rnn:
        cmd.append("--allow-rnn")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    combined = proc.stdout + "\n" + proc.stderr
    gb_lines = [ln.strip() for ln in combined.splitlines() if "graph break" in ln.lower() or "graph_break" in ln.lower()]
    print(f"graph_break_log_lines: {len(gb_lines)}", flush=True)
    seen = []
    for ln in gb_lines:
        clean = re.sub(r"\[[^\]]+\]", "", ln).strip()
        if clean not in seen:
            seen.append(clean)
    for idx, item in enumerate(seen[:20]):
        print(f"  [{idx+1}] {item[:220]}", flush=True)


def main() -> None:
    args = parse_args()
    enforce_fp32_precision()
    cfg_path = (
        Path(args.config)
        if args.config is not None
        else ROOT_DIR / "configs" / f"{args.variant}_r18_colab.yaml"
    )
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    cfg["variant"] = args.variant
    if args.compile_mode == "eager":
        cfg["use_torch_compile"] = "false"
    else:
        cfg["use_torch_compile"] = "auto"
        cfg["torch_compile_mode"] = args.compile_mode
    cfg["backward_group_chunks"] = int(args.backward_group_chunks)
    if args.allow_rnn:
        cfg["allow_rnn"] = True
    if args.anneal_chunk_size is not None:
        cfg["anneal_chunk_size"] = int(args.anneal_chunk_size)
    if args.no_prefetch:
        cfg["use_async_prefetch"] = False

    if torch.cuda.is_available():
        configure_cuda_memory(float(cfg.get("cuda_memory_fraction", 0.92)))
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    out_dir = ROOT_DIR / "scratch_profile_v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = UnrolledTrainerV4(cfg, output_dir=out_dir, device=device)

    tf32_mm = bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else False
    tf32_cudnn = bool(torch.backends.cudnn.allow_tf32) if device.type == "cuda" else False
    assert not tf32_mm and not tf32_cudnn, "TF32 MUST remain False!"

    if args.internal_graph_break_worker:
        trainer.train_one_step()
        trainer.train_one_step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print("GRAPH_BREAK_WORKER_DONE", flush=True)
        return

    print(
        f"[Profile V4] variant={args.variant} compile={trainer.compiled_active} "
        f"mode={args.compile_mode} chunk_len={trainer.bptt_chunk_len} "
        f"group_chunks={args.backward_group_chunks} "
        f"allow_tf32=(matmul={tf32_mm}, cudnn={tf32_cudnn})",
        flush=True,
    )

    use_prefetch = (not args.no_prefetch) and bool(cfg.get("use_async_prefetch", True))
    tp = _measure_trainer_throughput(
        trainer, warmup=args.warmup, steps=args.steps, use_prefetch=use_prefetch
    )
    print("\n=== 10-Step Real Throughput & A100 Telemetry ===", flush=True)
    for k, v in tp.items():
        print(f"  {k}: {v:.2f}", flush=True)
    eta_hr = (tp["mean_step_wall_ms"] / 1000.0) * 10000.0 / 3600.0
    print(f"  estimated_10k_steps_hours: {eta_hr:.2f} h", flush=True)

    stages = _profile_stage_breakdown(trainer, steps=5, use_prefetch=use_prefetch)
    print("\n=== Stage-by-Stage Timing Breakdown (mean over 5 steps) ===", flush=True)
    total_w = max(1e-6, stages["total_wall_ms"])
    for key in [
        "data_gen",
        "h2d",
        "radar",
        "backbone",
        "retract_sched",
        "loss",
        "backward",
        "optim_ema",
        "sync_or_log",
    ]:
        w_ms = stages[f"{key}_wall_ms"]
        g_ms = stages[f"{key}_gpu_ms"]
        print(
            f"  {key:14s} | cpu_wall={w_ms:8.2f} ms ({100.0*w_ms/total_w:5.1f}%) | gpu_event={g_ms:8.2f} ms ({100.0*g_ms/total_w:5.1f}%)",
            flush=True,
        )
    print(
        f"  {'TOTAL':14s} | step_wall={stages['total_wall_ms']:8.2f} ms | gpu_span={stages['total_gpu_span_ms']:8.2f} ms",
        flush=True,
    )

    if not args.skip_torch_profiler and device.type == "cuda":
        print("\n=== torch.profiler Top-15 Ops (1 steady-state step) ===", flush=True)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            trainer.train_one_step()
            torch.cuda.synchronize(device)
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15), flush=True)
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15), flush=True)

    if args.check_graph_breaks:
        args.config = str(cfg_path)
        _run_graph_break_audit(args)

    if args.profile_eval:
        print("\n=== Eval & SMC Slow-Thinking Timing ===", flush=True)
        trainer.ema_model.eval()
        eval_bs = int(cfg.get("eval_batch_size", 256))
        w16_eval = trainer.holdout_w16[:eval_bs].to(device)
        _, cb_eval = prepare_ground_truth_nodes(
            w16_eval,
            rounds=trainer.rounds,
            domain_mode=trainer.domain_mode,
            num_bits=trainer.num_bits,
            msg_bytes_len=trainer.msg_bytes_len,
        )
        mb_eval = ManifoldBoundary.from_clamped_boundary(cb_eval)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tz0 = time.perf_counter()
        trainer.solver.infer_deterministic_zero(mb_eval)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tz1 = time.perf_counter()
        trainer.solver.infer_deterministic_zero(mb_eval)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tz2 = time.perf_counter()
        print(
            f"  infer_deterministic_zero(bs={eval_bs}): cold={(tz1-tz0)*1000.0:.2f} ms | warm={(tz2-tz1)*1000.0:.2f} ms",
            flush=True,
        )

        w16_smc = trainer.holdout_w16[:4].to(device)
        dig_smc = sha256_digest(w16_smc, rounds=trainer.rounds)
        mb_smc = build_manifold_boundary(
            digest=dig_smc,
            rounds=trainer.rounds,
            domain_mode=trainer.domain_mode,
            num_bits=trainer.num_bits,
            msg_bytes_len=trainer.msg_bytes_len,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ts0 = time.perf_counter()
        trainer.solver.solve(
            target_digests=dig_smc,
            boundary=mb_smc,
            generator=torch.Generator(device=device).manual_seed(77777),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ts1 = time.perf_counter()
        trainer.solver.solve(
            target_digests=dig_smc,
            boundary=mb_smc,
            generator=torch.Generator(device=device).manual_seed(77777),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ts2 = time.perf_counter()
        print(
            f"  smc_solve(4x anneal_K={trainer.solver.num_candidates}, K_infer={trainer.solver.num_steps}): cold={(ts1-ts0)*1000.0:.2f} ms | warm={(ts2-ts1)*1000.0:.2f} ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
