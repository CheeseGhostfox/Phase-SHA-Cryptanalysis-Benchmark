"""Training & evaluation entrypoint for Phase-SHA V8A Torus Flow Unscrambler."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List

import torch

from v8a import assert_strict_fp32, configure_strict_fp32
from v8a.entropy_objective import (
    PeriodicMarginalDensity,
    RoundTransportObjective,
    compute_circular_metrics,
    compute_discrete_metrics,
    estimate_raw_input_baseline,
)
from v8a.preimage_probe import evaluate_frozen_preimage_probe
from v8a.sha256_stream import OnlineSHA256Stream
from v8a.torus_flow import TorusFlow, circular_abs_diff, dequantize_bits
from v8a.torus_flow_v8b import V8BTorusFlow


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _query_gpu_telemetry(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "dtype": str(torch.get_default_dtype()),
    }
    if not torch.cuda.is_available():
        return info
    info["gpu_name"] = torch.cuda.get_device_name(device)
    info["vram_alloc_gb"] = round(torch.cuda.memory_allocated(device) / 1e9, 3)
    info["vram_peak_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 3)
    info["vram_total_gb"] = round(torch.cuda.get_device_properties(device).total_memory / 1e9, 3)
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip():
            line = out.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                info["gpu_util_pct"] = float(parts[0])
                info["smi_vram_used_mb"] = float(parts[1])
                info["smi_vram_total_mb"] = float(parts[2])
                info["gpu_temp_c"] = float(parts[3])
    except Exception:
        pass
    return info


@torch.no_grad()
def _evaluate_holdout(
    flow: TorusFlow,
    density: PeriodicMarginalDensity,
    stream: OnlineSHA256Stream,
    eval_seed: int,
    draws: int = 4,
    batch_size: int = 2048,
) -> Dict[str, Any]:
    was_flow_train = flow.training
    was_dens_train = density.training
    flow.eval()
    density.eval()

    device = stream.device
    eval_gen = torch.Generator(device=device).manual_seed(int(eval_seed))
    h_eval = stream.eval_H_bits

    ext_sum = 0.0
    r1_acc = torch.zeros(256, device=device)
    r2_acc = torch.zeros(256, device=device)
    pcb_acc = torch.zeros(256, device=device)
    bij_err_max = 0.0

    for d in range(draws):
        th_out_chunks: List[torch.Tensor] = []
        nll_sum = torch.zeros(256, device=device)
        total_n = 0
        for start in range(0, h_eval.shape[0], batch_size):
            hb = h_eval[start : start + batch_size]
            th0 = dequantize_bits(hb, deterministic=False, generator=eval_gen)
            tho = flow(th0)
            if d == 0 and start == 0:
                th_rec = flow.inverse(tho[:512])
                bij_err_max = max(
                    bij_err_max, float(circular_abs_diff(th_rec, th0[:512]).max().item())
                )
            dout = density(tho)
            bs = hb.shape[0]
            nll_sum += dout["per_channel_nll"] * bs
            total_n += bs
            th_out_chunks.append(tho)

        th_all = torch.cat(th_out_chunks, dim=0)
        mean_nll = nll_sum / total_n
        pcb = (math.log(2.0 * math.pi) - mean_nll) / math.log(2.0)
        ext_sum += float(pcb.sum().item())
        pcb_acc += pcb
        cm = compute_circular_metrics(th_all)
        r1_acc += cm["R1"]
        r2_acc += cm["R2"]

    r1_avg = r1_acc / draws
    r2_avg = r2_acc / draws
    r_max = torch.maximum(r1_avg, r2_avg)
    pcb_avg = pcb_acc / draws
    disc = compute_discrete_metrics(flow, h_eval, batch_size=batch_size)
    norm_err = density.verify_normalization()

    if was_flow_train:
        flow.train()
    if was_dens_train:
        density.train()

    return {
        "eval_extracted_bits": ext_sum / draws,
        "collapsed_ch_35": int((r1_avg > 0.35).sum().item()),
        "collapsed_ch_55": int((r1_avg > 0.55).sum().item()),
        "collapsed_ch_max_35": int((r_max > 0.35).sum().item()),
        "collapsed_ch_max_55": int((r_max > 0.55).sum().item()),
        "collapsed_ch_disc_90": int(disc["collapsed_ch_disc_90"]),
        "max_R1": float(r1_avg.max().item()),
        "max_R2": float(r2_avg.max().item()),
        "mean_R1": float(r1_avg.mean().item()),
        "max_R_disc": float(disc["max_R_disc"]),
        "bijection_max_err": bij_err_max,
        "density_norm_err": norm_err,
        "R1_vec": [round(float(x), 4) for x in r1_avg.detach().cpu().tolist()],
        "R2_vec": [round(float(x), 4) for x in r2_avg.detach().cpu().tolist()],
        "R_disc_vec": [round(float(x), 4) for x in disc["R_disc"].detach().cpu().tolist()],
        "per_channel_bits": [round(float(x), 4) for x in pcb_avg.detach().cpu().tolist()],
    }


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    configure_strict_fp32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
        torch.cuda.reset_peak_memory_stats(device)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    telemetry_path = run_dir / "telemetry.json"
    if metrics_path.exists():
        metrics_path.unlink()

    log_tail: deque[str] = deque(maxlen=50)

    def log_msg(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_tail.append(line)

    stream = OnlineSHA256Stream(
        k_bits=args.k_bits,
        rounds=args.rounds,
        n_eval=args.eval_size,
        eval_seed=20260801 + args.rounds * 100 + args.k_bits,
        train_seed=20260901 + args.rounds * 100 + args.seed,
        device=device,
    )

    arch = getattr(args, "arch", "v8a")
    use_ckpt = bool(getattr(args, "use_checkpoint", False)) or (
        arch in ("v8b_t1", "v8b_t2") and (args.batch_size >= 2048 or args.rounds >= 6)
    )
    lambda_lip = float(getattr(args, "lambda_lip", -1.0))
    if lambda_lip < 0.0:
        lambda_lip = 2e-3 if arch in ("v8b_t1", "v8b_t2") else 0.0
    lip_sigma = float(getattr(args, "lip_sigma", 0.15))

    if arch == "v8a":
        flow = TorusFlow(
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            ff_dim=args.ff_dim,
            num_blocks=args.num_blocks,
            use_attn=not args.no_attn,
            mask_seed=42 + args.seed,
        ).to(device)
    else:
        flow = V8BTorusFlow(
            flow_layout=arch,
            rounds=args.rounds,
            num_layers=args.num_layers,
            c_bit=getattr(args, "c_bit", 128),
            c_mid=getattr(args, "c_mid", 96),
            d_glob=getattr(args, "d_glob", 384),
            ff_glob=getattr(args, "ff_glob", 512),
            num_arx_blocks=getattr(args, "num_arx_blocks", 2),
            use_checkpoint=use_ckpt,
        ).to(device)

    density = PeriodicMarginalDensity(
        num_channels=256,
        num_components=args.num_components,
        seed=args.seed,
    ).to(device)
    init_ckpt = getattr(args, "init_ckpt", "")
    if init_ckpt:
        ckpt_data = torch.load(init_ckpt, map_location=device, weights_only=False)
        flow.load_state_dict(ckpt_data["flow_state"], strict=False)
        if "density_state" in ckpt_data:
            density.load_state_dict(ckpt_data["density_state"], strict=False)
        log_msg(f"Loaded warm-start checkpoint from {init_ckpt}")

    assert_strict_fp32(flow)
    assert_strict_fp32(density)

    param_count = flow.param_count()
    theory_max_bits = 256 - args.k_bits
    run_name = args.run_name or run_dir.name
    log_msg(
        f"Init {run_name}: arch={arch}, mode={args.mode}, R={args.rounds}, k={args.k_bits}, "
        f"ops={len(flow.layers)}, params={param_count:,}, B={args.batch_size}, ckpt={use_ckpt}, lip={lambda_lip:.1e}"
    )

    # Honest raw input redundancy baseline
    train_base = stream.sample_train_batch(min(4096, max(2048, args.batch_size)))
    baseline = estimate_raw_input_baseline(
        eval_h_bits=stream.eval_H_bits,
        train_h_bits=train_base,
        num_components=args.num_components,
        steps=200,
        lr=0.03,
        seed=args.seed + 10,
    )
    base_bits = float(baseline["baseline_input_bits"])
    base_ch55 = int(baseline["baseline_collapsed_ch_55"])
    log_msg(f"Baseline raw input: {base_bits:.2f} bits (ch55={base_ch55}, theory_max={theory_max_bits})")

    transport_obj = None
    if args.mode == "round_transport":
        transport_obj = RoundTransportObjective(
            rounds=args.rounds,
            k_bits=args.k_bits,
            total_layers=len(flow.layers),
            custom_stage_slices=getattr(flow, "transport_stage_slices", None) or None,
            custom_target_rounds=getattr(flow, "transport_target_rounds", None) or None,
            custom_stage_kinds=getattr(flow, "transport_stage_kinds", None) or None,
        )
        log_msg(
            f"RoundTransport stages={transport_obj.num_stages}, target_rounds={transport_obj.target_rounds}, "
            f"lps={transport_obj.layers_per_stage}, transport_steps={args.transport_steps}"
        )

    opt = torch.optim.AdamW(
        [
            {"params": flow.parameters(), "lr": args.lr_phi, "weight_decay": args.weight_decay},
            {"params": density.parameters(), "lr": args.lr_psi, "weight_decay": 0.0},
        ]
    )

    warmup_steps = min(100, max(10, args.steps // 10))

    def _lr_phi_mult(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return 0.1 + 0.9 * (step_idx / max(1, warmup_steps))
        prog = (step_idx - warmup_steps) / max(1, args.steps - warmup_steps)
        return 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    def _lr_psi_mult(step_idx: int) -> float:
        prog = step_idx / max(1, args.steps)
        return 0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[_lr_phi_mult, _lr_psi_mult])

    best_eval_bits = -1e9
    best_flow_gain = -1e9
    latest_probe: Dict[str, Any] = {
        "w_bit_acc": 0.5,
        "w_bit_acc_linear": 0.5,
        "w_bit_acc_mlp": 0.5,
        "w_bit_acc_active": 0.5,
        "z_score": 0.0,
        "exact_pass_1": 0.0,
        "exact_pass_1_count": 0,
        "exact_pass_64": 0.0,
        "exact_pass_64_count": 0,
        "n_eval": args.eval_size,
    }
    latest_eval = _evaluate_holdout(flow, density, stream, eval_seed=args.seed + 500)

    def _publish_telemetry(
        step: int,
        status: str,
        train_bits: float,
        steps_per_sec: float,
        eta_sec: float,
        phase_label: str,
        tf_bit_acc: float = 0.0,
        rollout_bit_acc: float = 0.0,
    ) -> None:
        flow_gain = latest_eval["eval_extracted_bits"] - base_bits
        payload = {
            "run_name": run_name,
            "arch": arch,
            "status": status,
            "mode": args.mode,
            "phase": phase_label,
            "rounds": args.rounds,
            "k_bits": args.k_bits,
            "theory_max_bits": theory_max_bits,
            "step": step,
            "total_steps": args.steps,
            "transport_steps": args.transport_steps,
            "batch_size": args.batch_size,
            "num_layers": len(flow.layers),
            "hidden_dim": args.hidden_dim,
            "ff_dim": args.ff_dim,
            "lambda_lip": lambda_lip,
            "param_count": param_count,
            "baseline_input_bits": round(base_bits, 4),
            "baseline_ch55": base_ch55,
            "train_extracted_bits": round(train_bits, 4),
            "eval_extracted_bits": round(latest_eval["eval_extracted_bits"], 4),
            "flow_gain_bits": round(flow_gain, 4),
            "best_eval_bits": round(best_eval_bits, 4) if best_eval_bits > -1e8 else round(latest_eval["eval_extracted_bits"], 4),
            "best_flow_gain_bits": round(best_flow_gain, 4) if best_flow_gain > -1e8 else round(flow_gain, 4),
            "tf_bit_acc": round(tf_bit_acc, 4),
            "rollout_bit_acc": round(rollout_bit_acc, 4),
            "collapsed_ch_35": latest_eval["collapsed_ch_35"],
            "collapsed_ch_55": latest_eval["collapsed_ch_55"],
            "collapsed_ch_max_35": latest_eval["collapsed_ch_max_35"],
            "collapsed_ch_max_55": latest_eval["collapsed_ch_max_55"],
            "collapsed_ch_disc_90": latest_eval["collapsed_ch_disc_90"],
            "max_R1": round(latest_eval["max_R1"], 4),
            "max_R2": round(latest_eval["max_R2"], 4),
            "mean_R1": round(latest_eval["mean_R1"], 4),
            "max_R_disc": round(latest_eval["max_R_disc"], 4),
            "bijection_max_err": latest_eval["bijection_max_err"],
            "density_norm_err": latest_eval["density_norm_err"],
            "R1_vec": latest_eval["R1_vec"],
            "R2_vec": latest_eval["R2_vec"],
            "R_disc_vec": latest_eval["R_disc_vec"],
            "per_channel_bits": latest_eval["per_channel_bits"],
            "probe": latest_probe,
            "steps_per_sec": round(steps_per_sec, 2),
            "eta_sec": round(eta_sec, 1),
            "gpu": _query_gpu_telemetry(device),
            "log_tail": list(log_tail),
            "updated_at": time.time(),
        }
        _atomic_write_json(telemetry_path, payload)

    _publish_telemetry(0, "running", latest_eval["eval_extracted_bits"], 0.0, 0.0, "init")

    t_start = time.time()
    t_window = t_start
    window_steps = 0
    last_train_bits = latest_eval["eval_extracted_bits"]
    last_tf_acc = 0.0
    last_roll_acc = 0.0

    for step in range(1, args.steps + 1):
        flow.train()
        density.train()

        in_transport = (
            transport_obj is not None and args.transport_steps > 0 and step <= args.transport_steps
        )
        if args.mode == "round_transport":
            phase_label = "Phase-1 (round_transport)" if in_transport else "Phase-2 (e2e_unsupervised)"
        else:
            phase_label = "T1_unsupervised"

        if in_transport:
            h_bits, states_bits = stream.sample_train_batch(args.batch_size, return_states=True)
            prog = (step - 1) / max(1, args.transport_steps - 1)
            alpha = 0.1 + 0.9 * prog
            tout = transport_obj.compute_loss(
                flow=flow,
                density=density,
                h_bits=h_bits,
                states_bits=states_bits,
                alpha=alpha,
                generator=stream.train_gen,
                lip_sigma=(lip_sigma if lambda_lip > 0.0 else 0.0),
                progress=prog,
            )
            loss = tout["loss"]
            if lambda_lip > 0.0:
                loss = loss + lambda_lip * tout["loss_lip"]
            last_train_bits = float(tout["extracted_bits"].item())
            last_tf_acc = float(tout["tf_bit_acc"].item())
            last_roll_acc = float(tout["rollout_bit_acc"].item())
            extra_loss_info = {
                "loss_tf": float(tout["loss_tf"].item()),
                "loss_rollout": float(tout["loss_rollout"].item()),
                "tf_bit_acc": round(last_tf_acc, 4),
                "rollout_bit_acc": round(last_roll_acc, 4),
            }
        else:
            h_bits = stream.sample_train_batch(args.batch_size, return_states=False)
            th0 = dequantize_bits(h_bits, deterministic=False, generator=stream.train_gen)
            if lambda_lip > 0.0 and hasattr(flow, "flow_layout"):
                th_out, lip_val = flow(th0, return_lip=True, lip_sigma=lip_sigma)
            else:
                th_out = flow(th0)
                lip_val = torch.zeros((), device=device, dtype=torch.float32)
            mout = density(th_out)
            loss = mout["loss_marg"] + (lambda_lip * lip_val if lambda_lip > 0.0 else 0.0)
            last_train_bits = float(mout["extracted_bits"].item())
            extra_loss_info = {}

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss detected at step {step}: {loss.item()}")

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), args.grad_clip)
        opt.step()
        sched.step()

        window_steps += 1
        do_eval = (step % args.eval_every == 0) or (step == args.steps)
        do_probe = (step % args.probe_every == 0) or (step == args.steps)

        if do_eval:
            now = time.time()
            sps = window_steps / max(1e-6, now - t_window)
            eta = (args.steps - step) / max(1e-6, sps)
            t_window = now
            window_steps = 0

            latest_eval = _evaluate_holdout(flow, density, stream, eval_seed=args.seed + 500 + step)
            eval_bits = latest_eval["eval_extracted_bits"]
            flow_gain = eval_bits - base_bits

            if eval_bits > best_eval_bits:
                best_eval_bits = eval_bits
                best_flow_gain = flow_gain
                torch.save(
                    {
                        "step": step,
                        "flow_state": flow.state_dict(),
                        "density_state": density.state_dict(),
                        "eval_extracted_bits": eval_bits,
                        "flow_gain_bits": flow_gain,
                    },
                    run_dir / "best.pt",
                )

            if do_probe:
                latest_probe = evaluate_frozen_preimage_probe(
                    flow=flow,
                    stream=stream,
                    probe_train_size=args.probe_train_size,
                    probe_steps=350,
                    batch_size=512,
                    seed=args.seed + 10000 + step,
                )

            extra_trans = (
                f" tf={last_tf_acc:.4f} roll={last_roll_acc:.4f}"
                if in_transport
                else ""
            )
            log_msg(
                f"step {step:5d}/{args.steps} [{phase_label}] "
                f"train={last_train_bits:.2f}b eval={eval_bits:.2f}b (gain={flow_gain:+.2f}b) "
                f"ch35={latest_eval['collapsed_ch_35']} ch55={latest_eval['collapsed_ch_55']} "
                f"disc90={latest_eval['collapsed_ch_disc_90']} bij={latest_eval['bijection_max_err']:.2e}"
                f"{extra_trans} "
                f"w_acc={latest_probe['w_bit_acc']:.4f}(z={latest_probe['z_score']:.1f}) "
                f"p@1={latest_probe['exact_pass_1']:.4f} p@64={latest_probe['exact_pass_64']:.4f} "
                f"[{sps:.2f} it/s]"
            )

            _append_jsonl(
                metrics_path,
                {
                    "step": step,
                    "phase": phase_label,
                    "train_extracted_bits": round(last_train_bits, 4),
                    "eval_extracted_bits": round(eval_bits, 4),
                    "flow_gain_bits": round(flow_gain, 4),
                    "collapsed_ch_35": latest_eval["collapsed_ch_35"],
                    "collapsed_ch_55": latest_eval["collapsed_ch_55"],
                    "collapsed_ch_disc_90": latest_eval["collapsed_ch_disc_90"],
                    "max_R1": round(latest_eval["max_R1"], 4),
                    "max_R2": round(latest_eval["max_R2"], 4),
                    "bijection_max_err": latest_eval["bijection_max_err"],
                    "w_bit_acc": latest_probe["w_bit_acc"],
                    "z_score": latest_probe["z_score"],
                    "exact_pass_1": latest_probe["exact_pass_1"],
                    "exact_pass_64": latest_probe["exact_pass_64"],
                    "steps_per_sec": round(sps, 2),
                    **extra_loss_info,
                },
            )
            _publish_telemetry(
                step, "running", last_train_bits, sps, eta, phase_label,
                tf_bit_acc=last_tf_acc, rollout_bit_acc=last_roll_acc,
            )

    total_sec = time.time() - t_start
    torch.save(
        {
            "step": args.steps,
            "flow_state": flow.state_dict(),
            "density_state": density.state_dict(),
            "eval_extracted_bits": latest_eval["eval_extracted_bits"],
            "flow_gain_bits": latest_eval["eval_extracted_bits"] - base_bits,
        },
        run_dir / "final.pt",
    )
    _publish_telemetry(
        args.steps,
        "completed",
        last_train_bits,
        args.steps / max(1e-6, total_sec),
        0.0,
        "completed",
        tf_bit_acc=last_tf_acc,
        rollout_bit_acc=last_roll_acc,
    )

    summary = {
        "run_name": run_name,
        "arch": arch,
        "mode": args.mode,
        "rounds": args.rounds,
        "k_bits": args.k_bits,
        "steps": args.steps,
        "transport_steps": args.transport_steps,
        "param_count": param_count,
        "lambda_lip": lambda_lip,
        "tf_bit_acc": round(last_tf_acc, 4),
        "rollout_bit_acc": round(last_roll_acc, 4),
        "baseline_input_bits": round(base_bits, 4),
        "baseline_ch55": base_ch55,
        "eval_extracted_bits": round(latest_eval["eval_extracted_bits"], 4),
        "flow_gain_bits": round(latest_eval["eval_extracted_bits"] - base_bits, 4),
        "best_eval_bits": round(best_eval_bits, 4),
        "best_flow_gain_bits": round(best_flow_gain, 4),
        "collapsed_ch_35": latest_eval["collapsed_ch_35"],
        "collapsed_ch_55": latest_eval["collapsed_ch_55"],
        "collapsed_ch_disc_90": latest_eval["collapsed_ch_disc_90"],
        "max_R1": round(latest_eval["max_R1"], 4),
        "max_R2": round(latest_eval["max_R2"], 4),
        "bijection_max_err": latest_eval["bijection_max_err"],
        "w_bit_acc": latest_probe["w_bit_acc"],
        "w_bit_acc_linear": latest_probe["w_bit_acc_linear"],
        "w_bit_acc_mlp": latest_probe["w_bit_acc_mlp"],
        "w_bit_acc_active": latest_probe["w_bit_acc_active"],
        "z_score": latest_probe["z_score"],
        "exact_pass_1": latest_probe["exact_pass_1"],
        "exact_pass_1_count": latest_probe["exact_pass_1_count"],
        "exact_pass_64": latest_probe["exact_pass_64"],
        "exact_pass_64_count": latest_probe["exact_pass_64_count"],
        "wall_sec": round(total_sec, 1),
    }
    _atomic_write_json(run_dir / "summary.json", summary)
    log_msg(f"Completed {run_name} in {total_sec:.1f}s")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-SHA V8A/V8B Torus Flow Trainer")
    p.add_argument("--arch", choices=["v8a", "v8b_t1", "v8b_t2"], default="v8a")
    p.add_argument("--mode", choices=["unsupervised", "round_transport"], default="unsupervised")
    p.add_argument("--rounds", type=int, required=True)
    p.add_argument("--k_bits", type=int, default=32)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--transport_steps", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--num_layers", type=int, default=24)
    p.add_argument("--hidden_dim", type=int, default=384)
    p.add_argument("--ff_dim", type=int, default=512)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--c_bit", type=int, default=128)
    p.add_argument("--c_mid", type=int, default=96)
    p.add_argument("--d_glob", type=int, default=384)
    p.add_argument("--ff_glob", type=int, default=512)
    p.add_argument("--num_arx_blocks", type=int, default=2)
    p.add_argument("--lambda_lip", type=float, default=-1.0)
    p.add_argument("--lip_sigma", type=float, default=0.15)
    p.add_argument("--use_checkpoint", action="store_true")
    p.add_argument("--init_ckpt", type=str, default="")
    p.add_argument("--no_attn", action="store_true")
    p.add_argument("--num_components", type=int, default=12)
    p.add_argument("--lr_phi", type=float, default=5e-4)
    p.add_argument("--lr_psi", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--probe_every", type=int, default=1000)
    p.add_argument("--eval_size", type=int, default=4096)
    p.add_argument("--probe_train_size", type=int, default=32768)
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run_training(build_parser().parse_args())
