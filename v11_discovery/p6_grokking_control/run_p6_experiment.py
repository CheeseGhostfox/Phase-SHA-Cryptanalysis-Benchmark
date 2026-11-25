"""Main P6 runner: Real SHA-256 vs Random ROTR/K Controls + Grokking Dissection."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from cipher_variants import (  # noqa: E402
    build_variants,
    verify_real_sha256_parity,
)
from datasets_and_baselines import (  # noqa: E402
    TRACK_A_TARGET_NAMES,
    TrackAData,
    TrackBData,
    build_track_a_data,
    build_track_b_data,
    compute_auc_batched,
    compute_zero_param_baselines_a,
    compute_zero_param_baselines_b,
    evaluate_track_a,
    evaluate_track_b,
)
from models_and_dissection import (  # noqa: E402
    GrokkingResMLP,
    dissect_sae_and_ablate,
    dissect_track_a_walsh_fourier,
    dissect_track_b_walsh,
)


ALL_ROUNDS: List[int] = [4, 6, 8, 10, 12]
ALL_WDS: List[float] = [0.1, 0.5, 1.0, 2.0]
VARIANT_ORDER: List[str] = ["REAL_SHA256", "CTRL_RAND_1", "CTRL_RAND_2"]


def lock_strict_fp32() -> Dict[str, bool]:
    """Enforce pure IEEE-754 FP32 with TF32 completely disabled."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    return {
        "fp32_locked": True,
        "tf32_disabled": (
            (not bool(torch.backends.cuda.matmul.allow_tf32))
            and (not bool(torch.backends.cudnn.allow_tf32))
        ),
    }


def _weight_l2_norm(model: nn.Module) -> float:
    sq = 0.0
    for p in model.parameters():
        sq += float(p.detach().pow(2).sum().item())
    return math.sqrt(sq)


def _reset_parameters_seeded(model: nn.Module, seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


def _clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _materialize_model(
    d_in: int,
    d_out: int,
    state_dict: Dict[str, torch.Tensor],
    device: torch.device,
) -> GrokkingResMLP:
    m = GrokkingResMLP(d_in=d_in, d_out=d_out, d_model=192, n_blocks=4).to(device)
    m.load_state_dict(state_dict)
    m.eval()
    return m


def _train_track_a_run(
    model: GrokkingResMLP,
    train_fn: nn.Module,
    data: TrackAData,
    rounds: int,
    wd: float,
    variant_name: str,
    steps_base: int,
    max_steps: int,
    eval_every: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    allow_auto_extend: bool = True,
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    """Train one Track A (R, wd, variant) run and return summary dict + cloned state_dict."""
    t0 = time.perf_counter()
    _reset_parameters_seeded(model, seed, device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.98),
        weight_decay=wd,
        foreach=(device.type == "cuda"),
    )
    bce = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()

    x_tr = data.x_all[data.idx_train]
    yb_tr = data.y_bin_all[data.idx_train]
    yc_tr = data.y_cont_all[data.idx_train]

    x_ho = data.x_all[data.idx_holdout]
    yb_ho = data.y_bin_all[data.idx_holdout]
    yc_ho = data.y_cont_all[data.idx_holdout]

    n_tr = x_tr.shape[0]
    bs = min(batch_size, n_tr)
    gen = torch.Generator(device=device).manual_seed(seed + 17)

    trajectory: List[Dict[str, float]] = []
    t_mem: int | None = None
    t_grok: int | None = None
    min_wn = _weight_l2_norm(model)
    extended = False
    target_steps = steps_base

    step = 0
    last_train_loss = 0.0
    while step < target_steps:
        model.train()
        lr_now = 1e-3 * min(1.0, float(step + 1) / 200.0)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        idx = torch.randint(0, n_tr, (bs,), generator=gen, device=device)
        xb = x_tr[idx]
        ybb = yb_tr[idx]
        ycb = yc_tr[idx]

        out = train_fn(xb)
        loss_b = bce(out[:, :16], ybb)
        loss_c = mse_loss_fn(out[:, 16:18], ycb)
        loss = loss_b + 0.25 * loss_c

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, foreach=(device.type == "cuda")
        )
        opt.step()

        last_train_loss = float(loss.item())
        step += 1

        if (step % eval_every == 0) or (step == target_steps):
            model.eval()
            with torch.no_grad():
                tr_out = model(x_tr)
                tr_acc = float(
                    ((tr_out[:, :16] > 0.0).to(torch.float32) == yb_tr)
                    .to(torch.float32)
                    .mean()
                    .item()
                )
                ho_out = model(x_ho)
                ho_eval = evaluate_track_a(ho_out, yb_ho, yc_ho)

            wn = _weight_l2_norm(model)
            if wn < min_wn:
                min_wn = wn

            if t_mem is None and tr_acc >= 0.95:
                t_mem = step
            if t_grok is None and (
                ho_eval["digest12_z_null_auc"] >= 6.0
                or ho_eval["best_digest12_bit_z_null"] >= 6.0
                or ho_eval["digest12_auc"] >= 0.60
            ):
                t_grok = step

            trajectory.append(
                {
                    "step": step,
                    "train_acc": tr_acc,
                    "holdout_acc": float(ho_eval["digest12_acc"]),
                    "holdout_auc": float(ho_eval["digest12_auc"]),
                    "holdout_all16_auc": float(ho_eval["auc"]),
                    "holdout_mse": float(ho_eval["mse"]),
                    "weight_norm": wn,
                }
            )

            if (
                allow_auto_extend
                and (not extended)
                and (step == steps_base)
                and (rounds >= 6)
                and (max_steps > steps_base)
            ):
                emerging = (
                    ho_eval["digest12_auc"] >= 0.515
                    or abs(float(ho_eval["digest12_z_null_auc"])) >= 3.5
                    or float(ho_eval["best_digest12_bit_z_null"]) >= 3.5
                    or float(ho_eval["z_null_mse"]) >= 3.5
                )
                if (tr_acc >= 0.95) and emerging:
                    extended = True
                    target_steps = max_steps

    model.eval()
    with torch.no_grad():
        tr_out = model(x_tr)
        tr_bin_acc = float(
            ((tr_out[:, :16] > 0.0).to(torch.float32) == yb_tr).to(torch.float32).mean().item()
        )
        tr_bin_auc = float(compute_auc_batched(tr_out[:, :16].T, yb_tr.T).mean().item())
        tr_mse = float(mse_loss_fn(tr_out[:, 16:18], yc_tr).item())

        ho_out = model(x_ho)
        ho_eval = evaluate_track_a(ho_out, yb_ho, yc_ho)

    wall_sec = time.perf_counter() - t0
    final_wn = _weight_l2_norm(model)

    record = {
        "variant": variant_name,
        "rounds": rounds,
        "wd": wd,
        "train_final": {
            "acc": tr_bin_acc,
            "auc": tr_bin_auc,
            "mse": tr_mse,
            "loss": last_train_loss,
        },
        "holdout": ho_eval,
        "dynamics": {
            "t_mem": t_mem,
            "t_grok": t_grok,
            "min_weight_norm": min_wn,
            "final_weight_norm": final_wn,
            "extended": extended,
            "total_steps": step,
            "wall_sec": wall_sec,
            "trajectory": trajectory,
        },
        "retest": None,
    }
    return record, _clone_state_dict(model)


def _train_track_b_run(
    model: GrokkingResMLP,
    train_fn: nn.Module,
    data: TrackBData,
    rounds: int,
    wd: float,
    variant_name: str,
    steps_base: int,
    max_steps: int,
    eval_every: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    allow_auto_extend: bool = True,
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    """Train one Track B (R, wd, variant) run and return summary dict + cloned state_dict."""
    t0 = time.perf_counter()
    _reset_parameters_seeded(model, seed, device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.98),
        weight_decay=wd,
        foreach=(device.type == "cuda"),
    )
    bce = nn.BCEWithLogitsLoss()

    n_tr = data.x_train.shape[0]
    bs = min(batch_size, n_tr)
    gen = torch.Generator(device=device).manual_seed(seed + 31)

    trajectory: List[Dict[str, float]] = []
    t_mem: int | None = None
    t_grok: int | None = None
    min_wn = _weight_l2_norm(model)
    extended = False
    target_steps = steps_base

    def _eval_full(x: torch.Tensor) -> torch.Tensor:
        chunks = []
        for s in range(0, x.shape[0], 4096):
            chunks.append(model(x[s : s + 4096]).reshape(-1))
        return torch.cat(chunks, dim=0)

    step = 0
    last_train_loss = 0.0
    while step < target_steps:
        model.train()
        lr_now = 1e-3 * min(1.0, float(step + 1) / 200.0)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        idx = torch.randint(0, n_tr, (bs,), generator=gen, device=device)
        xb = data.x_train[idx]
        yb = data.y_train[idx]

        logits = train_fn(xb).reshape(-1)
        loss = bce(logits, yb)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, foreach=(device.type == "cuda")
        )
        opt.step()

        last_train_loss = float(loss.item())
        step += 1

        if (step % eval_every == 0) or (step == target_steps):
            model.eval()
            with torch.no_grad():
                tr_logits = _eval_full(data.x_train)
                tr_acc = float(
                    ((tr_logits > 0.0).to(torch.float32) == data.y_train)
                    .to(torch.float32)
                    .mean()
                    .item()
                )
                ho_logits = _eval_full(data.x_holdout)
                ho_eval = evaluate_track_b(ho_logits, data.y_holdout)

            wn = _weight_l2_norm(model)
            if wn < min_wn:
                min_wn = wn

            if t_mem is None and tr_acc >= 0.95:
                t_mem = step
            if t_grok is None and (
                ho_eval["z_null_auc"] >= 6.0 or ho_eval["auc"] >= 0.60
            ):
                t_grok = step

            trajectory.append(
                {
                    "step": step,
                    "train_acc": tr_acc,
                    "holdout_acc": float(ho_eval["acc"]),
                    "holdout_auc": float(ho_eval["auc"]),
                    "weight_norm": wn,
                }
            )

            if (
                allow_auto_extend
                and (not extended)
                and (step == steps_base)
                and (rounds >= 6)
                and (max_steps > steps_base)
            ):
                emerging = (
                    ho_eval["auc"] >= 0.515 or abs(float(ho_eval["z_null_auc"])) >= 3.5
                )
                if (tr_acc >= 0.95) and emerging:
                    extended = True
                    target_steps = max_steps

    model.eval()
    with torch.no_grad():
        tr_logits = _eval_full(data.x_train)
        tr_acc = float(
            ((tr_logits > 0.0).to(torch.float32) == data.y_train)
            .to(torch.float32)
            .mean()
            .item()
        )
        tr_auc = float(compute_auc_batched(tr_logits, data.y_train).item())
        ho_logits = _eval_full(data.x_holdout)
        ho_eval = evaluate_track_b(ho_logits, data.y_holdout)

    wall_sec = time.perf_counter() - t0
    final_wn = _weight_l2_norm(model)

    record = {
        "variant": variant_name,
        "rounds": rounds,
        "wd": wd,
        "train_final": {
            "acc": tr_acc,
            "auc": tr_auc,
            "loss": last_train_loss,
        },
        "holdout": ho_eval,
        "dynamics": {
            "t_mem": t_mem,
            "t_grok": t_grok,
            "min_weight_norm": min_wn,
            "final_weight_norm": final_wn,
            "extended": extended,
            "total_steps": step,
            "wall_sec": wall_sec,
            "trajectory": trajectory,
        },
        "retest": None,
    }
    return record, _clone_state_dict(model)


def _compute_z_vs_ctrl_pair(
    real_val: float,
    real_se: float,
    c1_val: float,
    c1_se: float,
    c2_val: float,
    c2_se: float,
    higher_is_better: bool = True,
) -> Dict[str, object]:
    """Compute z_vs_ctrl1, z_vs_ctrl2, and conservative z_vs_ctrl against strongest ctrl."""
    sign = 1.0 if higher_is_better else -1.0
    z1 = sign * (real_val - c1_val) / math.sqrt(real_se * real_se + c1_se * c1_se + 1e-12)
    z2 = sign * (real_val - c2_val) / math.sqrt(real_se * real_se + c2_se * c2_se + 1e-12)
    if higher_is_better:
        star = 1 if c1_val >= c2_val else 2
        c_star_val, c_star_se = (c1_val, c1_se) if star == 1 else (c2_val, c2_se)
    else:
        star = 1 if c1_val <= c2_val else 2
        c_star_val, c_star_se = (c1_val, c1_se) if star == 1 else (c2_val, c2_se)
    z_star = sign * (real_val - c_star_val) / math.sqrt(
        real_se * real_se + c_star_se * c_star_se + 1e-12
    )
    return {
        "delta_vs_ctrl1": float(real_val - c1_val),
        "delta_vs_ctrl2": float(real_val - c2_val),
        "z_vs_ctrl1": float(z1),
        "z_vs_ctrl2": float(z2),
        "ctrl_star": f"CTRL_RAND_{star}",
        "z_vs_ctrl_min": float(z_star),
    }


def _summarize_comparison(
    track: str,
    rec_real: Dict[str, object],
    rec_c1: Dict[str, object],
    rec_c2: Dict[str, object],
) -> Dict[str, object]:
    hr = rec_real["holdout"]
    hc1 = rec_c1["holdout"]
    hc2 = rec_c2["holdout"]

    if track == "track_a":
        comp_d12 = _compute_z_vs_ctrl_pair(
            float(hr["digest12_auc"]),
            float(hr["digest12_auc_pool_se"]),
            float(hc1["digest12_auc"]),
            float(hc1["digest12_auc_pool_se"]),
            float(hc2["digest12_auc"]),
            float(hc2["digest12_auc_pool_se"]),
            higher_is_better=True,
        )
        comp_all16 = _compute_z_vs_ctrl_pair(
            float(hr["auc"]),
            float(hr["auc_pool_se"]),
            float(hc1["auc"]),
            float(hc1["auc_pool_se"]),
            float(hc2["auc"]),
            float(hc2["auc_pool_se"]),
            higher_is_better=True,
        )
        comp_mse = _compute_z_vs_ctrl_pair(
            float(hr["mse"]),
            float(hr["mse_pool_se"]),
            float(hc1["mse"]),
            float(hc1["mse_pool_se"]),
            float(hc2["mse"]),
            float(hc2["mse_pool_se"]),
            higher_is_better=False,
        )
        base_comp = {
            "delta_auc_vs_ctrl1": comp_d12["delta_vs_ctrl1"],
            "delta_auc_vs_ctrl2": comp_d12["delta_vs_ctrl2"],
            "z_vs_ctrl1": comp_d12["z_vs_ctrl1"],
            "z_vs_ctrl2": comp_d12["z_vs_ctrl2"],
            "z_vs_ctrl_min": comp_d12["z_vs_ctrl_min"],
            "ctrl_star": comp_d12["ctrl_star"],
            "all16": comp_all16,
            "mse": comp_mse,
        }
    else:
        comp_b = _compute_z_vs_ctrl_pair(
            float(hr["auc"]),
            float(hr["auc_pool_se"]),
            float(hc1["auc"]),
            float(hc1["auc_pool_se"]),
            float(hc2["auc"]),
            float(hc2["auc_pool_se"]),
            higher_is_better=True,
        )
        base_comp = {
            "delta_auc_vs_ctrl1": comp_b["delta_vs_ctrl1"],
            "delta_auc_vs_ctrl2": comp_b["delta_vs_ctrl2"],
            "z_vs_ctrl1": comp_b["z_vs_ctrl1"],
            "z_vs_ctrl2": comp_b["z_vs_ctrl2"],
            "z_vs_ctrl_min": comp_b["z_vs_ctrl_min"],
            "ctrl_star": comp_b["ctrl_star"],
        }

    t_real = rec_real["dynamics"]["t_grok"]
    t_c1 = rec_c1["dynamics"]["t_grok"]
    t_c2 = rec_c2["dynamics"]["t_grok"]
    ctrl_t = [t for t in (t_c1, t_c2) if t is not None]
    t_ctrl_min = min(ctrl_t) if ctrl_t else None

    if t_real is not None and t_ctrl_min is not None:
        delta_t: object = int(t_ctrl_min - t_real)
    elif t_real is not None and t_ctrl_min is None:
        delta_t = "real_only"
    elif t_real is None and t_ctrl_min is not None:
        delta_t = "ctrl_only"
    else:
        delta_t = None

    base_comp["t_grok_real"] = t_real
    base_comp["t_grok_ctrl_min"] = t_ctrl_min
    base_comp["delta_t_grok"] = delta_t
    return base_comp


def _run_retest_track_a(
    models: Dict[str, GrokkingResMLP],
    datasets: Dict[str, TrackAData],
) -> Dict[str, Dict[str, object]]:
    out = {}
    res_map: Dict[str, Dict[str, object]] = {}
    for vname in VARIANT_ORDER:
        m = models[vname]
        d = datasets[vname]
        m.eval()
        with torch.no_grad():
            logits = m(d.x_all[d.idx_retest])
            ev = evaluate_track_a(logits, d.y_bin_all[d.idx_retest], d.y_cont_all[d.idx_retest])
        res_map[vname] = ev

    comp = _compute_z_vs_ctrl_pair(
        float(res_map["REAL_SHA256"]["digest12_auc"]),
        float(res_map["REAL_SHA256"]["digest12_auc_pool_se"]),
        float(res_map["CTRL_RAND_1"]["digest12_auc"]),
        float(res_map["CTRL_RAND_1"]["digest12_auc_pool_se"]),
        float(res_map["CTRL_RAND_2"]["digest12_auc"]),
        float(res_map["CTRL_RAND_2"]["digest12_auc_pool_se"]),
        higher_is_better=True,
    )
    for vname in VARIANT_ORDER:
        r = dict(res_map[vname])
        r["z_vs_ctrl_auc"] = comp["z_vs_ctrl_min"] if vname == "REAL_SHA256" else None
        out[vname] = r
    return out


def _run_retest_track_b(
    models: Dict[str, GrokkingResMLP],
    datasets: Dict[str, TrackBData],
) -> Dict[str, Dict[str, object]]:
    res_map: Dict[str, Dict[str, object]] = {}
    for vname in VARIANT_ORDER:
        m = models[vname]
        d = datasets[vname]
        m.eval()
        with torch.no_grad():
            chunks = [m(d.x_retest[s : s + 4096]).reshape(-1) for s in range(0, d.x_retest.shape[0], 4096)]
            logits = torch.cat(chunks, dim=0)
            ev = evaluate_track_b(logits, d.y_retest)
        res_map[vname] = ev

    comp = _compute_z_vs_ctrl_pair(
        float(res_map["REAL_SHA256"]["auc"]),
        float(res_map["REAL_SHA256"]["auc_pool_se"]),
        float(res_map["CTRL_RAND_1"]["auc"]),
        float(res_map["CTRL_RAND_1"]["auc_pool_se"]),
        float(res_map["CTRL_RAND_2"]["auc"]),
        float(res_map["CTRL_RAND_2"]["auc_pool_se"]),
        higher_is_better=True,
    )
    out = {}
    for vname in VARIANT_ORDER:
        r = dict(res_map[vname])
        r["z_vs_ctrl_auc"] = comp["z_vs_ctrl_min"] if vname == "REAL_SHA256" else None
        out[vname] = r
    return out


def _calibrate_and_compile(
    device: torch.device,
    run_a: bool,
    run_b: bool,
    n_runs_total: int,
    steps_override: int,
) -> Tuple[
    int,
    str,
    GrokkingResMLP | None,
    nn.Module | None,
    GrokkingResMLP | None,
    nn.Module | None,
]:
    """Compile model(s) once and calibrate step budget."""
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    model_a = GrokkingResMLP(14, 18, 192, 4).to(device) if run_a else None
    model_b = GrokkingResMLP(512, 1, 192, 4).to(device) if run_b else None

    if device.type == "cuda":
        train_fn_a = torch.compile(model_a) if model_a is not None else None
        train_fn_b = torch.compile(model_b) if model_b is not None else None
    else:
        train_fn_a = model_a
        train_fn_b = model_b

    if device.type != "cuda":
        return (steps_override if steps_override > 0 else 300), gpu_name, model_a, train_fn_a, model_b, train_fn_b

    # Warmup compile + calibrate
    variants = build_variants()
    if model_a is not None and train_fn_a is not None:
        d_a = build_track_a_data(4, variants["REAL_SHA256"], device)
        _train_track_a_run(
            model_a, train_fn_a, d_a, 4, 1.0, "WARMUP", 20, 20, 250, 2048, 1, device, False
        )
    if model_b is not None and train_fn_b is not None:
        d_b = build_track_b_data(4, variants["REAL_SHA256"], device)
        _train_track_b_run(
            model_b, train_fn_b, d_b, 4, 1.0, "WARMUP", 20, 20, 250, 2048, 2, device, False
        )
    else:
        # Need Track B timing only if Track B is not compiled? No, time whichever is active!
        d_b = None

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if model_b is not None and train_fn_b is not None and d_b is not None:
        _train_track_b_run(
            model_b, train_fn_b, d_b, 4, 1.0, "CAL", 400, 400, 250, 2048, 99, device, False
        )
    elif model_a is not None and train_fn_a is not None:
        _train_track_a_run(
            model_a, train_fn_a, d_a, 4, 1.0, "CAL", 400, 400, 250, 2048, 99, device, False
        )
    torch.cuda.synchronize()
    dt_400 = time.perf_counter() - t0
    sec_per_step = dt_400 / 400.0
    est_12k_min = (float(n_runs_total) * 12000.0 * sec_per_step) / 60.0
    est_15k_min = (float(n_runs_total) * 15000.0 * sec_per_step) / 60.0
    est_10k_min = (float(n_runs_total) * 10000.0 * sec_per_step) / 60.0

    if steps_override > 0:
        steps_base = steps_override
    elif ("A100" in gpu_name.upper()) and est_15k_min <= 40.0:
        steps_base = 15000
    elif est_12k_min > 50.0:
        steps_base = 10000
    else:
        steps_base = 12000

    print(
        f"[calibration] gpu={gpu_name}, n_runs={n_runs_total}, dt_400={dt_400:.3f}s, "
        f"est_10k={est_10k_min:.1f}m, est_12k={est_12k_min:.1f}m, est_15k={est_15k_min:.1f}m -> steps_base={steps_base}",
        flush=True,
    )
    return steps_base, gpu_name, model_a, train_fn_a, model_b, train_fn_b


def run_experiment(
    mode: str,
    out_dir: Path,
    track: str,
    rounds_list: List[int],
    steps_override: int,
    summary_name: str,
) -> None:
    t_start = time.perf_counter()
    fp32_info = lock_strict_fp32()

    run_a = track in ("all", "track_a")
    run_b = track in ("all", "track_b")

    if mode == "smoke":
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
            raise RuntimeError("Local smoke test must run with CUDA_VISIBLE_DEVICES=''")
        device = torch.device("cpu")
        rounds_list = [4]
        wd_list = [1.0]
        max_steps = 300
        eval_every = 150
        sae_steps = 200
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Full run requires CUDA GPU on Colab!")
        device = torch.device("cuda")
        wd_list = ALL_WDS
        max_steps = 24000
        eval_every = 250
        sae_steps = 2000

    n_runs_total = (
        len(rounds_list) * len(wd_list) * len(VARIANT_ORDER) * ((1 if run_a else 0) + (1 if run_b else 0))
    )

    verify_real_sha256_parity(device="cpu")
    print("[verify] REAL_SHA256 parity with src/sha256_ref.py PASSED.", flush=True)

    s_tie = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    y_tie = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    auc_tie = float(compute_auc_batched(s_tie, y_tie).item())
    assert abs(auc_tie - 2.0 / 3.0) < 1e-5, f"Tie AUC mismatch: {auc_tie}"
    print(f"[verify] Tie-aware batched AUC PASSED (tie_val={auc_tie:.4f}).", flush=True)

    steps_base, gpu_name, model_a, train_fn_a, model_b, train_fn_b = _calibrate_and_compile(
        device=device,
        run_a=run_a,
        run_b=run_b,
        n_runs_total=n_runs_total,
        steps_override=steps_override,
    )
    if mode == "smoke":
        max_steps = steps_base

    variants = build_variants()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "meta": {
            "mode": mode,
            "track_filter": track,
            "gpu_name": gpu_name,
            "fp32_locked": fp32_info["fp32_locked"],
            "tf32_disabled": fp32_info["tf32_disabled"],
            "rounds": rounds_list,
            "weight_decays": wd_list,
            "variants": VARIANT_ORDER,
            "ctrl_configs": {
                "CTRL_RAND_1": variants["CTRL_RAND_1"].to_meta_dict(),
                "CTRL_RAND_2": variants["CTRL_RAND_2"].to_meta_dict(),
            },
            "steps_base": steps_base,
            "max_steps_auto_extend": max_steps,
            "eval_every": eval_every,
            "batch_size": 2048,
            "sae_steps": sae_steps,
            "total_wall_sec": 0.0,
        },
        "track_a": {},
        "track_b": {},
        "comparisons": {"track_a": {}, "track_b": {}},
        "zero_param_baselines": {"track_a": {}, "track_b": {}},
        "dissections": [],
    }

    for r in rounds_list:
        r_key = str(r)
        print(f"\n=== Round R={r} (track={track}) ===", flush=True)

        ds_a: Dict[str, TrackAData] = (
            {vn: build_track_a_data(r, variants[vn], device) for vn in VARIANT_ORDER}
            if run_a
            else {}
        )
        ds_b: Dict[str, TrackBData] = (
            {vn: build_track_b_data(r, variants[vn], device) for vn in VARIANT_ORDER}
            if run_b
            else {}
        )

        if run_a:
            summary["zero_param_baselines"]["track_a"][r_key] = {
                vn: compute_zero_param_baselines_a(ds_a[vn]) for vn in VARIANT_ORDER
            }
            summary["track_a"][r_key] = {}
            summary["comparisons"]["track_a"][r_key] = {}
        if run_b:
            summary["zero_param_baselines"]["track_b"][r_key] = {
                vn: compute_zero_param_baselines_b(ds_b[vn]) for vn in VARIANT_ORDER
            }
            summary["track_b"][r_key] = {}
            summary["comparisons"]["track_b"][r_key] = {}

        r4_best_a: Dict[str, Tuple[float, float, Dict[str, object], Dict[str, torch.Tensor]]] = {}
        r4_best_b: Dict[str, Tuple[float, float, Dict[str, object], Dict[str, torch.Tensor]]] = {}

        for wd_idx, wd in enumerate(wd_list):
            wd_key = str(wd)
            if run_a:
                summary["track_a"][r_key][wd_key] = {}
            if run_b:
                summary["track_b"][r_key][wd_key] = {}

            states_a: Dict[str, Dict[str, torch.Tensor]] = {}
            states_b: Dict[str, Dict[str, torch.Tensor]] = {}

            for v_idx, vn in enumerate(VARIANT_ORDER):
                msg_parts = [f"[R={r} wd={wd} {vn}]"]

                if run_a and model_a is not None and train_fn_a is not None:
                    seed_a = 100000 + r * 1000 + wd_idx * 100 + v_idx
                    rec_a, sd_a = _train_track_a_run(
                        model=model_a,
                        train_fn=train_fn_a,
                        data=ds_a[vn],
                        rounds=r,
                        wd=wd,
                        variant_name=vn,
                        steps_base=steps_base,
                        max_steps=max_steps,
                        eval_every=eval_every,
                        batch_size=2048,
                        seed=seed_a,
                        device=device,
                        allow_auto_extend=(mode == "full"),
                    )
                    summary["track_a"][r_key][wd_key][vn] = rec_a
                    states_a[vn] = sd_a
                    msg_parts.append(
                        f"A: d12_auc={rec_a['holdout']['digest12_auc']:.4f} "
                        f"(best={rec_a['holdout']['best_bit_name']}:{rec_a['holdout']['best_bit_auc']:.4f}, "
                        f"mse={rec_a['holdout']['mse']:.4f}, ext={rec_a['dynamics']['extended']}, "
                        f"t={rec_a['dynamics']['wall_sec']:.1f}s)"
                    )

                if run_b and model_b is not None and train_fn_b is not None:
                    seed_b = 200000 + r * 1000 + wd_idx * 100 + v_idx
                    rec_b, sd_b = _train_track_b_run(
                        model=model_b,
                        train_fn=train_fn_b,
                        data=ds_b[vn],
                        rounds=r,
                        wd=wd,
                        variant_name=vn,
                        steps_base=steps_base,
                        max_steps=max_steps,
                        eval_every=eval_every,
                        batch_size=2048,
                        seed=seed_b,
                        device=device,
                        allow_auto_extend=(mode == "full"),
                    )
                    summary["track_b"][r_key][wd_key][vn] = rec_b
                    states_b[vn] = sd_b
                    msg_parts.append(
                        f"B: auc={rec_b['holdout']['auc']:.4f} "
                        f"(z={rec_b['holdout']['z_null_auc']:+.2f}, ext={rec_b['dynamics']['extended']}, "
                        f"t={rec_b['dynamics']['wall_sec']:.1f}s)"
                    )

                print(" | ".join(msg_parts), flush=True)

            if run_a:
                comp_a = _summarize_comparison(
                    "track_a",
                    summary["track_a"][r_key][wd_key]["REAL_SHA256"],
                    summary["track_a"][r_key][wd_key]["CTRL_RAND_1"],
                    summary["track_a"][r_key][wd_key]["CTRL_RAND_2"],
                )
                summary["comparisons"]["track_a"][r_key][wd_key] = comp_a
                if r >= 6:
                    ha = summary["track_a"][r_key][wd_key]["REAL_SHA256"]["holdout"]
                    trig_a = (
                        abs(float(ha["digest12_z_null_auc"])) >= 4.0
                        or float(ha["best_digest12_bit_z_null"]) >= 4.0
                        or float(ha["z_null_mse"]) >= 4.0
                        or abs(float(comp_a["z_vs_ctrl_min"])) >= 4.0
                    )
                    if trig_a:
                        mats_a = {
                            vn: _materialize_model(14, 18, states_a[vn], device)
                            for vn in VARIANT_ORDER
                        }
                        retest_a = _run_retest_track_a(mats_a, ds_a)
                        for vn in VARIANT_ORDER:
                            summary["track_a"][r_key][wd_key][vn]["retest"] = retest_a[vn]
                        b_idx = int(ha["best_digest12_bit_idx"])
                        for vn in ("REAL_SHA256", "CTRL_RAND_1"):
                            wf = dissect_track_a_walsh_fourier(mats_a[vn], ds_a[vn], b_idx)
                            sae_res = dissect_sae_and_ablate(
                                mats_a[vn],
                                ds_a[vn].x_all[ds_a[vn].idx_train],
                                ds_a[vn].x_all[ds_a[vn].idx_holdout],
                                ds_a[vn].y_bin_all[ds_a[vn].idx_holdout, b_idx],
                                target_idx=b_idx,
                                sae_steps=sae_steps,
                            )
                            summary["dissections"].append(
                                {
                                    "reason": "R_ge_6_trigger",
                                    "track": "track_a",
                                    "rounds": r,
                                    "wd": wd,
                                    "variant": vn,
                                    "walsh_fourier": wf,
                                    "sae": sae_res,
                                }
                            )
                            torch.save(states_a[vn], ckpt_dir / f"ckpt_A_R{r}_wd{wd}_{vn}.pt")
                else:
                    for vn in ("REAL_SHA256", "CTRL_RAND_1"):
                        score_a = float(summary["track_a"][r_key][wd_key][vn]["holdout"]["auc"])
                        if vn not in r4_best_a or score_a > r4_best_a[vn][0]:
                            r4_best_a[vn] = (
                                score_a,
                                wd,
                                summary["track_a"][r_key][wd_key][vn]["holdout"],
                                states_a[vn],
                            )

            if run_b:
                comp_b = _summarize_comparison(
                    "track_b",
                    summary["track_b"][r_key][wd_key]["REAL_SHA256"],
                    summary["track_b"][r_key][wd_key]["CTRL_RAND_1"],
                    summary["track_b"][r_key][wd_key]["CTRL_RAND_2"],
                )
                summary["comparisons"]["track_b"][r_key][wd_key] = comp_b
                if r >= 6:
                    hb = summary["track_b"][r_key][wd_key]["REAL_SHA256"]["holdout"]
                    trig_b = (
                        abs(float(hb["z_null_auc"])) >= 4.0
                        or abs(float(comp_b["z_vs_ctrl_min"])) >= 4.0
                    )
                    if trig_b:
                        mats_b = {
                            vn: _materialize_model(512, 1, states_b[vn], device)
                            for vn in VARIANT_ORDER
                        }
                        retest_b = _run_retest_track_b(mats_b, ds_b)
                        for vn in VARIANT_ORDER:
                            summary["track_b"][r_key][wd_key][vn]["retest"] = retest_b[vn]
                        for vn in ("REAL_SHA256", "CTRL_RAND_1"):
                            wf = dissect_track_b_walsh(mats_b[vn], ds_b[vn])
                            sae_res = dissect_sae_and_ablate(
                                mats_b[vn],
                                ds_b[vn].x_train,
                                ds_b[vn].x_holdout,
                                ds_b[vn].y_holdout,
                                target_idx=0,
                                sae_steps=sae_steps,
                            )
                            summary["dissections"].append(
                                {
                                    "reason": "R_ge_6_trigger",
                                    "track": "track_b",
                                    "rounds": r,
                                    "wd": wd,
                                    "variant": vn,
                                    "walsh_fourier": wf,
                                    "sae": sae_res,
                                }
                            )
                            torch.save(states_b[vn], ckpt_dir / f"ckpt_B_R{r}_wd{wd}_{vn}.pt")
                else:
                    for vn in ("REAL_SHA256", "CTRL_RAND_1"):
                        score_b = float(summary["track_b"][r_key][wd_key][vn]["holdout"]["auc"])
                        if vn not in r4_best_b or score_b > r4_best_b[vn][0]:
                            r4_best_b[vn] = (
                                score_b,
                                wd,
                                summary["track_b"][r_key][wd_key][vn]["holdout"],
                                states_b[vn],
                            )

        # Mandatory R=4 positive calibration dissections
        if r == 4:
            print("[dissection] Running mandatory R=4 positive calibration dissections...", flush=True)
            if run_a:
                for vn in ("REAL_SHA256", "CTRL_RAND_1"):
                    _, best_wd_a, ho_a, sd_a = r4_best_a[vn]
                    mod_a = _materialize_model(14, 18, sd_a, device)
                    per_bit = ho_a["per_bit_auc"]
                    e3_candidates = [(j, per_bit[TRACK_A_TARGET_NAMES[j]]) for j in range(12, 16)]
                    h4_candidates = [(j, per_bit[TRACK_A_TARGET_NAMES[j]]) for j in range(0, 6)]
                    best_e3_idx = max(e3_candidates, key=lambda t: t[1])[0]
                    best_h4_idx = max(h4_candidates, key=lambda t: t[1])[0]
                    for tag, b_idx in (("E3_best", best_e3_idx), ("H4_0_best", best_h4_idx)):
                        wf = dissect_track_a_walsh_fourier(mod_a, ds_a[vn], b_idx)
                        sae_res = dissect_sae_and_ablate(
                            mod_a,
                            ds_a[vn].x_all[ds_a[vn].idx_train],
                            ds_a[vn].x_all[ds_a[vn].idx_holdout],
                            ds_a[vn].y_bin_all[ds_a[vn].idx_holdout, b_idx],
                            target_idx=b_idx,
                            sae_steps=sae_steps,
                        )
                        summary["dissections"].append(
                            {
                                "reason": f"R4_positive_calibration_{tag}",
                                "track": "track_a",
                                "rounds": 4,
                                "wd": best_wd_a,
                                "variant": vn,
                                "walsh_fourier": wf,
                                "sae": sae_res,
                            }
                        )
                    torch.save(sd_a, ckpt_dir / f"ckpt_A_R4_wd{best_wd_a}_{vn}.pt")

            if run_b:
                for vn in ("REAL_SHA256", "CTRL_RAND_1"):
                    _, best_wd_b, _, sd_b = r4_best_b[vn]
                    mod_b = _materialize_model(512, 1, sd_b, device)
                    wf_b = dissect_track_b_walsh(mod_b, ds_b[vn])
                    sae_b = dissect_sae_and_ablate(
                        mod_b,
                        ds_b[vn].x_train,
                        ds_b[vn].x_holdout,
                        ds_b[vn].y_holdout,
                        target_idx=0,
                        sae_steps=sae_steps,
                    )
                    summary["dissections"].append(
                        {
                            "reason": "R4_positive_calibration",
                            "track": "track_b",
                            "rounds": 4,
                            "wd": best_wd_b,
                            "variant": vn,
                            "walsh_fourier": wf_b,
                            "sae": sae_b,
                        }
                    )
                    torch.save(sd_b, ckpt_dir / f"ckpt_B_R4_wd{best_wd_b}_{vn}.pt")

        summary["meta"]["total_wall_sec"] = time.perf_counter() - t_start
        with open(out_dir / summary_name, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[progress] Saved {summary_name} through R={r}, elapsed={summary['meta']['total_wall_sec']:.1f}s", flush=True)

    run_meta = {
        "command": " ".join(sys.argv),
        "mode": mode,
        "track": track,
        "rounds": rounds_list,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "gpu_name": gpu_name,
        "fp32_locked": fp32_info["fp32_locked"],
        "tf32_disabled": fp32_info["tf32_disabled"],
        "seeds": {
            "SEED_SPLIT_A": 20261106,
            "SEED_B_TRAIN": 20261116,
            "SEED_B_HOLD": 998244353,
            "SEED_B_RETEST": 777666555,
            "SEED_CTRL_1": 1000003,
            "SEED_CTRL_2": 2000003,
        },
        "total_wall_sec": summary["meta"]["total_wall_sec"],
        "colab_stopped": None,
    }
    meta_name = summary_name.replace("sweep_summary", "run_meta") if "sweep_summary" in summary_name else f"{Path(summary_name).stem}_meta.json"
    with open(out_dir / meta_name, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    print(f"[done] All finished in {run_meta['total_wall_sec']:.2f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="P6 Grokking Control Sweep")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--track", choices=["all", "track_a", "track_b"], default="all")
    parser.add_argument("--rounds", type=str, default="4,6,8,10,12")
    parser.add_argument("--steps-base", type=int, default=0)
    parser.add_argument("--summary-name", type=str, default="p6_sweep_summary.json")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_THIS_DIR / "results"),
    )
    args = parser.parse_args()
    rounds_list = [int(x.strip()) for x in args.rounds.split(",") if x.strip()]
    run_experiment(
        mode=args.mode,
        out_dir=Path(args.out_dir),
        track=args.track,
        rounds_list=rounds_list,
        steps_override=args.steps_base,
        summary_name=args.summary_name,
    )


if __name__ == "__main__":
    main()
