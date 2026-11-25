"""P10 full experiment runner (pure FP32, online samples, atomic JSON output):
- Reference verification: 100% bit-exact match with src.sha256_ref.sha256_trace + exact O(D) inverse check
- Part A: Tiny-bitwidth scaling law D_95(n) for Ugly-Algebra vs Human Hybrid basis
- Part B: Full 32-bit Ugly-Algebra error curve eps(D), bit-profile, & multi-op composition
- Part C: 256-bit SHA-256 single-round -> G^k / G^{-k} flattening vs isomorphic random control
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sha256_ref import SHA256_IV, SHA256_K, sha256_trace
from v11_discovery.p10_ugly_algebra.arx_ops import (
    enforce_fp32_lock,
    int_to_bits,
    make_cipher_profile,
    mask_n,
    op_add,
    op_rotr,
    op_xor,
    sha256_k_rounds_from_state,
    state_words_to_bits256,
)
from v11_discovery.p10_ugly_algebra.models import (
    HumanBasisOperatorBaseline,
    SingleRoundKoopmanFlattener,
    UglyAlgebraLifter,
)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _bit_acc(logits: torch.Tensor, target_bits: torch.Tensor) -> float:
    return float(((logits > 0.0) == (target_bits > 0.5)).to(torch.float32).mean().item())


def _sample_nbit_triple(
    batch: int, n_bits: int, device: torch.device, gen: torch.Generator
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hi = 1 << int(n_bits)
    x = torch.randint(0, hi, (batch,), dtype=torch.int64, device=device, generator=gen)
    y = torch.randint(0, hi, (batch,), dtype=torch.int64, device=device, generator=gen)
    w = torch.randint(0, hi, (batch,), dtype=torch.int64, device=device, generator=gen)
    return x, y, w


def verify_primitives_exact(device: torch.device) -> Dict[str, Any]:
    """Verify bit-exact agreement with src.sha256_ref and exact O(D) algebraic inverse."""
    gen = torch.Generator(device=device).manual_seed(20261125)
    w16 = torch.randint(0, 1 << 32, (32, 16), dtype=torch.int64, device=device, generator=gen)
    ref = sha256_trace(w16, rounds=16)
    s0 = torch.tensor(SHA256_IV, dtype=torch.int64, device=device).unsqueeze(0).expand(32, -1).clone()
    traj = sha256_k_rounds_from_state(s0, w16, tuple(SHA256_K), start_round=0, num_rounds=16)
    # After 16 rounds: S_16 = [A_16, A_15, A_14, A_13, E_16, E_15, E_14, E_13]
    s16_ref = ref.S_R
    s16_ours = traj[:, 16, :]
    max_diff = int((s16_ref ^ s16_ours).max().item())

    # Verify analytic inverse of InvertibleConditionedStep & SingleRoundKoopmanFlattener
    fl = SingleRoundKoopmanFlattener(dim=64, op_mode="invertible_coupling", controlled=True).to(device)
    z = torch.randn((16, 64), dtype=torch.float32, device=device)
    c = torch.randn((16, 128), dtype=torch.float32, device=device)
    z1 = fl.forward_step(z, ctrl=c)
    z_rec = fl.inverse_step(z1, ctrl=c)
    inv_err_coupling = float((z - z_rec).abs().max().item())

    fl_lin = SingleRoundKoopmanFlattener(dim=64, op_mode="linear_ortho", controlled=True).to(device)
    g_mat = fl_lin.step_op.get_matrix()
    z1_l = fl_lin.forward_step(z, ctrl=c, g_cache=g_mat)
    z_rec_l = fl_lin.inverse_step(z1_l, ctrl=c, g_cache=g_mat)
    inv_err_ortho = float((z - z_rec_l).abs().max().item())

    assert max_diff == 0, f"SHA-256 mismatch: max_diff={max_diff}"
    assert inv_err_coupling < 1e-4, f"coupling inverse err too high: {inv_err_coupling}"
    assert inv_err_ortho < 1e-4, f"ortho inverse err too high: {inv_err_ortho}"
    return {
        "sha256_ref_xor_diff": max_diff,
        "coupling_roundtrip_max_err": inv_err_coupling,
        "ortho_roundtrip_max_err": inv_err_ortho,
    }


def train_and_eval_arx_algebra(
    n_bits: int,
    dim: int,
    rank: int,
    method: str,
    steps: int,
    batch_size: int,
    eval_n: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """Train one coordinate system (UglyBilinear / UglyInvertible / HumanHybrid) and measure held-out accuracy."""
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    gen_train = torch.Generator(device=device).manual_seed(seed + 11)
    gen_eval = torch.Generator(device=device).manual_seed(seed + 9999)

    is_human = method == "human_hybrid"
    if is_human:
        model: nn.Module = HumanBasisOperatorBaseline(n_bits=n_bits, dim=dim, rank=rank, seed=seed).to(device)
    else:
        op_mode = "bilinear" if method == "ugly_bilinear" else "simple_invertible"
        model = UglyAlgebraLifter(n_bits=n_bits, dim=dim, rank=rank, op_mode=op_mode).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3 if not is_human else 3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=1e-4)

    def _encode_int(v: torch.Tensor) -> torch.Tensor:
        if is_human:
            return model.encode_int(v)  # type: ignore[operator]
        return model.encode(int_to_bits(v, n_bits))  # type: ignore[operator]

    model.train()
    for step in range(1, steps + 1):
        x, y, w = _sample_nbit_triple(batch_size, n_bits, device, gen_train)
        bx = int_to_bits(x, n_bits)
        by = int_to_bits(y, n_bits)
        b_xor = int_to_bits(op_xor(x, y, n_bits), n_bits)
        b_add = int_to_bits(op_add(x, y, n_bits), n_bits)
        b_rotr = int_to_bits(op_rotr(x, n_bits), n_bits)

        # 2-step & 3-step latent compositions without intermediate decoding!
        x_add_y = op_add(x, y, n_bits)
        r_add = op_rotr(x_add_y, n_bits)
        b_comp2 = int_to_bits(r_add, n_bits)
        b_comp3 = int_to_bits(op_xor(r_add, w, n_bits), n_bits)

        u = _encode_int(x)
        v = _encode_int(y)
        uw = _encode_int(w)

        u_xor = model.apply_xor(u, v)
        u_add = model.apply_add(u, v)
        u_rotr = model.apply_rotr(u)
        u_comp2 = model.apply_rotr(u_add)
        u_comp3 = model.apply_xor(u_comp2, uw)

        # Ground-truth latent targets (detached or direct consistency for encoder)
        with torch.no_grad() if is_human else torch.enable_grad():
            t_xor = _encode_int(op_xor(x, y, n_bits))
            t_add = _encode_int(x_add_y)
            t_rotr = _encode_int(op_rotr(x, n_bits))
            t_comp2 = _encode_int(r_add)

        loss_recon = F.binary_cross_entropy_with_logits(model.decode_logits(u), bx) + 0.5 * F.binary_cross_entropy_with_logits(
            model.decode_logits(v), by
        )
        loss_dec = (
            F.binary_cross_entropy_with_logits(model.decode_logits(u_xor), b_xor)
            + F.binary_cross_entropy_with_logits(model.decode_logits(u_add), b_add)
            + F.binary_cross_entropy_with_logits(model.decode_logits(u_rotr), b_rotr)
            + 0.7 * F.binary_cross_entropy_with_logits(model.decode_logits(u_comp2), b_comp2)
            + 0.5 * F.binary_cross_entropy_with_logits(model.decode_logits(u_comp3), b_comp3)
        )
        loss_lat = (
            F.mse_loss(u_xor, t_xor)
            + F.mse_loss(u_add, t_add)
            + F.mse_loss(u_rotr, t_rotr)
            + 0.5 * F.mse_loss(u_comp2, t_comp2)
        )
        # Prevent latent scale explosion/collapse
        loss_reg = 0.01 * ((u.pow(2).mean() - 1.0).abs()) if not is_human else u.sum() * 0.0
        loss = 1.5 * loss_recon + loss_dec + 0.5 * loss_lat + loss_reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()

    # Fresh held-out evaluation
    model.eval()
    with torch.no_grad():
        x, y, w = _sample_nbit_triple(eval_n, n_bits, device, gen_eval)
        bx = int_to_bits(x, n_bits)
        b_xor = int_to_bits(op_xor(x, y, n_bits), n_bits)
        b_add = int_to_bits(op_add(x, y, n_bits), n_bits)
        b_rotr = int_to_bits(op_rotr(x, n_bits), n_bits)
        x_add_y = op_add(x, y, n_bits)
        r_add = op_rotr(x_add_y, n_bits)
        b_comp2 = int_to_bits(r_add, n_bits)
        b_comp3 = int_to_bits(op_xor(r_add, w, n_bits), n_bits)

        u = _encode_int(x)
        v = _encode_int(y)
        uw = _encode_int(w)
        u_xor = model.apply_xor(u, v)
        u_add = model.apply_add(u, v)
        u_rotr = model.apply_rotr(u)
        u_comp2 = model.apply_rotr(u_add)
        u_comp3 = model.apply_xor(u_comp2, uw)

        t_add = _encode_int(x_add_y)
        t_xor = _encode_int(op_xor(x, y, n_bits))
        t_rotr = _encode_int(op_rotr(x, n_bits))
        rel_err_add = float((u_add - t_add).norm(dim=-1).mean() / (t_add.norm(dim=-1).mean() + 1e-8))
        rel_err_xor = float((u_xor - t_xor).norm(dim=-1).mean() / (t_xor.norm(dim=-1).mean() + 1e-8))
        rel_err_rotr = float((u_rotr - t_rotr).norm(dim=-1).mean() / (t_rotr.norm(dim=-1).mean() + 1e-8))

        lg_recon = model.decode_logits(u)
        lg_xor = model.decode_logits(u_xor)
        lg_add = model.decode_logits(u_add)
        lg_rotr = model.decode_logits(u_rotr)
        lg_c2 = model.decode_logits(u_comp2)
        lg_c3 = model.decode_logits(u_comp3)

        add_per_bit = ((lg_add > 0.0) == (b_add > 0.5)).to(torch.float32).mean(dim=0)
        half_b = max(1, n_bits // 2)
        add_low_half = float(add_per_bit[:half_b].mean().item())
        add_high_half = float(add_per_bit[half_b:].mean().item()) if n_bits > 1 else add_low_half

        recon_acc = _bit_acc(lg_recon, bx)
        xor_acc = _bit_acc(lg_xor, b_xor)
        add_acc = _bit_acc(lg_add, b_add)
        rotr_acc = _bit_acc(lg_rotr, b_rotr)
        comp2_acc = _bit_acc(lg_c2, b_comp2)
        comp3_acc = _bit_acc(lg_c3, b_comp3)
        joint_mean = (xor_acc + add_acc + rotr_acc) / 3.0

        inv_err = 0.0
        if method == "ugly_invertible":
            v_rec_add = model.op_add.inverse(u_add, cond=u)  # type: ignore[attr-defined]
            v_rec_xor = model.op_xor.inverse(u_xor, cond=u)  # type: ignore[attr-defined]
            inv_err = float(max((v - v_rec_add).abs().max().item(), (v - v_rec_xor).abs().max().item()))

    elapsed = time.perf_counter() - t0
    return {
        "n_bits": n_bits,
        "dim": dim,
        "rank": rank,
        "method": method,
        "n_params": int(n_params),
        "recon_acc": round(recon_acc, 5),
        "xor_acc": round(xor_acc, 5),
        "add_acc": round(add_acc, 5),
        "add_low_half_acc": round(add_low_half, 5),
        "add_high_half_acc": round(add_high_half, 5),
        "rotr_acc": round(rotr_acc, 5),
        "joint_mean_acc": round(joint_mean, 5),
        "comp2_acc": round(comp2_acc, 5),
        "comp3_acc": round(comp3_acc, 5),
        "latent_rel_err_add": round(rel_err_add, 5),
        "latent_rel_err_xor": round(rel_err_xor, 5),
        "latent_rel_err_rotr": round(rel_err_rotr, 5),
        "inv_roundtrip_max_err": round(inv_err, 7),
        "elapsed_sec": round(elapsed, 2),
    }


def train_and_eval_sha256_koopman(
    tag: str,
    dim: int,
    op_mode: str,
    controlled: bool,
    ctrl_cipher: bool,
    steps: int,
    batch_size: int,
    eval_n: int,
    rollout_k_max: int,
    eval_ks: List[int],
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """Train single-round G on 256-bit SHA-256 and evaluate G^k / G^{-k} up to k=16."""
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    gen_train = torch.Generator(device=device).manual_seed(seed + 17)
    gen_eval = torch.Generator(device=device).manual_seed(seed + 8888)

    prof = make_cipher_profile(ctrl=ctrl_cipher, seed=20261125 + seed)
    model = SingleRoundKoopmanFlattener(dim=dim, op_mode=op_mode, controlled=controlled).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=8e-5)

    # For autonomous mode, fix W_t to zero (or fixed word sequence); for controlled mode, fresh W_t each step
    k_tensor = torch.tensor(prof["k"], dtype=torch.int64, device=device)

    model.train()
    for step in range(1, steps + 1):
        # Progressive rollout curriculum: K grows 1 -> rollout_k_max
        frac = step / max(1, steps)
        cur_k = 1 if frac < 0.2 else (2 if frac < 0.4 else (min(4, rollout_k_max) if frac < 0.7 else rollout_k_max))

        s0 = torch.randint(0, 1 << 32, (batch_size, 8), dtype=torch.int64, device=device, generator=gen_train)
        if controlled:
            w_seq = torch.randint(0, 1 << 32, (batch_size, cur_k), dtype=torch.int64, device=device, generator=gen_train)
        else:
            w_seq = torch.zeros((batch_size, cur_k), dtype=torch.int64, device=device)

        traj = sha256_k_rounds_from_state(
            s0, w_seq, prof["k"], start_round=0, num_rounds=cur_k, s0_rots=prof["s0"], s1_rots=prof["s1"]
        )  # (B, cur_k+1, 8)
        bits_traj = [state_words_to_bits256(traj[:, r, :]) for r in range(cur_k + 1)]
        z_true = [model.encode_state(b) for b in bits_traj]

        ctrl_vecs: List[torch.Tensor | None] = []
        for r in range(cur_k):
            if controlled:
                wk = (w_seq[:, r] ^ k_tensor[r % 64]) & 0xFFFFFFFF
                ctrl_vecs.append(model.encode_ctrl(int_to_bits(wk, 32)))
            else:
                ctrl_vecs.append(None)

        g_cache = model.step_op.get_matrix() if op_mode == "linear_ortho" else None
        loss = 2.0 * F.binary_cross_entropy_with_logits(model.decode_logits(z_true[0]), bits_traj[0])

        # Forward G^k rollout from z_0
        zf = z_true[0]
        for r in range(cur_k):
            zf = model.forward_step(zf, ctrl=ctrl_vecs[r], g_cache=g_cache)
            w_r = 1.0 / (r + 1)
            loss = loss + w_r * (
                0.5 * F.mse_loss(zf, z_true[r + 1])
                + F.binary_cross_entropy_with_logits(model.decode_logits(zf), bits_traj[r + 1])
            )

        # Backward G^{-k} ("倒车回家") rollout from true endpoint z_true[cur_k] -> z_0
        zb = z_true[cur_k]
        for r in range(cur_k - 1, -1, -1):
            zb = model.inverse_step(zb, ctrl=ctrl_vecs[r], g_cache=g_cache)
        loss = loss + 1.0 * (
            0.5 * F.mse_loss(zb, z_true[0])
            + F.binary_cross_entropy_with_logits(model.decode_logits(zb), bits_traj[0])
        )
        loss = loss + 0.01 * (z_true[0].pow(2).mean() - 1.0).abs()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()

    # Held-out multi-step evaluation up to max(eval_ks)
    max_k = max(eval_ks)
    model.eval()
    per_k_metrics: Dict[str, Any] = {}
    with torch.no_grad():
        g_cache = model.step_op.get_matrix() if op_mode == "linear_ortho" else None
        s0 = torch.randint(0, 1 << 32, (eval_n, 8), dtype=torch.int64, device=device, generator=gen_eval)
        if controlled:
            w_seq = torch.randint(0, 1 << 32, (eval_n, max_k), dtype=torch.int64, device=device, generator=gen_eval)
        else:
            w_seq = torch.zeros((eval_n, max_k), dtype=torch.int64, device=device)

        traj = sha256_k_rounds_from_state(
            s0, w_seq, prof["k"], start_round=0, num_rounds=max_k, s0_rots=prof["s0"], s1_rots=prof["s1"]
        )
        bits_traj = [state_words_to_bits256(traj[:, r, :]) for r in range(max_k + 1)]
        z_true = [model.encode_state(b) for b in bits_traj]
        ctrl_vecs = []
        for r in range(max_k):
            if controlled:
                wk = (w_seq[:, r] ^ k_tensor[r % 64]) & 0xFFFFFFFF
                ctrl_vecs.append(model.encode_ctrl(int_to_bits(wk, 32)))
            else:
                ctrl_vecs.append(None)

        recon_acc = _bit_acc(model.decode_logits(z_true[0]), bits_traj[0])
        # Exact invertibility roundtrip check on step 0
        z1_test = model.forward_step(z_true[0], ctrl=ctrl_vecs[0], g_cache=g_cache)
        z0_rec_test = model.inverse_step(z1_test, ctrl=ctrl_vecs[0], g_cache=g_cache)
        inv_roundtrip_err = float((z_true[0] - z0_rec_test).abs().max().item())

        zf = z_true[0]
        fwd_lat_states = [zf]
        for r in range(max_k):
            zf = model.forward_step(zf, ctrl=ctrl_vecs[r], g_cache=g_cache)
            fwd_lat_states.append(zf)

        for k in eval_ks:
            fwd_acc = _bit_acc(model.decode_logits(fwd_lat_states[k]), bits_traj[k])
            fwd_rel_err = float(
                (fwd_lat_states[k] - z_true[k]).norm(dim=-1).mean() / (z_true[k].norm(dim=-1).mean() + 1e-8)
            )
            # Invert k steps from z_true[k] back to z_0
            zb = z_true[k]
            for r in range(k - 1, -1, -1):
                zb = model.inverse_step(zb, ctrl=ctrl_vecs[r], g_cache=g_cache)
            inv_acc = _bit_acc(model.decode_logits(zb), bits_traj[0])
            inv_rel_err = float((zb - z_true[0]).norm(dim=-1).mean() / (z_true[0].norm(dim=-1).mean() + 1e-8))
            z_fwd = (fwd_acc - 0.5) / math.sqrt(0.25 / (eval_n * 256))
            z_inv = (inv_acc - 0.5) / math.sqrt(0.25 / (eval_n * 256))
            per_k_metrics[str(k)] = {
                "fwd_bit_acc": round(fwd_acc, 5),
                "inv_bit_acc": round(inv_acc, 5),
                "fwd_latent_rel_err": round(fwd_rel_err, 5),
                "inv_latent_rel_err": round(inv_rel_err, 5),
                "z_fwd_vs_random": round(float(z_fwd), 2),
                "z_inv_vs_random": round(float(z_inv), 2),
            }

    elapsed = time.perf_counter() - t0
    return {
        "tag": tag,
        "dim": dim,
        "op_mode": op_mode,
        "controlled": controlled,
        "ctrl_cipher": ctrl_cipher,
        "n_params": int(n_params),
        "recon_acc": round(recon_acc, 5),
        "inv_roundtrip_max_err": round(inv_roundtrip_err, 7),
        "per_k": per_k_metrics,
        "elapsed_sec": round(elapsed, 2),
    }


def _fit_d95_scaling(part_a_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for method in sorted({r["method"] for r in part_a_rows}):
        by_n: Dict[int, int | None] = {}
        for n in sorted({r["n_bits"] for r in part_a_rows}):
            cands = [
                r
                for r in part_a_rows
                if r["method"] == method and r["n_bits"] == n and r["joint_mean_acc"] >= 0.95 and r["recon_acc"] >= 0.99
            ]
            by_n[str(n)] = min(r["dim"] for r in cands) if cands else None
        out[method] = by_n
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-SHA V11 P10 Ugly Algebra & Koopman Flattening")
    p.add_argument("--out_dir", type=str, default="v11_discovery/p10_ugly_algebra/results")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=20261125)
    args = p.parse_args()

    fp32_info = enforce_fp32_lock()
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "p10_results.json"

    ref_checks = verify_primitives_exact(device)
    print(f"[P10 Init] device={device} fp32={fp32_info} ref_checks={ref_checks}", flush=True)

    payload: Dict[str, Any] = {
        "experiment": "P10_Ugly_Algebra_And_Single_Round_Koopman",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "fp32_lock": fp32_info,
        "reference_verification": ref_checks,
        "status": "running",
        "part_a_scaling": [],
        "part_a_d95_summary": {},
        "part_b_32bit": [],
        "part_c_sha256_round": [],
    }
    if args.resume and res_path.exists():
        try:
            prev = json.loads(res_path.read_text(encoding="utf-8"))
            for key in ("part_a_scaling", "part_a_d95_summary", "part_b_32bit", "part_c_sha256_round"):
                if key in prev and isinstance(prev[key], type(payload[key])):
                    payload[key] = prev[key]
            print(
                f"[P10 Resume] loaded existing results: A={len(payload['part_a_scaling'])} "
                f"B={len(payload['part_b_32bit'])} C={len(payload['part_c_sha256_round'])}",
                flush=True,
            )
        except Exception as exc:
            print(f"[P10 Resume] failed to load {res_path}: {exc}; starting fresh", flush=True)
    _atomic_write_json(res_path, payload)

    if args.smoke:
        # Fast CPU smoke (<10s) validating every pipeline branch end-to-end
        r_a = train_and_eval_arx_algebra(4, 32, 32, "ugly_bilinear", 15, 64, 128, device, args.seed)
        r_b = train_and_eval_arx_algebra(32, 128, 64, "ugly_invertible", 10, 64, 128, device, args.seed)
        r_h = train_and_eval_arx_algebra(32, 128, 64, "human_hybrid", 10, 64, 128, device, args.seed)
        r_c = train_and_eval_sha256_koopman(
            "smoke_auto_ortho", 256, "linear_ortho", False, False, 10, 32, 64, 2, [1, 2, 4], device, args.seed
        )
        r_c2 = train_and_eval_sha256_koopman(
            "smoke_ctrl_inv", 256, "invertible_coupling", True, False, 10, 32, 64, 2, [1, 2, 4], device, args.seed
        )
        payload["part_a_scaling"] = [r_a]
        payload["part_a_d95_summary"] = _fit_d95_scaling([r_a])
        payload["part_b_32bit"] = [r_b, r_h]
        payload["part_c_sha256_round"] = [r_c, r_c2]
        payload["status"] = "smoke_passed"
        _atomic_write_json(res_path, payload)
        print("[P10 Smoke] ALL PASSED", flush=True)
        return

    done_a = {(r["n_bits"], r["dim"], r["method"]) for r in payload["part_a_scaling"]}
    done_b = {(r["dim"], r["method"]) for r in payload["part_b_32bit"]}
    done_c = {r["tag"] for r in payload["part_c_sha256_round"]}

    # ================= PART A: Tiny-bitwidth scaling law =================
    part_a_grid = {
        4: [32, 64],
        6: [32, 64, 128],
        8: [32, 64, 128, 256],
        10: [64, 128, 256, 512],
        12: [64, 128, 256, 512, 1024],
        16: [64, 128, 256, 512, 1024],
    }
    for nb, dims in part_a_grid.items():
        for dm in dims:
            rk = min(dm, max(64, dm // 2))
            ran_any = False
            for meth in ("ugly_bilinear", "ugly_invertible", "human_hybrid"):
                if (nb, dm, meth) in done_a:
                    continue
                row = train_and_eval_arx_algebra(
                    n_bits=nb,
                    dim=dm,
                    rank=rk,
                    method=meth,
                    steps=1000,
                    batch_size=512,
                    eval_n=4096,
                    device=device,
                    seed=args.seed + nb * 100 + dm,
                )
                payload["part_a_scaling"].append(row)
                done_a.add((nb, dm, meth))
                ran_any = True
                print(
                    f"[Part A n={nb:2d} D={dm:4d} {meth:15s}] "
                    f"recon={row['recon_acc']:.4f} xor={row['xor_acc']:.4f} add={row['add_acc']:.4f} "
                    f"rotr={row['rotr_acc']:.4f} comp3={row['comp3_acc']:.4f} ({row['elapsed_sec']}s)",
                    flush=True,
                )
            if ran_any:
                payload["part_a_d95_summary"] = _fit_d95_scaling(payload["part_a_scaling"])
                _atomic_write_json(res_path, payload)

    # ================= PART B: Full 32-bit approximation curve eps(D) =================
    part_b_dims = [128, 512, 1024, 2048, 4096]
    for dm in part_b_dims:
        rk = min(1024, dm // 2)
        bs = 512 if dm <= 2048 else 256
        steps_b = 2500
        for meth in ("ugly_bilinear", "ugly_invertible", "human_hybrid"):
            if (dm, meth) in done_b:
                continue
            row = train_and_eval_arx_algebra(
                n_bits=32,
                dim=dm,
                rank=rk,
                method=meth,
                steps=steps_b,
                batch_size=bs,
                eval_n=4096,
                device=device,
                seed=args.seed + 32000 + dm,
            )
            payload["part_b_32bit"].append(row)
            done_b.add((dm, meth))
            print(
                f"[Part B 32b D={dm:4d} {meth:15s}] "
                f"recon={row['recon_acc']:.4f} add={row['add_acc']:.4f} (low={row['add_low_half_acc']:.4f}, hi={row['add_high_half_acc']:.4f}) "
                f"xor={row['xor_acc']:.4f} comp3={row['comp3_acc']:.4f} rel_add={row['latent_rel_err_add']:.4f} ({row['elapsed_sec']}s)",
                flush=True,
            )
            _atomic_write_json(res_path, payload)

    # ================= PART C: 256-bit SHA-256 single-round -> G^k / G^{-k} =================
    eval_ks = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16]
    part_c_specs = [
        ("auto_linear_ortho_D1024_sha", 1024, "linear_ortho", False, False, 3000, 256),
        ("auto_invertible_D2048_sha", 2048, "invertible_coupling", False, False, 3000, 256),
        ("auto_invertible_D2048_rand_ctrl", 2048, "invertible_coupling", False, True, 3000, 256),
        ("ctrl_invertible_D2048_sha", 2048, "invertible_coupling", True, False, 3000, 256),
        ("ctrl_invertible_D2048_rand_ctrl", 2048, "invertible_coupling", True, True, 3000, 256),
        ("ctrl_invertible_D4096_sha", 4096, "invertible_coupling", True, False, 3000, 128),
    ]
    for tag, dm, mode, is_ctrl_w, is_rand_c, st, bs in part_c_specs:
        if tag in done_c:
            continue
        row = train_and_eval_sha256_koopman(
            tag=tag,
            dim=dm,
            op_mode=mode,
            controlled=is_ctrl_w,
            ctrl_cipher=is_rand_c,
            steps=st,
            batch_size=bs,
            eval_n=2048,
            rollout_k_max=6,
            eval_ks=eval_ks,
            device=device,
            seed=args.seed + 64000 + dm,
        )
        payload["part_c_sha256_round"].append(row)
        done_c.add(tag)
        k1 = row["per_k"]["1"]
        k4 = row["per_k"]["4"]
        k8 = row["per_k"]["8"]
        print(
            f"[Part C {tag}] recon={row['recon_acc']:.4f} rt_err={row['inv_roundtrip_max_err']:.2e} | "
            f"k1=(f:{k1['fwd_bit_acc']:.4f},i:{k1['inv_bit_acc']:.4f}) "
            f"k4=(f:{k4['fwd_bit_acc']:.4f},i:{k4['inv_bit_acc']:.4f}) "
            f"k8=(f:{k8['fwd_bit_acc']:.4f},i:{k8['inv_bit_acc']:.4f}) ({row['elapsed_sec']}s)",
            flush=True,
        )
        _atomic_write_json(res_path, payload)

    payload["status"] = "completed"
    _atomic_write_json(res_path, payload)
    print("[P10 Complete] All parts finished cleanly.", flush=True)


if __name__ == "__main__":
    main()
