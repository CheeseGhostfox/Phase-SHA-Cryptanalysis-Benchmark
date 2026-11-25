"""Four mandatory clean-room verification gates for Phase-SHA V8B on local RTX 4060."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil

import pytest
import torch

from v8a import assert_strict_fp32, configure_strict_fp32
from v8a.colab_queue_v8b import verify_v8b_param_budgets
from v8a.entropy_objective import PeriodicMarginalDensity, compute_circular_metrics
from v8a.sha256_stream import OnlineSHA256Stream, compute_round_t1_bits, sha256_digest_bits
from v8a.torus_flow import TWO_PI, circular_abs_diff, dequantize_bits
from v8a.torus_flow_v8b import (
    RegisterRollLayer,
    V8BCouplingLayer,
    V8BTorusFlow,
    _mask_words,
    _mask_words_bit_checkerboard,
)
from v8a.train_v8a import run_training


def _get_device() -> torch.device:
    configure_strict_fp32()
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for V8B gates")
    return torch.device("cuda")


def test_v8b_gate1_bijection_and_zero_logdet() -> None:
    """V8B Gate 1: Budget [20M-40M], exact bijection (<1e-5), zero log|det J| (<1e-5), and live training stability."""
    device = _get_device()

    budgets = verify_v8b_param_budgets()
    print("\n[V8B Gate1] Cloud param budgets:", {k: f"{v/1e6:.2f}M" for k, v in budgets.items()})

    flow_t1 = V8BTorusFlow(
        flow_layout="v8b_t1", rounds=5, num_layers=24, c_bit=128, c_mid=96, d_glob=360, ff_glob=480
    ).to(device)
    assert_strict_fp32(flow_t1)
    assert 20_000_000 <= flow_t1.param_count() <= 40_000_000

    flow = V8BTorusFlow(
        flow_layout="v8b_t2", rounds=3, c_bit=96, c_mid=64, d_glob=224, ff_glob=288
    ).to(device)
    assert_strict_fp32(flow)
    with torch.no_grad():
        for layer in flow.layers:
            if isinstance(layer, V8BCouplingLayer):
                layer.conditioner.arx_out.weight.normal_(0.0, 0.05)
                layer.conditioner.glob_out.weight.normal_(0.0, 0.02)
    flow.eval()

    gen = torch.Generator(device=device).manual_seed(801)
    theta = torch.rand((256, 256), device=device, generator=gen, dtype=torch.float32) * TWO_PI
    theta[0, :64] = 1e-5
    theta[0, 64:128] = TWO_PI - 1e-5

    with torch.no_grad():
        y = flow(theta)
        theta_rec = flow.inverse(y)
        err_fwd_inv = float(circular_abs_diff(theta_rec, theta).max().item())
        y_rec = flow(flow.inverse(theta))
        err_inv_fwd = float(circular_abs_diff(y_rec, theta).max().item())
        shift_rms = float(circular_abs_diff(y, theta).pow(2).mean().sqrt().item())

    print(
        f"[V8B Gate1] Perturbed bijection err: fwd->inv={err_fwd_inv:.2e}, "
        f"inv->fwd={err_inv_fwd:.2e} (RMS disp={shift_rms:.3f} rad)"
    )
    assert shift_rms > 0.5
    assert err_fwd_inv < 1e-5
    assert err_inv_fwd < 1e-5

    for idx in range(3):
        pt = theta[idx]
        jac = flow.compute_tangent_jacobian(pt)
        sign, logabsdet = torch.linalg.slogdet(jac.double())
        print(f"[V8B Gate1] Sample {idx}: sign={float(sign.item()):+.1f}, log|det J|={float(logabsdet.item()):.3e}")
        assert float(sign.item()) > 0.0
        assert abs(float(logabsdet.item())) < 1e-5

        if idx == 0:
            eps = 1e-3
            with torch.no_grad():
                for col in (0, 31, 65, 128, 177, 255):
                    e_vec = torch.zeros_like(pt)
                    e_vec[col] = eps
                    yp = flow((pt + e_vec).unsqueeze(0)).squeeze(0)
                    ym = flow((pt - e_vec).unsqueeze(0)).squeeze(0)
                    fd_col = (torch.remainder(yp - ym + math.pi, TWO_PI) - math.pi) / (2.0 * eps)
                    col_err = float((fd_col - jac[:, col]).abs().max().item())
                    assert col_err < 5e-3, f"FD Jacobian mismatch on col {col}: {col_err}"

    # Verify bij < 1e-5 holds after 100 steps of live training with lambda_lip=2e-3
    density = PeriodicMarginalDensity(256, 12).to(device)
    flow.train()
    opt = torch.optim.AdamW(
        [
            {"params": flow.parameters(), "lr": 5e-4, "weight_decay": 1e-5},
            {"params": density.parameters(), "lr": 0.03, "weight_decay": 0.0},
        ]
    )
    stream = OnlineSHA256Stream(k_bits=32, rounds=2, n_eval=512, device=device)
    for _ in range(100):
        th0 = dequantize_bits(stream.sample_train_batch(512), generator=stream.train_gen)
        tho, lip = flow(th0, return_lip=True, lip_sigma=0.15)
        loss = density(tho)["loss_marg"] + 2e-3 * lip
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 2.0)
        opt.step()

    flow.eval()
    with torch.no_grad():
        th_check = dequantize_bits(stream.eval_H_bits, generator=stream.eval_gen)
        bij_trained = float(circular_abs_diff(flow.inverse(flow(th_check)), th_check).max().item())
    print(f"[V8B Gate1] Post-100-step live-trained bijection err = {bij_trained:.2e}")
    assert bij_trained < 1e-5


def test_v8b_gate2_liouville_white_noise_anti_cheat() -> None:
    """V8B Gate 2: 300 steps on pure 256-bit white noise MUST extract < 0.5 bits, max_R1 < 0.08, bij < 1e-5."""
    device = _get_device()
    torch.manual_seed(802)

    stream = OnlineSHA256Stream(k_bits=32, rounds=1, n_eval=4096, device=device)
    flow = V8BTorusFlow(
        flow_layout="v8b_t1",
        num_layers=12,
        c_bit=64,
        c_mid=48,
        d_glob=192,
        ff_glob=256,
        num_arx_blocks=2,
    ).to(device)
    density = PeriodicMarginalDensity(256, 12).to(device)
    assert_strict_fp32(flow)
    assert_strict_fp32(density)

    opt = torch.optim.AdamW(
        [
            {"params": flow.parameters(), "lr": 5e-4, "weight_decay": 1e-5},
            {"params": density.parameters(), "lr": 0.03, "weight_decay": 0.0},
        ]
    )

    flow.train()
    density.train()
    for _ in range(300):
        bits = stream.sample_white_noise_batch(1024)
        th0 = dequantize_bits(bits, deterministic=False, generator=stream.train_gen)
        tho, lip = flow(th0, return_lip=True, lip_sigma=0.15)
        out = density(tho)
        loss = out["loss_marg"] + 2e-3 * lip
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 2.0)
        opt.step()

    flow.eval()
    density.eval()
    eval_gen = torch.Generator(device=device).manual_seed(99982)
    eval_bits = stream.sample_white_noise_batch(4096, generator=eval_gen)
    ext_sum = 0.0
    r1_acc = torch.zeros(256, device=device)
    r2_acc = torch.zeros(256, device=device)
    bij_err = 0.0
    with torch.no_grad():
        for d in range(4):
            th0 = dequantize_bits(eval_bits, deterministic=False, generator=eval_gen)
            tho = flow(th0)
            if d == 0:
                bij_err = float(circular_abs_diff(flow.inverse(tho[:512]), th0[:512]).max().item())
            ext_sum += float(density(tho)["extracted_bits"].item())
            cm = compute_circular_metrics(tho)
            r1_acc += cm["R1"]
            r2_acc += cm["R2"]

    extracted_bits = ext_sum / 4.0
    max_r1 = float((r1_acc / 4.0).max().item())
    max_r2 = float((r2_acc / 4.0).max().item())
    norm_err = density.verify_normalization()

    print(
        f"\n[V8B Gate2] White-noise holdout: extracted_bits={extracted_bits:.4f}, "
        f"max_R1={max_r1:.4f}, max_R2={max_r2:.4f}, norm_err={norm_err:.2e}, bij={bij_err:.2e}"
    )
    assert extracted_bits < 0.5
    assert max_r1 < 0.08
    assert max_r2 < 0.08
    assert norm_err < 1e-5
    assert bij_err < 1e-5


def test_v8b_gate3_single_round_transport_smoke() -> None:
    """V8B Gate 3: Half-round S_5 -> S_5^{half} -> S_4 sub-flow breaks d4>0.62, h4>0.62, tot>0.91, bij<1e-5."""
    device = _get_device()
    torch.manual_seed(803)

    # Exact V8B half-round -> full-round reversible circuit (5 coupling layers + Roll(-1))
    cell = torch.nn.ModuleList(
        [
            V8BCouplingLayer(_mask_words((0,)), c_bit=128, c_mid=96, d_glob=256, ff_glob=384),
            V8BCouplingLayer(_mask_words_bit_checkerboard((0,), 0), c_bit=128, c_mid=96, d_glob=256, ff_glob=384),
            V8BCouplingLayer(_mask_words((4,)), c_bit=128, c_mid=96, d_glob=256, ff_glob=384),
            V8BCouplingLayer(_mask_words((0,)), c_bit=128, c_mid=96, d_glob=256, ff_glob=384),
            V8BCouplingLayer(_mask_words_bit_checkerboard((0, 4), 1), c_bit=128, c_mid=96, d_glob=256, ff_glob=384),
            RegisterRollLayer(shift=-1),
        ]
    ).to(device)
    assert_strict_fp32(cell)

    stream = OnlineSHA256Stream(k_bits=32, rounds=5, n_eval=4096, device=device)
    eval_states = sha256_digest_bits(stream.eval_free_words, k_bits=32, rounds=5, return_states=True)[1]
    s5_eval = eval_states[4]
    s4_eval = eval_states[3]

    opt = torch.optim.AdamW(cell.parameters(), lr=3e-3, weight_decay=1e-5)
    total_steps = 650
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=3e-4)
    bit_pos = torch.arange(32, device=device, dtype=torch.float32)

    for step in range(1, total_steps + 1):
        cell.train()
        _, st_bits = stream.sample_train_batch(1024, return_states=True)
        s5_b, s4_b = st_bits[4], st_bits[3]
        t1_b = compute_round_t1_bits(s5_b)

        prog = (step - 1) / max(1, total_steps - 1)
        eps = 0.8 * max(0.0, 1.0 - 2.0 * prog)
        temp = 2.0 * max(0.0, 1.0 - 2.0 * prog)
        w_bit = torch.exp(temp * (bit_pos - 31.0))
        w_bit = w_bit / w_bit.mean()

        x = dequantize_bits(s5_b, deterministic=False, generator=stream.train_gen).to(torch.float64).view(-1, 8, 32)
        lip_sum = torch.zeros((), device=device)
        x_half = x
        for oi, op in enumerate(cell):
            if isinstance(op, V8BCouplingLayer):
                lip_sum = lip_sum + op.compute_local_lipschitz(x[:64].to(torch.float32).detach(), sigma=0.15)
            x = op(x)
            if oi == 1:
                x_half = x

        x_half32 = x_half.to(torch.float32)
        x_out32 = x.to(torch.float32)
        loss_half = ((1.0 - torch.cos(x_half32[:, 0, :] - math.pi * t1_b - eps)) * w_bit).sum(dim=-1).mean()
        s4_w = s4_b.view(-1, 8, 32)
        loss_dh = ((1.0 - torch.cos(x_out32[:, (3, 7), :] - math.pi * s4_w[:, (3, 7), :] - eps)) * w_bit.unsqueeze(0)).sum(dim=(1, 2)).mean()
        loss = loss_half + loss_dh + 2e-3 * (lip_sum / 5.0)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cell.parameters(), 2.0)
        opt.step()
        sched.step()

        if step % 150 == 0 or step == total_steps:
            with torch.no_grad():
                t1_acc = float(((torch.cos(x_half32[:, 0, :]) < 0.0).to(torch.float32) == t1_b).float().mean().item())
                pred = (torch.cos(x_out32) < 0.0).to(torch.float32)
                pw = (pred == s4_w).float().mean(dim=(0, 2)).tolist()
                tot = float((pred == s4_w).float().mean().item())
            print(
                f"  [V8B Gate3 step {step:3d}/{total_steps}] t1={t1_acc:.4f}, tot={tot:.4f}, "
                f"d4={pw[3]:.4f}, h4={pw[7]:.4f}"
            )

    cell.eval()
    with torch.no_grad():
        x_in = dequantize_bits(s5_eval, deterministic=False, generator=stream.eval_gen).to(torch.float64).view(-1, 8, 32)
        x = x_in
        for op in cell:
            x = op(x)
        x_out32 = x.to(torch.float32)
        pred_eval = (torch.cos(x_out32) < 0.0).to(torch.float32)
        tgt_eval = s4_eval.view(-1, 8, 32)
        pw_eval = (pred_eval == tgt_eval).float().mean(dim=(0, 2)).tolist()
        eval_bit_acc = float((pred_eval == tgt_eval).float().mean().item())
        x_rec = x
        for op in reversed(cell):
            x_rec = op.inverse(x_rec)
        bij_err = float(
            circular_abs_diff(x_rec.to(torch.float32).reshape(-1, 256), x_in.to(torch.float32).reshape(-1, 256)).max().item()
        )

    print(
        f"[V8B Gate3] Holdout S5->S4: eval_bit_acc={eval_bit_acc:.4f}, d4={pw_eval[3]:.4f}, "
        f"h4={pw_eval[7]:.4f}, bij_err={bij_err:.2e}, per_word={[round(v, 4) for v in pw_eval]}"
    )
    assert eval_bit_acc > 0.91
    assert pw_eval[3] > 0.62
    assert pw_eval[7] > 0.62
    assert bij_err < 1e-5


def test_v8b_gate4_r2_t1_and_t2_e2e_smoke() -> None:
    """V8B Gate 4: End-to-end local smoke verification for both V8B-T1 and V8B-T2 at R=2."""
    _get_device()
    tmp_root = Path(r"g:\SHA-LM\v8a\_tmp_test_v8b")
    shutil.rmtree(tmp_root, ignore_errors=True)
    try:
        args_t1 = argparse.Namespace(
            arch="v8b_t1",
            mode="unsupervised",
            rounds=2,
            k_bits=32,
            steps=350,
            transport_steps=0,
            batch_size=512,
            num_layers=12,
            hidden_dim=192,
            ff_dim=256,
            num_blocks=2,
            c_bit=64,
            c_mid=48,
            d_glob=192,
            ff_glob=256,
            num_arx_blocks=2,
            lambda_lip=2e-3,
            lip_sigma=0.15,
            use_checkpoint=False,
            init_ckpt="",
            no_attn=False,
            num_components=12,
            lr_phi=8e-4,
            lr_psi=0.03,
            weight_decay=1e-5,
            grad_clip=2.0,
            eval_every=175,
            probe_every=350,
            eval_size=2048,
            probe_train_size=8192,
            run_dir=str(tmp_root / "t1_r2"),
            run_name="smoke_v8b_t1_r2",
            seed=811,
        )
        res_t1 = run_training(args_t1)
        print(
            f"\n[V8B Gate4 T1] bits={res_t1['eval_extracted_bits']:.2f} "
            f"(gain=+{res_t1['flow_gain_bits']:.2f}), bij={res_t1['bijection_max_err']:.2e}, "
            f"w_acc={res_t1['w_bit_acc']:.4f}, p@1={res_t1['exact_pass_1']:.4f}"
        )
        assert res_t1["bijection_max_err"] < 5e-5
        assert res_t1["eval_extracted_bits"] > 50.0
        assert res_t1["flow_gain_bits"] > 5.0
        assert res_t1["exact_pass_1"] > 0.0

        args_t2 = argparse.Namespace(
            arch="v8b_t2",
            mode="round_transport",
            rounds=2,
            k_bits=32,
            steps=400,
            transport_steps=250,
            batch_size=512,
            num_layers=15,
            hidden_dim=192,
            ff_dim=256,
            num_blocks=2,
            c_bit=64,
            c_mid=48,
            d_glob=192,
            ff_glob=256,
            num_arx_blocks=2,
            lambda_lip=2e-3,
            lip_sigma=0.15,
            use_checkpoint=False,
            init_ckpt="",
            no_attn=False,
            num_components=12,
            lr_phi=1e-3,
            lr_psi=0.03,
            weight_decay=1e-5,
            grad_clip=2.0,
            eval_every=200,
            probe_every=400,
            eval_size=2048,
            probe_train_size=8192,
            run_dir=str(tmp_root / "t2_r2"),
            run_name="smoke_v8b_t2_r2",
            seed=812,
        )
        res_t2 = run_training(args_t2)
        print(
            f"[V8B Gate4 T2] bits={res_t2['eval_extracted_bits']:.2f} "
            f"(gain=+{res_t2['flow_gain_bits']:.2f}), bij={res_t2['bijection_max_err']:.2e}, "
            f"w_acc={res_t2['w_bit_acc']:.4f}, p@1={res_t2['exact_pass_1']:.4f}, "
            f"tf={res_t2['tf_bit_acc']:.4f}, roll={res_t2['rollout_bit_acc']:.4f}"
        )
        assert res_t2["bijection_max_err"] < 5e-5
        assert res_t2["eval_extracted_bits"] > 50.0
        assert res_t2["flow_gain_bits"] > 5.0
        assert res_t2["exact_pass_1"] > 0.0
        assert (tmp_root / "t2_r2" / "telemetry.json").exists()
        assert (tmp_root / "t2_r2" / "metrics.jsonl").exists()
        assert (tmp_root / "t2_r2" / "summary.json").exists()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
