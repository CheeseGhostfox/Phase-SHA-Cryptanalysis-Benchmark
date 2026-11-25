"""Four mandatory clean-room verification gates for Phase-SHA V8A on local RTX 4060."""

from __future__ import annotations

import hashlib
import math
import struct

import pytest
import torch

from v8a import assert_strict_fp32, configure_strict_fp32
from v8a.entropy_objective import (
    PeriodicMarginalDensity,
    compute_circular_metrics,
    estimate_raw_input_baseline,
)
from v8a.preimage_probe import evaluate_frozen_preimage_probe
from v8a.sha256_stream import (
    IV_TUPLE,
    MASK32,
    OnlineSHA256Stream,
    bits_to_words,
    build_padded_block_from_bytes,
    sha256_compress_words,
    sha256_digest_bits,
    sha256_reduced_reference_bytes,
    words_to_bits,
    words_to_bytes,
)
from v8a.torus_flow import (
    TWO_PI,
    TorusFlow,
    circular_abs_diff,
    dequantize_bits,
)


def _get_device() -> torch.device:
    configure_strict_fp32()
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for V8A gates")
    return torch.device("cuda")


def test_gate1_sha256_hashlib_consistency() -> None:
    """Gate 1: 100% bit-exact match with hashlib.sha256, reduced-round reference, and disjoint stream."""
    device = _get_device()

    # 1. Fixed FIPS vectors vs hashlib
    for msg in [b"", b"abc", b"a" * 55]:
        expected = hashlib.sha256(msg).digest()
        w16 = build_padded_block_from_bytes(msg, device=device)
        out_words = sha256_compress_words(w16, rounds=64, return_states=False)
        out_bytes = words_to_bytes(out_words)[0]
        assert out_bytes == expected, f"Mismatch on msg={msg!r}"
        assert sha256_reduced_reference_bytes(msg, rounds=64) == expected

    # 2. Random k=32 and k=64 batches at R=64 vs hashlib
    gen = torch.Generator(device=device).manual_seed(101)
    for k_bits in (32, 64):
        k_words = k_bits // 32
        words = torch.randint(0, 1 << 32, (1024, k_words), device=device, generator=gen, dtype=torch.int64)
        h_bits = sha256_digest_bits(words, k_bits=k_bits, rounds=64)
        h_bytes = words_to_bytes(bits_to_words(h_bits))
        msg_list = words_to_bytes(words)
        for m_b, got_b in zip(msg_list, h_bytes):
            assert got_b == hashlib.sha256(m_b).digest()

    # 3. Reduced rounds R in [1,2,4,8,16,32] vs pure-Python reference & intermediate states
    iv_t = torch.tensor(IV_TUPLE, dtype=torch.int64, device=device)
    sample_w = torch.randint(0, 1 << 32, (64, 1), device=device, generator=gen, dtype=torch.int64)
    sample_msgs = words_to_bytes(sample_w)
    for r in (1, 2, 4, 8, 16, 32):
        h_bits, states_bits = sha256_digest_bits(sample_w, k_bits=32, rounds=r, return_states=True)
        assert states_bits.shape == (r, 64, 256)
        last_state_words = bits_to_words(states_bits[-1])
        reconstructed_h = (last_state_words + iv_t.unsqueeze(0)) & MASK32
        assert torch.equal(reconstructed_h, bits_to_words(h_bits))
        h_bytes = words_to_bytes(bits_to_words(h_bits))
        for m_b, got_b in zip(sample_msgs, h_bytes):
            assert got_b == sha256_reduced_reference_bytes(m_b, rounds=r)

    # 4. Stream zero-overlap verification for k=32 and k=64
    for k_bits in (32, 64):
        stream = OnlineSHA256Stream(k_bits=k_bits, rounds=4, n_eval=4096, device=device)
        train_w = stream.sample_disjoint_words(16384)
        probe_w, _, _ = stream.sample_probe_train_set(8192)
        assert not bool(stream._in_eval_set(stream._pack_keys(train_w)).any())
        assert not bool(stream._in_eval_set(stream._pack_keys(probe_w)).any())


def test_gate2_torus_flow_bijection_and_zero_logdet() -> None:
    """Gate 2: Exact bijection (<1e-5 circular error), zero log|det J| (<1e-5), and normalized density."""
    device = _get_device()

    # Check default A100 parameter budget sits cleanly in 15M - 40M
    flow_full = TorusFlow(num_layers=24, hidden_dim=384, ff_dim=512, num_blocks=2, use_attn=True).to(device)
    assert_strict_fp32(flow_full)
    pcount = flow_full.param_count()
    print(f"\n[Gate2] Default 24-layer TorusFlow param_count={pcount:,}")
    assert 15_000_000 <= pcount <= 40_000_000

    # Non-identity perturbed 12-layer flow for strict bijection & Jacobian checks
    # Note: out_proj std=0.03 + skip_linear std=0.015 gives non-trivial >0.5 rad shifts without chaotic Lyapunov blow-up
    flow = TorusFlow(num_layers=12, hidden_dim=256, ff_dim=384, num_blocks=2, use_attn=True).to(device)
    assert_strict_fp32(flow)
    with torch.no_grad():
        for layer in flow.layers:
            layer.conditioner.out_proj.weight.normal_(mean=0.0, std=0.03)
            layer.conditioner.out_proj.bias.normal_(mean=0.0, std=0.03)
            layer.conditioner.skip_linear.weight.normal_(mean=0.0, std=0.015)
            layer.conditioner.skip_linear.bias.normal_(mean=0.0, std=0.015)
    flow.eval()

    gen = torch.Generator(device=device).manual_seed(202)
    theta = torch.rand((256, 256), device=device, generator=gen, dtype=torch.float32) * TWO_PI
    # Include tricky boundary angles near 0 and 2*pi
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
        f"[Gate2] Bijection circular error fwd->inv={err_fwd_inv:.2e}, "
        f"inv->fwd={err_inv_fwd:.2e} (non-trivial RMS displacement={shift_rms:.3f} rad)"
    )
    assert shift_rms > 0.5
    assert err_fwd_inv < 1e-5
    assert err_inv_fwd < 1e-5

    # Exact Riemannian tangent Jacobian slogdet + finite-difference column verification
    for idx in range(4):
        pt = theta[idx]
        jac = flow.compute_tangent_jacobian(pt)
        sign, logabsdet = torch.linalg.slogdet(jac.double())
        print(f"[Gate2] Sample {idx}: sign={float(sign.item()):+.1f}, log|det J|={float(logabsdet.item()):.3e}")
        assert float(sign.item()) > 0.0
        assert abs(float(logabsdet.item())) < 1e-5

        if idx == 0:
            eps = 1e-3
            with torch.no_grad():
                for col in (0, 17, 63, 128, 177, 255):
                    e_vec = torch.zeros_like(pt)
                    e_vec[col] = eps
                    y_plus = flow((pt + e_vec).unsqueeze(0)).squeeze(0)
                    y_minus = flow((pt - e_vec).unsqueeze(0)).squeeze(0)
                    diff = torch.remainder(y_plus - y_minus + math.pi, TWO_PI) - math.pi
                    fd_col = diff / (2.0 * eps)
                    col_err = float((fd_col - jac[:, col]).abs().max().item())
                    assert col_err < 5e-3, f"FD Jacobian mismatch on col {col}: {col_err}"

    density = PeriodicMarginalDensity(256, 12).to(device)
    norm_err = density.verify_normalization()
    print(f"[Gate2] Density integral max |I - 1| = {norm_err:.3e}")
    assert norm_err < 1e-5


def test_gate3_liouville_white_noise_anti_cheat() -> None:
    """Gate 3: 300 steps on pure 256-bit white noise MUST extract < 0.5 bits and max(R_i) < 0.08."""
    device = _get_device()
    torch.manual_seed(303)

    stream = OnlineSHA256Stream(k_bits=32, rounds=1, n_eval=4096, device=device)
    flow = TorusFlow(num_layers=12, hidden_dim=256, ff_dim=384, num_blocks=2, use_attn=True).to(device)
    density = PeriodicMarginalDensity(num_channels=256, num_components=12).to(device)
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
    for step in range(1, 301):
        bits = stream.sample_white_noise_batch(1024)
        theta_0 = dequantize_bits(bits, deterministic=False, generator=stream.train_gen)
        theta_out = flow(theta_0)
        out = density(theta_out)
        opt.zero_grad()
        out["loss_marg"].backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step()

    # Unseen holdout white-noise evaluation averaged over 4 fresh dequantization draws
    flow.eval()
    density.eval()
    eval_gen = torch.Generator(device=device).manual_seed(99991)
    eval_bits = stream.sample_white_noise_batch(4096, generator=eval_gen)

    ext_sum = 0.0
    r1_acc = torch.zeros(256, device=device)
    r2_acc = torch.zeros(256, device=device)
    draws = 4
    with torch.no_grad():
        for _ in range(draws):
            th0 = dequantize_bits(eval_bits, deterministic=False, generator=eval_gen)
            th_out = flow(th0)
            out = density(th_out)
            ext_sum += float(out["extracted_bits"].item())
            cm = compute_circular_metrics(th_out)
            r1_acc += cm["R1"]
            r2_acc += cm["R2"]

    extracted_bits = ext_sum / draws
    max_r1 = float((r1_acc / draws).max().item())
    max_r2 = float((r2_acc / draws).max().item())
    norm_err = density.verify_normalization()

    print(
        f"\n[Gate3] White-noise holdout: extracted_bits={extracted_bits:.4f}, "
        f"max_R1={max_r1:.4f}, max_R2={max_r2:.4f}, norm_err={norm_err:.2e}"
    )
    assert extracted_bits < 0.5
    assert max_r1 < 0.08
    assert max_r2 < 0.08
    assert norm_err < 1e-5


def _run_unsupervised_smoke(rounds: int, steps: int, seed: int) -> dict:
    device = _get_device()
    torch.manual_seed(seed)

    stream = OnlineSHA256Stream(
        k_bits=32, rounds=rounds, n_eval=4096, eval_seed=20260801 + rounds, train_seed=20260901 + rounds, device=device
    )
    train_base = stream.sample_train_batch(4096)
    baseline = estimate_raw_input_baseline(
        eval_h_bits=stream.eval_H_bits,
        train_h_bits=train_base,
        num_components=12,
        steps=200,
        lr=0.03,
        seed=seed + 10,
    )

    flow = TorusFlow(num_layers=12, hidden_dim=256, ff_dim=384, num_blocks=2, use_attn=True).to(device)
    density = PeriodicMarginalDensity(num_channels=256, num_components=12, seed=seed).to(device)
    assert_strict_fp32(flow)

    opt = torch.optim.AdamW(
        [
            {"params": flow.parameters(), "lr": 8e-4, "weight_decay": 0.0},
            {"params": density.parameters(), "lr": 0.03, "weight_decay": 0.0},
        ]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=8e-5)

    flow.train()
    density.train()
    for step in range(1, steps + 1):
        h_bits = stream.sample_train_batch(1024)
        theta_0 = dequantize_bits(h_bits, deterministic=False, generator=stream.train_gen)
        theta_out = flow(theta_0)
        out = density(theta_out)
        opt.zero_grad()
        out["loss_marg"].backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 2.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == steps:
            cm = compute_circular_metrics(theta_out.detach())
            print(
                f"  [R={rounds} step {step}/{steps}] train_bits={float(out['extracted_bits'].item()):.2f} "
                f"ch55={cm['collapsed_ch_55']}"
            )

    flow.eval()
    density.eval()
    eval_gen = torch.Generator(device=device).manual_seed(seed + 555)
    ext_sum = 0.0
    r1_acc = torch.zeros(256, device=device)
    draws = 4
    with torch.no_grad():
        for _ in range(draws):
            th0 = dequantize_bits(stream.eval_H_bits, deterministic=False, generator=eval_gen)
            th_out = flow(th0)
            ext_sum += float(density(th_out)["extracted_bits"].item())
            r1_acc += compute_circular_metrics(th_out)["R1"]

    eval_bits = ext_sum / draws
    r1_avg = r1_acc / draws
    ch55 = int((r1_avg > 0.55).sum().item())
    ch35 = int((r1_avg > 0.35).sum().item())
    flow_gain = eval_bits - baseline["baseline_input_bits"]

    probe_res = evaluate_frozen_preimage_probe(
        flow=flow,
        stream=stream,
        probe_train_size=16384,
        probe_steps=300,
        batch_size=512,
        seed=seed + 777,
    )

    res = {
        "rounds": rounds,
        "baseline_input_bits": baseline["baseline_input_bits"],
        "baseline_ch55": baseline["baseline_collapsed_ch_55"],
        "eval_extracted_bits": eval_bits,
        "flow_gain_bits": flow_gain,
        "collapsed_ch_35": ch35,
        "collapsed_ch_55": ch55,
        **probe_res,
    }
    print(
        f"[Gate4 R={rounds}] base={res['baseline_input_bits']:.2f}b (ch55={res['baseline_ch55']}) -> "
        f"eval={eval_bits:.2f}b (gain=+{flow_gain:.2f}b, ch35={ch35}, ch55={ch55}) | "
        f"w_acc={res['w_bit_acc']:.4f} (z={res['z_score']:.1f}), "
        f"pass@1={res['exact_pass_1']:.4f} ({res['exact_pass_1_count']}/{res['n_eval']}), "
        f"pass@64={res['exact_pass_64']:.4f}"
    )
    return res


def test_gate4_live_unscrambling_r1_and_r2() -> None:
    """Gate 4: Pure unsupervised R=1 & R=2 on RTX 4060 extract >50b, leap ch55, and hit exact_pass@1 > 0."""
    res_r1 = _run_unsupervised_smoke(rounds=1, steps=500, seed=401)
    assert res_r1["eval_extracted_bits"] > 50.0
    assert res_r1["flow_gain_bits"] > 7.0
    assert res_r1["collapsed_ch_55"] >= 197
    assert res_r1["collapsed_ch_55"] > res_r1["baseline_ch55"] + 4
    assert res_r1["exact_pass_1"] > 0.0

    res_r2 = _run_unsupervised_smoke(rounds=2, steps=600, seed=402)
    assert res_r2["eval_extracted_bits"] > 50.0
    assert res_r2["flow_gain_bits"] > 7.0
    assert res_r2["collapsed_ch_55"] >= 133
    assert res_r2["collapsed_ch_55"] > res_r2["baseline_ch55"] + 4
    assert res_r2["exact_pass_1"] > 0.0
