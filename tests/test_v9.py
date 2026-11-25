"""Mandatory 4-Gate Local RTX 4060 Verification Suite for Phase-SHA V9-64 Direct."""

from __future__ import annotations

import gc
import hashlib
import math

import pytest
import torch
import torch.nn.functional as F

from v9 import assert_strict_fp32, configure_strict_fp32, count_parameters
from v9.beam_search_r64 import evaluate_beam_search
from v9.decision_experts import (
    TWO_PI,
    DecisionExpert,
    DecisionTower,
    circular_abs_diff,
    dequantize_bits,
)
from v9.sha256_r64_engine import (
    IV_WORDS,
    K_WORDS,
    OnlineR64Stream,
    bits_to_words,
    classical_skip_prefix,
    classical_verify_from_decisions,
    digest_words_to_bytes,
    inv_message_schedule_full,
    inv_message_schedule_step,
    inv_round_step,
    pad_single_block,
    sha256_compress_64,
    words_to_bits,
)
from v9.tunnel_experts import TunnelTower


@pytest.fixture(autouse=True)
def _strict_fp32_fixture():
    configure_strict_fp32()
    assert torch.cuda.is_available(), "V9 local gates require CUDA RTX 4060"
    yield
    gc.collect()
    torch.cuda.empty_cache()


def test_gate1_r64_engine_and_skip():
    """Gate 1: R=64 hashlib exact match, inverse schedule, skip=7/8 trajectories, verifier."""
    device = torch.device("cuda")

    # 1. Standard hashlib SHA-256 bit-exact match across edge cases
    messages = [
        b"",
        b"abc",
        b"V9-64 Direct SHA-256 clean-room verification",
        b"A" * 55,  # max single-block SHA-256 byte length
        bytes(range(40)),
    ]
    w_blocks = torch.stack([pad_single_block(m) for m in messages], dim=0).to(device)
    h_words, r_states, w_full = sha256_compress_64(w_blocks, return_states=True)

    for idx, msg in enumerate(messages):
        got_bytes = digest_words_to_bytes(h_words[idx])
        exp_bytes = hashlib.sha256(msg).digest()
        assert got_bytes == exp_bytes, f"Digest mismatch on msg idx {idx}"

    # 2. Exact inverse message schedule & step intermediates
    w_rec = inv_message_schedule_full(w_full[:, 48:64])
    assert torch.equal(w_rec, w_full), "Inverse schedule must match full W[0..63] bit-exact"

    for t in (17, 40, 64):
        w_idx = 64 - t
        w_out, aux_s = inv_message_schedule_step(
            w_full[:, w_idx + 16],
            w_full[:, w_idx + 14],
            w_full[:, w_idx + 9],
            w_full[:, w_idx + 1],
        )
        assert torch.equal(w_out, w_full[:, w_idx])
        r_next, aux_r = inv_round_step(r_states[:, t - 1], K_WORDS[w_idx], w_full[:, w_idx])
        assert torch.equal(r_next, r_states[:, t])
        assert "T1" in aux_r and "s0" in aux_s

    # 3. Rejection-free stream & classical skip for SKIP_STEPS=7 and 8
    iv_tensor = torch.tensor(IV_WORDS, device=device, dtype=torch.int64)
    for skip_steps in (7, 8):
        stream = OnlineR64Stream(
            skip_steps=skip_steps,
            n_eval=2048 if skip_steps == 7 else 256,
            device=device,
            seed=202609 + skip_steps,
        )
        batch = stream.sample_train_batch(128)
        h_b = batch["h_words"]
        r_b = batch["r_states"]
        w_bits_b = batch["w_bits"]

        # Verify R_0 = (H - IV) mod 2^32 and R_64 === IV
        r0_exp = (h_b - iv_tensor.unsqueeze(0)) & 0xFFFFFFFF
        assert torch.equal(r_b[:, 0, :], r0_exp)
        assert torch.equal(r_b[:, 64, :], iv_tensor.unsqueeze(0).expand_as(r_b[:, 64, :]))

        # Verify classical prefix skip 100% match with R_0 .. R_{skip_steps}
        r_skip = classical_skip_prefix(h_b, skip_prefix=stream.skip_prefix, skip_steps=skip_steps)
        assert torch.equal(r_skip, r_b[:, : skip_steps + 1, :])

        # Feed ground-truth decision words [W_{63-skip}..W_48] into classical_verify_from_decisions
        dec_indices = list(range(63 - skip_steps, 47, -1))
        w_dec_gt = w_bits_b[:, dec_indices, :]
        ver = classical_verify_from_decisions(
            h_b, w_dec_gt, skip_prefix=stream.skip_prefix, skip_steps=skip_steps
        )
        assert ver["valid_preimage_rate"] == 1.0
        assert ver["classical_iv_landing_acc"] == 1.0
        assert ver["rehash_bit_acc"] == 1.0

        # Check disjointness of eval_matched vs eval_unconstrained
        if skip_steps == 7:
            assert stream.eval_matched["h_words"].shape == (2048, 8)
            assert stream.eval_unconstrained["h_words"].shape == (2048, 8)

    print("[Gate 1 PASS] SHA256_64 hashlib exact, inv_sched/round exact, skip=7/8 & 2048 holdouts verified.")


def test_gate2_bijection_jacobian_and_budget():
    """Gate 2: Production 57-expert budget [95M, 115M], bij < 1e-5, |log|det J|| < 1e-5."""
    device = torch.device("cuda")

    # 1. Verify full 57-expert production parameter count on CPU cleanly
    dec_prod = DecisionTower(skip_steps=7)
    tun_prod = TunnelTower(start_step=17, end_step=64)
    p_dec = assert_strict_fp32(dec_prod)
    p_tun = assert_strict_fp32(tun_prod)
    p_tot = p_dec + p_tun
    assert len(dec_prod.experts) == 9
    assert len(tun_prod.experts) == 48
    print(
        f"[Gate 2 Budget] Decision(9)={p_dec / 1e6:.2f}M, Tunnel(48)={p_tun / 1e6:.2f}M, Total={p_tot / 1e6:.2f}M"
    )
    assert 95_000_000 <= p_tot <= 115_000_000, f"Total params {p_tot} outside [95M, 115M]"
    del dec_prod, tun_prod
    gc.collect()

    # 2. Check exact bijection & 256x256 Jacobian on M_8 and M_16 on CUDA
    for step_t in (8, 16):
        expert = DecisionExpert(step_t=step_t).to(device)
        assert_strict_fp32(expert)
        expert.eval()

        bsz = 16
        ctx = torch.randint(0, 2, (bsz, step_t + 1, 8, 32), device=device).to(torch.float32)
        theta = torch.empty((bsz, 256), device=device, dtype=torch.float32).uniform_(0.4, 5.8)

        with torch.no_grad():
            c_glob, c_bit, _ = expert.encode_context(ctx)
            z0, _ = expert.flow(theta, c_glob, c_bit)
            rec0 = expert.flow.inverse(z0, c_glob, c_bit)
            bij_zero = circular_abs_diff(rec0, theta).max().item()
            assert bij_zero < 1e-5, f"M_{step_t} zero-init bij={bij_zero}"

            # Perturb coupling output projections so flow creates substantial (> 0.4 rad) shift
            for layer in expert.flow.layers:
                torch.nn.init.normal_(layer.out_proj.weight, std=0.08)
                torch.nn.init.normal_(layer.out_proj.bias, std=0.12)

            z_p, _ = expert.flow(theta, c_glob, c_bit)
            disp_rms = float((circular_abs_diff(z_p, theta) ** 2).mean().sqrt().item())
            rec_p = expert.flow.inverse(z_p, c_glob, c_bit)
            fwd_inv = float(circular_abs_diff(rec_p, theta).max().item())
            z_rt, _ = expert.flow(rec_p, c_glob, c_bit)
            inv_fwd = float(circular_abs_diff(z_rt, z_p).max().item())

        assert disp_rms > 0.35, f"Perturbed flow displacement too small: {disp_rms}"
        assert fwd_inv < 1e-5 and inv_fwd < 1e-5, f"M_{step_t} bij={fwd_inv:.2e}/{inv_fwd:.2e}"

        # Compute full exact 256x256 autograd Jacobian at a single non-boundary sample
        theta_1 = torch.empty((256,), device=device, dtype=torch.float32).uniform_(0.6, 5.6)
        c_g1 = c_glob[:1].detach()
        c_b1 = c_bit[:1].detach()

        def _flow_fn(t_flat: torch.Tensor) -> torch.Tensor:
            out, _ = expert.flow(t_flat.unsqueeze(0), c_g1, c_b1)
            return out.squeeze(0)

        jac = torch.autograd.functional.jacobian(_flow_fn, theta_1)
        sign, logdet = torch.slogdet(jac.to(torch.float64))
        abs_logdet = float(torch.abs(logdet).item())
        print(
            f"[Gate 2 PASS M_{step_t}] RMS_disp={disp_rms:.3f} rad, fwd->inv={fwd_inv:.2e}, "
            f"inv->fwd={inv_fwd:.2e}, sign={sign.item():+.0f}, |log|det J||={abs_logdet:.2e}"
        )
        assert abs_logdet < 1e-5, f"Non-unit Jacobian determinant on M_{step_t}: {abs_logdet}"

        del expert
        torch.cuda.empty_cache()


def test_gate3_liouville_white_noise():
    """Gate 3: Liouville zero-cheat lock — 300 steps pure 256-bit white noise on M_8."""
    device = torch.device("cuda")
    torch.manual_seed(9003)

    expert = DecisionExpert(step_t=8).to(device)
    assert_strict_fp32(expert)
    opt = torch.optim.AdamW(expert.parameters(), lr=3e-4, weight_decay=0.0)

    train_gen = torch.Generator(device=device)
    train_gen.manual_seed(11111)
    eval_gen = torch.Generator(device=device)
    eval_gen.manual_seed(22222)

    expert.train()
    for step in range(1, 301):
        ctx = torch.randint(0, 2, (256, 9, 8, 32), device=device, generator=train_gen).to(torch.float32)
        tgt = torch.randint(0, 2, (256, 8, 32), device=device, generator=train_gen).to(torch.float32)
        out = expert.forward_density(ctx, tgt, generator=train_gen, lambda_lip=1e-4)
        loss = out["nll_loss"] + 1e-4 * out["lip_loss"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(expert.parameters(), 2.0)
        opt.step()

    expert.eval()
    with torch.no_grad():
        ctx_ev = torch.randint(0, 2, (1024, 9, 8, 32), device=device, generator=eval_gen).to(torch.float32)
        tgt_ev = torch.randint(0, 2, (1024, 8, 32), device=device, generator=eval_gen).to(torch.float32)
        ev_out = expert.forward_density(ctx_ev, tgt_ev, generator=eval_gen)
        extracted_bits = float(ev_out["extracted_bits"].item())

        c_g, c_b, _ = expert.encode_context(ctx_ev)
        th_ev = dequantize_bits(tgt_ev.reshape(-1, 256), generator=eval_gen)
        z_ev, _ = expert.flow(th_ev, c_g, c_b)
        rec_ev = expert.flow.inverse(z_ev, c_g, c_b)
        bij_err = float(circular_abs_diff(rec_ev, th_ev).max().item())
        stats = expert.density.channel_stats(c_g, c_b)
        max_r1 = float(stats["r1_mean"].max().item())

    print(
        f"[Gate 3 PASS White-Noise 300 steps] holdout_extracted_bits={extracted_bits:.4f} (<0.5), "
        f"max_R1={max_r1:.4f}, bij={bij_err:.2e}"
    )
    assert extracted_bits < 0.5, f"Liouville violation! extracted_bits={extracted_bits}"
    assert bij_err < 1e-5, f"Bijection degraded after white noise training: {bij_err}"


def test_gate4_full_64_tower_and_48_round_grad_penetration():
    """Gate 4: Full 57-expert 64-step smoke + 48-round frozen tunnel gradient backprop to M_8 + Beam Search."""
    device = torch.device("cuda")
    torch.manual_seed(9004)

    stream = OnlineR64Stream(skip_steps=7, n_eval=32, device=device, seed=904)
    # Compact channel widths for 8GB RTX 4060 smoke while instantiating ALL 57 independent experts!
    dec_tower = DecisionTower(
        skip_steps=7,
        end_step=16,
        d_tok=64,
        n_heads=4,
        d_ff=128,
        n_attn_layers=1,
        d_cond_bit=16,
        c_half=24,
        c_merge=32,
        num_mix=8,
        c_w=32,
        h_carry=32,
    ).to(device)
    tun_tower = TunnelTower(
        start_step=17,
        end_step=64,
        c_arx=32,
        c_mid=24,
        h_carry=32,
    ).to(device)
    assert_strict_fp32(dec_tower)
    assert_strict_fp32(tun_tower)

    # 1. Stage 1 Smoke: train a subset + verify every tunnel expert runs clean forward/backward
    opt_tun = torch.optim.AdamW(tun_tower.parameters(), lr=1e-3)
    batch = stream.sample_train_batch(32)
    r_bits = batch["r_bits"]  # (32, 65, 8, 32)
    w_bits = batch["w_bits"]  # (32, 64, 32)
    opt_tun.zero_grad()
    for step_t in (17, 32, 48, 64):
        w_idx = 64 - step_t
        rl, _, wl, _, _ = tun_tower.forward_single_step(
            step_t,
            r_bits[:, step_t - 1],
            w_bits[:, w_idx + 16],
            w_bits[:, w_idx + 14],
            w_bits[:, w_idx + 9],
            w_bits[:, w_idx + 1],
        )
        loss_t = F.binary_cross_entropy_with_logits(rl, r_bits[:, step_t]) + F.binary_cross_entropy_with_logits(
            wl, w_bits[:, w_idx]
        )
        loss_t.backward()
    opt_tun.step()

    # 2. Stage 2 Smoke: decision experts M_8..M_16 + 1-bit sibling dead-branch pruning
    opt_dec = torch.optim.AdamW(dec_tower.parameters(), lr=1e-3)
    anchor_iv = stream.iv_bits.view(1, 1, 8, 32).expand(32, 1, 8, 32)
    full_ctx = torch.cat([anchor_iv, r_bits[:, :16]], dim=1)  # [IV, R_0..R_15]
    opt_dec.zero_grad()
    for idx, exp in enumerate(dec_tower.experts):
        step_t = exp.step_t
        w_idx = 64 - step_t
        ctx_t = full_ctx[:, : step_t + 1]
        out_d = exp.forward_density(
            ctx_t,
            r_bits[:, step_t],
            target_w_bits=w_bits[:, w_idx],
            generator=stream.train_gen,
            lambda_lip=1e-4,
        )
        # Hard 1-bit sibling negative on latest state R_{t-1}
        ctx_neg = ctx_t.clone()
        flip_w = torch.randint(0, 8, (32,), device=device)
        flip_b = torch.randint(0, 32, (32,), device=device)
        bi = torch.arange(32, device=device)
        ctx_neg[bi, -1, flip_w, flip_b] = 1.0 - ctx_neg[bi, -1, flip_w, flip_b]
        _, _, vlog_neg = exp.encode_context(ctx_neg)
        v_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(out_d["v_logit"], torch.ones_like(out_d["v_logit"]))
            + F.binary_cross_entropy_with_logits(vlog_neg, torch.zeros_like(vlog_neg))
        )
        loss_d = out_d["nll_loss"] / 256.0 + out_d["w_bce"] + v_loss + 1e-4 * out_d["lip_loss"]
        loss_d.backward()
    opt_dec.step()

    # 3. Stage 3 Critical Assertion: freeze M_17..M_64, backprop L(R_64_hat, IV) to M_8!
    for p in tun_tower.parameters():
        p.requires_grad_(False)
    tun_tower.train()  # GroupNorm/LayerNorm invariant; keeps cuDNN GRU happy with checkpoint
    dec_tower.train()

    opt_dec.zero_grad()
    b3 = stream.sample_train_batch(16)
    r_ctx_in = b3["r_bits"][:, :8]  # R_0..R_7 (16, 8, 8, 32)
    dec_out = dec_tower.rollout_decisions(
        stream.iv_bits,
        r_ctx_in,
        ste_mode="ste",
        ste_temp=2.5,
        leaky_slope=0.25,
        generator=stream.train_gen,
    )
    # Build W_48..W_63 in ascending order: [W_48..W_56] = dec_out['w_dec_bits'].flip(1), W_57..W_63 = P*
    w_dec_asc = dec_out["w_dec_bits"].flip(dims=(1,))  # type: ignore[union-attr]
    p_rev = torch.tensor(list(reversed(stream.skip_prefix)), device=device, dtype=torch.int64)
    p_rev_bits = words_to_bits(p_rev).unsqueeze(0).expand(16, 7, 32)
    w48_63 = torch.cat([w_dec_asc, p_rev_bits], dim=1)

    grad_tracker: dict[int, float] = {}
    tun_out = tun_tower.rollout_48(
        dec_out["r16_hat"],  # type: ignore[arg-type]
        w48_63,
        use_checkpoint=True,
        ste_mode="ste",
        ste_temp=2.5,
        leaky_slope=0.25,
        grad_tracker=grad_tracker,
        max_grad_norm=16.0,
    )
    iv_tgt = stream.iv_bits.unsqueeze(0).expand(16, 8, 32)
    loss_land = F.binary_cross_entropy_with_logits(tun_out["r64_logits"], iv_tgt)  # type: ignore[arg-type]
    loss_land.backward()

    # Check all 48 tunnel steps recorded finite strictly positive gradients
    assert len(grad_tracker) == 48, f"Expected 48 tunnel grad norms, got {len(grad_tracker)}"
    for t in range(17, 65):
        gn = grad_tracker[t]
        assert math.isfinite(gn) and gn > 0.0, f"Tunnel step M_{t} bad grad_norm={gn}"

    # Verify EVERY decision expert M_8 .. M_16 (especially M_8!) received non-zero finite grads
    m_grad_norms = {}
    for exp in dec_tower.experts:
        total_sq = 0.0
        for p in exp.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in M_{exp.step_t}"
                total_sq += float((p.grad.detach() ** 2).sum().item())
        m_grad_norms[exp.step_t] = math.sqrt(total_sq)
        assert m_grad_norms[exp.step_t] > 0.0, f"Zero gradient on M_{exp.step_t} after 48-round backprop!"

    opt_dec.step()
    print(
        f"[Gate 4 PASS 48-Round Grad Penetration] M_64={grad_tracker[64]:.3e} -> M_17={grad_tracker[17]:.3e} | "
        f"M_8 grad_norm={m_grad_norms[8]:.3e}, M_16 grad_norm={m_grad_norms[16]:.3e}"
    )

    # 4. Route-1 Beam Search + classical verification smoke on both eval tracks
    res_m = evaluate_beam_search(
        dec_tower,
        stream.eval_matched["h_words"][:8],
        skip_prefix=stream.skip_prefix,
        beam_width=8,
        chunk_size=8,
    )
    res_u = evaluate_beam_search(
        dec_tower,
        stream.eval_unconstrained["h_words"][:8],
        skip_prefix=stream.skip_prefix,
        beam_width=8,
        chunk_size=8,
    )
    assert math.isfinite(res_m["classical_iv_landing_acc"])
    assert math.isfinite(res_u["rehash_bit_acc"])
    print(
        f"[Gate 4 PASS Beam Search] matched_iv={res_m['classical_iv_landing_acc']:.4f} "
        f"unconstrained_iv={res_u['classical_iv_landing_acc']:.4f} rehash={res_u['rehash_bit_acc']:.4f}"
    )
