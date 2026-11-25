"""Acceptance test suite for Phase-SHA V3: BiGRU, Dynamic Latch, Chunked BPTT, SMC, VRAM."""

from __future__ import annotations

import inspect
import math
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.bit_grid_backbone import BitGridBackbone
from src.boundary_clamp import build_clamped_boundary
from src.dual_basis import DualBasisEngine
from src.flow_trainer import prepare_ground_truth_nodes
from src.sha256_ref import (
    configure_cuda_memory,
    expand_message_schedule,
    sample_random_w16,
    sha256_digest,
    words_to_spins,
)
from src.v3_solver import (
    UnrolledSolverV3,
    _eval_eq_e_candidates,
    _eval_eq_s_candidates,
    _eval_eq_w_for_e_candidates,
    _eval_eq_w_for_w_candidates,
    spin_add32,
    spin_expand_schedule,
)
from src.v3_trainer import (
    UnrolledTrainerV3,
    build_v3_init,
    compute_v3_step_loss,
    run_chunked_bptt_step,
)

if torch.cuda.is_available():
    configure_cuda_memory(0.80)


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_v3_bigru_bf16_safe_and_round16_to_round0_connectivity() -> None:
    """Verify Bit-Axis BiGRU runs safely with finite grads and round-axis connects 16<->0."""
    device = _get_device()
    rounds = 18
    backbone = BitGridBackbone(
        rounds=rounds,
        d_bit=48,
        n_heads=3,
        n_blocks=2,
        d_ff_mult=2,
        use_carry_gru=True,
    ).to(device)

    # Round-axis sparse mask must bidirectionally connect message-schedule hops
    mask = backbone.round_attn_mask[0, 0]
    for u, v in ((16, 0), (17, 1), (16, 1), (16, 9), (16, 14)):
        assert bool(mask[u, v].item())
        assert bool(mask[v, u].item())

    gen = torch.Generator(device=device).manual_seed(301)
    w16 = sample_random_w16(batch_size=8, num_bits=32, device=device, generator=gen)
    x_star, boundary = prepare_ground_truth_nodes(w16, rounds=rounds, num_bits=32)
    x_init = build_v3_init(boundary, zero_prob=0.5, generator=gen)

    engine = DualBasisEngine(rounds=rounds).to(device)
    lam = torch.ones(32, dtype=torch.float32, device=device)
    grad_radar, res_feat, _, _ = engine.compute_radar_and_features(
        x_init, boundary, lam=lam
    )

    optimizer = torch.optim.AdamW(backbone.parameters(), lr=1e-2)
    for step_idx in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits, h_out = backbone(
            x_nodes=x_init,
            grad_radar=grad_radar,
            clamp_mask=boundary.clamp_mask,
            res_features=res_feat,
            k_spins=engine.K_spins,
            step_frac=torch.zeros(8, device=device),
            lam=lam,
            h_prev=None,
        )
        assert logits.shape == (8, rounds, 3, 32)
        assert h_out.shape == (8, rounds, 32, 48)
        assert torch.isfinite(logits).all()

        loss = logits.square().mean() + 0.1 * h_out.square().mean()
        loss.backward()
        if step_idx == 0:
            # First step wakes up AdaLN-Zero gate gt_c from exact zero initialization
            optimizer.step()

    gru_grads = [
        p.grad
        for n, p in backbone.named_parameters()
        if "carry_gru" in n and p.grad is not None
    ]
    assert len(gru_grads) > 0
    for g in gru_grads:
        assert torch.isfinite(g).all()
        assert g.abs().sum().item() > 0.0


def test_v3_single_equation_evaluators_match_dual_basis_exact() -> None:
    """Verify fast single-equation candidate evaluators match dual_basis.compute_wrapped_deltas."""
    device = _get_device()
    rounds = 18
    gen = torch.Generator(device=device).manual_seed(302)
    w16 = sample_random_w16(batch_size=4, num_bits=32, device=device, generator=gen)
    x_star, boundary = prepare_ground_truth_nodes(w16, rounds=rounds, num_bits=32)
    engine = DualBasisEngine(rounds=rounds).to(device)

    # Add controlled continuous perturbations on multiple free words
    x_noisy = boundary.apply_clamp(
        x_star
        + 0.35
        * torch.randn(x_star.shape, dtype=torch.float32, device=device, generator=gen)
    )
    a_full, e_full, w_full = boundary.assemble_full(x_noisy)
    delta_e, delta_w, delta_s = engine.compute_wrapped_deltas(a_full, e_full, w_full)
    assert delta_s is not None
    res_e_gt = (2.0 * torch.sin(0.5 * delta_e).square()).sum(dim=-1)
    res_w_gt = (2.0 * torch.sin(0.5 * delta_w).square()).sum(dim=-1)
    res_s_gt = (2.0 * torch.sin(0.5 * delta_s).square()).sum(dim=-1)

    # Compare evaluators on exact and perturbed 2-candidate probes
    cand_s = torch.stack([w_full[:, 17], x_star[:, 17, 2]], dim=1)
    ev_s = _eval_eq_s_candidates(engine, w_full, 17, cand_s)
    assert torch.allclose(ev_s[:, 0], res_s_gt[:, 1], atol=1e-5)

    cand_a = torch.stack([a_full[:, 13], x_star[:, 10, 0]], dim=1)
    ev_a = _eval_eq_e_candidates(engine, a_full, e_full, 10, cand_a)
    assert torch.allclose(ev_a[:, 0], res_e_gt[:, 13], atol=1e-5)

    cand_e = torch.stack([e_full[:, 17], x_star[:, 14, 1]], dim=1)
    ev_e = _eval_eq_w_for_e_candidates(engine, a_full, e_full, w_full, 14, cand_e)
    assert torch.allclose(ev_e[:, 0], res_w_gt[:, 17], atol=1e-5)

    cand_w = torch.stack([w_full[:, 0], x_star[:, 0, 2]], dim=1)
    ev_w = _eval_eq_w_for_w_candidates(engine, a_full, e_full, 0, cand_w)
    assert torch.allclose(ev_w[:, 0], res_w_gt[:, 0], atol=1e-5)

    # Exact ground-truth trace must give < 1e-10 on all equations
    a_t, e_t, w_t = boundary.assemble_full(x_star)
    assert _eval_eq_s_candidates(engine, w_t, 17, w_t[:, 17:18]).max().item() < 1e-10
    assert _eval_eq_e_candidates(engine, a_t, e_t, 10, a_t[:, 13:14]).max().item() < 1e-10
    assert (
        _eval_eq_w_for_e_candidates(engine, a_t, e_t, w_t, 14, e_t[:, 17:18]).max().item()
        < 1e-10
    )
    assert _eval_eq_w_for_w_candidates(engine, a_t, e_t, 0, w_t[:, 0:1]).max().item() < 1e-10


def test_v3_dynamic_latch_flip_recovery_cascade_and_zero_leakage() -> None:
    """Verify 0/1/2-bit flip recovery cascades cleanly, 5-bit errors & W16 never latch, no x_star leak."""
    # Zero-leakage API check: none of the V3 init/solver methods take x_star
    for fn in (
        build_v3_init,
        UnrolledSolverV3._try_dynamic_latch,
        UnrolledSolverV3.infer_v3,
        UnrolledSolverV3.solve,
        UnrolledSolverV3.solve_batch_boundaries,
        UnrolledSolverV3.solve_single_boundary,
    ):
        sig = inspect.signature(fn)
        assert "x_star" not in sig.parameters

    device = _get_device()
    gen = torch.Generator(device=device).manual_seed(303)

    # Case 1: R=16 cascade (E12 with 2 flipped MSBs -> E11 with 1 flipped MSB -> A8 clean)
    engine16 = DualBasisEngine(rounds=16).to(device)
    model16 = BitGridBackbone(rounds=16, d_bit=24, n_heads=3, n_blocks=1, use_carry_gru=False).to(device)
    solver16 = UnrolledSolverV3(model=model16, dual_basis=engine16)

    w16_16 = sample_random_w16(batch_size=3, num_bits=32, device=device, generator=gen)
    x_star16, b16 = prepare_ground_truth_nodes(w16_16, rounds=16, num_bits=32)

    dyn_clamped = b16.x_clamped.clone()
    dyn_mask = b16.clamp_mask.clone()
    latched_logits = torch.zeros_like(dyn_clamped)
    x_state = b16.apply_clamp(torch.zeros_like(x_star16))
    logits = torch.zeros_like(x_star16)

    # Set high-confidence ground truth on E12, E11, A8
    for t_slot, ch_slot in ((12, 1), (11, 1), (8, 0)):
        logits[:, t_slot, ch_slot, :] = 4.0 * x_star16[:, t_slot, ch_slot, :]

    # Flip 2 high bits (30, 31) of E12 with lower confidence so top-5 covers them
    logits[:, 12, 1, 30] = -1.2 * x_star16[:, 12, 1, 30]
    logits[:, 12, 1, 31] = -1.3 * x_star16[:, 12, 1, 31]
    # Flip 1 bit (29) of E11
    logits[:, 11, 1, 29] = -1.1 * x_star16[:, 11, 1, 29]

    # Sample 2: deliberately corrupt 5 bits of E12 -> must NOT latch!
    for b_pos in (0, 5, 12, 20, 28):
        logits[2, 12, 1, b_pos] = -1.2 * x_star16[2, 12, 1, b_pos]

    x_state[:, 12, 1, :] = torch.tanh(logits[:, 12, 1, :])
    x_state[:, 11, 1, :] = torch.tanh(logits[:, 11, 1, :])
    x_state[:, 8, 0, :] = torch.tanh(logits[:, 8, 0, :])

    x_out = solver16._try_dynamic_latch(
        x=x_state,
        logits=logits,
        b_chunk=b16,
        dyn_clamped=dyn_clamped,
        dyn_mask=dyn_mask,
        latched_logits=latched_logits,
    )

    # Samples 0 and 1 must have cascaded and exact-latched E12, E11, AND A8!
    for b_i in (0, 1):
        for t_slot, ch_slot in ((12, 1), (11, 1), (8, 0)):
            assert (dyn_mask[b_i, t_slot, ch_slot] == 1.0).all()
            assert torch.equal(dyn_clamped[b_i, t_slot, ch_slot], x_star16[b_i, t_slot, ch_slot])
            assert torch.equal(x_out[b_i, t_slot, ch_slot], x_star16[b_i, t_slot, ch_slot])

    # Sample 2 (5-bit corruption) must remain unlatched!
    assert dyn_mask[2, 12, 1].sum().item() == 0.0
    assert dyn_mask[2, 11, 1].sum().item() == 0.0
    assert dyn_mask[2, 8, 0].sum().item() == 0.0

    # Case 2: R=18 3-hop cascade (W17 -> E14 -> A10) while W16 remains unlatched
    engine18 = DualBasisEngine(rounds=18).to(device)
    model18 = BitGridBackbone(rounds=18, d_bit=24, n_heads=3, n_blocks=1, use_carry_gru=False).to(device)
    solver18 = UnrolledSolverV3(model=model18, dual_basis=engine18)

    w16_18 = sample_random_w16(batch_size=2, num_bits=32, device=device, generator=gen)
    x_star18, b18 = prepare_ground_truth_nodes(w16_18, rounds=18, num_bits=32)

    dc18 = b18.x_clamped.clone()
    dm18 = b18.clamp_mask.clone()
    ll18 = torch.zeros_like(dc18)
    x18 = b18.apply_clamp(torch.zeros_like(x_star18))
    log18 = torch.zeros_like(x_star18)

    for t_slot, ch_slot in ((17, 2), (14, 1), (10, 0), (16, 2)):
        log18[:, t_slot, ch_slot, :] = 4.0 * x_star18[:, t_slot, ch_slot, :]
        x18[:, t_slot, ch_slot, :] = torch.tanh(log18[:, t_slot, ch_slot, :])

    solver18._try_dynamic_latch(x18, log18, b18, dc18, dm18, ll18)
    for t_slot, ch_slot in ((17, 2), (14, 1), (10, 0)):
        assert (dm18[:, t_slot, ch_slot] == 1.0).all()
        assert torch.equal(dc18[:, t_slot, ch_slot], x_star18[:, t_slot, ch_slot])
    # W16 prereq depends on unlocked W0 -> must NOT latch!
    assert dm18[:, 16, 2].sum().item() == 0.0


def test_v3_init_wavefront_loss_chunked_bptt_and_ema(tmp_path) -> None:
    """Verify zero-leak init ratio, 3x frontier weighting, multi-step BPTT grads, and EMA."""
    device = _get_device()
    gen = torch.Generator(device=device).manual_seed(304)
    rounds = 12

    w16 = sample_random_w16(batch_size=64, num_bits=32, device=device, generator=gen)
    x_star, boundary = prepare_ground_truth_nodes(w16, rounds=rounds, num_bits=32)
    x_init = build_v3_init(boundary, zero_prob=0.85, generator=gen)
    free_abs = (x_init * (1.0 - boundary.clamp_mask)).abs().sum(dim=(1, 2, 3))
    zero_frac = float((free_abs == 0.0).float().mean().item())
    assert 0.65 <= zero_frac <= 0.99

    # Check wavefront weighting triggers when some rounds fall in [0.55, 0.99] accuracy
    engine = DualBasisEngine(rounds=rounds).to(device)
    logits_mix = 2.0 * x_star.clone()
    # Force round 3 to ~75% accuracy
    logits_mix[:, 3, :, :8] = -2.0 * x_star[:, 3, :, :8]
    _, _, _, _, frontier_cnt, _, _ = compute_v3_step_loss(
        logits_mix, x_star, boundary, engine, wavefront_weight=3.0
    )
    assert frontier_cnt >= 1

    # Chunked BPTT (k_unroll=4, chunk_len=4) must backprop through h_prev_proj
    model = BitGridBackbone(
        rounds=rounds,
        d_bit=24,
        n_heads=3,
        n_blocks=1,
        d_ff_mult=2,
        use_carry_gru=True,
    ).to(device)
    metrics = run_chunked_bptt_step(
        model=model,
        dual_basis=engine,
        x_star=x_star[:8],
        boundary=prepare_ground_truth_nodes(w16[:8], rounds=rounds, num_bits=32)[1],
        x_init=x_init[:8],
        k_unroll=4,
        bptt_chunk_len=4,
        do_backward=True,
    )
    assert math.isfinite(metrics["loss"])
    for p in model.h_prev_proj.parameters():
        assert p.grad is not None and p.grad.abs().sum().item() > 0.0

    # Check EMA update
    cfg = {
        "version": "v3",
        "rounds": rounds,
        "domain_mode": "int_range",
        "num_bits": 32,
        "holdout_size": 256,
        "d_bit": 24,
        "n_heads": 3,
        "n_blocks": 1,
        "d_ff_mult": 2,
        "use_carry_gru": True,
        "K_unroll": 4,
        "bptt_chunk_len": 4,
        "K_infer": 8,
        "anneal_K": 32,
        "batch_size": 16,
        "ema_decay": 0.995,
    }
    tr = UnrolledTrainerV3(config=cfg, output_dir=tmp_path, device=device)
    ema_before = tr.ema_model.out_proj.weight.detach().clone()
    tr.train_one_step()
    assert tr.step == 1
    assert not torch.equal(tr.model.out_proj.weight.detach(), tr.ema_model.out_proj.weight.detach())
    assert not torch.equal(ema_before, tr.ema_model.out_proj.weight.detach())


def test_v3_smc_resampling_elitism_and_diversity() -> None:
    """Verify SMC preserves #1 elite, obeys clamping/latch masks, and sustains unique_ratio > 0.5."""
    device = _get_device()
    rounds = 16
    engine = DualBasisEngine(rounds=rounds).to(device)
    model = BitGridBackbone(
        rounds=rounds,
        d_bit=24,
        n_heads=3,
        n_blocks=1,
        d_ff_mult=2,
        use_carry_gru=True,
    ).to(device)
    solver = UnrolledSolverV3(
        model=model,
        dual_basis=engine,
        num_candidates=256,
        num_steps=16,
        chunk_size=1024,
        use_dynamic_latch=True,
        use_smc_resampling=True,
        smc_resample_every=4,
        smc_elite_k=64,
    )

    gen = torch.Generator(device=device).manual_seed(305)
    w16 = sample_random_w16(batch_size=2, num_bits=32, device=device, generator=gen)
    x_star, boundary = prepare_ground_truth_nodes(w16, rounds=rounds, num_bits=32)

    # Plant exact discrete solution at candidate index 173 for digest 0
    k = 256
    b_exp = boundary.repeat_for_candidates(k)
    x_cur = torch.tanh(0.8 * torch.randn((2 * k, rounds, 3, 32), device=device, generator=gen))
    x_cur = b_exp.apply_clamp(x_cur)
    x_cur[173] = x_star[0]
    dyn_clamped = b_exp.x_clamped.clone()
    dyn_mask = b_exp.clamp_mask.clone()
    latched_logits = torch.zeros_like(dyn_clamped)

    x_res, _, dc_res, dm_res, _ = solver._smc_resample(
        x=x_cur,
        h_prev=None,
        dyn_clamped=dyn_clamped,
        dyn_mask=dyn_mask,
        latched_logits=latched_logits,
        b_chunk=b_exp,
        num_digests=2,
        k_per_digest=k,
        temp=0.7,
        generator=gen,
    )
    # Digest 0 slot 0 MUST be the planted exact 0-energy preimage!
    assert torch.equal(x_res[0], x_star[0])

    # End-to-end solve with SMC active must maintain high diversity and valid clamps
    digests = sha256_digest(w16, rounds=rounds)
    res = solver.solve(digests, num_bits=32, generator=gen, return_last_trajectories=True)
    assert res.mean_unique_ratio > 0.50
    assert res.final_x_sample is not None
    b1 = build_clamped_boundary(digests[1], rounds=rounds, num_bits=32)
    assert ((res.final_x_sample - b1.x_clamped) * b1.clamp_mask).abs().max().item() == 0.0


def test_v3_v2_to_v3_partial_warm_start_and_v3_strict_warm_start(tmp_path) -> None:
    """Verify loading V2 checkpoints into V3 initializes only carry_gru fresh, then V3->V3 strict."""
    device = _get_device()
    src_v2 = Path("runs/r16_32b_v2/best.pt")
    if not src_v2.exists():
        pytest.skip("runs/r16_32b_v2/best.pt not found for warm-start test.")

    cfg = {
        "version": "v3",
        "rounds": 16,
        "domain_mode": "int_range",
        "num_bits": 32,
        "holdout_size": 256,
        "d_bit": 48,
        "n_heads": 3,
        "n_blocks": 2,
        "d_ff_mult": 2,
        "use_carry_gru": True,
        "K_unroll": 4,
        "bptt_chunk_len": 4,
        "K_infer": 8,
        "anneal_K": 32,
        "batch_size": 16,
    }
    tr = UnrolledTrainerV3(config=cfg, output_dir=tmp_path / "v3a", device=device)
    tr.load_weights_only(src_v2)
    assert tr.step == 0
    assert torch.equal(
        tr.model.out_proj.weight.detach(), tr.ema_model.out_proj.weight.detach()
    )
    ckpt_v3 = tr.save_checkpoint("best.pt")

    cfg18 = dict(cfg, rounds=18)
    tr18 = UnrolledTrainerV3(config=cfg18, output_dir=tmp_path / "v3b", device=device)
    tr18.load_weights_only(ckpt_v3)
    m = tr18.train_one_step()
    assert math.isfinite(m["loss"])


def test_v3_closed_loop_spin_schedule_exact_and_grad_flow() -> None:
    """Verify differentiable spin_add32/spin_expand_schedule match integer SHA-256 and flow grads to W0."""
    device = _get_device()
    gen = torch.Generator(device=device).manual_seed(306)
    w16 = sample_random_w16(batch_size=64, num_bits=32, device=device, generator=gen)

    # Exact adder check vs uint32 addition
    w_other = sample_random_w16(batch_size=64, num_bits=32, device=device, generator=gen)
    sum_int = (w16[:, 0] + w_other[:, 0]) & 0xFFFFFFFF
    sum_spin_gt = words_to_spins(sum_int, dtype=torch.float32)
    sum_spin_diff = spin_add32(
        words_to_spins(w16[:, 0], dtype=torch.float32),
        words_to_spins(w_other[:, 0], dtype=torch.float32),
    )
    assert torch.equal(sum_spin_diff, sum_spin_gt)

    # Exact schedule equivalence for R=18 and R=64
    for rounds in (18, 64):
        w_int = expand_message_schedule(w16, rounds=rounds)
        w_spin_gt = words_to_spins(w_int, dtype=torch.float32)
        w_spin_pred = spin_expand_schedule(w_spin_gt[:, :16, :], rounds=rounds)
        assert torch.equal(w_spin_pred, w_spin_gt)

    # Differentiable closed-loop gradients from Eq W(16) + W16 BCE straight back to W0 logits
    rounds = 18
    engine = DualBasisEngine(rounds=rounds).to(device)
    x_star, boundary = prepare_ground_truth_nodes(w16[:8], rounds=rounds, num_bits=32)
    for init_mode in ("zero", "saturated"):
        if init_mode == "zero":
            logits = torch.zeros_like(x_star, requires_grad=True)
        else:
            logits = (8.0 * (-x_star)).detach().requires_grad_(True)
        loss, _, phys, _, _, _, _ = compute_v3_step_loss(
            logits=logits,
            x_star=x_star,
            boundary=boundary,
            dual_basis=engine,
            alpha_phys=0.25,
            logit_reg_weight=0.0015,
            wavefront_weight=3.0,
            closed_loop_schedule=True,
        )
        assert math.isfinite(float(loss.item()))
        assert math.isfinite(phys)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        w0_grad_norm = float(logits.grad[:, 0, 2, :].abs().sum().item())
        assert w0_grad_norm > 0.0, f"Zero W0 gradient under closed_loop_schedule ({init_mode})"


def test_v3_dual_mask_isolation_and_v4d_calibrated_probs() -> None:
    """Verify neural backbone always sees static_mask despite dyn_mask latching, and check V4D probs."""
    device = _get_device()
    rounds = 16
    gen = torch.Generator(device=device).manual_seed(307)
    w16 = sample_random_w16(batch_size=2, num_bits=32, device=device, generator=gen)
    x_star, boundary = prepare_ground_truth_nodes(w16, rounds=rounds, num_bits=32)
    engine = DualBasisEngine(rounds=rounds).to(device)

    seen_masks: list[torch.Tensor] = []

    class SpyBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.real = BitGridBackbone(
                rounds=rounds, d_bit=24, n_heads=3, n_blocks=1, use_carry_gru=False
            ).to(device)

        def forward(
            self, x_nodes, grad_radar, clamp_mask, res_features, k_spins, step_frac, lam, h_prev=None
        ):
            seen_masks.append(clamp_mask.detach().clone())
            # Force exact ground truth logits on E12 so dyn_mask latches E12 mid-loop!
            logits, h_out = self.real(
                x_nodes, grad_radar, clamp_mask, res_features, k_spins, step_frac, lam, h_prev=h_prev
            )
            logits[:, 12, 1, :] = 4.0 * x_star[:, 12, 1, :]
            return logits, h_out

    spy = SpyBackbone().to(device)
    solver = UnrolledSolverV3(
        model=spy,  # type: ignore[arg-type]
        dual_basis=engine,
        num_steps=4,
        damp=0.5,
        use_dynamic_latch=True,
        latch_every=2,
        sync_h_damp=True,
    )
    x_zero = boundary.apply_clamp(torch.zeros_like(boundary.x_clamped))
    _, _, latch_ratio, dyn_mask = solver.infer_v3(
        boundary, x_zero, use_neural=True, return_dyn_mask=True
    )
    assert float(latch_ratio.min().item()) > 0.0
    assert (dyn_mask[:, 12, 1, :] == 1.0).all()
    for m_seen in seen_masks:
        assert torch.equal(m_seen, boundary.clamp_mask)

    # Check V4D calibrated probability conventions: P(bit=1) -> >0.99 for bit 1 (spin -1), <0.01 for bit 0
    spy_solver = UnrolledSolverV3(
        model=spy,  # type: ignore[arg-type]
        dual_basis=engine,
        num_steps=4,
        damp=0.5,
        use_dynamic_latch=True,
        latch_every=2,
        sync_h_damp=True,
    )
    dig16 = sha256_digest(w16, rounds=16)
    p16 = spy_solver.predict_calibrated_bit_probs(dig16, num_bits=32)
    e12_probs = p16["node_bit1_probs"][:, 12, 1, :]
    bit1_mask = x_star[:, 12, 1, :] < 0
    bit0_mask = x_star[:, 12, 1, :] > 0
    assert (e12_probs[bit1_mask] > 0.99).all()
    assert (e12_probs[bit0_mask] < 0.01).all()

    model18 = BitGridBackbone(
        rounds=18, d_bit=24, n_heads=3, n_blocks=1, d_ff_mult=2, use_carry_gru=True
    ).to(device)
    eng18 = DualBasisEngine(rounds=18).to(device)
    solver18 = UnrolledSolverV3(
        model=model18,
        dual_basis=eng18,
        num_steps=8,
        damp=0.25,
        sync_h_damp=True,
        closed_loop_schedule=True,
    )
    assert "x_star" not in inspect.signature(solver18.predict_calibrated_bit_probs).parameters
    dig18 = sha256_digest(w16, rounds=18)
    probs_out = solver18.predict_calibrated_bit_probs(dig18, num_bits=32)
    assert probs_out["w0_bit1_probs"].shape == (2, 32)
    assert probs_out["node_bit1_probs"].shape == (2, 18, 3, 32)
    assert probs_out["latched_mask"].shape == (2, 18, 3, 32)
    assert probs_out["latched_mask"].dtype == torch.bool
    assert float(probs_out["node_bit1_probs"].min().item()) >= 0.0
    assert float(probs_out["node_bit1_probs"].max().item()) <= 1.0


@pytest.mark.parametrize(
    "rounds,k_unroll",
    [(16, 14), (18, 12)],
)
def test_v3_vram_and_speed_stress_under_5_2gb(rounds: int, k_unroll: int, tmp_path) -> None:
    """Stress test Stage 3A (R=16, K_unroll=14) and Stage 3B (R=18 closed-loop, K_unroll=12) < 5.2 GB."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required for V3 VRAM stress test.")

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    is_r18 = rounds >= 18
    cfg = {
        "version": "v3",
        "rounds": rounds,
        "domain_mode": "int_range",
        "num_bits": 32,
        "holdout_size": 256,
        "eval_batch_size": 256,
        "d_bit": 48,
        "n_heads": 3,
        "n_blocks": 2,
        "d_ff_mult": 2,
        "use_carry_gru": True,
        "grad_checkpointing": False,
        "K_unroll": k_unroll,
        "bptt_chunk_len": 4,
        "damp": 0.28 if is_r18 else 0.30,
        "damp_jitter_min": 0.20 if is_r18 else None,
        "damp_jitter_max": 0.35 if is_r18 else None,
        "sync_h_damp": is_r18,
        "closed_loop_schedule": is_r18,
        "K_infer": 32,
        "infer_damp": 0.25 if is_r18 else 0.20,
        "anneal_K": 256,
        "anneal_chunk_size": 1024,
        "anneal_eval_samples": 4,
        "batch_size": 128,
        "use_dynamic_latch": True,
        "use_smc_resampling": True,
        "seed": 3000 + rounds,
    }
    tr = UnrolledTrainerV3(config=cfg, output_dir=tmp_path, device=device)

    t_last = 0.0
    for _ in range(2):
        t0 = time.perf_counter()
        m = tr.train_one_step()
        torch.cuda.synchronize(device)
        t_last = (time.perf_counter() - t0) * 1000.0
        assert math.isfinite(m["loss"])

    t_ev0 = time.perf_counter()
    ev = tr.evaluate_holdout(run_anneal=True)
    torch.cuda.synchronize(device)
    t_eval_s = time.perf_counter() - t_ev0

    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    print(
        f"\n[V3 Stress Test] R={rounds}, B=128, K_unroll={k_unroll}, closed_loop={is_r18}: "
        f"peak={peak_gb:.3f} GB, step={t_last:.1f} ms, eval(256+4x256x32)={t_eval_s:.2f}s, "
        f"uniq={ev['unique_ratio@256']:.3f}",
        flush=True,
    )
    assert ev["unique_ratio@256"] > 0.50
    assert peak_gb < 5.2, f"Peak VRAM {peak_gb:.3f} GB >= 5.2 GB!"
    assert t_last < 2500.0, f"Step time {t_last:.1f} ms >= 2500 ms!"
