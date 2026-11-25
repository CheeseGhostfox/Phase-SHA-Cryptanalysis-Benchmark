"""Mandatory acceptance test suite 1: algebraic and 2-adic equivalence tests."""

from __future__ import annotations

import hashlib
import os

import pytest
import torch

from src.boundary_clamp import build_clamped_boundary
from src.dual_basis import (
    DualBasisEngine,
    spin_Ch,
    spin_Maj,
    spin_Sigma0,
    spin_Sigma1,
    spin_rotr,
    spin_shr,
    spin_sigma0,
    spin_sigma1,
    spin_xor,
)
from src.sha256_ref import (
    MASK32,
    ch32,
    digest_words_to_bytes,
    maj32,
    pad_bytes_to_w16,
    rotr32,
    sample_random_w16,
    sha256_trace,
    shr32,
    sigma0_lower,
    sigma0_upper,
    sigma1_lower,
    sigma1_upper,
    spins_to_words,
    words_to_spins,
)


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_sha256_matches_hashlib_100_random_messages() -> None:
    """GPU-vectorized sha256_ref(R=64) must match hashlib.sha256 on 100 random byte strings."""
    device = _get_device()
    rng = torch.Generator(device="cpu").manual_seed(12345)

    messages = [b"", b"abc", b"a" * 55]
    for _ in range(97):
        length = int(torch.randint(0, 56, (1,), generator=rng).item())
        messages.append(os.urandom(length))

    assert len(messages) == 100
    w16_batch = torch.stack([pad_bytes_to_w16(m, device=device) for m in messages], dim=0)
    trace = sha256_trace(w16_batch, rounds=64)

    for idx, msg in enumerate(messages):
        expected = hashlib.sha256(msg).digest()
        actual = digest_words_to_bytes(trace.digest[idx])
        assert actual == expected, f"Mismatch on message index {idx} (len={len(msg)})"


@pytest.mark.parametrize("rounds", [10, 12, 16, 20, 64])
def test_boundary_clamp_backsolve_exact(rounds: int) -> None:
    """Back-solved A_{R-7..R-4} and clamped boundaries must match forward trace with 0 bit errors."""
    device = _get_device()
    gen = torch.Generator(device=device).manual_seed(100 + rounds)
    w16 = sample_random_w16(
        batch_size=32, mode="int_range", num_bits=32, device=device, generator=gen
    )

    trace = sha256_trace(w16, rounds=rounds)
    boundary = build_clamped_boundary(
        digest=trace.digest,
        rounds=rounds,
        domain_mode="int_range",
        num_bits=32,
    )

    # Left boundary A_{-3..0}, E_{-3..0}
    assert torch.equal(boundary.A_words_clamped[:, 0:4], trace.A_full[:, 0:4])
    assert torch.equal(boundary.E_words_clamped[:, 0:4], trace.E_full[:, 0:4])

    # Right boundary A_{R-3..R}, E_{R-3..R}
    assert torch.equal(
        boundary.A_words_clamped[:, rounds : rounds + 4],
        trace.A_full[:, rounds : rounds + 4],
    )
    assert torch.equal(
        boundary.E_words_clamped[:, rounds : rounds + 4],
        trace.E_full[:, rounds : rounds + 4],
    )

    # Analytical 4-round back-solved A_{R-7..R-4} (indices R-4 .. R-1)
    assert torch.equal(
        boundary.A_words_clamped[:, rounds - 4 : rounds],
        trace.A_full[:, rounds - 4 : rounds],
    )

    # Check exact free word counts in nodes: R-8 for A, R-4 for E
    clamp_mask = boundary.clamp_mask  # (B, R, 3, 32)
    free_a_words = (clamp_mask[0, :, 0, :].sum(dim=-1) == 0).sum().item()
    free_e_words = (clamp_mask[0, :, 1, :].sum(dim=-1) == 0).sum().item()
    assert free_a_words == rounds - 8
    assert free_e_words == rounds - 4

    # Verify x_star agrees with x_clamped everywhere clamp_mask == 1
    a_mid = words_to_spins(trace.A_full[:, 3 : 3 + rounds])
    e_mid = words_to_spins(trace.E_full[:, 3 : 3 + rounds])
    w_mid = words_to_spins(trace.W_full[:, :rounds])
    x_star = torch.stack([a_mid, e_mid, w_mid], dim=2)
    diff_on_clamped = ((x_star - boundary.x_clamped) * clamp_mask).abs().max().item()
    assert diff_on_clamped == 0.0


def test_fixed_len_bytes_domain_clamp_exact() -> None:
    """Verify fixed_len_bytes domain mode clamps padding and bit-length bits accurately."""
    device = _get_device()
    gen = torch.Generator(device=device).manual_seed(555)
    for msg_len in (0, 3, 4, 17, 55):
        w16 = sample_random_w16(
            batch_size=16,
            mode="fixed_len_bytes",
            msg_bytes_len=msg_len,
            device=device,
            generator=gen,
        )
        trace = sha256_trace(w16, rounds=20)
        boundary = build_clamped_boundary(
            digest=trace.digest,
            rounds=20,
            domain_mode="fixed_len_bytes",
            msg_bytes_len=msg_len,
        )
        w_spins = words_to_spins(trace.W_full[:, :20])
        diff = ((w_spins - boundary.W_full_clamped) * boundary.W_full_mask).abs().max().item()
        assert diff == 0.0
        # Total free bits in W_0..15 must equal msg_len * 8
        free_bits_in_w = int((1.0 - boundary.W_full_mask[0, :16, :]).sum().item())
        assert free_bits_in_w == msg_len * 8


def test_continuous_spin_ops_exact_on_discrete() -> None:
    """Bipolar spin operators must match integer uint32 bitwise ops on discrete spins."""
    device = _get_device()
    gen = torch.Generator(device=device).manual_seed(777)
    a = torch.randint(0, 1 << 32, (64,), dtype=torch.int64, device=device, generator=gen)
    b = torch.randint(0, 1 << 32, (64,), dtype=torch.int64, device=device, generator=gen)
    c = torch.randint(0, 1 << 32, (64,), dtype=torch.int64, device=device, generator=gen)

    sa = words_to_spins(a)
    sb = words_to_spins(b)
    sc = words_to_spins(c)

    assert torch.equal(spins_to_words(spin_xor(sa, sb)), (a ^ b) & MASK32)
    assert torch.equal(spins_to_words(spin_rotr(sa, 13)), rotr32(a, 13))
    assert torch.equal(spins_to_words(spin_shr(sa, 10)), shr32(a, 10))
    # Verify SHR pads upper bits with +1.0 (logical 0), never 0.0
    shifted = spin_shr(sa, 10)
    assert torch.all(shifted[:, 22:] == 1.0)

    assert torch.equal(spins_to_words(spin_Sigma0(sa)), sigma0_upper(a))
    assert torch.equal(spins_to_words(spin_Sigma1(sa)), sigma1_upper(a))
    assert torch.equal(spins_to_words(spin_sigma0(sa)), sigma0_lower(a))
    assert torch.equal(spins_to_words(spin_sigma1(sa)), sigma1_lower(a))
    assert torch.equal(spins_to_words(spin_Ch(sa, sb, sc)), ch32(a, b, c))
    assert torch.equal(spins_to_words(spin_Maj(sa, sb, sc)), maj32(a, b, c))


@pytest.mark.parametrize("rounds", [10, 12, 16, 20, 64])
def test_dual_basis_zero_residual_on_ground_truth(rounds: int) -> None:
    """Ground-truth discrete spins plugged into dual_basis.py yield E_phys < 1e-10."""
    device = _get_device()
    gen = torch.Generator(device=device).manual_seed(900 + rounds)
    w16 = sample_random_w16(
        batch_size=32, mode="int_range", num_bits=32, device=device, generator=gen
    )
    trace = sha256_trace(w16, rounds=rounds)

    a_spins = words_to_spins(trace.A_full, dtype=torch.float32)
    e_spins = words_to_spins(trace.E_full, dtype=torch.float32)
    w_spins = words_to_spins(trace.W_full, dtype=torch.float32)

    engine = DualBasisEngine(rounds=rounds).to(device)
    energy, features, stats = engine.compute_energy_and_features(a_spins, e_spins, w_spins)
    delta_e, delta_w, delta_s = engine.compute_wrapped_deltas(a_spins, e_spins, w_spins)

    assert energy.max().item() < 1e-10
    assert stats["per_scale"].max().item() < 1e-10
    assert features.abs().max().item() < 1e-5

    assert delta_e.abs().max().item() < 1e-5
    assert delta_w.abs().max().item() < 1e-5
    if rounds > 16:
        assert delta_s is not None
        assert delta_s.abs().max().item() < 1e-5
    else:
        assert delta_s is None
