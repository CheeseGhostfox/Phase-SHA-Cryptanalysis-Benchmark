"""Mandatory bit-exactness and zero-leakage tests for R=18 SHA-256 CNF encoder."""

from __future__ import annotations

import inspect

import pytest
import torch
from pysat.solvers import Cadical195

from src.sha256_cnf import SHA256CNFEncoder, compute_algebraic_known_words_r18
from src.sha256_ref import sample_random_w16, sha256_digest, sha256_trace


def test_r18_cnf_exact_sat_and_single_bit_flip_unsat() -> None:
    """Verify 10 random full 32-bit R=18 traces: true W0 is SAT (100% trace match) and 1-bit flip is UNSAT."""
    rounds = 18
    enc = SHA256CNFEncoder(rounds=rounds)
    gen = torch.Generator(device="cpu").manual_seed(181818)
    w16_batch = sample_random_w16(
        batch_size=10, mode="int_range", num_bits=32, device="cpu", generator=gen
    )
    traces = sha256_trace(w16_batch, rounds=rounds)

    # Zero-leakage API check
    assert "w0" not in inspect.signature(compute_algebraic_known_words_r18).parameters
    assert "x_star" not in inspect.signature(compute_algebraic_known_words_r18).parameters

    for i in range(10):
        w0_true = int(w16_batch[i, 0].item())
        dig_i = traces.digest[i]

        # Verify pure zero-leakage algebraic derivation matches exact ground-truth trace
        alg = compute_algebraic_known_words_r18(dig_i, domain_mode="int_range", num_bits=32)
        assert set(alg.keys()) == {
            (11, 0),
            (12, 0),
            (13, 0),
            (14, 0),
            (17, 2),
            (14, 1),
            (10, 0),
        }
        for (t, ch), val in alg.items():
            if ch == 0:
                assert val == int(traces.A_full[i, t + 3].item())
            elif ch == 1:
                assert val == int(traces.E_full[i, t + 3].item())
            elif ch == 2:
                assert val == int(traces.W_full[i, t].item())

        for include_alg in (False, True):
            unit_clauses, _ = enc.build_instance_unit_clauses(
                dig_i, domain_mode="int_range", num_bits=32, include_algebraic=include_alg
            )
            # 1. Fixing true W_0 must be SAT and recover exact internal A, E, W words bit-for-bit
            with Cadical195(bootstrap_with=enc.base_clauses) as solver:
                for uc in unit_clauses:
                    solver.add_clause(uc)
                assump_true = [
                    enc.W_vars[0][b] if ((w0_true >> b) & 1) else -enc.W_vars[0][b]
                    for b in range(32)
                ]
                assert solver.solve(assumptions=assump_true) is True
                model = solver.get_model()
                assert model is not None
                mset = set(model)
                assert enc.extract_w0_from_model(model) == w0_true
                for t in range(rounds):
                    assert enc.bits_to_word(enc.A_vars[t + 3], mset) == int(
                        traces.A_full[i, t + 3].item()
                    )
                    assert enc.bits_to_word(enc.E_vars[t + 3], mset) == int(
                        traces.E_full[i, t + 3].item()
                    )
                    assert enc.bits_to_word(enc.W_vars[t], mset) == int(
                        traces.W_full[i, t].item()
                    )

            # 2. Flipping 1 bit in W_0 (testing varied bit indices across 10 messages) -> UNSAT
            flip_bits = (0, 7, 15, 24, 31)
            b_flip = flip_bits[i % len(flip_bits)]
            w0_bad = w0_true ^ (1 << b_flip)
            # Sanity verify w0_bad doesn't produce an accidental collision (probability 2^-256)
            w16_bad = w16_batch[i : i + 1].clone()
            w16_bad[0, 0] = w0_bad
            assert not torch.equal(sha256_digest(w16_bad, rounds=rounds)[0], dig_i)

            with Cadical195(bootstrap_with=enc.base_clauses) as solver:
                for uc in unit_clauses:
                    solver.add_clause(uc)
                assump_bad = [
                    enc.W_vars[0][b] if ((w0_bad >> b) & 1) else -enc.W_vars[0][b]
                    for b in range(32)
                ]
                assert solver.solve(assumptions=assump_bad) is False
