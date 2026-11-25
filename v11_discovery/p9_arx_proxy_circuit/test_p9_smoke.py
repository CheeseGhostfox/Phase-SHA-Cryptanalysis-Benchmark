"""CPU-only fast unit & smoke checks for P9 ARX proxy circuit discovery."""

from __future__ import annotations

import os
import json
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import z3

from src.sha256_ref import expand_message_schedule, get_iv_tensor, sha256_trace
from v11_discovery.p9_arx_proxy_circuit.arx_operators import (
    OperatorSpec,
    evaluate_operator_targets,
    get_mini_arx_params,
    get_real_sha256_params,
    sample_input_words,
    words_to_packed_inputs,
)
from v11_discovery.p9_arx_proxy_circuit.circuit_slp import (
    AffineForm,
    AndGate,
    CircuitSLP,
    count_packed_matches,
    pack_bits_vector,
)
from v11_discovery.p9_arx_proxy_circuit.z3_verifier import build_z3_operator_target_bit, slp_to_z3_bool


def test_sha256_ref_alignment() -> None:
    """Verify Family A2 and Family B match src/sha256_ref.py bit-for-bit."""
    w16 = torch.randint(0, 1 << 32, (64, 16), dtype=torch.int64)
    trace2 = sha256_trace(w16, rounds=2)
    spec_a2 = OperatorSpec(family="A2", rounds=2, params=get_real_sha256_params(2))
    out_a2 = evaluate_operator_targets(spec_a2, w16[:, :2])
    for b in (0, 1, 7, 15, 23, 31):
        ref_a = (trace2.A_full[:, 2 + 3] >> b) & 1
        ref_e = (trace2.E_full[:, 2 + 3] >> b) & 1
        assert torch.equal(out_a2[f"a2[{b}]"], ref_a), f"Mismatch A2 a2[{b}]"
        assert torch.equal(out_a2[f"e2[{b}]"], ref_e), f"Mismatch A2 e2[{b}]"

    w_full = expand_message_schedule(w16, rounds=19)
    spec_b3 = OperatorSpec(family="B", rounds=3, params=get_real_sha256_params(3))
    out_b3 = evaluate_operator_targets(spec_b3, w16)
    for b in (0, 1, 7, 15, 23, 31):
        ref_w18 = (w_full[:, 18] >> b) & 1
        assert torch.equal(out_b3[f"W18[{b}]"], ref_w18), f"Mismatch B W18[{b}]"


def test_slp_packed_int_z3_alignment() -> None:
    """Verify bit-packed eval, scalar eval, and Z3 Bool compilation agree identically."""
    c = CircuitSLP(
        n_inputs=6,
        gates=[
            AndGate(AffineForm(0, (0, 1)), AffineForm(1, (2,))),
            AndGate(AffineForm(0, (6, 3)), AffineForm(0, (4, 5))),
        ],
        output=AffineForm(1, (0, 7)),
        name="test_c",
    )
    ok, msg, deg = c.validate()
    assert ok and deg == 3, f"Unexpected validate: {ok}, {msg}, {deg}"

    # Exhaustive 2^6 = 64 states packed into 1 int64 word!
    x_mat = torch.zeros((6, 64), dtype=torch.int64)
    for idx in range(64):
        for bit in range(6):
            x_mat[bit, idx] = (idx >> bit) & 1

    from v11_discovery.p9_arx_proxy_circuit.circuit_slp import pack_bits_matrix

    packed = pack_bits_matrix(x_mat)
    pred_packed = c.eval_packed(packed)

    z3_vars = [z3.Bool(f"x_{i}") for i in range(6)]
    z3_expr = slp_to_z3_bool(c, z3_vars)

    for idx in range(64):
        x_bits = [int(x_mat[b, idx].item()) for b in range(6)]
        v_int = c.eval_int(x_bits)
        v_pack = int((int(pred_packed[0].item()) >> idx) & 1)
        assert v_int == v_pack, f"Packed/int mismatch at {idx}: {v_int} != {v_pack}"

        s = z3.Solver()
        for b in range(6):
            s.add(z3_vars[b] == z3.BoolVal(bool(x_bits[b])))
        s.add(z3_expr)
        v_z3 = 1 if s.check() == z3.sat else 0
        assert v_int == v_z3, f"Z3 mismatch at {idx}"


def test_z3_mini_arx_alignment() -> None:
    """Verify Z3 BitVec Mini-ARX matches arx_operators.py on random vectors."""
    spec = OperatorSpec(family="C4", rounds=2, params=get_mini_arx_params(4, 2))
    words = sample_input_words(spec, 64, seed=99)
    targets = evaluate_operator_targets(spec, words)
    tname = "a2[1]"

    z3_vars = [z3.Bool(f"x_{i}") for i in range(spec.n_inputs)]
    z3_bit = build_z3_operator_target_bit(spec, tname, z3_vars)

    for row in range(8):
        s = z3.Solver()
        for w_i in range(spec.input_words_count):
            val = int(words[row, w_i].item())
            for b in range(4):
                s.add(z3_vars[w_i * 4 + b] == z3.BoolVal(bool((val >> b) & 1)))
        s.add(z3_bit)
        v_z3 = 1 if s.check() == z3.sat else 0
        v_ref = int(targets[tname][row].item())
        assert v_z3 == v_ref, f"Z3 Mini-ARX mismatch at row {row}: {v_z3} != {v_ref}"


if __name__ == "__main__":
    test_sha256_ref_alignment()
    print("PASS: test_sha256_ref_alignment")
    test_slp_packed_int_z3_alignment()
    print("PASS: test_slp_packed_int_z3_alignment")
    test_z3_mini_arx_alignment()
    print("PASS: test_z3_mini_arx_alignment")
