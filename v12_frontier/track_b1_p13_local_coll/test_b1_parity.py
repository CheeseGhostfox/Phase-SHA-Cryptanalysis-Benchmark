"""CPU parity & self-check suite for P13 generalized differential engine & SAT witness solver.

Run strictly on CPU:
  CUDA_VISIBLE_DEVICES="" g:\\SHA-LM\\.venv\\Scripts\\python.exe -u test_b1_parity.py
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parents[1]))

from generalized_diff import (  # noqa: E402
    GD_EQ,
    GD_ANY,
    GD_X,
    btc_double_sha256_ref,
    encode_pair_to_gdiff,
    forward_propagate_gdiff,
    gdiff_add2,
    gdiff_add_many_exact_carry,
    gdiff_ch,
    gdiff_maj,
    gdiff_sigma0_lower,
    gdiff_sigma0_upper,
    gdiff_sigma1_lower,
    gdiff_sigma1_upper,
    sha256_pair_exact_check,
    sha256_trace_custom_iv,
    verify_pair_against_gdiff,
)
from sat_witness import CNFBuilder, encode_sha256_trace, solve_differential_witness  # noqa: E402
from src.sha256_ref import (  # noqa: E402
    MASK32,
    SHA256_IV,
    digest_words_to_bytes,
    pad_bytes_to_w16,
    sha256_digest,
)


def test_fp32_locked() -> None:
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    print("[PASS] pure FP32 locked (allow_tf32=False)", flush=True)


def test_exhaustive_gdiff_gates() -> None:
    """Exhaustively check 16-state Ch/Maj/ADD2/ADD_many soundness on concrete pairs."""
    rng = random.Random(20261125)
    for _ in range(256):
        a = torch.randint(0, 1 << 32, (64,), dtype=torch.int64)
        ap = torch.randint(0, 1 << 32, (64,), dtype=torch.int64)
        b = torch.randint(0, 1 << 32, (64,), dtype=torch.int64)
        bp = torch.randint(0, 1 << 32, (64,), dtype=torch.int64)
        c = torch.randint(0, 1 << 32, (64,), dtype=torch.int64)
        cp = torch.randint(0, 1 << 32, (64,), dtype=torch.int64)

        ma = encode_pair_to_gdiff(a, ap)
        mb = encode_pair_to_gdiff(b, bp)
        mc = encode_pair_to_gdiff(c, cp)

        # Relax random bits to '?' (GD_ANY)
        relax = torch.randint(0, 2, (64, 32), dtype=torch.int64) == 1
        ma_r = torch.where(relax, torch.full_like(ma, GD_ANY), ma)

        # Exact reference operations
        from src.sha256_ref import ch32, maj32, sigma0_lower, sigma0_upper, sigma1_lower, sigma1_upper

        assert verify_pair_against_gdiff(ch32(a, b, c), ch32(ap, bp, cp), gdiff_ch(ma_r, mb, mc)).all()
        assert verify_pair_against_gdiff(maj32(a, b, c), maj32(ap, bp, cp), gdiff_maj(ma_r, mb, mc)).all()
        assert verify_pair_against_gdiff(sigma0_upper(a), sigma0_upper(ap), gdiff_sigma0_upper(ma_r)).all()
        assert verify_pair_against_gdiff(sigma1_upper(a), sigma1_upper(ap), gdiff_sigma1_upper(ma_r)).all()
        assert verify_pair_against_gdiff(sigma0_lower(a), sigma0_lower(ap), gdiff_sigma0_lower(ma_r)).all()
        assert verify_pair_against_gdiff(sigma1_lower(a), sigma1_lower(ap), gdiff_sigma1_lower(ma_r)).all()
        assert verify_pair_against_gdiff((a + b) & MASK32, (ap + bp) & MASK32, gdiff_add2(ma_r, mb)).all()

        sum5 = (a + b + c + a + b) & MASK32
        sum5p = (ap + bp + cp + ap + bp) & MASK32
        out5, _ = gdiff_add_many_exact_carry([ma_r, mb, mc, ma_r, mb])
        assert verify_pair_against_gdiff(sum5, sum5p, out5).all()
    print("[PASS] generalized 16-state gate & exact-carry ADD5 soundness (16,384 pairs)", flush=True)


def test_forward_gdiff_vs_sha256_ref() -> None:
    """Verify full-round forward_propagate_gdiff on concrete + partially relaxed traces vs sha256_ref."""
    gen = torch.Generator().manual_seed(1337)
    for rounds in (4, 8, 16, 32, 64):
        w1 = torch.randint(0, 1 << 32, (32, 16), dtype=torch.int64, generator=gen)
        w2 = w1.clone()
        # Inject sparse differences in W[0], W[1]
        w2[:, 0] ^= 0x80000001
        w2[:, 1] ^= 0x00010000

        pair = sha256_pair_exact_check(w1, w2, rounds=rounds)
        w16_mask = encode_pair_to_gdiff(w1, w2)
        # Also relax W[2][0..7] to '?'
        w16_mask[:, 2, :8] = GD_ANY
        gd = forward_propagate_gdiff(w16_mask, rounds=rounds)
        assert not gd.contradictions.any(), f"Contradiction at R={rounds}"
        assert verify_pair_against_gdiff(pair["tr"].digest, pair["tr_p"].digest, gd.digest).all()
        assert verify_pair_against_gdiff(pair["tr"].A_full, pair["tr_p"].A_full, gd.A_full).all()
        assert verify_pair_against_gdiff(pair["tr"].E_full, pair["tr_p"].E_full, gd.E_full).all()
    print("[PASS] forward_propagate_gdiff exact containment vs sha256_ref (R=4..64)", flush=True)


def test_btc_double_sha256_parity() -> None:
    """Verify btc_double_sha256_ref matches hashlib.sha256(hashlib.sha256(header))."""
    rng = random.Random(99)
    headers = []
    expected = []
    for _ in range(16):
        raw = bytes(rng.getrandbits(8) for _ in range(80))
        words = [int.from_bytes(raw[4 * i : 4 * (i + 1)], "big") for i in range(20)]
        headers.append(words)
        h1 = hashlib.sha256(raw).digest()
        h2 = hashlib.sha256(h1).digest()
        expected.append(h2)
    h_tensor = torch.tensor(headers, dtype=torch.int64)
    _, h2_tensor = btc_double_sha256_ref(h_tensor)
    for i in range(16):
        got = digest_words_to_bytes(h2_tensor[i])
        assert got == expected[i], f"BTC Double-SHA256 mismatch at idx={i}"
    print("[PASS] BTC 80-byte Double-SHA256 exact parity vs hashlib", flush=True)


def test_sat_single_trace_parity_r64() -> None:
    """Verify SAT Tseitin SHA-256 encoder on full R=64 produces xor_diff == 0 vs sha256_ref."""
    msg = b"v12-p13-sat-parity-check"
    w16_ref = pad_bytes_to_w16(msg, device="cpu")
    expected_digest = sha256_digest(w16_ref.unsqueeze(0), rounds=64)[0]

    cnf = CNFBuilder("cadical195")
    zero_var = cnf.new_var()
    cnf.fix_bit(zero_var, 0)
    tr = encode_sha256_trace(cnf, rounds=64, zero_var=zero_var)
    for i in range(8):
        cnf.fix_word(tr.iv_vars[i], SHA256_IV[i])
    for i in range(16):
        cnf.fix_word(tr.w16_vars[i], int(w16_ref[i].item()))

    assert cnf.solver.solve() is True
    mset = {lit for lit in cnf.solver.get_model() if lit > 0}
    sat_digest = torch.tensor(
        [cnf.decode_word(mset, tr.digest_vars[i]) for i in range(8)],
        dtype=torch.int64,
    )
    cnf.delete()

    xor_diff = int((sat_digest ^ expected_digest).sum().item())
    assert xor_diff == 0, f"SAT R=64 xor_diff={xor_diff}"
    print("[PASS] SAT Tseitin R=64 forward parity vs sha256_ref (xor_diff == 0)", flush=True)


def test_sat_differential_witness_smoke() -> None:
    """Solve a concrete reduced-round collision & truncated leading-zero witness via CaDiCaL195."""
    # 1. R=8 Semi-Free-Start collision witness (digest_xor_zero_words = 0..7)
    w_sfs = solve_differential_witness(
        rounds=8,
        iv_mode="semi_free",
        digest_xor_zero_words=list(range(8)),
        require_nonzero_msg_diff=True,
        max_solutions=1,
    )
    assert len(w_sfs) == 1 and w_sfs[0].sat
    res = w_sfs[0]
    tr1 = sha256_trace_custom_iv(torch.tensor([res.w16], dtype=torch.int64), iv=torch.tensor(res.iv, dtype=torch.int64), rounds=8)
    tr2 = sha256_trace_custom_iv(torch.tensor([res.w16_prime], dtype=torch.int64), iv=torch.tensor(res.iv_prime, dtype=torch.int64), rounds=8)
    xor_digest = (tr1.digest[0] ^ tr2.digest[0]).tolist()
    assert all(x == 0 for x in xor_digest), f"R=8 SFS collision verify failed: {xor_digest}"
    assert res.w16 != res.w16_prime
    print(
        f"[PASS] SAT R=8 SFS collision witness verified vs sha256_ref in {res.solve_seconds:.3f}s "
        f"(vars={res.num_vars}, clauses={res.num_clauses})",
        flush=True,
    )

    # 2. R=16 standard-IV BTC Block 2 truncated leading-zero (clz >= 8) differential pair
    # Fix W[0..3] with diff only in W[3][0], and require H[0] & H'[0] top 8 bits == 0
    w_gdiff = {
        0: [GD_EQ] * 32,
        1: [GD_EQ] * 32,
        2: [GD_EQ] * 32,
        3: [GD_X] + [GD_EQ] * 31,
    }
    w_btc = solve_differential_witness(
        rounds=16,
        iv_mode="standard",
        w_gdiff=w_gdiff,
        leading_zero_bits=8,
        btc_block2_padding=True,
        require_nonzero_msg_diff=True,
        max_solutions=2,
    )
    assert len(w_btc) == 2 and all(x.sat for x in w_btc)
    for idx, r in enumerate(w_btc):
        d1 = sha256_digest(torch.tensor([r.w16], dtype=torch.int64), rounds=16)[0]
        d2 = sha256_digest(torch.tensor([r.w16_prime], dtype=torch.int64), rounds=16)[0]
        assert d1.tolist() == r.digest
        assert d2.tolist() == r.digest_prime
        assert (d1[0].item() >> 24) == 0 and (d2[0].item() >> 24) == 0
        assert (r.w16[3] ^ r.w16_prime[3]) == 1
    assert w_btc[0].w16 != w_btc[1].w16
    print(
        f"[PASS] SAT R=16 BTC-padded clz>=8 differential witnesses (x2) verified in {w_btc[-1].solve_seconds:.3f}s",
        flush=True,
    )


def main() -> None:
    test_fp32_locked()
    test_exhaustive_gdiff_gates()
    test_forward_gdiff_vs_sha256_ref()
    test_btc_double_sha256_parity()
    test_sat_single_trace_parity_r64()
    test_sat_differential_witness_smoke()
    summary = {"all_passed": True, "solver": "cadical195", "fp32_locked": True}
    out_dir = _THIS_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parity_smoke.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[ALL PASS] P13 parity smoke saved to results/parity_smoke.json", flush=True)


if __name__ == "__main__":
    main()
