"""Exact chi=2 MPO ARX operators, gate-by-gate MPS-TEBD, exact SHA-256 (with full W-schedule),
and exact Schmidt / entanglement spectrum diagnostics (pure IEEE-754 FP32).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import torch

from src.sha256_ref import MASK32, SHA256_IV, SHA256_K, sha256_trace


def enforce_fp32_lock() -> Dict[str, bool]:
    """Enforce strict IEEE-754 FP32 with TF32 disabled."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    }


def _rotr32(x: torch.Tensor, r: int) -> torch.Tensor:
    xu = x & MASK32
    rot = r % 32
    if rot == 0:
        return xu
    return ((xu >> rot) | ((xu & ((1 << rot) - 1)) << (32 - rot))) & MASK32


def _shr32(x: torch.Tensor, r: int) -> torch.Tensor:
    return (x & MASK32) >> r


def make_cipher_profile(ctrl: bool = False, seed: int = 20261125) -> Dict[str, Tuple[int, ...]]:
    """Standard SHA-256 profile vs isomorphic random-rotation/random-constant clone."""
    if not ctrl:
        return {
            "s0": (2, 13, 22),
            "s1": (6, 11, 25),
            "w0": (7, 18, 3),
            "w1": (17, 19, 10),
            "k": tuple(SHA256_K),
        }
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    r0 = torch.randperm(31, generator=g)[:3] + 1
    r1 = torch.randperm(31, generator=g)[:3] + 1
    k_ctrl = torch.randint(0, 1 << 32, (64,), dtype=torch.int64, generator=g).tolist()
    return {
        "s0": tuple(int(v) for v in r0.tolist()),
        "s1": tuple(int(v) for v in r1.tolist()),
        "w0": (7, 18, 3),
        "w1": (17, 19, 10),
        "k": tuple(int(v) for v in k_ctrl),
    }


def expand_sha256_schedule(
    w16: torch.Tensor,
    num_rounds: int,
    w0_params: Tuple[int, int, int] = (7, 18, 3),
    w1_params: Tuple[int, int, int] = (17, 19, 10),
) -> torch.Tensor:
    """Exact SHA-256 message schedule expansion: (B, 16) -> (B, max(16, num_rounds))."""
    B = w16.shape[0]
    total = max(16, int(num_rounds))
    W = [w16[:, t] & MASK32 for t in range(16)]
    for t in range(16, total):
        w_m15 = W[t - 15]
        w_m2 = W[t - 2]
        s0 = _rotr32(w_m15, w0_params[0]) ^ _rotr32(w_m15, w0_params[1]) ^ _shr32(w_m15, w0_params[2])
        s1 = _rotr32(w_m2, w1_params[0]) ^ _rotr32(w_m2, w1_params[1]) ^ _shr32(w_m2, w1_params[2])
        W.append((W[t - 16] + s0 + W[t - 7] + s1) & MASK32)
    return torch.stack(W[:total], dim=1)


def sha256_round_step(
    s8: torch.Tensor,
    w_t: torch.Tensor,
    k_t: int | torch.Tensor,
    s0_rots: Tuple[int, int, int] = (2, 13, 22),
    s1_rots: Tuple[int, int, int] = (6, 11, 25),
) -> torch.Tensor:
    """Exact 1-round SHA-256 compression step: (B, 8) -> (B, 8)."""
    a, b, c, d, e, f, g, h = [s8[:, i] & MASK32 for i in range(8)]
    sig1 = _rotr32(e, s1_rots[0]) ^ _rotr32(e, s1_rots[1]) ^ _rotr32(e, s1_rots[2])
    ch = ((e & f) ^ ((~e & MASK32) & g)) & MASK32
    sig0 = _rotr32(a, s0_rots[0]) ^ _rotr32(a, s0_rots[1]) ^ _rotr32(a, s0_rots[2])
    maj = ((a & b) ^ (a & c) ^ (b & c)) & MASK32

    k_val = k_t if isinstance(k_t, torch.Tensor) else int(k_t)
    t1 = (h + sig1 + ch + k_val + (w_t & MASK32)) & MASK32
    t2 = (sig0 + maj) & MASK32
    new_e = (d + t1) & MASK32
    new_a = (t1 + t2) & MASK32
    return torch.stack([new_a, a, b, c, new_e, e, f, g], dim=1)


def sha256_run_rounds(
    s0: torch.Tensor,
    w16: torch.Tensor,
    num_rounds: int,
    prof: Dict[str, Tuple[int, ...]] | None = None,
    return_mid_round: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Run exact SHA-256 for num_rounds with full message schedule expansion."""
    if prof is None:
        prof = make_cipher_profile(ctrl=False)
    W = expand_sha256_schedule(w16, num_rounds, prof["w0"], prof["w1"])
    cur = s0 & MASK32
    mid_state = None
    for r in range(int(num_rounds)):
        cur = sha256_round_step(cur, W[:, r], prof["k"][r % 64], prof["s0"], prof["s1"])
        if return_mid_round is not None and (r + 1) == int(return_mid_round):
            mid_state = cur.clone()
    return cur, mid_state


def verify_sha256_exact_64(device: torch.device) -> int:
    """Verify bit-exact agreement with src.sha256_ref.sha256_trace across full 64 rounds."""
    gen = torch.Generator(device=device).manual_seed(20261125)
    w16 = torch.randint(0, 1 << 32, (32, 16), dtype=torch.int64, device=device, generator=gen)
    ref = sha256_trace(w16, rounds=64)
    s0 = torch.tensor(SHA256_IV, dtype=torch.int64, device=device).unsqueeze(0).expand(32, -1).clone()
    s64, _ = sha256_run_rounds(s0, w16, num_rounds=64, prof=make_cipher_profile(ctrl=False))
    return int((ref.S_R ^ s64).max().item())


# ==============================================================================
# PART 1: Exact chi=2 MPO Arithmetic & Gate-by-Gate MPS-TEBD
# ==============================================================================

def build_add_const_mpo(n_bits: int, const_val: int, device: torch.device) -> List[torch.Tensor]:
    """Exact chi=2 unitary MPO for x -> (x + const_val) mod 2^n_bits (LSB-first).

    Each site tensor W[j] has shape (chi_l, chi_r, d_out=2, d_in=2).
    """
    mpo: List[torch.Tensor] = []
    for j in range(n_bits):
        k_j = (int(const_val) >> j) & 1
        chi_l = 1 if j == 0 else 2
        chi_r = 1 if j == (n_bits - 1) else 2
        W = torch.zeros((chi_l, chi_r, 2, 2), dtype=torch.float32, device=device)
        for c_in in range(chi_l):
            for x_j in (0, 1):
                s = x_j + k_j + c_in
                z_j = s & 1
                c_out = (s >> 1) & 1
                if j == n_bits - 1:
                    W[c_in, 0, z_j, x_j] += 1.0
                else:
                    W[c_in, c_out, z_j, x_j] += 1.0
        mpo.append(W)
    return mpo


def build_xor_const_mpo(n_bits: int, const_val: int, device: torch.device) -> List[torch.Tensor]:
    """Exact chi=1 unitary MPO for x -> (x ^ const_val) (LSB-first)."""
    mpo: List[torch.Tensor] = []
    for j in range(n_bits):
        k_j = (int(const_val) >> j) & 1
        W = torch.zeros((1, 1, 2, 2), dtype=torch.float32, device=device)
        for x_j in (0, 1):
            W[0, 0, x_j ^ k_j, x_j] = 1.0
        mpo.append(W)
    return mpo


def build_reversible_2reg_add_mpo(n_bits: int, device: torch.device) -> List[torch.Tensor]:
    """Exact chi=2 unitary MPO for 2-register reversible addition (x, y) -> (x, (x + y) mod 2^n).

    Interleaved site encoding: physical index p = 2 * x_j + y_j in {0, 1, 2, 3} (d=4).
    """
    mpo: List[torch.Tensor] = []
    for j in range(n_bits):
        chi_l = 1 if j == 0 else 2
        chi_r = 1 if j == (n_bits - 1) else 2
        W = torch.zeros((chi_l, chi_r, 4, 4), dtype=torch.float32, device=device)
        for c_in in range(chi_l):
            for x_j in (0, 1):
                for y_j in (0, 1):
                    p_in = 2 * x_j + y_j
                    s = x_j + y_j + c_in
                    z_j = s & 1
                    c_out = (s >> 1) & 1
                    p_out = 2 * x_j + z_j
                    if j == n_bits - 1:
                        W[c_in, 0, p_out, p_in] += 1.0
                    else:
                        W[c_in, c_out, p_out, p_in] += 1.0
        mpo.append(W)
    return mpo


def build_reversible_2reg_xor_mpo(n_bits: int, device: torch.device) -> List[torch.Tensor]:
    """Exact chi=1 unitary MPO for 2-register reversible XOR (x, y) -> (x, x ^ y) on d=4 sites."""
    mpo: List[torch.Tensor] = []
    for _ in range(n_bits):
        W = torch.zeros((1, 1, 4, 4), dtype=torch.float32, device=device)
        for x_j in (0, 1):
            for y_j in (0, 1):
                W[0, 0, 2 * x_j + (x_j ^ y_j), 2 * x_j + y_j] = 1.0
        mpo.append(W)
    return mpo


def mpo_adjoint(mpo: List[torch.Tensor]) -> List[torch.Tensor]:
    """Exact algebraic inverse U^T for any real unitary MPO."""
    return [W.transpose(2, 3).contiguous() for W in mpo]


def product_state_mps(
    n_sites: int,
    site_vecs: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    """Create a chi=1 MPS from a list of (d,) site vectors."""
    assert len(site_vecs) == n_sites
    return [v.reshape(1, -1, 1).to(torch.float32).contiguous() for v in site_vecs]


def int_to_mps_1reg(val: int, n_bits: int, device: torch.device) -> List[torch.Tensor]:
    vecs = []
    for j in range(n_bits):
        v = torch.zeros(2, dtype=torch.float32, device=device)
        v[(int(val) >> j) & 1] = 1.0
        vecs.append(v)
    return product_state_mps(n_bits, vecs)


def int_pair_to_mps_2reg(x: int, y: int, n_bits: int, device: torch.device) -> List[torch.Tensor]:
    vecs = []
    for j in range(n_bits):
        v = torch.zeros(4, dtype=torch.float32, device=device)
        p = 2 * ((int(x) >> j) & 1) + ((int(y) >> j) & 1)
        v[p] = 1.0
        vecs.append(v)
    return product_state_mps(n_bits, vecs)


def uniform_superposition_mps(n_sites: int, d: int, device: torch.device) -> List[torch.Tensor]:
    v = torch.full((d,), 1.0 / math.sqrt(d), dtype=torch.float32, device=device)
    return product_state_mps(n_sites, [v.clone() for _ in range(n_sites)])


def mps_compress(mps: List[torch.Tensor], chi_max: int, cutoff: float = 1e-7) -> Tuple[List[torch.Tensor], float, int]:
    """Right-canonical QR sweep + left-to-right SVD truncation up to chi_max.

    Returns (compressed_mps, total_trunc_err_sq, max_bond_dim).
    """
    n = len(mps)
    tensors = [t.clone() for t in mps]
    # Right-to-left QR canonicalization
    for j in range(n - 1, 0, -1):
        cl, d, cr = tensors[j].shape
        mat = tensors[j].reshape(cl, d * cr).transpose(0, 1)  # (d*cr, cl)
        q, r = torch.linalg.qr(mat, mode="reduced")  # q: (d*cr, k), r: (k, cl)
        k = q.shape[1]
        tensors[j] = q.transpose(0, 1).reshape(k, d, cr)
        tensors[j - 1] = torch.einsum("abc,dc->abd", tensors[j - 1], r)

    # Left-to-right SVD compression
    trunc_err_sq = 0.0
    max_chi = 1
    for j in range(n - 1):
        cl, d, cr = tensors[j].shape
        mat = tensors[j].reshape(cl * d, cr)
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        keep_mask = s > cutoff
        k = int(keep_mask.sum().item())
        k = max(1, min(k, int(chi_max), int(s.shape[0])))
        if k < s.shape[0]:
            trunc_err_sq += float((s[k:] ** 2).sum().item())
        u_k = u[:, :k]
        s_k = s[:k]
        vh_k = vh[:k, :]
        tensors[j] = u_k.reshape(cl, d, k)
        sv = s_k.unsqueeze(1) * vh_k
        tensors[j + 1] = torch.einsum("ab,bcd->acd", sv, tensors[j + 1])
        if k > max_chi:
            max_chi = k
    return tensors, trunc_err_sq, max_chi


def apply_mpo_to_mps(
    mps: List[torch.Tensor],
    mpo: List[torch.Tensor],
    chi_max: int = 64,
) -> Tuple[List[torch.Tensor], float, int]:
    """Apply MPO to MPS and compress via exact canonical SVD."""
    n = len(mps)
    raw: List[torch.Tensor] = []
    for j in range(n):
        # mps[j]: (cl, d_in, cr), mpo[j]: (wl, wr, d_out, d_in)
        c = torch.einsum("abc,efdb->aedcf", mps[j], mpo[j])
        cl, wl, d_out, cr, wr = c.shape
        raw.append(c.reshape(cl * wl, d_out, cr * wr))
    return mps_compress(raw, chi_max=chi_max)


def apply_2site_gate_mps(
    mps: List[torch.Tensor],
    j: int,
    gate: torch.Tensor,
    chi_max: int,
) -> Tuple[List[torch.Tensor], float]:
    """Apply 2-site unitary gate of shape (d, d, d, d) to adjacent sites (j, j+1) and SVD compress."""
    a = mps[j]
    b = mps[j + 1]
    cl, d, _ = a.shape
    _, _, cr = b.shape
    theta = torch.einsum("abc,cde->abde", a, b)  # (cl, d, d, cr)
    theta_new = torch.einsum("xybd,abde->axye", gate, theta).reshape(cl * d, d * cr)
    u, s, vh = torch.linalg.svd(theta_new, full_matrices=False)
    k = max(1, min(int((s > 1e-7).sum().item()), int(chi_max), int(s.shape[0])))
    trunc_err_sq = float((s[k:] ** 2).sum().item()) if k < s.shape[0] else 0.0
    mps[j] = u[:, :k].reshape(cl, d, k)
    mps[j + 1] = (s[:k].unsqueeze(1) * vh[:k, :]).reshape(k, d, cr)
    return mps, trunc_err_sq


def build_swap_gate(d: int, device: torch.device) -> torch.Tensor:
    g = torch.zeros((d, d, d, d), dtype=torch.float32, device=device)
    for p in range(d):
        for q in range(d):
            g[q, p, p, q] = 1.0
    return g


def build_subreg_swap_gate_y(device: torch.device) -> torch.Tensor:
    """Swap only the y-subregister (leaving x untouched) between two adjacent d=4 sites."""
    g = torch.zeros((4, 4, 4, 4), dtype=torch.float32, device=device)
    for x1 in (0, 1):
        for y1 in (0, 1):
            for x2 in (0, 1):
                for y2 in (0, 1):
                    p1 = 2 * x1 + y1
                    p2 = 2 * x2 + y2
                    q1 = 2 * x1 + y2
                    q2 = 2 * x2 + y1
                    g[q1, q2, p1, p2] = 1.0
    return g


def apply_rotr_y_mps(
    mps: List[torch.Tensor],
    rot: int,
    chi_max: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], float]:
    """Rotate only the y sub-register right by `rot` bits via exact adjacent SWAP gates."""
    n = len(mps)
    r = int(rot) % n
    if r == 0:
        return mps, 0.0
    swap_y = build_subreg_swap_gate_y(device)
    total_err = 0.0
    # Right-rotate by 1 (LSB-first): bit 0 moves to bit n-1, bit j moves to j-1
    # Equivalently, sweep SWAP(0,1), SWAP(1,2), ..., SWAP(n-2, n-1) r times
    for _ in range(r):
        for j in range(n - 1):
            mps, e = apply_2site_gate_mps(mps, j, swap_y, chi_max=chi_max)
            total_err += e
    return mps, total_err


def apply_swap_xy_local_mps(mps: List[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    """Local chi=1 Feistel branch swap (x, y) -> (y, x) on every d=4 site."""
    m = torch.zeros((4, 4), dtype=torch.float32, device=device)
    for x in (0, 1):
        for y in (0, 1):
            m[2 * y + x, 2 * x + y] = 1.0
    return [torch.einsum("oi,aib->aob", m, t) for t in mps]


def mps_inner(mps_a: List[torch.Tensor], mps_b: List[torch.Tensor]) -> float:
    """Exact inner product <a|b> between two MPS."""
    env = torch.ones((1, 1), dtype=torch.float32, device=mps_a[0].device)
    for ta, tb in zip(mps_a, mps_b):
        # env: (al, bl), ta: (al, d, ar), tb: (bl, d, br) -> (ar, br)
        env = torch.einsum("ab,adc,bde->ce", env, ta, tb)
    return float(env[0, 0].item())


def mps_norm(mps: List[torch.Tensor]) -> float:
    return math.sqrt(max(0.0, mps_inner(mps, mps)))


def mps_decode_basis_int_2reg(mps: List[torch.Tensor]) -> Tuple[int, int, float]:
    """Read out argmax (x, y) and max probability from a basis-state 2-register MPS."""
    n = len(mps)
    x_val = 0
    y_val = 0
    # Exact marginal per site (for basis states, each site has a single peak)
    vec_l = torch.ones((1,), dtype=torch.float32, device=mps[0].device)
    # Use sequential contraction for exact basis readout (assuming chi=1 or near-basis)
    # First compress to get clean norms
    tensors, _, _ = mps_compress(mps, chi_max=8)
    # Compute exact site reduced density diagonals
    right_envs: List[torch.Tensor] = [torch.ones((1, 1), dtype=torch.float32, device=mps[0].device)]
    for j in range(n - 1, 0, -1):
        t = tensors[j]
        re = torch.einsum("adc,bde,ce->ab", t, t, right_envs[-1])
        right_envs.append(re)
    right_envs.reverse()

    left_env = torch.ones((1, 1), dtype=torch.float32, device=mps[0].device)
    min_prob = 1.0
    for j in range(n):
        t = tensors[j]
        # rho_diag[p] = sum_{a,b,c,e} left_env[a,b] t[a,p,c] t[b,p,e] right_env[c,e]
        diag = torch.einsum("ab,apc,bpe,ce->p", left_env, t, t, right_envs[j])
        diag = diag / (diag.sum() + 1e-12)
        p_idx = int(torch.argmax(diag).item())
        min_prob = min(min_prob, float(diag[p_idx].item()))
        x_val |= ((p_idx >> 1) & 1) << j
        y_val |= (p_idx & 1) << j
        left_env = torch.einsum("ab,adc,bde->ce", left_env, t, t)
    return x_val, y_val, min_prob


def verify_exact_mpo_primitives(device: torch.device) -> Dict[str, Any]:
    """Verify 0-error exact chi=2 MPO ADD and chi=1 XOR across n=4, 8, 16, 32 bits."""
    gen = torch.Generator(device="cpu").manual_seed(20261125)
    summary: Dict[str, Any] = {}
    for n_bits in (4, 8, 16, 32):
        mask = (1 << n_bits) - 1
        mpo_add = build_reversible_2reg_add_mpo(n_bits, device)
        mpo_add_inv = mpo_adjoint(mpo_add)
        mpo_xor = build_reversible_2reg_xor_mpo(n_bits, device)

        max_bit_mismatch = 0
        max_inv_err = 0.0
        max_bond_seen = 1
        n_trials = 8
        for _ in range(n_trials):
            x = int(torch.randint(0, 1 << min(n_bits, 30), (1,), generator=gen).item()) & mask
            y = int(torch.randint(0, 1 << min(n_bits, 30), (1,), generator=gen).item()) & mask
            if n_bits == 32:
                x = (x << 2) | int(torch.randint(0, 4, (1,), generator=gen).item())
                y = (y << 2) | int(torch.randint(0, 4, (1,), generator=gen).item())

            psi0 = int_pair_to_mps_2reg(x, y, n_bits, device)
            psi_add, err_add, chi_add = apply_mpo_to_mps(psi0, mpo_add, chi_max=8)
            x_out, y_out, _ = mps_decode_basis_int_2reg(psi_add)
            if x_out != x or y_out != ((x + y) & mask):
                max_bit_mismatch += 1

            psi_xor, _, _ = apply_mpo_to_mps(psi_add, mpo_xor, chi_max=8)
            x_x, y_x, _ = mps_decode_basis_int_2reg(psi_xor)
            if x_x != x or y_x != (x ^ ((x + y) & mask)):
                max_bit_mismatch += 1

            # Exact algebraic inverse U^T
            psi_rec, _, _ = apply_mpo_to_mps(psi_add, mpo_add_inv, chi_max=8)
            overlap = abs(mps_inner(psi0, psi_rec))
            max_inv_err = max(max_inv_err, abs(1.0 - overlap))
            max_bond_seen = max(max_bond_seen, chi_add)

        summary[str(n_bits)] = {
            "mpo_add_bond_dim": 2,
            "mpo_xor_bond_dim": 1,
            "trials": n_trials,
            "bit_mismatches": max_bit_mismatch,
            "max_inverse_overlap_err": round(max_inv_err, 8),
            "post_compression_max_chi_on_basis": max_bond_seen,
        }
    return summary


# ==============================================================================
# PART 2: Exact Statevector Entanglement Spectrum & MPS Compression Analysis
# ==============================================================================

def analyze_state_entanglement(
    psi: torch.Tensor,
    n_qubits: int,
    chi_list: Sequence[int] = (2, 4, 8, 16, 32, 64),
) -> Dict[str, Any]:
    """Compute exact Schmidt spectrum, von Neumann / Renyi-2 entropy, chi_99, and MPS fidelity F(chi)."""
    psi_f = psi.reshape(-1).to(torch.float32)
    norm = float(psi_f.norm().item())
    if norm < 1e-12:
        raise ValueError("Zero-norm wavefunction passed to analyze_state_entanglement")
    psi_f = psi_f / norm

    vn_per_cut: List[float] = []
    renyi2_per_cut: List[float] = []
    chi99_per_cut: List[int] = []
    chi999_per_cut: List[int] = []

    for b in range(1, n_qubits):
        mat = psi_f.reshape(1 << b, 1 << (n_qubits - b))
        s = torch.linalg.svdvals(mat)
        p = (s ** 2).clamp(min=0.0)
        p = p / (p.sum() + 1e-12)
        vn = float(-(p * torch.log2(p.clamp(min=1e-12))).sum().item())
        r2 = float(-torch.log2((p ** 2).sum().clamp(min=1e-12)).item())
        csum = torch.cumsum(p, dim=0)
        c99 = int((csum < 0.99).sum().item()) + 1
        c999 = int((csum < 0.999).sum().item()) + 1
        vn_per_cut.append(vn)
        renyi2_per_cut.append(r2)
        chi99_per_cut.append(min(c99, int(s.shape[0])))
        chi999_per_cut.append(min(c999, int(s.shape[0])))

    mid_idx = (n_qubits // 2) - 1
    # Exact sequential MPS-SVD truncation fidelity F(chi) = |<psi | psi_chi>|^2
    fidelity_by_chi: Dict[str, float] = {}
    for chi_max in chi_list:
        psi_rec = mps_truncate_statevec(psi_f, n_qubits=n_qubits, chi_max=int(chi_max))
        rec_norm = float(psi_rec.norm().item())
        if rec_norm < 1e-12:
            fid = 0.0
        else:
            ov = float(torch.dot(psi_f, psi_rec / rec_norm).item())
            fid = ov * ov
        fidelity_by_chi[str(chi_max)] = round(fid, 6)

    return {
        "mid_vn_entropy_bits": round(vn_per_cut[mid_idx], 5),
        "max_vn_entropy_bits": round(max(vn_per_cut), 5),
        "mean_vn_entropy_bits": round(sum(vn_per_cut) / len(vn_per_cut), 5),
        "mid_renyi2_bits": round(renyi2_per_cut[mid_idx], 5),
        "mid_chi99": int(chi99_per_cut[mid_idx]),
        "max_chi99": int(max(chi99_per_cut)),
        "mid_chi999": int(chi999_per_cut[mid_idx]),
        "fidelity_by_chi": fidelity_by_chi,
    }


def mps_truncate_statevec(psi: torch.Tensor, n_qubits: int, chi_max: int) -> torch.Tensor:
    """Decompose a normalized 2^n statevector into an MPS truncated at chi_max and reconstruct."""
    rem = psi.reshape(1, 2, -1)
    cores: List[torch.Tensor] = []
    chi_l = 1
    for j in range(n_qubits - 1):
        mat = rem.reshape(chi_l * 2, -1)
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        k = min(int(chi_max), int(s.shape[0]))
        cores.append(u[:, :k].reshape(chi_l, 2, k))
        rem = (s[:k].unsqueeze(1) * vh[:k, :]).reshape(k, 2, -1)
        chi_l = k
    cores.append(rem.reshape(chi_l, 2, 1))

    # Contract back to 2^n vector
    out = cores[0].reshape(2, -1)
    for j in range(1, n_qubits):
        out = torch.einsum("ab,bcd->acd", out, cores[j]).reshape(-1, cores[j].shape[2])
    return out.reshape(-1)


def build_sha256_solution_state(
    midstate_s8: torch.Tensor,
    base_w16: torch.Tensor,
    num_rounds: int,
    m_nonce_bits: int,
    k_zero_bits: int,
    prof: Dict[str, Tuple[int, ...]],
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """Build exact boundary-projected BTC mining wavefunction on 2^m_nonce_bits nonces
    for REAL 256-bit SHA-256 (full message schedule), projecting top k_zero_bits of A_R to 0.
    """
    n_states = 1 << int(m_nonce_bits)
    nonces = torch.arange(n_states, dtype=torch.int64, device=device)
    s0 = midstate_s8.to(device).unsqueeze(0).expand(n_states, -1)
    w16 = base_w16.to(device).unsqueeze(0).expand(n_states, -1).clone()
    # Inject m_nonce_bits into low bits of W_0 (standard nonce sweep)
    mask_m = (1 << int(m_nonce_bits)) - 1
    w16[:, 0] = (w16[:, 0] & (~mask_m & MASK32)) | nonces

    s_out, _ = sha256_run_rounds(s0, w16, num_rounds=num_rounds, prof=prof)
    a_out = s_out[:, 0] & MASK32
    # Bitcoin-style leading zero bits on big-endian 32-bit word A_R
    shift = 32 - int(k_zero_bits)
    valid = ((a_out >> shift) == 0).to(torch.float32)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        # Fallback to soft minimum Hamming weight if ultra-rare empty set
        hw = torch.zeros_like(a_out, dtype=torch.float32)
        top_bits = a_out >> shift
        for b in range(k_zero_bits):
            hw = hw + ((top_bits >> b) & 1).to(torch.float32)
        valid = torch.exp(-4.0 * hw)
    psi = valid / (valid.norm() + 1e-12)
    return psi, n_valid
