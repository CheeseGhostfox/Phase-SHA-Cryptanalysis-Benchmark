"""Phase-SHA V5C: Loop-Aware Tensor Network MPS Solver with Batched QR/TEBD & Exact Born Sampling.

Strictly aligned with Lean 4 Module 6C (formal_proofs/PhaseSHA/V5/TensorEntanglementBarrier.lean):
  - Pure FP32 everywhere (zero complex64, zero full-size serial batch SVD).
  - Loop-aware permutation bringing closed-loop pairs (W_0 <-> W_16, r_13 <-> r_16) to short bond distance.
  - Batched no-grad `torch.linalg.qr` canonicalization & TEBD 2-site imaginary-time evolution.
  - Exact autoregressive conditional Born sampling P(s_k | s_{<k}) (never independent marginal sign).
  - Tracks exact bipartite von Neumann entanglement entropy `ent_entropy` & constraint `ent_energy`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bit_grid_backbone import (
    BitGridBlock,
    build_rotr_channels,
    extract_grid_features,
    _modulate_grid,
)
from src.dual_basis import DualBasisEngine
from src.graph_backbone import build_round_sparse_attn_mask, sinusoidal_embedding
from src.sha256_ref import bits_to_words, sha256_digest, words_to_bits
from src.v4_boundary import ManifoldBoundary
from src.v5_interfaces import (
    EntanglementLayer,
    OuterSearchShell,
    SolveResultV5,
    compute_calibrated_probs_from_candidates,
    compute_calibrated_probs_from_logits,
    compute_radar_detached,
)


def build_loop_aware_word_order(rounds: int = 18) -> Tuple[List[Tuple[int, int]], List[int], List[Tuple[int, int, int]]]:
    """Construct closed-loop topology-aware node order mapping chain pos p -> (round t, channel ch).

    Channels: 0=A, 1=E, 2=W.
    Brings (W_0 <-> W_16), (W_1 <-> W_17), (W_{14,15} <-> W_{16,17}), and (r_13 <-> r_16)
    into immediate neighborhood to minimize bond cut-width in finite-chi MPS.

    Returns:
      order: list of (t, ch) pairs of length L = 3 * rounds
      inv_flat: mapping flat (t * 3 + ch) -> chain index p
      loop_pairs: list of (pos_a, pos_b, expected_parity_sign) adjacent/short loop bond specs for monitoring
    """
    selected: List[Tuple[int, int]] = []
    seen = set()

    def add(t: int, ch: int) -> None:
        if 0 <= t < rounds and (t, ch) not in seen:
            seen.add((t, ch))
            selected.append((t, ch))

    # 1. Message expansion closed-loop core: W_0 <-> W_16, W_1 <-> W_17, W_14/15/9/10
    if rounds > 16:
        add(0, 2)
        add(16, 2)
        add(1, 2)
        add(17, 2)
        for t in (15, 14, 10, 9, 2, 3):
            add(t, 2)
    else:
        add(0, 2)

    # 2. Compression closed-loop walls r_13 <-> r_16 (E, A, W near rounds-1 .. 10)
    for t in range(rounds - 1, max(9, rounds - 9) - 1, -1):
        add(t, 1)  # E_t
        add(t, 0)  # A_t
        add(t, 2)  # W_t

    # 3. Remaining rounds reverse from R-1 down to 0 preserving carry adjacency E_{t+1} <-> E_t
    for t in range(rounds - 1, -1, -1):
        add(t, 1)
        add(t, 0)
        add(t, 2)

    assert len(selected) == 3 * rounds
    inv_flat = [0] * (3 * rounds)
    for pos, (t, ch) in enumerate(selected):
        inv_flat[t * 3 + ch] = pos

    # Record core loop bond monitors (used for ent_energy & core cut entropy)
    loop_pairs: List[Tuple[int, int, int]] = []
    if rounds > 16:
        loop_pairs.append((inv_flat[0 * 3 + 2], inv_flat[16 * 3 + 2], 1))
        if rounds > 17:
            loop_pairs.append((inv_flat[1 * 3 + 2], inv_flat[17 * 3 + 2], 1))
        loop_pairs.append((inv_flat[13 * 3 + 1], inv_flat[16 * 3 + 1], 1))
    else:
        loop_pairs.append((0, 1, 1))

    return selected, inv_flat, loop_pairs


@torch.no_grad()
@torch.amp.autocast("cuda", enabled=False)
def _batched_orthonormalize_mgs2(mat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Fast pure-FP32 2-pass Modified Gram-Schmidt orthonormalization on (N, m, k) with small k.

    Eliminates tiny-matrix cuSOLVER `torch.linalg.qr` serialization on CUDA while staying
    singular-free on rank-deficient zero columns (e.g., exact Bell/GHZ states).
    """
    x = mat.to(torch.float32)
    _, _, k_cols = x.shape
    q_cols: List[torch.Tensor] = []
    for j in range(k_cols):
        v = x[:, :, j]
        if q_cols:
            q_prev = torch.stack(q_cols, dim=-1)
            q_prev_t = q_prev.transpose(-2, -1)
            c1 = torch.bmm(q_prev_t, v.unsqueeze(-1))
            v = v - torch.bmm(q_prev, c1).squeeze(-1)
            c2 = torch.bmm(q_prev_t, v.unsqueeze(-1))
            v = v - torch.bmm(q_prev, c2).squeeze(-1)
        nrm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        q_j = torch.where(nrm > eps, v / nrm.clamp_min(eps), torch.zeros_like(v))
        q_cols.append(q_j)
    return torch.stack(q_cols, dim=-1)


class MPSChain(EntanglementLayer):
    """Exact pure-FP32 1D Matrix Product State engine with batched QR & conditional Born sampling."""

    def __init__(self, num_sites: int, bond_dim: int = 8) -> None:
        if num_sites < 2:
            raise ValueError(f"num_sites must be >= 2, got {num_sites}")
        if bond_dim < 2:
            raise ValueError(f"bond_dim must be >= 2, got {bond_dim}")
        self.num_sites = int(num_sites)
        self.bond_dim = int(bond_dim)
        self._seed_q_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}

    def _get_seed_q(self, chi: int, device: torch.device) -> torch.Tensor:
        key = (int(chi), device)
        cached = self._seed_q_cache.get(key)
        if cached is None:
            seed = torch.zeros((2 * chi, chi), dtype=torch.float32, device=device)
            eye = torch.eye(chi, dtype=torch.float32, device=device)
            seed[:chi, :] = 0.70710677 * eye
            seed[chi:, :] = 0.70710677 * eye
            for j in range(chi):
                seed[(chi + j) % (2 * chi), (j + 1) % chi] += 0.13
            cached = _batched_orthonormalize_mgs2(seed.unsqueeze(0)).squeeze(0)
            self._seed_q_cache[key] = cached
        return cached

    def make_ghz_or_superposition(
        self,
        batch_size: int,
        c_amp: float | torch.Tensor,
        s_amp: float | torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Construct exact Rank-2 closed-loop GHZ MPS tensor of shape (B, L, chi, 2, chi).

        Matches Lean 4 `mpsCycleAmpR` / `mpsCycleAmpQ2`:
          Psi(all +1) = c_amp, Psi(all -1) = s_amp, all other configs = 0.
        Uses boundary vector e_0 = (1, 0, ...) on both left and right ends.
        """
        l_len = self.num_sites
        chi = self.bond_dim
        cores = torch.zeros((batch_size, l_len, chi, 2, chi), dtype=torch.float32, device=device)

        c_t = c_amp if isinstance(c_amp, torch.Tensor) else torch.full((batch_size,), float(c_amp), dtype=torch.float32, device=device)
        s_t = s_amp if isinstance(s_amp, torch.Tensor) else torch.full((batch_size,), float(s_amp), dtype=torch.float32, device=device)

        # Site 0 maps e_0 row -> [c * e_0 (for spin +1), s * e_1 (for spin -1)]
        cores[:, 0, 0, 0, 0] = c_t
        cores[:, 0, 0, 1, 1] = s_t
        # Middle sites 1 .. L-2 act as diagonal identity on channel 0 (+1) and channel 1 (-1)
        if l_len > 2:
            cores[:, 1 : l_len - 1, 0, 0, 0] = 1.0
            cores[:, 1 : l_len - 1, 1, 1, 1] = 1.0
        # Last site L-1 routes both channel 0 (+1) and channel 1 (-1) back to e_0 column 0
        cores[:, l_len - 1, 0, 0, 0] = 1.0
        cores[:, l_len - 1, 1, 1, 0] = 1.0
        return cores

    def init_superposition_from_spins(
        self,
        spins_init: torch.Tensor,
        branch_mix: float = 0.25,
    ) -> torch.Tensor:
        """Initialize weakly entangled MPS cores around initial continuous/clamped spin beliefs.

        spins_init: (B, L) in [-1, 1]. Returns cores: (B, L, chi, 2, chi) FP32.
        """
        bsz, l_len = spins_init.shape
        device = spins_init.device
        chi = self.bond_dim
        sf = spins_init.to(torch.float32).clamp(-1.0, 1.0)
        p0 = ((1.0 + sf) * 0.5).clamp(1e-4, 1.0 - 1e-4).sqrt()  # prob |+1>
        p1 = ((1.0 - sf) * 0.5).clamp(1e-4, 1.0 - 1e-4).sqrt()  # prob |-1>

        cores = torch.zeros((bsz, l_len, chi, 2, chi), dtype=torch.float32, device=device)
        # Main branch along bond 0, plus secondary entangled correlated channel along bond 1
        w0 = math.sqrt(max(1e-4, 1.0 - float(branch_mix)))
        w1 = math.sqrt(max(0.0, float(branch_mix)))
        cores[:, 0, 0, 0, 0] = w0 * p0[:, 0]
        cores[:, 0, 0, 1, 0] = w0 * p1[:, 0]
        cores[:, 0, 0, 0, 1] = w1 * p0[:, 0]
        cores[:, 0, 0, 1, 1] = -w1 * p1[:, 0]
        if l_len > 2:
            cores[:, 1 : l_len - 1, 0, 0, 0] = p0[:, 1 : l_len - 1]
            cores[:, 1 : l_len - 1, 0, 1, 0] = p1[:, 1 : l_len - 1]
            cores[:, 1 : l_len - 1, 1, 0, 1] = p0[:, 1 : l_len - 1]
            cores[:, 1 : l_len - 1, 1, 1, 1] = p1[:, 1 : l_len - 1]
            if chi > 2:
                # Weak cross-bond coupling seed so higher chi modes remain active
                cross = 0.08
                cores[:, 1 : l_len - 1, 0, 1, 1] = cross
                cores[:, 1 : l_len - 1, 1, 0, 0] = cross
                if chi >= 4:
                    cores[:, 1 : l_len - 1, 1, 0, 2] = cross
                    cores[:, 1 : l_len - 1, 2, 1, 1] = cross
                    cores[:, 1 : l_len - 1, 2, 0, 0] = 0.5
                    cores[:, 1 : l_len - 1, 2, 1, 2] = 0.5
        cores[:, l_len - 1, 0, 0, 0] = p0[:, l_len - 1]
        cores[:, l_len - 1, 0, 1, 0] = p1[:, l_len - 1]
        cores[:, l_len - 1, 1, 0, 0] = p0[:, l_len - 1]
        cores[:, l_len - 1, 1, 1, 0] = -p1[:, l_len - 1]
        return self._normalize_cores_scalar(cores)

    @staticmethod
    def _normalize_cores_scalar(cores: torch.Tensor) -> torch.Tensor:
        norms = torch.linalg.vector_norm(cores, dim=(-3, -2, -1), keepdim=True).clamp_min(1e-6)
        return cores / norms

    @torch.no_grad()
    def right_canonicalize_qr(self, cores: torch.Tensor) -> torch.Tensor:
        """Pure FP32 right-to-left batched QR gauge canonicalization (no SVD, no autograd through QR)."""
        bsz, l_len, chi, _, _ = cores.shape
        out = cores.to(torch.float32).clone()
        for k in range(l_len - 1, 0, -1):
            # Group (2 * chi, chi) per transpose -> LQ via QR on transpose of (chi, 2*chi)
            mat = out[:, k].reshape(bsz, chi, 2 * chi)  # (B, chi, 2*chi)
            q_t, r_t = torch.linalg.qr(mat.transpose(-2, -1), mode="reduced")  # q_t:(B,2*chi,chi), r_t:(B,chi,chi)
            out[:, k] = q_t.transpose(-2, -1).reshape(bsz, chi, 2, chi)
            # Absorb R = r_t^T from left into core k - 1 on right bond axis
            r_mat = r_t.transpose(-2, -1)  # (B, chi, chi)
            prev = torch.einsum("blsc,bcj->blsj", out[:, k - 1], r_mat)
            nrm = torch.linalg.vector_norm(prev, dim=(-3, -2, -1), keepdim=True).clamp_min(1e-8)
            out[:, k - 1] = prev / nrm
        return out

    def clamp_sites(
        self,
        cores: torch.Tensor,
        clamped_spins: torch.Tensor,
        clamp_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Project clamped sites onto their discrete computational basis |0> (+1) or |1> (-1)."""
        # clamped_spins: (B, L), clamp_mask: (B, L) in {0, 1}
        m = clamp_mask.to(torch.float32).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, L, 1, 1, 1)
        keep_p0 = (clamped_spins.to(torch.float32) >= 0.0).to(torch.float32)
        proj = torch.stack([keep_p0, 1.0 - keep_p0], dim=-1)  # (B, L, 2)
        proj_w = m * proj.unsqueeze(2).unsqueeze(-1) + (1.0 - m) * torch.ones_like(cores[:, :, :1, :, :1])
        out = cores.to(torch.float32) * proj_w
        return self._normalize_cores_scalar(out)

    @torch.amp.autocast("cuda", enabled=False)
    def apply_one_site_fields(
        self,
        cores: torch.Tensor,
        fields_b_l_2: torch.Tensor,
        clamp_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply differentiable 1-site imaginary-time diagonal factor exp(field_s) on each site."""
        w = torch.softmax(fields_b_l_2.to(torch.float32), dim=-1) * 2.0  # bounded positive ~1 around 0
        if clamp_mask is not None:
            m = clamp_mask.to(torch.float32).unsqueeze(-1)
            w = m + (1.0 - m) * w
        out = cores.to(torch.float32) * w.unsqueeze(2).unsqueeze(-1)
        return self._normalize_cores_scalar(out)

    @torch.amp.autocast("cuda", enabled=False)
    def _apply_disjoint_pair_gates_qr(
        self,
        cores: torch.Tensor,
        gates_pairs: torch.Tensor,
        start_parity: int,
    ) -> torch.Tensor:
        """Apply disjoint 2-site gates on pairs (start, start+1), ..., using no-grad QR subspace truncation.

        Differentiable through Theta(G, A_k, A_{k+1}); QR computes orthonormal truncation basis Q_L
        under torch.no_grad() to eliminate zero-singular-value QR backward singularities.
        """
        bsz, l_len, chi, _, _ = cores.shape
        even_idx = list(range(start_parity, l_len - 1, 2))
        if not even_idx:
            return cores
        n_p = len(even_idx)
        idx_a = torch.tensor(even_idx, dtype=torch.int64, device=cores.device)
        idx_b = idx_a + 1

        a_sel = cores[:, idx_a].to(torch.float32)  # (B, P, chi, 2, chi)
        b_sel = cores[:, idx_b].to(torch.float32)  # (B, P, chi, 2, chi)

        # Contract shared bond m: M[b, p, l, s1, s2, r]
        m_two = torch.einsum("bplsm,bpmtr->bplstr", a_sel, b_sel)  # (B, P, chi, 2, 2, chi)
        # Apply 2-site gate G[b, p, s1', s2', s1, s2]
        g = gates_pairs.to(torch.float32).view(bsz, n_p, 2, 2, 2, 2)
        theta = torch.einsum("bpuvst,bplstr->bpluvr", g, m_two)  # (B, P, chi, 2, 2, chi)

        # Permute to (B*P, chi*2, 2*chi) -> Theta_mat[row=(l, u), col=(v, r)]
        theta_mat = (
            theta.permute(0, 1, 2, 3, 4, 5).reshape(bsz * n_p, chi * 2, 2 * chi)
        )

        # No-grad batched subspace iteration finds optimal rank-chi left orthonormal basis U_L (B*P, 2*chi, chi)
        with torch.no_grad():
            seed_q = self._get_seed_q(chi, cores.device)
            q_r = seed_q.unsqueeze(0).expand(bsz * n_p, -1, -1)
            # 2 subspace iterations capture exact rank <= chi subspace cleanly (Bell/GHZ exact)
            t_t = theta_mat.transpose(-2, -1)
            q_l = _batched_orthonormalize_mgs2(torch.bmm(theta_mat, q_r))
            q_r = _batched_orthonormalize_mgs2(torch.bmm(t_t, q_l) + 1e-6 * q_r)
            q_l = _batched_orthonormalize_mgs2(torch.bmm(theta_mat, q_r) + 1e-6 * q_l)
            u_left = q_l.detach()  # (B*P, 2*chi, chi)

        # Differentiable projection onto U_L: B_right = U_L^T @ Theta_mat -> (B*P, chi, 2*chi)
        b_right = torch.bmm(u_left.transpose(-2, -1), theta_mat)
        # Keep non-negative norm balanced across sites A', B' cleanly
        b_norm = torch.linalg.vector_norm(b_right, dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        b_right_n = b_right / b_norm

        a_new = u_left.view(bsz, n_p, chi, 2, chi)
        # B_right shape (B, P, chi, 2, chi): note row is m in chi, col is (v, r) -> reshape(B, P, chi, 2, chi)
        b_new = b_right_n.view(bsz, n_p, chi, 2, chi)

        # Assemble updated cores without in-place mutation breaking autograd
        new_cores_list = [cores[:, i] for i in range(l_len)]
        for k_idx, pos_a in enumerate(even_idx):
            new_cores_list[pos_a] = a_new[:, k_idx]
            new_cores_list[pos_a + 1] = b_new[:, k_idx]
        return self._normalize_cores_scalar(torch.stack(new_cores_list, dim=1))

    @torch.amp.autocast("cuda", enabled=False)
    def evolve_step(
        self,
        cores: torch.Tensor,
        fields_b_l_2: Optional[torch.Tensor] = None,
        gates_even: Optional[torch.Tensor] = None,
        gates_odd: Optional[torch.Tensor] = None,
        clamped_spins: Optional[torch.Tensor] = None,
        clamp_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One batched TEBD imaginary-time evolution sweep (1-site + even bonds + odd bonds + clamp)."""
        cur = cores.to(torch.float32)
        if fields_b_l_2 is not None:
            cur = self.apply_one_site_fields(cur, fields_b_l_2, clamp_mask=clamp_mask)
        if gates_even is not None:
            cur = self._apply_disjoint_pair_gates_qr(cur, gates_even, start_parity=0)
        if gates_odd is not None:
            cur = self._apply_disjoint_pair_gates_qr(cur, gates_odd, start_parity=1)
        if clamped_spins is not None and clamp_mask is not None:
            cur = self.clamp_sites(cur, clamped_spins, clamp_mask)
        return cur

    @torch.amp.autocast("cuda", enabled=False)
    def _compute_left_grams(self, cores: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute left-to-right scaled density matrices L_k of shape (B, chi, chi) and log scales.

        L_0 = e_0 e_0^T; L_{k+1} = sum_s (A_k^s)^T L_k A_k^s.
        """
        bsz, l_len, chi, _, _ = cores.shape
        dev = cores.device
        cur = torch.zeros((bsz, chi, chi), dtype=torch.float32, device=dev)
        cur[:, 0, 0] = 1.0
        grams = [cur]
        log_scales: List[torch.Tensor] = []

        for k in range(l_len):
            a0 = cores[:, k, :, 0, :]
            a1 = cores[:, k, :, 1, :]
            nxt = torch.bmm(a0.transpose(-2, -1), torch.bmm(cur, a0)) + torch.bmm(
                a1.transpose(-2, -1), torch.bmm(cur, a1)
            )
            scale = nxt.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp_min(1e-12)
            cur = nxt / scale.view(bsz, 1, 1)
            grams.append(cur)
            log_scales.append(torch.log(scale))
        return grams, log_scales

    @torch.amp.autocast("cuda", enabled=False)
    def _compute_right_envs(self, cores: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute right-to-left scaled environment matrices E_k = sum_s A_k^s E_{k+1} (A_k^s)^T."""
        bsz, l_len, chi, _, _ = cores.shape
        dev = cores.device
        cur = torch.zeros((bsz, chi, chi), dtype=torch.float32, device=dev)
        cur[:, 0, 0] = 1.0
        envs: List[Optional[torch.Tensor]] = [None] * (l_len + 1)
        log_scales: List[Optional[torch.Tensor]] = [None] * l_len
        envs[l_len] = cur

        for k in range(l_len - 1, -1, -1):
            a0 = cores[:, k, :, 0, :]
            a1 = cores[:, k, :, 1, :]
            nxt = torch.bmm(a0, torch.bmm(cur, a0.transpose(-2, -1))) + torch.bmm(
                a1, torch.bmm(cur, a1.transpose(-2, -1))
            )
            scale = nxt.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp_min(1e-12)
            cur = nxt / scale.view(bsz, 1, 1)
            envs[k] = cur
            log_scales[k] = torch.log(scale)
        return envs, log_scales  # type: ignore[return-value]

    @torch.amp.autocast("cuda", enabled=False)
    def joint_log_prob(
        self, cores: torch.Tensor, target_spins: torch.Tensor
    ) -> torch.Tensor:
        """Compute exact differentiable log |<target_spins | psi>|^2 / <psi | psi> in pure FP32.

        target_spins: (B, L) in {-1, +1}. Never factorizes into mean-field marginals!
        """
        bsz, l_len, chi, _, _ = cores.shape
        dev = cores.device
        target_bits = (target_spins.to(torch.float32) < 0.0).to(torch.int64)  # 0->+1, 1->-1

        # Norm <psi | psi> tracked via scaled right environments log Z = sum(log_scales) + log E_0[0,0]
        envs, log_scales = self._compute_right_envs(cores)
        log_z = torch.log(envs[0][:, 0, 0].clamp_min(1e-12))
        for ls in log_scales:
            log_z = log_z + ls

        # Contract amplitude e_0^T A_0^{s_0} ... A_{L-1}^{s_{L-1}} e_0 with step scalar rescaling
        v = torch.zeros((bsz, chi), dtype=torch.float32, device=dev)
        v[:, 0] = 1.0
        log_amp_sq = torch.zeros((bsz,), dtype=torch.float32, device=dev)
        for k in range(l_len):
            a0 = cores[:, k, :, 0, :]
            a1 = cores[:, k, :, 1, :]
            bk = target_bits[:, k].view(bsz, 1, 1).expand(-1, chi, chi)
            ak = torch.where(bk > 0, a1, a0)
            v = torch.bmm(v.unsqueeze(1), ak).squeeze(1)
            if k < l_len - 1:
                nrm = torch.linalg.vector_norm(v, dim=-1).clamp_min(1e-12)
                log_amp_sq = log_amp_sq + 2.0 * torch.log(nrm)
                v = v / nrm.unsqueeze(-1)
        amp_last = v[:, 0].abs().clamp_min(1e-12)
        log_amp_sq = log_amp_sq + 2.0 * torch.log(amp_last)
        return log_amp_sq - log_z

    @torch.amp.autocast("cuda", enabled=False)
    def marginal_logits(self, cores: torch.Tensor, temp: float = 0.5) -> torch.Tensor:
        """Compute exact single-site reduced marginals converted to BCE logits (+ -> +1, - -> -1)."""
        bsz, l_len, _, _, _ = cores.shape
        l_grams, _ = self._compute_left_grams(cores)
        r_envs, _ = self._compute_right_envs(cores)
        logits_cols: List[torch.Tensor] = []
        for k in range(l_len):
            lk = l_grams[k]
            rk = r_envs[k + 1]
            a0 = cores[:, k, :, 0, :]
            a1 = cores[:, k, :, 1, :]
            m0 = torch.bmm(a0.transpose(-2, -1), torch.bmm(lk, a0))
            m1 = torch.bmm(a1.transpose(-2, -1), torch.bmm(lk, a1))
            p0 = (m0 * rk).sum(dim=(-2, -1)).clamp_min(1e-9)
            p1 = (m1 * rk).sum(dim=(-2, -1)).clamp_min(1e-9)
            # Recall training convention BCE(2 * logit, 0.5*(1+spin)) => logit = 0.25 * log(p0 / p1)
            logits_cols.append(float(temp) * 0.5 * torch.log(p0 / p1))
        return torch.stack(logits_cols, dim=1)  # (B, L)

    @torch.no_grad()
    def conditional_sample(
        self,
        cores: torch.Tensor,
        clamped_spins: Optional[torch.Tensor] = None,
        clamp_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        generator: Optional[torch.Generator] = None,
        num_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact sequential autoregressive Born sampling P(s_k | s_{<k}) updating left conditioned state.

        Never calls independent marginal sign! Supports drawing `num_samples` chains per MPS
        while computing right environments O(1) times without replicating `cores`.
        Returns:
          - If num_samples == 1: (sampled_spins (B, L) in {-1, +1}, sample_log_probs (B,))
          - If num_samples > 1:  (sampled_spins (B, K, L) in {-1, +1}, sample_log_probs (B, K))
        """
        bsz, l_len, chi, _, _ = cores.shape
        dev = cores.device
        k_samp = max(1, int(num_samples))
        r_envs, _ = self._compute_right_envs(cores)

        v_l = torch.zeros((bsz, k_samp, chi), dtype=torch.float32, device=dev)
        v_l[:, :, 0] = 1.0
        spins_list: List[torch.Tensor] = []
        log_p_acc = torch.zeros((bsz, k_samp), dtype=torch.float32, device=dev)

        for k in range(l_len):
            a0 = cores[:, k, :, 0, :]
            a1 = cores[:, k, :, 1, :]
            u0 = torch.matmul(v_l, a0)  # (B, K, chi)
            u1 = torch.matmul(v_l, a1)  # (B, K, chi)
            ek1 = r_envs[k + 1]
            w0 = (torch.matmul(u0, ek1) * u0).sum(dim=-1).clamp_min(0.0)
            w1 = (torch.matmul(u1, ek1) * u1).sum(dim=-1).clamp_min(0.0)

            tot = (w0 + w1).clamp_min(1e-12)
            p1 = (w1 / tot).clamp(0.0, 1.0)
            p0 = 1.0 - p1

            if deterministic:
                bit_k = p1 > p0
            else:
                u_rand = torch.rand((bsz, k_samp), dtype=torch.float32, device=dev, generator=generator)
                bit_k = u_rand < p1

            if clamped_spins is not None and clamp_mask is not None:
                c_m = (clamp_mask[:, k] > 0.5).unsqueeze(1)
                c_b = (clamped_spins[:, k] < 0.0).unsqueeze(1)
                bit_k = torch.where(c_m, c_b, bit_k)

            prob_chosen = torch.where(bit_k, p1, p0).clamp_min(1e-12)
            log_p_acc = log_p_acc + torch.log(prob_chosen)
            spin_k = torch.where(bit_k, -torch.ones_like(p1), torch.ones_like(p1))
            spins_list.append(spin_k)

            u_chosen = torch.where(bit_k.unsqueeze(-1), u1, u0)
            v_l = u_chosen / torch.linalg.vector_norm(u_chosen, dim=-1, keepdim=True).clamp_min(1e-8)

        spins_all = torch.stack(spins_list, dim=-1)  # (B, K, L)
        if k_samp == 1:
            return spins_all[:, 0, :], log_p_acc[:, 0]
        return spins_all, log_p_acc

    @torch.no_grad()
    def compute_entanglement_entropy(
        self, cores: torch.Tensor, cut_idx: Optional[int] = None
    ) -> torch.Tensor:
        """Compute exact bipartite von Neumann entanglement entropy S(rho_cut) across bond `cut_idx`."""
        bsz, l_len, _, _, _ = cores.shape
        c = max(0, min(l_len - 2, (l_len // 2) if cut_idx is None else int(cut_idx)))
        l_grams, _ = self._compute_left_grams(cores)
        r_envs, _ = self._compute_right_envs(cores)

        # Reduced density operator eigenvalues match spectrum of L_{c+1}^{1/2} E_{c+1} L_{c+1}^{1/2} / Tr(.)
        left_g = 0.5 * (l_grams[c + 1] + l_grams[c + 1].transpose(-2, -1))
        right_e = 0.5 * (r_envs[c + 1] + r_envs[c + 1].transpose(-2, -1))
        evals_l, evecs_l = torch.linalg.eigh(left_g)
        evals_l = evals_l.clamp_min(0.0).sqrt()
        sqrt_l = torch.bmm(evecs_l * evals_l.unsqueeze(-2), evecs_l.transpose(-2, -1))
        rho = torch.bmm(sqrt_l, torch.bmm(right_e, sqrt_l))
        rho = 0.5 * (rho + rho.transpose(-2, -1))
        lam = torch.linalg.eigvalsh(rho).clamp_min(0.0)
        p = lam / lam.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
        return entropy

    @torch.amp.autocast("cuda", enabled=False)
    def compute_entangled_energy(
        self, cores: torch.Tensor, pair_list: List[Tuple[int, int, int]]
    ) -> torch.Tensor:
        """Exact joint expectation energy sum_{(i,j)} <psi | (1 - s_i s_j)^2 / 4 | psi> on loop pairs."""
        bsz, l_len, chi, _, _ = cores.shape
        l_grams, _ = self._compute_left_grams(cores)
        r_envs, _ = self._compute_right_envs(cores)

        energies: List[torch.Tensor] = []
        for pos_a, pos_b, target_sign in pair_list:
            i, j = (pos_a, pos_b) if pos_a <= pos_b else (pos_b, pos_a)
            if i == j or i < 0 or j >= l_len:
                continue
            # Transfer Pauli Z operator: A^Z = A^(+1) (x) A^(+1) - A^(-1) (x) A^(-1) acting on Left Gram L_i
            a0_i, a1_i = cores[:, i, :, 0, :], cores[:, i, :, 1, :]
            g_norm = torch.bmm(a0_i.transpose(-2, -1), torch.bmm(l_grams[i], a0_i)) + torch.bmm(
                a1_i.transpose(-2, -1), torch.bmm(l_grams[i], a1_i)
            )
            g_z = torch.bmm(a0_i.transpose(-2, -1), torch.bmm(l_grams[i], a0_i)) - torch.bmm(
                a1_i.transpose(-2, -1), torch.bmm(l_grams[i], a1_i)
            )
            # Propagate identity transfer between i+1 .. j-1 keeping scale normalization matched!
            for mid in range(i + 1, j):
                am0, am1 = cores[:, mid, :, 0, :], cores[:, mid, :, 1, :]
                g_norm = torch.bmm(am0.transpose(-2, -1), torch.bmm(g_norm, am0)) + torch.bmm(
                    am1.transpose(-2, -1), torch.bmm(g_norm, am1)
                )
                g_z = torch.bmm(am0.transpose(-2, -1), torch.bmm(g_z, am0)) + torch.bmm(
                    am1.transpose(-2, -1), torch.bmm(g_z, am1)
                )
                sc = g_norm.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp_min(1e-12).view(bsz, 1, 1)
                g_norm = g_norm / sc
                g_z = g_z / sc

            a0_j, a1_j = cores[:, j, :, 0, :], cores[:, j, :, 1, :]
            n_fin = torch.bmm(a0_j.transpose(-2, -1), torch.bmm(g_norm, a0_j)) + torch.bmm(
                a1_j.transpose(-2, -1), torch.bmm(g_norm, a1_j)
            )
            zz_fin = torch.bmm(a0_j.transpose(-2, -1), torch.bmm(g_z, a0_j)) - torch.bmm(
                a1_j.transpose(-2, -1), torch.bmm(g_z, a1_j)
            )
            r_j = r_envs[j + 1]
            denom = (n_fin * r_j).sum(dim=(-2, -1)).clamp_min(1e-12)
            exp_zz = ((zz_fin * r_j).sum(dim=(-2, -1)) / denom).clamp(-1.0, 1.0)
            # Violation energy 0.5 * (1 - target_sign * <s_i s_j>) in [0, 1] -> zero on exact GHZ
            energies.append(0.5 * (1.0 - float(target_sign) * exp_zz))

        if not energies:
            return torch.zeros((bsz,), dtype=torch.float32, device=cores.device)
        return torch.stack(energies, dim=0).sum(dim=0)


class V5CMPSSolver(nn.Module, OuterSearchShell):
    """Loop-aware 32-bit-lane MPS solver with Bit-Grid + BiGRU neural controller (V3 warm-start compatible)."""

    BASE_DIM: int = 16
    ROT_DIM: int = BitGridBlock.ROT_DIM
    IN_DIM: int = BASE_DIM + ROT_DIM

    def __init__(
        self,
        rounds: int = 18,
        bond_dim: int = 8,
        d_bit: int = 48,
        n_heads: int = 3,
        n_blocks: int = 2,
        d_ff_mult: int = 2,
        share_blocks: bool = False,
        grad_checkpointing: bool = False,
        use_carry_gru: bool = True,
        grad_scale: float = 2.0,
        branch_mix: float = 0.25,
        num_candidates: int = 256,
        num_steps: int = 16,
        chunk_size: int = 16,
    ) -> None:
        super().__init__()
        self.rounds = int(rounds)
        self.bond_dim = int(bond_dim)
        self.d_bit = int(d_bit)
        self.grad_scale = float(grad_scale)
        self.branch_mix = float(branch_mix)
        self.default_num_candidates = int(num_candidates)
        self.default_num_steps = int(num_steps)
        self.default_chunk_size = int(chunk_size)

        order, inv_flat, loop_pairs = build_loop_aware_word_order(self.rounds)
        self.word_order = order
        self.inv_flat = inv_flat
        self.loop_pairs = loop_pairs
        self.register_buffer(
            "order_t", torch.tensor([t for t, _ in order], dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "order_ch", torch.tensor([ch for _, ch in order], dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "inv_flat_idx", torch.tensor(inv_flat, dtype=torch.int64), persistent=False
        )

        self.mps = MPSChain(num_sites=3 * self.rounds, bond_dim=self.bond_dim)
        self.dual_basis = DualBasisEngine(rounds=self.rounds)

        # Shared V3 core layers for lossless warm-start from stage3b_r18_calibrated_best.pt
        self.input_proj = nn.Linear(self.IN_DIM, d_bit)
        self.h_prev_norm = nn.LayerNorm(d_bit, eps=1e-6)
        self.h_prev_proj = nn.Linear(d_bit, d_bit)

        self.register_buffer(
            "round_pos_emb",
            sinusoidal_embedding(torch.arange(rounds, dtype=torch.float32), d_bit).view(
                1, rounds, 1, d_bit
            ),
            persistent=False,
        )
        self.register_buffer(
            "bit_pos_emb",
            sinusoidal_embedding(torch.arange(32, dtype=torch.float32), d_bit).view(
                1, 1, 32, d_bit
            ),
            persistent=False,
        )
        self.register_buffer(
            "round_attn_mask",
            build_round_sparse_attn_mask(rounds, num_registers=0),
            persistent=False,
        )

        self.step_mlp = nn.Sequential(
            nn.Linear(d_bit, d_bit),
            nn.SiLU(),
            nn.Linear(d_bit, d_bit),
        )
        self.lam_mlp = nn.Sequential(
            nn.Linear(32, d_bit),
            nn.SiLU(),
            nn.Linear(d_bit, d_bit),
        )

        if share_blocks:
            shared = BitGridBlock(
                d_bit=d_bit, n_heads=n_heads, d_ff_mult=d_ff_mult, use_carry_gru=use_carry_gru
            )
            self.blocks = nn.ModuleList([shared for _ in range(n_blocks)])
        else:
            self.blocks = nn.ModuleList(
                [
                    BitGridBlock(
                        d_bit=d_bit, n_heads=n_heads, d_ff_mult=d_ff_mult, use_carry_gru=use_carry_gru
                    )
                    for _ in range(n_blocks)
                ]
            )

        self.final_norm = nn.LayerNorm(d_bit, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_bit, 2 * d_bit, bias=True))
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)

        self.out_proj = nn.Linear(d_bit, 3)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

        # V5C TEBD coupling adapter head: outputs (ZZ coupling J, ZX/XZ bias) per site channel
        self.mps_gate_head = nn.Linear(d_bit, 6)
        nn.init.normal_(self.mps_gate_head.weight, std=0.02)
        nn.init.zeros_(self.mps_gate_head.bias)

    def float(self) -> "V5CMPSSolver":
        for name, mod in self.named_children():
            if name != "dual_basis":
                mod.float()
        return self

    def load_v3_warmstart(self, ckpt_path: str | Path, map_location: torch.device | str = "cpu") -> Dict[str, object]:
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        src_state = ckpt.get("ema_model_state", ckpt.get("model_state"))
        incompat = self.load_state_dict(src_state, strict=False)
        non_adapter_missing = [
            k for k in incompat.missing_keys if not k.startswith("mps_gate_head.")
        ]
        if incompat.unexpected_keys or non_adapter_missing:
            raise RuntimeError(
                f"V3->V5C mismatch: unexpected={incompat.unexpected_keys}, missing={non_adapter_missing}"
            )
        self.float()
        return {
            "loaded_keys": len(src_state),
            "missing_adapter_keys": list(incompat.missing_keys),
            "source_step": ckpt.get("step", "?"),
        }

    def nodes_to_chain(self, nodes: torch.Tensor) -> torch.Tensor:
        """Reorder (B, R, 3, 32) -> (B*32, L=3*R) along loop-aware topology."""
        bsz = nodes.shape[0]
        ordered = nodes[:, self.order_t, self.order_ch, :]  # (B, L, 32)
        return ordered.permute(0, 2, 1).reshape(bsz * 32, 3 * self.rounds)

    def chain_to_nodes(self, chain: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Restore chain (B*32, L=3*R) -> (B, R, 3, 32)."""
        l_len = 3 * self.rounds
        by_pos = chain.view(batch_size, 32, l_len).permute(0, 2, 1)  # (B, L, 32)
        flat = by_pos[:, self.inv_flat_idx, :]  # (B, 3*R, 32)
        return flat.view(batch_size, self.rounds, 3, 32)

    @torch.amp.autocast("cuda", enabled=False)
    def build_pair_gates(
        self,
        coupling_field: torch.Tensor,
        start_parity: int,
        tau: float = 0.25,
    ) -> torch.Tensor:
        """Build positive diagonal+off-diagonal 2-site imaginary-time TEBD gates (B*32, P, 4, 4)."""
        # coupling_field: (B*32, L, 2) -> col 0: ZZ strength, col 1: flip-exchange X1X2 strength
        l_len = coupling_field.shape[1]
        even_idx = list(range(start_parity, l_len - 1, 2))
        n_p = len(even_idx)
        idx_a = torch.tensor(even_idx, dtype=torch.int64, device=coupling_field.device)
        idx_b = idx_a + 1
        j_zz = 0.5 * torch.tanh(0.5 * (coupling_field[:, idx_a, 0] + coupling_field[:, idx_b, 0]))
        j_xx = 0.25 * torch.tanh(0.5 * (coupling_field[:, idx_a, 1] + coupling_field[:, idx_b, 1]))

        dev = coupling_field.device
        bm = coupling_field.shape[0]
        # Base identity + imaginary-time exp(-tau * H) parametrized smoothly on (4, 4)
        gates = torch.eye(4, dtype=torch.float32, device=dev).view(1, 1, 4, 4).expand(bm, n_p, 4, 4).clone()
        e_same = torch.exp(float(tau) * j_zz)
        e_diff = torch.exp(-float(tau) * j_zz)
        gates[:, :, 0, 0] = e_same
        gates[:, :, 3, 3] = e_same
        gates[:, :, 1, 1] = e_diff
        gates[:, :, 2, 2] = e_diff
        # Controlled superposition tunneling between (|++>, |-->) and (|+->, |-+>)
        t_term = float(tau) * j_xx
        gates[:, :, 0, 3] = t_term
        gates[:, :, 3, 0] = t_term
        gates[:, :, 1, 2] = t_term * 0.5
        gates[:, :, 2, 1] = t_term * 0.5
        return gates

    @torch.amp.autocast("cuda", enabled=False)
    def forward_step(
        self,
        cores: torch.Tensor,
        boundary: ManifoldBoundary,
        step_frac: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
        tau: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one neural-guided TEBD MPS evolution step returning (cores_next, logits_nodes, x_marginals, h_out, pair_gates_meta)."""
        bsz = boundary.batch_size
        r = self.rounds
        device = cores.device
        lam_full = torch.ones(32, dtype=torch.float32, device=device)

        # Single-site reduced expectations in [-1, +1] drive radar + grid backbone
        marg_logits_chain = self.mps.marginal_logits(cores, temp=0.5)
        marg_logits_nodes = self.chain_to_nodes(marg_logits_chain, bsz)
        x_spins = torch.tanh(2.0 * marg_logits_nodes)
        cb = boundary.to_clamped_boundary()
        x_spins = cb.apply_clamp(x_spins)

        grad_radar, res_feat = compute_radar_detached(
            self.dual_basis, x_spins, cb, lam=lam_full, grad_scale=self.grad_scale
        )
        base_feat, x_grid, grad_grid = extract_grid_features(
            x_nodes=x_spins,
            grad_radar=grad_radar,
            clamp_mask=cb.clamp_mask,
            res_features=res_feat,
            k_spins=self.dual_basis.K_spins,
        )
        rot_feat = build_rotr_channels(x_grid, grad_grid)
        grid_in = torch.cat([base_feat, rot_feat], dim=-1).to(torch.float32)

        step_b = step_frac.to(torch.float32).expand(bsz) if step_frac.ndim == 0 else step_frac.to(torch.float32)
        lam_b = lam_full.unsqueeze(0).expand(bsz, 32)
        cond = F.silu(
            self.step_mlp(sinusoidal_embedding(step_b * 1000.0, self.d_bit)) + self.lam_mlp(lam_b)
        )

        h = self.input_proj(grid_in) + self.round_pos_emb + self.bit_pos_emb
        if h_prev is not None:
            h = h + self.h_prev_proj(self.h_prev_norm(h_prev.to(torch.float32)))

        rot_in = rot_feat.to(torch.float32)
        for block in self.blocks:
            h = block(h, rot_in, cond, self.round_attn_mask)

        sh_f, sc_f = self.final_adaLN(cond).chunk(2, dim=-1)
        h_out = _modulate_grid(self.final_norm(h), sh_f, sc_f)

        logits_nodes = self.out_proj(h_out).permute(0, 1, 3, 2).contiguous()  # (B, R, 3, 32)
        mps_ctrl = self.mps_gate_head(h_out).view(bsz, r, 32, 3, 2).permute(0, 1, 3, 2, 4).contiguous()

        # Reorder logits as 1-site fields (h, -h) and mps_ctrl to loop-aware chain order
        f_nodes = torch.stack([logits_nodes, -logits_nodes], dim=-1)  # (B, R, 3, 32, 2)
        f_chain = f_nodes[:, self.order_t, self.order_ch, :, :].permute(0, 2, 1, 3).reshape(bsz * 32, 3 * r, 2)
        j_chain = mps_ctrl[:, self.order_t, self.order_ch, :, :].permute(0, 2, 1, 3).reshape(bsz * 32, 3 * r, 2)

        gates_even = self.build_pair_gates(j_chain, start_parity=0, tau=tau)
        gates_odd = self.build_pair_gates(j_chain, start_parity=1, tau=tau)
        clamp_s_chain = self.nodes_to_chain(cb.x_clamped)
        clamp_m_chain = self.nodes_to_chain(cb.clamp_mask)

        cores_next = self.mps.evolve_step(
            cores,
            fields_b_l_2=0.25 * f_chain,
            gates_even=gates_even,
            gates_odd=gates_odd,
            clamped_spins=clamp_s_chain,
            clamp_mask=clamp_m_chain,
        )
        return cores_next, logits_nodes, x_spins, h.to(torch.float32), marg_logits_nodes

    def init_cores(self, boundary: ManifoldBoundary, branch_mix: Optional[float] = None) -> torch.Tensor:
        cb = boundary.to_clamped_boundary()
        init_spins = cb.clamp_mask * cb.x_clamped
        s_chain = self.nodes_to_chain(init_spins)
        c_chain = self.nodes_to_chain(cb.clamp_mask)
        mix = self.branch_mix if branch_mix is None else float(branch_mix)
        cores = self.mps.init_superposition_from_spins(s_chain, branch_mix=mix)
        return self.mps.clamp_sites(cores, s_chain, c_chain)

    @torch.inference_mode()
    def infer_deterministic_zero(
        self,
        boundary: ManifoldBoundary,
        num_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Deterministic conditional Born decode from MPS superposition."""
        bsz = boundary.batch_size
        dev = boundary.device
        n_steps = self.default_num_steps if num_steps is None else int(num_steps)
        cores = self.init_cores(boundary)
        h_prev: Optional[torch.Tensor] = None
        logits_last: Optional[torch.Tensor] = None

        for s in range(n_steps):
            sf = torch.full((bsz,), float(s) / float(max(1, n_steps - 1)), dtype=torch.float32, device=dev)
            cores, logits_last, _, h_prev, _ = self.forward_step(cores, boundary, sf, h_prev=h_prev)

        cb = boundary.to_clamped_boundary()
        clamp_s = self.nodes_to_chain(cb.x_clamped)
        clamp_m = self.nodes_to_chain(cb.clamp_mask)
        sampled_chain, _ = self.mps.conditional_sample(
            cores, clamped_spins=clamp_s, clamp_mask=clamp_m, deterministic=True
        )
        x_disc = cb.apply_clamp(self.chain_to_nodes(sampled_chain, bsz))
        if logits_last is None:
            logits_last = self.chain_to_nodes(self.mps.marginal_logits(cores), bsz)

        a_d, e_d, w_d = cb.assemble_full(x_disc)
        lam_full = torch.ones(32, dtype=torch.float32, device=dev)
        _, _, stats = self.dual_basis.compute_energy_and_features(a_d, e_d, w_d, lam=lam_full)
        m1 = stats["per_scale"][:, 0]
        m32 = stats["per_scale"][:, 31]

        ent_e = self.mps.compute_entanglement_entropy(cores, cut_idx=1)
        loop_en = self.mps.compute_entangled_energy(cores, self.loop_pairs)
        zeros = torch.zeros_like(m1)
        extra = {
            "ent_entropy": float(ent_e.mean().item()),
            "ent_energy": float(loop_en.mean().item()),
        }
        return logits_last, x_disc, m1, m32, zeros, zeros, zeros, extra

    @torch.inference_mode()
    def solve(
        self,
        target_digests: Optional[torch.Tensor] = None,
        boundary: Optional[ManifoldBoundary] = None,
        generator: Optional[torch.Generator] = None,
        num_candidates: Optional[int] = None,
        num_steps: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> SolveResultV5:
        """Run TEBD evolution per digest and draw exact autoregressive conditional Born samples."""
        if boundary is None:
            if target_digests is None:
                raise ValueError("Either boundary or target_digests must be provided.")
            from src.v4_boundary import build_manifold_boundary

            boundary = build_manifold_boundary(target_digests, rounds=self.rounds)

        dev = boundary.device
        b_cnt = boundary.batch_size
        k_cand = self.default_num_candidates if num_candidates is None else int(num_candidates)
        n_steps = self.default_num_steps if num_steps is None else int(num_steps)
        c_size = self.default_chunk_size if chunk_size is None else int(chunk_size)
        if target_digests is None:
            target_digests = torch.zeros((b_cnt, 8), dtype=torch.int64, device=dev)

        lam_full = torch.ones(32, dtype=torch.float32, device=dev)
        all_disc: List[torch.Tensor] = []
        all_e: List[torch.Tensor] = []
        all_m1: List[torch.Tensor] = []
        all_m32: List[torch.Tensor] = []
        ent_list: List[float] = []
        en_list: List[float] = []

        for b0 in range(0, b_cnt, max(1, c_size)):
            b1 = min(b0 + max(1, c_size), b_cnt)
            b_cur = b1 - b0
            b_sub = ManifoldBoundary(
                rounds=boundary.rounds,
                A_full_clamped=boundary.A_full_clamped[b0:b1],
                A_full_mask=boundary.A_full_mask[b0:b1],
                E_full_clamped=boundary.E_full_clamped[b0:b1],
                E_full_mask=boundary.E_full_mask[b0:b1],
                W_full_clamped=boundary.W_full_clamped[b0:b1],
                W_full_mask=boundary.W_full_mask[b0:b1],
                A_words_clamped=boundary.A_words_clamped[b0:b1],
                E_words_clamped=boundary.E_words_clamped[b0:b1],
                W16_clamped_words=boundary.W16_clamped_words[b0:b1],
                W16_mask_bits=boundary.W16_mask_bits[b0:b1],
                digest_target_bits=(
                    None if boundary.digest_target_bits is None else boundary.digest_target_bits[b0:b1]
                ),
                digest_target_mask=(
                    None if boundary.digest_target_mask is None else boundary.digest_target_mask[b0:b1]
                ),
                iv_words=boundary.iv_words,
                is_full_digest_clamped=boundary.is_full_digest_clamped,
            )
            cores = self.init_cores(b_sub)
            h_prev = None
            for s in range(n_steps):
                sf = torch.full((b_cur,), float(s) / float(max(1, n_steps - 1)), dtype=torch.float32, device=dev)
                cores, _, _, h_prev, _ = self.forward_step(cores, b_sub, sf, h_prev=h_prev)

            ent_list.append(float(self.mps.compute_entanglement_entropy(cores, cut_idx=1).mean().item()))
            en_list.append(float(self.mps.compute_entangled_energy(cores, self.loop_pairs).mean().item()))

            cb_sub = b_sub.to_clamped_boundary()
            clamp_s = self.nodes_to_chain(cb_sub.x_clamped)
            clamp_m = self.nodes_to_chain(cb_sub.clamp_mask)

            sampled_chain, _ = self.mps.conditional_sample(
                cores,
                clamped_spins=clamp_s,
                clamp_mask=clamp_m,
                deterministic=False,
                generator=generator,
                num_samples=k_cand,
            )
            if k_cand == 1:
                sampled_flat = sampled_chain
            else:
                sampled_flat = (
                    sampled_chain.view(b_cur, 32, k_cand, 3 * self.rounds)
                    .permute(0, 2, 1, 3)
                    .reshape(b_cur * k_cand * 32, 3 * self.rounds)
                )
            cb_rep = cb_sub.repeat_for_candidates(k_cand)
            x_disc_rep = cb_rep.apply_clamp(self.chain_to_nodes(sampled_flat, b_cur * k_cand))
            a_d, e_d, w_d = cb_rep.assemble_full(x_disc_rep)
            tot_e, _, stats = self.dual_basis.compute_energy_and_features(a_d, e_d, w_d, lam=lam_full)
            all_disc.append(x_disc_rep.view(b_cur, k_cand, self.rounds, 3, 32))
            all_e.append(tot_e.view(b_cur, k_cand))
            all_m1.append(stats["per_scale"][:, 0].view(b_cur, k_cand))
            all_m32.append(stats["per_scale"][:, 31].view(b_cur, k_cand))

        x_disc_all = torch.cat(all_disc, dim=0)
        e_all = torch.cat(all_e, dim=0)
        m1_all = torch.cat(all_m1, dim=0)
        m32_all = torch.cat(all_m32, dim=0)

        k_w = min(16, self.rounds)
        w_pred_bits = (x_disc_all[:, :, :k_w, 2, :] < 0.0).to(torch.int64)
        if k_w < 16:
            pad_b = torch.zeros((b_cnt, k_cand, 16 - k_w, 32), dtype=torch.int64, device=dev)
            w_pred_bits = torch.cat([w_pred_bits, pad_b], dim=2)
        w16_m_int = (boundary.W16_mask_bits > 0.5).to(torch.int64).unsqueeze(1)
        bits_clamp = words_to_bits(boundary.W16_clamped_words).unsqueeze(1)
        w16_cand = bits_to_words(torch.where(w16_m_int > 0, bits_clamp, w_pred_bits))

        if boundary.is_full_digest_clamped:
            dig_cand = sha256_digest(w16_cand.reshape(-1, 16), rounds=self.rounds).view(b_cnt, k_cand, 8)
            hits_mat = (dig_cand == target_digests.unsqueeze(1)).all(dim=-1)
        elif boundary.digest_target_mask is not None and boundary.digest_target_bits is not None:
            dig_cand = sha256_digest(w16_cand.reshape(-1, 16), rounds=self.rounds).view(b_cnt, k_cand, 8)
            dig_bits = words_to_bits(dig_cand).to(torch.float32)
            tgt_b = boundary.digest_target_bits.unsqueeze(1).to(torch.float32)
            tgt_m = boundary.digest_target_mask.unsqueeze(1).to(torch.float32)
            hits_mat = ((dig_bits - tgt_b).abs() * tgt_m).sum(dim=(-2, -1)) < 0.5
        else:
            hits_mat = torch.zeros((b_cnt, k_cand), dtype=torch.bool, device=dev)

        num_hits = hits_mat.sum(dim=1)
        solved = num_hits > 0
        best_idx = m32_all.argmin(dim=1)
        if solved.any():
            best_idx = torch.where(hits_mat, m32_all - 1e6, m32_all).argmin(dim=1)

        b_idx = torch.arange(b_cnt, device=dev)
        w0_vals = w16_cand[:, :, 0].detach().cpu().tolist()
        uniq_ratios = torch.tensor(
            [float(len(set(row))) / float(max(1, k_cand)) for row in w0_vals],
            dtype=torch.float32,
            device=dev,
        )
        cal_probs, p_weights = compute_calibrated_probs_from_candidates(
            x_disc_all, m32_all, temperature=1.0
        )

        zeros = torch.zeros((b_cnt,), dtype=torch.float32, device=dev)
        return SolveResultV5(
            solved=solved,
            num_hits=num_hits,
            unique_ratio=uniq_ratios,
            latch_ratio=zeros,
            best_w16=w16_cand[b_idx, best_idx],
            best_energy=e_all[b_idx, best_idx],
            best_m1_energy=m1_all[b_idx, best_idx],
            best_m32_energy=m32_all[b_idx, best_idx],
            min_m32_energy=m32_all.min(dim=1).values,
            best_x_disc=x_disc_all[b_idx, best_idx],
            mean_align=zeros,
            mean_comm=zeros,
            calibrated_probs=cal_probs,
            particle_weights=p_weights,
            extra_metrics={
                "ent_entropy": float(sum(ent_list) / max(1, len(ent_list))),
                "ent_energy": float(sum(en_list) / max(1, len(en_list))),
            },
        )

    def export_calibrated_probs(self, solve_result: SolveResultV5) -> torch.Tensor:
        if solve_result.calibrated_probs is not None:
            return solve_result.calibrated_probs
        return compute_calibrated_probs_from_logits(1.5 * solve_result.best_x_disc)


__all__ = [
    "MPSChain",
    "V5CMPSSolver",
    "build_loop_aware_word_order",
]
