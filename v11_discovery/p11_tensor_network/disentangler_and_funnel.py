"""Neural Orthogonal Disentangler Circuit (O(4) Cayley gates, zero-collapse) and
Lyapunov Funnel Networks (endpoint-only vs intermediate-state trajectory).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from v11_discovery.p11_tensor_network.mpo_arx import (
    analyze_state_entanglement,
    mps_truncate_statevec,
)


class CayleyGate2Q(nn.Module):
    """Exact orthogonal 2-qubit gate U in O(4) via Cayley transform: U = (I - A)(I + A)^{-1}."""

    def __init__(self, init_scale: float = 0.03) -> None:
        super().__init__()
        self.raw = nn.Parameter(torch.randn((4, 4), dtype=torch.float32) * init_scale)

    def matrix(self) -> torch.Tensor:
        a = self.raw - self.raw.transpose(0, 1)
        eye = torch.eye(4, dtype=a.dtype, device=a.device)
        return torch.linalg.solve(eye + a, eye - a)


def apply_2q_gate_to_statebatch(
    psi: torch.Tensor,
    n_qubits: int,
    q1: int,
    q2: int,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Apply a (4, 4) orthogonal gate to qubit pair (q1, q2) on state batch (B, 2^n)."""
    B = psi.shape[0]
    shape = (B,) + (2,) * n_qubits
    t = psi.reshape(shape)
    # Move axes q1+1 and q2+1 to the last two positions
    ax1 = q1 + 1
    ax2 = q2 + 1
    perm = [i for i in range(n_qubits + 1) if i not in (ax1, ax2)] + [ax1, ax2]
    inv_perm = [0] * (n_qubits + 1)
    for pos, orig in enumerate(perm):
        inv_perm[orig] = pos

    t_p = t.permute(perm).contiguous()
    lead_shape = t_p.shape[:-2]
    t_mat = t_p.reshape(-1, 4) @ gate.transpose(0, 1)
    t_back = t_mat.reshape(*lead_shape, 2, 2).permute(inv_perm).contiguous()
    return t_back.reshape(B, -1)


class NeuralDisentanglerCircuit(nn.Module):
    """Analytically invertible orthogonal disentangler circuit U_theta composed of local & strided O(4) gates.

    Because every gate is strictly orthogonal (U^T U = I), norm is identically preserved and
    encoder/latent collapse is mathematically impossible.
    """

    def __init__(self, n_qubits: int = 16, n_layers: int = 6) -> None:
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.pairs: List[Tuple[int, int]] = []
        strides = [1, 1, 2, 4, 1, 3, 2, 1]
        for layer in range(self.n_layers):
            st = strides[layer % len(strides)]
            offset = (layer % 2) * st if st == 1 else (layer % st)
            used = set()
            for q in range(offset, self.n_qubits - st, 2 * st if st == 1 else 1):
                q2 = q + st
                if q2 < self.n_qubits and q not in used and q2 not in used:
                    self.pairs.append((q, q2))
                    used.add(q)
                    used.add(q2)
        self.gates = nn.ModuleList([CayleyGate2Q(init_scale=0.04) for _ in self.pairs])

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        out = psi
        for (q1, q2), g in zip(self.pairs, self.gates):
            out = apply_2q_gate_to_statebatch(out, self.n_qubits, q1, q2, g.matrix())
        return out

    def inverse(self, phi: torch.Tensor) -> torch.Tensor:
        out = phi
        for (q1, q2), g in zip(reversed(self.pairs), reversed(self.gates)):
            out = apply_2q_gate_to_statebatch(out, self.n_qubits, q1, q2, g.matrix().transpose(0, 1))
        return out

    def max_orthogonality_error(self) -> float:
        eye = torch.eye(4, dtype=torch.float32, device=self.gates[0].raw.device)
        max_e = 0.0
        with torch.no_grad():
            for g in self.gates:
                u = g.matrix()
                e = float((u.transpose(0, 1) @ u - eye).abs().max().item())
                if e > max_e:
                    max_e = e
        return max_e


def cut_renyi2_entropy_loss(
    phi: torch.Tensor,
    n_qubits: int,
    cuts: Sequence[int],
) -> torch.Tensor:
    """Smooth, SVD-free exact Renyi-2 entanglement entropy across specified bipartition cuts.

    S_2(rho_b) = -log2(Tr(rho_b^2)) = -log2(||M M^T||_F^2) where M = phi.reshape(B, 2^b, 2^{n-b}).
    """
    B = phi.shape[0]
    total = torch.zeros((), dtype=phi.dtype, device=phi.device)
    for b in cuts:
        dim_l = 1 << int(b)
        dim_r = 1 << (n_qubits - int(b))
        m = phi.reshape(B, dim_l, dim_r)
        if dim_l <= dim_r:
            gram = torch.bmm(m, m.transpose(1, 2))
        else:
            gram = torch.bmm(m.transpose(1, 2), m)
        purity = (gram * gram).sum(dim=(1, 2)).clamp(min=1e-12, max=1.0)
        s2 = -torch.log2(purity)
        total = total + s2.mean()
    return total / float(len(cuts))


def evaluate_mps_born_enrichment(
    psi_true: torch.Tensor,
    valid_mask: torch.Tensor,
    n_qubits: int,
    chi_list: Sequence[int],
    disentangler: NeuralDisentanglerCircuit | None = None,
) -> Dict[str, Any]:
    """Compute exact MPS truncation fidelity and exact Born-machine BTC hit enrichment ratio.

    Enrichment E = P_MPS(valid) / P_random(valid), where P_random(valid) = n_valid / 2^n.
    """
    n_states = 1 << n_qubits
    n_valid = float(valid_mask.sum().item())
    p_rand = max(n_valid / float(n_states), 1e-12)

    with torch.no_grad():
        if disentangler is not None:
            phi = disentangler(psi_true.unsqueeze(0)).squeeze(0)
        else:
            phi = psi_true
        ent = analyze_state_entanglement(phi, n_qubits=n_qubits, chi_list=chi_list)

        enrichment_by_chi: Dict[str, float] = {}
        hit_prob_by_chi: Dict[str, float] = {}
        for chi in chi_list:
            phi_trunc = mps_truncate_statevec(phi, n_qubits=n_qubits, chi_max=int(chi))
            if disentangler is not None:
                psi_hat = disentangler.inverse(phi_trunc.unsqueeze(0)).squeeze(0)
            else:
                psi_hat = phi_trunc
            probs = psi_hat * psi_hat
            probs = probs / (probs.sum() + 1e-12)
            p_hit = float((probs * valid_mask).sum().item())
            hit_prob_by_chi[str(chi)] = round(p_hit, 6)
            enrichment_by_chi[str(chi)] = round(p_hit / p_rand, 4)

    return {
        "entanglement": ent,
        "p_random": round(p_rand, 6),
        "hit_prob_by_chi": hit_prob_by_chi,
        "enrichment_by_chi": enrichment_by_chi,
    }


def train_neural_disentangler(
    psi_batch: torch.Tensor,
    n_qubits: int,
    n_layers: int = 6,
    steps: int = 250,
    lr: float = 2e-2,
    cuts: Sequence[int] | None = None,
    seed: int = 20261125,
) -> Tuple[NeuralDisentanglerCircuit, Dict[str, float]]:
    """Train NeuralDisentanglerCircuit on a batch of states (B=1 for local film, B>1 for amortized)."""
    torch.manual_seed(seed)
    device = psi_batch.device
    if cuts is None:
        cuts = sorted({max(2, n_qubits // 4), n_qubits // 2, min(n_qubits - 2, (3 * n_qubits) // 4)})
    circuit = NeuralDisentanglerCircuit(n_qubits=n_qubits, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(circuit.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=lr * 0.1)

    with torch.no_grad():
        loss_init = float(cut_renyi2_entropy_loss(psi_batch, n_qubits, cuts).item())

    B = psi_batch.shape[0]
    mb = min(B, 16)
    gen = torch.Generator(device=device).manual_seed(seed + 7)
    circuit.train()
    loss_final = loss_init
    for _ in range(steps):
        if B > mb:
            idx = torch.randint(0, B, (mb,), device=device, generator=gen)
            batch = psi_batch[idx]
        else:
            batch = psi_batch
        phi = circuit(batch)
        loss = cut_renyi2_entropy_loss(phi, n_qubits, cuts)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(circuit.parameters(), 2.0)
        opt.step()
        sched.step()
        loss_final = float(loss.item())

    circuit.eval()
    return circuit, {
        "renyi2_loss_init": round(loss_init, 5),
        "renyi2_loss_final": round(loss_final, 5),
        "ortho_max_err": round(circuit.max_orthogonality_error(), 7),
        "n_gates": len(circuit.gates),
        "n_params": sum(p.numel() for p in circuit.parameters()),
    }


# ==============================================================================
# PART 4: Lyapunov Energy Funnel (Endpoint-Only vs Intermediate-State Trajectory)
# ==============================================================================

class _ResBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class LyapunovFunnelNet(nn.Module):
    """Scalar energy function V_theta(candidate_bits, target_bits) -> R (lower = closer to valid solution)."""

    def __init__(self, in_bits: int, hidden: int = 384, n_blocks: int = 3) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_bits, hidden)
        self.blocks = nn.ModuleList([_ResBlock(hidden) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(1.0 - 2.0 * bits)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h).squeeze(-1)
