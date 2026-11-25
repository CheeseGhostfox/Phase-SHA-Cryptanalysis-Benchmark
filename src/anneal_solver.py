"""Inference engine: frequency-homotopy annealing (m=1 -> 32) with hard verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from src.boundary_clamp import ClampedBoundary, build_clamped_boundary
from src.dual_basis import DualBasisEngine
from src.graph_backbone import PhaseSHABackbone
from src.sha256_ref import sha256_digest


@dataclass
class AnnealSolveResult:
    """Results from frequency-homotopy annealing over a batch of target digests."""

    solved: torch.Tensor  # (B,) bool
    num_hits: torch.Tensor  # (B,) int64
    best_w16: torch.Tensor  # (B, 16) int64
    best_energy: torch.Tensor  # (B,) float32
    best_m1_energy: torch.Tensor  # (B,) float32
    best_m32_energy: torch.Tensor  # (B,) float32
    best_x_disc: torch.Tensor  # (B, R, 3, 32) float32
    final_x_sample: Optional[torch.Tensor] = None  # (K, R, 3, 32) for last digest

    @property
    def solve_rate(self) -> float:
        return float(self.solved.float().mean().item())


def homotopy_lambda(
    tau: float | torch.Tensor,
    gamma: float = 2.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute frequency homotopy mask lambda_m(tau) = sigmoid(gamma * (m_front(tau) - m))."""
    m = torch.arange(1, 33, dtype=torch.float32, device=device)
    if isinstance(tau, torch.Tensor):
        tau_t = tau.to(device=device, dtype=torch.float32)
        m_front = 1.0 + 31.0 * torch.pow(tau_t, 1.5)
        if m_front.ndim == 1:
            return torch.sigmoid(gamma * (m_front.unsqueeze(-1) - m.unsqueeze(0)))
        return torch.sigmoid(gamma * (m_front - m))
    m_front_f = 1.0 + 31.0 * (float(tau) ** 1.5)
    return torch.sigmoid(gamma * (m_front_f - m))


class AnnealSolver:
    """Frequency-homotopy ODE/SDE solver combining neural velocity and exact 2-adic gradients."""

    def __init__(
        self,
        model: PhaseSHABackbone,
        dual_basis: DualBasisEngine,
        num_candidates: int = 256,
        num_steps: int = 100,
        gamma: float = 2.0,
        eta_max: float = 0.05,
        beta_max: float = 0.10,
        grad_scale: float = 2.0,
        sde_noise: float = 0.0,
        chunk_size: int = 256,
        use_neural_velocity: bool = True,
    ) -> None:
        self.model = model
        self.dual_basis = dual_basis
        self.num_candidates = num_candidates
        self.num_steps = num_steps
        self.gamma = gamma
        self.eta_max = eta_max
        self.beta_max = beta_max
        self.grad_scale = grad_scale
        self.sde_noise = sde_noise
        self.chunk_size = chunk_size
        self.use_neural_velocity = use_neural_velocity

    def _run_trajectories_chunk(
        self,
        b_chunk: ClampedBoundary,
        use_neural: bool,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run homotopy annealing on a clamped boundary batch (N_traj, ...)."""
        device = b_chunk.device
        rounds = b_chunk.rounds
        n_traj = b_chunk.batch_size
        dt = 1.0 / float(self.num_steps)
        k_spins = self.dual_basis.K_spins
        full_lam = torch.ones(32, dtype=torch.float32, device=device)

        clamp_mask = b_chunk.clamp_mask
        free_mask = 1.0 - clamp_mask

        noise = torch.randn(
            (n_traj, rounds, 3, 32),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        x = b_chunk.apply_clamp(noise)

        for step_idx in range(self.num_steps):
            tau_val = float(step_idx) / float(self.num_steps)
            tau_tensor = torch.full(
                (n_traj,), tau_val, dtype=torch.float32, device=device
            )
            lam = homotopy_lambda(tau_val, gamma=self.gamma, device=device)

            grad_radar, res_feat, _, _, raw_grad = (
                self.dual_basis.compute_radar_and_features(
                    x,
                    b_chunk,
                    lam=lam,
                    grad_scale=self.grad_scale,
                    return_raw_grad=True,
                )
            )

            if use_neural:
                v_theta = self.model(
                    x_nodes=x,
                    grad_radar=grad_radar,
                    clamp_mask=clamp_mask,
                    res_features=res_feat,
                    k_spins=k_spins,
                    tau=tau_tensor,
                    lam=lam,
                )
            else:
                v_theta = torch.zeros_like(x)

            # Double-well binarization gradient: d/dx sum(1 - x^2)^2 = -4 x (1 - x^2)
            x_safe = torch.clamp(x, -2.0, 2.0)
            grad_bin = -4.0 * x_safe * (1.0 - x_safe * x_safe)

            eta_t = self.eta_max * (0.25 + 0.75 * tau_val)
            beta_t = self.beta_max * (tau_val * tau_val)
            total_grad = torch.clamp(raw_grad + beta_t * grad_bin, -10.0, 10.0)

            delta_x = (dt * v_theta - eta_t * total_grad) * free_mask
            if self.sde_noise > 0.0 and step_idx + 1 < self.num_steps:
                sigma_t = self.sde_noise * (1.0 - tau_val)
                sde_term = (
                    sigma_t
                    * (dt**0.5)
                    * torch.randn(
                        x.shape,
                        dtype=torch.float32,
                        device=device,
                        generator=generator,
                    )
                    * free_mask
                )
                delta_x = delta_x + sde_term

            x = b_chunk.apply_clamp(x + delta_x)

        cand_w16 = b_chunk.extract_candidate_w16(x)  # (n_traj, 16)
        cand_digest = sha256_digest(cand_w16, rounds=rounds)  # (n_traj, 8)

        x_disc = torch.where(
            x < 0,
            torch.full_like(x, -1.0),
            torch.full_like(x, 1.0),
        )
        x_disc = b_chunk.apply_clamp(x_disc)
        a_full, e_full, w_full = b_chunk.assemble_full(x_disc)
        energies, _, stats = self.dual_basis.compute_energy_and_features(
            a_full, e_full, w_full, lam=full_lam
        )
        return (
            cand_w16,
            cand_digest,
            x_disc,
            energies,
            stats["m1_energy"],
            stats["m32_energy"],
            x,
        )

    @torch.no_grad()
    def solve_single_boundary(
        self,
        boundary_one: ClampedBoundary,
        target_digest_one: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_trajectories: bool = False,
        use_neural_velocity: Optional[bool] = None,
    ) -> tuple[
        bool,
        int,
        torch.Tensor,
        float,
        float,
        float,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """Run K parallel annealing trajectories for a single digest (8,)."""
        use_neural = (
            self.use_neural_velocity
            if use_neural_velocity is None
            else bool(use_neural_velocity)
        )
        k_total = self.num_candidates
        chunk = max(1, min(self.chunk_size, k_total))

        was_training = self.model.training
        self.model.eval()

        best_energy_val = float("inf")
        best_m1_val = float("inf")
        best_m32_val = float("inf")
        best_w16 = boundary_one.W16_clamped_words[0].clone()
        best_x_disc = boundary_one.x_clamped[0].clone()
        total_hits = 0
        found_hit = False
        last_x_chunk: Optional[torch.Tensor] = None

        try:
            start = 0
            while start < k_total:
                cur_k = min(chunk, k_total - start)
                start += cur_k
                b_chunk = boundary_one.repeat_for_candidates(cur_k)
                cand_w16, cand_digest, x_disc, energies, m1_e, m32_e, x = (
                    self._run_trajectories_chunk(
                        b_chunk, use_neural=use_neural, generator=generator
                    )
                )

                matches = (cand_digest == target_digest_one.unsqueeze(0)).all(dim=-1)
                hits = int(matches.sum().item())
                total_hits += hits

                if hits > 0 and not found_hit:
                    first_hit = int(torch.nonzero(matches, as_tuple=False)[0].item())
                    best_w16 = cand_w16[first_hit].clone()
                    best_x_disc = x_disc[first_hit].clone()
                    best_energy_val = float(energies[first_hit].item())
                    best_m1_val = float(m1_e[first_hit].item())
                    best_m32_val = float(m32_e[first_hit].item())
                    found_hit = True
                elif not found_hit:
                    min_idx = int(torch.argmin(energies).item())
                    min_e = float(energies[min_idx].item())
                    if min_e < best_energy_val:
                        best_energy_val = min_e
                        best_m1_val = float(m1_e[min_idx].item())
                        best_m32_val = float(m32_e[min_idx].item())
                        best_w16 = cand_w16[min_idx].clone()
                        best_x_disc = x_disc[min_idx].clone()

                if return_trajectories:
                    last_x_chunk = x
        finally:
            if was_training:
                self.model.train()

        return (
            total_hits > 0,
            total_hits,
            best_w16,
            best_energy_val,
            best_m1_val,
            best_m32_val,
            best_x_disc,
            last_x_chunk,
        )

    @torch.no_grad()
    def solve_batch_boundaries(
        self,
        boundaries_batch: ClampedBoundary,
        target_digests_batch: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        use_neural_velocity: Optional[bool] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Solve M digests simultaneously with M*K parallel trajectories."""
        use_neural = (
            self.use_neural_velocity
            if use_neural_velocity is None
            else bool(use_neural_velocity)
        )
        m = boundaries_batch.batch_size
        k = self.num_candidates
        rounds = boundaries_batch.rounds

        was_training = self.model.training
        self.model.eval()
        try:
            b_expanded = boundaries_batch.repeat_for_candidates(k)
            cand_w16, cand_digest, x_disc, energies, m1_e, m32_e, _ = (
                self._run_trajectories_chunk(
                    b_expanded, use_neural=use_neural, generator=generator
                )
            )
        finally:
            if was_training:
                self.model.train()

        cand_w16_mk = cand_w16.view(m, k, 16)
        cand_digest_mk = cand_digest.view(m, k, 8)
        x_disc_mk = x_disc.view(m, k, rounds, 3, 32)
        energies_mk = energies.view(m, k)
        m1_mk = m1_e.view(m, k)
        m32_mk = m32_e.view(m, k)

        matches = (cand_digest_mk == target_digests_batch.unsqueeze(1)).all(dim=-1)
        hits = matches.sum(dim=-1)
        solved = hits > 0

        min_indices = torch.argmin(energies_mk, dim=-1)
        hit_indices = torch.argmax(matches.to(torch.int64), dim=-1)
        chosen_idx = torch.where(solved, hit_indices, min_indices)

        batch_idx = torch.arange(m, device=target_digests_batch.device)
        best_w16 = cand_w16_mk[batch_idx, chosen_idx]
        best_energy = energies_mk[batch_idx, chosen_idx]
        best_m1 = m1_mk[batch_idx, chosen_idx]
        best_m32 = m32_mk[batch_idx, chosen_idx]
        best_x_disc = x_disc_mk[batch_idx, chosen_idx]

        return solved, hits, best_w16, best_energy, best_m1, best_m32, best_x_disc

    def solve(
        self,
        target_digests: torch.Tensor,
        domain_mode: str = "int_range",
        num_bits: int = 32,
        msg_bytes_len: int = 4,
        generator: Optional[torch.Generator] = None,
        return_last_trajectories: bool = False,
        use_neural_velocity: Optional[bool] = None,
    ) -> AnnealSolveResult:
        """Solve preimage candidates for a batch of digests (B, 8) or (8,)."""
        if target_digests.ndim == 1:
            target_digests = target_digests.unsqueeze(0)
        bsz = target_digests.shape[0]
        device = target_digests.device
        rounds = self.dual_basis.rounds
        m_per_batch = max(1, self.chunk_size // self.num_candidates)

        if m_per_batch >= 2 and not return_last_trajectories and bsz > 1:
            solved_chunks = []
            hits_chunks = []
            w16_chunks = []
            e_chunks = []
            m1_chunks = []
            m32_chunks = []
            x_chunks = []
            for start in range(0, bsz, m_per_batch):
                dig_sub = target_digests[start : start + m_per_batch]
                b_sub = build_clamped_boundary(
                    digest=dig_sub,
                    rounds=rounds,
                    domain_mode=domain_mode,
                    num_bits=num_bits,
                    msg_bytes_len=msg_bytes_len,
                )
                s_i, h_i, w_i, e_i, m1_i, m32_i, x_i = self.solve_batch_boundaries(
                    b_sub,
                    dig_sub,
                    generator=generator,
                    use_neural_velocity=use_neural_velocity,
                )
                solved_chunks.append(s_i)
                hits_chunks.append(h_i)
                w16_chunks.append(w_i)
                e_chunks.append(e_i)
                m1_chunks.append(m1_i)
                m32_chunks.append(m32_i)
                x_chunks.append(x_i)

            return AnnealSolveResult(
                solved=torch.cat(solved_chunks, dim=0),
                num_hits=torch.cat(hits_chunks, dim=0),
                best_w16=torch.cat(w16_chunks, dim=0),
                best_energy=torch.cat(e_chunks, dim=0),
                best_m1_energy=torch.cat(m1_chunks, dim=0),
                best_m32_energy=torch.cat(m32_chunks, dim=0),
                best_x_disc=torch.cat(x_chunks, dim=0),
                final_x_sample=None,
            )

        solved_list = []
        hits_list = []
        best_w16_list = []
        best_energy_list = []
        best_m1_list = []
        best_m32_list = []
        best_x_list = []
        last_traj: Optional[torch.Tensor] = None

        for i in range(bsz):
            digest_i = target_digests[i]
            boundary_i = build_clamped_boundary(
                digest=digest_i,
                rounds=rounds,
                domain_mode=domain_mode,
                num_bits=num_bits,
                msg_bytes_len=msg_bytes_len,
            )
            solved_i, hits_i, w16_i, energy_i, m1_i, m32_i, x_disc_i, traj_i = (
                self.solve_single_boundary(
                    boundary_i,
                    digest_i,
                    generator=generator,
                    return_trajectories=(return_last_trajectories and i == bsz - 1),
                    use_neural_velocity=use_neural_velocity,
                )
            )
            solved_list.append(solved_i)
            hits_list.append(hits_i)
            best_w16_list.append(w16_i)
            best_energy_list.append(energy_i)
            best_m1_list.append(m1_i)
            best_m32_list.append(m32_i)
            best_x_list.append(x_disc_i)
            if traj_i is not None:
                last_traj = traj_i

        return AnnealSolveResult(
            solved=torch.tensor(solved_list, dtype=torch.bool, device=device),
            num_hits=torch.tensor(hits_list, dtype=torch.int64, device=device),
            best_w16=torch.stack(best_w16_list, dim=0),
            best_energy=torch.tensor(
                best_energy_list, dtype=torch.float32, device=device
            ),
            best_m1_energy=torch.tensor(
                best_m1_list, dtype=torch.float32, device=device
            ),
            best_m32_energy=torch.tensor(
                best_m32_list, dtype=torch.float32, device=device
            ),
            best_x_disc=torch.stack(best_x_list, dim=0),
            final_x_sample=last_traj,
        )
