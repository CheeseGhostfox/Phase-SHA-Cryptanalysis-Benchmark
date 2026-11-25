"""Phase-SHA V2 unrolled neural-physics inference engine (all 32 scales active)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from src.bit_grid_backbone import BitGridBackbone
from src.boundary_clamp import ClampedBoundary, build_clamped_boundary
from src.dual_basis import DualBasisEngine
from src.sha256_ref import sha256_digest


@dataclass
class SolveResultV2:
    """Results from unrolled neural-physics V2 inference over target digests."""

    solved: torch.Tensor  # (B,) bool
    num_hits: torch.Tensor  # (B,) int64
    unique_ratio: torch.Tensor  # (B,) float32
    best_w16: torch.Tensor  # (B, 16) int64
    best_energy: torch.Tensor  # (B,) float32
    best_m1_energy: torch.Tensor  # (B,) float32
    best_m32_energy: torch.Tensor  # (B,) float32
    best_x_disc: torch.Tensor  # (B, R, 3, 32) float32
    final_x_sample: Optional[torch.Tensor] = None  # (K, R, 3, 32)

    @property
    def solve_rate(self) -> float:
        return float(self.solved.float().mean().item())

    @property
    def mean_unique_ratio(self) -> float:
        return float(self.unique_ratio.float().mean().item())


class UnrolledSolverV2:
    """Iterative neural-physics refinement solver with full 32-scale 2-adic coupling."""

    def __init__(
        self,
        model: BitGridBackbone,
        dual_basis: DualBasisEngine,
        num_candidates: int = 256,
        num_steps: int = 32,
        damp: float = 0.10,
        eta_max: float = 0.02,
        beta_max: float = 0.25,
        sde_noise_min: float = 0.10,
        sde_noise_max: float = 0.45,
        grad_scale: float = 2.0,
        chunk_size: int = 1024,
        use_neural_velocity: bool = True,
    ) -> None:
        self.model = model
        self.dual_basis = dual_basis
        self.num_candidates = num_candidates
        self.num_steps = num_steps
        self.damp = damp
        self.eta_max = eta_max
        self.beta_max = beta_max
        self.sde_noise_min = sde_noise_min
        self.sde_noise_max = sde_noise_max
        self.grad_scale = grad_scale
        self.chunk_size = chunk_size
        self.use_neural_velocity = use_neural_velocity

    def _build_sigma_and_sde(
        self, num_digests: int, k_total: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigmas = torch.linspace(0.25, 1.80, k_total, device=device)
        sigmas[0] = 0.0
        sde = torch.linspace(
            self.sde_noise_min, self.sde_noise_max, k_total, device=device
        )
        sde[0] = 0.0
        return (
            sigmas.unsqueeze(0).expand(num_digests, k_total),
            sde.unsqueeze(0).expand(num_digests, k_total),
        )

    def _init_diverse_states(
        self,
        b_chunk: ClampedBoundary,
        sigmas_flat: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Initialize index-0 trajectory at 0 and 1..K-1 with diverse continuous noise."""
        device = b_chunk.device
        rounds = b_chunk.rounds
        n_traj = b_chunk.batch_size
        eps = torch.randn(
            (n_traj, rounds, 3, 32),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        x0 = torch.tanh(sigmas_flat.view(n_traj, 1, 1, 1) * eps)
        return b_chunk.apply_clamp(x0)

    def _refine_loop(
        self,
        b_chunk: ClampedBoundary,
        x_init: torch.Tensor,
        use_neural: bool,
        sde_scales: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run K_infer unrolled steps keeping all 32 2-adic scales active."""
        device = b_chunk.device
        n_traj = b_chunk.batch_size
        k_spins = self.dual_basis.K_spins
        full_lam = torch.ones(32, dtype=torch.float32, device=device)
        clamp_mask = b_chunk.clamp_mask
        free_mask = 1.0 - clamp_mask

        x = b_chunk.apply_clamp(x_init)
        h_prev: Optional[torch.Tensor] = None
        last_logits = torch.zeros_like(x)
        denom = float(max(1, self.num_steps - 1))

        for step_idx in range(self.num_steps):
            frac = float(step_idx) / denom
            step_tensor = torch.full((n_traj,), frac, dtype=torch.float32, device=device)

            grad_radar, res_feat, _, _, raw_grad = (
                self.dual_basis.compute_radar_and_features(
                    x,
                    b_chunk,
                    lam=full_lam,
                    grad_scale=self.grad_scale,
                    return_raw_grad=True,
                )
            )

            if use_neural:
                logits, h_prev = self.model(
                    x_nodes=x,
                    grad_radar=grad_radar,
                    clamp_mask=clamp_mask,
                    res_features=res_feat,
                    k_spins=k_spins,
                    step_frac=step_tensor,
                    lam=full_lam,
                    h_prev=h_prev,
                )
                last_logits = logits
                x_neural = torch.tanh(logits)
                x_base = (1.0 - self.damp) * x + self.damp * x_neural
            else:
                x_base = x

            x_safe = torch.clamp(x, -2.0, 2.0)
            grad_bin = -4.0 * x_safe * (1.0 - x_safe * x_safe)
            eta_s = self.eta_max * (0.2 + 0.8 * frac)
            beta_s = self.beta_max * (frac * frac)
            guide = torch.clamp(raw_grad + beta_s * grad_bin, -10.0, 10.0)

            temp = max(0.0, 1.0 - frac / 0.80)
            if sde_scales is not None and temp > 0.0:
                eps = torch.randn(
                    x.shape, dtype=torch.float32, device=device, generator=generator
                )
                sde_term = sde_scales.view(-1, 1, 1, 1) * temp * eps * free_mask
            else:
                sde_term = 0.0

            x_next = torch.clamp(
                x_base - eta_s * guide * free_mask + sde_term, -1.0, 1.0
            )
            x = b_chunk.apply_clamp(x_next)

        return x, last_logits

    @torch.no_grad()
    def infer_deterministic_zero(
        self,
        boundary: ClampedBoundary,
        use_neural_velocity: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run deterministic zero-init inference over a batch of boundaries."""
        use_neural = (
            self.use_neural_velocity
            if use_neural_velocity is None
            else bool(use_neural_velocity)
        )
        was_training = self.model.training
        self.model.eval()
        try:
            x_zero = boundary.apply_clamp(torch.zeros_like(boundary.x_clamped))
            x_final, last_logits = self._refine_loop(
                boundary, x_zero, use_neural=use_neural, sde_scales=None
            )
            x_disc = torch.where(
                x_final < 0,
                torch.full_like(x_final, -1.0),
                torch.full_like(x_final, 1.0),
            )
            x_disc = boundary.apply_clamp(x_disc)
            a_full, e_full, w_full = boundary.assemble_full(x_disc)
            full_lam = torch.ones(32, dtype=torch.float32, device=boundary.device)
            _, _, stats = self.dual_basis.compute_energy_and_features(
                a_full, e_full, w_full, lam=full_lam
            )
        finally:
            if was_training:
                self.model.train()

        return last_logits, x_disc, stats["m1_energy"], stats["m32_energy"]

    def _evaluate_chunk(
        self,
        b_chunk: ClampedBoundary,
        x_final: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        rounds = b_chunk.rounds
        full_lam = torch.ones(32, dtype=torch.float32, device=b_chunk.device)
        cand_w16 = b_chunk.extract_candidate_w16(x_final)
        cand_digest = sha256_digest(cand_w16, rounds=rounds)

        x_disc = torch.where(
            x_final < 0,
            torch.full_like(x_final, -1.0),
            torch.full_like(x_final, 1.0),
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
        torch.Tensor,
    ]:
        """Solve M digests in parallel using M*K trajectories."""
        use_neural = (
            self.use_neural_velocity
            if use_neural_velocity is None
            else bool(use_neural_velocity)
        )
        m = boundaries_batch.batch_size
        k = self.num_candidates
        rounds = boundaries_batch.rounds
        device = target_digests_batch.device

        was_training = self.model.training
        self.model.eval()
        try:
            b_exp = boundaries_batch.repeat_for_candidates(k)
            sigmas_mk, sde_mk = self._build_sigma_and_sde(m, k, device)
            x0 = self._init_diverse_states(
                b_exp, sigmas_flat=sigmas_mk.reshape(-1), generator=generator
            )
            x_final, _ = self._refine_loop(
                b_exp,
                x0,
                use_neural=use_neural,
                sde_scales=sde_mk.reshape(-1),
                generator=generator,
            )
            cand_w16, cand_digest, x_disc, energies, m1_e, m32_e = self._evaluate_chunk(
                b_exp, x_final
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

        batch_idx = torch.arange(m, device=device)
        best_w16 = cand_w16_mk[batch_idx, chosen_idx]
        best_energy = energies_mk[batch_idx, chosen_idx]
        best_m1 = m1_mk[batch_idx, chosen_idx]
        best_m32 = m32_mk[batch_idx, chosen_idx]
        best_x_disc = x_disc_mk[batch_idx, chosen_idx]

        unique_ratios = torch.tensor(
            [
                float(torch.unique(cand_w16_mk[i, :, 0]).numel()) / float(k)
                for i in range(m)
            ],
            dtype=torch.float32,
            device=device,
        )

        return (
            solved,
            hits,
            unique_ratios,
            best_w16,
            best_energy,
            best_m1,
            best_m32,
            best_x_disc,
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
        float,
        torch.Tensor,
        float,
        float,
        float,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """Solve a single digest (chunking K if chunk_size < K)."""
        use_neural = (
            self.use_neural_velocity
            if use_neural_velocity is None
            else bool(use_neural_velocity)
        )
        k_total = self.num_candidates
        chunk = max(1, min(self.chunk_size, k_total))
        device = boundary_one.device
        sigmas_all, sde_all = self._build_sigma_and_sde(1, k_total, device)
        sigmas_1d = sigmas_all[0]
        sde_1d = sde_all[0]

        was_training = self.model.training
        self.model.eval()

        best_energy_val = float("inf")
        best_m1_val = float("inf")
        best_m32_val = float("inf")
        best_w16 = boundary_one.W16_clamped_words[0].clone()
        best_x_disc = boundary_one.x_clamped[0].clone()
        total_hits = 0
        found_hit = False
        all_w0 = []
        last_x_chunk: Optional[torch.Tensor] = None

        try:
            start = 0
            while start < k_total:
                cur_k = min(chunk, k_total - start)
                sig_sub = sigmas_1d[start : start + cur_k]
                sde_sub = sde_1d[start : start + cur_k]
                start += cur_k
                b_chunk = boundary_one.repeat_for_candidates(cur_k)
                x0 = self._init_diverse_states(
                    b_chunk, sigmas_flat=sig_sub, generator=generator
                )
                x_final, _ = self._refine_loop(
                    b_chunk,
                    x0,
                    use_neural=use_neural,
                    sde_scales=sde_sub,
                    generator=generator,
                )
                cand_w16, cand_digest, x_disc, energies, m1_e, m32_e = (
                    self._evaluate_chunk(b_chunk, x_final)
                )
                all_w0.append(cand_w16[:, 0])

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
                    last_x_chunk = x_final
        finally:
            if was_training:
                self.model.train()

        unique_ratio = float(torch.unique(torch.cat(all_w0, dim=0)).numel()) / float(
            k_total
        )
        return (
            total_hits > 0,
            total_hits,
            unique_ratio,
            best_w16,
            best_energy_val,
            best_m1_val,
            best_m32_val,
            best_x_disc,
            last_x_chunk,
        )

    def solve(
        self,
        target_digests: torch.Tensor,
        domain_mode: str = "int_range",
        num_bits: int = 32,
        msg_bytes_len: int = 4,
        generator: Optional[torch.Generator] = None,
        return_last_trajectories: bool = False,
        use_neural_velocity: Optional[bool] = None,
    ) -> SolveResultV2:
        """Solve preimage candidates for a batch of digests (B, 8) or (8,)."""
        if target_digests.ndim == 1:
            target_digests = target_digests.unsqueeze(0)
        bsz = target_digests.shape[0]
        device = target_digests.device
        rounds = self.dual_basis.rounds
        m_per_batch = max(1, self.chunk_size // self.num_candidates)

        if m_per_batch >= 2 and not return_last_trajectories and bsz > 1:
            solved_l, hits_l, uniq_l, w16_l, e_l, m1_l, m32_l, x_l = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for start in range(0, bsz, m_per_batch):
                dig_sub = target_digests[start : start + m_per_batch]
                b_sub = build_clamped_boundary(
                    digest=dig_sub,
                    rounds=rounds,
                    domain_mode=domain_mode,
                    num_bits=num_bits,
                    msg_bytes_len=msg_bytes_len,
                )
                s_i, h_i, u_i, w_i, e_i, m1_i, m32_i, x_i = self.solve_batch_boundaries(
                    b_sub,
                    dig_sub,
                    generator=generator,
                    use_neural_velocity=use_neural_velocity,
                )
                solved_l.append(s_i)
                hits_l.append(h_i)
                uniq_l.append(u_i)
                w16_l.append(w_i)
                e_l.append(e_i)
                m1_l.append(m1_i)
                m32_l.append(m32_i)
                x_l.append(x_i)

            return SolveResultV2(
                solved=torch.cat(solved_l, dim=0),
                num_hits=torch.cat(hits_l, dim=0),
                unique_ratio=torch.cat(uniq_l, dim=0),
                best_w16=torch.cat(w16_l, dim=0),
                best_energy=torch.cat(e_l, dim=0),
                best_m1_energy=torch.cat(m1_l, dim=0),
                best_m32_energy=torch.cat(m32_l, dim=0),
                best_x_disc=torch.cat(x_l, dim=0),
                final_x_sample=None,
            )

        solved_l, hits_l, uniq_l, w16_l, e_l, m1_l, m32_l, x_l = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        last_traj: Optional[torch.Tensor] = None
        for i in range(bsz):
            dig_i = target_digests[i]
            b_i = build_clamped_boundary(
                digest=dig_i,
                rounds=rounds,
                domain_mode=domain_mode,
                num_bits=num_bits,
                msg_bytes_len=msg_bytes_len,
            )
            s_i, h_i, u_i, w_i, e_i, m1_i, m32_i, x_i, traj_i = (
                self.solve_single_boundary(
                    b_i,
                    dig_i,
                    generator=generator,
                    return_trajectories=(return_last_trajectories and i == bsz - 1),
                    use_neural_velocity=use_neural_velocity,
                )
            )
            solved_l.append(s_i)
            hits_l.append(h_i)
            uniq_l.append(u_i)
            w16_l.append(w_i)
            e_l.append(e_i)
            m1_l.append(m1_i)
            m32_l.append(m32_i)
            x_l.append(x_i)
            if traj_i is not None:
                last_traj = traj_i

        return SolveResultV2(
            solved=torch.tensor(solved_l, dtype=torch.bool, device=device),
            num_hits=torch.tensor(hits_l, dtype=torch.int64, device=device),
            unique_ratio=torch.tensor(uniq_l, dtype=torch.float32, device=device),
            best_w16=torch.stack(w16_l, dim=0),
            best_energy=torch.tensor(e_l, dtype=torch.float32, device=device),
            best_m1_energy=torch.tensor(m1_l, dtype=torch.float32, device=device),
            best_m32_energy=torch.tensor(m32_l, dtype=torch.float32, device=device),
            best_x_disc=torch.stack(x_l, dim=0),
            final_x_sample=last_traj,
        )
