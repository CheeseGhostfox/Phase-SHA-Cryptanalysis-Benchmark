"""Pure-FP32 orthogonal bases, 32-bit dual-group basis, Cayley parameterization, and sparsity metrics."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def build_sylvester_walsh(n: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Build normalized Sylvester Walsh-Hadamard matrix Q_WHT in O(n), float32."""
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a positive power of 2, got {n}")
    h = torch.tensor([[1.0]], dtype=torch.float64)
    while h.shape[0] < n:
        top = torch.cat([h, h], dim=1)
        bot = torch.cat([h, -h], dim=1)
        h = torch.cat([top, bot], dim=0)
    q = (h / math.sqrt(float(n))).to(torch.float32)
    return q.to(device).contiguous()


def build_real_dft(n: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Build normalized real 2-adic Fourier orthogonal basis Q_DFT in O(n), float32."""
    if n < 2 or (n % 2) != 0:
        raise ValueError(f"n must be even >= 2, got {n}")
    x = torch.arange(n, dtype=torch.float64).unsqueeze(1)  # (n, 1)
    q = torch.zeros((n, n), dtype=torch.float64)
    inv_sqrt_n = 1.0 / math.sqrt(float(n))
    sqrt_2_n = math.sqrt(2.0 / float(n))
    q[:, 0] = inv_sqrt_n
    half = n // 2
    for k in range(1, half):
        angle = (2.0 * math.pi * float(k) / float(n)) * x[:, 0]
        q[:, 2 * k - 1] = sqrt_2_n * torch.cos(angle)
        q[:, 2 * k] = sqrt_2_n * torch.sin(angle)
    sign_alt = torch.where((torch.arange(n) % 2) == 0, 1.0, -1.0).to(torch.float64)
    q[:, n - 1] = inv_sqrt_n * sign_alt
    return q.to(torch.float32).to(device).contiguous()


def build_hybrid_wht_dft_256(device: torch.device | str = "cpu") -> torch.Tensor:
    """Build explicit dual-group tensor product basis Q_WHT^(16) kron Q_DFT^(16) in O(256)."""
    q_wht16 = build_sylvester_walsh(16, device="cpu").to(torch.float64)
    q_dft16 = build_real_dft(16, device="cpu").to(torch.float64)
    q_hyb = torch.kron(q_wht16, q_dft16).to(torch.float32)
    return q_hyb.to(device).contiguous()


def build_haar_ortho(
    n: int, seed: int = 12345, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Sample a uniform Haar random orthogonal matrix in O(n), float32."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    g = torch.randn((n, n), dtype=torch.float64, generator=gen)
    q, r = torch.linalg.qr(g)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
    q = q * diag_sign.unsqueeze(0)
    return q.to(torch.float32).to(device).contiguous()


def build_256_init_bank(device: torch.device | str = "cpu") -> Dict[str, torch.Tensor]:
    """Return the 4 algebraic initializations in O(256)."""
    return {
        "init_std": torch.eye(256, dtype=torch.float32, device=device),
        "init_walsh": build_sylvester_walsh(256, device=device),
        "init_dft": build_real_dft(256, device=device),
        "init_hybrid": build_hybrid_wht_dft_256(device=device),
    }


def build_probe2_init_bank(
    device: torch.device | str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Return (Q_L_inits, Q_R_inits) for Probe II on O(93) x O(186)."""
    q_wht32 = build_sylvester_walsh(32, device=device)

    ql_std = torch.eye(93, dtype=torch.float32, device=device)
    qr_std = torch.eye(186, dtype=torch.float32, device=device)

    ql_wht = torch.eye(93, dtype=torch.float32, device=device)
    ql_wht[:32, :32] = q_wht32

    qr_wht = torch.eye(186, dtype=torch.float32, device=device)
    qr_wht[:32, :32] = q_wht32
    qr_wht[93:125, 93:125] = q_wht32

    ql_dict = {"init_std": ql_std, "init_wht_block": ql_wht}
    qr_dict = {"init_std": qr_std, "init_wht_block": qr_wht}
    return ql_dict, qr_dict


class DualGroup32Basis(nn.Module):
    """Exact 93-dim orthonormal F2 x Z/2^32Z basis Phi_dual(x) in pure real arithmetic."""

    def __init__(self) -> None:
        super().__init__()
        # Precompute analytical projection coeffs (alpha_m, beta_m) and 2x2 whitening matrices
        # for m = 3..32 (30 scales -> 60 dims). m = 2 contributes 1 dim (s_0 * s_1).
        scales = list(range(3, 33))
        masks = [(1 << m) - 1 for m in scales]
        two_pi_over_2m = [2.0 * math.pi / float(1 << m) for m in scales]
        alphas = []
        betas = []
        whiten_mats = []

        for m in scales:
            two_m = float(1 << m)
            alpha_m = 2.0 * math.sqrt(2.0) / two_m
            beta_m = alpha_m / math.tan(math.pi / two_m)
            rho_sq = alpha_m * alpha_m + beta_m * beta_m
            inv_sqrt_1mr = 1.0 / math.sqrt(1.0 - rho_sq)
            rho = math.sqrt(rho_sq)
            u0 = alpha_m / rho
            u1 = beta_m / rho
            coeff = inv_sqrt_1mr - 1.0
            # Sigma_m^{-1/2} = I + coeff * (u u^T)
            w_mat = [
                [1.0 + coeff * u0 * u0, coeff * u0 * u1],
                [coeff * u0 * u1, 1.0 + coeff * u1 * u1],
            ]
            alphas.append(alpha_m)
            betas.append(beta_m)
            whiten_mats.append(w_mat)

        self.register_buffer("bit_shifts", torch.arange(32, dtype=torch.int64), persistent=False)
        self.register_buffer("masks_m", torch.tensor(masks, dtype=torch.int64), persistent=False)
        self.register_buffer(
            "two_pi_inv_2m", torch.tensor(two_pi_over_2m, dtype=torch.float64), persistent=False
        )
        self.register_buffer("alphas", torch.tensor(alphas, dtype=torch.float64), persistent=False)
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float64), persistent=False)
        self.register_buffer(
            "whiten_mats", torch.tensor(whiten_mats, dtype=torch.float64), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Phi_dual(x) of shape (B, 93) float32 from uint32-in-int64 (B,)."""
        if x.ndim != 1:
            raise ValueError(f"Expected 1D tensor (B,), got {tuple(x.shape)}")
        xu = x.to(torch.int64) & 0xFFFFFFFF

        # 32 F2 spin bits (LSB-first)
        bits = (xu.unsqueeze(1) >> self.bit_shifts.unsqueeze(0)) & 1  # (B, 32)
        spins_f64 = 1.0 - 2.0 * bits.to(torch.float64)  # (B, 32)

        # m = 2 residual (1 dim): s_0 * s_1
        res_m2 = (spins_f64[:, 0] * spins_f64[:, 1]).unsqueeze(1)  # (B, 1)

        # m = 3..32 residuals (30 scales -> 60 dims)
        rem = (xu.unsqueeze(1) & self.masks_m.unsqueeze(0)).to(torch.float64)  # (B, 30)
        theta = rem * self.two_pi_inv_2m.unsqueeze(0)  # (B, 30) float64
        sqrt2 = math.sqrt(2.0)
        cm = sqrt2 * torch.cos(theta)
        qm = sqrt2 * torch.sin(theta)

        s_prev = spins_f64[:, 2:32]  # s_{m-1} for m = 3..32
        tc = cm - self.alphas.unsqueeze(0) * s_prev
        tq = qm - self.betas.unsqueeze(0) * s_prev

        # Apply analytical 2x2 whitening: (B, 30, 2)
        w = self.whiten_mats  # (30, 2, 2)
        z0 = w[:, 0, 0].unsqueeze(0) * tc + w[:, 0, 1].unsqueeze(0) * tq
        z1 = w[:, 1, 0].unsqueeze(0) * tc + w[:, 1, 1].unsqueeze(0) * tq
        res_3_32 = torch.stack([z0, z1], dim=-1).reshape(x.shape[0], 60)

        out = torch.cat([spins_f64, res_m2, res_3_32], dim=1)
        return out.to(torch.float32)


class BatchedCayleyOrthogonal(nn.Module):
    """Batched pure-FP32 Cayley parameterization U_theta = Q_init @ V_theta in O(d)."""

    def __init__(
        self,
        q_inits: torch.Tensor,
        init_std: float = 1e-3,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if q_inits.ndim == 2:
            q_inits = q_inits.unsqueeze(0)
        if q_inits.ndim != 3 or q_inits.shape[1] != q_inits.shape[2]:
            raise ValueError(f"Expected (K, d, d), got {tuple(q_inits.shape)}")

        self.num_inits = q_inits.shape[0]
        self.dim = q_inits.shape[1]
        self.register_buffer("q_inits", q_inits.to(torch.float32).contiguous())
        self.register_buffer(
            "eye",
            torch.eye(self.dim, dtype=torch.float32, device=q_inits.device).unsqueeze(0),
            persistent=False,
        )

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        a_init = torch.randn(
            (self.num_inits, self.dim, self.dim), dtype=torch.float32, generator=gen
        ) * init_std
        self.A_theta = nn.Parameter(a_init.to(q_inits.device))

    def forward(self) -> torch.Tensor:
        """Return (K, d, d) pure-FP32 orthogonal matrices U_theta."""
        s = 0.5 * (self.A_theta - self.A_theta.transpose(-2, -1))
        eye_2d = self.eye.squeeze(0)
        # Solve per-init 2D systems to avoid magma_sgetrf_batched bug in CUDA 13.0,
        # followed by one differentiable Newton-Schulz polar step to keep FP32
        # orthogonality error well below 1e-5 for d=256.
        mats = []
        for k in range(self.num_inits):
            v = torch.linalg.solve(eye_2d + s[k], eye_2d - s[k]).transpose(-2, -1)
            u = self.q_inits[k] @ v
            u = 0.5 * u @ (3.0 * eye_2d - u.transpose(-2, -1) @ u)
            mats.append(u)
        return torch.stack(mats, dim=0)


@torch.no_grad()
def check_orthogonal_conservation(
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    right_u: Optional[torch.Tensor] = None,
) -> Tuple[float, float, float]:
    """Compute (ortho_err, det_err, norm_err) on pure-FP32 matrix U (d, d)."""
    d = u.shape[0]
    eye = torch.eye(d, dtype=torch.float32, device=u.device)
    ut_u = u.transpose(-2, -1) @ u
    ortho_err = float(torch.linalg.matrix_norm(ut_u - eye, ord="fro").item())
    _, logabsdet = torch.linalg.slogdet(u)
    det_err = float(torch.abs(torch.exp(logabsdet) - 1.0).item())

    norm_err = 0.0
    if g is not None:
        if right_u is None:
            # G has shape (d, M); compare column l2^2 norms
            orig_sq = g.square().sum(dim=0)
            proj = u.transpose(-2, -1) @ g
            proj_sq = proj.square().sum(dim=0)
            norm_err = float(torch.max(torch.abs(proj_sq - orig_sq)).item())
        else:
            # G has shape (K, d_L, d_R); compare Frobenius^2 norms
            orig_sq = g.square().sum(dim=(-2, -1))
            proj = torch.einsum("li,kij,jr->klr", u, g, right_u)
            proj_sq = proj.square().sum(dim=(-2, -1))
            norm_err = float(torch.max(torch.abs(proj_sq - orig_sq)).item())

    return ortho_err, det_err, norm_err


def sparsity_metrics_tensor(
    z: torch.Tensor, top_k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-column (H_spec, R_l1_l2, E_topK) along dim=-2 for z of shape (..., d, M)."""
    d = z.shape[-2]
    sq = z.square()
    col_energy = sq.sum(dim=-2, keepdim=True)
    p = sq / (col_energy + 1e-12)
    h_col = -(p * torch.log(p + 1e-12)).sum(dim=-2) / math.log(float(d))
    r_col = z.abs().sum(dim=-2) / torch.sqrt(col_energy.squeeze(-2) + 1e-12)
    top_vals = torch.topk(p, k=min(top_k, d), dim=-2).values
    topk_col = top_vals.sum(dim=-2)
    return h_col, r_col, topk_col


def train_cayley_ortho_bank(
    q_init_dict: Dict[str, torch.Tensor],
    train_mat: torch.Tensor,
    mode: str,
    top_k: int,
    steps: int = 400,
    lr_max: float = 3e-2,
    lr_min: float = 1e-3,
    seed: int = 42,
) -> Tuple[str, torch.Tensor, Dict[str, float]]:
    """Train batched Cayley transforms over all initializations and select best by train loss.

    mode:
      - 'columns': train_mat is (d, M); transforms U^T @ train_mat -> (K, d, M)
      - 'conjugate': train_mat is (B_mats, d, d); transforms U^T @ T @ U -> (K, d*d, B_mats)
    """
    init_names = list(q_init_dict.keys())
    q_stack = torch.stack([q_init_dict[k] for k in init_names], dim=0)
    model = BatchedCayleyOrthogonal(q_stack, init_std=1e-3, seed=seed).to(train_mat.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr_min
    )

    d = q_stack.shape[1]
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        u_batch = model()  # (K, d, d)
        if mode == "columns":
            # (K, d, d)^T @ (d, M) -> (K, d, M)
            z = torch.matmul(u_batch.transpose(-2, -1), train_mat)
        elif mode == "conjugate":
            # train_mat: (B, d, d) -> Z: (K, B, d, d) -> (K, d*d, B)
            z_mat = torch.einsum("kij,bil,klm->kbjm", u_batch, train_mat, u_batch)
            z = z_mat.reshape(model.num_inits, train_mat.shape[0], d * d).transpose(-2, -1)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        h_col, r_col, _ = sparsity_metrics_tensor(z, top_k=top_k)
        losses = h_col.mean(dim=-1) + 0.05 * r_col.mean(dim=-1)
        losses.sum().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        u_batch = model()
        if mode == "columns":
            z = torch.matmul(u_batch.transpose(-2, -1), train_mat)
        else:
            z_mat = torch.einsum("kij,bil,klm->kbjm", u_batch, train_mat, u_batch)
            z = z_mat.reshape(model.num_inits, train_mat.shape[0], d * d).transpose(-2, -1)
        h_col, r_col, _ = sparsity_metrics_tensor(z, top_k=top_k)
        final_losses = (h_col.mean(dim=-1) + 0.05 * r_col.mean(dim=-1)).tolist()

    best_idx = int(min(range(len(init_names)), key=lambda i: final_losses[i]))
    best_name = init_names[best_idx]
    best_u = u_batch[best_idx].detach().clone()
    train_loss_map = {name: float(final_losses[i]) for i, name in enumerate(init_names)}
    return best_name, best_u, train_loss_map


def train_cayley_biortho_bank(
    ql_dict: Dict[str, torch.Tensor],
    qr_dict: Dict[str, torch.Tensor],
    train_mats: torch.Tensor,
    top_k: int = 32,
    steps: int = 400,
    lr_max: float = 3e-2,
    lr_min: float = 1e-3,
    seed: int = 42,
) -> Tuple[str, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Train batched biorthogonal (U_L, U_R) on train_mats (B_mats, 93, 186)."""
    init_names = list(ql_dict.keys())
    ql_stack = torch.stack([ql_dict[k] for k in init_names], dim=0)
    qr_stack = torch.stack([qr_dict[k] for k in init_names], dim=0)
    device = train_mats.device

    model_l = BatchedCayleyOrthogonal(ql_stack, init_std=1e-3, seed=seed).to(device)
    model_r = BatchedCayleyOrthogonal(qr_stack, init_std=1e-3, seed=seed + 1).to(device)

    params = list(model_l.parameters()) + list(model_r.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr_max, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr_min
    )

    dl = ql_stack.shape[1]
    dr = qr_stack.shape[1]
    n_mats = train_mats.shape[0]
    n_inits = len(init_names)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        ul = model_l()  # (K, 93, 93)
        ur = model_r()  # (K, 186, 186)
        z_mat = torch.einsum("kli,blj,kjr->kbir", ul, train_mats, ur)
        z = z_mat.reshape(n_inits, n_mats, dl * dr).transpose(-2, -1)
        h_col, r_col, _ = sparsity_metrics_tensor(z, top_k=top_k)
        losses = h_col.mean(dim=-1) + 0.05 * r_col.mean(dim=-1)
        losses.sum().backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        ul = model_l()
        ur = model_r()
        z_mat = torch.einsum("kli,blj,kjr->kbir", ul, train_mats, ur)
        z = z_mat.reshape(n_inits, n_mats, dl * dr).transpose(-2, -1)
        h_col, r_col, _ = sparsity_metrics_tensor(z, top_k=top_k)
        final_losses = (h_col.mean(dim=-1) + 0.05 * r_col.mean(dim=-1)).tolist()

    best_idx = int(min(range(n_inits), key=lambda i: final_losses[i]))
    best_name = init_names[best_idx]
    train_loss_map = {name: float(final_losses[i]) for i, name in enumerate(init_names)}
    return best_name, ul[best_idx].detach().clone(), ur[best_idx].detach().clone(), train_loss_map
