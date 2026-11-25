"""Neural Ugly-Algebra coordinate lifters and single-round SHA-256 Koopman flatteners (pure FP32)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from v11_discovery.p10_ugly_algebra.arx_ops import encode_human_hybrid_basis


class ResMLPBlock(nn.Module):
    def __init__(self, dim: int, expand: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expand),
            nn.GELU(),
            nn.Linear(dim * expand, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class FactoredBilinearOp(nn.Module):
    """Low-rank factored bilinear tensor operator: out(u, v) = W_out((W1 u) * (W2 v)) + L1 u + L2 v."""

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.scale = float(rank) ** -0.5
        self.w1 = nn.Linear(dim, rank, bias=False)
        self.w2 = nn.Linear(dim, rank, bias=False)
        self.w_out = nn.Linear(rank, dim, bias=True)
        self.lin_u = nn.Linear(dim, dim, bias=False)
        self.lin_v = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.w_out.weight, std=0.02)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        bi = self.w_out((self.w1(u) * self.w2(v)) * self.scale)
        return bi + self.lin_u(u) + self.lin_v(v)


class InvertibleConditionedStep(nn.Module):
    """Analytically O(D)-invertible 2-stage coupling operator y = G(v; cond) and exact v = G^{-1}(y; cond)."""

    def __init__(self, dim: int, cond_dim: int = 0, hidden: int | None = None) -> None:
        super().__init__()
        assert dim % 2 == 0, f"dim must be even, got {dim}"
        self.half = dim // 2
        self.cond_dim = int(cond_dim)
        hid = max(64, min(512, dim)) if hidden is None else int(hidden)
        in_d = self.half + self.cond_dim
        self.st1 = nn.Sequential(
            nn.LayerNorm(in_d),
            nn.Linear(in_d, hid),
            nn.GELU(),
            nn.Linear(hid, self.half * 2),
        )
        self.st2 = nn.Sequential(
            nn.LayerNorm(in_d),
            nn.Linear(in_d, hid),
            nn.GELU(),
            nn.Linear(hid, self.half * 2),
        )
        # Initialize near identity so multi-step rollouts start cleanly
        nn.init.zeros_(self.st1[-1].weight)
        nn.init.zeros_(self.st1[-1].bias)
        nn.init.zeros_(self.st2[-1].weight)
        nn.init.zeros_(self.st2[-1].bias)

    def forward(self, v: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        v_a, v_b = v[:, : self.half], v[:, self.half :]
        inp1 = torch.cat([v_b, cond], dim=-1) if cond is not None else v_b
        s1, t1 = self.st1(inp1).chunk(2, dim=-1)
        scale1 = torch.exp(torch.tanh(s1) * 0.5)
        y_a = v_a * scale1 + t1

        inp2 = torch.cat([y_a, cond], dim=-1) if cond is not None else y_a
        s2, t2 = self.st2(inp2).chunk(2, dim=-1)
        scale2 = torch.exp(torch.tanh(s2) * 0.5)
        y_b = v_b * scale2 + t2
        return torch.cat([y_a, y_b], dim=-1)

    def inverse(self, y: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        y_a, y_b = y[:, : self.half], y[:, self.half :]
        inp2 = torch.cat([y_a, cond], dim=-1) if cond is not None else y_a
        s2, t2 = self.st2(inp2).chunk(2, dim=-1)
        scale2 = torch.exp(-torch.tanh(s2) * 0.5)
        v_b = (y_b - t2) * scale2

        inp1 = torch.cat([v_b, cond], dim=-1) if cond is not None else v_b
        s1, t1 = self.st1(inp1).chunk(2, dim=-1)
        scale1 = torch.exp(-torch.tanh(s1) * 0.5)
        v_a = (y_a - t1) * scale1
        return torch.cat([v_a, v_b], dim=-1)


class UglyAlgebraLifter(nn.Module):
    """Learns an unrestricted high-D coordinate system Phi(x) flattening XOR, ADD, and ROTR simultaneously."""

    def __init__(self, n_bits: int, dim: int, rank: int, op_mode: str = "bilinear") -> None:
        super().__init__()
        self.n_bits = int(n_bits)
        self.dim = int(dim)
        self.op_mode = op_mode
        enc_hid = max(128, min(1024, max(dim, n_bits * 8)))

        self.encoder = nn.Sequential(
            nn.Linear(self.n_bits, enc_hid),
            nn.LayerNorm(enc_hid),
            nn.GELU(),
            ResMLPBlock(enc_hid),
            ResMLPBlock(enc_hid),
            nn.Linear(enc_hid, self.dim),
            nn.LayerNorm(self.dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.dim, max(64, enc_hid // 2)),
            nn.GELU(),
            nn.Linear(max(64, enc_hid // 2), self.n_bits),
        )

        if op_mode == "bilinear":
            self.op_xor = FactoredBilinearOp(self.dim, rank)
            self.op_add = FactoredBilinearOp(self.dim, rank)
            self.op_rotr = nn.Linear(self.dim, self.dim, bias=False)
        elif op_mode == "simple_invertible":
            self.op_xor = InvertibleConditionedStep(self.dim, cond_dim=self.dim)
            self.op_add = InvertibleConditionedStep(self.dim, cond_dim=self.dim)
            self.op_rotr = InvertibleConditionedStep(self.dim, cond_dim=0)
        else:
            raise ValueError(f"Unknown op_mode: {op_mode}")

    def encode(self, bits: torch.Tensor) -> torch.Tensor:
        spins = 1.0 - 2.0 * bits
        return self.encoder(spins)

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def apply_xor(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.op_mode == "bilinear":
            return self.op_xor(u, v)
        return self.op_xor(v, cond=u)

    def apply_add(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.op_mode == "bilinear":
            return self.op_add(u, v)
        return self.op_add(v, cond=u)

    def apply_rotr(self, u: torch.Tensor) -> torch.Tensor:
        return self.op_rotr(u)


class HumanBasisOperatorBaseline(nn.Module):
    """Fixed human Walsh+2-adic hybrid coordinate baseline with identical operator capacity."""

    def __init__(self, n_bits: int, dim: int, rank: int, seed: int = 1234) -> None:
        super().__init__()
        self.n_bits = int(n_bits)
        self.dim = int(dim)
        base_dim = 4 * self.n_bits
        if self.dim > base_dim:
            g = torch.Generator(device="cpu").manual_seed(seed)
            proj = torch.randn((base_dim, self.dim - base_dim), generator=g, dtype=torch.float32) * (
                1.0 / (base_dim**0.5)
            )
            self.register_buffer("proj_mat", proj)
        else:
            self.proj_mat = None

        self.op_xor = FactoredBilinearOp(self.dim, rank)
        self.op_add = FactoredBilinearOp(self.dim, rank)
        self.op_rotr = nn.Linear(self.dim, self.dim, bias=False)
        self.decoder = nn.Sequential(
            nn.Linear(self.dim, max(64, min(512, self.dim))),
            nn.GELU(),
            nn.Linear(max(64, min(512, self.dim)), self.n_bits),
        )

    def encode_int(self, x: torch.Tensor) -> torch.Tensor:
        return encode_human_hybrid_basis(x, self.n_bits, self.dim, self.proj_mat)

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def apply_xor(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.op_xor(u, v)

    def apply_add(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.op_add(u, v)

    def apply_rotr(self, u: torch.Tensor) -> torch.Tensor:
        return self.op_rotr(u)


class CayleyLinearStep(nn.Module):
    """Exact orthogonal linear step G = (I - A)(I + A)^{-1} with exact inverse G^{-1} = G^T."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.raw = nn.Parameter(torch.randn((dim, dim), dtype=torch.float32) * (0.05 / (dim**0.5)))

    def get_matrix(self) -> torch.Tensor:
        a = self.raw - self.raw.transpose(0, 1)
        eye = torch.eye(self.dim, dtype=a.dtype, device=a.device)
        # Solve (I + A) G^T = (I - A)^T => G = solve(I + A, I - A)
        return torch.linalg.solve(eye + a, eye - a)

    def forward(self, z: torch.Tensor, g_mat: torch.Tensor | None = None) -> torch.Tensor:
        g = self.get_matrix() if g_mat is None else g_mat
        return z @ g.transpose(0, 1)

    def inverse(self, z_next: torch.Tensor, g_mat: torch.Tensor | None = None) -> torch.Tensor:
        g = self.get_matrix() if g_mat is None else g_mat
        return z_next @ g


class SingleRoundKoopmanFlattener(nn.Module):
    """Maps 256-bit SHA-256 state S -> R^D and learns an exact-invertible single-round step G."""

    def __init__(
        self,
        dim: int = 1024,
        op_mode: str = "invertible_coupling",
        controlled: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.op_mode = op_mode
        self.controlled = bool(controlled)
        hid = max(256, min(1024, dim))

        self.encoder = nn.Sequential(
            nn.Linear(256, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            ResMLPBlock(hid),
            ResMLPBlock(hid),
            nn.Linear(hid, self.dim),
            nn.LayerNorm(self.dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.dim, hid),
            nn.GELU(),
            nn.Linear(hid, 256),
        )

        if self.controlled:
            self.ctrl_dim = 128
            self.ctrl_enc = nn.Sequential(
                nn.Linear(32, 128),
                nn.GELU(),
                nn.Linear(128, self.ctrl_dim),
                nn.LayerNorm(self.ctrl_dim),
            )
        else:
            self.ctrl_dim = 0
            self.ctrl_enc = None

        if op_mode == "linear_ortho":
            self.step_op = CayleyLinearStep(self.dim)
            if self.controlled:
                self.ctrl_proj = nn.Linear(self.ctrl_dim, self.dim)
        elif op_mode == "invertible_coupling":
            self.step_1 = InvertibleConditionedStep(self.dim, cond_dim=self.ctrl_dim)
            self.step_2 = InvertibleConditionedStep(self.dim, cond_dim=self.ctrl_dim)
        else:
            raise ValueError(f"Unknown op_mode: {op_mode}")

    def encode_state(self, bits256: torch.Tensor) -> torch.Tensor:
        return self.encoder(1.0 - 2.0 * bits256)

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def encode_ctrl(self, w_bits32: torch.Tensor | None) -> torch.Tensor | None:
        if (not self.controlled) or (w_bits32 is None) or (self.ctrl_enc is None):
            return None
        return self.ctrl_enc(1.0 - 2.0 * w_bits32)

    def forward_step(
        self, z: torch.Tensor, ctrl: torch.Tensor | None = None, g_cache: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.op_mode == "linear_ortho":
            out = self.step_op(z, g_mat=g_cache)
            if ctrl is not None:
                out = out + self.ctrl_proj(ctrl)
            return out
        z1 = self.step_1(z, cond=ctrl)
        return self.step_2(z1, cond=ctrl)

    def inverse_step(
        self, z_next: torch.Tensor, ctrl: torch.Tensor | None = None, g_cache: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.op_mode == "linear_ortho":
            inp = z_next - self.ctrl_proj(ctrl) if ctrl is not None else z_next
            return self.step_op.inverse(inp, g_mat=g_cache)
        z1 = self.step_2.inverse(z_next, cond=ctrl)
        return self.step_1.inverse(z1, cond=ctrl)
