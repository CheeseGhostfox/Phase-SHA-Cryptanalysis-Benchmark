"""Arbitrary bit-level mask boundary container and V4E partial-prefix adapter (strict FP32)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import torch

from src.boundary_clamp import ClampedBoundary, build_clamped_boundary
from src.dual_basis import spin_Ch, spin_Maj, spin_Sigma0, spin_Sigma1
from src.sha256_ref import (
    MASK32,
    get_iv_tensor,
    get_k_tensor,
    words_to_bits,
    words_to_spins,
)
from src.v3_solver import spin_add32
from src.v4_manifolds import BaseManifold, normalize_manifold


@dataclass
class ManifoldBoundary:
    """General mask-driven SHA-256 boundary supporting full digests and V4E partial prefixes."""

    rounds: int
    A_full_clamped: torch.Tensor  # (B, R + 4, 32) float32 spins
    A_full_mask: torch.Tensor  # (B, R + 4, 32) float32, 1=locked, 0=free
    E_full_clamped: torch.Tensor  # (B, R + 4, 32) float32 spins
    E_full_mask: torch.Tensor  # (B, R + 4, 32) float32
    W_full_clamped: torch.Tensor  # (B, R, 32) float32 spins
    W_full_mask: torch.Tensor  # (B, R, 32) float32
    A_words_clamped: torch.Tensor  # (B, R + 4) int64
    E_words_clamped: torch.Tensor  # (B, R + 4) int64
    W16_clamped_words: torch.Tensor  # (B, 16) int64
    W16_mask_bits: torch.Tensor  # (B, 16, 32) float32
    digest_target_bits: Optional[torch.Tensor] = None  # (B, 8, 32) float32 in {0, 1}
    digest_target_mask: Optional[torch.Tensor] = None  # (B, 8, 32) float32 in {0, 1}
    iv_words: Optional[torch.Tensor] = None  # (8,) int64
    is_full_digest_clamped: bool = True
    _has_free_right_cache: Optional[bool] = field(default=None, init=False, repr=False, compare=False)

    @property
    def batch_size(self) -> int:
        return int(self.A_full_clamped.shape[0])

    @property
    def device(self) -> torch.device:
        return self.A_full_clamped.device

    @property
    def x_clamped(self) -> torch.Tensor:
        r = self.rounds
        return torch.stack(
            [
                self.A_full_clamped[:, 3 : 3 + r],
                self.E_full_clamped[:, 3 : 3 + r],
                self.W_full_clamped[:, :r],
            ],
            dim=2,
        )

    @property
    def clamp_mask(self) -> torch.Tensor:
        r = self.rounds
        return torch.stack(
            [
                self.A_full_mask[:, 3 : 3 + r],
                self.E_full_mask[:, 3 : 3 + r],
                self.W_full_mask[:, :r],
            ],
            dim=2,
        )

    @property
    def has_free_right_boundary(self) -> bool:
        if self._has_free_right_cache is None:
            if self.is_full_digest_clamped:
                self._has_free_right_cache = False
            else:
                r = self.rounds
                self._has_free_right_cache = bool(
                    (self.A_full_mask[:, r + 3] < 0.5).any().item()
                    or (self.E_full_mask[:, r + 3] < 0.5).any().item()
                )
        return self._has_free_right_cache

    def prepare_for_manifold(self, manifold: BaseManifold) -> "PreparedManifoldBoundary":
        """Pre-embed static left/right boundaries and masks once per trajectory batch."""
        r = self.rounds
        clamp_m = self.clamp_mask.to(torch.float32)
        free_m = 1.0 - clamp_m
        clamp_m_exp = clamp_m.unsqueeze(-1)
        free_m_exp = free_m.unsqueeze(-1)
        u_clamped_masked = clamp_m_exp * manifold.embed_spins(self.x_clamped)
        u_a_left = manifold.embed_spins(self.A_full_clamped[:, 0:3])
        u_e_left = manifold.embed_spins(self.E_full_clamped[:, 0:3])
        has_free_right = self.has_free_right_boundary
        if not has_free_right:
            u_a_right = manifold.embed_spins(self.A_full_clamped[:, r + 3 : r + 4])
            u_e_right = manifold.embed_spins(self.E_full_clamped[:, r + 3 : r + 4])
            k_last_spins = None
        else:
            u_a_right = None
            u_e_right = None
            k_last_w = get_k_tensor(r, device=clamp_m.device)[r - 1 : r]
            k_last_spins = words_to_spins(k_last_w, dtype=torch.float32).expand(
                clamp_m.shape[0], 32
            )
        has_prefix = bool(
            self.digest_target_mask is not None
            and self.digest_target_bits is not None
            and self.iv_words is not None
            and float(self.digest_target_mask.max().item()) > 0.0
        )
        return PreparedManifoldBoundary(
            rounds=r,
            clamp_mask=clamp_m,
            free_mask=free_m,
            clamp_mask_exp=clamp_m_exp,
            free_mask_exp=free_m_exp,
            u_clamped_masked=u_clamped_masked,
            u_a_left=u_a_left,
            u_e_left=u_e_left,
            u_a_right=u_a_right,
            u_e_right=u_e_right,
            a_right_mask=self.A_full_mask[:, r + 3],
            e_right_mask=self.E_full_mask[:, r + 3],
            a_right_clamped=self.A_full_clamped[:, r + 3],
            e_right_clamped=self.E_full_clamped[:, r + 3],
            k_last_spins=k_last_spins,
            digest_target_bits=self.digest_target_bits,
            digest_target_mask=self.digest_target_mask,
            iv_words=self.iv_words,
            has_free_right=has_free_right,
            has_digest_prefix=has_prefix,
        )

    @classmethod
    def from_clamped_boundary(cls, cb: ClampedBoundary) -> "ManifoldBoundary":
        iv = get_iv_tensor(cb.device)
        return cls(
            rounds=cb.rounds,
            A_full_clamped=cb.A_full_clamped,
            A_full_mask=cb.A_full_mask,
            E_full_clamped=cb.E_full_clamped,
            E_full_mask=cb.E_full_mask,
            W_full_clamped=cb.W_full_clamped,
            W_full_mask=cb.W_full_mask,
            A_words_clamped=cb.A_words_clamped,
            E_words_clamped=cb.E_words_clamped,
            W16_clamped_words=cb.W16_clamped_words,
            W16_mask_bits=cb.W16_mask_bits,
            digest_target_bits=None,
            digest_target_mask=None,
            iv_words=iv,
            is_full_digest_clamped=True,
        )

    def to_clamped_boundary(self) -> ClampedBoundary:
        return ClampedBoundary(
            rounds=self.rounds,
            A_full_clamped=self.A_full_clamped,
            A_full_mask=self.A_full_mask,
            E_full_clamped=self.E_full_clamped,
            E_full_mask=self.E_full_mask,
            W_full_clamped=self.W_full_clamped,
            W_full_mask=self.W_full_mask,
            A_words_clamped=self.A_words_clamped,
            E_words_clamped=self.E_words_clamped,
            W16_clamped_words=self.W16_clamped_words,
            W16_mask_bits=self.W16_mask_bits,
        )

    def apply_clamp_manifold(
        self, u_nodes: torch.Tensor, manifold: BaseManifold
    ) -> torch.Tensor:
        """Hard-overwrite locked coordinates in u_nodes (B, R, 3, 32, D) with discrete poles."""
        mask = self.clamp_mask.unsqueeze(-1)  # (B, R, 3, 32, 1)
        u_clamped = manifold.embed_spins(self.x_clamped)
        return normalize_manifold(mask * u_clamped + (1.0 - mask) * u_nodes.to(torch.float32))

    def assemble_full_manifold(
        self, u_nodes: torch.Tensor, manifold: BaseManifold
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble full (U_A, U_E, U_W) supporting both full-digest and free right boundaries."""
        r = self.rounds
        u_f = u_nodes.to(torch.float32)
        u_a_mid = u_f[:, :, 0, :, :]  # A_{0..R-1}
        u_e_mid = u_f[:, :, 1, :, :]  # E_{0..R-1}
        u_w_full = u_f[:, :, 2, :, :]  # W_{0..R-1}

        u_a_left = manifold.embed_spins(self.A_full_clamped[:, 0:3])
        u_e_left = manifold.embed_spins(self.E_full_clamped[:, 0:3])
        u_a_pre = torch.cat([u_a_left, u_a_mid], dim=1)  # A_{-3..R-1}, len R+3
        u_e_pre = torch.cat([u_e_left, u_e_mid], dim=1)  # E_{-3..R-1}, len R+3

        if not self.has_free_right_boundary:
            u_a_right = manifold.embed_spins(self.A_full_clamped[:, r + 3 : r + 4])
            u_e_right = manifold.embed_spins(self.E_full_clamped[:, r + 3 : r + 4])
        else:
            # Synthesize A_R, E_R dynamically from round R-1 states for V4E partial-digest targets
            a_pre_s = manifold.project_spin(u_a_pre)
            e_pre_s = manifold.project_spin(u_e_pre)
            w_last_s = manifold.project_spin(u_w_full[:, r - 1])
            k_last_w = get_k_tensor(r, device=u_f.device)[r - 1 : r]
            k_last_s = words_to_spins(k_last_w, dtype=torch.float32).expand(
                u_f.shape[0], 32
            )

            s1_e = spin_Sigma1(e_pre_s[:, r + 2])
            ch_e = spin_Ch(e_pre_s[:, r + 2], e_pre_s[:, r + 1], e_pre_s[:, r])
            t1_s = spin_add32(
                spin_add32(spin_add32(e_pre_s[:, r - 1], s1_e), spin_add32(ch_e, k_last_s)),
                w_last_s,
            )
            s0_a = spin_Sigma0(a_pre_s[:, r + 2])
            maj_a = spin_Maj(a_pre_s[:, r + 2], a_pre_s[:, r + 1], a_pre_s[:, r])
            t2_s = spin_add32(s0_a, maj_a)
            e_r_dyn = spin_add32(a_pre_s[:, r - 1], t1_s)
            a_r_dyn = spin_add32(t1_s, t2_s)

            ma = self.A_full_mask[:, r + 3]
            me = self.E_full_mask[:, r + 3]
            a_r_s = ma * self.A_full_clamped[:, r + 3] + (1.0 - ma) * a_r_dyn
            e_r_s = me * self.E_full_clamped[:, r + 3] + (1.0 - me) * e_r_dyn
            u_a_right = manifold.embed_spins_with_equator(a_r_s).unsqueeze(1)
            u_e_right = manifold.embed_spins_with_equator(e_r_s).unsqueeze(1)

        u_a_full = torch.cat([u_a_pre, u_a_right], dim=1)
        u_e_full = torch.cat([u_e_pre, u_e_right], dim=1)
        return u_a_full, u_e_full, u_w_full

    def extract_candidate_w16(
        self, u_nodes: torch.Tensor, manifold: BaseManifold
    ) -> torch.Tensor:
        spins = manifold.project_spin(u_nodes)
        return self.to_clamped_boundary().extract_candidate_w16(spins)

    def repeat_for_candidates(self, num_candidates: int) -> "ManifoldBoundary":
        k = int(num_candidates)
        return ManifoldBoundary(
            rounds=self.rounds,
            A_full_clamped=self.A_full_clamped.repeat_interleave(k, dim=0),
            A_full_mask=self.A_full_mask.repeat_interleave(k, dim=0),
            E_full_clamped=self.E_full_clamped.repeat_interleave(k, dim=0),
            E_full_mask=self.E_full_mask.repeat_interleave(k, dim=0),
            W_full_clamped=self.W_full_clamped.repeat_interleave(k, dim=0),
            W_full_mask=self.W_full_mask.repeat_interleave(k, dim=0),
            A_words_clamped=self.A_words_clamped.repeat_interleave(k, dim=0),
            E_words_clamped=self.E_words_clamped.repeat_interleave(k, dim=0),
            W16_clamped_words=self.W16_clamped_words.repeat_interleave(k, dim=0),
            W16_mask_bits=self.W16_mask_bits.repeat_interleave(k, dim=0),
            digest_target_bits=(
                None
                if self.digest_target_bits is None
                else self.digest_target_bits.repeat_interleave(k, dim=0)
            ),
            digest_target_mask=(
                None
                if self.digest_target_mask is None
                else self.digest_target_mask.repeat_interleave(k, dim=0)
            ),
            iv_words=self.iv_words,
            is_full_digest_clamped=self.is_full_digest_clamped,
        )


@dataclass
class PreparedManifoldBoundary:
    """Pre-embedded masks & static boundaries eliminating per-step GPU syncs and allocations."""

    rounds: int
    clamp_mask: torch.Tensor
    free_mask: torch.Tensor
    clamp_mask_exp: torch.Tensor
    free_mask_exp: torch.Tensor
    u_clamped_masked: torch.Tensor
    u_a_left: torch.Tensor
    u_e_left: torch.Tensor
    u_a_right: Optional[torch.Tensor]
    u_e_right: Optional[torch.Tensor]
    a_right_mask: torch.Tensor
    e_right_mask: torch.Tensor
    a_right_clamped: torch.Tensor
    e_right_clamped: torch.Tensor
    k_last_spins: Optional[torch.Tensor]
    digest_target_bits: Optional[torch.Tensor]
    digest_target_mask: Optional[torch.Tensor]
    iv_words: Optional[torch.Tensor]
    has_free_right: bool
    has_digest_prefix: bool


@torch.amp.autocast("cuda", enabled=False)
def apply_clamp_prepared(
    u_nodes: torch.Tensor, prep: PreparedManifoldBoundary
) -> torch.Tensor:
    """Apply pre-embedded static boundary clamp without recomputing embeddings or masks."""
    return normalize_manifold(
        prep.u_clamped_masked + prep.free_mask_exp * u_nodes.to(torch.float32)
    )


@torch.amp.autocast("cuda", enabled=False)
def assemble_full_prepared(
    u_nodes: torch.Tensor,
    prep: PreparedManifoldBoundary,
    manifold: BaseManifold,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble (U_A, U_E, U_W) from PreparedManifoldBoundary with zero CPU-GPU sync."""
    r = prep.rounds
    u_f = u_nodes.to(torch.float32)
    u_a_mid = u_f[:, :, 0, :, :]
    u_e_mid = u_f[:, :, 1, :, :]
    u_w_full = u_f[:, :, 2, :, :]

    u_a_pre = torch.cat([prep.u_a_left, u_a_mid], dim=1)
    u_e_pre = torch.cat([prep.u_e_left, u_e_mid], dim=1)
    if not prep.has_free_right and prep.u_a_right is not None and prep.u_e_right is not None:
        return (
            torch.cat([u_a_pre, prep.u_a_right], dim=1),
            torch.cat([u_e_pre, prep.u_e_right], dim=1),
            u_w_full,
        )

    a_pre_s = manifold.project_spin(u_a_pre)
    e_pre_s = manifold.project_spin(u_e_pre)
    w_last_s = manifold.project_spin(u_w_full[:, r - 1])
    s1_e = spin_Sigma1(e_pre_s[:, r + 2])
    ch_e = spin_Ch(e_pre_s[:, r + 2], e_pre_s[:, r + 1], e_pre_s[:, r])
    t1_s = spin_add32(
        spin_add32(spin_add32(e_pre_s[:, r - 1], s1_e), spin_add32(ch_e, prep.k_last_spins)),
        w_last_s,
    )
    s0_a = spin_Sigma0(a_pre_s[:, r + 2])
    maj_a = spin_Maj(a_pre_s[:, r + 2], a_pre_s[:, r + 1], a_pre_s[:, r])
    t2_s = spin_add32(s0_a, maj_a)
    e_r_dyn = spin_add32(a_pre_s[:, r - 1], t1_s)
    a_r_dyn = spin_add32(t1_s, t2_s)

    a_r_s = prep.a_right_mask * prep.a_right_clamped + (1.0 - prep.a_right_mask) * a_r_dyn
    e_r_s = prep.e_right_mask * prep.e_right_clamped + (1.0 - prep.e_right_mask) * e_r_dyn
    u_a_right = manifold.embed_spins_with_equator(a_r_s).unsqueeze(1)
    u_e_right = manifold.embed_spins_with_equator(e_r_s).unsqueeze(1)
    return (
        torch.cat([u_a_pre, u_a_right], dim=1),
        torch.cat([u_e_pre, u_e_right], dim=1),
        u_w_full,
    )


def build_manifold_boundary(
    digest: torch.Tensor,
    rounds: int = 18,
    domain_mode: str = "int_range",
    num_bits: int = 32,
    msg_bytes_len: int = 4,
) -> ManifoldBoundary:
    """Build standard full-digest 4-round analytically clamped ManifoldBoundary."""
    cb = build_clamped_boundary(
        digest=digest,
        rounds=rounds,
        domain_mode=domain_mode,
        num_bits=num_bits,
        msg_bytes_len=msg_bytes_len,
    )
    return ManifoldBoundary.from_clamped_boundary(cb)


def build_arbitrary_mask_boundary(
    rounds: int,
    static_mask: torch.Tensor,
    target_spins: torch.Tensor,
    A_right_spins: Optional[torch.Tensor] = None,
    A_right_mask: Optional[torch.Tensor] = None,
    E_right_spins: Optional[torch.Tensor] = None,
    E_right_mask: Optional[torch.Tensor] = None,
    digest_target_bits: Optional[torch.Tensor] = None,
    digest_target_mask: Optional[torch.Tensor] = None,
    W16_clamped_words: Optional[torch.Tensor] = None,
    W16_mask_bits: Optional[torch.Tensor] = None,
) -> ManifoldBoundary:
    """Drive boundary strictly from arbitrary bit-level masks (foundation for V4E / RL)."""
    if static_mask.shape != target_spins.shape or static_mask.ndim != 4:
        raise ValueError(
            f"Expected matching (B, R, 3, 32) tensors, got {static_mask.shape} vs {target_spins.shape}"
        )
    bsz, r, ch, bits = static_mask.shape
    if r != rounds or ch != 3 or bits != 32:
        raise ValueError(f"Expected shape (B, {rounds}, 3, 32), got {tuple(static_mask.shape)}")

    device = static_mask.device
    iv = get_iv_tensor(device)

    a_full_c = torch.zeros((bsz, r + 4, 32), dtype=torch.float32, device=device)
    e_full_c = torch.zeros((bsz, r + 4, 32), dtype=torch.float32, device=device)
    a_full_m = torch.zeros((bsz, r + 4, 32), dtype=torch.float32, device=device)
    e_full_m = torch.zeros((bsz, r + 4, 32), dtype=torch.float32, device=device)

    # Left boundary fixed to standard SHA-256 IV
    iv_spins = words_to_spins(iv, dtype=torch.float32)  # (8, 32)
    a_full_c[:, 0] = iv_spins[3]
    a_full_c[:, 1] = iv_spins[2]
    a_full_c[:, 2] = iv_spins[1]
    a_full_m[:, 0:3, :] = 1.0
    e_full_c[:, 0] = iv_spins[7]
    e_full_c[:, 1] = iv_spins[6]
    e_full_c[:, 2] = iv_spins[5]
    e_full_m[:, 0:3, :] = 1.0

    # Middle R rounds strictly driven by arbitrary static_mask & target_spins
    a_full_c[:, 3 : 3 + r] = target_spins[:, :, 0, :] * static_mask[:, :, 0, :]
    a_full_m[:, 3 : 3 + r] = static_mask[:, :, 0, :]
    e_full_c[:, 3 : 3 + r] = target_spins[:, :, 1, :] * static_mask[:, :, 1, :]
    e_full_m[:, 3 : 3 + r] = static_mask[:, :, 1, :]
    w_full_c = target_spins[:, :, 2, :] * static_mask[:, :, 2, :]
    w_full_m = static_mask[:, :, 2, :]

    if A_right_spins is not None and A_right_mask is not None:
        a_full_c[:, r + 3] = A_right_spins.to(torch.float32) * A_right_mask.to(torch.float32)
        a_full_m[:, r + 3] = A_right_mask.to(torch.float32)
    if E_right_spins is not None and E_right_mask is not None:
        e_full_c[:, r + 3] = E_right_spins.to(torch.float32) * E_right_mask.to(torch.float32)
        e_full_m[:, r + 3] = E_right_mask.to(torch.float32)

    a_words = torch.zeros((bsz, r + 4), dtype=torch.int64, device=device)
    e_words = torch.zeros((bsz, r + 4), dtype=torch.int64, device=device)
    a_words[:, 0:4] = iv[torch.tensor([3, 2, 1, 0], device=device)]
    e_words[:, 0:4] = iv[torch.tensor([7, 6, 5, 4], device=device)]

    k = min(r, 16)
    if W16_clamped_words is None:
        from src.sha256_ref import bits_to_words

        w16_c = torch.zeros((bsz, 16), dtype=torch.int64, device=device)
        w16_bits = (w_full_c[:, :k, :] < 0).to(torch.int64) * (w_full_m[:, :k, :] > 0.5).to(torch.int64)
        w16_c[:, :k] = bits_to_words(w16_bits)
    else:
        w16_c = W16_clamped_words.to(device=device, dtype=torch.int64)

    if W16_mask_bits is None:
        w16_m = torch.zeros((bsz, 16, 32), dtype=torch.float32, device=device)
        w16_m[:, :k, :] = w_full_m[:, :k, :]
    else:
        w16_m = W16_mask_bits.to(device=device, dtype=torch.float32)

    full_right = bool(
        (a_full_m[:, r + 3] > 0.5).all().item() and (e_full_m[:, r + 3] > 0.5).all().item()
    )
    return ManifoldBoundary(
        rounds=r,
        A_full_clamped=a_full_c,
        A_full_mask=a_full_m,
        E_full_clamped=e_full_c,
        E_full_mask=e_full_m,
        W_full_clamped=w_full_c,
        W_full_mask=w_full_m,
        A_words_clamped=a_words,
        E_words_clamped=e_words,
        W16_clamped_words=w16_c,
        W16_mask_bits=w16_m,
        digest_target_bits=digest_target_bits,
        digest_target_mask=digest_target_mask,
        iv_words=iv,
        is_full_digest_clamped=full_right,
    )


def build_v4e_partial_prefix_boundary(
    rounds: int = 18,
    zero_prefix_bits: int = 16,
    target_words: Sequence[int] = (0,),
    free_w_indices: Sequence[int] = (0, 1, 2, 3),
    batch_size: int = 1,
    device: torch.device | str = "cpu",
) -> ManifoldBoundary:
    """Zero-modification V4E partial-preimage boundary with multiple free W_i words and partial digest mask.

    Matches Module 4 (PartialPreimagePoW.lean): defaults to target_words=(0,) (H_0) because
    H_0 / H_4 are strictly sensitive to the final round W_{R-1}, whereas H_7 is decoupled for R=18.
    Zero prefix bits are constrained at MSB positions [31, 30, ..., 32-z] of each selected word.
    """
    dev = torch.device(device)
    free_set = sorted(int(i) for i in free_w_indices)
    if not free_set or free_set[0] < 0 or free_set[-1] >= 15:
        raise ValueError(f"free_w_indices must be in [0, 14] leaving W_15 for bit length, got {free_set}")
    msg_words_cnt = free_set[-1] + 1

    # Construct valid SHA-256 padded W_{0..15} template
    w16_tmpl = torch.zeros(16, dtype=torch.int64, device=dev)
    w16_mask_tmpl = torch.ones((16, 32), dtype=torch.float32, device=dev)
    w16_tmpl[msg_words_cnt] = 0x80000000
    w16_tmpl[15] = msg_words_cnt * 32
    for idx in free_set:
        w16_mask_tmpl[idx, :] = 0.0

    w16_words = w16_tmpl.unsqueeze(0).expand(batch_size, 16).clone()
    w16_mask_bits = w16_mask_tmpl.unsqueeze(0).expand(batch_size, 16, 32).clone()

    static_mask = torch.zeros((batch_size, rounds, 3, 32), dtype=torch.float32, device=dev)
    target_spins = torch.zeros((batch_size, rounds, 3, 32), dtype=torch.float32, device=dev)
    k = min(rounds, 16)
    w16_spins = words_to_spins(w16_words[:, :k], dtype=torch.float32)
    static_mask[:, :k, 2, :] = w16_mask_bits[:, :k, :]
    target_spins[:, :k, 2, :] = w16_spins * w16_mask_bits[:, :k, :]

    # Lock IV words A_0 = IV[0], E_0 = IV[4] at round index 0? Wait: A_0/E_0 are outputs of round 0!
    # Left boundary (IV) sits at indices 0..3 -> A_{-3..0}, E_{-3..0} where A_0, E_0 ARE IV[0], IV[4] before round 0!
    # Note: In ClampedBoundary, A_words[:, 3] = IV[0] IS inside node slot t=0! Let's check: yes, clamp_mask[:,0,0]=1 in full mode!
    iv = get_iv_tensor(dev)
    iv_spins = words_to_spins(iv, dtype=torch.float32)
    static_mask[:, 0, 0, :] = 1.0
    target_spins[:, 0, 0, :] = iv_spins[0].unsqueeze(0)
    static_mask[:, 0, 1, :] = 1.0
    target_spins[:, 0, 1, :] = iv_spins[4].unsqueeze(0)

    digest_bits = torch.zeros((batch_size, 8, 32), dtype=torch.float32, device=dev)
    digest_mask = torch.zeros((batch_size, 8, 32), dtype=torch.float32, device=dev)
    z = int(zero_prefix_bits)
    if z > 0:
        for w_idx in target_words:
            if not (0 <= w_idx < 8):
                raise ValueError(f"target word index must be in [0, 7], got {w_idx}")
            z_word = min(32, z)
            # Big-endian leading zeros sit at MSB bit indices [32 - z_word .. 31]
            digest_mask[:, w_idx, 32 - z_word : 32] = 1.0
            digest_bits[:, w_idx, 32 - z_word : 32] = 0.0

    mb = build_arbitrary_mask_boundary(
        rounds=rounds,
        static_mask=static_mask,
        target_spins=target_spins,
        digest_target_bits=digest_bits,
        digest_target_mask=digest_mask,
        W16_clamped_words=w16_words,
        W16_mask_bits=w16_mask_bits,
    )
    mb.is_full_digest_clamped = False
    return mb
