"""Exact algebraic left/right boundary clamping and 4-round analytical back-solve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from src.sha256_ref import (
    MASK32,
    get_iv_tensor,
    maj32,
    sigma0_upper,
    words_to_bits,
    words_to_spins,
)


@dataclass
class ClampedBoundary:
    """Container for clamped spin templates, binary clamp masks, and exact boundary words."""

    rounds: int
    A_full_clamped: torch.Tensor  # (B, R + 4, 32) float32
    A_full_mask: torch.Tensor  # (B, R + 4, 32) float32, 1=locked, 0=free
    E_full_clamped: torch.Tensor  # (B, R + 4, 32) float32
    E_full_mask: torch.Tensor  # (B, R + 4, 32) float32
    W_full_clamped: torch.Tensor  # (B, R, 32) float32
    W_full_mask: torch.Tensor  # (B, R, 32) float32
    A_words_clamped: torch.Tensor  # (B, R + 4) int64
    E_words_clamped: torch.Tensor  # (B, R + 4) int64
    W16_clamped_words: torch.Tensor  # (B, 16) int64
    W16_mask_bits: torch.Tensor  # (B, 16, 32) float32

    @property
    def batch_size(self) -> int:
        return int(self.A_full_clamped.shape[0])

    @property
    def device(self) -> torch.device:
        return self.A_full_clamped.device

    @property
    def x_clamped(self) -> torch.Tensor:
        """Return node-aligned clamped spin template of shape (B, R, 3, 32)."""
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
        """Return node-aligned binary clamp mask of shape (B, R, 3, 32)."""
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
    def A_left(self) -> torch.Tensor:
        """Exterior left A states A_{-3}, A_{-2}, A_{-1} of shape (B, 3, 32)."""
        return self.A_full_clamped[:, 0:3]

    @property
    def A_right(self) -> torch.Tensor:
        """Exterior right A state A_R of shape (B, 1, 32)."""
        return self.A_full_clamped[:, self.rounds + 3 : self.rounds + 4]

    @property
    def E_left(self) -> torch.Tensor:
        """Exterior left E states E_{-3}, E_{-2}, E_{-1} of shape (B, 3, 32)."""
        return self.E_full_clamped[:, 0:3]

    @property
    def E_right(self) -> torch.Tensor:
        """Exterior right E state E_R of shape (B, 1, 32)."""
        return self.E_full_clamped[:, self.rounds + 3 : self.rounds + 4]

    def apply_clamp(self, x_nodes: torch.Tensor) -> torch.Tensor:
        """Hard-overwrite locked coordinates in x_nodes (B, R, 3, 32) with exact spins."""
        mask = self.clamp_mask
        return mask * self.x_clamped + (1.0 - mask) * x_nodes

    def assemble_full(
        self, x_nodes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble full (A_full, E_full, W_full) from node tensor (B, R, 3, 32)."""
        a_mid = x_nodes[:, :, 0, :]
        e_mid = x_nodes[:, :, 1, :]
        w_full = x_nodes[:, :, 2, :]

        a_full = torch.cat([self.A_left, a_mid, self.A_right], dim=1)
        e_full = torch.cat([self.E_left, e_mid, self.E_right], dim=1)
        return a_full, e_full, w_full

    def extract_candidate_w16(self, x_nodes: torch.Tensor) -> torch.Tensor:
        """Threshold node spins into a valid (B, 16) uint32-in-int64 message block."""
        from src.sha256_ref import bits_to_words

        r = self.rounds
        w_spins = x_nodes[:, : min(r, 16), 2, :]
        w_bits_pred = (w_spins < 0).to(torch.float32)

        w16_clamped_bits = words_to_bits(self.W16_clamped_words).to(torch.float32)
        w16_bits = w16_clamped_bits.clone()

        k = min(r, 16)
        mask_k = self.W16_mask_bits[:, :k, :]
        w16_bits[:, :k, :] = (
            mask_k * w16_clamped_bits[:, :k, :] + (1.0 - mask_k) * w_bits_pred
        )
        return bits_to_words(w16_bits.to(torch.int64))

    def repeat_for_candidates(self, num_candidates: int) -> "ClampedBoundary":
        """Repeat a batch-1 or batch-B boundary K times along batch dimension."""
        b = self.batch_size
        return ClampedBoundary(
            rounds=self.rounds,
            A_full_clamped=self.A_full_clamped.repeat_interleave(num_candidates, dim=0),
            A_full_mask=self.A_full_mask.repeat_interleave(num_candidates, dim=0),
            E_full_clamped=self.E_full_clamped.repeat_interleave(num_candidates, dim=0),
            E_full_mask=self.E_full_mask.repeat_interleave(num_candidates, dim=0),
            W_full_clamped=self.W_full_clamped.repeat_interleave(num_candidates, dim=0),
            W_full_mask=self.W_full_mask.repeat_interleave(num_candidates, dim=0),
            A_words_clamped=self.A_words_clamped.repeat_interleave(
                num_candidates, dim=0
            ),
            E_words_clamped=self.E_words_clamped.repeat_interleave(
                num_candidates, dim=0
            ),
            W16_clamped_words=self.W16_clamped_words.repeat_interleave(
                num_candidates, dim=0
            ),
            W16_mask_bits=self.W16_mask_bits.repeat_interleave(num_candidates, dim=0),
        )


def build_w16_domain_clamp(
    domain_mode: str = "int_range",
    num_bits: int = 32,
    msg_bytes_len: int = 4,
    device: torch.device | str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic (16,) clamped words and (16, 32) bit mask for input domain."""
    w16_words = torch.zeros(16, dtype=torch.int64, device=device)
    w16_mask_bits = torch.zeros((16, 32), dtype=torch.float32, device=device)

    if domain_mode == "int_range":
        if not (1 <= num_bits <= 32):
            raise ValueError(f"num_bits must be in [1, 32], got {num_bits}")
        # W_0: bits [0..num_bits-1] free, bits [num_bits..31] locked to 0
        if num_bits < 32:
            w16_mask_bits[0, num_bits:] = 1.0
        w16_words[1] = 0x80000000
        w16_words[15] = 32
        w16_mask_bits[1:16, :] = 1.0
        return w16_words, w16_mask_bits

    if domain_mode == "fixed_len_bytes":
        if not (0 <= msg_bytes_len <= 55):
            raise ValueError(
                f"msg_bytes_len must be in [0, 55], got {msg_bytes_len}"
            )
        byte_vals = torch.zeros(64, dtype=torch.int64, device=device)
        byte_mask = torch.zeros(64, dtype=torch.int64, device=device)

        if msg_bytes_len < 56:
            byte_vals[msg_bytes_len] = 0x80
            byte_mask[msg_bytes_len] = 0xFF
        if msg_bytes_len + 1 < 56:
            byte_vals[msg_bytes_len + 1 : 56] = 0x00
            byte_mask[msg_bytes_len + 1 : 56] = 0xFF

        bit_len = msg_bytes_len * 8
        for i in range(8):
            byte_vals[56 + i] = (bit_len >> (8 * (7 - i))) & 0xFF
            byte_mask[56 + i] = 0xFF

        bv = byte_vals.view(16, 4)
        bm = byte_mask.view(16, 4)
        w16_words = (
            (bv[:, 0] << 24) | (bv[:, 1] << 16) | (bv[:, 2] << 8) | bv[:, 3]
        ) & MASK32
        w16_mask_words = (
            (bm[:, 0] << 24) | (bm[:, 1] << 16) | (bm[:, 2] << 8) | bm[:, 3]
        ) & MASK32
        w16_mask_bits = words_to_bits(w16_mask_words).to(torch.float32)
        return w16_words, w16_mask_bits

    raise ValueError(f"Unsupported domain_mode: {domain_mode}")


def build_clamped_boundary(
    digest: torch.Tensor,
    rounds: int = 64,
    domain_mode: str = "int_range",
    num_bits: int = 32,
    msg_bytes_len: int = 4,
) -> ClampedBoundary:
    """Construct exact algebraic left/right clamped boundary with 4-round back-solve.

    Args:
        digest: Target digest words of shape (B, 8) or (8,) in uint32-in-int64.
        rounds: Number of SHA-256 rounds R in [8, 64] (primary scope [9, 64]).
        domain_mode: 'int_range' or 'fixed_len_bytes'.
        num_bits: Active bit width for 'int_range' W_0.
        msg_bytes_len: Message byte length for 'fixed_len_bytes'.
    """
    if not (8 <= rounds <= 64):
        raise ValueError(f"rounds must be in [8, 64], got {rounds}")

    if digest.ndim == 1:
        digest = digest.unsqueeze(0)
    if digest.ndim != 2 or digest.shape[1] != 8:
        raise ValueError(f"Expected digest shape (B, 8), got {tuple(digest.shape)}")

    batch_size = digest.shape[0]
    device = digest.device
    iv = get_iv_tensor(device)

    a_words = torch.zeros((batch_size, rounds + 4), dtype=torch.int64, device=device)
    e_words = torch.zeros((batch_size, rounds + 4), dtype=torch.int64, device=device)
    a_mask = torch.zeros((batch_size, rounds + 4, 32), dtype=torch.float32, device=device)
    e_mask = torch.zeros((batch_size, rounds + 4, 32), dtype=torch.float32, device=device)

    # 1. Left boundary from IV: A_{-3..0} = IV[3..0], E_{-3..0} = IV[7..4]
    a_words[:, 0] = iv[3]
    a_words[:, 1] = iv[2]
    a_words[:, 2] = iv[1]
    a_words[:, 3] = iv[0]
    e_words[:, 0] = iv[7]
    e_words[:, 1] = iv[6]
    e_words[:, 2] = iv[5]
    e_words[:, 3] = iv[4]
    a_mask[:, 0:4, :] = 1.0
    e_mask[:, 0:4, :] = 1.0

    # 2. Right boundary from digest: S_R = (H_out - IV) mod 2^32
    s_r = (digest.to(torch.int64) - iv.unsqueeze(0)) & MASK32
    a_words[:, rounds + 3] = s_r[:, 0]  # A_R
    a_words[:, rounds + 2] = s_r[:, 1]  # A_{R-1}
    a_words[:, rounds + 1] = s_r[:, 2]  # A_{R-2}
    a_words[:, rounds] = s_r[:, 3]  # A_{R-3}
    e_words[:, rounds + 3] = s_r[:, 4]  # E_R
    e_words[:, rounds + 2] = s_r[:, 5]  # E_{R-1}
    e_words[:, rounds + 1] = s_r[:, 6]  # E_{R-2}
    e_words[:, rounds] = s_r[:, 7]  # E_{R-3}
    a_mask[:, rounds : rounds + 4, :] = 1.0
    e_mask[:, rounds : rounds + 4, :] = 1.0

    # 3. Analytical 4-round back-solve for A_{R-4}, A_{R-5}, A_{R-6}, A_{R-7}
    # A_{t-3} = (E_{t+1} - A_{t+1} + Sigma0(A_t) + Maj(A_t, A_{t-1}, A_{t-2})) mod 2^32
    for t in (rounds - 1, rounds - 2, rounds - 3, rounds - 4):
        a_tm3 = (
            e_words[:, t + 4]
            - a_words[:, t + 4]
            + sigma0_upper(a_words[:, t + 3])
            + maj32(a_words[:, t + 3], a_words[:, t + 2], a_words[:, t + 1])
        ) & MASK32
        a_words[:, t] = a_tm3
        a_mask[:, t, :] = 1.0

    a_spins = words_to_spins(a_words, dtype=torch.float32) * a_mask
    e_spins = words_to_spins(e_words, dtype=torch.float32) * e_mask

    # 4. Domain / padding clamping on W_{0..15}
    w16_words_1d, w16_mask_2d = build_w16_domain_clamp(
        domain_mode=domain_mode,
        num_bits=num_bits,
        msg_bytes_len=msg_bytes_len,
        device=device,
    )
    w16_words = w16_words_1d.unsqueeze(0).expand(batch_size, 16).clone()
    w16_mask_bits = w16_mask_2d.unsqueeze(0).expand(batch_size, 16, 32).clone()

    w_full_clamped = torch.zeros(
        (batch_size, rounds, 32), dtype=torch.float32, device=device
    )
    w_full_mask = torch.zeros(
        (batch_size, rounds, 32), dtype=torch.float32, device=device
    )
    k = min(rounds, 16)
    w16_spins = words_to_spins(w16_words[:, :k], dtype=torch.float32)
    w_full_clamped[:, :k, :] = w16_spins * w16_mask_bits[:, :k, :]
    w_full_mask[:, :k, :] = w16_mask_bits[:, :k, :]

    return ClampedBoundary(
        rounds=rounds,
        A_full_clamped=a_spins,
        A_full_mask=a_mask,
        E_full_clamped=e_spins,
        E_full_mask=e_mask,
        W_full_clamped=w_full_clamped,
        W_full_mask=w_full_mask,
        A_words_clamped=a_words,
        E_words_clamped=e_words,
        W16_clamped_words=w16_words,
        W16_mask_bits=w16_mask_bits,
    )
