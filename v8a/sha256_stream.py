"""Online streaming reduced-round SHA-256 engine in vectorized PyTorch."""

from __future__ import annotations

import math
import struct
from typing import Tuple

import torch

from v8a import configure_strict_fp32

MASK32 = 0xFFFFFFFF


def _first_n_primes(n: int) -> list[int]:
    primes: list[int] = []
    cand = 2
    while len(primes) < n:
        limit = math.isqrt(cand)
        is_p = True
        for p in primes:
            if p > limit:
                break
            if cand % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(cand)
        cand += 1
    return primes


def _integer_cbrt(n: int) -> int:
    lo, hi = 0, 1 << ((n.bit_length() + 2) // 3 + 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid * mid * mid <= n:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _derive_sha256_constants() -> Tuple[tuple[int, ...], tuple[int, ...]]:
    primes = _first_n_primes(64)
    iv = tuple((math.isqrt(p << 128) >> 32) & MASK32 for p in primes[:8])
    k = tuple((_integer_cbrt(p << 192) >> 32) & MASK32 for p in primes)
    return iv, k


IV_TUPLE, K_TUPLE = _derive_sha256_constants()
_CONST_CACHE: dict[torch.device, Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_constants(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if device not in _CONST_CACHE:
        iv = torch.tensor(IV_TUPLE, dtype=torch.int64, device=device)
        k = torch.tensor(K_TUPLE, dtype=torch.int64, device=device)
        _CONST_CACHE[device] = (iv, k)
    return _CONST_CACHE[device]


def rotr32(x: torch.Tensor, n: int) -> torch.Tensor:
    """Unsigned 32-bit right rotation on int64 tensors masked to uint32."""
    xm = x & MASK32
    return ((xm >> n) | (xm << (32 - n))) & MASK32


def shr32(x: torch.Tensor, n: int) -> torch.Tensor:
    """Unsigned 32-bit logical right shift on int64 tensors."""
    return (x & MASK32) >> n


def expand_message_schedule(w16: torch.Tensor, rounds: int) -> torch.Tensor:
    """Expand [B, 16] uint32 message words to [B, max(16, rounds)] schedule."""
    if w16.ndim != 2 or w16.shape[1] != 16:
        raise ValueError(f"Expected w16 shape [B, 16], got {tuple(w16.shape)}")
    total_words = max(16, int(rounds))
    w_cols = [(w16[:, idx] & MASK32) for idx in range(16)]
    for t in range(16, total_words):
        w15 = w_cols[t - 15]
        w2 = w_cols[t - 2]
        s0 = rotr32(w15, 7) ^ rotr32(w15, 18) ^ shr32(w15, 3)
        s1 = rotr32(w2, 17) ^ rotr32(w2, 19) ^ shr32(w2, 10)
        w_t = (w_cols[t - 16] + s0 + w_cols[t - 7] + s1) & MASK32
        w_cols.append(w_t)
    return torch.stack(w_cols, dim=1)


def build_padded_block_from_words(free_words: torch.Tensor, k_bits: int) -> torch.Tensor:
    """Build single-block FIPS 180-4 padded [B, 16] uint32 words from free message words."""
    if k_bits not in (32, 64):
        raise ValueError(f"k_bits must be 32 or 64, got {k_bits}")
    fw = free_words.to(torch.int64) & MASK32
    if fw.ndim == 1:
        if k_bits != 32:
            raise ValueError("1D free_words is only valid for k_bits=32")
        fw = fw.unsqueeze(1)
    batch_size, ncols = fw.shape
    expected_cols = k_bits // 32
    if ncols != expected_cols:
        raise ValueError(f"Expected {expected_cols} words for k_bits={k_bits}, got {ncols}")

    block = torch.zeros((batch_size, 16), dtype=torch.int64, device=fw.device)
    block[:, :expected_cols] = fw
    block[:, expected_cols] = 0x80000000
    block[:, 15] = k_bits
    return block


def build_padded_block_from_bytes(data: bytes, device: torch.device | str = "cpu") -> torch.Tensor:
    """Build single-block [1, 16] uint32 tensor from raw bytes (len <= 55)."""
    if len(data) > 55:
        raise ValueError("Single-block helper supports at most 55 bytes")
    bit_len = len(data) * 8
    padded = bytearray(data)
    padded.append(0x80)
    while len(padded) < 56:
        padded.append(0)
    padded.extend(struct.pack(">Q", bit_len))
    words = struct.unpack(">16I", bytes(padded))
    return torch.tensor(words, dtype=torch.int64, device=device).unsqueeze(0)


def sha256_compress_words(
    w16: torch.Tensor,
    rounds: int = 64,
    return_states: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Run vectorized reduced-round SHA-256 compression on [B, 16] words."""
    if not (1 <= rounds <= 64):
        raise ValueError(f"rounds must be in [1, 64], got {rounds}")
    device = w16.device
    iv, k_vec = _get_constants(device)
    sched = expand_message_schedule(w16, rounds)

    a = iv[0].expand(w16.shape[0]).clone()
    b = iv[1].expand(w16.shape[0]).clone()
    c = iv[2].expand(w16.shape[0]).clone()
    d = iv[3].expand(w16.shape[0]).clone()
    e = iv[4].expand(w16.shape[0]).clone()
    f = iv[5].expand(w16.shape[0]).clone()
    g = iv[6].expand(w16.shape[0]).clone()
    h = iv[7].expand(w16.shape[0]).clone()

    states_list: list[torch.Tensor] = []
    for t in range(rounds):
        s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25)
        ch = (e & f) ^ ((~e & MASK32) & g)
        temp1 = (h + s1 + ch + k_vec[t] + sched[:, t]) & MASK32
        s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        temp2 = (s0 + maj) & MASK32

        h = g
        g = f
        f = e
        e = (d + temp1) & MASK32
        d = c
        c = b
        b = a
        a = (temp1 + temp2) & MASK32
        if return_states:
            states_list.append(torch.stack([a, b, c, d, e, f, g, h], dim=1))

    final_state = torch.stack([a, b, c, d, e, f, g, h], dim=1)
    digest_words = (final_state + iv.unsqueeze(0)) & MASK32
    if return_states:
        return digest_words, torch.stack(states_list, dim=0)
    return digest_words


def words_to_bits(words: torch.Tensor) -> torch.Tensor:
    """Convert [..., W] uint32 words to MSB-first float32 bits [..., 32*W]."""
    shifts = torch.arange(31, -1, -1, device=words.device, dtype=torch.int64)
    bits = ((words.unsqueeze(-1) >> shifts) & 1).to(torch.float32)
    return bits.reshape(*words.shape[:-1], words.shape[-1] * 32)


def bits_to_words(bits: torch.Tensor) -> torch.Tensor:
    """Convert MSB-first binary bits [..., 32*W] back to [..., W] int64 uint32 words."""
    if bits.shape[-1] % 32 != 0:
        raise ValueError(f"Bit dimension must be divisible by 32, got {bits.shape[-1]}")
    num_words = bits.shape[-1] // 32
    weights = (1 << torch.arange(31, -1, -1, device=bits.device, dtype=torch.int64))
    bit_int = (bits > 0.5).to(torch.int64).reshape(*bits.shape[:-1], num_words, 32)
    return (bit_int * weights).sum(dim=-1) & MASK32


def words_to_bytes(words: torch.Tensor) -> list[bytes]:
    """Convert [B, W] uint32 word tensor to big-endian byte strings."""
    w_cpu = (words.detach().cpu() & MASK32).tolist()
    fmt = ">" + ("I" * words.shape[1])
    return [struct.pack(fmt, *row) for row in w_cpu]


def sha256_digest_bits(
    free_words: torch.Tensor,
    k_bits: int,
    rounds: int,
    return_states: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Compute [B, 256] float32 digest bits (and optional [R, B, 256] states) from free words."""
    w16 = build_padded_block_from_words(free_words, k_bits=k_bits)
    if return_states:
        digest_words, states_words = sha256_compress_words(w16, rounds=rounds, return_states=True)
        return words_to_bits(digest_words), words_to_bits(states_words)
    digest_words = sha256_compress_words(w16, rounds=rounds, return_states=False)
    return words_to_bits(digest_words)


def compute_round_t1_bits(states_bits: torch.Tensor) -> torch.Tensor:
    """Compute exact SHA-256 half-round register T1 = a_t - (Sigma0(b_t) + Maj(b_t, c_t, d_t)) mod 2^32."""
    lead_shape = states_bits.shape[:-1]
    w = bits_to_words(states_bits.reshape(-1, 8, 32))
    a, b, c, d = w[:, 0], w[:, 1], w[:, 2], w[:, 3]
    s0 = rotr32(b, 2) ^ rotr32(b, 13) ^ rotr32(b, 22)
    maj = (b & c) ^ (b & d) ^ (c & d)
    t1 = (a - ((s0 + maj) & MASK32)) & MASK32
    return words_to_bits(t1.unsqueeze(1)).reshape(*lead_shape, 32)


def _rotr_py(x: int, n: int) -> int:
    x &= MASK32
    return ((x >> n) | (x << (32 - n))) & MASK32


def sha256_reduced_reference_bytes(msg: bytes, rounds: int = 64) -> bytes:
    """Independent pure-Python scalar reduced-round SHA-256 reference."""
    if not (1 <= rounds <= 64):
        raise ValueError(f"rounds must be in [1, 64], got {rounds}")
    bit_len = len(msg) * 8
    padded = bytearray(msg)
    padded.append(0x80)
    while (len(padded) % 64) != 56:
        padded.append(0)
    padded.extend(struct.pack(">Q", bit_len))

    h_acc = list(IV_TUPLE)
    for offset in range(0, len(padded), 64):
        w = list(struct.unpack(">16I", bytes(padded[offset : offset + 64])))
        for t in range(16, max(16, rounds)):
            s0 = _rotr_py(w[t - 15], 7) ^ _rotr_py(w[t - 15], 18) ^ ((w[t - 15] & MASK32) >> 3)
            s1 = _rotr_py(w[t - 2], 17) ^ _rotr_py(w[t - 2], 19) ^ ((w[t - 2] & MASK32) >> 10)
            w.append((w[t - 16] + s0 + w[t - 7] + s1) & MASK32)

        a, b, c, d, e, f, g, h = h_acc
        for t in range(rounds):
            s1 = _rotr_py(e, 6) ^ _rotr_py(e, 11) ^ _rotr_py(e, 25)
            ch = (e & f) ^ ((~e & MASK32) & g)
            temp1 = (h + s1 + ch + K_TUPLE[t] + w[t]) & MASK32
            s0 = _rotr_py(a, 2) ^ _rotr_py(a, 13) ^ _rotr_py(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (s0 + maj) & MASK32
            h = g
            g = f
            f = e
            e = (d + temp1) & MASK32
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & MASK32

        h_acc = [(h_acc[i] + val) & MASK32 for i, val in enumerate((a, b, c, d, e, f, g, h))]

    return struct.pack(">8I", *h_acc)


class OnlineSHA256Stream:
    """Infinite online GPU SHA-256 generator with strictly disjoint holdout eval set."""

    def __init__(
        self,
        k_bits: int = 32,
        rounds: int = 64,
        n_eval: int = 4096,
        eval_seed: int = 20260801,
        train_seed: int = 20260802,
        device: str | torch.device = "cuda",
    ) -> None:
        configure_strict_fp32()
        if k_bits not in (32, 64):
            raise ValueError(f"k_bits must be 32 or 64, got {k_bits}")
        if not (1 <= rounds <= 64):
            raise ValueError(f"rounds must be in [1, 64], got {rounds}")

        self.k_bits = int(k_bits)
        self.k_words = self.k_bits // 32
        self.rounds = int(rounds)
        self.n_eval = int(n_eval)
        self.device = torch.device(device)

        self.eval_gen = torch.Generator(device=self.device).manual_seed(int(eval_seed))
        self.train_gen = torch.Generator(device=self.device).manual_seed(int(train_seed))

        eval_words = self._sample_unique_words(self.n_eval, self.eval_gen)
        self.eval_free_words = eval_words
        self.eval_keys_sorted, _ = torch.sort(self._pack_keys(eval_words))
        self.eval_W_bits = words_to_bits(eval_words)
        self.eval_H_bits = sha256_digest_bits(eval_words, k_bits=self.k_bits, rounds=self.rounds)

    def _pack_keys(self, words: torch.Tensor) -> torch.Tensor:
        if self.k_words == 1:
            return words[:, 0] & MASK32
        return ((words[:, 0] & MASK32) << 32) | (words[:, 1] & MASK32)

    def _sample_unique_words(self, n: int, generator: torch.Generator) -> torch.Tensor:
        words = torch.randint(
            0, 1 << 32, (n, self.k_words), device=self.device, generator=generator, dtype=torch.int64
        )
        while True:
            keys = self._pack_keys(words)
            sorted_keys, order = torch.sort(keys)
            dup_mask_sorted = torch.zeros_like(sorted_keys, dtype=torch.bool)
            if sorted_keys.numel() > 1:
                dup_mask_sorted[1:] = sorted_keys[1:] == sorted_keys[:-1]
            if not bool(dup_mask_sorted.any()):
                return words
            dup_indices = order[dup_mask_sorted]
            words[dup_indices] = torch.randint(
                0,
                1 << 32,
                (dup_indices.numel(), self.k_words),
                device=self.device,
                generator=generator,
                dtype=torch.int64,
            )

    def _in_eval_set(self, keys: torch.Tensor) -> torch.Tensor:
        if self.n_eval == 0:
            return torch.zeros_like(keys, dtype=torch.bool)
        idx = torch.searchsorted(self.eval_keys_sorted, keys)
        idx_clamped = idx.clamp(max=self.n_eval - 1)
        return (idx < self.n_eval) & (self.eval_keys_sorted[idx_clamped] == keys)

    def sample_disjoint_words(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw n random free message words strictly disjoint from eval_keys_sorted."""
        gen = generator if generator is not None else self.train_gen
        words = torch.randint(
            0, 1 << 32, (n, self.k_words), device=self.device, generator=gen, dtype=torch.int64
        )
        while True:
            keys = self._pack_keys(words)
            bad = self._in_eval_set(keys)
            if not bool(bad.any()):
                return words
            n_bad = int(bad.sum().item())
            words[bad] = torch.randint(
                0, 1 << 32, (n_bad, self.k_words), device=self.device, generator=gen, dtype=torch.int64
            )

    def sample_train_batch(
        self, batch_size: int, return_states: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Generate a fresh online batch of 256-bit SHA-256 outputs (never returns preimage W)."""
        words = self.sample_disjoint_words(batch_size, self.train_gen)
        return sha256_digest_bits(
            words, k_bits=self.k_bits, rounds=self.rounds, return_states=return_states
        )

    def sample_probe_train_set(
        self, n_samples: int, seed: int = 12345
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate disjoint probe-training samples (free_words, W_bits, H_bits)."""
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        words = self.sample_disjoint_words(n_samples, gen)
        w_bits = words_to_bits(words)
        h_bits = sha256_digest_bits(words, k_bits=self.k_bits, rounds=self.rounds)
        return words, w_bits, h_bits

    def sample_white_noise_batch(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Generate i.i.d. uniform 256-bit white noise in {0.0, 1.0}."""
        gen = generator if generator is not None else self.train_gen
        return torch.randint(
            0, 2, (batch_size, 256), device=self.device, generator=gen, dtype=torch.int64
        ).to(torch.float32)
