"""Unified data arena, causal nibble codecs, verifier, and holdout evaluator for V7."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch

from src.sha256_ref import MASK32, SHA256Trace, sha256_digest, sha256_trace

V7_ROUNDS: Tuple[int, ...] = (2, 3, 4, 5, 6, 8)

# Token vocabulary shared by V7A and V7C (vocab_size = 32):
#   0..15 : 4-bit nibble values (LSB-first within every 32-bit word)
#   16..27: round conditioning token for R in 2..13  (tok = 16 + (R - 2))
#   28    : BOS / start-of-generation separator
#   29    : PAD
#   30..31: reserved special tokens
VOCAB_SIZE = 32
NIBBLE_MIN = 0
NIBBLE_MAX = 15
ROUND_TOKEN_BASE = 16
MAX_CODEC_ROUNDS = 13
BOS_TOKEN = 28
PAD_TOKEN = 29
DEFAULT_NIBBLES_PER_WORD = 8
DIGEST_PREFIX_LEN = 64  # 8 digest words * 8 LSB-first nibbles

_SHIFT4_CACHE: Dict[str, torch.Tensor] = {}
_SHIFT32_CACHE: Dict[str, torch.Tensor] = {}


def _get_shift4(device: torch.device) -> torch.Tensor:
    key = str(device)
    t = _SHIFT4_CACHE.get(key)
    if t is None:
        t = torch.arange(8, dtype=torch.int64, device=device) * 4
        _SHIFT4_CACHE[key] = t
    return t


def _get_shift32(device: torch.device) -> torch.Tensor:
    key = str(device)
    t = _SHIFT32_CACHE.get(key)
    if t is None:
        t = torch.arange(32, dtype=torch.int64, device=device)
        _SHIFT32_CACHE[key] = t
    return t


def round_to_token(rounds: int) -> int:
    """Map round count R (2..13) to its dedicated conditioning token id."""
    if not (2 <= int(rounds) <= MAX_CODEC_ROUNDS):
        raise ValueError(f"rounds must be in [2, {MAX_CODEC_ROUNDS}], got {rounds}")
    return ROUND_TOKEN_BASE + (int(rounds) - 2)


def pack_w_to_w16(w: torch.Tensor, rounds: int) -> torch.Tensor:
    """Convert `(..., R)` or `(..., 16)` int64 free-msg words into `(..., 16)` uint32-masked int64."""
    if w.shape[-1] == 16:
        out = (w & MASK32).to(torch.int64).clone()
        if rounds < 16:
            out[..., rounds:] = 0
        return out
    if w.shape[-1] != rounds:
        raise ValueError(f"Expected last dim {rounds} or 16, got shape {tuple(w.shape)}")
    out = torch.zeros((*w.shape[:-1], 16), dtype=torch.int64, device=w.device)
    out[..., :rounds] = w.to(torch.int64) & MASK32
    return out


def words_to_lsb_nibbles(words: torch.Tensor) -> torch.Tensor:
    """Encode `(...,)` int64 uint32 words into `(..., 8)` LSB-first 4-bit nibbles."""
    shifts = _get_shift4(words.device)
    return ((words.to(torch.int64) & MASK32).unsqueeze(-1) >> shifts) & 0xF


def lsb_nibbles_to_words(nibbles: torch.Tensor) -> torch.Tensor:
    """Decode `(..., 8)` LSB-first 4-bit nibbles back into `(...,)` int64 uint32 words."""
    if nibbles.shape[-1] != 8:
        raise ValueError(f"Expected last dim 8, got {tuple(nibbles.shape)}")
    shifts = _get_shift4(nibbles.device)
    return ((nibbles.to(torch.int64) & 0xF) << shifts).sum(dim=-1) & MASK32


def words_to_lsb_bits(words: torch.Tensor) -> torch.Tensor:
    """Convert `(...,)` int64 words into `(..., 32)` LSB-first `{0,1}` int64 bits."""
    shifts = _get_shift32(words.device)
    return ((words.to(torch.int64) & MASK32).unsqueeze(-1) >> shifts) & 1


def encode_digest_prefix(digest: torch.Tensor) -> torch.Tensor:
    """Encode `(B, 8)` target digests into `(B, 64)` LSB-first nibble tokens."""
    if digest.ndim != 2 or digest.shape[1] != 8:
        raise ValueError(f"Expected digest shape (B, 8), got {tuple(digest.shape)}")
    return words_to_lsb_nibbles(digest).reshape(digest.shape[0], DIGEST_PREFIX_LEN)


def build_prompt_tokens(digest: torch.Tensor, rounds: int) -> torch.Tensor:
    """Construct `(B, 66)` prompt `[H_nibbles(64), ROUND_R(1), BOS(1)]`."""
    b = digest.shape[0]
    h_tok = encode_digest_prefix(digest)
    meta = torch.tensor(
        [[round_to_token(rounds), BOS_TOKEN]],
        dtype=torch.int64,
        device=digest.device,
    ).expand(b, 2)
    return torch.cat([h_tok, meta], dim=1)


def encode_reverse_trace_targets(
    trace: SHA256Trace, w16: torch.Tensor, rounds: int
) -> torch.Tensor:
    """Encode ground-truth reverse trace `A_R,E_R..A_1,E_1, W_{R-1}..W_0` in LSB->MSB nibbles.

    Target sequence length = `24 * rounds` tokens (`0..15`):
    - First `16 * rounds`: for `r = rounds` down to `1`, `[A_r (8 nibbles), E_r (8 nibbles)]`.
    - Last  ` 8 * rounds`: for `r = rounds - 1` down to `0`, `W_r (8 nibbles)`.
    """
    b = w16.shape[0]
    state_words: List[torch.Tensor] = []
    # Note: trace.A_full[:, r + 3] is A_r after round (r - 1) for r = 1..rounds.
    for r in range(rounds, 0, -1):
        state_words.append(trace.A_full[:, r + 3])
        state_words.append(trace.E_full[:, r + 3])
    msg_words = [w16[:, r] & MASK32 for r in range(rounds - 1, -1, -1)]
    all_words = torch.stack(state_words + msg_words, dim=1)  # (B, 3 * rounds)
    return words_to_lsb_nibbles(all_words).reshape(b, 24 * rounds)


def decode_w16_from_reverse_trace(
    target_tokens: torch.Tensor, rounds: int
) -> torch.Tensor:
    """Extract `W_0..W_{R-1}` from `(..., 24 * rounds)` reverse-trace tokens and pad to 16 words."""
    expected_len = 24 * rounds
    if target_tokens.shape[-1] < expected_len:
        raise ValueError(
            f"Expected at least {expected_len} trace tokens for R={rounds}, got {target_tokens.shape[-1]}"
        )
    w_tok = target_tokens[..., 16 * rounds : 24 * rounds]
    leading = w_tok.shape[:-1]
    w_nib = w_tok.reshape(*leading, rounds, 8)
    w_rev = lsb_nibbles_to_words(w_nib)  # [W_{R-1}, ..., W_0]
    w_fwd = torch.flip(w_rev, dims=[-1])  # [W_0, ..., W_{R-1}]
    return pack_w_to_w16(w_fwd, rounds=rounds)


def decode_w16_from_words_nibbles(
    w_nibbles_rev: torch.Tensor, rounds: int
) -> torch.Tensor:
    """Decode `(..., 8 * rounds)` LSB-first nibbles ordered `W_{R-1}..W_0` into `(..., 16)` `w16`."""
    expected = 8 * rounds
    if w_nibbles_rev.shape[-1] != expected:
        raise ValueError(f"Expected {expected} nibbles, got {w_nibbles_rev.shape[-1]}")
    leading = w_nibbles_rev.shape[:-1]
    w_rev = lsb_nibbles_to_words(w_nibbles_rev.reshape(*leading, rounds, 8))
    w_fwd = torch.flip(w_rev, dims=[-1])
    return pack_w_to_w16(w_fwd, rounds=rounds)


def sample_online_sha_batch(
    batch_size: int,
    rounds: int,
    device: torch.device | str = "cuda",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, SHA256Trace, torch.Tensor]:
    """Sample fresh `(w16, trace, digest)` on the fly in `free_msg` domain (`W[0..R-1]`)."""
    dev = torch.device(device)
    if generator is not None and generator.device != dev:
        w_free = torch.randint(
            0,
            0x100000000,
            (batch_size, rounds),
            dtype=torch.int64,
            generator=generator,
            device=generator.device,
        ).to(dev)
    else:
        w_free = torch.randint(
            0,
            0x100000000,
            (batch_size, rounds),
            dtype=torch.int64,
            generator=generator,
            device=dev,
        )
    w16 = pack_w_to_w16(w_free, rounds=rounds)
    trace = sha256_trace(w16, rounds=rounds)
    return w16, trace, trace.digest


@dataclass(frozen=True)
class V7HoldoutSet:
    """Fixed unseen holdout digest set per round `R`.

    `_secret_w16` is strictly quarantined for post-inference diagnostic telemetry
    (`w0_bit_acc` / `w0_zscore`) and is NEVER passed to models or RL rewards.
    """

    rounds: int
    digest: torch.Tensor
    _secret_w16: torch.Tensor

    @property
    def n_eval(self) -> int:
        return int(self.digest.shape[0])


def build_v7_holdout(
    rounds: int,
    n_eval: int = 256,
    seed_base: int = 770000,
    device: torch.device | str = "cuda",
) -> V7HoldoutSet:
    """Build deterministic unseen holdout digests for round `rounds` (`seed_base` isolated)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed_base) + int(rounds) * 1009)
    w_free = torch.randint(
        0, 0x100000000, (int(n_eval), int(rounds)), dtype=torch.int64, generator=gen
    )
    w16 = pack_w_to_w16(w_free, rounds=rounds).to(device)
    digest = sha256_digest(w16, rounds=rounds)
    return V7HoldoutSet(rounds=int(rounds), digest=digest, _secret_w16=w16)


def compute_verifier_reward(
    w16_pred: torch.Tensor,
    digest_target: torch.Tensor,
    rounds: int,
) -> Dict[str, torch.Tensor]:
    """Compute strictly forward-verifiable pure-FP32 reward components (ZERO `W*` leakage).

    Supports arbitrary leading batch dims: `w16_pred` `(..., 16)`, `digest_target` `(..., 8)`.
    Reward = `r_bit + 2.5 * r_2adic_lcp + 1.0 * r_global_lcp + 10.0 * exact_match` in `[0, 14.5]`.
    """
    lead_shape = w16_pred.shape[:-1]
    flat_w16 = w16_pred.reshape(-1, 16)
    pred_digest = sha256_digest(flat_w16, rounds=rounds).reshape(*lead_shape, 8)
    diff = pred_digest ^ digest_target

    match_bits = (words_to_lsb_bits(diff) == 0).to(torch.float32)  # (..., 8, 32)
    r_bit = match_bits.mean(dim=(-2, -1))

    # 2-adic least-significant contiguous matching prefix per 32-bit digest word
    cum_word = torch.cumprod(match_bits, dim=-1)
    r_2adic_lcp = cum_word.sum(dim=-1).mean(dim=-1) / 32.0

    # Global contiguous LSB prefix across 256 digest bits
    flat_bits = match_bits.reshape(*lead_shape, 256)
    r_global_lcp = torch.cumprod(flat_bits, dim=-1).sum(dim=-1) / 256.0

    exact_match = (diff == 0).all(dim=-1).to(torch.float32)
    reward = r_bit + 2.5 * r_2adic_lcp + 1.0 * r_global_lcp + 10.0 * exact_match
    return {
        "reward": reward,
        "r_bit": r_bit,
        "r_2adic_lcp": r_2adic_lcp,
        "r_global_lcp": r_global_lcp,
        "exact_match": exact_match,
        "pred_digest": pred_digest,
    }


@torch.no_grad()
def evaluate_pure_neural_model(
    sample_fn: Callable[[torch.Tensor, int, int], Tuple[torch.Tensor, Optional[torch.Tensor]]],
    holdout: V7HoldoutSet,
    pass_k: int = 64,
    eval_batch_size: int = 32,
) -> Dict[str, float]:
    """Strictly evaluate any pure neural solver on `holdout.digest` via discrete `sha256_digest`.

    `sample_fn(digest_chunk, rounds, pass_k)` MUST only receive `(digest_chunk, rounds, pass_k)`
    and return `(w16_samples, optional_nll_per_sample)`:
      - `w16_samples`: `(b, pass_k, 16)` int64 tensor, where `w16_samples[:, 0]` is the greedy/top-1
        candidate used for `exact_pass@1`.
      - `optional_nll_per_sample`: optional scalar float or `(b,)` tensor (None if model has no NLL).
    """
    n_eval = holdout.n_eval
    rounds = holdout.rounds
    digest_all = holdout.digest
    w_secret_all = holdout._secret_w16

    hits_at_1 = 0
    hits_at_k = 0
    bit_acc_1_sum = 0.0
    bit_acc_bestk_sum = 0.0
    w0_match_bits = 0
    nll_values: List[float] = []

    for start in range(0, n_eval, eval_batch_size):
        end = min(n_eval, start + eval_batch_size)
        d_chunk = digest_all[start:end]
        b = int(d_chunk.shape[0])

        w16_cand, nll_opt = sample_fn(d_chunk, rounds, pass_k)
        if w16_cand.shape != (b, pass_k, 16):
            raise AssertionError(
                f"sample_fn must return shape ({b}, {pass_k}, 16), got {tuple(w16_cand.shape)}"
            )
        w16_cand = pack_w_to_w16(w16_cand, rounds=rounds)

        pred_digest = sha256_digest(w16_cand.reshape(b * pass_k, 16), rounds=rounds).reshape(
            b, pass_k, 8
        )
        diff = pred_digest ^ d_chunk.unsqueeze(1)
        exact = (diff == 0).all(dim=-1)  # (b, pass_k)
        match_bits = (words_to_lsb_bits(diff) == 0).to(torch.float32).mean(dim=(-2, -1))  # (b, pass_k)

        hits_at_1 += int(exact[:, 0].sum().item())
        hits_at_k += int(exact.any(dim=1).sum().item())
        bit_acc_1_sum += float(match_bits[:, 0].sum().item())
        bit_acc_bestk_sum += float(match_bits.max(dim=1).values.sum().item())

        # Post-hoc W_0 telemetry only (zero influence on model inputs/outputs)
        w0_pred = w16_cand[:, 0, 0]
        w0_true = w_secret_all[start:end, 0]
        w0_eq = (words_to_lsb_bits(w0_pred ^ w0_true) == 0).sum().item()
        w0_match_bits += int(w0_eq)

        if nll_opt is not None:
            if isinstance(nll_opt, torch.Tensor):
                nll_values.append(float(nll_opt.mean().item()))
            else:
                nll_values.append(float(nll_opt))

    total_w0_bits = n_eval * 32
    w0_acc = float(w0_match_bits) / float(total_w0_bits)
    w0_z = (float(w0_match_bits) - 0.5 * total_w0_bits) / math.sqrt(0.25 * total_w0_bits)

    return {
        "round_r": float(rounds),
        "eval_n_digests": float(n_eval),
        "exact_hits_1": float(hits_at_1),
        "exact_hits": float(hits_at_k),
        "exact_pass_1": float(hits_at_1) / float(n_eval),
        "exact_pass_64": float(hits_at_k) / float(n_eval),
        "hash_bit_acc": bit_acc_1_sum / float(n_eval),
        "hash_bit_acc_best64": bit_acc_bestk_sum / float(n_eval),
        "w0_bit_acc": w0_acc,
        "w0_zscore": w0_z,
        "joint_nll": (sum(nll_values) / len(nll_values)) if nll_values else 0.0,
    }


V7_CSV_FIELDS: Tuple[str, ...] = (
    "step",
    "wall_time",
    "model",
    "stage",
    "round_r",
    "r_c_neural",
    "train_loss",
    "joint_nll",
    "mean_reward",
    "lr",
    "peak_vram_gb",
    "step_time_ms",
    "exact_pass_1",
    "exact_pass_64",
    "exact_hits",
    "eval_n_digests",
    "hash_bit_acc",
    "hash_bit_acc_best64",
    "w0_bit_acc",
    "w0_zscore",
)


class V7CSVLogger:
    """Append-only structured CSV logger compatible with Colab & local dashboards."""

    def __init__(self, csv_path: Path | str) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.monotonic()
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=V7_CSV_FIELDS)
                writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        payload: Dict[str, Any] = {k: "" for k in V7_CSV_FIELDS}
        payload["wall_time"] = f"{time.monotonic() - self.t0:.2f}"
        for k, v in row.items():
            if k in payload and v is not None:
                if isinstance(v, float):
                    payload[k] = f"{v:.6g}"
                else:
                    payload[k] = v
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=V7_CSV_FIELDS)
            writer.writerow(payload)


__all__ = [
    "V7_ROUNDS",
    "VOCAB_SIZE",
    "ROUND_TOKEN_BASE",
    "BOS_TOKEN",
    "PAD_TOKEN",
    "DIGEST_PREFIX_LEN",
    "round_to_token",
    "pack_w_to_w16",
    "words_to_lsb_nibbles",
    "lsb_nibbles_to_words",
    "words_to_lsb_bits",
    "encode_digest_prefix",
    "build_prompt_tokens",
    "encode_reverse_trace_targets",
    "decode_w16_from_reverse_trace",
    "decode_w16_from_words_nibbles",
    "sample_online_sha_batch",
    "V7HoldoutSet",
    "build_v7_holdout",
    "compute_verifier_reward",
    "evaluate_pure_neural_model",
    "V7_CSV_FIELDS",
    "V7CSVLogger",
]
