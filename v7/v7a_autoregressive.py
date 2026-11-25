"""V7A: Reverse-Trace Autoregressive Causal Transformer Preimage Inverter.

Generates reverse intermediate states `(A_r, E_r)` for `r = R..1` followed by message
words `W_{R-1}..W_0`, strictly ordered LSB->MSB (4-bit nibbles) within every 32-bit word
to preserve 2-adic carry causality. Trained with exact Joint NLL only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from v7.v7_arena import (
    DIGEST_PREFIX_LEN,
    PAD_TOKEN,
    VOCAB_SIZE,
    build_prompt_tokens,
    decode_w16_from_reverse_trace,
    encode_reverse_trace_targets,
    sample_online_sha_batch,
)
from v7.v7_blocks import NibbleBitwiseEmbeddingFP32, RMSNormFP32, TransformerBlockFP32


@dataclass
class V7AConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1408
    max_seq_len: int = 384
    dropout: float = 0.0


def build_trace_role_metadata(
    prompt_len: int,
    n_target_tokens: int,
    rounds: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute purely positional metadata `(nib_pos, seg_role)` up to length `prompt_len + n_target_tokens`.

    When predicting the NEXT token at position `pos`, the decoder benefits from knowing the
    structural role (digest vs state vs W_word) and LSB nibble index (0..7) of that target slot.
    Strictly zero ground-truth value information is exposed.
    """
    total = prompt_len + n_target_tokens
    nib_pos = torch.full((total,), 8, dtype=torch.int64, device=device)
    seg_role = torch.full((total,), 1, dtype=torch.int64, device=device)

    # Prompt prefix H (0..63): digest nibbles
    h_len = min(DIGEST_PREFIX_LEN, prompt_len)
    nib_pos[:h_len] = torch.arange(h_len, device=device) % 8
    seg_role[:h_len] = 0

    # Target slots aligned at input index prompt_len - 1 + i  (predicting target[i])
    state_tokens = 16 * rounds
    for i in range(n_target_tokens):
        in_idx = prompt_len - 1 + i
        if 0 <= in_idx < total:
            # We store target slot i's nibble/segment in next_role/next_nib via shifted indices below
            pass
    # Build per-target-step arrays separately for explicit shift alignment in model.forward
    t_idx = torch.arange(n_target_tokens, device=device)
    tgt_nib = t_idx % 8
    tgt_seg = torch.where(t_idx < state_tokens, 2, 3)
    return nib_pos[:prompt_len], seg_role[:prompt_len], tgt_nib, tgt_seg


class V7AReverseTraceTransformer(nn.Module):
    """Pure-FP32 Causal RoPE + SwiGLU Autoregressive Reverse-Trace Inverter (~25.7M prod)."""

    def __init__(self, cfg: Optional[V7AConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or V7AConfig()
        c = self.cfg

        self.tok_embed = NibbleBitwiseEmbeddingFP32(c.vocab_size, c.d_model)
        # Current input token's structural metadata
        self.cur_nib_emb = nn.Embedding(9, c.d_model, dtype=torch.float32)
        self.cur_seg_emb = nn.Embedding(4, c.d_model, dtype=torch.float32)
        # Next-to-predict slot metadata (gives exact carry/nibble target cue before emitting logit)
        self.next_nib_emb = nn.Embedding(9, c.d_model, dtype=torch.float32)
        self.next_seg_emb = nn.Embedding(4, c.d_model, dtype=torch.float32)

        self.layers = nn.ModuleList(
            [
                TransformerBlockFP32(
                    d_model=c.d_model,
                    n_heads=c.n_heads,
                    d_ff=c.d_ff,
                    is_causal=True,
                    max_seq_len=c.max_seq_len,
                    dropout=c.dropout,
                )
                for _ in range(c.n_layers)
            ]
        )
        self.final_norm = RMSNormFP32(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False, dtype=torch.float32)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(mod: nn.Module) -> None:
        if isinstance(mod, (nn.Linear, nn.Embedding)):
            nn.init.normal_(mod.weight, mean=0.0, std=0.02)

    def _build_full_seq_metadata(
        self, prompt_len: int, tgt_len: int, rounds: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build causal current & target-slot structural indices of length `prompt_len + tgt_len`."""
        total = prompt_len + tgt_len
        cur_nib = torch.full((total,), 8, dtype=torch.int64, device=device)
        cur_seg = torch.full((total,), 1, dtype=torch.int64, device=device)
        next_nib = torch.full((total,), 8, dtype=torch.int64, device=device)
        next_seg = torch.full((total,), 1, dtype=torch.int64, device=device)

        h_len = min(DIGEST_PREFIX_LEN, prompt_len)
        cur_nib[:h_len] = torch.arange(h_len, device=device) % 8
        cur_seg[:h_len] = 0

        if tgt_len > 0:
            t_idx = torch.arange(tgt_len, device=device)
            tgt_nib = t_idx % 8
            tgt_seg = torch.where(t_idx < 16 * rounds, 2, 3)

            # Next-slot cue sits at positions prompt_len - 1 + i for i = 0..tgt_len-1
            max_next = min(tgt_len, total - (prompt_len - 1))
            if max_next > 0:
                next_nib[prompt_len - 1 : prompt_len - 1 + max_next] = tgt_nib[:max_next]
                next_seg[prompt_len - 1 : prompt_len - 1 + max_next] = tgt_seg[:max_next]

            # Current shifted target token metadata sits at positions prompt_len .. total-1
            cur_nib[prompt_len:total] = tgt_nib[:tgt_len]
            cur_seg[prompt_len:total] = tgt_seg[:tgt_len]

        return cur_nib, cur_seg, next_nib, next_seg

    def forward_logits(
        self,
        token_ids: torch.Tensor,
        rounds: int,
        prompt_len: int = 66,
    ) -> torch.Tensor:
        """Full-sequence causal forward pass returning `(B, L, vocab_size)` logits."""
        b, seq_len = token_ids.shape
        tgt_len = max(0, seq_len - prompt_len)
        cur_nib, cur_seg, next_nib, next_seg = self._build_full_seq_metadata(
            prompt_len=prompt_len, tgt_len=tgt_len, rounds=rounds, device=token_ids.device
        )
        x = (
            self.tok_embed(token_ids)
            + self.cur_nib_emb(cur_nib).unsqueeze(0)
            + self.cur_seg_emb(cur_seg).unsqueeze(0)
            + 0.5 * self.next_nib_emb(next_nib).unsqueeze(0)
            + 0.5 * self.next_seg_emb(next_seg).unsqueeze(0)
        )
        for layer in self.layers:
            x, _ = layer(x, kv_cache=None, rope_offset=0)
        x = self.final_norm(x)
        return self.lm_head(x)

    def compute_joint_nll_loss(
        self, digest: torch.Tensor, trace: Any, w16: torch.Tensor, rounds: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute exact Joint NLL (trajectory cross-entropy over `24 * R` LSB->MSB nibbles)."""
        prompt = build_prompt_tokens(digest, rounds=rounds)  # (B, 66)
        targets = encode_reverse_trace_targets(trace, w16, rounds=rounds)  # (B, 24*R)
        prompt_len = prompt.shape[1]
        tgt_len = targets.shape[1]

        # Autoregressive shifted input: [prompt(66), targets[:, :-1]] predicts targets[:, :]
        in_tokens = torch.cat([prompt, targets[:, :-1]], dim=1)
        logits = self.forward_logits(in_tokens, rounds=rounds, prompt_len=prompt_len)
        pred_logits = logits[:, prompt_len - 1 : prompt_len - 1 + tgt_len, :]

        # Joint NLL per sequence = sum over all 24*R nibbles, reported per token for cross-round comparability
        loss_per_tok = F.cross_entropy(
            pred_logits.reshape(-1, self.cfg.vocab_size),
            targets.reshape(-1),
            ignore_index=PAD_TOKEN,
            reduction="mean",
        )
        nll_seq = loss_per_tok.detach().item() * float(tgt_len)

        with torch.no_grad():
            preds = pred_logits[..., :16].argmax(dim=-1)
            tok_acc = float((preds == targets).to(torch.float32).mean().item())
            w_acc = float(
                (preds[:, 16 * rounds :] == targets[:, 16 * rounds :]).to(torch.float32).mean().item()
            )
        return loss_per_tok, {
            "tok_acc": tok_acc,
            "w_nibble_acc": w_acc,
            "joint_nll_total": nll_seq,
            "nll_per_tok": float(loss_per_tok.detach().item()),
        }

    def _step_kv(
        self,
        step_tok: torch.Tensor,
        cur_nib_id: int,
        cur_seg_id: int,
        next_nib_id: int,
        next_seg_id: int,
        kv_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
        rope_offset: int,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Single-token KV-cached decode step `(B,)` -> `(B, vocab_size)`."""
        b = step_tok.shape[0]
        dev = step_tok.device
        x = (
            self.tok_embed(step_tok).unsqueeze(1)
            + self.cur_nib_emb.weight[cur_nib_id].view(1, 1, -1)
            + self.cur_seg_emb.weight[cur_seg_id].view(1, 1, -1)
            + 0.5 * self.next_nib_emb.weight[next_nib_id].view(1, 1, -1)
            + 0.5 * self.next_seg_emb.weight[next_seg_id].view(1, 1, -1)
        )
        new_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for idx, layer in enumerate(self.layers):
            x, c = layer(x, kv_cache=kv_caches[idx], rope_offset=rope_offset)
            new_caches.append(c)
        logits = self.lm_head(self.final_norm(x)).squeeze(1)
        return logits, new_caches

    @staticmethod
    def _sample_top_p_nibble(
        logits_16: torch.Tensor, temperature: float, top_p: float
    ) -> torch.Tensor:
        """Sample nibble id in `0..15` with temperature and nucleus (top-p) filtering."""
        if temperature <= 0.0:
            return logits_16.argmax(dim=-1)
        scaled = logits_16 / float(temperature)
        probs = F.softmax(scaled, dim=-1)
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum_probs = sorted_probs.cumsum(dim=-1)
            mask = (cum_probs - sorted_probs) > float(top_p)
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            choice_in_sorted = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
            return sorted_idx.gather(-1, choice_in_sorted.unsqueeze(-1)).squeeze(-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def sample_preimages(
        self,
        digest: torch.Tensor,
        rounds: int,
        num_samples: int = 64,
        temperature: float = 0.75,
        top_p: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pure neural KV-cached sampling conditioned ONLY on `digest` `(B, 8)`.

        Candidate 0 per digest is strictly greedy (`pass@1`), candidates `1..K-1` are
        nucleus-sampled (`pass@64`). Zero classical post-processing or peel-back.
        """
        was_training = self.training
        self.eval()
        try:
            prompt = build_prompt_tokens(digest, rounds=rounds)  # (B, 66)
            b, prompt_len = prompt.shape
            tgt_len = 24 * rounds
            state_len = 16 * rounds
            dev = digest.device

            # 1. Prefill prefix tokens 0..64 first (65 tokens), keeping step 65 (BOS) to emit step 0
            cur_nib_p = torch.full((prompt_len - 1,), 8, dtype=torch.int64, device=dev)
            cur_seg_p = torch.full((prompt_len - 1,), 1, dtype=torch.int64, device=dev)
            h_len = min(DIGEST_PREFIX_LEN, prompt_len - 1)
            cur_nib_p[:h_len] = torch.arange(h_len, device=dev) % 8
            cur_seg_p[:h_len] = 0

            x_prefill = (
                self.tok_embed(prompt[:, :-1])
                + self.cur_nib_emb(cur_nib_p).unsqueeze(0)
                + self.cur_seg_emb(cur_seg_p).unsqueeze(0)
                + 0.5 * self.next_nib_emb.weight[8].view(1, 1, -1)
                + 0.5 * self.next_seg_emb.weight[1].view(1, 1, -1)
            )
            kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for layer in self.layers:
                x_prefill, c = layer(x_prefill, kv_cache=None, rope_offset=0)
                kv_caches.append(c)

            # 2. Expand KV cache to B * K before stepping BOS -> target[0..tgt_len-1]
            k_samples = int(num_samples)
            if k_samples > 1:
                kv_caches = [
                    (
                        ck.repeat_interleave(k_samples, dim=0),
                        cv.repeat_interleave(k_samples, dim=0),
                    )
                    for ck, cv in kv_caches
                ]
            greedy_mask = torch.zeros((b, k_samples), dtype=torch.bool, device=dev)
            greedy_mask[:, 0] = True
            greedy_flat = greedy_mask.reshape(-1)

            cur_tok = prompt[:, -1].repeat_interleave(k_samples)  # BOS token
            cur_nib_id, cur_seg_id = 8, 1
            rope_pos = prompt_len - 1

            generated = torch.empty((b * k_samples, tgt_len), dtype=torch.int64, device=dev)
            chosen_logprobs = torch.zeros((b * k_samples,), dtype=torch.float32, device=dev)

            for i in range(tgt_len):
                next_nib_id = i % 8
                next_seg_id = 2 if i < state_len else 3
                logits, kv_caches = self._step_kv(
                    step_tok=cur_tok,
                    cur_nib_id=cur_nib_id,
                    cur_seg_id=cur_seg_id,
                    next_nib_id=next_nib_id,
                    next_seg_id=next_seg_id,
                    kv_caches=kv_caches,
                    rope_offset=rope_pos,
                )
                logits_16 = logits[:, :16]
                log_probs_16 = F.log_softmax(logits_16, dim=-1)

                greedy_tok = logits_16.argmax(dim=-1)
                if k_samples > 1 and temperature > 0.0:
                    samp_tok = self._sample_top_p_nibble(
                        logits_16, temperature=temperature, top_p=top_p
                    )
                    next_tok = torch.where(greedy_flat, greedy_tok, samp_tok)
                else:
                    next_tok = greedy_tok

                generated[:, i] = next_tok
                chosen_logprobs += log_probs_16.gather(1, next_tok.unsqueeze(1)).squeeze(1)

                cur_tok = next_tok
                cur_nib_id = next_nib_id
                cur_seg_id = next_seg_id
                rope_pos += 1

            gen_shaped = generated.view(b, k_samples, tgt_len)
            w16 = decode_w16_from_reverse_trace(gen_shaped, rounds=rounds)
            nll_per_tok = (-chosen_logprobs.view(b, k_samples)[:, 0]) / float(tgt_len)
            return w16, nll_per_tok
        finally:
            if was_training:
                self.train()


def train_v7a_step(
    model: V7AReverseTraceTransformer,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    rounds: int,
    device: torch.device | str = "cuda",
    generator: Optional[torch.Generator] = None,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Run one pure-FP32 online training step on fresh SHA-256 reverse traces."""
    model.train()
    w16, trace, digest = sample_online_sha_batch(
        batch_size=batch_size, rounds=rounds, device=device, generator=generator
    )
    optimizer.zero_grad(set_to_none=True)
    loss, stats = model.compute_joint_nll_loss(digest=digest, trace=trace, w16=w16, rounds=rounds)
    loss.backward()
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    stats["train_loss"] = float(loss.detach().item())
    return stats


__all__ = [
    "V7AConfig",
    "V7AReverseTraceTransformer",
    "train_v7a_step",
]
