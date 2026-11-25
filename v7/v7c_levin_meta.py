"""V7C: Phase-Levin Universal ARX Meta-Inverter (`f_univ`) + Progressive Curriculum.

Two-stage neural preimage architecture:
- Program Bytecode Encoder: bidirectional RoPE+SwiGLU encoder over 32-bit ARX VM
  bytecode `(op, dst, a, b, imm)` pooled into 32 continuous condition tokens.
- Autoregressive Decoder: causal RoPE+SwiGLU decoder conditioned on `[P_cond(32),
  Y/H_nibbles(64), ROUND/BOS(2)]` generating inputs `X` (Stage 1 ARX programs) or
  full reverse traces `A_r, E_r, W_r` (Stage 2 compiled SHA-256).
- Optional offline `V6A` verified corpus mixing at `R=4..6` during Stage 2 training;
  100% pure-neural & detached from `V6A` during evaluation.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sha256_ref import MASK32, sha256_digest, sha256_trace
from v7.univ_arx_vm import (
    NUM_OPS,
    NUM_REGS,
    OP_NOP,
    compile_sha256_to_vm,
    sample_stage1_arx_curriculum_batch,
)
from v7.v7_arena import (
    BOS_TOKEN,
    DIGEST_PREFIX_LEN,
    PAD_TOKEN,
    VOCAB_SIZE,
    build_prompt_tokens,
    decode_w16_from_reverse_trace,
    encode_digest_prefix,
    encode_reverse_trace_targets,
    pack_w_to_w16,
    round_to_token,
    sample_online_sha_batch,
    words_to_lsb_bits,
    words_to_lsb_nibbles,
)
from v7.v7_blocks import NibbleBitwiseEmbeddingFP32, RMSNormFP32, TransformerBlockFP32


@dataclass
class V7CConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 512
    n_enc_layers: int = 2
    n_dec_layers: int = 7
    n_heads: int = 8
    d_ff: int = 1344
    n_prog_cond_tokens: int = 32
    max_prog_len: int = 256
    max_seq_len: int = 384
    dropout: float = 0.0


class BytecodeProgramEncoderFP32(nn.Module):
    """Encodes `(B, L_p, 5)` VM bytecode into `(B, n_cond, d_model)` continuous descriptors."""

    def __init__(self, cfg: V7CConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg
        self.op_emb = nn.Embedding(NUM_OPS, c.d_model, dtype=torch.float32)
        self.dst_emb = nn.Embedding(NUM_REGS, c.d_model, dtype=torch.float32)
        self.rega_emb = nn.Embedding(NUM_REGS, c.d_model, dtype=torch.float32)
        self.regb_emb = nn.Embedding(NUM_REGS + 1, c.d_model, dtype=torch.float32)  # 0 = imm (-1), 1..32 = r0..r31
        self.imm_bit_proj = nn.Linear(32, c.d_model, bias=False, dtype=torch.float32)
        self.in_norm = RMSNormFP32(c.d_model)

        self.layers = nn.ModuleList(
            [
                TransformerBlockFP32(
                    d_model=c.d_model,
                    n_heads=c.n_heads,
                    d_ff=c.d_ff,
                    is_causal=False,
                    max_seq_len=c.max_prog_len,
                    dropout=c.dropout,
                )
                for _ in range(c.n_enc_layers)
            ]
        )
        self.enc_norm = RMSNormFP32(c.d_model)

        # Learned query pooler collapsing arbitrary L_p into fixed n_prog_cond_tokens
        self.cond_queries = nn.Parameter(
            torch.randn(1, c.n_prog_cond_tokens, c.d_model, dtype=torch.float32) * 0.02
        )
        self.pool_q = nn.Linear(c.d_model, c.d_model, bias=False, dtype=torch.float32)
        self.pool_k = nn.Linear(c.d_model, c.d_model, bias=False, dtype=torch.float32)
        self.pool_v = nn.Linear(c.d_model, c.d_model, bias=False, dtype=torch.float32)
        self.pool_o = nn.Linear(c.d_model, c.d_model, bias=False, dtype=torch.float32)
        self.pool_norm = RMSNormFP32(c.d_model)

    def forward(self, prog: torch.Tensor) -> torch.Tensor:
        if prog.ndim == 2:
            prog = prog.unsqueeze(0)
        b, lp, _ = prog.shape
        op = prog[:, :, 0].clamp(0, NUM_OPS - 1)
        dst = prog[:, :, 1].clamp(0, NUM_REGS - 1)
        rega = prog[:, :, 2].clamp(0, NUM_REGS - 1)
        regb = (prog[:, :, 3] + 1).clamp(0, NUM_REGS)
        imm = prog[:, :, 4]

        imm_bits = words_to_lsb_bits(imm).to(torch.float32) * 2.0 - 1.0  # (B, L_p, 32)
        x = (
            self.op_emb(op)
            + 0.5 * self.dst_emb(dst)
            + 0.5 * self.rega_emb(rega)
            + 0.5 * self.regb_emb(regb)
            + 0.25 * self.imm_bit_proj(imm_bits)
        )
        x = self.in_norm(x)

        valid_mask = op != OP_NOP  # (B, L_p)
        # Guarantee at least slot 0 is unmasked in degenerate empty programs
        valid_mask[:, 0] = True
        for blk in self.layers:
            x, _ = blk(x, kv_cache=None, rope_offset=0, key_padding_mask=valid_mask)
        mem = self.enc_norm(x)

        q_in = self.cond_queries.expand(b, -1, -1)
        q = self.pool_q(q_in).view(b, self.cfg.n_prog_cond_tokens, self.cfg.n_heads, -1).transpose(1, 2)
        k = self.pool_k(mem).view(b, lp, self.cfg.n_heads, -1).transpose(1, 2)
        v = self.pool_v(mem).view(b, lp, self.cfg.n_heads, -1).transpose(1, 2)
        mask = valid_mask.unsqueeze(1).unsqueeze(2)
        pooled = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        pooled = pooled.transpose(1, 2).contiguous().view(b, self.cfg.n_prog_cond_tokens, self.cfg.d_model)
        return self.pool_norm(q_in + self.pool_o(pooled))


class V7CLevinMetaInverter(nn.Module):
    """Phase-Levin Universal Bytecode-Conditioned Preimage Inverter (~29.4M prod)."""

    def __init__(self, cfg: Optional[V7CConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or V7CConfig()
        c = self.cfg

        self.prog_encoder = BytecodeProgramEncoderFP32(c)
        self.tok_embed = NibbleBitwiseEmbeddingFP32(c.vocab_size, c.d_model)
        self.cur_nib_emb = nn.Embedding(9, c.d_model, dtype=torch.float32)
        self.cur_seg_emb = nn.Embedding(4, c.d_model, dtype=torch.float32)
        self.next_nib_emb = nn.Embedding(9, c.d_model, dtype=torch.float32)
        self.next_seg_emb = nn.Embedding(4, c.d_model, dtype=torch.float32)

        self.dec_layers = nn.ModuleList(
            [
                TransformerBlockFP32(
                    d_model=c.d_model,
                    n_heads=c.n_heads,
                    d_ff=c.d_ff,
                    is_causal=True,
                    max_seq_len=c.max_seq_len,
                    dropout=c.dropout,
                )
                for _ in range(c.n_dec_layers)
            ]
        )
        self.final_norm = RMSNormFP32(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False, dtype=torch.float32)
        self._sha_prog_cache: Dict[Tuple[int, str], torch.Tensor] = {}
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(mod: nn.Module) -> None:
        if isinstance(mod, (nn.Linear, nn.Embedding)):
            nn.init.normal_(mod.weight, mean=0.0, std=0.02)

    def get_compiled_sha_program(self, rounds: int, device: torch.device) -> torch.Tensor:
        key = (int(rounds), str(device))
        cached = self._sha_prog_cache.get(key)
        if cached is None:
            cached = compile_sha256_to_vm(rounds=int(rounds)).to(device)
            self._sha_prog_cache[key] = cached
        return cached

    def _build_dec_metadata(
        self,
        prompt_len: int,
        tgt_len: int,
        state_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            tgt_seg = torch.where(t_idx < state_len, 2, 3)

            max_next = min(tgt_len, total - (prompt_len - 1))
            if max_next > 0:
                next_nib[prompt_len - 1 : prompt_len - 1 + max_next] = tgt_nib[:max_next]
                next_seg[prompt_len - 1 : prompt_len - 1 + max_next] = tgt_seg[:max_next]
            cur_nib[prompt_len:total] = tgt_nib[:tgt_len]
            cur_seg[prompt_len:total] = tgt_seg[:tgt_len]
        return cur_nib, cur_seg, next_nib, next_seg

    def forward_decoder(
        self,
        prog: torch.Tensor,
        prompt_and_prev_targets: torch.Tensor,
        prompt_len: int,
        state_len: int,
    ) -> torch.Tensor:
        """Forward `[P_cond(32), prompt(66), targets[:-1]]` -> logits over `prompt+targets[:-1]`."""
        b, tok_len = prompt_and_prev_targets.shape
        dev = prompt_and_prev_targets.device
        tgt_len = max(0, tok_len - prompt_len)

        p_cond = self.prog_encoder(prog)
        if p_cond.shape[0] == 1 and b > 1:
            p_cond = p_cond.expand(b, -1, -1)

        cur_nib, cur_seg, next_nib, next_seg = self._build_dec_metadata(
            prompt_len=prompt_len, tgt_len=tgt_len, state_len=state_len, device=dev
        )
        tok_x = (
            self.tok_embed(prompt_and_prev_targets)
            + self.cur_nib_emb(cur_nib).unsqueeze(0)
            + self.cur_seg_emb(cur_seg).unsqueeze(0)
            + 0.5 * self.next_nib_emb(next_nib).unsqueeze(0)
            + 0.5 * self.next_seg_emb(next_seg).unsqueeze(0)
        )
        x = torch.cat([p_cond, tok_x], dim=1)
        for layer in self.dec_layers:
            x, _ = layer(x, kv_cache=None, rope_offset=0)
        x = self.final_norm(x[:, self.cfg.n_prog_cond_tokens :, :])
        return self.lm_head(x)

    def compute_stage1_loss(
        self,
        prog: torch.Tensor,
        x_inputs: torch.Tensor,
        y_outputs: torch.Tensor,
        n_inputs: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute exact Joint NLL over Stage 1 ARX inputs `X_{n_inputs-1}..X_0` (LSB nibbles)."""
        b = y_outputs.shape[0]
        dev = y_outputs.device
        y_pref = encode_digest_prefix(y_outputs)  # (B, 64)
        meta = torch.tensor(
            [[round_to_token(max(2, n_inputs)), BOS_TOKEN]],
            dtype=torch.int64,
            device=dev,
        ).expand(b, 2)
        prompt = torch.cat([y_pref, meta], dim=1)  # (B, 66)

        x_rev = torch.stack([x_inputs[:, r] & MASK32 for r in range(n_inputs - 1, -1, -1)], dim=1)
        targets = words_to_lsb_nibbles(x_rev).reshape(b, 8 * n_inputs)
        prompt_len = prompt.shape[1]
        tgt_len = targets.shape[1]

        in_tokens = torch.cat([prompt, targets[:, :-1]], dim=1)
        logits = self.forward_decoder(prog, in_tokens, prompt_len=prompt_len, state_len=0)
        pred_logits = logits[:, prompt_len - 1 : prompt_len - 1 + tgt_len, :]

        loss = F.cross_entropy(
            pred_logits.reshape(-1, self.cfg.vocab_size),
            targets.reshape(-1),
            ignore_index=PAD_TOKEN,
            reduction="mean",
        )
        with torch.no_grad():
            acc = float((pred_logits[..., :16].argmax(dim=-1) == targets).to(torch.float32).mean().item())
        return loss, {
            "train_loss": float(loss.detach().item()),
            "tok_acc": acc,
            "joint_nll_total": float(loss.detach().item()) * float(tgt_len),
        }

    def compute_sha_loss(
        self,
        digest: torch.Tensor,
        trace: Any,
        w16: torch.Tensor,
        rounds: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute Stage 2 SHA-256 Joint NLL conditioned on compiled `compile_sha256_to_vm(R)`."""
        prog = self.get_compiled_sha_program(rounds=rounds, device=digest.device)
        prompt = build_prompt_tokens(digest, rounds=rounds)
        targets = encode_reverse_trace_targets(trace, w16, rounds=rounds)
        prompt_len = prompt.shape[1]
        tgt_len = targets.shape[1]

        in_tokens = torch.cat([prompt, targets[:, :-1]], dim=1)
        logits = self.forward_decoder(
            prog, in_tokens, prompt_len=prompt_len, state_len=16 * rounds
        )
        pred_logits = logits[:, prompt_len - 1 : prompt_len - 1 + tgt_len, :]

        loss = F.cross_entropy(
            pred_logits.reshape(-1, self.cfg.vocab_size),
            targets.reshape(-1),
            ignore_index=PAD_TOKEN,
            reduction="mean",
        )
        with torch.no_grad():
            preds = pred_logits[..., :16].argmax(dim=-1)
            tok_acc = float((preds == targets).to(torch.float32).mean().item())
            w_acc = float(
                (preds[:, 16 * rounds :] == targets[:, 16 * rounds :]).to(torch.float32).mean().item()
            )
        return loss, {
            "train_loss": float(loss.detach().item()),
            "tok_acc": tok_acc,
            "w_nibble_acc": w_acc,
            "joint_nll_total": float(loss.detach().item()) * float(tgt_len),
        }

    @staticmethod
    def _sample_top_p_nibble(logits_16: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        if temperature <= 0.0:
            return logits_16.argmax(dim=-1)
        probs = F.softmax(logits_16 / float(temperature), dim=-1)
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum_probs = sorted_probs.cumsum(dim=-1)
            mask = (cum_probs - sorted_probs) > float(top_p)
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            choice = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
            return sorted_idx.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
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
        """Pure neural KV-cached sampling conditioned on compiled SHA bytecode + `digest` `(B, 8)`.

        100% detached from `V6A` during inference. Candidate 0 is strictly greedy (`pass@1`).
        """
        was_training = self.training
        self.eval()
        try:
            dev = digest.device
            prog = self.get_compiled_sha_program(rounds=rounds, device=dev)
            prompt = build_prompt_tokens(digest, rounds=rounds)  # (B, 66)
            b, prompt_len = prompt.shape
            n_cond = self.cfg.n_prog_cond_tokens
            tgt_len = 24 * rounds
            state_len = 16 * rounds

            p_cond = self.prog_encoder(prog).expand(b, -1, -1)

            cur_nib_p = torch.full((prompt_len - 1,), 8, dtype=torch.int64, device=dev)
            cur_seg_p = torch.full((prompt_len - 1,), 1, dtype=torch.int64, device=dev)
            h_len = min(DIGEST_PREFIX_LEN, prompt_len - 1)
            cur_nib_p[:h_len] = torch.arange(h_len, device=dev) % 8
            cur_seg_p[:h_len] = 0

            prefill_tok = (
                self.tok_embed(prompt[:, :-1])
                + self.cur_nib_emb(cur_nib_p).unsqueeze(0)
                + self.cur_seg_emb(cur_seg_p).unsqueeze(0)
                + 0.5 * self.next_nib_emb.weight[8].view(1, 1, -1)
                + 0.5 * self.next_seg_emb.weight[1].view(1, 1, -1)
            )
            x_prefill = torch.cat([p_cond, prefill_tok], dim=1)
            kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for layer in self.dec_layers:
                x_prefill, c = layer(x_prefill, kv_cache=None, rope_offset=0)
                kv_caches.append(c)

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

            cur_tok = prompt[:, -1].repeat_interleave(k_samples)
            cur_nib_id, cur_seg_id = 8, 1
            rope_pos = n_cond + prompt_len - 1

            generated = torch.empty((b * k_samples, tgt_len), dtype=torch.int64, device=dev)
            chosen_logprobs = torch.zeros((b * k_samples,), dtype=torch.float32, device=dev)

            for i in range(tgt_len):
                next_nib_id = i % 8
                next_seg_id = 2 if i < state_len else 3
                x = (
                    self.tok_embed(cur_tok).unsqueeze(1)
                    + self.cur_nib_emb.weight[cur_nib_id].view(1, 1, -1)
                    + self.cur_seg_emb.weight[cur_seg_id].view(1, 1, -1)
                    + 0.5 * self.next_nib_emb.weight[next_nib_id].view(1, 1, -1)
                    + 0.5 * self.next_seg_emb.weight[next_seg_id].view(1, 1, -1)
                )
                new_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
                for idx, layer in enumerate(self.dec_layers):
                    x, c = layer(x, kv_cache=kv_caches[idx], rope_offset=rope_pos)
                    new_caches.append(c)
                kv_caches = new_caches

                logits_16 = self.lm_head(self.final_norm(x)).squeeze(1)[:, :16]
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


class OfflineV6ACorpus:
    """In-memory store of verified offline V6A trajectories grouped by round `R`."""

    def __init__(self) -> None:
        self.by_round: Dict[int, torch.Tensor] = {}  # R -> (N, 16) int64 w16 CPU

    @classmethod
    def load(cls, path: Path | str) -> "OfflineV6ACorpus":
        corpus = cls()
        p = Path(path)
        if not p.exists():
            return corpus
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        buckets: Dict[int, List[List[int]]] = {}
        for item in data.get("records", []):
            r = int(item["rounds"])
            w16_list = [int(x) & MASK32 for x in item["w16"]]
            digest_list = [int(x) & MASK32 for x in item["digest"]]
            w_t = torch.tensor(w16_list, dtype=torch.int64).unsqueeze(0)
            d_t = torch.tensor(digest_list, dtype=torch.int64).unsqueeze(0)
            # Re-verify discrete sha256_digest on load so corrupt/fake records can never enter SFT
            if bool((sha256_digest(w_t, rounds=r) == d_t).all().item()):
                buckets.setdefault(r, []).append(w16_list)
        for r, rows in buckets.items():
            if rows:
                corpus.by_round[r] = torch.tensor(rows, dtype=torch.int64)
        return corpus

    def has_round(self, rounds: int) -> bool:
        return int(rounds) in self.by_round and self.by_round[int(rounds)].shape[0] > 0

    def sample_batch(
        self, rounds: int, batch_size: int, device: torch.device | str
    ) -> Tuple[torch.Tensor, Any, torch.Tensor]:
        pool = self.by_round[int(rounds)]
        idx = torch.randint(0, pool.shape[0], (batch_size,))
        w16 = pool[idx].to(device)
        trace = sha256_trace(w16, rounds=int(rounds))
        return w16, trace, trace.digest


def pretrain_v7c_stage1_step(
    model: V7CLevinMetaInverter,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    tier: int,
    n_inputs: int = 2,
    prog_len: int = 48,
    device: torch.device | str = "cuda",
    generator: Optional[torch.Generator] = None,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Run one Stage 1 `f_univ` random ARX curriculum training step."""
    model.train()
    prog, x_in, y_out = sample_stage1_arx_curriculum_batch(
        batch_size=batch_size,
        tier=tier,
        n_inputs=n_inputs,
        prog_len=prog_len,
        device=device,
        generator=generator,
    )
    optimizer.zero_grad(set_to_none=True)
    loss, stats = model.compute_stage1_loss(
        prog=prog, x_inputs=x_in, y_outputs=y_out, n_inputs=n_inputs
    )
    loss.backward()
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return stats


def train_v7c_stage2_step(
    model: V7CLevinMetaInverter,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    rounds: int,
    device: torch.device | str = "cuda",
    generator: Optional[torch.Generator] = None,
    v6a_corpus: Optional[OfflineV6ACorpus] = None,
    v6a_mix_prob: float = 0.0,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Run one Stage 2 SHA-256 curriculum step with optional offline V6A corpus mixing."""
    model.train()
    use_corpus = (
        v6a_corpus is not None
        and v6a_corpus.has_round(rounds)
        and (random.random() < float(v6a_mix_prob))
    )
    if use_corpus and v6a_corpus is not None:
        w16, trace, digest = v6a_corpus.sample_batch(
            rounds=rounds, batch_size=batch_size, device=device
        )
        src_tag = 1.0
    else:
        w16, trace, digest = sample_online_sha_batch(
            batch_size=batch_size, rounds=rounds, device=device, generator=generator
        )
        src_tag = 0.0

    optimizer.zero_grad(set_to_none=True)
    loss, stats = model.compute_sha_loss(
        digest=digest, trace=trace, w16=w16, rounds=rounds
    )
    loss.backward()
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    stats["v6a_corpus_used"] = src_tag
    return stats


__all__ = [
    "V7CConfig",
    "BytecodeProgramEncoderFP32",
    "V7CLevinMetaInverter",
    "OfflineV6ACorpus",
    "pretrain_v7c_stage1_step",
    "train_v7c_stage2_step",
]
