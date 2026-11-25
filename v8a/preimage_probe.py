"""Frozen torus-flow preimage probes and independent discrete forward verification."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from v8a import configure_strict_fp32
from v8a.sha256_stream import (
    OnlineSHA256Stream,
    bits_to_words,
    sha256_digest_bits,
    sha256_reduced_reference_bytes,
    words_to_bytes,
)
from v8a.torus_flow import TorusFlow, dequantize_bits, wrap_2pi


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallMLPProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActiveAxesProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def extract_torus_features(
    flow: TorusFlow,
    h_bits: torch.Tensor,
    batch_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract deterministic (u=0) disentangled torus angles and [cos, sin] features."""
    was_training = flow.training
    flow.eval()
    theta_chunks: List[torch.Tensor] = []
    for start in range(0, h_bits.shape[0], batch_size):
        theta_0 = dequantize_bits(h_bits[start : start + batch_size], deterministic=True)
        theta_chunks.append(flow(theta_0))
    theta_out = torch.cat(theta_chunks, dim=0)
    z = torch.cat([torch.cos(theta_out), torch.sin(theta_out)], dim=-1)
    if was_training:
        flow.train()
    return theta_out, z


def _fit_probe(
    model: nn.Module,
    z_train: torch.Tensor,
    w_train: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> nn.Module:
    device = z_train.device
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=lr * 0.05)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    n = z_train.shape[0]
    bs = min(batch_size, n)

    model.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), device=device, generator=gen)
        logits = model(z_train[idx])
        loss = F.binary_cross_entropy_with_logits(logits, w_train[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    return model


def _binomial_z_score(acc: float, total_bits: int) -> float:
    sigma = math.sqrt(0.25 / max(1, total_bits))
    return (acc - 0.5) / sigma


@torch.no_grad()
def _verify_hits_with_reference(
    pred_words: torch.Tensor,
    eval_H_bits: torch.Tensor,
    hit_indices: List[int],
    rounds: int,
    max_checks: int = 32,
) -> None:
    """Cross-verify GPU SHA-256 matches using independent pure-Python reference."""
    if not hit_indices:
        return
    sample_idx = hit_indices[:max_checks]
    sub_words = pred_words[sample_idx]
    msg_list = words_to_bytes(sub_words)
    target_words = bits_to_words(eval_H_bits[sample_idx])
    expected_bytes = words_to_bytes(target_words)
    for msg_b, exp_b in zip(msg_list, expected_bytes):
        ref_b = sha256_reduced_reference_bytes(msg_b, rounds=rounds)
        if ref_b != exp_b:
            raise AssertionError("Independent pure-Python SHA-256 verification mismatch!")


@torch.no_grad()
def _evaluate_exact_passes(
    probe: nn.Module,
    logits_eval: torch.Tensor,
    theta_eval: torch.Tensor,
    eval_H_bits: torch.Tensor,
    k_bits: int,
    rounds: int,
    seed: int,
    chunk_size: int = 16384,
) -> Dict[str, float | int]:
    device = eval_H_bits.device
    n_eval = eval_H_bits.shape[0]

    # 1. exact_pass@1
    hard_bits = (logits_eval > 0.0).to(torch.float32)
    hard_words = bits_to_words(hard_bits)
    h_pred_1 = sha256_digest_bits(hard_words, k_bits=k_bits, rounds=rounds)
    pass1_mask = torch.all(h_pred_1 == eval_H_bits, dim=-1)
    pass1_count = int(pass1_mask.sum().item())
    pass1_indices = torch.nonzero(pass1_mask, as_tuple=False).flatten().tolist()
    _verify_hits_with_reference(hard_words, eval_H_bits, pass1_indices, rounds=rounds)

    # 2. exact_pass@64:
    #    [0]: hard argmax
    #    [1..31]: all 31 non-empty bit-flip combinations of 5 lowest-|logit| bits
    #    [32..47]: 16 Bernoulli posterior samples
    #    [48..63]: 16 local latent perturbation samples around theta_eval
    cands = torch.empty((n_eval, 64, k_bits), dtype=torch.float32, device=device)
    cands[:, 0, :] = hard_bits

    uncertain_idx = torch.topk(-logits_eval.abs(), k=5, dim=-1).indices  # [N, 5]
    flip_masks = torch.tensor(
        [[(m >> b) & 1 for b in range(5)] for m in range(1, 32)],
        dtype=torch.float32,
        device=device,
    )  # [31, 5]
    base_31 = hard_bits.unsqueeze(1).expand(n_eval, 31, k_bits).clone()
    gathered_base = torch.gather(
        hard_bits, dim=1, index=uncertain_idx
    ).unsqueeze(1)  # [N, 1, 5]
    flipped_vals = torch.abs(gathered_base - flip_masks.unsqueeze(0))  # [N, 31, 5]
    scatter_idx = uncertain_idx.unsqueeze(1).expand(n_eval, 31, 5)
    base_31.scatter_(dim=2, index=scatter_idx, src=flipped_vals)
    cands[:, 1:32, :] = base_31

    gen = torch.Generator(device=device).manual_seed(int(seed) + 77)
    probs = torch.sigmoid(logits_eval).unsqueeze(1).expand(n_eval, 16, k_bits)
    cands[:, 32:48, :] = torch.bernoulli(probs, generator=gen)

    noise = torch.randn((n_eval, 16, 256), device=device, generator=gen) * 0.08
    theta_noisy = wrap_2pi(theta_eval.unsqueeze(1) + noise).reshape(n_eval * 16, 256)
    z_noisy = torch.cat([torch.cos(theta_noisy), torch.sin(theta_noisy)], dim=-1)
    noisy_logits = probe(z_noisy).reshape(n_eval, 16, k_bits)
    cands[:, 48:64, :] = (noisy_logits > 0.0).to(torch.float32)

    flat_words = bits_to_words(cands.reshape(n_eval * 64, k_bits))
    h_chunks: List[torch.Tensor] = []
    for start in range(0, flat_words.shape[0], chunk_size):
        h_chunks.append(
            sha256_digest_bits(flat_words[start : start + chunk_size], k_bits=k_bits, rounds=rounds)
        )
    h_cand = torch.cat(h_chunks, dim=0).reshape(n_eval, 64, 256)
    cand_matches = torch.all(h_cand == eval_H_bits.unsqueeze(1), dim=-1)  # [N, 64]
    pass64_mask = cand_matches.any(dim=1)
    pass64_count = int(pass64_mask.sum().item())

    if pass64_count > 0:
        hit_rows = torch.nonzero(pass64_mask, as_tuple=False).flatten()[:16].tolist()
        best_c = torch.argmax(cand_matches[hit_rows].to(torch.int64), dim=1).tolist()
        hit_words = torch.stack(
            [flat_words.reshape(n_eval, 64, -1)[r, c] for r, c in zip(hit_rows, best_c)],
            dim=0,
        )
        _verify_hits_with_reference(
            hit_words,
            eval_H_bits[hit_rows],
            list(range(len(hit_rows))),
            rounds=rounds,
        )

    return {
        "exact_pass_1": pass1_count / n_eval,
        "exact_pass_1_count": pass1_count,
        "exact_pass_64": pass64_count / n_eval,
        "exact_pass_64_count": pass64_count,
        "n_eval": n_eval,
    }


def evaluate_frozen_preimage_probe(
    flow: TorusFlow,
    stream: OnlineSHA256Stream,
    probe_train_size: int = 32768,
    probe_steps: int = 350,
    batch_size: int = 512,
    seed: int = 12345,
) -> Dict[str, float | int | List[int]]:
    """Freeze flow completely, train lightweight probes on disjoint train set, and verify on eval."""
    configure_strict_fp32()
    was_training = flow.training
    flow.eval()

    _, w_train, h_train = stream.sample_probe_train_set(probe_train_size, seed=seed)
    theta_train, z_train = extract_torus_features(flow, h_train, batch_size=1024)
    theta_eval, z_eval = extract_torus_features(flow, stream.eval_H_bits, batch_size=1024)

    # Identify top-k active (highest entropy / lowest R_disc) channels strictly on probe_train
    with torch.no_grad():
        c1 = torch.cos(theta_train).mean(dim=0)
        s1 = torch.sin(theta_train).mean(dim=0)
        r_disc_train = torch.sqrt(c1 * c1 + s1 * s1)
        active_axes = torch.argsort(r_disc_train, descending=False)[: stream.k_bits]
        active_feat_idx = torch.cat([active_axes, active_axes + 256])

    lin_probe = _fit_probe(
        LinearProbe(512, stream.k_bits),
        z_train,
        w_train,
        steps=min(300, probe_steps),
        batch_size=batch_size,
        lr=3e-3,
        seed=seed + 1,
    )
    mlp_probe = _fit_probe(
        SmallMLPProbe(512, stream.k_bits, hidden_dim=256),
        z_train,
        w_train,
        steps=probe_steps,
        batch_size=batch_size,
        lr=3e-3,
        seed=seed + 2,
    )
    act_probe = _fit_probe(
        ActiveAxesProbe(2 * stream.k_bits, stream.k_bits, hidden_dim=128),
        z_train.index_select(1, active_feat_idx),
        w_train,
        steps=probe_steps,
        batch_size=batch_size,
        lr=3e-3,
        seed=seed + 3,
    )

    with torch.no_grad():
        logits_lin = lin_probe(z_eval)
        logits_mlp = mlp_probe(z_eval)
        logits_act = act_probe(z_eval.index_select(1, active_feat_idx))

        acc_lin = float(((logits_lin > 0.0).to(torch.float32) == stream.eval_W_bits).float().mean().item())
        acc_mlp = float(((logits_mlp > 0.0).to(torch.float32) == stream.eval_W_bits).float().mean().item())
        acc_act = float(((logits_act > 0.0).to(torch.float32) == stream.eval_W_bits).float().mean().item())

        if acc_mlp >= acc_lin:
            best_probe, best_logits, best_name, best_acc = mlp_probe, logits_mlp, "mlp", acc_mlp
        else:
            best_probe, best_logits, best_name, best_acc = lin_probe, logits_lin, "linear", acc_lin

    pass_metrics = _evaluate_exact_passes(
        probe=best_probe,
        logits_eval=best_logits,
        theta_eval=theta_eval,
        eval_H_bits=stream.eval_H_bits,
        k_bits=stream.k_bits,
        rounds=stream.rounds,
        seed=seed,
    )

    if was_training:
        flow.train()

    total_eval_bits = stream.n_eval * stream.k_bits
    return {
        "w_bit_acc": best_acc,
        "w_bit_acc_linear": acc_lin,
        "w_bit_acc_mlp": acc_mlp,
        "w_bit_acc_active": acc_act,
        "best_probe_type": best_name,
        "z_score": _binomial_z_score(best_acc, total_eval_bits),
        "z_score_linear": _binomial_z_score(acc_lin, total_eval_bits),
        "z_score_active": _binomial_z_score(acc_act, total_eval_bits),
        "active_axes": active_axes.detach().cpu().tolist(),
        **pass_metrics,
    }
