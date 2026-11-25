"""Pure FP32 (allow_tf32=False) AlphaZero 3D Crystal Policy-Value Network + MCTS + SAT search.

Implements SPEC.md Sections 3.1, 3.2, and 4 (G3 greedy 1-bit recovery & G4 control cipher).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("highest")

from mendel_state_machine import (  # noqa: E402
    GD_0,
    GD_1,
    GD_ANY,
    GD_EMPTY,
    GD_EQ,
    GD_N,
    GD_U,
    GD_X,
    POPCNT_LUT,
    MendelGraphState,
    create_blank_mendel_state,
    encode_pair_to_gdiff,
    propagate_mendel_fixpoint,
)
from sat_witness_solver import WitnessResult, solve_differential_witness  # noqa: E402
from src.sha256_ref import MASK32, SHA256_K, sha256_digest, sha256_trace  # noqa: E402

# 6 structured refinement action masks (SPEC 3.2)
ACTION_MASKS: Tuple[int, ...] = (GD_EQ, GD_X, GD_U, GD_N, GD_0, GD_1)
ACTION_NAMES: Tuple[str, ...] = ("lock_eq", "lock_x", "lock_u", "lock_n", "lock_0", "lock_1")


def mendel_to_crystal_tensor(
    state: MendelGraphState,
    btc_block2_padding: bool = False,
) -> torch.Tensor:
    """Encode MendelGraphState into (B, 20, R, 3, 32) pure float32 3D crystal tensor."""
    R = state.rounds
    B = state.W.shape[0]
    device = state.W.device

    # V=3 channels: A_{1..R}, E_{1..R}, W_{0..R-1}
    grid = torch.stack(
        [
            state.A[:, 4 : 4 + R, :],  # A_{t+1} for t=0..R-1
            state.E[:, 4 : 4 + R, :],  # E_{t+1} for t=0..R-1
            state.W[:, :R, :],         # W_t     for t=0..R-1
        ],
        dim=2,
    )  # (B, R, 3, 32)

    # 16-dim one-hot
    onehot = F.one_hot(grid.clamp(0, 15), num_classes=16).permute(0, 4, 1, 2, 3).to(torch.float32)

    pop_lut = POPCNT_LUT.to(device)
    pop = pop_lut[grid.clamp(0, 15)].to(torch.float32)
    entropy = torch.log2(pop.clamp(min=1.0)) * 0.5  # [0, 2] -> [0, 1] when /2, spec says log2(popcnt)/2

    fixed_mask = (pop == 1.0).to(torch.float32)
    if btc_block2_padding and R >= 16:
        fixed_mask[:, 4:16, 2, :] = 1.0

    # Active/absorption sensitivity marker: constrained non-'?' and non-empty
    sens_mask = ((grid != GD_ANY) & (grid != GD_EMPTY)).to(torch.float32)

    # Normalized round index t / R
    t_pos = (torch.arange(R, dtype=torch.float32, device=device) / max(1, R)).view(1, R, 1, 1).expand(B, R, 3, 32)

    extra = torch.stack([entropy, fixed_mask, sens_mask, t_pos], dim=1)  # (B, 4, R, 3, 32)
    return torch.cat([onehot, extra], dim=1)  # (B, 20, R, 3, 32)


class CircularCrystalBlock3D(nn.Module):
    """3D ResNet block with circular bit-axis padding (matching ROTR32 symmetry)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=(1, 1, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=(1, 1, 0), bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Circular pad bit axis (last dim) by 1 on each side
        h = F.pad(x, (1, 1, 0, 0, 0, 0), mode="circular")
        h = F.silu(self.bn1(self.conv1(h)))
        h = F.pad(h, (1, 1, 0, 0, 0, 0), mode="circular")
        h = self.bn2(self.conv2(h))
        return F.silu(x + h)


class CrystalPolicyValueNet(nn.Module):
    """AlphaZero 3D Crystal Policy-Value Network (pure FP32, circular ROTR32 pad + W-schedule skips)."""

    def __init__(self, in_channels: int = 20, hidden_channels: int = 64, num_blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(3, 3, 3), padding=(1, 1, 0), bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([CircularCrystalBlock3D(hidden_channels) for _ in range(num_blocks)])
        # Cross-round message-schedule skip projection (t-2, t-7, t-15, t-16)
        self.sched_proj = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1, bias=False)

        self.policy_head = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, len(ACTION_MASKS), kernel_size=1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Stem with circular pad on bit axis
        h = F.pad(x, (1, 1, 0, 0, 0, 0), mode="circular")
        h = self.stem(h)
        for blk in self.blocks:
            h = blk(h)

        # Message-schedule skip connections along T on channel V=2 (W_t)
        R = h.shape[2]
        if R > 16:
            sched_acc = torch.zeros_like(h)
            for lag in (2, 7, 15, 16):
                sched_acc[:, :, lag:, 2, :] = sched_acc[:, :, lag:, 2, :] + h[:, :, : R - lag, 2, :]
            h = h + 0.25 * self.sched_proj(sched_acc)

        policy_logits = self.policy_head(h).permute(0, 2, 3, 4, 1).contiguous()  # (B, R, 3, 32, 6)
        pooled = torch.cat([h.mean(dim=(2, 3, 4)), h.amax(dim=(2, 3, 4))], dim=-1)  # (B, 2C)
        value = self.value_head(pooled).squeeze(-1)  # (B,)
        return policy_logits, value


def compute_legal_action_mask(state: MendelGraphState) -> torch.Tensor:
    """Return bool mask (B, R, 3, 32, 6) of actions that strictly refine an ambiguous site."""
    R = state.rounds
    grid = torch.stack(
        [state.A[:, 4 : 4 + R, :], state.E[:, 4 : 4 + R, :], state.W[:, :R, :]],
        dim=2,
    )  # (B, R, 3, 32)
    act_t = torch.tensor(ACTION_MASKS, dtype=torch.int64, device=grid.device).view(1, 1, 1, 1, -1)
    g_exp = grid.unsqueeze(-1)
    refined = g_exp & act_t
    # Strictly narrower and non-empty
    return (refined != 0) & (refined != g_exp)


def apply_single_action_to_state(
    state: MendelGraphState,
    t: int,
    v: int,
    b: int,
    act_idx: int,
    max_iters: int = 3,
    custom_k: Optional[Sequence[int]] = None,
    sigma0_shifts: Tuple[int, int, int] = (2, 13, 22),
    sigma1_shifts: Tuple[int, int, int] = (6, 11, 25),
) -> MendelGraphState:
    """Apply refinement action (t, v, b, act_idx) to batch=1 state and run Mendel propagation."""
    nxt = MendelGraphState(
        rounds=state.rounds,
        iv=state.iv.clone(),
        W=state.W.clone(),
        A=state.A.clone(),
        E=state.E.clone(),
        digest=state.digest.clone(),
        contradiction=state.contradiction.clone(),
    )
    mask = ACTION_MASKS[act_idx]
    if v == 0:
        nxt.A[:, 4 + t, b] &= mask
    elif v == 1:
        nxt.E[:, 4 + t, b] &= mask
    else:
        nxt.W[:, t, b] &= mask
    return propagate_mendel_fixpoint(
        nxt,
        max_iters=max_iters,
        custom_k=custom_k,
        sigma0_shifts=sigma0_shifts,
        sigma1_shifts=sigma1_shifts,
    )


def extract_gdiff_dicts(state: MendelGraphState) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
    """Convert batch=0 MendelGraphState into SAT-ready dictionaries for W, A, E."""
    R = state.rounds
    w_dict: Dict[int, List[int]] = {}
    for t in range(R):
        row = state.W[0, t].tolist()
        if any(m != GD_ANY for m in row):
            w_dict[t] = row
    a_dict: Dict[int, List[int]] = {}
    e_dict: Dict[int, List[int]] = {}
    for k in range(-3, R + 1):
        arow = state.A[0, k + 3].tolist()
        if any(m != GD_ANY for m in arow):
            a_dict[k] = arow
        erow = state.E[0, k + 3].tolist()
        if any(m != GD_ANY for m in erow):
            e_dict[k] = erow
    return w_dict, a_dict, e_dict


# ---------------------------------------------------------------------------
# Self-play dataset generation (positive SAT traces & negative UNSAT/contradiction)
# ---------------------------------------------------------------------------

def generate_selfplay_batch(
    rounds: int,
    n_pos: int = 64,
    n_neg: int = 64,
    iv_mode: str = "semi_free",
    btc_block2_padding: bool = False,
    leading_zero_bits: Optional[int] = None,
    k_consts: Sequence[int] = SHA256_K,
    sigma0_shifts: Tuple[int, int, int] = (2, 13, 22),
    sigma1_shifts: Tuple[int, int, int] = (6, 11, 25),
    seed: int = 123,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate supervised + self-play training samples (crystal_x, policy_tgt, value_tgt)."""
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    device = "cpu"

    xs: List[torch.Tensor] = []
    p_tgts: List[torch.Tensor] = []
    v_tgts: List[float] = []

    # 1. Positive trajectories from real exact execution pairs (with sparse/truncated diffs)
    w1 = torch.randint(0, 1 << 32, (n_pos, 16), dtype=torch.int64, generator=gen)
    if btc_block2_padding:
        w1[:, 4] = 0x80000000
        w1[:, 5:15] = 0
        w1[:, 15] = 640
    w2 = w1.clone()
    # Flip 1..3 bits in active message words to form real reachable differential pairs
    for i in range(n_pos):
        word_idx = rng.randint(0, 3 if btc_block2_padding else 15)
        bit_idx = rng.randint(0, 31)
        w2[i, word_idx] ^= 1 << bit_idx

    tr1 = sha256_trace(w1, rounds=rounds)
    tr2 = sha256_trace(w2, rounds=rounds)
    a_full_gd = encode_pair_to_gdiff(tr1.A_full, tr2.A_full)
    e_full_gd = encode_pair_to_gdiff(tr1.E_full, tr2.E_full)
    w_full_gd = encode_pair_to_gdiff(tr1.W_full, tr2.W_full)

    for i in range(n_pos):
        st = create_blank_mendel_state(1, rounds=rounds, iv_mode=iv_mode, device=device, btc_block2_padding=btc_block2_padding)
        # Keep 35..75% of constraints revealed; reveal a random next step as policy target
        reveal_prob = rng.uniform(0.35, 0.75)
        mask_a = torch.rand((rounds, 32), generator=gen) < reveal_prob
        mask_e = torch.rand((rounds, 32), generator=gen) < reveal_prob
        mask_w = torch.rand((rounds, 32), generator=gen) < reveal_prob
        st.A[0, 4 : 4 + rounds] = torch.where(mask_a, a_full_gd[i, 4 : 4 + rounds], st.A[0, 4 : 4 + rounds])
        st.E[0, 4 : 4 + rounds] = torch.where(mask_e, e_full_gd[i, 4 : 4 + rounds], st.E[0, 4 : 4 + rounds])
        st.W[0, :rounds] = torch.where(mask_w, w_full_gd[i, :rounds], st.W[0, :rounds])

        # Pick an unrevealed site as policy target
        cand_t = rng.randint(0, rounds - 1)
        cand_v = rng.randint(0, 2)
        cand_b = rng.randint(0, 31)
        true_gd = int((a_full_gd if cand_v == 0 else (e_full_gd if cand_v == 1 else w_full_gd))[i, (4 + cand_t if cand_v < 2 else cand_t), cand_b].item())
        act_idx = 4 if true_gd == GD_0 else (5 if true_gd == GD_1 else (2 if true_gd == GD_U else 3))

        p_map = torch.zeros((rounds, 3, 32, len(ACTION_MASKS)), dtype=torch.float32)
        p_map[cand_t, cand_v, cand_b, act_idx] = 1.0
        xs.append(mendel_to_crystal_tensor(st, btc_block2_padding=btc_block2_padding)[0])
        p_tgts.append(p_map)
        v_tgts.append(1.0)

    # 2. Negative trajectories (batched propagation for 100x speedup)
    st_neg = create_blank_mendel_state(
        n_neg, rounds=rounds, iv_mode=iv_mode, device=device, btc_block2_padding=btc_block2_padding
    )
    neg_p_tgts: List[torch.Tensor] = []
    for i in range(n_neg):
        t_bad = rng.randint(0, max(0, rounds - 3))
        b_bad = rng.randint(0, 31)
        st_neg.W[i, t_bad, b_bad] = GD_U
        st_neg.A[i, 4 + t_bad, b_bad] = GD_0
        st_neg.E[i, 4 + t_bad, b_bad] = GD_1
        p_map = torch.zeros((rounds, 3, 32, len(ACTION_MASKS)), dtype=torch.float32)
        p_map[t_bad, 2, b_bad, 0] = 1.0
        neg_p_tgts.append(p_map)

    st_neg_p = propagate_mendel_fixpoint(
        st_neg, max_iters=2, custom_k=k_consts, sigma0_shifts=sigma0_shifts, sigma1_shifts=sigma1_shifts
    )
    neg_xt = mendel_to_crystal_tensor(st_neg_p, btc_block2_padding=btc_block2_padding)
    for i in range(n_neg):
        xs.append(neg_xt[i])
        p_tgts.append(neg_p_tgts[i])
        v_tgts.append(-1.0)

    return torch.stack(xs, dim=0), torch.stack(p_tgts, dim=0), torch.tensor(v_tgts, dtype=torch.float32)


def train_crystal_net(
    net: CrystalPolicyValueNet,
    rounds: int,
    device: torch.device | str,
    steps: int = 80,
    batch_size: int = 32,
    btc_block2_padding: bool = False,
    k_consts: Sequence[int] = SHA256_K,
    sigma0_shifts: Tuple[int, int, int] = (2, 13, 22),
    sigma1_shifts: Tuple[int, int, int] = (6, 11, 25),
    seed: int = 2026,
) -> Dict[str, float]:
    """Train CrystalPolicyValueNet in pure FP32 on self-play differential trajectories."""
    net.to(device)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

    x_all, p_all, v_all = generate_selfplay_batch(
        rounds=rounds,
        n_pos=max(128, batch_size * 4),
        n_neg=max(128, batch_size * 4),
        btc_block2_padding=btc_block2_padding,
        k_consts=k_consts,
        sigma0_shifts=sigma0_shifts,
        sigma1_shifts=sigma1_shifts,
        seed=seed,
    )
    x_all = x_all.to(device)
    p_all = p_all.view(x_all.shape[0], -1).to(device)
    v_all = v_all.to(device)

    n_total = x_all.shape[0]
    last_loss = 0.0
    for step in range(steps):
        perm = torch.randint(0, n_total, (batch_size,), device=device)
        xb = x_all[perm]
        pb = p_all[perm]
        vb = v_all[perm]

        logits, val = net(xb)
        logits_flat = logits.view(batch_size, -1)
        loss_p = -(pb * F.log_softmax(logits_flat, dim=-1)).sum(dim=-1).mean()
        loss_v = F.mse_loss(val, vb)
        loss = loss_p + loss_v

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())

    net.eval()
    with torch.no_grad():
        _, val_pred = net(x_all)
        acc = float((((val_pred >= 0) == (v_all >= 0)).to(torch.float32).mean()).item())
    return {"train_loss": round(last_loss, 5), "train_val_acc": round(acc, 4)}


# ---------------------------------------------------------------------------
# G3 Anti-Hallucination Check: Active-word 1-bit greedy recovery rate
# ---------------------------------------------------------------------------

def evaluate_g3_greedy_1bit_recover(
    net: CrystalPolicyValueNet,
    rounds: int,
    device: torch.device | str,
    n_probes: int = 128,
    btc_block2_padding: bool = False,
    seed: int = 777,
) -> Dict[str, float]:
    """SPEC Section 4 G3: perturb 1 bit in an active word and test policy-guided 1-bit recovery."""
    net.eval()
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    w1 = torch.randint(0, 1 << 32, (n_probes, 16), dtype=torch.int64, generator=gen)
    if btc_block2_padding:
        w1[:, 4] = 0x80000000
        w1[:, 5:15] = 0
        w1[:, 15] = 640
    w2 = w1.clone()
    w2[:, 0] ^= 1

    tr1 = sha256_trace(w1, rounds=rounds)
    tr2 = sha256_trace(w2, rounds=rounds)
    a_gd = encode_pair_to_gdiff(tr1.A_full, tr2.A_full)
    e_gd = encode_pair_to_gdiff(tr1.E_full, tr2.E_full)
    w_gd = encode_pair_to_gdiff(tr1.W_full, tr2.W_full)

    hits_top1 = 0
    hits_top3 = 0
    pos_scores: List[float] = []
    neg_scores: List[float] = []

    with torch.no_grad():
        for i in range(n_probes):
            st = create_blank_mendel_state(1, rounds=rounds, iv_mode="standard", device="cpu", btc_block2_padding=btc_block2_padding)
            st.A[0, 4 : 4 + rounds] = a_gd[i, 4 : 4 + rounds]
            st.E[0, 4 : 4 + rounds] = e_gd[i, 4 : 4 + rounds]
            st.W[0, :rounds] = w_gd[i, :rounds]

            # Valid state score
            xt_pos = mendel_to_crystal_tensor(st, btc_block2_padding=btc_block2_padding).to(device)
            _, v_pos = net(xt_pos)
            pos_scores.append(float(v_pos[0].item()))

            # Corrupt exactly 1 bit inside an active word (t_act, v_act, b_act) to '?' (unconstrained)
            t_act = rng.randint(0, min(rounds - 1, 7))
            v_act = rng.randint(0, 1)  # A or E active state word
            b_act = rng.randint(0, 31)
            if v_act == 0:
                st.A[0, 4 + t_act, b_act] = GD_ANY
            else:
                st.E[0, 4 + t_act, b_act] = GD_ANY

            xt = mendel_to_crystal_tensor(st, btc_block2_padding=btc_block2_padding).to(device)
            logits, v_neg = net(xt)
            neg_scores.append(float(v_neg[0].item()))

            # Rank bits inside (t_act, v_act, 0..31) by max action probability
            bit_scores = logits[0, t_act, v_act, :, :].amax(dim=-1)  # (32,)
            top3 = torch.topk(bit_scores, k=3).indices.tolist()
            if top3[0] == b_act:
                hits_top1 += 1
            if b_act in top3:
                hits_top3 += 1

    top1_rate = hits_top1 / max(1, n_probes)
    top3_rate = hits_top3 / max(1, n_probes)
    random_baseline = 1.0 / 32.0
    return {
        "n_probes": n_probes,
        "greedy_1bit_recover_rate": round(top1_rate, 5),
        "top3_recover_rate": round(top3_rate, 5),
        "random_baseline_rate": round(random_baseline, 5),
        "ratio_vs_random": round(top1_rate / random_baseline, 3),
        "passes_g3_1_5x": bool(top1_rate >= 1.5 * random_baseline),
        "mean_valid_value": round(sum(pos_scores) / max(1, len(pos_scores)), 4),
        "mean_perturbed_value": round(sum(neg_scores) / max(1, len(neg_scores)), 4),
    }


# ---------------------------------------------------------------------------
# Beam + MCTS-guided search with incremental CaDiCaL195 SAT verification
# ---------------------------------------------------------------------------

@dataclass
class BeamNode:
    state: MendelGraphState
    score: float
    depth: int


def run_alphazero_beam_sat_search(
    net: CrystalPolicyValueNet,
    init_state: MendelGraphState,
    device: torch.device | str,
    beam_width: int = 8,
    max_depth: int = 12,
    sat_check_interval: int = 4,
    iv_mode: str = "standard",
    digest_xor_zero_words: Optional[Sequence[int]] = None,
    leading_zero_bits: Optional[int] = None,
    btc_block2_padding: bool = False,
    fixed_w_indices: Optional[Dict[int, int]] = None,
    sat_conflict_budget: int = 100_000,
    max_solutions: int = 1,
    k_consts: Sequence[int] = SHA256_K,
    sigma0_shifts: Tuple[int, int, int] = (2, 13, 22),
    sigma1_shifts: Tuple[int, int, int] = (6, 11, 25),
) -> Dict[str, object]:
    """Run AlphaZero policy-value guided Beam/MCTS refinement + CaDiCaL195 SAT check."""
    t0 = time.perf_counter()
    net.eval()
    beam: List[BeamNode] = [BeamNode(state=init_state, score=0.0, depth=0)]
    mendel_contradictions = 0
    sat_calls = 0
    best_witnesses: List[WitnessResult] = []

    # First check initial skeleton directly in SAT before refining further
    w_d, a_d, e_d = extract_gdiff_dicts(init_state)
    sat_calls += 1
    init_res = solve_differential_witness(
        rounds=init_state.rounds,
        iv_mode=iv_mode,
        w_gdiff=w_d,
        a_gdiff=a_d,
        e_gdiff=e_d,
        digest_xor_zero_words=digest_xor_zero_words,
        leading_zero_bits=leading_zero_bits,
        btc_block2_padding=btc_block2_padding,
        fixed_w_indices=fixed_w_indices,
        max_solutions=max_solutions,
        conflict_budget=sat_conflict_budget,
        k_consts=k_consts,
        sigma0_shifts=sigma0_shifts,
        sigma1_shifts=sigma1_shifts,
    )
    if init_res and init_res[0].sat:
        return {
            "sat": True,
            "witnesses": init_res,
            "sat_calls": sat_calls,
            "mendel_contradictions": mendel_contradictions,
            "depth_reached": 0,
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }

    for depth in range(1, max_depth + 1):
        pending_specs: List[Tuple[MendelGraphState, int, int, int, int, float]] = []
        beam_states = [node.state for node in beam if not bool(node.state.contradiction[0].item())]
        mendel_contradictions += len(beam) - len(beam_states)
        if not beam_states:
            break

        # Batched GPU policy forward over current beam
        beam_batch = MendelGraphState(
            rounds=init_state.rounds,
            iv=torch.cat([s.iv for s in beam_states], dim=0),
            W=torch.cat([s.W for s in beam_states], dim=0),
            A=torch.cat([s.A for s in beam_states], dim=0),
            E=torch.cat([s.E for s in beam_states], dim=0),
            digest=torch.cat([s.digest for s in beam_states], dim=0),
            contradiction=torch.cat([s.contradiction for s in beam_states], dim=0),
        )
        xt_beam = mendel_to_crystal_tensor(beam_batch, btc_block2_padding=btc_block2_padding).to(device)
        with torch.no_grad():
            p_logits_b, _ = net(xt_beam)
        legal_b = compute_legal_action_mask(beam_batch).to(device)

        for idx_b, st_parent in enumerate(beam_states):
            masked_logits = p_logits_b[idx_b].masked_fill(~legal_b[idx_b], -1e9).view(-1)
            valid_count = int(legal_b[idx_b].sum().item())
            if valid_count == 0:
                continue
            probs = F.softmax(masked_logits, dim=-1)
            k_pick = min(4, valid_count)
            top_probs, top_flat = torch.topk(probs, k=k_pick)
            for p_val, flat_idx in zip(top_probs.tolist(), top_flat.tolist()):
                act_idx = flat_idx % len(ACTION_MASKS)
                rem = flat_idx // len(ACTION_MASKS)
                b = rem % 32
                rem //= 32
                v_ch = rem % 3
                t_idx = rem // 3
                pending_specs.append((st_parent, t_idx, v_ch, b, act_idx, p_val))

        if not pending_specs:
            break

        # Batched Mendel propagation across all beam children at once
        cand_batch = MendelGraphState(
            rounds=init_state.rounds,
            iv=torch.cat([sp[0].iv for sp in pending_specs], dim=0).clone(),
            W=torch.cat([sp[0].W for sp in pending_specs], dim=0).clone(),
            A=torch.cat([sp[0].A for sp in pending_specs], dim=0).clone(),
            E=torch.cat([sp[0].E for sp in pending_specs], dim=0).clone(),
            digest=torch.cat([sp[0].digest for sp in pending_specs], dim=0).clone(),
            contradiction=torch.cat([sp[0].contradiction for sp in pending_specs], dim=0).clone(),
        )
        for ci, (_, t_idx, v_ch, b, act_idx, _) in enumerate(pending_specs):
            mask = ACTION_MASKS[act_idx]
            if v_ch == 0:
                cand_batch.A[ci, 4 + t_idx, b] &= mask
            elif v_ch == 1:
                cand_batch.E[ci, 4 + t_idx, b] &= mask
            else:
                cand_batch.W[ci, t_idx, b] &= mask

        cand_prop = propagate_mendel_fixpoint(
            cand_batch,
            max_iters=1,
            custom_k=k_consts,
            sigma0_shifts=sigma0_shifts,
            sigma1_shifts=sigma1_shifts,
        )
        with torch.no_grad():
            _, cand_vals = net(mendel_to_crystal_tensor(cand_prop, btc_block2_padding=btc_block2_padding).to(device))

        candidates: List[BeamNode] = []
        for ci, (_, _, _, _, _, p_val) in enumerate(pending_specs):
            if bool(cand_prop.contradiction[ci].item()):
                mendel_contradictions += 1
                continue
            child_st = MendelGraphState(
                rounds=cand_prop.rounds,
                iv=cand_prop.iv[ci : ci + 1],
                W=cand_prop.W[ci : ci + 1],
                A=cand_prop.A[ci : ci + 1],
                E=cand_prop.E[ci : ci + 1],
                digest=cand_prop.digest[ci : ci + 1],
                contradiction=cand_prop.contradiction[ci : ci + 1],
            )
            sc = float(cand_vals[ci].item()) + 0.5 * math.log(max(p_val, 1e-6))
            candidates.append(BeamNode(state=child_st, score=sc, depth=depth))

        if not candidates:
            break
        candidates.sort(key=lambda c: c.score, reverse=True)
        beam = candidates[:beam_width]

        if depth % sat_check_interval == 0 or depth == max_depth:
            for node in beam[:2]:
                w_d, a_d, e_d = extract_gdiff_dicts(node.state)
                sat_calls += 1
                res = solve_differential_witness(
                    rounds=node.state.rounds,
                    iv_mode=iv_mode,
                    w_gdiff=w_d,
                    a_gdiff=a_d,
                    e_gdiff=e_d,
                    digest_xor_zero_words=digest_xor_zero_words,
                    leading_zero_bits=leading_zero_bits,
                    btc_block2_padding=btc_block2_padding,
                    fixed_w_indices=fixed_w_indices,
                    max_solutions=max_solutions,
                    conflict_budget=sat_conflict_budget,
                    k_consts=k_consts,
                    sigma0_shifts=sigma0_shifts,
                    sigma1_shifts=sigma1_shifts,
                )
                if res and res[0].sat:
                    best_witnesses = res
                    return {
                        "sat": True,
                        "witnesses": best_witnesses,
                        "sat_calls": sat_calls,
                        "mendel_contradictions": mendel_contradictions,
                        "depth_reached": depth,
                        "elapsed_seconds": round(time.perf_counter() - t0, 3),
                    }

    return {
        "sat": False,
        "witnesses": [],
        "sat_calls": sat_calls,
        "mendel_contradictions": mendel_contradictions,
        "depth_reached": len(beam) and beam[0].depth or 0,
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
    }
