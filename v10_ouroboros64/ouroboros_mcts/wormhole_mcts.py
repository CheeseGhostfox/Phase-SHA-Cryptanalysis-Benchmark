"""
Phase-SHA V10「衔尾蛇-64」Milestone 3：轨 A 广义 6 符号空间虫洞 MCTS 发现器。

融合两大核心机制：
  1. 神经引导（加载 Milestone 2 Ouroboros64Net 权重）：
     - CDCLBranchPhaseScorer 提供候选比特的坍缩期望先验与条件极性 P(x=1 | crystal)；
     - CrystalEnergyCritic 为非冲突扩展叶子节点提供残余势能价值锚点。
  2. 符号精确双平面仲裁（GeneralizedSymbolEnv + C++17 ARX + GF(2) RREF）：
     - 每次扩展严格执行双平面精确闭包 propagate(use_gf2=True)；
     - 一旦分支在双平面上触发数学矛盾 (conflict=True)，立即将其精确剪枝，
       并等价导出该分支补元（root 层可证明的强制单子句 x != v -> x = 1-v）。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..ouroboros_core.engine import (
    C_PLANES,
    IV_WORDS,
    MASK32,
    MAX_ROUNDS,
    SYM_0,
    SYM_1,
    SYM_DASH,
    SYM_Q,
    OuroborosCore,
    TorchCrystalEngine,
    bits_to_words_np,
    get_core,
)
from ..ouroboros_net.heads import PLANE_COUNT
from ..ouroboros_net.model import Ouroboros64Net
from .symbol_env import GeneralizedSymbolEnv, SymbolStepResult


@dataclass
class MCTSAction:
    """单个符号空间分支动作或连贯多比特宏动作（坐标 + 目标赋值）。"""

    kind: int
    t: int
    word: int
    bit: int
    sym: int
    prior: float
    collapse_score: float = 0.0
    is_neutral: bool = False
    extra_assignments: Tuple[Tuple[int, int, int, int, int], ...] = ()

    @property
    def is_macro(self) -> bool:
        return len(self.extra_assignments) > 1

    def action_key(self) -> Tuple[Any, ...]:
        if self.extra_assignments:
            return self.extra_assignments
        return (self.kind, self.t, self.word, self.bit, self.sym)


class MCTSNode:
    """虫洞 MCTS 搜索树节点。"""

    def __init__(
        self,
        env: GeneralizedSymbolEnv,
        parent: Optional["MCTSNode"] = None,
        action: Optional[MCTSAction] = None,
        prior: float = 1.0,
    ) -> None:
        self.env = env
        self.parent = parent
        self.action = action
        self.prior = float(prior)
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: Dict[Tuple[Any, ...], "MCTSNode"] = {}
        self.untried_actions: List[MCTSAction] = []
        self.expanded: bool = False
        self.is_conflict: bool = bool(env.last_conflict)
        self.is_solved: bool = False
        self.unknown_count: int = env.count_unknown()
        self.active_carry_gates: int = env.count_active_carry_gates()
        self.splice_freedom: int = env.splice_freedom_bits()

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / float(self.visit_count)

    def ucb_score(self, c_puct: float = 1.35) -> float:
        parent_n = self.parent.visit_count if self.parent is not None else 1
        scale_u = max(1.0, abs(self.parent.q_value)) if self.parent is not None else 1.0
        u = c_puct * scale_u * self.prior * math.sqrt(max(1, parent_n)) / (1.0 + self.visit_count)
        # 对已证明数学矛盾的死胡同施加极大负分，彻底抑制重复访问
        if self.is_conflict:
            return -1e6
        return self.q_value + u


@dataclass
class MCTSSearchResult:
    """MCTS 虫洞搜索输出汇总（严禁任何内部真值泄漏）。"""

    rounds: int
    solved: bool
    w0_15: Optional[np.ndarray]
    simulations: int
    expansions: int
    pruned_conflicts: int
    root_unknown: int
    min_unknown_reached: int
    root_active_carry_gates: int
    min_active_carry_gates: int
    root_splice_freedom: int
    min_splice_freedom: int
    learned_units: List[Tuple[int, int, int, int, int]]
    phase_preferences: Dict[Tuple[int, int], Tuple[float, int]]
    elapsed_sec: float
    best_env_stats: Dict[str, Any] = field(default_factory=dict)
    learned_clauses: List[List[Tuple[int, int, int, int, int]]] = field(default_factory=list)
    root_visit_policy: List[Tuple[Tuple[int, int, int, int, int], float]] = field(default_factory=list)
    root_value_target: float = 0.0


class WormholeMCTS:
    """
    轨 A 神经-符号双平面虫洞 MCTS 搜索引擎。
    """

    def __init__(
        self,
        model: Optional[Ouroboros64Net] = None,
        torch_eng: Optional[TorchCrystalEngine] = None,
        core: Optional[OuroborosCore] = None,
        device: Optional[torch.device | str] = None,
        c_puct: float = 1.35,
        max_branch_candidates: int = 12,
        prop_max_iters: int = 8,
        use_gf2_root: Optional[bool] = None,
        gf2_interval: int = 4,
        enable_macro_actions: bool = True,
    ) -> None:
        self.core = core or get_core()
        self.model = model
        self.device = (
            torch.device(device)
            if device is not None
            else (next(model.parameters()).device if model is not None else torch.device("cpu"))
        )
        self.torch_eng = torch_eng or (TorchCrystalEngine(device=self.device) if model is not None else None)
        self.c_puct = float(c_puct)
        self.max_branch_candidates = int(max_branch_candidates)
        self.prop_max_iters = int(prop_max_iters)
        self.use_gf2_root = use_gf2_root
        self.gf2_interval = max(1, int(gf2_interval))
        self.enable_macro_actions = bool(enable_macro_actions)

    @torch.no_grad()
    def _neural_score_env(
        self,
        env: GeneralizedSymbolEnv,
        conflict_heatmap: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        将当前部分符号晶体与动态冲突热力图送入 Milestone 2 Ouroboros64Net 既有通道，返回：
          collapse_w: (16, 32) 预期坍缩得分
          phase_w:    (16, 32) 相位 log-odds P(bit=1)
          pred_le:    float 模型预测的残余对数势能 log1p(E)
        若未加载模型则退化为纯粹基于晶体边界距离的几何先验。
        """
        if self.model is None or self.torch_eng is None:
            # 无模型模式：越靠近 S[0]=IV (W[0..3]) 先验越强，鼓励由浅入深联动坍缩
            coll = np.zeros((16, 32), dtype=np.float32)
            for wi in range(16):
                coll[wi, :] = 1.0 / (1.0 + 0.15 * wi)
            phase = np.zeros((16, 32), dtype=np.float32)
            return coll, phase, 8.0

        was_train = self.model.training
        self.model.eval()
        r = env.rounds

        W_known = np.where((env.W_sym == SYM_0) | (env.W_sym == SYM_1), env.W_sym == SYM_1, 0).astype(np.uint8)
        S_known = np.where((env.S_sym[:r] == SYM_0) | (env.S_sym[:r] == SYM_1), env.S_sym[:r] == SYM_1, 0).astype(np.uint8)
        C_known = np.where((env.C_sym == SYM_0) | (env.C_sym == SYM_1), env.C_sym == SYM_1, 0).astype(np.uint8)

        S_words = np.zeros((1, r + 1, 8), dtype=np.uint32)
        S_words[0, :r] = bits_to_words_np(S_known)
        S_words[0, 0] = np.asarray(IV_WORDS, dtype=np.uint32)
        sr_words = ((env.H_target.astype(np.uint64) - S_words[0, 0].astype(np.uint64)) & MASK32).astype(np.uint32)
        S_words[0, r] = sr_words

        W_words = bits_to_words_np(W_known)[None, :]
        C_words = bits_to_words_np(C_known)[None, :]

        unk_np = np.ones((1, r, PLANE_COUNT, 32), dtype=np.float32)
        unk_np[0, :, 0, :] = ((env.W_sym != SYM_0) & (env.W_sym != SYM_1)).astype(np.float32)
        unk_np[0, :, 1:9, :] = ((env.S_sym[:r] != SYM_0) & (env.S_sym[:r] != SYM_1)).astype(np.float32)
        unk_np[0, :, 9:19, :] = ((env.C_sym != SYM_0) & (env.C_sym != SYM_1)).astype(np.float32)
        unk_np[0, 0, 1:9, :] = 0.0

        conf_t: Optional[torch.Tensor] = None
        if conflict_heatmap is not None:
            conf_arr = np.log1p(np.maximum(0.0, conflict_heatmap[:r])).astype(np.float32)
            conf_t = torch.from_numpy(conf_arr[None, ...]).to(self.device)

        out = self.model(
            torch.from_numpy(env.H_target[None, :].astype(np.int64)).to(self.device),
            torch.from_numpy(W_words.astype(np.int64)).to(self.device),
            torch.from_numpy(S_words.astype(np.int64)).to(self.device),
            torch.from_numpy(C_words.astype(np.int64)).to(self.device),
            rounds=r,
            unknown_mask=torch.from_numpy(unk_np).to(self.device),
            conflict_heatmap=conf_t,
            zero_defects=True,
            propose_clusters=False,
            return_entropy=False,
            crystal_engine=self.torch_eng,
        )
        coll_w = out["collapse_score"][0, :16, 0, :].cpu().numpy()
        phase_w = out["phase_logits"][0, :16, 0, :].cpu().numpy()
        pred_le = float(out["pred_log_energy"][0].item())
        if was_train:
            self.model.train()
        return coll_w, phase_w, pred_le

    @staticmethod
    def _record_conflict_heat(
        conflict_heatmap: np.ndarray,
        act: MCTSAction,
        rounds: int,
    ) -> None:
        """将触发双平面矛盾的坐标及其相邻进位与 ARX 滞后耦合热力动态回写至输入通道。"""
        plane = 0 if act.kind == 0 else (1 + act.word if act.kind == 1 else 9 + act.word)
        if not (0 <= act.t < rounds and 0 <= plane < PLANE_COUNT and 0 <= act.bit < 32):
            return
        conflict_heatmap[act.t, plane, act.bit] += 1.0
        for db in (-1, 1):
            nb = act.bit + db
            if 0 <= nb < 32:
                conflict_heatmap[act.t, plane, nb] += 0.35
        for lag in (1, 2, 7, 15, 16):
            for dt in (-lag, lag):
                nt = act.t + dt
                if 0 <= nt < rounds:
                    conflict_heatmap[nt, plane, act.bit] += 0.25

    def _build_candidate_actions(
        self,
        env: GeneralizedSymbolEnv,
        focus_splice: bool = False,
        conflict_heatmap: Optional[np.ndarray] = None,
    ) -> Tuple[List[MCTSAction], np.ndarray, np.ndarray, float]:
        """从当前未知符号变量提取 Top-K 高潜力分支动作列表与神经评估。"""
        coll_w, phase_w, pred_le = self._neural_score_env(env, conflict_heatmap=conflict_heatmap)
        unk_w = np.argwhere((env.W_sym[:16] != SYM_0) & (env.W_sym[:16] != SYM_1))

        scored_vars: List[Tuple[float, int, int, float, float]] = []
        for row in unk_w:
            wi, bb = int(row[0]), int(row[1])
            c_sc = float(coll_w[wi, bb])
            ph = float(phase_w[wi, bb])
            # 鼓励选择坍缩期望高、相位置信度高，且在低比特位处易于激发进位传播的自由变量
            total_sc = c_sc + 0.35 * abs(ph) + (0.15 if wi < 4 else 0.0)
            scored_vars.append((total_sc, wi, bb, c_sc, ph))

        # 若用于轨 A R=64 虫洞探测且 W 已高占比锁定，补充 S[splice_round] 中间拼接层未知变量
        splice_vars: List[Tuple[float, int, int, int]] = []
        if focus_splice:
            sr = env.splice_round
            unk_s = np.argwhere((env.S_sym[sr] != SYM_0) & (env.S_sym[sr] != SYM_1))
            for row in unk_s[:8]:
                ww, bb = int(row[0]), int(row[1])
                splice_vars.append((0.5, sr, ww, bb))

        scored_vars.sort(key=lambda x: x[0], reverse=True)
        top_k = max(1, self.max_branch_candidates // 2)
        chosen_w = scored_vars[:top_k]

        actions: List[MCTSAction] = []
        raw_logits: List[float] = []
        for sc, wi, bb, c_sc, ph in chosen_w:
            p1 = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, ph))))
            # 按首选相位先入队，同时保留备选反向分支实现完整二进制搜索树
            first_sym, second_sym = (SYM_1, SYM_0) if p1 >= 0.5 else (SYM_0, SYM_1)
            p_first = max(0.51, p1 if first_sym == SYM_1 else (1.0 - p1))
            p_second = max(0.05, 1.0 - p_first)
            actions.append(MCTSAction(0, wi, 0, bb, first_sym, p_first, c_sc))
            raw_logits.append(sc + math.log(p_first + 1e-4))
            actions.append(MCTSAction(0, wi, 0, bb, second_sym, p_second, c_sc))
            raw_logits.append(sc + math.log(p_second + 1e-4))

        for sc_s, sr, ww, bb in splice_vars[:4]:
            actions.append(MCTSAction(1, sr, ww, bb, SYM_1, 0.5, sc_s))
            raw_logits.append(sc_s)
            actions.append(MCTSAction(1, sr, ww, bb, SYM_0, 0.5, sc_s))
            raw_logits.append(sc_s)

        # 4-bit 连续 LSB 进位链宏动作：跨过 R>=32 或中间拼接层单比特传播饥饿壁垒
        if self.enable_macro_actions and (focus_splice or env.rounds >= 32):
            macro_added = 0
            # 优先针对中间拼接界面 S[splice_round] 的 a/e/d/h 关键寄存器连续 4-bit LSB 块
            if focus_splice:
                sr = env.splice_round
                splice_block: List[Tuple[int, int, int, int, int]] = []
                splice_block_alt: List[Tuple[int, int, int, int, int]] = []
                for ww in (0, 4, 3, 7):
                    unk_bits = [
                        b for b in range(32)
                        if int(env.S_sym[sr, ww, b]) not in (SYM_0, SYM_1)
                    ]
                    for b in unk_bits[:2]:
                        splice_block.append((1, sr, ww, b, SYM_0))
                        splice_block_alt.append((1, sr, ww, b, SYM_1))
                # 同步带上 W[splice_round] 连续未知低位形成真正跨字进位耦合块
                unk_wsr = [
                    b for b in range(32)
                    if int(env.W_sym[sr, b]) not in (SYM_0, SYM_1)
                ]
                for b in unk_wsr[:4]:
                    splice_block.append((0, sr, 0, b, SYM_0))
                    splice_block_alt.append((0, sr, 0, b, SYM_1))
                if len(splice_block) >= 4:
                    k0, t0_s, w0_s, b0_s, sym0_s = splice_block[0]
                    actions.insert(
                        0,
                        MCTSAction(
                            k0, t0_s, w0_s, b0_s, sym0_s, 0.7, 2.5,
                            extra_assignments=tuple(splice_block[:8]),
                        ),
                    )
                    raw_logits.insert(0, 2.5)
                    k1, t1_s, w1_s, b1_s, sym1_s = splice_block_alt[0]
                    actions.insert(
                        1,
                        MCTSAction(
                            k1, t1_s, w1_s, b1_s, sym1_s, 0.3, 1.8,
                            extra_assignments=tuple(splice_block_alt[:8]),
                        ),
                    )
                    raw_logits.insert(1, 1.8)
                    macro_added += 1

            # 为自由 W 字中最高评分未知字构建连续 4-bit LSB 宏动作
            w_by_word: Dict[int, List[Tuple[float, int, float]]] = {}
            for sc, wi, bb, _c_sc, ph in scored_vars:
                w_by_word.setdefault(wi, []).append((sc, bb, ph))
            best_wi = max(
                (wi for wi, items in w_by_word.items() if len(items) >= 4),
                key=lambda wi: sum(x[0] for x in sorted(w_by_word[wi], reverse=True)[:4]),
                default=-1,
            )
            if best_wi >= 0 and macro_added < 2:
                blk_items = sorted(w_by_word[best_wi], key=lambda x: x[1])[:4]
                w_blk = tuple(
                    (0, best_wi, 0, bb, SYM_1 if ph >= 0.0 else SYM_0)
                    for _, bb, ph in blk_items
                )
                w_blk_opp = tuple(
                    (0, best_wi, 0, bb, SYM_0 if ph >= 0.0 else SYM_1)
                    for _, bb, ph in blk_items
                )
                avg_sc = float(sum(x[0] for x in blk_items) / 4.0) + 1.2
                actions.append(
                    MCTSAction(
                        0, best_wi, 0, blk_items[0][1], w_blk[0][4], 0.65, avg_sc,
                        extra_assignments=w_blk,
                    )
                )
                raw_logits.append(avg_sc)
                actions.append(
                    MCTSAction(
                        0, best_wi, 0, blk_items[0][1], w_blk_opp[0][4], 0.35, avg_sc - 0.5,
                        extra_assignments=w_blk_opp,
                    )
                )
                raw_logits.append(avg_sc - 0.5)

        if raw_logits:
            mx = max(raw_logits)
            exps = [math.exp(max(-20.0, v - mx)) for v in raw_logits]
            s_exp = sum(exps) + 1e-12
            for act, ex in zip(actions, exps):
                act.prior = float(ex / s_exp)

        return actions, coll_w, phase_w, pred_le

    def _apply_action_to_env(
        self,
        env: GeneralizedSymbolEnv,
        act: MCTSAction,
        use_gf2: bool,
    ) -> SymbolStepResult:
        """统一执行单比特动作或多比特连贯宏动作并触发指定平面闭包传播。"""
        assigns = (
            act.extra_assignments
            if act.extra_assignments
            else ((act.kind, act.t, act.word, act.bit, act.sym),)
        )
        for k, t_i, w_i, b_i, s_i in assigns:
            env.set_symbol(
                int(k), int(t_i), int(w_i), int(b_i), int(s_i),
                is_neutral=act.is_neutral,
                run_propagate=False,
                use_gf2=use_gf2,
            )
        return env.propagate(use_gf2=use_gf2, max_iters=self.prop_max_iters)

    def _try_extract_preimage(self, env: GeneralizedSymbolEnv) -> Optional[np.ndarray]:
        """当 W[0..15] 全部确定为 {0,1} 时，立即构造原像并用 verify_preimage 严格核验。"""
        w16 = env.W_sym[:16]
        if np.all((w16 == SYM_0) | (w16 == SYM_1)):
            w_bits = (w16 == SYM_1).astype(np.uint8)
            w_words = bits_to_words_np(w_bits)
            ok = bool(self.core.verify_preimage(w_words, env.H_target, rounds=env.rounds)[0])
            if ok:
                return w_words
        return None

    def search(
        self,
        H_target: np.ndarray,
        rounds: int = 20,
        fixed_words: Optional[Dict[int, int]] = None,
        num_simulations: int = 48,
        time_limit_sec: float = 3.0,
        focus_splice: bool = False,
        fixed_bits: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> MCTSSearchResult:
        """
        执行双端焊死 + 双速精确传播 + 神经价值引导的虫洞 MCTS。
        """
        t0 = time.perf_counter()
        env_root = GeneralizedSymbolEnv(
            H_target=H_target,
            rounds=rounds,
            splice_round=max(1, rounds // 2),
            core=self.core,
        )
        # 注入指定已知固定整字与逐比特固定位（绝对不包含任何自由目标比特）
        fixed_bit_set = set()
        if fixed_words:
            for wi, word_val in fixed_words.items():
                u_val = int(word_val) & 0xFFFFFFFF
                for bb in range(32):
                    bit_sym = SYM_1 if ((u_val >> bb) & 1) else SYM_0
                    env_root.set_symbol(0, int(wi), 0, bb, bit_sym, run_propagate=False)
                    fixed_bit_set.add((int(wi), bb))
        if fixed_bits:
            for (wi, bb), bit_val in fixed_bits.items():
                w_i, b_i = int(wi), int(bb)
                bit_sym = SYM_1 if (int(bit_val) & 1) else SYM_0
                env_root.set_symbol(0, w_i, 0, b_i, bit_sym, run_propagate=False)
                fixed_bit_set.add((w_i, b_i))

        # 双速策略：R<=28 根节点执行满血 GF(2)+ARX 闭包；R>=32 首轮使用毫秒级 ARX 闭包保留树搜索预算
        use_gf2_root = (rounds <= 28) if self.use_gf2_root is None else bool(self.use_gf2_root)
        env_root.propagate(use_gf2=use_gf2_root, max_iters=self.prop_max_iters)

        # 收集所有在根节点即通过双平面推导锁死的精确 W[0..15] 单子句
        learned_units: List[Tuple[int, int, int, int, int]] = []
        seen_units = set()
        learned_clauses: List[List[Tuple[int, int, int, int, int]]] = []
        seen_clauses = set()
        for wi in range(16):
            for bb in range(32):
                if (wi, bb) in fixed_bit_set:
                    continue
                s_val = int(env_root.W_sym[wi, bb])
                if s_val in (SYM_0, SYM_1):
                    key = (0, wi, 0, bb, s_val)
                    if key not in seen_units:
                        seen_units.add(key)
                        learned_units.append(key)

        root = MCTSNode(env=env_root, prior=1.0)
        root_splice = root.splice_freedom

        best_env = env_root
        min_unk = root.unknown_count
        min_gates = root.active_carry_gates
        min_splice = root.splice_freedom

        solved_w: Optional[np.ndarray] = None
        if not root.is_conflict:
            solved_w = self._try_extract_preimage(env_root)
            if solved_w is not None:
                root.is_solved = True

        # 初始化动态冲突热力图（回灌至现有 conflict_heatmap 通道）与无上限叶评估基准
        conflict_heatmap = np.zeros((rounds, PLANE_COUNT, 32), dtype=np.float32)

        actions_r, coll_r, phase_r, root_pred_le = self._build_candidate_actions(
            env_root, focus_splice=focus_splice, conflict_heatmap=conflict_heatmap
        )
        root.untried_actions = actions_r
        phase_preferences: Dict[Tuple[int, int], Tuple[float, int]] = {}
        for wi in range(16):
            for bb in range(32):
                if (wi, bb) in fixed_bit_set:
                    continue
                logit = float(phase_r[wi, bb])
                conf = float(coll_r[wi, bb]) * abs(logit)
                pref_bit = 1 if logit >= 0.0 else 0
                phase_preferences[(wi, bb)] = (conf, pref_bit)

        sims = 0
        expansions = 0
        pruned_conflicts = 0

        if solved_w is None and not root.is_conflict:
            for _ in range(num_simulations):
                if time_limit_sec > 0.0 and (time.perf_counter() - t0) >= time_limit_sec:
                    break
                sims += 1
                node = root
                path = [node]

                # 1. Selection
                while node.expanded and not node.untried_actions and node.children and not node.is_conflict and not node.is_solved:
                    best_child = max(node.children.values(), key=lambda ch: ch.ucb_score(self.c_puct))
                    if best_child.is_conflict:
                        break
                    node = best_child
                    path.append(node)

                # 2. Expansion + 双速双平面闭包
                leaf_val = 0.0
                if not node.is_conflict and not node.is_solved:
                    if not node.expanded and not node.untried_actions:
                        acts, _, _, _ = self._build_candidate_actions(
                            node.env, focus_splice=focus_splice, conflict_heatmap=conflict_heatmap
                        )
                        node.untried_actions = acts
                        node.expanded = True

                    if node.untried_actions:
                        act = node.untried_actions.pop(0)
                        node.expanded = True
                        child_env = node.env.clone()
                        depth = len(path)
                        try:
                            step_res = self._apply_action_to_env(child_env, act, use_gf2=False)
                            rem_w = int(np.sum((child_env.W_sym[:16] != SYM_0) & (child_env.W_sym[:16] != SYM_1)))
                            if (
                                (not step_res.conflict)
                                and (rounds <= 28)
                                and (
                                    (depth % self.gf2_interval == 0)
                                    or (step_res.unit_prop_deductions >= 12)
                                    or (rem_w <= 8)
                                )
                            ):
                                step_res = child_env.propagate(use_gf2=True, max_iters=4)
                        except PermissionError:
                            step_res = SymbolStepResult(
                                conflict=True, unit_prop_deductions=0, gf2_deductions=0,
                                gf2_rank=0, gf2_conflicts=1, unknown_before=node.unknown_count,
                                unknown_after=node.unknown_count, active_carry_gates=node.active_carry_gates,
                                splice_freedom_bits=node.splice_freedom,
                            )

                        child = MCTSNode(env=child_env, parent=node, action=act, prior=act.prior)
                        child.is_conflict = bool(step_res.conflict)
                        node.children[act.action_key()] = child
                        path.append(child)
                        expansions += 1

                        if child.is_conflict:
                            pruned_conflicts += 1
                            leaf_val = -10.0
                            self._record_conflict_heat(conflict_heatmap, act, rounds)
                            # 仅对单比特分支导出严密单子句或深度-2 二元冲突子句
                            if node is root and (not act.is_macro) and act.kind == 0 and act.sym in (SYM_0, SYM_1):
                                opp = SYM_1 if act.sym == SYM_0 else SYM_0
                                k_opp = (act.kind, act.t, act.word, act.bit, opp)
                                if k_opp not in seen_units:
                                    seen_units.add(k_opp)
                                    learned_units.append(k_opp)
                            elif (
                                len(path) == 3
                                and node.action is not None
                                and (not node.action.is_macro)
                                and (not act.is_macro)
                                and node.action.kind == 0
                                and act.kind == 0
                                and node.action.sym in (SYM_0, SYM_1)
                                and act.sym in (SYM_0, SYM_1)
                            ):
                                opp1 = SYM_1 if node.action.sym == SYM_0 else SYM_0
                                opp2 = SYM_1 if act.sym == SYM_0 else SYM_0
                                cl = [
                                    (0, node.action.t, 0, node.action.bit, opp1),
                                    (0, act.t, 0, act.bit, opp2),
                                ]
                                ck = tuple(sorted(cl))
                                if ck not in seen_clauses:
                                    seen_clauses.add(ck)
                                    learned_clauses.append(cl)
                        else:
                            if child.unknown_count < min_unk:
                                min_unk = child.unknown_count
                                best_env = child_env
                            if child.active_carry_gates < min_gates:
                                min_gates = child.active_carry_gates
                            if child.splice_freedom < min_splice:
                                min_splice = child.splice_freedom

                            # 无硬封顶平滑价值：对数自由度坍缩 + 活跃门减少 + 拼接界压缩 + 神经能量压降
                            v_free = math.log1p(max(0, root.unknown_count - child.unknown_count))
                            v_gate = math.log1p(max(0, root.active_carry_gates - child.active_carry_gates))
                            v_splice = math.log1p(max(0, root_splice - child.splice_freedom))
                            _, _, child_pred_le = self._neural_score_env(
                                child_env, conflict_heatmap=conflict_heatmap
                            )
                            v_energy = math.log1p(max(0.0, root_pred_le - child_pred_le))
                            uncapped_val = 0.45 * v_free + 0.35 * v_gate + 0.25 * v_splice + 0.20 * v_energy

                            cand_w = self._try_extract_preimage(child_env)
                            if cand_w is not None:
                                child.is_solved = True
                                solved_w = cand_w
                                leaf_val = 50.0 + uncapped_val
                            else:
                                leaf_val = uncapped_val

                # 3. Backpropagation
                for nd in path:
                    nd.visit_count += 1
                    nd.value_sum += leaf_val

                if solved_w is not None:
                    break

        # 若发生扩展，用最优节点环境 + 动态冲突热力图刷新下游 CDCL 相位，并融合根 MCTS (Q_1 - Q_0) 偏置
        if expansions > 0 and solved_w is None:
            coll_b, phase_b, _ = self._neural_score_env(best_env, conflict_heatmap=conflict_heatmap)
            q_bias: Dict[Tuple[int, int], List[float]] = {}
            for ch in root.children.values():
                if ch.action is not None and (not ch.action.is_macro) and ch.action.kind == 0 and ch.visit_count >= 2:
                    coord = (ch.action.t, ch.action.bit)
                    pair_q = q_bias.setdefault(coord, [0.0, 0.0])
                    if ch.action.sym == SYM_1:
                        pair_q[1] = max(pair_q[1], ch.q_value)
                    else:
                        pair_q[0] = max(pair_q[0], ch.q_value)
            for (wi, bb) in list(phase_preferences.keys()):
                logit = float(phase_b[wi, bb])
                if (wi, bb) in q_bias:
                    logit += 0.35 * (q_bias[(wi, bb)][1] - q_bias[(wi, bb)][0])
                conf = float(coll_b[wi, bb]) * abs(logit)
                phase_preferences[(wi, bb)] = (conf, 1 if logit >= 0.0 else 0)

        # 导出根节点访问策略分布与最大子价值锚点，供 Milestone 3 策略/价值蒸馏消费
        total_root_visits = sum(ch.visit_count for ch in root.children.values() if ch.visit_count > 0)
        root_visit_policy: List[Tuple[Tuple[int, int, int, int, int], float]] = []
        if total_root_visits > 0:
            for ch in root.children.values():
                if ch.action is not None and (not ch.action.is_macro) and ch.visit_count > 0:
                    root_visit_policy.append(
                        (
                            (ch.action.kind, ch.action.t, ch.action.word, ch.action.bit, ch.action.sym),
                            float(ch.visit_count / total_root_visits),
                        )
                    )
        root_value_target = max(
            (ch.q_value for ch in root.children.values() if ch.visit_count > 0 and not ch.is_conflict),
            default=0.0,
        )

        return MCTSSearchResult(
            rounds=rounds,
            solved=(solved_w is not None),
            w0_15=solved_w,
            simulations=sims,
            expansions=expansions,
            pruned_conflicts=pruned_conflicts,
            root_unknown=root.unknown_count,
            min_unknown_reached=min_unk,
            root_active_carry_gates=root.active_carry_gates,
            min_active_carry_gates=min_gates,
            root_splice_freedom=root_splice,
            min_splice_freedom=min_splice,
            learned_units=learned_units,
            phase_preferences=phase_preferences,
            elapsed_sec=time.perf_counter() - t0,
            best_env_stats=best_env.get_stats(),
            learned_clauses=learned_clauses,
            root_visit_policy=root_visit_policy,
            root_value_target=float(root_value_target),
        )
