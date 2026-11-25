"""
Phase-SHA V10「衔尾蛇-64」轨 A 广义符号空间 `{0, 1, ?, -, u, n}` 虫洞发现环境接口（Step 3 MCTS 挂接桩）。

核心能力：
  - 维护完整 6 符号三维时空晶体状态：
      0: '0' 确定零
      1: '1' 确定一
      2: '?' 未知自由位
      3: '-' 双向差分相同但值待定
      4: 'u' 差分翻转 (1 -> 0)
      5: 'n' 差分翻转 (0 -> 1)
  - 严格锁定双端焊死边界 S[0]=IV 与 S[R]=(H-IV) mod 2^32；
  - 维护中间 Meet-in-the-Middle 拼接界面 splice_round（默认 R//2 = 32）、中性比特掩码（Neutral Bit Mask）
    与活跃模加进位门计数；
  - 无缝对接 OuroborosCore.propagate_symbols 执行双平面（GF(2) 高斯消元 + ARX 单元传播）精确坍缩。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..ouroboros_core.engine import (
    C_PLANES,
    IV_WORDS,
    MASK32,
    MAX_ROUNDS,
    SYM_0,
    SYM_1,
    SYM_DASH,
    SYM_N,
    SYM_Q,
    SYM_U,
    OuroborosCore,
    get_core,
    words_to_bits_np,
)


ALL_SYMBOLS = (SYM_0, SYM_1, SYM_Q, SYM_DASH, SYM_U, SYM_N)
SYMBOL_CHARS = {
    SYM_0: "0",
    SYM_1: "1",
    SYM_Q: "?",
    SYM_DASH: "-",
    SYM_U: "u",
    SYM_N: "n",
}


@dataclass
class SymbolStepResult:
    """单步广义符号赋值 + 双平面传播反馈。"""

    conflict: bool
    unit_prop_deductions: int
    gf2_deductions: int
    gf2_rank: int
    gf2_conflicts: int
    unknown_before: int
    unknown_after: int
    active_carry_gates: int
    splice_freedom_bits: int


@dataclass
class GeneralizedSymbolEnv:
    """
    面向轨 A 虫洞 MCTS 的 6 符号边界焊死时空晶体环境。
    """

    H_target: np.ndarray
    rounds: int = MAX_ROUNDS
    splice_round: Optional[int] = None
    core: OuroborosCore = field(default_factory=get_core)

    def __post_init__(self) -> None:
        self.H_target = np.asarray(self.H_target, dtype=np.uint32).reshape(8)
        self.rounds = int(self.rounds)
        if self.splice_round is None:
            self.splice_round = self.rounds // 2
        if not (0 < self.splice_round < self.rounds):
            raise ValueError(f"splice_round 必须位于 (0, {self.rounds})，收到 {self.splice_round}")

        self.W_sym = np.full((self.rounds, 32), SYM_Q, dtype=np.uint8)
        self.S_sym = np.full((self.rounds + 1, 8, 32), SYM_Q, dtype=np.uint8)
        self.C_sym = np.full((self.rounds, C_PLANES, 32), SYM_Q, dtype=np.uint8)

        # 中性比特掩码：记录该位点当前是否承担差分中性比特特性（供轨 A 高阶条件特征路径）
        self.neutral_mask_W = np.zeros((self.rounds, 32), dtype=bool)
        self.neutral_mask_S = np.zeros((self.rounds + 1, 8, 32), dtype=bool)

        # 精确确定化模 2^32 进位常量：C[:, :, b=0] = 0，以及前 16 轮未使用的 C[0..15, 0..2] = 0
        self.C_sym[:, :, 0] = SYM_0
        if self.rounds > 0:
            self.C_sym[: min(16, self.rounds), 0:3, :] = SYM_0

        # 硬焊双端边界：S[0] = IV，S[R] = (H_target - IV) mod 2^32
        s0_words = np.asarray(IV_WORDS, dtype=np.uint32)
        sr_words = ((self.H_target.astype(np.uint64) - s0_words.astype(np.uint64)) & MASK32).astype(np.uint32)
        self.S_sym[0] = words_to_bits_np(s0_words).astype(np.uint8)
        self.S_sym[self.rounds] = words_to_bits_np(sr_words).astype(np.uint8)

        self.last_conflict = False
        self.last_gf2_rank = 0
        self.last_unit_deductions = 0
        self.last_gf2_deductions = 0
        self.last_gf2_conflicts = 0

    def clone(self) -> "GeneralizedSymbolEnv":
        """深拷贝当前符号晶体状态供 MCTS 树展开回溯。"""
        cloned = GeneralizedSymbolEnv(
            H_target=self.H_target.copy(),
            rounds=self.rounds,
            splice_round=self.splice_round,
            core=self.core,
        )
        cloned.W_sym = self.W_sym.copy()
        cloned.S_sym = self.S_sym.copy()
        cloned.C_sym = self.C_sym.copy()
        cloned.neutral_mask_W = self.neutral_mask_W.copy()
        cloned.neutral_mask_S = self.neutral_mask_S.copy()
        cloned.last_conflict = self.last_conflict
        cloned.last_gf2_rank = self.last_gf2_rank
        cloned.last_unit_deductions = self.last_unit_deductions
        cloned.last_gf2_deductions = self.last_gf2_deductions
        cloned.last_gf2_conflicts = self.last_gf2_conflicts
        return cloned

    def _to_3sym_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """将 6 符号空间映射为 C++ 双平面核可消费的 {0, 1, ?} 三态精确约束视图。"""
        w3 = np.where((self.W_sym == SYM_0) | (self.W_sym == SYM_1), self.W_sym, SYM_Q).astype(np.uint8)
        s3 = np.where((self.S_sym == SYM_0) | (self.S_sym == SYM_1), self.S_sym, SYM_Q).astype(np.uint8)
        c3 = np.where((self.C_sym == SYM_0) | (self.C_sym == SYM_1), self.C_sym, SYM_Q).astype(np.uint8)
        return w3, s3, c3

    def _merge_propagated_3sym(
        self,
        w3_out: np.ndarray,
        s3_out: np.ndarray,
        c3_out: np.ndarray,
    ) -> None:
        """将 C++ 推导出的精确 {0, 1} 写回 6 符号阵列，同时保留未确定位上的 {-, u, n} 差分语义。"""
        w_det = (w3_out == SYM_0) | (w3_out == SYM_1)
        s_det = (s3_out == SYM_0) | (s3_out == SYM_1)
        c_det = (c3_out == SYM_0) | (c3_out == SYM_1)

        self.W_sym = np.where(w_det, w3_out, self.W_sym)
        self.S_sym = np.where(s_det, s3_out, self.S_sym)
        self.C_sym = np.where(c_det, c3_out, self.C_sym)

    def propagate(self, use_gf2: bool = True, max_iters: int = 16) -> SymbolStepResult:
        """运行双平面零退相干传播核并更新统计状态。"""
        unk_before = self.count_unknown()
        w3, s3, c3 = self._to_3sym_arrays()
        res = self.core.propagate_symbols(
            H_target=self.H_target,
            W_sym=w3,
            S_sym=s3,
            C_sym=c3,
            rounds=self.rounds,
            use_gf2=use_gf2,
            max_iters=max_iters,
        )
        if not res["conflict"]:
            self._merge_propagated_3sym(res["W_sym"], res["S_sym"], res["C_sym"])

        self.last_conflict = bool(res["conflict"])
        self.last_gf2_rank = int(res["gf2_rank"])
        self.last_unit_deductions = int(res["unit_prop_deductions"])
        self.last_gf2_deductions = int(res["gf2_deductions"])
        self.last_gf2_conflicts = int(res["gf2_conflicts"])

        return SymbolStepResult(
            conflict=self.last_conflict,
            unit_prop_deductions=self.last_unit_deductions,
            gf2_deductions=self.last_gf2_deductions,
            gf2_rank=self.last_gf2_rank,
            gf2_conflicts=self.last_gf2_conflicts,
            unknown_before=unk_before,
            unknown_after=self.count_unknown(),
            active_carry_gates=self.count_active_carry_gates(),
            splice_freedom_bits=self.splice_freedom_bits(),
        )

    def set_symbol(
        self,
        kind: int,
        t: int,
        word: int,
        bit: int,
        sym: int,
        is_neutral: bool = False,
        run_propagate: bool = True,
        use_gf2: bool = True,
    ) -> SymbolStepResult:
        """
        对单个晶体坐标设置广义符号（严格禁止覆盖焊死边界 S[0]/S[R] 与进位 b=0 常量）。
        """
        if sym not in ALL_SYMBOLS:
            raise ValueError(f"非法符号编号: {sym}，合法集合: {ALL_SYMBOLS}")
        if not (0 <= bit < 32):
            raise ValueError(f"bit 越界: {bit}")

        if kind == 0:
            if not (0 <= t < self.rounds):
                raise ValueError(f"W 轮数越界: {t}")
            self.W_sym[t, bit] = sym
            if is_neutral:
                self.neutral_mask_W[t, bit] = True
        elif kind == 1:
            if t == 0 or t == self.rounds:
                raise PermissionError(f"严禁修改硬焊死边界 S[{t}]")
            if not (1 <= t < self.rounds) or not (0 <= word < 8):
                raise ValueError(f"S 坐标越界: t={t}, word={word}")
            self.S_sym[t, word, bit] = sym
            if is_neutral:
                self.neutral_mask_S[t, word, bit] = True
        elif kind == 2:
            if not (0 <= t < self.rounds) or not (0 <= word < C_PLANES):
                raise ValueError(f"C 坐标越界: t={t}, word={word}")
            if bit == 0 or (t < 16 and word < 3):
                raise PermissionError(f"严禁修改确定进位常量位 C[{t},{word},{bit}]")
            self.C_sym[t, word, bit] = sym
        else:
            raise ValueError(f"未知坐标类型 kind={kind}（0=W, 1=S, 2=C）")

        if run_propagate:
            return self.propagate(use_gf2=use_gf2)
        unk = self.count_unknown()
        return SymbolStepResult(
            conflict=self.last_conflict,
            unit_prop_deductions=0,
            gf2_deductions=0,
            gf2_rank=self.last_gf2_rank,
            gf2_conflicts=self.last_gf2_conflicts,
            unknown_before=unk,
            unknown_after=unk,
            active_carry_gates=self.count_active_carry_gates(),
            splice_freedom_bits=self.splice_freedom_bits(),
        )

    def count_unknown(self) -> int:
        """统计尚未折叠为确定 {0, 1} 的位点总数。"""
        w_unk = int(np.sum((self.W_sym != SYM_0) & (self.W_sym != SYM_1)))
        s_unk = int(np.sum((self.S_sym != SYM_0) & (self.S_sym != SYM_1)))
        c_unk = int(np.sum((self.C_sym != SYM_0) & (self.C_sym != SYM_1)))
        return w_unk + s_unk + c_unk

    def count_active_carry_gates(self) -> int:
        """
        统计当前尚未确定进位比特的活跃模加门数量（衡量非线性进位链剩余活跃自由度）。
        """
        # 仅统计实际参与运算的有效进位门：b in 1..31，且排除 t<16 的 W 进位 C[0..15, 0..2]
        active_c = (self.C_sym[:, :, 1:] != SYM_0) & (self.C_sym[:, :, 1:] != SYM_1)
        if self.rounds > 0:
            active_c[: min(16, self.rounds), 0:3, :] = False
        return int(np.sum(active_c))

    def splice_freedom_bits(self) -> int:
        """返回中间 Meet-in-the-Middle 拼接界面 S[splice_round] 的剩余自由比特数 ∈ [0, 256]。"""
        s_mid = self.S_sym[self.splice_round]
        return int(np.sum((s_mid != SYM_0) & (s_mid != SYM_1)))

    def get_stats(self) -> Dict[str, Any]:
        """导出供轨 A MCTS 节点价值评估与可观测监控的完整符号统计字典。"""
        all_vals = np.concatenate(
            [self.W_sym.ravel(), self.S_sym.ravel(), self.C_sym.ravel()]
        )
        counts = {SYMBOL_CHARS[s]: int(np.sum(all_vals == s)) for s in ALL_SYMBOLS}
        return {
            "rounds": self.rounds,
            "splice_round": self.splice_round,
            "splice_freedom_bits": self.splice_freedom_bits(),
            "active_carry_gates": self.count_active_carry_gates(),
            "neutral_bits": int(np.sum(self.neutral_mask_W)) + int(np.sum(self.neutral_mask_S)),
            "conflict": self.last_conflict,
            "gf2_rank": self.last_gf2_rank,
            "symbol_counts": counts,
        }
