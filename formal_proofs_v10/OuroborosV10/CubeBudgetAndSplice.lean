/-!
# CubeBudgetAndSplice: Cube 盈亏不等式、`splice_block[:8]` 截断丢弃与消息调度 DAG 深度

对应 `v10_ouroboros64/ouroboros_mcts/dynamic_cdcl.py:390-436`、
`v10_ouroboros64/ouroboros_mcts/wormhole_mcts.py:312-350` 与 `crystal_core.hpp:1267-1275`：
1. Cube 剪枝期望冲突净收益充要不等式与零剪枝严格亏损；
2. `splice_block[:8]` 截断恒丢弃后置 `W[32]` 列表项；
3. `R=64` 晶体 76 倍膨胀、30,976 GF(2) 变量及 `R=26` 消息调度 5 阶 DAG 深度验证。
-/

namespace OuroborosV10.CubeBudgetAndSplice

/-- Cube 期望冲突净收益充要不等式（精确对齐 `p * saved > (1 - p) * wasted`）。 -/
theorem cube_profit_iff (p_num p_den saved wasted : Nat)
    (hDen : 0 < p_den) (hp : p_num ≤ p_den) :
    p_num * saved > (p_den - p_num) * wasted ↔
      ((p_num : Rat) / (p_den : Rat)) * (saved : Rat) >
        (1 - (p_num : Rat) / (p_den : Rat)) * (wasted : Rat) := by
  have hd : (0 : Rat) < (p_den : Rat) := Rat.natCast_lt_natCast.mpr hDen
  have hd_ne : (p_den : Rat) ≠ 0 := by grind
  have hinv : (0 : Rat) < (p_den : Rat)⁻¹ := Rat.inv_pos.mpr hd
  have hsub : ((p_den - p_num : Nat) : Rat) = (p_den : Rat) - (p_num : Rat) := by
    have h1 : (p_den - p_num) + p_num = p_den := Nat.sub_add_cancel hp
    have h2 : ((p_den - p_num + p_num : Nat) : Rat) = (p_den : Rat) :=
      congrArg (fun n : Nat => (n : Rat)) h1
    rw [Rat.natCast_add] at h2
    grind
  have hleft :
      ((p_num : Rat) / (p_den : Rat)) * (saved : Rat) =
        ((p_num * saved : Nat) : Rat) * (p_den : Rat)⁻¹ := by
    rw [Rat.natCast_mul, Rat.div_def]
    grind
  have hright :
      (1 - (p_num : Rat) / (p_den : Rat)) * (wasted : Rat) =
        (((p_den - p_num) * wasted : Nat) : Rat) * (p_den : Rat)⁻¹ := by
    rw [Rat.natCast_mul, hsub, Rat.div_def]
    have h1 : (1 - (p_num : Rat) * (p_den : Rat)⁻¹) =
        ((p_den : Rat) - (p_num : Rat)) * (p_den : Rat)⁻¹ := by
      have hmul_inv : (p_den : Rat) * (p_den : Rat)⁻¹ = 1 := Rat.mul_inv_cancel (p_den : Rat) hd_ne
      grind
    rw [h1]
    grind
  rw [hleft, hright]
  constructor
  · intro hnat
    have hrat : (((p_den - p_num) * wasted : Nat) : Rat) < ((p_num * saved : Nat) : Rat) :=
      Rat.natCast_lt_natCast.mpr hnat
    have hpos :
        0 < (((p_num * saved : Nat) : Rat) - (((p_den - p_num) * wasted : Nat) : Rat)) *
          (p_den : Rat)⁻¹ :=
      Rat.mul_pos (by grind) hinv
    grind
  · intro hrat
    rcases Nat.lt_or_ge ((p_den - p_num) * wasted) (p_num * saved) with hgt | hle
    · exact hgt
    · have hrat_le :
          ((p_num * saved : Nat) : Rat) ≤ (((p_den - p_num) * wasted : Nat) : Rat) :=
        Rat.natCast_le_natCast.mpr hle
      have hnonneg :
          0 ≤ ((((p_den - p_num) * wasted : Nat) : Rat) - ((p_num * saved : Nat) : Rat)) *
            (p_den : Rat)⁻¹ :=
        Rat.mul_nonneg (by grind) (Rat.le_of_lt hinv)
      grind

/-- 当 `p_num = 0`（剪枝失败率 100%）且浪费预算 `wasted > 0` 时，Cube 严格净亏损。 -/
theorem cube_zero_prune_strict_loss (p_den saved wasted : Nat)
    (hDen : 0 < p_den) (hW : 0 < wasted) :
    0 * saved < (p_den - 0) * wasted := by
  have hmul : 0 < p_den * wasted := Nat.mul_pos hDen hW
  simpa using hmul

/-- 零冲突成本（`wasted = 0`，如 P2 零冲突 `propagate` 探测）保证期望净收益恒非负。 -/
theorem cube_zero_cost_nonneg (p_num p_den saved : Nat)
    (hDen : 0 < p_den) (_hp : p_num ≤ p_den) :
    (1 - (p_num : Rat) / (p_den : Rat)) * (0 : Rat) ≤
      ((p_num : Rat) / (p_den : Rat)) * (saved : Rat) := by
  have hd_pos : (0 : Rat) < (p_den : Rat) := Rat.natCast_lt_natCast.mpr hDen
  have hnum : (0 : Rat) ≤ (p_num : Rat) := by grind
  have hsaved : (0 : Rat) ≤ (saved : Rat) := by grind
  have hdiv : (0 : Rat) ≤ (p_num : Rat) / (p_den : Rat) := by
    rw [Rat.div_def]
    exact Rat.mul_nonneg hnum (Rat.le_of_lt (Rat.inv_pos.mpr hd_pos))
  have hprod : (0 : Rat) ≤ ((p_num : Rat) / (p_den : Rat)) * (saved : Rat) :=
    Rat.mul_nonneg hdiv hsaved
  simpa using hprod

/-! ### `splice_block[:8]` 截断丢弃 `W[32]` 引理 (`wormhole_mcts.py:312-350`) -/

/-- 若 `s_list` 已有 8 项，则 `(s_list ++ w_list).take 8 = s_list`，后置 `w_list` 全部被截断。 -/
theorem splice_block_take8_drops_w32 {α : Type} (s_list w_list : List α)
    (hs : s_list.length = 8) :
    (s_list ++ w_list).take 8 = s_list := by
  rw [List.take_append_of_le_length (by omega)]
  exact List.take_of_length_le (by omega)

/-- 若 `w ∈ w_list` 且 `w ∉ s_list`（`s_list.length = 8`），则 `w` 必然不在截断后的 `take 8` 列表中。 -/
theorem splice_block_mem_w32_lost {α : Type} [DecidableEq α]
    (s_list w_list : List α) (hs : s_list.length = 8)
    (w : α) (_hw_in : w ∈ w_list) (hw_not_s : w ∉ s_list) :
    w ∉ (s_list ++ w_list).take 8 := by
  rw [splice_block_take8_drops_w32 s_list w_list hs]
  exact hw_not_s

/-! ### 离散维度恒等式与 `R=26` 消息调度 5 阶 DAG 深度 -/

/-- `R = 64` 时神经输入可见晶体 `64 * 19 * 32` 相对 512-bit 原像自由度的 76 倍膨胀。 -/
theorem crystal_expansion_76x : 64 * 19 * 32 = 76 * 512 := by
  rfl

/-- `R` 轮 GF(2) 全局变量数公式 `(15 * R + 8) * 32`（`crystal_core.hpp:1267-1275`）。 -/
def numGf2Vars (R : Nat) : Nat := (15 * R + 8) * 32

/-- `R = 64` 时全局 GF(2) 列数精确等于 `30976`。 -/
theorem num_gf2_vars_r64 : numGf2Vars 64 = 30976 := by
  rfl

/-- 合并两个可选依赖深度（取受未知变量影响分支的最大深度）。 -/
private def optMax : Option Nat → Option Nat → Option Nat
  | none, y => y
  | x, none => x
  | some a, some b => some (max a b)

/--
按结构燃料递归计算消息扩展依赖深度：
当 `W[0..14]` 固定而 `W[15]` 含自由未知位时，`W[t]` 的非线性依赖 DAG 深度
（`none` 表示由 `W[0..14]` 唯一确定）。
-/
def msgSchedDepthFuel : Nat → Nat → Option Nat
  | 0, _ => none
  | fuel + 1, t =>
    if t < 15 then none
    else if t = 15 then some 0
    else
      let d2 := msgSchedDepthFuel fuel (t - 2)
      let d7 := msgSchedDepthFuel fuel (t - 7)
      let d15 := msgSchedDepthFuel fuel (t - 15)
      let d16 := msgSchedDepthFuel fuel (t - 16)
      match optMax (optMax d2 d7) (optMax d15 d16) with
      | none => none
      | some d => some (d + 1)

/-- 展平的消息扩展 DAG 深度（常数值为 `0`）。 -/
def msgSchedDepth (t : Nat) : Nat :=
  (msgSchedDepthFuel (t + 1) t).getD 0

theorem dag_depth_w16_w18_w20 :
    msgSchedDepth 16 = 0 ∧ msgSchedDepth 18 = 0 ∧ msgSchedDepth 20 = 0 := by
  decide

theorem dag_depth_w17_w22 :
    msgSchedDepth 17 = 1 ∧ msgSchedDepth 22 = 1 := by
  decide

theorem dag_depth_w19_w24 :
    msgSchedDepth 19 = 2 ∧ msgSchedDepth 24 = 2 := by
  decide

theorem dag_depth_w21 : msgSchedDepth 21 = 3 := by
  decide

theorem dag_depth_w23 : msgSchedDepth 23 = 4 := by
  decide

/-- `R = 26` 顶层消息字 `W[25]` 恰好达到 5 阶非线性能量耦合链。 -/
theorem dag_depth_w25 : msgSchedDepth 25 = 5 := by
  decide

end OuroborosV10.CubeBudgetAndSplice
