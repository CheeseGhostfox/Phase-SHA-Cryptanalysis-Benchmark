import PhaseSHA.Core.BitVecIdentities

/-!
# Module 3: V3 动态数字锁存的无损性与开环无界续航定理

本模块在纯 Lean 4 标准库下对 Phase-SHA V3 的动态数字锁存（Dynamic Digital Latch）
与半径 `≤ 2` 局域比特翻转校验进行严格形式化：
1. 将一般 ARX 链及真实 SHA-256 单轮转移封装为显式双射算子 `BijectiveStep`；
2. **`exact_latch_preserves_solution_set`**：证明当边界离散状态满足零物理残差时，
   将其固化为新边界后的剩余开环子链与原链解集完全同构，且该边界传播误差严格归零；
3. **`unbounded_open_loop_propagation`**：构造可计算半径 `≤ 2` 局域纠错锁存算子
   `latchStepRadius2`，并通过对自由轮数 `L : Nat` 的数学归纳法证明：只要每轮预测误差
   落入纠错半径内，动态数字锁存即可将每步误差硬重置归零，实现任意深度 `L` 的无损续航。
-/

namespace PhaseSHA.Core.LatchingSoundness

open PhaseSHA.Core.BitVecIdentities

/-!
### 1. 显式双射算子与 SHA-256 单轮可逆性实例化
-/

/-- 显式双射单步转移：同时携带前向算子、精确逆算子以及左右互逆证明。 -/
structure BijectiveStep (α : Type) where
  fwd : α → α
  inv : α → α
  left_inv : ∀ x, inv (fwd x) = x
  right_inv : ∀ y, fwd (inv y) = y

theorem BijectiveStep.fwd_inj {α : Type} (step : BijectiveStep α) {x y : α}
    (h : step.fwd x = step.fwd y) : x = y := by
  have h1 := congrArg step.inv h
  rw [step.left_inv, step.left_inv] at h1
  exact h1

theorem BijectiveStep.fwd_eq_iff {α : Type} (step : BijectiveStep α) (x y : α) :
    step.fwd x = y ↔ x = step.inv y := by
  constructor
  · intro h
    have h1 := congrArg step.inv h
    rw [step.left_inv] at h1
    exact h1
  · intro h
    rw [h, step.right_inv]

/-- 真实 SHA-256 单轮状态的精确解析逆转移。 -/
@[inline] def sha256RoundInv (s' : SHAState) (k w : BitVec 32) : SHAState :=
  let a := s'.b
  let b := s'.c
  let c := s'.d
  let e := s'.f
  let f := s'.g
  let g := s'.h
  let t2 := Sigma0 a + maj32 a b c
  let t1 := s'.a - t2
  let d := s'.e - t1
  let h := t1 - (Sigma1 e + ch32 e f g + k + w)
  { a := a, b := b, c := c, d := d, e := e, f := f, g := g, h := h }

private theorem bv_recover_d (d t1 t2 : BitVec 32) :
    (d + t1) - ((t1 + t2) - t2) = d := by
  bv_decide

private theorem bv_recover_h (h x1 x2 x3 x4 t2 : BitVec 32) :
    (((h + x1 + x2 + x3 + x4) + t2) - t2) - (x1 + x2 + x3 + x4) = h := by
  bv_decide

private theorem bv_reconstruct_a (a t2 x1 x2 x3 x4 : BitVec 32) :
    (((a - t2) - (x1 + x2 + x3 + x4)) + x1 + x2 + x3 + x4) + t2 = a := by
  bv_decide

private theorem bv_reconstruct_e (e a t2 x1 x2 x3 x4 : BitVec 32) :
    (e - (a - t2)) + (((a - t2) - (x1 + x2 + x3 + x4)) + x1 + x2 + x3 + x4) = e := by
  bv_decide

theorem sha256RoundInv_round (s : SHAState) (k w : BitVec 32) :
    sha256RoundInv (sha256Round s k w) k w = s := by
  cases s with
  | mk a b c d e f g h =>
    simp only [sha256Round, sha256RoundInv]
    congr 1
    · exact bv_recover_d d
        (h + Sigma1 e + ch32 e f g + k + w)
        (Sigma0 a + maj32 a b c)
    · exact bv_recover_h h (Sigma1 e) (ch32 e f g) k w
        (Sigma0 a + maj32 a b c)

theorem sha256Round_inv (s' : SHAState) (k w : BitVec 32) :
    sha256Round (sha256RoundInv s' k w) k w = s' := by
  cases s' with
  | mk a b c d e f g h =>
    simp only [sha256RoundInv, sha256Round]
    congr 1
    · exact bv_reconstruct_a a
        (Sigma0 b + maj32 b c d)
        (Sigma1 f) (ch32 f g h) k w
    · exact bv_reconstruct_e e a
        (Sigma0 b + maj32 b c d)
        (Sigma1 f) (ch32 f g h) k w

/-- SHA-256 任意固定轮常量 `k` 与消息字 `w` 构成的显式双射单步。 -/
def sha256RoundBijective (k w : BitVec 32) : BijectiveStep SHAState where
  fwd s := sha256Round s k w
  inv s' := sha256RoundInv s' k w
  left_inv s := sha256RoundInv_round s k w
  right_inv s' := sha256Round_inv s' k w

/-!
### 2. 开环 ARX 双射链与精确锁存解集不变性 (`exact_latch_preserves_solution_set`)
-/

/-- 前向执行双射步序列。 -/
def evalChain {α : Type} : List (BijectiveStep α) → α → α
  | [], x => x
  | step :: rest, x => evalChain rest (step.fwd x)

/-- 逆向解析倒推双射步序列。 -/
def evalChainInv {α : Type} : List (BijectiveStep α) → α → α
  | [], y => y
  | step :: rest, y => step.inv (evalChainInv rest y)

theorem evalChain_append {α : Type} (l1 l2 : List (BijectiveStep α)) (x : α) :
    evalChain (l1 ++ l2) x = evalChain l2 (evalChain l1 x) := by
  induction l1 generalizing x with
  | nil => rfl
  | cons step rest ih =>
    simp only [List.cons_append, evalChain]
    exact ih (step.fwd x)

theorem evalChainInv_evalChain {α : Type} (steps : List (BijectiveStep α)) (x : α) :
    evalChainInv steps (evalChain steps x) = x := by
  induction steps generalizing x with
  | nil => rfl
  | cons step rest ih =>
    simp only [evalChain, evalChainInv]
    rw [ih (step.fwd x), step.left_inv]

theorem evalChain_evalChainInv {α : Type} (steps : List (BijectiveStep α)) (y : α) :
    evalChain steps (evalChainInv steps y) = y := by
  induction steps generalizing y with
  | nil => rfl
  | cons step rest ih =>
    simp only [evalChainInv, evalChain]
    rw [step.right_inv, ih y]

theorem evalChain_eq_iff {α : Type} (steps : List (BijectiveStep α)) (x y : α) :
    evalChain steps x = y ↔ x = evalChainInv steps y := by
  constructor
  · intro h
    have h1 := congrArg (evalChainInv steps) h
    rw [evalChainInv_evalChain] at h1
    exact h1
  · intro h
    rw [h, evalChain_evalChainInv]

/-- 一般开环 ARX 链解集谓词：仅初始消息字 `w0` 自由，其余各轮转移固定。 -/
def OpenLoopSolution {α W : Type} (init : α) (firstRound : W → α → α)
    (steps : List (BijectiveStep α)) (target : α) (w0 : W) : Prop :=
  evalChain steps (firstRound w0 init) = target

/-- 基于 `BitVec N` 前 `m` 个低位的汉明距离。 -/
def hammingDistPrefix {N : Nat} : Nat → BitVec N → BitVec N → Nat
  | 0, _, _ => 0
  | m + 1, x, y =>
    (if x.getLsbD m != y.getLsbD m then 1 else 0) + hammingDistPrefix m x y

/-- `BitVec N` 全宽汉明距离。 -/
@[inline] def hammingDist {N : Nat} (x y : BitVec N) : Nat :=
  hammingDistPrefix N x y

theorem hammingDistPrefix_self {N : Nat} (m : Nat) (x : BitVec N) :
    hammingDistPrefix m x x = 0 := by
  induction m with
  | zero => rfl
  | succ m ih =>
    simp [hammingDistPrefix, ih]

theorem hammingDist_self {N : Nat} (x : BitVec N) : hammingDist x x = 0 :=
  hammingDistPrefix_self N x

theorem hammingDistPrefix_eq_zero_iff {N : Nat} (m : Nat) (x y : BitVec N) :
    hammingDistPrefix m x y = 0 ↔ ∀ i, i < m → x.getLsbD i = y.getLsbD i := by
  induction m with
  | zero =>
    simp [hammingDistPrefix]
  | succ m ih =>
    simp only [hammingDistPrefix]
    constructor
    · intro h i hi
      have hHead : (if x.getLsbD m != y.getLsbD m then 1 else 0) = 0 := by omega
      have hTail : hammingDistPrefix m x y = 0 := by omega
      have hBit : x.getLsbD m = y.getLsbD m := by
        cases hx : x.getLsbD m <;> cases hy : y.getLsbD m <;> simp_all
      by_cases him : i = m
      · subst him; exact hBit
      · exact (ih.mp hTail) i (by omega)
    · intro hAll
      have hBit : x.getLsbD m = y.getLsbD m := hAll m (Nat.lt_succ_self m)
      have hTail : hammingDistPrefix m x y = 0 :=
        ih.mpr (fun i hi => hAll i (Nat.lt_trans hi (Nat.lt_succ_self m)))
      simp [hBit, hTail]

theorem hammingDist_eq_zero_iff {N : Nat} (x y : BitVec N) :
    hammingDist x y = 0 ↔ x = y := by
  rw [hammingDist, hammingDistPrefix_eq_zero_iff]
  constructor
  · intro h
    exact BitVec.eq_of_getLsbD_eq h
  · intro h i _
    rw [h]

theorem hammingDistPrefix_symm {N : Nat} (m : Nat) (x y : BitVec N) :
    hammingDistPrefix m x y = hammingDistPrefix m y x := by
  induction m with
  | zero => rfl
  | succ m ih =>
    simp only [hammingDistPrefix, ih]
    cases x.getLsbD m <;> cases y.getLsbD m <;> rfl

theorem hammingDist_symm {N : Nat} (x y : BitVec N) :
    hammingDist x y = hammingDist y x :=
  hammingDistPrefix_symm N x y

/--
**定理 1（V3 动态数字锁存的无损性与解集不变性）**：
若边界离散状态 `s_latch` 满足末轮零物理残差（`lastStep.fwd s_latch = target`），则：
1. 该新边界处的单步传播误差严格归零（`s_latch = lastStep.inv target`）；
2. 将 `s_latch` 固化为新边界后的剩余开环子链 `prefixSteps` 的解集与原全链
   `prefixSteps ++ [lastStep]` 的解集完全一致。
-/
theorem exact_latch_preserves_solution_set {α W : Type}
    (init : α) (firstRound : W → α → α)
    (prefixSteps : List (BijectiveStep α)) (lastStep : BijectiveStep α)
    (target s_latch : α)
    (h_res : lastStep.fwd s_latch = target) :
    s_latch = lastStep.inv target ∧
    ∀ w0 : W,
      OpenLoopSolution init firstRound (prefixSteps ++ [lastStep]) target w0 ↔
      OpenLoopSolution init firstRound prefixSteps s_latch w0 := by
  refine ⟨(lastStep.fwd_eq_iff s_latch target).mp h_res, ?_⟩
  intro w0
  simp only [OpenLoopSolution, evalChain_append, evalChain]
  constructor
  · intro h
    rw [← h_res] at h
    exact lastStep.fwd_inj h
  · intro h
    rw [h, h_res]

/--
一般内部切分点锁存无损性：在链中任意位置切分 `leftSteps ++ rightSteps`，
一旦中间边界 `s_mid` 满足右侧子链零残差 `evalChain rightSteps s_mid = target`，
则原全链解集与锁定 `s_mid` 后的左侧子链解集完全等价，且 `s_mid` 精确等于真值逆推状态。
-/
theorem exact_latch_interior_preserves_solution_set {α W : Type}
    (init : α) (firstRound : W → α → α)
    (leftSteps rightSteps : List (BijectiveStep α))
    (target s_mid : α)
    (h_res : evalChain rightSteps s_mid = target) :
    s_mid = evalChainInv rightSteps target ∧
    ∀ w0 : W,
      OpenLoopSolution init firstRound (leftSteps ++ rightSteps) target w0 ↔
      OpenLoopSolution init firstRound leftSteps s_mid w0 := by
  refine ⟨(evalChain_eq_iff rightSteps s_mid target).mp h_res, ?_⟩
  intro w0
  simp only [OpenLoopSolution, evalChain_append]
  constructor
  · intro h
    have h1 := (evalChain_eq_iff rightSteps (evalChain leftSteps (firstRound w0 init)) target).mp h
    have h2 := (evalChain_eq_iff rightSteps s_mid target).mp h_res
    exact h1.trans h2.symm
  · intro h
    rw [h, h_res]

/-- BitVec 实例化推论：零残差锁存状态与真值边界状态的汉明距离严格为 `0`。 -/
theorem exact_latch_bitvec_zero_hamming {N : Nat} {W : Type}
    (init : BitVec N) (firstRound : W → BitVec N → BitVec N)
    (prefixSteps : List (BijectiveStep (BitVec N)))
    (lastStep : BijectiveStep (BitVec N))
    (target s_latch : BitVec N)
    (h_res : lastStep.fwd s_latch = target) :
    hammingDist s_latch (lastStep.inv target) = 0 ∧
    ∀ w0 : W,
      OpenLoopSolution init firstRound (prefixSteps ++ [lastStep]) target w0 ↔
      OpenLoopSolution init firstRound prefixSteps s_latch w0 := by
  have hMain := exact_latch_preserves_solution_set
    init firstRound prefixSteps lastStep target s_latch h_res
  refine ⟨(hammingDist_eq_zero_iff s_latch (lastStep.inv target)).mpr hMain.1, hMain.2⟩

/-!
### 3. 局域半径 `≤ 2` 比特翻转物理校验与开环无界续航定理 (`unbounded_open_loop_propagation`)
-/

/-- 翻转 `BitVec N` 第 `i` 位。 -/
@[inline] def flipBitAt {N : Nat} (i : Nat) (x : BitVec N) : BitVec N :=
  x ^^^ (1#N <<< i)

theorem getLsbD_flipBitAt {N : Nat} (i j : Nat) (hi : i < N) (x : BitVec N) :
    (flipBitAt i x).getLsbD j = if j = i then !(x.getLsbD i) else x.getLsbD j := by
  have h0 : 0 < N := Nat.lt_of_le_of_lt (Nat.zero_le i) hi
  simp only [flipBitAt, BitVec.getLsbD_xor, BitVec.getLsbD_shiftLeft, BitVec.getLsbD_one]
  by_cases hj : j = i
  · subst hj
    simp [hi, h0]
  · simp only [hj, ↓reduceIte]
    have hBitFalse : (decide (j < N) && !decide (j < i) && (decide (0 < N) && decide (j - i = 0))) = false := by
      by_cases hlt : j < i
      · simp [hlt]
      · have hne : j - i ≠ 0 := by omega
        simp [hlt, hne]
    rw [hBitFalse]
    simp

theorem hammingDistPrefix_flipBitAt_above {N : Nat} (m k : Nat) (hk : k < N)
    (hmk : m ≤ k) (x y : BitVec N) :
    hammingDistPrefix m (flipBitAt k x) y = hammingDistPrefix m x y := by
  induction m with
  | zero => rfl
  | succ m' ih =>
    have hne : m' ≠ k := by omega
    simp only [hammingDistPrefix, getLsbD_flipBitAt k m' hk x, hne, ↓reduceIte]
    rw [ih (by omega)]

/-- 半径 `≤ r` 局域比特翻转物理校验搜索算子（对应 V3 16-candidate Top-5 局域翻转校验的完整推广）。 -/
def searchWithinRadius {N : Nat} (r idx : Nat)
    (check : BitVec N → Bool) (cur : BitVec N) : Option (BitVec N) :=
  match idx with
  | 0 => if check cur then some cur else none
  | m + 1 =>
    match searchWithinRadius r m check cur with
    | some res => some res
    | none =>
      match r with
      | 0 => none
      | r' + 1 => searchWithinRadius r' m check (flipBitAt m cur)

theorem searchWithinRadius_sound {N : Nat} (r idx : Nat)
    (check : BitVec N → Bool) (cur res : BitVec N)
    (h : searchWithinRadius r idx check cur = some res) :
    check res = true := by
  induction idx generalizing r cur with
  | zero =>
    simp only [searchWithinRadius] at h
    split at h
    · cases h
      assumption
    · cases h
  | succ m ih =>
    simp only [searchWithinRadius] at h
    cases h1 : searchWithinRadius r m check cur with
    | some v =>
      rw [h1] at h
      cases h
      exact ih r cur h1
    | none =>
      rw [h1] at h
      cases r with
      | zero => cases h
      | succ r' => exact ih r' (flipBitAt m cur) h

theorem searchWithinRadius_complete {N : Nat} (r idx : Nat)
    (check : BitVec N → Bool) (cur u : BitVec N)
    (hIdx : idx ≤ N)
    (hCheck : check u = true)
    (hDist : hammingDistPrefix idx cur u ≤ r)
    (hAbove : ∀ j, idx ≤ j → j < N → cur.getLsbD j = u.getLsbD j) :
    ∃ res, searchWithinRadius r idx check cur = some res := by
  induction idx generalizing r cur with
  | zero =>
    have hEq : cur = u := by
      apply BitVec.eq_of_getLsbD_eq
      intro j hj
      exact hAbove j (Nat.zero_le j) hj
    subst hEq
    exact ⟨cur, by simp [searchWithinRadius, hCheck]⟩
  | succ m ih =>
    have hmLt : m < N := Nat.lt_of_lt_of_le (Nat.lt_succ_self m) hIdx
    have hmLe : m ≤ N := Nat.le_of_lt hmLt
    by_cases hBit : cur.getLsbD m = u.getLsbD m
    · have hDist' : hammingDistPrefix m cur u ≤ r := by
        simpa [hammingDistPrefix, hBit] using hDist
      have hAbove' : ∀ j, m ≤ j → j < N → cur.getLsbD j = u.getLsbD j := by
        intro j hmj hjN
        by_cases hjm : j = m
        · subst hjm; exact hBit
        · exact hAbove j (by omega) hjN
      obtain ⟨res, hRes⟩ := ih r cur hmLe hDist' hAbove'
      exact ⟨res, by simp [searchWithinRadius, hRes]⟩
    · have hXor : (cur.getLsbD m != u.getLsbD m) = true := by
        cases hcx : cur.getLsbD m <;> cases huy : u.getLsbD m <;> simp_all
      have hDistSucc : 1 + hammingDistPrefix m cur u ≤ r := by
        simpa [hammingDistPrefix, hXor] using hDist
      cases r with
      | zero => omega
      | succ r' =>
        have hDist' : hammingDistPrefix m (flipBitAt m cur) u ≤ r' := by
          rw [hammingDistPrefix_flipBitAt_above m m hmLt (Nat.le_refl m) cur u]
          omega
        have hAbove' : ∀ j, m ≤ j → j < N → (flipBitAt m cur).getLsbD j = u.getLsbD j := by
          intro j hmj hjN
          rw [getLsbD_flipBitAt m j hmLt cur]
          by_cases hjm : j = m
          · subst hjm
            simp only [↓reduceIte]
            cases hcx : cur.getLsbD j <;> cases huy : u.getLsbD j <;> simp_all
          · simp only [hjm, ↓reduceIte]
            exact hAbove j (by omega) hjN
        cases hFirst : searchWithinRadius (r' + 1) m check cur with
        | some v => exact ⟨v, by simp [searchWithinRadius, hFirst]⟩
        | none =>
          obtain ⟨res, hRes⟩ := ih r' (flipBitAt m cur) hmLe hDist' hAbove'
          exact ⟨res, by simp [searchWithinRadius, hFirst, hRes]⟩

/-- V3 单步动态数字锁存算子：在预测值 `pred` 的半径 `≤ 2` 比特邻域内搜索精确满足物理方程的唯一解。 -/
def latchStepRadius2 {N : Nat} (step : BijectiveStep (BitVec N))
    (sNext pred : BitVec N) : Option (BitVec N) :=
  searchWithinRadius 2 N (fun u => decide (step.fwd u = sNext)) pred

theorem latchStepRadius2_sound {N : Nat} (step : BijectiveStep (BitVec N))
    (sNext pred res : BitVec N)
    (h : latchStepRadius2 step sNext pred = some res) :
    res = step.inv sNext := by
  have hCheck := searchWithinRadius_sound 2 N (fun u => decide (step.fwd u = sNext)) pred res h
  simp only [decide_eq_true_eq] at hCheck
  exact (step.fwd_eq_iff res sNext).mp hCheck

theorem latchStepRadius2_exact {N : Nat} (step : BijectiveStep (BitVec N))
    (sNext pred : BitVec N)
    (hRadius : hammingDist pred (step.inv sNext) ≤ 2) :
    latchStepRadius2 step sNext pred = some (step.inv sNext) := by
  have hCheck : decide (step.fwd (step.inv sNext) = sNext) = true := by
    simp [step.right_inv]
  have hDist : hammingDistPrefix N pred (step.inv sNext) ≤ 2 := hRadius
  have hAbove : ∀ j, N ≤ j → j < N → pred.getLsbD j = (step.inv sNext).getLsbD j := by
    intro j hjN1 hjN2
    omega
  obtain ⟨res, hRes⟩ :=
    searchWithinRadius_complete 2 N (fun u => decide (step.fwd u = sNext))
      pred (step.inv sNext) (Nat.le_refl N) hCheck hDist hAbove
  have hEq : res = step.inv sNext := latchStepRadius2_sound step sNext pred res hRes
  simpa [latchStepRadius2, hEq] using hRes

/-- 深度 `d` 处的真值逆推状态（`d = 0` 为输出边界 `target`）。 -/
def trueStateAt {N : Nat} (stepAt : Nat → BijectiveStep (BitVec N))
    (target : BitVec N) : Nat → BitVec N
  | 0 => target
  | d + 1 => (stepAt d).inv (trueStateAt stepAt target d)

/-- 深度 `d` 处的 V3 动态数字锁存推进状态。 -/
def latchedStateAt {N : Nat} (stepAt : Nat → BijectiveStep (BitVec N))
    (predAt : Nat → BitVec N) (target : BitVec N) : Nat → Option (BitVec N)
  | 0 => some target
  | d + 1 =>
    match latchedStateAt stepAt predAt target d with
    | none => none
    | some sPrev => latchStepRadius2 (stepAt d) sPrev (predAt d)

/-- 将逆推深度 `L` 组装为从自由端推向 `target` 的前向双射链（`stepAt (L-1)` 最先执行，`stepAt 0` 末轮直抵 `target`）。 -/
def buildForwardChain {N : Nat} (stepAt : Nat → BijectiveStep (BitVec N)) :
    Nat → List (BijectiveStep (BitVec N))
  | 0 => []
  | d + 1 => stepAt d :: buildForwardChain stepAt d

theorem evalChain_buildForwardChain_trueState {N : Nat}
    (stepAt : Nat → BijectiveStep (BitVec N)) (target : BitVec N) (L : Nat) :
    evalChain (buildForwardChain stepAt L) (trueStateAt stepAt target L) = target := by
  induction L with
  | zero => rfl
  | succ d ih =>
    simp only [buildForwardChain, evalChain, trueStateAt, (stepAt d).right_inv]
    exact ih

/--
**定理 2（开环无界续航定理：`unbounded_open_loop_propagation`）**：
对任意自由轮数 `L : Nat`、任意 `N`-bit 双射转移动力学 `stepAt` 与输出终态边界 `target`，
只要单步局域神经预测器 `predAt d` 在每一步均能将误差压回局部纠错半径（`hammingDist ≤ 2`）内：
1. 对任意深度 `d ≤ L`，动态数字锁存推进算子 `latchedStateAt` 均精确返回 `some (trueStateAt stepAt target d)`；
2. 沿整条 `L` 步开环轨迹，每一步的离散状态传播误差（汉明距离）均被硬重置为 `0`；
3. 从第 `L` 步锁定状态出发前向求值必精确命中目标终态 `target`。
因此动态数字锁存使开环精确求解深度脱离指数衰减的李雅普诺夫视界，实现任意深度 `L` 的无损推进。
-/
theorem unbounded_open_loop_propagation {N : Nat} (L : Nat)
    (stepAt : Nat → BijectiveStep (BitVec N))
    (predAt : Nat → BitVec N)
    (target : BitVec N)
    (hLocal : ∀ d, d < L →
      hammingDist (predAt d) (trueStateAt stepAt target (d + 1)) ≤ 2) :
    (∀ d, d ≤ L →
      ∃ s, latchedStateAt stepAt predAt target d = some s ∧
        s = trueStateAt stepAt target d ∧
        hammingDist s (trueStateAt stepAt target d) = 0) ∧
    (latchedStateAt stepAt predAt target L = some (trueStateAt stepAt target L) ∧
      evalChain (buildForwardChain stepAt L) (trueStateAt stepAt target L) = target) := by
  induction L with
  | zero =>
    refine ⟨?_, rfl, rfl⟩
    intro d hd
    have hd0 : d = 0 := Nat.le_zero.mp hd
    subst hd0
    exact ⟨target, rfl, rfl, hammingDist_self target⟩
  | succ k ih =>
    have hLocalPre : ∀ d, d < k →
        hammingDist (predAt d) (trueStateAt stepAt target (d + 1)) ≤ 2 :=
      fun d hd => hLocal d (Nat.lt_trans hd (Nat.lt_succ_self k))
    obtain ⟨ihAll, ihEnd, _⟩ := ih hLocalPre
    have hStepK :
        latchStepRadius2 (stepAt k) (trueStateAt stepAt target k) (predAt k) =
          some (trueStateAt stepAt target (k + 1)) := by
      simpa [trueStateAt] using
        latchStepRadius2_exact (stepAt k) (trueStateAt stepAt target k) (predAt k)
          (hLocal k (Nat.lt_succ_self k))
    have hSucc :
        latchedStateAt stepAt predAt target (k + 1) =
          some (trueStateAt stepAt target (k + 1)) := by
      simp [latchedStateAt, ihEnd, hStepK]
    refine ⟨?_, hSucc, evalChain_buildForwardChain_trueState stepAt target (k + 1)⟩
    intro d hd
    by_cases hdk : d = k + 1
    · subst hdk
      exact ⟨trueStateAt stepAt target (k + 1), hSucc, rfl, hammingDist_self _⟩
    · exact ihAll d (by omega)

end PhaseSHA.Core.LatchingSoundness

