import Std.Tactic.BVDecide

/-!
# Module 1: SHA-256 精确代数穿隧与无 W 钳位定理

本模块基于 Lean 4 原生 `BitVec 32` 精确形式化 SHA-256 的单轮压缩函数与消息扩展算子，
采用位向量 SAT/SMT 判决器 `bv_decide` 实现零占位符证明、零自定义公理的机器验证：
1. `four_round_clamp_w_invariant`：验证 `boundary_clamp.py` 中 4 轮解析倒推的无损性与 W 无关性；
2. `r18_a10_algebraic_tunneling`：验证 R=18 场景下 A_10 旁路恒等式对 W_16 与 E_13 的严格独立性；
3. `r18_ch_masking_shield`：验证 E_15[k] = 1 时 Ch(E_15, E_14, E_13)[k] 对未知 E_13[k] 的屏蔽效应
   以及触发屏蔽子空间与未触发子空间之间的 50% 精确对合双射。
-/

namespace PhaseSHA.Core.BitVecIdentities

/-- 32-bit 循环右移：精确对应 `src/sha256_ref.py` 中的 `rotr32`。 -/
@[inline] def rotr32 (x : BitVec 32) (n : Nat) : BitVec 32 :=
  if n = 0 then x else (x >>> n) ||| (x <<< (32 - n))

/-- 32-bit 逻辑右移：精确对应 `src/sha256_ref.py` 中的 `shr32`。 -/
@[inline] def shr32 (x : BitVec 32) (n : Nat) : BitVec 32 :=
  x >>> n

/-- SHA-256 选择函数 `Ch(e, f, g) = (e ∧ f) ⊕ (¬e ∧ g)`。 -/
@[inline] def ch32 (e f g : BitVec 32) : BitVec 32 :=
  (e &&& f) ^^^ ((~~~e) &&& g)

/-- SHA-256 多数函数 `Maj(a, b, c) = (a ∧ b) ⊕ (a ∧ c) ⊕ (b ∧ c)`。 -/
@[inline] def maj32 (a b c : BitVec 32) : BitVec 32 :=
  (a &&& b) ^^^ (a &&& c) ^^^ (b &&& c)

/-- SHA-256 大写 `Sigma0(x) = ROTR(x, 2) ⊕ ROTR(x, 13) ⊕ ROTR(x, 22)`。 -/
@[inline] def Sigma0 (x : BitVec 32) : BitVec 32 :=
  rotr32 x 2 ^^^ rotr32 x 13 ^^^ rotr32 x 22

/-- SHA-256 大写 `Sigma1(x) = ROTR(x, 6) ⊕ ROTR(x, 11) ⊕ ROTR(x, 25)`。 -/
@[inline] def Sigma1 (x : BitVec 32) : BitVec 32 :=
  rotr32 x 6 ^^^ rotr32 x 11 ^^^ rotr32 x 25

/-- SHA-256 小写 `sigma0(x) = ROTR(x, 7) ⊕ ROTR(x, 18) ⊕ SHR(x, 3)`。 -/
@[inline] def sigma0 (x : BitVec 32) : BitVec 32 :=
  rotr32 x 7 ^^^ rotr32 x 18 ^^^ shr32 x 3

/-- SHA-256 小写 `sigma1(x) = ROTR(x, 17) ⊕ ROTR(x, 19) ⊕ SHR(x, 10)`。 -/
@[inline] def sigma1 (x : BitVec 32) : BitVec 32 :=
  rotr32 x 17 ^^^ rotr32 x 19 ^^^ shr32 x 10

/-- SHA-256 消息扩展递推：`W_t = sigma1(W_{t-2}) + W_{t-7} + sigma0(W_{t-15}) + W_{t-16}`。 -/
@[inline] def msgScheduleNext (w_m2 w_m7 w_m15 w_m16 : BitVec 32) : BitVec 32 :=
  sigma1 w_m2 + w_m7 + sigma0 w_m15 + w_m16

/--
SHA-256 第 `t` 轮前的 8 字双轨状态：
`(a, b, c, d, e, f, g, h) = (A_t, A_{t-1}, A_{t-2}, A_{t-3}, E_t, E_{t-1}, E_{t-2}, E_{t-3})`。
-/
structure SHAState where
  a : BitVec 32
  b : BitVec 32
  c : BitVec 32
  d : BitVec 32
  e : BitVec 32
  f : BitVec 32
  g : BitVec 32
  h : BitVec 32
  deriving DecidableEq, Repr

/-- 标准 SHA-256 单轮状态转移。 -/
@[inline] def sha256Round (s : SHAState) (k w : BitVec 32) : SHAState :=
  let t1 := s.h + Sigma1 s.e + ch32 s.e s.f s.g + k + w
  let t2 := Sigma0 s.a + maj32 s.a s.b s.c
  { a := t1 + t2
    b := s.a
    c := s.b
    d := s.c
    e := s.d + t1
    f := s.e
    g := s.f
    h := s.g }

/-- 终态与初始向量 `IV` 模 `2^32` 逐字叠加形成摘要。 -/
@[inline] def addIV (s iv : SHAState) : SHAState :=
  { a := s.a + iv.a
    b := s.b + iv.b
    c := s.c + iv.c
    d := s.d + iv.d
    e := s.e + iv.e
    f := s.f + iv.f
    g := s.g + iv.g
    h := s.h + iv.h }

/-- 摘要剥离初始向量 `IV` 恢复终态 `S_R`。 -/
@[inline] def subIV (digest iv : SHAState) : SHAState :=
  { a := digest.a - iv.a
    b := digest.b - iv.b
    c := digest.c - iv.c
    d := digest.d - iv.d
    e := digest.e - iv.e
    f := digest.f - iv.f
    g := digest.g - iv.g
    h := digest.h - iv.h }

/-- 连续执行 4 轮 SHA-256 压缩转移。 -/
@[inline] def sha256FourRounds (s0 : SHAState)
    (k0 k1 k2 k3 w0 w1 w2 w3 : BitVec 32) : SHAState :=
  let s1 := sha256Round s0 k0 w0
  let s2 := sha256Round s1 k1 w1
  let s3 := sha256Round s2 k2 w2
  sha256Round s3 k3 w3

/--
单步无 `W` 钳位解析倒推公式（精确对齐 `boundary_clamp.py` 第 263–271 行）：
`A_{t-3} = E_{t+1} - A_{t+1} + Sigma0(A_t) + Maj(A_t, A_{t-1}, A_{t-2})`。
注意其函数签名完全不接收轮常量 `k` 或消息字 `w`。
-/
@[inline] def backSolveA (eNext aNext a0 a1 a2 : BitVec 32) : BitVec 32 :=
  eNext - aNext + Sigma0 a0 + maj32 a0 a1 a2

/--
由最终摘要 `digest` 与公开初始向量 `iv` 向前无损解析倒推 4 轮状态 `(A_{R-4}, A_{R-5}, A_{R-6}, A_{R-7})`。
-/
@[inline] def clampBackSolve4 (digest iv : SHAState) :
    BitVec 32 × BitVec 32 × BitVec 32 × BitVec 32 :=
  let sR := subIV digest iv
  let aR4 := backSolveA sR.e sR.a sR.b sR.c sR.d
  let aR5 := backSolveA sR.f sR.b sR.c sR.d aR4
  let aR6 := backSolveA sR.g sR.c sR.d aR4 aR5
  let aR7 := backSolveA sR.h sR.d aR4 aR5 aR6
  (aR4, aR5, aR6, aR7)

/-- R=18 中 `A_10` 的旁路解析公式，显式独立于 `W_16` 与 `E_13`。 -/
@[inline] def tunnelA10 (e14 a14 a13 a12 a11 : BitVec 32) : BitVec 32 :=
  e14 - a14 + Sigma0 a13 + maj32 a13 a12 a11

-- 轻量基础位向量辅助引理（拆分证明以控制 SAT 求解规模与内存占用）

private theorem add_sub_cancel_bv (x y : BitVec 32) : (x + y) - y = x := by
  bv_decide

private theorem elim_t1_bv (d t1 t2a t2b : BitVec 32) :
    (d + t1) - (t1 + (t2a + t2b)) + t2a + t2b = d := by
  bv_decide

theorem subIV_addIV (s iv : SHAState) : subIV (addIV s iv) iv = s := by
  cases s
  cases iv
  simp only [subIV, addIV, add_sub_cancel_bv]

theorem single_round_backsolve_w_invariant (s : SHAState) (k w : BitVec 32) :
    let s' := sha256Round s k w
    backSolveA s'.e s'.a s'.b s'.c s'.d = s.d := by
  simp only [sha256Round, backSolveA]
  exact elim_t1_bv s.d
    (s.h + Sigma1 s.e + ch32 s.e s.f s.g + k + w)
    (Sigma0 s.a) (maj32 s.a s.b s.c)

/-!
### 定理 1：4 轮无 W 钳位定理 (`four_round_clamp_w_invariant`)
-/

/--
**定理 1（4 轮解析倒推的无损性与 W 无关性）**：
对任意第 `R-4` 轮前状态 `s0`、任意初始向量 `iv`、任意轮常量 `k0..k3` 以及任意消息字 `w0..w3`，
仅依据公开 `iv` 与最终摘要 `digest = addIV (sha256FourRounds s0 k0..k3 w0..w3) iv` 进行 4 步解析倒推，
必定精确还原 `(A_{R-4}, A_{R-5}, A_{R-6}, A_{R-7}) = (s0.a, s0.b, s0.c, s0.d)`，
且与中间消息字 `W_{R-4..R-1}` 严格无关。
-/
theorem four_round_clamp_w_invariant
    (s0 iv : SHAState) (k0 k1 k2 k3 w0 w1 w2 w3 : BitVec 32) :
    clampBackSolve4
      (addIV (sha256FourRounds s0 k0 k1 k2 k3 w0 w1 w2 w3) iv) iv =
      (s0.a, s0.b, s0.c, s0.d) := by
  dsimp [clampBackSolve4]
  rw [subIV_addIV]
  dsimp [sha256FourRounds]
  have h3 : backSolveA
      (sha256Round (sha256Round (sha256Round (sha256Round s0 k0 w0) k1 w1) k2 w2) k3 w3).e
      (sha256Round (sha256Round (sha256Round (sha256Round s0 k0 w0) k1 w1) k2 w2) k3 w3).a
      (sha256Round (sha256Round (sha256Round (sha256Round s0 k0 w0) k1 w1) k2 w2) k3 w3).b
      (sha256Round (sha256Round (sha256Round (sha256Round s0 k0 w0) k1 w1) k2 w2) k3 w3).c
      (sha256Round (sha256Round (sha256Round (sha256Round s0 k0 w0) k1 w1) k2 w2) k3 w3).d =
      (sha256Round (sha256Round (sha256Round s0 k0 w0) k1 w1) k2 w2).d :=
    single_round_backsolve_w_invariant _ k3 w3
  rw [h3]
  have h2 : backSolveA
      (sha256Round (sha256Round (sha256Round (sha256Round s0 k0 w0) k1 w1) k2 w2) k3 w3).f
      (sha256Round (sha256Round (sha256Round (sha256Round s0 k0 w0) k1 w1) k2 w2) k3 w3).b
      (sha256Round (sha256Round (sha256Round (sha256Round s0 k0 w0) k1 w1) k2 w2) k3 w3).c
      (sha256Round (sha256Round (sha256Round (sha256Round s0 k0 w0) k1 w1) k2 w2) k3 w3).d
      (sha256Round (sha256Round (sha256Round s0 k0 w0) k1 w1) k2 w2).d =
      (sha256Round (sha256Round s0 k0 w0) k1 w1).d := by
    simpa [sha256Round] using
      single_round_backsolve_w_invariant
        (sha256Round (sha256Round s0 k0 w0) k1 w1) k2 w2
  rw [h2]
  have h1 : backSolveA
      (sha256Round (sha256Round (sha256Round (sha256Round s0 k0 w0) k1 w1) k2 w2) k3 w3).g
      (sha256Round (sha256Round (sha256Round (sha256Round s0 k0 w0) k1 w1) k2 w2) k3 w3).c
      (sha256Round (sha256Round (sha256Round (sha256Round s0 k0 w0) k1 w1) k2 w2) k3 w3).d
      (sha256Round (sha256Round (sha256Round s0 k0 w0) k1 w1) k2 w2).d
      (sha256Round (sha256Round s0 k0 w0) k1 w1).d =
      (sha256Round s0 k0 w0).d := by
    simpa [sha256Round] using
      single_round_backsolve_w_invariant (sha256Round s0 k0 w0) k1 w1
  rw [h1]
  have h0 : backSolveA
      (sha256Round (sha256Round (sha256Round (sha256Round s0 k0 w0) k1 w1) k2 w2) k3 w3).h
      (sha256Round (sha256Round (sha256Round (sha256Round s0 k0 w0) k1 w1) k2 w2) k3 w3).d
      (sha256Round (sha256Round (sha256Round s0 k0 w0) k1 w1) k2 w2).d
      (sha256Round (sha256Round s0 k0 w0) k1 w1).d
      (sha256Round s0 k0 w0).d =
      s0.d := by
    simpa [sha256Round] using single_round_backsolve_w_invariant s0 k0 w0
  rw [h0]
  simp [sha256Round]

/--
推论：对于任意两组消息字 `w0..w3` 与 `w0'..w3'`（乃至任意轮常量），只要经 4 轮后到达同一摘要 `digest`，
解析恢复出的 `(A_{R-4}, A_{R-5}, A_{R-6}, A_{R-7})` 必严格相同，证明无 W 钳位对中间 W 全空间不变。
-/
theorem four_round_clamp_independent_of_w
    (s0 s0' iv digest : SHAState)
    (k0 k1 k2 k3 k0' k1' k2' k3' : BitVec 32)
    (w0 w1 w2 w3 w0' w1' w2' w3' : BitVec 32)
    (h1 : addIV (sha256FourRounds s0 k0 k1 k2 k3 w0 w1 w2 w3) iv = digest)
    (h2 : addIV (sha256FourRounds s0' k0' k1' k2' k3' w0' w1' w2' w3') iv = digest) :
    (s0.a, s0.b, s0.c, s0.d) = (s0'.a, s0'.b, s0'.c, s0'.d) ∧
      (s0.a, s0.b, s0.c, s0.d) = clampBackSolve4 digest iv := by
  have eq1 := four_round_clamp_w_invariant s0 iv k0 k1 k2 k3 w0 w1 w2 w3
  have eq2 := four_round_clamp_w_invariant s0' iv k0' k1' k2' k3' w0' w1' w2' w3'
  rw [h1] at eq1
  rw [h2] at eq2
  exact ⟨eq1.symm.trans eq2, eq1.symm⟩

/-!
### 定理 2：R=18 A_10 精确代数穿隧定理 (`r18_a10_algebraic_tunneling`)
-/

/-- 直接由 `bv_decide` 证死的 32-bit 旁路穿隧核心位向量恒等式。 -/
theorem r18_a10_bv_identity
    (a13 a12 a11 a10 e13 e12 e11 e10 k13 w13 : BitVec 32) :
    let t1 := e10 + Sigma1 e13 + ch32 e13 e12 e11 + k13 + w13
    let t2 := Sigma0 a13 + maj32 a13 a12 a11
    let e14 := a10 + t1
    let a14 := t1 + t2
    e14 - a14 + Sigma0 a13 + maj32 a13 a12 a11 = a10 := by
  bv_decide

/--
**定理 2（R=18 `A_10` 精确代数穿隧定理）**：
在第 `t = 13` 轮前状态 `s13`（含 `A_13 = s13.a, A_12 = s13.b, A_11 = s13.c, A_10 = s13.d`）下：
1. 在全 `2^32` 空间上恒成立旁路等式：
   `A_10 = E_14 - A_14 + Sigma0(A_13) + Maj(A_13, A_12, A_11)`；
2. 该旁路公式与被锁死的 `E_13` 完全独立：任意替换 `E_13`（及 `E_12, E_11, E_10, K_13, W_13`）
   得到的新转移态 `s14'` 仍给出完全相同的 `A_10`；
3. 该旁路公式与第 16 轮扩展消息字 `W_16` 完全独立：即使 `W_16`（及 `W_14..W_17`）任意变化，
   从终态摘要经 4 轮无 W 钳位恢复的 `(A_14, A_13, A_12, A_11)` 保持不变，且 `tunnelA10` 仍精确还原 `A_10`。
-/
theorem r18_a10_algebraic_tunneling
    (s13 iv : SHAState)
    (k13 w13 k14 k15 k16 k17 w14 w15 w16 w16' w17 : BitVec 32)
    (e13' e12' e11' e10' k13' w13' : BitVec 32) :
    let s14 := sha256Round s13 k13 w13
    let s13' : SHAState :=
      { a := s13.a, b := s13.b, c := s13.c, d := s13.d
        e := e13', f := e12', g := e11', h := e10' }
    let s14' := sha256Round s13' k13' w13'
    let dig := addIV (sha256FourRounds s14 k14 k15 k16 k17 w14 w15 w16 w17) iv
    let dig' := addIV (sha256FourRounds s14 k14 k15 k16 k17 w14 w15 w16' w17) iv
    (tunnelA10 s14.e s14.a s13.a s13.b s13.c = s13.d) ∧
    (tunnelA10 s14'.e s14'.a s13'.a s13'.b s13'.c =
      tunnelA10 s14.e s14.a s13.a s13.b s13.c) ∧
    (clampBackSolve4 dig iv = (s14.a, s13.a, s13.b, s13.c) ∧
      clampBackSolve4 dig iv = clampBackSolve4 dig' iv) := by
  intro s14 s13' s14' dig dig'
  have hMain : tunnelA10 s14.e s14.a s13.a s13.b s13.c = s13.d := by
    dsimp [s14, tunnelA10]
    exact r18_a10_bv_identity s13.a s13.b s13.c s13.d s13.e s13.f s13.g s13.h k13 w13
  have hAlt : tunnelA10 s14'.e s14'.a s13'.a s13'.b s13'.c = s13.d := by
    dsimp [s14', s13', tunnelA10]
    exact r18_a10_bv_identity s13.a s13.b s13.c s13.d e13' e12' e11' e10' k13' w13'
  have hClamp1 := four_round_clamp_w_invariant s14 iv k14 k15 k16 k17 w14 w15 w16 w17
  have hClamp2 := four_round_clamp_w_invariant s14 iv k14 k15 k16 k17 w14 w15 w16' w17
  have hs14bcd : (s14.a, s14.b, s14.c, s14.d) = (s14.a, s13.a, s13.b, s13.c) := rfl
  refine ⟨hMain, hAlt.trans hMain.symm, ?_⟩
  refine ⟨?_, ?_⟩
  · exact hClamp1.trans hs14bcd
  · exact (hClamp1.trans hClamp2.symm)

/-!
### 定理 3：Ch 位级屏蔽与 50% 穿隧对合定理 (`r18_ch_masking_shield`)
-/

/-- 词级位掩码恒等式（`bv_decide` 全空间证死）：当掩码位为 1 时 `Ch` 严格等价于 `e14` 且与 `e13` 无关。 -/
theorem r18_ch_masking_word_bv (e15 e14 e13 e13' : BitVec 32) :
    (e15 &&& ch32 e15 e14 e13 = e15 &&& e14) ∧
    (e15 &&& ch32 e15 e14 e13 = e15 &&& ch32 e15 e14 e13') := by
  dsimp [ch32]
  bv_decide

/--
**定理 3（`Ch` 位级屏蔽护盾定理）**：
对任意 `e15, e14, e13, e13' : BitVec 32` 与任意位索引 `k < 32`，
一旦 `e15.getLsbD k = true`（即 `E_15[k] = 1`），则：
1. `Ch(E_15, E_14, E_13)[k] = E_14[k]`；
2. `Ch(E_15, E_14, E_13)[k]` 与未知变量 `E_13[k]`（乃至整个 `E_13` 字）严格无关。
-/
theorem r18_ch_masking_shield (e15 e14 e13 e13' : BitVec 32)
    (k : Nat) (hk : k < 32)
    (he15 : e15.getLsbD k = true) :
    (ch32 e15 e14 e13).getLsbD k = e14.getLsbD k ∧
    (ch32 e15 e14 e13).getLsbD k = (ch32 e15 e14 e13').getLsbD k := by
  have _ := hk
  simp only [ch32, BitVec.getLsbD_xor, BitVec.getLsbD_and, BitVec.getLsbD_not]
  rw [he15]
  simp

/-- 辅助翻转单比特算子：将 `x` 的第 `k` 位取反。 -/
@[inline] def flipBit32 (k : Fin 32) (x : BitVec 32) : BitVec 32 :=
  x ^^^ (1#32 <<< k.val)

private theorem xor_cancel_bv (x m : BitVec 32) : (x ^^^ m) ^^^ m = x := by
  bv_decide

theorem flipBit32_involutive (k : Fin 32) (x : BitVec 32) :
    flipBit32 k (flipBit32 k x) = x := by
  exact xor_cancel_bv x (1#32 <<< k.val)

theorem flipBit32_getLsbD (k : Fin 32) (x : BitVec 32) :
    (flipBit32 k x).getLsbD k.val = !(x.getLsbD k.val) := by
  have h1 : k.val < 32 := k.isLt
  simp [flipBit32, h1]

/--
**推论（50% 位级穿隧空间对合双射定理）**：
对任意位索引 `k : Fin 32`，`flipBit32 k` 在屏蔽激活集 `{e15 | e15[k] = 1}`
与未激活集 `{e15 | e15[k] = 0}` 之间建立了精确的自反双射，严格证明了 50% 穿隧比例。
-/
theorem r18_ch_masking_half_space_bijection (k : Fin 32) (e15 : BitVec 32) :
    flipBit32 k (flipBit32 k e15) = e15 ∧
    (e15.getLsbD k.val = true ↔ (flipBit32 k e15).getLsbD k.val = false) ∧
    (e15.getLsbD k.val = false ↔ (flipBit32 k e15).getLsbD k.val = true) := by
  refine ⟨flipBit32_involutive k e15, ?_, ?_⟩
  · rw [flipBit32_getLsbD]
    cases e15.getLsbD k.val <;> simp
  · rw [flipBit32_getLsbD]
    cases e15.getLsbD k.val <;> simp

end PhaseSHA.Core.BitVecIdentities
