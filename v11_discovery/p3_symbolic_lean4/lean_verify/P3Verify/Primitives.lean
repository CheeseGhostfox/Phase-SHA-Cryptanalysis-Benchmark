import Std.Tactic.BVDecide

/-!
# P3Verify.Primitives: SHA-256 与 BTC 双 SHA-256 核心 BitVec 32 原语与单轮 `bv_decide` 引理

与 `formal_proofs/PhaseSHA/Core/BitVecIdentities.lean` 及 `src/sha256_ref.py` 保持位级精确一致，
零外部 Mathlib 依赖，供 P3 跨轮与跨块恒等式 `bv_decide` 形式化验证使用。
-/

namespace P3Verify

@[inline] def rotr32 (x : BitVec 32) (n : Nat) : BitVec 32 :=
  if n = 0 then x else (x >>> n) ||| (x <<< (32 - n))

@[inline] def shr32 (x : BitVec 32) (n : Nat) : BitVec 32 :=
  x >>> n

@[inline] def shl32 (x : BitVec 32) (n : Nat) : BitVec 32 :=
  x <<< n

@[inline] def ch32 (e f g : BitVec 32) : BitVec 32 :=
  (e &&& f) ^^^ ((~~~e) &&& g)

@[inline] def maj32 (a b c : BitVec 32) : BitVec 32 :=
  (a &&& b) ^^^ (a &&& c) ^^^ (b &&& c)

@[inline] def Sigma0 (x : BitVec 32) : BitVec 32 :=
  rotr32 x 2 ^^^ rotr32 x 13 ^^^ rotr32 x 22

@[inline] def Sigma1 (x : BitVec 32) : BitVec 32 :=
  rotr32 x 6 ^^^ rotr32 x 11 ^^^ rotr32 x 25

@[inline] def sigma0 (x : BitVec 32) : BitVec 32 :=
  rotr32 x 7 ^^^ rotr32 x 18 ^^^ shr32 x 3

@[inline] def sigma1 (x : BitVec 32) : BitVec 32 :=
  rotr32 x 17 ^^^ rotr32 x 19 ^^^ shr32 x 10

@[inline] def msgScheduleNext (w_m2 w_m7 w_m15 w_m16 : BitVec 32) : BitVec 32 :=
  sigma1 w_m2 + w_m7 + sigma0 w_m15 + w_m16

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

@[inline] def addIV (s iv : SHAState) : SHAState :=
  { a := s.a + iv.a
    b := s.b + iv.b
    c := s.c + iv.c
    d := s.d + iv.d
    e := s.e + iv.e
    f := s.f + iv.f
    g := s.g + iv.g
    h := s.h + iv.h }

@[inline] def subIV (digest iv : SHAState) : SHAState :=
  { a := digest.a - iv.a
    b := digest.b - iv.b
    c := digest.c - iv.c
    d := digest.d - iv.d
    e := digest.e - iv.e
    f := digest.f - iv.f
    g := digest.g - iv.g
    h := digest.h - iv.h }

/-- 标准 SHA-256 初始向量 IV。 -/
@[reducible] def stdIV : SHAState :=
  { a := 0x6a09e667#32
    b := 0xbb67ae85#32
    c := 0x3c6ef372#32
    d := 0xa54ff53a#32
    e := 0x510e527f#32
    f := 0x9b05688c#32
    g := 0x1f83d9ab#32
    h := 0x5be0cd19#32 }

/-- 标准 SHA-256 轮常量 `K0..K19`。 -/
@[reducible] def K0  : BitVec 32 := 0x428a2f98#32
@[reducible] def K1  : BitVec 32 := 0x71374491#32
@[reducible] def K2  : BitVec 32 := 0xb5c0fbcf#32
@[reducible] def K3  : BitVec 32 := 0xe9b5dba5#32
@[reducible] def K4  : BitVec 32 := 0x3956c25b#32
@[reducible] def K5  : BitVec 32 := 0x59f111f1#32
@[reducible] def K6  : BitVec 32 := 0x923f82a4#32
@[reducible] def K7  : BitVec 32 := 0xab1c5ed5#32
@[reducible] def K8  : BitVec 32 := 0xd807aa98#32
@[reducible] def K9  : BitVec 32 := 0x12835b01#32
@[reducible] def K10 : BitVec 32 := 0x243185be#32
@[reducible] def K11 : BitVec 32 := 0x550c7dc3#32
@[reducible] def K12 : BitVec 32 := 0x72be5d74#32
@[reducible] def K13 : BitVec 32 := 0x80deb1fe#32
@[reducible] def K14 : BitVec 32 := 0x9bdc06a7#32
@[reducible] def K15 : BitVec 32 := 0xc19bf174#32
@[reducible] def K16 : BitVec 32 := 0xe49b69c1#32
@[reducible] def K17 : BitVec 32 := 0xefbe4786#32
@[reducible] def K18 : BitVec 32 := 0x0fc19dc6#32
@[reducible] def K19 : BitVec 32 := 0x240ca1cc#32

/-- BTC 双 SHA-256 第二块固定填充字 `W'[8..15]`。 -/
@[reducible] def btcPadW8  : BitVec 32 := 0x80000000#32
@[reducible] def btcPadW9  : BitVec 32 := 0x00000000#32
@[reducible] def btcPadW10 : BitVec 32 := 0x00000000#32
@[reducible] def btcPadW11 : BitVec 32 := 0x00000000#32
@[reducible] def btcPadW12 : BitVec 32 := 0x00000000#32
@[reducible] def btcPadW13 : BitVec 32 := 0x00000000#32
@[reducible] def btcPadW14 : BitVec 32 := 0x00000000#32
@[reducible] def btcPadW15 : BitVec 32 := 0x00000100#32

/-! ### 核心单步位向量 `bv_decide` 组合引理 -/

theorem carry_shield_add_eq_xor (x y : BitVec 32) (h : x &&& y = 0#32) :
    x + y = x ^^^ y := by
  bv_decide

theorem lsb_add_eq_xor (x y : BitVec 32) :
    1#32 &&& (x + y) = 1#32 &&& (x ^^^ y) := by
  bv_decide

theorem lsb_add_const_eq_sub_const (x c : BitVec 32) :
    1#32 &&& (x + c) = 1#32 &&& (x - c) := by
  bv_decide

theorem lsb_sub_eq_xor (x y : BitVec 32) :
    1#32 &&& (x - y) = 1#32 &&& (x ^^^ y) := by
  bv_decide

theorem bv_sub_add_assoc_cancel (d t1 t2 : BitVec 32) :
    ((d + t1) - (t1 + t2)) + t2 = d := by
  bv_decide

theorem bv_t1_extract_sum (h s1 ch k w t2 : BitVec 32) :
    ((((h + s1 + ch + k + w + t2) - t2) - s1) - ch) - k = h + w := by
  bv_decide

theorem bv_t1_extract_full (h s1 ch k w t2 : BitVec 32) :
    (((((h + s1 + ch + k + w + t2) - t2) - s1) - ch) - k) - w = h := by
  bv_decide

theorem bv_t1_extract_w_from_e (d h s1 ch k w : BitVec 32) :
    ((((((d + (h + s1 + ch + k + w)) - d) - s1) - ch) - k) - h) = w := by
  bv_decide

theorem bv_bridge_e_from_a (d t1 t2 : BitVec 32) :
    d + ((t1 + t2) - t2) = d + t1 := by
  bv_decide

theorem bv_bridge_t1_cancel (d t1 t2 : BitVec 32) :
    ((d + t1) - (t1 + t2)) + t2 = d := by
  bv_decide

theorem bv_splice_sub_iv (iv x : BitVec 32) :
    (iv + x) - iv = x := by
  bv_decide

theorem bv_sub_zero (x : BitVec 32) :
    x - 0#32 = x := by
  bv_decide

theorem bv_add_zero_left (x : BitVec 32) :
    0#32 + x = x := by
  bv_decide

theorem bv_add_zero_right (x : BitVec 32) :
    x + 0#32 = x := by
  bv_decide

theorem bv_five_round_sum_cancel (d t1_0 t2_0 w4 : BitVec 32) :
    ((d + t1_0) + w4) - (t1_0 + t2_0) + t2_0 = d + w4 := by
  bv_decide

theorem bv_five_round_w4_cancel (d t1_0 t2_0 w4 : BitVec 32) :
    (((d + t1_0) + w4) - (t1_0 + t2_0) + t2_0) - d = w4 := by
  bv_decide

theorem bv_btc_b2_r0_w0 (h0 s1_0 ch_0 k0 w0 t2_0 iv0 : BitVec 32) :
    ((((h0 + s1_0 + ch_0 + k0 + (iv0 + w0) + t2_0) - t2_0) - (h0 + s1_0 + ch_0 + k0)) - iv0) = w0 := by
  bv_decide

theorem bv_btc_b2_t1_recover_w (h s1 ch k w iv t2 : BitVec 32) :
    (((((h + s1 + ch + k + (iv + w) + t2) - t2) - s1) - ch) - k) - h - iv = w := by
  bv_decide

theorem bv_btc_b2_e_recover_w (d h s1 ch k w iv : BitVec 32) :
    ((((((d + (h + s1 + ch + k + (iv + w))) - d) - h) - s1) - ch) - k) - iv = w := by
  bv_decide

/-- 单轮 `backSolveA` 精确还原 `s.d`（精确匹配 AST 括号树 `(s'.e - s'.a) + (Sigma0 s'.b + maj32 s'.b s'.c s'.d)`）。 -/
theorem round_backSolveA (s : SHAState) (k w : BitVec 32) :
    let s' := sha256Round s k w
    (s'.e - s'.a) + (Sigma0 s'.b + maj32 s'.b s'.c s'.d) = s.d := by
  simp only [sha256Round]
  exact bv_sub_add_assoc_cancel s.d
    (s.h + Sigma1 s.e + ch32 s.e s.f s.g + k + w)
    (Sigma0 s.a + maj32 s.a s.b s.c)

/-- 单轮反向提取 `s.h + w`（不含 `w`）。 -/
theorem round_extract_hw (s : SHAState) (k w : BitVec 32) :
    let s' := sha256Round s k w
    (((s'.a - (Sigma0 s'.b + maj32 s'.b s'.c s'.d)) - Sigma1 s'.f) - ch32 s'.f s'.g s'.h) - k =
      s.h + w := by
  simp only [sha256Round]
  exact bv_t1_extract_sum s.h (Sigma1 s.e) (ch32 s.e s.f s.g) k w
    (Sigma0 s.a + maj32 s.a s.b s.c)

/-- 单轮反向还原 `s.h`（当 `w = 0#32` 且省略 `- 0#32` 时）。 -/
theorem round_extract_hw_zero (s : SHAState) (k : BitVec 32) :
    let s' := sha256Round s k 0#32
    (((s'.a - (Sigma0 s'.b + maj32 s'.b s'.c s'.d)) - Sigma1 s'.f) - ch32 s'.f s'.g s'.h) - k =
      s.h := by
  have h := round_extract_hw s k 0#32
  rwa [bv_add_zero_right] at h

/-- 单轮反向还原 `s.h`（已知 `k, w`）。 -/
theorem round_extract_h (s : SHAState) (k w : BitVec 32) :
    let s' := sha256Round s k w
    ((((s'.a - (Sigma0 s'.b + maj32 s'.b s'.c s'.d)) - Sigma1 s'.f) - ch32 s'.f s'.g s'.h) - k) - w =
      s.h := by
  simp only [sha256Round]
  exact bv_t1_extract_full s.h (Sigma1 s.e) (ch32 s.e s.f s.g) k w
    (Sigma0 s.a + maj32 s.a s.b s.c)

/-- 单轮正向前推 `s'.e`（仅依赖 `s.d, s'.a, s.a, s.b, s.c`，消去 `W` 与 `E`）。 -/
theorem round_bridge_e_from_a (s : SHAState) (k w : BitVec 32) :
    let s' := sha256Round s k w
    s.d + (s'.a - (Sigma0 s.a + maj32 s.a s.b s.c)) = s'.e := by
  simp only [sha256Round]
  exact bv_bridge_e_from_a s.d
    (s.h + Sigma1 s.e + ch32 s.e s.f s.g + k + w)
    (Sigma0 s.a + maj32 s.a s.b s.c)

/-- 单轮从 `s'.e` 反向提取 `w`（已知 `s.d, s.e, s.f, s.g, s.h, k`）。 -/
theorem round_extract_w_from_e (s : SHAState) (k w : BitVec 32) :
    let s' := sha256Round s k w
    (((((s'.e - s.d) - Sigma1 s.e) - ch32 s.e s.f s.g) - k) - s.h) = w := by
  simp only [sha256Round]
  exact bv_t1_extract_w_from_e s.d s.h (Sigma1 s.e) (ch32 s.e s.f s.g) k w

theorem bv_t1_extract_w_from_e_hfirst (d h s1 ch k w : BitVec 32) :
    ((((((d + (h + s1 + ch + k + w)) - d) - h) - s1) - ch) - k) = w := by
  bv_decide

/-- 单轮从 `s'.e` 反向提取 `w`（先减 `s.d` 再减 `s.h` 的左结合顺序）。 -/
theorem round_extract_w_from_e_hfirst (s : SHAState) (k w : BitVec 32) :
    let s' := sha256Round s k w
    (((((s'.e - s.d) - s.h) - Sigma1 s.e) - ch32 s.e s.f s.g) - k) = w := by
  simp only [sha256Round]
  exact bv_t1_extract_w_from_e_hfirst s.d s.h (Sigma1 s.e) (ch32 s.e s.f s.g) k w

theorem bv_t1_extract_w_from_a_r0 (h s1 ch k w t2 : BitVec 32) :
    ((h + s1 + ch + k + w + t2) - t2) - (((h + s1) + ch) + k) = w := by
  bv_decide

/-- 第 0 轮从 `s'.a` 反向提取 `w`（合并常量 `T1` 前缀）。 -/
theorem round_extract_w_from_a_r0 (s : SHAState) (k w : BitVec 32) :
    let s' := sha256Round s k w
    (s'.a - (Sigma0 s.a + maj32 s.a s.b s.c)) -
      (((s.h + Sigma1 s.e) + ch32 s.e s.f s.g) + k) = w := by
  simp only [sha256Round]
  exact bv_t1_extract_w_from_a_r0 s.h (Sigma1 s.e) (ch32 s.e s.f s.g) k w
    (Sigma0 s.a + maj32 s.a s.b s.c)

theorem bv_t1_extract_w_from_a_hfirst (h s1 ch k w t2 : BitVec 32) :
    ((((((h + s1 + ch + k + w + t2) - t2) - h) - s1) - ch) - k) = w := by
  bv_decide

/-- 从 `s'.a` 反向提取 `w`（先减 `s.h` 再减 `Sigma1`、`Ch`、`k`）。 -/
theorem round_extract_w_from_a_hfirst (s : SHAState) (k w : BitVec 32) :
    let s' := sha256Round s k w
    (((((s'.a - (Sigma0 s.a + maj32 s.a s.b s.c)) - s.h) - Sigma1 s.e) -
      ch32 s.e s.f s.g) - k) = w := by
  simp only [sha256Round]
  exact bv_t1_extract_w_from_a_hfirst s.h (Sigma1 s.e) (ch32 s.e s.f s.g) k w
    (Sigma0 s.a + maj32 s.a s.b s.c)

theorem lsb_p3_id_004_aux (u k d_s1 t2 w4 s0d : BitVec 32)
    (h : ((u - k - d_s1) + t2) = s0d + w4) :
    1#32 &&& (((u + k - d_s1) + t2)) = 1#32 &&& (s0d ^^^ w4) := by
  bv_decide

theorem lsb_xor_add_eq_sub_add (a b c : BitVec 32) :
    1#32 &&& ((a ^^^ b) + c) = 1#32 &&& ((a - b) + c) := by
  bv_decide

theorem lsb_sub_eq_add_left (a b : BitVec 32) :
    1#32 &&& (a - b) = 1#32 &&& (a + b) := by
  bv_decide

theorem lsb_add_sub_eq_add_add (a b c : BitVec 32) :
    1#32 &&& (a + (b - c)) = 1#32 &&& (a + (b + c)) := by
  bv_decide

theorem sigma1_zero : sigma1 0#32 = 0#32 := by
  decide

theorem wp16_collapse (wp1 wp0 : BitVec 32) :
    sigma0 wp1 + wp0 = msgScheduleNext btcPadW14 btcPadW9 wp1 wp0 := by
  dsimp only [msgScheduleNext, btcPadW14, btcPadW9]
  rw [sigma1_zero]
  bv_decide

theorem wp17_collapse (wp2 wp1 : BitVec 32) :
    (sigma1 btcPadW15 + sigma0 wp2) + wp1 = msgScheduleNext btcPadW15 btcPadW10 wp2 wp1 := by
  dsimp only [msgScheduleNext, btcPadW15, btcPadW10]
  bv_decide

theorem wp18_collapse (wp16 wp3 wp2 : BitVec 32) :
    (sigma1 wp16 + sigma0 wp3) + wp2 = msgScheduleNext wp16 btcPadW11 wp3 wp2 := by
  dsimp only [msgScheduleNext, btcPadW11]
  bv_decide

end P3Verify
