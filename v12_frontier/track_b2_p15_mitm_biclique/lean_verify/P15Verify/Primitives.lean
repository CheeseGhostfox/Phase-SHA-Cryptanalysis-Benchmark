import Std.Tactic.BVDecide

/-!
# P15Verify.Primitives: SHA-256 与 BTC Double-SHA256 (128 轮) 位向量原语与单步 `bv_decide` 引理

严格对齐 `src/sha256_ref.py` 及 `v11_discovery/p3_symbolic_lean4/lean_verify/P3Verify/Primitives.lean`，
零外部 Mathlib 依赖，供 P15 MitM / Biclique / Splice-and-Cut 独立块非干涉与截断前导零定理使用。
-/

namespace P15Verify

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

@[inline] def msgSchedulePrev (w_t w_m2 w_m7 w_m15 : BitVec 32) : BitVec 32 :=
  w_t - sigma1 w_m2 - w_m7 - sigma0 w_m15

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

@[inline] def backSolveA (s' : SHAState) : BitVec 32 :=
  (s'.e - s'.a) + (Sigma0 s'.b + maj32 s'.b s'.c s'.d)

@[inline] def sha256RoundInv (s' : SHAState) (k w : BitVec 32) : SHAState :=
  let t2 := Sigma0 s'.b + maj32 s'.b s'.c s'.d
  let t1 := s'.a - t2
  let d0 := s'.e - t1
  let h0 := (((t1 - Sigma1 s'.f) - ch32 s'.f s'.g s'.h) - k) - w
  { a := s'.b
    b := s'.c
    c := s'.d
    d := d0
    e := s'.f
    f := s'.g
    g := s'.h
    h := h0 }

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

/-- 标准 SHA-256 全部 64 轮常量 `K0..K63`。 -/
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
@[reducible] def K20 : BitVec 32 := 0x2de92c6f#32
@[reducible] def K21 : BitVec 32 := 0x4a7484aa#32
@[reducible] def K22 : BitVec 32 := 0x5cb0a9dc#32
@[reducible] def K23 : BitVec 32 := 0x76f988da#32
@[reducible] def K24 : BitVec 32 := 0x983e5152#32
@[reducible] def K25 : BitVec 32 := 0xa831c66d#32
@[reducible] def K26 : BitVec 32 := 0xb00327c8#32
@[reducible] def K27 : BitVec 32 := 0xbf597fc7#32
@[reducible] def K28 : BitVec 32 := 0xc6e00bf3#32
@[reducible] def K29 : BitVec 32 := 0xd5a79147#32
@[reducible] def K30 : BitVec 32 := 0x06ca6351#32
@[reducible] def K31 : BitVec 32 := 0x14292967#32
@[reducible] def K32 : BitVec 32 := 0x27b70a85#32
@[reducible] def K33 : BitVec 32 := 0x2e1b2138#32
@[reducible] def K34 : BitVec 32 := 0x4d2c6dfc#32
@[reducible] def K35 : BitVec 32 := 0x53380d13#32
@[reducible] def K36 : BitVec 32 := 0x650a7354#32
@[reducible] def K37 : BitVec 32 := 0x766a0abb#32
@[reducible] def K38 : BitVec 32 := 0x81c2c92e#32
@[reducible] def K39 : BitVec 32 := 0x92722c85#32
@[reducible] def K40 : BitVec 32 := 0xa2bfe8a1#32
@[reducible] def K41 : BitVec 32 := 0xa81a664b#32
@[reducible] def K42 : BitVec 32 := 0xc24b8b70#32
@[reducible] def K43 : BitVec 32 := 0xc76c51a3#32
@[reducible] def K44 : BitVec 32 := 0xd192e819#32
@[reducible] def K45 : BitVec 32 := 0xd6990624#32
@[reducible] def K46 : BitVec 32 := 0xf40e3585#32
@[reducible] def K47 : BitVec 32 := 0x106aa070#32
@[reducible] def K48 : BitVec 32 := 0x19a4c116#32
@[reducible] def K49 : BitVec 32 := 0x1e376c08#32
@[reducible] def K50 : BitVec 32 := 0x2748774c#32
@[reducible] def K51 : BitVec 32 := 0x34b0bcb5#32
@[reducible] def K52 : BitVec 32 := 0x391c0cb3#32
@[reducible] def K53 : BitVec 32 := 0x4ed8aa4a#32
@[reducible] def K54 : BitVec 32 := 0x5b9cca4f#32
@[reducible] def K55 : BitVec 32 := 0x682e6ff3#32
@[reducible] def K56 : BitVec 32 := 0x748f82ee#32
@[reducible] def K57 : BitVec 32 := 0x78a5636f#32
@[reducible] def K58 : BitVec 32 := 0x84c87814#32
@[reducible] def K59 : BitVec 32 := 0x8cc70208#32
@[reducible] def K60 : BitVec 32 := 0x90befffa#32
@[reducible] def K61 : BitVec 32 := 0xa4506ceb#32
@[reducible] def K62 : BitVec 32 := 0xbef9a3f7#32
@[reducible] def K63 : BitVec 32 := 0xc67178f2#32

/-- BTC Block 1b 固定 320-bit 尾部填充字 `W[4..15]` (`0x80000000, 0..0, 640`)。 -/
@[reducible] def btcB1PadW4  : BitVec 32 := 0x80000000#32
@[reducible] def btcB1PadW5  : BitVec 32 := 0x00000000#32
@[reducible] def btcB1PadW6  : BitVec 32 := 0x00000000#32
@[reducible] def btcB1PadW7  : BitVec 32 := 0x00000000#32
@[reducible] def btcB1PadW8  : BitVec 32 := 0x00000000#32
@[reducible] def btcB1PadW9  : BitVec 32 := 0x00000000#32
@[reducible] def btcB1PadW10 : BitVec 32 := 0x00000000#32
@[reducible] def btcB1PadW11 : BitVec 32 := 0x00000000#32
@[reducible] def btcB1PadW12 : BitVec 32 := 0x00000000#32
@[reducible] def btcB1PadW13 : BitVec 32 := 0x00000000#32
@[reducible] def btcB1PadW14 : BitVec 32 := 0x00000000#32
@[reducible] def btcB1PadW15 : BitVec 32 := 0x00000280#32

/-- BTC Block 2 固定 256-bit 尾部填充字 `W'[8..15]` (`0x80000000, 0..0, 256`)。 -/
@[reducible] def btcB2PadW8  : BitVec 32 := 0x80000000#32
@[reducible] def btcB2PadW9  : BitVec 32 := 0x00000000#32
@[reducible] def btcB2PadW10 : BitVec 32 := 0x00000000#32
@[reducible] def btcB2PadW11 : BitVec 32 := 0x00000000#32
@[reducible] def btcB2PadW12 : BitVec 32 := 0x00000000#32
@[reducible] def btcB2PadW13 : BitVec 32 := 0x00000000#32
@[reducible] def btcB2PadW14 : BitVec 32 := 0x00000000#32
@[reducible] def btcB2PadW15 : BitVec 32 := 0x00000100#32

/-! ### 核心单步 `bv_decide` 组合引理 -/

theorem bv_sub_add_assoc_cancel (d t1 t2 : BitVec 32) :
    ((d + t1) - (t1 + t2)) + t2 = d := by
  bv_decide

theorem bv_t1_extract_full (h s1 ch k w t2 : BitVec 32) :
    (((((h + s1 + ch + k + w + t2) - t2) - s1) - ch) - k) - w = h := by
  bv_decide

theorem bv_d_extract_from_e (d h s1 ch k w t2 : BitVec 32) :
    (d + (h + s1 + ch + k + w)) - ((h + s1 + ch + k + w + t2) - t2) = d := by
  bv_decide

theorem round_backSolveA (s : SHAState) (k w : BitVec 32) :
    backSolveA (sha256Round s k w) = s.d := by
  simp only [backSolveA, sha256Round]
  exact bv_sub_add_assoc_cancel s.d
    (s.h + Sigma1 s.e + ch32 s.e s.f s.g + k + w)
    (Sigma0 s.a + maj32 s.a s.b s.c)

theorem round_inv_exact (s : SHAState) (k w : BitVec 32) :
    sha256RoundInv (sha256Round s k w) k w = s := by
  simp only [sha256RoundInv, sha256Round]
  have hd : (s.d + (s.h + Sigma1 s.e + ch32 s.e s.f s.g + k + w)) -
      ((s.h + Sigma1 s.e + ch32 s.e s.f s.g + k + w +
        (Sigma0 s.a + maj32 s.a s.b s.c)) - (Sigma0 s.a + maj32 s.a s.b s.c)) = s.d :=
    bv_d_extract_from_e s.d s.h (Sigma1 s.e) (ch32 s.e s.f s.g) k w
      (Sigma0 s.a + maj32 s.a s.b s.c)
  have hh : ((((s.h + Sigma1 s.e + ch32 s.e s.f s.g + k + w +
      (Sigma0 s.a + maj32 s.a s.b s.c)) - (Sigma0 s.a + maj32 s.a s.b s.c)) -
      Sigma1 s.e) - ch32 s.e s.f s.g) - k - w = s.h :=
    bv_t1_extract_full s.h (Sigma1 s.e) (ch32 s.e s.f s.g) k w
      (Sigma0 s.a + maj32 s.a s.b s.c)
  cases s
  simp only [hd, hh]

theorem msg_schedule_inv_exact (w_m2 w_m7 w_m15 w_m16 : BitVec 32) :
    msgSchedulePrev (msgScheduleNext w_m2 w_m7 w_m15 w_m16) w_m2 w_m7 w_m15 = w_m16 := by
  dsimp only [msgSchedulePrev, msgScheduleNext]
  bv_decide

theorem sigma1_zero : sigma1 0#32 = 0#32 := by
  decide

theorem b1_w16_collapse (w1 w0 : BitVec 32) :
    msgScheduleNext btcB1PadW14 btcB1PadW9 w1 w0 = sigma0 w1 + w0 := by
  dsimp only [msgScheduleNext, btcB1PadW14, btcB1PadW9]
  rw [sigma1_zero]
  bv_decide

theorem b1_w17_collapse (w2 w1 : BitVec 32) :
    msgScheduleNext btcB1PadW15 btcB1PadW10 w2 w1 = sigma1 btcB1PadW15 + sigma0 w2 + w1 := by
  dsimp only [msgScheduleNext, btcB1PadW15, btcB1PadW10]
  bv_decide

theorem b2_wp16_collapse (wp1 wp0 : BitVec 32) :
    msgScheduleNext btcB2PadW14 btcB2PadW9 wp1 wp0 = sigma0 wp1 + wp0 := by
  dsimp only [msgScheduleNext, btcB2PadW14, btcB2PadW9]
  rw [sigma1_zero]
  bv_decide

theorem b2_wp17_collapse (wp2 wp1 : BitVec 32) :
    msgScheduleNext btcB2PadW15 btcB2PadW10 wp2 wp1 = sigma1 btcB2PadW15 + sigma0 wp2 + wp1 := by
  dsimp only [msgScheduleNext, btcB2PadW15, btcB2PadW10]
  bv_decide

theorem b2_wp18_collapse (wp16 wp3 wp2 : BitVec 32) :
    msgScheduleNext wp16 btcB2PadW11 wp3 wp2 = sigma1 wp16 + sigma0 wp3 + wp2 := by
  dsimp only [msgScheduleNext, btcB2PadW11]
  bv_decide

theorem b1_w18_collapse (w16 w3 w2 : BitVec 32) :
    msgScheduleNext w16 btcB1PadW7 w3 w2 = sigma1 w16 + sigma0 w3 + w2 := by
  dsimp only [msgScheduleNext, btcB1PadW7]
  bv_decide

/-! ### 精确前缀状态等价与 `sha256RoundInv` 非干涉传播引理 -/

def Eq7 (s1 s2 : SHAState) : Prop :=
  s1.a = s2.a ∧ s1.b = s2.b ∧ s1.c = s2.c ∧ s1.d = s2.d ∧
  s1.e = s2.e ∧ s1.f = s2.f ∧ s1.g = s2.g

def Eq6 (s1 s2 : SHAState) : Prop :=
  s1.a = s2.a ∧ s1.b = s2.b ∧ s1.c = s2.c ∧ s1.d = s2.d ∧
  s1.e = s2.e ∧ s1.f = s2.f

def Eq5 (s1 s2 : SHAState) : Prop :=
  s1.a = s2.a ∧ s1.b = s2.b ∧ s1.c = s2.c ∧ s1.d = s2.d ∧ s1.e = s2.e

def Eq4 (s1 s2 : SHAState) : Prop :=
  s1.a = s2.a ∧ s1.b = s2.b ∧ s1.c = s2.c ∧ s1.d = s2.d

def Eq3 (s1 s2 : SHAState) : Prop :=
  s1.a = s2.a ∧ s1.b = s2.b ∧ s1.c = s2.c

def Eq2 (s1 s2 : SHAState) : Prop :=
  s1.a = s2.a ∧ s1.b = s2.b

theorem inv_eq7_of_eq (s1 s2 : SHAState) (k w1 w2 : BitVec 32) (h : s1 = s2) :
    Eq7 (sha256RoundInv s1 k w1) (sha256RoundInv s2 k w2) := by
  subst h
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem inv_eq6_of_eq7 (s1 s2 : SHAState) (k w1 w2 : BitVec 32) (h : Eq7 s1 s2) :
    Eq6 (sha256RoundInv s1 k w1) (sha256RoundInv s2 k w2) := by
  rcases h with ⟨ha, hb, hc, hd, he, hf, hg⟩
  dsimp only [Eq6, sha256RoundInv]
  rw [ha, hb, hc, hd, he, hf, hg]
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem inv_eq5_of_eq6 (s1 s2 : SHAState) (k w1 w2 : BitVec 32) (h : Eq6 s1 s2) :
    Eq5 (sha256RoundInv s1 k w1) (sha256RoundInv s2 k w2) := by
  rcases h with ⟨ha, hb, hc, hd, he, hf⟩
  dsimp only [Eq5, sha256RoundInv]
  rw [ha, hb, hc, hd, he, hf]
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

theorem inv_eq4_of_eq5 (s1 s2 : SHAState) (k w1 w2 : BitVec 32) (h : Eq5 s1 s2) :
    Eq4 (sha256RoundInv s1 k w1) (sha256RoundInv s2 k w2) := by
  rcases h with ⟨ha, hb, hc, hd, he⟩
  dsimp only [Eq4, sha256RoundInv]
  rw [ha, hb, hc, hd, he]
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem inv_eq3_of_eq4 (s1 s2 : SHAState) (k w1 w2 : BitVec 32) (h : Eq4 s1 s2) :
    Eq3 (sha256RoundInv s1 k w1) (sha256RoundInv s2 k w2) := by
  rcases h with ⟨ha, hb, hc, hd⟩
  dsimp only [Eq3, sha256RoundInv]
  rw [hb, hc, hd]
  exact ⟨rfl, rfl, rfl⟩

theorem inv_eq2_of_eq3 (s1 s2 : SHAState) (k w1 w2 : BitVec 32) (h : Eq3 s1 s2) :
    Eq2 (sha256RoundInv s1 k w1) (sha256RoundInv s2 k w2) := by
  rcases h with ⟨ha, hb, hc⟩
  dsimp only [Eq2, sha256RoundInv]
  rw [hb, hc]
  exact ⟨rfl, rfl⟩

theorem subIV_eq5_of_eq5 (iv s1 s2 : SHAState) (h : Eq5 s1 s2) :
    Eq5 (subIV iv s1) (subIV iv s2) := by
  rcases h with ⟨ha, hb, hc, hd, he⟩
  dsimp only [Eq5, subIV]
  rw [ha, hb, hc, hd, he]
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

theorem fwd_ae_of_eq_prefix (s1 s2 : SHAState) (k w : BitVec 32) (h : s1 = s2) :
    (sha256Round s1 k w).a = (sha256Round s2 k w).a ∧
    (sha256Round s1 k w).e = (sha256Round s2 k w).e := by
  subst h
  exact ⟨rfl, rfl⟩

end P15Verify
