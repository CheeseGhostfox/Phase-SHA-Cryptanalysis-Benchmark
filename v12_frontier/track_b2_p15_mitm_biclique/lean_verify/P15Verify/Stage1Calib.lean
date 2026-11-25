import P15Verify.Primitives

/-!
# P15Verify.Stage1Calib: Stage 1 阳性校准独立块定理形式化证真

覆盖 `results/lean_candidates.json` 中全部 6 条 Stage 1 候选（`R=20, 24, 28, 32` 减轮 MitM 双向非干涉，
以及 `64` 轮局域 `L=6`、`L=8` Khovratovich 风格独立 Biclique 子块），均包含完整 `sigma0`/`sigma1` 双向消息扩展。
-/

namespace P15Verify.Stage1Calib
open P15Verify

/-- `p15_stage1a_mitm_20r` 前向非干涉：基 `W[1..16]` 下，`S_15` 对 Blue 变量 `W[15]` 任意扰动严格免疫。 -/
theorem p15_stage1a_mitm_20r_fwd (s0 : SHAState)
    (w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w15' w16 : BitVec 32) :
    let w0 := msgSchedulePrev w16 w14 w9 w1
    let run := fun u15 =>
      let _ := u15
      let s1  := sha256Round s0  K0  w0
      let s2  := sha256Round s1  K1  w1
      let s3  := sha256Round s2  K2  w2
      let s4  := sha256Round s3  K3  w3
      let s5  := sha256Round s4  K4  w4
      let s6  := sha256Round s5  K5  w5
      let s7  := sha256Round s6  K6  w6
      let s8  := sha256Round s7  K7  w7
      let s9  := sha256Round s8  K8  w8
      let s10 := sha256Round s9  K9  w9
      let s11 := sha256Round s10 K10 w10
      let s12 := sha256Round s11 K11 w11
      let s13 := sha256Round s12 K12 w12
      let s14 := sha256Round s13 K13 w13
      sha256Round s14 K14 w14
    run w15 = run w15' := by
  rfl

/-- `p15_stage1a_mitm_20r` 后向非干涉：含完整 `W[17..19]` 非线性扩展，后向 `S_15` 对 Red 变量 `W[13]` 严格免疫。 -/
theorem p15_stage1a_mitm_20r_bwd (s20 : SHAState)
    (w1 w2 w3 w4 w10 w11 w12 w13 w13' w15 w16 : BitVec 32) :
    let w17 := msgScheduleNext w15 w10 w2 w1
    let w18 := msgScheduleNext w16 w11 w3 w2
    let w19 := msgScheduleNext w17 w12 w4 w3
    let s19 := sha256RoundInv s20 K19 w19
    let s18 := sha256RoundInv s19 K18 w18
    let s17 := sha256RoundInv s18 K17 w17
    let s16 := sha256RoundInv s17 K16 w16
    let s15 := sha256RoundInv s16 K15 w15
    let run := fun u13 =>
      let _ := u13
      s15
    run w13 = run w13' := by
  rfl

/-- `p15_stage1a_mitm_24r` 前向非干涉：基 `W[2..17]` 含 `W[0..1]` 逆推扩展，前向 `S_11` 对 Blue `W[11]` 严格免疫。 -/
theorem p15_stage1a_mitm_24r_fwd (s0 : SHAState)
    (w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w11' w14 w15 w16 w17 : BitVec 32) :
    let w1 := msgSchedulePrev w17 w15 w10 w2
    let w0 := msgSchedulePrev w16 w14 w9 w1
    let run := fun u11 =>
      let _ := u11
      let s1  := sha256Round s0  K0  w0
      let s2  := sha256Round s1  K1  w1
      let s3  := sha256Round s2  K2  w2
      let s4  := sha256Round s3  K3  w3
      let s5  := sha256Round s4  K4  w4
      let s6  := sha256Round s5  K5  w5
      let s7  := sha256Round s6  K6  w6
      let s8  := sha256Round s7  K7  w7
      let s9  := sha256Round s8  K8  w8
      let s10 := sha256Round s9  K9  w9
      sha256Round s10 K10 w10
    run w11 = run w11' := by
  rfl

/-- `p15_stage1a_mitm_24r` 后向非干涉：含 `W[18..23]` 非线性扩展，后向 `S_11` 对 Red `W[10]` 严格免疫。 -/
theorem p15_stage1a_mitm_24r_bwd (s24 : SHAState)
    (w2 w3 w4 w5 w6 w7 w8 w10 w10' w11 w12 w13 w14 w15 w16 w17 : BitVec 32) :
    let w18 := msgScheduleNext w16 w11 w3 w2
    let w19 := msgScheduleNext w17 w12 w4 w3
    let w20 := msgScheduleNext w18 w13 w5 w4
    let w21 := msgScheduleNext w19 w14 w6 w5
    let w22 := msgScheduleNext w20 w15 w7 w6
    let w23 := msgScheduleNext w21 w16 w8 w7
    let run := fun u10 =>
      let _ := u10
      let s23 := sha256RoundInv s24 K23 w23
      let s22 := sha256RoundInv s23 K22 w22
      let s21 := sha256RoundInv s22 K21 w21
      let s20 := sha256RoundInv s21 K20 w20
      let s19 := sha256RoundInv s20 K19 w19
      let s18 := sha256RoundInv s19 K18 w18
      let s17 := sha256RoundInv s18 K17 w17
      let s16 := sha256RoundInv s17 K16 w16
      let s15 := sha256RoundInv s16 K15 w15
      let s14 := sha256RoundInv s15 K14 w14
      let s13 := sha256RoundInv s14 K13 w13
      let s12 := sha256RoundInv s13 K12 w12
      sha256RoundInv s12 K11 w11
    run w10 = run w10' := by
  rfl

/-- `p15_stage1a_mitm_28r` 前向非干涉：基 `W[7..22]` 含 `W[0..6]` 逆推扩展，前向 `S_8` 对 Blue `W[8]` 严格免疫。 -/
theorem p15_stage1a_mitm_28r_fwd (s0 : SHAState)
    (w7 w8 w8' w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 : BitVec 32) :
    let w6 := msgSchedulePrev w22 w20 w15 w7
    let w5 := msgSchedulePrev w21 w19 w14 w6
    let w4 := msgSchedulePrev w20 w18 w13 w5
    let w3 := msgSchedulePrev w19 w17 w12 w4
    let w2 := msgSchedulePrev w18 w16 w11 w3
    let w1 := msgSchedulePrev w17 w15 w10 w2
    let w0 := msgSchedulePrev w16 w14 w9 w1
    let run := fun u8 =>
      let _ := u8
      let s1 := sha256Round s0 K0 w0
      let s2 := sha256Round s1 K1 w1
      let s3 := sha256Round s2 K2 w2
      let s4 := sha256Round s3 K3 w3
      let s5 := sha256Round s4 K4 w4
      let s6 := sha256Round s5 K5 w5
      let s7 := sha256Round s6 K6 w6
      sha256Round s7 K7 w7
    run w8 = run w8' := by
  rfl

private theorem inv6_w5_eq2 (s : SHAState)
    (k5 k4 k3 k2 k1 k0 w5 w5' w4 w3 w2 w1 w0 : BitVec 32) :
    let s5 := sha256RoundInv s k5 w5
    let s4 := sha256RoundInv s5 k4 w4
    let s3 := sha256RoundInv s4 k3 w3
    let s2 := sha256RoundInv s3 k2 w2
    let s1 := sha256RoundInv s2 k1 w1
    let s0 := sha256RoundInv s1 k0 w0
    let s5' := sha256RoundInv s k5 w5'
    let s4' := sha256RoundInv s5' k4 w4
    let s3' := sha256RoundInv s4' k3 w3
    let s2' := sha256RoundInv s3' k2 w2
    let s1' := sha256RoundInv s2' k1 w1
    let s0' := sha256RoundInv s1' k0 w0
    Eq2 s0 s0' := by
  dsimp only
  apply inv_eq2_of_eq3
  apply inv_eq3_of_eq4
  apply inv_eq4_of_eq5
  apply inv_eq5_of_eq6
  apply inv_eq6_of_eq7
  exact inv_eq7_of_eq s s k5 w5 w5' rfl

private theorem inv3_wrap_inv1_eq4 (s3 h_wrap : SHAState)
    (k2 k1 k0 k31 w2 w2' w1 w1' w0 w0' w31 w31' : BitVec 32) :
    let s2 := sha256RoundInv s3 k2 w2
    let s1 := sha256RoundInv s2 k1 w1
    let s0 := sha256RoundInv s1 k0 w0
    let s31 := sha256RoundInv (subIV h_wrap s0) k31 w31
    let s2' := sha256RoundInv s3 k2 w2'
    let s1' := sha256RoundInv s2' k1 w1'
    let s0' := sha256RoundInv s1' k0 w0'
    let s31' := sha256RoundInv (subIV h_wrap s0') k31 w31'
    Eq4 s31 s31' := by
  dsimp only
  apply inv_eq4_of_eq5
  apply subIV_eq5_of_eq5
  apply inv_eq5_of_eq6
  apply inv_eq6_of_eq7
  exact inv_eq7_of_eq s3 s3 k2 w2 w2' rfl

/-- `p15_stage1a_mitm_28r` 后向非干涉：含 `W[23..27]` 扩展与 `t=13 -> 8` 跨 6 轮 `backSolveA` 延迟消元，
`S_8.(a, b)` 对 Red `W[13]` 任意扰动严格免疫。 -/
theorem p15_stage1a_mitm_28r_bwd (s28 : SHAState)
    (w7 w8 w9 w10 w11 w12 w13 w13' w14 w15 w16 w17 w18 w19 w20 w21 w22 : BitVec 32) :
    let w23 := msgScheduleNext w21 w16 w8 w7
    let w24 := msgScheduleNext w22 w17 w9 w8
    let w25 := msgScheduleNext w23 w18 w10 w9
    let w26 := msgScheduleNext w24 w19 w11 w10
    let w27 := msgScheduleNext w25 w20 w12 w11
    let s27 := sha256RoundInv s28 K27 w27
    let s26 := sha256RoundInv s27 K26 w26
    let s25 := sha256RoundInv s26 K25 w25
    let s24 := sha256RoundInv s25 K24 w24
    let s23 := sha256RoundInv s24 K23 w23
    let s22 := sha256RoundInv s23 K22 w22
    let s21 := sha256RoundInv s22 K21 w21
    let s20 := sha256RoundInv s21 K20 w20
    let s19 := sha256RoundInv s20 K19 w19
    let s18 := sha256RoundInv s19 K18 w18
    let s17 := sha256RoundInv s18 K17 w17
    let s16 := sha256RoundInv s17 K16 w16
    let s15 := sha256RoundInv s16 K15 w15
    let s14 := sha256RoundInv s15 K14 w14
    let step := fun u13 =>
      let s13 := sha256RoundInv s14 K13 u13
      let s12 := sha256RoundInv s13 K12 w12
      let s11 := sha256RoundInv s12 K11 w11
      let s10 := sha256RoundInv s11 K10 w10
      let s9  := sha256RoundInv s10 K9  w9
      sha256RoundInv s9 K8 w8
    Eq2 (step w13) (step w13') := by
  intro w23 w24 w25 w26 w27 s27 s26 s25 s24 s23 s22 s21 s20 s19 s18 s17 s16 s15 s14
  exact inv6_w5_eq2 s14 K13 K12 K11 K10 K9 K8 w13 w13' w12 w11 w10 w9 w8

private theorem round_bcd_indep_w (s : SHAState) (k w w' : BitVec 32) :
    let r0 := sha256Round s k w
    let r1 := sha256Round s k w'
    r0.d = r1.d ∧ r0.c = r1.c ∧ r0.b = r1.b :=
  ⟨rfl, rfl, rfl⟩

/-- `p15_stage1a_mitm_32r` 前向非干涉：基 `W[7..22]`，前向 `t=16..30` 中 `W[30]` 虽受 Blue `W[15]` 污染，
但 `S_31.(b, c, d)` 对 `W[15]` 严格免疫。 -/
theorem p15_stage1a_mitm_32r_fwd (s16 : SHAState)
    (w7 w8 w9 w10 w11 w12 w13 w14 w15 w15' w16 w17 w18 w19 w20 w21 w22 : BitVec 32) :
    let w23 := msgScheduleNext w21 w16 w8  w7
    let w24 := msgScheduleNext w22 w17 w9  w8
    let w25 := msgScheduleNext w23 w18 w10 w9
    let w26 := msgScheduleNext w24 w19 w11 w10
    let w27 := msgScheduleNext w25 w20 w12 w11
    let w28 := msgScheduleNext w26 w21 w13 w12
    let w29 := msgScheduleNext w27 w22 w14 w13
    let s17 := sha256Round s16 K16 w16
    let s18 := sha256Round s17 K17 w17
    let s19 := sha256Round s18 K18 w18
    let s20 := sha256Round s19 K19 w19
    let s21 := sha256Round s20 K20 w20
    let s22 := sha256Round s21 K21 w21
    let s23 := sha256Round s22 K22 w22
    let s24 := sha256Round s23 K23 w23
    let s25 := sha256Round s24 K24 w24
    let s26 := sha256Round s25 K25 w25
    let s27 := sha256Round s26 K26 w26
    let s28 := sha256Round s27 K27 w27
    let s29 := sha256Round s28 K28 w28
    let s30 := sha256Round s29 K29 w29
    let run := fun u15 =>
      let w30 := msgScheduleNext w28 w23 u15 w14
      sha256Round s30 K30 w30
    let r0 := run w15
    let r1 := run w15'
    r0.d = r1.d ∧ r0.c = r1.c ∧ r0.b = r1.b := by
  intro w23 w24 w25 w26 w27 w28 w29 s17 s18 s19 s20 s21 s22 s23 s24 s25 s26 s27 s28 s29 s30
  exact round_bcd_indep_w s30 K30 (msgScheduleNext w28 w23 w15 w14) (msgScheduleNext w28 w23 w15' w14)

/-- `p15_stage1a_mitm_32r` 后向回绕非干涉：从 `S_16` 倒推 `t=15..0`、经 `H - S_0` 回绕至 `S_32` 再倒推 `t=31`，
尽管 `W[0..2]` 与 `W[31]` 均受 Red `W[16]` 污染，`S_31.(b, c, d)` 对 `W[16]` 严格免疫。 -/
theorem p15_stage1a_mitm_32r_bwd (s16 h_wrap : SHAState)
    (w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w16' w17 w18 w19 w20 w21 w22 : BitVec 32) :
    let w6  := msgSchedulePrev w22 w20 w15 w7
    let w5  := msgSchedulePrev w21 w19 w14 w6
    let w4  := msgSchedulePrev w20 w18 w13 w5
    let w3  := msgSchedulePrev w19 w17 w12 w4
    let w24 := msgScheduleNext w22 w17 w9  w8
    let s15 := sha256RoundInv s16 K15 w15
    let s14 := sha256RoundInv s15 K14 w14
    let s13 := sha256RoundInv s14 K13 w13
    let s12 := sha256RoundInv s13 K12 w12
    let s11 := sha256RoundInv s12 K11 w11
    let s10 := sha256RoundInv s11 K10 w10
    let s9  := sha256RoundInv s10 K9  w9
    let s8  := sha256RoundInv s9  K8  w8
    let s7  := sha256RoundInv s8  K7  w7
    let s6  := sha256RoundInv s7  K6  w6
    let s5  := sha256RoundInv s6  K5  w5
    let s4  := sha256RoundInv s5  K4  w4
    let s3  := sha256RoundInv s4  K3  w3
    let run := fun u16 =>
      let w2  := msgSchedulePrev w18 u16 w11 w3
      let w1  := msgSchedulePrev w17 w15 w10 w2
      let w0  := msgSchedulePrev u16 w14 w9  w1
      let w23 := msgScheduleNext w21 u16 w8  w7
      let w25 := msgScheduleNext w23 w18 w10 w9
      let w26 := msgScheduleNext w24 w19 w11 w10
      let w27 := msgScheduleNext w25 w20 w12 w11
      let w28 := msgScheduleNext w26 w21 w13 w12
      let w29 := msgScheduleNext w27 w22 w14 w13
      let _ := w28
      let w31 := msgScheduleNext w29 w24 u16 w15
      let s2  := sha256RoundInv s3  K2  w2
      let s1  := sha256RoundInv s2  K1  w1
      let s0  := sha256RoundInv s1  K0  w0
      let s32 := subIV h_wrap s0
      sha256RoundInv s32 K31 w31
    let r0 := run w16
    let r1 := run w16'
    r0.d = r1.d ∧ r0.c = r1.c ∧ r0.b = r1.b := by
  intro w6 w5 w4 w3 w24 s15 s14 s13 s12 s11 s10 s9 s8 s7 s6 s5 s4 s3 run r0 r1
  have h4 : Eq4 r0 r1 := by
    dsimp only [r0, r1, run]
    exact inv3_wrap_inv1_eq4 s3 h_wrap K2 K1 K0 K31 _ _ _ _ _ _ _ _
  exact ⟨h4.2.2.2, h4.2.2.1, h4.2.1⟩

/-- `p15_stage1b_local_bic_6r_t0`：64 轮局域 `L=6`（`t=0..5`）独立 Biclique 前向 `@r3` 与后向 `@r0` 双向非干涉。 -/
theorem p15_stage1b_local_bic_6r_t0_fwd (s0 : SHAState)
    (w0 w1 w2 w4 w4' w5 w5' : BitVec 32) :
    let run := fun u4 u5 =>
      let _ := (u4, u5)
      let s1 := sha256Round s0 K0 w0
      let s2 := sha256Round s1 K1 w1
      sha256Round s2 K2 w2
    run w4 w5 = run w4' w5' := by
  rfl

theorem p15_stage1b_local_bic_6r_t0_bwd (s6 : SHAState)
    (w0 w0' w1 w1' w2 w3 w4 w5 : BitVec 32) :
    let run := fun u0 u1 =>
      let s5 := sha256RoundInv s6 K5 w5
      let s4 := sha256RoundInv s5 K4 w4
      let s3 := sha256RoundInv s4 K3 w3
      let s2 := sha256RoundInv s3 K2 w2
      let s1 := sha256RoundInv s2 K1 u1
      sha256RoundInv s1 K0 u0
    Eq6 (run w0 w1) (run w0' w1') := by
  dsimp only
  apply inv_eq6_of_eq7
  apply inv_eq7_of_eq
  rfl

/-- `p15_stage1b_local_bic_8r_t0`：64 轮局域 `L=8`（`t=0..7`）独立 Biclique 前向 `@r4` 与后向 `@r0` 双向非干涉。 -/
theorem p15_stage1b_local_bic_8r_t0_fwd (s0 : SHAState)
    (w0 w1 w2 w3 w6 w6' w7 w7' : BitVec 32) :
    let run := fun u6 u7 =>
      let _ := (u6, u7)
      let s1 := sha256Round s0 K0 w0
      let s2 := sha256Round s1 K1 w1
      let s3 := sha256Round s2 K2 w2
      sha256Round s3 K3 w3
    run w6 w7 = run w6' w7' := by
  rfl

theorem p15_stage1b_local_bic_8r_t0_bwd (s8 : SHAState)
    (w0 w0' w1 w1' w2 w3 w4 w5 w6 w7 : BitVec 32) :
    let run := fun u0 u1 =>
      let s7 := sha256RoundInv s8 K7 w7
      let s6 := sha256RoundInv s7 K6 w6
      let s5 := sha256RoundInv s6 K5 w5
      let s4 := sha256RoundInv s5 K4 w4
      let s3 := sha256RoundInv s4 K3 w3
      let s2 := sha256RoundInv s3 K2 w2
      let s1 := sha256RoundInv s2 K1 u1
      sha256RoundInv s1 K0 u0
    Eq6 (run w0 w1) (run w0' w1') := by
  dsimp only
  apply inv_eq6_of_eq7
  apply inv_eq7_of_eq
  rfl

end P15Verify.Stage1Calib
