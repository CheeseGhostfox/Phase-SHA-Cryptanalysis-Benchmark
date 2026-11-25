import P3Verify.Primitives

namespace P3Verify.TrackA1
open P3Verify

/-- `p3_id_001`：A1 5-round W_{t+4} recovery from (S_5, A_{t-3}) (elim 4W+4E)。 -/
theorem p3_id_001 (s0 : SHAState) (w0 w1 w2 w3 w4 : BitVec 32) :
    let s1 := sha256Round s0 K0 w0
    let s2 := sha256Round s1 K1 w1
    let s3 := sha256Round s2 K2 w2
    let s4 := sha256Round s3 K3 w3
    let s5 := sha256Round s4 K4 w4
    let v0 := Sigma0 s5.b
    let v1 := maj32 s5.b s5.c s5.d
    let v2 := v0 + v1
    let v3 := s5.a - v2
    let v4 := Sigma1 s5.f
    let v5 := v3 - v4
    let v6 := ch32 s5.f s5.g s5.h
    let v7 := v5 - v6
    let v8 := v7 - 0x3956c25b#32
    let v9 := s5.e - s5.a
    let v10 := v9 + v2
    let v11 := v8 - v10
    let v12 := s5.f - s5.b
    let v13 := Sigma0 s5.c
    let v14 := maj32 s5.c s5.d v10
    let v15 := v13 + v14
    let v16 := v12 + v15
    let v17 := Sigma0 v16
    let v18 := s5.g - s5.c
    let v19 := Sigma0 s5.d
    let v20 := maj32 s5.d v10 v16
    let v21 := v19 + v20
    let v22 := v18 + v21
    let v23 := s5.h - s5.d
    let v24 := Sigma0 v10
    let v25 := maj32 v10 v16 v22
    let v26 := v24 + v25
    let v27 := v23 + v26
    let v28 := maj32 v16 v22 v27
    let v29 := v17 + v28
    let v30 := v11 + v29
    let v31 := v30 - s0.d
    0xffffffff#32 &&& v31 = 0xffffffff#32 &&& (w4) := by
  intro s1 s2 s3 s4 s5 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31
  have hv10 : v10 = s1.a := by
    change (s5.e - s5.a) + ((Sigma0 s5.b) + (maj32 s5.b s5.c s5.d)) = s1.a
    exact round_backSolveA s4 K4 w4
  have hv16 : v16 = s0.a := by
    change (s5.f - s5.b) + ((Sigma0 s5.c) + (maj32 s5.c s5.d v10)) = s0.a
    rw [hv10]
    exact round_backSolveA s3 K3 w3
  have hv22 : v22 = s0.b := by
    change (s5.g - s5.c) + ((Sigma0 s5.d) + (maj32 s5.d v10 v16)) = s0.b
    rw [hv10, hv16]
    exact round_backSolveA s2 K2 w2
  have hv27 : v27 = s0.c := by
    change (s5.h - s5.d) + ((Sigma0 v10) + (maj32 v10 v16 v22)) = s0.c
    rw [hv10, hv16, hv22]
    exact round_backSolveA s1 K1 w1
  have hv30 : v30 = s0.d + w4 := by
    change (((((s5.a - ((Sigma0 s5.b) + (maj32 s5.b s5.c s5.d))) - (Sigma1 s5.f)) - (ch32 s5.f s5.g s5.h)) - 0x3956c25b#32) - v10) + ((Sigma0 v16) + (maj32 v16 v22 v27)) = s0.d + w4
    rw [round_extract_hw s4 K4 w4]
    rw [hv10, hv16, hv22, hv27]
    dsimp only [s1, sha256Round]
    exact bv_five_round_sum_cancel s0.d (s0.h + Sigma1 s0.e + ch32 s0.e s0.f s0.g + K0 + w0) (Sigma0 s0.a + maj32 s0.a s0.b s0.c) w4
  have hv31 : v31 = w4 := by
    change v30 - s0.d = w4
    rw [hv30]
    exact bv_splice_sub_iv s0.d w4
  rw [hv31]

/-- `p3_id_002`：A1 5-round bidirectional A_{t-3}+W_{t+4} coupling (elim 4W+4E)。 -/
theorem p3_id_002 (s0 : SHAState) (w0 w1 w2 w3 w4 : BitVec 32) :
    let s1 := sha256Round s0 K0 w0
    let s2 := sha256Round s1 K1 w1
    let s3 := sha256Round s2 K2 w2
    let s4 := sha256Round s3 K3 w3
    let s5 := sha256Round s4 K4 w4
    let v0 := Sigma0 s5.b
    let v1 := maj32 s5.b s5.c s5.d
    let v2 := v0 + v1
    let v3 := s5.a - v2
    let v4 := Sigma1 s5.f
    let v5 := v3 - v4
    let v6 := ch32 s5.f s5.g s5.h
    let v7 := v5 - v6
    let v8 := v7 - 0x3956c25b#32
    let v9 := s5.e - s5.a
    let v10 := v9 + v2
    let v11 := v8 - v10
    let v12 := s5.f - s5.b
    let v13 := Sigma0 s5.c
    let v14 := maj32 s5.c s5.d v10
    let v15 := v13 + v14
    let v16 := v12 + v15
    let v17 := Sigma0 v16
    let v18 := s5.g - s5.c
    let v19 := Sigma0 s5.d
    let v20 := maj32 s5.d v10 v16
    let v21 := v19 + v20
    let v22 := v18 + v21
    let v23 := s5.h - s5.d
    let v24 := Sigma0 v10
    let v25 := maj32 v10 v16 v22
    let v26 := v24 + v25
    let v27 := v23 + v26
    let v28 := maj32 v16 v22 v27
    let v29 := v17 + v28
    let v30 := v11 + v29
    0xffffffff#32 &&& v30 = 0xffffffff#32 &&& (s0.d + w4) := by
  intro s1 s2 s3 s4 s5 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30
  have hv10 : v10 = s1.a := by
    change (s5.e - s5.a) + ((Sigma0 s5.b) + (maj32 s5.b s5.c s5.d)) = s1.a
    exact round_backSolveA s4 K4 w4
  have hv16 : v16 = s0.a := by
    change (s5.f - s5.b) + ((Sigma0 s5.c) + (maj32 s5.c s5.d v10)) = s0.a
    rw [hv10]
    exact round_backSolveA s3 K3 w3
  have hv22 : v22 = s0.b := by
    change (s5.g - s5.c) + ((Sigma0 s5.d) + (maj32 s5.d v10 v16)) = s0.b
    rw [hv10, hv16]
    exact round_backSolveA s2 K2 w2
  have hv27 : v27 = s0.c := by
    change (s5.h - s5.d) + ((Sigma0 v10) + (maj32 v10 v16 v22)) = s0.c
    rw [hv10, hv16, hv22]
    exact round_backSolveA s1 K1 w1
  have hv30 : v30 = s0.d + w4 := by
    change (((((s5.a - ((Sigma0 s5.b) + (maj32 s5.b s5.c s5.d))) - (Sigma1 s5.f)) - (ch32 s5.f s5.g s5.h)) - 0x3956c25b#32) - v10) + ((Sigma0 v16) + (maj32 v16 v22 v27)) = s0.d + w4
    rw [round_extract_hw s4 K4 w4]
    rw [hv10, hv16, hv22, hv27]
    dsimp only [s1, sha256Round]
    exact bv_five_round_sum_cancel s0.d (s0.h + Sigma1 s0.e + ch32 s0.e s0.f s0.g + K0 + w0) (Sigma0 s0.a + maj32 s0.a s0.b s0.c) w4
  rw [hv30]

/-- `p3_id_003`：Track A1 5-round seed [mut] [Mask 1b]。 -/
theorem p3_id_003 (s0 : SHAState) (w0 w1 w2 w3 w4 : BitVec 32) :
    let s1 := sha256Round s0 K0 w0
    let s2 := sha256Round s1 K1 w1
    let s3 := sha256Round s2 K2 w2
    let s4 := sha256Round s3 K3 w3
    let s5 := sha256Round s4 K4 w4
    let v0 := Sigma0 s5.b
    let v1 := maj32 s5.b s5.c s5.d
    let v2 := v0 + v1
    let v3 := s5.a - v2
    let v4 := Sigma1 s5.f
    let v5 := v3 - v4
    let v6 := ch32 s5.f s5.g s5.h
    let v7 := v5 - v6
    let v8 := v7 - 0x3956c25b#32
    let v9 := s5.e - s5.a
    let v10 := v9 + v2
    let v11 := v8 - v10
    let v12 := s5.f - s5.b
    let v13 := Sigma0 s5.c
    let v14 := maj32 s5.c s5.d v10
    let v15 := v13 + v14
    let v16 := v12 + v15
    let v17 := Sigma0 v16
    let v18 := s5.g - s5.c
    let v19 := Sigma0 s5.d
    let v20 := maj32 s5.d v10 v16
    let v21 := v19 + v20
    let v22 := v18 + v21
    let v23 := s5.h - s5.d
    let v24 := Sigma0 v10
    let v25 := maj32 v10 v16 v22
    let v26 := v24 + v25
    let v27 := v23 + v26
    let v28 := maj32 v16 v22 v27
    let v29 := v17 + v28
    let v30 := v11 ^^^ v29
    0x00000001#32 &&& v30 = 0x00000001#32 &&& (s0.d + w4) := by
  intro s1 s2 s3 s4 s5 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30
  have hv10 : v10 = s1.a := by
    change (s5.e - s5.a) + ((Sigma0 s5.b) + (maj32 s5.b s5.c s5.d)) = s1.a
    exact round_backSolveA s4 K4 w4
  have hv16 : v16 = s0.a := by
    change (s5.f - s5.b) + ((Sigma0 s5.c) + (maj32 s5.c s5.d v10)) = s0.a
    rw [hv10]
    exact round_backSolveA s3 K3 w3
  have hv22 : v22 = s0.b := by
    change (s5.g - s5.c) + ((Sigma0 s5.d) + (maj32 s5.d v10 v16)) = s0.b
    rw [hv10, hv16]
    exact round_backSolveA s2 K2 w2
  have hv27 : v27 = s0.c := by
    change (s5.h - s5.d) + ((Sigma0 v10) + (maj32 v10 v16 v22)) = s0.c
    rw [hv10, hv16, hv22]
    exact round_backSolveA s1 K1 w1
  have hv8 : v8 = s1.e + w4 := by
    change (((s5.a - ((Sigma0 s5.b) + (maj32 s5.b s5.c s5.d))) - (Sigma1 s5.f)) - (ch32 s5.f s5.g s5.h)) - 0x3956c25b#32 = s1.e + w4
    exact round_extract_hw s4 K4 w4
  have hadd : v11 + v29 = s0.d + w4 := by
    change (v8 - v10) + ((Sigma0 v16) + (maj32 v16 v22 v27)) = s0.d + w4
    rw [hv8, hv10, hv16, hv22, hv27]
    dsimp only [s1, sha256Round]
    exact bv_five_round_sum_cancel s0.d (s0.h + Sigma1 s0.e + ch32 s0.e s0.f s0.g + K0 + w0) (Sigma0 s0.a + maj32 s0.a s0.b s0.c) w4
  change 1#32 &&& (v11 ^^^ v29) = 1#32 &&& (s0.d + w4)
  rw [← lsb_add_eq_xor, hadd]

/-- `p3_id_004`：Track A1 5-round seed + MaskZero(s0.d,s0.b) [Carry-shielded (s0.d&w4)==0 => XOR] [mut] [Mask 1b]。 -/
theorem p3_id_004 (s0 : SHAState) (w0 w1 w2 w3 w4 : BitVec 32) :
    let s1 := sha256Round s0 K0 w0
    let s2 := sha256Round s1 K1 w1
    let s3 := sha256Round s2 K2 w2
    let s4 := sha256Round s3 K3 w3
    let s5 := sha256Round s4 K4 w4
    let v0 := Sigma0 s5.b
    let v1 := maj32 s5.b s5.c s5.d
    let v2 := v0 + v1
    let v3 := s5.a - v2
    let v4 := Sigma1 s5.f
    let v5 := v3 - v4
    let v6 := ch32 s5.f s5.g s5.h
    let v7 := v5 - v6
    let v8 := v7 + 0x3956c25b#32
    let v9 := s5.e - s5.a
    let v10 := v9 + v2
    let v11 := v8 - v10
    let v12 := s5.f - s5.b
    let v13 := Sigma0 s5.c
    let v14 := maj32 s5.c s5.d v10
    let v15 := v13 + v14
    let v16 := v12 + v15
    let v17 := Sigma0 v16
    let v18 := s5.g - s5.c
    let v19 := Sigma0 s5.d
    let v20 := maj32 s5.d v10 v16
    let v21 := v19 + v20
    let v22 := v18 + v21
    let v23 := s5.h - s5.d
    let v24 := Sigma0 v10
    let v25 := maj32 v10 v16 v22
    let v26 := v24 + v25
    let v27 := v23 + v26
    let v28 := maj32 v16 v22 v27
    let v29 := v17 + v28
    let v30 := v11 + v29
    0x00000001#32 &&& v30 = 0x00000001#32 &&& (s0.d ^^^ w4) := by
  intro s1 s2 s3 s4 s5 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30
  have hv10 : v10 = s1.a := by
    change (s5.e - s5.a) + ((Sigma0 s5.b) + (maj32 s5.b s5.c s5.d)) = s1.a
    exact round_backSolveA s4 K4 w4
  have hv16 : v16 = s0.a := by
    change (s5.f - s5.b) + ((Sigma0 s5.c) + (maj32 s5.c s5.d v10)) = s0.a
    rw [hv10]
    exact round_backSolveA s3 K3 w3
  have hv22 : v22 = s0.b := by
    change (s5.g - s5.c) + ((Sigma0 s5.d) + (maj32 s5.d v10 v16)) = s0.b
    rw [hv10, hv16]
    exact round_backSolveA s2 K2 w2
  have hv27 : v27 = s0.c := by
    change (s5.h - s5.d) + ((Sigma0 v10) + (maj32 v10 v16 v22)) = s0.c
    rw [hv10, hv16, hv22]
    exact round_backSolveA s1 K1 w1
  have hv7 : v7 - K4 - v10 + (Sigma0 v16 + maj32 v16 v22 v27) = s0.d + w4 := by
    have hsub : v7 - K4 = s1.e + w4 := by
      change (((s5.a - ((Sigma0 s5.b) + (maj32 s5.b s5.c s5.d))) - (Sigma1 s5.f)) - (ch32 s5.f s5.g s5.h)) - K4 = s1.e + w4
      exact round_extract_hw s4 K4 w4
    rw [hsub, hv10, hv16, hv22, hv27]
    dsimp only [s1, sha256Round]
    exact bv_five_round_sum_cancel s0.d (s0.h + Sigma1 s0.e + ch32 s0.e s0.f s0.g + K0 + w0) (Sigma0 s0.a + maj32 s0.a s0.b s0.c) w4
  change 1#32 &&& ((v7 + K4 - v10) + (Sigma0 v16 + maj32 v16 v22 v27)) = 1#32 &&& (s0.d ^^^ w4)
  exact lsb_p3_id_004_aux v7 K4 v10 (Sigma0 v16 + maj32 v16 v22 v27) w4 s0.d hv7

/-- `p3_id_005`：Track A1 5-round seed + BitEq(s0.b[1]=0) [Carry-shielded (s0.d&w4)==0 => XOR]。 -/
theorem p3_id_005 (s0 : SHAState) (w0 w1 w2 w3 w4 : BitVec 32) (hcond : s0.d &&& w4 = 0#32) :
    let s1 := sha256Round s0 K0 w0
    let s2 := sha256Round s1 K1 w1
    let s3 := sha256Round s2 K2 w2
    let s4 := sha256Round s3 K3 w3
    let s5 := sha256Round s4 K4 w4
    let v0 := Sigma0 s5.b
    let v1 := maj32 s5.b s5.c s5.d
    let v2 := v0 + v1
    let v3 := s5.a - v2
    let v4 := Sigma1 s5.f
    let v5 := v3 - v4
    let v6 := ch32 s5.f s5.g s5.h
    let v7 := v5 - v6
    let v8 := v7 - 0x3956c25b#32
    let v9 := s5.e - s5.a
    let v10 := v9 + v2
    let v11 := v8 - v10
    let v12 := s5.f - s5.b
    let v13 := Sigma0 s5.c
    let v14 := maj32 s5.c s5.d v10
    let v15 := v13 + v14
    let v16 := v12 + v15
    let v17 := Sigma0 v16
    let v18 := s5.g - s5.c
    let v19 := Sigma0 s5.d
    let v20 := maj32 s5.d v10 v16
    let v21 := v19 + v20
    let v22 := v18 + v21
    let v23 := s5.h - s5.d
    let v24 := Sigma0 v10
    let v25 := maj32 v10 v16 v22
    let v26 := v24 + v25
    let v27 := v23 + v26
    let v28 := maj32 v16 v22 v27
    let v29 := v17 + v28
    let v30 := v11 + v29
    0xffffffff#32 &&& v30 = 0xffffffff#32 &&& (s0.d ^^^ w4) := by
  intro s1 s2 s3 s4 s5 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30
  have hv10 : v10 = s1.a := by
    change (s5.e - s5.a) + ((Sigma0 s5.b) + (maj32 s5.b s5.c s5.d)) = s1.a
    exact round_backSolveA s4 K4 w4
  have hv16 : v16 = s0.a := by
    change (s5.f - s5.b) + ((Sigma0 s5.c) + (maj32 s5.c s5.d v10)) = s0.a
    rw [hv10]
    exact round_backSolveA s3 K3 w3
  have hv22 : v22 = s0.b := by
    change (s5.g - s5.c) + ((Sigma0 s5.d) + (maj32 s5.d v10 v16)) = s0.b
    rw [hv10, hv16]
    exact round_backSolveA s2 K2 w2
  have hv27 : v27 = s0.c := by
    change (s5.h - s5.d) + ((Sigma0 v10) + (maj32 v10 v16 v22)) = s0.c
    rw [hv10, hv16, hv22]
    exact round_backSolveA s1 K1 w1
  have hv30 : v30 = s0.d + w4 := by
    change (((((s5.a - ((Sigma0 s5.b) + (maj32 s5.b s5.c s5.d))) - (Sigma1 s5.f)) - (ch32 s5.f s5.g s5.h)) - 0x3956c25b#32) - v10) + ((Sigma0 v16) + (maj32 v16 v22 v27)) = s0.d + w4
    rw [round_extract_hw s4 K4 w4]
    rw [hv10, hv16, hv22, hv27]
    dsimp only [s1, sha256Round]
    exact bv_five_round_sum_cancel s0.d (s0.h + Sigma1 s0.e + ch32 s0.e s0.f s0.g + K0 + w0) (Sigma0 s0.a + maj32 s0.a s0.b s0.c) w4
  rw [hv30]
  rw [carry_shield_add_eq_xor s0.d w4 hcond]

end P3Verify.TrackA1
