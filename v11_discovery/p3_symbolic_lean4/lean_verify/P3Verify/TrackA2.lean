import P3Verify.Primitives

namespace P3Verify.TrackA2
open P3Verify

/-- `p3_id_006`：Track A2 9-round padded A_{-3} reconstruction (elim W_{0..0}+4E)。 -/
theorem p3_id_006 (s0 : SHAState) (w0 : BitVec 32) :
    let w1 : BitVec 32 := 0x80000000#32
    let w2 : BitVec 32 := 0#32
    let w3 : BitVec 32 := 0#32
    let w4 : BitVec 32 := 0#32
    let w5 : BitVec 32 := 0#32
    let w6 : BitVec 32 := 0#32
    let w7 : BitVec 32 := 0#32
    let w8 : BitVec 32 := 0#32
    let s1 := sha256Round s0 K0 w0
    let s2 := sha256Round s1 K1 w1
    let s3 := sha256Round s2 K2 w2
    let s4 := sha256Round s3 K3 w3
    let s5 := sha256Round s4 K4 w4
    let s6 := sha256Round s5 K5 w5
    let s7 := sha256Round s6 K6 w6
    let s8 := sha256Round s7 K7 w7
    let s9 := sha256Round s8 K8 w8
    let v0 := s9.e - s9.a
    let v1 := Sigma0 s9.b
    let v2 := maj32 s9.b s9.c s9.d
    let v3 := v1 + v2
    let v4 := v0 + v3
    let v5 := s9.f - s9.b
    let v6 := Sigma0 s9.c
    let v7 := maj32 s9.c s9.d v4
    let v8 := v6 + v7
    let v9 := v5 + v8
    let v10 := Sigma0 v9
    let v11 := s9.g - s9.c
    let v12 := Sigma0 s9.d
    let v13 := maj32 s9.d v4 v9
    let v14 := v12 + v13
    let v15 := v11 + v14
    let v16 := s9.h - s9.d
    let v17 := Sigma0 v4
    let v18 := maj32 v4 v9 v15
    let v19 := v17 + v18
    let v20 := v16 + v19
    let v21 := maj32 v9 v15 v20
    let v22 := v10 + v21
    let v23 := v4 - v22
    let v24 := s9.b - v8
    let v25 := Sigma1 s9.g
    let v26 := v24 - v25
    let v27 := s9.a - v3
    let v28 := Sigma1 s9.f
    let v29 := v27 - v28
    let v30 := ch32 s9.f s9.g s9.h
    let v31 := v29 - v30
    let v32 := v31 - 0xd807aa98#32
    let v33 := v32 - 0x00000000#32
    let v34 := ch32 s9.g s9.h v33
    let v35 := v26 - v34
    let v36 := v35 - 0xab1c5ed5#32
    let v37 := v36 - 0x00000000#32
    let v38 := Sigma1 v37
    let v39 := v23 - v38
    let v40 := s9.c - v14
    let v41 := Sigma1 s9.h
    let v42 := v40 - v41
    let v43 := ch32 s9.h v33 v37
    let v44 := v42 - v43
    let v45 := v44 - 0x923f82a4#32
    let v46 := v45 - 0x00000000#32
    let v47 := s9.d - v19
    let v48 := Sigma1 v33
    let v49 := v47 - v48
    let v50 := ch32 v33 v37 v46
    let v51 := v49 - v50
    let v52 := v51 - 0x59f111f1#32
    let v53 := v52 - 0x00000000#32
    let v54 := ch32 v37 v46 v53
    let v55 := v39 - v54
    let v56 := v55 - 0x3956c25b#32
    let v57 := v33 - v4
    let v58 := v57 + v22
    let v59 := v56 - v58
    let v60 := v37 - v9
    let v61 := Sigma0 v15
    let v62 := maj32 v15 v20 v58
    let v63 := v61 + v62
    let v64 := v60 + v63
    let v65 := Sigma0 v64
    let v66 := v46 - v15
    let v67 := Sigma0 v20
    let v68 := maj32 v20 v58 v64
    let v69 := v67 + v68
    let v70 := v66 + v69
    let v71 := v53 - v20
    let v72 := Sigma0 v58
    let v73 := maj32 v58 v64 v70
    let v74 := v72 + v73
    let v75 := v71 + v74
    let v76 := maj32 v64 v70 v75
    let v77 := v65 + v76
    let v78 := v59 + v77
    let v79 := v78 - 0x00000000#32
    0xffffffff#32 &&& v79 = 0xffffffff#32 &&& (s0.d) := by
  intro w1 w2 w3 w4 w5 w6 w7 w8 s1 s2 s3 s4 s5 s6 s7 s8 s9 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52 v53 v54 v55 v56 v57 v58 v59 v60 v61 v62 v63 v64 v65 v66 v67 v68 v69 v70 v71 v72 v73 v74 v75 v76 v77 v78 v79
  have hv4 : v4 = s5.a := by
    change (s9.e - s9.a) + ((Sigma0 s9.b) + (maj32 s9.b s9.c s9.d)) = s5.a
    exact round_backSolveA s8 K8 w8
  have hv9 : v9 = s4.a := by
    change (s9.f - s9.b) + ((Sigma0 s9.c) + (maj32 s9.c s9.d v4)) = s4.a
    rw [hv4]
    exact round_backSolveA s7 K7 w7
  have hv15 : v15 = s3.a := by
    change (s9.g - s9.c) + ((Sigma0 s9.d) + (maj32 s9.d v4 v9)) = s3.a
    rw [hv4, hv9]
    exact round_backSolveA s6 K6 w6
  have hv20 : v20 = s2.a := by
    change (s9.h - s9.d) + ((Sigma0 v4) + (maj32 v4 v9 v15)) = s2.a
    rw [hv4, hv9, hv15]
    exact round_backSolveA s5 K5 w5
  have hv32 : v32 = s5.e := by
    change (((s9.a - ((Sigma0 s9.b) + (maj32 s9.b s9.c s9.d))) - (Sigma1 s9.f)) - (ch32 s9.f s9.g s9.h)) - 0xd807aa98#32 = s5.e
    exact round_extract_hw_zero s8 K8
  have hv33 : v33 = s5.e := by
    change v32 - 0x00000000#32 = s5.e
    rw [hv32]
    exact bv_sub_zero _
  have hv36 : v36 = s4.e := by
    change (((s9.b - ((Sigma0 s9.c) + (maj32 s9.c s9.d v4))) - (Sigma1 s9.g)) - (ch32 s9.g s9.h v33)) - 0xab1c5ed5#32 = s4.e
    rw [hv4, hv33]
    exact round_extract_hw_zero s7 K7
  have hv37 : v37 = s4.e := by
    change v36 - 0x00000000#32 = s4.e
    rw [hv36]
    exact bv_sub_zero _
  have hv45 : v45 = s3.e := by
    change (((s9.c - ((Sigma0 s9.d) + (maj32 s9.d v4 v9))) - (Sigma1 s9.h)) - (ch32 s9.h v33 v37)) - 0x923f82a4#32 = s3.e
    rw [hv4, hv9, hv33, hv37]
    exact round_extract_hw_zero s6 K6
  have hv46 : v46 = s3.e := by
    change v45 - 0x00000000#32 = s3.e
    rw [hv45]
    exact bv_sub_zero _
  have hv52 : v52 = s2.e := by
    change (((s9.d - ((Sigma0 v4) + (maj32 v4 v9 v15))) - (Sigma1 v33)) - (ch32 v33 v37 v46)) - 0x59f111f1#32 = s2.e
    rw [hv4, hv9, hv15, hv33, hv37, hv46]
    exact round_extract_hw_zero s5 K5
  have hv53 : v53 = s2.e := by
    change v52 - 0x00000000#32 = s2.e
    rw [hv52]
    exact bv_sub_zero _
  have hv56 : v56 = s1.e := by
    change (((v4 - ((Sigma0 v9) + (maj32 v9 v15 v20))) - (Sigma1 v37)) - (ch32 v37 v46 v53)) - 0x3956c25b#32 = s1.e
    rw [hv4, hv9, hv15, hv20, hv37, hv46, hv53]
    exact round_extract_hw_zero s4 K4
  have hv58 : v58 = s1.a := by
    change (v33 - v4) + ((Sigma0 v9) + (maj32 v9 v15 v20)) = s1.a
    rw [hv33, hv4, hv9, hv15, hv20]
    exact round_backSolveA s4 K4 w4
  have hv64 : v64 = s0.a := by
    change (v37 - v9) + ((Sigma0 v15) + (maj32 v15 v20 v58)) = s0.a
    rw [hv37, hv9, hv15, hv20, hv58]
    exact round_backSolveA s3 K3 w3
  have hv70 : v70 = s0.b := by
    change (v46 - v15) + ((Sigma0 v20) + (maj32 v20 v58 v64)) = s0.b
    rw [hv46, hv15, hv20, hv58, hv64]
    exact round_backSolveA s2 K2 w2
  have hv75 : v75 = s0.c := by
    change (v53 - v20) + ((Sigma0 v58) + (maj32 v58 v64 v70)) = s0.c
    rw [hv53, hv20, hv58, hv64, hv70]
    exact round_backSolveA s1 K1 w1
  have hv78 : v78 = s0.d := by
    change (v56 - v58) + ((Sigma0 v64) + (maj32 v64 v70 v75)) = s0.d
    rw [hv56, hv58, hv64, hv70, hv75]
    exact round_backSolveA s0 K0 w0
  have hv79 : v79 = s0.d := by
    change v78 - 0x00000000#32 = s0.d
    rw [hv78]
    exact bv_sub_zero _
  rw [hv79]

/-- `p3_id_007`：Track A2 7-round padded A_{-3} reconstruction (elim W_{0..0}+4E)。 -/
theorem p3_id_007 (s0 : SHAState) (w0 : BitVec 32) :
    let w1 : BitVec 32 := 0x80000000#32
    let w2 : BitVec 32 := 0#32
    let w3 : BitVec 32 := 0#32
    let w4 : BitVec 32 := 0#32
    let w5 : BitVec 32 := 0#32
    let w6 : BitVec 32 := 0#32
    let s1 := sha256Round s0 K0 w0
    let s2 := sha256Round s1 K1 w1
    let s3 := sha256Round s2 K2 w2
    let s4 := sha256Round s3 K3 w3
    let s5 := sha256Round s4 K4 w4
    let s6 := sha256Round s5 K5 w5
    let s7 := sha256Round s6 K6 w6
    let v0 := Sigma0 s7.d
    let v1 := s7.e - s7.a
    let v2 := Sigma0 s7.b
    let v3 := maj32 s7.b s7.c s7.d
    let v4 := v2 + v3
    let v5 := v1 + v4
    let v6 := s7.f - s7.b
    let v7 := Sigma0 s7.c
    let v8 := maj32 s7.c s7.d v5
    let v9 := v7 + v8
    let v10 := v6 + v9
    let v11 := maj32 s7.d v5 v10
    let v12 := v0 + v11
    let v13 := s7.c - v12
    let v14 := Sigma1 s7.h
    let v15 := v13 - v14
    let v16 := s7.a - v4
    let v17 := Sigma1 s7.f
    let v18 := v16 - v17
    let v19 := ch32 s7.f s7.g s7.h
    let v20 := v18 - v19
    let v21 := v20 - 0x923f82a4#32
    let v22 := v21 - 0x00000000#32
    let v23 := s7.b - v9
    let v24 := Sigma1 s7.g
    let v25 := v23 - v24
    let v26 := ch32 s7.g s7.h v22
    let v27 := v25 - v26
    let v28 := v27 - 0x59f111f1#32
    let v29 := v28 - 0x00000000#32
    let v30 := ch32 s7.h v22 v29
    let v31 := v15 - v30
    let v32 := v31 - 0x3956c25b#32
    let v33 := s7.g - s7.c
    let v34 := v33 + v12
    let v35 := v32 - v34
    let v36 := s7.h - s7.d
    let v37 := Sigma0 v5
    let v38 := maj32 v5 v10 v34
    let v39 := v37 + v38
    let v40 := v36 + v39
    let v41 := Sigma0 v40
    let v42 := v22 - v5
    let v43 := Sigma0 v10
    let v44 := maj32 v10 v34 v40
    let v45 := v43 + v44
    let v46 := v42 + v45
    let v47 := v29 - v10
    let v48 := Sigma0 v34
    let v49 := maj32 v34 v40 v46
    let v50 := v48 + v49
    let v51 := v47 + v50
    let v52 := maj32 v40 v46 v51
    let v53 := v41 + v52
    let v54 := v35 + v53
    let v55 := v54 - 0x00000000#32
    0xffffffff#32 &&& v55 = 0xffffffff#32 &&& (s0.d) := by
  intro w1 w2 w3 w4 w5 w6 s1 s2 s3 s4 s5 s6 s7 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52 v53 v54 v55
  have hv5 : v5 = s3.a := by
    change (s7.e - s7.a) + ((Sigma0 s7.b) + (maj32 s7.b s7.c s7.d)) = s3.a
    exact round_backSolveA s6 K6 w6
  have hv10 : v10 = s2.a := by
    change (s7.f - s7.b) + ((Sigma0 s7.c) + (maj32 s7.c s7.d v5)) = s2.a
    rw [hv5]
    exact round_backSolveA s5 K5 w5
  have hv21 : v21 = s3.e := by
    change (((s7.a - ((Sigma0 s7.b) + (maj32 s7.b s7.c s7.d))) - (Sigma1 s7.f)) - (ch32 s7.f s7.g s7.h)) - 0x923f82a4#32 = s3.e
    exact round_extract_hw_zero s6 K6
  have hv22 : v22 = s3.e := by
    change v21 - 0x00000000#32 = s3.e
    rw [hv21]
    exact bv_sub_zero _
  have hv28 : v28 = s2.e := by
    change (((s7.b - ((Sigma0 s7.c) + (maj32 s7.c s7.d v5))) - (Sigma1 s7.g)) - (ch32 s7.g s7.h v22)) - 0x59f111f1#32 = s2.e
    rw [hv5, hv22]
    exact round_extract_hw_zero s5 K5
  have hv29 : v29 = s2.e := by
    change v28 - 0x00000000#32 = s2.e
    rw [hv28]
    exact bv_sub_zero _
  have hv32 : v32 = s1.e := by
    change (((s7.c - ((Sigma0 s7.d) + (maj32 s7.d v5 v10))) - (Sigma1 s7.h)) - (ch32 s7.h v22 v29)) - 0x3956c25b#32 = s1.e
    rw [hv5, hv10, hv22, hv29]
    exact round_extract_hw_zero s4 K4
  have hv34 : v34 = s1.a := by
    change (s7.g - s7.c) + ((Sigma0 s7.d) + (maj32 s7.d v5 v10)) = s1.a
    rw [hv5, hv10]
    exact round_backSolveA s4 K4 w4
  have hv40 : v40 = s0.a := by
    change (s7.h - s7.d) + ((Sigma0 v5) + (maj32 v5 v10 v34)) = s0.a
    rw [hv5, hv10, hv34]
    exact round_backSolveA s3 K3 w3
  have hv46 : v46 = s0.b := by
    change (v22 - v5) + ((Sigma0 v10) + (maj32 v10 v34 v40)) = s0.b
    rw [hv22, hv5, hv10, hv34, hv40]
    exact round_backSolveA s2 K2 w2
  have hv51 : v51 = s0.c := by
    change (v29 - v10) + ((Sigma0 v34) + (maj32 v34 v40 v46)) = s0.c
    rw [hv29, hv10, hv34, hv40, hv46]
    exact round_backSolveA s1 K1 w1
  have hv54 : v54 = s0.d := by
    change (v32 - v34) + ((Sigma0 v40) + (maj32 v40 v46 v51)) = s0.d
    rw [hv32, hv34, hv40, hv46, hv51]
    exact round_backSolveA s0 K0 w0
  have hv55 : v55 = s0.d := by
    change v54 - 0x00000000#32 = s0.d
    rw [hv54]
    exact bv_sub_zero _
  rw [hv55]

/-- `p3_id_008`：Track A2 8-round padded A_{-3} reconstruction (elim W_{0..0}+4E)。 -/
theorem p3_id_008 (s0 : SHAState) (w0 : BitVec 32) :
    let w1 : BitVec 32 := 0x80000000#32
    let w2 : BitVec 32 := 0#32
    let w3 : BitVec 32 := 0#32
    let w4 : BitVec 32 := 0#32
    let w5 : BitVec 32 := 0#32
    let w6 : BitVec 32 := 0#32
    let w7 : BitVec 32 := 0#32
    let s1 := sha256Round s0 K0 w0
    let s2 := sha256Round s1 K1 w1
    let s3 := sha256Round s2 K2 w2
    let s4 := sha256Round s3 K3 w3
    let s5 := sha256Round s4 K4 w4
    let s6 := sha256Round s5 K5 w5
    let s7 := sha256Round s6 K6 w6
    let s8 := sha256Round s7 K7 w7
    let v0 := s8.e - s8.a
    let v1 := Sigma0 s8.b
    let v2 := maj32 s8.b s8.c s8.d
    let v3 := v1 + v2
    let v4 := v0 + v3
    let v5 := Sigma0 v4
    let v6 := s8.f - s8.b
    let v7 := Sigma0 s8.c
    let v8 := maj32 s8.c s8.d v4
    let v9 := v7 + v8
    let v10 := v6 + v9
    let v11 := s8.g - s8.c
    let v12 := Sigma0 s8.d
    let v13 := maj32 s8.d v4 v10
    let v14 := v12 + v13
    let v15 := v11 + v14
    let v16 := maj32 v4 v10 v15
    let v17 := v5 + v16
    let v18 := s8.d - v17
    let v19 := s8.a - v3
    let v20 := Sigma1 s8.f
    let v21 := v19 - v20
    let v22 := ch32 s8.f s8.g s8.h
    let v23 := v21 - v22
    let v24 := v23 - 0xab1c5ed5#32
    let v25 := v24 - 0x00000000#32
    let v26 := Sigma1 v25
    let v27 := v18 - v26
    let v28 := s8.b - v9
    let v29 := Sigma1 s8.g
    let v30 := v28 - v29
    let v31 := ch32 s8.g s8.h v25
    let v32 := v30 - v31
    let v33 := v32 - 0x923f82a4#32
    let v34 := v33 - 0x00000000#32
    let v35 := s8.c - v14
    let v36 := Sigma1 s8.h
    let v37 := v35 - v36
    let v38 := ch32 s8.h v25 v34
    let v39 := v37 - v38
    let v40 := v39 - 0x59f111f1#32
    let v41 := v40 - 0x00000000#32
    let v42 := ch32 v25 v34 v41
    let v43 := v27 - v42
    let v44 := v43 - 0x3956c25b#32
    let v45 := s8.h - s8.d
    let v46 := v45 + v17
    let v47 := v44 - v46
    let v48 := v25 - v4
    let v49 := Sigma0 v10
    let v50 := maj32 v10 v15 v46
    let v51 := v49 + v50
    let v52 := v48 + v51
    let v53 := Sigma0 v52
    let v54 := v34 - v10
    let v55 := Sigma0 v15
    let v56 := maj32 v15 v46 v52
    let v57 := v55 + v56
    let v58 := v54 + v57
    let v59 := v41 - v15
    let v60 := Sigma0 v46
    let v61 := maj32 v46 v52 v58
    let v62 := v60 + v61
    let v63 := v59 + v62
    let v64 := maj32 v52 v58 v63
    let v65 := v53 + v64
    let v66 := v47 + v65
    let v67 := v66 - 0x00000000#32
    0xffffffff#32 &&& v67 = 0xffffffff#32 &&& (s0.d) := by
  intro w1 w2 w3 w4 w5 w6 w7 s1 s2 s3 s4 s5 s6 s7 s8 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52 v53 v54 v55 v56 v57 v58 v59 v60 v61 v62 v63 v64 v65 v66 v67
  have hv4 : v4 = s4.a := by
    change (s8.e - s8.a) + ((Sigma0 s8.b) + (maj32 s8.b s8.c s8.d)) = s4.a
    exact round_backSolveA s7 K7 w7
  have hv10 : v10 = s3.a := by
    change (s8.f - s8.b) + ((Sigma0 s8.c) + (maj32 s8.c s8.d v4)) = s3.a
    rw [hv4]
    exact round_backSolveA s6 K6 w6
  have hv15 : v15 = s2.a := by
    change (s8.g - s8.c) + ((Sigma0 s8.d) + (maj32 s8.d v4 v10)) = s2.a
    rw [hv4, hv10]
    exact round_backSolveA s5 K5 w5
  have hv24 : v24 = s4.e := by
    change (((s8.a - ((Sigma0 s8.b) + (maj32 s8.b s8.c s8.d))) - (Sigma1 s8.f)) - (ch32 s8.f s8.g s8.h)) - 0xab1c5ed5#32 = s4.e
    exact round_extract_hw_zero s7 K7
  have hv25 : v25 = s4.e := by
    change v24 - 0x00000000#32 = s4.e
    rw [hv24]
    exact bv_sub_zero _
  have hv33 : v33 = s3.e := by
    change (((s8.b - ((Sigma0 s8.c) + (maj32 s8.c s8.d v4))) - (Sigma1 s8.g)) - (ch32 s8.g s8.h v25)) - 0x923f82a4#32 = s3.e
    rw [hv4, hv25]
    exact round_extract_hw_zero s6 K6
  have hv34 : v34 = s3.e := by
    change v33 - 0x00000000#32 = s3.e
    rw [hv33]
    exact bv_sub_zero _
  have hv40 : v40 = s2.e := by
    change (((s8.c - ((Sigma0 s8.d) + (maj32 s8.d v4 v10))) - (Sigma1 s8.h)) - (ch32 s8.h v25 v34)) - 0x59f111f1#32 = s2.e
    rw [hv4, hv10, hv25, hv34]
    exact round_extract_hw_zero s5 K5
  have hv41 : v41 = s2.e := by
    change v40 - 0x00000000#32 = s2.e
    rw [hv40]
    exact bv_sub_zero _
  have hv44 : v44 = s1.e := by
    change (((s8.d - ((Sigma0 v4) + (maj32 v4 v10 v15))) - (Sigma1 v25)) - (ch32 v25 v34 v41)) - 0x3956c25b#32 = s1.e
    rw [hv4, hv10, hv15, hv25, hv34, hv41]
    exact round_extract_hw_zero s4 K4
  have hv46 : v46 = s1.a := by
    change (s8.h - s8.d) + ((Sigma0 v4) + (maj32 v4 v10 v15)) = s1.a
    rw [hv4, hv10, hv15]
    exact round_backSolveA s4 K4 w4
  have hv52 : v52 = s0.a := by
    change (v25 - v4) + ((Sigma0 v10) + (maj32 v10 v15 v46)) = s0.a
    rw [hv25, hv4, hv10, hv15, hv46]
    exact round_backSolveA s3 K3 w3
  have hv58 : v58 = s0.b := by
    change (v34 - v10) + ((Sigma0 v15) + (maj32 v15 v46 v52)) = s0.b
    rw [hv34, hv10, hv15, hv46, hv52]
    exact round_backSolveA s2 K2 w2
  have hv63 : v63 = s0.c := by
    change (v41 - v15) + ((Sigma0 v46) + (maj32 v46 v52 v58)) = s0.c
    rw [hv41, hv15, hv46, hv52, hv58]
    exact round_backSolveA s1 K1 w1
  have hv66 : v66 = s0.d := by
    change (v44 - v46) + ((Sigma0 v52) + (maj32 v52 v58 v63)) = s0.d
    rw [hv44, hv46, hv52, hv58, hv63]
    exact round_backSolveA s0 K0 w0
  have hv67 : v67 = s0.d := by
    change v66 - 0x00000000#32 = s0.d
    rw [hv66]
    exact bv_sub_zero _
  rw [hv67]

/-- `p3_id_009`：Track A2 10-round padded A_{-3} reconstruction (elim W_{0..0}+4E)。 -/
theorem p3_id_009 (s0 : SHAState) (w0 : BitVec 32) :
    let w1 : BitVec 32 := 0x80000000#32
    let w2 : BitVec 32 := 0#32
    let w3 : BitVec 32 := 0#32
    let w4 : BitVec 32 := 0#32
    let w5 : BitVec 32 := 0#32
    let w6 : BitVec 32 := 0#32
    let w7 : BitVec 32 := 0#32
    let w8 : BitVec 32 := 0#32
    let w9 : BitVec 32 := 0#32
    let s1 := sha256Round s0 K0 w0
    let s2 := sha256Round s1 K1 w1
    let s3 := sha256Round s2 K2 w2
    let s4 := sha256Round s3 K3 w3
    let s5 := sha256Round s4 K4 w4
    let s6 := sha256Round s5 K5 w5
    let s7 := sha256Round s6 K6 w6
    let s8 := sha256Round s7 K7 w7
    let s9 := sha256Round s8 K8 w8
    let s10 := sha256Round s9 K9 w9
    let v0 := s10.f - s10.b
    let v1 := Sigma0 s10.c
    let v2 := s10.e - s10.a
    let v3 := Sigma0 s10.b
    let v4 := maj32 s10.b s10.c s10.d
    let v5 := v3 + v4
    let v6 := v2 + v5
    let v7 := maj32 s10.c s10.d v6
    let v8 := v1 + v7
    let v9 := v0 + v8
    let v10 := s10.g - s10.c
    let v11 := Sigma0 s10.d
    let v12 := maj32 s10.d v6 v9
    let v13 := v11 + v12
    let v14 := v10 + v13
    let v15 := Sigma0 v14
    let v16 := s10.h - s10.d
    let v17 := Sigma0 v6
    let v18 := maj32 v6 v9 v14
    let v19 := v17 + v18
    let v20 := v16 + v19
    let v21 := s10.a - v5
    let v22 := Sigma1 s10.f
    let v23 := v21 - v22
    let v24 := ch32 s10.f s10.g s10.h
    let v25 := v23 - v24
    let v26 := v25 - 0x12835b01#32
    let v27 := v26 - 0x00000000#32
    let v28 := v27 - v6
    let v29 := Sigma0 v9
    let v30 := maj32 v9 v14 v20
    let v31 := v29 + v30
    let v32 := v28 + v31
    let v33 := maj32 v14 v20 v32
    let v34 := v15 + v33
    let v35 := v9 - v34
    let v36 := s10.c - v13
    let v37 := Sigma1 s10.h
    let v38 := v36 - v37
    let v39 := s10.b - v8
    let v40 := Sigma1 s10.g
    let v41 := v39 - v40
    let v42 := ch32 s10.g s10.h v27
    let v43 := v41 - v42
    let v44 := v43 - 0xd807aa98#32
    let v45 := v44 - 0x00000000#32
    let v46 := ch32 s10.h v27 v45
    let v47 := v38 - v46
    let v48 := v47 - 0xab1c5ed5#32
    let v49 := v48 - 0x00000000#32
    let v50 := Sigma1 v49
    let v51 := v35 - v50
    let v52 := s10.d - v19
    let v53 := Sigma1 v27
    let v54 := v52 - v53
    let v55 := ch32 v27 v45 v49
    let v56 := v54 - v55
    let v57 := v56 - 0x923f82a4#32
    let v58 := v57 - 0x00000000#32
    let v59 := v6 - v31
    let v60 := Sigma1 v45
    let v61 := v59 - v60
    let v62 := ch32 v45 v49 v58
    let v63 := v61 - v62
    let v64 := v63 - 0x59f111f1#32
    let v65 := v64 - 0x00000000#32
    let v66 := ch32 v49 v58 v65
    let v67 := v51 - v66
    let v68 := v67 - 0x3956c25b#32
    let v69 := v45 - v9
    let v70 := v69 + v34
    let v71 := v68 - v70
    let v72 := v49 - v14
    let v73 := Sigma0 v20
    let v74 := maj32 v20 v32 v70
    let v75 := v73 + v74
    let v76 := v72 + v75
    let v77 := Sigma0 v76
    let v78 := v58 - v20
    let v79 := Sigma0 v32
    let v80 := maj32 v32 v70 v76
    let v81 := v79 + v80
    let v82 := v78 + v81
    let v83 := v65 - v32
    let v84 := Sigma0 v70
    let v85 := maj32 v70 v76 v82
    let v86 := v84 + v85
    let v87 := v83 + v86
    let v88 := maj32 v76 v82 v87
    let v89 := v77 + v88
    let v90 := v71 + v89
    let v91 := v90 - 0x00000000#32
    0xffffffff#32 &&& v91 = 0xffffffff#32 &&& (s0.d) := by
  intro w1 w2 w3 w4 w5 w6 w7 w8 w9 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52 v53 v54 v55 v56 v57 v58 v59 v60 v61 v62 v63 v64 v65 v66 v67 v68 v69 v70 v71 v72 v73 v74 v75 v76 v77 v78 v79 v80 v81 v82 v83 v84 v85 v86 v87 v88 v89 v90 v91
  have hv6 : v6 = s6.a := by
    change (s10.e - s10.a) + ((Sigma0 s10.b) + (maj32 s10.b s10.c s10.d)) = s6.a
    exact round_backSolveA s9 K9 w9
  have hv9 : v9 = s5.a := by
    change (s10.f - s10.b) + ((Sigma0 s10.c) + (maj32 s10.c s10.d v6)) = s5.a
    rw [hv6]
    exact round_backSolveA s8 K8 w8
  have hv14 : v14 = s4.a := by
    change (s10.g - s10.c) + ((Sigma0 s10.d) + (maj32 s10.d v6 v9)) = s4.a
    rw [hv6, hv9]
    exact round_backSolveA s7 K7 w7
  have hv20 : v20 = s3.a := by
    change (s10.h - s10.d) + ((Sigma0 v6) + (maj32 v6 v9 v14)) = s3.a
    rw [hv6, hv9, hv14]
    exact round_backSolveA s6 K6 w6
  have hv26 : v26 = s6.e := by
    change (((s10.a - ((Sigma0 s10.b) + (maj32 s10.b s10.c s10.d))) - (Sigma1 s10.f)) - (ch32 s10.f s10.g s10.h)) - 0x12835b01#32 = s6.e
    exact round_extract_hw_zero s9 K9
  have hv27 : v27 = s6.e := by
    change v26 - 0x00000000#32 = s6.e
    rw [hv26]
    exact bv_sub_zero _
  have hv32 : v32 = s2.a := by
    change (v27 - v6) + ((Sigma0 v9) + (maj32 v9 v14 v20)) = s2.a
    rw [hv27, hv6, hv9, hv14, hv20]
    exact round_backSolveA s5 K5 w5
  have hv44 : v44 = s5.e := by
    change (((s10.b - ((Sigma0 s10.c) + (maj32 s10.c s10.d v6))) - (Sigma1 s10.g)) - (ch32 s10.g s10.h v27)) - 0xd807aa98#32 = s5.e
    rw [hv6, hv27]
    exact round_extract_hw_zero s8 K8
  have hv45 : v45 = s5.e := by
    change v44 - 0x00000000#32 = s5.e
    rw [hv44]
    exact bv_sub_zero _
  have hv48 : v48 = s4.e := by
    change (((s10.c - ((Sigma0 s10.d) + (maj32 s10.d v6 v9))) - (Sigma1 s10.h)) - (ch32 s10.h v27 v45)) - 0xab1c5ed5#32 = s4.e
    rw [hv6, hv9, hv27, hv45]
    exact round_extract_hw_zero s7 K7
  have hv49 : v49 = s4.e := by
    change v48 - 0x00000000#32 = s4.e
    rw [hv48]
    exact bv_sub_zero _
  have hv57 : v57 = s3.e := by
    change (((s10.d - ((Sigma0 v6) + (maj32 v6 v9 v14))) - (Sigma1 v27)) - (ch32 v27 v45 v49)) - 0x923f82a4#32 = s3.e
    rw [hv6, hv9, hv14, hv27, hv45, hv49]
    exact round_extract_hw_zero s6 K6
  have hv58 : v58 = s3.e := by
    change v57 - 0x00000000#32 = s3.e
    rw [hv57]
    exact bv_sub_zero _
  have hv64 : v64 = s2.e := by
    change (((v6 - ((Sigma0 v9) + (maj32 v9 v14 v20))) - (Sigma1 v45)) - (ch32 v45 v49 v58)) - 0x59f111f1#32 = s2.e
    rw [hv6, hv9, hv14, hv20, hv45, hv49, hv58]
    exact round_extract_hw_zero s5 K5
  have hv65 : v65 = s2.e := by
    change v64 - 0x00000000#32 = s2.e
    rw [hv64]
    exact bv_sub_zero _
  have hv68 : v68 = s1.e := by
    change (((v9 - ((Sigma0 v14) + (maj32 v14 v20 v32))) - (Sigma1 v49)) - (ch32 v49 v58 v65)) - 0x3956c25b#32 = s1.e
    rw [hv9, hv14, hv20, hv32, hv49, hv58, hv65]
    exact round_extract_hw_zero s4 K4
  have hv70 : v70 = s1.a := by
    change (v45 - v9) + ((Sigma0 v14) + (maj32 v14 v20 v32)) = s1.a
    rw [hv45, hv9, hv14, hv20, hv32]
    exact round_backSolveA s4 K4 w4
  have hv76 : v76 = s0.a := by
    change (v49 - v14) + ((Sigma0 v20) + (maj32 v20 v32 v70)) = s0.a
    rw [hv49, hv14, hv20, hv32, hv70]
    exact round_backSolveA s3 K3 w3
  have hv82 : v82 = s0.b := by
    change (v58 - v20) + ((Sigma0 v32) + (maj32 v32 v70 v76)) = s0.b
    rw [hv58, hv20, hv32, hv70, hv76]
    exact round_backSolveA s2 K2 w2
  have hv87 : v87 = s0.c := by
    change (v65 - v32) + ((Sigma0 v70) + (maj32 v70 v76 v82)) = s0.c
    rw [hv65, hv32, hv70, hv76, hv82]
    exact round_backSolveA s1 K1 w1
  have hv90 : v90 = s0.d := by
    change (v68 - v70) + ((Sigma0 v76) + (maj32 v76 v82 v87)) = s0.d
    rw [hv68, hv70, hv76, hv82, hv87]
    exact round_backSolveA s0 K0 w0
  have hv91 : v91 = s0.d := by
    change v90 - 0x00000000#32 = s0.d
    rw [hv90]
    exact bv_sub_zero _
  rw [hv91]

/-- `p3_id_010`：Track A2 6-round padded A_{-3} reconstruction (elim W_{0..0}+4E)。 -/
theorem p3_id_010 (s0 : SHAState) (w0 : BitVec 32) :
    let w1 : BitVec 32 := 0x80000000#32
    let w2 : BitVec 32 := 0#32
    let w3 : BitVec 32 := 0#32
    let w4 : BitVec 32 := 0#32
    let w5 : BitVec 32 := 0#32
    let s1 := sha256Round s0 K0 w0
    let s2 := sha256Round s1 K1 w1
    let s3 := sha256Round s2 K2 w2
    let s4 := sha256Round s3 K3 w3
    let s5 := sha256Round s4 K4 w4
    let s6 := sha256Round s5 K5 w5
    let v0 := Sigma0 s6.c
    let v1 := s6.e - s6.a
    let v2 := Sigma0 s6.b
    let v3 := maj32 s6.b s6.c s6.d
    let v4 := v2 + v3
    let v5 := v1 + v4
    let v6 := maj32 s6.c s6.d v5
    let v7 := v0 + v6
    let v8 := s6.b - v7
    let v9 := Sigma1 s6.g
    let v10 := v8 - v9
    let v11 := s6.a - v4
    let v12 := Sigma1 s6.f
    let v13 := v11 - v12
    let v14 := ch32 s6.f s6.g s6.h
    let v15 := v13 - v14
    let v16 := v15 - 0x59f111f1#32
    let v17 := v16 - 0x00000000#32
    let v18 := ch32 s6.g s6.h v17
    let v19 := v10 - v18
    let v20 := v19 - 0x3956c25b#32
    let v21 := s6.f - s6.b
    let v22 := v21 + v7
    let v23 := v20 - v22
    let v24 := s6.g - s6.c
    let v25 := Sigma0 s6.d
    let v26 := maj32 s6.d v5 v22
    let v27 := v25 + v26
    let v28 := v24 + v27
    let v29 := Sigma0 v28
    let v30 := s6.h - s6.d
    let v31 := Sigma0 v5
    let v32 := maj32 v5 v22 v28
    let v33 := v31 + v32
    let v34 := v30 + v33
    let v35 := v17 - v5
    let v36 := Sigma0 v22
    let v37 := maj32 v22 v28 v34
    let v38 := v36 + v37
    let v39 := v35 + v38
    let v40 := maj32 v28 v34 v39
    let v41 := v29 + v40
    let v42 := v23 + v41
    let v43 := v42 - 0x00000000#32
    0xffffffff#32 &&& v43 = 0xffffffff#32 &&& (s0.d) := by
  intro w1 w2 w3 w4 w5 s1 s2 s3 s4 s5 s6 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43
  have hv5 : v5 = s2.a := by
    change (s6.e - s6.a) + ((Sigma0 s6.b) + (maj32 s6.b s6.c s6.d)) = s2.a
    exact round_backSolveA s5 K5 w5
  have hv16 : v16 = s2.e := by
    change (((s6.a - ((Sigma0 s6.b) + (maj32 s6.b s6.c s6.d))) - (Sigma1 s6.f)) - (ch32 s6.f s6.g s6.h)) - 0x59f111f1#32 = s2.e
    exact round_extract_hw_zero s5 K5
  have hv17 : v17 = s2.e := by
    change v16 - 0x00000000#32 = s2.e
    rw [hv16]
    exact bv_sub_zero _
  have hv20 : v20 = s1.e := by
    change (((s6.b - ((Sigma0 s6.c) + (maj32 s6.c s6.d v5))) - (Sigma1 s6.g)) - (ch32 s6.g s6.h v17)) - 0x3956c25b#32 = s1.e
    rw [hv5, hv17]
    exact round_extract_hw_zero s4 K4
  have hv22 : v22 = s1.a := by
    change (s6.f - s6.b) + ((Sigma0 s6.c) + (maj32 s6.c s6.d v5)) = s1.a
    rw [hv5]
    exact round_backSolveA s4 K4 w4
  have hv28 : v28 = s0.a := by
    change (s6.g - s6.c) + ((Sigma0 s6.d) + (maj32 s6.d v5 v22)) = s0.a
    rw [hv5, hv22]
    exact round_backSolveA s3 K3 w3
  have hv34 : v34 = s0.b := by
    change (s6.h - s6.d) + ((Sigma0 v5) + (maj32 v5 v22 v28)) = s0.b
    rw [hv5, hv22, hv28]
    exact round_backSolveA s2 K2 w2
  have hv39 : v39 = s0.c := by
    change (v17 - v5) + ((Sigma0 v22) + (maj32 v22 v28 v34)) = s0.c
    rw [hv17, hv5, hv22, hv28, hv34]
    exact round_backSolveA s1 K1 w1
  have hv42 : v42 = s0.d := by
    change (v20 - v22) + ((Sigma0 v28) + (maj32 v28 v34 v39)) = s0.d
    rw [hv20, hv22, hv28, hv34, hv39]
    exact round_backSolveA s0 K0 w0
  have hv43 : v43 = s0.d := by
    change v42 - 0x00000000#32 = s0.d
    rw [hv42]
    exact bv_sub_zero _
  rw [hv43]

/-- `p3_id_011`：Track A2 5-round 1-word padded A_{-3} invariant (elim w0+E_0..-3)。 -/
theorem p3_id_011 (s0 : SHAState) (w0 : BitVec 32) :
    let w1 : BitVec 32 := 0x80000000#32
    let w2 : BitVec 32 := 0#32
    let w3 : BitVec 32 := 0#32
    let w4 : BitVec 32 := 0#32
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
    let v9 := v8 - 0x00000000#32
    let v10 := s5.e - s5.a
    let v11 := v10 + v2
    let v12 := v9 - v11
    let v13 := s5.f - s5.b
    let v14 := Sigma0 s5.c
    let v15 := maj32 s5.c s5.d v11
    let v16 := v14 + v15
    let v17 := v13 + v16
    let v18 := Sigma0 v17
    let v19 := s5.g - s5.c
    let v20 := Sigma0 s5.d
    let v21 := maj32 s5.d v11 v17
    let v22 := v20 + v21
    let v23 := v19 + v22
    let v24 := s5.h - s5.d
    let v25 := Sigma0 v11
    let v26 := maj32 v11 v17 v23
    let v27 := v25 + v26
    let v28 := v24 + v27
    let v29 := maj32 v17 v23 v28
    let v30 := v18 + v29
    let v31 := v12 + v30
    0xffffffff#32 &&& v31 = 0xffffffff#32 &&& (s0.d) := by
  intro w1 w2 w3 w4 s1 s2 s3 s4 s5 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31
  have hv8 : v8 = s1.e := by
    change (((s5.a - ((Sigma0 s5.b) + (maj32 s5.b s5.c s5.d))) - (Sigma1 s5.f)) - (ch32 s5.f s5.g s5.h)) - 0x3956c25b#32 = s1.e
    exact round_extract_hw_zero s4 K4
  have hv9 : v9 = s1.e := by
    change v8 - 0x00000000#32 = s1.e
    rw [hv8]
    exact bv_sub_zero _
  have hv11 : v11 = s1.a := by
    change (s5.e - s5.a) + ((Sigma0 s5.b) + (maj32 s5.b s5.c s5.d)) = s1.a
    exact round_backSolveA s4 K4 w4
  have hv17 : v17 = s0.a := by
    change (s5.f - s5.b) + ((Sigma0 s5.c) + (maj32 s5.c s5.d v11)) = s0.a
    rw [hv11]
    exact round_backSolveA s3 K3 w3
  have hv23 : v23 = s0.b := by
    change (s5.g - s5.c) + ((Sigma0 s5.d) + (maj32 s5.d v11 v17)) = s0.b
    rw [hv11, hv17]
    exact round_backSolveA s2 K2 w2
  have hv28 : v28 = s0.c := by
    change (s5.h - s5.d) + ((Sigma0 v11) + (maj32 v11 v17 v23)) = s0.c
    rw [hv11, hv17, hv23]
    exact round_backSolveA s1 K1 w1
  have hv31 : v31 = s0.d := by
    change (v9 - v11) + ((Sigma0 v17) + (maj32 v17 v23 v28)) = s0.d
    rw [hv9, hv11, hv17, hv23, hv28]
    exact round_backSolveA s0 K0 w0
  rw [hv31]

/-- `p3_id_012`：A2 5-round bidirectional A_{t-3}+W_{t+4} coupling (elim 4W+4E)。 -/
theorem p3_id_012 (s0 : SHAState) (w0 w1 w2 w3 w4 : BitVec 32) :
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

/-- `p3_id_013`：Track A2 5-round 1-word padded A_{-3} invariant (elim w0+E_0..-3) [mut] [Mask 1b]。 -/
theorem p3_id_013 (s0 : SHAState) (w0 : BitVec 32) :
    let w1 : BitVec 32 := 0x80000000#32
    let w2 : BitVec 32 := 0#32
    let w3 : BitVec 32 := 0#32
    let w4 : BitVec 32 := 0#32
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
    let v9 := v8 - 0x00000000#32
    let v10 := s5.e - s5.a
    let v11 := v10 + v2
    let v12 := v9 ^^^ v11
    let v13 := s5.f - s5.b
    let v14 := Sigma0 s5.c
    let v15 := maj32 s5.c s5.d v11
    let v16 := v14 + v15
    let v17 := v13 + v16
    let v18 := Sigma0 v17
    let v19 := s5.g - s5.c
    let v20 := Sigma0 s5.d
    let v21 := maj32 s5.d v11 v17
    let v22 := v20 + v21
    let v23 := v19 + v22
    let v24 := s5.h - s5.d
    let v25 := Sigma0 v11
    let v26 := maj32 v11 v17 v23
    let v27 := v25 + v26
    let v28 := v24 + v27
    let v29 := maj32 v17 v23 v28
    let v30 := v18 + v29
    let v31 := v12 + v30
    0x00000001#32 &&& v31 = 0x00000001#32 &&& (s0.d) := by
  intro w1 w2 w3 w4 s1 s2 s3 s4 s5 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31
  have hv8 : v8 = s1.e := by
    change (((s5.a - ((Sigma0 s5.b) + (maj32 s5.b s5.c s5.d))) - (Sigma1 s5.f)) - (ch32 s5.f s5.g s5.h)) - 0x3956c25b#32 = s1.e
    exact round_extract_hw_zero s4 K4
  have hv9 : v9 = s1.e := by
    change v8 - 0x00000000#32 = s1.e
    rw [hv8]
    exact bv_sub_zero _
  have hv11 : v11 = s1.a := by
    change (s5.e - s5.a) + ((Sigma0 s5.b) + (maj32 s5.b s5.c s5.d)) = s1.a
    exact round_backSolveA s4 K4 w4
  have hv17 : v17 = s0.a := by
    change (s5.f - s5.b) + ((Sigma0 s5.c) + (maj32 s5.c s5.d v11)) = s0.a
    rw [hv11]
    exact round_backSolveA s3 K3 w3
  have hv23 : v23 = s0.b := by
    change (s5.g - s5.c) + ((Sigma0 s5.d) + (maj32 s5.d v11 v17)) = s0.b
    rw [hv11, hv17]
    exact round_backSolveA s2 K2 w2
  have hv28 : v28 = s0.c := by
    change (s5.h - s5.d) + ((Sigma0 v11) + (maj32 v11 v17 v23)) = s0.c
    rw [hv11, hv17, hv23]
    exact round_backSolveA s1 K1 w1
  change 1#32 &&& ((v9 ^^^ v11) + (Sigma0 v17 + maj32 v17 v23 v28)) = 1#32 &&& s0.d
  have hfin : (s1.e - s1.a) + (Sigma0 s0.a + maj32 s0.a s0.b s0.c) = s0.d := round_backSolveA s0 K0 w0
  rw [lsb_xor_add_eq_sub_add, hv9, hv11, hv17, hv23, hv28, hfin]

end P3Verify.TrackA2
