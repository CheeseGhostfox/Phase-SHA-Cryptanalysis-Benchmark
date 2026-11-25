import P3Verify.Primitives

namespace P3Verify.TrackB1
open P3Verify

/-- `p3_id_014`：Track B1 BTC splice (r1=0, r2=8) seed [mut] [FormII B1 non-interference]。 -/
theorem p3_id_014 (b1_s0 : SHAState) (b1_e' b1_f' b1_g' b1_h' : BitVec 32) :
    let b1_s0' : SHAState := { b1_s0 with e := b1_e', f := b1_f', g := b1_g', h := b1_h' }
    let wp0 := stdIV.a + b1_s0.a
    let wp1 := stdIV.b + b1_s0.b
    let wp2 := stdIV.c + b1_s0.c
    let wp3 := stdIV.d + b1_s0.d
    let wp4 := stdIV.e + b1_s0.e
    let wp5 := stdIV.f + b1_s0.f
    let wp6 := stdIV.g + b1_s0.g
    let wp7 := stdIV.h + b1_s0.h
    let b2_s1 := sha256Round stdIV K0 wp0
    let b2_s2 := sha256Round b2_s1 K1 wp1
    let b2_s3 := sha256Round b2_s2 K2 wp2
    let b2_s4 := sha256Round b2_s3 K3 wp3
    let b2_s5 := sha256Round b2_s4 K4 wp4
    let b2_s6 := sha256Round b2_s5 K5 wp5
    let b2_s7 := sha256Round b2_s6 K6 wp6
    let b2_s8 := sha256Round b2_s7 K7 wp7
    let wp0' := stdIV.a + b1_s0'.a
    let wp1' := stdIV.b + b1_s0'.b
    let wp2' := stdIV.c + b1_s0'.c
    let wp3' := stdIV.d + b1_s0'.d
    let wp4' := stdIV.e + b1_s0'.e
    let wp5' := stdIV.f + b1_s0'.f
    let wp6' := stdIV.g + b1_s0'.g
    let wp7' := stdIV.h + b1_s0'.h
    let b2_s1' := sha256Round stdIV K0 wp0'
    let b2_s2' := sha256Round b2_s1' K1 wp1'
    let b2_s3' := sha256Round b2_s2' K2 wp2'
    let b2_s4' := sha256Round b2_s3' K3 wp3'
    let b2_s5' := sha256Round b2_s4' K4 wp4'
    let b2_s6' := sha256Round b2_s5' K5 wp5'
    let b2_s7' := sha256Round b2_s6' K6 wp6'
    let b2_s8' := sha256Round b2_s7' K7 wp7'
    (0xffffffff#32 &&& ((b2_s8.e - b2_s8.a) + (Sigma0 b2_s8.b + maj32 b2_s8.b b2_s8.c b2_s8.d))) =
      (0xffffffff#32 &&& ((b2_s8'.e - b2_s8'.a) + (Sigma0 b2_s8'.b + maj32 b2_s8'.b b2_s8'.c b2_s8'.d))) := by
  intro b1_s0' wp0 wp1 wp2 wp3 wp4 wp5 wp6 wp7 b2_s1 b2_s2 b2_s3 b2_s4 b2_s5 b2_s6 b2_s7 b2_s8 wp0' wp1' wp2' wp3' wp4' wp5' wp6' wp7' b2_s1' b2_s2' b2_s3' b2_s4' b2_s5' b2_s6' b2_s7' b2_s8'
  have hL4 : (b2_s8.e - b2_s8.a) + (Sigma0 b2_s8.b + maj32 b2_s8.b b2_s8.c b2_s8.d) = b2_s4.a := round_backSolveA b2_s7 K7 wp7
  have hR4 : (b2_s8'.e - b2_s8'.a) + (Sigma0 b2_s8'.b + maj32 b2_s8'.b b2_s8'.c b2_s8'.d) = b2_s4'.a := round_backSolveA b2_s7' K7 wp7'
  rw [hL4, hR4]

/-- `p3_id_015`：Track B1 BTC splice (r2=8): reconstruct Block1 E_end from Block2 S'_8。 -/
theorem p3_id_015 (b1_s0 : SHAState) :
    let wp0 := stdIV.a + b1_s0.a
    let wp1 := stdIV.b + b1_s0.b
    let wp2 := stdIV.c + b1_s0.c
    let wp3 := stdIV.d + b1_s0.d
    let wp4 := stdIV.e + b1_s0.e
    let wp5 := stdIV.f + b1_s0.f
    let wp6 := stdIV.g + b1_s0.g
    let wp7 := stdIV.h + b1_s0.h
    let b2_s1 := sha256Round stdIV K0 wp0
    let b2_s2 := sha256Round b2_s1 K1 wp1
    let b2_s3 := sha256Round b2_s2 K2 wp2
    let b2_s4 := sha256Round b2_s3 K3 wp3
    let b2_s5 := sha256Round b2_s4 K4 wp4
    let b2_s6 := sha256Round b2_s5 K5 wp5
    let b2_s7 := sha256Round b2_s6 K6 wp6
    let b2_s8 := sha256Round b2_s7 K7 wp7
    let v0 := b2_s8.h - b2_s8.d
    let v1 := b2_s8.e - b2_s8.a
    let v2 := Sigma0 b2_s8.b
    let v3 := maj32 b2_s8.b b2_s8.c b2_s8.d
    let v4 := v2 + v3
    let v5 := v1 + v4
    let v6 := Sigma0 v5
    let v7 := b2_s8.f - b2_s8.b
    let v8 := Sigma0 b2_s8.c
    let v9 := maj32 b2_s8.c b2_s8.d v5
    let v10 := v8 + v9
    let v11 := v7 + v10
    let v12 := b2_s8.g - b2_s8.c
    let v13 := Sigma0 b2_s8.d
    let v14 := maj32 b2_s8.d v5 v11
    let v15 := v13 + v14
    let v16 := v12 + v15
    let v17 := maj32 v5 v11 v16
    let v18 := v6 + v17
    let v19 := v0 + v18
    let v20 := b2_s8.h - v19
    let v21 := Sigma0 0x6a09e667#32
    let v22 := maj32 0x6a09e667#32 0xbb67ae85#32 0x3c6ef372#32
    let v23 := v21 + v22
    let v24 := v19 - v23
    let v25 := 0xa54ff53a#32 + v24
    let v26 := v20 - v25
    let v27 := Sigma0 v11
    let v28 := maj32 v11 v16 v19
    let v29 := v27 + v28
    let v30 := v5 - v29
    let v31 := 0x6a09e667#32 + v30
    let v32 := Sigma1 v31
    let v33 := v26 - v32
    let v34 := Sigma0 v16
    let v35 := maj32 v16 v19 0x6a09e667#32
    let v36 := v34 + v35
    let v37 := v11 - v36
    let v38 := 0xbb67ae85#32 + v37
    let v39 := Sigma0 v19
    let v40 := maj32 v19 0x6a09e667#32 0xbb67ae85#32
    let v41 := v39 + v40
    let v42 := v16 - v41
    let v43 := 0x3c6ef372#32 + v42
    let v44 := ch32 v31 v38 v43
    let v45 := v33 - v44
    let v46 := v45 - 0x3956c25b#32
    let v47 := v46 - 0x510e527f#32
    0xffffffff#32 &&& v47 = 0xffffffff#32 &&& (b1_s0.e) := by
  intro wp0 wp1 wp2 wp3 wp4 wp5 wp6 wp7 b2_s1 b2_s2 b2_s3 b2_s4 b2_s5 b2_s6 b2_s7 b2_s8 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47
  have hv5 : v5 = b2_s4.a := by
    change (b2_s8.e - b2_s8.a) + ((Sigma0 b2_s8.b) + (maj32 b2_s8.b b2_s8.c b2_s8.d)) = b2_s4.a
    exact round_backSolveA b2_s7 K7 wp7
  have hv11 : v11 = b2_s3.a := by
    change (b2_s8.f - b2_s8.b) + ((Sigma0 b2_s8.c) + (maj32 b2_s8.c b2_s8.d v5)) = b2_s3.a
    rw [hv5]
    exact round_backSolveA b2_s6 K6 wp6
  have hv16 : v16 = b2_s2.a := by
    change (b2_s8.g - b2_s8.c) + ((Sigma0 b2_s8.d) + (maj32 b2_s8.d v5 v11)) = b2_s2.a
    rw [hv5, hv11]
    exact round_backSolveA b2_s5 K5 wp5
  have hv19 : v19 = b2_s1.a := by
    change (b2_s8.h - b2_s8.d) + ((Sigma0 v5) + (maj32 v5 v11 v16)) = b2_s1.a
    rw [hv5, hv11, hv16]
    exact round_backSolveA b2_s4 K4 wp4
  have hv25 : v25 = b2_s1.e := by
    change 0xa54ff53a#32 + (v19 - ((Sigma0 0x6a09e667#32) + (maj32 0x6a09e667#32 0xbb67ae85#32 0x3c6ef372#32))) = b2_s1.e
    rw [hv19]
    exact round_bridge_e_from_a stdIV K0 wp0
  have hv31 : v31 = b2_s4.e := by
    change 0x6a09e667#32 + (v5 - ((Sigma0 v11) + (maj32 v11 v16 v19))) = b2_s4.e
    rw [hv5, hv11, hv16, hv19]
    exact round_bridge_e_from_a b2_s3 K3 wp3
  have hv38 : v38 = b2_s3.e := by
    change 0xbb67ae85#32 + (v11 - ((Sigma0 v16) + (maj32 v16 v19 0x6a09e667#32))) = b2_s3.e
    rw [hv11, hv16, hv19]
    exact round_bridge_e_from_a b2_s2 K2 wp2
  have hv43 : v43 = b2_s2.e := by
    change 0x3c6ef372#32 + (v16 - ((Sigma0 v19) + (maj32 v19 0x6a09e667#32 0xbb67ae85#32))) = b2_s2.e
    rw [hv16, hv19]
    exact round_bridge_e_from_a b2_s1 K1 wp1
  have hv46 : v46 = wp4 := by
    change ((((b2_s8.h - v19) - v25) - (Sigma1 v31)) - (ch32 v31 v38 v43)) - 0x3956c25b#32 = wp4
    rw [hv19, hv25, hv31, hv38, hv43]
    exact round_extract_w_from_e_hfirst b2_s4 K4 wp4
  have hv47 : v47 = b1_s0.e := by
    change v46 - 0x510e527f#32 = b1_s0.e
    rw [hv46]
    exact bv_splice_sub_iv stdIV.e b1_s0.e
  rw [hv47]

/-- `p3_id_016`：Track B1 BTC deep cross-splice (r1=4, r2=8): 12-round Block1 reconstruction from S'_8。 -/
theorem p3_id_016 (b1_s0 : SHAState) (b1_w0 b1_w1 b1_w2 b1_w3 : BitVec 32) :
    let b1_s1 := sha256Round b1_s0 K0 b1_w0
    let b1_s2 := sha256Round b1_s1 K1 b1_w1
    let b1_s3 := sha256Round b1_s2 K2 b1_w2
    let b1_s4 := sha256Round b1_s3 K3 b1_w3
    let wp0 := stdIV.a + b1_s4.a
    let wp1 := stdIV.b + b1_s4.b
    let wp2 := stdIV.c + b1_s4.c
    let wp3 := stdIV.d + b1_s4.d
    let wp4 := stdIV.e + b1_s4.e
    let wp5 := stdIV.f + b1_s4.f
    let wp6 := stdIV.g + b1_s4.g
    let wp7 := stdIV.h + b1_s4.h
    let b2_s1 := sha256Round stdIV K0 wp0
    let b2_s2 := sha256Round b2_s1 K1 wp1
    let b2_s3 := sha256Round b2_s2 K2 wp2
    let b2_s4 := sha256Round b2_s3 K3 wp3
    let b2_s5 := sha256Round b2_s4 K4 wp4
    let b2_s6 := sha256Round b2_s5 K5 wp5
    let b2_s7 := sha256Round b2_s6 K6 wp6
    let b2_s8 := sha256Round b2_s7 K7 wp7
    let v0 := b2_s8.e - b2_s8.a
    let v1 := Sigma0 b2_s8.b
    let v2 := maj32 b2_s8.b b2_s8.c b2_s8.d
    let v3 := v1 + v2
    let v4 := v0 + v3
    let v5 := b2_s8.e - v4
    let v6 := b2_s8.f - b2_s8.b
    let v7 := Sigma0 b2_s8.c
    let v8 := maj32 b2_s8.c b2_s8.d v4
    let v9 := v7 + v8
    let v10 := v6 + v9
    let v11 := Sigma0 v10
    let v12 := b2_s8.g - b2_s8.c
    let v13 := Sigma0 b2_s8.d
    let v14 := maj32 b2_s8.d v4 v10
    let v15 := v13 + v14
    let v16 := v12 + v15
    let v17 := b2_s8.h - b2_s8.d
    let v18 := Sigma0 v4
    let v19 := maj32 v4 v10 v16
    let v20 := v18 + v19
    let v21 := v17 + v20
    let v22 := maj32 v10 v16 v21
    let v23 := v11 + v22
    let v24 := v4 - v23
    let v25 := 0x6a09e667#32 + v24
    let v26 := v5 - v25
    let v27 := Sigma1 b2_s8.f
    let v28 := v26 - v27
    let v29 := ch32 b2_s8.f b2_s8.g b2_s8.h
    let v30 := v28 - v29
    let v31 := v30 - 0xab1c5ed5#32
    let v32 := v31 - 0x5be0cd19#32
    let v33 := v24 - 0x510e527f#32
    let v34 := Sigma0 v16
    let v35 := maj32 v16 v21 0x6a09e667#32
    let v36 := v34 + v35
    let v37 := v10 - v36
    let v38 := 0xbb67ae85#32 + v37
    let v39 := Sigma1 v38
    let v40 := v33 - v39
    let v41 := Sigma0 v21
    let v42 := maj32 v21 0x6a09e667#32 0xbb67ae85#32
    let v43 := v41 + v42
    let v44 := v16 - v43
    let v45 := 0x3c6ef372#32 + v44
    let v46 := Sigma0 0x6a09e667#32
    let v47 := maj32 0x6a09e667#32 0xbb67ae85#32 0x3c6ef372#32
    let v48 := v46 + v47
    let v49 := v21 - v48
    let v50 := 0xa54ff53a#32 + v49
    let v51 := ch32 v38 v45 v50
    let v52 := v40 - v51
    let v53 := v52 - 0xe9b5dba5#32
    let v54 := v53 - 0xa54ff53a#32
    let v55 := v32 - v54
    let v56 := b2_s8.h - v21
    let v57 := v56 - v50
    let v58 := Sigma1 v25
    let v59 := v57 - v58
    let v60 := ch32 v25 v38 v45
    let v61 := v59 - v60
    let v62 := v61 - 0x3956c25b#32
    let v63 := v62 - 0x510e527f#32
    let v64 := Sigma1 0x510e527f#32
    let v65 := 0x5be0cd19#32 + v64
    let v66 := ch32 0x510e527f#32 0x9b05688c#32 0x1f83d9ab#32
    let v67 := v65 + v66
    let v68 := v67 + 0x428a2f98#32
    let v69 := v49 - v68
    let v70 := v69 - 0x6a09e667#32
    let v71 := v63 - v70
    let v72 := v44 - 0x1f83d9ab#32
    let v73 := Sigma1 v50
    let v74 := v72 - v73
    let v75 := ch32 v50 0x510e527f#32 0x9b05688c#32
    let v76 := v74 - v75
    let v77 := v76 - 0x71374491#32
    let v78 := v77 - 0xbb67ae85#32
    let v79 := Sigma0 v78
    let v80 := v37 - 0x9b05688c#32
    let v81 := Sigma1 v45
    let v82 := v80 - v81
    let v83 := ch32 v45 v50 0x510e527f#32
    let v84 := v82 - v83
    let v85 := v84 - 0xb5c0fbcf#32
    let v86 := v85 - 0x3c6ef372#32
    let v87 := maj32 v78 v86 v54
    let v88 := v79 + v87
    let v89 := v71 + v88
    let v90 := Sigma0 v89
    let v91 := b2_s8.g - v16
    let v92 := v91 - v45
    let v93 := Sigma1 b2_s8.h
    let v94 := v92 - v93
    let v95 := ch32 b2_s8.h v25 v38
    let v96 := v94 - v95
    let v97 := v96 - 0x59f111f1#32
    let v98 := v97 - 0x9b05688c#32
    let v99 := v98 - v78
    let v100 := Sigma0 v86
    let v101 := maj32 v86 v54 v89
    let v102 := v100 + v101
    let v103 := v99 + v102
    let v104 := b2_s8.f - v10
    let v105 := v104 - v38
    let v106 := Sigma1 b2_s8.g
    let v107 := v105 - v106
    let v108 := ch32 b2_s8.g b2_s8.h v25
    let v109 := v107 - v108
    let v110 := v109 - 0x923f82a4#32
    let v111 := v110 - 0x1f83d9ab#32
    let v112 := v111 - v86
    let v113 := Sigma0 v54
    let v114 := maj32 v54 v89 v103
    let v115 := v113 + v114
    let v116 := v112 + v115
    let v117 := maj32 v89 v103 v116
    let v118 := v90 + v117
    let v119 := v55 + v118
    0xffffffff#32 &&& v119 = 0xffffffff#32 &&& (b1_s0.d) := by
  intro b1_s1 b1_s2 b1_s3 b1_s4 wp0 wp1 wp2 wp3 wp4 wp5 wp6 wp7 b2_s1 b2_s2 b2_s3 b2_s4 b2_s5 b2_s6 b2_s7 b2_s8 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52 v53 v54 v55 v56 v57 v58 v59 v60 v61 v62 v63 v64 v65 v66 v67 v68 v69 v70 v71 v72 v73 v74 v75 v76 v77 v78 v79 v80 v81 v82 v83 v84 v85 v86 v87 v88 v89 v90 v91 v92 v93 v94 v95 v96 v97 v98 v99 v100 v101 v102 v103 v104 v105 v106 v107 v108 v109 v110 v111 v112 v113 v114 v115 v116 v117 v118 v119
  have hv4 : v4 = b2_s4.a := by
    change (b2_s8.e - b2_s8.a) + ((Sigma0 b2_s8.b) + (maj32 b2_s8.b b2_s8.c b2_s8.d)) = b2_s4.a
    exact round_backSolveA b2_s7 K7 wp7
  have hv10 : v10 = b2_s3.a := by
    change (b2_s8.f - b2_s8.b) + ((Sigma0 b2_s8.c) + (maj32 b2_s8.c b2_s8.d v4)) = b2_s3.a
    rw [hv4]
    exact round_backSolveA b2_s6 K6 wp6
  have hv16 : v16 = b2_s2.a := by
    change (b2_s8.g - b2_s8.c) + ((Sigma0 b2_s8.d) + (maj32 b2_s8.d v4 v10)) = b2_s2.a
    rw [hv4, hv10]
    exact round_backSolveA b2_s5 K5 wp5
  have hv21 : v21 = b2_s1.a := by
    change (b2_s8.h - b2_s8.d) + ((Sigma0 v4) + (maj32 v4 v10 v16)) = b2_s1.a
    rw [hv4, hv10, hv16]
    exact round_backSolveA b2_s4 K4 wp4
  have hv25 : v25 = b2_s4.e := by
    change 0x6a09e667#32 + (v4 - ((Sigma0 v10) + (maj32 v10 v16 v21))) = b2_s4.e
    rw [hv4, hv10, hv16, hv21]
    exact round_bridge_e_from_a b2_s3 K3 wp3
  have hv31 : v31 = wp7 := by
    change ((((b2_s8.e - v4) - v25) - (Sigma1 b2_s8.f)) - (ch32 b2_s8.f b2_s8.g b2_s8.h)) - 0xab1c5ed5#32 = wp7
    rw [hv4, hv25]
    exact round_extract_w_from_e_hfirst b2_s7 K7 wp7
  have hv32 : v32 = b1_s1.e := by
    change v31 - 0x5be0cd19#32 = b1_s1.e
    rw [hv31]
    exact bv_splice_sub_iv stdIV.h b1_s1.e
  have hv38 : v38 = b2_s3.e := by
    change 0xbb67ae85#32 + (v10 - ((Sigma0 v16) + (maj32 v16 v21 0x6a09e667#32))) = b2_s3.e
    rw [hv10, hv16, hv21]
    exact round_bridge_e_from_a b2_s2 K2 wp2
  have hv45 : v45 = b2_s2.e := by
    change 0x3c6ef372#32 + (v16 - ((Sigma0 v21) + (maj32 v21 0x6a09e667#32 0xbb67ae85#32))) = b2_s2.e
    rw [hv16, hv21]
    exact round_bridge_e_from_a b2_s1 K1 wp1
  have hv50 : v50 = b2_s1.e := by
    change 0xa54ff53a#32 + (v21 - ((Sigma0 0x6a09e667#32) + (maj32 0x6a09e667#32 0xbb67ae85#32 0x3c6ef372#32))) = b2_s1.e
    rw [hv21]
    exact round_bridge_e_from_a stdIV K0 wp0
  have hv53 : v53 = wp3 := by
    change ((((v4 - ((Sigma0 v10) + (maj32 v10 v16 v21))) - 0x510e527f#32) - (Sigma1 v38)) - (ch32 v38 v45 v50)) - 0xe9b5dba5#32 = wp3
    rw [hv4, hv10, hv16, hv21, hv38, hv45, hv50]
    exact round_extract_w_from_a_hfirst b2_s3 K3 wp3
  have hv54 : v54 = b1_s1.a := by
    change v53 - 0xa54ff53a#32 = b1_s1.a
    rw [hv53]
    exact bv_splice_sub_iv stdIV.d b1_s1.a
  have hv62 : v62 = wp4 := by
    change ((((b2_s8.h - v21) - v50) - (Sigma1 v25)) - (ch32 v25 v38 v45)) - 0x3956c25b#32 = wp4
    rw [hv21, hv50, hv25, hv38, hv45]
    exact round_extract_w_from_e_hfirst b2_s4 K4 wp4
  have hv63 : v63 = b1_s4.e := by
    change v62 - 0x510e527f#32 = b1_s4.e
    rw [hv62]
    exact bv_splice_sub_iv stdIV.e b1_s4.e
  have hv69 : v69 = wp0 := by
    change (v21 - ((Sigma0 0x6a09e667#32) + (maj32 0x6a09e667#32 0xbb67ae85#32 0x3c6ef372#32))) - (((0x5be0cd19#32 + (Sigma1 0x510e527f#32)) + (ch32 0x510e527f#32 0x9b05688c#32 0x1f83d9ab#32)) + 0x428a2f98#32) = wp0
    rw [hv21]
    exact round_extract_w_from_a_r0 stdIV K0 wp0
  have hv70 : v70 = b1_s4.a := by
    change v69 - 0x6a09e667#32 = b1_s4.a
    rw [hv69]
    exact bv_splice_sub_iv stdIV.a b1_s4.a
  have hv77 : v77 = wp1 := by
    change ((((v16 - ((Sigma0 v21) + (maj32 v21 0x6a09e667#32 0xbb67ae85#32))) - 0x1f83d9ab#32) - (Sigma1 v50)) - (ch32 v50 0x510e527f#32 0x9b05688c#32)) - 0x71374491#32 = wp1
    rw [hv16, hv21, hv50]
    exact round_extract_w_from_a_hfirst b2_s1 K1 wp1
  have hv78 : v78 = b1_s3.a := by
    change v77 - 0xbb67ae85#32 = b1_s3.a
    rw [hv77]
    exact bv_splice_sub_iv stdIV.b b1_s3.a
  have hv85 : v85 = wp2 := by
    change ((((v10 - ((Sigma0 v16) + (maj32 v16 v21 0x6a09e667#32))) - 0x9b05688c#32) - (Sigma1 v45)) - (ch32 v45 v50 0x510e527f#32)) - 0xb5c0fbcf#32 = wp2
    rw [hv10, hv16, hv21, hv45, hv50]
    exact round_extract_w_from_a_hfirst b2_s2 K2 wp2
  have hv86 : v86 = b1_s2.a := by
    change v85 - 0x3c6ef372#32 = b1_s2.a
    rw [hv85]
    exact bv_splice_sub_iv stdIV.c b1_s2.a
  have hv89 : v89 = b1_s0.a := by
    change (v63 - v70) + ((Sigma0 v78) + (maj32 v78 v86 v54)) = b1_s0.a
    rw [hv63, hv70, hv78, hv86, hv54]
    exact round_backSolveA b1_s3 K3 b1_w3
  have hv97 : v97 = wp5 := by
    change ((((b2_s8.g - v16) - v45) - (Sigma1 b2_s8.h)) - (ch32 b2_s8.h v25 v38)) - 0x59f111f1#32 = wp5
    rw [hv16, hv45, hv25, hv38]
    exact round_extract_w_from_e_hfirst b2_s5 K5 wp5
  have hv98 : v98 = b1_s3.e := by
    change v97 - 0x9b05688c#32 = b1_s3.e
    rw [hv97]
    exact bv_splice_sub_iv stdIV.f b1_s3.e
  have hv103 : v103 = b1_s0.b := by
    change (v98 - v78) + ((Sigma0 v86) + (maj32 v86 v54 v89)) = b1_s0.b
    rw [hv98, hv78, hv86, hv54, hv89]
    exact round_backSolveA b1_s2 K2 b1_w2
  have hv110 : v110 = wp6 := by
    change ((((b2_s8.f - v10) - v38) - (Sigma1 b2_s8.g)) - (ch32 b2_s8.g b2_s8.h v25)) - 0x923f82a4#32 = wp6
    rw [hv10, hv38, hv25]
    exact round_extract_w_from_e_hfirst b2_s6 K6 wp6
  have hv111 : v111 = b1_s2.e := by
    change v110 - 0x1f83d9ab#32 = b1_s2.e
    rw [hv110]
    exact bv_splice_sub_iv stdIV.g b1_s2.e
  have hv116 : v116 = b1_s0.c := by
    change (v111 - v86) + ((Sigma0 v54) + (maj32 v54 v89 v103)) = b1_s0.c
    rw [hv111, hv86, hv54, hv89, hv103]
    exact round_backSolveA b1_s1 K1 b1_w1
  have hv119 : v119 = b1_s0.d := by
    change (v32 - v54) + ((Sigma0 v89) + (maj32 v89 v103 v116)) = b1_s0.d
    rw [hv32, hv54, hv89, hv103, hv116]
    exact round_backSolveA b1_s0 K0 b1_w0
  rw [hv119]

/-- `p3_id_017`：Track B1 BTC splice (r2=7): reconstruct Block1 E_end from Block2 S'_7。 -/
theorem p3_id_017 (b1_s0 : SHAState) :
    let wp0 := stdIV.a + b1_s0.a
    let wp1 := stdIV.b + b1_s0.b
    let wp2 := stdIV.c + b1_s0.c
    let wp3 := stdIV.d + b1_s0.d
    let wp4 := stdIV.e + b1_s0.e
    let wp5 := stdIV.f + b1_s0.f
    let wp6 := stdIV.g + b1_s0.g
    let wp7 := stdIV.h + b1_s0.h
    let b2_s1 := sha256Round stdIV K0 wp0
    let b2_s2 := sha256Round b2_s1 K1 wp1
    let b2_s3 := sha256Round b2_s2 K2 wp2
    let b2_s4 := sha256Round b2_s3 K3 wp3
    let b2_s5 := sha256Round b2_s4 K4 wp4
    let b2_s6 := sha256Round b2_s5 K5 wp5
    let b2_s7 := sha256Round b2_s6 K6 wp6
    let v0 := b2_s7.g - b2_s7.c
    let v1 := Sigma0 b2_s7.d
    let v2 := b2_s7.e - b2_s7.a
    let v3 := Sigma0 b2_s7.b
    let v4 := maj32 b2_s7.b b2_s7.c b2_s7.d
    let v5 := v3 + v4
    let v6 := v2 + v5
    let v7 := b2_s7.f - b2_s7.b
    let v8 := Sigma0 b2_s7.c
    let v9 := maj32 b2_s7.c b2_s7.d v6
    let v10 := v8 + v9
    let v11 := v7 + v10
    let v12 := maj32 b2_s7.d v6 v11
    let v13 := v1 + v12
    let v14 := v0 + v13
    let v15 := b2_s7.g - v14
    let v16 := Sigma0 0x6a09e667#32
    let v17 := maj32 0x6a09e667#32 0xbb67ae85#32 0x3c6ef372#32
    let v18 := v16 + v17
    let v19 := v14 - v18
    let v20 := 0xa54ff53a#32 + v19
    let v21 := v15 - v20
    let v22 := Sigma1 b2_s7.h
    let v23 := v21 - v22
    let v24 := Sigma0 v11
    let v25 := maj32 v11 v14 0x6a09e667#32
    let v26 := v24 + v25
    let v27 := v6 - v26
    let v28 := 0xbb67ae85#32 + v27
    let v29 := Sigma0 v14
    let v30 := maj32 v14 0x6a09e667#32 0xbb67ae85#32
    let v31 := v29 + v30
    let v32 := v11 - v31
    let v33 := 0x3c6ef372#32 + v32
    let v34 := ch32 b2_s7.h v28 v33
    let v35 := v23 - v34
    let v36 := v35 - 0x3956c25b#32
    let v37 := v36 - 0x510e527f#32
    0xffffffff#32 &&& v37 = 0xffffffff#32 &&& (b1_s0.e) := by
  intro wp0 wp1 wp2 wp3 wp4 wp5 wp6 wp7 b2_s1 b2_s2 b2_s3 b2_s4 b2_s5 b2_s6 b2_s7 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37
  have hv6 : v6 = b2_s3.a := by
    change (b2_s7.e - b2_s7.a) + ((Sigma0 b2_s7.b) + (maj32 b2_s7.b b2_s7.c b2_s7.d)) = b2_s3.a
    exact round_backSolveA b2_s6 K6 wp6
  have hv11 : v11 = b2_s2.a := by
    change (b2_s7.f - b2_s7.b) + ((Sigma0 b2_s7.c) + (maj32 b2_s7.c b2_s7.d v6)) = b2_s2.a
    rw [hv6]
    exact round_backSolveA b2_s5 K5 wp5
  have hv14 : v14 = b2_s1.a := by
    change (b2_s7.g - b2_s7.c) + ((Sigma0 b2_s7.d) + (maj32 b2_s7.d v6 v11)) = b2_s1.a
    rw [hv6, hv11]
    exact round_backSolveA b2_s4 K4 wp4
  have hv20 : v20 = b2_s1.e := by
    change 0xa54ff53a#32 + (v14 - ((Sigma0 0x6a09e667#32) + (maj32 0x6a09e667#32 0xbb67ae85#32 0x3c6ef372#32))) = b2_s1.e
    rw [hv14]
    exact round_bridge_e_from_a stdIV K0 wp0
  have hv28 : v28 = b2_s3.e := by
    change 0xbb67ae85#32 + (v6 - ((Sigma0 v11) + (maj32 v11 v14 0x6a09e667#32))) = b2_s3.e
    rw [hv6, hv11, hv14]
    exact round_bridge_e_from_a b2_s2 K2 wp2
  have hv33 : v33 = b2_s2.e := by
    change 0x3c6ef372#32 + (v11 - ((Sigma0 v14) + (maj32 v14 0x6a09e667#32 0xbb67ae85#32))) = b2_s2.e
    rw [hv11, hv14]
    exact round_bridge_e_from_a b2_s1 K1 wp1
  have hv36 : v36 = wp4 := by
    change ((((b2_s7.g - v14) - v20) - (Sigma1 b2_s7.h)) - (ch32 b2_s7.h v28 v33)) - 0x3956c25b#32 = wp4
    rw [hv14, hv20, hv28, hv33]
    exact round_extract_w_from_e_hfirst b2_s4 K4 wp4
  have hv37 : v37 = b1_s0.e := by
    change v36 - 0x510e527f#32 = b1_s0.e
    rw [hv36]
    exact bv_splice_sub_iv stdIV.e b1_s0.e
  rw [hv37]

/-- `p3_id_018`：Track B1 BTC splice (r2=6): reconstruct Block1 A_{end-3} from Block2 S'_6。 -/
theorem p3_id_018 (b1_s0 : SHAState) :
    let wp0 := stdIV.a + b1_s0.a
    let wp1 := stdIV.b + b1_s0.b
    let wp2 := stdIV.c + b1_s0.c
    let wp3 := stdIV.d + b1_s0.d
    let wp4 := stdIV.e + b1_s0.e
    let wp5 := stdIV.f + b1_s0.f
    let wp6 := stdIV.g + b1_s0.g
    let wp7 := stdIV.h + b1_s0.h
    let b2_s1 := sha256Round stdIV K0 wp0
    let b2_s2 := sha256Round b2_s1 K1 wp1
    let b2_s3 := sha256Round b2_s2 K2 wp2
    let b2_s4 := sha256Round b2_s3 K3 wp3
    let b2_s5 := sha256Round b2_s4 K4 wp4
    let b2_s6 := sha256Round b2_s5 K5 wp5
    let v0 := Sigma0 b2_s6.d
    let v1 := b2_s6.e - b2_s6.a
    let v2 := Sigma0 b2_s6.b
    let v3 := maj32 b2_s6.b b2_s6.c b2_s6.d
    let v4 := v2 + v3
    let v5 := v1 + v4
    let v6 := b2_s6.f - b2_s6.b
    let v7 := Sigma0 b2_s6.c
    let v8 := maj32 b2_s6.c b2_s6.d v5
    let v9 := v7 + v8
    let v10 := v6 + v9
    let v11 := maj32 b2_s6.d v5 v10
    let v12 := v0 + v11
    let v13 := b2_s6.c - v12
    let v14 := v13 - 0x510e527f#32
    let v15 := Sigma1 b2_s6.h
    let v16 := v14 - v15
    let v17 := Sigma0 v10
    let v18 := maj32 v10 0x6a09e667#32 0xbb67ae85#32
    let v19 := v17 + v18
    let v20 := v5 - v19
    let v21 := 0x3c6ef372#32 + v20
    let v22 := Sigma0 0x6a09e667#32
    let v23 := maj32 0x6a09e667#32 0xbb67ae85#32 0x3c6ef372#32
    let v24 := v22 + v23
    let v25 := v10 - v24
    let v26 := 0xa54ff53a#32 + v25
    let v27 := ch32 b2_s6.h v21 v26
    let v28 := v16 - v27
    let v29 := v28 - 0xe9b5dba5#32
    let v30 := v29 - 0xa54ff53a#32
    0xffffffff#32 &&& v30 = 0xffffffff#32 &&& (b1_s0.d) := by
  intro wp0 wp1 wp2 wp3 wp4 wp5 wp6 wp7 b2_s1 b2_s2 b2_s3 b2_s4 b2_s5 b2_s6 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30
  have hv5 : v5 = b2_s2.a := by
    change (b2_s6.e - b2_s6.a) + ((Sigma0 b2_s6.b) + (maj32 b2_s6.b b2_s6.c b2_s6.d)) = b2_s2.a
    exact round_backSolveA b2_s5 K5 wp5
  have hv10 : v10 = b2_s1.a := by
    change (b2_s6.f - b2_s6.b) + ((Sigma0 b2_s6.c) + (maj32 b2_s6.c b2_s6.d v5)) = b2_s1.a
    rw [hv5]
    exact round_backSolveA b2_s4 K4 wp4
  have hv21 : v21 = b2_s2.e := by
    change 0x3c6ef372#32 + (v5 - ((Sigma0 v10) + (maj32 v10 0x6a09e667#32 0xbb67ae85#32))) = b2_s2.e
    rw [hv5, hv10]
    exact round_bridge_e_from_a b2_s1 K1 wp1
  have hv26 : v26 = b2_s1.e := by
    change 0xa54ff53a#32 + (v10 - ((Sigma0 0x6a09e667#32) + (maj32 0x6a09e667#32 0xbb67ae85#32 0x3c6ef372#32))) = b2_s1.e
    rw [hv10]
    exact round_bridge_e_from_a stdIV K0 wp0
  have hv29 : v29 = wp3 := by
    change ((((b2_s6.c - ((Sigma0 b2_s6.d) + (maj32 b2_s6.d v5 v10))) - 0x510e527f#32) - (Sigma1 b2_s6.h)) - (ch32 b2_s6.h v21 v26)) - 0xe9b5dba5#32 = wp3
    rw [hv5, hv10, hv21, hv26]
    exact round_extract_w_from_a_hfirst b2_s3 K3 wp3
  have hv30 : v30 = b1_s0.d := by
    change v29 - 0xa54ff53a#32 = b1_s0.d
    rw [hv29]
    exact bv_splice_sub_iv stdIV.d b1_s0.d
  rw [hv30]

/-- `p3_id_019`：Track B1 BTC cross-splice (r1=1, r2=5): reconstruct Block1 A_{-3} from S'_5 (elim b1_w0, b1_E)。 -/
theorem p3_id_019 (b1_s0 : SHAState) (b1_w0 : BitVec 32) :
    let b1_s1 := sha256Round b1_s0 K0 b1_w0
    let wp0 := stdIV.a + b1_s1.a
    let wp1 := stdIV.b + b1_s1.b
    let wp2 := stdIV.c + b1_s1.c
    let wp3 := stdIV.d + b1_s1.d
    let wp4 := stdIV.e + b1_s1.e
    let wp5 := stdIV.f + b1_s1.f
    let wp6 := stdIV.g + b1_s1.g
    let wp7 := stdIV.h + b1_s1.h
    let b2_s1 := sha256Round stdIV K0 wp0
    let b2_s2 := sha256Round b2_s1 K1 wp1
    let b2_s3 := sha256Round b2_s2 K2 wp2
    let b2_s4 := sha256Round b2_s3 K3 wp3
    let b2_s5 := sha256Round b2_s4 K4 wp4
    let v0 := b2_s5.e - b2_s5.a
    let v1 := Sigma0 b2_s5.b
    let v2 := maj32 b2_s5.b b2_s5.c b2_s5.d
    let v3 := v1 + v2
    let v4 := v0 + v3
    let v5 := b2_s5.e - v4
    let v6 := Sigma0 0x6a09e667#32
    let v7 := maj32 0x6a09e667#32 0xbb67ae85#32 0x3c6ef372#32
    let v8 := v6 + v7
    let v9 := v4 - v8
    let v10 := 0xa54ff53a#32 + v9
    let v11 := v5 - v10
    let v12 := Sigma1 b2_s5.f
    let v13 := v11 - v12
    let v14 := ch32 b2_s5.f b2_s5.g b2_s5.h
    let v15 := v13 - v14
    let v16 := v15 - 0x3956c25b#32
    let v17 := v16 - 0x510e527f#32
    let v18 := Sigma1 0x510e527f#32
    let v19 := 0x5be0cd19#32 + v18
    let v20 := ch32 0x510e527f#32 0x9b05688c#32 0x1f83d9ab#32
    let v21 := v19 + v20
    let v22 := v21 + 0x428a2f98#32
    let v23 := v9 - v22
    let v24 := v23 - 0x6a09e667#32
    let v25 := v17 - v24
    let v26 := Sigma0 v4
    let v27 := maj32 v4 0x6a09e667#32 0xbb67ae85#32
    let v28 := v26 + v27
    let v29 := b2_s5.d - v28
    let v30 := v29 - 0x1f83d9ab#32
    let v31 := Sigma1 v10
    let v32 := v30 - v31
    let v33 := ch32 v10 0x510e527f#32 0x9b05688c#32
    let v34 := v32 - v33
    let v35 := v34 - 0x71374491#32
    let v36 := v35 - 0xbb67ae85#32
    let v37 := Sigma0 v36
    let v38 := Sigma0 b2_s5.d
    let v39 := maj32 b2_s5.d v4 0x6a09e667#32
    let v40 := v38 + v39
    let v41 := b2_s5.c - v40
    let v42 := v41 - 0x9b05688c#32
    let v43 := Sigma1 b2_s5.h
    let v44 := v42 - v43
    let v45 := ch32 b2_s5.h v10 0x510e527f#32
    let v46 := v44 - v45
    let v47 := v46 - 0xb5c0fbcf#32
    let v48 := v47 - 0x3c6ef372#32
    let v49 := Sigma0 b2_s5.c
    let v50 := maj32 b2_s5.c b2_s5.d v4
    let v51 := v49 + v50
    let v52 := b2_s5.b - v51
    let v53 := v52 - 0x510e527f#32
    let v54 := Sigma1 b2_s5.g
    let v55 := v53 - v54
    let v56 := ch32 b2_s5.g b2_s5.h v10
    let v57 := v55 - v56
    let v58 := v57 - 0xe9b5dba5#32
    let v59 := v58 - 0xa54ff53a#32
    let v60 := maj32 v36 v48 v59
    let v61 := v37 + v60
    let v62 := v25 + v61
    0xffffffff#32 &&& v62 = 0xffffffff#32 &&& (b1_s0.d) := by
  intro b1_s1 wp0 wp1 wp2 wp3 wp4 wp5 wp6 wp7 b2_s1 b2_s2 b2_s3 b2_s4 b2_s5 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52 v53 v54 v55 v56 v57 v58 v59 v60 v61 v62
  have hv4 : v4 = b2_s1.a := by
    change (b2_s5.e - b2_s5.a) + ((Sigma0 b2_s5.b) + (maj32 b2_s5.b b2_s5.c b2_s5.d)) = b2_s1.a
    exact round_backSolveA b2_s4 K4 wp4
  have hv10 : v10 = b2_s1.e := by
    change 0xa54ff53a#32 + (v4 - ((Sigma0 0x6a09e667#32) + (maj32 0x6a09e667#32 0xbb67ae85#32 0x3c6ef372#32))) = b2_s1.e
    rw [hv4]
    exact round_bridge_e_from_a stdIV K0 wp0
  have hv16 : v16 = wp4 := by
    change ((((b2_s5.e - v4) - v10) - (Sigma1 b2_s5.f)) - (ch32 b2_s5.f b2_s5.g b2_s5.h)) - 0x3956c25b#32 = wp4
    rw [hv4, hv10]
    exact round_extract_w_from_e_hfirst b2_s4 K4 wp4
  have hv17 : v17 = b1_s1.e := by
    change v16 - 0x510e527f#32 = b1_s1.e
    rw [hv16]
    exact bv_splice_sub_iv stdIV.e b1_s1.e
  have hv23 : v23 = wp0 := by
    change (v4 - ((Sigma0 0x6a09e667#32) + (maj32 0x6a09e667#32 0xbb67ae85#32 0x3c6ef372#32))) - (((0x5be0cd19#32 + (Sigma1 0x510e527f#32)) + (ch32 0x510e527f#32 0x9b05688c#32 0x1f83d9ab#32)) + 0x428a2f98#32) = wp0
    rw [hv4]
    exact round_extract_w_from_a_r0 stdIV K0 wp0
  have hv24 : v24 = b1_s1.a := by
    change v23 - 0x6a09e667#32 = b1_s1.a
    rw [hv23]
    exact bv_splice_sub_iv stdIV.a b1_s1.a
  have hv35 : v35 = wp1 := by
    change ((((b2_s5.d - ((Sigma0 v4) + (maj32 v4 0x6a09e667#32 0xbb67ae85#32))) - 0x1f83d9ab#32) - (Sigma1 v10)) - (ch32 v10 0x510e527f#32 0x9b05688c#32)) - 0x71374491#32 = wp1
    rw [hv4, hv10]
    exact round_extract_w_from_a_hfirst b2_s1 K1 wp1
  have hv36 : v36 = b1_s0.a := by
    change v35 - 0xbb67ae85#32 = b1_s0.a
    rw [hv35]
    exact bv_splice_sub_iv stdIV.b b1_s0.a
  have hv47 : v47 = wp2 := by
    change ((((b2_s5.c - ((Sigma0 b2_s5.d) + (maj32 b2_s5.d v4 0x6a09e667#32))) - 0x9b05688c#32) - (Sigma1 b2_s5.h)) - (ch32 b2_s5.h v10 0x510e527f#32)) - 0xb5c0fbcf#32 = wp2
    rw [hv4, hv10]
    exact round_extract_w_from_a_hfirst b2_s2 K2 wp2
  have hv48 : v48 = b1_s0.b := by
    change v47 - 0x3c6ef372#32 = b1_s0.b
    rw [hv47]
    exact bv_splice_sub_iv stdIV.c b1_s0.b
  have hv58 : v58 = wp3 := by
    change ((((b2_s5.b - ((Sigma0 b2_s5.c) + (maj32 b2_s5.c b2_s5.d v4))) - 0x510e527f#32) - (Sigma1 b2_s5.g)) - (ch32 b2_s5.g b2_s5.h v10)) - 0xe9b5dba5#32 = wp3
    rw [hv4, hv10]
    exact round_extract_w_from_a_hfirst b2_s3 K3 wp3
  have hv59 : v59 = b1_s0.c := by
    change v58 - 0xa54ff53a#32 = b1_s0.c
    rw [hv58]
    exact bv_splice_sub_iv stdIV.d b1_s0.c
  have hv62 : v62 = b1_s0.d := by
    change (v17 - v24) + ((Sigma0 v36) + (maj32 v36 v48 v59)) = b1_s0.d
    rw [hv17, hv24, hv36, hv48, hv59]
    exact round_backSolveA b1_s0 K0 b1_w0
  rw [hv62]

/-- `p3_id_020`：Track B1 BTC cross-splice (r1=1, r2=6): reconstruct Block1 A_{-3} from S'_6 (elim b1_w0, b1_E)。 -/
theorem p3_id_020 (b1_s0 : SHAState) (b1_w0 : BitVec 32) :
    let b1_s1 := sha256Round b1_s0 K0 b1_w0
    let wp0 := stdIV.a + b1_s1.a
    let wp1 := stdIV.b + b1_s1.b
    let wp2 := stdIV.c + b1_s1.c
    let wp3 := stdIV.d + b1_s1.d
    let wp4 := stdIV.e + b1_s1.e
    let wp5 := stdIV.f + b1_s1.f
    let wp6 := stdIV.g + b1_s1.g
    let wp7 := stdIV.h + b1_s1.h
    let b2_s1 := sha256Round stdIV K0 wp0
    let b2_s2 := sha256Round b2_s1 K1 wp1
    let b2_s3 := sha256Round b2_s2 K2 wp2
    let b2_s4 := sha256Round b2_s3 K3 wp3
    let b2_s5 := sha256Round b2_s4 K4 wp4
    let b2_s6 := sha256Round b2_s5 K5 wp5
    let v0 := b2_s6.f - b2_s6.b
    let v1 := Sigma0 b2_s6.c
    let v2 := b2_s6.e - b2_s6.a
    let v3 := Sigma0 b2_s6.b
    let v4 := maj32 b2_s6.b b2_s6.c b2_s6.d
    let v5 := v3 + v4
    let v6 := v2 + v5
    let v7 := maj32 b2_s6.c b2_s6.d v6
    let v8 := v1 + v7
    let v9 := v0 + v8
    let v10 := b2_s6.f - v9
    let v11 := Sigma0 0x6a09e667#32
    let v12 := maj32 0x6a09e667#32 0xbb67ae85#32 0x3c6ef372#32
    let v13 := v11 + v12
    let v14 := v9 - v13
    let v15 := 0xa54ff53a#32 + v14
    let v16 := v10 - v15
    let v17 := Sigma1 b2_s6.g
    let v18 := v16 - v17
    let v19 := Sigma0 v9
    let v20 := maj32 v9 0x6a09e667#32 0xbb67ae85#32
    let v21 := v19 + v20
    let v22 := v6 - v21
    let v23 := 0x3c6ef372#32 + v22
    let v24 := ch32 b2_s6.g b2_s6.h v23
    let v25 := v18 - v24
    let v26 := v25 - 0x3956c25b#32
    let v27 := v26 - 0x510e527f#32
    let v28 := Sigma1 0x510e527f#32
    let v29 := 0x5be0cd19#32 + v28
    let v30 := ch32 0x510e527f#32 0x9b05688c#32 0x1f83d9ab#32
    let v31 := v29 + v30
    let v32 := v31 + 0x428a2f98#32
    let v33 := v14 - v32
    let v34 := v33 - 0x6a09e667#32
    let v35 := v27 - v34
    let v36 := v22 - 0x1f83d9ab#32
    let v37 := Sigma1 v15
    let v38 := v36 - v37
    let v39 := ch32 v15 0x510e527f#32 0x9b05688c#32
    let v40 := v38 - v39
    let v41 := v40 - 0x71374491#32
    let v42 := v41 - 0xbb67ae85#32
    let v43 := Sigma0 v42
    let v44 := Sigma0 v6
    let v45 := maj32 v6 v9 0x6a09e667#32
    let v46 := v44 + v45
    let v47 := b2_s6.d - v46
    let v48 := v47 - 0x9b05688c#32
    let v49 := Sigma1 v23
    let v50 := v48 - v49
    let v51 := ch32 v23 v15 0x510e527f#32
    let v52 := v50 - v51
    let v53 := v52 - 0xb5c0fbcf#32
    let v54 := v53 - 0x3c6ef372#32
    let v55 := Sigma0 b2_s6.d
    let v56 := maj32 b2_s6.d v6 v9
    let v57 := v55 + v56
    let v58 := b2_s6.c - v57
    let v59 := v58 - 0x510e527f#32
    let v60 := Sigma1 b2_s6.h
    let v61 := v59 - v60
    let v62 := ch32 b2_s6.h v23 v15
    let v63 := v61 - v62
    let v64 := v63 - 0xe9b5dba5#32
    let v65 := v64 - 0xa54ff53a#32
    let v66 := maj32 v42 v54 v65
    let v67 := v43 + v66
    let v68 := v35 + v67
    0xffffffff#32 &&& v68 = 0xffffffff#32 &&& (b1_s0.d) := by
  intro b1_s1 wp0 wp1 wp2 wp3 wp4 wp5 wp6 wp7 b2_s1 b2_s2 b2_s3 b2_s4 b2_s5 b2_s6 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52 v53 v54 v55 v56 v57 v58 v59 v60 v61 v62 v63 v64 v65 v66 v67 v68
  have hv6 : v6 = b2_s2.a := by
    change (b2_s6.e - b2_s6.a) + ((Sigma0 b2_s6.b) + (maj32 b2_s6.b b2_s6.c b2_s6.d)) = b2_s2.a
    exact round_backSolveA b2_s5 K5 wp5
  have hv9 : v9 = b2_s1.a := by
    change (b2_s6.f - b2_s6.b) + ((Sigma0 b2_s6.c) + (maj32 b2_s6.c b2_s6.d v6)) = b2_s1.a
    rw [hv6]
    exact round_backSolveA b2_s4 K4 wp4
  have hv15 : v15 = b2_s1.e := by
    change 0xa54ff53a#32 + (v9 - ((Sigma0 0x6a09e667#32) + (maj32 0x6a09e667#32 0xbb67ae85#32 0x3c6ef372#32))) = b2_s1.e
    rw [hv9]
    exact round_bridge_e_from_a stdIV K0 wp0
  have hv23 : v23 = b2_s2.e := by
    change 0x3c6ef372#32 + (v6 - ((Sigma0 v9) + (maj32 v9 0x6a09e667#32 0xbb67ae85#32))) = b2_s2.e
    rw [hv6, hv9]
    exact round_bridge_e_from_a b2_s1 K1 wp1
  have hv26 : v26 = wp4 := by
    change ((((b2_s6.f - v9) - v15) - (Sigma1 b2_s6.g)) - (ch32 b2_s6.g b2_s6.h v23)) - 0x3956c25b#32 = wp4
    rw [hv9, hv15, hv23]
    exact round_extract_w_from_e_hfirst b2_s4 K4 wp4
  have hv27 : v27 = b1_s1.e := by
    change v26 - 0x510e527f#32 = b1_s1.e
    rw [hv26]
    exact bv_splice_sub_iv stdIV.e b1_s1.e
  have hv33 : v33 = wp0 := by
    change (v9 - ((Sigma0 0x6a09e667#32) + (maj32 0x6a09e667#32 0xbb67ae85#32 0x3c6ef372#32))) - (((0x5be0cd19#32 + (Sigma1 0x510e527f#32)) + (ch32 0x510e527f#32 0x9b05688c#32 0x1f83d9ab#32)) + 0x428a2f98#32) = wp0
    rw [hv9]
    exact round_extract_w_from_a_r0 stdIV K0 wp0
  have hv34 : v34 = b1_s1.a := by
    change v33 - 0x6a09e667#32 = b1_s1.a
    rw [hv33]
    exact bv_splice_sub_iv stdIV.a b1_s1.a
  have hv41 : v41 = wp1 := by
    change ((((v6 - ((Sigma0 v9) + (maj32 v9 0x6a09e667#32 0xbb67ae85#32))) - 0x1f83d9ab#32) - (Sigma1 v15)) - (ch32 v15 0x510e527f#32 0x9b05688c#32)) - 0x71374491#32 = wp1
    rw [hv6, hv9, hv15]
    exact round_extract_w_from_a_hfirst b2_s1 K1 wp1
  have hv42 : v42 = b1_s0.a := by
    change v41 - 0xbb67ae85#32 = b1_s0.a
    rw [hv41]
    exact bv_splice_sub_iv stdIV.b b1_s0.a
  have hv53 : v53 = wp2 := by
    change ((((b2_s6.d - ((Sigma0 v6) + (maj32 v6 v9 0x6a09e667#32))) - 0x9b05688c#32) - (Sigma1 v23)) - (ch32 v23 v15 0x510e527f#32)) - 0xb5c0fbcf#32 = wp2
    rw [hv6, hv9, hv23, hv15]
    exact round_extract_w_from_a_hfirst b2_s2 K2 wp2
  have hv54 : v54 = b1_s0.b := by
    change v53 - 0x3c6ef372#32 = b1_s0.b
    rw [hv53]
    exact bv_splice_sub_iv stdIV.c b1_s0.b
  have hv64 : v64 = wp3 := by
    change ((((b2_s6.c - ((Sigma0 b2_s6.d) + (maj32 b2_s6.d v6 v9))) - 0x510e527f#32) - (Sigma1 b2_s6.h)) - (ch32 b2_s6.h v23 v15)) - 0xe9b5dba5#32 = wp3
    rw [hv6, hv9, hv23, hv15]
    exact round_extract_w_from_a_hfirst b2_s3 K3 wp3
  have hv65 : v65 = b1_s0.c := by
    change v64 - 0xa54ff53a#32 = b1_s0.c
    rw [hv64]
    exact bv_splice_sub_iv stdIV.d b1_s0.c
  have hv68 : v68 = b1_s0.d := by
    change (v27 - v34) + ((Sigma0 v42) + (maj32 v42 v54 v65)) = b1_s0.d
    rw [hv27, hv34, hv42, hv54, hv65]
    exact round_backSolveA b1_s0 K0 b1_w0
  rw [hv68]

end P3Verify.TrackB1
