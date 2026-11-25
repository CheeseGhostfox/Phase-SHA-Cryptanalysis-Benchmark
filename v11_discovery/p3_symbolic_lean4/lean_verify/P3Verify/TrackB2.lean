import P3Verify.Primitives

namespace P3Verify.TrackB2
open P3Verify

/-- `p3_id_021`：Track B2 Block2 W'[9..14]=0 collapse (t=12..18): invert across W'[16..18] eliminating Block1 E。 -/
theorem p3_id_021 (b1_s0 : SHAState) (b2_s12 : SHAState) :
    let wp0 := stdIV.a + b1_s0.a
    let wp1 := stdIV.b + b1_s0.b
    let wp2 := stdIV.c + b1_s0.c
    let wp3 := stdIV.d + b1_s0.d
    let wp4 := stdIV.e + b1_s0.e
    let wp5 := stdIV.f + b1_s0.f
    let wp6 := stdIV.g + b1_s0.g
    let wp7 := stdIV.h + b1_s0.h
    let wp12 : BitVec 32 := btcPadW12
    let wp13 : BitVec 32 := btcPadW13
    let wp14 : BitVec 32 := btcPadW14
    let wp15 : BitVec 32 := btcPadW15
    let wp16 := msgScheduleNext btcPadW14 btcPadW9 wp1 wp0
    let wp17 := msgScheduleNext btcPadW15 btcPadW10 wp2 wp1
    let wp18 := msgScheduleNext wp16 btcPadW11 wp3 wp2
    let b2_s13 := sha256Round b2_s12 K12 wp12
    let b2_s14 := sha256Round b2_s13 K13 wp13
    let b2_s15 := sha256Round b2_s14 K14 wp14
    let b2_s16 := sha256Round b2_s15 K15 wp15
    let b2_s17 := sha256Round b2_s16 K16 wp16
    let b2_s18 := sha256Round b2_s17 K17 wp17
    let b2_s19 := sha256Round b2_s18 K18 wp18
    let v0 := Sigma0 b2_s19.d
    let v1 := b2_s19.e - b2_s19.a
    let v2 := Sigma0 b2_s19.b
    let v3 := maj32 b2_s19.b b2_s19.c b2_s19.d
    let v4 := v2 + v3
    let v5 := v1 + v4
    let v6 := b2_s19.f - b2_s19.b
    let v7 := Sigma0 b2_s19.c
    let v8 := maj32 b2_s19.c b2_s19.d v5
    let v9 := v7 + v8
    let v10 := v6 + v9
    let v11 := maj32 b2_s19.d v5 v10
    let v12 := v0 + v11
    let v13 := b2_s19.c - v12
    let v14 := Sigma1 b2_s19.h
    let v15 := v13 - v14
    let v16 := b2_s19.a - v4
    let v17 := Sigma1 b2_s19.f
    let v18 := v16 - v17
    let v19 := ch32 b2_s19.f b2_s19.g b2_s19.h
    let v20 := v18 - v19
    let v21 := v20 - 0x0fc19dc6#32
    let v22 := 0xbb67ae85#32 + b1_s0.b
    let v23 := sigma0 v22
    let v24 := 0x6a09e667#32 + b1_s0.a
    let v25 := v23 + v24
    let v26 := sigma1 v25
    let v27 := 0xa54ff53a#32 + b1_s0.d
    let v28 := sigma0 v27
    let v29 := v26 + v28
    let v30 := 0x3c6ef372#32 + b1_s0.c
    let v31 := v29 + v30
    let v32 := v21 - v31
    let v33 := b2_s19.b - v9
    let v34 := Sigma1 b2_s19.g
    let v35 := v33 - v34
    let v36 := ch32 b2_s19.g b2_s19.h v32
    let v37 := v35 - v36
    let v38 := v37 - 0xefbe4786#32
    let v39 := sigma1 0x00000100#32
    let v40 := sigma0 v30
    let v41 := v39 + v40
    let v42 := v41 + v22
    let v43 := v38 - v42
    let v44 := ch32 b2_s19.h v32 v43
    let v45 := v15 - v44
    let v46 := v45 - 0xe49b69c1#32
    let v47 := v46 - v25
    let v48 := b2_s19.g - b2_s19.c
    let v49 := v48 + v12
    let v50 := v47 - v49
    let v51 := b2_s19.h - b2_s19.d
    let v52 := Sigma0 v5
    let v53 := maj32 v5 v10 v49
    let v54 := v52 + v53
    let v55 := v51 + v54
    let v56 := Sigma0 v55
    let v57 := v32 - v5
    let v58 := Sigma0 v10
    let v59 := maj32 v10 v49 v55
    let v60 := v58 + v59
    let v61 := v57 + v60
    let v62 := v43 - v10
    let v63 := Sigma0 v49
    let v64 := maj32 v49 v55 v61
    let v65 := v63 + v64
    let v66 := v62 + v65
    let v67 := maj32 v55 v61 v66
    let v68 := v56 + v67
    let v69 := v50 + v68
    0xffffffff#32 &&& v69 = 0xffffffff#32 &&& (b2_s12.d) := by
  intro wp0 wp1 wp2 wp3 wp4 wp5 wp6 wp7 wp12 wp13 wp14 wp15 wp16 wp17 wp18 b2_s13 b2_s14 b2_s15 b2_s16 b2_s17 b2_s18 b2_s19 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52 v53 v54 v55 v56 v57 v58 v59 v60 v61 v62 v63 v64 v65 v66 v67 v68 v69
  have hv5 : v5 = b2_s15.a := by
    change (b2_s19.e - b2_s19.a) + ((Sigma0 b2_s19.b) + (maj32 b2_s19.b b2_s19.c b2_s19.d)) = b2_s15.a
    exact round_backSolveA b2_s18 K18 wp18
  have hv10 : v10 = b2_s14.a := by
    change (b2_s19.f - b2_s19.b) + ((Sigma0 b2_s19.c) + (maj32 b2_s19.c b2_s19.d v5)) = b2_s14.a
    rw [hv5]
    exact round_backSolveA b2_s17 K17 wp17
  have hv22 : v22 = wp1 := by
    change 0xbb67ae85#32 + b1_s0.b = wp1
    rfl
  have hv24 : v24 = wp0 := by
    change 0x6a09e667#32 + b1_s0.a = wp0
    rfl
  have hv25 : v25 = wp16 := by
    change (sigma0 v22) + v24 = wp16
    rw [hv22, hv24]
    exact wp16_collapse wp1 wp0
  have hv27 : v27 = wp3 := by
    change 0xa54ff53a#32 + b1_s0.d = wp3
    rfl
  have hv30 : v30 = wp2 := by
    change 0x3c6ef372#32 + b1_s0.c = wp2
    rfl
  have hv31 : v31 = wp18 := by
    change ((sigma1 v25) + (sigma0 v27)) + v30 = wp18
    rw [hv25, hv27, hv30]
    exact wp18_collapse wp16 wp3 wp2
  have hv32 : v32 = b2_s15.e := by
    change ((((b2_s19.a - ((Sigma0 b2_s19.b) + (maj32 b2_s19.b b2_s19.c b2_s19.d))) - (Sigma1 b2_s19.f)) - (ch32 b2_s19.f b2_s19.g b2_s19.h)) - 0x0fc19dc6#32) - v31 = b2_s15.e
    rw [hv31]
    exact round_extract_h b2_s18 K18 wp18
  have hv42 : v42 = wp17 := by
    change ((sigma1 0x00000100#32) + (sigma0 v30)) + v22 = wp17
    rw [hv30, hv22]
    exact wp17_collapse wp2 wp1
  have hv43 : v43 = b2_s14.e := by
    change ((((b2_s19.b - ((Sigma0 b2_s19.c) + (maj32 b2_s19.c b2_s19.d v5))) - (Sigma1 b2_s19.g)) - (ch32 b2_s19.g b2_s19.h v32)) - 0xefbe4786#32) - v42 = b2_s14.e
    rw [hv5, hv32, hv42]
    exact round_extract_h b2_s17 K17 wp17
  have hv47 : v47 = b2_s13.e := by
    change ((((b2_s19.c - ((Sigma0 b2_s19.d) + (maj32 b2_s19.d v5 v10))) - (Sigma1 b2_s19.h)) - (ch32 b2_s19.h v32 v43)) - 0xe49b69c1#32) - v25 = b2_s13.e
    rw [hv5, hv10, hv32, hv43, hv25]
    exact round_extract_h b2_s16 K16 wp16
  have hv49 : v49 = b2_s13.a := by
    change (b2_s19.g - b2_s19.c) + ((Sigma0 b2_s19.d) + (maj32 b2_s19.d v5 v10)) = b2_s13.a
    rw [hv5, hv10]
    exact round_backSolveA b2_s16 K16 wp16
  have hv55 : v55 = b2_s12.a := by
    change (b2_s19.h - b2_s19.d) + ((Sigma0 v5) + (maj32 v5 v10 v49)) = b2_s12.a
    rw [hv5, hv10, hv49]
    exact round_backSolveA b2_s15 K15 wp15
  have hv61 : v61 = b2_s12.b := by
    change (v32 - v5) + ((Sigma0 v10) + (maj32 v10 v49 v55)) = b2_s12.b
    rw [hv32, hv5, hv10, hv49, hv55]
    exact round_backSolveA b2_s14 K14 wp14
  have hv66 : v66 = b2_s12.c := by
    change (v43 - v10) + ((Sigma0 v49) + (maj32 v49 v55 v61)) = b2_s12.c
    rw [hv43, hv10, hv49, hv55, hv61]
    exact round_backSolveA b2_s13 K13 wp13
  have hv69 : v69 = b2_s12.d := by
    change (v47 - v49) + ((Sigma0 v55) + (maj32 v55 v61 v66)) = b2_s12.d
    rw [hv47, hv49, hv55, hv61, hv66]
    exact round_backSolveA b2_s12 K12 wp12
  rw [hv69]

/-- `p3_id_022`：Track B2 Block2 W'[9..14]=0 collapse (t=12..18): invert across W'[16..18] eliminating Block1 E [mut] [mut] [Mask 1b]。 -/
theorem p3_id_022 (b1_s0 : SHAState) (b2_s12 : SHAState) :
    let wp0 := stdIV.a + b1_s0.a
    let wp1 := stdIV.b + b1_s0.b
    let wp2 := stdIV.c + b1_s0.c
    let wp3 := stdIV.d + b1_s0.d
    let wp4 := stdIV.e + b1_s0.e
    let wp5 := stdIV.f + b1_s0.f
    let wp6 := stdIV.g + b1_s0.g
    let wp7 := stdIV.h + b1_s0.h
    let wp12 : BitVec 32 := btcPadW12
    let wp13 : BitVec 32 := btcPadW13
    let wp14 : BitVec 32 := btcPadW14
    let wp15 : BitVec 32 := btcPadW15
    let wp16 := msgScheduleNext btcPadW14 btcPadW9 wp1 wp0
    let wp17 := msgScheduleNext btcPadW15 btcPadW10 wp2 wp1
    let wp18 := msgScheduleNext wp16 btcPadW11 wp3 wp2
    let b2_s13 := sha256Round b2_s12 K12 wp12
    let b2_s14 := sha256Round b2_s13 K13 wp13
    let b2_s15 := sha256Round b2_s14 K14 wp14
    let b2_s16 := sha256Round b2_s15 K15 wp15
    let b2_s17 := sha256Round b2_s16 K16 wp16
    let b2_s18 := sha256Round b2_s17 K17 wp17
    let b2_s19 := sha256Round b2_s18 K18 wp18
    let v0 := Sigma0 b2_s19.d
    let v1 := b2_s19.e - b2_s19.a
    let v2 := Sigma0 b2_s19.b
    let v3 := maj32 b2_s19.b b2_s19.c b2_s19.d
    let v4 := v2 + v3
    let v5 := v1 + v4
    let v6 := b2_s19.f - b2_s19.b
    let v7 := Sigma0 b2_s19.c
    let v8 := maj32 b2_s19.c b2_s19.d v5
    let v9 := v7 + v8
    let v10 := v6 + v9
    let v11 := maj32 b2_s19.d v5 v10
    let v12 := v0 + v11
    let v13 := b2_s19.c - v12
    let v14 := Sigma1 b2_s19.h
    let v15 := v13 - v14
    let v16 := b2_s19.a - v4
    let v17 := Sigma1 b2_s19.f
    let v18 := v16 - v17
    let v19 := ch32 b2_s19.f b2_s19.g b2_s19.h
    let v20 := v18 - v19
    let v21 := v20 - 0x0fc19dc6#32
    let v22 := 0xbb67ae85#32 + b1_s0.b
    let v23 := sigma0 v22
    let v24 := 0x6a09e667#32 + b1_s0.a
    let v25 := v23 + v24
    let v26 := sigma1 v25
    let v27 := 0xa54ff53a#32 + b1_s0.d
    let v28 := sigma0 v27
    let v29 := v26 + v28
    let v30 := 0x3c6ef372#32 + b1_s0.c
    let v31 := v29 + v30
    let v32 := v21 - v31
    let v33 := b2_s19.b - v9
    let v34 := Sigma1 b2_s19.g
    let v35 := v33 - v34
    let v36 := ch32 b2_s19.g b2_s19.h v32
    let v37 := v35 - v36
    let v38 := v37 - 0xefbe4786#32
    let v39 := sigma1 0x00000100#32
    let v40 := sigma0 v30
    let v41 := v39 + v40
    let v42 := v41 + v22
    let v43 := v38 - v42
    let v44 := ch32 b2_s19.h v32 v43
    let v45 := v15 - v44
    let v46 := v45 - 0xe49b69c1#32
    let v47 := v46 - v25
    let v48 := b2_s19.g - b2_s19.c
    let v49 := v48 + v12
    let v50 := v47 - v49
    let v51 := b2_s19.h - b2_s19.d
    let v52 := Sigma0 v5
    let v53 := maj32 v5 v10 v49
    let v54 := v52 + v53
    let v55 := v51 + v54
    let v56 := Sigma0 v55
    let v57 := v32 - v5
    let v58 := Sigma0 v10
    let v59 := maj32 v10 v49 v55
    let v60 := v58 + v59
    let v61 := v57 + v60
    let v62 := v43 - v10
    let v63 := Sigma0 v49
    let v64 := maj32 v49 v55 v61
    let v65 := v63 + v64
    let v66 := v62 + v65
    let v67 := maj32 v55 v61 v66
    let v68 := v56 + v67
    let v69 := v50 - v68
    0x00000001#32 &&& v69 = 0x00000001#32 &&& (b2_s12.d) := by
  intro wp0 wp1 wp2 wp3 wp4 wp5 wp6 wp7 wp12 wp13 wp14 wp15 wp16 wp17 wp18 b2_s13 b2_s14 b2_s15 b2_s16 b2_s17 b2_s18 b2_s19 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52 v53 v54 v55 v56 v57 v58 v59 v60 v61 v62 v63 v64 v65 v66 v67 v68 v69
  have hv5 : v5 = b2_s15.a := by
    change (b2_s19.e - b2_s19.a) + ((Sigma0 b2_s19.b) + (maj32 b2_s19.b b2_s19.c b2_s19.d)) = b2_s15.a
    exact round_backSolveA b2_s18 K18 wp18
  have hv10 : v10 = b2_s14.a := by
    change (b2_s19.f - b2_s19.b) + ((Sigma0 b2_s19.c) + (maj32 b2_s19.c b2_s19.d v5)) = b2_s14.a
    rw [hv5]
    exact round_backSolveA b2_s17 K17 wp17
  have hv22 : v22 = wp1 := by
    change 0xbb67ae85#32 + b1_s0.b = wp1
    rfl
  have hv24 : v24 = wp0 := by
    change 0x6a09e667#32 + b1_s0.a = wp0
    rfl
  have hv25 : v25 = wp16 := by
    change (sigma0 v22) + v24 = wp16
    rw [hv22, hv24]
    exact wp16_collapse wp1 wp0
  have hv27 : v27 = wp3 := by
    change 0xa54ff53a#32 + b1_s0.d = wp3
    rfl
  have hv30 : v30 = wp2 := by
    change 0x3c6ef372#32 + b1_s0.c = wp2
    rfl
  have hv31 : v31 = wp18 := by
    change ((sigma1 v25) + (sigma0 v27)) + v30 = wp18
    rw [hv25, hv27, hv30]
    exact wp18_collapse wp16 wp3 wp2
  have hv32 : v32 = b2_s15.e := by
    change ((((b2_s19.a - ((Sigma0 b2_s19.b) + (maj32 b2_s19.b b2_s19.c b2_s19.d))) - (Sigma1 b2_s19.f)) - (ch32 b2_s19.f b2_s19.g b2_s19.h)) - 0x0fc19dc6#32) - v31 = b2_s15.e
    rw [hv31]
    exact round_extract_h b2_s18 K18 wp18
  have hv42 : v42 = wp17 := by
    change ((sigma1 0x00000100#32) + (sigma0 v30)) + v22 = wp17
    rw [hv30, hv22]
    exact wp17_collapse wp2 wp1
  have hv43 : v43 = b2_s14.e := by
    change ((((b2_s19.b - ((Sigma0 b2_s19.c) + (maj32 b2_s19.c b2_s19.d v5))) - (Sigma1 b2_s19.g)) - (ch32 b2_s19.g b2_s19.h v32)) - 0xefbe4786#32) - v42 = b2_s14.e
    rw [hv5, hv32, hv42]
    exact round_extract_h b2_s17 K17 wp17
  have hv47 : v47 = b2_s13.e := by
    change ((((b2_s19.c - ((Sigma0 b2_s19.d) + (maj32 b2_s19.d v5 v10))) - (Sigma1 b2_s19.h)) - (ch32 b2_s19.h v32 v43)) - 0xe49b69c1#32) - v25 = b2_s13.e
    rw [hv5, hv10, hv32, hv43, hv25]
    exact round_extract_h b2_s16 K16 wp16
  have hv49 : v49 = b2_s13.a := by
    change (b2_s19.g - b2_s19.c) + ((Sigma0 b2_s19.d) + (maj32 b2_s19.d v5 v10)) = b2_s13.a
    rw [hv5, hv10]
    exact round_backSolveA b2_s16 K16 wp16
  have hv55 : v55 = b2_s12.a := by
    change (b2_s19.h - b2_s19.d) + ((Sigma0 v5) + (maj32 v5 v10 v49)) = b2_s12.a
    rw [hv5, hv10, hv49]
    exact round_backSolveA b2_s15 K15 wp15
  have hv61 : v61 = b2_s12.b := by
    change (v32 - v5) + ((Sigma0 v10) + (maj32 v10 v49 v55)) = b2_s12.b
    rw [hv32, hv5, hv10, hv49, hv55]
    exact round_backSolveA b2_s14 K14 wp14
  have hv66 : v66 = b2_s12.c := by
    change (v43 - v10) + ((Sigma0 v49) + (maj32 v49 v55 v61)) = b2_s12.c
    rw [hv43, hv10, hv49, hv55, hv61]
    exact round_backSolveA b2_s13 K13 wp13
  change 1#32 &&& ((v47 - v49) - (Sigma0 v55 + maj32 v55 v61 v66)) = 1#32 &&& b2_s12.d
  have hfin : (b2_s13.e - b2_s13.a) + (Sigma0 b2_s12.a + maj32 b2_s12.a b2_s12.b b2_s12.c) = b2_s12.d := round_backSolveA b2_s12 K12 wp12
  rw [lsb_sub_eq_add_left, hv47, hv49, hv55, hv61, hv66, hfin]

/-- `p3_id_023`：Track B2 Block2 W'[9..14]=0 collapse (t=12..18): invert across W'[16..18] eliminating Block1 E。 -/
theorem p3_id_023 (b1_s0 : SHAState) (b2_s12 : SHAState) :
    let wp0 := stdIV.a + b1_s0.a
    let wp1 := stdIV.b + b1_s0.b
    let wp2 := stdIV.c + b1_s0.c
    let wp3 := stdIV.d + b1_s0.d
    let wp4 := stdIV.e + b1_s0.e
    let wp5 := stdIV.f + b1_s0.f
    let wp6 := stdIV.g + b1_s0.g
    let wp7 := stdIV.h + b1_s0.h
    let wp12 : BitVec 32 := btcPadW12
    let wp13 : BitVec 32 := btcPadW13
    let wp14 : BitVec 32 := btcPadW14
    let wp15 : BitVec 32 := btcPadW15
    let wp16 := msgScheduleNext btcPadW14 btcPadW9 wp1 wp0
    let wp17 := msgScheduleNext btcPadW15 btcPadW10 wp2 wp1
    let wp18 := msgScheduleNext wp16 btcPadW11 wp3 wp2
    let b2_s13 := sha256Round b2_s12 K12 wp12
    let b2_s14 := sha256Round b2_s13 K13 wp13
    let b2_s15 := sha256Round b2_s14 K14 wp14
    let b2_s16 := sha256Round b2_s15 K15 wp15
    let b2_s17 := sha256Round b2_s16 K16 wp16
    let b2_s18 := sha256Round b2_s17 K17 wp17
    let b2_s19 := sha256Round b2_s18 K18 wp18
    let v0 := b2_s19.g - b2_s19.c
    let v1 := Sigma0 b2_s19.d
    let v2 := b2_s19.e - b2_s19.a
    let v3 := Sigma0 b2_s19.b
    let v4 := maj32 b2_s19.b b2_s19.c b2_s19.d
    let v5 := v3 + v4
    let v6 := v2 + v5
    let v7 := b2_s19.f - b2_s19.b
    let v8 := Sigma0 b2_s19.c
    let v9 := maj32 b2_s19.c b2_s19.d v6
    let v10 := v8 + v9
    let v11 := v7 + v10
    let v12 := maj32 b2_s19.d v6 v11
    let v13 := v1 + v12
    let v14 := v0 + v13
    let v15 := b2_s19.h - b2_s19.d
    let v16 := Sigma0 v6
    let v17 := maj32 v6 v11 v14
    let v18 := v16 + v17
    let v19 := v15 + v18
    let v20 := Sigma0 v19
    let v21 := b2_s19.a - v5
    let v22 := Sigma1 b2_s19.f
    let v23 := v21 - v22
    let v24 := ch32 b2_s19.f b2_s19.g b2_s19.h
    let v25 := v23 - v24
    let v26 := v25 - 0x0fc19dc6#32
    let v27 := 0xbb67ae85#32 + b1_s0.b
    let v28 := sigma0 v27
    let v29 := 0x6a09e667#32 + b1_s0.a
    let v30 := v28 + v29
    let v31 := sigma1 v30
    let v32 := 0xa54ff53a#32 + b1_s0.d
    let v33 := sigma0 v32
    let v34 := v31 + v33
    let v35 := 0x3c6ef372#32 + b1_s0.c
    let v36 := v34 + v35
    let v37 := v26 - v36
    let v38 := v37 - v6
    let v39 := Sigma0 v11
    let v40 := maj32 v11 v14 v19
    let v41 := v39 + v40
    let v42 := v38 + v41
    let v43 := b2_s19.b - v10
    let v44 := Sigma1 b2_s19.g
    let v45 := v43 - v44
    let v46 := ch32 b2_s19.g b2_s19.h v37
    let v47 := v45 - v46
    let v48 := v47 - 0xefbe4786#32
    let v49 := sigma1 0x00000100#32
    let v50 := sigma0 v35
    let v51 := v49 + v50
    let v52 := v51 + v27
    let v53 := v48 - v52
    let v54 := v53 - v11
    let v55 := Sigma0 v14
    let v56 := maj32 v14 v19 v42
    let v57 := v55 + v56
    let v58 := v54 + v57
    let v59 := maj32 v19 v42 v58
    let v60 := v20 + v59
    let v61 := v14 - v60
    let v62 := b2_s19.d - v18
    let v63 := Sigma1 v37
    let v64 := v62 - v63
    let v65 := b2_s19.c - v13
    let v66 := Sigma1 b2_s19.h
    let v67 := v65 - v66
    let v68 := ch32 b2_s19.h v37 v53
    let v69 := v67 - v68
    let v70 := v69 - 0xe49b69c1#32
    let v71 := v70 - v30
    let v72 := ch32 v37 v53 v71
    let v73 := v64 - v72
    let v74 := v73 - 0xc19bf174#32
    let v75 := v74 - 0x00000100#32
    let v76 := Sigma1 v75
    let v77 := v61 - v76
    let v78 := v6 - v41
    let v79 := Sigma1 v53
    let v80 := v78 - v79
    let v81 := ch32 v53 v71 v75
    let v82 := v80 - v81
    let v83 := v82 - 0x9bdc06a7#32
    let v84 := v83 - 0x00000000#32
    let v85 := v11 - v57
    let v86 := Sigma1 v71
    let v87 := v85 - v86
    let v88 := ch32 v71 v75 v84
    let v89 := v87 - v88
    let v90 := v89 - 0x80deb1fe#32
    let v91 := v90 - 0x00000000#32
    let v92 := ch32 v75 v84 v91
    let v93 := v77 - v92
    let v94 := v93 - 0x72be5d74#32
    let v95 := v94 - 0x00000000#32
    0xffffffff#32 &&& v95 = 0xffffffff#32 &&& (b2_s12.h) := by
  intro wp0 wp1 wp2 wp3 wp4 wp5 wp6 wp7 wp12 wp13 wp14 wp15 wp16 wp17 wp18 b2_s13 b2_s14 b2_s15 b2_s16 b2_s17 b2_s18 b2_s19 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52 v53 v54 v55 v56 v57 v58 v59 v60 v61 v62 v63 v64 v65 v66 v67 v68 v69 v70 v71 v72 v73 v74 v75 v76 v77 v78 v79 v80 v81 v82 v83 v84 v85 v86 v87 v88 v89 v90 v91 v92 v93 v94 v95
  have hv6 : v6 = b2_s15.a := by
    change (b2_s19.e - b2_s19.a) + ((Sigma0 b2_s19.b) + (maj32 b2_s19.b b2_s19.c b2_s19.d)) = b2_s15.a
    exact round_backSolveA b2_s18 K18 wp18
  have hv11 : v11 = b2_s14.a := by
    change (b2_s19.f - b2_s19.b) + ((Sigma0 b2_s19.c) + (maj32 b2_s19.c b2_s19.d v6)) = b2_s14.a
    rw [hv6]
    exact round_backSolveA b2_s17 K17 wp17
  have hv14 : v14 = b2_s13.a := by
    change (b2_s19.g - b2_s19.c) + ((Sigma0 b2_s19.d) + (maj32 b2_s19.d v6 v11)) = b2_s13.a
    rw [hv6, hv11]
    exact round_backSolveA b2_s16 K16 wp16
  have hv19 : v19 = b2_s12.a := by
    change (b2_s19.h - b2_s19.d) + ((Sigma0 v6) + (maj32 v6 v11 v14)) = b2_s12.a
    rw [hv6, hv11, hv14]
    exact round_backSolveA b2_s15 K15 wp15
  have hv27 : v27 = wp1 := by
    change 0xbb67ae85#32 + b1_s0.b = wp1
    rfl
  have hv29 : v29 = wp0 := by
    change 0x6a09e667#32 + b1_s0.a = wp0
    rfl
  have hv30 : v30 = wp16 := by
    change (sigma0 v27) + v29 = wp16
    rw [hv27, hv29]
    exact wp16_collapse wp1 wp0
  have hv32 : v32 = wp3 := by
    change 0xa54ff53a#32 + b1_s0.d = wp3
    rfl
  have hv35 : v35 = wp2 := by
    change 0x3c6ef372#32 + b1_s0.c = wp2
    rfl
  have hv36 : v36 = wp18 := by
    change ((sigma1 v30) + (sigma0 v32)) + v35 = wp18
    rw [hv30, hv32, hv35]
    exact wp18_collapse wp16 wp3 wp2
  have hv37 : v37 = b2_s15.e := by
    change ((((b2_s19.a - ((Sigma0 b2_s19.b) + (maj32 b2_s19.b b2_s19.c b2_s19.d))) - (Sigma1 b2_s19.f)) - (ch32 b2_s19.f b2_s19.g b2_s19.h)) - 0x0fc19dc6#32) - v36 = b2_s15.e
    rw [hv36]
    exact round_extract_h b2_s18 K18 wp18
  have hv42 : v42 = b2_s12.b := by
    change (v37 - v6) + ((Sigma0 v11) + (maj32 v11 v14 v19)) = b2_s12.b
    rw [hv37, hv6, hv11, hv14, hv19]
    exact round_backSolveA b2_s14 K14 wp14
  have hv52 : v52 = wp17 := by
    change ((sigma1 0x00000100#32) + (sigma0 v35)) + v27 = wp17
    rw [hv35, hv27]
    exact wp17_collapse wp2 wp1
  have hv53 : v53 = b2_s14.e := by
    change ((((b2_s19.b - ((Sigma0 b2_s19.c) + (maj32 b2_s19.c b2_s19.d v6))) - (Sigma1 b2_s19.g)) - (ch32 b2_s19.g b2_s19.h v37)) - 0xefbe4786#32) - v52 = b2_s14.e
    rw [hv6, hv37, hv52]
    exact round_extract_h b2_s17 K17 wp17
  have hv58 : v58 = b2_s12.c := by
    change (v53 - v11) + ((Sigma0 v14) + (maj32 v14 v19 v42)) = b2_s12.c
    rw [hv53, hv11, hv14, hv19, hv42]
    exact round_backSolveA b2_s13 K13 wp13
  have hv71 : v71 = b2_s13.e := by
    change ((((b2_s19.c - ((Sigma0 b2_s19.d) + (maj32 b2_s19.d v6 v11))) - (Sigma1 b2_s19.h)) - (ch32 b2_s19.h v37 v53)) - 0xe49b69c1#32) - v30 = b2_s13.e
    rw [hv6, hv11, hv37, hv53, hv30]
    exact round_extract_h b2_s16 K16 wp16
  have hv75 : v75 = b2_s12.e := by
    change ((((b2_s19.d - ((Sigma0 v6) + (maj32 v6 v11 v14))) - (Sigma1 v37)) - (ch32 v37 v53 v71)) - 0xc19bf174#32) - 0x00000100#32 = b2_s12.e
    rw [hv6, hv11, hv14, hv37, hv53, hv71]
    exact round_extract_h b2_s15 K15 wp15
  have hv83 : v83 = b2_s12.f := by
    change (((v6 - ((Sigma0 v11) + (maj32 v11 v14 v19))) - (Sigma1 v53)) - (ch32 v53 v71 v75)) - 0x9bdc06a7#32 = b2_s12.f
    rw [hv6, hv11, hv14, hv19, hv53, hv71, hv75]
    exact round_extract_hw_zero b2_s14 K14
  have hv84 : v84 = b2_s12.f := by
    change v83 - 0x00000000#32 = b2_s12.f
    rw [hv83]
    exact bv_sub_zero _
  have hv90 : v90 = b2_s12.g := by
    change (((v11 - ((Sigma0 v14) + (maj32 v14 v19 v42))) - (Sigma1 v71)) - (ch32 v71 v75 v84)) - 0x80deb1fe#32 = b2_s12.g
    rw [hv11, hv14, hv19, hv42, hv71, hv75, hv84]
    exact round_extract_hw_zero b2_s13 K13
  have hv91 : v91 = b2_s12.g := by
    change v90 - 0x00000000#32 = b2_s12.g
    rw [hv90]
    exact bv_sub_zero _
  have hv94 : v94 = b2_s12.h := by
    change (((v14 - ((Sigma0 v19) + (maj32 v19 v42 v58))) - (Sigma1 v75)) - (ch32 v75 v84 v91)) - 0x72be5d74#32 = b2_s12.h
    rw [hv14, hv19, hv42, hv58, hv75, hv84, hv91]
    exact round_extract_hw_zero b2_s12 K12
  have hv95 : v95 = b2_s12.h := by
    change v94 - 0x00000000#32 = b2_s12.h
    rw [hv94]
    exact bv_sub_zero _
  rw [hv95]

/-- `p3_id_024`：Track B2 Block2 W'[9..14]=0 collapse (t=12..17): invert across W'[16..17] eliminating Block1 E。 -/
theorem p3_id_024 (b1_s0 : SHAState) (b2_s12 : SHAState) :
    let wp0 := stdIV.a + b1_s0.a
    let wp1 := stdIV.b + b1_s0.b
    let wp2 := stdIV.c + b1_s0.c
    let wp3 := stdIV.d + b1_s0.d
    let wp4 := stdIV.e + b1_s0.e
    let wp5 := stdIV.f + b1_s0.f
    let wp6 := stdIV.g + b1_s0.g
    let wp7 := stdIV.h + b1_s0.h
    let wp12 : BitVec 32 := btcPadW12
    let wp13 : BitVec 32 := btcPadW13
    let wp14 : BitVec 32 := btcPadW14
    let wp15 : BitVec 32 := btcPadW15
    let wp16 := msgScheduleNext btcPadW14 btcPadW9 wp1 wp0
    let wp17 := msgScheduleNext btcPadW15 btcPadW10 wp2 wp1
    let wp18 := msgScheduleNext wp16 btcPadW11 wp3 wp2
    let b2_s13 := sha256Round b2_s12 K12 wp12
    let b2_s14 := sha256Round b2_s13 K13 wp13
    let b2_s15 := sha256Round b2_s14 K14 wp14
    let b2_s16 := sha256Round b2_s15 K15 wp15
    let b2_s17 := sha256Round b2_s16 K16 wp16
    let b2_s18 := sha256Round b2_s17 K17 wp17
    let v0 := Sigma0 b2_s18.c
    let v1 := b2_s18.e - b2_s18.a
    let v2 := Sigma0 b2_s18.b
    let v3 := maj32 b2_s18.b b2_s18.c b2_s18.d
    let v4 := v2 + v3
    let v5 := v1 + v4
    let v6 := maj32 b2_s18.c b2_s18.d v5
    let v7 := v0 + v6
    let v8 := b2_s18.b - v7
    let v9 := Sigma1 b2_s18.g
    let v10 := v8 - v9
    let v11 := b2_s18.a - v4
    let v12 := Sigma1 b2_s18.f
    let v13 := v11 - v12
    let v14 := ch32 b2_s18.f b2_s18.g b2_s18.h
    let v15 := v13 - v14
    let v16 := v15 - 0xefbe4786#32
    let v17 := sigma1 0x00000100#32
    let v18 := 0x3c6ef372#32 + b1_s0.c
    let v19 := sigma0 v18
    let v20 := v17 + v19
    let v21 := 0xbb67ae85#32 + b1_s0.b
    let v22 := v20 + v21
    let v23 := v16 - v22
    let v24 := ch32 b2_s18.g b2_s18.h v23
    let v25 := v10 - v24
    let v26 := v25 - 0xe49b69c1#32
    let v27 := sigma0 v21
    let v28 := 0x6a09e667#32 + b1_s0.a
    let v29 := v27 + v28
    let v30 := v26 - v29
    let v31 := b2_s18.f - b2_s18.b
    let v32 := v31 + v7
    let v33 := v30 - v32
    let v34 := b2_s18.g - b2_s18.c
    let v35 := Sigma0 b2_s18.d
    let v36 := maj32 b2_s18.d v5 v32
    let v37 := v35 + v36
    let v38 := v34 + v37
    let v39 := Sigma0 v38
    let v40 := b2_s18.h - b2_s18.d
    let v41 := Sigma0 v5
    let v42 := maj32 v5 v32 v38
    let v43 := v41 + v42
    let v44 := v40 + v43
    let v45 := v23 - v5
    let v46 := Sigma0 v32
    let v47 := maj32 v32 v38 v44
    let v48 := v46 + v47
    let v49 := v45 + v48
    let v50 := maj32 v38 v44 v49
    let v51 := v39 + v50
    let v52 := v33 + v51
    0xffffffff#32 &&& v52 = 0xffffffff#32 &&& (b2_s12.d) := by
  intro wp0 wp1 wp2 wp3 wp4 wp5 wp6 wp7 wp12 wp13 wp14 wp15 wp16 wp17 wp18 b2_s13 b2_s14 b2_s15 b2_s16 b2_s17 b2_s18 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52
  have hv5 : v5 = b2_s14.a := by
    change (b2_s18.e - b2_s18.a) + ((Sigma0 b2_s18.b) + (maj32 b2_s18.b b2_s18.c b2_s18.d)) = b2_s14.a
    exact round_backSolveA b2_s17 K17 wp17
  have hv18 : v18 = wp2 := by
    change 0x3c6ef372#32 + b1_s0.c = wp2
    rfl
  have hv21 : v21 = wp1 := by
    change 0xbb67ae85#32 + b1_s0.b = wp1
    rfl
  have hv22 : v22 = wp17 := by
    change ((sigma1 0x00000100#32) + (sigma0 v18)) + v21 = wp17
    rw [hv18, hv21]
    exact wp17_collapse wp2 wp1
  have hv23 : v23 = b2_s14.e := by
    change ((((b2_s18.a - ((Sigma0 b2_s18.b) + (maj32 b2_s18.b b2_s18.c b2_s18.d))) - (Sigma1 b2_s18.f)) - (ch32 b2_s18.f b2_s18.g b2_s18.h)) - 0xefbe4786#32) - v22 = b2_s14.e
    rw [hv22]
    exact round_extract_h b2_s17 K17 wp17
  have hv28 : v28 = wp0 := by
    change 0x6a09e667#32 + b1_s0.a = wp0
    rfl
  have hv29 : v29 = wp16 := by
    change (sigma0 v21) + v28 = wp16
    rw [hv21, hv28]
    exact wp16_collapse wp1 wp0
  have hv30 : v30 = b2_s13.e := by
    change ((((b2_s18.b - ((Sigma0 b2_s18.c) + (maj32 b2_s18.c b2_s18.d v5))) - (Sigma1 b2_s18.g)) - (ch32 b2_s18.g b2_s18.h v23)) - 0xe49b69c1#32) - v29 = b2_s13.e
    rw [hv5, hv23, hv29]
    exact round_extract_h b2_s16 K16 wp16
  have hv32 : v32 = b2_s13.a := by
    change (b2_s18.f - b2_s18.b) + ((Sigma0 b2_s18.c) + (maj32 b2_s18.c b2_s18.d v5)) = b2_s13.a
    rw [hv5]
    exact round_backSolveA b2_s16 K16 wp16
  have hv38 : v38 = b2_s12.a := by
    change (b2_s18.g - b2_s18.c) + ((Sigma0 b2_s18.d) + (maj32 b2_s18.d v5 v32)) = b2_s12.a
    rw [hv5, hv32]
    exact round_backSolveA b2_s15 K15 wp15
  have hv44 : v44 = b2_s12.b := by
    change (b2_s18.h - b2_s18.d) + ((Sigma0 v5) + (maj32 v5 v32 v38)) = b2_s12.b
    rw [hv5, hv32, hv38]
    exact round_backSolveA b2_s14 K14 wp14
  have hv49 : v49 = b2_s12.c := by
    change (v23 - v5) + ((Sigma0 v32) + (maj32 v32 v38 v44)) = b2_s12.c
    rw [hv23, hv5, hv32, hv38, hv44]
    exact round_backSolveA b2_s13 K13 wp13
  have hv52 : v52 = b2_s12.d := by
    change (v30 - v32) + ((Sigma0 v38) + (maj32 v38 v44 v49)) = b2_s12.d
    rw [hv30, hv32, hv38, hv44, hv49]
    exact round_backSolveA b2_s12 K12 wp12
  rw [hv52]

/-- `p3_id_025`：Track B2 Block2 W'[9..14]=0 collapse (t=13..18): invert across W'[16..18] eliminating Block1 E。 -/
theorem p3_id_025 (b1_s0 : SHAState) (b2_s13 : SHAState) :
    let wp0 := stdIV.a + b1_s0.a
    let wp1 := stdIV.b + b1_s0.b
    let wp2 := stdIV.c + b1_s0.c
    let wp3 := stdIV.d + b1_s0.d
    let wp4 := stdIV.e + b1_s0.e
    let wp5 := stdIV.f + b1_s0.f
    let wp6 := stdIV.g + b1_s0.g
    let wp7 := stdIV.h + b1_s0.h
    let wp12 : BitVec 32 := btcPadW12
    let wp13 : BitVec 32 := btcPadW13
    let wp14 : BitVec 32 := btcPadW14
    let wp15 : BitVec 32 := btcPadW15
    let wp16 := msgScheduleNext btcPadW14 btcPadW9 wp1 wp0
    let wp17 := msgScheduleNext btcPadW15 btcPadW10 wp2 wp1
    let wp18 := msgScheduleNext wp16 btcPadW11 wp3 wp2
    let b2_s14 := sha256Round b2_s13 K13 wp13
    let b2_s15 := sha256Round b2_s14 K14 wp14
    let b2_s16 := sha256Round b2_s15 K15 wp15
    let b2_s17 := sha256Round b2_s16 K16 wp16
    let b2_s18 := sha256Round b2_s17 K17 wp17
    let b2_s19 := sha256Round b2_s18 K18 wp18
    let v0 := Sigma0 b2_s19.c
    let v1 := b2_s19.e - b2_s19.a
    let v2 := Sigma0 b2_s19.b
    let v3 := maj32 b2_s19.b b2_s19.c b2_s19.d
    let v4 := v2 + v3
    let v5 := v1 + v4
    let v6 := maj32 b2_s19.c b2_s19.d v5
    let v7 := v0 + v6
    let v8 := b2_s19.b - v7
    let v9 := Sigma1 b2_s19.g
    let v10 := v8 - v9
    let v11 := b2_s19.a - v4
    let v12 := Sigma1 b2_s19.f
    let v13 := v11 - v12
    let v14 := ch32 b2_s19.f b2_s19.g b2_s19.h
    let v15 := v13 - v14
    let v16 := v15 - 0x0fc19dc6#32
    let v17 := 0xbb67ae85#32 + b1_s0.b
    let v18 := sigma0 v17
    let v19 := 0x6a09e667#32 + b1_s0.a
    let v20 := v18 + v19
    let v21 := sigma1 v20
    let v22 := 0xa54ff53a#32 + b1_s0.d
    let v23 := sigma0 v22
    let v24 := v21 + v23
    let v25 := 0x3c6ef372#32 + b1_s0.c
    let v26 := v24 + v25
    let v27 := v16 - v26
    let v28 := ch32 b2_s19.g b2_s19.h v27
    let v29 := v10 - v28
    let v30 := v29 - 0xefbe4786#32
    let v31 := sigma1 0x00000100#32
    let v32 := sigma0 v25
    let v33 := v31 + v32
    let v34 := v33 + v17
    let v35 := v30 - v34
    let v36 := b2_s19.f - b2_s19.b
    let v37 := v36 + v7
    let v38 := v35 - v37
    let v39 := b2_s19.g - b2_s19.c
    let v40 := Sigma0 b2_s19.d
    let v41 := maj32 b2_s19.d v5 v37
    let v42 := v40 + v41
    let v43 := v39 + v42
    let v44 := Sigma0 v43
    let v45 := b2_s19.h - b2_s19.d
    let v46 := Sigma0 v5
    let v47 := maj32 v5 v37 v43
    let v48 := v46 + v47
    let v49 := v45 + v48
    let v50 := v27 - v5
    let v51 := Sigma0 v37
    let v52 := maj32 v37 v43 v49
    let v53 := v51 + v52
    let v54 := v50 + v53
    let v55 := maj32 v43 v49 v54
    let v56 := v44 + v55
    let v57 := v38 + v56
    0xffffffff#32 &&& v57 = 0xffffffff#32 &&& (b2_s13.d) := by
  intro wp0 wp1 wp2 wp3 wp4 wp5 wp6 wp7 wp12 wp13 wp14 wp15 wp16 wp17 wp18 b2_s14 b2_s15 b2_s16 b2_s17 b2_s18 b2_s19 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52 v53 v54 v55 v56 v57
  have hv5 : v5 = b2_s15.a := by
    change (b2_s19.e - b2_s19.a) + ((Sigma0 b2_s19.b) + (maj32 b2_s19.b b2_s19.c b2_s19.d)) = b2_s15.a
    exact round_backSolveA b2_s18 K18 wp18
  have hv17 : v17 = wp1 := by
    change 0xbb67ae85#32 + b1_s0.b = wp1
    rfl
  have hv19 : v19 = wp0 := by
    change 0x6a09e667#32 + b1_s0.a = wp0
    rfl
  have hv20 : v20 = wp16 := by
    change (sigma0 v17) + v19 = wp16
    rw [hv17, hv19]
    exact wp16_collapse wp1 wp0
  have hv22 : v22 = wp3 := by
    change 0xa54ff53a#32 + b1_s0.d = wp3
    rfl
  have hv25 : v25 = wp2 := by
    change 0x3c6ef372#32 + b1_s0.c = wp2
    rfl
  have hv26 : v26 = wp18 := by
    change ((sigma1 v20) + (sigma0 v22)) + v25 = wp18
    rw [hv20, hv22, hv25]
    exact wp18_collapse wp16 wp3 wp2
  have hv27 : v27 = b2_s15.e := by
    change ((((b2_s19.a - ((Sigma0 b2_s19.b) + (maj32 b2_s19.b b2_s19.c b2_s19.d))) - (Sigma1 b2_s19.f)) - (ch32 b2_s19.f b2_s19.g b2_s19.h)) - 0x0fc19dc6#32) - v26 = b2_s15.e
    rw [hv26]
    exact round_extract_h b2_s18 K18 wp18
  have hv34 : v34 = wp17 := by
    change ((sigma1 0x00000100#32) + (sigma0 v25)) + v17 = wp17
    rw [hv25, hv17]
    exact wp17_collapse wp2 wp1
  have hv35 : v35 = b2_s14.e := by
    change ((((b2_s19.b - ((Sigma0 b2_s19.c) + (maj32 b2_s19.c b2_s19.d v5))) - (Sigma1 b2_s19.g)) - (ch32 b2_s19.g b2_s19.h v27)) - 0xefbe4786#32) - v34 = b2_s14.e
    rw [hv5, hv27, hv34]
    exact round_extract_h b2_s17 K17 wp17
  have hv37 : v37 = b2_s14.a := by
    change (b2_s19.f - b2_s19.b) + ((Sigma0 b2_s19.c) + (maj32 b2_s19.c b2_s19.d v5)) = b2_s14.a
    rw [hv5]
    exact round_backSolveA b2_s17 K17 wp17
  have hv43 : v43 = b2_s13.a := by
    change (b2_s19.g - b2_s19.c) + ((Sigma0 b2_s19.d) + (maj32 b2_s19.d v5 v37)) = b2_s13.a
    rw [hv5, hv37]
    exact round_backSolveA b2_s16 K16 wp16
  have hv49 : v49 = b2_s13.b := by
    change (b2_s19.h - b2_s19.d) + ((Sigma0 v5) + (maj32 v5 v37 v43)) = b2_s13.b
    rw [hv5, hv37, hv43]
    exact round_backSolveA b2_s15 K15 wp15
  have hv54 : v54 = b2_s13.c := by
    change (v27 - v5) + ((Sigma0 v37) + (maj32 v37 v43 v49)) = b2_s13.c
    rw [hv27, hv5, hv37, hv43, hv49]
    exact round_backSolveA b2_s14 K14 wp14
  have hv57 : v57 = b2_s13.d := by
    change (v35 - v37) + ((Sigma0 v43) + (maj32 v43 v49 v54)) = b2_s13.d
    rw [hv35, hv37, hv43, hv49, hv54]
    exact round_backSolveA b2_s13 K13 wp13
  rw [hv57]

/-- `p3_id_026`：Track B2 Block2 W'[9..14]=0 collapse (t=13..18): invert across W'[16..18] eliminating Block1 E [mut] [Mask 1b]。 -/
theorem p3_id_026 (b1_s0 : SHAState) (b2_s13 : SHAState) :
    let wp0 := stdIV.a + b1_s0.a
    let wp1 := stdIV.b + b1_s0.b
    let wp2 := stdIV.c + b1_s0.c
    let wp3 := stdIV.d + b1_s0.d
    let wp4 := stdIV.e + b1_s0.e
    let wp5 := stdIV.f + b1_s0.f
    let wp6 := stdIV.g + b1_s0.g
    let wp7 := stdIV.h + b1_s0.h
    let wp12 : BitVec 32 := btcPadW12
    let wp13 : BitVec 32 := btcPadW13
    let wp14 : BitVec 32 := btcPadW14
    let wp15 : BitVec 32 := btcPadW15
    let wp16 := msgScheduleNext btcPadW14 btcPadW9 wp1 wp0
    let wp17 := msgScheduleNext btcPadW15 btcPadW10 wp2 wp1
    let wp18 := msgScheduleNext wp16 btcPadW11 wp3 wp2
    let b2_s14 := sha256Round b2_s13 K13 wp13
    let b2_s15 := sha256Round b2_s14 K14 wp14
    let b2_s16 := sha256Round b2_s15 K15 wp15
    let b2_s17 := sha256Round b2_s16 K16 wp16
    let b2_s18 := sha256Round b2_s17 K17 wp17
    let b2_s19 := sha256Round b2_s18 K18 wp18
    let v0 := Sigma0 b2_s19.c
    let v1 := b2_s19.e - b2_s19.a
    let v2 := Sigma0 b2_s19.b
    let v3 := maj32 b2_s19.b b2_s19.c b2_s19.d
    let v4 := v2 + v3
    let v5 := v1 + v4
    let v6 := maj32 b2_s19.c b2_s19.d v5
    let v7 := v0 + v6
    let v8 := b2_s19.b - v7
    let v9 := Sigma1 b2_s19.g
    let v10 := v8 - v9
    let v11 := b2_s19.a - v4
    let v12 := Sigma1 b2_s19.f
    let v13 := v11 - v12
    let v14 := ch32 b2_s19.f b2_s19.g b2_s19.h
    let v15 := v13 - v14
    let v16 := v15 - 0x0fc19dc6#32
    let v17 := 0xbb67ae85#32 + b1_s0.b
    let v18 := sigma0 v17
    let v19 := 0x6a09e667#32 + b1_s0.a
    let v20 := v18 + v19
    let v21 := sigma1 v20
    let v22 := 0xa54ff53a#32 + b1_s0.d
    let v23 := sigma0 v22
    let v24 := v21 + v23
    let v25 := 0x3c6ef372#32 + b1_s0.c
    let v26 := v24 + v25
    let v27 := v16 - v26
    let v28 := ch32 b2_s19.g b2_s19.h v27
    let v29 := v10 - v28
    let v30 := v29 - 0xefbe4786#32
    let v31 := sigma1 0x00000100#32
    let v32 := sigma0 v25
    let v33 := v31 + v32
    let v34 := v33 + v17
    let v35 := v30 - v34
    let v36 := b2_s19.f - b2_s19.b
    let v37 := v36 + v7
    let v38 := v35 - v37
    let v39 := b2_s19.g - b2_s19.c
    let v40 := Sigma0 b2_s19.d
    let v41 := maj32 b2_s19.d v5 v37
    let v42 := v40 + v41
    let v43 := v39 + v42
    let v44 := Sigma0 v43
    let v45 := b2_s19.h - b2_s19.d
    let v46 := Sigma0 v5
    let v47 := maj32 v5 v37 v43
    let v48 := v46 + v47
    let v49 := v45 + v48
    let v50 := v27 - v5
    let v51 := Sigma0 v37
    let v52 := maj32 v37 v43 v49
    let v53 := v51 + v52
    let v54 := v50 + v53
    let v55 := maj32 v43 v49 v54
    let v56 := v44 - v55
    let v57 := v38 + v56
    0x00000001#32 &&& v57 = 0x00000001#32 &&& (b2_s13.d) := by
  intro wp0 wp1 wp2 wp3 wp4 wp5 wp6 wp7 wp12 wp13 wp14 wp15 wp16 wp17 wp18 b2_s14 b2_s15 b2_s16 b2_s17 b2_s18 b2_s19 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52 v53 v54 v55 v56 v57
  have hv5 : v5 = b2_s15.a := by
    change (b2_s19.e - b2_s19.a) + ((Sigma0 b2_s19.b) + (maj32 b2_s19.b b2_s19.c b2_s19.d)) = b2_s15.a
    exact round_backSolveA b2_s18 K18 wp18
  have hv17 : v17 = wp1 := by
    change 0xbb67ae85#32 + b1_s0.b = wp1
    rfl
  have hv19 : v19 = wp0 := by
    change 0x6a09e667#32 + b1_s0.a = wp0
    rfl
  have hv20 : v20 = wp16 := by
    change (sigma0 v17) + v19 = wp16
    rw [hv17, hv19]
    exact wp16_collapse wp1 wp0
  have hv22 : v22 = wp3 := by
    change 0xa54ff53a#32 + b1_s0.d = wp3
    rfl
  have hv25 : v25 = wp2 := by
    change 0x3c6ef372#32 + b1_s0.c = wp2
    rfl
  have hv26 : v26 = wp18 := by
    change ((sigma1 v20) + (sigma0 v22)) + v25 = wp18
    rw [hv20, hv22, hv25]
    exact wp18_collapse wp16 wp3 wp2
  have hv27 : v27 = b2_s15.e := by
    change ((((b2_s19.a - ((Sigma0 b2_s19.b) + (maj32 b2_s19.b b2_s19.c b2_s19.d))) - (Sigma1 b2_s19.f)) - (ch32 b2_s19.f b2_s19.g b2_s19.h)) - 0x0fc19dc6#32) - v26 = b2_s15.e
    rw [hv26]
    exact round_extract_h b2_s18 K18 wp18
  have hv34 : v34 = wp17 := by
    change ((sigma1 0x00000100#32) + (sigma0 v25)) + v17 = wp17
    rw [hv25, hv17]
    exact wp17_collapse wp2 wp1
  have hv35 : v35 = b2_s14.e := by
    change ((((b2_s19.b - ((Sigma0 b2_s19.c) + (maj32 b2_s19.c b2_s19.d v5))) - (Sigma1 b2_s19.g)) - (ch32 b2_s19.g b2_s19.h v27)) - 0xefbe4786#32) - v34 = b2_s14.e
    rw [hv5, hv27, hv34]
    exact round_extract_h b2_s17 K17 wp17
  have hv37 : v37 = b2_s14.a := by
    change (b2_s19.f - b2_s19.b) + ((Sigma0 b2_s19.c) + (maj32 b2_s19.c b2_s19.d v5)) = b2_s14.a
    rw [hv5]
    exact round_backSolveA b2_s17 K17 wp17
  have hv43 : v43 = b2_s13.a := by
    change (b2_s19.g - b2_s19.c) + ((Sigma0 b2_s19.d) + (maj32 b2_s19.d v5 v37)) = b2_s13.a
    rw [hv5, hv37]
    exact round_backSolveA b2_s16 K16 wp16
  have hv49 : v49 = b2_s13.b := by
    change (b2_s19.h - b2_s19.d) + ((Sigma0 v5) + (maj32 v5 v37 v43)) = b2_s13.b
    rw [hv5, hv37, hv43]
    exact round_backSolveA b2_s15 K15 wp15
  have hv54 : v54 = b2_s13.c := by
    change (v27 - v5) + ((Sigma0 v37) + (maj32 v37 v43 v49)) = b2_s13.c
    rw [hv27, hv5, hv37, hv43, hv49]
    exact round_backSolveA b2_s14 K14 wp14
  change 1#32 &&& ((v35 - v37) + (Sigma0 v43 - maj32 v43 v49 v54)) = 1#32 &&& b2_s13.d
  have hfin : (b2_s14.e - b2_s14.a) + (Sigma0 b2_s13.a + maj32 b2_s13.a b2_s13.b b2_s13.c) = b2_s13.d := round_backSolveA b2_s13 K13 wp13
  rw [lsb_add_sub_eq_add_add, hv35, hv37, hv43, hv49, hv54, hfin]

/-- `p3_id_027`：Track B2 Block2 W'[9..14]=0 collapse (t=12..17): invert across W'[16..17] eliminating Block1 E。 -/
theorem p3_id_027 (b1_s0 : SHAState) (b2_s12 : SHAState) :
    let wp0 := stdIV.a + b1_s0.a
    let wp1 := stdIV.b + b1_s0.b
    let wp2 := stdIV.c + b1_s0.c
    let wp3 := stdIV.d + b1_s0.d
    let wp4 := stdIV.e + b1_s0.e
    let wp5 := stdIV.f + b1_s0.f
    let wp6 := stdIV.g + b1_s0.g
    let wp7 := stdIV.h + b1_s0.h
    let wp12 : BitVec 32 := btcPadW12
    let wp13 : BitVec 32 := btcPadW13
    let wp14 : BitVec 32 := btcPadW14
    let wp15 : BitVec 32 := btcPadW15
    let wp16 := msgScheduleNext btcPadW14 btcPadW9 wp1 wp0
    let wp17 := msgScheduleNext btcPadW15 btcPadW10 wp2 wp1
    let wp18 := msgScheduleNext wp16 btcPadW11 wp3 wp2
    let b2_s13 := sha256Round b2_s12 K12 wp12
    let b2_s14 := sha256Round b2_s13 K13 wp13
    let b2_s15 := sha256Round b2_s14 K14 wp14
    let b2_s16 := sha256Round b2_s15 K15 wp15
    let b2_s17 := sha256Round b2_s16 K16 wp16
    let b2_s18 := sha256Round b2_s17 K17 wp17
    let v0 := b2_s18.f - b2_s18.b
    let v1 := Sigma0 b2_s18.c
    let v2 := b2_s18.e - b2_s18.a
    let v3 := Sigma0 b2_s18.b
    let v4 := maj32 b2_s18.b b2_s18.c b2_s18.d
    let v5 := v3 + v4
    let v6 := v2 + v5
    let v7 := maj32 b2_s18.c b2_s18.d v6
    let v8 := v1 + v7
    let v9 := v0 + v8
    let v10 := b2_s18.g - b2_s18.c
    let v11 := Sigma0 b2_s18.d
    let v12 := maj32 b2_s18.d v6 v9
    let v13 := v11 + v12
    let v14 := v10 + v13
    let v15 := Sigma0 v14
    let v16 := b2_s18.h - b2_s18.d
    let v17 := Sigma0 v6
    let v18 := maj32 v6 v9 v14
    let v19 := v17 + v18
    let v20 := v16 + v19
    let v21 := b2_s18.a - v5
    let v22 := Sigma1 b2_s18.f
    let v23 := v21 - v22
    let v24 := ch32 b2_s18.f b2_s18.g b2_s18.h
    let v25 := v23 - v24
    let v26 := v25 - 0xefbe4786#32
    let v27 := sigma1 0x00000100#32
    let v28 := 0x3c6ef372#32 + b1_s0.c
    let v29 := sigma0 v28
    let v30 := v27 + v29
    let v31 := 0xbb67ae85#32 + b1_s0.b
    let v32 := v30 + v31
    let v33 := v26 - v32
    let v34 := v33 - v6
    let v35 := Sigma0 v9
    let v36 := maj32 v9 v14 v20
    let v37 := v35 + v36
    let v38 := v34 + v37
    let v39 := maj32 v14 v20 v38
    let v40 := v15 + v39
    let v41 := v9 - v40
    let v42 := b2_s18.c - v13
    let v43 := Sigma1 b2_s18.h
    let v44 := v42 - v43
    let v45 := b2_s18.b - v8
    let v46 := Sigma1 b2_s18.g
    let v47 := v45 - v46
    let v48 := ch32 b2_s18.g b2_s18.h v33
    let v49 := v47 - v48
    let v50 := v49 - 0xe49b69c1#32
    let v51 := sigma0 v31
    let v52 := 0x6a09e667#32 + b1_s0.a
    let v53 := v51 + v52
    let v54 := v50 - v53
    let v55 := ch32 b2_s18.h v33 v54
    let v56 := v44 - v55
    let v57 := v56 - 0xc19bf174#32
    let v58 := v57 - 0x00000100#32
    let v59 := Sigma1 v58
    let v60 := v41 - v59
    let v61 := b2_s18.d - v19
    let v62 := Sigma1 v33
    let v63 := v61 - v62
    let v64 := ch32 v33 v54 v58
    let v65 := v63 - v64
    let v66 := v65 - 0x9bdc06a7#32
    let v67 := v66 - 0x00000000#32
    let v68 := v6 - v37
    let v69 := Sigma1 v54
    let v70 := v68 - v69
    let v71 := ch32 v54 v58 v67
    let v72 := v70 - v71
    let v73 := v72 - 0x80deb1fe#32
    let v74 := v73 - 0x00000000#32
    let v75 := ch32 v58 v67 v74
    let v76 := v60 - v75
    let v77 := v76 - 0x72be5d74#32
    let v78 := v77 - 0x00000000#32
    0xffffffff#32 &&& v78 = 0xffffffff#32 &&& (b2_s12.h) := by
  intro wp0 wp1 wp2 wp3 wp4 wp5 wp6 wp7 wp12 wp13 wp14 wp15 wp16 wp17 wp18 b2_s13 b2_s14 b2_s15 b2_s16 b2_s17 b2_s18 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52 v53 v54 v55 v56 v57 v58 v59 v60 v61 v62 v63 v64 v65 v66 v67 v68 v69 v70 v71 v72 v73 v74 v75 v76 v77 v78
  have hv6 : v6 = b2_s14.a := by
    change (b2_s18.e - b2_s18.a) + ((Sigma0 b2_s18.b) + (maj32 b2_s18.b b2_s18.c b2_s18.d)) = b2_s14.a
    exact round_backSolveA b2_s17 K17 wp17
  have hv9 : v9 = b2_s13.a := by
    change (b2_s18.f - b2_s18.b) + ((Sigma0 b2_s18.c) + (maj32 b2_s18.c b2_s18.d v6)) = b2_s13.a
    rw [hv6]
    exact round_backSolveA b2_s16 K16 wp16
  have hv14 : v14 = b2_s12.a := by
    change (b2_s18.g - b2_s18.c) + ((Sigma0 b2_s18.d) + (maj32 b2_s18.d v6 v9)) = b2_s12.a
    rw [hv6, hv9]
    exact round_backSolveA b2_s15 K15 wp15
  have hv20 : v20 = b2_s12.b := by
    change (b2_s18.h - b2_s18.d) + ((Sigma0 v6) + (maj32 v6 v9 v14)) = b2_s12.b
    rw [hv6, hv9, hv14]
    exact round_backSolveA b2_s14 K14 wp14
  have hv28 : v28 = wp2 := by
    change 0x3c6ef372#32 + b1_s0.c = wp2
    rfl
  have hv31 : v31 = wp1 := by
    change 0xbb67ae85#32 + b1_s0.b = wp1
    rfl
  have hv32 : v32 = wp17 := by
    change ((sigma1 0x00000100#32) + (sigma0 v28)) + v31 = wp17
    rw [hv28, hv31]
    exact wp17_collapse wp2 wp1
  have hv33 : v33 = b2_s14.e := by
    change ((((b2_s18.a - ((Sigma0 b2_s18.b) + (maj32 b2_s18.b b2_s18.c b2_s18.d))) - (Sigma1 b2_s18.f)) - (ch32 b2_s18.f b2_s18.g b2_s18.h)) - 0xefbe4786#32) - v32 = b2_s14.e
    rw [hv32]
    exact round_extract_h b2_s17 K17 wp17
  have hv38 : v38 = b2_s12.c := by
    change (v33 - v6) + ((Sigma0 v9) + (maj32 v9 v14 v20)) = b2_s12.c
    rw [hv33, hv6, hv9, hv14, hv20]
    exact round_backSolveA b2_s13 K13 wp13
  have hv52 : v52 = wp0 := by
    change 0x6a09e667#32 + b1_s0.a = wp0
    rfl
  have hv53 : v53 = wp16 := by
    change (sigma0 v31) + v52 = wp16
    rw [hv31, hv52]
    exact wp16_collapse wp1 wp0
  have hv54 : v54 = b2_s13.e := by
    change ((((b2_s18.b - ((Sigma0 b2_s18.c) + (maj32 b2_s18.c b2_s18.d v6))) - (Sigma1 b2_s18.g)) - (ch32 b2_s18.g b2_s18.h v33)) - 0xe49b69c1#32) - v53 = b2_s13.e
    rw [hv6, hv33, hv53]
    exact round_extract_h b2_s16 K16 wp16
  have hv58 : v58 = b2_s12.e := by
    change ((((b2_s18.c - ((Sigma0 b2_s18.d) + (maj32 b2_s18.d v6 v9))) - (Sigma1 b2_s18.h)) - (ch32 b2_s18.h v33 v54)) - 0xc19bf174#32) - 0x00000100#32 = b2_s12.e
    rw [hv6, hv9, hv33, hv54]
    exact round_extract_h b2_s15 K15 wp15
  have hv66 : v66 = b2_s12.f := by
    change (((b2_s18.d - ((Sigma0 v6) + (maj32 v6 v9 v14))) - (Sigma1 v33)) - (ch32 v33 v54 v58)) - 0x9bdc06a7#32 = b2_s12.f
    rw [hv6, hv9, hv14, hv33, hv54, hv58]
    exact round_extract_hw_zero b2_s14 K14
  have hv67 : v67 = b2_s12.f := by
    change v66 - 0x00000000#32 = b2_s12.f
    rw [hv66]
    exact bv_sub_zero _
  have hv73 : v73 = b2_s12.g := by
    change (((v6 - ((Sigma0 v9) + (maj32 v9 v14 v20))) - (Sigma1 v54)) - (ch32 v54 v58 v67)) - 0x80deb1fe#32 = b2_s12.g
    rw [hv6, hv9, hv14, hv20, hv54, hv58, hv67]
    exact round_extract_hw_zero b2_s13 K13
  have hv74 : v74 = b2_s12.g := by
    change v73 - 0x00000000#32 = b2_s12.g
    rw [hv73]
    exact bv_sub_zero _
  have hv77 : v77 = b2_s12.h := by
    change (((v9 - ((Sigma0 v14) + (maj32 v14 v20 v38))) - (Sigma1 v58)) - (ch32 v58 v67 v74)) - 0x72be5d74#32 = b2_s12.h
    rw [hv9, hv14, hv20, hv38, hv58, hv67, hv74]
    exact round_extract_hw_zero b2_s12 K12
  have hv78 : v78 = b2_s12.h := by
    change v77 - 0x00000000#32 = b2_s12.h
    rw [hv77]
    exact bv_sub_zero _
  rw [hv78]

/-- `p3_id_028`：Track B2 Block2 W'[9..14]=0 collapse (t=13..18): invert across W'[16..18] eliminating Block1 E。 -/
theorem p3_id_028 (b1_s0 : SHAState) (b2_s13 : SHAState) :
    let wp0 := stdIV.a + b1_s0.a
    let wp1 := stdIV.b + b1_s0.b
    let wp2 := stdIV.c + b1_s0.c
    let wp3 := stdIV.d + b1_s0.d
    let wp4 := stdIV.e + b1_s0.e
    let wp5 := stdIV.f + b1_s0.f
    let wp6 := stdIV.g + b1_s0.g
    let wp7 := stdIV.h + b1_s0.h
    let wp12 : BitVec 32 := btcPadW12
    let wp13 : BitVec 32 := btcPadW13
    let wp14 : BitVec 32 := btcPadW14
    let wp15 : BitVec 32 := btcPadW15
    let wp16 := msgScheduleNext btcPadW14 btcPadW9 wp1 wp0
    let wp17 := msgScheduleNext btcPadW15 btcPadW10 wp2 wp1
    let wp18 := msgScheduleNext wp16 btcPadW11 wp3 wp2
    let b2_s14 := sha256Round b2_s13 K13 wp13
    let b2_s15 := sha256Round b2_s14 K14 wp14
    let b2_s16 := sha256Round b2_s15 K15 wp15
    let b2_s17 := sha256Round b2_s16 K16 wp16
    let b2_s18 := sha256Round b2_s17 K17 wp17
    let b2_s19 := sha256Round b2_s18 K18 wp18
    let v0 := b2_s19.f - b2_s19.b
    let v1 := Sigma0 b2_s19.c
    let v2 := b2_s19.e - b2_s19.a
    let v3 := Sigma0 b2_s19.b
    let v4 := maj32 b2_s19.b b2_s19.c b2_s19.d
    let v5 := v3 + v4
    let v6 := v2 + v5
    let v7 := maj32 b2_s19.c b2_s19.d v6
    let v8 := v1 + v7
    let v9 := v0 + v8
    let v10 := b2_s19.g - b2_s19.c
    let v11 := Sigma0 b2_s19.d
    let v12 := maj32 b2_s19.d v6 v9
    let v13 := v11 + v12
    let v14 := v10 + v13
    let v15 := Sigma0 v14
    let v16 := b2_s19.h - b2_s19.d
    let v17 := Sigma0 v6
    let v18 := maj32 v6 v9 v14
    let v19 := v17 + v18
    let v20 := v16 + v19
    let v21 := b2_s19.a - v5
    let v22 := Sigma1 b2_s19.f
    let v23 := v21 - v22
    let v24 := ch32 b2_s19.f b2_s19.g b2_s19.h
    let v25 := v23 - v24
    let v26 := v25 - 0x0fc19dc6#32
    let v27 := 0xbb67ae85#32 + b1_s0.b
    let v28 := sigma0 v27
    let v29 := 0x6a09e667#32 + b1_s0.a
    let v30 := v28 + v29
    let v31 := sigma1 v30
    let v32 := 0xa54ff53a#32 + b1_s0.d
    let v33 := sigma0 v32
    let v34 := v31 + v33
    let v35 := 0x3c6ef372#32 + b1_s0.c
    let v36 := v34 + v35
    let v37 := v26 - v36
    let v38 := v37 - v6
    let v39 := Sigma0 v9
    let v40 := maj32 v9 v14 v20
    let v41 := v39 + v40
    let v42 := v38 + v41
    let v43 := maj32 v14 v20 v42
    let v44 := v15 + v43
    let v45 := v9 - v44
    let v46 := b2_s19.c - v13
    let v47 := Sigma1 b2_s19.h
    let v48 := v46 - v47
    let v49 := b2_s19.b - v8
    let v50 := Sigma1 b2_s19.g
    let v51 := v49 - v50
    let v52 := ch32 b2_s19.g b2_s19.h v37
    let v53 := v51 - v52
    let v54 := v53 - 0xefbe4786#32
    let v55 := sigma1 0x00000100#32
    let v56 := sigma0 v35
    let v57 := v55 + v56
    let v58 := v57 + v27
    let v59 := v54 - v58
    let v60 := ch32 b2_s19.h v37 v59
    let v61 := v48 - v60
    let v62 := v61 - 0xe49b69c1#32
    let v63 := v62 - v30
    let v64 := Sigma1 v63
    let v65 := v45 - v64
    let v66 := b2_s19.d - v19
    let v67 := Sigma1 v37
    let v68 := v66 - v67
    let v69 := ch32 v37 v59 v63
    let v70 := v68 - v69
    let v71 := v70 - 0xc19bf174#32
    let v72 := v71 - 0x00000100#32
    let v73 := v6 - v41
    let v74 := Sigma1 v59
    let v75 := v73 - v74
    let v76 := ch32 v59 v63 v72
    let v77 := v75 - v76
    let v78 := v77 - 0x9bdc06a7#32
    let v79 := v78 - 0x00000000#32
    let v80 := ch32 v63 v72 v79
    let v81 := v65 - v80
    let v82 := v81 - 0x80deb1fe#32
    let v83 := v82 - 0x00000000#32
    0xffffffff#32 &&& v83 = 0xffffffff#32 &&& (b2_s13.h) := by
  intro wp0 wp1 wp2 wp3 wp4 wp5 wp6 wp7 wp12 wp13 wp14 wp15 wp16 wp17 wp18 b2_s14 b2_s15 b2_s16 b2_s17 b2_s18 b2_s19 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 v29 v30 v31 v32 v33 v34 v35 v36 v37 v38 v39 v40 v41 v42 v43 v44 v45 v46 v47 v48 v49 v50 v51 v52 v53 v54 v55 v56 v57 v58 v59 v60 v61 v62 v63 v64 v65 v66 v67 v68 v69 v70 v71 v72 v73 v74 v75 v76 v77 v78 v79 v80 v81 v82 v83
  have hv6 : v6 = b2_s15.a := by
    change (b2_s19.e - b2_s19.a) + ((Sigma0 b2_s19.b) + (maj32 b2_s19.b b2_s19.c b2_s19.d)) = b2_s15.a
    exact round_backSolveA b2_s18 K18 wp18
  have hv9 : v9 = b2_s14.a := by
    change (b2_s19.f - b2_s19.b) + ((Sigma0 b2_s19.c) + (maj32 b2_s19.c b2_s19.d v6)) = b2_s14.a
    rw [hv6]
    exact round_backSolveA b2_s17 K17 wp17
  have hv14 : v14 = b2_s13.a := by
    change (b2_s19.g - b2_s19.c) + ((Sigma0 b2_s19.d) + (maj32 b2_s19.d v6 v9)) = b2_s13.a
    rw [hv6, hv9]
    exact round_backSolveA b2_s16 K16 wp16
  have hv20 : v20 = b2_s13.b := by
    change (b2_s19.h - b2_s19.d) + ((Sigma0 v6) + (maj32 v6 v9 v14)) = b2_s13.b
    rw [hv6, hv9, hv14]
    exact round_backSolveA b2_s15 K15 wp15
  have hv27 : v27 = wp1 := by
    change 0xbb67ae85#32 + b1_s0.b = wp1
    rfl
  have hv29 : v29 = wp0 := by
    change 0x6a09e667#32 + b1_s0.a = wp0
    rfl
  have hv30 : v30 = wp16 := by
    change (sigma0 v27) + v29 = wp16
    rw [hv27, hv29]
    exact wp16_collapse wp1 wp0
  have hv32 : v32 = wp3 := by
    change 0xa54ff53a#32 + b1_s0.d = wp3
    rfl
  have hv35 : v35 = wp2 := by
    change 0x3c6ef372#32 + b1_s0.c = wp2
    rfl
  have hv36 : v36 = wp18 := by
    change ((sigma1 v30) + (sigma0 v32)) + v35 = wp18
    rw [hv30, hv32, hv35]
    exact wp18_collapse wp16 wp3 wp2
  have hv37 : v37 = b2_s15.e := by
    change ((((b2_s19.a - ((Sigma0 b2_s19.b) + (maj32 b2_s19.b b2_s19.c b2_s19.d))) - (Sigma1 b2_s19.f)) - (ch32 b2_s19.f b2_s19.g b2_s19.h)) - 0x0fc19dc6#32) - v36 = b2_s15.e
    rw [hv36]
    exact round_extract_h b2_s18 K18 wp18
  have hv42 : v42 = b2_s13.c := by
    change (v37 - v6) + ((Sigma0 v9) + (maj32 v9 v14 v20)) = b2_s13.c
    rw [hv37, hv6, hv9, hv14, hv20]
    exact round_backSolveA b2_s14 K14 wp14
  have hv58 : v58 = wp17 := by
    change ((sigma1 0x00000100#32) + (sigma0 v35)) + v27 = wp17
    rw [hv35, hv27]
    exact wp17_collapse wp2 wp1
  have hv59 : v59 = b2_s14.e := by
    change ((((b2_s19.b - ((Sigma0 b2_s19.c) + (maj32 b2_s19.c b2_s19.d v6))) - (Sigma1 b2_s19.g)) - (ch32 b2_s19.g b2_s19.h v37)) - 0xefbe4786#32) - v58 = b2_s14.e
    rw [hv6, hv37, hv58]
    exact round_extract_h b2_s17 K17 wp17
  have hv63 : v63 = b2_s13.e := by
    change ((((b2_s19.c - ((Sigma0 b2_s19.d) + (maj32 b2_s19.d v6 v9))) - (Sigma1 b2_s19.h)) - (ch32 b2_s19.h v37 v59)) - 0xe49b69c1#32) - v30 = b2_s13.e
    rw [hv6, hv9, hv37, hv59, hv30]
    exact round_extract_h b2_s16 K16 wp16
  have hv72 : v72 = b2_s13.f := by
    change ((((b2_s19.d - ((Sigma0 v6) + (maj32 v6 v9 v14))) - (Sigma1 v37)) - (ch32 v37 v59 v63)) - 0xc19bf174#32) - 0x00000100#32 = b2_s13.f
    rw [hv6, hv9, hv14, hv37, hv59, hv63]
    exact round_extract_h b2_s15 K15 wp15
  have hv78 : v78 = b2_s13.g := by
    change (((v6 - ((Sigma0 v9) + (maj32 v9 v14 v20))) - (Sigma1 v59)) - (ch32 v59 v63 v72)) - 0x9bdc06a7#32 = b2_s13.g
    rw [hv6, hv9, hv14, hv20, hv59, hv63, hv72]
    exact round_extract_hw_zero b2_s14 K14
  have hv79 : v79 = b2_s13.g := by
    change v78 - 0x00000000#32 = b2_s13.g
    rw [hv78]
    exact bv_sub_zero _
  have hv82 : v82 = b2_s13.h := by
    change (((v9 - ((Sigma0 v14) + (maj32 v14 v20 v42))) - (Sigma1 v63)) - (ch32 v63 v72 v79)) - 0x80deb1fe#32 = b2_s13.h
    rw [hv9, hv14, hv20, hv42, hv63, hv72, hv79]
    exact round_extract_hw_zero b2_s13 K13
  have hv83 : v83 = b2_s13.h := by
    change v82 - 0x00000000#32 = b2_s13.h
    rw [hv82]
    exact bv_sub_zero _
  rw [hv83]

end P3Verify.TrackB2
