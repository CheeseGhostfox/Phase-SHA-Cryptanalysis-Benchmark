import P15Verify.Primitives

/-!
# P15Verify.Stage2BTC: BTC Double-SHA256 固定填充与跨块非干涉定理形式化证真

覆盖 `results/lean_candidates.json` 中全部 5 条 Stage 2 / BTC 候选，
以及 Block 1b 320-bit 固定填充下的 `W[16..17]` Nonce 无关性定理。
-/

namespace P15Verify.Stage2BTC
open P15Verify

/-- `p15_s2b_block2_16r_mitm` 前向非干涉：BTC Block 2（`S'_0 = stdIV`），`S'_1` 对 Blue `W'[7]` 严格免疫。 -/
theorem p15_s2b_block2_16r_mitm_fwd (w0 w7 w7' : BitVec 32) :
    let run := fun u7 =>
      let _ := u7
      sha256Round stdIV K0 w0
    run w7 = run w7' := by
  rfl

/-- `p15_s2b_block2_16r_mitm` 后向非干涉：含固定 256-bit 填充 `btcB2PadW8..15`，`S'_1` 对 Red `W'[0]` 严格免疫。 -/
theorem p15_s2b_block2_16r_mitm_bwd (s16 : SHAState)
    (w0 w0' w1 w2 w3 w4 w5 w6 w7 : BitVec 32) :
    let run := fun u0 =>
      let _ := u0
      let s15 := sha256RoundInv s16 K15 btcB2PadW15
      let s14 := sha256RoundInv s15 K14 btcB2PadW14
      let s13 := sha256RoundInv s14 K13 btcB2PadW13
      let s12 := sha256RoundInv s13 K12 btcB2PadW12
      let s11 := sha256RoundInv s12 K11 btcB2PadW11
      let s10 := sha256RoundInv s11 K10 btcB2PadW10
      let s9  := sha256RoundInv s10 K9  btcB2PadW9
      let s8  := sha256RoundInv s9  K8  btcB2PadW8
      let s7  := sha256RoundInv s8  K7  w7
      let s6  := sha256RoundInv s7  K6  w6
      let s5  := sha256RoundInv s6  K5  w5
      let s4  := sha256RoundInv s5  K4  w4
      let s3  := sha256RoundInv s4  K3  w3
      let s2  := sha256RoundInv s3  K2  w2
      sha256RoundInv s2 K1 w1
    run w0 = run w0' := by
  rfl

/-- `p15_btc_crossblock_wp4_7_decouple`（消息扩展部分）：BTC Block 2 固定填充下，
`W'[16], W'[17], W'[18]` 对 Block 1b `E`-轨输出 `W'[4..7]` 任意扰动严格免疫（`bv_decide` 证死）。 -/
theorem p15_btc_crossblock_wp4_7_msg (wp0 wp1 wp2 wp3 : BitVec 32) :
    let wp16 := msgScheduleNext btcB2PadW14 btcB2PadW9 wp1 wp0
    let wp17 := msgScheduleNext btcB2PadW15 btcB2PadW10 wp2 wp1
    let wp18 := msgScheduleNext wp16 btcB2PadW11 wp3 wp2
    (wp16 = sigma0 wp1 + wp0) ∧
    (wp17 = sigma1 btcB2PadW15 + sigma0 wp2 + wp1) ∧
    (wp18 = sigma1 (sigma0 wp1 + wp0) + sigma0 wp3 + wp2) := by
  refine ⟨b2_wp16_collapse wp1 wp0, b2_wp17_collapse wp2 wp1, ?_⟩
  rw [b2_wp16_collapse wp1 wp0]
  exact b2_wp18_collapse (sigma0 wp1 + wp0) wp3 wp2

/-- `p15_btc_crossblock_wp4_7_decouple`（后向状态部分）：从 Block 2 `S'_8` 经 `t'=7..4` 倒推 `S'_4`，
其 `A`-轨 `(A_4, A_3, A_2, A_1)` 对 Block 1b `E`-轨输出 `W'[4..7]` 任意扰动严格免疫。 -/
theorem p15_btc_crossblock_wp4_7_bwd (sp8 : SHAState)
    (wp4 wp5 wp6 wp7 wp4' wp5' wp6' wp7' : BitVec 32) :
    let run := fun u4 u5 u6 u7 =>
      let sp7 := sha256RoundInv sp8 K7 u7
      let sp6 := sha256RoundInv sp7 K6 u6
      let sp5 := sha256RoundInv sp6 K5 u5
      sha256RoundInv sp5 K4 u4
    Eq4 (run wp4 wp5 wp6 wp7) (run wp4' wp5' wp6' wp7') := by
  dsimp only
  apply inv_eq4_of_eq5
  apply inv_eq5_of_eq6
  apply inv_eq6_of_eq7
  apply inv_eq7_of_eq
  rfl

/-- `p15_backsolve_4r_chunk`：24 轮无调度独立块，`W[0]` 与 `W[20..23]` 在 `@r20` 严格双向非干涉。 -/
theorem p15_backsolve_4r_chunk_fwd (s0 : SHAState)
    (w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19
     w20 w21 w22 w23 w20' w21' w22' w23' : BitVec 32) :
    let run := fun u20 u21 u22 u23 =>
      let _ := (u20, u21, u22, u23)
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
      let s15 := sha256Round s14 K14 w14
      let s16 := sha256Round s15 K15 w15
      let s17 := sha256Round s16 K16 w16
      let s18 := sha256Round s17 K17 w17
      let s19 := sha256Round s18 K18 w18
      sha256Round s19 K19 w19
    run w20 w21 w22 w23 = run w20' w21' w22' w23' := by
  rfl

theorem p15_backsolve_4r_chunk_bwd (s24 : SHAState)
    (w0 w0' w20 w21 w22 w23 : BitVec 32) :
    let run := fun u0 =>
      let _ := u0
      let s23 := sha256RoundInv s24 K23 w23
      let s22 := sha256RoundInv s23 K22 w22
      let s21 := sha256RoundInv s22 K21 w21
      sha256RoundInv s21 K20 w20
    run w0 = run w0' := by
  rfl

/-- `p15_btc_b2_r0_16_chunk`：BTC Block 2 固定填充下，`W[0..1]` 与 `W[6..7]` 在 `@r6` 双向非干涉。 -/
theorem p15_btc_b2_r0_16_chunk_fwd (s0 : SHAState)
    (w0 w1 w2 w3 w4 w5 w6 w6' w7 w7' : BitVec 32) :
    let run := fun u6 u7 =>
      let _ := (u6, u7)
      let s1 := sha256Round s0 K0 w0
      let s2 := sha256Round s1 K1 w1
      let s3 := sha256Round s2 K2 w2
      let s4 := sha256Round s3 K3 w3
      let s5 := sha256Round s4 K4 w4
      sha256Round s5 K5 w5
    run w6 w7 = run w6' w7' := by
  rfl

theorem p15_btc_b2_r0_16_chunk_bwd (s16 : SHAState)
    (w0 w0' w1 w1' w6 w7 : BitVec 32) :
    let run := fun u0 u1 =>
      let _ := (u0, u1)
      let s15 := sha256RoundInv s16 K15 btcB2PadW15
      let s14 := sha256RoundInv s15 K14 btcB2PadW14
      let s13 := sha256RoundInv s14 K13 btcB2PadW13
      let s12 := sha256RoundInv s13 K12 btcB2PadW12
      let s11 := sha256RoundInv s12 K11 btcB2PadW11
      let s10 := sha256RoundInv s11 K10 btcB2PadW10
      let s9  := sha256RoundInv s10 K9  btcB2PadW9
      let s8  := sha256RoundInv s9  K8  btcB2PadW8
      let s7  := sha256RoundInv s8  K7  w7
      sha256RoundInv s7 K6 w6
    run w0 w1 = run w0' w1' := by
  rfl

/-- `p15_btc_b1b_r0_16_chunk`：BTC Block 1b（`W[4]=0x80000000, W[5..14]=0, W[15]=640`）`W[0..1]` 与 `W[2..3]` 在 `@r2` 双向非干涉。 -/
theorem p15_btc_b1b_r0_16_chunk_fwd (s0 : SHAState)
    (w0 w1 w2 w2' w3 w3' : BitVec 32) :
    let run := fun u2 u3 =>
      let _ := (u2, u3)
      let s1 := sha256Round s0 K0 w0
      sha256Round s1 K1 w1
    run w2 w3 = run w2' w3' := by
  rfl

theorem p15_btc_b1b_r0_16_chunk_bwd (s16 : SHAState)
    (w0 w0' w1 w1' w2 w3 : BitVec 32) :
    let run := fun u0 u1 =>
      let _ := (u0, u1)
      let s15 := sha256RoundInv s16 K15 btcB1PadW15
      let s14 := sha256RoundInv s15 K14 btcB1PadW14
      let s13 := sha256RoundInv s14 K13 btcB1PadW13
      let s12 := sha256RoundInv s13 K12 btcB1PadW12
      let s11 := sha256RoundInv s12 K11 btcB1PadW11
      let s10 := sha256RoundInv s11 K10 btcB1PadW10
      let s9  := sha256RoundInv s10 K9  btcB1PadW9
      let s8  := sha256RoundInv s9  K8  btcB1PadW8
      let s7  := sha256RoundInv s8  K7  btcB1PadW7
      let s6  := sha256RoundInv s7  K6  btcB1PadW6
      let s5  := sha256RoundInv s6  K5  btcB1PadW5
      let s4  := sha256RoundInv s5  K4  btcB1PadW4
      let s3  := sha256RoundInv s4  K3  w3
      sha256RoundInv s3 K2 w2
    run w0 w1 = run w0' w1' := by
  rfl

/-- BTC Block 1b 320-bit 固定填充下，`W[16]` 与 `W[17]` 对 Nonce `W[3]` 严格非干涉（`bv_decide` 证死）。 -/
theorem btc_b1b_w16_w17_indep_nonce (w0 w1 w2 w3 w3' : BitVec 32) :
    let w16 := msgScheduleNext btcB1PadW14 btcB1PadW9 w1 w0
    let w17 := msgScheduleNext btcB1PadW15 btcB1PadW10 w2 w1
    let w18 := fun u3 => msgScheduleNext w16 btcB1PadW7 u3 w2
    (w16 = sigma0 w1 + w0) ∧
    (w17 = sigma1 btcB1PadW15 + sigma0 w2 + w1) ∧
    (w18 w3 - sigma0 w3 = w18 w3' - sigma0 w3') := by
  refine ⟨b1_w16_collapse w1 w0, b1_w17_collapse w2 w1, ?_⟩
  dsimp only
  rw [b1_w16_collapse w1 w0, b1_w18_collapse, b1_w18_collapse]
  bv_decide

end P15Verify.Stage2BTC
