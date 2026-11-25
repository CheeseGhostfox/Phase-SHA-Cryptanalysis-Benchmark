import OuroborosV10

-- 内核公理审计：验证全部 6 大模块的核心定理仅依赖 Lean 4 标准内核公理，无任何占位符或自定义公理声明

-- 1. BoolGateLinearization
#print axioms OuroborosV10.BoolGateLinearization.bMaj_eq_or_form
#print axioms OuroborosV10.BoolGateLinearization.bCh_e_true
#print axioms OuroborosV10.BoolGateLinearization.bCh_e_false
#print axioms OuroborosV10.BoolGateLinearization.bCh_fg_eq
#print axioms OuroborosV10.BoolGateLinearization.bCh_fg_tf
#print axioms OuroborosV10.BoolGateLinearization.bCh_fg_ft
#print axioms OuroborosV10.BoolGateLinearization.bMaj_ab_eq
#print axioms OuroborosV10.BoolGateLinearization.bMaj_ac_eq
#print axioms OuroborosV10.BoolGateLinearization.bMaj_bc_eq
#print axioms OuroborosV10.BoolGateLinearization.bMaj_ab_ne
#print axioms OuroborosV10.BoolGateLinearization.bMaj_ac_ne
#print axioms OuroborosV10.BoolGateLinearization.bMaj_bc_ne
#print axioms OuroborosV10.BoolGateLinearization.bv32_ch_word_eq
#print axioms OuroborosV10.BoolGateLinearization.bv32_maj_word_eq

-- 2. RippleCarryAdder
#print axioms OuroborosV10.RippleCarryAdder.full_adder_arithmetic
#print axioms OuroborosV10.RippleCarryAdder.ripple_adder_exact_nat
#print axioms OuroborosV10.RippleCarryAdder.ripple_adder_mod_eq
#print axioms OuroborosV10.RippleCarryAdder.bv32_carry_fixed_point
#print axioms OuroborosV10.RippleCarryAdder.bv32_eval_add_carry_zero_iff

-- 3. PairAccDominance
#print axioms OuroborosV10.PairAccDominance.pair_acc_loss_iff_nat
#print axioms OuroborosV10.PairAccDominance.pair_acc_copy_dominates_nat
#print axioms OuroborosV10.PairAccDominance.pair_acc_copy_dominates_rat
#print axioms OuroborosV10.PairAccDominance.best_m3v2_empirical_copy_dominance
#print axioms OuroborosV10.PairAccDominance.false_confidence_sym_mul_lt
#print axioms OuroborosV10.PairAccDominance.false_confidence_nat_cross_mul

-- 4. NonlinearGuardBlocking
#print axioms OuroborosV10.NonlinearGuardBlocking.guarded_eq_preserves_sat
#print axioms OuroborosV10.NonlinearGuardBlocking.nonlinear_guard_blocks_unit_deduction

-- 5. CubeBudgetAndSplice
#print axioms OuroborosV10.CubeBudgetAndSplice.cube_profit_iff
#print axioms OuroborosV10.CubeBudgetAndSplice.cube_zero_prune_strict_loss
#print axioms OuroborosV10.CubeBudgetAndSplice.cube_zero_cost_nonneg
#print axioms OuroborosV10.CubeBudgetAndSplice.splice_block_take8_drops_w32
#print axioms OuroborosV10.CubeBudgetAndSplice.splice_block_mem_w32_lost
#print axioms OuroborosV10.CubeBudgetAndSplice.crystal_expansion_76x
#print axioms OuroborosV10.CubeBudgetAndSplice.num_gf2_vars_r64
#print axioms OuroborosV10.CubeBudgetAndSplice.dag_depth_w16_w18_w20
#print axioms OuroborosV10.CubeBudgetAndSplice.dag_depth_w17_w22
#print axioms OuroborosV10.CubeBudgetAndSplice.dag_depth_w19_w24
#print axioms OuroborosV10.CubeBudgetAndSplice.dag_depth_w21
#print axioms OuroborosV10.CubeBudgetAndSplice.dag_depth_w23
#print axioms OuroborosV10.CubeBudgetAndSplice.dag_depth_w25

-- 6. CNFEquisatisfiability
#print axioms OuroborosV10.CNFEquisatisfiability.unit_inject_equisat
#print axioms OuroborosV10.CNFEquisatisfiability.equiv_2lit_inject_equisat
#print axioms OuroborosV10.CNFEquisatisfiability.equiv_subst_equisat
#print axioms OuroborosV10.CNFEquisatisfiability.flp_unit_sound
#print axioms OuroborosV10.CNFEquisatisfiability.flp_pair_clause_sound
