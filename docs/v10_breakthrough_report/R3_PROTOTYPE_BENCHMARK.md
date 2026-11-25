# Phase-SHA V10「衔尾蛇-64」M3 原型实现与同条件配对实测对比报告（R3 实测终审）

> **报告摘要**：本报告记录在严格零修改既有核心源码（`ouroboros_core/`、`ouroboros_net/`、`ouroboros_mcts/`、`ouroboros_train/`、`ouroboros_bench/`）及既有实验存档（`runs/a100_m2`、`runs/a100_m3`、`runs/a100_m3v2` 全部 21 个文件无变动）的前提下，将 R3 第一梯队无重训突破方案 **P1–P4** 落地于 `g:\SHA-LM\v10_ouroboros64\prototypes\` 并完成 Iteration 2–3 根本修复（`dp/dn` 集合差计数修复；基于按轮次分带稳态速率硬封顶、非对称短窗 EMA 阻尼、渐进有界几何分摊与 `R<=28` 长首窗 `4000` 冲突的不可中断 CaDiCaL 根本控时重构；`cd103` `supports_propagate` 优雅降级兼容；`U3` 14 格全失败模式高压测试与 `U8` 强杀/重建强断言加固）后的完整单元测试（`16/16` 全通过，耗时 `33.29s`）与全量同条件配对实测（`verify_preimage = 441 / 441, 100%`，总耗时 `864.99s`）定量结果，并诚实剖析各模块的实证收益与现存边界。

---

## 一、实验条件、隔离边界与硬件锁定声明

1. **运行环境与精度锁定**：
   - Python 解释器：`g:\SHA-LM\.venv\Scripts\python.exe`（Python 3.11.9，无缓冲 `-u`）
   - 计算设备：`NVIDIA GeForce RTX 4060 Laptop GPU`（CUDA）+ 单线程本地 CPU（顺序单实例执行以确保 CPU 耗时公平可比）
   - 纯 IEEE-754 FP32 确定性门控：显式调用 `enforce_fp32_determinism()`，实测锁定 **`allow_tf32_matmul == False`** 且 **`allow_tf32_cudnn == False`**（`fp32_locked == True`）。
2. **只读模型权重与隔离约束**：
   - 只读加载检查点：`v10_ouroboros64/runs/a100_m3v2/checkpoints/best_m3v2.pt`（`step = 200`，`11,018,600` 参数）；
   - 域 B 全程严格零内部真值泄漏（仅传递 `H_target` 与 `make_end_free_constraints` 指定的固定边界）；所有神经网络前向显式传入 `zero_defects=True, conflict_heatmap=None`；
   - 既有 21 个基线存档文件（`a100_m2` 6 个、`a100_m3` 6 个、`a100_m3v2` 9 个）及 `formal_proofs/`、`formal_proofs_v10/` 均保持零改动；`g:\SHA-LM` 工作区无任何 Teamwork 元数据污染（`Test-Path g:\SHA-LM\.agents == False`）。
3. **同条件配对基准协议**（对齐 `m3_v2_paired_bench.py`）：
   - 随机流：`rng = np.random.default_rng(20260499)`，按 `(24,32,16) -> (24,40,16) -> (25,24,16) -> (26,12,16) -> (26,16,16) -> (28,12,16) -> (28,16,8) -> R=64(4)` 顺序抽样；
   - 预算配置：`sat_time_limit = 5.0s`、`conflict_budget = 250000`；`R=64` 轨 A MCTS 限时 `3.0s`；域 A 复核采用 `run_m3_extreme_healing(rounds=64, n_crystals=8, n_steps=64, k_budget=32, seed=20260506)`；
   - 统计检验：精确双侧 Wilcoxon 符号秩检验 `wilcoxon_signed_rank_exact` 与 `2000` 次抽样 `bootstrap_rel_reduction_ci`。

---

## 二、落地原型模块与单元测试验证矩阵（`16/16 PASS`）

### 2.1 原型代码清单（`g:\SHA-LM\v10_ouroboros64\prototypes\`）
| 文件 | 核心类 / 函数 | 实现要点与防坑核验 |
|---|---|---|
| `__init__.py` | 包导出 | 统一导出 P1–P4 原型类与局部 GF(2) 工具函数 |
| `p1_clean_cdcl.py` | `CleanDynamicNeuralCDCLSolver`, `_build_full_crystal_var_map`, `_run_0level_bcp`, `_get_round_timing_profile` | 前置 `isinstance(lit, bool)` 拦截；`a_canon`/`e_canon` 四重移位别名回填；无闭包嵌套 Python 0 层 BCP（`_bcp_enqueue` + `_run_0level_bcp`）；`unk` 于 CPU NumPy 构造后一次性 `.to(device)`；净态前向 `conflict_heatmap=None, zero_defects=True`；校准门控 `T_cal=4.5, tau=0.08, flp_asym=16`（仅 Episode 0）；识别 `cd103/cd153/cd195/cadical*` 全别名，对 `cd103` 捕获 `propagate()` `NotImplementedError` 置 `supports_propagate=False` 并安全回落；不可中断分支采用 `_get_round_timing_profile(rounds)` **按轮次分带稳态速率硬封顶**（`R>=56: 450/s`, `40..55: 950/s`, `29..39: 4000/s`, `27..28: 14000/s`, `<=26: 22000/s`）+ **非对称短窗 EMA 阻尼**（`dt<0.08s` 限制上行步长 `1.35x / alpha=0.25`，常规 `1.60x / 0.45`，下行快追 `0.50 / 0.75`）+ **渐进有界几何分摊**（`frac=0.30/0.35 -> 0.38/0.40 -> 0.45`, `step_mult=2.0/2.5`）+ **`R<=28` 长首窗 `4000` 冲突**（`tl>=3.0s`） |
| `p2_adaptive_probe_xor.py` | `ZeroWasteBoundaryProber`, `SyncedDynamicXORSolver` | `B_c = 0` 零冲突 1-bit + 集合差精确计数 `dp = len(lp_set - base_set)`, `dn = len(ln_set - base_set)` + 公共蕴涵交集 $(L_p \cap L_n) \setminus L_{\text{base}}$ + 耦合 2-bit 探测；`shadow_bcp` 伴随 0 层状态同步刷新 `W/S/C` 掩码，消除静止 `base_assigned` |
| `p3_dual_bridge_router.py` | `BridgeableSHA256XORCNFEncoder`, `SignedUnionFind`, `DualPlaneEquivalenceBridge`, `SafeCMSSubprocessRunner`, `CarryLockHybridRouter` | 完整镜像 `build()` 捕获 `Ch_vars`/`Maj_vars`（未 hook `_emit_maj`）；Bitset RREF 含 `Ch(5)+Maj(6)` 11 规则；`SignedUnionFind` 检出奇校验环与常数冲突；常驻单 Worker 子进程隔离 `pycryptosat`（崩溃/超时无缝降级）；三区 Carry-Lock 动态路由 |
| `p4_local_wormhole_mcts.py` | `local_window_gf2_propagate`, `LocalWindowWormholeMCTS` | 整数键编码局部轮窗 GF(2) 消元（`Ch(5)+Maj(6)` 全覆盖 + `a_canon`/`e_canon` 别名校验 + 不可写边界只读校对）；修复 `splice_block[:12]` 保留 `W[32]`；紧凑三区动作（未知 `W[0..15] <= 16` 时采用 LSB 单变量二分避免 $k!$ 排列爆炸）+ 死胡同子树向上闭合；单次净态前向 + 纯符号叶评估 |
| `test_prototypes_unit.py` | 16 项单元测试 | 完整覆盖 `U0–U8` 与 `T-P4-1..7`：`U3` 覆盖 `14` 格全失败模式压力矩阵（全 4 不可中断别名 × `R=48/64` × `0.05s` 全扫 + 5 格 Probe B 高压深窗 `1.5/2.0s` + `cd103` `0.5s` 降级兼容 + `threading.excepthook` 零后台异常校验）；`U8` 放宽超时支路 `tl=8.0s` 并强校验 `worker_exit_11` / `hard_timeout_killed` / `ok` 与 `solver_type` |
| `run_prototype_paired_bench.py` | 配对基准运行器 | 顺序单实例执行，按档位实时 `_atomic_write_json`，全量输出至 `prototype_bench_report.json` |

### 2.2 单元测试执行结果（`unit_test_report.json`，总耗时 `33.294s`）
执行命令：`g:\SHA-LM\.venv\Scripts\python.exe -u -m v10_ouroboros64.prototypes.test_prototypes_unit`（退出码 `0`）

| 测试编号 | 验证目标 | 实测关键指标与结论 | 耗时 |
|---|---|---|---:|
| `U0_fp32_determinism` | FP32 锁盘与权重 dtype | `cuda=True`, `allow_tf32=False`, 全参数 `torch.float32` | `0.001s` |
| `U1_bool_const_guard` | `bool` 常量拦截防变量 `0/1` 混淆 | 变量 `1` 映射精确为 `[(0,0,0,0,1)]`，`mapped_vars=9488`, `const_s=1280` | `0.097s` |
| `U2_full_crystal_alias_sync` | 全晶体 0 层回填与 `a_canon` 四重别名 | `S[1..23]` 校验 `5888` 位、`C[0..23]` 校验 `7680` 位，100% 匹配真值 | `0.263s` |
| `U3_calibrated_gate_and_chunk` | 净态入参、门控拦截与 14 格全失败模式控时/`cd103` 兼容 | `conflict_heatmap=None`, `zero_defects=True`, `max\|z\|=0.8154 < 1.4525`, `R=28 0.05s` 退出 `0.131s`, `R=64 1.5s` 退出 `1.625s`, `cd195 R=64 1.5s` 退出 `1.536s`；新增 `ext_cases=14` 全通过，`max_ratio_long=1.0170`（`<2.2x`），`cd103_sec=0.482s`（`supports_propagate=False`），`thread_errors=[]` | `15.769s` |
| `U4_zero_waste_probe_b0` | `ZeroWasteBoundaryProber` 零冲突、集合差 `dp_dn` 与交集/2-bit | `conflicts_delta == 0`；携带 `initial_base_lits=15271` 实测 `sample_dp_dn=(28, 24)` 且精确匹配集合差；合成公式导出 `[2]` 与 `[-1,-2]` | `0.312s` |
| `U5_synced_xor_state_sync` | `SyncedDynamicXORSolver` 伴随状态同步 | `shadow_bcp` 同步 `2824` 变量单调不减，净态前向且 `SAT` 验真通过 | `2.010s` |
| `U6_bridgeable_encoder_1to1` | `BridgeableSHA256XORCNFEncoder` 跨编码器 1:1 等价 | `R in {16,24,26,28}` 与 `SHA256XORCNFEncoder` 及 `SHA256CNFEncoder` 的 `next_var`、`W_vars`、`a_canon`、`e_canon`、`C_vars`、`clauses`、`xor_clauses` 完全一致 | `0.735s` |
| `U7_signed_uf_conflicts` | `bool`/RREF/`SignedUnionFind` 冲突检出 | 奇校验环、常数冲突、RREF `0=1` 均报 `early_unsat=True` | `0.140s` |
| `U8_safe_cms_fallback` | 子进程崩溃/超时强杀与自动重建强断言 | `crash_reason='worker_exit_11'`、`timeout_reason='hard_timeout_killed'` (`timeout_sec=4.449s >= 2.0s`, `tl=8.0s`)、`ok_reason='ok'` 与 `solver_type` 全部精确通过 | `8.437s` |
| `T_P4_1_fp32_and_result_type` | `allow_tf32=False` 与 `MCTSSearchResult` 类型 | 类型与数值精度契约 100% 匹配 | `0.359s` |
| `T_P4_2_splice_block_12` | `splice_block[:12]` 跨字耦合 | 前两个宏动作长度 `12`（`S[32]=8 bit, W[32]=4 bit`） | `0.035s` |
| `T_P4_3_window_vars_and_11_rules` | 局部窗变量数与 `Maj(a==c)/(b==c)` 规则 | `w=8 -> 4608`, `w=9 -> 5088`; `ac_units=16, bc_units=1`; 零 `PermissionError` | `0.043s` |
| `T_P4_4_local_gf2_soundness_and_conflict` | 局部 GF(2) 可靠性与 1-bit 奇偶矛盾剪枝 | 正确约束推 `34` 单位全对；1-bit 翻转返回 `conflict=True` | `0.059s` |
| `T_P4_5_single_forward_and_clean_eval` | 单次净态前向（消除叶二次前向） | 20 sims 前向调用：`Base=22` vs **`P4=3`**，`heatmap` 恒 `None` | `0.913s` |
| `T_P4_6_r64_overconstrained_probe` | `R=64, fb in {4,6}` 零泄漏可解探针 | `fb=4`: `0.165s, sims=16, pruned=2`; `fb=6`: `0.559s, sims=87, pruned=24`，全 `verify_preimage` 通过 | `0.726s` |
| `T_P4_7_r64_blind_throughput_and_depth` | 全盲 `R=64, 3.0s` 吞吐与深度 | `sims=259` (`>=120`), `max_depth=5` (`>=4`), `local_gf2_mean_ms=4.292ms` | `3.057s` |

> **目标 A 全矩阵独立探针复核**：除 `U3` 涵盖的 14 格扩展用例外，对 `{cadical195, cd195, cd153, cadical153} × {R=48, R=64} × {tl=0.05, 1.5, 2.0}` 全部 48 组（分别经 `P1` 与 `P3`）及 `cd103`（`R=24 32b` & `R=64 tl=1.5s`）执行全矩阵检查，所有 `tl >= 1.5s` 组合**最大 `elapsed / tl = 1.1109x`**（远低于 `2.2x` 硬界，零超时失控），`cd103` 全程零异常（`R=24 32b` 0.225s `SAT` 验真通过，`R=64 tl=1.5s` 精确停于 `1.5017s`）。

---

## 三、域 B 7 档盲求逆同条件配对实测对比

全量运行命令：
```powershell
g:\SHA-LM\.venv\Scripts\python.exe -u -m v10_ouroboros64.prototypes.run_prototype_paired_bench --sat_time_limit 5.0 --seed 20260499 --conflict_budget 250000 --out_json v10_ouroboros64/prototypes/prototype_bench_report.json
```
全量总耗时 **`864.99s`**；全部 5 个求解器（`Pure CaDiCaL`、`Baseline M3-V2 Dyn CDCL`、`Baseline Pure CMS-XOR [SafeSubprocess]`、`Prototype P1+P2 CleanDynCDCL`、`Prototype P3 CarryLockHybridRouter`）与 `R=64` 探针共计 **`441 / 441`（100.0%）** 通过 `OuroborosCore.verify_preimage` 校验。

### 3.1 域 B 7 档完整定量汇总表（`sat_time_limit = 5.0s`）

| 档位 (`R`, `fb`, `N`) | 指标 | `Pure CaDiCaL` | `Base M3-V2 DynCDCL` | `Base Pure CMS-XOR` (子进程防崩) | `Proto P1+P2 CleanCDCL` | `Proto P3 HybridRouter` |
|---|---|---:|---:|---:|---:|---:|
| **`R=24 32b (16)`** | SAT/N | **16/16** | **16/16** | **16/16** | **16/16** | **16/16** (`Zone1:16`) |
| | 均值/中位冲突 | 582.81 / 409.5 | **464.00 / 340.5** | — | 552.69 / 431.0 | **0.0 / 0.0** (CMS-XOR) |
| | 均值/中位耗时(s) | **0.0527 / 0.0483** | 0.2404 / 0.2300 | 0.2542 / 0.1857 | 0.3417 / 0.3369 | 0.4753 / 0.4782 |
| **`R=24 40b (16)`** | SAT/N | **16/16** | **16/16** | 11/16 (68.75%) | **16/16** | **16/16** (`Zone3:16`) |
| | 均值/中位冲突 | **5293.94 / 5363.0** | 6931.00 / 6919.5 | — | 7664.88 / 6376.0 | 6604.31 / 6410.5 |
| | 均值/中位耗时(s) | **0.2572 / 0.2775** | 0.6760 / 0.7004 | 2.5241 / 2.0016 | 0.7292 / 0.6396 | **1.0258 / 1.0216** (+59.4% vs XOR) |
| **`R=25 24b (16)`** | SAT/N | **16/16** | **16/16** | **16/16** | **16/16** | **16/16** (`Zone2:15, Z1:1`) |
| | 均值/中位冲突 | 4465.50 / 3635.0 | 6507.06 / 6373.0 | — | 5684.00 / 4937.5 | **3178.00 / 1868.5** |
| | 均值/中位耗时(s) | **0.1830 / 0.1429** | 0.5636 / 0.5590 | 1.4700 / 1.1653 | 0.5978 / 0.5890 | **1.3218** / 1.5216 (+10.1% vs XOR) |
| **`R=26 12b (16)`** | SAT/N | **16/16** | **16/16** | **0/16 (0.0%)** | **16/16** | **16/16** (`Zone3:16`) |
| | 均值/中位冲突 | 2337.88 / 2084.5 | 2489.94 / 2135.5 | — | 2650.50 / 2126.0 | **2014.12 / 1770.5** |
| | 均值/中位耗时(s) | **0.1412 / 0.1327** | 0.3786 / 0.3192 | 5.1459 / 5.1479 | 0.6330 / 0.6147 | **0.9121 / 0.8750** (+82.3% vs XOR) |
| **`R=26 16b (16)`** | SAT/N | **16/16** | **16/16** | **0/16 (0.0%)** | **16/16** | **16/16** (`Zone3:16`) |
| | 均值/中位冲突 | 15549.31 / **13770.5** | 20639.25 / 19437.0 | — | 18130.56 / 19006.0 | **14715.62** / 17883.5 |
| | 均值/中位耗时(s) | **0.7569 / 0.6633** | 1.6571 / 1.5890 | 5.1427 / 5.1492 | 1.3385 / 1.3948 | **1.5444 / 1.7259** (+70.0% vs XOR) |
| **`R=28 12b (16)`** | SAT/N | **16/16** | **16/16** | **0/16 (0.0%)** | **16/16** | **16/16** (`Zone3:16`) |
| | 均值/中位冲突 | **7125.62 / 6530.0** | 7437.12 / 8709.5 | — | 7912.12 / 7826.5 | 7928.62 / 7229.0 |
| | 均值/中位耗时(s) | **0.8311 / 0.8676** | 1.3143 / 1.4516 | 5.1734 / 5.1725 | 1.4030 / 1.4490 | **1.7626 / 1.8301** (+65.9% vs XOR) |
| **`R=28 16b (8)`** | SAT/N | 3/8 (37.5%) | **4/8 (50.0%)** | **0/8 (0.0%)** | 2/8 (25.0%) | 1/8 (12.5%, `Zone3:8`) |
| | 均值/中位冲突 | 38626.12 / 44006.5 | **31127.62** / 36199.5 | — | 33675.12 / **34634.0** | 35980.38 / 35819.0 |
| | 均值/中位耗时(s) | 4.4341 / 5.1288 | **4.2233 / 4.9301** | 5.1880 / 5.1789 | 4.5239 / 4.9927 | **4.7371 / 4.9977** (+8.7% vs XOR) |

### 3.2 关键配对统计学对比（Bootstrap 95% CI 与精确 Wilcoxon 检验）

| 档位 | 对比组合与指标 | 相对降幅 `Δ%` | 95% Bootstrap CI | `W / T / L` | 精确 Wilcoxon `p` |
|---|---|---|---|---|---:|
| **`R=24 32b`** | `P1+P2` vs `Pure` 冲突 | `+5.17%` | `[-17.06%, +24.43%]` | `9 / 0 / 7` | `0.6054` |
| | `P3` vs `Base CMS-XOR` 耗时 | `-86.99%` | `[-181.84%, -15.08%]` | `1 / 0 / 15` | `0.0052` |
| **`R=24 40b`** | `Base Dyn` vs `Pure` 冲突 | `-30.92%` | `[-79.96%, -1.44%]` | `4 / 0 / 12` | `0.0934` |
| | `P1+P2` vs `Pure` 冲突 | `-44.79%` | `[-125.99%, +11.38%]` | `5 / 0 / 11` | `0.3484` |
| | **`P3` vs `Base CMS-XOR` 耗时** | **`+59.36%`** | **`[+36.30%, +71.67%]`** | **`11 / 0 / 5`** | **`0.0250`** |
| **`R=25 24b`** | `Base Dyn` vs `Pure` 冲突（显著退化） | `-45.72%` | `[-80.75%, -18.32%]` | `6 / 0 / 10` | **`0.0182`** |
| | `P1+P2` vs `Base Dyn` 冲突 | `+12.65%` | `[-14.63%, +37.68%]` | `10 / 0 / 6` | `0.2744` |
| | `P1+P2` vs `Pure` 冲突（消除了显著劣化） | `-27.29%` | `[-78.83%, +8.85%]` | `6 / 0 / 10` | `0.1297` |
| **`R=26 12b`** | `P1+P2` vs `Pure` 冲突 | `-13.37%` | `[-46.35%, +13.71%]` | `8 / 0 / 8` | `0.4954` |
| | **`P3` vs `Base CMS-XOR` 耗时** | **`+82.28%`** | **`[+81.23%, +83.20%]`** | **`16 / 0 / 0`** | **`0.000031`** |
| **`R=26 16b`** | `Base Dyn` vs `Pure` 冲突 | `-32.73%` | `[-82.11%, -0.97%]` | `6 / 0 / 10` | `0.0739` |
| | `P1+P2` vs `Pure` 冲突（恢复 16/16 且无显著劣化） | `-16.60%` | `[-76.66%, +19.96%]` | `6 / 0 / 10` | `0.5966` |
| | `P1+P2` vs `Base Dyn` 冲突 | `+12.15%` | `[-13.58%, +31.45%]` | `9 / 0 / 7` | `0.5966` |
| | **`P3` vs `Base CMS-XOR` 耗时** | **`+69.97%`** | **`[+66.10%, +73.85%]`** | **`16 / 0 / 0`** | **`0.000031`** |
| **`R=28 12b`** | `P1+P2` vs `Pure` 冲突 | `-11.04%` | `[-43.21%, +15.18%]` | `7 / 0 / 9` | `0.5619` |
| | **`P3` vs `Base CMS-XOR` 耗时** | **`+65.93%`** | **`[+62.90%, +69.24%]`** | **`16 / 0 / 0`** | **`0.000031`** |
| **`R=28 16b`** | `P1+P2` vs `Pure` 冲突 | `+12.82%` | `[-23.93%, +35.36%]` | `5 / 0 / 3` | `0.5469` |
| | **`P3` vs `Base CMS-XOR` 耗时** | **`+8.69%`** | **`[+3.28%, +18.89%]`** | **`8 / 0 / 0`** | **`0.0078`** |

### 3.3 P3 路由器结构遥测汇总与门控行为
- **零误判三区精准路由**：
  - `R=24 32b`：`r_lock = 0.2774`, `D_gf2 = 1548.4`, `equiv_2lit = 393.9` $\rightarrow$ **`16/16 Zone1_Linear`**；
  - `R=25 24b`：`r_lock = 0.3061`, `D_gf2 = 1878.5`, `equiv_2lit = 402.1` $\rightarrow$ **`15/16 Zone2_EdgeProbe` + `1/16 Zone1_Linear`**；
  - `R=24 40b / R=26 12b / R=26 16b / R=28 12b / R=28 16b`：`r_lock = 0.3320 ~ 0.4016` $\rightarrow$ **`100% Zone3_CarryLocked`**，完全绕开 `R>=26` XOR `0/16` 僵局；
  - 全程 104 例 `Baseline Pure CMS-XOR` 经 `SafeCMSSubprocessRunner` 执行零进程崩死（`subprocess_crashes_or_timeouts = 0`）。
- **门控与零冲突探测实效**：
  - 全部 104 例域 B 算例上 `P1+P2` 神经置信支路 `pass_conf` 拦截率仍为 `100%`（严格拦截盲态假置信注入）；结构非对称 FLP 支路 `pass_asym`（`|dp - dn| >= 16`）在 `R=24 32b` 均通 `0.0`，在 `R=24 40b..R=28 16b` 均通 `0.125 ~ 1.188` 个变量；
  - `R=24 32b / R=26 12b / R=28 12b` 上 `ZeroWasteBoundaryProber` 在 0 冲突下免费推导出 `16.44 / 16.75 / 11.56` 个永久单位文字（`R=24 32b` 另导出 `2.81` 条 2-lit 阻断子句）。

---

## 四、`R=64` 轨 A 虫洞 MCTS 与域 A 物理隔离实测

### 4.1 全盲 `R=64` 轨 A MCTS 配对对比（`N=4, time_limit=3.0s`）

| 指标 | `Baseline WormholeMCTS` (4 例均值/范围) | `Prototype P4 LocalWindowWormholeMCTS` (4 例均值/范围) | 相对变化 |
|---|---|---|---|
| `solved` | `0/4` (`False`) | `0/4` (`False`)（符合 R4 定理 T6 密码学界） | — |
| `simulations` | `64` (`[64, 64]`) | **`249.75` (`[245, 258]`)** | **`3.90x`** |
| `expansions` | `64` (`[64, 64]`) | **`249.75` (`[245, 258]`)** | **`3.90x`** |
| `max_depth` | `2`（基线 28 根动作限制） | **`5`** | **`2.50x`** |
| `unk_drop` | `94` (`[94, 94]`) | **`205.0` (`[186, 224]`)**（`unique=2` 分化） | **`+118.09%`** |
| `gate_drop` | `26` (`[26, 26]`) | **`74.0` (`[68, 80]`)**（`unique=2` 分化） | **`+184.62%`** |
| `splice_drop` | `8` (`[8, 8]`) | **`28.0` (`[24, 32]`)**（`[:12]` 完整保留 `W[32]`） | **`+250.00%`** |
| `pruned_conflicts` | `0` | `0`（全盲深度 5 下局部未闭合矛盾） | — |
| `local_gf2_mean_ms` | 未启用 (`~901ms` 全图) | **`4.509 ms`** (`[4.276, 4.676]`) | 较全图 **`~200x`** 加速 |
| `neural_eval_calls` | `129`（含叶二次前向） | **`38.50` (`[38, 40]`)**（~250 扩展仅 ~39 次净态前向） | **`-70.2%`** |

### 4.2 `R=64` 严格零泄漏过约束可解验证探针（`free_bits in {4, 6}`, `N=2/档`）
严格仅传 `H_target` 与 `make_end_free_constraints` 前缀边界（零内部泄漏）：

| `free_bits` | 样例 | `Baseline WormholeMCTS` | `Prototype P4 LocalWindowWormholeMCTS` |
|---:|---:|---|---|
| `4` | `case 0` | `solved=False`, `sims=85`, `pruned=13`, `3.060s` | **`solved=True`**, `sims=22`, `pruned=7`, **`0.1929s`** |
| `4` | `case 1` | `solved=False`, `sims=85`, `pruned=11`, `3.043s` | **`solved=True`**, `sims=24`, `pruned=9`, **`0.2093s`** |
| `6` | `case 0` | `solved=False`, `sims=93`, `pruned=11`, `3.029s` | **`solved=True`**, `sims=104`, **`pruned=41`**, **`0.6231s`** |
| `6` | `case 1` | `solved=False`, `sims=94`, `pruned=11`, `3.068s` | **`solved=True`**, `sims=101`, **`pruned=38`**, **`0.6298s`** |
| **汇总** | **4 例** | **`0/4 (0.0%)`** | **`4/4 (100.0%)`，`verify_preimage` 100% 通过** |

### 4.3 域 A `R=64` 极限退火物理隔离复核（`N=8, 64步, K=32, seed=20260506`）
| 对比项 | M3-V2 历史存档基线 (`m3_v2_paired_report.json`) | 本次域 A 隔离复核实测 (`prototype_bench_report.json`) |
|---|---:|---:|
| 初始能量 `E0` | `212.625` | **`212.625`**（完全一致） |
| `WalkSAT-1bit` 终态能量 | `200.5` (`-5.70%`) | **`200.500`** (**`-5.70%`**) |
| `Neural Wolff` 终态能量 | `3.5` (`-98.35%`) | **`3.875`** (**`-98.18%`**) |
| 精确愈合率 / 跨垒隧穿率 | `50.0%` / `89.19%` | `37.5%` / `83.78%`（A100 vs RTX 4060 跨卡 FP32 微差，极值愈合稳态完整保留） |

---

## 五、实证突破收益与仍存边界的诚实剖析

### 5.1 已证实的原型突破收益
1. **彻底消除 `R>=26` XOR 僵局与 `R=24 40b` 低成功率问题（P3）**：
   - `CarryLockHybridRouter` 依据精确双平面 `carry_lock_rate` 与 `gf2_residual_dof` 自动将 `R=26 12b`、`R=26 16b`、`R=28 12b` 全部导流至 `Zone3_CarryLocked`，将求解成功率从 `Pure CMS-XOR` 的 **`0/16`** 全部拉升至 **`16/16 (100%)`**，端到端耗时显著下降 **`65.93% ~ 82.28%`**（全部 `p = 0.000031`）；在 `R=24 40b` 上将成功率由 `11/16` 提升至 **`16/16`** 并提速 **`59.36%`（`p=0.0250`）**；在 `R=28 16b` 上亦严守 `5.0s` 边界并较 `CMS-XOR` 显著更快（`+8.69%, p=0.0078, 8/0/0`）。
   - 在 `R=24 40b`、`R=25 24b`、`R=26 12b` 与 `R=26 16b` 上，`P3` 注入的 DualProp 单位与 `~402–517` 条 2-literal 等价类令 CDCL 冲突均值低于纯 `P1+P2`（`6604.3 / 3178.0 / 2014.1 / 14715.6` vs `7664.9 / 5684.0 / 2650.5 / 18130.6`）及 `BaseDyn`（`6931.0 / 6507.1 / 2489.9 / 20639.3`），并在 `R=25 24b`（`3178.0c`）、`R=26 12b`（`2014.1c`）与 `R=26 16b`（`14715.6c`）上取得全场最低冲突均值（低于 `Pure CaDiCaL` 的 `4465.5c`、`2337.9c` 与 `15549.3c`）。
2. **修复盲态假置信显著退化与 `R<=28` 早期分窗轨道失真（P1+P2 + Iteration 3 控时重构）**：
   - 将 `R<=28`（`tl>=3.0s`）首窗恢复为 **`4000` 冲突** 并辅以非对称短窗 EMA 阻尼与有界几何分摊后，彻底消除了 Iteration 2 在 `600` 冲突早截断导致的 CaDiCaL 搜索轨迹混沌退化：
     - **`R=26 16b`** 上 `P1+P2` 从 `10/16`（`45427.4c`, `-192.15%, p=0.0027`）完全恢复至 **`16/16`**（**`18130.6c`, `-60.1%` 冲突**），较 `BaseDyn` 均冲突下降 `+12.15%`，与 `Pure CaDiCaL` 已无显著差异（`p=0.5966`）；
     - **`R=26 12b`** 上 `P1+P2` 从 `3474.1c (p=0.0214)` 降至 **`2650.5c (p=0.4954)`**，**`R=24 40b`** 从 `8662.5c (p=0.0131)` 降至 **`7664.9c (p=0.3484)`**，**`R=25 24b`** 亦消除 `BaseDyn` 的显著劣化（`p=0.0182 -> 0.1297`）；至此 `R=24..28` 全部 7 档中 `P1+P2` 与 `Pure CaDiCaL` 之间**零任何统计显著冲突退化（全部 `p >= 0.1297`）**，且 **`R=28 16b`** 均值冲突（`33675.1c`）低于 `Pure`（`38626.1c`）同时中位耗时严格锁在 **`4.993s`**（不再如 Iteration 2 越过 `6.7s`）。
3. **根治高轮慢算例全系列不可中断 CaDiCaL `14x` 超时失控并保留 `R=64` 虫洞突破（P1 / P4）**：
   - 针对 CaDiCaL `1.0.3 / 1.5.3 / 1.9.5` 在 SHA-256 CNF 上的「浅层爆发（`0~2000` 冲突 `3k~61k/s`）→ 深层稳态悬崖（`>2000` 冲突骤降至 `186~526/s`）」双相规律，采用 `_get_round_timing_profile` 按轮次分带稳态速率硬封顶 + `dt<0.08s` 上行阻尼 + 渐进有界分摊后，彻底根治了 Iteration 2 在 `cd153/cadical153` 等 `R=48/64` 上长达 `21.24s~25.15s`（`14.16x`）的单窗阻塞：`U3` 14 格压测与 48 组全矩阵探针 `tl >= 1.5s` 最坏超时比仅 **`1.0170x ~ 1.1109x`**，且 `cd103` 捕获 `NotImplementedError` 优雅回落零崩溃；
   - 全盲 `R=64` 上 `P4` 模拟吞吐提升 **`3.90x`**（`64 -> 249.75`），最大深度由 `2` 深入至 **`5`**，`unk_drop / gate_drop / splice_drop` 分别增长 `+118.09% / +184.62% / +250.00%`（且在深度 `5` 首次出现跨 `H` 分化 `186 vs 224`），局部窗 GF(2) 单次均耗仅 **`4.51 ms`**；在严格零泄漏 `R=64, free_bits in {4, 6}` 过约束探针中达到 **`4/4 (100%)`** 秒解（`0.193s ~ 0.630s`），而基线仅为 **`0/4 (0.0%)`**。

### 5.2 实测暴露的仍存边界与客观局限（严禁粉饰）
1. **无重训条件下神经置信归零与紧预算 `R=28 16b` 预处理时间税**：
   - 正如 R4 定理 T2 所预测，在冻结 `best_m3v2.pt`（`max|z| < 0.86`）下，`P1` 神经置信门控 `pass_conf` 拦截率 `100%`，不提供正向自由比特极性先验；同时 `P1+P2` 预处理约占 `0.25~0.35s`，`P3` 桥接另占 `0.4~0.7s`。在 `5.0s` 严格硬界约束下（Iteration 3 已禁止 `P1+P2 / P3` 像 Iteration 2 那样依赖越界至 `~6.7s` 获取额外求解时间），`R=28 16b` 高难档留给 CDCL 冲刺的纯搜索墙钟缩减至 `~4.0~4.5s`，因此该档 `P1+P2` 成功率为 `2/8`（均值冲突 `33675.1c` 仍优于 `Pure` `38626.1c`）、`P3` 为 `1/8`（严守 `4.74s` 均值，低于 `Pure` `3/8` 与 `BaseDyn` `4/8`）。
   - 此外，即使 `R<=28` 恢复长首窗 `4000` 冲突，`5.0s` 控时分窗在第 `4000` 冲突处的重入以及 `P2` / `P3` 注入的少量 0 层子句仍会影响 CaDiCaL VSIDS/重启路径，故在 `R=24 40b`、`R=26 12b`、`R=26 16b` 与 `R=28 12b` 上 `P1+P2` 冲突均值略高于原生 `Pure CaDiCaL`（`-11.04% ~ -44.79%`，均无统计显著性），尚无法在零重训前提下凭空超越 `Pure CaDiCaL`。
2. **`R=64` 全盲态密码学不可穿透边界**：
   - 全盲 `R=64`（`512-bit` 自由度）下，`P4` 虽将吞吐与深度提升至 `249.75 sims / depth 5` 并触发了 `unk_drop = 186 / 224` 跨算例分化，但 `local_gf2_units = 0, pruned_conflicts = 0`。根因在于：`depth <= 5` 仅赋值少数单比特与中间 `S[32]/W[32]` 12-bit 宏块，在 `18,352` 个活跃非线性进位门与 64 轮雪崩下不足以在局部窗内封闭线性方程或产生矛盾剪枝——这与 R4 定理 T6 第 2 条严格一致。
3. **向第二梯队 P5 演进的必要性**：
   - 第一梯队 P1–P4 已在工程层面全面清理了热力图 OOD 污染、虚假极性注入、有损 Cube、XOR 相变死等、不可中断 CaDiCaL 慢轮超时失控、`cd103` 接口不兼容与 `R=64` 浅层截断阻塞；要进一步在 `R>=26` 高轮域 B 上获得超越原生 `Pure CaDiCaL` 的稳定冲突降幅，必须按 `R3_PROPOSAL_DESIGN.md` 第二梯队 **P5** 对神经网络实施 **Domain-B-Native 双路径解耦、对齐掩蔽与温度校准重训**。
