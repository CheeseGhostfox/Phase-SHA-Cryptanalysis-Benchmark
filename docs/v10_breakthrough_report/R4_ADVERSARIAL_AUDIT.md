# Phase-SHA V10「衔尾蛇-64」R4-2 独立对抗式交叉审证与定稿报告

> **文档定位**：本报告完整记录 Milestone 2 阶段针对 R1 卡点数学化归因、R3 突破方案推演（`R3_PROPOSAL_DESIGN.md`）、R4 严格定理总册（`R4_FORMAL_THEOREMS.md`）、符号/数值核验套件（`v10_ouroboros64/formal_checks/`）及 Lean 4.36.0 形式化证明工程（`formal_proofs_v10/`）开展的**独立对抗式交叉审证（R4-2）**全过程、4 组实测反例探针记录、法证完整性审计裁决，以及 Iteration 2 逐条修订闭环结果。

---

## 一、审证组织架构与独立性声明

### 1.1 三方并行独立审证分工
为避免「自我论证」偏差，M2 审证波次由三名完全隔离的独立子智能体并行执行：
1. **数学与工程对抗审证员（`teamwork_critic_m2_1`）**：逐行对照工作区源码（`cnf_cdcl.py`、`dynamic_cdcl.py`、`wormhole_mcts.py`、`crystal_core.hpp`、`heads.py`、`model.py`、`engine.py`、`trainer.py`），对 T1–T7 与 P1–P5 发起「假设有效性、反例构造、代码接口可达性、跨产物数值一致性」四维对抗攻击；并独立运行 FP32 盲态振幅与 `zero_defects` 全位点 GPU 探针。
2. **独立隔离复跑与边界探针工程师（`teamwork_challenger_m2_1`）**：在系统隔离临时目录中独立复跑 `v10_ouroboros64/formal_checks/` 全套脚本，独立编译并审计 `formal_proofs_v10/`（`lake build` + `VerifyAxioms.lean`），同时编写并运行 **Probe A/B/C/D 四组边界反例探针**，对 T2、T4、P3、P4 的极端边界进行实测证伪/验证。
3. **法证完整性审计员（`teamwork_auditor_m2_1`）**：在 Demo Mode 基线约束下执行四阶段法证核查：Lean 4.36.0 零作弊静态与内核公理审计、`formal_checks/` 反作弊/反硬编码审查与隔离复跑、报告可追溯性与密码学诚实边界审查，以及 `runs/a100_{m2,m3,m3v2}` 21 个基线存档 SHA-256 零篡改比对。

### 1.2 独立性与隔离性声明
- 三名审证员与原推导/实现工程师严格分离，独立运行全部验证命令；
- 所有临时探针脚本均运行于系统隔离临时目录并在汇报前彻底清理，未修改 `ouroboros_core/`、`ouroboros_net/`、`ouroboros_mcts/`、`ouroboros_train/`、`ouroboros_bench/`、`formal_proofs/`、`formal_proofs_v10/` 及 21 个基线存档；
- Iteration 1 审证结果：Critic 给出 `REQUEST_CHANGES`（共识别 **16 项意见**：8 项 Major、8 项 Minor，其中含 1 项阻塞 M3 落地的 P3 源码接口遗漏）；Challenger 给出 `CONFIRM / PASS`；Auditor 给出 `CLEAN`。Iteration 2 已针对 Critic 全 16 项逐一修订并重新锁盘 `formal_checks_summary.json`，实现 **16/16 闭环关闭**。

---

## 二、Critic 16 项逐条对抗攻防台账

### 2.1 总览表

| 编号 | 涉事模块 / 位置 | 质疑 / 反例核心 | 严重度 | Iteration 2 修订响应与关闭位置 | 状态 |
|---|---|---|---|---|---|
| **#1** | `T1` (`R4:44,48,55`) | `anchor_logits`（受 `flat_defect` 驱动）与 `flip_logits`（由 `flip_head(h)` 直接输出）机制混同；`flip_f1=0` 属权重分布偏移经验属性 | `Minor` | `R4 §T1.3(2) & §T1.4`：明确拆分为「代数恒等（Wolff 锚点归零）」与「分布偏移实测边界（584,448 位点 `max(flip_logits) <= -2.98`）」 | `CLOSED` |
| **#2** | `T1` (`R4:52–55`, `S2/B1`) | 混用 `S2`（`seed=20261124` 离散整数计数）与 `B1`（`seed+11` 实时前向）不同批次；未点明 `best_m3v2_cluster` $\alpha_1^* = 1.834395 > 1$ 的不可超越推论 | `Minor` | `R4 §T1.3(1) & §T1.4`：显式区分 `S2` 与 `B1` 批次，增补「当 $U-u_m \ge F$ 时 $\alpha_1^* \ge 1$，即使 100% 翻转准确率仍不可超越拷贝基线」推论 | `CLOSED` |
| **#3** | `T2` (`R4:65,83,86`) | 将确定性 CDCL（1-UIP + Phase Saving + VSIDS）混同于无学习独立随机重抽（$p_{\text{acc}}^{-k}$），且 `make_end_free` 相邻位实测相关性 `+0.035~+0.047` | `Major` | `R4 §T2.1(A2) & §T2.3(2)`：将 $p_{\text{acc}}^k / p_{\text{acc}}^{-k}$ 限定为「无冲突学习的独立 Guess-and-Determine 理想化对比模型」，与真实 CDCL 极性污染 $\Delta C_{\text{phase}}$ 及相依位 Fréchet–Hoeffding 界严格区分 | `CLOSED` |
| **#4** | `T2` (`R4:75–82,93`, `S3`) | 将批均值代入标量 `BCE` 忽视凸函数 Jensen 偏差；`BCE > ln 2` 充要条件需限定逐位或同质置信模型，低均值 BCE 亦可能掩盖噪声混合 | `Major` | `verify_symbolic_sympy.py` (`S3`) 重命名为 `r24_40b_plugin_bce` / `r25_24b_plugin_bce` 并附注 Jensen 偏差，增补 `p=0.52, q=0.70` 与混合分布反例；`R4 §T2` 严格区分 `S3` 插件近似与 `B2` 真实样本均值 BCE | `CLOSED` |
| **#5** | `T2` (`R4:88`) & `P1` (`R3:59`) | 把 `R25_24b` 均值 `0.165093` 误当最大值写出 `max|z| ≈ 0.69`（实测最大 `|z| = 0.8529~0.8565`, `max|p-0.5| = 0.2019`） | `Major` | `R3 §1.2(3)` 与 `R4 §T2.3(3)` 全部修正为实测 `max|z| ≈ 0.86`（`|p-0.5|_max ≈ 0.202`），给出安全下界 $T_{\text{cal}} > 2.66$（默认 `4.5` 具备约 `1.69x` 裕度） | `CLOSED` |
| **#6** | `T2` (`R4:80–89` vs `diag:185`) | `R=24 40b` 上 `no_any_set_phases=6489.0 > full=6250.25 > pure=4555.25`，证明分段重入独立引起退化，单归因 `set_phases` 失真 | `Major` | `R4 §T2.1(A2) & §T2.3 边界注记`：明确拆解总退化为极性污染 $\Delta C_{\text{phase}}$ 与 `solve_limited(3500->1.4x)` 分段重入混沌 $\Delta C_{\text{reentry}}$ 双分量，界定须由 P1 门控+长窗口协同消除 | `CLOSED` |
| **#7** | `T3` (`R4:102–116,125`) | 忽略失败 Cube 搜中学入普通 1-UIP 子句、VSIDS/Phase Saving 更新与热力图写入副作用，与 `R=26 16b case 2`（`full=37,718` vs `disable_cube=71,268`）冲突 | `Major` | `R4 §T3.1(A1), §T3.3(2) & 边界注记`：显式加入零副作用假设 $\mathbb{E}[\Delta C_{\text{side}}\mid\text{fail}]=0$，将结论 2 精确限定为「直接冲突预算净亏损 $-B_c$」，并据此解释 `case 2` 正向副作用 | `CLOSED` |
| **#8** | `T3` (`R4:102,129`) & `P2` (`R3:117`) | 源码 `dynamic_cdcl.py:408` 在 `used_confl>=7000` 触发时 `chunk>=6860`，故 `cube_budget = min(1500, max(500, chunk//2))` 恒为 `1500`，非 `[500, 1500]` 均匀变动 | `Minor` | `verify_symbolic_sympy.py` (`S4`)、`R3 §2.1/2.4` 与 `R4 §T3.1/3.4` 全部更正为默认触发恒取 **$B_c = 1500$**（仅当剩余总预算 `rem_b < 1500` 时截断至 `rem_b`） | `CLOSED` |
| **#9** | `T4` (`R4:171–173`) | 第 171 行同时使用未定义递归式的 $h(W_t)$ 与 $d(W_t)$；未说明 $\deg(c_b)=b+1$ 证明工具归属 | `Minor` | `R4 §T4.3(2) & §T4.4`：统一记号为 $d(W_t)$（与 Lean `CubeBudgetAndSplice.msgSchedDepth` 一致）并给出精确递归式；注明 $\deg(c_b)=b+1$ 由 SymPy `S1` + 数学归纳证明 | `CLOSED` |
| **#10** | `T4` (`R4:174,190`) | `NonlinearGuardBlocking.lean` 仅建模纯线性子系统 `sys : List LinEq`，未联合建模非线性门可行性约束 $\mathcal{N}$ | `Minor` | `R4 §T4.4`：明示 `NonlinearGuardBlocking.lean` 形式化纯线性子系统阻断性质，耦合非线性门双解见证（`c1_guard == (x0 & y0)`）由 Z3 `Z5` 补充 | `CLOSED` |
| **#11** | `T5` (`R4:233,310,317`) & `P4` (`R3:243`) | 报告保留早期计时（`531.403 ms / 1.750 ms / 303.59x / 7.815s` 及 `< 0.06%` 宣称），与工作区 `formal_checks_summary.json` 矛盾 | `Major` | 重新跑完 `run_formal_checks` 锁盘最新 `formal_checks_summary.json`（`2026-11-24T12:38:53Z`），`R3` 与 `R4` 全文同步更新为 **`901.539 ms / 2.141 ms / 421.12x / 10.126s`**，彻底删除过时数值与 `< 0.06%` 表述 | `CLOSED` |
| **#12** | `T5` (`R4:204,218–226`) & `P4` (`R3:276,317`) | 假设 A3 立方模型 $O(V^3)$ 与 `B5` 实际矩形欠定稀疏矩阵（$m_g/V_g=0.1818, m_w/V_w=0.1667$）不匹配 | `Major` | `R4 §T5.1(A3) & §T5.3(2)` 与 `R3 §4.5` 修正为矩形稀疏 RREF 标度 **$O(\mathrm{rank}\cdot m\cdot V)$**（`B5` 操作缩减比 $(5632/768)^2 \times (30976/4608) = 361.51\times$），说明仅在 $m/V=\text{const}$ 时退化为立方近似 | `CLOSED` |
| **#13** | `T7` (`R4:279–298`) | 核验 $\mathbb{E}[E_0] = 18,136$ 离散计数（`1536 + 16384 + 18352 = 36,272`）与期望线性性（无独立性假设漏洞） | `Minor` | Critic 核验确认完全精确，与 `B4` 实测 `18141.0625`（`0.0279%`）及 `smoke_analysis.json` (`18009.0`, $1.88\sigma$) 一致 | `CLOSED` |
| **#14** | `P3` (`R3:185` vs `cnf_cdcl.py:233,243`) | **阻塞级接口遗漏**：`SHA256CNFEncoder.build()` 将 `ch`/`mj` 声明为局部变量未保存为属性，隔离约束下 P3 无法直接读取 `Ch`/`Maj` 变量编号 | `Major` | `R3 §3.2(Step 0)` 显式规定 `p3_dual_bridge_router.py` 内建隔离子类/镜像编码器 **`BridgeableSHA256XORCNFEncoder`**，记录 `self.Ch_vars` 与 `self.Maj_vars` 并断言与原编码器 `1:1` 等价（零改 `cnf_cdcl.py`） | `CLOSED` |
| **#15** | `P1/P3` (`R3:53,183,187`) | 未显式拦截 `a_canon/e_canon/C_vars` 中的 Python `bool` 常量（`isinstance(False, int)` 为 `True` 致映射 `0/1` 崩溃）；未写出 Signed Union-Find 奇校验环冲突处理 | `Minor` | `R3 §1.2(1)` 与 `§3.2(Step 2–3)` 显式加入 `if isinstance(lit, bool):` 常量拦截规范，并补全 Signed Union-Find 遇奇校验环或与 `unit_clauses` 冲突时立即置 `early_unsat = True` 的规则 | `CLOSED` |
| **#16** | `T6` (`R4:254`) & `P3/P4` (`R3:190,241`) | 变量替换未规定原像逆恢复；`D_gf2 <= 2150/2450` 未注明校准范围；P4 闭区间 `w<=9` 与半开 `w=8` (`V_win=4608`) 口径未统一 | `Minor` | `R4 §T6.3(1)` 与 `R3 §3.2` 补充变量替换回代契约 $\tau'(y)=\tau(x)$；`R3 §3.2/3.3` 标明 `D_gf2` 阈值针对 `R in [22, 32], fb <= 40` 校准；`R3 §4.2` 与 `R4 §T5` 统一列出半开窗 `[28, 36)` (`w=8, V_win=4608`) 与对称闭窗 `[t_c-4, t_c+4]` (`w=9, V_win=5088`) | `CLOSED` |

---

## 三、Challenger 独立复跑与 4 组边界反例探针实测记录

`teamwork_challenger_m2_1` 在系统临时隔离目录独立运行了 `formal_checks`、Lean 4.36.0 编译审计以及 4 组对抗边界探针（执行完毕已彻底清理临时目录，工作区零污染）：

### 3.1 探针 A：T2 非对称/混合分布反例与 `T_cal = 4.5` 门控实测
1. **非对称过自信与临界阈值（$p = 0.52 > 0.5$）**：
   - SymPy/数值核验确认：当准确率略高于随机猜测（$p = 0.52$）而预测置信 $q = 0.70$ 时，$\mathrm{BCE}(0.52, 0.70) = 0.763378 > \ln 2 = 0.693147$（超额 `+0.070231`，精确等于 $D_{\mathrm{KL}}(p\|q) - D_{\mathrm{KL}}(p\|0.5)$）；
   - 解 $\mathrm{BCE}(0.52, q^*) = \ln 2$ 得临界过自信阈值为 **$q^* = 0.539957$**（即偏差 $\epsilon > 0.039957$ 即越界，已在 `verify_symbolic_sympy.py` `S3` 中固化）。
2. **混合置信分布（$p = 0.52$）**：
   - `50% q=0.70 + 50% q=0.30` $\implies \mathrm{BCE} = \mathbf{0.780324} > \ln 2$；
   - `70% q=0.52 + 30% q=0.90` $\implies \mathrm{BCE} = \mathbf{0.832651} > \ln 2$；
   - `80% q=0.52 + 20% q=0.75` $\implies \mathrm{BCE} = \mathbf{0.716881} > \ln 2$（其中 50/50 与 80/20 混合分布已写入 `S3` 自动化校验）。
3. **`best_m3v2.pt` 盲态实测与 `T_cal = 4.5` 封闭边界（`allow_tf32=False`）**：
   - `T_cal = 4.5, tau_phase = 0.08` 精确放行阈值为 $|z| \ge 2 T_{\text{cal}} \operatorname{atanh}(0.16) = \mathbf{1.452480}$（对应原始 $|q - 0.5| \ge 0.310380$）；
   - Challenger 在 `N=16` 上实测 `best_m3v2.pt`：`R=24 40b` `max|logit|=0.8122`、`R=25 24b` `max|logit|=0.8086`、`R=26 16b` `max|logit|=0.8529`，全部 **`gate_pass_rate = 0.000000`**；
   - Critic 同源 `seed+22, N=16` 探针进一步测得 `R=24 40b` `max|z|=0.8171 (adj_err_corr=+0.0349)`、`R=25 24b` `max|z|=0.8486 (adj_err_corr=+0.0468)`、`R=26 16b` `max|z|=0.8565 (max|p-0.5|=0.2019)`、`R=28 12b` `max|z|=0.7885`，同样 100% 被拦截（`frac(|z| >= 1.452480) = 0.0`），并据此锁定安全温度下界 $T_{\text{cal}} > 2.66$。
   - 同时探针验证若遭遇极端 OOD 置信 $q_{\text{raw}} = 0.85$（$|z| = 1.7346 \ge 1.4525$）且 $p=0.52$ 时将放行（`BCE=0.7039 > ln 2`），划定了 P1 在无重训下的理论 OOD 逃逸边界。

### 3.2 探针 B：T4 `W[16, 18, 20]` 零未知特例与 `W[26]` 双分支扩散相变
在真实 `SHA256XORCNFEncoder(rounds=26/27)` 与 `GeneralizedSymbolEnv.propagate(use_gf2=True, max_iters=16)`（`12b free`, `N=8`）上实测：
- **`W[16], W[18], W[20]` 精确已知**：在 `rounds=26` 与 `rounds=27` 上均精确为 **`32.0/32` 比特已知**、`active_msg_carries = 0.0/93`（`exact_match_16_18_20 = True`），直接证实 T4 依赖 DAG 断言；
- **`rounds=26`（顶层 5 阶字 `W[25]`）**：总未知位 `3514.25`、总活跃进位 `2192.0`（`t=22` `W_known=20.0/32`、`t=24` `2.2/32`、`t=25` `0.0/32`）；
- **`rounds=27`（`W[26]` 双 2 阶分支耦合）**：总未知位跃升至 `4130.0`、总活跃进位增至 `2512.25`（净增 **`+320.25` 门**），且 `S[27]` 替换 `S[26]` 作为终端边界后中段 `t=20..25` 失去反向锚定（`t=20` `active_ch_maj` 由 `47.4 -> 64.0`、`t=21` 由 `31.9 -> 48.1`、`t=24` 由 `16.6 -> 32.0`）。

### 3.3 探针 C：P4 `splice_block[:12]` vs `[:8]` 耦合锁定边界（Z3 对抗验证）
在 Z3 中构建第 32 轮低 4 位 `T1` 涟波进位压缩切片（`p1..p3 -> T1 -> e33, a33`）：
- **`[:8]` 截断（仅固定 `S[32,{0,4,3,7},0..1]`，`W[32,0..3]` 自由）**：即便固定上游 `S1, Ch, T2` 与 `S[32]` 高位，Z3 仍返回 **`sat`**（双解逃逸：`w32=9, t1=11, e33=0, a33=7` vs `w32=3, t1=5, e33=10, a33=1`）；
- **`[:12]` 完整块（同时固定 `W[32,0..3]`）+ 上游局部约束**：`T1[0..3]/e33[0..3]/a33[0..3]` 逃逸检查返回 **`unsat`**（彻底消除 `Z4` 自由度逃逸）；
- **对照组（撤除上游 `S1/Ch` 固定约束仅固定 `[:12]`）**：返回 **`sat`**，精确刻画了「`W[32]` 耦合锁定 `T1` 须以上游条件部分约束为前提」的工程边界。

### 3.4 探针 D：P3 Signed Union-Find 等价链与 `BitVec 32` 溢出边界
- **Signed Union-Find vs Z3 精确语义**：在偶取反一致环、奇取反矛盾环、自反矛盾 `x0^x0=1`、16 变量交错快捷矛盾及 **60 组随机等价图**（`n_vars=12`, `6..20` 条边，`14` 组 `sat_equiv` + `46` 组 `unsat`）上与 Z3 交叉检验，**`mismatches = 0 / 60`**；
- **`BitVec 32` 溢出边界复验**：显式验证 `0xFFFFFFFF+1`、`0x80000000+0x80000000`、`0xFFFFFFFF+0xFFFFFFFF` 等边界样例，且 Z3 全称否定 `(c != (maj(x,y,c)<<1)) || (c32 != ULT(x+y, x))` 返回 **`unsat`**。

---

## 四、Forensic Auditor 法证核验与 Iteration 2 终次锁盘汇总

### 4.1 法证审计裁决（`teamwork_auditor_m2_1`）
- **基线完整性模式**：Demo Mode（依据 `ORIGINAL_REQUEST.md`）。
- **最终法证裁决**：**`CLEAN`**（4 阶段全部 `PASS`）。
  - **Phase 1（静态零作弊与反预置扫描）**：`formal_proofs_v10/` 禁用词 `\bsorry\b|\bAdmitted\b|\baxiom\b|\bMathlib\b|\bLean\.ofReduceBool\b` 扫描 **0 命中**，`lake-manifest.json` `"packages": []`；`formal_checks/` 全为真实 `sympy` / `z3` / `torch` (`allow_tf32=False`) / C++ `core` 实时计算，无硬编码/空壳/预置结果。
  - **Phase 2（Lean 4.36.0 与 `formal_checks` 独立核验）**：
    - `g:\SHA-LM\formal_proofs_v10` 执行 `lake build` 返回 `Build completed successfully (16 jobs).`（退出码 `0`）；
    - 执行 `lake env lean VerifyAxioms.lean`（退出码 `0`），全部 **45 条核心定理**仅依赖 Lean 4 标准内核公理 `[propext, Classical.choice, Quot.sound]`（以及 4 条 32 位位向量定理附带的 `Std.Tactic.BVDecide` 原生反射项 `._native.bv_decide.ax_1_5`），**零 `sorryAx`、零自定义 `axiom`**。
  - **Phase 3（文档可追溯性与密码学诚实边界）**：全部源码路径、行号、定理标识符与日志数据均可直接追溯至工作区实测文件或 M1 审定记录；明确划定 `R=64` 密码学不可穿透边界与第一/第二梯队界限。
  - **Phase 4（只读隔离与 21 文件 SHA-256 核验）**：`runs/a100_{m2,m3,m3v2}` 全部 21 个基线存档与检查点文件 SHA-256 100% 零变动；既有核心源码（`ouroboros_core/` 等）与 `formal_proofs/` 零修改。

### 4.2 Iteration 2 终次锁盘核验结果（`formal_checks_summary.json`）
针对 Critic #4、#8、#11 修订了 `v10_ouroboros64/formal_checks/verify_symbolic_sympy.py` 后，于工作区执行全套自动化核验完成终次锁盘：
```powershell
& "g:\SHA-LM\.venv\Scripts\python.exe" -u -m v10_ouroboros64.formal_checks.run_formal_checks --out_json v10_ouroboros64\formal_checks\formal_checks_summary.json
```
- **锁盘时间戳（UTC）**：`2026-11-24T12:38:53Z`
- **退出码与总耗时**：`all_exit_zero = true`，`total_elapsed_sec = 10.126s`（`sympy`: `0`, `1.154s`; `z3`: `0`, `0.315s`; `torch_bounds`: `0`, `7.719s`）
- **FP32 锁定状态**：`allow_tf32_matmul = false`、`allow_tf32_cudnn = false`、`fp32_locked = true`
- **基线存档完整性**：`archive_integrity.files_checked = 21`、`all_unchanged = true`（`0` 改动、`0` 多余文件）
- **核心锁盘数值对齐校验**：
  - `S3`：`r24_40b_plugin_bce = 0.71397`、`r25_24b_plugin_bce = 0.706331`（附 `plugin_note`）、`asym_counterexample_p052_q070_bce = 0.763378`、`asym_p052_critical_q = 0.539957`、`mixture_p052_50_50_bce = 0.780324`、`mixture_p052_80_20_bce = 0.716881`；
  - `S4`：`r26_16b_default_bc = 1500`、`r26_16b_default_bc_loss = -1500.0`、`r26_16b_loss_range = [-1500.0, -500.0]`；
  - `B5`：全局 `R=64`（`30,976` 变量, `5,632` 方程, `rank = 5632`）耗时 **`901.539 ms`** vs 局部 `w=8` `[28, 36)`（`4,608` 变量, `768` 方程, `rank = 768`）耗时 **`2.141 ms`**，变量缩减 `6.7222x`，实测加速比 **`421.12x`**。

---

## 五、M2 终审放行结论与流转 M3 强制约束

1. **审证终裁**：Iteration 1 Critic 提出的 **16 项质疑（8 Major + 8 Minor）已 100% 闭环关闭**；`R3_PROPOSAL_DESIGN.md`、`R4_FORMAL_THEOREMS.md`、`R4_ADVERSARIAL_AUDIT.md`、`v10_ouroboros64/formal_checks/` 与 `formal_proofs_v10/` 在数值、逻辑、源码接口与 Lean 4.36.0 形式化映射上完全一致，**满足流转至 M3 原型实现与同条件配对实测对比的全部条件**。
2. **M3 必须遵守的硬性约束**：
   - **P3 编码器与 `bool` 拦截**：`p3_dual_bridge_router.py` 必须使用隔离 `BridgeableSHA256XORCNFEncoder`（零修改 `cnf_cdcl.py`），在所有文字映射前优先 `if isinstance(lit, bool):` 拦截常数，并在 Signed Union-Find 遇奇校验环/常数冲突时置 `early_unsat = True`；
   - **P1/P2/P4 安全边界**：强制 `allow_tf32=False`、`conflict_heatmap=None, zero_defects=True`、开局单次门控 `T_cal=4.5`（不再搜中覆写 `set_phases`）、长窗口连续预算消除 $\Delta C_{\text{reentry}}$、$B_c = 0$ 零冲突探测、局部 GF(2) 窗口 $w \le 9$，且所有 `SAT` 原像 100% 通过 `OuroborosCore.verify_preimage` 验真。
