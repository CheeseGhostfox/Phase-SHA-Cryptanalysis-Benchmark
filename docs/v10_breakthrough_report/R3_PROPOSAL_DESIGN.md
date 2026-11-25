# Phase-SHA V10「衔尾蛇-64」可行技术突破方案设计文档（R3 方案推演）

> **文档定位**：本方案直接承接 M1 源码/实测诊断报告（`docs/v10_breakthrough_report/R1_DIAGNOSIS_REPORT.md`）与 `v10_ouroboros64/diagnostics/diagnostics_summary.json` 的定量归因结果，提出针对五大核心卡点的技术突破方案 **P1–P5**。为严格保障与既有一线实验产物（`a100_m2`、`a100_m3`、`a100_m3v2`）的公平隔离对比，本报告明确划分为两大实施梯队：
> - **第一梯队（P1–P4：零重训即插即用原型突破方案）**：无需修改既有核心源码（`ouroboros_core/`、`ouroboros_net/`、`ouroboros_mcts/`、`ouroboros_train/`、`ouroboros_bench/`），无需重新训练主干权重，直接以隔离模块形式落地于 `v10_ouroboros64/prototypes/`，并与既有检查点 `best_m3v2.pt` / `ouroboros_a100_step4000.pt` 在纯 FP32（`allow_tf32=False`）、同随机流、同 `sat_time_limit` 与 100% `verify_preimage` 门控下开展 M3 配对实测对比；
> - **第二梯队（P5：需重训范式升级路线图）**：针对域 A/B 结构性输入割裂、Stage B 掩蔽分布错位与 `PairAcc` 恒等拷贝病灶的长期架构与训练目标重构规范，不参与无重训 M3 原型对比，但为后续迭代提供根本工程出路。

---

## 目录
- [总览：卡点与突破方案映射矩阵](#总览卡点与突破方案映射矩阵)
- [第一梯队：零重训原型突破方案（P1–P4）](#第一梯队零重训原型突破方案p1p4)
  - [方案 P1：动态 CDCL 净态修复、置信校准门控与长窗口连续预算（`CleanDynamicNeuralCDCL`）](#方案-p1动态-cdcl-净态修复置信校准门控与长窗口连续预算cleandynamicneuralcdcl)
  - [方案 P2：零虚耗边界探测与状态同步 XOR-Cube（`ZeroWasteProbeAndSyncXOR`）](#方案-p2零虚耗边界探测与状态同步-xor-cubezerowasteprobeandsyncxor)
  - [方案 P3：双平面预处理等价桥接与 Carry-Lock 感知混合路由（`DualPlaneBridgeRouter`）](#方案-p3双平面预处理等价桥接与-carry-lock-感知混合路由dualplanebridgerouter)
  - [方案 P4：`R=64` 虫洞 MCTS 局部窗口 GF(2) + `W[32]` 耦合修复 + 轻量化评估（`LocalWindowWormholeMCTS`）](#方案-p4r64-虫洞-mcts-局部窗口-gf2--w32-耦合修复--轻量化评估localwindowwormholemcts)
- [第二梯队：长期训练范式重构路线图（P5，需重训）](#第二梯队长期训练范式重构路线图p5需重训)
  - [方案 P5：Domain-B-Native 双路径解耦、对齐掩蔽与去拷贝残差目标](#方案-p5domain-b-native-双路径解耦对齐掩蔽与去拷贝残差目标)
- [M3 原型接口契约、隔离边界与验证准则](#m3-原型接口契约隔离边界与验证准则)

---

## 总览：卡点与突破方案映射矩阵

| 方案编号 | 所属梯队 | 针对 M1 核心卡点 | 落地路径（隔离） | 关键依赖与隔离约束 |
|---|---|---|---|---|
| **P1** | 第一梯队（无重训） | Dyn CDCL 丢失内部 `S/C/W[16:]`、OOD `conflict_heatmap`、`4.2x~8.4x` 假置信相位盲注、`solve_limited` 分段重入混沌 | `v10_ouroboros64/prototypes/p1_clean_cdcl.py` | 只读加载 `best_m3v2.pt` / `m2_step4000.pt`，禁用 TF32 |
| **P2** | 第一梯队（无重训） | `cube_depth=2` 默认每次触发虚耗 `1500` 冲突预算（仅尾段受 `rem_b` 截断）；`DynamicNeuralXORSolver` 循环恒传静止 `base_assigned` | `v10_ouroboros64/prototypes/p2_adaptive_probe_xor.py` | 零冲突 `Solver.propagate()` 探测；伴随 `shadow_bcp` 同步状态 |
| **P3** | 第一梯队（无重训） | 原生编码器未消费 DualProp 单位与 2-literal 等价类（且未导出 `Ch/Maj` 变量属性）；`R>=26` XOR 归零僵局；Windows `pycryptosat` 崩溃风险 | `v10_ouroboros64/prototypes/p3_dual_bridge_router.py` | 复用 `GeneralizedSymbolEnv` + `BridgeableSHA256XORCNFEncoder` + Python Bitset RREF + 子进程隔离 |
| **P4** | 第一梯队（无重训） | `R=64` 禁用 GF(2)、`splice_block[:8]` 截断丢弃 `W[32]`、双次前向、`94/26/8/0` 浅层死循环 | `v10_ouroboros64/prototypes/p4_local_wormhole_mcts.py` | 调用既有 C++ `core.solve_gf2_equations` 跑局部窗，诚实界定密码学边界 |
| **P5** | 第二梯队（需重训） | 域 A/B 104 维输入割裂、`PairAcc` 恒等拷贝支配（`0.9973 < 0.9983`）、Stage B 随机掩蔽错位、`conflict_heatmap` 零训练暴露 | 未来训练分支（规范文档，不改动 M1/M2/M3 存档） | 需完整 GPU 重训，不参与 M3 无重训基线配对对比 |

---

## 第一梯队：零重训原型突破方案（P1–P4）

### 方案 P1：动态 CDCL 净态修复、置信校准门控与长窗口连续预算（`CleanDynamicNeuralCDCL`）

#### 1.1 前提假设
1. **模型只读且不重训**：原型直接加载 `best_m3v2.pt` 或 `ouroboros_a100_step4000.pt`（强制 `enforce_fp32_determinism()`，`allow_tf32=False`）。
2. **OOD 热力图通道污染**：全仓训练器（`trainer.py`、`m3_trainer.py`、`m3_v2_trainer.py`）从未向通道 `[73..91]`（`conflict_heatmap`）传入非零张量（`diagnostics_summary.json` 验证该通道在训练样本中 100% 恒为 `0.0`），推理期通过 `_stamp_heat` 向 `[73]` 写入 `log1p(heat)` 属于严格分布外（OOD）激活。
3. **盲态置信膨胀与负优化因果**：M1 `diag_domain_gap.py` 与 `formal_checks_summary.json` 证实，在盲态 `R=24 40b` 与 `R=25 24b` 上，`best_m3v2.pt` 的相位准确率仅 `50.94%~53.13%`（$\text{BCE} \in [0.7065, 0.7548] > \ln 2$），但置信均值相对 `m2_step4000.pt` 膨胀 `4.2x~8.4x`，致使 `frac(|logit| >= 0.08) = 1.0`。向 CaDiCaL 注入此类虚假强极性会破坏原生 VSIDS/Phase Saving（如 `R=25 24b` 关闭 `set_phases` 后冲突数骤降 `-44.12%`, `p=0.0156`）。
4. **分段重入混沌效应（M1 Caveat 1）**：R1 报告 `R=24 40b` 消融显示，即便完全撤除 `set_phases`（`no_any_set_phases=6489.00`），按 `initial_chunk=3500 -> 4900 -> 6860` 频繁中断 `solver.solve_limited()` 仍显著劣于连续执行的 `pure_cadical=4555.25`（M1 `pure_3500_1.4x` 在 `R=26 16b` 亦飙升至 `40,935.83`）。因此必须避免低阈值分段重入。

#### 1.2 作用机制与接口设计

**源码修复锚点**：
- `v10_ouroboros64/ouroboros_mcts/dynamic_cdcl.py:66–86`（`_build_var_map_and_known_mask` 仅映射 `W[0..15]`）
- `dynamic_cdcl.py:134–189`（`_neural_score_partial` 将 `S[1..R-1]`、`C`、`W[16..R-1]` 全置零并标记 `unk=1.0`，且注入 OOD `conflict_heatmap`）
- `dynamic_cdcl.py:215, 371–387, 445–449`（盲注 `set_phases` 与 `3500 -> 1.4x -> 8000` 频繁重入）

**核心机制**：
1. **全晶体变量反向映射 `_build_full_crystal_var_map(enc: SHA256CNFEncoder)`**：
   - 将 `enc.W_vars[0..R-1][0..31]`、`enc.a_canon[1..R-1][0..31]`（及其向后移位别名 $S[t+w_s, w_s]$，$w_s \in \{0,1,2,3\}$）、`enc.e_canon[1..R-1][0..31]`（及移位别名 $S[t+w_s, 4+w_s]$）与 `enc.C_vars[0..R-1][0..9][1..31]` 全部建立双向 DIMACS 变量映射；
   - **`bool` 常量文字拦截规范**：`a_canon[-3..0, R-3..R]`、`e_canon[-3..0, R-3..R]` 及 `C_vars[*][*][0]` 中包含大量 Python `bool` 常量（`False`/`True`）。因 Python 中 `bool` 是 `int` 子类（`isinstance(False, int) is True` 且 `int(False)==0, int(True)==1`），所有变量映射扫描**必须先执行 `if isinstance(lit, bool):` 拦截**，将其直接作为已知常数位写入 `W_b/S_b/C_b`（`unk=0.0`），严禁将 `False/True` 误当 DIMACS 变量 `0` 或 `1`（`W_vars[0][0]`）；
   - 在开局 `solver.propagate()` 后，将全部已推导出的 0 层真值文字同步填入 `W_b, S_b, C_b` 并把对应 `unk` 置 `0.0`，彻底消除内部时空晶体的虚假盲区。
2. **清除 OOD 热力图通道**：推理调用固定传 `conflict_heatmap=None, zero_defects=True`，保证通道 `[46..91]` 全为 `0.0`。
3. **置信温度校准与不对称门控（Calibrated Phase Gate）**：
   $$\hat{p}_{\text{cal}}(v) = \sigma\!\left(\frac{\text{logit}(v)}{T_{\text{cal}}}\right), \quad \text{inject}(v) \iff \big|\hat{p}_{\text{cal}}(v) - 0.5\big| \ge \tau_{\text{phase}} \;\lor\; \big(|d^+(v) - d^-(v)| \ge \delta_{\text{asym}}\big)$$
   其中 $d^+(v), d^-(v)$ 为 0 冲突 1-bit FLP 正反传播推导增长量。默认参数 $T_{\text{cal}} = 4.5,\; \tau_{\text{phase}} = 0.08,\; \delta_{\text{asym}} = 16$（精确门限 $|z| \ge 2 T_{\text{cal}} \operatorname{atanh}(0.16) = 1.452480$）：
   - 对未重训模型 `best_m3v2.pt`，FP32 盲态独立探针实测平均 $|\text{logit}| \approx 0.57\sim 0.69$、最大实测 $|z| \approx 0.86$（Critic/Challenger 实测 `0.8529~0.8565`，对应 $|p - 0.5|_{\max} \approx 0.202$）；由此确定安全温度下界 $T_{\text{cal}} > 0.86 / (2 \operatorname{atanh}(0.16)) \approx 2.66$，默认 $T_{\text{cal}} = 4.5$ 具备约 **`1.69x`** 安全裕度，缩放后 $|\hat{p}_{\text{cal}}-0.5|_{\max} \le 0.048 < 0.08$（`gate_pass_rate = 0.0`），**确定性拦截全部虚假极性注入**；
   - 仅当某比特本身存在显著神经先验（$|z| \ge 1.452480$）或前瞻单步传播呈现强结构不对称时，才于开局 Episode 0 注入一次；**严禁在搜中任何阶段再次调用 `set_phases` 覆写 CaDiCaL 原生 Phase Saving**。
4. **单次预处理 + 长窗口连续预算**：
   - Episode 0 完成：注入 P3 双平面桥接子句 $\rightarrow$ 0 层 `propagate()` $\rightarrow$ 全晶体净态神经打分 $\rightarrow$ Top-$K_1$（$\le 8$）零冲突 FLP + 校准极性门控；
   - 随后采用 `long_chunk_conflicts = 120000`（配合 `threading.Timer(rem_sec, solver.interrupt)` + `solve_limited(expect_interrupt=True)` 硬壁钟保护）连续交付剩余冲突预算，彻底消除低阈值重入导致的重启调度破碎与轨迹混沌。

```python
# prototypes/p1_clean_cdcl.py 核心接口与伪代码
class CleanDynamicNeuralCDCLSolver:
    def __init__(
        self,
        model: Optional[Ouroboros64Net] = None,
        core: Optional[OuroborosCore] = None,
        device: Optional[torch.device | str] = None,
        solver_name: str = "cadical195",
        phase_cal_temp: float = 4.5,
        phase_conf_thresh: float = 0.08,
        flp_asym_thresh: int = 16,
        enable_flp: bool = True,
        flp_top_k: int = 8,
        long_chunk_conflicts: int = 120000,
    ) -> None: ...

    def solve(
        self,
        H_target: Sequence[int],
        rounds: int = 24,
        fixed_words: Optional[Dict[int, int]] = None,
        fixed_bits: Optional[Dict[Tuple[int, int], int]] = None,
        extra_clauses: Optional[Sequence[Sequence[int]]] = None,
        conflict_budget: int = 250000,
        time_limit_sec: float = 6.0,
    ) -> DynamicSolveResult:
        # 1. 构建 enc = SHA256CNFEncoder(rounds).build(...) + 注入 extra_clauses
        # 2. ok, prop_lits = solver.propagate() -> 填充完整 W/S/C 晶体 & unk 掩码
        # 3. 净态前向: model(..., conflict_heatmap=None, zero_defects=True)
        # 4. Top-K 1-bit FLP (0 冲突) + Calibrated Phase Gate (仅开局一次，若无过门限变量则不调用 set_phases)
        # 5. 长窗口连续预算 solver.conf_budget(min(long_chunk_conflicts, rem)) + Timer 硬中断
        # 6. SAT 时强制调用 core.verify_preimage(w0_15, ht_np, rounds) 验真
```

#### 1.3 适用边界
- 适用于全部 CaDiCaL 可解域 B 盲求逆档位（`R=24..28, free_bits=12..40`）以及 P3 Zone 2/3 回落求解；
- 当加载未经重训的 `best_m3v2.pt` 时自动充当「安全门卫」（拦截假极性，仅保留净态瓶颈排序与无损等价/FLP 剪枝）；当未来接入经 P5 校准重训的权重时，门控自动平滑放开真置信极性注入。

#### 1.4 预期收益
- 彻底消除 `R=25 24b`（`-45.72%`）、`R=24 40b`（`-30.92%`）与 `R=26 16b`（`-32.73%`）相对 `Pure CaDiCaL` 的冲突劣化，预期在 `R=25 24b` 上将平均冲突由 `8266.12` 降至 `<= 4600~5800`；
- 消除搜中反复 GPU 前向开销，大幅降低单样例平均求解壁钟耗时（预期减少 `40%~60%` 非求解器耗时）。

#### 1.5 潜在失效风险与缓解
- **风险 1（长窗口超时越界）**：`long_chunk_conflicts = 120000` 在高难度实例上可能超过 `time_limit_sec`；**缓解**：必须使用独立 `threading.Timer` 调用 `solver.interrupt()` 并通过 `try/finally: timer.cancel()` 确保精确定时退出。
- **风险 2（未重训模型无神经加速上限）**：门控拦截虚假极性后，未重训模型自身不能凭空提供正确的自由比特极性，加速主要来自消除负优化及配合 P2/P3 零冲突推导与等价替换。

---

### 方案 P2：零虚耗边界探测与状态同步 XOR-Cube（`ZeroWasteProbeAndSyncXOR`）

#### 2.1 前提假设
1. **有预算 Cube 探测在零剪枝区间存在确定直接预算亏损**（对应 R4 定理 T3）：`dynamic_cdcl.py:390–436`（第 `408` 行 `cube_budget = min(1500, max(500, chunk // 2))`）在 `used_confl >= 7000` 后才触发（此时 `initial_chunk = 3500` 已随 `1.4x` 倍增至 `chunk >= 6860`，`chunk // 2 >= 3430 > 1500`），因此原生触发时单次预算**恒等于 $B_c = 1500$**（仅当剩余总预算 `rem_b < 1500` 时受 `min(cube_budget, rem_b)` 截断至 `rem_b`）。在 `R=26 16b` 上触发的 `5` 次探测 `100%` 未剪枝（`unsolved_wasted=5, cores=0`），直接预算净亏损 `-1500`（`case 1/3` 总冲突净增 `+4,446 / +10,094`）。
2. **`propagate(assumptions=...)` 零冲突不变量**：PySAT `Solver.propagate(assumptions=...)` 仅运行 0 层布尔约束传播（BCP），不产生任何决策与冲突（$B_c = 0$），但其导出的单子句、2-bit 阻断子句及公共蕴涵交集具有永久全局有效性。
3. **`DynamicNeuralXORSolver` 静止状态断流**：`dynamic_cdcl.py:598, 602` 在 `while` 循环内恒定传入初始化 `base_assigned`，且 `pycryptosat.Solver` C API 未导出内部学习到的单位文字列表。

#### 2.2 作用机制与接口设计

**源码修复锚点**：
- `v10_ouroboros64/ouroboros_mcts/dynamic_cdcl.py:308–361, 390–436`（FLP 与有预算 `cube_depth=2`）
- `dynamic_cdcl.py:554–679`（`DynamicNeuralXORSolver` 静止 `base_assigned`）

**核心机制**：
1. **零虚耗边界探测器 `ZeroWasteBoundaryProber`**：
   - 默认**彻底关闭**消耗冲突预算的 `solve_limited(assumptions=cube)`（即 $B_c = 0$）；
   - 按结构边界邻接度（优先靠近已知位点的未知 `W/S` 变量）选取候选集 `cands[:12]`，对每个变量 $x$ 执行 0 冲突正反 `solver.propagate(assumptions=[±x])`：
     - 若单向冲突 $\Rightarrow$ 立即写入永久单位子句 `[-x]` 或 `[x]`；
     - **公共蕴涵交集提取（Stålmarck/BCP-Intersection）**：若 `ok_p` 与 `ok_n` 均成立，计算 $(L_p \cap L_n) \setminus L_{\text{base}}$。任何同时被 $x$ 和 $\neg x$ 蕴涵的文字 $y$ 满足 $(x \to y) \land (\neg x \to y) \models y$，立即免费写入永久单位子句 `[y]`；
     - **耦合邻近 2-bit 零冲突探测**：仅在同轮或消息扩展滞后窗口 $\{-2, -7, -15, -16\}$ 内选取 Top-4 高耦合变量对执行 `solver.propagate(assumptions=[s1, s2])`（$B_c = 0$），一旦失败立即写入精确二元阻断子句 `[-s1, -s2]`。
2. **双引擎状态同步 `SyncedDynamicXORSolver`**：
   - 维护一个**伴随 `PySAT CaDiCaL` 影子传播器 `shadow_bcp`**（绝不调用 `shadow_bcp.solve()`，仅用于 0 层 `propagate()` 状态追踪）；
   - 初始化时将 P3 产出的全量单位文字与 2-literal 等价子句同时注入 `cms`（CryptoMiniSat）与 `shadow_bcp`；
   - 每次搜索循环调用 `shadow_bcp.propagate()` 同步最新已推导的 `W/S/C` 全晶体未知掩码，跳过已确定变量后再以 `conflict_heatmap=None` 净态调用神经排序；
   - 任何来自 `shadow_bcp` 零冲突探测或 `cms.get_conflict()` 的新单位/二元子句均**双向同步**至 `cms` 与 `shadow_bcp`。

```python
# prototypes/p2_adaptive_probe_xor.py 核心接口
class ZeroWasteBoundaryProber:
    @staticmethod
    def probe(solver: Solver, cands: Sequence[int], n_max_1bit: int = 12,
              coupled_pairs: Optional[Sequence[Tuple[int, int]]] = None
              ) -> Dict[str, Any]:
        # 返回 {"unsat": bool, "units": List[int], "binary_clauses": List[List[int]], "deduced_vars": int}

class SyncedDynamicXORSolver:
    def solve(self, H_target, rounds=24, fixed_words=None, fixed_bits=None,
              extra_clauses=None, xor_clauses=None, time_limit_sec=6.0) -> DynamicSolveResult: ...
```

#### 2.3 适用边界
- `ZeroWasteBoundaryProber` 适用于所有 PySAT `CaDiCaL` 求解前/搜中边界强化；严格保证对未剪枝实例零冲突损耗；
- `SyncedDynamicXORSolver` 仅适用于 P3 路由器判定为 XOR 活跃的 Zone 1（`R<=25` 低进位阻塞区），禁止直接用于 `R>=26` 高阻塞区。

#### 2.4 预期收益
- 在 `R=26 16b` 等难例上彻底消除每次 `1500`（总计 `1500~3000+`）次 Cube 冲突预算虚耗（按定理 T3 保证直接冲突期望净收益 $\mathbb{E}[\Delta C_{\text{net}}] \ge 0$），并通过公共蕴涵交集在部分档位零成本锁定额外自由位；
- 消除 `DynamicNeuralXORSolver` 反复将已推导比特送入排序与冗余 Cube 的空转浪费。

#### 2.5 潜在失效风险与缓解
- **风险**：`shadow_bcp` 仅执行纯 CNF BCP，无法直接获知 `cms` 内部高斯消元在搜索途中动态发现的新增单位文字；**缓解**：依赖 P3 在开局通过完整 DualProp + Bitset RREF 先行抽干 0 层全局线性单位与 2-literal 等价类并同时喂给两端，最大程度压缩两者信息差。

---

### 方案 P3：双平面预处理等价桥接与 Carry-Lock 感知混合路由（`DualPlaneBridgeRouter`）

#### 3.1 前提假设
1. **原生编码器信息丢失**：`SHA256CNFEncoder` 与 `SHA256XORCNFEncoder`（`cnf_cdcl.py:60–257, 367–425`）仅对词级固定位做常量折叠，既未消费 `GeneralizedSymbolEnv` 双平面闭包（ARX + GF(2)）推导的中间单位文字，也未提取 GF(2) RREF 枢轴行中重量为 2 的 **2-literal 等价类**（$x_i \oplus x_j = c$）。
2. **相变双峰路由事实**：M1 `diag_xor_cliff.py` 定量证实：`R<=24 12b..32b` 及 `R=25 12b` 在双平面闭合后活跃进位阻塞率 $\le 27.5\%$、GF(2) 残余自由度 $\le 2045$，`CMS-XOR` 在 `0.012s~0.135s` 内 `100%` 秒解；而从 **`R=26 12b`** 起，未知扩散跨越满 11 轮压缩与 5 阶消息调度链，活跃进位阻塞率跃升至 **`33.31%`**（残余自由度 `2590.25`），`CMS-XOR` 降至 **`0/16`**，而纯 `CaDiCaL` 仅需 `2875` 冲突 / `0.19s` 全解。
3. **Windows `pycryptosat` 崩溃风险（M1 Caveat 2）**：M1 验证波观测到 Windows 下 `pycryptosat` 在高频触及 `time_limit` 边界时存在极低概率底层 `0xC0000005`（Access Violation）段错误风险，单进程直接调用存在拖垮整个评测基准进程的风险。

#### 3.2 作用机制与接口设计

**源码修复锚点**：
- `v10_ouroboros64/ouroboros_core/cnf_cdcl.py:233, 243, 367–425`（`SHA256CNFEncoder.build()` 中 `ch`/`mj` 仅为局部变量未保存为实例属性；`SHA256XORCNFEncoder` 未消费 DualProp 与等价类）
- `v10_ouroboros64/ouroboros_core/cpp/crystal_core.hpp:750–843, 1145–1190`（`GF2RREF::extract_unit_deductions` 仅抽取 `popcnt==1`）

**核心机制（隔离于 `v10_ouroboros64/prototypes/p3_dual_bridge_router.py`）**：
1. **双平面预处理与 2-Literal 等价桥接 `DualPlaneEquivalenceBridge.extract(...)`**：
   - **Step 0（隔离可桥接编码器 `BridgeableSHA256XORCNFEncoder`）**：因 `SHA256CNFEncoder.build()`（`cnf_cdcl.py:233, 243`）未保存 `Ch`/`Maj` 变量映射且严禁修改 `cnf_cdcl.py`，`p3_dual_bridge_router.py` 内建隔离子类或镜像编码器 **`BridgeableSHA256XORCNFEncoder`**，在构建过程中显式记录 `self.Ch_vars[t][0..31]` 与 `self.Maj_vars[t][0..31]`，并在单元测试中严格断言其总变量数 `num_vars`、子句列表 `clauses` 与 `xor_clauses` 和原 `SHA256XORCNFEncoder` **1:1 完全等价**（零修改 `cnf_cdcl.py`）。
   - **Step 1（C++ 双平面闭合）**：调用 `GeneralizedSymbolEnv.propagate(use_gf2=True, max_iters=16)`；若 `last_conflict == True`，直接返回 `early_unsat=True`。
   - **Step 2（全晶体单位文字导出与 `bool` 常量拦截）**：将确定化的 `W_sym / S_sym / C_sym`（含 `a_canon` 与 `e_canon`）映射至 `BridgeableSHA256XORCNFEncoder` 对应 DIMACS literal，形成 `unit_clauses`。**强制 `bool` 拦截规范**：对任意取出文字必须先判断 **`if isinstance(lit, bool):`**（防止 `isinstance(False, int)` 将 `False/True` 误当 DIMACS 变量 `0/1`）；若遇已知常量与期望真值冲突（如 `lit is False` 却要求取 `1`），立即触发 **`early_unsat = True`**。
   - **Step 3（Python 大整数 Bitset RREF 抽取 2-Literal 等价类）**：
     - 收集 `BridgeableSHA256XORCNFEncoder` 的全部 `xor_clauses` 及 DualProp 已确定条件触发的 `Ch`/`Maj` 条件线性化方程（通过 `self.Ch_vars` / `self.Maj_vars` 精确定位目标变量，与 `crystal_core.hpp:1148–1187` 对齐并补全 `Maj` 的 `a==c`、`b==c` 分支，共 `Ch` 5 条 + `Maj` 6 条）；
     - 代入全部 Step 2 已知变量后，利用 Python 原生任意精度 `int` 位向量执行 RREF 消元与全回代（枢轴 `lead = (row & -row).bit_length() - 1`、行重 `row.bit_count()`；若化简出现空变量行且常数项为 `1`，即 `0 = 1`，立即设置 `early_unsat = True`）；
     - 抽出全部重量为 1 的补充单位文字（$x_i = c$），以及全部重量为 2 的等价行（$x_i \oplus x_j = c$）；经**带符号并查集（Signed Union-Find）**合并传递闭包：若合并等价链时遇**奇校验环**（$x_i = \neg x_i$）或合并后的等价类取值与已知 `unit_clauses` / Step 2 常数矛盾，**立即置 `early_unsat = True`**；一致时导出等价 2-CNF 子句（$c=0 \implies [-l_i, l_j], [l_i, -l_j]$；$c=1 \implies [l_i, l_j], [-l_i, -l_j]$）及简化 `xor_clauses`；
     - **模型恢复与投影语义注记**：P3 主方案采用 **2-literal 桥接子句注入**（解集严格不变 $\mathrm{Sol}(\Phi') = \mathrm{Sol}(\Phi)$），原生保留全部 `W[0..15]` 变量标识；若未来扩展采用显式变量替换消元 $\Phi[y \leftarrow x]$，SAT 模型恢复提取 `w0_15` 前**必须**沿等价链回代 $\tau'(y) = \tau(x)$。
     - 同步精确计算结构门控指标：进位门阻塞率 $r_{\text{lock}} = N_{\text{active\_carry}} / ((10R - 48) \cdot 31)$（$R \ge 16$）与 GF(2) 残余自由度 $D_{\text{gf2}}$。
2. **Carry-Lock 感知三区混合路由 `CarryLockHybridRouter.solve(...)`**（注：$D_{\text{gf2}} \le 2150 / 2450$ 阈值为面向 `R in [22, 32], free_bits <= 40` 的 `make_end_free` 校准区间）：
   - **Zone 1（线性主导：$r_{\text{lock}} < 0.29 \land D_{\text{gf2}} \le 2150$）**：将单位+2-lit 等价子句注入隔离 `CMS-XOR` 求解（覆盖 `R<=24 12b..32b` 及 `R=25 12b`）。
   - **Zone 2（相变边缘：$0.29 \le r_{\text{lock}} < 0.325 \land D_{\text{gf2}} \le 2450$）**：先给隔离 `CMS-XOR` 分配短预算探针 `min(0.6s, 0.20 * tl)`（覆盖 `R=25 24b` 等边缘档）；若未 SAT 立即携带全量桥接单位+等价子句切换至 **`P1+P2 CleanDynamicNeuralCDCL`**。
   - **Zone 3（Carry-Lock 阻塞：$r_{\text{lock}} \ge 0.325 \lor D_{\text{gf2}} > 2450$）**：直接绕过 `CMS-XOR`（覆盖 `R>=26 12b..16b`、`R=28` 等），将 `100%` 时间预算投入 **`P1+P2 CleanDynamicNeuralCDCL`**（并已注入双平面单位与 2-literal 等价类）。
3. **Windows `pycryptosat` 安全子进程隔离 `SafeCMSSubprocessRunner`（响应 M1 Caveat 2）**：
   - 凡涉及限时调用 `pycryptosat`，默认通过独立隔离工作子进程执行（或经可配置保护开关封装 `subprocess.run(..., timeout=tl + 1.0)` 传递紧凑 JSON/二进制约束）；
   - 若子进程返回非零崩溃码（如 Windows `0xC0000005` / `3221225477`）或超时无解，主进程捕获异常而不崩溃，自动携带 Step 1–3 已抽出的单位+等价子句**无缝降级**至 `P1+P2 CleanDynamicNeuralCDCL` 继续求解；所有返回 `w0_15` 仍强制经 `core.verify_preimage` 验真。

```python
# prototypes/p3_dual_bridge_router.py 核心接口
class BridgeableSHA256XORCNFEncoder(SHA256XORCNFEncoder):
    # 隔离记录 self.Ch_vars[t][b] 与 self.Maj_vars[t][b]，与原生编码器 1:1 等价

class DualPlaneEquivalenceBridge:
    def extract(self, H_target, rounds: int, fixed_words=None, fixed_bits=None) -> BridgeExtractionResult:
        # 返回 early_unsat, unit_clauses, equiv_2lit_clauses, simplified_xors, carry_lock_rate, gf2_residual_dof

class CarryLockHybridRouter:
    def solve(self, H_target, rounds=26, fixed_words=None, fixed_bits=None,
              conflict_budget=250000, time_limit_sec=6.0, safe_cms_subprocess=True) -> RouterSolveResult: ...
```

#### 3.3 适用边界
- 阈值校准区间为 $R \in [22, 32],\; \text{free\_bits} \le 40$ 的 `make_end_free` 盲求逆（完整高效区间 $R \in [16, 32]$）；在 $R \in [16, 20]$ 极大自由度（如 `free_bits >= 256`）下应采用归一化自由度或按 $r_{\text{lock}}$ 路由以免误触发 `D_gf2` 上限；对于 $R \ge 48$，全局 Python Bitset RREF 矩阵规模过大且阻塞率极高，路由器强制关闭全局 RREF（交由 P4 局部窗口处理）。

#### 3.4 预期收益
- 在 `R<=25` 保留 `CMS-XOR` 的 `12~135ms` 极速优势并彻底杜绝 `pycryptosat` 进程崩溃；
- 在 `R=26 12b..16b`、`R=28 12b..16b` 上彻底扭转原 XOR `0/16` 归零僵局（绕过无效的 5s 高斯消元死等，直接切换桥接增强 CDCL），2-literal 等价类注入可实质化简 `CDCL` 初始搜索空间（依据定理 T6 保解集不变性）。

#### 3.5 潜在失效风险与缓解
- **风险 1（子进程启动延迟）**：Windows `spawn` 模式下启动 Python 子进程约耗 `0.15~0.25s`；**缓解**：可复用常驻单 worker 子进程池（若该 worker 因 `0xC0000005` 退出则自动拉起新 worker 并在当前例直接回退 CDCL）。
- **风险 2（Zone 2 探针损耗）**：Zone 2 短探针失败会占去至多 `0.6s`（约 `10%~20%` 时限）；**缓解**：严格限制在 `[0.29, 0.325)` 窄带内且上限封顶 `0.6s`。

---

### 方案 P4：`R=64` 虫洞 MCTS 局部窗口 GF(2) + `W[32]` 耦合修复 + 轻量化评估（`LocalWindowWormholeMCTS`）

#### 4.1 前提假设
1. **原生 `R=64` 虫洞 MCTS 卡死根因**：`wormhole_mcts.py:465, 555` 硬编码 `rounds <= 28` 禁用 GF(2)；`L337, 346` 的 `splice_block[:8]` 截断丢弃了全部 `W[32]` 比特；`L539` 与 `L621` 在单次扩展内执行两次 64 轮网络前向；前 28 次模拟全部耗尽在根节点 `depth=1`，最终 4 例 100% 停滞在 `unk_drop=94, gate_drop=26, splice_drop=8, pruned_conflicts=0`。
2. **全图 GF(2) 不可行性**：R2 实测与 R4 `formal_checks_summary.json` 证实，强开全图 `R=64` GF(2)（`30,976` 变量）单次耗时 `901.5ms~1895ms`（在 `3.0s` 预算内仅能跑 `1` 次模拟）且全盲态下 `gf2_deductions = 0`。
3. **密码学诚实边界（必须显式界定）**：**在严格零泄漏、512-bit 全盲 `R=64` 条件下，没有任何多项式时间的局部窗口线性化或符号传播能够直接穿透 64 轮模加进位雪崩并还原完整原像。** 本方案针对的是工程结构性死锁（`94/26/8/0` 同质浅层停滞），而非宣称攻破完整 64 轮 SHA-256 抗原像性。

#### 4.2 作用机制与接口设计

**源码修复锚点**：
- `v10_ouroboros64/ouroboros_mcts/wormhole_mcts.py:312–350`（`splice_block[:8]` 截断丢弃 `W[32]`）
- `wormhole_mcts.py:256–394`（根节点 28 个动作导致深度 1 耗尽预算）
- `wormhole_mcts.py:465, 555`（`rounds <= 28` 禁用 GF(2)）
- `wormhole_mcts.py:539, 621`（单次扩展双次 `_neural_score_env` 前向）

**核心机制（隔离于 `v10_ouroboros64/prototypes/p4_local_wormhole_mcts.py`）**：
1. **修复 `splice_block[:12]` 完整跨字耦合与紧凑多区配额**：
   - 将 `extra_assignments` 截断上限由 `[:8]` 修正为 `[:12]`，完整保留 `S[32, {0,4,3,7}, 0..1]`（8 bit）与 `W[32, 0..3]`（4 bit），从而在 $T_1 = h + \Sigma_1(e) + \text{Ch}(e,f,g) + K_{32} + W_{32}$ 上形成真实的跨字低位进位闭合；
   - 将每节点 `max_branch_candidates` 压缩至 `8~10`（总动作 $\le 12$），均匀配额至三大功能区：① **末端摘要锚定区**（`W[60..63]` / `S[61..63]`，直连 `S[64] = H - IV`，打破与 `H` 无关的同质轨迹）、② **中间拼接区**（`S[32]/W[32]`）、③ **起始注入区**（`W[0..3]`），使树搜索在 `10` 余次模拟内即可突破 `depth=1` 进入深层。
2. **局部轮窗 GF(2) 精确消元 `local_window_gf2_propagate(env, center_round, half_win=4)`**：
   - 围绕当前动作轮次 $t_c$ 选取局部窗口 $[t_L, t_R] = [\max(0, t_c - h), \min(R-1, t_c + h)]$（对称闭区间宽度 $w = t_R - t_L + 1 \le 9$，默认 $h=4$ 时满宽 $w=9$、$V_{\text{win}}=5088$、立方理论比 `225.65x`；而 `formal_checks` `B5` 基准测试采用对齐的半开窗 `[28, 36)`，$w=8$、$V_{\text{win}} = 4608$、理论立方比 `303.77x`、矩形操作比 `361.51x`）；
   - 收集窗口内消息调度（$t \ge 16$）及压缩线性方程，叠加已满足条件线性化前提的 `Ch`/`Maj` 规则，代入 `env` 中已知 `{0,1}` 位并压缩重标号至紧凑变量 $0 \dots V_{\text{win}}-1$（`w=8` 时 $V_{\text{win}} = 4608$，对比全局 `30,976` 缩减 **`6.7222x`**；`w=9` 时 $V_{\text{win}} = 5088$，缩减 `6.0881x`）；
   - 直接调用既有 C++17 接口 **`core.solve_gf2_equations(V_win, compact_eqs)`**（`engine.py:365–408`，最新 `formal_checks_summary.json` `B5` 实测 `w=8` 仅耗时 **`2.141 ms`**，较全局 `R=64` **`901.539 ms`** 实测加速 **`421.12x`**）：
     - 若返回 `conflict == True` $\implies$ 立即判定 `child.is_conflict = True` 并执行矛盾剪枝（`pruned_conflicts += 1`）与单/二元冲突学习；
     - 若返回新增 `units` $\implies$ 写回 `env` 并补跑 4 步轻量 `env.propagate(use_gf2=False, max_iters=4)` 触发级联传播。
3. **单次前向与零延迟物理叶评估**：
   - 彻底移除子节点生成时（原 `L621`）的第二次 `_neural_score_env(child_env)` 调用；
   - 非冲突叶节点价值改为纯符号物理即时收益：
     $$V_{\text{leaf}} = 0.45 \ln(1 + \Delta \text{unk}) + 0.35 \ln(1 + \Delta \text{gates}) + 0.25 \ln(1 + \Delta \text{splice}) + 0.15 \ln(1 + \text{gf2\_ded})$$
     仅当该节点被 UCB 选中并需要展开子动作时才执行一次净态神经前向（`conflict_heatmap=None`）。

```python
# prototypes/p4_local_wormhole_mcts.py 核心接口
def local_window_gf2_propagate(
    env: GeneralizedSymbolEnv,
    center_round: int,
    half_win: int = 4,
    core: Optional[OuroborosCore] = None,
) -> Dict[str, Any]:
    # 返回 {"conflict": bool, "units_applied": int, "rank": int, "window": (t_L, t_R), "num_vars": int, "elapsed_ms": float}

class LocalWindowWormholeMCTS(WormholeMCTS):
    def __init__(self, *args, half_win: int = 4, max_branch_candidates: int = 8,
                 use_local_gf2: bool = True, skip_leaf_neural_eval: bool = True, **kwargs): ...
```

#### 4.3 适用边界与密码学诚实界定
- **适用边界**：`R=64`（及 `R=48`）虫洞树搜索、中间拼接位点局部耦合消除、以及配合强已知前缀/内部部分泄漏条件下的加速寻优；
- **密码学边界**：对于完整 512-bit 自由度全盲 `R=64` SHA-256 实例，当所有内部轮次非线性进位门（`18,352` 门）仍以自由度高度耦合时，深度 $d \le 10$ 的局部窗口多项式传播无法推导出跨 64 轮全局原像。本方案旨在消除浅层同质停滞（`94/26/8/0`）、提高搜索深度、局部剪枝率与输入 $H$-敏感度，而非伪造 64 轮盲破能力。

#### 4.4 预期收益
- 在同等 `3.0s` 时限下，模拟次数预期由 `51` 次提升至 `150~350+` 次，最大搜索深度由 `2` 提升至 `>= 5~10`；
- 消除 4 个不同 $H_{\text{target}}$ 样本完全收敛至同一增量 `94/26/8/0` 的退化现象；在局部过约束分支上实现真实 `pruned_conflicts > 0`。

#### 4.5 潜在失效风险与缓解
- **风险**：当 `half_win >= 7`（$w \ge 15$）时，窗口矩阵稀疏 RREF 复杂度按 $O(\text{rank} \cdot m \cdot V)$ 快速膨胀会拉低 MCTS 模拟吞吐；**缓解**：严格限制 `half_win <= 4`（$w \le 9, V_{\text{win}} \le 5088$），且仅在局部 ARX 传播产生变化或深度达到触发步长时调用。

---

## 第二梯队：长期训练范式重构路线图（P5，需重训）

### 方案 P5：Domain-B-Native 双路径解耦、对齐掩蔽与去拷贝残差目标

> **梯队说明**：本方案涉及网络输入特征重构、损失函数重写与完整 GPU 重新训练，**严格归属第二梯队（需重训）**，不在 M3 无重训原型对标中强行使用，作为根本解决 R1 维度一三大结构病灶的长期规范。

#### 5.1 前提假设
1. 只要训练输入混合使用域 A 残差通道 `[46..72]`（当前列 L2 范数均值 `1.0611`，显著压倒双极主通道 `0.6226`），神经网络就会优先依赖「只存在于已知答案残差场中的捷径」，在 `zero_defects=True` 域 B 下特征信号必然坍缩。
2. 只要 `PairAcc` 在 `>99.8%` 未翻转样本上以全局 4 态交叉熵优化，`argmax` 恒等拷贝输入 $C_d$ 即成为难以逃离的局部极小吸引子。
3. 只要 `Stage B` 使用均匀随机掩蔽（`rng.random((16,32)) < reveal_prob`）配弱化 `ARX(max_iters=6)`，而推理期面临的是前缀焊死、后缀连续未知的 `make_end_free_constraints` 配强双平面推导，训练-推理分布错位就不可避免。

#### 5.2 核心重构规范
1. **Domain-B-Native 双头输入隔离架构**：
   - 将 `Ouroboros64Net` 输入投影拆分为独立域路径：**域 A 愈合投影**（消费全部 104 维含 `[46..72]` 残差通道）与 **域 B 盲求逆投影**（物理屏蔽 `[46..72]` 残差通道，仅接收 `[0..26]` 已知双极、`[27..45]` 完整 19 平面未知掩码、`[73..91]` 真搜中热力图与 `[92..103]` 常数/几何编码），禁止域 A 残差梯度反向污染域 B 特征提取器；
   - 训练期全程使用完整 `W/S/C` 推论回填（对齐 P1 `_build_full_crystal_var_map`），并在轨 1 在线 CDCL 中真实采集并注入轨迹 `conflict_heatmap`（覆盖 `W` 与 `S/C` 平面），彻底消灭通道 `[73..91]` 零训练暴露问题。
2. **Stage B 掩蔽分布与前置闭合对齐**：
   - 废除 `trainer.py:1295` 单一均匀随机掩蔽，构造混合分布训练课程：`70%` `make_end_free_constraints` 连续后缀未知（`fb` 覆盖 `[8, 64]`）+ `30%` 含中间 `splice_round` 锚定的分段块掩蔽；
   - 预处理统一使用 `GeneralizedSymbolEnv.propagate(use_gf2=(R<=28), max_iters=16)`（对 `R>28` 使用 P4 局部窗口 GF(2)），确保训练与推理期已知位分布 100% 一致。
3. **去恒等拷贝进位残差目标与温度校准正则**：
   - 将 `DefectHealingHead.pair_head` 的预测目标由绝对 4 态 $(c_b, c_{b+1})$ 重构为**翻转指示联合残差** $(\Delta c_b, \Delta c_{b+1}) = (c_b^{\text{clean}} \oplus c_b^{\text{dirty}}, c_{b+1}^{\text{clean}} \oplus c_{b+1}^{\text{dirty}})$，对未翻转对施加下采样权重或 Focal/Margin 损失：
     $$\mathcal{L}_{\text{pair\_anti\_copy}} = \mathcal{L}_{\text{CE}}(\text{hard\_flipped}) + \lambda_{\text{margin}} \cdot \text{softplus}\big(\text{NLL}_{\text{model}} - \text{NLL}_{\text{copy}} + \gamma\big)$$
     强制模型在真实翻转进位对与整体 NLL 上同时严格击败恒等拷贝基线；
   - 对 `phase_logits` 引入可微温度校准层与 **Brier / 校准熵下界正则**：当预测极性不可区分（如密码学高轮强随机目标）时惩罚过度自信尖峰（要求验证集盲态 $\text{BCE} \le \ln 2 + \epsilon$），从根源上杜绝 `BCE = 0.72~0.75 > ln 2` 的假置信输出。

#### 5.3 适用边界与潜在失效风险
- **适用边界**：面向下一代独立 GPU 重训里程碑（如 A100/H100 周期），严禁用于篡改本工作区已归档的 `a100_m2`、`a100_m3`、`a100_m3v2` 检查点。
- **失效风险**：在 $R \ge 32$ 强伪随机高轮盲区，任何神经网络都无法学到稳定的单比特全局极性；必须通过显式验证集 $\text{BCE} \le \ln 2$ 与校准误差（ECE）监控防止网络过拟合小规模随机批次噪声。

---

## M3 原型接口契约、隔离边界与验证准则

1. **目录与文件隔离**：所有第一梯队原型与配对基准脚本必须写入 `g:\SHA-LM\v10_ouroboros64\prototypes\`，**严禁修改** `ouroboros_core/`、`ouroboros_net/`、`ouroboros_mcts/`、`ouroboros_train/`、`ouroboros_bench/`、`runs/`、`formal_proofs/`。
2. **数值精度与确定性**：所有原型脚本入口必须调用 `enforce_fp32_determinism()` 并显式断言 `allow_tf32_matmul == False` 且 `allow_tf32_cudnn == False`。
3. **双域物理隔离与 100% 验真**：
   - 域 B 盲求逆严禁向任何输入或决策通道泄漏真实内部 `W/S/C`（除 `make_end_free_constraints` 指定的固定边界与 `H_target` 外）；
   - 所有声称 `SAT` 的原像 `w0_15` 必须 100% 通过 `OuroborosCore.verify_preimage(w0_15, H_target, rounds=rounds)` 程序化验真。
4. **R4 形式化约束边界**：M3 原型实现必须严格遵守 `R4_FORMAL_THEOREMS.md` 中的已证边界条件（例如 P2 探测保持 $B_c = 0$ 以保证直接冲突非负收益；P3 仅在 `Carry-Lock` 低阻塞区启用 `CMS-XOR` 并以 2-literal 等价子句保持解集不变；P4 窗口宽度受限于 $w \le 9$ 以按矩形稀疏 RREF $O(\text{rank} \cdot m \cdot V)$ 保障毫秒吞吐）。
