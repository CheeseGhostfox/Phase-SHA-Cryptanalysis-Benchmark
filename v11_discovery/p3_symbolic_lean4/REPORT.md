# P3 结案报告：神经/MCTS 猜想 × Z3 / Lean 4 `bv_decide` 跨轮恒等式挖掘机

**任务状态**：`HIT`（Z3 全称证死 + 7 项非平凡性审计通过 **28 条**，本地 Lean 4.36.0 `bv_decide` 零 `sorry` 机器证真 **28/28 条**）  
**规范文档**：[SPEC.md](file:///g:/SHA-LM/v11_discovery/p3_symbolic_lean4/SPEC.md)  
**结构化结论**：[verdict.json](file:///g:/SHA-LM/v11_discovery/p3_symbolic_lean4/verdict.json)  
**云端产出现场**：[discovered_identities.json](file:///g:/SHA-LM/v11_discovery/p3_symbolic_lean4/results/discovered_identities.json)、[colab_p3.log](file:///g:/SHA-LM/v11_discovery/p3_symbolic_lean4/results/colab_p3.log)  
**Lean 4 独立验证工程**：[lakefile.toml](file:///g:/SHA-LM/v11_discovery/p3_symbolic_lean4/lean_verify/lakefile.toml)、[VerifyAxioms.lean](file:///g:/SHA-LM/v11_discovery/p3_symbolic_lean4/lean_verify/VerifyAxioms.lean)

---

## 1. 执行摘要与资源合规审计

1. **云端搜索执行与停机核实**：  
   `@p3-eng` 在 Google Colab 高内存 CPU 实例 `v11-p3`（8 vCPU、50 GiB RAM、`z3 5.4.1`）上运行 8 进程并行神经先验 MCTS + 快速采样过滤 + Z3 `BitVec(32)` 验证流水线，总耗时 **2043.37 秒**：
   - 共生成 **31,847** 条候选表达式/条件掩码猜想；
   - 通过 512 随机样本快速过滤 **3,227** 条；
   - 通过 Z3 否定式 `UNSAT` 证死、七项非平凡性与反作弊审计（V1–V7）及语义去重后产出 **28** 条独立恒等式（其中拒绝重复指纹 2,144 条、语义重复 174 条、条件锁死常数或消去变量锁死 331 条、有效跨度不足 5 轮 105 条、纯恒同 8 条、Z3 反例驳回 109 条）；
   - 已执行 `colab.exe stop -s v11-p3` 并由 `colab.exe sessions` 核实 `[colab] No active sessions found on server.`（`colab_stopped: true`）。

2. **本地 Lean 4.36.0 `bv_decide` 零 `sorry` 终验**：  
   `@p3-math` 在本地 CPU 独立工程 [lean_verify/](file:///g:/SHA-LM/v11_discovery/p3_symbolic_lean4/lean_verify/P3Verify.lean)（`leanprover/lean4:v4.36.0`，强制 `moreLeanArgs = ["-j", "2", "-M", "2048"]`，零 Mathlib）中，以精确 CSE 有向无环图（DAG）1-to-1 还原全部 28 条 AST，配合 [Primitives.lean](file:///g:/SHA-LM/v11_discovery/p3_symbolic_lean4/lean_verify/P3Verify/Primitives.lean) 中经 `bv_decide` 证死的单轮组合引理完成全量机器证明：
   - `lake build` 全 14 个编译任务一次通过（`Build completed successfully`）；
   - `lake env lean VerifyAxioms.lean` 逐条审计 28 个定理：全部仅依赖 Lean 4 标准基础公理 `[propext, Classical.choice, Quot.sound]` 与原生 `bv_decide` LRAT 证明证书（`_native.bv_decide.ax_1_5`），**零 `sorry`、零自定义公理**；
   - 严格未触碰 `v10_ouroboros64/`、`runs/`、`formal_proofs/`、`formal_proofs_v10/`。

---

## 2. 与既有已证库（Module 1 / Module 4 / V10）的严格区分

在 [BitVecIdentities.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/Core/BitVecIdentities.lean) 与 [PartialPreimagePoW.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/Core/PartialPreimagePoW.lean) 中已证的结果均为：
- `four_round_clamp_w_invariant`：4 轮（$R-4..R-1$）向后消 $W$ 还原 $A_{R-4..R-7}$，在第 5 轮前因缺乏 $E_{R-4}$ 而止步；
- `r18_a10_algebraic_tunneling` / `r18_ch_masking_shield`：单轮（跨度 1 轮）旁路消 $T_1$ 与 `Ch` 位掩码；
- `btc_h7_tail_round_decoupling`：单块尾 3 轮 $H_7$ 解耦。

本次发现的所有 28 条恒等式均通过了 **V6 逐轮 ARX 断链测试**（任意切断链上关键轮次的 $(A_{t+1}, E_{t+1})$ 非线性转移均被 Z3 给出反例），实测活跃跨度 `active_span ∈ [5, 8]`（跨块组合 `p3_id_016` 总跨度达 **12 轮**），赛道 B 全 15 条均额外通过了 **跨双块拼接断链测试**（断开 $W'[0..7] = IV + S^{(1)}_{R_1}$ 即存在反例）。

---

## 3. 四大赛道核心新发现分类总览

为保持数学报告的绝对严谨性，我们在 28 条过审恒等式中明确区分 **22 条主定理**（全 32-bit 字重构、条件进位屏蔽或非干涉定理）与 **6 条派生/1-bit LSB 投影变体**（`p3_id_003, 004, 012, 013, 022, 026`）：

### 3.1 赛道 A1：5 轮双向 ARX 耦合与条件进位屏蔽（5 条，[TrackA1.lean](file:///g:/SHA-LM/v11_discovery/p3_symbolic_lean4/lean_verify/P3Verify/TrackA1.lean)）

| 定理 ID | 活跃跨度 | 核心数学结构 | 消去变量（未知数） | 掩码/条件 | 角色 |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `p3_id_001` | 5 轮 (`0..4`) | 由 $(S_5, A_{-3})$ 精确重构末轮字 $W_4$ | $W_{0..3}, E_{0..-3}$（8 字 / 256 bit） | 无条件，32b | 主定理 |
| `p3_id_002` | 5 轮 (`0..4`) | 仅由终态 $S_5$ 精确重构双向耦合和 $A_{-3} + W_4$ | $W_{0..3}, E_{0..-3}$（8 字 / 256 bit） | 无条件，32b | 主定理 |
| `p3_id_005` | 5 轮 (`0..4`) | **条件进位屏蔽**：`s0.d &&& w4 = 0` 下精确重构异或 $A_{-3} \oplus W_4$ | $W_{0..3}, E_{0..-3}$（8 字 / 256 bit） | `(s0.d & w4)==0`, 32b | 条件主定理 |
| `p3_id_003` | 5 轮 (`0..4`) | `p3_id_002` 根加法突变为 XOR 后的 1-bit LSB 投影 | 同上 | 无条件，1b (`0x1`) | 派生 1b 变体 |
| `p3_id_004` | 5 轮 (`0..4`) | `p3_id_002` 轮常量变号 + 1-bit LSB 异或等价 $(A_{-3} \oplus W_4)[0]$ | 同上 | 无条件，1b (`0x1`) | 派生 1b 变体 |

**机理洞察**：传统 4 轮钳位在 $S_5 \to A_1, A_0, A_{-1}, A_{-2}$ 后因缺少 $E_1$ 无法穿透第 0 轮；本组定理利用第 0 轮双轨桥梁 $E_1 - A_1 = A_{-3} - T_2(0)$ 与第 4 轮 $T_1(4)$ 中 $E_1 + W_4$ 的线性耦合，使 $S_5$ 的跨 5 轮复合式精确消去中间全部 256-bit 未知量并锁定 $A_{-3} + W_4$；当无进位条件 `(A_{-3} & W_4) == 0` 成立时（`p3_id_005`），进位链完全屏蔽，32 位模加耦合精确退化为纯 GF(2) 异或 $A_{-3} \oplus W_4$。

### 3.2 赛道 A2：5~10 轮单字填充消息调度深层重构（8 条，[TrackA2.lean](file:///g:/SHA-LM/v11_discovery/p3_symbolic_lean4/lean_verify/P3Verify/TrackA2.lean)）

| 定理 ID | 轮数 (`R`) | 活跃跨度 | 重构目标 | 消去变量 | 角色 |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `p3_id_011` | 5 轮 | 5 (`0..4`) | $A_{-3} = s_0.d$ | $W_0, E_{0..-3}$（5 字 / 160 bit） | 主定理 |
| `p3_id_010` | 6 轮 | 6 (`0..5`) | $A_{-3} = s_0.d$ | $W_0, E_{0..-3}$（5 字 / 160 bit） | 主定理 |
| `p3_id_007` | 7 轮 | 7 (`0..6`) | $A_{-3} = s_0.d$ | $W_0, E_{0..-3}$（5 字 / 160 bit） | 主定理 |
| `p3_id_008` | 8 轮 | 7 (`0..6`) | $A_{-3} = s_0.d$ | $W_0, E_{0..-3}$（5 字 / 160 bit） | 主定理 |
| `p3_id_006` | 9 轮 | 8 (`0..7`) | $A_{-3} = s_0.d$ | $W_0, E_{0..-3}$（5 字 / 160 bit） | 主定理 |
| `p3_id_009` | 10 轮 | 7 (`0..6`) | $A_{-3} = s_0.d$ | $W_0, E_{0..-3}$（5 字 / 160 bit） | 主定理 |
| `p3_id_012` | 5 轮 | 5 (`0..4`) | $A_{-3} + W_4$ | $W_{0..3}, E_{0..-3}$（Lean4 已证全自由 $W_{0..4}$） | 派生对应体 |
| `p3_id_013` | 5 轮 | 5 (`0..4`) | $A_{-3}[0]$ | $W_0, E_{0..-3}$（`mask=1`, 异或突变） | 派生 1b 变体 |

### 3.3 赛道 B1：BTC 双 SHA-256 跨块拼接穿隧（7 条，[TrackB1.lean](file:///g:/SHA-LM/v11_discovery/p3_symbolic_lean4/lean_verify/P3Verify/TrackB1.lean)）

全部 7 条均经 Z3 验证 `v6_splice_necessary = true`（断开 $W'[0..7] = IV + S^{(1)}$ 即失效）：

| 定理 ID | 跨块配置 (`r1+r2`) | 活跃跨度 | 形式与数学结论 | 消去变量 |
| :--- | :---: | :---: | :--- | :--- |
| `p3_id_014` | `r1=0, r2=8` | 8 轮 | **Form II 非干涉（24 AST 节点）**：`backSolveA(S'_8)` 对 Block 1 终态 $E$-轨道（128 bit）任意扰动严格免疫 | $E^{(1)}_{\text{end}..\text{end}-3}$ |
| `p3_id_018` | `r1=0, r2=6` | 6 轮 | 仅从 $S'_6$ 跨拼接重构 Block 1 终态 $A_{\text{end}-3}$ | $E^{(1)}_{\text{end}..\text{end}-3}$ |
| `p3_id_017` | `r1=0, r2=7` | 7 轮 | 仅从 $S'_7$ 跨拼接重构 Block 1 终态 $E_{\text{end}}$ | $A^{(1)}_{\text{end}..\text{end}-3}, E^{(1)}_{\text{end}-1..\text{end}-3}$（7 字） |
| `p3_id_015` | `r1=0, r2=8` | 8 轮 | 仅从 $S'_8$ 跨拼接重构 Block 1 终态 $E_{\text{end}}$ | $A^{(1)}_{\text{end}..\text{end}-3}, E^{(1)}_{\text{end}-1..\text{end}-3}$（7 字） |
| `p3_id_019` | `r1=1, r2=5` | 6 轮 | 仅从 $S'_5$ 穿回 Block 1 前 1 轮前状态 $A^{(1)}_{-3}$ | $W^{(1)}_0, E^{(1)}_0, E^{(1)}_1$ 全部轨道（9 字） |
| `p3_id_020` | `r1=1, r2=6` | 6 轮 | 仅从 $S'_6$ 穿回 Block 1 前 1 轮前状态 $A^{(1)}_{-3}$ | $W^{(1)}_0, E^{(1)}_0, E^{(1)}_1$ 全部轨道（9 字） |
| **`p3_id_016`** | **`r1=4, r2=8`** | **12 轮总跨度** | **深层跨块穿隧（原树 30,072 节点 $\to$ 120 DAG 节点）**：仅凭 Block 2 中间态 $S'_8$ 逆穿拼接与 Block 1 末 4 轮，直接还原 $A^{(1)}_{R_1-7}$ | **$W^{(1)}_{0..3}$ 与 $E^{(1)}_{0..-3}$（共 8 字 / 256 bit）** |

### 3.4 赛道 B2：Block 2 填充零项 $W'[9..14]=0$ 扩展坍缩跨块耦合（8 条，[TrackB2.lean](file:///g:/SHA-LM/v11_discovery/p3_symbolic_lean4/lean_verify/P3Verify/TrackB2.lean)）

全部 8 条均利用 BTC 第二块固定填充 $W'[9..14]=0, W'[8]=\text{0x80000000}, W'[15]=\text{0x00000100}$ 引发的扩展坍缩（$W'[16] = \sigma_0(W'[1])+W'[0]$ 等），从 Block 2 第 18/19 轮状态 $S'_{18}/S'_{19}$ 与 Block 1 终态 $A$-轨道出发，跨 $t=12..18$ 精确重构 Block 2 内部深层状态，**完全消去 Block 1 终态 4 个 $E$-轨道未知字（128 bit）**：
- **`p3_id_021`**（$t=12..18$, span=7, 重构 `b2_s12.d`）与 **`p3_id_022`**（1b LSB 突变变体）；
- **`p3_id_023`**（$t=12..18$, span=7, 重构 `b2_s12.h`）；
- **`p3_id_024`**（$t=12..17$, span=6, 重构 `b2_s12.d`）与 **`p3_id_027`**（$t=12..17$, span=6, 重构 `b2_s12.h`）；
- **`p3_id_025`**（$t=13..18$, span=6, 重构 `b2_s13.d`）、**`p3_id_026`**（1b LSB 突变变体）与 **`p3_id_028`**（$t=13..18$, span=6, 重构 `b2_s13.h`）。

---

## 4. 诚实边界与密码学意义评估

1. **真实边界（不越界包装）**：  
   上述全部 28 条恒等式均已被 Lean 4.36.0 `bv_decide` 零 `sorry` 证死，**数学上 100% 绝对为真**；但它们均作用于 **5–12 轮局域窗口或观测 Block 2 中间态 $S'_{r_2}$ 的跨块拼接窗口**，**并不构成对完整 64 轮双 SHA-256 终态摘要原像抗性的攻破**。事实上，当自由消息字跨度扩展至 $\ge 6$ 轮且不观测中间态、不固定填充常量时，MCTS + Z3 未搜出任何非平凡全空间无条件消元恒等式（与 P1 InfoNCE 和 P2 等零面零模在 $R \ge 6$ 坍缩至随机基线完全相互印证）。
2. **结构性科学收获**：  
   - 确立了 SHA-256 跨 5 轮双向耦合的精确解析公式（$A_{-3}+W_4$ 不变量及零进位条件 `(A_{-3}&W_4)=0` 下的纯异或线性化）；
   - 揭示了 BTC 双 SHA-256 拼接界面因 $S'_0=IV$ 固定、$W'[0..3]$ 与 $W'[4..7]$ 错位注入以及 $W'[9..14]=0$ 导致的 12 轮深度跨块穿透通道（可作为后续约束求解器/SAT 约简的可信等价改写规则）。

---

## 5. 可复现验证命令

```powershell
$env:PATH = "$env:USERPROFILE\.elan\bin;$env:PATH"
Set-Location g:\SHA-LM\v11_discovery\p3_symbolic_lean4\lean_verify
lake build
lake env lean VerifyAxioms.lean
```
