# B2-P15 结案报告：截断前导零专用 MitM / Biclique / Splice-and-Cut 自动搜索与 Lean 4 证真

**总体裁决**：`NULL`（在完整 64 轮 SHA-256 及 128 轮 BTC Double-SHA256 截断前导零目标 `clz(H[0]) >= k` 下，未发现任何超越暴力穷举 $2^k$ 的加速结构；但 **Stage 1 阳性校准全通过**、**Stage 2 四赛道全量边界测绘完毕**、**11/11 候选独立块已通过 Z3 与 Lean 4.36.0 `bv_decide` 零 `sorry` 形式化证真**）  
**设计规范**：[SPEC.md](file:///g:/SHA-LM/v12_frontier/track_b2_p15_mitm_biclique/SPEC.md)  
**结构化结论**：[verdict.json](file:///g:/SHA-LM/v12_frontier/track_b2_p15_mitm_biclique/verdict.json)  
**实测数据集**：[stage1_calibration.json](file:///g:/SHA-LM/v12_frontier/track_b2_p15_mitm_biclique/results/stage1_calibration.json)、[stage2_btc_search.json](file:///g:/SHA-LM/v12_frontier/track_b2_p15_mitm_biclique/results/stage2_btc_search.json)、[lean_candidates.json](file:///g:/SHA-LM/v12_frontier/track_b2_p15_mitm_biclique/results/lean_candidates.json)、[run_log.txt](file:///g:/SHA-LM/v12_frontier/track_b2_p15_mitm_biclique/results/run_log.txt)  
**Lean 4 形式化工程**：[lakefile.toml](file:///g:/SHA-LM/v12_frontier/track_b2_p15_mitm_biclique/lean_verify/lakefile.toml)、[Primitives.lean](file:///g:/SHA-LM/v12_frontier/track_b2_p15_mitm_biclique/lean_verify/P15Verify/Primitives.lean)、[Stage1Calib.lean](file:///g:/SHA-LM/v12_frontier/track_b2_p15_mitm_biclique/lean_verify/P15Verify/Stage1Calib.lean)、[Stage2BTC.lean](file:///g:/SHA-LM/v12_frontier/track_b2_p15_mitm_biclique/lean_verify/P15Verify/Stage2BTC.lean)、[VerifyAxioms.lean](file:///g:/SHA-LM/v12_frontier/track_b2_p15_mitm_biclique/lean_verify/VerifyAxioms.lean)

---

## 1. 执行环境、参考对拍与云端资源核实

1. **精确引擎对拍**：本地 CPU 单元测试 [test_step1_engine.py](file:///g:/SHA-LM/v12_frontier/track_b2_p15_mitm_biclique/test_step1_engine.py) 与云端 `40` 组对照均实现与 [sha256_ref.py](file:///g:/SHA-LM/src/sha256_ref.py) 及标准库 `hashlib.sha256`（含 80-byte BTC Double-SHA256）逐比特零误差对拍（`xor_diff == 0`）。
2. **云端执行与释放核实**：`@p15-eng` 在 Colab `v12-p15`（8 vCPU `--high-mem`）上运行 CP-SAT 四色模型与 Z3 `BitVec(32)` 三重校验，总耗时 **285.622 秒**；同步结果至 `results/` 后执行 `colab.exe stop -s v12-p15`，并经 `colab.exe sessions` 核实无 `v12-p15` 残留会话（`colab_stopped: true`，见 [run_log.txt](file:///g:/SHA-LM/v12_frontier/track_b2_p15_mitm_biclique/results/run_log.txt)）。
3. **本地 CPU Lean 4 独立构建**：在 `leanprover/lean4:v4.36.0`（`moreLeanArgs = ["-j", "2", "-M", "2048"]`，零 Mathlib）下运行 `lake build`（10 个作业一次性零警告通过）及 `lake env lean VerifyAxioms.lean`，全部 **32** 个声明仅依赖 Lean 基础公理 `[propext, Classical.choice, Quot.sound]` 与原生 `bv_decide` LRAT 证明证书，**零 `sorryAx`**。

---

## 2. Stage 1：零 Bug 阳性校准与全非线性消息扩展边界

### 2.1 Stage 1A：经典减轮 SHA-256 MitM 阳性校准（含完整 $\sigma_0, \sigma_1$ 双向扩展）

在标准单块 SHA-256（含完整双向非线性消息扩展 `W[t]` 与 `W[t-16]` 逆推）与 `ModeI_Wrap256`（$k = 256$）下，OR-Tools CP-SAT 在 $R \in \{16, 20, 24, 28, 32\}$ 全部自动搜出有效四色划分并 100% 通过 Z3 `BitVec(32)` V1–V3 证死：

| 轮数 $R$ | 消息基起点 $s_w$ | 匹配点 $t_{\text{match}}$ | Red / Blue 基字 | $(d_1, d_2)$ | 有效匹配 $m_{\text{eff}}$ | $\log_2 T_{\text{attack}}$ | 加速 $\Delta \log_2 T$ | 内存 $\log_2 M_{\text{words}}$ | Z3 耗时 (ms) | Lean 4 证真 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `16` | `0` | `1` | `W[0]` / `W[15]` | `(32, 32)` | `288` | `224.087463` | `+31.912537` | `35.0` | `4360.18` | — |
| `20` | `1` | `15` | `W[13]` / `W[15]` | `(32, 32)` | `288` | `224.070389` | `+31.929611` | `35.0` | `1215.53` | `p15_stage1a_mitm_20r_{fwd,bwd}` |
| `24` | `2` | `11` | `W[10]` / `W[11]` | `(32, 32)` | `288` | `224.058894` | `+31.941106` | `35.0` | `12800.08` | `p15_stage1a_mitm_24r_{fwd,bwd}` |
| `28` | `7` | `8` | `W[13]` / `W[8]` | `(32, 32)` | `96` | `224.050626` | `+31.949374` | `35.0` | `13592.44` | `p15_stage1a_mitm_28r_{fwd,bwd}` |
| `32` | `7` | `15` | `W[16]` / `W[15]` | `(32, 32)` | `96` | `224.044394` | `+31.955606` | `35.0` | `8072.98` | `p15_stage1a_mitm_32r_{fwd,bwd}` |

**核心结构洞察**：
- 在 `p15_stage1a_mitm_28r` 中，虽然 Red 变量 `W[13]` 在后向阶段进入第 `13` 轮，但经 `backSolveA` 跨 `t=13..8` 共 6 轮的 `E -> A` 延迟污染通道，直到 `S_8.(c..h)` 才被污染，`S_8.(a, b)` 仍严格独立于 `W[13]`（在 [Stage1Calib.lean](file:///g:/SHA-LM/v12_frontier/track_b2_p15_mitm_biclique/lean_verify/P15Verify/Stage1Calib.lean) 中经 `inv6_w5_eq2` 形式化证死）。
- 在 `p15_stage1a_mitm_32r` 中，前向 `t=16..30` 虽有 `W[30]` 被 Blue `W[15]` 污染，但仅波及 `S_31.(a, e)`；后向经 `S_16 -> S_0 -> H - S_0 -> S_31` 回绕，虽有 `W[0..2]` 与 `W[31]` 被 Red `W[16]` 污染，但经 `S_0.(a..e)` 与 `inv_eq4_of_eq5` 保护，`S_31.(b, c, d)` 仍实现双向严格非干涉。

### 2.2 Stage 1B：Khovratovich 风格 Biclique 校准与 64 轮穿透极限

1. **64 轮局域独立 Biclique 子块搜索与 Z3/Lean 4 证真**：
   - `L = 6`（轮 `[0, 5]`，Red `W[0,1]`，Blue `W[4,5]`，`d1=d2=64`）：Z3 `VERIFIED`（`15949.32 ms`），已在 Lean 4 中由 `p15_stage1b_local_bic_6r_t0_{fwd,bwd}` 证死；
   - `L = 8`（轮 `[0, 7]`，Red `W[0,1]`，Blue `W[6,7]`，`d1=d2=64`）：Z3 `VERIFIED`（`13846.91 ms`），已在 Lean 4 中由 `p15_stage1b_local_bic_8r_t0_{fwd,bwd}` 证死；
   - `L = 9`（轮 `[0, 8]`，Red `W[0,1]`，Blue `W[7,8]`）：Z3 返回 `PARTIAL`（`22414.99 ms`，未通过完全独立校验），精确锁定字级局域独立二分团长度上界 `L_bic <= 8`。
2. **字级全局连续弧段穿透上限**：
   - 在严格连续弧段 + 完整非线性 $\sigma_0, \sigma_1$ 消息扩展 + 32-bit 字级不降维条件下，CP-SAT 在 $R \in \{41, 45, 52, 64\}$（含 `fixed_iv=True` 原像与 `fixed_iv=False` 伪原像）均返回 **`INFEASIBLE`**，实测字级最大穿透上限 **`r_max_penetration_word = 32`**。
   - 同时 [stage1_calibration.json](file:///g:/SHA-LM/v12_frontier/track_b2_p15_mitm_biclique/results/stage1_calibration.json) 记录了经典的 8-bit 子字 Khovratovich 独立 Biclique 解析校准基线：$R=41$（$\log_2 T = 248.201945, \Delta = +7.798055$）、$R=45$（$\log_2 T = 248.185536, \Delta = +7.814464$）与 $R=52$（$\log_2 T = 248.281484, \Delta = +7.718516$），并指出 64 轮全原像依旧受限于消息扩展强耦合而无全局字级突破。

---

## 3. Stage 2：BTC Double-SHA256 128 轮截断前导零全量搜索

| 赛道 | 覆盖范围与配置数 | 有效加速命数 (`is_speedup=True`) | 命中的最高轮数与条件 | 64/128 轮及 $k \le 128$ 截断前导零结果 |
| :--- | :--- | :---: | :--- | :--- |
| **S2-A (`btc_block1b`)** | `W[4]=0x80000000, W[5..14]=0, W[15]=640`；`86` 组 | `2 / 86` | 仅 `R=16` `btc_block1b_hdr` 在全原像 `k=256`（$\Delta = +31.912537 / +31.866733$） | `Nonce-32` 全 `0` 命中；`R >= 20` 及全部 `k <= 240`（`8..128, 224, 240`）`0` 命中 |
| **S2-B (`btc_block2`)** | `W'[8]=0x80000000, W'[9..14]=0, W'[15]=256`；`42` 组 + 局域 Biclique | `4 / 42` | 仅 `R=16`（$\Delta = +31.912537 / +31.866733$）与 `R=20`（$\Delta = +31.929611 / +31.893085$）在 `k=256` | `R >= 24`（`24, 32, 48, 64`）及全部 `k <= 224`（`8..128, 224`）`0` 命中 |
| **S2-C (`double128`)** | 跨 Block 1b + Block 2 `40` 组（`(4+8)` 至 `(64+64)`） | `1 / 40` | 仅玩具窗口 `(R1=4, R2=8)` 全原像 `k=256`（`d1=32, d2=96, m_eff=96, log2_T=223.222392, delta=+32.716207`） | 全部 `(R1, R2) >= (8+8)` 至 `(64+64)=128` 均坍缩为 `d1=128, d2=0, delta=0.0`（`0` 加速） |
| **S2-D (`bitlevel`)** | Tier 2 位级进位截断 `48` 组（$R \in \{4..24\}, k \in \{8..32\}$） | `0 / 48` | 无任何命中（如 `R=4 k=8` 仅 $d_2=1, m_{\text{eff}}=6, \Delta = -113.152003$） | $R \ge 12$ 全部 `INFEASIBLE`；严格扣减 $\Delta_{\text{carry}}$ 后 `0` 命中 |

### 为何 `clz(H[0]) >= k` 无法被 MitM / Biclique 加速（机制性证伪）
1. **回绕通道阻断定理（Mode I）**：经典 Splice-and-Cut 依赖从终态 $S_R = H - IV$ 倒推，要求 $H$ 全 256 比特已知。当目标放宽为 `clz(H[0]) >= k`（$k < 256$）时，即使将 $u_{\text{blue}}$ 个自由输出比特并入 Blue 自由度，内部匹配切面仍须同时闭合全部 256 位状态方程，单组期望解数恒为 $2^{d_1 + d_2 - 256}$，导致 $T_{\text{attack}}^{\text{wrap}} \ge \min(c_{\text{red}}, c_{\text{blue}}) \cdot 2^{256 - \max(d_1, d_2)}$，在全部 BTC 实用难度 $k \in [32, 128]$ 上指数级劣于直接穷举 $2^k$。
2. **非回绕通道截断雪崩（Mode II + Tier 2）**：若不猜测未知的 $256 - k$ 输出位，后向分支从 `clz(H[0]) >= k` 仅获得 $A_R$ 的高 $k$ 位（`B_R..H_R` 全 `White`），经单轮 `backSolveA` 立即被 `E_R, B_R, C_R, D_R` 全白污染；而双前向分支汇合至 $A_R$ 在 Tier 2 位级搜索中于 `btc_block1b_hdr` 自 $R \ge 6$、`btc_block2` 自 $R \ge 12$ 起全部 `INFEASIBLE`。
3. **跨双块已知窗口的边界定位**：Z3 与 Lean 4 再次证死 BTC 跨块界面 `W'[4..7]` 解耦（`p15_btc_crossblock_wp4_7_msg` 与 `p15_btc_crossblock_wp4_7_bwd`）及 Block 1b `W[16..17]` 对 Nonce `W[3]` 的独立性（`btc_b1b_w16_w17_indep_nonce`），但这些结构仅限 Block 2 浅窗口（$t' \le 18$），在进入 Block 2 `t' >= 19` 完整非线性消息扩展及 Block 1b `t >= 18` 后全面热寂，无法将任何自由度跨过 128 轮压缩。

---

## 4. Lean 4.36.0 `bv_decide` 零 `sorry` 形式化证真清单

[lean_candidates.json](file:///g:/SHA-LM/v12_frontier/track_b2_p15_mitm_biclique/results/lean_candidates.json) 导出的全部 **11/11** 条候选（全部经 Z3 `VERIFIED`）均已在 [Stage1Calib.lean](file:///g:/SHA-LM/v12_frontier/track_b2_p15_mitm_biclique/lean_verify/P15Verify/Stage1Calib.lean) 与 [Stage2BTC.lean](file:///g:/SHA-LM/v12_frontier/track_b2_p15_mitm_biclique/lean_verify/P15Verify/Stage2BTC.lean) 中完成形式化定理编写与无 `sorry` 机器证真（连同 `Primitives.lean` 单步引理及 BTC Nonce `W[16..17]` 无关性定理，[VerifyAxioms.lean](file:///g:/SHA-LM/v12_frontier/track_b2_p15_mitm_biclique/lean_verify/VerifyAxioms.lean) 共审计 **32** 个定理）：

- **Stage 1A 减轮 MitM（4 组 / 8 定理）**：`p15_stage1a_mitm_20r_{fwd,bwd}`、`p15_stage1a_mitm_24r_{fwd,bwd}`、`p15_stage1a_mitm_28r_{fwd,bwd}`、`p15_stage1a_mitm_32r_{fwd,bwd}`；
- **Stage 1B 64 轮局域 Biclique（2 组 / 4 定理）**：`p15_stage1b_local_bic_6r_t0_{fwd,bwd}`、`p15_stage1b_local_bic_8r_t0_{fwd,bwd}`；
- **Stage 2 / BTC 固定填充与跨块非干涉（5 组 / 10 定理 + 1 扩展定理）**：`p15_s2b_block2_16r_mitm_{fwd,bwd}`、`p15_btc_crossblock_wp4_7_{msg,bwd}`、`p15_backsolve_4r_chunk_{fwd,bwd}`、`p15_btc_b2_r0_16_chunk_{fwd,bwd}`、`p15_btc_b1b_r0_16_chunk_{fwd,bwd}`、`btc_b1b_w16_w17_indep_nonce`。

### 可复现构建与公理审计命令
```powershell
$env:PATH = "$env:USERPROFILE\.elan\bin;$env:PATH"
Set-Location g:\SHA-LM\v12_frontier\track_b2_p15_mitm_biclique\lean_verify
lake build
lake env lean VerifyAxioms.lean
```
