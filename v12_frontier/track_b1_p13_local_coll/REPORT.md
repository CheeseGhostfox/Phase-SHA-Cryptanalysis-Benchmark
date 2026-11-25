# P13「AlphaZero 局部碰撞隧道 & 截断前导零定向差分」结案报告

> **任务代号**：Phase-SHA V12 · Track B1 · P13  
> **执笔与建模**：`@p13-math`（数学与实验契约 [SPEC.md](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/SPEC.md)）  
> **工程与计算**：`@p13-eng`（本地 CPU 门禁 + Colab `v12-p13` Tesla T4 纯 FP32 `allow_tf32=False`，总耗时 `1884.04s`）  
> **只读参考基准**：[sha256_ref.py](file:///g:/SHA-LM/src/sha256_ref.py)  
> **总结论裁决**：**`PARTIAL_POSITIVE`（阳性校准与 $R \le 16$ SAT 精确碰撞全线跑通并附带 50 组真迹见证；深轮 $R \ge 20$ 局部碰撞与 BTC 截断前导零全口径算力加速均严格证伪 `NULL`）**

---

## 一、核心裁决摘要与反幻觉结论

依据 [SPEC.md](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/SPEC.md) 第 5.2 节判定标准，P13 在严格执行五重反幻觉门禁（G1–G5）后给出如下客观裁决：

1. **B1-1 基础设施与阳性校准 100% 通过（Zero-Bug Pass）**：
   - 16 态 Mendel 广义位级带符号差分状态机通过 $1{,}000{,}000$ 组完备性自检（`soundness_violations = 0`）与 $100{,}000$ 组紧凑性自检（`exactness_mismatches = 0`），PySAT `CaDiCaL195` Tseitin 编码在完整 $R=64$ 轮 $10{,}000$ 组随机消息上与 [sha256_ref.py](file:///g:/SHA-LM/src/sha256_ref.py) 对拍 **`xor_mismatches = 0`**；
   - 成功搜出并落盘 **50 组**精确 SAT 见证消息对 $(M, M')$（[p13_witnesses.json](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/results/p13_witnesses.json)），经本地 CPU（`CUDA_VISIBLE_DEVICES=""`）独立调用 `sha256_ref.py` 逐比特验真 **50/50 全通过**。包含标准固定 `IV` 下 $R=16$ 满足 G2 非平凡约束的 **256-bit 精确碰撞**（`ladder_R16_standard`，`delta_H_hw = 0`, `t_min = 0`）。
2. **B1-2 赛道一（$R = 16 \to 32$ 局部碰撞阶梯）：在 $R \ge 20$ 撞壁，G3 揭示价值头前向流形伪信号**：
   - 在 $R=16$ 上 `free` / `semi_free` / `standard` 全部在 `0.16s~0.287s` 内求得精确碰撞；
   - 进入 $R \in \{20, 24, 28, 32\}$ 后，在 `sat_budget = 200,000` 约束下全部模式均 `sat = False`；其中 $R \ge 28$ 因消息扩展 $W[16..]$ 跨轮线性+非线性耦合回灌，单比特局部碰撞骨架直接在 Mendel 不动点传播阶段触发矛盾（`mendel_contradictions = 1`）。
   - **G3 反幻觉检验**：虽然 3D 晶体网络在训练集达到 `99.22%~100%` 分类精度且局部 `?` 孔洞策略定位在 $R=20..28$ 取得 `1.5x~3.75x` 的表观 `1-bit` 复原率（到 $R=32$ 坍缩为 `0.00781 = 0.25x`），但**价值头对合法前向轨迹与活跃位被破坏轨迹给出完全无区分的平均估值**（如 $R=20$: `0.9999 vs 0.9999`；$R=24$: `1.0 vs 1.0`），机制上复现了 V11 P11 Part 4 的“全局前向可达流形归属”伪信号。
3. **B1-2 赛道二（BTC 截断前导零定向差分隧道）：全口径摊还加速严格归零（`max speedup = 0.015935 << 1.0`）**：
   - **阳性反例（末轮作弊对照）**：`S1_trivial_last_round_R8_k12`（`t_min = 7`, `g2_nontrivial = False`）利用末轮直接注入取得 `p_hit = 0.0625 (256x)`, `z = 63.758`, `speedup_amortized = 2.396761`，验证了评分与计费管线对已知退化通道的高灵敏度；
   - **G2 非平凡组与 BTC 真实边界（`S1_g2_nontrivial`、`S2a_btc_w0_3`、`S2b_btc_nonce_only`、`S3_double_sha256`）**：虽然 SAT 能在 $R \le 24, k \le 12$（及 Double-SHA256 $R_1=R_2 \le 20, k \le 8$）算出满足顶 $k$ 位前导零的锚点消息对 $(M, M')$，但在“找到 1 个合法前导零解后沿暗道批量衍生 $B=256$ 个衍生候选”测试中：
     - 除浅轮 `S1_g2_nontrivial` 在 $k=8$（`R=12: p_hit=0.039062, z=9.018`；`R=16: p_hit=0.027344, z=6.012`）因固定 $W[4..15]$ 产生残余局部差分命中外，所有 `S2a`、`S2b` 与 `S3`（共 40 组配置）及 $R \ge 20$ `S1` 均落入随机涨落区间（$z \in [-3.098, +4.131] < 5.0$）；
     - 在按 `223,264.1 evals/sec` 实测吞吐如实计入搜索代价 $C_{\text{search}}$ 后，**全部 52 组非末轮作弊配置（含其中 38 组 SAT 锚点可解与 31 组严格满足 G2 跨度 `g2_nontrivial=True` 组）中，有效摊还加速比 `speedup_amortized > 1.0` 命中数为 `0/52`**（全局最大仅 `0.015935`，出现于 `S2a_btc_w0_3_R12_k4`；$R \ge 20$ 上 `S2a`/`S2b`/`S3` 最大仅 `0.001395`），与 `G4_CTRL_S2a_R16` 同构随机对照组（`speedup <= 0.000638`）行为一致。

---

## 二、B1-1 门禁实测：状态机百万自检与 SAT 精确对拍

全部自检脚本已在本地 CPU（`CUDA_VISIBLE_DEVICES=""`）严格通过：

| 检查项目 | 数据文件 | 样本规模 / 轮数 | 核心指标 | 状态 |
| :--- | :--- | :--- | :--- | :---: |
| Mendel 16 态完备性校验 | [mendel_selftest.json](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/results/mendel_selftest.json) | $1{,}000{,}000$ 随机约束 | `soundness_violations = 0`（`61.115s`） | PASS |
| Mendel 算子紧凑性校验 | [mendel_selftest.json](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/results/mendel_selftest.json) | $100{,}000$ 部分约束 | `exactness_mismatches = 0` | PASS |
| PySAT `CaDiCaL195` 对拍 | [sat_parity_10k.json](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/results/sat_parity_10k.json) | $10{,}000$ 组 $R=64$ | `xor_mismatches = 0`（`111.141s`） | PASS |
| 见证消息对独立 CPU 验真 | [p13_witnesses.json](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/results/p13_witnesses.json) | 50 组 SAT 见证对 | `50/50 sha256_ref_verified = True` | PASS |
| 云端实例停机释放核验 | [run_log.txt](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/results/run_log.txt) | 会话 `v12-p13`（Tesla T4） | `colab.exe sessions` 无残留实例 | PASS |

---

## 三、B1-2 赛道一：标准阶梯局部碰撞（$R=12 \to 32$）与 G3 检验

### 3.1 阳性校准与标准阶梯实测全表

数据溯源：[p13_ladder_results.json](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/results/p13_ladder_results.json)。

| 配置标签 | 轮数 $R$ | IV 模式 | SAT 可解 | 耗时 (s) | $t_{\min}$ | G2 合规 | 随机稀疏基线 $\min \mathrm{HW}(\Delta H)$ | 实测 $\mathrm{HW}(\Delta H)$ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `calib_R12_sfs` | 12 | `semi_free` | **True** | 0.2262 | 3 | False | — | **0** |
| `calib_R12_fs` | 12 | `free` | **True** | 0.1164 | 2 | False | — | **0** |
| `calib_R16_sfs` | 16 | `semi_free` | **True** | 0.1480 | 2 | **True** | — | **0** |
| `calib_R16_fs` | 16 | `free` | **True** | 0.1534 | 2 | False | — | **0** |
| `ladder_R16_free` | 16 | `free` | **True** | 0.160 | 0 | **True** | — | **0** |
| `ladder_R16_semi_free` | 16 | `semi_free` | **True** | 0.281 | 0 | **True** | — | **0** |
| `ladder_R16_standard` | 16 | `standard` | **True** | 0.287 | 0 | **True** | 96 | **0** |
| `ladder_R20_{free,sfs,std}` | 20 | 三模式全测 | False | 42.80~46.95 | — | — | 99 (std) | — |
| `ladder_R24_{free,sfs,std}` | 24 | 三模式全测 | False | 52.75~53.89 | — | — | 95 (std) | — |
| `ladder_R28_{free,sfs,std}` | 28 | 三模式全测 | False | 0.30~0.52 | — | — | 99 (std) | — |
| `ladder_R32_{free,sfs,std}` | 32 | 三模式全测 | False | 0.34~0.35 | — | — | 95 (std) | — |

**物理机制剖析**：
1. **$R \le 16$ 的可解边界**：由于 $R=16$ 尚不触发消息扩展递推 $W[16..]$，16 个消息字彼此完全独立自由，SAT 求解器能在 0.3 秒内利用 `W[0..8]` 的多步局部对消并保持 `W[9..15]` 零差，从而在标准固定 `IV` 下直接构造出 256-bit 精确碰撞（如 `ladder_R16_standard`）。
2. **$R \ge 20$ 与 $R \ge 28$ 的双重阻断**：在 $R=20, 24$ 上，$W[16..R-1]$ 引入 4~8 步跨轮 $\sigma_0/\sigma_1$ 耦合后，`sat_calls=5` 均在预算内 `UNSAT`；至 $R=28, 32$ 时，单点局部碰撞骨架（`step_t=0, bit_j=0`）由于在 $t=16..31$ 强制将扩散后的活跃差分压回零，**在 Mendel 传播层即触发不可满足矛盾（`mendel_contradictions=1`）**。

### 3.2 门禁 G3：3D 晶体策略-价值网络活跃字 1-bit 贪心复原检验

数据溯源：[p13_ladder_results.json](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/results/p13_ladder_results.json) `g3_evaluations`（每轮 `n_probes=128`，纯 FP32 `allow_tf32=False`）：

| 轮数 $R$ | 训练价值精度 `train_val_acc` | `greedy_1bit_recover_rate` | `top3_recover_rate` | 随机基线 | 倍率 (`ratio_vs_random`) | `mean_valid_value` vs `mean_perturbed_value` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **16** | 0.9922 | 0.03125 | 0.06250 | 0.03125 | 1.00× | `0.9914` vs `0.9914` |
| **20** | 1.0000 | 0.11719 | 0.32812 | 0.03125 | 3.75× | `0.9999` vs `0.9999` |
| **24** | 1.0000 | 0.04688 | 0.17969 | 0.03125 | 1.50× | `1.0000` vs `1.0000` |
| **28** | 1.0000 | 0.07031 | 0.15625 | 0.03125 | 2.25× | `0.9132` vs `0.9135` |
| **32** | 1.0000 | 0.00781 | 0.03125 | 0.03125 | **0.25×** | `0.9993` vs `0.9993` |

**反幻觉诊断**：表格清晰暴露了一个关键结构性事实——价值头在全轮次上的 `mean_valid_value` 与 `mean_perturbed_value` **小数点后四位几乎完全重合**（`0.9999 vs 0.9999`、`1.0 vs 1.0`），说明神经网络价值头并未学到“某一条差分路径距离合法碰撞有多远”的逆向能量梯度，而仅仅区分了“整张 3D 晶格是否看起来像一条前向演化轨迹”（与 V11 P11 Part 4 同一机制）。策略头在 $R=20..28$ 之所以出现 `1.5×~3.75×` 的 Top-1 定位，仅源于局部 3D 卷积识别出“一片已确定 `(0,1,u,n)` 邻域中突然缺失的单个 `?` 空洞”；到 $R=32$ 时连局部空洞判别也完全坍缩至 `0.25×`。

---

## 四、B1-2 赛道二：BTC 截断前导零定向差分隧道实验

参考吞吐率实测：`223,264.1 evals/sec`（[p13_btc_truncated_lz.json](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/results/p13_btc_truncated_lz.json)）。每组在求得 SAT 锚点对后，沿活跃通道随机扰动衍生 $B = 256$ 个不重复候选并按 SPEC 第 3.4 节全口径公式计入搜索等效代价 $C_{\text{search}}$。

### 4.1 四大场景核心对比表（含末轮作弊对照与 G4 随机对照）

| 场景分类 | 代表配置 / 覆盖范围 | SAT 可解数 | 显著偏移数 ($z \ge 5.0$) | 最高 $p_{\text{hit}}$ / 最高 $z$ | 最高全口径摊还加速比 | `effective_speedup > 1` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **阳性反例（末轮作弊）** | `S1_trivial_last_round` (`R=8, k=4,8,12`, `t_min=7`) | 3 / 3 | 3 / 3 | `0.0625` (`k=12`, **256×**) / `z=63.758` | **`2.396761`** (`k=12`) | **1 / 3（G2 否决）** |
| **S1 单块合规** | `S1_g2_nontrivial` (`R=12..24, k=4,8,12`) | 7 / 12 | 2 / 12 (`R=12,16 k=8`) | `0.039062` (`R12 k8`) / `z=9.018` | `0.015698` (`R12 k8`) | **0 / 12** |
| **S2a BTC Header** | `S2a_btc_w0_3` (`R=12..24, k=4..16`, 固定填充) | 12 / 16 | **0 / 16** | `0.125` (`R16 k4`) / `z=4.131` | `0.015935` (`R12 k4`) | **0 / 16** |
| **S2b BTC 仅 Nonce** | `S2b_btc_nonce_only` (`R=8..20, k=4..16`, 仅 `W[3]`) | 11 / 16 | **0 / 16** | `0.074219` (`R20 k4`) / `z=3.750` | `0.009492` (`R12 k4`) | **0 / 16** |
| **S3 Double-SHA256** | `S3_double_sha256` (`8x8..20x20, k=4,8`) | 8 / 8 | **0 / 8** | `0.066406` (`R20x20 k4`) / `z=0.258` | `0.000628` (`R20x20 k4`) | **0 / 8** |
| **G4 随机密码对照** | `G4_CTRL_S2a_R16` (`k=4, 8`) | 2 / 2 | **0 / 2** | `0.042969` (`k=4`) / `z=-1.002` | `0.000638` (`k=4`) | **0 / 2** |

### 4.2 关键机制结论（为何放松 $256-k$ 位仍无法构建深轮批量暗道）

1. **末轮通道 vs 非平凡多轮非线性扩散**：`S1_trivial_last_round_R8_k12` 之所以能达到 $256\times$ 边际命中与 $2.397\times$ 摊还加速，完全依赖在末轮 $t=7$ 修改 $W[7]$ 低比特直接控制 $A_8 = T_{1,7} + T_{2,7}$ 的最高 12 位进位（`t_min = 7`，零非线性轮隔离）。一旦通过 G2 禁止末轮直接扰动，即便仅要求 $H[0]$ 顶 $k$ 位差分为 0、彻底释放其余 $256-k$ 位（条件熵从 $256$ bits 降至 $4 \sim 16$ bits），经过 $R \ge 12$ 轮含 $A_{t+1}/E_{t+1}$ 跨轨回灌和 `ROTR` 环面混合后，**单个活跃消息字位翻转引起的差分雪崩仍然均匀覆盖 $H[0]$ 的最高 $k$ 位**。
2. **锚点搜索与中性暗道的鸿沟**：SAT 求解器确实能利用截断 $k \le 12$ 的松弛约束在数十秒内“找到一组满足 $\mathrm{clz}(H(M)[0]) \ge k$ 且 $\mathrm{clz}(H(M')[0]) \ge k$ 的孤立消息对 $(M, M')$”（例如 `S2a_btc_w0_3_R24_k12` 与 `S3_double_R20x20_k8`）；但这组解是针对特定进位链耦合高度特化的孤立点，而非一段可批量翻转其他比特而不破坏顶 $k$ 位前导零的**中性暗道（Neutral Tunnel）**——在衍生阶段一旦扰动其他比特，命中率立即塌回随机期望 $2^{-k}$，叠加搜索本身消耗的 $10^5 \sim 2.18 \times 10^7$ 次 SHA-256 等效算力，真实摊还加速比严格低于 $0.016 \ll 1.0$。

---

## 五、精选 SAT 精确见证消息对摘录（全部 50 组见 `p13_witnesses.json`）

以下见证均已在本地 CPU 经 `src/sha256_ref.py` 逐比特验真（完整 50 组十六进制记录存于 [p13_witnesses.json](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/results/p13_witnesses.json)）：

1. **`ladder_R16_standard`（标准固定 `IV`，$R=16$，256-bit 精确碰撞，`delta_H_hw = 0`, `t_min = 0`, `g2_nontrivial = True`）**：
   - `M` : `ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff`
   - `M'`: `fffffffe bbe8047f 145d1a6d 7397fb7c 965ffe7b 0d980374 00000001 00000005 00000000 ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff`
   - `H = H'`: `f20380b7 c7dc7b03 e386f61e 9785e654 209d9723 8d179d2a 6b53c975 681593ec`
2. **`calib_R16_sfs`（Semi-Free-Start，$R=16$，局域窗 `W[2, 5..12]` 256-bit 碰撞，`delta_H_hw = 0`, `g2_nontrivial = True`）**：
   - `IV = IV'`: `ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff`
   - `M` : `ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff`
   - `M'`: `ffffffff ffffffff fffffffe ffffffff ffffffff ae6aa460 2c617e47 6fb152b3 c0c9964f 8a966d23 6fdefd3c 2bd1143f 0728d833 ffffffff ffffffff ffffffff`
   - `H = H'`: `e4c76537 92747d32 e7be87e3 18c308c6 ab500384 def06975 bde6b5bc 788f4c6e`
3. **`S2a_btc_w0_3_R24_k12`（BTC Block 2 固定填充，$R=24$，顶 12 位前导零成对见证，`clz = (12, 12)`, `g2_nontrivial = True`）**：
   - `M` : `958ea5e6 08800002 00000480 00000000 80000000 00000000 ... 00000280`
   - `M'`: `922a9c51 2165b9c0 ee24724f bad21bc8 80000000 00000000 ... 00000280`
   - `H` : `000cdbcf b7d16168 ed894803 37358945 252b4cbc e427b8ee 209e0882 b22722c7`
   - `H'`: `000e5da9 f844816c 343069ac ed68bba1 635dac91 935c87e4 af1c1a5d b0f0d176`
4. **`S3_double_R20x20_k8`（BTC Double-SHA256 双块，$R_1=20, R_2=20$，顶 8 位前导零成对见证，`clz = (10, 10)`, `g2_nontrivial = True`）**：
   - `M` : `ffffffff ffffffff ffffffff 36bfffff 80000000 00000000 ... 00000280`
   - `M'`: `ffffffff ffffffff ffffffff 2e7fffff 80000000 00000000 ... 00000280`
   - `H` : `00386b62 040455f5 28d74ce3 6bcff05c baca1e03 c51b542c 19199dcd 9ba9f282`
   - `H'`: `00266b08 033d5a19 d7429ce5 1850f481 164cce3a b002c7bf 2cfef7d5 f12b2517`

---

## 六、资产索引与可复现说明

- **设计契约**：[SPEC.md](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/SPEC.md)
- **状态机与代码**：[mendel_state_machine.py](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/mendel_state_machine.py)、[sat_witness_solver.py](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/sat_witness_solver.py)、[alphazero_crystal_mcts.py](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/alphazero_crystal_mcts.py)、[run_p13_experiments.py](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/run_p13_experiments.py)
- **实测结果与日志**：[mendel_selftest.json](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/results/mendel_selftest.json)、[sat_parity_10k.json](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/results/sat_parity_10k.json)、[p13_ladder_results.json](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/results/p13_ladder_results.json)、[p13_btc_truncated_lz.json](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/results/p13_btc_truncated_lz.json)、[p13_witnesses.json](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/results/p13_witnesses.json)、[run_log.txt](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/results/run_log.txt)
- **结案裁决文件**：[verdict.json](file:///g:/SHA-LM/v12_frontier/track_b1_p13_local_coll/verdict.json)
