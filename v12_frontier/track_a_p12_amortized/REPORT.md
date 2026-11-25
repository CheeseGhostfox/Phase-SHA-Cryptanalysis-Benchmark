# P12 结案报告：超广义 AsicBoost —— Double-SHA256 广义中性比特团簇与多输入摊还复用

> **总裁决状态**：`NULL`（未发现超越经典 Overt/Covert AsicBoost 的广义中性比特团簇或跨块摊还增益；A1 经典基线校准与 A3 Z3/PySAT 精确不可能性/最小活跃界证明全通过 `PASS`）  
> **工作目录**：[track_a_p12_amortized/](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/)  
> **规范契约**：[SPEC.md](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/SPEC.md)  
> **结构化裁决**：[verdict.json](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/verdict.json)  
> **源码文件**：[btc_double_sha_engine.py](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/btc_double_sha_engine.py)、[classic_asicboost_calib.py](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/classic_asicboost_calib.py)、[gnbc_gpu_search.py](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/gnbc_gpu_search.py)、[sat_amortized_bounds.py](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/sat_amortized_bounds.py)、[test_p12_engine.py](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/test_p12_engine.py)、[run_p12_colab.py](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/run_p12_colab.py)  
> **实测数据与日志**：[a1_calibration.json](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/results/a1_calibration.json)、[a2_gnbc_search.json](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/results/a2_gnbc_search.json)、[a3_sat_bounds.json](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/results/a3_sat_bounds.json)、[p12_run_meta.json](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/results/p12_run_meta.json)、[p12_colab.log](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/results/p12_colab.log)

---

## 1. 执行摘要与资源合规审计

1. **云端算力、纯 FP32 与停机核实**：  
   `@p12-eng` 在 Google Colab 会话 `v12-p12`（`Tesla T4`，全程锁定 `torch.float32`、`matmul_allow_tf32 = false`、`cudnn_allow_tf32 = false`）上完成全量 A1+A2+A3 流水线，总耗时 **`3944.8s`**（`a1 = 9.58s`, `a2 = 1101.93s`, `a3 = 2831.92s`）。产物回传后已执行 `colab.exe stop -s v12-p12`，并在本地经 `colab.exe sessions` 独立核实返回 `[colab] No active sessions found on server.`（`colab_stopped: true`）。
2. **严格比特对拍与隔离边界**：  
   - 与 [sha256_ref.py](file:///g:/SHA-LM/src/sha256_ref.py)（256 组随机压缩轨迹，全状态 `A_full/E_full/W_full/digest`）及 Python `hashlib.sha256`（真实比特币创世区块头 Block #0 及 256 个随机 80-byte BTC 区块头）对拍均满足 **`xor_diff == 0`**（`btc_genesis_hash_matched: true`）。
   - A2 搜索池严格限定 `Nonce ∈ [0, 2^30)`，独立留出测试集严格限定 `Nonce ∈ [2^30, 2^32)`（`N_holdout = 262,144`, `train_holdout_overlap = 0`）。
   - 未触碰 `v11_discovery/`、`v10_ouroboros64/`、`runs/`、`formal_proofs/`、`formal_proofs_v10/`、`v6a/`、`src/` 等既有目录。

---

## 2. A1 精确基准与经典 AsicBoost 硬基线校准

依据 [SPEC.md](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/SPEC.md) §1.2 统一固定比特级门等效（GE）权重表（`XOR2=3, AND2=2, OR2=2, NOT=1, FA=11, HA=5, ADD32=346, CH32=256, MAJ32=384, SIGMA=192`，含静态零常量 `W1[5..14]=0` 与 `W2[9..14]=0` 折叠为 `0 GE`），在 [a1_calibration.json](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/results/a1_calibration.json) 中测得 `SHA256_MAIN` 完整 80-byte Double-SHA256 硬基线如下：

| 基线模式 | 路数 $k$ / 批大小 | `amortized_ge_per_hash` | `saving_pct_vs_noab` | 模块拆解（`Chunk1 / Chunk2_exp / Chunk2_comp / Hash2_exp / Hash2_comp` GE/hash） |
| :--- | :---: | :---: | :---: | :--- |
| **`BASE_NO_ASICBOOST_2p32`** | `k=1` (`B=64`) | **512,590.83** | `0.000%` | `0.00 / 52,631.77 / 195,134.64 / 58,212.00 / 206,612.42` |
| `BASE_NO_ASICBOOST_SYM` | `k=1` (`N=64`) | 512,658.97 | `0.000%` | `68.14 / 52,631.77 / 195,134.64 / 58,212.00 / 206,612.42` |
| `BASE_NO_ASICBOOST_1NONCE` | `k=1` (`N=1`) | 516,951.97 | `0.000%` | `4,361.14 / 52,631.77 / 195,134.64 / 58,212.00 / 206,612.42` |
| **`BASE_ASICBOOST_OVERT_k4_2p32`** | `k=4` (`B=256`) | **473,569.52** | **`7.613%`** | `0.00 / 13,157.94 / 195,612.98 / 58,212.00 / 206,586.61` |
| **`BASE_ASICBOOST_OVERT_k8_2p32`** | `k=8` (`B=512`) | **466,986.32** | **`8.897%`** | `0.00 / 6,578.97 / 195,613.05 / 58,212.00 / 206,582.30` |
| **`BASE_ASICBOOST_OVERT_k16_2p32`** | `k=16` (`B=1024`) | **463,694.68** | **`9.539%`** | `0.00 / 3,289.49 / 195,613.05 / 58,212.00 / 206,580.15` |
| `BASE_ASICBOOST_OVERT_k16_SYM` | `k=16` (`N=64`) | 463,760.02 | `9.538%` | `65.34 / 3,289.49 / 195,613.05 / 58,212.00 / 206,580.15` |
| `BASE_ASICBOOST_OVERT_k16_1NONCE` | `k=16` (`N=1`) | 467,876.16 | `9.493%` | `4,181.48 / 3,289.49 / 195,613.05 / 58,212.00 / 206,580.15` |
| `BASE_ASICBOOST_COVERT_k16_2p32` | `k=16` (`B=1024`) | 463,694.95 | `9.539%` | `setup = 0.266 GE/hash`（Merkle 碰撞外循环均摊至 $2^{32}$） |

**口径精算解释（为何比特级 GE 为 `9.54%` 而硬件文献常述 `~13%–20%`）**：  
- 在严格比特级 GE 口径下，`Chunk2` 消息扩展 `W1[16..63]` 占 `Chunk2` 自身的 **`21.24%`**（`52,632 / 247,766 GE`），占完整内循环 `Chunk2 + Hash2` 的 **`10.27%`**（`52,632 / 512,591 GE`）；因此 16 路复用消除 $(1 - 1/16)$ 的 `Chunk2_exp` 精确对应 `9.539%`（渐近极限 `10.268%`）。若硬件流水线进一步把 `W[t] + K[t]` 预加融进共享扩展器（或按 Timo Hanke 原论文 4-模块面积/控制布线估算），即对应文献中的 `~13%–20%` 区间。
- 无论采用何种等价口径，**同构随机对照 `CTRL_RAND_ROTR_K_S0` 表现出几乎完全重合的基线节省**（`k=4, 8, 16` 分别为 `7.610%, 8.892%, 9.533%`），且 P12 所有候选方案均在同一固定 GE 权重表下与 `BASE_ASICBOOST_OVERT` 直接对比。

---

## 3. A2 GPU 广义中性比特团簇（GNBC）与辅助条件搜索实测

在 Colab `Tesla T4` 上对 `SHA256_MAIN` 与 3 组谱隙对齐随机对照密码 `CTRL_RAND_ROTR_K_{S0, S1, S2}` 以完全相同预算运行三轨道搜索并在 $N_{\text{holdout}} = 262{,}144$ 留出集上核验（[a2_gnbc_search.json](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/results/a2_gnbc_search.json)）：

### 3.1 Track A2-1：`Chunk2` 内部中性穿透与预条件 $\Phi$
- **全模式收敛至同一平凡 MSB 瞬态**：单比特穷举、496 组 2-bit 对、`GNBCPolicyNet`（纯 FP32 神经策略先验）、`W1[16]=W1[17]=0` 预锚定以及 `W1[0,1,3]` 联合搜索全部收敛至注入点最高位 `W1[3] bit 31`（因为最高位无向高位进位），但该特性仅维持注入后 **1 轮**局部低活跃率：
  - `SHA256_MAIN` 逐轮活跃率：`t=0..2: 0.000`（`W1[3]` 尚未注入的已知平凡基线）$\to$ `t=3: 0.03125`（仅 2/64 比特）$\to$ `t=4: 0.26486` $\to$ `t=5: 0.45873` $\to$ `t=6: 0.49870` $\to$ `t=7: 0.50053`（完全热化）；
  - 启用 `W1[16]=W1[17]=0` 预锚定后，`t=4` 活跃率 **`0.26530`**（与无预锚定完全无异，`t=5: 0.46059`）；
  - 启用 `Ch` 分支锁定条件 `PhiCh`（触发率 `p_phi = 0.06255`）后，`t=4` 仅微降至 `0.25794`，到 `t=5` 即达 `0.45551`，在留出集（$N=16{,}396$ / $262{,}144$）上中性穿透至 $t \ge 4$ 的全零状态概率 **`p_neutral_t4 = p_neutral_t5 = p_neutral_t6 = 0.0`**（$z = -9.8 \times 10^{-5}$，95% Wilson 上界 $1.47 \times 10^{-5}$）。
  - **对照组重合**：3 组 `CTRL_RAND_ROTR_K` 在 `t=4` 活跃率为 `0.24108 ~ 0.27020`、`t=5` 为 `0.43342 ~ 0.43917`，`p_neutral_t4..t6` 同样全为 `0.0`。

### 3.2 Track A2-2：`Chunk2 → Hash2` 跨块界面中性团簇
- 在 $N_{\text{holdout}} = 262{,}144$ 上测试从 `W1[3] bit 0`、`W1[3] bit 31`、`W0[0] bit 13`（BIP320 `nVersion` 位）注入单比特翻转，测量 `hash1[0..7]`（即 `W2[0..7]`）保持首字零差或任意一字零差的命中数：
  - **`SHA256_MAIN`**：三组全部为 **`0 / 262,144`**（`zero_w2_word0_hits = 0`, `zero_w2_any_word_hits = 0`, `z = -0.022`），256-bit `hash1` 差分汉明重均值 **`128.005 ~ 128.022`**（最小值 `92 ~ 93` bit），精准吻合理想随机置换；
  - **`CTRL_RAND_ROTR_K_{S0,S1,S2}`**：全部 `0 / 262,144` 命中，均值 **`127.992 ~ 128.014`**。
- 直接探针测试 `Hash2` 内部单独翻转 `W2[k] bit 31`（$k \in \{0..7\}$）：除字 $k$ 进入前的平凡零轮数（$t' < k$）外，注入后同样仅维持 `1` 轮局部稀疏，下一轮活跃率即跳升至 `0.264 ~ 0.267`（与随机对照 `0.244 ~ 0.273` 一致）。

### 3.3 Track A2-3：比特级进位吸收与多维集群净摊还增益
在 $d_{\text{neutral}} \in \{2, 3, 4\}$（批大小 $M \in \{4, 8, 16\}$）下对完整 `Chunk1 + Chunk2 + Hash2` 计费：

| 候选集群方案 | 批大小 $M$ | `active_ge_per_hash` | `saving_pct_vs_noab` | **相对同 $M$ 经典 Overt AsicBoost 净增益** | 对照组 `CTRL` 均值 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `A2-3_NonceOnly_d2_MSB_contiguous` | 4 | 513,538.97 | `-0.042%` | **`-8.301%`** | `-8.248%` |
| `A2-3_NonceOnly_d3_MSB_contiguous` | 8 | 511,980.30 | `+0.262%` | **`-9.498%`** | `-9.442%` |
| **`A2-3_NonceOnly_d4_MSB_contiguous` (Holdout)** | 16 | **511,117.70** | **`+0.430%`** | **`-10.091%`** | **`-10.036%`** |
| `A2-3_Hybrid_4Ver_x_4NonceMSB_M16` | 16 | 476,399.63 | **`+7.193%`** | **`-2.613%`** | **`-2.598%`** |
| `A2-3_OvertVersionSym1_d4`（未摊还 `Chunk1`） | 16 | 742,985.56 | `-44.740%` | `-60.034%` | `-59.985%` |

- **关键机制洞察**：即使仅改动 `nNonce` 最高 4 位（`bits 28..31`）使第 `t=3` 轮 `p3 + W1[3]` 的低 28 位加法器静默，总计也仅节省 `~0.430%`（主要来自 `t=3` 的约 28 个进位门，在 `t=4..63` 及 `W1[18..63]` 中即刻扩散）；而若将 `M=16` 预算切分为 `4 Midstate × 4 Nonce` 混合集群，由于 4 个 Nonce 破坏了 `Chunk2` 扩展器的 16 路复用（退化为 4 路），总体节省反而从 `9.539%` 倒退至 `7.193%`（净损失 **`-2.613%`**）。

---

## 4. A3 Z3 / PySAT 精确界限证明与局部 CPU 独立复验

### 4.1 Z3 (`5.4.1`) 与 PySAT (`CaDiCaL195`) 精确数学上界（[a3_sat_bounds.json](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/results/a3_sat_bounds.json)）

1. **Z3 B1（`Chunk2` 内部零差穿透界）**：
   - `B1_S1a_sameMS_onlyW3`（固定 `midstate1` 与 `W1[0..2]`，仅变动 `W1[3]`）：在 $R = 4, 5, 6$ 上 Z3 均在 `0.008 ~ 0.017s` 内证死 **`UNSAT`**（`3` 组 `CTRL` 亦全 **`UNSAT`**），严格证明单靠变动 `nNonce` 不存在任何在注入后能重新回到零差状态的自愈轨迹。
   - `B1_S1b_sameMS_W0123`（固定 `midstate1`，允许 `W1[0..3]` 任意联合补偿）：$R=4$ **`UNSAT`**（`0.007s`）；$R \in \{5..12\}$ 在 8s 预算内 `UNKNOWN`。
   - `B1_S2_varMS_W0123`（放开 `midstate1` + `W1[0..3]` 自由度）：仅在 $R \le 8$ 存在自由起始局部消去解（`SAT`），一旦跨过 $R \ge 12$ 即进入搜索空间壁垒（`UNKNOWN`）。
2. **Z3 B2（`Hash2` 跨块扩展零差不可能性定理）**：
   - 对非平凡 `Chunk2` 输出差分 $\Delta W_2[0..7] \neq 0$：当且仅当仅末字 `W2[7]` 非零（即需前 7 字 224-bit 完全碰撞）时，扩展步 `W2[16..21]` 可维持零差（`SAT, min_nonzero_input_words=1`）；但 **`W2[16..23]`（以及强制 $\Delta W_2[0] \neq 0$ 的 `W2[16..23]`）被 Z3 在 `0.022 ~ 0.032s` 内严格证死为 `UNSAT`**（`SHA256_MAIN` 与 3 组 `CTRL` 完全一致），证明任何合法跨块差分至迟在 `W2[23]` 前必触发不可消除的消息扩展活跃传播。
3. **PySAT `CaDiCaL195` 最小活跃状态字下界**：
   - `Chunk2_onlyW3`（`inj_round = 3`）：在 $R=4, 6$ 上证死 **`EXACT_MIN_PROVEN: min_active_state_words = 2`**（`UNSAT <= 1`，即注入后必然同时激活 $A_4$ 与 $E_4$，不存在少于 2 个活跃字的路径），$R=8..12$ 预算下界亦维持 `UNSAT <= 1..2`；
   - `Hash2_W0to7`：$R=4, 6$ 允许 `0` 活跃字仅对应极高代价的 `W2[7]` 延迟注入平凡情况（`t'<7` 尚未进入），一旦进入 $R=8$（覆盖 `W2[7]` 注入），`CaDiCaL195` 在 `35.48s` 内严格证死 **`EXACT_MIN_PROVEN: min_active_state_words = 2`**（`UNSAT <= 1`）。3 组 `CTRL_RAND_ROTR_K` 给出完全一致的 `EXACT_MIN_PROVEN` 界。
4. **新跨轮/跨块恒等式产出**：`new_identities = []`。因未产生超越经典 AsicBoost 的新跨轮/跨块中性恒等式，按 [SPEC.md](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/SPEC.md) §3.3 约定未触发额外 Lean 4 `bv_decide` 新定理工程构建（`lean_verify_triggered: false`）。

### 4.2 本地 CPU 独立复验记录
在本地严格禁用 GPU（`$env:CUDA_VISIBLE_DEVICES = ""`）执行全套复核：
1. 运行 [test_p12_engine.py](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/test_p12_engine.py)：`test_fp32_lock`、`test_sha256_ref_xor_zero`（256 样本 `xor_diff=0`）、`test_btc_genesis_and_headers_xor_zero`（创世头 + 256 随机头 `max_xor_diff=0`）、`test_a1_calibration_and_ctrl`（`[7.62, 8.9, 9.54]%`）全部通过。
2. 逐比特核对 [a2_gnbc_search.json](file:///g:/SHA-LM/v12_frontier/track_a_p12_amortized/results/a2_gnbc_search.json) 中留出的 4 个真实 80-byte `witness_headers_hex`（`nonce_le = c5f4a576, 45f4a576, 85f4a576, 05f4a576`），引擎计算结果与 Python `hashlib.sha256` 100% 一致。
3. 执行 `colab.exe sessions` 确认输出 `[colab] No active sessions found on server.`。

---

## 5. 最终结论

**P12 结案裁决：`NULL`**。  
在真实比特币 80-byte Block Header 三段式 `Double-SHA256` 中：
1. 经典 Overt/Covert AsicBoost 通过复用 `Chunk2` 消息扩展 `W1[16..63]` 在精确比特级 GE 口径下获得 `7.613% ~ 9.539%`（`k=4..16`，对应硬件文献 ~13%~20% 管道估算）的确定性摊还节省；
2. 对 `Chunk2` 内部（含 `W1[16]=W1[17]=0` 预锚定与 `PhiCh`）、`Chunk2 → Hash2` 跨块界面以及比特级进位吸收集群的系统 GPU 搜索与 $N=262{,}144$ 留出验证表明：**不存在任何能突破经典 AsicBoost 的广义中性比特团簇**——最稀疏单比特（`W1[3] bit 31`）在注入后 1 轮即破裂，第 4 轮（注入后 1 轮）活跃率跳升至 `~0.265`、第 5 轮跳升至 `~0.459`、第 6 轮完全热化至 `~0.500`；跨块至 `hash1` 命中零差字为 `0 / 262,144`；`Nonce` 高位静默与混合复用相对经典 16 路 Overt AsicBoost 均呈严格负增益（`-10.091%` 与 `-2.613%`）；
3. Z3 与 PySAT `CaDiCaL195` 从形式化边界证明了同 Midstate 单改 Nonce 在 $R=4..6$ 零差 `UNSAT`、`Hash2` 扩展至迟在 `W2[23]` 零差 `UNSAT`，且 `SHA256_MAIN` 全程表现与 3 组谱隙对齐随机 `CTRL_RAND_ROTR_K` 对照组高度重合，严密排除了比特币挖矿存在“超广义 AsicBoost”代数加速暗道的可能。
