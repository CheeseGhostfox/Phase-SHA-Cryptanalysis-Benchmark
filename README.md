# Phase-SHA: A 14-Paradigm Empirical & Formal Cryptanalysis Benchmark for SHA-256 and Bitcoin Double-SHA256

[![Lean 4](https://img.shields.io/badge/Lean%204.36.0-483%20Theorems%20%7C%200%20sorry-brightgreen)](#四lean-4-形式化证明资产总览32-文件--483-定理引理--0-sorry)
[![Paradigms](https://img.shields.io/badge/Paradigms%20Audited-14%20(P1--P15)-blue)](#2-14-paradigm-benchmark-summary-v11--v12)
[![Verification](https://img.shields.io/badge/CPU%20Verify-50%2F50%20SAT%20%2B%2016%2F16%20BTC-success)](REPRODUCE.md)
[![Precision](https://img.shields.io/badge/GPU%20Precision-Pure%20IEEE--754%20FP32-informational)](#3-four-core-scientific-findings)
[![License](https://img.shields.io/badge/License-MIT%20%2F%20CC%20BY%204.0-orange)](LICENSE)

> **International Executive Summary (English)**  
> **Phase-SHA (`V1–V12`)** is a fully reproducible, anti-illusion-audited empirical and formal cryptanalysis benchmark evaluating **14 distinct algorithmic, statistical-physics, neural, hardware-amortization, SAT, and Meet-in-the-Middle paradigms (`V11 P1–P11` + `V12 P12, P13, P15`)** on reduced-round and full SHA-256 and Bitcoin Double-SHA256.  
> **What we rigorously establish (without overclaiming $\text{P} \neq \text{NP}$):**  
> 1. **Clean Empirical Falsification Across 14 Paradigms**: Under strict pure-FP32 controls and spectrum-aligned isomorphic random ciphers, **zero** neural, homotopy, Koopman, tensor-network disentangler, generalized neutral-bit, or SAT-tunnel methods beat brute-force expectations past effective compression depth $R_{\text{eff}} = 3\text{–}4$ (thermalization wall at $R = 6\text{–}8$).  
> 2. **Exact Positive Calibrations & 483 Zero-`sorry` Lean 4 Declarations**: Across **32 `.lean` files** (Lean `4.36.0`, zero Mathlib), we verify **483 theorem/lemma declarations (`338` public + `145` private/protected) with 0 `sorry`**, including 28 Z3-audited non-interference/cross-block identities (`P3`) forming a 23-round intermediate-state corridor, Stage 1A `16–32r` 4-color MitM & `64r L=6/8` Biclique bounds (`P15`), and **50/50** bit-verified SAT witnesses including an `R=16` standard-IV 256-bit exact collision (`P13`).  
> 3. **Counter-Intuitive Mechanistic Discoveries**: Exact zero-training $2^{16}$ SVD proves SHA-256's `(2,13,22)/(6,11,25)` rotations reach Page-Curve entanglement saturation at **$R=2\text{–}3$** (~2× entropy vs random clones at $R=2$); `ADD32 ⊗ ROTR` acts as a dual chaotic singularity ($\bar{\alpha}_c = 0.2400$, min-cut $\rho_{\text{cut}} = 0.8629$); and we expose the **"Forward-Manifold Membership Confounder"** where value/classifier heads score `0.969–0.9999` while yielding zero inverse speedup (`0/52` non-trivial BTC configs).  
> **10-Second CPU Verification**: Run `python scripts/verify_all_claims.py` (zero GPU, zero third-party dependencies). See [REPRODUCE.md](REPRODUCE.md).

---

## 一、项目全景与严格科学边界（简体中文总览）

**Phase-SHA (`V1–V12`)** 是一套面向全球密码学、形式化验证与机器学习机理研究社区公开的**全开源、零隐藏、一键本地 CPU 可复验**密码分析实证与形式化基准库。我们不将大量失败探索藏进抽屉（消除 *File-Drawer Effect*），而是把跨越 **V11（P1–P11）** 与 **V12（P12、P13、P15）** 共 **14 条独立攻防路线**、**4 套零 `sorry` Lean 4 形式化语料库（32 文件 / 483 条定理引理，含 338 公开 + 145 私有保护声明）**、**50/50 条 SAT 十六进制碰撞见证** 以及 **16 组 80-byte 比特币区块头实测数据** 全部原样公开，接受全世界的检验与复现。

### 严格不越界声明（No-Overclaim Scientific Boundary）
- **我们不声称**“无条件证明了世界上绝对不存在攻破 SHA-256 的多项式时间捷径”（在计算复杂性理论中，这类无条件结论等价于证明 $\text{P} \neq \text{NP}$）。
- **我们严格主张并交付三类硬成果**：
  1. **显式模型下的机器证明与不可行下界**：在 Lean 4.36.0 位向量核、Z3 SMT、PySAT (`CaDiCaL195`) 与 OR-Tools CP-SAT 字级模型下，给出可核查的精确 `UNSAT`、`EXACT_MIN_PROVEN`、字级穿透上限 `r_max = 32` 与 483 条零 `sorry` 定理/引理（338 公开 + 145 私有）；
  2. **14 条范式统一受控实证相变壁垒**：在纯 IEEE-754 FP32（`allow_tf32=False`）、零训练/留出重叠（`overlap=0`）与谱隙对齐随机对照密码（`CTRL_RAND_ROTR_K`）下，证明统计、神经、同伦、Koopman、张量网络、广义中性比特与截断 SAT 暗道在可控区间之外全面归零；
  3. **工业界/学术界反幻觉审计套件**：揭露并量化 6 类极具迷惑性的伪阳性信号（尤其是导致神经网络看起来“学到了 SHA-256 规律”的前向流形归属混淆器），提供开源 `scripts/verify_all_claims.py` 一键打假脚本。

---

## 二、14 条路线 (`P1–P15`) 全量实验裁决总表

### 2.1 V11 十一路探索（统计物理、代数表示、张量网络与符号证真）

| 编号 | 路线与目录 | 科学探索命题 | 最终裁决 | 核心实测证据与指标（纯 FP32 / 形式化） | 档案入口 |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **P1** | `p1_infonce` | 跨轮变分互信息望远镜 $I(f_\theta(S_0); g_\phi(S_R))$ | **`NULL`** | $R=4$ 阳性校准 `9.99673 bits`（`z=+5495.13`）；$R \in \{6..20\}$ 全面归零 `MI ∈ [-0.00006, +0.00002]b` | [REPORT](v11_discovery/p1_infonce/REPORT.md) · [verdict](v11_discovery/p1_infonce/verdict.json) |
| **P2** | `p2_zero_mode` | 3D 时空晶体等零面与零模波包（504 格点） | **`NULL`** | 剔除末轮线注后，84 个非平凡格点全面坍缩至 $2^{-(k+1)}$（`z ∈ [-3.20, +2.57]`），亮灯格点 `0/504` | [REPORT](v11_discovery/p2_zero_mode/REPORT.md) · [verdict](v11_discovery/p2_zero_mode/verdict.json) |
| **P3** | `p3_symbolic_lean4` | 神经/MCTS 猜想 $\times$ Z3 V1–V7 $\times$ Lean 4 自动证真 | **`HIT`** | 从 `31,847` 条候选中提纯并 **28/28** 零 `sorry` 证真（5–12 轮跨块恒等式，耦合 B1+B2 达 23 轮走廊） | [REPORT](v11_discovery/p3_symbolic_lean4/REPORT.md) · [verdict](v11_discovery/p3_symbolic_lean4/verdict.json) |
| **P4** | `p4_koopman_distinguisher` | 3D 晶体多体元组 Koopman 区分器与谱蒸馏 | **`NULL`** | $R=4$ `AUC=1.000`；$R=6$ `F2_mod2=0.88687`；$R \ge 8$ 主实验归零（`0.496~0.504`），随机对照 $R=8$ 仍泄漏 `0.5579` | [REPORT](v11_discovery/p4_koopman_distinguisher/REPORT.md) · [verdict](v11_discovery/p4_koopman_distinguisher/verdict.json) |
| **P5** | `p5_btc_weak_midstate` | BTC Block 2 固定填充 $\times$ 弱 Midstate 共振探测 | **`NULL`** | $t=0..6$ 局部进位可压至 `0.095~0.267`，Nonce `W[3]` 注入后 $t=7$ 跳至 `0.407`，`R=8..64/double64` 零统计偏置 | [REPORT](v11_discovery/p5_btc_weak_midstate/REPORT.md) · [verdict](v11_discovery/p5_btc_weak_midstate/verdict.json) |
| **P6** | `p6_grokking_control` | 强 `wd` 诱导 Grokking 顿悟 + 4× SAE 因果消融 | **`NULL`** | $R=4$ `REAL AUC=0.9098` (`L0=101.4`) vs `CTRL_1=1.0000` (`L0=8.2`)；$R \ge 6$ `AUC≈0.50`，SAE 膨胀为稠密表 (`L0=422.2`) | [REPORT](v11_discovery/p6_grokking_control/REPORT.md) · [verdict](v11_discovery/p6_grokking_control/verdict.json) |
| **P7** | `p7_dual_group_spectral` | $\mathbb{F}_2 \times \mathbb{Z}/2^{32}\mathbb{Z}$ 双群 Cayley 正交 $O(d)$ 基稀疏化 | **`NULL`** | $R=2$ 捕获 $(A_2-E_2)\bmod 2^{32}$ 消去（`H=0.2061`）；$R \ge 4$ 坍缩至 Haar 随机基（`H=0.868~0.934`, `|z|<=1.42`） | [REPORT](v11_discovery/p7_dual_group_spectral/REPORT.md) · [verdict](v11_discovery/p7_dual_group_spectral/verdict.json) |
| **P8** | `p8_anisotropic_homotopy` | 各向异性门控同伦分岔测绘与最小割集 | **`NULL`** | $R=4$ 亚渗流（`rho_cut=0.3828`）；$R \ge 6$ $\bar{\alpha}_c$ 锁死 `0.2400`，$\rho_{\text{cut}}=0.8629$ (`5302/6144`)，唯一奇点 `only_Add=0.1134` | [REPORT](v11_discovery/p8_anisotropic_homotopy/REPORT.md) · [verdict](v11_discovery/p8_anisotropic_homotopy/verdict.json) |
| **P9** | `p9_arx_proxy_circuit` | AlphaTensor 式多线性 + MCTS 紧凑低次概率代理电路 | **`NULL`** | 剔除退化 $R=2$ 与单步对称进位后，198 个真多轮复合位点 `max z=2.60`, Bonferroni `p_adj=1.00`, `0` 有效命中 | [REPORT](v11_discovery/p9_arx_proxy_circuit/REPORT.md) · [verdict](v11_discovery/p9_arx_proxy_circuit/verdict.json) |
| **P10** | `p10_ugly_algebra` | 无先验可逆代数 $\Phi_\theta$（32b）与 256b 联合迭代（Part C） | **`PARTIAL POSITIVE / NULL-END2END`** | 单步 32-bit `ugly_invertible` (`D=512`) 逃逸双线性 $D_{95}\propto 2^n$ 墙（`add=0.97124`, 胜过人工基 `2.5x`）；Part C 256b 全线常向量坍缩（`recon=0.500`） | [REPORT](v11_discovery/p10_ugly_algebra/REPORT.md) · [verdict](v11_discovery/p10_ugly_algebra/verdict.json) |
| **P11** | `p11_tensor_network` | 精确 $\chi=2$ MPO、$2^{16}$ 精确 SVD 谱、$O(4)$ 解纠缠与漏斗 | **`PARTIAL POSITIVE / DEEP-NULL`** | `rotr_zero R=6` 零误差；$R=2$ 真实 SHA-256 熵达随机对照 **2×**（`5.5384b vs 2.8867b`）并于 $R=3$ 锁死 Page-Curve；$O(4)$ 于 $R\ge 3$ 归零；漏斗揭露 `1v4=0.969` 伪信号 | [REPORT](v11_discovery/p11_tensor_network/REPORT.md) · [verdict](v11_discovery/p11_tensor_network/verdict.json) |

### 2.2 V12 三线前沿攻坚（比特币 Double-SHA256 硬件摊还、SAT 碰撞隧道与 MitM/Biclique）

| 编号 | 路线与目录 | 科学探索命题 | 最终裁决 | 核心实测证据与指标（本地 CPU 100% 复验通过） | 档案入口 |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **P12** | `track_a_p12_amortized` | 超广义 AsicBoost：广义中性比特团簇（GNBC）与多输入硬件 GE 摊还 | **`NULL`**（校准与 SAT 界 `PASS`） | 精准复现经典 Overt AsicBoost（`9.539%` 全内循环 / `21.24% Chunk2`）；全部神经/2-bit/`PhiCh` 仅撑 `r_neutral=1`（`p_neutral(t>=4)=0.0`）；跨块 `0/262,144`；Z3/PySAT 证死 `UNSAT` 与 `min_active=2` | [REPORT](v12_frontier/track_a_p12_amortized/REPORT.md) · [verdict](v12_frontier/track_a_p12_amortized/verdict.json) |
| **P13** | `track_b1_p13_local_coll` | AlphaZero 3D 晶体局部碰撞隧道 & BTC 截断前导零差分 | **`PARTIAL POSITIVE`**（深轮/BTC `NULL`） | Mendel 状态机 `100万` 次自检零错 + **50/50** SAT 见证验真；标准 `IV` $R=16$ `0.287s` 求得 256-bit 精确碰撞，$R \ge 20$ 全 `UNSAT`；G3 抓获 `0.9999` 混淆伪信号；BTC 截断 **`0/52`** 加速（最高 `0.0159x`） | [REPORT](v12_frontier/track_b1_p13_local_coll/REPORT.md) · [verdict](v12_frontier/track_b1_p13_local_coll/verdict.json) |
| **P15** | `track_b2_p15_mitm_biclique` | 截断前导零专用四色 MitM / Biclique 搜索 + Lean 4 证真 | **`NULL`**（校准与 Lean 4 `PASS`） | 复现 $R \in \{16..32\}$ 4-色 MitM（$\log_2 T=224.04 < 256$）与 64r `L=6/8` Biclique；锁定字级穿透上限 `r_max=32`（$R\ge 41$ `INFEASIBLE`）；Stage 2 BTC `216` 配置全部 $k \le 240$ **`0`** 加速；**32** 条 Lean 4 零 `sorry` 定理 | [REPORT](v12_frontier/track_b2_p15_mitm_biclique/REPORT.md) · [verdict](v12_frontier/track_b2_p15_mitm_biclique/verdict.json) |

> **跨阶段综合终审报告直达**：[V11 十一路专家整合总纲](v11_discovery/final_synthesis/V11_MASTER_SYNTHESIS.md) · [V12 三线终审整合总报告](v12_frontier/final_synthesis/V12_MASTER_REPORT.md) · [密码学深度耦合分析](v11_discovery/final_synthesis/01_crypt_coupling.md) · [统计物理五相图与有效场论](v11_discovery/final_synthesis/02_physics_phase_diagram.md)

---

## 三、四大跨学科反直觉科学发现（Core Findings）

1. **瞬时 Page-Curve 饱和与 `ADD32 ⊗ ROTR` 互补奇点定律（P8 × P11）**：纯模加无旋转（`rotr_zero`）在 $R=6$ 轮上可被精确 $\chi=2$ MPO 零误差（`0.00000`）压缩；纯旋转无模加仅属 $\chi=1$ 可逆置换；唯独当 `ADD32` 与 `ROTR` 交替非对易耦合时，SHA-256 仅用 2–3 轮即触顶随机预言机 Page-Curve 体积律上限，并形成 $\bar{\alpha}_c = 0.2400$、最小割集 $\rho_{\text{cut}} = 0.8629$ 的高维渗流屏障。
2. **真实 SHA-256 比随机对照快 1–2 轮热化（P4 × P6 × P7 × P11）**：$\Sigma_0=(2,13,22)$ 与 $\Sigma_1=(6,11,25)$ 实现了 $\mathbb{Z}/32\mathbb{Z}$ 环面近似等距三分与谱隙最大化。P11 零训练 $2^{16}$ SVD 证明真实 SHA-256 在 $R=2$ 时即制造出随机 `ROTR/K` 对照组 **近 2 倍纠缠熵（`5.5384b` vs `2.8867b`）与 6 倍 Schmidt 秩**，从而解释了为何随机对照组在 P4 ($R=8$) 与 P6 ($R=6$) 中反而还残留虚假统计泄漏。
3. **32-bit 双线性表示墙的可逃逸性与 256-bit 联合坍塌对偶（P10 × P6 × P7）**：在单步 32-bit `ADD+XOR+ROTR` 上，无先验双线性基遭遇 $D_{95}\propto 2^n$ 指数灾难，而解析可逆仿射耦合（`ugly_invertible`, `D=512`）可突破至 `add=0.97124`；但跨多轮串联 6 次 `ADD32` 令误差呈 Lyapunov 指数放大，最终与 Grokking SAE `L0` 爆炸（`101.4 -> 422.2`）共同刻画出表示空间与高维混淆相变的物理对偶。
4. **“前向可达流形归属混淆器”与反幻觉 7 条铁律（P2 × P9 × P11 × P12 × P13）**：揭示神经网络在密码分析评估中最隐蔽的造假级假象——分类器/价值头能够以 **`0.969 ~ 0.9999`** 的极高准确率识别样本是否落在前向合法轨迹流形内，却在贪心 1-bit 逆向修复与多候选采样上产生 **零有效加速**（`0/52` BTC 截断加速）。

---

## 四、Lean 4 形式化证明资产总览（32 文件 · 483 定理引理 · 0 `sorry`）

全部 4 套 Lean 4.36.0 工程均采用 **零 Mathlib、原生 `bv_decide` 位向量决策过程**（可在 `<= 2 GB` 内存的普通消费级 CPU 上快速重编译，无任何 `sorry`、`admit` 或自定义公理）：

- [formal_proofs/](formal_proofs/)（V3/V4 核心库：`12` `.lean` 文件，`180` 公开 / `313` 全部定理引理，`0 sorry`）
- [formal_proofs_v10/](formal_proofs_v10/)（V10 减轮库：`8` `.lean` 文件，`47` 公开 / `56` 全部定理引理，`0 sorry`）
- [v11_discovery/p3_symbolic_lean4/lean_verify/](v11_discovery/p3_symbolic_lean4/lean_verify/)（V11 P3 跨轮/跨块库：`7` `.lean` 文件，`67` 条定理引理 [覆盖 28 条目标定理]，`0 sorry`）
- [v12_frontier/track_b2_p15_mitm_biclique/lean_verify/](v12_frontier/track_b2_p15_mitm_biclique/lean_verify/)（V12 P15 MitM/Biclique 库：`5` `.lean` 文件，`44` 公开 / `47` 全部定理引理 [审计 32 条]，`0 sorry`）

---

## 五、快速一键复现与引用

```bash
# 10 秒零显卡零第三方依赖全量核验（SHA-256 对拍 + 50/50 SAT 见证 + 16/16 BTC 区块头 + 483 Lean 4 声明零 sorry）
CUDA_VISIBLE_DEVICES="" python -u scripts/verify_all_claims.py

# 如本机已安装 elan / Lean 4.36.0，追加真实编译器重构建与公理审计：
CUDA_VISIBLE_DEVICES="" python -u scripts/verify_all_claims.py --run-lean
```

- 完整复现手册：[REPRODUCE.md](REPRODUCE.md)
- 开源协议：[LICENSE](LICENSE) · 引用格式：[CITATION.cff](CITATION.cff)
