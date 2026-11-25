# P8「各向异性 / 复平面反绝热同伦分岔测绘与最小割集定位」结案报告 (`REPORT.md`)

> **工作目录**：`g:\SHA-LM\v11_discovery\p8_anisotropic_homotopy\`  
> **数学规范**：[SPEC.md](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/SPEC.md)  
> **最终裁决**：**`NULL`**（[verdict.json](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/verdict.json)）  
> **算力与合规审计**：Colab `v11-p8`（NVIDIA L4，双会话均 $\le 45$ 分钟），严格纯实数 `torch.float32` + `allow_tf32=False`（零 `complex64`），禁止本地 GPU，未改动 `v6a/`、`src/`、`v10_ouroboros64/` 等任何只读目录；已执行 `colab.exe stop -s v11-p8` 并经 `colab.exe sessions` 核实无残留会话（`colab_stopped: true`）。

---

## 核心结论摘要

针对 V6A 标量均匀同伦 $\alpha(s)=s$ 越过 $\alpha_c$ 后爆发分岔湍流的问题，P8 将标量 $\alpha$ 升维为三维时空各向异性门控场 $\alpha(t, g, b; s)$ 与纯实数 $2 \times 2$ 矩阵表示的复平面反绝热双伴随场 $(\alpha_{\text{re}}, \alpha_{\text{im}})$，在 **$R \in \{4, 6, 8, 12, 18, 24, 64\}$**、**真实 SHA-256 vs 同构随机 `ROTR/K` 对照组** 上对 9 条主同伦路径（含 4 项 `P6` 消融）、2D 边界松弛对比及单组件奇点/最小割集进行了全量测绘。核心发现如下：

1. **数学不变性 100% 守恒**：全部 13 条调度路径在 $s=1$ 端点均严格还原 V6A 精确 SHA-256（最大算子差异 $\le 1.19 \times 10^{-7}$，FP32 机器精度内），且真迹原像 $x^*$ 在全路径 $s \in [0, 1]$ 上保持严格不动点（`max_fixed_point_err = 0.0`），$\alpha=0$ 初态 $\mathbb{F}_2$ 线性违约率恒为 $0$。
2. **分岔临界线无法被各向异性或复绕行绕开**：在全部轮次 $R \in \{4, \dots, 64\}$ 上，**全部路径均触发失稳分岔**（`bifurcated = True`），`exact_hits = 0`：
   - 比特波前（`P1_bit_lsb_to_msb`, `P2_bit_msb_to_lsb`）与轮次波前（`P3_round_forward`, `P4_round_backward`）在 $R \ge 6$ 上将局部非线性压力提前集中于少数比特/轮次，导致有效临界开启密度 $\bar{\alpha}_c$ 从 `P0_uniform` 的 `0.2400` **提前跌落至 `0.0800–0.1600`**。
   - 复平面反绝热绕行（`P6_cd_complex_detour`）因双伴随通道在差异子空间内的反射放大效应（$\kappa_{\text{im}} + \lambda_{\text{cd}}\alpha_{\text{re}} > 1$），当 $s > 0.7$ 时引发虚部与实部耦合范数指数漂移爆炸（`drift_slope = +67.4 ~ +98.6`），终态 $E_{32}$ 显著恶化；纯规范速度阻尼 `P6_pure_damping`（$\gamma_0 = 0$）消除了漂移爆炸，但 $\bar{\alpha}_c$ 与终态残差与 `P0_uniform` 几乎无异（$R=64$: `E32_min = 109.75` vs `112.37`）。
   - **重要物理现象（`P5a` 与 `P7_learned_aniso` 的“虚假推迟”）**：`P5a_gate_bool_then_add` 与独立端到端优化的神经各向异性场 `P7_learned_aniso` 不约而同收敛到同一策略——**率先打开局部布尔门 `Ch/Maj`、极力压制长程 `ADD32` 进位门**（$R=64$ 训练后在 $s=0.5$ 处 $\alpha_{\text{Ch}}=0.1977,\; \alpha_{\text{Maj}}=0.1979,\; \alpha_{\text{Add}}=0.0576$）。当 `ADD32` 关闭时，V6A 线性骨架与 `Ch/Maj` 确可和平共处（`P5a` 在 $s=0.48$ 时 `ch_err=0.0006`, `maj_err=0.0014`, `gl_gap` 回落至 `0.0176–0.0211`），从而将全局平均开启密度 $\bar{\alpha}_c$ 抬高至 `0.348–0.427`；但**一旦 `ADD32` 进位门越过约 $0.20\text{–}0.28$ 临界阈值，系统立即爆发同等级别湍流**，且由于后期进位集中陡升，终态 $s=1$ 能量反而劣于均匀同伦 `P0`（$R=64$: `P5a E32_min = 118.55`, `P7 = 135.44` vs `P0 = 112.37`）。
3. **混沌回灌唯一奇点与普适 ARX 最小割集**：单组件消融与贪心搜索精确证明，分岔湍流的唯一主触发奇点是与 $\mathbb{F}_2$ 线性骨架耦合的 `ADD32` 进位链（$R=64$ 单开 `Ch` 线性违约 `0.0095`、`Maj` `0.0074`，均不破线；单开 `Add` 高达 **`0.1134`**、`gl_gap = 0.4210`），核心奇点槽位为 `u1 / u3 / E_next / A_next` 与 $R > 16$ 的消息扩展 `sched`。该进位回灌网络呈高度致密全向渗流结构：周期性位割（`stride=2` 切 50% 比特）完全失效；$R \ge 6$ 上须切断 **`82.5%–88.8%`** 的 $(t, g, b)$ 门（或截断 `28–32/32` 比特）才可恢复线性流形，不存在 SHA-256 专属低维割集。
4. **同构随机 `ROTR/K` 对照组完全重合**：真实 SHA-256（`main`）与随机 `ROTR/K` 克隆密码（`control`）在全部路径上的 $\bar{\alpha}_c$、固定边界跃迁 $\Delta E_{\text{hom}}$、终态 $E_{32}$ 与最小割集占比 $\rho_{\text{cut}}$ 均高度一致（例如 $R=64$ 下 `P7` $\bar{\alpha}_c$: `0.4041` vs `0.4041`；$\rho_{\text{cut}}$: `0.8629 ± 0.0009` vs `0.8626 ± 0.0006`, $z=+0.58$）。依据 [SPEC.md §5.2](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/SPEC.md)，正式裁决为 **`NULL`**。

---

## 1. 算力、不变性契约与工程实现核验

### 1.1 工程产物与停机核实
- 只读引用 `v6a/`（`v6a/v6a_homotopy.py` 与 `v6a/v6a_gl_rrr.py` 时间戳保持 `2026/11/23` 未变更），所有 P8 实现均位于 `g:\SHA-LM\v11_discovery\p8_anisotropic_homotopy\`：
  - [control_cipher.py](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/control_cipher.py)：同构随机 `ROTR/SHR/K` 对照密码（固定种子 `20261125`，重采样 $\Sigma_0=(11,21,25)$、$\Sigma_1=(15,20,24)$、$\sigma_0=(13,15,\text{shr}=2)$、$\sigma_1=(15,23,\text{shr}=4)$ 及 64 个随机轮常数 $K_{\text{ctrl}}$）。
  - [anisotropic_homotopy.py](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/anisotropic_homotopy.py)：时空各向异性 `AnisotropicGLRRREngine`、纯实数 $2 \times 2$ 双伴随矩阵耦合 `combine_B_dual`、可微单调样条 `LearnedAnisotropicField`（$K_{\text{bin}}=8$，零初态精确退化为 `P0_uniform`）。
  - [smoke_invariants.py](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/smoke_invariants.py) 与 [run_p8_experiments.py](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/run_p8_experiments.py)。
- Colab 运行日志（[run_log.txt](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/results/run_log.txt)）记录两轮 NVIDIA L4 会话（`allow_tf32=False`，每轮均 $< 45$ 分钟）：扫频总计 `3281.08s`，2D 边界对照 `327.16s`，割集消融 `862.18s`；结束同步后已执行 `colab.exe stop -s v11-p8` 并在 `colab.exe sessions` 中核实 `v11-p8` 消失。

### 1.2 端点精确性与真迹不动点守恒验证
本地 CPU（`CUDA_VISIBLE_DEVICES=""`，耗时 `1.672s`）输出的 [invariants_smoke.json](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/results/invariants_smoke.json) 与 Colab 全量运行检查证明：

| 检验项 | `R=4` (`main` / `control`) | `R=8` (`main` / `control`) | 契约要求 |
| :--- | :---: | :---: | :--- |
| 真迹 $x^*$ 2-adic 能量 `gt_e32_max` | `6.31e-30` / `7.10e-30` | `6.71e-30` / `1.62e-29` | $= 0$（双精度/FP32 精度内） |
| 真迹离散门违约 `gt_disc_viol_max` | `0` / `0` | `0` / `0` | 精确为 `0` |
| 端点 $s=1$ 与 V6A $P_B$ 最大差异 `s1_v6a_pb_max_diff` | `1.19e-7` / `0.0` | `1.19e-7` / `0.0` | $\le 10^{-6}$ |
| 全 13 路径 $s \in \{0, 0.25, 0.5, 0.75, 1\}$ 真迹不动点误差 | **`0.0`** / **`0.0`** | **`0.0`** / **`0.0`** | $< 10^{-6}$ |
| $\alpha=0$ GF(2) 精确初态线性违约 | `0` / `0` | `0` / `0` | 全量 $R=4..64$ 扫频均通过 $\le 10^{-6}$ 断言 |

---

## 2. 全轮次各向异性与反绝热同伦分岔测绘

数据来源：[homotopy_sweep.json](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/results/homotopy_sweep.json)（`free_msg`、`allow_peel_back=False`、`mask_m=256`、`s` 网格 26 点 $\Delta s=0.04$、每级 `8` 步 GL-RRR）。注：SPEC §4.2 的 `energy_spike` 项因 $E_{\text{hom}}(0)=0$ 导致首步 $s=0.04$ 在所有从零平滑线性增长的曲线上即触发 $E_{\text{hom}}(0.04) > 1.25 \times 0$；物理真实失稳阈值应看同步记录的 `bifurcation_instability`（`gl_gap > 0.14` 或 `drift_slope > 1e-2` 或 `lin_viol_rate > 0.02`）。

### 2.1 `R=4` 阳性校准与深轮代表点（`R=8, 18, 64`）全路径对比（真实 SHA-256 `main`）

| 轮数 | 路径 | 失稳临界 $s_c$ | 临界平均开启度 $\bar{\alpha}_c$ | 固定边界跃迁 $\Delta E_{\text{hom}}$ | 最大单步跃迁 $\Delta E_{\max}$ | 终态线性违约 `lin_viol` | 终态相隙 `gl_gap` | 终态漂移 `drift_slope` | 终态最优 $E_{32,\min}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`R=4`** | `P0_uniform` | `0.28` | `0.2800` | `0.0541` | `0.0183` | `0.0228` | `0.2342` | `+0.0023` | `0.0042` |
| **`R=4`** | `P1_bit_lsb_to_msb` | `0.84` | `0.8400` | `0.0167` | `0.0078` | `0.0083` | `0.1357` | `+0.0024` | `0.3893` |
| **`R=4`** | `P2_bit_msb_to_lsb` | `0.28` | `0.2800` | `0.0286` | `0.0081` | `0.0125` | `0.1725` | `+0.0033` | **`0.0000`** |
| **`R=4`** | `P4_round_backward` | `0.24` | `0.2400` | `0.0310` | `0.0183` | `0.0138` | `0.1814` | `+0.0028` | `0.0007` |
| **`R=8`** | `P0_uniform` | `0.24` | `0.2400` | `0.1410` | `0.0190` | `0.0666` | `0.3625` | `+0.0028` | **`3.1275`** |
| **`R=8`** | `P1_bit_lsb_to_msb` | `0.16` | `0.1600` | `0.1316` | `0.0135` | `0.0656` | `0.3589` | `+0.0023` | `6.2426` |
| **`R=8`** | `P2_bit_msb_to_lsb` | `0.16` | `0.1600` | `0.1414` | `0.0101` | `0.0686` | `0.3696` | `+0.0024` | `3.6705` |
| **`R=8`** | `P3_round_forward` | `0.16` | `0.1600` | `0.1500` | `0.0166` | `0.0724` | `0.3787` | `+0.0019` | `3.5454` |
| **`R=8`** | `P4_round_backward` | `0.16` | `0.1600` | `0.1438` | `0.0165` | `0.0709` | `0.3720` | `+0.0019` | `4.0445` |
| **`R=8`** | `P5a_gate_bool_then_add` | `0.28` | `0.3733` | `0.1496` | `0.0265` | `0.0709` | `0.3708` | `+0.0026` | `5.8163` |
| **`R=8`** | `P5b_gate_add_then_bool` | `0.16` | `0.1067` | `0.1718` | `0.0199` | `0.0830` | `0.4054` | `+0.0012` | `5.0024` |
| **`R=8`** | `P6_cd_complex_detour` | `0.24` | `0.2400` | `0.4784` | `0.0597` | `0.1893` | `0.6106` | **`+76.5940`** | `5.7138` |
| **`R=8`** | `P6_pure_damping` | `0.24` | `0.2400` | `0.1406` | `0.0190` | `0.0663` | `0.3605` | `+0.0027` | `5.2336` |
| **`R=8`** | `P7_learned_aniso` | `0.76` | `0.3483` | `0.1857` | `0.0232` | `0.0870` | `0.4062` | `+0.0016` | `5.1870` |
| **`R=18`** | `P0_uniform` | `0.24` | `0.2400` | `0.1482` | `0.0205` | `0.0754` | `0.3761` | `+0.0023` | `17.7088` |
| **`R=18`** | `P2_bit_msb_to_lsb` | `0.12` | `0.1200` | `0.1743` | `0.0109` | `0.0884` | `0.4143` | `+0.0014` | `15.8058` |
| **`R=18`** | `P5a_gate_bool_then_add` | `0.28` | `0.3733` | `0.1618` | `0.0278` | `0.0814` | `0.3925` | `+0.0022` | `15.5999` |
| **`R=18`** | `P6_cd_complex_detour` | `0.24` | `0.2400` | `0.5128` | `0.0776` | `0.2301` | `0.6287` | **`+71.0249`** | `26.6054` |
| **`R=18`** | `P7_learned_aniso` | `0.76` | `0.3786` | `0.1969` | `0.0323` | `0.0980` | `0.4266` | `+0.0014` | `24.0600` |
| **`R=64`** | `P0_uniform` | `0.24` | `0.2400` | `0.1667` | `0.0269` | `0.0882` | `0.4117` | `+0.0023` | `112.3715` |
| **`R=64`** | `P1_bit_lsb_to_msb` | `0.12` | `0.1200` | `0.2030` | `0.0146` | `0.1080` | `0.4597` | `+0.0008` | `144.5204` |
| **`R=64`** | `P2_bit_msb_to_lsb` | `0.12` | `0.1200` | `0.2070` | `0.0149` | `0.1092` | `0.4643` | `+0.0008` | `120.7950` |
| **`R=64`** | `P3_round_forward` | `0.12` | `0.1200` | `0.2074` | `0.0108` | `0.1098` | `0.4638` | `+0.0010` | `121.4469` |
| **`R=64`** | `P4_round_backward` | `0.08` | `0.0800` | `0.2085` | `0.0119` | `0.1093` | `0.4636` | `+0.0010` | `121.6581` |
| **`R=64`** | `P5a_gate_bool_then_add` | `0.32` | `0.4267` | `0.1829` | `0.0333` | `0.0959` | `0.4322` | `+0.0021` | `118.5501` |
| **`R=64`** | `P5b_gate_add_then_bool` | `0.12` | `0.0800` | `0.2006` | `0.0265` | `0.1066` | `0.4595` | `+0.0002` | `122.5770` |
| **`R=64`** | `P6_cd_complex_detour` | `0.24` | `0.2400` | `0.5851` | `0.1061` | `0.2962` | `0.6750` | **`+67.4203`** | `149.5839` |
| **`R=64`** | `P6_pure_damping` | `0.24` | `0.2400` | `0.1654` | `0.0262` | **`0.0872`** | **`0.4099`** | `+0.0022` | **`109.7513`** |
| **`R=64`** | `P7_learned_aniso` | `0.76` | `0.4041` | `0.2087` | `0.0434` | `0.1092` | `0.4566` | `+0.0016` | `135.4377` |

*(注：在 `R=4` 阳性校准上，`S_4` 与固定 `IV` 之间仅隔 4 轮，因此 `P2_bit_msb_to_lsb`、`P3`、`P4` 可将 `E32_min` 压低至 `0.0000~0.0011`；一旦进入深轮 $R \ge 6$，`R=6` `P0` 跃升至 `1.8723`，`R=12` `4.7537`，`R=24` `26.1874`，`R=64` `112.3715`，全路径 `exact_hits = 0`。)*

---

## 3. 动力学机理解析：为何各向异性与复绕行无法绕过分岔

### 3.1 `P5a` 与 `P7_learned_aniso` 的共性物理发现（布尔可解 vs 进位雪崩）
检视 `P5a_gate_bool_then_add` 在 $R=8, 18, 64$ 推进区间 $s \in [0.24, 0.64]$ 的精细轨迹，揭示了一个极其清晰的双相动力学：
1. 在 $s \in [0, 0.50]$ 期间，仅 `Ch` 与 `Maj` 线性打开至 `1.0`，而 `ADD32` 进位门严格关闭（$\alpha_{\text{Add}}=0$）。虽然 `gl_gap` 在 $s=0.28\text{–}0.32$ 一度擦过 `0.149` 阈值，但随后在 $s=0.40 \to 0.48$ **自发回落至极低平静平台**：以 $R=64$ 为例，$s=0.48$ 时 `ch_err` 降至 **`0.0006`**、`maj_err` 降至 **`0.0014`**、`gl_gap = 0.0211`、`lin_viol_rate = 0.0113`！这证明 $\mathbb{F}_2$ 线性骨架与纯布尔门 `Ch/Maj` 的联合流形在 64 轮上**并不存在不可调和的分岔壁垒**。
2. 然而当 $s > 0.50$、`ADD32` 进位开始开启时，即使仅仅打开 $\alpha_{\text{Add}} \approx 0.20\text{–}0.28$（$s=0.60\text{–}0.64$），`gl_gap` 立即从 `0.0211` 飙升至 `0.1722 -> 0.4322`，`lin_viol_rate` 飙升至 `0.0959`。反之，在 `P5b_gate_add_then_bool`（先开 `Add`）中，仅在 $s_c = 0.12\text{–}0.16$（$\alpha_{\text{Add}} = 0.24\text{–}0.32$，$\bar{\alpha}_c = 0.080\text{–}0.107$）就立即崩盘。
3. **神经场 `P7_learned_aniso` 的自发收敛行为**：从零初始化（`P0`）出发训练 100 步后，在全部 $R \in \{4, 6, 8, 12, 18, 24, 64\}$ 上，`P7` 自发将虚部 $\alpha_{\text{im}}$ 抑制为 `0.0000`，并将实部样条压成凹曲线——以 $R=64$ 在 $s=0.5$ 为例：
   - 门类平均：`Ch = 0.1977`, `Maj = 0.1979`, **`Add = 0.0576`**（把 `Add` 进位死死拖到 $s > 0.72$ 才被迫释放）；
   - 而轮次（`early=0.1505` vs `late=0.1513`）与比特带（`lsb8=0.1495` vs `msb8=0.1520`）几乎各向同性！
   - 结果导致 `P7` 在 $s=0.72$ 前维持低 `gl_gap = 0.1370`，到 $s=0.76$ ($\bar{\alpha}_c=0.4041$) `Add` 门集中开启时立刻剧烈分岔，且因后半程松弛步数不足，末态能量反而比 `P0_uniform` 恶化约 `20%`（$R=64$: `E32_min = 135.44` vs `112.37`）。

### 3.2 `P6` 复平面反绝热绕行消融分析
在 $R \in \{4, 8, 18, 64\}$ 上的 5 组 `P6` 消融（`P6_cd_complex_detour` $\gamma_0=0.4$、`P6_gamma02`、`P6_gamma07`、`P6_pure_complex`、`P6_pure_damping`）一致表明：
- **纯实数反绝热速度阻尼**（`P6_pure_damping`, $\gamma_0=0, \eta_{\text{cd}}=0.15$）可微弱缓和 $R=64$ 终态湍流（`E32_min = 109.75` vs `P0 112.37`，`lin_viol = 0.0872` vs `0.0882`），但无法改变临界阈值 $\bar{\alpha}_c = 0.2400$。
- **任意非零虚部耦合**（$\gamma_0 \in \{0.2, 0.4, 0.7\}$）均在 $s \ge 0.75$ 引发双伴随通道范数失控（`drift_slope = +8.7 ~ +98.6`）。数学根源在于 `combine_B_dual` 中差异子空间（$y_A^{\text{im}} = 0 \implies r_{\text{im}} = -x_{\text{im}}$）的离散迭代乘子为 $1 - \beta (1 - \kappa_{\text{im}}(s) - \lambda_{\text{cd}}\alpha_{\text{re}}(s))$：当 $s \to 1$ 使 $\kappa_{\text{im}} + \lambda_{\text{cd}}\alpha_{\text{re}} > 1$ 时该特征值绝对值突破 $1.0$，反而向实部通道注回发散震荡。

### 3.3 固定边界 `mask_m=256` vs 2D 边界松弛 (`mask_m: 64 -> 256`) 对照
在 [boundary_2d_compare.json](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/results/boundary_2d_compare.json)（$R \in \{4, 8, 18\}$）上，与固定 `m=256` 相比，2D 交错渐进锁边（`64 -> 256`）在各轮次与各路径上均录得**略高**的终态 `lin_viol_rate` 与 `delta_E_hom`（例如 $R=18$ `P0` 固定 `m=256` `dE=0.1453, lin=0.0741, E32=18.16` vs `2D` `dE=0.1538, lin=0.0775, E32=19.12`）。这表明在固定全摘要边界下观测到的 $\Delta E_{\text{hom}} \approx 0.14\text{–}0.17$ 能量跃迁并非强锁边人为导致的局部极小陷阱，而是高维非线性电路本身的体相（bulk）相变特征。

---

## 4. 最小进位回灌割集与触发奇点精确定位

数据来源：[cutset_ablation.json](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/results/cutset_ablation.json)（覆盖全部 $R \in \{4, 6, 8, 12, 18, 24, 64\} \times \{\text{main}, \text{control}\}$）。

### 4.1 单组件奇点探针（仅开单个门组 $\alpha=1$，其余保持 $\alpha=0$）

| 轮数 ($R$) | `only_Ch` (`lin` / `gap`) | `only_Maj` (`lin` / `gap`) | `only_Add` (`lin` / `gap`) | 最高触发 `ADD32` 槽位（按 `lin_viol` 降序） |
| :---: | :---: | :---: | :---: | :--- |
| `4` | `0.0004` / `0.0267` | `0.0000` / `0.0044` | **`0.0353` / `0.2851`** | `u3 (0.0490) > E_next (0.0443) > u1 (0.0433) > A_next (0.0362)` |
| `8` | `0.0060` / `0.0746` | `0.0052` / `0.0654` | **`0.0650` / `0.3172`** | `u3 (0.0557) > E_next (0.0490) > A_next (0.0367) > u1 (0.0336)` |
| `18` | `0.0089` / `0.0921` | `0.0082` / `0.0846` | **`0.0824` / `0.3472`** | `E_next (0.0393) > u3 (0.0366) > A_next (0.0324) > u1 (0.0301)` |
| `64` | `0.0095` / `0.1026` | `0.0074` / `0.0849` | **`0.1134` / `0.4210`** | **`sched (0.0297)`** `> E_next (0.0244) > u1 (0.0217) > A_next (0.0210)` |

- **触发机理一目了然**：直接连接 $\mathbb{F}_2$ 线性受限字（`S1`, 常量 `K`, `A/E` 移位寄存器及 $R>16$ 消息扩展字 `W_t`）的加法槽位 `u1, u3, E_next, A_next, sched` 在单组件激活时立即撕裂线性校验；而耦合到自由布尔辅助字 `Ch`/`Maj` 的 `u2 (0.0022–0.0034)`、`T2 (0.0015–0.0032)` 与 `ff (0.0000)` 几乎不引发冲突；比特带（4-bit nibble）则呈现全 32 位均匀分布（`n0..n6 ≈ 0.014–0.015`，`n7 ≈ 0.011` 仅因 MSB 最高位不向外传递进位而略低）。

### 4.2 反事实割集、轮深渗流与贪心最小割集占比 $\rho_{\text{cut}}$

| 轮数 ($R$) | 渗流轮深 $R_{\text{perc}}$ | 周期位割 `stride=2` (`lin_viol`) | 截断恢复所需最少比特数 (`cut_msb` / `cut_lsb`) | 贪心最小割集 `k_cut / n_tgb` | `main` 割集占比 $\rho_{\text{cut}}$ | `control` 割集占比 $\rho_{\text{cut}}$ | 主/对照 $z$-score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`4`** | `2` 轮 | `0.0443` (未恢复) | `20 / 32` | `138 / 384` | `0.3828 ± 0.0283` | `0.4134 ± 0.0439` | `-1.17` |
| **`6`** | `1` 轮 | `0.0675` (未恢复) | `28 / 32` | `479 / 576` | `0.8247 ± 0.0091` | `0.8125 ± 0.0099` | `+1.81` |
| **`8`** | `2` 轮 | `0.0746` (未恢复) | `28 / 32` | `663 / 768` | `0.8613 ± 0.0038` | `0.8571 ± 0.0141` | `+0.58` |
| **`12`** | `2` 轮 | `0.0828` (未恢复) | `28 / 32` | `1014 / 1152` | `0.8796 ± 0.0038` | `0.8774 ± 0.0035` | `+0.84` |
| **`18`** | `3` 轮 | `0.0871` (未恢复) | `32 / 32` (`lsb: 28`) | `1533 / 1728` | `0.8876 ± 0.0034` | `0.8838 ± 0.0049` | `+1.26` |
| **`24`** | `4` 轮 | `0.0891` (未恢复) | `32 / 32` (`lsb: 28`) | `2035 / 2304` | `0.8829 ± 0.0031` | `0.8908 ± 0.0025` | `-4.00` |
| **`64`** | `8` 轮 | `0.0945` (未恢复) | `32 / 32` (`lsb: 28`) | `5302 / 6144` | **`0.8629 ± 0.0009`** | **`0.8626 ± 0.0006`** | **`+0.58`** |

*(注：`R=24` 处因 `n_dig=4` 样本极小方差 `0.003` 使差值 `-0.0079` 呈现 `z=-4.00`，但绝对差异不足 `0.8%` 且在 `R=64` 彻底归零为 `z=+0.58`。)*  
**物理结论**：周期性切断 `ADD32` 进位链（即便每隔 1 位切断 50% 比特）反而保持或加重全局冲突（`lin_viol = 0.0746–0.0945`）；在深轮 $R \ge 6$ 中，必须关闭 **`82.5%–88.8%`** 的时空门坐标（对应 `62.9%–78.1%` 的非线性复制体）才能压制分岔，且在 $R=64$ 下即使关闭一半轮次（`tail_cuts w=32`）仍有 `lin_viol = 0.0587` 未恢复。

---

## 5. 真实 SHA-256 vs 随机 `ROTR/K` 对照组统计学审计

跨全部深轮 $R \in \{8, 12, 18, 24, 64\}$ 的双向对照检验汇总：
- **临界阈值 $\bar{\alpha}_c$**：确定性路径 `P0..P6` 在主/对照间 $|z_{\text{ctrl}}| \le 2.26$（绝大多数完全重合为 `0.00`）；`P7_learned_aniso` 在主/对照间绝对差距均 $< 0.0012$（$R=64$: `0.404138` vs `0.404149`）。
- **固定边界能量跃迁 $\Delta E_{\text{hom}}$**：全轮次全路径 $|z_{\text{ctrl}}| \le 2.63$（且 `P1` 在 $R=18$ 的 `z=-2.63`、`P7` 在 $R=64$ 的 `z=-2.15` 均无跨轮一致性）。
- **终态 2-adic 能量 $E_{32,\min}$**：全深轮绝大多数 $|z_{\text{ctrl}}| < 2.30$（仅 `P2` $R=18$ 出现 `18.92 vs 15.68`, $z=+3.34$ 的反向随机波动，且在 $R=24, 64$ 回落为 `z=+0.49, +1.44`）。
- **总结**：SHA-256 的各向异性同伦分岔临界线、跃迁幅度与进位渗流割集密度**与随机 `ROTR/K` 同构 ARX 密码完全同分布**。P8 未发现任何超越随机 ARX 的可利用结构弱点，正式结案为 **`NULL`**。

---

## 6. 交付文件索引

- 数学契约：[SPEC.md](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/SPEC.md)
- 最终裁决：[verdict.json](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/verdict.json)
- 结案报告：[REPORT.md](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/REPORT.md)
- 工程源码：[control_cipher.py](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/control_cipher.py)、[anisotropic_homotopy.py](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/anisotropic_homotopy.py)、[smoke_invariants.py](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/smoke_invariants.py)、[run_p8_experiments.py](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/run_p8_experiments.py)
- 原始实验产物：[invariants_smoke.json](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/results/invariants_smoke.json)、[homotopy_sweep.json](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/results/homotopy_sweep.json)、[boundary_2d_compare.json](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/results/boundary_2d_compare.json)、[cutset_ablation.json](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/results/cutset_ablation.json)、[run_log.txt](file:///g:/SHA-LM/v11_discovery/p8_anisotropic_homotopy/results/run_log.txt)
