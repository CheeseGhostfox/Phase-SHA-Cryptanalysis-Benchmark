# P8「各向异性 / 复平面反绝热同伦分岔测绘」数学契约与实验规范 (`SPEC.md`)

> **工作目录**：`g:\SHA-LM\v11_discovery\p8_anisotropic_homotopy\`（独占）  
> **只读参考接口（严禁修改）**：`g:\SHA-LM\v6a\v6a_homotopy.py`、`g:\SHA-LM\v6a\v6a_gl_rrr.py`、`g:\SHA-LM\v6a\__init__.py`、`g:\SHA-LM\src\sha256_ref.py`、`g:\SHA-LM\src\v5d_rrr.py`、`g:\SHA-LM\src\dual_basis.py`  
> **核心科学问题**：V6A 已证明在 $\alpha = 0$ 全 $\mathbb{F}_2$ 线性流形上真实原像 $x^*$ 能量精确为 $0$，且在标量均匀同伦 $\alpha(s) = s \in [0, 1]$ 上 $x^*$ 严格是不动点；然而一旦 $\alpha$ 越过临界阈值 $\alpha_c$，系统便爆发分岔湍流。本路线将标量 $\alpha$ 升维为**时空各向异性门控场 $\alpha(t, g, b; s)$** 与**反绝热复平面绕行调度 $(\alpha_{\text{re}}, \alpha_{\text{im}})$**（纯实数 $2 \times 2$ 矩阵表示 / 双伴随场），精确测绘跨轮次 $R \in \{4, 6, 8, 12, 18, 24, 64\}$ 的分岔临界线、GL-RRR 漂移率、固定边界能量跃迁，并定位触发混沌回灌的最小割集与奇点，检验能否在真实现实与同构随机 `ROTR/K` 对照组之间显著推迟或绕开分岔相变。

---

## 0. 硬性精度与资源约束

1. **纯实数 FP32 锁死**：
   - 所有脚本与评估模块必须在入口调用 `enforce_v6a_fp32()`（`torch.backends.cuda.matmul.allow_tf32 = False`、`torch.backends.cudnn.allow_tf32 = False`、`torch.set_float32_matmul_precision("highest")`）。
   - 全部状态张量、网络参数、复平面 $2\times 2$ 实数矩阵展开张量严格使用 `torch.float32`（2-adic 相位核 `P_T_fp64`、`theta_K_fp64` 沿用 V6A 只读缓冲区除外）；**严禁使用 `torch.complex64` / `complex128` / FP16 / BF16 / TF32**。
2. **禁止本地 GPU**：
   - 本地环境仅允许在 `CUDA_VISIBLE_DEVICES=""` 下执行秒级 CPU 语法检查与不变性冒烟测试；全部扫频、神经各向异性场优化、最小割集消融必须在 Colab GPU（会话名 `v11-p8`，L4 优先，T4 兜底，单会话总时长 $\le 45$ 分钟）完成。
3. **只读隔离**：
   - 严禁修改 `v6a/`、`src/`、`v10_ouroboros64/`、`runs/`、`formal_proofs/`、`formal_proofs_v10/` 及其他 `pX` 目录；所有新代码、实验产物与报告仅写入 `g:\SHA-LM\v11_discovery\p8_anisotropic_homotopy\`。

---

## 1. 与 V6A 同源的电路骨架、各向异性门控场与数学不变性契约

### 1.1 电路变量与复制体分解（完全复用 V6A）
直接复用 `v6a/v6a_homotopy.py` 的 `build_v6a_circuit_spec(rounds=R)`（支持 $R \in [4, 64]$）：
- **变量位向量 $v \in \mathbb{R}^{n_{\text{var\_bits}}}$**（$n_{\text{var\_bits}} = 32 \cdot n_{\text{var\_words}}$）：包含双轨状态字 $A_{-3..R}, E_{-3..R}$、消息字 $W_{0..R-1}$、常量 $K_{0..R-1}, \text{IV}_{0..7}, H_{0..7}, \text{CONST}_0$、轮次辅助字 `aux_round[t]`（`S1, Ch, S0, Maj, u1, u2, u3, T1, T2`）、消息扩展辅助字 `aux_sched[t-16]`（`s1, s0, v1, v2`）以及每个 `ADD32` 门的 31-bit 显式进位松弛字 `carry`（第 31 位结构性锁定为逻辑 `0` / 自旋 `+1.0`）。
- **Set-A 全局复制体向量 $x_{\text{rep}} = (r_{\text{lin}}, r_{\text{nl}}) \in \mathbb{R}^{n_{\text{rep\_bits}}}$**：
  - `r_lin`（$n_{\text{var\_bits}}$）：承受完整 $\mathbb{F}_2$ 线性骨架约束 $H_{\text{lin}} u = 0 \pmod 2$（涵盖全部 $\Sigma_0, \Sigma_1, \sigma_0, \sigma_1$ 与全部 `ADD32` 逐比特和校验 $a_b \oplus b_b \oplus c_{b-1} \oplus s_b = 0$）。
  - `r_nl`（$n_{\text{nl\_rep}}$）：按门顺次拼接三类非线性门复制体：
    1. **`Ch` 门组**（$N_{\text{ch}} = 32 R$ 个 4-元布尔真值表，按 $(t, b) \in \{0,\dots,R-1\} \times \{0,\dots,31\}$ 索引）；
    2. **`Maj` 门组**（$N_{\text{maj}} = 32 R$ 个 4-元布尔真值表，按 $(t, b) \in \{0,\dots,R-1\} \times \{0,\dots,31\}$ 索引）；
    3. **`ADD32 + Carry` 门组**（$N_{\text{add}} = 7 R + 3 \max(0, R - 16) + 8$ 个 32-bit 字四元组 $(a_w, b_w, s_w, c_w)$，展开为 $(N_{\text{add}}, 4, 32)$ 比特索引）。
- **边界与无作弊约束**：
  - 主测绘统一采用 **`domain_mode="free_msg"`、`allow_peel_back=False`、`mask_m=256`**（全 256-bit 摘要锁定），彻底杜绝 $R \le 16$ 的代数剥壳（peel-back）捷径。
  - 全部 $\alpha=0$ 初态严格由 `solve_alpha0_gf2_linear_skeleton`（稀疏单元传播 + 精确 Python-int GF(2) RREF）在给定锁定摘要边界下生成，保证初始线性违约精确为 $0$。

### 1.2 时空各向异性门控场 $\alpha(t, g, b; s)$ 定义
定义连续同伦参数 $s \in [0, 1]$ 下的三维张量门控场：
$$\alpha(t, g, b; s) \in [0, 1], \quad t \in \{0, \dots, R-1\},\; g \in \{\text{Ch}, \text{Maj}, \text{Add}\},\; b \in \{0, \dots, 31\}$$
对于 `ADD32` 门中的三类来源，轮次映射 $t(j) \in \{0, \dots, R-1\}$ 确定性定义为：
- 前 $7R$ 个轮次压缩加法（`u1, u2, u3, T1, T2, etp1, atp1`）：$t(j) = \lfloor j / 7 \rfloor \in \{0,\dots,R-1\}$；
- 消息扩展加法（$t_{\text{sched}} \in [16, R-1]$，每轮 3 个 `v1, v2, wt`）：$t(j) = t_{\text{sched}} \in \{16,\dots,R-1\}$；
- 末态前馈加法（$H_i = S_{R,i} + \text{IV}_i$，共 8 个）：统一归属末轮 $t(j) = R - 1$（且在门类消融中支持单独标记为 `ff` 子槽）。

由此将 $(t, g, b)$ 广播至与 `r_nl` 同构的复制体门控权重向量 $\boldsymbol{\alpha}_{\text{rep}}(s) \in [0, 1]^{n_{\text{nl\_rep}}}$：
- 对 `Ch` 四元组 $(t, b)$ 的 4 个复制体赋予 $\alpha(t, \text{Ch}, b; s)$；
- 对 `Maj` 四元组 $(t, b)$ 的 4 个复制体赋予 $\alpha(t, \text{Maj}, b; s)$；
- 对 `ADD32` 第 $j$ 门第 $b$ 比特的 4 个分量 $(a_b, b_b, s_b, c_b)$ 赋予 $\alpha(t(j), \text{Add}, b; s)$。

**各向异性非线性投影算子 $P_B^{(\text{nl}, \boldsymbol{\alpha})}$**：
先计算精确局部投影 $p_B^{(\text{nl})} = P_B^{(\text{nl})}(r_{\text{nl}})$（即 `project_ch`、`project_maj` 与 `viterbi_add32_with_carry_project`），再执行逐复制体凸插值：
$$P_B^{(\text{nl}, \boldsymbol{\alpha})}(r_{\text{nl}}) = \big(\mathbf{1} - \boldsymbol{\alpha}_{\text{rep}}(s)\big) \odot r_{\text{nl}} + \boldsymbol{\alpha}_{\text{rep}}(s) \odot P_B^{(\text{nl})}(r_{\text{nl}})$$
而线性骨架投影 $P_B^{(\text{lin})}$ 在所有 $s \in [0, 1]$ 上保持全强度（零校验子守卫 + GPU 阻尼 Min-Sum BP）。

### 1.3 反绝热复平面绕行调度（纯实数 $2 \times 2$ 矩阵 / 双伴随场表示）
为规避实轴 $\alpha \in [0, 1]$ 上的鞍点/折返奇点，将非线性同伦耦合复平面化：
$$\tilde{\boldsymbol{\alpha}}(s) = \boldsymbol{\alpha}_{\text{re}}(s) + i\,\boldsymbol{\alpha}_{\text{im}}(s), \qquad \boldsymbol{\alpha}_{\text{re}}(0)=\mathbf{0},\; \boldsymbol{\alpha}_{\text{re}}(1)=\mathbf{1},\; \boldsymbol{\alpha}_{\text{im}}(0)=\boldsymbol{\alpha}_{\text{im}}(1)=\mathbf{0}$$
严禁使用 `complex64`，改用等价的**纯实数 FP32 双通道（双伴随场）表示** $\mathbf{x} = (x_{\text{re}}, x_{\text{im}}) \in \mathbb{R}^{n_{\text{rep\_bits}}} \times \mathbb{R}^{n_{\text{rep\_bits}}}$（初始 $x_{\text{im}} = \mathbf{0}$）：
1. **Set-A 双通道共识投影**：
   - 实部 $y_A^{\text{re}} = P_A(x_{\text{re}}; \text{mask}, \text{vals})$（复用 V6A 共识加权与边界锁定）；
   - 虚部/伴随部 $y_A^{\text{im}} = P_A^{\text{adj}}(x_{\text{im}}; \text{mask})$：按同构边度对变量做共识平均，并将所有**锁定边界比特强制置 `0.0`**（保证外部边界无虚部漂移）。
   - 反射点：$r_{\text{re}} = 2 y_A^{\text{re}} - x_{\text{re}}$，$r_{\text{im}} = 2 y_A^{\text{im}} - x_{\text{im}}$。
2. **Set-B 线性骨架双通道投影**：
   - 实部 $y_B^{\text{lin, re}} = P_B^{(\text{lin})}(r_{\text{re}})$；
   - 虚部采用线性切空间阻尼投影 $y_B^{\text{lin, im}} = (1 - \kappa_{\text{im}}(s))\, r_{\text{im}}$（其中 $\kappa_{\text{im}}(s) \in [0.15, 1.0]$ 且强制 $\kappa_{\text{im}}(1) = 1.0$）。
3. **Set-B 非线性 $2 \times 2$ 实矩阵旋转耦合**：
   记精确非线性残差力 $\Delta_{\text{nl}}(r_{\text{re}}) = P_B^{(\text{nl})}(r_{\text{re}}) - r_{\text{re}}$，经复数乘法实矩阵 $M(\boldsymbol{\alpha}_{\text{re}}, \boldsymbol{\alpha}_{\text{im}}) = \begin{pmatrix} \boldsymbol{\alpha}_{\text{re}} & -\boldsymbol{\alpha}_{\text{im}} \\ \boldsymbol{\alpha}_{\text{im}} & \boldsymbol{\alpha}_{\text{re}} \end{pmatrix}$ 耦合双伴随反馈（阻尼系数 $\lambda_{\text{cd}} > 0$）：
   $$\begin{pmatrix} y_B^{\text{nl, re}} \\ y_B^{\text{nl, im}} \end{pmatrix} = \begin{pmatrix} r_{\text{re}} \\ (1 - \kappa_{\text{im}}(s))\, r_{\text{im}} \end{pmatrix} + \begin{pmatrix} \boldsymbol{\alpha}_{\text{re}}(s) & -\boldsymbol{\alpha}_{\text{im}}(s) \\ \boldsymbol{\alpha}_{\text{im}}(s) & \boldsymbol{\alpha}_{\text{re}}(s) \end{pmatrix} \begin{pmatrix} \Delta_{\text{nl}}(r_{\text{re}}) \\ -\lambda_{\text{cd}}\, r_{\text{im}} \end{pmatrix}$$
4. **反绝热规范速度阻尼（Counter-Diabatic Velocity Damping）**：
   在连续推进步 $s_{k-1} \to s_k$ 间加入可切换的绝热规范势阻尼修正（用于消融纯阻尼 vs 复平面旋转的各自贡献）：
   $$x_{\text{re}}^{(t+1)} = x_{\text{re}}^{(t)} + \beta \big(y_B^{\text{re}} - y_A^{\text{re}}\big) - \eta_{\text{cd}}\, \mathbf{1}_{\text{free}} \odot \big(x_{\text{re}}^{(t)} - x_{\text{re}}^{(t-1)}\big) \cdot \|\dot{\alpha}(s_k)\|$$
   仅在瞬时更新内使用前一步自由位差分；读出与真迹核验严格取自实部影态 $P_A(x_{\text{re}})$。

### 1.4 两大铁律不变性（必须通过自动化断言验证）
任何合法路径家族 $\mathcal{P}$（含随机/学习调度）必须在本地冒烟与 Colab 全量验证中满足：
1. **端点 $s = 1$ 严格还原精确 SHA-256**：
   - $\alpha_{\text{re}}(t, g, b; 1) \equiv 1.0$，$\alpha_{\text{im}}(t, g, b; 1) \equiv 0.0$，$\kappa_{\text{im}}(1) \equiv 1.0$，$\eta_{\text{cd}}(1) \equiv 0.0$；
   - 因此在 $s = 1$ 上算子 $P_B(\cdot; s=1)$ 与 `v6a_gl_rrr.py` 的标准 $P_B(\cdot; \alpha=1.0)$ **逐比特数学恒等**（实部离散违约、2-adic 能量 $E_{32}$ 与 `sha256_digest` 校验零偏差）。
2. **真迹 $x^*$ 全程精确不动点守恒**：
   - 对任意 $w_{16}$，令 $x^* = \text{embed\_ground\_truth\_vars}(w_{16}, R)$，构造 $\mathbf{x}^* = (\text{init\_replicas}(x^*), \mathbf{0})$。
   - 由于 $x^*$ 满足全部 GF(2) 校验与全部局部非线性门，故 $\Delta_{\text{nl}}(x^*_{\text{nl}}) = \mathbf{0}$ 且 $P_B^{(\text{lin})}(x^*_{\text{lin}}) = x^*_{\text{lin}}$。
   - 因此对**任意 $s \in [0, 1]$**、**任意 $\boldsymbol{\alpha}_{\text{re}} \in [0, 1]^{n_{\text{nl\_rep}}}$** 与**任意 $\boldsymbol{\alpha}_{\text{im}} \in \mathbb{R}^{n_{\text{nl\_rep}}}$**：
     $$P_A(\mathbf{x}^*) = \mathbf{x}^*, \qquad P_B(\mathbf{x}^*; s) = \mathbf{x}^* \implies \|\mathbf{x}_{t+1} - \mathbf{x}^*\|_\infty = 0$$
   - 冒烟测试必须对全部 9 条路径在 $s \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$ 实测 $\max |\mathbf{x}_{t+1} - \mathbf{x}^*| < 10^{-6}$。

---

## 2. 待对比同伦路径家族矩阵（9 条确定性与神经调度路径）

记平滑线性坡函数 $\text{ramp}(u) = \text{clip}(u, 0, 1)$。所有路径在 $s=0$ 均精确等于 $\mathbf{0}$，在 $s=1$ 均精确等于 $\mathbf{1}$（虚部均满足 $\alpha_{\text{im}}(0) = \alpha_{\text{im}}(1) = 0$）：

| 路径 ID | 类型 | 实部门控场 $\alpha_{\text{re}}(t, g, b; s)$ | 虚部/反绝热场 | 物理/密码学探测意图 |
| :--- | :--- | :--- | :--- | :--- |
| **`P0_uniform`** | 标量基线 | $s$ | $0$ | 复现 V6A 均匀同伦标尺与临界线 $\alpha_c$ |
| **`P1_bit_lsb_to_msb`** | 2-adic 低频波前 | $\text{ramp}(32s - b)$ | $0$ | 从 $b=0$ 到 $b=31$ 逐比特打开进位，检验低位先行能否抑制高阶进位雪崩 |
| **`P2_bit_msb_to_lsb`** | 逆序高位波前 | $\text{ramp}(32s - (31 - b))$ | $0$ | 从 $b=31$ 到 $b=0$ 逆向打开，与 P1 形成 2-adic 因果方向对照 |
| **`P3_round_forward`** | 轮次前向波前 | $\text{ramp}(Rs - t)$ | $0$ | 从初态 IV 侧 $t=0$ 向摘要侧 $t=R-1$ 逐轮开放非线性 |
| **`P4_round_backward`** | 轮次后向波前 | $\text{ramp}(Rs - (R - 1 - t))$ | $0$ | 从锁定摘要 $t=R-1$ 向 $t=0$ 逆向开放，测试尾部强约束先收敛能否压制分岔 |
| **`P5a_gate_bool_then_add`** | 分门类（布尔先） | `Ch, Maj`: $\text{ramp}(2s)$；`Add`: $\text{ramp}(2s - 1)$ | $0$ | 先激活局部布尔门 `Ch/Maj`，再开启长程模加进位 `Add` |
| **`P5b_gate_add_then_bool`** | 分门类（进位先） | `Add`: $\text{ramp}(2s)$；`Ch, Maj`: $\text{ramp}(2s - 1)$ | $0$ | 先开启模加进位 `Add`，再开启 `Ch/Maj`，剥离两类非线性的致湍权重 |
| **`P6_cd_complex_detour`** | 复平面反绝热绕行 | $s$ | $\alpha_{\text{im}} = \gamma_0 \sin(\pi s)$（主取 $\gamma_0=0.40$，并做 $\gamma_0 \in \{0.2, 0.4, 0.7\}$ 灵敏度子测）+ $\lambda_{\text{cd}}=0.5, \eta_{\text{cd}}=0.15$ | 检验复平面 $\sin(\pi s)$ 绕行 + 双伴随阻尼能否绕开实轴分岔奇点 |
| **`P7_learned_aniso`** | 神经可微学习场 | 参数化单调样条 $\alpha_\theta(t, g, b; s)$ | $\alpha_{\text{im},\theta}(t,g,b;s) = \gamma_{\max}\sin(\pi s)\tanh(\psi_\theta(t,g,b;s))$ | 在线端到端学习最优时空各向异性 + 复绕行轨迹（见第 3 节） |

---

## 3. `P7_learned_aniso` 可微各向异性调度场架构与训练契约

### 3.1 端点锁死的单调时空样条参数化
为从结构上保证对任意权重 $\theta$ 均严格满足 $\alpha_\theta(t, g, b; 0) \equiv 0$、$\alpha_\theta(t, g, b; 1) \equiv 1$ 及时间单调性 $\partial_s \alpha_\theta \ge 0$：
- 设分箱数 $K_{\text{bin}} = 8$（节点 $s_m = m / 8,\; m=0,\dots,8$）。
- 对每个门坐标 $(t, g, b)$ 维护可学习对数步长张量 $L_\theta \in \mathbb{R}^{R \times 3 \times 32 \times 8}$（按全零初始化，使初始场精确等于 `P0_uniform` $\alpha(s)=s$！）：
  $$w_{t,g,b,:} = \text{softmax}_k\big(L_\theta[t, g, b, :]\big) \in (0, 1)^8, \qquad A_{t,g,b}(s_m) = \sum_{k=0}^{m-1} w_{t,g,b,k}$$
  对任意 $s \in [s_m, s_{m+1}]$ 做线性插值得到 $\alpha_{\text{re},\theta}(t, g, b; s) \in [0, 1]$。
- 复绕行相位头 $\psi_\theta \in \mathbb{R}^{R \times 3 \times 32 \times 8}$ 同样从 $0$ 初始化，在节点间线性插值后乘以端点归零窗 $\gamma_{\max}\sin(\pi s)$（$\gamma_{\max} = 0.5$），严格满足 $\alpha_{\text{im},\theta}(0) = \alpha_{\text{im},\theta}(1) = 0$。

### 3.2 纯 FP32 可微展开（Unrolled STE-RRR）优化协议
- **严禁原像泄漏**：训练仅使用独立随机种子生成的在线目标摘要（`sha256_digest` 仅用于生成输入目标摘要，网络看不到真实 $w_{16}^*$，且测试使用完全隔离的留出种子摘要）。
- **可微梯度通路**：
  在推进网格 $s_\ell = \ell / L$（$L = 8$，每层展开 $T_{\text{unroll}} = 4$ 步）上：
  - `project_A` 与 $(1 - \boldsymbol{\alpha}_{\text{re},\theta}) r_{\text{nl}} + \boldsymbol{\alpha}_{\text{re},\theta} p_B^{(\text{nl})}$ 均为 $\boldsymbol{\alpha}_{\text{re},\theta}, \boldsymbol{\alpha}_{\text{im},\theta}$ 的**精确可微仿射/双线性函数**（保留局部投影 $p_B^{(\text{nl})}$ 停梯度分支上的显式梯度 $\frac{\partial y_{\text{nl}}}{\partial \boldsymbol{\alpha}_{\text{re}}} = p_B^{(\text{nl})} - r_{\text{re}}$ 与 $\frac{\partial y_{\text{nl}}}{\partial \boldsymbol{\alpha}_{\text{im}}}$，同时通过 `project_A` 和 $(1 - \boldsymbol{\alpha})$ 回传跨步长时序梯度）。
  - 训练目标最小化**分岔抑制与连续平稳度联合损失**：
    $$\mathcal{L}(\theta) = \frac{1}{L}\sum_{\ell=1}^{L} \Big(\text{gl\_gap}(s_\ell)^2 + 0.5\, E_{\text{hom}}(s_\ell)\Big) + 2.0 \sum_{\ell=1}^{L-1} \Big[\max\big(0,\; E_{\text{hom}}(s_{\ell+1}) - 1.05 E_{\text{hom}}(s_\ell)\big)\Big]^2 + 0.25\, \text{Var}_{\text{step}}\big(\|x_{t+1} - x_t\|_2\big)$$
  - 优化器：`Adam(lr=3e-2)`，每轮数 $R$ 在独立训练摘要上优化 `100` 步（`B = 8` 摘要 $\times$ `8` 轨 = `64` 轨；`R = 64` 时取 `4` 摘要 $\times$ `8` 轨 = `32` 轨），随后冻结 $\theta$ 在**隔离留出摘要集**上按第 4 节完全一致的协议测绘！

---

## 4. 扫频轮梯、分岔动力学指标与最小割集定位规范

### 4.1 扫频轮数与同构随机 `ROTR/K` 对照组
- **扫频轮数（7 个全覆盖点）**：
  $$R \in \{4,\; 6,\; 8,\; 12,\; 18,\; 24,\; 64\}$$
  其中 $R=4$ 承担阳性校准角色，$R \ge 8$ 为核心非线性混沌分岔区，$R=64$ 验证全轮次极限缩放规律。
- **同构随机 `ROTR/K` 克隆对照组 (`control`)**：
  构建完全同构的 `sha256_trace_control(w16, rounds=R)` 与对应 `V6ACircuitSpecControl`：
  1. 轮常数 $K_{\text{ctrl}}[0..63]$ 在固定种子下 i.i.d. 均匀采自 $\mathbb{Z}_{2^{32}}$；
  2. $\Sigma_0, \Sigma_1$ 保持 3 个 `ROTR` 偏移（从 $\{1,\dots,31\}$ 无放回重采样且不等于 SHA-256 标准三元组）；$\sigma_0, \sigma_1$ 保持 2 个 `ROTR` + 1 个 `SHR`（`ROTR` 从 $\{1,\dots,31\}$ 重采样且互异，`SHR` 从 $\{1,\dots,15\}$ 重采样且互异，均不等于标准值）；
  3. 保留 `IV`、`Ch`、`Maj`、全部 `ADD32` 拓扑与 8 字双轨寄存器结构不变；并同步校验克隆真迹 $x^*_{\text{ctrl}}$ 在克隆电路上满足 $\alpha \in [0,1]$ 不动点守恒。

### 4.2 分岔阈值 $\alpha_c$、GL-RRR 漂移率与固定边界能量跃迁测绘协议
对每个 $(R, \text{variant} \in \{\text{main}, \text{control}\}, \mathcal{P} \in \{\text{P0..P7}\})$：
- **隔离评估样本**：`N_digests = 8`（`R=64` 取 `4`）个独立留出随机消息生成的目标摘要，每个摘要并行演化 `K_traj = 16` 条 $\alpha=0$ GF(2) 精确解轨迹（共 $B = 128$ 轨；`R=64` 为 $64$ 轨），`beta = 0.85`（或轨迹带 `[0.75, 0.95]` 均匀分布，主对照一致）。
- **准静态 $s$-扫描网格**：$s_k = k / 25$（$k = 0, 1, \dots, 25$，$\Delta s = 0.04$），在每个水平 $s_k$ 上固定运行 `T_relax = 8` 步纯 GPU GL-RRR（`bp_iters=7`, 扫频中段关闭 CPU OSD-0 防止 `R=64` 阻塞，仅在 $s=0$ 验证线性零违约；并在 $s=1.0$ 结束后额外运行 1 次终局 `readout_shadow`）。
- **各向异性同伦能量 $E_{\text{hom}}(v_{\text{disc}}; s)$ 定义**（在 $\boldsymbol{\alpha} \equiv \alpha \mathbf{1}$ 时严格退化为 V6A `inst_E_hom`）：
  $$E_{\text{hom}}(s) = \text{lin\_viol\_rate} + \frac{1}{3}\Big(\bar{e}_{\text{Ch}}(\boldsymbol{\alpha}_{\text{re}}) + \bar{e}_{\text{Maj}}(\boldsymbol{\alpha}_{\text{re}}) + \bar{e}_{\text{Add}}(\boldsymbol{\alpha}_{\text{re}})\Big)$$
  其中每项加权违约率按该门在该比特上的当前实部激活度 $\alpha_{\text{re}}(t,g,b;s)$ 加权平均；同时独立记录**未加权全电路真实违约率** $E_{\text{full}}(s) = \text{lin\_viol\_rate} + \text{nl\_err\_rate}$、**真实 32 尺度 2-adic 能量** `inst_E32_true` 与 `inst_E_all32`。
- **分岔触发判据与防作弊双指标 $(s_c, \bar{\alpha}_c)$**：
  在连续推进 $s_k$（$k \ge 1$）中，首个同时/任一触发以下物理失稳信号的位置记为分岔点 $s_c$（若全程未触发则记 `bifurcated=False`, $s_c = \text{null}$）：
  1. **能量尖峰跃迁**：$E_{\text{hom}}(s_k) > 1.25\, E_{\text{hom}}(s_{k-1})$ 且 $E_{\text{hom}}(s_k) - E_{\text{hom}}(s_{k-1}) > 0.005$；
  2. **GL 相隙/漂移爆发**：`gl_gap`$(s_k) > 0.14$ 或线性漂移斜率 `drift_slope`$(s_k) > 10^{-2}$（`DriftMonitor`）；
  3. **线性骨架破裂**：`lin_viol_rate`$(s_k) > 0.02$（非线性回灌撕裂 $\mathbb{F}_2$ 线性支撑）。
  **强制同时汇报有效门开启密度**：
  $$\bar{\alpha}_c = \frac{1}{|\mathcal{G}_{\text{nl}}|}\sum_{(t,g,b)} \alpha_{\text{re}}(t, g, b; s_c)$$
  （严禁仅凭推迟时间参数 $s_c$ 而不检查 $\bar{\alpha}_c$ 宣称绕开分岔！）
- **固定边界能量跃迁与 2D 对比**：
  - 固定边界 `mask_m = 256` 下记录全路径总跃迁 $\Delta E_{\text{hom}} = E_{\text{hom}}(1) - E_{\text{hom}}(0)$ 与最大单步斜率 $\Delta E_{\max} = \max_k (E_{\text{hom}}(s_k) - E_{\text{hom}}(s_{k-1}))$，以及末态 $s=1$ 的 `drift_slope`、`gl_gap`、`inst_E32_true`、`disc_viol_min` 与 `exact_hits`。
  - 另在代表轮次 $R \in \{4, 8, 18\}$ 上追加 **2D 边界松弛对比**（`mask_m: 64 -> 256` 交错 vs 固定 `mask_m=256`），量化边界锁定本身贡献的跃迁份额。

### 4.3 最小进位回灌割集与触发奇点定位协议
在所有轮次 $R \in \{4, 6, 8, 12, 18, 24, 64\}$（`main` 与 `control` 同步）上，从 $\alpha \equiv \mathbf{0}$ 精确线性流形出发执行三层精确定位：
1. **单组件注入奇点扫描（Single-Component Singularity Probe）**：
   保持其余全部门 $\alpha = 0$，仅将单个门类（`only_Ch`, `only_Maj`, `only_Add`）、单个 `ADD32` 槽位（`u1, u2, u3, T1, T2, E_next, A_next`, `sched`, `ff`）或单个比特带（4-bit nibble band $b \in [4m, 4m+3]$）设为 $\alpha = 1$，运行 `16` 步 GL-RRR，测绘 `lin_viol_rate`、`gl_gap`、`drift_slope` 敏感度张量 $S_{\text{inject}}(g, \text{slot}, b)$，识别首轮非线性冲突触发源。
2. **反事实消融与最小回灌割集搜索（Minimal Feedback Cut-Set $\mathcal{C}_{\min}$）**：
   从全激活状态 $\alpha \equiv \mathbf{1}$ 逆向切断特定子集 $\mathcal{C}$（令 $\alpha|_{\mathcal{C}} = 0$，其余全部门保持 $\alpha = 1$），测绘以下结构性割集恢复线性流形（`lin_viol_rate < 0.02` 且 `drift_slope < 1e-2`）的能力：
   - **比特位阻断割（Bit-Carry Cut）**：按 $b$ 切掉单比特或连续比特窗（如每隔 $k$ 比特切断一次 `ADD32` 进位链，或切断最高/最低若干比特），测绘进位长程相关性截断所需的最小比特割密度；
   - **轮次截断割（Round-Depth Cut）**：依次从尾部/中部切断 $w \in \{1, 2, \dots\}$ 轮或仅激活前 $R_{\text{act}} \in \{1,\dots,R\}$ 轮，精确测绘分岔渗流阈值轮深 $R_{\text{perc}}$；
   - **贪心最小门割集（Greedy Minimal Cut-Set）**：基于一阶敏度 $S(t, g, b)$ 从大到小逐步将 $(t, g, b)$ 置 `0`（步长按块二分细化），直至 `lin_viol_rate < 0.02` 且 `gl_gap < 0.14`，记录最小割集占比 $\rho_{\text{cut}} = |\mathcal{C}_{\min}| / N_{\text{nl\_bits}}$ 及其在 $(t, g, b)$ 空间中的支撑分布。

---

## 5. 算力预算与裁决门限（Colab L4/T4 $\le 45$ 分钟）

### 5.1 运行时长预算规划
- `R ∈ {4, 6, 8, 12, 18, 24}`（6 轮 $\times$ 2 变体 $\times$ 9 路径 $\times$ 200 步，$B=128$）+ `R=64`（2 变体 $\times$ 9 路径 $\times$ 200 步，$B=64$）= `25,200` RRR 步，预计 ~3–5 分钟；
- `P7_learned_aniso` 在 7 轮 $\times$ 2 变体共 14 组上各训练 `100` 步（$L=8 \times 4$ 展开），预计 ~5–8 分钟；
- 最小割集与单组件奇点扫描（7 轮 $\times$ 2 变体）+ 2D 边界对照，预计 ~4–6 分钟；
- 总预估耗时 **12–20 分钟**（远低于 45 分钟单会话上限）；`solve_alpha0_gf2_linear_skeleton` 在每 `(R, variant)` 仅求解 1 次并缓存至全部 9 路径复用，避免重复 CPU RREF 开销。

### 5.2 双向科学裁决门限（Verdict Criteria）
在独立留出目标摘要上，对 $R \ge 8$ 汇总全部路径与对照指标：
- **`HIT`（发现可显著绕开/推迟 SHA-256 分岔的各向异性或复绕行调度，或发现 SHA-256 专属低维割集）**：
  在任意 $R \ge 8$ 上**同时满足**：
  1. 至少一条路径 $\mathcal{P} \in \{\text{P1..P7}\}$ 实现有效门开启密度显著推迟 $\bar{\alpha}_c(\mathcal{P}) - \bar{\alpha}_c(\text{P0}) \ge 0.20$（或全程不触发分岔），且 $s=1$ 终态真实 `inst_E32_true` / `gl_gap` / `exact_hits` 显著优于 `P0_uniform`；
  2. 上述分岔推迟或最小割集占比 $\rho_{\text{cut}}$ 在**真实 SHA-256 vs 同构随机 ROTR/K 对照组**之间存在显著统计学分离（跨独立摘要检验 $|z_{\text{ctrl}}| \ge 4.0$）。
- **`BORDERLINE`（局部相移或通用 ARX 割集结构，未突破普适性或终点瓶颈）**：
  满足以下任一（且未达 `HIT`）：
  1. 在 $R \ge 8$ 上观测到 $\Delta \bar{\alpha}_c \ge 0.15$ 的分岔推迟或能量跃迁缓和，但终态 $s=1$ 仍崩塌至与 `P0_uniform` 同等湍流能量，或真实 SHA-256 与随机 `ROTR/K` 对照组表现一致（$|z_{\text{ctrl}}| < 3.0$）；
  2. 最小割集实验揭示出精确但对 SHA-256 与随机 ARX 普适的进位渗流奇点，或仅在过渡轮 $R=6$ 达显著改善。
- **`NULL`（各向异性与复平面反绝热同伦未推迟/绕开深轮分岔，且 SHA-256 与随机克隆分岔线完全一致）**：
  在全部 $R \ge 8$ 上，没有任何路径实现有效的 $\bar{\alpha}_c$ 推迟与终态能量/命中改善（仅重参数化 $s$ 而在相同 $\bar{\alpha}_c$ 爆发湍流），且真实 SHA-256 与同构随机 `ROTR/K` 对照组的 $\bar{\alpha}_c$、GL-RRR 漂移斜率、固定边界能量跃迁及最小割集占比 $\rho_{\text{cut}}$ 在统计误差内完全一致（$|z_{\text{ctrl}}| < 2.5$）。

---

## 6. 交付物清单（全部落盘于 `v11_discovery/p8_anisotropic_homotopy/`）

1. `@p8-eng` 工程产物：
   - 核心实现与测绘脚本（严格纯 FP32、禁止修改 `v6a/`）
   - `results/invariants_smoke.json`（本地 CPU 冒烟：端点 $s=1$ 精确性 + 真迹 $x^*$ 全路径不动点守恒验证）
   - `results/homotopy_sweep.json`（$R \in \{4,6,8,12,18,24,64\} \times \{\text{main}, \text{control}\} \times \{\text{P0..P7}\}$ 完整分岔曲线、$s_c$、$\bar{\alpha}_c$、漂移率、能量跃迁与终局指标）
   - `results/cutset_ablation.json`（单组件奇点敏感度张量 + 比特/轮次/贪心最小进位回灌割集定位）
   - `results/boundary_2d_compare.json`（$R \in \{4,8,18\}$ 固定 `mask_m=256` vs 2D `mask_m: 64->256` 能量跃迁对照）
   - `results/run_log.txt`（Colab 执行日志与 `colab.exe stop -s v11-p8` 停机核实记录）
2. `@p8-math` 结案产物（待 Colab 数据全量同步且核实停机后撰就）：
   - `verdict.json`（含 `"colab_stopped": true` 与全部核心定量指标）
   - `REPORT.md`（简体中文完整数学剖析与结案报告）
