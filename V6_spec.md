【Phase-SHA V6A 工程实现指令：Ouroboros 全栈反置乱引擎（双同伦 × 全局 GL-RRR × 瓶颈真纠缠 MPS × 高维 BM 逃逸 × 零刹车神经导航）】

在严格不破坏现有 V3 / V4 / V5 代码与测试的前提下，请在 `g:\SHA-LM` 中构建 **Phase-SHA V6A (`v6a/`)**。V6A 彻底终结“随机冷空投 + 局域单门松弛 + 边缘 BCE 互掐”的 L 型死水范式，将双同伦追踪、非局域全局反射、真纠缠瓶颈张量网络、高维正交逃逸与零刹车神经导航叠加为单一闭环破壁系统。

---

### 一、从 V4/V5 尸检提取的五条最高铁律（违者视为验收失败）

1. **彻底废除单比特边缘 BCE**：严禁在主优化目标、MPS 训练或 RRR 引导中使用单比特边缘交叉熵（它已证实会勒死 RRR 步长 $\beta$ 并把 MPS 纠缠熵 $S_{\text{ent}}$ 压成 `0.000`）。
2. **严禁历史极值伪装实时进度**：日志与监控面板必须严格区分 **瞬时活体状态能量 `inst_E32` / `inst_mean_E`** 与 **历史最佳 `hist_min_E32`**，严禁将 running-min 记作 `phys_loss`。
3. **死守纯血 FP32 + 40GB A100 显存安全**：全程 `allow_tf32 = False`、纯实数 `torch.float32`（零 `complex64`、零半精度）；默认云端配置峰值显存严控在 `20–32 GB`（完美兼容 Colab A100-40GB，同时提供 `--gpu_tier 80g` 放大并行轨迹数）。
4. **严禁远端 Colab 联调**：禁止编写或执行任何调用 Colab 授权/远程子进程的脚本；本地仅做纯代码 + RTX 4060 极简验收，云端测速与全量运行由我亲自执行。
5. **向下兼容现有监控面板**：输出 `train.log` 保留面板所需基础字段，同时附加 V6A 专属核心字段：`inst_E32`, `hist_min`, `alpha`, `mask_m`, `S_ent`, `gl_gap`, `bifurc`, `exact_hits`, `dt`, `wall_s`。

---

### 二、V6A 五层叠加引擎核心规格 (`v6a/`)

#### 1. `v6a_homotopy.py` — 2D 双轴连续同伦与精确代数剥离引擎
- **确定性边界剥离（Exact Algebraic Peel-Back）**：给定任意轮数 $R$ 与摘要边界，启动时先在毫秒内执行精确代数反推，将外围完全决定的状态位（如 `R=18` 中 `r17/r15/r14` 的确定性路径与固定常量）硬锁为不可变常量掩码 `clamped_mask`，绝不浪费算力重算外围轮。
- **同伦轴 ①：非线性/进位强度 $\alpha \in [0, 1]$**：
  - 将 32 位模加 `ADD32(a, b)` 参数化为 `XOR(a, b)` 与完整进位链的连续/软门同伦，将 `Ch` / `Maj` 参数化为线性代理到真实非线性门的同伦；
  - **$\alpha = 0$ 零能起跳点**：在 $\alpha = 0$ 时整张 $R$ 轮计算图退化为严格纯 $\mathbb{F}_2$ 线性系统，通过精确 GF(2) 高斯消元在毫秒内求得 `inst_E32 == 0.000` 的完美精确解作为同伦起点！
- **同伦轴 ②：可控 $x \cdot 0$ 摘要掩码光圈 $m \in [32, 256]$**：
  - 支持从仅激活最低位/最浅约束的 $m$ 位摘要开始（其余摘要位乘以 `0` 释放成宽阔解大陆），随追踪稳定逐步收紧光圈至完整 256 位。
- **自适应推进与回退机制**：仅当活跃轨迹群的瞬时能量 `inst_E32 < eps_advance` 时推进 $(\alpha, m)$ 一步；若推进一步后发生剧烈能量跃升（检测到分岔点 `bifurc += 1`），自动减半步长并触发 **L3 瓶颈 MPS 分支分裂 + L4 高维正交逃逸 + L2 GL-RRR 脉冲爆发**。

#### 2. `v6a_gl_rrr.py` — 非局域全局反射重炮（Global-Linear + Exact-Viterbi RRR）
- **升级投影 $P_B$ 为非局域全局算子**：
  - **全局线性 $\mathbb{F}_2$ 骨架投影**：将全图所有旋转、移位、$\Sigma_0, \Sigma_1, \sigma_0, \sigma_1$ 与纯 XOR 关系打包为稀疏奇偶校验矩阵 $H$，在 GPU 上执行批量向量化阻尼置信传播（Batched Damped Min-Sum BP，默认 6–8 轮消息传递）叠加周期性有序统计译码（OSD）投影，使单步反射的信息传播半径覆盖全图而非单个门；
  - **精确 32-bit 进位 Viterbi 投影**：复刻并加速 V5D 中全向量化 $O(32)$ 精确进位链投影（按当前同伦 $\alpha$ 调制进位约束权重，在 $\alpha=1$ 时严格退化为精确模 $2^{32}$ 整数解）；
  - **严格阴影读出与防漂移**：所有能量、校验与候选解严格从共识阴影 $P_A(x)$ 读出，步长硬夹断在 $\beta \in [0.65, 1.15]$（绝不降速锁死），保留线性漂移雷达 `drift_flag`。

#### 3. `v6a_bottleneck_mps.py` — 靶向闭环瓶颈 MPS（纯血 Joint-NLL Born Machine）
- **极速瓶颈聚焦（解决 5s/step 性能灾难）**：不再对全图几千变量建链！自动提取闭环冲突最密集的核心切面子集（$W_0$ 的 32 bit + `r13/r16` 及其关键进位耦合变量，控制在 `64–128` 节点、默认键维 $\chi \in \{8, 16\}$），全部使用批量小矩阵 `torch.linalg.qr` 与并行双体门收缩，确保单次更新开销 `< 50 ms`。
- **真·多体纠缠动力学（彻底根治 `S_ent = 0.000`）**：
  - 随机满秩高纠缠初始化（强制起步 `S_ent > 0.5`）；
  - **纯联合波函数训练与投影**：直接对当前同伦阶段收集到的高价值低能候选轨迹集合 $\{x^{(k)}\}$ 优化精确**联合负对数似然（Joint NLL）** $-\log \frac{|\langle x^{(k)} | \psi \rangle|^2}{\langle \psi | \psi \rangle}$，并直接将局部闭环门作为多体投影张量作用在波函数上；
  - **分岔期精确条件采样**：遇到分岔壁垒时，通过精确自回归链式坍缩 $P(s_i \mid s_{<i})$ 从纠缠态一次性吐出多条互斥但各自内部自洽的分支种子，注入 GL-RRR 轨迹池并发探索。

#### 4. `v6a_bm_escape.py` — Burer-Monteiro $\mathbb{S}^{N-1}$ 高维正交逃逸层
- 维护轻量高维辅助嵌入 $V \in \mathbb{S}^{N-1}$（默认 $N=16$，锚定实轴 $e_0$），不直接跑慢速全图梯度，而是作为 **Stagnation Triggered Escape**：当任意轨迹的瞬时 `gl_gap` 连续 $T_{\text{stag}}$ 步降幅小于阈值时，计算其低维主子空间的正交补空间，施加受控半角有理 Cayley 切向踢腿并映射回 RRR 状态空间，强行把死在玻璃坑里的粒子掀出鞍点。

#### 5. `v6a_navigator.py` — 零刹车神经导航员（Predictor + VSIDS 聚火）
- **Predictor 头（切空间预测器）**：轻量残差 MLP/图聚合层，输入当前同伦点状态与参数增量 $\Delta \alpha$，预测新同伦点下的解轨初值 $x(\alpha + \Delta \alpha)$，用真实校正后位移 $\|x_{\text{corr}} - x_{\text{pred}}\|_2^2$ 自监督在线训练（全程零 BCE）。
- **VSIDS 冲突加权头**：根据各变量近期在 GL-RRR 里的投影间隙震荡频率，动态输出共识投影 $P_A$ 的正权重 $w_i \in [0.5, 4.0]$，把全局反射动能集中压向 `r13 / r16` 等核心冲突闭环，且**物理硬锁死无权修改全局步长 $\beta$ 下限**。

---

### 三、Phase-0 临界相变雷达与主入口 (`train_v6a.py`)

1. **强制内置 Phase-0 扫频标定模式 (`--mode calibrate`)**：
   - 自动按顺序对 `R = 4, 6, 8, 10, 12` 跑极速标定（每轮上限 2 分钟，找到精确 `inst_E32 == 0.000` 原像即提前报捷晋级）；
   - 输出 `calibration_report.json` 与 `phase_transition_curve.png`，明确标出 V6A 在哪个临界轮数 $R_c$ 出现首次分岔断裂、真实破壁时的瞬时能量闪崩形态，以及各模块消融贡献（纯 GL-RRR vs +同伦 vs +MPS分岔 vs +BM逃逸）。
2. **主攻模式 (`--mode attack --rounds 18`)**：
   - 默认加载 `R=18` 统一留出考卷（并兼容 `R=20` 与 `v4_boundary.py` 前导零掩码），支持 `--resume` 断点无损续跑与分岔快照保存。
3. **严格自动化单测 (`tests/test_v6a.py`)**：
   - 验证 $\alpha=0$ 纯线性 SHA-256 精确 GF(2) 求解在 `R=4` 和 `R=18` 上均达到 `inst_E32 == 0.0`（零误差）；
   - 验证 GL-RRR 全局线性 BP 投影在多跳 XOR 链上的单步收敛速度显著快于单门投影，且真迹为严格不动点；
   - 验证瓶颈 MPS 在闭环 Bell/GHZ 约束下 `S_ent > 0.5`（绝不坍缩回 `0.000`），且 Joint-NLL 梯度无 NaN、条件采样 100% 合法；
   - 验证 BM 逃逸踢腿严格保范且不破坏外围 `clamped_mask`；
   - **硬核活体通关验收**：在本地 RTX 4060 上，必须通过 **`R=4` 完整闭环精确求逆测试（`exact_hits > 0` 且真实 SHA-256 前向哈希 100% 验真通过）**，绝不允许只靠假指标过关！
4. **交付物**：
   - 本地 4060 跑通全部 `tests/test_v6a.py` 与 `R=4` 精确破壁验真 + `R=18` 20 步极简显存/速度冒烟；
   - 打包自包含 `colab_bundle_v6a/`，附带 Colab A100-40GB / 80GB 一键启动命令（先自动跑 10 分钟 `calibrate` 扫出 $R_c$，紧接着无缝点火 `attack --rounds 18`）。

### 执行纪律
严格按 **`v6a_homotopy.py` → `v6a_gl_rrr.py` → `v6a_bottleneck_mps.py` → `v6a_bm_escape.py` & `v6a_navigator.py` → `train_v6a.py` + `tests/test_v6a.py`** 模块化推进，先跑通本地 `R=4` 精确求逆（100% 真实哈希命中）再打包交付，彻底杜绝空转代码。

---

### 四、V6A-v2 2D 交错同伦推进路径与门控口径（定稿备忘）
1. **重标定双门控**：`eps_advance = 0.018`（支持 `0.015–0.025`）、`max_ticks_per_alpha = 12`、`min_step_alpha = 0.01`；分岔基于滑窗基线 `pre_jump_baseline` 触发 `e_cur > pre_jump_baseline * 1.25 and (e_cur - pre_jump_baseline) > 0.004`（移除旧 `> eps_advance` 一票否决），命中后自动减半步长并半步回退，激发 **瓶颈 MPS 分裂 + BM 历史正交踢腿 + 3 步 GL-RRR 高能脉冲**。
2. **2D 交错推进路径**：摘要光圈按 `(64 → 96 → 128 → 160 → 192 → 224 → 256)` 渐进锁位（`mask_m` 优先级在首个 `alpha=0 -> step_alpha` 激活非线性后启动：每累积 3 次 `alpha` 推进、或稳定 `< eps_advance` 达 2 次 `alpha` 推进、或 `alpha == 1.0` 时自动收紧一档 `mask_m`，且 `bnd_changed` 后立即触发新边界钳位覆写与全局 OSD-0 传导）；`exact_hits` 永远严格校验 256-bit 全摘要相等，禁止部分掩码虚假计数。
3. **V6A-v2.1 紧急热修记录（全尺度口径 + $2^K$ GPU 变焦精修 + 停滞分岔 MPS + 显存防漏）**：
   - `readout_shadow()` 在保留 `inst_E32_true`（严格对齐 V3/V4/V5）的同时新增全 32 尺度能量 `inst_E_all32` 与离散总违约数 `disc_viol_min`，以 `per_traj_rank = disc_viol_cnt + 0.05 * e_all32` 消除 `m=32` 高位恰好吻合导致的误排序；
   - `endgame_polish()` 升级为无 Python 内循环的 GPU 全向量化 $2^K$（默认 `K=12` / 4,096 组合，显存 `>= 20GB` 时对榜首粒子追加 `K=14` / 16,384 组合）低置信度消息比特翻转变焦验真；
   - `HomotopySchedule2D` 移除 `2.0x` 线性倍增抬槛并标定为 `spike_factor=1.20, spike_abs_margin=0.003`，叠加连续 `3` 个 `mps_every` 周期能量降幅 `< 2%` 触发停滞分岔与 MPS 联合 NLL 训练/注入（彻底激活 `bifurc > 0`, `mps_calls > 0` 与动态 `S_ent`）；
   - `ZeroBrakeNavigator.train_on_successful_jump()` 引入 `max_train_samples=128` 子采样上限，消除 `1,024` 轨迹全批 `Conv1d` 反传带来的 `10.9 GB` 峰值尖峰，且 `solve_single_digest_v6a()` 与 `run_frontier_push()` 每轮结束显式 `del` + `gc.collect(); torch.cuda.empty_cache()` 彻底阻断跨轮显存泄漏。
