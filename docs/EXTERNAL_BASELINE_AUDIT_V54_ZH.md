# OC-RAP 外部 baseline 复现审计与 v54 source-port 说明

> 结论先行：论文的三域划分与 baseline taxonomy 是成立的，但必须把“benchmark interface fairness”和“source fidelity”分开报告。Safe 域本轮提供的 4 个源码包可以做 source-level port；Near-contact 与 Contact 当前代码仍主要是论文机制/目标函数级 adapter，不能写成“官方复现”。

## 1. 我对论文主线的理解

当前论文的核心不是为 safe / near-contact / contact 写三个 planner，而是同一个部署策略在三种状态下都维持“可恢复性”：

- **Safe**：不只追求无碰撞，还要保留后续 recovery reserve；此时 signed recovery margin 应主要为正。
- **Near-contact**：在多模态/遮挡/低 headroom 下，重点是不要把本来仍可恢复的状态推入不可恢复区域；因此应关注 collision risk、clearance/TTC、recovery-opportunity preservation，以及 oracle-to-deployable gap。
- **Contact**：接触发生后，目标从“保留 reserve”转成偿还 recovery debt，抑制 penetration/re-contact/secondary collision，并恢复可稳定、可重新进入正常行驶的状态。
- **关键公平性约束**：regime 只应作为数据分层、监督与评价分层；不能成为运行时输入、router、不同阈值/预算或不同 fallback 的依据。否则“统一 planner”主张会被破坏。

论文里 OC-MERO / oracle-vs-deployable recoverability、signed margin、RIFA fixed-top-K admission 等设计，都服务于这个“一个 planner、三个操作区间”的主线。

## 2. 数据集性质是否支撑这三个 regime

reports 中 test split 的统计与上述定义一致：

| Regime | test scenes / groups / samples | oracle artifact fraction | negative deployable fraction | mean oracle gap | 解释 |
|---|---:|---:|---:|---:|---|
| Safe | 175 / 402 / 3216 | 0.000 | 0.069 | 0.000 | 几乎没有 latent-root deployability ambiguity，适合常规 planner 对比 |
| Near-contact | 250 / 595 / 4723 | 0.244 | 0.488 | 0.422 | 接近一半候选出现负 deployable margin，同时存在显著 oracle gap，适合 risk/contingency/filter 方法 |
| Contact | 209 / 747 / 6687 | 0.218 | 0.444 | 0.298 | 全部样本属于 post-contact，恢复债务/二次碰撞控制成为核心 |

训练集同样表现出明显分层：Safe 的 oracle gap 恒为 0；Near-contact 的 mean gap≈0.326；Contact 的 mean gap≈0.228。三个数据域不是简单按碰撞标签硬切，而是确实在 recoverability/observability 上有不同统计性质。

## 3. 图 1/2/3 主表 baseline 与论文对应关系

### Safe regime

| 代码名 | 论文/来源 | 当前应如何表述 |
|---|---|---|
| `gameformer_lite` | Huang et al., **GameFormer**, ICCV 2023 | **v54: source-ported / interface-adapted**；不能称官方 checkpoint 复现 |
| `plantf` | Cheng et al., **Rethinking Imitation-based Planner(s) for Autonomous Driving**, ICRA 2024 | **v54: source-ported / interface-adapted** |
| `pluto` | Cheng et al., **PLUTO**, 2024 | **v54: source-ported / interface-adapted** |
| `pdm_closed` | Dauner et al., **Parting with Misconceptions about Learning-based Vehicle Motion Planning**, CoRL 2023 | **source-scorer / geometry-adapted** |
| `pdm_hybrid` | 同上，PDM-Hybrid | **2 s executed prefix 内 source-semantics faithful；8 s learned tail 当前 benchmark 不可见** |
| `idm` | Treiber, Hennecke, Helbing, IDM, Phys. Rev. E 2000 | equation-core finite-lattice projection |

### Near-contact regime

| 代码名 | 论文/来源 | 当前应如何表述 |
|---|---|---|
| `marc_lite` | Li et al., **MARC: Multipolicy and Risk-aware Contingency Planning**, RA-L 2023 | mechanism-inspired candidate-lattice adaptation |
| `racp_lite` | Mustafa et al., **RACP: Risk-Aware Contingency Planning with Multi-Modal Predictions** | mechanism-inspired adaptation。注意：DOI 是 2024，但 IEEE T-IV 最终卷期为 **2025, 10(1):228–243** |
| `robust_scenario_mpc` | Batkovic et al., **A Robust Scenario MPC Approach for Uncertain Multi-Modal Obstacles**, IEEE CSL 2021 | mechanism-inspired finite-lattice scenario-MPC adaptation |
| `predictive_safety_filter` | Wabersich & Zeilinger, **A Predictive Safety Filter...**, Automatica 2021 | mechanism-inspired finite-lattice PSF |
| `dr_cvar_safety_filter` | Safaoui & Summers, **Distributionally Robust CVaR-Based Safety Filtering...**, ICRA 2024 | mechanism-inspired approximation；当前没有完整 DRO half-space + continuous MPC correction |
| `conformal_predictive_safety_filter` | Strawn, Ayanian, Lindemann, **Conformal Predictive Safety Filter...**, 2023 | mechanism-inspired approximation；当前是 split-conformal admission，不是作者 RL safety-filter pipeline |

### Contact regime

| 代码名 | 论文/来源 | 当前应如何表述 |
|---|---|---|
| `postimpact_mpc_lite` | Wang et al., **Integrated Post-Impact Planning and Active Safety Control for Autonomous Vehicles**, T-IV 2023 | objective-level finite-lattice adaptation |
| `post_crash_braking` | Lu et al., **A System for Autonomous Braking of a Vehicle Following Collision**, SAE 2017 | objective-level braking/stabilization adaptation |
| `postimpact_motion_tvlqr` | Wang et al., **Post-Impact Motion Planning and Tracking Control for Autonomous Vehicles**, CJME 2022 | objective-level APF/TVLQR adaptation |
| `post_collision_restoration` | Ghosh et al., **Post-Collision Trajectory Restoration...**, 2026 preprint | objective-level restoration adaptation |
| `compensatory_postimpact_mpc` | Cao et al., **Compensatory MPC for Post-impact Trajectory Tracking...**, 2021 | objective-level FCC-MPC adaptation |
| `robust_postimpact_control` | Ao et al., **Advanced Post-impact Safety and Stability Control for Electric Vehicles**, IET ITS 2022 | objective-level sliding-mode/QP adaptation |

## 4. 现有 v53 是否“诚实复现”

### Safe：v53 有几个必须修正的地方

**GameFormer v53** 保留了 level-k/game-theoretic 味道，但并不是官方架构复现。v53 配置是 4-layer encoder、batch 64、weight decay 2e-4，而上传源码的关键公开配置/实现是 6-layer 256-d Transformer、两层 LSTM actor encoder、6-mode GMM、level-wise interaction decoder，并且 previous-level future encoder 有 `detach()`。所以 v53 若写成 “GameFormer reproduction” 会过强；写 “GameFormer-inspired” 才诚实。

**PlanTF v53** 偏差更明显：v53 用 192-d、state dropout 0.25、直接 candidate scoring；上传源码是 128-d、4 layers、8 heads、state dropout 0.75、6-mode native trajectory decoder，并有 source-specific best-ADE regression + mode CE。因此原 v53 只能算 PlanTF-inspired。

**PLUTO v53** 也不能称 source reproduction：v53 用 192-d / 8 heads / 5×5 query，并把 contrastive loss 默认设成 0.35；上传源码是 128-d、4-head、reference-line × 12-mode decoder。更重要的是，官方训练 README 的 full-dataset 默认配置 **不启用 CIL**，CIL 是显式可选项；因此在缺少源 triplet/augmentation pipeline 时人为强行加入 contrastive loss 不够诚实。

**PDM-Closed v53** 至少使用了 progress/TTC/comfort 5/5/2 的表面结构，因此方向对，但没有严格复现 PDMScorer 的 multiplicative safety gate、gated progress normalization、nuPlan geometry/scorer semantics。更合适的名字是 “PDM-Closed source-structure projection”。

**PDM-Hybrid v53 是本轮最重要的科学性问题。** v53 训练了一个 learned candidate head 并把它直接加到当前候选选择上；而上传的 tuPlan Garage 源码里，PDM-Hybrid 在 `correction_horizon=2.0 s` 之前直接沿用 PDM-Closed，学习到的 PDM-Offset 只作用于 2 s 之后的长时域尾部。你当前 closed-loop 是短前缀执行 + 高频 replanning，所以用 learned head 改变前 2 s action 会制造一个原方法不存在的优势。v54 已把这个问题修掉。

**IDM** 本身不存在官方 deep-learning preprocessing 对齐问题，只要明确“continuous IDM acceleration 被投影到共享 executable lattice”，就是可接受的 equation-core baseline。

### Near-contact：taxonomy 合理，但当前不能叫源码复现

这一组把“极端情况 planner”扩展为 contingency planner + risk-sensitive planner + safety filter，我认为是合理的。MARC/RACP直接对应多模态 contingency，Robust Scenario MPC 对应 uncertainty-aware robust planning；PSF/DR-CVaR/CPSF 是对任意 nominal planner 做最小干预的 safety-layer 家族。它们不是同一架构类别，但都回答同一个 benchmark 问题：**在 near-contact observable state 下，给定相同可执行动作空间，谁能更好地避免不可恢复选择。**

公平性前提是：filters 必须拿到同样的 nominal reference / candidate set，只用 observation 与 calibration split，不得读取 OC-RAP teacher root/recovery tensor。当前代码按这个原则设计是对的。但在没有源码逐行接入前，MARC/RACP/Scenario MPC/PSF/DR-CVaR/CPSF 都应该标 “mechanism-inspired/interface-adapted”，而不是 “reproduced”。

### Contact：研究问题正确，但“论文复现”表述尤其要谨慎

Contact 这组六篇论文都确实研究 first impact 之后的 braking / trajectory restoration / stability / secondary-collision avoidance，因此与论文的 recovery-debt 主线高度相关。问题是这些工作大量依赖 **车辆动力学、轮胎/侧偏、AFS、差动扭矩、轮端故障、impact-induced state、控制分配器**，而 WOMD/Waymax 不提供这些原生状态。

因此，把这些方法投影到同一 candidate lattice 对“benchmark interface fairness”是好事，但对“source fidelity”是有损的。建议论文主表/附录明确加一句：**“Post-impact baselines are objective/mechanism-level ports to the shared Waymax executable-candidate interface; source-native wheel/impact dynamics are unavailable in WOMD.”** 这样审稿人很难指责你把有限动作空间 surrogate 冒充原论文的 full controller。

## 5. `external_baselines.zip` 四个源码包到底是什么

| 源码包 | 对应外部 baseline | 本轮用途 |
|---|---|---|
| `GameFormer-main.zip` | GameFormer | 复现 source actor encoder / scene Transformer / level-k multimodal decoder，再投影到 OC-RAP candidates |
| `planTF-main.zip` | PlanTF | 复现 source scene/state encoder 与 6-mode trajectory decoder，再投影到 OC-RAP candidates |
| `pluto-main.zip` | PLUTO | 复现 reference-line × mode planning decoder 与 source training defaults，再映射到 executable lattice |
| `tuplan_garage-main.zip` | PDM-Closed **和** PDM-Hybrid | 复现 PDMScorer structure；同时校正 Hybrid 的 2 s correction-horizon semantics |

所以“4 个源码包”对应 **5 个 safe 主表 baseline row**；IDM 不需要外部源码包。

## 6. v54 做了什么

### 6.1 GameFormer source port

- 两层 LSTM actor history encoder，8-state input、256 hidden。
- 6-layer / 8-head / 256-d scene Transformer。
- 6-mode GMM trajectory head。
- 4-level joint multi-agent interaction refinement。
- previous-level future encoder 保留 source `detach()` semantics。
- 每一级都可计算 native-style trajectory loss；由于 OC-RAP sample 没有作者格式的 all-neighbor future target，source-native GMM loss 诚实地只对 ego supervision。
- native trajectory 最后通过 ADE-based mode-to-candidate projection 映射到共享 executable lattice，不让 teacher recovery label 进入输入。

**已知差距**：官方公开 repo 有 WOMD interaction prediction 与 WOMD open-loop planning，但明确不提供 WOMD closed-loop；v54 因此是 source-derived Waymax port，不是作者官方 closed-loop implementation。当前 20-step benchmark 也短于 native 80-step target。

### 6.2 PlanTF source port

- 128-d、4-layer、8-head。
- 9-channel actor history；6-channel state attention。
- state dropout=0.75。
- 6-mode source-shaped trajectory decoder。
- best-ADE SmoothL1 + mode CE。
- AdamW decay/no-decay grouping + 3-epoch warmup cosine，lr=1e-3，wd=1e-4。
- agent truncation 改成 **ego first + observed current-distance nearest neighbors**，避免 source_max_agents speed cap 按 WOMD 存储顺序任意裁人。

**已知差距**：WOMD-derived contract 只有 11 history steps，而 source config 是 21；为速度把 source_max_agents 设成 16（source feature builder 最大 32）；NATTEN local attention 因依赖不可用改成三层 local-convolution FPN；80-step target 缩为 20-step；没有 source auxiliary neighbor-future target。

### 6.3 PLUTO source port

- 128-d、4-layer encoder / 4-layer decoder、4 heads。
- 12 longitudinal/lateral planning modes per reference line。
- reference-to-reference、mode-to-mode、scene cross-attention 的 source-style decoder structure。
- native trajectory regression + flattened reference×mode classification。
- 使用作者 README 的 full-dataset training defaults：25 epochs、lr=1e-3、warmup=3、wd=1e-4；**不伪造默认 CIL**。

**已知差距**：11 vs source 21 history；source_max_agents=16 vs source max 48；NATTEN 替换；ESDF collision loss / auxiliary agent-future loss 因原始 geometry/target 不存在而不伪造。

### 6.4 PDM-Closed / PDM-Hybrid

`pdm_closed` 现在按 source PDMScorer 的结构做 shared-lattice projection：

- multiplicative safety gate；
- progress 先 gated，再按同场景最大正 progress 归一化；
- progress/TTC/comfort = 5/5/2 加权；
- direction / drivable / TTC / comfort 都用 WOMD observable geometry 做显式 proxy；
- 无学习，不读取 OC-RAP recovery teacher。

`pdm_hybrid` 当前 **故意与 PDM-Closed 的 current prefix/action 相同**。这不是“退化了”，而是对上传 source 里 `correction_horizon=2.0 s` 的忠实处理。若以后要让 PDM-Hybrid 展示 source-native hybrid advantage，应增加 8 s trajectory output/evaluation，而不能训练一个 head 去改前 2 s action。

## 7. 性能优化与双卡调度

本轮没有伪造任何 GPU speedup 数字，因为当前容器看不到你的 `/data0/...` 数据，也没有可用的两张 GPU 做完整 wall-clock benchmark。代码层面的确定性优化包括：

- source scene/map/actor tensor **每个 scene-time group 只构建一次**，不随 candidate 重复构建；
- source planner 先输出 native multimodal trajectory，再一次性投影到 candidates；避免为每个 candidate 重新跑整套 scene encoder；
- PlanTF/PLUTO 用 nearest-neighbor actor cap（16）控制显存/吞吐，同时保持 source 风格的 actor selection；
- 三个 non-learning safe methods (`pdm_closed`, `pdm_hybrid`, `idm`) 只做一次 train/val registration scan；
- CUDA 下优先 fused AdamW，若运行环境不支持自动 fallback；
- AMP dtype 保持 auto；
- safe launcher 的 `MAX_PARALLEL` 强制不超过 2，并按 GPU completion 动态续排，所以两张卡始终最多各跑一个 method；
- closed-loop 继续使用已有 JAX compilation cache、`XLA_PYTHON_CLIENT_PREALLOCATE=false`、Waymax JIT rollout 等优化。

## 8. 原命令兼容性

你的四条入口命令不需要改。尤其 safe 命令仍然是：

```bash
OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
CUDA_DEVICES=0,1 \
RUN=/home/senzeyu2/code/OC-RAP/runs/safe_external_v50 \
CL_MAX_SCENARIOS=0 \
DO_TRAIN=true DO_OFFLINE=false DO_CLOSED_LOOP=true \
bash scripts/run_safe_regime_external_baselines.sh
```

v54 会把 v53 的旧 learned checkpoint 识别为 implementation version 不匹配，从而自动重训 source-port checkpoint，避免静默复用不兼容模型。

Near-contact / Contact / all-regime launcher 的 CLI contract 本轮保持不变。本轮 source-level 改写集中在这 4 个上传源码包对应的 Safe methods；Near/Contact 仍先保持现有 mechanism/objective adapters，等下一轮有源码再逐个替换。

## 9. 验证状态

在当前容器内完成：

- source-port synthetic full forward + native loss + backward；
- observation-only invariance；
- PDM-Hybrid 与 PDM-Closed 的 short-prefix equality；
- external baseline loss numerics；
- GameFormer input contract；
- regime dataset filtering；
- CUDA runtime contract；
- launcher contract；
- Python `py_compile`；
- 4 个主 launcher `bash -n`。

最终 external-related test suite：**28 passed**。

没有完成、也没有声称完成：你机器上真实 `/data0/...` WOMD/OC-RAP 的 full training、full closed-loop、最终指标与 wall-clock speedup。当前容器没有这些数据目录/GPU，必须在你的两卡机器上用原命令做最终 benchmark。

## 10. 论文里建议采用的措辞

推荐把主表说明拆成两层：

- **Benchmark fairness**：所有方法使用相同 observable state、相同 executable candidate/action interface、相同 split/closed-loop evaluator；不允许 teacher recovery/root tensor 泄漏。
- **Source fidelity**：GameFormer/PlanTF/PLUTO = source-ported/interface-adapted；PDM = source-structure/geometry-adapted；Near = mechanism-inspired until source ports are available；Contact = objective/mechanism-level due missing post-impact wheel/impact states。

这样能同时守住公平性和学术诚实性，而不是为了“看起来像完整复现”隐去无法从 WOMD 恢复的原始状态。

## 11. 下一轮建议寻找的四个源码/论文

下一轮按 Near-contact 主表顺序，建议先准备：

1. **MARC** — *MARC: Multipolicy and Risk-aware Contingency Planning for Autonomous Driving*, RA-L 2023。优先作者源码/补充材料；若没有公开 repo，就给论文 PDF + supplement + 作者 simulation code。
2. **RACP** — *RACP: Risk-Aware Contingency Planning with Multi-Modal Predictions*, DOI `10.1109/TIV.2024.3411530`，final T-IV 2025。请下载作者官方 `KhMustafa/Risk-aware-contingency-planning-with-multi-modal-predictions`。
3. **Robust Scenario MPC** — Batkovic et al., *A Robust Scenario MPC Approach for Uncertain Multi-Modal Obstacles*, IEEE CSL 2021, DOI `10.1109/LCSYS.2020.3006819`。优先作者 solver/simulation scripts + paper/supplement。
4. **Predictive Safety Filter** — Wabersich & Zeilinger, *A Predictive Safety Filter for Learning-Based Control of Constrained Nonlinear Dynamical Systems*, Automatica 2021, DOI `10.1016/j.automatica.2021.109597`。优先作者 MPC implementation / associated code + paper/supplement。

拿到这四套后，下一轮可以把 Near-contact 前四个从 “mechanism-inspired” 升级到尽可能 source-derived 的共同 Waymax interface；再下一轮自然接 DR-CVaR + CPSF + 两个 Contact source packages。
