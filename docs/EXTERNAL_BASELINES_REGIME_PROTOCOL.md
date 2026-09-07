# OC-RAP 外部 Baseline：三 Regime 复现、数据接入与评测协议（v56）

本文件对应 `post-collision-ocrap-v48.58-rifa-revised.tex` 的实验意图，并约束 `safe / near_contact / contact` 三类外部 baseline 的**选择、数据使用、运行方式、指标发布与复现表述**。

## 1. 与论文主张的关系

OC-RAP 自身是一个 **regime-agnostic runtime policy**：Safe / Near-Contact / Contact 是监督与评测 strata，而不是运行时 router、专家 ID、阈值或预算输入。外部 baseline 的三张对比表则按实验目的拆开：

- **Safe**：比较常规规划器的闭环安全、舒适性、任务保持和非必要干预。
- **Near-Contact**：比较针对不确定性、低安全裕度和极端交互的 contingency / robust / safety-filter 方法。
- **Contact**：比较接触之后的稳定、脱离、再碰撞抑制和 post-impact recovery 方法。

为了避免数据泄漏，每个外部方法只读取其所在 regime 的训练/验证/测试数据。唯一需要统计校准的主表方法 `conformal_predictive_safety_filter` 只使用 `calibration_near_contact` 与其对应的 WOMD **standard validation** raw future，按论文 Algorithm 1 拟合逐预测步冻结 conformal radii `C_h`，绝不使用 test future/label 做阈值或半径选择。闭环压力测试仍可使用 WOMD `validation_interactive`，二者职责严格分离。

## 2. 最终主表：每个 Regime 六个

### Safe（常规闭环规划）

| method id | 对应工作 | 本仓库实现 | 训练 | 选择理由 |
|---|---|---|---|---|
| `gameformer_lite` | GameFormer / GameFormer-Planner | level-k interaction + candidate planning adapter | `train_safe`, `val_safe` | 公开规划实现明确支持 open/closed-loop；交互式学习规划代表 |
| `plantf` | PlanTF | pure-IL state/vector attention + state-dropout candidate adapter | `train_safe`, `val_safe` | 纯学习 planning baseline，nuPlan 公开训练/benchmark 管线 |
| `pluto` | PLUTO | maneuver-query + imitation + contrastive candidate adapter | `train_safe`, `val_safe` | 强 IL planner；保留 maneuver query/CIL 核心 |
| `pdm_closed` | PDM-Closed | IDM-like proposal preference + safety/progress/comfort scoring | 无神经训练；只验证 `train_safe/val_safe` 数据契约 | 强规则闭环 planner，适合 nominal/safe 对比 |
| `pdm_hybrid` | PDM-Hybrid | PDM rule score + learned long-horizon refinement | `train_safe`, `val_safe` | 规则闭环 + 学习 forecast/refinement 的混合对照 |
| `idm` | Intelligent Driver Model | IDM acceleration -> 公共 executable candidate lattice 投影 | 无神经训练；只验证 `train_safe/val_safe` 数据契约 | 标准纵向交互基线，简单且可解释 |

Safe legacy/control（不占 6 个主表名额）：`nominal_replay`, `wayformer_bc`, `betopnet_lite`。

**为何降级 BeTop**：公开仓库当前提供 WOMD prediction pipeline，而 nuPlan planning pipeline 仍列为 TODO，因此把现有 `betopnet_lite` 说成严格公开 planning 复现并不成立。  
**为何降级 Wayformer**：Wayformer 是 motion forecasting 架构；现有代码是 route-conditioned planning adaptation，而不是作者公开 planner 的端到端复现。

### Near-Contact（低裕度 / 不确定性 / 极端交互）

| method id | 对应工作 | 保留的 paper core | 参数拟合 |
|---|---|---|---|
| `marc_lite` | MARC | semantic multipolicy、动态 branchpoint、non-anticipative prefix、risk-aware contingency | 无学习；数据契约验证 |
| `racp_lite` | RACP | multimodal Bayesian belief、contingent plan、probabilistic risk | 无学习；数据契约验证 |
| `robust_scenario_mpc` | Batkovic et al. Robust Scenario MPC | multimodal scenario expected cost + worst/robust constraint | 无学习；数据契约验证 |
| `predictive_safety_filter` | Wabersich & Zeilinger PSF | proposed action 的 minimal correction、stage/terminal backup safety | 无学习；数据契约验证 |
| `dr_cvar_safety_filter` | Safaoui & Summers DR-CVaR filter | 官方源码中的逐障碍/逐时刻 DR-CVaR safe halfspace；对 released affine-loss/unconstrained-support 子问题使用等价 closed-form `g*=empirical_CVaR+eps/alpha-delta`；公共 candidate lattice 上执行 MPC tracking/minimal-correction objective | 无神经训练；数据契约验证 |
| `conformal_predictive_safety_filter` | Strawn et al. CPSF | Algorithm 1 的逐 horizon `C_h`（含 `(N+1)` infinity sentinel 和 `delta/T`）；Eq. (7) `||tau_bar^j-xhat|| >= C_h+epsilon`；最小轨迹偏离投影 | **仅 `calibration_near_contact` + 对应 WOMD standard validation raw future** |

Near legacy/diagnostic：`gameformer_lite`, `expected_risk_filter`, `cvar_risk_filter`, `dro_cvar_filter`, `oracle_recovery_filter`。

- `gameformer_lite` 是通用 learned planner，不应占“极端/低裕度”主表名额。
- 原 `dro_cvar_filter` 只是 `CVaR + ambiguity_radius * dispersion / alpha` 的 Wasserstein-inspired surrogate，不是论文中的完整 DRO 安全半空间/MPC，因此保留 legacy 名称但不再作为主 DR-CVaR baseline。
- `oracle_recovery_filter` 使用 teacher tensor，只能做不可部署 upper bound，不能和 deployable baseline 混入主表。

### Contact（post-impact / secondary-collision avoidance）

| method id | 对应工作 | 保留的 paper core | 参数拟合 |
|---|---|---|---|
| `postimpact_mpc_lite` | Wang et al. 2023 Integrated Post-Impact Planning & Active Safety Control | SBD decision、MPC output/control objective、front/rear axle octagonal adhesion constraints、constant-velocity rhombus obstacle exclusion、LTR constraint、Magic-Formula/friction-similarity tire model、PSO tire-force allocation | 无学习；数据契约验证 |
| `post_crash_braking` | Lu et al. 2017 Post-Impact Braking | collision-triggered autonomous braking、stable-stop intent | 无学习；数据契约验证 |
| `postimpact_motion_tvlqr` | Wang et al. 2022 Post-Impact Motion Planning/TVLQR | quintic `X/Y/psi` planning family + terminal equalities、APF obstacle/road objective、sideslip stability objective、acceleration/rear-axle force constraints、paper Q/R TVLQR、full Magic Formula + friction similarity、nonlinear tire-force allocation | 无学习；数据契约验证 |
| `post_collision_restoration` | Ghosh et al. 2026 | steering + tractive-force restoration、lateral/yaw recovery | 无学习；数据契约验证 |
| `compensatory_postimpact_mpc` | Cao et al. 2021 FCC-MPC | lateral/yaw deviation attenuation、AFS/differential-torque compensation objectives | 无学习；数据契约验证 |
| `robust_postimpact_control` | Ao et al. 2022 | sliding-surface stability recovery + fault-tolerant allocation objective | 无学习；数据契约验证 |

Contact legacy：`severity_minimization`。该工作包含不可避免碰撞前的 collision-severity planning 与 post-impact motion，不能作为纯 post-contact controller 主表项。

## 3. “严格复现”的表述边界

这里必须区分两种 fidelity：

1. **作者原生 benchmark/native implementation**：例如官方 GameFormer-Planner、PlanTF、PLUTO、tuPlan Garage 基于 nuPlan 的数据预处理、地图、proposal generator、trajectory decoder、车辆动力学或 simulator 接口。
2. **本仓库 common-action-space paper-core adapter**：为了让所有方法在你的 WOMD/Waymax regime 数据和同一组 executable candidate prefixes 上公平比较，保留论文的关键决策结构/目标，但把原生连续优化、nuPlan proposal generator、wheel-force/torque actuator、CarSim vehicle states 等替换为公共候选集上的 scoring/admission/projection。

因此论文中建议使用类似：

> “We implement paper-core adaptations of external planners in a shared WOMD/Waymax executable-candidate interface; these adapters preserve the transferable decision mechanism of each method but are not checkpoint-compatible or bit-for-bit reproductions of the authors’ native nuPlan/CommonRoad/CarSim pipelines.”

不要把 `plantf/pluto/PDM/contact MPC` adapter 写成“authors' official implementation on WOMD”。`tools/audit_external_baseline_fidelity.py` 会生成可审计 manifest，逐项列出 `core_retained` 和 `known_gaps`。

## 4. 数据集接入协议

默认根目录：

```text
/data0/senzeyu2/dataset/OCRAP/
  train_safe/               val_safe/               calibration_safe/               test_safe/
  train_near_contact/       val_near_contact/       calibration_near_contact/       test_near_contact/
  train_contact/            val_contact/            calibration_contact/            test_contact/
```

实际使用矩阵：

| Regime | train | val | calibration | test |
|---|---|---|---|---|
| Safe | `train_safe` | `val_safe` | 不使用（六个主表方法无需 calibration） | `test_safe` |
| Near | `train_near_contact` | `val_near_contact` | **只用于 conformal filter：`calibration_near_contact`** | `test_near_contact` |
| Contact | `train_contact` | `val_contact` | 不使用（六个主表方法无需 calibration） | `test_contact` |

Safe 的 4 个学习方法的 imitation target **严格为 logged nominal candidate**；`allow_teacher_supervision=false`。不会因为 `feasible`, `R_dep`, `R_orc`, `hard_violation` 等 OC-RAP teacher/certificate 字段而改换训练 target。

Near/Contact 的优化器/规则/滤波方法没有神经网络需要训练，`train-baseline` 步骤只验证对应 regime 的 `train/val` grouped dataset 可读取并落盘 provenance/config contract；不要为了形式上“有训练”而对 test 调参数。

### Waymax 原始 TFRecord 为什么仍出现在 closed-loop 命令中

`ocrap.cli closed-loop` 的动力学 rollout 需要原始 WOMD TFRecord；这不是跨 regime 训练/测试泄漏。真正限定评测样本的是：

```text
closed_loop.bucket_dataset = test_<regime>
closed_loop.bucket_split   = test
closed_loop.require_bucket_targets = true
```

因此 simulator 从 raw WOMD 恢复状态，但只执行 `test_safe` / `test_near_contact` / `test_contact` 中列出的 scene-time targets。

## 5. 指标发布协议

Raw Waymax artifact 可以包含 superset；论文主表只读取 `tools/summarize_external_closed_loop.py` 的 regime-specific whitelist。

### Safe：普通闭环指标，不发布 post-contact 指标

主要输出：

- `collision_scene_rate`, `offroad_scene_rate`
- `minimum_clearance_m`, `scene_min_clearance_m_median`, `scene_min_clearance_m_p05`
- `minimum_ttc_s`, `scene_ttc_s_median`, `scene_ttc_s_p05`
- `acceleration_abs_p95_mps2`, `jerk_p95`, `yaw_rate_p95`
- `closed_loop_bounded_NUP`, `closed_loop_nominal_deviation`
- `intervention_rate`, `intervention_scene_rate`

### Near-Contact：闭环 + 低裕度/极端指标，不发布 post-contact 指标

主要输出：

- 普通安全：collision/offroad/min-clearance/min-TTC + scene p05
- 低裕度暴露：`near_contact_exposure_rate/duration`, `critical_ttc_exposure_rate/duration`, longest exposure
- 恢复趋势：`terminal_clearance_m`, `clearance_recovery_gain_m`, `terminal_ttc_s`, `ttc_recovery_gain_s`
- 持续风险积分：`clearance_deficit_auc_m_s`, `ttc_deficit_auc_s2`
- 你的 recovery 指标：`closed_loop_FRA_exec`, `closed_loop_DRS`, `closed_loop_ODG`, `closed_loop_bounded_NUP`
- intervention rate

默认 `CL_LABEL_MODE=selected` 只在选择完成之后标注 executed candidate，selector 本身仍 observation-only；这适合大规模跑。`closed_loop_FRA_cand` 只有在 exhaustive teacher labels 可用时才完整，因此默认结果可能为 null。若 Near 主表必须报告 full-candidate FRA，使用 `CL_LABEL_MODE=all` 做最终小规模/算力足够的 authoritative audit；不要用这些 teacher labels 反向调 baseline 参数。

### Contact：只发布 post-contact recovery/stability 指标

主要输出：

- `post_contact_terminal_clearance_m`
- `post_contact_free_space_auc_normalized_m`
- `post_contact_clearance_gain_m`
- `post_contact_escape_scene_rate`, `time_to_post_contact_escape_s`
- `recontact_scene_rate`, `secondary_overlap_scene_rate`
- `new_stable_stop_scene_rate`, `new_stable_stop_quality_scene_rate`
- `post_contact_overlap_duration_s`, `post_contact_overlap_rate`
- `post_contact_clearance_deficit_auc_m_s`
- `post_contact_clearance_m_mean/max`

Contact 的 rollout 仍是闭环 simulator execution，但 publication summary **不混入 ordinary Safe/Near closed-loop metrics**。

## 6. 运行命令

假设代码位于 `/home/senzeyu2/code/OC-RAP`。

### 6.1 一次跑完三类

```bash
cd /home/senzeyu2/code/OC-RAP

OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
CUDA_DEVICES=0,1 \
OUT=/home/senzeyu2/code/OC-RAP/runs/external_baselines_v50 \
MAX_SCENARIOS=0 \
MAX_STEPS=40 \
DO_TRAIN_SAFE=true \
DO_TRAIN_NEAR=true \
DO_TRAIN_CONTACT=true \
DO_CALIBRATE_NEAR=true \
DO_OFFLINE=false \
DO_CLOSED_LOOP=true \
bash scripts/run_all_regime_external_baselines_optimized.sh
```

`MAX_SCENARIOS=0` 按 runner 的全量语义执行 bucket 中所有目标；调试时可设 `10`/`50`。

### 6.2 Safe：训练

```bash
cd /home/senzeyu2/code/OC-RAP
OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
CUDA_DEVICES=0,1 \
RUN=/home/senzeyu2/code/OC-RAP/runs/safe_external_v50 \
DO_TRAIN=true DO_OFFLINE=false DO_CLOSED_LOOP=false \
bash scripts/run_safe_regime_external_baselines.sh
```

训练 checkpoint：

```text
$RUN/checkpoints/gameformer_lite/best.pt
$RUN/checkpoints/plantf/best.pt
$RUN/checkpoints/pluto/best.pt
$RUN/checkpoints/pdm_hybrid/best.pt
```

PDM-Closed/IDM 不产生神经 checkpoint；会写 `$RUN/train_contract/<method>/...`。

### 6.3 Safe：最终 test/closed-loop

```bash
OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
CUDA_DEVICES=0,1 \
RUN=/home/senzeyu2/code/OC-RAP/runs/safe_external_v50 \
CL_MAX_SCENARIOS=0 \
DO_TRAIN=false DO_OFFLINE=false DO_CLOSED_LOOP=true \
bash scripts/run_safe_regime_external_baselines.sh
```

主表 summary：`$RUN/closed_loop_summary.json`。

### 6.4 Near：train-contract + conformal calibration

```bash
OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
CUDA_DEVICES=0,1 \
RUN=/home/senzeyu2/code/OC-RAP/runs/near_external_v50 \
DO_TRAIN=true DO_CALIBRATE=true DO_OFFLINE=false DO_CLOSED_LOOP=false \
bash scripts/run_near_contact_external_baselines_2gpu_optimized.sh
```

校准产物：`$RUN/conformal_near_contact_calibration.json`。v56 中产物保存逐 horizon `conformal_prediction_intervals_m`，而不是旧版单一 scalar collision-risk threshold；同时记录 `delta`、mission horizon `T`、prediction horizon `H`、样本量、raw-WOMD pattern、数据/配置 fingerprint、calibration unit 与 exchangeability caveat。

默认 launcher 维持 `CONFORMAL_CALIBRATION_UNIT=group` 以兼容既有命令；若要把有限样本保证严格提升到 WOMD scene 级独立性，设置 `CONFORMAL_CALIBRATION_UNIT=scene_max`。注意论文的 `delta_bar=delta/T` 与 `(N+1)` infinity sentinel 是硬数学条件：当 calibration scene 数不足以支持指定 `delta/T` 时，脚本会 fail closed/报告 infinity，而不会偷偷截断 quantile。此时应增加独立 calibration scenes、缩短 mission horizon，或在论文允许的实验设定下增大 `delta`，不能用 test 数据“补样本”。

### 6.5 Near：最终 test/closed-loop（推荐大规模）

```bash
OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
CUDA_DEVICES=0,1 \
RUN=/home/senzeyu2/code/OC-RAP/runs/near_external_v50 \
CL_MAX_SCENARIOS=0 \
CL_LABEL_MODE=selected \
DO_TRAIN=false DO_CALIBRATE=false REUSE_CALIBRATION=true \
DO_OFFLINE=false DO_CLOSED_LOOP=true \
bash scripts/run_near_contact_external_baselines_2gpu_optimized.sh
```

若论文主表必须有完整 `FRA_cand`：

```bash
CL_LABEL_MODE=all CL_MAX_SCENARIOS=0 \
DO_TRAIN=false DO_CALIBRATE=false REUSE_CALIBRATION=true \
DO_OFFLINE=false DO_CLOSED_LOOP=true \
bash scripts/run_near_contact_external_baselines_2gpu_optimized.sh
```

注意：`all` 会对每个候选计算昂贵 teacher recovery labels，明显更慢；它只用于**评测标注**，不会输入 external selector。

### 6.6 Contact：train-contract

```bash
OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
CUDA_DEVICES=0,1 \
RUN=/home/senzeyu2/code/OC-RAP/runs/contact_external_v50 \
DO_TRAIN=true DO_OFFLINE=false DO_CLOSED_LOOP=false \
bash scripts/run_contact_external_baselines.sh
```

### 6.7 Contact：最终 post-impact test

```bash
OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
CUDA_DEVICES=0,1 \
RUN=/home/senzeyu2/code/OC-RAP/runs/contact_external_v50 \
CL_MAX_SCENARIOS=0 \
DO_TRAIN=false DO_OFFLINE=false DO_CLOSED_LOOP=true \
bash scripts/run_contact_external_baselines.sh
```

主表 summary：`$RUN/closed_loop_summary.json`，其中只保留 post-contact 指标。

### 6.8 可选 offline diagnostic

把任一脚本的 `DO_OFFLINE=true` 即可在对应 `test_<regime>` 上输出 candidate/offline diagnostic。**论文三 regime 主表以 closed-loop regime summary 为准**，不要把 offline 字段和 main closed-loop/post-impact 指标混在同一张表。

## 7. 关键代码位置

```text
src/ocrap/external_baselines/models.py          # GameFormer / PlanTF / PLUTO / PDM-Hybrid learned adapters
src/ocrap/external_baselines/policies.py        # rule/MPC/filter/post-impact selectors
src/ocrap/external_baselines/observed_risk.py   # observation-only multimodal risk
src/ocrap/external_baselines/data.py            # grouped dataset + no-teacher imitation target
src/ocrap/external_baselines/train.py           # learned training + deterministic data-contract validation
src/ocrap/external_baselines/evaluate.py        # offline evaluation
src/ocrap/external_baselines/provenance.py      # paper/code/fidelity registry + 6x3 main-table contract
src/ocrap/simulation/closed_loop_runner.py       # runtime method registration + physical metrics

tools/calibrate_external_baselines.py            # near conformal calibration only
tools/summarize_external_closed_loop.py          # regime-specific publication metric whitelist
tools/audit_external_baseline_fidelity.py        # provenance/fidelity manifest
tools/build_external_baseline_run_index.py        # three-regime run index

scripts/run_safe_regime_external_baselines.sh
scripts/run_near_contact_external_baselines_2gpu_optimized.sh
scripts/run_contact_external_baselines.sh
scripts/run_all_regime_external_baselines_optimized.sh
```

## 8. 已修正的原实现问题

1. 原 `run_external_baselines.sh` 曾使用跨 regime 的 mixed train/val；现在只作为三 regime launcher wrapper，不再混合训练集。
2. Safe 主表移除 BeTop/Wayformer，加入 PlanTF/PLUTO/PDM-Closed/PDM-Hybrid/IDM，最终六个。
3. Near 主表移除通用 GameFormer、generic expected/CVaR 和旧 `dro_cvar_filter` surrogate；v56 的 `dr_cvar_safety_filter` 已按 Safaoui/Summers 官方源码恢复 DR-CVaR safe-halfspace construction，`conformal_predictive_safety_filter` 已按 Algorithm 1 + Eq. (7) 恢复逐 horizon conformal tube，而不是 scalar risk threshold。
4. Contact 主表移除 pre-impact-heavy `severity_minimization`，补 Wang’22、Cao’21、Ao’22 三类纯 post-impact 方法，最终六个。
5. 修复旧 summary 把 `scene_min_clearance_m_p05` / `scene_ttc_s_p05` 当顶层字段读取的问题；它们实际位于 `waymax_metrics`。
6. Safe imitation target 不再根据 OC-RAP `feasible` annotation 回退到别的 candidate；严格 imitation logged nominal。
7. `oracle_recovery_filter` 只保留 diagnostic，默认主运行不执行。
8. CPSF 的逐 horizon conformal radii 只由 `calibration_near_contact` + 对应 WOMD standard-validation future 生成，test 不参与 calibration；standard validation 与 closed-loop `validation_interactive` source 被显式分离。
9. Wang 2023 Contact port 恢复 paper SBD、octagonal adhesion、constant-velocity rhombus、LTR gate、Magic-Formula friction scaling 与 PSO allocator；轮胎模型按论文拟合单位使用 `F_z[kN]` 与 `alpha[deg]`，而不是错误的 N/rad。
10. Wang 2022 Contact port 恢复 quintic/APF/terminal constraints、TVLQR、full Magic Formula 与 nonlinear allocation；APF 使用论文中的固定 perceived obstacle coordinates `(X_b,Y_b)`，不会额外注入 OC-RAP learned predictor。

## 9. 本环境已经完成的验证 / 尚不能替你完成的验证

已通过：

- 修改过的 Python 文件 `py_compile`。
- 5 个 launcher 的 `bash -n`。
- `tools/audit_external_baseline_fidelity.py`：每个 regime 恰好 6 个 main baseline。
- 新增 learned adapter synthetic forward：GameFormer/PlanTF/PLUTO/PDM-Hybrid 输出统一 24-candidate logits。
- 18 个 main-table selector synthetic smoke test 均可返回有效 candidate。
- external non-oracle policy 对 teacher label mutation 的回归测试。
- launcher/index + observation-only policy 针对性 pytest：14 passed。
- regime summary synthetic test：Safe/Near 能从嵌套 `waymax_metrics` 正确取 scene p05；Contact 只保留 post-contact 字段。

当前沙箱没有挂载你机器的 `/data0/senzeyu2/dataset/OCRAP` 和完整 Waymax runtime，所以这里无法替你做最终 TFRecord/Waymax 全量执行。第一次在服务器运行时建议先用 `CL_MAX_SCENARIOS=2` 做三 regime smoke，再设 `0` 全量。
