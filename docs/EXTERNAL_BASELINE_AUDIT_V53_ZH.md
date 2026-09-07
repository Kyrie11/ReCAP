# OC-RAP 外部 Baseline 审计与 v53 修复说明

## 1. 论文目标与三种 regime

OC-RAP 的主张不是训练三个 regime-specific planner，而是用同一 policy、同一 signed recovery semantics 和同一 recovery library 跨 Safe / Near-Contact / Contact 工作。Safe 主要检查 nominal utility preservation / non-interference；Near-Contact 检查碰撞前低 headroom、TTC/clearance 和 deployability；Contact 检查碰撞后的稳定、逃逸、re-contact、secondary collision 和 terminal clearance。

数据报告与该设计一致：Safe 的 oracle gap 基本为 0，Near/Contact 则显著包含 observation aliasing / oracle-to-deployable gap；Contact 的全部样本处于 post-contact stratum。因此外部 baseline 也应按“常规 planner / 极端风险 planner / 碰后恢复控制”分层，而不是把同一个 generic risk scorer 复制三次。

## 2. Contact 为什么没有 checkpoint

结论：**六个 Contact 主表 baseline 本身都是非学习式 optimization/rule/controller adapters，因此没有 `.pt` checkpoint 是设计行为，不是神经网络训练失败。**

`src/ocrap/external_baselines/train.py::train_external_baseline` 会把这六个方法识别为 `non_learning_filter_or_planner`，读取并验证 train/val grouped dataset，然后写 `train_summary.json`；不会产生虚假的权重文件。

但旧 `scripts/run_contact_external_baselines.sh` 确实还有一个 launcher bug：它完全没有消费 `DO_TRAIN`，所以连上述 validation/registration 都没有运行。v53 已修复：`DO_TRAIN=true` 会一次扫描 train/val 并为六个 contact 方法写 registration summary；仍然不会生成 `.pt`。

## 3. 当前 6×3 主表方法与复现等级

| Regime | Baseline | 对应论文/来源 | 当前实现等级 | 作者源码状态 |
|---|---|---|---|---|
| Safe | GameFormer | Huang et al., ICCV 2023 | mechanism / architecture adapter | 官方 WOMD prediction + open-loop planning repo 可用；不含 WOMD closed-loop |
| Safe | PlanTF | Cheng et al., ICRA 2024 | mechanism / architecture adapter | 官方 nuPlan repo 可用 |
| Safe | PLUTO | Cheng et al., 2024 | mechanism / architecture adapter | 官方 nuPlan repo + checkpoint 可用 |
| Safe | PDM-Closed | Dauner et al., CoRL 2023 | equation/paper-core candidate-lattice adapter | tuPlan Garage 可用 |
| Safe | PDM-Hybrid | Dauner et al., CoRL 2023 | mechanism adapter | tuPlan Garage 可用 |
| Safe | IDM | Treiber et al., PRE 2000 | equation-core projection | 无需学习；连续控制投影到共同 candidate lattice |
| Near | MARC | Multipolicy and Risk-aware Contingency Planning | mechanism adapter | 未确认作者官方代码 |
| Near | RACP | Mustafa et al., T-IV 2024 | mechanism adapter | 作者 GitHub 可用，含 Branch MPC/Frenet planner |
| Near | Robust Scenario MPC | Batkovic et al. | mechanism adapter | 未确认作者官方代码 |
| Near | Predictive Safety Filter | Wabersich & Zeilinger | mechanism adapter | 原论文代码未确认；存在相关 MPSC/SLS 实现但不是同一论文源码 |
| Near | DR-CVaR Safety Filter | Safaoui et al., ICRA 2024 | mechanism adapter | TSummersLab 官方 GitHub 可用 |
| Near | Conformal Predictive Safety Filter | Strawn et al. | mechanism adapter | 未确认作者官方代码 |
| Contact | Integrated Post-impact MPC | Wang et al., T-IV 2023 | **objective-level adaptation** | 未确认公开作者代码 |
| Contact | Autonomous Post-crash Braking | Lu et al., SAE 2017 | **objective/rule-level adaptation** | 未确认公开作者代码 |
| Contact | Post-impact Motion + TVLQR | Wang et al., CJME 2022 | **objective-level adaptation** | 未确认公开作者代码 |
| Contact | Post-collision Restoration | Ghosh et al., arXiv 2026 | **objective-level adaptation** | 未确认公开作者代码；当前是 preprint |
| Contact | Compensatory Post-impact MPC | Cao et al., 2021 | **objective-level adaptation** | 未确认公开作者代码 |
| Contact | Robust Post-impact Control | Ao et al., IET ITS 2022 | **objective-level adaptation** | 未确认公开作者代码 |

**重要措辞：**这些方法都可以作为“paper-inspired / mechanism-level adaptation under a common WOMD candidate interface”比较，但在没有把作者原始 optimizer/controller/feature stack 接进来前，不应写成 “faithful reproduction” 或 “official implementation”。v53 已把 provenance 的过强措辞降级，并显式记录 known gaps。

## 4. 发现并修复的代码/协议问题

1. **Launcher 与主表脱节。** 旧 Safe/Near/Contact scripts 没有运行 provenance 中的当前 6×3 主表。v53 三个 launcher 已严格对齐 `MAIN_TABLE_BY_REGIME`，并加静态 regression test 防止以后再次漂移。
2. **Near calibration flag 是假开关。** 旧 near launcher 声明 `DO_CALIBRATE` 但没有真正执行 conformal calibration。v53 在 `calibration_near_contact` 上做 held-out split calibration，保存带 config fingerprint / split / alpha 的 artifact，测试只读取冻结阈值，不使用 test label。
3. **Near 空 admissible-set 会错误退化成 utility maximization。** 旧实现当任何 candidate 都没通过安全 admission 时，又在所有 feasible candidate 上按原 score 选，可能挑回最高效用但最高风险轨迹。v53 对六个 Near 主表方法加入 method-specific fail-closed safest fallback，并在 `reason` 中显式记录 `empty_admissible_set_safest_fallback`。
4. **Contact `DO_TRAIN` 未执行。** v53 现在一次性扫描/验证 Contact train/val，并生成每个 non-learning baseline 的 `train_summary.json`；没有 `.pt` 是预期。
5. **复现 provenance 过度宣称。** GameFormer/PlanTF/PLUTO/BeTop/near risk planners/contact controllers 的 fidelity wording 已按实际代码下调，并补充源码可用性/缺口。

## 5. 两卡并行与速度优化

现有命令接口保持兼容。三个 regime 默认 `CUDA_DEVICES=0,1`、`MAX_PARALLEL=2`；每次最多两个方法分别占一张卡。调度优先使用 Bash `wait -n -p` 动态回收空闲 GPU，因此某个 planner 先完成后会立即补下一个，而不是等待同一固定 batch 的慢任务。

同时加入：非学习 baseline 一次共享 train/val scan；JAX persistent compilation cache；关闭 XLA 预分配；限制 OMP/MKL/OpenBLAS/TF CPU 线程，避免两进程争抢；closed-loop 支持完成态跳过/partial resume；默认关闭无关 teacher-future metrics；保持 Waymax JIT scan rollout 开关。

## 6. 下一步“源码级”升级优先级

### Safe
最值得先做 **GameFormer → PlanTF → PDM/PLUTO**。这些都有作者/官方仓库，能把目前的 candidate-adapter 升级为“官方 feature/model/optimizer + OCRAP interface bridge”。BeTop 的公开仓库截至当前只完整发布 WOMD prediction，nuPlan planning 仍标 TODO，因此目前不建议把它算成主表的“源码级 planner”。

### Near-Contact
最值得先做 **RACP + DR-CVaR**，二者有明确公开源码，可以真正接入 Branch MPC / safe-halfspace + safety filter，而不是只保留风险目标。之后可增加 **EUDM (ICRA 2020)** 或 **EPSILON (T-RO 2021)**：两者有公开代码，适合 highly-interactive / uncertainty-aware near-contact；若要增加 CCF-B 会议 baseline，EUDM 尤其合适。

### Contact
这个方向的文献主要来自 T-IV / T-ITS / Vehicle System Dynamics / IET ITS / vehicle-control journals，而不是 CCF A/B 计算机会议。强行要求“每个 Contact baseline 都来自 CCF-A/B”会扭曲科学对比。更合理的是保留 5--6 个领域权威 post-impact controllers，并明确它们是 source-level、equation-level 还是 objective-level reproduction。当前六个仍需作者代码（如果存在）或按论文动力学/优化方程做更深实现。

## 7. 建议用户后续提供的源码包

如果你能下载到作者源码，请优先上传以下 repo zip：

- GameFormer (`MCZhi/GameFormer`)
- PlanTF (`jchengai/planTF`)
- PLUTO (`jchengai/pluto`)
- tuPlan Garage (`autonomousvision/tuplan_garage`)
- RACP (`KhMustafa/Risk-aware-contingency-planning-with-multi-modal-predictions`)
- DR-CVaR (`TSummersLab/dr-cvar-safety_filtering`)
- EUDM / EPSILON（若决定扩到 >6 Near baselines）

对 MARC、Robust Scenario MPC、原始 Predictive Safety Filter、Conformal PSF，以及六个 Contact 方法，我目前没有确认到可直接复用的作者官方仓库；如果你找到，请上传 zip，我建议下一轮把它们从 objective/mechanism adapter 升级成 source-level port。

## 8. 测试状态

v53 的 external-baseline targeted tests 已覆盖：observation-only policy、empty-admission fallback、CUDA runtime contract、GameFormer input contract、loss numerics、regime dataset filter、launcher/main-table contract。完整项目测试套件中存在与本次 baseline 修改无关的旧失败/长耗时，因此这里只把 targeted suite 作为本次改动的回归依据，不宣称全仓库测试全部通过。
