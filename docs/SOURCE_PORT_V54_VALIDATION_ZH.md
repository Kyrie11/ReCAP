# Source-port v54 验证与运行说明

## 版本目的

v54 只解决本轮用户提供的 4 个源码包：GameFormer、PlanTF、PLUTO、tuPlan Garage（PDM-Closed/PDM-Hybrid）。Near-contact 与 Contact launcher/API 保持兼容，但其论文级 source-port 留到后续轮次。

## 兼容性

原 `外部baseline训练和测试.txt` 中四条命令均不需要更改。Safe learned checkpoint 新增 `implementation_version=source_port_v54` contract；旧 v53 checkpoint 会被判定为不兼容并自动重训（若 `DO_TRAIN=true`）。

## 双 GPU 语义

Safe launcher 读取 `CUDA_DEVICES=0,1`，并将 `MAX_PARALLEL` 上限固定为 2。learned/non-learning/offline/closed-loop 的 method queue 都采用“一 method / 一 GPU”的两并发调度；某 GPU 结束后继续领取下一个 method，而不是单个 method 跨两卡 DDP。

这符合用户要求的“每次训练和测试时两两分别运行在两张卡上”。

## 验证命令

```bash
cd /path/to/OC-RAP-baselines-sourceported-v54
export PYTHONPATH=$PWD/src

pytest -q \
  tests/models/test_external_baseline_loss_numerics.py \
  tests/models/test_external_gameformer_input_contract.py \
  tests/models/test_external_observation_only_policies.py \
  tests/models/test_external_regime_dataset_filter.py \
  tests/models/test_external_source_ports_v54.py \
  tests/test_external_baseline_cuda_runtime.py \
  tests/test_external_baseline_launcher_contract.py

bash -n scripts/run_all_regime_external_baselines_optimized.sh
bash -n scripts/run_safe_regime_external_baselines.sh
bash -n scripts/run_near_contact_external_baselines_2gpu_optimized.sh
bash -n scripts/run_contact_external_baselines.sh
```

当前容器结果：`28 passed`。

## 在用户机器上建议先做的 smoke test

正式 `CL_MAX_SCENARIOS=0` 前，先把 Safe 的 `CL_MAX_SCENARIOS=8` 或 16，确认：

- 三个 learned method 自动重训并产生 `source_port_v54` checkpoint；
- PDM-Hybrid 不再要求 `.pt` checkpoint；
- PDM-Hybrid short-prefix action 与 PDM-Closed 一致是预期行为；
- 两张卡同时各跑一个 method，显存没有互相抢占；
- closed-loop artifact 通过 `tools/check_closed_loop_artifact.py`。

然后再恢复你原来的 `CL_MAX_SCENARIOS=0` 全量命令。

## 不能由当前容器代替验证的事项

- `/data0/senzeyu2/dataset/OCRAP/...` 的真实 I/O 吞吐；
- 原始 WOMD tfrecord / Waymax closed-loop compatibility；
- 两张实际 GPU 的 wall-clock、显存、利用率；
- 论文最终表格指标。

因此 v54 的速度优化是代码路径优化，不附带未经实测的百分比 speedup。
