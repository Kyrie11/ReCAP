# OC-RAP v60：A30/JAX/Waymax/PyTorch 兼容运行时与外部 Baseline 无算法改动加速说明

## 1. 结论先行

本版本针对双 NVIDIA A30（24 GB）、Driver 570.211.01、`nvidia-smi` 显示 CUDA 12.8 的机器，采用以下原则：

1. **JAX 使用 CUDA 12 pip-managed runtime**，固定 `jax/jaxlib/jax-cuda12-plugin/jax-cuda12-pjrt==0.6.2`。
2. **PyTorch 使用 CUDA 12.8 wheel**，固定 `torch==2.8.0`；与 JAX 共用兼容的 CUDA 12.8 / cuDNN 9.10.2.21 / NCCL 2.27.3 等 NVIDIA pip wheels。
3. **TensorFlow 只承担 WOMD/Waymax TFRecord 输入角色**：安装 `tensorflow==2.18.1`，但不安装 `tensorflow[and-cuda]`，并在 Waymax 导入前通过 TensorFlow 自身 API 隐藏 GPU。不要同时安装 `tensorflow` 和 `tensorflow-cpu`。
4. **Waymax 使用官方 GitHub main 安装方式**；安装时受本仓库 constraints 约束，防止它反向升级 JAX/Flax/TensorFlow 数值栈。
5. **不要设置 `JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1`**。当前报错明确涉及 cuDNN 过旧时的多 GPU 矩阵乘正确性问题，跳过检查会把“不能启动”变成“可能静默算错”。
6. 启动器统一 `env -u LD_LIBRARY_PATH`，避免 `/usr/local/cuda` 或旧 cuDNN 抢在 pip-managed CUDA 之前被动态链接器加载。

推荐直接运行：

```bash
conda create -n ocrap-a30 python=3.10.16 pip -y
conda activate ocrap-a30
cd /path/to/OC-RAP-v60-a30-runtime-speed-optimized
bash scripts/install_a30_py310_runtime.sh
```

验证：

```bash
env -u LD_LIBRARY_PATH OCRAP_TENSORFLOW_CPU_ONLY=1 \
  python tools/check_jax_waymax_runtime.py --require-gpu --check-waymax
```

## 2. 对论文核心思想的理解

论文的核心不是再增加一个 Safe/Near/Contact 路由器，而是把“接触前保存可恢复性”和“接触后执行恢复”放进同一个 **observation-consistent recoverability** 逻辑中。Safe、Near-contact、Contact 是评估 strata，不是策略状态。

关键动机是解决 branch-wise oracle 的 deployability gap：如果不同 latent future root 在当前部署观测下不可区分，策略不能根据隐藏 root 选择不同 recovery option。因此 OC-MERO 先固定 recovery option，在 observation-compatible latent roots 上做 lower-tail/LCVaR 评价，再在可区分 observation anchor 内选择 option，防止 hidden-root-specific cheating。

同一个 signed recoverability 对象贯穿三类场景：先建立 support，再保留正 reserve；若已经产生负 recovery debt，则在 support 存在后偿还。最终 admission 是非补偿式的 support--reserve 约束，正 utility 不能补偿 support 丢失。RIFA 将“相对 proposal 排名”和“绝对 recovery admission”分开；CNRO 是 candidate-conditioned、native interaction orientation 的审计而非另一个 planner。

这意味着外部 baseline 的加速也必须遵守同样的实验语义：不能减少搜索预算、不能改控制律、不能使用额外 teacher/future 信息、不能更改 candidate lattice、不能把 oracle 信息泄漏给部署 baseline。

## 3. 数据集规模与性能含义

上传的 `reports.zip` 中，主要 dataset 规模为：

| Regime | Split | samples | groups | scenes | candidates/group | feasible rate |
|---|---:|---:|---:|---:|---:|---:|
| Safe | train | 20,000 | 2,500 | 1,171 | 8.000 | 0.91345 |
| Safe | val | 2,328 | 291 | 132 | 8.000 | 0.92397 |
| Safe | test | 3,216 | 402 | 175 | 8.000 | 0.92755 |
| Safe | calibration | 2,544 | 318 | 135 | 8.000 | 0.95087 |
| Near | train | 13,324 | 1,800 | 600 | 7.402 | 0.87864 |
| Near | val | 3,445 | 433 | 176 | 7.956 | 0.93498 |
| Near | test | 4,723 | 595 | 250 | 7.938 | 0.89943 |
| Near | calibration | 6,039 | 765 | 316 | 7.894 | 0.86190 |
| Contact | train | 16,790 | 2,000 | 500 | 8.395 | 0.86992 |
| Contact | val | 6,477 | 723 | 211 | 8.959 | 0.91771 |
| Contact | test | 6,687 | 747 | 209 | 8.952 | 0.89652 |
| Contact | calibration | 16,843 | 1,896 | 543 | 8.883 | 0.87829 |

注意 Contact 目录里还有旧的 `traincontact.json`；当前应使用 `train_contact.json` 对应的 16,790 / 2,000 / 500 数据集。

性能上，这些规模并不大到足以解释 PlanTF/GameFormer “非常慢”。因此训练慢更像是 **小 kernel + Python launch + CUDA host sync + 重复数据物化**，而不是纯 FLOPs 不够。Contact test 只有 747 个 group；若 `compensatory_postimpact_mpc` offline 仍非常慢，也不可能主要来自它本身亚毫秒级 selector，而更像 evaluator 重复构造 risk profile / 重复 dataset pass。

## 4. 外部 baseline 原理、实际瓶颈与安全加速边界

### 4.1 Safe regime

| Baseline | 当前代码中的原理/实现 | 主要成本 | v60 处理 |
|---|---|---|---|
| GameFormer | 8-state 2-layer LSTM actor encoder；256-D 6-layer Transformer scene fusion；6-mode GMM；多层 level-k joint refinement；trajectory-to-candidate projection | actor-indexed decoder 原来对每个 actor Python 循环执行 MHA/FFN/predictor；batch=16，很多小 CUDA kernel；每 batch 多次 host sync | 将 actor 维展平为 `B*A` 做完全等价 batched MHA/FFN/predictor；保留 actor-specific mask；训练通用 host-sync 优化 |
| PlanTF | 128-D 4-layer scene Transformer；6-channel state attention；6-mode trajectory decoder；best-ADE SmoothL1 + mode classification | 模型本身不大，旧 gradient clipping 会对每个 parameter 把完整 gradient 转 float64 并逐次同步；state/map encoder 有数据依赖 `any()` host sync；冗余 legacy history H2D | foreach gradient norm；一次 epoch metric D2H；state affine fuse；去除 map/actor host branch；source port 不再构造未消费的 legacy history |
| PLUTO | 128-D scene encoder；reference-line × 12-mode、4-layer decoder，多种 cross-attention | 与 PlanTF 共用 scene encoder；decoder 本身较重但已 tensorized | 共享 scene encoder 加速；去冗余 history；不改 12 mode / decoder depth |
| PDM-Closed | source-style PDM scorer，安全 gate 后乘法聚合 progress/TTC/comfort | 主要是几何/候选打分，便宜 | 不改算法；不值得 GPU 化 |
| PDM-Hybrid | 当前短 prefix 接口内前 2 s 与 PDM-Closed 一致；learned tail 不可用 | 同上 | 不改 |
| IDM | desired acceleration / gap / relative-speed 方程，连续控制投影到公共 candidate | 极便宜 | 不改 |

GameFormer 的 v60 eval 数值等价验证（同一 state_dict）：logits 最大绝对误差约 `2.38e-7`，ego trajectory `5.07e-7`，scores `1.49e-7`；当前 CPU synthetic forward 约 `18.81 ms -> 7.46 ms`（约 2.52x）。这不是 A30 实测速度，只说明去 Python actor loop 的方向有效。

PlanTF eval 数值等价验证：logits `3.58e-7`、trajectory `5.36e-7`、probability `2.01e-7`；当前 CPU synthetic forward 约 `7.51 ms -> 6.54 ms`。A30 上真正更重要的是去掉每 batch 多次 CUDA host sync。

### 4.2 Near-contact regime

| Baseline | 原理/逻辑 | 性能结论 |
|---|---|---|
| MARC-lite | multi-policy + risk-aware contingency；policy-conditioned scenario responses；CVaR 风险权重 | 主要成本来自共享 actor-risk forecast；多 baseline 分开 offline 时会重复付费 |
| RACP-lite | shared prefix + branch contingent recourse，分支可区分前满足 nonanticipativity | candidate-tree 枚举中等；共享 risk context 很重要 |
| robust_scenario_mpc | 对多 obstacle modes 做 robust constraint satisfaction；不可区分阶段共享控制 | beam/candidate-tree 枚举中等，不建议降低 beam/search budget |
| predictive_safety_filter | 原 proposed input 可行则保持，否则最小 executable correction | 很便宜 |
| DR-CVaR safety filter | affine safe-halfspace + DR-CVaR 几何 | 已用闭式解替代大量 tiny CVXPY solve，当前不是主要瓶颈 |
| conformal predictive safety filter | calibration 上拟合 horizon-wise conformal intervals，test 使用冻结校准 | calibration 是一次数据遍历；filter 本身便宜 |

v60 的 Near offline 把 6 个 non-learning baseline 放到 **一次 evaluator dataset pass** 中，risk forecast/context 共享，然后再拆回历史 `eval_near_contact_<method>.json` 文件名。它只改变调度和缓存，不改变每个 baseline 的 selector。

### 4.3 Contact regime

| Baseline | 原理/逻辑 | 瓶颈与处理 |
|---|---|---|
| postimpact_mpc_lite | Wang-2023-style integrated MPC + SBD + PSO wheel/steer allocation；500 particles × 8 iterations；prediction horizon 10；control horizon 3 | 是 Contact 真正重 kernel；v59 已跨 candidates/knots 向量化。v60 **没有**减少 particle/iteration/horizon，所以算法预算不变；offline 中单独作为 heavy bundle 与 fast bundle 并行 |
| post_crash_braking | post-crash brake / stable-stop rule | predictor-free、非常便宜；v60 不再提前算全 candidates risk profiles |
| postimpact_motion_tvlqr | quintic/APF reference + TVLQR + nonlinear tire allocation | v59 已 batch DARE/TVLQR；不是主要投诉瓶颈 |
| post_collision_restoration | post-collision trajectory restoration structured adapter | predictor-free、便宜；共享 evaluator overhead |
| compensatory_postimpact_mpc | Cao-2021-inspired architecture；公开材料不足以 equation-exact reproduction，代码明确是 source-limited structured adapter | controller microkernel 约 0.75 ms；offline 慢主要是 evaluator/risk/dataset overhead，而不是 controller；v60 修复重复 risk profile 构造和重复 dataset pass |
| robust_postimpact_control | Ao-2022 sliding-mode + exact 4-wheel box-QP/force allocation | 旧实现每次 Python 循环 3^4=81 active sets；v60 批量计算同样的 81 个 exact faces，保持 objective/tie rule |

Robust postimpact exact box-QP 在 200 个随机等价用例上的最大 `u` 误差约 `3.28e-10`，最大 objective 误差 `3.73e-9`；当前 CPU controller-like batch 从约 `2.571 ms -> 0.233 ms`（约 11.05x）。

### 4.4 为什么 compensatory offline 慢但 controller 不慢

v59 中 predictor-free controller 虽然 selection 不需要预测 actor risk，evaluator 仍为了输出 selected risk metrics，可能对每个 method 单独调用 `observed_risk_profile`。多个 Contact method 在同一 scene-time group 内往往选中相同或少数几个 candidates，于是同一个 actor forecast 被重复构造。

v60 先完成所有 method selection，再收集 **unique selected candidate index**，一次批量构造这些 selected risk profiles；如果某 baseline 本来就需要 full profiles，则直接复用已经存在的 profiles。Contact offline 又把 6 次完整 dataset pass 降到 2 次：PSO heavy 一次，其他 fast methods 一次，并行执行。

## 5. JAX 报错根因

当前错误不是“机器没有 CUDA JAX”。从日志看，`jax-cuda12-plugin==0.6.2` 已经被发现并进入 `jax_plugins.xla_cuda12.initialize()`，只是插件自检发现：

- loaded cuDNN = 9.1 (`90100`)
- loaded cublasLt = 12.9 (`120902`)
- JAX 0.6.2 针对该组合有 multi-GPU correctness guard，要求 cuDNN >= 9.10.1

插件初始化失败后，JAX 才退回 CPU，于是后面的 “CUDA-enabled jaxlib is not installed” 是二次/泛化提示，不是第一根因。

当前环境还有更明显的污染：同时出现 `tensorflow==2.12.0` 和 `tensorflow-cpu==2.18.0`。这两个 distribution 提供同一个 `tensorflow` import namespace，不应共存。再叠加系统 `LD_LIBRARY_PATH` 时，很容易把 pip JAX 自带的 CUDA/cuDNN 替换为旧系统库。

`nvidia-smi` 里的 `CUDA Version: 12.8` 表示 **Driver 能支持到的 CUDA API 级别**，不等于 `/usr/local/cuda` 一定安装了 CUDA 12.8 toolkit。对这台机器更稳的是 pip-managed CUDA，不依赖系统 toolkit。

另外 Driver 570.x 适合 CUDA 12 JAX wheel；当前 JAX 官方 CUDA 13 Linux driver floor 是 580，所以这里不要装 `jax[cuda13]`。

## 6. 锁定版本

`constraints/a30_py310_runtime.txt` 固定：

```text
Python              3.10.16
numpy               1.26.4
scipy               1.13.1
ml-dtypes           0.5.1
protobuf            4.25.8
jax                  0.6.2
jaxlib               0.6.2
jax-cuda12-plugin    0.6.2
jax-cuda12-pjrt      0.6.2
tensorflow           2.18.1
flax                 0.10.6
chex                 0.1.89
optax                0.2.4
orbax-checkpoint     0.11.12
torch                2.8.0 + cu128 wheel
nvidia-cublas-cu12   12.8.4.1
nvidia-cudnn-cu12    9.10.2.21
nvidia-nccl-cu12     2.27.3
...其余 CUDA 12.8 NVIDIA wheels 见 constraints 文件
```

选择 Python 3.10 是因为 JAX 0.6.2 要求 Python >=3.10，而本仓库/Waymax 路径也要求 >=3.10；同时 Torch cu128 有 cp310 Linux wheel，TensorFlow 2.18.1 支持 Python 3.10。

`numpy==1.26.4` 同时满足 JAX 0.6.2 的 `numpy>=1.26` 和 TensorFlow 2.18.1 的范围；`ml-dtypes==0.5.1` 同时满足 JAX 的 `>=0.5.0` 与 TensorFlow 的约束。

关键共享项是 `nvidia-cudnn-cu12==9.10.2.21`：它高于当前 JAX 报错要求的 9.10.1，并与 Torch 2.8 cu128 栈兼容。

## 7. 为什么不用 tensorflow-cpu

直觉上可以装 `tensorflow-cpu`，但 Waymax 的 package metadata 声明的是 `tensorflow>=2.11`。只装 `tensorflow-cpu` 时，pip 仍可能认为 `tensorflow` requirement 未满足并再次安装 `tensorflow`，最终又回到两个 distribution 共存的污染状态。

因此本版本采用：

```bash
pip install tensorflow==2.18.1
```

但 **不使用** `[and-cuda]` extra，并在 OC-RAP 的 Waymax loader 中：

```python
tf.config.set_visible_devices([], "GPU")
```

这样 TensorFlow 实际承担 CPU 输入流水线，JAX/Torch 仍能看到 A30。不要设置全局 `CUDA_VISIBLE_DEVICES=""` 来禁 TF GPU，因为那也会把 GPU 从 JAX 隐藏。

## 8. 完整安装步骤

### 8.1 推荐的一键方式

```bash
conda deactivate 2>/dev/null || true
conda create -n ocrap-a30 python=3.10.16 pip -y
conda activate ocrap-a30

cd /home/senzeyu2/code/OC-RAP-v60-a30-runtime-speed-optimized
bash scripts/install_a30_py310_runtime.sh
```

### 8.2 等价的手工方式

```bash
python -m pip install --upgrade pip setuptools wheel

python -m pip uninstall -y \
  jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt \
  jax-cuda13-plugin jax-cuda13-pjrt \
  tensorflow tensorflow-cpu tensorflow-intel \
  torch torchvision torchaudio

unset LD_LIBRARY_PATH

python -m pip install -c constraints/a30_py310_runtime.txt \
  torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

python -m pip install -c constraints/a30_py310_runtime.txt \
  "jax[cuda12]==0.6.2"

python -m pip install -c constraints/a30_py310_runtime.txt \
  tensorflow==2.18.1 numpy==1.26.4 scipy==1.13.1 \
  ml-dtypes==0.5.1 protobuf==4.25.8

python -m pip install -c constraints/a30_py310_runtime.txt \
  flax==0.10.6 chex==0.1.89 optax==0.2.4 orbax-checkpoint==0.11.12

python -m pip install -c constraints/a30_py310_runtime.txt \
  "git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax"

python -m pip install -e . --no-deps
python -m pip install "PyYAML>=6.0" "tqdm>=4.66" "crc32c>=2.3"
python -m pip check
```

如果只运行已经构建好的 OCRAP regime dataset + Waymax TFExample closed-loop，上述栈足够。仓库里 legacy `Scenario` proto reader 还支持另一路径；只有你直接读取 serialized `waymo_open_dataset.protos.Scenario` 时才需要额外提供 `waymo_open_dataset.protos.scenario_pb2`。不要为了这个 legacy parser 随意安装一个会强制降级 TensorFlow 的旧 `waymo-open-dataset-tf-*` wheel。

## 9. 安装后的强制验证

```bash
env -u LD_LIBRARY_PATH OCRAP_TENSORFLOW_CPU_ONLY=1 \
python tools/check_jax_waymax_runtime.py --require-gpu --check-waymax
```

再执行：

```bash
python - <<'PY'
import jax
import torch
import tensorflow as tf
print("JAX:", jax.__version__, jax.devices())
print("Torch:", torch.__version__, torch.version.cuda,
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
tf.config.set_visible_devices([], "GPU")
print("TF:", tf.__version__, tf.config.get_visible_devices("GPU"))
PY
```

期望：

- JAX devices 中出现 NVIDIA A30 GPU；
- Torch `torch.cuda.is_available()` 为 True；
- TensorFlow visible GPU 为空；
- `pip check` 无冲突；
- 不再存在 `tensorflow-cpu`、JAX CUDA13 plugin 或旧 JAX plugin；
- `LD_LIBRARY_PATH` 在运行 baseline 的 launcher 进程内为空。

## 10. v60 代码改动

### 10.1 `train.py`

- 原 v59 `_stable_clip_grad_norm_` 对每个 parameter gradient 做完整 float64 转换和 finiteness 检查，CUDA 上会产生许多同步和 FP64 memory traffic。
- v60 使用 `torch._foreach_norm` 获取 per-parameter norm，只有小标量向量使用 float64 做跨参数 reduction，只做一次 `.item()` host sync。
- 非有限异常路径仍回退到 v59 的保守 elementwise float64 逻辑。
- loss finiteness 改为一次 stack/device reduction；epoch metric 在 device 上累计，epoch 结束只拷贝一次。

随机梯度验证：reference norm `1560.56383309`、optimized `1560.56246477`；clipped gradient 最大绝对差 `1.49e-8`、最大相对差 `9.81e-7`。当前 CPU norm-only microbenchmark 约 `23.68 ms -> 0.705 ms`（33.6x）；这不是完整 A30 training speedup。

### 10.2 `source_ports.py`

- PlanTF state 的 6 个独立 `Linear(1,D)` 保留原 parameters/state_dict，但 forward 合并为 broadcast affine。
- actor validity 不再用 `bool(tensor.any())` 触发 GPU->CPU sync；最多 16 个 actor slot 直接 dense temporal FPN，再 mask invalid。
- map speed valid/unknown 分支完全 tensorize，去掉两个 `any()` sync。
- all-masked attention row repair tensorize。
- GameFormer InitialDecoder 和每层 level-k decoder 从 actor Python loop 改成 `B*A` batch，保留每个 actor 的 own-future attention mask。

注意：eval/inference 数值已验证 allclose；training 中 dropout 的随机数消费顺序会因 batch vectorization 改变，因此 **固定 seed 下不保证与 v59 bitwise 相同训练轨迹**，但 dropout 分布、objective、网络结构、搜索预算和输入信息完全不变。

### 10.3 `data.py` / `evaluate.py`

GameFormer/PlanTF/PLUTO 的 `source_port_v54` 实际读取 `source_*` actor/map tensors 和 `prefix_traj`，旧的 `ego_history/neighbor_history/neighbor_valid` 对它们完全未使用。v60 不再构造、candidate-repeat、collate、pin-memory、H2D copy 这些冗余 tensor。legacy 非 source adapters 保持旧行为。

### 10.4 Contact exact box-QP

`robust_postimpact_control` 的 4-wheel QP 仍遍历数学上完全相同的 81 个 active-set face，但把 Python for-loop 编译成 cached affine KKT maps 后用 NumPy `einsum` 一次评估。没有改变约束、objective、bound、tolerance 或 first-min tie semantics。

### 10.5 offline launcher

Near：6 个方法一次 dataset pass，再拆出历史文件名。

Contact：

- heavy process: `postimpact_mpc_lite`
- fast process: 其余 Contact methods

两进程并行，6 次 dataset pass 降为 2 次。原 `DO_OFFLINE=false` 时完全不触发这部分逻辑。

## 11. 原命令兼容性

原有统一、Safe、Near、Contact launcher 的环境变量和入口脚本均保留。v60 只在 launcher 内部增加 runtime 环境清理和 offline batching；不要求修改你现有命令。

尤其要注意：你上传的统一命令默认 `DO_OFFLINE=false`，因此它主要训练 + closed-loop；若要验证 Contact offline 加速，需要显式设置 `DO_OFFLINE=true`。

## 12. 验证结果

本环境没有 A30 GPU，因此不能在这里伪造 A30 wall-clock；已做的是 CPU 数值等价、语法、契约和单元测试：

- 关键修改文件 `py_compile` 通过。
- unified/Safe/Near/Contact/install launcher `bash -n` 通过。
- 与 external baseline 相关的 72 个定向测试通过。
- 完整 `python -m pytest` 在原仓库就存在的历史测试依赖 `run_v47_two_gpu_fast_commands.txt` 缺失处失败；在停止前 186 tests passed。该缺文件不属于本次 v60 改动。
- 三份 microbenchmark/equivalence JSON 位于 `docs/v60_*_benchmark.json`。

## 13. 不建议做的“加速”

为了不改变 baseline 算法，本版本刻意没有做：

- 不降低 GameFormer level、mode 数、Transformer depth；
- 不降低 PlanTF/PLUTO mode 数或 future horizon；
- 不减少 Contact PSO 的 500 particles / 8 iterations；
- 不减少 robust scenario MPC beam/search budget；
- 不缩短 conformal calibration horizon；
- 不把 predictor-free baseline 换成近似 surrogate；
- 不用 `JAX_SKIP_CUDA_CONSTRAINTS_CHECK`；
- 不给 TensorFlow 安装另一套 CUDA/cuDNN；
- 不为了吞吐量使用 oracle/future/teacher 信息。

因此 v60 的加速属于 implementation-level：vectorization、减少 host sync、删除死数据搬运、shared computation、批量 exact active-set algebra、process scheduling。
