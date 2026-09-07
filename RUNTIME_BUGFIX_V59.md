# OC-RAP external baseline runtime bugfix v59

This patch keeps the v58 external-baseline algorithm/fidelity definitions intact and fixes runtime/data-loading/environment failure modes observed on the server.

## 1. Safe regime: AMP scatter dtype mismatch

Observed failure:

`RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and BFloat16 for the source.`

Root cause: `SourcePointsEncoder` (and two sibling source-port scatter paths) allocated FP32 destination buffers from source storage tensors, while PyTorch autocast legitimately returned BF16/FP16 encoded features. `Tensor[index] = value` does not implicitly reconcile these dtypes.

Fix:
- allocate scatter buffers from the computed encoder output (`enc.new_zeros(...)`), not the FP32 storage tensor;
- make `SourceAgentEncoder` choose its runtime buffer dtype from the active encoded branch;
- explicitly cast source map speed embeddings into the destination dtype.

This preserves the network architecture, weights, loss, and AMP policy. It only makes the implementation dtype-correct.

## 2. Near-contact CPSF calibration: WOMD payload type mismatch

Observed failure: `validation_tfexample.tfrecord-*` was sent to `scenario_pb2.Scenario.ParseFromString`, causing `google.protobuf.message.DecodeError`.

Root cause: WOMD `uncompressed/tf_example/.../validation_tfexample.tfrecord@150` contains TFExample records and is the format consumed by Waymax. The legacy Scenario-proto reader is a different data path.

Fix:
- CPSF calibration now reuses `iter_waymax_womd_scenarios_selected` / `iter_waymax_womd_scenarios` from the production Waymax loader;
- use `source_scenario_index` for sparse targeted replay when available, preserving the same object truncation/order/SDC mapping used to build OC-RAP datasets;
- cross-check canonical/official scene identity and fail closed on provenance mismatch;
- calibration defaults to `CUDA_VISIBLE_DEVICES=''` and `JAX_PLATFORMS=cpu` because it does not require a GPU;
- the legacy Scenario reader now detects a TFExample payload best-effort and raises an actionable format-mismatch error;
- fixed serialization of valid `source_scenario_index=0` (previous `value or -1` logic converted zero to -1). Existing datasets remain compatible via the persisted `__wx########` scene suffix fallback.

This changes no CPSF conformal formula or calibration statistic; it only recovers the correct raw trajectories from the correct record format.

## 3. Contact closed-loop: JAX CUDA PJRT mismatch

Observed failure: `Mismatched PJRT plugin PJRT API version (0.58) and framework PJRT API version (0.54)` before any contact policy is evaluated.

Root cause: the installed JAX framework/jaxlib and CUDA PJRT/plugin packages are from incompatible releases. This is an environment error, not a post-impact planner error.

Fix in code:
- Safe/Near/Contact launchers run `tools/check_jax_waymax_runtime.py --require-gpu` before expensive end-to-end work and write a JSON diagnostic in the run directory;
- CPU-only registration/calibration paths force the CPU JAX backend, so an unrelated broken CUDA plugin cannot break those tasks;
- the preflight prints installed JAX/JAXlib/CUDA-plugin/Waymax/TensorFlow versions and a clean-repair hint.

The GPU environment itself must still be repaired; silently falling back Contact Waymax evaluation to CPU would hide a deployment error and distort runtime comparisons.

### Recommended environment repair

First inspect the driver and current stack:

```bash
nvidia-smi
python tools/check_jax_waymax_runtime.py --require-gpu
python -m pip check
```

For an NVIDIA driver compatible with CUDA 12 (Linux driver >= 525), the least ambiguous clean repair is:

```bash
python -m pip uninstall -y \
  jax jaxlib \
  jax-cuda12-plugin jax-cuda12-pjrt \
  jax-cuda13-plugin jax-cuda13-pjrt
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda12]"
python -m pip check
python tools/check_jax_waymax_runtime.py --require-gpu
```

If the machine has a sufficiently new driver for CUDA 13 (Linux driver >= 580), use `"jax[cuda13]"` instead. If you intentionally depend on a system-installed CUDA toolkit rather than pip-managed CUDA libraries, use the matching `jax[cuda12-local]` or `jax[cuda13-local]` extra.

For pip-managed CUDA wheels, JAX recommends avoiding an `LD_LIBRARY_PATH` that overrides its bundled NVIDIA libraries. If the reinstall still fails, test:

```bash
env -u LD_LIBRARY_PATH python tools/check_jax_waymax_runtime.py --require-gpu
```

Waymax itself can be refreshed separately if needed:

```bash
python -m pip install --upgrade \
  'git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax'
```

Do not reinstall protobuf/TensorFlow to fix the Near calibration DecodeError: that failure was a code-level TFExample-vs-Scenario parser mismatch.

## 4. Additional latent training issue found by the scan

The legacy/generic `GameFormerFutureEncoder` detached its internal predicted trajectory before the feature MLP, which made the generic adapter unable to propagate trajectory gradients. The detach was removed **only in the legacy/generic adapter**.

The main-table source-faithful GameFormer port (`GameFormerFutureEncoderSource` in `source_ports.py`) still preserves the source/paper detach semantics. Therefore the main GameFormer reproduction is unchanged.

## 5. Validation

- External-baseline / Waymax / launcher / loss / source-port regression suite: **104/104 passed**.
- Current v48.105-v48.111 main-algorithm regression suite: **44/44 passed** (8 existing PyTorch Transformer nested-tensor warnings only).
- Total targeted tests: **148/148 passed**.
- Safe/Near/Contact/all-regime launchers pass `bash -n`.
- `src`, `tools`, and tests pass `compileall`.

The real `/data0/...` WOMD/OC-RAP dataset and the server CUDA stack are not mounted in this environment, so the final full Waymax GPU run must be re-executed on the server after the JAX stack is repaired.
