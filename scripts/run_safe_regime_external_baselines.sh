#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# shellcheck source=scripts/lib/v50_runtime.sh
source scripts/lib/v50_runtime.sh

: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${TRAIN_SAFE:=$OCRAP_ROOT/train_safe}"
: "${VAL_SAFE:=$OCRAP_ROOT/val_safe}"
: "${TEST_SAFE:=$OCRAP_ROOT/test_safe}"
: "${RUN:=runs/safe_external_baselines}"
: "${WOMD_VAL:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150}"
: "${CL_WOMD:=$WOMD_VAL}"
: "${WOMD_NUM_SHARDS:=150}"
CL_WOMD="$(v50_normalize_womd_spec "$CL_WOMD" "$WOMD_NUM_SHARDS")"
: "${CL_MAX_SCENARIOS:=50}"
: "${CL_BUCKET_DATASET:=$TEST_SAFE}"
: "${CL_BUCKET_SPLIT:=test}"
: "${CL_MAX_TARGETS_PER_SCENE:=1}"
: "${CL_TARGET_KEYS_FILE:=}"
: "${CL_RENDER_TRACE:=false}"
: "${CL_RENDER_MAX_AGENTS:=48}"
: "${CL_PREFLIGHT:=true}"
: "${JAX_RUNTIME_PREFLIGHT:=true}"
: "${CL_MAX_STEPS:=40}"
: "${CL_REPLAN_INTERVAL_STEPS:=1}"
: "${CL_NUM_CANDIDATES:=24}"
: "${CL_LABEL_MODE:=fast}"
: "${CL_AUDIT_EVERY_N_STEPS:=0}"
: "${CL_SAVE_PARTIAL:=true}"
: "${CL_PROFILE_TIMING:=true}"
: "${CL_RESUME_FORCE:=false}"
: "${CL_PARTIAL_WRITE_EVERY_SCENES:=32}"
: "${CL_PROGRESS_EVERY_STEPS:=10}"
: "${SKIP_COMPLETE_METHODS:=true}"
: "${DO_TRAIN:=true}"
: "${FORCE_RETRAIN_SAFE:=false}"
: "${DO_OFFLINE:=true}"
: "${DO_CLOSED_LOOP:=true}"
: "${RUN_NOMINAL_CONTROL:=true}"
: "${RUN_LEGACY_SAFE:=false}"
: "${CUDA_DEVICES:=0,1}"
: "${MAX_PARALLEL:=2}"                    # two jobs at a time by default
: "${USE_DYNAMIC_SCHEDULER:=auto}"
: "${OCRAP_SDPA_BACKEND:=safe}"
: "${OCRAP_AMP_DTYPE:=auto}"
: "${CHECKPOINT_ROOT:=$RUN/checkpoints}"
: "${TRAIN_NUM_WORKERS_PER_JOB:=0}"       # 0 = auto from host CPU count

IFS=',' read -r -a GPU_LIST <<< "$CUDA_DEVICES"
((${#GPU_LIST[@]})) || GPU_LIST=(0 1)
((MAX_PARALLEL >= 1)) || MAX_PARALLEL=1
((MAX_PARALLEL <= 2)) || MAX_PARALLEL=2
((MAX_PARALLEL <= ${#GPU_LIST[@]})) || MAX_PARALLEL="${#GPU_LIST[@]}"
CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 8)"
: "${THREADS_PER_JOB:=$(( CPU_COUNT / (2 * MAX_PARALLEL) ))}"
((THREADS_PER_JOB >= 1)) || THREADS_PER_JOB=1
((THREADS_PER_JOB <= 8)) || THREADS_PER_JOB=8
if ((TRAIN_NUM_WORKERS_PER_JOB <= 0)); then
  TRAIN_NUM_WORKERS_PER_JOB=$(( CPU_COUNT / (2 * MAX_PARALLEL) ))
  ((TRAIN_NUM_WORKERS_PER_JOB >= 2)) || TRAIN_NUM_WORKERS_PER_JOB=2
  ((TRAIN_NUM_WORKERS_PER_JOB <= 8)) || TRAIN_NUM_WORKERS_PER_JOB=8
fi
: "${JAX_CACHE_DIR:=$RUN/.jax_compilation_cache}"
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"
export RUN CL_WOMD
mkdir -p "$RUN" "$CHECKPOINT_ROOT" "$JAX_CACHE_DIR"

if v50_bool_true "$DO_CLOSED_LOOP" && v50_bool_true "$CL_PREFLIGHT"; then
  preflight_target_args=()
  if [[ -n "$CL_TARGET_KEYS_FILE" ]]; then
    preflight_target_args=(--target-keys-file "$CL_TARGET_KEYS_FILE" --require-target-keys)
  fi
  python tools/check_closed_loop_dataset_support.py \
    --dataset "$CL_BUCKET_DATASET" --split "$CL_BUCKET_SPLIT" --womd-pattern "$CL_WOMD" \
    --expected-source-role auto "${preflight_target_args[@]}" \
    --output "$RUN/closed_loop_dataset_support.json"
fi

# method|config|kind|checkpoint
# These are the six main-table Safe methods declared in provenance.py.
SPECS=(
  "gameformer_lite|configs/external_baselines/gameformer_lite.yaml|learned|$CHECKPOINT_ROOT/gameformer_lite/best.pt|source_port_v54"
  "plantf|configs/external_baselines/plantf.yaml|learned|$CHECKPOINT_ROOT/plantf/best.pt|source_port_v54"
  "pluto|configs/external_baselines/pluto.yaml|learned|$CHECKPOINT_ROOT/pluto/best.pt|source_port_v54"
  "pdm_closed|configs/external_baselines/pdm_closed.yaml|nonlearning||"
  "pdm_hybrid|configs/external_baselines/pdm_hybrid.yaml|nonlearning||"
  "idm|configs/external_baselines/idm.yaml|nonlearning||"
)
# Wayformer and BeTop are architecture/topology controls rather than Safe
# main-table planners.  They are opt-in so the historical command remains
# unchanged, while RUN_LEGACY_SAFE=true trains/evaluates them through the same
# dataset and complete Safe metric contract.
if v50_bool_true "$RUN_LEGACY_SAFE"; then
  SPECS+=(
    "wayformer_bc|configs/external_baselines/wayformer_bc.yaml|learned|$CHECKPOINT_ROOT/wayformer_bc/best.pt|architecture_port_v58"
    "betopnet_lite|configs/external_baselines/betopnet_lite.yaml|learned|$CHECKPOINT_ROOT/betopnet_lite/best.pt|source_backed_topology_adapter_v58"
  )
fi

run_env_gpu() {
  local gpu="$1"; shift
  local cache="$JAX_CACHE_DIR/gpu_${gpu//[^[:alnum:]_.-]/_}"
  mkdir -p "$cache"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$gpu" OCRAP_TENSORFLOW_CPU_ONLY=1 \
    OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB" \
    OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" NUMEXPR_NUM_THREADS="$THREADS_PER_JOB" \
    TF_NUM_INTRAOP_THREADS="$THREADS_PER_JOB" TF_NUM_INTEROP_THREADS=2 MALLOC_ARENA_MAX=4 \
    XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE" \
    TF_FORCE_GPU_ALLOW_GROWTH=true JAX_ENABLE_X64=0 \
    JAX_COMPILATION_CACHE_DIR="$cache" JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
    PYTHONUNBUFFERED=1 "$@"
}

if v50_bool_true "$DO_CLOSED_LOOP" && v50_bool_true "$JAX_RUNTIME_PREFLIGHT"; then
  echo "[PREFLIGHT] validating JAX/Waymax GPU runtime on CUDA device ${GPU_LIST[0]}"
  run_env_gpu "${GPU_LIST[0]}" python tools/check_jax_waymax_runtime.py \
    --require-gpu --output "$RUN/jax_waymax_runtime_preflight.json"
fi

checkpoint_valid() {
  local ckpt="$1" expected_impl="${2:-source_port_v54}"
  [[ -n "$ckpt" && -f "$ckpt" ]] && python tools/validate_external_checkpoint.py \
    --checkpoint "$ckpt" --require-deployable-contract \
    --require-implementation-version "$expected_impl" >/dev/null 2>&1
}

# Validate/register the three rule/optimization baselines with ONE train/val scan.
# No .pt file is expected for these methods.
if v50_bool_true "$DO_TRAIN"; then
  python -u tools/register_external_nonlearning_baselines.py \
    --dataset "$TRAIN_SAFE" --val-dataset "$VAL_SAFE" \
    --specs "pdm_closed=configs/external_baselines/pdm_closed.yaml,pdm_hybrid=configs/external_baselines/pdm_hybrid.yaml,idm=configs/external_baselines/idm.yaml" \
    --output-root "$CHECKPOINT_ROOT" \
    2>&1 | tee "$RUN/register_nonlearning_safe.log"
fi

prepare_or_offline_method() {
  local spec="$1" gpu="$2" method config kind ckpt expected_impl train_dir
  IFS='|' read -r method config kind ckpt expected_impl <<< "$spec"
  expected_impl="${expected_impl:-source_port_v54}"
  if [[ "$kind" == learned ]]; then
    if ! v50_bool_true "$DO_OFFLINE" && v50_bool_true "$DO_CLOSED_LOOP" && v50_bool_true "$SKIP_COMPLETE_METHODS" \
        && python tools/check_closed_loop_artifact.py --output "$RUN/closed_loop_${method}.json" --quiet; then
      echo "[REUSE] safe method=$method already has a complete closed-loop artifact; checkpoint preparation skipped"
      return 0
    fi
    if v50_bool_true "$FORCE_RETRAIN_SAFE" || ! checkpoint_valid "$ckpt" "$expected_impl"; then
      if ! v50_bool_true "$DO_TRAIN"; then
        echo "Missing/invalid checkpoint and training disabled: $ckpt" >&2
        return 2
      fi
      train_dir="$(dirname "$ckpt")"; mkdir -p "$train_dir"
      echo "[TRAIN] safe method=$method gpu=$gpu"
      run_env_gpu "$gpu" python -u -m ocrap.cli train-baseline \
        --config "$config" --dataset "$TRAIN_SAFE" --val-dataset "$VAL_SAFE" \
        --baseline "$method" --output "$train_dir" \
        --set external_baselines.training.distributed=false \
        --set "external_baselines.training.num_workers=$TRAIN_NUM_WORKERS_PER_JOB" \
        --set external_baselines.training.tqdm=false \
        --set "external_baselines.training.sdpa_backend=$OCRAP_SDPA_BACKEND" \
        --set "external_baselines.training.amp_dtype=$OCRAP_AMP_DTYPE" \
        2>&1 | tee "$RUN/train_${method}.log"
      checkpoint_valid "$ckpt" "$expected_impl" || { echo "Training produced an invalid checkpoint: $ckpt" >&2; return 2; }
    else
      echo "[REUSE] validated checkpoint $ckpt"
    fi
  fi

  if v50_bool_true "$DO_OFFLINE"; then
    local checkpoint_args=()
    [[ "$kind" == learned ]] && checkpoint_args=(--checkpoint "$ckpt")
    echo "[OFFLINE] safe method=$method gpu=$gpu"
    run_env_gpu "$gpu" python -u -m ocrap.cli evaluate-baseline \
      --config "$config" --dataset "$TEST_SAFE" "${checkpoint_args[@]}" \
      --split test --output "$RUN/eval_safe_${method}.json" --baselines "$method" \
      2>&1 | tee "$RUN/eval_safe_${method}.log"
  fi
}

supports_wait_pid_capture() {
  help wait 2>/dev/null | grep -Eq -- '(^|[[:space:]])-p([[:space:]]|[[:punct:]])'
}

run_queue_dynamic() {
  local runner="$1"; shift; local -a items=("$@")
  local next=0 active=0 failed=0 done_pid status gpu item i
  declare -A PID_GPU=() PID_ITEM=()
  launch_one() {
    local x="$1" g="$2"
    "$runner" "$x" "$g" & local p=$!
    PID_GPU[$p]="$g"; PID_ITEM[$p]="$x"; active=$((active+1))
  }
  for ((i=0; i<MAX_PARALLEL && next<${#items[@]}; i++)); do
    launch_one "${items[$next]}" "${GPU_LIST[$i]}"; next=$((next+1))
  done
  while ((active>0)); do
    done_pid=""; if wait -n -p done_pid; then status=0; else status=$?; fi
    gpu="${PID_GPU[$done_pid]}"; item="${PID_ITEM[$done_pid]}"
    unset 'PID_GPU[$done_pid]' 'PID_ITEM[$done_pid]'; active=$((active-1))
    if ((status!=0)); then echo "[ERROR] ${item%%|*} failed on GPU $gpu (status=$status)" >&2; failed=1; fi
    if ((next<${#items[@]})); then launch_one "${items[$next]}" "$gpu"; next=$((next+1)); fi
  done
  return "$failed"
}

run_queue_fixed() {
  local runner="$1"; shift; local -a items=("$@")
  local base j idx failed=0; local -a pids=() names=()
  for ((base=0; base<${#items[@]}; base+=MAX_PARALLEL)); do
    pids=(); names=()
    for ((j=0; j<MAX_PARALLEL && base+j<${#items[@]}; j++)); do
      idx=$((base+j)); "$runner" "${items[$idx]}" "${GPU_LIST[$j]}" &
      pids+=("$!"); names+=("${items[$idx]%%|*}")
    done
    for j in "${!pids[@]}"; do wait "${pids[$j]}" || { echo "[ERROR] ${names[$j]} failed" >&2; failed=1; }; done
  done
  return "$failed"
}

run_queue() {
  local runner="$1"; shift; local use_dynamic=false
  case "${USE_DYNAMIC_SCHEDULER,,}" in
    1|true|yes|on) supports_wait_pid_capture || { echo "USE_DYNAMIC_SCHEDULER requested but Bash lacks wait -p" >&2; return 2; }; use_dynamic=true ;;
    auto|'') supports_wait_pid_capture && use_dynamic=true ;;
    0|false|no|off) use_dynamic=false ;;
    *) echo "Invalid USE_DYNAMIC_SCHEDULER=$USE_DYNAMIC_SCHEDULER" >&2; return 2 ;;
  esac
  if [[ "$use_dynamic" == true ]]; then run_queue_dynamic "$runner" "$@"; else run_queue_fixed "$runner" "$@"; fi
}

if v50_bool_true "$DO_TRAIN" || v50_bool_true "$DO_OFFLINE" || v50_bool_true "$DO_CLOSED_LOOP"; then
  run_queue prepare_or_offline_method "${SPECS[@]}"
fi

if v50_bool_true "$DO_OFFLINE" && v50_bool_true "$RUN_NOMINAL_CONTROL"; then
  env CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 python -u -m ocrap.cli evaluate-baseline \
    --config configs/external_baselines/nominal_log_replay.yaml \
    --dataset "$TEST_SAFE" --split test \
    --output "$RUN/eval_safe_nominal_log_replay.json" \
    --baselines nominal_replay,log_replay \
    2>&1 | tee "$RUN/eval_safe_nominal_log_replay.log"
fi

run_closed_loop_method() {
  local spec="$1" gpu="$2" method config kind ckpt expected_impl runtime_method
  IFS='|' read -r method config kind ckpt expected_impl <<< "$spec"
  expected_impl="${expected_impl:-source_port_v54}"
  runtime_method="$method"
  [[ "$method" == nominal_replay ]] && runtime_method=nominal
  local output="$RUN/closed_loop_${method}.json"
  if v50_bool_true "$SKIP_COMPLETE_METHODS" && python tools/check_closed_loop_artifact.py --output "$output" --quiet; then
    echo "[REUSE] safe closed-loop method=$method is already complete: $output"
    return 0
  fi
  local checkpoint_args=() target_args=()
  if [[ "$kind" == learned ]]; then
    checkpoint_valid "$ckpt" "$expected_impl" || { echo "Missing/invalid checkpoint: $ckpt" >&2; return 2; }
    checkpoint_args=(--checkpoint "$ckpt")
  fi
  if [[ -n "$CL_TARGET_KEYS_FILE" ]]; then
    target_args=(--set "closed_loop.target_keys_file=$CL_TARGET_KEYS_FILE" --set closed_loop.require_target_keys=true)
  fi
  echo "[START] safe closed-loop method=$method gpu=$gpu"
  run_env_gpu "$gpu" python -u -m ocrap.cli closed-loop \
    --config "$config" --dataset "$CL_WOMD" "${checkpoint_args[@]}" --output "$output" \
    --set "closed_loop.method=$runtime_method" \
    --set "closed_loop.max_scenarios=$CL_MAX_SCENARIOS" \
    --set "closed_loop.max_bucket_targets=$CL_MAX_SCENARIOS" \
    --set "closed_loop.bucket_dataset=$CL_BUCKET_DATASET" \
    --set "closed_loop.bucket_split=$CL_BUCKET_SPLIT" \
    --set closed_loop.require_bucket_targets=true \
    --set "closed_loop.max_targets_per_scene=$CL_MAX_TARGETS_PER_SCENE" \
    --set "closed_loop.render_trace=$CL_RENDER_TRACE" \
    --set "closed_loop.render_max_agents=$CL_RENDER_MAX_AGENTS" \
    --set "closed_loop.max_steps=$CL_MAX_STEPS" \
    --set "closed_loop.replan_interval_steps=$CL_REPLAN_INTERVAL_STEPS" \
    --set "closed_loop.label_mode=$CL_LABEL_MODE" \
    --set closed_loop.force_teacher_baselines=false \
    --set "closed_loop.num_candidate_prefixes=$CL_NUM_CANDIDATES" \
    --set "closed_loop.audit_every_n_steps=$CL_AUDIT_EVERY_N_STEPS" \
    --set "closed_loop.save_partial=$CL_SAVE_PARTIAL" \
    --set "closed_loop.resume_force=$CL_RESUME_FORCE" \
    --set "closed_loop.partial_write_every_scenes=$CL_PARTIAL_WRITE_EVERY_SCENES" \
    --set "closed_loop.progress_every_steps=$CL_PROGRESS_EVERY_STEPS" \
    --set closed_loop.result_scene_detail=metrics \
    --set closed_loop.scene_journal_detail=metrics \
    --set closed_loop.memory_scene_detail=metrics \
    --set closed_loop.include_scenes_in_result=false \
    --set closed_loop.include_scenes_in_partial=false \
    --set "closed_loop.profile_timing=$CL_PROFILE_TIMING" \
    --set waymax.dataloader_include_sdc_paths=false \
    --set waymax.compute_future_metrics=false \
    --set waymax.teacher_metrics_stride=0 \
    --set waymax.use_jit_scan_rollouts=true \
    "${target_args[@]}" \
    2>&1 | tee "$RUN/closed_loop_${method}.log"
  echo "[DONE] safe closed-loop method=$method gpu=$gpu"
}

if v50_bool_true "$DO_CLOSED_LOOP"; then
  CLOSED_LOOP_SPECS=("${SPECS[@]}")
  if v50_bool_true "$RUN_NOMINAL_CONTROL"; then
    CLOSED_LOOP_SPECS+=("nominal_replay|configs/external_baselines/nominal_log_replay.yaml|nonlearning|")
  fi
  run_queue run_closed_loop_method "${CLOSED_LOOP_SPECS[@]}"
fi

SUMMARY_METHODS=()
for spec in "${SPECS[@]}"; do
  IFS='|' read -r _method _config _kind _ckpt _impl <<< "$spec"
  SUMMARY_METHODS+=("$_method")
done
if v50_bool_true "$RUN_NOMINAL_CONTROL"; then SUMMARY_METHODS+=(nominal_replay); fi
SUMMARY_METHODS_CSV="$(IFS=,; echo "${SUMMARY_METHODS[*]}")"
python tools/summarize_external_closed_loop.py \
  --run "$RUN" --regime safe --output "$RUN/closed_loop_summary.json" \
  --methods "$SUMMARY_METHODS_CSV" --womd-spec "$CL_WOMD"

python - <<'PY'
import glob, json, os
run=os.environ['RUN']; offline=[]
for p in sorted(glob.glob(os.path.join(run,'eval_safe_*.json'))):
    try: d=json.load(open(p))
    except Exception: continue
    for method, values in (d.get('methods') or {}).items(): offline.append({'method':method, **values})
try:
    closed_doc=json.load(open(os.path.join(run,'closed_loop_summary.json')))
except Exception:
    closed_doc={'methods':[], 'missing_methods':[]}
out=os.path.join(run,'safe_external_baselines_summary.json')
json.dump({'offline':offline,'closed_loop':closed_doc.get('methods',[]),'missing_methods':closed_doc.get('missing_methods',[]),'womd_spec':os.environ.get('CL_WOMD')}, open(out,'w'), indent=2)
print({'event':'safe_external_baselines_summary','output':out,'offline_methods':len(offline),'closed_loop_methods':len(closed_doc.get('methods',[]))})
PY