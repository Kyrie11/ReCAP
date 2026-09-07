#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# shellcheck source=scripts/lib/v50_runtime.sh
source scripts/lib/v50_runtime.sh

: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${TRAIN_CONTACT:=$OCRAP_ROOT/train_contact}"
: "${VAL_CONTACT:=$OCRAP_ROOT/val_contact}"
: "${TEST_CONTACT:=$OCRAP_ROOT/test_contact}"
: "${RUN:=runs/contact_external_baselines}"
: "${WOMD_VAL_INTERACTIVE:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord@150}"
: "${CL_WOMD:=$WOMD_VAL_INTERACTIVE}"
: "${WOMD_NUM_SHARDS:=150}"
CL_WOMD="$(v50_normalize_womd_spec "$CL_WOMD" "$WOMD_NUM_SHARDS")"
: "${CL_MAX_SCENARIOS:=50}"
: "${CL_BUCKET_DATASET:=$TEST_CONTACT}"
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
: "${CL_NUM_RECOVERY_OPTIONS:=12}"
: "${CL_LABEL_MODE:=fast}"
: "${CL_AUDIT_EVERY_N_STEPS:=0}"
: "${CL_SAVE_PARTIAL:=true}"
: "${CL_PROFILE_TIMING:=true}"
: "${CL_RESUME_FORCE:=false}"
: "${CL_PARTIAL_WRITE_EVERY_SCENES:=32}"
: "${CL_PROGRESS_EVERY_STEPS:=10}"
: "${SKIP_COMPLETE_METHODS:=true}"
: "${DO_TRAIN:=true}"
: "${DO_OFFLINE:=true}"
: "${DO_CLOSED_LOOP:=true}"
: "${RUN_LEGACY_CONTACT:=false}"
: "${CUDA_DEVICES:=0,1}"
: "${MAX_PARALLEL:=2}"
: "${USE_DYNAMIC_SCHEDULER:=auto}"

IFS=',' read -r -a GPU_LIST <<< "$CUDA_DEVICES"
((${#GPU_LIST[@]})) || GPU_LIST=(0 1)
((MAX_PARALLEL >= 1)) || MAX_PARALLEL=1
((MAX_PARALLEL <= 2)) || MAX_PARALLEL=2
((MAX_PARALLEL <= ${#GPU_LIST[@]})) || MAX_PARALLEL="${#GPU_LIST[@]}"
CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 8)"
: "${THREADS_PER_JOB:=$(( CPU_COUNT / (2 * MAX_PARALLEL) ))}"
((THREADS_PER_JOB >= 1)) || THREADS_PER_JOB=1
((THREADS_PER_JOB <= 8)) || THREADS_PER_JOB=8
: "${JAX_CACHE_DIR:=$RUN/.jax_compilation_cache}"
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"
export RUN CL_WOMD
mkdir -p "$RUN" "$JAX_CACHE_DIR"

CONFIG=configs/external_baselines/contact_external_baselines.yaml
# Exactly the six post-contact main-table methods declared in provenance.py.
METHODS=(
  postimpact_mpc_lite
  post_crash_braking
  postimpact_motion_tvlqr
  post_collision_restoration
  compensatory_postimpact_mpc
  robust_postimpact_control
)
if v50_bool_true "$RUN_LEGACY_CONTACT"; then
  echo "[DEPRECATED] severity_minimization is a pre-impact unavoidable-collision planner and is now evaluated under Near-contact via RUN_LEGACY_NEAR=true. Keeping Contact execution only for v57 result compatibility; do not report it as a source-faithful Contact baseline." >&2
  METHODS+=(severity_minimization)
fi
METHODS_CSV="$(IFS=,; echo "${METHODS[*]}")"

run_env_gpu() {
  local gpu="$1"; shift
  local cache="$JAX_CACHE_DIR/gpu_${gpu//[^[:alnum:]_.-]/_}"
  mkdir -p "$cache"
  env CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB" \
    OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" NUMEXPR_NUM_THREADS="$THREADS_PER_JOB" \
    TF_NUM_INTRAOP_THREADS="$THREADS_PER_JOB" TF_NUM_INTEROP_THREADS=2 MALLOC_ARENA_MAX=4 \
    XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE" \
    TF_FORCE_GPU_ALLOW_GROWTH=true JAX_ENABLE_X64=0 \
    JAX_COMPILATION_CACHE_DIR="$cache" JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
    PYTHONUNBUFFERED=1 "$@"
}
run_env_cpu() {
  local cache="$JAX_CACHE_DIR/cpu"; mkdir -p "$cache"
  env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
    OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB" \
    OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" NUMEXPR_NUM_THREADS="$THREADS_PER_JOB" \
    TF_NUM_INTRAOP_THREADS="$THREADS_PER_JOB" TF_NUM_INTEROP_THREADS=2 MALLOC_ARENA_MAX=4 \
    JAX_COMPILATION_CACHE_DIR="$cache" PYTHONUNBUFFERED=1 "$@"
}

if v50_bool_true "$DO_CLOSED_LOOP" && v50_bool_true "$JAX_RUNTIME_PREFLIGHT"; then
  echo "[PREFLIGHT] validating JAX/Waymax GPU runtime on CUDA device ${GPU_LIST[0]}"
  run_env_gpu "${GPU_LIST[0]}" python tools/check_jax_waymax_runtime.py \
    --require-gpu --output "$RUN/jax_waymax_runtime_preflight.json"
fi

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

# IMPORTANT: all current Contact baselines are optimization/rule/controller
# adapters. Their training contract is dataset validation + train_summary.json;
# no neural checkpoint is expected. One shared dataset scan registers them all.
if v50_bool_true "$DO_TRAIN"; then
  run_env_cpu python -u tools/register_external_nonlearning_baselines.py \
    --config "$CONFIG" --dataset "$TRAIN_CONTACT" --val-dataset "$VAL_CONTACT" \
    --baselines "$METHODS_CSV" --output-root "$RUN" \
    2>&1 | tee "$RUN/register_nonlearning_contact.log"
fi

supports_wait_pid_capture() {
  help wait 2>/dev/null | grep -Eq -- '(^|[[:space:]])-p([[:space:]]|[[:punct:]])'
}
run_queue_dynamic() {
  local runner="$1"; shift; local -a items=("$@")
  local next=0 active=0 failed=0 done_pid status gpu item i
  declare -A PID_GPU=() PID_ITEM=()
  launch_one() { local x="$1" g="$2"; "$runner" "$x" "$g" & local p=$!; PID_GPU[$p]="$g"; PID_ITEM[$p]="$x"; active=$((active+1)); }
  for ((i=0;i<MAX_PARALLEL && next<${#items[@]};i++)); do launch_one "${items[$next]}" "${GPU_LIST[$i]}"; next=$((next+1)); done
  while ((active>0)); do
    done_pid=""; if wait -n -p done_pid; then status=0; else status=$?; fi
    gpu="${PID_GPU[$done_pid]}"; item="${PID_ITEM[$done_pid]}"; unset 'PID_GPU[$done_pid]' 'PID_ITEM[$done_pid]'; active=$((active-1))
    if ((status!=0)); then echo "[ERROR] $item failed on GPU $gpu (status=$status)" >&2; failed=1; fi
    if ((next<${#items[@]})); then launch_one "${items[$next]}" "$gpu"; next=$((next+1)); fi
  done
  return "$failed"
}
run_queue_fixed() {
  local runner="$1"; shift; local -a items=("$@") pids=() names=(); local base j idx failed=0
  for ((base=0;base<${#items[@]};base+=MAX_PARALLEL)); do
    pids=(); names=()
    for ((j=0;j<MAX_PARALLEL && base+j<${#items[@]};j++)); do idx=$((base+j)); "$runner" "${items[$idx]}" "${GPU_LIST[$j]}" & pids+=("$!"); names+=("${items[$idx]}"); done
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

eval_one() {
  local method="$1" gpu="$2"
  echo "[OFFLINE] contact method=$method gpu=$gpu"
  run_env_gpu "$gpu" python -u -m ocrap.cli evaluate-baseline \
    --config "$CONFIG" --dataset "$TEST_CONTACT" --split test \
    --output "$RUN/eval_contact_${method}.json" --baselines "$method" \
    2>&1 | tee "$RUN/eval_contact_${method}.log"
}
if v50_bool_true "$DO_OFFLINE"; then run_queue eval_one "${METHODS[@]}"; fi

run_closed_loop_method() {
  local method="$1" gpu="$2" target_args=()
  local output="$RUN/closed_loop_${method}.json"
  if v50_bool_true "$SKIP_COMPLETE_METHODS" && python tools/check_closed_loop_artifact.py --output "$output" --quiet; then
    echo "[REUSE] contact closed-loop method=$method is already complete: $output"
    return 0
  fi
  if [[ -n "$CL_TARGET_KEYS_FILE" ]]; then target_args=(--set "closed_loop.target_keys_file=$CL_TARGET_KEYS_FILE" --set closed_loop.require_target_keys=true); fi
  echo "[START] contact method=$method gpu=$gpu"
  run_env_gpu "$gpu" python -u -m ocrap.cli closed-loop \
    --config "$CONFIG" --dataset "$CL_WOMD" --output "$output" \
    --set "closed_loop.method=$method" \
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
    --set closed_loop.external_sparse_labels=true \
    --set closed_loop.exhaustive_teacher_labels=false \
    --set "closed_loop.num_candidate_prefixes=$CL_NUM_CANDIDATES" \
    --set "closed_loop.num_recovery_options=$CL_NUM_RECOVERY_OPTIONS" \
    --set "closed_loop.save_partial=$CL_SAVE_PARTIAL" \
    --set "closed_loop.partial_write_every_scenes=$CL_PARTIAL_WRITE_EVERY_SCENES" \
    --set "closed_loop.progress_every_steps=$CL_PROGRESS_EVERY_STEPS" \
    --set closed_loop.result_scene_detail=metrics \
    --set closed_loop.scene_journal_detail=metrics \
    --set closed_loop.memory_scene_detail=metrics \
    --set closed_loop.include_scenes_in_result=false \
    --set closed_loop.include_scenes_in_partial=false \
    --set "closed_loop.profile_timing=$CL_PROFILE_TIMING" \
    --set "closed_loop.audit_every_n_steps=$CL_AUDIT_EVERY_N_STEPS" \
    --set "closed_loop.resume_force=$CL_RESUME_FORCE" \
    --set waymax.dataloader_include_sdc_paths=false \
    --set waymax.compute_future_metrics=false \
    --set waymax.teacher_metrics_stride=0 \
    --set waymax.use_jit_scan_rollouts=true \
    "${target_args[@]}" \
    2>&1 | tee "$RUN/closed_loop_${method}.log"
  echo "[DONE] contact method=$method gpu=$gpu"
}

if v50_bool_true "$DO_CLOSED_LOOP"; then run_queue run_closed_loop_method "${METHODS[@]}"; fi

python tools/summarize_external_closed_loop.py \
  --run "$RUN" --regime contact --output "$RUN/closed_loop_summary.json" \
  --methods "$METHODS_CSV" --womd-spec "$CL_WOMD"
