#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck source=scripts/lib/v50_runtime.sh
source scripts/lib/v50_runtime.sh

: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${OUT:=runs/all_regime_external_baselines_v50}"
: "${CUDA_DEVICES:=0,1}"
: "${OCRAP_SDPA_BACKEND:=safe}"
: "${OCRAP_AMP_DTYPE:=auto}"
: "${MAX_SCENARIOS:=0}"
: "${MAX_STEPS:=40}"
: "${RUN_SAFE:=1}"
: "${RUN_NEAR:=1}"
: "${RUN_CONTACT:=1}"
: "${RUN_LEGACY_SAFE:=false}"
: "${RUN_LEGACY_NEAR:=false}"
: "${RUN_LEGACY_CONTACT:=false}"
: "${DO_TRAIN_SAFE:=true}"       # train only missing/invalid learned checkpoints
: "${DO_TRAIN_NEAR:=true}"
: "${DO_TRAIN_CONTACT:=true}"
: "${DO_CALIBRATE_NEAR:=true}"
: "${FORCE_RECALIBRATE_NEAR:=false}"
: "${FORCE_RETRAIN_ALL:=false}"
: "${DO_OFFLINE:=true}"
: "${DO_CLOSED_LOOP:=true}"
: "${RENDER_TRACES:=false}"      # full runs are metric-only; trace only selected 10 scenes later
: "${RUN_ORACLE_CLOSED_LOOP:=false}"
: "${CONTINUE_AFTER_REGIME_FAILURE:=true}"
: "${SKIP_COMPLETE_METHODS:=true}"
: "${USE_DYNAMIC_SCHEDULER:=auto}"
: "${WOMD_NUM_SHARDS:=150}"
: "${WOMD_VAL:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150}"
: "${WOMD_VAL_INTERACTIVE:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord@150}"
: "${SAFE_CL_WOMD:=$WOMD_VAL}"
: "${NEAR_CL_WOMD:=$WOMD_VAL_INTERACTIVE}"
: "${CONTACT_CL_WOMD:=$WOMD_VAL_INTERACTIVE}"
: "${EXTERNAL_CHECKPOINT_ROOT:=$OUT/checkpoints}"
: "${SAFE_CHECKPOINT_ROOT:=$EXTERNAL_CHECKPOINT_ROOT/safe}"
: "${NEAR_CHECKPOINT_ROOT:=$EXTERNAL_CHECKPOINT_ROOT/near}"

SAFE_CL_WOMD="$(v50_normalize_womd_spec "$SAFE_CL_WOMD" "$WOMD_NUM_SHARDS")"
NEAR_CL_WOMD="$(v50_normalize_womd_spec "$NEAR_CL_WOMD" "$WOMD_NUM_SHARDS")"
CONTACT_CL_WOMD="$(v50_normalize_womd_spec "$CONTACT_CL_WOMD" "$WOMD_NUM_SHARDS")"
mkdir -p "$OUT" "$OUT/safe" "$OUT/near" "$OUT/contact" "$SAFE_CHECKPOINT_ROOT" "$NEAR_CHECKPOINT_ROOT"

write_phase() {
  local regime="$1" status="$2" rc="$3" started="$4" ended="$5"
  python - "$OUT/$regime.phase.json" "$regime" "$status" "$rc" "$started" "$ended" <<'PY'
import json, pathlib, sys
path=pathlib.Path(sys.argv[1]); path.parent.mkdir(parents=True,exist_ok=True)
json.dump({'regime':sys.argv[2],'status':sys.argv[3],'exit_code':int(sys.argv[4]),'started_at':sys.argv[5],'ended_at':sys.argv[6]},path.open('w'),indent=2)
PY
}

FINALIZED=0
finalize_index() {
  local rc=$?
  [[ "$FINALIZED" == 1 ]] && return 0
  FINALIZED=1
  set +e
  python tools/build_external_baseline_run_index.py \
    --root "$OUT" --closed-loop-enabled "$DO_CLOSED_LOOP" \
    --oracle-enabled "$RUN_ORACLE_CLOSED_LOOP" --launcher-exit-code "$rc"
  return 0
}
trap finalize_index EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
# Publish a recoverable index before starting long-running workers.
python tools/build_external_baseline_run_index.py \
  --root "$OUT" --closed-loop-enabled "$DO_CLOSED_LOOP" \
  --oracle-enabled "$RUN_ORACLE_CLOSED_LOOP" --launcher-exit-code 1 >/dev/null || true

failed=0
if ! python tools/audit_external_baseline_fidelity.py \
  --output-json "$OUT/EXTERNAL_BASELINE_FIDELITY.json" \
  --output-md "$OUT/EXTERNAL_BASELINE_FIDELITY.md"; then
  echo "[ERROR] baseline fidelity audit failed" >&2; failed=1
fi

common=(
  OCRAP_ROOT="$OCRAP_ROOT"
  CUDA_DEVICES="$CUDA_DEVICES"
  OCRAP_SDPA_BACKEND="$OCRAP_SDPA_BACKEND"
  OCRAP_AMP_DTYPE="$OCRAP_AMP_DTYPE"
  CL_MAX_SCENARIOS="$MAX_SCENARIOS"
  CL_MAX_STEPS="$MAX_STEPS"
  CL_SAVE_PARTIAL=true
  CL_RESUME_FORCE=false
  CL_RENDER_TRACE="$RENDER_TRACES"
  DO_OFFLINE="$DO_OFFLINE"
  DO_CLOSED_LOOP="$DO_CLOSED_LOOP"
  SKIP_COMPLETE_METHODS="$SKIP_COMPLETE_METHODS"
  WOMD_NUM_SHARDS="$WOMD_NUM_SHARDS"
)

run_regime() {
  local regime="$1" enabled="$2" started ended rc status
  started="$(v50_iso_now)"
  if [[ "$enabled" != 1 ]]; then
    write_phase "$regime" skipped 0 "$started" "$(v50_iso_now)"; return 0
  fi
  write_phase "$regime" running 0 "$started" ""
  case "$regime" in
    safe)
      if env "${common[@]}" RUN="$OUT/safe" DO_TRAIN="$DO_TRAIN_SAFE" \
        FORCE_RETRAIN_SAFE="$FORCE_RETRAIN_ALL" CHECKPOINT_ROOT="$SAFE_CHECKPOINT_ROOT" \
        RUN_LEGACY_SAFE="$RUN_LEGACY_SAFE" CL_WOMD="$SAFE_CL_WOMD" CL_RENDER_TRACE=false \
        bash scripts/run_safe_regime_external_baselines.sh \
        > >(tee "$OUT/safe.launcher.log") 2>&1; then rc=0; else rc=$?; fi
      ;;
    near)
      if env "${common[@]}" RUN="$OUT/near" DO_TRAIN="$DO_TRAIN_NEAR" \
        DO_CALIBRATE="$DO_CALIBRATE_NEAR" FORCE_RECALIBRATE="$FORCE_RECALIBRATE_NEAR" \
        RUN_ORACLE_CLOSED_LOOP="$RUN_ORACLE_CLOSED_LOOP" RUN_LEGACY_NEAR="$RUN_LEGACY_NEAR" CL_WOMD="$NEAR_CL_WOMD" \
        USE_DYNAMIC_SCHEDULER="$USE_DYNAMIC_SCHEDULER" \
        bash scripts/run_near_contact_external_baselines_2gpu_optimized.sh \
        > >(tee "$OUT/near.launcher.log") 2>&1; then rc=0; else rc=$?; fi
      ;;
    contact)
      if env "${common[@]}" RUN="$OUT/contact" DO_TRAIN="$DO_TRAIN_CONTACT" CL_WOMD="$CONTACT_CL_WOMD" \
        RUN_LEGACY_CONTACT="$RUN_LEGACY_CONTACT" USE_DYNAMIC_SCHEDULER="$USE_DYNAMIC_SCHEDULER" \
        bash scripts/run_contact_external_baselines.sh \
        > >(tee "$OUT/contact.launcher.log") 2>&1; then rc=0; else rc=$?; fi
      ;;
    *) echo "unknown regime: $regime" >&2; rc=2 ;;
  esac
  ended="$(v50_iso_now)"; status=complete; ((rc==0)) || status=failed
  write_phase "$regime" "$status" "$rc" "$started" "$ended"
  if ((rc!=0)); then
    failed=1
    echo "[ERROR] $regime regime failed with exit code $rc; see $OUT/$regime.launcher.log" >&2
    v50_bool_true "$CONTINUE_AFTER_REGIME_FAILURE" || return "$rc"
  fi
  return 0
}

run_regime safe "$RUN_SAFE" || failed=1
run_regime near "$RUN_NEAR" || failed=1
run_regime contact "$RUN_CONTACT" || failed=1

# Write once before EXIT as well, so the index is visible immediately.
python tools/build_external_baseline_run_index.py \
  --root "$OUT" --closed-loop-enabled "$DO_CLOSED_LOOP" \
  --oracle-enabled "$RUN_ORACLE_CLOSED_LOOP" --launcher-exit-code "$failed"
FINALIZED=1

((failed==0)) || exit 1
