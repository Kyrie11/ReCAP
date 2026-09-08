#!/usr/bin/env bash
# V48.111 evidence-frozen deployable submission harness.
# Scientific contract:
#   - CNRO remains audit-only and is NEVER injected by this launcher.
#   - Evaluate the frozen V48.80 checkpoint/calibration owner used by the V48.111 audit chain.
#   - Fail closed on checkpoint mechanism drift, policy-order drift, or calibration drift.
#   - No finetuning, no recalibration, no dataset reconstruction, no regime router.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
MODEL_RUN="${MODEL_RUN:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
VARIANTS="${VARIANTS:-balanced,precision}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
MAX_SCENARIOS="${MAX_SCENARIOS:-0}"
MAX_STEPS="${MAX_STEPS:-40}"
NUM_CANDIDATES="${NUM_CANDIDATES:-24}"
NUM_RECOVERY_OPTIONS="${NUM_RECOVERY_OPTIONS:-12}"
RUN_CNRO_AUDIT="${RUN_CNRO_AUDIT:-0}"
DEPLOYMENT_PROFILE="${DEPLOYMENT_PROFILE:-v48_80_evidence_frozen}"
OUT_ROOT="${OUT_ROOT:-$BASE_OUT/ocrap_v48_111_submission_three_regime}"
SUMMARY_JSON="$OUT_ROOT/V48.111-SUBMISSION-THREE-REGIME-SUMMARY.json"
SUMMARY_CSV="$OUT_ROOT/V48.111-SUBMISSION-THREE-REGIME-SUMMARY.csv"
SUMMARY_MD="$OUT_ROOT/V48.111-SUBMISSION-THREE-REGIME-SUMMARY.md"

mkdir -p "$OUT_ROOT"

[[ "$DEPLOYMENT_PROFILE" == "v48_80_evidence_frozen" ]] || { echo "Unsupported deployment profile: $DEPLOYMENT_PROFILE" >&2; exit 30; }
[[ "$RUN_CNRO_AUDIT" == 0 ]] || { echo "RUN_CNRO_AUDIT must remain 0 in the deployable submission launcher. Run the audit launcher separately." >&2; exit 30; }

# Runtime provenance for the actual V48.111 source tree.
python tools/check_v48_111_runtime_code_contract.py \
  --repo "$REPO" \
  --output "$OUT_ROOT/OC-RAP-v48.111-runtime-code-contract.submission.json"

IFS=',' read -r -a variants <<< "$VARIANTS"
((${#variants[@]})) || { echo 'No variants requested.' >&2; exit 2; }

for variant in "${variants[@]}"; do
  variant="$(echo "$variant" | xargs)"
  [[ -n "$variant" ]] || continue
  ckpt="$MODEL_RUN/candidates/$variant/model_v48_trac_sr/best.pt"
  gamma="$MODEL_RUN/candidates/$variant/calibration/gamma_rec_by_bucket_v48.json"
  [[ -f "$ckpt" ]] || { echo "Missing frozen checkpoint: $ckpt" >&2; exit 30; }
  [[ -f "$gamma" ]] || { echo "Missing frozen bucket calibration: $gamma" >&2; exit 30; }

  # Freeze the actual deployed stack before any held-out closed-loop execution.
  # This prevents a late checkpoint swap from silently re-opening a historically STOP mechanism.
  python tools/check_v48_111_deployable_stack.py \
    --model-run "$MODEL_RUN" \
    --variant "$variant" \
    --output "$OUT_ROOT/V48.111-DEPLOYABLE-STACK-${variant}.json"

  out="$OUT_ROOT/$variant"
  echo "[V48.111 submission] evaluating variant=$variant -> $out"
  MODEL_RUN="$MODEL_RUN" \
  MODEL_VARIANT="$variant" \
  CUDA_DEVICES="$CUDA_DEVICES" \
  OCRAP_ROOT="$OCRAP_ROOT" \
  OUT="$out" \
  MAX_SCENARIOS="$MAX_SCENARIOS" \
  MAX_STEPS="$MAX_STEPS" \
  NUM_CANDIDATES="$NUM_CANDIDATES" \
  NUM_RECOVERY_OPTIONS="$NUM_RECOVERY_OPTIONS" \
  ALLOW_DIAGNOSTIC_RC20=1 \
  RUN_SAFE=1 RUN_NEAR=1 RUN_CONTACT=1 \
  SKIP_COMPLETE_REGIMES=true FINALIZE_COMPLETE_JOURNALS=true \
  bash scripts/run_ocrap_three_regime_closed_loop.sh

done

python tools/summarize_v48_111_submission_three_regime.py \
  --root "$OUT_ROOT" \
  --variants "$VARIANTS" \
  --output-json "$SUMMARY_JSON" \
  --output-csv "$SUMMARY_CSV" \
  --output-md "$SUMMARY_MD"

printf '\nV48.111 submission evaluation complete.\nSummary JSON: %s\nSummary CSV: %s\nSummary Markdown: %s\n' \
  "$SUMMARY_JSON" "$SUMMARY_CSV" "$SUMMARY_MD"
