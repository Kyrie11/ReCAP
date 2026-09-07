#!/usr/bin/env bash
# V48.109 OC-RCSO: preregistered raw candidate x scene convex relational orientation audit after V48.108.1 RAW_ACTION_PATHWAY_STOP.
# Audit only: no planner/source/Stage-I/root-decoder parameters are trained.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"; export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
REFERENCE_A="${V48109_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
L80_RUN="${V48109_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V93_AUDIT="${V48109_V93_AUDIT:-$BASE_OUT/OC-RAP-v48.93-factor-mediation-audit.jsonl}"
V108_PIPELINE="${V48109_V108_PIPELINE:-$BASE_OUT/OC-RAP-v48.108-PIPELINE_COMPLETE.json}"
V108_COMPARE="${V48109_V108_COMPARE:-$BASE_OUT/OC-RAP-v48.108-DCP-DRFC-BCDE-RIFA-OC-RPAP-comparison.json}"
TRAIN_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl"; DEV_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl"
CERT_INDEX="${V48109_CERT_INDEX:-$BASE_OUT/OC-RAP-v48.96-certificate-teacher-pcd-index.jsonl}"
CACHE="${V48109_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_109_rcso_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.109-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.109-RCSO-balanced.json"; POUT="$BASE_OUT/OC-RAP-v48.109-RCSO-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.109-RCSO-balanced.pt"; PSTATE="$BASE_OUT/OC-RAP-v48.109-RCSO-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.109-DCP-DRFC-BCDE-RIFA-OC-RCSO-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.109-PIPELINE_COMPLETE.json"; AUDITS_ZIP="$BASE_OUT/OC-RAP-v48.109-OC-RCSO-audits.zip"
mkdir -p "$BASE_OUT" "$CACHE"; rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$AUDITS_ZIP"

python tools/check_v48_109_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V108_PIPELINE" "$V108_COMPARE" <<'PY'
import json,pathlib,sys
p,c=map(pathlib.Path,sys.argv[1:])
for x in (p,c):
    if not x.is_file(): raise SystemExit(f'missing V48.109 prerequisite {x}')
pd=json.loads(p.read_text()); cd=json.loads(c.read_text()); d=cd.get('preregistered_decision') or {}
if not(pd.get('valid') and pd.get('attribution_ready') and pd.get('engineering_version')=='v48.108.1-OC-RPAP' and pd.get('preregistered_status')=='RAW_ACTION_PATHWAY_STOP'):
    raise SystemExit('V48.108.1 RAW_ACTION_PATHWAY_STOP pipeline prerequisite missing')
if not(cd.get('valid') and cd.get('attribution_ready') and d.get('status')=='RAW_ACTION_PATHWAY_STOP' and d.get('projection_structural_injectivity') is True and d.get('next_branch')=='raw_candidate_pathway_insufficient_then_preregister_raw_candidate_scene_relational_interaction_audit_no_training_or_source_sweep'):
    raise SystemExit('V48.108.1 raw candidate-scene relational branch prerequisite missing')
PY
for f in "$TRAIN_INDEX" "$DEV_INDEX" "$CERT_INDEX" "$V93_AUDIT"; do [[ -s "$f" ]] || { echo "missing prerequisite $f" >&2; exit 30; }; done

run_one(){
  local v="$1"; local gpu="$2"; local out="$3"; local state="$4"
  local ckpt="$L80_RUN/candidates/$v/model_v48_trac_sr/best.pt"
  [[ -f "$ckpt" ]] || { echo "missing L80 checkpoint $ckpt" >&2; return 30; }
  CUDA_VISIBLE_DEVICES="$gpu" python tools/run_v48_109_raw_candidate_scene_orientation_audit.py \
    --checkpoint "$ckpt" --train-index "$TRAIN_INDEX" --dev-index "$DEV_INDEX" --certificate-index "$CERT_INDEX" \
    --v93-audit "$V93_AUDIT" --cache-dir "$CACHE/$v" --device cuda --variant "$v" --output "$out" --state-output "$state"
}
set +e
run_one balanced "$GPU0" "$BOUT" "$BSTATE" & p0=$!
run_one precision "$GPU1" "$POUT" "$PSTATE" & p1=$!
wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
[[ $r0 == 0 && $r1 == 0 ]] || { echo "V48.109 RCSO run failure balanced=$r0 precision=$r1" >&2; exit 30; }
python tools/compare_v48_109_rcso.py --balanced "$BOUT" --precision "$POUT" --v108-pipeline "$V108_PIPELINE" --v108-comparison "$V108_COMPARE" --output "$COMPARE"
python tools/check_v48_109_pipeline_complete.py --runtime "$RUNTIME" --balanced "$BOUT" --precision "$POUT" --balanced-state "$BSTATE" --precision-state "$PSTATE" --comparison "$COMPARE" --v48-108-pipeline "$V108_PIPELINE" --v48-108-comparison "$V108_COMPARE" --output "$COMPLETE"
cd "$BASE_OUT"; zip -qj "$AUDITS_ZIP" "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE"
printf 'V48.109 complete. Upload:\n%s\n%s\n%s\n' "$BOUT" "$POUT" "$AUDITS_ZIP"
