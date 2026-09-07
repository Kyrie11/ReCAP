#!/usr/bin/env bash
# V48.110 OC-CATO: candidate-to-agent active-constraint topology audit after V48.109 RAW_SCENE_RELATIONAL_STOP.
# Audit only: no planner/source/Stage-I/root-decoder parameters are trained.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"; export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
REFERENCE_A="${V48110_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
L80_RUN="${V48110_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V93_AUDIT="${V48110_V93_AUDIT:-$BASE_OUT/OC-RAP-v48.93-factor-mediation-audit.jsonl}"
V109_PIPELINE="${V48110_V109_PIPELINE:-$BASE_OUT/OC-RAP-v48.109-PIPELINE_COMPLETE.json}"
V109_COMPARE="${V48110_V109_COMPARE:-$BASE_OUT/OC-RAP-v48.109-DCP-DRFC-BCDE-RIFA-OC-RCSO-comparison.json}"
TRAIN_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl"; DEV_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl"
CERT_INDEX="${V48110_CERT_INDEX:-$BASE_OUT/OC-RAP-v48.96-certificate-teacher-pcd-index.jsonl}"
CACHE="${V48110_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_110_cato_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.110-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.110-CATO-balanced.json"; POUT="$BASE_OUT/OC-RAP-v48.110-CATO-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.110-CATO-balanced.pt"; PSTATE="$BASE_OUT/OC-RAP-v48.110-CATO-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.110-DCP-DRFC-BCDE-RIFA-OC-CATO-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.110-PIPELINE_COMPLETE.json"; AUDITS_ZIP="$BASE_OUT/OC-RAP-v48.110-OC-CATO-audits.zip"
mkdir -p "$BASE_OUT" "$CACHE"; rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$AUDITS_ZIP"

python tools/check_v48_110_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V109_PIPELINE" "$V109_COMPARE" <<'PY'
import hashlib,json,pathlib,sys
p,c=map(pathlib.Path,sys.argv[1:]); want='4aa8d8846a39de6fa3797464ce3e9587c148872a7a46d5a8f118fcba9c983627'
for x in (p,c):
    if not x.is_file(): raise SystemExit(f'missing V48.110 prerequisite {x}')
if hashlib.sha256(c.read_bytes()).hexdigest()!=want: raise SystemExit('V48.109 authoritative comparison SHA mismatch')
pd=json.loads(p.read_text()); cd=json.loads(c.read_text()); d=cd.get('preregistered_decision') or {}
if not(pd.get('valid') and pd.get('attribution_ready') and pd.get('engineering_version')=='v48.109.0-OC-RCSO' and pd.get('preregistered_status')=='RAW_SCENE_RELATIONAL_STOP'):
    raise SystemExit('V48.109 RAW_SCENE_RELATIONAL_STOP pipeline prerequisite missing')
if not(cd.get('valid') and cd.get('attribution_ready') and d.get('status')=='RAW_SCENE_RELATIONAL_STOP' and d.get('next_branch')=='close_bilinear_relational_orientation_then_preregister_candidate_to_agent_constraint_topology_audit_no_training_or_source_sweep'):
    raise SystemExit('V48.109 candidate-to-agent topology branch prerequisite missing')
PY
for f in "$TRAIN_INDEX" "$DEV_INDEX" "$CERT_INDEX" "$V93_AUDIT"; do [[ -s "$f" ]] || { echo "missing prerequisite $f" >&2; exit 30; }; done

run_one(){
  local v="$1"; local gpu="$2"; local out="$3"; local state="$4"
  local ckpt="$L80_RUN/candidates/$v/model_v48_trac_sr/best.pt"
  [[ -f "$ckpt" ]] || { echo "missing L80 checkpoint $ckpt" >&2; return 30; }
  CUDA_VISIBLE_DEVICES="$gpu" python tools/run_v48_110_candidate_agent_topology_orientation_audit.py \
    --checkpoint "$ckpt" --train-index "$TRAIN_INDEX" --dev-index "$DEV_INDEX" --certificate-index "$CERT_INDEX" \
    --v93-audit "$V93_AUDIT" --cache-dir "$CACHE/$v" --device cuda --variant "$v" --output "$out" --state-output "$state"
}
set +e
run_one balanced "$GPU0" "$BOUT" "$BSTATE" & p0=$!
run_one precision "$GPU1" "$POUT" "$PSTATE" & p1=$!
wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
[[ $r0 == 0 && $r1 == 0 ]] || { echo "V48.110 CATO run failure balanced=$r0 precision=$r1" >&2; exit 30; }
python tools/compare_v48_110_cato.py --balanced "$BOUT" --precision "$POUT" --v109-pipeline "$V109_PIPELINE" --v109-comparison "$V109_COMPARE" --output "$COMPARE"
python tools/check_v48_110_pipeline_complete.py --runtime "$RUNTIME" --balanced "$BOUT" --precision "$POUT" --balanced-state "$BSTATE" --precision-state "$PSTATE" --comparison "$COMPARE" --v48-109-pipeline "$V109_PIPELINE" --v48-109-comparison "$V109_COMPARE" --output "$COMPLETE"
cd "$BASE_OUT"; zip -qj "$AUDITS_ZIP" "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE"
printf 'V48.110 complete. Upload:\n%s\n%s\n%s\n' "$BOUT" "$POUT" "$AUDITS_ZIP"
