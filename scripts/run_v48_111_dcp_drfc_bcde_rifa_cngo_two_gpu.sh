#!/usr/bin/env bash
# V48.111 OC-CNGO: constraint-native signed-gap / normal-flow audit after V48.110 CATO STOP.
# Audit only: no planner/source/Stage-I/root-decoder parameters are trained.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"; export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
REFERENCE_A="${V48111_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
L80_RUN="${V48111_L80:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
V93_AUDIT="${V48111_V93_AUDIT:-$BASE_OUT/OC-RAP-v48.93-factor-mediation-audit.jsonl}"
V110_PIPELINE="${V48111_V110_PIPELINE:-$BASE_OUT/OC-RAP-v48.110-PIPELINE_COMPLETE.json}"
V110_COMPARE="${V48111_V110_COMPARE:-$BASE_OUT/OC-RAP-v48.110-DCP-DRFC-BCDE-RIFA-OC-CATO-comparison.json}"
V110_B="${V48111_V110_BALANCED:-$BASE_OUT/OC-RAP-v48.110-CATO-balanced.json}"
V110_P="${V48111_V110_PRECISION:-$BASE_OUT/OC-RAP-v48.110-CATO-precision.json}"
TRAIN_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl"; DEV_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl"
CERT_INDEX="${V48111_CERT_INDEX:-$BASE_OUT/OC-RAP-v48.96-certificate-teacher-pcd-index.jsonl}"
CACHE="${V48111_INPUT_CACHE:-$BASE_OUT/.ocrap_v48_111_cngo_cache}"
RUNTIME="$BASE_OUT/OC-RAP-v48.111-runtime-code-contract.json"
BOUT="$BASE_OUT/OC-RAP-v48.111-CNGO-balanced.json"; POUT="$BASE_OUT/OC-RAP-v48.111-CNGO-precision.json"
BSTATE="$BASE_OUT/OC-RAP-v48.111-CNGO-balanced.pt"; PSTATE="$BASE_OUT/OC-RAP-v48.111-CNGO-precision.pt"
COMPARE="$BASE_OUT/OC-RAP-v48.111-DCP-DRFC-BCDE-RIFA-OC-CNGO-comparison.json"
COMPLETE="$BASE_OUT/OC-RAP-v48.111-PIPELINE_COMPLETE.json"; AUDITS_ZIP="$BASE_OUT/OC-RAP-v48.111-OC-CNGO-audits.zip"
mkdir -p "$BASE_OUT" "$CACHE"; rm -f "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE" "$AUDITS_ZIP"

python tools/check_v48_111_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V110_PIPELINE" "$V110_COMPARE" "$V110_B" "$V110_P" <<'PY'
import hashlib,json,pathlib,sys
p,c,b,q=map(pathlib.Path,sys.argv[1:])
want_c='5bb9bbac2b5a88cb9419308804afdfce22643cd986df284224e1c9f3617e1c9d'
want_b='850c56e1d8fb015a2a05011390642d3c770288133a8a115c86aaed38e283daf5'
want_q='fd57374d98e3c13abde70829ee41426ba3d487b38d6cbecbfcfd5ba882738ed9'
for x in (p,c,b,q):
    if not x.is_file(): raise SystemExit(f'missing V48.111 prerequisite {x}')
if hashlib.sha256(c.read_bytes()).hexdigest()!=want_c: raise SystemExit('V48.110 authoritative comparison SHA mismatch')
if hashlib.sha256(b.read_bytes()).hexdigest()!=want_b: raise SystemExit('V48.110 balanced result SHA mismatch')
if hashlib.sha256(q.read_bytes()).hexdigest()!=want_q: raise SystemExit('V48.110 precision result SHA mismatch')
pd=json.loads(p.read_text()); cd=json.loads(c.read_text()); d=cd.get('preregistered_decision') or {}
if not(pd.get('valid') and pd.get('attribution_ready') and pd.get('engineering_version')=='v48.110.0-OC-CATO' and pd.get('preregistered_status')=='CANDIDATE_AGENT_TOPOLOGY_STOP'):
    raise SystemExit('V48.110 CANDIDATE_AGENT_TOPOLOGY_STOP pipeline prerequisite missing')
if not(cd.get('valid') and cd.get('attribution_ready') and d.get('status')=='CANDIDATE_AGENT_TOPOLOGY_STOP' and d.get('next_branch')=='close_coordinatewise_agent_topology_then_preregister_constraint_native_candidate_agent_geometry_audit_no_training_or_source_sweep'):
    raise SystemExit('V48.110 constraint-native geometry branch prerequisite missing')
PY
for f in "$TRAIN_INDEX" "$DEV_INDEX" "$CERT_INDEX" "$V93_AUDIT"; do [[ -s "$f" ]] || { echo "missing prerequisite $f" >&2; exit 30; }; done

run_one(){
  local v="$1"; local gpu="$2"; local out="$3"; local state="$4"
  local ckpt="$L80_RUN/candidates/$v/model_v48_trac_sr/best.pt"
  [[ -f "$ckpt" ]] || { echo "missing L80 checkpoint $ckpt" >&2; return 30; }
  CUDA_VISIBLE_DEVICES="$gpu" python tools/run_v48_111_constraint_native_geometry_orientation_audit.py \
    --checkpoint "$ckpt" --train-index "$TRAIN_INDEX" --dev-index "$DEV_INDEX" --certificate-index "$CERT_INDEX" \
    --v93-audit "$V93_AUDIT" --cache-dir "$CACHE/$v" --device cuda --variant "$v" --output "$out" --state-output "$state"
}
set +e
run_one balanced "$GPU0" "$BOUT" "$BSTATE" & p0=$!
run_one precision "$GPU1" "$POUT" "$PSTATE" & p1=$!
wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
[[ $r0 == 0 && $r1 == 0 ]] || { echo "V48.111 CNGO run failure balanced=$r0 precision=$r1" >&2; exit 30; }
python tools/compare_v48_111_cngo.py \
  --balanced "$BOUT" --precision "$POUT" \
  --v110-pipeline "$V110_PIPELINE" --v110-comparison "$V110_COMPARE" --v110-balanced "$V110_B" --v110-precision "$V110_P" \
  --output "$COMPARE"
python tools/check_v48_111_pipeline_complete.py \
  --runtime "$RUNTIME" --balanced "$BOUT" --precision "$POUT" --balanced-state "$BSTATE" --precision-state "$PSTATE" \
  --comparison "$COMPARE" --v48-110-pipeline "$V110_PIPELINE" --v48-110-comparison "$V110_COMPARE" --output "$COMPLETE"
cd "$BASE_OUT"; zip -qj "$AUDITS_ZIP" "$RUNTIME" "$BOUT" "$POUT" "$BSTATE" "$PSTATE" "$COMPARE" "$COMPLETE"
printf 'V48.111 complete. Upload:\n%s\n%s\n%s\n' "$BOUT" "$POUT" "$AUDITS_ZIP"
