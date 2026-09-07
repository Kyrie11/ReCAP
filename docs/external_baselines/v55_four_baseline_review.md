# OC-RAP external-baseline reproduction review - v55

## Scope

This revision audits and optimizes the four uploaded references in `external_baselines.zip`:

1. MARC - Li et al., *MARC: Multipolicy and Risk-aware Contingency Planning for Autonomous Driving*.
2. RACP - Mustafa et al., *RACP: Risk-Aware Contingency Planning with Multi-Modal Predictions*; the uploaded archive is the authors' source repository.
3. Robust Scenario MPC - Batkovic et al., *A Robust Scenario MPC Approach for Uncertain Multi-Modal Obstacles*.
4. Predictive Safety Filter (PSF) - Wabersich and Zeilinger, *A Predictive Safety Filter for Learning-Based Control of Constrained Nonlinear Dynamical Systems*.

All four are Near-Contact main-table methods. The public reporting name intentionally contains `candidate-lattice port` where the original continuous optimizer cannot be reproduced under the shared OC-RAP executable-candidate interface.

## Reproduction contract

The goal is not to claim checkpoint/API compatibility with methods whose native optimizer, simulator or dataset is different. The auditable contract is:

- retain the paper/source mechanism that changes the decision rule;
- give every Near-Contact method the same observation-only multi-modal prediction interface;
- project continuous controls/trajectory trees only onto the same executable candidate lattice;
- never use OC-RAP teacher recoverability tensors during deployable selection;
- keep paper/source gaps explicit in `src/ocrap/external_baselines/provenance.py` and the generated fidelity manifest;
- evaluate all methods with the same Waymax closed-loop candidate-generation and metric path.

## Main changes

### MARC

Previous issue: the dynamic branch point was derived from ego candidate-vs-nominal divergence, which is not the paper definition.

v55:

- constructs semantic policy families;
- forms policy-conditioned mode responses from executable candidates;
- computes the latest branch time from pairwise divergence of those scenario-conditioned ego futures;
- keeps a shared prefix and mode-specific compatible continuation;
- applies upper-tail CVaR only to the safety component and probability expectation to non-safety costs;
- enumerates the finite candidate lattice instead of pretending to reproduce the original LP+iLQR bi-level optimizer.

Reporting class: **paper-core / optimizer-interface adapted**.

### RACP

Previous issue: the port did not retain the released source's shared-plan + belief-weighted contingent-plan cost topology.

v55:

- follows the uploaded source's shared-plan plus mode-conditioned contingent-tail structure;
- uses the shared observation-only mode belief as branch weights;
- preserves non-anticipativity on the shared prefix;
- preserves the source Frenet 2.0 s shared + 1.0 s contingent relative timing as a normalized 2:1 split on OC-RAP's shorter common horizon;
- vectorizes the candidate-by-mode recourse search.

The CommonRoad/Frenet planner, CasADi/OSQP branch MPC and backup-controller propagation are not reproduced. This is stated explicitly rather than hidden.

Reporting class: **source-structured / optimizer-interface adapted**.

### Robust Scenario MPC

Previous issue: one global latest branch time was used. The paper instead constrains each mode pair to share inputs only until that pair is distinguishable.

v55:

- computes pairwise mode-distinguishability times;
- enforces pairwise non-anticipative input/state compatibility before each pair-specific split;
- allows mode-dependent recourse after distinction;
- requires physical/collision constraints for every mode while mode probabilities affect the expected objective;
- removes the unpublished internal-risk hard gate from the default paper-core configuration;
- uses a bounded beam search over mode-candidate tuples to avoid combinatorial Python loops.

The source paper's tube/reachable-set tightening and recursive-feasibility theorem are **not** claimed for this point-hypothesis candidate port.

Reporting class: **paper-core / optimizer-interface adapted**.

### Predictive Safety Filter

Previous issue: the implementation mixed in a CBF gate although the cited Wabersich-Zeilinger PSF is an MPC-style predictive safety filter, not a CBF-QP baseline.

v55:

- treats the logged nominal candidate as the proposed input;
- returns it unchanged when finite-horizon state/input constraints and the terminal backup condition are satisfied;
- otherwise selects the admitted executable candidate with minimum normalized control-sequence deviation;
- uses a stopping-distance backup certificate as an explicit terminal-safe-set proxy;
- keeps old CBF-named aliases for command compatibility only; no CBF mechanism is attributed to the paper.

Reporting class: **paper-core / optimizer-interface adapted**.

## Dataset and information-flow audit

The launchers consume the split directories under a single `OCRAP_ROOT`:

- `train_near_contact`
- `val_near_contact`
- `calibration_near_contact`
- `test_near_contact`

The four audited selectors use candidate kinematics/controls, route utility, prefix physical feasibility, visible actor history, and the shared observation-only predictor. `m_star`, `r_orc_star`, `r_dep_star`, `hard_violation` and `harm_proxy` are not deployable selector inputs. A regression test mutates these teacher/evaluation fields and requires the selected action and score of every non-oracle external method to remain unchanged.

`feasible` is safe to use here because it is the prefix generator's finite/control/route-topology executable flag; it is not the OC-RAP deployable-recoverability teacher label.

The conformal baseline remains the only Near-Contact method whose algorithm requires a held-out calibration step; it fits its threshold on `calibration_near_contact`, not the test split. `DO_TRAIN=true` for these six non-learning Near methods is deliberately a registration/data-validation step and produces no neural checkpoint.

## Metrics retained in external closed-loop summaries

Near-Contact compact summaries now retain, rather than silently drop:

- collision/off-road scene and step rates;
- minimum clearance and TTC, including lower-tail scene statistics;
- Near-Contact/critical-TTC/near-zero-clearance exposure rates and durations;
- exposure episode count and longest exposure run;
- time-to-minimum, terminal clearance/TTC, recovery gains and deficit AUCs;
- closed-loop FRA_exec, FRA_cand, DRS, ODG and bounded NUP;
- intervention statistics and comfort statistics;
- the complete nested `waymax_metrics` dictionary for forward compatibility.

Safe and Contact publication summaries were also expanded so the external-baseline reporting layer cannot drop their regime-specific metrics.

Important: OC-RAP's source-calibration diagnostics such as signed recoverability-margin calibration are not fabricated for external methods. With `CL_LABEL_MODE=selected`, teacher-derived extreme metrics are attached only after action selection. A paired selected-vs-nominal harmful-selection confidence bound would require a separately preregistered teacher audit and is not claimed by this revision.

## Performance changes

The shared observation-risk forecast is constructed once per candidate group and all candidates are scored in a vectorized batch. A 24-candidate synthetic apples-to-apples microbenchmark with observation-conditioned reweighting disabled gives:

- v54 median: 11.17 ms/group
- v55 median: 6.46 ms/group
- speedup: about 1.73x
- the checked risk outputs are numerically identical in this legacy-prior comparison.

MARC and RACP recourse evaluation were also changed from Python candidate/mode loops to broadcasted reductions; Robust Scenario MPC uses a bounded beam instead of full Cartesian enumeration. The existing launcher already keeps the 2-GPU dynamic queue, resumable outputs and persistent JAX cache.

## Verification completed in this sandbox

- 48 focused external-baseline / launcher / closed-loop / metric-contract tests passed.
- `bash -n` passed for Safe, Near-Contact and Contact external launchers.
- fidelity manifest regenerated successfully for all 26 registered main/legacy/diagnostic entries.
- paper/source mechanism tests cover PSF zero intervention when safe, MARC scenario-derived branch time, Robust Scenario MPC pairwise non-anticipativity, and no default unpublished loss constraint.

The sandbox does **not** mount `/data0/senzeyu2/dataset/OCRAP` or the raw WOMD directory, so a real end-to-end Waymax rollout could not be executed here. Run the unchanged user commands on the training host before freezing paper numbers.

## Command compatibility

No command-line change is required. In particular, the existing Near-Contact command remains valid:

```bash
OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP \
CUDA_DEVICES=0,1 \
RUN=/home/senzeyu2/code/OC-RAP/runs/near_external_v50 \
CL_MAX_SCENARIOS=0 \
CL_LABEL_MODE=selected \
DO_TRAIN=true DO_CALIBRATE=true \
DO_OFFLINE=false DO_CLOSED_LOOP=true \
bash scripts/run_near_contact_external_baselines_2gpu_optimized.sh
```

Safe and Contact launchers are untouched at their public command interface.

## Recommended next four audits

1. `dr_cvar_safety_filter` - obtain the Safaoui & Summers ICRA 2024 paper and preferably the official `TSummersLab/dr-cvar-safety_filtering` source. Current implementation is only a mechanism-inspired Wasserstein-CVaR bound and is the highest-priority Near-Contact remaining risk.
2. `conformal_predictive_safety_filter` - obtain the Strawn, Ayanian & Lindemann RA-L 2023 paper and any author source if available. Current scalar binary admission is materially weaker than trajectory-wise conformal uncertainty intervals plus a learned filter.
3. `postimpact_mpc_lite` - obtain Wang et al., *Integrated Post-Impact Planning and Active Safety Control for Autonomous Vehicles* (T-IV 2023), plus source if available. Current implementation is objective-level only.
4. `postimpact_motion_tvlqr` - obtain *Post-Impact Motion Planning and Tracking Control for Autonomous Vehicles* (2022), plus source if available. Current implementation compresses polynomial/APF planning + TVLQR + torque allocation into candidate-level proxies.

These four give the best next-round payoff: they finish the two unaudited Near-Contact main methods and start with the two most algorithmically structured Contact methods.
