# OC-RAP v58 - Wayformer / BeTop / severity-minimization source-port audit

This revision audits four requested external baselines against the newly supplied papers/source and reliable public primary material:

- `wayformer_bc`
- `betopnet_lite`
- `severity_minimization`
- `compensatory_postimpact_mpc`

The first three receive substantive source/paper-core changes. `compensatory_postimpact_mpc` intentionally remains at the v57 source-limited ceiling because no reliable full-equation source or official implementation could be identified; adding unverified equations would reduce rather than improve fidelity.

## Wayformer (`wayformer_bc`)

Wayformer is a motion-forecasting architecture, not a native planner. v58 therefore reports it only as a **Safe-regime architecture/ego-BC control**, never as an official Wayformer planner reproduction.

v58 replaces the old candidate-token Transformer with a paper-structured adapter:

- homogeneous **early fusion** of observed agent-history and vector-map tokens;
- trainable **latent-query** scene bottleneck;
- learned output queries and repeated self/cross-attention decoder;
- multimodal **bivariate-Gaussian GMM** ego-future distribution;
- nearest-mode GMM NLL plus mode-classification loss;
- executable-candidate ranking by GMM trajectory likelihood.

The paper's large WOMD operator budget is explicitly projected for closed-loop runtime: the shipped v58 config uses 96 latent queries and 2 repeated decoder layers instead of the paper-scale 192/8. The operator is preserved, but checkpoint equivalence is not claimed. No author-official Wayformer code/checkpoint was identified; the uploaded BeTop reproduction and the public KIT-MRT reimplementation were used only as cross-checks.

## BeTop (`betopnet_lite`)

v58 is source-backed by the uploaded official OpenDriveLab repository. It restores the mechanisms that materially define BeTop topology reasoning:

- official behavior-braid segment-intersection operator;
- official nearest-map-polyline braid labeling with the 3 m gate;
- separate source/target `TopoFuser` projections and additive previous-topology residual;
- detached topology prediction before top-K local attention;
- source focal hard-topology mining and valid-edge loss normalization;
- six iterative topology-reasoning layers and top-K=32;
- paper Appendix-C short-horizon repulsive-potential planning term with `t_b=3` and `lambda_m=0.5`, projected onto the common executable candidate set.

The official repository currently provides the WOMD prediction implementation while the nuPlan planning pipeline is still TODO. Consequently v58 does **not** claim an official BeTop planner reproduction or source-checkpoint compatibility for the planning head. Existing OC-RAP NPZs omit other-agent GT futures, so the exact braid operator is applied to an observation-only constant-velocity future proxy; the default topology-loss weight is therefore 10 rather than the source prediction setting `weight_top=100`.

## Severity minimization (`severity_minimization`)

Mapped to Parseh, Nybacka & Asplund, *Motion planning for autonomous vehicles with the inclusion of post-impact motions for minimising collision risk* (Vehicle System Dynamics, 2023). The source method is **pre-impact unavoidable-collision motion planning**, not a post-contact recovery controller. v58 therefore moves its recommended OC-RAP evaluation from historical Contact legacy to **Near-contact legacy/control**. The old Contact flag is retained only for result compatibility and emits a deprecation warning.

v58 replaces the generic severity proxy with a paper-core collision/post-impact candidate port:

- vectorized rectangle SAT broad phase;
- exact Sutherland-Hodgman overlap polygon at first impact;
- overlap-polygon centroid POI;
- source-style overlap-intersection contact-plane construction with a documented degenerate SAT fallback;
- Kudlich-Slibar-style sticking/sliding collision impulse classification;
- source Table-A1 mass/inertia/geometry/friction/tire parameters;
- source-style 3DOF post-impact target rollout;
- Eq. (25) target-vehicle severity objective using lateral deviation, modulo-pi heading, yaw rate and sideslip with the source demonstration's equal weights.

The paper assumes a known target trajectory and uses an implicit 3DOF full-impact `fsolve` model including collision-interval tire terms. Serialized deployable OC-RAP samples intentionally omit other-agent futures. v58 therefore uses the nearest observed target plus constant-velocity extrapolation and a planar rigid-body sticking/sliding impact projection before the 3DOF rollout. These two interface substitutions are explicit in diagnostics/provenance and equation-exact full-impact equivalence is not claimed.

## Compensatory post-impact MPC (`compensatory_postimpact_mpc`)

The reliable primary source remains Cao et al., *Compensatory model predictive control for post-impact trajectory tracking via active front steering and differential torque vectoring* (2021). Public primary metadata/abstract verifies FCC-MPC, reverse steering, differential torque vectoring, constraint transformation for deteriorated post-impact states, and time-varying saturation on input/input-rate/slip ratio. Searches did not expose the full equations, weights, transform or an official implementation.

v58 therefore **retains the v57 source-limited structured adapter** rather than inventing an equation-level controller. It remains predictor-free, teacher-free, uses the observed-history pre-impact reference frame, and keeps the v57 fast dispatch. Diagnostics continue to state `source_limited_abstract_structured` and `requires_full_paper_for_equation_exact_port=True`.

## Regime/data/metric contract

- `wayformer_bc`, `betopnet_lite`: Safe legacy/architecture controls, enabled with `RUN_LEGACY_SAFE=true`. They use the same Safe train/val/test datasets and full Safe closed-loop publication metric contract.
- `severity_minimization`: Near-contact legacy/control, enabled with `RUN_LEGACY_NEAR=true`. It uses the same Near train/val/calibration/test/Waymax evaluator path and full Near publication metric contract.
- `compensatory_postimpact_mpc`: unchanged Contact main-table method and full Contact metric contract.
- `RUN_LEGACY_CONTACT=true` retains only the historical severity execution path and prints a clear source-regime mismatch warning.
- `run_all_regime_external_baselines_optimized.sh` now forwards `RUN_LEGACY_SAFE`, `RUN_LEGACY_NEAR` and `RUN_LEGACY_CONTACT` without changing defaults.

## Runtime optimization

No paper mechanism is removed silently for speed.

- Wayformer pools padded map points per polyline before homogeneous fusion and uses an explicitly declared 96-latent/2-decoder runtime budget.
- BeTop caches candidate-independent actor/map sorting and scene context once per scene-time group.
- Severity uses vectorized SAT for all candidate/time pairs and pays polygon clipping/contact geometry only at the first impact of colliding candidates.
- Cao retains the v57 early predictor/risk-profile bypass; no unsupported numerical solver was added.
- AMP, pinned-memory workers and existing two-GPU launcher compatibility are preserved for learned controls.

Synthetic CPU microbenchmarks are recorded in `docs/external_baselines/v58_microbenchmark.json`; they are kernel measurements, not full WOMD/Waymax wall-clock claims.

## Verification

- 68/68 relevant source-port, observation-only, loss, data-filter, closed-loop hot-path, CUDA/runtime, metric-summary and launcher-contract tests pass.
- Safe, Near-contact, Contact and all-regime shell launchers pass `bash -n`.
- New tests cover BeTop braid/map operators and TopoFuser residuals, Wayformer scene-conditioned GMM outputs and native loss backpropagation, Parseh collision/post-impact semantics and teacher/predictor invariance, Cao source-limited diagnostics, and v58 regime/reporting contracts.

The execution environment does not mount the user's `/data0/...` datasets, so v58 does not claim a full real-data Waymax benchmark run. The unchanged/extended server launchers should be used for final end-to-end wall-clock and metric measurements.
