# OC-RAP v57: Contact external baseline literature mapping and implementation audit

## Scope

This pass starts from the uploaded v56 tree and audits four Contact-regime baselines against primary online material:

| OC-RAP baseline | Primary paper/source | Online source quality | v57 reporting level |
|---|---|---|---|
| `post_crash_braking` | Jianbo Lu et al., *A System for Autonomous Braking of a Vehicle Following Collision*, SAE 2017-01-1581, 2017, DOI 10.4271/2017-01-1581 | SAE primary abstract + related Ford inventor patent full text | paper-core PIB semantics / patent-supported interface adaptation |
| `post_collision_restoration` | Samsaptak Ghosh, M. Felix Orlando, Sohom Chakrabarty, *Post-Collision Trajectory Restoration for a Single-track Ackermann Vehicle using Heuristic Steering and Tractive Force Functions*, arXiv:2602.08444, 2026 | full arXiv preprint/source | paper-core mathematical port / source ambiguity documented |
| `compensatory_postimpact_mpc` | Mingcong Cao et al., *Compensatory model predictive control for post-impact trajectory tracking via active front steering and differential torque vectoring*, Proc. IMechE Part D 235(4):903-919, 2021, DOI 10.1177/0954407020979087 | reliable SAGE metadata/abstract, but not enough full equations for equation-exact reconstruction | source-limited structured adapter; **not equation-exact** |
| `robust_postimpact_control` | Di Ao et al., *Advanced post-impact safety and stability control for electric vehicles*, IET Intelligent Transport Systems, 2022, DOI 10.1049/itr2.12230 | full open Wiley article with equations/tables | paper-core mathematical port / plant-interface adapted |

Primary links:

- SAE Lu 2017: https://saemobilus.sae.org/papers/a-system-autonomous-braking-a-vehicle-following-collision-2017-01-1581
- Ford related patent: https://patents.google.com/patent/US9205815B2/en
- Ghosh 2026: https://arxiv.org/abs/2602.08444
- Cao 2021: https://doi.org/10.1177/0954407020979087
- Ao 2022: https://doi.org/10.1049/itr2.12230

## 1. Post-crash braking -> Lu et al. 2017 PIB/PIBA

The SAE paper presents two post-impact brake functions. PIBA assists a driver who is already braking; PIB autonomously requests braking up to ABS level when the driver is not braking. WOMD does not expose brake-pedal/driver-intent status, so the benchmark cannot honestly distinguish PIBA from PIB. v57 therefore explicitly evaluates the autonomous no-driver PIB branch rather than synthesizing a driver state.

The related Ford patent family shares the Lu/Hammoud/Clark/Hofmann/Lakehal-Ayat/Farmer/Shomsky/Schaefer inventor set and provides additional implementation examples: autonomous brake pressure can be limited by ABS, post-collision kinetic energy is reduced, undesirable motion below about 8 km/h can be treated as no longer requiring PIB, and termination can occur after stabilization/stationary state or an example ~2.5 s interval. These numbers are labelled **patent-supported adapter defaults**, not SAE-paper numerical parameters.

v57 implementation:

- collision/post-contact trigger is supplied by the Contact benchmark state, not re-detected from unavailable airbag/RCM signals;
- commanded longitudinal deceleration is `min(mu*g, configured actuator max)` and projected to the common executable candidate set;
- steering is only a tie-breaker, not a fabricated obstacle planner;
- release occurs after the motion/timeout condition;
- no learned actor predictor, risk-profile objective, teacher future or recovery oracle enters the selection.

This is substantially more faithful than the v56 `stable_stop` risk-weighted heuristic, but it is still not a hydraulic ABS simulator. The honest name is **Post-impact braking (Lu et al. PIB port)**.

## 2. Post-collision restoration -> Ghosh et al. 2026

The arXiv paper proposes an open-loop recovery law for a generalized single-track Ackermann model, jointly commanding steering and tractive force. v57 ports the reported absolute-time steering/force laws rather than rescaling them to the OC-RAP horizon:

- steering direction composition (paper Eq. 10);
- first sine steering pulse (Eq. 13);
- second sine steering-pulse interface (Eq. 16);
- initial + compensatory tractive force (Eq. 18);
- compensatory sine force pulse (Eq. 21).

A source ambiguity is deliberately not hidden: the paper equation requires a second-pulse amplitude `A2`, while the reported parameter table includes an unexplained `K1` and does not report `A2`. v57 defaults `A2=0` and exposes `source_A2_unreported=True` / the table `K1` diagnostic. It does **not** silently equate `K1` with `A2`. The reported cases also do not define an online case classifier; v57 uses an observed initial-yaw-rate threshold only as a clearly labelled benchmark adapter.

WOMD has no wheel tractive-force state, so `F_i` is represented by the common nominal longitudinal acceleration and the paper's compensatory force becomes `F_c/m` as an incremental acceleration reference. The selection objective fits the open-loop law only; it intentionally does not add a learned future-agent predictor because the source method does not use one.

## 3. Compensatory post-impact MPC -> Cao et al. 2021

The reliable SAGE primary page establishes the following mechanisms: compensatory MPC with feedforward-feedback compensation (FCC), sufficient reverse steering plus differential torque vectoring, a constraint transformation for collision-deteriorated states that may start outside the traditional stability envelope, and time-varying saturation on input, input rate and slip ratio. It also states faster attenuation of lateral/yaw deviation and CarSim-Simulink evaluation.

However, the accessible online material used in this pass does not expose enough of the full controller equations, model matrices, cost weights, exact constraint transform or wheel allocation details to support an equation-exact implementation. v57 therefore deliberately **downgrades the claim** instead of inventing missing formulas.

The current v57 adapter retains only source-verified structure:

- reverse-steering/FCC preference against the observed collision-induced lateral/yaw state;
- differential-yaw-action intent through executable steering/acceleration candidates;
- a contracting admissible envelope as an explicitly labelled benchmark projection of “constraint transformation”;
- limits on input, input rate and a conservative longitudinal-utilisation proxy because wheel slip is not observable in WOMD;
- lateral, heading, yaw-rate and sideslip attenuation objective;
- observed-history pre-impact reference path, with no teacher future and no learned obstacle predictor.

The diagnostics contain `fidelity=source_limited_abstract_structured` and `requires_full_paper_for_equation_exact_port=True`. **This is the one main-table method for which I recommend uploading the full paper or source code next.**

## 4. Robust post-impact control -> Ao et al. 2022

The Wiley full article is sufficiently complete for a formula-level port. v57 implements the published two-level controller:

- Eqs. 17-19: yaw-rate/course and lateral-deviation errors and `s=c1*e1+c2*e2`;
- Eqs. 21-23: vehicle/impact decomposition, reaching law and requested additional yaw moment;
- Eqs. 26-27: in-wheel-motor fault gain model;
- Eqs. 28-34: tyre-utilisation objective, force/yaw mapping and constrained quadratic allocation;
- source vehicle/controller parameters including `m=1270 kg`, `Iz=1536.7 kg m^2`, `a=1.015 m`, `b=1.895 m`, track `1.675 m`, wheel radius `0.325 m`, `Kf=50000 N/rad`, `Kr=65000 N/rad`, `mu=0.85`, `k1=0.25`, `k2=0.005`, `c1=0.6`, `c2=1`, allocation `xi=0.5`.

The benchmark begins from observed post-impact states and does not expose the impact-force history, so the explicit impact-force contribution is zero after the observed impact interval. WOMD also has no IWM fault-diagnosis signal or four wheel torques; healthy normalized motor gain is the default, and the published QP is used as an actuator-realizability certificate/ranking term before Waymax executes the common acceleration/steering action.

The 4-wheel box QP is solved exactly for this small dimension using batch active-set enumeration. All 3^4 active/bound statuses are pre-factored and cached, making the paper-core implementation materially faster without changing the QP objective or bounds.

## Dataset and closed-loop integration

These four methods are Contact-only main-table methods. v57 keeps all existing aliases and launcher commands unchanged. The key interface fixes are:

- all four dispatch **before** the generic learned multimodal risk-profile path, so they do not receive an extra predictor not present in the cited algorithms;
- post-impact lateral/course controllers use a pre-impact path estimated from **observed ego history only**. The OC-RAP history is current-ego-centric, so using the history line preserves the post-impact displacement/course error; using the current disturbed pose as reference would incorrectly zero that error;
- repeated closed-loop replanning receives only a runtime `elapsed_contact_s = step_idx * contact_dt`, allowing Lu/Ghosh absolute-time laws to progress without modifying stored dataset labels;
- no `hard_violation`, `harm_proxy`, `m_star`, future trajectory, recovery teacher or other oracle target is read by these source ports.

The real `/data0/...` datasets are not mounted in the execution environment used for this audit, so this pass validates the data/launcher/selection contracts but does **not** claim a full real-WOMD Waymax run.

## Contact publication metrics

All four selected policies flow through the same existing Contact-regime Waymax evaluator and publication summarizer. The summary contract retained by regression tests includes overlap episode count/duration, longest overlap run, post-contact terminal clearance, post-contact free-space AUC and normalized AUC, clearance gain and time-to-peak clearance, escape scene rate/time, re-contact scene rate/count, secondary-overlap rate, stable-stop rate/quality/time, post-contact overlap duration/rate, clearance-deficit AUC, plus shared comfort/kinematic diagnostics. No method-specific surrogate replaces the common Contact metric contract.

## Verification and performance

58/58 relevant regression tests pass, including new tests for PIB activation/termination, Ghosh absolute-time pulse semantics/source ambiguity, observed-history reference frame, exact Ao QP correctness, source sliding surface/gains, predictor/teacher independence and explicit Cao fidelity diagnostics. Contact/Near/all-regime launchers pass `bash -n`.

Synthetic CPU selector benchmark with 24 executable candidates (not full Waymax wall clock):

| method | v56 median | v57 median | median speedup |
|---|---:|---:|---:|
| post-crash braking | 7.91 ms | 0.68 ms | 11.6x |
| post-collision restoration | 11.77 ms | 0.65 ms | 18.0x |
| compensatory post-impact MPC | 7.46 ms | 0.73 ms | 10.3x |
| robust post-impact control | 7.50 ms | 3.67 ms | 2.0x |

The first three gains mostly come from early bypass of the irrelevant learned risk predictor/profile. Ao is computationally heavier in v57 because it now actually evaluates the published sliding-mode + four-wheel QP structure; vectorization and factorization caching still halve the selector time relative to v56.

## Other external baselines that still merit audit

### Priority 1: `severity_minimization` (legacy Contact)

Corresponding paper: Masoumeh Parseh, Mikael Nybacka, Fredrik Asplund, **Motion planning for autonomous vehicles with the inclusion of post-impact motions for minimising collision risk**, *Vehicle System Dynamics*, 61(6):1707-1733, DOI 10.1080/00423114.2022.2088396 (published online 2022; volume 2023). Open full text is available.

The paper is more than a scalar crash-severity score: it combines a motion-planning trajectory library, vehicle dynamics, accident reconstruction/collision modelling, and a 4DOF post-impact longitudinal/lateral/yaw/roll model. The current OC-RAP `severity_minimization` is only a finite-lattice severity/stability proxy. v57 therefore downgrades its provenance to **objective-level / requires post-impact-model source port**. If this method appears in any reported table or appendix, it is the next strongest candidate for a paper-core port. Uploading the paper is optional because an open full text is available online, but author code would help if it exists.

### Priority 2: `betopnet_lite` (legacy Safe)

Corresponding paper: **Reasoning Multi-Agent Behavioral Topology for Interactive Autonomous Driving (BeTop/BeTopNet)**, NeurIPS 2024, with official OpenDriveLab repository. The official repository currently states that the full WOMD **prediction** implementation is provided while the **nuPlan planning pipeline remains TODO**. Therefore the current candidate-lattice implementation must remain labelled a **BeTop topology-aware adapter**, not an official/reproduced planner. If you need it in a planner comparison table, either obtain unreleased/updated planning code or keep it as an architecture/mechanism ablation.

### Priority 3: `wayformer_bc` (legacy Safe)

Corresponding paper: **Wayformer: Motion Forecasting via Simple & Efficient Attention Networks**, arXiv:2207.05844 (2022). The paper is explicitly a **motion-forecasting** model, not a planning algorithm. The current `wayformer_bc` may be useful as a Wayformer-style encoder + route-conditioned BC control, but calling it “Wayformer planner reproduction” would be inaccurate. Recommended reporting name remains **Wayformer-style route BC** or move it to an architecture ablation rather than external-planner main comparison.

### Controls that do not require paper reproduction

`nominal_replay`, `expected_risk_filter`, `cvar_risk_filter`, `dro_cvar_filter`, and `oracle_recovery_filter` are benchmark controls/diagnostics, not faithful external-paper reproductions. They should be labelled respectively as replay/control/surrogate/oracle audit. In particular `dro_cvar_filter` is intentionally a fast Wasserstein-inspired surrogate and should never be conflated with the now source-ported `dr_cvar_safety_filter`.

## Recommended next upload

The highest-value next upload is the **full Cao 2021 compensatory MPC paper** (and any author/source code if available), because it is the only current **main-table** method still limited by unavailable equations. After that, if `severity_minimization` is reported, its source/code is the next fidelity target. BeTop/Wayformer are mainly attribution/category issues unless you intend to promote them into the main planner comparison.
