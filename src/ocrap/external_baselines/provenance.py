from __future__ import annotations

"""Literature/code provenance and fidelity contracts for external baselines.

The registry is intentionally explicit: an OC-RAP candidate-lattice adaptation
must never be reported as an author's official implementation.  `core_retained`
describes the paper mechanisms implemented in this repository; `known_gaps`
lists components that remain outside the fair executable-candidate protocol.
"""

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class BaselineProvenance:
    canonical_name: str
    aliases: tuple[str, ...]
    regimes: tuple[str, ...]
    paper_title: str
    paper_year: int | None
    paper_url: str | None
    official_code_url: str | None
    implementation_kind: str
    fidelity: str
    core_retained: tuple[str, ...]
    known_gaps: tuple[str, ...]
    reporting_name: str

    def to_dict(self) -> dict:
        return asdict(self)


REGISTRY: tuple[BaselineProvenance, ...] = (
    BaselineProvenance(
        "nominal_replay", ("nominal", "nominal_replay", "log_replay"), ("safe",),
        "Logged trajectory replay (dataset control; not a paper implementation)", None, None, None,
        "dataset control", "exact",
        ("replays the logged nominal candidate",), (), "Logged nominal replay",
    ),
    BaselineProvenance(
        "wayformer_bc", ("route_bc", "route_bc_lite", "waymax_bc", "waymax_bc_lite", "wayformer_bc", "wayformer_style_bc", "route_bc_wayformer"), ("safe",),
        "Wayformer: Motion Forecasting via Simple & Efficient Attention Networks", 2022,
        "https://arxiv.org/abs/2207.05844", None,
        "paper-architecture early-fusion/GMM ego-BC projection", "architecture-faithful / task-interface adapted",
        ("homogeneous early fusion of observable agent-history and vector-map tokens", "trainable latent-query scene bottleneck", "repeated learned-query self/cross-attention decoder", "bivariate-Gaussian multimodal ego-future distribution", "source/paper nearest-mode GMM NLL plus mode-classification objective", "GMM-likelihood projection onto the common executable candidate lattice"),
        ("Wayformer is a motion-forecasting method rather than a planner, so v58 explicitly treats it as an ego-BC architecture control", "the original 8 s forecasting horizon is projected to the OC-RAP executable-prefix horizon", "the paper WOMD benchmark uses 192 encoder latent queries and eight decoder layers; v58 explicitly uses 96 latent queries and two repeated self/cross decoder layers as a runtime budget while preserving the operator", "map points are pooled per polyline before homogeneous fusion to avoid paying attention over padded map-point tensors", "the v58 implementation was cross-checked against the paper and the Wayformer reproduction bundled in the uploaded BeTop repository; no author-official Waymo Wayformer training code/checkpoint was identified", "not checkpoint-compatible with any external Wayformer implementation"),
        "Wayformer early-fusion GMM ego-BC adapter",
    ),
    BaselineProvenance(
        "gameformer_lite", ("gameformer", "gameformer_lite", "gameformer_levelk"), ("safe", "near"),
        "GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving", 2023,
        "https://arxiv.org/abs/2303.05760", "https://github.com/MCZhi/GameFormer",
        "source-derived WOMD-to-common-lattice port", "source-ported / interface-adapted",
        ("8-state two-layer LSTM actor encoder", "256-d six-layer Transformer scene fusion", "six-mode GMM decoder", "joint multi-agent level-k interaction refinement", "detached previous-level future encoder", "deep level-wise trajectory loss", "trajectory-to-executable-candidate projection"),
        ("OC-RAP observable preprocessing replaces the author's released tensors", "the official public repository provides WOMD interaction prediction/open-loop planning rather than a drop-in Waymax/WOMD closed-loop planner", "global ego-centric WOMD map tokens replace actor-indexed GameFormer lane tensors", "the common interface supplies a 20-step executable prefix rather than the native 80-step target", "OC-RAP does not expose native all-neighbor future targets to this adapter so source-native GMM supervision is ego-only", "not checkpoint-compatible with the official repository"),
        "GameFormer (source-ported)",
    ),
    BaselineProvenance(
        "plantf", ("plantf", "plan_tf", "plantf_adapter"), ("safe",),
        "Rethinking Imitation-based Planner for Autonomous Driving", 2024,
        "https://arxiv.org/abs/2309.10443", "https://github.com/jchengai/planTF",
        "source-derived nuPlan-to-WOMD port", "source-ported / interface-adapted",
        ("128-d four-layer scene Transformer", "nine-channel actor-history encoding", "six-channel state attention with 0.75 state dropout", "source-shaped six-mode trajectory decoder", "best-ADE SmoothL1 plus mode classification", "source AdamW decay split and warmup-cosine schedule", "trajectory-to-executable-candidate projection"),
        ("nuPlan feature builder/map object classes are reconstructed from observable WOMD tensors", "the WOMD-derived contract exposes 11 history steps rather than the native 21-step PlanTF history", "source_max_agents=16 is an explicit speed cap below the public feature-builder maximum of 32; retained neighbors are chosen by observed current distance as in the source builder", "NATTEN NeighborhoodAttention1D is replaced by a dependency-free local-convolution FPN with the same three-level topology", "the common interface truncates the native 80-step horizon to 20 executable steps", "native auxiliary future-agent prediction loss is unavailable because OC-RAP external samples do not carry that target", "not checkpoint-compatible with official weights"),
        "PlanTF (source-ported)",
    ),
    BaselineProvenance(
        "pluto", ("pluto", "pluto_adapter"), ("safe",),
        "PLUTO: Pushing the Limit of Imitation Learning-based Planning for Autonomous Driving", 2024,
        "https://arxiv.org/abs/2404.14327", "https://github.com/jchengai/pluto",
        "source-derived nuPlan-to-WOMD port", "source-ported / interface-adapted",
        ("128-d four-layer source-style scene encoder", "reference-line x 12-mode four-layer planning decoder", "reference-to-reference/mode-to-mode/scene cross attention", "global reference-mode probability selection", "source-shaped trajectory regression plus flattened classification", "source AdamW decay split and warmup-cosine schedule"),
        ("nuPlan reference-line/map builders are replaced by observable WOMD vectors and the common executable candidate prefixes", "the WOMD-derived contract exposes 11 history steps rather than the native 21-step PLUTO history", "source_max_agents=16 is an explicit speed cap below the public feature-builder maximum of 48; retained neighbors are chosen by observed current distance", "NATTEN temporal attention is replaced by a dependency-free local-convolution FPN", "native 80-step horizon is truncated to 20 executable steps", "the public trainer defaults contrastive loss off and this port therefore does not fabricate triplets", "native ESDF collision loss and auxiliary agent-future loss are not reproduced because their source targets/geometry are unavailable", "not checkpoint-compatible with official weights"),
        "PLUTO (source-ported)",
    ),
    BaselineProvenance(
        "pdm_closed", ("pdm_closed", "pdm_closed_adapter"), ("safe",),
        "Parting with Misconceptions about Learning-based Vehicle Motion Planning", 2023,
        "https://arxiv.org/abs/2306.07962", "https://github.com/autonomousvision/tuplan_garage",
        "source-scorer projection onto common executable lattice", "source-structure / geometry-adapted",
        ("PDMScorer multiplicative safety gate", "gated progress normalization", "source 5/5/2 progress-TTC-comfort weighted aggregation", "route-progress/direction/drivable observable proxies", "non-learning closed-loop selection"),
        ("BatchIDMPolicy plus lateral-offset proposal generation and kinematic simulation are replaced by the shared OC-RAP executable proposals", "nuPlan polygon at-fault/drivable/direction and exact TTC geometry are approximated from WOMD route/predicted occupancy", "PDM emergency-brake override is not independently reconstructed"),
        "PDM-Closed (source-scorer port)",
    ),
    BaselineProvenance(
        "pdm_hybrid", ("pdm_hybrid", "pdm_hybrid_adapter"), ("safe",),
        "Parting with Misconceptions about Learning-based Vehicle Motion Planning", 2023,
        "https://arxiv.org/abs/2306.07962", "https://github.com/autonomousvision/tuplan_garage",
        "source short-horizon semantics projected onto common executable lattice", "source-faithful within executed prefix / long-horizon tail unavailable",
        ("PDM-Closed proposal selection semantics", "source correction-horizon contract: uncorrected PDM-Closed trajectory through 2.0 s", "non-learning current action under the OC-RAP short-prefix replan protocol"),
        ("the learned PDM-Offset tail after the 2.0 s correction horizon is not observable in the current 2 s executed-prefix comparison", "therefore PDM-Hybrid is intentionally expected to match PDM-Closed on current-action metrics", "a faithful hybrid advantage requires an 8 s trajectory target/evaluation path rather than a learned logit that changes the pre-2 s action"),
        "PDM-Hybrid (source prefix semantics)",
    ),
    BaselineProvenance(
        "idm", ("idm", "idm_planner"), ("safe",),
        "Congested Traffic States in Empirical Observations and Microscopic Simulations (Intelligent Driver Model)", 2000,
        "https://doi.org/10.1103/PhysRevE.62.1805", None,
        "finite-lattice IDM control projection", "equation-core adaptation",
        ("IDM desired acceleration", "front-gap and relative-speed interaction", "comfortable headway/deceleration parameters"),
        ("continuous IDM acceleration is projected onto the nearest feasible executable candidate", "lateral path choice comes from the common candidate set"),
        "IDM projection",
    ),
    BaselineProvenance(
        "betopnet_lite", ("betop", "betop_lite", "betopnet", "betopnet_lite"), ("safe",),
        "Reasoning Multi-Agent Behavioral Topology for Interactive Autonomous Driving", 2024,
        "https://arxiv.org/abs/2409.18031", "https://github.com/OpenDriveLab/BeTop",
        "uploaded-source-backed topology-aware candidate-lattice adaptation", "source-backed core-mechanism / unreleased-planner-interface adapted",
        ("official behavior-braid x-time segment-intersection label operator", "official nearest-map-polyline braid target", "source-structured separate source/target TopoFuser with iterative residual topology features", "detached top-k topology indexing before local attention", "source focal hard-topology mining and valid-edge normalization", "six iterative topology-reasoning layers", "Appendix-C planning inference structure with t_b=3 and lambda_m=0.5 short-term repulsive-potential cost projected onto executable candidates"),
        ("the official repository releases the WOMD prediction pipeline but explicitly leaves the nuPlan planning pipeline TODO, so this is not claimed as an official BeTop planner reproduction", "existing OC-RAP NPZs intentionally omit other-agent ground-truth future trajectories; actor braid labels therefore apply the exact source operator to an observation-only constant-velocity target extrapolation", "the source Eq. (6) learned joint-prediction recombination and M-by-MJ branched planning head cannot be checkpoint-faithfully reconstructed from the released planning code; v58 preserves the published inference-time short-term potential term but uses the common candidate lattice instead of claiming the unreleased native head", "because actor topology labels are a CV proxy, the shipped topology loss weight is 10 rather than the source prediction setting weight_top=100; rebuilt datasets with source-equivalent future topology labels may restore 100", "the executable-candidate selector is not checkpoint-compatible with official BeTopNet"),
        "BeTop source-backed topology adapter",
    ),
    BaselineProvenance(
        "marc_lite", ("marc", "marc_lite", "marc_contingency"), ("near",),
        "MARC: Multipolicy and Risk-aware Contingency Planning for Autonomous Driving", 2023,
        "https://arxiv.org/abs/2308.12021", None,
        "paper-core finite-lattice contingency projection", "paper-core / optimizer-interface adapted",
        ("semantic multi-policy families", "policy-conditioned scenario responses", "scene-level dynamic branch point from scenario-conditioned ego futures", "shared-plus-contingent trajectory tree", "CVaR safety-cost reweighting with user risk tolerance", "policy-family selection"),
        ("the paper's closed-loop critical-scenario renderer is represented by mode-conditioned executable responses from the shared observation-only predictor", "the LP/iLQR bi-level continuous optimizer is replaced by exact enumeration over the fixed executable candidate lattice", "no author code/checkpoint was identified", "the shared predictor is an OC-RAP interface component and is not attributed to MARC"),
        "MARC (paper-core candidate-lattice port)",
    ),
    BaselineProvenance(
        "racp_lite", ("racp", "racp_lite", "risk_aware_contingency"), ("near",),
        "RACP: Risk-Aware Contingency Planning with Multi-Modal Predictions", 2024,
        "https://arxiv.org/abs/2402.17387", "https://github.com/KhMustafa/Risk-aware-contingency-planning-with-multi-modal-predictions",
        "uploaded-source-structured finite-lattice port", "source-structured / optimizer-interface adapted",
        ("shared-plan plus branch-weighted contingent-plan cost topology from the released planner", "multi-modal branch weights/beliefs", "non-anticipative shared prefix", "probability-weighted collision-risk cost", "mode-conditioned executable contingent continuations"),
        ("CommonRoad/Frenet trajectory generation, CasADi/OSQP branch-MPC dynamics, and backup-controller propagation are replaced by enumeration over OC-RAP executable candidates", "the released multi-stage branch-MPC is projected to a shared-prefix plus mode-conditioned-tail candidate tree because the common executable lattice does not expose the source optimizer's native branch variables", "the source Frenet 2.0 s shared + 1.0 s contingent timing is preserved only as a normalized 2:1 split on the shorter common horizon, not as source-native absolute timing", "the released belief updater/prediction format cannot be consumed directly from WOMD, so all baselines use the same observation-conditioned interface predictor", "not checkpoint/API compatible with the source repository", "the user bibliography cites the IEEE T-IV article as 2024; some bibliographic services may display the final volume year separately"),
        "RACP (source-structured candidate-lattice port)",
    ),
    BaselineProvenance(
        "robust_scenario_mpc", ("robust_scenario_mpc", "scenario_mpc", "batkovic_scenario_mpc"), ("near",),
        "A Robust Scenario MPC Approach for Uncertain Multi-Modal Obstacles", 2021,
        "https://doi.org/10.1109/LCSYS.2020.3006819", None,
        "paper-core finite-lattice scenario-policy projection", "paper-core / optimizer-interface adapted",
        ("multi-modal obstacle scenarios and probabilities", "pairwise mode distinguishability", "pairwise non-anticipative input tying until each mode pair becomes distinguishable", "mode-dependent recourse after distinction", "hard all-mode state/collision constraint satisfaction", "probability-weighted expected-cost objective"),
        ("reachable-set/tube tightening and the nonlinear continuous MPC solve are replaced by explicit scenario checks and a bounded beam search over executable candidate-tree tuples", "point-valued shared predictor hypotheses approximate the paper's mode reachable sets", "internal risk loss is used in the objective only unless an experiment explicitly enables a legacy loss guard", "recursive-feasibility theorem assumptions are therefore not claimed for the adapter"),
        "Robust scenario MPC (paper-core candidate-lattice port)",
    ),
    BaselineProvenance(
        "dr_cvar_safety_filter", ("dr_cvar_safety_filter", "distributionally_robust_cvar_filter", "safaoui_dr_cvar_filter"), ("near",),
        "Distributionally Robust CVaR-Based Safety Filtering for Motion Planning in Uncertain Environments", 2024,
        "https://arxiv.org/abs/2309.08821", "https://github.com/TSummersLab/dr-cvar-safety_filtering",
        "official-source-structured DR-CVaR safe-halfspace/MPC port onto common executable lattice", "source-structured paper-core / interface-adapted",
        ("released DRCVaRHalfspace affine-loss formulation", "per-obstacle/per-horizon DR-CVaR safe-halfspaces", "Wasserstein ambiguity radius and alpha-tail CVaR", "released MPCFilter Q/QT/R reference-tracking objective", "hard halfspace admission before minimum-cost executable projection"),
        ("the released continuous DTVehicle/CVXPY MPC is projected onto OC-RAP executable candidates", "the affine DRCVaRHalfspace solve uses its algebraically equivalent closed form instead of thousands of tiny CVXPY solves", "the source synthetic Gaussian obstacle samples are replaced by samples from the shared observation-only multimodal WOMD predictor", "not checkpoint/solver-output identical to the authors' synthetic drone experiments"),
        "DR-CVaR safety filter (source-structured port)",
    ),
    BaselineProvenance(
        "conformal_predictive_safety_filter", ("conformal_predictive_safety_filter", "conformal_safety_filter", "cpsf"), ("near",),
        "Conformal Predictive Safety Filter for RL Controllers in Dynamic Environments", 2023,
        "https://arxiv.org/abs/2306.02551", None,
        "paper-core CPSF Algorithm-1/Eq.-7 port onto common executable lattice", "paper-core mathematical port / predictor-and-solver adapted",
        ("held-out raw-future conformal calibration", "joint-agent L2 nonconformity per prediction horizon", "delta/T Bonferroni allocation", "explicit (N+1)-th infinity sentinel and exact finite-sample quantile", "per-horizon conformal tubes", "Eq.-7 hard geometric separation constraints", "minimum-deviation safety projection"),
        ("the paper permits an arbitrary trajectory predictor; OC-RAP uses the shared observation-only multimodal predictor instead of retraining the paper's 2x128 LSTM", "the paper's 3x128 learned approximation of the safety-filter optimization is replaced by exact enumeration of Eq. (7) over the common executable candidate lattice", "formal coverage additionally depends on the exchangeability/independence of the chosen OC-RAP calibration unit; the artifact records that unit and finite-sample validity diagnostics"),
        "Conformal predictive safety filter (paper-core port)",
    ),
    BaselineProvenance(
        "expected_risk_filter", ("expected_risk", "expected_risk_filter", "expected_risk_planner"), ("near",),
        "Expected-risk constrained planning (generic control baseline)", None, None, None,
        "observation-only risk filter", "exact protocol definition",
        ("multimodal collision loss expectation", "risk-threshold admission", "utility-risk selection"), (), "Expected-risk filter",
    ),
    BaselineProvenance(
        "cvar_risk_filter", ("cvar_risk", "cvar_risk_filter", "cvar_planner"), ("near",),
        "CVaR-constrained planning (generic risk-sensitive baseline)", None, None, None,
        "observation-only risk filter", "exact protocol definition",
        ("weighted upper-tail CVaR", "risk-threshold admission", "utility-risk selection"), (), "CVaR risk filter",
    ),
    BaselineProvenance(
        "dro_cvar_filter", ("dro_cvar", "dro_cvar_filter", "dro_cvar_safety_filter", "dr_cvar_filter"), ("near",),
        "Distributionally robust CVaR planning (generic baseline)", None, None, None,
        "observation-only robust-risk surrogate", "explicit surrogate",
        ("CVaR tail risk", "ambiguity-radius dispersion penalty", "constrained selection"),
        ("dispersion penalty is a fast Wasserstein-inspired surrogate rather than a full inner ambiguity optimization",),
        "DRO-CVaR filter",
    ),
    BaselineProvenance(
        "predictive_safety_filter", ("predictive_safety_filter", "psf", "cbf_backup_filter", "predictive_cbf_backup", "backup_cbf_filter"), ("near",),
        "A Predictive Safety Filter for Learning-Based Control of Constrained Nonlinear Dynamical Systems", 2021,
        "https://arxiv.org/abs/1812.05506", None,
        "paper-core finite-lattice predictive safety filter", "paper-core / optimizer-interface adapted",
        ("accept proposed input unchanged when the finite-horizon safety problem is feasible", "state and input constraints across the prediction horizon", "terminal safe/backup-set condition", "minimum-input-deviation correction when intervention is necessary"),
        ("candidate enumeration replaces the online nonlinear safety-filter MPC", "the terminal safe set is represented by a conservative stopping-distance backup certificate because WOMD does not provide the paper's verified terminal controller/model uncertainty set", "legacy CBF-named aliases are accepted for command compatibility only; no CBF condition is attributed to the cited paper"),
        "Predictive safety filter (paper-core candidate-lattice port)",
    ),
    BaselineProvenance(
        "oracle_recovery_filter", ("oracle_filter", "oracle_recovery_filter", "branchwise_oracle_filter", "oracle_branchwise_recovery"), ("near",),
        "OC-RAP teacher-only oracle upper bound (not an external baseline)", None, None, None,
        "non-deployable audit upper bound", "exact teacher audit",
        ("branch-wise existential recovery option", "oracle order: option maximization before latent-root aggregation"),
        ("uses teacher tensors and must never be reported as deployable",), "Teacher oracle upper bound",
    ),
    BaselineProvenance(
        "postimpact_mpc_lite", ("postimpact_mpc", "postimpact_mpc_lite", "post_impact_mpc_lite", "postimpact_mpc_paper", "integrated_postimpact_mpc"), ("contact",),
        "Integrated Post-Impact Planning and Active Safety Control for Autonomous Vehicles", 2023,
        "https://doi.org/10.1109/TIV.2023.3236150", None, "paper-core planning-integrated MPC/PSO port onto common executable lattice", "paper-core mathematical port / dynamics-interface adapted",
        ("safe-braking-distance brake-vs-avoid decision", "planning-integrated MPC Qy/Ru tracking objective", "front/rear axle octagon road-adhesion inequalities", "paper Eq.-15 constant-velocity/current-lane rhombus obstacle exclusion", "LTR <= 0.9 rollover-stability constraint", "paper Table-I simplified Magic-Formula tire model with Fz[kN]/slip[deg] and friction-similarity scaling", "500-particle/8-iteration PSO wheel-force/front-steer allocation objective and tire saturation"),
        ("OC-RAP/WOMD does not expose the paper's suspension roll states, so LTRsim is evaluated with the paper-compatible quasi-static small-angle load-transfer proxy", "continuous linear MPC decision variables are projected onto the common executable candidate trajectories", "PSO wheel-level allocation is used as an actuator-feasibility certificate while Waymax executes the common acceleration/steering interface", "the paper's constant-velocity obstacle model is retained rather than replaced by OC-RAP's learned predictor", "no CarSim/HIL plant or post-impact damage/impulse estimator is available in WOMD"),
        "Integrated post-impact MPC + PSO (paper-core port)",
    ),
    BaselineProvenance(
        "post_crash_braking", ("post_crash_braking", "post_crash_braking_rule", "stable_stop", "stable_stop_rule", "postcrash_stable_stop"), ("contact",),
        "A System for Autonomous Braking of a Vehicle Following Collision", 2017,
        "https://doi.org/10.4271/2017-01-1581", None,
        "paper-core autonomous PIB branch projected onto executable lattice", "paper-core / patent-supported interface adaptation",
        ("collision-triggered autonomous post-impact braking", "no-driver PIB branch", "braking up to ABS capability", "kinetic-energy reduction and stop/stabilization termination"),
        ("WOMD has no brake-pedal/driver-intent signal so PIBA cannot be distinguished and the benchmark explicitly evaluates the autonomous no-driver PIB branch", "hydraulic pressure, wheel-slip and ABS actuator dynamics are unavailable; ABS-level braking is projected onto executable longitudinal-acceleration candidates", "the 8 km/h undesirable-motion threshold and 2.5 s termination default are supported by the related Ford inventor patent family and are not attributed as SAE-paper numerical parameters"),
        "Post-impact braking (Lu et al. PIB port)",
    ),
    BaselineProvenance(
        "post_collision_restoration", ("post_collision_restoration", "trajectory_restoration", "post_collision_trajectory_restoration", "post_collision_restoration_heuristic", "ackermann_restoration"), ("contact",),
        "Post-Collision Trajectory Restoration for a Single-track Ackermann Vehicle using Heuristic Steering and Tractive Force Functions", 2026,
        "https://arxiv.org/abs/2602.08444", None, "paper-core open-loop steering/tractive-force law projected onto executable lattice", "paper-core mathematical port / source-ambiguity documented",
        ("Eq. (10) steering-direction composition", "Eq. (13) first sine steering pulse", "Eq. (16) second sine steering pulse interface", "Eq. (18) initial-plus-compensatory tractive-force law", "Eq. (21) compensatory sine force pulse", "absolute paper-time pulse schedule and reported A1/Ac/Kdir parameters"),
        ("the 2026 source is an arXiv preprint and no official implementation was identified", "Table 2 reports an undefined K1 while Eq. (16) requires A2 and does not report A2; v57 leaves A2=0 by default rather than inventing a mapping", "WOMD has no wheel tractive-force state, so Fi is represented by the common nominal longitudinal acceleration and Fc/m is the explicit incremental lattice projection", "the paper controller is open-loop; OC-RAP does not add learned future-agent risk to its selection objective", "the paper reports separate tuned cases but no online case classifier; v57 selects the reported case from observed initial post-impact yaw-rate magnitude using an explicitly documented benchmark adapter threshold"),
        "Post-collision restoration (Ghosh et al. paper-core port)",
    ),
    BaselineProvenance(
        "postimpact_motion_tvlqr", ("postimpact_motion_tvlqr", "postimpact_motion_planning", "wang2022_postimpact", "postimpact_tvlqr"), ("contact",),
        "Post-Impact Motion Planning and Tracking Control for Autonomous Vehicles", 2022,
        "https://doi.org/10.1186/s10033-022-00745-w", None,
        "paper-core quintic/APF/TVLQR/allocation port onto common executable lattice", "paper-core mathematical port / optimizer-interface adapted",
        ("three quintic X/Y/psi reference functions with initial and terminal equalities", "paper artificial-potential-field objective using fixed perceived obstacle coordinates plus road-boundary potential and sideslip-stability term", "resultant-acceleration and rear-axle lateral-force constraints", "time-varying local linearization and Riccati TVLQR with paper Q/R", "paper Table-I full Magic-Formula tire model with Fz[kN]/slip[deg] and friction-similarity scaling", "nonlinear wheel-force/front-steer allocation objective with combined-tire constraints"),
        ("the paper's fmincon trajectory optimizer is replaced by equality-constrained quintic projection of every common executable candidate", "the paper's fixed scenario terminal lane target is supplied by each common candidate endpoint before enforcing zero terminal lateral/yaw rates", "the wheel-level SQP allocator is replaced by deterministic vectorized projected search of the same residual/combined-tire objective for runtime tractability", "the paper's APF uses fixed obstacle coordinates, so this port intentionally does not inject OC-RAP's learned motion predictor", "WOMD/Waymax does not expose wheel forces, collision-damage states, or the original high-fidelity vehicle plant"),
        "Post-impact quintic + TVLQR (paper-core port)",
    ),
    BaselineProvenance(
        "compensatory_postimpact_mpc", ("compensatory_postimpact_mpc", "cao_postimpact_mpc"), ("contact",),
        "Compensatory Model Predictive Control for Post-impact Trajectory Tracking via Active Front Steering and Differential Torque Vectoring", 2021,
        "https://doi.org/10.1177/0954407020979087", None,
        "source-limited FCC-MPC structured adaptation", "objective-level / full-equation source unavailable",
        ("feedforward-feedback compensation structure", "reverse-steering response", "differential-yaw-action intent", "constraint transformation for initially deteriorated states", "time-varying-style constraints on input/input rate/slip proxy", "lateral/yaw deviation attenuation"),
        ("the reliable online primary source exposes the abstract/metadata but not the full controller equations, weights or constraint-transform formula; v57 therefore does not claim equation-exact reproduction", "WOMD has no wheel torque, wheel speed or measured slip-ratio states; active-front-steering/differential-torque-vectoring are projected through executable acceleration/steering candidates and an explicitly labelled longitudinal-utilisation proxy", "the pre-impact path frame is estimated only from observed ego history because WOMD does not expose the paper vehicle reference-state interface", "a full paper or official source upload is required for a second-pass equation-level port"),
        "Compensatory post-impact FCC-MPC (source-limited adapter)",
    ),
    BaselineProvenance(
        "robust_postimpact_control", ("robust_postimpact_control", "postimpact_sliding_mode", "ao_postimpact_control"), ("contact",),
        "Advanced Post-impact Safety and Stability Control for Electric Vehicles", 2022,
        "https://doi.org/10.1049/itr2.12230", None,
        "paper-core sliding-mode + exact small-QP allocation port", "paper-core mathematical port / plant-interface adapted",
        ("Eqs. (17-19) course/yaw-rate and lateral-deviation sliding surface", "Eqs. (21-23) reaching law and requested additional yaw moment", "Eqs. (26-27) in-wheel-motor fault-gain model", "Eqs. (28-34) tire-utilisation objective, force/yaw mapping and constrained quadratic allocation", "source Table-1 vehicle parameters and Table-2 controller gains"),
        ("OC-RAP begins from observed post-impact states and does not expose the impact point/force history, so explicit impact-force terms in U(t) are zero after the observed impact interval", "WOMD has no IWM fault diagnosis; healthy fault factors are the default but user-supplied factors use the same published allocation", "Waymax executes common acceleration/steering rather than four in-wheel torques; the source QP is therefore an actuator-realizability certificate/ranking term", "the pre-impact original-road frame is estimated only from observed ego history so the published lateral-deviation/course-angle states remain meaningful after impact", "the published motor-gain fault model is normalized to unit healthy gain because WOMD does not expose motor-electrical calibration", "the accessible source specifies bounded motor torques but a universal numeric wheel-torque maximum is not recoverable for the benchmark, so that bound remains an explicit adapter parameter"),
        "Advanced post-impact SMC + fault-tolerant QP (paper-core port)",
    ),
    BaselineProvenance(
        "severity_minimization", ("severity_minimization", "severity_minimization_planner", "unavoidable_collision_planner", "crash_mitigation_planner", "uc_severity_planner"), ("near",),
        "Motion planning for autonomous vehicles with the inclusion of post-impact motions for minimising collision risk", 2023,
        "https://doi.org/10.1080/00423114.2022.2088396", None, "paper-core collision/post-impact severity candidate port", "paper-core mathematical adapter / collision-geometry and temporal-interface adapted",
        ("Kudlich-Slibar full-versus-sliding impact classification and restitution/friction impulse mechanics", "source Table-A1 target/ego mass, inertia, geometry and tire parameters", "3DOF post-impact target-vehicle rollout with source simplified Magic-Formula lateral tire forces", "Eq. (25) target-vehicle collision cost using lateral deviation, modulo-pi heading, yaw-rate and sideslip components with equal source demonstration weights", "selection of the lowest predicted target-vehicle post-impact collision cost over the executable trajectory library"),
        ("the paper is a *pre-impact unavoidable-collision* motion planner; v58 therefore moves the recommended OC-RAP evaluation from the historical Contact legacy bucket to Near-contact legacy/control, while keeping the old Contact launcher flag only as a deprecated compatibility path", "the paper assumes the target future trajectory is known, whereas serialized deployable OC-RAP samples omit other-agent futures; v58 uses the nearest observed target with constant-velocity extrapolation and no teacher future", "v58 reproduces the paper's Sutherland-Hodgman overlap-polygon centroid and overlap-intersection-vertex contact-plane construction after a vectorized SAT broad phase; degenerate corner/tangent contacts fall back to the minimum-penetration SAT axis", "for source full impacts the paper solves an implicit 3DOF collision system with fsolve including collision-interval tire terms; v58 uses a planar rigid-body sticking/sliding impulse projection at the common action interface before the source-style 3DOF post-impact rollout and does not claim equation-exact full-impact equivalence", "the native offline dynamic-single-track optimal-control trajectory library is replaced only by the common executable candidate library"),
        "Unavoidable-collision severity minimization (paper-core legacy adapter)",
    ),
)


MAIN_TABLE_BY_REGIME: dict[str, tuple[str, ...]] = {
    "safe": ("gameformer_lite", "plantf", "pluto", "pdm_closed", "pdm_hybrid", "idm"),
    "near": ("marc_lite", "racp_lite", "robust_scenario_mpc", "predictive_safety_filter", "dr_cvar_safety_filter", "conformal_predictive_safety_filter"),
    "contact": ("postimpact_mpc_lite", "post_crash_braking", "postimpact_motion_tvlqr", "post_collision_restoration", "compensatory_postimpact_mpc", "robust_postimpact_control"),
}

LEGACY_OR_DIAGNOSTIC_BY_REGIME: dict[str, tuple[str, ...]] = {
    "safe": ("nominal_replay", "wayformer_bc", "betopnet_lite"),
    "near": ("gameformer_lite", "expected_risk_filter", "cvar_risk_filter", "dro_cvar_filter", "oracle_recovery_filter", "severity_minimization"),
    "contact": (),
}


def find_provenance(name: str) -> BaselineProvenance | None:
    key = str(name).strip().lower()
    for item in REGISTRY:
        if key == item.canonical_name or key in item.aliases:
            return item
    return None


def registry_dict(regimes: Iterable[str] | None = None) -> list[dict]:
    selected = set(str(x).lower() for x in regimes) if regimes is not None else None
    return [x.to_dict() for x in REGISTRY if selected is None or selected.intersection(x.regimes)]
