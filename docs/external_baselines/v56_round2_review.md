# OC-RAP v56 - External baseline source/paper-core reproduction pass

This revision audits and replaces the v55 implementations of four external baselines:

1. `dr_cvar_safety_filter` - Safaoui/Summers DR-CVaR safety filtering (official source available)
2. `conformal_predictive_safety_filter` - Strawn/Ayanian/Lindemann CPSF (paper only)
3. `postimpact_mpc_lite` - Wang et al. 2023 Integrated Post-Impact Planning and Active Safety Control (paper only)
4. `postimpact_motion_tvlqr` - Wang et al. 2022 Post-Impact Motion Planning + TVLQR (paper only)

The original Safe/Near/Contact launcher interface is preserved.

## Near-contact

### DR-CVaR safety filter

- Replaced the v55 generic `CVaR + ambiguity penalty` selector with the structure in the authors' released repository.
- Reproduces `DRCVaRHalfspace`: one safe halfspace per obstacle and prediction step.
- For the released affine loss with unconstrained support, uses the exact algebraic solution
  `g* = empirical_CVaR_alpha(r - h^T xi) + epsilon/alpha - delta`.
  This is equivalent to the source CVXPY subproblem and avoids thousands of tiny online solves.
- Preserves source defaults: horizon 10, sample count 20, alpha 0.2, Wasserstein radius 0.05, loss bound 0.1, Q=2, QT=5, R=1.
- Reproduces the released `MPCFilter` objective topology, including its interior-state index range.
- Native continuous DTVehicle/CVXPY MPC is projected onto the common executable candidate lattice and is explicitly disclosed as an interface adaptation.

### Conformal Predictive Safety Filter

- Replaced the v55 scalar `hard_violation`/collision-risk calibration with Algorithm 1 from the paper.
- Calibration computes per-horizon joint-agent L2 nonconformity from held-out raw WOMD futures.
- Implements `delta_bar = delta / T`, `p = ceil((N+1)(1-delta_bar))`, and the explicit `(N+1)`-th `+infinity` sentinel exactly.
- Runtime admission implements Eq. (7): `||tau_bar^j_{t+h} - xhat_{t+h}|| >= C_h + epsilon` with `epsilon=0.5 m` by default, and minimizes deviation from the nominal candidate.
- No extra OC-RAP vehicle radii are added to Eq. (7).
- `calibration_near_contact` is matched to the WOMD **standard validation** source used to construct that dataset. Closed-loop may continue using `validation_interactive`.
- Fixed a critical raw-track mapping issue: OC-RAP serializes agents in `[SDC, other raw tracks...]` order, whereas raw WOMD retains original track indices. Calibration now reconstructs the builder's exact order before reading future states and retains a fail-closed current-position alignment check.
- Calibration artifacts record the chosen exchangeability unit. `group` preserves the existing launcher behavior; `scene_max` is available as a stricter clustered/scene-level adaptation.

## Contact

### Integrated Post-Impact Planning + Active Safety Control (Wang 2023)

Restored the paper mechanisms that were only heuristics in v55:

- energy-based safe braking distance and brake-vs-lane-avoid decision;
- paper `Qy`/`Ru` planning-integrated MPC objective projected over the executable lattice;
- front/rear axle octagonal road-adhesion inequalities;
- Eq. (15) constant-velocity/current-lane obstacle model and rhombus exclusion (`Sl=6 m`, `Sw=2 m`);
- `LTR <= 0.9` rollover gate (quasi-static proxy because WOMD has no roll state/suspension channels);
- Table-I simplified Magic Formula and Eq. (39) friction-similarity law;
- PSO wheel-force/front-steer allocation with the paper's `N=500`, `TR=8`, `c1=c2=3`, `w: 0.9 -> 0.4`, and objective weights `[9,1,10,1]`;
- Table-III vehicle values available in the paper (m=1610 kg, Iz=2059 kg m^2, Lf=1.05 m, Lr=1.61 m, track=1.565 m, wheel radius=0.347 m, wheel inertia=0.9 kg m^2).

Critical physics correction: the paper's fitted Magic-Formula coefficients are evaluated with vertical load in **kN** and slip angle in **degrees**. v56 converts WOMD/OC-RAP N/rad values before evaluating the tire model, then applies the paper friction-similarity law. Combined-slip limits continue using physical force in N.

The paper obstacle model is retained directly; this method no longer receives the benchmark learned/multimodal predictor.

### Post-Impact Motion Planning + TVLQR (Wang 2022)

Restored:

- three quintic `X(t), Y(t), psi(t)` reference functions;
- initial equalities and terminal `Ydot(tf)=0`, `psidot(tf)=0` constraints;
- APF obstacle and road-boundary objective with `Dr=1.7 m`, `Ds=1.0 m`;
- sideslip/stability objective and paper weights `k1=k2=k3=1`, `k4=0.9`;
- resultant-acceleration and rear-axle lateral-force constraints;
- time-varying local linearization and Riccati TVLQR with paper `Q` and `R`;
- full Table-I Magic Formula plus friction-similarity and combined-tire ellipse;
- nonlinear wheel-force/front-steer allocation with paper residual weights `[9,1,10]`.

The APF now follows the paper literally and uses fixed perceived obstacle coordinates `(X_b,Y_b)`; it no longer injects OC-RAP's learned future predictor. The original paper's `fmincon` trajectory optimization and SQP wheel allocator are projected to the common executable lattice / deterministic vectorized projected search and are disclosed as such.

The same kN/degree unit correction is applied to the Wang-2022 Magic Formula.

## Runtime optimization

No paper search budget was reduced for acceleration.

- The four paper ports dispatch before legacy scalar observed-risk profiles.
- Near DR-CVaR/CPSF build only the shared prediction context; full expected/CVaR/worst-risk profiles for all candidates are skipped.
- Contact Wang-2022/2023 ports are predictor-free, matching the papers and removing another unnecessary forecast pass.
- Wang-2023 PSO is batched over candidate/control-knot dimensions while preserving 500 particles x 8 iterations.
- Wang-2022 6x6 Riccati recursions use batched 3x3 solves across candidates.
- Wang-2022 nonlinear tire allocation is batched across candidate/control-knot dimensions.
- Non-learning training is still one shared train/val data-contract scan, not six redundant scans.

Synthetic 24-candidate CPU microbenchmark (`docs/external_baselines/v56_microbenchmark.json`):

- observation context only: ~0.62 ms vs full risk profiles+context: ~4.85 ms (~7.86x avoided setup cost);
- DR-CVaR selector: ~4.42 ms;
- CPSF selector: ~3.43 ms;
- Wang-2023 PSO scalar kernel: ~193.9 ms -> batched ~109.5 ms (~1.77x);
- Wang-2022 Riccati scalar: ~63.0 ms -> batched ~6.64 ms (~9.48x);
- Wang-2022 allocation scalar: ~67.3 ms -> batched ~16.1 ms (~4.19x);
- end-to-end Wang-2023 candidate selection: ~108 ms;
- end-to-end Wang-2022 candidate selection: ~39.8 ms.

These are microbenchmarks, not WOMD/Waymax end-to-end wall-clock claims.

## Publication metric contract

The launchers call `tools/summarize_external_closed_loop.py` so publication summaries retain the complete regime contract:

- Near: collision/offroad, clearance/TTC tails, near-contact/critical-TTC exposure episode metrics, near-zero clearance, recovery timing/gain/AUC, FRA_exec/FRA_cand, DRS, ODG, bounded NUP, comfort and intervention.
- Contact: overlap episodes/duration, post-contact free-space AUC/gain, escape/time, re-contact, secondary overlap, stable-stop/quality/time, post-contact overlap/clearance deficit, and comfort.

## Verification in this environment

- 59 relevant external-baseline/Waymax/closed-loop tests passed (55 main regression tests + 4 additional Waymax/input-contract tests).
- Near, Contact, and all-regime shell launchers pass `bash -n`.
- New mechanism regression tests cover DR-CVaR Wasserstein conservatism, CPSF exact finite-sample quantile and Eq. (7), CPSF raw WOMD track-order restoration, Wang-2023 constant-velocity obstacle prediction and tire units, Wang-2022 static APF semantics/quintic terminal equalities/tire units, batched Riccati equivalence, and deterministic batched allocators.

The current execution environment does not mount the user's `/data0/...` OC-RAP/WOMD datasets, so real full-dataset Waymax closed-loop runs were not claimed here. Run the unchanged regime launchers on the target server for final end-to-end numbers.
