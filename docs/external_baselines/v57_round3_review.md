# OC-RAP v57 - Contact source/paper-core baseline audit

This revision audits and replaces the v56 implementations of four Contact-regime external baselines while preserving the existing launcher/registry aliases:

- `post_crash_braking`
- `post_collision_restoration`
- `compensatory_postimpact_mpc`
- `robust_postimpact_control`

## Source mapping and fidelity

### Post-crash braking

Mapped to Lu et al., **A System for Autonomous Braking of a Vehicle Following Collision**, SAE Technical Paper 2017-01-1581 (2017). The paper establishes PIBA (driver-braking assist) and PIB (autonomous braking up to ABS when the driver is not braking). Because WOMD has no brake-pedal/driver-intent channel, OC-RAP explicitly evaluates the no-driver PIB branch. The related Ford inventor patent family is used only to support adapter defaults such as the 8 km/h undesirable-motion example and 2.5 s termination example; these numbers are not attributed to the SAE paper.

Implementation: collision-triggered ABS-level longitudinal deceleration projected onto executable candidates, stop/stabilize termination, no learned predictor or teacher future.

### Post-collision restoration

Mapped to Ghosh, Orlando, Chakrabarty, **Post-Collision Trajectory Restoration for a Single-track Ackermann Vehicle using Heuristic Steering and Tractive Force Functions**, arXiv:2602.08444 (2026).

Implementation: absolute-time Eq.-10/13/16 steering pulses plus Eq.-18/21 tractive-force composition projected onto the common acceleration/steering interface. The source Table 2 reports an undefined `K1`, whereas Eq. (16) requires `A2` and the paper does not report `A2`; v57 therefore keeps `A2=0` by default and records the ambiguity instead of inventing a mapping. The paper reports separate tuned cases but no online classifier, so OC-RAP's yaw-rate-based case selector is explicitly marked as a benchmark adapter.

### Compensatory post-impact MPC

Mapped to Cao et al., **Compensatory model predictive control for post-impact trajectory tracking via active front steering and differential torque vectoring**, Proc. IMechE Part D 235(4), 903-919 (2021), DOI 10.1177/0954407020979087.

Reliable online primary material exposes the abstract and verifies FCC-MPC, reverse steering, differential torque vectoring, constraint transformation for deteriorated states, and time-varying saturation on input/input rate/slip ratio. The full equations, weights and exact transform are not reliably available in the online material used for this pass. v57 therefore deliberately labels this implementation `source_limited_abstract_structured`, implements only those verified mechanisms as a finite-lattice adapter, and does **not** claim equation-exact reproduction. A full-paper/source upload is required for a second equation-level pass.

### Robust post-impact control

Mapped to Ao et al., **Advanced post-impact safety and stability control for electric vehicles**, IET Intelligent Transport Systems (2022), DOI 10.1049/itr2.12230. The Wiley article exposes the full controller equations.

Implementation reproduces Eqs. (17-23) sliding-mode upper controller, Eqs. (26-27) IWM fault model, and Eqs. (28-34) tire-utilisation/fault-tolerant quadratic allocation with source Tables 1-2 parameters. The four-wheel box QP is solved exactly for the benchmark's small dimension by vectorized active-set enumeration; active-set factorizations are cached across replans. WOMD lacks IWM diagnosis/torque actuation, so healthy normalized motor gains are the default and the QP is used as an actuator-realizability certificate/ranking term before Waymax executes the common acceleration/steering command.

## Data and state-interface corrections

- The four ports dispatch before the legacy learned multimodal risk-profile path. This avoids injecting predictor capability absent from these cited methods and removes unnecessary runtime work.
- Post-impact lateral deviation/course-angle controllers use a pre-impact path frame estimated from **observed ego history only**. Using the current disturbed post-impact pose as the frame origin would artificially set the initial lateral/course error near zero. No future/teacher labels are used.
- Closed-loop runner injects only runtime elapsed-contact time (`step_idx * contact_dt`) so absolute-time PIB/restoration laws progress correctly over repeated replanning. This value is not a dataset label.
- Existing Contact dataset filters and launcher aliases remain unchanged.

## Publication metrics

All four methods continue through the same Contact-regime Waymax evaluator and publication summarizer. Therefore they emit the complete common Contact contract rather than method-specific surrogate metrics: overlap episodes/duration, longest overlap run, post-contact terminal clearance, free-space AUC and normalized AUC, clearance gain/time-to-peak, escape rate/time, re-contact rate/count, secondary-overlap rate, stable-stop rate/quality/time, post-contact overlap duration/rate, clearance-deficit AUC, and comfort/kinematic diagnostics.

## Runtime optimization

No paper mechanism/search budget is reduced for speed.

- Predictor/risk-profile early bypass for all four ports.
- Vectorized candidate scoring.
- Ao 4-wheel QP solved in batch over candidates with cached active-set factorizations.
- Non-neural methods retain the existing launcher-compatible `DO_TRAIN=true` behavior as shared data-contract validation rather than gradient training.

Synthetic 24-candidate CPU selector medians (`docs/external_baselines/v57_microbenchmark.json`):

- post-crash braking: 7.91 ms (v56) -> 0.68 ms (v57), ~11.6x
- post-collision restoration: 11.77 ms -> 0.65 ms, ~18.0x
- compensatory post-impact MPC: 7.46 ms -> 0.73 ms, ~10.3x
- robust post-impact control: 7.50 ms -> 3.67 ms, ~2.0x while restoring the paper SMC + exact small QP structure

These are selector microbenchmarks, not full WOMD/Waymax wall-clock numbers.

## Verification

- 58/58 relevant external-baseline, Waymax hot-path, data-filter, metric-summary and launcher-contract tests pass.
- Contact, Near-contact and all-regime shell launchers pass `bash -n`.
- New tests cover: PIB activation/termination; absolute restoration timeline and source ambiguity; observed-history pre-impact reference frame; Ao source sliding surface/gains; exact small box-QP correctness/bounds; predictor/teacher independence; and explicit Cao source-limited fidelity diagnostics.

The execution environment does not mount the target `/data0/...` datasets, so v57 does not claim a full real-data Waymax run. The unchanged server launchers should be used for final end-to-end measurements.
