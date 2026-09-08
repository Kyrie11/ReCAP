# Near-contact WOMD provenance repair

This repair is intentionally data/runtime-only. It does not change the six Near-contact baseline algorithms, the CPSF nonconformity score, the conformal quantile, candidate generation, recovery semantics, or closed-loop metrics.

## Failure mode

Canonical OC-RAP datasets may contain legacy scene identities of the form `waymax_<hash>__wx########`. Current WOMD/Waymax loading retains the official `scenario/id` as the primary raw scene id while also reproducing the legacy hash as `raw.metadata["legacy_scenario_id"]`.

The previous CPSF calibration cross-check compared calibration sample aliases against only the current raw `scenario_id`, `original_scenario_id`, and `official_scenario_id`. It omitted `legacy_scenario_id`, so the same raw scene could be rejected as a provenance mismatch after the identity migration.

`source_scenario_index` is also record-order provenance, not stable scene identity. It is now treated as a sparse-replay hint that must be verified by stable identity aliases. If the hint is stale, calibration falls back to identity matching rather than consuming the wrong future or failing immediately.

## Closed-loop source contract

The canonical `val_near_contact` / `test_near_contact` builder in this repository consumes standard WOMD validation TFExamples. The optimized Near launcher therefore defaults closed-loop replay to the same standard validation source. `validation_interactive` remains available only as an explicit `CL_WOMD=...` override for a bucket actually built from that source.

## Calibration output diagnostics

The generated conformal artifact now records:

- `source_index_verified_groups`
- `identity_fallback_groups`
- `source_index_mismatch_count`
- `source_index_mismatch_examples`
- `scene_identity_policy`

Any unresolved planning-decision group still fails closed; no calibration residual is computed from an unverified raw scene.
