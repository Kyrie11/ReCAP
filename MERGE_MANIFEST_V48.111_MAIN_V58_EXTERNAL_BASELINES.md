# Merge manifest: v48.111 main algorithm + v58 external baselines

This tree was built with the v48.111 submission-mode archive as the authoritative main-algorithm base and the v58 source-ported archive as the authoritative external-baseline implementation.

## Merge rule

- Preserved the v48.111 main-algorithm tree byte-for-byte outside the explicit external-baseline allowlist.
- Replaced `src/ocrap/external_baselines/` and `configs/external_baselines/` wholesale from v58.
- Replaced all external-baseline launchers and their direct support tools from v58.
- Replaced the v58 external-baseline regression tests and fidelity documentation.
- Preserved v48.111 root README/main-algorithm design documents.
- `src/ocrap/simulation/closed_loop_runner.py` received only the v58 external-baseline runtime changes; after the surgical merge this file is byte-identical to the v58 version, while all other non-allowlisted v48.111 files remain unchanged.

## Validation

- Non-allowlisted v48.111 files changed: 0.
- Dedicated v58 baseline files copied with mismatches: 0.
- Python compileall for external-baseline/runtime files: PASS.
- External-baseline/Waymax/launcher regression tests: 77 passed.
- v48.105-v48.111 main-algorithm regression tests: 44 passed (8 existing PyTorch warnings).
- External-baseline shell launchers: `bash -n` PASS.

See `MERGE_EXTERNAL_BASELINE_PATHS.txt` for the explicit replacement allowlist.
