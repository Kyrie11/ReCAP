from __future__ import annotations

import re
from pathlib import Path

from ocrap.external_baselines.provenance import MAIN_TABLE_BY_REGIME

ROOT = Path(__file__).resolve().parents[1]


def _block(text: str, var: str) -> str:
    m = re.search(rf"{re.escape(var)}=\(\n(?P<body>.*?)\n\)", text, flags=re.S)
    assert m, f"missing bash array {var}"
    return m.group("body")


def _method_lines(block: str) -> tuple[str, ...]:
    out: list[str] = []
    for raw in block.splitlines():
        line = raw.strip().strip('"').strip("'")
        if not line or line.startswith("#"):
            continue
        # Safe SPECS entries use method|config|kind|checkpoint.
        out.append(line.split("|", 1)[0])
    return tuple(out)


def test_regime_launchers_match_main_table_registry() -> None:
    safe = (ROOT / "scripts/run_safe_regime_external_baselines.sh").read_text()
    near = (ROOT / "scripts/run_near_contact_external_baselines_2gpu_optimized.sh").read_text()
    contact = (ROOT / "scripts/run_contact_external_baselines.sh").read_text()

    assert _method_lines(_block(safe, "SPECS")) == MAIN_TABLE_BY_REGIME["safe"]
    assert _method_lines(_block(near, "METHODS")) == MAIN_TABLE_BY_REGIME["near"]
    assert _method_lines(_block(contact, "METHODS")) == MAIN_TABLE_BY_REGIME["contact"]


def test_launchers_are_two_gpu_bounded_and_wire_train_calibration() -> None:
    for rel in (
        "scripts/run_safe_regime_external_baselines.sh",
        "scripts/run_near_contact_external_baselines_2gpu_optimized.sh",
        "scripts/run_contact_external_baselines.sh",
    ):
        text = (ROOT / rel).read_text()
        assert ': "${CUDA_DEVICES:=0,1}"' in text
        assert ': "${MAX_PARALLEL:=2}"' in text
        assert '((MAX_PARALLEL <= 2)) || MAX_PARALLEL=2' in text

    near = (ROOT / "scripts/run_near_contact_external_baselines_2gpu_optimized.sh").read_text()
    assert 'tools/calibrate_external_baselines.py' in near
    assert 'DO_CALIBRATE' in near
    assert 'CONFORMAL_INTERVALS' in near
    assert 'conformal_prediction_intervals_m' in near
    assert 'validation/validation_tfexample.tfrecord@150' in near

    contact = (ROOT / "scripts/run_contact_external_baselines.sh").read_text()
    assert 'tools/register_external_nonlearning_baselines.py' in contact
    assert 'DO_TRAIN' in contact

    master = (ROOT / "scripts/run_all_regime_external_baselines_optimized.sh").read_text()
    assert 'DO_TRAIN="$DO_TRAIN_CONTACT"' in master
    assert 'DO_CALIBRATE="$DO_CALIBRATE_NEAR"' in master
