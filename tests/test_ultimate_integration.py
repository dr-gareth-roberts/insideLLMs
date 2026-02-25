"""Integration tests for Ultimate mode (receipts, integrity, attestations)."""

from pathlib import Path

import pytest

from insideLLMs.config_types import RunConfig
from insideLLMs.models import DummyModel
from insideLLMs.probes import LogicProbe
from insideLLMs.runtime.runner import ProbeRunner


def test_ultimate_mode_emits_integrity_and_attestations(tmp_path: Path) -> None:
    """With run_mode=ultimate, run dir gets integrity/, attestations/, receipts/."""
    model = DummyModel(canned_response="4")
    probe = LogicProbe()
    runner = ProbeRunner(model=model, probe=probe)
    run_dir = tmp_path / "ultimate_run"
    config = RunConfig(
        emit_run_artifacts=True,
        run_dir=str(run_dir),
        run_mode="ultimate",
        validate_output=False,
    )
    runner.run(
        [{"messages": [{"role": "user", "content": "What is 2+2?"}]}],
        config=config,
    )
    assert (run_dir / "records.jsonl").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "receipts" / "calls.jsonl").exists()
    assert (run_dir / "integrity" / "records.merkle.json").exists()
    assert (run_dir / "integrity" / "receipts.merkle.json").exists()
    assert (run_dir / "integrity" / "bundle_id.txt").exists()
    assert (run_dir / "attestations" / "00.source.dsse.json").exists()
    assert (run_dir / "attestations" / "04.execution.dsse.json").exists()
    assert (run_dir / "attestations" / "09.publish.dsse.json").exists()
