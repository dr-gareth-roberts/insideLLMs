"""Tests for policy engine."""

from pathlib import Path

import pytest

from insideLLMs.policy.engine import run_policy


def test_run_policy_passes_when_artifacts_present(tmp_path: Path) -> None:
    """Policy passes when manifest, records, and attestations 00-07 exist."""
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "records.jsonl").write_text('{"id": 1}\n')
    att_dir = tmp_path / "attestations"
    att_dir.mkdir()
    for name in ["00.source", "01.env", "02.dataset", "03.promptset", "04.execution", "05.scoring", "06.report", "07.claims"]:
        (att_dir / f"{name}.dsse.json").write_text("{}")

    verdict = run_policy(tmp_path)
    assert verdict["passed"] is True
    assert verdict["reasons"] == []
    assert verdict["checks"]["manifest"] is True
    assert verdict["checks"]["records"] is True


def test_run_policy_fails_when_manifest_missing(tmp_path: Path) -> None:
    """Policy fails and reports reason when manifest.json is missing."""
    (tmp_path / "records.jsonl").write_text("{}")
    att_dir = tmp_path / "attestations"
    att_dir.mkdir()
    for name in ["00.source", "01.env", "02.dataset", "03.promptset", "04.execution", "05.scoring", "06.report", "07.claims"]:
        (att_dir / f"{name}.dsse.json").write_text("{}")

    verdict = run_policy(tmp_path)
    assert verdict["passed"] is False
    assert "manifest.json missing" in verdict["reasons"]
    assert verdict["checks"]["manifest"] is False


def test_run_policy_fails_when_attestation_missing(tmp_path: Path) -> None:
    """Policy fails when a required attestation is missing."""
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "records.jsonl").write_text("{}")
    att_dir = tmp_path / "attestations"
    att_dir.mkdir()
    # Only write 00-06, missing 07.claims
    for name in ["00.source", "01.env", "02.dataset", "03.promptset", "04.execution", "05.scoring", "06.report"]:
        (att_dir / f"{name}.dsse.json").write_text("{}")

    verdict = run_policy(tmp_path)
    assert verdict["passed"] is False
    assert any("07.claims" in r for r in verdict["reasons"])
