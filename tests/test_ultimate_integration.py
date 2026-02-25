"""Integration tests for Ultimate mode (receipts, integrity, attestations)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from insideLLMs.attestations import parse_dsse_envelope
from insideLLMs.config_types import RunConfig
from insideLLMs.models import DummyModel
from insideLLMs.probes import LogicProbe
from insideLLMs.publish.oras import PushResult
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
    assert (run_dir / "attestations" / "08.policy.dsse.json").exists()
    assert (run_dir / "attestations" / "09.publish.dsse.json").exists()
    assert (run_dir / "policy" / "verdict.json").exists()

    # Attestation 08 should have populated policy verdict data
    att_08 = json.loads((run_dir / "attestations" / "08.policy.dsse.json").read_text())
    statement, _ = parse_dsse_envelope(att_08)
    pred = statement.get("predicate", {})
    assert "passed" in pred
    assert "verdict_digest" in pred
    assert "reasons" in pred


@patch("insideLLMs.runtime._ultimate.push_run_oci")
def test_ultimate_mode_with_publish_oci_populates_attestation_09(
    mock_push: object, tmp_path: Path
) -> None:
    """With publish_oci_ref, attestation 09 gets oci_ref and oci_digest."""
    mock_push.return_value = PushResult(ref="ghcr.io/org/repo:tag", digest="sha256:abc123")

    model = DummyModel(canned_response="4")
    probe = LogicProbe()
    runner = ProbeRunner(model=model, probe=probe)
    run_dir = tmp_path / "ultimate_run"
    config = RunConfig(
        emit_run_artifacts=True,
        run_dir=str(run_dir),
        run_mode="ultimate",
        validate_output=False,
        publish_oci_ref="ghcr.io/org/repo:tag",
    )
    runner.run(
        [{"messages": [{"role": "user", "content": "What is 2+2?"}]}],
        config=config,
    )

    att_09 = json.loads((run_dir / "attestations" / "09.publish.dsse.json").read_text())
    statement, _ = parse_dsse_envelope(att_09)
    pred = statement.get("predicate", {})
    assert pred.get("oci_ref") == "ghcr.io/org/repo:tag"
    assert pred.get("oci_digest") == "sha256:abc123"


@patch("insideLLMs.runtime._ultimate.submit_statement")
def test_ultimate_mode_with_scitt_persists_receipts(
    mock_submit: object, tmp_path: Path
) -> None:
    """With scitt_service_url, submit 04 and 07, persist receipts to receipts/scitt/."""
    mock_submit.return_value = {
        "status": "success",
        "statement_digest": "sha256:abc",
        "receipt": {"proof": "inclusion"},
        "service_url": "https://scitt.example.com",
    }

    model = DummyModel(canned_response="4")
    probe = LogicProbe()
    runner = ProbeRunner(model=model, probe=probe)
    run_dir = tmp_path / "ultimate_run"
    config = RunConfig(
        emit_run_artifacts=True,
        run_dir=str(run_dir),
        run_mode="ultimate",
        validate_output=False,
        scitt_service_url="https://scitt.example.com",
    )
    runner.run(
        [{"messages": [{"role": "user", "content": "What is 2+2?"}]}],
        config=config,
    )

    assert (run_dir / "receipts" / "scitt" / "04.execution.receipt.json").exists()
    assert (run_dir / "receipts" / "scitt" / "07.claims.receipt.json").exists()
    assert mock_submit.call_count == 2
