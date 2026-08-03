"""Tests for policy engine."""

import json
from pathlib import Path

import pytest

from insideLLMs.crypto import digest_obj
from insideLLMs.policy.engine import run_policy

ALL_ATTESTATIONS = [
    "00.source",
    "01.env",
    "02.dataset",
    "03.promptset",
    "04.execution",
    "05.scoring",
    "06.report",
    "07.claims",
]


def _complete_run_dir(tmp_path: Path) -> Path:
    """Build a run directory that passes policy with no transparency receipts."""
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "records.jsonl").write_text('{"id": 1}\n')
    att_dir = tmp_path / "attestations"
    att_dir.mkdir()
    for name in ALL_ATTESTATIONS:
        (att_dir / f"{name}.dsse.json").write_text("{}")
    return tmp_path


def _well_formed_receipt(run_dir: Path, att_name: str) -> dict:
    """Build a structurally well-formed receipt for an existing attestation."""
    att_path = run_dir / "attestations" / f"{att_name}.dsse.json"
    stmt_digest = digest_obj(
        json.loads(att_path.read_text(encoding="utf-8")),
        purpose="scitt_submission",
    )["digest"]
    return {
        "status": "success",
        "statement_digest": stmt_digest,
        "receipt": {"entry_id": "abc123"},
    }


def test_run_policy_passes_when_artifacts_present(tmp_path: Path) -> None:
    """Policy passes when manifest, records, and attestations 00-07 exist."""
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "records.jsonl").write_text('{"id": 1}\n')
    att_dir = tmp_path / "attestations"
    att_dir.mkdir()
    for name in [
        "00.source",
        "01.env",
        "02.dataset",
        "03.promptset",
        "04.execution",
        "05.scoring",
        "06.report",
        "07.claims",
    ]:
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
    for name in [
        "00.source",
        "01.env",
        "02.dataset",
        "03.promptset",
        "04.execution",
        "05.scoring",
        "06.report",
        "07.claims",
    ]:
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
    for name in [
        "00.source",
        "01.env",
        "02.dataset",
        "03.promptset",
        "04.execution",
        "05.scoring",
        "06.report",
    ]:
        (att_dir / f"{name}.dsse.json").write_text("{}")

    verdict = run_policy(tmp_path)
    assert verdict["passed"] is False
    assert any("07.claims" in r for r in verdict["reasons"])


def test_run_policy_fails_when_scitt_receipt_missing_for_present_attestation(
    tmp_path: Path,
) -> None:
    """A transparency dir that omits a receipt must fail, not pass silently.

    Regression: previously the ``receipt and attestation`` / ``receipt only``
    branches left the attestation-present-receipt-missing case unhandled, so no
    ``scitt_*`` check was recorded and the verdict stayed ``passed=True`` — a
    run with an incomplete transparency record read as compliant.
    """
    run_dir = _complete_run_dir(tmp_path)
    scitt_dir = run_dir / "receipts" / "scitt"
    scitt_dir.mkdir(parents=True)
    # 07 has a receipt; 04 does not, even though its attestation exists.
    (scitt_dir / "07.claims.receipt.json").write_text(
        json.dumps(_well_formed_receipt(run_dir, "07.claims"))
    )

    verdict = run_policy(run_dir)

    assert verdict["passed"] is False
    assert verdict["checks"]["scitt_04.execution"] is False
    assert any("scitt receipt 04.execution missing" in r for r in verdict["reasons"])
    # The receipt that is present and well-formed still records a passing check.
    assert verdict["checks"]["scitt_07.claims"] is True


def test_run_policy_fails_when_all_scitt_receipts_missing(tmp_path: Path) -> None:
    """An empty transparency dir fails closed for every expected receipt."""
    run_dir = _complete_run_dir(tmp_path)
    (run_dir / "receipts" / "scitt").mkdir(parents=True)

    verdict = run_policy(run_dir)

    assert verdict["passed"] is False
    for att_name in ("04.execution", "07.claims"):
        assert verdict["checks"][f"scitt_{att_name}"] is False
        assert any(f"scitt receipt {att_name} missing" in r for r in verdict["reasons"])


def test_run_policy_fails_when_scitt_receipt_has_no_attestation(tmp_path: Path) -> None:
    """A receipt without its attestation still fails (pre-existing behaviour)."""
    run_dir = _complete_run_dir(tmp_path)
    receipt = _well_formed_receipt(run_dir, "04.execution")
    (run_dir / "attestations" / "04.execution.dsse.json").unlink()
    scitt_dir = run_dir / "receipts" / "scitt"
    scitt_dir.mkdir(parents=True)
    (scitt_dir / "04.execution.receipt.json").write_text(json.dumps(receipt))
    (scitt_dir / "07.claims.receipt.json").write_text(
        json.dumps(_well_formed_receipt(run_dir, "07.claims"))
    )

    verdict = run_policy(run_dir)

    assert verdict["passed"] is False
    assert verdict["checks"]["scitt_04.execution"] is False
    assert any("has no attestation" in r for r in verdict["reasons"])


def test_run_policy_fails_when_scitt_receipt_malformed(tmp_path: Path) -> None:
    """A present but structurally invalid receipt fails."""
    run_dir = _complete_run_dir(tmp_path)
    scitt_dir = run_dir / "receipts" / "scitt"
    scitt_dir.mkdir(parents=True)
    for att_name in ("04.execution", "07.claims"):
        (scitt_dir / f"{att_name}.receipt.json").write_text(
            json.dumps({"status": "failed", "statement_digest": "", "receipt": {}})
        )

    verdict = run_policy(run_dir)

    assert verdict["passed"] is False
    assert verdict["checks"]["scitt_04.execution"] is False
    assert any("malformed" in r for r in verdict["reasons"])


def test_run_policy_passes_with_complete_well_formed_receipts(tmp_path: Path) -> None:
    """The happy path: every expected receipt present and well-formed."""
    run_dir = _complete_run_dir(tmp_path)
    scitt_dir = run_dir / "receipts" / "scitt"
    scitt_dir.mkdir(parents=True)
    for att_name in ("04.execution", "07.claims"):
        (scitt_dir / f"{att_name}.receipt.json").write_text(
            json.dumps(_well_formed_receipt(run_dir, att_name))
        )

    verdict = run_policy(run_dir)

    assert verdict["passed"] is True
    assert verdict["reasons"] == []
    assert verdict["checks"]["scitt_04.execution"] is True
    assert verdict["checks"]["scitt_07.claims"] is True
