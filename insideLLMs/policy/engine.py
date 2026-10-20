"""Policy engine: run checks and emit verdict (pass/fail + reasons)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from insideLLMs.crypto import digest_obj
from insideLLMs.policy._claims import check_claims_quality
from insideLLMs.transparency.scitt_client import receipt_looks_well_formed


def run_policy(run_dir: Path | str, *, require_claims_pass: bool = False) -> dict[str, Any]:
    """Run artifact-completeness checks on a run directory and return a verdict.

    Checks for required artifacts (manifest.json, records.jsonl), core
    attestations (00-07), integrity roots, and SCITT receipt well-formedness.
    Receipt checks are structural only — no cryptographic receipt
    verification (countersignature/inclusion proof) is performed.

    Args:
        run_dir: Path to the run directory.
        require_claims_pass: Also require complete, consistent passing results in
            claims.json and verification.json. Defaults to structural checks only.

    Returns:
        Verdict with passed, reasons and checks. When requested, structural_passed
        and claims_quality report the independent results; passed is their
        conjunction. The legacy assurance label describes structural provenance;
        claims_quality is labelled metric_acceptance, not scientific validity.
    """
    if not isinstance(require_claims_pass, bool):
        raise ValueError("require_claims_pass must be a bool")
    run_dir = Path(run_dir)
    verdict: dict[str, Any] = {
        "passed": True,
        "reasons": [],
        "checks": {},
        "assurance": "structural",
        "signature_authenticity": "not_checked",
        "scitt_authenticity": "not_checked",
    }

    # Required artifacts
    manifest_path = run_dir / "manifest.json"
    records_path = run_dir / "records.jsonl"
    attestations_dir = run_dir / "attestations"
    integrity_dir = run_dir / "integrity"

    if not manifest_path.exists():
        verdict["passed"] = False
        verdict["reasons"].append("manifest.json missing")
        verdict["checks"]["manifest"] = False
    else:
        verdict["checks"]["manifest"] = True

    if not records_path.exists():
        verdict["passed"] = False
        verdict["reasons"].append("records.jsonl missing")
        verdict["checks"]["records"] = False
    else:
        verdict["checks"]["records"] = True

    # Core attestations (00-07) should exist when policy runs post-attestation
    required_attestations = [
        "00.source",
        "01.env",
        "02.dataset",
        "03.promptset",
        "04.execution",
        "05.scoring",
        "06.report",
        "07.claims",
    ]
    for name in required_attestations:
        att_path = attestations_dir / f"{name}.dsse.json"
        if not att_path.exists():
            verdict["passed"] = False
            verdict["reasons"].append(f"attestation {name} missing")
            verdict["checks"][f"attestation_{name}"] = False
        else:
            verdict["checks"][f"attestation_{name}"] = True

    # Integrity roots (optional but expected in Ultimate mode)
    if (integrity_dir / "records.merkle.json").exists():
        verdict["checks"]["integrity_records"] = True
    else:
        verdict["checks"]["integrity_records"] = False

    # SCITT receipts: when receipts/scitt/ exists, structurally check 04 and 07
    # receipts (well-formedness only; not cryptographic verification)
    scitt_dir = run_dir / "receipts" / "scitt"
    if scitt_dir.exists():
        for att_name in ("04.execution", "07.claims"):
            receipt_path = scitt_dir / f"{att_name}.receipt.json"
            att_path = attestations_dir / f"{att_name}.dsse.json"
            # Fail closed on every asymmetry: a transparency directory that is
            # missing a receipt is an incomplete record, not a passing one. Each
            # branch records a check so no combination can pass silently.
            if not receipt_path.exists():
                verdict["passed"] = False
                verdict["reasons"].append(f"scitt receipt {att_name} missing")
                verdict["checks"][f"scitt_{att_name}"] = False
            elif not att_path.exists():
                verdict["passed"] = False
                verdict["reasons"].append(f"scitt receipt {att_name} has no attestation")
                verdict["checks"][f"scitt_{att_name}"] = False
            else:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                stmt_digest = digest_obj(
                    json.loads(att_path.read_text(encoding="utf-8")),
                    purpose="scitt_submission",
                )["digest"]
                if receipt_looks_well_formed(receipt, stmt_digest):
                    verdict["checks"][f"scitt_{att_name}"] = True
                else:
                    verdict["passed"] = False
                    verdict["reasons"].append(f"scitt receipt {att_name} malformed")
                    verdict["checks"][f"scitt_{att_name}"] = False

    if require_claims_pass:
        verdict["structural_passed"] = verdict["passed"]
        quality = check_claims_quality(run_dir)
        verdict["claims_quality"] = quality
        verdict["checks"]["claims_quality"] = quality["passed"]
        verdict["passed"] = verdict["structural_passed"] and quality["passed"]
        verdict["reasons"].extend(f"claims quality: {reason}" for reason in quality["reasons"])

    return verdict
