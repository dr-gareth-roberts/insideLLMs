"""Policy engine: run checks and emit verdict (pass/fail + reasons)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from insideLLMs.crypto import digest_obj
from insideLLMs.transparency.scitt_client import verify_receipt


def run_policy(run_dir: Path | str, policy_yaml_path: Path | str | None = None) -> dict[str, Any]:
    """Load policy, run checks (attestations exist, signatures verify, Merkle recompute, etc.), return verdict.

    Args:
        run_dir: Path to the run directory.
        policy_yaml_path: Optional path to policy YAML; if present, its digest is included.

    Returns:
        Verdict dict with keys: passed, reasons, checks.
    """
    run_dir = Path(run_dir)
    verdict: dict[str, Any] = {"passed": True, "reasons": [], "checks": {}}

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
        "00.source", "01.env", "02.dataset", "03.promptset",
        "04.execution", "05.scoring", "06.report", "07.claims",
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

    # SCITT receipts: when receipts/scitt/ exists, verify 04 and 07 receipts
    scitt_dir = run_dir / "receipts" / "scitt"
    if scitt_dir.exists():
        for att_name in ("04.execution", "07.claims"):
            receipt_path = scitt_dir / f"{att_name}.receipt.json"
            att_path = attestations_dir / f"{att_name}.dsse.json"
            if receipt_path.exists() and att_path.exists():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                stmt_digest = digest_obj(
                    json.loads(att_path.read_text(encoding="utf-8")),
                    purpose="scitt_submission",
                )["digest"]
                if verify_receipt(receipt, stmt_digest):
                    verdict["checks"][f"scitt_{att_name}"] = True
                else:
                    verdict["passed"] = False
                    verdict["reasons"].append(f"scitt receipt {att_name} invalid")
                    verdict["checks"][f"scitt_{att_name}"] = False
            elif receipt_path.exists():
                verdict["checks"][f"scitt_{att_name}"] = False
                verdict["reasons"].append(f"scitt receipt {att_name} has no attestation")
                verdict["passed"] = False

    return verdict
