"""Model identity: declared + measured (fingerprint suite, drift detection).

Ultimate mode captures model identity and fingerprint for execution attestation.
"""

from __future__ import annotations

from typing import Any


def collect_declared_identity(model: Any) -> dict[str, Any]:
    """Collect declared identity (provider, model_id, endpoint, params) from a model."""
    out: dict[str, Any] = {}
    if hasattr(model, "model_name"):
        out["model_id"] = getattr(model, "model_name", None)
    if hasattr(model, "provider"):
        out["provider"] = getattr(model, "provider", None)
    return out


def run_fingerprint_suite(model: Any) -> dict[str, Any]:
    """Run fingerprint suite (response hashes to stable prompts); return report (stub)."""
    return {"status": "stub", "fingerprints": []}


def detect_drift(report_a: dict[str, Any], report_b: dict[str, Any]) -> dict[str, Any]:
    """Compare two fingerprint reports for drift (stub)."""
    return {"drift_detected": False, "status": "stub"}
