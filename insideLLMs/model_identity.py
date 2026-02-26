"""Model identity: declared + measured (fingerprint suite, drift detection).

Ultimate mode captures model identity and fingerprint for execution attestation.
"""

from __future__ import annotations

import hashlib
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
    """Run fingerprint suite (response hashes to stable prompts); return report."""
    identity = collect_declared_identity(model)

    # Stable prompts designed to elicit deterministic responses
    stable_prompts = ["What is 2+2?", "Count to 3: 1, 2, "]

    fingerprints = []

    for prompt in stable_prompts:
        try:
            # Force temperature to 0 for determinism if the model supports it
            if hasattr(model, "generate"):
                response = model.generate(prompt, temperature=0.0)
            else:
                response = "unsupported_model_interface"

            # Hash the response
            resp_bytes = str(response).encode("utf-8")
            resp_hash = hashlib.sha256(resp_bytes).hexdigest()

            fingerprints.append({"prompt": prompt, "response_hash": resp_hash})
        except Exception as e:
            fingerprints.append({"prompt": prompt, "error": str(e)})

    return {
        "status": "success",
        "model_id": identity.get("model_id"),
        "provider": identity.get("provider"),
        "fingerprints": fingerprints,
    }


def detect_drift(report_a: dict[str, Any], report_b: dict[str, Any]) -> dict[str, Any]:
    """Compare two fingerprint reports for drift."""
    fps_a = {
        fp["prompt"]: fp.get("response_hash")
        for fp in report_a.get("fingerprints", [])
        if "response_hash" in fp
    }
    fps_b = {
        fp["prompt"]: fp.get("response_hash")
        for fp in report_b.get("fingerprints", [])
        if "response_hash" in fp
    }

    differences = []

    for prompt, hash_a in fps_a.items():
        if prompt in fps_b:
            hash_b = fps_b[prompt]
            if hash_a != hash_b:
                differences.append({"prompt": prompt, "hash_a": hash_a, "hash_b": hash_b})

    return {"status": "success", "drift_detected": len(differences) > 0, "differences": differences}
