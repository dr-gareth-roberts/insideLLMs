"""OpenVEX emission from scan/context."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid


def _load_manifest(run_dir: Path | str) -> dict[str, Any]:
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def emit_openvex(run_dir: Path | str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate OpenVEX document."""
    manifest = _load_manifest(run_dir)
    context = context or {}
    
    statements = []
    
    # We create a default VEX statement asserting "not_affected" for the evaluation run
    # If the user provides specific vulnerabilities in context, we would map them here.
    
    components = []
    if model := manifest.get("model"):
        components.append(
            {"@id": f"pkg:ml/{model.get('provider', 'unknown')}/{model.get('model_id', 'unknown')}"}
        )
        
    if probe := manifest.get("probe"):
        components.append(
            {"@id": f"pkg:insidellms/probe/{probe.get('probe_id', 'unknown')}"}
        )
        
    if dataset := manifest.get("dataset"):
        components.append(
            {"@id": f"pkg:data/{dataset.get('dataset_id', 'unknown')}"}
        )
        
    timestamp = datetime.now(timezone.utc).isoformat()
    
    if components:
        statements.append({
            "vulnerability": {"name": "any"}, # Catch-all for basic attestation
            "products": components,
            "status": "not_affected",
            "justification": "inline_mitigations_already_exist",
            "impact_statement": "The evaluation framework runs the components in an isolated environment."
        })
        
    return {
        "@context": "https://openvex.dev/ns/v0.2.0",
        "@id": f"https://insidellms.dev/openvex/{uuid.uuid4().hex}",
        "author": "InsideLLMs Framework",
        "timestamp": timestamp,
        "version": 1,
        "statements": statements
    }
