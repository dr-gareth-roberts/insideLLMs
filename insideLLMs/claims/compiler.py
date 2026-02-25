"""Claims compiler: read claims.yaml, compute effects/CIs, emit claims.json + verification.json (stub)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _evaluate(operator: str, threshold: float, value: float) -> bool:
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    if operator == "==":
        return value == threshold
    raise ValueError(f"Unknown operator: {operator}")


def compile_claims(claims_yaml_path: Path | str, run_dir: Path | str) -> dict[str, Any]:
    """Read claims.yaml, compute using statistics, emit claims.json and verification.json."""
    claims_path = Path(claims_yaml_path)
    run_dir = Path(run_dir)

    with claims_path.open() as f:
        claims_doc = yaml.safe_load(f)

    claims = claims_doc.get("claims", [])

    summary_path = run_dir / "summary.json"
    with summary_path.open() as f:
        summary = json.load(f)

    metrics = summary.get("metrics", {})

    verification = {}
    for claim in claims:
        claim_id = claim["id"]
        metric_name = claim["metric"]
        operator = claim["operator"]
        threshold = claim["threshold"]

        # Assume summary.json structure: "metrics": {"accuracy": {"mean": 0.95}}
        metric_val = metrics.get(metric_name, {}).get("mean")

        if metric_val is None:
            passed = False
            error = f"Metric {metric_name} not found in summary"
        else:
            try:
                passed = _evaluate(operator, threshold, metric_val)
                error = None
            except Exception as e:
                passed = False
                error = str(e)

        verification[claim_id] = {
            "passed": passed,
            "metric_value": metric_val,
            "threshold": threshold,
            "operator": operator,
        }
        if error:
            verification[claim_id]["error"] = error

    # Emit claims.json (the parsed claims)
    claims_out = run_dir / "claims.json"
    with claims_out.open("w") as f:
        json.dump(claims_doc, f, indent=2)

    # Emit verification.json
    verification_out = run_dir / "verification.json"
    with verification_out.open("w") as f:
        json.dump(verification, f, indent=2)

    return {"status": "success", "verification": verification}
