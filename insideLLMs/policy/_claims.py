"""Fail-closed acceptance of recorded claim metrics, independent of authenticity."""

from __future__ import annotations

import json
import math
import operator
import os
from pathlib import Path
from typing import Any

from insideLLMs._artifact_snapshot import _open_directory, _read_regular

_COMPARATORS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate fields rather than silently discarding earlier evidence."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    """Reject Python's nonstandard JSON NaN and infinity extensions."""
    raise ValueError(f"Non-finite JSON constant: {value}")


def _read_object(run_dir: Path, name: str) -> dict[str, Any]:
    """Read a UTF-8 JSON object through the existing no-follow evidence boundary."""
    descriptor = _open_directory(run_dir)
    try:
        raw = _read_regular(name, descriptor)
    finally:
        os.close(descriptor)
    value = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant
    )
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _finite_number(value: Any) -> bool:
    """Accept JSON numbers without bool coercion or conversion of large integers."""
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _claim_failure(claim: dict[str, Any], result: Any) -> str | None:
    """Check declared thresholds against their recorded, explicitly passing result."""
    metric = claim.get("metric")
    if not isinstance(metric, str) or not metric.strip():
        return "declared metric must be a nonempty string"
    comparator = claim.get("operator")
    if not isinstance(comparator, str) or comparator not in _COMPARATORS:
        return "declared operator is unsupported"
    threshold = claim.get("threshold")
    if not _finite_number(threshold):
        return "declared threshold must be a finite number"
    if not isinstance(result, dict):
        return "verification result must be an object"
    if type(result.get("passed")) is not bool:
        return "verification passed must be a boolean"
    if result.get("error") not in (None, ""):
        return "verification reports an evaluation error"
    value = result.get("metric_value")
    if not _finite_number(value) or not _finite_number(result.get("threshold")):
        return "verification metric_value and threshold must be finite numbers"
    if result.get("operator") != comparator or result["threshold"] != threshold:
        return "verification operator or threshold differs from the declared claim"
    # This comparison checks recorded values only. It does not reconstruct metrics
    # from records/summary, establish provenance, or validate the scientific claim.
    if not result["passed"]:
        return "claim did not pass its recorded threshold evaluation"
    if not _COMPARATORS[comparator](value, threshold):
        return "recorded metric fails the threshold despite passed=true"
    return None


def check_claims_quality(run_dir: Path) -> dict[str, Any]:
    """Require complete, consistent passing compiler evidence for nonempty claims.

    This is metric acceptance of supplied evidence, not statistical recomputation,
    scientific validity, or cryptographic authentication. Callers must keep the
    source stable across reads; Ultimate evaluates its private publication copy.
    """
    reasons: list[str] = []
    checks: dict[str, bool] = {}
    verdict: dict[str, Any] = {
        "assurance": "metric_acceptance",
        "passed": False,
        "reasons": reasons,
        "checks": checks,
    }
    documents: dict[str, dict[str, Any]] = {}
    for name in ("claims.json", "verification.json"):
        try:
            documents[name] = _read_object(run_dir, name)
        except FileNotFoundError:
            reasons.append(f"{name} missing")
        except OSError:
            reasons.append(f"{name} unreadable (regular no-follow evidence required)")
        except (ValueError, RecursionError) as exc:
            reasons.append(f"{name} malformed: {exc}")
    if reasons:
        return verdict

    claims = documents["claims.json"].get("claims")
    if not isinstance(claims, list) or not claims:
        reasons.append("claims.json must declare a nonempty claims list")
        return verdict

    declared: dict[str, dict[str, Any]] = {}
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            reasons.append(f"claim at index {index} must be an object")
            continue
        claim_id = claim.get("id")
        if not isinstance(claim_id, str) or not claim_id.strip():
            reasons.append(f"claim at index {index} must have a nonempty string id")
        elif claim_id in declared:
            reasons.append(f"duplicate declared claim id: {claim_id}")
        else:
            declared[claim_id] = claim

    verification = documents["verification.json"]
    for claim_id in sorted(declared):
        if claim_id not in verification:
            failure = "verification result missing"
        else:
            failure = _claim_failure(declared[claim_id], verification[claim_id])
        checks[claim_id] = failure is None
        if failure is not None:
            reasons.append(f"claim {claim_id}: {failure}")
    for claim_id in sorted(verification.keys() - declared.keys()):
        reasons.append(f"verification result has no declared claim: {claim_id}")
    verdict["passed"] = bool(checks) and not reasons
    return verdict
