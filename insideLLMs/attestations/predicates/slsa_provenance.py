"""SLSA provenance predicate for evaluation steps.

Describes how artifacts were produced: builder, invocation, materials.
Used for execution, scoring, report, and publish attestations.
"""

from __future__ import annotations

from typing import Any


def build_slsa_provenance_predicate(
    builder: dict[str, Any],
    invocation: dict[str, Any] | None = None,
    materials: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal SLSA provenance v1 predicate.

    Parameters
    ----------
    builder : dict
        Builder identity, e.g. {"id": "https://insidellms.dev/runner/1.0"}.
    invocation : dict or None
        Config source, environment (e.g. config_source, parameters).
    materials : list[dict] or None
        Input artifacts (digests, URIs).
    metadata : dict or None
        buildInvocationId, buildStartedOn, buildFinishedOn, etc.

    Returns
    -------
    dict
        Predicate suitable for predicateType SLSA provenance v1.
    """
    out: dict[str, Any] = {
        "buildType": "https://insidellms.dev/eval/v1",
        "builder": builder,
    }
    if invocation is not None:
        out["invocation"] = invocation
    if materials is not None:
        out["materials"] = materials
    if metadata is not None:
        out["metadata"] = metadata
    return out
