"""in-toto Statement structure for attestations.

Statement = subject (list of digest descriptors) + predicateType + predicate.
Subject describes the artifacts (e.g. records.jsonl, manifest.json);
predicate holds step-specific claims (e.g. SLSA provenance).
"""

from __future__ import annotations

from typing import Any

# Predicate type for SLSA provenance v1
PREDICATE_TYPE_SLSA_PROVENANCE = "https://slsa.dev/provenance/v1"
# Custom predicate for evaluation steps (e.g. source, env, execution)
PREDICATE_TYPE_EVAL_STEP = "https://insidellms.dev/attestations/eval-step/v1"


def build_statement(
    subject: list[dict[str, Any]],
    predicate_type: str,
    predicate: dict[str, Any],
) -> dict[str, Any]:
    """Build an in-toto Statement.

    Parameters
    ----------
    subject : list[dict]
        List of subject descriptors. Each typically has name, digest (e.g. sha256:hex).
    predicate_type : str
        URI of the predicate type (e.g. SLSA provenance, custom eval step).
    predicate : dict
        Predicate-specific payload (builder, materials, invocation, etc.).

    Returns
    -------
    dict
        Statement with _type, subject, predicateType, predicate.
    """
    return {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": subject,
        "predicateType": predicate_type,
        "predicate": predicate,
    }
