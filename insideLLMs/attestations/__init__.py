"""Attestation chain: DSSE envelopes and in-toto Statements for verifiable evaluation."""

from insideLLMs.attestations.dsse import (
    PAYLOAD_TYPE_IN_TOTO,
    build_dsse_envelope,
    pae,
    parse_dsse_envelope,
)
from insideLLMs.attestations.statement import (
    PREDICATE_TYPE_EVAL_STEP,
    PREDICATE_TYPE_SLSA_PROVENANCE,
    build_statement,
)

__all__ = [
    "PAYLOAD_TYPE_IN_TOTO",
    "build_dsse_envelope",
    "pae",
    "parse_dsse_envelope",
    "PREDICATE_TYPE_EVAL_STEP",
    "PREDICATE_TYPE_SLSA_PROVENANCE",
    "build_statement",
]
