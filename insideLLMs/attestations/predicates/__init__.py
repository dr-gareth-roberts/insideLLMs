"""Attestation predicates (SLSA provenance, privacy, custom eval steps)."""

from insideLLMs.attestations.predicates.slsa_provenance import build_slsa_provenance_predicate

__all__ = ["build_slsa_provenance_predicate"]
