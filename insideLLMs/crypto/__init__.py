"""Cryptographic canonicalization and Merkle commitments for verifiable evaluation.

This package provides:
- Canonicalization contract (one true byte representation per object)
- Digest utilities with purpose and version metadata
- Merkle tree roots for records, receipts, dataset, promptset
- Content-addressed run bundle ID
"""

from insideLLMs.crypto.canonical import (
    DEFAULT_ALGO,
    DEFAULT_CANON_VERSION,
    SUPPORTED_ALGOS,
    canonical_json_bytes,
    digest_bytes,
    digest_obj,
    run_bundle_id,
)
from insideLLMs.crypto.merkle import (
    merkle_root_from_items,
    merkle_root_from_jsonl,
)

__all__ = [
    "DEFAULT_ALGO",
    "DEFAULT_CANON_VERSION",
    "SUPPORTED_ALGOS",
    "canonical_json_bytes",
    "digest_bytes",
    "digest_obj",
    "run_bundle_id",
    "merkle_root_from_items",
    "merkle_root_from_jsonl",
]
