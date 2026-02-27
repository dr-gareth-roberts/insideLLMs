"""Canonicalization and digest utilities for verifiable evaluation evidence.

This module provides a versioned canonicalization contract and digest functions
so every digest in the system carries algo, canon_version, purpose, and
created_by (insideLLMs version). Used for records, receipts, attestations,
and run bundle IDs.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from insideLLMs._serialization import stable_json_dumps

try:
    import insideLLMs as _pkg

    _LIBRARY_VERSION: Optional[str] = getattr(_pkg, "__version__", None)
except ImportError:
    _LIBRARY_VERSION = None

DEFAULT_CANON_VERSION = "canon_v1"
DEFAULT_ALGO = "sha256"
SUPPORTED_ALGOS = ("sha256",)


def canonical_json_bytes(obj: Any, canon_version: str = DEFAULT_CANON_VERSION, *, strict: bool = False) -> bytes:
    """Return the canonical byte representation of an object for hashing.

    Uses insideLLMs stable JSON serialization so equivalent structures produce
    identical bytes. Every digest in the system should be produced from
    canonical bytes to ensure reproducibility.

    Parameters
    ----------
    obj : Any
        JSON-serializable object to canonicalize.
    canon_version : str, default "canon_v1"
        Version tag for the canonicalization scheme.
    strict : bool, default False
        If True, raise on non-JSON-serializable values.

    Returns
    -------
    bytes
        UTF-8 encoded canonical JSON (sort_keys, no whitespace).

    Raises
    ------
    StrictSerializationError
        When strict=True and obj contains non-serializable values.
    """
    if canon_version != DEFAULT_CANON_VERSION:
        raise ValueError(f"Unsupported canon_version: {canon_version!r}")
    return stable_json_dumps(obj, strict=strict).encode("utf-8")


def digest_bytes(data: bytes, algo: str = DEFAULT_ALGO) -> str:
    """Compute the hex digest of raw bytes.

    Parameters
    ----------
    data : bytes
        Raw bytes to hash.
    algo : str, default "sha256"
        Hash algorithm. Only sha256 is guaranteed supported.

    Returns
    -------
    str
        Lowercase hexadecimal digest string.

    Raises
    ------
    ValueError
        If algo is not supported.
    """
    if algo not in SUPPORTED_ALGOS:
        raise ValueError(f"Unsupported digest algo: {algo!r}. Supported: {SUPPORTED_ALGOS}")
    if algo == "sha256":
        return hashlib.sha256(data).hexdigest()
    raise ValueError(f"Unsupported digest algo: {algo!r}")


def digest_obj(
    obj: Any,
    algo: str = DEFAULT_ALGO,
    canon_version: str = DEFAULT_CANON_VERSION,
    purpose: str = "record",
    created_by: Optional[str] = None,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Canonicalize an object and return a digest descriptor.

    Every digest in the system carries algo, canon_version, purpose,
    and created_by (insideLLMs version) for auditability.

    Parameters
    ----------
    obj : Any
        Object to canonicalize and hash.
    algo : str, default "sha256"
        Hash algorithm.
    canon_version : str, default "canon_v1"
        Canonicalization version.
    purpose : str, default "record"
        Purpose tag (e.g. record, receipt, dataset_example, attestation).
    created_by : Optional[str], default None
        Set to insideLLMs version when None.

    Returns
    -------
    dict
        With keys: digest (hex string), algo, canon_version, purpose, created_by.
    """
    raw = canonical_json_bytes(obj, canon_version=canon_version, strict=strict)
    digest = digest_bytes(raw, algo=algo)
    return {
        "digest": digest,
        "algo": algo,
        "canon_version": canon_version,
        "purpose": purpose,
        "created_by": created_by if created_by is not None else _LIBRARY_VERSION,
    }


def run_bundle_id(
    manifest_digest: str,
    roots: dict[str, str],
    attestation_digests: list[str],
    algo: str = DEFAULT_ALGO,
) -> str:
    """Compute content-addressed run bundle ID from manifest, Merkle roots, and attestation digests.

    run_bundle_id = hash(manifest_digest + sorted root names/values + sorted attestation digests).
    This is what you publish, sign, and diff.

    Parameters
    ----------
    manifest_digest : str
        Hex digest of the canonical manifest.
    roots : dict[str, str]
        Named Merkle roots (e.g. records_merkle_root, receipts_merkle_root).
    attestation_digests : list[str]
        Digests of each attestation envelope (e.g. 00.source, ..., 09.publish).

    Returns
    -------
    str
        Hex digest (bundle ID).
    """
    if algo != DEFAULT_ALGO:
        raise ValueError(f"Unsupported algo for run_bundle_id: {algo!r}")
    parts: list[bytes] = [manifest_digest.encode("utf-8")]
    for key in sorted(roots):
        parts.append(key.encode("utf-8"))
        parts.append(roots[key].encode("utf-8"))
    for d in sorted(attestation_digests):
        parts.append(d.encode("utf-8"))
    payload = b"".join(parts)
    return hashlib.sha256(payload).hexdigest()
