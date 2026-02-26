"""Merkle tree commitments for tamper evidence and selective disclosure.

Builds Merkle trees from ordered lists of canonicalized items and emits
root (hex), count, algo, and canon_version. Used for records.jsonl,
receipts, dataset examples, and prompt sets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from insideLLMs.crypto.canonical import (
    DEFAULT_ALGO,
    DEFAULT_CANON_VERSION,
    SUPPORTED_ALGOS,
    canonical_json_bytes,
    digest_bytes,
)


def _hash_pair(left: str, right: str, algo: str) -> str:
    """Hash two hex digests together for Merkle tree construction."""
    if algo != "sha256":
        raise ValueError(f"Unsupported algo: {algo!r}")
    payload = (left + right).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _merkle_root_from_hashes(leaf_hashes: list[str], algo: str) -> str:
    """Compute Merkle root from an ordered list of leaf hashes.

    Leaves are hashed in pairs; if odd number, duplicate the last.
    Repeated until a single root remains.
    """
    if not leaf_hashes:
        return digest_bytes(b"", algo=algo)
    current = list(leaf_hashes)
    while len(current) > 1:
        next_level: list[str] = []
        for i in range(0, len(current), 2):
            left = current[i]
            right = current[i + 1] if i + 1 < len(current) else current[i]
            next_level.append(_hash_pair(left, right, algo))
        current = next_level
    return current[0]


def merkle_root_from_items(
    items: list[Any],
    canonicalize_fn: Callable[[Any], bytes] | None = None,
    algo: str = DEFAULT_ALGO,
    canon_version: str = DEFAULT_CANON_VERSION,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Build Merkle tree from ordered list of items; return root manifest.

    Each item is canonicalized to bytes (default: canonical_json_bytes),
    then hashed to form leaves. The tree is built and the root returned
    with count, algo, and canon_version.

    Parameters
    ----------
    items : list[Any]
        Ordered list of items (e.g. record dicts, receipt dicts).
    canonicalize_fn : Callable[[Any], bytes] or None
        If None, uses canonical_json_bytes with canon_version and strict.
    algo : str, default "sha256"
        Hash algorithm.
    canon_version : str, default "canon_v1"
        Canonicalization version (used when canonicalize_fn is None).
    strict : bool, default False
        Passed to canonical_json_bytes when canonicalize_fn is None.

    Returns
    -------
    dict
        Keys: root (hex), count, algo, canon_version.
    """
    if algo not in SUPPORTED_ALGOS:
        raise ValueError(f"Unsupported algo: {algo!r}")

    if canonicalize_fn is None:

        def _canon(item: Any) -> bytes:
            return canonical_json_bytes(item, canon_version=canon_version, strict=strict)

        canonicalize_fn = _canon

    leaf_hashes = [digest_bytes(canonicalize_fn(item), algo=algo) for item in items]
    root = _merkle_root_from_hashes(leaf_hashes, algo=algo)
    return {
        "root": root,
        "count": len(items),
        "algo": algo,
        "canon_version": canon_version,
    }


def merkle_root_from_jsonl(
    path: Path | str,
    canonicalize_fn: Callable[[Any], bytes] | None = None,
    algo: str = DEFAULT_ALGO,
    canon_version: str = DEFAULT_CANON_VERSION,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Compute Merkle root from a JSONL file (one JSON object per line).

    Lines are parsed as JSON, then each object is treated as an item
    for merkle_root_from_items. Order is line order.

    Parameters
    ----------
    path : Path or str
        Path to the JSONL file.
    canonicalize_fn : Callable[[Any], bytes] or None
        Same as in merkle_root_from_items.
    algo : str
        Hash algorithm.
    canon_version : str
        Canonicalization version.
    strict : bool
        Passed to canonicalization when canonicalize_fn is None.

    Returns
    -------
    dict
        Same as merkle_root_from_items: root, count, algo, canon_version.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    items: list[Any] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return merkle_root_from_items(
        items,
        canonicalize_fn=canonicalize_fn,
        algo=algo,
        canon_version=canon_version,
        strict=strict,
    )
