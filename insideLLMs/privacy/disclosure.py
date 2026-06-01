"""Merkle inclusion proofs for selective disclosure.

Security notes
--------------
- Proofs are **direction-aware** (each step records whether the sibling is the
  left or right node), so :func:`verify_inclusion_proof` can reconstruct the root
  without out-of-band knowledge of the tree shape.
- Leaf and internal-node hashes currently share the same hash construction (no
  leaf/node domain-separation prefix). This matches the artifact-spine
  ``records_merkle_root`` so existing roots stay reproducible. Adding leaf/node
  domain separation (to fully close second-preimage ambiguity) changes every
  root and therefore requires a canon-version bump; it is tracked as a separate,
  versioned hardening rather than silently breaking existing artifacts.
"""

from __future__ import annotations

from typing import Any

from insideLLMs.crypto.canonical import (
    DEFAULT_ALGO,
    DEFAULT_CANON_VERSION,
    canonical_json_bytes,
)
from insideLLMs.crypto.merkle import _hash_pair, _leaf_digest, merkle_root_from_items


def merkle_inclusion_proof(leaf_index: int, leaves: list[Any], root: str) -> list[dict[str, Any]]:
    """Return a direction-aware sibling path proving inclusion of a leaf.

    Parameters
    ----------
    leaf_index : int
        Index of the leaf to prove.
    leaves : list[Any]
        The full list of items in the tree.
    root : str
        The expected Merkle root.

    Returns
    -------
    list[dict[str, Any]]
        Ordered proof steps from leaf to root. Each step is
        ``{"sibling": <hash>, "sibling_is_left": <bool>}`` where
        ``sibling_is_left`` indicates the sibling is the *left* operand when
        combined with the running node (i.e. the proven node is a right child at
        that level). Pass this directly to :func:`verify_inclusion_proof`.

    Raises
    ------
    IndexError
        If leaf_index is out of bounds.
    ValueError
        If the computed root from the leaves does not match the provided root.
    """
    if not leaves:
        raise ValueError("Cannot generate proof for empty leaves list")

    if leaf_index < 0 or leaf_index >= len(leaves):
        raise IndexError(f"Leaf index {leaf_index} out of bounds for {len(leaves)} leaves")

    # First, verify the root matches
    computed_root_info = merkle_root_from_items(leaves)
    if computed_root_info["root"] != root:
        raise ValueError(f"Root mismatch. Expected {root}, got {computed_root_info['root']}")

    algo = DEFAULT_ALGO
    canon_version = DEFAULT_CANON_VERSION
    leaf_hashes = [
        _leaf_digest(canonical_json_bytes(item), algo, canon_version=canon_version)
        for item in leaves
    ]

    proof: list[dict[str, Any]] = []
    current_level = list(leaf_hashes)
    current_index = leaf_index

    while len(current_level) > 1:
        is_right_child = current_index % 2 != 0
        sibling_index = current_index - 1 if is_right_child else current_index + 1

        # Odd number of nodes: the last node is duplicated and paired with itself.
        if sibling_index >= len(current_level):
            sibling_index = current_index

        # The sibling is the left operand exactly when the proven node is a right child.
        proof.append({"sibling": current_level[sibling_index], "sibling_is_left": is_right_child})

        next_level = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            right = current_level[i + 1] if i + 1 < len(current_level) else current_level[i]
            next_level.append(_hash_pair(left, right, algo, canon_version))

        current_level = next_level
        current_index //= 2

    return proof


def verify_inclusion_proof(
    leaf: Any,
    proof: list[dict[str, Any]],
    root: str,
    *,
    algo: str = DEFAULT_ALGO,
    canon_version: str = DEFAULT_CANON_VERSION,
) -> bool:
    """Verify a direction-aware inclusion proof against a Merkle root.

    Parameters
    ----------
    leaf : Any
        The original leaf item (canonicalized and hashed the same way as during
        tree construction).
    proof : list[dict[str, Any]]
        The proof returned by :func:`merkle_inclusion_proof`.
    root : str
        The expected Merkle root.
    algo : str
        Hash algorithm (defaults to the canonical algorithm).
    canon_version : str
        Tree-construction version (must match the version used to build the root).

    Returns
    -------
    bool
        True if the proof reconstructs ``root`` from ``leaf``, else False.
    """
    node = _leaf_digest(canonical_json_bytes(leaf), algo, canon_version=canon_version)
    for step in proof:
        sibling = step["sibling"]
        if step["sibling_is_left"]:
            node = _hash_pair(sibling, node, algo, canon_version)
        else:
            node = _hash_pair(node, sibling, algo, canon_version)
    return node == root
