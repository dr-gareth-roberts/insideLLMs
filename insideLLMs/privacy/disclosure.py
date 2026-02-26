"""Merkle inclusion proofs for selective disclosure."""

from __future__ import annotations

from typing import Any

from insideLLMs.crypto.canonical import DEFAULT_ALGO, canonical_json_bytes, digest_bytes
from insideLLMs.crypto.merkle import _hash_pair, merkle_root_from_items


def merkle_inclusion_proof(leaf_index: int, leaves: list[Any], root: str) -> list[str]:
    """Return sibling path for inclusion proof.

    Generates a proof that the leaf at `leaf_index` is part of the Merkle tree
    that hashes to `root`.

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
    list[str]
        List of sibling hashes forming the inclusion proof path.

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

    # Generate leaf hashes
    algo = DEFAULT_ALGO
    leaf_hashes = [digest_bytes(canonical_json_bytes(item), algo=algo) for item in leaves]

    proof = []
    current_level = list(leaf_hashes)
    current_index = leaf_index

    while len(current_level) > 1:
        next_level = []

        # Determine sibling for the current node
        is_right_child = current_index % 2 != 0
        sibling_index = current_index - 1 if is_right_child else current_index + 1

        # Handle odd number of nodes (last node duplicates itself)
        if sibling_index >= len(current_level):
            sibling_index = current_index

        proof.append(current_level[sibling_index])

        # Build next level
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            right = current_level[i + 1] if i + 1 < len(current_level) else current_level[i]
            next_level.append(_hash_pair(left, right, algo))

        current_level = next_level
        current_index //= 2

    return proof
