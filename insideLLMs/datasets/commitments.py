"""Dataset commitments: Merkle root over canonicalized examples."""

from __future__ import annotations

from typing import Any

from insideLLMs.crypto.merkle import merkle_root_from_items


def dataset_merkle_root(examples: list[Any]) -> dict[str, Any]:
    """Compute Merkle root over canonicalized dataset examples."""
    return merkle_root_from_items(examples)
