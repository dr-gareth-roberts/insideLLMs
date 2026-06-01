import pytest

from insideLLMs.crypto.merkle import merkle_root_from_items
from insideLLMs.privacy.disclosure import merkle_inclusion_proof, verify_inclusion_proof


def test_merkle_inclusion_proof_round_trips_through_verifier():
    leaves = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
    root = merkle_root_from_items(leaves)["root"]

    # Every leaf produces a direction-aware proof that the verifier accepts.
    for i, leaf in enumerate(leaves):
        proof = merkle_inclusion_proof(i, leaves, root)
        assert len(proof) == 2  # tree of height 2
        assert all("sibling" in step and "sibling_is_left" in step for step in proof)
        assert verify_inclusion_proof(leaf, proof, root) is True


def test_verify_inclusion_proof_rejects_wrong_leaf():
    leaves = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
    root = merkle_root_from_items(leaves)["root"]
    proof = merkle_inclusion_proof(0, leaves, root)
    # A different leaf with the same proof must not verify (direction bits + hash).
    assert verify_inclusion_proof({"id": 999}, proof, root) is False


def test_merkle_inclusion_proof_invalid_index():
    leaves = [{"id": 1}]
    root = merkle_root_from_items(leaves)["root"]

    with pytest.raises(IndexError):
        merkle_inclusion_proof(1, leaves, root)


def test_merkle_inclusion_proof_invalid_root():
    leaves = [{"id": 1}, {"id": 2}]

    with pytest.raises(ValueError, match="Root mismatch"):
        merkle_inclusion_proof(0, leaves, "wrong_root")
