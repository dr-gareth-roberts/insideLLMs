import pytest
from insideLLMs.privacy.disclosure import merkle_inclusion_proof
from insideLLMs.crypto.merkle import merkle_root_from_items, _hash_pair
from insideLLMs.crypto.canonical import digest_bytes, canonical_json_bytes

def test_merkle_inclusion_proof():
    leaves = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
    root_info = merkle_root_from_items(leaves)
    root = root_info["root"]
    
    # Test for leaf 0
    proof_0 = merkle_inclusion_proof(0, leaves, root)
    assert len(proof_0) == 2 # tree of height 2
    
    # Verify proof 0 manually
    leaf_0_hash = digest_bytes(canonical_json_bytes(leaves[0]))
    h1 = _hash_pair(leaf_0_hash, proof_0[0], "sha256")
    h2 = _hash_pair(h1, proof_0[1], "sha256")
    assert h2 == root
    
    # Test for leaf 2
    proof_2 = merkle_inclusion_proof(2, leaves, root)
    assert len(proof_2) == 2
    
    # Verify proof 2 manually
    leaf_2_hash = digest_bytes(canonical_json_bytes(leaves[2]))
    h1 = _hash_pair(leaf_2_hash, proof_2[0], "sha256")
    h2 = _hash_pair(proof_2[1], h1, "sha256")
    assert h2 == root

def test_merkle_inclusion_proof_invalid_index():
    leaves = [{"id": 1}]
    root = merkle_root_from_items(leaves)["root"]
    
    with pytest.raises(IndexError):
        merkle_inclusion_proof(1, leaves, root)

def test_merkle_inclusion_proof_invalid_root():
    leaves = [{"id": 1}, {"id": 2}]
    
    with pytest.raises(ValueError, match="Root mismatch"):
        merkle_inclusion_proof(0, leaves, "wrong_root")
