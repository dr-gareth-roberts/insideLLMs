"""Tests for crypto canonicalization and digest utilities."""

import pytest

from insideLLMs.crypto.canonical import (
    DEFAULT_ALGO,
    DEFAULT_CANON_VERSION,
    canonical_json_bytes,
    digest_bytes,
    digest_obj,
    run_bundle_id,
)


def test_canonical_json_bytes_deterministic() -> None:
    """Equivalent structures produce identical bytes."""
    a = canonical_json_bytes({"b": 2, "a": 1})
    b = canonical_json_bytes({"a": 1, "b": 2})
    assert a == b
    assert isinstance(a, bytes)
    assert a.decode("utf-8") == '{"a":1,"b":2}'


def test_canonical_json_bytes_canon_version() -> None:
    """Only canon_v1 is supported."""
    canonical_json_bytes({"x": 1}, canon_version="canon_v1")
    with pytest.raises(ValueError, match="Unsupported canon_version"):
        canonical_json_bytes({"x": 1}, canon_version="canon_v2")


def test_digest_bytes_sha256() -> None:
    """digest_bytes returns hex sha256."""
    d = digest_bytes(b"hello")
    assert len(d) == 64
    assert all(c in "0123456789abcdef" for c in d)


def test_digest_bytes_unsupported_algo() -> None:
    """Unsupported algo raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported digest algo"):
        digest_bytes(b"x", algo="md5")


def test_digest_obj_metadata() -> None:
    """digest_obj returns dict with algo, canon_version, purpose, created_by."""
    out = digest_obj({"k": "v"}, purpose="record")
    assert "digest" in out
    assert out["algo"] == DEFAULT_ALGO
    assert out["canon_version"] == DEFAULT_CANON_VERSION
    assert out["purpose"] == "record"
    assert "created_by" in out


def test_digest_obj_deterministic() -> None:
    """Same object yields same digest."""
    obj = {"a": [1, 2], "b": {"x": 3}}
    d1 = digest_obj(obj, purpose="record")
    d2 = digest_obj(obj, purpose="record")
    assert d1["digest"] == d2["digest"]


def test_digest_obj_purpose_and_created_by() -> None:
    """Explicit purpose and created_by are stored."""
    out = digest_obj({"x": 1}, purpose="attestation", created_by="1.0.0")
    assert out["purpose"] == "attestation"
    assert out["created_by"] == "1.0.0"


def test_run_bundle_id_deterministic() -> None:
    """Same inputs produce same bundle ID."""
    r = run_bundle_id("abc123", {"records": "r1", "receipts": "r2"}, ["d1", "d2"])
    assert len(r) == 64
    assert r == run_bundle_id("abc123", {"records": "r1", "receipts": "r2"}, ["d1", "d2"])


def test_run_bundle_id_sorted_roots_and_digests() -> None:
    """Bundle ID is order-independent for roots and attestation digests."""
    a = run_bundle_id("m", {"b": "2", "a": "1"}, ["z", "y"])
    b = run_bundle_id("m", {"a": "1", "b": "2"}, ["y", "z"])
    assert a == b
