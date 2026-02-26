"""Tests for crypto Merkle tree utilities."""

from pathlib import Path

import pytest

from insideLLMs.crypto.merkle import merkle_root_from_items, merkle_root_from_jsonl


def test_merkle_root_from_items_empty() -> None:
    """Empty list yields root of empty bytes."""
    out = merkle_root_from_items([])
    assert out["root"]
    assert out["count"] == 0
    assert out["algo"] == "sha256"
    assert out["canon_version"] == "canon_v1"


def test_merkle_root_from_items_single() -> None:
    """Single item: root is hash of that item's canonical form."""
    out = merkle_root_from_items([{"id": 1}])
    assert len(out["root"]) == 64
    assert out["count"] == 1


def test_merkle_root_from_items_deterministic() -> None:
    """Same items in same order yield same root."""
    items = [{"a": 1}, {"a": 2}, {"a": 3}]
    r1 = merkle_root_from_items(items)["root"]
    r2 = merkle_root_from_items(items)["root"]
    assert r1 == r2


def test_merkle_root_from_items_order_matters() -> None:
    """Different order yields different root."""
    r1 = merkle_root_from_items([{"a": 1}, {"a": 2}])["root"]
    r2 = merkle_root_from_items([{"a": 2}, {"a": 1}])["root"]
    assert r1 != r2


def test_merkle_root_from_jsonl(tmp_path: Path) -> None:
    """Merkle root from JSONL file matches root from list of parsed objects."""
    path = tmp_path / "data.jsonl"
    path.write_text('{"x":1}\n{"x":2}\n', encoding="utf-8")
    out = merkle_root_from_jsonl(path)
    assert out["count"] == 2
    assert out["root"]
    expected = merkle_root_from_items([{"x": 1}, {"x": 2}])
    assert out["root"] == expected["root"]


def test_merkle_root_from_jsonl_missing_file() -> None:
    """Missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        merkle_root_from_jsonl("/nonexistent/path.jsonl")
