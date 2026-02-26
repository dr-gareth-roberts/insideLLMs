import pytest

from insideLLMs.datasets.tuf_client import fetch_dataset


def test_fetch_dataset():
    path, proof = fetch_dataset("dummy_ds", "1.0", base_url="https://example.com")
    assert path.exists()
    assert proof["status"] == "verified"
