import pytest

from insideLLMs.datasets.tuf_client import fetch_dataset


def test_fetch_dataset_with_explicit_mock():
    path, proof = fetch_dataset("dummy_ds", "1.0", base_url="https://example.com", allow_mock=True)
    assert path.exists()
    assert proof["status"] in {"verified", "mock-verified"}


def test_fetch_dataset_refuses_implicit_mock():
    with pytest.raises(RuntimeError, match="refusing mock verification"):
        fetch_dataset("dummy_ds", "1.0", base_url="https://example.com")
