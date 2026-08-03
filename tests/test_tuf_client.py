import pytest

from insideLLMs.datasets.tuf_client import fetch_dataset


def test_fetch_dataset_with_explicit_mock():
    path, proof = fetch_dataset("dummy_ds", "1.0", base_url="https://example.com", allow_mock=True)
    assert path.exists()
    assert proof["status"] == "mock"
    assert proof["method"] == "mock"
    assert proof["verified"] is False


def test_fetch_dataset_refuses_without_explicit_mock():
    """Real TUF verification is not implemented: allow_mock=False must always refuse."""
    with pytest.raises(RuntimeError, match="refusing mock verification"):
        fetch_dataset("dummy_ds", "1.0", base_url="https://example.com")


def test_fetch_dataset_never_reports_verified():
    """The mock path must never masquerade as TUF verification."""
    _, proof = fetch_dataset("dummy_ds", "1.0", allow_mock=True)
    assert proof["status"] != "verified"
    assert proof["method"] != "tuf.ngclient"
