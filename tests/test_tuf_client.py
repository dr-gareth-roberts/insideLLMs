import builtins

import pytest

from insideLLMs.datasets.tuf_client import fetch_dataset


def test_fetch_dataset_with_explicit_mock():
    path, proof = fetch_dataset("dummy_ds", "1.0", base_url="https://example.com", allow_mock=True)
    assert path.exists()
    assert proof["status"] in {"verified", "mock-verified"}


def test_fetch_dataset_refuses_implicit_mock():
    """When tuf is unavailable, allow_mock=False must refuse the mock path."""
    real_import = builtins.__import__

    def _block_tuf(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tuf" or name.startswith("tuf."):
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(builtins, "__import__", _block_tuf)
        with pytest.raises(RuntimeError, match="refusing mock verification"):
            fetch_dataset("dummy_ds", "1.0", base_url="https://example.com")
