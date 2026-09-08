"""Optional crypto absence must skip encryption tests without breaking collection."""

import builtins
import runpy
from pathlib import Path

import pytest


def test_staging_module_skips_explicitly_when_crypto_is_absent(monkeypatch):
    original_import = builtins.__import__

    def without_crypto(name, *args, **kwargs):
        if name == "cryptography" or name.startswith("cryptography."):
            raise ModuleNotFoundError("No module named 'cryptography'", name="cryptography")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_crypto)
    with pytest.raises(pytest.skip.Exception, match="cryptography"):
        runpy.run_path(str(Path(__file__).with_name("test_encryption_staging.py")))
