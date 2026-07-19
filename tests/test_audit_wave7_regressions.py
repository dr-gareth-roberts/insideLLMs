"""Regression tests for MONSTER_LOOP wave 7 fixes."""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path


def test_visualization_shim_sunset_documented_consistently():
    """IMPORT_PATHS, CHANGELOG, and shim docstring must agree on v2.0.0 removal."""
    repo_root = Path(__file__).resolve().parents[1]
    import_paths = (repo_root / "docs" / "IMPORT_PATHS.md").read_text(encoding="utf-8")
    changelog = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
    shim_doc = (repo_root / "insideLLMs" / "visualization.py").read_text(encoding="utf-8")

    assert "Deprecated; removal in v2.0.0" in import_paths
    assert "removed in v2.0.0" in changelog
    assert "removed in v2.0.0" in shim_doc
    assert "DeprecationWarning" in shim_doc
    assert "indefinitely" not in shim_doc.lower()
    assert "is not deprecated" not in shim_doc


def test_visualization_shim_emits_deprecation_warning_on_import():
    """CHANGELOG migration timeline requires a DeprecationWarning on shim import."""
    sys.modules.pop("insideLLMs.visualization", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        mod = importlib.import_module("insideLLMs.visualization")

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "expected DeprecationWarning when importing insideLLMs.visualization"
    message = str(deprecations[0].message)
    assert "v2.0.0" in message
    assert "insideLLMs.analysis.visualization" in message
    # Shim still aliases the canonical module object.
    canonical = importlib.import_module("insideLLMs.analysis.visualization")
    assert mod is canonical
