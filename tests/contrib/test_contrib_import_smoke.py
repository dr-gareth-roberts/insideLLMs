"""Import/smoke guarantee for every ``insideLLMs.contrib`` module.

Contrib modules are research/analysis utilities that are not part of the core
pipeline. The contract enforced here: each one must import cleanly with only the
core dependencies present. Heavy third-party packages must be imported lazily
(inside functions), not at module top level, so a missing *optional external*
dependency is tolerated (the case is skipped). A broken *internal* import
(a missing ``insideLLMs.*`` module/attribute) is always a failure.
"""

import importlib
import pkgutil

import pytest

import insideLLMs.contrib as contrib_pkg

_CONTRIB_MODULES = sorted(
    name
    for _, name, is_pkg in pkgutil.iter_modules(contrib_pkg.__path__)
    if not is_pkg and not name.startswith("_")
)


def test_contrib_modules_discovered():
    """The contrib package should expose a non-trivial set of modules."""
    assert len(_CONTRIB_MODULES) >= 30, _CONTRIB_MODULES


@pytest.mark.parametrize("module_name", _CONTRIB_MODULES)
def test_contrib_module_imports(module_name):
    """Each contrib module imports; only a missing external dep may skip it."""
    qualified = f"insideLLMs.contrib.{module_name}"
    try:
        module = importlib.import_module(qualified)
    except ImportError as exc:
        missing_root = (exc.name or "").split(".")[0]
        if not missing_root or missing_root.startswith("insideLLMs"):
            raise AssertionError(f"{qualified} has a broken internal import: {exc!r}") from exc
        pytest.skip(f"{qualified} needs optional dependency '{missing_root}' (not installed)")

    assert module is not None
    assert module.__name__ == qualified
