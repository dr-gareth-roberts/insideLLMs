"""Import smoke tests for every ``insideLLMs.contrib`` module."""

import importlib
import pkgutil

import pytest

import insideLLMs.contrib as contrib_package


def _discover_contrib_modules() -> list[str]:
    prefix = f"{contrib_package.__name__}."
    return sorted(
        module_info.name
        for module_info in pkgutil.walk_packages(contrib_package.__path__, prefix=prefix)
        if not module_info.name.rsplit(".", maxsplit=1)[-1].startswith("_")
    )


CONTRIB_MODULES = _discover_contrib_modules()


def test_contrib_module_discovery_is_recursive() -> None:
    """Discovery includes modules from both nested contrib packages."""
    assert "insideLLMs.contrib.claims.compiler" in CONTRIB_MODULES
    assert "insideLLMs.contrib.security.openvex" in CONTRIB_MODULES


@pytest.mark.parametrize("module_name", CONTRIB_MODULES)
def test_contrib_module_imports(module_name: str) -> None:
    """Every contrib module imports unless an optional dependency is absent."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        missing_root = (exc.name or "").split(".", maxsplit=1)[0]
        if not missing_root or missing_root == "insideLLMs":
            raise AssertionError(f"{module_name} has a broken internal import: {exc!r}") from exc
        pytest.skip(f"{module_name} needs optional dependency {missing_root!r}")

    assert module.__name__ == module_name
