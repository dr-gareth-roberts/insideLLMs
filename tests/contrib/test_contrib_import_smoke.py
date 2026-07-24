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
    """Every contrib module imports with the project's core dependencies."""
    module = importlib.import_module(module_name)
    assert module.__name__ == module_name
