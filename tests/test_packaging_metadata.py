import re
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")


def _optional_dependencies() -> dict[str, list[str]]:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def test_all_extra_references_only_declared_extras() -> None:
    extras = _optional_dependencies()
    referenced: set[str] = set()

    for requirement in extras["all"]:
        match = re.fullmatch(r"insideLLMs\[([^]]+)\]", requirement)
        if match:
            referenced.update(name.strip() for name in match.group(1).split(","))

    undeclared = referenced - set(extras)
    assert undeclared == set(), f"all references undeclared extras: {sorted(undeclared)}"


def test_crypto_extra_installs_fernet_dependency() -> None:
    assert _optional_dependencies()["crypto"] == ["cryptography>=41.0.0"]


def test_base_install_includes_validation_dependency() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    assert "pydantic>=2.0.0,<3" in project["dependencies"]
    assert "Development Status :: 4 - Beta" in project["classifiers"]
