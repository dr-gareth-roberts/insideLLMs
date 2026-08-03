"""Tests for optional-extra import smoke checks."""

import pytest

from scripts import check_optional_imports


def test_each_declared_feature_extra_has_imports() -> None:
    assert set(check_optional_imports.EXTRA_IMPORTS) == {
        "openai",
        "anthropic",
        "huggingface",
        "signing",
        "crypto",
        "nlp",
        "visualization",
        "langchain",
        "serving",
        "providers",
    }
    assert all(
        check_optional_imports.imports_for(extra) for extra in check_optional_imports.EXTRA_IMPORTS
    )


def test_all_imports_are_deduplicated() -> None:
    modules = check_optional_imports.imports_for("all")
    assert len(modules) == len(set(modules))


def test_unknown_extra_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown extra"):
        check_optional_imports.imports_for("missing")


def test_check_imports_reports_failures(monkeypatch) -> None:
    def fake_import(module: str):
        if module == "broken":
            raise ImportError("not installed")
        return object()

    monkeypatch.setattr(check_optional_imports.importlib, "import_module", fake_import)

    assert check_optional_imports.check_imports(("working", "broken")) == [
        "broken: ImportError: not installed"
    ]
