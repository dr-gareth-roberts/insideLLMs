from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import insideLLMs

pytestmark = pytest.mark.contract


def test_docs_audit_script_passes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "audit_docs.py"
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"audit_docs.py failed with code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def test_exported_api_names_match_exact_public_api_index_rows() -> None:
    from scripts.audit_docs import _get_documented_api_names, _get_exported_api_names

    exported = _get_exported_api_names()
    api_reference = (Path(__file__).resolve().parents[1] / "API_REFERENCE.md").read_text(
        encoding="utf-8"
    )
    documented = _get_documented_api_names(api_reference)

    assert exported
    assert documented == exported, (
        f"missing rows: {sorted(exported - documented)}; "
        f"stale rows: {sorted(documented - exported)}"
    )


def test_documented_api_names_only_accepts_exact_first_column_rows() -> None:
    from scripts.audit_docs import _get_documented_api_names

    markdown = """# API

Budget is mentioned in prose but has no index row.

## Public API Index

| Name | Import Path | Summary |
|---|---|---|
| `Model` | `from insideLLMs import Model` | Base model. |

## Other

| `ProbeRunner` | not part of the API index | Must not count. |
"""

    assert _get_documented_api_names(markdown) == {"Model"}


def test_lazy_import_extraction_fails_closed_when_map_is_missing(tmp_path, monkeypatch) -> None:
    from scripts.audit_docs import _get_lazy_import_names

    fake_init = tmp_path / "__init__.py"
    fake_init.write_text("__all__ = ['Model']\n", encoding="utf-8")
    monkeypatch.setattr(insideLLMs, "__file__", str(fake_init))

    with pytest.raises(RuntimeError, match="_LAZY_IMPORTS"):
        _get_lazy_import_names()
