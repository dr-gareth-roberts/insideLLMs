from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.architecture_evidence import (
    API_MANIFEST_PATH,
    IMPORT_GRAPH_PATH,
    PUBLIC_STATUSES,
    build_api_manifest,
    collect_api_usage,
    generated_outputs,
    scan_imports,
)


def test_import_scanner_resolves_relative_imports(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text(
        "from ..contrib import retrieval\nfrom .. import models\nfrom insideLLMs import contrib\n",
        encoding="utf-8",
    )

    assert {
        reference.imported
        for reference in scan_imports(module, containing_package=("insideLLMs", "inference"))
    } == {
        "insideLLMs.contrib",
        "insideLLMs.contrib.retrieval",
        "insideLLMs.models",
    }


def test_import_scanner_resolves_nested_relative_imports(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("from ... import models\n", encoding="utf-8")

    assert {
        reference.imported
        for reference in scan_imports(
            module, containing_package=("insideLLMs", "inference", "nested")
        )
    } == {"insideLLMs.models"}


def test_import_scanner_honours_an_explicit_empty_package(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("from . import local_helper\n", encoding="utf-8")

    assert {reference.imported for reference in scan_imports(module, containing_package=())} == {
        "local_helper"
    }


def test_import_scanner_finds_literal_dynamic_imports(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text(
        "import importlib\n"
        '_LAZY_IMPORTS = {"Feature": "insideLLMs.contrib.feature"}\n'
        'importlib.import_module("openai")\n'
        '_lazy_import_factory("insideLLMs.models.base", "Model")\n'
        'label = "insideLLMs.not_an_import"\n',
        encoding="utf-8",
    )

    imports = {reference.imported for reference in scan_imports(module, containing_package=())}

    assert "insideLLMs.contrib.feature" in imports
    assert "insideLLMs.models.base" in imports
    assert "openai" in imports
    assert "insideLLMs.not_an_import" not in imports


def test_api_manifest_covers_every_root_export_once() -> None:
    manifest = build_api_manifest()
    names = [symbol["name"] for symbol in manifest["symbols"]]

    assert names == sorted(set(names))
    assert len(names) == manifest["summary"]["symbol_count"]
    assert all(symbol["status"] in PUBLIC_STATUSES for symbol in manifest["symbols"])
    assert all(symbol["providers"] for symbol in manifest["symbols"])


def test_checked_in_architecture_evidence_is_current() -> None:
    outputs = generated_outputs()

    assert outputs[API_MANIFEST_PATH] == API_MANIFEST_PATH.read_text(encoding="utf-8")
    assert outputs[IMPORT_GRAPH_PATH] == IMPORT_GRAPH_PATH.read_text(encoding="utf-8")
    for path, expected in outputs.items():
        assert path.read_text(encoding="utf-8") == expected, f"stale generated evidence: {path}"


def test_api_usage_counts_only_git_tracked_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evidence generated in a working checkout must match CI's clean checkout.

    Regression: a gitignored, locally generated ``benchmarks/codemap.md`` was counted
    as a consumer of ``model_registry``, so CI's regenerated evidence never matched.
    """
    # Git hooks export these, which would redirect the scratch repo's git calls.
    for variable in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(variable, raising=False)
    docs = tmp_path / "docs"
    docs.mkdir()
    (tmp_path / ".gitignore").write_text("codemap.md\n", encoding="utf-8")
    (docs / "guide.md").write_text("from insideLLMs import ProbeRunner\n", encoding="utf-8")
    (docs / "codemap.md").write_text("from insideLLMs import model_registry\n", encoding="utf-8")
    (docs / "notes.md").write_text("from insideLLMs import DummyModel\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", ".gitignore", "docs/guide.md"], cwd=tmp_path, check=True)
    monkeypatch.setattr("scripts.architecture_evidence.REPO_ROOT", tmp_path)
    monkeypatch.setattr("scripts.architecture_evidence.API_STATUS_DOC_PATH", docs / "API_STATUS.md")

    usage = collect_api_usage()

    assert usage["ProbeRunner"]["paths"] == ["docs/guide.md"]
    assert "model_registry" not in usage  # gitignored
    assert "DummyModel" not in usage  # untracked


def test_import_graph_summary_is_not_vacuous() -> None:
    graph = json.loads(IMPORT_GRAPH_PATH.read_text(encoding="utf-8"))

    assert graph["summary"]["module_count"] > 0
    assert graph["summary"]["internal_edge_count"] > 0
