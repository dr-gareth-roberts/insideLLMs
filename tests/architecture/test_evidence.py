from __future__ import annotations

import json
from pathlib import Path

from scripts.architecture_evidence import (
    API_MANIFEST_PATH,
    IMPORT_GRAPH_PATH,
    PUBLIC_STATUSES,
    build_api_manifest,
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


def test_import_graph_summary_is_not_vacuous() -> None:
    graph = json.loads(IMPORT_GRAPH_PATH.read_text(encoding="utf-8"))

    assert graph["summary"]["module_count"] > 0
    assert graph["summary"]["internal_edge_count"] > 0
