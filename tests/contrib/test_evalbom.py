import json
from pathlib import Path

from insideLLMs.contrib.evalbom import emit_cyclonedx, emit_spdx3


def test_emit_cyclonedx(tmp_path):
    manifest = {
        "model": {"model_id": "test-model", "provider": "test-provider"},
        "probe": {"probe_id": "test-probe"},
        "dataset": {"dataset_id": "test-dataset"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    bom = emit_cyclonedx(tmp_path)
    assert bom["bomFormat"] == "CycloneDX"
    assert len(bom["components"]) > 0
    assert any(c["name"] == "test-model" for c in bom["components"])


def test_emit_spdx3(tmp_path):
    manifest = {"model": {"model_id": "test-model", "provider": "test-provider"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    bom = emit_spdx3(tmp_path)
    assert bom["spdxVersion"] == "SPDX-3.0"
    assert len(bom["elements"]) > 0
    assert any(e["name"] == "test-model" for e in bom["elements"] if e["type"] == "Software")


def test_emit_evalbom_edges(tmp_path):
    # no manifest
    assert emit_cyclonedx(tmp_path)["components"] == []
    assert emit_spdx3(tmp_path)["elements"] == []

    # invalid JSON → swallowed
    (tmp_path / "manifest.json").write_text("{not-json", encoding="utf-8")
    assert emit_cyclonedx(tmp_path)["components"] == []
    assert emit_spdx3(tmp_path)["elements"] == []

    # model without model_id skipped; probe+dataset present
    manifest = {
        "model": {"provider": "p"},
        "probe": {"probe_id": "pr"},
        "dataset": {"dataset_id": "ds"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cdx = emit_cyclonedx(tmp_path)
    assert [c["name"] for c in cdx["components"]] == ["pr", "ds"]
    spdx = emit_spdx3(tmp_path)
    assert [e["name"] for e in spdx["elements"]] == ["pr", "ds"]
