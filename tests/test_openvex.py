import json
from pathlib import Path

from insideLLMs.security.openvex import emit_openvex


def test_emit_openvex(tmp_path):
    manifest = {
        "model": {"model_id": "test-model", "provider": "test-provider"},
        "probe": {"probe_id": "test-probe"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    vex = emit_openvex(tmp_path)
    assert vex["@context"] == "https://openvex.dev/ns/v0.2.0"
    assert "statements" in vex
    assert isinstance(vex["statements"], list)

    # It should have a statement for the model
    statements = vex["statements"]
    assert any(s["products"][0]["@id"].endswith("test-model") for s in statements)
