"""Regression tests for doctor behavior with missing optional dependencies."""

from __future__ import annotations

import argparse
import json


def test_cmd_doctor_json_handles_missing_optional_dependencies(monkeypatch, capsys) -> None:
    """Doctor JSON mode should return warnings instead of crashing."""
    from insideLLMs.cli.commands import doctor as doctor_module

    monkeypatch.setattr(doctor_module, "_has_module", lambda _module: False)
    monkeypatch.setattr(doctor_module, "_check_nltk_resource", lambda _path: False)

    args = argparse.Namespace(format="json", fail_on_warn=False, capabilities=False)
    rc = doctor_module.cmd_doctor(args)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert "checks" in payload
    assert "warnings" in payload
    assert any(item["name"] == "nltk" and item["ok"] is False for item in payload["checks"])
