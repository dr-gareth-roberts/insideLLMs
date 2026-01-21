"""Tests for CI-style diff gating behavior."""

import json
from pathlib import Path

from insideLLMs.cli import main


def _write_minimal_run_config(tmp_path: Path, *, canned_response: str) -> Path:
    import yaml

    data_path = tmp_path / "data.jsonl"
    data_path.write_text('{"question": "Is 2+2=4?"}\n', encoding="utf-8")

    config_path = tmp_path / f"config_{canned_response}.yaml"
    config = {
        "model": {"type": "dummy", "args": {"canned_response": canned_response}},
        "probe": {"type": "logic", "args": {}},
        "dataset": {"path": str(data_path), "format": "jsonl"},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_diff_fail_on_changes_exit_code(tmp_path: Path) -> None:
    base_cfg = _write_minimal_run_config(tmp_path, canned_response="A")
    head_cfg = _write_minimal_run_config(tmp_path, canned_response="B")

    run_dir_a = tmp_path / "run_a"
    run_dir_b = tmp_path / "run_b"

    rc_a = main(["run", str(base_cfg), "--format", "summary", "--run-dir", str(run_dir_a)])
    rc_b = main(["run", str(head_cfg), "--format", "summary", "--run-dir", str(run_dir_b)])
    assert rc_a == 0
    assert rc_b == 0

    diff_path = tmp_path / "diff.json"
    rc = main(
        [
            "diff",
            str(run_dir_a),
            str(run_dir_b),
            "--format",
            "json",
            "--output",
            str(diff_path),
            "--fail-on-changes",
        ]
    )
    assert rc == 2

    payload = json.loads(diff_path.read_text(encoding="utf-8"))
    assert payload["schema_version"]
    assert isinstance(payload["changes"], list)
