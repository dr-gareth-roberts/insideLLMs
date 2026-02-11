"""Tests for high-level runtime workflow helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from insideLLMs.runtime.workflows import diff_run_dirs, run_harness_to_dir


def test_run_harness_to_dir_delegates_to_cli(tmp_path: Path) -> None:
    config_path = tmp_path / "harness.yaml"
    config_path.write_text("models: []\n", encoding="utf-8")
    run_dir = tmp_path / "run"

    with patch("insideLLMs.cli.commands.harness.cmd_harness", return_value=0) as cmd_harness:
        code = run_harness_to_dir(config_path, run_dir, track_project="wf-test")

    assert code == 0
    args = cmd_harness.call_args.args[0]
    assert args.config == str(config_path)
    assert args.run_dir == str(run_dir)
    assert args.strict_serialization is True
    assert args.deterministic_artifacts is True
    assert args.track_project == "wf-test"


def test_diff_run_dirs_delegates_to_cli(tmp_path: Path) -> None:
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"

    with patch("insideLLMs.cli.commands.diff.cmd_diff", return_value=2) as cmd_diff:
        code = diff_run_dirs(
            run_a,
            run_b,
            fail_on_changes=True,
            fail_on_trace_drift=True,
            output_format="json",
            output_path=tmp_path / "diff.json",
            output_fingerprint_ignore=["latency_ms"],
        )

    assert code == 2
    args = cmd_diff.call_args.args[0]
    assert args.run_dir_a == str(run_a)
    assert args.run_dir_b == str(run_b)
    assert args.fail_on_changes is True
    assert args.fail_on_trace_drift is True
    assert args.format == "json"
    assert args.output == str(tmp_path / "diff.json")
    assert args.output_fingerprint_ignore == ["latency_ms"]
