"""Tests for high-level runtime workflow helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from insideLLMs.runtime.workflows import diff_run_dirs, run_harness_to_dir


def test_run_harness_to_dir_delegates_to_cli(tmp_path: Path) -> None:
    config_path = tmp_path / "harness.yaml"
    config_path.write_text("models: []\n", encoding="utf-8")
    run_dir = tmp_path / "run"

    with patch("insideLLMs.cli.commands.harness.cmd_harness", return_value=0) as cmd_harness:
        code = run_harness_to_dir(config_path, run_dir, track_project="wf-test")

    assert code == 0
    args = cmd_harness.call_args.args[0]
    assert args.config == str(config_path.resolve())
    assert args.run_dir == str(run_dir.resolve())
    assert args.strict_serialization is True
    assert args.deterministic_artifacts is True
    assert args.track_project == "wf-test"


def test_run_harness_to_dir_uses_default_flags(tmp_path: Path) -> None:
    config_path = tmp_path / "harness.yaml"
    config_path.write_text("models: []\n", encoding="utf-8")

    with patch("insideLLMs.cli.commands.harness.cmd_harness", return_value=0) as cmd_harness:
        code = run_harness_to_dir(config_path, tmp_path / "run")

    assert code == 0
    args = cmd_harness.call_args.args[0]
    assert args.quiet is True
    assert args.verbose is False
    assert args.validate_output is True
    assert args.deterministic_artifacts is True
    assert args.strict_serialization is True


def test_run_harness_to_dir_overrides_flags_and_kwargs(tmp_path: Path) -> None:
    config_path = tmp_path / "harness.yaml"
    config_path.write_text("models: []\n", encoding="utf-8")

    with patch("insideLLMs.cli.commands.harness.cmd_harness", return_value=0) as cmd_harness:
        code = run_harness_to_dir(
            config_path,
            tmp_path / "run",
            quiet=False,
            verbose=True,
            validate_output=False,
            deterministic_artifacts=False,
            strict_serialization=False,
            schema_version="v2",
            validation_mode="strict",
        )

    assert code == 0
    args = cmd_harness.call_args.args[0]
    assert args.quiet is False
    assert args.verbose is True
    assert args.validate_output is False
    assert args.deterministic_artifacts is False
    assert args.strict_serialization is False
    assert args.schema_version == "v2"
    assert args.validation_mode == "strict"


def test_run_harness_to_dir_rejects_conflicting_verbosity(tmp_path: Path) -> None:
    config_path = tmp_path / "harness.yaml"
    config_path.write_text("models: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_harness_to_dir(config_path, tmp_path / "run", verbose=True, quiet=True)


def test_run_harness_to_dir_rejects_missing_config(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_harness_to_dir(tmp_path / "missing.yaml", tmp_path / "run")


def test_run_harness_to_dir_rejects_invalid_validation_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "harness.yaml"
    config_path.write_text("models: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="validation_mode"):
        run_harness_to_dir(config_path, tmp_path / "run", validation_mode="bad")  # type: ignore[arg-type]


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
    assert args.run_dir_a == str(run_a.resolve())
    assert args.run_dir_b == str(run_b.resolve())
    assert args.fail_on_changes is True
    assert args.fail_on_trace_drift is True
    assert args.format == "json"
    assert args.output == str((tmp_path / "diff.json").resolve())
    assert args.output_fingerprint_ignore == ["latency_ms"]


def test_diff_run_dirs_rejects_invalid_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="limit"):
        diff_run_dirs(tmp_path / "a", tmp_path / "b", limit=0)


def test_diff_run_dirs_repeated_calls_are_stateless(tmp_path: Path) -> None:
    with patch("insideLLMs.cli.commands.diff.cmd_diff", return_value=0) as cmd_diff:
        assert diff_run_dirs(tmp_path / "a", tmp_path / "b", output_fingerprint_ignore=["a"]) == 0
        assert diff_run_dirs(tmp_path / "a", tmp_path / "b") == 0

    first = cmd_diff.call_args_list[0].args[0]
    second = cmd_diff.call_args_list[1].args[0]
    assert first.output_fingerprint_ignore == ["a"]
    assert second.output_fingerprint_ignore == []
