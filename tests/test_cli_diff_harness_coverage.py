"""Comprehensive coverage tests for cli/commands/diff.py and cli/commands/harness.py."""

import argparse
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.cli.commands.diff import cmd_diff
from insideLLMs.cli.commands.harness import cmd_harness

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as JSONL to the given file path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def _make_record(
    *,
    model_id: str = "m1",
    probe_id: str = "p1",
    example_id: str = "ex-1",
    status: str = "success",
    primary_metric: str | None = "score",
    scores: dict[str, float] | None = None,
    output: Any = None,
    output_text: str | None = None,
    run_id: str | None = "run-1",
    custom: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal result record with sensible defaults."""
    rec: dict[str, Any] = {
        "schema_version": "1.0.1",
        "run_id": run_id,
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:00:01+00:00",
        "model": {"model_id": model_id, "provider": "local", "params": {}},
        "probe": {"probe_id": probe_id, "probe_version": "1.0.0", "params": {}},
        "example_id": example_id,
        "status": status,
        "scores": scores if scores is not None else {"score": 0.9},
        "usage": {},
        "custom": custom if custom is not None else {},
    }
    if primary_metric is not None:
        rec["primary_metric"] = primary_metric
    if output is not None:
        rec["output"] = output
    if output_text is not None:
        rec["output_text"] = output_text
    return rec


def _diff_namespace(
    run_dir_a: str = "",
    run_dir_b: str = "",
    fmt: str = "text",
    output: str | None = None,
    limit: int = 50,
    fail_on_regressions: bool = False,
    fail_on_changes: bool = False,
    fail_on_trace_violations: bool = False,
    fail_on_trace_drift: bool = False,
    output_fingerprint_ignore: list[str] | None = None,
) -> argparse.Namespace:
    """Build an argparse.Namespace that matches what cmd_diff expects."""
    return argparse.Namespace(
        run_dir_a=run_dir_a,
        run_dir_b=run_dir_b,
        format=fmt,
        output=output,
        limit=limit,
        fail_on_regressions=fail_on_regressions,
        fail_on_changes=fail_on_changes,
        fail_on_trace_violations=fail_on_trace_violations,
        fail_on_trace_drift=fail_on_trace_drift,
        output_fingerprint_ignore=output_fingerprint_ignore,
    )


def _harness_namespace(
    config: str = "",
    verbose: bool = False,
    quiet: bool = False,
    run_id: str | None = None,
    schema_version: str = "1.0.1",
    strict_serialization: bool = False,
    run_root: str | None = None,
    run_dir: str | None = None,
    output_dir: str | None = None,
    track: str | None = None,
    track_project: str = "default",
    overwrite: bool = False,
    validate_output: bool = False,
    validation_mode: str = "strict",
    deterministic_artifacts: bool = True,
    skip_report: bool = False,
    report_title: str | None = None,
) -> argparse.Namespace:
    """Build an argparse.Namespace that matches what cmd_harness expects."""
    return argparse.Namespace(
        config=config,
        verbose=verbose,
        quiet=quiet,
        run_id=run_id,
        schema_version=schema_version,
        strict_serialization=strict_serialization,
        run_root=run_root,
        run_dir=run_dir,
        output_dir=output_dir,
        track=track,
        track_project=track_project,
        overwrite=overwrite,
        validate_output=validate_output,
        validation_mode=validation_mode,
        deterministic_artifacts=deterministic_artifacts,
        skip_report=skip_report,
        report_title=report_title,
    )


# ===========================================================================
#  DIFF COMMAND TESTS
# ===========================================================================


class TestCmdDiffErrorPaths:
    """Test error paths in cmd_diff."""

    def test_run_dir_a_missing(self, tmp_path: Path) -> None:
        """Non-existent run_dir_a returns 1."""
        args = _diff_namespace(
            run_dir_a=str(tmp_path / "nonexistent"),
            run_dir_b=str(tmp_path),
        )
        assert cmd_diff(args) == 1

    def test_run_dir_b_missing(self, tmp_path: Path) -> None:
        """Non-existent run_dir_b returns 1."""
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        args = _diff_namespace(
            run_dir_a=str(dir_a),
            run_dir_b=str(tmp_path / "nonexistent"),
        )
        assert cmd_diff(args) == 1

    def test_records_a_missing(self, tmp_path: Path) -> None:
        """Missing records.jsonl in dir_a returns 1."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        _write_records(dir_b / "records.jsonl", [_make_record()])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b))
        assert cmd_diff(args) == 1

    def test_records_b_missing(self, tmp_path: Path) -> None:
        """Missing records.jsonl in dir_b returns 1."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        _write_records(dir_a / "records.jsonl", [_make_record()])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b))
        assert cmd_diff(args) == 1

    def test_invalid_jsonl(self, tmp_path: Path) -> None:
        """Malformed JSONL returns 1."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "records.jsonl").write_text("not json\n", encoding="utf-8")
        _write_records(dir_b / "records.jsonl", [_make_record()])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b))
        assert cmd_diff(args) == 1

    def test_empty_records(self, tmp_path: Path) -> None:
        """Empty records in either directory returns 1."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        _write_records(dir_a / "records.jsonl", [])
        _write_records(dir_b / "records.jsonl", [_make_record()])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b))
        assert cmd_diff(args) == 1

    def test_output_with_text_format_warns(self, tmp_path: Path, capsys) -> None:
        """--output with non-json format triggers a warning."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        _write_records(dir_a / "records.jsonl", [_make_record()])
        _write_records(dir_b / "records.jsonl", [_make_record()])
        args = _diff_namespace(
            run_dir_a=str(dir_a),
            run_dir_b=str(dir_b),
            output=str(tmp_path / "out.txt"),
            fmt="text",
        )
        rc = cmd_diff(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "only used with" in captured.out


class TestCmdDiffTextFormat:
    """Test the text-format output of cmd_diff covering comparison branches."""

    def _setup_dirs(self, tmp_path: Path, records_a, records_b):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        _write_records(dir_a / "records.jsonl", records_a)
        _write_records(dir_b / "records.jsonl", records_b)
        return dir_a, dir_b

    def test_identical_records_no_changes(self, tmp_path: Path) -> None:
        """Two identical runs should produce 0 regressions, 0 changes."""
        rec = _make_record()
        dir_a, dir_b = self._setup_dirs(tmp_path, [rec], [rec])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b))
        assert cmd_diff(args) == 0

    def test_status_regression_and_improvement(self, tmp_path: Path, capsys) -> None:
        """Status changes are detected as regressions or improvements."""
        rec_a1 = _make_record(model_id="m1", example_id="e1", status="success")
        rec_b1 = _make_record(model_id="m1", example_id="e1", status="error")
        # Improvement: error -> success
        rec_a2 = _make_record(model_id="m2", example_id="e2", status="error")
        rec_b2 = _make_record(model_id="m2", example_id="e2", status="success")
        dir_a, dir_b = self._setup_dirs(tmp_path, [rec_a1, rec_a2], [rec_b1, rec_b2])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b))
        rc = cmd_diff(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Regressions" in captured.out
        assert "Improvements" in captured.out

    def test_metric_regression_and_improvement(self, tmp_path: Path, capsys) -> None:
        """Score changes are detected as metric regressions or improvements."""
        rec_a = _make_record(example_id="e1", scores={"score": 0.9}, primary_metric="score")
        rec_b = _make_record(example_id="e1", scores={"score": 0.7}, primary_metric="score")
        # Improvement
        rec_a2 = _make_record(
            model_id="m2", example_id="e2", scores={"score": 0.5}, primary_metric="score"
        )
        rec_b2 = _make_record(
            model_id="m2", example_id="e2", scores={"score": 0.8}, primary_metric="score"
        )
        dir_a, dir_b = self._setup_dirs(tmp_path, [rec_a, rec_a2], [rec_b, rec_b2])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b))
        rc = cmd_diff(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Regressions" in captured.out
        assert "Improvements" in captured.out

    def test_only_in_baseline_and_comparison(self, tmp_path: Path, capsys) -> None:
        """Records only in one run are reported."""
        rec_a = _make_record(model_id="m1", example_id="e1")
        rec_b = _make_record(model_id="m2", example_id="e2")
        dir_a, dir_b = self._setup_dirs(tmp_path, [rec_a], [rec_b])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b))
        rc = cmd_diff(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Missing in Comparison" in captured.out
        assert "New in Comparison" in captured.out

    def test_output_text_changed(self, tmp_path: Path, capsys) -> None:
        """Different output_text triggers 'output changed' entry."""
        rec_a = _make_record(example_id="e1", output_text="hello")
        rec_b = _make_record(example_id="e1", output_text="world")
        dir_a, dir_b = self._setup_dirs(tmp_path, [rec_a], [rec_b])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b))
        rc = cmd_diff(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Other Changes" in captured.out

    def test_output_fingerprint_changed_structured(self, tmp_path: Path, capsys) -> None:
        """Different structured output triggers fingerprint diff."""
        rec_a = _make_record(example_id="e1", output={"key": "val_a"})
        rec_b = _make_record(example_id="e1", output={"key": "val_b"})
        dir_a, dir_b = self._setup_dirs(tmp_path, [rec_a], [rec_b])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b))
        rc = cmd_diff(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Other Changes" in captured.out

    def test_output_fingerprint_one_none(self, tmp_path: Path, capsys) -> None:
        """When one record has output and the other has None, 'structured' change."""
        rec_a = _make_record(example_id="e1", output={"data": 1})
        rec_b = _make_record(example_id="e1")
        # rec_b has no output field, so fingerprint_b will be None
        dir_a, dir_b = self._setup_dirs(tmp_path, [rec_a], [rec_b])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b))
        rc = cmd_diff(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Other Changes" in captured.out

    def test_metrics_not_comparable(self, tmp_path: Path, capsys) -> None:
        """Different primary metrics -> metrics not comparable."""
        rec_a = _make_record(
            example_id="e1",
            primary_metric="accuracy",
            scores={"accuracy": 0.8},
        )
        rec_b = _make_record(
            example_id="e1",
            primary_metric="f1",
            scores={"f1": 0.7},
        )
        dir_a, dir_b = self._setup_dirs(tmp_path, [rec_a], [rec_b])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b))
        rc = cmd_diff(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Other Changes" in captured.out

    def test_metric_key_missing(self, tmp_path: Path, capsys) -> None:
        """Metric key sets differ -> metric_key_missing change entry."""
        rec_a = _make_record(
            example_id="e1",
            primary_metric="score",
            scores={"score": 0.9, "extra_a": 0.5},
        )
        rec_b = _make_record(
            example_id="e1",
            primary_metric="score",
            scores={"score": 0.9, "extra_b": 0.6},
        )
        dir_a, dir_b = self._setup_dirs(tmp_path, [rec_a], [rec_b])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b))
        rc = cmd_diff(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Other Changes" in captured.out

    def test_status_changed_non_success(self, tmp_path: Path, capsys) -> None:
        """Status change between two non-success statuses is a general change."""
        rec_a = _make_record(example_id="e1", status="error", scores={})
        rec_b = _make_record(example_id="e1", status="timeout", scores={})
        dir_a, dir_b = self._setup_dirs(tmp_path, [rec_a], [rec_b])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b))
        rc = cmd_diff(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Other Changes" in captured.out

    def test_trace_drift(self, tmp_path: Path, capsys) -> None:
        """Trace fingerprint changes are reported as trace drifts."""
        rec_a = _make_record(
            example_id="e1",
            custom={"trace_fingerprint": "sha256:aaaaaaaaaaaa"},
        )
        rec_b = _make_record(
            example_id="e1",
            custom={"trace_fingerprint": "sha256:bbbbbbbbbbbb"},
        )
        dir_a, dir_b = self._setup_dirs(tmp_path, [rec_a], [rec_b])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b))
        rc = cmd_diff(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Trace" in captured.out

    def test_trace_violation_increase(self, tmp_path: Path, capsys) -> None:
        """Increased trace violations are reported."""
        rec_a = _make_record(
            example_id="e1",
            custom={"trace_violations": [{"rule": "r1"}]},
        )
        rec_b = _make_record(
            example_id="e1",
            custom={"trace_violations": [{"rule": "r1"}, {"rule": "r2"}]},
        )
        dir_a, dir_b = self._setup_dirs(tmp_path, [rec_a], [rec_b])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b))
        rc = cmd_diff(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Trace Violation" in captured.out

    def test_duplicate_records_warned(self, tmp_path: Path, capsys) -> None:
        """Duplicate keys in a run trigger warnings."""
        rec = _make_record(example_id="e1")
        dir_a, dir_b = self._setup_dirs(tmp_path, [rec, rec], [rec, rec])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b))
        rc = cmd_diff(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "duplicate" in captured.out.lower()

    def test_limit_exceeded(self, tmp_path: Path, capsys) -> None:
        """When items exceed --limit, the 'and N more' message is printed."""
        records_a = [_make_record(model_id=f"m{i}", example_id=f"e{i}") for i in range(5)]
        # Make all of them only-in-baseline by having an empty comparison
        records_b = [_make_record(model_id="other", example_id="other")]
        dir_a, dir_b = self._setup_dirs(tmp_path, records_a, records_b)
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b), limit=2)
        rc = cmd_diff(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "more" in captured.out

    def test_fingerprint_ignore_keys(self, tmp_path: Path) -> None:
        """output_fingerprint_ignore strips volatile keys before comparison."""
        rec_a = _make_record(
            example_id="e1",
            output={"data": "same", "timestamp": "2026-01-01"},
        )
        rec_b = _make_record(
            example_id="e1",
            output={"data": "same", "timestamp": "2026-02-02"},
        )
        dir_a, dir_b = self._setup_dirs(tmp_path, [rec_a], [rec_b])
        # Without ignore: fingerprints differ
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b), fmt="json")
        rc = cmd_diff(args)
        assert rc == 0

        # With ignore: fingerprints match (the only volatile key is stripped)
        args2 = _diff_namespace(
            run_dir_a=str(dir_a),
            run_dir_b=str(dir_b),
            fmt="json",
            output_fingerprint_ignore=["timestamp"],
        )
        rc2 = cmd_diff(args2)
        assert rc2 == 0

    def test_fingerprint_ignore_comma_separated(self, tmp_path: Path) -> None:
        """output_fingerprint_ignore supports comma-separated values."""
        rec_a = _make_record(example_id="e1", output={"a": 1, "b": 2, "ts": "x"})
        rec_b = _make_record(example_id="e1", output={"a": 1, "b": 2, "ts": "y"})
        dir_a, dir_b = self._setup_dirs(tmp_path, [rec_a], [rec_b])
        args = _diff_namespace(
            run_dir_a=str(dir_a),
            run_dir_b=str(dir_b),
            fmt="json",
            output_fingerprint_ignore=["ts, extra"],
        )
        rc = cmd_diff(args)
        assert rc == 0


class TestCmdDiffJsonFormat:
    """Test JSON output format of cmd_diff."""

    def _setup_dirs(self, tmp_path, records_a, records_b):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        _write_records(dir_a / "records.jsonl", records_a)
        _write_records(dir_b / "records.jsonl", records_b)
        return dir_a, dir_b

    def test_json_stdout(self, tmp_path: Path, capsys) -> None:
        """JSON output is printed to stdout when no --output given."""
        rec = _make_record()
        dir_a, dir_b = self._setup_dirs(tmp_path, [rec], [rec])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b), fmt="json")
        rc = cmd_diff(args)
        assert rc == 0
        captured = capsys.readouterr()
        report = json.loads(captured.out)
        assert "counts" in report
        assert report["counts"]["regressions"] == 0

    def test_json_to_file(self, tmp_path: Path) -> None:
        """JSON output is written to file when --output given."""
        rec_a = _make_record(example_id="e1", scores={"score": 0.9}, primary_metric="score")
        rec_b = _make_record(example_id="e1", scores={"score": 0.7}, primary_metric="score")
        dir_a, dir_b = self._setup_dirs(tmp_path, [rec_a], [rec_b])
        out_path = tmp_path / "diff.json"
        args = _diff_namespace(
            run_dir_a=str(dir_a),
            run_dir_b=str(dir_b),
            fmt="json",
            output=str(out_path),
        )
        rc = cmd_diff(args)
        assert rc == 0
        assert out_path.exists()
        report = json.loads(out_path.read_text(encoding="utf-8"))
        assert report["counts"]["regressions"] == 1

    def test_json_comprehensive_report(self, tmp_path: Path, capsys) -> None:
        """JSON report captures all types: regressions, improvements, changes, only_*."""
        records_a = [
            # Will be a regression (metric drop)
            _make_record(
                model_id="m1", example_id="e1", scores={"score": 0.9}, primary_metric="score"
            ),
            # Will be an improvement (metric rise)
            _make_record(
                model_id="m1", example_id="e2", scores={"score": 0.5}, primary_metric="score"
            ),
            # Will be only_a
            _make_record(model_id="m1", example_id="e3"),
            # Will be status regression
            _make_record(model_id="m1", example_id="e4", status="success"),
            # Will be status improvement
            _make_record(model_id="m1", example_id="e5", status="error", scores={}),
        ]
        records_b = [
            _make_record(
                model_id="m1", example_id="e1", scores={"score": 0.7}, primary_metric="score"
            ),
            _make_record(
                model_id="m1", example_id="e2", scores={"score": 0.8}, primary_metric="score"
            ),
            # e3 missing from b -> only_a
            _make_record(model_id="m1", example_id="e4", status="error", scores={}),
            _make_record(model_id="m1", example_id="e5", status="success"),
            # only_b
            _make_record(model_id="m1", example_id="e6"),
        ]
        dir_a, dir_b = self._setup_dirs(tmp_path, records_a, records_b)
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b), fmt="json")
        rc = cmd_diff(args)
        assert rc == 0
        report = json.loads(capsys.readouterr().out)
        assert report["counts"]["regressions"] >= 2  # metric + status
        assert report["counts"]["improvements"] >= 2  # metric + status
        assert report["counts"]["only_baseline"] >= 1
        assert report["counts"]["only_candidate"] >= 1


class TestCmdDiffFailFlags:
    """Test exit-code fail flags for cmd_diff."""

    def _setup_regression_dirs(self, tmp_path: Path):
        """Set up dirs where there is a regression."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        rec_a = _make_record(example_id="e1", scores={"score": 0.9}, primary_metric="score")
        rec_b = _make_record(example_id="e1", scores={"score": 0.7}, primary_metric="score")
        _write_records(dir_a / "records.jsonl", [rec_a])
        _write_records(dir_b / "records.jsonl", [rec_b])
        return dir_a, dir_b

    def test_fail_on_regressions_json(self, tmp_path: Path) -> None:
        dir_a, dir_b = self._setup_regression_dirs(tmp_path)
        args = _diff_namespace(
            run_dir_a=str(dir_a),
            run_dir_b=str(dir_b),
            fmt="json",
            fail_on_regressions=True,
        )
        assert cmd_diff(args) == 2

    def test_fail_on_regressions_text(self, tmp_path: Path) -> None:
        dir_a, dir_b = self._setup_regression_dirs(tmp_path)
        args = _diff_namespace(
            run_dir_a=str(dir_a),
            run_dir_b=str(dir_b),
            fmt="text",
            fail_on_regressions=True,
        )
        assert cmd_diff(args) == 2

    def test_fail_on_changes_with_only_a(self, tmp_path: Path) -> None:
        """fail_on_changes triggers when there are only_a entries."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        rec_a = _make_record(model_id="m1", example_id="e1")
        _write_records(dir_a / "records.jsonl", [rec_a])
        _write_records(dir_b / "records.jsonl", [_make_record(model_id="m2", example_id="e2")])
        # JSON format
        args = _diff_namespace(
            run_dir_a=str(dir_a),
            run_dir_b=str(dir_b),
            fmt="json",
            fail_on_changes=True,
        )
        assert cmd_diff(args) == 2

    def test_fail_on_changes_text_format(self, tmp_path: Path) -> None:
        """fail_on_changes works in text format too."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        rec_a = _make_record(example_id="e1", output_text="hello")
        rec_b = _make_record(example_id="e1", output_text="world")
        _write_records(dir_a / "records.jsonl", [rec_a])
        _write_records(dir_b / "records.jsonl", [rec_b])
        args = _diff_namespace(
            run_dir_a=str(dir_a),
            run_dir_b=str(dir_b),
            fmt="text",
            fail_on_changes=True,
        )
        assert cmd_diff(args) == 2

    def test_fail_on_trace_violations_json(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        rec_a = _make_record(example_id="e1", custom={"trace_violations": []})
        rec_b = _make_record(
            example_id="e1",
            custom={"trace_violations": [{"rule": "r1"}]},
        )
        _write_records(dir_a / "records.jsonl", [rec_a])
        _write_records(dir_b / "records.jsonl", [rec_b])
        args = _diff_namespace(
            run_dir_a=str(dir_a),
            run_dir_b=str(dir_b),
            fmt="json",
            fail_on_trace_violations=True,
        )
        assert cmd_diff(args) == 3

    def test_fail_on_trace_violations_text(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        rec_a = _make_record(example_id="e1", custom={"trace_violations": []})
        rec_b = _make_record(
            example_id="e1",
            custom={"trace_violations": [{"rule": "r1"}]},
        )
        _write_records(dir_a / "records.jsonl", [rec_a])
        _write_records(dir_b / "records.jsonl", [rec_b])
        args = _diff_namespace(
            run_dir_a=str(dir_a),
            run_dir_b=str(dir_b),
            fmt="text",
            fail_on_trace_violations=True,
        )
        assert cmd_diff(args) == 3

    def test_fail_on_trace_drift_json(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        rec_a = _make_record(
            example_id="e1",
            custom={"trace_fingerprint": "sha256:aaaa1111bbbb"},
        )
        rec_b = _make_record(
            example_id="e1",
            custom={"trace_fingerprint": "sha256:cccc2222dddd"},
        )
        _write_records(dir_a / "records.jsonl", [rec_a])
        _write_records(dir_b / "records.jsonl", [rec_b])
        args = _diff_namespace(
            run_dir_a=str(dir_a),
            run_dir_b=str(dir_b),
            fmt="json",
            fail_on_trace_drift=True,
        )
        assert cmd_diff(args) == 4

    def test_fail_on_trace_drift_text(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        rec_a = _make_record(
            example_id="e1",
            custom={"trace_fingerprint": "sha256:aaaa1111bbbb"},
        )
        rec_b = _make_record(
            example_id="e1",
            custom={"trace_fingerprint": "sha256:cccc2222dddd"},
        )
        _write_records(dir_a / "records.jsonl", [rec_a])
        _write_records(dir_b / "records.jsonl", [rec_b])
        args = _diff_namespace(
            run_dir_a=str(dir_a),
            run_dir_b=str(dir_b),
            fmt="text",
            fail_on_trace_drift=True,
        )
        assert cmd_diff(args) == 4


class TestCmdDiffRecordIdentity:
    """Test record_identity and record_summary internal logic via JSON output."""

    def test_custom_replicate_key(self, tmp_path: Path, capsys) -> None:
        """Records with custom replicate_key are included in identity."""
        rec_a = _make_record(
            example_id="e1",
            custom={
                "replicate_key": "rep-1",
                "harness": {"model_id": "m1", "probe_type": "p1"},
            },
        )
        rec_b = _make_record(
            example_id="e1",
            output_text="changed",
            custom={
                "replicate_key": "rep-1",
                "harness": {"model_id": "m1", "probe_type": "p1"},
            },
        )
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        _write_records(dir_a / "records.jsonl", [rec_a])
        _write_records(dir_b / "records.jsonl", [rec_b])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b), fmt="json")
        rc = cmd_diff(args)
        assert rc == 0
        report = json.loads(capsys.readouterr().out)
        # Should have a common key and detect change
        assert report["counts"]["common"] >= 1

    def test_record_with_non_dict_custom(self, tmp_path: Path, capsys) -> None:
        """Records where custom is not a dict are handled gracefully."""
        rec_a = _make_record(example_id="e1", custom=None)  # type: ignore[arg-type]
        rec_a["custom"] = "not a dict"
        rec_b = _make_record(example_id="e1")
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        _write_records(dir_a / "records.jsonl", [rec_a])
        _write_records(dir_b / "records.jsonl", [rec_b])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b), fmt="json")
        # Should not crash
        rc = cmd_diff(args)
        assert rc == 0

    def test_non_dict_scores(self, tmp_path: Path, capsys) -> None:
        """Records with non-dict scores are handled."""
        rec_a = _make_record(example_id="e1", primary_metric=None)
        rec_a["scores"] = "bad"
        rec_b = _make_record(example_id="e1", primary_metric=None)
        rec_b["scores"] = "bad"
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        _write_records(dir_a / "records.jsonl", [rec_a])
        _write_records(dir_b / "records.jsonl", [rec_b])
        args = _diff_namespace(run_dir_a=str(dir_a), run_dir_b=str(dir_b), fmt="json")
        rc = cmd_diff(args)
        assert rc == 0


# ===========================================================================
#  HARNESS COMMAND TESTS
# ===========================================================================


def _minimal_harness_result(
    config: dict[str, Any] | None = None,
    records: list[dict[str, Any]] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Return a mock result from run_harness_from_config."""
    if config is None:
        config = {
            "models": [{"type": "dummy", "args": {}}],
            "probes": [{"type": "logic", "args": {}}],
            "dataset": {"format": "jsonl", "path": "/tmp/data.jsonl", "input_field": "question"},
            "max_examples": 1,
        }
    if records is None:
        records = [
            {
                "schema_version": "1.0.1",
                "run_id": run_id or "harness-test",
                "started_at": "2026-01-01T00:00:00+00:00",
                "completed_at": "2026-01-01T00:00:01+00:00",
                "model": {"model_id": "dummy", "provider": "local", "params": {}},
                "probe": {"probe_id": "logic", "probe_version": "1.0.0", "params": {}},
                "example_id": "ex-1",
                "status": "success",
                "scores": {"score": 1.0},
                "usage": {},
                "custom": {
                    "harness": {
                        "model_id": "dummy",
                        "model_name": "Dummy",
                        "model_type": "dummy",
                        "probe_name": "Logic",
                        "probe_type": "logic",
                        "example_index": 0,
                        "experiment_id": "exp-1",
                    },
                },
            },
        ]

    from insideLLMs.types import (
        ExperimentResult,
        ModelInfo,
        ProbeCategory,
        ProbeResult,
        ResultStatus,
    )

    experiments = [
        ExperimentResult(
            experiment_id="exp-1",
            model_info=ModelInfo(name="Dummy", provider="local", model_id="dummy"),
            probe_name="Logic",
            probe_category=ProbeCategory.REASONING,
            results=[
                ProbeResult(
                    input="What is 2+2?",
                    output="4",
                    status=ResultStatus.SUCCESS,
                )
            ],
            score=None,
            started_at=None,
            completed_at=None,
            config={},
        )
    ]

    return {
        "config": config,
        "config_snapshot": config,
        "records": records,
        "summary": {
            "by_model": {"dummy": {"success_rate": {"mean": 1.0}}},
            "by_probe": {"logic": {"success_rate": {"mean": 1.0}}},
        },
        "experiments": experiments,
        "run_id": run_id,
        "strict_serialization": False,
        "deterministic_artifacts": True,
    }


class TestCmdHarnessErrorPaths:
    """Test error paths in cmd_harness."""

    def test_config_not_found(self, tmp_path: Path) -> None:
        args = _harness_namespace(config=str(tmp_path / "nonexistent.yaml"))
        assert cmd_harness(args) == 1

    def test_run_harness_exception(self, tmp_path: Path) -> None:
        """General exception during harness run returns 1."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models: []\nprobes: []\n", encoding="utf-8")
        args = _harness_namespace(config=str(config_path))
        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="test-run-id",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value={"output_dir": str(tmp_path / "results")},
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 1

    def test_run_harness_exception_verbose(self, tmp_path: Path, capsys) -> None:
        """Verbose mode prints traceback on error."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models: []\nprobes: []\n", encoding="utf-8")
        args = _harness_namespace(config=str(config_path), verbose=True)
        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="test-run-id",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value={"output_dir": str(tmp_path / "results")},
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 1
        captured = capsys.readouterr()
        assert "Traceback" in captured.err or "boom" in captured.err


class TestCmdHarnessSuccessPaths:
    """Test successful harness execution with mocked run_harness_from_config."""

    def _run_harness_mocked(
        self,
        tmp_path: Path,
        *,
        run_id: str = "harness-test",
        run_dir: str | None = None,
        run_root: str | None = None,
        output_dir: str | None = None,
        skip_report: bool = False,
        quiet: bool = False,
        schema_version: str = "1.0.1",
        report_title: str | None = None,
        validate_output: bool = False,
        overwrite: bool = False,
        deterministic_artifacts: bool = True,
        strict_serialization: bool = False,
        verbose: bool = False,
        track: str | None = None,
        track_project: str = "default",
    ) -> int:
        """Helper to run cmd_harness with mocked dependencies."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        effective_run_dir = run_dir or str(tmp_path / "run_out")
        args = _harness_namespace(
            config=str(config_path),
            run_id=run_id,
            run_dir=effective_run_dir,
            run_root=run_root,
            output_dir=output_dir,
            skip_report=skip_report,
            quiet=quiet,
            schema_version=schema_version,
            report_title=report_title,
            validate_output=validate_output,
            overwrite=overwrite,
            deterministic_artifacts=deterministic_artifacts,
            strict_serialization=strict_serialization,
            verbose=verbose,
            track=track,
            track_project=track_project,
        )
        result = _minimal_harness_result(run_id=run_id)

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value=run_id,
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
        ):
            return cmd_harness(args)

    def test_basic_success(self, tmp_path: Path) -> None:
        """Basic harness run writes all expected artifacts."""
        rc = self._run_harness_mocked(tmp_path)
        assert rc == 0
        run_dir = tmp_path / "run_out"
        assert (run_dir / "records.jsonl").exists()
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "config.resolved.yaml").exists()
        assert (run_dir / ".insidellms_run").exists()

    def test_skip_report(self, tmp_path: Path) -> None:
        """With --skip-report, report.html is not created."""
        rc = self._run_harness_mocked(tmp_path, skip_report=True)
        assert rc == 0
        run_dir = tmp_path / "run_out"
        assert not (run_dir / "report.html").exists()

    def test_quiet_mode(self, tmp_path: Path, capsys) -> None:
        """Quiet mode suppresses final output."""
        rc = self._run_harness_mocked(tmp_path, quiet=True)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Run written to:" not in captured.out

    def test_custom_report_title(self, tmp_path: Path) -> None:
        """Custom report title is passed through."""
        rc = self._run_harness_mocked(tmp_path, report_title="Custom Title")
        assert rc == 0
        run_dir = tmp_path / "run_out"
        report = (run_dir / "report.html").read_text(encoding="utf-8")
        assert "Custom Title" in report

    def test_schema_version_100(self, tmp_path: Path) -> None:
        """Schema version 1.0.0 should not include run_completed."""
        rc = self._run_harness_mocked(tmp_path, schema_version="1.0.0")
        assert rc == 0
        run_dir = tmp_path / "run_out"
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert "run_completed" not in manifest

    def test_schema_version_101(self, tmp_path: Path) -> None:
        """Schema version 1.0.1 should include run_completed=True."""
        rc = self._run_harness_mocked(tmp_path, schema_version="1.0.1")
        assert rc == 0
        run_dir = tmp_path / "run_out"
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest.get("run_completed") is True

    def test_manifest_run_id(self, tmp_path: Path) -> None:
        """Manifest run_id matches the --run-id argument."""
        rc = self._run_harness_mocked(tmp_path, run_id="custom-id")
        assert rc == 0
        run_dir = tmp_path / "run_out"
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["run_id"] == "custom-id"

    def test_records_run_id_stamped(self, tmp_path: Path) -> None:
        """Records' run_id is overwritten to match resolved_run_id."""
        rc = self._run_harness_mocked(tmp_path, run_id="stamp-id")
        assert rc == 0
        run_dir = tmp_path / "run_out"
        first_line = (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()[0]
        record = json.loads(first_line)
        assert record["run_id"] == "stamp-id"

    def test_legacy_results_symlink(self, tmp_path: Path) -> None:
        """results.jsonl is created alongside records.jsonl."""
        rc = self._run_harness_mocked(tmp_path)
        assert rc == 0
        run_dir = tmp_path / "run_out"
        assert (run_dir / "results.jsonl").exists()

    def test_non_deterministic_artifacts(self, tmp_path: Path) -> None:
        """When deterministic_artifacts=False, python_version and platform are set."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id="nd-test",
            run_dir=str(run_dir_path),
            deterministic_artifacts=False,
        )
        result = _minimal_harness_result(run_id="nd-test")
        result["deterministic_artifacts"] = False

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="nd-test",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0
        manifest = json.loads((run_dir_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["python_version"] is not None
        assert manifest["platform"] is not None


class TestCmdHarnessOutputDirPrecedence:
    """Test output directory resolution precedence."""

    def _run_with_dir_options(
        self, tmp_path, *, run_dir=None, output_dir=None, run_root=None, run_id="test-id"
    ):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "models:\n- type: dummy\nprobes:\n- type: logic\noutput_dir: config_results\n",
            encoding="utf-8",
        )
        args = _harness_namespace(
            config=str(config_path),
            run_id=run_id,
            run_dir=run_dir,
            output_dir=output_dir,
            run_root=run_root,
        )
        result = _minimal_harness_result(run_id=run_id)
        result["config"]["output_dir"] = "config_results"

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value=run_id,
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
        ):
            rc = cmd_harness(args)
        return rc

    def test_run_dir_takes_precedence(self, tmp_path: Path) -> None:
        """--run-dir overrides everything."""
        run_dir = str(tmp_path / "explicit_run_dir")
        rc = self._run_with_dir_options(tmp_path, run_dir=run_dir)
        assert rc == 0
        assert (Path(run_dir) / "manifest.json").exists()

    def test_output_dir_legacy(self, tmp_path: Path) -> None:
        """--output-dir (legacy) is used when --run-dir not given."""
        out_dir = str(tmp_path / "legacy_out")
        rc = self._run_with_dir_options(tmp_path, output_dir=out_dir)
        assert rc == 0
        assert (Path(out_dir) / "manifest.json").exists()

    def test_run_root_with_run_id(self, tmp_path: Path) -> None:
        """--run-root + run_id forms the directory."""
        root = str(tmp_path / "root")
        rc = self._run_with_dir_options(tmp_path, run_root=root, run_id="abc")
        assert rc == 0
        assert (Path(root) / "abc" / "manifest.json").exists()


class TestCmdHarnessTracking:
    """Test tracking integration paths."""

    def test_tracking_local(self, tmp_path: Path) -> None:
        """Local tracking creates tracker and logs metrics."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id="track-local-test",
            run_dir=str(run_dir_path),
            track="local",
            track_project="my-project",
        )
        result = _minimal_harness_result(run_id="track-local-test")
        mock_tracker = MagicMock()

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="track-local-test",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
            patch(
                "insideLLMs.experiment_tracking.create_tracker",
                return_value=mock_tracker,
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0
        mock_tracker.start_run.assert_called_once()
        mock_tracker.log_metrics.assert_called_once()
        mock_tracker.end_run.assert_called_once_with(status="finished")

    def test_tracking_wandb_kwargs(self, tmp_path: Path) -> None:
        """wandb tracking passes project kwarg."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id="track-wandb",
            run_dir=str(run_dir_path),
            track="wandb",
            track_project="wandb-proj",
        )
        result = _minimal_harness_result(run_id="track-wandb")
        mock_tracker = MagicMock()

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="track-wandb",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
            patch(
                "insideLLMs.experiment_tracking.create_tracker",
                return_value=mock_tracker,
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0
        mock_tracker.start_run.assert_called_once()

    def test_tracking_mlflow_kwargs(self, tmp_path: Path) -> None:
        """mlflow tracking passes experiment_name kwarg."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id="track-mlflow",
            run_dir=str(run_dir_path),
            track="mlflow",
            track_project="mlflow-exp",
        )
        result = _minimal_harness_result(run_id="track-mlflow")
        mock_tracker = MagicMock()

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="track-mlflow",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
            patch(
                "insideLLMs.experiment_tracking.create_tracker",
                return_value=mock_tracker,
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0

    def test_tracking_tensorboard_kwargs(self, tmp_path: Path) -> None:
        """tensorboard tracking passes log_dir kwarg."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id="track-tb",
            run_dir=str(run_dir_path),
            track="tensorboard",
            track_project="tb-proj",
        )
        result = _minimal_harness_result(run_id="track-tb")
        mock_tracker = MagicMock()

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="track-tb",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
            patch(
                "insideLLMs.experiment_tracking.create_tracker",
                return_value=mock_tracker,
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0

    def test_tracking_creation_failure(self, tmp_path: Path) -> None:
        """Tracking creation failure falls back gracefully."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id="track-fail",
            run_dir=str(run_dir_path),
            track="local",
            track_project="proj",
        )
        result = _minimal_harness_result(run_id="track-fail")

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="track-fail",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
            patch(
                "insideLLMs.experiment_tracking.create_tracker",
                side_effect=ImportError("no tracking"),
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0

    def test_tracking_log_error(self, tmp_path: Path) -> None:
        """Tracking log error during post-run logging is handled gracefully."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id="track-log-err",
            run_dir=str(run_dir_path),
            track="local",
            track_project="proj",
        )
        result = _minimal_harness_result(run_id="track-log-err")
        mock_tracker = MagicMock()
        mock_tracker.log_metrics.side_effect = RuntimeError("log fail")

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="track-log-err",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
            patch(
                "insideLLMs.experiment_tracking.create_tracker",
                return_value=mock_tracker,
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0

    def test_tracker_cleanup_on_error(self, tmp_path: Path) -> None:
        """Tracker.end_run(status='failed') is called on run exception."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id="track-cleanup",
            run_dir=str(run_dir_path),
            track="local",
            track_project="proj",
        )
        mock_tracker = MagicMock()

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="track-cleanup",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value={"output_dir": str(tmp_path / "results")},
            ),
            patch(
                "insideLLMs.experiment_tracking.create_tracker",
                return_value=mock_tracker,
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 1
        mock_tracker.end_run.assert_called_once_with(status="failed")


class TestCmdHarnessConfigSnapshot:
    """Test config_snapshot and determinism option resolution."""

    def test_fallback_config_snapshot(self, tmp_path: Path) -> None:
        """When config_snapshot not in result, it's built from config."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id="snap-test",
            run_dir=str(run_dir_path),
        )
        result = _minimal_harness_result(run_id="snap-test")
        result["config_snapshot"] = None  # Force fallback path

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="snap-test",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0

    def test_fallback_determinism_options(self, tmp_path: Path) -> None:
        """When strict_serialization/deterministic_artifacts missing, resolved from config."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id="determ-test",
            run_dir=str(run_dir_path),
        )
        result = _minimal_harness_result(run_id="determ-test")
        # Force fallback by removing type info
        result["strict_serialization"] = "not a bool"
        result["deterministic_artifacts"] = "not a bool"

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="determ-test",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0

    def test_run_id_from_result(self, tmp_path: Path) -> None:
        """When no --run-id, uses result['run_id']."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id=None,  # No explicit run_id
            run_dir=str(run_dir_path),
        )
        result = _minimal_harness_result(run_id="result-derived-id")

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="derived-from-config",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0
        manifest = json.loads((run_dir_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["run_id"] == "result-derived-id"

    def test_run_id_fallback_to_deterministic(self, tmp_path: Path) -> None:
        """When result has no run_id and derive returns empty, use deterministic."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id=None,
            run_dir=str(run_dir_path),
        )
        result = _minimal_harness_result(run_id=None)
        result["run_id"] = None

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0
        # A deterministic run_id should have been generated
        manifest = json.loads((run_dir_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["run_id"] is not None
        assert len(manifest["run_id"]) > 0


class TestCmdHarnessProgressCallback:
    """Test progress callback behavior."""

    def test_progress_callback_verbose(self, tmp_path: Path) -> None:
        """Verbose mode enables the progress callback."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id="progress-test",
            run_dir=str(run_dir_path),
            verbose=True,
            quiet=False,
        )
        result = _minimal_harness_result(run_id="progress-test")

        captured_callback = {}

        def mock_run(*_args, **kwargs):
            cb = kwargs.get("progress_callback")
            captured_callback["cb"] = cb
            if cb:
                cb(0, 10)
                cb(5, 10)
                cb(10, 10)
            return result

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                side_effect=mock_run,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="progress-test",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0
        assert captured_callback.get("cb") is not None

    def test_progress_callback_quiet_suppressed(self, tmp_path: Path) -> None:
        """Quiet mode suppresses progress callback output."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id="quiet-progress",
            run_dir=str(run_dir_path),
            verbose=True,
            quiet=True,
        )
        result = _minimal_harness_result(run_id="quiet-progress")

        def mock_run(*_args, **kwargs):
            cb = kwargs.get("progress_callback")
            # With verbose=True but quiet=True, callback should no-op
            if cb:
                cb(1, 10)  # Should not crash even though quiet
            return result

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                side_effect=mock_run,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="quiet-progress",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0


class TestCmdHarnessDatasetSpec:
    """Test dataset spec construction from various config shapes."""

    def test_dataset_spec_fields(self, tmp_path: Path) -> None:
        """Dataset spec picks up name, version, hash, provenance from config."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id="ds-test",
            run_dir=str(run_dir_path),
        )
        result = _minimal_harness_result(run_id="ds-test")
        result["config"]["dataset"] = {
            "name": "my_dataset",
            "version": "2.0",
            "hash": "sha256:abc123",
            "provenance": "hf",
            "path": "/data/file.jsonl",
        }

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="ds-test",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0
        manifest = json.loads((run_dir_path / "manifest.json").read_text(encoding="utf-8"))
        ds = manifest["dataset"]
        assert ds["dataset_id"] == "my_dataset"
        assert ds["dataset_version"] == "2.0"
        assert ds["dataset_hash"] == "sha256:abc123"

    def test_dataset_spec_fallback_fields(self, tmp_path: Path) -> None:
        """Dataset spec falls back to alternative field names."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id="ds-fallback",
            run_dir=str(run_dir_path),
        )
        result = _minimal_harness_result(run_id="ds-fallback")
        result["config"]["dataset"] = {
            "dataset": "alt_name",
            "split": "test",
            "dataset_hash": "sha256:def456",
            "source": "local",
        }

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="ds-fallback",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0
        manifest = json.loads((run_dir_path / "manifest.json").read_text(encoding="utf-8"))
        ds = manifest["dataset"]
        assert ds["dataset_id"] == "alt_name"
        assert ds["dataset_version"] == "test"
        assert ds["dataset_hash"] == "sha256:def456"

    def test_non_dict_dataset_config(self, tmp_path: Path) -> None:
        """Non-dict dataset config is handled gracefully."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id="ds-none",
            run_dir=str(run_dir_path),
        )
        result = _minimal_harness_result(run_id="ds-none")
        result["config"]["dataset"] = "just_a_string"

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="ds-none",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0

    def test_non_list_models_probes(self, tmp_path: Path) -> None:
        """Non-list models/probes config is handled gracefully."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id="bad-lists",
            run_dir=str(run_dir_path),
        )
        result = _minimal_harness_result(run_id="bad-lists")
        result["config"]["models"] = "not a list"
        result["config"]["probes"] = None

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="bad-lists",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0


class TestCmdHarnessOverwrite:
    """Test overwrite behavior for harness run directories."""

    def test_overwrite_existing_dir(self, tmp_path: Path) -> None:
        """With --overwrite and sentinel, existing dir is cleared."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        run_dir_path.mkdir()
        (run_dir_path / ".insidellms_run").write_text("marker\n", encoding="utf-8")
        (run_dir_path / "old_artifact.txt").write_text("old", encoding="utf-8")

        args = _harness_namespace(
            config=str(config_path),
            run_id="overwrite-test",
            run_dir=str(run_dir_path),
            overwrite=True,
        )
        result = _minimal_harness_result(run_id="overwrite-test")

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="overwrite-test",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0
        assert not (run_dir_path / "old_artifact.txt").exists()
        assert (run_dir_path / "manifest.json").exists()


class TestCmdHarnessReportFallback:
    """Test report generation fallback when visualization module is unavailable."""

    def test_fallback_to_basic_report(self, tmp_path: Path) -> None:
        """When visualization import fails, basic report is generated."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id="report-fallback",
            run_dir=str(run_dir_path),
            skip_report=False,
        )
        result = _minimal_harness_result(run_id="report-fallback")

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="report-fallback",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
            patch.dict(
                "sys.modules",
                {"insideLLMs.visualization": None},
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0
        report = (run_dir_path / "report.html").read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in report
        assert "Behavioural Probe Report" in report


class TestCmdHarnessNonDictConfig:
    """Test harness when load_config returns a non-dict."""

    def test_non_dict_config_for_output_dir(self, tmp_path: Path) -> None:
        """When load_config returns a non-dict, output_dir defaults to 'results'."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id="nondict-test",
            run_dir=str(run_dir_path),
        )
        result = _minimal_harness_result(run_id="nondict-test")

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="nondict-test",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value="not a dict",
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0


class TestCmdHarnessValidateOutput:
    """Test --validate-output flag exercises schema validation paths."""

    def test_validate_output_manifest_and_summary(self, tmp_path: Path) -> None:
        """validate_output triggers OutputValidator on manifest and summary."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id="validate-test",
            run_dir=str(run_dir_path),
            validate_output=True,
            validation_mode="warn",
            skip_report=True,
        )
        result = _minimal_harness_result(run_id="validate-test")

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="validate-test",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0


class TestCmdHarnessConfigDefaultDir:
    """Test the config_default_dir fallback (no --run-dir, --output-dir, --run-root)."""

    def test_falls_back_to_config_output_dir(self, tmp_path: Path) -> None:
        """When no explicit dir options, uses config output_dir."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        expected_dir = tmp_path / "my_results"
        args = _harness_namespace(
            config=str(config_path),
            run_id="fallback-dir-test",
            run_dir=None,
            output_dir=None,
            run_root=None,
        )
        result = _minimal_harness_result(run_id="fallback-dir-test")
        result["config"]["output_dir"] = str(expected_dir)

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="fallback-dir-test",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0
        assert expected_dir.exists()
        assert (expected_dir / "manifest.json").exists()


class TestCmdHarnessTrackerEndRunFailure:
    """Test tracker.end_run() failure during exception handling."""

    def test_tracker_end_run_raises_attribute_error(self, tmp_path: Path) -> None:
        """When tracker.end_run raises AttributeError, harness still returns 1."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        args = _harness_namespace(
            config=str(config_path),
            run_id="tracker-end-fail",
            run_dir=str(tmp_path / "run_out"),
            track="local",
            track_project="proj",
        )
        mock_tracker = MagicMock()
        mock_tracker.end_run.side_effect = AttributeError("no end_run")

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="tracker-end-fail",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value={"output_dir": str(tmp_path / "results")},
            ),
            patch(
                "insideLLMs.experiment_tracking.create_tracker",
                return_value=mock_tracker,
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 1

    def test_tracker_end_run_raises_runtime_error(self, tmp_path: Path) -> None:
        """When tracker.end_run raises RuntimeError, harness still returns 1."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        args = _harness_namespace(
            config=str(config_path),
            run_id="tracker-end-rt",
            run_dir=str(tmp_path / "run_out"),
            track="local",
            track_project="proj",
        )
        mock_tracker = MagicMock()
        mock_tracker.end_run.side_effect = RuntimeError("end failed")

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="tracker-end-rt",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value={"output_dir": str(tmp_path / "results")},
            ),
            patch(
                "insideLLMs.experiment_tracking.create_tracker",
                return_value=mock_tracker,
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 1


class TestCmdHarnessLegacySymlinkFallback:
    """Test legacy results.jsonl fallback when symlink fails."""

    def test_symlink_failure_falls_back_to_copy(self, tmp_path: Path) -> None:
        """When os.symlink fails, fallback to hardlink or copy."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n- type: dummy\nprobes:\n- type: logic\n", encoding="utf-8")
        run_dir_path = tmp_path / "run_out"
        args = _harness_namespace(
            config=str(config_path),
            run_id="sym-fallback",
            run_dir=str(run_dir_path),
        )
        result = _minimal_harness_result(run_id="sym-fallback")

        with (
            patch(
                "insideLLMs.cli.commands.harness.run_harness_from_config",
                return_value=result,
            ),
            patch(
                "insideLLMs.cli.commands.harness.derive_run_id_from_config_path",
                return_value="sym-fallback",
            ),
            patch(
                "insideLLMs.cli.commands.harness.load_config",
                return_value=result["config"],
            ),
            patch(
                "os.symlink",
                side_effect=OSError("symlink not supported"),
            ),
            patch(
                "os.link",
                side_effect=OSError("hardlink not supported"),
            ),
        ):
            rc = cmd_harness(args)
        assert rc == 0
        # results.jsonl should exist (via shutil.copyfile fallback)
        assert (run_dir_path / "results.jsonl").exists()
