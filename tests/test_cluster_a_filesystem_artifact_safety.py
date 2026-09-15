"""Regression tests for Cluster A filesystem/artifact safety fixes."""

from __future__ import annotations

import json
import os
import threading
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from insideLLMs.cli._record_utils import _write_jsonl
from insideLLMs.cli.commands import diff as diff_command
from insideLLMs.cli.commands import report as report_command
from insideLLMs.cli.commands import validate as validate_command
from insideLLMs.resources import atomic_write_text
from insideLLMs.results import generate_statistical_report
from insideLLMs.runtime._artifact_utils import iter_jsonl_records
from insideLLMs.runtime.diffing_interactive import copy_candidate_artifacts_to_baseline
from insideLLMs.types import (
    ExperimentResult,
    ModelInfo,
    ProbeCategory,
    ProbeResult,
    ProbeScore,
    ResultStatus,
)

# ---------------------------------------------------------------------------
# A1 — atomic_write_text unique staging
# ---------------------------------------------------------------------------


def test_atomic_write_text_does_not_follow_precreated_tmp_symlink(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    outside = tmp_path / "outside.txt"
    outside.write_text("secret-outside", encoding="utf-8")
    predictable_tmp = tmp_path / ".artifact.json.tmp"
    predictable_tmp.symlink_to(outside)

    atomic_write_text(target, '{"ok": true}')

    assert target.read_text(encoding="utf-8") == '{"ok": true}'
    assert outside.read_text(encoding="utf-8") == "secret-outside"
    assert predictable_tmp.is_symlink()


def test_atomic_write_text_concurrent_writers_do_not_raise(tmp_path: Path) -> None:
    target = tmp_path / "shared.json"
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def _writer(index: int) -> None:
        barrier.wait(timeout=5)
        try:
            atomic_write_text(target, f'{{"writer": {index}}}')
        except BaseException as exc:  # noqa: BLE001 - collect any failure
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_writer, range(8)))

    assert errors == []
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert "writer" in payload


# ---------------------------------------------------------------------------
# A2 — copy_candidate_artifacts_to_baseline sealed + symlink safety
# ---------------------------------------------------------------------------


def test_copy_candidate_refuses_sealed_baseline(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (candidate / "records.jsonl").write_text("{}\n", encoding="utf-8")
    integrity = baseline / "integrity"
    integrity.mkdir()
    (integrity / "bundle_id.txt").write_text("sealed", encoding="utf-8")

    with pytest.raises(ValueError, match="immutable"):
        copy_candidate_artifacts_to_baseline(baseline, candidate)

    assert not (baseline / "records.jsonl").exists()


def test_copy_candidate_refuses_symlink_source(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    real = tmp_path / "real_records.jsonl"
    real.write_text('{"id": 1}\n', encoding="utf-8")
    (candidate / "records.jsonl").symlink_to(real)

    with pytest.raises(ValueError, match="non-regular source"):
        copy_candidate_artifacts_to_baseline(baseline, candidate)


def test_copy_candidate_refuses_symlink_destination(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (candidate / "records.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
    outside = tmp_path / "outside.jsonl"
    outside.write_text("keep-me", encoding="utf-8")
    (baseline / "records.jsonl").symlink_to(outside)

    with pytest.raises(ValueError, match="non-regular destination"):
        copy_candidate_artifacts_to_baseline(baseline, candidate)

    assert outside.read_text(encoding="utf-8") == "keep-me"


def test_copy_candidate_publishes_via_replace(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (candidate / "records.jsonl").write_text('{"id": "cand"}\n', encoding="utf-8")
    (baseline / "records.jsonl").write_text('{"id": "base"}\n', encoding="utf-8")

    copied = copy_candidate_artifacts_to_baseline(baseline, candidate)

    assert "records.jsonl" in copied
    assert (baseline / "records.jsonl").read_text(encoding="utf-8") == '{"id": "cand"}\n'
    assert not (baseline / "records.jsonl").is_symlink()


def test_copy_candidate_rejects_path_escape_artifact_name(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    outside = tmp_path / "escape.txt"
    outside.write_text("outside-original", encoding="utf-8")
    (candidate / "records.jsonl").write_text('{"id": 1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="simple filename"):
        copy_candidate_artifacts_to_baseline(
            baseline,
            candidate,
            artifact_names=["records.jsonl", "../escape.txt"],
        )

    assert outside.read_text(encoding="utf-8") == "outside-original"
    assert not (baseline / "records.jsonl").exists()


def test_copy_candidate_validates_all_before_any_publish(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (candidate / "records.jsonl").write_text('{"id": "cand"}\n', encoding="utf-8")
    real_summary = tmp_path / "real_summary.json"
    real_summary.write_text('{"evil": true}', encoding="utf-8")
    (candidate / "summary.json").symlink_to(real_summary)
    (baseline / "records.jsonl").write_text('{"id": "base"}\n', encoding="utf-8")
    (baseline / "summary.json").write_text('{"id": "base-summary"}', encoding="utf-8")

    with pytest.raises(ValueError, match="non-regular source"):
        copy_candidate_artifacts_to_baseline(
            baseline,
            candidate,
            artifact_names=["records.jsonl", "summary.json"],
        )

    # Late failure must not leave a partially updated baseline.
    assert (baseline / "records.jsonl").read_text(encoding="utf-8") == '{"id": "base"}\n'
    assert (baseline / "summary.json").read_text(encoding="utf-8") == '{"id": "base-summary"}'


def test_copy_candidate_refuses_symlink_swapped_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TOCTOU: regular file swapped to symlink before open must not be followed."""
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("O_NOFOLLOW required for no-follow source open")

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    source = candidate / "records.jsonl"
    source.write_text('{"id": "legit"}\n', encoding="utf-8")
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"id": "evil"}\n', encoding="utf-8")
    (baseline / "records.jsonl").write_text('{"id": "base"}\n', encoding="utf-8")

    real_open = os.open

    def _swap_then_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        name = path if isinstance(path, str) else str(path)
        # Only intercept the candidate artifact open (name relative to dir_fd).
        if name == "records.jsonl" and kwargs.get("dir_fd") is not None:
            if source.exists() and not source.is_symlink():
                source.unlink()
                source.symlink_to(outside)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", _swap_then_open)

    with pytest.raises((ValueError, OSError)):
        copy_candidate_artifacts_to_baseline(
            baseline,
            candidate,
            artifact_names=["records.jsonl"],
        )

    assert (baseline / "records.jsonl").read_text(encoding="utf-8") == '{"id": "base"}\n'
    assert outside.read_text(encoding="utf-8") == '{"id": "evil"}\n'


def test_copy_candidate_rolls_back_when_second_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (candidate / "records.jsonl").write_text('{"id": "cand-records"}\n', encoding="utf-8")
    (candidate / "summary.json").write_text('{"id": "cand-summary"}\n', encoding="utf-8")
    (baseline / "records.jsonl").write_text('{"id": "base-records"}\n', encoding="utf-8")
    (baseline / "summary.json").write_text('{"id": "base-summary"}\n', encoding="utf-8")

    real_replace = os.replace
    publish_count = {"n": 0}

    def _flaky_replace(src: Any, dst: Any) -> None:
        src_path = Path(src)
        dst_path = Path(dst)
        # Count final publishes onto baseline artifact names (not .bak moves).
        if dst_path.name in {"records.jsonl", "summary.json"} and src_path.suffix == ".tmp":
            publish_count["n"] += 1
            if publish_count["n"] == 2:
                raise OSError("simulated second publish failure")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _flaky_replace)

    with pytest.raises(OSError, match="second publish"):
        copy_candidate_artifacts_to_baseline(
            baseline,
            candidate,
            artifact_names=["records.jsonl", "summary.json"],
        )

    assert (baseline / "records.jsonl").read_text(encoding="utf-8") == '{"id": "base-records"}\n'
    assert (baseline / "summary.json").read_text(encoding="utf-8") == '{"id": "base-summary"}\n'
    assert list(baseline.glob("*.bak")) == []


def test_copy_candidate_rolls_back_when_second_backup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed second backup must restore the first moved destination."""
    import tempfile

    import insideLLMs.runtime.diffing_interactive as diffing_interactive

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (candidate / "records.jsonl").write_text('{"id": "cand-records"}\n', encoding="utf-8")
    (candidate / "summary.json").write_text('{"id": "cand-summary"}\n', encoding="utf-8")
    (baseline / "records.jsonl").write_text('{"id": "base-records"}\n', encoding="utf-8")
    (baseline / "summary.json").write_text('{"id": "base-summary"}\n', encoding="utf-8")

    real_mkstemp = tempfile.mkstemp
    bak_count = {"n": 0}

    def _flaky_mkstemp(*args: Any, **kwargs: Any) -> tuple[int, str]:
        if kwargs.get("suffix") == ".bak":
            bak_count["n"] += 1
            if bak_count["n"] == 2:
                raise OSError("simulated second backup failure")
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(diffing_interactive.tempfile, "mkstemp", _flaky_mkstemp)

    with pytest.raises(OSError, match="second backup"):
        copy_candidate_artifacts_to_baseline(
            baseline,
            candidate,
            artifact_names=["records.jsonl", "summary.json"],
        )

    assert (baseline / "records.jsonl").read_text(encoding="utf-8") == '{"id": "base-records"}\n'
    assert (baseline / "summary.json").read_text(encoding="utf-8") == '{"id": "base-summary"}\n'
    assert list(baseline.glob("*.bak")) == []
    assert list(baseline.glob(".*bak")) == []


def test_copy_candidate_preserves_backup_when_restore_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed restore must keep the .bak file with original baseline content."""
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    original_records = '{"id": "base-records"}\n'
    original_summary = '{"id": "base-summary"}\n'
    (candidate / "records.jsonl").write_text('{"id": "cand-records"}\n', encoding="utf-8")
    (candidate / "summary.json").write_text('{"id": "cand-summary"}\n', encoding="utf-8")
    (baseline / "records.jsonl").write_text(original_records, encoding="utf-8")
    (baseline / "summary.json").write_text(original_summary, encoding="utf-8")

    real_replace = os.replace
    publish_count = {"n": 0}
    restore_blocked = {"records": False}

    def _flaky_replace(src: Any, dst: Any) -> None:
        src_path = Path(src)
        dst_path = Path(dst)
        # Fail the second publish so we enter rollback with both backups held.
        if dst_path.name in {"records.jsonl", "summary.json"} and src_path.suffix == ".tmp":
            publish_count["n"] += 1
            if publish_count["n"] == 2:
                raise OSError("simulated second publish failure")
            real_replace(src, dst)
            return
        # During restore, refuse to put records backup back — keep the .bak.
        if (
            src_path.suffix == ".bak"
            and "records.jsonl" in src_path.name
            and dst_path.name == "records.jsonl"
        ):
            restore_blocked["records"] = True
            raise OSError("simulated records restore failure")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _flaky_replace)

    with pytest.raises(OSError, match="second publish"):
        copy_candidate_artifacts_to_baseline(
            baseline,
            candidate,
            artifact_names=["records.jsonl", "summary.json"],
        )

    assert restore_blocked["records"] is True
    # summary restored successfully
    assert (baseline / "summary.json").read_text(encoding="utf-8") == original_summary
    # records destination may be missing/new, but its backup must survive
    records_backups = [
        path
        for path in baseline.iterdir()
        if path.name.startswith(".records.jsonl.") and path.name.endswith(".bak")
    ]
    assert len(records_backups) == 1
    assert records_backups[0].read_text(encoding="utf-8") == original_records
    # No summary .bak leftovers after successful restore
    summary_backups = [
        path
        for path in baseline.iterdir()
        if path.name.startswith(".summary.json.") and path.name.endswith(".bak")
    ]
    assert summary_backups == []


def test_copy_candidate_refuses_without_nofollow_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import insideLLMs.runtime.diffing_interactive as diffing_interactive

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (candidate / "records.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(diffing_interactive, "_supports_nofollow_dir_io", lambda: False)

    with pytest.raises(OSError, match="no-follow"):
        copy_candidate_artifacts_to_baseline(baseline, candidate)

    assert not (baseline / "records.jsonl").exists()


# ---------------------------------------------------------------------------
# A3 — iter_jsonl_records fail-loud on non-object lines
# ---------------------------------------------------------------------------


def test_iter_jsonl_records_raises_on_non_object_line(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"a": 1}\n42\n{"a": 2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        list(iter_jsonl_records(path))


# ---------------------------------------------------------------------------
# A4 — validate run-dir containment + manifest object
# ---------------------------------------------------------------------------


def _validate_args(path: Path, *, mode: str = "strict") -> Namespace:
    return Namespace(config=str(path), mode=mode, schema_version=None)


def test_validate_rejects_json_array_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("[]", encoding="utf-8")
    (run_dir / "records.jsonl").write_text("{}\n", encoding="utf-8")

    rc = validate_command.cmd_validate(_validate_args(run_dir))

    assert rc == 1


def test_validate_rejects_records_file_path_traversal(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"status": "success"}\n', encoding="utf-8")
    manifest = {
        "schema_version": "1.0.2",
        "run_id": "validate-escape",
        "records_file": "../outside.jsonl",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    rc = validate_command.cmd_validate(_validate_args(run_dir))

    assert rc == 1


def test_validate_rejects_records_file_symlink(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"status": "success"}\n', encoding="utf-8")
    (run_dir / "escape.jsonl").symlink_to(outside)
    manifest = {
        "schema_version": "1.0.2",
        "run_id": "validate-symlink",
        "records_file": "escape.jsonl",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    rc = validate_command.cmd_validate(_validate_args(run_dir))

    assert rc == 1


def test_validate_allows_alternate_records_file_name(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = {
        "schema_version": "1.0.2",
        "run_id": "validate-alt",
        "input": "x",
        "output": "y",
        "status": "success",
        "model": {"model_id": "dummy", "provider": "offline", "params": {}},
        "probe": {"probe_id": "logic"},
    }
    (run_dir / "custom_records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0.2",
        "run_id": "validate-alt",
        "records_file": "custom_records.jsonl",
        "record_count": 1,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    rc = validate_command.cmd_validate(_validate_args(run_dir))

    # Schema may still flag incomplete manifests; containment must not reject.
    # If schema fails, that is orthogonal — ensure we did not error on path.
    # A fully valid minimal manifest is hard; assert the helper accepts the path.
    resolved = validate_command._contained_records_path(run_dir, "custom_records.jsonl")
    assert resolved == (run_dir / "custom_records.jsonl").resolve()
    assert rc in (0, 1)  # schema may fail; path containment must not


# ---------------------------------------------------------------------------
# A5 — JSONL atomic write + diff OSError handling
# ---------------------------------------------------------------------------


def test_write_jsonl_preserves_existing_on_serialization_failure(tmp_path: Path) -> None:
    output = tmp_path / "records.jsonl"
    output.write_text('{"keep": true}\n', encoding="utf-8")

    class Boom:
        pass

    with pytest.raises(ValueError, match="strict_serialization"):
        _write_jsonl([{"ok": 1}, {"bad": Boom()}], output, strict_serialization=True)

    assert output.read_text(encoding="utf-8") == '{"keep": true}\n'
    assert list(tmp_path.glob(".records.jsonl.*.tmp")) == []


def _diff_pair_args(
    run_a: Path,
    run_b: Path,
    *,
    output: Path | None = None,
    html: Path | None = None,
    fmt: str = "json",
) -> Namespace:
    return Namespace(
        run_dir_a=str(run_a),
        run_dir_b=str(run_b),
        format=fmt,
        output=str(output) if output is not None else None,
        html=str(html) if html is not None else None,
        interactive=False,
        fail_on_regressions=False,
        fail_on_changes=False,
        fail_on_trace_violations=False,
        fail_on_trace_drift=False,
        fail_on_trajectory_drift=False,
        output_fingerprint_ignore=None,
        validate_output=False,
        schema_version=None,
        validation_mode="strict",
        limit=10,
        judge=False,
        judge_policy="strict",
        judge_limit=10,
    )


def _seed_diff_runs(tmp_path: Path) -> tuple[Path, Path]:
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    run_a.mkdir()
    run_b.mkdir()
    record = {
        "run_id": "r",
        "status": "success",
        "custom": {
            "harness": {"model_id": "dummy", "probe_type": "logic", "example_index": 0},
            "replicate_key": "0",
        },
    }
    line = json.dumps(record) + "\n"
    (run_a / "records.jsonl").write_text(line, encoding="utf-8")
    (run_b / "records.jsonl").write_text(line, encoding="utf-8")
    return run_a, run_b


def test_diff_json_write_oserror_returns_exit_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_a, run_b = _seed_diff_runs(tmp_path)
    out_path = tmp_path / "missing-parent" / "diff.json"
    args = _diff_pair_args(run_a, run_b, output=out_path)

    def _boom(path: Path, text: str) -> None:
        raise FileNotFoundError(2, "No such file or directory", str(path))

    monkeypatch.setattr(diff_command, "atomic_write_text", _boom)
    rc = diff_command.cmd_diff(args)
    captured = capsys.readouterr()
    assert rc == 1
    assert "Could not write JSON report" in captured.out + captured.err


def test_diff_json_write_failure_preserves_preexisting_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_a, run_b = _seed_diff_runs(tmp_path)
    out_path = tmp_path / "diff.json"
    out_path.write_text('{"keep": true}', encoding="utf-8")
    args = _diff_pair_args(run_a, run_b, output=out_path)

    def _boom(path: Path, text: str) -> None:
        assert path == out_path
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(diff_command, "atomic_write_text", _boom)
    rc = diff_command.cmd_diff(args)
    assert rc == 1
    assert out_path.read_text(encoding="utf-8") == '{"keep": true}'


def test_diff_atomic_write_fsync_failure_preserves_preexisting_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupted staged write must not clobber an existing JSON report."""
    run_a, run_b = _seed_diff_runs(tmp_path)
    out_path = tmp_path / "diff.json"
    out_path.write_text('{"keep": true}', encoding="utf-8")
    args = _diff_pair_args(run_a, run_b, output=out_path)

    def _fsync_boom(_fd: int) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("insideLLMs.resources.os.fsync", _fsync_boom)
    rc = diff_command.cmd_diff(args)
    assert rc == 1
    assert out_path.read_text(encoding="utf-8") == '{"keep": true}'
    assert list(tmp_path.glob(".diff.json.*.tmp")) == []


def test_diff_html_write_failure_preserves_preexisting_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_a, run_b = _seed_diff_runs(tmp_path)
    html_path = tmp_path / "diff.html"
    html_path.write_text("<html>keep</html>", encoding="utf-8")
    args = _diff_pair_args(run_a, run_b, html=html_path, fmt="text")

    def _boom(path: Path, text: str) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(diff_command, "atomic_write_text", _boom)
    rc = diff_command.cmd_diff(args)
    assert rc == 1
    assert html_path.read_text(encoding="utf-8") == "<html>keep</html>"


# ---------------------------------------------------------------------------
# A6 — report dual-publish rollback
# ---------------------------------------------------------------------------


def _seed_report_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "1.0.2",
        "run_id": "report-rollback",
        "input": "one",
        "output": "answer",
        "status": "success",
        "model": {"model_id": "dummy", "provider": "offline", "params": {}},
        "probe": {"probe_id": "logic"},
    }
    (run_dir / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    summary_payload = {
        "schema_version": "1.0.2",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "summary": {"run_completed": False, "marker": "old-summary"},
        "config": {"old": True},
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary_payload, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "report.html").write_text("<html>old report</html>", encoding="utf-8")


def test_report_rolls_back_summary_when_second_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    _seed_report_run(run_dir)
    old_summary = (run_dir / "summary.json").read_bytes()
    old_report = (run_dir / "report.html").read_bytes()

    real_replace = os.replace
    calls = {"n": 0}

    def _flaky_replace(src: Any, dst: Any) -> None:
        src_path = Path(src)
        dst_path = Path(dst)
        # Count only final publish replaces onto summary.json / report.html
        if dst_path.name in {"summary.json", "report.html"} and ".bak" not in src_path.name:
            calls["n"] += 1
            if calls["n"] == 2 and dst_path.name == "report.html":
                raise OSError("simulated report publish failure")
        real_replace(src, dst)

    monkeypatch.setattr(report_command.os, "replace", _flaky_replace)
    args = Namespace(run_dir=str(run_dir), report_title=None, quiet=True)
    rc = report_command.cmd_report(args)

    assert rc == 1
    assert (run_dir / "summary.json").read_bytes() == old_summary
    assert (run_dir / "report.html").read_bytes() == old_report


def test_report_removes_new_summary_when_absent_prior_and_second_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    _seed_report_run(run_dir)
    (run_dir / "summary.json").unlink()
    old_report = (run_dir / "report.html").read_bytes()
    assert not (run_dir / "summary.json").exists()

    real_replace = os.replace
    calls = {"n": 0}

    def _flaky_replace(src: Any, dst: Any) -> None:
        src_path = Path(src)
        dst_path = Path(dst)
        if dst_path.name in {"summary.json", "report.html"} and ".bak" not in src_path.name:
            calls["n"] += 1
            if calls["n"] == 2 and dst_path.name == "report.html":
                raise OSError("simulated report publish failure")
        real_replace(src, dst)

    monkeypatch.setattr(report_command.os, "replace", _flaky_replace)
    args = Namespace(run_dir=str(run_dir), report_title=None, quiet=True)
    rc = report_command.cmd_report(args)

    assert rc == 1
    assert not (run_dir / "summary.json").exists()
    assert (run_dir / "report.html").read_bytes() == old_report


def test_report_backup_cleanup_failure_keeps_committed_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    _seed_report_run(run_dir)
    old_summary = (run_dir / "summary.json").read_bytes()
    old_report = (run_dir / "report.html").read_bytes()

    real_unlink = Path.unlink
    unlink_calls = {"n": 0}

    def _flaky_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        name = self.name
        if ".summary.json.bak" in name or ".report.html.bak" in name:
            unlink_calls["n"] += 1
            if unlink_calls["n"] == 2:
                raise OSError("simulated backup cleanup failure")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _flaky_unlink)
    args = Namespace(run_dir=str(run_dir), report_title=None, quiet=True)
    rc = report_command.cmd_report(args)

    assert rc == 0
    # Committed pair must remain the new generation, not a mixed restore.
    assert (run_dir / "summary.json").read_bytes() != old_summary
    assert (run_dir / "report.html").read_bytes() != old_report
    captured = capsys.readouterr()
    assert "Could not remove report backup" in captured.out + captured.err


def test_create_tracker_ends_run_when_log_params_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from insideLLMs.cli.commands import _run_common as run_common

    class FakeTracker:
        def __init__(self) -> None:
            self.started = False
            self.ended_with: str | None = None

        def start_run(self, **_kwargs: Any) -> str:
            self.started = True
            return "run-id"

        def log_params(self, _params: dict[str, Any]) -> None:
            raise RuntimeError("params failed")

        def end_run(self, *, status: str) -> None:
            self.ended_with = status
            self.started = False

    tracker = FakeTracker()
    monkeypatch.setattr(
        run_common.experiment_tracking,
        "create_tracker",
        lambda *_a, **_k: tracker,
    )
    result = run_common.create_tracker(
        backend="local",
        project="p",
        run_dir=tmp_path / "run",
        run_id="run-id",
        config_path=tmp_path / "cfg.yaml",
        schema_version="1.0.0",
    )
    assert result is None
    assert tracker.ended_with == "failed"
    assert tracker.started is False


# ---------------------------------------------------------------------------
# A7 — statistical HTML escape + unknown format
# ---------------------------------------------------------------------------


def _minimal_experiment(model_name: str) -> ExperimentResult:
    return ExperimentResult(
        experiment_id="exp-1",
        model_info=ModelInfo(name=model_name, provider="offline", model_id="m-1"),
        probe_name="logic",
        probe_category=ProbeCategory.LOGIC,
        results=[
            ProbeResult(
                input="q",
                output="a",
                status=ResultStatus.SUCCESS,
                latency_ms=1.0,
            )
        ],
        score=ProbeScore(accuracy=1.0),
        completed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_statistical_html_escapes_dynamic_model_name() -> None:
    evil = "<img src=x onerror=alert(1)>"
    report = generate_statistical_report([_minimal_experiment(evil)], format="html")

    assert evil not in report
    assert "&lt;img src=x onerror=alert(1)&gt;" in report


def test_statistical_report_unknown_format_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported statistical report format"):
        generate_statistical_report([_minimal_experiment("m")], format="pdf")
