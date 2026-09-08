"""Behavioural checks for unhealthy-run exits and strict comparison policies."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from insideLLMs.cli import main
from insideLLMs.models import DummyModel
from insideLLMs.runtime._run_health import assess_run_health, check_run_directory_health
from insideLLMs.runtime.diffing import (
    DiffGatePolicy,
    build_diff_computation,
    compute_diff_exit_code,
)


def _config(tmp_path: Path, command: str) -> Path:
    (tmp_path / "data.jsonl").write_text('{"question":"One?"}\n{"question":"Two?"}\n')
    config = {"dataset": {"format": "jsonl", "path": "data.jsonl"}}
    if command == "harness":
        config.update(models=[{"type": "dummy"}], probes=[{"type": "logic"}])
    else:
        config.update(model={"type": "dummy"}, probe={"type": "logic"})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("command", ["run", "harness"])
def test_all_error_run_exits_nonzero_and_retains_artifacts(tmp_path, monkeypatch, command):
    def fail_provider(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(DummyModel, "generate", fail_provider)
    config = _config(tmp_path, command)
    run_dir = tmp_path / "run"
    args = [command, str(config), "--run-dir", str(run_dir), "--quiet"]
    if command == "harness":
        args.append("--skip-report")
    assert main(args) == 1
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert len(records) == 2
    assert all(record["status"] == "error" for record in records)
    assert manifest["run_completed"] is True
    assert manifest["custom"]["health"]["healthy"] is False
    assert check_run_directory_health(run_dir)["healthy"] is False


@pytest.mark.parametrize("status", ["error", "timeout", "skipped", "unknown"])
def test_health_rejects_every_non_success_status(status):
    health = assess_run_health([{"status": "success"}, {"status": status}], expected_count=2)
    assert health["healthy"] is False
    assert health["status_counts"][status] == 1


def test_health_rejects_empty_incomplete_and_truncated_successful_runs():
    assert assess_run_health([])["healthy"] is False
    assert assess_run_health([{"status": "success"}], run_completed=False)["healthy"] is False
    truncated = assess_run_health([{"status": "success"}], expected_count=2)
    assert truncated["healthy"] is False
    assert "Expected 2 records, found 1" in truncated["reasons"]


def test_action_rejects_identical_old_all_error_runs_before_comparison(tmp_path):
    for label in ("baseline", "candidate"):
        directory = tmp_path / label
        directory.mkdir()
        (directory / "records.jsonl").write_text('{"status":"error"}\n')
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "run_completed": True,
                    "record_count": 1,
                    "success_count": 0,
                    "error_count": 1,
                }
            )
        )
    report_path = tmp_path / "diff.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/github_action_health.py",
            str(tmp_path / "baseline"),
            str(tmp_path / "candidate"),
            "--output",
            str(report_path),
            "--baseline-exit",
            "0",
            "--candidate-exit",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    report = json.loads(report_path.read_text())
    assert set(report["health_failures"]) == {"baseline", "candidate"}


def test_persisted_health_detects_manifest_count_mismatch(tmp_path):
    (tmp_path / "records.jsonl").write_text('{"status":"success"}\n')
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "run_completed": True,
                "record_count": 2,
                "success_count": 2,
                "error_count": 0,
            }
        )
    )
    health = check_run_directory_health(tmp_path)
    assert health["healthy"] is False
    assert "Manifest record_count does not match records" in health["reasons"]


def _record(score=0.5, custom=None):
    return {
        "input": "Question",
        "status": "success",
        "output": "Answer",
        "scores": {"accuracy": score},
        "primary_metric": "accuracy",
        "custom": custom or {},
    }


@pytest.mark.parametrize(
    "candidate",
    [
        _record(score=0.9),
        _record(custom={"trace_fingerprint": "different"}),
        _record(custom={"trace_violations": [{"rule": "new"}]}),
    ],
)
def test_strict_policy_gates_differences_excluded_by_legacy_policy(candidate):
    computation = build_diff_computation(
        records_baseline=[_record()],
        records_candidate=[candidate],
        baseline_label="baseline",
        candidate_label="candidate",
    )
    assert computation.has_differences
    assert compute_diff_exit_code(computation, DiffGatePolicy(fail_on_any_difference=True)) == 2


def test_duplicate_record_identity_is_invalid_even_without_gate_flags():
    with pytest.raises(ValueError, match="Duplicate record identity"):
        build_diff_computation(
            records_baseline=[_record(), _record()],
            records_candidate=[_record()],
            baseline_label="a",
            candidate_label="b",
        )


@pytest.mark.parametrize(
    "candidate",
    [
        {**_record(), "scores": {}, "primary_metric": None},
        {**_record(), "scores": {"other": 0.5}, "primary_metric": "other"},
    ],
)
def test_regression_gate_fails_when_numeric_score_evidence_is_lost(candidate):
    computation = build_diff_computation(
        records_baseline=[_record()],
        records_candidate=[candidate],
        baseline_label="a",
        candidate_label="b",
    )
    assert compute_diff_exit_code(computation) == 0
    assert compute_diff_exit_code(computation, DiffGatePolicy(fail_on_regressions=True)) == 1
    assert compute_diff_exit_code(computation, DiffGatePolicy(fail_on_any_difference=True)) == 2


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("side", ["baseline", "candidate"])
def test_nonfinite_scores_are_invalid_on_either_side(score, side):
    records = {"baseline": [_record()], "candidate": [_record()]}
    records[side][0]["scores"]["accuracy"] = score
    with pytest.raises(ValueError, match="Non-finite score"):
        build_diff_computation(
            records_baseline=records["baseline"],
            records_candidate=records["candidate"],
            baseline_label="a",
            candidate_label="b",
        )


@pytest.mark.parametrize("value", [None, True, "not numeric"])
def test_identical_malformed_primary_evidence_is_invalid(value):
    record = {**_record(), "scores": {"accuracy": value}}
    with pytest.raises(ValueError, match="primary_metric must name a finite numeric score"):
        build_diff_computation(
            records_baseline=[record],
            records_candidate=[record],
            baseline_label="a",
            candidate_label="b",
        )


def test_missing_declared_primary_score_is_invalid():
    record = {**_record(), "scores": {}}
    with pytest.raises(ValueError, match="primary_metric must name a finite numeric score"):
        build_diff_computation(
            records_baseline=[record],
            records_candidate=[record],
            baseline_label="a",
            candidate_label="b",
        )


def test_legacy_null_score_is_invalid_but_explicitly_unscored_records_are_allowed():
    record = {**_record(), "scores": {"score": None}, "primary_metric": None}
    with pytest.raises(ValueError, match="Legacy 'score'"):
        build_diff_computation(
            records_baseline=[record],
            records_candidate=[record],
            baseline_label="a",
            candidate_label="b",
        )
    record["scores"] = {}
    computation = build_diff_computation(
        records_baseline=[record],
        records_candidate=[record],
        baseline_label="a",
        candidate_label="b",
    )
    assert not computation.has_differences


def test_comment_report_excludes_untrusted_text_and_rejects_invalid_counts():
    path = Path(__file__).resolve().parents[1] / "scripts/github_action_pr_report.py"
    spec = importlib.util.spec_from_file_location("pr_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    event = {"pull_request": {"number": 42, "head": {"sha": "a" * 40}}}
    diff = {"error": "@everyone injected", "output": "unsafe", "counts": {"regressions": 1}}
    report = module.build_report(event, diff, 1)
    assert "injected" not in json.dumps(report)
    assert report["counts"]["regressions"] == 1
    with pytest.raises(ValueError, match="Invalid count"):
        module.build_report(event, {"counts": {"regressions": "malicious"}}, 1)
