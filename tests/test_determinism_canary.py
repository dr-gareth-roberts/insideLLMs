import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from insideLLMs.runtime.runner import (
    _deterministic_base_time,
    _deterministic_item_times,
    _fingerprint_value,
    _replicate_key,
)
from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION

pytestmark = pytest.mark.determinism


def _write_harness_records(
    run_dir: Path,
    run_id: str,
    outputs: list[Any],
    *,
    inputs: list[dict[str, Any]] | None = None,
    example_ids: list[str] | None = None,
    scores: list[float] | None = None,
    scores_payloads: list[dict[str, Any]] | None = None,
    primary_metrics: list[str | None] | None = None,
    statuses: list[str] | None = None,
    errors: list[str | None] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    base_time = _deterministic_base_time(run_id)

    records = []
    score_values = scores or [1.0] * len(outputs)
    status_values = statuses or ["success"] * len(outputs)
    error_values = errors or [None] * len(outputs)
    for index, output_text in enumerate(outputs):
        score = score_values[index] if index < len(score_values) else score_values[-1]
        status = status_values[index] if index < len(status_values) else status_values[-1]
        error = error_values[index] if index < len(error_values) else error_values[-1]
        started_at, completed_at = _deterministic_item_times(base_time, index)
        input_item = inputs[index] if inputs and index < len(inputs) else {"question": f"Q{index}?"}
        example_id = example_ids[index] if example_ids and index < len(example_ids) else str(index)
        input_hash = _fingerprint_value(input_item)
        model_spec = {"model_id": "dummy-1", "provider": "dummy", "params": {}}
        probe_spec = {"probe_id": "logic", "probe_version": None, "params": {}}
        dataset_spec = {
            "dataset_id": "canary-ds",
            "dataset_version": None,
            "dataset_hash": None,
            "provenance": "jsonl",
            "params": {},
        }
        replicate_key = _replicate_key(
            model_spec=model_spec,
            probe_spec=probe_spec,
            dataset_spec=dataset_spec,
            example_id=example_id,
            record_index=index,
            input_hash=input_hash,
        )
        output_value = output_text
        output_text_value = output_text if isinstance(output_text, str) else None

        primary_metric = (
            primary_metrics[index] if primary_metrics and index < len(primary_metrics) else "score"
        )
        score_payload = (
            scores_payloads[index]
            if scores_payloads and index < len(scores_payloads)
            else {"score": score}
        )

        records.append(
            {
                "schema_version": DEFAULT_SCHEMA_VERSION,
                "run_id": run_id,
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "model": model_spec,
                "probe": probe_spec,
                "dataset": dataset_spec,
                "example_id": example_id,
                "input": input_item,
                "output": output_value,
                "output_text": output_text_value,
                "scores": score_payload,
                "primary_metric": primary_metric,
                "usage": {},
                "latency_ms": 1.0,
                "status": status,
                "error": error,
                "error_type": "RuntimeError" if error else None,
                "custom": {
                    "replicate_key": replicate_key,
                    "record_index": index,
                    "harness": {
                        "experiment_id": "exp-canary",
                        "model_type": "dummy",
                        "model_name": "DummyModel",
                        "model_id": "dummy-1",
                        "probe_type": "logic",
                        "probe_name": "LogicProbe",
                        "probe_category": "logic",
                        "dataset": "canary-ds",
                        "dataset_format": "jsonl",
                        "example_index": index,
                    },
                },
            }
        )

    records_path = run_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _run_cli(
    args: list[str],
    env: dict[str, str],
    cwd: Path,
    *,
    expected_code: int = 0,
) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "insideLLMs.cli", *args],
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == expected_code, result.stderr
    return result.stdout


def _seeded_env(seed: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = seed
    return env


def test_report_hash_seed_determinism(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_id = "canary-harness"

    run_dir_a = tmp_path / "report_seed0"
    run_dir_b = tmp_path / "report_seed1"
    _write_harness_records(run_dir_a, run_id, ["A", "B"])
    _write_harness_records(run_dir_b, run_id, ["A", "B"])

    _run_cli(["report", str(run_dir_a)], _seeded_env("0"), repo_root)
    _run_cli(["report", str(run_dir_b)], _seeded_env("1"), repo_root)

    summary_a = (run_dir_a / "summary.json").read_text(encoding="utf-8")
    summary_b = (run_dir_b / "summary.json").read_text(encoding="utf-8")
    report_a = (run_dir_a / "report.html").read_text(encoding="utf-8")
    report_b = (run_dir_b / "report.html").read_text(encoding="utf-8")

    assert summary_a == summary_b
    assert report_a == report_b


def test_diff_hash_seed_determinism(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_id = "canary-harness"

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_harness_records(baseline_dir, run_id, ["A", "B"])
    _write_harness_records(
        candidate_dir,
        run_id,
        ["A", "C"],
        statuses=["success", "error"],
        errors=[None, "simulated failure"],
    )

    out_a = _run_cli(
        ["diff", str(baseline_dir), str(candidate_dir)],
        _seeded_env("0"),
        repo_root,
    )
    out_b = _run_cli(
        ["diff", str(baseline_dir), str(candidate_dir)],
        _seeded_env("1"),
        repo_root,
    )

    assert out_a == out_b


def test_diff_fail_on_regressions_exit_code(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_id = "canary-harness"

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_harness_records(baseline_dir, run_id, ["A", "B"])
    _write_harness_records(
        candidate_dir,
        run_id,
        ["A", "C"],
        statuses=["success", "error"],
        errors=[None, "simulated failure"],
    )

    _run_cli(
        ["diff", str(baseline_dir), str(candidate_dir), "--fail-on-regressions"],
        _seeded_env("0"),
        repo_root,
        expected_code=2,
    )


def test_diff_handles_replicates(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_id = "canary-harness"

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    repeated_inputs = [{"question": "Same prompt"}] * 2

    _write_harness_records(baseline_dir, run_id, ["A", "A"], inputs=repeated_inputs)
    _write_harness_records(candidate_dir, run_id, ["A", "B"], inputs=repeated_inputs)

    output = _run_cli(
        ["diff", str(baseline_dir), str(candidate_dir)],
        _seeded_env("0"),
        repo_root,
    )

    assert "Common keys: 2" in output


def test_diff_ignores_output_keys(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_id = "canary-harness"

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"

    baseline_output = {"answer": "A", "timestamp": "2024-01-01T00:00:00Z"}
    candidate_output = {"answer": "A", "timestamp": "2024-01-02T00:00:00Z"}

    _write_harness_records(baseline_dir, run_id, [baseline_output])
    _write_harness_records(candidate_dir, run_id, [candidate_output])

    output = _run_cli(
        [
            "diff",
            str(baseline_dir),
            str(candidate_dir),
            "--output-fingerprint-ignore",
            "timestamp",
        ],
        _seeded_env("0"),
        repo_root,
    )

    assert "Other changes: 0" in output
    assert "output changed" not in output


def test_diff_metric_mismatch_reason(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_id = "canary-harness"

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"

    _write_harness_records(
        baseline_dir,
        run_id,
        ["same"],
        scores_payloads=[{"score": 1.0}],
        primary_metrics=["score"],
    )
    _write_harness_records(
        candidate_dir,
        run_id,
        ["same"],
        scores_payloads=[{"accuracy": 0.9}],
        primary_metrics=["accuracy"],
    )

    output = _run_cli(
        ["diff", str(baseline_dir), str(candidate_dir)],
        _seeded_env("0"),
        repo_root,
    )

    assert "metrics not comparable" in output
    assert "primary_metric_mismatch" in output
    assert "baseline.primary_metric='score'" in output
    assert "candidate.primary_metric='accuracy'" in output
    assert "score" in output
    assert "accuracy" in output


def test_diff_missing_metric_key(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_id = "canary-harness"

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"

    _write_harness_records(
        baseline_dir,
        run_id,
        ["same"],
        scores_payloads=[{"score": 1.0, "latency_ms": 10.0}],
        primary_metrics=["score"],
    )
    _write_harness_records(
        candidate_dir,
        run_id,
        ["same"],
        scores_payloads=[{"score": 1.0}],
        primary_metrics=["score"],
    )

    output = _run_cli(
        ["diff", str(baseline_dir), str(candidate_dir)],
        _seeded_env("0"),
        repo_root,
    )

    assert "metric_key_missing" in output
    assert "latency_ms" in output


def test_diff_json_output(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_id = "canary-harness"

    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"

    _write_harness_records(
        baseline_dir,
        run_id,
        ["same"],
        scores_payloads=[{"score": 1.0}],
        primary_metrics=["score"],
    )
    _write_harness_records(
        candidate_dir,
        run_id,
        ["same"],
        scores_payloads=[{"accuracy": 0.9}],
        primary_metrics=["accuracy"],
    )

    output = _run_cli(
        ["diff", str(baseline_dir), str(candidate_dir), "--format", "json"],
        _seeded_env("0"),
        repo_root,
    )

    payload = json.loads(output)
    assert payload["counts"]["other_changes"] == 1
    assert payload["changes"][0]["kind"] == "metrics_not_comparable"
    assert payload["changes"][0]["reason"] == "primary_metric_mismatch"
