import json
import os
import subprocess
import sys
from pathlib import Path

from insideLLMs.runner import _deterministic_base_time, _deterministic_item_times
from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION


def _write_harness_records(run_dir: Path, run_id: str, outputs: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    base_time = _deterministic_base_time(run_id)

    records = []
    for index, output_text in enumerate(outputs):
        started_at, completed_at = _deterministic_item_times(base_time, index)
        records.append(
            {
                "schema_version": DEFAULT_SCHEMA_VERSION,
                "run_id": run_id,
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "model": {"model_id": "dummy-1", "provider": "dummy", "params": {}},
                "probe": {"probe_id": "logic", "probe_version": None, "params": {}},
                "dataset": {
                    "dataset_id": "canary-ds",
                    "dataset_version": None,
                    "dataset_hash": None,
                    "provenance": "jsonl",
                    "params": {},
                },
                "example_id": str(index),
                "input": {"question": f"Q{index}?"},
                "output": output_text,
                "output_text": output_text,
                "scores": {"score": 1.0},
                "primary_metric": "score",
                "usage": {},
                "latency_ms": 1.0,
                "status": "success",
                "error": None,
                "error_type": None,
                "custom": {
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
                    }
                },
            }
        )

    records_path = run_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _run_cli(args: list[str], env: dict[str, str], cwd: Path) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "insideLLMs.cli", *args],
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
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
    _write_harness_records(candidate_dir, run_id, ["A", "C"])

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
