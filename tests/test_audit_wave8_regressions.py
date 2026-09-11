"""Regression tests for the first REVIEW_LOOP wave-8 increment."""

from __future__ import annotations

import json
import tracemalloc
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterator

import pytest

from insideLLMs.cli._record_utils import _read_jsonl_records, iter_jsonl_records
from insideLLMs.runtime.diffing import build_diff_computation


def _record(index: int, *, run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "success",
        "custom": {
            "harness": {
                "model_id": "dummy",
                "probe_type": "logic",
                "example_index": index,
            },
            "replicate_key": str(index),
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_iter_jsonl_records_is_lazy_and_skips_non_dict_values(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"id": 1}\nnull\n[2]\n{"id": 2}\n', encoding="utf-8")

    records = iter_jsonl_records(path)

    assert not isinstance(records, list)
    assert next(records) == {"id": 1}
    assert list(records) == [{"id": 2}]


def test_jsonl_readers_preserve_list_compatibility_and_line_errors(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"id": 1}\n', encoding="utf-8")

    assert _read_jsonl_records(path) == [{"id": 1}]

    path.write_text('{"id": 1}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        list(iter_jsonl_records(path))


def test_diff_consumes_single_pass_iterators_and_preserves_run_ids() -> None:
    baseline: Iterator[dict[str, Any]] = (_record(index, run_id="baseline") for index in range(3))
    candidate: Iterator[dict[str, Any]] = (_record(index, run_id="candidate") for index in range(3))

    computation = build_diff_computation(
        records_baseline=baseline,
        records_candidate=candidate,
        baseline_label="baseline",
        candidate_label="candidate",
    )

    assert computation.diff_report["counts"]["common"] == 3
    assert computation.diff_report["run_ids"] == {
        "baseline": ["baseline"],
        "candidate": ["candidate"],
    }


def test_cli_diff_uses_streaming_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from insideLLMs.cli.commands import diff as diff_command

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    _write_jsonl(run_a / "records.jsonl", [_record(0, run_id="run")])
    _write_jsonl(run_b / "records.jsonl", [_record(0, run_id="run")])

    def streaming_reader(path: Path) -> Iterator[dict[str, Any]]:
        with path.open(encoding="utf-8") as records_file:
            for line in records_file:
                if line.strip():
                    yield json.loads(line)

    monkeypatch.setattr(diff_command, "iter_jsonl_records", streaming_reader)
    monkeypatch.setattr(
        diff_command,
        "_read_jsonl_records",
        lambda _path: pytest.fail("diff must not materialize records with the list reader"),
        raising=False,
    )

    args = Namespace(
        run_dir_a=str(run_a),
        run_dir_b=str(run_b),
        format="text",
        output=None,
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
    )

    assert diff_command.cmd_diff(args) == 0


def test_streaming_diff_peak_memory_is_bounded_for_100k_records() -> None:
    tracemalloc.start()
    try:
        computation = build_diff_computation(
            records_baseline=(_record(index, run_id="baseline") for index in range(100_000)),
            records_candidate=(_record(index, run_id="candidate") for index in range(100_000)),
            baseline_label="baseline",
            candidate_label="candidate",
        )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert computation.diff_report["counts"]["common"] == 100_000
    assert peak < 128 * 1024 * 1024
