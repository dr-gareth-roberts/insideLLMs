"""Tests for shared CLI diff computation engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from insideLLMs.cli._record_utils import _read_jsonl_records
from insideLLMs.runtime.diffing import (
    DiffGatePolicy,
    build_diff_computation,
    compute_diff_exit_code,
    judge_diff_report,
)
from insideLLMs.runtime.diffing_interactive import (
    build_interactive_review_lines,
    copy_candidate_artifacts_to_baseline,
    prompt_accept_snapshot,
)


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")


def _record(
    *,
    model_id: str = "m1",
    probe_id: str = "p1",
    example_id: str = "e1",
    status: str = "success",
    score: float = 0.9,
    output_text: str | None = None,
    custom: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "schema_version": "1.0.1",
        "run_id": "run-1",
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:00:01+00:00",
        "model": {"model_id": model_id, "provider": "local", "params": {}},
        "probe": {"probe_id": probe_id, "probe_version": "1.0.0", "params": {}},
        "example_id": example_id,
        "status": status,
        "primary_metric": "score",
        "scores": {"score": score},
        "usage": {},
        "custom": custom if custom is not None else {},
    }
    if output_text is not None:
        rec["output_text"] = output_text
    return rec


def test_build_diff_computation_metric_regression_contract(tmp_path: Path) -> None:
    a_path = tmp_path / "a" / "records.jsonl"
    b_path = tmp_path / "b" / "records.jsonl"
    _write_records(a_path, [_record(score=0.9)])
    _write_records(b_path, [_record(score=0.7)])

    computation = build_diff_computation(
        records_baseline=_read_jsonl_records(a_path),
        records_candidate=_read_jsonl_records(b_path),
        baseline_label=str(a_path.parent),
        candidate_label=str(b_path.parent),
    )

    assert computation.diff_report["counts"]["regressions"] == 1
    assert len(computation.regressions) == 1
    assert computation.has_differences is True
    assert computation.diff_report["baseline"] == str(a_path.parent)
    assert computation.diff_report["candidate"] == str(b_path.parent)


def test_build_diff_computation_output_ignore_keys_contract(tmp_path: Path) -> None:
    a_path = tmp_path / "a" / "records.jsonl"
    b_path = tmp_path / "b" / "records.jsonl"
    rec_a = _record()
    rec_b = _record()
    rec_a["output"] = {"data": "same", "ts": "a"}
    rec_b["output"] = {"data": "same", "ts": "b"}
    _write_records(a_path, [rec_a])
    _write_records(b_path, [rec_b])

    without_ignore = build_diff_computation(
        records_baseline=_read_jsonl_records(a_path),
        records_candidate=_read_jsonl_records(b_path),
        baseline_label="a",
        candidate_label="b",
    )
    with_ignore = build_diff_computation(
        records_baseline=_read_jsonl_records(a_path),
        records_candidate=_read_jsonl_records(b_path),
        baseline_label="a",
        candidate_label="b",
        output_fingerprint_ignore=["ts"],
    )

    assert without_ignore.diff_report["counts"]["other_changes"] >= 1
    assert with_ignore.diff_report["counts"]["other_changes"] == 0


def test_build_diff_computation_detects_only_baseline_and_candidate(tmp_path: Path) -> None:
    a_path = tmp_path / "a" / "records.jsonl"
    b_path = tmp_path / "b" / "records.jsonl"
    _write_records(a_path, [_record(model_id="m1", example_id="only-a")])
    _write_records(b_path, [_record(model_id="m2", example_id="only-b")])

    computation = build_diff_computation(
        records_baseline=_read_jsonl_records(a_path),
        records_candidate=_read_jsonl_records(b_path),
        baseline_label="a",
        candidate_label="b",
    )

    assert computation.diff_report["counts"]["only_baseline"] == 1
    assert computation.diff_report["counts"]["only_candidate"] == 1
    assert len(computation.only_baseline) == 1
    assert len(computation.only_candidate) == 1


def test_build_diff_computation_validates_diff_schema_when_enabled(tmp_path: Path) -> None:
    a_path = tmp_path / "a" / "records.jsonl"
    b_path = tmp_path / "b" / "records.jsonl"
    rec_a = _record(output_text="hello", custom={"trace_violations": []})
    rec_b = _record(output_text="world", custom={"trace_violations": [{"rule": "x"}]})
    _write_records(a_path, [rec_a])
    _write_records(b_path, [rec_b])

    computation = build_diff_computation(
        records_baseline=_read_jsonl_records(a_path),
        records_candidate=_read_jsonl_records(b_path),
        baseline_label="a",
        candidate_label="b",
        validate_output=True,
        validation_mode="strict",
    )

    assert computation.diff_report["counts"]["trace_violation_increases"] == 1


def test_build_diff_computation_detects_trajectory_drifts(tmp_path: Path) -> None:
    a_path = tmp_path / "a" / "records.jsonl"
    b_path = tmp_path / "b" / "records.jsonl"
    rec_a = _record(output_text="same")
    rec_b = _record(output_text="same")
    rec_a["custom"] = {
        "trace": {
            "events": [
                {
                    "seq": 0,
                    "kind": "tool_call_start",
                    "payload": {"tool_name": "search", "arguments": {"query": "alpha"}},
                }
            ]
        }
    }
    rec_b["custom"] = {
        "trace": {
            "events": [
                {
                    "seq": 0,
                    "kind": "tool_call_start",
                    "payload": {"tool_name": "search", "arguments": {"query": "beta"}},
                }
            ]
        }
    }
    _write_records(a_path, [rec_a])
    _write_records(b_path, [rec_b])

    computation = build_diff_computation(
        records_baseline=_read_jsonl_records(a_path),
        records_candidate=_read_jsonl_records(b_path),
        baseline_label="a",
        candidate_label="b",
    )

    assert computation.diff_report["counts"]["trajectory_drifts"] == 1
    assert len(computation.trajectory_drifts) == 1
    assert computation.diff_report["trajectory_drifts"][0]["kind"] == "trajectory_drift"


def test_judge_diff_report_balanced_policy_marks_changes_for_review(tmp_path: Path) -> None:
    a_path = tmp_path / "a" / "records.jsonl"
    b_path = tmp_path / "b" / "records.jsonl"
    _write_records(a_path, [_record(output_text="hello")])
    _write_records(b_path, [_record(output_text="world")])

    computation = build_diff_computation(
        records_baseline=_read_jsonl_records(a_path),
        records_candidate=_read_jsonl_records(b_path),
        baseline_label="a",
        candidate_label="b",
    )

    judged = judge_diff_report(computation.diff_report, policy="balanced")
    assert judged.breaking is False
    assert judged.review_count >= 1


def test_runtime_package_re_exports_diffing_public_api() -> None:
    import insideLLMs.runtime as runtime

    assert callable(runtime.build_diff_computation)
    assert callable(runtime.judge_diff_report)
    assert callable(runtime.compute_diff_exit_code)
    assert runtime.DiffGatePolicy is not None


def test_public_diffing_facade_exports_symbols() -> None:
    import insideLLMs.diffing as public_diffing

    assert callable(public_diffing.build_diff_computation)
    assert callable(public_diffing.compute_diff_exit_code)
    assert callable(public_diffing.print_interactive_review)
    assert public_diffing.DiffGatePolicy is not None


def test_compute_diff_exit_code_contract(tmp_path: Path) -> None:
    reg_base = tmp_path / "reg_base" / "records.jsonl"
    reg_head = tmp_path / "reg_head" / "records.jsonl"
    _write_records(reg_base, [_record(output_text="hello", score=0.9)])
    _write_records(reg_head, [_record(output_text="world", score=0.7)])
    regression = build_diff_computation(
        records_baseline=_read_jsonl_records(reg_base),
        records_candidate=_read_jsonl_records(reg_head),
        baseline_label="reg_base",
        candidate_label="reg_head",
    )

    trace_v_base = tmp_path / "trace_v_base" / "records.jsonl"
    trace_v_head = tmp_path / "trace_v_head" / "records.jsonl"
    _write_records(
        trace_v_base,
        [_record(output_text="same", score=0.9, custom={"trace_violations": []})],
    )
    _write_records(
        trace_v_head,
        [_record(output_text="same", score=0.9, custom={"trace_violations": [{"rule": "r1"}]})],
    )
    trace_violations = build_diff_computation(
        records_baseline=_read_jsonl_records(trace_v_base),
        records_candidate=_read_jsonl_records(trace_v_head),
        baseline_label="trace_v_base",
        candidate_label="trace_v_head",
    )

    trace_d_base = tmp_path / "trace_d_base" / "records.jsonl"
    trace_d_head = tmp_path / "trace_d_head" / "records.jsonl"
    _write_records(
        trace_d_base,
        [_record(output_text="same", score=0.9, custom={"trace_fingerprint": "sha256:aaaa"})],
    )
    _write_records(
        trace_d_head,
        [_record(output_text="same", score=0.9, custom={"trace_fingerprint": "sha256:bbbb"})],
    )
    trace_drift = build_diff_computation(
        records_baseline=_read_jsonl_records(trace_d_base),
        records_candidate=_read_jsonl_records(trace_d_head),
        baseline_label="trace_d_base",
        candidate_label="trace_d_head",
    )

    traj_base = tmp_path / "traj_base" / "records.jsonl"
    traj_head = tmp_path / "traj_head" / "records.jsonl"
    _write_records(
        traj_base,
        [
            _record(
                output_text="same",
                score=0.9,
                custom={
                    "trace": {
                        "events": [
                            {
                                "seq": 0,
                                "kind": "tool_call_start",
                                "payload": {"tool_name": "search", "arguments": {"query": "alpha"}},
                            }
                        ]
                    }
                },
            )
        ],
    )
    _write_records(
        traj_head,
        [
            _record(
                output_text="same",
                score=0.9,
                custom={
                    "trace": {
                        "events": [
                            {
                                "seq": 0,
                                "kind": "tool_call_start",
                                "payload": {"tool_name": "search", "arguments": {"query": "beta"}},
                            }
                        ]
                    }
                },
            )
        ],
    )
    trajectory_drift = build_diff_computation(
        records_baseline=_read_jsonl_records(traj_base),
        records_candidate=_read_jsonl_records(traj_head),
        baseline_label="traj_base",
        candidate_label="traj_head",
    )

    assert compute_diff_exit_code(regression) == 0
    assert compute_diff_exit_code(regression, DiffGatePolicy(fail_on_regressions=True)) == 2
    assert compute_diff_exit_code(regression, DiffGatePolicy(fail_on_changes=True)) == 2
    assert (
        compute_diff_exit_code(trace_violations, DiffGatePolicy(fail_on_trace_violations=True)) == 3
    )
    assert compute_diff_exit_code(trace_drift, DiffGatePolicy(fail_on_trace_drift=True)) == 4
    assert (
        compute_diff_exit_code(trajectory_drift, DiffGatePolicy(fail_on_trajectory_drift=True)) == 5
    )


def test_interactive_helpers_contract(tmp_path: Path) -> None:
    lines = build_interactive_review_lines(
        {
            "regressions": [
                {
                    "model_id": "m1",
                    "probe_id": "p1",
                    "example_id": "e1",
                    "kind": "output_changed",
                    "baseline": {"status": "success", "output": {"preview": "a"}},
                    "candidate": {"status": "success", "output": {"preview": "b"}},
                }
            ],
            "changes": [],
            "only_baseline": [],
            "only_candidate": [],
        },
        limit=5,
    )
    assert any("example e1" in line for line in lines)

    assert prompt_accept_snapshot(input_func=lambda _prompt: "y") is True
    assert prompt_accept_snapshot(input_func=lambda _prompt: "n") is False

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (candidate / "records.jsonl").write_text("{}", encoding="utf-8")
    copied = copy_candidate_artifacts_to_baseline(baseline, candidate)
    assert "records.jsonl" in copied
    assert (baseline / "records.jsonl").exists()
