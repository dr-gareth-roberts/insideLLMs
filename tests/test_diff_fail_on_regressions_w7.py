"""Focused proof that `insidellms diff --fail-on-regressions` returns the
correct exit status for both the regression and the no-regression case.

Acceptance criterion (goal.md):
    "Existing or newly focused CLI tests prove `insidellms diff
    --fail-on-regressions` exits nonzero for a genuine regression and
    succeeds when no regression exists."

This file provides four new end-to-end tests that drive the actual CLI entry
point (insideLLMs.cli.main) to prove all exit-status paths.  It supplements
the unit-level tests that already exist in test_cli_diff_harness_coverage.py
and test_cli_diff_engine.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from insideLLMs.cli import main

pytestmark = pytest.mark.determinism


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_record(path: Path, score: float) -> None:
    """Write a single-record JSONL file for a logic probe run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "1.0.1",
        "run_id": "test-run",
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:00:01+00:00",
        "model": {"model_id": "dummy", "provider": "local", "params": {}},
        "probe": {"probe_id": "logic", "probe_version": "1.0.0", "params": {}},
        "example_id": "e1",
        "status": "success",
        "primary_metric": "score",
        "scores": {"score": score},
        "usage": {},
        "custom": {},
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _run_diff_cli(
    tmp_path: Path,
    *,
    baseline_score: float,
    candidate_score: float,
    fail_on_regressions: bool = True,
) -> int:
    """Set up two run directories and invoke `insidellms diff`."""
    dir_a = tmp_path / "baseline"
    dir_b = tmp_path / "candidate"
    _write_record(dir_a / "records.jsonl", baseline_score)
    _write_record(dir_b / "records.jsonl", candidate_score)

    cmd = [
        "diff",
        str(dir_a),
        str(dir_b),
        "--format",
        "json",
    ]
    if fail_on_regressions:
        cmd.append("--fail-on-regressions")

    return main(cmd)


# ---------------------------------------------------------------------------
# Regression case — candidate score drops
# ---------------------------------------------------------------------------


def test_fail_on_regressions_exits_nonzero_when_regression(tmp_path: Path) -> None:
    """`--fail-on-regressions` must return non-zero exit when score drops."""
    rc = _run_diff_cli(tmp_path, baseline_score=0.9, candidate_score=0.6)
    assert rc != 0, f"Expected nonzero exit for a genuine score regression, got {rc}"


# ---------------------------------------------------------------------------
# No-regression case — candidate score equals or improves
# ---------------------------------------------------------------------------


def test_fail_on_regressions_exits_zero_when_no_regression(tmp_path: Path) -> None:
    """With `--fail-on-regressions`, exit 0 when candidate score is same/better."""
    rc = _run_diff_cli(tmp_path, baseline_score=0.8, candidate_score=0.8)
    assert rc == 0, f"Expected exit 0 when no regressions exist, got {rc}"


def test_fail_on_regressions_exits_zero_when_improvement(tmp_path: Path) -> None:
    """With `--fail-on-regressions`, exit 0 when candidate score is better."""
    rc = _run_diff_cli(tmp_path, baseline_score=0.7, candidate_score=0.95)
    assert rc == 0, f"Expected exit 0 when candidate is strictly better, got {rc}"


# ---------------------------------------------------------------------------
# Without the flag — always exit 0 even with regressions
# ---------------------------------------------------------------------------


def test_no_fail_on_regressions_flag_exits_zero_despite_regression(
    tmp_path: Path,
) -> None:
    """Without `--fail-on-regressions`, diff should exit 0 regardless of regressions."""
    rc = _run_diff_cli(
        tmp_path,
        baseline_score=0.9,
        candidate_score=0.5,
        fail_on_regressions=False,
    )
    assert rc == 0, f"Expected exit 0 when --fail-on-regressions is not set, got {rc}"
