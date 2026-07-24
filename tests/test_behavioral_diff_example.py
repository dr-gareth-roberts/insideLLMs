"""Smoke test for the offline behavioural-diff walkthrough."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples" / "diff"


def _run_cli(*args: str, expected_code: int = 0) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env.pop("FORCE_COLOR", None)
    result = subprocess.run(
        [sys.executable, "-m", "insideLLMs.cli", "--no-color", *args],
        cwd=REPOSITORY_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == expected_code, result.stderr or result.stdout
    return result


def test_behavioral_diff_walkthrough(tmp_path: Path) -> None:
    """The documented baseline, candidate, gate, and JSON flow all execute."""
    baseline_run = tmp_path / "baseline"
    candidate_run = tmp_path / "candidate"
    json_report = tmp_path / "diff.json"

    _run_cli("run", str(EXAMPLE_ROOT / "baseline.yaml"), "--run-dir", str(baseline_run))
    _run_cli("run", str(EXAMPLE_ROOT / "candidate.yaml"), "--run-dir", str(candidate_run))

    report = _run_cli("diff", str(baseline_run), str(candidate_run))
    assert "change" in report.stdout.lower()

    _run_cli(
        "diff",
        str(baseline_run),
        str(candidate_run),
        "--fail-on-changes",
        expected_code=2,
    )
    _run_cli(
        "diff",
        str(baseline_run),
        str(candidate_run),
        "--format",
        "json",
        "--output",
        str(json_report),
    )

    payload = json.loads(json_report.read_text(encoding="utf-8"))
    assert payload["counts"]["other_changes"] == 1
    assert len(payload["changes"]) == 1
    assert payload["changes"][0]["kind"] == "output_changed"
