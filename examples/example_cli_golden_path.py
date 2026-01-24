"""End-to-end CLI workflow (offline + deterministic).

This script runs the same harness twice (baseline/candidate) and diffs the outputs.
It uses `ci/harness.yaml`, which is configured to run entirely offline using `DummyModel`.

Run:
    python examples/example_cli_golden_path.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_cli(repo_root: Path, *args: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "insideLLMs.cli", *args],
        cwd=str(repo_root),
        check=True,
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "ci" / "harness.yaml"

    run_root = repo_root / ".tmp" / "runs"
    baseline_dir = run_root / "golden_path_baseline"
    candidate_dir = run_root / "golden_path_candidate"
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"Config: {config_path}")
    print(f"Baseline: {baseline_dir}")
    print(f"Candidate: {candidate_dir}")

    print("\n[1/4] Run baseline harness")
    _run_cli(
        repo_root,
        "harness",
        str(config_path),
        "--run-dir",
        str(baseline_dir),
        "--overwrite",
        "--skip-report",
    )

    print("\n[2/4] Build baseline report")
    _run_cli(repo_root, "report", str(baseline_dir))

    print("\n[3/4] Run candidate harness")
    _run_cli(
        repo_root,
        "harness",
        str(config_path),
        "--run-dir",
        str(candidate_dir),
        "--overwrite",
        "--skip-report",
    )

    print("\n[4/4] Diff baseline vs candidate (should be identical)")
    _run_cli(
        repo_root,
        "diff",
        str(baseline_dir),
        str(candidate_dir),
        "--fail-on-changes",
    )

    print("\nDone.")
    print(f"- Baseline report: {baseline_dir / 'report.html'}")
    print(f"- Candidate report: {candidate_dir / 'report.html'}")


if __name__ == "__main__":
    main()

