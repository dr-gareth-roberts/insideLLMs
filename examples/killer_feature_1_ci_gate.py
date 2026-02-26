"""Killer Feature #1: deterministic behavioral CI gate.

Runs the same harness twice and fails if any behavior changed.
"""

from __future__ import annotations

from pathlib import Path

from insideLLMs.runtime import diff_run_dirs, run_harness_to_dir


def _run_harness(config: Path, run_dir: Path) -> int:
    return run_harness_to_dir(config, run_dir, track_project="killer-feature-1")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = repo_root / "ci" / "harness.yaml"
    run_root = repo_root / ".tmp" / "runs" / "killer-feature-1"
    baseline = run_root / "baseline"
    candidate = run_root / "candidate"

    if _run_harness(config, baseline) != 0:
        raise SystemExit(1)
    if _run_harness(config, candidate) != 0:
        raise SystemExit(1)

    raise SystemExit(diff_run_dirs(baseline, candidate, fail_on_changes=True))


if __name__ == "__main__":
    main()
