"""Killer Feature #1: deterministic behavioural CI gate.

Runs the same harness twice, then fails if any behaviour changed.
This is the core production safety workflow many teams want in CI.

Run:
    python examples/killer_features/feature_1_ci_behavior_gate.py
"""

from __future__ import annotations

import sys
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    run_root = repo_root / ".tmp" / "killer_features" / "feature_1"
    baseline = run_root / "baseline"
    candidate = run_root / "candidate"

    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    harness_cfg = repo_root / "ci" / "harness.yaml"

    run([
        "python",
        "-m",
        "insideLLMs.cli",
        "harness",
        str(harness_cfg),
        "--run-dir",
        str(baseline),
        "--overwrite",
    ])

    run([
        "python",
        "-m",
        "insideLLMs.cli",
        "harness",
        str(harness_cfg),
        "--run-dir",
        str(candidate),
        "--overwrite",
    ])

    run([
        "python",
        "-m",
        "insideLLMs.cli",
        "diff",
        str(baseline),
        str(candidate),
        "--fail-on-changes",
    ])


if __name__ == "__main__":
    main()
