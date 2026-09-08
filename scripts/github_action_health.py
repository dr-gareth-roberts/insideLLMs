"""Fail CI before diffing if either persisted run is unhealthy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from insideLLMs.runtime._run_health import check_run_directory_health


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-exit", required=True, type=int)
    parser.add_argument("--candidate-exit", required=True, type=int)
    args = parser.parse_args()
    failures: dict[str, list[str]] = {}
    for label, directory, exit_code in (
        ("baseline", args.baseline, args.baseline_exit),
        ("candidate", args.candidate, args.candidate_exit),
    ):
        reasons = []
        if exit_code:
            reasons.append(f"Harness exited with code {exit_code}")
        try:
            reasons.extend(check_run_directory_health(directory)["reasons"])
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            reasons.append(f"Invalid or missing run artifacts: {exc}")
        if reasons:
            failures[label] = reasons
    if failures:
        args.output.write_text(
            json.dumps({"error": "Run health check failed", "health_failures": failures}, indent=2),
            encoding="utf-8",
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
