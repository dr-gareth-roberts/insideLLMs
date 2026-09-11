"""Create a small allowlisted report for a separate trusted comment workflow."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

COUNT_FIELDS = (
    "common",
    "regressions",
    "improvements",
    "other_changes",
    "only_baseline",
    "only_candidate",
    "trace_drifts",
    "trace_violation_increases",
    "trajectory_drifts",
)


def build_report(event: dict[str, Any], diff: dict[str, Any], exit_code: int) -> dict[str, Any]:
    """Exclude all model output, error text, labels, URLs, and paths."""
    pr = event.get("pull_request") or {}
    number = pr.get("number")
    sha = (pr.get("head") or {}).get("sha")
    if type(number) is not int or number <= 0:
        raise ValueError("Pull request number is required")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise ValueError("Pull request head SHA is required")
    raw_counts = diff.get("counts") or {}
    counts = {}
    for field in COUNT_FIELDS:
        value = raw_counts.get(field, 0)
        if type(value) is not int or not 0 <= value <= 1_000_000_000:
            raise ValueError(f"Invalid count for {field}")
        counts[field] = value
    return {"pr_number": number, "head_sha": sha, "exit_code": exit_code, "counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True, type=Path)
    parser.add_argument("--diff", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exit-code", required=True, type=int, choices=range(6))
    args = parser.parse_args()
    event = json.loads(args.event.read_text(encoding="utf-8"))
    if not event.get("pull_request"):
        return 0
    report = build_report(event, json.loads(args.diff.read_text(encoding="utf-8")), args.exit_code)
    args.output.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
