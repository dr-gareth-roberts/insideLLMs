"""Killer Feature #2: trace-aware diff gate.

Builds synthetic baseline/candidate records with identical outputs but different
trace fingerprints / violation counts, then enforces trace-level gating.

Run:
    python examples/killer_features/feature_2_trace_drift_gate.py
"""

from __future__ import annotations

import sys
import argparse
import shutil
from pathlib import Path

def _record(trace_fp: str, violations: int) -> dict[str, object]:
    return {
        "run_id": "trace-demo",
        "status": "success",
        "primary_metric": "score",
        "scores": {"score": 1.0},
        "input": {"prompt": "Summarize this changelog."},
        "output": {"text": "Summary is stable."},
        "example_id": "0",
        "model": {"model_id": "demo-model"},
        "probe": {"probe_id": "instruction_following"},
        "custom": {
            "trace_fingerprint": trace_fp,
            "trace_violations": [{"rule": "tool_schema"}] * violations,
        },
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from insideLLMs.cli._record_utils import _write_jsonl
    from insideLLMs.cli.commands.diff import cmd_diff

    run_root = repo_root / ".tmp" / "killer_features" / "feature_2"
    baseline = run_root / "baseline"
    candidate = run_root / "candidate"

    if run_root.exists():
        shutil.rmtree(run_root)
    baseline.mkdir(parents=True, exist_ok=True)
    candidate.mkdir(parents=True, exist_ok=True)

    _write_jsonl([_record("sha256:" + "a" * 64, 0)], baseline / "records.jsonl")
    _write_jsonl([_record("sha256:" + "b" * 64, 1)], candidate / "records.jsonl")

    args = argparse.Namespace(
        run_dir_a=str(baseline),
        run_dir_b=str(candidate),
        format="text",
        output=None,
        limit=25,
        fail_on_regressions=False,
        fail_on_changes=False,
        output_fingerprint_ignore=[],
        fail_on_trace_violations=True,
        fail_on_trace_drift=True,
    )

    exit_code = cmd_diff(args)
    if exit_code == 0:
        raise SystemExit("Expected trace gate to fail, but it passed.")

    print(f"Trace gate triggered as expected (exit code {exit_code}).")


if __name__ == "__main__":
    main()
