"""Killer Feature #2: trace-aware diff guardrails.

Creates a baseline run, clones it, injects synthetic trace drift/violations,
then uses diff guardrails to fail on trace integrity changes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from insideLLMs._serialization import stable_json_dumps
from insideLLMs.runtime import diff_run_dirs, run_harness_to_dir


def _run_harness(config: Path, run_dir: Path) -> int:
    return run_harness_to_dir(config, run_dir, track_project="killer-feature-2")


def _inject_trace_drift(records_path: Path) -> None:
    lines = records_path.read_text(encoding="utf-8").splitlines()
    mutated: list[str] = []
    for idx, line in enumerate(lines):
        record = json.loads(line)
        custom = record.get("custom") if isinstance(record.get("custom"), dict) else {}
        trace = custom.get("trace") if isinstance(custom.get("trace"), dict) else {}
        trace["fingerprint"] = {"value": f"sha256:{idx:064x}"}
        trace["violations"] = [{"rule": "tool_policy", "detail": "synthetic violation"}]
        custom["trace"] = trace
        record["custom"] = custom
        mutated.append(stable_json_dumps(record, strict=True))
    records_path.write_text("\n".join(mutated) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = repo_root / "ci" / "harness.yaml"
    run_root = repo_root / ".tmp" / "runs" / "killer-feature-2"
    baseline = run_root / "baseline"
    candidate = run_root / "candidate"

    if _run_harness(config, baseline) != 0:
        raise SystemExit(1)

    if candidate.exists():
        shutil.rmtree(candidate)
    shutil.copytree(baseline, candidate)
    _inject_trace_drift(candidate / "records.jsonl")

    raise SystemExit(
        diff_run_dirs(
            baseline,
            candidate,
            fail_on_changes=False,
            fail_on_trace_violations=True,
            fail_on_trace_drift=True,
        )
    )


if __name__ == "__main__":
    main()
