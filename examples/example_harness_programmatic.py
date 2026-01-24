"""Run a harness programmatically (no CLI).

This shows how to use `insideLLMs.runtime.runner.run_harness_from_config` directly.
It runs the offline CI harness config and prints a small summary.

Run:
    python examples/example_harness_programmatic.py
"""

from __future__ import annotations

from pathlib import Path

from insideLLMs.runtime.runner import run_harness_from_config


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "ci" / "harness.yaml"

    result = run_harness_from_config(config_path)

    print(f"run_id: {result.get('run_id')}")
    print(f"records: {len(result.get('records', []))}")
    print(f"experiments: {len(result.get('experiments', []))}")

    summary = result.get("summary") or {}
    if isinstance(summary, dict):
        print("summary keys:", ", ".join(sorted(summary.keys())))


if __name__ == "__main__":
    main()
