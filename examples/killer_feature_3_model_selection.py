"""Killer Feature #3: model-selection workflow with statistical reporting.

Runs a harness with multiple model families and emits a statistical report
that can be attached to model selection reviews.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from insideLLMs.results import generate_statistical_report
from insideLLMs.runtime import run_harness_from_config
from insideLLMs.schemas import DEFAULT_SCHEMA_VERSION


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_root = repo_root / ".tmp" / "runs" / "killer-feature-3"
    config_path = run_root / "harness.yaml"
    report_path = run_root / "statistical_report.md"
    run_root.mkdir(parents=True, exist_ok=True)

    config = {
        "models": [
            {
                "type": "dummy",
                "args": {
                    "name": "dummy-safe",
                    "response_prefix": "[safe]",
                },
            },
            {
                "type": "dummy",
                "args": {
                    "name": "dummy-creative",
                    "response_prefix": "[creative]",
                },
            },
            {
                "type": "openrouter",
                "args": {
                    "model_name": "openai/gpt-4o-mini",
                    "timeout": 120,
                },
            },
        ],
        "probes": [
            {"type": "logic", "args": {}},
            {"type": "factuality", "args": {}},
            {"type": "instruction_following", "args": {}},
        ],
        "dataset": {"format": "jsonl", "path": str(repo_root / "examples" / "probe_battery.jsonl")},
        "max_examples": 3,
        "generation": {"temperature": 0.0, "seed": 42, "max_tokens": 256},
        "report_title": "Killer Feature #3: Model Selection",
        "output_dir": str(run_root / "harness-output"),
    }

    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_harness_from_config(
        config_path,
        validate_output=True,
        schema_version=DEFAULT_SCHEMA_VERSION,
        validation_mode="strict",
        strict_serialization=True,
        deterministic_artifacts=True,
    )

    generate_statistical_report(
        result["experiments"],
        output_path=str(report_path),
        format="markdown",
        confidence_level=0.95,
    )

    print(f"Harness run id: {result['run_id']}")
    print(f"Experiments: {len(result['experiments'])}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
