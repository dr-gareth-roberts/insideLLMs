"""Killer Feature #3: multi-model tournament + statistical report.

Creates a harness with multiple model families (OpenRouter via OpenAI-compatible
API when available, otherwise DummyModel fallback), runs probes, and emits a
statistical report for decision-making.

Run (OpenRouter):
    export OPENROUTER_API_KEY="..."
    export OPENROUTER_MODELS="openai/gpt-4o-mini,anthropic/claude-3.5-sonnet"
    python examples/killer_features/feature_3_model_tournament.py

Run (offline fallback):
    python examples/killer_features/feature_3_model_tournament.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import yaml

def _model_matrix() -> list[dict[str, object]]:
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_models = os.getenv("OPENROUTER_MODELS", "").replace(",", " ").split()

    if openrouter_key and openrouter_models:
        return [
            {
                "type": "openrouter",
                "args": {
                    "model_name": model_name,
                    "timeout": 600,
                },
            }
            for model_name in openrouter_models
        ]

    # Offline-safe fallback for local development / CI demonstration.
    return [
        {
            "type": "dummy",
            "args": {"name": "Dummy-Strict", "canned_response": "I cannot comply with unsafe requests."},
        },
        {
            "type": "dummy",
            "args": {"name": "Dummy-Echo", "response_prefix": "[EchoModel]"},
        },
    ]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from insideLLMs.results import generate_statistical_report
    from insideLLMs.runtime import run_harness_from_config
    from insideLLMs.schemas import DEFAULT_SCHEMA_VERSION

    run_root = repo_root / ".tmp" / "killer_features" / "feature_3"
    cfg_path = run_root / "harness.yaml"
    run_root.mkdir(parents=True, exist_ok=True)

    config = {
        "models": _model_matrix(),
        "probes": [
            {"type": "logic", "args": {}},
            {"type": "factuality", "args": {}},
            {"type": "prompt_injection", "args": {}},
            {"type": "instruction_following", "args": {}},
            {"type": "code_generation", "args": {"language": "python"}},
        ],
        "dataset": {"format": "jsonl", "path": str((repo_root / "examples" / "probe_battery.jsonl").resolve())},
        "max_examples": 3,
        "generation": {"temperature": 0.0, "seed": 42, "max_tokens": 512},
        "report_title": "Killer Feature #3 - Model Tournament",
        "output_dir": str(run_root / "run"),
    }
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_harness_from_config(
        cfg_path,
        validate_output=True,
        schema_version=DEFAULT_SCHEMA_VERSION,
        validation_mode="strict",
        strict_serialization=True,
        deterministic_artifacts=True,
    )

    report_path = run_root / "statistical_report.md"
    generate_statistical_report(
        result["experiments"],
        output_path=str(report_path),
        format="markdown",
        confidence_level=0.95,
    )

    print(f"experiments: {len(result['experiments'])}")
    print(f"records: {len(result['records'])}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
