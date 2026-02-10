"""Advanced OpenRouter harness workflow with schema validation, stats, and diffing.

This example demonstrates:
  - OpenRouter OpenAI-compatible model support (via OPENROUTER_API_KEY)
  - Programmatic harness execution with schema validation + deterministic artifacts
  - Statistical report generation from experiment results
  - Diffing run artifacts to detect regressions

Run:
    export OPENROUTER_API_KEY="..."
    export OPENROUTER_MODELS="openai/gpt-4o-mini,anthropic/claude-3.5-sonnet"
    python examples/example_openrouter_advanced.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import yaml

from insideLLMs.cli.commands.diff import cmd_diff
from insideLLMs.cli._record_utils import _write_jsonl
from insideLLMs.results import generate_statistical_report
from insideLLMs.runtime import run_harness_from_config
from insideLLMs.schemas import DEFAULT_SCHEMA_VERSION


def _split_models(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def _probe_battery() -> list[dict[str, object]]:
    return [
        {"type": "logic", "args": {}},
        {"type": "factuality", "args": {}},
        {"type": "prompt_injection", "args": {}},
        {"type": "instruction_following", "args": {}},
        {"type": "code_generation", "args": {"language": "python"}},
    ]


def _write_config(
    path: Path,
    *,
    models: Iterable[str],
    dataset_path: Path,
    output_dir: Path,
) -> None:
    config = {
        "models": [
            {
                "type": "openrouter",
                "args": {
                    "model_name": model_name,
                    "timeout": 600,
                },
            }
            for model_name in models
        ],
        "probes": _probe_battery(),
        "dataset": {"format": "jsonl", "path": str(dataset_path)},
        "max_examples": 3,
        "generation": {"temperature": 0.0, "seed": 42, "max_tokens": 512},
        "report_title": "OpenRouter Advanced Harness",
        "output_dir": str(output_dir),
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def _write_records(run_dir: Path, records: list[dict[str, object]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(records, run_dir / "records.jsonl", strict_serialization=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        help="Comma- or space-separated list of OpenRouter models. Defaults to OPENROUTER_MODELS.",
    )
    parser.add_argument(
        "--run-root",
        default=".tmp/runs/openrouter",
        help="Root directory for baseline/candidate runs.",
    )
    parser.add_argument(
        "--dataset",
        default="examples/probe_battery.jsonl",
        help="Path to the probe battery dataset.",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("Set OPENROUTER_API_KEY before running this example.")

    model_source = args.models or os.environ.get("OPENROUTER_MODELS")
    if not model_source:
        raise RuntimeError("Set OPENROUTER_MODELS or pass --models to run this example.")

    models = _split_models(model_source)
    if not models:
        raise RuntimeError("No models provided after parsing OPENROUTER_MODELS.")

    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (repo_root / dataset_path).resolve()
    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = (repo_root / run_root).resolve()

    baseline_dir = run_root / "baseline"
    candidate_dir = run_root / "candidate"
    baseline_config = run_root / "harness_baseline.yaml"
    candidate_config = run_root / "harness_candidate.yaml"

    _write_config(
        baseline_config,
        models=models,
        dataset_path=dataset_path,
        output_dir=baseline_dir,
    )
    _write_config(
        candidate_config,
        models=models,
        dataset_path=dataset_path,
        output_dir=candidate_dir,
    )

    baseline_result = run_harness_from_config(
        baseline_config,
        validate_output=True,
        schema_version=DEFAULT_SCHEMA_VERSION,
        validation_mode="strict",
        strict_serialization=True,
        deterministic_artifacts=True,
    )
    candidate_result = run_harness_from_config(
        candidate_config,
        validate_output=True,
        schema_version=DEFAULT_SCHEMA_VERSION,
        validation_mode="strict",
        strict_serialization=True,
        deterministic_artifacts=True,
    )

    _write_records(baseline_dir, baseline_result["records"])
    _write_records(candidate_dir, candidate_result["records"])

    report_path = run_root / "statistical_report.md"
    generate_statistical_report(
        baseline_result["experiments"],
        output_path=str(report_path),
        format="markdown",
        confidence_level=0.95,
    )

    diff_args = argparse.Namespace(
        run_dir_a=str(baseline_dir),
        run_dir_b=str(candidate_dir),
        format="text",
        output=None,
        limit=25,
        fail_on_regressions=False,
        fail_on_changes=True,
        output_fingerprint_ignore=[],
        fail_on_trace_violations=False,
        fail_on_trace_drift=False,
    )
    exit_code = cmd_diff(diff_args)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
