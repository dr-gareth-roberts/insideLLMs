"""Run an Ollama Cloud harness twice and diff the artifacts programmatically.

This example builds a harness config for all Ollama Cloud models listed in
`OLLAMA_CLOUD_MODELS`, runs the full probe battery twice, and then diffs the
baseline vs candidate run directories.

Run:
    export OLLAMA_API_KEY="..."
    export OLLAMA_CLOUD_MODELS="llama3.1:cloud,qwen2.5:cloud,mistral:cloud"
    python examples/example_ollama_cloud_harness_diff.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import yaml

from insideLLMs.cli.commands.diff import cmd_diff
from insideLLMs.cli.commands.harness import cmd_harness
from insideLLMs.schemas import DEFAULT_SCHEMA_VERSION


def _split_models(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def _probe_battery() -> list[dict[str, object]]:
    return [
        {"type": "logic", "args": {}},
        {"type": "factuality", "args": {}},
        {"type": "bias", "args": {"bias_dimension": "gender"}},
        {"type": "prompt_injection", "args": {}},
        {"type": "jailbreak", "args": {}},
        {"type": "attack", "args": {"attack_type": "general"}},
        {"type": "instruction_following", "args": {}},
        {"type": "constraint_compliance", "args": {"constraint_type": "word_limit", "limit": 80}},
        {"type": "multi_step_task", "args": {}},
        {"type": "code_generation", "args": {"language": "python"}},
        {"type": "code_explanation", "args": {"detail_level": "medium"}},
        {"type": "code_debug", "args": {"language": "python"}},
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
                "type": "ollama",
                "args": {
                    "model_name": model_name,
                    "base_url": "https://ollama.com",
                    "timeout": 600,
                },
            }
            for model_name in models
        ],
        "probes": _probe_battery(),
        "dataset": {"format": "jsonl", "path": str(dataset_path)},
        "max_examples": 3,
        "generation": {"temperature": 0.0, "seed": 42, "max_tokens": 512},
        "report_title": "Ollama Cloud All-Model Benchmark",
        "output_dir": str(output_dir),
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        help="Comma- or space-separated list of Ollama Cloud models. Defaults to OLLAMA_CLOUD_MODELS.",
    )
    parser.add_argument(
        "--run-root",
        default=".tmp/runs/ollama-cloud",
        help="Root directory for baseline/candidate runs.",
    )
    parser.add_argument(
        "--dataset",
        default="examples/probe_battery.jsonl",
        help="Path to the probe battery dataset.",
    )
    args = parser.parse_args()

    import os

    if not os.environ.get("OLLAMA_API_KEY"):
        raise RuntimeError("Set OLLAMA_API_KEY before running this example.")

    model_source = args.models or os.environ.get("OLLAMA_CLOUD_MODELS")
    if not model_source:
        raise RuntimeError("Set OLLAMA_CLOUD_MODELS or pass --models to run this example.")

    models = _split_models(model_source)
    if not models:
        raise RuntimeError("No models provided after parsing OLLAMA_CLOUD_MODELS.")

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

    baseline_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

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

    harness_args = argparse.Namespace(
        config=str(baseline_config),
        run_id=None,
        run_root=None,
        run_dir=str(baseline_dir),
        output_dir=None,
        overwrite=True,
        resume=False,
        validate_output=True,
        schema_version=DEFAULT_SCHEMA_VERSION,
        validation_mode="strict",
        strict_serialization=True,
        deterministic_artifacts=True,
        verbose=False,
        quiet=True,
        track=None,
        track_project="ollama-cloud",
    )
    cmd_harness(harness_args)

    harness_args.config = str(candidate_config)
    harness_args.run_dir = str(candidate_dir)
    cmd_harness(harness_args)

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
