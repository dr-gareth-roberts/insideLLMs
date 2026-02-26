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
import json
import os
from pathlib import Path
from typing import Iterable

from insideLLMs.runtime import diff_run_dirs, run_harness_to_dir


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
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


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

    if not os.environ.get("OLLAMA_API_KEY"):
        raise RuntimeError("Set OLLAMA_API_KEY before running this example.")

    if not Path("insideLLMs").exists():
        raise RuntimeError(
            "Run this example from the repository root (or install insideLLMs in editable mode)."
        )

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

    if run_harness_to_dir(baseline_config, baseline_dir, track_project="ollama-cloud") != 0:
        raise SystemExit(1)
    if run_harness_to_dir(candidate_config, candidate_dir, track_project="ollama-cloud") != 0:
        raise SystemExit(1)

    exit_code = diff_run_dirs(baseline_dir, candidate_dir, fail_on_changes=True)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
