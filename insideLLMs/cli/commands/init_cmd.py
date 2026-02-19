"""Init command: generate sample configuration files."""

import argparse
import json
from pathlib import Path
from typing import Any

from .._output import (
    Colors,
    colorize,
    print_header,
    print_info,
    print_key_value,
    print_success,
)


def cmd_init(args: argparse.Namespace) -> int:
    """Execute the init command."""
    import yaml

    print_header("Initialize Experiment Configuration")

    # Base configuration
    config: dict[str, Any] = {
        "model": {
            "type": args.model,
            "args": {},
        },
        "probe": {
            "type": args.probe,
            "args": {},
        },
        "dataset": {
            "format": "jsonl",
            "path": "data/questions.jsonl",
        },
    }

    # Add model-specific args hints
    model_hints = {
        "openai": {"model_name": "gpt-4"},
        "openrouter": {"model_name": "openai/gpt-4o-mini"},
        "anthropic": {"model_name": "claude-3-opus-20240229"},
        "cohere": {"model_name": "command-r-plus"},
        "gemini": {"model_name": "gemini-pro"},
        "huggingface": {"model_name": "gpt2"},
        "ollama": {"model_name": "llama2"},
    }
    if args.model in model_hints:
        config["model"]["args"] = model_hints[args.model]

    # Apply template enhancements
    if args.template == "benchmark":
        config["benchmark"] = {
            "datasets": ["reasoning", "math", "coding"],
            "max_examples_per_dataset": 10,
        }
    elif args.template == "tracking":
        config["tracking"] = {
            "backend": "local",
            "project": "my-experiment",
            "log_dir": "./experiments",
        }
    elif args.template == "full":
        config["benchmark"] = {
            "datasets": ["reasoning", "math", "coding", "safety"],
            "max_examples_per_dataset": 20,
        }
        config["tracking"] = {
            "backend": "local",
            "project": "my-experiment",
            "log_dir": "./experiments",
        }
        config["async"] = {
            "enabled": True,
            "concurrency": 5,
        }
        config["output"] = {
            "format": "json",
            "path": "results/experiment_results.json",
            "html_report": True,
        }

    output_path = Path(args.output)

    if output_path.suffix in (".yaml", ".yml"):
        content = yaml.dump(config, default_flow_style=False, sort_keys=False)
    else:
        content = json.dumps(config, indent=2)

    output_path.write_text(content)
    print_success(f"Created config: {output_path}")
    print_key_value("Template", args.template)
    print_key_value("Model", args.model)
    print_key_value("Probe", args.probe)

    # Create sample data directory and file
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    sample_data_path = data_dir / "questions.jsonl"
    if not sample_data_path.exists():
        sample_data = [
            {"question": "What is 2+2?", "reference_answer": "4"},
            {"question": "What is the capital of France?", "reference_answer": "Paris"},
            {"question": "Who wrote Romeo and Juliet?", "reference_answer": "William Shakespeare"},
            {
                "question": "If all cats are mammals, and all mammals are animals, are all cats animals?",
                "reference_answer": "Yes",
            },
            {"question": "What is the chemical symbol for water?", "reference_answer": "H2O"},
        ]
        with open(sample_data_path, "w") as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")
        print_success(f"Created sample data: {sample_data_path}")

    print()
    print_info("Next steps:")
    print(f"  1. Edit {colorize(str(output_path), Colors.CYAN)} to customize your experiment")
    print(f"  2. Run: {colorize(f'insidellms run {output_path}', Colors.GREEN)}")

    return 0
