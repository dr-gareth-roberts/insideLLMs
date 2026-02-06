"""Compare command: compare multiple models on identical inputs."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

from insideLLMs.registry import ensure_builtins_registered, model_registry

from .._output import (
    Colors,
    colorize,
    print_error,
    print_header,
    print_key_value,
    print_subheader,
    print_success,
    print_warning,
)
from .._record_utils import _json_default


def cmd_compare(args: argparse.Namespace) -> int:
    """Execute the compare command."""
    ensure_builtins_registered()

    try:
        output_format = getattr(args, "format", "table") or "table"

        print_header("Model Comparison")

        models = [m.strip() for m in str(args.models).split(",") if m.strip()]
        if not models:
            print_error("No models provided via --models")
            return 1
        print_key_value("Models", ", ".join(models))

        inputs: list[str] = []
        if args.input:
            inputs = [str(args.input)]
        elif args.input_file:
            input_path = Path(args.input_file)
            if not input_path.exists():
                print_error(f"Input file not found: {input_path}")
                return 1

            try:
                if input_path.suffix == ".json":
                    with open(input_path, encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        inputs = [
                            d.get("input", d.get("question", str(d)))
                            if isinstance(d, dict)
                            else str(d)
                            for d in data
                        ]
                    elif isinstance(data, dict):
                        inputs = [data.get("input", data.get("question", str(data)))]
                    else:
                        inputs = [str(data)]
                elif input_path.suffix == ".jsonl":
                    with open(input_path, encoding="utf-8") as f:
                        for line_no, line in enumerate(f, start=1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                            except json.JSONDecodeError as e:
                                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
                            inputs.append(
                                data.get("input", data.get("question", str(data)))
                                if isinstance(data, dict)
                                else str(data)
                            )
                else:
                    with open(input_path, encoding="utf-8") as f:
                        inputs = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print_error(f"Could not read inputs from {input_path}: {e}")
                return 1
        else:
            print_error("Please provide --input or --input-file")
            return 1

        print_key_value("Inputs", len(inputs))

        # Instantiate models once (best-effort).
        model_instances: dict[str, Any] = {}
        for model_name in models:
            try:
                model_or_factory = model_registry.get(model_name)
                model_instances[model_name] = (
                    model_or_factory() if isinstance(model_or_factory, type) else model_or_factory
                )
            except Exception as e:
                model_instances[model_name] = None
                print_warning(f"Could not load model '{model_name}': {e}")

        results: list[dict[str, Any]] = []
        for inp in inputs:
            inp_display = inp[:50] + "..." if len(inp) > 50 else inp
            if output_format == "table":
                print_subheader(f"Input: {inp_display}")

            item: dict[str, Any] = {"input": inp, "models": []}
            for model_name in models:
                model = model_instances.get(model_name)
                if model is None:
                    item["models"].append(
                        {
                            "model": model_name,
                            "response": None,
                            "latency_ms": None,
                            "error": "model_init_failed",
                        }
                    )
                    continue

                try:
                    start = time.time()
                    response = model.generate(inp)
                    elapsed = time.time() - start
                    item["models"].append(
                        {
                            "model": model_name,
                            "response": response,
                            "latency_ms": elapsed * 1000,
                            "error": None,
                        }
                    )

                    if output_format == "table":
                        print(f"\n  {colorize(model_name, Colors.CYAN)}:")
                        preview = str(response)
                        print(f"    {preview[:100]}{'...' if len(preview) > 100 else ''}")
                        print(f"    {colorize(f'({elapsed * 1000:.1f}ms)', Colors.DIM)}")
                except Exception as e:
                    item["models"].append(
                        {
                            "model": model_name,
                            "response": None,
                            "latency_ms": None,
                            "error": str(e),
                        }
                    )
                    if output_format == "table":
                        print(f"\n  {colorize(model_name, Colors.RED)}: Error - {e}")

            results.append(item)

        if output_format == "json":
            payload = json.dumps(results, indent=2, default=_json_default)
            if args.output:
                Path(args.output).write_text(payload, encoding="utf-8")
                print_success(f"Results saved to: {args.output}")
            else:
                print(payload)
            return 0

        if output_format == "markdown":
            rows: list[dict[str, Any]] = []
            for item in results:
                for model_item in item["models"]:
                    rows.append(
                        {
                            "input": item["input"],
                            "model": model_item["model"],
                            "response": model_item.get("response") or "",
                            "error": model_item.get("error") or "",
                            "latency_ms": model_item.get("latency_ms"),
                        }
                    )

            def escape_cell(value: Any) -> str:
                text = "" if value is None else str(value)
                return text.replace("|", "\\|").replace("\n", "<br>")

            header = "| Input | Model | Response | Error | Latency (ms) |"
            sep = "|---|---|---|---|---|"
            lines = [header, sep]
            for row in rows:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            escape_cell(row["input"]),
                            escape_cell(row["model"]),
                            escape_cell(row["response"]),
                            escape_cell(row["error"]),
                            escape_cell(
                                f"{float(row['latency_ms']):.1f}"
                                if isinstance(row.get("latency_ms"), (int, float))
                                else ""
                            ),
                        ]
                    )
                    + " |"
                )
            md = "\n".join(lines)

            if args.output:
                Path(args.output).write_text(md, encoding="utf-8")
                print_success(f"Results saved to: {args.output}")
            else:
                print(md)
            return 0

        # table output already printed; optionally write JSON to file.
        if args.output:
            payload = json.dumps(results, indent=2, default=_json_default)
            Path(args.output).write_text(payload, encoding="utf-8")
            print_success(f"Results saved to: {args.output}")

        return 0

    except Exception as e:
        print_error(f"Comparison error: {e}")
        return 1
