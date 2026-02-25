"""Export command: export results to various formats."""

import argparse
import json
import os
from pathlib import Path

from insideLLMs.privacy.redaction import redact_pii
from insideLLMs.results import results_to_markdown

from .._output import print_error, print_header, print_key_value, print_success


def _load_results(input_path: Path) -> list:
    """Load results from JSON or JSONL file."""
    with open(input_path) as f:
        if input_path.suffix.lower() == ".jsonl":
            results = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            results = data if isinstance(data, list) else [data]
    return results


def cmd_export(args: argparse.Namespace) -> int:
    """Execute the export command."""
    print_header("Export Results")

    input_path = Path(args.input)
    if not input_path.exists():
        print_error(f"Input file not found: {input_path}")
        return 1

    print_key_value("Input", input_path)
    print_key_value("Format", args.format)

    try:
        results = _load_results(input_path)

        if getattr(args, "redact_pii", False):
            results = redact_pii(results)
            print_key_value("Redact PII", "enabled")

        output_path = args.output
        if not output_path:
            output_path = input_path.stem + f".{args.format}"

        if args.format == "csv":
            import csv

            if results:
                keys = results[0].keys()
                with open(output_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(results)

        elif args.format == "markdown":
            content = results_to_markdown(results)
            with open(output_path, "w") as f:
                f.write(content)

        elif args.format == "html":
            print_error(
                "HTML export requires plotly and ExperimentResult format; "
                "use `insidellms report <run_dir>` to produce report.html."
            )
            return 1

        elif args.format == "latex":
            # Generate LaTeX table
            if results:
                keys = list(results[0].keys())
                lines = [
                    "\\begin{table}[h]",
                    "\\centering",
                    "\\begin{tabular}{" + "l" * len(keys) + "}",
                    "\\hline",
                    " & ".join(keys) + " \\\\",
                    "\\hline",
                ]
                for r in results[:20]:  # Limit rows
                    values = [str(r.get(k, ""))[:30] for k in keys]
                    lines.append(" & ".join(values) + " \\\\")
                lines.extend(
                    [
                        "\\hline",
                        "\\end{tabular}",
                        "\\caption{Experiment Results}",
                        "\\end{table}",
                    ]
                )
                with open(output_path, "w") as f:
                    f.write("\n".join(lines))

        elif args.format == "jsonl":
            with open(output_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r, default=str) + "\n")

        if getattr(args, "encrypt", False):
            key_env = getattr(args, "encryption_key_env", "INSIDELLMS_ENCRYPTION_KEY")
            key_b64 = os.environ.get(key_env)
            if not key_b64:
                print_error(
                    f"Encryption requested but {key_env} is not set. "
                    "Set the env var with a Fernet key (base64)."
                )
                return 1
            if args.format != "jsonl":
                print_error("--encrypt is only supported for JSONL format")
                return 1
            try:
                from insideLLMs.privacy.encryption import encrypt_jsonl

                encrypt_jsonl(output_path, key=key_b64.encode())
                print_key_value("Encrypted", "yes")
            except RuntimeError as e:
                print_error(f"Encryption failed: {e}")
                return 1

        print_success(f"Exported to: {output_path}")
        return 0

    except Exception as e:
        print_error(f"Export error: {e}")
        return 1
