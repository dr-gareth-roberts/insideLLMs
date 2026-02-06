"""Export command: export results to various formats."""

import argparse
import json
from pathlib import Path

from insideLLMs.results import results_to_markdown

from .._output import print_error, print_header, print_key_value, print_success


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
        with open(input_path) as f:
            results = json.load(f)

        if not isinstance(results, list):
            results = [results]

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

        print_success(f"Exported to: {output_path}")
        return 0

    except Exception as e:
        print_error(f"Export error: {e}")
        return 1
