"""Export command: export results to various formats."""

import argparse
import json
import os
import tempfile
import unicodedata
from pathlib import Path

from insideLLMs.privacy.redaction import redact_pii
from insideLLMs.results import results_to_markdown

from .._output import print_error, print_header, print_key_value, print_success

_LATEX_LITERAL_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "%": r"\%",
    "_": r"\_",
    "^": r"\textasciicircum{}",
    "~": r"\textasciitilde{}",
}


def _escape_latex_literal(value: object) -> str:
    """Render a value as literal text without changing exporter syntax."""
    escaped: list[str] = []
    for character in str(value):
        if unicodedata.category(character) == "Cc" or character in "\u2028\u2029":
            escaped.append(" ")
        else:
            escaped.append(_LATEX_LITERAL_ESCAPES.get(character, character))
    return "".join(escaped)


def _load_results(input_path: Path) -> list:
    """Load results from JSON or JSONL file."""
    with open(input_path, encoding="utf-8") as f:
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

    staged_output: Path | None = None
    try:
        results = _load_results(input_path)

        if getattr(args, "redact_pii", False):
            results = redact_pii(results)
            print_key_value("Redact PII", "enabled")

        output_path = args.output
        if not output_path:
            output_path = input_path.stem + f".{args.format}"
        output_path = Path(output_path)

        # Validate encryption preconditions BEFORE writing any plaintext to disk,
        # otherwise a missing key / unsupported format leaves cleartext on disk.
        encrypt_requested = getattr(args, "encrypt", False)
        key_b64 = None
        if encrypt_requested:
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

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as staged:
                staged_output = Path(staged.name)

        write_path = staged_output if staged_output is not None else output_path

        if args.format == "csv":
            import csv

            if results:
                keys = results[0].keys()
                with open(write_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(results)

        elif args.format == "markdown":
            content = results_to_markdown(results)
            with open(write_path, "w", encoding="utf-8") as f:
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
                    " & ".join(_escape_latex_literal(key) for key in keys) + " \\\\",
                    "\\hline",
                ]
                for r in results[:20]:  # Limit rows
                    values = [_escape_latex_literal(str(r.get(k, ""))[:30]) for k in keys]
                    lines.append(" & ".join(values) + " \\\\")
                lines.extend(
                    [
                        "\\hline",
                        "\\end{tabular}",
                        "\\caption{Experiment Results}",
                        "\\end{table}",
                    ]
                )
                with open(write_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))

        elif args.format == "jsonl":
            from insideLLMs._serialization import stable_json_dumps

            with open(write_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(stable_json_dumps(r) + "\n")

        if encrypt_requested:
            try:
                from insideLLMs.privacy.encryption import encrypt_jsonl

                if key_b64 is None or staged_output is None:
                    print_error("Encryption key and staging path are required")
                    return 1
                encrypt_jsonl(staged_output, key=key_b64.encode())
                os.replace(staged_output, output_path)
                staged_output = None
                print_key_value("Encrypted", "yes")
            except Exception as e:
                print_error(f"Encryption failed: {e}")
                return 1

        print_success(f"Exported to: {output_path}")
        return 0

    except Exception as e:
        print_error(f"Export error: {e}")
        return 1
    finally:
        if staged_output is not None:
            staged_output.unlink(missing_ok=True)
