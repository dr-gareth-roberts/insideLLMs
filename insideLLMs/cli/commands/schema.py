"""Schema command: inspect and validate versioned output schemas."""

import argparse
import json
from pathlib import Path
from typing import Any

from insideLLMs.schemas import DEFAULT_SCHEMA_VERSION

from .._output import print_error, print_header, print_key_value, print_success, print_warning
from .._record_utils import _json_default


def cmd_schema(args: argparse.Namespace) -> int:
    """Inspect/dump/validate versioned output schemas."""

    from insideLLMs.schemas import OutputValidationError, OutputValidator, SchemaRegistry

    registry = SchemaRegistry()

    op = getattr(args, "op", None) or "list"
    name = getattr(args, "name", None)
    version = getattr(args, "version", DEFAULT_SCHEMA_VERSION)

    # Shortcut UX: `insidellms schema <SchemaName>` -> dump schema
    if op not in {"list", "dump", "validate"}:
        name = name or op
        op = "dump"

    if op == "list":
        print_header("Available Output Schemas")
        schema_names = [
            registry.RUNNER_ITEM,
            registry.RUNNER_OUTPUT,
            registry.RESULT_RECORD,
            registry.RUN_MANIFEST,
            registry.HARNESS_RECORD,
            registry.HARNESS_SUMMARY,
            registry.HARNESS_EXPLAIN,
            registry.BENCHMARK_SUMMARY,
            registry.COMPARISON_REPORT,
            registry.DIFF_REPORT,
            registry.EXPORT_METADATA,
            registry.CUSTOM_TRACE,
        ]
        for name in schema_names:
            versions = registry.available_versions(name)
            if not versions:
                continue
            print_key_value(name, ", ".join(versions))
        return 0

    if op == "dump":
        if not name:
            print_error(
                "Missing schema name. Use: insidellms schema dump --name <SchemaName> [--version X]"
            )
            return 1
        try:
            schema = registry.get_json_schema(name, version)
        except Exception as e:
            print_error(f"Could not dump schema {name}@{version}: {e}")
            return 2

        payload = json.dumps(schema, indent=2, default=_json_default)
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(payload)
            print_success(f"Schema written to: {out_path}")
        else:
            print(payload)
        return 0

    if op == "validate":
        if not name:
            print_error(
                "Missing schema name. Use: insidellms schema validate --name <SchemaName> -i <file>"
            )
            return 1
        if not getattr(args, "input", None):
            print_error("Missing --input for schema validate")
            return 1

        in_path = Path(args.input)
        if not in_path.exists():
            print_error(f"Input file not found: {in_path}")
            return 1

        validator = OutputValidator(registry)
        errors = 0

        def validate_one(obj: Any) -> None:
            nonlocal errors
            try:
                validator.validate(
                    name,
                    obj,
                    schema_version=version,
                    mode="strict",
                )
            except OutputValidationError as e:
                errors += 1
                if args.mode == "warn":
                    print_warning(str(e))
                else:
                    raise

        try:
            if in_path.suffix.lower() == ".jsonl":
                with open(in_path) as f:
                    for line_no, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError as e:
                            errors += 1
                            if args.mode == "warn":
                                print_warning(f"Invalid JSON on line {line_no}: {e}")
                                continue
                            raise
                        validate_one(obj)
            else:
                obj = json.loads(in_path.read_text())
                if isinstance(obj, list):
                    for item in obj:
                        validate_one(item)
                else:
                    validate_one(obj)
        except Exception as e:
            if args.mode == "warn":
                print_warning(f"Validation completed with errors: {e}")
            else:
                print_error(f"Validation failed: {e}")
                return 1

        if errors:
            if args.mode == "warn":
                print_warning(f"Validated with {errors} error(s) (warn mode)")
                return 0
            print_error(f"Validation failed with {errors} error(s)")
            return 1

        print_success("Validation OK")
        return 0

    print_error(f"Unknown schema op: {op}")
    return 1
