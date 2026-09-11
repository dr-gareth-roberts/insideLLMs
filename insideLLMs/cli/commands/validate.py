"""Validate command: validate configuration files or run directories."""

import argparse
import json
from pathlib import Path

import yaml

from insideLLMs.schemas import DEFAULT_SCHEMA_VERSION

from .._output import (
    print_error,
    print_header,
    print_key_value,
    print_success,
    print_warning,
)


def cmd_validate(args: argparse.Namespace) -> int:
    """Execute the validate command."""
    target_path = Path(args.config)
    if not target_path.exists():
        print_error(f"Path not found: {target_path}")
        return 1

    # ---------------------------------------------------------------------
    # Run directory validation (manifest.json + records.jsonl)
    # ---------------------------------------------------------------------
    if target_path.is_dir() or target_path.name == "manifest.json":
        run_dir = target_path if target_path.is_dir() else target_path.parent
        manifest_path = (
            target_path if target_path.name == "manifest.json" else run_dir / "manifest.json"
        )

        print_header("Validate Run Directory")
        print_key_value("Run dir", run_dir)
        print_key_value("Manifest", manifest_path)

        if not manifest_path.exists():
            print_error(f"manifest.json not found: {manifest_path}")
            return 1

        from insideLLMs.schemas import OutputValidationError, OutputValidator, SchemaRegistry

        registry = SchemaRegistry()
        validator = OutputValidator(registry)

        errors = 0

        def _handle_error(msg: str) -> None:
            nonlocal errors
            errors += 1
            if args.mode == "warn":
                print_warning(msg)
            else:
                print_error(msg)

        try:
            manifest_obj = json.loads(manifest_path.read_text())
        except Exception as e:
            # An unreadable/corrupt manifest is a structural failure (not a schema
            # mismatch); fail hard regardless of mode rather than silently passing
            # and skipping all record validation.
            print_error(f"Could not read manifest JSON: {e}")
            return 1

        # Determine schema version: CLI override > manifest.schema_version > manifest.schemas[name]
        schema_version = (
            args.schema_version
            or manifest_obj.get("schema_version")
            or manifest_obj.get("schemas", {}).get(registry.RUN_MANIFEST)
            or DEFAULT_SCHEMA_VERSION
        )

        print_key_value("Schema version", schema_version)

        # Validate manifest
        try:
            validator.validate(
                registry.RUN_MANIFEST,
                manifest_obj,
                schema_version=schema_version,
                mode="strict",
            )
        except OutputValidationError as e:
            _handle_error(f"Manifest schema mismatch: {e}")
            if args.mode != "warn":
                return 1

        # Validate records
        records_file = manifest_obj.get("records_file") or "records.jsonl"
        records_path = run_dir / records_file
        print_key_value("Records", records_path)

        if not records_path.exists():
            _handle_error(f"records file not found: {records_path}")
            return 0 if args.mode == "warn" else 1

        try:
            with open(records_path, encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        _handle_error(f"Invalid JSON on line {line_no}: {e}")
                        if args.mode != "warn":
                            return 1
                        continue
                    try:
                        validator.validate(
                            registry.RESULT_RECORD,
                            obj,
                            schema_version=schema_version,
                            mode="strict",
                        )
                    except OutputValidationError as e:
                        _handle_error(f"Record line {line_no} schema mismatch: {e}")
                        if args.mode != "warn":
                            return 1
        except Exception as e:
            _handle_error(f"Error reading records: {e}")
            return 0 if args.mode == "warn" else 1

        # Strict mode returns early on the first record error. Accumulated
        # errors only remain in warn mode.
        if errors:
            print_warning(f"Validation completed with {errors} error(s) (warn mode)")
            return 0

        print_success("Validation OK")
        return 0

    print_header("Validate Configuration")
    print_key_value("Config", target_path)
    try:
        from insideLLMs.config_schema import normalize_runtime_config, resolve_dataset_path
        from insideLLMs.runtime._config_loader import load_config

        config = normalize_runtime_config(load_config(target_path), check_registry=True)
        dataset = config["dataset"]
        if dataset["format"] in ("csv", "jsonl"):
            dataset_path = resolve_dataset_path(dataset["path"], target_path.parent)
            if not dataset_path.is_file():
                raise ValueError(f"Dataset file not found: {dataset_path}")
        print_success("Configuration is valid!")
        return 0
    except (ValueError, OSError, TypeError, yaml.YAMLError) as exc:
        print_error(f"Validation error: {exc}")
        return 1
