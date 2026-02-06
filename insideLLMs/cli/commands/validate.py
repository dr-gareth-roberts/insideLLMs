"""Validate command: validate configuration files or run directories."""

import argparse
import json
from pathlib import Path

from insideLLMs.registry import ensure_builtins_registered, model_registry, probe_registry
from insideLLMs.schemas import DEFAULT_SCHEMA_VERSION

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
            _handle_error(f"Could not read manifest JSON: {e}")
            return 0 if args.mode == "warn" else 1

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

        if errors:
            if args.mode == "warn":
                print_warning(f"Validation completed with {errors} error(s) (warn mode)")
                return 0
            print_error(f"Validation failed with {errors} error(s)")
            return 1

        print_success("Validation OK")
        return 0

    # ---------------------------------------------------------------------
    # Config validation (legacy)
    # ---------------------------------------------------------------------
    print_header("Validate Configuration")
    config_path = target_path
    print_key_value("Config", config_path)

    try:
        import yaml

        # Load config
        with open(config_path) as f:
            config = yaml.safe_load(f) if config_path.suffix in (".yaml", ".yml") else json.load(f)

        config_errors: list[str] = []
        warnings: list[str] = []

        # Validate model
        if "model" not in config:
            config_errors.append("Missing required field: model")
        else:
            model_config = config["model"]
            if "type" not in model_config:
                config_errors.append("Missing model.type")
            else:
                ensure_builtins_registered()
                if model_config["type"] not in model_registry.list():
                    config_errors.append(f"Unknown model type: {model_config['type']}")

        # Validate probe
        if "probe" not in config:
            config_errors.append("Missing required field: probe")
        else:
            probe_config = config["probe"]
            if "type" not in probe_config:
                config_errors.append("Missing probe.type")
            else:
                ensure_builtins_registered()
                if probe_config["type"] not in probe_registry.list():
                    config_errors.append(f"Unknown probe type: {probe_config['type']}")

        # Validate dataset
        if "dataset" not in config:
            warnings.append("No dataset specified (will use builtin)")
        else:
            ds_config = config["dataset"]
            if "path" in ds_config:
                ds_path = Path(ds_config["path"])
                if not ds_path.exists():
                    warnings.append(f"Dataset file not found: {ds_path}")

        # Report results
        if config_errors:
            print_subheader("Errors")
            for e in config_errors:
                print(f"  {colorize('ERROR', Colors.RED)} {e}")

        if warnings:
            print_subheader("Warnings")
            for w in warnings:
                print(f"  {colorize('WARN', Colors.YELLOW)} {w}")

        if not config_errors:
            print()
            print_success("Configuration is valid!")
            return 0
        else:
            print()
            print_error(f"Configuration has {len(config_errors)} error(s)")
            return 1

    except Exception as e:
        print_error(f"Validation error: {e}")
        return 1
