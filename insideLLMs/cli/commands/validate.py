"""Validate command: validate configuration files or run directories."""

import argparse
import json
import os
import stat
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


def _contained_records_path(run_dir: Path, records_file: object) -> Path:
    """Resolve records_file under run_dir with containment and no symlink follow.

    Schema v1 permits arbitrary records_file basenames/relative paths, but the
    resolved target must remain inside the run directory. Absolute paths, parent
    traversal, and symlink destinations are rejected.
    """
    if not isinstance(records_file, str) or not records_file.strip():
        raise ValueError("records_file must be a non-empty string")
    relative = Path(records_file)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"records_file must stay inside the run directory: {records_file}")
    run_dir_resolved = run_dir.resolve()
    candidate = run_dir / relative
    # Resolve the parent only so a final-component symlink is not followed for
    # containment; then reject the file itself if it is a symlink.
    parent_resolved = candidate.parent.resolve()
    try:
        parent_resolved.relative_to(run_dir_resolved)
    except ValueError as exc:
        raise ValueError(
            f"records_file must stay inside the run directory: {records_file}"
        ) from exc
    records_path = parent_resolved / candidate.name
    if os.path.lexists(records_path) and records_path.is_symlink():
        raise ValueError(f"records_file must not be a symlink: {records_file}")
    return records_path


def _open_records_nofollow(path: str, flags: int) -> int:
    """Open the final component without following a raced-in symlink."""
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise OSError("Records validation requires no-follow file I/O")
    descriptor = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("records_file must be a regular file")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


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

        if not isinstance(manifest_obj, dict):
            print_error("manifest.json must be a JSON object")
            return 1

        # Determine schema version: CLI override > manifest.schema_version > manifest.schemas[name]
        manifest_schemas = manifest_obj.get("schemas", {})
        # Let schema validation report malformed nested values in strict/warn
        # mode, without crashing during version discovery.
        schema_versions = manifest_schemas if isinstance(manifest_schemas, dict) else {}
        schema_version = (
            args.schema_version
            or manifest_obj.get("schema_version")
            or schema_versions.get(registry.RUN_MANIFEST)
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

        # Validate records — schema v1 permits arbitrary records_file names, but
        # the resolved path must stay inside the run directory (no traversal or
        # symlink escape).
        records_file = manifest_obj.get("records_file") or "records.jsonl"
        try:
            records_path = _contained_records_path(run_dir, records_file)
        except ValueError as e:
            _handle_error(str(e))
            return 0 if args.mode == "warn" else 1
        print_key_value("Records", records_path)

        if not records_path.exists():
            _handle_error(f"records file not found: {records_path}")
            return 0 if args.mode == "warn" else 1

        try:
            with open(records_path, encoding="utf-8", opener=_open_records_nofollow) as f:
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
