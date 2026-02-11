"""High-level workflow helpers for deterministic harness + diff usage.

These helpers provide a stable Python API surface for common CLI workflows,
so callers do not need to construct CLI argparse namespaces directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from insideLLMs.schemas import DEFAULT_SCHEMA_VERSION


def run_harness_to_dir(
    config_path: str | Path,
    run_dir: str | Path,
    *,
    overwrite: bool = True,
    validate_output: bool = True,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    validation_mode: str = "strict",
    strict_serialization: bool = True,
    deterministic_artifacts: bool = True,
    verbose: bool = False,
    quiet: bool = True,
    track: Optional[str] = None,
    track_project: str = "insideLLMs",
) -> int:
    """Run a harness config into a specific run directory.

    Returns the same exit code semantics as ``insidellms harness``.
    """
    from insideLLMs.cli.commands.harness import cmd_harness

    args = argparse.Namespace(
        config=str(config_path),
        run_id=None,
        run_root=None,
        run_dir=str(run_dir),
        output_dir=None,
        overwrite=overwrite,
        resume=False,
        validate_output=validate_output,
        schema_version=schema_version,
        validation_mode=validation_mode,
        strict_serialization=strict_serialization,
        deterministic_artifacts=deterministic_artifacts,
        verbose=verbose,
        quiet=quiet,
        track=track,
        track_project=track_project,
    )
    return cmd_harness(args)


def diff_run_dirs(
    run_dir_a: str | Path,
    run_dir_b: str | Path,
    *,
    fail_on_regressions: bool = False,
    fail_on_changes: bool = True,
    fail_on_trace_violations: bool = False,
    fail_on_trace_drift: bool = False,
    limit: int = 25,
    output_format: str = "text",
    output_path: Optional[str | Path] = None,
    output_fingerprint_ignore: Optional[list[str]] = None,
) -> int:
    """Diff two run directories.

    Returns the same exit code semantics as ``insidellms diff``.
    """
    from insideLLMs.cli.commands.diff import cmd_diff

    args = argparse.Namespace(
        run_dir_a=str(run_dir_a),
        run_dir_b=str(run_dir_b),
        format=output_format,
        output=str(output_path) if output_path is not None else None,
        limit=limit,
        fail_on_regressions=fail_on_regressions,
        fail_on_changes=fail_on_changes,
        output_fingerprint_ignore=output_fingerprint_ignore or [],
        fail_on_trace_violations=fail_on_trace_violations,
        fail_on_trace_drift=fail_on_trace_drift,
    )
    return cmd_diff(args)
