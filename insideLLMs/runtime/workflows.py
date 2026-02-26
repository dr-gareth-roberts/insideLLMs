"""High-level workflow helpers for deterministic harness + diff usage.

These helpers provide a stable Python API surface for common CLI workflows,
so callers do not need to construct CLI argparse namespaces directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Optional

from insideLLMs.schemas import DEFAULT_SCHEMA_VERSION

_TRACK_CHOICES = {"local", "wandb", "mlflow", "tensorboard"}
_VALIDATION_MODE_CHOICES = {"strict", "warn"}
_DIFF_FORMAT_CHOICES = {"text", "json"}


def _coerce_path(value: str | Path, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty path.")
    return str(Path(text).expanduser().absolute())


def run_harness_to_dir(
    config_path: str | Path,
    run_dir: str | Path,
    *,
    overwrite: bool = True,
    validate_output: bool = True,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    validation_mode: Literal["strict", "warn"] = "strict",
    strict_serialization: Optional[bool] = None,
    deterministic_artifacts: Optional[bool] = None,
    verbose: bool = False,
    quiet: bool = True,
    track: Optional[str] = None,
    track_project: str = "insidellms",
) -> int:
    """Run a harness config into a specific run directory.

    Returns the same exit code semantics as ``insidellms harness``.
    """
    from insideLLMs.cli.commands.harness import cmd_harness

    resolved_config = _coerce_path(config_path, name="config_path")
    if not Path(resolved_config).exists():
        raise FileNotFoundError(f"Config file not found: {resolved_config}")

    resolved_run_dir = _coerce_path(run_dir, name="run_dir")

    if validation_mode not in _VALIDATION_MODE_CHOICES:
        choices = ", ".join(sorted(_VALIDATION_MODE_CHOICES))
        raise ValueError(f"validation_mode must be one of: {choices}")

    if track is not None and track not in _TRACK_CHOICES:
        choices = ", ".join(sorted(_TRACK_CHOICES))
        raise ValueError(f"track must be one of: {choices}")

    args = argparse.Namespace(
        config=resolved_config,
        run_id=None,
        run_root=None,
        run_dir=resolved_run_dir,
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
    fail_on_changes: bool = False,
    fail_on_trace_violations: bool = False,
    fail_on_trace_drift: bool = False,
    limit: int = 25,
    output_format: Literal["text", "json"] = "text",
    output_path: Optional[str | Path] = None,
    output_fingerprint_ignore: Optional[list[str]] = None,
) -> int:
    """Diff two run directories.

    Returns the same exit code semantics as ``insidellms diff``.
    """
    from insideLLMs.cli.commands.diff import cmd_diff

    resolved_a = _coerce_path(run_dir_a, name="run_dir_a")
    resolved_b = _coerce_path(run_dir_b, name="run_dir_b")

    if output_format not in _DIFF_FORMAT_CHOICES:
        choices = ", ".join(sorted(_DIFF_FORMAT_CHOICES))
        raise ValueError(f"output_format must be one of: {choices}")

    if limit <= 0:
        raise ValueError("limit must be a positive integer")

    resolved_output = None
    if output_path is not None:
        resolved_output = _coerce_path(output_path, name="output_path")

    args = argparse.Namespace(
        run_dir_a=resolved_a,
        run_dir_b=resolved_b,
        format=output_format,
        output=resolved_output,
        limit=limit,
        fail_on_regressions=fail_on_regressions,
        fail_on_changes=fail_on_changes,
        output_fingerprint_ignore=list(output_fingerprint_ignore or []),
        fail_on_trace_violations=fail_on_trace_violations,
        fail_on_trace_drift=fail_on_trace_drift,
    )
    return cmd_diff(args)
