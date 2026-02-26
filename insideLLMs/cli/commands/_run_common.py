"""Shared helpers for run/harness CLI commands.

This module centralizes common command behaviors so `insidellms run` and
`insidellms harness` stay aligned over time.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import insideLLMs.experiment_tracking as experiment_tracking

from .._output import print_warning


def resolve_harness_output_dir(
    args: argparse.Namespace, config: dict[str, Any], resolved_run_id: str
) -> Path:
    """Resolve the effective output directory for harness artifacts.

    Args:
        args: Parsed CLI arguments from `insidellms harness`.
        config: Loaded harness config dict.
        resolved_run_id: Final run identifier for this execution.

    Returns:
        Absolute output path where harness artifacts should be written.
    """

    effective_run_root = Path(args.run_root).expanduser().absolute() if args.run_root else None
    config_default_dir = Path(config.get("output_dir", "results")).expanduser().absolute()

    if args.run_dir:
        return Path(args.run_dir).expanduser().absolute()
    if args.output_dir:
        return Path(args.output_dir).expanduser().absolute()
    if effective_run_root is not None:
        return effective_run_root / resolved_run_id
    return config_default_dir


def create_tracker(
    *,
    backend: str | None,
    project: str,
    run_dir: Path,
    run_id: str,
    config_path: Path,
    schema_version: str,
) -> Any | None:
    """Create and initialize an experiment tracker for a run.

    Args:
        backend: Tracker backend name (`local`, `wandb`, `mlflow`, `tensorboard`), or None.
        project: Project/experiment name for the tracker.
        run_dir: Effective run directory.
        run_id: Resolved run ID.
        config_path: Source config file path.
        schema_version: Active schema version.

    Returns:
        Initialized tracker instance, or None when tracking is disabled/unavailable.
    """

    if not backend:
        return None

    try:
        tracking_root = run_dir.parent / "tracking"
        tracker_kwargs: dict[str, Any] = {}

        if backend == "local":
            tracker_kwargs["output_dir"] = str(tracking_root)
            tracker_kwargs["config"] = experiment_tracking.TrackingConfig(project=project)
        elif backend == "wandb":
            tracker_kwargs["project"] = project
        elif backend == "mlflow":
            tracker_kwargs["experiment_name"] = project
        elif backend == "tensorboard":
            tracker_kwargs["log_dir"] = str(tracking_root / "tensorboard" / project)
            tracker_kwargs["config"] = experiment_tracking.TrackingConfig(project=project)

        tracker = experiment_tracking.create_tracker(backend, **tracker_kwargs)
        tracker.start_run(run_name=run_id, run_id=run_id)
        tracker.log_params(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "config_path": str(config_path),
                "schema_version": schema_version,
            }
        )
        return tracker
    except Exception as exc:
        print_warning(f"Tracking disabled: {exc}")
        return None


def iter_standard_run_artifacts(run_dir: Path) -> tuple[Path, ...]:
    """Return the canonical artifact set used by run and harness commands.

    Args:
        run_dir: Directory that contains run artifacts.

    Returns:
        Tuple of standard artifact paths.
    """

    return (
        run_dir / "manifest.json",
        run_dir / "records.jsonl",
        run_dir / "config.resolved.yaml",
        run_dir / "summary.json",
        run_dir / "report.html",
    )
