"""Run command: execute experiments from configuration files."""

import argparse
import asyncio
import json
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from insideLLMs.results import results_to_markdown, save_results_json
from insideLLMs.runtime.runner import (
    derive_run_id_from_config_path,
    run_experiment_from_config,
    run_experiment_from_config_async,
)
from insideLLMs.types import ExperimentResult

from .._output import (
    Colors,
    ProgressBar,
    colorize,
    print_error,
    print_header,
    print_info,
    print_key_value,
    print_subheader,
    print_success,
    print_warning,
)
from .._record_utils import _json_default


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the run command."""
    config_path = Path(args.config)

    if not config_path.exists():
        print_error(f"Config file not found: {config_path}")
        return 1

    print_header("Running Experiment")
    print_key_value("Config", config_path)

    # Create progress bar if verbose
    progress_bar: Optional[ProgressBar] = None

    def progress_callback(current: int, total: int) -> None:
        nonlocal progress_bar
        if args.verbose and not args.quiet:
            if progress_bar is None:
                progress_bar = ProgressBar(total, prefix="Evaluating")
            progress_bar.update(current)

    try:
        start_time = time.time()
        tracker = None

        # Resolve deterministic run artifact location (used for both runner and UX hints)
        env_run_root = os.environ.get("INSIDELLMS_RUN_ROOT")
        default_run_root = (
            Path(env_run_root).expanduser().absolute()
            if env_run_root
            else Path.home() / ".insidellms" / "runs"
        )
        if args.run_id:
            resolved_run_id = args.run_id
        else:
            resolved_run_id = derive_run_id_from_config_path(
                config_path,
                schema_version=args.schema_version,
                strict_serialization=args.strict_serialization,
            )
        # Use absolute paths to reduce surprise when users provide relative paths.
        effective_run_root = (
            Path(args.run_root).expanduser().absolute() if args.run_root else default_run_root
        )
        effective_run_dir = (
            Path(args.run_dir).expanduser().absolute()
            if args.run_dir
            else (effective_run_root / resolved_run_id)
        )

        if args.track:
            try:
                from insideLLMs.experiment_tracking import TrackingConfig, create_tracker

                tracking_root = effective_run_dir.parent / "tracking"

                tracker_kwargs: dict[str, Any] = {}
                if args.track == "local":
                    tracker_kwargs["output_dir"] = str(tracking_root)
                    tracker_kwargs["config"] = TrackingConfig(project=args.track_project)
                elif args.track == "wandb":
                    tracker_kwargs["project"] = args.track_project
                elif args.track == "mlflow":
                    tracker_kwargs["experiment_name"] = args.track_project
                elif args.track == "tensorboard":
                    tracker_kwargs["log_dir"] = str(
                        tracking_root / "tensorboard" / args.track_project
                    )
                    tracker_kwargs["config"] = TrackingConfig(project=args.track_project)

                tracker = create_tracker(args.track, **tracker_kwargs)
                tracker.start_run(run_name=resolved_run_id, run_id=resolved_run_id)
                tracker.log_params(
                    {
                        "run_id": resolved_run_id,
                        "run_dir": str(effective_run_dir),
                        "config_path": str(config_path),
                        "schema_version": args.schema_version,
                    }
                )
            except Exception as e:
                print_warning(f"Tracking disabled: {e}")
                tracker = None

        if args.use_async:
            print_info(f"Using async execution with concurrency={args.concurrency}")
            results = asyncio.run(
                run_experiment_from_config_async(
                    config_path,
                    concurrency=args.concurrency,
                    progress_callback=progress_callback if args.verbose else None,
                    validate_output=args.validate_output,
                    schema_version=args.schema_version,
                    validation_mode=args.validation_mode,
                    emit_run_artifacts=True,
                    run_dir=str(effective_run_dir) if args.run_dir else None,
                    run_root=str(effective_run_root),
                    run_id=resolved_run_id,
                    overwrite=bool(args.overwrite),
                    resume=bool(args.resume),
                    strict_serialization=args.strict_serialization,
                    deterministic_artifacts=args.deterministic_artifacts,
                )
            )
        else:
            results = run_experiment_from_config(
                config_path,
                progress_callback=progress_callback if args.verbose else None,
                validate_output=args.validate_output,
                schema_version=args.schema_version,
                validation_mode=args.validation_mode,
                emit_run_artifacts=True,
                run_dir=str(effective_run_dir) if args.run_dir else None,
                run_root=str(effective_run_root),
                run_id=resolved_run_id,
                overwrite=bool(args.overwrite),
                resume=bool(args.resume),
                strict_serialization=args.strict_serialization,
                deterministic_artifacts=args.deterministic_artifacts,
            )

        elapsed = time.time() - start_time

        if progress_bar:
            progress_bar.finish()

        experiment_result = results if isinstance(results, ExperimentResult) else None
        if experiment_result is not None:
            raw_results = [
                {
                    "input": r.input,
                    "output": r.output,
                    "error": r.error,
                    "status": r.status.value if isinstance(r.status, Enum) else str(r.status),
                    "latency_ms": r.latency_ms,
                    "metadata": r.metadata,
                }
                for r in experiment_result.results
            ]
            success_count = experiment_result.success_count
            error_count = experiment_result.error_count
            total = experiment_result.total_count
        else:
            raw_results = results
            success_count = sum(1 for r in results if r.get("status") == "success")
            error_count = sum(1 for r in results if r.get("status") == "error")
            total = len(results)

        # Output results
        if args.output:
            save_results_json(
                results,
                args.output,
                validate_output=args.validate_output,
                schema_version=args.schema_version,
                validation_mode=args.validation_mode,
            )
            print_success(f"Results saved to: {args.output}")

        if args.format == "json":
            print(json.dumps(results, indent=2, default=_json_default))
        elif args.format == "markdown":
            print(results_to_markdown(raw_results))
        elif args.format == "summary":
            # Minimal summary output
            print(
                f"OK {success_count}/{total} successful ({success_count / max(1, total) * 100:.1f}%)"
            )
        else:  # table format
            print_subheader("Results Summary")
            print_key_value("Total items", total)
            print_key_value(
                "Successful", f"{success_count} ({success_count / max(1, total) * 100:.1f}%)"
            )
            print_key_value("Errors", f"{error_count} ({error_count / max(1, total) * 100:.1f}%)")
            print_key_value("Duration", f"{elapsed:.2f}s")

            if raw_results:
                latencies = [
                    r.get("latency_ms", 0) for r in raw_results if r.get("status") == "success"
                ]
                if latencies:
                    avg_latency = sum(latencies) / len(latencies)
                    min_latency = min(latencies)
                    max_latency = max(latencies)
                    print_key_value("Avg latency", f"{avg_latency:.1f}ms")
                    print_key_value("Min/Max", f"{min_latency:.1f}ms / {max_latency:.1f}ms")

            # Show first few results
            print_subheader("Sample Results")
            for i, r in enumerate(raw_results[:5]):
                status_icon = (
                    colorize("OK", Colors.GREEN)
                    if r.get("status") == "success"
                    else colorize("FAIL", Colors.RED)
                )
                inp = str(r.get("input", ""))[:50]
                if len(str(r.get("input", ""))) > 50:
                    inp += "..."
                print(f"  {status_icon} [{i + 1}] {inp}")

            if len(raw_results) > 5:
                print(colorize(f"  ... and {len(raw_results) - 5} more", Colors.DIM))

        if tracker is not None:
            try:
                if experiment_result is not None:
                    tracker.log_experiment_result(experiment_result)

                metrics: dict[str, float] = {
                    "wall_time_seconds": float(elapsed),
                    "total_count": float(total),
                    "success_count": float(success_count),
                    "error_count": float(error_count),
                }

                latency_values = [
                    float(latency_ms)
                    for latency_ms in (
                        r.get("latency_ms") for r in raw_results if r.get("status") == "success"
                    )
                    if isinstance(latency_ms, (int, float))
                ]
                if latency_values:
                    metrics["avg_latency_ms"] = float(sum(latency_values) / len(latency_values))
                    metrics["min_latency_ms"] = float(min(latency_values))
                    metrics["max_latency_ms"] = float(max(latency_values))

                tracker.log_metrics(metrics)

                for artifact in (
                    effective_run_dir / "manifest.json",
                    effective_run_dir / "records.jsonl",
                    effective_run_dir / "config.resolved.yaml",
                    effective_run_dir / "summary.json",
                    effective_run_dir / "report.html",
                ):
                    if artifact.exists():
                        tracker.log_artifact(str(artifact), artifact_name=artifact.name)

                if args.output and Path(args.output).exists():
                    tracker.log_artifact(
                        str(Path(args.output)), artifact_name=Path(args.output).name
                    )

                tracker.end_run(status="finished")
                tracker = None
            except Exception as e:
                print_warning(f"Tracking error: {e}")

        # UX sugar: make it obvious where artifacts landed and how to validate.
        # Keep stdout JSON clean when --format json.
        if not args.quiet:
            hint_stream = sys.stderr if args.format == "json" else sys.stdout
            print(f"\nRun written to: {effective_run_dir}", file=hint_stream)
            print(f"Validate with: insidellms validate {effective_run_dir}", file=hint_stream)

        return 0

    except Exception as e:
        if "tracker" in locals() and tracker is not None:
            try:
                tracker.end_run(status="failed")
            except (AttributeError, RuntimeError):
                pass
        print_error(f"Error running experiment: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1
