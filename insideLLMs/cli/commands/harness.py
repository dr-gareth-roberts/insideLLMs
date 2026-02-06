"""Harness command: run cross-model behavioural probe harnesses."""

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional

from insideLLMs._serialization import (
    StrictSerializationError,
)
from insideLLMs.runtime.runner import (
    derive_run_id_from_config_path,
    load_config,
    run_harness_from_config,
)

from .._output import (
    ProgressBar,
    print_error,
    print_header,
    print_key_value,
    print_success,
    print_warning,
)
from .._record_utils import _write_jsonl
from .._report_builder import _build_basic_harness_report


def cmd_harness(args: argparse.Namespace) -> int:
    """Execute the harness command."""
    config_path = Path(args.config)

    if not config_path.exists():
        print_error(f"Config file not found: {config_path}")
        return 1

    print_header("Running Behavioural Harness")
    print_key_value("Config", config_path)

    progress_bar: Optional[ProgressBar] = None

    def progress_callback(current: int, total: int) -> None:
        nonlocal progress_bar
        if not args.verbose or args.quiet:
            return
        if progress_bar is None:
            progress_bar = ProgressBar(total, prefix="Evaluating")
        progress_bar.update(current)

    tracker = None

    try:
        start_time = time.time()
        resolved_run_id = args.run_id or derive_run_id_from_config_path(
            config_path,
            schema_version=args.schema_version,
            strict_serialization=args.strict_serialization,
        )

        # Precompute output_dir for tracking. Must match the precedence used later when emitting
        # standardized run artifacts.
        config = load_config(config_path)
        effective_run_root = Path(args.run_root).expanduser().absolute() if args.run_root else None
        config_default_dir = (
            Path(config.get("output_dir", "results")).expanduser().absolute()
            if isinstance(config, dict)
            else Path("results").expanduser().absolute()
        )
        if args.run_dir:
            output_dir = Path(args.run_dir).expanduser().absolute()
        elif args.output_dir:
            output_dir = Path(args.output_dir).expanduser().absolute()
        elif effective_run_root is not None:
            output_dir = effective_run_root / resolved_run_id
        else:
            output_dir = config_default_dir

        if args.track:
            try:
                from insideLLMs.experiment_tracking import TrackingConfig, create_tracker

                tracking_root = output_dir.parent / "tracking"

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
                        "run_dir": str(output_dir),
                        "config_path": str(config_path),
                        "schema_version": args.schema_version,
                    }
                )
            except Exception as e:
                print_warning(f"Tracking disabled: {e}")
                tracker = None

        result = run_harness_from_config(
            config_path,
            progress_callback=progress_callback if args.verbose else None,
            validate_output=args.validate_output,
            schema_version=args.schema_version,
            validation_mode=args.validation_mode,
            strict_serialization=args.strict_serialization,
            deterministic_artifacts=args.deterministic_artifacts,
        )
        elapsed = time.time() - start_time

        if progress_bar:
            progress_bar.finish()

        # -----------------------------------------------------------------
        # Determine harness run directory and emit standardized run artifacts
        # -----------------------------------------------------------------
        from insideLLMs.runtime.runner import (
            _build_resolved_config_snapshot,
            _deterministic_base_time,
            _deterministic_run_id_from_config_snapshot,
            _deterministic_run_times,
            _prepare_run_dir,
            _resolve_determinism_options,
            _serialize_value,
        )

        config_snapshot = result.get("config_snapshot")
        if not isinstance(config_snapshot, dict):
            config_snapshot = _build_resolved_config_snapshot(result["config"], config_path.parent)

        strict_serialization = result.get("strict_serialization")
        deterministic_artifacts = result.get("deterministic_artifacts")
        if not isinstance(strict_serialization, bool) or not isinstance(
            deterministic_artifacts, bool
        ):
            strict_serialization, deterministic_artifacts = _resolve_determinism_options(
                config_snapshot,
                strict_override=args.strict_serialization,
                deterministic_artifacts_override=args.deterministic_artifacts,
            )

        if args.run_id:
            resolved_run_id = args.run_id
        else:
            resolved_run_id = result.get("run_id") or resolved_run_id
            if not resolved_run_id:
                try:
                    resolved_run_id = _deterministic_run_id_from_config_snapshot(
                        config_snapshot,
                        schema_version=args.schema_version,
                        strict_serialization=strict_serialization,
                    )
                except StrictSerializationError as exc:
                    raise ValueError(
                        "strict_serialization requires JSON-stable values in the resolved harness config."
                    ) from exc

        # Absolute paths reduce surprise when users pass relative paths.
        effective_run_root = Path(args.run_root).expanduser().absolute() if args.run_root else None
        config_default_dir = (
            Path(result["config"].get("output_dir", "results")).expanduser().absolute()
        )

        # Precedence: --run-dir > --output-dir (legacy alias) > --run-root/run-id > config output_dir
        if args.run_dir:
            output_dir = Path(args.run_dir).expanduser().absolute()
        elif args.output_dir:
            output_dir = Path(args.output_dir).expanduser().absolute()
        elif effective_run_root is not None:
            output_dir = effective_run_root / resolved_run_id
        else:
            output_dir = config_default_dir

        def _semver_tuple(version: str) -> tuple[int, int, int]:
            from insideLLMs.schemas.registry import semver_tuple

            return semver_tuple(version)

        def _atomic_write_text(path: Path, text: str) -> None:
            from insideLLMs.resources import atomic_write_text

            atomic_write_text(path, text)

        def _atomic_write_yaml(path: Path, data: Any) -> None:
            import yaml

            content = yaml.safe_dump(
                _serialize_value(data, strict=strict_serialization),
                sort_keys=True,
                default_flow_style=False,
                allow_unicode=True,
            )
            _atomic_write_text(path, content)

        def _ensure_run_sentinel(run_dir_path: Path) -> None:
            marker = run_dir_path / ".insidellms_run"
            if not marker.exists():
                try:
                    marker.write_text("insideLLMs run directory\n", encoding="utf-8")
                except (IOError, OSError):
                    pass

        # Prepare run dir with the same safety policy as `insidellms run`.
        _prepare_run_dir(output_dir, overwrite=bool(args.overwrite), run_root=effective_run_root)
        _ensure_run_sentinel(output_dir)

        # Write resolved config snapshot for reproducibility.
        _atomic_write_yaml(output_dir / "config.resolved.yaml", config_snapshot)

        # Canonical record stream for validate/run-dir tooling.
        records_path = output_dir / "records.jsonl"

        # Ensure records' run_id matches the directory's run_id.
        for record in result.get("records", []):
            if isinstance(record, dict):
                record["run_id"] = resolved_run_id

        _write_jsonl(result["records"], records_path, strict_serialization=strict_serialization)
        print_success(f"Records written to: {records_path}")

        # Backward compatibility: keep results.jsonl alongside records.jsonl.
        legacy_results_path = output_dir / "results.jsonl"
        try:
            os.symlink("records.jsonl", legacy_results_path)
        except (OSError, NotImplementedError):
            try:
                os.link(records_path, legacy_results_path)
            except (OSError, IOError):
                shutil.copyfile(records_path, legacy_results_path)

        ds_cfg = result.get("config", {}).get("dataset")
        ds_cfg = ds_cfg if isinstance(ds_cfg, dict) else {}
        resolved_ds_cfg = config_snapshot.get("dataset")
        resolved_ds_cfg = resolved_ds_cfg if isinstance(resolved_ds_cfg, dict) else ds_cfg
        dataset_spec = {
            "dataset_id": resolved_ds_cfg.get("name")
            or resolved_ds_cfg.get("dataset")
            or resolved_ds_cfg.get("path")
            or resolved_ds_cfg.get("dataset_id"),
            "dataset_version": resolved_ds_cfg.get("version")
            or resolved_ds_cfg.get("split")
            or resolved_ds_cfg.get("dataset_version"),
            "dataset_hash": resolved_ds_cfg.get("hash") or resolved_ds_cfg.get("dataset_hash"),
            "provenance": resolved_ds_cfg.get("provenance")
            or resolved_ds_cfg.get("source")
            or resolved_ds_cfg.get("format"),
            "params": resolved_ds_cfg,
        }

        models_cfg = result.get("config", {}).get("models")
        models_cfg = models_cfg if isinstance(models_cfg, list) else []
        model_types = [m.get("type") for m in models_cfg if isinstance(m, dict) and m.get("type")]

        probes_cfg = result.get("config", {}).get("probes")
        probes_cfg = probes_cfg if isinstance(probes_cfg, list) else []
        probe_types = [p.get("type") for p in probes_cfg if isinstance(p, dict) and p.get("type")]

        record_count = len(result.get("records", []))
        success_count = sum(1 for r in result.get("records", []) if r.get("status") == "success")
        error_count = sum(1 for r in result.get("records", []) if r.get("status") == "error")

        run_base_time = _deterministic_base_time(resolved_run_id)
        started_at, completed_at = _deterministic_run_times(run_base_time, record_count)
        created_at = started_at

        python_version = None if deterministic_artifacts else sys.version.split()[0]
        platform_info = None if deterministic_artifacts else platform.platform()

        def _serialize_manifest(value: Any) -> Any:
            return _serialize_value(value, strict=strict_serialization)

        manifest: dict[str, Any] = {
            "schema_version": args.schema_version,
            "run_id": resolved_run_id,
            "created_at": created_at,
            "started_at": started_at,
            "completed_at": completed_at,
            "library_version": None,
            "python_version": python_version,
            "platform": platform_info,
            "command": None,
            "model": {
                "model_id": "harness",
                "provider": "insideLLMs",
                "params": {"model_count": len(model_types), "models": model_types},
            },
            "probe": {
                "probe_id": "harness",
                "probe_version": None,
                "params": {"probe_count": len(probe_types), "probes": probe_types},
            },
            "dataset": dataset_spec,
            "record_count": record_count,
            "success_count": success_count,
            "error_count": error_count,
            "records_file": "records.jsonl",
            "schemas": {"RunManifest": args.schema_version, "ResultRecord": args.schema_version},
            "custom": {
                "harness": {
                    "models": models_cfg,
                    "probes": probes_cfg,
                    "dataset": resolved_ds_cfg,
                    "max_examples": result.get("config", {}).get("max_examples"),
                    "experiment_count": len(result.get("experiments", [])),
                    "legacy_results_file": "results.jsonl",
                },
                "determinism": {
                    "strict_serialization": strict_serialization,
                    "deterministic_artifacts": deterministic_artifacts,
                },
            },
        }

        if _semver_tuple(args.schema_version) >= (1, 0, 1):
            manifest["run_completed"] = True

        try:
            import insideLLMs

            manifest["library_version"] = getattr(insideLLMs, "__version__", None)
        except (ImportError, AttributeError):
            pass

        if args.validate_output:
            from insideLLMs.schemas import OutputValidator, SchemaRegistry

            validator = OutputValidator(SchemaRegistry())
            validator.validate(
                SchemaRegistry.RUN_MANIFEST,
                manifest,
                schema_version=args.schema_version,
                mode=args.validation_mode,
            )

        manifest_path = output_dir / "manifest.json"
        _atomic_write_text(
            manifest_path,
            json.dumps(
                _serialize_manifest(manifest),
                sort_keys=True,
                indent=2,
                default=_serialize_manifest,
            ),
        )
        print_success(f"Manifest written to: {manifest_path}")

        summary_path = output_dir / "summary.json"
        report_path = output_dir / "report.html"

        summary_payload = {
            "schema_version": args.schema_version,
            "generated_at": created_at,
            "summary": result["summary"],
            "config": result["config"],
        }

        if args.validate_output:
            from insideLLMs.schemas import OutputValidator, SchemaRegistry

            validator = OutputValidator(SchemaRegistry())
            validator.validate(
                SchemaRegistry.HARNESS_SUMMARY,
                summary_payload,
                schema_version=args.schema_version,
                mode=args.validation_mode,
            )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                _serialize_manifest(summary_payload),
                f,
                indent=2,
                default=_serialize_manifest,
                sort_keys=True,
            )
        print_success(f"Summary written to: {summary_path}")

        if not args.skip_report:
            report_title = args.report_title or result["config"].get(
                "report_title", "Behavioural Probe Report"
            )
            try:
                from insideLLMs.visualization import create_interactive_html_report

                create_interactive_html_report(
                    result["experiments"],
                    title=report_title,
                    save_path=str(report_path),
                    generated_at=created_at,
                )
            except ImportError:
                report_html = _build_basic_harness_report(
                    result["experiments"],
                    result["summary"],
                    report_title,
                    generated_at=created_at,
                )
                with open(report_path, "w") as f:
                    f.write(report_html)

            print_success(f"Report written to: {report_path}")

        print_key_value("Elapsed", f"{elapsed:.2f}s")

        if tracker is not None:
            try:
                metrics: dict[str, float] = {
                    "wall_time_seconds": float(elapsed),
                    "record_count": float(record_count),
                    "success_count": float(success_count),
                    "error_count": float(error_count),
                    "experiment_count": float(len(result.get("experiments", []))),
                }
                tracker.log_metrics(metrics)

                tracker.log_params(
                    {
                        "model_count": len(model_types),
                        "probe_count": len(probe_types),
                        "dataset_id": dataset_spec.get("dataset_id"),
                        "dataset_version": dataset_spec.get("dataset_version"),
                        "dataset_hash": dataset_spec.get("dataset_hash"),
                        "dataset_provenance": dataset_spec.get("provenance"),
                    }
                )

                for artifact in (
                    output_dir / "manifest.json",
                    output_dir / "records.jsonl",
                    output_dir / "config.resolved.yaml",
                    output_dir / "summary.json",
                    output_dir / "report.html",
                ):
                    if artifact.exists():
                        tracker.log_artifact(str(artifact), artifact_name=artifact.name)

                tracker.end_run(status="finished")
                tracker = None
            except Exception as e:
                print_warning(f"Tracking error: {e}")

        if not args.quiet:
            print(f"\nRun written to: {output_dir}")
            print(f"Validate with: insidellms validate {output_dir}")
        return 0

    except Exception as e:
        if tracker is not None:
            try:
                tracker.end_run(status="failed")
            except (AttributeError, RuntimeError):
                pass
        print_error(f"Error running harness: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1
