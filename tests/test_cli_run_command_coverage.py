import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

from insideLLMs.cli.commands.run import cmd_run
from insideLLMs.types import ExperimentResult, ModelInfo, ProbeCategory, ProbeResult, ResultStatus


def _make_run_args(config_path: Path, **overrides):
    defaults = {
        "config": str(config_path),
        "verbose": False,
        "quiet": False,
        "run_id": None,
        "run_root": None,
        "run_dir": None,
        "schema_version": "1.0.1",
        "strict_serialization": False,
        "deterministic_artifacts": True,
        "track": None,
        "track_project": "test-project",
        "use_async": False,
        "concurrency": 4,
        "validate_output": False,
        "validation_mode": "warn",
        "output": None,
        "format": "table",
        "overwrite": False,
        "resume": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: {type: dummy}\nprobe: {type: dummy}\ndataset: {type: inline, items: []}\n")
    return config_path


def test_cmd_run_sync_table_output_tracking_and_artifacts(tmp_path, capsys):
    config_path = _write_config(tmp_path)
    run_dir = tmp_path / "run-dir"
    output_file = tmp_path / "results.json"

    tracker = MagicMock()

    def _fake_run_experiment(_config_path, progress_callback=None, **kwargs):
        if progress_callback is not None:
            progress_callback(1, 6)
            progress_callback(6, 6)

        artifact_dir = Path(kwargs["run_dir"])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "manifest.json",
            "records.jsonl",
            "config.resolved.yaml",
            "summary.json",
            "report.html",
        ):
            (artifact_dir / name).write_text("{}")

        return [
            {"input": "alpha", "status": "success", "latency_ms": 12.0, "output": "A"},
            {"input": "beta", "status": "success", "latency_ms": 8.0, "output": "B"},
            {"input": "gamma", "status": "error", "latency_ms": None, "output": ""},
            {"input": "delta", "status": "success", "latency_ms": 5.0, "output": "D"},
            {"input": "epsilon", "status": "error", "latency_ms": None, "output": ""},
            {"input": "z" * 60, "status": "success", "latency_ms": 15.0, "output": "Z"},
        ]

    def _fake_save_results_json(results, output_path, **_kwargs):
        Path(output_path).write_text("[]")
        assert results

    args = _make_run_args(
        config_path,
        verbose=True,
        quiet=False,
        run_id="run-1",
        run_root=str(tmp_path / "run-root"),
        run_dir=str(run_dir),
        track="tensorboard",
        output=str(output_file),
        format="table",
    )

    with patch(
        "insideLLMs.cli.commands.run.run_experiment_from_config",
        side_effect=_fake_run_experiment,
    ), patch(
        "insideLLMs.cli.commands.run.save_results_json",
        side_effect=_fake_save_results_json,
    ), patch(
        "insideLLMs.experiment_tracking.create_tracker",
        return_value=tracker,
    ), patch("insideLLMs.experiment_tracking.TrackingConfig"):
        rc = cmd_run(args)

    captured = capsys.readouterr()
    assert rc == 0
    assert "Results Summary" in captured.out
    assert "Sample Results" in captured.out
    assert output_file.exists()

    tracker.start_run.assert_called_once_with(run_name="run-1", run_id="run-1")
    tracker.log_metrics.assert_called_once()
    tracker.end_run.assert_called_once_with(status="finished")
    logged_artifacts = [call.kwargs.get("artifact_name") for call in tracker.log_artifact.call_args_list]
    assert "manifest.json" in logged_artifacts
    assert output_file.name in logged_artifacts


def test_cmd_run_async_json_prints_hints_to_stderr(tmp_path, capsys):
    config_path = _write_config(tmp_path)

    async def _fake_async_run(_config_path, **kwargs):
        run_dir = Path(kwargs["run_root"]) / kwargs["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        return [{"input": "hello", "output": "world", "status": "success", "latency_ms": 1.0}]

    args = _make_run_args(
        config_path,
        use_async=True,
        run_root=str(tmp_path / "runs"),
        format="json",
        quiet=False,
    )

    with patch(
        "insideLLMs.cli.commands.run.run_experiment_from_config_async",
        side_effect=_fake_async_run,
    ), patch(
        "insideLLMs.cli.commands.run.derive_run_id_from_config_path",
        return_value="derived-run-id",
    ) as derive_run_id:
        rc = cmd_run(args)

    captured = capsys.readouterr()
    assert rc == 0
    derive_run_id.assert_called_once()
    assert "Run written to:" in captured.err
    assert "Validate with: insidellms validate" in captured.err


def test_cmd_run_handles_experiment_result_object(tmp_path, capsys):
    config_path = _write_config(tmp_path)

    experiment_result = ExperimentResult(
        experiment_id="exp-1",
        model_info=ModelInfo(name="Dummy", provider="test", model_id="dummy"),
        probe_name="logic",
        probe_category=ProbeCategory.LOGIC,
        results=[
            ProbeResult(input="q1", output="a1", status=ResultStatus.SUCCESS, latency_ms=10.0),
            ProbeResult(input="q2", output=None, status=ResultStatus.ERROR, error="boom"),
        ],
    )

    args = _make_run_args(
        config_path,
        quiet=True,
        format="summary",
        run_id="experiment-result",
        run_root=str(tmp_path / "runs"),
    )

    with patch(
        "insideLLMs.cli.commands.run.run_experiment_from_config",
        return_value=experiment_result,
    ):
        rc = cmd_run(args)

    captured = capsys.readouterr()
    assert rc == 0
    assert "OK 1/2 successful" in captured.out


def test_cmd_run_tracking_setup_failure_is_nonfatal(tmp_path, capsys):
    config_path = _write_config(tmp_path)

    args = _make_run_args(
        config_path,
        track="wandb",
        quiet=True,
        format="summary",
        run_id="tracking-disabled",
        run_root=str(tmp_path / "runs"),
    )

    with patch(
        "insideLLMs.experiment_tracking.create_tracker",
        side_effect=RuntimeError("tracker unavailable"),
    ), patch("insideLLMs.cli.commands.run.run_experiment_from_config", return_value=[]):
        rc = cmd_run(args)

    captured = capsys.readouterr()
    assert rc == 0
    assert "Tracking disabled" in captured.out


def test_cmd_run_tracker_logging_error_is_nonfatal(tmp_path, capsys):
    config_path = _write_config(tmp_path)

    tracker = MagicMock()
    tracker.log_metrics.side_effect = RuntimeError("metrics exploded")

    args = _make_run_args(
        config_path,
        track="local",
        run_id="tracker-log-error",
        run_dir=str(tmp_path / "run-dir"),
        run_root=str(tmp_path / "runs"),
        quiet=True,
        format="summary",
    )

    with patch("insideLLMs.experiment_tracking.create_tracker", return_value=tracker), patch(
        "insideLLMs.experiment_tracking.TrackingConfig"
    ), patch("insideLLMs.cli.commands.run.run_experiment_from_config", return_value=[]):
        rc = cmd_run(args)

    captured = capsys.readouterr()
    assert rc == 0
    assert "Tracking error" in captured.out


def test_cmd_run_outer_exception_ends_failed_tracker_and_prints_traceback(tmp_path, capsys):
    config_path = _write_config(tmp_path)

    tracker = MagicMock()
    tracker.end_run.side_effect = RuntimeError("end_run failed")

    args = _make_run_args(
        config_path,
        track="local",
        run_id="outer-exception",
        run_root=str(tmp_path / "runs"),
        verbose=True,
    )

    with patch("insideLLMs.experiment_tracking.create_tracker", return_value=tracker), patch(
        "insideLLMs.experiment_tracking.TrackingConfig"
    ), patch(
        "insideLLMs.cli.commands.run.run_experiment_from_config",
        side_effect=RuntimeError("run boom"),
    ), patch("traceback.print_exc") as print_exc:
        rc = cmd_run(args)

    captured = capsys.readouterr()
    assert rc == 1
    assert "Error running experiment" in captured.err
    assert print_exc.called
