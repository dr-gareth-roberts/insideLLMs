"""Additional tests for experiment_tracking to raise coverage from ~58% to 90%+.

Focuses on WandBTracker, MLflowTracker, TensorBoardTracker (all mocked),
MultiTracker edge cases, create_tracker factory for all backends,
auto_track decorator edge cases, and error-handling branches.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.experiment_tracking import (
    ExperimentTracker,
    LocalFileTracker,
    MultiTracker,
    TrackingConfig,
    auto_track,
    create_tracker,
)
from insideLLMs.types import (
    ExperimentResult,
    ModelInfo,
    ProbeCategory,
    ProbeResult,
    ProbeScore,
    ResultStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_experiment() -> ExperimentResult:
    """Create a minimal ExperimentResult for testing."""
    results = [
        ProbeResult(
            input="q",
            output="a",
            status=ResultStatus.SUCCESS,
            latency_ms=10.0,
        )
    ]
    return ExperimentResult(
        experiment_id="exp-cov",
        model_info=ModelInfo(name="TestModel", provider="test", model_id="tm-1"),
        probe_name="CovProbe",
        probe_category=ProbeCategory.LOGIC,
        results=results,
        score=ProbeScore(accuracy=0.8, precision=0.75),
    )


def _mock_experiment_with_duration() -> ExperimentResult:
    """Create an ExperimentResult that has duration_seconds set via timestamps."""
    from datetime import datetime, timedelta

    start = datetime(2025, 1, 1, 0, 0, 0)
    results = [
        ProbeResult(
            input="q",
            output="a",
            status=ResultStatus.SUCCESS,
            latency_ms=10.0,
        )
    ]
    return ExperimentResult(
        experiment_id="exp-dur",
        model_info=ModelInfo(name="DurModel", provider="test", model_id="dm-1"),
        probe_name="DurProbe",
        probe_category=ProbeCategory.LOGIC,
        results=results,
        score=None,
        started_at=start,
        completed_at=start + timedelta(seconds=42.5),
    )


def _mock_experiment_no_score_no_duration() -> ExperimentResult:
    """Create an ExperimentResult with no score and no duration."""
    results = [
        ProbeResult(
            input="q",
            output="a",
            status=ResultStatus.SUCCESS,
            latency_ms=10.0,
        )
    ]
    return ExperimentResult(
        experiment_id="exp-no",
        model_info=ModelInfo(name="NoModel", provider="test", model_id="nm-1"),
        probe_name="NoProbe",
        probe_category=ProbeCategory.LOGIC,
        results=results,
        score=None,
    )


# ===================================================================
# WandBTracker
# ===================================================================


class TestWandBTrackerFull:
    """Full coverage for WandBTracker with mocked wandb module."""

    @pytest.fixture(autouse=True)
    def _patch_wandb(self):
        """Make WANDB_AVAILABLE True and provide a mock wandb module."""
        self.mock_wandb = MagicMock()
        # wandb.init returns a run object with an .id attribute
        self.mock_run = MagicMock()
        self.mock_run.id = "wandb-run-123"
        self.mock_wandb.init.return_value = self.mock_run
        self.mock_wandb.Table = MagicMock()
        self.mock_wandb.Artifact = MagicMock()

        with (
            patch("insideLLMs.experiment_tracking.WANDB_AVAILABLE", True),
            patch("insideLLMs.experiment_tracking.wandb", self.mock_wandb, create=True),
        ):
            from insideLLMs.experiment_tracking import WandBTracker

            self.WandBTracker = WandBTracker
            yield

    # -- init ----------------------------------------------------------

    def test_init_default(self):
        tracker = self.WandBTracker()
        assert tracker.config.project == "insideLLMs"
        assert tracker.entity is None
        assert tracker._run is None

    def test_init_with_project_and_entity(self):
        tracker = self.WandBTracker(project="proj", entity="team")
        assert tracker.config.project == "proj"
        assert tracker.entity == "team"

    def test_init_with_config_overrides_project(self):
        cfg = TrackingConfig(project="old", experiment_name="exp1")
        tracker = self.WandBTracker(project="new", config=cfg)
        assert tracker.config.project == "new"
        assert tracker.config.experiment_name == "exp1"

    def test_init_extra_kwargs_stored(self):
        tracker = self.WandBTracker(mode="offline", group="grp")
        assert tracker.wandb_kwargs == {"mode": "offline", "group": "grp"}

    # -- start_run / end_run -------------------------------------------

    def test_start_run_basic(self):
        tracker = self.WandBTracker(project="p")
        run_id = tracker.start_run(run_name="my-run")
        assert run_id == "wandb-run-123"
        assert tracker._run_active is True
        assert tracker._step == 0
        self.mock_wandb.init.assert_called_once()

    def test_start_run_finishes_existing_when_not_nested(self):
        tracker = self.WandBTracker()
        tracker.start_run(run_name="first")
        # Second call should finish the first run
        tracker.start_run(run_name="second")
        self.mock_wandb.finish.assert_called_once()

    def test_start_run_nested_does_not_finish_existing(self):
        tracker = self.WandBTracker()
        tracker.start_run(run_name="parent")
        self.mock_wandb.finish.reset_mock()
        tracker.start_run(run_name="child", nested=True)
        self.mock_wandb.finish.assert_not_called()

    def test_start_run_with_run_id(self):
        tracker = self.WandBTracker()
        tracker.start_run(run_id="resume-id")
        call_kwargs = self.mock_wandb.init.call_args[1]
        assert call_kwargs["id"] == "resume-id"

    def test_start_run_uses_config_experiment_name_as_fallback(self):
        cfg = TrackingConfig(experiment_name="cfg-name", tags=["t1"], notes="n")
        tracker = self.WandBTracker(config=cfg)
        tracker.start_run()
        call_kwargs = self.mock_wandb.init.call_args[1]
        assert call_kwargs["name"] == "cfg-name"
        assert call_kwargs["tags"] == ["t1"]
        assert call_kwargs["notes"] == "n"

    def test_end_run_finished(self):
        tracker = self.WandBTracker()
        tracker.start_run()
        tracker.end_run(status="finished")
        self.mock_wandb.finish.assert_called_with(exit_code=0)
        assert tracker._run_active is False
        assert tracker._run is None

    def test_end_run_failed(self):
        tracker = self.WandBTracker()
        tracker.start_run()
        tracker.end_run(status="failed")
        self.mock_wandb.finish.assert_called_with(exit_code=1)

    def test_end_run_noop_when_inactive(self):
        tracker = self.WandBTracker()
        # end_run without start_run should be harmless (no error)
        tracker.end_run()
        self.mock_wandb.finish.assert_not_called()

    # -- log_metrics ---------------------------------------------------

    def test_log_metrics_with_step(self):
        tracker = self.WandBTracker()
        tracker.start_run()
        tracker.log_metrics({"acc": 0.9}, step=5)
        self.mock_wandb.log.assert_called_once_with({"acc": 0.9}, step=5)

    def test_log_metrics_auto_step_increments(self):
        tracker = self.WandBTracker()
        tracker.start_run()
        tracker.log_metrics({"a": 1.0})
        tracker.log_metrics({"b": 2.0})
        calls = self.mock_wandb.log.call_args_list
        assert calls[0][1]["step"] == 0
        assert calls[1][1]["step"] == 1

    def test_log_metrics_raises_without_run(self):
        tracker = self.WandBTracker()
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_metrics({"x": 1.0})

    # -- log_params ----------------------------------------------------

    def test_log_params(self):
        tracker = self.WandBTracker()
        tracker.start_run()
        tracker.log_params({"lr": 0.01})
        self.mock_wandb.config.update.assert_called_once_with({"lr": 0.01})

    def test_log_params_raises_without_run(self):
        tracker = self.WandBTracker()
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_params({"lr": 0.01})

    # -- log_artifact --------------------------------------------------

    def test_log_artifact(self):
        tracker = self.WandBTracker()
        tracker.start_run()
        tracker.log_artifact("/tmp/model.pt", artifact_name="best", artifact_type="model")
        self.mock_wandb.Artifact.assert_called_once_with("best", type="model")
        artifact_instance = self.mock_wandb.Artifact.return_value
        artifact_instance.add_file.assert_called_once_with("/tmp/model.pt")
        self.mock_wandb.log_artifact.assert_called_once_with(artifact_instance)

    def test_log_artifact_defaults(self):
        tracker = self.WandBTracker()
        tracker.start_run()
        tracker.log_artifact("/data/results.json")
        self.mock_wandb.Artifact.assert_called_once_with("results.json", type="file")

    def test_log_artifact_raises_without_run(self):
        tracker = self.WandBTracker()
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_artifact("/tmp/x.txt")

    # -- log_table -----------------------------------------------------

    def test_log_table_inferred_columns(self):
        tracker = self.WandBTracker()
        tracker.start_run()
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        tracker.log_table("results", data)
        self.mock_wandb.Table.assert_called_once_with(columns=["a", "b"])
        table_inst = self.mock_wandb.Table.return_value
        assert table_inst.add_data.call_count == 2

    def test_log_table_explicit_columns(self):
        tracker = self.WandBTracker()
        tracker.start_run()
        data = [{"a": 1, "b": 2, "c": 3}]
        tracker.log_table("tbl", data, columns=["a", "c"])
        self.mock_wandb.Table.assert_called_once_with(columns=["a", "c"])

    def test_log_table_empty_data(self):
        tracker = self.WandBTracker()
        tracker.start_run()
        tracker.log_table("empty", [])
        self.mock_wandb.Table.assert_not_called()

    def test_log_table_raises_without_run(self):
        tracker = self.WandBTracker()
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_table("t", [{"x": 1}])

    # -- watch_model ---------------------------------------------------

    def test_watch_model(self):
        tracker = self.WandBTracker()
        tracker.start_run()
        model = MagicMock()
        tracker.watch_model(model, log_freq=50)
        self.mock_wandb.watch.assert_called_once_with(model, log_freq=50)

    def test_watch_model_raises_without_run(self):
        tracker = self.WandBTracker()
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.watch_model(MagicMock())

    # -- context manager -----------------------------------------------

    def test_context_manager_finished(self):
        tracker = self.WandBTracker()
        with tracker:
            tracker.log_metrics({"x": 1.0})
        assert tracker._run_active is False

    def test_context_manager_failed_on_exception(self):
        tracker = self.WandBTracker()
        with pytest.raises(ValueError):
            with tracker:
                raise ValueError("boom")
        self.mock_wandb.finish.assert_called_with(exit_code=1)

    # -- log_experiment_result -----------------------------------------

    def test_log_experiment_result(self):
        tracker = self.WandBTracker()
        tracker.start_run()
        exp = _mock_experiment()
        tracker.log_experiment_result(exp, prefix="eval_")
        # Should have called log (for metrics) and config.update (for params)
        assert self.mock_wandb.log.called
        assert self.mock_wandb.config.update.called

    def test_log_experiment_result_with_duration(self):
        tracker = self.WandBTracker()
        tracker.start_run()
        exp = _mock_experiment_with_duration()
        tracker.log_experiment_result(exp)
        logged_metrics = self.mock_wandb.log.call_args[0][0]
        assert "duration_seconds" in logged_metrics

    def test_log_experiment_result_no_score(self):
        tracker = self.WandBTracker()
        tracker.start_run()
        exp = _mock_experiment_no_score_no_duration()  # score=None, no duration
        tracker.log_experiment_result(exp, prefix="p_")
        logged_metrics = self.mock_wandb.log.call_args[0][0]
        assert "p_accuracy" not in logged_metrics  # score is None
        assert "p_duration_seconds" not in logged_metrics  # no duration


# ===================================================================
# MLflowTracker
# ===================================================================


class TestMLflowTrackerFull:
    """Full coverage for MLflowTracker with mocked mlflow module."""

    @pytest.fixture(autouse=True)
    def _patch_mlflow(self):
        self.mock_mlflow = MagicMock()
        # mlflow.start_run returns a run with .info.run_id / .info.experiment_id
        mock_run_info = SimpleNamespace(run_id="mlflow-run-456", experiment_id="exp-789")
        self.mock_mlflow.start_run.return_value = SimpleNamespace(info=mock_run_info)
        # Sub-modules
        self.mock_mlflow.pytorch = MagicMock()
        self.mock_mlflow.tensorflow = MagicMock()
        self.mock_mlflow.pyfunc = MagicMock()

        with (
            patch("insideLLMs.experiment_tracking.MLFLOW_AVAILABLE", True),
            patch("insideLLMs.experiment_tracking.mlflow", self.mock_mlflow, create=True),
        ):
            from insideLLMs.experiment_tracking import MLflowTracker

            self.MLflowTracker = MLflowTracker
            yield

    # -- init ----------------------------------------------------------

    def test_init_default(self):
        tracker = self.MLflowTracker()
        assert tracker.tracking_uri is None
        assert tracker._experiment_id is None

    def test_init_with_tracking_uri(self):
        tracker = self.MLflowTracker(tracking_uri="http://localhost:5000")
        self.mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        assert tracker.tracking_uri == "http://localhost:5000"

    def test_init_with_experiment_name(self):
        tracker = self.MLflowTracker(experiment_name="my-exp")
        assert tracker.config.experiment_name == "my-exp"

    def test_init_with_config(self):
        cfg = TrackingConfig(project="proj", experiment_name="e")
        tracker = self.MLflowTracker(config=cfg)
        assert tracker.config.experiment_name == "e"

    # -- start_run / end_run -------------------------------------------

    def test_start_run(self):
        tracker = self.MLflowTracker()
        run_id = tracker.start_run(run_name="test-run")
        assert run_id == "mlflow-run-456"
        assert tracker._run_active is True
        assert tracker._experiment_id == "exp-789"
        self.mock_mlflow.set_experiment.assert_called_once()

    def test_start_run_uses_project_when_no_experiment_name(self):
        tracker = self.MLflowTracker()
        tracker.config.experiment_name = None
        tracker.start_run()
        self.mock_mlflow.set_experiment.assert_called_with(tracker.config.project)

    def test_start_run_passes_tags_and_notes(self):
        cfg = TrackingConfig(tags=["a", "b"], notes="hello")
        tracker = self.MLflowTracker(config=cfg)
        tracker.start_run(run_name="r", nested=True)
        call_kwargs = self.mock_mlflow.start_run.call_args[1]
        assert call_kwargs["nested"] is True
        assert call_kwargs["tags"] == {"a": "true", "b": "true"}
        assert call_kwargs["description"] == "hello"

    def test_end_run_finished(self):
        tracker = self.MLflowTracker()
        tracker.start_run()
        tracker.end_run(status="finished")
        self.mock_mlflow.end_run.assert_called_once_with(status="FINISHED")
        assert tracker._run_active is False

    def test_end_run_failed(self):
        tracker = self.MLflowTracker()
        tracker.start_run()
        tracker.end_run(status="failed")
        self.mock_mlflow.end_run.assert_called_with(status="FAILED")

    def test_end_run_killed(self):
        tracker = self.MLflowTracker()
        tracker.start_run()
        tracker.end_run(status="killed")
        self.mock_mlflow.end_run.assert_called_with(status="KILLED")

    def test_end_run_unknown_status_defaults_finished(self):
        tracker = self.MLflowTracker()
        tracker.start_run()
        tracker.end_run(status="unknown_status")
        self.mock_mlflow.end_run.assert_called_with(status="FINISHED")

    def test_end_run_noop_when_inactive(self):
        tracker = self.MLflowTracker()
        tracker.end_run()
        self.mock_mlflow.end_run.assert_not_called()

    # -- log_metrics ---------------------------------------------------

    def test_log_metrics_with_step(self):
        tracker = self.MLflowTracker()
        tracker.start_run()
        tracker.log_metrics({"loss": 0.5}, step=10)
        self.mock_mlflow.log_metrics.assert_called_once_with({"loss": 0.5}, step=10)

    def test_log_metrics_auto_step(self):
        tracker = self.MLflowTracker()
        tracker.start_run()
        tracker.log_metrics({"a": 1.0})
        tracker.log_metrics({"b": 2.0})
        calls = self.mock_mlflow.log_metrics.call_args_list
        assert calls[0][1]["step"] == 0
        assert calls[1][1]["step"] == 1

    def test_log_metrics_raises_without_run(self):
        tracker = self.MLflowTracker()
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_metrics({"x": 1.0})

    # -- log_params ----------------------------------------------------

    def test_log_params_converts_to_string(self):
        tracker = self.MLflowTracker()
        tracker.start_run()
        tracker.log_params({"lr": 0.01, "epochs": 10})
        self.mock_mlflow.log_params.assert_called_once_with({"lr": "0.01", "epochs": "10"})

    def test_log_params_raises_without_run(self):
        tracker = self.MLflowTracker()
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_params({"x": "y"})

    # -- log_artifact --------------------------------------------------

    def test_log_artifact(self):
        tracker = self.MLflowTracker()
        tracker.start_run()
        tracker.log_artifact("/tmp/results.json")
        self.mock_mlflow.log_artifact.assert_called_once_with("/tmp/results.json")

    def test_log_artifact_raises_without_run(self):
        tracker = self.MLflowTracker()
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_artifact("/tmp/x.txt")

    # -- log_model -----------------------------------------------------

    def test_log_model_pytorch(self):
        tracker = self.MLflowTracker()
        tracker.start_run()
        model = MagicMock()
        tracker.log_model(model, "mymodel", model_flavor="pytorch")
        self.mock_mlflow.pytorch.log_model.assert_called_once_with(model, "mymodel")

    def test_log_model_tensorflow(self):
        tracker = self.MLflowTracker()
        tracker.start_run()
        model = MagicMock()
        tracker.log_model(model, "tfmodel", model_flavor="tensorflow")
        self.mock_mlflow.tensorflow.log_model.assert_called_once_with(model, "tfmodel")

    def test_log_model_pyfunc_default(self):
        tracker = self.MLflowTracker()
        tracker.start_run()
        model = MagicMock()
        tracker.log_model(model, "pfmodel")
        self.mock_mlflow.pyfunc.log_model.assert_called_once_with("pfmodel", python_model=model)

    def test_log_model_raises_without_run(self):
        tracker = self.MLflowTracker()
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_model(MagicMock(), "m")

    # -- register_model ------------------------------------------------

    def test_register_model(self):
        tracker = self.MLflowTracker()
        tracker.start_run()
        tracker.register_model("runs:/abc/model", "prod-model")
        self.mock_mlflow.register_model.assert_called_once_with("runs:/abc/model", "prod-model")

    # -- context manager -----------------------------------------------

    def test_context_manager_finished(self):
        tracker = self.MLflowTracker()
        with tracker:
            tracker.log_metrics({"x": 1.0})
        assert tracker._run_active is False

    def test_context_manager_failed(self):
        tracker = self.MLflowTracker()
        with pytest.raises(ZeroDivisionError):
            with tracker:
                _ = 1 / 0
        self.mock_mlflow.end_run.assert_called_with(status="FAILED")

    # -- log_experiment_result -----------------------------------------

    def test_log_experiment_result(self):
        tracker = self.MLflowTracker()
        tracker.start_run()
        tracker.log_experiment_result(_mock_experiment(), prefix="p_")
        assert self.mock_mlflow.log_metrics.called
        assert self.mock_mlflow.log_params.called


# ===================================================================
# TensorBoardTracker
# ===================================================================


class TestTensorBoardTrackerFull:
    """Full coverage for TensorBoardTracker with mocked SummaryWriter."""

    @pytest.fixture(autouse=True)
    def _patch_tb(self):
        self.MockWriter = MagicMock()
        self.writer_instance = MagicMock()
        self.MockWriter.return_value = self.writer_instance

        with (
            patch("insideLLMs.experiment_tracking.TENSORBOARD_AVAILABLE", True),
            patch("insideLLMs.experiment_tracking.SummaryWriter", self.MockWriter, create=True),
        ):
            from insideLLMs.experiment_tracking import TensorBoardTracker

            self.TensorBoardTracker = TensorBoardTracker
            yield

    # -- init ----------------------------------------------------------

    def test_init_default(self):
        tracker = self.TensorBoardTracker()
        assert tracker.log_dir == "./runs"
        assert tracker._writer is None

    def test_init_custom_log_dir(self):
        tracker = self.TensorBoardTracker(log_dir="/data/tb")
        assert tracker.log_dir == "/data/tb"

    def test_init_with_config(self):
        cfg = TrackingConfig(project="proj", experiment_name="exp")
        tracker = self.TensorBoardTracker(config=cfg)
        assert tracker.config.experiment_name == "exp"

    # -- start_run / end_run -------------------------------------------

    def test_start_run_with_name(self):
        tracker = self.TensorBoardTracker(log_dir="/tmp/tb")
        run_id = tracker.start_run(run_name="run1")
        assert run_id == "run1"
        assert tracker._run_active is True
        assert tracker._step == 0
        self.MockWriter.assert_called_once_with(log_dir="/tmp/tb/run1")

    def test_start_run_uses_config_name(self):
        cfg = TrackingConfig(experiment_name="cfg-exp")
        tracker = self.TensorBoardTracker(config=cfg)
        run_id = tracker.start_run()
        assert run_id == "cfg-exp"

    def test_start_run_auto_name_when_none(self):
        tracker = self.TensorBoardTracker()
        run_id = tracker.start_run()
        # Should be an ISO datetime string
        assert len(run_id) > 0
        assert tracker._run_active is True

    def test_end_run(self):
        tracker = self.TensorBoardTracker()
        tracker.start_run(run_name="r")
        tracker.end_run()
        self.writer_instance.close.assert_called_once()
        assert tracker._run_active is False
        assert tracker._writer is None

    def test_end_run_when_no_writer(self):
        tracker = self.TensorBoardTracker()
        # Should not raise even if no writer exists
        tracker.end_run()
        assert tracker._run_active is False

    # -- log_metrics ---------------------------------------------------

    def test_log_metrics_with_step(self):
        tracker = self.TensorBoardTracker()
        tracker.start_run(run_name="r")
        tracker.log_metrics({"loss": 0.5, "acc": 0.9}, step=3)
        calls = self.writer_instance.add_scalar.call_args_list
        assert len(calls) == 2
        # Verify the calls include the correct step
        call_args_set = {(c[0][0], c[0][1], c[0][2]) for c in calls}
        assert ("loss", 0.5, 3) in call_args_set
        assert ("acc", 0.9, 3) in call_args_set

    def test_log_metrics_auto_step(self):
        tracker = self.TensorBoardTracker()
        tracker.start_run(run_name="r")
        tracker.log_metrics({"a": 1.0})
        tracker.log_metrics({"b": 2.0})
        calls = self.writer_instance.add_scalar.call_args_list
        assert calls[0][0][2] == 0  # step 0
        assert calls[1][0][2] == 1  # step 1

    def test_log_metrics_raises_without_run(self):
        tracker = self.TensorBoardTracker()
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_metrics({"x": 1.0})

    # -- log_params ----------------------------------------------------

    def test_log_params(self):
        tracker = self.TensorBoardTracker()
        tracker.start_run(run_name="r")
        tracker.log_params({"model": "gpt-4", "temp": 0.7})
        self.writer_instance.add_text.assert_called_once()
        text_arg = self.writer_instance.add_text.call_args[0][1]
        assert "model: gpt-4" in text_arg
        assert "temp: 0.7" in text_arg

    def test_log_params_raises_without_run(self):
        tracker = self.TensorBoardTracker()
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_params({"x": "y"})

    # -- log_artifact --------------------------------------------------

    def test_log_artifact(self):
        tracker = self.TensorBoardTracker()
        tracker.start_run(run_name="r")
        tracker.log_artifact("/data/model.pt", artifact_name="best")
        self.writer_instance.add_text.assert_called_with("artifact/best", "/data/model.pt")

    def test_log_artifact_default_name(self):
        tracker = self.TensorBoardTracker()
        tracker.start_run(run_name="r")
        tracker.log_artifact("/data/results.json")
        self.writer_instance.add_text.assert_called_with(
            "artifact/results.json", "/data/results.json"
        )

    def test_log_artifact_raises_without_run(self):
        tracker = self.TensorBoardTracker()
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_artifact("/tmp/x.txt")

    # -- log_histogram -------------------------------------------------

    def test_log_histogram_with_step(self):
        tracker = self.TensorBoardTracker()
        tracker.start_run(run_name="r")
        values = [1.0, 2.0, 3.0]
        tracker.log_histogram("weights", values, step=5)
        self.writer_instance.add_histogram.assert_called_once_with("weights", values, 5)

    def test_log_histogram_auto_step(self):
        tracker = self.TensorBoardTracker()
        tracker.start_run(run_name="r")
        tracker.log_histogram("w", [1.0])
        self.writer_instance.add_histogram.assert_called_once_with("w", [1.0], 0)

    def test_log_histogram_raises_without_run(self):
        tracker = self.TensorBoardTracker()
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_histogram("w", [1.0])

    # -- context manager -----------------------------------------------

    def test_context_manager_finished(self):
        tracker = self.TensorBoardTracker()
        with tracker:
            tracker.log_metrics({"x": 1.0})
        assert tracker._run_active is False

    def test_context_manager_failed(self):
        tracker = self.TensorBoardTracker()
        with pytest.raises(TypeError):
            with tracker:
                raise TypeError("oops")
        assert tracker._run_active is False

    # -- log_experiment_result -----------------------------------------

    def test_log_experiment_result(self):
        tracker = self.TensorBoardTracker()
        tracker.start_run(run_name="r")
        tracker.log_experiment_result(_mock_experiment())
        assert self.writer_instance.add_scalar.called
        assert self.writer_instance.add_text.called


# ===================================================================
# MultiTracker edge cases
# ===================================================================


class TestMultiTrackerEdgeCases:
    """Additional edge-case tests for MultiTracker."""

    def test_empty_trackers_list(self):
        multi = MultiTracker([])
        run_id = multi.start_run(run_name="empty")
        assert run_id == ""
        multi.end_run()
        assert multi._run_active is False

    def test_log_artifact_to_all(self, tmp_path):
        t1 = LocalFileTracker(output_dir=str(tmp_path / "a"))
        t2 = LocalFileTracker(output_dir=str(tmp_path / "b"))
        multi = MultiTracker([t1, t2])
        multi.start_run(run_name="art-test")

        artifact = tmp_path / "file.txt"
        artifact.write_text("data")
        multi.log_artifact(str(artifact), artifact_name="f")

        assert len(t1._artifacts) == 1
        assert len(t2._artifacts) == 1
        multi.end_run()

    def test_log_experiment_result_to_all(self, tmp_path):
        t1 = LocalFileTracker(output_dir=str(tmp_path / "a"))
        t2 = LocalFileTracker(output_dir=str(tmp_path / "b"))
        multi = MultiTracker([t1, t2])
        multi.start_run(run_name="exp-test")
        multi.log_experiment_result(_mock_experiment())
        assert len(t1._metrics) > 0
        assert len(t2._metrics) > 0
        multi.end_run()

    def test_context_manager_ends_all_on_exception(self, tmp_path):
        t1 = LocalFileTracker(output_dir=str(tmp_path / "a"))
        t2 = LocalFileTracker(output_dir=str(tmp_path / "b"))
        multi = MultiTracker([t1, t2])
        with pytest.raises(RuntimeError):
            with multi:
                raise RuntimeError("fail")
        assert not t1._run_active
        assert not t2._run_active

    def test_partial_failure_in_log_metrics(self, tmp_path):
        """If one tracker fails during log_metrics, the exception propagates."""
        good = LocalFileTracker(output_dir=str(tmp_path / "good"))
        bad = MagicMock(spec=ExperimentTracker)
        bad.start_run.return_value = "bad-id"
        bad.log_metrics.side_effect = RuntimeError("backend down")

        multi = MultiTracker([good, bad])
        multi.start_run(run_name="partial")

        # The exception from the second tracker should propagate
        with pytest.raises(RuntimeError, match="backend down"):
            multi.log_metrics({"x": 1.0})

    def test_run_id_from_first_tracker(self, tmp_path):
        t1 = LocalFileTracker(output_dir=str(tmp_path / "a"))
        t2 = LocalFileTracker(output_dir=str(tmp_path / "b"))
        multi = MultiTracker([t1, t2])
        run_id = multi.start_run(run_name="check-id")
        assert run_id == "check-id"
        assert multi._run_id == "check-id"
        multi.end_run()


# ===================================================================
# create_tracker factory
# ===================================================================


class TestCreateTrackerFactory:
    """Tests for create_tracker with all backends."""

    def test_create_local(self, tmp_path):
        tracker = create_tracker("local", output_dir=str(tmp_path))
        assert isinstance(tracker, LocalFileTracker)

    def test_create_wandb(self):
        with (
            patch("insideLLMs.experiment_tracking.WANDB_AVAILABLE", True),
            patch("insideLLMs.experiment_tracking.wandb", MagicMock(), create=True),
        ):
            from insideLLMs.experiment_tracking import WandBTracker

            tracker = create_tracker("wandb", project="test-proj")
            assert isinstance(tracker, WandBTracker)

    def test_create_mlflow(self):
        with (
            patch("insideLLMs.experiment_tracking.MLFLOW_AVAILABLE", True),
            patch("insideLLMs.experiment_tracking.mlflow", MagicMock(), create=True),
        ):
            from insideLLMs.experiment_tracking import MLflowTracker

            tracker = create_tracker("mlflow", experiment_name="test")
            assert isinstance(tracker, MLflowTracker)

    def test_create_tensorboard(self):
        with (
            patch("insideLLMs.experiment_tracking.TENSORBOARD_AVAILABLE", True),
            patch("insideLLMs.experiment_tracking.SummaryWriter", MagicMock(), create=True),
        ):
            from insideLLMs.experiment_tracking import TensorBoardTracker

            tracker = create_tracker("tensorboard", log_dir="/tmp/tb")
            assert isinstance(tracker, TensorBoardTracker)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            create_tracker("neptune")


# ===================================================================
# auto_track decorator
# ===================================================================


class TestAutoTrackDecorator:
    """Edge-case tests for auto_track."""

    def test_returns_dict_with_numeric_metrics(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))

        @auto_track(tracker, experiment_name="num-metrics")
        def compute():
            return {"acc": 0.95, "loss": 0.05}

        result = compute()
        assert result == {"acc": 0.95, "loss": 0.05}
        data = tracker.load_run("num-metrics")
        assert data["final_state"]["status"] == "finished"

    def test_returns_dict_with_non_numeric_values_only(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))

        @auto_track(tracker, experiment_name="non-num")
        def compute():
            return {"model": "gpt-4", "status": "ok"}

        result = compute()
        assert result == {"model": "gpt-4", "status": "ok"}
        data = tracker.load_run("non-num")
        # No numeric metrics to log, but metrics list may be empty
        assert data["final_state"]["status"] == "finished"

    def test_returns_dict_mixed_values(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))

        @auto_track(tracker, experiment_name="mixed")
        def compute():
            return {"acc": 0.9, "model": "gpt-4", "count": 10}

        result = compute()
        assert result["acc"] == 0.9
        data = tracker.load_run("mixed")
        # Should have logged only numeric keys
        logged_metrics = data["metrics"]
        assert len(logged_metrics) == 1  # one log_metrics call
        assert "acc" in logged_metrics[0]["metrics"]
        assert "count" in logged_metrics[0]["metrics"]
        assert "model" not in logged_metrics[0]["metrics"]

    def test_returns_non_dict(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))

        @auto_track(tracker, experiment_name="non-dict")
        def compute():
            return 42

        result = compute()
        assert result == 42
        data = tracker.load_run("non-dict")
        assert data["final_state"]["status"] == "finished"
        assert len(data["metrics"]) == 0

    def test_returns_none(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))

        @auto_track(tracker, experiment_name="none-return")
        def compute():
            return None

        result = compute()
        assert result is None
        data = tracker.load_run("none-return")
        assert data["final_state"]["status"] == "finished"

    def test_uses_function_name_when_no_experiment_name(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))

        @auto_track(tracker)
        def my_custom_eval():
            return {"score": 0.8}

        my_custom_eval()
        runs = tracker.list_runs()
        assert "my_custom_eval" in runs

    def test_propagates_args(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))

        @auto_track(tracker, experiment_name="args-test")
        def add(a, b):
            return {"sum": a + b}

        result = add(3, 4)
        assert result["sum"] == 7

    def test_propagates_kwargs(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))

        @auto_track(tracker, experiment_name="kwargs-test")
        def greet(name="world"):
            return {"length": len(name)}

        result = greet(name="test")
        assert result["length"] == 4

    def test_error_marks_failed_and_reraises(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))

        @auto_track(tracker, experiment_name="err-test")
        def broken():
            raise TypeError("broken")

        with pytest.raises(TypeError, match="broken"):
            broken()

        data = tracker.load_run("err-test")
        assert data["final_state"]["status"] == "failed"

    def test_empty_dict_return(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))

        @auto_track(tracker, experiment_name="empty-dict")
        def compute():
            return {}

        result = compute()
        assert result == {}
        data = tracker.load_run("empty-dict")
        assert data["final_state"]["status"] == "finished"
        # Empty dict has no numeric metrics, so nothing logged
        assert len(data["metrics"]) == 0


# ===================================================================
# LocalFileTracker additional branches
# ===================================================================


class TestLocalFileTrackerAdditional:
    """Cover remaining branches in LocalFileTracker."""

    def test_log_artifact_without_run_raises(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_artifact("/tmp/file.txt")

    def test_load_run_not_found(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Run not found"):
            tracker.load_run("nonexistent")

    def test_list_runs_empty_project(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        runs = tracker.list_runs()
        assert runs == []

    def test_start_run_with_run_id(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        run_id = tracker.start_run(run_name="name", run_id="custom-id")
        assert run_id == "custom-id"
        tracker.end_run()

    def test_start_run_auto_name(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        run_id = tracker.start_run()  # no name, no config name
        assert len(run_id) > 0  # auto-generated timestamp name
        tracker.end_run()

    def test_end_run_when_not_active(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        # Should not raise
        tracker.end_run()
        assert tracker._run_active is False

    def test_log_metrics_with_explicit_step(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        tracker.start_run(run_name="step-test")
        tracker.log_metrics({"a": 1.0}, step=42)
        assert tracker._metrics[0]["step"] == 42
        # Internal step counter should NOT have been incremented
        assert tracker._step == 0
        tracker.end_run()

    def test_context_manager_exit_on_exception(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        tracker.config.experiment_name = "ctx-fail"
        with pytest.raises(ValueError):
            with tracker:
                raise ValueError("boom")
        data = tracker.load_run("ctx-fail")
        assert data["final_state"]["status"] == "failed"

    def test_log_experiment_result_no_score_no_duration(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        tracker.start_run(run_name="no-score")
        exp = _mock_experiment_no_score_no_duration()  # score=None, no duration
        tracker.log_experiment_result(exp)
        metrics = tracker._metrics[0]["metrics"]
        # No score metrics
        assert "accuracy" not in metrics
        # No duration
        assert "duration_seconds" not in metrics
        tracker.end_run()

    def test_log_experiment_result_with_duration(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        tracker.start_run(run_name="with-dur")
        exp = _mock_experiment_with_duration()  # has started_at/completed_at
        tracker.log_experiment_result(exp)
        metrics = tracker._metrics[0]["metrics"]
        assert "duration_seconds" in metrics
        assert metrics["duration_seconds"] == 42.5
        tracker.end_run()

    def test_log_params_incremental(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        tracker.start_run(run_name="inc-params")
        tracker.log_params({"a": 1})
        tracker.log_params({"b": 2})
        assert tracker._params == {"a": 1, "b": 2}
        tracker.end_run()

    def test_artifact_copied_to_run_dir(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        tracker.start_run(run_name="cp-test")

        src = tmp_path / "source.txt"
        src.write_text("hello")
        tracker.log_artifact(str(src))

        # Verify file was copied
        run_dir = tmp_path / "insideLLMs" / "cp-test" / "artifacts"
        assert (run_dir / "source.txt").exists()
        assert (run_dir / "source.txt").read_text() == "hello"
        tracker.end_run()

    def test_load_run_partial_files(self, tmp_path):
        """If some JSON files are missing, load_run still returns available data."""
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        tracker.start_run(run_name="partial")
        tracker.end_run()

        # Remove one file to test partial loading
        run_dir = tmp_path / "insideLLMs" / "partial"
        (run_dir / "artifacts.json").unlink()

        data = tracker.load_run("partial")
        assert "metadata" in data
        assert "artifacts" not in data


# ===================================================================
# TrackingConfig additional
# ===================================================================


class TestTrackingConfigAdditional:
    """Cover remaining TrackingConfig fields."""

    def test_all_fields(self):
        config = TrackingConfig(
            project="p",
            experiment_name="e",
            tags=["t"],
            notes="n",
            log_artifacts=False,
            log_code=True,
            auto_log_metrics=False,
        )
        assert config.project == "p"
        assert config.experiment_name == "e"
        assert config.tags == ["t"]
        assert config.notes == "n"
        assert config.log_artifacts is False
        assert config.log_code is True
        assert config.auto_log_metrics is False

    def test_default_values(self):
        config = TrackingConfig()
        assert config.log_code is False
        assert config.auto_log_metrics is True
        assert config.notes is None


# ===================================================================
# ExperimentTracker base class (abstract protocol)
# ===================================================================


class TestExperimentTrackerBase:
    """Test the ExperimentTracker abstract base class behavior."""

    def test_enter_calls_start_run_with_config_name(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        tracker.config.experiment_name = "from-config"
        tracker.__enter__()
        assert tracker._run_active is True
        assert tracker._run_id == "from-config"
        tracker.__exit__(None, None, None)

    def test_exit_finished_when_no_exception(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        tracker.__enter__()
        tracker.__exit__(None, None, None)
        assert tracker._run_active is False

    def test_exit_failed_when_exception(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        tracker.config.experiment_name = "exit-fail"
        tracker.__enter__()
        tracker.__exit__(ValueError, ValueError("oops"), None)
        data = tracker.load_run("exit-fail")
        assert data["final_state"]["status"] == "failed"

    def test_score_to_metrics_all_fields(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        score = ProbeScore(
            accuracy=0.9,
            precision=0.85,
            recall=0.88,
            f1_score=0.865,
            mean_latency_ms=100.0,
            total_tokens=500,
            error_rate=0.1,
        )
        m = tracker._score_to_metrics(score, prefix="x_")
        assert m == {
            "x_accuracy": 0.9,
            "x_precision": 0.85,
            "x_recall": 0.88,
            "x_f1_score": 0.865,
            "x_mean_latency_ms": 100.0,
            "x_total_tokens": 500.0,
            "x_error_rate": 0.1,
        }

    def test_score_to_metrics_minimal(self, tmp_path):
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        score = ProbeScore()
        m = tracker._score_to_metrics(score)
        # ProbeScore defaults error_rate to 0.0, so it will be present
        assert "accuracy" not in m
        assert "precision" not in m
        assert "recall" not in m
        assert "f1_score" not in m
        assert "mean_latency_ms" not in m
        assert "total_tokens" not in m
        assert m.get("error_rate") == 0.0


# ===================================================================
# Unavailable library import errors
# ===================================================================


class TestUnavailableLibraries:
    """Test ImportError when optional libraries are not installed."""

    def test_wandb_unavailable(self):
        with patch("insideLLMs.experiment_tracking.WANDB_AVAILABLE", False):
            from insideLLMs.experiment_tracking import WandBTracker

            with pytest.raises(ImportError, match="wandb is required"):
                WandBTracker()

    def test_mlflow_unavailable(self):
        with patch("insideLLMs.experiment_tracking.MLFLOW_AVAILABLE", False):
            from insideLLMs.experiment_tracking import MLflowTracker

            with pytest.raises(ImportError, match="mlflow is required"):
                MLflowTracker()

    def test_tensorboard_unavailable(self):
        with patch("insideLLMs.experiment_tracking.TENSORBOARD_AVAILABLE", False):
            from insideLLMs.experiment_tracking import TensorBoardTracker

            with pytest.raises(ImportError, match="tensorboard"):
                TensorBoardTracker()
