"""Tests for experiment tracking integrations."""

from unittest.mock import patch

import pytest

from insideLLMs.experiment_tracking import (
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


# Helper to create mock experiment results
def create_mock_experiment() -> ExperimentResult:
    """Create a mock experiment result for testing."""
    results = [
        ProbeResult(
            input=f"test input {i}",
            output=f"test output {i}",
            status=ResultStatus.SUCCESS,
            latency_ms=100.0 + i,
        )
        for i in range(10)
    ]

    return ExperimentResult(
        experiment_id="test_exp_001",
        model_info=ModelInfo(
            name="GPT-4",
            provider="openai",
            model_id="gpt-4-turbo",
        ),
        probe_name="LogicProbe",
        probe_category=ProbeCategory.LOGIC,
        results=results,
        score=ProbeScore(
            accuracy=0.9,
            precision=0.85,
            recall=0.88,
            f1_score=0.865,
            mean_latency_ms=105.0,
        ),
    )


class TestTrackingConfig:
    """Tests for TrackingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrackingConfig()
        assert config.project == "insideLLMs"
        assert config.experiment_name is None
        assert config.tags == []
        assert config.log_artifacts is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = TrackingConfig(
            project="my-project",
            experiment_name="test-exp",
            tags=["tag1", "tag2"],
            notes="Test notes",
        )
        assert config.project == "my-project"
        assert config.experiment_name == "test-exp"
        assert len(config.tags) == 2


class TestLocalFileTracker:
    """Tests for LocalFileTracker."""

    def test_start_end_run(self, tmp_path):
        """Test starting and ending a run."""
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        run_id = tracker.start_run(run_name="test-run")

        assert tracker._run_active is True
        assert run_id == "test-run"

        tracker.end_run()
        assert tracker._run_active is False

    def test_log_metrics(self, tmp_path):
        """Test logging metrics."""
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        tracker.start_run(run_name="test-run")

        tracker.log_metrics({"accuracy": 0.95, "loss": 0.05})
        tracker.log_metrics({"accuracy": 0.97, "loss": 0.03}, step=1)

        assert len(tracker._metrics) == 2
        assert tracker._metrics[0]["metrics"]["accuracy"] == 0.95

        tracker.end_run()

    def test_log_params(self, tmp_path):
        """Test logging parameters."""
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        tracker.start_run(run_name="test-run")

        tracker.log_params({"model": "gpt-4", "temperature": 0.7})

        assert tracker._params["model"] == "gpt-4"
        assert tracker._params["temperature"] == 0.7

        tracker.end_run()

    def test_log_artifact(self, tmp_path):
        """Test logging artifacts."""
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        tracker.start_run(run_name="test-run")

        # Create a test artifact
        artifact_path = tmp_path / "test_artifact.txt"
        artifact_path.write_text("test content")

        tracker.log_artifact(str(artifact_path), artifact_name="my-artifact")

        assert len(tracker._artifacts) == 1
        assert tracker._artifacts[0]["name"] == "my-artifact"

        tracker.end_run()

    def test_context_manager(self, tmp_path):
        """Test using tracker as context manager."""
        config = TrackingConfig(experiment_name="context-test")
        tracker = LocalFileTracker(output_dir=str(tmp_path), config=config)

        with tracker:
            tracker.log_metrics({"test": 1.0})

        assert tracker._run_active is False

    def test_log_experiment_result(self, tmp_path):
        """Test logging ExperimentResult."""
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        tracker.start_run(run_name="exp-result-test")

        exp = create_mock_experiment()
        tracker.log_experiment_result(exp, prefix="eval_")

        # Should have logged metrics and params
        assert len(tracker._metrics) > 0
        assert "eval_model_name" in tracker._params

        tracker.end_run()

    def test_load_run(self, tmp_path):
        """Test loading a previous run."""
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        tracker.start_run(run_name="load-test")
        tracker.log_metrics({"accuracy": 0.9})
        tracker.log_params({"model": "test"})
        tracker.end_run()

        # Load the run
        data = tracker.load_run("load-test")

        assert "metadata" in data
        assert "metrics" in data
        assert "params" in data
        assert data["params"]["model"] == "test"

    def test_list_runs(self, tmp_path):
        """Test listing all runs."""
        tracker = LocalFileTracker(output_dir=str(tmp_path))

        # Create multiple runs
        for name in ["run1", "run2", "run3"]:
            tracker.start_run(run_name=name)
            tracker.end_run()

        runs = tracker.list_runs()
        assert len(runs) == 3
        assert "run1" in runs
        assert "run2" in runs
        assert "run3" in runs

    def test_error_without_active_run(self, tmp_path):
        """Test that operations fail without active run."""
        tracker = LocalFileTracker(output_dir=str(tmp_path))

        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_metrics({"test": 1.0})

        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_params({"test": "value"})

    def test_files_saved_on_end(self, tmp_path):
        """Test that files are saved when run ends."""
        tracker = LocalFileTracker(output_dir=str(tmp_path))
        tracker.start_run(run_name="save-test")
        tracker.log_metrics({"accuracy": 0.95})
        tracker.log_params({"model": "gpt-4"})
        tracker.end_run()

        run_dir = tmp_path / "insideLLMs" / "save-test"
        assert (run_dir / "metadata.json").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "params.json").exists()
        assert (run_dir / "final_state.json").exists()


class TestMultiTracker:
    """Tests for MultiTracker."""

    def test_logs_to_all_trackers(self, tmp_path):
        """Test that metrics are logged to all trackers."""
        tracker1 = LocalFileTracker(output_dir=str(tmp_path / "tracker1"))
        tracker2 = LocalFileTracker(output_dir=str(tmp_path / "tracker2"))

        multi = MultiTracker([tracker1, tracker2])
        multi.start_run(run_name="multi-test")

        multi.log_metrics({"accuracy": 0.9})
        multi.log_params({"model": "test"})

        # Both trackers should have the metrics
        assert len(tracker1._metrics) == 1
        assert len(tracker2._metrics) == 1
        assert tracker1._params["model"] == "test"
        assert tracker2._params["model"] == "test"

        multi.end_run()

    def test_context_manager(self, tmp_path):
        """Test MultiTracker as context manager."""
        tracker1 = LocalFileTracker(output_dir=str(tmp_path / "t1"))
        tracker2 = LocalFileTracker(output_dir=str(tmp_path / "t2"))

        config = TrackingConfig(experiment_name="multi-context")
        multi = MultiTracker([tracker1, tracker2], config=config)

        with multi:
            multi.log_metrics({"test": 1.0})

        assert not tracker1._run_active
        assert not tracker2._run_active


class TestCreateTracker:
    """Tests for create_tracker factory function."""

    def test_create_local_tracker(self, tmp_path):
        """Test creating local tracker."""
        tracker = create_tracker("local", output_dir=str(tmp_path))
        assert isinstance(tracker, LocalFileTracker)

    def test_invalid_backend(self):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_tracker("invalid_backend")


class TestAutoTrack:
    """Tests for auto_track decorator."""

    def test_basic_tracking(self, tmp_path):
        """Test auto_track decorator."""
        tracker = LocalFileTracker(output_dir=str(tmp_path))

        @auto_track(tracker, experiment_name="auto-test")
        def compute_metrics():
            return {"accuracy": 0.95, "loss": 0.05}

        result = compute_metrics()

        assert result["accuracy"] == 0.95

        # Verify tracking happened
        runs = tracker.list_runs()
        assert "auto-test" in runs

        data = tracker.load_run("auto-test")
        assert data["final_state"]["status"] == "finished"

    def test_tracking_on_error(self, tmp_path):
        """Test that errors are tracked."""
        tracker = LocalFileTracker(output_dir=str(tmp_path))

        @auto_track(tracker, experiment_name="error-test")
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        # Run should be marked as failed
        data = tracker.load_run("error-test")
        assert data["final_state"]["status"] == "failed"


class TestScoreToMetrics:
    """Tests for _score_to_metrics conversion."""

    def test_full_score_conversion(self, tmp_path):
        """Test converting full ProbeScore to metrics."""
        tracker = LocalFileTracker(output_dir=str(tmp_path))

        score = ProbeScore(
            accuracy=0.9,
            precision=0.85,
            recall=0.88,
            f1_score=0.865,
            mean_latency_ms=100.0,
            total_tokens=1000,
            error_rate=0.1,
        )

        metrics = tracker._score_to_metrics(score, prefix="test_")

        assert metrics["test_accuracy"] == 0.9
        assert metrics["test_precision"] == 0.85
        assert metrics["test_recall"] == 0.88
        assert metrics["test_f1_score"] == 0.865
        assert metrics["test_mean_latency_ms"] == 100.0
        assert metrics["test_total_tokens"] == 1000.0
        assert metrics["test_error_rate"] == 0.1

    def test_partial_score_conversion(self, tmp_path):
        """Test converting partial ProbeScore to metrics."""
        tracker = LocalFileTracker(output_dir=str(tmp_path))

        score = ProbeScore(accuracy=0.9)

        metrics = tracker._score_to_metrics(score)

        assert metrics["accuracy"] == 0.9
        assert "precision" not in metrics
        assert "recall" not in metrics


class TestWandBTracker:
    """Tests for WandBTracker (mocked)."""

    def test_wandb_unavailable(self):
        """Test that WandBTracker raises error when wandb not installed."""
        with patch("insideLLMs.experiment_tracking.WANDB_AVAILABLE", False):
            from insideLLMs.experiment_tracking import WandBTracker

            with pytest.raises(ImportError, match="wandb is required"):
                WandBTracker(project="test")


class TestMLflowTracker:
    """Tests for MLflowTracker (mocked)."""

    def test_mlflow_unavailable(self):
        """Test that MLflowTracker raises error when mlflow not installed."""
        with patch("insideLLMs.experiment_tracking.MLFLOW_AVAILABLE", False):
            from insideLLMs.experiment_tracking import MLflowTracker

            with pytest.raises(ImportError, match="mlflow is required"):
                MLflowTracker()


class TestTensorBoardTracker:
    """Tests for TensorBoardTracker (mocked)."""

    def test_tensorboard_unavailable(self):
        """Test that TensorBoardTracker raises error when not installed."""
        with patch("insideLLMs.experiment_tracking.TENSORBOARD_AVAILABLE", False):
            from insideLLMs.experiment_tracking import TensorBoardTracker

            with pytest.raises(ImportError, match="tensorboard"):
                TensorBoardTracker()
