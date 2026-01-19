"""Experiment tracking integration for insideLLMs.

This module provides integrations with popular experiment tracking platforms:
- Weights & Biases (W&B)
- MLflow
- TensorBoard
- Local file-based tracking

Example:
    >>> from insideLLMs import ExperimentTracker, WandBTracker
    >>>
    >>> # Using W&B
    >>> tracker = WandBTracker(project="llm-evaluation")
    >>> tracker.start_run("my-experiment")
    >>> tracker.log_metrics({"accuracy": 0.95, "latency_ms": 150.0})
    >>> tracker.end_run()
    >>>
    >>> # Using context manager
    >>> with WandBTracker(project="llm-evaluation") as tracker:
    ...     tracker.log_metrics({"accuracy": 0.95})
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from insideLLMs.types import ExperimentResult, ProbeScore

# Check for optional dependencies
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import mlflow
    from mlflow.entities import Metric, Param

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_AVAILABLE = False


@dataclass
class TrackingConfig:
    """Configuration for experiment tracking.

    Attributes:
        project: Project name for grouping experiments.
        experiment_name: Name of the current experiment/run.
        tags: Tags for categorizing the experiment.
        notes: Optional notes about the experiment.
        log_artifacts: Whether to log artifacts (files, models).
        log_code: Whether to log source code.
        auto_log_metrics: Whether to auto-log common metrics.
    """

    project: str = "insideLLMs"
    experiment_name: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    notes: Optional[str] = None
    log_artifacts: bool = True
    log_code: bool = False
    auto_log_metrics: bool = True


class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking integrations.

    Subclass this to create custom tracking integrations.

    Example:
        >>> class MyTracker(ExperimentTracker):
        ...     def log_metrics(self, metrics, step=None):
        ...         print(f"Metrics: {metrics}")
    """

    def __init__(self, config: Optional[TrackingConfig] = None):
        """Initialize the tracker.

        Args:
            config: Tracking configuration.
        """
        self.config = config or TrackingConfig()
        self._run_active = False
        self._run_id: Optional[str] = None
        self._step = 0

    @abstractmethod
    def start_run(
        self,
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        nested: bool = False,
    ) -> str:
        """Start a new tracking run.

        Args:
            run_name: Name for the run.
            run_id: Specific run ID to resume.
            nested: Whether this is a nested run.

        Returns:
            The run ID.
        """
        pass

    @abstractmethod
    def end_run(self, status: str = "finished") -> None:
        """End the current run.

        Args:
            status: Final status (finished, failed, killed).
        """
        pass

    @abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to the tracker.

        Args:
            metrics: Dictionary of metric names to values.
            step: Optional step number.
        """
        pass

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters/config to the tracker.

        Args:
            params: Dictionary of parameter names to values.
        """
        pass

    @abstractmethod
    def log_artifact(
        self,
        artifact_path: str,
        artifact_name: Optional[str] = None,
        artifact_type: Optional[str] = None,
    ) -> None:
        """Log an artifact file.

        Args:
            artifact_path: Path to the artifact file.
            artifact_name: Name for the artifact.
            artifact_type: Type of artifact (model, data, etc).
        """
        pass

    def log_experiment_result(
        self,
        result: ExperimentResult,
        prefix: str = "",
    ) -> None:
        """Log an ExperimentResult from insideLLMs.

        Args:
            result: The experiment result to log.
            prefix: Prefix for metric names.
        """
        # Log basic metrics
        metrics = {
            f"{prefix}success_rate": result.success_rate,
            f"{prefix}total_count": float(result.total_count),
            f"{prefix}success_count": float(result.success_count),
            f"{prefix}error_count": float(result.error_count),
        }

        # Log score metrics if available
        if result.score:
            score_metrics = self._score_to_metrics(result.score, prefix)
            metrics.update(score_metrics)

        # Log duration if available
        if result.duration_seconds is not None:
            metrics[f"{prefix}duration_seconds"] = result.duration_seconds

        self.log_metrics(metrics)

        # Log params
        params = {
            f"{prefix}experiment_id": result.experiment_id,
            f"{prefix}model_name": result.model_info.name,
            f"{prefix}model_provider": result.model_info.provider,
            f"{prefix}probe_name": result.probe_name,
            f"{prefix}probe_category": result.probe_category.value,
        }
        self.log_params(params)

    def _score_to_metrics(
        self,
        score: ProbeScore,
        prefix: str = "",
    ) -> dict[str, float]:
        """Convert ProbeScore to metrics dict."""
        metrics = {}
        if score.accuracy is not None:
            metrics[f"{prefix}accuracy"] = score.accuracy
        if score.precision is not None:
            metrics[f"{prefix}precision"] = score.precision
        if score.recall is not None:
            metrics[f"{prefix}recall"] = score.recall
        if score.f1_score is not None:
            metrics[f"{prefix}f1_score"] = score.f1_score
        if score.mean_latency_ms is not None:
            metrics[f"{prefix}mean_latency_ms"] = score.mean_latency_ms
        if score.total_tokens is not None:
            metrics[f"{prefix}total_tokens"] = float(score.total_tokens)
        if score.error_rate is not None:
            metrics[f"{prefix}error_rate"] = score.error_rate
        return metrics

    def __enter__(self) -> "ExperimentTracker":
        """Context manager entry."""
        self.start_run(run_name=self.config.experiment_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        status = "failed" if exc_type else "finished"
        self.end_run(status=status)


class WandBTracker(ExperimentTracker):
    """Weights & Biases experiment tracker.

    Example:
        >>> tracker = WandBTracker(project="llm-eval")
        >>> with tracker:
        ...     tracker.log_metrics({"accuracy": 0.95})
        ...     tracker.log_params({"model": "gpt-4"})
    """

    def __init__(
        self,
        project: str = "insideLLMs",
        entity: Optional[str] = None,
        config: Optional[TrackingConfig] = None,
        **wandb_kwargs,
    ):
        """Initialize W&B tracker.

        Args:
            project: W&B project name.
            entity: W&B entity (team or user).
            config: Tracking configuration.
            **wandb_kwargs: Additional kwargs passed to wandb.init.
        """
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is required for WandBTracker. Install with: pip install wandb")

        if config is None:
            config = TrackingConfig(project=project)
        else:
            config.project = project

        super().__init__(config)
        self.entity = entity
        self.wandb_kwargs = wandb_kwargs
        self._run = None

    def start_run(
        self,
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        nested: bool = False,
    ) -> str:
        """Start a W&B run."""
        if self._run_active and not nested:
            wandb.finish()

        self._run = wandb.init(
            project=self.config.project,
            entity=self.entity,
            name=run_name or self.config.experiment_name,
            id=run_id,
            tags=self.config.tags,
            notes=self.config.notes,
            reinit=True,
            **self.wandb_kwargs,
        )

        self._run_active = True
        self._run_id = self._run.id
        self._step = 0

        return self._run_id

    def end_run(self, status: str = "finished") -> None:
        """End the W&B run."""
        if self._run_active:
            wandb.finish(exit_code=0 if status == "finished" else 1)
            self._run_active = False
            self._run = None

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to W&B."""
        if not self._run_active:
            raise RuntimeError("No active run. Call start_run() first.")

        if step is None:
            step = self._step
            self._step += 1

        wandb.log(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params to W&B config."""
        if not self._run_active:
            raise RuntimeError("No active run. Call start_run() first.")

        wandb.config.update(params)

    def log_artifact(
        self,
        artifact_path: str,
        artifact_name: Optional[str] = None,
        artifact_type: Optional[str] = None,
    ) -> None:
        """Log an artifact to W&B."""
        if not self._run_active:
            raise RuntimeError("No active run. Call start_run() first.")

        artifact_type = artifact_type or "file"
        artifact_name = artifact_name or Path(artifact_path).name

        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)

    def log_table(
        self,
        table_name: str,
        data: list[dict[str, Any]],
        columns: Optional[list[str]] = None,
    ) -> None:
        """Log a table to W&B.

        Args:
            table_name: Name for the table.
            data: List of dictionaries with row data.
            columns: Column names (inferred if not provided).
        """
        if not self._run_active:
            raise RuntimeError("No active run. Call start_run() first.")

        if not data:
            return

        columns = columns or list(data[0].keys())
        table = wandb.Table(columns=columns)

        for row in data:
            table.add_data(*[row.get(c) for c in columns])

        wandb.log({table_name: table})

    def watch_model(self, model: Any, log_freq: int = 100) -> None:
        """Watch a model for gradient/parameter logging.

        Args:
            model: PyTorch model to watch.
            log_freq: How often to log.
        """
        if not self._run_active:
            raise RuntimeError("No active run. Call start_run() first.")

        wandb.watch(model, log_freq=log_freq)


class MLflowTracker(ExperimentTracker):
    """MLflow experiment tracker.

    Example:
        >>> tracker = MLflowTracker(tracking_uri="http://localhost:5000")
        >>> with tracker:
        ...     tracker.log_metrics({"accuracy": 0.95})
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        config: Optional[TrackingConfig] = None,
    ):
        """Initialize MLflow tracker.

        Args:
            tracking_uri: MLflow tracking server URI.
            experiment_name: Name of the MLflow experiment.
            config: Tracking configuration.
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "mlflow is required for MLflowTracker. Install with: pip install mlflow"
            )

        if config is None:
            config = TrackingConfig(experiment_name=experiment_name)

        super().__init__(config)

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.tracking_uri = tracking_uri
        self._experiment_id: Optional[str] = None

    def start_run(
        self,
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        nested: bool = False,
    ) -> str:
        """Start an MLflow run."""
        # Set experiment
        exp_name = self.config.experiment_name or self.config.project
        mlflow.set_experiment(exp_name)

        # Start run
        run = mlflow.start_run(
            run_name=run_name,
            run_id=run_id,
            nested=nested,
            tags=dict.fromkeys(self.config.tags, "true"),
            description=self.config.notes,
        )

        self._run_active = True
        self._run_id = run.info.run_id
        self._experiment_id = run.info.experiment_id
        self._step = 0

        return self._run_id

    def end_run(self, status: str = "finished") -> None:
        """End the MLflow run."""
        if self._run_active:
            status_map = {
                "finished": "FINISHED",
                "failed": "FAILED",
                "killed": "KILLED",
            }
            mlflow.end_run(status=status_map.get(status, "FINISHED"))
            self._run_active = False

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to MLflow."""
        if not self._run_active:
            raise RuntimeError("No active run. Call start_run() first.")

        if step is None:
            step = self._step
            self._step += 1

        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params to MLflow."""
        if not self._run_active:
            raise RuntimeError("No active run. Call start_run() first.")

        # Convert non-string values to strings
        str_params = {k: str(v) for k, v in params.items()}
        mlflow.log_params(str_params)

    def log_artifact(
        self,
        artifact_path: str,
        artifact_name: Optional[str] = None,
        artifact_type: Optional[str] = None,
    ) -> None:
        """Log an artifact to MLflow."""
        if not self._run_active:
            raise RuntimeError("No active run. Call start_run() first.")

        mlflow.log_artifact(artifact_path)

    def log_model(
        self,
        model: Any,
        model_name: str,
        model_flavor: str = "pyfunc",
    ) -> None:
        """Log a model to MLflow.

        Args:
            model: The model object.
            model_name: Name for the model.
            model_flavor: MLflow model flavor.
        """
        if not self._run_active:
            raise RuntimeError("No active run. Call start_run() first.")

        if model_flavor == "pytorch":
            mlflow.pytorch.log_model(model, model_name)
        elif model_flavor == "tensorflow":
            mlflow.tensorflow.log_model(model, model_name)
        else:
            mlflow.pyfunc.log_model(model_name, python_model=model)

    def register_model(
        self,
        model_uri: str,
        model_name: str,
    ) -> None:
        """Register a model in the MLflow registry.

        Args:
            model_uri: URI of the logged model.
            model_name: Name for the registered model.
        """
        mlflow.register_model(model_uri, model_name)


class TensorBoardTracker(ExperimentTracker):
    """TensorBoard experiment tracker.

    Example:
        >>> tracker = TensorBoardTracker(log_dir="./runs/exp1")
        >>> with tracker:
        ...     tracker.log_metrics({"loss": 0.5}, step=0)
    """

    def __init__(
        self,
        log_dir: str = "./runs",
        config: Optional[TrackingConfig] = None,
    ):
        """Initialize TensorBoard tracker.

        Args:
            log_dir: Directory for TensorBoard logs.
            config: Tracking configuration.
        """
        if not TENSORBOARD_AVAILABLE:
            raise ImportError(
                "tensorboard or tensorboardX is required. Install with: pip install tensorboard"
            )

        super().__init__(config)
        self.log_dir = log_dir
        self._writer: Optional[SummaryWriter] = None

    def start_run(
        self,
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        nested: bool = False,
    ) -> str:
        """Start a TensorBoard run."""
        run_name = run_name or self.config.experiment_name or datetime.now().isoformat()
        run_dir = os.path.join(self.log_dir, run_name)

        self._writer = SummaryWriter(log_dir=run_dir)
        self._run_active = True
        self._run_id = run_name
        self._step = 0

        return self._run_id

    def end_run(self, status: str = "finished") -> None:
        """End the TensorBoard run."""
        if self._writer:
            self._writer.close()
            self._writer = None
        self._run_active = False

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to TensorBoard."""
        if not self._run_active or not self._writer:
            raise RuntimeError("No active run. Call start_run() first.")

        if step is None:
            step = self._step
            self._step += 1

        for name, value in metrics.items():
            self._writer.add_scalar(name, value, step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params to TensorBoard as text."""
        if not self._run_active or not self._writer:
            raise RuntimeError("No active run. Call start_run() first.")

        # Log params as text
        params_text = "\n".join(f"{k}: {v}" for k, v in params.items())
        self._writer.add_text("params", params_text)

    def log_artifact(
        self,
        artifact_path: str,
        artifact_name: Optional[str] = None,
        artifact_type: Optional[str] = None,
    ) -> None:
        """Log artifact path as text (TensorBoard doesn't support artifacts)."""
        if not self._run_active or not self._writer:
            raise RuntimeError("No active run. Call start_run() first.")

        artifact_name = artifact_name or Path(artifact_path).name
        self._writer.add_text(f"artifact/{artifact_name}", artifact_path)

    def log_histogram(
        self,
        tag: str,
        values: Any,
        step: Optional[int] = None,
    ) -> None:
        """Log a histogram to TensorBoard.

        Args:
            tag: Name for the histogram.
            values: Values to plot.
            step: Step number.
        """
        if not self._run_active or not self._writer:
            raise RuntimeError("No active run. Call start_run() first.")

        if step is None:
            step = self._step

        self._writer.add_histogram(tag, values, step)


class LocalFileTracker(ExperimentTracker):
    """Simple local file-based experiment tracker.

    Useful for offline tracking or when no external service is available.

    Example:
        >>> tracker = LocalFileTracker(output_dir="./experiments")
        >>> with tracker:
        ...     tracker.log_metrics({"accuracy": 0.95})
    """

    def __init__(
        self,
        output_dir: str = "./experiments",
        config: Optional[TrackingConfig] = None,
    ):
        """Initialize local file tracker.

        Args:
            output_dir: Directory for experiment logs.
            config: Tracking configuration.
        """
        super().__init__(config)
        self.output_dir = Path(output_dir)
        self._run_dir: Optional[Path] = None
        self._metrics: list[dict[str, Any]] = []
        self._params: dict[str, Any] = {}
        self._artifacts: list[str] = []

    def start_run(
        self,
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        nested: bool = False,
    ) -> str:
        """Start a local tracking run."""
        run_name = (
            run_name or self.config.experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        run_id = run_id or run_name

        self._run_dir = self.output_dir / self.config.project / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self._run_active = True
        self._run_id = run_id
        self._step = 0
        self._metrics = []
        self._params = {}
        self._artifacts = []

        # Save run metadata
        metadata = {
            "run_id": run_id,
            "run_name": run_name,
            "project": self.config.project,
            "tags": self.config.tags,
            "notes": self.config.notes,
            "started_at": datetime.now().isoformat(),
        }
        self._save_json(self._run_dir / "metadata.json", metadata)

        return run_id

    def end_run(self, status: str = "finished") -> None:
        """End the local tracking run."""
        if self._run_active and self._run_dir:
            # Save final state
            final_state = {
                "status": status,
                "ended_at": datetime.now().isoformat(),
                "total_steps": self._step,
                "metrics_count": len(self._metrics),
            }
            self._save_json(self._run_dir / "final_state.json", final_state)

            # Save all metrics
            self._save_json(self._run_dir / "metrics.json", self._metrics)

            # Save all params
            self._save_json(self._run_dir / "params.json", self._params)

            # Save artifacts list
            self._save_json(self._run_dir / "artifacts.json", self._artifacts)

        self._run_active = False
        self._run_dir = None

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to local file."""
        if not self._run_active:
            raise RuntimeError("No active run. Call start_run() first.")

        if step is None:
            step = self._step
            self._step += 1

        entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        self._metrics.append(entry)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params to local file."""
        if not self._run_active:
            raise RuntimeError("No active run. Call start_run() first.")

        self._params.update(params)

    def log_artifact(
        self,
        artifact_path: str,
        artifact_name: Optional[str] = None,
        artifact_type: Optional[str] = None,
    ) -> None:
        """Log artifact by copying to run directory."""
        if not self._run_active or not self._run_dir:
            raise RuntimeError("No active run. Call start_run() first.")

        import shutil

        artifact_name = artifact_name or Path(artifact_path).name
        artifacts_dir = self._run_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        dest_path = artifacts_dir / artifact_name
        shutil.copy2(artifact_path, dest_path)

        self._artifacts.append(
            {
                "name": artifact_name,
                "type": artifact_type,
                "path": str(dest_path),
                "original_path": artifact_path,
            }
        )

    def _save_json(self, path: Path, data: Any) -> None:
        """Save data to JSON file."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_run(self, run_id: str) -> dict[str, Any]:
        """Load a previous run's data.

        Args:
            run_id: The run ID to load.

        Returns:
            Dictionary with run data.
        """
        run_dir = self.output_dir / self.config.project / run_id

        if not run_dir.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")

        data = {}
        for filename in [
            "metadata.json",
            "metrics.json",
            "params.json",
            "artifacts.json",
            "final_state.json",
        ]:
            filepath = run_dir / filename
            if filepath.exists():
                with open(filepath) as f:
                    key = filename.replace(".json", "")
                    data[key] = json.load(f)

        return data

    def list_runs(self) -> list[str]:
        """List all runs in the project.

        Returns:
            List of run IDs.
        """
        project_dir = self.output_dir / self.config.project
        if not project_dir.exists():
            return []

        return [d.name for d in project_dir.iterdir() if d.is_dir()]


class MultiTracker(ExperimentTracker):
    """Tracker that logs to multiple backends simultaneously.

    Example:
        >>> tracker = MultiTracker([
        ...     WandBTracker(project="my-project"),
        ...     LocalFileTracker(output_dir="./logs"),
        ... ])
        >>> with tracker:
        ...     tracker.log_metrics({"accuracy": 0.95})
    """

    def __init__(
        self,
        trackers: list[ExperimentTracker],
        config: Optional[TrackingConfig] = None,
    ):
        """Initialize multi-tracker.

        Args:
            trackers: List of trackers to use.
            config: Tracking configuration (applied to all if not set individually).
        """
        super().__init__(config)
        self.trackers = trackers

    def start_run(
        self,
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        nested: bool = False,
    ) -> str:
        """Start runs on all trackers."""
        run_ids = []
        for tracker in self.trackers:
            rid = tracker.start_run(run_name=run_name, run_id=run_id, nested=nested)
            run_ids.append(rid)

        self._run_active = True
        self._run_id = run_ids[0] if run_ids else None
        return self._run_id or ""

    def end_run(self, status: str = "finished") -> None:
        """End runs on all trackers."""
        for tracker in self.trackers:
            tracker.end_run(status=status)
        self._run_active = False

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to all trackers."""
        for tracker in self.trackers:
            tracker.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params to all trackers."""
        for tracker in self.trackers:
            tracker.log_params(params)

    def log_artifact(
        self,
        artifact_path: str,
        artifact_name: Optional[str] = None,
        artifact_type: Optional[str] = None,
    ) -> None:
        """Log artifact to all trackers."""
        for tracker in self.trackers:
            tracker.log_artifact(artifact_path, artifact_name, artifact_type)


def create_tracker(
    backend: str = "local",
    **kwargs,
) -> ExperimentTracker:
    """Factory function to create a tracker.

    Args:
        backend: Tracker backend ('wandb', 'mlflow', 'tensorboard', 'local').
        **kwargs: Backend-specific arguments.

    Returns:
        Configured ExperimentTracker instance.

    Example:
        >>> tracker = create_tracker("wandb", project="my-project")
        >>> tracker = create_tracker("local", output_dir="./logs")
    """
    backends = {
        "wandb": WandBTracker,
        "mlflow": MLflowTracker,
        "tensorboard": TensorBoardTracker,
        "local": LocalFileTracker,
    }

    if backend not in backends:
        raise ValueError(f"Unknown backend: {backend}. Choose from: {list(backends.keys())}")

    return backends[backend](**kwargs)


def auto_track(
    tracker: ExperimentTracker,
    experiment_name: Optional[str] = None,
):
    """Decorator to automatically track a function's execution.

    Args:
        tracker: The tracker to use.
        experiment_name: Name for the experiment.

    Example:
        >>> @auto_track(WandBTracker(project="my-project"))
        ... def train_model():
        ...     return {"accuracy": 0.95}
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            name = experiment_name or func.__name__
            tracker.start_run(run_name=name)

            try:
                result = func(*args, **kwargs)

                # If result is a dict of metrics, log them
                if isinstance(result, dict):
                    numeric_metrics = {
                        k: v for k, v in result.items() if isinstance(v, (int, float))
                    }
                    if numeric_metrics:
                        tracker.log_metrics(numeric_metrics)

                tracker.end_run(status="finished")
                return result
            except Exception:
                tracker.end_run(status="failed")
                raise

        return wrapper

    return decorator
