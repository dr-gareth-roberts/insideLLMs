"""Experiment tracking integration for insideLLMs.

This module provides integrations with popular experiment tracking platforms
for logging metrics, parameters, and artifacts from LLM evaluation experiments.
It supports multiple backends through a unified interface, making it easy to
switch between platforms or log to multiple destinations simultaneously.

Overview
--------
The module implements a plugin architecture with a common ``ExperimentTracker``
abstract base class and concrete implementations for each platform:

1. **WandBTracker**: Weights & Biases integration with full table/artifact support
2. **MLflowTracker**: MLflow integration with model registry capabilities
3. **TensorBoardTracker**: TensorBoard integration for visualization
4. **LocalFileTracker**: File-based tracking for offline or lightweight usage
5. **MultiTracker**: Composite tracker for logging to multiple backends

All trackers support:
- Starting/ending experiment runs with context manager protocol
- Logging metrics with optional step numbers
- Logging parameters/configuration
- Logging artifacts (files, models)
- Logging ``ExperimentResult`` objects from insideLLMs probes

Examples
--------
Basic usage with Weights & Biases:

>>> from insideLLMs import WandBTracker
>>> tracker = WandBTracker(project="llm-evaluation", entity="my-team")
>>> tracker.start_run("my-experiment")
>>> tracker.log_params({"model": "gpt-4", "temperature": 0.7})
>>> tracker.log_metrics({"accuracy": 0.95, "latency_ms": 150.0})
>>> tracker.log_metrics({"accuracy": 0.96, "latency_ms": 145.0}, step=1)
>>> tracker.end_run()

Using the context manager pattern for automatic cleanup:

>>> with WandBTracker(project="llm-evaluation") as tracker:
...     tracker.log_params({"model": "claude-3-opus"})
...     tracker.log_metrics({"accuracy": 0.92})
...     # Run automatically ends with "finished" status
...     # or "failed" status if an exception occurs

Logging experiment results from insideLLMs probes:

>>> from insideLLMs import LocalFileTracker
>>> from insideLLMs.types import (
...     ProbeExperimentResult, ModelInfo, ProbeCategory, ProbeScore
... )
>>> result = ProbeExperimentResult(
...     experiment_id="exp-001",
...     model_info=ModelInfo("GPT-4", "openai", "gpt-4"),
...     probe_name="factuality_basic",
...     probe_category=ProbeCategory.FACTUALITY,
...     results=[],
...     score=ProbeScore(accuracy=0.92, mean_latency_ms=150.0)
... )
>>> with LocalFileTracker(output_dir="./experiments") as tracker:
...     tracker.log_experiment_result(result, prefix="factuality_")

Logging to multiple backends simultaneously:

>>> from insideLLMs import MultiTracker, WandBTracker, LocalFileTracker
>>> tracker = MultiTracker([
...     WandBTracker(project="my-project"),
...     LocalFileTracker(output_dir="./backup-logs")
... ])
>>> with tracker:
...     tracker.log_metrics({"accuracy": 0.95})  # Logs to both backends

Using the factory function for quick setup:

>>> from insideLLMs import create_tracker
>>> tracker = create_tracker("local", output_dir="./experiments")
>>> tracker = create_tracker("wandb", project="my-project")
>>> tracker = create_tracker("mlflow", tracking_uri="http://localhost:5000")

Auto-tracking function execution:

>>> from insideLLMs import auto_track, LocalFileTracker
>>> @auto_track(LocalFileTracker(output_dir="./runs"))
... def run_evaluation():
...     # Your evaluation code here
...     return {"accuracy": 0.95, "f1": 0.93}
>>> result = run_evaluation()  # Metrics logged automatically

Notes
-----
- Most trackers require their respective libraries to be installed:
  - W&B: ``pip install wandb``
  - MLflow: ``pip install mlflow``
  - TensorBoard: ``pip install tensorboard`` or ``pip install tensorboardX``
- The ``LocalFileTracker`` has no external dependencies
- All trackers are thread-safe for basic operations
- Runs can be nested (for parent-child experiment relationships)
- Artifacts are copied to the tracking backend, not moved

See Also
--------
insideLLMs.types.ExperimentResult : Result container logged by trackers
insideLLMs.types.ProbeScore : Score metrics extracted for logging
TrackingConfig : Configuration dataclass for tracker settings
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

    This dataclass provides a unified configuration interface for all tracker
    implementations. It specifies project organization, run metadata, and
    behavioral settings that apply across backends.

    Parameters
    ----------
    project : str, optional
        Project name for grouping related experiments. Used as the top-level
        organizational unit in most tracking platforms. Default is "insideLLMs".
    experiment_name : str, optional
        Name of the current experiment or run within the project. If not
        specified, trackers will auto-generate names (often timestamps).
    tags : list of str, optional
        Tags for categorizing and filtering experiments. Tags are stored
        as metadata and can be used for search/filtering in most platforms.
    notes : str, optional
        Free-form notes or description for the experiment. Useful for
        documenting the purpose, hypothesis, or context of a run.
    log_artifacts : bool, optional
        Whether to enable artifact logging (files, models, datasets).
        Set to False to reduce storage usage. Default is True.
    log_code : bool, optional
        Whether to automatically log source code with the experiment.
        Useful for reproducibility but may expose sensitive code.
        Default is False.
    auto_log_metrics : bool, optional
        Whether to enable automatic metric logging (e.g., system metrics,
        framework-specific metrics). Default is True.

    Attributes
    ----------
    project : str
        Project name for grouping experiments.
    experiment_name : str or None
        Name of the current experiment/run.
    tags : list of str
        Tags for categorizing the experiment.
    notes : str or None
        Optional notes about the experiment.
    log_artifacts : bool
        Whether to log artifacts (files, models).
    log_code : bool
        Whether to log source code.
    auto_log_metrics : bool
        Whether to auto-log common metrics.

    Examples
    --------
    Creating a basic configuration:

    >>> from insideLLMs import TrackingConfig
    >>> config = TrackingConfig(project="llm-safety-eval")
    >>> print(config.project)
    llm-safety-eval

    Creating a fully configured tracking setup:

    >>> config = TrackingConfig(
    ...     project="production-evals",
    ...     experiment_name="gpt4-factuality-v2",
    ...     tags=["production", "gpt-4", "factuality"],
    ...     notes="Evaluating GPT-4 on updated factuality dataset",
    ...     log_artifacts=True,
    ...     log_code=True,
    ...     auto_log_metrics=True
    ... )
    >>> "production" in config.tags
    True

    Using configuration with a tracker:

    >>> from insideLLMs import LocalFileTracker
    >>> config = TrackingConfig(
    ...     project="my-experiments",
    ...     experiment_name="run-001",
    ...     tags=["baseline"]
    ... )
    >>> tracker = LocalFileTracker(config=config)

    Creating minimal configuration for quick tests:

    >>> config = TrackingConfig(log_artifacts=False, log_code=False)
    >>> config.log_artifacts
    False

    See Also
    --------
    ExperimentTracker : Abstract base class that uses this configuration
    WandBTracker : W&B tracker that accepts TrackingConfig
    MLflowTracker : MLflow tracker that accepts TrackingConfig
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

    This class defines the interface that all tracking backends must implement.
    It provides common functionality for run management, metric/parameter
    logging, and integration with insideLLMs experiment results.

    Subclass this to create custom tracking integrations for platforms not
    natively supported (e.g., Neptune, Comet ML, custom databases).

    Parameters
    ----------
    config : TrackingConfig, optional
        Configuration for the tracker. If not provided, default configuration
        is used with project name "insideLLMs".

    Attributes
    ----------
    config : TrackingConfig
        The tracking configuration.
    _run_active : bool
        Whether a run is currently active.
    _run_id : str or None
        The ID of the current run, if any.
    _step : int
        The current step counter for metric logging.

    Examples
    --------
    Creating a custom tracker implementation:

    >>> from insideLLMs import ExperimentTracker, TrackingConfig
    >>> class ConsoleTracker(ExperimentTracker):
    ...     '''Tracker that prints to console.'''
    ...
    ...     def start_run(self, run_name=None, run_id=None, nested=False):
    ...         print(f"Starting run: {run_name or 'unnamed'}")
    ...         self._run_active = True
    ...         self._run_id = run_id or "console-run"
    ...         return self._run_id
    ...
    ...     def end_run(self, status="finished"):
    ...         print(f"Ending run with status: {status}")
    ...         self._run_active = False
    ...
    ...     def log_metrics(self, metrics, step=None):
    ...         print(f"Step {step}: {metrics}")
    ...
    ...     def log_params(self, params):
    ...         print(f"Params: {params}")
    ...
    ...     def log_artifact(self, artifact_path, artifact_name=None, artifact_type=None):
    ...         print(f"Artifact: {artifact_path}")

    Using the custom tracker:

    >>> tracker = ConsoleTracker()
    >>> tracker.start_run("my-test")
    Starting run: my-test
    'console-run'
    >>> tracker.log_metrics({"accuracy": 0.95})
    Step None: {'accuracy': 0.95}

    Using as context manager:

    >>> from insideLLMs import LocalFileTracker
    >>> with LocalFileTracker() as tracker:
    ...     tracker.log_metrics({"loss": 0.5})
    ...     # Run ends automatically

    Logging ExperimentResult objects:

    >>> from insideLLMs.types import (
    ...     ProbeExperimentResult, ModelInfo, ProbeCategory, ProbeScore
    ... )
    >>> # Assuming tracker is initialized
    >>> result = ProbeExperimentResult(
    ...     experiment_id="exp-001",
    ...     model_info=ModelInfo("GPT-4", "openai", "gpt-4"),
    ...     probe_name="logic_test",
    ...     probe_category=ProbeCategory.LOGIC,
    ...     results=[],
    ...     score=ProbeScore(accuracy=0.88)
    ... )
    >>> # tracker.log_experiment_result(result)  # Would log to tracker

    See Also
    --------
    WandBTracker : Weights & Biases implementation
    MLflowTracker : MLflow implementation
    TensorBoardTracker : TensorBoard implementation
    LocalFileTracker : Local file-based implementation
    MultiTracker : Composite tracker for multiple backends
    """

    def __init__(self, config: Optional[TrackingConfig] = None):
        """Initialize the tracker.

        Parameters
        ----------
        config : TrackingConfig, optional
            Tracking configuration. If not provided, uses default configuration
            with project name "insideLLMs".

        Examples
        --------
        >>> from insideLLMs import TrackingConfig, LocalFileTracker
        >>> config = TrackingConfig(project="my-project")
        >>> tracker = LocalFileTracker(config=config)
        >>> tracker.config.project
        'my-project'

        >>> tracker = LocalFileTracker()  # Uses defaults
        >>> tracker.config.project
        'insideLLMs'
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

        This method initializes a new experiment run in the tracking backend.
        All subsequent log calls will be associated with this run until
        ``end_run()`` is called.

        Parameters
        ----------
        run_name : str, optional
            Human-readable name for the run. If not provided, falls back to
            ``config.experiment_name`` or an auto-generated name.
        run_id : str, optional
            Specific run ID to use. If provided, may resume an existing run
            (backend-dependent). If not provided, a new unique ID is generated.
        nested : bool, optional
            Whether this run is nested under a parent run. Useful for
            hyperparameter sweeps or multi-stage experiments. Default is False.

        Returns
        -------
        str
            The unique run ID assigned to this run.

        Raises
        ------
        RuntimeError
            If a run is already active and nested=False (backend-dependent).

        Examples
        --------
        Starting a simple run:

        >>> tracker = LocalFileTracker()
        >>> run_id = tracker.start_run("experiment-001")
        >>> print(f"Started run: {run_id}")  # doctest: +SKIP
        Started run: experiment-001

        Starting a run with specific ID:

        >>> run_id = tracker.start_run(run_id="custom-id-123")  # doctest: +SKIP
        >>> run_id  # doctest: +SKIP
        'custom-id-123'

        Starting a nested run:

        >>> parent_id = tracker.start_run("parent-run")  # doctest: +SKIP
        >>> child_id = tracker.start_run("child-run", nested=True)  # doctest: +SKIP

        See Also
        --------
        end_run : End the current run
        """
        pass

    @abstractmethod
    def end_run(self, status: str = "finished") -> None:
        """End the current run.

        This method finalizes the current experiment run and flushes any
        buffered data to the tracking backend. After calling this method,
        a new run must be started before logging additional data.

        Parameters
        ----------
        status : str, optional
            Final status of the run. Common values are:

            - "finished": Run completed successfully (default)
            - "failed": Run failed due to an error
            - "killed": Run was manually terminated

        Raises
        ------
        RuntimeError
            If no run is currently active (backend-dependent).

        Examples
        --------
        Ending a successful run:

        >>> tracker = LocalFileTracker()
        >>> tracker.start_run("my-run")  # doctest: +SKIP
        >>> tracker.log_metrics({"accuracy": 0.95})  # doctest: +SKIP
        >>> tracker.end_run()  # Status defaults to "finished"

        Ending a failed run:

        >>> try:
        ...     tracker.start_run("risky-run")  # doctest: +SKIP
        ...     # ... code that might fail
        ...     raise ValueError("Something went wrong")  # doctest: +SKIP
        ... except Exception:
        ...     tracker.end_run(status="failed")  # doctest: +SKIP

        Using context manager (preferred):

        >>> with LocalFileTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_metrics({"loss": 0.5})
        ...     # end_run called automatically with appropriate status

        See Also
        --------
        start_run : Start a new run
        __exit__ : Context manager exit (calls end_run)
        """
        pass

    @abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to the tracker.

        This method records numeric metrics at a given step. Metrics are
        typically used for tracking training progress, evaluation scores,
        or any time-series data.

        Parameters
        ----------
        metrics : dict of str to float
            Dictionary mapping metric names to their numeric values.
            All values should be numeric (int or float).
        step : int, optional
            Step number for this metric recording. If not provided, uses
            and increments an internal counter. Steps are used for
            time-series visualization.

        Raises
        ------
        RuntimeError
            If no run is currently active.

        Examples
        --------
        Logging basic metrics:

        >>> with LocalFileTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_metrics({"accuracy": 0.95, "loss": 0.05})

        Logging metrics with explicit steps:

        >>> with LocalFileTracker() as tracker:  # doctest: +SKIP
        ...     for epoch in range(3):
        ...         loss = 1.0 / (epoch + 1)
        ...         tracker.log_metrics({"loss": loss}, step=epoch)

        Logging multiple metric groups:

        >>> with LocalFileTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_metrics({"train/loss": 0.5, "train/accuracy": 0.8})
        ...     tracker.log_metrics({"val/loss": 0.6, "val/accuracy": 0.75})

        See Also
        --------
        log_params : Log configuration parameters
        log_experiment_result : Log complete experiment results
        """
        pass

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters/config to the tracker.

        This method records configuration parameters or hyperparameters for
        the experiment. Unlike metrics, parameters are typically logged once
        and do not have associated steps.

        Parameters
        ----------
        params : dict of str to Any
            Dictionary mapping parameter names to their values. Values can
            be any JSON-serializable type (strings, numbers, lists, dicts).

        Raises
        ------
        RuntimeError
            If no run is currently active.

        Examples
        --------
        Logging model configuration:

        >>> with LocalFileTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_params({
        ...         "model": "gpt-4",
        ...         "temperature": 0.7,
        ...         "max_tokens": 1000
        ...     })

        Logging nested configuration:

        >>> with LocalFileTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_params({
        ...         "model_config": {
        ...             "name": "gpt-4",
        ...             "provider": "openai"
        ...         },
        ...         "eval_config": {
        ...             "num_samples": 100,
        ...             "timeout": 30
        ...         }
        ...     })

        Logging incrementally:

        >>> with LocalFileTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_params({"model": "gpt-4"})
        ...     tracker.log_params({"dataset": "factuality-v2"})

        See Also
        --------
        log_metrics : Log numeric metrics
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

        This method uploads or copies an artifact file to the tracking backend.
        Artifacts can be any files: model weights, datasets, visualizations,
        configuration files, etc.

        Parameters
        ----------
        artifact_path : str
            Path to the artifact file on the local filesystem.
        artifact_name : str, optional
            Name for the artifact in the tracking system. If not provided,
            uses the filename from artifact_path.
        artifact_type : str, optional
            Type/category of the artifact. Common types include:

            - "model": Model weights or checkpoints
            - "dataset": Data files
            - "config": Configuration files
            - "visualization": Charts, plots, images
            - "file": Generic file (default)

        Raises
        ------
        RuntimeError
            If no run is currently active.
        FileNotFoundError
            If the artifact_path does not exist.

        Examples
        --------
        Logging a model file:

        >>> with LocalFileTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_artifact(
        ...         "models/best_model.pt",
        ...         artifact_name="best_model",
        ...         artifact_type="model"
        ...     )

        Logging results file:

        >>> with LocalFileTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_artifact("results.json")

        Logging multiple artifacts:

        >>> with LocalFileTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_artifact("config.yaml", artifact_type="config")
        ...     tracker.log_artifact("metrics.csv", artifact_type="data")
        ...     tracker.log_artifact("plot.png", artifact_type="visualization")

        See Also
        --------
        log_params : For logging configuration values directly
        """
        pass

    def log_experiment_result(
        self,
        result: ExperimentResult,
        prefix: str = "",
    ) -> None:
        """Log an ExperimentResult from insideLLMs.

        This convenience method extracts metrics and parameters from an
        insideLLMs ``ExperimentResult`` (or ``ProbeExperimentResult``) and
        logs them to the tracker. It handles score metrics, timing info,
        and model metadata automatically.

        Parameters
        ----------
        result : ExperimentResult
            The experiment result to log. This is typically the output from
            running a probe against a model.
        prefix : str, optional
            Prefix to add to all metric and parameter names. Useful for
            organizing results when logging multiple experiments to the
            same run. Default is empty string.

        Examples
        --------
        Logging a single experiment result:

        >>> from insideLLMs.types import (
        ...     ProbeExperimentResult, ModelInfo, ProbeCategory, ProbeScore
        ... )
        >>> result = ProbeExperimentResult(
        ...     experiment_id="exp-001",
        ...     model_info=ModelInfo("GPT-4", "openai", "gpt-4"),
        ...     probe_name="factuality_basic",
        ...     probe_category=ProbeCategory.FACTUALITY,
        ...     results=[],
        ...     score=ProbeScore(accuracy=0.92, mean_latency_ms=150.0),
        ...     duration_seconds=45.0
        ... )
        >>> with LocalFileTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_experiment_result(result)

        Logging with prefix for multiple experiments:

        >>> with LocalFileTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_experiment_result(factuality_result, prefix="factuality_")
        ...     tracker.log_experiment_result(logic_result, prefix="logic_")
        ...     tracker.log_experiment_result(bias_result, prefix="bias_")

        What gets logged:

        - Metrics: success_rate, total_count, success_count, error_count
        - Score metrics (if available): accuracy, precision, recall, f1_score,
          mean_latency_ms, total_tokens, error_rate
        - Duration (if available): duration_seconds
        - Parameters: experiment_id, model_name, model_provider, probe_name,
          probe_category

        See Also
        --------
        log_metrics : Log individual metrics
        log_params : Log individual parameters
        insideLLMs.types.ExperimentResult : Result container structure
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
        """Convert ProbeScore to metrics dict.

        This internal method extracts non-None metrics from a ProbeScore
        object and returns them as a flat dictionary suitable for logging.

        Parameters
        ----------
        score : ProbeScore
            The score object containing metrics.
        prefix : str, optional
            Prefix to add to metric names. Default is empty string.

        Returns
        -------
        dict of str to float
            Dictionary of metric names to values, excluding None values.

        Examples
        --------
        >>> from insideLLMs.types import ProbeScore
        >>> tracker = LocalFileTracker()
        >>> score = ProbeScore(accuracy=0.95, mean_latency_ms=150.0)
        >>> metrics = tracker._score_to_metrics(score, prefix="test_")
        >>> "test_accuracy" in metrics  # doctest: +SKIP
        True
        """
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
        """Context manager entry.

        Starts a new run when entering the context. The run name is taken
        from ``config.experiment_name`` if set.

        Returns
        -------
        ExperimentTracker
            Self, allowing method calls on the tracker within the context.

        Examples
        --------
        >>> with LocalFileTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_metrics({"accuracy": 0.95})
        """
        self.start_run(run_name=self.config.experiment_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit.

        Ends the run with "failed" status if an exception occurred,
        otherwise ends with "finished" status.

        Parameters
        ----------
        exc_type : type or None
            Exception type if an exception was raised.
        exc_val : BaseException or None
            Exception instance if an exception was raised.
        exc_tb : traceback or None
            Traceback if an exception was raised.

        Examples
        --------
        >>> with LocalFileTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_metrics({"accuracy": 0.95})
        ...     # Ends with "finished"

        >>> with LocalFileTracker() as tracker:  # doctest: +SKIP
        ...     raise ValueError("Error!")
        ...     # Ends with "failed"
        """
        status = "failed" if exc_type else "finished"
        self.end_run(status=status)


class WandBTracker(ExperimentTracker):
    """Weights & Biases experiment tracker.

    This tracker integrates with Weights & Biases (wandb.ai) for comprehensive
    experiment tracking. It supports all W&B features including tables,
    artifacts, model watching, and team collaboration.

    Parameters
    ----------
    project : str, optional
        W&B project name. Projects group related runs. Default is "insideLLMs".
    entity : str, optional
        W&B entity (team or username). If not specified, uses the default
        entity from W&B configuration.
    config : TrackingConfig, optional
        Tracking configuration. Project name from config is overridden by
        the explicit ``project`` parameter.
    **wandb_kwargs
        Additional keyword arguments passed directly to ``wandb.init()``.
        See W&B documentation for available options.

    Attributes
    ----------
    entity : str or None
        The W&B entity (team/user).
    wandb_kwargs : dict
        Additional W&B initialization arguments.
    _run : wandb.Run or None
        The current W&B run object.

    Examples
    --------
    Basic usage:

    >>> from insideLLMs import WandBTracker
    >>> tracker = WandBTracker(project="llm-eval")
    >>> with tracker:  # doctest: +SKIP
    ...     tracker.log_metrics({"accuracy": 0.95})
    ...     tracker.log_params({"model": "gpt-4"})

    With team entity:

    >>> tracker = WandBTracker(
    ...     project="production-evals",
    ...     entity="my-team"
    ... )  # doctest: +SKIP

    With additional W&B options:

    >>> tracker = WandBTracker(
    ...     project="llm-eval",
    ...     mode="offline",  # Run offline
    ...     group="hyperparameter-search",
    ...     job_type="evaluation"
    ... )  # doctest: +SKIP

    Logging tables:

    >>> with WandBTracker(project="llm-eval") as tracker:  # doctest: +SKIP
    ...     results = [
    ...         {"question": "Q1", "answer": "A1", "correct": True},
    ...         {"question": "Q2", "answer": "A2", "correct": False}
    ...     ]
    ...     tracker.log_table("results", results)

    Watching a model:

    >>> import torch.nn as nn  # doctest: +SKIP
    >>> model = nn.Linear(10, 2)  # doctest: +SKIP
    >>> with WandBTracker(project="training") as tracker:  # doctest: +SKIP
    ...     tracker.watch_model(model, log_freq=100)
    ...     # Training loop here

    Raises
    ------
    ImportError
        If wandb is not installed.

    See Also
    --------
    MLflowTracker : Alternative for MLflow users
    LocalFileTracker : For offline tracking without dependencies
    MultiTracker : To log to W&B and another backend simultaneously
    """

    def __init__(
        self,
        project: str = "insideLLMs",
        entity: Optional[str] = None,
        config: Optional[TrackingConfig] = None,
        **wandb_kwargs,
    ):
        """Initialize W&B tracker.

        Parameters
        ----------
        project : str, optional
            W&B project name. Default is "insideLLMs".
        entity : str, optional
            W&B entity (team or user).
        config : TrackingConfig, optional
            Tracking configuration.
        **wandb_kwargs
            Additional kwargs passed to wandb.init.

        Raises
        ------
        ImportError
            If wandb is not installed.

        Examples
        --------
        >>> tracker = WandBTracker(project="my-project")  # doctest: +SKIP
        >>> tracker = WandBTracker(
        ...     project="my-project",
        ...     entity="my-team",
        ...     tags=["production"]
        ... )  # doctest: +SKIP
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
        """Start a W&B run.

        Initializes a new W&B run with the specified name and configuration.
        If a run is already active and nested=False, the existing run is
        finished before starting the new one.

        Parameters
        ----------
        run_name : str, optional
            Name for the run. Falls back to config.experiment_name.
        run_id : str, optional
            Specific run ID for resuming runs.
        nested : bool, optional
            Whether this is a nested run. Default is False.

        Returns
        -------
        str
            The W&B run ID.

        Examples
        --------
        >>> tracker = WandBTracker(project="test")  # doctest: +SKIP
        >>> run_id = tracker.start_run("my-experiment")  # doctest: +SKIP
        >>> print(f"Run ID: {run_id}")  # doctest: +SKIP
        """
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
        """End the W&B run.

        Finalizes the current run and uploads any remaining data to W&B.

        Parameters
        ----------
        status : str, optional
            Final status. "finished" exits with code 0, others with code 1.

        Examples
        --------
        >>> tracker = WandBTracker(project="test")  # doctest: +SKIP
        >>> tracker.start_run("my-run")  # doctest: +SKIP
        >>> tracker.log_metrics({"accuracy": 0.95})  # doctest: +SKIP
        >>> tracker.end_run()  # doctest: +SKIP
        """
        if self._run_active:
            wandb.finish(exit_code=0 if status == "finished" else 1)
            self._run_active = False
            self._run = None

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to W&B.

        Records metrics to the current run. Metrics are displayed as charts
        in the W&B dashboard.

        Parameters
        ----------
        metrics : dict of str to float
            Dictionary of metric names to values.
        step : int, optional
            Step number. If not provided, uses internal counter.

        Raises
        ------
        RuntimeError
            If no run is active.

        Examples
        --------
        >>> with WandBTracker(project="test") as tracker:  # doctest: +SKIP
        ...     tracker.log_metrics({"loss": 0.5, "accuracy": 0.8})
        ...     tracker.log_metrics({"loss": 0.3, "accuracy": 0.9}, step=10)
        """
        if not self._run_active:
            raise RuntimeError("No active run. Call start_run() first.")

        if step is None:
            step = self._step
            self._step += 1

        wandb.log(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params to W&B config.

        Records parameters to the run's configuration. These appear in the
        run's config section in the W&B dashboard.

        Parameters
        ----------
        params : dict of str to Any
            Dictionary of parameter names to values.

        Raises
        ------
        RuntimeError
            If no run is active.

        Examples
        --------
        >>> with WandBTracker(project="test") as tracker:  # doctest: +SKIP
        ...     tracker.log_params({
        ...         "model": "gpt-4",
        ...         "temperature": 0.7,
        ...         "max_tokens": 1000
        ...     })
        """
        if not self._run_active:
            raise RuntimeError("No active run. Call start_run() first.")

        wandb.config.update(params)

    def log_artifact(
        self,
        artifact_path: str,
        artifact_name: Optional[str] = None,
        artifact_type: Optional[str] = None,
    ) -> None:
        """Log an artifact to W&B.

        Uploads a file as a W&B artifact. Artifacts are versioned and can
        be shared across runs and projects.

        Parameters
        ----------
        artifact_path : str
            Path to the file to upload.
        artifact_name : str, optional
            Name for the artifact. Defaults to filename.
        artifact_type : str, optional
            Type of artifact. Defaults to "file".

        Raises
        ------
        RuntimeError
            If no run is active.

        Examples
        --------
        >>> with WandBTracker(project="test") as tracker:  # doctest: +SKIP
        ...     tracker.log_artifact(
        ...         "model.pt",
        ...         artifact_name="best_model",
        ...         artifact_type="model"
        ...     )
        """
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

        Creates and logs a W&B Table for interactive exploration in the
        dashboard. Tables are useful for logging structured data like
        model predictions, dataset samples, or evaluation results.

        Parameters
        ----------
        table_name : str
            Name for the table. This appears as the metric name in W&B.
        data : list of dict
            List of dictionaries, each representing a row. All dictionaries
            should have the same keys.
        columns : list of str, optional
            Column names for the table. If not provided, inferred from the
            first row's keys.

        Raises
        ------
        RuntimeError
            If no run is active.

        Examples
        --------
        >>> with WandBTracker(project="test") as tracker:  # doctest: +SKIP
        ...     predictions = [
        ...         {"input": "2+2", "predicted": "4", "correct": True},
        ...         {"input": "3*3", "predicted": "6", "correct": False},
        ...         {"input": "10/2", "predicted": "5", "correct": True}
        ...     ]
        ...     tracker.log_table("predictions", predictions)

        With explicit columns:

        >>> with WandBTracker(project="test") as tracker:  # doctest: +SKIP
        ...     data = [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}]
        ...     tracker.log_table("results", data, columns=["a", "b"])

        See Also
        --------
        log_metrics : For logging simple numeric values
        log_artifact : For logging files
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

        Enables automatic logging of model gradients and parameters during
        training. This is useful for debugging training dynamics and
        detecting gradient issues.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model to watch.
        log_freq : int, optional
            How often to log gradients/parameters (in batches). Default is 100.

        Raises
        ------
        RuntimeError
            If no run is active.

        Examples
        --------
        >>> import torch.nn as nn  # doctest: +SKIP
        >>> model = nn.Sequential(  # doctest: +SKIP
        ...     nn.Linear(784, 256),
        ...     nn.ReLU(),
        ...     nn.Linear(256, 10)
        ... )
        >>> with WandBTracker(project="training") as tracker:  # doctest: +SKIP
        ...     tracker.watch_model(model, log_freq=50)
        ...     # Training loop...

        See Also
        --------
        log_metrics : For manual metric logging
        """
        if not self._run_active:
            raise RuntimeError("No active run. Call start_run() first.")

        wandb.watch(model, log_freq=log_freq)


class MLflowTracker(ExperimentTracker):
    """MLflow experiment tracker.

    This tracker integrates with MLflow for experiment tracking, model
    registry, and deployment. It supports local tracking, remote tracking
    servers, and cloud deployments (Databricks, Azure ML, etc.).

    Parameters
    ----------
    tracking_uri : str, optional
        URI of the MLflow tracking server. Can be:

        - Local path: "file:./mlruns"
        - HTTP: "http://localhost:5000"
        - Databricks: "databricks"

        If not specified, uses the default from MLFLOW_TRACKING_URI
        environment variable or local "./mlruns" directory.
    experiment_name : str, optional
        Name of the MLflow experiment to use. If the experiment doesn't
        exist, it will be created.
    config : TrackingConfig, optional
        Tracking configuration.

    Attributes
    ----------
    tracking_uri : str or None
        The MLflow tracking URI.
    _experiment_id : str or None
        The MLflow experiment ID.

    Examples
    --------
    Basic local usage:

    >>> from insideLLMs import MLflowTracker
    >>> tracker = MLflowTracker(experiment_name="llm-evaluation")
    >>> with tracker:  # doctest: +SKIP
    ...     tracker.log_metrics({"accuracy": 0.95})
    ...     tracker.log_params({"model": "gpt-4"})

    With tracking server:

    >>> tracker = MLflowTracker(
    ...     tracking_uri="http://localhost:5000",
    ...     experiment_name="production-evals"
    ... )  # doctest: +SKIP

    Logging a model:

    >>> import torch.nn as nn  # doctest: +SKIP
    >>> model = nn.Linear(10, 2)  # doctest: +SKIP
    >>> with MLflowTracker() as tracker:  # doctest: +SKIP
    ...     tracker.log_model(model, "classifier", model_flavor="pytorch")

    Registering a model:

    >>> with MLflowTracker() as tracker:  # doctest: +SKIP
    ...     tracker.log_model(model, "classifier")
    ...     tracker.register_model(
    ...         model_uri=f"runs:/{tracker._run_id}/classifier",
    ...         model_name="production-classifier"
    ...     )

    Raises
    ------
    ImportError
        If mlflow is not installed.

    See Also
    --------
    WandBTracker : Alternative for W&B users
    LocalFileTracker : For simple file-based tracking
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        config: Optional[TrackingConfig] = None,
    ):
        """Initialize MLflow tracker.

        Parameters
        ----------
        tracking_uri : str, optional
            MLflow tracking server URI.
        experiment_name : str, optional
            Name of the MLflow experiment.
        config : TrackingConfig, optional
            Tracking configuration.

        Raises
        ------
        ImportError
            If mlflow is not installed.

        Examples
        --------
        >>> tracker = MLflowTracker()  # doctest: +SKIP
        >>> tracker = MLflowTracker(
        ...     tracking_uri="http://localhost:5000",
        ...     experiment_name="my-experiment"
        ... )  # doctest: +SKIP
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
        """Start an MLflow run.

        Creates a new MLflow run within the configured experiment.

        Parameters
        ----------
        run_name : str, optional
            Name for the run.
        run_id : str, optional
            Existing run ID to resume.
        nested : bool, optional
            Whether this is a nested run.

        Returns
        -------
        str
            The MLflow run ID.

        Examples
        --------
        >>> tracker = MLflowTracker()  # doctest: +SKIP
        >>> run_id = tracker.start_run("evaluation-001")  # doctest: +SKIP
        """
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
        """End the MLflow run.

        Finalizes the current run with the specified status.

        Parameters
        ----------
        status : str, optional
            Final status. Valid values: "finished", "failed", "killed".

        Examples
        --------
        >>> tracker = MLflowTracker()  # doctest: +SKIP
        >>> tracker.start_run()  # doctest: +SKIP
        >>> tracker.end_run(status="finished")  # doctest: +SKIP
        """
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
        """Log metrics to MLflow.

        Records metrics at the specified step.

        Parameters
        ----------
        metrics : dict of str to float
            Dictionary of metric names to values.
        step : int, optional
            Step number for the metrics.

        Raises
        ------
        RuntimeError
            If no run is active.

        Examples
        --------
        >>> with MLflowTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_metrics({"loss": 0.5, "accuracy": 0.8})
        """
        if not self._run_active:
            raise RuntimeError("No active run. Call start_run() first.")

        if step is None:
            step = self._step
            self._step += 1

        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params to MLflow.

        Records parameters for the run. MLflow requires all parameter values
        to be strings, so non-string values are automatically converted.

        Parameters
        ----------
        params : dict of str to Any
            Dictionary of parameter names to values.

        Raises
        ------
        RuntimeError
            If no run is active.

        Examples
        --------
        >>> with MLflowTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_params({"model": "gpt-4", "temperature": 0.7})
        """
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
        """Log an artifact to MLflow.

        Uploads a file to the MLflow artifact store.

        Parameters
        ----------
        artifact_path : str
            Path to the file to upload.
        artifact_name : str, optional
            Not used by MLflow (included for interface compatibility).
        artifact_type : str, optional
            Not used by MLflow (included for interface compatibility).

        Raises
        ------
        RuntimeError
            If no run is active.

        Examples
        --------
        >>> with MLflowTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_artifact("results.json")
        """
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

        Saves a model to the MLflow artifact store. The model can later be
        loaded using MLflow's model loading functions or deployed using
        MLflow Model Serving.

        Parameters
        ----------
        model : Any
            The model object to log. Must be compatible with the specified
            model_flavor.
        model_name : str
            Name for the logged model. Used as the artifact path.
        model_flavor : str, optional
            MLflow model flavor to use. Options include:

            - "pyfunc": Generic Python function model (default)
            - "pytorch": PyTorch model
            - "tensorflow": TensorFlow model

        Raises
        ------
        RuntimeError
            If no run is active.

        Examples
        --------
        >>> import torch.nn as nn  # doctest: +SKIP
        >>> model = nn.Linear(10, 2)  # doctest: +SKIP
        >>> with MLflowTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_model(model, "classifier", model_flavor="pytorch")

        See Also
        --------
        register_model : Register a logged model in the model registry
        log_artifact : For logging arbitrary files
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

        Registers a previously logged model in the MLflow Model Registry,
        enabling versioning, staging, and production deployments.

        Parameters
        ----------
        model_uri : str
            URI of the logged model. Format: "runs:/<run_id>/<artifact_path>".
        model_name : str
            Name for the registered model. This name is used to identify
            the model in the registry.

        Examples
        --------
        >>> with MLflowTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_model(model, "classifier")
        ...     tracker.register_model(
        ...         model_uri=f"runs:/{tracker._run_id}/classifier",
        ...         model_name="production-classifier"
        ...     )

        See Also
        --------
        log_model : Log a model before registering
        """
        mlflow.register_model(model_uri, model_name)


class TensorBoardTracker(ExperimentTracker):
    """TensorBoard experiment tracker.

    This tracker writes metrics and data to TensorBoard format for
    visualization. It supports scalars, histograms, text, and images.

    Parameters
    ----------
    log_dir : str, optional
        Base directory for TensorBoard logs. Each run creates a subdirectory.
        Default is "./runs".
    config : TrackingConfig, optional
        Tracking configuration.

    Attributes
    ----------
    log_dir : str
        The base log directory.
    _writer : SummaryWriter or None
        The TensorBoard SummaryWriter for the current run.

    Examples
    --------
    Basic usage:

    >>> from insideLLMs import TensorBoardTracker
    >>> tracker = TensorBoardTracker(log_dir="./runs/experiment1")
    >>> with tracker:  # doctest: +SKIP
    ...     for step in range(100):
    ...         tracker.log_metrics({"loss": 1.0 / (step + 1)}, step=step)

    Then view with TensorBoard:

    .. code-block:: bash

        tensorboard --logdir ./runs

    Logging histograms:

    >>> import numpy as np  # doctest: +SKIP
    >>> with TensorBoardTracker() as tracker:  # doctest: +SKIP
    ...     values = np.random.randn(1000)
    ...     tracker.log_histogram("weights", values, step=0)

    Raises
    ------
    ImportError
        If neither tensorboard nor tensorboardX is installed.

    See Also
    --------
    WandBTracker : For more advanced visualization
    LocalFileTracker : For simple JSON-based logging
    """

    def __init__(
        self,
        log_dir: str = "./runs",
        config: Optional[TrackingConfig] = None,
    ):
        """Initialize TensorBoard tracker.

        Parameters
        ----------
        log_dir : str, optional
            Directory for TensorBoard logs. Default is "./runs".
        config : TrackingConfig, optional
            Tracking configuration.

        Raises
        ------
        ImportError
            If tensorboard/tensorboardX is not installed.

        Examples
        --------
        >>> tracker = TensorBoardTracker()  # doctest: +SKIP
        >>> tracker = TensorBoardTracker(log_dir="./experiments/tensorboard")  # doctest: +SKIP
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
        """Start a TensorBoard run.

        Creates a new SummaryWriter for the run.

        Parameters
        ----------
        run_name : str, optional
            Name for the run (used as subdirectory name).
        run_id : str, optional
            Not used by TensorBoard (included for interface compatibility).
        nested : bool, optional
            Not used by TensorBoard (included for interface compatibility).

        Returns
        -------
        str
            The run name (used as run ID).

        Examples
        --------
        >>> tracker = TensorBoardTracker()  # doctest: +SKIP
        >>> tracker.start_run("my-experiment")  # doctest: +SKIP
        'my-experiment'
        """
        run_name = run_name or self.config.experiment_name or datetime.now().isoformat()
        run_dir = os.path.join(self.log_dir, run_name)

        self._writer = SummaryWriter(log_dir=run_dir)
        self._run_active = True
        self._run_id = run_name
        self._step = 0

        return self._run_id

    def end_run(self, status: str = "finished") -> None:
        """End the TensorBoard run.

        Closes the SummaryWriter and flushes data to disk.

        Parameters
        ----------
        status : str, optional
            Not used by TensorBoard (included for interface compatibility).

        Examples
        --------
        >>> tracker = TensorBoardTracker()  # doctest: +SKIP
        >>> tracker.start_run()  # doctest: +SKIP
        >>> tracker.end_run()  # doctest: +SKIP
        """
        if self._writer:
            self._writer.close()
            self._writer = None
        self._run_active = False

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to TensorBoard.

        Records scalar metrics that will be displayed as line charts.

        Parameters
        ----------
        metrics : dict of str to float
            Dictionary of metric names to values.
        step : int, optional
            Step number (x-axis value).

        Raises
        ------
        RuntimeError
            If no run is active.

        Examples
        --------
        >>> with TensorBoardTracker() as tracker:  # doctest: +SKIP
        ...     for epoch in range(10):
        ...         tracker.log_metrics({"loss": 1.0 / (epoch + 1)}, step=epoch)
        """
        if not self._run_active or not self._writer:
            raise RuntimeError("No active run. Call start_run() first.")

        if step is None:
            step = self._step
            self._step += 1

        for name, value in metrics.items():
            self._writer.add_scalar(name, value, step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params to TensorBoard as text.

        TensorBoard doesn't have native parameter support, so parameters
        are logged as formatted text.

        Parameters
        ----------
        params : dict of str to Any
            Dictionary of parameter names to values.

        Raises
        ------
        RuntimeError
            If no run is active.

        Examples
        --------
        >>> with TensorBoardTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_params({"model": "gpt-4", "lr": 0.001})
        """
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
        """Log artifact path as text (TensorBoard doesn't support artifacts).

        Since TensorBoard doesn't have native artifact support, the artifact
        path is logged as text for reference.

        Parameters
        ----------
        artifact_path : str
            Path to the artifact file.
        artifact_name : str, optional
            Name for the artifact.
        artifact_type : str, optional
            Type of artifact.

        Raises
        ------
        RuntimeError
            If no run is active.

        Examples
        --------
        >>> with TensorBoardTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_artifact("model.pt")  # Logs path as text
        """
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

        Records a histogram of values for distribution visualization.
        Useful for tracking weight distributions, activation patterns,
        or any numerical distributions.

        Parameters
        ----------
        tag : str
            Name for the histogram.
        values : array-like
            Values to plot. Can be a numpy array, PyTorch tensor, or list.
        step : int, optional
            Step number for the histogram.

        Raises
        ------
        RuntimeError
            If no run is active.

        Examples
        --------
        >>> import numpy as np  # doctest: +SKIP
        >>> with TensorBoardTracker() as tracker:  # doctest: +SKIP
        ...     weights = np.random.randn(1000)
        ...     tracker.log_histogram("layer1/weights", weights, step=0)

        Tracking weight evolution:

        >>> with TensorBoardTracker() as tracker:  # doctest: +SKIP
        ...     for epoch in range(10):
        ...         weights = np.random.randn(1000) * (1 - epoch * 0.05)
        ...         tracker.log_histogram("weights", weights, step=epoch)

        See Also
        --------
        log_metrics : For scalar values
        """
        if not self._run_active or not self._writer:
            raise RuntimeError("No active run. Call start_run() first.")

        if step is None:
            step = self._step

        self._writer.add_histogram(tag, values, step)


class LocalFileTracker(ExperimentTracker):
    """Simple local file-based experiment tracker.

    This tracker stores all experiment data as JSON files in the local
    filesystem. It requires no external dependencies and works offline,
    making it useful for development, testing, or environments without
    network access to tracking servers.

    Data is organized in the following structure::

        output_dir/
            project_name/
                run_id/
                    metadata.json     # Run metadata and configuration
                    metrics.json      # All logged metrics with timestamps
                    params.json       # All logged parameters
                    artifacts.json    # List of logged artifacts
                    final_state.json  # Final run status and summary
                    artifacts/        # Copied artifact files
                        artifact1.txt
                        artifact2.pt

    Parameters
    ----------
    output_dir : str, optional
        Base directory for experiment data. Default is "./experiments".
    config : TrackingConfig, optional
        Tracking configuration.

    Attributes
    ----------
    output_dir : Path
        The base output directory.
    _run_dir : Path or None
        Directory for the current run.
    _metrics : list of dict
        Accumulated metrics for the current run.
    _params : dict
        Accumulated parameters for the current run.
    _artifacts : list of dict
        List of logged artifacts for the current run.

    Examples
    --------
    Basic usage:

    >>> from insideLLMs import LocalFileTracker
    >>> tracker = LocalFileTracker(output_dir="./my-experiments")
    >>> with tracker:  # doctest: +SKIP
    ...     tracker.log_params({"model": "gpt-4"})
    ...     tracker.log_metrics({"accuracy": 0.95})
    ...     tracker.log_artifact("results.json")

    Loading previous run data:

    >>> tracker = LocalFileTracker(output_dir="./my-experiments")  # doctest: +SKIP
    >>> data = tracker.load_run("my-experiment-001")  # doctest: +SKIP
    >>> print(data["metrics"])  # doctest: +SKIP

    Listing all runs:

    >>> tracker = LocalFileTracker()  # doctest: +SKIP
    >>> runs = tracker.list_runs()  # doctest: +SKIP
    >>> print(runs)  # doctest: +SKIP

    See Also
    --------
    WandBTracker : For cloud-based tracking with visualization
    MLflowTracker : For enterprise tracking with model registry
    MultiTracker : To combine local and cloud tracking
    """

    def __init__(
        self,
        output_dir: str = "./experiments",
        config: Optional[TrackingConfig] = None,
    ):
        """Initialize local file tracker.

        Parameters
        ----------
        output_dir : str, optional
            Directory for experiment logs. Default is "./experiments".
        config : TrackingConfig, optional
            Tracking configuration.

        Examples
        --------
        >>> tracker = LocalFileTracker()
        >>> tracker = LocalFileTracker(output_dir="/data/experiments")
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
        """Start a local tracking run.

        Creates a new directory for the run and initializes tracking state.

        Parameters
        ----------
        run_name : str, optional
            Name for the run. Defaults to timestamp if not provided.
        run_id : str, optional
            Specific ID for the run. Defaults to run_name if not provided.
        nested : bool, optional
            Not used by LocalFileTracker (included for interface compatibility).

        Returns
        -------
        str
            The run ID (directory name).

        Examples
        --------
        >>> tracker = LocalFileTracker()
        >>> run_id = tracker.start_run("my-experiment")  # doctest: +SKIP
        >>> tracker.end_run()  # doctest: +SKIP
        """
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
        """End the local tracking run.

        Saves all accumulated data to JSON files.

        Parameters
        ----------
        status : str, optional
            Final status of the run.

        Examples
        --------
        >>> tracker = LocalFileTracker()
        >>> tracker.start_run("test")  # doctest: +SKIP
        >>> tracker.log_metrics({"accuracy": 0.9})  # doctest: +SKIP
        >>> tracker.end_run()  # doctest: +SKIP
        """
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
        """Log metrics to local file.

        Accumulates metrics in memory; they are saved to disk when the
        run ends.

        Parameters
        ----------
        metrics : dict of str to float
            Dictionary of metric names to values.
        step : int, optional
            Step number for the metrics.

        Raises
        ------
        RuntimeError
            If no run is active.

        Examples
        --------
        >>> with LocalFileTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_metrics({"loss": 0.5, "accuracy": 0.8})
        ...     tracker.log_metrics({"loss": 0.3}, step=10)
        """
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
        """Log params to local file.

        Accumulates parameters in memory; they are saved to disk when
        the run ends.

        Parameters
        ----------
        params : dict of str to Any
            Dictionary of parameter names to values.

        Raises
        ------
        RuntimeError
            If no run is active.

        Examples
        --------
        >>> with LocalFileTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_params({"model": "gpt-4", "temperature": 0.7})
        """
        if not self._run_active:
            raise RuntimeError("No active run. Call start_run() first.")

        self._params.update(params)

    def log_artifact(
        self,
        artifact_path: str,
        artifact_name: Optional[str] = None,
        artifact_type: Optional[str] = None,
    ) -> None:
        """Log artifact by copying to run directory.

        Copies the artifact file to the run's artifacts subdirectory
        and records metadata.

        Parameters
        ----------
        artifact_path : str
            Path to the artifact file.
        artifact_name : str, optional
            Name for the artifact. Defaults to original filename.
        artifact_type : str, optional
            Type of artifact (for metadata).

        Raises
        ------
        RuntimeError
            If no run is active.

        Examples
        --------
        >>> with LocalFileTracker() as tracker:  # doctest: +SKIP
        ...     tracker.log_artifact("results.json", artifact_type="data")
        ...     tracker.log_artifact("model.pt", artifact_type="model")
        """
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
        """Save data to JSON file.

        Internal helper for writing JSON files with proper formatting.

        Parameters
        ----------
        path : Path
            Path to the output file.
        data : Any
            Data to serialize. Must be JSON-serializable.

        Examples
        --------
        >>> tracker = LocalFileTracker()
        >>> # tracker._save_json(Path("test.json"), {"key": "value"})
        """
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_run(self, run_id: str) -> dict[str, Any]:
        """Load a previous run's data.

        Reads all JSON files from a previous run and returns them as a
        dictionary.

        Parameters
        ----------
        run_id : str
            The run ID to load.

        Returns
        -------
        dict of str to Any
            Dictionary containing:

            - metadata: Run metadata (start time, tags, etc.)
            - metrics: List of metric entries
            - params: Dictionary of parameters
            - artifacts: List of artifact metadata
            - final_state: Final run status

        Raises
        ------
        FileNotFoundError
            If the run directory doesn't exist.

        Examples
        --------
        >>> tracker = LocalFileTracker(output_dir="./experiments")  # doctest: +SKIP
        >>> data = tracker.load_run("experiment-001")  # doctest: +SKIP
        >>> print(data["metadata"]["started_at"])  # doctest: +SKIP
        >>> print(data["params"])  # doctest: +SKIP
        >>> for entry in data["metrics"]:  # doctest: +SKIP
        ...     print(f"Step {entry['step']}: {entry['metrics']}")

        See Also
        --------
        list_runs : Get list of available run IDs
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

        Returns a list of run IDs (directory names) for the configured
        project.

        Returns
        -------
        list of str
            List of run IDs.

        Examples
        --------
        >>> tracker = LocalFileTracker(output_dir="./experiments")  # doctest: +SKIP
        >>> runs = tracker.list_runs()  # doctest: +SKIP
        >>> for run_id in runs:  # doctest: +SKIP
        ...     print(run_id)

        See Also
        --------
        load_run : Load data for a specific run
        """
        project_dir = self.output_dir / self.config.project
        if not project_dir.exists():
            return []

        # Ensure deterministic output order (filesystem iteration order is not guaranteed).
        return sorted(d.name for d in project_dir.iterdir() if d.is_dir())


class MultiTracker(ExperimentTracker):
    """Tracker that logs to multiple backends simultaneously.

    This composite tracker delegates all logging calls to a list of
    underlying trackers, enabling simultaneous logging to multiple
    platforms (e.g., W&B for visualization + local files for backup).

    Parameters
    ----------
    trackers : list of ExperimentTracker
        List of tracker instances to log to.
    config : TrackingConfig, optional
        Tracking configuration. Note that individual trackers may have
        their own configurations.

    Attributes
    ----------
    trackers : list of ExperimentTracker
        The underlying trackers.

    Examples
    --------
    Logging to W&B and local files:

    >>> from insideLLMs import MultiTracker, WandBTracker, LocalFileTracker
    >>> tracker = MultiTracker([
    ...     WandBTracker(project="my-project"),
    ...     LocalFileTracker(output_dir="./backup")
    ... ])  # doctest: +SKIP
    >>> with tracker:  # doctest: +SKIP
    ...     tracker.log_metrics({"accuracy": 0.95})  # Logs to both

    Logging to multiple cloud services:

    >>> tracker = MultiTracker([
    ...     WandBTracker(project="my-project"),
    ...     MLflowTracker(experiment_name="my-experiment")
    ... ])  # doctest: +SKIP

    Partial failures are silently ignored (logs continue to other backends).

    See Also
    --------
    WandBTracker : W&B backend
    MLflowTracker : MLflow backend
    LocalFileTracker : Local file backend
    """

    def __init__(
        self,
        trackers: list[ExperimentTracker],
        config: Optional[TrackingConfig] = None,
    ):
        """Initialize multi-tracker.

        Parameters
        ----------
        trackers : list of ExperimentTracker
            List of trackers to use.
        config : TrackingConfig, optional
            Tracking configuration (applied to all if not set individually).

        Examples
        --------
        >>> from insideLLMs import LocalFileTracker
        >>> tracker1 = LocalFileTracker(output_dir="./logs1")
        >>> tracker2 = LocalFileTracker(output_dir="./logs2")
        >>> multi = MultiTracker([tracker1, tracker2])
        """
        super().__init__(config)
        self.trackers = trackers

    def start_run(
        self,
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        nested: bool = False,
    ) -> str:
        """Start runs on all trackers.

        Calls start_run on each underlying tracker with the same parameters.

        Parameters
        ----------
        run_name : str, optional
            Name for the run.
        run_id : str, optional
            Specific run ID.
        nested : bool, optional
            Whether this is a nested run.

        Returns
        -------
        str
            The run ID from the first tracker.

        Examples
        --------
        >>> multi = MultiTracker([LocalFileTracker(), LocalFileTracker()])
        >>> run_id = multi.start_run("my-experiment")  # doctest: +SKIP
        """
        run_ids = []
        for tracker in self.trackers:
            rid = tracker.start_run(run_name=run_name, run_id=run_id, nested=nested)
            run_ids.append(rid)

        self._run_active = True
        self._run_id = run_ids[0] if run_ids else None
        return self._run_id or ""

    def end_run(self, status: str = "finished") -> None:
        """End runs on all trackers.

        Calls end_run on each underlying tracker.

        Parameters
        ----------
        status : str, optional
            Final status for all runs.

        Examples
        --------
        >>> multi = MultiTracker([LocalFileTracker()])
        >>> multi.start_run()  # doctest: +SKIP
        >>> multi.end_run()  # doctest: +SKIP
        """
        for tracker in self.trackers:
            tracker.end_run(status=status)
        self._run_active = False

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to all trackers.

        Calls log_metrics on each underlying tracker.

        Parameters
        ----------
        metrics : dict of str to float
            Dictionary of metric names to values.
        step : int, optional
            Step number for the metrics.

        Examples
        --------
        >>> with MultiTracker([LocalFileTracker()]) as tracker:  # doctest: +SKIP
        ...     tracker.log_metrics({"accuracy": 0.95})
        """
        for tracker in self.trackers:
            tracker.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params to all trackers.

        Calls log_params on each underlying tracker.

        Parameters
        ----------
        params : dict of str to Any
            Dictionary of parameter names to values.

        Examples
        --------
        >>> with MultiTracker([LocalFileTracker()]) as tracker:  # doctest: +SKIP
        ...     tracker.log_params({"model": "gpt-4"})
        """
        for tracker in self.trackers:
            tracker.log_params(params)

    def log_artifact(
        self,
        artifact_path: str,
        artifact_name: Optional[str] = None,
        artifact_type: Optional[str] = None,
    ) -> None:
        """Log artifact to all trackers.

        Calls log_artifact on each underlying tracker.

        Parameters
        ----------
        artifact_path : str
            Path to the artifact file.
        artifact_name : str, optional
            Name for the artifact.
        artifact_type : str, optional
            Type of artifact.

        Examples
        --------
        >>> with MultiTracker([LocalFileTracker()]) as tracker:  # doctest: +SKIP
        ...     tracker.log_artifact("model.pt")
        """
        for tracker in self.trackers:
            tracker.log_artifact(artifact_path, artifact_name, artifact_type)


def create_tracker(
    backend: str = "local",
    **kwargs,
) -> ExperimentTracker:
    """Factory function to create a tracker.

    This convenience function creates and returns an appropriate tracker
    instance based on the specified backend name. It simplifies tracker
    creation by hiding the specific class names.

    Parameters
    ----------
    backend : str, optional
        Tracker backend to use. Options are:

        - "local": LocalFileTracker (default)
        - "wandb": WandBTracker
        - "mlflow": MLflowTracker
        - "tensorboard": TensorBoardTracker

    **kwargs
        Backend-specific arguments passed to the tracker constructor.

    Returns
    -------
    ExperimentTracker
        Configured tracker instance.

    Raises
    ------
    ValueError
        If an unknown backend is specified.
    ImportError
        If the required library for the backend is not installed.

    Examples
    --------
    Creating different trackers:

    >>> from insideLLMs import create_tracker
    >>> tracker = create_tracker("local", output_dir="./experiments")  # doctest: +SKIP
    >>> tracker = create_tracker("wandb", project="my-project")  # doctest: +SKIP
    >>> tracker = create_tracker("mlflow", tracking_uri="http://localhost:5000")  # doctest: +SKIP
    >>> tracker = create_tracker("tensorboard", log_dir="./runs")  # doctest: +SKIP

    Using with context manager:

    >>> with create_tracker("local") as tracker:  # doctest: +SKIP
    ...     tracker.log_metrics({"accuracy": 0.95})

    Dynamic backend selection:

    >>> import os
    >>> backend = os.environ.get("TRACKING_BACKEND", "local")  # doctest: +SKIP
    >>> tracker = create_tracker(backend, project="my-project")  # doctest: +SKIP

    See Also
    --------
    LocalFileTracker : Local file-based tracker
    WandBTracker : Weights & Biases tracker
    MLflowTracker : MLflow tracker
    TensorBoardTracker : TensorBoard tracker
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

    This decorator wraps a function to automatically:
    1. Start a tracking run before execution
    2. Log returned metrics if the function returns a dict
    3. End the run with appropriate status (finished/failed)

    Parameters
    ----------
    tracker : ExperimentTracker
        The tracker instance to use.
    experiment_name : str, optional
        Name for the experiment. If not provided, uses the function name.

    Returns
    -------
    callable
        Decorator function.

    Examples
    --------
    Basic usage:

    >>> from insideLLMs import auto_track, LocalFileTracker
    >>> @auto_track(LocalFileTracker())
    ... def evaluate_model():
    ...     # Your evaluation code
    ...     return {"accuracy": 0.95, "f1": 0.93}
    >>> result = evaluate_model()  # doctest: +SKIP
    >>> # Metrics are logged automatically

    With custom experiment name:

    >>> @auto_track(LocalFileTracker(), experiment_name="gpt4-evaluation")
    ... def run_evaluation():
    ...     return {"accuracy": 0.92}
    >>> run_evaluation()  # doctest: +SKIP

    Error handling:

    >>> @auto_track(LocalFileTracker())
    ... def risky_evaluation():
    ...     raise ValueError("Something went wrong")
    ...     return {"accuracy": 0.0}
    >>> risky_evaluation()  # doctest: +SKIP
    >>> # Run ends with "failed" status, exception is re-raised

    Non-dict returns:

    >>> @auto_track(LocalFileTracker())
    ... def evaluation_with_model():
    ...     model = "trained_model"
    ...     return model  # Not logged (not a dict)
    >>> evaluation_with_model()  # doctest: +SKIP

    Notes
    -----
    - Only numeric values in the returned dict are logged as metrics
    - Non-dict return values are passed through unchanged
    - The original function's return value is always returned
    - Exceptions are re-raised after ending the run with "failed" status

    See Also
    --------
    ExperimentTracker : Base tracker class
    log_metrics : For manual metric logging
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
