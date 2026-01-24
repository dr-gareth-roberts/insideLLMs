"""
Experiment orchestration utilities for batch LLM experiments.

This module provides a comprehensive framework for running, managing, and analyzing
batch experiments with Large Language Models (LLMs). It supports sequential and
parallel execution, experiment queuing with priority scheduling, grid search over
parameter combinations, A/B comparison experiments, and detailed metrics aggregation.

Overview
--------
The orchestration system is built around several key concepts:

* **ExperimentConfig**: Defines a single experiment with prompt, model, and parameters
* **ExperimentRun**: Records the execution and results of an experiment
* **BatchRunner**: Executes batches of experiments (sync or parallel)
* **AsyncBatchRunner**: Async version for high-concurrency workloads
* **ExperimentQueue**: Priority-based queue for experiment scheduling
* **ExperimentGrid**: Generate experiments from parameter combinations
* **ComparisonExperiment**: A/B testing multiple prompts on same inputs
* **ExperimentOrchestrator**: High-level workflow coordinator

Key Features
------------
* Sequential and parallel batch execution with configurable workers
* Automatic retry on failure with configurable retry limits
* Progress callbacks for real-time monitoring
* Metrics calculation and aggregation across experiments
* Priority-based experiment queuing
* Grid search over models and parameters
* A/B comparison experiments
* Markdown report generation

Examples
--------
Basic batch execution with a simple executor:

>>> def my_executor(prompt: str, **kwargs) -> str:
...     # Your LLM call here
...     return f"Response to: {prompt}"
>>>
>>> configs = [
...     ExperimentConfig(id="exp_0", name="test_0", prompt="Hello", model_id="gpt-4"),
...     ExperimentConfig(id="exp_1", name="test_1", prompt="World", model_id="gpt-4"),
... ]
>>> runner = BatchRunner(executor=my_executor, max_workers=2)
>>> result = runner.run(configs, parallel=True)
>>> print(f"Success rate: {result.success_rate:.1%}")
Success rate: 100.0%

Running a grid search experiment:

>>> grid = ExperimentGrid(
...     base_prompt="Translate to French: Hello",
...     model_ids=["gpt-4", "claude-3"],
...     parameters={"temperature": [0.0, 0.5, 1.0]},
... )
>>> orchestrator = ExperimentOrchestrator(executor=my_executor)
>>> result = orchestrator.run_grid(grid, parallel=True, max_workers=4)
>>> print(f"Total experiments: {len(grid)}")
Total experiments: 6

A/B comparison of prompts:

>>> comparison = ComparisonExperiment(
...     prompts={
...         "formal": "Please summarize: {input}",
...         "casual": "Give me a quick summary of: {input}",
...     },
...     model_id="gpt-4",
...     inputs=["Document 1 content", "Document 2 content"],
... )
>>> result = orchestrator.run_comparison(comparison)
>>> formal_runs = result.get_by_tag("formal")
>>> casual_runs = result.get_by_tag("casual")

Using the experiment queue for custom scheduling:

>>> queue = ExperimentQueue()
>>> queue.add(ExperimentConfig(id="low", name="low", prompt="Low priority", model_id="gpt-4", priority=1))
>>> queue.add(ExperimentConfig(id="high", name="high", prompt="High priority", model_id="gpt-4", priority=10))
>>> next_exp = queue.next()  # Returns "high" due to higher priority
>>> print(next_exp.id)
high

Notes
-----
* All timestamps use `datetime.now()` for simplicity; consider using UTC in production
* The `ModelExecutor` protocol expects a callable that takes (prompt, **kwargs) -> str
* Parallel execution uses `ThreadPoolExecutor`; for async use `AsyncBatchRunner`
* Metrics calculators receive the prompt, response, and config for flexible analysis
* Batch IDs are auto-generated with timestamps if not provided

See Also
--------
* `insideLLMs.core` : Core experiment definitions and types
* `insideLLMs.analysis` : Analysis utilities for experiment results
* `insideLLMs.visualization` : Visualization tools for experiment data

References
----------
.. [1] Python concurrent.futures documentation
   https://docs.python.org/3/library/concurrent.futures.html
.. [2] Python asyncio documentation
   https://docs.python.org/3/library/asyncio.html
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Optional,
    Protocol,
)


class ExperimentStatus(Enum):
    """
    Enumeration of possible experiment execution states.

    This enum represents the lifecycle states of an experiment from creation
    through completion. Experiments start as PENDING, transition to RUNNING
    during execution, and end in one of COMPLETED, FAILED, or CANCELLED states.
    The PAUSED state is available for experiments that support interruption.

    Attributes
    ----------
    PENDING : str
        Experiment is queued but not yet started. This is the initial state
        for all experiments before execution begins.
    RUNNING : str
        Experiment is currently being executed. The executor is processing
        the prompt and awaiting a response from the model.
    COMPLETED : str
        Experiment finished successfully with a valid response. The response
        and metrics are available in the ExperimentRun.
    FAILED : str
        Experiment encountered an error during execution. The error message
        is available in the ExperimentRun.error field.
    CANCELLED : str
        Experiment was cancelled before completion. This typically occurs
        when a batch is interrupted or when using queue cancellation.
    PAUSED : str
        Experiment execution is temporarily suspended. This state is
        available for experiments that support pause/resume functionality.

    Examples
    --------
    Checking experiment status:

    >>> run = ExperimentRun(config=config)
    >>> if run.status == ExperimentStatus.PENDING:
    ...     print("Experiment waiting to start")
    Experiment waiting to start

    Filtering completed experiments:

    >>> completed = [r for r in runs if r.status == ExperimentStatus.COMPLETED]

    Using status in a state machine:

    >>> if run.status == ExperimentStatus.FAILED:
    ...     print(f"Error: {run.error}")
    ...     if can_retry:
    ...         run.status = ExperimentStatus.PENDING  # Queue for retry
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class OrchestrationExperimentConfig:
    """
    Configuration for a single experiment in the orchestration framework.

    This dataclass defines all parameters needed to execute a single experiment,
    including the prompt to send, the model to use, additional parameters,
    and metadata for organization and filtering. The priority field enables
    priority-based scheduling when using ExperimentQueue.

    Parameters
    ----------
    id : str
        Unique identifier for this experiment. Used for tracking and
        deduplication. Should be unique within a batch.
    name : str
        Human-readable name for the experiment. Displayed in reports
        and used for filtering. Can contain spaces and special characters.
    prompt : str
        The prompt text to send to the model. This is the primary input
        for the experiment. Can contain placeholders for templating.
    model_id : str
        Identifier of the model to use for this experiment. The format
        depends on your executor implementation (e.g., "gpt-4", "claude-3").
    parameters : dict[str, Any], optional
        Additional parameters passed to the executor. Common parameters
        include temperature, max_tokens, top_p, etc. Default is empty dict.
    tags : list[str], optional
        Tags for categorizing and filtering experiments. Useful for
        grouping related experiments and filtering results. Default is empty list.
    metadata : dict[str, Any], optional
        Arbitrary metadata associated with the experiment. Stored with
        results for later analysis. Default is empty dict.
    priority : int, optional
        Priority level for queue scheduling. Higher values indicate higher
        priority. Experiments with higher priority are executed first
        when using ExperimentQueue. Default is 0.

    Attributes
    ----------
    id : str
        The experiment's unique identifier.
    name : str
        Human-readable experiment name.
    prompt : str
        The prompt text for the experiment.
    model_id : str
        The target model identifier.
    parameters : dict[str, Any]
        Execution parameters for the model.
    tags : list[str]
        Categorization tags.
    metadata : dict[str, Any]
        Additional metadata.
    priority : int
        Queue scheduling priority.

    Examples
    --------
    Creating a basic experiment configuration:

    >>> config = OrchestrationExperimentConfig(
    ...     id="exp_001",
    ...     name="Translation Test",
    ...     prompt="Translate 'Hello' to French",
    ...     model_id="gpt-4",
    ... )
    >>> print(config.id)
    exp_001

    Configuration with parameters and tags:

    >>> config = OrchestrationExperimentConfig(
    ...     id="exp_002",
    ...     name="Creative Writing",
    ...     prompt="Write a haiku about programming",
    ...     model_id="claude-3-sonnet",
    ...     parameters={"temperature": 0.8, "max_tokens": 100},
    ...     tags=["creative", "poetry", "test"],
    ...     metadata={"author": "test_user", "category": "creative"},
    ... )
    >>> print(config.parameters["temperature"])
    0.8

    High-priority experiment for queue scheduling:

    >>> urgent_config = OrchestrationExperimentConfig(
    ...     id="urgent_001",
    ...     name="Urgent Analysis",
    ...     prompt="Analyze critical data",
    ...     model_id="gpt-4",
    ...     priority=100,  # High priority
    ... )
    >>> queue.add(urgent_config)  # Will be processed before lower priority

    See Also
    --------
    ExperimentRun : Records the execution results of a config.
    ExperimentQueue : Priority-based scheduling using the priority field.
    ExperimentGrid : Generates multiple configs from parameter combinations.
    """

    id: str
    name: str
    prompt: str
    model_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher = higher priority


# Backward compatibility alias
ExperimentConfig = OrchestrationExperimentConfig


@dataclass
class ExperimentRun:
    """
    Record of a single experiment execution with results and metrics.

    This dataclass captures all information about an experiment's execution,
    including the original configuration, execution status, response from
    the model, timing information, any errors encountered, and computed
    metrics. It serves as the primary result type for experiment execution.

    Parameters
    ----------
    config : OrchestrationExperimentConfig
        The configuration that was used to run this experiment. Contains
        the prompt, model, and parameters.
    status : ExperimentStatus, optional
        Current status of the experiment. Default is PENDING.
    response : str or None, optional
        The response received from the model. None if experiment hasn't
        completed or failed. Default is None.
    latency_ms : float, optional
        Time taken for the model to respond in milliseconds. Measured
        from prompt submission to response receipt. Default is 0.0.
    error : str or None, optional
        Error message if the experiment failed. None for successful
        experiments. Default is None.
    started_at : datetime or None, optional
        Timestamp when execution began. None if not yet started.
        Default is None.
    completed_at : datetime or None, optional
        Timestamp when execution finished. None if not yet completed.
        Default is None.
    metrics : dict[str, float], optional
        Custom metrics computed for this experiment. Populated by
        MetricsCalculator if one is configured. Default is empty dict.

    Attributes
    ----------
    config : OrchestrationExperimentConfig
        The experiment's configuration.
    status : ExperimentStatus
        Current execution status.
    response : str or None
        Model response text.
    latency_ms : float
        Response time in milliseconds.
    error : str or None
        Error message if failed.
    started_at : datetime or None
        Execution start timestamp.
    completed_at : datetime or None
        Execution end timestamp.
    metrics : dict[str, float]
        Computed metrics for this run.
    duration_ms : float
        Property: total duration in milliseconds.
    is_complete : bool
        Property: whether experiment has finished.

    Examples
    --------
    Creating a run record for an experiment:

    >>> config = ExperimentConfig(
    ...     id="exp_001", name="Test", prompt="Hello", model_id="gpt-4"
    ... )
    >>> run = ExperimentRun(config=config)
    >>> print(run.status)
    ExperimentStatus.PENDING

    Checking if an experiment completed successfully:

    >>> if run.is_complete and run.status == ExperimentStatus.COMPLETED:
    ...     print(f"Response: {run.response}")
    ...     print(f"Latency: {run.latency_ms:.1f}ms")
    Response: Hello! How can I help you?
    Latency: 245.3ms

    Accessing metrics after execution:

    >>> run.status = ExperimentStatus.COMPLETED
    >>> run.response = "Generated text here"
    >>> run.metrics = {"word_count": 42, "sentiment": 0.8}
    >>> print(f"Word count: {run.metrics['word_count']}")
    Word count: 42

    Handling failed experiments:

    >>> if run.status == ExperimentStatus.FAILED:
    ...     print(f"Experiment {run.config.id} failed: {run.error}")
    Experiment exp_001 failed: API rate limit exceeded

    Calculating duration from timestamps:

    >>> run.started_at = datetime(2024, 1, 1, 12, 0, 0)
    >>> run.completed_at = datetime(2024, 1, 1, 12, 0, 1, 500000)
    >>> print(f"Duration: {run.duration_ms:.1f}ms")
    Duration: 1500.0ms

    See Also
    --------
    ExperimentConfig : The configuration used to create a run.
    ExperimentBatchResult : Aggregates multiple runs.
    BatchRunner : Executes configs and produces runs.
    """

    config: OrchestrationExperimentConfig
    status: ExperimentStatus = ExperimentStatus.PENDING
    response: Optional[str] = None
    latency_ms: float = 0.0
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """
        Calculate the total experiment duration in milliseconds.

        Computes duration from timestamps if both started_at and completed_at
        are available. Falls back to latency_ms if timestamps are not set.
        This provides a more accurate total duration including any overhead
        beyond the model response time.

        Returns
        -------
        float
            Duration in milliseconds. Returns 0.0 if timing data is unavailable.

        Examples
        --------
        Duration from timestamps:

        >>> run.started_at = datetime(2024, 1, 1, 12, 0, 0)
        >>> run.completed_at = datetime(2024, 1, 1, 12, 0, 2)
        >>> print(run.duration_ms)
        2000.0

        Fallback to latency when timestamps unavailable:

        >>> run = ExperimentRun(config=config, latency_ms=500.0)
        >>> print(run.duration_ms)
        500.0
        """
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return self.latency_ms

    @property
    def is_complete(self) -> bool:
        """
        Check if the experiment has finished execution.

        An experiment is considered complete when it has reached a terminal
        state: COMPLETED (success), FAILED (error), or CANCELLED (interrupted).
        Experiments in PENDING, RUNNING, or PAUSED states are not complete.

        Returns
        -------
        bool
            True if experiment is in a terminal state, False otherwise.

        Examples
        --------
        Checking completion status:

        >>> run = ExperimentRun(config=config, status=ExperimentStatus.PENDING)
        >>> print(run.is_complete)
        False
        >>> run.status = ExperimentStatus.COMPLETED
        >>> print(run.is_complete)
        True

        Filtering incomplete experiments:

        >>> incomplete = [r for r in runs if not r.is_complete]
        """
        return self.status in (
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
        )


@dataclass
class ExperimentBatchResult:
    """
    Aggregated results from a batch of experiment executions.

    This dataclass collects results from multiple experiments run together,
    providing aggregate statistics, filtering capabilities, and metrics
    aggregation. It serves as the primary result type for BatchRunner and
    ExperimentOrchestrator operations.

    Parameters
    ----------
    batch_id : str
        Unique identifier for this batch. Auto-generated with timestamp
        if not provided to runner methods.
    runs : list[ExperimentRun]
        List of individual experiment run records in this batch.
    started_at : datetime, optional
        Timestamp when batch execution began. Default is current time.
    completed_at : datetime or None, optional
        Timestamp when batch execution finished. None until all
        experiments complete. Default is None.

    Attributes
    ----------
    batch_id : str
        The batch's unique identifier.
    runs : list[ExperimentRun]
        All experiment runs in the batch.
    started_at : datetime
        Batch start timestamp.
    completed_at : datetime or None
        Batch completion timestamp.
    total : int
        Property: total number of experiments.
    completed : int
        Property: count of successfully completed experiments.
    failed : int
        Property: count of failed experiments.
    success_rate : float
        Property: ratio of completed to total finished.
    avg_latency_ms : float
        Property: average latency of completed experiments.

    Examples
    --------
    Creating a batch result manually:

    >>> batch = ExperimentBatchResult(
    ...     batch_id="batch_001",
    ...     runs=[run1, run2, run3],
    ... )
    >>> print(f"Total experiments: {batch.total}")
    Total experiments: 3

    Accessing aggregate statistics:

    >>> result = runner.run(configs)
    >>> print(f"Success rate: {result.success_rate:.1%}")
    >>> print(f"Average latency: {result.avg_latency_ms:.1f}ms")
    >>> print(f"Completed: {result.completed}/{result.total}")
    Success rate: 95.0%
    Average latency: 312.5ms
    Completed: 19/20

    Filtering runs by status:

    >>> failed_runs = result.get_by_status(ExperimentStatus.FAILED)
    >>> for run in failed_runs:
    ...     print(f"Failed: {run.config.id} - {run.error}")
    Failed: exp_005 - Timeout exceeded

    Filtering runs by tag:

    >>> translation_runs = result.get_by_tag("translation")
    >>> avg_latency = sum(r.latency_ms for r in translation_runs) / len(translation_runs)

    Filtering runs by model:

    >>> gpt4_runs = result.get_by_model("gpt-4")
    >>> claude_runs = result.get_by_model("claude-3")

    Aggregating metrics across runs:

    >>> metrics = result.aggregate_metrics()
    >>> print(f"Word count - mean: {metrics['word_count']['mean']:.1f}")
    >>> print(f"Word count - range: {metrics['word_count']['min']}-{metrics['word_count']['max']}")
    Word count - mean: 156.3
    Word count - range: 42-312

    See Also
    --------
    ExperimentRun : Individual experiment results.
    BatchRunner : Creates batch results from experiment execution.
    ExperimentOrchestrator : High-level interface returning batch results.
    """

    batch_id: str
    runs: list[ExperimentRun]
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    @property
    def total(self) -> int:
        """
        Get the total number of experiments in this batch.

        Returns
        -------
        int
            Total count of all experiments regardless of status.

        Examples
        --------
        >>> print(f"Total experiments: {batch.total}")
        Total experiments: 100
        """
        return len(self.runs)

    @property
    def completed(self) -> int:
        """
        Get the count of successfully completed experiments.

        Returns
        -------
        int
            Number of experiments with COMPLETED status.

        Examples
        --------
        >>> print(f"Completed: {batch.completed}/{batch.total}")
        Completed: 95/100
        """
        return sum(1 for r in self.runs if r.status == ExperimentStatus.COMPLETED)

    @property
    def failed(self) -> int:
        """
        Get the count of failed experiments.

        Returns
        -------
        int
            Number of experiments with FAILED status.

        Examples
        --------
        >>> print(f"Failed experiments: {batch.failed}")
        Failed experiments: 5
        """
        return sum(1 for r in self.runs if r.status == ExperimentStatus.FAILED)

    @property
    def success_rate(self) -> float:
        """
        Calculate the success rate of finished experiments.

        Computes the ratio of completed experiments to total finished
        experiments (completed + failed). Pending, running, and cancelled
        experiments are not included in the calculation.

        Returns
        -------
        float
            Success rate as a decimal (0.0 to 1.0). Returns 0.0 if no
            experiments have finished.

        Examples
        --------
        >>> print(f"Success rate: {batch.success_rate:.1%}")
        Success rate: 95.0%

        >>> if batch.success_rate < 0.9:
        ...     print("Warning: High failure rate detected")
        """
        total = self.completed + self.failed
        return self.completed / total if total > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """
        Calculate average latency of successfully completed experiments.

        Computes the mean latency in milliseconds across all experiments
        with COMPLETED status. Failed and pending experiments are excluded.

        Returns
        -------
        float
            Average latency in milliseconds. Returns 0.0 if no experiments
            completed successfully.

        Examples
        --------
        >>> print(f"Average latency: {batch.avg_latency_ms:.1f}ms")
        Average latency: 312.5ms
        """
        completed_runs = [r for r in self.runs if r.status == ExperimentStatus.COMPLETED]
        if not completed_runs:
            return 0.0
        return sum(r.latency_ms for r in completed_runs) / len(completed_runs)

    def get_by_status(self, status: ExperimentStatus) -> list[ExperimentRun]:
        """
        Filter experiment runs by their execution status.

        Parameters
        ----------
        status : ExperimentStatus
            The status to filter by.

        Returns
        -------
        list[ExperimentRun]
            List of runs matching the specified status.

        Examples
        --------
        Get all failed experiments for retry:

        >>> failed = batch.get_by_status(ExperimentStatus.FAILED)
        >>> for run in failed:
        ...     print(f"Retry needed: {run.config.id}")

        Get all pending experiments:

        >>> pending = batch.get_by_status(ExperimentStatus.PENDING)
        >>> print(f"{len(pending)} experiments still waiting")
        """
        return [r for r in self.runs if r.status == status]

    def get_by_tag(self, tag: str) -> list[ExperimentRun]:
        """
        Filter experiment runs by a configuration tag.

        Parameters
        ----------
        tag : str
            The tag to filter by. Must match exactly.

        Returns
        -------
        list[ExperimentRun]
            List of runs whose configs contain the specified tag.

        Examples
        --------
        Filter translation experiments:

        >>> translation_runs = batch.get_by_tag("translation")
        >>> print(f"Found {len(translation_runs)} translation experiments")

        Compare experiments by category:

        >>> formal = batch.get_by_tag("formal")
        >>> casual = batch.get_by_tag("casual")
        >>> formal_avg = sum(r.latency_ms for r in formal) / len(formal)
        >>> casual_avg = sum(r.latency_ms for r in casual) / len(casual)
        """
        return [r for r in self.runs if tag in r.config.tags]

    def get_by_model(self, model_id: str) -> list[ExperimentRun]:
        """
        Filter experiment runs by model identifier.

        Parameters
        ----------
        model_id : str
            The model ID to filter by. Must match exactly.

        Returns
        -------
        list[ExperimentRun]
            List of runs configured to use the specified model.

        Examples
        --------
        Compare model performance:

        >>> gpt4_runs = batch.get_by_model("gpt-4")
        >>> claude_runs = batch.get_by_model("claude-3-sonnet")
        >>> gpt4_avg = sum(r.latency_ms for r in gpt4_runs) / len(gpt4_runs)
        >>> claude_avg = sum(r.latency_ms for r in claude_runs) / len(claude_runs)
        >>> print(f"GPT-4 avg: {gpt4_avg:.1f}ms, Claude avg: {claude_avg:.1f}ms")
        """
        return [r for r in self.runs if r.config.model_id == model_id]

    def aggregate_metrics(self) -> dict[str, dict[str, float]]:
        """
        Aggregate metrics across all successfully completed runs.

        Computes summary statistics (mean, min, max, count) for each
        metric present in the completed runs. Only runs with COMPLETED
        status are included in the aggregation.

        Returns
        -------
        dict[str, dict[str, float]]
            Dictionary mapping metric names to their statistics. Each
            metric has keys: 'mean', 'min', 'max', 'count'.

        Examples
        --------
        Get aggregate statistics for all metrics:

        >>> metrics = batch.aggregate_metrics()
        >>> for name, stats in metrics.items():
        ...     print(f"{name}: mean={stats['mean']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]")
        word_count: mean=156.30, range=[42.00, 312.00]
        sentiment: mean=0.72, range=[0.15, 0.95]

        Access specific metric statistics:

        >>> if 'response_length' in metrics:
        ...     avg_length = metrics['response_length']['mean']
        ...     print(f"Average response: {avg_length:.0f} chars")
        """
        completed = self.get_by_status(ExperimentStatus.COMPLETED)
        if not completed:
            return {}

        all_metrics: dict[str, list[float]] = {}
        for run in completed:
            for name, value in run.metrics.items():
                if name not in all_metrics:
                    all_metrics[name] = []
                all_metrics[name].append(value)

        result = {}
        for name, values in all_metrics.items():
            result[name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
        return result


class ProgressCallback(Protocol):
    """
    Protocol defining the interface for progress callbacks.

    Progress callbacks are invoked during batch execution to report
    on experiment completion status. This enables real-time monitoring,
    progress bars, logging, and other feedback mechanisms.

    The callback receives the count of completed experiments, total
    experiments, and optionally the most recently completed run.

    Methods
    -------
    __call__(completed, total, current=None)
        Called when an experiment completes.

    Examples
    --------
    Simple progress printer:

    >>> def print_progress(completed: int, total: int, current=None):
    ...     pct = completed / total * 100
    ...     print(f"Progress: {completed}/{total} ({pct:.1f}%)")
    >>> runner = BatchRunner(executor=my_executor)
    >>> runner.on_progress(print_progress)
    >>> result = runner.run(configs)
    Progress: 1/10 (10.0%)
    Progress: 2/10 (20.0%)
    ...

    Progress with status tracking:

    >>> def track_progress(completed: int, total: int, current=None):
    ...     if current and current.status == ExperimentStatus.FAILED:
    ...         print(f"FAILED: {current.config.id} - {current.error}")
    ...     else:
    ...         print(f"Completed: {current.config.id}")

    Progress bar integration (e.g., with tqdm):

    >>> from tqdm import tqdm
    >>> pbar = tqdm(total=len(configs))
    >>> def update_bar(completed: int, total: int, current=None):
    ...     pbar.update(1)
    >>> runner.on_progress(update_bar)
    >>> result = runner.run(configs)
    >>> pbar.close()

    See Also
    --------
    BatchRunner.on_progress : Set a progress callback on the runner.
    AsyncBatchRunner.on_progress : Set a progress callback for async execution.
    """

    def __call__(
        self,
        completed: int,
        total: int,
        current: Optional[ExperimentRun] = None,
    ) -> None:
        """
        Called when progress updates during batch execution.

        Parameters
        ----------
        completed : int
            Number of experiments completed so far.
        total : int
            Total number of experiments in the batch.
        current : ExperimentRun or None, optional
            The most recently completed experiment run, if available.

        Returns
        -------
        None

        Examples
        --------
        >>> def callback(completed: int, total: int, current=None):
        ...     print(f"{completed}/{total}")
        ...     if current:
        ...         print(f"  Last: {current.config.name}")
        """
        ...


class ModelExecutor(Protocol):
    """
    Protocol defining the interface for model execution functions.

    A model executor is a callable that takes a prompt string and optional
    keyword arguments, and returns the model's response as a string. This
    abstraction allows the orchestration framework to work with any LLM
    backend or API.

    The executor should handle all model-specific logic including API calls,
    authentication, retries (if not using BatchRunner's retry), and response
    parsing.

    Methods
    -------
    __call__(prompt, **kwargs)
        Execute a prompt and return the model's response.

    Examples
    --------
    Simple executor wrapping an API client:

    >>> def openai_executor(prompt: str, **kwargs) -> str:
    ...     response = openai.ChatCompletion.create(
    ...         model=kwargs.get("model", "gpt-4"),
    ...         messages=[{"role": "user", "content": prompt}],
    ...         temperature=kwargs.get("temperature", 0.7),
    ...     )
    ...     return response.choices[0].message.content

    Executor with error handling:

    >>> def robust_executor(prompt: str, **kwargs) -> str:
    ...     try:
    ...         return api_client.complete(prompt, **kwargs)
    ...     except RateLimitError:
    ...         time.sleep(60)
    ...         return api_client.complete(prompt, **kwargs)

    Mock executor for testing:

    >>> def mock_executor(prompt: str, **kwargs) -> str:
    ...     return f"Mock response to: {prompt[:50]}..."

    Using with BatchRunner:

    >>> runner = BatchRunner(executor=openai_executor, max_workers=4)
    >>> result = runner.run(configs, parallel=True)

    See Also
    --------
    BatchRunner : Uses ModelExecutor for experiment execution.
    AsyncBatchRunner : Uses async version of executor.
    ExperimentOrchestrator : High-level interface using executor.
    """

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """
        Execute a prompt and return the model's response.

        Parameters
        ----------
        prompt : str
            The prompt text to send to the model.
        **kwargs : Any
            Additional parameters passed from ExperimentConfig.parameters.
            Common parameters include temperature, max_tokens, etc.

        Returns
        -------
        str
            The model's response text.

        Raises
        ------
        Exception
            Any exception raised will be caught by BatchRunner and
            recorded in the ExperimentRun.error field.

        Examples
        --------
        >>> response = executor("What is 2+2?", temperature=0.0)
        >>> print(response)
        The answer is 4.
        """
        ...


class MetricsCalculator(Protocol):
    """
    Protocol defining the interface for metrics calculation functions.

    A metrics calculator takes the prompt, response, and experiment config,
    and returns a dictionary of metric names to float values. This enables
    custom quality metrics, response analysis, and automated evaluation.

    Common metrics include response length, sentiment scores, specific
    pattern matches, semantic similarity, or task-specific quality measures.

    Methods
    -------
    __call__(prompt, response, config)
        Calculate metrics for a response.

    Examples
    --------
    Simple word count metric:

    >>> def word_count_metrics(prompt: str, response: str, config) -> dict[str, float]:
    ...     words = len(response.split())
    ...     chars = len(response)
    ...     return {"word_count": words, "char_count": chars}

    Sentiment analysis metric:

    >>> def sentiment_metrics(prompt: str, response: str, config) -> dict[str, float]:
    ...     # Using a hypothetical sentiment analyzer
    ...     score = sentiment_analyzer.analyze(response)
    ...     return {"sentiment": score.compound, "positive": score.pos, "negative": score.neg}

    Task-specific metric:

    >>> def translation_metrics(prompt: str, response: str, config) -> dict[str, float]:
    ...     # Check if response is in expected language
    ...     lang = detect_language(response)
    ...     expected = config.metadata.get("target_language", "en")
    ...     return {
    ...         "correct_language": 1.0 if lang == expected else 0.0,
    ...         "response_length": len(response),
    ...     }

    Combining multiple metrics:

    >>> def combined_metrics(prompt: str, response: str, config) -> dict[str, float]:
    ...     metrics = {}
    ...     metrics.update(word_count_metrics(prompt, response, config))
    ...     metrics.update(sentiment_metrics(prompt, response, config))
    ...     metrics["relevance"] = compute_relevance(prompt, response)
    ...     return metrics

    Using with BatchRunner:

    >>> runner = BatchRunner(
    ...     executor=my_executor,
    ...     metrics_calculator=combined_metrics,
    ... )
    >>> result = runner.run(configs)
    >>> print(result.runs[0].metrics)
    {'word_count': 42.0, 'sentiment': 0.75, 'relevance': 0.92}

    See Also
    --------
    BatchRunner : Uses MetricsCalculator to compute run metrics.
    ExperimentBatchResult.aggregate_metrics : Aggregates metrics across runs.
    """

    def __call__(
        self,
        prompt: str,
        response: str,
        config: ExperimentConfig,
    ) -> dict[str, float]:
        """
        Calculate metrics for an experiment response.

        Parameters
        ----------
        prompt : str
            The original prompt sent to the model.
        response : str
            The response received from the model.
        config : ExperimentConfig
            The experiment configuration, useful for accessing metadata
            or expected values.

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their computed values.
            All values should be numeric (float).

        Examples
        --------
        >>> metrics = calculator("Hello", "Hi there!", config)
        >>> print(metrics)
        {'response_length': 10.0, 'sentiment': 0.8}
        """
        ...


class ExperimentQueue:
    """
    Priority-based queue for managing experiment execution order.

    This class provides a queue data structure for scheduling experiments
    with priority-based ordering. Experiments with higher priority values
    are executed first. The queue tracks pending, running, and completed
    experiments, enabling progress monitoring and result collection.

    The queue is useful for scenarios where experiments need to be
    dynamically added or prioritized, or when implementing custom
    execution loops.

    Parameters
    ----------
    None

    Attributes
    ----------
    pending_count : int
        Property: number of experiments waiting to be executed.
    running_count : int
        Property: number of experiments currently being executed.
    completed_count : int
        Property: number of experiments that have finished.

    Examples
    --------
    Basic queue usage:

    >>> queue = ExperimentQueue()
    >>> queue.add(ExperimentConfig(id="1", name="Low", prompt="...", model_id="gpt-4", priority=1))
    >>> queue.add(ExperimentConfig(id="2", name="High", prompt="...", model_id="gpt-4", priority=10))
    >>> queue.add(ExperimentConfig(id="3", name="Medium", prompt="...", model_id="gpt-4", priority=5))
    >>>
    >>> # Higher priority comes first
    >>> next_config = queue.next()
    >>> print(next_config.name)
    High

    Processing a queue:

    >>> queue = ExperimentQueue()
    >>> queue.add_batch(configs)
    >>>
    >>> while not queue.is_empty():
    ...     config = queue.next()
    ...     if config:
    ...         run = execute_experiment(config)
    ...         queue.complete(run)
    >>>
    >>> results = queue.get_results()
    >>> print(f"Processed {len(results)} experiments")

    Monitoring queue state:

    >>> print(f"Pending: {queue.pending_count}")
    >>> print(f"Running: {queue.running_count}")
    >>> print(f"Completed: {queue.completed_count}")
    Pending: 5
    Running: 2
    Completed: 10

    Dynamic priority adjustment:

    >>> # Add urgent experiment that jumps the queue
    >>> urgent = ExperimentConfig(
    ...     id="urgent", name="Urgent", prompt="...",
    ...     model_id="gpt-4", priority=1000
    ... )
    >>> queue.add(urgent)  # Will be next to execute

    See Also
    --------
    BatchRunner : Alternative for running batches without manual queue management.
    ExperimentConfig.priority : The priority field used for ordering.
    """

    def __init__(self):
        """
        Initialize an empty experiment queue.

        Creates internal storage for pending, running, and completed
        experiments. The queue is ready to accept experiments via
        add() or add_batch() after initialization.

        Examples
        --------
        >>> queue = ExperimentQueue()
        >>> print(queue.pending_count)
        0
        """
        self._pending: list[ExperimentConfig] = []
        self._running: dict[str, ExperimentConfig] = {}
        self._completed: list[ExperimentRun] = []

    def add(self, config: ExperimentConfig) -> None:
        """
        Add a single experiment to the queue.

        The experiment is inserted into the pending list and the list
        is re-sorted by priority (descending). Higher priority values
        result in earlier execution.

        Parameters
        ----------
        config : ExperimentConfig
            The experiment configuration to add to the queue.

        Returns
        -------
        None

        Examples
        --------
        Add experiments with different priorities:

        >>> queue.add(ExperimentConfig(id="low", name="Low", prompt="...", model_id="gpt-4", priority=1))
        >>> queue.add(ExperimentConfig(id="high", name="High", prompt="...", model_id="gpt-4", priority=100))
        >>> print(queue.next().id)  # High priority first
        high

        Add default priority (0):

        >>> queue.add(ExperimentConfig(id="default", name="Default", prompt="...", model_id="gpt-4"))
        """
        self._pending.append(config)
        # Sort by priority (higher priority first)
        self._pending.sort(key=lambda x: -x.priority)

    def add_batch(self, configs: list[ExperimentConfig]) -> None:
        """
        Add multiple experiments to the queue.

        Convenience method for adding a list of experiments. Each
        experiment is added individually with priority sorting.

        Parameters
        ----------
        configs : list[ExperimentConfig]
            List of experiment configurations to add.

        Returns
        -------
        None

        Examples
        --------
        Add a batch of experiments:

        >>> configs = [
        ...     ExperimentConfig(id="1", name="Exp 1", prompt="...", model_id="gpt-4"),
        ...     ExperimentConfig(id="2", name="Exp 2", prompt="...", model_id="gpt-4"),
        ...     ExperimentConfig(id="3", name="Exp 3", prompt="...", model_id="gpt-4"),
        ... ]
        >>> queue.add_batch(configs)
        >>> print(queue.pending_count)
        3
        """
        for config in configs:
            self.add(config)

    def next(self) -> Optional[ExperimentConfig]:
        """
        Get the next experiment to execute.

        Removes and returns the highest-priority pending experiment.
        The experiment is moved to the running set. Returns None if
        no pending experiments remain.

        Returns
        -------
        ExperimentConfig or None
            The next experiment config to execute, or None if queue is empty.

        Examples
        --------
        Process experiments in priority order:

        >>> while (config := queue.next()) is not None:
        ...     run = execute_experiment(config)
        ...     queue.complete(run)

        Check for available work:

        >>> config = queue.next()
        >>> if config is None:
        ...     print("Queue exhausted")
        """
        if not self._pending:
            return None
        config = self._pending.pop(0)
        self._running[config.id] = config
        return config

    def complete(self, run: ExperimentRun) -> None:
        """
        Mark an experiment as complete with its results.

        Removes the experiment from the running set and adds the
        run record to the completed list for later retrieval.

        Parameters
        ----------
        run : ExperimentRun
            The completed experiment run with results.

        Returns
        -------
        None

        Examples
        --------
        Complete an experiment:

        >>> config = queue.next()
        >>> # ... execute experiment ...
        >>> run = ExperimentRun(config=config, status=ExperimentStatus.COMPLETED, response="...")
        >>> queue.complete(run)
        >>> print(queue.completed_count)
        1
        """
        if run.config.id in self._running:
            del self._running[run.config.id]
        self._completed.append(run)

    @property
    def pending_count(self) -> int:
        """
        Get the number of pending experiments.

        Returns
        -------
        int
            Count of experiments waiting to be executed.

        Examples
        --------
        >>> print(f"Waiting: {queue.pending_count}")
        Waiting: 5
        """
        return len(self._pending)

    @property
    def running_count(self) -> int:
        """
        Get the number of currently running experiments.

        Returns
        -------
        int
            Count of experiments currently being executed.

        Examples
        --------
        >>> print(f"In progress: {queue.running_count}")
        In progress: 2
        """
        return len(self._running)

    @property
    def completed_count(self) -> int:
        """
        Get the number of completed experiments.

        Returns
        -------
        int
            Count of experiments that have finished (success or failure).

        Examples
        --------
        >>> print(f"Done: {queue.completed_count}")
        Done: 10
        """
        return len(self._completed)

    def is_empty(self) -> bool:
        """
        Check if the queue has no more work.

        Returns True only when there are no pending experiments and
        no running experiments. Completed experiments do not affect
        this status.

        Returns
        -------
        bool
            True if no pending or running experiments, False otherwise.

        Examples
        --------
        Check queue status:

        >>> if queue.is_empty():
        ...     print("All work complete")
        ... else:
        ...     print(f"Remaining: {queue.pending_count + queue.running_count}")
        """
        return len(self._pending) == 0 and len(self._running) == 0

    def get_results(self) -> list[ExperimentRun]:
        """
        Get all completed experiment runs.

        Returns a copy of the completed runs list. The original list
        remains in the queue for potential further inspection.

        Returns
        -------
        list[ExperimentRun]
            List of all completed experiment runs.

        Examples
        --------
        Retrieve results after processing:

        >>> while not queue.is_empty():
        ...     config = queue.next()
        ...     run = execute_experiment(config)
        ...     queue.complete(run)
        >>>
        >>> results = queue.get_results()
        >>> successful = [r for r in results if r.status == ExperimentStatus.COMPLETED]
        >>> print(f"Success: {len(successful)}/{len(results)}")
        """
        return list(self._completed)


class BatchRunner:
    """
    Execute batches of experiments with configurable parallelism and retry.

    BatchRunner provides the primary mechanism for running multiple experiments
    efficiently. It supports both sequential and parallel execution modes,
    automatic retry on failure, progress callbacks, and optional metrics
    calculation.

    For parallel execution, ThreadPoolExecutor is used to run multiple
    experiments concurrently. The max_workers parameter controls the
    degree of parallelism.

    Parameters
    ----------
    executor : ModelExecutor
        Callable that executes prompts. Receives (prompt, **kwargs) and
        returns the response string.
    metrics_calculator : MetricsCalculator or None, optional
        Optional callable to compute metrics for each response. If provided,
        metrics are stored in ExperimentRun.metrics. Default is None.
    max_workers : int, optional
        Maximum number of parallel workers for parallel execution.
        Only used when running with parallel=True. Default is 1.
    retry_on_failure : bool, optional
        Whether to automatically retry failed experiments. Default is False.
    max_retries : int, optional
        Maximum number of retry attempts per experiment when retry_on_failure
        is True. Default is 3.

    Attributes
    ----------
    executor : ModelExecutor
        The configured model executor.
    metrics_calculator : MetricsCalculator or None
        The configured metrics calculator.
    max_workers : int
        Maximum parallel workers.
    retry_on_failure : bool
        Whether retry is enabled.
    max_retries : int
        Maximum retry attempts.

    Examples
    --------
    Basic sequential execution:

    >>> def my_executor(prompt: str, **kwargs) -> str:
    ...     return llm_client.complete(prompt, **kwargs)
    >>>
    >>> runner = BatchRunner(executor=my_executor)
    >>> configs = [
    ...     ExperimentConfig(id="1", name="Test 1", prompt="Hello", model_id="gpt-4"),
    ...     ExperimentConfig(id="2", name="Test 2", prompt="World", model_id="gpt-4"),
    ... ]
    >>> result = runner.run(configs)
    >>> print(f"Completed: {result.completed}")
    Completed: 2

    Parallel execution with progress tracking:

    >>> def log_progress(completed: int, total: int, current=None):
    ...     print(f"Progress: {completed}/{total}")
    >>>
    >>> runner = BatchRunner(executor=my_executor, max_workers=4)
    >>> runner.on_progress(log_progress)
    >>> result = runner.run(configs, parallel=True)
    Progress: 1/10
    Progress: 2/10
    ...

    With metrics calculation:

    >>> def calc_metrics(prompt: str, response: str, config) -> dict[str, float]:
    ...     return {"word_count": len(response.split())}
    >>>
    >>> runner = BatchRunner(
    ...     executor=my_executor,
    ...     metrics_calculator=calc_metrics,
    ... )
    >>> result = runner.run(configs)
    >>> print(result.runs[0].metrics)
    {'word_count': 42}

    With retry on failure:

    >>> runner = BatchRunner(
    ...     executor=unreliable_executor,
    ...     retry_on_failure=True,
    ...     max_retries=3,
    ... )
    >>> result = runner.run(configs)

    Custom batch ID:

    >>> result = runner.run(configs, batch_id="my_experiment_2024")
    >>> print(result.batch_id)
    my_experiment_2024

    See Also
    --------
    AsyncBatchRunner : Async version for high-concurrency workloads.
    ExperimentOrchestrator : Higher-level interface with grid and comparison support.
    ExperimentBatchResult : The result type returned by run methods.
    """

    def __init__(
        self,
        executor: ModelExecutor,
        metrics_calculator: Optional[MetricsCalculator] = None,
        max_workers: int = 1,
        retry_on_failure: bool = False,
        max_retries: int = 3,
    ):
        """
        Initialize the batch runner with execution configuration.

        Parameters
        ----------
        executor : ModelExecutor
            Function to execute prompts. Receives (prompt, **kwargs).
        metrics_calculator : MetricsCalculator or None, optional
            Function to calculate metrics for responses. Default is None.
        max_workers : int, optional
            Maximum parallel workers for parallel execution. Default is 1.
        retry_on_failure : bool, optional
            Whether to retry failed experiments. Default is False.
        max_retries : int, optional
            Maximum retry attempts. Default is 3.

        Examples
        --------
        Basic initialization:

        >>> runner = BatchRunner(executor=my_executor)

        Full configuration:

        >>> runner = BatchRunner(
        ...     executor=my_executor,
        ...     metrics_calculator=my_metrics,
        ...     max_workers=8,
        ...     retry_on_failure=True,
        ...     max_retries=5,
        ... )
        """
        self.executor = executor
        self.metrics_calculator = metrics_calculator
        self.max_workers = max_workers
        self.retry_on_failure = retry_on_failure
        self.max_retries = max_retries
        self._progress_callback: Optional[ProgressCallback] = None

    def on_progress(self, callback: ProgressCallback) -> "BatchRunner":
        """
        Set a callback for progress updates during execution.

        The callback is invoked after each experiment completes with
        the current progress and the most recent run. This enables
        progress bars, logging, and real-time monitoring.

        Parameters
        ----------
        callback : ProgressCallback
            Function called with (completed, total, current_run) after
            each experiment finishes.

        Returns
        -------
        BatchRunner
            Self for method chaining.

        Examples
        --------
        Simple progress logging:

        >>> def log_progress(completed: int, total: int, current=None):
        ...     print(f"{completed}/{total} complete")
        >>>
        >>> runner = BatchRunner(executor=my_executor)
        >>> runner.on_progress(log_progress)
        >>> result = runner.run(configs)
        1/5 complete
        2/5 complete
        ...

        Chained configuration:

        >>> runner = (
        ...     BatchRunner(executor=my_executor, max_workers=4)
        ...     .on_progress(progress_callback)
        ... )
        >>> result = runner.run(configs, parallel=True)
        """
        self._progress_callback = callback
        return self

    def _execute_single(self, config: ExperimentConfig, retry_count: int = 0) -> ExperimentRun:
        """
        Execute a single experiment with optional retry.

        Internal method that handles the execution of one experiment,
        including timing, error handling, metrics calculation, and retry logic.

        Parameters
        ----------
        config : ExperimentConfig
            The experiment configuration to execute.
        retry_count : int, optional
            Current retry attempt number. Used internally for recursion.
            Default is 0.

        Returns
        -------
        ExperimentRun
            The completed experiment run with results or error.

        Examples
        --------
        This is an internal method. Use run(), run_sequential(), or
        run_parallel() instead.

        >>> # Internal usage:
        >>> run = runner._execute_single(config)
        >>> print(run.status)
        ExperimentStatus.COMPLETED
        """
        run = ExperimentRun(config=config, status=ExperimentStatus.RUNNING)
        run.started_at = datetime.now()

        try:
            start_time = time.time()
            response = self.executor(config.prompt, **config.parameters)
            run.latency_ms = (time.time() - start_time) * 1000
            run.response = response
            run.status = ExperimentStatus.COMPLETED

            # Calculate metrics if calculator provided
            if self.metrics_calculator and response:
                run.metrics = self.metrics_calculator(config.prompt, response, config)

        except Exception as e:
            run.error = str(e)
            run.status = ExperimentStatus.FAILED

            # Retry if configured
            if self.retry_on_failure and retry_count < self.max_retries:
                return self._execute_single(config, retry_count + 1)

        run.completed_at = datetime.now()
        return run

    def run_sequential(
        self, configs: list[ExperimentConfig], batch_id: Optional[str] = None
    ) -> ExperimentBatchResult:
        """
        Run experiments sequentially (one at a time).

        Executes each experiment in order, waiting for completion before
        starting the next. This is the safest execution mode, suitable for
        rate-limited APIs or when order matters.

        Parameters
        ----------
        configs : list[ExperimentConfig]
            List of experiment configurations to execute.
        batch_id : str or None, optional
            Custom batch identifier. If None, auto-generated with timestamp.
            Default is None.

        Returns
        -------
        ExperimentBatchResult
            Results containing all experiment runs.

        Examples
        --------
        Basic sequential execution:

        >>> runner = BatchRunner(executor=my_executor)
        >>> result = runner.run_sequential(configs)
        >>> print(f"Completed: {result.completed}/{result.total}")
        Completed: 10/10

        With custom batch ID:

        >>> result = runner.run_sequential(configs, batch_id="sequential_run_001")
        >>> print(result.batch_id)
        sequential_run_001

        Sequential with progress:

        >>> runner.on_progress(lambda c, t, r: print(f"{c}/{t}"))
        >>> result = runner.run_sequential(configs)
        1/10
        2/10
        ...
        """
        batch_id = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        runs = []
        total = len(configs)

        for i, config in enumerate(configs):
            run = self._execute_single(config)
            runs.append(run)

            if self._progress_callback:
                self._progress_callback(i + 1, total, run)

        result = ExperimentBatchResult(batch_id=batch_id, runs=runs)
        result.completed_at = datetime.now()
        return result

    def run_parallel(
        self, configs: list[ExperimentConfig], batch_id: Optional[str] = None
    ) -> ExperimentBatchResult:
        """
        Run experiments in parallel using a thread pool.

        Executes experiments concurrently using ThreadPoolExecutor with
        up to max_workers threads. This provides significant speedup for
        I/O-bound operations like API calls.

        Parameters
        ----------
        configs : list[ExperimentConfig]
            List of experiment configurations to execute.
        batch_id : str or None, optional
            Custom batch identifier. If None, auto-generated with timestamp.
            Default is None.

        Returns
        -------
        ExperimentBatchResult
            Results containing all experiment runs.

        Notes
        -----
        Results may be returned in a different order than the input configs
        due to parallel execution. Use config.id for matching if order matters.

        Examples
        --------
        Parallel execution with 4 workers:

        >>> runner = BatchRunner(executor=my_executor, max_workers=4)
        >>> result = runner.run_parallel(configs)
        >>> print(f"Completed in parallel: {result.completed}")
        Completed in parallel: 10

        High concurrency for fast APIs:

        >>> runner = BatchRunner(executor=fast_executor, max_workers=16)
        >>> result = runner.run_parallel(large_config_list)
        >>> print(f"Avg latency: {result.avg_latency_ms:.1f}ms")
        Avg latency: 45.2ms
        """
        batch_id = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        runs = []
        total = len(configs)
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._execute_single, config): config for config in configs}

            for future in as_completed(futures):
                run = future.result()
                runs.append(run)
                completed += 1

                if self._progress_callback:
                    self._progress_callback(completed, total, run)

        result = ExperimentBatchResult(batch_id=batch_id, runs=runs)
        result.completed_at = datetime.now()
        return result

    def run(
        self,
        configs: list[ExperimentConfig],
        batch_id: Optional[str] = None,
        parallel: bool = False,
    ) -> ExperimentBatchResult:
        """
        Run experiments with automatic mode selection.

        Convenience method that chooses between sequential and parallel
        execution based on the parallel flag and max_workers setting.
        If parallel=True and max_workers > 1, uses parallel execution;
        otherwise falls back to sequential.

        Parameters
        ----------
        configs : list[ExperimentConfig]
            List of experiment configurations to execute.
        batch_id : str or None, optional
            Custom batch identifier. If None, auto-generated with timestamp.
            Default is None.
        parallel : bool, optional
            Whether to run in parallel mode. Default is False.

        Returns
        -------
        ExperimentBatchResult
            Results containing all experiment runs.

        Examples
        --------
        Default sequential execution:

        >>> runner = BatchRunner(executor=my_executor)
        >>> result = runner.run(configs)

        Enable parallel execution:

        >>> runner = BatchRunner(executor=my_executor, max_workers=4)
        >>> result = runner.run(configs, parallel=True)

        Force sequential even with multiple workers:

        >>> runner = BatchRunner(executor=my_executor, max_workers=4)
        >>> result = runner.run(configs, parallel=False)  # Sequential
        """
        if parallel and self.max_workers > 1:
            return self.run_parallel(configs, batch_id)
        return self.run_sequential(configs, batch_id)


class AsyncBatchRunner:
    """
    Asynchronous batch runner for high-concurrency experiment execution.

    AsyncBatchRunner provides async/await support for running experiments
    concurrently using asyncio. This is ideal for high-concurrency workloads
    where you need to maximize throughput without the overhead of thread pools.

    The max_concurrent parameter controls the maximum number of experiments
    running simultaneously using a semaphore for rate limiting.

    Parameters
    ----------
    executor : Callable[..., Any]
        Async callable that executes prompts. Must be awaitable and receive
        (prompt, **kwargs), returning the response string.
    metrics_calculator : MetricsCalculator or None, optional
        Optional callable to compute metrics for each response. Note that
        this should be a sync function; it's called after awaiting the
        executor. Default is None.
    max_concurrent : int, optional
        Maximum number of experiments to run concurrently. Uses an asyncio
        Semaphore for rate limiting. Default is 5.

    Attributes
    ----------
    executor : Callable[..., Any]
        The configured async executor.
    metrics_calculator : MetricsCalculator or None
        The configured metrics calculator.
    max_concurrent : int
        Maximum concurrent executions.

    Examples
    --------
    Basic async execution:

    >>> async def async_executor(prompt: str, **kwargs) -> str:
    ...     async with aiohttp.ClientSession() as session:
    ...         async with session.post(API_URL, json={"prompt": prompt}) as resp:
    ...             data = await resp.json()
    ...             return data["response"]
    >>>
    >>> runner = AsyncBatchRunner(executor=async_executor, max_concurrent=10)
    >>> result = await runner.run(configs)
    >>> print(f"Completed: {result.completed}")
    Completed: 100

    With rate limiting:

    >>> # Limit to 5 concurrent requests to avoid rate limits
    >>> runner = AsyncBatchRunner(executor=async_executor, max_concurrent=5)
    >>> result = await runner.run(configs)

    With metrics calculation:

    >>> def calc_metrics(prompt: str, response: str, config) -> dict[str, float]:
    ...     return {"length": len(response)}
    >>>
    >>> runner = AsyncBatchRunner(
    ...     executor=async_executor,
    ...     metrics_calculator=calc_metrics,
    ...     max_concurrent=10,
    ... )
    >>> result = await runner.run(configs)

    Integration with asyncio event loop:

    >>> async def main():
    ...     runner = AsyncBatchRunner(executor=async_executor)
    ...     result = await runner.run(configs)
    ...     return result
    >>>
    >>> result = asyncio.run(main())

    See Also
    --------
    BatchRunner : Synchronous version using ThreadPoolExecutor.
    ExperimentOrchestrator : High-level interface (sync only).
    """

    def __init__(
        self,
        executor: Callable[..., Any],  # Async executor
        metrics_calculator: Optional[MetricsCalculator] = None,
        max_concurrent: int = 5,
    ):
        """
        Initialize the async batch runner.

        Parameters
        ----------
        executor : Callable[..., Any]
            Async function to execute prompts. Must return awaitable.
        metrics_calculator : MetricsCalculator or None, optional
            Function to calculate metrics for responses. Default is None.
        max_concurrent : int, optional
            Maximum concurrent executions. Default is 5.

        Examples
        --------
        Basic initialization:

        >>> runner = AsyncBatchRunner(executor=async_executor)

        With all options:

        >>> runner = AsyncBatchRunner(
        ...     executor=async_executor,
        ...     metrics_calculator=my_metrics,
        ...     max_concurrent=20,
        ... )
        """
        self.executor = executor
        self.metrics_calculator = metrics_calculator
        self.max_concurrent = max_concurrent
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._progress_callback: Optional[ProgressCallback] = None

    def on_progress(self, callback: ProgressCallback) -> "AsyncBatchRunner":
        """
        Set a callback for progress updates.

        Parameters
        ----------
        callback : ProgressCallback
            Function called with (completed, total, current_run) after
            each experiment finishes.

        Returns
        -------
        AsyncBatchRunner
            Self for method chaining.

        Examples
        --------
        >>> def log_progress(completed, total, current=None):
        ...     print(f"{completed}/{total}")
        >>>
        >>> runner = AsyncBatchRunner(executor=async_executor)
        >>> runner.on_progress(log_progress)
        """
        self._progress_callback = callback
        return self

    async def _execute_single(self, config: ExperimentConfig) -> ExperimentRun:
        """
        Execute a single experiment asynchronously.

        Internal method that handles async execution of one experiment
        with semaphore-based concurrency control.

        Parameters
        ----------
        config : ExperimentConfig
            The experiment configuration to execute.

        Returns
        -------
        ExperimentRun
            The completed experiment run with results or error.

        Examples
        --------
        This is an internal method. Use run() instead.

        >>> # Internal usage:
        >>> run = await runner._execute_single(config)
        """
        run = ExperimentRun(config=config, status=ExperimentStatus.RUNNING)
        run.started_at = datetime.now()

        if self._semaphore:
            await self._semaphore.acquire()

        try:
            start_time = time.time()
            response = await self.executor(config.prompt, **config.parameters)
            run.latency_ms = (time.time() - start_time) * 1000
            run.response = response
            run.status = ExperimentStatus.COMPLETED

            if self.metrics_calculator and response:
                run.metrics = self.metrics_calculator(config.prompt, response, config)

        except Exception as e:
            run.error = str(e)
            run.status = ExperimentStatus.FAILED
        finally:
            if self._semaphore:
                self._semaphore.release()

        run.completed_at = datetime.now()
        return run

    async def run(
        self, configs: list[ExperimentConfig], batch_id: Optional[str] = None
    ) -> ExperimentBatchResult:
        """
        Run experiments asynchronously with concurrency control.

        Executes all experiments concurrently using asyncio.gather with
        a semaphore limiting the number of simultaneous executions.

        Parameters
        ----------
        configs : list[ExperimentConfig]
            List of experiment configurations to execute.
        batch_id : str or None, optional
            Custom batch identifier. If None, auto-generated with timestamp.
            Default is None.

        Returns
        -------
        ExperimentBatchResult
            Results containing all experiment runs.

        Examples
        --------
        Basic async execution:

        >>> async def main():
        ...     runner = AsyncBatchRunner(executor=async_executor, max_concurrent=10)
        ...     result = await runner.run(configs)
        ...     print(f"Success rate: {result.success_rate:.1%}")
        >>>
        >>> asyncio.run(main())
        Success rate: 95.0%

        With custom batch ID:

        >>> result = await runner.run(configs, batch_id="async_batch_001")
        >>> print(result.batch_id)
        async_batch_001
        """
        batch_id = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        tasks = [self._execute_single(config) for config in configs]
        runs = await asyncio.gather(*tasks)

        result = ExperimentBatchResult(batch_id=batch_id, runs=list(runs))
        result.completed_at = datetime.now()
        return result


@dataclass
class ExperimentGrid:
    """
    Generate experiments from combinations of models and parameters.

    ExperimentGrid provides a declarative way to define grid search experiments
    over models and parameter values. It generates all combinations of the
    specified models and parameters, creating a complete experiment configuration
    for each combination.

    This is useful for hyperparameter tuning, model comparison, and systematic
    exploration of the parameter space.

    Parameters
    ----------
    base_prompt : str
        The prompt template used for all experiments in the grid.
    model_ids : list[str]
        List of model identifiers to test.
    parameters : dict[str, list[Any]], optional
        Dictionary mapping parameter names to lists of values to try.
        All combinations are generated (Cartesian product). Default is empty dict.
    name_template : str, optional
        Template for experiment names. Can include {index}, {model}, and
        any parameter names as format fields. Default is "exp_{index}".

    Attributes
    ----------
    base_prompt : str
        The prompt template.
    model_ids : list[str]
        Models to test.
    parameters : dict[str, list[Any]]
        Parameter combinations to explore.
    name_template : str
        Template for naming experiments.

    Examples
    --------
    Simple model comparison:

    >>> grid = ExperimentGrid(
    ...     base_prompt="Explain quantum computing",
    ...     model_ids=["gpt-4", "claude-3", "gemini-pro"],
    ... )
    >>> configs = grid.generate()
    >>> print(len(configs))  # 3 experiments (one per model)
    3

    Grid search over parameters:

    >>> grid = ExperimentGrid(
    ...     base_prompt="Write a poem about nature",
    ...     model_ids=["gpt-4"],
    ...     parameters={
    ...         "temperature": [0.0, 0.5, 1.0],
    ...         "max_tokens": [50, 100, 200],
    ...     },
    ... )
    >>> configs = grid.generate()
    >>> print(len(configs))  # 1 model x 3 temps x 3 max_tokens = 9
    9

    Full grid with multiple models and parameters:

    >>> grid = ExperimentGrid(
    ...     base_prompt="Summarize: {text}",
    ...     model_ids=["gpt-4", "claude-3"],
    ...     parameters={
    ...         "temperature": [0.0, 0.7],
    ...         "top_p": [0.9, 1.0],
    ...     },
    ...     name_template="summary_{model}_t{temperature}_p{top_p}",
    ... )
    >>> configs = grid.generate()
    >>> print(len(configs))  # 2 models x 2 temps x 2 top_p = 8
    8
    >>> print(configs[0].name)
    summary_gpt-4_t0.0_p0.9

    Using with orchestrator:

    >>> orchestrator = ExperimentOrchestrator(executor=my_executor)
    >>> result = orchestrator.run_grid(grid, parallel=True, max_workers=4)
    >>> print(f"Grid completed: {result.completed}/{result.total}")

    Checking grid size before execution:

    >>> print(f"Grid will generate {len(grid)} experiments")
    Grid will generate 8 experiments

    See Also
    --------
    ComparisonExperiment : Alternative for comparing prompts rather than parameters.
    ExperimentOrchestrator.run_grid : Execute a grid experiment.
    """

    base_prompt: str
    model_ids: list[str]
    parameters: dict[str, list[Any]] = field(default_factory=dict)
    name_template: str = "exp_{index}"

    def generate(self) -> list[ExperimentConfig]:
        """
        Generate all experiment configurations from the grid.

        Creates a configuration for each combination of models and parameters
        using the Cartesian product. Each config includes metadata with the
        grid index and parameter values.

        Returns
        -------
        list[ExperimentConfig]
            List of all generated experiment configurations.

        Examples
        --------
        Generate and inspect configurations:

        >>> grid = ExperimentGrid(
        ...     base_prompt="Hello",
        ...     model_ids=["gpt-4", "claude-3"],
        ...     parameters={"temperature": [0.0, 1.0]},
        ... )
        >>> configs = grid.generate()
        >>> for cfg in configs:
        ...     print(f"{cfg.id}: {cfg.model_id}, temp={cfg.parameters.get('temperature')}")
        grid_0: gpt-4, temp=0.0
        grid_1: gpt-4, temp=1.0
        grid_2: claude-3, temp=0.0
        grid_3: claude-3, temp=1.0
        """
        import itertools

        configs = []
        index = 0

        # Generate parameter combinations
        if self.parameters:
            param_keys = list(self.parameters.keys())
            param_values = [self.parameters[k] for k in param_keys]
            param_combos = list(itertools.product(*param_values))
        else:
            param_combos = [()]
            param_keys = []

        for model_id in self.model_ids:
            for param_combo in param_combos:
                params = dict(zip(param_keys, param_combo)) if param_keys else {}

                config = ExperimentConfig(
                    id=f"grid_{index}",
                    name=self.name_template.format(index=index, model=model_id, **params),
                    prompt=self.base_prompt,
                    model_id=model_id,
                    parameters=params,
                    metadata={
                        "grid_index": index,
                        "model": model_id,
                        **params,
                    },
                )
                configs.append(config)
                index += 1

        return configs

    def __len__(self) -> int:
        """
        Get the total number of experiments in the grid.

        Calculates the size without generating all configurations,
        useful for estimating execution time.

        Returns
        -------
        int
            Total number of experiments (models x parameter combinations).

        Examples
        --------
        >>> grid = ExperimentGrid(
        ...     base_prompt="Test",
        ...     model_ids=["gpt-4", "claude-3"],
        ...     parameters={"temperature": [0.0, 0.5, 1.0]},
        ... )
        >>> print(f"Grid size: {len(grid)}")
        Grid size: 6
        """
        import itertools

        if self.parameters:
            param_count = len(list(itertools.product(*self.parameters.values())))
        else:
            param_count = 1
        return len(self.model_ids) * param_count


@dataclass
class ComparisonExperiment:
    """
    Define an A/B comparison experiment with multiple prompts on same inputs.

    ComparisonExperiment enables comparing different prompt strategies by
    running each prompt variant against the same set of inputs. This is useful
    for prompt engineering, evaluating different phrasings, and A/B testing.

    Each prompt is tagged with its name, enabling easy filtering and analysis
    of results by prompt variant.

    Parameters
    ----------
    prompts : dict[str, str]
        Dictionary mapping prompt names to prompt templates. Templates should
        contain the input_placeholder for substitution.
    model_id : str
        The model to use for all experiments.
    inputs : list[str]
        List of input values to substitute into each prompt template.
    input_placeholder : str, optional
        The placeholder string in prompts to replace with inputs.
        Default is "{input}".

    Attributes
    ----------
    prompts : dict[str, str]
        The prompt name to template mapping.
    model_id : str
        The target model.
    inputs : list[str]
        Input values for substitution.
    input_placeholder : str
        The placeholder for substitution.

    Examples
    --------
    Compare formal vs casual prompts:

    >>> comparison = ComparisonExperiment(
    ...     prompts={
    ...         "formal": "Please provide a formal summary of: {input}",
    ...         "casual": "Give me the quick gist of: {input}",
    ...         "technical": "Provide a technical analysis of: {input}",
    ...     },
    ...     model_id="gpt-4",
    ...     inputs=[
    ...         "Machine learning is a subset of AI...",
    ...         "The stock market opened higher today...",
    ...     ],
    ... )
    >>> configs = comparison.generate()
    >>> print(len(configs))  # 3 prompts x 2 inputs = 6
    6

    Custom placeholder:

    >>> comparison = ComparisonExperiment(
    ...     prompts={
    ...         "v1": "Translate {{text}} to French",
    ...         "v2": "French translation: {{text}}",
    ...     },
    ...     model_id="gpt-4",
    ...     inputs=["Hello world", "Good morning"],
    ...     input_placeholder="{{text}}",
    ... )
    >>> configs = comparison.generate()

    Analyzing results by prompt variant:

    >>> result = orchestrator.run_comparison(comparison)
    >>> formal = result.get_by_tag("formal")
    >>> casual = result.get_by_tag("casual")
    >>>
    >>> formal_avg_latency = sum(r.latency_ms for r in formal) / len(formal)
    >>> casual_avg_latency = sum(r.latency_ms for r in casual) / len(casual)
    >>> print(f"Formal: {formal_avg_latency:.1f}ms, Casual: {casual_avg_latency:.1f}ms")

    Using with orchestrator:

    >>> orchestrator = ExperimentOrchestrator(executor=my_executor)
    >>> result = orchestrator.run_comparison(comparison, parallel=True)
    >>> print(f"Comparison completed: {result.success_rate:.1%}")

    See Also
    --------
    ExperimentGrid : Alternative for parameter grid search.
    ExperimentOrchestrator.run_comparison : Execute a comparison experiment.
    ExperimentBatchResult.get_by_tag : Filter results by prompt name.
    """

    prompts: dict[str, str]  # name -> prompt
    model_id: str
    inputs: list[str]  # Input values to substitute
    input_placeholder: str = "{input}"

    def generate(self) -> list[ExperimentConfig]:
        """
        Generate experiment configurations for all prompt/input combinations.

        Creates a configuration for each combination of prompt and input,
        with the prompt name stored as a tag for later filtering.

        Returns
        -------
        list[ExperimentConfig]
            List of all generated experiment configurations.

        Examples
        --------
        Generate and inspect configurations:

        >>> comparison = ComparisonExperiment(
        ...     prompts={"short": "Summarize: {input}", "long": "Detailed summary: {input}"},
        ...     model_id="gpt-4",
        ...     inputs=["Text 1", "Text 2"],
        ... )
        >>> configs = comparison.generate()
        >>> for cfg in configs:
        ...     print(f"{cfg.name}: tags={cfg.tags}, prompt={cfg.prompt[:30]}...")
        short_0: tags=['short'], prompt=Summarize: Text 1...
        long_1: tags=['long'], prompt=Detailed summary: Text 1...
        short_2: tags=['short'], prompt=Summarize: Text 2...
        long_3: tags=['long'], prompt=Detailed summary: Text 2...
        """
        configs = []
        index = 0

        for input_value in self.inputs:
            for prompt_name, prompt_template in self.prompts.items():
                prompt = prompt_template.replace(self.input_placeholder, input_value)

                config = ExperimentConfig(
                    id=f"compare_{index}",
                    name=f"{prompt_name}_{index}",
                    prompt=prompt,
                    model_id=self.model_id,
                    tags=[prompt_name],
                    metadata={
                        "prompt_name": prompt_name,
                        "input": input_value,
                    },
                )
                configs.append(config)
                index += 1

        return configs


class ExperimentOrchestrator:
    """
    High-level orchestrator for complex experiment workflows.

    ExperimentOrchestrator provides a unified interface for running various
    types of experiments including grid searches, A/B comparisons, and
    custom configuration lists. It manages batch results across multiple
    runs and provides aggregate reporting.

    The orchestrator maintains a history of all batches, enabling cross-batch
    analysis and comprehensive reporting.

    Parameters
    ----------
    executor : ModelExecutor
        Callable that executes prompts. Receives (prompt, **kwargs) and
        returns the response string.

    Attributes
    ----------
    executor : ModelExecutor
        The configured model executor.

    Examples
    --------
    Basic usage with grid experiment:

    >>> def my_executor(prompt: str, **kwargs) -> str:
    ...     return llm_client.complete(prompt, **kwargs)
    >>>
    >>> orchestrator = ExperimentOrchestrator(executor=my_executor)
    >>>
    >>> grid = ExperimentGrid(
    ...     base_prompt="Explain AI",
    ...     model_ids=["gpt-4", "claude-3"],
    ...     parameters={"temperature": [0.0, 0.5, 1.0]},
    ... )
    >>> result = orchestrator.run_grid(grid, parallel=True, max_workers=4)
    >>> print(f"Grid completed: {result.success_rate:.1%}")

    Running comparison experiments:

    >>> comparison = ComparisonExperiment(
    ...     prompts={"formal": "Please explain: {input}", "casual": "Explain: {input}"},
    ...     model_id="gpt-4",
    ...     inputs=["quantum physics", "machine learning"],
    ... )
    >>> result = orchestrator.run_comparison(comparison)

    With metrics calculation:

    >>> def my_metrics(prompt: str, response: str, config) -> dict[str, float]:
    ...     return {"word_count": len(response.split())}
    >>>
    >>> orchestrator = ExperimentOrchestrator(executor=my_executor)
    >>> orchestrator.set_metrics_calculator(my_metrics)
    >>> result = orchestrator.run_grid(grid)
    >>> print(result.aggregate_metrics())

    Managing multiple batches:

    >>> # Run multiple experiments
    >>> result1 = orchestrator.run_grid(grid1)
    >>> result2 = orchestrator.run_grid(grid2)
    >>> result3 = orchestrator.run_comparison(comparison)
    >>>
    >>> # List all batches
    >>> print(orchestrator.list_batches())
    ['batch_20240101_120000', 'batch_20240101_120500', 'batch_20240101_121000']
    >>>
    >>> # Get aggregate metrics across all batches
    >>> all_metrics = orchestrator.aggregate_all()

    Generating a report:

    >>> report = orchestrator.generate_report()
    >>> print(report)
    # Experiment Orchestration Report
    ...

    See Also
    --------
    BatchRunner : Lower-level batch execution.
    ExperimentGrid : Define grid search experiments.
    ComparisonExperiment : Define A/B comparison experiments.
    """

    def __init__(self, executor: ModelExecutor):
        """
        Initialize the experiment orchestrator.

        Parameters
        ----------
        executor : ModelExecutor
            Function to execute prompts.

        Examples
        --------
        >>> orchestrator = ExperimentOrchestrator(executor=my_executor)
        """
        self.executor = executor
        self._batches: dict[str, ExperimentBatchResult] = {}
        self._metrics_calculator: Optional[MetricsCalculator] = None

    def set_metrics_calculator(self, calculator: MetricsCalculator) -> "ExperimentOrchestrator":
        """
        Set the metrics calculator for all experiments.

        The calculator will be used for all subsequent experiment runs
        to compute custom metrics for each response.

        Parameters
        ----------
        calculator : MetricsCalculator
            Function to calculate metrics from responses.

        Returns
        -------
        ExperimentOrchestrator
            Self for method chaining.

        Examples
        --------
        >>> def calc_metrics(prompt: str, response: str, config) -> dict[str, float]:
        ...     return {"length": len(response), "words": len(response.split())}
        >>>
        >>> orchestrator = (
        ...     ExperimentOrchestrator(executor=my_executor)
        ...     .set_metrics_calculator(calc_metrics)
        ... )
        >>> result = orchestrator.run_grid(grid)
        >>> print(result.runs[0].metrics)
        {'length': 456, 'words': 78}
        """
        self._metrics_calculator = calculator
        return self

    def run_grid(
        self,
        grid: ExperimentGrid,
        parallel: bool = False,
        max_workers: int = 1,
    ) -> ExperimentBatchResult:
        """
        Execute a grid search experiment.

        Generates all configurations from the grid and executes them.
        Results are stored in the orchestrator's batch history.

        Parameters
        ----------
        grid : ExperimentGrid
            The grid configuration defining models and parameters.
        parallel : bool, optional
            Whether to run experiments in parallel. Default is False.
        max_workers : int, optional
            Maximum parallel workers. Default is 1.

        Returns
        -------
        ExperimentBatchResult
            Results from executing the grid.

        Examples
        --------
        Sequential grid execution:

        >>> grid = ExperimentGrid(
        ...     base_prompt="Test prompt",
        ...     model_ids=["gpt-4"],
        ...     parameters={"temperature": [0.0, 0.5, 1.0]},
        ... )
        >>> result = orchestrator.run_grid(grid)
        >>> print(f"Completed: {result.completed}/{result.total}")

        Parallel grid execution:

        >>> result = orchestrator.run_grid(grid, parallel=True, max_workers=4)
        >>> print(f"Avg latency: {result.avg_latency_ms:.1f}ms")
        """
        configs = grid.generate()
        runner = BatchRunner(
            executor=self.executor,
            metrics_calculator=self._metrics_calculator,
            max_workers=max_workers,
        )
        result = runner.run(configs, parallel=parallel)
        self._batches[result.batch_id] = result
        return result

    def run_comparison(
        self,
        comparison: ComparisonExperiment,
        parallel: bool = False,
        max_workers: int = 1,
    ) -> ExperimentBatchResult:
        """
        Execute an A/B comparison experiment.

        Generates all configurations from the comparison and executes them.
        Results are tagged by prompt variant for easy filtering.

        Parameters
        ----------
        comparison : ComparisonExperiment
            The comparison configuration defining prompts and inputs.
        parallel : bool, optional
            Whether to run experiments in parallel. Default is False.
        max_workers : int, optional
            Maximum parallel workers. Default is 1.

        Returns
        -------
        ExperimentBatchResult
            Results from executing the comparison.

        Examples
        --------
        Run comparison and analyze:

        >>> comparison = ComparisonExperiment(
        ...     prompts={"formal": "Please summarize: {input}", "casual": "Sum up: {input}"},
        ...     model_id="gpt-4",
        ...     inputs=["Document 1", "Document 2"],
        ... )
        >>> result = orchestrator.run_comparison(comparison)
        >>>
        >>> formal = result.get_by_tag("formal")
        >>> casual = result.get_by_tag("casual")
        >>> print(f"Formal avg: {sum(r.latency_ms for r in formal)/len(formal):.1f}ms")
        >>> print(f"Casual avg: {sum(r.latency_ms for r in casual)/len(casual):.1f}ms")
        """
        configs = comparison.generate()
        runner = BatchRunner(
            executor=self.executor,
            metrics_calculator=self._metrics_calculator,
            max_workers=max_workers,
        )
        result = runner.run(configs, parallel=parallel)
        self._batches[result.batch_id] = result
        return result

    def run_configs(
        self,
        configs: list[ExperimentConfig],
        parallel: bool = False,
        max_workers: int = 1,
        batch_id: Optional[str] = None,
    ) -> ExperimentBatchResult:
        """
        Execute a custom list of experiment configurations.

        Provides direct access to run arbitrary configurations without
        using grid or comparison generators.

        Parameters
        ----------
        configs : list[ExperimentConfig]
            List of experiment configurations to execute.
        parallel : bool, optional
            Whether to run experiments in parallel. Default is False.
        max_workers : int, optional
            Maximum parallel workers. Default is 1.
        batch_id : str or None, optional
            Custom batch identifier. Default is None (auto-generated).

        Returns
        -------
        ExperimentBatchResult
            Results from executing the configs.

        Examples
        --------
        Run custom configurations:

        >>> configs = [
        ...     ExperimentConfig(id="1", name="Test 1", prompt="Hello", model_id="gpt-4"),
        ...     ExperimentConfig(id="2", name="Test 2", prompt="World", model_id="gpt-4"),
        ... ]
        >>> result = orchestrator.run_configs(configs, batch_id="custom_batch")
        >>> print(result.batch_id)
        custom_batch
        """
        runner = BatchRunner(
            executor=self.executor,
            metrics_calculator=self._metrics_calculator,
            max_workers=max_workers,
        )
        result = runner.run(configs, batch_id=batch_id, parallel=parallel)
        self._batches[result.batch_id] = result
        return result

    def get_batch(self, batch_id: str) -> Optional[ExperimentBatchResult]:
        """
        Retrieve batch results by ID.

        Parameters
        ----------
        batch_id : str
            The batch identifier to look up.

        Returns
        -------
        ExperimentBatchResult or None
            The batch results, or None if not found.

        Examples
        --------
        >>> result = orchestrator.run_grid(grid)
        >>> later = orchestrator.get_batch(result.batch_id)
        >>> print(later.success_rate)
        0.95
        """
        return self._batches.get(batch_id)

    def list_batches(self) -> list[str]:
        """
        Get all batch IDs from the orchestrator history.

        Returns
        -------
        list[str]
            List of all batch identifiers.

        Examples
        --------
        >>> orchestrator.run_grid(grid1)
        >>> orchestrator.run_grid(grid2)
        >>> print(orchestrator.list_batches())
        ['batch_20240101_120000', 'batch_20240101_120500']
        """
        return list(self._batches.keys())

    def aggregate_all(self) -> dict[str, dict[str, float]]:
        """
        Aggregate metrics across all batches in history.

        Combines metrics from all completed experiments across all
        batches, computing summary statistics for each metric.

        Returns
        -------
        dict[str, dict[str, float]]
            Dictionary mapping metric names to their statistics.
            Each metric has keys: 'mean', 'min', 'max', 'count'.

        Examples
        --------
        >>> orchestrator.run_grid(grid1)
        >>> orchestrator.run_grid(grid2)
        >>>
        >>> metrics = orchestrator.aggregate_all()
        >>> for name, stats in metrics.items():
        ...     print(f"{name}: mean={stats['mean']:.2f}")
        word_count: mean=156.30
        sentiment: mean=0.72
        """
        all_metrics: dict[str, list[float]] = {}

        for batch in self._batches.values():
            for run in batch.get_by_status(ExperimentStatus.COMPLETED):
                for name, value in run.metrics.items():
                    if name not in all_metrics:
                        all_metrics[name] = []
                    all_metrics[name].append(value)

        result = {}
        for name, values in all_metrics.items():
            result[name] = {
                "mean": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "count": len(values),
            }
        return result

    def generate_report(self) -> str:
        """
        Generate a Markdown summary report of all experiments.

        Creates a comprehensive report including per-batch statistics,
        overall summary, and aggregate metrics across all batches.

        Returns
        -------
        str
            Markdown-formatted report string.

        Examples
        --------
        >>> orchestrator.run_grid(grid)
        >>> orchestrator.run_comparison(comparison)
        >>>
        >>> report = orchestrator.generate_report()
        >>> print(report)
        # Experiment Orchestration Report

        **Total Batches:** 2

        ## Batch: batch_20240101_120000

        - Total runs: 6
        - Completed: 6
        - Failed: 0
        - Success rate: 100.0%
        - Avg latency: 312.5ms
        ...

        >>> # Save to file
        >>> with open("report.md", "w") as f:
        ...     f.write(report)
        """
        lines = [
            "# Experiment Orchestration Report",
            "",
            f"**Total Batches:** {len(self._batches)}",
            "",
        ]

        total_runs = 0
        total_completed = 0
        total_failed = 0

        for batch_id, batch in self._batches.items():
            total_runs += batch.total
            total_completed += batch.completed
            total_failed += batch.failed

            lines.extend(
                [
                    f"## Batch: {batch_id}",
                    "",
                    f"- Total runs: {batch.total}",
                    f"- Completed: {batch.completed}",
                    f"- Failed: {batch.failed}",
                    f"- Success rate: {batch.success_rate:.1%}",
                    f"- Avg latency: {batch.avg_latency_ms:.1f}ms",
                    "",
                ]
            )

        lines.extend(
            [
                "## Summary",
                "",
                f"- Total experiments: {total_runs}",
                f"- Total completed: {total_completed}",
                f"- Total failed: {total_failed}",
                f"- Overall success rate: {total_completed / total_runs:.1%}"
                if total_runs
                else "- Overall success rate: N/A",
                "",
            ]
        )

        # Add aggregate metrics
        agg = self.aggregate_all()
        if agg:
            lines.extend(
                [
                    "## Aggregate Metrics",
                    "",
                ]
            )
            for metric, stats in agg.items():
                lines.append(
                    f"- **{metric}**: mean={stats['mean']:.3f}, "
                    f"min={stats['min']:.3f}, max={stats['max']:.3f}"
                )

        return "\n".join(lines)


def create_experiment_configs(
    prompts: list[str],
    model_id: str,
    prefix: str = "exp",
) -> list[ExperimentConfig]:
    """
    Create experiment configurations from a list of prompts.

    Convenience function for quickly creating experiment configurations
    from a list of prompt strings. Each prompt gets a sequential ID
    and name based on the provided prefix.

    Parameters
    ----------
    prompts : list[str]
        List of prompt strings to create experiments for.
    model_id : str
        The model identifier to use for all experiments.
    prefix : str, optional
        Prefix for experiment IDs and names. Default is "exp".

    Returns
    -------
    list[ExperimentConfig]
        List of experiment configurations, one per prompt.

    Examples
    --------
    Basic usage:

    >>> prompts = ["Hello", "How are you?", "What is AI?"]
    >>> configs = create_experiment_configs(prompts, model_id="gpt-4")
    >>> print(len(configs))
    3
    >>> print(configs[0].id)
    exp_0

    Custom prefix:

    >>> configs = create_experiment_configs(
    ...     prompts=["Test 1", "Test 2"],
    ...     model_id="claude-3",
    ...     prefix="test",
    ... )
    >>> print(configs[0].id)
    test_0
    >>> print(configs[1].name)
    test_1

    Using with BatchRunner:

    >>> configs = create_experiment_configs(prompts, "gpt-4")
    >>> runner = BatchRunner(executor=my_executor)
    >>> result = runner.run(configs)

    See Also
    --------
    run_quick_batch : Higher-level function that creates and runs configs.
    ExperimentConfig : The configuration class being created.
    """
    return [
        ExperimentConfig(
            id=f"{prefix}_{i}",
            name=f"{prefix}_{i}",
            prompt=prompt,
            model_id=model_id,
        )
        for i, prompt in enumerate(prompts)
    ]


def run_quick_batch(
    prompts: list[str],
    executor: ModelExecutor,
    model_id: str = "default",
) -> ExperimentBatchResult:
    """
    Quickly run a batch of prompts with minimal configuration.

    High-level convenience function for running experiments with
    minimal setup. Creates configurations from prompts and executes
    them sequentially. For more control, use BatchRunner directly.

    Parameters
    ----------
    prompts : list[str]
        List of prompt strings to execute.
    executor : ModelExecutor
        Function to execute prompts.
    model_id : str, optional
        Model identifier for the experiments. Default is "default".

    Returns
    -------
    ExperimentBatchResult
        Results from executing all prompts.

    Examples
    --------
    Minimal batch execution:

    >>> def my_executor(prompt: str, **kwargs) -> str:
    ...     return llm.complete(prompt)
    >>>
    >>> prompts = ["What is AI?", "Explain machine learning", "Define neural networks"]
    >>> result = run_quick_batch(prompts, my_executor)
    >>> print(f"Success rate: {result.success_rate:.1%}")
    Success rate: 100.0%

    With custom model ID:

    >>> result = run_quick_batch(prompts, my_executor, model_id="gpt-4")
    >>> for run in result.runs:
    ...     print(f"{run.config.id}: {run.latency_ms:.1f}ms")

    Quick testing during development:

    >>> # Fast iteration during prompt development
    >>> test_prompts = ["Version A: {input}", "Version B: {input}"]
    >>> result = run_quick_batch(test_prompts, mock_executor)
    >>> for run in result.runs:
    ...     print(f"{run.config.prompt[:30]}... -> {run.response[:50]}...")

    See Also
    --------
    create_experiment_configs : Lower-level config creation.
    BatchRunner : For more control over execution.
    ExperimentOrchestrator : For complex workflows.
    """
    configs = create_experiment_configs(prompts, model_id)
    runner = BatchRunner(executor=executor)
    return runner.run(configs)


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Older code and tests may import BatchResult. The canonical name is
# ExperimentBatchResult.
BatchResult = ExperimentBatchResult
