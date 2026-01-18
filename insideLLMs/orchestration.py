"""
Experiment orchestration utilities for batch LLM experiments.

Provides tools for:
- Batch experiment execution
- Experiment scheduling and queuing
- Progress tracking and reporting
- Result aggregation across experiments
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
    """Status of an experiment."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    id: str
    name: str
    prompt: str
    model_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher = higher priority


@dataclass
class ExperimentRun:
    """Record of a single experiment execution."""

    config: ExperimentConfig
    status: ExperimentStatus = ExperimentStatus.PENDING
    response: Optional[str] = None
    latency_ms: float = 0.0
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Get experiment duration in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return self.latency_ms

    @property
    def is_complete(self) -> bool:
        """Check if experiment is complete."""
        return self.status in (
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
        )


@dataclass
class BatchResult:
    """Results from a batch of experiments."""

    batch_id: str
    runs: list[ExperimentRun]
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    @property
    def total(self) -> int:
        return len(self.runs)

    @property
    def completed(self) -> int:
        return sum(1 for r in self.runs if r.status == ExperimentStatus.COMPLETED)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.runs if r.status == ExperimentStatus.FAILED)

    @property
    def success_rate(self) -> float:
        total = self.completed + self.failed
        return self.completed / total if total > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        completed_runs = [r for r in self.runs if r.status == ExperimentStatus.COMPLETED]
        if not completed_runs:
            return 0.0
        return sum(r.latency_ms for r in completed_runs) / len(completed_runs)

    def get_by_status(self, status: ExperimentStatus) -> list[ExperimentRun]:
        """Get runs with a specific status."""
        return [r for r in self.runs if r.status == status]

    def get_by_tag(self, tag: str) -> list[ExperimentRun]:
        """Get runs with a specific tag."""
        return [r for r in self.runs if tag in r.config.tags]

    def get_by_model(self, model_id: str) -> list[ExperimentRun]:
        """Get runs for a specific model."""
        return [r for r in self.runs if r.config.model_id == model_id]

    def aggregate_metrics(self) -> dict[str, dict[str, float]]:
        """Aggregate metrics across all completed runs."""
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
    """Protocol for progress callbacks."""

    def __call__(
        self,
        completed: int,
        total: int,
        current: Optional[ExperimentRun] = None,
    ) -> None:
        """Called when progress updates."""
        ...


class ModelExecutor(Protocol):
    """Protocol for model execution."""

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """Execute a prompt and return response."""
        ...


class MetricsCalculator(Protocol):
    """Protocol for calculating metrics from responses."""

    def __call__(
        self,
        prompt: str,
        response: str,
        config: ExperimentConfig,
    ) -> dict[str, float]:
        """Calculate metrics for a response."""
        ...


class ExperimentQueue:
    """Queue for managing experiment execution order."""

    def __init__(self):
        """Initialize the queue."""
        self._pending: list[ExperimentConfig] = []
        self._running: dict[str, ExperimentConfig] = {}
        self._completed: list[ExperimentRun] = []

    def add(self, config: ExperimentConfig) -> None:
        """Add an experiment to the queue."""
        self._pending.append(config)
        # Sort by priority (higher priority first)
        self._pending.sort(key=lambda x: -x.priority)

    def add_batch(self, configs: list[ExperimentConfig]) -> None:
        """Add multiple experiments."""
        for config in configs:
            self.add(config)

    def next(self) -> Optional[ExperimentConfig]:
        """Get the next experiment to run."""
        if not self._pending:
            return None
        config = self._pending.pop(0)
        self._running[config.id] = config
        return config

    def complete(self, run: ExperimentRun) -> None:
        """Mark an experiment as complete."""
        if run.config.id in self._running:
            del self._running[run.config.id]
        self._completed.append(run)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._pending) == 0 and len(self._running) == 0

    def get_results(self) -> list[ExperimentRun]:
        """Get all completed runs."""
        return list(self._completed)


class BatchRunner:
    """Run batches of experiments with various execution modes."""

    def __init__(
        self,
        executor: ModelExecutor,
        metrics_calculator: Optional[MetricsCalculator] = None,
        max_workers: int = 1,
        retry_on_failure: bool = False,
        max_retries: int = 3,
    ):
        """Initialize the batch runner.

        Args:
            executor: Function to execute prompts.
            metrics_calculator: Optional function to calculate metrics.
            max_workers: Maximum parallel workers.
            retry_on_failure: Whether to retry failed experiments.
            max_retries: Maximum retry attempts.
        """
        self.executor = executor
        self.metrics_calculator = metrics_calculator
        self.max_workers = max_workers
        self.retry_on_failure = retry_on_failure
        self.max_retries = max_retries
        self._progress_callback: Optional[ProgressCallback] = None

    def on_progress(self, callback: ProgressCallback) -> "BatchRunner":
        """Set progress callback.

        Args:
            callback: Function called on progress updates.

        Returns:
            Self for chaining.
        """
        self._progress_callback = callback
        return self

    def _execute_single(self, config: ExperimentConfig, retry_count: int = 0) -> ExperimentRun:
        """Execute a single experiment."""
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
    ) -> BatchResult:
        """Run experiments sequentially.

        Args:
            configs: List of experiment configurations.
            batch_id: Optional batch identifier.

        Returns:
            Batch results.
        """
        batch_id = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        runs = []
        total = len(configs)

        for i, config in enumerate(configs):
            run = self._execute_single(config)
            runs.append(run)

            if self._progress_callback:
                self._progress_callback(i + 1, total, run)

        result = BatchResult(batch_id=batch_id, runs=runs)
        result.completed_at = datetime.now()
        return result

    def run_parallel(
        self, configs: list[ExperimentConfig], batch_id: Optional[str] = None
    ) -> BatchResult:
        """Run experiments in parallel.

        Args:
            configs: List of experiment configurations.
            batch_id: Optional batch identifier.

        Returns:
            Batch results.
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

        result = BatchResult(batch_id=batch_id, runs=runs)
        result.completed_at = datetime.now()
        return result

    def run(
        self,
        configs: list[ExperimentConfig],
        batch_id: Optional[str] = None,
        parallel: bool = False,
    ) -> BatchResult:
        """Run experiments.

        Args:
            configs: List of experiment configurations.
            batch_id: Optional batch identifier.
            parallel: Whether to run in parallel.

        Returns:
            Batch results.
        """
        if parallel and self.max_workers > 1:
            return self.run_parallel(configs, batch_id)
        return self.run_sequential(configs, batch_id)


class AsyncBatchRunner:
    """Async version of batch runner."""

    def __init__(
        self,
        executor: Callable[..., Any],  # Async executor
        metrics_calculator: Optional[MetricsCalculator] = None,
        max_concurrent: int = 5,
    ):
        """Initialize async batch runner.

        Args:
            executor: Async function to execute prompts.
            metrics_calculator: Optional function to calculate metrics.
            max_concurrent: Maximum concurrent executions.
        """
        self.executor = executor
        self.metrics_calculator = metrics_calculator
        self.max_concurrent = max_concurrent
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._progress_callback: Optional[ProgressCallback] = None

    def on_progress(self, callback: ProgressCallback) -> "AsyncBatchRunner":
        """Set progress callback."""
        self._progress_callback = callback
        return self

    async def _execute_single(self, config: ExperimentConfig) -> ExperimentRun:
        """Execute a single experiment asynchronously."""
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
    ) -> BatchResult:
        """Run experiments asynchronously.

        Args:
            configs: List of experiment configurations.
            batch_id: Optional batch identifier.

        Returns:
            Batch results.
        """
        batch_id = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        tasks = [self._execute_single(config) for config in configs]
        runs = await asyncio.gather(*tasks)

        result = BatchResult(batch_id=batch_id, runs=list(runs))
        result.completed_at = datetime.now()
        return result


@dataclass
class ExperimentGrid:
    """Define a grid of experiments from parameter combinations."""

    base_prompt: str
    model_ids: list[str]
    parameters: dict[str, list[Any]] = field(default_factory=dict)
    name_template: str = "exp_{index}"

    def generate(self) -> list[ExperimentConfig]:
        """Generate all experiment configurations."""
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
        """Get total number of experiments."""
        import itertools

        if self.parameters:
            param_count = len(list(itertools.product(*self.parameters.values())))
        else:
            param_count = 1
        return len(self.model_ids) * param_count


@dataclass
class ComparisonExperiment:
    """Experiment comparing multiple prompts on the same inputs."""

    prompts: dict[str, str]  # name -> prompt
    model_id: str
    inputs: list[str]  # Input values to substitute
    input_placeholder: str = "{input}"

    def generate(self) -> list[ExperimentConfig]:
        """Generate experiment configurations."""
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
    """High-level orchestrator for complex experiment workflows."""

    def __init__(self, executor: ModelExecutor):
        """Initialize the orchestrator.

        Args:
            executor: Function to execute prompts.
        """
        self.executor = executor
        self._batches: dict[str, BatchResult] = {}
        self._metrics_calculator: Optional[MetricsCalculator] = None

    def set_metrics_calculator(self, calculator: MetricsCalculator) -> "ExperimentOrchestrator":
        """Set the metrics calculator.

        Args:
            calculator: Function to calculate metrics.

        Returns:
            Self for chaining.
        """
        self._metrics_calculator = calculator
        return self

    def run_grid(
        self,
        grid: ExperimentGrid,
        parallel: bool = False,
        max_workers: int = 1,
    ) -> BatchResult:
        """Run a grid of experiments.

        Args:
            grid: Experiment grid configuration.
            parallel: Whether to run in parallel.
            max_workers: Maximum parallel workers.

        Returns:
            Batch results.
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
    ) -> BatchResult:
        """Run a comparison experiment.

        Args:
            comparison: Comparison experiment configuration.
            parallel: Whether to run in parallel.
            max_workers: Maximum parallel workers.

        Returns:
            Batch results.
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
    ) -> BatchResult:
        """Run a list of experiment configs.

        Args:
            configs: List of experiment configurations.
            parallel: Whether to run in parallel.
            max_workers: Maximum parallel workers.
            batch_id: Optional batch identifier.

        Returns:
            Batch results.
        """
        runner = BatchRunner(
            executor=self.executor,
            metrics_calculator=self._metrics_calculator,
            max_workers=max_workers,
        )
        result = runner.run(configs, batch_id=batch_id, parallel=parallel)
        self._batches[result.batch_id] = result
        return result

    def get_batch(self, batch_id: str) -> Optional[BatchResult]:
        """Get batch results by ID."""
        return self._batches.get(batch_id)

    def list_batches(self) -> list[str]:
        """List all batch IDs."""
        return list(self._batches.keys())

    def aggregate_all(self) -> dict[str, dict[str, float]]:
        """Aggregate metrics across all batches."""
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
        """Generate a summary report of all experiments."""
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
    """Create experiment configs from a list of prompts.

    Args:
        prompts: List of prompts.
        model_id: Model to use.
        prefix: ID prefix.

    Returns:
        List of experiment configurations.
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
) -> BatchResult:
    """Quick utility to run a batch of prompts.

    Args:
        prompts: List of prompts to run.
        executor: Function to execute prompts.
        model_id: Model identifier.

    Returns:
        Batch results.
    """
    configs = create_experiment_configs(prompts, model_id)
    runner = BatchRunner(executor=executor)
    return runner.run(configs)
