"""Experiment runner with YAML/JSON config support.

This module provides tools for running LLM experiments, either programmatically
or from configuration files. Supports async execution for parallel processing.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

from insideLLMs.models.base import Model
from insideLLMs.probes.base import Probe
from insideLLMs.registry import (
    NotFoundError,
    dataset_registry,
    ensure_builtins_registered,
    model_registry,
    probe_registry,
)
from insideLLMs.types import (
    ConfigDict,
    ExperimentResult,
    ProbeCategory,
    ProbeResult,
    ProbeScore,
    ResultStatus,
)


class ProbeRunner:
    """Runs a probe against a model on a dataset.

    The ProbeRunner orchestrates the execution of probes, handling error
    recovery, progress tracking, and result aggregation.

    Attributes:
        model: The model to test.
        probe: The probe to run.

    Example:
        >>> runner = ProbeRunner(model, probe)
        >>> results = runner.run(dataset)
        >>> print(f"Success rate: {runner.success_rate:.1%}")
    """

    def __init__(self, model: Model, probe: Probe):
        """Initialize the runner.

        Args:
            model: The model to test.
            probe: The probe to run.
        """
        self.model = model
        self.probe = probe
        self._results: List[Dict[str, Any]] = []

    def run(
        self,
        prompt_set: List[Any],
        *,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        stop_on_error: bool = False,
        **probe_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Run the probe on the model for each item in the prompt set.

        Args:
            prompt_set: List of inputs to test.
            progress_callback: Optional callback(current, total) for progress updates.
            stop_on_error: If True, stop execution on first error.
            **probe_kwargs: Additional arguments passed to the probe.

        Returns:
            A list of result dictionaries with 'input', 'output', and optionally 'error'.
        """
        import time

        results = []
        total = len(prompt_set)

        for i, item in enumerate(prompt_set):
            if progress_callback:
                progress_callback(i, total)

            start_time = time.perf_counter()
            try:
                output = self.probe.run(self.model, item, **probe_kwargs)
                latency_ms = (time.perf_counter() - start_time) * 1000
                results.append({
                    "input": item,
                    "output": output,
                    "latency_ms": latency_ms,
                    "status": "success",
                })
            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                error_result = {
                    "input": item,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "latency_ms": latency_ms,
                    "status": "error",
                }
                results.append(error_result)

                if stop_on_error:
                    break

        if progress_callback:
            progress_callback(total, total)

        self._results = results
        return results

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of the last run."""
        if not self._results:
            return 0.0
        successes = sum(1 for r in self._results if r.get("status") == "success")
        return successes / len(self._results)

    @property
    def error_count(self) -> int:
        """Count errors from the last run."""
        return sum(1 for r in self._results if r.get("status") == "error")


class AsyncProbeRunner:
    """Async version of ProbeRunner for parallel execution.

    Use this when you need to run probes concurrently, which is especially
    useful for API-based models that can handle multiple requests in parallel.

    Example:
        >>> runner = AsyncProbeRunner(model, probe)
        >>> results = await runner.run(dataset, concurrency=10)
    """

    def __init__(self, model: Model, probe: Probe):
        """Initialize the async runner.

        Args:
            model: The model to test.
            probe: The probe to run.
        """
        self.model = model
        self.probe = probe
        self._results: List[Dict[str, Any]] = []

    async def run(
        self,
        prompt_set: List[Any],
        *,
        concurrency: int = 5,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **probe_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Run the probe on all items with controlled concurrency.

        Args:
            prompt_set: List of inputs to test.
            concurrency: Maximum number of concurrent executions.
            progress_callback: Optional callback(current, total) for progress updates.
            **probe_kwargs: Additional arguments passed to the probe.

        Returns:
            A list of result dictionaries.
        """
        import time

        semaphore = asyncio.Semaphore(concurrency)
        results: List[Dict[str, Any]] = [None] * len(prompt_set)  # type: ignore
        completed = 0
        total = len(prompt_set)

        async def run_single(index: int, item: Any) -> None:
            nonlocal completed
            async with semaphore:
                start_time = time.perf_counter()
                try:
                    # Run in thread pool for sync models
                    loop = asyncio.get_event_loop()
                    output = await loop.run_in_executor(
                        None,
                        lambda: self.probe.run(self.model, item, **probe_kwargs),
                    )
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    results[index] = {
                        "input": item,
                        "output": output,
                        "latency_ms": latency_ms,
                        "status": "success",
                    }
                except Exception as e:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    results[index] = {
                        "input": item,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "latency_ms": latency_ms,
                        "status": "error",
                    }
                finally:
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)

        tasks = [run_single(i, item) for i, item in enumerate(prompt_set)]
        await asyncio.gather(*tasks)

        self._results = results
        return results


def run_probe(
    model: Model,
    probe: Probe,
    prompt_set: List[Any],
    **probe_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Convenience function to run a probe on a model.

    Args:
        model: The model to test.
        probe: The probe to run.
        prompt_set: List of inputs to test.
        **probe_kwargs: Additional arguments for the probe.

    Returns:
        A list of result dictionaries.
    """
    runner = ProbeRunner(model, probe)
    return runner.run(prompt_set, **probe_kwargs)


async def run_probe_async(
    model: Model,
    probe: Probe,
    prompt_set: List[Any],
    concurrency: int = 5,
    **probe_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Convenience function to run a probe asynchronously.

    Args:
        model: The model to test.
        probe: The probe to run.
        prompt_set: List of inputs to test.
        concurrency: Maximum concurrent executions.
        **probe_kwargs: Additional arguments for the probe.

    Returns:
        A list of result dictionaries.
    """
    runner = AsyncProbeRunner(model, probe)
    return await runner.run(prompt_set, concurrency=concurrency, **probe_kwargs)


def load_config(path: Union[str, Path]) -> ConfigDict:
    """Load a configuration file.

    Supports YAML (.yaml, .yml) and JSON (.json) formats.

    Args:
        path: Path to the configuration file.

    Returns:
        The parsed configuration dictionary.

    Raises:
        ValueError: If the file format is not supported.
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix in (".yaml", ".yml"):
        with open(path) as f:
            return yaml.safe_load(f)
    elif path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")


def _resolve_path(path: str, base_dir: Path) -> Path:
    """Resolve a path relative to a base directory."""
    p = Path(path)
    if p.is_absolute():
        return p
    return base_dir / p


def _create_model_from_config(config: ConfigDict) -> Model:
    """Create a model instance from configuration.

    Uses the registry if available, falls back to direct imports.
    """
    ensure_builtins_registered()

    model_type = config["type"]
    model_args = config.get("args", {})

    try:
        return model_registry.get(model_type, **model_args)
    except NotFoundError:
        # Fallback to direct import for backwards compatibility
        from insideLLMs.models import (
            AnthropicModel,
            DummyModel,
            HuggingFaceModel,
            OpenAIModel,
        )

        model_map = {
            "dummy": DummyModel,
            "openai": OpenAIModel,
            "huggingface": HuggingFaceModel,
            "anthropic": AnthropicModel,
        }

        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}")

        return model_map[model_type](**model_args)


def _create_probe_from_config(config: ConfigDict) -> Probe:
    """Create a probe instance from configuration.

    Uses the registry if available, falls back to direct imports.
    """
    ensure_builtins_registered()

    probe_type = config["type"]
    probe_args = config.get("args", {})

    try:
        return probe_registry.get(probe_type, **probe_args)
    except NotFoundError:
        # Fallback to direct import for backwards compatibility
        from insideLLMs.probes import (
            AttackProbe,
            BiasProbe,
            FactualityProbe,
            LogicProbe,
        )

        probe_map = {
            "logic": LogicProbe,
            "bias": BiasProbe,
            "attack": AttackProbe,
            "factuality": FactualityProbe,
        }

        if probe_type not in probe_map:
            raise ValueError(f"Unknown probe type: {probe_type}")

        return probe_map[probe_type](**probe_args)


def _load_dataset_from_config(config: ConfigDict, base_dir: Path) -> List[Any]:
    """Load a dataset from configuration."""
    ensure_builtins_registered()

    format_type = config["format"]

    if format_type in ("csv", "jsonl"):
        path = _resolve_path(config["path"], base_dir)
        try:
            loader = dataset_registry.get_factory(format_type)
            return loader(str(path))
        except NotFoundError:
            from insideLLMs.dataset_utils import load_csv_dataset, load_jsonl_dataset

            if format_type == "csv":
                return load_csv_dataset(str(path))
            else:
                return load_jsonl_dataset(str(path))

    elif format_type == "hf":
        try:
            loader = dataset_registry.get_factory("hf")
            return loader(config["name"], split=config.get("split", "test"))
        except NotFoundError:
            from insideLLMs.dataset_utils import load_hf_dataset

            return load_hf_dataset(config["name"], split=config.get("split", "test"))

    else:
        raise ValueError(f"Unknown dataset format: {format_type}")


def run_experiment_from_config(
    config_path: Union[str, Path],
    *,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """Run an experiment from a configuration file.

    The configuration file should specify:
    - model: type and args
    - probe: type and args
    - dataset: format, path/name, and optional split

    Args:
        config_path: Path to the YAML or JSON configuration file.
        progress_callback: Optional callback for progress updates.

    Returns:
        A list of result dictionaries.

    Example config (YAML):
        ```yaml
        model:
          type: openai
          args:
            model_name: gpt-4
        probe:
          type: factuality
          args: {}
        dataset:
          format: jsonl
          path: data/questions.jsonl
        ```
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    base_dir = config_path.parent

    model = _create_model_from_config(config["model"])
    probe = _create_probe_from_config(config["probe"])
    dataset = _load_dataset_from_config(config["dataset"], base_dir)

    runner = ProbeRunner(model, probe)
    return runner.run(dataset, progress_callback=progress_callback)


async def run_experiment_from_config_async(
    config_path: Union[str, Path],
    *,
    concurrency: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """Run an experiment asynchronously from a configuration file.

    Like run_experiment_from_config, but uses async execution for
    parallel processing.

    Args:
        config_path: Path to the configuration file.
        concurrency: Maximum number of concurrent executions.
        progress_callback: Optional callback for progress updates.

    Returns:
        A list of result dictionaries.
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    base_dir = config_path.parent

    model = _create_model_from_config(config["model"])
    probe = _create_probe_from_config(config["probe"])
    dataset = _load_dataset_from_config(config["dataset"], base_dir)

    runner = AsyncProbeRunner(model, probe)
    return await runner.run(dataset, concurrency=concurrency, progress_callback=progress_callback)


def create_experiment_result(
    model: Model,
    probe: Probe,
    results: List[Dict[str, Any]],
    config: Optional[ConfigDict] = None,
) -> ExperimentResult:
    """Create a structured ExperimentResult from raw results.

    Args:
        model: The model that was tested.
        probe: The probe that was run.
        results: The raw result dictionaries.
        config: Optional configuration used for the experiment.

    Returns:
        An ExperimentResult with structured data and scores.
    """
    probe_results = []
    for r in results:
        status = ResultStatus.SUCCESS if r.get("status") == "success" else ResultStatus.ERROR
        probe_results.append(
            ProbeResult(
                input=r.get("input"),
                output=r.get("output"),
                status=status,
                error=r.get("error"),
                latency_ms=r.get("latency_ms"),
            )
        )

    # Calculate scores
    score = probe.score(probe_results) if hasattr(probe, "score") else None

    # Determine category
    category = getattr(probe, "category", ProbeCategory.CUSTOM)
    if isinstance(category, str):
        category = ProbeCategory(category)

    return ExperimentResult(
        experiment_id=str(uuid.uuid4())[:8],
        model_info=model.info(),
        probe_name=probe.name,
        probe_category=category,
        results=probe_results,
        score=score,
        started_at=datetime.now(),
        completed_at=datetime.now(),
        config=config or {},
    )
