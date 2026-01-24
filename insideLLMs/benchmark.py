"""Benchmarking tools for comparing LLM models and probes.

This module provides comprehensive benchmarking utilities for evaluating and
comparing the performance of language models across standardized probes. It
enables systematic comparison of multiple models on the same tasks, or multiple
probes on the same model, with detailed metrics collection and result persistence.

Benchmark outputs are part of the library's public, serialized API surface.
They include a ``schema_version`` field so downstream tooling can validate and
evolve contracts in a SemVer-compatible way.

Key Features:
    - **Model Comparison**: Run the same probe across multiple LLMs to compare
      performance metrics like latency, success rate, and throughput.
    - **Probe Comparison**: Evaluate a single model across different probes to
      understand its strengths and weaknesses across various tasks.
    - **Metrics Collection**: Automatically calculates success rates, error rates,
      latencies, and timing statistics for each benchmark run.
    - **Result Persistence**: Save benchmark results to JSON files for later
      analysis, visualization, or integration with other tools.
    - **Schema Versioning**: All outputs include a schema version for forward
      compatibility and validation.

Classes:
    ModelBenchmark: Compare multiple models on a single probe and dataset.
    ProbeBenchmark: Compare multiple probes on a single model and dataset.

Functions:
    _to_serializable: Internal helper to convert objects to JSON-serializable form.

Example: Comparing multiple models on a logic probe
    >>> from insideLLMs.benchmark import ModelBenchmark
    >>> from insideLLMs.models import OpenAIModel, AnthropicModel
    >>> from insideLLMs.probes.logic import SyllogismProbe
    >>>
    >>> # Create models to compare
    >>> models = [
    ...     OpenAIModel(model_name="gpt-4"),
    ...     AnthropicModel(model_name="claude-3-opus"),
    ... ]
    >>>
    >>> # Create the benchmark
    >>> probe = SyllogismProbe(name="syllogism_test")
    >>> benchmark = ModelBenchmark(models, probe, name="Logic Comparison")
    >>>
    >>> # Define test prompts
    >>> prompts = [
    ...     "All humans are mortal. Socrates is human. Is Socrates mortal?",
    ...     "All birds can fly. Penguins are birds. Can penguins fly?",
    ... ]
    >>>
    >>> # Run the benchmark
    >>> results = benchmark.run(prompts)
    Benchmarking gpt-4...
    Benchmarking claude-3-opus...
    >>>
    >>> # Analyze results
    >>> comparison = benchmark.compare_models()
    >>> print(f"Fastest model: {comparison['rankings']['total_time'][0]}")
    Fastest model: gpt-4
    >>>
    >>> # Save for later analysis
    >>> benchmark.save_results("logic_benchmark.json")

Example: Comparing multiple probes on a single model
    >>> from insideLLMs.benchmark import ProbeBenchmark
    >>> from insideLLMs.models import OpenAIModel
    >>> from insideLLMs.probes.logic import SyllogismProbe
    >>> from insideLLMs.probes.factuality import FactualityProbe
    >>>
    >>> # Create probes to compare
    >>> probes = [
    ...     SyllogismProbe(name="logic"),
    ...     FactualityProbe(name="facts"),
    ... ]
    >>>
    >>> # Create the benchmark
    >>> model = OpenAIModel(model_name="gpt-4")
    >>> benchmark = ProbeBenchmark(model, probes, name="GPT-4 Evaluation")
    >>>
    >>> # Run across all probes
    >>> results = benchmark.run(test_prompts)
    Benchmarking logic...
    Benchmarking facts...
    >>>
    >>> # Check probe performance
    >>> for probe_result in results["probes"]:
    ...     print(f"{probe_result['probe']}: {probe_result['metrics']['success_rate']:.2%}")
    logic: 95.00%
    facts: 88.00%

Example: Full benchmark workflow with persistence
    >>> from insideLLMs.benchmark import ModelBenchmark
    >>> from insideLLMs.models import OpenAIModel
    >>> from insideLLMs.probes.bias import GenderBiasProbe
    >>> import json
    >>>
    >>> # Setup
    >>> models = [OpenAIModel(model_name="gpt-4"), OpenAIModel(model_name="gpt-3.5-turbo")]
    >>> probe = GenderBiasProbe(name="gender_bias")
    >>> benchmark = ModelBenchmark(models, probe, name="Bias Comparison")
    >>>
    >>> # Run with custom schema version
    >>> results = benchmark.run(bias_prompts, schema_version="1.0.1")
    >>>
    >>> # Save results
    >>> benchmark.save_results("bias_benchmark.json")
    >>>
    >>> # Later: load and analyze
    >>> with open("bias_benchmark.json") as f:
    ...     saved_results = json.load(f)
    >>> print(f"Schema version: {saved_results['schema_version']}")
    Schema version: 1.0.1

Example: Accessing detailed metrics
    >>> # After running a benchmark
    >>> results = benchmark.run(prompts)
    >>>
    >>> # Access per-model metrics
    >>> for model_result in results["models"]:
    ...     model_name = model_result["model"]["name"]
    ...     metrics = model_result["metrics"]
    ...     print(f"Model: {model_name}")
    ...     print(f"  Total time: {metrics['total_time']:.2f}s")
    ...     print(f"  Avg per item: {metrics['avg_time_per_item']:.3f}s")
    ...     print(f"  Success rate: {metrics['success_rate']:.2%}")
    ...     print(f"  Mean latency: {metrics['mean_latency_ms']:.1f}ms")
    Model: gpt-4
      Total time: 15.23s
      Avg per item: 0.305s
      Success rate: 98.00%
      Mean latency: 287.5ms
    Model: gpt-3.5-turbo
      Total time: 8.45s
      Avg per item: 0.169s
      Success rate: 92.00%
      Mean latency: 156.2ms

Notes:
    - Benchmark results include timestamps for tracking when evaluations were run.
    - The ``schema_version`` field enables validation of result formats against
      known schemas, ensuring backward compatibility.
    - For large prompt sets, consider using models with async support and running
      probes in parallel for better throughput.
    - Results are stored in memory until explicitly saved; call ``save_results()``
      to persist them to disk.

See Also:
    insideLLMs.runner: The underlying probe execution engine.
    insideLLMs.probes.base: Base classes for creating custom probes.
    insideLLMs.models.base: Model interfaces and protocols.
    insideLLMs.schemas.constants: Schema version constants for validation.
"""

import json
import time
from dataclasses import asdict, is_dataclass
from typing import Any

from insideLLMs.models import Model
from insideLLMs.probes import Probe
from insideLLMs.runner import run_probe
from insideLLMs.schemas.constants import DEFAULT_SCHEMA_VERSION


def _to_serializable(obj: Any) -> Any:
    """Convert an object to a JSON-serializable form.

    This internal helper function ensures that dataclass instances and other
    complex objects can be serialized to JSON for benchmark result storage.
    It recursively converts dataclasses to dictionaries while leaving other
    types unchanged.

    Args:
        obj (Any): The object to convert. Can be any type, but dataclass
            instances receive special handling.

    Returns:
        Any: The JSON-serializable form of the object. For dataclass instances,
            returns a dictionary representation via ``dataclasses.asdict()``.
            For all other types, returns the object unchanged.

    Example: Converting a ModelInfo dataclass
        >>> from insideLLMs.types import ModelInfo
        >>> info = ModelInfo(name="gpt-4", provider="OpenAI", model_id="gpt-4")
        >>> serialized = _to_serializable(info)
        >>> isinstance(serialized, dict)
        True
        >>> serialized["name"]
        'gpt-4'

    Example: Passing through primitive types
        >>> _to_serializable("hello")
        'hello'
        >>> _to_serializable(42)
        42
        >>> _to_serializable([1, 2, 3])
        [1, 2, 3]

    Example: Handling dataclass type vs instance
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class MyData:
        ...     value: int
        >>>
        >>> # Instance gets converted
        >>> _to_serializable(MyData(value=5))
        {'value': 5}
        >>>
        >>> # Type itself is NOT converted (would fail)
        >>> _to_serializable(MyData) is MyData
        True

    Note:
        This function checks ``is_dataclass(obj) and not isinstance(obj, type)``
        to distinguish between dataclass instances (which should be converted)
        and dataclass types themselves (which should not be converted).

    See Also:
        dataclasses.asdict: The standard library function used for conversion.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return obj


class ModelBenchmark:
    """Benchmark multiple models on the same probe and dataset.

    This class provides a standardized way to compare the performance of
    multiple language models on identical evaluation tasks. It runs the same
    probe across all configured models, collecting detailed metrics for each,
    and provides methods for comparing and persisting results.

    Use ModelBenchmark when you want to:
        - Compare different model providers (OpenAI vs Anthropic vs local)
        - Evaluate different model sizes (GPT-4 vs GPT-3.5-turbo)
        - Test model versions after updates or fine-tuning
        - Build leaderboards for specific capabilities

    The benchmark collects the following metrics for each model:
        - **total_time**: Wall-clock time to process all prompts
        - **avg_time_per_item**: Average time per prompt
        - **total_items**: Number of prompts processed
        - **success_rate**: Proportion of successful responses
        - **error_rate**: Proportion of failed responses
        - **mean_latency_ms**: Average response latency in milliseconds

    Parameters:
        models (list[Model]): List of model instances to benchmark. Each model
            must implement the Model protocol with a ``generate()`` method.
            Order determines the sequence of benchmark execution.
        probe (Probe): The probe to use for evaluation. All models are tested
            with this same probe, ensuring consistent evaluation criteria.
        name (str): Human-readable name for this benchmark run. Used in
            result output and saved files. Defaults to "Model Benchmark".

    Attributes:
        models (list[Model]): The list of models being benchmarked.
        probe (Probe): The probe used for evaluation.
        name (str): The benchmark name/identifier.
        results (dict[str, Any]): Dictionary containing benchmark results after
            ``run()`` is called. Empty until the benchmark is executed.

    Example: Basic model comparison
        >>> from insideLLMs.benchmark import ModelBenchmark
        >>> from insideLLMs.models import OpenAIModel
        >>> from insideLLMs.probes.logic import SyllogismProbe
        >>>
        >>> # Setup models to compare
        >>> models = [
        ...     OpenAIModel(model_name="gpt-4"),
        ...     OpenAIModel(model_name="gpt-3.5-turbo"),
        ... ]
        >>>
        >>> # Create benchmark
        >>> probe = SyllogismProbe(name="logic_test")
        >>> benchmark = ModelBenchmark(models, probe, name="GPT Comparison")
        >>>
        >>> # Run benchmark
        >>> prompts = ["Is this valid: All A are B. C is A. Therefore C is B."]
        >>> results = benchmark.run(prompts)
        Benchmarking gpt-4...
        Benchmarking gpt-3.5-turbo...
        >>>
        >>> # Check results
        >>> print(f"Benchmark: {results['name']}")
        Benchmark: GPT Comparison

    Example: Comparing models from different providers
        >>> from insideLLMs.models import OpenAIModel, AnthropicModel, CohereModel
        >>>
        >>> models = [
        ...     OpenAIModel(model_name="gpt-4"),
        ...     AnthropicModel(model_name="claude-3-opus"),
        ...     CohereModel(model_name="command-r-plus"),
        ... ]
        >>>
        >>> benchmark = ModelBenchmark(
        ...     models=models,
        ...     probe=factuality_probe,
        ...     name="Multi-Provider Factuality Test"
        ... )
        >>>
        >>> results = benchmark.run(factual_questions)
        >>>
        >>> # Get rankings by speed
        >>> comparison = benchmark.compare_models()
        >>> print("Speed ranking:", comparison["rankings"]["total_time"])
        Speed ranking: ['gpt-4', 'claude-3-opus', 'command-r-plus']

    Example: Full workflow with analysis
        >>> # Create and run benchmark
        >>> benchmark = ModelBenchmark(models, probe, name="Quality vs Speed")
        >>> results = benchmark.run(test_prompts)
        >>>
        >>> # Analyze each model
        >>> for model_result in results["models"]:
        ...     name = model_result["model"]["name"]
        ...     metrics = model_result["metrics"]
        ...     print(f"{name}:")
        ...     print(f"  Success: {metrics['success_rate']:.1%}")
        ...     print(f"  Latency: {metrics['mean_latency_ms']:.0f}ms")
        gpt-4:
          Success: 98.5%
          Latency: 245ms
        gpt-3.5-turbo:
          Success: 89.2%
          Latency: 156ms
        >>>
        >>> # Compare and rank
        >>> comparison = benchmark.compare_models()
        >>> print(f"Most accurate: {comparison['rankings']['success_rate'][0]}")
        Most accurate: gpt-4
        >>>
        >>> # Persist results
        >>> benchmark.save_results("benchmark_results.json")

    Example: Accessing raw per-prompt results
        >>> results = benchmark.run(prompts)
        >>>
        >>> # Get individual responses for first model
        >>> first_model_results = results["models"][0]["results"]
        >>> for i, result in enumerate(first_model_results[:3]):
        ...     status = result.get("status", "unknown")
        ...     latency = result.get("latency_ms", 0)
        ...     print(f"Prompt {i}: status={status}, latency={latency:.1f}ms")
        Prompt 0: status=success, latency=234.5ms
        Prompt 1: status=success, latency=189.2ms
        Prompt 2: status=error, latency=0.0ms

    Example: Using with custom probe kwargs
        >>> benchmark = ModelBenchmark(models, probe)
        >>>
        >>> # Pass additional arguments to the probe
        >>> results = benchmark.run(
        ...     prompts,
        ...     temperature=0.0,  # Deterministic for comparison
        ...     max_tokens=500,
        ... )

    Note:
        Models are benchmarked sequentially in the order provided. For parallel
        execution across models, consider using multiple threads or processes
        with separate ModelBenchmark instances.

    Warning:
        The ``results`` attribute is overwritten each time ``run()`` is called.
        Call ``save_results()`` before running again if you need to preserve
        previous results.

    See Also:
        ProbeBenchmark: For comparing multiple probes on a single model.
        insideLLMs.runner.run_probe: The underlying probe execution function.
    """

    def __init__(self, models: list[Model], probe: Probe, name: str = "Model Benchmark"):
        """Initialize a model benchmark.

        Creates a new benchmark instance configured to compare the specified
        models using the given probe. The benchmark is ready to run after
        initialization.

        Args:
            models (list[Model]): List of model instances to benchmark.
                Must contain at least one model. Each model should implement
                the Model protocol (have ``generate()`` and ``info()`` methods).
            probe (Probe): The probe to use for evaluation. This probe will
                be run against each model in sequence with the same inputs.
            name (str): Human-readable name for this benchmark run. Appears
                in the output results and saved files. Useful for identifying
                different benchmark runs. Defaults to "Model Benchmark".

        Example: Basic initialization
            >>> from insideLLMs.benchmark import ModelBenchmark
            >>> from insideLLMs.models import OpenAIModel
            >>> from insideLLMs.probes.logic import SyllogismProbe
            >>>
            >>> models = [OpenAIModel(model_name="gpt-4")]
            >>> probe = SyllogismProbe(name="logic")
            >>> benchmark = ModelBenchmark(models, probe)
            >>> benchmark.name
            'Model Benchmark'

        Example: With custom name
            >>> benchmark = ModelBenchmark(
            ...     models=[model1, model2],
            ...     probe=my_probe,
            ...     name="Production Model Comparison Q1 2024"
            ... )
            >>> benchmark.name
            'Production Model Comparison Q1 2024'

        Example: Verifying initialization state
            >>> benchmark = ModelBenchmark(models, probe, name="Test")
            >>> len(benchmark.models)
            2
            >>> benchmark.probe.name
            'logic'
            >>> benchmark.results  # Empty until run() is called
            {}

        Note:
            The ``results`` attribute is initialized as an empty dict and
            populated when ``run()`` is called.
        """
        self.models = models
        self.probe = probe
        self.name = name
        self.results = {}

    def run(
        self,
        prompt_set: list[Any],
        *,
        schema_version: str = DEFAULT_SCHEMA_VERSION,
        **probe_kwargs,
    ) -> dict[str, Any]:
        """Run the benchmark on all configured models.

        Executes the probe against each model in sequence using the provided
        prompt set, collecting performance metrics and individual results.
        Progress is printed to stdout as each model is benchmarked.

        The method:
            1. Initializes the result structure with metadata
            2. Iterates through each model
            3. Runs the probe with the prompt set for each model
            4. Calculates metrics (success rate, latency, timing)
            5. Stores results and updates the instance ``results`` attribute

        Args:
            prompt_set (list[Any]): The list of prompts/inputs to evaluate.
                Each item is passed to the probe's run method. The format
                depends on the probe being used (strings for simple probes,
                dicts for complex probes with structured input).
            schema_version (str): Schema version to include in results for
                validation compatibility. Defaults to DEFAULT_SCHEMA_VERSION
                from insideLLMs.schemas.constants. Use this to lock results
                to a specific schema version.
            **probe_kwargs: Additional keyword arguments passed to the probe
                during execution. Common arguments include:
                - temperature (float): Sampling temperature for the model
                - max_tokens (int): Maximum response length
                - timeout (float): Request timeout in seconds

        Returns:
            dict[str, Any]: A dictionary containing the complete benchmark results
                with the following structure::

                    {
                        "schema_version": "1.0.1",
                        "name": "My Benchmark",
                        "probe": "probe_name",
                        "models": [
                            {
                                "model": {"name": "gpt-4", "provider": "OpenAI", ...},
                                "results": [...],  # Per-prompt results
                                "metrics": {
                                    "total_time": 15.23,
                                    "avg_time_per_item": 0.305,
                                    "total_items": 50,
                                    "success_rate": 0.98,
                                    "error_rate": 0.02,
                                    "mean_latency_ms": 287.5
                                }
                            },
                            ...
                        ],
                        "timestamp": 1704067200.0
                    }

        Example: Basic benchmark run
            >>> benchmark = ModelBenchmark(models, probe)
            >>> prompts = [
            ...     "What is the capital of France?",
            ...     "What is 2 + 2?",
            ...     "Who wrote Romeo and Juliet?",
            ... ]
            >>> results = benchmark.run(prompts)
            Benchmarking gpt-4...
            Benchmarking gpt-3.5-turbo...
            >>> results["name"]
            'Model Benchmark'
            >>> len(results["models"])
            2

        Example: With custom schema version
            >>> results = benchmark.run(
            ...     prompts,
            ...     schema_version="1.0.0"
            ... )
            >>> results["schema_version"]
            '1.0.0'

        Example: Passing probe parameters
            >>> results = benchmark.run(
            ...     prompts,
            ...     temperature=0.0,  # Deterministic responses
            ...     max_tokens=100,   # Limit response length
            ... )

        Example: Processing results after run
            >>> results = benchmark.run(prompts)
            >>>
            >>> # Check metrics for each model
            >>> for model_data in results["models"]:
            ...     model_name = model_data["model"]["name"]
            ...     success_rate = model_data["metrics"]["success_rate"]
            ...     print(f"{model_name}: {success_rate:.1%} success")
            gpt-4: 98.0% success
            gpt-3.5-turbo: 92.0% success

        Example: Accessing individual prompt results
            >>> results = benchmark.run(prompts)
            >>>
            >>> # Get first model's per-prompt results
            >>> first_model = results["models"][0]
            >>> for i, result in enumerate(first_model["results"]):
            ...     print(f"Prompt {i}: {result.get('status', 'unknown')}")
            Prompt 0: success
            Prompt 1: success
            Prompt 2: error

        Example: Analyzing timing data
            >>> results = benchmark.run(prompts)
            >>>
            >>> for model_data in results["models"]:
            ...     metrics = model_data["metrics"]
            ...     print(f"{model_data['model']['name']}:")
            ...     print(f"  Total: {metrics['total_time']:.2f}s")
            ...     print(f"  Per item: {metrics['avg_time_per_item']*1000:.0f}ms")
            ...     print(f"  API latency: {metrics['mean_latency_ms']:.0f}ms")

        Note:
            This method prints progress to stdout. For quiet operation, redirect
            stdout or modify the print statements.

        Warning:
            Running a benchmark with many prompts or slow models can take
            significant time. Consider testing with a small prompt set first.

        See Also:
            compare_models: Analyze and rank models after running.
            save_results: Persist results to a JSON file.
        """
        benchmark_results = {
            "schema_version": schema_version,
            "name": self.name,
            "probe": self.probe.name,
            "models": [],
            "timestamp": time.time(),
        }

        for model in self.models:
            model_info = model.info()
            print(f"Benchmarking {model_info.name}...")

            start_time = time.time()
            results = run_probe(
                model,
                self.probe,
                prompt_set,
                schema_version=schema_version,
                **probe_kwargs,
            )
            end_time = time.time()

            # Calculate metrics from the results
            success_count = sum(1 for r in results if r.get("status") == "success")
            error_count = len(results) - success_count
            success_rate = success_count / len(results) if results else 0.0
            error_rate = error_count / len(results) if results else 0.0
            latencies = [r.get("latency_ms", 0) for r in results]
            mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

            model_result = {
                "model": _to_serializable(model_info),
                "results": results,
                "metrics": {
                    "total_time": end_time - start_time,
                    "avg_time_per_item": (end_time - start_time) / len(prompt_set)
                    if prompt_set
                    else 0,
                    "total_items": len(prompt_set),
                    "success_rate": success_rate,
                    "error_rate": error_rate,
                    "mean_latency_ms": mean_latency,
                },
            }

            benchmark_results["models"].append(model_result)

        self.results = benchmark_results
        return benchmark_results

    def save_results(self, path: str) -> None:
        """Save benchmark results to a JSON file.

        Persists the current benchmark results to the specified file path
        in JSON format with readable indentation. The saved file contains
        all benchmark metadata, per-model results, and metrics.

        This method should be called after ``run()`` has been executed.
        If called before running the benchmark, an empty or incomplete
        results dictionary will be saved.

        Args:
            path (str): The file path where results should be saved.
                Can be absolute or relative to the current working directory.
                The file will be created if it doesn't exist, or overwritten
                if it does.

        Returns:
            None

        Raises:
            OSError: If the file cannot be written (permissions, disk full, etc.).
            TypeError: If results contain non-JSON-serializable objects that
                weren't properly converted by ``_to_serializable()``.

        Example: Basic save operation
            >>> benchmark = ModelBenchmark(models, probe)
            >>> results = benchmark.run(prompts)
            >>> benchmark.save_results("results.json")
            >>>
            >>> # Verify file was created
            >>> import os
            >>> os.path.exists("results.json")
            True

        Example: Save with descriptive filename
            >>> from datetime import datetime
            >>> timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            >>> filename = f"benchmark_{benchmark.name}_{timestamp}.json"
            >>> benchmark.save_results(filename)

        Example: Save to specific directory
            >>> import os
            >>> output_dir = "benchmark_results"
            >>> os.makedirs(output_dir, exist_ok=True)
            >>> benchmark.save_results(os.path.join(output_dir, "latest.json"))

        Example: Load and analyze saved results
            >>> benchmark.save_results("my_benchmark.json")
            >>>
            >>> # Later, load and analyze
            >>> import json
            >>> with open("my_benchmark.json") as f:
            ...     saved = json.load(f)
            >>> print(f"Benchmark: {saved['name']}")
            >>> print(f"Models tested: {len(saved['models'])}")
            >>> for m in saved["models"]:
            ...     print(f"  - {m['model']['name']}: {m['metrics']['success_rate']:.1%}")

        Example: Organizing benchmark history
            >>> # Save each run with timestamp for history
            >>> import time
            >>> results = benchmark.run(prompts)
            >>> timestamp = int(time.time())
            >>> benchmark.save_results(f"benchmarks/run_{timestamp}.json")
            >>>
            >>> # Also save as "latest" for easy access
            >>> benchmark.save_results("benchmarks/latest.json")

        Note:
            The JSON file is formatted with 2-space indentation for readability.
            For compact output, you could modify the method or save manually
            using ``json.dump(benchmark.results, f)``.

        Warning:
            This method overwrites existing files without confirmation.
            Use unique filenames or check for existing files if preservation
            is important.

        See Also:
            run: Execute the benchmark to populate results before saving.
        """
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)

    def compare_models(self) -> dict[str, Any]:
        """Compare models based on benchmark results and generate rankings.

        Analyzes the benchmark results to extract comparative metrics and
        generate rankings across different performance dimensions. This method
        provides a summary view of how models performed relative to each other.

        The comparison includes:
            - **total_time**: Ranks models by total wall-clock execution time
              (lower is better, fastest model first)
            - **avg_time_per_item**: Ranks models by average time per prompt
              (lower is better)
            - **success_rate**: Ranks models by proportion of successful responses
              (higher is better, most successful model first)

        This method must be called after ``run()`` has executed successfully.

        Returns:
            dict[str, Any]: A dictionary containing comparison data with the
                following structure::

                    {
                        "schema_version": "1.0.1",
                        "name": "Benchmark Name",
                        "metrics": {
                            "total_time": {"gpt-4": 15.2, "gpt-3.5": 8.5},
                            "avg_time_per_item": {"gpt-4": 0.30, "gpt-3.5": 0.17},
                            "success_rate": {"gpt-4": 0.98, "gpt-3.5": 0.92}
                        },
                        "rankings": {
                            "total_time": ["gpt-3.5", "gpt-4"],  # Fastest first
                            "avg_time_per_item": ["gpt-3.5", "gpt-4"],
                            "success_rate": ["gpt-4", "gpt-3.5"]  # Best first
                        }
                    }

        Raises:
            ValueError: If ``run()`` has not been called yet, or if the results
                are empty or malformed (missing "models" key).

        Example: Basic comparison after benchmark
            >>> benchmark = ModelBenchmark(models, probe)
            >>> benchmark.run(prompts)
            >>> comparison = benchmark.compare_models()
            >>>
            >>> # Check rankings
            >>> print(f"Fastest: {comparison['rankings']['total_time'][0]}")
            Fastest: gpt-3.5-turbo
            >>> print(f"Most accurate: {comparison['rankings']['success_rate'][0]}")
            Most accurate: gpt-4

        Example: Accessing raw metrics for analysis
            >>> comparison = benchmark.compare_models()
            >>>
            >>> # Get all model times
            >>> times = comparison["metrics"]["total_time"]
            >>> for model, time_sec in times.items():
            ...     print(f"{model}: {time_sec:.2f}s")
            gpt-4: 15.23s
            gpt-3.5-turbo: 8.45s
            >>>
            >>> # Calculate speedup ratio
            >>> speedup = times["gpt-4"] / times["gpt-3.5-turbo"]
            >>> print(f"gpt-4 is {speedup:.1f}x slower")
            gpt-4 is 1.8x slower

        Example: Finding the best model by criteria
            >>> comparison = benchmark.compare_models()
            >>>
            >>> # Best overall (fastest AND accurate)
            >>> speed_rank = comparison["rankings"]["total_time"]
            >>> accuracy_rank = comparison["rankings"]["success_rate"]
            >>>
            >>> # Simple combined ranking (average position)
            >>> models = comparison["metrics"]["total_time"].keys()
            >>> combined = {}
            >>> for m in models:
            ...     speed_pos = speed_rank.index(m)
            ...     accuracy_pos = accuracy_rank.index(m)
            ...     combined[m] = (speed_pos + accuracy_pos) / 2
            >>> best = min(combined, key=combined.get)
            >>> print(f"Best overall: {best}")

        Example: Generating a comparison report
            >>> comparison = benchmark.compare_models()
            >>>
            >>> print(f"Benchmark: {comparison['name']}")
            >>> print(f"Schema: {comparison['schema_version']}")
            >>> print()
            >>> print("Rankings:")
            >>> print(f"  Speed:    {' > '.join(comparison['rankings']['total_time'])}")
            >>> print(f"  Accuracy: {' > '.join(comparison['rankings']['success_rate'])}")
            Benchmark: Model Comparison
            Schema: 1.0.1

            Rankings:
              Speed:    gpt-3.5-turbo > gpt-4
              Accuracy: gpt-4 > gpt-3.5-turbo

        Example: Error handling for premature call
            >>> benchmark = ModelBenchmark(models, probe)
            >>> # Forgot to call run()!
            >>> try:
            ...     comparison = benchmark.compare_models()
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: No benchmark results available. Run the benchmark first.

        Note:
            Rankings are sorted lists of model names. For speed metrics,
            models are sorted ascending (faster is better). For success
            rate, models are sorted descending (higher is better).

        See Also:
            run: Execute the benchmark to generate results before comparing.
        """
        if not self.results or "models" not in self.results:
            raise ValueError("No benchmark results available. Run the benchmark first.")

        comparison = {
            "schema_version": self.results.get("schema_version", DEFAULT_SCHEMA_VERSION),
            "name": self.name,
            "metrics": {},
            "rankings": {},
        }

        # Extract metrics for comparison
        model_metrics = {}
        for model_result in self.results["models"]:
            model_name = model_result["model"]["name"]
            model_metrics[model_name] = model_result["metrics"]

        # Compare total time
        total_times = {name: metrics["total_time"] for name, metrics in model_metrics.items()}
        comparison["metrics"]["total_time"] = total_times
        comparison["rankings"]["total_time"] = sorted(
            total_times.keys(), key=lambda x: total_times[x]
        )

        # Compare average time per item
        avg_times = {name: metrics["avg_time_per_item"] for name, metrics in model_metrics.items()}
        comparison["metrics"]["avg_time_per_item"] = avg_times
        comparison["rankings"]["avg_time_per_item"] = sorted(
            avg_times.keys(), key=lambda x: avg_times[x]
        )

        # Compare success rate
        success_rates = {name: metrics["success_rate"] for name, metrics in model_metrics.items()}
        comparison["metrics"]["success_rate"] = success_rates
        comparison["rankings"]["success_rate"] = sorted(
            success_rates.keys(), key=lambda x: success_rates[x], reverse=True
        )

        return comparison


class ProbeBenchmark:
    """Benchmark multiple probes on the same model and dataset.

    This class provides a standardized way to evaluate a single language model
    across multiple different probes (evaluation tasks). It runs all configured
    probes against the same model with the same inputs, enabling assessment of
    the model's strengths and weaknesses across different capabilities.

    Use ProbeBenchmark when you want to:
        - Create a comprehensive model capability profile
        - Identify strengths and weaknesses of a specific model
        - Validate model performance across different task types
        - Build model evaluation reports or scorecards

    The benchmark collects the following metrics for each probe:
        - **total_time**: Wall-clock time to complete all prompts with this probe
        - **avg_time_per_item**: Average time per prompt
        - **total_items**: Number of prompts processed
        - **success_rate**: Proportion of responses without errors

    Parameters:
        model (Model): The model instance to benchmark. Must implement the
            Model protocol with a ``generate()`` method. This same model
            is used for all probe evaluations.
        probes (list[Probe]): List of probe instances to run. Each probe
            tests a different aspect of model capability. Order determines
            the sequence of benchmark execution.
        name (str): Human-readable name for this benchmark run. Used in
            result output and saved files. Defaults to "Probe Benchmark".

    Attributes:
        model (Model): The model being evaluated.
        probes (list[Probe]): The list of probes used for evaluation.
        name (str): The benchmark name/identifier.
        results (dict[str, Any]): Dictionary containing benchmark results after
            ``run()`` is called. Empty until the benchmark is executed.

    Example: Basic probe benchmark
        >>> from insideLLMs.benchmark import ProbeBenchmark
        >>> from insideLLMs.models import OpenAIModel
        >>> from insideLLMs.probes.logic import SyllogismProbe
        >>> from insideLLMs.probes.factuality import FactualityProbe
        >>>
        >>> # Setup probes to evaluate
        >>> probes = [
        ...     SyllogismProbe(name="logic"),
        ...     FactualityProbe(name="facts"),
        ... ]
        >>>
        >>> # Create benchmark for a single model
        >>> model = OpenAIModel(model_name="gpt-4")
        >>> benchmark = ProbeBenchmark(model, probes, name="GPT-4 Capabilities")
        >>>
        >>> # Run benchmark
        >>> results = benchmark.run(test_prompts)
        Benchmarking logic...
        Benchmarking facts...
        >>>
        >>> # Check results
        >>> print(f"Benchmark: {results['name']}")
        Benchmark: GPT-4 Capabilities

    Example: Comprehensive model evaluation
        >>> from insideLLMs.probes.logic import SyllogismProbe
        >>> from insideLLMs.probes.bias import GenderBiasProbe
        >>> from insideLLMs.probes.factuality import FactualityProbe
        >>> from insideLLMs.probes.code import CodeGenerationProbe
        >>>
        >>> # Test multiple capabilities
        >>> probes = [
        ...     SyllogismProbe(name="logical_reasoning"),
        ...     GenderBiasProbe(name="bias_detection"),
        ...     FactualityProbe(name="factual_accuracy"),
        ...     CodeGenerationProbe(name="code_generation"),
        ... ]
        >>>
        >>> benchmark = ProbeBenchmark(
        ...     model=model,
        ...     probes=probes,
        ...     name="Comprehensive Model Evaluation"
        ... )
        >>>
        >>> results = benchmark.run(evaluation_prompts)
        >>>
        >>> # Generate capability report
        >>> for probe_result in results["probes"]:
        ...     name = probe_result["probe"]
        ...     success = probe_result["metrics"]["success_rate"]
        ...     print(f"{name}: {success:.1%}")
        logical_reasoning: 96.5%
        bias_detection: 88.2%
        factual_accuracy: 91.3%
        code_generation: 94.7%

    Example: Comparing performance across probe types
        >>> results = benchmark.run(prompts)
        >>>
        >>> # Find best and worst performing areas
        >>> probe_scores = [
        ...     (p["probe"], p["metrics"]["success_rate"])
        ...     for p in results["probes"]
        ... ]
        >>> probe_scores.sort(key=lambda x: x[1], reverse=True)
        >>>
        >>> print(f"Strongest: {probe_scores[0][0]} ({probe_scores[0][1]:.1%})")
        >>> print(f"Weakest: {probe_scores[-1][0]} ({probe_scores[-1][1]:.1%})")
        Strongest: logical_reasoning (96.5%)
        Weakest: bias_detection (88.2%)

    Example: Saving and loading results
        >>> benchmark = ProbeBenchmark(model, probes)
        >>> results = benchmark.run(prompts)
        >>> benchmark.save_results("model_evaluation.json")
        >>>
        >>> # Later: load and analyze
        >>> import json
        >>> with open("model_evaluation.json") as f:
        ...     saved = json.load(f)
        >>> print(f"Model: {saved['model']['name']}")
        >>> print(f"Probes tested: {len(saved['probes'])}")

    Example: Using with custom probe kwargs
        >>> benchmark = ProbeBenchmark(model, probes)
        >>>
        >>> # Pass parameters to all probes
        >>> results = benchmark.run(
        ...     prompts,
        ...     temperature=0.0,  # Deterministic
        ...     max_tokens=200,
        ... )

    Note:
        Probes are executed sequentially in the order provided. Each probe
        runs all prompts before the next probe starts.

    Warning:
        The ``results`` attribute is overwritten each time ``run()`` is called.
        Call ``save_results()`` before running again if you need to preserve
        previous results.

    See Also:
        ModelBenchmark: For comparing multiple models on a single probe.
        insideLLMs.runner.run_probe: The underlying probe execution function.
    """

    def __init__(self, model: Model, probes: list[Probe], name: str = "Probe Benchmark"):
        """Initialize a probe benchmark.

        Creates a new benchmark instance configured to evaluate the specified
        model across all given probes. The benchmark is ready to run after
        initialization.

        Args:
            model (Model): The model instance to benchmark. Must implement
                the Model protocol (have ``generate()`` and ``info()`` methods).
                This single model is tested against all probes.
            probes (list[Probe]): List of probe instances to run against the
                model. Each probe evaluates a different capability or aspect
                of model behavior. Must contain at least one probe.
            name (str): Human-readable name for this benchmark run. Appears
                in the output results and saved files. Useful for identifying
                different benchmark runs. Defaults to "Probe Benchmark".

        Example: Basic initialization
            >>> from insideLLMs.benchmark import ProbeBenchmark
            >>> from insideLLMs.models import OpenAIModel
            >>> from insideLLMs.probes.logic import SyllogismProbe
            >>>
            >>> model = OpenAIModel(model_name="gpt-4")
            >>> probes = [SyllogismProbe(name="logic")]
            >>> benchmark = ProbeBenchmark(model, probes)
            >>> benchmark.name
            'Probe Benchmark'

        Example: With custom name
            >>> benchmark = ProbeBenchmark(
            ...     model=my_model,
            ...     probes=[probe1, probe2],
            ...     name="Model Capability Assessment Q1 2024"
            ... )
            >>> benchmark.name
            'Model Capability Assessment Q1 2024'

        Example: Verifying initialization state
            >>> benchmark = ProbeBenchmark(model, probes, name="Test")
            >>> benchmark.model.name
            'gpt-4'
            >>> len(benchmark.probes)
            2
            >>> benchmark.results  # Empty until run() is called
            {}

        Note:
            The ``results`` attribute is initialized as an empty dict and
            populated when ``run()`` is called.
        """
        self.model = model
        self.probes = probes
        self.name = name
        self.results = {}

    def run(
        self,
        prompt_set: list[Any],
        *,
        schema_version: str = DEFAULT_SCHEMA_VERSION,
        **kwargs,
    ) -> dict[str, Any]:
        """Run the benchmark on all configured probes.

        Executes each probe against the model in sequence using the provided
        prompt set, collecting performance metrics and individual results.
        Progress is printed to stdout as each probe is benchmarked.

        The method:
            1. Initializes the result structure with model metadata
            2. Iterates through each probe
            3. Runs the probe with the prompt set against the model
            4. Calculates metrics (success rate, timing)
            5. Stores results and updates the instance ``results`` attribute

        Args:
            prompt_set (list[Any]): The list of prompts/inputs to evaluate.
                Each item is passed to every probe's run method. The format
                should be compatible with all probes in the benchmark
                (strings for simple probes, dicts for complex probes).
            schema_version (str): Schema version to include in results for
                validation compatibility. Defaults to DEFAULT_SCHEMA_VERSION
                from insideLLMs.schemas.constants. Use this to lock results
                to a specific schema version.
            **kwargs: Additional keyword arguments passed to all probes
                during execution. Common arguments include:
                - temperature (float): Sampling temperature for the model
                - max_tokens (int): Maximum response length
                - timeout (float): Request timeout in seconds

        Returns:
            dict[str, Any]: A dictionary containing the complete benchmark results
                with the following structure::

                    {
                        "schema_version": "1.0.1",
                        "name": "My Benchmark",
                        "model": {"name": "gpt-4", "provider": "OpenAI", ...},
                        "probes": [
                            {
                                "probe": "logic_probe",
                                "results": [...],  # Per-prompt results
                                "metrics": {
                                    "total_time": 15.23,
                                    "avg_time_per_item": 0.305,
                                    "total_items": 50,
                                    "success_rate": 0.98
                                }
                            },
                            ...
                        ],
                        "timestamp": 1704067200.0
                    }

        Example: Basic probe benchmark run
            >>> benchmark = ProbeBenchmark(model, probes)
            >>> prompts = [
            ...     "Is this logically valid: All A are B. C is A. Therefore C is B.",
            ...     "What is the capital of France?",
            ...     "Explain recursion.",
            ... ]
            >>> results = benchmark.run(prompts)
            Benchmarking logic...
            Benchmarking factuality...
            >>> results["name"]
            'Probe Benchmark'
            >>> len(results["probes"])
            2

        Example: With custom schema version
            >>> results = benchmark.run(
            ...     prompts,
            ...     schema_version="1.0.0"
            ... )
            >>> results["schema_version"]
            '1.0.0'

        Example: Passing model parameters to all probes
            >>> results = benchmark.run(
            ...     prompts,
            ...     temperature=0.0,  # Deterministic responses
            ...     max_tokens=100,   # Limit response length
            ... )

        Example: Processing results after run
            >>> results = benchmark.run(prompts)
            >>>
            >>> # Check metrics for each probe
            >>> for probe_data in results["probes"]:
            ...     probe_name = probe_data["probe"]
            ...     success_rate = probe_data["metrics"]["success_rate"]
            ...     print(f"{probe_name}: {success_rate:.1%} success")
            logic: 95.0% success
            factuality: 88.0% success

        Example: Accessing individual results per probe
            >>> results = benchmark.run(prompts)
            >>>
            >>> # Get first probe's per-prompt results
            >>> first_probe = results["probes"][0]
            >>> print(f"Probe: {first_probe['probe']}")
            >>> for i, result in enumerate(first_probe["results"][:3]):
            ...     has_error = "error" in result
            ...     print(f"  Prompt {i}: {'error' if has_error else 'success'}")

        Example: Creating a capability heatmap
            >>> results = benchmark.run(prompts)
            >>>
            >>> # Build capability summary
            >>> capabilities = {}
            >>> for probe_data in results["probes"]:
            ...     name = probe_data["probe"]
            ...     metrics = probe_data["metrics"]
            ...     capabilities[name] = {
            ...         "success_rate": metrics["success_rate"],
            ...         "avg_latency_ms": metrics["avg_time_per_item"] * 1000,
            ...     }
            >>> print(capabilities)

        Note:
            This method prints progress to stdout. For quiet operation, redirect
            stdout or modify the print statements.

        Warning:
            Running a benchmark with many prompts or slow probes can take
            significant time. Consider testing with a small prompt set first.

        See Also:
            save_results: Persist results to a JSON file.
        """
        benchmark_results = {
            "schema_version": schema_version,
            "name": self.name,
            "model": _to_serializable(self.model.info()),
            "probes": [],
            "timestamp": time.time(),
        }

        for probe in self.probes:
            print(f"Benchmarking {probe.name}...")

            start_time = time.time()
            results = run_probe(
                self.model,
                probe,
                prompt_set,
                schema_version=schema_version,
                **kwargs,
            )
            end_time = time.time()

            probe_result = {
                "probe": probe.name,
                "results": results,
                "metrics": {
                    "total_time": end_time - start_time,
                    "avg_time_per_item": (end_time - start_time) / len(prompt_set)
                    if prompt_set
                    else 0,
                    "total_items": len(prompt_set),
                    "success_rate": sum(1 for r in results if "error" not in r) / len(results)
                    if results
                    else 0,
                },
            }

            benchmark_results["probes"].append(probe_result)

        self.results = benchmark_results
        return benchmark_results

    def save_results(self, path: str) -> None:
        """Save benchmark results to a JSON file.

        Persists the current benchmark results to the specified file path
        in JSON format with readable indentation. The saved file contains
        all benchmark metadata, model information, per-probe results, and
        metrics.

        This method should be called after ``run()`` has been executed.
        If called before running the benchmark, an empty or incomplete
        results dictionary will be saved.

        Args:
            path (str): The file path where results should be saved.
                Can be absolute or relative to the current working directory.
                The file will be created if it doesn't exist, or overwritten
                if it does.

        Returns:
            None

        Raises:
            OSError: If the file cannot be written (permissions, disk full, etc.).
            TypeError: If results contain non-JSON-serializable objects that
                weren't properly converted by ``_to_serializable()``.

        Example: Basic save operation
            >>> benchmark = ProbeBenchmark(model, probes)
            >>> results = benchmark.run(prompts)
            >>> benchmark.save_results("probe_results.json")
            >>>
            >>> # Verify file was created
            >>> import os
            >>> os.path.exists("probe_results.json")
            True

        Example: Save with descriptive filename
            >>> from datetime import datetime
            >>> model_name = benchmark.model.name.replace("/", "-")
            >>> timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            >>> filename = f"probe_benchmark_{model_name}_{timestamp}.json"
            >>> benchmark.save_results(filename)

        Example: Save to specific directory
            >>> import os
            >>> output_dir = "benchmark_results/probes"
            >>> os.makedirs(output_dir, exist_ok=True)
            >>> benchmark.save_results(os.path.join(output_dir, "latest.json"))

        Example: Load and analyze saved results
            >>> benchmark.save_results("my_evaluation.json")
            >>>
            >>> # Later, load and analyze
            >>> import json
            >>> with open("my_evaluation.json") as f:
            ...     saved = json.load(f)
            >>> print(f"Model: {saved['model']['name']}")
            >>> print(f"Probes tested: {len(saved['probes'])}")
            >>> for p in saved["probes"]:
            ...     print(f"  - {p['probe']}: {p['metrics']['success_rate']:.1%}")

        Example: Archiving evaluation history
            >>> # Save each evaluation with timestamp
            >>> import time
            >>> results = benchmark.run(prompts)
            >>> timestamp = int(time.time())
            >>> benchmark.save_results(f"evaluations/run_{timestamp}.json")
            >>>
            >>> # Also save as "latest" for easy access
            >>> benchmark.save_results("evaluations/latest.json")

        Note:
            The JSON file is formatted with 2-space indentation for readability.
            The file includes the full model metadata, making it self-contained
            for later analysis.

        Warning:
            This method overwrites existing files without confirmation.
            Use unique filenames or check for existing files if preservation
            is important.

        See Also:
            run: Execute the benchmark to populate results before saving.
        """
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
