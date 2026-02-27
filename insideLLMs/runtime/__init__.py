"""Execution runtime utilities for LLM experiments.

This package provides the core runtime infrastructure for executing, monitoring,
and reproducing LLM experiments. It consolidates four major subsystems into a
unified interface for running probes, composing model pipelines, collecting
telemetry, and ensuring experiment reproducibility.

Overview
--------
The runtime package is organized into four specialized modules:

1. **runner** - Experiment execution with YAML/JSON config support
2. **pipeline** - Composable middleware for model capabilities
3. **observability** - Telemetry, tracing, and monitoring
4. **reproducibility** - Seed management and experiment snapshots

All public symbols from these modules are re-exported here for convenient access.

Modules
-------
runner
    Provides :class:`ProbeRunner` and :class:`AsyncProbeRunner` for executing
    probes against models, along with configuration-driven experiment execution
    via :func:`run_experiment_from_config` and :func:`run_harness_from_config`.

pipeline
    Implements the middleware pattern through :class:`ModelPipeline` and
    :class:`AsyncModelPipeline`, with built-in middleware for caching
    (:class:`CacheMiddleware`), rate limiting (:class:`RateLimitMiddleware`),
    retry logic (:class:`RetryMiddleware`), cost tracking
    (:class:`CostTrackingMiddleware`), and execution tracing
    (:class:`TraceMiddleware`).

observability
    Offers comprehensive monitoring via :class:`TelemetryCollector` for
    lightweight call recording, :class:`TracedModel` for automatic
    instrumentation, and :class:`OTelTracedModel` for OpenTelemetry
    distributed tracing integration.

reproducibility
    Ensures experiment repeatability through :class:`SeedManager` for
    cross-library seed management, :class:`ExperimentSnapshot` for
    state capture, and :class:`DeterministicExecutor` for guaranteed
    deterministic execution.

Key Classes
-----------
ProbeRunner
    Synchronous runner for sequential probe execution against a model.

AsyncProbeRunner
    Asynchronous runner for concurrent probe execution with configurable
    parallelism and progress callbacks.

ModelPipeline
    Wraps a model with composable middleware for caching, rate limiting,
    retries, cost tracking, and more.

AsyncModelPipeline
    Async-first pipeline with batch processing and streaming result support.

Middleware
    Abstract base class for implementing custom pipeline middleware.

CacheMiddleware
    Response caching with LRU eviction and optional TTL expiration.

RateLimitMiddleware
    Token bucket rate limiting to prevent API quota exhaustion.

RetryMiddleware
    Exponential backoff retry logic for transient failures.

CostTrackingMiddleware
    Token usage and cost estimation for API calls.

TraceMiddleware
    Execution trace capture for debugging and replay.

TracingConfig
    Configuration dataclass for tracing and observability settings.

TelemetryCollector
    In-memory collector for call records with statistics aggregation.

CallRecord
    Immutable record of a single model invocation with timing and metadata.

TracedModel
    Model wrapper that automatically records all operations to telemetry.

OTelTracedModel
    Model wrapper with OpenTelemetry distributed tracing spans.

SeedManager
    Manages random seeds across Python, NumPy, PyTorch, and TensorFlow.

ExperimentSnapshot
    Complete capture of experiment state for reproducibility.

EnvironmentCapture
    Captures Python version, packages, and environment variables.

ConfigVersionManager
    Tracks and diffs configuration versions over time.

DeterministicExecutor
    Executes functions with guaranteed seed state.

ExperimentRegistry
    Registry for tracking and querying experiments by ID, name, or tags.

Key Functions
-------------
run_probe(model, probe, prompts, ...)
    Simple function to run a probe on a model and return results.

run_experiment_from_config(config_path, ...)
    Execute an experiment defined in a YAML or JSON configuration file.

run_harness_from_config(config_path, ...)
    Run a multi-model, multi-probe evaluation harness from configuration.

run_harness_to_dir(config_path, run_dir, ...)
    Stable workflow helper for running harness configs into explicit run dirs.

diff_run_dirs(run_dir_a, run_dir_b, ...)
    Stable workflow helper for diffing run directories with CI-oriented flags.

load_config(path)
    Load and validate a YAML or JSON experiment configuration file.

create_experiment_result(...)
    Create a structured ExperimentResult from runner execution data.

instrument_model(model, config, collector)
    Wrap a model with automatic tracing instrumentation.

setup_otel_tracing(config)
    Configure OpenTelemetry tracing with Jaeger or OTLP export.

get_collector()
    Get the global TelemetryCollector instance.

set_collector(collector)
    Set the global TelemetryCollector instance.

trace_call(model_name, operation, prompt, ...)
    Context manager for manually tracing a model call.

trace_function(operation_name, include_args)
    Decorator for tracing arbitrary function calls.

estimate_tokens(text, model)
    Estimate token count for text (heuristic-based).

set_global_seed(seed)
    Set random seed across all available libraries (Python, NumPy, etc.).

create_seed_manager(seed)
    Create a SeedManager instance for controlled seed management.

capture_snapshot(name, config, seed)
    Capture a complete ExperimentSnapshot of current state.

save_snapshot(snapshot, path)
    Persist an ExperimentSnapshot to disk.

load_snapshot(path)
    Load an ExperimentSnapshot from disk.

capture_environment()
    Capture current environment information (Python, OS, packages).

compare_environments(env1, env2)
    Compare two EnvironmentInfo snapshots and report differences.

diff_configs(config1, config2)
    Compute differences between two configuration dictionaries.

check_reproducibility(snapshot, seed_manager)
    Analyze and report on reproducibility status.

run_deterministic(func, seed, *args, **kwargs)
    Execute a function with deterministic seed state.

derive_seed(base_seed, key)
    Derive a deterministic seed from a base seed and string key.

derive_run_id_from_config_path(config_path, schema_version)
    Generate a deterministic run ID from a configuration file.

Examples
--------
Basic probe execution with ProbeRunner:

    >>> from insideLLMs.models import OpenAIModel
    >>> from insideLLMs.probes import FactualityProbe
    >>> from insideLLMs.runtime import ProbeRunner
    >>>
    >>> model = OpenAIModel(model_name="gpt-4")
    >>> probe = FactualityProbe()
    >>> runner = ProbeRunner(model, probe)
    >>>
    >>> prompts = [
    ...     {"messages": [{"role": "user", "content": "What is 2+2?"}]},
    ...     {"messages": [{"role": "user", "content": "Who wrote Hamlet?"}]},
    ... ]
    >>> results = runner.run(prompts)
    >>> print(f"Success rate: {runner.success_rate:.1%}")

Async execution with concurrency control:

    >>> import asyncio
    >>> from insideLLMs.runtime import AsyncProbeRunner
    >>>
    >>> async def run_concurrent():
    ...     model = OpenAIModel(model_name="gpt-4")
    ...     probe = FactualityProbe()
    ...     runner = AsyncProbeRunner(model, probe)
    ...     results = await runner.run(prompts, concurrency=10)
    ...     return results
    >>>
    >>> results = asyncio.run(run_concurrent())

Building a model pipeline with middleware:

    >>> from insideLLMs.runtime import (
    ...     ModelPipeline,
    ...     CacheMiddleware,
    ...     RateLimitMiddleware,
    ...     RetryMiddleware,
    ...     CostTrackingMiddleware,
    ... )
    >>>
    >>> pipeline = ModelPipeline(
    ...     OpenAIModel(model_name="gpt-4"),
    ...     middlewares=[
    ...         CacheMiddleware(cache_size=500, ttl_seconds=3600),
    ...         RateLimitMiddleware(requests_per_minute=60),
    ...         RetryMiddleware(max_retries=3),
    ...         CostTrackingMiddleware(),
    ...     ],
    ... )
    >>> response = pipeline.generate("Explain transformers")
    >>> print(pipeline.info())  # Includes middleware stats

Instrumenting a model for telemetry:

    >>> from insideLLMs.runtime import (
    ...     instrument_model,
    ...     get_collector,
    ...     TracingConfig,
    ... )
    >>>
    >>> config = TracingConfig(log_prompts=False, log_responses=False)
    >>> traced = instrument_model(model, config)
    >>> response = traced.generate("Hello, world!")
    >>>
    >>> collector = get_collector()
    >>> stats = collector.get_stats()
    >>> print(f"Total calls: {stats['total_calls']}")
    >>> print(f"Avg latency: {stats['avg_latency_ms']:.2f}ms")

Setting up OpenTelemetry distributed tracing:

    >>> from insideLLMs.runtime import (
    ...     TracingConfig,
    ...     setup_otel_tracing,
    ...     OTelTracedModel,
    ... )
    >>>
    >>> config = TracingConfig(
    ...     service_name="my-llm-app",
    ...     jaeger_endpoint="http://localhost:14268/api/traces",
    ...     sample_rate=0.1,
    ... )
    >>> setup_otel_tracing(config)
    >>> traced = OTelTracedModel(model, config)

Ensuring reproducibility with seed management:

    >>> from insideLLMs.runtime import (
    ...     SeedManager,
    ...     capture_snapshot,
    ...     check_reproducibility,
    ... )
    >>>
    >>> # Set seeds across all libraries
    >>> seed_manager = SeedManager(global_seed=42)
    >>> results = seed_manager.set_all_seeds()
    >>> print(f"Libraries seeded: {seed_manager.get_libraries_seeded()}")
    >>>
    >>> # Capture experiment state
    >>> snapshot = capture_snapshot(
    ...     name="my-experiment",
    ...     config={"model": "gpt-4", "temperature": 0.7},
    ...     seed=42,
    ... )
    >>> snapshot.save("experiment_snapshot.json")
    >>>
    >>> # Check reproducibility status
    >>> report = check_reproducibility(snapshot, seed_manager)
    >>> print(f"Reproducibility level: {report.level.value}")

Running experiments from configuration:

    >>> from insideLLMs.runtime import (
    ...     run_experiment_from_config,
    ...     run_harness_from_config,
    ...     load_config,
    ... )
    >>>
    >>> # Single experiment from config file
    >>> results = run_experiment_from_config("experiment.yaml")
    >>>
    >>> # Multi-model harness
    >>> harness_results = run_harness_from_config("harness.yaml")

Deterministic function execution:

    >>> from insideLLMs.runtime import run_deterministic, derive_seed
    >>>
    >>> def stochastic_function(data):
    ...     import random
    ...     return random.choice(data)
    >>>
    >>> # Always produces the same result
    >>> result, metadata = run_deterministic(
    ...     stochastic_function,
    ...     seed=12345,
    ...     data=["a", "b", "c"],
    ... )
    >>>
    >>> # Derive sub-seeds for different operations
    >>> embedding_seed = derive_seed(base_seed=42, key="embedding")
    >>> sampling_seed = derive_seed(base_seed=42, key="sampling")

Async pipeline with batch processing:

    >>> from insideLLMs.runtime import AsyncModelPipeline
    >>>
    >>> async def batch_process():
    ...     pipeline = AsyncModelPipeline(
    ...         model,
    ...         middlewares=[CacheMiddleware(), RateLimitMiddleware()],
    ...     )
    ...
    ...     # Process many prompts efficiently
    ...     prompts = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    ...     results = await pipeline.abatch_generate(
    ...         prompts,
    ...         max_concurrency=3,
    ...     )
    ...
    ...     # Or stream results as they complete
    ...     async for idx, result in pipeline.agenerate_stream_results(prompts):
    ...         print(f"Prompt {idx}: {result[:50]}...")
    ...
    ...     return results

Notes
-----
- The runner module provides the primary entry points for executing probes
  and experiments, either programmatically or from configuration files.

- Pipeline middleware is executed in order for requests and reverse order
  for responses, following the chain of responsibility pattern.

- The observability module works without OpenTelemetry for simple use cases;
  distributed tracing requires: ``pip install opentelemetry-sdk``

- Token estimation uses a heuristic (~4 characters per token). For precise
  counts, use the model's native tokenizer.

- The global TelemetryCollector has a default limit of 10,000 records with
  automatic LRU pruning.

- Reproducibility requires setting seeds BEFORE any stochastic operations.
  The SeedManager handles Python, NumPy, PyTorch, and TensorFlow.

- Experiment snapshots capture environment info for debugging but may
  contain sensitive data (env vars). Use capture_env_vars=False for
  production.

See Also
--------
insideLLMs.models : Model implementations (OpenAI, Anthropic, etc.)
insideLLMs.probes : Probe definitions for evaluating model behavior
insideLLMs.types : Type definitions (ExperimentResult, ProbeResult, etc.)
insideLLMs.config_types : Configuration dataclasses (RunConfig, etc.)
insideLLMs.tracing : Low-level trace recording utilities
"""

from insideLLMs.runtime.async_io import async_write_lines, async_write_text  # noqa: F401
from insideLLMs.runtime.diffing import *  # noqa: F401,F403
from insideLLMs.runtime.observability import *  # noqa: F401,F403
from insideLLMs.runtime.pipeline import *  # noqa: F401,F403
from insideLLMs.runtime.reproducibility import *  # noqa: F401,F403
from insideLLMs.runtime.runner import *  # noqa: F401,F403
from insideLLMs.runtime.timeout_wrapper import run_with_timeout  # noqa: F401
