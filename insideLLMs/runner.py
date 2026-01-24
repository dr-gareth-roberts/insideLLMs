"""Compatibility shim for insideLLMs.runtime.runner.

This module provides backward-compatible imports for the experiment runner
infrastructure. All functionality has been consolidated into the
``insideLLMs.runtime.runner`` module, and this shim re-exports all symbols
to preserve legacy import paths.

Overview
--------
The runner module provides tools for executing LLM probes/experiments:

- **ProbeRunner**: Synchronous runner for sequential probe execution
- **AsyncProbeRunner**: Asynchronous runner for concurrent probe execution
- **run_probe**: Convenience function for simple probe execution
- **run_probe_async**: Async version of run_probe
- **run_experiment_from_config**: Run experiments from YAML/JSON config files
- **run_experiment_from_config_async**: Async version
- **run_harness_from_config**: Run multi-model/multi-probe comparison harness
- **load_config**: Load configuration files without running

Migration Guide
---------------
New code should import directly from ``insideLLMs.runtime.runner``::

    # Recommended (new style)
    from insideLLMs.runtime.runner import ProbeRunner, AsyncProbeRunner

    # Deprecated (legacy style, still works via this shim)
    from insideLLMs.runner import ProbeRunner, AsyncProbeRunner

Both import styles work identically; this shim ensures backward compatibility.

Examples
--------
Basic probe execution using the synchronous runner:

    >>> from insideLLMs.runner import ProbeRunner
    >>> from insideLLMs.models import OpenAIModel
    >>> from insideLLMs.probes import FactualityProbe
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
    Success rate: 100.0%

Using the convenience function for one-off execution:

    >>> from insideLLMs.runner import run_probe
    >>>
    >>> results = run_probe(model, probe, prompts)
    >>> print(f"Processed {len(results)} prompts")
    Processed 2 prompts

Async execution with concurrency for faster API calls:

    >>> import asyncio
    >>> from insideLLMs.runner import AsyncProbeRunner
    >>>
    >>> async def run_concurrent():
    ...     runner = AsyncProbeRunner(model, probe)
    ...     results = await runner.run(prompts, concurrency=10)
    ...     return results
    >>>
    >>> results = asyncio.run(run_concurrent())

Running experiments from YAML configuration:

    >>> from insideLLMs.runner import run_experiment_from_config
    >>>
    >>> # experiment.yaml defines model, probe, and dataset
    >>> results = run_experiment_from_config("experiment.yaml")
    >>> print(f"Completed {len(results)} examples")

Getting structured ExperimentResult with metrics:

    >>> from insideLLMs.runner import run_experiment_from_config
    >>>
    >>> experiment = run_experiment_from_config(
    ...     "experiment.yaml",
    ...     return_experiment=True
    ... )
    >>> print(f"Experiment ID: {experiment.experiment_id}")
    >>> print(f"Score: {experiment.score}")

Running a multi-model, multi-probe harness:

    >>> from insideLLMs.runner import run_harness_from_config
    >>>
    >>> # harness.yaml defines multiple models and probes
    >>> results = run_harness_from_config("harness.yaml")
    >>> print(f"Total records: {len(results['records'])}")
    >>> print(f"Experiments: {len(results['experiments'])}")
    >>>
    >>> # Access the summary report
    >>> for model_name, scores in results['summary'].items():
    ...     print(f"{model_name}: {scores}")

Loading configuration without running:

    >>> from insideLLMs.runner import load_config
    >>>
    >>> config = load_config("experiment.yaml")
    >>> print(f"Model type: {config['model']['type']}")
    >>> print(f"Probe type: {config['probe']['type']}")

Re-exported Symbols
-------------------
Classes
~~~~~~~
ProbeRunner
    Synchronous runner for executing probes sequentially. Use when order
    matters or API rate limits are a concern.

AsyncProbeRunner
    Asynchronous runner for concurrent probe execution. Significantly faster
    for API-based models that can handle parallel requests.

_RunnerBase
    Base class for runners (internal use). Provides shared properties like
    ``success_rate``, ``error_count``, ``last_run_id``, and ``last_experiment``.

Functions
~~~~~~~~~
run_probe(model, probe, prompt_set, *, config=None, **probe_kwargs)
    Convenience function for simple synchronous probe execution.

run_probe_async(model, probe, prompt_set, *, config=None, concurrency=None, **probe_kwargs)
    Convenience function for async probe execution with concurrency.

run_experiment_from_config(config_path, **kwargs)
    Run an experiment defined in a YAML or JSON configuration file.

run_experiment_from_config_async(config_path, *, concurrency=5, **kwargs)
    Async version of run_experiment_from_config.

run_harness_from_config(config_path, **kwargs)
    Run a cross-model, cross-probe harness from configuration.

load_config(path)
    Load an experiment configuration file (YAML or JSON).

create_experiment_result(model, probe, results, config=None, **kwargs)
    Create a structured ExperimentResult from raw results.

derive_run_id_from_config_path(config_path, *, schema_version=...)
    Compute the deterministic run ID for a configuration file.

Type Aliases
~~~~~~~~~~~~
ProgressCallback
    Union type for progress callback signatures. Supports both legacy
    ``(current, total)`` and rich ``ProgressInfo`` callbacks.

Notes
-----
This module is a thin compatibility shim. All implementation resides in
``insideLLMs.runtime.runner``. The shim uses dynamic attribute copying
to ensure all symbols (including any future additions) are automatically
re-exported.

The runner infrastructure supports:

- **Deterministic run IDs**: Generated from input configuration for
  reproducibility and caching.
- **Resumable runs**: Interrupted experiments can be resumed from where
  they left off using the ``resume=True`` option.
- **Artifact emission**: Records and manifests can be written to disk
  for analysis and debugging.
- **Schema validation**: Outputs can be validated against versioned schemas.
- **Progress tracking**: Callbacks receive progress information including
  current item, elapsed time, and ETA.

See Also
--------
insideLLMs.runtime.runner : The canonical module with full implementation.
insideLLMs.config_types.RunConfig : Configuration options for runners.
insideLLMs.types.ExperimentResult : Structured experiment results.
insideLLMs.types.ProbeResult : Individual probe execution results.

Warnings
--------
This module exists solely for backward compatibility. New code should
import directly from ``insideLLMs.runtime.runner``.
"""

from insideLLMs.runtime import runner as _runner

# Re-export all public + private symbols to preserve legacy imports.
# This dynamic approach ensures any new symbols added to the runtime
# module are automatically available here without manual updates.
for _name in dir(_runner):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_runner, _name)

del _name, _runner
