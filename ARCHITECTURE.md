# insideLLMs Architecture

This document describes the current architecture and execution flows in insideLLMs.
It is intended for contributors and advanced users who want to understand how
models, probes, runners, and supporting utilities fit together.

Diagrams are rendered with Mermaid (supported by GitHub Markdown).

## High-Level Architecture

```mermaid
graph TD
  subgraph EntryPoints[Entry Points]
    CLI[CLI: insidellms]
    API[Python API]
  end

  subgraph CoreRuntime[Core Runtime]
    Runner[ProbeRunner / AsyncProbeRunner]
    Probe[Probe.run / Probe.run_batch]
    Model[Model.generate / chat / stream]
    Types[Types + Results]
  end

  subgraph Registries[Registry Layer]
    Registry[Registry]
    ModelReg[model_registry]
    ProbeReg[probe_registry]
    DatasetReg[dataset_registry]
  end

  subgraph DataLayer[Data + Datasets]
    DatasetLoaders[dataset_utils: CSV/JSONL/HF]
    BenchDatasets[benchmark_datasets]
  end

  subgraph Providers[Model Providers]
    OpenAI[OpenAI SDK]
    Anthropic[Anthropic SDK]
    HF[Transformers / HF]
    Local[Local: llama.cpp / ollama / vLLM]
  end

  subgraph Infra[Infra Utilities]
    Cache[Caching]
    RateLimit[Rate Limiting]
    Cost[Cost Tracking]
    Stream[Streaming Utilities]
  end

  CLI --> Runner
  API --> Runner
  Runner --> Probe
  Probe --> Model
  Model --> Providers
  Runner --> Types

  Registry --> ModelReg
  Registry --> ProbeReg
  Registry --> DatasetReg
  Runner --> Registry
  Runner --> DatasetLoaders

  BenchDatasets --> DatasetLoaders

  Model -. optional .-> Cache
  Model -. optional .-> RateLimit
  Model -. optional .-> Cost
  Model -. optional .-> Stream
```

Notes:
- The core flow runs through `ProbeRunner` and `Probe.run` into `Model.generate`.
- Registry and dataset loaders power config-driven and programmatic creation.
- Infra utilities exist as standalone modules and are not currently enforced by the runner.

## Target Architecture (Proposed)

This is a proposed architecture that makes infra capabilities first-class and
standardizes result types across runners and benchmarks.

Key deltas from current:
- Introduce a model pipeline with composable middleware (retry, rate limiting, caching, cost, streaming).
- Standardize all execution to return `ModelResponse` and `ExperimentResult`.
- Make batch and async execution explicit in the runner contract.

```mermaid
graph TD
  subgraph EntryPoints[Entry Points]
    CLI[CLI: insidellms]
    API[Python API]
  end

  subgraph Orchestration[Execution Orchestration]
    Runner[ExperimentRunner]
    Planner[Probe Scheduler]
  end

  subgraph Probing[Probing]
    Probe[Probe.run / run_batch]
  end

  subgraph Pipeline[Model Pipeline]
    MW[Middleware Chain]
    Retry[Retry + Backoff]
    RateLimit[Rate Limiting]
    Cache[Caching]
    Cost[Cost Tracking]
    Stream[Streaming Adapter]
    Trace[Tracing + Metrics]
  end

  subgraph Providers[Model Providers]
    Adapter[Provider Adapter]
    Provider[OpenAI / Anthropic / HF / Local]
  end

  subgraph Results[Results + Reporting]
    Result[ExperimentResult]
    Export[Export + Visualization]
  end

  CLI --> Runner
  API --> Runner
  Runner --> Planner
  Planner --> Probe
  Probe --> MW
  MW --> Retry
  MW --> RateLimit
  MW --> Cache
  MW --> Cost
  MW --> Stream
  MW --> Trace
  MW --> Adapter
  Adapter --> Provider
  Runner --> Result
  Result --> Export
```

## Proposed Model Pipeline Flow

```mermaid
sequenceDiagram
  participant P as Probe
  participant MP as ModelPipeline
  participant C as Cache
  participant RL as RateLimiter
  participant A as Provider Adapter
  participant CT as CostTracker

  P->>MP: generate(prompt, params)
  MP->>C: lookup(key)
  alt Cache hit
    C-->>MP: cached response
  else Cache miss
    MP->>RL: acquire()
    RL-->>MP: ok
    MP->>A: request(prompt, params)
    A-->>MP: response + usage
    MP->>C: store(key, response)
  end
  MP->>CT: record(usage)
  MP-->>P: ModelResponse
```

## Model Pipeline API Sketch (Proposed)

This sketch illustrates the intended API shape for composable model middleware.

```python
from insideLLMs.models import OpenAIModel
from insideLLMs.pipeline import (
    ModelPipeline,
    CacheMiddleware,
    RateLimitMiddleware,
    RetryMiddleware,
    CostTrackingMiddleware,
)

base_model = OpenAIModel(model_name="gpt-4o")

pipeline = ModelPipeline(
    base_model,
    middlewares=[
        CacheMiddleware(),
        RateLimitMiddleware(),
        RetryMiddleware(),
        CostTrackingMiddleware(),
    ],
)

response = pipeline.generate("Explain transformers in one paragraph.")
print(response.content)
```

## Proposed Streaming Flow

```mermaid
sequenceDiagram
  participant U as User Code
  participant MP as ModelPipeline
  participant A as Provider Adapter
  participant SB as StreamBuffer
  participant SM as StreamMetrics

  U->>MP: stream(prompt, params)
  MP->>A: stream_request(...)
  loop chunks
    A-->>MP: chunk
    MP->>SB: add_chunk(chunk)
    MP->>SM: update_metrics()
    MP-->>U: yield chunk
  end
  MP->>SM: finalize()
```

## Proposed Cost Tracking Flow

```mermaid
sequenceDiagram
  participant MP as ModelPipeline
  participant A as Provider Adapter
  participant CT as UsageTracker

  MP->>A: request(prompt, params)
  A-->>MP: response + usage
  MP->>CT: record(model, tokens, cost)
  CT-->>MP: ok
```

## Core Execution Flow (ProbeRunner)

```mermaid
sequenceDiagram
  participant U as User Code
  participant R as ProbeRunner
  participant P as Probe
  participant M as Model

  U->>R: run(prompt_set, ...)
  loop Each item
    R->>P: run(model, item, **probe_kwargs)
    P->>M: generate(prompt)
    M-->>P: response
    P-->>R: output
    R-->>U: append result
  end
  R-->>U: results list
```

Key behaviour:
- `ProbeRunner` iterates a dataset and calls `Probe.run` per item.
- `Probe.run` is responsible for formatting the prompt and calling the model.
- Results are returned as a list of dictionaries with input/output/error/latency.

## Config-Driven Execution Flow

```mermaid
sequenceDiagram
  participant CLI as CLI (insidellms)
  participant Runner as run_experiment_from_config
  participant Registry as Registry
  participant DS as Dataset Loader
  participant PR as ProbeRunner

  CLI->>Runner: run <config.yaml>
  Runner->>Runner: load_config(path)
  Runner->>Registry: model_registry.get(...)
  Runner->>Registry: probe_registry.get(...)
  Runner->>DS: load_csv/jsonl/hf
  Runner->>PR: run(dataset)
  PR-->>CLI: results
```

## Benchmark Flow (ModelBenchmark)

```mermaid
sequenceDiagram
  participant U as User Code
  participant B as ModelBenchmark
  participant R as run_probe
  participant P as Probe
  participant M as Model

  U->>B: run(prompt_set)
  loop Each model
    B->>R: run_probe(model, probe, dataset)
    R->>P: run(model, item)
    P->>M: generate(prompt)
    M-->>P: response
    P-->>R: output
    R-->>B: results + metrics
  end
  B-->>U: benchmark_results
```

## Supporting Subsystems

- **Registry** (`insideLLMs/registry.py`): Central registration system for models, probes, and dataset loaders.
- **Results & Export** (`insideLLMs/results.py`, `insideLLMs/types.py`): Structured experiment results and export helpers.
- **Infra Utilities** (`insideLLMs/caching_unified.py`, `insideLLMs/rate_limiting.py`, `insideLLMs/cost_tracking.py`, `insideLLMs/streaming.py`): Optional utilities that can be wired into model wrappers.
- **Prompt Tooling** (`insideLLMs/templates.py`, `insideLLMs/prompt_utils.py`, `insideLLMs/template_versioning.py`): Templates and versioning for prompt engineering workflows.

## Extension Points

- **Models**: Implement `Model.generate` (and optional `chat`/`stream`) in `insideLLMs/models/base.py`.
- **Probes**: Implement `Probe.run` (and optionally `run_batch`/`score`) in `insideLLMs/probes/base.py`.
- **Datasets**: Add loaders to `insideLLMs/dataset_utils.py` and register in `insideLLMs/registry.py`.
