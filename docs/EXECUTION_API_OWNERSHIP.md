# Execution API ownership

This document names the canonical owner of each execution concern. Similar names
in different modules are not sufficient reason to merge behavior: consolidation
must preserve lifecycle, safety, result, and compatibility contracts.

## Canonical ownership

| Concern | Canonical owner | Compatibility boundary |
|---|---|---|
| Provider model interfaces | `insideLLMs.models` and `insideLLMs.models.base` | Provider SDK details stay behind `Model`/`AsyncModel`. |
| Model construction | `insideLLMs.registry` and `insideLLMs.runtime._config_loader` | `InferenceClient.from_model_config()` delegates here; inference does not create providers. |
| Model-call middleware | `insideLLMs.runtime.pipeline` (`insideLLMs.pipeline` facade) | Retry, rate limiting, response caching, cost tracking, and transport tracing wrap model calls before inference strategies see them. |
| Generic retry utilities | `insideLLMs.retry` | Application-function decorators are distinct from model middleware and idempotent tool-action retries. |
| Cache stores and eviction | `insideLLMs.caching` | Storage lifecycle, TTL, scopes, and eviction live here. |
| Semantic response caching | `insideLLMs.semantic_cache` | Approximate reuse has separate correctness and invalidation policy. |
| Stable-prefix composition and identity | `insideLLMs.inference.prefix_cache` | It partitions prompts and composes tenant-safe keys; it is not a cache store. |
| Retrieval storage and search | Application retriever or `insideLLMs.contrib.retrieval` | `inference.rerank_and_assemble()` consumes a structural `document.id`/`document.content` contract and does not depend on a retrieval implementation. |
| Online inference strategies | `insideLLMs.inference` and `InferenceClient` | One-shot, self-consistency, verifier-first Best-of-N, repair, escalation, DAGs, search, and tool policy return the shared inference contracts. |
| Comparative ensemble analysis | `insideLLMs.contrib.ensemble` | Multi-model aggregation, similarity grouping, and evaluation reports remain analysis APIs. Its legacy `BEST_OF_N` mode delegates deterministic score ranking to `inference.best_of_n`. |
| Semantic model dispatch | `insideLLMs.contrib.routing` | Query-to-route matching is distinct from confidence-driven adaptive escalation. |
| Agent experiments | `insideLLMs.contrib.agents` | ReAct/memory/agent-loop research remains in `contrib`; production tool actions use inference allowlists, budgets, idempotency, and timeout rules. |
| Prompt compression and ablation | `insideLLMs.optimization` | Deterministic prompt analysis is distinct from evaluator-driven evolutionary artifact search. |

## Dependency direction

The allowed direction is:

```text
provider SDKs -> models -> runtime pipeline -> inference client -> application
                                      ^
                                      |
                          cache/retry/trace infrastructure
```

The strategy core accepts protocols and callbacks. It must not import provider
SDKs, provider model implementations, or `contrib`. The one exception is the
lazy runtime configuration import inside `InferenceClient.from_model_config()`.
`tests/inference/test_architecture.py` enforces this boundary.

## Compatibility policy

1. Existing `contrib`, retry, cache, routing, and optimization APIs remain
   backward compatible until a behavior-equivalent replacement exists.
2. Similar terminology alone does not justify delegation or deprecation.
3. A migration must first prove matching inputs, outputs, failure handling,
   accounting, cancellation, and determinism with contract tests.
4. New application-facing online generation features enter through
   `InferenceClient` and return `InferenceResult`.
5. New transport retries, provider caching, rate limiting, and cost tracking
   enter through runtime middleware rather than strategy-specific loops.
6. Deprecations require a tested adapter, migration example, and dated removal
   window; no silent redirects.

## Audited non-duplicates

- `contrib.ensemble.AggregationMethod.BEST_OF_N` scores pre-collected model
  outputs. `inference.select_best()` generates bounded candidates and applies
  ordered fail-closed verifiers. Their orchestration contracts remain distinct,
  but both now delegate score-once deterministic tie-breaking to
  `inference.best_of_n.rank_candidates()`.
- `contrib.ensemble` majority voting groups multi-model responses by similarity.
  Inference self-consistency samples one model sequentially and can stop when
  the leader is mathematically uncatchable.
- `contrib.routing.SemanticRouter` selects a route from query similarity and
  patterns. `inference.escalate_adaptively()` executes a policy-ordered model
  ladder based on candidate confidence.
- `runtime.pipeline.RetryMiddleware` retries transport-level model failures.
  `inference.tools` retries only explicitly idempotent actions and only for
  transport-like exceptions.
- `runtime.pipeline.CacheMiddleware` and `insideLLMs.caching` store responses.
  `inference.prefix_cache` only creates stable prompt partitions, keys, and
  telemetry.

The first Phase 2 consolidation slice therefore locks these boundaries rather
than forcing incompatible public APIs through an adapter that would alter
behavior.

## Retrieval migration example

Inference reranking previously imported the concrete `contrib.retrieval`
result type. It now uses structural protocols, so existing contrib results and
application-owned result objects both work without conversion:

```python
from dataclasses import dataclass

from insideLLMs.inference import rerank_and_assemble


@dataclass
class Document:
    id: str
    content: str


@dataclass
class SearchHit:
    document: Document


evidence = await rerank_and_assemble(
    "Which policy applies?",
    [SearchHit(Document("policy-7", "Approved policy text"))],
    rerank=lambda query, document: 1.0,
    top_k=1,
    max_tokens=100,
)
```

This removes the core-to-`contrib` dependency without deprecating or changing
`insideLLMs.contrib.retrieval.Document` and `RetrievalResult`.

## Legacy Best-of-N compatibility

`ResponseAggregator` remains synchronous and continues to return
`AggregatedOutput`. For `AggregationMethod.BEST_OF_N`, it converts each legacy
output to an index-keyed candidate and delegates only deterministic ranking to
the inference-owned kernel. Its explicit `input_order` policy preserves the
historical `max()` behavior—including first-input ties and order-sensitive NaN
scores—while evaluating each scorer exactly once.

Applications needing generation budgets, hard verifiers, blinded judging,
trace events, or spend accounting should migrate to `InferenceClient.best_of_n()`.
