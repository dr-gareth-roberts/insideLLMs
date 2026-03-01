"""Model pipeline with composable middleware for advanced model capabilities.

This module provides a middleware-based pipeline architecture for wrapping
language models with cross-cutting concerns such as caching, rate limiting,
retry logic, cost tracking, and execution tracing. The pipeline follows the
chain-of-responsibility pattern, allowing middleware to be composed in any
order to build sophisticated model interaction patterns.

The module supports both synchronous and asynchronous execution patterns,
with async methods providing true concurrent processing and batch operations
with progress tracking.

Overview
--------
The pipeline architecture consists of three main components:

1. **Middleware** - Interceptors that process requests/responses:
   - :class:`CacheMiddleware` - Response caching with TTL and LRU eviction
   - :class:`RateLimitMiddleware` - Token bucket rate limiting
   - :class:`RetryMiddleware` - Exponential backoff retry logic
   - :class:`CostTrackingMiddleware` - Token usage and cost estimation
   - :class:`TraceMiddleware` - Execution tracing and recording
   - :class:`PassthroughMiddleware` - Base for observation-only middleware

2. **Pipelines** - Wrappers that chain middleware together:
   - :class:`ModelPipeline` - Standard pipeline with sync/async support
   - :class:`AsyncModelPipeline` - Extended async features with batch processing

3. **Base Classes** - For creating custom middleware:
   - :class:`Middleware` - Abstract base for all middleware

Middleware Execution Order
--------------------------
Middleware executes in the order added to the pipeline for requests, and
in reverse order for responses. This enables patterns like:

    Request flow:  Cache -> RateLimit -> Retry -> Model
    Response flow: Model -> Retry -> RateLimit -> Cache

Examples
--------
Basic pipeline with caching and retry:

>>> from insideLLMs.models import OpenAIModel
>>> from insideLLMs.pipeline import (
...     ModelPipeline,
...     CacheMiddleware,
...     RetryMiddleware,
... )
>>>
>>> model = OpenAIModel(model_name="gpt-4")
>>> pipeline = ModelPipeline(
...     model,
...     middlewares=[
...         CacheMiddleware(cache_size=100, ttl_seconds=3600),
...         RetryMiddleware(max_retries=3),
...     ],
... )
>>> response = pipeline.generate("What is machine learning?")
>>> print(response[:100])
Machine learning is a subset of artificial intelligence...

Full-featured production pipeline:

>>> from insideLLMs.pipeline import (
...     ModelPipeline,
...     CacheMiddleware,
...     RateLimitMiddleware,
...     RetryMiddleware,
...     CostTrackingMiddleware,
...     TraceMiddleware,
... )
>>>
>>> # Create a production-ready pipeline with all features
>>> cost_tracker = CostTrackingMiddleware()
>>> trace_middleware = TraceMiddleware(run_id="prod_run_001")
>>>
>>> pipeline = ModelPipeline(
...     model,
...     middlewares=[
...         trace_middleware,           # Trace all operations
...         CacheMiddleware(            # Cache responses for 1 hour
...             cache_size=1000,
...             ttl_seconds=3600,
...         ),
...         RateLimitMiddleware(        # Limit to 60 req/min
...             requests_per_minute=60,
...             burst_size=10,
...         ),
...         RetryMiddleware(            # Retry with exponential backoff
...             max_retries=3,
...             initial_delay=1.0,
...             max_delay=30.0,
...         ),
...         cost_tracker,               # Track costs
...     ],
...     name="production_pipeline",
... )
>>>
>>> # Generate with full middleware chain
>>> response = pipeline.generate("Explain transformers architecture")
>>>
>>> # Check cache performance
>>> print(f"Cache hit rate: {pipeline.middlewares[1].hit_rate:.2%}")
>>>
>>> # Check cost statistics
>>> stats = cost_tracker.get_stats()
>>> print(f"Total tokens: {stats['total_tokens']}")
>>> print(f"Estimated cost: ${stats['estimated_cost_usd']:.4f}")

Async pipeline with batch processing:

>>> import asyncio
>>> from insideLLMs.pipeline import AsyncModelPipeline, CacheMiddleware
>>>
>>> async def process_batch():
...     pipeline = AsyncModelPipeline(
...         model,
...         middlewares=[CacheMiddleware(cache_size=500)],
...     )
...
...     prompts = [
...         "What is Python?",
...         "What is JavaScript?",
...         "What is Rust?",
...     ]
...
...     # Batch process with concurrency limit
...     results = await pipeline.abatch_generate(
...         prompts,
...         max_concurrency=5,
...         return_exceptions=True,
...     )
...
...     for prompt, result in zip(prompts, results):
...         print(f"Q: {prompt[:30]}... A: {result[:50]}...")
...
...     return results
>>>
>>> results = asyncio.run(process_batch())

Streaming with progress callbacks:

>>> async def stream_with_progress():
...     pipeline = AsyncModelPipeline(model, middlewares=[])
...
...     def on_progress(completed: int, total: int):
...         print(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")
...
...     def on_result(index: int, result: str):
...         print(f"Got result {index}: {result[:50]}...")
...
...     prompts = ["Q1", "Q2", "Q3", "Q4", "Q5"]
...     results = await pipeline.agenerate_with_callback(
...         prompts,
...         max_concurrency=3,
...         on_progress=on_progress,
...         on_result=on_result,
...     )
...     return results

Using TraceMiddleware for debugging:

>>> from insideLLMs.pipeline import ModelPipeline, TraceMiddleware
>>> from insideLLMs.tracing import TraceEventKind
>>>
>>> trace_mw = TraceMiddleware(run_id="debug_run_001", example_id="ex_001")
>>> pipeline = ModelPipeline(model, middlewares=[trace_mw])
>>>
>>> response = pipeline.generate("Hello, world!")
>>>
>>> # Examine trace events
>>> for event in trace_mw.recorder.events:
...     print(f"{event.kind.name}: {event.data}")
GENERATE_START: {'prompt': 'Hello, world!'}
GENERATE_END: {'response': 'Hello! How can I help you today?'}
>>>
>>> # Reset for new run
>>> trace_mw.reset(run_id="debug_run_002")

Custom middleware for logging:

>>> from insideLLMs.pipeline import PassthroughMiddleware
>>> import logging
>>>
>>> class LoggingMiddleware(PassthroughMiddleware):
...     '''Middleware that logs all requests and responses.'''
...
...     def __init__(self, logger_name: str = "insideLLMs"):
...         super().__init__()
...         self.logger = logging.getLogger(logger_name)
...
...     def process_generate(self, prompt: str, **kwargs):
...         self.logger.info(f"Request: {prompt[:50]}...")
...         response = super().process_generate(prompt, **kwargs)
...         self.logger.info(f"Response: {response[:50]}...")
...         return response
>>>
>>> pipeline = ModelPipeline(
...     model,
...     middlewares=[LoggingMiddleware("my_app"), RetryMiddleware()],
... )

Chat conversation through pipeline:

>>> from insideLLMs.models.base import ChatMessage
>>>
>>> pipeline = ModelPipeline(model, middlewares=[CacheMiddleware()])
>>>
>>> messages = [
...     ChatMessage(role="system", content="You are a helpful assistant."),
...     ChatMessage(role="user", content="What is the capital of France?"),
... ]
>>>
>>> response = pipeline.chat(messages)
>>> print(response)
The capital of France is Paris.
>>>
>>> # Continue the conversation
>>> messages.append(ChatMessage(role="assistant", content=response))
>>> messages.append(ChatMessage(role="user", content="What about Germany?"))
>>> response = pipeline.chat(messages)
>>> print(response)
The capital of Germany is Berlin.

Error handling with retry middleware:

>>> from insideLLMs.exceptions import ModelError, RateLimitError
>>>
>>> pipeline = ModelPipeline(
...     model,
...     middlewares=[
...         RetryMiddleware(max_retries=3, initial_delay=1.0),
...     ],
... )
>>>
>>> try:
...     response = pipeline.generate("Complex query...")
... except ModelError as e:
...     print(f"Failed after retries: {e}")
...     print(f"Total retries attempted: {pipeline.middlewares[0].total_retries}")

Notes
-----
**Middleware Order Matters**

The order of middleware affects behavior. Recommended order:

1. TraceMiddleware (first to capture everything)
2. CacheMiddleware (before rate limiting to avoid unnecessary waits)
3. RateLimitMiddleware (after cache to not count cache hits)
4. RetryMiddleware (wraps actual model calls)
5. CostTrackingMiddleware (last to measure actual usage)

**Thread Safety**

- Async methods use locks for thread-safe access to shared state
- Cache, rate limiter, and cost tracker maintain internal async locks
- Sync methods are not thread-safe; use async for concurrent access

**Cost Estimation**

CostTrackingMiddleware provides approximate costs based on:
- Token estimation: ~4 characters per token
- Pricing data from 2024 (subject to change)
- Model name matching for known pricing tiers

**Performance Considerations**

- CacheMiddleware uses in-memory dict with LRU eviction
- RateLimitMiddleware implements token bucket algorithm
- RetryMiddleware adds jitter to prevent thundering herd
- Batch operations use semaphores for concurrency control

See Also
--------
insideLLMs.models.base : Base model classes and protocols.
insideLLMs.tracing : Trace recording for debugging and analysis.
insideLLMs.exceptions : Error types for pipeline failures.

References
----------
.. [1] Chain of Responsibility Pattern
   https://en.wikipedia.org/wiki/Chain-of-responsibility_pattern

.. [2] Token Bucket Algorithm
   https://en.wikipedia.org/wiki/Token_bucket
"""

import warnings as _warnings

_warnings.warn(
    "Importing from 'insideLLMs.pipeline' is deprecated. "
    "Use 'from insideLLMs.runtime.pipeline import ...' instead. "
    "This shim will be removed in v1.0.",
    DeprecationWarning,
    stacklevel=2,
)

from insideLLMs.runtime.pipeline import *  # noqa: E402,F401,F403

