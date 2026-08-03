"""Composable model pipeline exposed through a stable compatibility facade.

Implementations are separated by responsibility under ``runtime._pipeline``;
this module retains the established public imports.
"""

from insideLLMs.runtime._pipeline.caching import CacheMiddleware
from insideLLMs.runtime._pipeline.core import AsyncModelPipeline, ModelPipeline
from insideLLMs.runtime._pipeline.cost import CostTrackingMiddleware
from insideLLMs.runtime._pipeline.middleware import Middleware, PassthroughMiddleware
from insideLLMs.runtime._pipeline.rate_limit import RateLimitMiddleware
from insideLLMs.runtime._pipeline.retry import RetryMiddleware
from insideLLMs.runtime._pipeline.tracing import TraceMiddleware

__all__ = [
    "Middleware",
    "PassthroughMiddleware",
    "TraceMiddleware",
    "CacheMiddleware",
    "RateLimitMiddleware",
    "RetryMiddleware",
    "CostTrackingMiddleware",
    "ModelPipeline",
    "AsyncModelPipeline",
]

# Preserve the pre-refactor public identity for users and serialized references.
for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
