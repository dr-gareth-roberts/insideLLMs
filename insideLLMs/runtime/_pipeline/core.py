"""Synchronous and asynchronous model-pipeline orchestration."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any, Callable, Optional

from insideLLMs.exceptions import ModelError
from insideLLMs.models.base import AsyncModelProtocol, ChatMessage, Model, ModelProtocol
from insideLLMs.runtime._pipeline.caching import CacheMiddleware
from insideLLMs.runtime._pipeline.cost import CostTrackingMiddleware
from insideLLMs.runtime._pipeline.middleware import Middleware
from insideLLMs.runtime._pipeline.retry import RetryMiddleware


class ModelPipeline(Model):
    """A model wrapper that composes multiple middleware for enhanced capabilities.

    The pipeline executes middleware in order for requests and reverse order
    for responses, following the chain of responsibility pattern.

    Args:
        base_model: The underlying model to wrap.
        middlewares: List of middleware to apply (in order).
        name: Optional name for the pipeline (defaults to base model name).

    Example:
        >>> pipeline = ModelPipeline(
        ...     OpenAIModel("gpt-4"),
        ...     middlewares=[
        ...         CacheMiddleware(cache_size=500),
        ...         RateLimitMiddleware(requests_per_minute=60),
        ...         RetryMiddleware(max_retries=3),
        ...         CostTrackingMiddleware(),
        ...     ],
        ... )
        >>> response = pipeline.generate("Hello, world!")
    """

    def __init__(
        self,
        base_model: ModelProtocol,
        middlewares: Optional[list[Middleware]] = None,
        name: Optional[str] = None,
    ):
        """Initialize the pipeline."""
        self.base_model = base_model
        self.middlewares = middlewares or []

        # Chain middleware together
        prev: Optional[Middleware] = None
        for middleware in self.middlewares:
            middleware.model = base_model
            if prev:
                prev.next_middleware = middleware
            prev = middleware

        # Initialize base Model
        pipeline_name = name or f"{base_model.name}_pipeline"
        model_id = getattr(base_model, "model_id", base_model.name)
        super().__init__(name=pipeline_name, model_id=model_id)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate through the middleware pipeline."""
        if self.middlewares:
            return self.middlewares[0].process_generate(prompt, **kwargs)
        return self.base_model.generate(prompt, **kwargs)

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Chat through the middleware pipeline."""
        if self.middlewares:
            return self.middlewares[0].process_chat(messages, **kwargs)
        if hasattr(self.base_model, "chat"):
            return self.base_model.chat(messages, **kwargs)
        raise ModelError("Base model does not support chat")

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream through the middleware pipeline."""
        if self.middlewares:
            yield from self.middlewares[0].process_stream(prompt, **kwargs)
        elif hasattr(self.base_model, "stream"):
            yield from self.base_model.stream(prompt, **kwargs)
        else:
            raise ModelError("Base model does not support streaming")

    # Async methods

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously generate through the middleware pipeline."""
        if self.middlewares:
            return await self.middlewares[0].aprocess_generate(prompt, **kwargs)
        if isinstance(self.base_model, AsyncModelProtocol):
            return await self.base_model.agenerate(prompt, **kwargs)
        # Fall back to executor for sync model
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.base_model.generate(prompt, **kwargs))

    async def achat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Asynchronously chat through the middleware pipeline."""
        if self.middlewares:
            return await self.middlewares[0].aprocess_chat(messages, **kwargs)
        if hasattr(self.base_model, "achat"):
            return await self.base_model.achat(messages, **kwargs)
        if hasattr(self.base_model, "chat"):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: self.base_model.chat(messages, **kwargs)
            )
        raise ModelError("Base model does not support chat")

    async def astream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Asynchronously stream through the middleware pipeline."""
        if self.middlewares:
            async for chunk in self.middlewares[0].aprocess_stream(prompt, **kwargs):
                yield chunk
        elif hasattr(self.base_model, "astream"):
            async for chunk in self.base_model.astream(prompt, **kwargs):
                yield chunk
        elif hasattr(self.base_model, "stream"):
            loop = asyncio.get_running_loop()
            chunks = await loop.run_in_executor(
                None, lambda: list(self.base_model.stream(prompt, **kwargs))
            )
            for chunk in chunks:
                yield chunk
        else:
            raise ModelError("Base model does not support streaming")

    async def abatch_generate(
        self,
        prompts: list[str],
        *,
        max_concurrency: int = 10,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        """Generate responses for multiple prompts concurrently.

        Args:
            prompts: List of prompts to process.
            max_concurrency: Maximum concurrent requests.
            return_exceptions: If True, include exceptions in results as strings.
            **kwargs: Additional arguments for generation.

        Returns:
            List of responses in same order as prompts.

        Example:
            >>> results = await pipeline.abatch_generate(
            ...     ["Q1", "Q2", "Q3"],
            ...     max_concurrency=5
            ... )
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        results: list[Any] = [None] * len(prompts)

        async def process(index: int, prompt: str) -> None:
            async with semaphore:
                try:
                    results[index] = await self.agenerate(prompt, **kwargs)
                except Exception as e:
                    if return_exceptions:
                        results[index] = f"Error: {e}"
                    else:
                        raise

        tasks = [asyncio.create_task(process(i, prompt)) for i, prompt in enumerate(prompts)]
        await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        return results

    def info(self) -> dict[str, Any]:
        """Get pipeline information including middleware stats."""
        from dataclasses import asdict, is_dataclass

        base_info = self.base_model.info()

        # Convert ModelInfo dataclass to dict if needed
        if is_dataclass(base_info):
            base_info_dict = asdict(base_info)
        else:
            base_info_dict = base_info

        pipeline_info = {
            **base_info_dict,
            "pipeline": True,
            "middleware_count": len(self.middlewares),
            "middlewares": [type(m).__name__ for m in self.middlewares],
        }

        # Add middleware-specific stats
        for middleware in self.middlewares:
            if isinstance(middleware, CacheMiddleware):
                pipeline_info["cache_hit_rate"] = middleware.hit_rate
                pipeline_info["cache_hits"] = middleware.hits
                pipeline_info["cache_misses"] = middleware.misses
            elif isinstance(middleware, RetryMiddleware):
                pipeline_info["total_retries"] = middleware.total_retries
            elif isinstance(middleware, CostTrackingMiddleware):
                pipeline_info["cost_stats"] = middleware.get_stats()

        return pipeline_info


class AsyncModelPipeline(ModelPipeline):
    """Async-first model pipeline optimized for concurrent workloads.

    Extends ModelPipeline with additional async features like batch processing
    with progress tracking and advanced concurrency control.

    Example:
        >>> async def main():
        ...     pipeline = AsyncModelPipeline(
        ...         base_model,
        ...         middlewares=[CacheMiddleware(), RateLimitMiddleware()],
        ...     )
        ...     # Single request
        ...     response = await pipeline.agenerate("Hello")
        ...
        ...     # Batch processing with progress
        ...     async for result in pipeline.agenerate_stream_results(prompts):
        ...         print(f"Got result: {result}")
    """

    async def agenerate_with_callback(
        self,
        prompts: list[str],
        *,
        max_concurrency: int = 10,
        on_progress: Optional[Callable[[int, int], None]] = None,
        on_result: Optional[Callable[[int, str], None]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Generate with progress and result callbacks.

        Args:
            prompts: List of prompts to process.
            max_concurrency: Maximum concurrent requests.
            on_progress: Callback(completed, total) for progress updates.
            on_result: Callback(index, result) when each result completes.
            **kwargs: Additional arguments for generation.

        Returns:
            List of responses in order.
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        results: list[Any] = [None] * len(prompts)
        completed = 0
        total = len(prompts)

        async def process(index: int, prompt: str) -> None:
            nonlocal completed
            async with semaphore:
                try:
                    result = await self.agenerate(prompt, **kwargs)
                    results[index] = result
                    if on_result:
                        on_result(index, result)
                except Exception as e:
                    results[index] = f"Error: {e}"
                    if on_result:
                        on_result(index, f"Error: {e}")
                finally:
                    completed += 1
                    if on_progress:
                        on_progress(completed, total)

        tasks = [asyncio.create_task(process(i, prompt)) for i, prompt in enumerate(prompts)]
        await asyncio.gather(*tasks)
        return results

    async def agenerate_stream_results(
        self,
        prompts: list[str],
        *,
        max_concurrency: int = 10,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[int, str]]:
        """Generate and yield results as they complete.

        Yields results in completion order (not input order).

        Args:
            prompts: List of prompts to process.
            max_concurrency: Maximum concurrent requests.
            **kwargs: Additional arguments for generation.

        Yields:
            Tuples of (index, result) as each completes.

        Example:
            >>> async for idx, result in pipeline.agenerate_stream_results(prompts):
            ...     print(f"Prompt {idx}: {result[:50]}...")
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()

        async def process(index: int, prompt: str) -> None:
            async with semaphore:
                try:
                    result = await self.agenerate(prompt, **kwargs)
                    await queue.put((index, result))
                except Exception as e:
                    await queue.put((index, f"Error: {e}"))

        tasks = [asyncio.create_task(process(i, prompt)) for i, prompt in enumerate(prompts)]

        # Yield results as they complete
        for _ in range(len(prompts)):
            result = await queue.get()
            yield result

        # Ensure all tasks complete
        await asyncio.gather(*tasks)

    async def amap(
        self,
        prompts: list[str],
        *,
        max_concurrency: int = 10,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> list[tuple[str, Optional[str], Optional[Exception]]]:
        """Map prompts to responses with detailed error handling.

        Args:
            prompts: List of prompts to process.
            max_concurrency: Maximum concurrent requests.
            timeout: Optional timeout per request in seconds.
            **kwargs: Additional arguments for generation.

        Returns:
            List of (prompt, response, error) tuples.
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        results: list[tuple[str, Optional[str], Optional[Exception]]] = []

        async def process(prompt: str) -> tuple[str, Optional[str], Optional[Exception]]:
            async with semaphore:
                try:
                    if timeout:
                        result = await asyncio.wait_for(
                            self.agenerate(prompt, **kwargs), timeout=timeout
                        )
                    else:
                        result = await self.agenerate(prompt, **kwargs)
                    return (prompt, result, None)
                except Exception as e:
                    return (prompt, None, e)

        tasks = [asyncio.create_task(process(prompt)) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        return list(results)
