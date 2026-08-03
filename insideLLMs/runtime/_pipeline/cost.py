"""Request and token cost tracking middleware."""

import asyncio
from typing import Any

from insideLLMs.exceptions import ModelError
from insideLLMs.models.base import AsyncModelProtocol
from insideLLMs.runtime._pipeline.middleware import Middleware


class CostTrackingMiddleware(Middleware):
    """Middleware for tracking API costs and token usage.

    Tracks requests, tokens, and estimated costs for model usage.
    """

    # Approximate costs per 1K tokens (as of 2024, subject to change)
    COST_PER_1K_TOKENS = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }

    def __init__(self) -> None:
        """Initialize cost tracking."""
        super().__init__()
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.estimated_cost = 0.0

    def _estimate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on model and token counts."""
        # Try to match model name to known pricing
        model_key = None
        for key in self.COST_PER_1K_TOKENS:
            if key in model_name.lower():
                model_key = key
                break

        if not model_key:
            return 0.0  # Unknown model

        pricing = self.COST_PER_1K_TOKENS[model_key]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost

    def process_generate(self, prompt: str, **kwargs: Any) -> str:
        """Track costs for generation."""
        # Generate response
        if self.next_middleware:
            response = self.next_middleware.process_generate(prompt, **kwargs)
        elif self.model:
            response = self.model.generate(prompt, **kwargs)
        else:
            raise ModelError("No model available in pipeline")

        # Track usage
        self.total_requests += 1

        # Rough token estimation (4 chars ≈ 1 token)
        input_tokens = len(prompt) // 4
        output_tokens = len(response) // 4

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        # Estimate cost
        if self.model:
            model_name = getattr(self.model, "model_id", "unknown")
            cost = self._estimate_cost(model_name, input_tokens, output_tokens)
            self.estimated_cost += cost

        return response

    async def aprocess_generate(self, prompt: str, **kwargs: Any) -> str:
        """Async track costs for generation."""
        # Generate response
        if self.next_middleware:
            response = await self.next_middleware.aprocess_generate(prompt, **kwargs)
        elif self.model:
            if isinstance(self.model, AsyncModelProtocol):
                response = await self.model.agenerate(prompt, **kwargs)
            else:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None, lambda: self.model.generate(prompt, **kwargs)
                )
        else:
            raise ModelError("No model available in pipeline")

        # Track usage (thread-safe via lock)
        async with self._get_lock():
            self.total_requests += 1

            # Rough token estimation (4 chars ≈ 1 token)
            input_tokens = len(prompt) // 4
            output_tokens = len(response) // 4

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            # Estimate cost
            if self.model:
                model_name = getattr(self.model, "model_id", "unknown")
                cost = self._estimate_cost(model_name, input_tokens, output_tokens)
                self.estimated_cost += cost

        return response

    def _get_lock(self) -> asyncio.Lock:
        """Get or create an async lock for thread-safe tracking."""
        if not hasattr(self, "_lock"):
            self._lock = asyncio.Lock()
        return self._lock

    def get_stats(self) -> dict[str, Any]:
        """Get cost tracking statistics."""
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": round(self.estimated_cost, 4),
        }
