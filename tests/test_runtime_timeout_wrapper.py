"""Tests for insideLLMs/runtime/timeout_wrapper.py module.

This module tests the timeout wrapper utilities used for probe execution
with configurable timeouts.
"""

import asyncio

import pytest

from insideLLMs.exceptions import ProbeExecutionError
from insideLLMs.runtime.timeout_wrapper import run_with_timeout


class TestRunWithTimeout:
    """Tests for run_with_timeout function."""

    @pytest.mark.asyncio
    async def test_completes_within_timeout(self):
        """Test that fast coroutines complete successfully."""

        async def fast_coro():
            await asyncio.sleep(0.01)
            return "success"

        result = await run_with_timeout(fast_coro, timeout=1.0)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_no_timeout_completes(self):
        """Test that None timeout allows completion."""

        async def slow_coro():
            await asyncio.sleep(0.05)
            return "completed"

        result = await run_with_timeout(slow_coro, timeout=None)
        assert result == "completed"

    @pytest.mark.asyncio
    async def test_timeout_raises_probe_execution_error(self):
        """Test that timeout raises ProbeExecutionError."""

        async def slow_coro():
            await asyncio.sleep(10.0)
            return "should not reach"

        with pytest.raises(ProbeExecutionError) as exc_info:
            await run_with_timeout(slow_coro, timeout=0.05)

        assert "timed out" in str(exc_info.value).lower()
        assert "0.05" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_with_context(self):
        """Test that context probe_type is used in error."""

        async def slow_coro():
            await asyncio.sleep(10.0)
            return "should not reach"

        context = {"model": "gpt-4", "probe": "factuality", "example_id": "123"}

        with pytest.raises(ProbeExecutionError) as exc_info:
            await run_with_timeout(slow_coro, timeout=0.05, context=context)

        # Probe type from context should be in error details
        assert exc_info.value.details["probe_type"] == "factuality"
        assert "timed out" in exc_info.value.details["reason"]

    @pytest.mark.asyncio
    async def test_zero_timeout_immediate_timeout(self):
        """Test that zero timeout triggers immediate timeout."""

        async def instant_coro():
            return "instant"

        # Zero timeout should still timeout since await has overhead
        with pytest.raises(ProbeExecutionError):
            await run_with_timeout(instant_coro, timeout=0.0)

    @pytest.mark.asyncio
    async def test_very_small_timeout(self):
        """Test behavior with very small timeout."""

        async def quick_coro():
            return "quick"

        # Very small but non-zero timeout may or may not complete
        # depending on system load - we just verify it doesn't crash
        try:
            result = await run_with_timeout(quick_coro, timeout=0.001)
            assert result == "quick"
        except ProbeExecutionError:
            pass  # Also acceptable

    @pytest.mark.asyncio
    async def test_coroutine_exception_propagates(self):
        """Test that exceptions from coroutine propagate correctly."""

        async def failing_coro():
            raise ValueError("Coroutine error")

        with pytest.raises(ValueError, match="Coroutine error"):
            await run_with_timeout(failing_coro, timeout=1.0)

    @pytest.mark.asyncio
    async def test_coroutine_exception_with_no_timeout(self):
        """Test exception propagation with no timeout."""

        async def failing_coro():
            raise RuntimeError("No timeout error")

        with pytest.raises(RuntimeError, match="No timeout error"):
            await run_with_timeout(failing_coro, timeout=None)

    @pytest.mark.asyncio
    async def test_return_value_preserved(self):
        """Test that various return types are preserved."""

        async def return_dict():
            return {"key": "value", "nested": {"a": 1}}

        async def return_list():
            return [1, 2, 3]

        async def return_none():
            return None

        dict_result = await run_with_timeout(return_dict, timeout=1.0)
        assert dict_result == {"key": "value", "nested": {"a": 1}}

        list_result = await run_with_timeout(return_list, timeout=1.0)
        assert list_result == [1, 2, 3]

        none_result = await run_with_timeout(return_none, timeout=1.0)
        assert none_result is None

    @pytest.mark.asyncio
    async def test_context_empty_uses_unknown_probe(self):
        """Test that unknown probe_type is used when no context provided."""

        async def slow_coro():
            await asyncio.sleep(10.0)

        with pytest.raises(ProbeExecutionError) as exc_info:
            await run_with_timeout(slow_coro, timeout=0.05, context=None)

        # Should use "unknown" as probe_type when no context
        assert exc_info.value.details["probe_type"] == "unknown"

    @pytest.mark.asyncio
    async def test_timeout_cancels_coroutine(self):
        """Test that the coroutine is properly cancelled on timeout."""
        cleanup_called = False

        async def coro_with_cleanup():
            nonlocal cleanup_called
            try:
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                cleanup_called = True
                raise

        with pytest.raises(ProbeExecutionError):
            await run_with_timeout(coro_with_cleanup, timeout=0.05)

        # Give a moment for cleanup
        await asyncio.sleep(0.01)
        assert cleanup_called


class TestRunWithTimeoutEdgeCases:
    """Edge case tests for run_with_timeout."""

    @pytest.mark.asyncio
    async def test_negative_timeout_treated_as_immediate(self):
        """Test that negative timeout is handled."""

        async def quick_coro():
            return "result"

        # Negative timeout should behave like zero/immediate timeout
        with pytest.raises(ProbeExecutionError):
            await run_with_timeout(quick_coro, timeout=-1.0)

    @pytest.mark.asyncio
    async def test_large_timeout_value(self):
        """Test that large timeout values work correctly."""

        async def quick_coro():
            return "fast"

        # Very large timeout should not cause issues
        result = await run_with_timeout(quick_coro, timeout=999999.0)
        assert result == "fast"

    @pytest.mark.asyncio
    async def test_concurrent_timeout_operations(self):
        """Test multiple concurrent timeout operations."""

        async def variable_delay(delay: float, value: str):
            await asyncio.sleep(delay)
            return value

        async def run_one(delay, value, timeout):
            try:
                return await run_with_timeout(lambda: variable_delay(delay, value), timeout=timeout)
            except ProbeExecutionError:
                return "timeout"

        results = await asyncio.gather(
            run_one(0.01, "fast", 1.0),
            run_one(10.0, "slow", 0.05),
            run_one(0.02, "medium", 1.0),
        )

        assert results[0] == "fast"
        assert results[1] == "timeout"
        assert results[2] == "medium"

    @pytest.mark.asyncio
    async def test_timeout_precision(self):
        """Test that timeout is reasonably precise."""
        import time

        async def sleep_coro():
            await asyncio.sleep(10.0)

        start = time.perf_counter()
        with pytest.raises(ProbeExecutionError):
            await run_with_timeout(sleep_coro, timeout=0.1)
        elapsed = time.perf_counter() - start

        # Should timeout around 0.1s (with some tolerance)
        assert 0.08 <= elapsed <= 0.3

    @pytest.mark.asyncio
    async def test_context_with_probe_key(self):
        """Test context with probe key is properly extracted."""

        async def slow_coro():
            await asyncio.sleep(10.0)

        context = {
            "probe": "test_probe",
            "model": "test_model",
        }

        with pytest.raises(ProbeExecutionError) as exc_info:
            await run_with_timeout(slow_coro, timeout=0.05, context=context)

        # Probe type should be extracted from context
        assert exc_info.value.details["probe_type"] == "test_probe"


class TestRunWithTimeoutIntegration:
    """Integration tests for timeout wrapper with realistic scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_simulated_probe_execution(self):
        """Test timeout behavior simulating probe execution."""

        async def simulate_probe(prompt: str, model_delay: float):
            await asyncio.sleep(model_delay)
            return {"input": prompt, "output": f"Response to: {prompt}"}

        # Fast model response
        result = await run_with_timeout(
            lambda: simulate_probe("Hello", 0.01),
            timeout=1.0,
            context={"probe": "test", "model": "fast"},
        )
        assert result["output"] == "Response to: Hello"

        # Slow model times out
        with pytest.raises(ProbeExecutionError):
            await run_with_timeout(
                lambda: simulate_probe("Hello", 10.0),
                timeout=0.05,
                context={"probe": "test", "model": "slow"},
            )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_batch_with_mixed_timeouts(self):
        """Test batch processing where some items timeout."""
        results = []
        errors = []

        async def process_item(item_id: int, delay: float):
            await asyncio.sleep(delay)
            return f"processed_{item_id}"

        items = [
            (0, 0.01),  # Fast
            (1, 10.0),  # Timeout
            (2, 0.02),  # Fast
            (3, 10.0),  # Timeout
            (4, 0.01),  # Fast
        ]

        for item_id, delay in items:
            try:
                result = await run_with_timeout(
                    lambda i=item_id, d=delay: process_item(i, d),
                    timeout=0.1,
                    context={"probe": f"probe_{item_id}"},
                )
                results.append(result)
            except ProbeExecutionError:
                errors.append(item_id)

        assert len(results) == 3
        assert len(errors) == 2
        assert set(errors) == {1, 3}
