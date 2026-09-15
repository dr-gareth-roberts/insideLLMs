"""Regression tests for Cluster B resilience primitive fixes (B1–B7)."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from insideLLMs.async_utils import (
    AsyncTokenBucketRateLimiter,
    AsyncWorkerPool,
    first_completed,
    for_each_async,
    run_async,
)
from insideLLMs.caching import (
    DiskCache,
    InMemoryCache,
    StrategyCache,
    generate_cache_key,
)
from insideLLMs.rate_limiting import (
    TokenBucketRateLimiter,
    circuit_protected,
    with_retry,
)
from insideLLMs.retry import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitState,
)

# --- B1: async decorators await lambda-returned coroutines ---


@pytest.mark.asyncio
async def test_with_retry_async_decorator_executes_underlying_attempts() -> None:
    calls = {"n": 0}

    @with_retry(max_retries=2, base_delay=0.0)
    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError(f"fail-{calls['n']}")
        return "ok"

    assert await flaky() == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_circuit_protected_async_decorator_returns_value_once() -> None:
    @circuit_protected(failure_threshold=3, recovery_timeout=0.01)
    async def ok_fn() -> str:
        return "circuit-ok"

    assert await ok_fn() == "circuit-ok"


# --- B2: CircuitBreaker half-open recovery with success_threshold > max_calls ---


def test_circuit_breaker_recovers_with_default_success_threshold() -> None:
    """Default config needs 2 half-open successes but only 1 concurrent call.

    _half_open_calls must be an in-flight gauge (decremented after each call)
    or the breaker rejects forever after the first recovery success.
    """
    config = CircuitBreakerConfig(
        failure_threshold=1,
        success_threshold=2,
        reset_timeout=0.05,
        half_open_max_calls=1,
    )
    circuit = CircuitBreaker("recover", config=config)

    with pytest.raises(RuntimeError):
        circuit.execute(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert circuit.state == CircuitState.OPEN

    time.sleep(0.06)
    assert circuit.state == CircuitState.HALF_OPEN

    assert circuit.execute(lambda: "probe-1") == "probe-1"
    assert circuit.state == CircuitState.HALF_OPEN

    # Second sequential success must be admitted (gauge released after first).
    assert circuit.execute(lambda: "probe-2") == "probe-2"
    assert circuit.state == CircuitState.CLOSED

    # Further calls must not be stuck open forever.
    assert circuit.execute(lambda: "steady") == "steady"


def test_circuit_breaker_half_open_gauge_releases_on_failure() -> None:
    config = CircuitBreakerConfig(
        failure_threshold=1,
        success_threshold=2,
        reset_timeout=0.05,
        half_open_max_calls=1,
    )
    circuit = CircuitBreaker("fail-gauge", config=config)
    with pytest.raises(RuntimeError):
        circuit.execute(lambda: (_ for _ in ()).throw(RuntimeError("open")))
    time.sleep(0.06)

    with pytest.raises(RuntimeError):
        circuit.execute(lambda: (_ for _ in ()).throw(RuntimeError("still-bad")))
    assert circuit.state == CircuitState.OPEN
    assert circuit._half_open_calls == 0


def test_circuit_breaker_nested_contexts_drain_half_open_gauge() -> None:
    """Nested ``with breaker`` frames must each release their own half-open slot."""
    config = CircuitBreakerConfig(
        failure_threshold=1,
        success_threshold=2,
        reset_timeout=0.05,
        half_open_max_calls=2,
    )
    circuit = CircuitBreaker("nested", config=config)
    with pytest.raises(RuntimeError):
        circuit.execute(lambda: (_ for _ in ()).throw(RuntimeError("trip")))
    time.sleep(0.06)
    assert circuit.state == CircuitState.HALF_OPEN

    with circuit:
        with circuit:
            pass  # two nested half-open successes

    assert circuit._half_open_calls == 0
    assert circuit.state == CircuitState.CLOSED


def test_circuit_breaker_concurrent_contexts_no_half_open_leak() -> None:
    """Concurrent context managers must not leak half-open capacity."""
    config = CircuitBreakerConfig(
        failure_threshold=1,
        success_threshold=2,
        reset_timeout=0.05,
        half_open_max_calls=2,
    )
    circuit = CircuitBreaker("concurrent-ctx", config=config)
    with pytest.raises(RuntimeError):
        circuit.execute(lambda: (_ for _ in ()).throw(RuntimeError("trip")))
    time.sleep(0.06)
    assert circuit.state == CircuitState.HALF_OPEN

    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            with circuit:
                barrier.wait(timeout=2.0)
                time.sleep(0.02)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive()

    assert errors == []
    assert circuit._half_open_calls == 0
    assert circuit.state == CircuitState.CLOSED


def test_circuit_breaker_nested_inner_failure_counts_once() -> None:
    """Inner half-open failure must trip open once and drain the inner slot."""
    config = CircuitBreakerConfig(
        failure_threshold=1,
        success_threshold=2,
        reset_timeout=0.05,
        half_open_max_calls=2,
    )
    circuit = CircuitBreaker("nested-fail", config=config)
    with pytest.raises(RuntimeError):
        circuit.execute(lambda: (_ for _ in ()).throw(RuntimeError("trip")))
    time.sleep(0.06)
    assert circuit.state == CircuitState.HALF_OPEN

    with pytest.raises(RuntimeError):
        with circuit:
            with circuit:
                raise RuntimeError("inner-fail")

    assert circuit.state == CircuitState.OPEN
    assert circuit._half_open_calls == 0
    # Outer frame still exits after inner failure; gauge must not stick.
    with pytest.raises(CircuitBreakerOpen):
        with circuit:
            pass


@pytest.mark.asyncio
async def test_circuit_breaker_overlapping_async_tasks_respect_half_open_max() -> None:
    """Overlapping asyncio tasks share a thread; slots must stay task-local.

    With half_open_max_calls=1, only one concurrent probe may hold a slot.
    A shared threading.local stack would let a second task steal the first
    task's frame and admit a third probe incorrectly.
    """
    config = CircuitBreakerConfig(
        failure_threshold=1,
        success_threshold=3,
        reset_timeout=0.05,
        half_open_max_calls=1,
    )
    circuit = CircuitBreaker("async-overlap", config=config)
    with pytest.raises(RuntimeError):
        circuit.execute(lambda: (_ for _ in ()).throw(RuntimeError("trip")))
    await asyncio.sleep(0.06)
    assert circuit.state == CircuitState.HALF_OPEN

    hold = asyncio.Event()
    released = asyncio.Event()
    outcomes: list[str] = []

    async def holder() -> None:
        try:
            with circuit:
                outcomes.append("held")
                hold.set()
                await released.wait()
        except CircuitBreakerOpen:
            outcomes.append("holder-open")

    async def contender() -> None:
        await hold.wait()
        try:
            with circuit:
                outcomes.append("contender-entered")
        except CircuitBreakerOpen:
            outcomes.append("contender-rejected")

    async def third_probe() -> None:
        await hold.wait()
        # While holder still owns the only half-open slot, a third admit
        # must fail even if contender has already bounced.
        try:
            with circuit:
                outcomes.append("third-entered")
        except CircuitBreakerOpen:
            outcomes.append("third-rejected")

    t_holder = asyncio.create_task(holder())
    await hold.wait()
    t_contender = asyncio.create_task(contender())
    t_third = asyncio.create_task(third_probe())
    await asyncio.sleep(0.05)
    assert circuit._half_open_calls == 1
    released.set()
    await asyncio.gather(t_holder, t_contender, t_third)

    assert "held" in outcomes
    assert outcomes.count("contender-rejected") + outcomes.count("third-rejected") >= 2
    assert "contender-entered" not in outcomes
    assert "third-entered" not in outcomes
    assert circuit._half_open_calls == 0


# --- B3: blocking acquire under contention ---


def test_token_bucket_blocking_acquire_under_contention() -> None:
    limiter = TokenBucketRateLimiter(rate=50.0, capacity=1)
    results: list[bool] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            results.append(limiter.acquire(tokens=1, block=True))
        except BaseException as exc:  # noqa: BLE001 - collect for assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive(), "blocking acquire hung under contention"

    assert not errors
    assert results == [True, True, True, True]
    assert limiter._stats.allowed_requests == 4


# --- B4: async_utils stop_on_error, first_completed, workers, token cap ---


@pytest.mark.asyncio
async def test_for_each_async_stop_on_error_skips_remaining_with_concurrency_1() -> None:
    seen: list[int] = []

    async def boom(item: int) -> None:
        seen.append(item)
        if item == 0:
            raise ValueError("stop")

    errors = await for_each_async(
        boom,
        [0, 1, 2, 3, 4],
        max_concurrency=1,
        stop_on_error=True,
    )
    assert len(errors) == 1
    assert seen == [0]


@pytest.mark.asyncio
async def test_first_completed_cancels_pending_children() -> None:
    cancelled = {"slow": False}

    async def fast() -> str:
        return "fast"

    async def slow() -> str:
        try:
            await asyncio.sleep(10.0)
            return "slow"
        except asyncio.CancelledError:
            cancelled["slow"] = True
            raise

    result = await first_completed([fast(), slow()], cancel_remaining=True)
    assert result == "fast"
    # Allow cancellation to settle.
    await asyncio.sleep(0.05)
    assert cancelled["slow"] is True


@pytest.mark.asyncio
async def test_first_completed_parent_cancellation_drains_children() -> None:
    """Parent CancelledError must cancel+drain children (BaseException path)."""
    child_started = asyncio.Event()
    child_cancelled = {"n": 0}
    child_tasks: list[asyncio.Task[None]] = []

    async def sticky() -> str:
        child_tasks.append(asyncio.current_task())  # type: ignore[arg-type]
        child_started.set()
        try:
            await asyncio.sleep(30.0)
            return "never"
        except asyncio.CancelledError:
            child_cancelled["n"] += 1
            raise

    async def race() -> str:
        return await first_completed([sticky(), sticky()], cancel_remaining=True)

    parent = asyncio.create_task(race())
    await asyncio.wait_for(child_started.wait(), timeout=2.0)
    # Let both children schedule.
    await asyncio.sleep(0.05)
    parent.cancel()
    with pytest.raises(asyncio.CancelledError):
        await parent

    # No child task may still be running after parent cancellation settles.
    await asyncio.sleep(0.05)
    assert child_cancelled["n"] >= 1
    still_alive = [t for t in child_tasks if t is not None and not t.done()]
    assert still_alive == []


def test_async_worker_pool_rejects_zero_workers() -> None:
    async def noop(_item: int) -> int:
        return _item

    with pytest.raises(ValueError, match="num_workers"):
        AsyncWorkerPool(noop, num_workers=0)


@pytest.mark.asyncio
async def test_async_token_bucket_rejects_tokens_above_burst() -> None:
    limiter = AsyncTokenBucketRateLimiter(rate=10.0, burst=5)
    with pytest.raises(ValueError, match="burst"):
        await limiter.acquire(tokens=6)


# --- B5: run_async nested loop ---


@pytest.mark.asyncio
async def test_run_async_from_running_loop_raises_and_closes_coro() -> None:
    import warnings

    async def sample() -> str:
        return "never"

    coro = sample()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="running event loop"):
            run_async(coro)
        # Give the interpreter a chance to emit "never awaited" if leaked.
        await asyncio.sleep(0)

    never_awaited = [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning) and "never awaited" in str(w.message)
    ]
    assert never_awaited == []


def test_run_async_without_running_loop() -> None:
    async def sample() -> int:
        return 42

    assert run_async(sample()) == 42


# --- B6: cache key collision ---


def test_generate_cache_key_no_delimiter_collision() -> None:
    # Pipe-joined format previously collided; JSON canonical form must not.
    # Note: key format change invalidates older cache entries (acceptable).
    left = generate_cache_key("p|model:m")
    right = generate_cache_key("p", model="m")
    assert left != right

    # Determinism for identical inputs.
    assert generate_cache_key("hello", model="gpt", params={"t": 0.1}) == generate_cache_key(
        "hello", model="gpt", params={"t": 0.1}
    )


# --- B7: update-at-capacity and disk logical size ---


def test_inmemory_cache_update_existing_key_at_capacity_does_not_evict_other() -> None:
    cache: InMemoryCache[str] = InMemoryCache(max_size=2)
    cache.set("a", "1")
    cache.set("b", "2")
    cache.set("a", "1-updated")  # at capacity, replacing 'a'
    assert cache.get("a") == "1-updated"
    assert cache.get("b") == "2"


def test_strategy_cache_update_existing_key_at_capacity_does_not_evict_other() -> None:
    cache = StrategyCache(max_size=2)
    cache.set("a", "1")
    cache.set("b", "2")
    cache.set("a", "1-updated")
    assert cache.get("a").value == "1-updated"
    assert cache.get("b").value == "2"


def test_disk_cache_update_existing_key_at_capacity_keeps_other(tmp_path: Path) -> None:
    db = tmp_path / "cache.db"
    # Tiny logical budget so two small values fill capacity for new keys.
    cache: DiskCache[str] = DiskCache(path=db, max_size_mb=1)
    cache._max_size_bytes = 20  # type: ignore[attr-defined]
    cache.set("a", "xxxx")  # 6 chars JSON ~ '"xxxx"'
    cache.set("b", "yyyy")
    # Force capacity full by setting max below current logical size.
    cache._max_size_bytes = cache._logical_payload_bytes()  # type: ignore[attr-defined]
    cache.set("a", "zzzz")
    assert cache.get("a") == "zzzz"
    assert cache.get("b") == "yyyy"


def test_disk_cache_evicts_by_logical_payload_size(tmp_path: Path) -> None:
    db = tmp_path / "logical.db"
    cache: DiskCache[str] = DiskCache(path=db, max_size_mb=1)
    # Budget fits a few small values; many inserts must shrink logical size.
    cache._max_size_bytes = 200  # type: ignore[attr-defined]

    for i in range(30):
        cache.set(f"k{i}", "x" * 20)

    logical = cache._logical_payload_bytes()  # type: ignore[attr-defined]
    assert logical <= cache._max_size_bytes  # type: ignore[attr-defined]
    assert cache.stats().evictions > 0
    # Newest keys should still be present after LRU eviction of older ones.
    assert cache.get("k29") is not None


def test_disk_cache_grow_on_update_enforces_budget(tmp_path: Path) -> None:
    """Updating an existing key with a larger value must still enforce size."""
    db = tmp_path / "grow.db"
    cache: DiskCache[str] = DiskCache(path=db, max_size_mb=1)
    cache._max_size_bytes = 40  # type: ignore[attr-defined]

    cache.set("keep", "aa")
    cache.set("grow", "bb")
    # Replace with a much larger payload — previously bypassed eviction.
    cache.set("grow", "x" * 80)

    logical = cache._logical_payload_bytes()  # type: ignore[attr-defined]
    assert logical <= cache._max_size_bytes  # type: ignore[attr-defined]
    assert cache.stats().evictions > 0


def test_disk_cache_logical_size_uses_utf8_bytes(tmp_path: Path) -> None:
    """Multi-byte Unicode must count octets, not SQLite text characters."""
    db = tmp_path / "utf8.db"
    cache: DiskCache[str] = DiskCache(path=db, max_size_mb=1)
    # Three multi-byte codepoints: each is 3 UTF-8 bytes.
    value = "\u4e2d\u6587\u6d4b"  # 中文测 — 3 chars, 9 UTF-8 bytes
    cache.set("m", value)
    stored = json.dumps(value, ensure_ascii=False)
    char_len = len(stored)
    byte_len = len(stored.encode("utf-8"))
    assert byte_len > char_len
    assert cache._logical_payload_bytes() == byte_len  # type: ignore[attr-defined]

    # Character LENGTH would under-count; blob LENGTH matches UTF-8 octets.
    conn = cache._get_conn()  # type: ignore[attr-defined]
    row = conn.execute(
        "SELECT LENGTH(value) AS chars, LENGTH(CAST(value AS BLOB)) AS octets "
        "FROM cache WHERE key = ?",
        ("m",),
    ).fetchone()
    assert row["octets"] == byte_len
    assert row["chars"] == char_len
    assert row["octets"] > row["chars"]


def test_token_bucket_total_requests_counts_once_per_acquire() -> None:
    """Blocking retry spins must not inflate total_requests."""
    limiter = TokenBucketRateLimiter(rate=100.0, capacity=1)
    limiter._tokens = 0.0
    before = limiter._stats.total_requests

    refill_n = {"n": 0}

    def refill_side_effect() -> None:
        refill_n["n"] += 1
        if refill_n["n"] >= 2:
            limiter._tokens = 1.0

    with (
        patch.object(limiter, "_refill", side_effect=refill_side_effect),
        patch("insideLLMs.rate_limiting.time.sleep", return_value=None),
    ):
        assert limiter.acquire(tokens=1, block=True) is True

    assert refill_n["n"] >= 2  # loop spun more than once
    # One logical acquire → +1 total_requests even though the loop spun twice.
    assert limiter._stats.total_requests == before + 1
