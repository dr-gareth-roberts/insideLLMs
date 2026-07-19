"""W7-0008 slice 12: semantic_cache/retry/schemas/bias/ultimate gaps."""

from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from insideLLMs.models import DummyModel
from insideLLMs.probes.bias import BiasProbe
from insideLLMs.retry import (
    BackoffStrategy,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitState,
    RetryConfig,
    execute_with_retry,
)
from insideLLMs.runtime._ultimate import (
    _load_normalized_receipts_for_merkle,
    run_ultimate_post_artifact,
)
from insideLLMs.schemas.registry import SchemaRegistry, semver_tuple


def test_semantic_cache_cosine_and_redis_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover cosine fallback + RedisCache(client=) without reloading the module.

    Reloading semantic_cache poisons other suites that hold RedisCache class
    references from the original module object (patch targets the new module).
    """
    import insideLLMs.semantic_cache as sc

    # numpy path (zero-norm + happy)
    fake_np = types.SimpleNamespace(
        array=lambda x: x,
        dot=lambda a, b: sum(i * j for i, j in zip(a, b)),
        linalg=types.SimpleNamespace(norm=lambda a: 0.0 if not any(a) else 1.0),
    )
    monkeypatch.setattr(sc, "NUMPY_AVAILABLE", True)
    monkeypatch.setattr(sc, "np", fake_np)
    assert sc.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert sc.cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    # pure-python path
    monkeypatch.setattr(sc, "NUMPY_AVAILABLE", False)
    assert sc.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert sc.cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    monkeypatch.setattr(sc, "REDIS_AVAILABLE", True)
    monkeypatch.setattr(sc, "redis", types.SimpleNamespace(Redis=MagicMock))
    client = MagicMock()
    cache = sc.RedisCache(client=client)
    assert cache._client is client
    # no-client path constructs redis.Redis(...)
    cache2 = sc.RedisCache()
    assert cache2._client is not None


def test_retry_else_backoff_and_circuit_half_open() -> None:
    cfg = RetryConfig(max_retries=0, strategy=BackoffStrategy.CONSTANT)
    # force unknown strategy via monkeypatch on instance
    cfg.strategy = "unknown"  # type: ignore[assignment]
    assert cfg.calculate_delay(1) >= 0

    cb = CircuitBreaker(
        name="t",
        config=CircuitBreakerConfig(failure_threshold=1, reset_timeout=0.01, half_open_max_calls=1),
    )
    # trip open
    with pytest.raises(RuntimeError):
        cb.execute(lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert cb._state == CircuitState.OPEN
    # force half-open
    cb._state = CircuitState.HALF_OPEN
    cb._half_open_calls = 0
    with pytest.raises(RuntimeError):
        cb.execute(lambda: (_ for _ in ()).throw(RuntimeError("y")))
    assert cb._state == CircuitState.OPEN

    cb2 = CircuitBreaker(
        name="t2",
        config=CircuitBreakerConfig(failure_threshold=5, reset_timeout=60, half_open_max_calls=1),
    )
    cb2._state = CircuitState.HALF_OPEN
    cb2._half_open_calls = 1
    with pytest.raises(CircuitBreakerOpen):
        cb2.execute(lambda: 1)


def test_schema_registry_edges() -> None:
    assert semver_tuple("x.y.z") == (0, 0, 0)

    reg = SchemaRegistry()
    with pytest.raises(KeyError):
        reg.get_model(reg.CUSTOM_TRACE, schema_version="bad@version")

    migrated = reg.migrate(
        reg.RUN_MANIFEST,
        {"schema_version": "1.0.0", "run_id": "r"},
        from_version="1.0.0",
        to_version="1.0.1",
        custom_migration=lambda d: {**d, "extra": 1},
    )
    assert migrated["schema_version"] == "1.0.1"
    assert migrated["run_completed"] is False
    assert migrated["extra"] == 1


def test_bias_probe_dict_pair_shapes() -> None:
    probe = BiasProbe()
    model = DummyModel()

    r1 = probe.run(model, {"pairs": [("a", "b")]})
    assert r1
    r2 = probe.run(model, {"prompt_pairs": [("c", "d")]})
    assert r2
    r3 = probe.run(model, {"prompt_a": "x", "prompt_b": "y"})
    assert r3
    r4 = probe.run(model, [{"a": "p", "b": "q"}])
    assert r4

    with pytest.raises(ValueError, match="prompt_a"):
        probe.run(model, {"foo": 1})
    with pytest.raises(ValueError, match="list of"):
        probe.run(model, "not-pairs")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pair dict"):
        probe.run(model, [{"prompt_a": "only"}])
    with pytest.raises(ValueError, match="2-item"):
        probe.run(model, [("only",)])


def test_ultimate_receipts_and_provided_roots(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts" / "calls.jsonl"
    receipts.parent.mkdir(parents=True)
    receipts.write_text(
        "\n"
        + json.dumps({"id": "1", "latency_ms": 12.0})
        + "\n\n"
        + json.dumps({"id": "2"})
        + "\n",
        encoding="utf-8",
    )
    loaded = _load_normalized_receipts_for_merkle(receipts)
    assert loaded[0]["latency_ms"] is None

    run = tmp_path / "run"
    run.mkdir()
    (run / "records.jsonl").write_text(json.dumps({"a": 1}) + "\n", encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({"run_id": "r"}), encoding="utf-8")
    run_ultimate_post_artifact(
        run,
        records_merkle_root="aa" * 32,
        receipts_merkle_root="bb" * 32,
        dataset_merkle_root="cc" * 32,
        promptset_merkle_root="dd" * 32,
        insidellms_version="0.0.0",
    )
    assert (run / "integrity" / "records.merkle.json").exists()
    assert (run / "integrity" / "dataset.merkle.json").exists()


def test_execute_with_retry_non_retryable() -> None:
    def boom():
        raise ValueError("no")

    with pytest.raises(ValueError):
        execute_with_retry(boom, (), {}, RetryConfig())
