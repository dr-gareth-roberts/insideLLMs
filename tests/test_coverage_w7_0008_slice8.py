"""W7-0008 slice 8: config_types/retry/tokens/sync_runner measured gaps."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from insideLLMs.config_types import ProgressInfo, RunConfigBuilder
from insideLLMs.retry import (
    BackoffStrategy,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitState,
    RetryConfig,
)
from insideLLMs.tokens import SimpleTokenizer, TokenDistribution, VocabCoverage


def test_run_config_builder_remaining_setters() -> None:
    cfg = (
        RunConfigBuilder()
        .with_validation(enabled=True, schema_version="1.0.1", mode="lenient")
        .with_artifacts(
            enabled=True, run_dir="/tmp/r", run_root="/tmp", run_id="id", overwrite=True
        )
        .with_concurrency(2)
        .with_error_handling(stop_on_error=True)
        .with_stop_on_error(False)
        .with_dataset_info({"name": "d"})
        .with_config_snapshot({"model": "m"})
        .with_message_storage(True)
        .with_determinism(strict_serialization=True, deterministic_artifacts=True)
        .with_resume(True)
        .with_probe_batch(enabled=True, batch_workers=3)
        .with_output(return_experiment=False)
        .build()
    )
    assert cfg.schema_version == "1.0.1"
    assert cfg.resume is True
    assert cfg.batch_workers == 3
    assert cfg.use_probe_batch is True

    info = ProgressInfo(current=0, total=0, elapsed_seconds=0.0)
    assert info.is_complete is True
    info2 = ProgressInfo(current=1, total=2, elapsed_seconds=1.0)
    assert info2.is_complete is False


def test_retry_fibonacci_and_circuit_breaker() -> None:
    cfg = RetryConfig(
        strategy=BackoffStrategy.FIBONACCI, initial_delay=0.01, max_delay=1.0, jitter=False
    )
    assert cfg.calculate_delay(1) >= 0
    assert cfg.calculate_delay(5) >= cfg.calculate_delay(3)
    # unknown strategy fallback
    cfg2 = RetryConfig(strategy=BackoffStrategy.LINEAR, initial_delay=0.1, jitter=False)
    assert cfg2.calculate_delay(2) >= 0

    breaker = CircuitBreaker(
        "t",
        CircuitBreakerConfig(
            failure_threshold=1,
            reset_timeout=0.05,
            success_threshold=1,
            half_open_max_calls=1,
        ),
    )
    # trip open
    try:
        with breaker:
            raise ConnectionError("fail")
    except ConnectionError:
        pass
    assert breaker.is_open
    with pytest.raises(CircuitBreakerOpen):
        with breaker:
            pass

    # wait for half-open
    time.sleep(0.06)
    breaker._check_state_transition()
    assert breaker.state == CircuitState.HALF_OPEN
    # succeed once to close
    with breaker:
        pass
    assert breaker.state == CircuitState.CLOSED

    # half_open max calls
    b2 = CircuitBreaker(
        "t2",
        CircuitBreakerConfig(failure_threshold=1, reset_timeout=0.01, half_open_max_calls=1),
    )
    try:
        with b2:
            raise RuntimeError("x")
    except RuntimeError:
        pass
    time.sleep(0.02)
    b2._check_state_transition()
    with b2:
        pass
    # force half open and exceed
    b2._state = CircuitState.HALF_OPEN
    b2._half_open_calls = b2.config.half_open_max_calls
    with pytest.raises(CircuitBreakerOpen):
        with b2:
            pass
    b2.reset()
    assert b2.state == CircuitState.CLOSED


def test_tokens_distribution_and_vocab_edges() -> None:
    dist = TokenDistribution(frequencies={}, total_tokens=0)
    assert dist.frequency_of("x") == 0
    assert dist.relative_frequency("x") == 0.0
    assert dist.tokens_above_frequency(1) == []
    assert dist.hapax_legomena() == []
    assert dist.entropy() == 0.0

    dist2 = TokenDistribution(frequencies={"a": 1, "b": 2}, total_tokens=3)
    assert dist2.relative_frequency("a") > 0
    assert "a" in dist2.hapax_legomena()
    assert dist2.entropy() >= 0

    cov = VocabCoverage(
        text_vocab={"a", "b"},
        reference_vocab={"a", "c"},
        covered={"a"},
        uncovered={"b"},
    )
    assert 0 < cov.coverage_ratio <= 1
    assert cov.oov_ratio >= 0
    d = cov.to_dict()
    assert "coverage_ratio" in d

    tok = SimpleTokenizer()
    ids = tok.encode("Hello, world!")
    assert tok.decode(ids)
    assert tok.tokenize("Hi there")
    assert tok.vocab_size >= 1


def test_sync_runner_strict_serialization_error() -> None:
    from insideLLMs._serialization import StrictSerializationError
    from insideLLMs.models import DummyModel
    from insideLLMs.probes.logic import LogicProbe
    from insideLLMs.runtime._sync_runner import ProbeRunner

    runner = ProbeRunner(DummyModel(), LogicProbe())
    with patch(
        "insideLLMs.runtime._sync_runner._deterministic_run_id_from_inputs",
        side_effect=StrictSerializationError("bad"),
    ):
        with pytest.raises(ValueError, match="JSON-stable"):
            runner.run(
                ["hi"],
                emit_run_artifacts=False,
                strict_serialization=True,
                run_id=None,
            )
