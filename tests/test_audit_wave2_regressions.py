"""Regression tests for production-quality audit wave 2 (medium-severity) fixes.

Each test pins a behavior that was previously incorrect; they should fail if the
corresponding fix is reverted.
"""

import pytest


# M7 — generate_cache_key: dict/list-valued kwargs must hash by content.
def test_cache_key_deterministic_for_dict_kwargs():
    from insideLLMs.caching import generate_cache_key

    k1 = generate_cache_key("p", meta={"b": 2, "a": 1})
    k2 = generate_cache_key("p", meta={"a": 1, "b": 2})
    assert k1 == k2


# M12 — CachedModel must not crash on a StrategyCache cache hit.
def test_cached_model_strategy_cache_hit_does_not_crash():
    from insideLLMs.caching import CachedModel, ModelResponse, StrategyCache

    class _Model:
        model_id = "stub"

        def generate(self, prompt, **kwargs):
            return ModelResponse(content="hi", model="stub")

    cached = CachedModel(_Model(), cache=StrategyCache())
    first = cached.generate("hello", temperature=0)
    second = cached.generate("hello", temperature=0)  # cache hit path (previously crashed)
    assert second.content == first.content == "hi"


# M2 — retry attempts count reflects real calls, not max_retries+1.
def test_retry_attempts_count_on_non_retryable_error():
    from insideLLMs.rate_limiting import RateLimitRetryConfig, RetryHandler

    handler = RetryHandler(
        RateLimitRetryConfig(max_retries=5, base_delay=0, retryable_errors=[KeyError])
    )

    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise ValueError("not retryable")

    result = handler.execute(fn)
    assert result.success is False
    assert calls["n"] == 1
    assert result.attempts == 1


# M15 — acquiring more tokens than capacity fails loudly.
def test_token_bucket_rejects_impossible_request():
    from insideLLMs.rate_limiting import TokenBucketRateLimiter

    limiter = TokenBucketRateLimiter(rate=1000.0, capacity=5)
    with pytest.raises(ValueError):
        limiter.acquire(tokens=10, block=True)


# M13 — _serialize_for_json must canonicalize sets to sorted lists.
def test_serialize_for_json_sorts_sets():
    from insideLLMs.results import _serialize_for_json

    assert _serialize_for_json({"tags": {"b", "a", "c"}}) == {"tags": ["a", "b", "c"]}


# M11 — code-block extraction handles languages with regex metachars.
def test_code_extraction_handles_cpp_language():
    from insideLLMs.probes.code import CodeGenerationProbe

    probe = CodeGenerationProbe(language="c++")
    code = probe.extract_code("Here is code:\n```c++\nint main(){return 0;}\n```")
    assert code == "int main(){return 0;}"


# M23 — two addresses in one string redact to two tokens, preserving between text.
def test_mask_pii_handles_multiple_addresses():
    from insideLLMs.safety import mask_pii

    masked = mask_pii("visit 456 oak ave then 789 elm road")
    assert masked.count("[ADDRESS]") == 2
    assert "then" in masked


# M26 — a long legitimate refusal is not flagged as a successful attack.
def test_attack_probe_long_refusal_not_flagged():
    from insideLLMs.probes.attack import AttackProbe

    probe = AttackProbe()
    refusal = (
        "I will not help with that. It would be wrong and unsafe. "
        "Please consider a safer alternative instead. "
    ) * 6
    analysis = probe._analyze_response(refusal, "do something bad")
    assert analysis["attack_succeeded"] is False


# M14 — trace_fingerprint is order-independent across recorders sharing seq.
def test_trace_fingerprint_order_independent():
    from insideLLMs.trace.tracing import TraceRecorder, trace_fingerprint

    r1 = TraceRecorder()
    r1.record("a", {"v": 1})
    r2 = TraceRecorder()
    r2.record("b", {"v": 2})
    events = r1.events + r2.events
    assert trace_fingerprint(events) == trace_fingerprint(list(reversed(events)))


# M5 — instruction probes report a meaningful (non-zero) accuracy.
def test_instruction_probe_accuracy_is_meaningful():
    from insideLLMs.probes.instruction import ConstraintComplianceProbe

    probe = ConstraintComplianceProbe(constraint_type="word_limit", limit=3)
    compliant = probe.evaluate_single("two words", None)
    violation = probe.evaluate_single("this is five words here", None)
    assert compliant.metadata["is_correct"] is True
    assert violation.metadata["is_correct"] is False
    score = probe.score([compliant, violation])
    assert score.accuracy == pytest.approx(0.5)
