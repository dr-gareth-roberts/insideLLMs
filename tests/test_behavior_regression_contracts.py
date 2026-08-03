"""Regression contracts for previously incorrect user-visible behavior."""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import pytest


# generate_cache_key: dict/list-valued kwargs must hash by content.
def test_cache_key_deterministic_for_dict_kwargs():
    from insideLLMs.caching import generate_cache_key

    k1 = generate_cache_key("p", meta={"b": 2, "a": 1})
    k2 = generate_cache_key("p", meta={"a": 1, "b": 2})
    assert k1 == k2


# CachedModel must not crash on a StrategyCache cache hit.
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


# retry attempts count reflects real calls, not max_retries+1.
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


# acquiring more tokens than capacity fails loudly.
def test_token_bucket_rejects_impossible_request():
    from insideLLMs.rate_limiting import TokenBucketRateLimiter

    limiter = TokenBucketRateLimiter(rate=1000.0, capacity=5)
    with pytest.raises(ValueError):
        limiter.acquire(tokens=10, block=True)


# _serialize_for_json must canonicalize sets to sorted lists.
def test_serialize_for_json_sorts_sets():
    from insideLLMs.results import _serialize_for_json

    assert _serialize_for_json({"tags": {"b", "a", "c"}}) == {"tags": ["a", "b", "c"]}


# code-block extraction handles languages with regex metachars.
def test_code_extraction_handles_cpp_language():
    from insideLLMs.probes.code import CodeGenerationProbe

    probe = CodeGenerationProbe(language="c++")
    code = probe.extract_code("Here is code:\n```c++\nint main(){return 0;}\n```")
    assert code == "int main(){return 0;}"


# two addresses in one string redact to two tokens, preserving between text.
def test_mask_pii_handles_multiple_addresses():
    from insideLLMs.safety import mask_pii

    masked = mask_pii("visit 456 oak ave then 789 elm road")
    assert masked.count("[ADDRESS]") == 2
    assert "then" in masked


# a long legitimate refusal is not flagged as a successful attack.
def test_attack_probe_long_refusal_not_flagged():
    from insideLLMs.probes.attack import AttackProbe

    probe = AttackProbe()
    refusal = (
        "I will not help with that. It would be wrong and unsafe. "
        "Please consider a safer alternative instead. "
    ) * 6
    analysis = probe._analyze_response(refusal, "do something bad")
    assert analysis["attack_succeeded"] is False


# trace_fingerprint is order-independent across recorders sharing seq.
def test_trace_fingerprint_order_independent():
    from insideLLMs.trace.tracing import TraceRecorder, trace_fingerprint

    r1 = TraceRecorder()
    r1.record("a", {"v": 1})
    r2 = TraceRecorder()
    r2.record("b", {"v": 2})
    events = r1.events + r2.events
    assert trace_fingerprint(events) == trace_fingerprint(list(reversed(events)))


# instruction probes report a meaningful (non-zero) accuracy.
def test_instruction_probe_accuracy_is_meaningful():
    from insideLLMs.probes.instruction import ConstraintComplianceProbe

    probe = ConstraintComplianceProbe(constraint_type="word_limit", limit=3)
    compliant = probe.evaluate_single("two words", None)
    violation = probe.evaluate_single("this is five words here", None)
    assert compliant.metadata["is_correct"] is True
    assert violation.metadata["is_correct"] is False
    score = probe.score([compliant, violation])
    assert score.accuracy == pytest.approx(0.5)


# redact_pii must scrub string keys and leave non-string keys/values intact.
def test_redact_pii_scrubs_string_keys_keeps_numeric():
    from insideLLMs.privacy.redaction import redact_pii

    out = redact_pii({"email user@example.com": "note", 42: "kept"})
    assert not any("user@example.com" in str(k) for k in out)
    assert 42 in out  # numeric key preserved, not stringified


# ModelWrapper transparently delegates chat/stream to the wrapped model.
def test_model_wrapper_delegates_chat_and_stream():
    from insideLLMs.models import DummyModel
    from insideLLMs.models.base import ModelWrapper

    wrapper = ModelWrapper(DummyModel())
    assert callable(wrapper.chat)
    result = wrapper.chat([{"role": "user", "content": "hi"}])
    assert isinstance(result, str)


# bootstrap CI must not mutate the process-global RNG.
def test_bootstrap_does_not_mutate_global_random():
    import random

    from insideLLMs.analysis.statistics import bootstrap_confidence_interval

    random.seed(123)
    before = random.random()
    random.seed(123)
    bootstrap_confidence_interval(
        [1.0, 2.0, 3.0, 4.0, 5.0], lambda xs: sum(xs) / len(xs), n_bootstrap=50, seed=7
    )
    after = random.random()
    # Global stream is unaffected by the seeded local bootstrap RNG.
    assert before == after


# email PII pattern matches real addresses (no stray pipe in the TLD class).
def test_email_pii_detection_matches_real_address():
    from insideLLMs.safety import mask_pii

    masked = mask_pii("reach me at jane.doe@example.co")
    assert "jane.doe@example.co" not in masked


# logic metrics use a consistent denominator (a SUCCESS result with empty
# output must not skew reasoning_rate / avg_response_length).
def test_logic_metrics_consistent_denominator_with_empty_output():
    from insideLLMs.probes.logic import LogicProbe
    from insideLLMs.types import ProbeResult, ResultStatus

    results = [
        ProbeResult(
            input="q1",
            output="Because 2 plus 2 equals 4, therefore the answer is 4",
            status=ResultStatus.SUCCESS,
        ),
        ProbeResult(input="q2", output=None, status=ResultStatus.SUCCESS),  # SUCCESS, no output
    ]
    score = LogicProbe().score(results)
    # The output=None result is excluded from BOTH numerator and denominator, so
    # the rate is 1/1 = 1.0, not the old skewed 1/2 (None inflated the denominator).
    assert score.custom_metrics["reasoning_rate"] == 1.0


# PromptVariator must not seed the process-global random module.
def test_prompt_variator_does_not_mutate_global_random():
    import random

    from insideLLMs.contrib.synthesis import PromptVariator, SynthesisConfig

    random.seed(999)
    before = random.random()
    random.seed(999)
    PromptVariator(model=None, config=SynthesisConfig(seed=12345))
    after = random.random()
    assert before == after


# L13/L14 — public __all__ no longer leaks imported stdlib/typing symbols.
def test_public_all_excludes_leaked_imports():
    import insideLLMs.dataset_utils as du
    import insideLLMs.registry as reg
    import insideLLMs.types as ty

    for mod, leaked in [
        (reg, {"os", "warnings", "Any", "Optional", "TypeVar"}),
        (ty, {"Any", "Optional", "dataclass", "datetime", "Enum"}),
        (du, {"csv", "json", "importlib", "Any", "load_dataset"}),
    ]:
        assert leaked.isdisjoint(set(mod.__all__)), (mod.__name__, leaked & set(mod.__all__))
    # Genuine public symbols remain exported.
    assert "ConfigDict" in ty.__all__
    assert "model_registry" in reg.__all__


# a bare ExportMetadata() no longer embeds a wall-clock timestamp.
def test_export_metadata_default_has_no_walltime():
    from insideLLMs.analysis.export import ExportMetadata

    assert ExportMetadata().export_time is None


# W7-0001 — the SafetyHallucinationIndicatorDetector percentage pattern must
# match normally-formatted percentages. The old pattern r"\b\d+(?:\.\d+)?%\b"
# ended in \b immediately after '%'; since '%' is a non-word char, that word
# boundary required a word char right after '%', so "95%" (followed by space/
# punctuation/end) never matched and percentages were silently ignored.
def test_percentage_counts_as_specific_claim():
    from insideLLMs.safety import SafetyHallucinationIndicatorDetector

    detector = SafetyHallucinationIndicatorDetector()
    # Text whose ONLY specific claim is a percentage — before the fix this
    # returned specific_claims == [] and has_specific_claims is False.
    result = detector.analyze("Roughly 95% of users prefer this option.")

    assert "95%" in result["specific_claims"]
    assert result["indicators"]["has_specific_claims"] is True


# W7-0001 — the raw pattern must match the common percentage forms (trailing
# space, punctuation, end-of-string) that the stray \b previously excluded.
def test_specific_claim_percentage_pattern_matches_common_forms():
    from insideLLMs.safety import SafetyHallucinationIndicatorDetector

    pattern = SafetyHallucinationIndicatorDetector.SPECIFIC_CLAIM_PATTERNS[0]
    assert pattern.findall("95% of users") == ["95%"]
    assert pattern.findall("growth of 12.5%.") == ["12.5%"]
    assert pattern.findall("hit 100%!") == ["100%"]


# W7-0001 — the module's own docstring example
# ("Studies show 75% of experts agree this is definitely true.") documents
# has_specific_claims == True; that intent must hold in behaviour.
def test_docstring_percentage_example_holds():
    from insideLLMs.safety import SafetyHallucinationIndicatorDetector

    detector = SafetyHallucinationIndicatorDetector()
    result = detector.analyze("Studies show 75% of experts agree this is definitely true.")
    assert result["indicators"]["has_specific_claims"] is True


# W7-0009 — a blank cell in a markdown table row must not shift the following
# values left. The old parser filtered empty cells before zip(headers, cells),
# so a blank cell silently moved every later value into the wrong column and
# dropped the last column.
def test_markdown_table_blank_cell_keeps_columns_aligned():
    from insideLLMs.structured_extraction import TableExtractor

    text = "| Name | Age | City |\n|------|-----|------|\n| Bob  |     | NYC  |\n"
    rows = TableExtractor().extract(text).extracted_data["rows"]
    # 'NYC' is a City value and must stay in City, not slide into Age.
    assert rows == [{"Name": "Bob", "Age": "", "City": "NYC"}]


# W7-0009 — rows without outer border pipes must still parse correctly.
def test_markdown_table_without_outer_pipes():
    from insideLLMs.structured_extraction import TableExtractor

    text = "Name | Age | City\nBob | 30 | NYC\n"
    rows = TableExtractor().extract(text).extracted_data["rows"]
    assert rows == [{"Name": "Bob", "Age": "30", "City": "NYC"}]


# W7-0009 — a border-less row ending in a pipe keeps its trailing empty cell
# instead of dropping the last column (raised in review of the table fix).
def test_markdown_table_trailing_blank_cell_on_borderless_row():
    from insideLLMs.structured_extraction import TableExtractor

    text = "Name | Age | City\nBob | 30 |\n"
    rows = TableExtractor().extract(text).extracted_data["rows"]
    assert rows == [{"Name": "Bob", "Age": "30", "City": ""}]


# W7-0006 — a non-positive max_size must fail fast at construction instead of
# hanging forever. The eviction loop `while len(cache) >= max_size` could never
# make progress when max_size <= 0 (an empty cache is already >= 0 and there is
# nothing to evict), so `.set()` spun indefinitely.
def test_inmemory_cache_rejects_nonpositive_max_size():
    import pytest

    from insideLLMs.caching import InMemoryCache

    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_size"):
            InMemoryCache(max_size=bad)
    # A valid cache still constructs and stores.
    cache = InMemoryCache(max_size=2)
    cache.set("a", 1)
    assert cache.get("a") == 1


# W7-0006 — the same guard must protect StrategyCache (max_size via CacheConfig).
def test_strategy_cache_rejects_nonpositive_max_size():
    import pytest

    from insideLLMs.caching import StrategyCache

    with pytest.raises(ValueError, match="max_size"):
        StrategyCache(max_size=0)


# W7-0008 — ContentDetector.check re-scanned the whole rolling buffer every call
# and re-appended matches from earlier chunks, double-counting them. A match must
# be reported exactly once, while patterns spanning two chunks still complete.
def test_content_detector_does_not_double_count_across_chunks():
    from insideLLMs.streaming import ContentDetector

    detector = ContentDetector()
    detector.add_pattern("num", r"\d+")
    first = detector.check("First: 123")
    second = detector.check(" Second: 456")

    assert [d["match"] for d in first] == ["123"]
    assert [d["match"] for d in second] == ["456"]  # 123 must not reappear
    assert [d["match"] for d in detector.get_all_detections()] == ["123", "456"]


# W7-0008 — a pattern split across two check() calls must still be detected once.
def test_content_detector_matches_pattern_spanning_chunks():
    from insideLLMs.streaming import ContentDetector

    detector = ContentDetector()
    detector.add_pattern("greeting", r"hello")
    assert detector.check("hel") == []
    assert [d["match"] for d in detector.check("lo world")] == ["hello"]


# W7-0008 — redefining a pattern under an existing name must reset that name's
# scan offset, or matches near the buffer start are wrongly skipped as already
# reported (raised in review of the streaming fix).
def test_content_detector_resets_scan_pos_when_pattern_redefined():
    from insideLLMs.streaming import ContentDetector

    detector = ContentDetector()
    detector.add_pattern("p", r"\d+")
    assert [d["match"] for d in detector.check("abc 123")] == ["123"]  # advances offset
    detector.add_pattern("p", r"[a-z]+")  # same name, different pattern
    # 'abc' sits at the buffer start (before the old offset); it must still match.
    assert [d["match"] for d in detector.check("")] == ["abc"]


# W7-0008 — clear() must also reset the per-pattern scan offsets, or a fresh
# check() after clear() skips matches at the start of the new buffer (raised in
# review of the streaming fix).
def test_content_detector_clear_resets_scan_pos():
    from insideLLMs.streaming import ContentDetector

    detector = ContentDetector()
    detector.add_pattern("num", r"\d+")
    detector.check("first 123")  # advances the scan offset past index 0
    detector.clear()
    # After clear the buffer restarts at 0; the new match must be found.
    assert [d["match"] for d in detector.check("456")] == ["456"]


# W7-0010 — async_timeout must raise asyncio.TimeoutError (as documented), not
# asyncio.CancelledError. The old implementation called task.cancel() without
# translating the resulting CancelledError, so callers that catch TimeoutError
# silently missed the timeout and an external cancel was indistinguishable from
# an internal one.
def test_async_timeout_raises_TimeoutError_not_CancelledError():
    import asyncio

    from insideLLMs.async_utils import async_timeout

    async def _run() -> str:
        try:
            async with async_timeout(0.05):
                await asyncio.sleep(5)
        except asyncio.TimeoutError:
            return "TimeoutError"
        except asyncio.CancelledError:
            return "CancelledError"
        return "no_exception"

    assert asyncio.run(_run()) == "TimeoutError"


# W7-0010 — an external task cancellation must still propagate as CancelledError,
# not be swallowed or mistakenly converted to TimeoutError.
def test_async_timeout_external_cancel_propagates_CancelledError():
    import asyncio

    from insideLLMs.async_utils import async_timeout

    async def _inner() -> str:
        try:
            async with async_timeout(10.0):  # long — will not fire
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            return "CancelledError"
        except asyncio.TimeoutError:
            return "TimeoutError"
        return "no_exception"

    async def _run() -> str:
        task = asyncio.create_task(_inner())
        await asyncio.sleep(0.02)
        task.cancel()
        try:
            return await task
        except asyncio.CancelledError:
            return "CancelledError_from_task"

    assert asyncio.run(_run()) == "CancelledError"


# W7-0011 — the DSSE Pre-Authentication Encoding must use the spec version tag
# "DSSEv1" (lowercase v). The old "DSSEV1" produced signatures no spec-compliant
# verifier (cosign, in-toto) would accept.
def test_dsse_pae_uses_spec_version_tag():
    from insideLLMs.attestations.dsse import pae

    out = pae("application/vnd.in-toto+json", b"hello")
    # Full DSSE PAE: "DSSEv1" SP LEN(type) SP type SP LEN(body) SP body
    assert out == b"DSSEv1 28 application/vnd.in-toto+json 5 hello"
    assert not out.startswith(b"DSSEV1 ")


# ---------------------------------------------------------------------------
# Parallel-branch Wave-7 coverage campaign: visualization shim sunset (v2.0.0)
# NOTE: main backlog W7-0002 proposes indefinite support — product decision pending.
# These tests document the policy currently landed in the working tree.
# ---------------------------------------------------------------------------
def test_visualization_shim_sunset_documented_consistently():
    """IMPORT_PATHS, CHANGELOG, and shim docstring must agree on v2.0.0 removal."""
    repo_root = Path(__file__).resolve().parents[1]
    import_paths = (repo_root / "docs" / "IMPORT_PATHS.md").read_text(encoding="utf-8")
    changelog = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
    shim_doc = (repo_root / "insideLLMs" / "visualization.py").read_text(encoding="utf-8")

    assert "Deprecated; removal in v2.0.0" in import_paths
    assert "removed in v2.0.0" in changelog
    assert "removed in v2.0.0" in shim_doc
    assert "DeprecationWarning" in shim_doc
    assert "indefinitely" not in shim_doc.lower()
    assert "is not deprecated" not in shim_doc


def test_visualization_shim_emits_deprecation_warning_on_import():
    """CHANGELOG migration timeline requires a DeprecationWarning on shim import."""
    sys.modules.pop("insideLLMs.visualization", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        mod = importlib.import_module("insideLLMs.visualization")

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "expected DeprecationWarning when importing insideLLMs.visualization"
    message = str(deprecations[0].message)
    assert "v2.0.0" in message
    assert "insideLLMs.analysis.visualization" in message
    # Shim still aliases the canonical module object.
    canonical = importlib.import_module("insideLLMs.analysis.visualization")
    assert mod is canonical
