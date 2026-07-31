import pytest

from insideLLMs.inference import PromptParts
from insideLLMs.inference.prefix_cache import compose_cached_prompt, record_cache_usage


def test_prefix_composition_is_canonical_and_tenant_isolated() -> None:
    parts = PromptParts(stable=("system",), dynamic=("question",), evidence=("source",))

    first = compose_cached_prompt(
        parts,
        tools={"zeta": {"type": "object"}, "alpha": {"type": "string"}},
        schemas={"response": {"required": ["answer"]}},
        tenant_id="tenant-a",
    )
    reordered = compose_cached_prompt(
        parts,
        tools={"alpha": {"type": "string"}, "zeta": {"type": "object"}},
        schemas={"response": {"required": ["answer"]}},
        tenant_id="tenant-a",
    )
    other_tenant = compose_cached_prompt(
        parts,
        tools={"alpha": {"type": "string"}, "zeta": {"type": "object"}},
        schemas={"response": {"required": ["answer"]}},
        tenant_id="tenant-b",
    )

    assert first.text == reordered.text
    assert first.cache_key == reordered.cache_key
    assert first.cache_key != other_tenant.cache_key
    assert first.stable_prefix.endswith('"zeta":{"type":"object"}}')
    assert first.metadata == {"cache_control": "exact-prefix", "tenant_id": "tenant-a"}

    telemetry = record_cache_usage(first, input_tokens=100, cached_tokens=75)
    assert telemetry.cache_hit is True
    assert telemetry.cached_token_ratio == 0.75
    assert telemetry.cache_key == first.cache_key


def test_prefix_cache_key_uses_unambiguous_tenant_encoding() -> None:
    first = compose_cached_prompt(PromptParts(stable=("\0b",)), tenant_id="a")
    assert first.cache_key

    with pytest.raises(ValueError, match="tenant_id"):
        compose_cached_prompt(PromptParts(stable=("system",)), tenant_id="")
    with pytest.raises(ValueError, match="tenant_id"):
        compose_cached_prompt(PromptParts(stable=("b",)), tenant_id="a\0")
