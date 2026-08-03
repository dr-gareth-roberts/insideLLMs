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
        model_id="m1",
    )
    reordered = compose_cached_prompt(
        parts,
        tools={"alpha": {"type": "string"}, "zeta": {"type": "object"}},
        schemas={"response": {"required": ["answer"]}},
        tenant_id="tenant-a",
        model_id="m1",
    )
    other_tenant = compose_cached_prompt(
        parts,
        tools={"alpha": {"type": "string"}, "zeta": {"type": "object"}},
        schemas={"response": {"required": ["answer"]}},
        tenant_id="tenant-b",
        model_id="m1",
    )

    assert first.text == reordered.text
    assert first.cache_key == reordered.cache_key
    assert first.cache_key != other_tenant.cache_key
    assert first.stable_prefix.endswith('"zeta":{"type":"object"}}')
    assert first.metadata == {
        "cache_control": "exact-prefix",
        "tenant_id": "tenant-a",
        "model_id": "m1",
    }

    telemetry = record_cache_usage(first, input_tokens=100, cached_tokens=75)
    assert telemetry.cache_hit is True
    assert telemetry.cached_token_ratio == 0.75
    assert telemetry.cache_key == first.cache_key


def test_prefix_cache_key_uses_unambiguous_tenant_encoding() -> None:
    first = compose_cached_prompt(PromptParts(stable=("\0b",)), tenant_id="a", model_id="m1")
    assert first.cache_key
    # Field boundaries must not collapse: moving characters between the tenant
    # and the stable prefix has to change the key.
    shifted = compose_cached_prompt(PromptParts(stable=("b",)), tenant_id="a", model_id="m1")
    assert first.cache_key != shifted.cache_key
    # model_id participates in cache identity, so two models cannot collide.
    scoped = compose_cached_prompt(
        PromptParts(stable=("b",)), tenant_id="a", model_id="gpt-4o-mini:temp=0"
    )
    other_model = compose_cached_prompt(
        PromptParts(stable=("b",)), tenant_id="a", model_id="other:temp=1"
    )
    assert scoped.cache_key != shifted.cache_key
    assert scoped.cache_key != other_model.cache_key
    assert scoped.metadata["model_id"] == "gpt-4o-mini:temp=0"

    with pytest.raises(ValueError, match="tenant_id"):
        compose_cached_prompt(PromptParts(stable=("system",)), tenant_id="", model_id="m1")
    with pytest.raises(ValueError, match="tenant_id"):
        compose_cached_prompt(PromptParts(stable=("b",)), tenant_id="a\0", model_id="m1")


def test_model_id_is_required_and_prevents_cross_model_collision() -> None:
    """model_id must be supplied; omitting it previously collided silently.

    Regression: model_id defaulted to "", so two different models sharing a
    stable prefix and tenant produced byte-identical cache keys. A shared
    KV/response cache would then serve one model's completion for another
    model's request. The parameter is now required — there is no default that
    cannot be silently wrong.
    """
    parts = PromptParts(stable=("system",), dynamic=("q",))

    with pytest.raises(TypeError):
        compose_cached_prompt(parts, tenant_id="t")  # type: ignore[call-arg]

    for bad in ("", "   ", "m\0id"):
        with pytest.raises(ValueError, match="model_id"):
            compose_cached_prompt(parts, tenant_id="t", model_id=bad)

    a = compose_cached_prompt(parts, tenant_id="t", model_id="gpt-4o-mini:temp=0")
    b = compose_cached_prompt(parts, tenant_id="t", model_id="claude:temp=1")
    assert a.cache_key != b.cache_key
    assert a.metadata["model_id"] == "gpt-4o-mini:temp=0"
