"""Canonical prompt composition for exact provider/KV prefix caching."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .schemas import PromptParts


@dataclass(frozen=True)
class CachedPrompt:
    text: str
    stable_prefix: str
    cache_key: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class PrefixCacheTelemetry:
    cache_key: str
    input_tokens: int
    cached_tokens: int
    cache_hit: bool
    cached_token_ratio: float


def _canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compose_cached_prompt(
    parts: PromptParts,
    *,
    tools: dict[str, Any] | None = None,
    schemas: dict[str, Any] | None = None,
    tenant_id: str,
    model_id: str = "",
) -> CachedPrompt:
    """Put immutable sections first and scope exact-cache identity by tenant.

    Pass ``model_id`` (model name plus any generation parameters that change
    outputs, e.g. ``"gpt-4o-mini:temp=0"``) whenever the cache_key feeds a
    shared KV/response cache: without it, different models with the same stable
    prefix collide on one key.
    """

    if not tenant_id.strip() or "\0" in tenant_id:
        raise ValueError("tenant_id must be non-empty and contain no NUL characters")
    stable_sections = list(parts.stable)
    if schemas:
        stable_sections.append(f"Schemas: {_canonical(schemas)}")
    if tools:
        stable_sections.append(f"Tools: {_canonical(tools)}")
    stable_prefix = "\n\n".join(stable_sections)
    # Skip an empty stable prefix so the composed text matches
    # PromptParts.compose() for the same parts (no spurious leading separator).
    sections = (
        (stable_prefix, *parts.dynamic, *parts.evidence)
        if stable_prefix
        else (
            *parts.dynamic,
            *parts.evidence,
        )
    )
    text = "\n\n".join(sections)
    cache_identity = json.dumps(
        [tenant_id, model_id, stable_prefix], separators=(",", ":"), ensure_ascii=False
    ).encode()
    digest = hashlib.sha256(cache_identity).hexdigest()
    metadata = {"cache_control": "exact-prefix", "tenant_id": tenant_id}
    if model_id:
        metadata["model_id"] = model_id
    return CachedPrompt(
        text=text,
        stable_prefix=stable_prefix,
        cache_key=digest,
        metadata=metadata,
    )


def record_cache_usage(
    prompt: CachedPrompt, *, input_tokens: int, cached_tokens: int
) -> PrefixCacheTelemetry:
    """Normalize provider cache counters into portable telemetry."""

    if input_tokens < 0 or not 0 <= cached_tokens <= input_tokens:
        raise ValueError("cached_tokens must be between zero and input_tokens")
    return PrefixCacheTelemetry(
        cache_key=prompt.cache_key,
        input_tokens=input_tokens,
        cached_tokens=cached_tokens,
        cache_hit=cached_tokens > 0,
        cached_token_ratio=cached_tokens / input_tokens if input_tokens else 0.0,
    )
