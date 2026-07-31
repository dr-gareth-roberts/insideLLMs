"""Broad retrieval followed by query-aware reranking and context assembly."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Awaitable

from insideLLMs.contrib.retrieval import RetrievalResult

from ._callbacks import resolve


@dataclass(frozen=True)
class AssembledEvidence:
    context: str
    source_ids: tuple[str, ...]
    reranked_scores: dict[str, float]
    used_tokens: int = 0


async def rerank_and_assemble(
    query: str,
    broad_results: Sequence[RetrievalResult],
    *,
    rerank: Callable[[str, Any], float | Awaitable[float]],
    top_k: int,
    max_characters: int | None = None,
    max_tokens: int | None = None,
    token_count: Callable[[str], int] = lambda text: len(text.split()),
    diversity_key: Callable[[Any], str] | None = None,
    max_per_diversity_group: int | None = None,
) -> AssembledEvidence:
    """Dedupe, rerank, budget, then move the runner-up to the far boundary."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if max_characters is None and max_tokens is None:
        raise ValueError("a character or token budget is required")
    if max_characters is not None and max_characters < 1:
        raise ValueError("max_characters must be positive")
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if max_per_diversity_group is not None and max_per_diversity_group < 1:
        raise ValueError("max_per_diversity_group must be positive")
    unique: dict[str, Any] = {}
    for result in broad_results:
        source_id = str(result.document.id)
        unique.setdefault(source_id, result.document)

    scored = [
        (source_id, document, await resolve(rerank(query, document)))
        for source_id, document in unique.items()
    ]
    scored.sort(key=lambda item: (-item[2], item[0]))

    selected: list[tuple[str, Any, float]] = []
    used_characters = 0
    used_tokens = 0
    group_counts: dict[str, int] = {}
    for item in scored:
        if len(selected) >= top_k:
            break
        rendered = f"[{item[0]}] {item[1].content}"
        extra_characters = len(rendered) + (2 if selected else 0)
        extra_tokens = token_count(rendered) + (token_count("\n\n") if selected else 0)
        group = diversity_key(item[1]) if diversity_key is not None else item[0]
        if (
            max_per_diversity_group is not None
            and group_counts.get(group, 0) >= max_per_diversity_group
        ):
            continue
        if max_characters is not None and used_characters + extra_characters > max_characters:
            continue
        if max_tokens is not None and used_tokens + extra_tokens > max_tokens:
            continue
        selected.append(item)
        used_characters += extra_characters
        used_tokens += extra_tokens
        group_counts[group] = group_counts.get(group, 0) + 1

    if len(selected) > 2:
        ordered = [selected[0], *selected[2:], selected[1]]
    else:
        ordered = selected
    return AssembledEvidence(
        context="\n\n".join(
            f"[{source_id}] {document.content}" for source_id, document, _ in ordered
        ),
        source_ids=tuple(source_id for source_id, _, _ in ordered),
        reranked_scores={source_id: score for source_id, _, score in scored},
        used_tokens=used_tokens,
    )
