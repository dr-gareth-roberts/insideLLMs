from insideLLMs.contrib.retrieval import Document, RetrievalResult
from insideLLMs.inference.retrieval import rerank_and_assemble


async def test_retrieval_reranks_deduplicates_and_places_best_evidence_at_boundaries() -> None:
    broad = [
        RetrievalResult(Document("weak", id="weak"), score=0.9, rank=1),
        RetrievalResult(Document("best evidence", id="best"), score=0.2, rank=2),
        RetrievalResult(Document("second evidence", id="second"), score=0.1, rank=3),
        RetrievalResult(Document("best duplicate", id="best"), score=0.0, rank=4),
    ]

    scores = {"weak": 0.1, "best": 1.0, "second": 0.8}
    result = await rerank_and_assemble(
        "query",
        broad,
        rerank=lambda query, document: scores[document.id],
        top_k=3,
        max_characters=80,
    )

    assert result.source_ids == ("best", "weak", "second")
    assert result.context.startswith("[best] best evidence")
    assert result.context.endswith("[second] second evidence")
    assert result.reranked_scores == {"best": 1.0, "second": 0.8, "weak": 0.1}


async def test_retrieval_enforces_token_budget_and_source_diversity() -> None:
    broad = [
        RetrievalResult(Document("one two", id="a", metadata={"source": "same"}), 0.9),
        RetrievalResult(Document("three four", id="b", metadata={"source": "same"}), 0.8),
        RetrievalResult(Document("five six", id="c", metadata={"source": "other"}), 0.7),
    ]

    result = await rerank_and_assemble(
        "query",
        broad,
        rerank=lambda query, document: {"a": 0.9, "b": 0.8, "c": 0.7}[document.id],
        top_k=2,
        max_tokens=6,
        token_count=lambda text: len(text.split()),
        diversity_key=lambda document: str(document.metadata["source"]),
        max_per_diversity_group=1,
    )

    assert result.source_ids == ("a", "c")
    assert result.used_tokens == 6


async def test_retrieval_token_budget_covers_rendered_separators() -> None:
    broad = [
        RetrievalResult(Document("x", id="a"), 1.0),
        RetrievalResult(Document("y", id="b"), 0.9),
    ]

    result = await rerank_and_assemble(
        "query",
        broad,
        rerank=lambda query, document: {"a": 1.0, "b": 0.9}[document.id],
        top_k=2,
        max_tokens=12,
        token_count=len,
    )

    assert result.context == "[a] x\n\n[b] y"
    assert result.used_tokens == len(result.context) == 12
