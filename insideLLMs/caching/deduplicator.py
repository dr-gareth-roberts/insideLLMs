"""
Response Deduplicator
=====================

Identifies and deduplicates identical or similar responses.
"""

from __future__ import annotations

from typing import Any, Optional


class ResponseDeduplicator:
    """Identifies and deduplicates identical or similar responses.

    ResponseDeduplicator tracks responses and detects duplicates based on
    exact matching or configurable similarity thresholds. Useful for
    identifying redundant API calls or response patterns.

    Parameters
    ----------
    similarity_threshold : float
        Threshold for considering responses as duplicates. Range 0.0 to 1.0.
        1.0 means exact match only, lower values allow fuzzy matching.
        Default is 1.0 (exact match).

    Attributes
    ----------
    similarity_threshold : float
        The configured similarity threshold.
    _responses : list[tuple[str, str, Any]]
        List of (prompt, response, metadata) tuples for unique responses.

    Examples
    --------
    Exact match deduplication:

        >>> from insideLLMs.caching import ResponseDeduplicator
        >>> dedup = ResponseDeduplicator()  # threshold=1.0 (exact)
        >>> is_dup, idx = dedup.add("p1", "Hello, World!")
        >>> print(f"Duplicate: {is_dup}")
        Duplicate: False
        >>> is_dup, idx = dedup.add("p2", "Hello, World!")
        >>> print(f"Duplicate: {is_dup}, matches index: {idx}")
        Duplicate: True, matches index: 0

    Similarity-based deduplication:

        >>> dedup = ResponseDeduplicator(similarity_threshold=0.8)
        >>> dedup.add("p1", "The quick brown fox jumps over the lazy dog")
        >>> is_dup, idx = dedup.add("p2", "The quick brown fox jumps over a lazy dog")
        >>> print(f"Similar enough: {is_dup}")  # True - high word overlap
        Similar enough: True

    Adding metadata:

        >>> dedup = ResponseDeduplicator()
        >>> dedup.add("prompt", "response", metadata={"model": "gpt-4", "tokens": 50})
        >>> unique = dedup.get_unique_responses()
        >>> for prompt, response, meta in unique:
        ...     print(f"Model: {meta['model']}")

    Getting unique responses:

        >>> dedup = ResponseDeduplicator()
        >>> dedup.add("p1", "Response A")
        >>> dedup.add("p2", "Response B")
        >>> dedup.add("p3", "Response A")  # Duplicate
        >>> unique = dedup.get_unique_responses()
        >>> print(f"Unique count: {len(unique)}")
        Unique count: 2

    Clearing stored responses:

        >>> dedup = ResponseDeduplicator()
        >>> for i in range(100):
        ...     dedup.add(f"p{i}", f"Response {i}")
        >>> print(len(dedup.get_unique_responses()))
        100
        >>> dedup.clear()
        >>> print(len(dedup.get_unique_responses()))
        0

    Use case - detecting redundant API calls:

        >>> dedup = ResponseDeduplicator(similarity_threshold=0.95)
        >>> responses = ["Answer A", "Answer A", "Answer B", "Answer A again"]
        >>> for i, resp in enumerate(responses):
        ...     is_dup, _ = dedup.add(f"prompt_{i}", resp)
        ...     if is_dup:
        ...         print(f"Response {i} is redundant")

    Notes
    -----
    - Similarity is calculated using Jaccard index (word overlap).
    - For exact matching (threshold=1.0), uses simple string comparison.
    - Memory usage grows linearly with unique responses.

    See Also
    --------
    PromptCache.find_similar : Similar functionality in cache context.
    """

    def __init__(self, similarity_threshold: float = 1.0):
        self.similarity_threshold = similarity_threshold
        self._responses: list[tuple[str, str, Any]] = []
        self._duplicate_count = 0

    def add(
        self,
        prompt: str,
        response: str,
        metadata: Optional[Any] = None,
    ) -> tuple[bool, Optional[int]]:
        """Add response, returning whether it's a duplicate."""
        for i, (_, existing_response, _) in enumerate(self._responses):
            if self._is_duplicate(response, existing_response):
                self._duplicate_count += 1
                return True, i

        self._responses.append((prompt, response, metadata))
        return False, None

    def _is_duplicate(self, response1: str, response2: str) -> bool:
        """Check if two responses are duplicates."""
        if self.similarity_threshold == 1.0:
            return response1 == response2

        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)
        similarity = intersection / union if union > 0 else 0.0

        return similarity >= self.similarity_threshold

    def get_unique_responses(self) -> list[tuple[str, str, Any]]:
        """Get all unique responses."""
        return self._responses.copy()

    def get_duplicate_count(self) -> int:
        """Get count of duplicates found."""
        return self._duplicate_count

    def clear(self):
        """Clear stored responses."""
        self._responses.clear()
        self._duplicate_count = 0


__all__ = ["ResponseDeduplicator"]
