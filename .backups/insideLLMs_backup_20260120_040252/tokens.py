"""
Token analysis and embedding utilities for LLM exploration.

Provides tools for:
- Token counting and estimation
- Token distribution analysis
- Vocabulary coverage analysis
- Token-level statistics
- Embedding utilities and similarity
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Optional,
    Protocol,
)


class TokenizerType(Enum):
    """Types of tokenizers for estimation."""

    GPT4 = "gpt4"
    GPT3 = "gpt3"
    CLAUDE = "claude"
    LLAMA = "llama"
    SIMPLE = "simple"  # Whitespace-based


@dataclass
class TokenStats:
    """Statistics about tokens in text."""

    total_tokens: int
    unique_tokens: int
    char_count: int
    word_count: int
    avg_token_length: float
    tokens_per_word: float
    token_frequencies: dict[str, int] = field(default_factory=dict)

    @property
    def token_diversity(self) -> float:
        """Ratio of unique tokens to total tokens."""
        return self.unique_tokens / self.total_tokens if self.total_tokens > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tokens": self.total_tokens,
            "unique_tokens": self.unique_tokens,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "avg_token_length": self.avg_token_length,
            "tokens_per_word": self.tokens_per_word,
            "token_diversity": self.token_diversity,
        }


@dataclass
class TokenDistribution:
    """Token frequency distribution analysis."""

    frequencies: dict[str, int]
    total_tokens: int

    @property
    def vocabulary_size(self) -> int:
        """Number of unique tokens."""
        return len(self.frequencies)

    def top_tokens(self, n: int = 10) -> list[tuple[str, int]]:
        """Get top N most frequent tokens."""
        return sorted(self.frequencies.items(), key=lambda x: x[1], reverse=True)[:n]

    def bottom_tokens(self, n: int = 10) -> list[tuple[str, int]]:
        """Get bottom N least frequent tokens."""
        return sorted(self.frequencies.items(), key=lambda x: x[1])[:n]

    def frequency_of(self, token: str) -> int:
        """Get frequency of a specific token."""
        return self.frequencies.get(token, 0)

    def relative_frequency(self, token: str) -> float:
        """Get relative frequency of a token."""
        if self.total_tokens == 0:
            return 0.0
        return self.frequencies.get(token, 0) / self.total_tokens

    def tokens_above_frequency(self, min_freq: int) -> list[str]:
        """Get tokens with frequency above threshold."""
        return [t for t, f in self.frequencies.items() if f >= min_freq]

    def hapax_legomena(self) -> list[str]:
        """Get tokens that appear exactly once."""
        return [t for t, f in self.frequencies.items() if f == 1]

    def entropy(self) -> float:
        """Calculate Shannon entropy of token distribution."""
        if self.total_tokens == 0:
            return 0.0

        entropy = 0.0
        for count in self.frequencies.values():
            if count > 0:
                p = count / self.total_tokens
                entropy -= p * math.log2(p)
        return entropy


@dataclass
class VocabCoverage:
    """Analysis of vocabulary coverage."""

    text_vocab: set[str]
    reference_vocab: set[str]
    covered: set[str]
    uncovered: set[str]

    @property
    def coverage_ratio(self) -> float:
        """Ratio of text vocab covered by reference vocab."""
        if not self.text_vocab:
            return 1.0
        return len(self.covered) / len(self.text_vocab)

    @property
    def oov_ratio(self) -> float:
        """Out-of-vocabulary ratio."""
        return 1.0 - self.coverage_ratio

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text_vocab_size": len(self.text_vocab),
            "reference_vocab_size": len(self.reference_vocab),
            "covered_count": len(self.covered),
            "uncovered_count": len(self.uncovered),
            "coverage_ratio": self.coverage_ratio,
            "oov_ratio": self.oov_ratio,
        }


class Tokenizer(Protocol):
    """Protocol for tokenizer implementations."""

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        ...

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text to token strings."""
        ...


class SimpleTokenizer:
    """Simple whitespace and punctuation-based tokenizer."""

    def __init__(self, lowercase: bool = True):
        """Initialize tokenizer.

        Args:
            lowercase: Whether to lowercase tokens.
        """
        self.lowercase = lowercase
        self._vocab: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._next_id = 0

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text to tokens.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens.
        """
        if self.lowercase:
            text = text.lower()

        # Split on whitespace and punctuation
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode.

        Returns:
            List of token IDs.
        """
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = self._next_id
                self._id_to_token[self._next_id] = token
                self._next_id += 1
            ids.append(self._vocab[token])
        return ids

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded text.
        """
        return " ".join(self._id_to_token.get(t, "<UNK>") for t in tokens)

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size."""
        return len(self._vocab)


class TokenEstimator:
    """Estimate token counts for different models."""

    # Average characters per token for different models
    CHARS_PER_TOKEN = {
        TokenizerType.GPT4: 4.0,
        TokenizerType.GPT3: 4.0,
        TokenizerType.CLAUDE: 3.5,
        TokenizerType.LLAMA: 3.8,
        TokenizerType.SIMPLE: 5.0,
    }

    # Adjustment factors for different content types
    CONTENT_ADJUSTMENTS = {
        "code": 0.7,  # Code tends to have more tokens
        "prose": 1.0,
        "technical": 0.85,
        "chat": 1.1,  # Conversational text has fewer tokens
    }

    def __init__(self, tokenizer_type: TokenizerType = TokenizerType.GPT4):
        """Initialize estimator.

        Args:
            tokenizer_type: Type of tokenizer to estimate for.
        """
        self.tokenizer_type = tokenizer_type
        self.chars_per_token = self.CHARS_PER_TOKEN.get(tokenizer_type, 4.0)

    def estimate_tokens(self, text: str, content_type: str = "prose") -> int:
        """Estimate token count for text.

        Args:
            text: Text to estimate.
            content_type: Type of content (code, prose, technical, chat).

        Returns:
            Estimated token count.
        """
        char_count = len(text)
        adjustment = self.CONTENT_ADJUSTMENTS.get(content_type, 1.0)
        estimated = char_count / (self.chars_per_token * adjustment)
        return max(1, round(estimated))

    def estimate_cost(
        self,
        text: str,
        price_per_1k_tokens: float,
        content_type: str = "prose",
    ) -> float:
        """Estimate cost for processing text.

        Args:
            text: Text to process.
            price_per_1k_tokens: Price per 1000 tokens.
            content_type: Type of content.

        Returns:
            Estimated cost.
        """
        tokens = self.estimate_tokens(text, content_type)
        return (tokens / 1000) * price_per_1k_tokens

    def fits_context(
        self,
        text: str,
        context_limit: int,
        content_type: str = "prose",
    ) -> bool:
        """Check if text fits within context limit.

        Args:
            text: Text to check.
            context_limit: Maximum tokens allowed.
            content_type: Type of content.

        Returns:
            True if text fits within limit.
        """
        return self.estimate_tokens(text, content_type) <= context_limit

    def split_to_chunks(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int = 0,
        content_type: str = "prose",
    ) -> list[str]:
        """Split text into chunks that fit token limit.

        Args:
            text: Text to split.
            max_tokens: Maximum tokens per chunk.
            overlap_tokens: Number of overlapping tokens.
            content_type: Type of content.

        Returns:
            List of text chunks.
        """
        # Calculate approximate characters per chunk
        adjustment = self.CONTENT_ADJUSTMENTS.get(content_type, 1.0)
        chars_per_chunk = int(max_tokens * self.chars_per_token * adjustment)
        overlap_chars = int(overlap_tokens * self.chars_per_token * adjustment)

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chars_per_chunk, text_len)

            # Try to break at sentence or word boundary
            if end < text_len:
                # Look for sentence boundary
                for boundary in [". ", "! ", "? ", "\n"]:
                    boundary_pos = text.rfind(boundary, start, end)
                    if boundary_pos > start:
                        end = boundary_pos + len(boundary)
                        break
                else:
                    # Look for word boundary
                    space_pos = text.rfind(" ", start, end)
                    if space_pos > start:
                        end = space_pos + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap_chars if overlap_chars > 0 else end

        return chunks


class TokenAnalyzer:
    """Comprehensive token analysis utilities."""

    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        """Initialize analyzer.

        Args:
            tokenizer: Optional tokenizer to use.
        """
        self.tokenizer = tokenizer or SimpleTokenizer()

    def analyze(self, text: str) -> TokenStats:
        """Analyze tokens in text.

        Args:
            text: Text to analyze.

        Returns:
            Token statistics.
        """
        tokens = self.tokenizer.tokenize(text)
        word_count = len(text.split())

        frequencies = Counter(tokens)
        total_tokens = len(tokens)
        unique_tokens = len(frequencies)

        total_length = sum(len(t) for t in tokens)
        avg_length = total_length / total_tokens if total_tokens > 0 else 0

        return TokenStats(
            total_tokens=total_tokens,
            unique_tokens=unique_tokens,
            char_count=len(text),
            word_count=word_count,
            avg_token_length=avg_length,
            tokens_per_word=total_tokens / word_count if word_count > 0 else 0,
            token_frequencies=dict(frequencies),
        )

    def get_distribution(self, text: str) -> TokenDistribution:
        """Get token frequency distribution.

        Args:
            text: Text to analyze.

        Returns:
            Token distribution.
        """
        tokens = self.tokenizer.tokenize(text)
        frequencies = Counter(tokens)
        return TokenDistribution(
            frequencies=dict(frequencies),
            total_tokens=len(tokens),
        )

    def compare_distributions(self, text1: str, text2: str) -> dict[str, Any]:
        """Compare token distributions between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Comparison results.
        """
        dist1 = self.get_distribution(text1)
        dist2 = self.get_distribution(text2)

        vocab1 = set(dist1.frequencies.keys())
        vocab2 = set(dist2.frequencies.keys())

        shared = vocab1 & vocab2
        only_in_1 = vocab1 - vocab2
        only_in_2 = vocab2 - vocab1

        # Calculate cosine similarity of distributions
        common_tokens = shared
        if common_tokens:
            dot_product = sum(dist1.frequencies[t] * dist2.frequencies[t] for t in common_tokens)
            mag1 = math.sqrt(sum(f**2 for f in dist1.frequencies.values()))
            mag2 = math.sqrt(sum(f**2 for f in dist2.frequencies.values()))
            cosine_sim = dot_product / (mag1 * mag2) if mag1 * mag2 > 0 else 0
        else:
            cosine_sim = 0.0

        return {
            "text1_vocab_size": len(vocab1),
            "text2_vocab_size": len(vocab2),
            "shared_vocab_size": len(shared),
            "only_in_text1": len(only_in_1),
            "only_in_text2": len(only_in_2),
            "jaccard_similarity": len(shared) / len(vocab1 | vocab2) if vocab1 | vocab2 else 0,
            "cosine_similarity": cosine_sim,
            "entropy_text1": dist1.entropy(),
            "entropy_text2": dist2.entropy(),
        }

    def analyze_vocabulary_coverage(self, text: str, reference_vocab: set[str]) -> VocabCoverage:
        """Analyze vocabulary coverage against reference.

        Args:
            text: Text to analyze.
            reference_vocab: Reference vocabulary set.

        Returns:
            Vocabulary coverage analysis.
        """
        tokens = self.tokenizer.tokenize(text)
        text_vocab = set(tokens)

        covered = text_vocab & reference_vocab
        uncovered = text_vocab - reference_vocab

        return VocabCoverage(
            text_vocab=text_vocab,
            reference_vocab=reference_vocab,
            covered=covered,
            uncovered=uncovered,
        )


@dataclass
class EmbeddingSimilarity:
    """Results of embedding similarity comparison."""

    similarity: float
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


class EmbeddingUtils:
    """Utilities for working with text embeddings."""

    @staticmethod
    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Cosine similarity (-1 to 1).
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same dimension")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a**2 for a in vec1))
        mag2 = math.sqrt(sum(b**2 for b in vec2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    @staticmethod
    def euclidean_distance(vec1: list[float], vec2: list[float]) -> float:
        """Calculate Euclidean distance between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Euclidean distance.
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same dimension")

        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

    @staticmethod
    def manhattan_distance(vec1: list[float], vec2: list[float]) -> float:
        """Calculate Manhattan distance between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Manhattan distance.
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same dimension")

        return sum(abs(a - b) for a, b in zip(vec1, vec2))

    @staticmethod
    def normalize(vec: list[float]) -> list[float]:
        """Normalize vector to unit length.

        Args:
            vec: Vector to normalize.

        Returns:
            Normalized vector.
        """
        magnitude = math.sqrt(sum(v**2 for v in vec))
        if magnitude == 0:
            return vec
        return [v / magnitude for v in vec]

    @staticmethod
    def average_embeddings(embeddings: list[list[float]]) -> list[float]:
        """Calculate average of multiple embeddings.

        Args:
            embeddings: List of embedding vectors.

        Returns:
            Average embedding.
        """
        if not embeddings:
            return []

        dim = len(embeddings[0])
        avg = [0.0] * dim

        for emb in embeddings:
            for i, v in enumerate(emb):
                avg[i] += v

        n = len(embeddings)
        return [v / n for v in avg]

    @staticmethod
    def find_most_similar(
        query: list[float],
        candidates: list[tuple[str, list[float]]],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Find most similar embeddings to query.

        Args:
            query: Query embedding.
            candidates: List of (id, embedding) tuples.
            top_k: Number of top results to return.

        Returns:
            List of (id, similarity) tuples.
        """
        similarities = []
        for id_, embedding in candidates:
            sim = EmbeddingUtils.cosine_similarity(query, embedding)
            similarities.append((id_, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


@dataclass
class TokenBudget:
    """Track token budget for operations."""

    total_budget: int
    used_tokens: int = 0
    reserved_tokens: int = 0

    @property
    def available(self) -> int:
        """Available tokens."""
        return self.total_budget - self.used_tokens - self.reserved_tokens

    @property
    def utilization(self) -> float:
        """Budget utilization ratio."""
        return self.used_tokens / self.total_budget if self.total_budget > 0 else 0

    def can_afford(self, tokens: int) -> bool:
        """Check if budget allows for tokens."""
        return tokens <= self.available

    def spend(self, tokens: int) -> bool:
        """Spend tokens from budget.

        Args:
            tokens: Tokens to spend.

        Returns:
            True if successful, False if over budget.
        """
        if not self.can_afford(tokens):
            return False
        self.used_tokens += tokens
        return True

    def reserve(self, tokens: int) -> bool:
        """Reserve tokens for future use.

        Args:
            tokens: Tokens to reserve.

        Returns:
            True if successful.
        """
        if tokens > self.available:
            return False
        self.reserved_tokens += tokens
        return True

    def release_reservation(self, tokens: int) -> None:
        """Release reserved tokens."""
        self.reserved_tokens = max(0, self.reserved_tokens - tokens)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_budget": self.total_budget,
            "used_tokens": self.used_tokens,
            "reserved_tokens": self.reserved_tokens,
            "available": self.available,
            "utilization": self.utilization,
        }


class ContextWindowManager:
    """Manage content within context window limits."""

    def __init__(
        self,
        max_tokens: int,
        estimator: Optional[TokenEstimator] = None,
    ):
        """Initialize manager.

        Args:
            max_tokens: Maximum context window tokens.
            estimator: Token estimator to use.
        """
        self.max_tokens = max_tokens
        self.estimator = estimator or TokenEstimator()
        self.budget = TokenBudget(total_budget=max_tokens)

    def add_content(
        self,
        content: str,
        priority: int = 0,
        content_type: str = "prose",
    ) -> bool:
        """Try to add content to context.

        Args:
            content: Content to add.
            priority: Content priority (higher = more important).
            content_type: Type of content.

        Returns:
            True if content fits, False otherwise.
        """
        tokens = self.estimator.estimate_tokens(content, content_type)
        return self.budget.spend(tokens)

    def remaining_tokens(self) -> int:
        """Get remaining token budget."""
        return self.budget.available

    def fits(self, content: str, content_type: str = "prose") -> bool:
        """Check if content fits in remaining budget.

        Args:
            content: Content to check.
            content_type: Type of content.

        Returns:
            True if content fits.
        """
        tokens = self.estimator.estimate_tokens(content, content_type)
        return self.budget.can_afford(tokens)

    def truncate_to_fit(
        self,
        content: str,
        reserve_tokens: int = 0,
        content_type: str = "prose",
    ) -> str:
        """Truncate content to fit remaining budget.

        Args:
            content: Content to truncate.
            reserve_tokens: Tokens to reserve for other content.
            content_type: Type of content.

        Returns:
            Truncated content.
        """
        available = self.budget.available - reserve_tokens
        if available <= 0:
            return ""

        # Estimate characters that fit
        adjustment = self.estimator.CONTENT_ADJUSTMENTS.get(content_type, 1.0)
        max_chars = int(available * self.estimator.chars_per_token * adjustment)

        if len(content) <= max_chars:
            return content

        # Truncate at word boundary
        truncated = content[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]

        return truncated + "..."

    def reset(self) -> None:
        """Reset context budget."""
        self.budget = TokenBudget(total_budget=self.max_tokens)


# Convenience functions


def estimate_tokens(
    text: str,
    tokenizer_type: TokenizerType = TokenizerType.GPT4,
    content_type: str = "prose",
) -> int:
    """Estimate token count for text.

    Args:
        text: Text to estimate.
        tokenizer_type: Type of tokenizer.
        content_type: Type of content.

    Returns:
        Estimated token count.
    """
    estimator = TokenEstimator(tokenizer_type)
    return estimator.estimate_tokens(text, content_type)


def analyze_tokens(text: str) -> TokenStats:
    """Analyze tokens in text.

    Args:
        text: Text to analyze.

    Returns:
        Token statistics.
    """
    analyzer = TokenAnalyzer()
    return analyzer.analyze(text)


def get_token_distribution(text: str) -> TokenDistribution:
    """Get token distribution for text.

    Args:
        text: Text to analyze.

    Returns:
        Token distribution.
    """
    analyzer = TokenAnalyzer()
    return analyzer.get_distribution(text)


def split_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 0,
    tokenizer_type: TokenizerType = TokenizerType.GPT4,
) -> list[str]:
    """Split text into chunks by token limit.

    Args:
        text: Text to split.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Overlapping tokens between chunks.
        tokenizer_type: Type of tokenizer for estimation.

    Returns:
        List of text chunks.
    """
    estimator = TokenEstimator(tokenizer_type)
    return estimator.split_to_chunks(text, max_tokens, overlap_tokens)


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity.
    """
    return EmbeddingUtils.cosine_similarity(vec1, vec2)
