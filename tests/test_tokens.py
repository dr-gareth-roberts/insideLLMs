"""Tests for token analysis and embedding utilities."""

import math

import pytest

from insideLLMs.tokens import (
    ContextWindowManager,
    EmbeddingUtils,
    SimpleTokenizer,
    TokenAnalyzer,
    TokenBudget,
    TokenDistribution,
    TokenEstimator,
    TokenizerType,
    TokenStats,
    VocabCoverage,
    analyze_tokens,
    cosine_similarity,
    estimate_tokens,
    get_token_distribution,
    split_by_tokens,
)


class TestTokenizerType:
    """Tests for TokenizerType enum."""

    def test_all_types_exist(self):
        """Test that all types are defined."""
        assert TokenizerType.GPT4.value == "gpt4"
        assert TokenizerType.CLAUDE.value == "claude"
        assert TokenizerType.LLAMA.value == "llama"
        assert TokenizerType.SIMPLE.value == "simple"


class TestTokenStats:
    """Tests for TokenStats."""

    def test_basic_stats(self):
        """Test basic statistics."""
        stats = TokenStats(
            total_tokens=100,
            unique_tokens=50,
            char_count=500,
            word_count=80,
            avg_token_length=5.0,
            tokens_per_word=1.25,
        )

        assert stats.total_tokens == 100
        assert stats.unique_tokens == 50
        assert stats.token_diversity == 0.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = TokenStats(
            total_tokens=10,
            unique_tokens=5,
            char_count=50,
            word_count=8,
            avg_token_length=5.0,
            tokens_per_word=1.25,
        )

        d = stats.to_dict()
        assert d["total_tokens"] == 10
        assert d["token_diversity"] == 0.5


class TestTokenDistribution:
    """Tests for TokenDistribution."""

    def test_top_tokens(self):
        """Test getting top tokens."""
        dist = TokenDistribution(
            frequencies={"the": 10, "a": 5, "is": 3},
            total_tokens=18,
        )

        top = dist.top_tokens(2)
        assert top[0] == ("the", 10)
        assert top[1] == ("a", 5)

    def test_bottom_tokens(self):
        """Test getting bottom tokens."""
        dist = TokenDistribution(
            frequencies={"the": 10, "a": 5, "is": 3},
            total_tokens=18,
        )

        bottom = dist.bottom_tokens(1)
        assert bottom[0] == ("is", 3)

    def test_relative_frequency(self):
        """Test relative frequency."""
        dist = TokenDistribution(
            frequencies={"the": 10},
            total_tokens=100,
        )

        assert dist.relative_frequency("the") == 0.1
        assert dist.relative_frequency("missing") == 0.0

    def test_hapax_legomena(self):
        """Test finding words appearing once."""
        dist = TokenDistribution(
            frequencies={"common": 5, "rare": 1, "unique": 1},
            total_tokens=7,
        )

        hapax = dist.hapax_legomena()
        assert "rare" in hapax
        assert "unique" in hapax
        assert "common" not in hapax

    def test_entropy(self):
        """Test entropy calculation."""
        # Uniform distribution has maximum entropy
        dist = TokenDistribution(
            frequencies={"a": 10, "b": 10, "c": 10, "d": 10},
            total_tokens=40,
        )
        entropy = dist.entropy()
        assert entropy > 0
        assert abs(entropy - 2.0) < 0.001  # log2(4) = 2

    def test_vocabulary_size(self):
        """Test vocabulary size property."""
        dist = TokenDistribution(
            frequencies={"a": 1, "b": 2, "c": 3},
            total_tokens=6,
        )
        assert dist.vocabulary_size == 3


class TestVocabCoverage:
    """Tests for VocabCoverage."""

    def test_full_coverage(self):
        """Test full vocabulary coverage."""
        coverage = VocabCoverage(
            text_vocab={"a", "b", "c"},
            reference_vocab={"a", "b", "c", "d"},
            covered={"a", "b", "c"},
            uncovered=set(),
        )

        assert coverage.coverage_ratio == 1.0
        assert coverage.oov_ratio == 0.0

    def test_partial_coverage(self):
        """Test partial coverage."""
        coverage = VocabCoverage(
            text_vocab={"a", "b", "x"},
            reference_vocab={"a", "b", "c"},
            covered={"a", "b"},
            uncovered={"x"},
        )

        assert abs(coverage.coverage_ratio - 2/3) < 0.001

    def test_to_dict(self):
        """Test conversion to dictionary."""
        coverage = VocabCoverage(
            text_vocab={"a", "b"},
            reference_vocab={"a", "b", "c"},
            covered={"a", "b"},
            uncovered=set(),
        )

        d = coverage.to_dict()
        assert d["text_vocab_size"] == 2
        assert d["coverage_ratio"] == 1.0


class TestSimpleTokenizer:
    """Tests for SimpleTokenizer."""

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        tokenizer = SimpleTokenizer()
        tokens = tokenizer.tokenize("Hello, world!")

        assert "hello" in tokens
        assert "world" in tokens
        assert "," in tokens

    def test_tokenize_case_sensitive(self):
        """Test case-sensitive tokenization."""
        tokenizer = SimpleTokenizer(lowercase=False)
        tokens = tokenizer.tokenize("Hello World")

        assert "Hello" in tokens
        assert "World" in tokens

    def test_encode_decode(self):
        """Test encoding and decoding."""
        tokenizer = SimpleTokenizer()
        text = "hello world"

        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)

        assert decoded == text

    def test_vocab_size(self):
        """Test vocabulary size tracking."""
        tokenizer = SimpleTokenizer()
        tokenizer.encode("hello world")

        assert tokenizer.vocab_size == 2

        tokenizer.encode("hello there")
        assert tokenizer.vocab_size == 3


class TestTokenEstimator:
    """Tests for TokenEstimator."""

    def test_estimate_tokens(self):
        """Test token estimation."""
        estimator = TokenEstimator(TokenizerType.GPT4)
        text = "This is a test sentence with some words."

        tokens = estimator.estimate_tokens(text)
        assert tokens > 0
        # Approximately 40 chars / 4 chars per token = 10
        assert 5 <= tokens <= 15

    def test_content_type_adjustment(self):
        """Test content type affects estimation."""
        estimator = TokenEstimator()
        text = "def foo():\n    return 42"

        code_tokens = estimator.estimate_tokens(text, "code")
        prose_tokens = estimator.estimate_tokens(text, "prose")

        # Code should have more tokens
        assert code_tokens > prose_tokens

    def test_estimate_cost(self):
        """Test cost estimation."""
        estimator = TokenEstimator()
        text = "A" * 4000  # ~1000 tokens

        cost = estimator.estimate_cost(text, price_per_1k_tokens=0.01)
        assert 0.005 <= cost <= 0.02

    def test_fits_context(self):
        """Test context limit checking."""
        estimator = TokenEstimator()

        short_text = "Hello world"
        assert estimator.fits_context(short_text, 1000)

        long_text = "x" * 50000  # ~12500 tokens
        assert not estimator.fits_context(long_text, 1000)

    def test_split_to_chunks(self):
        """Test splitting to chunks."""
        estimator = TokenEstimator()
        text = "This is a test. " * 100  # Long text

        chunks = estimator.split_to_chunks(text, max_tokens=50)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) > 0


class TestTokenAnalyzer:
    """Tests for TokenAnalyzer."""

    def test_analyze(self):
        """Test basic analysis."""
        analyzer = TokenAnalyzer()
        text = "The quick brown fox jumps over the lazy dog."

        stats = analyzer.analyze(text)

        assert stats.total_tokens > 0
        assert stats.unique_tokens > 0
        assert stats.char_count == len(text)

    def test_get_distribution(self):
        """Test distribution analysis."""
        analyzer = TokenAnalyzer()
        text = "the cat and the dog"

        dist = analyzer.get_distribution(text)

        assert dist.frequencies["the"] == 2
        assert dist.frequencies["cat"] == 1

    def test_compare_distributions(self):
        """Test comparing distributions."""
        analyzer = TokenAnalyzer()
        text1 = "the quick brown fox"
        text2 = "the slow brown bear"

        comparison = analyzer.compare_distributions(text1, text2)

        assert comparison["shared_vocab_size"] > 0
        assert 0 <= comparison["jaccard_similarity"] <= 1
        assert 0 <= comparison["cosine_similarity"] <= 1

    def test_vocabulary_coverage(self):
        """Test vocabulary coverage analysis."""
        analyzer = TokenAnalyzer()
        text = "hello world"
        reference = {"hello", "world", "foo", "bar"}

        coverage = analyzer.analyze_vocabulary_coverage(text, reference)

        assert coverage.coverage_ratio == 1.0


class TestEmbeddingUtils:
    """Tests for EmbeddingUtils."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        vec = [1.0, 2.0, 3.0]
        sim = EmbeddingUtils.cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.0001

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        sim = EmbeddingUtils.cosine_similarity(vec1, vec2)
        assert abs(sim) < 0.0001

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors."""
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]
        sim = EmbeddingUtils.cosine_similarity(vec1, vec2)
        assert abs(sim - (-1.0)) < 0.0001

    def test_euclidean_distance(self):
        """Test Euclidean distance."""
        vec1 = [0.0, 0.0]
        vec2 = [3.0, 4.0]
        dist = EmbeddingUtils.euclidean_distance(vec1, vec2)
        assert abs(dist - 5.0) < 0.0001

    def test_manhattan_distance(self):
        """Test Manhattan distance."""
        vec1 = [0.0, 0.0]
        vec2 = [3.0, 4.0]
        dist = EmbeddingUtils.manhattan_distance(vec1, vec2)
        assert abs(dist - 7.0) < 0.0001

    def test_normalize(self):
        """Test vector normalization."""
        vec = [3.0, 4.0]
        normalized = EmbeddingUtils.normalize(vec)

        magnitude = math.sqrt(sum(v ** 2 for v in normalized))
        assert abs(magnitude - 1.0) < 0.0001

    def test_average_embeddings(self):
        """Test averaging embeddings."""
        embeddings = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
        avg = EmbeddingUtils.average_embeddings(embeddings)

        assert abs(avg[0] - 3.0) < 0.0001
        assert abs(avg[1] - 4.0) < 0.0001

    def test_find_most_similar(self):
        """Test finding most similar embeddings."""
        query = [1.0, 0.0]
        candidates = [
            ("a", [1.0, 0.0]),  # Most similar
            ("b", [0.0, 1.0]),  # Orthogonal
            ("c", [0.7, 0.7]),  # Somewhat similar
        ]

        results = EmbeddingUtils.find_most_similar(query, candidates, top_k=2)

        assert results[0][0] == "a"
        assert results[0][1] > 0.9

    def test_dimension_mismatch_error(self):
        """Test error on dimension mismatch."""
        vec1 = [1.0, 2.0]
        vec2 = [1.0, 2.0, 3.0]

        with pytest.raises(ValueError):
            EmbeddingUtils.cosine_similarity(vec1, vec2)


class TestTokenBudget:
    """Tests for TokenBudget."""

    def test_basic_operations(self):
        """Test basic budget operations."""
        budget = TokenBudget(total_budget=1000)

        assert budget.available == 1000
        assert budget.can_afford(500)

    def test_spend(self):
        """Test spending tokens."""
        budget = TokenBudget(total_budget=1000)

        assert budget.spend(300)
        assert budget.available == 700
        assert budget.used_tokens == 300

    def test_spend_over_budget(self):
        """Test spending over budget."""
        budget = TokenBudget(total_budget=100)

        assert not budget.spend(150)
        assert budget.used_tokens == 0

    def test_reserve(self):
        """Test reserving tokens."""
        budget = TokenBudget(total_budget=1000)

        assert budget.reserve(200)
        assert budget.available == 800
        assert budget.reserved_tokens == 200

    def test_release_reservation(self):
        """Test releasing reservation."""
        budget = TokenBudget(total_budget=1000)
        budget.reserve(200)
        budget.release_reservation(100)

        assert budget.reserved_tokens == 100

    def test_utilization(self):
        """Test utilization calculation."""
        budget = TokenBudget(total_budget=1000)
        budget.spend(500)

        assert budget.utilization == 0.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        budget = TokenBudget(total_budget=1000, used_tokens=300)
        d = budget.to_dict()

        assert d["total_budget"] == 1000
        assert d["used_tokens"] == 300


class TestContextWindowManager:
    """Tests for ContextWindowManager."""

    def test_add_content(self):
        """Test adding content."""
        manager = ContextWindowManager(max_tokens=1000)

        assert manager.add_content("Hello world")
        assert manager.remaining_tokens() < 1000

    def test_fits(self):
        """Test checking if content fits."""
        manager = ContextWindowManager(max_tokens=100)

        assert manager.fits("Hello world")
        assert not manager.fits("x" * 1000)

    def test_truncate_to_fit(self):
        """Test truncating content to fit."""
        manager = ContextWindowManager(max_tokens=10)

        long_text = "This is a very long text that will not fit in the context window."
        truncated = manager.truncate_to_fit(long_text)

        assert len(truncated) < len(long_text)
        assert truncated.endswith("...")

    def test_reset(self):
        """Test resetting context."""
        manager = ContextWindowManager(max_tokens=1000)
        manager.add_content("Hello world")
        initial_remaining = manager.remaining_tokens()

        manager.reset()

        assert manager.remaining_tokens() == 1000


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_estimate_tokens(self):
        """Test estimate_tokens function."""
        tokens = estimate_tokens("Hello world", TokenizerType.GPT4)
        assert tokens > 0

    def test_analyze_tokens(self):
        """Test analyze_tokens function."""
        stats = analyze_tokens("The quick brown fox")
        assert stats.total_tokens > 0

    def test_get_token_distribution(self):
        """Test get_token_distribution function."""
        dist = get_token_distribution("hello hello world")
        assert dist.frequencies["hello"] == 2

    def test_split_by_tokens(self):
        """Test split_by_tokens function."""
        text = "This is a test. " * 50
        chunks = split_by_tokens(text, max_tokens=20)
        assert len(chunks) > 1

    def test_cosine_similarity(self):
        """Test cosine_similarity function."""
        sim = cosine_similarity([1, 0], [1, 0])
        assert abs(sim - 1.0) < 0.0001


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_text(self):
        """Test with empty text."""
        analyzer = TokenAnalyzer()
        stats = analyzer.analyze("")

        assert stats.total_tokens == 0

    def test_unicode_text(self):
        """Test with unicode text."""
        analyzer = TokenAnalyzer()
        stats = analyzer.analyze("Hello 世界! 🌍")

        assert stats.total_tokens > 0

    def test_zero_vector(self):
        """Test with zero vector."""
        vec = [0.0, 0.0]
        normalized = EmbeddingUtils.normalize(vec)
        assert normalized == vec

    def test_single_token(self):
        """Test with single token."""
        analyzer = TokenAnalyzer()
        dist = analyzer.get_distribution("hello")

        assert dist.vocabulary_size == 1
        assert dist.entropy() == 0.0
