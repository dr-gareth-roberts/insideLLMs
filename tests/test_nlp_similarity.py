"""Tests for insideLLMs/nlp/similarity.py module."""

import pytest

from insideLLMs.nlp.similarity import (
    cosine_similarity_texts,
    hamming_distance,
    jaccard_similarity,
    jaro_similarity,
    jaro_winkler_similarity,
    levenshtein_distance,
    longest_common_subsequence,
    simple_tokenize,
)


class TestSimpleTokenize:
    """Tests for simple_tokenize helper function."""

    def test_basic_tokenization(self):
        """Test basic word tokenization."""
        result = simple_tokenize("hello world")
        assert result == ["hello", "world"]

    def test_empty_string(self):
        """Test tokenizing empty string."""
        result = simple_tokenize("")
        assert result == []

    def test_single_word(self):
        """Test tokenizing single word."""
        result = simple_tokenize("hello")
        assert result == ["hello"]

    def test_multiple_spaces(self):
        """Test tokenizing with multiple spaces."""
        result = simple_tokenize("hello   world")
        # split() collapses multiple spaces
        assert result == ["hello", "world"]


class TestJaccardSimilarity:
    """Tests for jaccard_similarity function."""

    def test_identical_texts(self):
        """Test Jaccard similarity of identical texts."""
        similarity = jaccard_similarity("hello world", "hello world")
        assert similarity == 1.0

    def test_completely_different(self):
        """Test Jaccard similarity of completely different texts."""
        similarity = jaccard_similarity("hello world", "foo bar")
        assert similarity == 0.0

    def test_partial_overlap(self):
        """Test Jaccard similarity with partial overlap."""
        similarity = jaccard_similarity("hello world", "hello universe")
        # "hello" is common, "world" and "universe" are different
        assert 0 < similarity < 1

    def test_empty_texts(self):
        """Test Jaccard similarity with empty texts."""
        similarity = jaccard_similarity("", "")
        assert similarity == 0.0

    def test_one_empty_text(self):
        """Test Jaccard similarity with one empty text."""
        similarity = jaccard_similarity("hello", "")
        assert similarity == 0.0

    def test_custom_tokenizer(self):
        """Test with custom tokenizer."""
        def char_tokenizer(text):
            return list(text)

        similarity = jaccard_similarity("abc", "abd", tokenizer=char_tokenizer)
        # a, b common; c, d different -> 2/4 = 0.5
        assert similarity == 0.5


class TestLevenshteinDistance:
    """Tests for levenshtein_distance function."""

    def test_identical_strings(self):
        """Test distance between identical strings."""
        distance = levenshtein_distance("hello", "hello")
        assert distance == 0

    def test_single_substitution(self):
        """Test distance with single substitution."""
        distance = levenshtein_distance("cat", "car")
        assert distance == 1

    def test_single_insertion(self):
        """Test distance with single insertion."""
        distance = levenshtein_distance("cat", "cats")
        assert distance == 1

    def test_single_deletion(self):
        """Test distance with single deletion."""
        distance = levenshtein_distance("cats", "cat")
        assert distance == 1

    def test_empty_string(self):
        """Test distance with empty string."""
        distance = levenshtein_distance("hello", "")
        assert distance == 5

    def test_both_empty(self):
        """Test distance between two empty strings."""
        distance = levenshtein_distance("", "")
        assert distance == 0

    def test_completely_different(self):
        """Test distance between completely different strings."""
        distance = levenshtein_distance("abc", "xyz")
        assert distance == 3

    def test_longer_strings(self):
        """Test with longer strings."""
        distance = levenshtein_distance("kitten", "sitting")
        assert distance == 3  # k->s, e->i, +g


class TestJaroSimilarity:
    """Tests for jaro_similarity function."""

    def test_identical_strings(self):
        """Test Jaro similarity of identical strings."""
        similarity = jaro_similarity("hello", "hello")
        assert similarity == 1.0

    def test_completely_different(self):
        """Test Jaro similarity of completely different strings."""
        similarity = jaro_similarity("abc", "xyz")
        assert similarity == 0.0

    def test_empty_strings(self):
        """Test Jaro similarity of empty strings."""
        similarity = jaro_similarity("", "")
        assert similarity == 1.0

    def test_one_empty_string(self):
        """Test Jaro similarity with one empty string."""
        similarity = jaro_similarity("hello", "")
        assert similarity == 0.0

    def test_similar_strings(self):
        """Test Jaro similarity of similar strings."""
        similarity = jaro_similarity("martha", "marhta")
        assert 0.9 < similarity < 1.0

    def test_partial_match(self):
        """Test Jaro similarity with partial match."""
        similarity = jaro_similarity("dwayne", "duane")
        assert 0.7 < similarity < 1.0


class TestJaroWinklerSimilarity:
    """Tests for jaro_winkler_similarity function."""

    def test_identical_strings(self):
        """Test Jaro-Winkler similarity of identical strings."""
        similarity = jaro_winkler_similarity("hello", "hello")
        assert similarity == 1.0

    def test_common_prefix_boost(self):
        """Test that common prefix boosts similarity."""
        jaro = jaro_similarity("prefix_abc", "prefix_xyz")
        jaro_winkler = jaro_winkler_similarity("prefix_abc", "prefix_xyz")
        assert jaro_winkler >= jaro

    def test_no_common_prefix(self):
        """Test with no common prefix."""
        similarity = jaro_winkler_similarity("abc", "xyz")
        assert similarity == 0.0

    def test_custom_scaling(self):
        """Test with custom scaling factor."""
        sim1 = jaro_winkler_similarity("test", "text", scaling=0.1)
        sim2 = jaro_winkler_similarity("test", "text", scaling=0.2)
        # Higher scaling should increase similarity for strings with common prefix
        assert sim2 >= sim1


class TestHammingDistance:
    """Tests for hamming_distance function."""

    def test_identical_strings(self):
        """Test Hamming distance of identical strings."""
        distance = hamming_distance("hello", "hello")
        assert distance == 0

    def test_one_difference(self):
        """Test Hamming distance with one difference."""
        distance = hamming_distance("hello", "hella")
        assert distance == 1

    def test_all_different(self):
        """Test Hamming distance when all chars different."""
        distance = hamming_distance("abc", "xyz")
        assert distance == 3

    def test_unequal_length_raises(self):
        """Test that unequal lengths raise ValueError."""
        with pytest.raises(ValueError):
            hamming_distance("hello", "hi")

    def test_empty_strings(self):
        """Test Hamming distance of empty strings."""
        distance = hamming_distance("", "")
        assert distance == 0


class TestLongestCommonSubsequence:
    """Tests for longest_common_subsequence function."""

    def test_identical_strings(self):
        """Test LCS of identical strings."""
        lcs = longest_common_subsequence("hello", "hello")
        assert lcs == 5

    def test_completely_different(self):
        """Test LCS of completely different strings."""
        lcs = longest_common_subsequence("abc", "xyz")
        assert lcs == 0

    def test_partial_match(self):
        """Test LCS with partial match."""
        lcs = longest_common_subsequence("abcdef", "ace")
        assert lcs == 3  # "ace" is common

    def test_empty_string(self):
        """Test LCS with empty string."""
        lcs = longest_common_subsequence("hello", "")
        assert lcs == 0

    def test_both_empty(self):
        """Test LCS of two empty strings."""
        lcs = longest_common_subsequence("", "")
        assert lcs == 0

    def test_classic_example(self):
        """Test classic LCS example."""
        lcs = longest_common_subsequence("AGGTAB", "GXTXAYB")
        assert lcs == 4  # "GTAB"


class TestCosineSimilarityTexts:
    """Tests for cosine_similarity_texts function."""

    @pytest.fixture
    def sklearn_available(self):
        """Check if sklearn is available."""
        try:
            import sklearn  # noqa: F401
            return True
        except ImportError:
            pytest.skip("sklearn not available")

    def test_identical_texts(self, sklearn_available):
        """Test cosine similarity of identical texts."""
        similarity = cosine_similarity_texts("hello world", "hello world")
        assert similarity == 1.0

    def test_similar_texts(self, sklearn_available):
        """Test cosine similarity of similar texts."""
        similarity = cosine_similarity_texts(
            "the cat sat on the mat",
            "the cat sat on the rug"
        )
        assert 0.5 < similarity < 1.0

    def test_different_texts(self, sklearn_available):
        """Test cosine similarity of different texts."""
        similarity = cosine_similarity_texts(
            "apple banana orange",
            "car truck bicycle"
        )
        assert similarity < 0.5

    def test_result_is_float(self, sklearn_available):
        """Test that result is a float."""
        similarity = cosine_similarity_texts("hello", "world")
        assert isinstance(similarity, float)
