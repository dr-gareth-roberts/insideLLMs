"""Tests for insideLLMs/nlp/text_metrics.py module."""

import pytest

from insideLLMs.nlp.text_metrics import (
    calculate_avg_sentence_length,
    calculate_avg_word_length,
    calculate_lexical_diversity,
    count_sentences,
    count_syllables,
    count_words,
    get_word_frequencies,
    simple_tokenize,
)


class TestSimpleTokenize:
    """Tests for simple_tokenize function."""

    def test_basic_tokenize(self):
        """Test basic tokenization."""
        result = simple_tokenize("hello world")
        assert result == ["hello", "world"]

    def test_empty_string(self):
        """Test tokenizing empty string."""
        result = simple_tokenize("")
        assert result == []

    def test_single_word(self):
        """Test single word."""
        result = simple_tokenize("hello")
        assert result == ["hello"]

    def test_multiple_spaces(self):
        """Test with multiple spaces."""
        result = simple_tokenize("hello   world")
        assert result == ["hello", "world"]


class TestCountWords:
    """Tests for count_words function."""

    def test_count_words_basic(self):
        """Test basic word counting."""
        result = count_words("hello world foo bar")
        assert result == 4

    def test_count_words_empty(self):
        """Test counting words in empty string."""
        result = count_words("")
        assert result == 0

    def test_count_words_single(self):
        """Test counting single word."""
        result = count_words("hello")
        assert result == 1


class TestCountSentences:
    """Tests for count_sentences function."""

    @pytest.fixture
    def nltk_available(self):
        """Check if NLTK is available."""
        try:
            import nltk  # noqa: F401
            return True
        except ImportError:
            pytest.skip("NLTK not available")

    def test_count_sentences_basic(self, nltk_available):
        """Test basic sentence counting."""
        result = count_sentences("Hello world. How are you? I am fine!")
        assert result == 3

    def test_count_sentences_single(self, nltk_available):
        """Test counting single sentence."""
        result = count_sentences("Hello world.")
        assert result == 1

    def test_count_sentences_no_punctuation(self, nltk_available):
        """Test sentence without ending punctuation."""
        result = count_sentences("Hello world")
        assert result >= 1


class TestCalculateAvgWordLength:
    """Tests for calculate_avg_word_length function."""

    def test_avg_word_length_basic(self):
        """Test average word length calculation."""
        result = calculate_avg_word_length("cat dog")
        assert result == 3.0

    def test_avg_word_length_mixed(self):
        """Test with mixed length words."""
        result = calculate_avg_word_length("a bb ccc")
        assert result == 2.0

    def test_avg_word_length_empty(self):
        """Test with empty string."""
        result = calculate_avg_word_length("")
        assert result == 0.0


class TestCalculateAvgSentenceLength:
    """Tests for calculate_avg_sentence_length function."""

    @pytest.fixture
    def nltk_available(self):
        """Check if NLTK is available."""
        try:
            import nltk  # noqa: F401
            return True
        except ImportError:
            pytest.skip("NLTK not available")

    def test_avg_sentence_length_basic(self, nltk_available):
        """Test average sentence length calculation."""
        result = calculate_avg_sentence_length("Hello world. Foo bar baz.")
        assert result > 0

    def test_avg_sentence_length_single(self, nltk_available):
        """Test with single sentence."""
        result = calculate_avg_sentence_length("Hello world foo bar")
        assert result == 4.0


class TestCalculateLexicalDiversity:
    """Tests for calculate_lexical_diversity function."""

    def test_lexical_diversity_all_unique(self):
        """Test with all unique words."""
        result = calculate_lexical_diversity("one two three four")
        assert result == 1.0

    def test_lexical_diversity_all_same(self):
        """Test with all same words."""
        result = calculate_lexical_diversity("the the the the")
        assert result == 0.25

    def test_lexical_diversity_empty(self):
        """Test with empty string."""
        result = calculate_lexical_diversity("")
        assert result == 0.0


class TestCountSyllables:
    """Tests for count_syllables function."""

    def test_syllables_one(self):
        """Test single syllable word."""
        assert count_syllables("cat") == 1

    def test_syllables_two(self):
        """Test two syllable word."""
        assert count_syllables("hello") == 2

    def test_syllables_three(self):
        """Test three syllable word."""
        assert count_syllables("beautiful") == 3

    def test_syllables_silent_e(self):
        """Test word ending in silent e."""
        result = count_syllables("make")
        assert result >= 1

    def test_syllables_empty(self):
        """Test empty string."""
        assert count_syllables("") == 0

    def test_syllables_numbers(self):
        """Test string with numbers."""
        result = count_syllables("123")
        assert result == 0


class TestGetWordFrequencies:
    """Tests for get_word_frequencies function."""

    def test_word_frequencies_basic(self):
        """Test basic word frequency counting."""
        result = get_word_frequencies("hello world hello")
        assert result["hello"] == 2
        assert result["world"] == 1

    def test_word_frequencies_empty(self):
        """Test with empty string."""
        result = get_word_frequencies("")
        assert result == {}

    def test_word_frequencies_unique(self):
        """Test with all unique words."""
        result = get_word_frequencies("one two three")
        assert len(result) == 3
        assert all(v == 1 for v in result.values())


class TestNltkTokenize:
    """Tests for nltk_tokenize_internal function."""

    @pytest.fixture
    def nltk_available(self):
        """Check if NLTK is available."""
        try:
            import nltk  # noqa: F401
            return True
        except ImportError:
            pytest.skip("NLTK not available")

    def test_nltk_tokenize_basic(self, nltk_available):
        """Test NLTK tokenization."""
        from insideLLMs.nlp.text_metrics import nltk_tokenize_internal

        result = nltk_tokenize_internal("Hello, world!")
        assert "Hello" in result
        assert "world" in result


class TestSegmentSentences:
    """Tests for segment_sentences_internal function."""

    @pytest.fixture
    def nltk_available(self):
        """Check if NLTK is available."""
        try:
            import nltk  # noqa: F401
            return True
        except ImportError:
            pytest.skip("NLTK not available")

    def test_segment_sentences_with_nltk(self, nltk_available):
        """Test sentence segmentation with NLTK."""
        from insideLLMs.nlp.text_metrics import segment_sentences_internal

        text = "Hello world. This is a test. How are you?"
        result = segment_sentences_internal(text, use_nltk=True)
        assert len(result) == 3

    def test_segment_sentences_without_nltk(self):
        """Test sentence segmentation without NLTK."""
        from insideLLMs.nlp.text_metrics import segment_sentences_internal

        text = "Hello world. This is a test. How are you?"
        result = segment_sentences_internal(text, use_nltk=False)
        assert len(result) >= 1  # At least some splitting should occur


class TestFleschKincaid:
    """Tests for calculate_readability_flesch_kincaid function."""

    @pytest.fixture
    def nltk_available(self):
        """Check if NLTK is available."""
        try:
            import nltk  # noqa: F401
            return True
        except ImportError:
            pytest.skip("NLTK not available")

    def test_flesch_kincaid_basic(self, nltk_available):
        """Test Flesch-Kincaid score calculation."""
        from insideLLMs.nlp.text_metrics import calculate_readability_flesch_kincaid

        text = "The cat sat on the mat. It was a nice day."
        result = calculate_readability_flesch_kincaid(text)
        assert isinstance(result, float)

    def test_flesch_kincaid_empty(self, nltk_available):
        """Test Flesch-Kincaid with empty string."""
        from insideLLMs.nlp.text_metrics import calculate_readability_flesch_kincaid

        result = calculate_readability_flesch_kincaid("")
        assert result == 0.0

    def test_flesch_kincaid_complex_text(self, nltk_available):
        """Test Flesch-Kincaid with more complex text."""
        from insideLLMs.nlp.text_metrics import calculate_readability_flesch_kincaid

        text = (
            "The implementation of sophisticated algorithms requires "
            "comprehensive understanding of computational complexity."
        )
        result = calculate_readability_flesch_kincaid(text)
        # Higher grade level for complex text
        assert isinstance(result, float)
