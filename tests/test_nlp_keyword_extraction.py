"""Tests for insideLLMs/nlp/keyword_extraction.py module."""

import pytest

from insideLLMs.nlp.keyword_extraction import (
    extract_keywords_textrank,
    extract_keywords_tfidf,
    nltk_tokenize_for_keyword,
    remove_stopwords_for_keyword,
    segment_sentences_for_keyword,
)


class TestCheckFunctions:
    """Tests for dependency check functions."""

    def test_check_nltk_available(self):
        """Test check_nltk when NLTK is available."""
        try:
            import nltk

            nltk.data.find("tokenizers/punkt")
            nltk.data.find("corpora/stopwords")
        except (ImportError, LookupError):
            pytest.skip("NLTK not available")

        from insideLLMs.nlp.keyword_extraction import check_nltk

        check_nltk()  # Should not raise

    def test_check_sklearn_available(self):
        """Test check_sklearn when sklearn is available."""
        try:
            import sklearn  # noqa: F401
        except ImportError:
            pytest.skip("sklearn not available")

        from insideLLMs.nlp.keyword_extraction import check_sklearn

        check_sklearn()  # Should not raise


class TestSegmentSentences:
    """Tests for segment_sentences_for_keyword function."""

    @pytest.fixture
    def nltk_available(self):
        """Check if NLTK is available."""
        try:
            import nltk

            nltk.data.find("tokenizers/punkt")
            return True
        except (ImportError, LookupError):
            pytest.skip("NLTK not available")

    def test_basic_sentence_segmentation(self, nltk_available):
        """Test basic sentence segmentation."""
        text = "Hello world. This is a test. Another sentence here."
        result = segment_sentences_for_keyword(text)
        assert len(result) >= 3

    def test_single_sentence(self, nltk_available):
        """Test with single sentence."""
        text = "This is a single sentence."
        result = segment_sentences_for_keyword(text)
        assert len(result) >= 1

    def test_empty_text(self, nltk_available):
        """Test with empty text."""
        text = ""
        result = segment_sentences_for_keyword(text)
        assert isinstance(result, list)


class TestNltkTokenize:
    """Tests for nltk_tokenize_for_keyword function."""

    @pytest.fixture
    def nltk_available(self):
        """Check if NLTK is available."""
        try:
            import nltk

            nltk.data.find("tokenizers/punkt")
            return True
        except (ImportError, LookupError):
            pytest.skip("NLTK not available")

    def test_basic_tokenization(self, nltk_available):
        """Test basic tokenization."""
        text = "Hello world, this is a test."
        result = nltk_tokenize_for_keyword(text)
        assert "Hello" in result
        assert "world" in result

    def test_handles_punctuation(self, nltk_available):
        """Test handling of punctuation."""
        text = "Hello, world!"
        result = nltk_tokenize_for_keyword(text)
        assert isinstance(result, list)


class TestRemoveStopwords:
    """Tests for remove_stopwords_for_keyword function."""

    @pytest.fixture
    def nltk_available(self):
        """Check if NLTK is available."""
        try:
            import nltk

            nltk.data.find("corpora/stopwords")
            return True
        except (ImportError, LookupError):
            pytest.skip("NLTK not available")

    def test_removes_english_stopwords(self, nltk_available):
        """Test removing English stopwords."""
        tokens = ["the", "cat", "is", "on", "the", "mat"]
        result = remove_stopwords_for_keyword(tokens)
        assert "the" not in result
        assert "is" not in result
        assert "cat" in result
        assert "mat" in result

    def test_empty_list(self, nltk_available):
        """Test with empty list."""
        result = remove_stopwords_for_keyword([])
        assert result == []

    def test_all_stopwords(self, nltk_available):
        """Test with all stopwords."""
        tokens = ["the", "is", "a", "an", "of"]
        result = remove_stopwords_for_keyword(tokens)
        assert len(result) < len(tokens)


class TestExtractKeywordsTfidf:
    """Tests for extract_keywords_tfidf function."""

    @pytest.fixture
    def dependencies_available(self):
        """Check if required dependencies are available."""
        try:
            import nltk
            import sklearn

            nltk.data.find("tokenizers/punkt")
            return True
        except (ImportError, LookupError):
            pytest.skip("Required dependencies not available")

    def test_extracts_keywords(self, dependencies_available):
        """Test basic keyword extraction."""
        text = "Machine learning is great. Machine learning algorithms work well. Deep learning is a subset of machine learning."
        result = extract_keywords_tfidf(text, num_keywords=5)
        assert isinstance(result, list)
        assert len(result) <= 5

    def test_custom_num_keywords(self, dependencies_available):
        """Test with custom number of keywords."""
        text = "Python programming is fun. Python is easy to learn. Programming in Python is productive."
        result = extract_keywords_tfidf(text, num_keywords=3)
        assert len(result) <= 3

    def test_empty_text(self, dependencies_available):
        """Test with empty text."""
        text = ""
        result = extract_keywords_tfidf(text)
        assert result == []

    def test_short_text(self, dependencies_available):
        """Test with very short text."""
        text = "Hello"
        result = extract_keywords_tfidf(text)
        assert isinstance(result, list)


class TestExtractKeywordsTextrank:
    """Tests for extract_keywords_textrank function."""

    @pytest.fixture
    def nltk_available(self):
        """Check if NLTK is available."""
        try:
            import nltk

            nltk.data.find("tokenizers/punkt")
            nltk.data.find("corpora/stopwords")
            return True
        except (ImportError, LookupError):
            pytest.skip("NLTK not available")

    def test_extracts_keywords(self, nltk_available):
        """Test basic TextRank keyword extraction."""
        text = "Machine learning algorithms are powerful. Deep learning is a subset of machine learning. Neural networks are used in deep learning."
        result = extract_keywords_textrank(text, num_keywords=5)
        assert isinstance(result, list)
        assert len(result) <= 5

    def test_custom_num_keywords(self, nltk_available):
        """Test with custom number of keywords."""
        text = "Python programming is excellent. Python code is readable. Programming in Python is enjoyable."
        result = extract_keywords_textrank(text, num_keywords=3)
        assert len(result) <= 3

    def test_custom_window_size(self, nltk_available):
        """Test with custom window size."""
        text = "Data science involves statistics and programming. Machine learning uses statistical methods."
        result = extract_keywords_textrank(text, num_keywords=5, window_size=3)
        assert isinstance(result, list)

    def test_empty_text(self, nltk_available):
        """Test with empty text."""
        text = ""
        result = extract_keywords_textrank(text)
        assert result == []

    def test_stopwords_only(self, nltk_available):
        """Test with text containing only stopwords."""
        text = "the is a an of the is a"
        result = extract_keywords_textrank(text)
        assert result == []

    def test_single_word(self, nltk_available):
        """Test with single content word."""
        text = "Python"
        result = extract_keywords_textrank(text)
        assert isinstance(result, list)
