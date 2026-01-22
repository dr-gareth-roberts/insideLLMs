"""Tests for insideLLMs/nlp/keyword_extraction.py module."""

import pytest


class TestSegmentSentencesForKeyword:
    """Tests for segment_sentences_for_keyword function."""

    @pytest.fixture
    def nltk_available(self):
        """Check if NLTK is available."""
        try:
            import nltk  # noqa: F401

            return True
        except ImportError:
            pytest.skip("NLTK not available")

    def test_segment_sentences_basic(self, nltk_available):
        """Test basic sentence segmentation."""
        from insideLLMs.nlp.keyword_extraction import segment_sentences_for_keyword

        result = segment_sentences_for_keyword("Hello world. How are you?")
        assert len(result) == 2

    def test_segment_sentences_single(self, nltk_available):
        """Test single sentence."""
        from insideLLMs.nlp.keyword_extraction import segment_sentences_for_keyword

        result = segment_sentences_for_keyword("Hello world")
        assert len(result) == 1


class TestNltkTokenizeForKeyword:
    """Tests for nltk_tokenize_for_keyword function."""

    @pytest.fixture
    def nltk_available(self):
        """Check if NLTK is available."""
        try:
            import nltk  # noqa: F401

            return True
        except ImportError:
            pytest.skip("NLTK not available")

    def test_tokenize_basic(self, nltk_available):
        """Test basic tokenization."""
        from insideLLMs.nlp.keyword_extraction import nltk_tokenize_for_keyword

        result = nltk_tokenize_for_keyword("Hello world foo bar")
        assert "Hello" in result
        assert "world" in result


class TestRemoveStopwordsForKeyword:
    """Tests for remove_stopwords_for_keyword function."""

    @pytest.fixture
    def nltk_available(self):
        """Check if NLTK is available."""
        try:
            import nltk  # noqa: F401

            return True
        except ImportError:
            pytest.skip("NLTK not available")

    def test_remove_stopwords_basic(self, nltk_available):
        """Test basic stopword removal."""
        from insideLLMs.nlp.keyword_extraction import remove_stopwords_for_keyword

        tokens = ["the", "cat", "sat", "on", "the", "mat"]
        result = remove_stopwords_for_keyword(tokens)

        assert "the" not in result
        assert "on" not in result
        assert "cat" in result
        assert "sat" in result


class TestExtractKeywordsTfidf:
    """Tests for extract_keywords_tfidf function."""

    @pytest.fixture
    def deps_available(self):
        """Check if sklearn and NLTK are available."""
        try:
            import nltk  # noqa: F401
            import sklearn  # noqa: F401

            return True
        except ImportError:
            pytest.skip("sklearn or NLTK not available")

    def test_extract_keywords_basic(self, deps_available):
        """Test basic keyword extraction."""
        from insideLLMs.nlp.keyword_extraction import extract_keywords_tfidf

        text = """
        Machine learning is a subset of artificial intelligence.
        Machine learning algorithms learn from data.
        Deep learning is a subset of machine learning.
        Neural networks are used in deep learning.
        """
        result = extract_keywords_tfidf(text, num_keywords=5)

        assert isinstance(result, list)
        assert len(result) <= 5

    def test_extract_keywords_empty(self, deps_available):
        """Test with empty text."""
        from insideLLMs.nlp.keyword_extraction import extract_keywords_tfidf

        result = extract_keywords_tfidf("")
        assert result == []

    def test_extract_keywords_stopwords_only(self, deps_available):
        """Test with text containing only stopwords."""
        from insideLLMs.nlp.keyword_extraction import extract_keywords_tfidf

        result = extract_keywords_tfidf("the the the a a a")
        assert result == []


class TestExtractKeywordsTextrank:
    """Tests for extract_keywords_textrank function."""

    @pytest.fixture
    def nltk_available(self):
        """Check if NLTK is available."""
        try:
            import nltk  # noqa: F401

            return True
        except ImportError:
            pytest.skip("NLTK not available")

    def test_extract_keywords_basic(self, nltk_available):
        """Test basic TextRank keyword extraction."""
        from insideLLMs.nlp.keyword_extraction import extract_keywords_textrank

        text = """
        Machine learning algorithms learn patterns from data.
        Deep learning uses neural networks for complex patterns.
        Artificial intelligence includes machine learning techniques.
        Data science applies machine learning to solve problems.
        """
        result = extract_keywords_textrank(text, num_keywords=5)

        assert isinstance(result, list)
        assert len(result) <= 5

    def test_extract_keywords_empty(self, nltk_available):
        """Test with empty text."""
        from insideLLMs.nlp.keyword_extraction import extract_keywords_textrank

        result = extract_keywords_textrank("")
        assert result == []

    def test_extract_keywords_custom_window(self, nltk_available):
        """Test with custom window size."""
        from insideLLMs.nlp.keyword_extraction import extract_keywords_textrank

        text = "Python programming language data science machine learning"
        result = extract_keywords_textrank(text, num_keywords=3, window_size=2)

        assert isinstance(result, list)
        assert len(result) <= 3

    def test_extract_keywords_single_word(self, nltk_available):
        """Test with single word after stopword removal."""
        from insideLLMs.nlp.keyword_extraction import extract_keywords_textrank

        result = extract_keywords_textrank("python")
        # Single word with no co-occurrences may return empty
        assert isinstance(result, list)
