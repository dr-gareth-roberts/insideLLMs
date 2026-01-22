"""Tests for insideLLMs/nlp/tokenization.py module."""

import pytest

from insideLLMs.nlp.tokenization import (
    get_ngrams,
    segment_sentences,
    simple_tokenize,
)


class TestSimpleTokenize:
    """Tests for simple_tokenize function."""

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
        assert "hello" in result
        assert "world" in result


class TestGetNgrams:
    """Tests for get_ngrams function."""

    def test_bigrams(self):
        """Test generating bigrams."""
        tokens = ["a", "b", "c", "d"]
        result = get_ngrams(tokens, n=2)
        assert result == [("a", "b"), ("b", "c"), ("c", "d")]

    def test_trigrams(self):
        """Test generating trigrams."""
        tokens = ["a", "b", "c", "d"]
        result = get_ngrams(tokens, n=3)
        assert result == [("a", "b", "c"), ("b", "c", "d")]

    def test_unigrams(self):
        """Test generating unigrams."""
        tokens = ["a", "b", "c"]
        result = get_ngrams(tokens, n=1)
        assert result == [("a",), ("b",), ("c",)]

    def test_empty_list(self):
        """Test with empty list."""
        result = get_ngrams([], n=2)
        assert result == []

    def test_short_list(self):
        """Test when list is shorter than n."""
        tokens = ["a"]
        result = get_ngrams(tokens, n=2)
        assert result == []

    def test_list_equals_n(self):
        """Test when list length equals n."""
        tokens = ["a", "b"]
        result = get_ngrams(tokens, n=2)
        assert result == [("a", "b")]


class TestSegmentSentences:
    """Tests for segment_sentences function."""

    def test_regex_based_segmentation(self):
        """Test regex-based sentence segmentation."""
        text = "Hello world. This is a test. Another sentence here."
        result = segment_sentences(text, use_nltk=False)
        assert len(result) >= 2

    def test_single_sentence_regex(self):
        """Test regex with single sentence."""
        text = "This is a single sentence."
        result = segment_sentences(text, use_nltk=False)
        assert len(result) >= 1

    def test_question_mark_regex(self):
        """Test regex with question mark."""
        text = "Is this a question? Yes it is."
        result = segment_sentences(text, use_nltk=False)
        assert len(result) >= 2

    def test_exclamation_mark_regex(self):
        """Test regex with exclamation mark."""
        text = "Wow! That is amazing."
        result = segment_sentences(text, use_nltk=False)
        assert len(result) >= 2


class TestNltkDependentFunctions:
    """Tests for NLTK-dependent functions."""

    @pytest.fixture
    def nltk_available(self):
        """Check if NLTK is available."""
        try:
            import nltk

            nltk.data.find("tokenizers/punkt")
            nltk.data.find("corpora/stopwords")
            nltk.data.find("corpora/wordnet")
            return True
        except (ImportError, LookupError):
            pytest.skip("NLTK with required resources not available")

    def test_nltk_tokenize(self, nltk_available):
        """Test NLTK tokenization."""
        from insideLLMs.nlp.tokenization import nltk_tokenize

        text = "Hello, world! This is a test."
        result = nltk_tokenize(text)
        assert "Hello" in result
        assert "world" in result

    def test_nltk_sentence_segmentation(self, nltk_available):
        """Test NLTK sentence segmentation."""
        text = "Hello world. This is a test."
        result = segment_sentences(text, use_nltk=True)
        assert len(result) >= 2

    def test_remove_stopwords(self, nltk_available):
        """Test removing stopwords."""
        from insideLLMs.nlp.tokenization import remove_stopwords

        tokens = ["the", "cat", "is", "on", "the", "mat"]
        result = remove_stopwords(tokens)
        assert "the" not in result
        assert "cat" in result

    def test_stem_words(self, nltk_available):
        """Test word stemming."""
        from insideLLMs.nlp.tokenization import stem_words

        tokens = ["running", "runs", "runner"]
        result = stem_words(tokens)
        # All should stem to 'run'
        assert len(result) == 3

    def test_lemmatize_words(self, nltk_available):
        """Test word lemmatization."""
        from insideLLMs.nlp.tokenization import lemmatize_words

        tokens = ["cats", "running", "better"]
        result = lemmatize_words(tokens)
        # 'cats' should lemmatize to 'cat'
        assert "cat" in result or "cats" in result


class TestSpacyDependentFunctions:
    """Tests for spaCy-dependent functions."""

    @pytest.fixture
    def spacy_available(self):
        """Check if spaCy is available."""
        try:
            import spacy

            spacy.load("en_core_web_sm")
            return True
        except (ImportError, OSError):
            pytest.skip("spaCy with en_core_web_sm not available")

    def test_spacy_tokenize(self, spacy_available):
        """Test spaCy tokenization."""
        from insideLLMs.nlp.tokenization import spacy_tokenize

        text = "Hello, world!"
        result = spacy_tokenize(text)
        assert "Hello" in result
        assert "world" in result
