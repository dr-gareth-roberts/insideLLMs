"""Tests for insideLLMs/nlp/dependencies.py module."""

import pytest


class TestEnsureNltk:
    """Tests for ensure_nltk function."""

    def test_nltk_available(self):
        """Test when NLTK is available."""
        try:
            import nltk  # noqa: F401
        except ImportError:
            pytest.skip("NLTK not available")

        from insideLLMs.nlp.dependencies import ensure_nltk
        ensure_nltk.cache_clear()
        result = ensure_nltk(())
        # Should return nltk module
        assert result is not None

    def test_nltk_with_punkt_resource(self):
        """Test ensuring NLTK with punkt tokenizer."""
        try:
            import nltk
            nltk.data.find("tokenizers/punkt")
        except (ImportError, LookupError):
            pytest.skip("NLTK with punkt not available")

        from insideLLMs.nlp.dependencies import ensure_nltk
        ensure_nltk.cache_clear()
        result = ensure_nltk(("tokenizers/punkt",))
        assert result is not None

    def test_nltk_with_multiple_resources(self):
        """Test ensuring NLTK with multiple resources."""
        try:
            import nltk
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("corpora/stopwords")
        except (ImportError, LookupError):
            pytest.skip("NLTK resources not available")

        from insideLLMs.nlp.dependencies import ensure_nltk
        ensure_nltk.cache_clear()
        result = ensure_nltk(("tokenizers/punkt", "corpora/stopwords"))
        assert result is not None


class TestEnsureSpacy:
    """Tests for ensure_spacy function."""

    def test_spacy_model_not_found(self):
        """Test ImportError when spaCy model not found."""
        try:
            import spacy  # noqa: F401
        except ImportError:
            pytest.skip("spaCy not installed")

        from insideLLMs.nlp.dependencies import ensure_spacy
        ensure_spacy.cache_clear()

        with pytest.raises(ImportError, match="model.*not found"):
            ensure_spacy("nonexistent_model_xyz_12345")

    def test_spacy_available_with_model(self):
        """Test when spaCy and model are available."""
        try:
            import spacy
            spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            pytest.skip("spaCy or en_core_web_sm not available")

        from insideLLMs.nlp.dependencies import ensure_spacy
        ensure_spacy.cache_clear()
        result = ensure_spacy("en_core_web_sm")
        assert result is not None


class TestEnsureSklearn:
    """Tests for ensure_sklearn function."""

    def test_sklearn_available(self):
        """Test when sklearn is available."""
        try:
            import sklearn  # noqa: F401
        except ImportError:
            pytest.skip("sklearn not available")

        from insideLLMs.nlp.dependencies import ensure_sklearn
        ensure_sklearn.cache_clear()
        result = ensure_sklearn()
        assert result is None  # Returns None on success


class TestEnsureGensim:
    """Tests for ensure_gensim function."""

    def test_gensim_available(self):
        """Test when gensim is available."""
        try:
            import gensim  # noqa: F401
        except ImportError:
            pytest.skip("gensim not available")

        from insideLLMs.nlp.dependencies import ensure_gensim
        ensure_gensim.cache_clear()
        result = ensure_gensim()
        assert result is None  # Returns None on success
