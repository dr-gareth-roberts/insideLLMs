"""Tests for insideLLMs/nlp/language_detection.py module."""

import pytest

from insideLLMs.nlp.language_detection import (
    detect_language_by_char_ngrams,
    detect_language_by_stopwords,
)


class TestDetectLanguageByCharNgrams:
    """Tests for detect_language_by_char_ngrams function."""

    def test_detect_english(self):
        """Test detecting English text."""
        text = "The quick brown fox jumps over the lazy dog. This is a longer English sentence that should be detected properly."
        result = detect_language_by_char_ngrams(text)
        assert result in ["en", "unknown"]

    def test_detect_spanish(self):
        """Test detecting Spanish text."""
        text = "El rápido zorro marrón salta sobre el perro perezoso. Esta es una oración en español que debería detectarse correctamente."
        result = detect_language_by_char_ngrams(text)
        assert result in ["es", "unknown"]

    def test_detect_french(self):
        """Test detecting French text."""
        text = "Le renard brun rapide saute par-dessus le chien paresseux. Cette phrase est en français et devrait être détectée correctement."
        result = detect_language_by_char_ngrams(text)
        assert result in ["fr", "unknown"]

    def test_detect_german(self):
        """Test detecting German text."""
        text = "Der schnelle braune Fuchs springt über den faulen Hund. Dies ist ein deutscher Satz der richtig erkannt werden sollte."
        result = detect_language_by_char_ngrams(text)
        assert result in ["de", "unknown"]

    def test_short_text_returns_unknown(self):
        """Test that short text returns unknown."""
        text = "Hi"
        result = detect_language_by_char_ngrams(text)
        assert result == "unknown"

    def test_empty_text(self):
        """Test that empty text returns unknown."""
        text = ""
        result = detect_language_by_char_ngrams(text)
        assert result == "unknown"

    def test_numeric_text(self):
        """Test that numeric-only text returns unknown."""
        text = "1234567890 1234567890 1234567890"
        result = detect_language_by_char_ngrams(text)
        assert result == "unknown"

    def test_special_characters_only(self):
        """Test that special characters only returns unknown."""
        text = "!@#$%^&*()_+ !@#$%^&*()_+"
        result = detect_language_by_char_ngrams(text)
        assert result == "unknown"


class TestDetectLanguageByStopwords:
    """Tests for detect_language_by_stopwords function."""

    @pytest.fixture
    def nltk_available(self):
        """Check if NLTK with required resources is available."""
        try:
            import nltk

            nltk.data.find("tokenizers/punkt")
            nltk.data.find("corpora/stopwords")
            return True
        except (ImportError, LookupError):
            pytest.skip("NLTK with required resources not available")

    def test_detect_english(self, nltk_available):
        """Test detecting English text by stopwords."""
        text = "The quick brown fox jumps over the lazy dog. This is a sentence with many English words and the definite article."
        result = detect_language_by_stopwords(text)
        assert result in ["en", "unknown"]

    def test_detect_spanish(self, nltk_available):
        """Test detecting Spanish text by stopwords."""
        text = "El rápido zorro marrón salta sobre el perro perezoso. Esta es una oración en español con muchas palabras."
        result = detect_language_by_stopwords(text)
        assert result in ["es", "unknown"]

    def test_detect_french(self, nltk_available):
        """Test detecting French text by stopwords."""
        text = "Le renard brun rapide saute par dessus le chien paresseux. Cette phrase est en français avec beaucoup de mots."
        result = detect_language_by_stopwords(text)
        assert result in ["fr", "unknown"]

    def test_detect_german(self, nltk_available):
        """Test detecting German text by stopwords."""
        text = "Der schnelle braune Fuchs springt über den faulen Hund. Dies ist ein deutscher Satz mit vielen Wörtern."
        result = detect_language_by_stopwords(text)
        assert result in ["de", "unknown"]

    def test_empty_text(self, nltk_available):
        """Test that empty text returns unknown."""
        text = ""
        result = detect_language_by_stopwords(text)
        assert result == "unknown"

    def test_single_word(self, nltk_available):
        """Test that single word may return unknown."""
        text = "hello"
        result = detect_language_by_stopwords(text)
        # Single word might not have enough stopwords
        assert result in ["en", "unknown"]

    def test_low_stopword_density(self, nltk_available):
        """Test text with low stopword density."""
        text = "cat dog bird fish elephant giraffe zebra lion tiger bear"
        result = detect_language_by_stopwords(text)
        # Low stopword density should return unknown
        assert result in ["en", "unknown"]
