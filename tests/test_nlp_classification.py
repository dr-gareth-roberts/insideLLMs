"""Tests for insideLLMs/nlp/classification.py module."""

import pytest


class TestCheckFunctions:
    """Tests for dependency check functions."""

    def test_check_nltk_available(self):
        """Test check_nltk when NLTK with VADER is available."""
        try:
            import nltk

            nltk.data.find("sentiment/vader_lexicon")
        except (ImportError, LookupError):
            pytest.skip("NLTK with VADER lexicon not available")

        from insideLLMs.nlp.classification import check_nltk

        check_nltk()  # Should not raise

    def test_check_sklearn_available(self):
        """Test check_sklearn when sklearn is available."""
        try:
            import sklearn  # noqa: F401
        except ImportError:
            pytest.skip("sklearn not available")

        from insideLLMs.nlp.classification import check_sklearn

        check_sklearn()  # Should not raise


class TestNaiveBayesClassify:
    """Tests for naive_bayes_classify function."""

    @pytest.fixture
    def sklearn_available(self):
        """Check if sklearn is available."""
        try:
            import sklearn  # noqa: F401

            return True
        except ImportError:
            pytest.skip("sklearn not available")

    def test_basic_classification(self, sklearn_available):
        """Test basic classification."""
        from insideLLMs.nlp.classification import naive_bayes_classify

        train_texts = [
            "I love this product",
            "This is great",
            "Amazing quality",
            "I hate this",
            "Terrible product",
            "Very bad",
        ]
        train_labels = ["positive", "positive", "positive", "negative", "negative", "negative"]
        test_texts = ["Great product", "Bad quality"]

        result = naive_bayes_classify(train_texts, train_labels, test_texts)

        assert len(result) == 2
        assert all(label in ["positive", "negative"] for label in result)

    def test_returns_list(self, sklearn_available):
        """Test that result is a list."""
        from insideLLMs.nlp.classification import naive_bayes_classify

        train_texts = ["good", "bad"]
        train_labels = ["pos", "neg"]
        test_texts = ["good"]

        result = naive_bayes_classify(train_texts, train_labels, test_texts)

        assert isinstance(result, list)


class TestSvmClassify:
    """Tests for svm_classify function."""

    @pytest.fixture
    def sklearn_available(self):
        """Check if sklearn is available."""
        try:
            import sklearn  # noqa: F401

            return True
        except ImportError:
            pytest.skip("sklearn not available")

    def test_basic_classification(self, sklearn_available):
        """Test basic SVM classification."""
        from insideLLMs.nlp.classification import svm_classify

        train_texts = [
            "I love this product",
            "This is great",
            "Amazing quality",
            "I hate this",
            "Terrible product",
            "Very bad",
        ]
        train_labels = ["positive", "positive", "positive", "negative", "negative", "negative"]
        test_texts = ["Great product", "Bad quality"]

        result = svm_classify(train_texts, train_labels, test_texts)

        assert len(result) == 2
        assert all(label in ["positive", "negative"] for label in result)

    def test_returns_list(self, sklearn_available):
        """Test that result is a list."""
        from insideLLMs.nlp.classification import svm_classify

        train_texts = ["good", "bad"]
        train_labels = ["pos", "neg"]
        test_texts = ["good"]

        result = svm_classify(train_texts, train_labels, test_texts)

        assert isinstance(result, list)


class TestSentimentAnalysisBasic:
    """Tests for sentiment_analysis_basic function."""

    @pytest.fixture
    def nltk_available(self):
        """Check if NLTK with VADER is available."""
        try:
            import nltk

            nltk.data.find("sentiment/vader_lexicon")
            return True
        except (ImportError, LookupError):
            pytest.skip("NLTK with VADER lexicon not available")

    def test_positive_sentiment(self, nltk_available):
        """Test detecting positive sentiment."""
        from insideLLMs.nlp.classification import sentiment_analysis_basic

        result = sentiment_analysis_basic("I love this! It's amazing and wonderful!")
        assert result == "positive"

    def test_negative_sentiment(self, nltk_available):
        """Test detecting negative sentiment."""
        from insideLLMs.nlp.classification import sentiment_analysis_basic

        result = sentiment_analysis_basic("I hate this! It's terrible and awful!")
        assert result == "negative"

    def test_neutral_sentiment(self, nltk_available):
        """Test detecting neutral sentiment."""
        from insideLLMs.nlp.classification import sentiment_analysis_basic

        result = sentiment_analysis_basic("The cat is on the table.")
        assert result == "neutral"

    def test_returns_valid_sentiment(self, nltk_available):
        """Test that result is a valid sentiment."""
        from insideLLMs.nlp.classification import sentiment_analysis_basic

        result = sentiment_analysis_basic("Some text here.")
        assert result in ["positive", "negative", "neutral"]
