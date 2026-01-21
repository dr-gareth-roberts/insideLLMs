"""Tests for insideLLMs/nlp/classification.py module."""

import pytest


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

    def test_classify_basic(self, sklearn_available):
        """Test basic Naive Bayes classification."""
        from insideLLMs.nlp.classification import naive_bayes_classify

        train_texts = [
            "I love this movie",
            "This film is great",
            "Excellent movie experience",
            "I hate this film",
            "Terrible movie",
            "Worst film ever",
        ]
        train_labels = ["positive", "positive", "positive", "negative", "negative", "negative"]
        test_texts = ["I love it", "I hate it"]

        result = naive_bayes_classify(train_texts, train_labels, test_texts)

        assert len(result) == 2
        assert all(label in ["positive", "negative"] for label in result)

    def test_classify_returns_list(self, sklearn_available):
        """Test that result is a list."""
        from insideLLMs.nlp.classification import naive_bayes_classify

        train_texts = ["hello", "world", "foo", "bar"]
        train_labels = ["a", "a", "b", "b"]
        test_texts = ["hello world"]

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

    def test_classify_basic(self, sklearn_available):
        """Test basic SVM classification."""
        from insideLLMs.nlp.classification import svm_classify

        train_texts = [
            "sports game football soccer",
            "basketball tennis baseball",
            "running swimming athletics",
            "politics government election",
            "president senate congress",
            "voting democracy policy",
        ]
        train_labels = ["sports", "sports", "sports", "politics", "politics", "politics"]
        test_texts = ["football match", "election results"]

        result = svm_classify(train_texts, train_labels, test_texts)

        assert len(result) == 2
        assert all(label in ["sports", "politics"] for label in result)

    def test_classify_multiple_classes(self, sklearn_available):
        """Test SVM with multiple classes."""
        from insideLLMs.nlp.classification import svm_classify

        train_texts = ["cat dog pet", "car truck vehicle", "apple banana fruit"] * 3
        train_labels = ["animals", "vehicles", "food"] * 3
        test_texts = ["dog", "car", "banana"]

        result = svm_classify(train_texts, train_labels, test_texts)

        assert len(result) == 3


class TestSentimentAnalysisBasic:
    """Tests for sentiment_analysis_basic function."""

    @pytest.fixture
    def nltk_available(self):
        """Check if NLTK is available."""
        try:
            import nltk  # noqa: F401
            return True
        except ImportError:
            pytest.skip("NLTK not available")

    def test_positive_sentiment(self, nltk_available):
        """Test positive sentiment detection."""
        from insideLLMs.nlp.classification import sentiment_analysis_basic

        result = sentiment_analysis_basic("I love this! It's amazing and wonderful!")
        assert result == "positive"

    def test_negative_sentiment(self, nltk_available):
        """Test negative sentiment detection."""
        from insideLLMs.nlp.classification import sentiment_analysis_basic

        result = sentiment_analysis_basic("I hate this! It's terrible and awful!")
        assert result == "negative"

    def test_neutral_sentiment(self, nltk_available):
        """Test neutral sentiment detection."""
        from insideLLMs.nlp.classification import sentiment_analysis_basic

        result = sentiment_analysis_basic("The meeting is at 3pm.")
        assert result == "neutral"

    def test_returns_string(self, nltk_available):
        """Test that result is a string."""
        from insideLLMs.nlp.classification import sentiment_analysis_basic

        result = sentiment_analysis_basic("Some text here")
        assert isinstance(result, str)
        assert result in ["positive", "negative", "neutral"]
