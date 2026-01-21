"""Tests for insideLLMs/nlp/feature_extraction.py module."""

import pytest


class TestCheckFunctions:
    """Tests for dependency check functions."""

    def test_check_sklearn_available(self):
        """Test check_sklearn when sklearn is available."""
        try:
            import sklearn  # noqa: F401
        except ImportError:
            pytest.skip("sklearn not available")

        from insideLLMs.nlp.feature_extraction import check_sklearn
        check_sklearn()  # Should not raise

    def test_check_gensim_available(self):
        """Test check_gensim when gensim is available."""
        try:
            import gensim  # noqa: F401
        except ImportError:
            pytest.skip("gensim not available")

        from insideLLMs.nlp.feature_extraction import check_gensim
        check_gensim()  # Should not raise

    def test_check_spacy_available(self):
        """Test check_spacy when spaCy is available."""
        try:
            import spacy
            spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            pytest.skip("spaCy or en_core_web_sm not available")

        from insideLLMs.nlp.feature_extraction import check_spacy
        result = check_spacy()
        assert result is not None


class TestCreateBow:
    """Tests for create_bow function."""

    @pytest.fixture
    def sklearn_available(self):
        """Check if sklearn is available."""
        try:
            import sklearn  # noqa: F401
            return True
        except ImportError:
            pytest.skip("sklearn not available")

    def test_create_bow_basic(self, sklearn_available):
        """Test basic bag-of-words creation."""
        from insideLLMs.nlp.feature_extraction import create_bow

        texts = ["hello world", "world foo", "hello foo bar"]
        vectors, features = create_bow(texts)

        assert len(vectors) == 3
        assert len(features) > 0
        assert "hello" in features
        assert "world" in features

    def test_create_bow_max_features(self, sklearn_available):
        """Test bag-of-words with max_features."""
        from insideLLMs.nlp.feature_extraction import create_bow

        texts = ["a b c d e f g h i j", "a b c d"]
        vectors, features = create_bow(texts, max_features=5)

        assert len(features) <= 5


class TestCreateTfidf:
    """Tests for create_tfidf function."""

    @pytest.fixture
    def sklearn_available(self):
        """Check if sklearn is available."""
        try:
            import sklearn  # noqa: F401
            return True
        except ImportError:
            pytest.skip("sklearn not available")

    def test_create_tfidf_basic(self, sklearn_available):
        """Test basic TF-IDF creation."""
        from insideLLMs.nlp.feature_extraction import create_tfidf

        texts = ["hello world", "world foo", "hello foo bar"]
        vectors, features = create_tfidf(texts)

        assert len(vectors) == 3
        assert len(features) > 0
        # TF-IDF values should be floats
        assert all(isinstance(v, float) for row in vectors for v in row)

    def test_create_tfidf_max_features(self, sklearn_available):
        """Test TF-IDF with max_features."""
        from insideLLMs.nlp.feature_extraction import create_tfidf

        texts = ["a b c d e f g h i j", "a b c d"]
        vectors, features = create_tfidf(texts, max_features=5)

        assert len(features) <= 5


class TestCreateWordEmbeddings:
    """Tests for create_word_embeddings function."""

    @pytest.fixture
    def gensim_available(self):
        """Check if gensim is available."""
        try:
            import gensim  # noqa: F401
            return True
        except ImportError:
            pytest.skip("gensim not available")

    def test_create_word_embeddings_basic(self, gensim_available):
        """Test basic word embeddings creation."""
        from insideLLMs.nlp.feature_extraction import create_word_embeddings

        sentences = [
            ["hello", "world", "foo"],
            ["hello", "bar", "baz"],
            ["world", "foo", "bar"],
        ]
        embeddings = create_word_embeddings(sentences, vector_size=10)

        assert isinstance(embeddings, dict)
        assert len(embeddings) > 0
        # Each embedding should be a list of floats
        for word, vec in embeddings.items():
            assert len(vec) == 10

    def test_create_word_embeddings_custom_params(self, gensim_available):
        """Test word embeddings with custom parameters."""
        from insideLLMs.nlp.feature_extraction import create_word_embeddings

        sentences = [["a", "b", "c"], ["a", "d", "e"], ["b", "c", "d"]]
        embeddings = create_word_embeddings(
            sentences, vector_size=50, window=3, min_count=1
        )

        for vec in embeddings.values():
            assert len(vec) == 50


class TestExtractPosTags:
    """Tests for extract_pos_tags function."""

    @pytest.fixture
    def spacy_available(self):
        """Check if spaCy and model are available."""
        try:
            import spacy  # noqa: F401
            # Try to load the model
            spacy.load("en_core_web_sm")
            return True
        except (ImportError, OSError):
            pytest.skip("spaCy or en_core_web_sm not available")

    def test_extract_pos_tags_basic(self, spacy_available):
        """Test basic POS tagging."""
        from insideLLMs.nlp.feature_extraction import extract_pos_tags

        result = extract_pos_tags("The cat sat on the mat.")

        assert isinstance(result, list)
        assert len(result) > 0
        # Each result should be (word, pos_tag)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2


class TestExtractDependencies:
    """Tests for extract_dependencies function."""

    @pytest.fixture
    def spacy_available(self):
        """Check if spaCy and model are available."""
        try:
            import spacy  # noqa: F401
            spacy.load("en_core_web_sm")
            return True
        except (ImportError, OSError):
            pytest.skip("spaCy or en_core_web_sm not available")

    def test_extract_dependencies_basic(self, spacy_available):
        """Test basic dependency extraction."""
        from insideLLMs.nlp.feature_extraction import extract_dependencies

        result = extract_dependencies("The cat sat on the mat.")

        assert isinstance(result, list)
        assert len(result) > 0
        # Each result should be (word, dep_relation, head)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 3
