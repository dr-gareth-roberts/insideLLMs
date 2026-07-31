"""Un-omit path: chunking / tokenization / classification with mocks."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.nlp import chunking, classification, tokenization


def test_chunking_all_strategies_and_errors() -> None:
    assert chunking.split_by_char_count("", 5) == []
    assert chunking.split_by_char_count("hi", 10) == ["hi"]
    assert chunking.split_by_char_count("abcdefghij", 4, overlap=2)
    with pytest.raises(ValueError, match="positive"):
        chunking.split_by_char_count("x", 0)
    with pytest.raises(ValueError, match="overlap"):
        chunking.split_by_char_count("x", 2, overlap=2)

    assert chunking.split_by_word_count("", 2) == []
    assert chunking.split_by_word_count("a b c d", 2, overlap=1)
    with pytest.raises(ValueError):
        chunking.split_by_word_count("a", 0)
    with pytest.raises(ValueError):
        chunking.split_by_word_count("a b", 1, overlap=1)

    assert chunking.split_by_sentence(
        "One. Two. Three.", sentences_per_chunk=2, use_nltk_for_segmentation=False
    )
    # empty / no sentence boundary → may yield ['']
    assert chunking.split_by_sentence("", 1, use_nltk_for_segmentation=False) in ([], [""])
    with pytest.raises(ValueError):
        chunking.split_by_sentence("A.", 0, use_nltk_for_segmentation=False)
    with pytest.raises(ValueError):
        chunking.split_by_sentence("A. B.", 1, overlap=1, use_nltk_for_segmentation=False)

    with patch.object(
        chunking, "segment_sentences_internal_for_chunking", return_value=["S1.", "S2."]
    ):
        out = chunking.split_by_sentence("ignored", 1, use_nltk_for_segmentation=True)
        assert out == ["S1.", "S2."]
    with patch.object(chunking, "segment_sentences_internal_for_chunking", return_value=[]):
        assert chunking.split_by_sentence("x", 1, use_nltk_for_segmentation=False) == []

    assert chunking.sliding_window_chunks("a b c d e", 3, 2)
    assert chunking.sliding_window_chunks("short", 10, 1) == ["short"]
    assert chunking.sliding_window_chunks("", 2, 1) == []
    with pytest.raises(ValueError):
        chunking.sliding_window_chunks("a", 0, 1)
    with pytest.raises(ValueError):
        chunking.sliding_window_chunks("a", 1, 0)

    with patch.object(chunking, "ensure_nltk") as ens:
        chunking.check_nltk()
        ens.assert_called()


def _install_nltk_stubs(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    nltk = types.ModuleType("nltk")
    tokenize = types.ModuleType("nltk.tokenize")
    corpus = types.ModuleType("nltk.corpus")
    stem = types.ModuleType("nltk.stem")
    sentiment = types.ModuleType("nltk.sentiment")
    vader = types.ModuleType("nltk.sentiment.vader")

    tokenize.word_tokenize = MagicMock(return_value=["Hello", ","])
    tokenize.wordpunct_tokenize = MagicMock(return_value=["Hi", "!"])
    tokenize.sent_tokenize = MagicMock(return_value=["A.", "B."])

    stopwords = MagicMock()
    stopwords.words.return_value = ["the", "a"]
    corpus.stopwords = stopwords

    class PorterStemmer:
        def stem(self, token):
            return token[:3]

    class WordNetLemmatizer:
        def lemmatize(self, token):
            return token.rstrip("s")

    stem.PorterStemmer = PorterStemmer
    stem.WordNetLemmatizer = WordNetLemmatizer

    class SentimentIntensityAnalyzer:
        def polarity_scores(self, text):
            t = (text or "").lower()
            if "bad" in t or "hate" in t:
                return {"compound": -0.8}
            if "good" in t or "love" in t:
                return {"compound": 0.8}
            return {"compound": 0.0}

    sentiment.SentimentIntensityAnalyzer = SentimentIntensityAnalyzer
    vader.SentimentIntensityAnalyzer = SentimentIntensityAnalyzer
    nltk.tokenize = tokenize
    nltk.corpus = corpus
    nltk.stem = stem
    nltk.sentiment = sentiment

    for name, mod in [
        ("nltk", nltk),
        ("nltk.tokenize", tokenize),
        ("nltk.corpus", corpus),
        ("nltk.stem", stem),
        ("nltk.sentiment", sentiment),
        ("nltk.sentiment.vader", vader),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    return tokenize


def test_tokenization_regex_ngrams_and_mocked_nltk_spacy(monkeypatch: pytest.MonkeyPatch) -> None:
    assert tokenization.word_tokenize_regex("Hello, World!", lowercase=True) == ["hello", "world"]
    assert tokenization.word_tokenize_regex("Hello, World!", lowercase=False) == ["Hello", "World"]
    assert tokenization.get_ngrams(["a", "b", "c"], 2)
    assert tokenization.segment_sentences("Hi. Bye.", use_nltk=False)

    tok = _install_nltk_stubs(monkeypatch)
    with patch.object(tokenization, "_ensure_nltk_tokenization"):
        assert tokenization.nltk_tokenize("Hello,") == ["Hello", ","]

        tok.word_tokenize.side_effect = LookupError("no punkt")
        assert tokenization.nltk_tokenize("Hi!") == ["Hi", "!"]
        tok.word_tokenize.side_effect = None
        tok.word_tokenize.return_value = ["Hello", ","]

        assert tokenization.segment_sentences("A. B.", use_nltk=True) == ["A.", "B."]
        tok.sent_tokenize.side_effect = LookupError("x")
        out = tokenization.segment_sentences("A. B.", use_nltk=True)
        assert isinstance(out, list)
        tok.sent_tokenize.side_effect = None
        tok.sent_tokenize.return_value = ["A.", "B."]

        assert tokenization.remove_stopwords(["the", "cat", "a"]) == ["cat"]
        assert tokenization.stem_words(["running", "jumps"]) == ["run", "jum"]
        assert tokenization.lemmatize_words(["cats", "dogs"]) == ["cat", "dog"]

        # stopwords LookupError → RuntimeError
        from nltk.corpus import stopwords as sw

        sw.words.side_effect = LookupError("missing")
        with pytest.raises(RuntimeError, match="stopwords"):
            tokenization.remove_stopwords(["the"])
        sw.words.side_effect = None
        sw.words.return_value = ["the", "a"]

    # Cover _ensure_nltk_tokenization body (calls ensure_nltk)
    with patch.object(tokenization, "ensure_nltk") as ens:
        tokenization._ensure_nltk_tokenization()
        ens.assert_called()

    # lemmatize LookupError path
    with patch.object(tokenization, "_ensure_nltk_tokenization"):
        lem = MagicMock()
        lem.lemmatize.side_effect = LookupError("no wordnet")
        with patch("nltk.stem.WordNetLemmatizer", return_value=lem):
            with pytest.raises(RuntimeError, match="WordNet"):
                tokenization.lemmatize_words(["cats"])

    nlp = MagicMock(return_value=[SimpleNamespace(text="Hello"), SimpleNamespace(text="!")])
    with patch.object(tokenization, "ensure_spacy", return_value=nlp):
        assert tokenization.spacy_tokenize("Hello!") == ["Hello", "!"]


def test_classification_mocked_sklearn_and_vader(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_nltk_stubs(monkeypatch)

    sklearn = types.ModuleType("sklearn")
    pipeline_mod = types.ModuleType("sklearn.pipeline")
    fe = types.ModuleType("sklearn.feature_extraction")
    text_mod = types.ModuleType("sklearn.feature_extraction.text")
    nb_mod = types.ModuleType("sklearn.naive_bayes")
    svm_mod = types.ModuleType("sklearn.svm")

    pipe = MagicMock()
    pipe.fit.return_value = pipe
    pipe.predict.return_value = MagicMock(tolist=lambda: ["spam", "ham"])

    class Pipeline:
        def __init__(self, steps):
            self.steps = steps

        def fit(self, *a, **k):
            return pipe.fit(*a, **k)

        def predict(self, *a, **k):
            return pipe.predict(*a, **k)

    pipeline_mod.Pipeline = Pipeline
    text_mod.CountVectorizer = MagicMock
    text_mod.TfidfVectorizer = MagicMock
    nb_mod.MultinomialNB = MagicMock
    svm_mod.LinearSVC = MagicMock
    fe.text = text_mod
    sklearn.pipeline = pipeline_mod
    sklearn.feature_extraction = fe
    sklearn.naive_bayes = nb_mod
    sklearn.svm = svm_mod

    for name, mod in [
        ("sklearn", sklearn),
        ("sklearn.pipeline", pipeline_mod),
        ("sklearn.feature_extraction", fe),
        ("sklearn.feature_extraction.text", text_mod),
        ("sklearn.naive_bayes", nb_mod),
        ("sklearn.svm", svm_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    with patch.object(classification, "ensure_sklearn"):
        out = classification.naive_bayes_classify(
            ["buy now", "hello"],
            ["spam", "ham"],
            ["win prize", "meeting"],
        )
        assert out == ["spam", "ham"]
        out = classification.svm_classify(
            ["buy now", "hello"],
            ["spam", "ham"],
            ["win prize", "meeting"],
        )
        assert out == ["spam", "ham"]

    with patch.object(classification, "_ensure_vader"):
        assert classification.sentiment_analysis_basic("good") == "positive"
        assert classification.sentiment_analysis_basic("bad") == "negative"
        assert classification.sentiment_analysis_basic("meh") == "neutral"

    assert callable(classification.check_sklearn)
    with patch.object(classification, "ensure_nltk", return_value=MagicMock()) as ens:
        classification.check_nltk()
        ens.assert_called()
