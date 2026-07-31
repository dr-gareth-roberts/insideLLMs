"""Un-omit path: dependencies, feature_extraction, keyword_extraction (mocked SDKs)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def test_dependencies_ensure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    from insideLLMs.nlp import dependencies as deps

    # clear caches
    deps.ensure_nltk.cache_clear()
    deps.ensure_spacy.cache_clear()
    deps.ensure_sklearn.cache_clear()
    deps.ensure_gensim.cache_clear()

    # ImportError nltk
    real_import = __import__

    def block_nltk(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "nltk" or name.startswith("nltk."):
            raise ImportError("no nltk")
        return real_import(name, globals, locals, fromlist, level)

    deps._ensure_nltk_cached.cache_clear()
    with patch("builtins.__import__", side_effect=block_nltk):
        with pytest.raises(ImportError, match="NLTK"):
            deps._ensure_nltk_cached(())

    # success path with mocked nltk + punkt expansion + NLTK_DATA
    nltk = types.ModuleType("nltk")
    nltk_data = types.SimpleNamespace(path=[], find=MagicMock(side_effect=LookupError("missing")))
    nltk.data = nltk_data
    nltk.download = MagicMock()
    monkeypatch.setitem(sys.modules, "nltk", nltk)
    monkeypatch.setenv("NLTK_DATA", "/tmp/fake_nltk_data")
    deps._ensure_nltk_cached.cache_clear()
    # find succeeds after first miss for second resource
    calls = {"n": 0}

    def find(res):
        calls["n"] += 1
        if "punkt_tab" in res or calls["n"] > 2:
            return True
        raise LookupError(res)

    nltk_data.find = find
    out = deps._ensure_nltk_cached(("tokenizers/punkt",))
    assert out is nltk
    assert nltk.download.called
    assert deps.ensure_nltk(("tokenizers/punkt",)) is nltk

    # spacy ImportError + OSError model missing
    deps.ensure_spacy.cache_clear()

    def block_spacy(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "spacy":
            raise ImportError("no spacy")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=block_spacy):
        with pytest.raises(ImportError, match="spaCy is not installed"):
            deps.ensure_spacy("en_core_web_sm")

    spacy = types.ModuleType("spacy")
    spacy.load = MagicMock(side_effect=OSError("missing model"))
    monkeypatch.setitem(sys.modules, "spacy", spacy)
    deps.ensure_spacy.cache_clear()
    with pytest.raises(ImportError, match="not found"):
        deps.ensure_spacy("en_core_web_sm")

    spacy.load = MagicMock(return_value="NLP")
    deps.ensure_spacy.cache_clear()
    assert deps.ensure_spacy("en_core_web_sm") == "NLP"

    # sklearn / gensim
    deps.ensure_sklearn.cache_clear()
    sklearn = types.ModuleType("sklearn")
    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    assert deps.ensure_sklearn() is None

    def block_sk(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ImportError("no")
        return real_import(name, globals, locals, fromlist, level)

    deps.ensure_sklearn.cache_clear()
    # remove from cache modules temporarily
    saved = sys.modules.pop("sklearn", None)
    with patch("builtins.__import__", side_effect=block_sk):
        with pytest.raises(ImportError, match="scikit-learn"):
            deps.ensure_sklearn()
    if saved is not None:
        sys.modules["sklearn"] = saved

    deps.ensure_gensim.cache_clear()
    gensim = types.ModuleType("gensim")
    monkeypatch.setitem(sys.modules, "gensim", gensim)
    assert deps.ensure_gensim() is None

    def block_g(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "gensim" or name.startswith("gensim."):
            raise ImportError("no")
        return real_import(name, globals, locals, fromlist, level)

    deps.ensure_gensim.cache_clear()
    saved_g = sys.modules.pop("gensim", None)
    with patch("builtins.__import__", side_effect=block_g):
        with pytest.raises(ImportError, match="gensim"):
            deps.ensure_gensim()
    if saved_g is not None:
        sys.modules["gensim"] = saved_g


def test_feature_extraction_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    from insideLLMs.nlp import feature_extraction as fe

    # bow / tfidf via sklearn stubs
    class Mat:
        def __init__(self, rows):
            self.rows = rows

        def toarray(self):
            return MagicMock(tolist=lambda: self.rows)

    class FeatNames(list):
        def tolist(self):
            return list(self)

    class CV:
        def __init__(self, **k):
            pass

        def fit_transform(self, texts):
            return Mat([[1, 0], [0, 1]])

        def get_feature_names_out(self):
            return FeatNames(["a", "b"])

    class TV(CV):
        pass

    sk = types.ModuleType("sklearn")
    sk_fe = types.ModuleType("sklearn.feature_extraction")
    sk_text = types.ModuleType("sklearn.feature_extraction.text")
    sk_text.CountVectorizer = CV
    sk_text.TfidfVectorizer = TV
    sk.feature_extraction = sk_fe
    sk_fe.text = sk_text
    for name, mod in [
        ("sklearn", sk),
        ("sklearn.feature_extraction", sk_fe),
        ("sklearn.feature_extraction.text", sk_text),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    with patch.object(fe, "ensure_sklearn"):
        vectors, vocab = fe.create_bow(["a", "b"])
        assert vocab == ["a", "b"]
        vectors2, vocab2 = fe.create_tfidf(["a", "b"])
        assert vocab2 == ["a", "b"]

    # word embeddings via gensim stub
    class WV:
        index_to_key = ["cat"]

        def __getitem__(self, w):
            return MagicMock(tolist=lambda: [0.1, 0.2])

    class W2V:
        def __init__(self, *a, **k):
            self.wv = WV()

    gensim = types.ModuleType("gensim")
    models = types.ModuleType("gensim.models")
    models.Word2Vec = W2V
    gensim.models = models
    monkeypatch.setitem(sys.modules, "gensim", gensim)
    monkeypatch.setitem(sys.modules, "gensim.models", models)
    with patch.object(fe, "ensure_gensim"):
        emb = fe.create_word_embeddings([["cat", "dog"]], vector_size=2)
        assert "cat" in emb

    # pos + dependencies via spaCy stub
    tok = MagicMock()
    tok.text = "Hello"
    tok.pos_ = "INTJ"
    tok.dep_ = "ROOT"
    tok.head = tok
    doc = MagicMock()
    doc.__iter__ = lambda self: iter([tok])
    nlp = MagicMock(return_value=doc)
    with patch.object(fe, "ensure_spacy", return_value=nlp):
        tags = fe.extract_pos_tags("Hello")
        assert tags
        deps_out = fe.extract_dependencies("Hello")
        assert deps_out


def test_keyword_extraction_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    from insideLLMs.nlp import keyword_extraction as ke

    with patch.object(ke, "ensure_nltk"):
        ke._ensure_nltk_keyword()

    class Mat:
        def sum(self, axis=0):
            return MagicMock(A1=[0.5, 0.9, 0.1])

    class TV:
        def __init__(self, **k):
            pass

        def fit_transform(self, sentences):
            return Mat()

        def get_feature_names_out(self):
            arr = MagicMock()
            arr.any.return_value = True
            arr.__iter__ = lambda self: iter(["alpha", "beta", "gamma"])
            return arr

    sk = types.ModuleType("sklearn")
    sk_fe = types.ModuleType("sklearn.feature_extraction")
    sk_text = types.ModuleType("sklearn.feature_extraction.text")
    sk_text.TfidfVectorizer = TV
    sk.feature_extraction = sk_fe
    sk_fe.text = sk_text
    for name, mod in [
        ("sklearn", sk),
        ("sklearn.feature_extraction", sk_fe),
        ("sklearn.feature_extraction.text", sk_text),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    with patch.object(ke, "ensure_sklearn"):
        with patch.object(ke, "segment_sentences_for_keyword", return_value=[]):
            assert ke.extract_keywords_tfidf("x") == []
        with patch.object(
            ke, "segment_sentences_for_keyword", return_value=["hello world", "world peace"]
        ):
            kws = ke.extract_keywords_tfidf("hello world. world peace.", num_keywords=2)
            assert kws[:1] == ["beta"] or len(kws) == 2

        # ValueError from vectorizer
        class BadTV(TV):
            def fit_transform(self, s):
                raise ValueError("empty")

        sk_text.TfidfVectorizer = BadTV
        with patch.object(ke, "segment_sentences_for_keyword", return_value=["hi"]):
            assert ke.extract_keywords_tfidf("hi") == []

        # empty feature_names → []
        class EmptyTV(TV):
            def fit_transform(self, s):
                return Mat()

            def get_feature_names_out(self):
                arr = MagicMock()
                arr.any.return_value = False
                return arr

        sk_text.TfidfVectorizer = EmptyTV
        with patch.object(ke, "segment_sentences_for_keyword", return_value=["hi there"]):
            assert ke.extract_keywords_tfidf("hi there") == []

    # textrank with stubs
    with patch.object(ke, "_ensure_nltk_keyword"):
        with patch.object(ke, "nltk_tokenize_for_keyword", return_value=[]):
            with patch.object(ke, "remove_stopwords_for_keyword", return_value=[]):
                assert ke.extract_keywords_textrank("x") == []
        with patch.object(
            ke,
            "nltk_tokenize_for_keyword",
            return_value=["alpha", "beta", "alpha", "gamma", "beta"],
        ):
            with patch.object(ke, "remove_stopwords_for_keyword", side_effect=lambda t: t):
                kws = ke.extract_keywords_textrank(
                    "alpha beta alpha gamma beta", num_keywords=2, window_size=3
                )
                assert len(kws) <= 2
        # single token → empty graph
        with patch.object(ke, "nltk_tokenize_for_keyword", return_value=["only"]):
            with patch.object(ke, "remove_stopwords_for_keyword", side_effect=lambda t: t):
                assert ke.extract_keywords_textrank("only", window_size=4) == []
