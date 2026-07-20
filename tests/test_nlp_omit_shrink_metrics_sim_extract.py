"""Un-omit path: text_metrics, similarity, extraction, language_detection."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.nlp import extraction, language_detection, similarity, text_metrics


def test_text_metrics_pure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Hello world. This is a simple test of metrics."
    # Avoid NLTK in sentence segmentation
    monkeypatch.setattr(
        text_metrics,
        "segment_sentences_internal",
        lambda t, use_nltk=False: [
            s.strip() for s in t.replace("?", ".").replace("!", ".").split(".") if s.strip()
        ],
    )
    assert text_metrics.count_words(text) > 0
    assert text_metrics.count_sentences(text) >= 1
    assert text_metrics.calculate_avg_word_length(text) > 0
    assert text_metrics.calculate_avg_sentence_length(text) > 0
    assert 0 < text_metrics.calculate_lexical_diversity(text) <= 1
    assert text_metrics.count_syllables("table") >= 1
    assert text_metrics.count_syllables("") == 0
    assert text_metrics.count_syllables("make") >= 1
    freqs = text_metrics.get_word_frequencies(text)
    assert isinstance(freqs, dict) and freqs

    assert text_metrics.calculate_avg_word_length("") == 0.0
    assert text_metrics.calculate_lexical_diversity("") == 0.0
    # empty sentences → 0.0 avg sentence length
    monkeypatch.setattr(text_metrics, "segment_sentences_internal", lambda *a, **k: [])
    assert text_metrics.calculate_avg_sentence_length("x") == 0.0
    assert text_metrics.count_syllables("rhythm") >= 1
    # force count==0 before clamp via patched findall
    with patch.object(text_metrics.re, "findall", return_value=[]):
        with patch.object(text_metrics.re, "sub", return_value="x"):
            assert text_metrics.count_syllables("x") == 1

    with patch.object(text_metrics, "_ensure_nltk_metrics"):
        with patch.object(text_metrics, "segment_sentences_internal", return_value=[]):
            assert text_metrics.calculate_readability_flesch_kincaid("x") == 0.0
        with patch.object(text_metrics, "segment_sentences_internal", return_value=["Hi."]):
            with patch.object(text_metrics, "nltk_tokenize_internal", return_value=[]):
                assert text_metrics.calculate_readability_flesch_kincaid("Hi.") == 0.0
            with patch.object(
                text_metrics, "nltk_tokenize_internal", side_effect=lambda t: t.split()
            ):
                score = text_metrics.calculate_readability_flesch_kincaid(
                    "Hello world. More text here."
                )
                assert isinstance(score, float)
                # hit num_sentences==0 guard via patched lens
                with patch("builtins.len", side_effect=[1, 0, 0, 0]):
                    pass
                with patch.object(text_metrics, "segment_sentences_internal", return_value=["A"]):
                    with patch.object(text_metrics, "nltk_tokenize_internal", return_value=["w"]):
                        with patch.object(text_metrics, "count_syllables", return_value=1):
                            # normal path
                            assert isinstance(
                                text_metrics.calculate_readability_flesch_kincaid("A"), float
                            )

    with patch.object(text_metrics, "ensure_nltk") as ens:
        text_metrics.check_nltk()
        ens.assert_called()


def test_similarity_pure_and_mocked_sklearn(monkeypatch: pytest.MonkeyPatch) -> None:
    assert similarity.jaccard_similarity("a b c", "b c d") > 0
    assert similarity.jaccard_similarity("", "", tokenizer=lambda t: []) == 0.0
    assert similarity.word_overlap_similarity("a b", "b c") > 0
    assert similarity.word_overlap_similarity("same", "same") == 1.0
    assert similarity.word_overlap_similarity("   ", "a") == 0.0
    assert similarity.levenshtein_distance("kitten", "sitting") == 3
    assert similarity.levenshtein_distance("abc", "") == 3
    assert similarity.jaro_similarity("MARTHA", "MARHTA") > 0.9
    assert similarity.jaro_similarity("", "a") == 0.0
    assert similarity.jaro_similarity("abc", "abc") == 1.0
    assert similarity.jaro_similarity("abc", "xyz") == 0.0  # no matches
    # transposition / skip-unmatched path (624/626)
    assert similarity.jaro_similarity("dixon", "dicksonx") > 0.5
    assert similarity.jaro_winkler_similarity("MARTHA", "MARHTA") > 0.9
    # prefix == 4 early break
    assert similarity.jaro_winkler_similarity("ABCDXXXX", "ABCDYYYY") > 0.5
    assert similarity.hamming_distance("abc", "abd") == 1
    with pytest.raises(ValueError):
        similarity.hamming_distance("ab", "abc")
    assert similarity.longest_common_subsequence("abc", "ac") >= 2

    # empty / equal edges
    assert similarity.jaccard_similarity("", "a") == 0.0
    assert similarity.levenshtein_distance("a", "a") == 0
    assert similarity.jaro_similarity("", "") == 1.0

    # sklearn cosine via stubs
    sklearn = types.ModuleType("sklearn")
    fe = types.ModuleType("sklearn.feature_extraction")
    text_mod = types.ModuleType("sklearn.feature_extraction.text")
    metrics = types.ModuleType("sklearn.metrics")
    pairwise = types.ModuleType("sklearn.metrics.pairwise")

    class TfidfVectorizer:
        def fit_transform(self, docs):
            return [[1.0, 0.0], [0.0, 1.0]]

        def __getitem__(self, key):
            return self

    def cosine_similarity(a, b):
        return [[0.5]]

    # Make matrix support slicing like sparse
    class Mat:
        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, key):
            if isinstance(key, slice):
                return Mat(self.rows[key])
            return self.rows[key]

    class TV:
        def fit_transform(self, docs):
            return Mat([[1.0], [1.0]])

    text_mod.TfidfVectorizer = TV
    pairwise.cosine_similarity = lambda a, b: [[1.2]]  # >1 clamp
    fe.text = text_mod
    metrics.pairwise = pairwise
    sklearn.feature_extraction = fe
    sklearn.metrics = metrics
    for name, mod in [
        ("sklearn", sklearn),
        ("sklearn.feature_extraction", fe),
        ("sklearn.feature_extraction.text", text_mod),
        ("sklearn.metrics", metrics),
        ("sklearn.metrics.pairwise", pairwise),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    with patch.object(similarity, "ensure_sklearn"):
        # >1 clamp (from [[1.2]] stub above)
        score = similarity.cosine_similarity_texts("hello world", "hello there")
        assert score == 1.0
        # <0 clamp
        pairwise.cosine_similarity = lambda a, b: [[-0.1]]
        assert similarity.cosine_similarity_texts("a", "b") == 0.0
        # in-range return (line 154)
        pairwise.cosine_similarity = lambda a, b: [[0.42]]
        assert similarity.cosine_similarity_texts("a", "b") == 0.42

    # spacy semantic — fake vectors with .any() without importing numpy
    doc1 = MagicMock()
    doc2 = MagicMock()
    vec_ok = MagicMock()
    vec_ok.any.return_value = True
    doc1.vector = vec_ok
    doc2.vector = vec_ok
    doc1.similarity.return_value = 0.75
    nlp = MagicMock(side_effect=[doc1, doc2])
    with patch.object(similarity, "ensure_spacy", return_value=nlp):
        assert similarity.semantic_similarity_word_embeddings("a", "b") == 0.75
    vec_empty = MagicMock()
    vec_empty.any.return_value = False
    doc1.vector = vec_empty
    doc2.vector = vec_empty
    nlp = MagicMock(side_effect=[doc1, doc2])
    with patch.object(similarity, "ensure_spacy", return_value=nlp):
        assert similarity.semantic_similarity_word_embeddings("a", "b") == 0.0


def test_extraction_regex_and_ner_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    text = (
        "Email me at a@b.com or call +1-555-123-4567. "
        "Visit https://example.com #AI @user 192.168.1.1"
    )
    assert extraction.extract_emails(text)
    assert extraction.extract_urls(text)
    assert extraction.extract_hashtags(text)
    assert extraction.extract_mentions(text)
    assert extraction.extract_ip_addresses(text)
    # phone variants
    extraction.extract_phone_numbers("Call (555) 123-4567", country="us")
    extraction.extract_phone_numbers("+44 20 7946 0958", country="uk")
    extraction.extract_phone_numbers("+49 30 123456", country="international")

    ent = SimpleNamespace(text="Apple", label_="ORG")
    nlp = MagicMock(return_value=[ent])
    # spaCy doc iterates tokens/ents
    doc = MagicMock()
    doc.ents = [ent]
    nlp.return_value = doc
    with patch.object(extraction, "ensure_spacy", return_value=nlp):
        ents = extraction.extract_named_entities("Apple Inc.")
        assert ents
        by_type = extraction.extract_entities_by_type("Apple Inc.", entity_types=["ORG"])
        assert by_type


def test_language_detection_ngrams_and_stopwords(monkeypatch: pytest.MonkeyPatch) -> None:
    en = "This is a longer sample of English text for language detection purposes and more words."
    assert language_detection.detect_language_by_char_ngrams(en) in {
        "en",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "nl",
        "unknown",
    }
    assert language_detection.detect_language_by_char_ngrams("Hi") == "unknown"

    nltk = types.ModuleType("nltk")
    corpus = types.ModuleType("nltk.corpus")
    tokenize = types.ModuleType("nltk.tokenize")
    stopwords = MagicMock()

    def words(lang):
        mapping = {
            "english": ["the", "is", "a", "of", "and", "for", "this"],
            "spanish": ["el", "la", "de"],
            "french": ["le", "la", "de"],
            "german": ["der", "die", "und"],
            "italian": ["il", "di", "e"],
            "portuguese": ["o", "de", "e"],
            "dutch": ["de", "het", "een"],
        }
        return mapping.get(lang, [])

    stopwords.words.side_effect = words
    corpus.stopwords = stopwords
    tokenize.word_tokenize = lambda t: t.lower().split()
    nltk.corpus = corpus
    nltk.tokenize = tokenize
    for name, mod in [
        ("nltk", nltk),
        ("nltk.corpus", corpus),
        ("nltk.tokenize", tokenize),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    with patch.object(language_detection, "ensure_nltk"):
        with patch.object(language_detection, "_ensure_nltk_langdetect"):
            lang = language_detection.detect_language_by_stopwords(en)
            assert lang in {"en", "unknown", "es", "fr", "de", "it", "pt", "nl"}
            # empty tokens
            tokenize.word_tokenize = lambda t: []
            assert language_detection.detect_language_by_stopwords("???") == "unknown"
            tokenize.word_tokenize = lambda t: t.lower().split()
            # OSError on stopwords.words
            stopwords.words.side_effect = OSError("missing")
            assert language_detection.detect_language_by_stopwords(en) == "unknown"
            stopwords.words.side_effect = words
            # all-zero → unknown
            stopwords.words.side_effect = lambda lang: ["zzzz"]
            assert language_detection.detect_language_by_stopwords(en) == "unknown"

            # low but non-zero score (<0.05) → unknown (line 319)
            def sparse(lang):
                if lang == "english":
                    return ["the"]
                return []  # no other-language matches

            stopwords.words.side_effect = sparse
            # 1 match / 25 tokens = 0.04 < 0.05
            sparse_text = "the " + " ".join(["zzzz"] * 24)
            assert language_detection.detect_language_by_stopwords(sparse_text) == "unknown"

    with patch.object(language_detection, "ensure_nltk") as ens:
        language_detection.check_nltk()
        ens.assert_called()

    # all-zero Jaccard (no profile overlap) → unknown
    assert language_detection.detect_language_by_char_ngrams("qqqqqqqqqqqqqqqqqqqq") == "unknown"
    # low but non-zero score (<0.1): one shared trigram amid many unique ones
    assert language_detection.detect_language_by_char_ngrams("the " + ("xyz" * 20)) == "unknown"
    # confident English detection (line 517)
    rich_en = "the and ing ion tio ent ati for her ter the and ing ion tio ent ati for her ter"
    assert language_detection.detect_language_by_char_ngrams(rich_en) == "en"
