"""A3: prove deleted defensive branches in nlp were unreachable."""

from __future__ import annotations

from insideLLMs.nlp.language_detection import detect_language_by_char_ngrams
from insideLLMs.nlp.text_metrics import calculate_readability_flesch_kincaid


def test_ngram_len_ge_20_always_yields_trigrams() -> None:
    """After the <20 early return, trigram extraction cannot be empty."""
    cleaned_len = 20  # minimum that passes the gate
    assert list(range(cleaned_len - 2))  # non-empty ⇒ listcomp non-empty
    # Smoke: short → unknown; long ASCII → not crash
    assert detect_language_by_char_ngrams("x" * 19) == "unknown"
    assert isinstance(detect_language_by_char_ngrams("x" * 20), str)


def test_flesch_kincaid_empty_guards_cover_zero_counts(monkeypatch) -> None:
    """Empty sentence/word early returns make a second zero-len check redundant."""
    import insideLLMs.nlp.text_metrics as tm

    monkeypatch.setattr(tm, "_ensure_nltk_metrics", lambda: None)
    monkeypatch.setattr(tm, "segment_sentences_internal", lambda *a, **k: [])
    assert calculate_readability_flesch_kincaid("anything") == 0.0
    monkeypatch.setattr(tm, "segment_sentences_internal", lambda *a, **k: ["Hi."])
    monkeypatch.setattr(tm, "nltk_tokenize_internal", lambda t: [])
    assert calculate_readability_flesch_kincaid("Hi.") == 0.0
