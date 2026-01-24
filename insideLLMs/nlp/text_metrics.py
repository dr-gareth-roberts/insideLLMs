import re
from collections import Counter
from typing import Callable

from insideLLMs.nlp.dependencies import ensure_nltk
from insideLLMs.nlp.tokenization import (
    nltk_tokenize,
    segment_sentences,
    simple_tokenize,
)


def _ensure_nltk_metrics():
    """Ensure NLTK and required resources are available."""
    ensure_nltk(("tokenizers/punkt",))


# Backward compatibility alias and imports
check_nltk = _ensure_nltk_metrics
segment_sentences_internal = segment_sentences
nltk_tokenize_internal = nltk_tokenize


def count_words(text: str, tokenizer: Callable = simple_tokenize) -> int:
    """Count the number of words in text."""
    tokens = tokenizer(text)
    return len(tokens)


def count_sentences(text: str) -> int:
    """Count the number of sentences in text."""
    sentences = segment_sentences_internal(text)
    return len(sentences)


def calculate_avg_word_length(text: str, tokenizer: Callable = simple_tokenize) -> float:
    """Calculate the average word length in text."""
    tokens = tokenizer(text)
    if not tokens:
        return 0.0
    return sum(len(token) for token in tokens) / len(tokens)


def calculate_avg_sentence_length(text: str, tokenizer: Callable = simple_tokenize) -> float:
    """Calculate the average sentence length (in words) in text."""
    sentences = segment_sentences_internal(text)
    if not sentences:
        return 0.0

    sentence_lengths = [len(tokenizer(sentence)) for sentence in sentences]
    return sum(sentence_lengths) / len(sentences)


def calculate_lexical_diversity(text: str, tokenizer: Callable = simple_tokenize) -> float:
    """Calculate the lexical diversity (type-token ratio) of text."""
    tokens = tokenizer(text)
    if not tokens:
        return 0.0

    unique_tokens = set(tokens)
    return len(unique_tokens) / len(tokens)


def count_syllables(word: str) -> int:
    """Count the number of syllables in a word (heuristic)."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0

    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e"):
        count -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in "aeiouy":
        count += 1
    if count == 0:
        count = 1

    return count


def calculate_readability_flesch_kincaid(text: str) -> float:
    """Calculate the Flesch-Kincaid Grade Level readability score."""
    _ensure_nltk_metrics()

    sentences = segment_sentences_internal(text, use_nltk=True)
    if not sentences:
        return 0.0

    words = nltk_tokenize_internal(text)
    if not words:
        return 0.0

    syllables = sum(count_syllables(word) for word in words)

    num_words = len(words)
    num_sentences = len(sentences)
    if num_sentences == 0 or num_words == 0:
        return 0.0

    return 0.39 * (num_words / num_sentences) + 11.8 * (syllables / num_words) - 15.59


def get_word_frequencies(text: str, tokenizer: Callable = simple_tokenize) -> dict[str, int]:
    """Get word frequencies in text."""
    tokens = tokenizer(text)
    return dict(Counter(tokens))
