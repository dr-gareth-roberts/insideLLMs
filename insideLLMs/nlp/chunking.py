import re
from typing import Callable

from insideLLMs.nlp.dependencies import ensure_nltk
from insideLLMs.nlp.tokenization import segment_sentences, simple_tokenize


def _ensure_nltk_chunking():
    """Ensure NLTK sentence tokenizer is available."""
    ensure_nltk(("tokenizers/punkt",))


# Backward compatibility aliases
check_nltk = _ensure_nltk_chunking
simple_tokenize_for_chunking = simple_tokenize
segment_sentences_internal_for_chunking = segment_sentences


def split_by_char_count(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """Split text into chunks of specified character length."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def split_by_word_count(
    text: str,
    words_per_chunk: int,
    overlap: int = 0,
    tokenizer: Callable = simple_tokenize_for_chunking,
) -> list[str]:
    """Split text into chunks with a specified number of words."""
    if words_per_chunk <= 0:
        raise ValueError("words_per_chunk must be positive")
    if overlap >= words_per_chunk:
        raise ValueError("overlap must be less than words_per_chunk")

    tokens = tokenizer(text)
    if not tokens:
        return []

    chunks = []
    start = 0
    tokens_len = len(tokens)

    while start < tokens_len:
        end = min(start + words_per_chunk, tokens_len)
        chunks.append(" ".join(tokens[start:end]))
        start += words_per_chunk - overlap

    return chunks


def split_by_sentence(
    text: str, sentences_per_chunk: int, overlap: int = 0, use_nltk_for_segmentation: bool = True
) -> list[str]:
    """Split text into chunks with a specified number of sentences."""
    if sentences_per_chunk <= 0:
        raise ValueError("sentences_per_chunk must be positive")
    if overlap >= sentences_per_chunk:
        raise ValueError("overlap must be less than sentences_per_chunk")

    sentences = segment_sentences_internal_for_chunking(text, use_nltk=use_nltk_for_segmentation)
    if not sentences:
        return []

    chunks = []
    start = 0
    sentences_len = len(sentences)

    while start < sentences_len:
        end = min(start + sentences_per_chunk, sentences_len)
        chunks.append(" ".join(sentences[start:end]))
        start += sentences_per_chunk - overlap

    return chunks


def sliding_window_chunks(
    text: str, window_size: int, step_size: int, tokenizer: Callable = simple_tokenize_for_chunking
) -> list[str]:
    """Create sliding window chunks of text."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    tokens = tokenizer(text)
    if len(tokens) < window_size:
        return [" ".join(tokens)] if tokens else []

    chunks = []
    for i in range(0, len(tokens) - window_size + 1, step_size):
        chunks.append(" ".join(tokens[i : i + window_size]))

    return chunks
