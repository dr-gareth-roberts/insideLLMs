import re
from typing import Callable, List

from insideLLMs.nlp.dependencies import ensure_nltk


def check_nltk():
    """Ensure NLTK sentence tokenizer is available."""
    ensure_nltk(("tokenizers/punkt",))


def simple_tokenize_for_chunking(text: str) -> List[str]:
    """Simple word tokenization by splitting on whitespace."""
    return text.split()


def segment_sentences_internal_for_chunking(text: str, use_nltk: bool = True) -> List[str]:
    """Split text into sentences."""
    if use_nltk:
        check_nltk()
        from nltk.tokenize import sent_tokenize

        return sent_tokenize(text)
    pattern = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s")
    return pattern.split(text)


def split_by_char_count(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
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


def split_by_word_count(text: str, words_per_chunk: int, overlap: int = 0, tokenizer: Callable = simple_tokenize_for_chunking) -> List[str]:
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


def split_by_sentence(text: str, sentences_per_chunk: int, overlap: int = 0, use_nltk_for_segmentation: bool = True) -> List[str]:
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


def sliding_window_chunks(text: str, window_size: int, step_size: int, tokenizer: Callable = simple_tokenize_for_chunking) -> List[str]:
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
