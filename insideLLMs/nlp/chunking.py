"""
Text Chunking Utilities for NLP Processing.

This module provides functions for splitting text into chunks using various strategies,
which is essential for processing long documents with language models that have context
length limitations, or for creating overlapping segments for search and retrieval.

Chunking Strategies
-------------------
The module supports several chunking approaches:

1. **Character-based chunking** (`split_by_char_count`): Split text at exact character
   boundaries. Useful when you need precise control over chunk size in bytes/characters.

2. **Word-based chunking** (`split_by_word_count`): Split text by word count, preserving
   word boundaries. Better for maintaining readability than character-based splitting.

3. **Sentence-based chunking** (`split_by_sentence`): Split text by sentence count,
   preserving sentence boundaries. Best for maintaining semantic coherence.

4. **Sliding window chunking** (`sliding_window_chunks`): Create overlapping chunks
   using a sliding window approach. Ideal for search/retrieval where context overlap
   is important.

Examples
--------
Basic character-based chunking:

>>> from insideLLMs.nlp.chunking import split_by_char_count
>>> text = "Hello world! This is a test."
>>> split_by_char_count(text, chunk_size=10)
['Hello worl', 'd! This is', ' a test.']

Word-based chunking with overlap:

>>> from insideLLMs.nlp.chunking import split_by_word_count
>>> text = "The quick brown fox jumps over the lazy dog"
>>> split_by_word_count(text, words_per_chunk=3, overlap=1)
['The quick brown', 'brown fox jumps', 'jumps over the', 'the lazy dog']

Sentence-based chunking:

>>> from insideLLMs.nlp.chunking import split_by_sentence
>>> text = "First sentence. Second sentence. Third sentence. Fourth sentence."
>>> split_by_sentence(text, sentences_per_chunk=2, use_nltk_for_segmentation=False)
['First sentence. Second sentence.', 'Third sentence. Fourth sentence.']

Sliding window for search applications:

>>> from insideLLMs.nlp.chunking import sliding_window_chunks
>>> text = "one two three four five six"
>>> sliding_window_chunks(text, window_size=3, step_size=2)
['one two three', 'three four five', 'five six']

Notes
-----
- All functions handle edge cases gracefully (empty text, text shorter than chunk size)
- Overlap parameters allow creating chunks that share content at boundaries
- NLTK's punkt tokenizer is used by default for sentence segmentation (with fallback)
"""
import re
from typing import Callable

from insideLLMs.nlp.dependencies import ensure_nltk
from insideLLMs.nlp.tokenization import segment_sentences, simple_tokenize


def _ensure_nltk_chunking():
    """
    Ensure NLTK sentence tokenizer is available.

    This internal function checks that the NLTK punkt tokenizer data is
    downloaded and available. It is called automatically when sentence-based
    chunking with NLTK is requested.

    The punkt tokenizer is required for accurate sentence boundary detection,
    especially for text containing abbreviations (Dr., Mr., etc.), decimal
    numbers, and other edge cases that simple regex-based approaches handle
    poorly.

    Note:
        This function is exposed as `check_nltk` for backward compatibility.
    """
    ensure_nltk(("tokenizers/punkt",))


# Backward compatibility aliases
check_nltk = _ensure_nltk_chunking
simple_tokenize_for_chunking = simple_tokenize
segment_sentences_internal_for_chunking = segment_sentences


def split_by_char_count(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """
    Split text into chunks of specified character length.

    This function divides text into fixed-size character chunks, optionally with
    overlapping regions between consecutive chunks. Character-based chunking is
    useful when you need precise control over the byte/character size of chunks,
    such as when working with APIs that have strict character limits.

    Args:
        text: The input text to be chunked.
        chunk_size: Maximum number of characters per chunk. Must be positive.
        overlap: Number of characters to overlap between consecutive chunks.
            Must be less than chunk_size. Defaults to 0 (no overlap).

    Returns:
        A list of string chunks. Returns an empty list if text is empty.
        The last chunk may be shorter than chunk_size if the remaining
        text is insufficient.

    Raises:
        ValueError: If chunk_size is not positive.
        ValueError: If overlap is greater than or equal to chunk_size.

    Examples:
        Basic chunking without overlap:

        >>> split_by_char_count("Hello, World!", chunk_size=5)
        ['Hello', ', Wor', 'ld!']

        Chunking with overlap to preserve context at boundaries:

        >>> split_by_char_count("abcdefghij", chunk_size=4, overlap=2)
        ['abcd', 'cdef', 'efgh', 'ghij']

        Handling text shorter than chunk_size:

        >>> split_by_char_count("Hi", chunk_size=10)
        ['Hi']

        Empty text returns empty list:

        >>> split_by_char_count("", chunk_size=5)
        []

    Note:
        This function splits at exact character positions, which may break words
        or even multi-byte Unicode characters. For word-preserving chunking, use
        `split_by_word_count` instead.
    """
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
    """
    Split text into chunks with a specified number of words.

    This function divides text into chunks based on word count, preserving word
    boundaries. This is generally preferable to character-based chunking when
    working with natural language, as it maintains word integrity and readability.

    Args:
        text: The input text to be chunked.
        words_per_chunk: Maximum number of words per chunk. Must be positive.
        overlap: Number of words to overlap between consecutive chunks.
            Must be less than words_per_chunk. Defaults to 0 (no overlap).
        tokenizer: A callable that takes text and returns a list of tokens/words.
            Defaults to `simple_tokenize`, which splits on whitespace and
            punctuation.

    Returns:
        A list of string chunks, where each chunk contains up to `words_per_chunk`
        words joined by single spaces. Returns an empty list if text is empty
        or contains no tokens.

    Raises:
        ValueError: If words_per_chunk is not positive.
        ValueError: If overlap is greater than or equal to words_per_chunk.

    Examples:
        Basic word-based chunking:

        >>> split_by_word_count("The quick brown fox jumps over the lazy dog", words_per_chunk=3)
        ['The quick brown', 'fox jumps over', 'the lazy dog']

        Chunking with overlap for context preservation:

        >>> split_by_word_count("one two three four five six", words_per_chunk=3, overlap=1)
        ['one two three', 'three four five', 'five six']

        Using a custom tokenizer:

        >>> custom_tokenizer = lambda t: t.lower().split()
        >>> split_by_word_count("Hello World Test", words_per_chunk=2, tokenizer=custom_tokenizer)
        ['hello world', 'test']

        Text with fewer words than chunk size:

        >>> split_by_word_count("Short text", words_per_chunk=10)
        ['Short text']

    Note:
        The tokenizer determines how words are identified. The default tokenizer
        handles punctuation intelligently, but you can provide a custom tokenizer
        for specialized use cases (e.g., keeping contractions together).
    """
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
    """
    Split text into chunks with a specified number of sentences.

    This function divides text into chunks based on sentence count, preserving
    sentence boundaries. This approach maintains the highest level of semantic
    coherence, as sentences represent complete thoughts. Ideal for summarization,
    question-answering, and other NLP tasks where context matters.

    Args:
        text: The input text to be chunked.
        sentences_per_chunk: Maximum number of sentences per chunk. Must be positive.
        overlap: Number of sentences to overlap between consecutive chunks.
            Must be less than sentences_per_chunk. Defaults to 0 (no overlap).
        use_nltk_for_segmentation: If True (default), uses NLTK's punkt tokenizer
            for more accurate sentence boundary detection. If False, falls back
            to a simple regex-based approach that splits on `.!?` followed by
            whitespace.

    Returns:
        A list of string chunks, where each chunk contains up to `sentences_per_chunk`
        sentences joined by single spaces. Returns an empty list if text is empty
        or contains no sentences.

    Raises:
        ValueError: If sentences_per_chunk is not positive.
        ValueError: If overlap is greater than or equal to sentences_per_chunk.

    Examples:
        Basic sentence-based chunking:

        >>> text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        >>> split_by_sentence(text, sentences_per_chunk=2, use_nltk_for_segmentation=False)
        ['First sentence. Second sentence.', 'Third sentence. Fourth sentence.']

        Chunking with overlap for context continuity:

        >>> text = "Sentence one. Sentence two. Sentence three. Sentence four."
        >>> split_by_sentence(text, sentences_per_chunk=2, overlap=1, use_nltk_for_segmentation=False)
        ['Sentence one. Sentence two.', 'Sentence two. Sentence three.', 'Sentence three. Sentence four.']

        Using NLTK for better sentence detection (handles abbreviations, etc.):

        >>> text = "Dr. Smith went to Washington D.C. He arrived at 3 p.m."
        >>> # With NLTK: correctly handles "Dr." and "D.C." as non-sentence-ending
        >>> split_by_sentence(text, sentences_per_chunk=1, use_nltk_for_segmentation=True)  # doctest: +SKIP
        ['Dr. Smith went to Washington D.C.', 'He arrived at 3 p.m.']

        Handling text with complex punctuation:

        >>> text = "What time is it? It's noon! Great, let's go."
        >>> split_by_sentence(text, sentences_per_chunk=2, use_nltk_for_segmentation=False)
        ['What time is it? It's noon!', 'Great, let's go.']

    Note:
        NLTK's punkt tokenizer handles edge cases like abbreviations (Dr., Mr., etc.),
        decimal numbers (3.14), and URLs better than simple regex-based approaches.
        However, it requires the NLTK data to be downloaded (handled automatically).
    """
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
    """
    Create sliding window chunks of text.

    This function creates overlapping chunks by sliding a fixed-size window across
    the tokenized text. The overlap is controlled by the step size: smaller step
    sizes create more overlap, while a step size equal to window size creates
    non-overlapping chunks.

    This approach is particularly useful for:
    - Search and retrieval systems where relevant content may span chunk boundaries
    - Creating training data for models that benefit from context overlap
    - Text analysis where you want to ensure no information is lost at boundaries

    Args:
        text: The input text to be chunked.
        window_size: Number of tokens/words in each window. Must be positive.
        step_size: Number of tokens to advance the window. Must be positive.
            - step_size < window_size: Creates overlapping chunks
            - step_size == window_size: Creates non-overlapping chunks
            - step_size > window_size: Creates gaps (tokens are skipped)
        tokenizer: A callable that takes text and returns a list of tokens/words.
            Defaults to `simple_tokenize`.

    Returns:
        A list of string chunks, where each chunk contains exactly `window_size`
        tokens joined by single spaces. If the text has fewer tokens than
        `window_size`, returns a single chunk with all tokens (or empty list
        if no tokens).

    Raises:
        ValueError: If window_size is not positive.
        ValueError: If step_size is not positive.

    Examples:
        Basic sliding window with 50% overlap (step_size = window_size / 2):

        >>> sliding_window_chunks("one two three four five six", window_size=4, step_size=2)
        ['one two three four', 'three four five six']

        Sliding window with 66% overlap (step_size = window_size / 3):

        >>> sliding_window_chunks("a b c d e f g", window_size=3, step_size=1)
        ['a b c', 'b c d', 'c d e', 'd e f', 'e f g']

        Non-overlapping windows (step_size equals window_size):

        >>> sliding_window_chunks("1 2 3 4 5 6", window_size=2, step_size=2)
        ['1 2', '3 4', '5 6']

        Text shorter than window size returns single chunk:

        >>> sliding_window_chunks("short text", window_size=10, step_size=5)
        ['short text']

        Using with custom tokenizer for case-insensitive processing:

        >>> lower_tokenizer = lambda t: t.lower().split()
        >>> sliding_window_chunks("Hello World Test Case", window_size=2, step_size=1, tokenizer=lower_tokenizer)
        ['hello world', 'world test', 'test case']

    Note:
        Unlike `split_by_word_count` with overlap, this function ensures all chunks
        (except possibly the last) have exactly `window_size` tokens. It stops when
        there aren't enough remaining tokens to fill a complete window.
    """
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
