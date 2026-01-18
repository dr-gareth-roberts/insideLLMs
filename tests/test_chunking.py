"""Tests for text chunking functionality."""

import pytest

from insideLLMs.nlp.chunking import (
    split_by_char_count,
    split_by_sentence,
    split_by_word_count,
)


def test_split_by_char_count_with_overlap():
    """Test character-based chunking with overlap.

    With chunk_size=4 and overlap=1:
    - "abcd" (0:4)
    - "defg" (3:7) - starts at position 3 due to overlap
    - "ghij" (6:10) - starts at position 6
    - "j" (9:10) - last partial chunk

    The implementation correctly includes partial chunks at the end.
    """
    chunks = split_by_char_count("abcdefghij", chunk_size=4, overlap=1)
    # Chunk positions: 0-4, 3-7, 6-10, 9-10
    assert len(chunks) == 4
    assert chunks[0] == "abcd"
    assert chunks[1] == "defg"
    assert chunks[2] == "ghij"
    assert chunks[3] == "j"  # Partial final chunk


def test_split_by_char_count_no_overlap():
    """Test character chunking without overlap."""
    chunks = split_by_char_count("abcdefgh", chunk_size=4, overlap=0)
    assert chunks == ["abcd", "efgh"]


def test_split_by_char_count_exact_fit():
    """Test when text fits exactly into chunks."""
    chunks = split_by_char_count("abcdef", chunk_size=3, overlap=0)
    assert chunks == ["abc", "def"]


def test_split_by_word_count_with_overlap():
    """Test word-based chunking with overlap."""
    text = "one two three four five six"
    chunks = split_by_word_count(text, words_per_chunk=3, overlap=1)
    # With overlap=1, step is 2 words
    # Chunk 1: words 0-2 (one two three)
    # Chunk 2: words 2-4 (three four five)
    # Chunk 3: words 4-5 (five six)
    assert len(chunks) == 3
    assert chunks[0] == "one two three"
    assert chunks[1] == "three four five"
    assert chunks[2] == "five six"


def test_split_by_word_count_no_overlap():
    """Test word chunking without overlap."""
    text = "one two three four five six"
    chunks = split_by_word_count(text, words_per_chunk=2, overlap=0)
    assert chunks == ["one two", "three four", "five six"]


def test_split_by_sentence_with_overlap_no_nltk():
    """Test sentence-based chunking with overlap, without NLTK."""
    text = "First sentence. Second sentence. Third sentence."
    chunks = split_by_sentence(
        text, sentences_per_chunk=2, overlap=1, use_nltk_for_segmentation=False
    )
    # With overlap=1, step is 1 sentence
    # Chunk 1: sentences 0-1 (First, Second)
    # Chunk 2: sentences 1-2 (Second, Third)
    # Chunk 3: sentence 2 (Third)
    assert len(chunks) == 3
    assert "First sentence" in chunks[0]
    assert "Second sentence" in chunks[0]
    assert "Second sentence" in chunks[1]
    assert "Third sentence" in chunks[1]
    assert "Third sentence" in chunks[2]


def test_split_by_sentence_no_overlap():
    """Test sentence chunking without overlap."""
    text = "First. Second. Third. Fourth."
    chunks = split_by_sentence(
        text, sentences_per_chunk=2, overlap=0, use_nltk_for_segmentation=False
    )
    assert len(chunks) == 2


def test_empty_text():
    """Test chunking with empty text."""
    assert split_by_char_count("", chunk_size=4) == []
    assert split_by_word_count("", words_per_chunk=4) == []
    # Note: split_by_sentence may return [''] for empty text due to regex behavior
    result = split_by_sentence("", sentences_per_chunk=2, use_nltk_for_segmentation=False)
    assert result == [] or result == [""]


@pytest.mark.parametrize(
    "func,args",
    [
        (split_by_char_count, {"text": "abc", "chunk_size": 2, "overlap": 2}),
        (split_by_word_count, {"text": "a b c", "words_per_chunk": 2, "overlap": 2}),
        (split_by_sentence, {"text": "A. B.", "sentences_per_chunk": 1, "overlap": 1}),
    ],
)
def test_overlap_validation(func, args):
    """Test that overlap >= chunk_size raises ValueError."""
    with pytest.raises(ValueError):
        func(**args)


@pytest.mark.parametrize(
    "func,args",
    [
        (split_by_char_count, {"text": "abc", "chunk_size": 0}),
        (split_by_word_count, {"text": "a b c", "words_per_chunk": 0}),
        (
            split_by_sentence,
            {"text": "A. B.", "sentences_per_chunk": 0, "use_nltk_for_segmentation": False},
        ),
    ],
)
def test_invalid_chunk_size(func, args):
    """Test that chunk_size <= 0 raises ValueError."""
    with pytest.raises(ValueError):
        func(**args)
