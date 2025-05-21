from typing import List, Callable
import re # For segment_sentences_internal if NLTK is not used

# Optional dependencies for segment_sentences_internal if NLTK is used
try:
    import nltk
    from nltk.tokenize import sent_tokenize as nltk_sent_tokenize_for_chunking
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# ===== Dependency Management (for segment_sentences_internal) =====
def check_nltk_for_chunking():
    if not NLTK_AVAILABLE:
        raise ImportError("NLTK is not installed. Please install it with: pip install nltk")
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

# ===== Helper functions (copied for now) =====

def simple_tokenize_for_chunking(text: str) -> List[str]:
    """Simple word tokenization by splitting on whitespace."""
    return text.split()

def segment_sentences_internal_for_chunking(text: str, use_nltk: bool = True) -> List[str]:
    """Split text into sentences. (Internal helper for this module)"""
    if use_nltk:
        check_nltk_for_chunking()
        return nltk_sent_tokenize_for_chunking(text)
    else:
        pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
        return pattern.split(text)

# ===== Text Chunking and Splitting =====

def split_by_char_count(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """Split text into chunks of specified character length.

    Args:
        text: Input text
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
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
        if start >= text_len and overlap > 0 and len(chunks[-1]) < chunk_size : # Ensure last chunk is not just overlap
             if text_len > chunk_size and (text_len - (start - (chunk_size - overlap))) > 0 : # check if there is remaining part
                  pass # Allow smaller last chunk if it's the remainder
             elif len(chunks) > 1 and start - (chunk_size - overlap) + chunk_size > text_len: # handle overlap carefully for last chunk
                  # if the remaining text is smaller than chunk_size, take it all
                  # this logic might need further refinement based on exact desired behavior for overlaps at the end
                  pass


    return chunks

def split_by_word_count(text: str, words_per_chunk: int, overlap: int = 0, tokenizer: Callable = simple_tokenize_for_chunking) -> List[str]:
    """Split text into chunks with specified number of words.

    Args:
        text: Input text
        words_per_chunk: Maximum number of words per chunk
        overlap: Number of overlapping words between chunks
        tokenizer: Function to tokenize text

    Returns:
        List of text chunks
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
        chunks.append(' '.join(tokens[start:end]))
        start += words_per_chunk - overlap
        # Similar to split_by_char_count, ensure the last chunk handling with overlap is correct
        # if start >= tokens_len and overlap > 0 and len(tokens[start-(words_per_chunk-overlap):end]) < words_per_chunk:
        #    pass


    return chunks

def split_by_sentence(text: str, sentences_per_chunk: int, overlap: int = 0, use_nltk_for_segmentation: bool = True) -> List[str]:
    """Split text into chunks with specified number of sentences.

    Args:
        text: Input text
        sentences_per_chunk: Maximum number of sentences per chunk
        overlap: Number of overlapping sentences between chunks
        use_nltk_for_segmentation: whether to use NLTK for sentence segmentation.

    Returns:
        List of text chunks
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
        chunks.append(' '.join(sentences[start:end]))
        start += sentences_per_chunk - overlap
        # Similar overlap check for the last chunk might be needed

    return chunks

def sliding_window_chunks(text: str, window_size: int, step_size: int, tokenizer: Callable = simple_tokenize_for_chunking) -> List[str]:
    """Create sliding window chunks of text.

    Args:
        text: Input text
        window_size: Size of the window in tokens
        step_size: Number of tokens to slide the window
        tokenizer: Function to tokenize text

    Returns:
        List of text chunks
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    if step_size <= 0:
        raise ValueError("step_size must be positive")

    tokens = tokenizer(text)
    if len(tokens) < window_size:
        # If text is shorter than window_size, return the text as a single chunk
        # or an empty list, depending on desired behavior.
        # Current behavior in original code is to return [text], which might not be tokenized.
        # Returning [' '.join(tokens)] would be more consistent.
        return [' '.join(tokens)] if tokens else []


    chunks = []
    for i in range(0, len(tokens) - window_size + 1, step_size):
        chunks.append(' '.join(tokens[i:i+window_size]))

    return chunks

# Re-initialize NLTK_AVAILABLE based on successful import
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
