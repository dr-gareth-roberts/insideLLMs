import re
from typing import List, Dict, Callable
from collections import Counter

# Optional dependencies
try:
    import nltk
    from nltk.tokenize import word_tokenize as nltk_word_tokenize_for_metrics # Renamed to avoid conflict
    from nltk.tokenize import sent_tokenize as nltk_sent_tokenize_for_metrics # Renamed
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# ===== Dependency Management =====

def check_nltk():
    """Check if NLTK is available and download required resources if needed."""
    if not NLTK_AVAILABLE:
        raise ImportError(
            "NLTK is not installed. Please install it with: pip install nltk"
        )
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    # Removed stopwords and wordnet download as they are not directly used by these metric functions
    # but are needed by calculate_readability_flesch_kincaid -> count_syllables -> nltk_tokenize
    # and Flesch-Kincaid uses word_tokenize.

# ===== Helper functions (copied from tokenization.py for now) =====

def simple_tokenize(text: str) -> List[str]:
    """Simple word tokenization by splitting on whitespace.

    Args:
        text: Input text to tokenize

    Returns:
        List of tokens
    """
    return text.split()

def segment_sentences_internal(text: str, use_nltk: bool = True) -> List[str]:
    """Split text into sentences. (Internal helper for this module)

    Args:
        text: Input text to segment
        use_nltk: Whether to use NLTK (True) or a simple regex (False)

    Returns:
        List of sentences
    """
    if use_nltk:
        check_nltk()
        return nltk_sent_tokenize_for_metrics(text)
    else:
        # Simple regex-based sentence segmentation
        pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
        return pattern.split(text)

def nltk_tokenize_internal(text: str) -> List[str]:
    """Tokenize text using NLTK's word_tokenize. (Internal helper for this module)"""
    check_nltk()
    return nltk_word_tokenize_for_metrics(text)


# ===== Text Statistics and Metrics =====

def count_words(text: str, tokenizer: Callable = simple_tokenize) -> int:
    """Count the number of words in text.

    Args:
        text: Input text
        tokenizer: Function to tokenize text

    Returns:
        Number of words
    """
    tokens = tokenizer(text)
    return len(tokens)

def count_sentences(text: str) -> int:
    """Count the number of sentences in text.

    Args:
        text: Input text

    Returns:
        Number of sentences
    """
    sentences = segment_sentences_internal(text)
    return len(sentences)

def calculate_avg_word_length(text: str, tokenizer: Callable = simple_tokenize) -> float:
    """Calculate the average word length in text.

    Args:
        text: Input text
        tokenizer: Function to tokenize text

    Returns:
        Average word length
    """
    tokens = tokenizer(text)
    if not tokens:
        return 0.0
    return sum(len(token) for token in tokens) / len(tokens)

def calculate_avg_sentence_length(text: str, tokenizer: Callable = simple_tokenize) -> float:
    """Calculate the average sentence length (in words) in text.

    Args:
        text: Input text
        tokenizer: Function to tokenize text

    Returns:
        Average sentence length
    """
    sentences = segment_sentences_internal(text)
    if not sentences:
        return 0.0

    sentence_lengths = [len(tokenizer(sentence)) for sentence in sentences]
    return sum(sentence_lengths) / len(sentences) if sentences else 0.0


def calculate_lexical_diversity(text: str, tokenizer: Callable = simple_tokenize) -> float:
    """Calculate the lexical diversity (type-token ratio) of text.

    Args:
        text: Input text
        tokenizer: Function to tokenize text

    Returns:
        Lexical diversity score
    """
    tokens = tokenizer(text)
    if not tokens:
        return 0.0

    unique_tokens = set(tokens)
    return len(unique_tokens) / len(tokens)

def count_syllables(word: str) -> int:
    """Count the number of syllables in a word.

    This is a simple heuristic and may not be accurate for all words.

    Args:
        word: Input word

    Returns:
        Estimated number of syllables
    """
    word = word.lower()
    # Remove non-alphanumeric characters
    word = re.sub(r'[^a-z]', '', word)

    # Count vowel groups
    if not word:
        return 0

    # Count vowel sequences as syllables
    count = len(re.findall(r'[aeiouy]+', word))

    # Adjust for common patterns
    if word.endswith('e'):
        count -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in 'aeiouy':
        count += 1
    if count == 0 and len(word) > 0 : # Ensure word is not empty before assigning 1
        count = 1

    return count

def calculate_readability_flesch_kincaid(text: str) -> float:
    """Calculate the Flesch-Kincaid Grade Level readability score.

    Args:
        text: Input text

    Returns:
        Flesch-Kincaid Grade Level score
    """
    check_nltk() # NLTK is needed for tokenizing words and sentences for this formula
    
    sentences = segment_sentences_internal(text, use_nltk=True)
    if not sentences:
        return 0.0

    # Use the internal NLTK tokenizer for words to ensure consistency with syllable counting
    words = nltk_tokenize_internal(text) 
    if not words:
        return 0.0

    syllables = sum(count_syllables(word) for word in words)
    
    num_words = len(words)
    num_sentences = len(sentences)

    if num_sentences == 0 or num_words == 0: # Avoid division by zero
        return 0.0

    # Flesch-Kincaid Grade Level formula
    return 0.39 * (num_words / num_sentences) + 11.8 * (syllables / num_words) - 15.59


def get_word_frequencies(text: str, tokenizer: Callable = simple_tokenize) -> Dict[str, int]:
    """Get word frequencies in text.

    Args:
        text: Input text
        tokenizer: Function to tokenize text

    Returns:
        Dictionary mapping words to their frequencies
    """
    tokens = tokenizer(text)
    return dict(Counter(tokens))

# Re-initialize NLTK_AVAILABLE based on successful import
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
