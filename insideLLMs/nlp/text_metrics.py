"""Text metrics module for analyzing and measuring text characteristics.

This module provides a comprehensive set of functions for computing various
text metrics including word counts, sentence counts, readability scores,
lexical diversity, and word frequencies. These metrics are useful for
text analysis, content evaluation, and natural language processing tasks.

The module supports both simple regex-based tokenization and NLTK-based
tokenization for more accurate results. Most functions accept a custom
tokenizer parameter for flexibility.

Example:
    Basic usage for analyzing text metrics:

    >>> from insideLLMs.nlp.text_metrics import count_words, count_sentences
    >>> text = "Hello world. This is a test."
    >>> count_words(text)
    6
    >>> count_sentences(text)
    2

    Using readability analysis:

    >>> from insideLLMs.nlp.text_metrics import calculate_readability_flesch_kincaid
    >>> text = "The cat sat on the mat. It was a sunny day."
    >>> score = calculate_readability_flesch_kincaid(text)
    >>> print(f"Grade level: {score:.1f}")
    Grade level: 1.3

    Analyzing lexical diversity:

    >>> from insideLLMs.nlp.text_metrics import calculate_lexical_diversity
    >>> text = "the quick brown fox jumps over the lazy dog"
    >>> diversity = calculate_lexical_diversity(text)
    >>> print(f"Lexical diversity: {diversity:.2f}")
    Lexical diversity: 0.89

Functions:
    count_words: Count the number of words in text.
    count_sentences: Count the number of sentences in text.
    calculate_avg_word_length: Calculate average word length.
    calculate_avg_sentence_length: Calculate average sentence length in words.
    calculate_lexical_diversity: Calculate type-token ratio.
    count_syllables: Count syllables in a single word.
    calculate_readability_flesch_kincaid: Calculate Flesch-Kincaid grade level.
    get_word_frequencies: Get frequency distribution of words.
"""

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
    """Ensure NLTK and required resources are available for text metrics.

    This internal function checks and downloads the necessary NLTK data
    packages required for accurate text metric calculations. It specifically
    ensures the punkt tokenizer is available for sentence segmentation.

    The function is called automatically by functions that require NLTK,
    such as calculate_readability_flesch_kincaid().

    Returns:
        None

    Raises:
        ImportError: If NLTK cannot be imported or installed.

    Example:
        >>> _ensure_nltk_metrics()  # Downloads punkt if not present
        >>> # No output on success, NLTK resources now available

        >>> # Called internally by readability functions:
        >>> from insideLLMs.nlp.text_metrics import calculate_readability_flesch_kincaid
        >>> calculate_readability_flesch_kincaid("Test sentence.")
        -5.4...

    Note:
        This is a private function (prefixed with underscore) and is not
        intended for direct use. Use the public API functions instead.
    """
    ensure_nltk(("tokenizers/punkt",))


# Backward compatibility alias and imports
check_nltk = _ensure_nltk_metrics
segment_sentences_internal = segment_sentences
nltk_tokenize_internal = nltk_tokenize


def count_words(text: str, tokenizer: Callable = simple_tokenize) -> int:
    """Count the number of words in the given text.

    Tokenizes the input text and returns the total number of tokens (words).
    By default, uses simple regex-based tokenization that splits on whitespace
    and punctuation, but a custom tokenizer can be provided for more
    sophisticated word boundary detection.

    Args:
        text: The input text string to analyze. Can be any length from
            empty string to full documents.
        tokenizer: A callable that takes a string and returns a list of
            tokens. Defaults to simple_tokenize which uses regex-based
            tokenization. Can be replaced with nltk_tokenize for more
            accurate results with complex text.

    Returns:
        int: The number of words (tokens) in the text. Returns 0 for
            empty strings or strings with no valid tokens.

    Examples:
        Basic word counting:

        >>> count_words("Hello world")
        2
        >>> count_words("The quick brown fox jumps over the lazy dog.")
        9

        Handling empty or whitespace-only text:

        >>> count_words("")
        0
        >>> count_words("   ")
        0

        Text with punctuation (punctuation is not counted as words):

        >>> count_words("Hello, world! How are you?")
        5
        >>> count_words("It's a beautiful day.")
        5

        Using NLTK tokenizer for more accurate results:

        >>> from insideLLMs.nlp.tokenization import nltk_tokenize
        >>> count_words("Dr. Smith's appointment is at 3:00 p.m.", tokenizer=nltk_tokenize)
        8

        Multi-line text:

        >>> text = '''First line here.
        ... Second line follows.
        ... Third line ends.'''
        >>> count_words(text)
        9

    Note:
        The word count may vary depending on the tokenizer used. The default
        simple_tokenize treats contractions as separate tokens (e.g., "don't"
        becomes "don" and "t"), while nltk_tokenize may handle them differently.
    """
    tokens = tokenizer(text)
    return len(tokens)


def count_sentences(text: str) -> int:
    """Count the number of sentences in the given text.

    Segments the input text into sentences and returns the count. Uses
    the internal sentence segmentation function which can detect sentence
    boundaries based on punctuation patterns (periods, exclamation marks,
    question marks) and common abbreviations.

    Args:
        text: The input text string to analyze. Can contain multiple
            sentences separated by standard punctuation. Empty strings
            return 0.

    Returns:
        int: The number of sentences detected in the text. Returns 0
            for empty strings or strings without sentence-ending punctuation.

    Examples:
        Basic sentence counting:

        >>> count_sentences("Hello world.")
        1
        >>> count_sentences("Hello world. How are you?")
        2

        Multiple sentence types:

        >>> count_sentences("Is this a question? Yes! It is.")
        3
        >>> count_sentences("Wow! Amazing! Incredible!")
        3

        Handling edge cases:

        >>> count_sentences("")
        0
        >>> count_sentences("No ending punctuation")
        1

        Text with abbreviations (handled correctly):

        >>> count_sentences("Dr. Smith went to the store. He bought milk.")
        2
        >>> count_sentences("The U.S.A. is large. Very large indeed.")
        2

        Multi-paragraph text:

        >>> text = '''First paragraph here.
        ...
        ... Second paragraph follows. With two sentences.'''
        >>> count_sentences(text)
        3

    Note:
        Sentence detection is heuristic-based and may not be perfect for
        all edge cases. For more accurate sentence segmentation with
        complex text (e.g., legal documents, scientific papers), consider
        using NLTK-based segmentation directly.
    """
    sentences = segment_sentences_internal(text)
    return len(sentences)


def calculate_avg_word_length(text: str, tokenizer: Callable = simple_tokenize) -> float:
    """Calculate the average word length in characters for the given text.

    Tokenizes the input text and computes the mean length of all tokens.
    Word length is measured in characters. This metric can indicate text
    complexity, with longer average word lengths typically suggesting
    more sophisticated vocabulary.

    Args:
        text: The input text string to analyze. Can be any length.
        tokenizer: A callable that takes a string and returns a list of
            tokens. Defaults to simple_tokenize. The tokenizer choice
            affects how words are split and thus the calculated average.

    Returns:
        float: The average number of characters per word. Returns 0.0
            for empty strings or text with no valid tokens.

    Examples:
        Basic average word length:

        >>> calculate_avg_word_length("cat dog")
        3.0
        >>> calculate_avg_word_length("hello world")
        5.0

        Mixed word lengths:

        >>> calculate_avg_word_length("I am here")
        2.0
        >>> calculate_avg_word_length("The extraordinary circumstances")
        9.0

        Empty or whitespace text:

        >>> calculate_avg_word_length("")
        0.0
        >>> calculate_avg_word_length("   ")
        0.0

        Comparing simple vs complex text:

        >>> simple_text = "The cat sat on the mat."
        >>> calculate_avg_word_length(simple_text)
        2.5
        >>> complex_text = "The sophisticated algorithm demonstrates exceptional performance."
        >>> avg_len = calculate_avg_word_length(complex_text)
        >>> avg_len > 8  # Complex text has longer words
        True

        Using custom tokenizer:

        >>> from insideLLMs.nlp.tokenization import nltk_tokenize
        >>> text = "It's a wonderful life!"
        >>> calculate_avg_word_length(text, tokenizer=nltk_tokenize)
        4.25

    Note:
        The calculation includes all tokens returned by the tokenizer.
        Punctuation handling depends on the tokenizer used - simple_tokenize
        excludes punctuation, while nltk_tokenize may include it as separate
        tokens.
    """
    tokens = tokenizer(text)
    if not tokens:
        return 0.0
    return sum(len(token) for token in tokens) / len(tokens)


def calculate_avg_sentence_length(text: str, tokenizer: Callable = simple_tokenize) -> float:
    """Calculate the average sentence length measured in words.

    Segments the text into sentences, counts the words in each sentence,
    and returns the mean sentence length. This metric is commonly used
    in readability analysis - longer average sentence lengths typically
    indicate more complex, harder-to-read text.

    Args:
        text: The input text string to analyze. Should contain one or
            more sentences with proper punctuation.
        tokenizer: A callable that takes a string and returns a list of
            tokens. Defaults to simple_tokenize. Used to count words
            within each sentence.

    Returns:
        float: The average number of words per sentence. Returns 0.0
            for empty strings or text with no detected sentences.

    Examples:
        Basic sentence length calculation:

        >>> calculate_avg_sentence_length("Hello world.")
        2.0
        >>> calculate_avg_sentence_length("Hello. World.")
        1.0

        Multiple sentences with varying lengths:

        >>> text = "I am here. The quick brown fox jumps over the lazy dog."
        >>> calculate_avg_sentence_length(text)
        5.5
        >>> text = "Short. Medium length. This is a longer sentence here."
        >>> avg = calculate_avg_sentence_length(text)
        >>> round(avg, 2)
        3.0

        Empty text handling:

        >>> calculate_avg_sentence_length("")
        0.0
        >>> calculate_avg_sentence_length("   ")
        0.0

        Comparing writing styles:

        >>> hemingway = "The man fished. He caught nothing. The sun set."
        >>> calculate_avg_sentence_length(hemingway)
        2.67
        >>> academic = "The comprehensive analysis demonstrates significant findings. Furthermore, the methodology employed rigorous standards."
        >>> avg = calculate_avg_sentence_length(academic)
        >>> avg > 5  # Academic writing has longer sentences
        True

        Using custom tokenizer:

        >>> from insideLLMs.nlp.tokenization import nltk_tokenize
        >>> text = "Dr. Smith's report is ready. Please review it."
        >>> calculate_avg_sentence_length(text, tokenizer=nltk_tokenize)
        4.0

    Note:
        This function combines sentence segmentation and word tokenization,
        so results depend on both the sentence boundary detection and the
        chosen tokenizer. For most accurate results with complex text,
        consider using nltk_tokenize.
    """
    sentences = segment_sentences_internal(text)
    if not sentences:
        return 0.0

    sentence_lengths = [len(tokenizer(sentence)) for sentence in sentences]
    return sum(sentence_lengths) / len(sentences)


def calculate_lexical_diversity(text: str, tokenizer: Callable = simple_tokenize) -> float:
    """Calculate the lexical diversity (type-token ratio) of the text.

    Computes the ratio of unique words (types) to total words (tokens).
    This metric measures vocabulary richness - a higher ratio indicates
    more diverse vocabulary usage with fewer repeated words, while a
    lower ratio indicates more repetition.

    The type-token ratio (TTR) ranges from 0 to 1:
    - 1.0: Every word is unique (maximum diversity)
    - 0.5: Half the words are unique
    - Closer to 0: High repetition of words

    Args:
        text: The input text string to analyze. Longer texts typically
            have lower TTR due to natural word repetition.
        tokenizer: A callable that takes a string and returns a list of
            tokens. Defaults to simple_tokenize. Note that case-sensitive
            tokenization will treat "The" and "the" as different words.

    Returns:
        float: The type-token ratio between 0.0 and 1.0. Returns 0.0
            for empty strings or text with no valid tokens.

    Examples:
        Maximum diversity (all unique words):

        >>> calculate_lexical_diversity("the quick brown fox")
        1.0
        >>> calculate_lexical_diversity("one two three four five")
        1.0

        Repetitive text (lower diversity):

        >>> calculate_lexical_diversity("the the the the")
        0.25
        >>> calculate_lexical_diversity("hello hello world world")
        0.5

        Empty text handling:

        >>> calculate_lexical_diversity("")
        0.0
        >>> calculate_lexical_diversity("   ")
        0.0

        Comparing writing samples:

        >>> creative = "vibrant sunset painted golden hues across mountains"
        >>> calculate_lexical_diversity(creative)
        1.0
        >>> repetitive = "the cat and the dog and the bird"
        >>> div = calculate_lexical_diversity(repetitive)
        >>> round(div, 2)
        0.62

        Real-world text example:

        >>> text = "To be or not to be that is the question"
        >>> diversity = calculate_lexical_diversity(text)
        >>> round(diversity, 2)
        0.8

        Case sensitivity demonstration:

        >>> text = "The the THE"
        >>> calculate_lexical_diversity(text)  # Case-sensitive by default
        1.0

    Note:
        Type-token ratio is sensitive to text length - longer texts
        naturally have more word repetition and thus lower TTR values.
        For comparing texts of different lengths, consider using
        standardized measures like MSTTR (Mean Segmental TTR) or
        analyzing fixed-length samples.
    """
    tokens = tokenizer(text)
    if not tokens:
        return 0.0

    unique_tokens = set(tokens)
    return len(unique_tokens) / len(tokens)


def count_syllables(word: str) -> int:
    """Count the number of syllables in a single word using heuristics.

    Estimates syllable count using English language patterns. The algorithm:
    1. Converts to lowercase and removes non-alphabetic characters
    2. Counts vowel groups (consecutive vowels count as one syllable)
    3. Adjusts for silent 'e' at word endings
    4. Handles special cases like '-le' endings (e.g., "table")
    5. Ensures minimum of 1 syllable for non-empty words

    This is a heuristic approach optimized for English text. While not
    perfect for all words (especially borrowed words or proper nouns),
    it provides reasonable estimates for readability calculations.

    Args:
        word: A single word string to analyze. Can contain mixed case
            and punctuation (non-alphabetic characters are stripped).

    Returns:
        int: The estimated number of syllables. Returns 0 for empty
            strings or strings with no alphabetic characters. Returns
            minimum of 1 for any word with letters.

    Examples:
        Single syllable words:

        >>> count_syllables("cat")
        1
        >>> count_syllables("the")
        1
        >>> count_syllables("strengths")
        1

        Multi-syllable words:

        >>> count_syllables("hello")
        2
        >>> count_syllables("beautiful")
        3
        >>> count_syllables("extraordinary")
        5

        Words ending in silent 'e':

        >>> count_syllables("make")
        1
        >>> count_syllables("complete")
        2
        >>> count_syllables("demonstrate")
        3

        Words ending in '-le' (pronounced syllable):

        >>> count_syllables("table")
        2
        >>> count_syllables("simple")
        2
        >>> count_syllables("capable")
        3

        Handling punctuation and case:

        >>> count_syllables("Hello!")
        2
        >>> count_syllables("WORLD")
        1
        >>> count_syllables("don't")
        1

        Edge cases:

        >>> count_syllables("")
        0
        >>> count_syllables("123")
        0
        >>> count_syllables("a")
        1

    Note:
        This heuristic may not be accurate for:
        - Words borrowed from other languages (e.g., "naive" -> 2, not 1)
        - Proper nouns with unusual spellings
        - Technical terms or neologisms
        For applications requiring high accuracy, consider using a
        pronunciation dictionary like CMU Pronouncing Dictionary.
    """
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
    """Calculate the Flesch-Kincaid Grade Level readability score.

    Computes a readability score that corresponds to a U.S. school grade
    level. The formula considers average sentence length and average
    syllables per word:

        Grade Level = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59

    Higher scores indicate more difficult text requiring higher education
    levels to understand. The score roughly corresponds to the U.S. grade
    level needed to comprehend the text.

    Score interpretation:
    - Below 0: Very easy (rare, typically incomplete sentences)
    - 0-5: Elementary school level
    - 6-8: Middle school level
    - 9-12: High school level
    - 13-16: College level
    - 17+: Graduate/professional level

    Args:
        text: The input text string to analyze. Should contain complete
            sentences for accurate scoring. Requires NLTK punkt tokenizer
            (downloaded automatically if not present).

    Returns:
        float: The Flesch-Kincaid Grade Level score. Returns 0.0 for
            empty strings or text with no valid sentences/words.
            Negative scores are possible for very simple text.

    Examples:
        Simple text (low grade level):

        >>> text = "The cat sat. The dog ran."
        >>> score = calculate_readability_flesch_kincaid(text)
        >>> score < 3  # Elementary level
        True

        Moderate complexity:

        >>> text = "The weather today is pleasant and warm. Many people enjoy outdoor activities."
        >>> score = calculate_readability_flesch_kincaid(text)
        >>> 3 < score < 8  # Middle school level
        True

        Complex text (high grade level):

        >>> text = "The epistemological implications of quantum mechanics fundamentally challenge classical deterministic paradigms."
        >>> score = calculate_readability_flesch_kincaid(text)
        >>> score > 12  # College level or higher
        True

        Empty text handling:

        >>> calculate_readability_flesch_kincaid("")
        0.0
        >>> calculate_readability_flesch_kincaid("   ")
        0.0

        Comparing different writing styles:

        >>> children_book = "See the dog. The dog runs. Run, dog, run!"
        >>> legal_text = "Notwithstanding the aforementioned provisions, the contractual obligations herein shall remain enforceable."
        >>> children_score = calculate_readability_flesch_kincaid(children_book)
        >>> legal_score = calculate_readability_flesch_kincaid(legal_text)
        >>> legal_score > children_score
        True

    Note:
        This function requires NLTK and will automatically download the
        punkt tokenizer if not already present. The score is calibrated
        for English text and may not be meaningful for other languages.

    See Also:
        count_syllables: The syllable counting function used internally.
        calculate_avg_sentence_length: Related readability metric.
    """
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
    """Get the frequency distribution of words in the text.

    Tokenizes the input text and counts the occurrences of each unique
    token. Returns a dictionary mapping each word to its count. This is
    useful for vocabulary analysis, keyword extraction, and understanding
    word usage patterns.

    Args:
        text: The input text string to analyze. Can be any length.
        tokenizer: A callable that takes a string and returns a list of
            tokens. Defaults to simple_tokenize. The tokenizer determines
            how words are split and whether case is preserved.

    Returns:
        dict[str, int]: A dictionary where keys are unique words (tokens)
            and values are their occurrence counts. Returns an empty dict
            for empty strings or text with no valid tokens.

    Examples:
        Basic word frequency:

        >>> get_word_frequencies("hello world")
        {'hello': 1, 'world': 1}
        >>> freqs = get_word_frequencies("the cat and the dog")
        >>> freqs['the']
        2
        >>> freqs['cat']
        1

        Counting repeated words:

        >>> text = "to be or not to be"
        >>> freqs = get_word_frequencies(text)
        >>> freqs['to']
        2
        >>> freqs['be']
        2
        >>> freqs['or']
        1

        Empty text handling:

        >>> get_word_frequencies("")
        {}
        >>> get_word_frequencies("   ")
        {}

        Finding most common words:

        >>> text = "the quick brown fox jumps over the lazy dog the"
        >>> freqs = get_word_frequencies(text)
        >>> max(freqs, key=freqs.get)  # Most frequent word
        'the'
        >>> freqs['the']
        3

        Case sensitivity (default tokenizer is case-sensitive):

        >>> freqs = get_word_frequencies("The the THE")
        >>> len(freqs)  # Three different tokens
        3

        Using NLTK tokenizer:

        >>> from insideLLMs.nlp.tokenization import nltk_tokenize
        >>> text = "It's a beautiful day! Isn't it?"
        >>> freqs = get_word_frequencies(text, tokenizer=nltk_tokenize)
        >>> 'beautiful' in freqs
        True

        Analyzing word distribution:

        >>> text = "one fish two fish red fish blue fish"
        >>> freqs = get_word_frequencies(text)
        >>> freqs['fish']
        4
        >>> sum(freqs.values())  # Total words
        8

    Note:
        The frequency dictionary is not sorted. To get words sorted by
        frequency, use: sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        For large texts, consider using collections.Counter directly for
        additional functionality like most_common().
    """
    tokens = tokenizer(text)
    return dict(Counter(tokens))
