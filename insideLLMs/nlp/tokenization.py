"""
Tokenization and Text Segmentation Utilities.

This module provides a comprehensive suite of text tokenization and preprocessing
functions for natural language processing tasks. It includes simple whitespace-based
tokenizers, regex-based tokenizers, and advanced tokenizers powered by NLTK and spaCy.

The module also provides text normalization utilities including stemming, lemmatization,
stopword removal, and n-gram generation.

Dependencies:
    - NLTK: Used for word_tokenize, sent_tokenize, stopwords, stemming, and lemmatization.
      Required resources are automatically downloaded on first use.
    - spaCy: Used for advanced tokenization with linguistic features.
      Requires a language model (e.g., 'en_core_web_sm') to be installed.

Example Usage:
    >>> from insideLLMs.nlp.tokenization import simple_tokenize, nltk_tokenize
    >>>
    >>> # Simple whitespace tokenization
    >>> simple_tokenize("Hello world!")
    ['Hello', 'world!']
    >>>
    >>> # NLTK-based tokenization (handles punctuation)
    >>> nltk_tokenize("Hello, world!")
    ['Hello', ',', 'world', '!']
    >>>
    >>> # Full preprocessing pipeline
    >>> from insideLLMs.nlp.tokenization import (
    ...     nltk_tokenize, remove_stopwords, lemmatize_words
    ... )
    >>> text = "The cats are running quickly through the gardens"
    >>> tokens = nltk_tokenize(text.lower())
    >>> tokens = remove_stopwords(tokens)
    >>> tokens = lemmatize_words(tokens)
    >>> tokens
    ['cat', 'running', 'quickly', 'garden']

Functions:
    - simple_tokenize: Basic whitespace splitting
    - word_tokenize_regex: Regex-based word extraction
    - nltk_tokenize: NLTK's word_tokenize wrapper
    - spacy_tokenize: spaCy-based tokenization
    - segment_sentences: Split text into sentences
    - get_ngrams: Generate n-grams from tokens
    - remove_stopwords: Filter out common stopwords
    - stem_words: Apply Porter stemming
    - lemmatize_words: Apply WordNet lemmatization
"""

import re

from insideLLMs.nlp.dependencies import ensure_nltk, ensure_spacy


def _ensure_nltk_tokenization():
    """Ensure NLTK and required resources are available."""
    ensure_nltk(("tokenizers/punkt", "corpora/stopwords", "corpora/wordnet"))


# Backward compatibility aliases
check_nltk = _ensure_nltk_tokenization
check_spacy = ensure_spacy


# ===== Tokenization and Segmentation =====


def simple_tokenize(text: str) -> list[str]:
    """
    Perform simple word tokenization by splitting on whitespace.

    This is the fastest tokenization method but does not handle punctuation
    intelligently. Punctuation remains attached to adjacent words.
    Best suited for quick preprocessing when punctuation handling is not critical.

    Args:
        text: The input text string to tokenize.

    Returns:
        A list of tokens (words) split by whitespace characters.
        Empty strings are not included in the result.

    Examples:
        >>> simple_tokenize("Hello world")
        ['Hello', 'world']

        >>> # Note: punctuation stays attached to words
        >>> simple_tokenize("Hello, world!")
        ['Hello,', 'world!']

        >>> # Multiple spaces are handled correctly
        >>> simple_tokenize("Hello    world")
        ['Hello', 'world']

        >>> # Empty string returns empty list
        >>> simple_tokenize("")
        []

        >>> # Tabs and newlines are also split points
        >>> simple_tokenize("Hello\\tworld\\nfoo")
        ['Hello', 'world', 'foo']

    Note:
        - This function uses Python's built-in str.split() with no arguments,
          which splits on any whitespace and removes empty strings.
        - For punctuation-aware tokenization, use nltk_tokenize() or
          word_tokenize_regex() instead.
        - Performance: O(n) where n is the length of the input text.
    """
    return text.split()


def word_tokenize_regex(text: str, lowercase: bool = True) -> list[str]:
    """
    Tokenize text using word boundary regex pattern.

    This function extracts alphanumeric words using the \\b\\w+\\b regex pattern,
    which matches sequences of word characters (letters, digits, underscores)
    bounded by word boundaries. This approach automatically strips punctuation
    from words, making it more robust than simple whitespace splitting.

    Args:
        text: The input text string to tokenize.
        lowercase: If True, convert all tokens to lowercase before returning.
            Defaults to True for case-insensitive text processing.

    Returns:
        A list of word tokens extracted from the text. Punctuation is excluded.

    Examples:
        >>> word_tokenize_regex("Hello, world!")
        ['hello', 'world']

        >>> # Preserve original case
        >>> word_tokenize_regex("Hello, World!", lowercase=False)
        ['Hello', 'World']

        >>> # Numbers are included as tokens
        >>> word_tokenize_regex("I have 3 apples and 2 oranges.")
        ['i', 'have', '3', 'apples', 'and', '2', 'oranges']

        >>> # Underscores are part of word characters
        >>> word_tokenize_regex("variable_name is snake_case")
        ['variable_name', 'is', 'snake_case']

        >>> # Contractions are split at the apostrophe
        >>> word_tokenize_regex("I don't know")
        ['i', 'don', 't', 'know']

        >>> # Hyphenated words are split
        >>> word_tokenize_regex("state-of-the-art")
        ['state', 'of', 'the', 'art']

    Note:
        - The \\w pattern matches [a-zA-Z0-9_], so underscores are kept but
          hyphens and apostrophes cause word splits.
        - For better handling of contractions and hyphenated words, consider
          using nltk_tokenize() or spacy_tokenize().
        - No external dependencies required; uses Python's built-in re module.
        - Performance: O(n) where n is the length of the input text.
    """
    if lowercase:
        return re.findall(r"\b\w+\b", text.lower())
    return re.findall(r"\b\w+\b", text)


def nltk_tokenize(text: str) -> list[str]:
    """
    Tokenize text using NLTK's word_tokenize function.

    This function provides linguistically-informed tokenization that properly
    handles punctuation, contractions, and other complex tokenization cases.
    It uses NLTK's Punkt tokenizer, which is trained on multilingual text.

    Args:
        text: The input text string to tokenize.

    Returns:
        A list of tokens including words and punctuation as separate tokens.

    Examples:
        >>> nltk_tokenize("Hello, world!")
        ['Hello', ',', 'world', '!']

        >>> # Contractions are split intelligently
        >>> nltk_tokenize("I don't know")
        ['I', 'do', "n't", 'know']

        >>> # Punctuation becomes separate tokens
        >>> nltk_tokenize("Dr. Smith went to Washington, D.C.")
        ['Dr.', 'Smith', 'went', 'to', 'Washington', ',', 'D.C', '.']

        >>> # Handles possessives
        >>> nltk_tokenize("John's book")
        ['John', "'s", 'book']

        >>> # Complex sentence with multiple punctuation
        >>> nltk_tokenize("Wait... what?!")
        ['Wait', '...', 'what', '?', '!']

    Note:
        - Requires NLTK to be installed. The function automatically downloads
          required NLTK resources (punkt tokenizer) on first use.
        - First call may be slower due to resource downloading and model loading.
        - For performance-critical applications with simple text, consider
          simple_tokenize() or word_tokenize_regex() instead.
        - The tokenizer preserves punctuation as separate tokens, which is useful
          for tasks that need punctuation information.

    Raises:
        ImportError: If NLTK is not installed.
    """
    _ensure_nltk_tokenization()
    from nltk.tokenize import word_tokenize, wordpunct_tokenize

    try:
        return word_tokenize(text)
    except LookupError:
        # Gracefully degrade when punkt resources are unavailable.
        return wordpunct_tokenize(text)


def spacy_tokenize(text: str, model_name: str = "en_core_web_sm") -> list[str]:
    """
    Tokenize text using spaCy's language processing pipeline.

    This function provides advanced tokenization using spaCy's statistical models,
    which are trained on large corpora and handle complex linguistic phenomena
    including multi-word expressions, abbreviations, and special characters.

    Args:
        text: The input text string to tokenize.
        model_name: The spaCy language model to use. Defaults to 'en_core_web_sm'
            (small English model). Other options include:
            - 'en_core_web_md': Medium English model (more accurate)
            - 'en_core_web_lg': Large English model (most accurate)
            - Language-specific models like 'de_core_news_sm' for German

    Returns:
        A list of token strings extracted from the text.

    Examples:
        >>> spacy_tokenize("Hello, world!")
        ['Hello', ',', 'world', '!']

        >>> # Handles URLs and emails
        >>> spacy_tokenize("Visit https://example.com or email test@email.com")
        ['Visit', 'https://example.com', 'or', 'email', 'test@email.com']

        >>> # Contractions handling
        >>> spacy_tokenize("I don't think we'll go")
        ['I', 'do', "n't", 'think', 'we', "'ll", 'go']

        >>> # Numbers and currency
        >>> spacy_tokenize("The price is $19.99")
        ['The', 'price', 'is', '$', '19.99']

        >>> # Using a different model
        >>> spacy_tokenize("Guten Tag!", model_name="de_core_news_sm")
        ['Guten', 'Tag', '!']

    Note:
        - Requires spaCy to be installed along with the specified language model.
          The function will attempt to download the model if not present.
        - First call with a new model is slower due to model loading. The model
          is cached for subsequent calls.
        - spaCy provides richer token information (POS tags, dependencies, etc.)
          but this function only returns token text. For full token objects,
          use spaCy directly.
        - Memory usage is higher than simpler tokenizers due to the full NLP
          pipeline being loaded.

    Raises:
        ImportError: If spaCy is not installed.
        OSError: If the specified language model cannot be loaded or downloaded.
    """
    nlp = ensure_spacy(model_name)
    doc = nlp(text)
    return [token.text for token in doc]


def segment_sentences(text: str, use_nltk: bool = True) -> list[str]:
    """
    Split text into individual sentences.

    This function provides sentence boundary detection using either NLTK's
    trained Punkt tokenizer or a simple regex-based approach. The NLTK approach
    is more accurate for complex cases like abbreviations and titles.

    Args:
        text: The input text string containing one or more sentences.
        use_nltk: If True, use NLTK's sent_tokenize (more accurate).
            If False, use a simple regex-based approach (no dependencies).
            Defaults to True.

    Returns:
        A list of sentence strings extracted from the text.

    Examples:
        >>> segment_sentences("Hello world. How are you?")
        ['Hello world.', 'How are you?']

        >>> # Handles abbreviations correctly with NLTK
        >>> segment_sentences("Dr. Smith went home. He was tired.")
        ['Dr. Smith went home.', 'He was tired.']

        >>> # Multiple punctuation types
        >>> segment_sentences("What time is it? I need to know! Please tell me.")
        ['What time is it?', 'I need to know!', 'Please tell me.']

        >>> # Using regex fallback (less accurate with abbreviations)
        >>> segment_sentences("Hello world. How are you?", use_nltk=False)
        ['Hello world.', 'How are you?']

        >>> # Single sentence returns list with one element
        >>> segment_sentences("Just one sentence here.")
        ['Just one sentence here.']

    Note:
        - The NLTK approach (use_nltk=True) handles edge cases like:
          - Abbreviations (Dr., Mr., Mrs., etc.)
          - Decimal numbers (3.14)
          - Ellipsis (...)
          - Initials (J. K. Rowling)
        - The regex approach (use_nltk=False) is faster but may incorrectly
          split on abbreviations or decimal points.
        - For production use, the NLTK approach is recommended for accuracy.
        - Empty input returns a list containing the empty string.

    Raises:
        ImportError: If use_nltk=True and NLTK is not installed.
    """
    # Simple regex-based sentence segmentation
    pattern = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s")
    if use_nltk:
        _ensure_nltk_tokenization()
        from nltk.tokenize import sent_tokenize

        try:
            return sent_tokenize(text)
        except LookupError:
            # Gracefully degrade when punkt resources are unavailable.
            return pattern.split(text)
    return pattern.split(text)


def get_ngrams(tokens: list[str], n: int = 2) -> list[tuple[str, ...]]:
    """
    Generate n-grams from a list of tokens.

    An n-gram is a contiguous sequence of n items from a sequence of tokens.
    This function creates sliding windows of size n over the token list,
    returning each window as a tuple.

    Args:
        tokens: A list of token strings to generate n-grams from.
        n: The size of each n-gram (number of tokens per gram).
            Common values:
            - n=1: unigrams (single tokens)
            - n=2: bigrams (pairs of consecutive tokens)
            - n=3: trigrams (triplets of consecutive tokens)
            Defaults to 2 (bigrams).

    Returns:
        A list of tuples, where each tuple contains n consecutive tokens
        from the input list.

    Examples:
        >>> get_ngrams(["I", "love", "Python"])
        [('I', 'love'), ('love', 'Python')]

        >>> # Generate trigrams
        >>> get_ngrams(["the", "quick", "brown", "fox"], n=3)
        [('the', 'quick', 'brown'), ('quick', 'brown', 'fox')]

        >>> # Unigrams (single tokens as tuples)
        >>> get_ngrams(["hello", "world"], n=1)
        [('hello',), ('world',)]

        >>> # If n > len(tokens), returns empty list
        >>> get_ngrams(["one", "two"], n=5)
        []

        >>> # Empty input returns empty output
        >>> get_ngrams([], n=2)
        []

        >>> # Full pipeline example
        >>> from insideLLMs.nlp.tokenization import simple_tokenize, get_ngrams
        >>> text = "I love natural language processing"
        >>> tokens = simple_tokenize(text.lower())
        >>> get_ngrams(tokens, n=2)
        [('i', 'love'), ('love', 'natural'), ('natural', 'language'), ('language', 'processing')]

    Note:
        - The number of n-grams generated is len(tokens) - n + 1.
        - For n=1, this effectively wraps each token in a single-element tuple.
        - No external dependencies required.
        - Performance: O(n * m) where m is the number of tokens.
        - N-grams are commonly used for:
          - Language modeling
          - Text classification features
          - Phrase detection
          - Collocation extraction
    """
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def remove_stopwords(tokens: list[str], language: str = "english") -> list[str]:
    """
    Remove stopwords from a list of tokens.

    Stopwords are common words (like "the", "is", "at") that typically don't
    carry significant meaning and are often filtered out in text processing.
    This function uses NLTK's curated stopword lists for various languages.

    Args:
        tokens: A list of token strings to filter.
        language: The language for the stopword list. Defaults to "english".
            Supported languages include: arabic, azerbaijani, danish, dutch,
            english, finnish, french, german, greek, hungarian, indonesian,
            italian, kazakh, nepali, norwegian, portuguese, romanian, russian,
            slovene, spanish, swedish, tajik, turkish.

    Returns:
        A new list containing only tokens that are not stopwords.
        Original token casing is preserved; comparison is case-insensitive.

    Examples:
        >>> remove_stopwords(["the", "quick", "brown", "fox"])
        ['quick', 'brown', 'fox']

        >>> # Case-insensitive matching, original case preserved
        >>> remove_stopwords(["The", "CAT", "is", "HERE"])
        ['CAT', 'HERE']

        >>> # Empty tokens list
        >>> remove_stopwords([])
        []

        >>> # All stopwords returns empty list
        >>> remove_stopwords(["the", "a", "an", "is", "are"])
        []

        >>> # Using different language
        >>> remove_stopwords(["le", "chat", "est", "noir"], language="french")
        ['chat', 'noir']

        >>> # Full pipeline example
        >>> from insideLLMs.nlp.tokenization import nltk_tokenize, remove_stopwords
        >>> text = "The quick brown fox jumps over the lazy dog"
        >>> tokens = nltk_tokenize(text.lower())
        >>> remove_stopwords(tokens)
        ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']

    Note:
        - Requires NLTK to be installed. The function automatically downloads
          the stopwords corpus on first use.
        - Stopword matching is case-insensitive, but the original token casing
          is preserved in the output.
        - Punctuation tokens are NOT removed by this function (they are not
          in the stopword list). Filter punctuation separately if needed.
        - Consider whether removing stopwords is appropriate for your task;
          some NLP tasks (like sentiment analysis) benefit from keeping them.
        - The stopword set is created fresh on each call. For processing many
          documents, consider caching the stopword set externally.

    Raises:
        ImportError: If NLTK is not installed.
        OSError: If the specified language's stopword list is not available.
    """
    _ensure_nltk_tokenization()
    from nltk.corpus import stopwords  # Local import after ensuring nltk

    stop_words_set = set(stopwords.words(language))
    return [token for token in tokens if token.lower() not in stop_words_set]


def stem_words(tokens: list[str]) -> list[str]:
    """
    Apply stemming to a list of tokens using the Porter stemmer algorithm.

    Stemming reduces words to their root form by removing suffixes using
    rule-based heuristics. The Porter stemmer is one of the most widely used
    stemming algorithms for English text. Note that stems may not be valid
    dictionary words.

    Args:
        tokens: A list of token strings to stem.

    Returns:
        A list of stemmed tokens in the same order as the input.
        All stems are lowercase regardless of input casing.

    Examples:
        >>> stem_words(["running", "runs", "ran"])
        ['run', 'run', 'ran']

        >>> # Stems are not always valid words
        >>> stem_words(["studies", "studying", "studied"])
        ['studi', 'studi', 'studi']

        >>> # Different word forms reduce to same stem
        >>> stem_words(["connect", "connected", "connecting", "connection"])
        ['connect', 'connect', 'connect', 'connect']

        >>> # Handles various suffixes
        >>> stem_words(["happiness", "happily", "happy"])
        ['happi', 'happili', 'happi']

        >>> # Empty list returns empty list
        >>> stem_words([])
        []

        >>> # Full pipeline example
        >>> from insideLLMs.nlp.tokenization import (
        ...     word_tokenize_regex, remove_stopwords, stem_words
        ... )
        >>> text = "The cats are running through the gardens"
        >>> tokens = word_tokenize_regex(text)
        >>> tokens = remove_stopwords(tokens)
        >>> stem_words(tokens)
        ['cat', 'run', 'garden']

    Note:
        - Requires NLTK to be installed.
        - The Porter stemmer is aggressive and may produce non-word stems
          (e.g., "studies" -> "studi"). For more linguistically valid results,
          consider using lemmatize_words() instead.
        - Stemming is language-specific; this function uses an English stemmer.
        - Stemming is irreversible - you cannot recover the original word form.
        - The stemmer is created fresh on each call. For processing many
          documents, consider creating the stemmer once externally.
        - Common use cases include:
          - Search engine indexing (matching query variants)
          - Text classification feature reduction
          - Information retrieval

    Raises:
        ImportError: If NLTK is not installed.

    See Also:
        lemmatize_words: For dictionary-based normalization that produces
            valid words.
    """
    _ensure_nltk_tokenization()
    from nltk.stem import PorterStemmer  # Local import after ensuring nltk

    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def lemmatize_words(tokens: list[str]) -> list[str]:
    """
    Apply lemmatization to a list of tokens using the WordNet lemmatizer.

    Lemmatization reduces words to their dictionary base form (lemma) using
    vocabulary lookup and morphological analysis. Unlike stemming, lemmatization
    produces valid dictionary words and considers the word's part of speech.

    Args:
        tokens: A list of token strings to lemmatize.

    Returns:
        A list of lemmatized tokens in the same order as the input.
        Each token is reduced to its base dictionary form.

    Examples:
        >>> lemmatize_words(["cats", "dogs", "children"])
        ['cat', 'dog', 'child']

        >>> # Verb forms (default POS is noun, so verbs may not fully lemmatize)
        >>> lemmatize_words(["running", "runs", "ran"])
        ['running', 'run', 'ran']

        >>> # Preserves words that are already in base form
        >>> lemmatize_words(["happy", "cat", "run"])
        ['happy', 'cat', 'run']

        >>> # Handles irregular plurals
        >>> lemmatize_words(["mice", "geese", "feet"])
        ['mouse', 'goose', 'foot']

        >>> # Case sensitivity - input case is preserved
        >>> lemmatize_words(["Cats", "DOGS"])
        ['Cats', 'DOGS']

        >>> # Empty list returns empty list
        >>> lemmatize_words([])
        []

        >>> # Full preprocessing pipeline
        >>> from insideLLMs.nlp.tokenization import (
        ...     nltk_tokenize, remove_stopwords, lemmatize_words
        ... )
        >>> text = "The cats were running through the gardens"
        >>> tokens = nltk_tokenize(text.lower())
        >>> tokens = remove_stopwords(tokens)
        >>> lemmatize_words(tokens)
        ['cat', 'running', 'garden']

    Note:
        - Requires NLTK to be installed. The function automatically downloads
          the WordNet corpus on first use.
        - The default WordNet lemmatizer assumes all words are nouns. For better
          results with verbs, adjectives, and adverbs, provide the POS tag to
          the lemmatizer (requires using NLTK's WordNetLemmatizer directly).
        - Lemmatization is slower than stemming but produces more meaningful
          results for human-readable output.
        - Unlike stemming, lemmatization preserves the semantic meaning of words.
        - The lemmatizer is case-sensitive; "Cats" will not lemmatize to "cat"
          unless lowercased first.
        - Common use cases include:
          - Text normalization for NLP models
          - Search query expansion
          - Document similarity computation
          - Named entity recognition preprocessing

    Raises:
        ImportError: If NLTK is not installed.

    See Also:
        stem_words: For faster but less accurate rule-based normalization.
    """
    _ensure_nltk_tokenization()
    from nltk.stem import WordNetLemmatizer  # Local import after ensuring nltk

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]
