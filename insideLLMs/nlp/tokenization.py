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
    """Simple word tokenization by splitting on whitespace."""
    return text.split()


def word_tokenize_regex(text: str, lowercase: bool = True) -> list[str]:
    """Tokenize text using word boundary regex.

    This extracts alphanumeric words using \\b\\w+\\b pattern.
    More robust than simple whitespace splitting for handling punctuation.

    Args:
        text: Input text to tokenize
        lowercase: If True, convert tokens to lowercase (default: True)

    Returns:
        List of word tokens
    """
    if lowercase:
        return re.findall(r"\b\w+\b", text.lower())
    return re.findall(r"\b\w+\b", text)


def nltk_tokenize(text: str) -> list[str]:
    """Tokenize text using NLTK's word_tokenize."""
    _ensure_nltk_tokenization()
    from nltk.tokenize import word_tokenize  # Local import after ensuring nltk

    return word_tokenize(text)


def spacy_tokenize(text: str, model_name: str = "en_core_web_sm") -> list[str]:
    """Tokenize text using spaCy."""
    nlp = ensure_spacy(model_name)
    doc = nlp(text)
    return [token.text for token in doc]


def segment_sentences(text: str, use_nltk: bool = True) -> list[str]:
    """Split text into sentences."""
    if use_nltk:
        _ensure_nltk_tokenization()
        from nltk.tokenize import sent_tokenize  # Local import after ensuring nltk

        return sent_tokenize(text)
    # Simple regex-based sentence segmentation
    pattern = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s")
    return pattern.split(text)


def get_ngrams(tokens: list[str], n: int = 2) -> list[tuple[str, ...]]:
    """Generate n-grams from a list of tokens."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def remove_stopwords(tokens: list[str], language: str = "english") -> list[str]:
    """Remove stopwords from a list of tokens."""
    _ensure_nltk_tokenization()
    from nltk.corpus import stopwords  # Local import after ensuring nltk

    stop_words_set = set(stopwords.words(language))
    return [token for token in tokens if token.lower() not in stop_words_set]


def stem_words(tokens: list[str]) -> list[str]:
    """Apply stemming to a list of tokens using Porter stemmer."""
    _ensure_nltk_tokenization()
    from nltk.stem import PorterStemmer  # Local import after ensuring nltk

    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def lemmatize_words(tokens: list[str]) -> list[str]:
    """Apply lemmatization to a list of tokens using WordNet lemmatizer."""
    _ensure_nltk_tokenization()
    from nltk.stem import WordNetLemmatizer  # Local import after ensuring nltk

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]
