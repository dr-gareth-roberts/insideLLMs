import re
from typing import List, Tuple

from insideLLMs.nlp.dependencies import ensure_nltk, ensure_spacy


# ===== Dependency Management =====

def check_nltk():
    """Ensure NLTK and required resources are available."""
    ensure_nltk(("tokenizers/punkt", "corpora/stopwords", "corpora/wordnet"))


def check_spacy(model_name: str = "en_core_web_sm"):
    """Ensure spaCy and the requested model are available."""
    return ensure_spacy(model_name)


# ===== Tokenization and Segmentation =====

def simple_tokenize(text: str) -> List[str]:
    """Simple word tokenization by splitting on whitespace."""
    return text.split()


def nltk_tokenize(text: str) -> List[str]:
    """Tokenize text using NLTK's word_tokenize."""
    check_nltk()
    from nltk.tokenize import word_tokenize  # Local import after ensuring nltk

    return word_tokenize(text)


def spacy_tokenize(text: str, model_name: str = "en_core_web_sm") -> List[str]:
    """Tokenize text using spaCy."""
    nlp = check_spacy(model_name)
    doc = nlp(text)
    return [token.text for token in doc]


def segment_sentences(text: str, use_nltk: bool = True) -> List[str]:
    """Split text into sentences."""
    if use_nltk:
        check_nltk()
        from nltk.tokenize import sent_tokenize  # Local import after ensuring nltk

        return sent_tokenize(text)
    # Simple regex-based sentence segmentation
    pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
    return pattern.split(text)


def get_ngrams(tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
    """Generate n-grams from a list of tokens."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def remove_stopwords(tokens: List[str], language: str = "english") -> List[str]:
    """Remove stopwords from a list of tokens."""
    check_nltk()
    from nltk.corpus import stopwords  # Local import after ensuring nltk

    stop_words_set = set(stopwords.words(language))
    return [token for token in tokens if token.lower() not in stop_words_set]


def stem_words(tokens: List[str]) -> List[str]:
    """Apply stemming to a list of tokens using Porter stemmer."""
    check_nltk()
    from nltk.stem import PorterStemmer  # Local import after ensuring nltk

    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def lemmatize_words(tokens: List[str]) -> List[str]:
    """Apply lemmatization to a list of tokens using WordNet lemmatizer."""
    check_nltk()
    from nltk.stem import WordNetLemmatizer  # Local import after ensuring nltk

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]
