import re
from typing import List, Tuple

# Optional dependencies
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = False  # Set to False initially, will be True when a model is loaded
    SPACY_MODEL = None
except ImportError:
    SPACY_AVAILABLE = False
    SPACY_MODEL = None

# ===== Dependency Management =====

def check_nltk():
    """Check if NLTK is available and download required resources if needed."""
    if not NLTK_AVAILABLE:
        raise ImportError(
            "NLTK is not installed. Please install it with: pip install nltk"
        )

    # Download required NLTK resources if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

def check_spacy(model_name: str = "en_core_web_sm"):
    """Check if spaCy is available and load the specified model if needed.

    Args:
        model_name: Name of the spaCy model to load

    Returns:
        The loaded spaCy model
    """
    global SPACY_AVAILABLE, SPACY_MODEL

    if not SPACY_AVAILABLE: # This was the original check, but it's always False here.
                            # We should check for the spacy module itself first.
        try:
            import spacy as spacy_check_module
        except ImportError:
             raise ImportError(
                "spaCy is not installed. Please install it with: pip install spacy"
            )
        # If we reach here, spacy is installed, so set SPACY_AVAILABLE to True
        # This global variable modification is tricky across modules.
        # For now, let's assume this function is self-contained or SPACY_AVAILABLE
        # is managed correctly at a higher level if these files become part of a package.


    if SPACY_MODEL is None or SPACY_MODEL.meta['name'] != model_name:
        try:
            # Ensure spacy is actually imported before trying to load
            if 'spacy' not in globals() and 'spacy' not in locals():
                 import spacy as spacy_load_module # import with a different alias if needed
                 SPACY_MODEL = spacy_load_module.load(model_name)
            else:
                 SPACY_MODEL = spacy.load(model_name)
            # SPACY_AVAILABLE = True # This should be set once spacy itself is confirmed.
                                  # The global SPACY_AVAILABLE might be an issue.
        except OSError:
            # Try to re-assign SPACY_AVAILABLE in case it was True but model loading failed
            # global SPACY_AVAILABLE # Required to modify global variable
            # SPACY_AVAILABLE = False # Indicate that the required model is not available
            raise ImportError(
                f"spaCy model '{model_name}' not found. "
                f"Please install it with: python -m spacy download {model_name}"
            )
        except NameError: # If spacy was not imported
             raise ImportError(
                "spaCy is not installed. Please install it with: pip install spacy"
            )


    return SPACY_MODEL

# ===== Tokenization and Segmentation =====

def simple_tokenize(text: str) -> List[str]:
    """Simple word tokenization by splitting on whitespace.

    Args:
        text: Input text to tokenize

    Returns:
        List of tokens
    """
    return text.split()

def nltk_tokenize(text: str) -> List[str]:
    """Tokenize text using NLTK's word_tokenize.

    Args:
        text: Input text to tokenize

    Returns:
        List of tokens
    """
    check_nltk()
    return word_tokenize(text)

def spacy_tokenize(text: str, model_name: str = "en_core_web_sm") -> List[str]:
    """Tokenize text using spaCy.

    Args:
        text: Input text to tokenize
        model_name: Name of the spaCy model to use

    Returns:
        List of tokens
    """
    nlp = check_spacy(model_name)
    doc = nlp(text)
    return [token.text for token in doc]

def segment_sentences(text: str, use_nltk: bool = True) -> List[str]:
    """Split text into sentences.

    Args:
        text: Input text to segment
        use_nltk: Whether to use NLTK (True) or a simple regex (False)

    Returns:
        List of sentences
    """
    if use_nltk:
        check_nltk()
        return sent_tokenize(text)
    else:
        # Simple regex-based sentence segmentation
        pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
        return pattern.split(text)

def get_ngrams(tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
    """Generate n-grams from a list of tokens.

    Args:
        tokens: List of tokens
        n: Size of n-grams

    Returns:
        List of n-grams as tuples
    """
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def remove_stopwords(tokens: List[str], language: str = 'english') -> List[str]:
    """Remove stopwords from a list of tokens.

    Args:
        tokens: List of tokens
        language: Language for stopwords

    Returns:
        List of tokens with stopwords removed
    """
    check_nltk()
    stop_words_set = set(stopwords.words(language))
    return [token for token in tokens if token.lower() not in stop_words_set]

def stem_words(tokens: List[str]) -> List[str]:
    """Apply stemming to a list of tokens using Porter stemmer.

    Args:
        tokens: List of tokens

    Returns:
        List of stemmed tokens
    """
    check_nltk()
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def lemmatize_words(tokens: List[str]) -> List[str]:
    """Apply lemmatization to a list of tokens using WordNet lemmatizer.

    Args:
        tokens: List of tokens

    Returns:
        List of lemmatized tokens
    """
    check_nltk()
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

# Re-initialize SPACY_AVAILABLE based on successful import of spacy module
# This is a workaround for the global variable issue.
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
