from typing import List, Dict, Tuple, Optional

# Optional dependencies
try:
    import spacy
    SPACY_AVAILABLE = False 
    SPACY_MODEL = None
except ImportError:
    SPACY_AVAILABLE = False
    SPACY_MODEL = None

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import gensim
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

# ===== Dependency Management =====

def check_spacy(model_name: str = "en_core_web_sm"):
    """Check if spaCy is available and load the specified model if needed."""
    global SPACY_AVAILABLE, SPACY_MODEL
    if not SPACY_AVAILABLE:
        try:
            import spacy as spacy_check_module
            SPACY_AVAILABLE = True
        except ImportError:
            raise ImportError("spaCy is not installed. Please install it with: pip install spacy")

    if SPACY_MODEL is None or SPACY_MODEL.meta['name'] != model_name:
        try:
            if 'spacy' not in globals() and 'spacy' not in locals():
                import spacy as spacy_load_module
                SPACY_MODEL = spacy_load_module.load(model_name)
            elif SPACY_MODEL is None or SPACY_MODEL.meta['name'] != model_name :
                 SPACY_MODEL = spacy.load(model_name)
        except OSError:
            raise ImportError(
                f"spaCy model '{model_name}' not found. "
                f"Please install it with: python -m spacy download {model_name}"
            )
        except NameError:
            raise ImportError("spaCy is not installed. Please install it with: pip install spacy")
    return SPACY_MODEL

def check_sklearn():
    """Check if scikit-learn is available."""
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is not installed. Please install it with: pip install scikit-learn"
        )

def check_gensim():
    """Check if gensim is available."""
    if not GENSIM_AVAILABLE:
        raise ImportError(
            "gensim is not installed. Please install it with: pip install gensim"
        )

# ===== Feature Extraction =====

def create_bow(texts: List[str], max_features: Optional[int] = None) -> Tuple[List[List[int]], List[str]]:
    """Create bag-of-words representation of texts.

    Args:
        texts: List of text documents
        max_features: Maximum number of features (words)

    Returns:
        Tuple of (document-term matrix, feature names)
    """
    check_sklearn()
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return X.toarray().tolist(), vectorizer.get_feature_names_out().tolist()

def create_tfidf(texts: List[str], max_features: Optional[int] = None) -> Tuple[List[List[float]], List[str]]:
    """Create TF-IDF representation of texts.

    Args:
        texts: List of text documents
        max_features: Maximum number of features (words)

    Returns:
        Tuple of (document-term matrix, feature names)
    """
    check_sklearn()
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return X.toarray().tolist(), vectorizer.get_feature_names_out().tolist()

def create_word_embeddings(sentences: List[List[str]],
                          vector_size: int = 100,
                          window: int = 5,
                          min_count: int = 1) -> Dict[str, List[float]]:
    """Create word embeddings using Word2Vec.

    Args:
        sentences: List of tokenized sentences
        vector_size: Dimensionality of the word vectors
        window: Maximum distance between current and predicted word
        min_count: Minimum count of words to consider

    Returns:
        Dictionary mapping words to their vector representations
    """
    check_gensim()
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count)
    return {word: model.wv[word].tolist() for word in model.wv.index_to_key}

def extract_pos_tags(text: str, model_name: str = "en_core_web_sm") -> List[Tuple[str, str]]:
    """Extract part-of-speech tags from text using spaCy.

    Args:
        text: Input text
        model_name: Name of the spaCy model to use

    Returns:
        List of (token, POS tag) tuples
    """
    nlp = check_spacy(model_name)
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

def extract_dependencies(text: str, model_name: str = "en_core_web_sm") -> List[Tuple[str, str, str]]:
    """Extract dependency relations from text using spaCy.

    Args:
        text: Input text
        model_name: Name of the spaCy model to use

    Returns:
        List of (token, dependency relation, head) tuples
    """
    nlp = check_spacy(model_name)
    doc = nlp(text)
    return [(token.text, token.dep_, token.head.text) for token in doc]

# Re-initialize SPACY_AVAILABLE based on successful import of spacy module
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
