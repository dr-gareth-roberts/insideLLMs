from typing import Dict, List, Optional, Tuple

from insideLLMs.nlp.dependencies import ensure_gensim, ensure_sklearn, ensure_spacy


# ===== Dependency Management =====

def check_spacy(model_name: str = "en_core_web_sm"):
    """Ensure spaCy and the requested model are available."""
    return ensure_spacy(model_name)


def check_sklearn():
    """Ensure scikit-learn is available."""
    ensure_sklearn()


def check_gensim():
    """Ensure gensim is available."""
    ensure_gensim()


# ===== Feature Extraction =====

def create_bow(texts: List[str], max_features: Optional[int] = None) -> Tuple[List[List[int]], List[str]]:
    """Create bag-of-words representation of texts."""
    check_sklearn()
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return X.toarray().tolist(), vectorizer.get_feature_names_out().tolist()


def create_tfidf(texts: List[str], max_features: Optional[int] = None) -> Tuple[List[List[float]], List[str]]:
    """Create TF-IDF representation of texts."""
    check_sklearn()
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return X.toarray().tolist(), vectorizer.get_feature_names_out().tolist()


def create_word_embeddings(
    sentences: List[List[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
) -> Dict[str, List[float]]:
    """Create word embeddings using Word2Vec."""
    check_gensim()
    from gensim.models import Word2Vec

    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count)
    return {word: model.wv[word].tolist() for word in model.wv.index_to_key}


def extract_pos_tags(text: str, model_name: str = "en_core_web_sm") -> List[Tuple[str, str]]:
    """Extract part-of-speech tags from text using spaCy."""
    nlp = check_spacy(model_name)
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]


def extract_dependencies(text: str, model_name: str = "en_core_web_sm") -> List[Tuple[str, str, str]]:
    """Extract dependency relations from text using spaCy."""
    nlp = check_spacy(model_name)
    doc = nlp(text)
    return [(token.text, token.dep_, token.head.text) for token in doc]
