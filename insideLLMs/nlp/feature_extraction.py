from typing import Optional

from insideLLMs.nlp.dependencies import ensure_gensim, ensure_sklearn, ensure_spacy


# Backward compatibility aliases
check_spacy = ensure_spacy
check_sklearn = ensure_sklearn
check_gensim = ensure_gensim


# ===== Feature Extraction =====


def create_bow(
    texts: list[str], max_features: Optional[int] = None
) -> tuple[list[list[int]], list[str]]:
    """Create bag-of-words representation of texts."""
    ensure_sklearn()
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(texts)
    return X.toarray().tolist(), vectorizer.get_feature_names_out().tolist()


def create_tfidf(
    texts: list[str], max_features: Optional[int] = None
) -> tuple[list[list[float]], list[str]]:
    """Create TF-IDF representation of texts."""
    ensure_sklearn()
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(texts)
    return X.toarray().tolist(), vectorizer.get_feature_names_out().tolist()


def create_word_embeddings(
    sentences: list[list[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
) -> dict[str, list[float]]:
    """Create word embeddings using Word2Vec."""
    ensure_gensim()
    from gensim.models import Word2Vec

    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count)
    return {word: model.wv[word].tolist() for word in model.wv.index_to_key}


def extract_pos_tags(text: str, model_name: str = "en_core_web_sm") -> list[tuple[str, str]]:
    """Extract part-of-speech tags from text using spaCy."""
    nlp = ensure_spacy(model_name)
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]


def extract_dependencies(
    text: str, model_name: str = "en_core_web_sm"
) -> list[tuple[str, str, str]]:
    """Extract dependency relations from text using spaCy."""
    nlp = ensure_spacy(model_name)
    doc = nlp(text)
    return [(token.text, token.dep_, token.head.text) for token in doc]
