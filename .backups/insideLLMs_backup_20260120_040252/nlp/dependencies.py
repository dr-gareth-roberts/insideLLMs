"""Shared dependency helpers for NLP modules."""

from collections.abc import Iterable
from functools import lru_cache


@lru_cache
def ensure_nltk(resources: Iterable[str] = ()):
    """Import NLTK and ensure required resources are available."""
    try:
        import nltk
    except ImportError as exc:
        raise ImportError(
            "NLTK is not installed. Please install it with: pip install nltk"
        ) from exc

    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            # resource is like "tokenizers/punkt" -> download last segment
            nltk.download(resource.split("/")[-1])
    return nltk


@lru_cache
def ensure_spacy(model_name: str = "en_core_web_sm"):
    """Import spaCy and load the requested model (cached)."""
    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "spaCy is not installed. Please install it with: pip install spacy"
        ) from exc

    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise ImportError(
            f"spaCy model '{model_name}' not found. "
            f"Please install it with: python -m spacy download {model_name}"
        ) from exc


@lru_cache
def ensure_sklearn():
    """Import scikit-learn."""
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is not installed. Please install it with: pip install scikit-learn"
        ) from exc
    return None


@lru_cache
def ensure_gensim():
    """Import gensim."""
    try:
        import gensim  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "gensim is not installed. Please install it with: pip install gensim"
        ) from exc
    return None
