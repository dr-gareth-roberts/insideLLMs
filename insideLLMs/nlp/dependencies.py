"""Shared dependency helpers for NLP modules."""

from collections.abc import Iterable
from functools import lru_cache
import os
from pathlib import Path


@lru_cache
def ensure_nltk(resources: Iterable[str] = ()):
    """Import NLTK and ensure required resources are available."""
    try:
        import nltk
    except ImportError as exc:
        raise ImportError(
            "NLTK is not installed. Please install it with: pip install nltk"
        ) from exc

    expanded_resources: list[str] = []
    for resource in resources:
        expanded_resources.append(resource)
        # NLTK 3.8+ may require `punkt_tab` for `sent_tokenize` even when `punkt` exists.
        if resource == "tokenizers/punkt" or resource.startswith("tokenizers/punkt/"):
            expanded_resources.append("tokenizers/punkt_tab")

    nltk_data_env = os.environ.get("NLTK_DATA", "")
    download_dir = None
    if nltk_data_env:
        # NLTK_DATA can be a path list; pick the first entry as a download target.
        first_path = nltk_data_env.split(os.pathsep)[0]
        if first_path:
            download_dir = first_path
            Path(download_dir).mkdir(parents=True, exist_ok=True)

    for resource in expanded_resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            # resource is like "tokenizers/punkt" -> download last segment
            package = resource.strip("/").split("/")[-1]
            nltk.download(package, download_dir=download_dir, quiet=True)
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
