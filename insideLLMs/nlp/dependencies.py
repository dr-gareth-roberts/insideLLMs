"""
Shared dependency helpers for NLP modules.

This module provides lazy-loading utilities for optional NLP library dependencies.
Rather than requiring all NLP libraries to be installed upfront, these helper
functions allow modules to import dependencies only when needed and provide
clear, actionable error messages when dependencies are missing.

All functions are cached using ``functools.lru_cache`` to ensure that:
    - Import checks happen only once per session
    - Resource downloads (for NLTK) happen only once
    - Model loading (for spaCy) happens only once

Overview
--------
The module provides four main dependency helpers:

    - ``ensure_nltk``: Import NLTK and optionally download required resources
    - ``ensure_spacy``: Import spaCy and load a language model
    - ``ensure_sklearn``: Import scikit-learn
    - ``ensure_gensim``: Import gensim

Examples
--------
Basic usage to ensure NLTK is available with punkt tokenizer:

>>> from insideLLMs.nlp.dependencies import ensure_nltk
>>> nltk = ensure_nltk(resources=("tokenizers/punkt",))
>>> from nltk.tokenize import sent_tokenize
>>> sentences = sent_tokenize("Hello world. How are you?")
>>> len(sentences)
2

Loading a spaCy model for NLP processing:

>>> from insideLLMs.nlp.dependencies import ensure_spacy
>>> nlp = ensure_spacy("en_core_web_sm")
>>> doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
>>> [(ent.text, ent.label_) for ent in doc.ents]
[('Apple', 'ORG'), ('U.K.', 'GPE'), ('$1 billion', 'MONEY')]

Verifying scikit-learn is available before using it:

>>> from insideLLMs.nlp.dependencies import ensure_sklearn
>>> ensure_sklearn()
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> vectorizer = TfidfVectorizer()

Notes
-----
Environment Variables
    NLTK_DATA : str, optional
        Colon-separated list of paths where NLTK should look for data.
        If set, the first path is used as the download directory for
        any missing resources. The directory will be created if it
        doesn't exist.

Caching Behavior
    All functions use ``@lru_cache`` with unbounded cache size. This means:

    - First call performs the import/download/load
    - Subsequent calls return the cached result immediately
    - Cache persists for the lifetime of the Python process
    - To clear the cache, call ``function_name.cache_clear()``

NLTK 3.8+ Compatibility
    The ``ensure_nltk`` function automatically handles NLTK 3.8+ which
    requires ``punkt_tab`` in addition to ``punkt`` for sentence tokenization.
    When ``tokenizers/punkt`` is requested, ``tokenizers/punkt_tab`` is
    automatically added to the download list.

See Also
--------
nltk : Natural Language Toolkit for Python
spacy : Industrial-strength NLP library
sklearn : Machine learning library for Python
gensim : Topic modeling and document similarity library
"""

import os
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path


@lru_cache
def ensure_nltk(resources: Iterable[str] = ()):
    """
    Import NLTK and ensure required resources are available.

    This function performs a lazy import of NLTK and optionally downloads
    any required resources (corpora, tokenizers, taggers, etc.) that are
    not already present on the system. Downloads are performed quietly
    to avoid cluttering output.

    Parameters
    ----------
    resources : Iterable[str], optional
        An iterable of NLTK resource paths to ensure are available.
        Resource paths follow NLTK's data path convention, e.g.,
        ``"tokenizers/punkt"`` or ``"corpora/wordnet"``. If a resource
        is not found locally, it will be downloaded automatically.
        Default is an empty tuple (no resources downloaded).

    Returns
    -------
    module
        The imported ``nltk`` module, ready for use.

    Raises
    ------
    ImportError
        If NLTK is not installed. The error message includes the
        installation command: ``pip install nltk``.

    Examples
    --------
    Import NLTK without downloading any resources:

    >>> nltk = ensure_nltk()
    >>> nltk.__name__
    'nltk'

    Ensure the punkt tokenizer is available for sentence splitting:

    >>> nltk = ensure_nltk(resources=("tokenizers/punkt",))
    >>> from nltk.tokenize import sent_tokenize
    >>> text = "Dr. Smith went to the store. He bought apples."
    >>> sentences = sent_tokenize(text)
    >>> sentences
    ['Dr. Smith went to the store.', 'He bought apples.']

    Ensure multiple resources are available:

    >>> nltk = ensure_nltk(resources=(
    ...     "tokenizers/punkt",
    ...     "taggers/averaged_perceptron_tagger",
    ...     "corpora/wordnet",
    ... ))
    >>> from nltk import pos_tag, word_tokenize
    >>> pos_tag(word_tokenize("The quick brown fox"))
    [('The', 'DT'), ('quick', 'JJ'), ('brown', 'NN'), ('fox', 'NN')]

    Using a custom NLTK_DATA directory via environment variable:

    >>> import os
    >>> os.environ["NLTK_DATA"] = "/custom/nltk_data"
    >>> ensure_nltk.cache_clear()  # Clear cache to pick up new env var
    >>> nltk = ensure_nltk(resources=("tokenizers/punkt",))
    # Resources will be downloaded to /custom/nltk_data if not present

    Handling the case when NLTK is not installed:

    >>> try:
    ...     nltk = ensure_nltk()
    ... except ImportError as e:
    ...     print(f"NLTK not available: {e}")
    ...     # Fall back to alternative implementation

    Notes
    -----
    NLTK 3.8+ Compatibility
        Starting with NLTK 3.8, the sentence tokenizer requires both
        ``punkt`` and ``punkt_tab`` resources. This function automatically
        adds ``punkt_tab`` when ``punkt`` is requested, ensuring
        compatibility with both older and newer NLTK versions.

    Download Directory
        If the ``NLTK_DATA`` environment variable is set, resources are
        downloaded to the first path in the colon-separated list. The
        directory is created automatically if it doesn't exist.

    Caching
        Results are cached using ``@lru_cache``. Call
        ``ensure_nltk.cache_clear()`` to reset the cache and force
        re-import and re-download checks.

    See Also
    --------
    ensure_spacy : Import spaCy and load a language model.
    nltk.download : NLTK's built-in download function.
    nltk.data.find : NLTK's resource finder function.
    """
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
    """
    Import spaCy and load the requested language model.

    This function performs a lazy import of spaCy and loads the specified
    language model. The loaded model is cached, so subsequent calls with
    the same model name return the cached model instance immediately.

    Parameters
    ----------
    model_name : str, optional
        The name of the spaCy model to load. This can be:

        - A model package name (e.g., ``"en_core_web_sm"``)
        - A path to a model directory
        - A shortcut link name

        Default is ``"en_core_web_sm"``, the small English model.

    Returns
    -------
    spacy.language.Language
        The loaded spaCy language model (``nlp`` object), ready for
        processing text with ``nlp(text)``.

    Raises
    ------
    ImportError
        If spaCy is not installed. The error message includes the
        installation command: ``pip install spacy``.
    ImportError
        If the specified model is not found. The error message includes
        the installation command: ``python -m spacy download <model_name>``.

    Examples
    --------
    Load the default English model:

    >>> nlp = ensure_spacy()
    >>> doc = nlp("Hello, world!")
    >>> [token.text for token in doc]
    ['Hello', ',', 'world', '!']

    Load a specific model for named entity recognition:

    >>> nlp = ensure_spacy("en_core_web_sm")
    >>> doc = nlp("Apple Inc. was founded by Steve Jobs in Cupertino.")
    >>> for ent in doc.ents:
    ...     print(f"{ent.text}: {ent.label_}")
    Apple Inc.: ORG
    Steve Jobs: PERSON
    Cupertino: GPE

    Load a larger model for better accuracy:

    >>> nlp = ensure_spacy("en_core_web_lg")
    >>> doc = nlp("The bank by the river bank")
    >>> # Larger models provide better word vectors for disambiguation

    Load a model for a different language:

    >>> nlp_de = ensure_spacy("de_core_news_sm")
    >>> doc = nlp_de("Berlin ist die Hauptstadt von Deutschland.")
    >>> [ent.text for ent in doc.ents]
    ['Berlin', 'Deutschland']

    Handle missing model gracefully:

    >>> try:
    ...     nlp = ensure_spacy("en_core_web_trf")
    ... except ImportError as e:
    ...     print("Transformer model not installed, using small model")
    ...     nlp = ensure_spacy("en_core_web_sm")

    Use the model for dependency parsing:

    >>> nlp = ensure_spacy()
    >>> doc = nlp("The quick brown fox jumps over the lazy dog.")
    >>> for token in doc:
    ...     print(f"{token.text}: {token.dep_} -> {token.head.text}")
    The: det -> fox
    quick: amod -> fox
    brown: amod -> fox
    fox: nsubj -> jumps
    jumps: ROOT -> jumps
    over: prep -> jumps
    the: det -> dog
    lazy: amod -> dog
    dog: pobj -> over
    .: punct -> jumps

    Notes
    -----
    Available Models
        Common spaCy models include:

        - ``en_core_web_sm``: Small English model (~12 MB)
        - ``en_core_web_md``: Medium English model with word vectors (~40 MB)
        - ``en_core_web_lg``: Large English model with word vectors (~560 MB)
        - ``en_core_web_trf``: Transformer-based English model (highest accuracy)

        See https://spacy.io/models for the full list of available models.

    Caching
        The loaded model is cached using ``@lru_cache``. This means:

        - Multiple calls with the same model name share one model instance
        - Different model names result in separate cached instances
        - Call ``ensure_spacy.cache_clear()`` to unload all cached models

    Memory Considerations
        spaCy models can be memory-intensive. The transformer models
        (``*_trf``) require significantly more memory. Consider using
        smaller models for development or when processing many documents.

    See Also
    --------
    ensure_nltk : Import NLTK and download resources.
    spacy.load : spaCy's built-in model loader.
    spacy.blank : Create a blank model without pre-trained weights.
    """
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
    """
    Import scikit-learn and verify it is available.

    This function performs a lazy import of scikit-learn to verify the
    library is installed. Unlike ``ensure_nltk`` and ``ensure_spacy``,
    this function does not return the module or load any models; it
    simply validates that scikit-learn can be imported.

    Returns
    -------
    None
        This function returns ``None`` on success. The purpose is to
        validate the import, not to return the module.

    Raises
    ------
    ImportError
        If scikit-learn is not installed. The error message includes
        the installation command: ``pip install scikit-learn``.

    Examples
    --------
    Verify scikit-learn is available before importing submodules:

    >>> ensure_sklearn()
    >>> from sklearn.feature_extraction.text import TfidfVectorizer
    >>> vectorizer = TfidfVectorizer(max_features=1000)

    Use in a feature extraction pipeline:

    >>> ensure_sklearn()
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> from sklearn.decomposition import LatentDirichletAllocation
    >>>
    >>> corpus = [
    ...     "Machine learning is fascinating.",
    ...     "Natural language processing uses machine learning.",
    ...     "Deep learning is a subset of machine learning.",
    ... ]
    >>> vectorizer = CountVectorizer()
    >>> X = vectorizer.fit_transform(corpus)
    >>> X.shape
    (3, 10)

    Conditional import with fallback:

    >>> try:
    ...     ensure_sklearn()
    ...     from sklearn.metrics.pairwise import cosine_similarity
    ...     USE_SKLEARN = True
    ... except ImportError:
    ...     USE_SKLEARN = False
    ...     print("scikit-learn not available, using fallback")

    Use for text classification:

    >>> ensure_sklearn()
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.feature_extraction.text import TfidfVectorizer
    >>>
    >>> pipeline = Pipeline([
    ...     ('tfidf', TfidfVectorizer()),
    ...     ('clf', MultinomialNB()),
    ... ])
    >>> # pipeline.fit(texts, labels)

    Validate before using clustering algorithms:

    >>> ensure_sklearn()
    >>> from sklearn.cluster import KMeans
    >>> from sklearn.feature_extraction.text import TfidfVectorizer
    >>>
    >>> documents = ["doc one", "doc two", "doc three"]
    >>> vectorizer = TfidfVectorizer()
    >>> X = vectorizer.fit_transform(documents)
    >>> kmeans = KMeans(n_clusters=2, random_state=42)
    >>> clusters = kmeans.fit_predict(X)

    Notes
    -----
    Why No Module Return?
        Unlike ``ensure_nltk`` which returns the nltk module, this function
        returns ``None`` because scikit-learn's submodules are typically
        imported directly (e.g., ``from sklearn.feature_extraction.text
        import TfidfVectorizer``). Returning the top-level module would
        not be useful for most use cases.

    Caching
        The import check is cached using ``@lru_cache``. After the first
        successful call, subsequent calls return immediately without
        re-importing. Call ``ensure_sklearn.cache_clear()`` to force
        a re-check.

    See Also
    --------
    ensure_gensim : Import gensim for topic modeling.
    ensure_nltk : Import NLTK for text processing.
    sklearn.feature_extraction.text : Text feature extraction utilities.
    """
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is not installed. Please install it with: pip install scikit-learn"
        ) from exc
    return None


@lru_cache
def ensure_gensim():
    """
    Import gensim and verify it is available.

    This function performs a lazy import of gensim to verify the library
    is installed. Gensim is used for topic modeling, document similarity,
    and word vector operations.

    Returns
    -------
    None
        This function returns ``None`` on success. The purpose is to
        validate the import, not to return the module.

    Raises
    ------
    ImportError
        If gensim is not installed. The error message includes the
        installation command: ``pip install gensim``.

    Examples
    --------
    Verify gensim is available before using Word2Vec:

    >>> ensure_gensim()
    >>> from gensim.models import Word2Vec
    >>>
    >>> sentences = [
    ...     ["machine", "learning", "is", "fun"],
    ...     ["natural", "language", "processing"],
    ...     ["deep", "learning", "neural", "networks"],
    ... ]
    >>> model = Word2Vec(sentences, vector_size=100, min_count=1)

    Use for topic modeling with LDA:

    >>> ensure_gensim()
    >>> from gensim import corpora
    >>> from gensim.models import LdaModel
    >>>
    >>> documents = [
    ...     ["human", "machine", "interface"],
    ...     ["survey", "user", "computer", "system"],
    ...     ["graph", "trees", "algorithms"],
    ... ]
    >>> dictionary = corpora.Dictionary(documents)
    >>> corpus = [dictionary.doc2bow(doc) for doc in documents]
    >>> lda = LdaModel(corpus, num_topics=2, id2word=dictionary)

    Load pre-trained word vectors:

    >>> ensure_gensim()
    >>> from gensim.models import KeyedVectors
    >>> # vectors = KeyedVectors.load_word2vec_format('vectors.bin', binary=True)
    >>> # similar = vectors.most_similar('king')

    Document similarity with Doc2Vec:

    >>> ensure_gensim()
    >>> from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    >>>
    >>> documents = [
    ...     TaggedDocument(["machine", "learning"], [0]),
    ...     TaggedDocument(["deep", "learning"], [1]),
    ... ]
    >>> model = Doc2Vec(documents, vector_size=50, min_count=1, epochs=10)

    Conditional import with fallback:

    >>> try:
    ...     ensure_gensim()
    ...     from gensim.models import Word2Vec
    ...     HAS_GENSIM = True
    ... except ImportError:
    ...     HAS_GENSIM = False
    ...     print("gensim not available, using alternative")

    Use for text similarity:

    >>> ensure_gensim()
    >>> from gensim import similarities
    >>> from gensim import corpora
    >>>
    >>> texts = [["computer", "science"], ["data", "science"]]
    >>> dictionary = corpora.Dictionary(texts)
    >>> corpus = [dictionary.doc2bow(text) for text in texts]
    >>> index = similarities.SparseMatrixSimilarity(corpus, num_features=len(dictionary))

    Notes
    -----
    Why No Module Return?
        Similar to ``ensure_sklearn``, this function returns ``None``
        because gensim submodules are typically imported directly
        (e.g., ``from gensim.models import Word2Vec``).

    Common Use Cases
        Gensim is particularly useful for:

        - Word embeddings (Word2Vec, FastText)
        - Topic modeling (LDA, LSI)
        - Document similarity
        - Phrase detection

    Memory Considerations
        Gensim models, especially word vector models, can be memory-intensive.
        Pre-trained models like Google's Word2Vec (3 million words, 300
        dimensions) require several gigabytes of RAM.

    Caching
        The import check is cached using ``@lru_cache``. Call
        ``ensure_gensim.cache_clear()`` to force a re-check.

    See Also
    --------
    ensure_sklearn : Import scikit-learn for machine learning.
    ensure_nltk : Import NLTK for text processing.
    gensim.models.Word2Vec : Word2Vec implementation.
    gensim.models.LdaModel : Latent Dirichlet Allocation.
    """
    try:
        import gensim  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "gensim is not installed. Please install it with: pip install gensim"
        ) from exc
    return None
