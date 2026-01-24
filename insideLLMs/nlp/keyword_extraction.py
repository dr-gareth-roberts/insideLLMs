"""
Keyword Extraction Module for Natural Language Processing.

This module provides implementations of popular keyword extraction algorithms
for identifying the most important terms in text documents. It includes both
statistical (TF-IDF) and graph-based (TextRank) approaches.

Overview
--------
Keyword extraction is a fundamental NLP task that identifies the most relevant
terms in a document. This module provides two complementary approaches:

1. **TF-IDF (Term Frequency-Inverse Document Frequency)**: A statistical measure
   that evaluates word importance based on frequency within a document relative
   to a corpus (here, sentences serve as the corpus).

2. **TextRank**: A graph-based ranking algorithm inspired by PageRank that
   identifies keywords through co-occurrence relationships in sliding windows.

Key Features
------------
- Automatic handling of NLTK and scikit-learn dependencies
- Stopword removal and text preprocessing
- Configurable number of keywords to extract
- Adjustable window size for TextRank algorithm

Examples
--------
Basic TF-IDF keyword extraction:

>>> from insideLLMs.nlp.keyword_extraction import extract_keywords_tfidf
>>> text = '''
...     Machine learning is a subset of artificial intelligence.
...     Deep learning is a subset of machine learning.
...     Neural networks are fundamental to deep learning systems.
... '''
>>> keywords = extract_keywords_tfidf(text, num_keywords=5)
>>> print(keywords)
['learning', 'machine', 'deep', 'subset', 'neural']

TextRank keyword extraction with custom window size:

>>> from insideLLMs.nlp.keyword_extraction import extract_keywords_textrank
>>> text = '''
...     Python is a versatile programming language.
...     Python supports multiple programming paradigms.
...     Many developers prefer Python for data science.
... '''
>>> keywords = extract_keywords_textrank(text, num_keywords=3, window_size=3)
>>> print(keywords)
['python', 'programming', 'data']

Comparing both methods on the same text:

>>> from insideLLMs.nlp.keyword_extraction import (
...     extract_keywords_tfidf,
...     extract_keywords_textrank,
... )
>>> article = '''
...     Climate change affects global ecosystems. Rising temperatures
...     impact biodiversity. Scientists study climate patterns to
...     understand environmental changes. Conservation efforts aim
...     to protect vulnerable species from climate impacts.
... '''
>>> tfidf_keywords = extract_keywords_tfidf(article, num_keywords=4)
>>> textrank_keywords = extract_keywords_textrank(article, num_keywords=4)
>>> print(f"TF-IDF: {tfidf_keywords}")
TF-IDF: ['climate', 'changes', 'environmental', 'biodiversity']
>>> print(f"TextRank: {textrank_keywords}")
TextRank: ['climate', 'species', 'environmental', 'scientists']

Notes
-----
- TF-IDF works best on longer documents with multiple sentences, as it
  uses sentence-level document frequency calculations.
- TextRank performs better on texts with natural co-occurrence patterns
  and may capture semantic relationships between terms.
- Both methods automatically handle stopword removal and lowercasing.
- Empty or very short texts may return fewer keywords than requested.

Dependencies
------------
- NLTK: Required for tokenization and stopword removal
- scikit-learn: Required for TF-IDF vectorization

See Also
--------
insideLLMs.nlp.tokenization : Tokenization utilities used by this module
insideLLMs.nlp.dependencies : Dependency management for NLP packages

References
----------
.. [1] Salton, G., & Buckley, C. (1988). Term-weighting approaches in
       automatic text retrieval. Information Processing & Management.
.. [2] Mihalcea, R., & Tarau, P. (2004). TextRank: Bringing Order into
       Texts. Proceedings of EMNLP.
"""

from collections import defaultdict

from insideLLMs.nlp.dependencies import ensure_nltk, ensure_sklearn
from insideLLMs.nlp.tokenization import (
    nltk_tokenize,
    remove_stopwords,
    segment_sentences,
)


def _ensure_nltk_keyword():
    """
    Ensure NLTK and required resources are available for keyword extraction.

    This internal function checks that NLTK is installed and downloads
    the necessary resources (punkt tokenizer and stopwords corpus) if
    they are not already available.

    Returns
    -------
    None
        This function has no return value but may raise ImportError if
        NLTK cannot be installed.

    Raises
    ------
    ImportError
        If NLTK is not installed and cannot be automatically installed.

    Notes
    -----
    This function is called automatically by keyword extraction functions.
    Users typically do not need to call it directly.

    Examples
    --------
    >>> from insideLLMs.nlp.keyword_extraction import _ensure_nltk_keyword
    >>> _ensure_nltk_keyword()  # Ensures NLTK resources are available
    """
    ensure_nltk(("tokenizers/punkt", "corpora/stopwords"))


# Backward compatibility aliases
# These aliases are provided for backward compatibility with older code that
# imported these functions directly from this module before refactoring.
check_nltk = _ensure_nltk_keyword
"""Alias for _ensure_nltk_keyword. Deprecated: use _ensure_nltk_keyword instead."""

check_sklearn = ensure_sklearn
"""Alias for ensure_sklearn from dependencies module. Deprecated."""

segment_sentences_for_keyword = segment_sentences
"""Alias for segment_sentences from tokenization module. Deprecated."""

nltk_tokenize_for_keyword = nltk_tokenize
"""Alias for nltk_tokenize from tokenization module. Deprecated."""

remove_stopwords_for_keyword = remove_stopwords
"""Alias for remove_stopwords from tokenization module. Deprecated."""


# ===== Keyword Extraction =====


def extract_keywords_tfidf(text: str, num_keywords: int = 5) -> list[str]:
    """
    Extract keywords from text using TF-IDF (Term Frequency-Inverse Document Frequency).

    This function identifies the most important keywords in a text by computing
    TF-IDF scores across sentences. Words that appear frequently in some sentences
    but not others receive higher scores, making them good keyword candidates.

    Parameters
    ----------
    text : str
        The input text from which to extract keywords. Should contain at least
        one complete sentence for meaningful results. Longer texts with multiple
        sentences produce more reliable keyword rankings.
    num_keywords : int, optional
        The maximum number of keywords to return. Default is 5. If the text
        contains fewer unique non-stopword terms than requested, fewer keywords
        will be returned.

    Returns
    -------
    list[str]
        A list of keywords sorted by TF-IDF score in descending order. Returns
        an empty list if the text is empty, contains only stopwords, or cannot
        be processed.

    Raises
    ------
    ImportError
        If scikit-learn is not installed and cannot be automatically installed.

    Examples
    --------
    Extract keywords from a technical paragraph:

    >>> from insideLLMs.nlp.keyword_extraction import extract_keywords_tfidf
    >>> text = '''
    ...     Natural language processing enables computers to understand text.
    ...     Machine learning algorithms power modern NLP systems.
    ...     Deep learning has revolutionized natural language understanding.
    ... '''
    >>> keywords = extract_keywords_tfidf(text, num_keywords=5)
    >>> print(keywords)
    ['learning', 'natural', 'language', 'deep', 'machine']

    Extract keywords from a news article excerpt:

    >>> article = '''
    ...     The stock market experienced significant volatility today.
    ...     Investors reacted to economic indicators and policy announcements.
    ...     Technology stocks led the market decline. Energy sector remained stable.
    ...     Analysts predict continued market uncertainty in coming weeks.
    ... '''
    >>> keywords = extract_keywords_tfidf(article, num_keywords=4)
    >>> print(keywords)
    ['market', 'stocks', 'sector', 'volatility']

    Handle edge case with short text:

    >>> short_text = "Hello world."
    >>> extract_keywords_tfidf(short_text, num_keywords=3)
    ['hello', 'world']

    Handle empty or stopword-only text:

    >>> extract_keywords_tfidf("")
    []
    >>> extract_keywords_tfidf("the and or but")
    []

    Extract more keywords than available:

    >>> text = "Python programming is fun."
    >>> keywords = extract_keywords_tfidf(text, num_keywords=10)
    >>> len(keywords) <= 10  # May return fewer than requested
    True

    Notes
    -----
    The algorithm works as follows:

    1. Segment the input text into sentences using NLTK's sentence tokenizer.
    2. Create a TF-IDF matrix where each sentence is a document.
    3. Sum TF-IDF scores across all sentences for each term.
    4. Sort terms by aggregated score and return the top N.

    This approach treats each sentence as a separate document, which allows
    the IDF component to identify terms that are distinctive within the text.
    Terms appearing in every sentence will have lower IDF scores.

    The function uses scikit-learn's TfidfVectorizer with English stopwords
    removed automatically.

    See Also
    --------
    extract_keywords_textrank : Graph-based keyword extraction using TextRank
    insideLLMs.nlp.tokenization.segment_sentences : Sentence segmentation utility
    """
    ensure_sklearn()
    from sklearn.feature_extraction.text import TfidfVectorizer

    sentences = segment_sentences_for_keyword(text)
    if not sentences:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()
    if not feature_names.any():
        return []

    aggregated_scores = tfidf_matrix.sum(axis=0).A1
    sorted_words = sorted(
        zip(feature_names, aggregated_scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return [word for word, _ in sorted_words[:num_keywords]]


def extract_keywords_textrank(text: str, num_keywords: int = 5, window_size: int = 4) -> list[str]:
    """
    Extract keywords from text using the TextRank algorithm.

    TextRank is a graph-based ranking algorithm for keyword extraction, inspired
    by Google's PageRank. It builds a co-occurrence graph of words and iteratively
    computes importance scores based on the structure of word relationships.

    Parameters
    ----------
    text : str
        The input text from which to extract keywords. The text is lowercased
        and tokenized before processing. Works best with coherent text where
        related words appear near each other.
    num_keywords : int, optional
        The maximum number of keywords to return. Default is 5. If the text
        contains fewer unique non-stopword terms than requested, fewer keywords
        will be returned.
    window_size : int, optional
        The size of the sliding window used to establish co-occurrence
        relationships between words. Default is 4. Larger windows capture
        broader relationships but may introduce noise; smaller windows are
        more precise but may miss some connections.

    Returns
    -------
    list[str]
        A list of keywords sorted by TextRank score in descending order.
        All keywords are lowercase. Returns an empty list if the text is
        empty or contains only stopwords.

    Raises
    ------
    ImportError
        If NLTK is not installed and cannot be automatically installed.

    Examples
    --------
    Basic keyword extraction from a research abstract:

    >>> from insideLLMs.nlp.keyword_extraction import extract_keywords_textrank
    >>> abstract = '''
    ...     Graph neural networks have emerged as powerful tools for learning
    ...     on structured data. These networks combine graph structure with
    ...     neural network architectures to capture complex relationships.
    ...     Applications include social network analysis and molecular property
    ...     prediction.
    ... '''
    >>> keywords = extract_keywords_textrank(abstract, num_keywords=5)
    >>> print(keywords)
    ['networks', 'graph', 'neural', 'network', 'structured']

    Adjusting window size for different granularity:

    >>> text = '''
    ...     Python developers often use pandas for data analysis.
    ...     Data scientists prefer Python for machine learning tasks.
    ...     Python libraries simplify complex data processing workflows.
    ... '''
    >>> # Smaller window - tighter word relationships
    >>> keywords_small = extract_keywords_textrank(text, num_keywords=3, window_size=2)
    >>> print(keywords_small)
    ['python', 'data', 'learning']
    >>> # Larger window - broader relationships
    >>> keywords_large = extract_keywords_textrank(text, num_keywords=3, window_size=6)
    >>> print(keywords_large)
    ['python', 'data', 'developers']

    Extract keywords from a product review:

    >>> review = '''
    ...     This laptop has excellent battery life and a beautiful display.
    ...     The keyboard feels responsive and comfortable for long typing sessions.
    ...     Performance is smooth for everyday tasks. Great value for the price.
    ... '''
    >>> keywords = extract_keywords_textrank(review, num_keywords=4)
    >>> print(keywords)
    ['battery', 'keyboard', 'display', 'performance']

    Handle edge cases:

    >>> extract_keywords_textrank("")
    []
    >>> extract_keywords_textrank("the and or")  # Only stopwords
    []
    >>> extract_keywords_textrank("Python")  # Single word
    []

    Compare different window sizes on the same text:

    >>> text = "Machine learning algorithms process large datasets efficiently."
    >>> for ws in [2, 4, 6]:
    ...     kw = extract_keywords_textrank(text, num_keywords=3, window_size=ws)
    ...     print(f"Window {ws}: {kw}")
    Window 2: ['learning', 'machine', 'algorithms']
    Window 4: ['learning', 'machine', 'large']
    Window 6: ['learning', 'datasets', 'machine']

    Notes
    -----
    The TextRank algorithm works as follows:

    1. **Tokenization**: The text is lowercased and tokenized into words.
       Stopwords are removed to focus on content words.

    2. **Graph Construction**: A weighted undirected graph is built where:
       - Each unique word is a node
       - Edges connect words that co-occur within the sliding window
       - Edge weights represent co-occurrence frequency

    3. **Ranking**: The PageRank-style algorithm iteratively computes scores:

       .. math::

           WS(V_i) = (1-d) + d \\times \\sum_{V_j \\in In(V_i)}
                     \\frac{w_{ji}}{\\sum_{V_k \\in Out(V_j)} w_{jk}} \\times WS(V_j)

       where:
       - :math:`d` is the damping factor (0.85)
       - :math:`w_{ji}` is the edge weight between nodes
       - :math:`In(V_i)` is the set of nodes pointing to :math:`V_i`

    4. **Selection**: Nodes are sorted by final score and top N are returned.

    The implementation uses 10 iterations, which typically provides good
    convergence for most texts.

    Choosing Window Size
    ~~~~~~~~~~~~~~~~~~~~
    - **window_size=2-3**: Very strict co-occurrence, captures only adjacent
      or nearly adjacent words. Best for technical terms.
    - **window_size=4-5** (default): Balanced approach that captures semantic
      relationships while limiting noise.
    - **window_size=6+**: Broader relationships, may capture thematic keywords
      but can introduce spurious connections.

    See Also
    --------
    extract_keywords_tfidf : Statistical keyword extraction using TF-IDF
    insideLLMs.nlp.tokenization.nltk_tokenize : Tokenization utility
    insideLLMs.nlp.tokenization.remove_stopwords : Stopword removal utility

    References
    ----------
    .. [1] Mihalcea, R., & Tarau, P. (2004). TextRank: Bringing Order into
           Texts. Proceedings of the 2004 Conference on Empirical Methods
           in Natural Language Processing (EMNLP).
    """
    _ensure_nltk_keyword()

    tokens = nltk_tokenize_for_keyword(text.lower())
    tokens = remove_stopwords_for_keyword(tokens)
    if not tokens:
        return []

    graph = defaultdict(lambda: defaultdict(int))
    # Populates weighted co-occurrence graph from token window
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i : i + window_size]
        for j in range(len(window)):
            # Populates weighted co-occurrence graph from token window
            for k in range(j + 1, len(window)):
                if window[j] == window[k]:
                    continue
                graph[window[j]][window[k]] += 1
                graph[window[k]][window[j]] += 1

    if not graph:
        return []

    ranks = dict.fromkeys(graph, 1.0)
    num_iterations = 10
    damping = 0.85

    # Iteratively refines keyword ranks until convergence
    for _ in range(num_iterations):
        new_ranks = ranks.copy()
        for node in graph:
            rank_sum = 0
            # Accumulates neighbor rank contribution to current node
            for neighbor in graph[node]:
                neighbor_links_sum = sum(graph[neighbor].values())
                if neighbor_links_sum > 0:
                    rank_sum += (graph[node][neighbor] / neighbor_links_sum) * ranks[neighbor]
            new_ranks[node] = (1 - damping) + damping * rank_sum
        ranks = new_ranks

    sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_ranks[:num_keywords]]
