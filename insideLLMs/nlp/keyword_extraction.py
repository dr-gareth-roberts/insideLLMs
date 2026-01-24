from collections import defaultdict

from insideLLMs.nlp.dependencies import ensure_nltk, ensure_sklearn
from insideLLMs.nlp.tokenization import (
    nltk_tokenize,
    remove_stopwords,
    segment_sentences,
)


def _ensure_nltk_keyword():
    """Ensure NLTK and required resources are available."""
    ensure_nltk(("tokenizers/punkt", "corpora/stopwords"))


# Backward compatibility aliases
check_nltk = _ensure_nltk_keyword
check_sklearn = ensure_sklearn
segment_sentences_for_keyword = segment_sentences
nltk_tokenize_for_keyword = nltk_tokenize
remove_stopwords_for_keyword = remove_stopwords


# ===== Keyword Extraction =====


def extract_keywords_tfidf(text: str, num_keywords: int = 5) -> list[str]:
    """Extract keywords from text using TF-IDF."""
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
    """Extract keywords from text using a simple TextRank implementation."""
    _ensure_nltk_keyword()

    tokens = nltk_tokenize_for_keyword(text.lower())
    tokens = remove_stopwords_for_keyword(tokens)
    if not tokens:
        return []

    graph = defaultdict(lambda: defaultdict(int))
    # Populates weighted co‑occurrence graph from token window
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i : i + window_size]
        for j in range(len(window)):
            # Populates weighted co‑occurrence graph from token window
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
