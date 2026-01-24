from typing import Callable

from insideLLMs.nlp.dependencies import ensure_sklearn, ensure_spacy
from insideLLMs.nlp.tokenization import simple_tokenize


# Backward compatibility aliases
check_sklearn = ensure_sklearn
check_spacy = ensure_spacy


# ===== Text Similarity =====


def cosine_similarity_texts(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts using TF-IDF."""
    ensure_sklearn()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    similarity = sklearn_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    score = float(similarity[0][0])
    # Numerical precision can push results slightly outside [0, 1].
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def jaccard_similarity(text1: str, text2: str, tokenizer: Callable = simple_tokenize) -> float:
    """Calculate Jaccard similarity between two texts."""
    tokens1 = set(tokenizer(text1))
    tokens2 = set(tokenizer(text2))

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    if not union:
        return 0.0

    return len(intersection) / len(union)


def word_overlap_similarity(text1: str, text2: str) -> float:
    """Calculate simple word overlap (Jaccard) similarity between two texts.

    This is a lightweight version of jaccard_similarity that uses lowercase
    whitespace splitting. Useful for quick similarity checks.

    Args:
        text1: First text to compare.
        text2: Second text to compare.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    if text1 == text2:
        return 1.0

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def levenshtein_distance(text1: str, text2: str) -> int:
    """Calculate Levenshtein (edit) distance between two texts."""
    if len(text1) < len(text2):
        return levenshtein_distance(text2, text1)

    if not text2:
        return len(text1)

    previous_row = range(len(text2) + 1)
    for i, c1 in enumerate(text1):
        current_row = [i + 1]
        for j, c2 in enumerate(text2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def semantic_similarity_word_embeddings(
    text1: str, text2: str, model_name: str = "en_core_web_sm"
) -> float:
    """Calculate semantic similarity between two texts using word embeddings."""
    nlp = ensure_spacy(model_name)
    doc1 = nlp(text1)
    doc2 = nlp(text2)

    if not doc1.vector.any() or not doc2.vector.any():
        return 0.0

    return doc1.similarity(doc2)


def jaro_similarity(s1: str, s2: str) -> float:
    """Calculate Jaro similarity between two strings."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    len_s1, len_s2 = len(s1), len(s2)
    match_distance = max(len_s1, len_s2) // 2 - 1
    match_distance = max(0, match_distance)

    s1_matches = [False] * len_s1
    s2_matches = [False] * len_s2

    matches = 0
    for i in range(len_s1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len_s2)

        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    t = 0
    k = 0
    for i in range(len_s1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            t += 1
        k += 1

    t /= 2
    return (matches / len_s1 + matches / len_s2 + (matches - t) / matches) / 3.0


def jaro_winkler_similarity(s1: str, s2: str, scaling: float = 0.1) -> float:
    """Calculate Jaro-Winkler similarity between two strings."""
    jaro_sim = jaro_similarity(s1, s2)
    prefix = 0
    max_prefix = 4
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            prefix += 1
        else:
            break
        if prefix == max_prefix:
            break

    return jaro_sim + prefix * scaling * (1 - jaro_sim)


def hamming_distance(s1: str, s2: str) -> int:
    """Calculate Hamming distance between two strings of equal length."""
    if len(s1) != len(s2):
        raise ValueError("Strings must be of equal length for Hamming distance.")
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))


def longest_common_subsequence(text1: str, text2: str) -> int:
    """Calculate length of the longest common subsequence between two strings."""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]
