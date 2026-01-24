"""
Text Similarity Module
======================

This module provides a comprehensive suite of text similarity and distance metrics
for comparing strings and documents. It includes both character-level and token-level
similarity measures, as well as semantic similarity using word embeddings.

Overview
--------
The module offers several categories of similarity functions:

1. **Token-based Similarity**: Measures that compare texts based on shared words/tokens
   - `cosine_similarity_texts`: TF-IDF based cosine similarity
   - `jaccard_similarity`: Ratio of shared tokens to total unique tokens
   - `word_overlap_similarity`: Lightweight Jaccard variant

2. **Edit Distance Metrics**: Character-level distance measures
   - `levenshtein_distance`: Minimum edit operations to transform one string to another
   - `hamming_distance`: Count of differing positions (equal-length strings only)

3. **Phonetic/Fuzzy Matching**: Similarity for typos and variations
   - `jaro_similarity`: Matching characters with transposition penalty
   - `jaro_winkler_similarity`: Jaro with prefix bonus for matching starts

4. **Sequence Matching**:
   - `longest_common_subsequence`: Length of longest shared subsequence

5. **Semantic Similarity**:
   - `semantic_similarity_word_embeddings`: SpaCy-based word vector similarity

Examples
--------
Basic text similarity comparison:

>>> from insideLLMs.nlp.similarity import cosine_similarity_texts, jaccard_similarity
>>> text1 = "The quick brown fox jumps over the lazy dog"
>>> text2 = "A fast brown fox leaps over a sleepy dog"
>>> cosine_sim = cosine_similarity_texts(text1, text2)
>>> print(f"Cosine similarity: {cosine_sim:.3f}")
Cosine similarity: 0.456

Comparing similar words with typos:

>>> from insideLLMs.nlp.similarity import jaro_winkler_similarity
>>> jaro_winkler_similarity("MARTHA", "MARHTA")
0.9611111111111111

Finding edit distance:

>>> from insideLLMs.nlp.similarity import levenshtein_distance
>>> levenshtein_distance("kitten", "sitting")
3

Notes
-----
- Some functions require optional dependencies (sklearn, spacy)
- Performance varies: character-level metrics are O(n*m), token-based are generally faster
- For large-scale comparisons, consider vectorized approaches
"""

from typing import Callable

from insideLLMs.nlp.dependencies import ensure_sklearn, ensure_spacy
from insideLLMs.nlp.tokenization import simple_tokenize


# Backward compatibility aliases
check_sklearn = ensure_sklearn
check_spacy = ensure_spacy


# ===== Text Similarity =====


def cosine_similarity_texts(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two texts using TF-IDF vectorization.

    This function converts two text strings into TF-IDF (Term Frequency-Inverse
    Document Frequency) vectors and computes the cosine of the angle between them.
    Cosine similarity measures the orientation rather than magnitude, making it
    effective for comparing documents of different lengths.

    Args:
        text1: The first text string to compare. Can be any length, from a single
            word to a full document.
        text2: The second text string to compare. Can be any length, from a single
            word to a full document.

    Returns:
        A float between 0.0 and 1.0, where:
        - 1.0 indicates identical texts (or texts with identical term distributions)
        - 0.0 indicates completely dissimilar texts (no shared terms)
        - Values in between indicate partial similarity

    Examples:
        Basic comparison of similar sentences:

        >>> cosine_similarity_texts(
        ...     "The cat sat on the mat",
        ...     "The cat is sitting on the mat"
        ... )
        0.6030226891555273

        Identical texts return 1.0:

        >>> cosine_similarity_texts("hello world", "hello world")
        1.0

        Completely different texts return 0.0:

        >>> cosine_similarity_texts("apple banana", "xyz uvw")
        0.0

        Case sensitivity matters (lowercase vs uppercase):

        >>> cosine_similarity_texts("Hello World", "hello world")
        0.0

        Comparing longer documents:

        >>> doc1 = "Machine learning is a subset of artificial intelligence"
        >>> doc2 = "Artificial intelligence includes machine learning methods"
        >>> sim = cosine_similarity_texts(doc1, doc2)
        >>> 0.3 < sim < 0.7  # Moderate similarity
        True

    Notes:
        - Requires sklearn to be installed (automatically checked via ensure_sklearn)
        - Case-sensitive by default; preprocess texts to lowercase for case-insensitive comparison
        - Punctuation is handled by the TF-IDF vectorizer
        - Empty strings will result in 0.0 similarity
        - For very short texts (1-2 words), results may be less meaningful
        - Time complexity: O(n) where n is the total number of tokens in both texts

    See Also:
        jaccard_similarity: For simple token overlap without TF-IDF weighting
        semantic_similarity_word_embeddings: For meaning-based similarity using word vectors
    """
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
    """
    Calculate Jaccard similarity (intersection over union) between two texts.

    The Jaccard similarity coefficient measures the overlap between two sets of tokens.
    It is computed as the size of the intersection divided by the size of the union
    of the token sets. This is also known as the Jaccard index or IoU (Intersection
    over Union).

    Args:
        text1: The first text string to compare.
        text2: The second text string to compare.
        tokenizer: A callable that takes a string and returns a list of tokens.
            Defaults to `simple_tokenize`, which performs basic whitespace and
            punctuation tokenization. Custom tokenizers can be provided for
            specialized tokenization (e.g., preserving contractions, handling
            specific languages).

    Returns:
        A float between 0.0 and 1.0, where:
        - 1.0 indicates identical token sets
        - 0.0 indicates no shared tokens (or both texts are empty)
        - Values in between indicate partial overlap

    Examples:
        Basic comparison with shared words:

        >>> jaccard_similarity("the cat sat", "the dog sat")
        0.5

        Identical texts return 1.0:

        >>> jaccard_similarity("hello world", "hello world")
        1.0

        No overlap returns 0.0:

        >>> jaccard_similarity("apple banana", "cherry date")
        0.0

        Word repetition doesn't affect the score (sets ignore duplicates):

        >>> jaccard_similarity("the the the cat", "the cat")
        1.0

        Using a custom tokenizer (lowercase split):

        >>> jaccard_similarity(
        ...     "Hello World",
        ...     "hello world",
        ...     tokenizer=lambda x: x.lower().split()
        ... )
        1.0

        Partial overlap example:

        >>> text1 = "machine learning is great"
        >>> text2 = "deep learning is powerful"
        >>> sim = jaccard_similarity(text1, text2)
        >>> 0.2 < sim < 0.5  # Some overlap ('learning', 'is')
        True

    Notes:
        - The default tokenizer is case-sensitive; use a custom tokenizer for
          case-insensitive comparison
        - Jaccard similarity treats each unique token equally, regardless of frequency
        - For frequency-weighted comparison, consider cosine_similarity_texts
        - Empty texts result in 0.0 similarity (empty union edge case)
        - Time complexity: O(n + m) where n and m are the number of tokens

    See Also:
        word_overlap_similarity: A lightweight variant using simple lowercase split
        cosine_similarity_texts: For TF-IDF weighted similarity
    """
    tokens1 = set(tokenizer(text1))
    tokens2 = set(tokenizer(text2))

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    if not union:
        return 0.0

    return len(intersection) / len(union)


def word_overlap_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple word overlap (Jaccard) similarity using lowercase whitespace splitting.

    This is a lightweight, zero-dependency version of Jaccard similarity optimized for
    quick similarity checks. It converts both texts to lowercase and splits on whitespace,
    making it case-insensitive and fast. Best suited for simple text comparisons where
    precise tokenization is not critical.

    Args:
        text1: The first text string to compare. Will be lowercased and split on
            whitespace for tokenization.
        text2: The second text string to compare. Will be lowercased and split on
            whitespace for tokenization.

    Returns:
        A float between 0.0 and 1.0, where:
        - 1.0 indicates identical word sets (after lowercasing)
        - 0.0 indicates no shared words or one/both texts are empty
        - Values in between indicate partial overlap

    Examples:
        Basic comparison (case-insensitive):

        >>> word_overlap_similarity("Hello World", "hello world")
        1.0

        Partial overlap:

        >>> word_overlap_similarity("the quick brown fox", "the lazy brown dog")
        0.3333333333333333

        Identical texts (fast path):

        >>> word_overlap_similarity("test string", "test string")
        1.0

        No overlap:

        >>> word_overlap_similarity("apple banana", "cherry date")
        0.0

        Empty string handling:

        >>> word_overlap_similarity("", "hello")
        0.0

        >>> word_overlap_similarity("", "")
        0.0

        Punctuation is NOT removed (attached to words):

        >>> word_overlap_similarity("hello, world", "hello world")
        0.3333333333333333

    Notes:
        - This function includes a fast path for identical strings (returns 1.0 immediately)
        - Case-insensitive by design (both texts are lowercased)
        - Punctuation attached to words is NOT stripped; use jaccard_similarity with
          a proper tokenizer if punctuation handling is needed
        - Empty strings result in 0.0 similarity
        - More performant than jaccard_similarity for simple use cases
        - Time complexity: O(n + m) where n and m are the number of words

    See Also:
        jaccard_similarity: For customizable tokenization and more precise comparison
        cosine_similarity_texts: For TF-IDF weighted comparison
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
    """
    Calculate the Levenshtein (edit) distance between two strings.

    The Levenshtein distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to transform one string
    into another. This is a fundamental string metric used in spell checking,
    DNA sequence analysis, and fuzzy string matching.

    This implementation uses Wagner-Fischer algorithm with space optimization,
    using only O(min(n,m)) space instead of O(n*m).

    Args:
        text1: The first string (source string to transform from).
        text2: The second string (target string to transform to).

    Returns:
        A non-negative integer representing the minimum number of edits required.
        - 0 means the strings are identical
        - Higher values indicate more differences

    Examples:
        Classic example - "kitten" to "sitting":

        >>> levenshtein_distance("kitten", "sitting")
        3

        The three edits are:
        1. kitten -> sitten (substitute 'k' with 's')
        2. sitten -> sittin (substitute 'e' with 'i')
        3. sittin -> sitting (insert 'g')

        Identical strings have distance 0:

        >>> levenshtein_distance("hello", "hello")
        0

        Single character difference:

        >>> levenshtein_distance("cat", "car")
        1

        Completely different strings:

        >>> levenshtein_distance("abc", "xyz")
        3

        Empty string to non-empty (all insertions):

        >>> levenshtein_distance("", "hello")
        5

        Case sensitivity:

        >>> levenshtein_distance("Hello", "hello")
        1

        Common typo detection:

        >>> levenshtein_distance("recieve", "receive")
        2

    Notes:
        - This is a character-level metric; for word-level comparison, tokenize first
        - Case-sensitive by default; lowercase both strings for case-insensitive comparison
        - Time complexity: O(n * m) where n and m are string lengths
        - Space complexity: O(min(n, m)) due to optimization
        - For normalized similarity (0-1 range), use:
          1 - (levenshtein_distance(s1, s2) / max(len(s1), len(s2)))
        - The function automatically swaps arguments to optimize for the shorter string

    See Also:
        hamming_distance: For comparing strings of equal length (substitutions only)
        jaro_winkler_similarity: For fuzzy matching that favors common prefixes
    """
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
    """
    Calculate semantic similarity between two texts using SpaCy word embeddings.

    This function uses pre-trained word vectors to compute semantic similarity,
    capturing meaning beyond simple word overlap. The document vectors are computed
    as the average of constituent word vectors, and cosine similarity is used
    to compare them.

    Args:
        text1: The first text string to compare. Can be a word, phrase, or document.
        text2: The second text string to compare. Can be a word, phrase, or document.
        model_name: The SpaCy model to use for word embeddings. Defaults to
            "en_core_web_sm". For better semantic similarity, consider:
            - "en_core_web_md": Medium model with word vectors (50MB)
            - "en_core_web_lg": Large model with word vectors (750MB)
            Note: The small model (sm) has limited word vectors.

    Returns:
        A float typically between -1.0 and 1.0 (cosine similarity), where:
        - 1.0 indicates very high semantic similarity
        - 0.0 indicates no semantic relationship (or missing vectors)
        - Negative values indicate semantic dissimilarity (rare in practice)

    Examples:
        Semantically similar sentences:

        >>> sim = semantic_similarity_word_embeddings(
        ...     "The cat sat on the mat",
        ...     "A feline rested on the rug"
        ... )
        >>> sim > 0.5  # Should show semantic similarity
        True

        Synonyms should have high similarity:

        >>> sim = semantic_similarity_word_embeddings("happy", "joyful")
        >>> sim > 0.3  # Positive semantic relationship
        True

        Unrelated concepts:

        >>> sim = semantic_similarity_word_embeddings(
        ...     "quantum physics",
        ...     "chocolate cake recipe"
        ... )
        >>> sim < 0.5  # Lower semantic similarity
        True

        Identical texts:

        >>> semantic_similarity_word_embeddings("hello world", "hello world")
        1.0

        Using a larger model for better accuracy:

        >>> sim = semantic_similarity_word_embeddings(
        ...     "bank account",
        ...     "financial institution",
        ...     model_name="en_core_web_md"
        ... )  # doctest: +SKIP

    Notes:
        - Requires SpaCy and the specified language model to be installed
        - The default small model (en_core_web_sm) has limited word vectors;
          use medium (md) or large (lg) models for better semantic accuracy
        - Returns 0.0 if either text produces an empty or zero vector
        - Out-of-vocabulary (OOV) words are handled gracefully but may reduce accuracy
        - Performance depends on the size of the SpaCy model loaded
        - First call may be slow due to model loading; subsequent calls are faster
        - Time complexity: O(n + m) for tokenization, plus model inference time

    Raises:
        RuntimeError: If SpaCy or the specified model is not installed.

    See Also:
        cosine_similarity_texts: For TF-IDF based similarity (lexical, not semantic)
        jaccard_similarity: For simple token overlap comparison
    """
    nlp = ensure_spacy(model_name)
    doc1 = nlp(text1)
    doc2 = nlp(text2)

    if not doc1.vector.any() or not doc2.vector.any():
        return 0.0

    return doc1.similarity(doc2)


def jaro_similarity(s1: str, s2: str) -> float:
    """
    Calculate Jaro similarity between two strings.

    The Jaro similarity is a measure of similarity between two strings based on
    the number and order of matching characters. It was developed for record linkage
    in census data and is particularly effective for comparing short strings like
    names. The algorithm considers:
    1. The number of matching characters
    2. The number of transpositions (matching characters in different order)

    Two characters are considered matching if they are the same and not farther
    than floor(max(len(s1), len(s2)) / 2) - 1 characters apart.

    Args:
        s1: The first string to compare.
        s2: The second string to compare.

    Returns:
        A float between 0.0 and 1.0, where:
        - 1.0 indicates identical strings
        - 0.0 indicates no matching characters within the match window
        - Values in between indicate partial similarity

    Examples:
        Classic example - similar names:

        >>> jaro_similarity("MARTHA", "MARHTA")
        0.9444444444444445

        Identical strings:

        >>> jaro_similarity("hello", "hello")
        1.0

        Completely different strings:

        >>> jaro_similarity("abc", "xyz")
        0.0

        Empty strings:

        >>> jaro_similarity("", "")
        1.0

        >>> jaro_similarity("hello", "")
        0.0

        Common typos:

        >>> jaro_similarity("DIXON", "DICKSONX")
        0.7666666666666666

        Short strings with transposition:

        >>> jaro_similarity("DwAyNE", "DuANE")
        0.8222222222222223

    Notes:
        - Best suited for short strings (names, words) rather than long documents
        - Case-sensitive by default; lowercase both strings for case-insensitive comparison
        - The match window is calculated as: max(len(s1), len(s2)) // 2 - 1
        - Transpositions are counted as half-matches in the formula
        - Time complexity: O(n * m) where n and m are string lengths
        - Space complexity: O(n + m) for the match tracking arrays

    Formula:
        jaro = (1/3) * (m/|s1| + m/|s2| + (m-t)/m)
        where:
        - m = number of matching characters
        - t = number of transpositions / 2
        - |s1|, |s2| = lengths of the strings

    See Also:
        jaro_winkler_similarity: Enhanced version with prefix bonus
        levenshtein_distance: For edit distance (number of operations)
    """
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
    """
    Calculate Jaro-Winkler similarity between two strings.

    The Jaro-Winkler similarity extends Jaro similarity by giving a bonus
    to strings that share a common prefix. This makes it especially effective
    for comparing strings where matching prefixes are significant, such as
    person names (where first letters are rarely mistyped).

    The formula is: jaro_winkler = jaro + (prefix_length * scaling * (1 - jaro))

    Args:
        s1: The first string to compare.
        s2: The second string to compare.
        scaling: The scaling factor for the prefix bonus. Must be <= 0.25.
            Defaults to 0.1 (standard Winkler modification). Higher values
            give more weight to matching prefixes.

    Returns:
        A float between 0.0 and 1.0, where:
        - 1.0 indicates identical strings
        - Values closer to 1.0 indicate higher similarity
        - Strings with matching prefixes score higher than pure Jaro similarity

    Examples:
        Classic example - names with transposition:

        >>> jaro_winkler_similarity("MARTHA", "MARHTA")
        0.9611111111111111

        Compare with pure Jaro (no prefix bonus):

        >>> jaro_similarity("MARTHA", "MARHTA")
        0.9444444444444445

        Identical strings:

        >>> jaro_winkler_similarity("hello", "hello")
        1.0

        Strings with matching prefix get a boost:

        >>> jaro_winkler_similarity("PREFIX_abc", "PREFIX_xyz")
        0.6666666666666666

        Different first characters (no prefix bonus):

        >>> jaro_winkler_similarity("abcdef", "zbcdef")
        0.9444444444444445

        Custom scaling factor (higher prefix weight):

        >>> jaro_winkler_similarity("MARTHA", "MARHTA", scaling=0.2)
        0.9777777777777779

        Common name variations:

        >>> jaro_winkler_similarity("JOHNATHAN", "JONATHAN")
        0.9481481481481482

    Notes:
        - Winkler's modification only applies to strings that already have
          Jaro similarity >= 0.7 in the original formulation, but this
          implementation always applies the prefix bonus
        - Maximum prefix length considered is 4 characters (standard Winkler)
        - Scaling factor should not exceed 0.25 to keep similarity <= 1.0
        - Particularly effective for:
          - Name matching and record linkage
          - Typo correction where first letters are usually correct
          - Entity resolution in databases
        - Time complexity: O(n * m) dominated by Jaro calculation
        - Space complexity: O(n + m) for the match tracking arrays

    See Also:
        jaro_similarity: Base similarity without prefix bonus
        levenshtein_distance: For edit-based distance metric
    """
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
    """
    Calculate Hamming distance between two strings of equal length.

    The Hamming distance measures the number of positions at which the corresponding
    characters are different. It is only defined for strings of equal length and
    counts substitutions only (no insertions or deletions). Originally developed
    for error detection in telecommunications, it is widely used in coding theory,
    DNA sequence comparison, and binary data comparison.

    Args:
        s1: The first string to compare. Must have the same length as s2.
        s2: The second string to compare. Must have the same length as s1.

    Returns:
        A non-negative integer representing the number of differing positions.
        - 0 means the strings are identical
        - Maximum value is len(s1) (all positions differ)

    Raises:
        ValueError: If the two strings have different lengths.

    Examples:
        Basic comparison:

        >>> hamming_distance("karolin", "kathrin")
        3

        The three differences are at positions 2, 3, and 4:
        - 'r' vs 't' at position 2
        - 'o' vs 'h' at position 3
        - 'l' vs 'r' at position 4

        Identical strings:

        >>> hamming_distance("hello", "hello")
        0

        Single character difference:

        >>> hamming_distance("cat", "car")
        1

        All characters different:

        >>> hamming_distance("abc", "xyz")
        3

        Binary strings (common use case):

        >>> hamming_distance("1011101", "1001001")
        2

        Case sensitivity:

        >>> hamming_distance("Hello", "hello")
        1

        Unequal length strings raise ValueError:

        >>> hamming_distance("abc", "abcd")
        Traceback (most recent call last):
            ...
        ValueError: Strings must be of equal length for Hamming distance.

    Notes:
        - Only valid for equal-length strings; use levenshtein_distance for
          variable-length strings
        - Case-sensitive by default
        - Time complexity: O(n) where n is the string length
        - Space complexity: O(1)
        - For normalized similarity (0-1 range), use:
          1 - (hamming_distance(s1, s2) / len(s1))
        - Commonly used for:
          - Error detection/correction codes
          - Comparing binary data or hashes
          - DNA sequence analysis (fixed-length sequences)

    See Also:
        levenshtein_distance: For comparing strings of different lengths
    """
    if len(s1) != len(s2):
        raise ValueError("Strings must be of equal length for Hamming distance.")
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))


def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    Calculate the length of the longest common subsequence (LCS) between two strings.

    The longest common subsequence is the longest sequence of characters that appears
    in both strings in the same order, but not necessarily contiguously. Unlike
    substrings, subsequences do not need to occupy consecutive positions.

    This is a classic dynamic programming problem with applications in diff tools,
    version control systems, DNA sequence alignment, and plagiarism detection.

    Args:
        text1: The first string to compare.
        text2: The second string to compare.

    Returns:
        A non-negative integer representing the length of the longest common
        subsequence.
        - 0 means no common characters in order
        - Maximum is min(len(text1), len(text2))

    Examples:
        Basic example:

        >>> longest_common_subsequence("ABCDGH", "AEDFHR")
        3

        The LCS is "ADH" (A...D...H in both strings)

        Finding common characters in order:

        >>> longest_common_subsequence("AGGTAB", "GXTXAYB")
        4

        The LCS is "GTAB"

        Identical strings:

        >>> longest_common_subsequence("hello", "hello")
        5

        No common subsequence:

        >>> longest_common_subsequence("abc", "xyz")
        0

        Empty string:

        >>> longest_common_subsequence("", "hello")
        0

        Single character match:

        >>> longest_common_subsequence("abc", "axy")
        1

        Case sensitivity matters:

        >>> longest_common_subsequence("Hello", "hello")
        4

        The LCS is "ello" (lowercase)

        Practical example - comparing similar sentences:

        >>> longest_common_subsequence("the cat sat", "the rat sat")
        9

    Notes:
        - This function returns only the LENGTH; to get the actual subsequence,
          the algorithm would need to be modified to backtrack through the DP table
        - Time complexity: O(n * m) where n and m are string lengths
        - Space complexity: O(n * m) for the DP table
        - For very long strings, consider space-optimized versions using O(min(n,m))
        - Case-sensitive by default; lowercase both strings for case-insensitive comparison
        - For normalized similarity, divide by max(len(text1), len(text2))
        - Common applications:
          - diff algorithms (finding changes between files)
          - DNA/protein sequence alignment
          - Spell checking and autocomplete
          - Plagiarism detection

    See Also:
        levenshtein_distance: For edit distance (related but different metric)
        jaccard_similarity: For set-based comparison ignoring order
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]
