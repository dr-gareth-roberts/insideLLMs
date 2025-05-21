from typing import Callable, List

# Optional dependencies
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity # Renamed
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = False 
    SPACY_MODEL = None
except ImportError:
    SPACY_AVAILABLE = False
    SPACY_MODEL = None

# ===== Dependency Management =====

def check_sklearn():
    """Check if scikit-learn is available."""
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is not installed. Please install it with: pip install scikit-learn"
        )

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
        except NameError: # Should be caught by SPACY_AVAILABLE check
            raise ImportError("spaCy is not installed. Please install it with: pip install spacy")
    return SPACY_MODEL

# ===== Helper functions (copied from tokenization.py for now) =====
def simple_tokenize(text: str) -> List[str]:
    """Simple word tokenization by splitting on whitespace.

    Args:
        text: Input text to tokenize

    Returns:
        List of tokens
    """
    return text.split()


# ===== Text Similarity =====

def cosine_similarity_texts(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts using TF-IDF.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Cosine similarity score
    """
    check_sklearn()

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    similarity = sklearn_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return float(similarity[0][0])

def jaccard_similarity(text1: str, text2: str, tokenizer: Callable = simple_tokenize) -> float:
    """Calculate Jaccard similarity between two texts.

    Args:
        text1: First text
        text2: Second text
        tokenizer: Function to tokenize text

    Returns:
        Jaccard similarity score
    """
    tokens1 = set(tokenizer(text1))
    tokens2 = set(tokenizer(text2))

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    if not union:
        return 0.0

    return len(intersection) / len(union)

def levenshtein_distance(text1: str, text2: str) -> int:
    """Calculate Levenshtein (edit) distance between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Levenshtein distance
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

def semantic_similarity_word_embeddings(text1: str,
                                       text2: str,
                                       model_name: str = "en_core_web_sm") -> float:
    """Calculate semantic similarity between two texts using word embeddings.

    Args:
        text1: First text
        text2: Second text
        model_name: Name of the spaCy model to use

    Returns:
        Semantic similarity score
    """
    nlp = check_spacy(model_name)
    doc1 = nlp(text1)
    doc2 = nlp(text2)

    if not doc1.vector.any() or not doc2.vector.any(): # Check if vectors are non-zero
        # Fallback or warning if vectors are zero (e.g. out-of-vocabulary words for all tokens)
        # This can happen if the text is empty or all words are OOV for the spaCy model.
        # Depending on desired behavior, could return 0.0 or raise a warning.
        return 0.0


    return doc1.similarity(doc2)

def jaro_similarity(s1: str, s2: str) -> float:
    """Calculate Jaro similarity between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Jaro similarity score (0-1)
    """
    # If both strings are empty, return 1.0
    if not s1 and not s2:
        return 1.0

    # If one string is empty, return 0.0
    if not s1 or not s2:
        return 0.0

    # If strings are identical, return 1.0
    if s1 == s2:
        return 1.0

    len_s1, len_s2 = len(s1), len(s2)

    # Maximum distance for matching characters
    match_distance = max(len_s1, len_s2) // 2 - 1
    match_distance = max(0, match_distance)  # Ensure non-negative

    # Arrays to track matched characters
    s1_matches = [False] * len_s1
    s2_matches = [False] * len_s2

    # Count matching characters
    matches = 0
    for i in range(len_s1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len_s2)

        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    # If no matches, return 0.0
    if matches == 0:
        return 0.0

    # Count transpositions
    transpositions = 0
    k = 0
    for i in range(len_s1):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

    transpositions = transpositions // 2

    # Calculate Jaro similarity
    return (matches / len_s1 + matches / len_s2 + (matches - transpositions) / matches) / 3.0

def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """Calculate Jaro-Winkler similarity between two strings.

    Args:
        s1: First string
        s2: Second string
        p: Scaling factor for prefix matches (default: 0.1)

    Returns:
        Jaro-Winkler similarity score (0-1)
    """
    # Calculate Jaro similarity
    jaro_sim = jaro_similarity(s1, s2)

    # Calculate length of common prefix (up to 4 characters)
    prefix_len = 0
    max_prefix = min(4, min(len(s1), len(s2)))
    for i in range(max_prefix):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    # Calculate Jaro-Winkler similarity
    return jaro_sim + (prefix_len * p * (1 - jaro_sim))

def hamming_distance(s1: str, s2: str) -> int:
    """Calculate Hamming distance between two strings of equal length.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Hamming distance
    """
    if len(s1) != len(s2):
        raise ValueError("Strings must be of equal length")

    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def longest_common_subsequence(s1: str, s2: str) -> str:
    """Find the longest common subsequence of two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Longest common subsequence
    """
    len_s1, len_s2 = len(s1), len(s2)

    # Create a table to store lengths of LCS for all subproblems
    dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]

    # Fill the dp table
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # Reconstruct the LCS
    lcs = []
    i, j = len_s1, len_s2
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            lcs.append(s1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    return ''.join(reversed(lcs))

# Re-initialize SKLEARN_AVAILABLE and SPACY_AVAILABLE
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
