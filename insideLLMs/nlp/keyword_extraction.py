from collections import defaultdict
from typing import List

# Optional dependencies
try:
    import nltk
    from nltk.tokenize import word_tokenize as nltk_word_tokenize_for_keyword, sent_tokenize as nltk_sent_tokenize_for_keyword
    from nltk.corpus import stopwords as nltk_stopwords_for_keyword
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ===== Dependency Management =====

def check_nltk_for_keyword():
    """Check if NLTK is available and download required resources if needed."""
    if not NLTK_AVAILABLE:
        raise ImportError("NLTK is not installed. Please install it with: pip install nltk")
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def check_sklearn_for_keyword():
    """Check if scikit-learn is available."""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is not installed. Please install it with: pip install scikit-learn")

# ===== Helper functions (copied/adapted) =====

def segment_sentences_for_keyword(text: str) -> List[str]:
    """Split text into sentences using NLTK."""
    check_nltk_for_keyword()
    return nltk_sent_tokenize_for_keyword(text)

def nltk_tokenize_for_keyword(text: str) -> List[str]:
    """Tokenize text using NLTK's word_tokenize."""
    check_nltk_for_keyword()
    return nltk_word_tokenize_for_keyword(text)

def remove_stopwords_for_keyword(tokens: List[str], language: str = 'english') -> List[str]:
    """Remove stopwords from a list of tokens."""
    check_nltk_for_keyword()
    stop_words = set(nltk_stopwords_for_keyword.words(language))
    return [token for token in tokens if token.lower() not in stop_words]


# ===== Keyword Extraction =====

def extract_keywords_tfidf(text: str, num_keywords: int = 5) -> List[str]:
    """Extract keywords from text using TF-IDF.

    Args:
        text: Input text
        num_keywords: Number of keywords to extract

    Returns:
        List of extracted keywords
    """
    check_sklearn_for_keyword()

    # Split text into sentences
    sentences = segment_sentences_for_keyword(text)
    if not sentences:
        return []

    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english') # Using sklearn's built-in stop words
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError: # Can happen if all words are stop words
        return []


    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    if not feature_names.any():
        return []

    # Calculate average TF-IDF score for each word by summing scores across sentences
    # This is a simplified approach. A more common way is to average document vectors or sum term scores.
    word_scores = defaultdict(float)
    # Sum TF-IDF scores for each term across all sentences (documents in this context)
    # This treats each sentence as a document and aggregates scores for terms.
    aggregated_scores = tfidf_matrix.sum(axis=0).A1 # Sum scores for each term, .A1 converts to 1D array
    
    for i, term_score in enumerate(aggregated_scores):
        word_scores[feature_names[i]] = term_score


    # Sort words by score and return top keywords
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_words[:num_keywords]]

def extract_keywords_textrank(text: str, num_keywords: int = 5, window_size: int = 4) -> List[str]:
    """Extract keywords from text using TextRank algorithm.

    Args:
        text: Input text
        num_keywords: Number of keywords to extract
        window_size: Window size for co-occurrence

    Returns:
        List of extracted keywords
    """
    check_nltk_for_keyword()

    # Tokenize and remove stopwords
    tokens = nltk_tokenize_for_keyword(text.lower())
    tokens = remove_stopwords_for_keyword(tokens)
    if not tokens:
        return []

    # Build co-occurrence graph
    graph = defaultdict(lambda: defaultdict(int))
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i+window_size]
        for j in range(len(window)):
            for k in range(j+1, len(window)):
                # Filter out self-loops if they occur due to tokenization issues or very small windows
                if window[j] != window[k]:
                    graph[window[j]][window[k]] += 1
                    graph[window[k]][window[j]] += 1
    
    if not graph:
        return []

    # Run TextRank algorithm
    ranks = {node: 1.0 for node in graph}
    num_iterations = 10 # Standard number of iterations
    damping = 0.85      # Standard damping factor

    for _ in range(num_iterations):
        new_ranks = ranks.copy() # Calculate new ranks based on current ranks
        for node in graph:
            rank_sum = 0
            for neighbor in graph[node]:
                # Sum of weights of edges from neighbor
                neighbor_links_sum = sum(graph[neighbor].values())
                if neighbor_links_sum > 0:
                    # Add rank from neighbor, weighted by edge weight / total weight from neighbor
                    rank_sum += (graph[node][neighbor] / neighbor_links_sum) * ranks[neighbor]
            
            new_ranks[node] = (1 - damping) + damping * rank_sum
        ranks = new_ranks


    # Sort words by rank and return top keywords
    sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    return [word for word, rank in sorted_ranks[:num_keywords]]

# Re-initialize NLTK_AVAILABLE and SKLEARN_AVAILABLE
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
