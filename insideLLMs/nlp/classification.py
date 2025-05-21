from typing import List

# Optional dependencies
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ===== Dependency Management =====

def check_nltk():
    """Check if NLTK is available and download required resources if needed."""
    if not NLTK_AVAILABLE:
        raise ImportError(
            "NLTK is not installed. Please install it with: pip install nltk"
        )
    # Download vader_lexicon for sentiment_analysis_basic
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

def check_sklearn():
    """Check if scikit-learn is available."""
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is not installed. Please install it with: pip install scikit-learn"
        )

# ===== Basic Text Classification =====

def naive_bayes_classify(train_texts: List[str],
                        train_labels: List[str],
                        test_texts: List[str]) -> List[str]:
    """Classify texts using Naive Bayes.

    Args:
        train_texts: List of training texts
        train_labels: List of training labels
        test_texts: List of texts to classify

    Returns:
        List of predicted labels for test_texts
    """
    check_sklearn()
    
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])

    pipeline.fit(train_texts, train_labels)
    return pipeline.predict(test_texts).tolist()

def svm_classify(train_texts: List[str],
                train_labels: List[str],
                test_texts: List[str]) -> List[str]:
    """Classify texts using Support Vector Machine.

    Args:
        train_texts: List of training texts
        train_labels: List of training labels
        test_texts: List of texts to classify

    Returns:
        List of predicted labels for test_texts
    """
    check_sklearn()

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', LinearSVC())
    ])

    pipeline.fit(train_texts, train_labels)
    return pipeline.predict(test_texts).tolist()

def sentiment_analysis_basic(text: str) -> str:
    """Perform basic sentiment analysis using a lexicon-based approach.

    Args:
        text: Input text

    Returns:
        Sentiment label ('positive', 'negative', or 'neutral')
    """
    check_nltk()
    
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)

    if sentiment_scores['compound'] >= 0.05:
        return 'positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

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
