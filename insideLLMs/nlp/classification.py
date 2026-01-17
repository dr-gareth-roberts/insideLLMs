from typing import List

from insideLLMs.nlp.dependencies import ensure_nltk, ensure_sklearn


# ===== Dependency Management =====

def check_nltk():
    """Ensure NLTK and the VADER lexicon are available."""
    ensure_nltk(("sentiment/vader_lexicon",))


def check_sklearn():
    """Ensure scikit-learn is available."""
    ensure_sklearn()


# ===== Basic Text Classification =====

def naive_bayes_classify(train_texts: List[str], train_labels: List[str], test_texts: List[str]) -> List[str]:
    """Classify texts using Naive Bayes."""
    check_sklearn()
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB()),
    ])

    pipeline.fit(train_texts, train_labels)
    return pipeline.predict(test_texts).tolist()


def svm_classify(train_texts: List[str], train_labels: List[str], test_texts: List[str]) -> List[str]:
    """Classify texts using Support Vector Machine."""
    check_sklearn()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", LinearSVC()),
    ])

    pipeline.fit(train_texts, train_labels)
    return pipeline.predict(test_texts).tolist()


def sentiment_analysis_basic(text: str) -> str:
    """Perform basic sentiment analysis using a lexicon-based approach."""
    check_nltk()
    from nltk.sentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)

    if sentiment_scores["compound"] >= 0.05:
        return "positive"
    if sentiment_scores["compound"] <= -0.05:
        return "negative"
    return "neutral"
