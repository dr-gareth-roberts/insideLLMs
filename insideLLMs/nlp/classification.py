from insideLLMs.nlp.dependencies import ensure_nltk, ensure_sklearn


def _ensure_vader():
    """Ensure NLTK and the VADER lexicon are available."""
    ensure_nltk(("sentiment/vader_lexicon",))


# Backward compatibility aliases
check_nltk = _ensure_vader
check_sklearn = ensure_sklearn


# ===== Basic Text Classification =====


def naive_bayes_classify(
    train_texts: list[str], train_labels: list[str], test_texts: list[str]
) -> list[str]:
    """Classify texts using Naive Bayes."""
    ensure_sklearn()
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [
            ("vectorizer", CountVectorizer()),
            ("classifier", MultinomialNB()),
        ]
    )

    pipeline.fit(train_texts, train_labels)
    return pipeline.predict(test_texts).tolist()


def svm_classify(
    train_texts: list[str], train_labels: list[str], test_texts: list[str]
) -> list[str]:
    """Classify texts using Support Vector Machine."""
    ensure_sklearn()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    pipeline = Pipeline(
        [
            ("vectorizer", TfidfVectorizer()),
            ("classifier", LinearSVC()),
        ]
    )

    pipeline.fit(train_texts, train_labels)
    return pipeline.predict(test_texts).tolist()


def sentiment_analysis_basic(text: str) -> str:
    """Perform basic sentiment analysis using a lexicon-based approach."""
    _ensure_vader()
    from nltk.sentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)

    if sentiment_scores["compound"] >= 0.05:
        return "positive"
    if sentiment_scores["compound"] <= -0.05:
        return "negative"
    return "neutral"
