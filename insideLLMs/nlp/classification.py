"""
Text Classification Module
===========================

This module provides text classification utilities including machine learning-based
classifiers and sentiment analysis tools. It supports common classification tasks
such as spam detection, topic categorization, and sentiment analysis.

Key Features
------------
- Naive Bayes classification with bag-of-words features
- SVM classification with TF-IDF features
- Lexicon-based sentiment analysis using VADER

Dependencies
------------
- scikit-learn: For Naive Bayes and SVM classifiers
- NLTK: For VADER sentiment analysis

Examples
--------
Basic spam classification with Naive Bayes:

>>> train_texts = [
...     "Get rich quick! Buy now!",
...     "Limited time offer! Click here!",
...     "Meeting scheduled for tomorrow",
...     "Please review the attached document"
... ]
>>> train_labels = ["spam", "spam", "ham", "ham"]
>>> test_texts = ["Win a free prize today!", "Lunch meeting at noon"]
>>> predictions = naive_bayes_classify(train_texts, train_labels, test_texts)
>>> predictions
['spam', 'ham']

Sentiment analysis:

>>> sentiment_analysis_basic("I love this product!")
'positive'
>>> sentiment_analysis_basic("This is terrible.")
'negative'
>>> sentiment_analysis_basic("The meeting is at 3pm.")
'neutral'
"""

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
    """
    Classify texts using Multinomial Naive Bayes with bag-of-words features.

    This function implements a text classification pipeline that uses
    CountVectorizer for feature extraction and Multinomial Naive Bayes
    for classification. It is particularly effective for text classification
    tasks where word frequency is a good indicator of class membership.

    Args:
        train_texts: A list of training text samples. Each string represents
            a document to be used for training the classifier.
        train_labels: A list of labels corresponding to each training text.
            Must have the same length as train_texts.
        test_texts: A list of text samples to classify. The trained model
            will predict labels for these texts.

    Returns:
        A list of predicted labels for each text in test_texts. The labels
        will be from the same set as train_labels.

    Raises:
        ValueError: If train_texts and train_labels have different lengths.
        ValueError: If train_texts or test_texts is empty.

    Examples:
        Spam detection:

        >>> train_texts = [
        ...     "Buy cheap products now! Limited offer!",
        ...     "Congratulations! You won a prize!",
        ...     "Can we schedule a meeting for Monday?",
        ...     "Please find the quarterly report attached"
        ... ]
        >>> train_labels = ["spam", "spam", "ham", "ham"]
        >>> test_texts = ["Win free money today!", "Project deadline reminder"]
        >>> naive_bayes_classify(train_texts, train_labels, test_texts)
        ['spam', 'ham']

        Topic classification:

        >>> sports_docs = ["The team won the championship game",
        ...                "The player scored a touchdown"]
        >>> tech_docs = ["The new CPU has faster processing speed",
        ...              "The software update fixes bugs"]
        >>> train_texts = sports_docs + tech_docs
        >>> train_labels = ["sports", "sports", "tech", "tech"]
        >>> test_texts = ["The quarterback threw a perfect pass"]
        >>> naive_bayes_classify(train_texts, train_labels, test_texts)
        ['sports']

        Multi-class classification with more categories:

        >>> train_texts = [
        ...     "The stock market crashed today",
        ...     "Investors are worried about inflation",
        ...     "New restaurant opened downtown",
        ...     "Best pizza I ever had",
        ...     "Rain expected throughout the week",
        ...     "Temperatures will drop below freezing"
        ... ]
        >>> train_labels = ["finance", "finance", "food", "food", "weather", "weather"]
        >>> test_texts = ["Bond yields are rising", "Sunny skies ahead"]
        >>> naive_bayes_classify(train_texts, train_labels, test_texts)
        ['finance', 'weather']

        Binary sentiment classification:

        >>> positive = ["Great product, love it!", "Excellent quality"]
        >>> negative = ["Terrible experience", "Would not recommend"]
        >>> train_texts = positive + negative
        >>> train_labels = ["positive", "positive", "negative", "negative"]
        >>> test_texts = ["Amazing service!", "Very disappointed"]
        >>> naive_bayes_classify(train_texts, train_labels, test_texts)
        ['positive', 'negative']

    Notes:
        - Uses CountVectorizer for bag-of-words feature extraction
        - Multinomial Naive Bayes assumes features are word counts
        - Works well with small to medium training sets
        - Fast training and prediction times
        - May underperform on datasets with complex linguistic patterns

    See Also:
        svm_classify: SVM-based classifier with TF-IDF features
        sentiment_analysis_basic: Lexicon-based sentiment analysis
    """
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
    """
    Classify texts using Linear Support Vector Machine with TF-IDF features.

    This function implements a text classification pipeline that uses
    TfidfVectorizer for feature extraction and Linear SVM for classification.
    TF-IDF (Term Frequency-Inverse Document Frequency) weighting helps
    emphasize distinctive words while downweighting common terms.

    Args:
        train_texts: A list of training text samples. Each string represents
            a document to be used for training the classifier.
        train_labels: A list of labels corresponding to each training text.
            Must have the same length as train_texts.
        test_texts: A list of text samples to classify. The trained model
            will predict labels for these texts.

    Returns:
        A list of predicted labels for each text in test_texts. The labels
        will be from the same set as train_labels.

    Raises:
        ValueError: If train_texts and train_labels have different lengths.
        ValueError: If train_texts or test_texts is empty.

    Examples:
        Spam detection with SVM:

        >>> train_texts = [
        ...     "URGENT: You have won a lottery! Claim now!",
        ...     "Click here for exclusive discount offer!",
        ...     "The project meeting is rescheduled to Friday",
        ...     "Please review the attached expense report"
        ... ]
        >>> train_labels = ["spam", "spam", "ham", "ham"]
        >>> test_texts = ["Claim your free gift card!", "Budget review tomorrow"]
        >>> svm_classify(train_texts, train_labels, test_texts)
        ['spam', 'ham']

        News article categorization:

        >>> politics = ["The senator proposed a new bill in congress",
        ...             "Election results show a close race"]
        >>> science = ["Researchers discovered a new species of bacteria",
        ...            "The experiment confirmed the hypothesis"]
        >>> train_texts = politics + science
        >>> train_labels = ["politics", "politics", "science", "science"]
        >>> test_texts = ["New legislation passed by the committee"]
        >>> svm_classify(train_texts, train_labels, test_texts)
        ['politics']

        Customer feedback classification:

        >>> train_texts = [
        ...     "The product broke after one day",
        ...     "Shipping took way too long",
        ...     "Excellent customer support",
        ...     "Fast delivery and great packaging",
        ...     "Average product, nothing special",
        ...     "It works as expected, okay quality"
        ... ]
        >>> train_labels = ["complaint", "complaint", "praise", "praise",
        ...                 "neutral", "neutral"]
        >>> test_texts = ["My order arrived damaged", "Love this company!"]
        >>> svm_classify(train_texts, train_labels, test_texts)
        ['complaint', 'praise']

        Language detection:

        >>> english = ["The weather is beautiful today",
        ...            "I went to the store yesterday"]
        >>> spanish = ["El clima es hermoso hoy",
        ...            "Fui a la tienda ayer"]
        >>> french = ["Le temps est magnifique aujourd'hui",
        ...           "Je suis alle au magasin hier"]
        >>> train_texts = english + spanish + french
        >>> train_labels = ["en", "en", "es", "es", "fr", "fr"]
        >>> test_texts = ["The book is on the table", "El libro esta en la mesa"]
        >>> svm_classify(train_texts, train_labels, test_texts)
        ['en', 'es']

    Notes:
        - Uses TfidfVectorizer for TF-IDF weighted feature extraction
        - Linear SVM is effective for high-dimensional sparse data like text
        - Generally achieves higher accuracy than Naive Bayes on complex texts
        - Slightly slower training time compared to Naive Bayes
        - Works well even with imbalanced class distributions

    See Also:
        naive_bayes_classify: Faster Naive Bayes classifier
        sentiment_analysis_basic: Lexicon-based sentiment analysis
    """
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
    """
    Perform sentiment analysis using NLTK's VADER lexicon-based approach.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically
    attuned to sentiments expressed in social media. It uses a lexicon of
    sentiment-related words and applies rules to handle intensity modifiers,
    negations, and punctuation.

    Args:
        text: The input text to analyze for sentiment. Can be a single
            sentence, paragraph, or longer document. Works best with
            informal text such as social media posts, reviews, or comments.

    Returns:
        A string indicating the sentiment category:
        - "positive": Compound score >= 0.05
        - "negative": Compound score <= -0.05
        - "neutral": Compound score between -0.05 and 0.05

    Examples:
        Clearly positive sentiment:

        >>> sentiment_analysis_basic("I love this product! It's amazing!")
        'positive'
        >>> sentiment_analysis_basic("Best experience ever, highly recommend!")
        'positive'
        >>> sentiment_analysis_basic("Fantastic service, exceeded expectations!")
        'positive'

        Clearly negative sentiment:

        >>> sentiment_analysis_basic("This is terrible, worst purchase ever.")
        'negative'
        >>> sentiment_analysis_basic("I hate waiting in long lines.")
        'negative'
        >>> sentiment_analysis_basic("Disappointed with the poor quality.")
        'negative'

        Neutral or factual statements:

        >>> sentiment_analysis_basic("The meeting is scheduled for 3pm.")
        'neutral'
        >>> sentiment_analysis_basic("The package arrived yesterday.")
        'neutral'
        >>> sentiment_analysis_basic("The report contains five sections.")
        'neutral'

        Social media style text with emoticons and emphasis:

        >>> sentiment_analysis_basic("SO EXCITED for the concert!!! :)")
        'positive'
        >>> sentiment_analysis_basic("ugh... worst day ever :(")
        'negative'
        >>> sentiment_analysis_basic("Just finished lunch.")
        'neutral'

    Notes:
        - VADER handles emoticons, slang, and social media conventions
        - Capitalization increases sentiment intensity (e.g., "GREAT" vs "great")
        - Punctuation affects intensity (e.g., "good!" vs "good")
        - Negation is handled (e.g., "not good" is detected as negative)
        - The compound score combines positive, negative, and neutral scores
        - No training data required; uses a pre-built sentiment lexicon
        - Best suited for short, informal text; may be less accurate on
          formal or domain-specific content

    See Also:
        naive_bayes_classify: Train custom sentiment classifier
        svm_classify: Train SVM-based sentiment classifier
    """
    _ensure_vader()
    from nltk.sentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)

    if sentiment_scores["compound"] >= 0.05:
        return "positive"
    if sentiment_scores["compound"] <= -0.05:
        return "negative"
    return "neutral"
